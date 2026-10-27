"""``opt.init_jitter``: a seeded perturbation of the start, off by default."""

import subprocess
import sys
import zlib
from importlib.resources import files as _res_files
from pathlib import Path

import jax
import numpy as np
import pytest
import yaml

from tabascal.config import load_config, yaml_load
from tabascal.jitter import (
    SEED_TAG,
    describe_init_jitter,
    jitter_init,
    resolve_init_jitter,
    validate_init_jitter,
)
from tabascal.seeds import component_key, describe_random_draws

from tests.test_seeds import COMPONENTS


def start():
    """A start shaped like a model's: two ast blocks, two rfi blocks and a gain block."""
    rng = np.random.default_rng(0)
    return {
        "ast_k_r_base": jax.numpy.zeros((4, 1, 3)),
        "ast_k_i_base": jax.numpy.zeros((4, 1, 3)),
        "rfi_k_r_base": jax.numpy.asarray(rng.normal(size=(2, 3, 1, 5))),
        "rfi_k_i_base": jax.numpy.asarray(rng.normal(size=(2, 3, 1, 5))),
        "gains_amp_base": jax.numpy.asarray(rng.normal(size=(3, 1, 1))),
    }


def draw(seed, name, like):
    key = jax.random.fold_in(component_key(seed, SEED_TAG), zlib.crc32(name.encode()))
    return np.asarray(jax.random.normal(key, like.shape, like.dtype))


# ---------------------------------------------------------------------------
# Off
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", [None, 0, 0.0, {"rfi": 0}], ids=["null", "0", "0.0", "zero-block"])
def test_off_returns_the_very_start_it_was_given(spec):
    """Not a copy: the same arrays, so a run without jitter is bit-identical."""
    params = start()
    out = jitter_init(params, spec, seed=7)
    assert all(out[k] is params[k] for k in params)


def test_off_is_described_as_nothing():
    assert describe_init_jitter(None) is None


# ---------------------------------------------------------------------------
# The draw
# ---------------------------------------------------------------------------


def test_a_number_is_an_absolute_scale_on_every_block(exact_rtol):
    params = start()
    out = jitter_init(params, 0.05, seed=7)
    for name, x in params.items():
        # Against the sum rather than the difference, which in single precision
        # keeps the start's round-off and none of the jitter's digits.
        expected = np.asarray(x) + 0.05 * draw(7, name, x)
        np.testing.assert_allclose(np.asarray(out[name]), expected, rtol=exact_rtol, atol=exact_rtol)


def test_a_relative_scale_is_a_fraction_of_the_rms_of_the_start(exact_rtol):
    params = start()
    out = jitter_init(params, {"rfi": {"relative": 0.1}}, seed=7)
    for name in ("rfi_k_r_base", "rfi_k_i_base"):
        x = np.asarray(params[name])
        sigma = 0.1 * np.sqrt(np.mean(x**2))
        np.testing.assert_allclose(
            np.asarray(out[name]), x + sigma * draw(7, name, params[name]),
            rtol=exact_rtol, atol=exact_rtol,
        )
    for name in ("ast_k_r_base", "ast_k_i_base", "gains_amp_base"):
        assert out[name] is params[name]


def test_the_longest_block_wins():
    params = start()
    out = jitter_init(params, {"ast": 0.1, "ast_k_r_base": 0.5}, seed=7)
    np.testing.assert_allclose(
        np.asarray(out["ast_k_r_base"]), 0.5 * draw(7, "ast_k_r_base", params["ast_k_r_base"]), rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(out["ast_k_i_base"]), 0.1 * draw(7, "ast_k_i_base", params["ast_k_i_base"]), rtol=1e-6
    )


def test_a_block_prefix_stops_at_an_underscore():
    """``rf`` is not a prefix of ``rfi_k_r_base`` in this sense, so it names nothing."""
    with pytest.raises(ValueError, match="rf"):
        resolve_init_jitter({"rf": 0.1}, start())


@pytest.mark.parametrize("seeds", [(7, 7), (7, 8)], ids=["same", "different"])
def test_the_jitter_is_identical_exactly_when_the_seed_is(seeds):
    a, b = (jitter_init(start(), 0.05, seed=s) for s in seeds)
    same = all(np.array_equal(a[k], b[k]) for k in a)
    assert same == (seeds[0] == seeds[1])


def test_a_parameter_draw_does_not_depend_on_the_other_parameters():
    params = start()
    alone = jitter_init({"gains_amp_base": params["gains_amp_base"]}, 0.05, seed=7)
    together = jitter_init(params, 0.05, seed=7)
    assert np.array_equal(alone["gains_amp_base"], together["gains_amp_base"])


def test_the_draw_is_not_the_one_ast_init_sample_takes():
    """The jitter folds its own tag in, so it is independent of a sampled start."""
    x = start()["ast_k_r_base"]
    ast_key_draw = np.asarray(jax.random.normal(component_key(7, "ast"), x.shape, x.dtype))
    assert not np.array_equal(draw(7, "ast_k_r_base", x), ast_key_draw)


@pytest.mark.parametrize("kind", ["absolute", "relative"])
def test_padded_dummy_sources_stay_dark_and_do_not_change_the_real_draw(kind):
    """Padding 2 real sources to 4 for the devices moves neither the real rows nor the dummies."""
    params = start()
    rfi = params["rfi_k_r_base"]
    padded = dict(params, rfi_k_r_base=jax.numpy.concatenate([rfi, jax.numpy.zeros_like(rfi)]))
    spec = {"rfi_k_r_base": {kind: 0.1}}

    unpadded_out = jitter_init(params, spec, seed=7, n_rfi=2, n_rfi_real=2)
    padded_out = jitter_init(padded, spec, seed=7, n_rfi=4, n_rfi_real=2)

    np.testing.assert_array_equal(padded_out["rfi_k_r_base"][:2], unpadded_out["rfi_k_r_base"])
    np.testing.assert_array_equal(padded_out["rfi_k_r_base"][2:], 0)


# ---------------------------------------------------------------------------
# What is refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec",
    [True, -0.1, float("nan"), float("inf"), "0.1", {"rfi": {"relative": -1}},
     {"rfi": {"scale": 0.1}}, {"rfi": {"relative": 0.1, "absolute": 0.1}}, {"rfi": "big"}],
)
def test_a_malformed_spec_is_refused_by_name(spec):
    with pytest.raises(ValueError, match=r"opt\.init_jitter"):
        validate_init_jitter(spec)


def test_a_block_that_names_no_parameter_is_refused():
    with pytest.raises(ValueError, match="gainz"):
        jitter_init(start(), {"gainz": 0.1}, seed=7)


def test_a_relative_jitter_on_a_block_that_starts_at_zero_is_refused():
    """ast starts at its zero prior mean here, so a relative jitter would be no jitter."""
    with pytest.raises(ValueError, match="ast_k_r_base"):
        jitter_init(start(), {"ast": {"relative": 0.1}}, seed=7)


def test_load_config_refuses_a_malformed_spec(tmp_path):
    path = tmp_path / "user.yaml"
    path.write_text(yaml.dump({"model": {"components": COMPONENTS}, "opt": {"init_jitter": -1}}))
    with pytest.raises(ValueError, match=r"opt\.init_jitter"):
        load_config(str(path))


def test_the_base_config_ships_it_off():
    from importlib.resources import files

    base = yaml_load(str(files("tabascal").joinpath("data/config/tab_config_base.yaml")))
    assert base["opt"]["init_jitter"] is None


# ---------------------------------------------------------------------------
# The run header
# ---------------------------------------------------------------------------


def loaded(tmp_path, jitter, ast_init="prior", seed=4242):
    path = tmp_path / "user.yaml"
    path.write_text(yaml.dump({
        "model": {"components": COMPONENTS},
        "ast": {"init": ast_init},
        "rfi": {"init": "prior"},
        "inference": {"seed": seed},
        "opt": {"init_jitter": jitter},
    }))
    return load_config(str(path))


def test_the_line_names_the_jitter_and_the_seed(tmp_path):
    line = describe_random_draws(loaded(tmp_path, {"rfi": {"relative": 0.01}, "ast": 0.05}))

    assert "no random draws" not in line
    assert "rfi relative 0.01" in line and "ast absolute 0.05" in line
    assert "4242" in line


def test_the_line_names_a_sampled_start_and_the_jitter_together(tmp_path):
    line = describe_random_draws(loaded(tmp_path, 0.05, ast_init="sample"))

    assert "ast init sampled" in line and "every block absolute 0.05" in line


# ---------------------------------------------------------------------------
# End to end: the runner jitters the start it fits from
# ---------------------------------------------------------------------------

FIT_SEEDS = (1, 7)


@pytest.fixture(scope="module")
def jittered_fits(tmp_path_factory):
    """Three MAP iterations on the 8A sim with a jittered start, at two seeds.

    Returns ``{seed: (stdout, out_dir, config_path)}``.
    """
    import tabsim
    from huggingface_hub import snapshot_download

    from tests.test_tabascal_pipeline import _copy_sim, read_and_modify_yaml

    try:
        data = Path(snapshot_download(
            repo_id="epfl-radio-astro/rfi-simulations",
            repo_type="dataset",
            revision=f"tabsim_v{tabsim.__version__}",
        ))
    except Exception as e:  # pragma: no cover - offline
        pytest.skip(f"8A simulation not available: {e}")

    template = Path(__file__).parent / "data" / "tab_target.yaml"
    base = yaml_load(template)
    script = Path(__file__).parent.parent / "tabascal" / "scripts" / "run_tabascal.py"
    precision = "double" if jax.config.read("jax_enable_x64") else "single"

    fits = {}
    for seed in FIT_SEEDS:
        work = tmp_path_factory.mktemp(f"jitter{seed}")
        out_dir = _copy_sim(data, work)
        config_path = work / "tab_target.yaml"
        read_and_modify_yaml(
            {
                "model": {"components": COMPONENTS, "precision": precision},
                "inference": {"opt": True, "seed": seed},
                "opt": {"epsilon": 1e-3, "max_iter": 3, "dual_run": False, "guide": "map",
                        "init_jitter": 0.1},
                "ast": {**base["ast"], "init": "prior"},
                "rfi": {**base["rfi"], "init": "prior"},
            },
            template,
            config_path,
        )
        result = subprocess.run(
            [
                sys.executable, str(script), "run",
                "-c", str(config_path),
                "-od", str(out_dir),
                "--extra-orbit-dir", str(_res_files("tabascal").joinpath("data/tles")),
                "-nl",
            ],
            capture_output=True, text=True, cwd=work, check=False,
        )
        assert result.returncode == 0, f"seed {seed} run failed: {result.stderr}"
        fits[seed] = (result.stdout, out_dir, config_path)

    return fits


def test_with_a_jitter_the_seed_moves_the_start(jittered_fits):
    """The deterministic-init twin of this run is bitwise identical across seeds (test_seeds)."""
    import xarray as xr

    first, second = (
        xr.open_zarr(str(jittered_fits[s][1] / "results" / "init_pred_Custom.zarr"))
        for s in FIT_SEEDS
    )
    assert any(not np.array_equal(first[n].values, second[n].values) for n in first.data_vars)


@pytest.mark.parametrize("seed", FIT_SEEDS)
def test_the_run_header_carries_the_jitter(jittered_fits, seed):
    stdout, _, config_path = jittered_fits[seed]

    line = describe_random_draws(load_config(str(config_path)))
    assert "jittered" in line and line in stdout
