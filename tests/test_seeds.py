"""``inference.seed``: one seed, a fixed key per drawing component (#256)."""

import subprocess
import sys
import warnings
from importlib.resources import files as _res_files
from pathlib import Path

import jax
import numpy as np
import pytest
import yaml

from tabascal.components.ast_vis import GPVisAst
from tabascal.components.rfi_signal import ComplexRFIConstAntFine, ComplexRFIVarAntFine
from tabascal.config import load_config, yaml_load

from tests.components.test_rfi_signal import make_rfi_config
from tests.test_base_config import base_args, make_ast_config

#: Stands for a config with no ``inference.seed`` key at all.
ABSENT = object()

#: The seed a null or absent ``inference.seed`` means.
DEFAULT_SEED = 1

RFI_CLASSES = [ComplexRFIVarAntFine, ComplexRFIConstAntFine]

COMPONENTS = [
    "trajectory:FixedOrbitFine",
    "rfi_signal:ComplexRFIVarAntFine",
    "rfi_vis:RiemannVisFine",
    "ast_vis:GPVisAst",
    "gains:UnitaryGains",
]


def _set_seed(args, seed):
    inference = args.setdefault("inference", {})
    inference.pop("seed", None)
    if seed is not ABSENT:
        inference["seed"] = seed


def ast_start(tmp_path, seed, init="sample"):
    """``GPVisAst`` set up with ``ast.init`` and ``inference.seed``."""
    args = base_args(tmp_path)
    args["ast"]["init"] = init
    _set_seed(args, seed)
    comp = GPVisAst()
    comp.setup(make_ast_config(args))
    return comp


def rfi_start(cls, seed, init="sample", r_seed=None):
    """An RFI signal component with no dummy sources, set up on the mock config."""
    config = make_rfi_config(n_rfi=3, n_rfi_real=3, init=init)
    if r_seed is not None:
        config.args["rfi"]["r_seed"] = r_seed
    _set_seed(config.args, seed)
    comp = cls()
    comp.setup(config)
    return comp


START = {
    "ast": lambda tmp_path, seed, **kw: ast_start(tmp_path, seed, **kw),
    "rfi": lambda tmp_path, seed, **kw: rfi_start(ComplexRFIVarAntFine, seed, **kw),
}


def base_latent(comp):
    """The component's initial base (whitened) latent, as one complex array."""
    (part,) = {k.removesuffix("_r_base").removesuffix("_i_base") for k in comp.init_params_base}
    params = comp.init_params_base
    return np.asarray(params[f"{part}_r_base"]) + 1j * np.asarray(params[f"{part}_i_base"])


def write_config(tmp_path, **sections):
    path = tmp_path / "user.yaml"
    path.write_text(yaml.dump({"model": {"components": COMPONENTS}, **sections}))
    return path


def loaded(tmp_path, ast_init="prior", rfi_init="prior", seed=None):
    return load_config(
        str(write_config(
            tmp_path, ast={"init": ast_init}, rfi={"init": rfi_init}, inference={"seed": seed}
        ))
    )


# ---------------------------------------------------------------------------
# The helpers
# ---------------------------------------------------------------------------


def test_the_run_key_is_the_prng_key_of_the_seed():
    from tabascal.seeds import run_key

    assert np.array_equal(run_key(5), jax.random.PRNGKey(5))


@pytest.mark.parametrize("tag", ["ast", "rfi"])
def test_a_component_key_is_the_seed_folded_with_the_crc_of_its_tag(tag):
    import zlib

    from tabascal.seeds import component_key

    expected = jax.random.fold_in(jax.random.PRNGKey(5), zlib.crc32(tag.encode()))
    assert np.array_equal(component_key(5, tag), expected)


# ---------------------------------------------------------------------------
# The `init: sample` draws
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [ABSENT, None, 0, 7], ids=["absent", "null", "0", "7"])
@pytest.mark.parametrize(
    "tag, make",
    [("ast", lambda tmp_path, seed: ast_start(tmp_path, seed))]
    + [("rfi", lambda tmp_path, seed, c=c: rfi_start(c, seed)) for c in RFI_CLASSES],
    ids=["ast"] + [f"rfi-{c.__name__}" for c in RFI_CLASSES],
)
def test_a_sample_start_is_the_draw_from_its_own_component_key(
    tmp_path, tag, make, seed, exact_rtol
):
    """The draw depends on (seed, tag) alone, so no other component can shift it."""
    from tabascal.seeds import component_key

    comp = make(tmp_path, seed)
    start = base_latent(comp)
    expected_seed = DEFAULT_SEED if seed in (ABSENT, None) else seed
    draw = jax.random.normal(component_key(expected_seed, tag), start.shape, dtype=complex)

    np.testing.assert_allclose(start, np.asarray(draw), rtol=exact_rtol, atol=exact_rtol)


@pytest.mark.parametrize("seeds", [(7, 7), (7, 8)], ids=["same", "different"])
@pytest.mark.parametrize("tag", ["ast", "rfi"])
def test_the_sample_start_is_identical_exactly_when_the_seed_is(tmp_path, tag, seeds):
    first, second = (base_latent(START[tag](tmp_path, s)) for s in seeds)

    assert np.array_equal(first, second) == (seeds[0] == seeds[1])


def test_rfi_r_seed_has_no_effect_on_the_draw(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        set_ = rfi_start(ComplexRFIVarAntFine, 7, r_seed=99)
    unset = rfi_start(ComplexRFIVarAntFine, 7, r_seed=None)

    assert np.array_equal(base_latent(set_), base_latent(unset))


@pytest.mark.parametrize(
    "tag, init",
    [("ast", i) for i in ("prior", "zeros", "data")]
    + [("rfi", i) for i in ("prior", "zeros", "ones")],
)
def test_a_deterministic_start_does_not_depend_on_the_seed(tmp_path, tag, init):
    first, second = (base_latent(START[tag](tmp_path, s, init=init)) for s in (1, 7))

    assert np.array_equal(first, second)


# ---------------------------------------------------------------------------
# The config keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [None, 0, 7])
def test_a_null_or_integer_seed_loads(tmp_path, seed):
    loaded(tmp_path, seed=seed)


@pytest.mark.parametrize("seed", [True, 1.5, "7"])
def test_a_seed_that_is_not_an_integer_is_refused_by_name(tmp_path, seed):
    with pytest.raises(ValueError, match=r"inference\.seed"):
        loaded(tmp_path, seed=seed)


@pytest.mark.parametrize("section", ["rfi", "gains"])
def test_a_set_r_seed_warns_that_inference_seed_replaces_it(tmp_path, section):
    with pytest.warns(FutureWarning) as record:
        load_config(str(write_config(tmp_path, **{section: {"r_seed": 5}})))

    messages = [str(w.message) for w in record if issubclass(w.category, FutureWarning)]
    assert any(
        section in m and "r_seed" in m and "inference.seed" in m for m in messages
    ), messages


@pytest.mark.parametrize(
    "sections",
    [{"rfi": {"r_seed": None}, "gains": {"r_seed": None}}, {}],
    ids=["null", "absent"],
)
def test_a_null_or_absent_r_seed_does_not_warn(tmp_path, sections):
    """The absent case also fails if the base config ever ships an r_seed again."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        config = load_config(str(write_config(tmp_path, **sections)))
    if not sections:
        assert not any("r_seed" in config[s] for s in ("rfi", "gains"))


# ---------------------------------------------------------------------------
# What the run header says
# ---------------------------------------------------------------------------


def test_with_no_sampled_init_the_line_says_there_are_no_random_draws(tmp_path):
    from tabascal.seeds import describe_random_draws

    assert "no random draws" in describe_random_draws(loaded(tmp_path, seed=4242))


@pytest.mark.parametrize(
    "ast_init, rfi_init, sampled",
    [
        ("sample", "prior", ["ast"]),
        ("prior", "sample", ["rfi"]),
        ("sample", "sample", ["ast", "rfi"]),
    ],
)
def test_the_line_names_each_sampled_section_and_the_seed(tmp_path, ast_init, rfi_init, sampled):
    from tabascal.seeds import describe_random_draws

    line = describe_random_draws(loaded(tmp_path, ast_init, rfi_init, seed=4242))

    assert "no random draws" not in line
    assert "4242" in line
    assert all(section in line for section in sampled), line


@pytest.mark.parametrize("n_proc, says", [(1, "prior plots draw"), (2, "prior plots skipped")])
def test_the_line_says_whether_prior_plots_draw(tmp_path, monkeypatch, n_proc, says):
    """The runner skips prior plots in a multi-process run, and the line follows it."""
    from tabascal.seeds import describe_random_draws

    config = loaded(tmp_path, seed=4242)
    config["plots"]["prior"] = True
    monkeypatch.setattr(jax, "process_count", lambda: n_proc)

    assert says in describe_random_draws(config)


def test_the_runner_takes_its_run_key_from_inference_seed(monkeypatch):
    """Stops the run at the key; a hard-coded key never reaches the spy and fails."""
    from types import SimpleNamespace

    from tabascal.scripts import _run_tabascal_impl as impl

    class Stop(Exception):
        pass

    seen = []

    def spy(seed):
        seen.append(seed)
        raise Stop

    monkeypatch.setattr(impl, "run_key", spy)
    monkeypatch.setattr(impl, "_resolve_paths", lambda *a, **k: SimpleNamespace(ms_path=None, log_path=None))

    with pytest.raises(Stop):
        impl.tabascal_subtraction({"inference": {"seed": 7}}, out_dir=None, log=False)
    assert seen == [7]


# ---------------------------------------------------------------------------
# End to end: a run with no stochastic path, at two seeds
# ---------------------------------------------------------------------------

FIT_SEEDS = (1, 7)


@pytest.fixture(scope="module")
def deterministic_fits(tmp_path_factory):
    """Three MAP iterations on the 8A sim at each of two seeds; ~35 s per run.

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
        work = tmp_path_factory.mktemp(f"seed{seed}")
        out_dir = _copy_sim(data, work)
        config_path = work / "tab_target.yaml"
        read_and_modify_yaml(
            {
                "model": {"components": COMPONENTS, "precision": precision},
                "inference": {"opt": True, "seed": seed},
                "opt": {"epsilon": 1e-3, "max_iter": 3, "dual_run": False, "guide": "map"},
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


@pytest.mark.parametrize("product", ["init_pred", "map_pred"])
def test_with_no_stochastic_path_the_seed_changes_no_output(deterministic_fits, product):
    """Every array the run writes is bitwise identical across the two seeds."""
    import xarray as xr

    first, second = (
        xr.open_zarr(str(deterministic_fits[s][1] / "results" / f"{product}_Custom.zarr"))
        for s in FIT_SEEDS
    )

    assert set(first.data_vars) == set(second.data_vars)
    for name in first.data_vars:
        assert np.array_equal(first[name].values, second[name].values), name


@pytest.mark.parametrize("seed", FIT_SEEDS)
def test_the_run_header_carries_the_random_draws_line(deterministic_fits, seed):
    from tabascal.seeds import describe_random_draws

    stdout, _, config_path = deterministic_fits[seed]

    assert describe_random_draws(load_config(str(config_path))) in stdout
