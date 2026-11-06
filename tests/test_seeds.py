"""``inference.seed``: one seed, a fixed key per drawing component (#256)."""

import subprocess
import sys
import warnings
import zlib
from importlib.resources import files as _res_files
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest
import yaml

from tabascal.components.ast_vis import GPVisAst
from tabascal.components.rfi_signal import ComplexRFIConstAntFine, ComplexRFIVarAntFine
from tabascal.config import load_config, yaml_load

from tests.components.test_rfi_signal import make_rfi_config
from tests.test_base_config import base_args, build_stubbed_tab_config, make_ast_config

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


def ast_start(tmp_path, seed, init="sample", components=COMPONENTS):
    """``GPVisAst`` set up with ``ast.init`` and ``inference.seed``."""
    args = base_args(tmp_path)
    args["ast"]["init"] = init
    args["model"]["components"] = list(components)
    _set_seed(args, seed)
    comp = GPVisAst()
    comp.setup(make_ast_config(args))
    return comp


def rfi_start(cls, seed, init="sample", r_seed=None, components=COMPONENTS):
    """An RFI signal component with no dummy sources, set up on the mock config."""
    config = make_rfi_config(n_rfi=3, n_rfi_real=3, init=init)
    if r_seed is not None:
        config.args["rfi"]["r_seed"] = r_seed
    config.args["model"] = {"components": list(components)}
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


def write_config(tmp_path, components=COMPONENTS, **sections):
    path = tmp_path / "user.yaml"
    path.write_text(yaml.dump({"model": {"components": components}, **sections}))
    return path


def loaded(tmp_path, ast_init="prior", rfi_init="prior", seed=None, components=COMPONENTS):
    inference = {} if seed is ABSENT else {"inference": {"seed": seed}}
    return load_config(str(write_config(
        tmp_path, components, ast={"init": ast_init}, rfi={"init": rfi_init}, **inference
    )))


# ---------------------------------------------------------------------------
# The `init: sample` draws
# ---------------------------------------------------------------------------


def test_the_run_key_is_the_prng_key_of_the_seed():
    from tabascal.seeds import run_key

    assert np.array_equal(run_key(5), jax.random.PRNGKey(5))


@pytest.mark.parametrize("seed", [ABSENT, None, 0, 7], ids=["absent", "null", "0", "7"])
@pytest.mark.parametrize(
    "tag, make",
    [("ast", lambda tmp_path, seed: ast_start(tmp_path, seed))]
    + [("rfi", lambda tmp_path, seed, c=c: rfi_start(c, seed)) for c in RFI_CLASSES],
    ids=["ast"] + [f"rfi-{c.__name__}" for c in RFI_CLASSES],
)
def test_a_sample_start_is_the_draw_from_the_seed_folded_with_the_crc_of_its_tag(
    tmp_path, tag, make, seed, exact_rtol
):
    start = base_latent(make(tmp_path, seed))
    expected_seed = DEFAULT_SEED if seed in (ABSENT, None) else seed
    key = jax.random.fold_in(jax.random.PRNGKey(expected_seed), zlib.crc32(tag.encode()))
    draw = jax.random.normal(key, start.shape, dtype=complex)

    np.testing.assert_allclose(start, np.asarray(draw), rtol=exact_rtol, atol=exact_rtol)


@pytest.mark.parametrize("tag", ["ast", "rfi"])
def test_adding_removing_or_moving_another_component_never_shifts_a_draw(tmp_path, tag):
    lists = [COMPONENTS, ["gains:ConstGains", *COMPONENTS], COMPONENTS[1:], COMPONENTS[::-1]]
    first, *others = (base_latent(START[tag](tmp_path, 7, components=c)) for c in lists)

    assert all(np.array_equal(first, other) for other in others)


@pytest.mark.parametrize("r_seed", [99, "not a seed"])
@pytest.mark.parametrize("cls", RFI_CLASSES)
def test_an_old_rfi_r_seed_has_no_effect_on_the_draw(cls, r_seed):
    set_, unset = (rfi_start(cls, 7, r_seed=r) for r in (r_seed, None))

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


@pytest.mark.parametrize(
    "seed, expected", [(ABSENT, 1), (None, 1), (0, 0), (7, 7)], ids=["absent", "null", "0", "7"]
)
def test_load_config_normalises_the_seed(tmp_path, seed, expected):
    assert loaded(tmp_path, seed=seed)["inference"]["seed"] == expected


NOT_INTEGERS = [True, 1.0, 1.5, "7"]


@pytest.mark.parametrize("seed", NOT_INTEGERS)
def test_a_seed_that_is_not_an_integer_is_refused_by_name(tmp_path, seed):
    with pytest.raises(ValueError, match=r"inference\.seed"):
        loaded(tmp_path, seed=seed)


@pytest.mark.parametrize("seed", NOT_INTEGERS)
@pytest.mark.parametrize("tag", ["ast", "rfi"])
def test_a_seed_that_is_not_an_integer_is_refused_without_load_config(tmp_path, tag, seed):
    """A config built in code skips load_config; even a deterministic start refuses it."""
    with pytest.raises(RuntimeError, match=r"inference\.seed"):
        START[tag](tmp_path, seed, init="prior")


@pytest.mark.parametrize("route", ["loaded", "built in code"])
@pytest.mark.parametrize("section", ["rfi", "gains"])
def test_a_set_r_seed_warns_once_into_a_tabconfig(tmp_path, monkeypatch, section, route):
    """Loaded (load_config warns, TabConfig must not again) or built in code (TabConfig warns)."""
    with pytest.warns(FutureWarning) as record:
        if route == "loaded":
            config = load_config(str(write_config(tmp_path, [], **{section: {"r_seed": 5}})))
        else:
            config = base_args(tmp_path)
            config[section]["r_seed"] = 5
        build_stubbed_tab_config(config, monkeypatch)

    (warned,) = [w for w in record if "r_seed" in str(w.message)]
    assert issubclass(warned.category, FutureWarning)
    assert f"{section}.r_seed" in str(warned.message) and "inference.seed" in str(warned.message)


@pytest.mark.parametrize(
    "sections",
    [{"rfi": {"r_seed": None}, "gains": {"r_seed": None}}, {}],
    ids=["null", "absent"],
)
def test_a_null_or_absent_r_seed_does_not_warn(tmp_path, sections):
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        load_config(str(write_config(tmp_path, **sections)))


# ---------------------------------------------------------------------------
# What the run header says
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ast_init, rfi_init, drop, sampled",
    [
        ("prior", "prior", None, []),
        ("sample", "prior", None, ["ast"]),
        ("prior", "sample", None, ["rfi"]),
        ("sample", "sample", None, ["ast", "rfi"]),
        ("sample", "sample", "ast_vis", ["rfi"]),
        ("sample", "sample", "rfi_signal", ["ast"]),
    ],
)
def test_the_line_names_exactly_the_sections_that_draw(
    tmp_path, ast_init, rfi_init, drop, sampled
):
    """A section draws when it asks for ``init: sample`` and the model has its component."""
    from tabascal.seeds import describe_random_draws

    components = [c for c in COMPONENTS if drop is None or not c.startswith(drop)]
    line = describe_random_draws(loaded(tmp_path, ast_init, rfi_init, 4242, components))

    assert ("no random draws" in line) != bool(sampled), line
    assert ("4242" in line) == bool(sampled), line
    assert [t for t in ("ast", "rfi") if f"{t} init" in line] == sampled, line


@pytest.mark.parametrize(
    "n_proc, says", [(1, "prior plots draw from inference.seed 4242"), (3, "prior plots skipped")]
)
def test_the_line_says_whether_prior_plots_draw(tmp_path, monkeypatch, n_proc, says):
    """The runner skips prior plots in a multi-process run, and the line follows it."""
    from tabascal.seeds import describe_random_draws

    config = loaded(tmp_path, seed=4242)
    config["plots"]["prior"] = True
    monkeypatch.setattr(jax, "process_count", lambda: n_proc)

    assert says in describe_random_draws(config)


# ---------------------------------------------------------------------------
# The runner, with the model and its outputs stubbed
# ---------------------------------------------------------------------------


def run_stubbed(tmp_path, monkeypatch, capsys, seed, n):
    """The runner on a loaded config with prior plots on; the keys each consumer got."""
    from tabascal.scripts import _run_tabascal_impl as impl

    config = loaded(tmp_path, seed=seed)
    config["plots"]["prior"] = True
    run_dir = tmp_path / f"run{n}"
    run_dir.mkdir()
    paths = SimpleNamespace(
        **dict.fromkeys(["name", "f_name", "ms_path", "log_path", "plot_dir", "map_path",
                         "init_pred_path", "used_orbits_path"]),
        run_dir=str(run_dir),
    )
    model = SimpleNamespace(prob_model=None, init_params={}, state={}, constants={})
    keys = {}
    stubs = {
        **dict.fromkeys(["_print_run_header", "_record_resolved_settings", "save_orbits_for_reuse",
                         "_print_model_summary", "write_results_xds"], lambda *a, **k: None),
        "_resolve_paths": lambda *a, **k: paths,
        "build_model": lambda *a: (SimpleNamespace(vis_obs=None), model),
        "load_truth": lambda *a: {},
        "init_predict": lambda _c, _m, key, *a, **k: keys.setdefault("init", key),
        "nlog_like_and_post": lambda *a, **k: (0.0, 0.0),
        "run_opt": lambda _c, _m, subkeys, *a, **k: (keys.setdefault("opt", subkeys), None, {}, None),
    }
    for name, stub in stubs.items():
        monkeypatch.setattr(impl, name, stub)
    monkeypatch.setattr(
        "tabascal.plot.plot_prior", lambda _c, _m, _t, _n, key, *a, **k: keys.setdefault("prior", key)
    )

    impl.tabascal_subtraction(config, out_dir=None, log=False)
    return keys, capsys.readouterr().out, config, run_dir


def test_the_seed_reaches_every_key_consumer_the_header_and_run_json(tmp_path, monkeypatch, capsys):
    """Unset and 1 hand the init prediction, prior plots and optimiser the same keys; 7 others."""
    from tabascal.run_dir import read_record
    from tabascal.seeds import describe_random_draws

    runs = [run_stubbed(tmp_path, monkeypatch, capsys, s, n) for n, s in enumerate((ABSENT, 1, 7))]
    (unset, *_), (one, *_), (seven, *_) = runs

    for consumer in ("init", "prior", "opt"):
        assert np.array_equal(unset[consumer], one[consumer]), consumer
        assert not np.array_equal(one[consumer], seven[consumer]), consumer
    assert [read_record(run_dir)["seed"] for *_, run_dir in runs] == [1, 1, 7]
    assert all(describe_random_draws(config) in out for _, out, config, _ in runs)


# ---------------------------------------------------------------------------
# End to end: a run with no stochastic path, at two seeds
# ---------------------------------------------------------------------------

FIT_SEEDS = (1, 7)


@pytest.fixture(scope="module")
def deterministic_fits(tmp_path_factory):
    """Three MAP iterations on the 8A sim at each of two seeds; ~35 s per run.

    Returns ``{seed: run_dir}``.
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
                "-n", "seed",
                "--extra-orbit-dir", str(_res_files("tabascal").joinpath("data/tles")),
                "-nl",
            ],
            capture_output=True, text=True, cwd=work, check=False,
        )
        assert result.returncode == 0, f"seed {seed} run failed: {result.stderr}"
        fits[seed] = out_dir / "runs" / "latest"

    return fits


@pytest.mark.parametrize("product", ["init_pred", "map_pred"])
def test_with_no_stochastic_path_the_seed_changes_no_output(deterministic_fits, product):
    """Every array the run writes is bitwise identical across the two seeds."""
    import xarray as xr

    first, second = (
        xr.open_zarr(str(deterministic_fits[s] / "results" / f"{product}.zarr"))
        for s in FIT_SEEDS
    )

    assert set(first.data_vars) == set(second.data_vars)
    for name in first.data_vars:
        assert np.array_equal(first[name].values, second[name].values), name
