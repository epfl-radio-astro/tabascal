"""``opt.schedule``, ``opt.stop`` and ``opt.checkpoints`` (GitHub #82).

- **The old path is unchanged.** With no schedule, ``dual_run`` still runs two
  ``adabelief`` phases, bit for bit.
- **One compilation.** A schedule builds one optimiser, so the step is traced
  once where ``dual_run`` traces it twice.
- **The step sizes are the ones asked for**, and the optimiser takes them.
- **Checkpoints are written at their steps and reload** to the parameters the
  optimiser held then.
- **The stop triggers** once the loss stops moving, and not before ``min_steps``.
"""

import numpy as np
import optax
import pytest

from tabascal import tab_tools
from tabascal.opt_schedule import (
    checkpoint_path,
    converged,
    describe_schedule,
    load_checkpoint,
    step_size,
    validate_opt,
    validate_schedule,
)
from tabascal.tab_tools import run_custom_svi

from tests.test_loss_trace import _toy_problem

EPS = 1e-2


def _by_hand(toy, phases):
    """``_map_step`` driven by hand: ``phases`` is ``[(optimizer, n_steps), ...]``.

    Returns the losses and the parameters after every step.
    """
    params, losses, history = toy.init_params, [], []
    for optimizer, n in phases:
        opt_state = optimizer.init(params)
        for _ in range(n):
            params, opt_state, loss = tab_tools._map_step(
                toy.prob_model, optimizer, params, opt_state,
                toy.state, toy.constants, toy.vis_obs,
            )
            losses.append(float(loss))
            history.append(params)
    return losses, history


def _svi(toy, max_iter, **kwargs):
    kwargs.setdefault("dual_run", False)
    return run_custom_svi(
        prob_model=toy.prob_model,
        obs_data=toy.vis_obs,
        max_iter=max_iter,
        init_params=toy.init_params,
        epsilon=EPS,
        state=toy.state,
        constants=toy.constants,
        **kwargs,
    )


def _same(got: dict, ref: dict, suffix="_auto_loc"):
    for key, value in ref.items():
        assert np.array_equal(np.asarray(got[key + suffix]), np.asarray(value)), key


def _counting(toy):
    """``toy`` with a model that counts how often it is traced."""
    calls = {"n": 0}
    model = toy.prob_model

    def prob_model(*args, **kwargs):
        calls["n"] += 1
        return model(*args, **kwargs)

    toy.prob_model = prob_model
    return calls


# ---------------------------------------------------------------------------
# The old path
# ---------------------------------------------------------------------------


def test_with_no_schedule_dual_run_is_the_two_phases_it_always_was():
    toy = _toy_problem(0.1)
    ref_losses, history = _by_hand(toy, [(optax.adabelief(EPS), 3), (optax.adabelief(EPS / 10), 3)])

    results = _svi(toy, 3, dual_run=True)

    assert list(np.asarray(results.losses)) == ref_losses
    _same(results.params, history[-1])


def test_dual_run_traces_the_step_twice():
    """What #82 is about: the second phase's new optimiser is a new static argument."""
    toy = _toy_problem(0.1)
    calls = _counting(toy)
    _svi(toy, 3, dual_run=True)
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# The schedule
# ---------------------------------------------------------------------------


def test_a_step_schedule_holds_epsilon_for_n_steps_then_drops_as_dual_run_does():
    schedule = step_size(validate_schedule({"kind": "step", "boundaries": [4]}, 8), EPS, 8)
    sizes = [float(schedule(count)) for count in range(8)]
    np.testing.assert_allclose(sizes, [EPS] * 4 + [EPS / 10] * 4, rtol=1e-6)


def test_a_step_schedule_multiplies_at_every_boundary():
    schedule = step_size(
        validate_schedule({"kind": "step", "boundaries": [2, 4], "factor": 0.5}, 6), EPS, 6
    )
    sizes = [float(schedule(count)) for count in range(6)]
    np.testing.assert_allclose(sizes, [EPS, EPS, EPS / 2, EPS / 2, EPS / 4, EPS / 4], rtol=1e-6)


def test_an_exponential_schedule_reaches_its_final_factor_at_max_iter():
    schedule = step_size(validate_schedule({"kind": "exponential", "final_factor": 0.01}, 10), EPS, 10)
    sizes = np.array([float(schedule(count)) for count in range(11)])
    np.testing.assert_allclose(sizes, EPS * 0.01 ** (np.arange(11) / 10), rtol=1e-5)


def test_a_cosine_schedule_falls_by_half_a_cosine_to_its_final_factor():
    schedule = step_size(validate_schedule({"kind": "cosine", "final_factor": 0.1}, 10), EPS, 10)
    sizes = np.array([float(schedule(count)) for count in range(11)])
    t = np.arange(11) / 10
    np.testing.assert_allclose(sizes, EPS * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * t))), rtol=1e-5)


def test_a_constant_schedule_holds_epsilon():
    schedule = step_size(validate_schedule({"kind": "constant"}, 5), EPS, 5)
    np.testing.assert_allclose([float(schedule(c)) for c in range(5)], [EPS] * 5, rtol=1e-6)


def test_the_scheduled_run_takes_the_scheduled_steps():
    """The run is one adabelief on that schedule, bit for bit."""
    toy = _toy_problem(0.1)
    schedule = validate_schedule({"kind": "step", "boundaries": [3]}, 6)
    ref_losses, history = _by_hand(toy, [(optax.adabelief(step_size(schedule, EPS, 6)), 6)])

    results = _svi(toy, 6, schedule=schedule)

    assert list(np.asarray(results.losses)) == ref_losses
    _same(results.params, history[-1])


def test_a_schedule_traces_the_step_once():
    toy = _toy_problem(0.1)
    calls = _counting(toy)
    _svi(toy, 6, schedule=validate_schedule({"kind": "step", "boundaries": [3]}, 6))
    assert calls["n"] == 1


def test_a_schedule_keeps_the_moments_across_the_drop():
    """Unlike dual_run, which restarts them: the two diverge after the boundary only."""
    toy = _toy_problem(0.1)
    dual = np.asarray(_svi(toy, 3, dual_run=True).losses)
    scheduled = np.asarray(
        _svi(toy, 6, schedule=validate_schedule({"kind": "step", "boundaries": [3]}, 6)).losses
    )
    np.testing.assert_allclose(scheduled[:3], dual[:3], rtol=1e-6)
    assert not np.allclose(scheduled[3:], dual[3:], rtol=1e-9, atol=0)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dual_run", [False, True], ids=["schedule", "dual_run"])
def test_checkpoints_are_written_at_their_steps_and_reload(tmp_path, dual_run):
    toy = _toy_problem(0.1)
    if dual_run:
        kwargs = {"dual_run": True}
        phases = [(optax.adabelief(EPS), 3), (optax.adabelief(EPS / 10), 3)]
    else:
        schedule = validate_schedule({"kind": "step", "boundaries": [3]}, 6)
        kwargs = {"schedule": schedule}
        phases = [(optax.adabelief(step_size(schedule, EPS, 6)), 6)]
    ref_losses, history = _by_hand(toy, phases)

    steps = (2, 4, 6)  # 4 is in dual_run's second phase
    results = _svi(
        toy, 3 if dual_run else 6, checkpoints=steps,
        checkpoint_path=lambda step: checkpoint_path(str(tmp_path), "map_pred_Custom", step),
        **kwargs,
    )

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        f"map_pred_Custom_step{s}.npz" for s in steps
    ]
    for step in steps:
        params, saved_step, loss = load_checkpoint(checkpoint_path(str(tmp_path), "map_pred_Custom", step))
        assert saved_step == step
        assert loss == ref_losses[step - 1]
        _same(params, history[step - 1], suffix="")
    _same(results.params, load_checkpoint(checkpoint_path(str(tmp_path), "map_pred_Custom", 6))[0])


def test_no_checkpoint_is_written_by_default(tmp_path):
    toy = _toy_problem(0.1)
    _svi(toy, 3)
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# The stop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "losses, stop, expected",
    [
        ([4, 4, 2, 2], {"window": 2, "rtol": 0.1, "min_steps": 0}, False),
        ([2, 2, 2, 2], {"window": 2, "rtol": 0.1, "min_steps": 0}, True),
        ([2, 2, 2.1, 2.1], {"window": 2, "rtol": 0.1, "min_steps": 0}, True),
        ([2, 2, 2.1, 2.1], {"window": 2, "rtol": 0.01, "min_steps": 0}, False),
        ([2, 2, 2], {"window": 2, "rtol": 0.1, "min_steps": 0}, False),  # two windows not yet full
        ([2, 2, 2, 2], {"window": 2, "rtol": 0.1, "min_steps": 5}, False),
        ([-2, -2, -2.1, -2.1], {"window": 2, "rtol": 0.1, "min_steps": 0}, True),  # magnitude
    ],
)
def test_converged_compares_the_means_of_the_last_two_windows(losses, stop, expected):
    assert converged(losses, stop) is expected


def test_the_stop_ends_the_run_once_the_loss_stops_moving():
    toy = _toy_problem(0.1)
    stop = {"window": 10, "rtol": 1e-3, "min_steps": 50}
    schedule = validate_schedule({"kind": "constant"}, 2000)
    losses = list(np.asarray(_svi(toy, 2000, schedule=schedule, stop=stop).losses))

    assert 50 <= len(losses) < 2000
    assert converged(losses, stop)
    assert not converged(losses[:-1], stop)


def test_the_stop_waits_for_min_steps():
    toy = _toy_problem(0.1)
    stop = {"window": 2, "rtol": 1.0, "min_steps": 30}  # any two windows agree to 100 %
    schedule = validate_schedule({"kind": "constant"}, 100)
    assert len(_svi(toy, 100, schedule=schedule, stop=stop).losses) == 30


def test_a_stop_without_a_schedule_is_refused():
    toy = _toy_problem(0.1)
    with pytest.raises(ValueError, match="schedule"):
        _svi(toy, 3, stop={"window": 2, "rtol": 0.1, "min_steps": 0})


# ---------------------------------------------------------------------------
# What the config may say
# ---------------------------------------------------------------------------


def opt(**kwargs):
    return {"epsilon": 0.2, "max_iter": 1000, "dual_run": True, **kwargs}


def test_off_by_default():
    assert validate_opt(opt()) == {"schedule": None, "stop": None, "checkpoints": []}


def test_the_defaults_fill_in():
    checked = validate_opt(opt(
        schedule={"kind": "step", "boundaries": [500]},
        stop={"window": 50, "rtol": 1e-4},
    ))
    assert checked["schedule"] == {"kind": "step", "boundaries": [500], "factor": 0.1}
    assert checked["stop"] == {"window": 50, "rtol": 1e-4, "min_steps": 0}


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"schedule": "cosine"}, "opt.schedule"),
        ({"schedule": {"kind": "linear"}}, "opt.schedule"),
        ({"schedule": {"kind": "step"}}, "boundaries"),
        ({"schedule": {"kind": "step", "boundaries": [1000]}}, "max_iter"),
        ({"schedule": {"kind": "step", "boundaries": [500, 200]}}, "increase"),
        ({"schedule": {"kind": "step", "boundaries": [500], "factor": 0}}, "factor"),
        ({"schedule": {"kind": "step", "boundaries": [500], "decay": 0.1}}, "decay"),
        ({"schedule": {"kind": "exponential"}}, "final_factor"),
        ({"schedule": {"kind": "exponential", "final_factor": 0}}, "final_factor"),
        ({"schedule": {"kind": "cosine", "final_factor": 1.5}}, "final_factor"),
        ({"stop": {"window": 10, "rtol": 1e-4}}, "needs opt.schedule"),
        ({"schedule": {"kind": "constant"}, "stop": {"window": 0, "rtol": 1e-4}}, "window"),
        ({"schedule": {"kind": "constant"}, "stop": {"window": 10, "rtol": True}}, "rtol"),
        ({"schedule": {"kind": "constant"}, "stop": {"window": 10}}, "rtol"),
        ({"checkpoints": [0]}, "checkpoints"),
        ({"checkpoints": [1.5]}, "checkpoints"),
        ({"checkpoints": 1000}, "checkpoints"),
        ({"checkpoints": [2001]}, "2001"),  # dual_run runs 2 * max_iter
        ({"schedule": {"kind": "constant"}, "checkpoints": [1001]}, "1001"),
    ],
)
def test_a_malformed_opt_key_is_refused_by_name(kwargs, match):
    with pytest.raises(ValueError, match=match):
        validate_opt(opt(**kwargs))


def test_dual_run_checkpoints_may_reach_into_the_second_phase():
    assert validate_opt(opt(checkpoints=[2000, 1000]))["checkpoints"] == [1000, 2000]


def test_the_base_config_ships_every_new_key_off():
    from importlib.resources import files

    from tabascal.config import yaml_load

    base = yaml_load(str(files("tabascal").joinpath("data/config/tab_config_base.yaml")))["opt"]
    assert base["schedule"] is None and base["stop"] is None and base["checkpoints"] is None
    assert base["dual_run"] is True


@pytest.mark.parametrize(
    "kwargs, says",
    [
        ({}, "(dual_run)"),
        ({"schedule": {"kind": "step", "boundaries": [1000]}, "max_iter": 2000}, "one compiled step"),
        ({"schedule": {"kind": "constant"}, "stop": {"window": 10, "rtol": 1e-4}}, "stop when"),
        ({"checkpoints": [500]}, "saved at steps [500]"),
    ],
)
def test_the_log_line_names_the_optimiser(kwargs, says):
    assert says in describe_schedule(opt(**kwargs))


# ---------------------------------------------------------------------------
# End to end: the runner writes the checkpoints beside the MAP prediction
# ---------------------------------------------------------------------------

E2E_COMPONENTS = [
    "trajectory:FixedOrbitFine",
    "rfi_signal:ComplexRFIVarAntFine",
    "rfi_vis:RiemannVisFine",
    "ast_vis:GPVisAst",
    "gains:UnitaryGains",
]


@pytest.fixture(scope="module")
def scheduled_fit(tmp_path_factory):
    """Four scheduled MAP steps on the 8A sim, with checkpoints at 2 and 4; ~35 s."""
    import subprocess
    import sys
    from importlib.resources import files as _res_files
    from pathlib import Path

    import jax
    import tabsim
    from huggingface_hub import snapshot_download

    from tabascal.config import yaml_load
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

    work = tmp_path_factory.mktemp("scheduled")
    out_dir = _copy_sim(data, work)
    config_path = work / "tab_target.yaml"
    read_and_modify_yaml(
        {
            "model": {"components": E2E_COMPONENTS, "precision": precision},
            "inference": {"opt": True},
            "opt": {
                "epsilon": 1e-3, "max_iter": 4, "dual_run": True, "guide": "map",
                "schedule": {"kind": "step", "boundaries": [2]},
                "checkpoints": [2, 4],
            },
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
    assert result.returncode == 0, result.stderr
    return result.stdout, out_dir


def test_the_runner_writes_each_checkpoint_beside_the_map_prediction(scheduled_fit):
    import xarray as xr

    stdout, out_dir = scheduled_fit
    results = out_dir / "results"
    (map_zarr,) = results.glob("map_pred*.zarr")
    stem = map_zarr.name.removesuffix(".zarr")

    assert sorted(p.name for p in (results / "checkpoints").iterdir()) == [
        f"{stem}_step2.npz", f"{stem}_step4.npz"
    ]
    assert "Optimiser : adabelief, step size 0.001, x0.1 at steps [2], 4 steps, one compiled step" in stdout
    # dual_run: true is not read with a schedule: four steps, not eight.
    params, step, _ = load_checkpoint(str(results / "checkpoints" / f"{stem}_step4.npz"))
    assert step == 4 and params
    assert xr.open_zarr(str(map_zarr)).sizes  # the run finished its writes
