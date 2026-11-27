"""``opt.optimizer``, ``opt.schedule``, ``opt.stop`` and ``opt.checkpoints`` (GitHub #82).

- **Default path.** With ``optimizer`` and ``schedule`` both null, the fit runs
  one ``adabelief`` at constant ``epsilon``, bit for bit.
- **One compilation.** One optimiser for the whole fit traces the step once.
- **The step sizes are the ones asked for**, and the optimiser asked for takes them.
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
    validate_optimizer,
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
    kwargs.setdefault("epsilon", EPS)
    return run_custom_svi(
        prob_model=toy.prob_model,
        obs_data=toy.vis_obs,
        max_iter=max_iter,
        init_params=toy.init_params,
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
# Default (unscheduled) path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("eps", [EPS, 1])
def test_with_no_schedule_runs_constant_adabelief(eps):
    toy = _toy_problem(0.1)
    ref_losses, history = _by_hand(toy, [(optax.adabelief(float(eps)), 6)])

    results = _svi(toy, 6, epsilon=eps)

    assert list(np.asarray(results.losses)) == ref_losses
    _same(results.params, history[-1])


def test_unscheduled_run_traces_the_step_once():
    toy = _toy_problem(0.1)
    calls = _counting(toy)
    _svi(toy, 6)
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# The schedule
# ---------------------------------------------------------------------------


def test_a_step_schedule_holds_epsilon_for_n_steps_then_drops():
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


@pytest.mark.parametrize(
    "schedule, max_iter, expected",
    [
        ({"kind": "linear", "final_factor": 0.2}, 4, [1, 0.8, 0.6, 0.4, 0.2]),
        ({"kind": "step", "boundaries": [1, 3], "factor": [0.5, 0.1]}, 4, [1, 0.5, 0.5, 0.05, 0.05]),
        # Boundaries count from the start of the fit, warmup included.
        ({"kind": "step", "boundaries": [3], "warmup_steps": 2}, 4, [0, 0.5, 1, 0.1, 0.1]),
        # The decay runs over the steps the warmup leaves, and still ends at max_iter.
        ({"kind": "cosine", "warmup_steps": 2, "warmup_factor": 0.5}, 4, [0.5, 0.75, 1, 0.5, 0]),
    ],
    ids=["linear", "factor-per-boundary", "step-after-warmup", "cosine-after-warmup"],
)
def test_the_other_schedules_give_their_step_sizes(schedule, max_iter, expected):
    sizes = step_size(validate_schedule(schedule, max_iter), EPS, max_iter)
    np.testing.assert_allclose(
        [float(sizes(c)) for c in range(max_iter + 1)], EPS * np.array(expected), rtol=1e-6, atol=1e-12
    )


@pytest.mark.parametrize("with_schedule", [False, True], ids=["constant", "schedule"])
def test_the_run_takes_the_optimiser_asked_for(with_schedule):
    """Its arguments and the clip included, with or without a schedule, bit for bit."""
    toy = _toy_problem(0.1)
    optimizer = validate_optimizer({"name": "adamw", "b1": 0.8, "weight_decay": 1e-3, "clip_norm": 0.5})

    def adamw(lr):
        return optax.chain(optax.clip_by_global_norm(0.5), optax.adamw(lr, b1=0.8, weight_decay=1e-3))

    if with_schedule:
        schedule = validate_schedule({"kind": "cosine", "warmup_steps": 2}, 6)
        ref_losses, history = _by_hand(toy, [(adamw(step_size(schedule, EPS, 6)), 6)])
        results = _svi(toy, 6, schedule=schedule, optimizer=optimizer)
    else:
        ref_losses, history = _by_hand(toy, [(adamw(EPS), 6)])
        results = _svi(toy, 6, optimizer=optimizer)

    assert list(np.asarray(results.losses)) == ref_losses
    _same(results.params, history[-1])


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


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_schedule", [False, True], ids=["constant", "schedule"])
def test_checkpoints_are_written_at_their_steps_and_reload(tmp_path, with_schedule):
    toy = _toy_problem(0.1)
    if with_schedule:
        schedule = validate_schedule({"kind": "step", "boundaries": [3]}, 6)
        kwargs = {"schedule": schedule}
        phases = [(optax.adabelief(step_size(schedule, EPS, 6)), 6)]
    else:
        kwargs = {}
        phases = [(optax.adabelief(EPS), 6)]
    ref_losses, history = _by_hand(toy, phases)

    steps = (2, 4, 6)
    results = _svi(
        toy, 6, checkpoints=steps,
        checkpoint_path=lambda step: checkpoint_path(str(tmp_path), "map_pred_Custom", step),
        **kwargs,
    )

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        f"map_pred_Custom_step{s}.npz" for s in steps
    ]
    for step in steps:
        params, saved_step, trace_loss = load_checkpoint(checkpoint_path(str(tmp_path), "map_pred_Custom", step))
        assert saved_step == step
        assert trace_loss == ref_losses[step - 1]  # the trace's entry, evaluated before the update
        _same(params, history[step - 1], suffix="")
    _same(results.params, load_checkpoint(checkpoint_path(str(tmp_path), "map_pred_Custom", 6))[0])


def test_a_failed_checkpoint_write_is_raised_on_every_rank(tmp_path, monkeypatch):
    """Rank 0 hands its error to fail_together; a rank that does not write still meets it."""
    import tabascal.opt_schedule as opt_schedule
    from tabascal.opt_schedule import save_checkpoint

    seen = []
    monkeypatch.setattr(opt_schedule, "fail_together", lambda failure, what: seen.append(failure))
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("")
    params = {"a": np.zeros(3)}

    save_checkpoint(str(blocker / "x_step1.npz"), params, 1, 0.0, write=True)
    save_checkpoint(str(blocker / "x_step1.npz"), params, 1, 0.0, write=False)

    assert isinstance(seen[0], OSError) and seen[1] is None


def test_a_failed_checkpoint_write_raises_in_a_single_process_run(tmp_path):
    from tabascal.opt_schedule import save_checkpoint

    blocker = tmp_path / "not_a_dir"
    blocker.write_text("")
    with pytest.raises(OSError):
        save_checkpoint(str(blocker / "x_step1.npz"), {"a": np.zeros(3)}, 1, 0.0, write=True)


def test_no_checkpoint_is_written_by_default(monkeypatch):
    calls = []
    monkeypatch.setattr(tab_tools, "save_checkpoint", lambda *a, **k: calls.append(a))
    _svi(_toy_problem(0.1), 3)
    assert calls == []


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
        ([float("inf"), 1.0], {"window": 1, "rtol": 1e-4, "min_steps": 0}, False),
        ([float("-inf"), 1.0], {"window": 1, "rtol": 1e-4, "min_steps": 0}, False),
        ([1.0, float("nan")], {"window": 1, "rtol": 1e-4, "min_steps": 0}, False),
        ([1e308, 1e308, 1.0, 1.0], {"window": 2, "rtol": 1e-4, "min_steps": 0}, False),  # sum overflows
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


def test_the_stop_works_without_an_explicit_schedule():
    toy = _toy_problem(0.1)
    stop = {"window": 2, "rtol": 1.0, "min_steps": 10}
    assert len(_svi(toy, 100, stop=stop).losses) == 10


# ---------------------------------------------------------------------------
# What the config may say
# ---------------------------------------------------------------------------


def opt(**kwargs):
    return {"epsilon": 0.2, "max_iter": 1000, **kwargs}


def test_off_by_default():
    assert validate_opt(opt()) == {"optimizer": None, "schedule": None, "stop": None, "checkpoints": []}


def test_the_defaults_fill_in():
    checked = validate_opt(opt(
        schedule={"kind": "step", "boundaries": [500]},
        stop={"window": 50, "rtol": 1e-4},
    ))
    assert checked["schedule"] == {
        "kind": "step", "boundaries": [500], "factor": 0.1, "warmup_steps": 0, "warmup_factor": 0.0
    }
    assert checked["stop"] == {"window": 50, "rtol": 1e-4, "min_steps": 0}
    assert validate_optimizer("adam") == {"name": "adam", "clip_norm": None, "args": {}}
    assert validate_optimizer({"b1": 0.95}) == {"name": "adabelief", "clip_norm": None, "args": {"b1": 0.95}}


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"schedule": "cosine"}, "opt.schedule"),
        ({"schedule": {"kind": "polynomial"}}, "opt.schedule"),
        ({"schedule": {"kind": "step"}}, "boundaries"),
        ({"schedule": {"kind": "step", "boundaries": [1000]}}, "max_iter"),
        ({"schedule": {"kind": "step", "boundaries": [500, 200]}}, "increase"),
        ({"schedule": {"kind": "step", "boundaries": [500], "factor": 0}}, "factor"),
        ({"schedule": {"kind": "step", "boundaries": [500], "decay": 0.1}}, "decay"),
        ({"schedule": {"kind": "exponential"}}, "final_factor"),
        ({"schedule": {"kind": "exponential", "final_factor": 0}}, "final_factor"),
        ({"schedule": {"kind": "cosine", "final_factor": 1.5}}, "final_factor"),
        ({"schedule": {"kind": "step", "boundaries": [200, 500], "factor": [0.1]}}, "one per boundary"),
        ({"schedule": {"kind": "constant", "warmup_steps": 1000}}, "warmup_steps"),
        ({"schedule": {"kind": "step", "boundaries": [100], "warmup_steps": 100}}, "after warmup_steps"),
        ({"schedule": {"kind": "constant", "warmup_factor": 2}}, "warmup_factor"),
        ({"optimizer": "adamm"}, "opt.optimizer.name"),
        ({"optimizer": ["adam"]}, "opt.optimizer"),
        ({"optimizer": {"name": "adam", "learning_rate": 0.1}}, "opt.epsilon"),
        ({"optimizer": {"name": "sgd", "b1": 0.9}}, r"\['b1'\]"),
        ({"optimizer": {"name": "sgd", "nesterov": 1}}, "true or false"),
        ({"optimizer": {"name": "adam", "b1": True}}, "b1"),
        ({"optimizer": {"name": "adam", "eps": None}}, "eps"),
        ({"optimizer": {"name": "adam", "b1": 1}}, "b1"),
        ({"optimizer": {"name": "adam", "b2": -0.1}}, "b2"),
        ({"optimizer": {"name": "adam", "eps_root": -1}}, "eps_root"),
        ({"optimizer": {"name": "sgd", "momentum": 1.0}}, "momentum"),
        ({"optimizer": {"name": "adamw", "weight_decay": -1e-4}}, "weight_decay"),
        ({"optimizer": {"name": "radam", "threshold": 4}}, "threshold"),
        ({"optimizer": {"name": "adam", "eps": 0}}, "eps"),
        ({"optimizer": {"name": "adam", "eps": 1e-400}}, "eps"),  # 0 once parsed
        ({"optimizer": {"name": "rmsprop", "centered": True, "bias_correction": True}}, "centered"),
        ({"optimizer": {"clip_norm": 0}}, "clip_norm"),
        ({"stop": {"window": 0, "rtol": 1e-4}}, "window"),
        ({"stop": {"window": 10, "rtol": True}}, "rtol"),
        ({"stop": {"window": 10}}, "rtol"),
        ({"checkpoints": [0]}, "checkpoints"),
        ({"checkpoints": [1.5]}, "checkpoints"),
        ({"checkpoints": 1000}, "checkpoints"),
        ({"checkpoints": [1001]}, "1001"),
        ({"schedule": {"kind": "constant"}, "checkpoints": [1001]}, "1001"),
    ],
)
def test_a_malformed_opt_key_is_refused_by_name(kwargs, match):
    with pytest.raises(ValueError, match=match):
        validate_opt(opt(**kwargs))


def test_every_numeric_optimiser_argument_has_a_range():
    import inspect

    from tabascal import opt_schedule as o

    for name in o.OPTIMIZERS:
        params = inspect.signature(getattr(optax, name)).parameters
        numeric = {k for k, p in params.items() if k not in o._NOT_FROM_CONFIG and not isinstance(p.default, bool)}
        assert numeric <= set(o._ARG_RANGES), (name, numeric - set(o._ARG_RANGES))


@pytest.mark.parametrize(
    "optimizer",
    [
        {"name": "adam", "b1": 0.99999999},  # 1
        {"name": "adam", "eps": 1e-40},  # subnormal
        {"name": "sgd", "clip_norm": 1e-50},  # 0
        {"name": "adamw", "weight_decay": 1e40},  # inf
    ],
)
def test_single_precision_checks_the_value_it_will_hold(optimizer, monkeypatch):
    import jax

    monkeypatch.setattr(jax.config, "read", lambda name: True)
    validate_optimizer(optimizer)  # fine in double precision
    monkeypatch.setattr(jax.config, "read", lambda name: False)
    with pytest.raises(ValueError, match="opt.optimizer"):
        validate_optimizer(optimizer)


def test_checkpoints_are_sorted_and_deduplicated():
    assert validate_opt(opt(checkpoints=[1000, 500]))["checkpoints"] == [500, 1000]


def test_the_schema_default_ships_every_new_key_off():
    from tabascal.schema import OptConfig

    cfg = OptConfig()
    assert all(getattr(cfg, k) is None for k in ("optimizer", "schedule", "stop", "checkpoints"))


@pytest.mark.parametrize(
    "kwargs, says",
    [
        ({}, "adabelief, step size 0.2 for 1000 steps"),
        ({"schedule": {"kind": "step", "boundaries": [1000]}, "max_iter": 2000}, "one compiled step"),
        ({"stop": {"window": 10, "rtol": 1e-4}}, "stop when"),
        ({"checkpoints": [500]}, "saved at steps [500]"),
        ({"optimizer": {"name": "sgd", "momentum": 0.9, "clip_norm": 1}},
         "sgd (momentum=0.9, gradient norm clipped at 1), step size 0.2 for 1000 steps"),
        ({"schedule": {"kind": "linear", "warmup_steps": 10}},
         "warmup from x0 over 10 steps, then linear decay to x0"),
        ({"schedule": {"kind": "step", "boundaries": [200, 500], "factor": [0.5, 0.1]}}, "x0.5/0.1 at steps"),
    ],
)
def test_the_log_line_names_the_optimiser(kwargs, says):
    assert says in describe_schedule(opt(**kwargs))


# ---------------------------------------------------------------------------
# End to end: the runner writes the checkpoints beside the MAP prediction
# ---------------------------------------------------------------------------

E2E_COMPONENTS = [
    "trajectory.FixedOrbitFine",
    "rfi_signal.ComplexRFIVarAntFine",
    "rfi_vis.RiemannVisFine",
    "ast_vis.GPVisAst",
    "gains.UnitaryGains",
]


@pytest.fixture(scope="module")
def scheduled_fit(tmp_path_factory):
    """Four scheduled, clipped adam steps on the 8A sim, with checkpoints at 2 and 4; ~35 s."""
    import subprocess
    import sys
    from importlib.resources import files as _res_files
    from pathlib import Path

    import jax

    from tabascal.config import yaml_load
    from tests.test_tabascal_pipeline import (
        _copy_sim, download_test_simulations, read_and_modify_yaml,
    )

    try:
        data = download_test_simulations()
    except Exception as e:  # pragma: no cover - offline
        pytest.skip(f"8A simulation not available: {e}")

    template = Path(__file__).parent / "data" / "tab_target.yaml"
    base = yaml_load(template)
    script = Path(__file__).parent.parent / "tabascal" / "scripts" / "run_tabascal.py"
    precision = "double" if jax.config.read("jax_enable_x64") else "single"

    def _with_prior_init(comps):
        out = []
        for entry in comps:
            if isinstance(entry, dict):
                ((ref, params),) = entry.items()
                params = dict(params or {})
                if ref.startswith(("rfi_signal.", "ast_vis.GPVisAst")):
                    params["init"] = "prior"
                out.append({ref: params})
            else:
                out.append(entry)
        return out

    work = tmp_path_factory.mktemp("scheduled")
    out_dir = _copy_sim(data, work)
    config_path = work / "tab_target.yaml"
    read_and_modify_yaml(
        {
            "model": {
                "components": _with_prior_init(base["model"]["components"]),
                "precision": precision,
            },
            "opt": {
                "epsilon": 1e-3, "max_iter": 4,
                "optimizer": {"name": "adam", "clip_norm": 10.0},
                "schedule": {"kind": "step", "boundaries": [2]},
                "checkpoints": [2, 4],
            },
        },
        template,
        config_path,
    )
    result = subprocess.run(
        [
            sys.executable, str(script), "run",
            "-c", str(config_path),
            "-od", str(out_dir),
            "-n", "sched",
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
    (run_dir,) = (out_dir / "runs").glob("sched_*")
    results = run_dir / "results"
    map_zarr = results / "map_pred.zarr"
    stem = "map_pred"

    assert sorted(p.name for p in (results / "checkpoints").iterdir()) == [
        f"{stem}_step2.npz", f"{stem}_step4.npz"
    ]
    assert ("Optimiser : adam (gradient norm clipped at 10), step size 0.001, x0.1 at steps [2], "
            "4 steps, one compiled step") in stdout
    params, step, _ = load_checkpoint(str(results / "checkpoints" / f"{stem}_step4.npz"))
    assert step == 4 and params
    assert xr.open_zarr(str(map_zarr)).sizes  # the run finished its writes
