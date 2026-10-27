"""One optimiser for the whole fit: a step-size schedule, a stop and checkpoints.

``opt.dual_run`` fits in two phases, each with its own ``adabelief``. The
optimiser is a static argument of the compiled step, so the second phase
compiles again and starts from fresh moments, which shows as a bump in the
loss at the switch (GitHub #82). ``opt.schedule`` builds one optimiser whose
step size follows a schedule instead: one compiled step, and moments that run
through every step.

``opt.stop`` ends the fit once the loss has stopped moving, and
``opt.checkpoints`` writes the parameters at given steps, so one long run can
be scored at several step counts. All three are off by default, and with all
three off the optimiser is the two-phase one it always was.
"""

import math
import os
from numbers import Integral, Real
from typing import Dict, List, Optional, Sequence

import jax
import numpy as np
import optax

#: The schedules ``opt.schedule.kind`` names.
SCHEDULE_KINDS = ("constant", "step", "exponential", "cosine")

#: The keys each schedule takes, and the default of each optional one.
_SCHEDULE_KEYS = {
    "constant": {},
    "step": {"boundaries": None, "factor": 0.1},
    "exponential": {"final_factor": None},
    "cosine": {"final_factor": 0.0},
}

#: The keys ``opt.stop`` takes, and the default of each optional one.
_STOP_KEYS = {"window": None, "rtol": None, "min_steps": 0}


def _number(value, key: str, low: float, high: float = math.inf, low_open=False) -> float:
    # A bool is a Real to Python, and `factor: true` was not meant as 1.
    ok = (
        not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)
        and (value > low if low_open else value >= low) and value <= high
    )
    if not ok:
        bound = f"{'>' if low_open else '>='} {low:g}" + (f" and <= {high:g}" if high < math.inf else "")
        raise ValueError(f"opt.{key} must be a finite number {bound}, not {value!r}.")
    return float(value)


def _count(value, key: str, low: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < low:
        raise ValueError(f"opt.{key} must be an integer >= {low}, not {value!r}.")
    return int(value)


def _keys(section, allowed: Dict, key: str) -> Dict:
    """``section`` with its defaults filled in; unknown and missing keys refused by name."""
    if not isinstance(section, dict):
        raise ValueError(f"opt.{key} must be a mapping or null, not {section!r}.")
    unknown = sorted(set(section) - set(allowed))
    if unknown:
        raise ValueError(f"opt.{key} does not take {unknown}; it takes {sorted(allowed)}.")
    out = {k: section.get(k, default) for k, default in allowed.items()}
    missing = sorted(k for k, v in out.items() if v is None)
    if missing:
        raise ValueError(f"opt.{key} needs {missing}.")
    return out


def validate_schedule(schedule, max_iter: int) -> Optional[Dict]:
    """``opt.schedule`` with its defaults filled in, or None when it is off."""
    if schedule is None:
        return None
    if not isinstance(schedule, dict) or schedule.get("kind") not in SCHEDULE_KINDS:
        raise ValueError(
            f"opt.schedule must be null or a mapping whose kind is one of "
            f"{list(SCHEDULE_KINDS)}, not {schedule!r}."
        )
    kind = schedule["kind"]
    out = _keys({k: v for k, v in schedule.items() if k != "kind"}, _SCHEDULE_KEYS[kind], "schedule")

    if kind == "step":
        boundaries = out["boundaries"]
        if not isinstance(boundaries, (list, tuple)) or not boundaries:
            raise ValueError(f"opt.schedule.boundaries must be a non-empty list of steps, not {boundaries!r}.")
        boundaries = [_count(b, "schedule.boundaries", 1) for b in boundaries]
        if boundaries != sorted(set(boundaries)) or boundaries[-1] >= max_iter:
            raise ValueError(
                f"opt.schedule.boundaries must increase strictly and stay below "
                f"max_iter ({max_iter}), not {boundaries}."
            )
        out["boundaries"] = boundaries
        out["factor"] = _number(out["factor"], "schedule.factor", 0, low_open=True)
    elif kind == "exponential":
        out["final_factor"] = _number(out["final_factor"], "schedule.final_factor", 0, 1, low_open=True)
    elif kind == "cosine":
        out["final_factor"] = _number(out["final_factor"], "schedule.final_factor", 0, 1)
    return {"kind": kind, **out}


def validate_stop(stop) -> Optional[Dict]:
    """``opt.stop`` with its defaults filled in, or None when it is off."""
    if stop is None:
        return None
    out = _keys(stop, _STOP_KEYS, "stop")
    out["window"] = _count(out["window"], "stop.window", 1)
    out["rtol"] = _number(out["rtol"], "stop.rtol", 0, low_open=True)
    out["min_steps"] = _count(out["min_steps"], "stop.min_steps", 0)
    return out


def total_steps(opt_config: Dict) -> int:
    """Steps the fit runs if nothing stops it: one phase with a schedule, else one or two."""
    if opt_config.get("schedule") is not None:
        return opt_config["max_iter"]
    return opt_config["max_iter"] * (2 if opt_config["dual_run"] else 1)


def validate_checkpoints(checkpoints, n_steps: int) -> List[int]:
    """``opt.checkpoints`` as a sorted list of steps, empty when it is off."""
    if checkpoints is None:
        return []
    if not isinstance(checkpoints, (list, tuple)):
        raise ValueError(f"opt.checkpoints must be a list of steps or null, not {checkpoints!r}.")
    steps = sorted({_count(s, "checkpoints", 1) for s in checkpoints})
    if steps and steps[-1] > n_steps:
        raise ValueError(
            f"opt.checkpoints asks for step {steps[-1]}, but the fit runs {n_steps} "
            "steps at most."
        )
    return steps


def validate_opt(opt_config: Dict) -> Dict:
    """Check ``opt.schedule``, ``opt.stop`` and ``opt.checkpoints`` before the run starts.

    Returns them normalised, under the same keys. ``stop`` needs ``schedule``:
    with ``dual_run`` it is not clear whether a stop should end the first
    phase or the fit.
    """
    max_iter = opt_config["max_iter"]
    schedule = validate_schedule(opt_config.get("schedule"), max_iter)
    stop = validate_stop(opt_config.get("stop"))
    if stop is not None and schedule is None:
        raise ValueError(
            "opt.stop needs opt.schedule: with the two-phase dual_run optimiser it is "
            "not clear whether it should end the first phase or the fit. Set a "
            "schedule, e.g. {kind: step, boundaries: [N], factor: 0.1} for dual_run's drop."
        )
    checkpoints = validate_checkpoints(opt_config.get("checkpoints"), total_steps(opt_config))
    return {"schedule": schedule, "stop": stop, "checkpoints": checkpoints}


def step_size(schedule: Dict, epsilon: float, max_iter: int):
    """The optax schedule ``opt.schedule`` describes, starting from ``epsilon``.

    Evaluated at the optimiser's count, the number of steps already taken, so a
    ``step`` boundary at N keeps the first N steps at ``epsilon`` and drops from
    step N + 1 on -- where ``dual_run`` drops for ``max_iter: N``.
    """
    kind = schedule["kind"]
    if kind == "constant":
        return optax.constant_schedule(epsilon)
    if kind == "step":
        return optax.piecewise_constant_schedule(
            epsilon, {b: schedule["factor"] for b in schedule["boundaries"]}
        )
    if kind == "exponential":
        return optax.exponential_decay(epsilon, max_iter, schedule["final_factor"])
    return optax.cosine_decay_schedule(epsilon, max_iter, alpha=schedule["final_factor"])


def describe_schedule(opt_config: Dict) -> str:
    """The optimiser as the run log names it."""
    epsilon, max_iter = opt_config["epsilon"], opt_config["max_iter"]
    checked = validate_opt(opt_config)
    schedule = checked["schedule"]
    if schedule is None:
        line = f"adabelief, step size {epsilon:g} for {max_iter} steps"
        if opt_config["dual_run"]:
            line += f", then a new adabelief at {epsilon / 10:g} for {max_iter} (dual_run)"
    else:
        kind = schedule["kind"]
        detail = {
            "constant": "constant",
            "step": f"x{schedule.get('factor', 0):g} at steps {schedule.get('boundaries')}",
            "exponential": f"exponential decay to x{schedule.get('final_factor', 0):g}",
            "cosine": f"cosine decay to x{schedule.get('final_factor', 0):g}",
        }[kind]
        line = f"adabelief, step size {epsilon:g}, {detail}, {max_iter} steps, one compiled step"
    if checked["stop"]:
        s = checked["stop"]
        line += (
            f"; stop when the mean loss moves by < {s['rtol']:g} (relative) between "
            f"windows of {s['window']} steps, after {s['min_steps']}"
        )
    if checked["checkpoints"]:
        line += f"; parameters saved at steps {checked['checkpoints']}"
    return line


def converged(losses: Sequence[float], stop: Dict) -> bool:
    """True once the mean loss of the last window is within ``rtol`` of the one before.

    Relative to the magnitude of the earlier window's mean. Needs ``min_steps``
    steps and two full windows first.
    """
    n, w = len(losses), stop["window"]
    if n < max(stop["min_steps"], 2 * w):
        return False
    before = sum(losses[n - 2 * w:n - w]) / w
    after = sum(losses[n - w:]) / w
    return abs(after - before) <= stop["rtol"] * abs(before)


def _to_host(x) -> np.ndarray:
    # A leaf sharded across processes is not addressable here as a whole; the
    # gather is a collective, so every rank must reach it, not only rank 0.
    if isinstance(x, jax.Array) and not x.is_fully_addressable:
        from jax.experimental import multihost_utils
        return np.asarray(multihost_utils.process_allgather(x, tiled=True))
    return np.asarray(x)


def checkpoint_path(directory: str, stem: str, step: int) -> str:
    return os.path.join(directory, f"{stem}_step{step}.npz")


def save_checkpoint(path: str, params: Dict, step: int, loss: float, write: bool) -> None:
    """Write the parameters after ``step`` steps, with that step's loss.

    Every rank gathers, since that can be a collective; only ``write`` (rank 0)
    writes. Parameter names never start with an underscore, so the two
    ``_``-keys cannot collide with them.
    """
    arrays = {name: _to_host(value) for name, value in params.items()}
    if not write:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, _step=step, _loss=loss, **arrays)


def load_checkpoint(path: str):
    """``(params, step, loss)`` from a file :func:`save_checkpoint` wrote.

    ``params`` are keyed as the model names them, without ``_auto_loc``, which
    is what ``Predictive`` and ``init_predict`` take.
    """
    with np.load(path) as f:
        params = {k: f[k] for k in f.files if not k.startswith("_")}
        return params, int(f["_step"]), float(f["_loss"])
