"""One optimiser for the whole fit: its kind, a step-size schedule, a stop and checkpoints.

``opt.dual_run`` fits in two phases. The second restarts the optimiser at a
tenth of the step size, from fresh moments, which GitHub #82 reports as a
bump in the loss at the switch. ``opt.schedule`` builds one optimiser whose
step size follows a schedule instead, with moments that run through every
step.

``opt.optimizer`` picks the optax optimiser and its hyperparameters, with
or without a schedule, and can clip the gradient's global norm. ``opt.stop``
ends the fit once the loss has stopped moving, and ``opt.checkpoints`` writes
the parameters at given steps, so one long run can be scored at several step
counts. All four are off by default, and with all four off the optimiser is the
two-phase ``adabelief`` it always was.
"""

import inspect
import math
import os
from numbers import Integral, Real
from typing import Dict, List, Optional, Sequence

import jax
import numpy as np
import optax

from tabascal.distributed import fail_together

#: The optax optimisers ``opt.optimizer.name`` names. Each takes the step size
#: as its first argument; their other scalar arguments may be set in the config.
OPTIMIZERS = (
    "adabelief", "adam", "adamw", "amsgrad", "nadam", "radam", "adamax", "yogi",
    "lion", "lamb", "novograd", "rmsprop", "sgd", "adagrad",
)

#: Arguments of those optimisers that are not scalars a YAML file can give.
_NOT_FROM_CONFIG = ("learning_rate", "mask", "mu_dtype", "accumulator_dtype")

#: The range each of their numeric arguments must lie in, as ``(low, high,
#: low_open)`` with ``high`` excluded: a decay rate of 1 never forgets its
#: start, and the first update divides by zero, as it does without a positive
#: eps. radam's rectification takes the
#: square root of ``rho_t - 4`` once ``rho_t`` passes ``threshold``, so that
#: must be above 4. Every numeric argument of OPTIMIZERS is here.
_ARG_RANGES = {
    **dict.fromkeys(("b1", "b2", "b3", "decay", "momentum"), (0.0, 1.0, False)),
    # eps keeps the update's denominator off zero; eps_root only shifts it.
    "eps": (0.0, math.inf, True),
    **dict.fromkeys(
        ("eps_root", "weight_decay", "initial_scale", "initial_accumulator_value"),
        (0.0, math.inf, False),
    ),
    "threshold": (4.0, math.inf, True),
}

#: The schedules ``opt.schedule.kind`` names.
SCHEDULE_KINDS = ("constant", "step", "exponential", "cosine", "linear")

#: The keys each schedule takes, and the default of each optional one.
_SCHEDULE_KEYS = {
    "constant": {},
    "step": {"boundaries": None, "factor": 0.1},
    "exponential": {"final_factor": None},
    "cosine": {"final_factor": 0.0},
    "linear": {"final_factor": 0.0},
}

#: Keys every schedule takes: a linear warmup from warmup_factor * epsilon.
_WARMUP_KEYS = {"warmup_steps": 0, "warmup_factor": 0.0}

#: The keys ``opt.stop`` takes, and the default of each optional one.
_STOP_KEYS = {"window": None, "rtol": None, "min_steps": 0}


def _number(value, key: str, low: float, high: float = math.inf, low_open=False, high_open=False) -> float:
    # A bool is a Real to Python, and `factor: true` was not meant as 1.
    ok = not isinstance(value, bool) and isinstance(value, Real)
    if ok:
        # Checked as the run will hold it: in single precision b1: 0.99999999
        # is 1, 1e40 is inf, and a positive 1e-40 is a subnormal that flushes to 0.
        dtype = np.float64 if jax.config.read("jax_enable_x64") else np.float32
        with np.errstate(over="ignore"):
            held = float(np.asarray(value, dtype=dtype))
        ok = (
            math.isfinite(held)
            and (held > low if low_open else held >= low)
            and (held < high if high_open else held <= high)
            and not (low_open and low == 0 and held < np.finfo(dtype).tiny)
        )
    if not ok:
        bound = f"{'>' if low_open else '>='} {low:g}" + (
            f" and {'<' if high_open else '<='} {high:g}" if high < math.inf else ""
        )
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


def validate_optimizer(optimizer) -> Optional[Dict]:
    """``opt.optimizer`` as ``{name, clip_norm, args}``, or None when it is off.

    A name alone (``adam``) is that optimiser with optax's defaults; a mapping
    adds its scalar arguments by their optax names, and ``clip_norm``. A
    mapping without ``name`` is ``adabelief``, the optimiser of the default.
    """
    if optimizer is None:
        return None
    if isinstance(optimizer, str):
        optimizer = {"name": optimizer}
    if not isinstance(optimizer, dict):
        raise ValueError(f"opt.optimizer must be a name, a mapping or null, not {optimizer!r}.")
    args = dict(optimizer)
    name = args.pop("name", "adabelief")
    if name not in OPTIMIZERS:
        raise ValueError(f"opt.optimizer.name must be one of {list(OPTIMIZERS)}, not {name!r}.")
    clip_norm = args.pop("clip_norm", None)
    if clip_norm is not None:
        clip_norm = _number(clip_norm, "optimizer.clip_norm", 0, low_open=True)

    params = inspect.signature(getattr(optax, name)).parameters
    allowed = sorted(k for k in params if k not in _NOT_FROM_CONFIG)
    unknown = sorted(set(args) - set(allowed))
    if unknown:
        raise ValueError(
            f"opt.optimizer ({name}) does not take {unknown}; it takes "
            f"{['name', 'clip_norm'] + allowed}. The step size is opt.epsilon."
        )
    for key, value in args.items():
        default = params[key].default
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise ValueError(f"opt.optimizer.{key} must be true or false, not {value!r}.")
        elif not (value is None and default is None):
            low, high, low_open = _ARG_RANGES[key]
            args[key] = _number(
                value, f"optimizer.{key}", low, high, low_open=low_open, high_open=high < math.inf
            )
    if name == "rmsprop" and args.get("centered") and args.get("bias_correction"):
        # Bias-correcting the mean and the square separately can leave their
        # difference, the variance it takes the square root of, negative.
        raise ValueError("opt.optimizer (rmsprop) cannot take centered and bias_correction together.")
    return {"name": name, "clip_norm": clip_norm, "args": args}


def build_optimizer(optimizer: Optional[Dict], learning_rate) -> optax.GradientTransformation:
    """The optax optimiser ``opt.optimizer`` describes, at ``learning_rate``.

    ``learning_rate`` is a number or an optax schedule. ``None`` is
    ``optax.adabelief(learning_rate)``, exactly what the fit always built.
    """
    if optimizer is None:
        return optax.adabelief(learning_rate)
    transform = getattr(optax, optimizer["name"])(learning_rate, **optimizer["args"])
    if optimizer["clip_norm"] is not None:
        transform = optax.chain(optax.clip_by_global_norm(optimizer["clip_norm"]), transform)
    return transform


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
    out = _keys(
        {k: v for k, v in schedule.items() if k != "kind"},
        {**_SCHEDULE_KEYS[kind], **_WARMUP_KEYS}, "schedule",
    )
    warmup = out["warmup_steps"] = _count(out["warmup_steps"], "schedule.warmup_steps", 0)
    if warmup >= max_iter:
        raise ValueError(
            f"opt.schedule.warmup_steps ({warmup}) must stay below max_iter ({max_iter})."
        )
    out["warmup_factor"] = _number(out["warmup_factor"], "schedule.warmup_factor", 0, 1)

    if kind == "step":
        boundaries = out["boundaries"]
        if not isinstance(boundaries, (list, tuple)) or not boundaries:
            raise ValueError(f"opt.schedule.boundaries must be a non-empty list of steps, not {boundaries!r}.")
        boundaries = [_count(b, "schedule.boundaries", 1) for b in boundaries]
        if boundaries != sorted(set(boundaries)) or boundaries[-1] >= max_iter or boundaries[0] <= warmup:
            raise ValueError(
                f"opt.schedule.boundaries must increase strictly, stay below "
                f"max_iter ({max_iter}) and come after warmup_steps ({warmup}), not {boundaries}."
            )
        out["boundaries"] = boundaries
        factor = out["factor"]
        if isinstance(factor, (list, tuple)):
            if len(factor) != len(boundaries):
                raise ValueError(
                    f"opt.schedule.factor must be one number, or one per boundary "
                    f"({len(boundaries)}), not {factor!r}."
                )
            out["factor"] = [_number(f, "schedule.factor", 0, low_open=True) for f in factor]
        else:
            out["factor"] = _number(factor, "schedule.factor", 0, low_open=True)
    elif kind == "exponential":
        out["final_factor"] = _number(out["final_factor"], "schedule.final_factor", 0, 1, low_open=True)
    elif kind in ("cosine", "linear"):
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
    """Check ``opt.optimizer``, ``opt.schedule``, ``opt.stop`` and ``opt.checkpoints`` before the run starts.

    Returns them normalised, under the same keys. ``stop`` needs ``schedule``:
    with ``dual_run`` it is not clear whether a stop should end the first
    phase or the fit.
    """
    max_iter = opt_config["max_iter"]
    optimizer = validate_optimizer(opt_config.get("optimizer"))
    schedule = validate_schedule(opt_config.get("schedule"), max_iter)
    stop = validate_stop(opt_config.get("stop"))
    if stop is not None and schedule is None:
        raise ValueError(
            "opt.stop needs opt.schedule: with the two-phase dual_run optimiser it is "
            "not clear whether it should end the first phase or the fit. Set a "
            "schedule, e.g. {kind: step, boundaries: [N], factor: 0.1} for dual_run's drop."
        )
    checkpoints = validate_checkpoints(opt_config.get("checkpoints"), total_steps(opt_config))
    return {"optimizer": optimizer, "schedule": schedule, "stop": stop, "checkpoints": checkpoints}


def step_size(schedule: Dict, epsilon: float, max_iter: int):
    """The optax schedule ``opt.schedule`` describes, starting from ``epsilon``.

    Evaluated at the optimiser's count, the number of steps already taken, so a
    ``step`` boundary at N keeps the first N steps at ``epsilon`` and drops from
    step N + 1 on -- where ``dual_run`` drops for ``max_iter: N``.

    A warmup of W steps rises linearly from ``warmup_factor * epsilon`` and
    reaches ``epsilon`` at step W + 1. The decays then run over the remaining
    ``max_iter - W`` steps, still ending at ``max_iter``, and ``step``
    boundaries stay counted from the start of the fit.
    """
    kind, warmup = schedule["kind"], schedule.get("warmup_steps", 0)
    n = max_iter - warmup
    if kind == "constant":
        main = optax.constant_schedule(epsilon)
    elif kind == "step":
        factor = schedule["factor"]
        factors = factor if isinstance(factor, list) else [factor] * len(schedule["boundaries"])
        main = optax.piecewise_constant_schedule(
            epsilon, {b - warmup: f for b, f in zip(schedule["boundaries"], factors)}
        )
    elif kind == "exponential":
        main = optax.exponential_decay(epsilon, n, schedule["final_factor"])
    elif kind == "linear":
        main = optax.linear_schedule(epsilon, schedule["final_factor"] * epsilon, n)
    else:
        main = optax.cosine_decay_schedule(epsilon, n, alpha=schedule["final_factor"])
    if warmup == 0:
        return main
    rise = optax.linear_schedule(schedule["warmup_factor"] * epsilon, epsilon, warmup)
    return optax.join_schedules([rise, main], [warmup])


def describe_schedule(opt_config: Dict) -> str:
    """The optimiser as the run log names it."""
    epsilon, max_iter = opt_config["epsilon"], opt_config["max_iter"]
    checked = validate_opt(opt_config)
    optimizer, schedule = checked["optimizer"], checked["schedule"]
    name = "adabelief" if optimizer is None else optimizer["name"]
    if optimizer is not None:
        settings = [f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in optimizer["args"].items()]
        if optimizer["clip_norm"] is not None:
            settings.append(f"gradient norm clipped at {optimizer['clip_norm']:g}")
        if settings:
            name += f" ({', '.join(settings)})"
    if schedule is None:
        line = f"{name}, step size {epsilon:g} for {max_iter} steps"
        if opt_config["dual_run"]:
            line += f", then a new {name.split(' ')[0]} at {epsilon / 10:g} for {max_iter} (dual_run)"
    else:
        kind = schedule["kind"]
        factor = schedule.get("factor")
        factor = (
            "x" + "/".join(f"{f:g}" for f in factor) if isinstance(factor, list)
            else f"x{factor:g}" if factor is not None else ""
        )
        detail = {
            "constant": "constant",
            "step": f"{factor} at steps {schedule.get('boundaries')}",
            "exponential": f"exponential decay to x{schedule.get('final_factor', 0):g}",
            "cosine": f"cosine decay to x{schedule.get('final_factor', 0):g}",
            "linear": f"linear decay to x{schedule.get('final_factor', 0):g}",
        }[kind]
        if schedule["warmup_steps"]:
            detail = (
                f"warmup from x{schedule['warmup_factor']:g} over "
                f"{schedule['warmup_steps']} steps, then {detail}"
            )
        line = f"{name}, step size {epsilon:g}, {detail}, {max_iter} steps, one compiled step"
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
    # inf <= rtol * inf is True: a window that overflowed is no baseline to converge on.
    if not (math.isfinite(before) and math.isfinite(after)):
        return False
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


def save_checkpoint(path: str, params: Dict, step: int, trace_loss: float, write: bool) -> None:
    """Write the parameters after ``step`` steps.

    ``trace_loss`` is the loss the trace records for that step, which the step
    evaluated before its own update -- the loss at the parameters after step
    ``step - 1``, not at these. It is there to line a checkpoint up with the
    trace; score a checkpoint from its parameters.

    Every rank gathers, since that can be a collective; only ``write`` (rank 0)
    writes, and every rank then meets in :func:`fail_together`, so a failed
    write ends the run on every process rather than stranding the others in
    the next step. Parameter names never start with an underscore, so the
    ``_``-keys cannot collide with them.
    """
    arrays = {name: _to_host(value) for name, value in params.items()}
    failure = None
    if write:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            np.savez(path, _step=step, _trace_loss=trace_loss, **arrays)
        except Exception as err:
            failure = err
    fail_together(failure, f"write the checkpoint {path}")


def load_checkpoint(path: str):
    """``(params, step, trace_loss)`` from a file :func:`save_checkpoint` wrote.

    ``params`` are keyed as the model names them, without ``_auto_loc``, which
    is what ``Predictive`` and ``init_predict`` take.
    """
    with np.load(path) as f:
        params = {k: f[k] for k in f.files if not k.startswith("_")}
        return params, int(f["_step"]), float(f["_trace_loss"])
