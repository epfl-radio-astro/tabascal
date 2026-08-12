"""Schema validation for the tabascal config YAML (the ``-c`` option).

The config is a deep merge of the packaged defaults
(``tabascal/data/config/tab_config_base.yaml``) and the user's file — see
:func:`tabascal.config.load_config`. Nothing about that merge is checked: a typo
merges cleanly and the run silently uses the default, and a missing key surfaces
much later as a ``KeyError`` inside a component's ``setup``, re-wrapped as
``RuntimeError("<Component> setup failed: ...")`` — after the expensive MS read
and TLE fetch.

:func:`validate_config` runs immediately after the merge and *before* anything
expensive, collecting **every** problem and raising a single :class:`ConfigError`.
That mirrors the fail-fast convention already used by
:func:`tabascal.scripts._run_tabascal_impl.assert_precision_supported` and
:func:`tabascal.truth.require_truth`.

Three layers of checking:

1. **Static schema** (:data:`SCHEMA`) — the set of known keys and their types,
   enums and ranges. Derived from the base config *plus the actual readers*, not
   from ``docs/config.md`` (which has drifted). Unknown keys are an error, with a
   ``did you mean ...?`` suggestion.
2. **Per-component requirements** — components declare ``required_config`` and
   ``config_choices`` (see :class:`tabascal.components.Component`), validated
   against the selected ``model.components``.
3. **Data-derived defaults** (:func:`resolve_gains_defaults`,
   :func:`resolve_rfi_defaults`) — resolved at component-setup time rather than
   load time, because they are computed from the MS data (the extent of the
   frequency/time axes, the observed visibility amplitude), which does not exist
   until ``TabConfig`` has read the measurement set.

Some known keys have no reader anywhere in the package but appear in every
shipped example config and in the docs (``rfi.pow_spec``, ``satellites.ric_std``,
``fisher.*``, ...). Those are marked ``inert=True``: type-checked and accepted,
reported once as a note, never an error. Erroring on them would reject every
config in ``examples/`` and ``tests/data/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

Number = (int, float)


class ConfigError(Exception):
    """Raised when the config YAML is malformed or fails validation."""


# ---------------------------------------------------------------------------
# Schema primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Field:
    """Specification for a single config key.

    Only what this config actually needs — deliberately not a general-purpose
    schema language.

    Attributes
    ----------
    types
        Accepted Python types. ``bool`` is never accepted for a numeric field
        even though ``isinstance(True, int)`` is True in Python; pass
        ``types=(bool,)`` explicitly for genuine flags.
    choices
        Permitted values. Combined with ``types`` as an *or*: a value passes if
        it is one of ``choices`` **or** an instance of ``types``. That is what
        lets ``ast.mean`` be either a number or one of ``"data"``/``"zeros"``.
        For a plain enum leave ``types`` empty, so ``choices`` is the only way
        through -- otherwise any string would satisfy ``types=(str,)`` and the
        enum would never be enforced.
    item
        Element type for list-valued keys; a tuple to accept several.
    gt, ge
        Exclusive / inclusive lower bounds for numeric keys.
    inert
        Known key that no code currently reads. Accepted and type-checked, then
        reported once as a note.
    help
        Appended to the error message for this key.
    """

    types: Tuple[type, ...] = ()
    choices: Tuple[Any, ...] = ()
    item: Optional[Union[type, Tuple[type, ...]]] = None
    gt: Optional[float] = None
    ge: Optional[float] = None
    inert: bool = False
    help: str = ""


def _type_names(types: Sequence[type]) -> str:
    if set(types) == {int, float}:
        return "a number"
    names = {
        bool: "a boolean",
        int: "an integer",
        float: "a float",
        str: "a string",
        list: "a list",
        dict: "a mapping",
    }
    return " or ".join(names.get(t, t.__name__) for t in types)


# ---------------------------------------------------------------------------
# The schema
# ---------------------------------------------------------------------------

# Shorthands for the recurring shapes.
_BOOL = Field(types=(bool,))
_STR = Field(types=(str,))
_NUM = Field(types=(int, float))
_POS_NUM = Field(types=(int, float), gt=0)
_POS_INT = Field(types=(int,), ge=1)
_NUM_LIST = Field(types=(list,), item=(float, int))  # type: ignore[arg-type]

# Valid values for the `init` / `mean` keys, unioned across all components. The
# per-component subsets are declared on the components themselves (via
# ``config_choices``) because they genuinely differ — e.g. FourierTimeFreqGPAst
# accepts ``data``/``prior``/``truth``/``sample`` and raises on anything else,
# while the older Fourier ast components silently treat anything that is not
# ``prior``/``truth`` as ``sample``.
_INIT_CHOICES = ("data", "est", "prior", "truth", "truth_mean", "sample", "zeros", "ones")
_MEAN_CHOICES = ("data", "est", "prior", "truth", "truth_mean", "zeros")

SCHEMA: Dict[str, Dict[str, Any]] = {
    "model": {
        "components": Field(
            types=(list,),
            item=str,
            help="each entry is a 'module:Class' reference, e.g. 'gains:UnitaryGains'",
        ),
        "precision": Field(choices=("single", "double")),
        "name": Field(types=(str,), help="set by the run; rarely useful in a config"),
    },
    "data": {
        "sim_dir": Field(types=(str,), help="or pass -s/--sim_dir on the command line"),
        "ms_path": Field(types=(str,), help="or pass -ms/--ms_path on the command line"),
        "zarr_path": Field(types=(str,), help="derived from sim_dir by the run"),
        "freq": Field(types=(int, float), help="null selects all frequency channels"),
        "data_col": _STR,
        "corr": Field(types=(), choices=("xx", "xy", "yx", "yy")),
        "noise": Field(types=(int, float), gt=0, help="per-visibility noise in Jy"),
        "flags": _BOOL,
    },
    "plots": {
        "init": _BOOL,
        "truth": _BOOL,
        "prior": _BOOL,
        "opt": _BOOL,
        "losses": _BOOL,
        "prior_samples": _POS_INT,
    },
    "inference": {
        "opt": _BOOL,
        "fisher": Field(types=(bool,), inert=True),
        "mcmc": Field(types=(bool,), inert=True),
    },
    "opt": {
        "epsilon": _POS_NUM,
        "max_iter": Field(types=(int,), ge=0),
        "guide": Field(types=(), choices=("map",)),
        "dual_run": _BOOL,
    },
    "fisher": {
        "max_cg_iter": Field(types=(int,), ge=1, inert=True),
        "n_samples": Field(types=(int,), ge=1, inert=True),
    },
    "ast": {
        "init": Field(types=(), choices=_INIT_CHOICES),
        "mean": Field(types=(int, float), choices=_MEAN_CHOICES),
        "freq_pad_factor": Field(types=(int, float), ge=1),
        "time_pad_factor": Field(types=(int, float), ge=1),
        "pow_spec": {
            "p0": _POS_NUM,
            "gamma": _POS_NUM,
            "gammas": _NUM_LIST,
            "k0": _POS_NUM,
            "k0_freq": _POS_NUM,
            "fov_deg": Field(
                types=(int, float),
                gt=0,
                help="null derives the field of view from the dish diameter in the MS",
            ),
            "cutoff": _POS_NUM,
        },
    },
    "rfi": {
        "init": Field(types=(), choices=_INIT_CHOICES),
        "mean": Field(types=(int, float), choices=_MEAN_CHOICES),
        "est": Field(types=(str,), help="path to a previously saved RFI estimate"),
        "var": Field(types=(int, float), gt=0, help="RFI signal variance in Jy"),
        "corr_time": Field(types=(int, float), gt=0, help="seconds"),
        "corr_freq": Field(types=(int, float), gt=0, help="Hz"),
        "freq_pad_factor": Field(types=(int, float), ge=1),
        "time_pad_factor": Field(types=(int, float), ge=1),
        "time_int_factor": _POS_NUM,
        "freq_int_samples": _POS_INT,
        "n_int_freq": _POS_INT,
        "n_int_time": _POS_INT,
        "min_time_bins": _POS_INT,
        "max_time_bins": _POS_INT,
        "r_seed": Field(types=(int,)),
        "pow_spec": {
            "p0": Field(types=(int, float), gt=0, inert=True),
            "k0s": Field(types=(list,), item=(float, int), inert=True),  # type: ignore[arg-type]
            "gammas": Field(types=(list,), item=(float, int), inert=True),  # type: ignore[arg-type]
            "cutoff": Field(types=(int, float), gt=0, inert=True),
        },
    },
    "satellites": {
        "norad_ids": Field(types=(list,), item=int),
        "spacetrack_path": _STR,
        "extra_tle_dir": Field(types=(str,), help="or pass --extra-tle-dir on the command line"),
        "norad_ids_path": Field(types=(str,), inert=True),
        "tle_dir": Field(types=(str,), inert=True),
        "tle_offset": Field(types=(int, float), inert=True),
        "sat_ids": Field(types=(list,), inert=True),
        "ole_path": Field(types=(str,), inert=True),
        "ric_std": Field(types=(int, float), gt=0, inert=True),
    },
    "gains": {
        "amp_mean": Field(types=(int, float), gt=0),
        "phase_mean": _NUM,
        "amp_std": Field(types=(int, float), ge=0, help="percent of amp_mean"),
        "phase_std": Field(types=(int, float), ge=0, help="degrees"),
        "amp_corr_freq": Field(types=(int, float), gt=0, help="Hz"),
        "amp_corr_time": Field(types=(int, float), gt=0, help="seconds"),
        "phase_corr_freq": Field(types=(int, float), gt=0, help="Hz"),
        "phase_corr_time": Field(types=(int, float), gt=0, help="seconds"),
        "r_seed": Field(types=(int,)),
        "init": Field(types=(str,), inert=True),
        "corr_time": Field(types=(int, float), inert=True),
    },
}


# ---------------------------------------------------------------------------
# Problem collection
# ---------------------------------------------------------------------------


@dataclass
class _Problems:
    """Accumulates every problem so one error can report them all."""

    errors: List[Tuple[str, str]] = field(default_factory=list)
    inert: List[str] = field(default_factory=list)

    def add(self, path: str, message: str, help: str = "") -> None:
        self.errors.append((path, f"{message} — {help}" if help else message))

    def render(self, path: Optional[str]) -> str:
        where = f" in {path}" if path else ""
        width = max(len(p) for p, _ in self.errors)
        lines = [f"invalid configuration{where}", ""]
        lines += [f"  {p.ljust(width)} : {m}" for p, m in self.errors]
        return "\n".join(lines)


def _describe(value: Any) -> str:
    return repr(value)


def _check_value(spec: Field, path: str, value: Any, problems: _Problems) -> None:
    """Type / enum / range checks for one non-null value."""
    if value in spec.choices:
        # An explicit choice always wins, whatever its type (this is how the
        # numeric 0 in `ast.mean: 0` and the string "zeros" both pass).
        return

    if spec.types:
        # isinstance(True, int) is True, so a boolean would silently satisfy any
        # numeric field. Reject it unless bool was asked for explicitly.
        is_bool = isinstance(value, bool)
        ok = isinstance(value, spec.types) and (bool in spec.types or not is_bool)
        if not ok:
            expected = _type_names(spec.types)
            if spec.choices:
                expected += " or one of " + ", ".join(repr(c) for c in spec.choices)
            problems.add(path, f"expected {expected}, got {_describe(value)}", spec.help)
            return
    elif spec.choices:
        allowed = ", ".join(repr(c) for c in spec.choices)
        problems.add(path, f"{_describe(value)} is not one of {allowed}", spec.help)
        return

    if spec.item is not None and isinstance(value, list):
        item_types = spec.item if isinstance(spec.item, tuple) else (spec.item,)
        for i, item in enumerate(value):
            ok = isinstance(item, item_types) and (
                bool in item_types or not isinstance(item, bool)
            )
            if not ok:
                problems.add(
                    f"{path}[{i}]",
                    f"expected {_type_names(item_types)}, got {_describe(item)}",
                    spec.help,
                )

    if isinstance(value, Number) and not isinstance(value, bool):
        if spec.gt is not None and not value > spec.gt:
            problems.add(
                path, f"expected a number > {spec.gt:g}, got {_describe(value)}", spec.help
            )
        if spec.ge is not None and not value >= spec.ge:
            problems.add(
                path, f"expected a number >= {spec.ge:g}, got {_describe(value)}", spec.help
            )


def _suggest(key: str, candidates: Sequence[str]) -> str:
    match = get_close_matches(key, list(candidates), n=1, cutoff=0.6)
    return f" (did you mean {match[0]!r}?)" if match else ""


def _walk(node: Any, schema: Dict[str, Any], prefix: str, problems: _Problems) -> None:
    """Recursively check ``node`` against ``schema``."""
    if not isinstance(node, dict):
        problems.add(prefix or "<top level>", f"expected a mapping, got {_describe(node)}")
        return

    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else str(key)

        if key not in schema:
            problems.add(path, f"unknown key{_suggest(str(key), schema.keys())}")
            continue

        spec = schema[key]

        if isinstance(spec, dict):
            _walk(value, spec, path, problems)
            continue

        if spec.inert:
            problems.inert.append(path)

        # `null` is how the base config spells "derive this from the data", so it
        # is always allowed here. Keys that genuinely cannot be null are caught
        # by the per-component `required_config` check below.
        if value is None:
            continue

        _check_value(spec, path, value, problems)


# ---------------------------------------------------------------------------
# Cross-field and component-driven checks
# ---------------------------------------------------------------------------


def _lookup(config: Dict, dotted: str) -> Tuple[bool, Any]:
    """Return ``(found, value)`` for a dotted config path."""
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def _check_components(config: Dict, problems: _Problems) -> None:
    """Validate the keys the *selected* components require.

    Imports the component classes named in ``model.components`` and checks each
    class's ``required_config`` (paths that must be set) and ``config_choices``
    (per-component enums). Import failures are reported here too, so a mistyped
    component reference is caught at the same time as everything else rather
    than blowing up in ``Model.__init__``.
    """
    from tabascal.imports import import_components

    found, components = _lookup(config, "model.components")
    if not found or not components:
        problems.add("model.components", "required: list at least one model component")
        return
    if not isinstance(components, list):
        return  # already reported by the schema walk

    try:
        classes = import_components(components)
    except ImportError as e:
        problems.add("model.components", str(e).replace("\n", "\n      "))
        return

    for cls in classes:
        for dotted in getattr(cls, "required_config", ()):
            found, value = _lookup(config, dotted)
            if not found or value is None:
                problems.add(dotted, f"required by {cls.__name__} but not set")

        for dotted, choices in getattr(cls, "config_choices", {}).items():
            found, value = _lookup(config, dotted)
            if found and value is not None and value not in choices:
                allowed = ", ".join(repr(c) for c in choices)
                problems.add(
                    dotted,
                    f"{_describe(value)} is not supported by {cls.__name__}; "
                    f"choose from {allowed}",
                )


def _check_cross_field(config: Dict, problems: _Problems) -> None:
    """Checks that span more than one key."""
    _, min_bins = _lookup(config, "rfi.min_time_bins")
    _, max_bins = _lookup(config, "rfi.max_time_bins")
    if isinstance(min_bins, int) and isinstance(max_bins, int) and min_bins > max_bins:
        problems.add(
            "rfi.min_time_bins",
            f"must be <= rfi.max_time_bins ({max_bins}), got {min_bins}",
        )

    _, prior = _lookup(config, "plots.prior")
    _, samples = _lookup(config, "plots.prior_samples")
    if prior and not samples:
        problems.add(
            "plots.prior_samples",
            "must be a positive integer when plots.prior is enabled",
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def validate_config(
    config: Dict, path: Optional[str] = None, report_inert: bool = False
) -> List[str]:
    """Validate a fully-merged config dictionary.

    Every problem is collected before anything is raised, so one run surfaces the
    whole list rather than making the user fix them one at a time.

    Parameters
    ----------
    config : Dict
        The config after :func:`tabascal.config.deep_update` has merged the user
        file over the packaged defaults.
    path : str, optional
        Path of the user's config file, used in the error message.
    report_inert : bool, optional
        Print the recognised-but-unread keys. Off by default: the packaged base
        config carries several of them, so every run would print the note.
        ``tabascal validate-config`` turns it on.

    Returns
    -------
    List[str]
        The dotted paths of the recognised-but-unread keys that were present.

    Raises
    ------
    ConfigError
        If any key is unknown, of the wrong type, out of range, or required by a
        selected component but unset.
    """
    problems = _Problems()

    _walk(config, SCHEMA, "", problems)
    if not problems.errors:
        # Only meaningful once the structure is known-good: these read values by
        # path and would otherwise pile confusing errors on top of real ones.
        _check_cross_field(config, problems)
        _check_components(config, problems)

    if problems.errors:
        problems.errors.sort()
        raise ConfigError(problems.render(path))

    inert = sorted(problems.inert)
    if report_inert and inert:
        print(
            "\nNote: these keys are recognised but not currently read by any "
            "component:\n  " + "\n  ".join(inert)
        )
    return inert


# ---------------------------------------------------------------------------
# Data-derived defaults
# ---------------------------------------------------------------------------
#
# These run at component-setup time, not load time: their defaults come from the
# measurement set (the extent of the frequency/time axes, the observed
# visibility amplitude), which is not available until TabConfig has read it.
# The static type checking they used to do is now handled by the schema above,
# so all that is left here is "fill in what the user left null".


def _extent(x, dx) -> float:
    """Span of the array ``x``, falling back to ``dx`` for a zero-extent axis."""
    ext = float(x.max() - x.min())
    return float(dx) if ext == 0.0 else ext


def resolve_rfi_defaults(
    rfi_config: Dict, vis_obs, freqs, chan_width: float, times, int_time: float
) -> Dict:
    """Fill in the data-derived ``rfi`` defaults, in place.

    ``var`` defaults to the largest observed visibility amplitude, and the two
    correlation scales to half the extent of the frequency/time axes.

    Note the ``is None`` tests: an explicitly configured ``0`` is honoured. The
    previous implementation used ``if not x``, which silently replaced any
    falsy value — including a deliberate zero — with the estimate.
    """
    if rfi_config.get("r_seed") is None:
        rfi_config["r_seed"] = 1

    if rfi_config.get("var") is None:
        rfi_config["var"] = float(abs(vis_obs).max())
    else:
        rfi_config["var"] = float(rfi_config["var"])

    if rfi_config.get("corr_freq") is None:
        rfi_config["corr_freq"] = _extent(freqs, chan_width) / 2
    else:
        rfi_config["corr_freq"] = float(rfi_config["corr_freq"])

    if rfi_config.get("corr_time") is None:
        rfi_config["corr_time"] = _extent(times, int_time) / 2
    else:
        rfi_config["corr_time"] = float(rfi_config["corr_time"])

    print()
    print(f"Using RFI var : {rfi_config['var']:.1e} Jy")
    print(f"Using RFI corr_freq : {rfi_config['corr_freq']/1e3:.1f} kHz")
    print(f"Using RFI corr_time : {rfi_config['corr_time']:.1f} s")

    return rfi_config


def resolve_gains_defaults(
    gains_config: Dict, freqs, chan_width: float, times, int_time: float
) -> Dict:
    """Fill in the data-derived ``gains`` defaults, in place.

    ``amp_std`` is given as a percentage of ``amp_mean`` and ``phase_std`` in
    degrees; both are converted here to the absolute/radian values the GP uses.
    The correlation scales default to the full extent of the frequency/time axes.

    As in :func:`resolve_rfi_defaults`, the ``is None`` tests mean an explicit
    ``0`` is honoured rather than replaced by the estimate.
    """
    import jax.numpy as jnp

    if gains_config.get("r_seed") is None:
        gains_config["r_seed"] = 2

    if gains_config.get("amp_mean") is None:
        gains_config["amp_mean"] = 1.0
    else:
        gains_config["amp_mean"] = float(gains_config["amp_mean"])

    if gains_config.get("amp_std") is None:
        gains_config["amp_std"] = gains_config["amp_mean"] / 100  # 1 %
    else:
        gains_config["amp_std"] = float(gains_config["amp_std"]) / 100 * gains_config["amp_mean"]

    if gains_config.get("amp_corr_freq") is None:
        gains_config["amp_corr_freq"] = _extent(freqs, chan_width)
    else:
        gains_config["amp_corr_freq"] = float(gains_config["amp_corr_freq"])

    if gains_config.get("amp_corr_time") is None:
        gains_config["amp_corr_time"] = _extent(times, int_time)
    else:
        gains_config["amp_corr_time"] = float(gains_config["amp_corr_time"])

    if gains_config.get("phase_mean") is None:
        gains_config["phase_mean"] = 0.0
    else:
        gains_config["phase_mean"] = float(gains_config["phase_mean"])

    if gains_config.get("phase_std") is None:
        gains_config["phase_std"] = float(jnp.deg2rad(1))
    else:
        gains_config["phase_std"] = float(jnp.deg2rad(gains_config["phase_std"]))

    if gains_config.get("phase_corr_freq") is None:
        gains_config["phase_corr_freq"] = _extent(freqs, chan_width)
    else:
        gains_config["phase_corr_freq"] = float(gains_config["phase_corr_freq"])

    if gains_config.get("phase_corr_time") is None:
        gains_config["phase_corr_time"] = _extent(times, int_time)
    else:
        gains_config["phase_corr_time"] = float(gains_config["phase_corr_time"])

    print()
    print(f"Using Gains amplitude mean : {gains_config['amp_mean']:.1f}")
    print(f"Using Gains amplitude std : {gains_config['amp_std']*100/gains_config['amp_mean']:.1f} %")
    print(f"Using Gains amplitude corr_freq : {gains_config['amp_corr_freq']/1e3:.1f} kHz")
    print(f"Using Gains amplitude corr_time : {gains_config['amp_corr_time']:.1f} s")
    print()
    print(f"Using Gains phase mean : {jnp.rad2deg(gains_config['phase_mean']):.1f} degrees")
    print(f"Using Gains phase std : {jnp.rad2deg(gains_config['phase_std']):.1f} degrees")
    print(f"Using Gains phase corr_freq : {gains_config['phase_corr_freq']/1e3:.1f} kHz")
    print(f"Using Gains phase corr_time : {gains_config['phase_corr_time']:.1f} s")

    return gains_config
