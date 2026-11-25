"""Pydantic v2 configuration schemas for TABASCAL.

Every top-level config section (``model``, ``data``, ``plots``, ``inference``,
``opt``, ``satellites``) and every :class:`~tabascal.components.Component`
subclass declares its own strict Pydantic model with field descriptions and
range/type checks. Common fields shared by related components are expressed via
Python mixin inheritance here — never as shared global YAML sections on the
user side.

This module deliberately avoids importing JAX, NumPyro, Optax, or Dask at
module import time so lightweight CLI commands (``tabascal -h``, ``tab2MS``,
``tabascal rfi-per-sat``) pay no accelerator import cost.
"""

from __future__ import annotations

import difflib
import math
import os
import sys
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Annotated, Any, ClassVar, FrozenSet, Literal, Optional, Tuple, Type, TypeVar, Union

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


# ---------------------------------------------------------------------------
# Public error type & formatting
# ---------------------------------------------------------------------------

#: Top-level sections from the pre-component sectioned layout that moved onto
#: individual components under ``model.components``.
_OLD_LAYOUT_KEYS = ("ast", "rfi", "gains")


class ConfigValidationError(ValueError):
    """Raised when a TABASCAL configuration fails schema validation."""

    def __init__(self, message: str, *, cause: Optional[ValidationError] = None):
        super().__init__(message)
        self.validation_error = cause


class NullForDefaultError(ValueError):
    """``null`` written on an option that has no "none": the message is about ``key``."""

    def __init__(self, key: str, message: str):
        super().__init__(message)
        self.key = key


def _extra_key_hint(
    key: str,
    model_cls: Optional[type],
    *,
    component_cls: Optional[type] = None,
) -> str:
    """Suggest a typo fix or the component/section that actually owns ``key``."""
    if not isinstance(key, str) or model_cls is None:
        return ""
    known = sorted(getattr(model_cls, "model_fields", ()))
    close = difflib.get_close_matches(key, known, n=1)
    if close:
        return f" Did you mean {close[0]!r}?"

    if component_cls is not None:
        # Check whether the misplaced option belongs to a top-level section
        # (e.g. norad_ids belongs in 'satellites') or to another component.
        section_owners = [
            sec_name
            for sec_name, sec_cls in TOP_LEVEL_SECTIONS.items()
            if key in sec_cls.model_fields
        ]
        if section_owners:
            return (
                f" {key!r} belongs in the top-level {', '.join(repr(s) for s in section_owners)} "
                "section, not on this component."
            )
        from tabascal.components import in_tree_components

        owners = sorted(
            ref
            for ref, other in in_tree_components().items()
            if other is not component_cls and key in getattr(other, "Config", EmptyComponentConfig).model_fields
        )
        if owners:
            return (
                f" {key!r} is an option of {', '.join(owners)}, not of this "
                "component; set it there."
            )
    return ""


def _resolve_model_for_loc(root_cls: Optional[type], loc: tuple) -> Optional[type]:
    """Walk ``root_cls`` along ``loc[:-1]`` to find the model that rejected ``loc[-1]``."""
    if root_cls is None or len(loc) <= 1:
        return root_cls
    # Direct section lookup on RunConfig.
    if root_cls is RunConfig and isinstance(loc[0], str):
        sec_map = {"model": ModelConfig, **TOP_LEVEL_SECTIONS}
        if len(loc) == 2 and loc[0] in sec_map:
            return sec_map[loc[0]]
    return None


def format_validation_error(
    err: ValidationError,
    *,
    prefix: str = "",
    model_cls: Optional[type] = None,
    component_cls: Optional[type] = None,
) -> ConfigValidationError:
    """Turn a Pydantic :class:`ValidationError` into a readable :class:`ConfigValidationError`."""
    lines = []
    for item in err.errors():
        raw_loc = tuple(item.get("loc", ()))
        cause = (item.get("ctx") or {}).get("error")
        if isinstance(cause, NullForDefaultError):
            # The message names the key for a bare pydantic error; here it is
            # moved onto the location instead.
            raw_loc = raw_loc + (cause.key,)
            item = {**item, "msg": str(cause)[len(f"{cause.key}: "):]}
        loc_parts = []
        for part in raw_loc:
            if isinstance(part, int):
                if loc_parts:
                    loc_parts[-1] = f"{loc_parts[-1]}[{part}]"
                else:
                    loc_parts.append(f"[{part}]")
            else:
                loc_parts.append(str(part))
        loc = ".".join(loc_parts)
        if prefix:
            loc = f"{prefix}.{loc}" if loc else prefix
        msg = item.get("msg", "Invalid value")
        # Pydantic prefixes custom ValueError messages with "Value error, ".
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, "):]
        if item.get("type") == "literal_error" and "input" in item:
            msg = f"{msg} (got {item['input']!r})"
        elif item.get("type") == "extra_forbidden" and raw_loc:
            bad_key = raw_loc[-1]
            target_cls = (
                model_cls
                if component_cls is not None
                else _resolve_model_for_loc(model_cls, raw_loc)
            )
            hint = _extra_key_hint(
                str(bad_key),
                target_cls,
                component_cls=component_cls if len(raw_loc) == 1 else None,
            )
            if target_cls is not None:
                takes = ", ".join(target_cls.model_fields) or "no options"
                msg = f"{msg} (takes: {takes}).{hint}"
            elif hint:
                msg = f"{msg}.{hint}"
        # Errors accumulated inside a field validator (e.g. model.components)
        # already format one bullet per offending component.
        if msg.lstrip().startswith("- "):
            for subline in msg.splitlines():
                lines.append(f"  {subline.strip()}")
        else:
            lines.append(f"  - {loc}: {msg}" if loc else f"  - {msg}")
    header = "Configuration validation failed:\n" if len(lines) > 1 else "Configuration validation failed: "
    body = "\n".join(lines) if len(lines) > 1 else lines[0].lstrip("  - ") if lines else str(err)
    return ConfigValidationError(header + body, cause=err)


# ---------------------------------------------------------------------------
# Strict primitive validators (reject bools/strings where numbers are expected)
# ---------------------------------------------------------------------------

def _check_bool(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"must be a boolean (true or false), got {value!r}")
    return value


def _check_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"must be an integer, got {value!r}")
    return int(value)


def _check_positive_int(value: Any) -> int:
    out = _check_int(value)
    if out < 1:
        raise ValueError(f"must be an integer >= 1, got {value!r}")
    return out


def _check_non_negative_int(value: Any) -> int:
    out = _check_int(value)
    if out < 0:
        raise ValueError(f"must be an integer >= 0, got {value!r}")
    return out


def _check_whole_positive_int(value: Any) -> int:
    """Accept an integer >= 1 or a finite whole float (e.g. 128.0); reject bool/fractions."""
    if isinstance(value, bool):
        raise ValueError(f"must be a whole number >= 1, got {value!r}")
    if isinstance(value, Integral):
        out = int(value)
    elif isinstance(value, Real):
        try:
            f = float(value)
        except OverflowError:
            raise ValueError(f"is too large to be a whole number: {value!r}") from None
        if not math.isfinite(f) or f != int(f):
            raise ValueError(f"must be a whole number >= 1, got {value!r}")
        out = int(f)
    else:
        raise ValueError(f"must be a whole number >= 1, got {value!r}")
    if out < 1:
        raise ValueError(f"must be a whole number >= 1, got {value!r}")
    return out


def _check_finite_float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"must be a finite number, got {value!r}")
    try:
        out = float(value)
    except OverflowError:
        raise ValueError(f"is too large to be a floating-point number: {value!r}") from None
    if not math.isfinite(out):
        raise ValueError(f"must be a finite number, got {value!r}")
    return out


def _check_positive_float(value: Any) -> float:
    out = _check_finite_float(value)
    if out <= 0.0:
        raise ValueError(f"must be finite and greater than zero, got {value!r}")
    return out


def _check_non_negative_float(value: Any) -> float:
    out = _check_finite_float(value)
    if out < 0.0:
        raise ValueError(f"must be finite and >= 0, got {value!r}")
    return out


def _check_float_ge_1(value: Any) -> float:
    out = _check_finite_float(value)
    if out < 1.0:
        raise ValueError(f"must be a finite number >= 1, got {value!r}")
    return out


def _check_cutoff_float(value: Any) -> float:
    out = _check_positive_float(value)
    if out >= 1.0:
        raise ValueError(
            f"must be strictly positive and below 1 (at 1 or above every Fourier mode is cut), got {value!r}"
        )
    return out


def _is_plain_ordered_sequence(value: Any) -> bool:
    """True for list, tuple, plain np.ndarray (not subclasses), or jax.Array."""
    if isinstance(value, (list, tuple)):
        return True
    np_mod = sys.modules.get("numpy")
    if np_mod is not None and type(value) is np_mod.ndarray:
        return True
    jax_mod = sys.modules.get("jax")
    if jax_mod is not None and isinstance(value, jax_mod.Array):
        return True
    return False


def _check_positive_pair(value: Any) -> Tuple[float, float]:
    """An ordered pair of positive numbers, one per axis, each as a float.

    Floats, as the pre-schema ``gp_cov`` validator returned them: the pair is
    used as an exponent, and jax routes an integer exponent through
    ``lax.integer_pow`` and a float one through ``lax.pow``, which differ in
    the last bit. A config that writes ``gammas: [5, 5]`` therefore runs exactly
    the model it ran before. The integer defaults some components keep are
    their own, and never pass through here (see ``BaseGPRFI.default_gammas``).
    """
    if not _is_plain_ordered_sequence(value):
        raise ValueError(
            f"must be an ordered pair of two positive numbers [freq, time], got {value!r}"
        )
    if hasattr(value, "ndim"):
        if value.ndim != 1:
            raise ValueError(
                f"must be a 1-D ordered pair of two positive numbers [freq, time], got {value!r}"
            )
        items = value.tolist()
    else:
        items = list(value)
    if len(items) != 2:
        raise ValueError(
            f"must have exactly 2 entries [freq, time], got {len(items)}: {value!r}"
        )
    return (_check_positive_float(items[0]), _check_positive_float(items[1]))


def _check_path_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str):
        raise ValueError(f"must be a path string or null, got {value!r}")
    return value


def _check_non_empty_str(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"must be a non-empty string, got {value!r}")
    return value


def _as_norad_id(value: Any, where: str) -> int:
    """Coerce one entry to a positive integral NORAD catalogue ID."""
    if isinstance(value, bool):
        raise ValueError(f"{where}: {value!r} is not a NORAD catalogue ID")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{where}: empty NORAD catalogue ID")
        try:
            value = int(text)
        except ValueError as e:
            raise ValueError(f"{where}: {value!r} is not a NORAD catalogue ID") from e
    if isinstance(value, Integral):
        out = int(value)
    elif isinstance(value, Real):
        as_float = float(value)
        if not math.isfinite(as_float) or as_float != round(as_float):
            raise ValueError(
                f"{where}: {value!r} is not a finite integer NORAD catalogue ID"
            )
        out = int(round(as_float))
    else:
        raise ValueError(f"{where}: {value!r} is not a NORAD catalogue ID")
    if out <= 0:
        raise ValueError(
            f"{where}: NORAD catalogue IDs must be positive, got {out}"
        )
    return out


StrictBool = Annotated[bool, BeforeValidator(_check_bool)]
StrictInt = Annotated[int, BeforeValidator(_check_int)]
PositiveInt = Annotated[int, BeforeValidator(_check_positive_int)]
NonNegativeInt = Annotated[int, BeforeValidator(_check_non_negative_int)]
WholePositiveInt = Annotated[int, BeforeValidator(_check_whole_positive_int)]
FiniteFloat = Annotated[float, BeforeValidator(_check_finite_float)]
PositiveFloat = Annotated[float, BeforeValidator(_check_positive_float)]
NonNegativeFloat = Annotated[float, BeforeValidator(_check_non_negative_float)]
FloatGe1 = Annotated[float, BeforeValidator(_check_float_ge_1)]
CutoffFloat = Annotated[float, BeforeValidator(_check_cutoff_float)]
PositivePair = Annotated[Tuple[float, float], BeforeValidator(_check_positive_pair)]
OptionalPathStr = Annotated[Optional[str], BeforeValidator(_check_path_str)]
PathStr = Annotated[str, BeforeValidator(_check_path_str)]
NonEmptyStr = Annotated[str, BeforeValidator(_check_non_empty_str)]


# ---------------------------------------------------------------------------
# Serialization helper
# ---------------------------------------------------------------------------

def _to_plain_yaml(obj: Any) -> Any:
    """Convert Pydantic/Python structures into YAML-safe plain types."""
    if isinstance(obj, BaseModel):
        return {k: _to_plain_yaml(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    if isinstance(obj, tuple):
        return [_to_plain_yaml(v) for v in obj]
    if isinstance(obj, list):
        return [_to_plain_yaml(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_plain_yaml(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# Strict base model & code-level shared component mixins
# ---------------------------------------------------------------------------

def field_allows_none(field_info: Any) -> bool:
    """Whether a field's type admits ``None``, i.e. whether ``null`` means something for it."""
    annotation = field_info.annotation
    if annotation is Any or annotation is type(None):
        return True
    return _type_admits_none(annotation)


def _type_admits_none(annotation: Any) -> bool:
    from typing import get_args

    return any(
        arg is type(None) or (get_args(arg) and _type_admits_none(arg))
        for arg in get_args(annotation)
    )


class StrictConfigModel(BaseModel):
    """Base schema for every TABASCAL configuration section and component.

    ``null`` means "none" or "off", and only on the options where that is a
    setting -- ``min_elevation: null`` is no elevation mask. It never means "the
    default": a default is had by leaving the key out, and an option whose
    default is computed names the computation (``corr_freq_hz: half_band``).
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    #: Field name -> path in the pre-component sectioned config, used by
    #: ``tabascal migrate-config`` to move options onto the owning component.
    legacy_keys: ClassVar[dict[str, tuple[str, ...]]] = {}

    #: Fields for which a legacy ``null`` meant "this component's default" even
    #: though ``null`` is a setting of the field now, so ``tabascal
    #: migrate-config`` omits them. A legacy ``null`` on a field that does not
    #: take one is omitted without being listed here.
    legacy_null_is_default: ClassVar[FrozenSet[str]] = frozenset()

    #: Field name -> what a legacy ``null`` meant, for the fields whose ``null``
    #: was a computed default and is now spelled by name.
    legacy_null_value: ClassVar[dict[str, Any]] = {}

    #: Fields whose type does not admit ``None`` but which read ``null`` as
    #: "none" all the same (``norad_ids: null`` is no satellites).
    null_is_none: ClassVar[FrozenSet[str]] = frozenset()

    @model_validator(mode="before")
    @classmethod
    def _reject_null_as_default(cls, data: Any) -> Any:
        # A model validator rather than a field one, so it runs before every
        # field's own validators and the message is this one whatever they say.
        if not isinstance(data, Mapping):
            return data
        for key, value in data.items():
            if value is not None or key in cls.null_is_none:
                continue
            field_info = cls.model_fields.get(key) if isinstance(key, str) else None
            if field_info is None or field_info.is_required() or field_allows_none(field_info):
                continue
            default = field_info.get_default(call_default_factory=True)
            raise NullForDefaultError(
                key,
                f"{key}: null is not a setting of this option. Leave the key out to "
                f"use its default ({_to_plain_yaml(default)!r}).",
            )
        return data

    @classmethod
    def input_path_fields(cls) -> dict[str, dict[str, Any]]:
        """Fields tagged as input file/directory paths in their Field metadata."""
        out: dict[str, dict[str, Any]] = {}
        for name, field_info in cls.model_fields.items():
            extra = field_info.json_schema_extra
            if isinstance(extra, dict) and extra.get("input_path"):
                out[name] = dict(extra)
        return out

    def requires_truth(self, ref: str = "") -> dict[str, str]:
        """Simulation truth quantities required by this config block (quantity -> reason)."""
        return {}

    def to_yaml_dict(self) -> dict[str, Any]:
        """Plain YAML-safe dictionary representation of this model."""
        return {name: _to_plain_yaml(getattr(self, name)) for name in type(self).model_fields}


class EmptyComponentConfig(StrictConfigModel):
    """Schema for components that accept no configuration parameters."""


#: Provisional defaults mirrored from ``tabascal.orbit_config`` (kept here so
#: importing ``schema`` does not pull in casacore/dask/skyfield).
DEFAULT_REMOTE_MAX_AGE_DAYS = 3.0
DEFAULT_CACHE_REUSE_MAX_AGE_DAYS = 1.0


class TrajectoryConfig(StrictConfigModel):
    """Code-level mixin for satellite trajectory components that propagate configured orbits."""

    min_elevation: Optional[FiniteFloat] = Field(
        default=0.0,
        description=(
            "Elevation in degrees below which a satellite's RFI signal is masked to zero. "
            "0 masks a satellite only while it is below the geometric horizon; null "
            "disables masking. A satellite never above it stops the run."
        ),
    )

    legacy_keys: ClassVar[dict[str, tuple[str, ...]]] = {
        "min_elevation": ("rfi", "min_elevation"),
    }


class FreqIntegrationMixin(StrictConfigModel):
    """Code-level mixin for RFI visibility components that integrate over frequency samples."""

    n_int_freq: PositiveInt = Field(
        default=1,
        description=(
            "Integration samples per frequency channel of the RFI visibility. Given "
            "directly: there is no estimate on this axis."
        ),
    )

    legacy_keys: ClassVar[dict[str, tuple[str, ...]]] = {
        "n_int_freq": ("rfi", "n_int_freq"),
    }


class FineTimeIntegrationMixin(FreqIntegrationMixin):
    """Code-level mixin for fine-grid RFI visibility components."""

    time_int_factor: PositiveFloat = Field(
        default=1.0,
        description=(
            "Factor on the number of integration samples per time step of the fine grid "
            "over its estimate from the observation's RFI sampling requirements. The "
            "count itself is always estimated and cannot be given directly."
        ),
    )

    legacy_keys: ClassVar[dict[str, tuple[str, ...]]] = {
        **FreqIntegrationMixin.legacy_keys,
        "time_int_factor": ("rfi", "time_int_factor"),
    }


class VariableFineTimeIntegrationMixin(FineTimeIntegrationMixin):
    """Code-level mixin for variable-stride fine-grid RFI visibility components."""

    min_time_bins: PositiveInt = Field(
        default=1,
        description=(
            "Minimum number of stride groups the per-baseline RFI sampling rates are "
            "binned into."
        ),
    )
    max_time_bins: PositiveInt = Field(
        default=30,
        description=(
            "Maximum number of stride groups the per-baseline RFI sampling rates are "
            "binned into."
        ),
    )

    legacy_keys: ClassVar[dict[str, tuple[str, ...]]] = {
        **FineTimeIntegrationMixin.legacy_keys,
        "min_time_bins": ("rfi", "min_time_bins"),
        "max_time_bins": ("rfi", "max_time_bins"),
    }

    @model_validator(mode="after")
    def _check_time_bins(self) -> "VariableFineTimeIntegrationMixin":
        if self.min_time_bins > self.max_time_bins:
            raise ValueError(
                f"min_time_bins ({self.min_time_bins}) must not exceed "
                f"max_time_bins ({self.max_time_bins})"
            )
        return self


def _check_std_value(value: Any, *, named: Tuple[str, ...] = ("data",)) -> Any:
    if isinstance(value, str):
        if value in named:
            return value
        options = " or ".join(repr(n) for n in named)
        raise ValueError(f"std must be a positive number or {options}, got {value!r}")
    return _check_positive_float(value)


class FourierGPConfig(StrictConfigModel):
    """Code-level mixin shared by Fourier-domain Gaussian process components."""

    std: Optional[Union[PositiveFloat, Literal["data"]]] = Field(
        default=None,
        description="Prior standard deviation in Jy, or 'data' to estimate from the observed visibilities.",
    )
    corr_freq: Optional[PositiveFloat] = Field(
        default=None,
        description="Correlation bandwidth in Hz along the frequency axis.",
    )
    gammas: PositivePair = Field(
        default=(5.0, 5.0),
        description="Power-law roll-off exponents along the (frequency, time) axes.",
    )
    cutoff: CutoffFloat = Field(
        default=1e-6,
        description="Relative power threshold below which Fourier k-modes are dropped.",
    )
    freq_pad_factor: FloatGe1 = Field(
        default=2.0,
        description="Zero-padding factor along the frequency axis of the Fourier domain (>= 1).",
    )
    time_pad_factor: FloatGe1 = Field(
        default=2.0,
        description="Zero-padding factor along the time axis of the Fourier domain (>= 1).",
    )


#: ``rfi_signal`` ``std``: twice the largest observed visibility (GitHub #227).
MAX_VIS = "max_vis"
#: ``rfi_signal`` ``corr_freq``: half the observed band.
HALF_BAND = "half_band"
#: ``rfi_signal`` ``corr_time``: half the observation.
HALF_DURATION = "half_duration"


class RFIGPConfig(FourierGPConfig):
    """Code-level mixin shared by all GP-based RFI signal components."""

    std: Union[PositiveFloat, Literal["data", "max_vis"]] = Field(
        default=MAX_VIS,
        description=(
            "Width of the prior on one RFI source, in Jy (rms|V|). 'data' measures the "
            "total rms|V| of the observed visibilities (sky and noise included); "
            "'max_vis' is twice the largest observed visibility."
        ),
    )
    corr_freq: Union[PositiveFloat, Literal["half_band"]] = Field(
        default=1e6,
        description=(
            "Bandwidth, in Hz, over which the RFI signal stays correlated (knee along "
            "the frequency axis). 'half_band' uses half the observed band."
        ),
    )
    corr_time: Union[PositiveFloat, Literal["half_duration"]] = Field(
        default=24.0,
        description=(
            "Time, in seconds, over which the RFI signal stays correlated (knee along "
            "the time axis). 'half_duration' uses half the observation."
        ),
    )
    gammas: Optional[PositivePair] = Field(
        default=None,
        description=(
            "Roll-off exponents of the RFI power spectrum on the (frequency, time) axes. "
            "null is the component's own: [3, 3] for ComplexRFIVarAnt and "
            "ComplexRFIVarAntFine, [100, 100] for ComplexRFIConstAnt and "
            "ComplexRFIConstAntFine."
        ),
    )
    cutoff: CutoffFloat = Field(
        default=1e-9,
        description=(
            "Relative power below which a k-mode is dropped from the RFI latent grid; "
            "the retained spectrum is re-normalised after the cut."
        ),
    )
    signal_type: Literal["complex", "real"] = Field(
        default="complex",
        description=(
            "Signal space of the RFI amplitude: 'complex' (default) or 'real' (one "
            "Hartley coefficient per retained mode)."
        ),
    )
    init: Literal["prior", "est", "matched-filter", "truth", "zeros", "ones", "sample"] = Field(
        default="sample",
        description=(
            "Where the RFI signal parameters start: 'prior', 'est' (light curves in "
            "est), 'matched-filter' (alias 'mf'), 'truth', 'zeros' (or 0), 'ones' (or 1), "
            "or 'sample'."
        ),
    )
    mean: Literal["zeros", "data", "est", "matched-filter"] = Field(
        default="zeros",
        description=(
            "Mean of the prior over the RFI signal parameters: 'zeros' (or 0), 'data', "
            "'est', or 'matched-filter' (alias 'mf')."
        ),
    )
    est: OptionalPathStr = Field(
        default=None,
        description=(
            "Path to a measured light-curve file (.npz or .zarr), read by init or mean "
            "'est'. Not needed for 'matched-filter'."
        ),
        json_schema_extra={"input_path": True},
    )

    legacy_keys: ClassVar[dict[str, tuple[str, ...]]] = {
        "init": ("rfi", "init"),
        "mean": ("rfi", "mean"),
        "est": ("rfi", "est"),
        "std": ("rfi", "gp_cov", "std"),
        "corr_freq": ("rfi", "gp_cov", "corr_freq"),
        "corr_time": ("rfi", "gp_cov", "corr_time"),
        "gammas": ("rfi", "gp_cov", "gammas"),
        "cutoff": ("rfi", "cutoff"),
        "signal_type": ("rfi", "signal_type"),
        "freq_pad_factor": ("rfi", "freq_pad_factor"),
        "time_pad_factor": ("rfi", "time_pad_factor"),
    }
    legacy_null_is_default: ClassVar[FrozenSet[str]] = frozenset({"gammas"})
    legacy_null_value: ClassVar[dict[str, Any]] = {
        "std": MAX_VIS,
        "corr_freq": HALF_BAND,
        "corr_time": HALF_DURATION,
    }

    @field_validator("std", mode="before")
    @classmethod
    def _validate_std(cls, value: Any) -> Any:
        return _check_std_value(value, named=("data", MAX_VIS))

    @field_validator("init", mode="before")
    @classmethod
    def _normalise_init(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError(f"init must be one of prior/est/matched-filter/truth/zeros/ones/sample, got {value!r}")
        if value == 0:
            return "zeros"
        if value == 1:
            return "ones"
        if value == "mf":
            return "matched-filter"
        return value

    @field_validator("mean", mode="before")
    @classmethod
    def _normalise_mean(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError(f"mean must be one of zeros/data/est/matched-filter (or 0), got {value!r}")
        if value == 0:
            return "zeros"
        if value == "mf":
            return "matched-filter"
        return value

    def requires_truth(self, ref: str = "") -> dict[str, str]:
        if self.init == "truth":
            label = f"{ref}.init: truth" if ref else "rfi_signal.init: truth"
            return {"rfi_A": label}
        return {}


# ---------------------------------------------------------------------------
# Top-level section schemas
# ---------------------------------------------------------------------------

class DataConfig(StrictConfigModel):
    """Observation, I/O, noise, flags, and external calibration settings."""

    out_dir: OptionalPathStr = Field(
        default=None,
        description=(
            "Directory where runs/<name>_<timestamp>_<token>/ is created. Defaults to "
            "the directory containing ms_path; given without ms_path, treated as a "
            "tab-sim simulation directory."
        ),
    )
    truth_zarr: OptionalPathStr = Field(
        default=None,
        description=(
            "Path to a tab-sim simulation truth zarr, required by init: truth and "
            "plots.truth. Defaults to <MS directory>/<MS name>.zarr."
        ),
        json_schema_extra={"input_path": True, "resolved_by_runner": True},
    )
    ms_path: OptionalPathStr = Field(
        default=None,
        description="Path to the Measurement Set to model (overridden by -ms/--ms_path).",
        json_schema_extra={
            "input_path": True,
            "resolved_by_runner": True,
            "fingerprint_kind": "ms",
        },
    )
    freq: Optional[PositiveFloat] = Field(
        default=None,
        description=(
            "Single frequency in Hz to select (resolved to the nearest channel inside "
            "the band). null reads every channel."
        ),
    )
    skip_ms_write: StrictBool = Field(
        default=False,
        description=(
            "Keep results in zarr without writing back to the MS; tab2MS can export "
            "them later."
        ),
    )
    row_chunk: Optional[PositiveInt] = Field(
        default=None,
        description="Positive row chunk size for MS reads and writes; null keeps dask-ms defaults.",
    )
    data_col: NonEmptyStr = Field(
        default="DATA",
        description="MS data column to model.",
    )
    corr: NonEmptyStr = Field(
        default="xx",
        description="Correlation to model, matched by identity against POLARIZATION::CORR_TYPE.",
    )
    noise: Optional[Union[PositiveFloat, str]] = Field(
        default=None,
        description=(
            "Per-visibility noise override (positive scalar in Jy or path to an .npz), "
            "in the frame of data_col before gain_table is applied. null reads SIGMA_SPECTRUM/SIGMA."
        ),
        json_schema_extra={"input_path": True},
    )
    gain_table: Optional[Union[str, list[str]]] = Field(
        default=None,
        description=(
            "Path to a CASA calibration table, or an ordered list of them, divided out "
            "of data and noise at read time."
        ),
        json_schema_extra={"input_path": True, "is_path_list": True},
    )
    flags: StrictBool = Field(
        default=False,
        description=(
            "Exclude the MS's own flags from the likelihood. Default false so flagged "
            "RFI emitters can be recovered."
        ),
    )
    save_rfi_per_sat: StrictBool = Field(
        default=False,
        description=(
            "Also store per-satellite RFI visibilities (rfi_vis_src) in the results "
            "zarr for per-NORAD diagnostics."
        ),
    )

    @field_validator("noise", mode="before")
    @classmethod
    def _validate_noise(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, os.PathLike):
            value = os.fspath(value)
        if isinstance(value, str):
            if not value:
                raise ValueError("data.noise path must not be empty; use null to read from the MS")
            return value
        if isinstance(value, bool):
            raise ValueError(
                f"data.noise = {value!r} is neither a number nor a path to an .npz of noise values."
            )
        return _check_positive_float(value)

    @field_validator("gain_table", mode="before")
    @classmethod
    def _validate_gain_table(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, os.PathLike):
            value = os.fspath(value)
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("data.gain_table path must not be empty")
            return value
        if isinstance(value, (list, tuple)):
            out = []
            for i, item in enumerate(value):
                if isinstance(item, os.PathLike):
                    item = os.fspath(item)
                if not isinstance(item, str) or not item.strip():
                    raise ValueError(
                        f"data.gain_table[{i}] must be a non-empty path string, got {item!r}"
                    )
                out.append(item)
            return out
        raise ValueError(
            f"data.gain_table must be a path string, a list of paths, or null, got {value!r}"
        )


class PlotsConfig(StrictConfigModel):
    """Diagnostic plotting toggles."""

    init: StrictBool = Field(
        default=False,
        description="Plot astronomical/RFI visibilities and gains at the initial parameters.",
    )
    truth: StrictBool = Field(
        default=False,
        description="Overlay simulation truth on diagnostic plots (requires a tab-sim zarr).",
    )
    prior: StrictBool = Field(
        default=False,
        description="Plot prior predictive distributions over visibilities and gains.",
    )
    opt: StrictBool = Field(
        default=False,
        description="Plot astronomical/RFI visibilities and gains at the optimised MAP parameters.",
    )
    losses: StrictBool = Field(
        default=False,
        description="Plot the optimiser trace (loss, plus chi^2 and NRMSE when trace_path is set).",
    )
    prior_samples: PositiveInt = Field(
        default=100,
        description="Number of prior samples drawn when plots.prior is true.",
    )


class InferenceConfig(StrictConfigModel):
    """Top-level inference controls."""

    opt: StrictBool = Field(
        default=True,
        description="Whether to run MAP optimisation.",
    )
    seed: StrictInt = Field(
        default=1,
        description=(
            "Single random seed for the run; 0 is a valid seed. Each sampling "
            "component folds its own fixed tag into this seed."
        ),
    )

    @property
    def effective_seed(self) -> int:
        return self.seed


class OptConfig(StrictConfigModel):
    """MAP optimisation parameters."""

    epsilon: PositiveFloat = Field(
        default=1e-2,
        description="Base optimiser step size.",
    )
    max_iter: NonNegativeInt = Field(
        default=500,
        description="Maximum optimisation iterations (per phase under dual_run, or total under schedule).",
    )
    guide: Literal["map"] = Field(
        default="map",
        description="Optimisation guide type (currently only 'map').",
    )
    dual_run: StrictBool = Field(
        default=True,
        description="Run a second optimisation phase at epsilon / 10 (ignored when schedule is set).",
    )
    optimizer: Optional[Union[str, dict[str, Any]]] = Field(
        default=None,
        description=(
            "Optax optimiser name or mapping of optimiser arguments plus optional "
            "clip_norm. null uses adabelief with optax defaults."
        ),
    )
    schedule: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Single-optimiser step-size schedule (step, exponential, cosine, linear, "
            "constant) with optional warmup_steps and warmup_factor. null keeps dual_run."
        ),
    )
    stop: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Early-stopping plateau criterion {window, rtol, min_steps}; requires schedule."
        ),
    )
    checkpoints: Optional[list[PositiveInt]] = Field(
        default=None,
        description="Iteration counts at which to write parameter checkpoints under results/checkpoints/.",
    )
    init_jitter: Optional[Union[NonNegativeFloat, dict[str, Any]]] = Field(
        default=None,
        description=(
            "Seeded perturbation added to the initial parameters: a scalar in prior "
            "widths or a per-block mapping of absolute/relative scales."
        ),
    )
    trace_path: OptionalPathStr = Field(
        default=None,
        description=(
            "Path for the per-iteration metrics trace (.npz with chi^2 and truth NRMSEs); "
            "relative paths resolve inside the run directory. Overridden by TAB_LOSS_TRACE."
        ),
    )

    @field_validator("init_jitter", mode="before")
    @classmethod
    def _validate_init_jitter_field(cls, value: Any) -> Any:
        if value is None:
            return None
        # Pure-Python check (no JAX required) mirroring tabascal.jitter.validate_init_jitter.
        if not isinstance(value, dict):
            return _check_non_negative_float(value)
        out: dict[str, Any] = {}
        for block, spec in value.items():
            if isinstance(spec, dict):
                if len(spec) != 1 or next(iter(spec)) not in ("absolute", "relative"):
                    raise ValueError(
                        f"opt.init_jitter.{block} must be a number or one of "
                        f"{{relative: r}} / {{absolute: a}}, not {spec!r}."
                    )
                ((kind, scale),) = spec.items()
                out[str(block)] = {kind: _check_non_negative_float(scale)}
            else:
                out[str(block)] = _check_non_negative_float(spec)
        return out


class SatellitesConfig(StrictConfigModel):
    """Satellite catalogue and orbit-record resolution settings."""

    null_is_none: ClassVar[FrozenSet[str]] = frozenset({"norad_ids"})

    norad_ids: list[PositiveInt] = Field(
        default_factory=list,
        description=(
            "NORAD catalogue IDs of satellites to model. Every listed ID must resolve "
            "to an acceptable orbit record. Empty is valid only when no satellite "
            "trajectory component is in model.components."
        ),
    )
    norad_ids_path: OptionalPathStr = Field(
        default=None,
        description=(
            "Path to a text file of NORAD IDs (one per line, '#' comments ignored). "
            "Takes precedence over norad_ids; CLI -np/--norad-path takes precedence over both."
        ),
        json_schema_extra={"input_path": True},
    )
    extra_orbit_dir: OptionalPathStr = Field(
        default=None,
        description=(
            "Directory of local orbit files (TLE or OMM) searched per NORAD ID before "
            "the managed cache and SatChecker."
        ),
        json_schema_extra={"input_path": True, "fingerprint_kind": "orbits"},
    )
    extra_orbit_max_age_days: Optional[NonNegativeFloat] = Field(
        default=None,
        description=(
            "Max |record epoch - observation epoch| in days for extra_orbit_dir files. "
            "null is unlimited (keeps exact used_orbits replay working)."
        ),
    )
    remote_max_age_days: Optional[NonNegativeFloat] = Field(
        default=DEFAULT_REMOTE_MAX_AGE_DAYS,
        description=(
            "Hard ceiling in days on the age of a record accepted from SatChecker or "
            "the managed cache. null removes the ceiling."
        ),
    )
    cache_reuse_max_age_days: Optional[NonNegativeFloat] = Field(
        default=DEFAULT_CACHE_REUSE_MAX_AGE_DAYS,
        description=(
            "Age in days below which a cached SatChecker record avoids a new remote "
            "request. null always reuses the nearest acceptable cached record."
        ),
    )

    @field_validator("norad_ids", mode="before")
    @classmethod
    def _normalise_norad_ids(cls, value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, (str, bytes)):
            raise ValueError(f"norad_ids must be a list of NORAD catalogue IDs, got {value!r}")
        if hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, Sequence):
            try:
                value = list(value)
            except TypeError as e:
                raise ValueError(
                    f"norad_ids must be a list of NORAD catalogue IDs, got {value!r}"
                ) from e
        out: list[int] = []
        seen: set[int] = set()
        for i, item in enumerate(value):
            nid = _as_norad_id(item, f"norad_ids[{i}]")
            if nid not in seen:
                seen.add(nid)
                out.append(nid)
        return out

    @model_validator(mode="after")
    def _check_age_bounds(self) -> "SatellitesConfig":
        if (
            self.cache_reuse_max_age_days is not None
            and self.remote_max_age_days is not None
            and self.cache_reuse_max_age_days > self.remote_max_age_days
        ):
            raise ValueError(
                f"cache_reuse_max_age_days ({self.cache_reuse_max_age_days:g}) must not "
                f"exceed remote_max_age_days ({self.remote_max_age_days:g})"
            )
        return self


#: Non-model top-level sections of a run config, keyed by section name.
TOP_LEVEL_SECTIONS: dict[str, Type[StrictConfigModel]] = {
    "data": DataConfig,
    "plots": PlotsConfig,
    "inference": InferenceConfig,
    "opt": OptConfig,
    "satellites": SatellitesConfig,
}


# ---------------------------------------------------------------------------
# Component specifications & ModelConfig
# ---------------------------------------------------------------------------

TConfig = TypeVar("TConfig", bound=StrictConfigModel)


class ComponentSpec(BaseModel):
    """One resolved and validated entry in ``model.components``."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    ref: str
    cls: type
    config: StrictConfigModel

    def instantiate(self):
        return self.cls(self.config)

    def to_yaml_entry(self) -> Union[str, dict[str, dict[str, Any]]]:
        dumped = self.config.to_yaml_dict()
        if not dumped:
            return self.ref
        return {self.ref: dumped}


def _parse_component_entry(entry: Any, index: int) -> ComponentSpec:
    """Resolve and validate one ``model.components`` entry."""
    if isinstance(entry, ComponentSpec):
        return entry

    if isinstance(entry, str):
        raw_ref = entry
        raw_params: Mapping[str, Any] = {}
    elif isinstance(entry, Mapping):
        if len(entry) != 1:
            raise ValueError(
                f"model.components[{index}] must be a component name string or a "
                f"single-key mapping '{{module.Class: {{...}}}}', got keys {list(entry)}."
            )
        ((raw_ref, val),) = entry.items()
        if not isinstance(raw_ref, str):
            raise ValueError(
                f"model.components[{index}] key must be a component reference string, "
                f"got {raw_ref!r}."
            )
        if val is None:
            raw_params = {}
        elif isinstance(val, Mapping):
            raw_params = val
        else:
            raise ValueError(
                f"model.components[{index}] ({raw_ref}) parameters must be a mapping "
                f"or null, got {type(val).__name__} ({val!r})."
            )
    else:
        raise ValueError(
            f"model.components[{index}] must be a component name string or a "
            f"single-key mapping '{{module.Class: {{...}}}}', got {type(entry).__name__}."
        )

    from tabascal.imports import canonical_component_ref, import_components

    try:
        cls = import_components([raw_ref])[0]
    except ImportError as exc:
        raise ValueError(f"model.components[{index}]: {exc}") from exc

    ref = canonical_component_ref(raw_ref, cls)
    schema_cls: Type[StrictConfigModel] = getattr(cls, "Config", EmptyComponentConfig)
    try:
        validated = schema_cls.model_validate(dict(raw_params))
    except ValidationError as err:
        formatted = format_validation_error(
            err,
            prefix=f"model.components[{index}] ({ref})",
            model_cls=schema_cls,
            component_cls=cls,
        )
        raise ValueError(str(formatted)) from err

    return ComponentSpec(ref=ref, cls=cls, config=validated)


def _component_error_lines(msg: str) -> list[str]:
    """Extract bullet lines from a component parse error for multi-error reporting."""
    prefix = "Configuration validation failed:"
    body = msg[len(prefix):].strip() if msg.startswith(prefix) else msg.strip()
    out = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        out.append(stripped if stripped.startswith("- ") else f"- {stripped}")
    return out


class ModelConfig(StrictConfigModel):
    """Forward model precision and ordered component list."""

    null_is_none: ClassVar[FrozenSet[str]] = frozenset({"components"})

    precision: Literal["single", "double"] = Field(
        default="single",
        description="Floating-point precision of the model: 'single' (fp32) or 'double' (fp64).",
    )
    components: list[ComponentSpec] = Field(
        default_factory=list,
        description=(
            "Ordered list of forward-model components, each a 'module.Class' reference "
            "or a single-key mapping '{module.Class: {options...}}'."
        ),
    )

    @field_validator("components", mode="before")
    @classmethod
    def _parse_components(cls, value: Any) -> list[ComponentSpec]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(f"model.components must be a list, got {type(value).__name__}")
        specs: list[ComponentSpec] = []
        error_lines: list[str] = []
        for i, entry in enumerate(value):
            try:
                specs.append(_parse_component_entry(entry, i))
            except ValueError as err:
                error_lines.extend(_component_error_lines(str(err)))
        if error_lines:
            raise ValueError("\n".join(error_lines))
        return specs

    def find_all(self, config_type: Type[TConfig]) -> list[Tuple[ComponentSpec, TConfig]]:
        """Every ``(spec, config)`` whose config is an instance of ``config_type``."""
        return [
            (spec, spec.config)
            for spec in self.components
            if isinstance(spec.config, config_type)
        ]

    def find_one(self, config_type: Type[TConfig]) -> Optional[TConfig]:
        """The single component config of ``config_type``, or ``None`` if absent."""
        matches = self.find_all(config_type)
        if len(matches) > 1:
            refs = [spec.ref for spec, _ in matches]
            raise ValueError(
                f"Expected at most one component with {config_type.__name__}, found {refs}."
            )
        return matches[0][1] if matches else None

    def find_shared(self, config_type: Type[TConfig]) -> Optional[TConfig]:
        """The first component config of ``config_type``, or ``None`` if absent.

        For settings several components may carry: the model validator has
        already checked that every component of ``config_type`` agrees on
        them, so any one of them answers for all.
        """
        matches = self.find_all(config_type)
        return matches[0][1] if matches else None

    def find_one_spec(self, config_type: Type[TConfig]) -> Optional[ComponentSpec]:
        """The single :class:`ComponentSpec` whose config is ``config_type``, or ``None``."""
        matches = self.find_all(config_type)
        if len(matches) > 1:
            refs = [spec.ref for spec, _ in matches]
            raise ValueError(
                f"Expected at most one component with {config_type.__name__}, found {refs}."
            )
        return matches[0][0] if matches else None

    def component_classes(self) -> list[type]:
        return [spec.cls for spec in self.components]

    def component_class_names(self) -> list[str]:
        return [spec.cls.__name__ for spec in self.components]

    def instantiate_components(self) -> list[Any]:
        return [spec.instantiate() for spec in self.components]

    @model_validator(mode="after")
    def _validate_model_consistency(self) -> "ModelConfig":
        # 1. At most one orbit trajectory component.
        traj_matches = self.find_all(TrajectoryConfig)
        if len(traj_matches) > 1:
            refs = [spec.ref for spec, _ in traj_matches]
            raise ValueError(
                f"model.components has multiple orbit trajectory components ({refs}); "
                "include at most one."
            )

        # 2. If multiple RFI visibility components exist, their integration settings must agree.
        freq_matches = self.find_all(FreqIntegrationMixin)
        if len(freq_matches) > 1:
            n_freqs = {cfg.n_int_freq for _, cfg in freq_matches}
            if len(n_freqs) > 1:
                raise ValueError(
                    "RFI visibility components disagree on n_int_freq: "
                    + ", ".join(f"{spec.ref}={cfg.n_int_freq}" for spec, cfg in freq_matches)
                )
        fine_matches = self.find_all(FineTimeIntegrationMixin)
        if len(fine_matches) > 1:
            factors = {c.time_int_factor for _, c in fine_matches}
            if len(factors) > 1:
                raise ValueError(
                    "Fine-grid RFI visibility components disagree on time_int_factor."
                )
        var_matches = self.find_all(VariableFineTimeIntegrationMixin)
        if len(var_matches) > 1:
            bins = {(c.min_time_bins, c.max_time_bins) for _, c in var_matches}
            if len(bins) > 1:
                raise ValueError(
                    "Variable fine-grid RFI visibility components disagree on "
                    "min_time_bins / max_time_bins."
                )
        # The RFI pad factors set the shared fine grid, so every RFI signal
        # component has to agree on them too.
        rfi_matches = self.find_all(RFIGPConfig)
        if len(rfi_matches) > 1:
            pads = {(c.freq_pad_factor, c.time_pad_factor) for _, c in rfi_matches}
            if len(pads) > 1:
                raise ValueError(
                    "RFI signal components disagree on freq_pad_factor / time_pad_factor: "
                    + ", ".join(
                        f"{spec.ref}=({cfg.freq_pad_factor}, {cfg.time_pad_factor})"
                        for spec, cfg in rfi_matches
                    )
                )

        if not self.components:
            return self

        # 3. Precision compatibility.
        if self.precision != "double":
            offenders = sorted(
                spec.cls.__name__
                for spec in self.components
                if getattr(spec.cls, "requires_double", False)
            )
            if offenders:
                raise ValueError(
                    "model.precision is 'single', but these components require double "
                    f"precision: {', '.join(offenders)}. "
                    "Set model.precision to 'double' to use them."
                )

        # 4. Dependency order & fine/data grid consistency.
        from tabascal.components import BASE_STATE_KEYS, validate_component_order

        validate_component_order(self.instantiate_components(), BASE_STATE_KEYS)

        # 5. ConstGains with free flux scale and fitted amplitude requires FixedDiscreteSky.
        names = set(self.component_class_names())
        for spec in self.components:
            if spec.cls.__name__ == "ConstGains":
                cfg = spec.config
                fit_amp = cfg.fit in ("complex", "amplitude")
                if not cfg.fix_flux_scale and fit_amp and "FixedDiscreteSky" not in names:
                    raise ValueError(
                        f"{spec.ref}.fix_flux_scale is false, which frees the overall gain "
                        "amplitude, but the model has no fixed-flux sky to set the flux "
                        "scale against. Add ast_signal.FixedDiscreteSky (with "
                        "ast_vis.DiscreteSkyVis) to model.components, or leave "
                        "fix_flux_scale true."
                    )

        return self

    def to_yaml_dict(self) -> dict[str, Any]:
        return {
            "precision": self.precision,
            "components": [spec.to_yaml_entry() for spec in self.components],
        }


# ---------------------------------------------------------------------------
# Top-level RunConfig
# ---------------------------------------------------------------------------

class RunConfig(StrictConfigModel):
    """Complete validated configuration for a TABASCAL run."""

    model: ModelConfig = Field(
        description="Forward model precision and ordered list of configured components.",
    )
    data: DataConfig = Field(
        default_factory=DataConfig,
        description="Observation, I/O, noise, flags, and external calibration settings.",
    )
    plots: PlotsConfig = Field(
        default_factory=PlotsConfig,
        description="Diagnostic plotting toggles.",
    )
    inference: InferenceConfig = Field(
        default_factory=InferenceConfig,
        description="Top-level inference controls and random seed.",
    )
    opt: OptConfig = Field(
        default_factory=OptConfig,
        description="MAP optimisation parameters.",
    )
    satellites: SatellitesConfig = Field(
        default_factory=SatellitesConfig,
        description="Satellite NORAD IDs and orbit-record resolution settings.",
    )

    @model_validator(mode="after")
    def _validate_cross_sections(self) -> "RunConfig":
        # Validate opt.optimizer / schedule / stop / checkpoints against model.precision
        # whenever any of them is configured (keeps lightweight default configs fast).
        if (
            self.opt.optimizer is not None
            or self.opt.schedule is not None
            or self.opt.stop is not None
            or self.opt.checkpoints is not None
        ):
            from tabascal.opt_schedule import validate_opt

            validate_opt(self.opt, precision=self.model.precision)
        return self

    def to_yaml_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_yaml_dict(),
            "data": self.data.to_yaml_dict(),
            "plots": self.plots.to_yaml_dict(),
            "inference": self.inference.to_yaml_dict(),
            "opt": self.opt.to_yaml_dict(),
            "satellites": self.satellites.to_yaml_dict(),
        }


def validate_config(raw: Any) -> RunConfig:
    """Validate a raw config mapping (or existing :class:`RunConfig`) into :class:`RunConfig`.

    Raises :class:`ConfigValidationError` (a :class:`ValueError` subclass) with
    clear dotted field paths on any schema violation.
    """
    if isinstance(raw, RunConfig):
        return raw
    if not isinstance(raw, Mapping):
        raise ConfigValidationError(
            f"Configuration must be a YAML mapping, got {type(raw).__name__}."
        )
    old = [key for key in _OLD_LAYOUT_KEYS if key in raw]
    if old:
        raise ConfigValidationError(
            f"This config is in the old sectioned layout ({', '.join(old)} at the top level). "
            "Each component now takes its own options under its entry in 'model.components'. "
            "Convert it with: tabascal migrate-config <old.yaml> -o <new.yaml>"
        )
    # Strip top-level bare section headers (e.g. `data:` with nothing under it) ONLY for
    # known optional sections that have defaults, while forbidding unknown sections.
    cleaned = dict(raw)
    for section in TOP_LEVEL_SECTIONS:
        if section in cleaned and cleaned[section] is None:
            cleaned[section] = {}
    try:
        return RunConfig.model_validate(cleaned)
    except ValidationError as err:
        raise format_validation_error(err, model_cls=RunConfig) from None
