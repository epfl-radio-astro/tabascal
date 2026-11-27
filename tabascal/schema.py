"""Pydantic v2 configuration schemas for TABASCAL.

Every top-level config section (``model``, ``data``, ``plots``, ``opt``,
plus the top-level ``seed``) and every
:class:`~tabascal.components.Component` subclass declares its own strict
Pydantic model with field descriptions and range/type checks. Common fields
shared by related components are expressed via Python mixin inheritance here —
never as shared global YAML sections on the user side.

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
    ValidationInfo,
    field_validator,
    model_validator,
)


# ---------------------------------------------------------------------------
# Public error type & formatting
# ---------------------------------------------------------------------------

#: Top-level sections from the pre-component sectioned layout that moved onto
#: individual components under ``model.components``.
_OLD_LAYOUT_KEYS = ("ast", "rfi", "gains", "satellites")


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
        # (e.g. data_col belongs in 'data') or to another component.
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


def _child_models(annotation: Any) -> list[type]:
    """Extract every Pydantic ``BaseModel`` subclass referenced by ``annotation``."""
    from typing import get_args

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    found: list[type] = []
    for arg in get_args(annotation):
        for sub in _child_models(arg):
            if sub not in found:
                found.append(sub)
    return found


def _match_union_member(candidates: list[type], tag: Any) -> Optional[type]:
    """Match a Pydantic discriminated-union branch tag to its model class."""
    from typing import get_args

    for cls in candidates:
        kind_field = cls.model_fields.get("kind")
        if kind_field is not None and tag in get_args(kind_field.annotation):
            return cls
        if cls.__name__ == tag:
            return cls
    return None


def _resolve_model_for_loc(root_cls: Optional[type], loc: tuple) -> Optional[type]:
    """Walk ``root_cls`` along ``loc[:-1]`` to find the model that rejected ``loc[-1]``."""
    if root_cls is None or len(loc) <= 1:
        return root_cls
    current: Optional[type] = root_cls
    candidates: Optional[list[type]] = None
    for prev, part in zip((None, *loc), loc[:-1]):
        if isinstance(part, int):
            continue
        if candidates is not None:
            matched = _match_union_member(candidates, part)
            if matched is not None:
                current = matched
                candidates = None
                continue
            return None
        if current is None or not (isinstance(current, type) and issubclass(current, BaseModel)):
            return None
        field_info = current.model_fields.get(part) if isinstance(part, str) else None
        if field_info is None:
            # Discriminator tag on a non-model union (e.g. 'inline' on point_sources)
            # leaves the resolved element model intact.
            if _is_hidden_union_tag(prev, part):
                continue
            return None
        models = _child_models(field_info.annotation)
        if len(models) == 1:
            current = models[0]
        elif len(models) > 1:
            current = None
            candidates = models
        else:
            return None
    return current


#: Internal Pydantic ``Tag(...)`` names omitted when rendering user-facing error
#: paths, keyed by the field whose union they tag. A tag is hidden only directly
#: after its field, so a user key spelled the same elsewhere is still shown.
_HIDDEN_UNION_TAGS: dict[str, frozenset[str]] = {
    "point_sources": frozenset({"file", "inline"}),
}


def _is_hidden_union_tag(prev: Any, part: Any) -> bool:
    return isinstance(prev, str) and part in _HIDDEN_UNION_TAGS.get(prev, ())


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
        for prev, part in zip((None, *raw_loc), raw_loc):
            if _is_hidden_union_tag(prev, part):
                continue
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
            target_cls = _resolve_model_for_loc(model_cls, raw_loc)
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
    their own defaults as well, written as floats too.
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
    if not value.strip():
        raise ValueError("path must not be empty; use null when unset")
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
        extra = getattr(obj, "__pydantic_extra__", None) or {}
        return {k: _to_plain_yaml(v) for k, v in {**obj.__dict__, **extra}.items()}
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
    setting -- ``min_elevation_deg: null`` is no elevation mask. It never means "the
    default": a default is had by leaving the key out, and an option whose
    default is computed names the computation (``corr_freq_hz: half_band``).
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

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

    null_is_none: ClassVar[FrozenSet[str]] = frozenset({"norad_ids"})

    norad_ids: list[PositiveInt] = Field(
        default_factory=list,
        description=(
            "NORAD catalogue IDs of satellites to model. Every listed ID must resolve "
            "to an acceptable orbit record. Provide this, norad_ids_path, or CLI -np/--norad-path."
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
    min_elevation_deg: Optional[FiniteFloat] = Field(
        default=0.0,
        description=(
            "Elevation in degrees below which a satellite's RFI signal is masked to zero. "
            "0 masks a satellite only while it is below the geometric horizon; null "
            "disables masking. A satellite never above it stops the run."
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
    def _check_age_bounds(self) -> "TrajectoryConfig":
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


class FreqIntegrationMixin(StrictConfigModel):
    """Code-level mixin for RFI visibility components that integrate over frequency samples."""

    n_int_freq: PositiveInt = Field(
        default=1,
        description=(
            "Integration samples per frequency channel of the RFI visibility. Given "
            "directly: there is no estimate on this axis."
        ),
    )


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
        raise ValueError(f"std_jy must be a positive number or {options}, got {value!r}")
    return _check_positive_float(value)


class FourierGPConfig(StrictConfigModel):
    """Code-level mixin shared by Fourier-domain Gaussian process components."""

    _std_named: ClassVar[Tuple[str, ...]] = ("data",)

    std_jy: Union[PositiveFloat, Literal["data"]] = Field(
        default="data",
        description=(
            "Width of the prior in Jy (rms|V|). 'data' measures it from the observed "
            "visibilities; a positive number sets it directly."
        ),
    )
    corr_freq_hz: Optional[PositiveFloat] = Field(
        default=None,
        description=(
            "Bandwidth, in Hz, over which the signal stays correlated along frequency. "
            "null disables roll-off along the frequency axis."
        ),
    )
    gammas: PositivePair = Field(
        default=(5.0, 5.0),
        description="Power-law roll-off exponents past the knee on the (frequency, time) axes.",
    )
    cutoff: CutoffFloat = Field(
        default=1e-6,
        description=(
            "Relative power below which a k-mode is dropped from the latent grid; "
            "the retained spectrum is re-normalised after the cut."
        ),
    )
    freq_pad_factor: FloatGe1 = Field(
        default=2.0,
        description="Zero-padding factor along the frequency axis of the Fourier domain (>= 1).",
    )
    time_pad_factor: FloatGe1 = Field(
        default=2.0,
        description="Zero-padding factor along the time axis of the Fourier domain (>= 1).",
    )

    @field_validator("std_jy", mode="before")
    @classmethod
    def _validate_std(cls, value: Any) -> Any:
        return _check_std_value(value, named=cls._std_named)


#: ``rfi_signal`` ``std_jy``: twice the largest observed visibility (GitHub #227).
MAX_VIS = "max_vis"
#: ``rfi_signal`` ``corr_freq_hz``: half the observed band.
HALF_BAND = "half_band"
#: ``rfi_signal`` ``corr_time_s``: half the observation.
HALF_DURATION = "half_duration"


class RFIGPConfig(FourierGPConfig):
    """Configuration shared by all GP-based RFI signal components."""

    _std_named: ClassVar[Tuple[str, ...]] = ("data", MAX_VIS)

    std_jy: Union[PositiveFloat, Literal["data", "max_vis"]] = Field(
        default="data",
        description=(
            "Width of the prior on one RFI source, in Jy (rms|V|). 'data' measures the "
            "total rms|V| of the observed visibilities (sky and noise included); "
            "'max_vis' is twice the largest observed visibility."
        ),
    )
    corr_freq_hz: Union[PositiveFloat, Literal["half_band"]] = Field(
        default=1e6,
        description=(
            "Bandwidth, in Hz, over which the RFI signal stays correlated (knee along "
            "the frequency axis). 'half_band' uses half the observed band."
        ),
    )
    corr_time_s: Union[PositiveFloat, Literal["half_duration"]] = Field(
        default=24.0,
        description=(
            "Time, in seconds, over which the RFI signal stays correlated (knee along "
            "the time axis). 'half_duration' uses half the observation."
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
            "est), 'matched-filter', 'truth', 'zeros', 'ones', or 'sample'."
        ),
    )
    mean: Literal["zeros", "data", "est", "matched-filter"] = Field(
        default="zeros",
        description=(
            "Mean of the prior over the RFI signal parameters: 'zeros', 'data', "
            "'est', or 'matched-filter'."
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

    @field_validator("init", "mean", mode="before")
    @classmethod
    def _no_aliases(cls, value: Any, info: ValidationInfo) -> Any:
        # bool is an int, so `init: true` would otherwise read as a number.
        if isinstance(value, (bool, int, float)):
            raise ValueError(
                f"{info.field_name} takes a name, not {value!r}: "
                + ("'zeros', 'ones', " if info.field_name == "init" else "'zeros', ")
                + "'matched-filter' and the rest are spelled out."
            )
        return value

    @model_validator(mode="after")
    def _est_is_read_exactly_when_given(self) -> "RFIGPConfig":
        reads = [key for key in ("init", "mean") if getattr(self, key) == "est"]
        if reads and self.est is None:
            raise ValueError(
                f"{' and '.join(f'{k}: est' for k in reads)} read the light curves in "
                "the file 'est' names, and est is not set. Give est a path, or use "
                "'matched-filter', which measures them from the data."
            )
        if not reads and self.est is not None:
            raise ValueError(
                f"est is set ({self.est!r}) but nothing reads it: only init: est and "
                "mean: est do. Set one of them, or leave est out."
            )
        return self

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
    freq_hz: Optional[PositiveFloat] = Field(
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
    noise_jy: Optional[Union[PositiveFloat, str]] = Field(
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
    exclude_flagged: StrictBool = Field(
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

    @field_validator("noise_jy", mode="before")
    @classmethod
    def _validate_noise(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, os.PathLike):
            value = os.fspath(value)
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("data.noise_jy path must not be empty; use null to read from the MS")
            return value
        if isinstance(value, bool):
            raise ValueError(
                f"data.noise_jy = {value!r} is neither a number nor a path to an .npz of noise values."
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


#: The optax optimisers ``opt.optimizer`` may name (``tabascal.opt_schedule``
#: builds them; listed here so the schema does not import optax).
OPTIMIZER_NAMES = (
    "adabelief", "adam", "adamw", "amsgrad", "nadam", "radam", "adamax", "yogi",
    "lion", "lamb", "novograd", "rmsprop", "sgd", "adagrad",
)


class OptimizerConfig(StrictConfigModel):
    """``opt.optimizer``: an optax optimiser, its scalar arguments and a gradient clip.

    The arguments besides ``name`` and ``clip_norm`` are the optimiser's own,
    under their optax names, so they differ by optimiser; which ones it takes,
    and the range each must lie in at the run's precision, is checked against
    optax itself by :func:`tabascal.opt_schedule.validate_optimizer`.
    """

    model_config = ConfigDict(extra="allow")

    name: Literal[OPTIMIZER_NAMES] = Field(  # type: ignore[valid-type]
        default="adabelief",
        description="The optax optimiser.",
    )
    clip_norm: Optional[PositiveFloat] = Field(
        default=None,
        description="Clip the gradient's global norm to this before the optimiser sees it; null is no clip.",
    )

    @model_validator(mode="before")
    @classmethod
    def _a_name_alone(cls, data: Any) -> Any:
        return {"name": data} if isinstance(data, str) else data

    def to_yaml_dict(self) -> dict[str, Any]:
        return _to_plain_yaml(self)


class _ScheduleBase(StrictConfigModel):
    warmup_steps: NonNegativeInt = Field(
        default=0,
        description="Steps over which the step size rises linearly from warmup_factor * epsilon.",
    )
    warmup_factor: FiniteFloat = Field(
        default=0.0,
        description="Where the warmup starts, as a fraction of epsilon, in [0, 1].",
    )


    def to_yaml_dict(self) -> dict[str, Any]:
        dumped = super().to_yaml_dict()
        return {"kind": dumped.pop("kind"), **dumped}


class ConstantSchedule(_ScheduleBase):
    """``epsilon`` throughout."""

    kind: Literal["constant"]


class StepSchedule(_ScheduleBase):
    """``epsilon`` multiplied by ``factor`` after each step in ``boundaries``."""

    kind: Literal["step"]
    boundaries: list[PositiveInt] = Field(
        description="Steps after which the step size drops; increasing, below max_iter.",
    )
    factor: Union[PositiveFloat, list[PositiveFloat]] = Field(
        default=0.1,
        description="The drop at every boundary, or one per boundary.",
    )


class ExponentialSchedule(_ScheduleBase):
    """``epsilon * final_factor ** (step / max_iter)``."""

    kind: Literal["exponential"]
    final_factor: FiniteFloat = Field(description="The step size at max_iter, as a fraction of epsilon, in (0, 1].")


class CosineSchedule(_ScheduleBase):
    """Half a cosine from ``epsilon`` to ``final_factor * epsilon`` at ``max_iter``."""

    kind: Literal["cosine"]
    final_factor: FiniteFloat = Field(default=0.0, description="The step size at max_iter, as a fraction of epsilon, in [0, 1].")


class LinearSchedule(_ScheduleBase):
    """A straight line from ``epsilon`` to ``final_factor * epsilon`` at ``max_iter``."""

    kind: Literal["linear"]
    final_factor: FiniteFloat = Field(default=0.0, description="The step size at max_iter, as a fraction of epsilon, in [0, 1].")


ScheduleConfig = Annotated[
    Union[ConstantSchedule, StepSchedule, ExponentialSchedule, CosineSchedule, LinearSchedule],
    Field(discriminator="kind"),
]


class StopConfig(StrictConfigModel):
    """``opt.stop``: end the fit once the loss has stopped moving."""

    window: PositiveInt = Field(description="Steps in each of the two windows whose mean losses are compared.")
    rtol: FiniteFloat = Field(description="Largest relative change between the two means that counts as stopped; > 0.")
    min_steps: NonNegativeInt = Field(default=0, description="Steps the fit runs before it may stop.")


class OptConfig(StrictConfigModel):
    """MAP optimisation parameters."""

    epsilon: PositiveFloat = Field(
        default=1e-2,
        description="Base optimiser step size.",
    )
    max_iter: NonNegativeInt = Field(
        default=1000,
        description=(
            "Optimisation steps in all: under dual_run the first half at epsilon and "
            "the second at epsilon / 10. 0 runs no fit, only the initial prediction."
        ),
    )
    dual_run: StrictBool = Field(
        default=True,
        description="Run a second optimisation phase at epsilon / 10 (ignored when schedule is set).",
    )
    optimizer: Optional[OptimizerConfig] = Field(
        default=None,
        description=(
            "Optax optimiser name or mapping of optimiser arguments plus optional "
            "clip_norm. null uses adabelief with optax defaults."
        ),
    )
    schedule: Optional[ScheduleConfig] = Field(
        default=None,
        description=(
            "Single-optimiser step-size schedule (step, exponential, cosine, linear, "
            "constant) with optional warmup_steps and warmup_factor. null keeps dual_run."
        ),
    )
    stop: Optional[StopConfig] = Field(
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

    @model_validator(mode="after")
    def _dual_run_or_schedule(self) -> "OptConfig":
        if self.schedule is not None and "dual_run" in self.model_fields_set:
            raise ValueError(
                "dual_run and schedule are two ways of running the fit, and with a "
                "schedule dual_run is not read. Leave dual_run out; "
                "{kind: step, boundaries: [N], factor: 0.1} is dual_run's drop."
            )
        return self

    def to_yaml_dict(self) -> dict[str, Any]:
        dumped = super().to_yaml_dict()
        if self.schedule is not None:
            # Not read with a schedule, and refused beside one when written.
            dumped.pop("dual_run")
        return dumped

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


#: Non-model top-level sections of a run config, keyed by section name.
TOP_LEVEL_SECTIONS: dict[str, Type[StrictConfigModel]] = {
    "data": DataConfig,
    "plots": PlotsConfig,
    "opt": OptConfig,
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

    if ":" in raw_ref:
        raise ValueError(
            f"model.components[{index}]: component references are written module.Class, "
            f"not {raw_ref!r}; write {raw_ref.replace(':', '.')!r}."
        )
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


#: The component kinds a model takes at most one of, identified by the state
#: key they write, and why.
_ONE_PER_KIND = {
    "rfi_signal": ("rfi_A", "each writes the RFI amplitude rfi_A, so the last one listed replaces the others."),
    "rfi_vis": ("vis_rfi", "each adds the RFI visibilities into vis_rfi, so the RFI would be counted once per component."),
    "gains": ("vis_obs", "each writes the observed visibilities vis_obs, so the last one listed replaces the others."),
}


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

        # 2. At most one component of each kind that would overwrite or double
        # another: a second rfi_signal replaces rfi_A, leaving the first one's
        # parameters fitting nothing; a second rfi_vis adds the RFI into vis_rfi
        # again; a second gains component replaces vis_obs.
        for kind, (state_key, why) in _ONE_PER_KIND.items():
            refs = [
                spec.ref
                for spec in self.components
                if state_key in getattr(spec.cls, "output_shapes", {})
            ]
            if len(refs) > 1:
                raise ValueError(
                    f"model.components has {len(refs)} {kind} components ({', '.join(refs)}); "
                    f"include at most one: {why}"
                )

        # 2b. No component class may appear twice: Model keys constants by class
        # name and parameter sites are global, so a second instance collides with
        # or doubles the first.
        seen_cls: set[type] = set()
        for spec in self.components:
            if spec.cls in seen_cls:
                raise ValueError(
                    f"model.components lists {spec.ref} more than once; "
                    "include each component class at most once."
                )
            seen_cls.add(spec.cls)

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

        # 4. Run in dependency order, whatever order they were listed in, and
        # check missing producers / fine-vs-data grid consistency citing the
        # user's written indices.
        from tabascal.components import (
            BASE_STATE_KEYS,
            dependency_order,
            validate_component_order,
        )

        order = dependency_order([spec.cls for spec in self.components])
        ordered_specs = [self.components[i] for i in order]
        validate_component_order(
            [spec.instantiate() for spec in ordered_specs],
            BASE_STATE_KEYS,
            source_indices=order,
        )
        # object.__setattr__, not plain assignment: under validate_assignment=True
        # a plain assignment re-validates the model, which runs this validator
        # again from inside itself, without end. The sorted list is the one just
        # checked, so bypassing validation skips nothing.
        object.__setattr__(self, "components", ordered_specs)

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
    seed: StrictInt = Field(
        default=1,
        description=(
            "The one random seed of the run; 0 is a valid seed. Each component that "
            "draws folds its own fixed tag into it, as do the start jitter and prior plots."
        ),
    )
    opt: OptConfig = Field(
        default_factory=OptConfig,
        description="MAP optimisation parameters.",
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
            "seed": self.seed,
            "opt": self.opt.to_yaml_dict(),
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
            "Each component now takes its own options under its entry in 'model.components'."
        )
    if "inference" in raw:
        raise ConfigValidationError(
            "The inference section is gone: the seed is the top-level key 'seed', and "
            "'opt.max_iter: 0' runs no fit, which is what 'inference.opt: false' did."
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
