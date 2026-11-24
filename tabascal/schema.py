"""Pydantic v2 configuration schemas for TABASCAL.

Every top-level config section (``data``, ``plots``, ``inference``, ``opt``,
``model``) and every :class:`~tabascal.components.Component` subclass declares
its own strict Pydantic model. Common fields shared by related components are
expressed via Python mixin inheritance here — never as shared global YAML
sections on the user side.

This module deliberately avoids importing JAX, NumPyro, Optax, or Dask at
module import time so lightweight CLI commands (``tabascal -h``, ``tab2MS``,
``tabascal rfi-per-sat``) pay no accelerator import cost.
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Annotated, Any, ClassVar, Literal, Optional, Tuple, Type, TypeVar, Union

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
# Public error type
# ---------------------------------------------------------------------------

class ConfigValidationError(ValueError):
    """Raised when a TABASCAL configuration fails schema validation."""

    def __init__(self, message: str, *, cause: Optional[ValidationError] = None):
        super().__init__(message)
        self.validation_error = cause


def format_validation_error(err: ValidationError, *, prefix: str = "") -> ConfigValidationError:
    """Turn a Pydantic :class:`ValidationError` into a readable :class:`ConfigValidationError`."""
    lines = []
    for item in err.errors():
        loc_parts = []
        for part in item.get("loc", ()):
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


def _check_positive_number(value: Any) -> Union[int, float]:
    """Finite positive number, keeping ints as int (for lax.integer_pow bit-exactness)."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"must be a positive finite number, got {value!r}")
    if isinstance(value, Integral):
        out_i = int(value)
        try:
            as_f = float(out_i)
        except OverflowError:
            raise ValueError(f"is too large to be a floating-point number: {value!r}") from None
        if not math.isfinite(as_f) or out_i <= 0:
            raise ValueError(f"must be finite and greater than zero, got {value!r}")
        return out_i
    return _check_positive_float(value)


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


def _check_positive_pair(value: Any) -> Tuple[Union[int, float], Union[int, float]]:
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
    return (_check_positive_number(items[0]), _check_positive_number(items[1]))


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
PositiveNumber = Annotated[Union[int, float], BeforeValidator(_check_positive_number)]
PositivePair = Annotated[
    Tuple[Union[int, float], Union[int, float]],
    BeforeValidator(_check_positive_pair),
]
OptionalPathStr = Annotated[Optional[str], BeforeValidator(_check_path_str)]
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

class StrictConfigModel(BaseModel):
    """Base schema for every TABASCAL configuration section and component."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

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


class OrbitSourceConfig(StrictConfigModel):
    """Code-level mixin for satellite trajectory components that consume TLE/OMM orbits."""

    norad_ids: list[PositiveInt] = Field(default_factory=list)
    norad_ids_path: OptionalPathStr = Field(
        default=None,
        json_schema_extra={"input_path": True},
    )
    extra_orbit_dir: OptionalPathStr = Field(
        default=None,
        json_schema_extra={"input_path": True, "fingerprint_kind": "orbits"},
    )
    extra_orbit_max_age_days: Optional[NonNegativeFloat] = None
    remote_max_age_days: Optional[NonNegativeFloat] = DEFAULT_REMOTE_MAX_AGE_DAYS
    cache_reuse_max_age_days: Optional[NonNegativeFloat] = DEFAULT_CACHE_REUSE_MAX_AGE_DAYS
    min_elevation: Optional[FiniteFloat] = 0.0

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
    def _check_age_bounds(self) -> "OrbitSourceConfig":
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

    n_int_freq: PositiveInt = 1


class FineTimeIntegrationMixin(FreqIntegrationMixin):
    """Code-level mixin for fine-grid RFI visibility components."""

    time_int_factor: PositiveFloat = 1.0
    min_time_bins: PositiveInt = 1
    max_time_bins: PositiveInt = 30

    @model_validator(mode="after")
    def _check_time_bins(self) -> "FineTimeIntegrationMixin":
        if self.min_time_bins > self.max_time_bins:
            raise ValueError(
                f"min_time_bins ({self.min_time_bins}) must not exceed "
                f"max_time_bins ({self.max_time_bins})"
            )
        return self


def _check_std_value(value: Any, *, allow_none: bool) -> Any:
    if value is None:
        if allow_none:
            return None
        raise ValueError("std is required and cannot be null; give a positive number in Jy or 'data'")
    if isinstance(value, str):
        if value == "data":
            return "data"
        raise ValueError(f"std must be a positive number or 'data', got {value!r}")
    return _check_positive_float(value)


class FourierGPConfig(StrictConfigModel):
    """Code-level mixin shared by Fourier-domain Gaussian process components."""

    std: Optional[Union[PositiveFloat, Literal["data"]]] = None
    corr_freq: Optional[PositiveFloat] = None
    gammas: PositivePair = (5.0, 5.0)
    cutoff: CutoffFloat = 1e-6
    freq_pad_factor: FloatGe1 = 2.0
    time_pad_factor: FloatGe1 = 2.0

    @field_validator("cutoff", mode="before")
    @classmethod
    def _default_cutoff(cls, value: Any) -> Any:
        if value is None:
            return cls.model_fields["cutoff"].default
        return value


class RFIGPConfig(FourierGPConfig):
    """Code-level mixin shared by all GP-based RFI signal components."""

    std: Optional[Union[PositiveFloat, Literal["data"]]] = None
    corr_freq: Optional[PositiveFloat] = 1e6
    corr_time: Optional[PositiveFloat] = 24.0
    gammas: PositivePair = (3, 3)
    cutoff: CutoffFloat = 1e-9
    signal_type: Literal["complex", "real"] = "complex"
    init: Literal["prior", "est", "matched-filter", "truth", "zeros", "ones", "sample"] = "sample"
    mean: Literal["zeros", "data", "est", "matched-filter"] = "zeros"
    est: OptionalPathStr = Field(
        default=None,
        json_schema_extra={"input_path": True},
    )

    @field_validator("std", mode="before")
    @classmethod
    def _validate_std(cls, value: Any) -> Any:
        return _check_std_value(value, allow_none=True)

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
            label = f"{ref}.init: truth" if ref else "rfi.init: truth"
            return {"rfi_A": label}
        return {}


# ---------------------------------------------------------------------------
# Top-level section schemas
# ---------------------------------------------------------------------------

class DataConfig(StrictConfigModel):
    """Observation, I/O, noise, flags, and external calibration settings."""

    out_dir: OptionalPathStr = None
    truth_zarr: OptionalPathStr = Field(
        default=None,
        json_schema_extra={"input_path": True, "resolved_by_runner": True},
    )
    ms_path: OptionalPathStr = Field(
        default=None,
        json_schema_extra={
            "input_path": True,
            "resolved_by_runner": True,
            "fingerprint_kind": "ms",
        },
    )
    freq: Optional[PositiveFloat] = None
    skip_ms_write: StrictBool = False
    row_chunk: Optional[PositiveInt] = None
    data_col: NonEmptyStr = "DATA"
    corr: NonEmptyStr = "xx"
    noise: Optional[Union[PositiveFloat, str]] = Field(
        default=None,
        json_schema_extra={"input_path": True},
    )
    gain_table: Optional[Union[str, list[str]]] = Field(
        default=None,
        json_schema_extra={"input_path": True, "is_path_list": True},
    )
    flags: StrictBool = False
    save_rfi_per_sat: StrictBool = False

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

    init: StrictBool = False
    truth: StrictBool = False
    prior: StrictBool = False
    opt: StrictBool = False
    losses: StrictBool = False
    prior_samples: PositiveInt = 100


class InferenceConfig(StrictConfigModel):
    """Top-level inference controls."""

    opt: StrictBool = True
    seed: Optional[StrictInt] = None

    @property
    def effective_seed(self) -> int:
        return 1 if self.seed is None else self.seed


class OptConfig(StrictConfigModel):
    """MAP optimisation parameters."""

    epsilon: PositiveFloat = 1e-2
    max_iter: NonNegativeInt = 500
    guide: Literal["map"] = "map"
    dual_run: StrictBool = True
    optimizer: Optional[Union[str, dict[str, Any]]] = None
    schedule: Optional[dict[str, Any]] = None
    stop: Optional[dict[str, Any]] = None
    checkpoints: Optional[list[PositiveInt]] = None
    init_jitter: Optional[Union[NonNegativeFloat, dict[str, Any]]] = None
    trace_path: OptionalPathStr = None

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
        raise ValueError(str(exc)) from exc

    ref = canonical_component_ref(raw_ref, cls)
    schema_cls: Type[StrictConfigModel] = getattr(cls, "Config", EmptyComponentConfig)
    try:
        validated = schema_cls.model_validate(dict(raw_params))
    except ValidationError as err:
        formatted = format_validation_error(err, prefix=f"model.components[{index}] ({ref})")
        raise ValueError(str(formatted)) from err

    return ComponentSpec(ref=ref, cls=cls, config=validated)


class ModelConfig(StrictConfigModel):
    """Forward model precision and ordered component list."""

    precision: Literal["single", "double"] = "single"
    components: list[ComponentSpec] = Field(default_factory=list)

    @field_validator("components", mode="before")
    @classmethod
    def _parse_components(cls, value: Any) -> list[ComponentSpec]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(f"model.components must be a list, got {type(value).__name__}")
        return [_parse_component_entry(entry, i) for i, entry in enumerate(value)]

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
        # 1. At most one orbit-source trajectory component.
        orbit_matches = self.find_all(OrbitSourceConfig)
        if len(orbit_matches) > 1:
            refs = [spec.ref for spec, _ in orbit_matches]
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
            keys = {(c.time_int_factor, c.min_time_bins, c.max_time_bins) for _, c in fine_matches}
            if len(keys) > 1:
                raise ValueError(
                    "Fine-grid RFI visibility components disagree on time integration settings."
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

    model: ModelConfig
    data: DataConfig = Field(default_factory=DataConfig)
    plots: PlotsConfig = Field(default_factory=PlotsConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    opt: OptConfig = Field(default_factory=OptConfig)

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
    # Strip top-level bare section headers (e.g. `data:` with nothing under it) ONLY for
    # known optional sections that have defaults, while forbidding unknown sections.
    cleaned = dict(raw)
    for section in ("data", "plots", "inference", "opt"):
        if section in cleaned and cleaned[section] is None:
            cleaned[section] = {}
    try:
        return RunConfig.model_validate(cleaned)
    except ValidationError as err:
        raise format_validation_error(err) from None
