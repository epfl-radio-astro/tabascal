"""One normalisation path for the ``satellites`` TLE configuration.

Both the preflight check and the actual resolution build their inputs here, so a
configuration can never be interpreted two different ways, and every malformed
value surfaces as :class:`TLEConfigurationError` — which the CLI renders as a
one-line message instead of a traceback.

The module also owns :func:`ms_observation_epoch_jd`, the single Measurement Set
observation-epoch derivation. Preflight and execution must agree exactly on the
epoch: it sets every TLE age comparison, so two slightly different means could
make different acceptance decisions. The helper mirrors
:func:`tabascal.tab_tools.read_ms` — one time
per integration, with the same seconds-versus-days unit guard.

Three age settings exist and are deliberately kept distinct:

``extra_tle_max_age_days``
    Acceptance of explicit user/replay files (``extra_tle_dir``). ``null`` by
    default so exact ``used_tles_*.json`` replay is always possible; remote
    service policy must never constrain it.
``remote_tle_max_age_days``
    Emergency acceptance ceiling for SatChecker records and its managed cache.
``tle_cache_reuse_max_age_days``
    Age below which a cached SatChecker record avoids a new nearest-TLE request.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Optional

import numpy as np

from tabascal.satchecker import SatCheckerError as TLEError
from tabascal.time import mjd_to_jd


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Provisional hard ceiling on the age of a TLE accepted from SatChecker or its
#: managed cache. Substantially below the seven days an earlier proposal used and
#: inside the range the catalogue investigation actually evaluated. It is an
#: emergency backstop against obviously unsuitable remote records, *not* a promise
#: of three-day positional accuracy — see issue #101 for the calibrated,
#: observation-specific replacement.
DEFAULT_REMOTE_TLE_MAX_AGE_DAYS = 3.0

#: A cached record this close to the observation is good enough to avoid a new
#: nearest-TLE request. This is a request/latency trade-off, not the hard safety
#: ceiling below.
DEFAULT_TLE_CACHE_REUSE_MAX_AGE_DAYS = 1.0

#: Trajectory components that consume resolved TLEs, so an empty NORAD ID list is
#: a configuration error rather than a satellite-free model.
_TLE_TRAJECTORY_COMPONENTS = frozenset(
    {"FixedOrbit", "SGP4LEOOrbit", "SGP4LEONoDragOrbit", "KeplerOrbit"}
)


class TLEConfigurationError(TLEError, ValueError):
    """A TLE-related configuration value is missing, malformed or out of range.

    Subclasses :class:`~tabascal.satchecker.client.SatCheckerError` (aliased as
    ``tabascal.tle.TLEError``) so the CLI's existing handler prints it without a
    traceback, and :class:`ValueError` because that is what a bad argument to
    these helpers has always raised.
    """


# ---------------------------------------------------------------------------
# Scalar validation
# ---------------------------------------------------------------------------

def _as_finite_float(value, name: str) -> float:
    """Coerce *value* to a finite float or raise :class:`TLEConfigurationError`."""
    if isinstance(value, bool) or not isinstance(value, (Real, str)):
        raise TLEConfigurationError(
            f"{name} must be a number, got {value!r}"
        )
    try:
        out = float(value)
    except (TypeError, ValueError) as e:
        raise TLEConfigurationError(f"{name} must be a number, got {value!r}") from e
    if not math.isfinite(out):
        raise TLEConfigurationError(
            f"{name} must be finite, got {value!r}"
        )
    return out


def validate_age_days(value, name: str) -> Optional[float]:
    """Validate an age limit in days: ``None`` (no limit) or a non-negative number.

    ``None`` is an explicit expert opt-out. Negative, non-numeric, NaN and
    infinite values are configuration errors, never silently coerced.
    """
    if value is None:
        return None
    out = _as_finite_float(value, name)
    if out < 0:
        raise TLEConfigurationError(
            f"{name} must be null or a non-negative number of days, got {value!r}"
        )
    return out


# ---------------------------------------------------------------------------
# NORAD ID validation
# ---------------------------------------------------------------------------

def _as_norad_id(value, where: str) -> int:
    """Coerce one entry to a positive integral NORAD catalogue ID."""
    if isinstance(value, bool):
        raise TLEConfigurationError(f"{where}: {value!r} is not a NORAD catalogue ID")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise TLEConfigurationError(f"{where}: empty NORAD catalogue ID")
        try:
            value = int(text)
        except ValueError as e:
            raise TLEConfigurationError(
                f"{where}: {value!r} is not a NORAD catalogue ID"
            ) from e
    if isinstance(value, Integral):
        out = int(value)
    elif isinstance(value, Real):
        as_float = float(value)
        # Reject NaN/inf and fractional values before int(): a fractional ID would
        # truncate to a *different* satellite and an infinity raises inside numpy.
        if not math.isfinite(as_float) or as_float != round(as_float):
            raise TLEConfigurationError(
                f"{where}: {value!r} is not a finite integer NORAD catalogue ID"
            )
        out = int(round(as_float))
    else:
        raise TLEConfigurationError(f"{where}: {value!r} is not a NORAD catalogue ID")
    if out <= 0:
        raise TLEConfigurationError(
            f"{where}: NORAD catalogue IDs must be positive, got {out}"
        )
    return out


def normalise_norad_ids(values, source: str = "satellites.norad_ids") -> list[int]:
    """Return validated, order-preserving, de-duplicated NORAD IDs.

    ``None`` normalises to an empty list — a satellite-free model is a legitimate
    configuration, and the caller decides whether the *selected model* makes that
    an error (see :func:`model_requires_tles`). Every other malformed value raises
    :class:`TLEConfigurationError` before any NumPy/Pandas conversion, so a
    fractional or non-numeric entry can never reach the resolver.
    """
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        raise TLEConfigurationError(
            f"{source} must be a list of NORAD catalogue IDs, got {values!r}"
        )
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if not isinstance(values, Sequence):
        try:
            values = list(values)
        except TypeError as e:
            raise TLEConfigurationError(
                f"{source} must be a list of NORAD catalogue IDs, got {values!r}"
            ) from e

    out: list[int] = []
    seen: set[int] = set()
    for i, value in enumerate(values):
        nid = _as_norad_id(value, f"{source}[{i}]")
        if nid not in seen:
            seen.add(nid)
            out.append(nid)
    return out


def read_norad_ids_file(path) -> list[int]:
    """Read one NORAD catalogue ID per line from *path*.

    Blank lines and ``#`` comments are ignored; anything else must be a single
    positive integer. Errors name the offending file *and line number* so a typo
    in a long list is trivially located. IDs are de-duplicated with their first
    occurrence's order preserved.
    """
    path = Path(path)
    try:
        text = path.read_text()
    except OSError as e:
        raise TLEConfigurationError(f"NORAD ID file could not be read ({path}): {e}") from e

    out: list[int] = []
    seen: set[int] = set()
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        nid = _as_norad_id(line, f"{path}:{lineno}")
        if nid not in seen:
            seen.add(nid)
            out.append(nid)
    return out


# ---------------------------------------------------------------------------
# Model introspection
# ---------------------------------------------------------------------------

def model_requires_tles(config: dict) -> bool:
    """True when the configured model includes a TLE-consuming trajectory component."""
    components = (config.get("model") or {}).get("components") or []
    for component in components:
        name = str(component).rsplit(":", 1)[-1].strip()
        if name in _TLE_TRAJECTORY_COMPONENTS:
            return True
    return False


# ---------------------------------------------------------------------------
# Normalised configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TLEConfig:
    """Fully validated TLE configuration shared by preflight and resolution."""

    norad_ids: list[int] = field(default_factory=list)
    extra_tle_dir: Optional[str] = None
    extra_tle_max_age_days: Optional[float] = None
    remote_tle_max_age_days: Optional[float] = DEFAULT_REMOTE_TLE_MAX_AGE_DAYS
    cache_reuse_max_age_days: Optional[float] = DEFAULT_TLE_CACHE_REUSE_MAX_AGE_DAYS


def normalise_tle_config(
    config: dict,
    norad_ids_path_override: Optional[str] = None,
    require_ids: Optional[bool] = None,
) -> TLEConfig:
    """Validate the ``satellites`` section of *config* into a :class:`TLEConfig`.

    NORAD IDs come from ``satellites.norad_ids_path`` when set and from
    ``satellites.norad_ids`` otherwise, with *norad_ids_path_override* (the CLI's
    ``-np/--norad-path``) taking precedence over both. ``require_ids`` defaults to
    :func:`model_requires_tles`, so a model with a TLE-based trajectory component
    but no satellites is reported as a configuration error rather than failing
    later with an empty frame.
    """
    satellites = config.get("satellites") or {}

    ids_path = norad_ids_path_override or satellites.get("norad_ids_path")
    if ids_path:
        norad_ids = read_norad_ids_file(ids_path)
        source = str(ids_path)
    else:
        norad_ids = normalise_norad_ids(satellites.get("norad_ids"))
        source = "satellites.norad_ids"

    if require_ids is None:
        require_ids = model_requires_tles(config)
    if require_ids and not norad_ids:
        raise TLEConfigurationError(
            "The configured model includes a TLE-based trajectory component, but "
            f"no NORAD catalogue IDs were given ({source} is empty or null). Add "
            "the satellites to model, or select a model without a TLE trajectory "
            "component."
        )

    extra_tle_dir = satellites.get("extra_tle_dir") or None

    remote_max_age = validate_age_days(
        satellites.get("remote_tle_max_age_days", DEFAULT_REMOTE_TLE_MAX_AGE_DAYS),
        "remote_tle_max_age_days",
    )
    cache_reuse_age = validate_age_days(
        satellites.get(
            "tle_cache_reuse_max_age_days", DEFAULT_TLE_CACHE_REUSE_MAX_AGE_DAYS
        ),
        "tle_cache_reuse_max_age_days",
    )
    if (
        cache_reuse_age is not None
        and remote_max_age is not None
        and cache_reuse_age > remote_max_age
    ):
        raise TLEConfigurationError(
            f"tle_cache_reuse_max_age_days ({cache_reuse_age:g}) must not exceed "
            f"remote_tle_max_age_days ({remote_max_age:g})"
        )

    return TLEConfig(
        norad_ids=norad_ids,
        extra_tle_dir=str(extra_tle_dir) if extra_tle_dir else None,
        extra_tle_max_age_days=validate_age_days(
            satellites.get("extra_tle_max_age_days"), "extra_tle_max_age_days"
        ),
        remote_tle_max_age_days=remote_max_age,
        cache_reuse_max_age_days=cache_reuse_age,
    )


# ---------------------------------------------------------------------------
# Measurement Set observation epoch
# ---------------------------------------------------------------------------

def _ms_time_column(ms_path: str) -> np.ndarray:
    """Raw ``TIME`` column of *ms_path*.

    Isolated so the epoch helper (and everything built on it) can be exercised
    offline by patching this one casacore-touching seam.
    """
    from casacore.tables import table as _ms_table

    with _ms_table(str(ms_path), readonly=True, ack=False) as t:
        return np.asarray(t.getcol("TIME"), dtype=float)


def ms_integration_times_mjd(ms_path: str) -> np.ndarray:
    """One observation time per integration, in MJD days.

    Mirrors :func:`tabascal.tab_tools.read_ms`, which takes a single timestamp per
    integration (not one per visibility row) and divides by 86400 when consecutive
    timestamps are more than 0.5 apart — the signature of a ``TIME`` column stored
    in seconds rather than days. The unique-times form used here equals
    ``TIME.reshape(n_time, n_bl)[:, 0]`` for a well-formed MS while remaining
    correct when the row count is not an exact multiple of the baseline count.

    Negative (pre-1858) and pre-1970 epochs are supported: the guard reads the
    *spacing* of consecutive samples, which is positive regardless of sign. A
    single-integration MS has no spacing to read, so its unit is inferred from
    magnitude instead — an MJD day number is at most ~1e5 in any plausible
    observing era, while the same instant in seconds is ~1e9.
    """
    times = np.unique(_ms_time_column(ms_path))
    if times.size == 0:
        raise TLEError(f"Measurement Set has an empty TIME column: {ms_path}")
    if times.size > 1:
        in_seconds = (times[1] - times[0]) > 0.5
    else:
        in_seconds = abs(times[0]) > 1e5
    return times / 86400.0 if in_seconds else times


def ms_observation_epoch_jd(ms_path: str) -> float:
    """Mean observation epoch of *ms_path* as a UTC Julian Date.

    This exact value is what the canonical cache bucket and every TLE age check
    are computed from, in both preflight and execution.
    """
    return float(mjd_to_jd(ms_integration_times_mjd(ms_path).mean()))


def observation_epoch_jd(times_jd) -> float:
    """Mean observation epoch (UTC JD) of an already-read time array.

    Execution reaches the epoch through the times :func:`tabascal.tab_tools.read_ms`
    returned rather than by re-reading the MS; going through this one helper keeps
    the reduction identical to :func:`ms_observation_epoch_jd`.
    """
    return float(np.atleast_1d(np.asarray(times_jd, dtype=float)).mean())
