"""One normalisation path for the ``satellites`` orbit configuration.

Both the preflight check and the actual resolution build their inputs here, so a
configuration can never be interpreted two different ways, and every malformed
value surfaces as :class:`TLEConfigurationError` — which the CLI renders as a
one-line message instead of a traceback.

The module also owns :func:`ms_observation_epoch_jd`, the single Measurement Set
observation-epoch derivation. Preflight and execution must agree exactly on the
epoch: it sets every TLE age comparison, so two slightly different means could
make different acceptance decisions. The helper mirrors
:func:`tabascal.ms.read_ms` — one time per integration, converted through the
same :func:`tabascal.ms.times_to_mjd`, and normalised onto UTC from the same
declared time scale, so the epoch is the instant the observation happened rather
than whatever number the ``TIME`` column happens to hold.

Three age settings exist and are deliberately kept distinct:

``extra_orbit_max_age_days``
    Acceptance of explicit user/replay files (``extra_orbit_dir``). ``null`` by
    default so exact ``used_orbits_*.json`` replay is always possible; remote
    service policy must never constrain it.
``remote_max_age_days``
    Emergency acceptance ceiling for SatChecker records and its managed cache.
``cache_reuse_max_age_days``
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

from satchecker_client import SatCheckerError as TLEError
from tabascal.ms import (
    infer_time_unit,
    read_time_scale,
    read_time_unit,
    times_to_mjd,
)
from tabascal.time import mjd_to_jd, to_utc_jd


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Provisional hard ceiling on the age of a TLE accepted from SatChecker or its
#: managed cache. Substantially below the seven days an earlier proposal used and
#: inside the range the catalogue investigation actually evaluated. It is an
#: emergency backstop against obviously unsuitable remote records, *not* a promise
#: of three-day positional accuracy — see issue #101 for the calibrated,
#: observation-specific replacement.
DEFAULT_REMOTE_MAX_AGE_DAYS = 3.0

#: A cached record this close to the observation is good enough to avoid a new
#: nearest-TLE request. This is a request/latency trade-off, not the hard safety
#: ceiling below.
DEFAULT_CACHE_REUSE_MAX_AGE_DAYS = 1.0

#: Trajectory components that consume resolved TLEs, so an empty NORAD ID list is
#: a configuration error rather than a satellite-free model.
_TLE_TRAJECTORY_COMPONENTS = frozenset({"FixedOrbit", "Orbit", "NoDragOrbit"})


class TLEConfigurationError(TLEError, ValueError):
    """A TLE-related configuration value is missing, malformed or out of range.

    Subclasses :class:`~satchecker_client.client.SatCheckerError` (aliased as
    ``tabascal.orbit.TLEError``) so the CLI's existing handler prints it without a
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
        # import_components accepts both "module:Class" and "module.Class", so
        # both have to reach the guard — a dotted reference that slipped past it
        # would build a silently satellite-free model instead of erroring.
        name = str(component).replace(":", ".").rsplit(".", 1)[-1].strip()
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
    extra_orbit_dir: Optional[str] = None
    extra_orbit_max_age_days: Optional[float] = None
    remote_max_age_days: Optional[float] = DEFAULT_REMOTE_MAX_AGE_DAYS
    cache_reuse_max_age_days: Optional[float] = DEFAULT_CACHE_REUSE_MAX_AGE_DAYS


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

    extra_orbit_dir = satellites.get("extra_orbit_dir") or None

    remote_max_age = validate_age_days(
        satellites.get("remote_max_age_days", DEFAULT_REMOTE_MAX_AGE_DAYS),
        "remote_max_age_days",
    )
    cache_reuse_age = validate_age_days(
        satellites.get(
            "cache_reuse_max_age_days", DEFAULT_CACHE_REUSE_MAX_AGE_DAYS
        ),
        "cache_reuse_max_age_days",
    )
    if (
        cache_reuse_age is not None
        and remote_max_age is not None
        and cache_reuse_age > remote_max_age
    ):
        raise TLEConfigurationError(
            f"cache_reuse_max_age_days ({cache_reuse_age:g}) must not exceed "
            f"remote_max_age_days ({remote_max_age:g})"
        )

    return TLEConfig(
        norad_ids=norad_ids,
        extra_orbit_dir=str(extra_orbit_dir) if extra_orbit_dir else None,
        extra_orbit_max_age_days=validate_age_days(
            satellites.get("extra_orbit_max_age_days"), "extra_orbit_max_age_days"
        ),
        remote_max_age_days=remote_max_age,
        cache_reuse_max_age_days=cache_reuse_age,
    )


# ---------------------------------------------------------------------------
# Measurement Set observation epoch
# ---------------------------------------------------------------------------

def _ms_times_and_scale(ms_path: str) -> tuple:
    """Raw ``TIME`` column of *ms_path*, and the scale and unit it declares.

    All three come out of one casacore open, so preflight still reaches the
    Measurement Set through a single seam — the one tests patch to run offline.
    The scale is read rather than assumed for the same reason
    :func:`tabascal.ms.read_ms` reads it: an epoch on the wrong scale is a wrong
    instant, not a rounding difference.

    The unit is read for the same reason again, and the two keywords come out of
    the one ``getcolkeywords`` call. It used to be deliberately left unread here
    so that this path and ``read_ms`` would share the heuristic and could not
    classify one MS two ways -- but ``read_ms`` honours a declaration, so not
    reading it was what created that divergence rather than what closed it. An
    MS declaring seconds while storing day numbers was read on the declaration
    by the run and on the magnitudes by the TLE age checks.

    ``None`` where the MS declares nothing usable, which is the case the
    heuristic exists for.
    """
    from casacore.tables import table as _ms_table

    with _ms_table(str(ms_path), readonly=True, ack=False) as t:
        times = np.asarray(t.getcol("TIME"), dtype=float)
        keywords = {"TIME": t.getcolkeywords("TIME")}

    return times, read_time_scale(keywords), read_time_unit(keywords)


def _integration_times_mjd(
    times: np.ndarray, ms_path: str, unit: Optional[str] = None
) -> np.ndarray:
    """One time per integration, in MJD days, from an already-read column.

    A declared *unit* is honoured, as :func:`tabascal.ms.read_ms` honours one.
    Inferred, it is read from every row and the values from the distinct ones.
    The
    heuristic in :func:`tabascal.ms.infer_time_unit` weighs the times by how
    often they occur -- that is what lets a handful of unfilled rows be
    outvoted -- and deduplicating before it runs throws exactly that away: a
    column of one real timestamp per baseline plus stray rows at 0 and 1 is
    three distinct values whose median is 1, and the whole observation reads as
    days. Only the vote needs the duplicates, so only the vote gets them; this
    is the whole main table, and scaling it to reach the same answer would cost
    both the memory and the exactness of deduplicating the raw values.
    """

    times = np.asarray(times)
    if times.size == 0:
        raise TLEError(f"Measurement Set has an empty TIME column: {ms_path}")

    return times_to_mjd(np.unique(times), unit or infer_time_unit(times))


def ms_integration_times_mjd(ms_path: str) -> np.ndarray:
    """One observation time per integration, in MJD days, as the MS stores them.

    Mirrors :func:`tabascal.ms.read_ms`, which takes a single timestamp per
    integration (not one per visibility row), and converts through the same
    :func:`tabascal.ms.times_to_mjd` so the two cannot read one MS on two
    different units. The unique-times form used here equals
    ``TIME.reshape(n_time, n_bl)[:, 0]`` for a well-formed MS while remaining
    correct when the row count is not an exact multiple of the baseline count.

    Left on the scale the MS declares, exactly as ``read_ms`` leaves
    ``times_mjd``: this reports the column, and only the *epoch* below —
    a physical instant — is moved onto UTC.

    The column's declared ``QuantumUnits`` are read here as ``read_ms`` reads
    them, so the declaration settles the unit for both and the shared heuristic
    is what is left for an MS that declares nothing. The heuristic is
    order-insensitive, so neither path can be swayed by the order an MS stores
    its timestep blocks in either.
    """
    times, _, unit = _ms_times_and_scale(ms_path)

    return _integration_times_mjd(times, ms_path, unit)


def ms_observation_epoch_jd(ms_path: str) -> float:
    """Mean observation epoch of *ms_path* as a UTC Julian Date.

    This exact value is what the canonical cache bucket and every TLE age check
    are computed from, in both preflight and execution — and it is an *instant*,
    so the declared scale is honoured here as it is in the reader. A TAI column
    read as UTC would put every age comparison and nearest-record decision 37 s
    from the instant the fit then propagates at.

    The times are normalised per sample and averaged afterwards, through the
    same :func:`observation_epoch_jd` execution applies to what ``read_ms``
    returned, so the two reductions are identical by construction — including
    across a leap second, where averaging first and shifting once would differ
    by up to half a second.
    """
    times, scale, unit = _ms_times_and_scale(ms_path)
    times_mjd = _integration_times_mjd(times, ms_path, unit)

    return observation_epoch_jd(to_utc_jd(mjd_to_jd(times_mjd), scale))


def observation_epoch_jd(times_jd) -> float:
    """Mean observation epoch (UTC JD) of an already-read time array.

    Execution reaches the epoch through the times :func:`tabascal.ms.read_ms`
    returned rather than by re-reading the MS; going through this one helper keeps
    the reduction identical to :func:`ms_observation_epoch_jd`.
    """
    return float(np.atleast_1d(np.asarray(times_jd, dtype=float)).mean())
