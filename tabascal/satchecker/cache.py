"""Deterministic catalogue-cache policy and text-backed storage for SatChecker.

This module owns three things and nothing else:

1. the *deterministic cache key* — :func:`canonical_epoch_jd` snaps an arbitrary
   observation epoch to the midpoint of a fixed, globally-anchored UTC bucket, so
   the catalogue reused for a request depends only on the request and the bucket
   width, never on what happens to already be cached;
2. a small :class:`CatalogueCache` interface (get/store a snapshot, get/store the
   per-snapshot fallback records) that a future SQLite backend can implement
   without changing callers;
3. :class:`TextCatalogueCache`, a human-readable JSON implementation using a
   versioned row-oriented envelope, written atomically.

It knows nothing about orbit propagation, TLE-element parsing, JAX, or casacore.
It reads legacy pandas-oriented ``<YYYY-MM-DD>-*.json`` files only as *input*
(:func:`read_legacy_tle_records`); new managed snapshots never use that format.

Bucket policy (a deliberate approximation, documented for users): with the
default two-hour bucket the catalogue epoch differs from the requested epoch by
at most one hour, so a cached record is nearest to the *canonical bucket epoch*,
not necessarily nearest to the exact observation. Two observations less than two
hours apart can still straddle a boundary and use different snapshots.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from glob import glob
from numbers import Integral
from pathlib import Path
from typing import Optional

import pandas as pd

from tabascal.time import jd_to_datetime


SCHEMA_VERSION = 1

# Default width (hours) of the fixed UTC catalogue-reuse bucket. The single
# source of truth — config, orchestration and components import this rather
# than repeating the literal.
DEFAULT_CATALOGUE_INTERVAL_HOURS = 2.0

# Columns a normalised TLE record must expose to be a usable cache row.
_REQUIRED_COLUMNS = ("NORAD_CAT_ID", "TLE_LINE1", "TLE_LINE2")

# Completeness a full snapshot must satisfy: actual >= expected * fraction. Mirrors
# the client's zip-acceptance policy (kept here so the cache validates independently
# and the two modules stay decoupled).
CATALOGUE_MIN_FRACTION = 0.99

# A stored ``catalogue_epoch_jd`` must match the requested canonical epoch to within
# this many days. Canonical epochs of distinct buckets are >= 1 s apart (see
# ``_MIN_INTERVAL_US``), so ~86 ms of slack for the float round-trip cannot admit a
# neighbouring bucket while still catching a genuinely wrong embedded epoch.
_EPOCH_MATCH_TOL_DAYS = 1e-6

_UNIX_EPOCH_JD = 2440587.5
_DAY_US = 86_400 * 1_000_000  # microseconds per day
# Smallest supported bucket step. At >= 1 s the second-rounded canonical stamps of
# distinct buckets can never collide (their midpoints are always >= 1 s apart).
_MIN_INTERVAL_US = 1_000_000


# ---------------------------------------------------------------------------
# Deterministic cache key: fixed, globally-anchored UTC buckets
# ---------------------------------------------------------------------------

def _interval_us(interval_hours: float) -> int:
    us = int(round(float(interval_hours) * 3600 * 1_000_000))
    if us < _MIN_INTERVAL_US:
        raise ValueError(
            f"tle_catalogue_interval_hours must be at least "
            f"{_MIN_INTERVAL_US / 3.6e9:g} h (1 second), got {interval_hours!r}"
        )
    return us


def canonical_epoch_jd(
    epoch_jd: float, interval_hours: float = DEFAULT_CATALOGUE_INTERVAL_HOURS
) -> float:
    """Snap *epoch_jd* to the midpoint of its fixed UTC bucket.

    Buckets are anchored at the Unix epoch and are *interval_hours* wide. The
    calculation is done in integer microseconds since 1970-01-01T00:00:00Z, not
    in floating-point Julian dates, so the mapping is exact and reproducible.

    With ``interval_hours=2`` a request at 11:23 UTC and one at 10:05 UTC both map
    to the 11:00 UTC catalogue epoch; a request at 12:15 maps to 13:00. The
    returned catalogue epoch is at most ``interval_hours / 2`` hours from the
    requested epoch.
    """
    step = _interval_us(interval_hours)
    unix_us = round((float(epoch_jd) - _UNIX_EPOCH_JD) * _DAY_US)
    bucket = unix_us // step  # floor division: correct for the bucket boundary
    midpoint_us = bucket * step + step // 2
    return _UNIX_EPOCH_JD + midpoint_us / _DAY_US


def canonical_stamp(catalogue_epoch_jd: float) -> str:
    """Filename-safe UTC timestamp (``YYYYMMDDThhmmssZ``) for a catalogue epoch.

    Rounded to the nearest second: the canonical epoch is exact in integer
    microseconds, but the float Julian-date round-trip can land a few
    microseconds short of a whole second (e.g. 12:59:59.999999 for a 13:00:00
    bucket midpoint). Rounding keeps the filename stable and deterministic.
    """
    dt = jd_to_datetime(catalogue_epoch_jd)
    if dt.microsecond:
        dt = dt.replace(microsecond=0)
        if jd_to_datetime(catalogue_epoch_jd).microsecond >= 500_000:
            dt += timedelta(seconds=1)
    return dt.strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# Snapshot value object + cache interface
# ---------------------------------------------------------------------------

@dataclass
class CatalogueSnapshot:
    """A validated full catalogue at one canonical epoch, plus provenance.

    ``records`` is an in-memory pandas frame for convenience; the *stored* form is
    a row-oriented JSON list, so no pandas column-orientation detail leaks through
    the cache interface (a future SQLite backend can store rows directly).
    """

    catalogue_epoch_jd: float
    records: pd.DataFrame
    requested_epoch_jd: Optional[float] = None
    fetched_at: Optional[str] = None
    expected_count: Optional[int] = None
    actual_count: Optional[int] = None
    service_version: Optional[str] = None
    schema_version: int = SCHEMA_VERSION


class CatalogueCache(ABC):
    """Storage interface for canonical catalogue snapshots.

    Deliberately narrow so a future indexed/SQLite backend can replace the text
    implementation without touching the orchestration in :mod:`tabascal.tle`.
    Snapshots are keyed by canonical catalogue epoch (see :func:`canonical_epoch_jd`).
    ``*_extra`` handles the per-satellite fallback records fetched for IDs missing
    from a bulk snapshot; they are associated with the same canonical epoch so a
    later run over the same request reuses them.
    """

    @abstractmethod
    def get_snapshot(self, catalogue_epoch_jd: float) -> Optional[CatalogueSnapshot]:
        """Return the stored snapshot for the epoch, or ``None`` on a miss/invalid file."""

    @abstractmethod
    def store_snapshot(self, snapshot: CatalogueSnapshot) -> None:
        """Persist a validated snapshot atomically."""

    @abstractmethod
    def get_extra(self, catalogue_epoch_jd: float) -> pd.DataFrame:
        """Return per-satellite fallback records for the epoch (may be empty)."""

    @abstractmethod
    def store_extra(self, catalogue_epoch_jd: float, records: pd.DataFrame) -> None:
        """Merge fallback records into the epoch's fallback store atomically."""


# ---------------------------------------------------------------------------
# Text (JSON) implementation
# ---------------------------------------------------------------------------

def _has_required_columns(df: pd.DataFrame) -> bool:
    return all(col in df.columns for col in _REQUIRED_COLUMNS)


class CacheValidationError(ValueError):
    """Raised when a cache envelope is structurally or semantically invalid."""


def _required_count(env: dict, field: str) -> int:
    """Return a required non-negative JSON integer count, or raise validation error."""
    value = env.get(field)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise CacheValidationError(f"{field} must be a non-negative integer, got {value!r}")
    value = int(value)
    if value < 0:
        raise CacheValidationError(f"{field} must be non-negative, got {value}")
    return value


def _validate_envelope(
    env, expected_epoch_jd: Optional[float], *, full_snapshot: bool
) -> pd.DataFrame:
    """Validate a cache envelope and return its records frame, or raise.

    Shared by the read and write paths so a snapshot can never be *stored* in a
    shape that would later be rejected on *load*. Checks common to snapshots and
    fallback (``-extra``) envelopes:

    * ``schema_version`` equals the supported :data:`SCHEMA_VERSION` (an
      unsupported/future schema is rejected, not silently read as v1);
    * ``catalogue_epoch_jd`` is finite and matches *expected_epoch_jd* (when given)
      within :data:`_EPOCH_MATCH_TOL_DAYS`, so a valid document under the wrong
      filename is not accepted;
    * the required TLE columns are present with no null values and numeric NORAD IDs.

    Full snapshots additionally require ``actual_count == len(records)`` and, when
    ``expected_count`` is known, ``actual >= expected * CATALOGUE_MIN_FRACTION``.
    Fallback envelopes carry multiple records for different IDs, so those two
    count rules do not apply to them.
    """
    if not isinstance(env, dict):
        raise CacheValidationError("cache envelope is not a JSON object")
    if env.get("schema_version") != SCHEMA_VERSION:
        raise CacheValidationError(
            f"unsupported schema_version {env.get('schema_version')!r} "
            f"(supported: {SCHEMA_VERSION})"
        )

    stored_epoch = env.get("catalogue_epoch_jd")
    if expected_epoch_jd is not None:
        if not isinstance(stored_epoch, (int, float)) or not math.isfinite(float(stored_epoch)):
            raise CacheValidationError("catalogue_epoch_jd is missing or non-finite")
        if abs(float(stored_epoch) - float(expected_epoch_jd)) > _EPOCH_MATCH_TOL_DAYS:
            raise CacheValidationError(
                f"catalogue_epoch_jd {stored_epoch} does not match requested "
                f"{expected_epoch_jd}"
            )

    records = pd.DataFrame(env.get("records") or [])
    if not len(records) or not _has_required_columns(records):
        raise CacheValidationError("cache envelope has no usable records")
    for col in _REQUIRED_COLUMNS:
        if records[col].isnull().any():
            raise CacheValidationError(f"cache envelope has null values in {col}")
    try:
        pd.to_numeric(records["NORAD_CAT_ID"])
    except (ValueError, TypeError) as e:
        raise CacheValidationError(f"NORAD_CAT_ID is not numeric: {e}") from e

    if full_snapshot:
        actual = _required_count(env, "actual_count")
        if actual != len(records):
            raise CacheValidationError(
                f"actual_count {actual} != {len(records)} stored records"
            )
        expected = _required_count(env, "expected_count")
        if expected == 0:
            raise CacheValidationError("expected_count cannot be zero for a non-empty snapshot")
        if len(records) < expected * CATALOGUE_MIN_FRACTION:
            raise CacheValidationError(
                f"snapshot incomplete: {len(records)} of {expected} records "
                f"(< {CATALOGUE_MIN_FRACTION:.0%})"
            )
    return records


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write *payload* as JSON to *path* atomically (temp file + ``os.replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class TextCatalogueCache(CatalogueCache):
    """JSON-file catalogue cache under a single directory.

    Snapshot files are ``catalogue-<stamp>.json`` and fallback files are
    ``catalogue-<stamp>-extra.json`` where ``<stamp>`` is the canonical UTC epoch.
    Both use a versioned, row-oriented envelope. A file that is missing, partial,
    unparseable, or lacks the required columns is treated as a miss and will be
    overwritten — a partial write can never look like a valid cache hit.
    """

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)

    # -- paths --
    def _snapshot_path(self, catalogue_epoch_jd: float) -> Path:
        return self.cache_dir / f"catalogue-{canonical_stamp(catalogue_epoch_jd)}.json"

    def _extra_path(self, catalogue_epoch_jd: float) -> Path:
        return self.cache_dir / f"catalogue-{canonical_stamp(catalogue_epoch_jd)}-extra.json"

    # -- snapshots --
    def get_snapshot(self, catalogue_epoch_jd: float) -> Optional[CatalogueSnapshot]:
        path = self._snapshot_path(catalogue_epoch_jd)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                env = json.load(f)
        except (OSError, ValueError):
            return None
        try:
            records = _validate_envelope(env, catalogue_epoch_jd, full_snapshot=True)
        except CacheValidationError:
            return None  # invalid/incomplete/wrong-epoch → cache miss, fetch afresh
        return CatalogueSnapshot(
            catalogue_epoch_jd=env.get("catalogue_epoch_jd", catalogue_epoch_jd),
            records=records,
            requested_epoch_jd=env.get("requested_epoch_jd"),
            fetched_at=env.get("fetched_at"),
            expected_count=env.get("expected_count"),
            actual_count=env.get("actual_count"),
            service_version=env.get("service_version"),
            schema_version=env.get("schema_version", SCHEMA_VERSION),
        )

    def store_snapshot(self, snapshot: CatalogueSnapshot) -> None:
        actual = (
            snapshot.actual_count
            if snapshot.actual_count is not None
            else len(snapshot.records)
        )
        expected = (
            snapshot.expected_count
            if snapshot.expected_count is not None
            else actual
        )
        env = {
            "schema_version": snapshot.schema_version,
            "requested_epoch_jd": snapshot.requested_epoch_jd,
            "catalogue_epoch_jd": snapshot.catalogue_epoch_jd,
            "fetched_at": snapshot.fetched_at or _utc_now_iso(),
            "expected_count": expected,
            "actual_count": actual,
            "service_version": snapshot.service_version,
            "records": snapshot.records.to_dict(orient="records"),
        }
        # Validate before writing so a caller cannot persist a snapshot that would
        # later be rejected on load (raises before any file is created).
        _validate_envelope(env, snapshot.catalogue_epoch_jd, full_snapshot=True)
        # Normalise Integral subclasses before passing the envelope to json.dump.
        env["actual_count"] = int(env["actual_count"])
        env["expected_count"] = int(env["expected_count"])
        _atomic_write_json(self._snapshot_path(snapshot.catalogue_epoch_jd), env)

    # -- fallback records --
    def get_extra(self, catalogue_epoch_jd: float) -> pd.DataFrame:
        path = self._extra_path(catalogue_epoch_jd)
        if not path.exists():
            return pd.DataFrame()
        try:
            with open(path) as f:
                env = json.load(f)
        except (OSError, ValueError):
            return pd.DataFrame()
        try:
            return _validate_envelope(env, catalogue_epoch_jd, full_snapshot=False)
        except CacheValidationError:
            return pd.DataFrame()

    def store_extra(self, catalogue_epoch_jd: float, records: pd.DataFrame) -> None:
        """Merge *records* into the epoch's fallback store atomically.

        Note: this is a read-merge-write cycle. Atomic replacement guarantees no
        partial file is ever observed, but it does *not* serialise two processes
        merging concurrently — the later writer can overwrite the earlier one's new
        records. Cross-process locking is intentionally deferred to a future
        indexed/SQLite backend; single-process runs are unaffected.
        """
        if not len(records):
            return
        merged = records
        existing = self.get_extra(catalogue_epoch_jd)
        if len(existing):
            merged = pd.concat([existing, records], ignore_index=True)
        merged = merged.drop_duplicates(
            subset=["NORAD_CAT_ID", "TLE_LINE1", "TLE_LINE2"]
        )
        env = {
            "schema_version": SCHEMA_VERSION,
            "catalogue_epoch_jd": catalogue_epoch_jd,
            "fetched_at": _utc_now_iso(),
            "records": merged.to_dict(orient="records"),
        }
        _validate_envelope(env, catalogue_epoch_jd, full_snapshot=False)
        _atomic_write_json(self._extra_path(catalogue_epoch_jd), env)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Legacy local-file reader (explicit user data, e.g. extra_tle_dir fixtures)
# ---------------------------------------------------------------------------

def read_legacy_tle_records(directory) -> pd.DataFrame:
    """Read every ``*.json`` TLE file in *directory* into one normalised frame.

    Supports the legacy pandas-oriented ``<YYYY-MM-DD>-*.json`` Space-Track cache
    format (and the bundled test fixtures). Files that do not parse, or that lack
    the required ``NORAD_CAT_ID``/``TLE_LINE1``/``TLE_LINE2`` columns (e.g. a new
    managed-snapshot envelope), are skipped. Only the identity and TLE-line
    columns are retained; orbital elements are parsed locally downstream so any
    stale element columns in the file are ignored.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return pd.DataFrame()
    frames = []
    for path in sorted(glob(str(directory / "*.json"))):
        try:
            df = pd.read_json(path)
        except (ValueError, OSError):
            continue
        if not _has_required_columns(df):
            continue
        keep = [c for c in ("NORAD_CAT_ID", "OBJECT_NAME", "TLE_LINE1", "TLE_LINE2") if c in df.columns]
        frames.append(df[keep].copy())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
