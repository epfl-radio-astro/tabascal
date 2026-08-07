"""TABASCAL TLE orchestration and local orbital-element parsing.

TLEs are sourced from the IAU CPS SatChecker service via
:mod:`tabascal.satchecker` — no account or credentials are required. This module
is the TABASCAL adapter: it resolves each requested NORAD ID against an ordered
set of sources, applies the configurable age policy, drives the deterministic
catalogue cache, and parses OMM-style orbital elements locally from the two TLE
lines. All filtering and element computation is done locally.

Source precedence is resolved **independently per NORAD ID**:

  1. ``extra_tle_dir`` — user-supplied local TLE files. The record whose TLE-line
     epoch is closest to the observation epoch is chosen; it is accepted only if
     within ``extra_tle_max_age_days`` (``None`` = unlimited). An accepted record
     wins outright — later sources are not consulted for that ID.
  2. Managed canonical catalogue — one deterministic snapshot per fixed UTC bucket
     (see :func:`tabascal.satchecker.cache.canonical_epoch_jd`), fetched from
     SatChecker on a miss and cached atomically.
  3. Per-satellite SatChecker fallback — for IDs still missing from the bulk
     snapshot, and for *all* remaining IDs when the service reports an empty
     catalogue at the epoch (its ``tles-at-epoch`` endpoint has a data horizon;
     recent observations can fall beyond it while ``get-nearest-tle`` still
     resolves). The records are associated with the same canonical snapshot so a
     later run over the same request reuses them.

Catalogue reuse follows the bucket policy: the cached record is nearest to the
canonical bucket epoch, not necessarily nearest to the exact observation epoch.
Cache contents cannot change the result for a fixed request and policy.
"""

from __future__ import annotations

import math
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from platformdirs import user_cache_path

import numpy as np
import pandas as pd

from tabascal import satchecker
from tabascal.satchecker import (
    DEFAULT_CATALOGUE_INTERVAL_HOURS,
    CatalogueSnapshot,
    TextCatalogueCache,
    canonical_epoch_jd,
    canonical_stamp,
    read_legacy_tle_records,
)
from tabascal.satchecker import SatCheckerError as TLEError  # noqa: F401  back-compat alias
from tabascal.time import datetime_to_jd, jd_to_datetime


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MU_KM3_S2 = 398600.4418  # Earth gravitational parameter, km^3/s^2
_THROTTLE_SECONDS = 1.0   # delay between per-satellite fallback requests
# DEFAULT_CATALOGUE_INTERVAL_HOURS is imported from tabascal.satchecker above —
# the bucket policy's single source of truth.

# A TLE line-1 epoch is quantised to ~1e-8 day (8 decimal places of a day, ~0.9 ms),
# and the datetime<->JD round-trip adds only sub-microsecond-day noise (measured
# ~3.7e-9 day). This tolerance covers one epoch quantum plus that slack (~2.6 ms), so
# ``extra_tle_max_age_days: 0`` accepts a record matching the observation to TLE
# precision while rejecting one several ms away — matching the documented semantics.
_AGE_TOL_DAYS = 3e-8


# ---------------------------------------------------------------------------
# TLE cache directory helpers
# ---------------------------------------------------------------------------

def tle_cache_dir() -> Path:
    """Return the managed TLE cache directory, creating it if needed.

    The directory is resolved in priority order:
    1. ``TLE_CACHE_DIR`` environment variable (if set).
    2. The platform user-cache directory (e.g. ``~/.cache/tle-cache`` on Linux,
       ``~/Library/Caches/tle-cache`` on macOS).
    """
    p = Path(os.environ.get("TLE_CACHE_DIR") or user_cache_path("tle-cache"))
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------

def _validate_max_age(extra_tle_max_age_days) -> Optional[float]:
    """Validate ``extra_tle_max_age_days``: ``None`` or a non-negative float."""
    if extra_tle_max_age_days is None:
        return None
    value = float(extra_tle_max_age_days)
    if value < 0 or math.isnan(value):
        raise ValueError(
            f"extra_tle_max_age_days must be null or a non-negative number, "
            f"got {extra_tle_max_age_days!r}"
        )
    return value


# ---------------------------------------------------------------------------
# Local TLE parsing (elements derived from the two TLE lines)
# ---------------------------------------------------------------------------

def _parse_exp_field(field: str) -> float:
    """Parse a TLE exponential field (e.g. '-11606-4' -> -0.11606e-4)."""
    s = field.strip()
    if not s or s in ("+00000-0", "00000-0", "00000+0"):
        return 0.0
    sign = 1.0
    if s[0] in "+-":
        sign = -1.0 if s[0] == "-" else 1.0
        s = s[1:]
    mantissa = s[:-2].replace(" ", "")
    exponent = int(s[-2:])
    if not mantissa:
        return 0.0
    return sign * float("0." + mantissa) * (10.0 ** exponent)


def _tle_epoch_jd(line1: str) -> float:
    """UTC Julian Date of a TLE epoch (line 1 columns 19-32)."""
    epoch_year = int(line1[18:20])
    epoch_day = float(line1[20:32])
    year = 2000 + epoch_year if epoch_year < 57 else 1900 + epoch_year
    dt = datetime(year, 1, 1) + timedelta(days=epoch_day - 1.0)
    return datetime_to_jd(dt)


def parse_tle_elements(line1: str, line2: str) -> dict:
    """Derive OMM-style orbital elements from a TLE pair.

    Angles are in degrees, mean motion in rev/day and the semi-major axis in km
    — matching the units Space-Track's OMM reported, so downstream consumers are
    unchanged. ``SEMIMAJOR_AXIS`` is computed from the mean motion via Kepler's
    third law (reproduces the Space-Track OMM value).
    """
    inclination = float(line2[8:16])
    raan = float(line2[17:25])
    eccentricity = float("0." + line2[26:33].strip())
    arg_pericenter = float(line2[34:42])
    mean_anomaly = float(line2[43:51])
    mean_motion = float(line2[52:63])  # rev/day
    bstar = _parse_exp_field(line1[53:61])

    n_rad_s = mean_motion * 2.0 * math.pi / 86400.0
    semimajor_axis = (_MU_KM3_S2 / n_rad_s ** 2) ** (1.0 / 3.0)

    return {
        "INCLINATION": inclination,
        "RA_OF_ASC_NODE": raan,
        "ECCENTRICITY": eccentricity,
        "ARG_OF_PERICENTER": arg_pericenter,
        "MEAN_ANOMALY": mean_anomaly,
        "MEAN_MOTION": mean_motion,
        "BSTAR": bstar,
        "SEMIMAJOR_AXIS": semimajor_axis,
        "EPOCH_JD": _tle_epoch_jd(line1),
    }


def _add_parsed_elements(tles: pd.DataFrame) -> pd.DataFrame:
    """Populate OMM-style element columns by parsing each row's TLE lines.

    Columns are assigned (overwriting any element columns already present in a
    legacy Space-Track cache file) so the locally parsed values always win and
    no duplicate columns are produced.
    """
    tles = tles.copy()
    parsed = pd.DataFrame(
        [parse_tle_elements(r["TLE_LINE1"], r["TLE_LINE2"]) for _, r in tles.iterrows()],
        index=tles.index,
    )
    for col in parsed.columns:
        tles[col] = parsed[col]
    return tles


# ---------------------------------------------------------------------------
# Per-ID source resolution
# ---------------------------------------------------------------------------

def _select_from_extra_dir(
    extra_tle_dir: str,
    wanted: set[int],
    obs_epoch_jd: float,
    max_age_days: Optional[float],
) -> dict[int, dict]:
    """Resolve IDs from ``extra_tle_dir`` with per-ID nearest + age policy.

    Returns ``{norad_id: record}`` only for IDs whose nearest local TLE is within
    ``max_age_days`` of *obs_epoch_jd* (``None`` = unlimited). The age is measured
    from the TLE line-1 epoch, not the filename or file modification time.
    """
    resolved: dict[int, dict] = {}
    records = read_legacy_tle_records(extra_tle_dir)
    if not len(records):
        return resolved
    records = records.copy()
    records["NORAD_CAT_ID"] = pd.to_numeric(records["NORAD_CAT_ID"]).astype(int)
    records = records[records["NORAD_CAT_ID"].isin(wanted)]
    if not len(records):
        return resolved
    records["EPOCH_JD"] = records["TLE_LINE1"].map(_tle_epoch_jd)

    for nid, group in records.groupby("NORAD_CAT_ID"):
        best = group.loc[(group["EPOCH_JD"] - obs_epoch_jd).abs().idxmin()]
        age = abs(float(best["EPOCH_JD"]) - obs_epoch_jd)
        if max_age_days is None or age <= max_age_days + _AGE_TOL_DAYS:
            resolved[int(nid)] = best.to_dict()
            print(f"  {nid}: from extra_tle_dir (epoch {age:.3f} d from observation)")
        else:
            print(
                f"  {nid}: extra_tle_dir record rejected — {age:.3f} d old "
                f"> extra_tle_max_age_days={max_age_days}; trying managed catalogue"
            )
    return resolved


def _select_from_records(
    records: pd.DataFrame,
    wanted: set[int],
) -> dict[int, dict]:
    """One record per wanted ID from a normalised catalogue/fallback frame."""
    resolved: dict[int, dict] = {}
    if not len(records):
        return resolved
    records = records.copy()
    records["NORAD_CAT_ID"] = pd.to_numeric(records["NORAD_CAT_ID"]).astype(int)
    match = records[records["NORAD_CAT_ID"].isin(wanted)]
    for nid, group in match.groupby("NORAD_CAT_ID"):
        resolved[int(nid)] = group.iloc[0].to_dict()
    return resolved


def _fetch_missing_ids(missing: list[int], epoch_jd: float) -> pd.DataFrame:
    """Per-satellite fallback fetch for IDs absent from the bulk catalogue."""
    rows: list[pd.DataFrame] = []
    for i, nid in enumerate(missing):
        if i:
            time.sleep(_THROTTLE_SECONDS)
        try:
            rec = satchecker.fetch_nearest_tle(nid, epoch_jd)
        except TLEError as e:
            print(f"  fallback fetch failed for {nid}: {e}")
            continue
        if len(rec):
            rows.append(rec)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _ensure_snapshot(
    cache: TextCatalogueCache,
    catalogue_epoch_jd: float,
    obs_epoch_jd: float,
) -> Optional[CatalogueSnapshot]:
    """Return the canonical snapshot, downloading + caching atomically on a miss.

    Returns ``None`` when the service is reachable but reports an *empty*
    catalogue at this epoch (its ``tles-at-epoch`` endpoint has a data horizon and
    recent epochs can fall beyond it). The caller then resolves satellites through
    the per-ID fallback instead. Transport failures still raise.
    """
    snapshot = cache.get_snapshot(catalogue_epoch_jd)
    if snapshot is not None:
        print(f"  managed catalogue cached at {canonical_stamp(catalogue_epoch_jd)}")
        return snapshot

    print(
        f"Fetching TLE catalogue from SatChecker for canonical epoch "
        f"{canonical_stamp(catalogue_epoch_jd)} ..."
    )
    try:
        result = satchecker.fetch_full_catalogue(catalogue_epoch_jd)
    except satchecker.EmptyCatalogueError as e:
        print(f"  {e}")
        print("  falling back to per-satellite TLE lookups for all requested IDs")
        return None
    snapshot = CatalogueSnapshot(
        catalogue_epoch_jd=catalogue_epoch_jd,
        records=result.records,
        requested_epoch_jd=obs_epoch_jd,
        expected_count=result.expected_count,
        actual_count=result.actual_count,
        service_version=result.service_version,
    )
    cache.store_snapshot(snapshot)
    print(f"Saved {len(result.records)} TLEs for {canonical_stamp(catalogue_epoch_jd)}")
    return snapshot


# ---------------------------------------------------------------------------
# Public orchestration
# ---------------------------------------------------------------------------

def get_tles_by_id(
    norad_ids: list[int],
    times_jd,
    extra_tle_dir: Optional[str] = None,
    extra_tle_max_age_days: Optional[float] = None,
    catalogue_interval_hours: float = DEFAULT_CATALOGUE_INTERVAL_HOURS,
) -> pd.DataFrame:
    """Return TLE records + locally parsed elements for *norad_ids*.

    Source precedence is resolved independently per NORAD ID: ``extra_tle_dir``
    (subject to ``extra_tle_max_age_days``), then the managed canonical catalogue
    snapshot, then the per-satellite SatChecker fallback. One row per resolved ID
    is returned, with OMM-style element columns parsed locally from the TLE lines.

    Parameters
    ----------
    norad_ids:
        NORAD catalogue IDs to select.
    times_jd:
        Observation time(s) as UTC Julian Date(s); scalar or array. The mean is
        used as the requested observation epoch, which sets both the extra-dir age
        comparison and (after bucketing) the canonical catalogue epoch.
    extra_tle_dir:
        Optional user-supplied directory of local TLE files searched first, per ID.
    extra_tle_max_age_days:
        Maximum absolute difference, in days, between a local TLE's epoch and the
        observation epoch for it to be accepted. ``None`` = unlimited; ``0`` =
        exact-epoch only; negative is a configuration error.
    catalogue_interval_hours:
        Width of the fixed UTC catalogue-reuse bucket (default 2 h).
    """
    max_age = _validate_max_age(extra_tle_max_age_days)
    wanted = {int(x) for x in np.atleast_1d(np.asarray(norad_ids)).astype(int)}
    if not wanted:
        return pd.DataFrame()

    obs_epoch_jd = float(np.atleast_1d(np.asarray(times_jd, dtype=float)).mean())
    catalogue_epoch_jd = canonical_epoch_jd(obs_epoch_jd, catalogue_interval_hours)

    print(f"TLE requested epoch    : {jd_to_datetime(obs_epoch_jd).isoformat()} UTC")
    print(
        f"TLE catalogue epoch    : {jd_to_datetime(catalogue_epoch_jd).isoformat()} UTC "
        f"(nearest to a {catalogue_interval_hours:g} h bucket, not the exact observation)"
    )

    resolved: dict[int, dict] = {}

    # 1. extra_tle_dir (per-ID precedence + age policy)
    if extra_tle_dir:
        print(f"TLE extra dir          : {Path(extra_tle_dir).resolve()}")
        resolved.update(_select_from_extra_dir(extra_tle_dir, wanted, obs_epoch_jd, max_age))

    remaining = wanted - set(resolved)

    # 2 + 3. managed canonical snapshot, then per-satellite fallback
    if remaining:
        cache = TextCatalogueCache(tle_cache_dir())
        snapshot = _ensure_snapshot(cache, catalogue_epoch_jd, obs_epoch_jd)
        if snapshot is not None:
            resolved.update(_select_from_records(snapshot.records, remaining))
            remaining = wanted - set(resolved)

        if remaining:
            cached_extra = cache.get_extra(catalogue_epoch_jd)
            resolved.update(_select_from_records(cached_extra, remaining))
            remaining = wanted - set(resolved)

        if remaining:
            missing = sorted(remaining)
            print(f"Fetching {len(missing)} missing TLE(s) individually from SatChecker: {missing}")
            fetched = _fetch_missing_ids(missing, catalogue_epoch_jd)
            if len(fetched):
                cache.store_extra(catalogue_epoch_jd, fetched)
                resolved.update(_select_from_records(fetched, remaining))
                remaining = wanted - set(resolved)

    still_missing = sorted(wanted - set(resolved))
    if still_missing:
        print(f"TLEs not found         : {still_missing}")
    if not resolved:
        return pd.DataFrame()

    tles = pd.DataFrame([resolved[nid] for nid in sorted(resolved)])
    tles["NORAD_CAT_ID"] = pd.to_numeric(tles["NORAD_CAT_ID"]).astype(int)
    tles = _add_parsed_elements(tles)
    return tles.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reproducibility: persist the TLEs a run actually used
# ---------------------------------------------------------------------------

def save_tles_for_reuse(path, norad_ids, tles) -> Optional[str]:
    """Write the TLE lines a run used to *path* in ``extra_tle_dir`` format.

    The file is a pandas-oriented JSON with ``NORAD_CAT_ID``, ``TLE_LINE1`` and
    ``TLE_LINE2`` columns — exactly what :func:`read_legacy_tle_records` reads —
    so a later run can reproduce this run's trajectory priors by passing the
    file's directory via ``--extra-tle-dir`` (with the default unlimited
    ``extra_tle_max_age_days``), independent of the shared cache or any change
    in what SatChecker serves.

    ``norad_ids`` and ``tles`` are the aligned arrays produced by the element
    fetchers (one ``(line1, line2)`` pair per ID). Returns the written path, or
    ``None`` when there is nothing to save.
    """
    tle_pairs = np.atleast_2d(np.asarray(tles)) if tles is not None else np.empty((0, 2))
    ids = list(norad_ids or [])
    if not ids or not tle_pairs.size:
        return None
    df = pd.DataFrame(
        {
            "NORAD_CAT_ID": [int(n) for n in ids],
            "TLE_LINE1": tle_pairs[:, 0],
            "TLE_LINE2": tle_pairs[:, 1],
        }
    )
    path = str(path)
    df.to_json(path)
    return path


# ---------------------------------------------------------------------------
# Preflight check
# ---------------------------------------------------------------------------

def _ms_mean_epoch_jd(ms_path: str) -> float:
    """Mean observation epoch (UTC Julian Date) from an MS ``TIME`` column.

    Isolated so :func:`preflight_tle_check` can be exercised offline by patching
    this one casacore-touching seam.
    """
    from casacore.tables import table as _ms_table
    with _ms_table(ms_path, readonly=True, ack=False) as t:
        return float(t.getcol("TIME").mean()) / 86400.0 + 2400000.5


def preflight_tle_check(
    norad_ids: list[int],
    ms_path: str,
    extra_tle_dir: Optional[str] = None,
    extra_tle_max_age_days: Optional[float] = None,
    catalogue_interval_hours: float = DEFAULT_CATALOGUE_INTERVAL_HOURS,
) -> None:
    """Report whether the required canonical catalogue snapshot is cached locally.

    Reads the mean observation epoch from the MS ``TIME`` column (MJD seconds),
    reports the requested epoch and the deterministic canonical catalogue epoch it
    maps to, and states whether that snapshot is already cached or will be
    downloaded from SatChecker (which needs no credentials). Purely informational
    — no exception is raised for a cache miss or configuration issue.
    """
    if not norad_ids:
        return

    max_age = _validate_max_age(extra_tle_max_age_days)

    mean_epoch_jd = _ms_mean_epoch_jd(ms_path)

    catalogue_epoch_jd = canonical_epoch_jd(mean_epoch_jd, catalogue_interval_hours)
    cache = TextCatalogueCache(tle_cache_dir())
    wanted = {int(x) for x in norad_ids}

    print(f"Preflight TLE check    : NORAD IDs {sorted(wanted)}")
    print(f"TLE requested epoch    : {jd_to_datetime(mean_epoch_jd).isoformat()} UTC")
    print(
        f"TLE catalogue epoch    : {jd_to_datetime(catalogue_epoch_jd).isoformat()} UTC "
        f"({catalogue_interval_hours:g} h bucket)"
    )

    # Account for IDs already covered by extra_tle_dir so we don't wrongly promise a
    # download when no managed snapshot is actually needed.
    remaining = set(wanted)
    if extra_tle_dir:
        print(f"TLE extra dir          : {Path(extra_tle_dir).resolve()}"
              f"  (max age {extra_tle_max_age_days} d)")
        from_extra = _select_from_extra_dir(extra_tle_dir, wanted, mean_epoch_jd, max_age)
        remaining -= set(from_extra)

    if not remaining:
        print("  all requested TLEs available from extra_tle_dir — no download needed")
    elif cache.get_snapshot(catalogue_epoch_jd) is not None:
        print(f"  managed catalogue cached at {canonical_stamp(catalogue_epoch_jd)}")
    else:
        print(f"  managed catalogue not cached — will download from SatChecker")
    print(
        "  To search additional local TLE directories:\n"
        "    --extra-tle-dir <dir>  (CLI flag)\n"
        "    TLE_CACHE_DIR=<dir>    (env var, overrides managed cache location)"
    )
