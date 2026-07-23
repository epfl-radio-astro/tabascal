"""TLE fetching, caching, and local orbital-element parsing for tabascal.

TLEs are sourced from the IAU CPS **SatChecker** service
(https://satchecker.cps.iau.org/) — no account or credentials are required.

Strategy: on each run the *full* TLE catalogue for the observation date(s) is
downloaded once (``/tools/tles-at-epoch/``, one nearest TLE per satellite),
cached locally, and then all filtering by NORAD ID and orbital-element
computation is done locally by parsing the two TLE lines. The bulk download is
validated against the reported ``total_results`` (the zip is occasionally
truncated); any requested NORAD ID still missing is fetched individually via
``/tools/get-nearest-tle/`` (throttled) and cached alongside the catalogue.

TLE search order (each date is resolved independently):
  1. User-supplied directory  (--extra-tle-dir <dir> CLI flag)
  2. Managed cache directory  (default: platformdirs user cache / tle-cache,
                               override via TLE_CACHE_DIR env var)
  3. SatChecker API           (the full catalogue is saved to the managed cache)

Cache files are named ``<YYYY-MM-DD>-catalogue.json`` — one deterministic file
per UTC date, so repeated runs over the same date reuse the cache instead of
creating duplicates. The ``<YYYY-MM-DD>-*.json`` glob is retained so legacy
Space-Track cache files (and the bundled test fixtures) are still discovered;
only ``NORAD_CAT_ID`` and the two TLE lines are read from any cache file.
"""

from __future__ import annotations

import io
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path
from typing import Optional

from platformdirs import user_cache_path

import numpy as np
import pandas as pd

from tabascal.time import datetime_to_jd, jd_to_datetime


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SATCHECKER_BASE_URL = "https://satchecker.cps.iau.org/tools"
_USER_AGENT = "tabascal-tle/1.0"
_REQUEST_TIMEOUT = 300  # seconds — the full catalogue zip is a few MB
_MU_KM3_S2 = 398600.4418  # Earth gravitational parameter, km^3/s^2
_THROTTLE_SECONDS = 1.0   # delay between per-satellite fallback requests
_CATALOGUE_MIN_FRACTION = 0.99  # accept a zip download this complete vs total_results

# Columns the cached catalogue is normalised to (a superset of what any
# downstream consumer reads from a cache file).
_CATALOGUE_COLUMNS = [
    "NORAD_CAT_ID",
    "OBJECT_NAME",
    "EPOCH",
    "TLE_LINE1",
    "TLE_LINE2",
    "DATA_SOURCE",
    "DATE_COLLECTED",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TLEError(RuntimeError):
    """Raised when TLEs cannot be fetched or no data is available."""


# ---------------------------------------------------------------------------
# TLE directory helpers
# ---------------------------------------------------------------------------

def tle_cache_dir() -> Path:
    """Return the TLE cache directory, creating it if needed.

    The directory is resolved in priority order:
    1. ``TLE_CACHE_DIR`` environment variable (if set).
    2. The platform user-cache directory (e.g. ``~/.cache/tle-cache`` on Linux,
       ``~/Library/Caches/tle-cache`` on macOS).
    """
    p = Path(os.environ.get("TLE_CACHE_DIR") or user_cache_path("tle-cache"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _search_dirs(extra_tle_dir: Optional[str]) -> list[Path]:
    """Directories searched for cached catalogues (extra first, then managed cache)."""
    cache_dir = tle_cache_dir()
    if extra_tle_dir:
        return [Path(extra_tle_dir).resolve(), cache_dir]
    return [cache_dir]


# ---------------------------------------------------------------------------
# SatChecker catalogue fetch
# ---------------------------------------------------------------------------

def _http_get(url: str, timeout: int = _REQUEST_TIMEOUT) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (urllib.error.URLError, TimeoutError) as e:
        raise TLEError(f"SatChecker request failed ({url}): {e}") from e


def _normalise_catalogue(records: pd.DataFrame) -> pd.DataFrame:
    """Rename SatChecker fields to the cached-catalogue column names."""
    rename = {
        "satellite_id": "NORAD_CAT_ID",
        "satellite_name": "OBJECT_NAME",
        "epoch": "EPOCH",
        "tle_line1": "TLE_LINE1",
        "tle_line2": "TLE_LINE2",
        "data_source": "DATA_SOURCE",
        "date_collected": "DATE_COLLECTED",
    }
    df = records.rename(columns=rename)
    for col in _CATALOGUE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[_CATALOGUE_COLUMNS].copy()
    df["NORAD_CAT_ID"] = pd.to_numeric(df["NORAD_CAT_ID"]).astype(int)
    return df.reset_index(drop=True)


def _fetch_catalogue_zip(epoch_jd: float) -> pd.DataFrame:
    q = urllib.parse.urlencode({"epoch": repr(float(epoch_jd)), "format": "zip"})
    url = f"{SATCHECKER_BASE_URL}/tles-at-epoch/?{q}"
    raw = _http_get(url)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = zf.namelist()
        if not names:
            raise TLEError("SatChecker returned an empty zip archive.")
        with zf.open(names[0]) as f:
            df = pd.read_csv(f)
    return _normalise_catalogue(df)


def _catalogue_total(epoch_jd: float) -> int:
    """Expected number of catalogue records at *epoch_jd* (from JSON total_results)."""
    q = urllib.parse.urlencode(
        {"epoch": repr(float(epoch_jd)), "format": "json", "per_page": 1, "page": 1}
    )
    payload = json.loads(_http_get(f"{SATCHECKER_BASE_URL}/tles-at-epoch/?{q}", timeout=120))
    obj = payload[0] if isinstance(payload, list) else payload
    return int(obj.get("total_results", 0))


def _fetch_catalogue_json(epoch_jd: float, per_page: int = 5000) -> pd.DataFrame:
    """Paginated JSON fallback for the full catalogue at *epoch_jd*."""
    frames: list[pd.DataFrame] = []
    page = 1
    while True:
        q = urllib.parse.urlencode(
            {
                "epoch": repr(float(epoch_jd)),
                "format": "json",
                "per_page": per_page,
                "page": page,
            }
        )
        url = f"{SATCHECKER_BASE_URL}/tles-at-epoch/?{q}"
        payload = json.loads(_http_get(url))
        obj = payload[0] if isinstance(payload, list) else payload
        rows = obj.get("data", [])
        if rows:
            frames.append(pd.DataFrame(rows))
        total = int(obj.get("total_results", 0))
        if page * per_page >= total or not rows:
            break
        page += 1

    if not frames:
        raise TLEError("SatChecker returned no TLE records.")
    return _normalise_catalogue(pd.concat(frames, ignore_index=True))


def fetch_full_catalogue(epoch_jd: float) -> pd.DataFrame:
    """Download the full TLE catalogue nearest *epoch_jd* from SatChecker.

    Uses the efficient ``format=zip`` endpoint, validating the row count against
    the ``total_results`` reported by the JSON endpoint (the zip response is
    occasionally truncated). A short download is retried, then falls back to the
    paginated ``format=json`` endpoint.

    Returns
    -------
    pd.DataFrame
        One row per satellite in the catalogue, columns ``_CATALOGUE_COLUMNS``.
    """
    try:
        expected = _catalogue_total(epoch_jd)
    except Exception:
        expected = 0

    last_err: Optional[Exception] = None
    for _ in range(2):
        try:
            df = _fetch_catalogue_zip(epoch_jd)
        except Exception as e:  # pragma: no cover - network dependent
            last_err = e
            continue
        if not expected or len(df) >= expected * _CATALOGUE_MIN_FRACTION:
            return df
        last_err = TLEError(
            f"SatChecker zip truncated ({len(df)} of {expected} records) — retrying"
        )
        print(f"  {last_err}")

    # zip repeatedly short or failing → paginated JSON (complete but slower)
    try:
        return _fetch_catalogue_json(epoch_jd)
    except Exception as json_err:
        raise TLEError(
            f"Failed to fetch TLE catalogue from SatChecker: {json_err} "
            f"(zip attempt: {last_err})"
        ) from json_err


def fetch_nearest_tle(norad_id: int, epoch_jd: float) -> pd.DataFrame:
    """Fetch the single TLE nearest *epoch_jd* for one satellite from SatChecker.

    Used as a per-satellite fallback for NORAD IDs missing from the bulk
    catalogue. Returns an empty DataFrame if SatChecker has no record.
    """
    q = urllib.parse.urlencode(
        {"id": int(norad_id), "id_type": "catalog", "epoch": repr(float(epoch_jd))}
    )
    payload = json.loads(_http_get(f"{SATCHECKER_BASE_URL}/get-nearest-tle/?{q}", timeout=120))
    obj = payload[0] if isinstance(payload, list) else payload
    rows = obj.get("orbital_data") or obj.get("tle_data") or []
    if not rows:
        return pd.DataFrame()
    return _normalise_catalogue(pd.DataFrame(rows))


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
# Date helpers
# ---------------------------------------------------------------------------

def _spanned_dates(times_jd) -> list[str]:
    """UTC calendar dates (YYYY-MM-DD) covered by the observation times."""
    times = np.atleast_1d(np.asarray(times_jd, dtype=float))
    d0 = jd_to_datetime(times.min()).date()
    d1 = jd_to_datetime(times.max()).date()
    dates, d = [], d0
    while d <= d1:
        dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return dates


def _date_query_jd(date_str: str) -> float:
    """Deterministic query epoch (noon UTC) for a catalogue date."""
    dt = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(hours=12)
    return datetime_to_jd(dt)


# ---------------------------------------------------------------------------
# Cached catalogue access
# ---------------------------------------------------------------------------

def _load_cached_catalogue(date_str: str, search_dirs: list[Path]) -> pd.DataFrame:
    """Return any locally cached catalogue rows for *date_str* (may be empty)."""
    frames = []
    for d in search_dirs:
        for path in glob(str(d / f"{date_str}-*.json")):
            try:
                frames.append(pd.read_json(path))
            except Exception:
                continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _ensure_catalogue(date_str: str, search_dirs: list[Path]) -> pd.DataFrame:
    """Return the catalogue for *date_str*, downloading + caching if absent."""
    cached = _load_cached_catalogue(date_str, search_dirs)
    if len(cached):
        return cached

    print(f"Fetching TLE catalogue from SatChecker for {date_str} ...")
    catalogue = fetch_full_catalogue(_date_query_jd(date_str))
    save_path = tle_cache_dir() / f"{date_str}-catalogue.json"
    catalogue.to_json(save_path)
    print(f"Saved {len(catalogue)} TLEs to {save_path}")
    return catalogue


def _append_extra_cache(date_str: str, extra: pd.DataFrame) -> None:
    """Merge per-satellite fallback records into the date's ``-extra`` cache file.

    Kept in a separate ``<date>-extra.json`` file (still matched by the
    ``<date>-*.json`` glob) so future runs reuse them without re-querying, and
    the bulk catalogue file is never rewritten.
    """
    path = tle_cache_dir() / f"{date_str}-extra.json"
    if path.exists():
        try:
            extra = pd.concat([pd.read_json(path), extra], ignore_index=True)
        except Exception:
            pass
    extra = extra.drop_duplicates(subset=["NORAD_CAT_ID", "TLE_LINE1", "TLE_LINE2"])
    extra.to_json(path)


def _fetch_missing_ids(missing: list[int], epoch_jd: float) -> pd.DataFrame:
    """Per-satellite fallback fetch for IDs absent from the bulk catalogue."""
    rows: list[pd.DataFrame] = []
    for i, nid in enumerate(missing):
        if i:
            time.sleep(_THROTTLE_SECONDS)
        try:
            rec = fetch_nearest_tle(nid, epoch_jd)
        except TLEError as e:
            print(f"  fallback fetch failed for {nid}: {e}")
            continue
        if len(rec):
            rows.append(rec)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def get_tles_by_id(
    norad_ids: list[int],
    times_jd,
    extra_tle_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Return TLE records + parsed elements for *norad_ids* near the observation.

    The full catalogue for each UTC date spanned by *times_jd* is loaded from a
    local cache (downloading from SatChecker on a miss), filtered to the
    requested NORAD IDs, and the orbital elements are parsed locally from the
    TLE lines. One row per satellite is returned — the epoch closest to the mean
    observation time.

    Parameters
    ----------
    norad_ids:
        NORAD catalogue IDs to select.
    times_jd:
        Observation time(s) as UTC Julian Date(s); scalar or array. Determines
        the spanned catalogue date(s) and the target epoch for nearest-TLE
        selection.
    extra_tle_dir:
        Optional user-supplied directory searched before the managed cache.
    """
    search_dirs = _search_dirs(extra_tle_dir)
    print(f"TLE search dirs        : {[str(d) for d in search_dirs]}")

    wanted = set(int(x) for x in np.atleast_1d(np.asarray(norad_ids)).astype(int))
    epoch_mean = float(np.atleast_1d(np.asarray(times_jd, dtype=float)).mean())
    dates = _spanned_dates(times_jd)

    frames = [_ensure_catalogue(d, search_dirs) for d in dates]
    catalogue = pd.concat([f for f in frames if len(f)], ignore_index=True)
    if not len(catalogue):
        return pd.DataFrame()

    catalogue["NORAD_CAT_ID"] = pd.to_numeric(catalogue["NORAD_CAT_ID"]).astype(int)
    tles = catalogue[catalogue["NORAD_CAT_ID"].isin(wanted)].copy()

    found = set(tles["NORAD_CAT_ID"].unique())
    print(f"Catalogue TLEs matched : {len(found)} / {len(wanted)}")

    # --- per-satellite fallback for IDs absent from the bulk catalogue ---
    missing = sorted(wanted - found)
    if missing:
        print(f"Fetching {len(missing)} missing TLE(s) individually from SatChecker: {missing}")
        extra = _fetch_missing_ids(missing, epoch_mean)
        if len(extra):
            extra["NORAD_CAT_ID"] = pd.to_numeric(extra["NORAD_CAT_ID"]).astype(int)
            _append_extra_cache(jd_to_datetime(epoch_mean).strftime("%Y-%m-%d"), extra)
            tles = pd.concat([tles, extra], ignore_index=True)
            found = set(tles["NORAD_CAT_ID"].unique())

    still_missing = sorted(wanted - found)
    if still_missing:
        print(f"TLEs not found         : {still_missing}")

    if not len(tles):
        return pd.DataFrame()

    tles = tles.drop_duplicates(subset=["NORAD_CAT_ID", "TLE_LINE1", "TLE_LINE2"])
    tles = _add_parsed_elements(tles)
    tles = _get_closest_times(tles, epoch_mean)
    return tles.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Preflight check
# ---------------------------------------------------------------------------

def preflight_tle_check(
    norad_ids: list[int],
    ms_path: str,
    extra_tle_dir: Optional[str] = None,
) -> None:
    """Report whether the required TLE catalogue(s) are cached locally.

    Reads the mean observation epoch from the MS ``TIME`` column (MJD seconds)
    and lists, per spanned UTC date, whether a catalogue is already cached or
    will be downloaded from SatChecker (which needs no credentials). Purely
    informational — no exception is raised for a cache miss.
    """
    if not norad_ids:
        return

    from casacore.tables import table as _ms_table
    with _ms_table(ms_path, readonly=True, ack=False) as t:
        mean_epoch_jd = float(t.getcol("TIME").mean()) / 86400.0 + 2400000.5

    dates = _spanned_dates(mean_epoch_jd)
    search_dirs = _search_dirs(extra_tle_dir)
    print(f"Preflight TLE check    : dates {dates}, NORAD IDs {sorted(norad_ids)}")
    print(f"TLE search dirs        : {[str(d) for d in search_dirs]}")

    for date_str in dates:
        cached = _load_cached_catalogue(date_str, search_dirs)
        if len(cached):
            print(f"  {date_str}: catalogue cached ({len(cached)} TLEs)")
        else:
            print(f"  {date_str}: not cached — will download from SatChecker")
    print(
        "  To search additional local TLE directories:\n"
        "    --extra-tle-dir <dir>  (CLI flag)\n"
        "    TLE_CACHE_DIR=<dir>    (env var, overrides managed cache location)"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_closest_times(
    df: pd.DataFrame,
    target_jd: float,
    id_col: str = "NORAD_CAT_ID",
    time_col: str = "EPOCH_JD",
) -> pd.DataFrame:
    """Return one row per NORAD ID — the one whose epoch is closest to *target_jd*."""
    df = df.copy()
    df["time_diff"] = df[time_col] - target_jd
    df["time_diff_abs"] = df[time_col].sub(target_jd).abs()
    return df.loc[df.groupby(id_col)["time_diff_abs"].idxmin()]
