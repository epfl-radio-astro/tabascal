"""TLE fetching, caching, and Space-Track credential handling for tabascal.

TLE search order on each run (each step only runs for IDs not yet found):
  1. User-supplied directory  (--extra-tle-dir <dir> CLI flag)
  2. Managed cache directory  (default: platformdirs user cache / tle-cache,
                               override via TLE_CACHE_DIR env var)
  3. Space-Track API          (new records are saved to the managed cache)

Cache files are named <YYYY-MM-DD>-<random-id>.json.

Space-Track credentials are loaded in priority order:
  1. Environment variables  SPACETRACK_USER and SPACETRACK_PASSWORD
  2. Config file            ~/.config/tabascal/spacetrack_login.yaml
                            (written by: tabascal spacetrack-login)
"""

from __future__ import annotations

import json
import os
import random
import string
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Optional

from platformdirs import user_cache_path, user_config_path

import numpy as np
import pandas as pd
import yaml
from astropy.time import Time
from spacetrack import SpaceTrackClient
import spacetrack.operators as op


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


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TLEError(RuntimeError):
    """Raised when TLEs cannot be fetched due to missing credentials or data."""


# ---------------------------------------------------------------------------
# Credential helpers
# ---------------------------------------------------------------------------

def spacetrack_config_path() -> Path:
    """Return the path to the Space-Track credentials config file."""
    return user_config_path("tabascal") / "spacetrack_login.yaml"


def save_spacetrack_credentials(username: str, password: str) -> Path:
    """Write Space-Track credentials to the user config file.

    Parameters
    ----------
    username:
        Space-Track account email address.
    password:
        Space-Track account password.

    Returns
    -------
    Path
        The path where the credentials were written.
    """
    path = spacetrack_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump({"username": username, "password": password}, f)
    return path


# ---------------------------------------------------------------------------
# Credential loading
# ---------------------------------------------------------------------------

def load_spacetrack_credentials() -> tuple[str, str]:
    """Load Space-Track credentials, raising if none are configured.

    Checks in priority order:

    1. Environment variables ``SPACETRACK_USER`` and ``SPACETRACK_PASSWORD``.
    2. Config file at ``{spacetrack_config_path()}``.

    Raises
    ------
    RuntimeError
        When no credentials are found, with instructions for how to set them.
    """
    user = os.environ.get("SPACETRACK_USER")
    password = os.environ.get("SPACETRACK_PASSWORD")
    if user and password:
        return user, password

    config_path = spacetrack_config_path()
    if config_path.exists():
        with open(config_path) as f:
            creds = yaml.safe_load(f)
        user = creds.get("username")
        password = creds.get("password")
        if user and password:
            return user, password

    raise TLEError(
        "Space-Track credentials not found.\n\n"
        "Set them via environment variables:\n"
        "  SPACETRACK_USER=<email> SPACETRACK_PASSWORD=<password>\n"
        f"or save them to {spacetrack_config_path()} by running:\n"
        "  tabascal spacetrack-login"
    )


def print_spacetrack_status() -> None:
    """Print where Space-Track credentials are configured, or warn if not found."""
    user = os.environ.get("SPACETRACK_USER")
    password = os.environ.get("SPACETRACK_PASSWORD")
    if user and password:
        print("Space-Track credentials : environment variables (SPACETRACK_USER, SPACETRACK_PASSWORD)")
        return

    config_path = spacetrack_config_path()
    if config_path.exists():
        try:
            with open(config_path) as f:
                creds = yaml.safe_load(f)
            if creds.get("username") and creds.get("password"):
                print(f"Space-Track credentials : {config_path}")
                return
        except Exception:
            pass

    print(
        f"Space-Track credentials : not configured — remote TLE fetching will fail\n"
        f"  Set via : SPACETRACK_USER=<email> SPACETRACK_PASSWORD=<password>\n"
        f"  Or run  : tabascal spacetrack-login\n"
    )


def preflight_tle_check(
    norad_ids: list[int],
    ms_path: str,
    extra_tle_dir: Optional[str] = None,
) -> None:
    """Fail fast if TLEs are not locally available and credentials are not configured.

    Reads the mean observation epoch from the MS ``TIME`` column (MJD seconds),
    then searches local TLE dirs using the same date-matched pattern as
    ``get_tles_by_id``.  If any NORAD IDs are missing locally, Space-Track
    credentials are verified immediately so the caller can abort before doing
    any expensive computation.

    Parameters
    ----------
    norad_ids:
        NORAD catalogue IDs required for the run.
    ms_path:
        Path to the Measurement Set.
    extra_tle_dir:
        Optional user-supplied directory searched before the managed cache.
    """
    if not norad_ids:
        return

    from casacore.tables import table as _ms_table
    with _ms_table(ms_path, readonly=True, ack=False) as t:
        mean_epoch_jd = float(t.getcol("TIME").mean()) / 86400.0 + 2400000.5

    epoch_str = Time(mean_epoch_jd, format="jd", scale="ut1").strftime("%Y-%m-%d")
    print(f"Preflight TLE check    : epoch {epoch_str}, NORAD IDs {sorted(norad_ids)}")

    cache_dir = tle_cache_dir()
    search_dirs = [Path(extra_tle_dir).resolve(), cache_dir] if extra_tle_dir else [cache_dir]
    print(f"TLE search dirs        : {[str(d) for d in search_dirs]}")

    found_ids: set[int] = set()
    for d in search_dirs:
        for json_path in glob(str(d / f"{epoch_str}-*.json")):
            try:
                df = pd.read_json(json_path)
                if "NORAD_CAT_ID" in df.columns:
                    found_ids.update(int(x) for x in df["NORAD_CAT_ID"].unique())
            except Exception:
                pass

    missing = sorted(set(int(n) for n in norad_ids) - found_ids)
    if missing:
        print(f"TLEs not in local dirs : {missing} — verifying Space-Track credentials")
        print(
            "  To search additional local TLE directories:\n"
            "    --extra-tle-dir <dir>  (CLI flag)\n"
            "    TLE_CACHE_DIR=<dir>    (env var, overrides managed cache location)"
        )
        load_spacetrack_credentials()  # raises RuntimeError if not configured
        print(f"Space-Track credentials verified — will fetch {len(missing)} satellite(s) during run")
    else:
        print(f"TLEs found locally     : all {len(norad_ids)} satellite(s) covered")


# ---------------------------------------------------------------------------
# Space-Track client
# ---------------------------------------------------------------------------

def get_space_track_client(username: str, password: str) -> SpaceTrackClient:
    """Return an authenticated SpaceTrackClient."""
    return SpaceTrackClient(identity=username, password=password)


# ---------------------------------------------------------------------------
# Raw API fetch
# ---------------------------------------------------------------------------

def fetch_tle_data(
    st_client: SpaceTrackClient,
    norad_ids: list[int],
    epoch_jd: float,
    window_days: float = 1.0,
    limit: int = 2000,
) -> pd.DataFrame:
    """Fetch TLE data for the given NORAD IDs around *epoch_jd*.

    Parameters
    ----------
    st_client:
        Authenticated SpaceTrackClient.
    norad_ids:
        NORAD catalogue IDs to query.
    epoch_jd:
        Julian date of the observation epoch.
    window_days:
        Half-width of the time window around *epoch_jd* (days).
    limit:
        Maximum number of records to return per request.
    """
    start_time = Time(epoch_jd - window_days, format="jd", scale="ut1").datetime
    end_time   = Time(epoch_jd + window_days, format="jd", scale="ut1").datetime
    date_range = op.inclusive_range(start_time, end_time)

    raw = st_client.gp_history(
        norad_cat_id=norad_ids, epoch=date_range, limit=limit, format="json"
    )
    return pd.DataFrame(json.loads(raw))


# ---------------------------------------------------------------------------
# Caching fetch
# ---------------------------------------------------------------------------

def _random_id(size: int = 6) -> str:
    return "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(size))


def get_tles_by_id(
    norad_ids: list[int],
    epoch_jd: float,
    window_days: float = 1.0,
    limit: int = 2000,
    extra_tle_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch TLE records for the given NORAD IDs, using a local JSON cache.

    Searches for already-cached records in *extra_tle_dir* (if given) first,
    then in the managed cache directory (``tle_cache_dir()``).  Only IDs not
    found locally are fetched from Space-Track; credentials are loaded via
    ``load_spacetrack_credentials()`` only when a remote fetch is required.
    New records are saved to the managed cache, never to *extra_tle_dir*.

    Parameters
    ----------
    norad_ids:
        NORAD catalogue IDs to fetch.
    epoch_jd:
        Julian date of the observation epoch (used to select the nearest TLE).
    window_days:
        Half-width of the time window used when querying Space-Track.
    limit:
        Maximum records per API request.
    extra_tle_dir:
        Optional user-supplied directory searched before the managed cache.

    Returns
    -------
    pd.DataFrame
        One row per requested satellite, the row whose epoch is closest to
        *epoch_jd*.
    """
    cache_dir = tle_cache_dir()
    search_dirs = [Path(extra_tle_dir).resolve(), cache_dir] if extra_tle_dir else [cache_dir]

    print(f"TLE search dirs        : {[str(d) for d in search_dirs]}")

    remaining: list[int] = list(np.array(list(set(norad_ids))).astype(int))
    n_ids_start = len(remaining)
    epoch_str = Time(epoch_jd, format="jd", scale="ut1").strftime("%Y-%m-%d")

    # --- load from local dirs (extra first, then cache) ---
    tles_local = pd.DataFrame()
    local_ids: list[int] = []
    for d in search_dirs:
        if not remaining:
            break
        tle_paths = glob(str(d / f"{epoch_str}-*.json"))
        if tle_paths:
            found = pd.concat([pd.read_json(p) for p in tle_paths])
            found = found[found["NORAD_CAT_ID"].isin(remaining)]
            if len(found):
                tles_local = pd.concat([tles_local, found]) if len(tles_local) else found
                new_ids = list(found["NORAD_CAT_ID"].unique())
                local_ids.extend(new_ids)
                remaining = list(set(remaining) - set(new_ids))
    print(f"Local TLEs loaded  : {len(local_ids)}")

    # --- fetch remaining IDs from Space-Track, save to cache ---
    max_ids = 500
    tles_remote = pd.DataFrame()
    remote_ids: list[int] = []
    if remaining:
        username, password = load_spacetrack_credentials()
        client = get_space_track_client(username, password)
        chunks = [
            remaining[i : i + max_ids] for i in range(0, len(remaining), max_ids)
        ]
        frames = [fetch_tle_data(client, chunk, epoch_jd, window_days, limit) for chunk in chunks]
        non_empty = [f for f in frames if len(f) > 0]
        if non_empty:
            tles_remote = pd.concat(non_empty)
            tles_remote["Fetch_Timestamp"] = Time.now().fits
            remote_ids = list(tles_remote["NORAD_CAT_ID"].unique())

            save_path = cache_dir / f"{epoch_str}-{_random_id()}.json"
            tles_remote.to_json(save_path)
            print(f"Saving remotely obtained TLEs to {save_path}")

    n_not_found = n_ids_start - len(remote_ids) - len(local_ids)
    print(f"Remote TLEs loaded : {len(remote_ids)}")
    print(f"TLEs not found     : {n_not_found}")
    if n_not_found > 0:
        print(
            "  To search additional local TLE directories:\n"
            "    --extra-tle-dir <dir>  (CLI flag)\n"
            "    TLE_CACHE_DIR=<dir>    (env var, overrides managed cache location)"
        )

    # --- merge and post-process ---
    frames = [f for f in [tles_local, tles_remote] if len(f) > 0]
    if not frames:
        return pd.DataFrame()

    tles = pd.concat(frames)
    tles.reset_index(drop=True, inplace=True)
    tles["EPOCH_JD"] = tles["EPOCH"].apply(lambda x: Time(spacetrack_time_to_isot(x)).jd)
    tles = _type_cast_tles(tles)
    tles = _get_closest_times(tles, epoch_jd)
    return tles


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def spacetrack_time_to_isot(spacetrack_time: str) -> str:
    """Convert a Space-Track epoch string to ISOT format."""
    if "T" in spacetrack_time:
        return spacetrack_time if "." in spacetrack_time else spacetrack_time + ".000"
    dt = datetime.strptime(spacetrack_time, "%Y-%m-%d %H:%M:%S")
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000")


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


def _type_cast_tles(tles: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "NORAD_CAT_ID", "EPOCH_MICROSECONDS", "MEAN_MOTION", "ECCENTRICITY",
        "INCLINATION", "RA_OF_ASC_NODE", "ARG_OF_PERICENTER", "MEAN_ANOMALY",
        "EPHEMERIS_TYPE", "ELEMENT_SET_NO", "REV_AT_EPOCH", "BSTAR",
        "MEAN_MOTION_DOT", "MEAN_MOTION_DDOT", "FILE", "OBJECT_NUMBER",
        "SEMIMAJOR_AXIS", "PERIOD", "APOGEE", "PERIGEE",
    ]
    for col in numeric_cols:
        if col in tles.columns:
            tles[col] = pd.to_numeric(tles[col])
    if "DECAYED" in tles.columns:
        tles["DECAYED"] = pd.to_numeric(tles["DECAYED"]).astype(bool)
    return tles
