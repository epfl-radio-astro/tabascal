"""TLE fetching, caching, and Space-Track credential handling for tabascal.

Provides the same caching behaviour as tabsim.tle so that tabascal can
eventually be used independently of tabsim.

TLE cache directory (default: tabascal/data/tles/):
  - JSON files are named  <YYYY-MM-DD>-<random-id>.json
  - On each call to get_tles_by_id the cache is checked first; only the
    NORAD IDs not found locally are fetched from Space-Track.

Space-Track credentials are searched in priority order:
  1. <tle_dir>/spacetrack_login.yaml
  2. ~/.credentials/spacetrack_login.yaml
  3. ./spacetrack_login.yaml   (current working directory)
"""

from __future__ import annotations

import json
import os
import random
import string
from datetime import datetime
from glob import glob
from importlib.resources import files
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from astropy.time import Time
from spacetrack import SpaceTrackClient
import spacetrack.operators as op


# ---------------------------------------------------------------------------
# TLE directory helpers
# ---------------------------------------------------------------------------

def make_tle_dir(tle_dir: Optional[str] = None) -> str:
    """Return the absolute path to the TLE cache directory, creating it if needed.

    Parameters
    ----------
    tle_dir:
        Explicit directory path.  When *None* the package-bundled default
        (``tabascal/data/tles/``) is used.
    """
    if tle_dir:
        tle_dir = os.path.abspath(tle_dir)
    else:
        tle_dir = files("tabascal.data").joinpath("tles").__str__()

    os.makedirs(tle_dir, exist_ok=True)
    return tle_dir


# ---------------------------------------------------------------------------
# Credential loading
# ---------------------------------------------------------------------------

def load_spacetrack_credentials(
    tle_dir: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Load Space-Track credentials from a YAML file.

    Searches the following locations in priority order:

    1. ``<tle_dir>/spacetrack_login.yaml``
    2. ``~/.credentials/spacetrack_login.yaml``
    3. ``./spacetrack_login.yaml`` (current working directory)

    Parameters
    ----------
    tle_dir:
        Directory that may contain ``spacetrack_login.yaml``.  When *None*
        the package-bundled TLE data directory is used.

    Returns
    -------
    tuple
        ``(username, password)`` on success, ``(None, None)`` if no
        credentials file is found.
    """
    tle_dir_path = make_tle_dir(tle_dir)

    search_paths = [
        os.path.join(tle_dir_path, "spacetrack_login.yaml"),
        os.path.join(os.path.expanduser("~"), ".credentials", "spacetrack_login.yaml"),
        os.path.join(os.getcwd(), "spacetrack_login.yaml"),
    ]

    for cred_path in search_paths:
        if os.path.exists(cred_path):
            try:
                with open(cred_path) as f:
                    creds = yaml.safe_load(f)
                username = creds.get("username")
                password = creds.get("password")
                if username and password:
                    print(f"Space-Track credentials loaded from : {cred_path}")
                    return username, password
            except Exception:
                print(f"Warning: Could not load credentials from {cred_path}")

    print("No Space-Track credentials loaded.")
    return None, None


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
    username: str,
    password: str,
    norad_ids: list[int],
    epoch_jd: float,
    window_days: float = 1.0,
    limit: int = 2000,
    tle_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch TLE records for the given NORAD IDs, using a local JSON cache.

    Already-cached records for the requested epoch date are loaded from disk;
    only the remaining IDs are fetched from Space-Track and then saved to the
    cache.

    Parameters
    ----------
    username, password:
        Space-Track credentials.
    norad_ids:
        NORAD catalogue IDs to fetch.
    epoch_jd:
        Julian date of the observation epoch (used to select the nearest TLE).
    window_days:
        Half-width of the time window used when querying Space-Track.
    limit:
        Maximum records per API request.
    tle_dir:
        Local cache directory.  Defaults to ``tabascal/data/tles/``.

    Returns
    -------
    pd.DataFrame
        One row per requested satellite, the row whose epoch is closest to
        *epoch_jd*.
    """
    tle_dir = make_tle_dir(tle_dir)
    norad_ids = list(np.array(list(set(norad_ids))).astype(int))
    n_ids_start = len(norad_ids)
    epoch_str = Time(epoch_jd, format="jd", scale="ut1").strftime("%Y-%m-%d")

    # --- load from cache ---
    tles_local = pd.DataFrame()
    tle_paths = glob(os.path.join(tle_dir, f"{epoch_str}-*.json"))
    local_ids: list[int] = []
    if tle_paths:
        tles_local = pd.concat([pd.read_json(p) for p in tle_paths])
        tles_local = tles_local[tles_local["NORAD_CAT_ID"].isin(norad_ids)]
        local_ids = list(tles_local["NORAD_CAT_ID"].unique())
        norad_ids = list(set(norad_ids) - set(local_ids))
    print(f"Local TLEs loaded  : {len(local_ids)}")

    # --- fetch missing IDs from Space-Track ---
    max_ids = 500
    tles_remote = pd.DataFrame()
    remote_ids: list[int] = []
    if norad_ids:
        client = get_space_track_client(username, password)
        chunks = [
            norad_ids[i : i + max_ids] for i in range(0, len(norad_ids), max_ids)
        ]
        frames = [fetch_tle_data(client, chunk, epoch_jd, window_days, limit) for chunk in chunks]
        non_empty = [f for f in frames if len(f) > 0]
        if non_empty:
            tles_remote = pd.concat(non_empty)
            tles_remote["Fetch_Timestamp"] = Time.now().fits
            remote_ids = list(tles_remote["NORAD_CAT_ID"].unique())

            save_path = os.path.join(tle_dir, f"{epoch_str}-{_random_id()}.json")
            tles_remote.to_json(save_path)
            print(f"Saving remotely obtained TLEs to {save_path}")

    print(f"Remote TLEs loaded : {len(remote_ids)}")
    print(f"TLEs not found     : {n_ids_start - len(remote_ids) - len(local_ids)}")

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
