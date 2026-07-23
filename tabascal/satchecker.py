"""Client for the IAU CPS SatChecker TLE service (https://satchecker.cps.iau.org/).

Self-contained: this module talks to the SatChecker HTTP API and returns pandas
DataFrames with a normalised column set. No account or credentials are required.
It knows nothing about tabascal's caching or orbital-element parsing — that lives
in :mod:`tabascal.tle`.

Endpoints used:
  - ``GET /tools/tles-at-epoch/``    full TLE catalogue nearest an epoch (zip/json)
  - ``GET /tools/get-nearest-tle/``  single nearest TLE for one satellite

Returned frames use these columns (SatChecker fields mapped to the OMM-style
names the rest of tabascal expects): ``NORAD_CAT_ID``, ``OBJECT_NAME``,
``EPOCH``, ``TLE_LINE1``, ``TLE_LINE2``, ``DATA_SOURCE``, ``DATE_COLLECTED``.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse
import urllib.request
import zipfile

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://satchecker.cps.iau.org/tools"
USER_AGENT = "tabascal-tle/1.0"
REQUEST_TIMEOUT = 300  # seconds — the full catalogue zip is a few MB
CATALOGUE_MIN_FRACTION = 0.99  # accept a zip download this complete vs total_results

# Columns the normalised TLE frames expose.
CATALOGUE_COLUMNS = [
    "NORAD_CAT_ID",
    "OBJECT_NAME",
    "EPOCH",
    "TLE_LINE1",
    "TLE_LINE2",
    "DATA_SOURCE",
    "DATE_COLLECTED",
]

# SatChecker response field -> normalised column name.
_FIELD_RENAME = {
    "satellite_id": "NORAD_CAT_ID",
    "satellite_name": "OBJECT_NAME",
    "epoch": "EPOCH",
    "tle_line1": "TLE_LINE1",
    "tle_line2": "TLE_LINE2",
    "data_source": "DATA_SOURCE",
    "date_collected": "DATE_COLLECTED",
}


class SatCheckerError(RuntimeError):
    """Raised when SatChecker cannot be reached or returns no usable data."""


# ---------------------------------------------------------------------------
# Low-level HTTP + normalisation
# ---------------------------------------------------------------------------

def _http_get(url: str, timeout: int = REQUEST_TIMEOUT) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (urllib.error.URLError, TimeoutError) as e:
        raise SatCheckerError(f"SatChecker request failed ({url}): {e}") from e


def _normalise(records: pd.DataFrame) -> pd.DataFrame:
    """Rename SatChecker fields to the normalised catalogue columns."""
    df = records.rename(columns=_FIELD_RENAME)
    for col in CATALOGUE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[CATALOGUE_COLUMNS].copy()
    df["NORAD_CAT_ID"] = pd.to_numeric(df["NORAD_CAT_ID"]).astype(int)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Catalogue endpoints
# ---------------------------------------------------------------------------

def catalogue_total(epoch_jd: float) -> int:
    """Number of catalogue records reported at *epoch_jd* (JSON ``total_results``)."""
    q = urllib.parse.urlencode(
        {"epoch": repr(float(epoch_jd)), "format": "json", "per_page": 1, "page": 1}
    )
    payload = json.loads(_http_get(f"{BASE_URL}/tles-at-epoch/?{q}", timeout=120))
    obj = payload[0] if isinstance(payload, list) else payload
    return int(obj.get("total_results", 0))


def _fetch_catalogue_zip(epoch_jd: float) -> pd.DataFrame:
    q = urllib.parse.urlencode({"epoch": repr(float(epoch_jd)), "format": "zip"})
    raw = _http_get(f"{BASE_URL}/tles-at-epoch/?{q}")
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = zf.namelist()
        if not names:
            raise SatCheckerError("SatChecker returned an empty zip archive.")
        with zf.open(names[0]) as f:
            df = pd.read_csv(f)
    return _normalise(df)


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
        payload = json.loads(_http_get(f"{BASE_URL}/tles-at-epoch/?{q}"))
        obj = payload[0] if isinstance(payload, list) else payload
        rows = obj.get("data", [])
        if rows:
            frames.append(pd.DataFrame(rows))
        total = int(obj.get("total_results", 0))
        if page * per_page >= total or not rows:
            break
        page += 1

    if not frames:
        raise SatCheckerError("SatChecker returned no TLE records.")
    return _normalise(pd.concat(frames, ignore_index=True))


def fetch_full_catalogue(epoch_jd: float) -> pd.DataFrame:
    """Download the full TLE catalogue nearest *epoch_jd* from SatChecker.

    Uses the efficient ``format=zip`` endpoint, validating the row count against
    the ``total_results`` reported by the JSON endpoint (the zip response is
    occasionally truncated). A short download is retried, then falls back to the
    paginated ``format=json`` endpoint.
    """
    try:
        expected = catalogue_total(epoch_jd)
    except Exception:
        expected = 0

    last_err: Exception | None = None
    for _ in range(2):
        try:
            df = _fetch_catalogue_zip(epoch_jd)
        except Exception as e:  # pragma: no cover - network dependent
            last_err = e
            continue
        if not expected or len(df) >= expected * CATALOGUE_MIN_FRACTION:
            return df
        last_err = SatCheckerError(
            f"SatChecker zip truncated ({len(df)} of {expected} records) — retrying"
        )
        print(f"  {last_err}")

    # zip repeatedly short or failing → paginated JSON (complete but slower)
    try:
        return _fetch_catalogue_json(epoch_jd)
    except Exception as json_err:
        raise SatCheckerError(
            f"Failed to fetch TLE catalogue from SatChecker: {json_err} "
            f"(zip attempt: {last_err})"
        ) from json_err


def fetch_nearest_tle(norad_id: int, epoch_jd: float) -> pd.DataFrame:
    """Fetch the single TLE nearest *epoch_jd* for one satellite.

    Returns an empty DataFrame if SatChecker has no record for the satellite.
    """
    q = urllib.parse.urlencode(
        {"id": int(norad_id), "id_type": "catalog", "epoch": repr(float(epoch_jd))}
    )
    payload = json.loads(_http_get(f"{BASE_URL}/get-nearest-tle/?{q}", timeout=120))
    obj = payload[0] if isinstance(payload, list) else payload
    rows = obj.get("orbital_data") or obj.get("tle_data") or []
    if not rows:
        return pd.DataFrame()
    return _normalise(pd.DataFrame(rows))
