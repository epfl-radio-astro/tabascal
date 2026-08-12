"""Client for the IAU CPS SatChecker nearest-TLE service.

Self-contained transport layer: this module talks to the SatChecker HTTP API and
returns pandas DataFrames with a normalised column set. No account or credentials
are required. It knows nothing about tabascal's caching, cache-key policy, or
orbital-element parsing — those live in :mod:`tabascal.satchecker.cache` and
:mod:`tabascal.tle`. It deliberately imports nothing from tabascal, JAX, or
casacore so it stays a candidate for extraction into a standalone client.

Endpoint used: ``GET /tools/get-nearest-tle/`` for the TLE whose epoch is
closest to the requested epoch for one satellite.

Returned frames use these columns (SatChecker fields mapped to the OMM-style
names the rest of tabascal expects): ``NORAD_CAT_ID``, ``OBJECT_NAME``,
``EPOCH``, ``TLE_LINE1``, ``TLE_LINE2``, ``DATA_SOURCE``, ``DATE_COLLECTED``.
"""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://satchecker.cps.iau.org/tools"
# Identify ourselves to the SatChecker operators, with a contact URL.
USER_AGENT = "tabascal-tle/1.0 (+https://github.com/epfl-radio-astro/tabascal)"
REQUEST_TIMEOUT = 120  # seconds

# Columns the normalised TLE frames expose.
TLE_COLUMNS = [
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
    """Raised when SatChecker cannot be reached or returns no usable data.

    The two subclasses below separate the failures a caller can usefully route
    around from the ones it cannot; catching this base class treats them alike and
    is only appropriate at a top-level boundary.
    """


class SatCheckerTransportError(SatCheckerError):
    """The service could not be reached: connection, TLS, timeout or mid-read failure.

    Whole-service, not per-request: every other satellite's lookup would go to the
    same unreachable host. :func:`tabascal.satchecker.service.fetch_nearest_batch`
    treats the first one as the answer for the whole batch and abandons the
    requests still queued, rather than paying the request timeout once per
    configured satellite to learn the same thing.
    """


class SatCheckerResponseError(SatCheckerError):
    """The service answered, but the response is unusable: malformed or incomplete.

    Per-request, not whole-service: the host is up and answering, so it says
    nothing about the other satellites. Callers record it against the one ID and
    carry on with the rest of the batch.
    """


# ---------------------------------------------------------------------------
# Low-level HTTP + normalisation
# ---------------------------------------------------------------------------

# HTTP statuses that mean service-level backoff rather than a bad individual request.
_BACKOFF_STATUSES = frozenset({429})


def _status_error(url: str, error: urllib.error.HTTPError) -> SatCheckerError:
    """Classify an HTTP status response as a response or a transport failure.

    An HTTP status means the service *answered*, so it is not automatically a
    transport failure — but it is only worth trying a different endpoint when the
    server was rejecting this particular request rather than failing wholesale:

    * 4xx (except 429) — this individual request was rejected.
    * 429 and 5xx — the service is rate-limiting or failing server-side.
    """
    status = getattr(error, "code", None)
    detail = f"SatChecker returned HTTP {status} ({url}): {error.reason}"
    if status is not None and 400 <= status < 500 and status not in _BACKOFF_STATUSES:
        return SatCheckerResponseError(detail)
    return SatCheckerTransportError(detail)


def _http_get(url: str, timeout: int = REQUEST_TIMEOUT) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        # Must precede URLError: HTTPError subclasses it, and unlike its siblings
        # it means the service replied rather than that it could not be reached.
        raise _status_error(url, e) from e
    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
        http.client.HTTPException,
    ) as e:
        raise SatCheckerTransportError(f"SatChecker request failed ({url}): {e}") from e


def _load_json(raw: bytes, url: str):
    """Parse a JSON response, wrapping malformed payloads as ``SatCheckerError``."""
    try:
        return json.loads(raw)
    except (ValueError, TypeError) as e:
        raise SatCheckerResponseError(f"SatChecker returned invalid JSON ({url}): {e}") from e


def _as_object(payload, url: str) -> dict:
    """Return the JSON object carrying the response fields.

    SatChecker sometimes wraps the payload in a single-element list. An empty list
    or a scalar is a malformed response for the endpoints that read named fields,
    so raise ``SatCheckerError`` rather than letting a raw ``IndexError`` /
    ``AttributeError`` escape.
    """
    obj = payload[0] if isinstance(payload, list) and payload else payload
    if not isinstance(obj, dict):
        raise SatCheckerResponseError(
            f"SatChecker returned an unexpected response shape ({url}): "
            f"{type(payload).__name__}"
        )
    return obj


def _normalise(records: pd.DataFrame) -> pd.DataFrame:
    """Rename SatChecker fields to the normalised TLE columns.

    Response rows are validated here so schema problems surface as
    :class:`SatCheckerError` (the module's error contract) rather than as raw
    pandas exceptions: satellite IDs must be present and numeric, and both TLE
    lines must be present.
    """
    df = records.rename(columns=_FIELD_RENAME)
    for col in TLE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[TLE_COLUMNS].copy()
    try:
        ids = pd.to_numeric(df["NORAD_CAT_ID"])
    except (ValueError, TypeError) as e:
        raise SatCheckerResponseError(
            f"SatChecker response has non-numeric satellite IDs: {e}"
        ) from e
    if ids.isnull().any():
        raise SatCheckerResponseError(
            "SatChecker response is missing satellite IDs (satellite_id)"
        )
    # Require finite integers before casting: a fractional ID would silently
    # truncate to a different satellite, and infinities raise inside astype().
    if not np.isfinite(ids.to_numpy(dtype=float)).all():
        raise SatCheckerResponseError("SatChecker response has non-finite satellite IDs")
    if (ids != ids.round()).any():
        bad = ids[ids != ids.round()].unique()[:5]
        raise SatCheckerResponseError(
            f"SatChecker response has non-integer satellite IDs: {list(bad)}"
        )
    try:
        df["NORAD_CAT_ID"] = ids.astype(int)
    except (ValueError, TypeError, OverflowError) as e:
        raise SatCheckerResponseError(f"SatChecker satellite IDs are not usable: {e}") from e
    for col in ("TLE_LINE1", "TLE_LINE2"):
        if df[col].isnull().any():
            raise SatCheckerResponseError(f"SatChecker response is missing {col} values")
    return df.reset_index(drop=True)


def fetch_nearest_tle(norad_id: int, epoch_jd: float) -> pd.DataFrame:
    """Fetch the single TLE nearest *epoch_jd* for one satellite.

    Returns an empty DataFrame if SatChecker has no record for the satellite.
    """
    url = f"{BASE_URL}/get-nearest-tle/?" + urllib.parse.urlencode(
        {"id": int(norad_id), "id_type": "catalog", "epoch": repr(float(epoch_jd))}
    )
    payload = _load_json(_http_get(url), url)
    if isinstance(payload, list) and not payload:
        return pd.DataFrame()  # empty list == no record for this satellite
    obj = _as_object(payload, url)
    rows = obj.get("orbital_data") or obj.get("tle_data") or []
    if not rows:
        return pd.DataFrame()
    # The endpoint normally returns a list of row objects, but accepting a single
    # row object costs nothing and keeps pandas' raw "all scalar values" ValueError
    # from escaping the client's SatCheckerError contract.
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise SatCheckerResponseError(
            f"SatChecker returned unexpected nearest-TLE rows ({url}): "
            f"{type(rows).__name__}"
        )
    try:
        records = pd.DataFrame.from_records(rows)
    except (ValueError, TypeError) as e:
        raise SatCheckerResponseError(
            f"SatChecker nearest-TLE rows could not be read ({url}): {e}"
        ) from e
    return _normalise(records)
