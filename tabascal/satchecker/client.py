"""Client for the IAU CPS SatChecker nearest-element service.

Self-contained transport layer: this module talks to the SatChecker HTTP API and
returns pandas DataFrames with a normalised column set. No account or credentials
are required. It knows nothing about tabascal's caching, cache-key policy, or
source-precedence policy — those live in :mod:`tabascal.satchecker.cache` and
:mod:`tabascal.tle`. It deliberately imports nothing from tabascal, JAX, or
casacore so it stays a candidate for extraction into a standalone client.

Two endpoints, because SatChecker keeps two archives:

``GET /tools/get-nearest-tle/``
    The TLE whose epoch is closest to the requested one. This archive is frozen:
    its last record is from 2026-07-11 and it will never gain another.
``GET /tools/get-nearest-omm/``
    The OMM element set whose epoch is closest to the requested one. This
    archive begins at the handover, twelve hours after the TLE archive ends, and
    grows forward only.

SatChecker 1.7.0 made the split because Celestrak is dropping Alpha-5 notation
to preserve the original TLE format, which leaves catalogue numbers above 99999
with no TLE representation at all. Which endpoint to ask is
:mod:`tabascal.satchecker.service`'s decision, not this module's.

Neither archive reports "I have nothing that old". ``get-nearest-omm`` answers a
2021 request with its earliest 2026-07-11 record, 4.6 years off epoch, and says
nothing about the discrepancy. Callers are expected to check the epoch they got
against the epoch they asked for; this module only reports what came back.

Returned frames use OMM-style column names throughout. Both kinds carry
``NORAD_CAT_ID``, ``OBJECT_NAME``, ``EPOCH``, ``DATA_SOURCE``,
``DATE_COLLECTED`` and ``RECORD_KIND``; TLE frames add ``TLE_LINE1`` /
``TLE_LINE2``, and OMM frames add ``OBJECT_ID`` and the seven element columns.
"""

from __future__ import annotations

import email.utils
import http.client
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from .records import KIND_FIELD, KIND_OMM, KIND_TLE, OMM_ELEMENT_COLUMNS


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

# Columns the normalised OMM frames expose.
OMM_COLUMNS = [
    "NORAD_CAT_ID",
    "OBJECT_NAME",
    "OBJECT_ID",
    "EPOCH",
    *OMM_ELEMENT_COLUMNS,
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

# The same, for the OMM rows. The row-level ``epoch`` is deliberately *not*
# mapped: SatChecker spells it "2026-08-13 03:34:14 UTC" there, which is neither
# ISO 8601 nor sub-second, while the nested element object carries the same
# instant as "2026-08-13T03:34:14.082240". The nested one is what we lift.
_FIELD_RENAME_OMM = {
    "satellite_id": "NORAD_CAT_ID",
    "satellite_name": "OBJECT_NAME",
    "data_source": "DATA_SOURCE",
    "date_collected": "DATE_COLLECTED",
}

# Fields lifted out of each row's nested ``orbital_elements`` object. SatChecker
# already names them in OMM style, so this is a move rather than a translation.
# The object also carries CLASSIFICATION_TYPE, ELEMENT_SET_NO, EPHEMERIS_TYPE,
# MEAN_MOTION_DOT, MEAN_MOTION_DDOT and REV_AT_EPOCH; nothing downstream reads
# them, so they are dropped rather than stored and never used.
_OMM_LIFTED_FIELDS = ("EPOCH", "OBJECT_ID", *OMM_ELEMENT_COLUMNS)


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


class SatCheckerRateLimitError(SatCheckerTransportError):
    """The service asked this client to slow down (HTTP 429).

    A *transport* error by classification, because the thing it tells us is about
    the service and not about the satellite we happened to ask for: the next
    request is unwelcome too. Being a subclass is what stops a batch dead on the
    first 429 instead of working through the rest of the list.

    ``retry_after`` carries the service's own ``Retry-After`` hint in seconds when
    it sends one, so the message can say when the run is worth repeating. TABASCAL
    reports it rather than sleeping on it: an unattended preflight that quietly
    blocks for an interval the service chose is worse than one that stops and says
    why.
    """

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class SatCheckerResponseError(SatCheckerError):
    """The service answered, but the response is unusable: malformed or incomplete.

    Per-request, not whole-service: the host is up and answering, so it says
    nothing about the other satellites. Callers record it against the one ID and
    carry on with the rest of the batch.

    ``status`` is the HTTP status when the failure came from one (``None`` for a
    malformed body). A run of identical statuses with no success in between is
    how a caller recognises that "per-request" has stopped being true — a WAF
    answering 403 to everything, or a renamed endpoint answering 404.
    """

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# Low-level HTTP + normalisation
# ---------------------------------------------------------------------------

# HTTP statuses that mean service-level backoff rather than a bad individual request.
_BACKOFF_STATUSES = frozenset({429})


def _retry_after_seconds(error: urllib.error.HTTPError) -> Optional[float]:
    """Seconds to wait, from a ``Retry-After`` header in either permitted form.

    RFC 9110 allows delta-seconds (``120``) or an HTTP-date
    (``Wed, 21 Oct 2026 07:28:00 GMT``); both appear in the wild. An absent,
    malformed or already-elapsed value yields ``None`` / ``0.0`` rather than an
    exception — a bad hint must never turn into a second failure on top of the
    one being reported.
    """
    headers = getattr(error, "headers", None)
    raw = headers.get("Retry-After") if headers is not None else None
    if not raw:
        return None
    raw = str(raw).strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        stamp = email.utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if stamp is None:
        return None
    if stamp.tzinfo is None:  # an HTTP-date without a zone is GMT
        stamp = stamp.replace(tzinfo=timezone.utc)
    return max(0.0, (stamp - datetime.now(timezone.utc)).total_seconds())


def _status_error(url: str, error: urllib.error.HTTPError) -> SatCheckerError:
    """Classify an HTTP status response as a response or a transport failure.

    An HTTP status means the service *answered*, so it is not automatically a
    transport failure — but it is only worth trying a different endpoint when the
    server was rejecting this particular request rather than failing wholesale:

    * 4xx (except 429) — this individual request was rejected.
    * 429 — the service is asking this client to back off, and says so about the
      client rather than about the satellite requested.
    * 5xx — the service is failing server-side.
    """
    status = getattr(error, "code", None)
    detail = f"SatChecker returned HTTP {status} ({url}): {error.reason}"
    if status in _BACKOFF_STATUSES:
        retry_after = _retry_after_seconds(error)
        hint = (
            f"; it asks for {retry_after:g} s before the next request"
            if retry_after is not None
            else ""
        )
        return SatCheckerRateLimitError(
            f"SatChecker returned HTTP {status} — it is rate-limiting this "
            f"client{hint} ({url}): {error.reason}",
            retry_after=retry_after,
        )
    if status is not None and 400 <= status < 500:
        return SatCheckerResponseError(detail, status=status)
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


def _checked_ids(df: pd.DataFrame) -> pd.Series:
    """Satellite IDs from a normalised frame, as usable integers.

    Shared by both kinds: whatever the record format, an ID that is absent,
    non-numeric, non-finite or fractional makes the row unusable in the same way
    — a fractional ID would silently truncate to a *different* satellite, and an
    infinity raises inside ``astype``.
    """
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
        return ids.astype(int)
    except (ValueError, TypeError, OverflowError) as e:
        raise SatCheckerResponseError(f"SatChecker satellite IDs are not usable: {e}") from e


def _project(records: pd.DataFrame, columns: list[str], kind: str) -> pd.DataFrame:
    """Reduce a raw frame to *columns*, stamping the record kind onto every row.

    Missing columns are filled rather than raising, so a response that simply
    omits an optional field stays usable; the required ones are checked by the
    per-kind normalisers below.
    """
    df = records.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = None
    df = df[columns].copy()
    # Stated rather than inferred. Inference exists for user-supplied files that
    # cannot carry the field; a response we parsed ourselves knows what it is.
    df[KIND_FIELD] = kind
    return df


def _normalise(records: pd.DataFrame) -> pd.DataFrame:
    """Rename SatChecker fields to the normalised TLE columns.

    Response rows are validated here so schema problems surface as
    :class:`SatCheckerError` (the module's error contract) rather than as raw
    pandas exceptions: satellite IDs must be present and numeric, and both TLE
    lines must be present.
    """
    df = _project(records.rename(columns=_FIELD_RENAME), TLE_COLUMNS, KIND_TLE)
    df["NORAD_CAT_ID"] = _checked_ids(df)
    for col in ("TLE_LINE1", "TLE_LINE2"):
        if df[col].isnull().any():
            raise SatCheckerResponseError(f"SatChecker response is missing {col} values")
    return df.reset_index(drop=True)


def _lift_orbital_elements(rows: list[dict], url: str) -> list[dict]:
    """Flatten each row's nested ``orbital_elements`` object onto the row itself.

    SatChecker nests the elements one level down and already names them in OMM
    style, so this is a move rather than a translation. Doing it before the
    frame is built keeps a nested ``dict`` out of a pandas cell, where it would
    survive every column check and only fail much later.
    """
    lifted = []
    for row in rows:
        nested = row.get("orbital_elements")
        if not isinstance(nested, dict):
            raise SatCheckerResponseError(
                f"SatChecker OMM row has no orbital_elements object ({url}): "
                f"{type(nested).__name__}"
            )
        flat = {key: value for key, value in row.items() if key != "orbital_elements"}
        for field in _OMM_LIFTED_FIELDS:
            if field in nested:
                flat[field] = nested[field]
        lifted.append(flat)
    return lifted


def _normalise_omm(records: pd.DataFrame) -> pd.DataFrame:
    """Rename SatChecker fields to the normalised OMM columns.

    The same contract as :func:`_normalise` — schema problems surface as
    :class:`SatCheckerError` — over a different required set. There is no
    checksum and no second copy of the identifier to verify here; range and
    finiteness checks on the elements happen in
    :mod:`tabascal.satchecker.records`, which is also where the reasoning about
    that gap lives.
    """
    df = _project(records.rename(columns=_FIELD_RENAME_OMM), OMM_COLUMNS, KIND_OMM)
    df["NORAD_CAT_ID"] = _checked_ids(df)
    for col in ("EPOCH", *OMM_ELEMENT_COLUMNS):
        if df[col].isnull().any():
            raise SatCheckerResponseError(f"SatChecker response is missing {col} values")
    return df.reset_index(drop=True)


def _fetch_nearest_rows(endpoint: str, norad_id: int, epoch_jd: float):
    """Row dicts from one of the ``get-nearest-*`` endpoints, plus the URL used.

    Returns ``(None, url)`` when the service has no record for the satellite —
    which it signals either as an empty top-level list or as an empty
    ``orbital_data``, both observed. Note that "no record" here means *no record
    at all*: neither endpoint reports that it has nothing near the epoch asked
    for. ``get-nearest-omm`` answers a 2021 request with its earliest 2026
    record. Judging the epoch is the caller's job.
    """
    url = f"{BASE_URL}/{endpoint}/?" + urllib.parse.urlencode(
        {"id": int(norad_id), "id_type": "catalog", "epoch": repr(float(epoch_jd))}
    )
    payload = _load_json(_http_get(url), url)
    if isinstance(payload, list) and not payload:
        return None, url  # empty list == no record for this satellite
    obj = _as_object(payload, url)
    rows = obj.get("orbital_data") or obj.get("tle_data") or []
    if not rows:
        return None, url
    # The endpoint normally returns a list of row objects, but accepting a single
    # row object costs nothing and keeps pandas' raw "all scalar values" ValueError
    # from escaping the client's SatCheckerError contract.
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise SatCheckerResponseError(
            f"SatChecker returned unexpected {endpoint} rows ({url}): "
            f"{type(rows).__name__}"
        )
    return rows, url


def _frame(rows: list[dict], endpoint: str, url: str) -> pd.DataFrame:
    try:
        return pd.DataFrame.from_records(rows)
    except (ValueError, TypeError) as e:
        raise SatCheckerResponseError(
            f"SatChecker {endpoint} rows could not be read ({url}): {e}"
        ) from e


def fetch_nearest_tle(norad_id: int, epoch_jd: float) -> pd.DataFrame:
    """Fetch the single TLE nearest *epoch_jd* for one satellite.

    Returns an empty DataFrame if SatChecker has no record for the satellite.
    Note that the TLE archive is frozen at 2026-07-11, so for any observation
    after that this returns the last TLE ever published for the satellite,
    however far from the requested epoch that is.
    """
    rows, url = _fetch_nearest_rows("get-nearest-tle", norad_id, epoch_jd)
    if rows is None:
        return pd.DataFrame()
    return _normalise(_frame(rows, "get-nearest-tle", url))


def fetch_nearest_omm(norad_id: int, epoch_jd: float) -> pd.DataFrame:
    """Fetch the single OMM element set nearest *epoch_jd* for one satellite.

    Returns an empty DataFrame if SatChecker has no record for the satellite.
    The OMM archive begins at the 2026-07-12 handover, and a request for an
    earlier epoch is answered with its *earliest* record rather than with
    nothing — so a pre-handover caller gets a confident-looking element set that
    may be years off. The age ceiling in :mod:`tabascal.tle` is what rejects it.
    """
    rows, url = _fetch_nearest_rows("get-nearest-omm", norad_id, epoch_jd)
    if rows is None:
        return pd.DataFrame()
    lifted = _lift_orbital_elements(rows, url)
    return _normalise_omm(_frame(lifted, "get-nearest-omm", url))
