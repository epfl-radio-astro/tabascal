"""Client for the IAU CPS SatChecker TLE service (https://satchecker.cps.iau.org/).

Self-contained transport layer: this module talks to the SatChecker HTTP API and
returns pandas DataFrames with a normalised column set. No account or credentials
are required. It knows nothing about tabascal's caching, cache-key policy, or
orbital-element parsing — those live in :mod:`tabascal.satchecker.cache` and
:mod:`tabascal.tle`. It deliberately imports nothing from tabascal, JAX, or
casacore so it stays a candidate for extraction into a standalone client.

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
from dataclasses import dataclass
from typing import Optional

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


@dataclass(frozen=True)
class CatalogueResult:
    """A full-catalogue download plus the metadata needed to validate/cache it.

    ``records`` carries the normalised TLE frame; the counts let the caller store
    (and later audit) how complete the download was, and ``service_version`` /
    ``source`` record where it came from. This is transport-level provenance only
    — no cache-key or file-format policy lives here.
    """

    records: pd.DataFrame
    expected_count: int
    actual_count: int
    service_version: Optional[str] = None
    source: str = "zip"  # "zip" or "json"


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


def _load_json(raw: bytes, url: str):
    """Parse a JSON response, wrapping malformed payloads as ``SatCheckerError``."""
    try:
        return json.loads(raw)
    except (ValueError, TypeError) as e:
        raise SatCheckerError(f"SatChecker returned invalid JSON ({url}): {e}") from e


def _as_object(payload, url: str) -> dict:
    """Return the JSON object carrying the response fields.

    SatChecker sometimes wraps the payload in a single-element list. An empty list
    or a scalar is a malformed response for the endpoints that read named fields,
    so raise ``SatCheckerError`` rather than letting a raw ``IndexError`` /
    ``AttributeError`` escape.
    """
    obj = payload[0] if isinstance(payload, list) and payload else payload
    if not isinstance(obj, dict):
        raise SatCheckerError(
            f"SatChecker returned an unexpected response shape ({url}): "
            f"{type(payload).__name__}"
        )
    return obj


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

def catalogue_info(epoch_jd: float) -> tuple[int, Optional[str]]:
    """Return ``(total_results, service_version)`` reported at *epoch_jd*.

    A single cheap JSON request (``per_page=1``) used to size and provenance-tag a
    subsequent full download. ``service_version`` is ``None`` when the service does
    not report one.
    """
    url = f"{BASE_URL}/tles-at-epoch/?" + urllib.parse.urlencode(
        {"epoch": repr(float(epoch_jd)), "format": "json", "per_page": 1, "page": 1}
    )
    payload = _load_json(_http_get(url, timeout=120), url)
    obj = _as_object(payload, url)
    version = obj.get("version") or obj.get("service_version") or obj.get("api_version")
    return int(obj.get("total_results", 0)), (str(version) if version else None)


def catalogue_total(epoch_jd: float) -> int:
    """Number of catalogue records reported at *epoch_jd* (JSON ``total_results``)."""
    return catalogue_info(epoch_jd)[0]


def _reject_repeated_rows(df: pd.DataFrame, source: str) -> None:
    """Reject *fully identical* TLE rows (same NORAD ID and both TLE lines).

    An identical repeated row is the signature of corrupted/repeated content (e.g.
    JSON pagination serving the same page twice), so it invalidates the download.
    Duplicate NORAD IDs with *different* TLE lines are deliberately tolerated: the
    service may legitimately carry more than one record per object near an epoch,
    and downstream selection already takes one record per ID.
    """
    dup = df.duplicated(subset=["NORAD_CAT_ID", "TLE_LINE1", "TLE_LINE2"])
    if dup.any():
        sample = ", ".join(str(int(x)) for x in df.loc[dup, "NORAD_CAT_ID"].unique()[:5])
        raise SatCheckerError(
            f"SatChecker {source} catalogue contains identical repeated TLE rows "
            f"(for example NORAD {sample}) — treating as corrupted content"
        )


def _fetch_catalogue_zip(epoch_jd: float) -> pd.DataFrame:
    q = urllib.parse.urlencode({"epoch": repr(float(epoch_jd)), "format": "zip"})
    raw = _http_get(f"{BASE_URL}/tles-at-epoch/?{q}")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            names = zf.namelist()
            if not names:
                raise SatCheckerError("SatChecker returned an empty zip archive.")
            with zf.open(names[0]) as f:
                df = pd.read_csv(f)
    except (zipfile.BadZipFile, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise SatCheckerError(f"SatChecker zip response could not be read: {e}") from e
    df = _normalise(df)
    _reject_repeated_rows(df, "zip")
    return df


def _fetch_catalogue_json(epoch_jd: float, per_page: int = 5000) -> pd.DataFrame:
    """Paginated JSON fallback for the full catalogue at *epoch_jd*.

    Completion is decided by *record count*, never by ``page * per_page``: the
    service's effective page size may be smaller than the value requested, so the
    accumulator pages until it has collected ``total_results`` rows. A response is
    rejected (``SatCheckerError``) — and therefore never cached — when it is
    incomplete or internally inconsistent:

    * ``total_results`` changes between pages;
    * a page returns no rows before the total is reached; or
    * the final accumulated count does not equal ``total_results``.
    """
    frames: list[pd.DataFrame] = []
    collected = 0
    total: Optional[int] = None
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
        url = f"{BASE_URL}/tles-at-epoch/?{q}"
        payload = _load_json(_http_get(url), url)
        obj = _as_object(payload, url)

        try:
            page_total = int(obj.get("total_results", 0))
        except (TypeError, ValueError) as e:
            raise SatCheckerError(
                f"SatChecker returned invalid total_results at page {page}: "
                f"{obj.get('total_results')!r}"
            ) from e
        if page_total < 0:
            raise SatCheckerError(
                f"SatChecker returned negative total_results at page {page}: {page_total}"
            )

        response_page = obj.get("page")
        if response_page is not None:
            try:
                response_page = int(response_page)
            except (TypeError, ValueError) as e:
                raise SatCheckerError(
                    f"SatChecker returned invalid page metadata: {response_page!r}"
                ) from e
            if response_page != page:
                raise SatCheckerError(
                    f"SatChecker returned page {response_page} when page {page} was requested"
                )
        if total is None:
            total = page_total
        elif page_total != total:
            raise SatCheckerError(
                f"SatChecker reported inconsistent total_results across pages "
                f"({total} then {page_total}) at page {page}"
            )

        rows = obj.get("data") or []
        if rows:
            frames.append(pd.DataFrame(rows))
            collected += len(rows)

        if total <= 0 or collected >= total:
            break
        if not rows:
            raise SatCheckerError(
                f"SatChecker JSON pagination returned an empty page at page {page} "
                f"after {collected} of {total} records — treating as truncated"
            )
        page += 1

    if not frames:
        raise SatCheckerError("SatChecker returned no TLE records.")
    df = _normalise(pd.concat(frames, ignore_index=True))
    if total is not None and len(df) != total:
        raise SatCheckerError(
            f"SatChecker JSON catalogue incomplete: {len(df)} of {total} records"
        )
    _reject_repeated_rows(df, "JSON")
    return df


def fetch_full_catalogue(epoch_jd: float) -> CatalogueResult:
    """Download the full TLE catalogue nearest *epoch_jd* from SatChecker.

    Uses the efficient ``format=zip`` endpoint, validating the row count against
    the ``total_results`` reported by the JSON endpoint (the zip response is
    occasionally truncated). A short download is retried, then falls back to the
    paginated ``format=json`` endpoint.

    Returns a :class:`CatalogueResult` carrying the normalised frame plus the
    expected/actual counts and service version, so the caller can decide whether
    the download is complete enough to cache and can record its provenance.
    """
    info_error: Exception | None = None
    try:
        expected, version = catalogue_info(epoch_jd)
    except Exception as e:
        expected, version = None, None
        info_error = e

    last_err: Exception | None = info_error
    # A ZIP has no independent completeness metadata. Only use it when the cheap
    # JSON info request supplied a positive expected count; otherwise go straight
    # to the self-validating paginated JSON path.
    if expected is not None and expected > 0:
        for _ in range(2):
            try:
                # raises on unreadable zips and identical repeated rows alike
                df = _fetch_catalogue_zip(epoch_jd)
            except Exception as e:
                last_err = e
                continue
            if len(df) >= expected * CATALOGUE_MIN_FRACTION:
                return CatalogueResult(df, expected, len(df), version, "zip")
            last_err = SatCheckerError(
                f"SatChecker zip truncated ({len(df)} of {expected} records) — retrying"
            )
            print(f"  {last_err}")

    # zip repeatedly short or failing → paginated JSON. _fetch_catalogue_json
    # guarantees an internally complete result (it raises otherwise), so the JSON
    # result is exactly complete: expected == actual == len(df).
    try:
        df = _fetch_catalogue_json(epoch_jd)
    except Exception as json_err:
        raise SatCheckerError(
            f"Failed to fetch TLE catalogue from SatChecker: {json_err} "
            f"(zip attempt: {last_err})"
        ) from json_err
    return CatalogueResult(df, len(df), len(df), version, "json")


def fetch_nearest_tle(norad_id: int, epoch_jd: float) -> pd.DataFrame:
    """Fetch the single TLE nearest *epoch_jd* for one satellite.

    Returns an empty DataFrame if SatChecker has no record for the satellite.
    """
    url = f"{BASE_URL}/get-nearest-tle/?" + urllib.parse.urlencode(
        {"id": int(norad_id), "id_type": "catalog", "epoch": repr(float(epoch_jd))}
    )
    payload = _load_json(_http_get(url, timeout=120), url)
    if isinstance(payload, list) and not payload:
        return pd.DataFrame()  # empty list == no record for this satellite
    obj = _as_object(payload, url)
    rows = obj.get("orbital_data") or obj.get("tle_data") or []
    if not rows:
        return pd.DataFrame()
    return _normalise(pd.DataFrame(rows))
