"""IAU CPS SatChecker access for tabascal, split by responsibility.

- :mod:`tabascal.satchecker.client` — HTTP transport and response normalisation
  for both nearest-record endpoints; no tabascal/JAX/casacore imports, so it can
  be extracted later.
- :mod:`tabascal.satchecker.tle_parse` — TLE line parsing, and the element range
  checks both record kinds share.
- :mod:`tabascal.satchecker.records` — what a record is, when it is valid, and
  what it means; the only place either format is named.
- :mod:`tabascal.satchecker.cache` — validated per-NORAD record storage.
- :mod:`tabascal.satchecker.service` — endpoint selection, bounded concurrent
  acquisition, response validation, and resilient cache writes.

TABASCAL-specific orchestration (source precedence, ``extra_orbit_dir`` age policy,
remote-age acceptance, and complete-coverage enforcement) lives in
:mod:`tabascal.orbit`.

The names most callers need are re-exported here.
"""

from .client import (
    BASE_URL,
    HANDOVER_JD,
    OMM_COLUMNS,
    TLE_COLUMNS,
    SatCheckerError,
    SatCheckerRateLimitError,
    SatCheckerResponseError,
    SatCheckerTransportError,
    fetch_nearest_omm,
    fetch_nearest_tle,
)
from .cache import (
    CacheValidationError,
    TextOrbitCache,
    read_legacy_tle_records,
)
from .records import (
    KIND_OMM,
    KIND_TLE,
    KIND_FIELD,
    RecordKindError,
    record_elements,
    record_epoch_jd,
    record_kind,
    validate_record,
)
from .service import (
    MAX_WORKERS,
    NearestBatchResult,
    fetch_nearest_batch,
    nearest_endpoints_for,
    store_or_warn,
)

__all__ = [
    "BASE_URL",
    "HANDOVER_JD",
    "OMM_COLUMNS",
    "TLE_COLUMNS",
    "fetch_nearest_omm",
    "nearest_endpoints_for",
    "SatCheckerError",
    "SatCheckerRateLimitError",
    "SatCheckerResponseError",
    "SatCheckerTransportError",
    "fetch_nearest_tle",
    "CacheValidationError",
    "TextOrbitCache",
    "read_legacy_tle_records",
    "KIND_OMM",
    "KIND_TLE",
    "KIND_FIELD",
    "RecordKindError",
    "record_elements",
    "record_epoch_jd",
    "record_kind",
    "validate_record",
    "MAX_WORKERS",
    "NearestBatchResult",
    "fetch_nearest_batch",
    "store_or_warn",
]
