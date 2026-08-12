"""IAU CPS SatChecker access for tabascal, split by responsibility.

- :mod:`tabascal.satchecker.client` — HTTP transport and response normalisation;
  no tabascal/JAX/casacore imports, so it can be extracted later.
- :mod:`tabascal.satchecker.cache` — validated per-NORAD TLE storage.
- :mod:`tabascal.satchecker.service` — bounded concurrent acquisition, response
  validation, and resilient cache writes.

TABASCAL-specific orchestration (source precedence, ``extra_tle_dir`` age policy,
remote-age acceptance, and complete-coverage enforcement) lives in
:mod:`tabascal.tle`.

The names most callers need are re-exported here.
"""

from .client import (
    BASE_URL,
    TLE_COLUMNS,
    SatCheckerError,
    SatCheckerResponseError,
    SatCheckerTransportError,
    fetch_nearest_tle,
)
from .cache import (
    CacheValidationError,
    TextTLECache,
    read_legacy_tle_records,
)
from .service import MAX_WORKERS, NearestBatchResult, fetch_nearest_batch, store_or_warn

__all__ = [
    "BASE_URL",
    "TLE_COLUMNS",
    "SatCheckerError",
    "SatCheckerResponseError",
    "SatCheckerTransportError",
    "fetch_nearest_tle",
    "CacheValidationError",
    "TextTLECache",
    "read_legacy_tle_records",
    "MAX_WORKERS",
    "NearestBatchResult",
    "fetch_nearest_batch",
    "store_or_warn",
]
