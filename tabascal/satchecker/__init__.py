"""IAU CPS SatChecker access for tabascal, split by responsibility.

- :mod:`tabascal.satchecker.client` — HTTP transport and response normalisation;
  no tabascal/JAX/casacore imports, so it can be extracted later.
- :mod:`tabascal.satchecker.cache` — deterministic canonical-epoch cache key, the
  :class:`CatalogueCache` interface, and a text (JSON) implementation.
- :mod:`tabascal.satchecker.service` — endpoint fallback, response validation,
  throttled per-satellite acquisition, and resilient cache writes.

TABASCAL-specific orchestration (source precedence, ``extra_tle_dir`` age policy,
remote-age acceptance, and complete-coverage enforcement) lives in
:mod:`tabascal.tle`.

The names most callers need are re-exported here so ``from tabascal import
satchecker`` and ``satchecker.fetch_full_catalogue(...)`` keep working.
"""

from .client import (
    BASE_URL,
    CATALOGUE_COLUMNS,
    CatalogueResult,
    EmptyCatalogueError,
    SatCheckerError,
    SatCheckerResponseError,
    SatCheckerTransportError,
    catalogue_info,
    catalogue_total,
    fetch_full_catalogue,
    fetch_nearest_tle,
)
from .cache import (
    DEFAULT_CATALOGUE_INTERVAL_HOURS,
    PROVISIONAL,
    STABLE,
    CacheValidationError,
    CatalogueCache,
    CatalogueSnapshot,
    TextCatalogueCache,
    canonical_epoch_jd,
    canonical_stamp,
    catalogue_state,
    read_legacy_tle_records,
    utc_now_jd,
)
from .service import ensure_snapshot, fetch_nearest_batch, store_or_warn

__all__ = [
    "BASE_URL",
    "CATALOGUE_COLUMNS",
    "CatalogueResult",
    "EmptyCatalogueError",
    "SatCheckerError",
    "SatCheckerResponseError",
    "SatCheckerTransportError",
    "catalogue_info",
    "catalogue_total",
    "fetch_full_catalogue",
    "fetch_nearest_tle",
    "DEFAULT_CATALOGUE_INTERVAL_HOURS",
    "PROVISIONAL",
    "STABLE",
    "CacheValidationError",
    "CatalogueCache",
    "CatalogueSnapshot",
    "TextCatalogueCache",
    "canonical_epoch_jd",
    "canonical_stamp",
    "catalogue_state",
    "read_legacy_tle_records",
    "utc_now_jd",
    "ensure_snapshot",
    "fetch_nearest_batch",
    "store_or_warn",
]
