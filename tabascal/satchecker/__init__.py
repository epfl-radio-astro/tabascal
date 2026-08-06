"""IAU CPS SatChecker access for tabascal, split by responsibility.

- :mod:`tabascal.satchecker.client` — HTTP transport and response normalisation;
  no tabascal/JAX/casacore imports, so it can be extracted later.
- :mod:`tabascal.satchecker.cache` — deterministic canonical-epoch cache key, the
  :class:`CatalogueCache` interface, and a text (JSON) implementation.

TABASCAL-specific orchestration (source precedence, ``extra_tle_dir`` age policy,
local orbital-element parsing) lives in :mod:`tabascal.tle`.

The names most callers need are re-exported here so ``from tabascal import
satchecker`` and ``satchecker.fetch_full_catalogue(...)`` keep working.
"""

from tabascal.satchecker.client import (
    BASE_URL,
    CATALOGUE_COLUMNS,
    CatalogueResult,
    EmptyCatalogueError,
    SatCheckerError,
    catalogue_info,
    catalogue_total,
    fetch_full_catalogue,
    fetch_nearest_tle,
)
from tabascal.satchecker.cache import (
    CacheValidationError,
    CatalogueCache,
    CatalogueSnapshot,
    TextCatalogueCache,
    canonical_epoch_jd,
    canonical_stamp,
    read_legacy_tle_records,
)

__all__ = [
    "BASE_URL",
    "CATALOGUE_COLUMNS",
    "CatalogueResult",
    "EmptyCatalogueError",
    "SatCheckerError",
    "catalogue_info",
    "catalogue_total",
    "fetch_full_catalogue",
    "fetch_nearest_tle",
    "CacheValidationError",
    "CatalogueCache",
    "CatalogueSnapshot",
    "TextCatalogueCache",
    "canonical_epoch_jd",
    "canonical_stamp",
    "read_legacy_tle_records",
]
