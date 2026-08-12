"""Resilient SatChecker acquisition built on the HTTP client and cache.

This module owns provider-specific orchestration: endpoint fallback, request
throttling, SatChecker failure classification, response-row filtering, and
cache writes. Applications can therefore consume this service without knowing
which SatChecker route answered or how its failures should be handled.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from . import client
from .cache import (
    CATALOGUE_MIN_FRACTION,
    PROVISIONAL,
    CatalogueSnapshot,
    TextCatalogueCache,
    canonical_stamp,
)
from .tle_parse import validate_tle_pair


THROTTLE_SECONDS = 0.2
MAX_CONSECUTIVE_FAILURES = 3


def validated_records(
    records: pd.DataFrame, context: str, log: Callable[[str], None] = print
) -> pd.DataFrame:
    """Return SatChecker rows whose TLE pair parses and matches its row ID."""
    if not len(records):
        return records.copy()

    valid_indices = []
    for index, row in records.iterrows():
        try:
            norad_id = int(row["NORAD_CAT_ID"])
            embedded_id = validate_tle_pair(row["TLE_LINE1"], row["TLE_LINE2"])
            if embedded_id != norad_id:
                raise ValueError(
                    f"TLE lines belong to satellite {embedded_id}, not {norad_id}"
                )
        except (KeyError, ValueError, TypeError, OverflowError) as error:
            log(
                f"  {context}: invalid service record "
                f"{row.get('NORAD_CAT_ID', 'unknown')!r} rejected — {error}"
            )
            continue
        valid_indices.append(index)
    return records.loc[valid_indices].reset_index(drop=True)


def store_or_warn(
    action: Callable[[], None],
    target: Path,
    what: str,
    log: Callable[[str], None] = print,
) -> bool:
    """Perform a cache write without losing fetched data to an I/O failure."""
    try:
        action()
        return True
    except OSError as error:
        log(
            f"  warning: could not write the {what} to {target} ({error}); "
            "continuing with the validated records in memory, without reusable "
            "cache state"
        )
        return False


def ensure_snapshot(
    cache: TextCatalogueCache,
    catalogue_epoch_jd: float,
    requested_epoch_jd: float,
    state: str,
    provisional_cache_hours: float,
    *,
    fetch_full_catalogue: Callable[[float], client.CatalogueResult] = (
        client.fetch_full_catalogue
    ),
    log: Callable[[str], None] = print,
) -> Optional[CatalogueSnapshot]:
    """Return a cached/fetched catalogue, or ``None`` for an unusable response.

    Transport failures propagate because another route on the same unavailable
    service should not be stormed. Response failures return ``None`` so callers
    can use SatChecker's per-satellite endpoint.
    """
    max_age = None if state != PROVISIONAL else provisional_cache_hours
    snapshot = cache.get_snapshot(catalogue_epoch_jd, state, max_age)
    if snapshot is not None:
        label = "" if state != PROVISIONAL else f" ({state})"
        log(f"  managed catalogue cached at {canonical_stamp(catalogue_epoch_jd)}{label}")
        return snapshot

    log(
        "Fetching TLE catalogue from SatChecker for canonical epoch "
        f"{canonical_stamp(catalogue_epoch_jd)} ..."
    )
    if state == PROVISIONAL:
        log(
            "  this epoch has not settled — the catalogue may still be filling "
            f"upstream, so it will be cached provisionally for "
            f"{provisional_cache_hours:g} h only"
        )
    try:
        result = fetch_full_catalogue(catalogue_epoch_jd)
    except client.SatCheckerTransportError:
        raise
    except client.SatCheckerResponseError as error:
        log(f"  {error}")
        log("  falling back to per-satellite TLE lookups for all requested IDs")
        return None

    records = validated_records(result.records, "managed catalogue", log)
    actual_count = len(records)
    if not actual_count:
        log(
            "  managed catalogue has no valid TLE rows — falling back to "
            "per-satellite lookups"
        )
        return None
    if actual_count < result.expected_count * CATALOGUE_MIN_FRACTION:
        log(
            f"  managed catalogue has {actual_count} valid rows of "
            f"{result.expected_count} expected (< {CATALOGUE_MIN_FRACTION:.0%}) "
            "— not caching; falling back to per-satellite lookups"
        )
        return None

    snapshot = CatalogueSnapshot(
        catalogue_epoch_jd=catalogue_epoch_jd,
        records=records,
        requested_epoch_jd=requested_epoch_jd,
        expected_count=result.expected_count,
        actual_count=actual_count,
        service_version=result.service_version,
        state=state,
    )
    stored = store_or_warn(
        lambda: cache.store_snapshot(snapshot),
        cache.snapshot_path(catalogue_epoch_jd, state),
        f"{state} catalogue snapshot",
        log,
    )
    if stored:
        suffix = (
            ""
            if state != PROVISIONAL
            else f" ({state}, expires in {provisional_cache_hours:g} h)"
        )
        log(f"Saved {actual_count} TLEs for {canonical_stamp(catalogue_epoch_jd)}{suffix}")
    return snapshot


def fetch_nearest_batch(
    norad_ids: list[int],
    epoch_jd: float,
    required: set[int] = frozenset(),
    *,
    fetch_nearest_tle: Callable[[int, float], pd.DataFrame] = client.fetch_nearest_tle,
    throttle_seconds: float = THROTTLE_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Fetch nearest TLEs with bounded failures and outage-aware throttling."""
    rows: list[pd.DataFrame] = []
    unasked_required = set(required)
    consecutive_failures = 0
    for index, norad_id in enumerate(norad_ids):
        if index:
            sleep(throttle_seconds)
        try:
            record = fetch_nearest_tle(norad_id, epoch_jd)
        except client.SatCheckerTransportError:
            if unasked_required:
                raise
            log(
                "  per-satellite endpoint unreachable — skipping the remaining "
                f"{len(norad_ids) - index} freshness upgrade(s); every satellite "
                "is already resolved, so the run continues on its catalogue records"
            )
            break
        except client.SatCheckerError as error:
            unasked_required.discard(norad_id)
            log(f"  per-satellite fetch failed for {norad_id}: {error}")
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                log(
                    "  stopping per-satellite fetches after "
                    f"{consecutive_failures} consecutive failures — the remaining "
                    f"{len(norad_ids) - index - 1} would almost certainly fail too"
                )
                break
            continue

        unasked_required.discard(norad_id)
        consecutive_failures = 0
        if len(record):
            record = validated_records(
                record, f"per-satellite fetch for {norad_id}", log
            )
            if len(record):
                rows.append(record)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
