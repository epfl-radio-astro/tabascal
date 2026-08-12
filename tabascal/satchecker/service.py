"""Bounded concurrent acquisition built on the SatChecker HTTP client.

This module owns provider-specific orchestration: bounded concurrency, failure
collection, response-row filtering, and resilient cache writes.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from . import client
from .tle_parse import validate_tle_pair


MAX_WORKERS = 5


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


@dataclass
class NearestBatchResult:
    """Validated nearest-TLE rows and per-ID request failures."""

    records: pd.DataFrame = field(default_factory=pd.DataFrame)
    errors: dict[int, client.SatCheckerError] = field(default_factory=dict)


def _abandon_pending(futures: dict) -> list[int]:
    """Cancel every request that has not started yet; return the IDs given up on.

    ``Future.cancel`` succeeds only for work still queued, so the handful already
    in flight run to completion and are reported normally. That is enough: the
    queue holds everything beyond ``max_workers``, which is the part that turns an
    outage into a long wait.
    """
    return sorted(
        norad_id for future, norad_id in futures.items() if future.cancel()
    )


def fetch_nearest_batch(
    norad_ids: list[int],
    epoch_jd: float,
    *,
    fetch_nearest_tle: Callable[[int, float], pd.DataFrame] = client.fetch_nearest_tle,
    max_workers: int = MAX_WORKERS,
    log: Callable[[str], None] = print,
) -> NearestBatchResult:
    """Fetch exact-epoch nearest TLEs with at most *max_workers* requests in flight.

    A transport failure means the service itself is unreachable, and every
    remaining ID would query that same service. The first one therefore abandons
    the queued requests instead of working through them: at the 120 s request
    timeout, a few hundred satellites would otherwise spend hours timing out one
    batch at a time before preflight could report the outage it already knew
    about after the first reply.

    A *response* failure is per-request — the service is answering — so it is
    recorded and the remaining IDs proceed.
    """
    ids = list(dict.fromkeys(int(value) for value in norad_ids))
    if not ids:
        return NearestBatchResult()
    if max_workers < 1:
        raise ValueError(f"max_workers must be positive, got {max_workers}")

    rows: dict[int, pd.DataFrame] = {}
    errors: dict[int, client.SatCheckerError] = {}
    worker_count = min(max_workers, len(ids))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(fetch_nearest_tle, norad_id, epoch_jd): norad_id
            for norad_id in ids
        }
        unreachable = False
        for future in as_completed(futures):
            norad_id = futures[future]
            if future.cancelled():
                continue
            try:
                record = future.result()
            except client.SatCheckerTransportError as error:
                errors[norad_id] = error
                log(f"  nearest-TLE fetch failed for {norad_id}: {error}")
                if not unreachable:
                    unreachable = True
                    abandoned = _abandon_pending(futures)
                    for pending_id in abandoned:
                        errors[pending_id] = client.SatCheckerTransportError(
                            f"SatChecker request for {pending_id} abandoned: the "
                            f"service is unreachable ({error})"
                        )
                    if abandoned:
                        log(
                            f"  SatChecker is unreachable; abandoning "
                            f"{len(abandoned)} queued request(s) rather than "
                            "waiting for each to time out"
                        )
                continue
            except client.SatCheckerError as error:
                errors[norad_id] = error
                log(f"  nearest-TLE fetch failed for {norad_id}: {error}")
                continue
            if not len(record):
                continue
            record = validated_records(record, f"nearest-TLE fetch for {norad_id}", log)
            record = record[record["NORAD_CAT_ID"] == norad_id].reset_index(drop=True)
            if len(record):
                rows[norad_id] = record
            else:
                errors[norad_id] = client.SatCheckerResponseError(
                    f"SatChecker returned no valid record for requested NORAD ID {norad_id}"
                )

    records = (
        pd.concat([rows[norad_id] for norad_id in ids if norad_id in rows], ignore_index=True)
        if rows
        else pd.DataFrame()
    )
    return NearestBatchResult(records=records, errors=errors)
