"""Bounded concurrent acquisition built on the SatChecker HTTP client.

This module owns provider-specific orchestration: bounded concurrency, failure
collection, response-row filtering, and resilient cache writes.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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


def _outage_summary(cause: client.SatCheckerError) -> str:
    """How to describe an outage in the log: rate limit or plain unreachability."""
    retry_after = getattr(cause, "retry_after", None)
    if isinstance(cause, client.SatCheckerRateLimitError):
        wait_hint = (
            f" It asked for {retry_after:g} s before the next request."
            if retry_after is not None
            else ""
        )
        return f"SatChecker is rate-limiting this client.{wait_hint}"
    return "SatChecker is unreachable."


def fetch_nearest_batch(
    norad_ids: list[int],
    epoch_jd: float,
    *,
    fetch_nearest_tle: Callable[[int, float], pd.DataFrame] = client.fetch_nearest_tle,
    max_workers: int = MAX_WORKERS,
    log: Callable[[str], None] = print,
) -> NearestBatchResult:
    """Fetch exact-epoch nearest TLEs with at most *max_workers* requests in flight.

    Requests are submitted incrementally — the in-flight set is topped back up as
    each one lands — rather than queued all at once. That is what makes the
    outage bound hold: a transport failure means the service itself is unreachable
    (or has asked us to stop), so every remaining ID would be a request we already
    know is unwelcome, and nothing further is sent. At most *max_workers* requests
    can be in flight when the first failure is seen, so that is the most an outage
    can ever cost, whatever the worker count and however fast the failures return.

    Queueing everything up front and cancelling the remainder does not achieve
    this: with fast failures — a refused connection, or an HTTP 429 arriving in
    one round trip — the pool's workers drain the queue faster than the cancel
    can catch it, and a large run still hits an already-failing service hundreds
    of times.

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
    unsent = iter(ids)
    outage: client.SatCheckerError | None = None

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        in_flight: dict = {}

        def top_up() -> None:
            """Refill the in-flight set from the IDs not yet sent."""
            while len(in_flight) < worker_count:
                norad_id = next(unsent, None)
                if norad_id is None:
                    return
                future = executor.submit(fetch_nearest_tle, norad_id, epoch_jd)
                in_flight[future] = norad_id

        top_up()
        while in_flight:
            done, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
            for future in done:
                norad_id = in_flight.pop(future)
                try:
                    record = future.result()
                except client.SatCheckerTransportError as error:
                    errors[norad_id] = error
                    log(f"  nearest-TLE fetch failed for {norad_id}: {error}")
                    if outage is None:
                        outage = error
                    continue
                except client.SatCheckerError as error:
                    errors[norad_id] = error
                    log(f"  nearest-TLE fetch failed for {norad_id}: {error}")
                    continue
                if not len(record):
                    continue
                record = validated_records(
                    record, f"nearest-TLE fetch for {norad_id}", log
                )
                record = record[record["NORAD_CAT_ID"] == norad_id].reset_index(drop=True)
                if len(record):
                    rows[norad_id] = record
                else:
                    errors[norad_id] = client.SatCheckerResponseError(
                        f"SatChecker returned no valid record for requested NORAD ID "
                        f"{norad_id}"
                    )
            # Stop feeding a service that has already told us it cannot serve us.
            if outage is None:
                top_up()

    if outage is not None:
        summary = _outage_summary(outage)
        abandoned = list(unsent)
        for norad_id in abandoned:
            errors[norad_id] = client.SatCheckerTransportError(
                f"SatChecker request for {norad_id} was never sent: {summary} "
                f"({outage})"
            )
        if abandoned:
            log(
                f"  {summary} Not sending the remaining {len(abandoned)} "
                "request(s) — every one would add load to a service that has "
                "already failed or asked us to back off"
            )

    records = (
        pd.concat([rows[norad_id] for norad_id in ids if norad_id in rows], ignore_index=True)
        if rows
        else pd.DataFrame()
    )
    return NearestBatchResult(records=records, errors=errors)
