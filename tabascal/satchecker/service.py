"""Bounded concurrent acquisition built on the SatChecker HTTP client.

This module owns provider-specific orchestration: bounded concurrency, failure
collection, response-row filtering, and resilient cache writes.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from . import client
from .records import KIND_OMM, KIND_TLE, validate_record


MAX_WORKERS = 5

#: The two nearest-record endpoints: log label, and the name of the client
#: function that calls it. Held by *name* rather than by reference so the
#: transport is resolved on each call — that keeps ``client.fetch_nearest_*`` the
#: single seam a test (or a caller wanting a different transport) patches, which
#: binding the functions at import time would quietly break.
_ENDPOINTS = {
    KIND_TLE: ("nearest-TLE", "fetch_nearest_tle"),
    KIND_OMM: ("nearest-OMM", "fetch_nearest_omm"),
}


def nearest_endpoints_for(epoch_jd: float) -> list[tuple[str, Callable]]:
    """The endpoints to try for *epoch_jd*, best first.

    SatChecker's two archives do not overlap: the TLE one is frozen at
    2026-07-11 and the OMM one begins twelve hours later. So the observation
    epoch decides which to ask, and :data:`client.HANDOVER_JD` is the dividing
    line.

    Both are always returned, in order, because neither endpoint reports "I have
    nothing that near". ``get-nearest-omm`` answers a 2021 request with its
    earliest 2026 record; ``get-nearest-tle`` answers a 2027 request with the
    last TLE ever published. The caller cannot tell a good answer from a clamped
    one without checking the epoch it got, and when the answer turns out to be
    unusable the *other* endpoint is where the record actually lives. Trying it
    costs one request and is what makes the handover date a hint rather than a
    cutoff — including if SatChecker ever backfills OMM history, which would
    otherwise leave us silently preferring TLEs for periods with better OMM.
    """
    order = (
        (KIND_OMM, KIND_TLE)
        if float(epoch_jd) >= client.HANDOVER_JD
        else (KIND_TLE, KIND_OMM)
    )
    return [
        (label, getattr(client, function))
        for label, function in (_ENDPOINTS[kind] for kind in order)
    ]

#: Identical per-request rejections, with no success in between, before the batch
#: concludes it is facing a wall rather than that many absent satellites. A 4xx is
#: normally per-request — "no such catalogue entry" is a legitimate 404 — so this
#: has to be high enough that a handful of genuinely unknown IDs at the head of a
#: list cannot trip it, and low enough that a service rejecting everything is not
#: asked once per configured satellite.
RESPONSE_WALL_THRESHOLD = 10


def validated_records(
    records: pd.DataFrame, context: str, log: Callable[[str], None] = print
) -> pd.DataFrame:
    """Return SatChecker rows that validate and belong to the ID they claim.

    Kind-agnostic: a TLE row must parse, checksum and carry the same satellite
    identifier in both lines *and* in the row; an OMM row must carry finite,
    in-range elements and a plausible epoch. The identifier cross-check is
    vacuous for OMM — see :func:`tabascal.satchecker.records.validate_record`.
    """
    if not len(records):
        return records.copy()

    valid_indices = []
    for index, row in records.iterrows():
        try:
            norad_id = int(row["NORAD_CAT_ID"])
            embedded_id = validate_record(row)
            if embedded_id != norad_id:
                raise ValueError(
                    f"record belongs to satellite {embedded_id}, not {norad_id}"
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
    """Validated nearest-record rows and per-ID request failures."""

    records: pd.DataFrame = field(default_factory=pd.DataFrame)
    errors: dict[int, client.SatCheckerError] = field(default_factory=dict)
    #: Set when the batch stopped early because the service itself was the
    #: problem — unreachable, rate-limiting, or answering every request alike.
    #: Distinct from ``errors``, which is per-ID and says nothing about whether
    #: another request is worth making. A caller that would otherwise retry
    #: against a different endpoint must check this first: the service being
    #: down is not a reason to ask it a different question.
    outage: Optional[client.SatCheckerError] = None


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
    fetch_nearest: Callable[[int, float], pd.DataFrame] = client.fetch_nearest_tle,
    endpoint: str = "nearest-TLE",
    max_workers: int = MAX_WORKERS,
    log: Callable[[str], None] = print,
) -> NearestBatchResult:
    """Fetch exact-epoch nearest records with at most *max_workers* in flight.

    *fetch_nearest* is one of the client's two endpoint functions and *endpoint*
    is its label for logs; :func:`nearest_endpoints_for` pairs them. Everything
    below is identical for either, because the two endpoints differ only in what
    they return, never in how they fail.

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
    outage_summary: str | None = None
    wall_status: int | None = None
    wall_count = 0

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        in_flight: dict = {}

        def top_up() -> None:
            """Refill the in-flight set from the IDs not yet sent."""
            while len(in_flight) < worker_count:
                norad_id = next(unsent, None)
                if norad_id is None:
                    return
                future = executor.submit(fetch_nearest, norad_id, epoch_jd)
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
                    log(f"  {endpoint} fetch failed for {norad_id}: {error}")
                    if outage is None:
                        outage, outage_summary = error, _outage_summary(error)
                    continue
                except client.SatCheckerError as error:
                    errors[norad_id] = error
                    log(f"  {endpoint} fetch failed for {norad_id}: {error}")
                    status = getattr(error, "status", None)
                    if status is not None and status == wall_status:
                        wall_count += 1
                    else:
                        wall_status, wall_count = status, 1
                    if outage is None and wall_count >= RESPONSE_WALL_THRESHOLD:
                        outage = client.SatCheckerTransportError(
                            f"SatChecker answered HTTP {wall_status} to "
                            f"{wall_count} consecutive requests without a single "
                            "success; treating it as a service-level block rather "
                            "than that many individually missing satellites"
                        )
                        outage_summary = (
                            f"SatChecker is rejecting every request with "
                            f"HTTP {wall_status}."
                        )
                        log(f"  {outage}")
                    continue
                # The service answered properly, so any run of rejections was
                # about those satellites and not about us.
                wall_status, wall_count = None, 0
                if not len(record):
                    continue
                record = validated_records(
                    record, f"{endpoint} fetch for {norad_id}", log
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
        summary = outage_summary or _outage_summary(outage)
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
    return NearestBatchResult(records=records, errors=errors, outage=outage)
