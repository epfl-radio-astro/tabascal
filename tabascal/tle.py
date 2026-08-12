"""TABASCAL TLE orchestration and local orbital-element parsing.

TLEs are sourced from the IAU CPS SatChecker service via
:mod:`tabascal.satchecker` — no account or credentials are required. This module
is the TABASCAL adapter: it resolves each requested NORAD ID against an ordered
set of sources, applies the configurable age policies, drives the deterministic
catalogue cache, and parses OMM-style orbital elements locally from the two TLE
lines. All filtering and element computation is done locally.

Source precedence is resolved **independently per NORAD ID**:

  1. ``extra_tle_dir`` — user-supplied local TLE files. The record whose TLE-line
     epoch is closest to the observation epoch is chosen; it is accepted only if
     within ``extra_tle_max_age_days`` (``None`` = unlimited). An accepted record
     wins outright — later sources are not consulted for that ID. This is *your*
     data: the remote service's age policy never applies to it, so exact replay
     of a previous run's ``used_tles_*.json`` is always possible.
  2. Managed canonical catalogue — one deterministic snapshot per fixed UTC bucket
     (see :func:`tabascal.satchecker.cache.canonical_epoch_jd`), fetched from
     SatChecker on a miss and cached atomically. A catalogue whose epoch has not
     yet settled (``tle_catalogue_settle_days``) is cached only *provisionally*.
  3. Per-satellite SatChecker fallback — for IDs still missing from the bulk
     snapshot, for *all* remaining IDs when the service reports an empty or
     otherwise unusable catalogue at the epoch (its ``tles-at-epoch`` endpoint has
     a data horizon; recent observations can fall beyond it while
     ``get-nearest-tle`` still resolves), and for IDs whose bulk record is older
     than ``remote_tle_target_age_days``. That last case exists because the bulk
     endpoint returns the newest record at or *before* the requested epoch and so
     cannot see a closer one just after it. The records are associated with the
     same canonical snapshot so a later run over the same request reuses them.

Two rules govern what is then accepted:

* **Age ceiling.** Every record from source 2 or 3 must lie within
  ``remote_tle_max_age_days`` of the observation epoch. The epoch is parsed
  locally from TLE line 1 — never taken from a provider field — and every
  accepted record's provider, epoch, signed offset and absolute age is logged.
* **Complete coverage.** Every configured NORAD ID must end up with an accepted
  TLE. If even one does not, resolution raises :class:`TLEError` during preflight,
  before the expensive subtraction begins, naming every failing ID and the
  remedies. TABASCAL never silently shrinks the requested satellite model.

Catalogue reuse follows the bucket policy: the cached record is nearest to the
canonical bucket epoch, not necessarily nearest to the exact observation epoch.
Cache contents cannot change the result for a fixed request and policy.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from platformdirs import user_cache_path

import numpy as np
import pandas as pd

from tabascal import satchecker
from tabascal.satchecker import (
    DEFAULT_CATALOGUE_INTERVAL_HOURS,
    PROVISIONAL,
    STABLE,
    CatalogueSnapshot,
    TextCatalogueCache,
    canonical_epoch_jd,
    canonical_stamp,
    catalogue_state,
    read_legacy_tle_records,
    utc_now_jd,
)
from tabascal.satchecker import SatCheckerError as TLEError  # noqa: F401  back-compat alias
from tabascal.satchecker.cache import CATALOGUE_MIN_FRACTION

# The TLE parser lives in tabascal.satchecker.tle_parse so cache validation and
# element extraction exercise the *same* code; re-exported here under this
# module's historical names.
from tabascal.satchecker.tle_parse import (
    parse_tle_elements,  # noqa: F401  re-export
    tle_epoch_jd as _tle_epoch_jd,
    validate_tle_pair,
)
from tabascal.tle_config import (  # noqa: F401  re-exported for callers
    DEFAULT_REMOTE_TLE_MAX_AGE_DAYS,
    DEFAULT_REMOTE_TLE_TARGET_AGE_DAYS,
    DEFAULT_TLE_CATALOGUE_SETTLE_DAYS,
    DEFAULT_TLE_PROVISIONAL_CACHE_HOURS,
    TLEConfig,
    TLEConfigurationError,
    ms_observation_epoch_jd,
    normalise_norad_ids,
    normalise_tle_config,
    observation_epoch_jd,
    validate_age_days,
)
from tabascal.time import jd_to_datetime


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Delay between per-satellite requests. SatChecker publishes "100 per second,
# 2000 per minute" on these routes, so this is ~6.7x more conservative than the
# service permits — deliberately, because its rate limiter is configured to fail
# open and this is a shared public facility run by IAU CPS. It is not a rate the
# service has asked for; it is us choosing to be a good citizen.
_THROTTLE_SECONDS = 0.2
# DEFAULT_CATALOGUE_INTERVAL_HOURS is imported from tabascal.satchecker above —
# the bucket policy's single source of truth.

# A TLE line-1 epoch is quantised to ~1e-8 day (8 decimal places of a day, ~0.9 ms),
# and the datetime<->JD round-trip adds only sub-microsecond-day noise (measured
# ~3.7e-9 day). This tolerance covers one epoch quantum plus that slack (~2.6 ms), so
# ``extra_tle_max_age_days: 0`` accepts a record matching the observation to TLE
# precision while rejecting one several ms away — matching the documented semantics.
_AGE_TOL_DAYS = 3e-8

# Consecutive per-satellite failures after which the loop gives up. A fault in the
# request rather than the service surfaces as a response error on every satellite
# alike, so error typing alone cannot bound that case; this can.
_MAX_CONSECUTIVE_FAILURES = 3

# Above this many remote records the per-satellite log lines are replaced by a
# grouped summary; set ``TABASCAL_TLE_LOG_DETAIL=1`` to force the full listing.
_GROUPED_LOG_THRESHOLD = 12
_LOG_DETAIL_ENV = "TABASCAL_TLE_LOG_DETAIL"

# Preflight and execution must derive the same observation epoch. This tolerance
# (~86 ms) absorbs float-summation noise while still catching a genuine divergence
# such as a different unit guard or a different per-integration row selection.
_EPOCH_AGREEMENT_TOL_DAYS = 1e-6

# Source labels used in logs, errors and provenance.
_SRC_EXTRA = "extra_tle_dir"
_SRC_CATALOGUE = "managed catalogue"
_SRC_CACHED_FALLBACK = "cached per-satellite record"
_SRC_FALLBACK = "SatChecker per-satellite"


# ---------------------------------------------------------------------------
# TLE cache directory helpers
# ---------------------------------------------------------------------------

def tle_cache_dir() -> Path:
    """Return the managed TLE cache directory, creating it if possible.

    The directory is resolved in priority order:
    1. ``TLE_CACHE_DIR`` environment variable (if set).
    2. The platform user-cache directory (e.g. ``~/.cache/tle-cache`` on Linux,
       ``~/Library/Caches/tle-cache`` on macOS).

    A directory that cannot be created (read-only filesystem, no permission,
    quota) is *not* an error here: the path is returned regardless, reads then
    miss and writes are reported and skipped, so a run with a valid fetch is
    never lost to an unusable cache location.
    """
    p = Path(os.environ.get("TLE_CACHE_DIR") or user_cache_path("tle-cache"))
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return p


# ---------------------------------------------------------------------------
# Configuration validation (back-compat shim over tabascal.tle_config)
# ---------------------------------------------------------------------------

def _validate_max_age(extra_tle_max_age_days) -> Optional[float]:
    """Validate ``extra_tle_max_age_days``: ``None`` or a non-negative number."""
    return validate_age_days(extra_tle_max_age_days, "extra_tle_max_age_days")


# ---------------------------------------------------------------------------
# Resolution results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedTLE:
    """One accepted TLE, with everything needed to explain *why* it was accepted."""

    norad_id: int
    record: dict
    source: str
    provider: Optional[str]
    epoch_jd: float
    offset_days: float          # signed: TLE epoch minus observation epoch

    @property
    def age_days(self) -> float:
        return abs(self.offset_days)

    @property
    def remote(self) -> bool:
        """True for records that came from the service or its managed cache."""
        return self.source != _SRC_EXTRA


@dataclass(frozen=True)
class RejectedTLE:
    """The best (nearest-epoch) candidate that was found but not acceptable."""

    norad_id: int
    source: str
    provider: Optional[str]
    epoch_jd: Optional[float]
    offset_days: Optional[float]
    reason: str

    @property
    def age_days(self) -> Optional[float]:
        return None if self.offset_days is None else abs(self.offset_days)


@dataclass
class TLEResolution:
    """The authoritative outcome of resolving one run's satellites.

    Produced once, during preflight, and consumed unchanged by execution — so the
    coverage decision is made exactly once and the model is built from exactly the
    records that decision was made about.
    """

    requested: list[int]
    obs_epoch_jd: float
    catalogue_epoch_jd: float
    remote_max_age_days: Optional[float]
    resolved: dict[int, ResolvedTLE] = field(default_factory=dict)
    rejected: dict[int, RejectedTLE] = field(default_factory=dict)

    @property
    def missing(self) -> list[int]:
        """Requested IDs with no accepted TLE, in the order they were requested."""
        return [nid for nid in self.requested if nid not in self.resolved]

    @property
    def complete(self) -> bool:
        return not self.missing

    def records(self) -> list[dict]:
        """Accepted raw records (identity + TLE lines + provenance), in requested order."""
        return [dict(self.resolved[nid].record) for nid in self.requested if nid in self.resolved]

    def frame(self) -> pd.DataFrame:
        """Accepted records plus locally parsed orbital elements, in requested order."""
        return _finalise_records(self.records())


# ---------------------------------------------------------------------------
# Record helpers
# ---------------------------------------------------------------------------

def _add_parsed_elements(tles: pd.DataFrame) -> pd.DataFrame:
    """Populate OMM-style element columns by parsing each row's TLE lines.

    Columns are assigned (overwriting any element columns already present in a
    legacy Space-Track cache file) so the locally parsed values always win and
    no duplicate columns are produced.
    """
    tles = tles.copy()
    parsed = pd.DataFrame(
        [parse_tle_elements(r["TLE_LINE1"], r["TLE_LINE2"]) for _, r in tles.iterrows()],
        index=tles.index,
    )
    for col in parsed.columns:
        tles[col] = parsed[col]
    return tles


def _finalise_records(records: list[dict]) -> pd.DataFrame:
    """Turn accepted raw records into the element frame the components consume."""
    if not records:
        return pd.DataFrame()
    tles = pd.DataFrame(records)
    tles["NORAD_CAT_ID"] = pd.to_numeric(tles["NORAD_CAT_ID"]).astype(int)
    return _add_parsed_elements(tles).reset_index(drop=True)


def _validated_service_records(records: pd.DataFrame, context: str) -> pd.DataFrame:
    """Return service rows whose TLE pair parses and matches its catalogue ID.

    Managed-cache validation intentionally remains strict and all-or-nothing. This
    boundary is different: a single bad row in a remote bulk response must not make
    valid, requested satellites unusable. Invalid service rows are reported and
    omitted before any cache write or downstream selection.
    """
    if not len(records):
        return records.copy()

    valid_indices = []
    for idx, row in records.iterrows():
        try:
            nid = int(row["NORAD_CAT_ID"])
            embedded_id = validate_tle_pair(row["TLE_LINE1"], row["TLE_LINE2"])
            if embedded_id != nid:
                raise ValueError(
                    f"TLE lines belong to satellite {embedded_id}, not {nid}"
                )
        except (KeyError, ValueError, TypeError, OverflowError) as e:
            shown_id = row.get("NORAD_CAT_ID", "unknown")
            print(f"  {context}: invalid service record {shown_id!r} rejected — {e}")
            continue
        valid_indices.append(idx)
    return records.loc[valid_indices].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-ID source resolution
# ---------------------------------------------------------------------------

def _select_from_extra_dir(
    extra_tle_dir: str,
    wanted: set[int],
    obs_epoch_jd: float,
    max_age_days: Optional[float],
) -> tuple[dict[int, ResolvedTLE], dict[int, RejectedTLE]]:
    """Resolve IDs from ``extra_tle_dir`` with per-ID nearest + age policy.

    Returns the IDs whose nearest local TLE is within ``max_age_days`` of
    *obs_epoch_jd* (``None`` = unlimited), plus the rejected near-misses. The age
    is measured from the TLE line-1 epoch, not the filename or file modification
    time.
    """
    resolved: dict[int, ResolvedTLE] = {}
    rejected: dict[int, RejectedTLE] = {}
    records = read_legacy_tle_records(extra_tle_dir)
    if not len(records):
        return resolved, rejected
    records = records.copy()
    numeric_ids = pd.to_numeric(records["NORAD_CAT_ID"], errors="coerce")
    valid_ids = numeric_ids.notnull() & np.isfinite(numeric_ids)
    valid_ids &= numeric_ids == numeric_ids.round()
    records = records.loc[valid_ids].copy()
    records["NORAD_CAT_ID"] = numeric_ids.loc[valid_ids].astype(int)
    records = records[records["NORAD_CAT_ID"].isin(wanted)]
    if not len(records):
        return resolved, rejected

    valid_rows = []
    for _, row in records.iterrows():
        nid = int(row["NORAD_CAT_ID"])
        try:
            embedded_id = validate_tle_pair(row["TLE_LINE1"], row["TLE_LINE2"])
            if embedded_id != nid:
                raise ValueError(
                    f"TLE lines belong to satellite {embedded_id}, not {nid}"
                )
            epoch_jd = _tle_epoch_jd(row["TLE_LINE1"])
        except (ValueError, TypeError) as e:
            print(f"  {nid}: invalid extra_tle_dir record rejected — {e}")
            continue
        valid_row = row.copy()
        valid_row["EPOCH_JD"] = epoch_jd
        valid_rows.append(valid_row)
    if not valid_rows:
        return resolved, rejected
    records = pd.DataFrame(valid_rows)

    for nid, group in records.groupby("NORAD_CAT_ID"):
        best = group.loc[(group["EPOCH_JD"] - obs_epoch_jd).abs().idxmin()]
        epoch_jd = float(best["EPOCH_JD"])
        offset = epoch_jd - obs_epoch_jd
        record = {k: v for k, v in best.to_dict().items() if k != "EPOCH_JD"}
        if max_age_days is None or abs(offset) <= max_age_days + _AGE_TOL_DAYS:
            resolved[int(nid)] = ResolvedTLE(
                norad_id=int(nid),
                record=record,
                source=_SRC_EXTRA,
                provider=None,
                epoch_jd=epoch_jd,
                offset_days=offset,
            )
            print(
                f"  {nid}: from extra_tle_dir "
                f"(epoch {abs(offset):.3f} d from observation)"
            )
        else:
            rejected[int(nid)] = RejectedTLE(
                norad_id=int(nid),
                source=_SRC_EXTRA,
                provider=None,
                epoch_jd=epoch_jd,
                offset_days=offset,
                reason=f"extra_tle_max_age_days={max_age_days}",
            )
            print(
                f"  {nid}: extra_tle_dir record rejected — {abs(offset):.3f} d old "
                f"> extra_tle_max_age_days={max_age_days}; trying managed catalogue"
            )
    return resolved, rejected


def _select_from_records(
    records: pd.DataFrame,
    wanted: set[int],
    reference_epoch_jd: float,
) -> dict[int, dict]:
    """One record per wanted ID from a normalised catalogue/fallback frame.

    The service may legitimately carry several distinct TLEs for one NORAD ID.
    When it does, the record whose line-1 epoch is nearest *reference_epoch_jd*
    (the canonical catalogue epoch) is chosen, so the selection is deterministic
    and independent of the service's row order.
    """
    resolved: dict[int, dict] = {}
    if not len(records):
        return resolved
    records = records.copy()
    records["NORAD_CAT_ID"] = pd.to_numeric(records["NORAD_CAT_ID"]).astype(int)
    match = records[records["NORAD_CAT_ID"].isin(wanted)]
    for nid, group in match.groupby("NORAD_CAT_ID"):
        if len(group) > 1:
            offsets = (group["TLE_LINE1"].map(_tle_epoch_jd) - reference_epoch_jd).abs()
            best = group.loc[offsets.idxmin()]
        else:
            best = group.iloc[0]
        resolved[int(nid)] = best.to_dict()
    return resolved


def _accept_remote(
    candidates: dict[int, dict],
    source: str,
    obs_epoch_jd: float,
    max_age_days: Optional[float],
    resolved: dict[int, ResolvedTLE],
    rejected: dict[int, RejectedTLE],
) -> None:
    """Apply the remote age ceiling to *candidates*, updating accept/reject maps.

    The epoch is parsed locally from TLE line 1 — a provider's own ``epoch`` field
    is never trusted — and compared against the actual mean observation epoch
    rather than the canonical catalogue bucket the record was selected against.
    A rejected candidate is remembered (nearest one wins) so the coverage error can
    report exactly how close the best available record was; it is never silently
    re-admitted once the remaining sources are exhausted.

    One rule covers both filling a gap and improving on what is already held: a
    candidate replaces the incumbent only when it is *strictly fresher*. That makes
    the freshness upgrade pass safe by construction — a failed or staler upgrade
    leaves the existing record untouched — and it also stops a later source from
    quietly downgrading an earlier one.
    """
    for nid, record in candidates.items():
        provider = record.get("DATA_SOURCE") or None
        incumbent = resolved.get(nid)
        try:
            epoch_jd = _tle_epoch_jd(record["TLE_LINE1"])
        except (KeyError, ValueError, TypeError) as e:
            if incumbent is None:
                rejected[nid] = RejectedTLE(
                    nid, source, provider, None, None, f"unparseable TLE epoch: {e}"
                )
            continue
        offset = epoch_jd - obs_epoch_jd
        if max_age_days is not None and abs(offset) > max_age_days + _AGE_TOL_DAYS:
            # Only worth reporting when nothing acceptable is held for this ID;
            # an over-age upgrade candidate is simply discarded.
            if incumbent is None:
                previous = rejected.get(nid)
                if previous is None or previous.age_days is None or abs(offset) < previous.age_days:
                    rejected[nid] = RejectedTLE(
                        norad_id=nid,
                        source=source,
                        provider=provider,
                        epoch_jd=epoch_jd,
                        offset_days=offset,
                        reason=f"remote_tle_max_age_days={max_age_days:g}",
                    )
            continue
        if incumbent is not None and incumbent.age_days <= abs(offset):
            continue  # no improvement — keep what we have
        resolved[nid] = ResolvedTLE(
            norad_id=nid,
            record=record,
            source=source,
            provider=provider,
            epoch_jd=epoch_jd,
            offset_days=offset,
        )
        rejected.pop(nid, None)


def _fetch_per_satellite(
    norad_ids: list[int], epoch_jd: float, required: set[int] = frozenset()
) -> pd.DataFrame:
    """Fetch one record per ID from the per-satellite endpoint.

    Serves both purposes the endpoint has here: filling IDs the bulk catalogue
    lacks, and upgrading IDs whose bulk record is needlessly old. A satellite the
    endpoint cannot answer for is simply omitted — the caller decides whether that
    is a coverage failure or a declined upgrade.

    *required* names the IDs the run cannot proceed without; everything else in
    *norad_ids* is an optional freshness upgrade. A transport failure ends the loop
    either way, since the service is down and continuing would issue one doomed
    request per remaining satellite — but it is only *re-raised* while some
    required ID has not yet been asked about. An outage during a batch of pure
    upgrades must not fail a run whose satellites are all already resolved, and any
    upgrades obtained before the outage are kept.
    """
    rows: list[pd.DataFrame] = []
    # IDs the run needs and that we have not yet managed to ask the service about.
    # A satellite that was asked and simply has no record leaves this set: that is
    # a coverage failure, reported far better by the coverage check than by a
    # transport error about a *different* satellite.
    unasked_required = set(required)
    consecutive_failures = 0
    for i, nid in enumerate(norad_ids):
        if i:
            time.sleep(_THROTTLE_SECONDS)
        try:
            rec = satchecker.fetch_nearest_tle(nid, epoch_jd)
        except satchecker.SatCheckerTransportError:
            if unasked_required:
                raise
            print(
                f"  per-satellite endpoint unreachable — skipping the remaining "
                f"{len(norad_ids) - i} freshness upgrade(s); every satellite is "
                f"already resolved, so the run continues on its catalogue records"
            )
            break
        except TLEError as e:
            unasked_required.discard(nid)  # asked; it failed for this satellite
            print(f"  per-satellite fetch failed for {nid}: {e}")
            consecutive_failures += 1
            if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                # Independent of how the failure was typed: a fault affecting the
                # request itself rather than the service (a malformed epoch, say)
                # produces a *response* error on every satellite alike, and working
                # through hundreds of them would learn nothing. Stop rather than
                # raise, so the IDs simply go unresolved and the coverage check
                # reports all of them with its usual diagnostics.
                print(
                    f"  stopping per-satellite fetches after "
                    f"{consecutive_failures} consecutive failures — the remaining "
                    f"{len(norad_ids) - i - 1} would almost certainly fail too"
                )
                break
            continue
        unasked_required.discard(nid)
        consecutive_failures = 0
        if len(rec):
            rec = _validated_service_records(rec, f"per-satellite fetch for {nid}")
            if len(rec):
                rows.append(rec)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Managed snapshot (stable / provisional)
# ---------------------------------------------------------------------------

def _store_or_warn(action, target: Path, what: str) -> bool:
    """Run a cache write, downgrading environmental failures to a warning.

    A read-only filesystem, a full quota or a missing permission must not discard
    a fetch that already succeeded: the validated in-memory records still serve
    this run, only the reusable cache state is lost. Validation and programming
    errors are *not* environmental and keep propagating — a snapshot that fails
    its own validation is a bug, not a disk problem, and the atomic writer has
    already removed any partial temporary file.
    """
    try:
        action()
        return True
    except OSError as e:
        print(
            f"  warning: could not write the {what} to {target} ({e}); continuing "
            f"with the validated records in memory, without reusable cache state"
        )
        return False


def _ensure_snapshot(
    cache: TextCatalogueCache,
    catalogue_epoch_jd: float,
    obs_epoch_jd: float,
    state: str,
    provisional_cache_hours: float,
) -> Optional[CatalogueSnapshot]:
    """Return the canonical snapshot, downloading + caching atomically on a miss.

    *state* decides how the result may be persisted. A :data:`STABLE` epoch (older
    than ``tle_catalogue_settle_days``) yields the permanent, immutable snapshot.
    A :data:`PROVISIONAL` one — a recent or future epoch whose upstream catalogue
    may still be filling — is stored under its own filename with a short expiry,
    and is never promoted: once the epoch settles, the next successful request
    writes a fresh stable snapshot instead.

    Returns ``None`` when the service is reachable but its catalogue is unusable
    at this epoch (empty, malformed, or too incomplete to cache). The caller then
    resolves satellites through the per-ID fallback instead. Transport failures
    still raise.
    """
    max_age = None if state == STABLE else provisional_cache_hours
    snapshot = cache.get_snapshot(catalogue_epoch_jd, state, max_age)
    if snapshot is not None:
        label = "" if state == STABLE else f" ({state})"
        print(
            f"  managed catalogue cached at "
            f"{canonical_stamp(catalogue_epoch_jd)}{label}"
        )
        return snapshot

    print(
        f"Fetching TLE catalogue from SatChecker for canonical epoch "
        f"{canonical_stamp(catalogue_epoch_jd)} ..."
    )
    if state == PROVISIONAL:
        print(
            "  this epoch has not settled — the catalogue may still be filling "
            f"upstream, so it will be cached provisionally for "
            f"{provisional_cache_hours:g} h only"
        )
    try:
        result = satchecker.fetch_full_catalogue(catalogue_epoch_jd)
    except satchecker.SatCheckerTransportError:
        raise  # service unreachable: per-satellite lookups would only storm it
    except satchecker.SatCheckerResponseError as e:
        # The service answered but the catalogue is unusable. get-nearest-tle is a
        # different endpoint on a service we know is up, so it is worth trying.
        print(f"  {e}")
        print("  falling back to per-satellite TLE lookups for all requested IDs")
        return None
    records = _validated_service_records(result.records, "managed catalogue")
    actual_count = len(records)
    if not actual_count:
        print(
            "  managed catalogue has no valid TLE rows — falling back to "
            "per-satellite lookups"
        )
        return None
    if actual_count < result.expected_count * CATALOGUE_MIN_FRACTION:
        print(
            f"  managed catalogue has {actual_count} valid rows of "
            f"{result.expected_count} expected (< {CATALOGUE_MIN_FRACTION:.0%}) "
            "— not caching; falling back to per-satellite lookups"
        )
        return None
    snapshot = CatalogueSnapshot(
        catalogue_epoch_jd=catalogue_epoch_jd,
        records=records,
        requested_epoch_jd=obs_epoch_jd,
        expected_count=result.expected_count,
        actual_count=actual_count,
        service_version=result.service_version,
        state=state,
    )
    stored = _store_or_warn(
        lambda: cache.store_snapshot(snapshot),
        cache.snapshot_path(catalogue_epoch_jd, state),
        f"{state} catalogue snapshot",
    )
    if stored:
        print(
            f"Saved {len(records)} TLEs for {canonical_stamp(catalogue_epoch_jd)}"
            + ("" if state == STABLE else f" ({state}, expires in {provisional_cache_hours:g} h)")
        )
    return snapshot


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _detail_requested() -> bool:
    return os.environ.get(_LOG_DETAIL_ENV, "").strip().lower() not in ("", "0", "false", "no")


def _describe(entry: ResolvedTLE) -> str:
    provider = f" [{entry.provider}]" if entry.provider else ""
    return (
        f"  {entry.norad_id}: {entry.source}{provider} "
        f"epoch {jd_to_datetime(entry.epoch_jd).isoformat()} UTC, "
        f"offset {entry.offset_days:+.4f} d, age {entry.age_days:.4f} d"
    )


def _report_remote_selection(resolution: TLEResolution) -> None:
    """Log provider, epoch, signed offset and age for every accepted remote TLE.

    Small ID sets get one line each. Larger ones get a grouped summary — an
    all-Starlink run would otherwise bury the rest of the log — with the oldest
    records still named individually, and the full listing available on demand via
    ``TABASCAL_TLE_LOG_DETAIL=1``.
    """
    remote = [e for e in resolution.resolved.values() if e.remote]
    if not remote:
        return
    limit = resolution.remote_max_age_days
    limit_text = "no limit" if limit is None else f"limit {limit:g} d"

    if len(remote) <= _GROUPED_LOG_THRESHOLD or _detail_requested():
        print(f"TLE remote records     : {len(remote)} accepted ({limit_text})")
        for entry in sorted(remote, key=lambda e: e.norad_id):
            print(_describe(entry))
        return

    ages = np.array([e.age_days for e in remote])
    by_source: dict[str, int] = {}
    for entry in remote:
        key = entry.source + (f" [{entry.provider}]" if entry.provider else "")
        by_source[key] = by_source.get(key, 0) + 1
    print(f"TLE remote records     : {len(remote)} accepted ({limit_text})")
    for key, count in sorted(by_source.items(), key=lambda kv: -kv[1]):
        print(f"  {count} from {key}")
    print(
        f"  age vs observation   : min {ages.min():.4f} d, "
        f"median {np.median(ages):.4f} d, max {ages.max():.4f} d"
    )
    oldest = sorted(remote, key=lambda e: -e.age_days)[:5]
    print("  oldest               : " + ", ".join(
        f"{e.norad_id} ({e.offset_days:+.4f} d)" for e in oldest
    ))
    print(f"  (set {_LOG_DETAIL_ENV}=1 for a per-satellite listing)")


def _coverage_error(resolution: TLEResolution) -> TLEError:
    """Build the actionable error raised when some configured ID has no TLE."""
    missing = resolution.missing
    lines = [
        f"TLEs could not be resolved for {len(missing)} of "
        f"{len(resolution.requested)} configured satellites at observation epoch "
        f"{jd_to_datetime(resolution.obs_epoch_jd).isoformat()} UTC "
        f"(catalogue bucket {canonical_stamp(resolution.catalogue_epoch_jd)}):"
    ]
    for nid in missing:
        bad = resolution.rejected.get(nid)
        if bad is None:
            lines.append(
                f"  {nid}: no record found in extra_tle_dir, the managed "
                f"catalogue, or SatChecker"
            )
        elif bad.age_days is None:
            lines.append(f"  {nid}: best candidate unusable — {bad.reason}")
        else:
            provider = f", provider {bad.provider}" if bad.provider else ""
            lines.append(
                f"  {nid}: best candidate is {bad.age_days:.3f} d from the "
                f"observation (epoch {jd_to_datetime(bad.epoch_jd).isoformat()} "
                f"UTC, from {bad.source}{provider}) — rejected by {bad.reason}"
            )
    limit = resolution.remote_max_age_days
    lines += [
        "",
        f"The remote age ceiling in force is remote_tle_max_age_days="
        f"{'null (disabled)' if limit is None else f'{limit:g}'}. Remedies:",
        "  - put an acceptable TLE for these satellites in a directory and pass "
        "--extra-tle-dir <dir> (or set satellites.extra_tle_dir)",
        "  - deliberately change satellites.remote_tle_max_age_days (null removes "
        "the ceiling entirely; this is an expert opt-out, not a default)",
        "  - remove these NORAD IDs from satellites.norad_ids",
        "",
        "TABASCAL will not silently omit a configured satellite from the RFI "
        "model: the run stops here rather than subtracting an incomplete one.",
    ]
    return TLEError("\n".join(lines))


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def resolve_tles(
    norad_ids,
    obs_epoch_jd: float,
    extra_tle_dir: Optional[str] = None,
    extra_tle_max_age_days: Optional[float] = None,
    remote_tle_max_age_days: Optional[float] = DEFAULT_REMOTE_TLE_MAX_AGE_DAYS,
    remote_tle_target_age_days: Optional[float] = DEFAULT_REMOTE_TLE_TARGET_AGE_DAYS,
    catalogue_interval_hours: float = DEFAULT_CATALOGUE_INTERVAL_HOURS,
    catalogue_settle_days: Optional[float] = DEFAULT_TLE_CATALOGUE_SETTLE_DAYS,
    provisional_cache_hours: float = DEFAULT_TLE_PROVISIONAL_CACHE_HOURS,
    clock=None,
) -> TLEResolution:
    """Resolve every requested NORAD ID at *obs_epoch_jd*, without raising on gaps.

    Returns the full :class:`TLEResolution` — accepted records, rejected
    near-misses and the epochs everything was judged against. Callers decide what
    an incomplete result means; :func:`require_complete_coverage` is the policy
    TABASCAL runs use.
    """
    clock = clock or utc_now_jd
    requested = normalise_norad_ids(norad_ids)
    extra_max_age = validate_age_days(extra_tle_max_age_days, "extra_tle_max_age_days")
    remote_max_age = validate_age_days(remote_tle_max_age_days, "remote_tle_max_age_days")
    target_age = validate_age_days(remote_tle_target_age_days, "remote_tle_target_age_days")
    obs_epoch_jd = float(obs_epoch_jd)
    catalogue_epoch = canonical_epoch_jd(obs_epoch_jd, catalogue_interval_hours)

    resolution = TLEResolution(
        requested=requested,
        obs_epoch_jd=obs_epoch_jd,
        catalogue_epoch_jd=catalogue_epoch,
        remote_max_age_days=remote_max_age,
    )
    if not requested:
        return resolution

    print(f"TLE requested epoch    : {jd_to_datetime(obs_epoch_jd).isoformat()} UTC")
    print(
        f"TLE catalogue epoch    : {canonical_stamp(catalogue_epoch)} "
        f"(nearest to a {catalogue_interval_hours:g} h bucket, not the exact "
        f"observation)"
    )

    wanted = set(requested)

    # 1. extra_tle_dir (per-ID precedence + its own age policy)
    if extra_tle_dir:
        print(
            f"TLE extra dir          : {Path(extra_tle_dir).resolve()} "
            f"(max age {'unlimited' if extra_max_age is None else f'{extra_max_age:g} d'})"
        )
        from_extra, extra_rejected = _select_from_extra_dir(
            extra_tle_dir, wanted, obs_epoch_jd, extra_max_age
        )
        resolution.resolved.update(from_extra)
        resolution.rejected.update(extra_rejected)

    remaining = wanted - set(resolution.resolved)

    # 2 + 3. managed canonical snapshot, then per-satellite fallback
    if remaining:
        cache = TextCatalogueCache(tle_cache_dir(), clock=clock)
        state = catalogue_state(catalogue_epoch, clock(), catalogue_settle_days)
        # Stable cache entries are immutable and permanent; provisional ones exist
        # only to avoid refetching within a session, so they carry an expiry.
        extra_max_age_hours = None if state == STABLE else provisional_cache_hours
        snapshot = _ensure_snapshot(
            cache, catalogue_epoch, obs_epoch_jd, state, provisional_cache_hours
        )
        if snapshot is not None:
            _accept_remote(
                _select_from_records(snapshot.records, remaining, catalogue_epoch),
                _SRC_CATALOGUE,
                obs_epoch_jd,
                remote_max_age,
                resolution.resolved,
                resolution.rejected,
            )
            remaining = wanted - set(resolution.resolved)

        # The per-satellite endpoint serves two needs from here on: IDs the bulk
        # catalogue lacks, and IDs whose bulk record is needlessly old (see
        # _stale_catalogue_ids). Both are satisfied from the same cache read and
        # the same fetch, so an upgrade costs nothing extra once a run is already
        # making per-satellite requests.
        wanted_per_satellite = remaining | _stale_catalogue_ids(resolution, target_age)

        if wanted_per_satellite:
            # Fallback records carry the same state and expiry as the snapshot
            # they stand in for. An unsettled epoch reaches the per-satellite
            # endpoint *because* its bulk catalogue is empty, so without this the
            # one result the settling policy never revisits would be exactly the
            # one it exists to revisit.
            _accept_remote(
                _select_from_records(
                    cache.get_extra(catalogue_epoch, state, extra_max_age_hours),
                    wanted_per_satellite,
                    catalogue_epoch,
                ),
                _SRC_CACHED_FALLBACK,
                obs_epoch_jd,
                remote_max_age,
                resolution.resolved,
                resolution.rejected,
            )
            remaining = wanted - set(resolution.resolved)
            wanted_per_satellite = remaining | _stale_catalogue_ids(resolution, target_age)

        if wanted_per_satellite:
            to_fetch = sorted(wanted_per_satellite)
            n_missing = len(remaining)
            n_upgrade = len(to_fetch) - n_missing
            reasons = (
                [f"{n_missing} missing"] if n_missing else []
            ) + ([f"{n_upgrade} older than {target_age:g} d"] if n_upgrade else [])
            print(
                f"Fetching {len(to_fetch)} TLE(s) individually from SatChecker "
                f"({', '.join(reasons)}): {to_fetch}"
            )
            fetched = _fetch_per_satellite(to_fetch, catalogue_epoch, remaining)
            if len(fetched):
                _store_or_warn(
                    lambda: cache.store_extra(catalogue_epoch, fetched, state),
                    cache.extra_path(catalogue_epoch, state),
                    "per-satellite fallback records",
                )
                # _accept_remote replaces only on a strict improvement, so a
                # satellite the endpoint could not better keeps its bulk record
                # and an upgrade can never turn into a coverage failure.
                _accept_remote(
                    _select_from_records(fetched, wanted_per_satellite, catalogue_epoch),
                    _SRC_FALLBACK,
                    obs_epoch_jd,
                    remote_max_age,
                    resolution.resolved,
                    resolution.rejected,
                )

    _report_remote_selection(resolution)
    return resolution


def _stale_catalogue_ids(
    resolution: TLEResolution, target_age_days: Optional[float]
) -> set[int]:
    """IDs holding a bulk-catalogue record older than *target_age_days*.

    Only records from the bulk catalogue qualify. SatChecker's ``tles-at-epoch``
    returns the newest record at or *before* the requested epoch, so it cannot see
    a closer record that happens to fall after it; measured over 32 GNSS
    satellites, that made its records about twice the age of the best available
    and, in the worst case, 4.5 days against 1.1. ``get-nearest-tle`` has no such
    restriction, so re-asking it for the stale ones recovers the difference.

    Records already obtained from the per-satellite endpoint are excluded — they
    are already the nearest the service holds, so re-requesting them would spend a
    request to learn nothing. ``None`` disables the upgrade pass entirely.
    """
    if target_age_days is None:
        return set()
    return {
        nid
        for nid, entry in resolution.resolved.items()
        if entry.source == _SRC_CATALOGUE and entry.age_days > target_age_days
    }


def require_complete_coverage(resolution: TLEResolution) -> TLEResolution:
    """Return *resolution* unchanged, or raise the actionable coverage error.

    Enforced during preflight so an unresolvable satellite stops the run *before*
    the expensive subtraction, and enforced only once — execution consumes the
    same resolution rather than making a second, potentially different decision.
    """
    if resolution.requested and not resolution.complete:
        raise _coverage_error(resolution)
    return resolution


# ---------------------------------------------------------------------------
# Public orchestration
# ---------------------------------------------------------------------------

def get_tles_by_id(
    norad_ids,
    times_jd,
    extra_tle_dir: Optional[str] = None,
    extra_tle_max_age_days: Optional[float] = None,
    remote_tle_max_age_days: Optional[float] = DEFAULT_REMOTE_TLE_MAX_AGE_DAYS,
    remote_tle_target_age_days: Optional[float] = DEFAULT_REMOTE_TLE_TARGET_AGE_DAYS,
    catalogue_interval_hours: float = DEFAULT_CATALOGUE_INTERVAL_HOURS,
    catalogue_settle_days: Optional[float] = DEFAULT_TLE_CATALOGUE_SETTLE_DAYS,
    provisional_cache_hours: float = DEFAULT_TLE_PROVISIONAL_CACHE_HOURS,
    clock=None,
) -> pd.DataFrame:
    """Resolve TLEs for *norad_ids*, sharing one resolution across all processes.

    Multi-process runs resolve on process 0 only and broadcast the accepted raw
    records (or the failure) to every worker, so the provider sees exactly one
    fetch per run — even when the cache could not be written and workers would
    otherwise have found nothing to read. Single-process runs resolve directly.

    Returns one row per requested ID, in the requested order, with OMM-style
    element columns parsed locally from the TLE lines. Raises :class:`TLEError`
    unless every requested ID resolved.
    """
    return resolve_shared(
        lambda: require_complete_coverage(
            resolve_tles(
                norad_ids,
                observation_epoch_jd(times_jd),
                extra_tle_dir=extra_tle_dir,
                extra_tle_max_age_days=extra_tle_max_age_days,
                remote_tle_max_age_days=remote_tle_max_age_days,
                remote_tle_target_age_days=remote_tle_target_age_days,
                catalogue_interval_hours=catalogue_interval_hours,
                catalogue_settle_days=catalogue_settle_days,
                provisional_cache_hours=provisional_cache_hours,
                clock=clock or utc_now_jd,
            )
        )
    ).frame()


# ---------------------------------------------------------------------------
# Multi-process sharing
# ---------------------------------------------------------------------------

def resolve_shared(resolve) -> TLEResolution:
    """Run *resolve* on process 0 only and share its outcome with every process.

    Every entry point that resolves TLEs must go through here, not just the
    element fetch: in a multi-process launch every rank builds its own
    :class:`~tabascal.config.TabConfig` and would otherwise reach the resolver
    directly, so a cache miss (or an unwritable cache) would have each rank
    download the catalogue and issue its own fallback requests. Divergent
    outcomes are worse still — some ranks exiting while others go on to a JAX
    collective is a hang, not an error.

    Workers therefore never call the provider themselves: whatever process 0
    decided — the accepted records or the failure — is what every process acts
    on, so the run either proceeds from one identical satellite set or fails
    coherently everywhere. Only the raw identity/TLE-line columns and the epochs
    cross the wire; every process re-derives the orbital elements locally, so the
    parsed values are bit-identical rather than serialisation-rounded. Rejection
    diagnostics stay on process 0, which has already formatted them into the
    error text being shared.

    Single-process runs call *resolve* directly and are unaffected.
    """
    from tabascal import distributed

    if distributed.process_count() == 1:
        return resolve()

    payload = None
    if distributed.is_process_0():
        try:
            message = _resolution_to_wire(resolve())
        except Exception as e:  # reported identically on every process below
            message = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        payload = json.dumps(message).encode()

    message = json.loads(distributed.broadcast_bytes_from_rank0(payload, "tle-fetch"))
    if not message.get("ok"):
        raise TLEError(
            "TLE resolution failed on process 0; every process is stopping with "
            f"the same result.\n{message.get('error')}"
        )
    return _resolution_from_wire(message)


_WIRE_COLUMNS = ("NORAD_CAT_ID", "OBJECT_NAME", "TLE_LINE1", "TLE_LINE2", "DATA_SOURCE")


def _wire_record(record: dict) -> dict:
    """JSON-safe projection of one record onto the columns workers actually need."""
    out = {"NORAD_CAT_ID": int(record["NORAD_CAT_ID"])}
    for col in _WIRE_COLUMNS[1:]:
        value = record.get(col)
        out[col] = None if value is None or pd.isna(value) else str(value)
    return out


def _resolution_to_wire(resolution: TLEResolution) -> dict:
    """Serialise an accepted resolution for the broadcast.

    ``json`` round-trips a Python float through its ``repr``, so the epochs and
    offsets survive exactly — the workers judge the run against the same numbers
    process 0 did.
    """
    return {
        "ok": True,
        "requested": [int(nid) for nid in resolution.requested],
        "obs_epoch_jd": float(resolution.obs_epoch_jd),
        "catalogue_epoch_jd": float(resolution.catalogue_epoch_jd),
        "remote_max_age_days": (
            None if resolution.remote_max_age_days is None
            else float(resolution.remote_max_age_days)
        ),
        "resolved": [
            {
                "norad_id": int(entry.norad_id),
                "record": _wire_record(entry.record),
                "source": entry.source,
                "provider": entry.provider,
                "epoch_jd": float(entry.epoch_jd),
                "offset_days": float(entry.offset_days),
            }
            for entry in (resolution.resolved[nid] for nid in resolution.requested
                          if nid in resolution.resolved)
        ],
    }


def _resolution_from_wire(message: dict) -> TLEResolution:
    """Rebuild process 0's resolution on a worker."""
    return TLEResolution(
        requested=[int(nid) for nid in message["requested"]],
        obs_epoch_jd=message["obs_epoch_jd"],
        catalogue_epoch_jd=message["catalogue_epoch_jd"],
        remote_max_age_days=message["remote_max_age_days"],
        resolved={
            int(entry["norad_id"]): ResolvedTLE(
                norad_id=int(entry["norad_id"]),
                record=entry["record"],
                source=entry["source"],
                provider=entry["provider"],
                epoch_jd=entry["epoch_jd"],
                offset_days=entry["offset_days"],
            )
            for entry in message["resolved"]
        },
    )


# ---------------------------------------------------------------------------
# Reproducibility: persist the TLEs a run actually used
# ---------------------------------------------------------------------------

def save_tles_for_reuse(path, norad_ids, tles) -> Optional[str]:
    """Write the TLE lines a run used to *path* in ``extra_tle_dir`` format.

    The file is a pandas-oriented JSON with ``NORAD_CAT_ID``, ``TLE_LINE1`` and
    ``TLE_LINE2`` columns — exactly what :func:`read_legacy_tle_records` reads —
    so a later run can reproduce this run's trajectory priors by passing the
    file's directory via ``--extra-tle-dir`` (with the default unlimited
    ``extra_tle_max_age_days``), independent of the shared cache, of what
    SatChecker serves by then, and of the remote age ceiling.

    ``norad_ids`` and ``tles`` are the aligned arrays produced by the element
    fetchers (one ``(line1, line2)`` pair per ID). Returns the written path, or
    ``None`` when there is nothing to save.
    """
    tle_pairs = np.atleast_2d(np.asarray(tles)) if tles is not None else np.empty((0, 2))
    ids = list(norad_ids or [])
    if not ids or not tle_pairs.size:
        return None
    df = pd.DataFrame(
        {
            "NORAD_CAT_ID": [int(n) for n in ids],
            "TLE_LINE1": tle_pairs[:, 0],
            "TLE_LINE2": tle_pairs[:, 1],
        }
    )
    path = str(path)
    df.to_json(path)
    return path


# ---------------------------------------------------------------------------
# Preflight check
# ---------------------------------------------------------------------------

def _ms_mean_epoch_jd(ms_path: str) -> float:
    """Mean observation epoch (UTC Julian Date) of an MS.

    Thin alias over :func:`tabascal.tle_config.ms_observation_epoch_jd`, kept as
    the single seam tests patch to exercise preflight offline.
    """
    return ms_observation_epoch_jd(ms_path)


def preflight_tle_check(
    tle_config: TLEConfig,
    ms_path: str,
    clock=None,
) -> TLEResolution:
    """Resolve every configured satellite before the run commits to any real work.

    This is the *authoritative* resolution and the single enforcement point for
    complete coverage: it reads the observation epoch from the MS, resolves each
    NORAD ID through the configured sources, reports what was accepted and from
    where, and raises :class:`TLEError` naming every failure if any configured ID
    has no acceptable TLE. It runs before the visibilities are read and long
    before subtraction, so a missing or unusably stale satellite costs seconds
    rather than a whole job.

    The returned resolution is what execution then builds the model from — it must
    not re-resolve, or the model could differ from what was checked.

    Multi-process runs reach this through every rank's own ``TabConfig``, so the
    resolution goes through :func:`resolve_shared`: process 0 does the work and
    the rest receive its outcome, rather than each rank downloading the catalogue
    and reaching its own coverage verdict.
    """
    clock = clock or utc_now_jd
    if not tle_config.norad_ids:
        return TLEResolution(
            requested=[],
            obs_epoch_jd=float("nan"),
            catalogue_epoch_jd=float("nan"),
            remote_max_age_days=tle_config.remote_tle_max_age_days,
        )

    print(f"Preflight TLE check    : NORAD IDs {tle_config.norad_ids}")

    resolution = resolve_shared(
        lambda: require_complete_coverage(
            resolve_tles(
                tle_config.norad_ids,
                _ms_mean_epoch_jd(ms_path),
                extra_tle_dir=tle_config.extra_tle_dir,
                extra_tle_max_age_days=tle_config.extra_tle_max_age_days,
                remote_tle_max_age_days=tle_config.remote_tle_max_age_days,
                remote_tle_target_age_days=tle_config.remote_tle_target_age_days,
                catalogue_interval_hours=tle_config.catalogue_interval_hours,
                catalogue_settle_days=tle_config.catalogue_settle_days,
                provisional_cache_hours=tle_config.provisional_cache_hours,
                clock=clock,
            )
        )
    )

    n_extra = sum(1 for e in resolution.resolved.values() if not e.remote)
    print(
        f"TLE preflight OK       : {len(resolution.resolved)} of "
        f"{len(resolution.requested)} satellites resolved "
        f"({n_extra} from extra_tle_dir, "
        f"{len(resolution.resolved) - n_extra} from SatChecker/managed cache)"
    )
    print(
        "  Local TLE files are searched with --extra-tle-dir <dir>; "
        "TLE_CACHE_DIR=<dir> relocates the managed cache and is not an "
        "additional source."
    )
    return resolution


def check_epoch_agreement(resolution: TLEResolution, times_jd) -> None:
    """Verify execution's observation epoch matches the one preflight resolved at.

    The epoch sets both the canonical cache bucket and every age comparison, so a
    divergence between the preflight MS read and the times
    :func:`tabascal.tab_tools.read_ms` returned would mean the run was checked
    against one epoch and modelled at another. Raise rather than silently
    re-resolving: a second resolution could reach a different coverage decision.
    """
    if not resolution.requested:
        return
    execution_epoch = observation_epoch_jd(times_jd)
    if abs(execution_epoch - resolution.obs_epoch_jd) <= _EPOCH_AGREEMENT_TOL_DAYS:
        return
    raise TLEError(
        "Observation epoch disagreement between the TLE preflight check and the "
        "Measurement Set read:\n"
        f"  preflight : {jd_to_datetime(resolution.obs_epoch_jd).isoformat()} UTC\n"
        f"  execution : {jd_to_datetime(execution_epoch).isoformat()} UTC\n"
        "The TLEs were selected and age-checked against the preflight epoch, so "
        "the run stops rather than model a different one. This usually means the "
        "MS TIME column is inconsistent (mixed units, or a row count that is not "
        "a whole number of integrations)."
    )
