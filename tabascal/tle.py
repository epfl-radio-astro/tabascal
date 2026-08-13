"""TABASCAL TLE orchestration and local orbital-element parsing.

TLEs are sourced from the IAU CPS SatChecker service via
:mod:`tabascal.satchecker` — no account or credentials are required. This module
is the TABASCAL adapter: it resolves each requested NORAD ID against an ordered
set of sources, applies the configurable age policies, drives the per-satellite
cache, and parses OMM-style orbital elements locally from the two TLE
lines. All filtering and element computation is done locally.

Source precedence is resolved **independently per NORAD ID**:

  1. ``extra_orbit_dir`` — user-supplied local TLE files. The record whose TLE-line
     epoch is closest to the observation epoch is chosen; it is accepted only if
     within ``extra_orbit_max_age_days`` (``None`` = unlimited). An accepted record
     wins outright — later sources are not consulted for that ID. This is *your*
     data: the remote service's age policy never applies to it, so exact replay
     of a previous run's ``used_orbits_*.json`` is always possible.
  2. Per-satellite cache — the cached record whose TLE epoch is closest to the
     observation. If it is within ``cache_reuse_max_age_days``, it avoids a
     network request. An older record within the hard ceiling remains an offline
     fallback while TABASCAL asks SatChecker for something closer.
  3. SatChecker ``get-nearest-tle`` — exact-epoch lookups run with bounded
     concurrency for the remaining IDs. Valid responses are merged into the
     per-NORAD cache and may serve nearby observations later.

Two rules govern what is then accepted:

* **Age ceiling.** Every record from source 2 or 3 must lie within
  ``remote_max_age_days`` of the observation epoch. The epoch is parsed
  locally from TLE line 1 — never taken from a provider field — and every
  accepted record's provider, epoch, signed offset and absolute age is logged.
* **Complete coverage.** Every configured NORAD ID must end up with an accepted
  TLE. If even one does not, resolution raises :class:`TLEError` during preflight,
  before the expensive subtraction begins, naming every failing ID and the
  remedies. TABASCAL never silently shrinks the requested satellite model.

"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from platformdirs import user_cache_path

import numpy as np
import pandas as pd

from tabascal import satchecker
from tabascal.satchecker import (
    TextOrbitCache,
    read_legacy_tle_records,
)
from tabascal.satchecker import SatCheckerError as TLEError  # noqa: F401  back-compat alias

# The TLE parser lives in tabascal.satchecker.tle_parse so cache validation and
# element extraction exercise the *same* code; re-exported here under this
# module's historical names.
from tabascal.satchecker.tle_parse import (
    parse_tle_elements,  # noqa: F401  re-export
    tle_epoch_jd as _tle_epoch_jd,  # noqa: F401  re-export
    validate_tle_pair,  # noqa: F401  re-export
)
# Format dispatch. Nothing below this line asks whether a record is a TLE or an
# OMM: it asks for its epoch, its elements, or whether it is valid, and these
# three answer for either kind.
from tabascal.satchecker.records import (
    KIND_FIELD,
    KIND_OMM,
    KIND_TLE,
    OMM_ELEMENT_COLUMNS,
    record_elements,
    record_epoch_jd,
    record_kind,
    validate_record,
)
from tabascal.tle_config import (  # noqa: F401  re-exported for callers
    DEFAULT_REMOTE_MAX_AGE_DAYS,
    DEFAULT_CACHE_REUSE_MAX_AGE_DAYS,
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

# A TLE line-1 epoch is quantised to ~1e-8 day (8 decimal places of a day, ~0.9 ms),
# and the datetime<->JD round-trip adds only sub-microsecond-day noise (measured
# ~3.7e-9 day). This tolerance covers one epoch quantum plus that slack (~2.6 ms), so
# ``extra_orbit_max_age_days: 0`` accepts a record matching the observation to TLE
# precision while rejecting one several ms away — matching the documented semantics.
_AGE_TOL_DAYS = 3e-8

# Above this many remote records the per-satellite log lines are replaced by a
# grouped summary; set ``TABASCAL_TLE_LOG_DETAIL=1`` to force the full listing.
_GROUPED_LOG_THRESHOLD = 12
_LOG_DETAIL_ENV = "TABASCAL_TLE_LOG_DETAIL"

# Preflight and execution must derive the same observation epoch. This tolerance
# (~86 ms) absorbs float-summation noise while still catching a genuine divergence
# such as a different unit guard or a different per-integration row selection.
_EPOCH_AGREEMENT_TOL_DAYS = 1e-6

# Source labels used in logs, errors and provenance.
_SRC_EXTRA = "extra_orbit_dir"
_SRC_CACHE = "managed per-satellite cache"
# Qualified with the endpoint that answered, so a log or a coverage error says
# which of the two archives a record came from.
_SRC_SATCHECKER = "SatChecker"


# ---------------------------------------------------------------------------
# TLE cache directory helpers
# ---------------------------------------------------------------------------

def orbit_cache_dir() -> Path:
    """Return the managed TLE cache directory, creating it if possible.

    The directory is resolved in priority order:
    1. ``ORBIT_CACHE_DIR`` environment variable (if set).
    2. The platform user-cache directory (e.g. ``~/.cache/orbit-cache`` on Linux,
       ``~/Library/Caches/orbit-cache`` on macOS).

    A directory that cannot be created (read-only filesystem, no permission,
    quota) is *not* an error here: the path is returned regardless, reads then
    miss and writes are reported and skipped, so a run with a valid fetch is
    never lost to an unusable cache location.
    """
    p = Path(os.environ.get("ORBIT_CACHE_DIR") or user_cache_path("orbit-cache"))
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return p


# ---------------------------------------------------------------------------
# Configuration validation (back-compat shim over tabascal.tle_config)
# ---------------------------------------------------------------------------

def _validate_max_age(extra_orbit_max_age_days) -> Optional[float]:
    """Validate ``extra_orbit_max_age_days``: ``None`` or a non-negative number."""
    return validate_age_days(extra_orbit_max_age_days, "extra_orbit_max_age_days")


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
    remote_max_age_days: Optional[float]
    resolved: dict[int, ResolvedTLE] = field(default_factory=dict)
    rejected: dict[int, RejectedTLE] = field(default_factory=dict)
    #: Why the service could not answer for an ID, when it was asked and failed.
    #: Kept separate from ``rejected``: a rejection is a record we saw and judged,
    #: whereas this is the absence of an answer. Without it a coverage failure
    #: during an outage reads as "this satellite does not exist", which is a
    #: different problem with different remedies.
    service_errors: dict[int, Exception] = field(default_factory=dict)

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
    """Populate OMM-style element columns by deriving them from each row.

    A TLE row is parsed from its two lines; an OMM row's element columns are
    read directly, with the semi-major axis recomputed from the mean motion so
    both kinds agree. Columns are assigned (overwriting any element columns
    already present in a legacy Space-Track cache file) so the locally derived
    values always win and no duplicate columns are produced.
    """
    tles = tles.copy()
    parsed = pd.DataFrame(
        [record_elements(r) for _, r in tles.iterrows()],
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


# ---------------------------------------------------------------------------
# Per-ID source resolution
# ---------------------------------------------------------------------------

def _select_from_extra_dir(
    extra_orbit_dir: str,
    wanted: set[int],
    obs_epoch_jd: float,
    max_age_days: Optional[float],
) -> tuple[dict[int, ResolvedTLE], dict[int, RejectedTLE]]:
    """Resolve IDs from ``extra_orbit_dir`` with per-ID nearest + age policy.

    Returns the IDs whose nearest local TLE is within ``max_age_days`` of
    *obs_epoch_jd* (``None`` = unlimited), plus the rejected near-misses. The age
    is measured from the TLE line-1 epoch, not the filename or file modification
    time.
    """
    resolved: dict[int, ResolvedTLE] = {}
    rejected: dict[int, RejectedTLE] = {}
    records = read_legacy_tle_records(extra_orbit_dir)
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
            embedded_id = validate_record(row)
            if embedded_id != nid:
                raise ValueError(
                    f"record belongs to satellite {embedded_id}, not {nid}"
                )
            epoch_jd = record_epoch_jd(row)
        except (ValueError, TypeError) as e:
            print(f"  {nid}: invalid extra_orbit_dir record rejected — {e}")
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
                f"  {nid}: from extra_orbit_dir "
                f"(epoch {abs(offset):.3f} d from observation)"
            )
        else:
            rejected[int(nid)] = RejectedTLE(
                norad_id=int(nid),
                source=_SRC_EXTRA,
                provider=None,
                epoch_jd=epoch_jd,
                offset_days=offset,
                reason=f"extra_orbit_max_age_days={max_age_days}",
            )
            print(
                f"  {nid}: extra_orbit_dir record rejected — {abs(offset):.3f} d old "
                f"> extra_orbit_max_age_days={max_age_days}; trying managed cache"
            )
    return resolved, rejected


def _select_from_records(
    records: pd.DataFrame,
    wanted: set[int],
    reference_epoch_jd: float,
) -> dict[int, dict]:
    """One record per wanted ID from a normalised record frame.

    The service may legitimately carry several distinct records for one NORAD
    ID. When it does, the one whose epoch is nearest *reference_epoch_jd* is
    chosen, so the selection is deterministic and independent of row order. The
    epoch comes from :func:`~tabascal.satchecker.records.record_epoch_jd`, which
    is a row-wise call rather than a column map because a frame may mix kinds
    for one satellite around the archive handover.
    """
    resolved: dict[int, dict] = {}
    if not len(records):
        return resolved
    records = records.copy()
    records["NORAD_CAT_ID"] = pd.to_numeric(records["NORAD_CAT_ID"]).astype(int)
    match = records[records["NORAD_CAT_ID"].isin(wanted)]
    for nid, group in match.groupby("NORAD_CAT_ID"):
        if len(group) > 1:
            epochs = pd.Series(
                [record_epoch_jd(row) for _, row in group.iterrows()],
                index=group.index,
                dtype=float,
            )
            offsets = (epochs - reference_epoch_jd).abs()
            best = group.loc[offsets.idxmin()]
        else:
            best = group.iloc[0]
        resolved[int(nid)] = best.to_dict()
    return resolved


def _cached_candidates(
    cache: TextOrbitCache, wanted: set[int], obs_epoch_jd: float
) -> dict[int, dict]:
    """Select the nearest validated cached record independently for each ID."""
    selected: dict[int, dict] = {}
    for norad_id in sorted(wanted):
        records = cache.get(norad_id)
        if records.empty:
            continue
        candidates = _select_from_records(records, {norad_id}, obs_epoch_jd)
        if norad_id in candidates:
            selected[norad_id] = candidates[norad_id]
    return selected


def _accept_remote(
    candidates: dict[int, dict],
    source: str,
    obs_epoch_jd: float,
    max_age_days: Optional[float],
    resolved: dict[int, ResolvedTLE],
    rejected: dict[int, RejectedTLE],
) -> None:
    """Apply the remote age ceiling to *candidates*, updating accept/reject maps.

    The epoch comes from :func:`~tabascal.satchecker.records.record_epoch_jd`
    and is compared against the actual mean observation epoch. For a TLE that
    means re-deriving it from line 1 — a provider's own ``epoch`` field is never
    trusted. An OMM record has no lines to re-derive from, so its ``EPOCH`` is
    parsed and range-checked instead; that is a real reduction in what can be
    caught here, and is why the plausibility window exists.
    A rejected candidate is remembered (nearest one wins) so the coverage error can
    report exactly how close the best available record was; it is never silently
    re-admitted once the remaining sources are exhausted.

    One rule covers both filling a gap and improving on what is already held: a
    candidate replaces the incumbent only when it is *strictly fresher*. That makes
    refresh safe by construction — a failed or staler response leaves the existing
    record untouched — and it also stops a later source from
    quietly downgrading an earlier one.
    """
    for nid, record in candidates.items():
        provider = record.get("DATA_SOURCE") or None
        incumbent = resolved.get(nid)
        try:
            epoch_jd = record_epoch_jd(record)
        except (KeyError, ValueError, TypeError) as e:
            # Never displace a rejection that carries a real epoch and offset:
            # "the best candidate was 4.2 d away" tells the user what to do about
            # it, "unparseable" does not. The over-age branch below is symmetric
            # — it replaces an epoch-less rejection when it has a measurable one.
            if incumbent is None and nid not in rejected:
                rejected[nid] = RejectedTLE(
                    nid, source, provider, None, None, f"unparseable epoch: {e}"
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
                        reason=f"remote_max_age_days={max_age_days:g}",
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
        f"{jd_to_datetime(resolution.obs_epoch_jd).isoformat()} UTC:"
    ]
    for nid in missing:
        bad = resolution.rejected.get(nid)
        failure = resolution.service_errors.get(nid)
        if bad is None:
            if failure is not None:
                lines.append(f"  {nid}: SatChecker could not answer — {failure}")
            else:
                lines.append(
                    f"  {nid}: no record found in extra_orbit_dir, the managed "
                    f"per-satellite cache, or SatChecker"
                )
            continue
        if bad.age_days is None:
            lines.append(f"  {nid}: best candidate unusable — {bad.reason}")
        else:
            provider = f", provider {bad.provider}" if bad.provider else ""
            lines.append(
                f"  {nid}: best candidate is {bad.age_days:.3f} d from the "
                f"observation (epoch {jd_to_datetime(bad.epoch_jd).isoformat()} "
                f"UTC, from {bad.source}{provider}) — rejected by {bad.reason}"
            )
        if failure is not None:
            # Both matter: how close the best record was, *and* that a fresher
            # one could not be requested.
            lines.append(f"      SatChecker could not be asked for a closer one — {failure}")

    limit = resolution.remote_max_age_days
    lines += [
        "",
        f"The remote age ceiling in force is remote_max_age_days="
        f"{'null (disabled)' if limit is None else f'{limit:g}'}. Remedies:",
    ]
    # A service failure is not the user's configuration being wrong, so lead with
    # the remedy that actually applies before the ones that change the model.
    if resolution.service_errors:
        retry_after = max(
            (
                seconds
                for seconds in (
                    getattr(error, "retry_after", None)
                    for error in resolution.service_errors.values()
                )
                if seconds is not None
            ),
            default=None,
        )
        when = (
            f" It asked for {retry_after:g} s before the next request."
            if retry_after is not None
            else ""
        )
        lines.append(
            f"  - SatChecker did not answer for "
            f"{len(resolution.service_errors)} of these.{when} Re-run when the "
            "service is reachable; nothing about the configuration need change"
        )
    lines += [
        "  - put an acceptable TLE for these satellites in a directory and pass "
        "--extra-orbit-dir <dir> (or set satellites.extra_orbit_dir)",
        "  - deliberately change satellites.remote_max_age_days (null removes "
        "the ceiling entirely; this is an expert opt-out, not a default)",
        "  - remove these NORAD IDs from satellites.norad_ids",
        "",
        "TABASCAL will not silently omit a configured satellite from the RFI "
        "model: the run stops here rather than subtracting an incomplete one.",
    ]
    return TLEError("\n".join(lines))


# ---------------------------------------------------------------------------
# Service acquisition
# ---------------------------------------------------------------------------

def _fetch_from_service(
    to_fetch: list[int],
    obs_epoch_jd: float,
    remote_max_age: Optional[float],
    cache,
    resolution: TLEResolution,
    max_workers: int,
) -> None:
    """Ask SatChecker for *to_fetch*, falling back to its other archive.

    SatChecker keeps two archives that do not overlap — TLEs up to 2026-07-11,
    OMM from 2026-07-12 — so the observation epoch decides which endpoint to ask
    first. In the common case that is the whole story: one request per satellite,
    answered from the right archive.

    The fallback exists because neither endpoint reports "I have nothing that
    near". Ask ``get-nearest-omm`` for a 2021 epoch and it returns its earliest
    2026 record with nothing to flag the 4.6-year gap; ask ``get-nearest-tle``
    for a 2027 epoch and it returns the last TLE ever published. Both are
    rejected here by the age ceiling, which is exactly the signal that the
    record wanted lives in the *other* archive. Within a few days either side of
    the handover that is the normal case, not an exceptional one.

    An unresolved ID after the first pass therefore earns one more request. What
    does **not** earn one is an outage: a transport failure, an HTTP 429, or a
    uniform wall of rejections means the service cannot serve us, and asking a
    down service a different question is still asking a down service. So the
    batch's ``outage`` stops the loop, while a per-ID response failure or an
    over-age record does not.

    Each pass merges its valid records into the cache before they are judged, so
    a record rejected on age is still available offline to a later run whose
    epoch it does suit.
    """
    remaining = list(to_fetch)
    endpoints = satchecker.nearest_endpoints_for(obs_epoch_jd)

    for attempt, (endpoint, fetch_nearest) in enumerate(endpoints):
        if not remaining:
            return
        if attempt:
            print(
                f"  {len(remaining)} ID(s) unresolved from {endpoints[0][0]}; "
                f"trying {endpoint} — the archives meet at "
                f"{jd_to_datetime(satchecker.HANDOVER_JD).date()} and an "
                "observation near that boundary can fall either side of it"
            )
        else:
            print(
                f"Fetching {len(remaining)} nearest record(s) from SatChecker "
                f"{endpoint} with up to {min(max_workers, len(remaining))} "
                f"concurrent requests: {remaining}"
            )

        batch = satchecker.fetch_nearest_batch(
            remaining,
            obs_epoch_jd,
            fetch_nearest=fetch_nearest,
            endpoint=endpoint,
            max_workers=max_workers,
        )
        if not batch.records.empty:
            for norad_id, records in batch.records.groupby("NORAD_CAT_ID"):
                satchecker.store_or_warn(
                    lambda nid=int(norad_id), rows=records: cache.store(nid, rows),
                    cache.path(int(norad_id)),
                    f"orbit cache for NORAD {int(norad_id)}",
                )
            _accept_remote(
                _select_from_records(batch.records, set(remaining), obs_epoch_jd),
                f"{_SRC_SATCHECKER} ({endpoint})",
                obs_epoch_jd,
                remote_max_age,
                resolution.resolved,
                resolution.rejected,
            )

        # Keep why the service could not answer, for the IDs still without a
        # record. Discarding it makes an outage indistinguishable from a
        # satellite that genuinely has no record — the same error text, but
        # remedies that do not include the only one that works: try again.
        for norad_id, error in batch.errors.items():
            if norad_id not in resolution.resolved:
                resolution.service_errors[norad_id] = error

        if batch.outage is not None:
            return
        remaining = [nid for nid in remaining if nid not in resolution.resolved]
        # An ID the fallback resolved is no longer a service failure, whatever
        # the first pass recorded against it.
        for norad_id in list(resolution.service_errors):
            if norad_id in resolution.resolved:
                del resolution.service_errors[norad_id]


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def resolve_tles(
    norad_ids,
    obs_epoch_jd: float,
    extra_orbit_dir: Optional[str] = None,
    extra_orbit_max_age_days: Optional[float] = None,
    remote_max_age_days: Optional[float] = DEFAULT_REMOTE_MAX_AGE_DAYS,
    cache_reuse_max_age_days: Optional[float] = DEFAULT_CACHE_REUSE_MAX_AGE_DAYS,
    max_workers: int = satchecker.MAX_WORKERS,
) -> TLEResolution:
    """Resolve every requested NORAD ID at *obs_epoch_jd*, without raising on gaps.

    Returns the full :class:`TLEResolution` — accepted records, rejected
    near-misses and the epochs everything was judged against. Callers decide what
    an incomplete result means; :func:`require_complete_coverage` is the policy
    TABASCAL runs use.
    """
    requested = normalise_norad_ids(norad_ids)
    extra_max_age = validate_age_days(extra_orbit_max_age_days, "extra_orbit_max_age_days")
    remote_max_age = validate_age_days(remote_max_age_days, "remote_max_age_days")
    reuse_max_age = validate_age_days(
        cache_reuse_max_age_days, "cache_reuse_max_age_days"
    )
    if reuse_max_age is not None and remote_max_age is not None and reuse_max_age > remote_max_age:
        raise TLEConfigurationError(
            f"cache_reuse_max_age_days ({reuse_max_age:g}) must not exceed "
            f"remote_max_age_days ({remote_max_age:g})"
        )
    obs_epoch_jd = float(obs_epoch_jd)

    resolution = TLEResolution(
        requested=requested,
        obs_epoch_jd=obs_epoch_jd,
        remote_max_age_days=remote_max_age,
    )
    if not requested:
        return resolution

    print(f"TLE requested epoch    : {jd_to_datetime(obs_epoch_jd).isoformat()} UTC")

    wanted = set(requested)

    # 1. extra_orbit_dir (per-ID precedence + its own age policy)
    if extra_orbit_dir:
        print(
            f"TLE extra dir          : {Path(extra_orbit_dir).resolve()} "
            f"(max age {'unlimited' if extra_max_age is None else f'{extra_max_age:g} d'})"
        )
        # A directory that is not there was almost certainly meant to be. Staying
        # silent turns a typo in a replay path into a run that quietly models
        # different satellites than the ones asked for, while the line above
        # implies the directory was searched.
        if not Path(extra_orbit_dir).is_dir():
            print(
                "  warning: this extra_orbit_dir does not exist (or is not a "
                "directory); no local TLEs will be found there. Check the path "
                "if you meant to supply your own TLEs."
            )
        from_extra, extra_rejected = _select_from_extra_dir(
            extra_orbit_dir, wanted, obs_epoch_jd, extra_max_age
        )
        resolution.resolved.update(from_extra)
        resolution.rejected.update(extra_rejected)

    remaining = wanted - set(resolution.resolved)

    # 2. Managed per-NORAD cache. A sufficiently close record is a cache hit. An
    # older but still acceptable record is retained as an offline fallback while
    # the service is asked whether it now has something closer.
    if remaining:
        cache = TextOrbitCache(orbit_cache_dir())
        cached = _cached_candidates(cache, remaining, obs_epoch_jd)

        # Every acceptable cached record becomes its ID's incumbent *before* any
        # request goes out. Two things depend on that: it is what gives
        # _accept_remote's strictly-fresher rule something to compare a response
        # against (otherwise a staler response would be accepted unopposed), and
        # it is the offline fallback if the request never comes back.
        _accept_remote(
            cached,
            _SRC_CACHE,
            obs_epoch_jd,
            remote_max_age,
            resolution.resolved,
            resolution.rejected,
        )

        # Whether to still ask the service is a *separate* question from whether
        # we already hold something usable. Only a record that is both within the
        # reuse threshold and actually accepted suppresses the request: without
        # the intersection, `cache_reuse_max_age_days: null` would make every
        # cached record a hit — including ones the hard ceiling then rejects —
        # and the ID would never be fetched at all.
        near_enough_to_reuse = {
            norad_id
            for norad_id, record in cached.items()
            if reuse_max_age is None
            or abs(record_epoch_jd(record) - obs_epoch_jd)
            <= reuse_max_age + _AGE_TOL_DAYS
        }
        to_fetch = sorted(remaining - (near_enough_to_reuse & set(resolution.resolved)))

        # 3. Exact-epoch nearest lookups for cache misses/stale cache candidates,
        # against whichever archive the observation epoch falls in — with the
        # other one as a fallback. See _fetch_from_service.
        if to_fetch:
            _fetch_from_service(
                to_fetch,
                obs_epoch_jd,
                remote_max_age,
                cache,
                resolution,
                max_workers,
            )

            # A service failure — or a response no fresher than what we hold —
            # does not invalidate a cached record within the hard ceiling. Those
            # records are already the incumbents, so nothing has to be recovered
            # here; report the ones the request did not improve on.
            retained = [
                nid
                for nid in to_fetch
                if nid in resolution.resolved
                and resolution.resolved[nid].source == _SRC_CACHE
            ]
            if retained:
                print(
                    f"  SatChecker did not improve {len(retained)} ID(s); "
                    "continuing with acceptable cached records"
                )

    _report_remote_selection(resolution)
    return resolution


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
    extra_orbit_dir: Optional[str] = None,
    extra_orbit_max_age_days: Optional[float] = None,
    remote_max_age_days: Optional[float] = DEFAULT_REMOTE_MAX_AGE_DAYS,
    cache_reuse_max_age_days: Optional[float] = DEFAULT_CACHE_REUSE_MAX_AGE_DAYS,
    max_workers: int = satchecker.MAX_WORKERS,
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
                extra_orbit_dir=extra_orbit_dir,
                extra_orbit_max_age_days=extra_orbit_max_age_days,
                remote_max_age_days=remote_max_age_days,
                cache_reuse_max_age_days=cache_reuse_max_age_days,
                max_workers=max_workers,
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
    issue its own requests. Divergent
    outcomes are worse still — some ranks exiting while others go on to a JAX
    collective is a hang, not an error.

    Workers therefore never call the provider themselves: whatever process 0
    decided — the accepted records or the failure — is what every process acts
    on, so the run either proceeds from one identical satellite set or fails
    coherently everywhere. Only raw record columns and the epochs cross the
    wire; every process re-derives the orbital elements locally, so the parsed
    values are bit-identical rather than serialisation-rounded. For a TLE that
    means re-parsing the two lines; for an OMM, whose elements have no line
    encoding to re-parse, it means the element values themselves must survive
    the hop exactly — see :func:`_wire_record`. Rejection diagnostics stay on
    process 0, which has already formatted them into the error text being
    shared.

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


#: Identity and provenance, carried whatever the kind.
_WIRE_COMMON_COLUMNS = ("OBJECT_NAME", "DATA_SOURCE")

#: Per-kind text payload — everything the worker re-derives its elements from,
#: for the kind that encodes them as text.
_WIRE_TEXT_COLUMNS = {
    KIND_TLE: ("TLE_LINE1", "TLE_LINE2"),
    KIND_OMM: ("EPOCH", "OBJECT_ID"),
}

#: Per-kind numeric payload. These cross the wire as JSON *numbers*; see
#: :func:`_wire_record`.
_WIRE_NUMBER_COLUMNS = {
    KIND_TLE: (),
    KIND_OMM: OMM_ELEMENT_COLUMNS,
}


def _wire_record(record: dict) -> dict:
    """JSON-safe projection of one record onto the columns workers actually need.

    A TLE crosses as its two lines and every rank re-parses them, so the ranks
    compute independently and still agree bit for bit — they are running the
    same parser over the same 69 characters. An OMM record has no lines: the
    element values *themselves* have to survive the hop, and if they do not, the
    ranks diverge in their trajectory priors with no error anywhere. That is
    wrong science rather than a crash, so the two kinds are projected
    differently on purpose.

    OMM elements are therefore emitted as JSON **numbers**, not strings.
    ``json.dump`` writes a float through ``repr``, which in Python 3 is the
    shortest representation that round-trips exactly, so ``json.loads`` returns
    the identical float. Routing them through the ``str(value)`` path would
    happen to round-trip too, but a mixed-type projection — some numbers as
    text, some as numbers — is what a later reader gets wrong, so it is avoided
    rather than relied on.
    """
    kind = record_kind(record)
    out = {"NORAD_CAT_ID": int(record["NORAD_CAT_ID"]), KIND_FIELD: kind}
    for col in _WIRE_COMMON_COLUMNS + _WIRE_TEXT_COLUMNS[kind]:
        value = record.get(col)
        out[col] = None if value is None or pd.isna(value) else str(value)
    for col in _WIRE_NUMBER_COLUMNS[kind]:
        out[col] = float(record[col])
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

#: What each kind needs written out to be readable back as itself. A TLE needs
#: only its lines — every element is encoded in them. An OMM needs its epoch and
#: its seven elements, because nothing else carries them.
_REPLAY_COLUMNS = {
    KIND_TLE: (KIND_FIELD, "OBJECT_NAME", "TLE_LINE1", "TLE_LINE2"),
    KIND_OMM: (KIND_FIELD, "OBJECT_NAME", "OBJECT_ID", "EPOCH", *OMM_ELEMENT_COLUMNS),
}


def _replay_record(norad_id: int, record: dict) -> dict:
    """One record projected onto the columns a replay file needs.

    Derived columns are dropped: ``EPOCH_JD`` and ``SEMIMAJOR_AXIS`` are computed
    from the others on every read, so writing them would create a second copy
    that a later edit could silently contradict.
    """
    kind = record_kind(record)
    out = {"NORAD_CAT_ID": int(norad_id), KIND_FIELD: kind}
    for column in _REPLAY_COLUMNS[kind]:
        value = record.get(column)
        if value is not None and not pd.isna(value):
            out[column] = value
    return out


def save_orbits_for_reuse(path, norad_ids, records) -> Optional[str]:
    """Write the orbit records a run used to *path* in ``extra_orbit_dir`` format.

    The file is a pandas-oriented JSON carrying, per record, exactly what
    :func:`read_legacy_tle_records` needs to read it back as the same record — a
    TLE's two lines, or an OMM's epoch and elements. A later run reproduces this
    run's trajectory priors by passing the file's directory via
    ``--extra-orbit-dir`` (with the default unlimited
    ``extra_orbit_max_age_days``), independent of the shared cache, of what
    SatChecker serves by then, and of the remote age ceiling.

    ``RECORD_KIND`` is written explicitly. Inference exists for exports we did
    not write; for a file tabascal produced itself there is no reason to make a
    later reader guess.

    ``norad_ids`` and ``records`` are the aligned sequences produced by the
    element fetchers. Returns the written path, or ``None`` when there is
    nothing to save.
    """
    ids = list(norad_ids or [])
    rows = list(records) if records is not None else []
    if not ids or not rows:
        return None
    frame = pd.DataFrame(
        [_replay_record(nid, record) for nid, record in zip(ids, rows)]
    )
    path = str(path)
    frame.to_json(path)
    return path


#: Historical name, from when every record was a TLE.
save_tles_for_reuse = save_orbits_for_reuse


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
    the rest receive its outcome, rather than each rank contacting SatChecker
    and reaching its own coverage verdict.
    """
    if not tle_config.norad_ids:
        return TLEResolution(
            requested=[],
            obs_epoch_jd=float("nan"),
            remote_max_age_days=tle_config.remote_max_age_days,
        )

    print(f"Preflight TLE check    : NORAD IDs {tle_config.norad_ids}")

    resolution = resolve_shared(
        lambda: require_complete_coverage(
            resolve_tles(
                tle_config.norad_ids,
                _ms_mean_epoch_jd(ms_path),
                extra_orbit_dir=tle_config.extra_orbit_dir,
                extra_orbit_max_age_days=tle_config.extra_orbit_max_age_days,
                remote_max_age_days=tle_config.remote_max_age_days,
                cache_reuse_max_age_days=tle_config.cache_reuse_max_age_days,
            )
        )
    )

    n_extra = sum(1 for e in resolution.resolved.values() if not e.remote)
    print(
        f"TLE preflight OK       : {len(resolution.resolved)} of "
        f"{len(resolution.requested)} satellites resolved "
        f"({n_extra} from extra_orbit_dir, "
        f"{len(resolution.resolved) - n_extra} from SatChecker/managed cache)"
    )
    print(
        "  Local TLE files are searched with --extra-orbit-dir <dir>; "
        "ORBIT_CACHE_DIR=<dir> relocates the managed cache and is not an "
        "additional source."
    )
    return resolution


def check_epoch_agreement(resolution: TLEResolution, times_jd) -> None:
    """Verify execution's observation epoch matches the one preflight resolved at.

    The epoch sets every age comparison, so a divergence between the preflight MS
    read and the times
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
