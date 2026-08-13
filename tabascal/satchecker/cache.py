"""Validated per-NORAD cache for immutable SatChecker orbit records.

Each satellite has one small, atomically-written JSON file containing every
validated record learned for it. A resolver can reuse one record for multiple
nearby observation epochs by comparing its epoch with the configurable
cache-reuse age. There are no catalogue buckets, snapshots, or settling states.

A single satellite's file may hold both record kinds at once, and around the
2026-07-12 archive handover it usually will: the last TLEs SatChecker ever
published for it, and the OMM element sets that follow. Which one a given
observation gets is decided by epoch distance in :mod:`tabascal.orbit`, not here.
Everything below that has to be kind-aware is: what columns a record must have,
what makes two records duplicates, and what "valid" means.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Callable

import pandas as pd

from .records import (
    KIND_FIELD,
    KIND_OMM,
    KIND_TLE,
    OMM_ELEMENT_COLUMNS,
    record_kind,
    validate_record,
)


#: Bumped from 1 when records stopped being TLE-only. The bump costs nothing:
#: :meth:`TextOrbitCache.get` already treats an unusable file as a warned cache
#: miss, so v1 files self-evict and are re-fetched with a clear log line instead
#: of needing a migration path.
SCHEMA_VERSION = 2

#: What a record of each kind must carry to be worth validating at all.
REQUIRED_COLUMNS_BY_KIND = {
    KIND_TLE: ("NORAD_CAT_ID", "TLE_LINE1", "TLE_LINE2"),
    KIND_OMM: ("NORAD_CAT_ID", "EPOCH", *OMM_ELEMENT_COLUMNS),
}

#: What makes two records for one satellite the same record. A TLE is identified
#: by its lines, which encode the epoch; an OMM has no lines, so its epoch is the
#: identifying field.
DEDUPE_COLUMNS_BY_KIND = {
    KIND_TLE: ("NORAD_CAT_ID", "TLE_LINE1", "TLE_LINE2", "DATA_SOURCE"),
    KIND_OMM: ("NORAD_CAT_ID", "EPOCH", "DATA_SOURCE"),
}

#: The TLE column set, under its historical name.
REQUIRED_COLUMNS = REQUIRED_COLUMNS_BY_KIND[KIND_TLE]


class CacheValidationError(ValueError):
    """A per-satellite cache file is structurally or semantically invalid."""


def _validated_ids(frame: pd.DataFrame, expected_norad_id: int) -> pd.Series:
    """Satellite IDs from a cache frame, checked and belonging to one satellite."""
    if "NORAD_CAT_ID" not in frame.columns:
        raise CacheValidationError("orbit cache has no NORAD_CAT_ID column")
    if frame["NORAD_CAT_ID"].isnull().any():
        raise CacheValidationError("orbit cache has null values in NORAD_CAT_ID")
    try:
        ids = pd.to_numeric(frame["NORAD_CAT_ID"])
    except (TypeError, ValueError) as error:
        raise CacheValidationError(f"NORAD_CAT_ID is not numeric: {error}") from error
    if any(not math.isfinite(float(value)) or float(value) != round(float(value)) for value in ids):
        raise CacheValidationError("NORAD_CAT_ID contains non-finite or non-integer values")
    ids = ids.astype(int)
    if set(ids) != {int(expected_norad_id)}:
        raise CacheValidationError(
            f"orbit cache for {expected_norad_id} contains records for another satellite"
        )
    return ids


def _validated_records(records, expected_norad_id: int) -> pd.DataFrame:
    """Return a validated record frame belonging entirely to one NORAD ID.

    Checked row by row rather than column by column, because one file may hold
    both kinds: a column-level null check would reject a TLE row for having no
    ``MEAN_MOTION`` and an OMM row for having no ``TLE_LINE1``, when neither is
    a defect.
    """
    frame = pd.DataFrame(records)
    if frame.empty:
        raise CacheValidationError("orbit cache has no usable records")
    frame["NORAD_CAT_ID"] = _validated_ids(frame, expected_norad_id)

    for _, row in frame.iterrows():
        try:
            kind = record_kind(row)
            missing = [
                column
                for column in REQUIRED_COLUMNS_BY_KIND[kind]
                if column not in frame.columns or pd.isna(row[column])
            ]
            if missing:
                raise ValueError(f"{kind} record is missing {', '.join(missing)}")
            embedded_id = validate_record(row)
        except (TypeError, ValueError) as error:
            raise CacheValidationError(
                f"invalid record for {expected_norad_id}: {error}"
            ) from error
        if embedded_id != int(expected_norad_id):
            raise CacheValidationError(
                f"record belongs to satellite {embedded_id}, not {expected_norad_id}"
            )
    return frame.reset_index(drop=True)


def _drop_duplicates_per_kind(frame: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate within each kind, on the columns that identify that kind.

    One set of dedupe columns cannot serve both: dedupe on the TLE lines and
    every OMM record collapses to one (they all have null lines); dedupe on
    ``EPOCH`` and two genuinely different TLEs published for the same instant
    would be merged.
    """
    if frame.empty:
        return frame
    kinds = frame.apply(_kind_for_grouping, axis=1)
    kept = []
    for kind, group in frame.groupby(kinds, sort=False):
        columns = [
            column
            for column in DEDUPE_COLUMNS_BY_KIND.get(kind, ())
            if column in group.columns
        ]
        kept.append(group.drop_duplicates(subset=columns, keep="last") if columns else group)
    return pd.concat(kept).sort_index()


#: Group label for a record whose kind cannot be determined. It must survive
#: grouping rather than be dropped here: _validated_records is what rejects it,
#: and it does so with a message that says why.
_UNKNOWN_KIND = "unknown"


def _kind_for_grouping(row) -> str:
    try:
        return record_kind(row)
    except ValueError:
        return _UNKNOWN_KIND


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write *payload* atomically so a partial cache file is never observed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


class TextOrbitCache:
    """Human-readable, per-NORAD JSON cache under one directory.

    One file per satellite, holding records of either kind. Around the archive
    handover a file will typically hold both: the satellite's last TLEs and its
    first OMM element sets.
    """

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)

    def path(self, norad_id: int) -> Path:
        return self.cache_dir / f"orbit-{int(norad_id)}.json"

    def get(self, norad_id: int, log: Callable[[str], None] = print) -> pd.DataFrame:
        """Return all validated cached records for *norad_id*, or an empty frame.

        An absent file is an ordinary miss and says nothing. A file that exists
        but cannot be used is reported: silently treating it as a miss costs a
        network request on every run with nothing to indicate why, and a cache
        that never takes hold is otherwise invisible.

        This is also the whole migration path for the schema bump. A v1 file
        fails the version check, gets reported as unusable, and is overwritten by
        the next successful fetch — so nothing has to convert it.
        """
        path = self.path(norad_id)
        if not path.exists():
            return pd.DataFrame()
        try:
            with open(path) as handle:
                envelope = json.load(handle)
            if not isinstance(envelope, dict):
                raise CacheValidationError("orbit cache envelope is not an object")
            if envelope.get("schema_version") != SCHEMA_VERSION:
                raise CacheValidationError(
                    f"unsupported schema_version {envelope.get('schema_version')!r} "
                    f"(this tabascal writes {SCHEMA_VERSION})"
                )
            if envelope.get("norad_id") != int(norad_id):
                raise CacheValidationError("orbit cache envelope has the wrong NORAD ID")
            return _validated_records(envelope.get("records") or [], int(norad_id))
        except (OSError, ValueError, TypeError) as error:
            log(
                f"  warning: cached orbit file {path} is unusable ({error}); "
                "treating it as a cache miss"
            )
            return pd.DataFrame()

    def store(self, norad_id: int, records: pd.DataFrame) -> None:
        """Merge newly fetched immutable records into one satellite's cache.

        Concatenating kinds widens the frame — a TLE row gains null element
        columns and an OMM row gains null lines — which is why deduplication and
        validation are both per-row rather than per-column.
        """
        if records.empty:
            return
        norad_id = int(norad_id)
        incoming = records.copy()
        if "FETCHED_AT" not in incoming.columns:
            incoming["FETCHED_AT"] = datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        existing = self.get(norad_id)
        merged = (
            pd.concat([existing, incoming], ignore_index=True)
            if not existing.empty
            else incoming
        )
        merged = _drop_duplicates_per_kind(merged)
        merged = _validated_records(merged, norad_id)
        envelope = {
            "schema_version": SCHEMA_VERSION,
            "norad_id": norad_id,
            "records": merged.to_dict(orient="records"),
        }
        _atomic_write_json(self.path(norad_id), envelope)


#: Columns worth carrying out of an explicit user/replay file. Everything a
#: record of either kind is validated and resolved from, and nothing else.
_LEGACY_KEEP_COLUMNS = (
    "NORAD_CAT_ID",
    "OBJECT_NAME",
    "OBJECT_ID",
    KIND_FIELD,
    "TLE_LINE1",
    "TLE_LINE2",
    "EPOCH",
    *OMM_ELEMENT_COLUMNS,
)


def read_legacy_tle_records(directory) -> pd.DataFrame:
    """Read explicit user/replay ``*.json`` orbit tables from *directory*.

    A file qualifies if it exposes the required columns for *either* kind, so a
    Space-Track ``gp`` / ``gp_history`` export drops in unconverted whether it
    carries TLE lines, OMM element columns, or — as those exports usually do —
    both. No ``RECORD_KIND`` is needed: it is inferred, and a file carrying both
    resolves as a TLE, whose lines are the stronger thing to validate against.

    ``EPOCH`` is kept for the OMM records that have no other epoch. It is still
    ignored for TLEs, which re-derive theirs from line 1.

    Managed ``orbit-<NORAD>.json`` envelopes are intentionally not interpreted as
    explicit input.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return pd.DataFrame()
    frames = []
    for path in sorted(glob(str(directory / "*.json"))):
        try:
            frame = pd.read_json(path)
        except (ValueError, OSError):
            continue
        if not any(
            all(column in frame.columns for column in required)
            for required in REQUIRED_COLUMNS_BY_KIND.values()
        ):
            continue
        keep = [column for column in _LEGACY_KEEP_COLUMNS if column in frame.columns]
        frames.append(frame[keep].copy())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
