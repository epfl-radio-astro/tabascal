"""Validated per-NORAD cache for immutable SatChecker TLE records.

Each satellite has one small, atomically-written JSON file containing every
validated TLE learned for it. A resolver can reuse one record for multiple nearby
observation epochs by comparing the epoch encoded in line 1 with its configurable
cache-reuse age. There are no catalogue buckets, snapshots, or settling states.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from glob import glob
from pathlib import Path

import pandas as pd

from .tle_parse import validate_tle_pair


SCHEMA_VERSION = 1
REQUIRED_COLUMNS = ("NORAD_CAT_ID", "TLE_LINE1", "TLE_LINE2")


class CacheValidationError(ValueError):
    """A per-satellite cache file is structurally or semantically invalid."""


def _validated_records(records, expected_norad_id: int) -> pd.DataFrame:
    """Return a validated record frame belonging entirely to one NORAD ID."""
    frame = pd.DataFrame(records)
    if frame.empty or not all(column in frame.columns for column in REQUIRED_COLUMNS):
        raise CacheValidationError("TLE cache has no usable records")
    for column in REQUIRED_COLUMNS:
        if frame[column].isnull().any():
            raise CacheValidationError(f"TLE cache has null values in {column}")
    try:
        ids = pd.to_numeric(frame["NORAD_CAT_ID"])
    except (TypeError, ValueError) as error:
        raise CacheValidationError(f"NORAD_CAT_ID is not numeric: {error}") from error
    if any(not math.isfinite(float(value)) or float(value) != round(float(value)) for value in ids):
        raise CacheValidationError("NORAD_CAT_ID contains non-finite or non-integer values")
    frame["NORAD_CAT_ID"] = ids.astype(int)
    if set(frame["NORAD_CAT_ID"]) != {int(expected_norad_id)}:
        raise CacheValidationError(
            f"TLE cache for {expected_norad_id} contains records for another satellite"
        )

    for line1, line2 in zip(frame["TLE_LINE1"], frame["TLE_LINE2"]):
        try:
            embedded_id = validate_tle_pair(line1, line2)
        except (TypeError, ValueError) as error:
            raise CacheValidationError(f"invalid TLE for {expected_norad_id}: {error}") from error
        if embedded_id != int(expected_norad_id):
            raise CacheValidationError(
                f"TLE lines belong to satellite {embedded_id}, not {expected_norad_id}"
            )
    return frame.reset_index(drop=True)


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


class TextTLECache:
    """Human-readable, per-NORAD JSON cache under one directory."""

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)

    def path(self, norad_id: int) -> Path:
        return self.cache_dir / f"tle-{int(norad_id)}.json"

    def get(self, norad_id: int) -> pd.DataFrame:
        """Return all validated cached records for *norad_id*, or an empty frame."""
        path = self.path(norad_id)
        if not path.exists():
            return pd.DataFrame()
        try:
            with open(path) as handle:
                envelope = json.load(handle)
            if not isinstance(envelope, dict):
                raise CacheValidationError("TLE cache envelope is not an object")
            if envelope.get("schema_version") != SCHEMA_VERSION:
                raise CacheValidationError(
                    f"unsupported schema_version {envelope.get('schema_version')!r}"
                )
            if envelope.get("norad_id") != int(norad_id):
                raise CacheValidationError("TLE cache envelope has the wrong NORAD ID")
            return _validated_records(envelope.get("records") or [], int(norad_id))
        except (OSError, ValueError, TypeError):
            return pd.DataFrame()

    def store(self, norad_id: int, records: pd.DataFrame) -> None:
        """Merge newly fetched immutable records into one satellite's cache."""
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
        dedupe_columns = [
            column
            for column in ("NORAD_CAT_ID", "TLE_LINE1", "TLE_LINE2", "DATA_SOURCE")
            if column in merged.columns
        ]
        merged = merged.drop_duplicates(subset=dedupe_columns, keep="last")
        merged = _validated_records(merged, norad_id)
        envelope = {
            "schema_version": SCHEMA_VERSION,
            "norad_id": norad_id,
            "records": merged.to_dict(orient="records"),
        }
        _atomic_write_json(self.path(norad_id), envelope)


def read_legacy_tle_records(directory) -> pd.DataFrame:
    """Read explicit user/replay ``*.json`` TLE tables from *directory*.

    Managed ``tle-<NORAD>.json`` envelopes are intentionally not interpreted as
    explicit input. Only pandas-oriented files exposing the three required columns
    are collected.
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
        if not all(column in frame.columns for column in REQUIRED_COLUMNS):
            continue
        keep = [
            column
            for column in ("NORAD_CAT_ID", "OBJECT_NAME", "TLE_LINE1", "TLE_LINE2")
            if column in frame.columns
        ]
        frames.append(frame[keep].copy())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
