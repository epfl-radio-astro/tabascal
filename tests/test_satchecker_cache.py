"""Offline tests for the deterministic canonical-epoch policy and text cache.

Covers the canonical-epoch bucketing, cache-key determinism, the versioned text
envelope (validation, atomic replace, legacy reads) — all without network access.
"""

import json

import pandas as pd
import pytest

from tabascal.satchecker.cache import (
    CatalogueSnapshot,
    TextCatalogueCache,
    canonical_epoch_jd,
    canonical_stamp,
    read_legacy_tle_records,
)

from .tle_helpers import jd, make_catalogue_df, write_legacy_tle_file


def _snapshot_env(epoch_jd, pairs, **overrides):
    """A well-formed snapshot envelope dict, with fields overridable for tests."""
    records = make_catalogue_df(pairs).to_dict(orient="records")
    env = {
        "schema_version": 1,
        "requested_epoch_jd": epoch_jd,
        "catalogue_epoch_jd": epoch_jd,
        "fetched_at": "2023-02-21T13:00:00Z",
        "expected_count": len(pairs),
        "actual_count": len(pairs),
        "service_version": "1.6.0",
        "records": records,
    }
    env.update(overrides)
    return env


def _write_env(path, env):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(env))


# ---------------------------------------------------------------------------
# Canonical epoch (fixed UTC buckets)
# ---------------------------------------------------------------------------

class TestCanonicalEpoch:

    def test_same_bucket_maps_to_same_epoch(self):
        a = canonical_epoch_jd(jd(2023, 2, 21, 10, 5))
        b = canonical_epoch_jd(jd(2023, 2, 21, 11, 23))
        assert canonical_stamp(a) == canonical_stamp(b) == "20230221T110000Z"

    def test_opposite_sides_of_boundary_differ(self):
        before = canonical_epoch_jd(jd(2023, 2, 21, 11, 59, 59))
        after = canonical_epoch_jd(jd(2023, 2, 21, 12, 0, 1))
        assert canonical_stamp(before) == "20230221T110000Z"
        assert canonical_stamp(after) == "20230221T130000Z"
        assert canonical_stamp(before) != canonical_stamp(after)

    @pytest.mark.parametrize("h,mi", [(0, 0), (1, 59), (10, 5), (12, 15), (23, 30), (13, 0)])
    def test_offset_from_midpoint_at_most_one_hour(self, h, mi):
        epoch = jd(2023, 2, 21, h, mi)
        offset_hours = abs(canonical_epoch_jd(epoch, 2.0) - epoch) * 24
        assert offset_hours <= 1.0 + 1e-9

    def test_bucket_width_is_configurable(self):
        # A 6-hour bucket snaps 09:00 and 11:00 together (midpoint 09:00 in
        # [06:00, 12:00)), where a 2-hour bucket would separate them.
        six_a = canonical_stamp(canonical_epoch_jd(jd(2023, 2, 21, 7), 6.0))
        six_b = canonical_stamp(canonical_epoch_jd(jd(2023, 2, 21, 11), 6.0))
        assert six_a == six_b == "20230221T090000Z"
        two_a = canonical_stamp(canonical_epoch_jd(jd(2023, 2, 21, 7), 2.0))
        two_b = canonical_stamp(canonical_epoch_jd(jd(2023, 2, 21, 11), 2.0))
        assert two_a != two_b

    def test_key_independent_of_cache_state(self, tmp_path):
        # The cache key depends only on the request + width, never on what files
        # already exist: computing it twice, with an unrelated snapshot written in
        # between, yields the same stamp.
        epoch = jd(2023, 2, 21, 12, 30)
        first = canonical_stamp(canonical_epoch_jd(epoch))
        cache = TextCatalogueCache(tmp_path)
        cache.store_snapshot(
            CatalogueSnapshot(
                catalogue_epoch_jd=canonical_epoch_jd(jd(2020, 1, 1, 3)),
                records=make_catalogue_df([(25544, jd(2020, 1, 1, 3))]),
            )
        )
        assert canonical_stamp(canonical_epoch_jd(epoch)) == first

    def test_non_positive_interval_rejected(self):
        with pytest.raises(ValueError):
            canonical_epoch_jd(jd(2023, 2, 21, 12), 0)
        with pytest.raises(ValueError):
            canonical_epoch_jd(jd(2023, 2, 21, 12), -2)

    def test_subsecond_interval_rejected(self):
        # A bucket narrower than 1 s could collide with the whole-second filename
        # stamp, so it is rejected rather than silently allowed.
        with pytest.raises(ValueError):
            canonical_epoch_jd(jd(2023, 2, 21, 12), 0.0001)  # 0.36 s

    @pytest.mark.parametrize("interval", [2.0, 1.0, 0.5, 6.0])
    def test_distinct_canonical_epochs_have_distinct_stamps(self, interval, tmp_path):
        # The second-rounded filename stamp must be injective over distinct
        # canonical epochs for every supported bucket width.
        cache = TextCatalogueCache(tmp_path)
        by_stamp = {}
        for minutes in range(0, 1440, 3):
            epoch = jd(2023, 2, 21, 0, 0) + minutes / 1440.0
            canon = canonical_epoch_jd(epoch, interval)
            key = round((canon - 2440587.5) * 86_400 * 1_000_000)  # exact identity
            path = cache._snapshot_path(canon)
            if path in by_stamp:
                assert by_stamp[path] == key, "distinct canonical epochs share a path"
            by_stamp[path] = key


# ---------------------------------------------------------------------------
# Semantic envelope validation (load and store paths)
# ---------------------------------------------------------------------------

class TestEnvelopeValidation:

    _EP = None  # set per-test via canonical_epoch_jd

    def _canon(self):
        return canonical_epoch_jd(jd(2023, 2, 21, 12, 30))

    def test_unsupported_schema_is_a_miss(self, tmp_path):
        cache = TextCatalogueCache(tmp_path)
        canon = self._canon()
        _write_env(cache._snapshot_path(canon),
                   _snapshot_env(canon, [(25544, jd(2023, 2, 21, 13))], schema_version=999))
        assert cache.get_snapshot(canon) is None

    def test_wrong_embedded_epoch_is_a_miss(self, tmp_path):
        cache = TextCatalogueCache(tmp_path)
        canon_a = self._canon()
        canon_b = canonical_epoch_jd(jd(2023, 2, 22, 6, 30))  # a different bucket
        # Valid document, correct filename for A, but epoch field claims B.
        _write_env(cache._snapshot_path(canon_a),
                   _snapshot_env(canon_a, [(25544, jd(2023, 2, 21, 13))],
                                 catalogue_epoch_jd=canon_b))
        assert cache.get_snapshot(canon_a) is None

    def test_actual_count_mismatch_is_a_miss(self, tmp_path):
        cache = TextCatalogueCache(tmp_path)
        canon = self._canon()
        _write_env(cache._snapshot_path(canon),
                   _snapshot_env(canon, [(1, jd(2023, 2, 21, 13)), (2, jd(2023, 2, 21, 13))],
                                 actual_count=10))
        assert cache.get_snapshot(canon) is None

    def test_incomplete_snapshot_is_a_miss(self, tmp_path):
        cache = TextCatalogueCache(tmp_path)
        canon = self._canon()
        pairs = [(i, jd(2023, 2, 21, 13)) for i in range(50)]
        _write_env(cache._snapshot_path(canon),
                   _snapshot_env(canon, pairs, expected_count=100))
        assert cache.get_snapshot(canon) is None

    def test_completeness_threshold_boundary(self, tmp_path):
        cache = TextCatalogueCache(tmp_path)
        canon = self._canon()
        # 99 of 100 (== 99%) accepted; 98 of 100 (< 99%) rejected.
        ok = [(i, jd(2023, 2, 21, 13)) for i in range(99)]
        _write_env(cache._snapshot_path(canon), _snapshot_env(canon, ok, expected_count=100))
        assert cache.get_snapshot(canon) is not None

        bad = [(i, jd(2023, 2, 21, 13)) for i in range(98)]
        _write_env(cache._snapshot_path(canon), _snapshot_env(canon, bad, expected_count=100))
        assert cache.get_snapshot(canon) is None

    def test_null_required_value_is_a_miss(self, tmp_path):
        cache = TextCatalogueCache(tmp_path)
        canon = self._canon()
        env = _snapshot_env(canon, [(25544, jd(2023, 2, 21, 13))])
        env["records"][0]["TLE_LINE1"] = None
        _write_env(cache._snapshot_path(canon), env)
        assert cache.get_snapshot(canon) is None

    def test_store_side_validation_rejects_and_writes_nothing(self, tmp_path):
        cache = TextCatalogueCache(tmp_path)
        canon = self._canon()
        snap = CatalogueSnapshot(
            catalogue_epoch_jd=canon,
            records=make_catalogue_df([(1, jd(2023, 2, 21, 13)), (2, jd(2023, 2, 21, 13))]),
            actual_count=10,  # inconsistent with the two stored records
        )
        with pytest.raises(ValueError):
            cache.store_snapshot(snap)
        assert list(tmp_path.glob("catalogue-*.json")) == []

    def test_invalid_fallback_envelope_is_a_miss(self, tmp_path):
        cache = TextCatalogueCache(tmp_path)
        canon = self._canon()
        # Unsupported schema.
        _write_env(cache._extra_path(canon),
                   _snapshot_env(canon, [(25544, jd(2023, 2, 21, 13))], schema_version=999))
        assert cache.get_extra(canon).empty
        # Wrong embedded epoch.
        other = canonical_epoch_jd(jd(2023, 2, 22, 6, 30))
        _write_env(cache._extra_path(canon),
                   _snapshot_env(canon, [(25544, jd(2023, 2, 21, 13))], catalogue_epoch_jd=other))
        assert cache.get_extra(canon).empty


# ---------------------------------------------------------------------------
# Text cache round-trip, validation, atomicity
# ---------------------------------------------------------------------------

class TestTextCatalogueCache:

    def _snapshot(self, epoch_jd, pairs):
        return CatalogueSnapshot(
            catalogue_epoch_jd=canonical_epoch_jd(epoch_jd),
            records=make_catalogue_df(pairs),
            requested_epoch_jd=epoch_jd,
            expected_count=len(pairs),
            actual_count=len(pairs),
            service_version="1.6.0",
        )

    def test_store_then_get_round_trip(self, tmp_path):
        cache = TextCatalogueCache(tmp_path)
        epoch = jd(2023, 2, 21, 12, 30)
        snap = self._snapshot(epoch, [(25544, jd(2023, 2, 21, 13)), (38833, jd(2023, 2, 21, 13))])
        cache.store_snapshot(snap)

        got = cache.get_snapshot(canonical_epoch_jd(epoch))
        assert got is not None
        assert sorted(got.records["NORAD_CAT_ID"]) == [25544, 38833]
        assert got.expected_count == 2
        assert got.service_version == "1.6.0"

    def test_envelope_is_row_oriented_and_versioned(self, tmp_path):
        cache = TextCatalogueCache(tmp_path)
        epoch = jd(2023, 2, 21, 12, 30)
        cache.store_snapshot(self._snapshot(epoch, [(25544, jd(2023, 2, 21, 13))]))
        path = tmp_path / f"catalogue-{canonical_stamp(canonical_epoch_jd(epoch))}.json"
        env = json.loads(path.read_text())
        assert env["schema_version"] == 1
        assert isinstance(env["records"], list)  # row-oriented, not pandas columns
        assert env["records"][0]["NORAD_CAT_ID"] == 25544
        assert env["fetched_at"]  # provenance stamped on write

    def test_miss_returns_none(self, tmp_path):
        cache = TextCatalogueCache(tmp_path)
        assert cache.get_snapshot(canonical_epoch_jd(jd(2023, 2, 21, 12))) is None

    def test_invalid_or_partial_file_is_ignored(self, tmp_path):
        cache = TextCatalogueCache(tmp_path)
        epoch = jd(2023, 2, 21, 12, 30)
        path = tmp_path / f"catalogue-{canonical_stamp(canonical_epoch_jd(epoch))}.json"
        path.write_text('{"schema_version": 1, "records": [')  # truncated JSON
        assert cache.get_snapshot(canonical_epoch_jd(epoch)) is None

    def test_envelope_without_records_is_ignored(self, tmp_path):
        cache = TextCatalogueCache(tmp_path)
        epoch = jd(2023, 2, 21, 12, 30)
        path = tmp_path / f"catalogue-{canonical_stamp(canonical_epoch_jd(epoch))}.json"
        path.write_text('{"schema_version": 1, "records": []}')
        assert cache.get_snapshot(canonical_epoch_jd(epoch)) is None

    def test_store_is_atomic_no_tmp_left_behind(self, tmp_path):
        cache = TextCatalogueCache(tmp_path)
        epoch = jd(2023, 2, 21, 12, 30)
        cache.store_snapshot(self._snapshot(epoch, [(25544, jd(2023, 2, 21, 13))]))
        leftovers = [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"]
        assert leftovers == []
        assert (tmp_path / f"catalogue-{canonical_stamp(canonical_epoch_jd(epoch))}.json").exists()

    def test_extra_records_merge_and_dedupe(self, tmp_path):
        cache = TextCatalogueCache(tmp_path)
        epoch = canonical_epoch_jd(jd(2023, 2, 21, 12, 30))
        cache.store_extra(epoch, make_catalogue_df([(11111, jd(2023, 2, 21, 13))]))
        cache.store_extra(epoch, make_catalogue_df([(11111, jd(2023, 2, 21, 13)), (22222, jd(2023, 2, 21, 13))]))
        extra = cache.get_extra(epoch)
        assert sorted(extra["NORAD_CAT_ID"].unique()) == [11111, 22222]
        assert len(extra) == 2  # duplicate 11111 collapsed


# ---------------------------------------------------------------------------
# Legacy file reader
# ---------------------------------------------------------------------------

class TestLegacyReader:

    def test_reads_pandas_oriented_files(self, tmp_path):
        write_legacy_tle_file(tmp_path / "2023-02-21-navstar.json", [(20452, jd(2023, 2, 21, 13))])
        write_legacy_tle_file(tmp_path / "2023-02-21-galileo.json", [(38833, jd(2023, 2, 21, 13))])
        df = read_legacy_tle_records(tmp_path)
        assert sorted(df["NORAD_CAT_ID"]) == [20452, 38833]
        assert set(["NORAD_CAT_ID", "TLE_LINE1", "TLE_LINE2"]).issubset(df.columns)

    def test_skips_files_without_required_columns(self, tmp_path):
        (tmp_path / "unrelated.json").write_text('{"foo": {"0": 1}}')
        write_legacy_tle_file(tmp_path / "2023-02-21-navstar.json", [(20452, jd(2023, 2, 21, 13))])
        df = read_legacy_tle_records(tmp_path)
        assert list(df["NORAD_CAT_ID"]) == [20452]

    def test_missing_directory_returns_empty(self, tmp_path):
        assert read_legacy_tle_records(tmp_path / "nope").empty

    def test_bundled_fixtures_are_readable(self):
        from importlib.resources import files as _files
        bundled = str(_files("tabascal").joinpath("data/tles"))
        df = read_legacy_tle_records(bundled)
        assert {20452, 38833}.issubset(set(pd.to_numeric(df["NORAD_CAT_ID"]).astype(int)))
