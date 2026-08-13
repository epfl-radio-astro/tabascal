"""Offline tests for the per-NORAD immutable-record cache."""

import json

import pytest

from tabascal.satchecker.cache import SCHEMA_VERSION, TextOrbitCache
from tabascal.satchecker.records import record_epoch_jd

from .tle_helpers import (  # noqa: F401
    block_network,
    jd,
    make_catalogue_df,
    make_omm_catalogue_df,
)


EPOCH = jd(2023, 1, 1)


def test_store_and_get_round_trip(tmp_path):
    cache = TextOrbitCache(tmp_path)
    cache.store(25544, make_catalogue_df([(25544, EPOCH)]))
    loaded = cache.get(25544)
    assert len(loaded) == 1
    assert loaded.loc[0, "NORAD_CAT_ID"] == 25544


def test_records_at_distinct_epochs_are_merged(tmp_path):
    cache = TextOrbitCache(tmp_path)
    cache.store(25544, make_catalogue_df([(25544, EPOCH - 1)]))
    cache.store(25544, make_catalogue_df([(25544, EPOCH + 1)]))
    assert len(cache.get(25544)) == 2


def test_identical_records_are_deduplicated(tmp_path):
    cache = TextOrbitCache(tmp_path)
    record = make_catalogue_df([(25544, EPOCH)])
    cache.store(25544, record)
    cache.store(25544, record)
    assert len(cache.get(25544)) == 1


def test_wrong_satellite_cannot_be_stored(tmp_path):
    cache = TextOrbitCache(tmp_path)
    try:
        cache.store(25544, make_catalogue_df([(43013, EPOCH)]))
    except ValueError as error:
        assert "another satellite" in str(error)
    else:
        raise AssertionError("wrong-satellite record was cached")


def test_corrupt_or_wrong_schema_file_is_a_cache_miss(tmp_path):
    cache = TextOrbitCache(tmp_path)
    cache.path(25544).write_text("not-json")
    assert cache.get(25544).empty
    cache.path(25544).write_text(json.dumps({"schema_version": SCHEMA_VERSION + 1}))
    assert cache.get(25544).empty


def test_unusable_file_is_reported_but_an_absent_one_is_not(tmp_path):
    """A cache that never takes hold must not be invisible.

    Silently re-fetching every run gives the user nothing to debug, so an
    existing-but-unusable file warns while a plain miss stays quiet.
    """
    cache = TextOrbitCache(tmp_path)
    messages = []

    assert cache.get(25544, log=messages.append).empty
    assert messages == []

    cache.path(25544).write_text("not-json")
    assert cache.get(25544, log=messages.append).empty
    assert len(messages) == 1
    assert "unusable" in messages[0]
    assert str(cache.path(25544)) in messages[0]


def test_write_is_an_envelope_not_a_pandas_orientation(tmp_path):
    cache = TextOrbitCache(tmp_path)
    cache.store(25544, make_catalogue_df([(25544, EPOCH)]))
    payload = json.loads(cache.path(25544).read_text())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["norad_id"] == 25544
    assert isinstance(payload["records"], list)


# ---------------------------------------------------------------------------
# Schema v2: two kinds in one file
# ---------------------------------------------------------------------------

class TestMixedKindCache:
    """A satellite's file spans the archive handover, so it holds both kinds.

    Almost everything the cache does had one implicit assumption in it — that
    every record has TLE lines. Concatenating an OMM record onto a TLE frame
    widens the columns and fills the gaps with nulls, so any check that looks at
    a whole column rather than a row now sees a defect that is not one.
    """

    def test_both_kinds_coexist_for_one_satellite(self, tmp_path):
        cache = TextOrbitCache(tmp_path)
        cache.store(25544, make_catalogue_df([(25544, EPOCH)]))
        cache.store(25544, make_omm_catalogue_df([(25544, EPOCH + 40)]))
        loaded = cache.get(25544)
        assert len(loaded) == 2
        assert set(loaded["RECORD_KIND"]) == {"tle", "omm"}

    def test_an_omm_record_round_trips_alone(self, tmp_path):
        cache = TextOrbitCache(tmp_path)
        cache.store(25544, make_omm_catalogue_df([(25544, EPOCH)]))
        loaded = cache.get(25544)
        assert len(loaded) == 1
        assert record_epoch_jd(loaded.loc[0]) == pytest.approx(EPOCH, abs=1e-6)

    def test_omm_elements_survive_the_json_file_exactly(self, tmp_path):
        # Same hazard as the rank broadcast: json writes a float through repr,
        # which round-trips exactly, but only if it goes as a number.
        cache = TextOrbitCache(tmp_path)
        stored = make_omm_catalogue_df([(25544, EPOCH)])
        cache.store(25544, stored)
        loaded = cache.get(25544)
        for column in (
            "INCLINATION",
            "RA_OF_ASC_NODE",
            "ECCENTRICITY",
            "ARG_OF_PERICENTER",
            "MEAN_ANOMALY",
            "MEAN_MOTION",
            "BSTAR",
        ):
            assert loaded.loc[0, column] == stored.loc[0, column], column

    def test_omm_records_dedupe_on_epoch_not_on_absent_lines(self, tmp_path):
        # Deduping on the TLE lines would collapse every OMM record into one,
        # because they all share the same null lines.
        cache = TextOrbitCache(tmp_path)
        cache.store(25544, make_omm_catalogue_df([(25544, EPOCH)]))
        cache.store(25544, make_omm_catalogue_df([(25544, EPOCH + 1)]))
        cache.store(25544, make_omm_catalogue_df([(25544, EPOCH + 2)]))
        assert len(cache.get(25544)) == 3

    def test_identical_omm_records_are_still_deduplicated(self, tmp_path):
        cache = TextOrbitCache(tmp_path)
        record = make_omm_catalogue_df([(25544, EPOCH)])
        cache.store(25544, record)
        cache.store(25544, record)
        assert len(cache.get(25544)) == 1

    def test_tle_dedupe_is_unaffected_by_the_presence_of_omm_rows(self, tmp_path):
        cache = TextOrbitCache(tmp_path)
        tle = make_catalogue_df([(25544, EPOCH)])
        cache.store(25544, tle)
        cache.store(25544, make_omm_catalogue_df([(25544, EPOCH + 40)]))
        cache.store(25544, tle)
        loaded = cache.get(25544)
        assert len(loaded) == 2
        assert list(loaded["RECORD_KIND"]).count("tle") == 1

    def test_a_corrupted_omm_element_makes_the_file_a_miss(self, tmp_path):
        cache = TextOrbitCache(tmp_path)
        cache.store(25544, make_omm_catalogue_df([(25544, EPOCH)]))
        payload = json.loads(cache.path(25544).read_text())
        payload["records"][0]["INCLINATION"] = 999.0
        cache.path(25544).write_text(json.dumps(payload))
        messages = []
        assert cache.get(25544, log=messages.append).empty
        assert "inclination out of range" in messages[0]

    def test_a_wrong_satellite_omm_record_cannot_be_stored(self, tmp_path):
        cache = TextOrbitCache(tmp_path)
        with pytest.raises(ValueError, match="another satellite"):
            cache.store(25544, make_omm_catalogue_df([(43013, EPOCH)]))


class TestSchemaVersionBump:

    def test_the_schema_version_is_two(self):
        assert SCHEMA_VERSION == 2

    def test_a_v1_file_self_evicts_with_a_warning(self, tmp_path):
        # The whole migration path. A v1 envelope is structurally fine and its
        # records are valid TLEs — it is rejected purely on version, warned
        # about, and replaced by the next fetch. Nothing converts it.
        cache = TextOrbitCache(tmp_path)
        v1 = {
            "schema_version": 1,
            "norad_id": 25544,
            "records": make_catalogue_df([(25544, EPOCH)]).to_dict(orient="records"),
        }
        cache.path(25544).write_text(json.dumps(v1))

        messages = []
        assert cache.get(25544, log=messages.append).empty
        assert len(messages) == 1
        assert "unusable" in messages[0]
        assert "schema_version 1" in messages[0]

    def test_the_next_fetch_replaces_the_evicted_file(self, tmp_path):
        cache = TextOrbitCache(tmp_path)
        cache.path(25544).write_text(
            json.dumps({"schema_version": 1, "norad_id": 25544, "records": []})
        )
        cache.store(25544, make_catalogue_df([(25544, EPOCH)]))
        assert json.loads(cache.path(25544).read_text())["schema_version"] == 2
        assert len(cache.get(25544)) == 1
