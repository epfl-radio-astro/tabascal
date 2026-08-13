"""Offline tests for the per-NORAD immutable-record cache."""

import json

from tabascal.satchecker.cache import SCHEMA_VERSION, TextOrbitCache

from .tle_helpers import block_network, jd, make_catalogue_df  # noqa: F401


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
