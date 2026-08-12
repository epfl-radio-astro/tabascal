"""Offline acceptance, caching, and concurrency tests for nearest-only TLE resolution."""

import threading
import time

import pandas as pd
import pytest

from tabascal import tle
from tabascal.satchecker import client, SatCheckerTransportError
from tabascal.satchecker.cache import TextTLECache
from tabascal.satchecker.service import fetch_nearest_batch

from .tle_helpers import block_network, jd, make_catalogue_df  # noqa: F401


OBS = jd(2023, 2, 21)


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TLE_CACHE_DIR", str(tmp_path))
    return tmp_path


def _service(monkeypatch, epochs):
    calls = []

    def fetch(norad_id, epoch):
        calls.append((norad_id, epoch))
        value = epochs.get(norad_id)
        return pd.DataFrame() if value is None else make_catalogue_df([(norad_id, value)])

    monkeypatch.setattr(tle.satchecker, "fetch_nearest_tle", fetch)
    return calls


def test_cache_miss_fetches_exact_observation_epoch_and_caches(cache_dir, monkeypatch):
    calls = _service(monkeypatch, {25544: OBS - 0.1})
    result = tle.resolve_tles([25544], OBS)
    assert calls == [(25544, OBS)]
    assert result.resolved[25544].source == "SatChecker nearest-TLE"
    assert len(TextTLECache(cache_dir).get(25544)) == 1


def test_close_cache_hit_makes_no_request(cache_dir, monkeypatch):
    TextTLECache(cache_dir).store(25544, make_catalogue_df([(25544, OBS - 0.5)]))
    calls = _service(monkeypatch, {25544: OBS})
    result = tle.resolve_tles([25544], OBS, cache_reuse_max_age_days=1)
    assert calls == []
    assert result.resolved[25544].source == "managed per-satellite cache"


def test_old_acceptable_cache_is_refreshed(cache_dir, monkeypatch):
    TextTLECache(cache_dir).store(25544, make_catalogue_df([(25544, OBS - 2)]))
    calls = _service(monkeypatch, {25544: OBS - 0.1})
    result = tle.resolve_tles(
        [25544], OBS, cache_reuse_max_age_days=1, remote_tle_max_age_days=3
    )
    assert calls == [(25544, OBS)]
    assert result.resolved[25544].age_days == pytest.approx(0.1, abs=2e-8)
    assert len(TextTLECache(cache_dir).get(25544)) == 2


def test_old_acceptable_cache_is_offline_fallback(cache_dir, monkeypatch):
    TextTLECache(cache_dir).store(25544, make_catalogue_df([(25544, OBS - 2)]))

    def down(*args):
        raise SatCheckerTransportError("offline")

    monkeypatch.setattr(tle.satchecker, "fetch_nearest_tle", down)
    result = tle.resolve_tles(
        [25544], OBS, cache_reuse_max_age_days=1, remote_tle_max_age_days=3
    )
    assert result.complete
    assert result.resolved[25544].source == "managed per-satellite cache"


def test_record_beyond_hard_ceiling_is_not_recovered(cache_dir, monkeypatch):
    TextTLECache(cache_dir).store(25544, make_catalogue_df([(25544, OBS - 4)]))
    _service(monkeypatch, {25544: OBS - 4})
    result = tle.resolve_tles(
        [25544], OBS, cache_reuse_max_age_days=1, remote_tle_max_age_days=3
    )
    assert not result.complete
    with pytest.raises(tle.TLEError, match="4.000 d"):
        tle.require_complete_coverage(result)


def test_null_reuse_age_always_reuses_acceptable_cache(cache_dir, monkeypatch):
    TextTLECache(cache_dir).store(25544, make_catalogue_df([(25544, OBS - 2)]))
    calls = _service(monkeypatch, {25544: OBS})
    assert tle.resolve_tles(
        [25544], OBS, cache_reuse_max_age_days=None, remote_tle_max_age_days=3
    ).complete
    assert calls == []


def test_null_reuse_age_still_fetches_when_cache_is_over_age(cache_dir, monkeypatch):
    """`null` reuse must not suppress the request for an *unacceptable* record.

    Reuse is unlimited, so the 10-day-old cached record is "near enough to reuse"
    — but the hard ceiling rejects it. Suppressing the request on the strength of
    a record that is then thrown away would fail the run with a fresh TLE sitting
    on the service.
    """
    TextTLECache(cache_dir).store(25544, make_catalogue_df([(25544, OBS - 10)]))
    calls = _service(monkeypatch, {25544: OBS - 0.1})
    result = tle.resolve_tles(
        [25544], OBS, cache_reuse_max_age_days=None, remote_tle_max_age_days=3
    )
    assert calls == [(25544, OBS)]
    assert result.complete
    assert result.resolved[25544].source == "SatChecker nearest-TLE"


def test_staler_service_response_does_not_displace_fresher_cache(cache_dir, monkeypatch):
    """The strictly-fresher rule must govern the cache-vs-service comparison.

    The cached record is too old to suppress the request but closer to the
    observation than what the service returns, so it has to survive the refresh.
    """
    TextTLECache(cache_dir).store(25544, make_catalogue_df([(25544, OBS - 2)]))
    calls = _service(monkeypatch, {25544: OBS - 2.5})
    result = tle.resolve_tles(
        [25544], OBS, cache_reuse_max_age_days=1, remote_tle_max_age_days=3
    )
    assert calls == [(25544, OBS)]
    assert result.resolved[25544].source == "managed per-satellite cache"
    assert result.resolved[25544].age_days == pytest.approx(2.0, abs=2e-8)


def test_extra_directory_has_precedence(cache_dir, tmp_path, monkeypatch):
    extra = tmp_path / "extra"
    extra.mkdir()
    make_catalogue_df([(25544, OBS - 0.2)]).to_json(extra / "manual.json")
    calls = _service(monkeypatch, {25544: OBS})
    result = tle.resolve_tles([25544], OBS, extra_tle_dir=str(extra))
    assert result.resolved[25544].source == "extra_tle_dir"
    assert calls == []


def test_batch_uses_bounded_concurrency_and_preserves_input_order():
    active = 0
    maximum = 0
    lock = threading.Lock()

    def fetch(norad_id, epoch):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return make_catalogue_df([(norad_id, epoch)])

    result = fetch_nearest_batch([5, 4, 3, 2, 1], OBS, fetch_nearest_tle=fetch, max_workers=2)
    assert maximum == 2
    assert result.records["NORAD_CAT_ID"].tolist() == [5, 4, 3, 2, 1]


def test_batch_collects_one_failure_without_losing_other_ids():
    def fetch(norad_id, epoch):
        if norad_id == 2:
            raise SatCheckerTransportError("offline")
        return make_catalogue_df([(norad_id, epoch)])

    result = fetch_nearest_batch([1, 2, 3], OBS, fetch_nearest_tle=fetch)
    assert result.records["NORAD_CAT_ID"].tolist() == [1, 3]
    assert isinstance(result.errors[2], SatCheckerTransportError)


def test_batch_abandons_queued_requests_once_the_service_is_unreachable():
    """An outage must cost a bounded number of requests, not one per satellite.

    Only the ``max_workers`` already in flight run; the rest are abandoned rather
    than each paying the full request timeout to rediscover the same outage.
    """
    attempted = []
    lock = threading.Lock()

    def dead_service(norad_id, epoch):
        with lock:
            attempted.append(norad_id)
        time.sleep(0.05)
        raise SatCheckerTransportError("connection refused")

    result = fetch_nearest_batch(
        list(range(1, 41)), OBS, fetch_nearest_tle=dead_service, max_workers=2
    )
    assert len(attempted) < 40
    # Every requested ID still gets an error, so coverage reporting names them all.
    assert set(result.errors) == set(range(1, 41))
    assert all(
        isinstance(error, SatCheckerTransportError) for error in result.errors.values()
    )


def test_batch_response_errors_do_not_abandon_the_rest():
    """A malformed reply is one satellite's problem: the service is still up."""
    def fetch(norad_id, epoch):
        if norad_id % 2:
            raise client.SatCheckerResponseError("malformed")
        return make_catalogue_df([(norad_id, epoch)])

    result = fetch_nearest_batch(
        list(range(1, 11)), OBS, fetch_nearest_tle=fetch, max_workers=2
    )
    assert result.records["NORAD_CAT_ID"].tolist() == [2, 4, 6, 8, 10]
    assert set(result.errors) == {1, 3, 5, 7, 9}
