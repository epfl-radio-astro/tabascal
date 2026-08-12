"""Offline acceptance, caching, and concurrency tests for nearest-only TLE resolution."""

import threading
import time

import pandas as pd
import pytest

from tabascal import tle
from tabascal.satchecker import client, SatCheckerTransportError
from tabascal.satchecker import service
from tabascal.satchecker.cache import TextTLECache
from tabascal.satchecker.service import fetch_nearest_batch

from .tle_helpers import block_network, jd, make_catalogue_df, make_tle  # noqa: F401


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


def test_unparseable_candidate_does_not_erase_a_measurable_rejection():
    """The coverage error must keep the rejection that can quantify the miss.

    "The best candidate was 4 d away" tells the user which knob to turn;
    "unparseable TLE epoch" does not. Both validating layers upstream drop
    unparseable rows, so this is defence in depth exercised directly.
    """
    line1, line2 = make_tle(25544, OBS - 4)
    resolved, rejected = {}, {}

    tle._accept_remote(
        {25544: {"TLE_LINE1": line1, "TLE_LINE2": line2}},
        "managed per-satellite cache", OBS, 3.0, resolved, rejected,
    )
    assert rejected[25544].age_days == pytest.approx(4.0, abs=2e-8)

    tle._accept_remote(
        {25544: {"TLE_LINE1": "garbage", "TLE_LINE2": "garbage"}},
        "SatChecker nearest-TLE", OBS, 3.0, resolved, rejected,
    )
    assert rejected[25544].age_days == pytest.approx(4.0, abs=2e-8)


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


@pytest.mark.parametrize("workers", [2, 5, 16])
@pytest.mark.parametrize("latency", [0.0, 0.02])
def test_outage_costs_at_most_max_workers_requests(workers, latency):
    """An outage must cost exactly ``max_workers`` requests — no more, ever.

    The bound has to hold at *any* worker count and *any* failure latency. That
    is why requests are submitted incrementally: queueing all of them and
    cancelling the remainder leaks badly when failures return fast, because the
    pool's workers drain the queue faster than the cancel catches it.

    ``latency=0`` is the hard case (a refused connection); 20 ms stands in for a
    429 arriving in one round trip.
    """
    attempted = []
    lock = threading.Lock()

    def dead_service(norad_id, epoch):
        with lock:
            attempted.append(norad_id)
        if latency:
            time.sleep(latency)
        raise SatCheckerTransportError("connection refused")

    result = fetch_nearest_batch(
        list(range(1, 201)), OBS, fetch_nearest_tle=dead_service,
        max_workers=workers, log=lambda _m: None,
    )
    assert len(attempted) <= workers
    # Every requested ID still gets an error, so coverage reporting names them all.
    assert set(result.errors) == set(range(1, 201))
    assert all(
        isinstance(error, SatCheckerTransportError) for error in result.errors.values()
    )


def test_incremental_submission_still_fetches_every_id_when_healthy():
    """Topping the in-flight set up must not drop IDs off the end of the batch."""
    result = fetch_nearest_batch(
        list(range(1, 51)), OBS,
        fetch_nearest_tle=lambda nid, epoch: make_catalogue_df([(nid, epoch)]),
        max_workers=4,
    )
    assert result.records["NORAD_CAT_ID"].tolist() == list(range(1, 51))
    assert result.errors == {}


def test_rate_limit_stops_the_batch_and_reports_the_wait():
    """A 429 is the service asking us to stop; the rest of the list must not go out."""
    attempted = []
    lock = threading.Lock()

    def limited(norad_id, epoch):
        with lock:
            attempted.append(norad_id)
        raise client.SatCheckerRateLimitError("slow down", retry_after=90.0)

    result = fetch_nearest_batch(
        list(range(1, 201)), OBS, fetch_nearest_tle=limited, max_workers=3,
        log=lambda _m: None,
    )
    assert len(attempted) <= 3
    assert set(result.errors) == set(range(1, 201))


def test_service_failure_is_not_reported_as_a_missing_satellite(cache_dir, monkeypatch):
    """An outage and a non-existent satellite are different problems.

    Reporting "no record found" when the service was simply refusing us asserts
    something untrue about the catalogue and offers remedies — supply local TLEs,
    relax the ceiling, drop the ID — that all miss the only one that works.
    """
    def rate_limited(norad_id, epoch):
        raise client.SatCheckerRateLimitError(
            "HTTP 429 — it is rate-limiting this client", retry_after=90.0
        )

    monkeypatch.setattr(tle.satchecker, "fetch_nearest_tle", rate_limited)
    result = tle.resolve_tles([25544], OBS)
    assert 25544 in result.service_errors

    with pytest.raises(tle.TLEError) as caught:
        tle.require_complete_coverage(result)
    text = str(caught.value)
    assert "no record found" not in text
    assert "rate-limiting" in text
    assert "90 s" in text            # the Retry-After hint reaches the error
    assert "Re-run when the service is reachable" in text


def test_service_failure_is_reported_alongside_an_over_age_candidate(
    cache_dir, monkeypatch
):
    """Both facts matter: how close the best record was, and that it could not be improved."""
    TextTLECache(cache_dir).store(25544, make_catalogue_df([(25544, OBS - 9)]))

    def down(norad_id, epoch):
        raise SatCheckerTransportError("connection refused")

    monkeypatch.setattr(tle.satchecker, "fetch_nearest_tle", down)
    result = tle.resolve_tles([25544], OBS, remote_tle_max_age_days=3)

    with pytest.raises(tle.TLEError) as caught:
        tle.require_complete_coverage(result)
    text = str(caught.value)
    assert "9.000 d from the observation" in text     # the rejected candidate
    assert "could not be asked for a closer one" in text  # and why nothing better came


def test_uniform_rejection_is_recognised_as_a_wall_not_missing_satellites():
    """A service answering 4xx to everything must not be asked once per satellite.

    A 4xx is normally per-request — an unknown catalogue ID legitimately 404s —
    so this can only be inferred from a run of identical statuses with no success
    in between, not from the first one.
    """
    attempted = []
    lock = threading.Lock()

    def walled(norad_id, epoch):
        with lock:
            attempted.append(norad_id)
        raise client.SatCheckerResponseError("blocked", status=403)

    result = fetch_nearest_batch(
        list(range(1, 201)), OBS, fetch_nearest_tle=walled, max_workers=5,
        log=lambda _m: None,
    )
    assert len(attempted) <= service.RESPONSE_WALL_THRESHOLD + 5
    assert set(result.errors) == set(range(1, 201))


def test_a_few_missing_satellites_do_not_trip_the_wall_detector():
    """Genuinely absent catalogue entries must not abort the healthy remainder."""
    missing = {3, 7, 11}

    def fetch(norad_id, epoch):
        if norad_id in missing:
            raise client.SatCheckerResponseError("no such object", status=404)
        return make_catalogue_df([(norad_id, epoch)])

    result = fetch_nearest_batch(
        list(range(1, 41)), OBS, fetch_nearest_tle=fetch, max_workers=5,
        log=lambda _m: None,
    )
    assert set(result.errors) == missing
    assert result.records["NORAD_CAT_ID"].tolist() == [
        nid for nid in range(1, 41) if nid not in missing
    ]


def test_missing_extra_tle_dir_is_reported(cache_dir, monkeypatch, capsys):
    """A typo in a replay path must not silently become a different RFI model."""
    _service(monkeypatch, {25544: OBS})
    tle.resolve_tles([25544], OBS, extra_tle_dir="/nonexistent/typo-dir")
    assert "does not exist" in capsys.readouterr().out


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
