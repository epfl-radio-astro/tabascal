"""Offline acceptance, caching, and concurrency tests for nearest-only TLE resolution."""

import threading
import time

import pandas as pd
import pytest

from tabascal import tle
from tabascal.satchecker import client, SatCheckerTransportError
from tabascal.satchecker import service
from tabascal.satchecker.cache import TextOrbitCache
from tabascal.satchecker.service import fetch_nearest_batch

from .tle_helpers import (  # noqa: F401
    block_network,
    jd,
    make_catalogue_df,
    make_omm_catalogue_df,
    make_tle,
)


OBS = jd(2023, 2, 21)


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ORBIT_CACHE_DIR", str(tmp_path))
    return tmp_path


def _service(monkeypatch, epochs, omm_epochs=None):
    """Stub both nearest endpoints and record every request made.

    ``epochs`` maps NORAD ID to the epoch the TLE archive holds for it and
    ``omm_epochs`` does the same for the OMM archive; an absent entry means that
    archive has no record for that satellite. Both are stubbed even when a test
    cares about only one, because an ID the first endpoint cannot resolve is
    retried against the other — leaving the second live would reach the network.
    """
    calls = []

    def endpoint(archive, builder):
        def fetch(norad_id, epoch):
            calls.append((norad_id, epoch))
            value = archive.get(norad_id)
            return pd.DataFrame() if value is None else builder([(norad_id, value)])

        return fetch

    monkeypatch.setattr(client, "fetch_nearest_tle", endpoint(epochs, make_catalogue_df))
    monkeypatch.setattr(
        client, "fetch_nearest_omm", endpoint(omm_epochs or {}, make_omm_catalogue_df)
    )
    return calls


def _service_down(monkeypatch, error):
    """Make both endpoints fail the same way, and count the attempts."""
    attempts = []

    def down(norad_id, epoch):
        attempts.append(norad_id)
        raise error

    monkeypatch.setattr(client, "fetch_nearest_tle", down)
    monkeypatch.setattr(client, "fetch_nearest_omm", down)
    return attempts


def test_cache_miss_fetches_exact_observation_epoch_and_caches(cache_dir, monkeypatch):
    calls = _service(monkeypatch, {25544: OBS - 0.1})
    result = tle.resolve_tles([25544], OBS)
    assert calls == [(25544, OBS)]
    assert result.resolved[25544].source == "SatChecker (nearest-TLE)"
    assert len(TextOrbitCache(cache_dir).get(25544)) == 1


def test_close_cache_hit_makes_no_request(cache_dir, monkeypatch):
    TextOrbitCache(cache_dir).store(25544, make_catalogue_df([(25544, OBS - 0.5)]))
    calls = _service(monkeypatch, {25544: OBS})
    result = tle.resolve_tles([25544], OBS, cache_reuse_max_age_days=1)
    assert calls == []
    assert result.resolved[25544].source == "managed per-satellite cache"


def test_old_acceptable_cache_is_refreshed(cache_dir, monkeypatch):
    TextOrbitCache(cache_dir).store(25544, make_catalogue_df([(25544, OBS - 2)]))
    calls = _service(monkeypatch, {25544: OBS - 0.1})
    result = tle.resolve_tles(
        [25544], OBS, cache_reuse_max_age_days=1, remote_max_age_days=3
    )
    assert calls == [(25544, OBS)]
    assert result.resolved[25544].age_days == pytest.approx(0.1, abs=2e-8)
    assert len(TextOrbitCache(cache_dir).get(25544)) == 2


def test_old_acceptable_cache_is_offline_fallback(cache_dir, monkeypatch):
    TextOrbitCache(cache_dir).store(25544, make_catalogue_df([(25544, OBS - 2)]))

    _service_down(monkeypatch, SatCheckerTransportError("offline"))
    result = tle.resolve_tles(
        [25544], OBS, cache_reuse_max_age_days=1, remote_max_age_days=3
    )
    assert result.complete
    assert result.resolved[25544].source == "managed per-satellite cache"


def test_record_beyond_hard_ceiling_is_not_recovered(cache_dir, monkeypatch):
    TextOrbitCache(cache_dir).store(25544, make_catalogue_df([(25544, OBS - 4)]))
    _service(monkeypatch, {25544: OBS - 4})
    result = tle.resolve_tles(
        [25544], OBS, cache_reuse_max_age_days=1, remote_max_age_days=3
    )
    assert not result.complete
    with pytest.raises(tle.TLEError, match="4.000 d"):
        tle.require_complete_coverage(result)


def test_null_reuse_age_always_reuses_acceptable_cache(cache_dir, monkeypatch):
    TextOrbitCache(cache_dir).store(25544, make_catalogue_df([(25544, OBS - 2)]))
    calls = _service(monkeypatch, {25544: OBS})
    assert tle.resolve_tles(
        [25544], OBS, cache_reuse_max_age_days=None, remote_max_age_days=3
    ).complete
    assert calls == []


def test_null_reuse_age_still_fetches_when_cache_is_over_age(cache_dir, monkeypatch):
    """`null` reuse must not suppress the request for an *unacceptable* record.

    Reuse is unlimited, so the 10-day-old cached record is "near enough to reuse"
    — but the hard ceiling rejects it. Suppressing the request on the strength of
    a record that is then thrown away would fail the run with a fresh TLE sitting
    on the service.
    """
    TextOrbitCache(cache_dir).store(25544, make_catalogue_df([(25544, OBS - 10)]))
    calls = _service(monkeypatch, {25544: OBS - 0.1})
    result = tle.resolve_tles(
        [25544], OBS, cache_reuse_max_age_days=None, remote_max_age_days=3
    )
    assert calls == [(25544, OBS)]
    assert result.complete
    assert result.resolved[25544].source == "SatChecker (nearest-TLE)"


def test_staler_service_response_does_not_displace_fresher_cache(cache_dir, monkeypatch):
    """The strictly-fresher rule must govern the cache-vs-service comparison.

    The cached record is too old to suppress the request but closer to the
    observation than what the service returns, so it has to survive the refresh.
    """
    TextOrbitCache(cache_dir).store(25544, make_catalogue_df([(25544, OBS - 2)]))
    calls = _service(monkeypatch, {25544: OBS - 2.5})
    result = tle.resolve_tles(
        [25544], OBS, cache_reuse_max_age_days=1, remote_max_age_days=3
    )
    assert calls == [(25544, OBS)]
    assert result.resolved[25544].source == "managed per-satellite cache"
    assert result.resolved[25544].age_days == pytest.approx(2.0, abs=2e-8)


def test_extra_directory_has_precedence(cache_dir, tmp_path, monkeypatch):
    extra = tmp_path / "extra"
    extra.mkdir()
    make_catalogue_df([(25544, OBS - 0.2)]).to_json(extra / "manual.json")
    calls = _service(monkeypatch, {25544: OBS})
    result = tle.resolve_tles([25544], OBS, extra_orbit_dir=str(extra))
    assert result.resolved[25544].source == "extra_orbit_dir"
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
        "SatChecker (nearest-TLE)", OBS, 3.0, resolved, rejected,
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

    result = fetch_nearest_batch([5, 4, 3, 2, 1], OBS, fetch_nearest=fetch, max_workers=2)
    assert maximum == 2
    assert result.records["NORAD_CAT_ID"].tolist() == [5, 4, 3, 2, 1]


def test_batch_collects_one_failure_without_losing_other_ids():
    def fetch(norad_id, epoch):
        if norad_id == 2:
            raise SatCheckerTransportError("offline")
        return make_catalogue_df([(norad_id, epoch)])

    result = fetch_nearest_batch([1, 2, 3], OBS, fetch_nearest=fetch)
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
        list(range(1, 201)), OBS, fetch_nearest=dead_service,
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
        fetch_nearest=lambda nid, epoch: make_catalogue_df([(nid, epoch)]),
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
        list(range(1, 201)), OBS, fetch_nearest=limited, max_workers=3,
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
    _service_down(
        monkeypatch,
        client.SatCheckerRateLimitError(
            "HTTP 429 — it is rate-limiting this client", retry_after=90.0
        ),
    )
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
    TextOrbitCache(cache_dir).store(25544, make_catalogue_df([(25544, OBS - 9)]))

    _service_down(monkeypatch, SatCheckerTransportError("connection refused"))
    result = tle.resolve_tles([25544], OBS, remote_max_age_days=3)

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
        list(range(1, 201)), OBS, fetch_nearest=walled, max_workers=5,
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
        list(range(1, 41)), OBS, fetch_nearest=fetch, max_workers=5,
        log=lambda _m: None,
    )
    assert set(result.errors) == missing
    assert result.records["NORAD_CAT_ID"].tolist() == [
        nid for nid in range(1, 41) if nid not in missing
    ]


def test_missing_extra_tle_dir_is_reported(cache_dir, monkeypatch, capsys):
    """A typo in a replay path must not silently become a different RFI model."""
    _service(monkeypatch, {25544: OBS})
    tle.resolve_tles([25544], OBS, extra_orbit_dir="/nonexistent/typo-dir")
    assert "does not exist" in capsys.readouterr().out


def test_batch_response_errors_do_not_abandon_the_rest():
    """A malformed reply is one satellite's problem: the service is still up."""
    def fetch(norad_id, epoch):
        if norad_id % 2:
            raise client.SatCheckerResponseError("malformed")
        return make_catalogue_df([(norad_id, epoch)])

    result = fetch_nearest_batch(
        list(range(1, 11)), OBS, fetch_nearest=fetch, max_workers=2
    )
    assert result.records["NORAD_CAT_ID"].tolist() == [2, 4, 6, 8, 10]
    assert set(result.errors) == {1, 3, 5, 7, 9}


# ---------------------------------------------------------------------------
# Endpoint selection and boundary failover
# ---------------------------------------------------------------------------

BEFORE_HANDOVER = jd(2026, 6, 1)
AFTER_HANDOVER = jd(2026, 8, 1)
#: The last TLE SatChecker ever published, and the first OMM, twelve hours later.
LAST_TLE = jd(2026, 7, 11, 7, 33)
FIRST_OMM = jd(2026, 7, 11, 19, 56)


class TestEndpointSelection:
    """Which archive to ask, given when the observation was.

    The two archives do not overlap: TLEs stop at 2026-07-11 and OMM starts
    twelve hours later. So the observation epoch alone decides which endpoint is
    worth asking first, and asking the wrong one first costs a request rather
    than a wrong answer.
    """

    def _labels(self, epoch_jd):
        return [label for label, _ in service.nearest_endpoints_for(epoch_jd)]

    def test_a_pre_handover_epoch_asks_the_tle_archive_first(self):
        assert self._labels(BEFORE_HANDOVER)[0] == "nearest-TLE"

    def test_a_post_handover_epoch_asks_the_omm_archive_first(self):
        assert self._labels(AFTER_HANDOVER)[0] == "nearest-OMM"

    def test_the_boundary_itself_belongs_to_the_omm_archive(self):
        assert self._labels(client.HANDOVER_JD)[0] == "nearest-OMM"
        assert self._labels(client.HANDOVER_JD - 1e-6)[0] == "nearest-TLE"

    def test_both_endpoints_are_always_offered(self):
        # Neither endpoint reports "nothing that near", so the other is always
        # worth a try before concluding a satellite cannot be resolved.
        for epoch in (BEFORE_HANDOVER, AFTER_HANDOVER):
            assert len(self._labels(epoch)) == 2
            assert set(self._labels(epoch)) == {"nearest-TLE", "nearest-OMM"}

    def test_the_handover_matches_satcheckers_changelog(self):
        from tabascal.time import jd_to_datetime

        assert jd_to_datetime(client.HANDOVER_JD).date().isoformat() == "2026-07-12"


class TestBoundaryFailover:

    def test_a_post_handover_run_resolves_from_omm(self, cache_dir, monkeypatch):
        calls = _service(
            monkeypatch, {}, omm_epochs={25544: AFTER_HANDOVER - 0.1}
        )
        result = tle.resolve_tles([25544], AFTER_HANDOVER)
        assert result.complete
        assert result.resolved[25544].source == "SatChecker (nearest-OMM)"
        assert calls == [(25544, AFTER_HANDOVER)]  # one request, right archive

    def test_a_pre_handover_run_resolves_from_tle(self, cache_dir, monkeypatch):
        calls = _service(monkeypatch, {25544: BEFORE_HANDOVER - 0.1})
        result = tle.resolve_tles([25544], BEFORE_HANDOVER)
        assert result.complete
        assert result.resolved[25544].source == "SatChecker (nearest-TLE)"
        assert calls == [(25544, BEFORE_HANDOVER)]

    def test_an_empty_primary_response_falls_over_to_the_other_archive(
        self, cache_dir, monkeypatch
    ):
        # The OMM archive has nothing for this satellite at all; the TLE archive
        # does. Without the failover the run would fail with a usable record
        # sitting on the service.
        calls = _service(
            monkeypatch, {25544: AFTER_HANDOVER - 0.1}, omm_epochs={}
        )
        result = tle.resolve_tles([25544], AFTER_HANDOVER)
        assert result.complete
        assert result.resolved[25544].source == "SatChecker (nearest-TLE)"
        assert len(calls) == 2

    def test_an_over_age_primary_response_falls_over(self, cache_dir, monkeypatch):
        # The real boundary case. An observation just after the handover asks
        # OMM first; if the OMM archive has nothing near it, the last TLE — a
        # day the other side of the boundary — is the better answer.
        obs = client.HANDOVER_JD + 0.5
        calls = _service(
            monkeypatch,
            {25544: LAST_TLE},
            omm_epochs={25544: obs + 30},  # far outside the ceiling
        )
        result = tle.resolve_tles([25544], obs, remote_max_age_days=3)
        assert result.complete
        assert result.resolved[25544].source == "SatChecker (nearest-TLE)"
        assert len(calls) == 2

    def test_the_failover_works_in_the_other_direction_too(
        self, cache_dir, monkeypatch
    ):
        # The backfill case. SatChecker now sources OMM from Space-Track as well
        # as Celestrak, and Space-Track's OMM history runs years deep, so OMM
        # may appear for pre-handover epochs. A hardcoded cutoff would keep
        # preferring a stale TLE; the failover picks the record up instead.
        obs = jd(2026, 3, 1)
        calls = _service(
            monkeypatch,
            {25544: obs - 40},          # TLE archive has only something stale
            omm_epochs={25544: obs - 0.1},   # backfilled OMM is right there
        )
        result = tle.resolve_tles([25544], obs, remote_max_age_days=3)
        assert result.complete
        assert result.resolved[25544].source == "SatChecker (nearest-OMM)"
        assert len(calls) == 2

    def test_a_clamped_pre_handover_omm_response_is_rejected_with_its_true_offset(
        self, cache_dir, monkeypatch
    ):
        # get-nearest-omm answers a 2021 request with its earliest 2026 record.
        # Confirmed live: the response looks entirely healthy. The age ceiling
        # is the only thing between it and a 4.6-year-stale trajectory.
        archival = jd(2021, 11, 1)
        _service(monkeypatch, {}, omm_epochs={25544: FIRST_OMM})
        result = tle.resolve_tles([25544], archival, remote_max_age_days=3)
        assert not result.complete
        rejected = result.rejected[25544]
        assert rejected.age_days == pytest.approx(FIRST_OMM - archival, abs=1e-3)
        with pytest.raises(tle.TLEError, match="best candidate is 17"):
            tle.require_complete_coverage(result)

    def test_no_failover_when_the_primary_already_resolved_everything(
        self, cache_dir, monkeypatch
    ):
        calls = _service(
            monkeypatch,
            {25544: AFTER_HANDOVER},
            omm_epochs={25544: AFTER_HANDOVER - 0.1},
        )
        tle.resolve_tles([25544], AFTER_HANDOVER)
        assert len(calls) == 1

    def test_only_the_unresolved_ids_are_retried(self, cache_dir, monkeypatch):
        calls = _service(
            monkeypatch,
            {25544: AFTER_HANDOVER - 0.1, 43013: AFTER_HANDOVER - 0.1},
            omm_epochs={43013: AFTER_HANDOVER - 0.1},
        )
        result = tle.resolve_tles([25544, 43013], AFTER_HANDOVER)
        assert result.complete
        # Both asked once on OMM; only the unresolved one asked again on TLE.
        assert calls == [
            (25544, AFTER_HANDOVER),
            (43013, AFTER_HANDOVER),
            (25544, AFTER_HANDOVER),
        ]

    def test_a_failover_resolution_clears_the_first_passs_service_error(
        self, cache_dir, monkeypatch
    ):
        def omm_rejects(norad_id, epoch):
            raise client.SatCheckerResponseError("no such record", status=404)

        monkeypatch.setattr(client, "fetch_nearest_omm", omm_rejects)
        monkeypatch.setattr(
            client,
            "fetch_nearest_tle",
            lambda nid, epoch: make_catalogue_df([(nid, AFTER_HANDOVER - 0.1)]),
        )
        result = tle.resolve_tles([25544], AFTER_HANDOVER)
        assert result.complete
        assert result.service_errors == {}


class TestFailoverIsNotForOutages:
    """A service that cannot serve us is not asked a different question.

    This is the distinction the whole failover rests on. A response-level miss
    means the service answered and this archive has nothing usable — so the
    other archive is worth a request. A transport failure, a 429, or a uniform
    wall of rejections means the service itself is unavailable, and a second
    round of requests would add load to something already failing while
    learning nothing.
    """

    def test_a_transport_failure_does_not_trigger_the_failover(
        self, cache_dir, monkeypatch
    ):
        attempts = _service_down(monkeypatch, SatCheckerTransportError("offline"))
        result = tle.resolve_tles([25544], AFTER_HANDOVER)
        assert not result.complete
        assert attempts == [25544]  # asked once, not once per archive

    def test_a_rate_limit_does_not_trigger_the_failover(self, cache_dir, monkeypatch):
        attempts = _service_down(
            monkeypatch,
            client.SatCheckerRateLimitError(
                "SatChecker returned HTTP 429 — it is rate-limiting this client",
                retry_after=60.0,
            ),
        )
        result = tle.resolve_tles([25544], AFTER_HANDOVER)
        assert attempts == [25544]
        with pytest.raises(tle.TLEError, match="rate-limiting"):
            tle.require_complete_coverage(result)

    def test_an_outage_mid_batch_abandons_both_archives(self, cache_dir, monkeypatch):
        attempts = _service_down(monkeypatch, SatCheckerTransportError("offline"))
        ids = list(range(1, 51))
        tle.resolve_tles(ids, AFTER_HANDOVER, max_workers=2)
        # Bounded by max_workers on the first archive, and never retried on the
        # second: 50 IDs must not become 100 requests to a service that is down.
        assert len(attempts) <= 2

    def test_a_uniform_wall_does_not_trigger_the_failover(self, cache_dir, monkeypatch):
        attempts = _service_down(
            monkeypatch, client.SatCheckerResponseError("forbidden", status=403)
        )
        ids = list(range(1, 61))
        tle.resolve_tles(ids, AFTER_HANDOVER, max_workers=2)
        # The wall detector converts a run of identical 4xx into an outage, which
        # then stops the second archive being asked at all.
        assert len(attempts) < 2 * len(ids)
        assert len(attempts) <= service.RESPONSE_WALL_THRESHOLD + 2

    def test_a_per_id_response_failure_still_allows_the_failover(
        self, cache_dir, monkeypatch
    ):
        # One 404 is per-request, not a wall: the service is up and says it has
        # no OMM for this satellite. The TLE archive is worth asking.
        def omm_404(norad_id, epoch):
            raise client.SatCheckerResponseError("not found", status=404)

        monkeypatch.setattr(client, "fetch_nearest_omm", omm_404)
        monkeypatch.setattr(
            client,
            "fetch_nearest_tle",
            lambda nid, epoch: make_catalogue_df([(nid, AFTER_HANDOVER - 0.1)]),
        )
        result = tle.resolve_tles([25544], AFTER_HANDOVER)
        assert result.complete
        assert result.resolved[25544].source == "SatChecker (nearest-TLE)"
