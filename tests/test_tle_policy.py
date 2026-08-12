"""Offline tests for the TLE acceptance policies added for PR #92.

Three separate policies live here and are deliberately tested apart:

* ``remote_tle_max_age_days`` — the provisional hard ceiling on how old a
  SatChecker (or managed-cache) record may be. Explicit local files are exempt.
* complete coverage — every configured NORAD ID must resolve, or the run stops
  before subtraction with an actionable error.
* ``tle_catalogue_settle_days`` / ``tle_provisional_cache_hours`` — whether a
  downloaded catalogue may become an immutable snapshot or only a short-lived
  provisional one.

Plus the environmental-failure and multi-process behaviour that keeps a
successful fetch from being lost. Configuration normalisation itself is tested in
``test_tle_config.py``, alongside the module it belongs to.

Everything is offline: the SatChecker transport is monkeypatched, the managed
cache is a temp directory, and the wall clock is injected.
"""

import json

import pandas as pd
import pytest

from tabascal import distributed, tle
from tabascal.satchecker import cache as cache_mod
from tabascal.satchecker import SatCheckerResponseError, SatCheckerTransportError
from tabascal.satchecker.client import CatalogueResult
from tabascal.tle_config import TLEConfig

from .tle_helpers import (
    block_network,  # noqa: F401  autouse fixture: no live SatChecker access
    jd,
    make_catalogue_df,
    write_legacy_tle_file,
)

TLEError = tle.TLEError

_OBS = jd(2023, 2, 21, 12, 30)
_SETTLED_NOW = jd(2023, 6, 1)      # 100 days after the observation: settled
_UNSETTLED_NOW = jd(2023, 3, 1)    # 8 days after: still inside the 45-day window


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    d = tmp_path / "managed"
    monkeypatch.setenv("TLE_CACHE_DIR", str(d))
    return d


def _serve_catalogue(monkeypatch, pairs, counter=None):
    """Install a bulk catalogue containing exactly *pairs*."""
    def fake(epoch_jd):
        if counter is not None:
            counter["full"] = counter.get("full", 0) + 1
        df = make_catalogue_df(pairs)
        return CatalogueResult(df, len(df), len(df), "1.6.0", "zip")

    monkeypatch.setattr(tle.satchecker, "fetch_full_catalogue", fake)


def _serve_empty_catalogue(monkeypatch, counter=None):
    from tabascal.satchecker import EmptyCatalogueError

    def fake(epoch_jd):
        if counter is not None:
            counter["full"] = counter.get("full", 0) + 1
        raise EmptyCatalogueError("no records at this epoch")

    monkeypatch.setattr(tle.satchecker, "fetch_full_catalogue", fake)


def _serve_nearest(monkeypatch, pairs, counter=None):
    lookup = dict(pairs)

    def fake(norad_id, epoch_jd):
        if counter is not None:
            counter["near"] = counter.get("near", 0) + 1
        if norad_id in lookup:
            return make_catalogue_df([(norad_id, lookup[norad_id])])
        return pd.DataFrame()

    monkeypatch.setattr(tle.satchecker, "fetch_nearest_tle", fake)
    monkeypatch.setattr(tle.time, "sleep", lambda seconds: None)


def _no_network(monkeypatch):
    """Any provider call from here on is a test failure."""
    def boom(*args, **kwargs):
        raise AssertionError("no provider request may be made")

    monkeypatch.setattr(tle.satchecker, "fetch_full_catalogue", boom)
    monkeypatch.setattr(tle.satchecker, "fetch_nearest_tle", boom)


def _resolve(norad_ids, now=_SETTLED_NOW, **kwargs):
    return tle.resolve_tles(norad_ids, _OBS, clock=lambda: now, **kwargs)


# ---------------------------------------------------------------------------
# Remote age ceiling
# ---------------------------------------------------------------------------

class TestRemoteAgePolicy:

    def test_fresh_bulk_record_accepted(self, cache_dir, monkeypatch):
        _serve_catalogue(monkeypatch, [(25544, jd(2023, 2, 21, 13))])
        entry = _resolve([25544]).resolved[25544]
        assert entry.source == "managed catalogue"
        assert entry.age_days < 0.03

    def test_fresh_per_id_record_accepted(self, cache_dir, monkeypatch):
        _serve_empty_catalogue(monkeypatch)
        _serve_nearest(monkeypatch, [(25544, jd(2023, 2, 21, 13))])
        entry = _resolve([25544]).resolved[25544]
        assert entry.source == "SatChecker per-satellite"
        assert entry.age_days < 0.03

    def test_fresh_managed_cache_record_accepted(self, cache_dir, monkeypatch):
        _serve_catalogue(monkeypatch, [(25544, jd(2023, 2, 21, 13))])
        _resolve([25544])
        _no_network(monkeypatch)  # second pass must come from the stored snapshot
        entry = _resolve([25544]).resolved[25544]
        assert entry.source == "managed catalogue"
        assert entry.age_days < 0.03

    def test_record_exactly_at_the_threshold_is_accepted(self, cache_dir, monkeypatch):
        _serve_catalogue(monkeypatch, [(25544, jd(2023, 2, 20, 12, 30))])  # exactly 1 d
        resolution = _resolve(
            [25544], remote_tle_max_age_days=1.0, remote_tle_target_age_days=None
        )
        assert resolution.complete
        assert resolution.resolved[25544].age_days == pytest.approx(1.0, abs=1e-6)

    def test_older_record_beyond_the_threshold_is_rejected(self, cache_dir, monkeypatch):
        _serve_catalogue(monkeypatch, [(25544, jd(2023, 1, 21, 12, 30))])  # 31 d old
        _serve_nearest(monkeypatch, [])
        resolution = _resolve([25544])
        assert resolution.missing == [25544]
        rejected = resolution.rejected[25544]
        assert rejected.age_days == pytest.approx(31.0, abs=1e-3)
        assert rejected.offset_days < 0  # the TLE predates the observation
        assert "remote_tle_max_age_days" in rejected.reason

    def test_future_record_beyond_the_threshold_is_rejected(self, cache_dir, monkeypatch):
        # A TLE from well *after* the observation is just as unusable as an old one.
        _serve_catalogue(monkeypatch, [(25544, jd(2023, 3, 21, 12, 30))])
        _serve_nearest(monkeypatch, [])
        resolution = _resolve([25544])
        assert resolution.missing == [25544]
        assert resolution.rejected[25544].offset_days > 0

    def test_null_ceiling_is_an_explicit_expert_override(self, cache_dir, monkeypatch):
        _serve_catalogue(monkeypatch, [(25544, jd(2023, 1, 21, 12, 30))])  # 31 d old
        resolution = _resolve(
            [25544], remote_tle_max_age_days=None, remote_tle_target_age_days=None
        )
        assert resolution.complete
        assert resolution.resolved[25544].age_days == pytest.approx(31.0, abs=1e-3)
        assert resolution.remote_max_age_days is None

    def test_local_replay_is_unaffected_by_the_remote_ceiling(
        self, cache_dir, tmp_path, monkeypatch
    ):
        # The exact-replay workflow: a saved used_tles file from a year-old run
        # must still resolve with the ceiling at its default.
        extra = tmp_path / "run_tles"
        extra.mkdir()
        old_epoch = jd(2022, 11, 13, 12, 30)  # ~100 days before the observation
        write_legacy_tle_file(extra / "used_tles_Custom.json", [(25544, old_epoch)])
        _no_network(monkeypatch)

        resolution = _resolve([25544], extra_tle_dir=str(extra))

        assert resolution.complete
        assert resolution.resolved[25544].source == "extra_tle_dir"
        assert resolution.resolved[25544].age_days > 90

    def test_rejected_record_is_never_re_admitted_after_exhaustion(
        self, cache_dir, monkeypatch
    ):
        # The stale bulk record is the *only* thing the service has. Once the
        # per-ID endpoint also comes up short, the run must fail rather than fall
        # back to the record the age policy already refused.
        _serve_catalogue(monkeypatch, [(25544, jd(2023, 1, 21, 12, 30))])
        _serve_nearest(monkeypatch, [])
        with pytest.raises(TLEError, match="rejected by remote_tle_max_age_days"):
            tle.get_tles_by_id([25544], _OBS, clock=lambda: _SETTLED_NOW)

    def test_every_source_reports_the_same_epoch_and_age_fields(
        self, cache_dir, tmp_path, monkeypatch
    ):
        extra = tmp_path / "extra"
        extra.mkdir()
        write_legacy_tle_file(extra / "local.json", [(111, jd(2023, 2, 21, 11))])
        _serve_catalogue(monkeypatch, [(222, jd(2023, 2, 21, 13))])
        _serve_nearest(monkeypatch, [(333, jd(2023, 2, 21, 14))])

        resolution = _resolve([111, 222, 333], extra_tle_dir=str(extra))

        assert resolution.complete
        sources = {nid: e.source for nid, e in resolution.resolved.items()}
        assert sources == {
            111: "extra_tle_dir",
            222: "managed catalogue",
            333: "SatChecker per-satellite",
        }
        for entry in resolution.resolved.values():
            assert entry.epoch_jd > 2_400_000
            assert entry.age_days == abs(entry.offset_days)
            assert entry.age_days < 1.0


class TestRemoteAgeLogging:

    def test_small_sets_report_every_record(self, cache_dir, monkeypatch, capsys):
        _serve_catalogue(monkeypatch, [(25544, jd(2023, 2, 21, 13))])
        _resolve([25544])
        out = capsys.readouterr().out
        assert "25544: managed catalogue [test]" in out
        assert "offset +" in out and "age " in out

    def test_large_sets_are_grouped_with_the_oldest_named(
        self, cache_dir, monkeypatch, capsys
    ):
        pairs = [(1000 + i, jd(2023, 2, 21, 13) - i * 0.1) for i in range(30)]
        _serve_catalogue(monkeypatch, pairs)
        _resolve([nid for nid, _ in pairs], remote_tle_target_age_days=None)
        out = capsys.readouterr().out
        assert "30 accepted (limit 3 d)" in out
        assert "age vs observation" in out
        assert "oldest" in out
        assert "1029" in out                       # the oldest record is named
        assert "1005: managed catalogue" not in out  # but not every one of them
        assert "TABASCAL_TLE_LOG_DETAIL=1" in out

    def test_detail_env_forces_the_full_listing(self, cache_dir, monkeypatch, capsys):
        monkeypatch.setenv("TABASCAL_TLE_LOG_DETAIL", "1")
        pairs = [(1000 + i, jd(2023, 2, 21, 13) - i * 0.1) for i in range(30)]
        _serve_catalogue(monkeypatch, pairs)
        _resolve([nid for nid, _ in pairs], remote_tle_target_age_days=None)
        out = capsys.readouterr().out
        assert "1005: managed catalogue" in out
        assert "1029: managed catalogue" in out


# ---------------------------------------------------------------------------
# Complete coverage
# ---------------------------------------------------------------------------

class TestCompleteCoverage:

    def test_mixed_outcomes_produce_one_error_naming_every_failure(
        self, cache_dir, monkeypatch
    ):
        _serve_catalogue(
            monkeypatch,
            [
                (111, jd(2023, 2, 21, 13)),          # fresh
                (222, jd(2023, 1, 21, 12, 30)),      # 31 d: too old
                (333, jd(2023, 1, 1, 12, 30)),       # 51 d: too old
            ],
        )
        _serve_nearest(monkeypatch, [])

        with pytest.raises(TLEError) as exc_info:
            tle.get_tles_by_id([111, 222, 333, 444], _OBS, clock=lambda: _SETTLED_NOW)

        message = str(exc_info.value)
        assert "3 of 4 configured satellites" in message
        assert "222" in message and "333" in message and "444" in message
        assert "no record found" in message       # 444 was never served at all
        assert "31." in message and "51." in message  # how close the best ones were

    def test_one_stale_id_in_a_large_list_stops_the_run(self, cache_dir, monkeypatch):
        pairs = [(1000 + i, jd(2023, 2, 21, 13)) for i in range(40)]
        pairs[17] = (1017, jd(2023, 1, 1, 12, 30))  # one stale record
        _serve_catalogue(monkeypatch, pairs)
        _serve_nearest(monkeypatch, [])

        with pytest.raises(TLEError) as exc_info:
            tle.get_tles_by_id([nid for nid, _ in pairs], _OBS, clock=lambda: _SETTLED_NOW)
        assert "1017" in str(exc_info.value)

    def test_removing_the_offending_id_lets_the_same_run_proceed(
        self, cache_dir, monkeypatch
    ):
        pairs = [(1000 + i, jd(2023, 2, 21, 13)) for i in range(40)]
        pairs[17] = (1017, jd(2023, 1, 1, 12, 30))
        _serve_catalogue(monkeypatch, pairs)
        _serve_nearest(monkeypatch, [])

        wanted = [nid for nid, _ in pairs if nid != 1017]
        df = tle.get_tles_by_id(wanted, _OBS, clock=lambda: _SETTLED_NOW)
        assert list(df["NORAD_CAT_ID"]) == wanted

    def test_zero_accepted_ids_uses_the_same_structured_error(
        self, cache_dir, monkeypatch
    ):
        _serve_catalogue(monkeypatch, [])
        _serve_nearest(monkeypatch, [])
        with pytest.raises(TLEError) as exc_info:
            tle.get_tles_by_id([111, 222], _OBS, clock=lambda: _SETTLED_NOW)
        message = str(exc_info.value)
        assert "2 of 2 configured satellites" in message
        assert "remove these NORAD IDs" in message

    def test_requested_id_list_is_never_silently_mutated(self, cache_dir, monkeypatch):
        _serve_catalogue(monkeypatch, [(111, jd(2023, 2, 21, 13))])
        _serve_nearest(monkeypatch, [])
        resolution = _resolve([111, 222, 333])
        assert resolution.requested == [111, 222, 333]
        assert resolution.missing == [222, 333]

    def test_result_rows_follow_the_requested_order(self, cache_dir, monkeypatch):
        pairs = [(333, jd(2023, 2, 21, 13)), (111, jd(2023, 2, 21, 13)), (222, jd(2023, 2, 21, 13))]
        _serve_catalogue(monkeypatch, pairs)
        df = tle.get_tles_by_id([333, 111, 222], _OBS, clock=lambda: _SETTLED_NOW)
        assert list(df["NORAD_CAT_ID"]) == [333, 111, 222]

    def test_no_configured_ids_resolves_to_an_empty_frame(self, cache_dir, monkeypatch):
        _no_network(monkeypatch)
        assert tle.get_tles_by_id([], _OBS).empty


# ---------------------------------------------------------------------------
# Freshness upgrade pass
# ---------------------------------------------------------------------------

class TestFreshnessUpgrade:
    """SatChecker's bulk endpoint returns the newest record at or *before* the
    requested epoch, so it cannot see a closer record just after it. Records
    older than ``remote_tle_target_age_days`` are re-asked of the per-satellite
    endpoint, which has no such restriction."""

    _STALE = jd(2023, 2, 19, 12, 30)   # 2 d before the observation
    _FRESH = jd(2023, 2, 21, 14, 0)    # 1.5 h after it — the bulk cannot return this

    def test_stale_bulk_record_is_upgraded(self, cache_dir, monkeypatch):
        _serve_catalogue(monkeypatch, [(25544, self._STALE)])
        counter = {}
        _serve_nearest(monkeypatch, [(25544, self._FRESH)], counter)

        entry = _resolve([25544]).resolved[25544]

        assert counter["near"] == 1
        assert entry.source == "SatChecker per-satellite"
        assert entry.age_days < 0.1          # was 2 d from the catalogue
        assert entry.offset_days > 0         # the record the bulk endpoint cannot see

    def test_fresh_bulk_record_is_left_alone(self, cache_dir, monkeypatch):
        _serve_catalogue(monkeypatch, [(25544, jd(2023, 2, 21, 13))])  # 0.02 d
        counter = {}
        _serve_nearest(monkeypatch, [(25544, self._FRESH)], counter)

        entry = _resolve([25544]).resolved[25544]

        assert counter.get("near", 0) == 0   # no request spent
        assert entry.source == "managed catalogue"

    def test_upgrade_keeps_the_bulk_record_when_no_better(self, cache_dir, monkeypatch):
        # The per-satellite endpoint answers, but with something no fresher.
        _serve_catalogue(monkeypatch, [(25544, self._STALE)])
        _serve_nearest(monkeypatch, [(25544, jd(2023, 2, 18, 12, 30))])  # 3 d, worse

        entry = _resolve([25544]).resolved[25544]

        assert entry.source == "managed catalogue"
        assert entry.age_days == pytest.approx(2.0, abs=1e-3)

    def test_failed_upgrade_never_loses_the_satellite(self, cache_dir, monkeypatch):
        # A declined upgrade must not turn an otherwise complete run into a
        # coverage failure.
        _serve_catalogue(monkeypatch, [(25544, self._STALE)])
        _serve_nearest(monkeypatch, [])       # endpoint has nothing

        resolution = _resolve([25544])

        assert resolution.complete
        assert resolution.resolved[25544].source == "managed catalogue"

    def test_upgrade_never_admits_a_record_beyond_the_ceiling(
        self, cache_dir, monkeypatch
    ):
        _serve_catalogue(monkeypatch, [(25544, self._STALE)])
        _serve_nearest(monkeypatch, [(25544, jd(2023, 1, 1, 12, 30))])  # 51 d

        resolution = _resolve([25544])

        assert resolution.complete
        assert resolution.resolved[25544].age_days == pytest.approx(2.0, abs=1e-3)

    def test_upgrades_are_cached_and_reused_offline(self, cache_dir, monkeypatch):
        _serve_catalogue(monkeypatch, [(25544, self._STALE)])
        counter = {}
        _serve_nearest(monkeypatch, [(25544, self._FRESH)], counter)
        _resolve([25544])
        assert counter["near"] == 1

        _no_network(monkeypatch)
        entry = _resolve([25544]).resolved[25544]
        assert entry.source == "cached per-satellite record"
        assert entry.age_days < 0.1

    def test_null_target_disables_the_pass(self, cache_dir, monkeypatch):
        _serve_catalogue(monkeypatch, [(25544, self._STALE)])
        counter = {}
        _serve_nearest(monkeypatch, [(25544, self._FRESH)], counter)

        entry = _resolve([25544], remote_tle_target_age_days=None).resolved[25544]

        assert counter.get("near", 0) == 0
        assert entry.source == "managed catalogue"

    def test_extra_dir_records_are_never_upgraded(
        self, cache_dir, tmp_path, monkeypatch
    ):
        # Your own files are outside remote policy entirely, however old.
        extra = tmp_path / "extra"
        extra.mkdir()
        write_legacy_tle_file(extra / "local.json", [(25544, jd(2022, 11, 13, 12, 30))])
        _no_network(monkeypatch)

        entry = _resolve([25544], extra_tle_dir=str(extra)).resolved[25544]

        assert entry.source == "extra_tle_dir"
        assert entry.age_days > 90

    def test_missing_and_stale_share_one_fetch_pass(self, cache_dir, monkeypatch, capsys):
        _serve_catalogue(monkeypatch, [(111, self._STALE)])      # present but stale
        counter = {}
        _serve_nearest(monkeypatch, [(111, self._FRESH), (222, self._FRESH)], counter)

        resolution = _resolve([111, 222])                        # 222 absent from bulk

        assert counter["near"] == 2
        assert resolution.complete
        out = capsys.readouterr().out
        assert "1 missing" in out and "1 older than 1 d" in out

    def test_endpoint_outage_during_upgrades_does_not_fail_the_run(
        self, cache_dir, monkeypatch, capsys
    ):
        # Every satellite already has an acceptable catalogue record; the
        # per-satellite calls are pure optional upgrades. An outage there must not
        # convert a fully resolved set into a failed run.
        _serve_catalogue(monkeypatch, [(111, self._STALE), (222, self._STALE)])

        def down(norad_id, epoch_jd):
            raise SatCheckerTransportError("connection refused")

        monkeypatch.setattr(tle.satchecker, "fetch_nearest_tle", down)
        monkeypatch.setattr(tle.time, "sleep", lambda seconds: None)

        resolution = _resolve([111, 222])

        assert resolution.complete
        assert all(e.source == "managed catalogue" for e in resolution.resolved.values())
        assert "per-satellite endpoint unreachable" in capsys.readouterr().out

    def test_endpoint_outage_still_fails_when_a_satellite_is_missing(
        self, cache_dir, monkeypatch
    ):
        # Mixed batch: 222 is absent from the catalogue, so the per-satellite call
        # is load-bearing rather than optional and the outage is fatal.
        _serve_catalogue(monkeypatch, [(111, self._STALE)])

        def down(norad_id, epoch_jd):
            raise SatCheckerTransportError("connection refused")

        monkeypatch.setattr(tle.satchecker, "fetch_nearest_tle", down)
        monkeypatch.setattr(tle.time, "sleep", lambda seconds: None)

        with pytest.raises(SatCheckerTransportError):
            _resolve([111, 222])

    def test_upgrades_obtained_before_an_outage_are_kept(self, cache_dir, monkeypatch):
        _serve_catalogue(
            monkeypatch, [(111, self._STALE), (222, self._STALE), (333, self._STALE)]
        )
        seen = []

        def flaky(norad_id, epoch_jd):
            seen.append(norad_id)
            if norad_id == 111:
                return make_catalogue_df([(111, self._FRESH)])
            raise SatCheckerTransportError("connection refused")

        monkeypatch.setattr(tle.satchecker, "fetch_nearest_tle", flaky)
        monkeypatch.setattr(tle.time, "sleep", lambda seconds: None)

        resolution = _resolve([111, 222, 333])

        assert resolution.complete
        assert resolution.resolved[111].source == "SatChecker per-satellite"
        assert resolution.resolved[111].age_days < 0.1      # the upgrade survived
        assert resolution.resolved[222].source == "managed catalogue"
        assert seen == [111, 222]                            # loop stopped at the outage

    def test_a_satellite_asked_about_and_absent_is_not_a_transport_failure(
        self, cache_dir, monkeypatch
    ):
        # 222 is required and the endpoint answered "no record"; a later outage on
        # an optional upgrade must then surface as the coverage error naming 222,
        # not as a transport error about a different satellite.
        _serve_catalogue(monkeypatch, [(333, self._STALE)])
        order = []

        def flaky(norad_id, epoch_jd):
            order.append(norad_id)
            if norad_id == 222:
                return pd.DataFrame()          # asked; genuinely nothing there
            raise SatCheckerTransportError("connection refused")

        monkeypatch.setattr(tle.satchecker, "fetch_nearest_tle", flaky)
        monkeypatch.setattr(tle.time, "sleep", lambda seconds: None)

        resolution = _resolve([222, 333])

        assert resolution.missing == [222]
        assert resolution.resolved[333].source == "managed catalogue"

    def test_records_already_from_the_per_id_endpoint_are_not_re_requested(
        self, cache_dir, monkeypatch
    ):
        # An empty catalogue sends everything to the per-satellite endpoint; the
        # answer is already the nearest the service holds, so even if it is older
        # than the target it must not be asked for twice.
        _serve_empty_catalogue(monkeypatch)
        counter = {}
        _serve_nearest(monkeypatch, [(25544, self._STALE)], counter)

        entry = _resolve([25544]).resolved[25544]

        assert counter["near"] == 1
        assert entry.age_days == pytest.approx(2.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Catalogue settling: stable vs provisional
# ---------------------------------------------------------------------------

def _snapshot_files(cache_dir):
    return sorted(p.name for p in cache_dir.glob("catalogue-*.json"))


class TestCatalogueSettling:

    def test_settled_epoch_is_stored_and_reused_immutably(self, cache_dir, monkeypatch):
        counter = {}
        _serve_catalogue(monkeypatch, [(25544, jd(2023, 2, 21, 13))], counter)
        _resolve([25544], now=_SETTLED_NOW)
        assert counter["full"] == 1
        names = _snapshot_files(cache_dir)
        assert len(names) == 1 and "provisional" not in names[0]

        # Reused forever, without a further request, even much later.
        _no_network(monkeypatch)
        assert _resolve([25544], now=_SETTLED_NOW + 3650).complete

    def test_recent_epoch_is_only_provisional(self, cache_dir, monkeypatch):
        _serve_catalogue(monkeypatch, [(25544, jd(2023, 2, 21, 13))])
        assert _resolve([25544], now=_UNSETTLED_NOW).complete
        names = _snapshot_files(cache_dir)
        assert names == [names[0]] and names[0].endswith("-provisional.json")

    def test_future_epoch_is_only_provisional(self, cache_dir, monkeypatch):
        _serve_catalogue(monkeypatch, [(25544, jd(2023, 2, 21, 13))])
        # "Now" precedes the observation: nothing about that catalogue is settled.
        assert _resolve([25544], now=_OBS - 1).complete
        assert all(n.endswith("-provisional.json") for n in _snapshot_files(cache_dir))

    def test_provisional_result_is_reused_before_expiry(self, cache_dir, monkeypatch):
        counter = {}
        _serve_catalogue(monkeypatch, [(25544, jd(2023, 2, 21, 13))], counter)
        _resolve([25544], now=_UNSETTLED_NOW)
        _resolve([25544], now=_UNSETTLED_NOW + 6 / 24)  # 6 h later, inside 12 h
        assert counter["full"] == 1

    def test_provisional_result_is_refreshed_after_expiry(self, cache_dir, monkeypatch):
        counter = {}
        _serve_catalogue(monkeypatch, [(25544, jd(2023, 2, 21, 13))], counter)
        _resolve([25544], now=_UNSETTLED_NOW)
        _resolve([25544], now=_UNSETTLED_NOW + 13 / 24)  # 13 h later, past 12 h
        assert counter["full"] == 2

    def test_fallback_records_follow_the_snapshot_state(self, cache_dir, monkeypatch):
        # An unsettled epoch reaches the per-satellite endpoint *because* its bulk
        # catalogue is empty, so a provisional response cached under the stable
        # name would be the one result the settling policy never revisited.
        _serve_empty_catalogue(monkeypatch)
        _serve_nearest(monkeypatch, [(25544, jd(2023, 2, 21, 13))])
        _resolve([25544], now=_UNSETTLED_NOW)
        names = sorted(p.name for p in cache_dir.glob("catalogue-*-extra*.json"))
        assert names == [n for n in names if n.endswith("-extra-provisional.json")]

    def test_provisional_fallback_is_reused_before_expiry(self, cache_dir, monkeypatch):
        counter = {}
        _serve_empty_catalogue(monkeypatch)
        _serve_nearest(monkeypatch, [(25544, jd(2023, 2, 21, 13))], counter)
        _resolve([25544], now=_UNSETTLED_NOW)
        _resolve([25544], now=_UNSETTLED_NOW + 6 / 24)
        assert counter["near"] == 1

    def test_provisional_fallback_is_refetched_after_expiry(self, cache_dir, monkeypatch):
        # Without an expiry this record would be reused forever: TLE age is
        # measured against the fixed observation epoch, so nothing else about it
        # ever goes stale, even as upstream ingestion settles.
        counter = {}
        _serve_empty_catalogue(monkeypatch)
        _serve_nearest(monkeypatch, [(25544, jd(2023, 2, 21, 13))], counter)
        _resolve([25544], now=_UNSETTLED_NOW)
        _resolve([25544], now=_UNSETTLED_NOW + 13 / 24)
        assert counter["near"] == 2

    def test_settled_fallback_is_kept_indefinitely(self, cache_dir, monkeypatch):
        # A settled epoch's fallback record is immutable, like its snapshot.
        counter = {}
        _serve_empty_catalogue(monkeypatch)
        _serve_nearest(monkeypatch, [(25544, jd(2023, 2, 21, 13))], counter)
        _resolve([25544], now=_SETTLED_NOW)
        _resolve([25544], now=_SETTLED_NOW + 3650)
        assert counter["near"] == 1
        assert list(cache_dir.glob("catalogue-*-extra.json"))

    def test_settled_run_ignores_a_provisional_fallback(self, cache_dir, monkeypatch):
        counter = {}
        _serve_empty_catalogue(monkeypatch)
        _serve_nearest(monkeypatch, [(25544, jd(2023, 2, 21, 13))], counter)
        _resolve([25544], now=_UNSETTLED_NOW)
        _resolve([25544], now=_SETTLED_NOW)
        assert counter["near"] == 2  # refetched for the stable store
        assert list(cache_dir.glob("catalogue-*-extra.json"))
        assert list(cache_dir.glob("catalogue-*-extra-provisional.json"))

    def test_provisional_is_never_promoted_by_aging_in_place(
        self, cache_dir, monkeypatch
    ):
        # The provisional file written while the epoch was recent must not become
        # the trusted stable snapshot once the epoch settles.
        counter = {}
        _serve_catalogue(monkeypatch, [(25544, jd(2023, 2, 21, 13))], counter)
        _resolve([25544], now=_UNSETTLED_NOW)
        assert counter["full"] == 1

        _resolve([25544], now=_SETTLED_NOW)
        assert counter["full"] == 2  # refetched for the stable snapshot
        names = _snapshot_files(cache_dir)
        assert len(names) == 2  # the provisional file is left alone, not renamed
        assert any(not n.endswith("-provisional.json") for n in names)

    def test_a_stable_file_holding_a_provisional_envelope_is_a_miss(
        self, cache_dir, monkeypatch
    ):
        # Hand-moving a provisional document to the stable filename must not work.
        _serve_catalogue(monkeypatch, [(25544, jd(2023, 2, 21, 13))])
        _resolve([25544], now=_UNSETTLED_NOW)
        provisional = next(cache_dir.glob("*-provisional.json"))
        stable = cache_dir / provisional.name.replace("-provisional", "")
        stable.write_text(provisional.read_text())

        cache = cache_mod.TextCatalogueCache(cache_dir, clock=lambda: _SETTLED_NOW)
        canon = cache_mod.canonical_epoch_jd(_OBS)
        assert cache.get_snapshot(canon, cache_mod.STABLE) is None

    def test_previous_schema_snapshots_are_not_trusted(self, cache_dir, monkeypatch):
        # A snapshot written by the pre-settling-policy revision was stored under
        # the stable name however recent its catalogue was; the schema bump makes
        # it a miss rather than letting it become trusted by aging in place.
        cache_dir.mkdir(parents=True, exist_ok=True)
        canon = cache_mod.canonical_epoch_jd(_OBS)
        stamp = cache_mod.canonical_stamp(canon)
        records = make_catalogue_df([(25544, jd(2023, 2, 21, 13))]).to_dict("records")
        (cache_dir / f"catalogue-{stamp}.json").write_text(json.dumps({
            "schema_version": 1,
            "catalogue_epoch_jd": canon,
            "fetched_at": "2023-02-21T13:00:00Z",
            "expected_count": 1,
            "actual_count": 1,
            "records": records,
        }))
        counter = {}
        _serve_catalogue(monkeypatch, [(25544, jd(2023, 2, 21, 13))], counter)
        _resolve([25544], now=_SETTLED_NOW)
        assert counter["full"] == 1  # the v1 file was ignored and refetched

    def test_settle_days_null_treats_everything_as_stable(self, cache_dir, monkeypatch):
        _serve_catalogue(monkeypatch, [(25544, jd(2023, 2, 21, 13))])
        _resolve([25544], now=_UNSETTLED_NOW, catalogue_settle_days=None)
        assert all(not n.endswith("-provisional.json") for n in _snapshot_files(cache_dir))


# ---------------------------------------------------------------------------
# Environmental cache-write failures
# ---------------------------------------------------------------------------

def _break_cache_writes(monkeypatch, error=None):
    def boom(path, payload):
        raise error or OSError(30, "Read-only file system")

    monkeypatch.setattr(cache_mod, "_atomic_write_json", boom)


class TestCacheWriteFailures:

    def test_write_failure_preserves_the_successful_fetch(
        self, cache_dir, monkeypatch, capsys
    ):
        _serve_catalogue(monkeypatch, [(25544, jd(2023, 2, 21, 13))])
        _break_cache_writes(monkeypatch)

        df = tle.get_tles_by_id([25544], _OBS, clock=lambda: _SETTLED_NOW)

        assert list(df["NORAD_CAT_ID"]) == [25544]
        out = capsys.readouterr().out
        assert "could not write the stable catalogue snapshot" in out
        assert "without reusable cache state" in out
        assert not list(cache_dir.glob("catalogue-*.json"))

    def test_fallback_write_failure_preserves_the_records(self, cache_dir, monkeypatch):
        _serve_empty_catalogue(monkeypatch)
        _serve_nearest(monkeypatch, [(25544, jd(2023, 2, 21, 13))])
        _break_cache_writes(monkeypatch)

        df = tle.get_tles_by_id([25544], _OBS, clock=lambda: _SETTLED_NOW)
        assert list(df["NORAD_CAT_ID"]) == [25544]

    def test_unwritable_cache_directory_does_not_abort_the_run(
        self, tmp_path, monkeypatch
    ):
        # The directory itself cannot even be created.
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("")
        monkeypatch.setenv("TLE_CACHE_DIR", str(blocker / "cache"))
        _serve_catalogue(monkeypatch, [(25544, jd(2023, 2, 21, 13))])

        df = tle.get_tles_by_id([25544], _OBS, clock=lambda: _SETTLED_NOW)
        assert list(df["NORAD_CAT_ID"]) == [25544]

    def test_validation_failures_stay_loud(self, cache_dir, monkeypatch):
        # A snapshot that fails its own validation is a bug, not a disk problem,
        # and must not be swallowed as an environmental failure.
        _break_cache_writes(monkeypatch, error=ValueError("actual_count mismatch"))
        _serve_catalogue(monkeypatch, [(25544, jd(2023, 2, 21, 13))])
        with pytest.raises(ValueError, match="actual_count mismatch"):
            tle.get_tles_by_id([25544], _OBS, clock=lambda: _SETTLED_NOW)


# ---------------------------------------------------------------------------
# Multi-process coherence
# ---------------------------------------------------------------------------

class _FakeCluster:
    """A two-process cluster whose ranks are driven one after the other.

    ``broadcast_bytes_from_rank0`` is replaced by a shared buffer: rank 0 fills
    it, every other rank reads it. That is exactly the contract the real
    implementation provides, without needing a JAX distributed runtime.
    """

    def __init__(self, monkeypatch, n_processes=2):
        self.buffer = None
        self.rank = 0
        self.broadcasts = 0
        monkeypatch.setattr(distributed, "process_count", lambda: n_processes)
        monkeypatch.setattr(distributed, "is_process_0", lambda: self.rank == 0)
        monkeypatch.setattr(distributed, "broadcast_bytes_from_rank0", self._broadcast)

    def _broadcast(self, payload, name):
        self.broadcasts += 1
        if payload is not None:
            self.buffer = payload
        return self.buffer


class TestMultiProcessResolution:

    def test_workers_reuse_rank0_result_when_the_cache_cannot_be_written(
        self, cache_dir, monkeypatch
    ):
        # The failure mode this replaces: with nothing on disk for a worker to
        # read, each rank would have gone to the provider itself.
        counter = {}
        _serve_catalogue(monkeypatch, [(25544, jd(2023, 2, 21, 13))], counter)
        _break_cache_writes(monkeypatch)
        cluster = _FakeCluster(monkeypatch)

        frames = []
        for rank in (0, 1):
            cluster.rank = rank
            frames.append(tle.get_tles_by_id([25544], _OBS, clock=lambda: _SETTLED_NOW))

        assert counter["full"] == 1  # exactly one provider fetch for the whole run
        assert list(frames[0]["NORAD_CAT_ID"]) == list(frames[1]["NORAD_CAT_ID"]) == [25544]
        assert frames[0]["TLE_LINE1"].iloc[0] == frames[1]["TLE_LINE1"].iloc[0]

    def test_workers_receive_bit_identical_elements(self, cache_dir, monkeypatch):
        # Only the raw TLE lines cross the wire; each rank re-parses the elements,
        # so nothing is lost to JSON rounding.
        _serve_catalogue(monkeypatch, [(25544, jd(2023, 2, 21, 13))])
        _break_cache_writes(monkeypatch)
        cluster = _FakeCluster(monkeypatch)

        cluster.rank = 0
        rank0 = tle.get_tles_by_id([25544], _OBS, clock=lambda: _SETTLED_NOW)
        cluster.rank = 1
        worker = tle.get_tles_by_id([25544], _OBS, clock=lambda: _SETTLED_NOW)

        for column in ("EPOCH_JD", "SEMIMAJOR_AXIS", "MEAN_MOTION", "BSTAR"):
            assert rank0[column].iloc[0] == worker[column].iloc[0]

    def test_all_ranks_fail_coherently(self, cache_dir, monkeypatch):
        _serve_catalogue(monkeypatch, [])
        _serve_nearest(monkeypatch, [])
        cluster = _FakeCluster(monkeypatch)

        errors = []
        for rank in (0, 1):
            cluster.rank = rank
            with pytest.raises(TLEError) as exc_info:
                tle.get_tles_by_id([25544], _OBS, clock=lambda: _SETTLED_NOW)
            errors.append(str(exc_info.value))

        assert "25544" in errors[0]
        assert "process 0" in errors[1] and "25544" in errors[1]

    def test_workers_never_call_the_provider(self, cache_dir, monkeypatch):
        _serve_catalogue(monkeypatch, [(25544, jd(2023, 2, 21, 13))])
        cluster = _FakeCluster(monkeypatch)
        cluster.rank = 0
        tle.get_tles_by_id([25544], _OBS, clock=lambda: _SETTLED_NOW)

        _no_network(monkeypatch)
        cluster.rank = 1
        df = tle.get_tles_by_id([25544], _OBS, clock=lambda: _SETTLED_NOW)
        assert list(df["NORAD_CAT_ID"]) == [25544]

    def test_single_process_does_not_broadcast(self, cache_dir, monkeypatch):
        _serve_catalogue(monkeypatch, [(25544, jd(2023, 2, 21, 13))])
        cluster = _FakeCluster(monkeypatch, n_processes=1)
        tle.get_tles_by_id([25544], _OBS, clock=lambda: _SETTLED_NOW)
        assert cluster.broadcasts == 0

    def test_preflight_is_shared_not_repeated_per_rank(self, cache_dir, monkeypatch):
        # Every rank builds its own TabConfig and so reaches preflight. Without
        # sharing, each would download the catalogue and reach its own coverage
        # verdict — one provider fetch per rank, and divergent verdicts leaving
        # some ranks exiting while others enter a JAX collective and hang.
        counter = {}
        _serve_catalogue(monkeypatch, [(25544, jd(2023, 2, 21, 13))], counter)
        _break_cache_writes(monkeypatch)
        monkeypatch.setattr(tle, "_ms_mean_epoch_jd", lambda ms: _OBS)
        cluster = _FakeCluster(monkeypatch)
        config = TLEConfig(norad_ids=[25544])

        resolutions = []
        for rank in (0, 1):
            cluster.rank = rank
            resolutions.append(
                tle.preflight_tle_check(config, "ignored.ms", clock=lambda: _SETTLED_NOW)
            )

        assert counter["full"] == 1
        rank0, worker = resolutions
        assert worker.requested == rank0.requested == [25544]
        assert worker.obs_epoch_jd == rank0.obs_epoch_jd
        assert worker.catalogue_epoch_jd == rank0.catalogue_epoch_jd
        assert worker.remote_max_age_days == rank0.remote_max_age_days
        assert worker.resolved[25544].epoch_jd == rank0.resolved[25544].epoch_jd
        assert worker.resolved[25544].offset_days == rank0.resolved[25544].offset_days
        assert worker.frame()["TLE_LINE1"].iloc[0] == rank0.frame()["TLE_LINE1"].iloc[0]

    def test_preflight_never_reads_the_ms_on_a_worker(self, cache_dir, monkeypatch):
        _serve_catalogue(monkeypatch, [(25544, jd(2023, 2, 21, 13))])
        cluster = _FakeCluster(monkeypatch)
        config = TLEConfig(norad_ids=[25544])

        monkeypatch.setattr(tle, "_ms_mean_epoch_jd", lambda ms: _OBS)
        cluster.rank = 0
        tle.preflight_tle_check(config, "ignored.ms", clock=lambda: _SETTLED_NOW)

        def boom(ms_path):
            raise AssertionError("a worker must not resolve, or read the MS, itself")

        monkeypatch.setattr(tle, "_ms_mean_epoch_jd", boom)
        _no_network(monkeypatch)
        cluster.rank = 1
        assert tle.preflight_tle_check(
            config, "ignored.ms", clock=lambda: _SETTLED_NOW
        ).complete

    def test_preflight_coverage_failure_stops_every_rank(self, cache_dir, monkeypatch):
        _serve_catalogue(monkeypatch, [])
        _serve_nearest(monkeypatch, [])
        monkeypatch.setattr(tle, "_ms_mean_epoch_jd", lambda ms: _OBS)
        cluster = _FakeCluster(monkeypatch)
        config = TLEConfig(norad_ids=[25544])

        for rank in (0, 1):
            cluster.rank = rank
            with pytest.raises(TLEError, match="25544"):
                tle.preflight_tle_check(config, "ignored.ms", clock=lambda: _SETTLED_NOW)
