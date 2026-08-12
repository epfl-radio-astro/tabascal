"""Offline tests for the TABASCAL TLE orchestration (``tabascal.tle``).

Exercises per-NORAD-ID source precedence, the ``extra_tle_dir`` age policy, the
managed canonical-snapshot cache, and the per-satellite fallback — all with the
SatChecker transport mocked and the managed cache pointed at a temp directory.
"""

import json
import urllib.error
from datetime import datetime

import pytest

from tabascal import tle
from tabascal.tle_config import TLEConfig
from tabascal.satchecker import (
    EmptyCatalogueError,
    SatCheckerError,
    SatCheckerResponseError,
    SatCheckerTransportError,
    client,
)
from tabascal.satchecker.client import CatalogueResult
from tabascal.time import datetime_to_jd

from .tle_helpers import (
    block_network,  # noqa: F401  autouse fixture: no live SatChecker access
    jd,
    make_catalogue_df,
    make_info_json,
    make_json_page,
    make_tle,
    write_legacy_tle_file,
)

TLEError = tle.TLEError

_OBS = jd(2023, 2, 21, 12, 30)
# A clock far enough past the observation that its catalogue counts as settled
# under the default 45-day policy, so the default tests exercise the stable path.
_SETTLED_NOW = jd(2023, 6, 1)

_ELEMENT_COLUMNS = [
    "SEMIMAJOR_AXIS", "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER", "MEAN_ANOMALY", "MEAN_MOTION", "BSTAR", "EPOCH_JD",
]


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Point the managed cache at an isolated temp directory."""
    d = tmp_path / "managed"
    monkeypatch.setenv("TLE_CACHE_DIR", str(d))
    return d


def _install_full(monkeypatch, pairs, counter):
    def fake(epoch_jd):
        counter["full"] += 1
        df = make_catalogue_df(pairs)
        return CatalogueResult(df, len(df), len(df), "1.6.0", "zip")
    monkeypatch.setattr(tle.satchecker, "fetch_full_catalogue", fake)


def _install_full_raise(monkeypatch):
    def fake(epoch_jd):
        raise SatCheckerError("network down")
    monkeypatch.setattr(tle.satchecker, "fetch_full_catalogue", fake)


def _install_full_empty(monkeypatch, counter):
    """Service reachable but no catalogue at the epoch (beyond its data horizon)."""
    def fake(epoch_jd):
        counter["full"] += 1
        raise EmptyCatalogueError("no records at this epoch")
    monkeypatch.setattr(tle.satchecker, "fetch_full_catalogue", fake)


def _install_nearest(monkeypatch, pairs, counter):
    lookup = {nid: ep for nid, ep in pairs}

    def fake(norad_id, epoch_jd):
        counter["near"] += 1
        if norad_id in lookup:
            return make_catalogue_df([(norad_id, lookup[norad_id])])
        import pandas as pd
        return pd.DataFrame()
    monkeypatch.setattr(tle.satchecker, "fetch_nearest_tle", fake)


def _install_nearest_raise(monkeypatch):
    def fake(norad_id, epoch_jd):
        raise SatCheckerError("network down")
    monkeypatch.setattr(tle.satchecker, "fetch_nearest_tle", fake)


# ---------------------------------------------------------------------------
# Managed catalogue + caching
# ---------------------------------------------------------------------------

class TestManagedCatalogue:

    def test_no_ids_needs_no_provider(self, cache_dir, monkeypatch):
        _install_full_raise(monkeypatch)
        assert tle.get_tles_by_id([], _OBS).empty

    def test_fetch_then_parse_elements(self, cache_dir, monkeypatch):
        counter = {"full": 0}
        _install_full(monkeypatch, [(25544, jd(2023, 2, 21, 13))], counter)
        df = tle.get_tles_by_id([25544], _OBS)
        assert list(df["NORAD_CAT_ID"]) == [25544]
        assert set(_ELEMENT_COLUMNS).issubset(df.columns)  # elements consumed by trajectory.py
        assert counter["full"] == 1

    def test_valid_cache_hit_does_no_http(self, cache_dir, monkeypatch):
        counter = {"full": 0}
        _install_full(monkeypatch, [(25544, jd(2023, 2, 21, 13))], counter)
        tle.get_tles_by_id([25544], _OBS)
        assert counter["full"] == 1
        # Second call: any HTTP attempt would raise; it must come from cache.
        _install_full_raise(monkeypatch)
        df = tle.get_tles_by_id([25544], _OBS)
        assert list(df["NORAD_CAT_ID"]) == [25544]

    def test_failed_download_leaves_no_snapshot(self, cache_dir, monkeypatch):
        _install_full_raise(monkeypatch)
        with pytest.raises(SatCheckerError):
            tle.get_tles_by_id([25544], _OBS)
        assert not list(cache_dir.glob("catalogue-*.json"))

    def test_result_has_one_row_per_id(self, cache_dir, monkeypatch):
        counter = {"full": 0}
        pairs = [(25544, jd(2023, 2, 21, 13)), (38833, jd(2023, 2, 21, 13))]
        _install_full(monkeypatch, pairs, counter)
        df = tle.get_tles_by_id([25544, 38833], _OBS)
        assert sorted(df["NORAD_CAT_ID"]) == [25544, 38833]
        assert len(df) == 2

    def test_invalid_bulk_row_below_threshold_falls_back_without_cache_failure(
        self, cache_dir, monkeypatch
    ):
        records = make_catalogue_df([(25544, _OBS), (38833, _OBS)])
        records.loc[1, "TLE_LINE2"] = "not a TLE"
        monkeypatch.setattr(
            tle.satchecker,
            "fetch_full_catalogue",
            lambda epoch: CatalogueResult(records, 2, 2, "1.6.0", "zip"),
        )
        counter = {"near": 0}
        _install_nearest(monkeypatch, [(25544, _OBS)], counter)

        df = tle.get_tles_by_id([25544], _OBS)

        assert list(df["NORAD_CAT_ID"]) == [25544]
        assert counter["near"] == 1
        assert not list(cache_dir.glob("catalogue-*[0-9]Z.json"))

    def test_invalid_bulk_row_is_removed_before_acceptable_snapshot_is_cached(
        self, cache_dir, monkeypatch
    ):
        records = make_catalogue_df([(nid, _OBS) for nid in range(1000, 1100)])
        records.loc[records["NORAD_CAT_ID"] == 1099, "TLE_LINE1"] = "not a TLE"
        monkeypatch.setattr(
            tle.satchecker,
            "fetch_full_catalogue",
            lambda epoch: CatalogueResult(records, 100, 100, "1.6.0", "zip"),
        )

        df = tle.get_tles_by_id([1000], _OBS)

        assert list(df["NORAD_CAT_ID"]) == [1000]
        snapshot_files = list(cache_dir.glob("catalogue-*[0-9]Z.json"))
        assert len(snapshot_files) == 1
        envelope = json.loads(snapshot_files[0].read_text())
        assert envelope["expected_count"] == 100
        assert envelope["actual_count"] == 99
        assert len(envelope["records"]) == 99


# ---------------------------------------------------------------------------
# Per-satellite fallback
# ---------------------------------------------------------------------------

class TestFallback:

    def test_missing_id_uses_individual_fallback(self, cache_dir, monkeypatch):
        cfull, cnear = {"full": 0}, {"near": 0}
        _install_full(monkeypatch, [(25544, jd(2023, 2, 21, 13))], cfull)
        _install_nearest(monkeypatch, [(99999, jd(2023, 2, 21, 13))], cnear)
        df = tle.get_tles_by_id([25544, 99999], _OBS)
        assert sorted(df["NORAD_CAT_ID"]) == [25544, 99999]
        assert cnear["near"] == 1

    def test_fallback_record_reused_from_cache(self, cache_dir, monkeypatch):
        cfull, cnear = {"full": 0}, {"near": 0}
        _install_full(monkeypatch, [(25544, jd(2023, 2, 21, 13))], cfull)
        _install_nearest(monkeypatch, [(99999, jd(2023, 2, 21, 13))], cnear)
        tle.get_tles_by_id([25544, 99999], _OBS)
        assert cnear["near"] == 1
        # Second run: both bulk snapshot and fallback come from cache; no HTTP.
        _install_full_raise(monkeypatch)
        _install_nearest_raise(monkeypatch)
        df = tle.get_tles_by_id([25544, 99999], _OBS)
        assert sorted(df["NORAD_CAT_ID"]) == [25544, 99999]

    def test_empty_catalogue_falls_back_per_satellite(self, cache_dir, monkeypatch):
        # Live-observed: tles-at-epoch reports zero records for epochs beyond the
        # service's ingest horizon while get-nearest-tle still resolves. The whole
        # request must succeed via the per-satellite path, not hard-fail.
        cfull, cnear = {"full": 0}, {"near": 0}
        _install_full_empty(monkeypatch, cfull)
        _install_nearest(monkeypatch, [(25544, _OBS), (38833, _OBS)], cnear)
        df = tle.get_tles_by_id([25544, 38833], _OBS)
        assert sorted(df["NORAD_CAT_ID"]) == [25544, 38833]
        assert cfull["full"] == 1 and cnear["near"] == 2
        assert "SEMIMAJOR_AXIS" in df.columns  # elements parsed as usual
        # No snapshot file may exist, but the fallback records are cached.
        assert not list(cache_dir.glob("catalogue-*[0-9]Z.json"))
        assert list(cache_dir.glob("catalogue-*-extra.json"))

    def test_empty_catalogue_fallback_reused_from_cache(self, cache_dir, monkeypatch):
        cfull, cnear = {"full": 0}, {"near": 0}
        _install_full_empty(monkeypatch, cfull)
        _install_nearest(monkeypatch, [(25544, _OBS)], cnear)
        tle.get_tles_by_id([25544], _OBS)
        assert cnear["near"] == 1
        # Second run: catalogue still empty, but the fallback record comes from
        # the extra cache — no further per-satellite requests.
        _install_nearest_raise(monkeypatch)
        df = tle.get_tles_by_id([25544], _OBS)
        assert list(df["NORAD_CAT_ID"]) == [25544]
        assert cfull["full"] == 2  # catalogue re-probed (cheap), fallback cached

    def test_transport_failure_still_fails_fast(self, cache_dir, monkeypatch):
        # A plain SatCheckerError (service unreachable) must NOT trigger the
        # per-satellite fallback — it propagates immediately.
        _install_full_raise(monkeypatch)
        _install_nearest_raise(monkeypatch)
        with pytest.raises(SatCheckerError):
            tle.get_tles_by_id([25544], _OBS)

    def test_invalid_fallback_row_isolated_but_coverage_still_enforced(
        self, cache_dir, monkeypatch
    ):
        # One bad row must not poison the *other* satellites' records — the valid
        # one is still cached — but the run cannot proceed with a satellite short.
        counter = {"full": 0}
        _install_full_empty(monkeypatch, counter)

        def nearest(norad_id, epoch_jd):
            rec = make_catalogue_df([(norad_id, _OBS)])
            if norad_id == 38833:
                rec.loc[0, "TLE_LINE2"] = "not a TLE"
            return rec

        monkeypatch.setattr(tle.satchecker, "fetch_nearest_tle", nearest)
        monkeypatch.setattr(tle.time, "sleep", lambda seconds: None)

        with pytest.raises(TLEError, match="38833"):
            tle.get_tles_by_id([25544, 38833], _OBS)

        extra_files = list(cache_dir.glob("catalogue-*-extra.json"))
        assert len(extra_files) == 1
        assert "25544" in extra_files[0].read_text()

    def test_transport_failure_never_storms_the_per_id_endpoint(
        self, cache_dir, monkeypatch
    ):
        # A service outage must not be followed by one request per satellite.
        def down(epoch_jd):
            raise SatCheckerTransportError("connection refused")

        monkeypatch.setattr(tle.satchecker, "fetch_full_catalogue", down)
        counter = {"near": 0}
        _install_nearest(monkeypatch, [(n, _OBS) for n in range(100, 140)], counter)

        with pytest.raises(SatCheckerTransportError):
            tle.get_tles_by_id(list(range(100, 140)), _OBS)
        assert counter["near"] == 0

    def test_response_failure_uses_the_per_id_fallback(self, cache_dir, monkeypatch):
        # The service is up but its catalogue response is unusable — a different
        # endpoint on a working service is worth trying.
        def malformed(epoch_jd):
            raise SatCheckerResponseError("catalogue payload was not valid JSON")

        monkeypatch.setattr(tle.satchecker, "fetch_full_catalogue", malformed)
        counter = {"near": 0}
        _install_nearest(monkeypatch, [(25544, _OBS)], counter)

        df = tle.get_tles_by_id([25544], _OBS)
        assert list(df["NORAD_CAT_ID"]) == [25544]
        assert counter["near"] == 1

    def test_per_id_transport_failure_stops_the_loop(self, cache_dir, monkeypatch):
        counter = {"full": 0}
        _install_full_empty(monkeypatch, counter)
        attempts = {"n": 0}

        def nearest(norad_id, epoch_jd):
            attempts["n"] += 1
            raise SatCheckerTransportError("connection refused")

        monkeypatch.setattr(tle.satchecker, "fetch_nearest_tle", nearest)
        monkeypatch.setattr(tle.time, "sleep", lambda seconds: None)

        with pytest.raises(SatCheckerTransportError):
            tle.get_tles_by_id(list(range(100, 120)), _OBS)
        assert attempts["n"] == 1  # stopped at the first unreachable request

    def test_bulk_http_client_error_uses_the_per_id_fallback(
        self, cache_dir, monkeypatch
    ):
        # HTTPError subclasses URLError, so without explicit handling a 404 on the
        # bulk endpoint was typed as transport and suppressed the fallback — even
        # though the service had plainly answered and the other route still worked.
        import urllib.request

        def boom(req, timeout=None):
            raise urllib.error.HTTPError("http://x/tles-at-epoch/", 404, "no", {}, None)

        monkeypatch.setattr(urllib.request, "urlopen", boom)
        counter = {"near": 0}
        _install_nearest(monkeypatch, [(25544, _OBS)], counter)

        df = tle.get_tles_by_id([25544], _OBS)

        assert list(df["NORAD_CAT_ID"]) == [25544]
        assert counter["near"] == 1

    def test_bulk_http_server_error_still_fails_fast(self, cache_dir, monkeypatch):
        import urllib.request

        def boom(req, timeout=None):
            raise urllib.error.HTTPError("http://x/tles-at-epoch/", 503, "down", {}, None)

        monkeypatch.setattr(urllib.request, "urlopen", boom)
        counter = {"near": 0}
        _install_nearest(monkeypatch, [(n, _OBS) for n in range(100, 140)], counter)

        with pytest.raises(SatCheckerTransportError):
            tle.get_tles_by_id(list(range(100, 140)), _OBS)
        assert counter["near"] == 0

    def test_repeated_per_id_response_failures_stop_the_loop(
        self, cache_dir, monkeypatch, capsys
    ):
        # Error typing cannot bound a fault in the *request* rather than the
        # service: a malformed epoch is a response error on every satellite alike.
        counter = {"full": 0}
        _install_full_empty(monkeypatch, counter)
        attempts = {"n": 0}

        def always_bad(norad_id, epoch_jd):
            attempts["n"] += 1
            raise SatCheckerResponseError("epoch parameter rejected")

        monkeypatch.setattr(tle.satchecker, "fetch_nearest_tle", always_bad)
        monkeypatch.setattr(tle.time, "sleep", lambda seconds: None)

        with pytest.raises(TLEError) as exc_info:
            tle.get_tles_by_id(list(range(100, 160)), _OBS)

        assert attempts["n"] == 3  # not 60
        assert "stopping per-satellite fetches" in capsys.readouterr().out
        assert "60 configured satellites" in str(exc_info.value)

    def test_an_isolated_per_id_failure_does_not_stop_the_loop(
        self, cache_dir, monkeypatch
    ):
        counter = {"full": 0}
        _install_full_empty(monkeypatch, counter)
        attempts = {"n": 0}

        def flaky(norad_id, epoch_jd):
            attempts["n"] += 1
            if norad_id == 102:
                raise SatCheckerResponseError("one bad record")
            return make_catalogue_df([(norad_id, _OBS)])

        monkeypatch.setattr(tle.satchecker, "fetch_nearest_tle", flaky)
        monkeypatch.setattr(tle.time, "sleep", lambda seconds: None)

        with pytest.raises(TLEError, match="102"):
            tle.get_tles_by_id([100, 101, 102, 103, 104], _OBS)
        assert attempts["n"] == 5  # the loop ran to completion


class TestDuplicateNoradSelection:
    """When the catalogue legitimately carries several TLEs for one NORAD ID,
    the record nearest the canonical epoch must win — independent of row order."""

    # canonical epoch for _OBS (12:30 in the default 2 h bucket) is 13:00 UTC
    _NEAR = jd(2023, 2, 21, 13, 30)   # 0.5 h from canonical epoch
    _FAR = jd(2023, 2, 21, 4, 0)      # 9 h from canonical epoch

    @pytest.mark.parametrize("order", ["far_first", "near_first"])
    def test_nearest_to_canonical_epoch_wins(self, cache_dir, monkeypatch, order):
        pairs = [(25544, self._FAR), (25544, self._NEAR)]
        if order == "near_first":
            pairs = pairs[::-1]
        counter = {"full": 0}
        _install_full(monkeypatch, pairs, counter)
        df = tle.get_tles_by_id([25544], _OBS)
        expected_l1, _ = make_tle(25544, self._NEAR)
        assert len(df) == 1
        assert df["TLE_LINE1"].iloc[0] == expected_l1  # same result either order


# ---------------------------------------------------------------------------
# extra_tle_dir precedence + age policy
# ---------------------------------------------------------------------------

class TestExtraDirPrecedence:

    def test_fresh_extra_wins_over_closer_managed(self, cache_dir, tmp_path, monkeypatch):
        extra = tmp_path / "extra"
        extra.mkdir()
        extra_epoch = jd(2023, 2, 20, 12, 30)  # one day before the observation
        write_legacy_tle_file(extra / "local.json", [(25544, extra_epoch)])
        counter = {"full": 0}
        # Managed snapshot has a record exactly at the observation (closer), but
        # extra_tle_dir must still win for that ID — so it is never fetched.
        _install_full(monkeypatch, [(25544, _OBS)], counter)
        df = tle.get_tles_by_id([25544], _OBS, extra_tle_dir=str(extra))
        expected_l1, _ = make_tle(25544, extra_epoch)
        assert df.loc[df["NORAD_CAT_ID"] == 25544, "TLE_LINE1"].iloc[0] == expected_l1
        assert counter["full"] == 0

    def test_stale_extra_falls_through_to_managed(self, cache_dir, tmp_path, monkeypatch):
        extra = tmp_path / "extra"
        extra.mkdir()
        write_legacy_tle_file(extra / "local.json", [(25544, jd(2023, 2, 11, 12, 30))])  # 10 d old
        counter = {"full": 0}
        _install_full(monkeypatch, [(25544, _OBS)], counter)
        df = tle.get_tles_by_id([25544], _OBS, extra_tle_dir=str(extra), extra_tle_max_age_days=1)
        expected_l1, _ = make_tle(25544, _OBS)
        assert df.loc[df["NORAD_CAT_ID"] == 25544, "TLE_LINE1"].iloc[0] == expected_l1
        assert counter["full"] == 1

    def test_mislabeled_extra_tle_falls_through_to_managed(
        self, cache_dir, tmp_path, monkeypatch
    ):
        extra = tmp_path / "extra"
        extra.mkdir()
        local = make_catalogue_df([(38833, _OBS)])
        local["NORAD_CAT_ID"] = 25544  # metadata and embedded TLE IDs disagree
        local.to_json(extra / "mislabeled.json")

        counter = {"full": 0}
        _install_full(monkeypatch, [(25544, _OBS)], counter)
        df = tle.get_tles_by_id([25544], _OBS, extra_tle_dir=str(extra))

        expected_l1, _ = make_tle(25544, _OBS)
        assert df["TLE_LINE1"].iloc[0] == expected_l1
        assert counter["full"] == 1

    def test_stale_extra_missing_in_managed_uses_fallback(self, cache_dir, tmp_path, monkeypatch):
        extra = tmp_path / "extra"
        extra.mkdir()
        write_legacy_tle_file(extra / "local.json", [(25544, jd(2023, 2, 11, 12, 30))])  # 10 d old
        cfull, cnear = {"full": 0}, {"near": 0}
        _install_full(monkeypatch, [(99999, _OBS)], cfull)  # non-empty, but lacks the ID
        _install_nearest(monkeypatch, [(25544, _OBS)], cnear)
        df = tle.get_tles_by_id([25544], _OBS, extra_tle_dir=str(extra), extra_tle_max_age_days=1)
        expected_l1, _ = make_tle(25544, _OBS)
        assert df.loc[df["NORAD_CAT_ID"] == 25544, "TLE_LINE1"].iloc[0] == expected_l1
        assert cnear["near"] == 1

    def test_null_max_age_is_unlimited(self, cache_dir, tmp_path, monkeypatch):
        extra = tmp_path / "extra"
        extra.mkdir()
        old_epoch = jd(2022, 11, 13, 12, 30)  # ~100 days before the observation
        write_legacy_tle_file(extra / "local.json", [(25544, old_epoch)])
        _install_full_raise(monkeypatch)  # must not be reached
        df = tle.get_tles_by_id([25544], _OBS, extra_tle_dir=str(extra), extra_tle_max_age_days=None)
        expected_l1, _ = make_tle(25544, old_epoch)
        assert df.loc[df["NORAD_CAT_ID"] == 25544, "TLE_LINE1"].iloc[0] == expected_l1

    def test_zero_max_age_accepts_exact_epoch(self, cache_dir, tmp_path, monkeypatch):
        extra = tmp_path / "extra"
        extra.mkdir()
        write_legacy_tle_file(extra / "local.json", [(25544, _OBS)])  # exact observation epoch
        _install_full_raise(monkeypatch)
        df = tle.get_tles_by_id([25544], _OBS, extra_tle_dir=str(extra), extra_tle_max_age_days=0)
        assert list(df["NORAD_CAT_ID"]) == [25544]

    def test_zero_max_age_rejects_offset_epoch(self, cache_dir, tmp_path, monkeypatch):
        extra = tmp_path / "extra"
        extra.mkdir()
        write_legacy_tle_file(extra / "local.json", [(25544, jd(2023, 2, 22, 12, 30))])  # 1 day off
        counter = {"full": 0}
        _install_full(monkeypatch, [(25544, _OBS)], counter)
        df = tle.get_tles_by_id([25544], _OBS, extra_tle_dir=str(extra), extra_tle_max_age_days=0)
        expected_l1, _ = make_tle(25544, _OBS)  # came from managed, not the offset extra
        assert df.loc[df["NORAD_CAT_ID"] == 25544, "TLE_LINE1"].iloc[0] == expected_l1
        assert counter["full"] == 1

    def test_negative_max_age_is_a_config_error(self, cache_dir, tmp_path, monkeypatch):
        with pytest.raises(ValueError):
            tle.get_tles_by_id([25544], _OBS, extra_tle_max_age_days=-1)

    def test_age_measured_from_line1_not_filename(self, cache_dir, tmp_path, monkeypatch):
        extra = tmp_path / "extra"
        extra.mkdir()
        # Filename implies year 2099, but the TLE epoch is the observation day.
        write_legacy_tle_file(extra / "2099-01-01-misleading.json", [(25544, _OBS)])
        _install_full_raise(monkeypatch)
        df = tle.get_tles_by_id([25544], _OBS, extra_tle_dir=str(extra), extra_tle_max_age_days=1)
        assert list(df["NORAD_CAT_ID"]) == [25544]

    def test_precedence_resolved_per_id(self, cache_dir, tmp_path, monkeypatch):
        extra = tmp_path / "extra"
        extra.mkdir()
        extra_epoch = jd(2023, 2, 20, 12, 30)
        write_legacy_tle_file(extra / "local.json", [(25544, extra_epoch)])  # only 25544 locally
        counter = {"full": 0}
        _install_full(monkeypatch, [(25544, _OBS), (38833, _OBS)], counter)
        df = tle.get_tles_by_id([25544, 38833], _OBS, extra_tle_dir=str(extra))
        l1_25544, _ = make_tle(25544, extra_epoch)  # from extra
        l1_38833, _ = make_tle(38833, _OBS)          # from managed
        assert df.loc[df["NORAD_CAT_ID"] == 25544, "TLE_LINE1"].iloc[0] == l1_25544
        assert df.loc[df["NORAD_CAT_ID"] == 38833, "TLE_LINE1"].iloc[0] == l1_38833
        assert counter["full"] == 1


# ---------------------------------------------------------------------------
# Exact-epoch tolerance (extra_tle_max_age_days: 0)
# ---------------------------------------------------------------------------

class TestExactEpochTolerance:
    """The observation epochs here are built directly from datetimes, not via the
    implementation's TLE helper, so conversion error cannot be hidden."""

    def test_within_tle_precision_accepted_at_zero(self, cache_dir, tmp_path, monkeypatch):
        extra = tmp_path / "extra"
        extra.mkdir()
        t = datetime_to_jd(datetime(2023, 2, 21, 12, 30, 0))
        write_legacy_tle_file(extra / "local.json", [(25544, t)])
        _install_full_raise(monkeypatch)  # accepted from extra -> no fetch
        obs = t + 1e-9  # ~0.09 ms, inside the ~2.6 ms tolerance
        df = tle.get_tles_by_id([25544], obs, extra_tle_dir=str(extra), extra_tle_max_age_days=0)
        assert list(df["NORAD_CAT_ID"]) == [25544]

    def test_beyond_tle_precision_rejected_at_zero(self, cache_dir, tmp_path, monkeypatch):
        extra = tmp_path / "extra"
        extra.mkdir()
        t = datetime_to_jd(datetime(2023, 2, 21, 12, 30, 0))
        write_legacy_tle_file(extra / "local.json", [(25544, t)])
        counter = {"full": 0}
        _install_full(monkeypatch, [(25544, _OBS)], counter)  # managed carries it
        obs = t + 10e-3 / 86400.0  # 10 ms: far under the old 86.4 ms, over the new tol
        df = tle.get_tles_by_id([25544], obs, extra_tle_dir=str(extra), extra_tle_max_age_days=0)
        assert list(df["NORAD_CAT_ID"]) == [25544]
        assert counter["full"] == 1  # extra rejected -> managed consulted

    def test_configured_nonzero_boundary(self, cache_dir, tmp_path, monkeypatch):
        extra = tmp_path / "extra"
        extra.mkdir()
        # obs 2023-02-21 12:30. A: 0.979 d away (inside 1 d); B: 1.021 d away (outside).
        write_legacy_tle_file(
            extra / "local.json",
            [(111, jd(2023, 2, 20, 13, 0)), (222, jd(2023, 2, 20, 12, 0))],
        )
        counter = {"full": 0}
        _install_full(monkeypatch, [(222, _OBS)], counter)  # only the rejected ID
        df = tle.get_tles_by_id(
            [111, 222], _OBS, extra_tle_dir=str(extra), extra_tle_max_age_days=1.0
        )
        assert df.loc[df["NORAD_CAT_ID"] == 111, "TLE_LINE1"].iloc[0] == make_tle(111, jd(2023, 2, 20, 13, 0))[0]
        assert df.loc[df["NORAD_CAT_ID"] == 222, "TLE_LINE1"].iloc[0] == make_tle(222, _OBS)[0]
        assert counter["full"] == 1


# ---------------------------------------------------------------------------
# Partial JSON catalogue is never cached (Issue 1, orchestration side)
# ---------------------------------------------------------------------------

class TestPartialCatalogueNotCached:

    def test_incomplete_download_raises_and_leaves_no_snapshot(self, cache_dir, monkeypatch):
        # Real client path: zip unreadable, then JSON pagination terminates early.
        def handler(url):
            if "format=zip" in url:
                return b"not a zip"
            if "per_page=1" in url:
                return make_info_json(total=4)
            if "&page=2" in url:
                return make_json_page([], total=4)  # premature empty page
            return make_json_page([(1, _OBS), (2, _OBS)], total=4)

        monkeypatch.setattr(
            client, "_http_get", lambda url, timeout=client.REQUEST_TIMEOUT: handler(url)
        )
        with pytest.raises(SatCheckerError):
            tle.get_tles_by_id([25544], _OBS)
        assert list(cache_dir.glob("catalogue-*.json")) == []


# ---------------------------------------------------------------------------
# Preflight source awareness
# ---------------------------------------------------------------------------

class TestPreflight:

    def _config(self, **overrides):
        fields = {"norad_ids": [25544]}
        fields.update(overrides)
        return TLEConfig(**fields)

    def test_full_extra_coverage_needs_no_provider(
        self, cache_dir, tmp_path, monkeypatch, capsys
    ):
        extra = tmp_path / "extra"
        extra.mkdir()
        write_legacy_tle_file(extra / "local.json", [(25544, _OBS)])
        monkeypatch.setattr(tle, "_ms_mean_epoch_jd", lambda ms: _OBS)
        _install_full_raise(monkeypatch)  # any provider call would fail the test

        resolution = tle.preflight_tle_check(
            self._config(extra_tle_dir=str(extra), extra_tle_max_age_days=0),
            "ignored.ms",
            clock=lambda: _SETTLED_NOW,
        )

        assert resolution.complete
        assert "1 from extra_tle_dir" in capsys.readouterr().out

    def test_preflight_resolves_and_reports_remote_records(
        self, cache_dir, monkeypatch, capsys
    ):
        counter = {"full": 0}
        _install_full(monkeypatch, [(25544, jd(2023, 2, 21, 13))], counter)
        monkeypatch.setattr(tle, "_ms_mean_epoch_jd", lambda ms: _OBS)

        resolution = tle.preflight_tle_check(
            self._config(), "ignored.ms", clock=lambda: _SETTLED_NOW
        )

        out = capsys.readouterr().out
        assert counter["full"] == 1
        assert list(resolution.resolved) == [25544]
        assert "TLE preflight OK" in out
        assert "age" in out  # every accepted remote record's age is reported

    def test_preflight_stops_before_subtraction_on_a_missing_id(
        self, cache_dir, monkeypatch
    ):
        counter = {"full": 0}
        _install_full(monkeypatch, [(25544, _OBS)], counter)
        _install_nearest(monkeypatch, [], {"near": 0})
        monkeypatch.setattr(tle, "_ms_mean_epoch_jd", lambda ms: _OBS)
        monkeypatch.setattr(tle.time, "sleep", lambda seconds: None)

        with pytest.raises(TLEError) as exc_info:
            tle.preflight_tle_check(
                self._config(norad_ids=[25544, 99999]),
                "ignored.ms",
                clock=lambda: _SETTLED_NOW,
            )
        message = str(exc_info.value)
        assert "99999" in message
        assert "--extra-tle-dir" in message
        assert "remote_tle_max_age_days" in message

    def test_no_configured_ids_does_no_work(self, cache_dir, monkeypatch):
        def boom(ms_path):
            raise AssertionError("the MS must not be read when no IDs are configured")

        monkeypatch.setattr(tle, "_ms_mean_epoch_jd", boom)
        resolution = tle.preflight_tle_check(TLEConfig(), "ignored.ms")
        assert resolution.requested == []
        assert resolution.complete

    def test_extra_dir_selection_is_logged_once(
        self, cache_dir, tmp_path, monkeypatch, capsys
    ):
        # Preflight used to run the extra-directory selection twice purely to
        # print it, duplicating every per-satellite line.
        extra = tmp_path / "extra"
        extra.mkdir()
        write_legacy_tle_file(extra / "local.json", [(25544, _OBS)])
        monkeypatch.setattr(tle, "_ms_mean_epoch_jd", lambda ms: _OBS)
        _install_full_raise(monkeypatch)

        tle.preflight_tle_check(
            self._config(extra_tle_dir=str(extra)),
            "ignored.ms",
            clock=lambda: _SETTLED_NOW,
        )

        out = capsys.readouterr().out
        assert out.count("25544: from extra_tle_dir") == 1
        assert out.count("TLE extra dir") == 1

    def test_cache_key_in_logs_matches_the_created_filename(
        self, cache_dir, monkeypatch, capsys
    ):
        counter = {"full": 0}
        _install_full(monkeypatch, [(25544, jd(2023, 2, 21, 13))], counter)
        monkeypatch.setattr(tle, "_ms_mean_epoch_jd", lambda ms: _OBS)

        tle.preflight_tle_check(
            self._config(), "ignored.ms", clock=lambda: _SETTLED_NOW
        )

        out = capsys.readouterr().out
        written = list(cache_dir.glob("catalogue-*[0-9]Z.json"))
        assert len(written) == 1
        stamp = written[0].stem.removeprefix("catalogue-")
        assert stamp in out


class TestEpochAgreement:
    """Preflight and execution must judge the run at the same instant."""

    def _resolution(self, obs_epoch_jd):
        return tle.TLEResolution(
            requested=[25544],
            obs_epoch_jd=obs_epoch_jd,
            catalogue_epoch_jd=obs_epoch_jd,
            remote_max_age_days=3.0,
        )

    def test_matching_epochs_pass(self):
        tle.check_epoch_agreement(self._resolution(_OBS), [_OBS - 1e-9, _OBS + 1e-9])

    def test_divergent_epochs_raise(self):
        with pytest.raises(TLEError, match="Observation epoch disagreement"):
            tle.check_epoch_agreement(self._resolution(_OBS), [_OBS + 0.5])

    def test_no_satellites_needs_no_agreement(self):
        empty = tle.TLEResolution(
            requested=[], obs_epoch_jd=float("nan"),
            catalogue_epoch_jd=float("nan"), remote_max_age_days=None,
        )
        tle.check_epoch_agreement(empty, [_OBS])


# ---------------------------------------------------------------------------
# Reproducibility: save_tles_for_reuse round-trip
# ---------------------------------------------------------------------------

class TestSaveTlesForReuse:

    def test_round_trip_through_extra_tle_dir(self, cache_dir, tmp_path, monkeypatch):
        # Save the TLEs "a run used", then reproduce the resolution offline from
        # the saved file alone — the exact reproducibility workflow.
        out = tmp_path / "results"
        out.mkdir()
        epoch = jd(2023, 2, 21, 13)
        pairs = [make_tle(25544, epoch), make_tle(38833, epoch)]
        path = tle.save_tles_for_reuse(
            out / "used_tles_Custom.json", [25544, 38833], pairs
        )
        assert path is not None

        _install_full_raise(monkeypatch)  # no network allowed
        df = tle.get_tles_by_id([25544, 38833], _OBS, extra_tle_dir=str(out))
        assert sorted(df["NORAD_CAT_ID"]) == [25544, 38833]
        assert df.loc[df["NORAD_CAT_ID"] == 25544, "TLE_LINE1"].iloc[0] == pairs[0][0]

    def test_nothing_to_save_returns_none(self, tmp_path):
        assert tle.save_tles_for_reuse(tmp_path / "x.json", [], None) is None
        assert not (tmp_path / "x.json").exists()


# ---------------------------------------------------------------------------
# Config validation helper
# ---------------------------------------------------------------------------

class TestValidateMaxAge:

    @pytest.mark.parametrize("value", [None, 0, 0.5, 30, 100.0])
    def test_accepts_none_and_non_negative(self, value):
        assert tle._validate_max_age(value) == (None if value is None else float(value))

    @pytest.mark.parametrize("value", [-1, -0.001])
    def test_rejects_negative(self, value):
        with pytest.raises(ValueError):
            tle._validate_max_age(value)
