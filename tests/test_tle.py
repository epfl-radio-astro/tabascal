"""Offline tests for the TABASCAL TLE orchestration (``tabascal.tle``).

Exercises per-NORAD-ID source precedence, the ``extra_tle_dir`` age policy, the
managed canonical-snapshot cache, and the per-satellite fallback — all with the
SatChecker transport mocked and the managed cache pointed at a temp directory.
"""

from datetime import datetime

import pytest

from tabascal import tle
from tabascal.satchecker import EmptyCatalogueError, SatCheckerError, client
from tabascal.satchecker.client import CatalogueResult
from tabascal.time import datetime_to_jd

from .tle_helpers import (
    jd,
    make_catalogue_df,
    make_info_json,
    make_json_page,
    make_tle,
    write_legacy_tle_file,
)

_OBS = jd(2023, 2, 21, 12, 30)

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

    def test_full_extra_coverage_promises_no_download(self, cache_dir, tmp_path, monkeypatch, capsys):
        extra = tmp_path / "extra"
        extra.mkdir()
        write_legacy_tle_file(extra / "local.json", [(25544, _OBS)])
        monkeypatch.setattr(tle, "_ms_mean_epoch_jd", lambda ms: _OBS)
        tle.preflight_tle_check(
            [25544], "ignored.ms", extra_tle_dir=str(extra), extra_tle_max_age_days=0
        )
        out = capsys.readouterr().out
        assert "will download" not in out
        assert "no download needed" in out

    def test_missing_extra_promises_download(self, cache_dir, monkeypatch, capsys):
        monkeypatch.setattr(tle, "_ms_mean_epoch_jd", lambda ms: _OBS)
        tle.preflight_tle_check([25544], "ignored.ms")
        out = capsys.readouterr().out
        assert "will download" in out


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
