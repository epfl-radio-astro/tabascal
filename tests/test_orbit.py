"""Focused tests for TLE source precedence, coverage, parsing, and sharing."""

import json

import pandas as pd
import pytest

from tabascal import orbit

tle = orbit

from .tle_helpers import (  # noqa: F401
    block_network,
    jd,
    make_catalogue_df,
    make_omm_catalogue_df,
    write_legacy_omm_file,
    write_legacy_tle_file,
)


OBS = jd(2023, 2, 21)


def test_selects_nearest_valid_extra_record(tmp_path):
    write_legacy_tle_file(
        tmp_path / "records.json", [(25544, OBS - 2), (25544, OBS + 0.25)]
    )
    resolved, rejected = tle._select_from_extra_dir(str(tmp_path), {25544}, OBS, 1)
    assert rejected == {}
    assert resolved[25544].offset_days == pytest.approx(0.25, abs=2e-8)


def test_extra_record_outside_limit_is_reported(tmp_path):
    write_legacy_tle_file(tmp_path / "records.json", [(25544, OBS - 2)])
    resolved, rejected = tle._select_from_extra_dir(str(tmp_path), {25544}, OBS, 1)
    assert resolved == {}
    assert rejected[25544].age_days == pytest.approx(2, abs=2e-8)


def test_remote_epoch_is_parsed_from_line_one_not_provider_field():
    record = make_catalogue_df([(25544, OBS - 0.5)]).iloc[0].to_dict()
    record["EPOCH"] = "1900-01-01"
    resolution = tle.TLEResolution([25544], OBS, 3)
    tle._accept_remote(
        {25544: record}, "test", OBS, 3, resolution.resolved, resolution.rejected
    )
    assert resolution.resolved[25544].age_days == pytest.approx(0.5, abs=2e-8)


def test_fresher_candidate_replaces_incumbent_but_staler_does_not():
    resolution = tle.TLEResolution([25544], OBS, 3)
    old = make_catalogue_df([(25544, OBS - 2)]).iloc[0].to_dict()
    fresh = make_catalogue_df([(25544, OBS - 0.2)]).iloc[0].to_dict()
    tle._accept_remote({25544: old}, "old", OBS, 3, resolution.resolved, resolution.rejected)
    tle._accept_remote({25544: fresh}, "fresh", OBS, 3, resolution.resolved, resolution.rejected)
    tle._accept_remote({25544: old}, "old", OBS, 3, resolution.resolved, resolution.rejected)
    assert resolution.resolved[25544].source == "fresh"


def test_complete_coverage_names_every_missing_id():
    resolution = tle.TLEResolution([1, 2], OBS, 3)
    with pytest.raises(tle.TLEError) as caught:
        tle.require_complete_coverage(resolution)
    assert "1:" in str(caught.value)
    assert "2:" in str(caught.value)


def test_finalise_records_parses_elements():
    records = make_catalogue_df([(25544, OBS)]).to_dict(orient="records")
    frame = tle._finalise_records(records)
    for column in ("EPOCH_JD", "SEMIMAJOR_AXIS", "ECCENTRICITY", "MEAN_MOTION"):
        assert column in frame


def test_resolution_wire_round_trip():
    record = make_catalogue_df([(25544, OBS)]).iloc[0].to_dict()
    resolution = tle.TLEResolution([25544], OBS, 3)
    tle._accept_remote(
        {25544: record}, "test", OBS, 3, resolution.resolved, resolution.rejected
    )
    restored = tle._resolution_from_wire(
        json.loads(json.dumps(tle._resolution_to_wire(resolution)))
    )
    assert restored.requested == [25544]
    assert restored.resolved[25544].record["TLE_LINE1"] == record["TLE_LINE1"]


def test_save_orbits_for_reuse(tmp_path):
    records = make_catalogue_df([(25544, OBS)]).to_dict(orient="records")
    path = tmp_path / "used.json"
    tle.save_orbits_for_reuse(path, [25544], records)
    loaded = pd.read_json(path)
    assert loaded["NORAD_CAT_ID"].tolist() == [25544]


# ---------------------------------------------------------------------------
# extra_orbit_dir accepts either kind
# ---------------------------------------------------------------------------

class TestExtraDirRecordKinds:
    """``extra_orbit_dir`` is the user's own data, in whichever format they have.

    Space-Track publishes OMM as well as TLEs and its OMM history runs years
    deep, so an archival run whose epoch predates SatChecker's OMM archive can
    still be served from a local file. Neither format carries a kind field in
    those exports, so both are inferred.
    """

    def test_an_omm_file_resolves(self, tmp_path):
        write_legacy_omm_file(tmp_path / "omm.json", [(25544, OBS + 0.25)])
        resolved, rejected = tle._select_from_extra_dir(str(tmp_path), {25544}, OBS, 1)
        assert rejected == {}
        assert resolved[25544].offset_days == pytest.approx(0.25, abs=1e-6)

    def test_the_nearest_record_wins_across_kinds(self, tmp_path):
        # Precedence is by epoch distance, not by format.
        write_legacy_tle_file(tmp_path / "a-tle.json", [(25544, OBS - 2)])
        write_legacy_omm_file(tmp_path / "b-omm.json", [(25544, OBS + 0.25)])
        resolved, _ = tle._select_from_extra_dir(str(tmp_path), {25544}, OBS, 1)
        assert resolved[25544].offset_days == pytest.approx(0.25, abs=1e-6)

    def test_a_tle_still_wins_when_it_is_the_nearer_one(self, tmp_path):
        write_legacy_tle_file(tmp_path / "a-tle.json", [(25544, OBS - 0.1)])
        write_legacy_omm_file(tmp_path / "b-omm.json", [(25544, OBS + 2)])
        resolved, _ = tle._select_from_extra_dir(str(tmp_path), {25544}, OBS, 1)
        assert resolved[25544].offset_days == pytest.approx(-0.1, abs=1e-6)

    def test_an_omm_file_with_a_corrupt_element_is_rejected(self, tmp_path):
        frame = make_omm_catalogue_df([(25544, OBS)]).drop(columns=["RECORD_KIND"])
        frame.loc[0, "INCLINATION"] = 999.0
        frame.to_json(tmp_path / "omm.json")
        resolved, rejected = tle._select_from_extra_dir(str(tmp_path), {25544}, OBS, 1)
        assert resolved == {}

    def test_an_omm_element_frame_is_finalised(self, tmp_path):
        write_legacy_omm_file(tmp_path / "omm.json", [(25544, OBS)])
        resolved, _ = tle._select_from_extra_dir(str(tmp_path), {25544}, OBS, 1)
        frame = tle._finalise_records([resolved[25544].record])
        for column in ("EPOCH_JD", "SEMIMAJOR_AXIS", "ECCENTRICITY", "MEAN_MOTION"):
            assert column in frame
        assert frame.loc[0, "SEMIMAJOR_AXIS"] > 6000.0


# ---------------------------------------------------------------------------
# Replay round-trip
# ---------------------------------------------------------------------------

class TestReplayRoundTrip:
    """A run's records must be readable back as the same records.

    This is what makes a run reproducible independently of the shared cache, of
    what SatChecker serves by then, and of the age ceiling. A TLE needs only its
    two lines written out — every element is encoded in them. An OMM needs its
    epoch and its seven elements, because nothing else carries them, which is
    why the file format could not stay a table of line pairs.
    """

    def _round_trip(self, tmp_path, records):
        ids = [int(r["NORAD_CAT_ID"]) for r in records]
        assert tle.save_orbits_for_reuse(tmp_path / "used.json", ids, records)
        resolved, _ = tle._select_from_extra_dir(str(tmp_path), set(ids), OBS, None)
        return resolved

    @pytest.mark.parametrize("kind", ["tle", "omm"])
    def test_a_saved_record_resolves_again(self, tmp_path, kind):
        from .tle_helpers import make_record

        record = make_record(kind, 25544, OBS - 0.3)
        resolved = self._round_trip(tmp_path, [record])
        assert resolved[25544].offset_days == pytest.approx(-0.3, abs=1e-6)

    @pytest.mark.parametrize("kind", ["tle", "omm"])
    def test_the_replayed_elements_are_identical(self, tmp_path, kind):
        from tabascal.satchecker.records import record_elements

        from .tle_helpers import make_record

        record = make_record(kind, 25544, OBS)
        before = record_elements(record)
        after = record_elements(self._round_trip(tmp_path, [record])[25544].record)
        assert before == after

    def test_a_mixed_run_replays_both_kinds(self, tmp_path):
        from .tle_helpers import make_omm, make_tle_record

        records = [make_tle_record(25544, OBS), make_omm(43013, OBS - 0.1)]
        resolved = self._round_trip(tmp_path, records)
        assert set(resolved) == {25544, 43013}
        assert resolved[43013].offset_days == pytest.approx(-0.1, abs=1e-6)

    def test_the_kind_is_written_explicitly(self, tmp_path):
        from .tle_helpers import make_omm

        record = make_omm(25544, OBS)
        tle.save_orbits_for_reuse(tmp_path / "used.json", [25544], [record])
        written = pd.read_json(tmp_path / "used.json")
        assert written.loc[0, "RECORD_KIND"] == "omm"

    def test_derived_columns_are_not_written(self, tmp_path):
        # EPOCH_JD and SEMIMAJOR_AXIS are recomputed on every read; a second
        # stored copy could silently disagree with the elements it came from.
        from .tle_helpers import make_omm

        record = tle._finalise_records([make_omm(25544, OBS)]).to_dict(
            orient="records"
        )[0]
        tle.save_orbits_for_reuse(tmp_path / "used.json", [25544], [record])
        written = pd.read_json(tmp_path / "used.json")
        assert "EPOCH_JD" not in written.columns
        assert "SEMIMAJOR_AXIS" not in written.columns

    def test_a_tle_replay_file_carries_no_element_columns(self, tmp_path):
        # _finalise_records adds derived elements to a TLE row too. Writing them
        # would make the file describe the same orbit twice.
        record = tle._finalise_records(
            make_catalogue_df([(25544, OBS)]).to_dict(orient="records")
        ).to_dict(orient="records")[0]
        tle.save_orbits_for_reuse(tmp_path / "used.json", [25544], [record])
        written = pd.read_json(tmp_path / "used.json")
        assert "MEAN_MOTION" not in written.columns
        assert written.loc[0, "TLE_LINE1"].startswith("1 25544")

    def test_nothing_to_save_writes_no_file(self, tmp_path):
        assert tle.save_orbits_for_reuse(tmp_path / "used.json", [], []) is None
        assert tle.save_orbits_for_reuse(tmp_path / "used.json", [25544], None) is None
        assert not (tmp_path / "used.json").exists()

    def test_the_historical_name_still_works(self, tmp_path):
        records = make_catalogue_df([(25544, OBS)]).to_dict(orient="records")
        assert tle.save_tles_for_reuse(tmp_path / "used.json", [25544], records)
