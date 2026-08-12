"""Focused tests for TLE source precedence, coverage, parsing, and sharing."""

import json

import pandas as pd
import pytest

from tabascal import tle

from .tle_helpers import block_network, jd, make_catalogue_df, write_legacy_tle_file  # noqa: F401


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


def test_save_tles_for_reuse(tmp_path):
    frame = make_catalogue_df([(25544, OBS)])
    path = tmp_path / "used.json"
    tle.save_tles_for_reuse(
        path, [25544], frame[["TLE_LINE1", "TLE_LINE2"]].to_numpy()
    )
    loaded = pd.read_json(path)
    assert loaded["NORAD_CAT_ID"].tolist() == [25544]
