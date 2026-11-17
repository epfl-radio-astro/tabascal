"""The bundled Space-Track records must stay acceptable to the shared parser.

The parser itself is tested in the satchecker-client package; this guards the
enforcement against the data tabascal actually ships: every bundled record must
remain acceptable.
"""

from datetime import datetime, timedelta
from importlib.resources import files
from pathlib import Path

import yaml
from satchecker_client import read_legacy_tle_records
from satchecker_client.records import record_epoch_jd, record_kind
from satchecker_client.tle_parse import validate_tle_pair

REFRAME_DATA = Path(__file__).parents[1] / "ci" / "reframe" / "data"


def test_real_bundled_records_satisfy_both_checks():
    df = read_legacy_tle_records(str(files("tabascal").joinpath("data/tles")))
    assert len(df) > 200
    for _, row in df.iterrows():
        validate_tle_pair(row["TLE_LINE1"], row["TLE_LINE2"])


def test_the_benchmark_bundles_an_orbit_for_every_satellite_it_simulates():
    """The ReFrame benchmark must never need SatChecker.

    Its tabsim config simulates offline from ``orbits/``, so a satellite with no
    record there, or one too far from the observation to be the record SatChecker
    would have served, fails the simulation (or quietly degrades it) on CI.
    """
    sim = yaml.safe_load((REFRAME_DATA / "sim_target_196A.yaml").read_text())
    sats = sim["rfi_sources"]["tle_satellite"]
    assert sats["offline"] is True
    orbit_dir = REFRAME_DATA / sats["extra_orbit_dir"]

    ids = [
        int(line.split("#")[0].split()[0])
        for line in (REFRAME_DATA / sats["norad_ids_path"]).read_text().splitlines()
        if line.split("#")[0].strip()
    ]
    obs = sim["observation"]
    start = datetime.fromisoformat(obs["start_time_isot"])
    mid = start + timedelta(seconds=obs["n_time"] * obs["int_time"] / 2)
    mid_jd = (mid - datetime(2000, 1, 1, 12)).total_seconds() / 86400 + 2451545.0

    records = read_legacy_tle_records(str(orbit_dir)).to_dict("records")
    by_id = {int(r["NORAD_CAT_ID"]): r for r in records}
    assert sorted(by_id) == sorted(ids)
    for nid, record in by_id.items():
        assert record_kind(record) == "omm", nid
        assert abs(record_epoch_jd(record) - mid_jd) < 0.25, nid

    for name in ("tab_target_analytic.yaml", "tab_target_riemann.yaml"):
        tab = yaml.safe_load((REFRAME_DATA / name).read_text())
        assert REFRAME_DATA / tab["satellites"]["extra_orbit_dir"] == orbit_dir, name
