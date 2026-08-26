"""The bundled Space-Track records must stay acceptable to the shared parser.

The parser itself is tested in the satchecker-client package; this guards the
enforcement against the data tabascal actually ships: every bundled record must
remain acceptable.
"""

from importlib.resources import files

from satchecker_client import read_legacy_tle_records
from satchecker_client.tle_parse import validate_tle_pair


def test_real_bundled_records_satisfy_both_checks():
    df = read_legacy_tle_records(str(files("tabascal").joinpath("data/tles")))
    assert len(df) > 200
    for _, row in df.iterrows():
        validate_tle_pair(row["TLE_LINE1"], row["TLE_LINE2"])
