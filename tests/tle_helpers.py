"""Shared builders for the offline SatChecker / TLE tests.

Everything here is synthetic and offline: TLE lines are derived from a real ISS
template (so they parse through :func:`tabascal.tle.parse_tle_elements`), with
only the NORAD ID and epoch varied. No network access is involved.

:func:`block_network` enforces that. Import it into a test module and it becomes
an autouse fixture there, so a test that forgets to stub the transport fails
loudly instead of quietly querying the live SatChecker service (which would make
the suite slow, flaky, and dependent on what a third party happens to serve).
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from datetime import datetime

import pandas as pd
import pytest

from tabascal.time import datetime_to_jd, jd_to_datetime


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """Fail any test that reaches the network instead of stubbing the transport.

    Tests that need to exercise the real ``_http_get`` wrapping patch
    ``urlopen`` themselves; a later ``monkeypatch.setattr`` simply replaces this
    one, so those keep working.
    """
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "this test attempted a live network request; stub "
            "tabascal.satchecker.client._http_get (or urlopen) instead"
        )

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)

# Real public ISS TLE (epoch 2008-264); used only as a fixed-width template.
_L1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
_L2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"


def make_tle(norad_id: int, epoch_jd: float) -> tuple[str, str]:
    """Return a (line1, line2) TLE pair for *norad_id* with the given UTC epoch.

    The NORAD ID and the line-1 epoch field are substituted into the ISS template
    at their fixed columns, so the pair parses and its parsed epoch round-trips to
    ``epoch_jd`` (to ~ms). Other elements are the template's and are irrelevant to
    the cache/precedence/age logic under test.
    """
    nid = f"{int(norad_id):05d}"
    dt = jd_to_datetime(epoch_jd)
    yy = dt.year % 100
    doy = (dt - datetime(dt.year, 1, 1)).total_seconds() / 86400.0 + 1.0
    epoch_field = f"{yy:02d}{doy:012.8f}"  # 2 + 12 = 14 columns (line1[18:32])
    line1 = "1 " + nid + _L1[7:18] + epoch_field + _L1[32:]
    line2 = "2 " + nid + _L2[7:]
    return line1, line2


def jd(year, month, day, hour=0, minute=0, second=0) -> float:
    """UTC calendar time -> Julian Date."""
    return datetime_to_jd(datetime(year, month, day, hour, minute, second))


# ---------------------------------------------------------------------------
# Normalised frames (as returned by the client / stored in snapshots)
# ---------------------------------------------------------------------------

def make_catalogue_df(pairs) -> pd.DataFrame:
    """Normalised catalogue frame from ``[(norad_id, epoch_jd), ...]``."""
    rows = []
    for nid, ep in pairs:
        l1, l2 = make_tle(nid, ep)
        rows.append(
            {
                "NORAD_CAT_ID": int(nid),
                "OBJECT_NAME": f"SAT-{nid}",
                "EPOCH": jd_to_datetime(ep).isoformat(),
                "TLE_LINE1": l1,
                "TLE_LINE2": l2,
                "DATA_SOURCE": "test",
                "DATE_COLLECTED": None,
            }
        )
    return pd.DataFrame(rows)


def write_legacy_tle_file(path, pairs) -> None:
    """Write a legacy pandas-oriented ``*.json`` TLE file (Space-Track style)."""
    make_catalogue_df(pairs).to_json(path)


# ---------------------------------------------------------------------------
# Raw SatChecker HTTP payloads (un-normalised field names)
# ---------------------------------------------------------------------------

def _raw_rows(pairs) -> list[dict]:
    rows = []
    for nid, ep in pairs:
        l1, l2 = make_tle(nid, ep)
        rows.append(
            {
                "satellite_id": int(nid),
                "satellite_name": f"SAT-{nid}",
                "epoch": jd_to_datetime(ep).isoformat(),
                "tle_line1": l1,
                "tle_line2": l2,
                "data_source": "test",
                "date_collected": None,
            }
        )
    return rows


def make_zip_bytes(pairs) -> bytes:
    """A ``format=zip`` response: a single CSV of raw records inside a zip."""
    df = pd.DataFrame(_raw_rows(pairs))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("tles.csv", df.to_csv(index=False))
    return buf.getvalue()


def make_empty_zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w"):
        pass
    return buf.getvalue()


def make_info_json(total: int, version: str | None = "1.6.0") -> bytes:
    payload = {"total_results": total}
    if version is not None:
        payload["version"] = version
    import json

    return json.dumps(payload).encode()


def make_json_page(pairs, total: int) -> bytes:
    import json

    return json.dumps({"total_results": total, "data": _raw_rows(pairs)}).encode()


def make_nearest_json(pairs) -> bytes:
    import json

    return json.dumps({"orbital_data": _raw_rows(pairs)}).encode()
