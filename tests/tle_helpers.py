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

import urllib.request
from datetime import datetime

import pandas as pd
import pytest

from tabascal.satchecker.records import KIND_OMM, KIND_TLE
from tabascal.satchecker.tle_parse import parse_tle_elements, tle_checksum
from tabascal.time import datetime_to_jd, jd_to_datetime


#: Parametrisation for tests that must hold for both record formats. Use with
#: :func:`make_record`, which builds either kind for the same satellite.
both_kinds = pytest.mark.parametrize("kind", [KIND_TLE, KIND_OMM])


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


def with_checksum(line: str) -> str:
    """Return *line* with its column-69 modulo-10 checksum recomputed.

    Substituting a NORAD ID or epoch into the template invalidates the original
    checksum, and the parser rejects a bad one — as it should, since that is how a
    single-character corruption is caught. Fixtures therefore have to recompute it
    exactly as a real TLE producer would.
    """
    body = line[:68]
    return body + str(tle_checksum(body))


def make_tle(norad_id: int, epoch_jd: float) -> tuple[str, str]:
    """Return a (line1, line2) TLE pair for *norad_id* with the given UTC epoch.

    The NORAD ID and the line-1 epoch field are substituted into the ISS template
    at their fixed columns and the checksums are recomputed, so the pair validates
    and its parsed epoch round-trips to ``epoch_jd`` (to ~ms). Other elements are
    the template's and are irrelevant to the cache/precedence/age logic under test.
    """
    nid = f"{int(norad_id):05d}"
    dt = jd_to_datetime(epoch_jd)
    yy = dt.year % 100
    doy = (dt - datetime(dt.year, 1, 1)).total_seconds() / 86400.0 + 1.0
    epoch_field = f"{yy:02d}{doy:012.8f}"  # 2 + 12 = 14 columns (line1[18:32])
    line1 = "1 " + nid + _L1[7:18] + epoch_field + _L1[32:]
    line2 = "2 " + nid + _L2[7:]
    return with_checksum(line1), with_checksum(line2)


def jd(year, month, day, hour=0, minute=0, second=0) -> float:
    """UTC calendar time -> Julian Date."""
    return datetime_to_jd(datetime(year, month, day, hour, minute, second))


# ---------------------------------------------------------------------------
# OMM records
# ---------------------------------------------------------------------------

def make_omm(norad_id: int, epoch_jd: float, **overrides) -> dict:
    """Return an OMM record for *norad_id* with the given UTC epoch.

    The elements are the ones :func:`make_tle` encodes for the same satellite,
    read back through the TLE parser, so a test can build either kind for one
    satellite and expect the same trajectory out of both. The epoch is written
    exactly rather than quantised to the TLE line's 8-decimal day field, which
    is the one place the two kinds legitimately differ (by under a millisecond).

    *overrides* replace or add fields, for the malformed-record cases.
    """
    line1, line2 = make_tle(norad_id, epoch_jd)
    elements = parse_tle_elements(line1, line2)
    record = {
        "RECORD_KIND": KIND_OMM,
        "NORAD_CAT_ID": int(norad_id),
        "OBJECT_NAME": f"SAT-{norad_id}",
        "OBJECT_ID": "1998-067A",
        "EPOCH": jd_to_datetime(epoch_jd).isoformat(),
        "DATA_SOURCE": "test",
        "DATE_COLLECTED": None,
    }
    for column in (
        "INCLINATION",
        "RA_OF_ASC_NODE",
        "ECCENTRICITY",
        "ARG_OF_PERICENTER",
        "MEAN_ANOMALY",
        "MEAN_MOTION",
        "BSTAR",
    ):
        record[column] = elements[column]
    record.update(overrides)
    return record


def make_tle_record(norad_id: int, epoch_jd: float, **overrides) -> dict:
    """Return a TLE record for *norad_id*, shaped like the client's output."""
    line1, line2 = make_tle(norad_id, epoch_jd)
    record = {
        "RECORD_KIND": KIND_TLE,
        "NORAD_CAT_ID": int(norad_id),
        "OBJECT_NAME": f"SAT-{norad_id}",
        "EPOCH": jd_to_datetime(epoch_jd).isoformat(),
        "TLE_LINE1": line1,
        "TLE_LINE2": line2,
        "DATA_SOURCE": "test",
        "DATE_COLLECTED": None,
    }
    record.update(overrides)
    return record


def make_record(kind: str, norad_id: int, epoch_jd: float, **overrides) -> dict:
    """Build either kind of record, for the ``both_kinds`` parametrisation."""
    builder = make_tle_record if kind == KIND_TLE else make_omm
    return builder(norad_id, epoch_jd, **overrides)


# ---------------------------------------------------------------------------
# Normalised frames (as returned by the client / stored in snapshots)
# ---------------------------------------------------------------------------

def make_catalogue_df(pairs) -> pd.DataFrame:
    """Normalised TLE frame from ``[(norad_id, epoch_jd), ...]``.

    Shaped as the client returns it, ``RECORD_KIND`` included — a response we
    parsed ourselves knows what kind it is and says so.
    """
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
                "RECORD_KIND": KIND_TLE,
            }
        )
    return pd.DataFrame(rows)


def write_legacy_tle_file(path, pairs) -> None:
    """Write a legacy pandas-oriented ``*.json`` TLE file (Space-Track style).

    Deliberately *without* ``RECORD_KIND``: a Space-Track ``gp`` export carries
    no such field, and the promise that one can be dropped into
    ``extra_orbit_dir`` unconverted rests on the kind being inferable. Writing
    the field here would retire that path from the suite.
    """
    make_catalogue_df(pairs).drop(columns=["RECORD_KIND"]).to_json(path)


def write_legacy_omm_file(path, pairs) -> None:
    """Write a legacy pandas-oriented ``*.json`` OMM file, with no kind field."""
    make_omm_catalogue_df(pairs).drop(columns=["RECORD_KIND"]).to_json(path)


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


def make_nearest_json(pairs) -> bytes:
    import json

    return json.dumps({"orbital_data": _raw_rows(pairs)}).encode()


def _raw_omm_rows(pairs) -> list[dict]:
    """Raw ``get-nearest-omm`` rows, shaped as the live service returns them.

    Two details are copied deliberately from a real 1.7.0 response. The elements
    sit one level down in an ``orbital_elements`` object, already in OMM naming.
    And the epoch appears *twice* in different spellings: the row-level ``epoch``
    is ``"2026-08-13 03:34:14 UTC"`` — not ISO 8601, truncated to the second —
    while the nested ``EPOCH`` is ``"2026-08-13T03:34:14.082240"``. The client
    lifts the nested one; a fixture that only carried the tidy spelling would let
    a regression through.
    """
    rows = []
    for nid, ep in pairs:
        l1, l2 = make_tle(nid, ep)
        elements = parse_tle_elements(l1, l2)
        stamp = jd_to_datetime(ep)
        rows.append(
            {
                "satellite_id": int(nid),
                "satellite_name": f"SAT-{nid}",
                "data_source": "test",
                "date_collected": stamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "epoch": stamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "orbital_elements": {
                    "EPOCH": stamp.isoformat(),
                    "NORAD_CAT_ID": int(nid),
                    "OBJECT_ID": "1998-067A",
                    "OBJECT_NAME": f"SAT-{nid}",
                    "CLASSIFICATION_TYPE": "U",
                    "ELEMENT_SET_NO": 999,
                    "EPHEMERIS_TYPE": 0,
                    "MEAN_MOTION_DOT": 3.778e-05,
                    "MEAN_MOTION_DDOT": 0.0,
                    "REV_AT_EPOCH": 58058,
                    "INCLINATION": elements["INCLINATION"],
                    "RA_OF_ASC_NODE": elements["RA_OF_ASC_NODE"],
                    "ECCENTRICITY": elements["ECCENTRICITY"],
                    "ARG_OF_PERICENTER": elements["ARG_OF_PERICENTER"],
                    "MEAN_ANOMALY": elements["MEAN_ANOMALY"],
                    "MEAN_MOTION": elements["MEAN_MOTION"],
                    "BSTAR": elements["BSTAR"],
                },
            }
        )
    return rows


def make_nearest_omm_json(pairs) -> bytes:
    """A ``get-nearest-omm`` payload, in the single-element-list wrapping the
    live service uses."""
    import json

    return json.dumps(
        [{"orbital_data": _raw_omm_rows(pairs), "version": "1.7.0"}]
    ).encode()


def make_omm_catalogue_df(pairs) -> pd.DataFrame:
    """Normalised OMM frame, as the client returns it."""
    return pd.DataFrame([make_omm(nid, ep) for nid, ep in pairs])
