"""The MS phase centre is read from FIELD::PHASE_DIR in its declared frame, as J2000."""

import os

import numpy as np
import pytest

from tabascal.ms import read_ms

tables = pytest.importorskip("casacore.tables")
erfa = pytest.importorskip("erfa")

RA, DEC = np.deg2rad(30.0), np.deg2rad(-30.0)  # the ICRS truth, which tabascal calls J2000
UNIX = 1568538661.0 + np.array([0.0, 8.0, 16.0])  # 2019-09-15
MID_TT = 2440587.5 + (UNIX[1] + 69.184) / 86400.0


def _write_ms(path, dirs, ref="J2000", field_id=0, source=(RA, DEC), num_poly=0, ephemeris_id=None):
    """Three antennas, times and channels; FIELD row i holds ``dirs[i]``, rows read field ``field_id``.

    ``num_poly`` and ``ephemeris_id`` are per FIELD row or one for all; ``ephemeris_id=None`` leaves
    the optional EPHEMERIS_ID column out.
    """
    polys = np.broadcast_to(num_poly, len(dirs))
    ms = tables.default_ms(path, tables.maketabdesc(
        tables.makearrcoldesc("DATA", 0j, ndim=2, shape=[3, 1], valuetype="complex")))
    a1, a2 = np.triu_indices(3, k=1)
    n = len(UNIX) * len(a1)
    ms.addrows(n)
    ms.putcol("TIME", np.repeat((UNIX / 86400.0 + 40587.0) * 86400.0, len(a1)))
    ms.putcol("ANTENNA1", np.tile(a1, len(UNIX)).astype(np.int32))
    ms.putcol("ANTENNA2", np.tile(a2, len(UNIX)).astype(np.int32))
    ms.putcol("FIELD_ID", np.full(n, field_id, np.int32))
    ms.putcol("INTERVAL", np.full(n, 8.0))
    ms.putcol("UVW", np.zeros((n, 3)))
    ms.putcol("SIGMA", np.ones((n, 1)))
    ms.putcol("WEIGHT", np.ones((n, 1)))
    ms.putcol("DATA", np.ones((n, 3, 1), complex))
    ms.putcol("FLAG", np.zeros((n, 3, 1), bool))
    for sub, cols in {
        "ANTENNA": {"POSITION": np.array([[5109360.1, 2006852.6, -3238948.1]] * 3), "DISH_DIAMETER": np.full(3, 13.5)},
        "SPECTRAL_WINDOW": {"CHAN_FREQ": np.linspace(1e9, 1.002e9, 3)[None], "CHAN_WIDTH": np.full((1, 3), 1e6),
                            "NUM_CHAN": np.array([3], np.int32)},
        "POLARIZATION": {"CORR_TYPE": np.array([[9]], np.int32), "NUM_CORR": np.array([1], np.int32)},
        "DATA_DESCRIPTION": {"SPECTRAL_WINDOW_ID": np.array([0], np.int32), "POLARIZATION_ID": np.array([0], np.int32)},
        "FIELD": {"NUM_POLY": polys.astype(np.int32)},
    }.items():
        with tables.table(os.path.join(path, sub), readonly=False, ack=False) as tb:
            tb.addrows(len(next(iter(cols.values()))))
            for name, val in cols.items():
                tb.putcol(name, val)
            if sub == "FIELD":
                for row, (d, p) in enumerate(zip(dirs, polys)):
                    tb.putcell("PHASE_DIR", row, np.repeat(np.array(d, float)[None], p + 1, axis=0))
                if ephemeris_id is not None:
                    tb.addcols(tables.makescacoldesc("EPHEMERIS_ID", 0))
                    tb.putcol("EPHEMERIS_ID", np.broadcast_to(ephemeris_id, len(dirs)).astype(np.int32))
                info = {"type": "direction", "Ref": ref} if ref else {"type": "direction"}
                tb.putcolkeyword("PHASE_DIR", "MEASINFO", info)
    with tables.default_ms_subtable("SOURCE", os.path.join(path, "SOURCE")) as src:
        src.addrows(1)
        src.putcol("DIRECTION", np.array([source]))
    ms.putkeyword("SOURCE", "Table: " + os.path.join(path, "SOURCE"))
    ms.close()
    return path


def _fk5(ra, dec):
    """ICRS to FK5 J2000 by the IAU 2006 frame bias, the oracle for the J2000 case."""
    return erfa.c2s(erfa.bp06(2451545.0, 0.0)[0] @ erfa.s2c(ra, dec))


def _declared(frame):
    """The ICRS truth as ``frame`` spells it, from pyerfa."""
    if frame == "J2000":
        return _fk5(RA, DEC)
    if frame == "B1950":
        return erfa.fk524(*_fk5(RA, DEC), 0.0, 0.0, 0.0, 0.0)[:2]
    if frame == "GALACTIC":
        return erfa.icrs2g(RA, DEC)
    if frame == "APP":
        ri, di, eo = erfa.atci13(RA, DEC, 0.0, 0.0, 0.0, 0.0, MID_TT, 0.0)
        return erfa.anp(ri - eo), di
    return RA, DEC


def _sep_arcsec(ms, ra, dec):
    return np.rad2deg(erfa.seps(np.deg2rad(float(ms["ra"])), np.deg2rad(float(ms["dec"])), ra, dec)) * 3600


@pytest.mark.parametrize("frame, tol", [("J2000", 1e-3), ("ICRS", 1e-3), ("B1950", 0.5), ("GALACTIC", 0.5), ("APP", 0.5)])
def test_declared_frame_is_converted_to_j2000(tmp_path, frame, tol):
    """FIELD::PHASE_DIR in any supported frame reads back as the J2000 direction it names."""
    declared = _declared(frame)
    ms = read_ms(_write_ms(str(tmp_path / "f.ms"), [declared], frame, source=declared))
    assert _sep_arcsec(ms, RA, DEC) < tol


@pytest.mark.parametrize("ref, num_poly, ephemeris_id, match", [
    ("AZEL", 0, None, "AZEL"), ("HADEC", 0, None, "HADEC"), ("JMEAN", 0, None, "JMEAN"), ("SUN", 0, None, "SUN"),
    ("J2000", 1, None, "NUM_POLY"), ("J2000", 0, 0, "EPHEMERIS_ID")])
def test_unsupported_phase_centre_is_refused(tmp_path, ref, num_poly, ephemeris_id, match):
    """Frames that cannot become a fixed J2000 direction, and moving centres, raise."""
    with pytest.raises(ValueError, match=match):
        read_ms(_write_ms(str(tmp_path / "f.ms"), [(RA, DEC)], ref, num_poly=num_poly, ephemeris_id=ephemeris_id))


def test_missing_ref_is_read_as_j2000_with_a_warning(tmp_path):
    with pytest.warns(UserWarning, match="PHASE_DIR"):
        ms = read_ms(_write_ms(str(tmp_path / "f.ms"), [(RA, DEC)], ref=None))
    assert _sep_arcsec(ms, RA, DEC) < 0.1


def test_phase_centre_comes_from_the_field_being_read(tmp_path):
    """Field 1's rows are read, so neither FIELD row 0 nor SOURCE row 0 is the centre."""
    other = (RA + 0.1, DEC + 0.1)
    ms = read_ms(_write_ms(str(tmp_path / "f.ms"), [other, (RA, DEC)], field_id=1, source=other))
    assert _sep_arcsec(ms, RA, DEC) < 0.1


def test_a_fixed_field_is_read_beside_a_polynomial_one(tmp_path):
    """Only the field being read is checked and shaped; EPHEMERIS_ID = -1 is fixed."""
    other = (RA + 0.1, DEC + 0.1)
    path = _write_ms(str(tmp_path / "f.ms"), [other, (RA, DEC)], "ICRS", field_id=1, source=other,
                     num_poly=[1, 0], ephemeris_id=-1)
    assert _sep_arcsec(read_ms(path), RA, DEC) < 1e-3
