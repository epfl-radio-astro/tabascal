"""Tests for tabascal.time."""

from datetime import datetime

import pytest
import numpy as np

from tabascal.time import (
    secs_to_days,
    days_to_secs,
    jd_to_mjd,
    mjd_to_jd,
    gast_deg,
    jd_to_datetime,
    datetime_to_jd,
)


# ---------------------------------------------------------------------------
# tabascal.time
# ---------------------------------------------------------------------------

class TestTime:

    def test_secs_to_days_roundtrip(self):
        secs = 12345.6
        assert days_to_secs(secs_to_days(secs)) == pytest.approx(secs)

    def test_secs_to_days_scaling(self):
        assert secs_to_days(86400.0) == pytest.approx(1.0)
        assert days_to_secs(1.0) == pytest.approx(86400.0)

    def test_jd_to_mjd_roundtrip(self):
        jd = 2451545.0
        assert mjd_to_jd(jd_to_mjd(jd)) == pytest.approx(jd)

    def test_jd_to_mjd_known_value(self):
        # J2000.0 is JD 2451545.0 = MJD 51544.5
        assert jd_to_mjd(2451545.0) == pytest.approx(51544.5)

    def test_mjd_to_jd_known_value(self):
        assert mjd_to_jd(51544.5) == pytest.approx(2451545.0)

    def test_offset_is_2400000_5(self):
        jd = 2459000.0
        assert jd_to_mjd(jd) == pytest.approx(jd - 2400000.5)
        assert mjd_to_jd(jd - 2400000.5) == pytest.approx(jd)

    def test_array_inputs(self):
        jds = np.array([2451545.0, 2451546.0, 2451547.0])
        mjds = jd_to_mjd(jds)
        assert np.allclose(mjd_to_jd(mjds), jds)


# ---------------------------------------------------------------------------
# tabascal.time — jd_to_datetime / datetime_to_jd
# ---------------------------------------------------------------------------

class TestJdDatetime:

    def test_j2000_known_value(self):
        # JD 2451545.0 == 2000-01-01T12:00:00 UTC
        dt = jd_to_datetime(2451545.0)
        assert dt == datetime(2000, 1, 1, 12, 0, 0)

    def test_datetime_to_jd_known_value(self):
        assert datetime_to_jd(datetime(2000, 1, 1, 12, 0, 0)) == pytest.approx(2451545.0)

    def test_unix_epoch(self):
        assert jd_to_datetime(2440587.5) == datetime(1970, 1, 1, 0, 0, 0)

    def test_roundtrip(self):
        jd = 2461112.369018021
        assert datetime_to_jd(jd_to_datetime(jd)) == pytest.approx(jd, abs=1e-9)

    def test_isoformat_parse_roundtrip(self):
        # Mirrors the TLE EPOCH -> EPOCH_JD path in tle.py
        isot = "2026-03-12T20:51:23.157"
        jd = datetime_to_jd(datetime.fromisoformat(isot))
        assert jd == pytest.approx(2461112.369018021, abs=1e-9)

    def test_tz_aware_input_treated_as_utc(self):
        from datetime import timezone
        aware = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert datetime_to_jd(aware) == pytest.approx(2451545.0)


# ---------------------------------------------------------------------------
# tabascal.time — gast_deg
# ---------------------------------------------------------------------------

class TestGastDeg:

    def test_shape_and_range(self):
        jds = np.array([2451545.0, 2451545.5, 2451546.0])
        gast = gast_deg(jds)
        assert gast.shape == (3,)
        assert np.all((gast >= 0) & (gast < 360))

    def test_advances_with_sidereal_rate(self):
        # Over one solar day GAST advances by ~360.985 degrees (one full turn
        # plus the ~0.9856 deg/day the stars gain on the Sun), i.e. ~0.985 deg
        # net after wrapping.
        g0 = gast_deg(np.array([2451545.0]))[0]
        g1 = gast_deg(np.array([2451546.0]))[0]
        assert (g1 - g0) % 360 == pytest.approx(0.9856, abs=0.05)

    def test_is_apparent_not_mean(self):
        # GAST = GMST + equation of the equinoxes. Cross-check against skyfield's
        # own GMST: the two must differ (by up to ~18 arcsec) and agree to within
        # the equation-of-equinoxes magnitude.
        from skyfield.api import load

        jd = 2451545.3
        jd_whole = np.floor(jd)
        jd_frac = jd - jd_whole
        ts = load.timescale()
        t_sf = ts._utc_jd(jd_whole, jd_frac)
        gmst_deg = float(np.asarray(t_sf.gmst)) * 15.0

        gast = gast_deg(np.array([jd]))[0]
        diff_arcsec = abs((gast - gmst_deg) * 3600.0)
        assert diff_arcsec > 0.0          # genuinely apparent, not mean
        assert diff_arcsec < 20.0         # equation of equinoxes is < ~18 arcsec

    def test_uses_utc_not_ut1(self):
        # Interpreting the JD as UTC (not UT1) shifts GAST by the UT1-UTC offset.
        # Confirm gast_deg matches the UTC interpretation, not the UT1 one.
        from skyfield.api import load

        jd = 2451545.3
        ts = load.timescale()
        gast_ut1 = float(np.asarray(ts.ut1_jd(jd).gast)) * 15.0

        gast = gast_deg(np.array([jd]))[0]
        # The two interpretations differ at the milli-degree level (DUT1 ~ 0.9 s).
        assert gast != pytest.approx(gast_ut1, abs=1e-6)


# ---------------------------------------------------------------------------
# skyfield private-API smoke test
# ---------------------------------------------------------------------------

def test_skyfield_utc_jd_whole_fraction_available():
    """Guard the private skyfield API that gast_deg / itrs_to_gcrs_sf depend on.

    Both call ``ts._utc_jd(whole, fraction)`` (a private method, used to feed UTC
    Julian Dates split into whole + fractional parts for full f64 precision). The
    skyfield pin in pyproject.toml is bounded for exactly this reason; if a
    resolved version drops or changes the method, fail loudly here in CI instead
    of deep inside a run.
    """
    from skyfield.api import load

    ts = load.timescale()
    assert hasattr(ts, "_utc_jd"), "skyfield removed Timescale._utc_jd"

    # J2000.0 = JD 2451545.0; whole + fraction must reconstruct the same instant.
    t_split = ts._utc_jd(2451545.0, 0.0)
    t_whole = ts._utc_jd(2451544.0, 1.0)
    assert float(np.asarray(t_split.gast)) == pytest.approx(
        float(np.asarray(t_whole.gast)), abs=1e-9
    )
