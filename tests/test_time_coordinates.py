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
    skyfield_time,
    TIME_SCALES,
    timescale,
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
# tabascal.time — skyfield_time
# ---------------------------------------------------------------------------

class TestSkyfieldTime:
    """The single entry point from Julian Dates to skyfield times."""

    JD = 2451545.3

    def test_timescale_is_memoised(self):
        """One timescale, reused -- so every call site shares the same one."""
        assert timescale() is timescale()

    def test_matches_the_explicit_whole_fraction_split(self):
        """Equivalent to the inline ``ts._utc_jd(floor, frac)`` it replaced.

        Pins the refactor: this is the expression that was duplicated across
        time.py and the three trajectory.py call sites.
        """
        ts = timescale()
        expected = ts._utc_jd(np.floor(self.JD), self.JD - np.floor(self.JD))

        assert float(np.asarray(skyfield_time(self.JD).tt)) == float(
            np.asarray(expected.tt)
        )

    def test_preserves_the_whole_fraction_split(self):
        """The integer day is carried separately, not folded into one float."""
        t = skyfield_time(np.array([self.JD]))
        assert float(np.asarray(t.whole)[0]) == np.floor(self.JD)

    def test_fraction_is_the_exact_remainder(self):
        """Nothing is lost between the input JD and what skyfield receives.

        The split cannot recover precision the input f64 never had -- a JD of
        ~2.5e6 is already quantised to ~5e-10 days before it arrives. What it
        does guarantee is that no *further* precision is dropped on the way in:
        the fraction handed over is exactly ``jd - floor(jd)``.
        """
        jd = np.array([self.JD, self.JD + 0.25])
        t = skyfield_time(jd)

        # TT runs ahead of UTC by the leap seconds + 32.184 s, constant here, so
        # the UTC fraction is recovered by removing that same offset from both.
        offset = np.asarray(t.tt_fraction) - (jd - np.floor(jd))
        assert offset[0] == pytest.approx(offset[1], abs=1e-15)

    def test_gast_deg_is_consistent_with_it(self):
        """gast_deg reads its time through the same entry point."""
        times = np.array([self.JD, self.JD + 0.25])
        direct = np.asarray(skyfield_time(times).gast) * 15.0

        np.testing.assert_allclose(gast_deg(times), direct, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# tabascal.time — time scales
# ---------------------------------------------------------------------------

class TestTimeScale:
    """``scale`` selects how a Julian Date is interpreted, not how it prints."""

    JD = 2451545.3

    def _tt_seconds(self, scale):
        t = skyfield_time(self.JD, scale)
        return float(np.asarray(t.tt)) * 86400.0

    def test_utc_is_the_default(self):
        """Omitting the scale reads UTC, so existing callers are unchanged."""
        assert float(np.asarray(skyfield_time(self.JD).tt)) == float(
            np.asarray(skyfield_time(self.JD, "utc").tt)
        )

    @pytest.mark.parametrize("scale", sorted(TIME_SCALES))
    def test_every_supported_scale_constructs(self, scale):
        assert skyfield_time(self.JD, scale) is not None

    def test_scale_is_case_insensitive(self):
        assert self._tt_seconds("TAI") == self._tt_seconds("tai")

    def test_unsupported_scale_is_rejected(self):
        with pytest.raises(ValueError, match="Unsupported time scale 'gmst'"):
            skyfield_time(self.JD, "gmst")

    def test_reading_utc_as_tai_shifts_by_the_leap_seconds(self):
        """The whole point: the same JD on a different scale is a different instant.

        At J2000 the TAI-UTC offset was 32 leap seconds. A satellite at a typical
        LEO ground-track speed of ~7.5 km/s moves ~240 km in that time, so this
        is a wrong position rather than a rounding difference -- and nothing
        raises to say so.
        """
        offset = self._tt_seconds("utc") - self._tt_seconds("tai")

        assert offset == pytest.approx(32.0, abs=1e-6)

    def test_reading_utc_as_tt_shifts_by_leap_seconds_plus_32_184(self):
        offset = self._tt_seconds("utc") - self._tt_seconds("tt")

        assert offset == pytest.approx(32.0 + 32.184, abs=1e-3)

    def test_et_is_an_alias_for_tt(self):
        """CASA's legacy spelling of TT."""
        assert self._tt_seconds("et") == self._tt_seconds("tt")

    def test_reading_utc_as_ut1_shifts_by_dut1(self):
        """Sub-second, but still ~2.7 km of LEO track."""
        offset = abs(self._tt_seconds("utc") - self._tt_seconds("ut1"))

        assert 0.0 < offset < 0.9

    def test_gast_deg_accepts_a_scale(self):
        direct = np.asarray(skyfield_time(self.JD, "tai").gast) * 15.0

        np.testing.assert_allclose(gast_deg(self.JD, "tai"), direct, rtol=0, atol=0)

    def test_gast_deg_differs_between_scales(self):
        assert gast_deg(self.JD, "utc") != pytest.approx(
            float(gast_deg(self.JD, "tai")), abs=1e-9
        )


# ---------------------------------------------------------------------------
# skyfield private-API smoke test
# ---------------------------------------------------------------------------

def test_skyfield_utc_jd_whole_fraction_available():
    """Guard the private skyfield API that ``skyfield_time`` depends on.

    ``skyfield_time`` calls ``ts._utc_jd(whole, fraction)`` (a private method,
    used to feed UTC Julian Dates split into whole + fractional parts for full
    f64 precision), and is now the only caller in the package. The skyfield pin
    in pyproject.toml is bounded for exactly this reason; if a resolved version
    drops or changes the method, fail loudly here in CI instead of deep inside
    a run.
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
