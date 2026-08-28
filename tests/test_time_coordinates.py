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
    to_utc_jd,
    to_utc_mjd,
    utc_offset_days,
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

    def test_array_and_scalar_agree(self):
        """A scalar stays 0-d and a length-1 array stays 1-d.

        Callers index the result (``sf_times[0]`` in ``get_satellite_elevations``)
        or broadcast it against a time axis, so a shape quietly gained or lost on
        the way through is an error somewhere further on.
        """
        tt_array = np.asarray(skyfield_time(np.array([self.JD])).tt)
        tt_scalar = np.asarray(skyfield_time(self.JD).tt)

        assert tt_array.shape == (1,) and tt_scalar.shape == ()
        assert tt_array[0] == float(tt_scalar)

    def test_the_split_lands_on_the_unsplit_instant(self):
        """The whole/fraction split is a precision trick, not an epoch shift.

        It must name exactly the instant the unsplit Julian Date does -- to well
        inside the ~40 us the unsplit form can resolve in the first place.
        """
        jd = 2460574.123456789
        split = float(np.asarray(skyfield_time(jd).tt))
        unsplit = float(np.asarray(timescale()._utc_jd(jd, 0.0).tt))

        assert (split - unsplit) * 86400.0 == pytest.approx(0.0, abs=1e-4)


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
        with pytest.raises(ValueError, match="Unsupported time scale 'nonsense'"):
            skyfield_time(self.JD, "nonsense")

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

    def test_matches_astropy_utc(self):
        """The one external anchor that ``scale="utc"`` really is UTC.

        Every other check here is skyfield against skyfield, which would agree
        with itself just as happily on the wrong scale. astropy knows the leap
        seconds independently, so TT - UTC coming out as those leap seconds plus
        32.184 s ties the default to the scale it claims to be.
        """
        atime = pytest.importorskip("astropy.time")

        jds = np.array([2451545.3, 2460574.9])
        from_skyfield = np.asarray(skyfield_time(jds).tt)
        from_astropy = atime.Time(jds, format="jd", scale="utc").tt.jd

        assert from_skyfield == pytest.approx(from_astropy, abs=1e-9)  # < 0.1 ms


class TestToUtcJd:
    """Moving a declared scale onto UTC, once, where the times are read.

    Everything downstream reads a Julian Date as UTC -- skyfield through
    ``skyfield_time``'s default, ``sgp4jax.itrf_to_gcrf``, which has no scale
    concept to be told otherwise, and the TLE epoch checks. Normalising at the
    boundary puts all of them on the instant the MS actually names, without a
    ``scale`` argument threaded through any of them.
    """

    #: 2025-01-01T00:00:00 UTC, when TAI - UTC was 37 s.
    JD = 2460676.5

    #: 2017-01-01T00:00:00 UTC: the most recent leap second, 36 s -> 37 s.
    LEAP_JD = 2457754.5

    def _shift_secs(self, jd, scale):
        """The shift measured end to end, off the returned Julian Dates.

        Differencing two ~2.5e6 Julian Dates cannot resolve better than the ~40 us
        f64 holds there, so the tolerances below are 0.1 ms. The exact offset is
        checked against :func:`utc_offset_days`, which never leaves the fraction.
        """

        return (to_utc_jd(jd, scale) - np.asarray(jd, dtype=float)) * 86400.0

    def test_utc_is_left_exactly_alone(self):
        """Bit-identical, not merely close.

        The common case is a UTC-declared MS, and it must read exactly as it did
        before the scale was honoured -- so it goes through no arithmetic at all.
        """
        jd = self.JD + np.arange(4) * 8.0 / 86400.0

        np.testing.assert_array_equal(to_utc_jd(jd, "utc"), jd)

    def test_utc_is_the_default(self):
        jd = self.JD + np.arange(3) * 8.0 / 86400.0

        np.testing.assert_array_equal(to_utc_jd(jd), jd)

    def test_tai_moves_back_by_the_leap_seconds(self):
        """A TAI reading names an instant 37 s before the same number read as UTC."""
        assert self._shift_secs(self.JD, "tai") == pytest.approx(-37.0, abs=1e-4)
        assert utc_offset_days(self.JD, "tai") * 86400.0 == pytest.approx(
            -37.0, abs=1e-9
        )

    def test_tt_moves_back_by_the_leap_seconds_plus_32_184(self):
        assert self._shift_secs(self.JD, "tt") == pytest.approx(-69.184, abs=1e-3)

    def test_ut1_moves_back_by_dut1(self):
        """Sub-second, and still ~2.7 km of LEO ground track."""
        assert 0.0 < abs(self._shift_secs(self.JD, "ut1")) < 0.9

    @pytest.mark.parametrize("scale", sorted(TIME_SCALES))
    def test_the_result_names_the_original_instant(self, scale):
        """The whole point: read back as UTC, it is the instant the MS declared."""
        declared = float(np.asarray(skyfield_time(self.JD, scale).tt))
        normalised = float(np.asarray(skyfield_time(to_utc_jd(self.JD, scale)).tt))

        assert (normalised - declared) * 86400.0 == pytest.approx(0.0, abs=1e-4)

    def test_a_leap_second_inside_the_observation_is_followed(self):
        """The offset is per sample, not one constant for the array.

        An observation straddling a leap second is offset by 36 s on one side of
        it and 37 s on the other; a single offset for the array would put half
        the samples a second wrong.
        """
        jds = self.LEAP_JD + np.array([-0.5, 0.5])
        offset_secs = utc_offset_days(jds, "tai") * 86400.0

        assert offset_secs == pytest.approx([-36.0, -37.0], abs=1e-9)
        assert self._shift_secs(jds, "tai") == pytest.approx([-36.0, -37.0], abs=1e-4)

    def test_the_shift_is_right_to_the_julian_date_quantum(self):
        """What the returned value is worth, not how it is arrived at.

        A Julian Date near 2.5e6 is spaced ~40 us apart in f64, so that is the
        floor on any single returned JD and no arrangement of the arithmetic
        beats it. What the day-fraction route buys is that the conversion costs
        nothing *beyond* that floor: the returned shift is the leap seconds to
        within one representable step, where recomputing the date from a
        skyfield accessor would spend the floor twice over.
        """
        jd = self.JD + np.arange(4) * 8.0 / 86400.0

        np.testing.assert_allclose(
            to_utc_jd(jd, "tai"),
            jd - 37.0 / 86400.0,
            rtol=0,
            atol=float(np.spacing(jd).max()),  # one representable step, ~40 us
        )

    def test_the_offset_itself_never_leaves_the_fraction(self):
        """The offset is picosecond-clean even though the JD it lands on is not.

        It is a difference of O(1) day fractions, so it carries far more digits
        than the ~2.5e6-magnitude sum it is then added to. That is the half of
        the calculation that is worth being exact, and it is.
        """
        jd = self.JD + np.arange(4) * 8.0 / 86400.0
        offset_secs = utc_offset_days(jd, "tai") * 86400.0

        np.testing.assert_allclose(offset_secs, -37.0, rtol=0, atol=1e-9)

    def test_the_offset_is_zero_on_utc(self):
        assert utc_offset_days(self.JD, "utc") == 0.0

    def test_shapes_are_preserved(self):
        assert np.asarray(to_utc_jd(self.JD, "tai")).shape == ()
        assert to_utc_jd(np.array([self.JD]), "tai").shape == (1,)

    def test_scale_is_case_insensitive(self):
        assert to_utc_jd(self.JD, "TAI") == to_utc_jd(self.JD, "tai")

    def test_an_unsupported_scale_is_rejected(self):
        """Rejected here as it is in ``skyfield_time``, and for the same reason."""
        with pytest.raises(ValueError, match="Unsupported time scale 'nonsense'"):
            to_utc_jd(self.JD, "nonsense")

    def test_a_sidereal_reference_says_it_is_not_a_scale(self):
        with pytest.raises(ValueError, match="valid Measurement Set epoch reference"):
            to_utc_jd(self.JD, "gast")


class TestToUtcMjd:
    """The MJD counterpart, for day numbers that are compared to another source's.

    A Modified Julian Date is a number until a scale says what it counts, so an
    MJD that leaves an observation -- into a light-curve file, say -- has to name
    a scale for anything else to sample it at the right instant. UTC is the one
    tabascal states, and this is what puts a declared column on it.
    """

    #: 2025-01-01T00:00:00 UTC, when TAI - UTC was 37 s.
    MJD = 60676.0

    #: 2017-01-01T00:00:00 UTC: the most recent leap second, 36 s -> 37 s.
    LEAP_MJD = 57754.0

    def _times(self, n_time=4, step=8.0):
        return self.MJD + np.arange(n_time) * step / 86400.0

    @staticmethod
    def _shift_secs(mjd):
        """The shift measured end to end, off the returned day numbers."""

        return (to_utc_mjd(mjd, "tai") - np.asarray(mjd, dtype=float)) * 86400.0

    @staticmethod
    def _quantum_secs(mjd):
        """One representable step at that magnitude, in seconds (~0.6 us).

        The floor on any single returned MJD, and so on any shift differenced
        back out of one. The offset itself is checked against
        :func:`utc_offset_days`, which never leaves the day fraction.
        """

        return float(np.spacing(np.asarray(mjd, dtype=float)).max()) * 86400.0

    def test_utc_is_left_exactly_alone(self):
        """Bit-identical, not merely close.

        The overwhelmingly common case is a UTC-declared MS, whose light curves
        must be sampled at exactly the coordinates they were before the scale was
        honoured -- so it goes through no arithmetic at all.
        """
        mjd = self._times()

        np.testing.assert_array_equal(to_utc_mjd(mjd, "utc"), mjd)

    def test_utc_is_the_default(self):
        mjd = self._times()

        np.testing.assert_array_equal(to_utc_mjd(mjd), mjd)

    def test_tai_moves_back_by_the_leap_seconds(self):
        """The same numbers read as TAI name an instant 37 s earlier."""
        mjd = self._times()

        np.testing.assert_allclose(
            self._shift_secs(mjd), -37.0, rtol=0, atol=self._quantum_secs(mjd)
        )

    def test_tt_moves_back_by_the_leap_seconds_plus_32_184(self):
        shift = (to_utc_mjd(self.MJD, "tt") - self.MJD) * 86400.0

        assert shift == pytest.approx(-69.184, abs=self._quantum_secs(self.MJD))

    def test_it_names_the_same_instant_as_the_julian_date_route(self):
        """Same answer as ``to_utc_jd``, to the ~40 us that route resolves to."""
        mjd = self._times()
        via_jd = jd_to_mjd(to_utc_jd(mjd_to_jd(mjd), "tai"))

        np.testing.assert_allclose(to_utc_mjd(mjd, "tai"), via_jd, rtol=0, atol=1e-9)

    def test_it_keeps_the_digits_the_julian_date_route_spends(self):
        """Why it converts at MJD magnitude instead of routing through a JD.

        An MJD near 6e4 is spaced sub-microsecond (~0.6 us) apart in f64; a JD
        near 2.5e6 is spaced ~40 us apart. Going out to a JD and back rounds twice at the
        coarser magnitude, and on a UTC MS that is a shift applied to a column
        that needed no conversion at all.
        """
        mjd = self.MJD + np.arange(4) * 8.5 / 86400.0
        round_tripped = jd_to_mjd(mjd_to_jd(mjd))

        # The route not taken really does lose digits, so there is something here
        # to keep...
        assert np.any(round_tripped != mjd)
        # ...and the one taken keeps them.
        np.testing.assert_array_equal(to_utc_mjd(mjd, "utc"), mjd)

    def test_a_leap_second_inside_the_observation_is_followed(self):
        """Per sample, as ``utc_offset_days`` is: 36 s one side of it, 37 s the other."""
        mjd = self.LEAP_MJD + np.array([-0.5, 0.5])

        assert self._shift_secs(mjd) == pytest.approx(
            [-36.0, -37.0], abs=self._quantum_secs(mjd)
        )

    def test_shapes_are_preserved(self):
        assert np.asarray(to_utc_mjd(self.MJD, "tai")).shape == ()
        assert to_utc_mjd(np.array([self.MJD]), "tai").shape == (1,)

    def test_scale_is_case_insensitive(self):
        assert to_utc_mjd(self.MJD, "TAI") == to_utc_mjd(self.MJD, "tai")

    def test_an_unsupported_scale_is_rejected(self):
        with pytest.raises(ValueError, match="Unsupported time scale 'nonsense'"):
            to_utc_mjd(self.MJD, "nonsense")

    def test_a_sidereal_reference_says_it_is_not_a_scale(self):
        with pytest.raises(ValueError, match="valid Measurement Set epoch reference"):
            to_utc_mjd(self.MJD, "gast")


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


class TestCasacoreScaleNames:
    """casacore names several scales more than once; all spellings are accepted.

    The point of ``scale`` is to forward whatever an MS declares, so rejecting
    the spelling casacore actually writes would defeat it. Verified against
    casacore's own epoch code list, which reports:

        LAST LMST GMST1 GAST UT1 UT2 UTC TAI TDT TCG TDB TCB IAT GMST TT ET UT

    with TDT listed before its TT/ET synonyms, IAT alongside TAI, and UT
    alongside UT1.
    """

    JD = 2451545.3

    def _tt(self, scale):
        return float(np.asarray(skyfield_time(self.JD, scale).tt))

    @pytest.mark.parametrize("alias", ["tdt", "tt", "et"])
    def test_terrestrial_time_spellings_agree(self, alias):
        """TDT is casacore's canonical name; TT and ET are its synonyms."""
        assert self._tt(alias) == self._tt("tt")

    def test_tdt_is_accepted(self):
        """The canonical spelling an MS is most likely to carry."""
        assert "tdt" in TIME_SCALES

    @pytest.mark.parametrize("alias", ["tai", "iat"])
    def test_atomic_time_spellings_agree(self, alias):
        assert self._tt(alias) == self._tt("tai")

    @pytest.mark.parametrize("alias", ["ut1", "ut"])
    def test_universal_time_spellings_agree(self, alias):
        assert self._tt(alias) == self._tt("ut1")

    @pytest.mark.parametrize(
        "scale", ["gast", "gmst1", "gmst", "last", "lmst", "ut2", "tcg", "tcb"]
    )
    def test_scales_we_cannot_interpret_say_so(self, scale):
        """Rejected as unsupported, not as a typo.

        These are real MS epoch references -- sidereal angles, and relativistic
        scales skyfield offers no constructor for -- so the error should not
        imply the name is wrong.
        """
        with pytest.raises(ValueError, match="valid Measurement Set epoch reference"):
            skyfield_time(self.JD, scale)

    def test_a_genuine_typo_still_reads_as_one(self):
        with pytest.raises(ValueError, match="Unsupported time scale 'utcc'"):
            skyfield_time(self.JD, "utcc")
