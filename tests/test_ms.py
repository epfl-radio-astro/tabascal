"""Tests for tabascal.ms — row layout, correlation resolution, time scales."""

from datetime import datetime, timedelta

import numpy as np
import pytest

from tabascal.ms import (
    CORR_TYPES,
    read_ms,
    DEFAULT_TIME_SCALE,
    fitted_correlation,
    grid_to_rows,
    into_corr,
    ms_layout,
    partition_noise,
    partition_polarization,
    partition_setup,
    read_ms,
    read_time_scale,
    read_time_unit,
    rows_to_grid,
    resolve_correlation,
    resolve_data_description,
    times_to_mjd,
)
from tabascal.time import mjd_to_jd


class _FakePol:
    """One grouped POLARIZATION row.

    dask-ms with ``group_cols="__row__"`` yields one dataset per row, each
    keeping a leading row axis of length 1 -- so CORR_TYPE is (1, n_corr) and
    rows of differing width never have to share a shape.
    """

    def __init__(self, corr_type):
        self.CORR_TYPE = _FakeVar(np.asarray([corr_type]))


class _FakeVar:
    def __init__(self, values):
        self.data = values if hasattr(values, "compute") else _FakeData(values)


class _FakeData:
    def __init__(self, values):
        self._values = values

    def __getitem__(self, idx):
        return _FakeData(self._values[idx])

    def compute(self):
        return self._values


@pytest.fixture
def polarization(monkeypatch):
    """Patch the POLARIZATION read with a chosen CORR_TYPE row."""

    def _install(*rows):
        def fake_xds_from_table(path, group_cols=None):
            assert path.endswith("::POLARIZATION")
            assert group_cols == "__row__", "rows must be grouped, see variable shapes"
            return [_FakePol(r) for r in rows]

        monkeypatch.setattr("tabascal.ms.xds_from_table", fake_xds_from_table)

    return _install


# ---------------------------------------------------------------------------
# The case this exists for: a single-correlation MS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("corr", ["xx", "yy", "ll", "i"])
def test_single_correlation_ms_resolves_to_index_zero(polarization, corr):
    """An MS holding one polarisation has it at index 0, whatever it is.

    The conventional {xx: 0, ..., yy: 3} table returns 3 for 'yy', which is off
    the end of a length-1 correlation axis.
    """
    polarization([CORR_TYPES[corr]])

    assert resolve_correlation("fake.ms", corr) == 0


def test_single_correlation_ms_rejects_a_correlation_it_does_not_hold(polarization):
    """Asking for XX on a YY-only MS is an error, not a silent read of YY."""
    polarization([CORR_TYPES["yy"]])

    with pytest.raises(ValueError, match="does not contain correlation 'xx'"):
        resolve_correlation("fake.ms", "xx")


def test_the_error_names_what_the_ms_actually_holds(polarization):
    polarization([CORR_TYPES["yy"]])

    with pytest.raises(ValueError, match="It holds: yy"):
        resolve_correlation("fake.ms", "xx")


# ---------------------------------------------------------------------------
# The other layouts the positional table gets wrong
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "corr, expected", [("xx", 0), ("xy", 1), ("yx", 2), ("yy", 3)]
)
def test_full_four_correlation_ms_matches_the_conventional_order(
    polarization, corr, expected
):
    """The case the old positional table handled; unchanged."""
    polarization([CORR_TYPES[c] for c in ("xx", "xy", "yx", "yy")])

    assert resolve_correlation("fake.ms", corr) == expected


@pytest.mark.parametrize("corr, expected", [("xx", 0), ("yy", 1)])
def test_two_correlation_ms(polarization, corr, expected):
    """An (XX, YY) MS puts YY at index 1, where the positional table says 3."""
    polarization([CORR_TYPES["xx"], CORR_TYPES["yy"]])

    assert resolve_correlation("fake.ms", corr) == expected


def test_non_conventional_ordering_is_followed(polarization):
    """The axis order is read, not assumed."""
    polarization([CORR_TYPES["yy"], CORR_TYPES["xx"]])

    assert resolve_correlation("fake.ms", "xx") == 1
    assert resolve_correlation("fake.ms", "yy") == 0


def test_circular_correlations(polarization):
    polarization([CORR_TYPES[c] for c in ("rr", "rl", "lr", "ll")])

    assert resolve_correlation("fake.ms", "ll") == 3


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def test_correlation_name_is_case_insensitive(polarization):
    polarization([CORR_TYPES["yy"]])

    assert resolve_correlation("fake.ms", "YY") == 0


def test_unknown_correlation_name_is_rejected(polarization):
    polarization([CORR_TYPES["xx"]])

    with pytest.raises(ValueError, match="Unknown correlation 'zz'"):
        resolve_correlation("fake.ms", "zz")


def test_unreadable_polarization_raises(monkeypatch):
    """No positional fallback: guessing is the bug this function removes.

    POLARIZATION is a required subtable. Falling back to the conventional
    {xx: 0, ..., yy: 3} ordering would return 3 for a single-correlation MS --
    off the end of its axis, and exactly the failure #128 is about.
    """

    def broken(path, group_cols=None):
        raise RuntimeError("no such table")

    monkeypatch.setattr("tabascal.ms.xds_from_table", broken)

    with pytest.raises(ValueError, match="correlation layout is unknown"):
        resolve_correlation("fake.ms", "yy")


# ---------------------------------------------------------------------------
# Time scale, from the TIME column's MEASINFO record
# ---------------------------------------------------------------------------

def _keywords(ref=None, column="TIME", units=("s",)):
    """Column keywords as dask-ms returns them, optionally declaring a scale.

    ``units`` is the ``QuantumUnits`` declaration; ``None`` leaves it off, which
    is what a column written without one looks like.
    """

    measinfo = {"type": "epoch"}
    if ref is not None:
        measinfo["Ref"] = ref

    keywords = {column: {"MEASINFO": measinfo}}
    if units is not None:
        keywords[column]["QuantumUnits"] = units

    return keywords


def test_reads_the_declared_scale():
    assert read_time_scale(_keywords("UTC")) == "utc"


@pytest.mark.parametrize("ref", ["UTC", "TAI", "TT", "UT1", "TDB"])
def test_every_declared_scale_is_returned_lowercased(ref):
    assert read_time_scale(_keywords(ref)) == ref.lower()


def test_a_non_utc_scale_is_reported_as_declared():
    """Not silently coerced to UTC -- the caller decides what to do about it."""
    assert read_time_scale(_keywords("TAI")) == "tai"


def test_missing_measinfo_ref_falls_back_with_a_warning():
    with pytest.warns(UserWarning, match="no MEASINFO Ref"):
        assert read_time_scale(_keywords(None)) == DEFAULT_TIME_SCALE


def test_missing_column_falls_back_with_a_warning():
    with pytest.warns(UserWarning, match="no MEASINFO Ref"):
        assert read_time_scale({}) == DEFAULT_TIME_SCALE


def test_none_keywords_fall_back():
    with pytest.warns(UserWarning, match="no MEASINFO Ref"):
        assert read_time_scale(None) == DEFAULT_TIME_SCALE


def test_a_different_column_can_be_read():
    keywords = _keywords("TAI", column="TIME_CENTROID")

    assert read_time_scale(keywords, column="TIME_CENTROID") == "tai"


# ---------------------------------------------------------------------------
# Time unit: what the numbers in a TIME column mean
# ---------------------------------------------------------------------------

def _in_memory_ms(times, n_ant=3, n_freq=2, int_time=8.0):
    """A whole MS partition and its subtables, in memory.

    ``read_ms`` reaches casacore only through ``xds_from_ms``/``xds_from_table``,
    so standing those two up covers the reader end to end without an MS on disk.
    Row-major and dask-backed, as a real partition is.
    """

    import dask.array as da
    import xarray as xr

    a1_bl, a2_bl = np.triu_indices(n_ant, k=1)
    n_bl = len(a1_bl)
    n_time = len(times)
    n_row = n_time * n_bl

    column = lambda values, dims: (dims, da.from_array(np.asarray(values)))
    row_shape = lambda *rest: (n_row, *rest)

    partition = xr.Dataset(
        data_vars={
            "TIME": column(np.repeat(np.asarray(times, float), n_bl), ["row"]),
            "ANTENNA1": column(np.tile(a1_bl, n_time), ["row"]),
            "ANTENNA2": column(np.tile(a2_bl, n_time), ["row"]),
            "INTERVAL": column(np.full(n_row, int_time), ["row"]),
            "UVW": column(np.zeros(row_shape(3)), ["row", "uvw"]),
            "DATA": column(
                np.ones(row_shape(n_freq, 1), dtype=complex), ["row", "chan", "corr"]
            ),
            "FLAG": column(
                np.zeros(row_shape(n_freq, 1), bool), ["row", "chan", "corr"]
            ),
            "SIGMA": column(np.ones(row_shape(1)), ["row", "corr"]),
        }
    )

    subtables = {
        "ANTENNA": xr.Dataset(
            data_vars={
                "POSITION": column(np.zeros((n_ant, 3)), ["row", "xyz"]),
                "DISH_DIAMETER": column(np.full(n_ant, 35.0), ["row"]),
            }
        ),
        # Grouped by row, so the leading axis of length 1 that read_ms indexes.
        "SPECTRAL_WINDOW": xr.Dataset(
            data_vars={
                "CHAN_FREQ": column(
                    np.linspace(1.0e9, 1.1e9, n_freq)[None], ["row", "chan"]
                ),
                "CHAN_WIDTH": column(np.full((1, n_freq), 1.0e6), ["row", "chan"]),
            }
        ),
        "SOURCE": xr.Dataset(
            data_vars={"DIRECTION": column([[0.35, -0.5]], ["row", "radec"])}
        ),
        "DATA_DESCRIPTION": xr.Dataset(
            data_vars={
                "SPECTRAL_WINDOW_ID": column([0], ["row"]),
                "POLARIZATION_ID": column([0], ["row"]),
            }
        ),
        "POLARIZATION": xr.Dataset(
            data_vars={"CORR_TYPE": column([[CORR_TYPES["xx"]]], ["row", "corr"])}
        ),
    }

    return partition, subtables


@pytest.fixture
def run_reader(monkeypatch):
    """Run ``read_ms`` over an in-memory MS holding the given times."""

    def _run(times, keywords=None, **kwargs):
        partition, subtables = _in_memory_ms(times, **kwargs)

        # Declares its scale and not its unit. The unit is what the default
        # cases are about, so it stays undeclared; leaving the *scale*
        # undeclared as well would only add the reader's fallback warning to
        # every one of them.
        if keywords is None:
            keywords = _keywords("UTC", units=None)

        monkeypatch.setattr(
            "tabascal.ms.xds_from_ms",
            lambda path, column_keywords=False: ([partition], keywords),
        )
        monkeypatch.setattr(
            "tabascal.ms.xds_from_table",
            lambda path, group_cols=None: [subtables[path.split("::")[-1]]],
        )

        return read_ms("in-memory.ms")

    return _run


class TestTimeUnits:
    """The one rule for reading a TIME column as MJD days.

    ``read_ms`` and the epoch helper in ``orbit_config`` both go through
    :func:`times_to_mjd`, so an MS cannot be read on one unit by the preflight
    check and on another by the run itself.
    """

    #: 2025-01-01T00:00:00 UTC, the era the array cases are built around.
    MJD = 60676.0

    #: 1965-03-04T06:00:00 UTC: before the Unix epoch, still a positive MJD.
    MJD_1965 = 38823.25

    #: 1800-01-01T00:00:00 UTC: before the MJD epoch, so a negative day number.
    MJD_1800 = -21504.0

    def _days(self, mjd=None, n_time=4, step=8.0):
        """``n_time`` integrations ``step`` seconds apart, in MJD days."""

        return (self.MJD if mjd is None else mjd) + np.arange(n_time) * step / 86400.0

    # -- the magnitude rule, on a normal multi-integration MS ---------------

    def test_seconds_are_converted_to_days(self):
        days = self._days()

        np.testing.assert_allclose(times_to_mjd(days * 86400.0), days, rtol=1e-15)

    def test_days_are_left_alone(self):
        days = self._days()

        np.testing.assert_array_equal(times_to_mjd(days), days)

    def test_a_pre_1858_epoch_is_read_in_either_unit(self):
        """Its MJD day number is negative; the magnitude the rule reads is not."""
        days = self._days(mjd=self.MJD_1800)

        np.testing.assert_allclose(times_to_mjd(days * 86400.0), days, rtol=1e-15)
        np.testing.assert_array_equal(times_to_mjd(days), days)

    def test_a_long_integration_is_still_seconds(self):
        """An hour-long sample: the cadence has no say, only the magnitude."""
        days = self._days(step=3600.0)

        np.testing.assert_allclose(times_to_mjd(days * 86400.0), days, rtol=1e-15)

    def test_two_integrations_are_read_like_any_other_column(self):
        """One rule for every length of column; nothing special happens here."""
        days = self._days(n_time=2)

        np.testing.assert_allclose(times_to_mjd(days * 86400.0), days, rtol=1e-15)
        np.testing.assert_array_equal(times_to_mjd(days), days)

    # -- the cases a spacing rule got wrong, in both directions -------------

    @pytest.mark.parametrize("step", [0.5, 0.25, 0.008])
    def test_a_short_integration_is_still_seconds(self, step):
        """Issue #208: 0.5 s is a common correlator dump time, not a half day.

        A spacing at or below 0.5 used to read as days outright, so an MS with
        0.5 s integrations came back unconverted -- ``TIME`` values of ~5.16e9
        taken for day numbers, which overflowed ``timedelta`` in the preflight
        TLE epoch check. Every integration shorter than 0.5 s was misread the
        same way.
        """
        days = self._days(step=step)

        np.testing.assert_allclose(times_to_mjd(days * 86400.0), days, rtol=1e-15)

    @pytest.mark.parametrize("cadence", [0.5, 1.0, 7.0])
    def test_a_coarse_cadence_in_days_is_still_days(self, cadence):
        """The other direction, and the reason the spacing rule went entirely.

        A day-numbered column stepping by half a day has the same spacing as a
        0.5 s integration in seconds, so no spacing test can part those two. A
        column stepping by a day or a week -- nights concatenated -- steps well
        past the old half-day threshold and was read as seconds outright: MJD
        60676 came back as MJD 0.7, wrong by the whole of the common era.
        Magnitude parts all of them: ~6e4 against ~5e9.
        """
        days = self.MJD + np.arange(3) * cadence

        np.testing.assert_array_equal(times_to_mjd(days), days)

    @pytest.mark.parametrize("cadence", [0.5, 1.0, 7.0])
    def test_a_coarse_cadence_in_seconds_is_still_seconds(self, cadence):
        """The same columns in the other unit. A guard, not a discriminator.

        The old spacing rule read these as seconds too, so this passes either
        side of the change; it is here so that the pairing above cannot be
        satisfied by a rule that simply calls everything days.
        """
        days = self.MJD + np.arange(3) * cadence

        np.testing.assert_allclose(times_to_mjd(days * 86400.0), days, rtol=1e-15)

    # -- one entry does not decide a column ---------------------------------

    def test_an_unfilled_row_does_not_turn_seconds_into_days(self):
        """casacore leaves TIME at zero in a row added and never filled.

        Reading the *smallest* magnitude, that row alone would have read a
        column of seconds as days -- issue #208's overflow again, by another
        door. The median needs half the column before it moves.
        """
        seconds = self._days(n_time=8, step=0.5) * 86400.0
        with_unfilled_row = np.concatenate([[0.0], seconds])

        converted = times_to_mjd(with_unfilled_row)

        np.testing.assert_allclose(converted[1:], seconds / 86400.0, rtol=1e-15)

    @pytest.mark.parametrize("sentinel", [np.inf, -np.inf, np.nan])
    def test_a_non_finite_entry_does_not_decide_the_unit(self, sentinel):
        """Dropped rather than ranked, so it reaches neither end of the order."""
        days = self._days(n_time=8)
        with_sentinel = np.concatenate([[sentinel], days])

        np.testing.assert_array_equal(times_to_mjd(with_sentinel)[1:], days)
        np.testing.assert_allclose(
            times_to_mjd(np.concatenate([[sentinel], days * 86400.0]))[1:],
            days,
            rtol=1e-15,
        )

    def test_a_column_of_nothing_but_sentinels_reads_as_days(self):
        """Garbage in: left as it arrived rather than scaled by 86400."""
        all_nan = np.full(4, np.nan)

        assert np.isnan(times_to_mjd(all_nan)).all()

    def test_the_rule_does_not_depend_on_row_order(self):
        """``ms_layout`` permits timestep blocks that do not ascend."""
        days = self._days()[::-1]

        np.testing.assert_allclose(times_to_mjd(days * 86400.0), days, rtol=1e-15)
        np.testing.assert_array_equal(times_to_mjd(days), days)

    # -- a single integration, which is no different -----------------------

    @pytest.mark.parametrize("mjd_name", ["MJD", "MJD_1965", "MJD_1800"])
    def test_single_integration_in_seconds(self, mjd_name):
        mjd = getattr(self, mjd_name)

        np.testing.assert_allclose(
            times_to_mjd(np.array([mjd * 86400.0])), [mjd], rtol=1e-15
        )

    @pytest.mark.parametrize("mjd_name", ["MJD", "MJD_1965", "MJD_1800"])
    def test_single_integration_in_days(self, mjd_name):
        mjd = getattr(self, mjd_name)

        np.testing.assert_array_equal(times_to_mjd(np.array([mjd])), [mjd])

    @pytest.mark.parametrize("sign", [1, -1])
    def test_the_magnitude_boundary_is_strict(self, sign):
        """A magnitude of exactly 1e5 reads as days: the test is ``> 1e5``."""
        at_the_limit = np.array([sign * 1e5])

        np.testing.assert_array_equal(times_to_mjd(at_the_limit), at_the_limit)

    def test_an_empty_column_converts_to_nothing(self):
        assert times_to_mjd(np.array([])).size == 0

    # -- a declared unit outranks the heuristic -----------------------------

    def test_a_declared_day_unit_beats_the_spacing_rule(self):
        """Spacing says seconds; the MS says days, and the MS is authoritative."""
        days = self._days()

        np.testing.assert_array_equal(times_to_mjd(days * 86400.0, "d"), days * 86400.0)

    def test_a_declared_second_unit_beats_the_spacing_rule(self):
        days = self._days()

        np.testing.assert_allclose(times_to_mjd(days, "s"), days / 86400.0, rtol=1e-15)

    def test_a_declared_unit_beats_the_magnitude_fallback(self):
        one = np.array([self.MJD])

        np.testing.assert_array_equal(times_to_mjd(one, "d"), [self.MJD])

    # -- reading the declaration -------------------------------------------

    def test_reads_the_declared_unit(self):
        assert read_time_unit(_keywords("UTC")) == "s"

    @pytest.mark.parametrize(
        "declared, expected",
        [
            ("s", "s"), ("S", "s"), (" s ", "s"), ("seconds", "s"),
            ("d", "d"), ("day", "d"), ("days", "d"),
        ],
    )
    def test_unit_spellings(self, declared, expected):
        assert read_time_unit(_keywords(units=[declared])) == expected

    @pytest.mark.parametrize("declaration", [np.array(["s"]), "s"])
    def test_the_keyword_is_read_as_casacore_stored_it(self, declaration):
        """dask-ms hands the keyword back as it was written: array or scalar."""
        assert read_time_unit(_keywords(units=declaration)) == "s"

    def test_an_undeclared_unit_leaves_it_to_the_heuristic(self):
        assert read_time_unit(_keywords("UTC", units=None)) is None

    def test_an_empty_declaration_leaves_it_to_the_heuristic(self):
        assert read_time_unit(_keywords(units=[])) is None

    def test_a_missing_column_leaves_it_to_the_heuristic(self):
        assert read_time_unit({}) is None

    def test_none_keywords_leave_it_to_the_heuristic(self):
        assert read_time_unit(None) is None

    def test_an_unrecognised_unit_is_reported(self):
        """Ignoring a declaration silently is what the warning exists to stop."""
        with pytest.warns(UserWarning, match="rad"):
            assert read_time_unit(_keywords(units=["rad"])) is None

    def test_a_recognised_unit_is_read_without_a_warning(self, recwarn):
        assert read_time_unit(_keywords(units=["d"])) == "d"
        assert not recwarn.list

    def test_a_different_column_can_be_read(self):
        keywords = _keywords(units=["d"], column="TIME_CENTROID")

        assert read_time_unit(keywords, column="TIME_CENTROID") == "d"

    # -- through the reader -------------------------------------------------

    def test_read_ms_converts_seconds_to_mjd(self, run_reader):
        days = self._days(n_time=3)

        data = run_reader(days * 86400.0)

        np.testing.assert_allclose(data["times_mjd"], days, rtol=1e-15)

    def test_read_ms_reads_a_single_integration_ms(self, run_reader):
        """The IndexError of issue #148: one integration has no spacing to read."""
        data = run_reader(np.array([self.MJD * 86400.0]))

        assert data["n_time"] == 1
        np.testing.assert_allclose(data["times_mjd"], [self.MJD], rtol=1e-15)

    def test_read_ms_honours_the_declared_unit(self, run_reader):
        """Declared seconds, spacing of days: the reader follows the declaration."""
        times = self._days(n_time=3)

        data = run_reader(times, keywords=_keywords("UTC", units=["s"]))

        np.testing.assert_allclose(data["times_mjd"], times / 86400.0, rtol=1e-15)

    def test_the_reader_and_preflight_agree_on_a_descending_column(
        self, run_reader, monkeypatch
    ):
        """The units cannot part company over the order the MS stores its blocks.

        ``ms_layout`` permits timestep blocks that do not ascend, and the
        preflight epoch helper sorts the column where ``read_ms`` keeps it in
        block order. Only that order may differ between them: a unit read one
        way for the TLE age checks and another for the fit would age the orbits
        against an epoch the run never used.
        """
        from tabascal import orbit_config

        days = self._days(n_time=3)[::-1]
        n_bl = len(np.triu_indices(3, k=1)[0])
        monkeypatch.setattr(
            orbit_config,
            "_ms_times_and_scale",
            lambda ms: (np.repeat(days * 86400.0, n_bl), "utc"),
        )

        from_reader = run_reader(days * 86400.0)["times_mjd"]
        from_preflight = orbit_config.ms_integration_times_mjd("in-memory.ms")

        np.testing.assert_allclose(from_reader, days, rtol=1e-15)
        np.testing.assert_allclose(np.sort(from_reader), from_preflight, rtol=1e-15)

    def test_the_preflight_epoch_of_a_half_second_ms_is_the_observation(
        self, run_reader, monkeypatch
    ):
        """Issue #208 as it was actually met: an OverflowError out of the epoch.

        The unit misread as days made ``ms_observation_epoch_jd`` return a
        Julian Date of ~5.16e9, which ``jd_to_datetime`` cannot express --
        ``OverflowError: Python int too large to convert to C int``, raised
        before the run had read a visibility. The epoch has to be the instant
        the MS was taken, not merely a number the conversion survives, so this
        checks the datetime and not just the absence of the raise.
        """
        from tabascal import orbit_config
        from tabascal.time import jd_to_datetime

        days = self._days(n_time=4, step=0.5)
        n_bl = len(np.triu_indices(3, k=1)[0])
        monkeypatch.setattr(
            orbit_config,
            "_ms_times_and_scale",
            lambda ms: (np.repeat(days * 86400.0, n_bl), "utc"),
        )

        epoch_jd = orbit_config.ms_observation_epoch_jd("in-memory.ms")

        # The conversion first, so an unconverted column raises here as it did
        # in the field, rather than being caught a line earlier as a number.
        # Not exact: a JD is ~2.46e6 days, so a double resolves it to ~50 us,
        # which is far below an integration and far above a millisecond.
        instant = jd_to_datetime(epoch_jd)
        assert abs(instant - datetime(2025, 1, 1, 0, 0, 0, 750000)) < timedelta(
            milliseconds=1
        )
        np.testing.assert_allclose(epoch_jd, mjd_to_jd(days.mean()), rtol=1e-15)


class TestDeclaredTimeScale:
    """The ``TIME`` column says what scale its numbers are on; the reader obeys it.

    The declaration is honoured by normalising to UTC once, here at the
    boundary, rather than by threading a scale through the trajectory maths.
    Everything past ``read_ms`` -- skyfield, ``sgp4jax.itrf_to_gcrf`` (which has
    no scale concept at all) and the TLE epoch checks -- reads UTC Julian Dates,
    so one conversion covers all of them.
    """

    #: 2025-01-01T00:00:00 UTC as an MJD, when TAI - UTC was 37 s.
    MJD = 60676.0

    #: Leap seconds at that epoch. Fixed forever: a future leap second changes
    #: the offset from then on, never the one that applied in 2025.
    LEAP_SECS = 37.0

    def _times(self, n_time=4, step=8.0):
        """``n_time`` integrations ``step`` seconds apart, in MS seconds."""

        return (self.MJD + np.arange(n_time) * step / 86400.0) * 86400.0

    def _read_both(self, run_reader):
        """The same TIME column, declared UTC and declared TAI."""

        times = self._times()

        return (
            run_reader(times, keywords=_keywords("UTC")),
            run_reader(times, keywords=_keywords("TAI")),
        )

    def test_the_scale_leaves_the_reader(self, run_reader):
        data = run_reader(self._times(), keywords=_keywords("TAI"))

        assert data["time_scale"] == "tai"

    def test_a_utc_ms_reads_exactly_as_it_did_before(self, run_reader):
        """Bit-identical: the common case gains no arithmetic and loses no digits."""
        data = run_reader(self._times(), keywords=_keywords("UTC"))

        np.testing.assert_array_equal(data["times_jd"], mjd_to_jd(data["times_mjd"]))

    def test_the_declared_times_are_kept_as_declared(self, run_reader):
        """``times_mjd`` is the MS's own column in days, so it stays on the MS's
        own scale -- and the preflight helper reports it the same way."""
        utc, tai = self._read_both(run_reader)

        np.testing.assert_array_equal(tai["times_mjd"], utc["times_mjd"])

    def test_a_tai_ms_is_normalised_to_utc(self, run_reader):
        """The same numbers on TAI name an instant 37 s earlier than on UTC."""
        utc, tai = self._read_both(run_reader)

        shift = (tai["times_jd"] - utc["times_jd"]) * 86400.0

        np.testing.assert_allclose(shift, -self.LEAP_SECS, atol=1e-3)

    def test_a_non_utc_ms_no_longer_only_warns(self, run_reader, capsys):
        """The placeholder that stood in for the behaviour until it existed."""
        run_reader(self._times(), keywords=_keywords("TAI"))

        assert "#133" not in capsys.readouterr().out

    def test_a_scale_that_cannot_be_interpreted_stops_the_read(self, run_reader):
        """Better than reading a sidereal angle as if it were a timestamp.

        Raised before the visibilities are read, and before anything is
        modelled: there is no defensible thing to do with an epoch reference
        tabascal cannot place on a time line.
        """
        with pytest.raises(ValueError, match="valid Measurement Set epoch reference"):
            run_reader(self._times(), keywords=_keywords("GAST"))

    # -- the acceptance check: it reaches the satellites --------------------

    def test_a_tai_ms_moves_the_satellite_by_the_leap_seconds(self, run_reader):
        """Where the scale has to arrive, and what it costs when it does not.

        The satellite is propagated over the times each read produced. Declaring
        TAI must land it where it is 37 s before the UTC reading puts it -- which
        is ~285 km along a LEO ground track, not a rounding difference. This is
        the check that fails if the declaration is dropped anywhere between the
        ``MEASINFO`` record and the propagator.
        """
        from tabascal.components.trajectory import get_satellite_positions

        from .tle_helpers import make_tle_record

        records = [make_tle_record(25544, mjd_to_jd(self.MJD))]
        utc, tai = self._read_both(run_reader)

        from_tai = get_satellite_positions(records, tai["times_jd"])
        from_utc = get_satellite_positions(records, utc["times_jd"])
        expected = get_satellite_positions(
            records, utc["times_jd"] - self.LEAP_SECS / 86400.0
        )

        # Metres, against ~7.7 km/s of orbital motion: the two ways of naming
        # the instant differ only by the ~40 us an f64 Julian Date resolves to.
        np.testing.assert_allclose(from_tai, expected, rtol=0, atol=1.0)

        # ~7.7 km/s for 37 s, along a curved track: ~285 km, and bounded on both
        # sides so a shift of the wrong size fails as loudly as no shift at all.
        moved = np.linalg.norm(from_utc - from_tai, axis=-1)
        assert moved.min() > 280e3 and moved.max() < 290e3

    # -- and what it must not disturb on the way ----------------------------

    def test_preflight_and_the_reader_land_on_one_epoch_for_a_tai_ms(
        self, run_reader, monkeypatch
    ):
        """Both sides of the agreement check are on UTC, and both are correct.

        Preflight runs before ``read_ms`` and reads the same ``TIME`` column
        through casacore -- including its ``MEASINFO`` record, so it normalises
        the same way. That keeps the two comparable *and* puts the TLE
        selection and age limits at the instant the observation actually
        happened. Left on the declared scale, preflight would age its records
        against an epoch 37 s from the one the fit propagates at.
        """
        from tabascal import orbit_config
        from tabascal.orbit import TLEResolution, check_epoch_agreement

        times = self._times()
        n_bl = len(np.triu_indices(3, k=1)[0])
        monkeypatch.setattr(
            orbit_config,
            "_ms_times_and_scale",
            lambda ms: (np.repeat(times, n_bl), "tai"),
        )
        preflight_epoch = orbit_config.ms_observation_epoch_jd("in-memory.ms")

        data = run_reader(times, keywords=_keywords("TAI"))

        # The same instant from both reads, to the ~40 us an f64 JD resolves to.
        assert (
            preflight_epoch - float(np.mean(data["times_jd"]))
        ) * 86400.0 == pytest.approx(0.0, abs=1e-3)

        # And it is the shifted instant, not the declared number read as UTC.
        declared = float(np.mean(mjd_to_jd(data["times_mjd"])))
        assert (preflight_epoch - declared) * 86400.0 == pytest.approx(
            -self.LEAP_SECS, abs=1e-3
        )

        check_epoch_agreement(
            TLEResolution(
                requested=[25544],
                obs_epoch_jd=preflight_epoch,
                remote_max_age_days=None,
            ),
            data["times_jd"],
        )

    def test_the_config_carries_the_scale_and_the_utc_times(self, run_reader):
        """``TabConfig`` used to unpack every key of the read but this one.

        Called on a stand-in rather than a real ``TabConfig``, which would want
        an MS on disk and a TLE preflight to construct -- neither of which has
        anything to say about the time scale.
        """
        from types import SimpleNamespace

        from tabascal.config import TabConfig

        # Leaves the in-memory MS patched in behind ``read_ms`` for the call below.
        data = run_reader(self._times(), keywords=_keywords("TAI"))

        config = SimpleNamespace(ms_path="in-memory.ms")
        TabConfig.read_ms_params(config, None, "xx", "DATA")

        assert config.time_scale == "tai"
        np.testing.assert_array_equal(config.times_jd, data["times_jd"])
        np.testing.assert_array_equal(config.times_mjd, data["times_mjd"])


# ---------------------------------------------------------------------------
# DATA_DESCRIPTION: which subtable rows the data actually uses
# ---------------------------------------------------------------------------

class _FakeDataDesc:
    def __init__(self, spw_ids, pol_ids):
        self.SPECTRAL_WINDOW_ID = _FakeVar(np.asarray(spw_ids))
        self.POLARIZATION_ID = _FakeVar(np.asarray(pol_ids))


@pytest.fixture
def data_description(monkeypatch):
    """Patch the DATA_DESCRIPTION read with a chosen id mapping."""

    def _install(spw_ids, pol_ids):
        def fake_xds_from_table(path, group_cols=None):
            assert path.endswith("::DATA_DESCRIPTION")
            return [_FakeDataDesc(spw_ids, pol_ids)]

        monkeypatch.setattr("tabascal.ms.xds_from_table", fake_xds_from_table)

    return _install


class TestResolveDataDescription:
    """An MS does not tie its data to row 0 of SPECTRAL_WINDOW / POLARIZATION."""

    def test_single_setup_resolves_to_zero(self, data_description):
        """The common case, and why assuming 0 usually works."""
        data_description([0], [0])

        assert resolve_data_description("fake.ms", 0) == (0, 0)

    def test_a_later_data_desc_id_selects_its_own_rows(self, data_description):
        """The bug: hardcoding row 0 reads another setup's configuration."""
        data_description([0, 1, 2], [0, 1, 1])

        assert resolve_data_description("fake.ms", 1) == (1, 1)
        assert resolve_data_description("fake.ms", 2) == (2, 1)

    def test_spw_and_pol_ids_are_independent(self, data_description):
        """Several windows can share one polarization setup, and vice versa."""
        data_description([3, 4], [1, 1])

        assert resolve_data_description("fake.ms", 0) == (3, 1)
        assert resolve_data_description("fake.ms", 1) == (4, 1)

    def test_unreadable_table_falls_back_with_a_warning(self, monkeypatch, capsys):
        def broken(path, group_cols=None):
            raise RuntimeError("no such table")

        monkeypatch.setattr("tabascal.ms.xds_from_table", broken)

        assert resolve_data_description("fake.ms", 1) == (0, 0)
        assert "Warning" in capsys.readouterr().out


class TestCorrelationUsesTheRightPolarizationRow:

    def test_pol_id_selects_the_row(self, polarization):
        """Row 1 holds YY only; row 0 holds the full four."""
        polarization(
            [CORR_TYPES[c] for c in ("xx", "xy", "yx", "yy")],
            [CORR_TYPES["yy"]],
        )

        assert resolve_correlation("fake.ms", "yy", pol_id=0) == 3
        assert resolve_correlation("fake.ms", "yy", pol_id=1) == 0

    def test_reading_row_zero_would_accept_an_absent_correlation(self, polarization):
        """The failure the fix prevents: row 0 holds XX, the data's row does not."""
        polarization([CORR_TYPES["xx"]], [CORR_TYPES["yy"]])

        assert resolve_correlation("fake.ms", "xx", pol_id=0) == 0
        with pytest.raises(ValueError, match="does not contain correlation 'xx'"):
            resolve_correlation("fake.ms", "xx", pol_id=1)

    def test_defaults_to_row_zero(self, polarization):
        polarization([CORR_TYPES["xx"]], [CORR_TYPES["yy"]])

        assert resolve_correlation("fake.ms", "xx") == 0


class TestHeterogeneousPolarizationRows:
    """Rows of differing NUM_CORR: the case an ungrouped read cannot represent.

    CORR_TYPE is a variable-shaped CASA column. Read ungrouped, dask-ms describes
    the whole subtable with one exemplar row's width and fails on any row that
    differs -- so a four-correlation setup beside a YY-only one breaks precisely
    the single-correlation support #128 adds.
    """

    def test_wide_row_beside_narrow_row(self, polarization):
        polarization(
            [CORR_TYPES[c] for c in ("xx", "xy", "yx", "yy")],   # 4 wide
            [CORR_TYPES["yy"]],                                   # 1 wide
        )

        assert resolve_correlation("fake.ms", "yy", pol_id=0) == 3
        assert resolve_correlation("fake.ms", "yy", pol_id=1) == 0

    def test_narrow_row_first(self, polarization):
        """Order must not matter; neither row is the exemplar."""
        polarization(
            [CORR_TYPES["yy"]],
            [CORR_TYPES[c] for c in ("xx", "xy", "yx", "yy")],
        )

        assert resolve_correlation("fake.ms", "yy", pol_id=0) == 0
        assert resolve_correlation("fake.ms", "yy", pol_id=1) == 3

    def test_two_correlation_row_beside_four(self, polarization):
        polarization(
            [CORR_TYPES[c] for c in ("xx", "xy", "yx", "yy")],
            [CORR_TYPES["xx"], CORR_TYPES["yy"]],
        )

        assert resolve_correlation("fake.ms", "yy", pol_id=1) == 1
        with pytest.raises(ValueError, match="does not contain correlation 'xy'"):
            resolve_correlation("fake.ms", "xy", pol_id=1)


# ---------------------------------------------------------------------------
# Row layout: the (n_time, n_bl) reshape every reader and writer relies on
# ---------------------------------------------------------------------------

class _FakeRows:
    """An MS partition stripped to the three columns ms_layout reads.

    dask-backed, as the daskms dataset is, in row chunks that deliberately do
    not divide the baseline count: the layout checks must reshape across chunk
    boundaries the way they will on a real MS.
    """

    ROW_CHUNK = 5

    def __init__(self, a1, a2, times=None, attrs=None):
        import dask.array as da

        a1, a2 = np.asarray(a1), np.asarray(a2)
        if times is None:
            times = np.zeros(len(a1))

        column = lambda values: _FakeVar(da.from_array(values, chunks=self.ROW_CHUNK))
        self.ANTENNA1 = column(a1)
        self.ANTENNA2 = column(a2)
        self.TIME = column(np.asarray(times))
        self.attrs = {} if attrs is None else attrs


def _time_major(n_ant: int = 4, n_time: int = 3):
    """A well-formed time-major partition and its one block of pairs."""

    a1_bl, a2_bl = np.triu_indices(n_ant, k=1)
    n_bl = len(a1_bl)
    times = np.repeat(np.arange(n_time, dtype=float), n_bl)

    return _FakeRows(np.tile(a1_bl, n_time), np.tile(a2_bl, n_time), times), a1_bl, a2_bl


class TestMSLayout:
    """The four facts the reshape needs, derived once for reader and writer."""

    def test_derives_the_grid_and_the_antenna_pairs(self):
        xds, a1_bl, a2_bl = _time_major(n_ant=4, n_time=3)

        layout = ms_layout(xds)

        assert (layout.n_time, layout.n_bl) == (3, 6)
        np.testing.assert_array_equal(layout.a1, a1_bl)
        np.testing.assert_array_equal(layout.a2, a2_bl)

    def test_reads_both_antenna_columns(self):
        """The regression guard: reading ANTENNA1 twice was the original bug."""
        a1_col, a2_col = np.triu_indices(4, k=1)

        layout = ms_layout(_FakeRows(a1_col, a2_col))

        np.testing.assert_array_equal(layout.a1, a1_col)
        np.testing.assert_array_equal(layout.a2, a2_col)
        assert not np.array_equal(layout.a1, layout.a2)

    def test_takes_only_the_first_baseline_block(self):
        """The columns repeat per timestep; one block of baselines is wanted."""
        xds, a1_bl, a2_bl = _time_major(n_time=3)

        layout = ms_layout(xds)

        assert len(layout.a1) == len(a1_bl) and len(layout.a2) == len(a2_bl)
        np.testing.assert_array_equal(layout.a2, a2_bl)

    def test_ragged_row_counts_are_rejected(self):
        """Rows that do not divide into whole timesteps break the reshape."""
        a1_bl, a2_bl = np.triu_indices(4, k=1)
        xds = _FakeRows(
            np.tile(a1_bl, 2)[:-1],
            np.tile(a2_bl, 2)[:-1],
            np.repeat([0.0, 1.0], 6)[:-1],
        )

        with pytest.raises(ValueError, match="whole number of baselines"):
            ms_layout(xds)

    def test_baseline_major_ordering_is_rejected(self):
        """All times of one baseline first repeats each pair down the rows."""
        a1_bl, a2_bl = np.triu_indices(4, k=1)
        n_time = 3
        xds = _FakeRows(
            np.repeat(a1_bl, n_time),
            np.repeat(a2_bl, n_time),
            np.tile(np.arange(n_time, dtype=float), len(a1_bl)),
        )

        with pytest.raises(ValueError, match="not ordered time-major"):
            ms_layout(xds)

    def test_partially_repeated_pairs_are_rejected(self):
        a1_col, a2_col = np.triu_indices(4, k=1)
        a1_col, a2_col = a1_col.copy(), a2_col.copy()
        a1_col[-1], a2_col[-1] = a1_col[0], a2_col[0]   # one duplicate pair

        with pytest.raises(ValueError, match="distinct antenna pairs"):
            ms_layout(_FakeRows(a1_col, a2_col))

    def test_a_permuted_timestep_is_rejected(self):
        a1_bl, a2_bl = np.triu_indices(4, k=1)
        perm = np.array([3, 1, 0, 5, 4, 2])
        xds = _FakeRows(
            np.concatenate([a1_bl, a1_bl[perm]]),
            np.concatenate([a2_bl, a2_bl[perm]]),
            np.repeat([0.0, 1.0], 6),
        )

        with pytest.raises(ValueError, match="differs between timesteps"):
            ms_layout(xds)

    def test_interleaved_timesteps_are_rejected(self):
        """Pairs repeat per block and the rows reshape cleanly -- and every
        visibility still lands on the wrong timestamp."""
        a1_bl, a2_bl = np.triu_indices(3, k=1)          # 3 baselines
        xds = _FakeRows(
            np.tile(a1_bl, 2),
            np.tile(a2_bl, 2),
            np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        )

        with pytest.raises(ValueError, match="interleave timesteps"):
            ms_layout(xds)

    def test_blocks_out_of_ascending_time_order_are_accepted(self):
        """Only block-constancy matters: the time axis follows the blocks."""
        a1_bl, a2_bl = np.triu_indices(4, k=1)
        n_bl = len(a1_bl)
        xds = _FakeRows(
            np.tile(a1_bl, 3), np.tile(a2_bl, 3), np.repeat([2.0, 0.0, 1.0], n_bl)
        )

        layout = ms_layout(xds)

        assert layout.n_time == 3
        np.testing.assert_array_equal(layout.a1, a1_bl)


    def test_never_holds_a_full_column_in_memory(self):
        """Nothing larger than a chunk, or the n_time unique times, is ever a
        task result: the checks over the full columns are reductions.
        """
        import dask
        from dask.callbacks import Callback

        xds, _, _ = _time_major(n_ant=6, n_time=4)   # 60 rows, chunks of 5
        n_row = xds.TIME.data.shape[0]
        largest = []

        class Largest(Callback):
            def _posttask(self, key, result, dsk, state, worker_id):
                largest.append(int(getattr(result, "size", 0)))

        with dask.config.set(scheduler="synchronous"), Largest():
            layout = ms_layout(xds)

        assert layout.n_bl == 15 and layout.n_time == 4
        assert max(largest) < n_row
        assert max(largest) <= max(_FakeRows.ROW_CHUNK, layout.n_bl, layout.n_time) * 2

# ---------------------------------------------------------------------------
# Which subtable rows a partition uses
# ---------------------------------------------------------------------------

class TestPartitionSetup:
    """The partition names its own DATA_DESC_ID; row 0 is only a convention."""

    def test_the_partitions_id_is_resolved(self, data_description):
        data_description([0, 1], [0, 2])
        xds, _, _ = _time_major()
        xds.attrs = {"DATA_DESC_ID": 1}

        assert partition_setup("fake.ms", xds) == (1, 2)
        assert partition_polarization("fake.ms", xds) == 2

    def test_a_partition_without_the_attribute_falls_back_to_zero(
        self, data_description
    ):
        data_description([3], [4])
        xds, _, _ = _time_major()

        assert partition_setup("fake.ms", xds) == (3, 4)
        assert partition_polarization("fake.ms", xds) == 4


# ---------------------------------------------------------------------------
# Which noise column a partition's noise comes from
# ---------------------------------------------------------------------------

N_TIME_N, N_BL_N, N_FREQ_N = 4, 3, 2


def _noise_partition(sigma=None, sigma_spectrum=None):
    """An MS partition stripped to the noise columns, as a real dataset.

    An ``xr.Dataset`` rather than a stub: the chain asks whether a column is
    there at all, and a real dataset raises ``AttributeError`` for a missing
    variable and hands back a dask array for a present one -- exactly what
    dask-ms does, and the two halves the chain turns on.
    """

    import dask.array as da
    import xarray as xr

    data = {}
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        data["SIGMA"] = (("row", "corr"), da.from_array(sigma, chunks=(5, -1)))
    if sigma_spectrum is not None:
        sigma_spectrum = np.asarray(sigma_spectrum, dtype=float)
        data["SIGMA_SPECTRUM"] = (
            ("row", "chan", "corr"),
            da.from_array(sigma_spectrum, chunks=(5, -1, -1)),
        )

    return xr.Dataset(data)


def _sigma_rows(per_bl, n_corr=1):
    """A SIGMA column holding ``per_bl``, repeated over ``N_TIME_N`` timesteps."""

    per_bl = np.asarray(per_bl, dtype=float)

    return np.repeat(
        np.tile(per_bl, N_TIME_N)[:, None], n_corr, axis=1
    )


class _UnfilledColumn:
    """A column CASA has declared and never written a cell of.

    Reading one raises rather than returning anything, which is what makes an
    optional column's absence two cases instead of one.
    """

    class _Data:
        def compute(self):
            raise RuntimeError("no array in row 0")

    data = _Data()


def _spectrum_rows(per_bl_freq, n_corr=1):
    """A SIGMA_SPECTRUM column holding ``per_bl_freq``, repeated over time."""

    per_bl_freq = np.asarray(per_bl_freq, dtype=float)
    rows = np.tile(per_bl_freq, (N_TIME_N, 1, 1)).reshape(-1, per_bl_freq.shape[1])

    return np.repeat(rows[:, :, None], n_corr, axis=2)


def _sigma_rows_time(per_bl_time, n_corr=1):
    """A SIGMA column that changes over time, ``(n_bl, n_time)`` in.

    Time-major, like every other column: row ``t * n_bl + b`` is baseline ``b``
    at time ``t``.
    """

    per_bl_time = np.asarray(per_bl_time, dtype=float)

    return np.repeat(per_bl_time.T.reshape(-1, 1), n_corr, axis=1)


def _spectrum_rows_time(per_bl_freq_time, n_corr=1):
    """A SIGMA_SPECTRUM that changes over time, ``(n_bl, n_freq, n_time)`` in."""

    arr = np.asarray(per_bl_freq_time, dtype=float)

    return np.repeat(grid_to_rows(arr, arr.shape[1]), n_corr, axis=2)


class TestPartitionNoise:
    """The resolution chain ``read_ms`` reads its noise through.

    Exercised here rather than through ``read_ms`` itself, which needs a real MS
    on disk -- five subtables of it -- to reach these four lines. The chain is
    the whole of the decision; ``read_ms`` only passes it the partition and the
    grid it has already derived.
    """

    PER_BL = np.array([1.0, 2.0, 4.0])
    PER_BL_FREQ = np.array([[1.0, 10.0], [2.0, 20.0], [4.0, 40.0]])

    def _call(self, xds, **kwargs):
        return partition_noise(
            xds, N_TIME_N, N_BL_N, N_FREQ_N, **kwargs
        )

    def test_a_spectrum_gives_a_per_baseline_channel_noise(self):
        """Frequency-dependent by default: the band is not flat and the MS says so."""
        xds = _noise_partition(sigma_spectrum=_spectrum_rows(self.PER_BL_FREQ))

        out = self._call(xds)

        assert out.shape == (N_BL_N, N_FREQ_N)
        np.testing.assert_allclose(out, self.PER_BL_FREQ)

    def test_sigma_is_used_when_there_is_no_spectrum(self):
        """Most MSs carry only SIGMA, and per baseline is still better than a scalar."""
        xds = _noise_partition(sigma=_sigma_rows(self.PER_BL))

        out = self._call(xds)

        assert out.shape == (N_BL_N,)
        np.testing.assert_allclose(out, self.PER_BL)

    def test_the_spectrum_wins_when_both_are_there(self):
        """SIGMA is the band-averaged version of the same measurement."""
        xds = _noise_partition(
            sigma=_sigma_rows(np.full(N_BL_N, 99.0)),
            sigma_spectrum=_spectrum_rows(self.PER_BL_FREQ),
        )

        np.testing.assert_allclose(self._call(xds), self.PER_BL_FREQ)

    def test_an_empty_spectrum_falls_through_to_sigma(self, capsys):
        """A column of zeros is a column that was never filled in."""
        xds = _noise_partition(
            sigma=_sigma_rows(self.PER_BL),
            sigma_spectrum=_spectrum_rows(np.zeros((N_BL_N, N_FREQ_N))),
        )

        out = self._call(xds)

        np.testing.assert_allclose(out, self.PER_BL)
        assert "SIGMA_SPECTRUM" in capsys.readouterr().out

    def test_a_spectrum_of_other_channels_falls_through_to_sigma(self, capsys):
        """It cannot line up with the visibilities, so it cannot weight them."""
        xds = _noise_partition(
            sigma=_sigma_rows(self.PER_BL),
            sigma_spectrum=_spectrum_rows(np.ones((N_BL_N, N_FREQ_N + 3))),
        )

        out = self._call(xds)

        np.testing.assert_allclose(out, self.PER_BL)
        assert "channels" in capsys.readouterr().out

    def test_the_requested_correlation_is_selected_from_the_spectrum(self):
        col = np.concatenate(
            [
                _spectrum_rows(self.PER_BL_FREQ),
                _spectrum_rows(self.PER_BL_FREQ * 10),
            ],
            axis=2,
        )

        np.testing.assert_allclose(
            self._call(_noise_partition(sigma_spectrum=col), corr_idx=1),
            self.PER_BL_FREQ * 10,
        )

    def test_the_requested_correlation_is_selected_from_sigma(self):
        col = np.stack(
            [_sigma_rows(self.PER_BL)[:, 0], _sigma_rows(self.PER_BL * 10)[:, 0]],
            axis=1,
        )

        np.testing.assert_allclose(
            self._call(_noise_partition(sigma=col), corr_idx=1), self.PER_BL * 10
        )

    def test_a_declared_but_unfilled_spectrum_falls_through_to_sigma(self, capsys):
        """SIGMA_SPECTRUM is optional, and CASA writes the column before any of
        its cells: unreadable is the same as absent, not a reason to stop."""
        from types import SimpleNamespace

        xds = SimpleNamespace(
            SIGMA=_FakeVar(_sigma_rows(self.PER_BL)),
            SIGMA_SPECTRUM=_UnfilledColumn(),
        )

        np.testing.assert_allclose(self._call(xds), self.PER_BL)
        assert "could not be read" in capsys.readouterr().out

    def test_neither_column_leaves_the_noise_unset(self, capsys):
        """Never invented -- and never terminal here either.

        ``data.noise`` is read *after* the MS, so a read that raised would take
        the override's turn away and the documented recovery could not happen.
        The read says why it found nothing and returns None; ``TabConfig`` gives
        the override its chance and stops only if there is still no noise.
        """
        assert self._call(_noise_partition()) is None
        assert "neither" in capsys.readouterr().out

    def test_an_unreadable_sigma_leaves_the_noise_unset(self, capsys):
        """Said in tabascal's words, not as an opaque casacore error from three
        frames down -- and said as a warning, since the override may yet fix it."""
        from types import SimpleNamespace

        assert self._call(SimpleNamespace(SIGMA=_UnfilledColumn())) is None
        assert "could not be read" in capsys.readouterr().out

    def test_both_columns_empty_leaves_the_noise_unset(self, capsys):
        xds = _noise_partition(
            sigma=_sigma_rows(np.zeros(N_BL_N)),
            sigma_spectrum=_spectrum_rows(np.zeros((N_BL_N, N_FREQ_N))),
        )

        assert self._call(xds) is None
        assert "data.noise" in capsys.readouterr().out

    def test_a_malformed_sigma_is_still_an_error(self, capsys):
        """Deferring the *empty* column is not deferring a broken one: a row
        count that disagrees with the grid means the reader and the MS describe
        different observations, which no data.noise value makes right."""
        xds = _noise_partition(sigma=_sigma_rows(self.PER_BL))

        with pytest.raises(ValueError, match="does not match the observation grid"):
            partition_noise(xds, N_TIME_N + 1, N_BL_N, N_FREQ_N)

    def test_a_time_varying_spectrum_keeps_the_time_axis(self):
        """A column that changes over time is the MS saying the noise changed,
        so the chain hands back the whole ``(n_bl, n_freq, n_time)`` grid."""
        per_bl_freq_time = 1.0 + np.arange(
            N_BL_N * N_FREQ_N * N_TIME_N, dtype=float
        ).reshape(N_BL_N, N_FREQ_N, N_TIME_N)

        out = self._call(
            _noise_partition(sigma_spectrum=_spectrum_rows_time(per_bl_freq_time))
        )

        assert out.shape == (N_BL_N, N_FREQ_N, N_TIME_N)
        np.testing.assert_allclose(out, per_bl_freq_time)

    def test_a_time_varying_spectrum_still_wins_over_sigma(self):
        """The order of preference is about which column is more resolved, and
        a time-resolved spectrum is the most resolved answer there is."""
        per_bl_freq_time = 1.0 + np.arange(
            N_BL_N * N_FREQ_N * N_TIME_N, dtype=float
        ).reshape(N_BL_N, N_FREQ_N, N_TIME_N)
        xds = _noise_partition(
            sigma=_sigma_rows(np.full(N_BL_N, 99.0)),
            sigma_spectrum=_spectrum_rows_time(per_bl_freq_time),
        )

        np.testing.assert_allclose(self._call(xds), per_bl_freq_time)

    def test_a_time_varying_sigma_is_read_when_there_is_no_spectrum(self):
        """``(n_bl, 1, n_time)`` -- never ``(n_bl, n_time)``, which with
        n_freq == n_time nothing downstream could tell from a channel axis."""
        per_bl_time = np.outer(self.PER_BL, 1.0 + np.arange(N_TIME_N, dtype=float))

        out = self._call(_noise_partition(sigma=_sigma_rows_time(per_bl_time)))

        assert out.shape == (N_BL_N, 1, N_TIME_N)
        np.testing.assert_allclose(out[:, 0, :], per_bl_time)

    def test_a_time_varying_but_empty_spectrum_falls_through_to_sigma(self, capsys):
        """The fallthrough does not care how the column varies, only that none
        of it is a noise."""
        xds = _noise_partition(
            sigma=_sigma_rows(self.PER_BL),
            sigma_spectrum=_spectrum_rows_time(
                -1.0
                - np.arange(N_BL_N * N_FREQ_N * N_TIME_N, dtype=float).reshape(
                    N_BL_N, N_FREQ_N, N_TIME_N
                )
            ),
        )

        out = self._call(xds)

        np.testing.assert_allclose(out, self.PER_BL)
        assert "SIGMA_SPECTRUM" in capsys.readouterr().out

    def test_a_malformed_spectrum_is_not_fallen_through(self):
        """A row count that does not match the grid means the reader and the MS
        disagree about the observation; reading SIGMA instead would bury that.

        The error must name SIGMA_SPECTRUM: falling through would raise the same
        sentence about SIGMA and point the reader at the wrong column.
        """
        xds = _noise_partition(
            sigma=_sigma_rows(self.PER_BL),
            sigma_spectrum=_spectrum_rows(self.PER_BL_FREQ),
        )

        with pytest.raises(ValueError, match="SIGMA_SPECTRUM has 12 rows"):
            partition_noise(xds, N_TIME_N + 1, N_BL_N, N_FREQ_N)


# ---------------------------------------------------------------------------
# Row/channel <-> (bl, freq, time)
# ---------------------------------------------------------------------------

class TestRowGridMapping:
    """The reshape the reader and the writer each used to spell out."""

    @pytest.fixture
    def grid(self):
        """A ``(bl, freq, time)`` array with every element distinguishable."""
        rng = np.random.default_rng(21)
        shape = (6, 3, 4)                       # 6 baselines, 3 chans, 4 times

        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    def test_round_trips(self, grid):
        n_bl, n_freq, n_time = grid.shape

        rows = grid_to_rows(grid, n_freq)
        back = rows_to_grid(rows[:, :, 0], n_time, n_bl, n_freq)

        np.testing.assert_array_equal(back, grid)

    def test_rows_are_time_major(self, grid):
        """Row ``t * n_bl + b`` holds baseline ``b`` at time ``t``."""
        n_bl, n_freq, n_time = grid.shape
        rows = grid_to_rows(grid, n_freq)

        assert rows.shape == (n_time * n_bl, n_freq, 1)
        for t in range(n_time):
            for b in range(n_bl):
                np.testing.assert_array_equal(
                    rows[t * n_bl + b, :, 0], grid[b, :, t]
                )

    def test_rows_to_grid_reproduces_the_readers_old_expression(self, grid):
        """The inline reshape/transpose read_ms used to carry."""
        n_bl, n_freq, n_time = grid.shape
        col = grid_to_rows(grid, n_freq)[:, :, 0]

        expected = np.transpose(col.reshape(n_time, n_bl, n_freq), (1, 2, 0))

        np.testing.assert_array_equal(rows_to_grid(col, n_time, n_bl, n_freq), expected)

    def test_grid_to_rows_reproduces_the_writers_old_expression(self, grid):
        """The inline transpose/reshape _to_ms_column used to carry."""
        n_freq = grid.shape[1]

        expected = np.transpose(grid, (2, 0, 1)).reshape(-1, n_freq, 1)

        np.testing.assert_array_equal(grid_to_rows(grid, n_freq), expected)

    def test_a_wider_correlation_axis_is_kept(self, grid):
        n_freq = grid.shape[1]

        assert grid_to_rows(grid, n_freq, n_corr=1).shape[2] == 1

    def test_works_on_dask_and_jax_arrays(self, grid, exact_rtol):
        """One mapping for the writer's dask arrays and the reader's jax ones."""
        da = pytest.importorskip("dask.array")
        jnp = pytest.importorskip("jax.numpy")
        n_bl, n_freq, n_time = grid.shape

        expected = grid_to_rows(grid, n_freq)

        np.testing.assert_allclose(
            np.asarray(grid_to_rows(da.from_array(grid, chunks=(3, 3, 2)), n_freq)),
            expected,
        )
        np.testing.assert_allclose(
            np.asarray(grid_to_rows(jnp.asarray(grid), n_freq)), expected, rtol=exact_rtol
        )


# ---------------------------------------------------------------------------
# Placing one fitted correlation on the MS's correlation axis
# ---------------------------------------------------------------------------

class TestIntoCorr:
    """One fitted correlation placed on a wider MS correlation axis."""

    @pytest.fixture
    def col(self):
        """A result on a length-1 correlation axis, as the writer builds it."""
        return np.arange(6, dtype=np.complex64).reshape(3, 2, 1) + 1.0

    def test_a_single_correlation_ms_is_left_alone(self, col):
        out = into_corr(col, 0, 1, 0)

        assert out is col

    def test_the_result_lands_on_the_fitted_correlation(self, col):
        out = into_corr(col, 2, 4, 0)

        assert out.shape == (3, 2, 4)
        np.testing.assert_array_equal(out[:, :, 2:3], col)

    def test_a_scalar_fill_covers_the_others(self, col):
        out = into_corr(col, 2, 4, 0)

        np.testing.assert_array_equal(out[:, :, [0, 1, 3]], 0.0)

    def test_an_array_fill_passes_its_own_values_through(self, col):
        """The data-frame columns keep the data on the correlations not fitted."""
        fill = (100 + np.arange(24)).astype(np.complex64).reshape(3, 2, 4)

        out = into_corr(col, 2, 4, fill)

        np.testing.assert_array_equal(out[:, :, [0, 1, 3]], fill[:, :, [0, 1, 3]])
        np.testing.assert_array_equal(out[:, :, 2:3], col)

    def test_the_dtype_is_preserved(self, col):
        assert into_corr(col, 2, 4, 0).dtype == np.complex64

    def test_works_on_dask_arrays(self, col):
        """write_results_ms passes dask arrays through this."""
        da = pytest.importorskip("dask.array")
        fill = (100 + np.arange(24)).astype(np.complex64).reshape(3, 2, 4)

        out = into_corr(
            da.from_array(col, chunks=(3, 2, 1)),
            2,
            4,
            da.from_array(fill, chunks=(3, 2, 4)),
        )

        np.testing.assert_array_equal(
            np.asarray(out), into_corr(col, 2, 4, fill)
        )


class TestFittedCorrelation:
    """Which correlation the results belong to, resolved by name."""

    @pytest.fixture
    def resolver(self, monkeypatch):
        """Stand in for the casacore-backed resolver, recording the name."""
        seen = {}

        def _resolve(ms_path, name, pol_id=0):
            seen["name"] = name
            seen["pol_id"] = pol_id
            return 3

        monkeypatch.setattr("tabascal.ms.resolve_correlation", _resolve)

        return seen

    def test_the_argument_is_resolved_by_name(self, resolver):
        assert fitted_correlation("ms", None, "yy", 4) == 3
        assert resolver["name"] == "yy"

    def test_the_zarr_attribute_is_used_when_no_argument_is_given(self, resolver):
        assert fitted_correlation("ms", "xy", None, 4) == 3
        assert resolver["name"] == "xy"

    def test_the_argument_wins_over_the_attribute(self, resolver):
        fitted_correlation("ms", "xx", "yy", 4)

        assert resolver["name"] == "yy"

    def test_a_single_correlation_ms_needs_no_name(self, resolver):
        """One correlation, one answer -- and nothing to resolve."""
        assert fitted_correlation("ms", None, None, 1) == 0
        assert "name" not in resolver

    def test_a_nameless_zarr_on_a_wide_ms_is_rejected(self, resolver):
        """Guessing would write the results into the wrong polarisation."""
        with pytest.raises(ValueError, match="does not record which one"):
            fitted_correlation("ms", None, None, 4)

    def test_the_error_says_how_to_fix_it(self, resolver):
        with pytest.raises(ValueError, match="tab2MS -c xx"):
            fitted_correlation("ms", None, None, 2)

    def test_the_partition_polarization_row_is_forwarded(self, resolver):
        """Row 0 is a convention, not where this partition's data lives."""
        fitted_correlation("ms", None, "yy", 4, pol_id=2)

        assert resolver["pol_id"] == 2

    def test_the_row_defaults_to_zero(self, resolver):
        fitted_correlation("ms", None, "yy", 4)

        assert resolver["pol_id"] == 0

    def test_an_index_off_the_partition_axis_is_rejected(self, resolver):
        """A wider POLARIZATION row than the data: index 3 on a 2-corr axis.

        Without this, into_corr would match nothing and silently write zero
        models and untouched data everywhere.
        """
        with pytest.raises(ValueError, match="resolves to index 3"):
            fitted_correlation("ms", None, "yy", 2)


# ---------------------------------------------------------------------------
# Backwards compatibility for the move out of tab_tools
# ---------------------------------------------------------------------------

class TestMovedNamesStayImportable:
    """read_ms and get_observation_data_type moved to tabascal.ms.

    The old import path keeps working so the move does not break callers that
    predate it, but warns so it does not become the permanent home.
    """

    @pytest.mark.parametrize("name", ["read_ms", "get_observation_data_type"])
    def test_old_import_path_still_resolves(self, name):
        import tabascal.ms
        import tabascal.tab_tools

        with pytest.warns(DeprecationWarning, match="moved to tabascal.ms"):
            moved = getattr(tabascal.tab_tools, name)

        assert moved is getattr(tabascal.ms, name)

    def test_unknown_attribute_still_raises_attribute_error(self):
        import tabascal.tab_tools

        with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
            tabascal.tab_tools.nonexistent

    def test_no_warning_when_importing_from_the_new_home(self, recwarn):
        import importlib

        importlib.reload(importlib.import_module("tabascal.ms"))

        assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]


# ---------------------------------------------------------------------------
# Channel selection
# ---------------------------------------------------------------------------

N_ANT_C, N_TIME_C, N_FREQ_C, N_CORR_C = 4, 3, 5, 2

#: The channel `-f` should land on, and the frequency asked for to reach it.
TARGET_CHAN = 2


def _channel_ms(n_ant=N_ANT_C, n_time=N_TIME_C, n_freq=N_FREQ_C, n_corr=N_CORR_C,
                sigma_spectrum=True, spectrum_chans=None, widths=None, sigma=True):
    """A whole MS partition and its subtables, in memory.

    Real ``xr.Dataset``s over dask arrays rather than stubs: channel selection
    has to survive the same lazy indexing a daskms column does, and every column
    it touches -- data, flags and the noise spectrum -- has to come back on the
    same channel axis or the visibilities are weighted by another channel's noise.
    """

    import dask.array as da
    import xarray as xr

    a1_bl, a2_bl = np.triu_indices(n_ant, k=1)
    n_bl = len(a1_bl)
    n_row = n_time * n_bl

    # Value encodes (row, channel, correlation), so a mis-selected axis cannot
    # coincide with the right answer.
    grid = (
        np.arange(n_row)[:, None, None] * 1000.0
        + np.arange(n_freq)[None, :, None] * 10.0
        + np.arange(n_corr)[None, None, :]
    )
    # Centres follow the widths, so a non-uniform window really is one: each
    # centre sits half a width past the previous channel's edge.
    widths = np.full(n_freq, 1e6) if widths is None else np.asarray(widths, float)
    edges = np.concatenate([[0.0], np.cumsum(widths)])
    freqs = 1.4e9 + 0.5 * (edges[:-1] + edges[1:])

    columns = {
        "DATA": (("row", "chan", "corr"), da.from_array(grid + 0j, chunks=(5, -1, -1))),
        "FLAG": (
            ("row", "chan", "corr"),
            da.from_array(
                np.broadcast_to(
                    (np.arange(n_row)[:, None, None]
                     + np.arange(n_freq)[None, :, None]) % 7 == 0,
                    (n_row, n_freq, n_corr),
                ).copy(),
                chunks=(5, -1, -1),
            ),
        ),
        "ANTENNA1": (("row",), da.from_array(np.tile(a1_bl, n_time), chunks=5)),
        "ANTENNA2": (("row",), da.from_array(np.tile(a2_bl, n_time), chunks=5)),
        "TIME": (
            ("row",),
            da.from_array(
                np.repeat(60000.0 + np.arange(n_time) / 86400.0, n_bl), chunks=5
            ),
        ),
        "INTERVAL": (("row",), da.from_array(np.full(n_row, 2.0), chunks=5)),
        "UVW": (("row", "uvw"), da.from_array(np.zeros((n_row, 3)), chunks=(5, -1))),
    }
    if sigma:
        # The band-averaged fallback, distinct from every channel of the spectrum.
        columns["SIGMA"] = (
            ("row", "corr"),
            da.from_array(
                np.repeat(
                    np.tile(np.arange(n_bl) + 0.5, n_time)[:, None], n_corr, axis=1
                ),
                chunks=(5, -1),
            ),
        )
    if sigma_spectrum:
        # Per (baseline, channel), constant over time, and distinct per channel.
        # `spectrum_chans` lets the column disagree with the data about the band.
        n_spec = n_freq if spectrum_chans is None else spectrum_chans
        per_bl_freq = (
            (np.arange(n_bl)[:, None] + 1.0) * (np.arange(n_spec)[None, :] + 1.0)
        )
        rows = np.tile(per_bl_freq, (n_time, 1, 1)).reshape(-1, n_spec)
        # Its own channel dimension, so the column is free to disagree with the
        # data about how many channels the observation has -- which is the whole
        # point of `spectrum_chans`, and what a real MS (separate columns, not one
        # xarray Dataset) allows without comment.
        columns["SIGMA_SPECTRUM"] = (
            ("row", "spec_chan", "corr"),
            da.from_array(
                np.repeat(rows[:, :, None], n_corr, axis=2), chunks=(5, -1, -1)
            ),
        )

    xds = xr.Dataset(columns)
    ant = xr.Dataset(
        {
            "POSITION": (("row", "xyz"), da.from_array(np.arange(n_ant * 3.0).reshape(n_ant, 3))),
            "DISH_DIAMETER": (("row",), da.from_array(np.full(n_ant, 35.0))),
        }
    )
    spec = xr.Dataset(
        {
            "CHAN_FREQ": (("row", "chan"), da.from_array(freqs[None, :])),
            "CHAN_WIDTH": (("row", "chan"), da.from_array(widths[None, :])),
        }
    )
    src = xr.Dataset(
        {"DIRECTION": (("row", "radec"), da.from_array(np.deg2rad([[30.0, -30.0]])))}
    )

    return xds, ant, spec, src, freqs, widths


@pytest.fixture
def channel_ms(monkeypatch):
    """Install an in-memory MS and return the arrays it was built from."""

    def _install(**kwargs):
        xds, ant, spec, src, freqs, widths = _channel_ms(**kwargs)
        import tabascal.ms as ms_mod

        tables = {"::ANTENNA": [ant], "::SPECTRAL_WINDOW": [spec], "::SOURCE": [src]}

        def fake_from_table(path, group_cols=None):
            for suffix, value in tables.items():
                if path.endswith(suffix):
                    return value
            raise AssertionError(f"unexpected subtable {path}")

        monkeypatch.setattr(
            ms_mod, "xds_from_ms",
            lambda path, column_keywords=False: (
                [xds], {"TIME": {"MEASINFO": {"Ref": "UTC"}}}
            ),
        )
        monkeypatch.setattr(ms_mod, "xds_from_table", fake_from_table)
        # Resolved elsewhere and separately tested; what is under test here is
        # which channels come back, not which correlation.
        monkeypatch.setattr(ms_mod, "partition_setup", lambda path, x: (0, 0))
        monkeypatch.setattr(ms_mod, "resolve_correlation", lambda path, corr, pol: 1)

        return xds, freqs, widths

    return _install


class TestReadMSChannelSelection:
    """``freq`` / ``chans`` select channels, and everything follows them.

    The selection used to be half-wired: ``freq`` produced a scalar index that
    ``len()`` could not measure, and the data columns ignored it entirely, so the
    option was advertised and unusable. What it must do now is narrow every
    channel-indexed array together -- data, flags, frequencies and the noise --
    since a noise left on the full band weights the visibilities by a channel
    they did not come from.
    """

    def test_the_whole_band_is_read_by_default(self, channel_ms):
        _, freqs, _ = channel_ms()

        ms = read_ms("fake.ms")

        assert ms["n_freq"] == N_FREQ_C
        np.testing.assert_allclose(ms["freqs"], freqs)
        assert ms["vis_obs"].shape[1] == N_FREQ_C

    def test_freq_selects_the_nearest_single_channel(self, channel_ms):
        _, freqs, _ = channel_ms()
        # Deliberately off-centre, so "nearest" is doing the work.
        asked = freqs[TARGET_CHAN] + 3e5

        ms = read_ms("fake.ms", freq=asked)

        assert ms["n_freq"] == 1
        np.testing.assert_allclose(ms["freqs"], freqs[TARGET_CHAN : TARGET_CHAN + 1])

    def test_the_selected_channel_carries_its_own_visibilities(self, channel_ms):
        channel_ms()
        full = read_ms("fake.ms")

        one = read_ms("fake.ms", freq=float(np.asarray(full["freqs"])[TARGET_CHAN]))

        assert one["vis_obs"].shape == (full["n_bl"], 1, full["n_time"])
        np.testing.assert_allclose(
            np.asarray(one["vis_obs"]),
            np.asarray(full["vis_obs"])[:, TARGET_CHAN : TARGET_CHAN + 1],
        )

    def test_the_flags_follow_the_selection(self, channel_ms):
        channel_ms()
        full = read_ms("fake.ms")

        one = read_ms("fake.ms", freq=float(np.asarray(full["freqs"])[TARGET_CHAN]))

        np.testing.assert_array_equal(
            np.asarray(one["flags"]),
            np.asarray(full["flags"])[:, TARGET_CHAN : TARGET_CHAN + 1],
        )

    def test_the_noise_follows_the_selection(self, channel_ms):
        """A SIGMA_SPECTRUM left on the full band would weight the wrong channel."""
        channel_ms()
        full = read_ms("fake.ms")

        one = read_ms("fake.ms", freq=float(np.asarray(full["freqs"])[TARGET_CHAN]))

        assert np.asarray(one["noise"]).shape == (full["n_bl"], 1)
        np.testing.assert_allclose(
            np.asarray(one["noise"]),
            np.asarray(full["noise"])[:, TARGET_CHAN : TARGET_CHAN + 1],
        )

    def test_an_explicit_channel_list_is_honoured(self, channel_ms):
        channel_ms()
        full = read_ms("fake.ms")

        some = read_ms("fake.ms", chans=np.array([0, 3]))

        assert some["n_freq"] == 2
        np.testing.assert_allclose(
            np.asarray(some["freqs"]), np.asarray(full["freqs"])[[0, 3]]
        )
        np.testing.assert_allclose(
            np.asarray(some["vis_obs"]), np.asarray(full["vis_obs"])[:, [0, 3]]
        )

    def test_a_channel_off_the_end_is_refused(self, channel_ms):
        """Out of range, rather than wrapping round to another channel."""
        channel_ms()

        with pytest.raises(ValueError, match="off the"):
            read_ms("fake.ms", chans=np.array([0, N_FREQ_C]))

    def test_asking_for_the_whole_band_in_order_is_the_default_read(self, channel_ms):
        """No selection to make, so none is made -- and the result is identical."""
        channel_ms()

        explicit = read_ms("fake.ms", chans=np.arange(N_FREQ_C))
        default = read_ms("fake.ms")

        np.testing.assert_allclose(
            np.asarray(explicit["vis_obs"]), np.asarray(default["vis_obs"])
        )
        np.testing.assert_allclose(
            np.asarray(explicit["noise"]), np.asarray(default["noise"])
        )

    def test_the_channel_widths_come_back_per_channel(self, channel_ms):
        """A spectral window need not be uniform, so one width cannot describe it."""
        widths = np.array([1e6, 2e6, 5e5, 4e6, 1e6])
        _, _, built = channel_ms(widths=widths)

        ms = read_ms("fake.ms")

        np.testing.assert_allclose(np.asarray(ms["chan_widths"]), built)

    def test_the_widths_follow_the_selection(self, channel_ms):
        widths = np.array([1e6, 2e6, 5e5, 4e6, 1e6])
        channel_ms(widths=widths)

        ms = read_ms("fake.ms", chans=np.array([1, 3]))

        np.testing.assert_allclose(np.asarray(ms["chan_widths"]), widths[[1, 3]])
        # The scalar stays what it always was: the first width being read.
        np.testing.assert_allclose(float(np.asarray(ms["chan_width"])), widths[1])


class TestReadMSFrequencyRequest:
    """`freq` names a channel of *this* band, or it is not a request this MS can serve.

    ``argmin`` always returns an index, so without a range check a frequency from
    another subband -- or a units slip, GHz for Hz -- reads the nearest edge
    channel and reports nothing. The estimate then comes back on a channel nobody
    asked for.
    """

    def test_a_frequency_outside_the_band_is_refused(self, channel_ms):
        _, freqs, _ = channel_ms()

        with pytest.raises(ValueError, match="outside"):
            read_ms("fake.ms", freq=2.4e9)

    def test_the_error_names_the_request_and_the_band(self, channel_ms):
        _, freqs, _ = channel_ms()

        with pytest.raises(ValueError) as excinfo:
            read_ms("fake.ms", freq=2.4e9)

        message = str(excinfo.value)
        assert "2400" in message                      # the frequency asked for, MHz
        assert f"{freqs[0] / 1e6:.4f}" in message      # and the band it is not in
        assert f"{freqs[-1] / 1e6:.4f}" in message

    @pytest.mark.parametrize("freq", [float("nan"), float("-nan")])
    def test_a_non_finite_frequency_is_refused(self, channel_ms, freq):
        """NaN walks through the range check rather than failing it.

        Every comparison against NaN is False, so ``offset > half a channel`` is
        False too and the request is accepted -- silently, on channel 0, which
        ``argmin`` returns for a NaN distance. It has to be caught before the
        band check rather than by it.
        """
        _, freqs, _ = channel_ms()

        with pytest.raises(ValueError, match="data.freq"):
            read_ms("fake.ms", freq=freq)

    @pytest.mark.parametrize(
        "freq", ["1400000000", [1.4e9, 1.41e9], True, 10**400]
    )
    def test_a_frequency_that_is_not_a_single_number_is_refused(self, channel_ms, freq):
        """A string or a list gets no further than the type check.

        Left to the arithmetic, a string raises TypeError out of numpy and a list
        raises an ambiguous-truth-value error, neither of which says which config
        key was wrong. Several channels are `chans`, not a list here.
        """
        _, freqs, _ = channel_ms()

        with pytest.raises(ValueError, match="data.freq"):
            read_ms("fake.ms", freq=freq)

    def test_an_infinite_frequency_is_refused_by_name(self, channel_ms):
        """Infinity would fail the band check on its own, so a bare
        ``pytest.raises(ValueError)`` here would pass with or without the type
        guard. What is asserted is therefore *which* rejection fires: the guard
        runs first, so the message names the key rather than the band.
        """
        _, freqs, _ = channel_ms()

        with pytest.raises(ValueError, match="data.freq"):
            read_ms("fake.ms", freq=float("inf"))

    def test_a_frequency_inside_the_edge_channel_is_accepted(self, channel_ms):
        """Half a channel past the last centre is still that channel."""
        _, freqs, widths = channel_ms()

        ms = read_ms("fake.ms", freq=float(freqs[-1] + 0.4 * widths[-1]))

        assert ms["n_freq"] == 1
        np.testing.assert_allclose(np.asarray(ms["freqs"]), freqs[-1:])

    def test_a_frequency_just_past_the_edge_channel_is_not(self, channel_ms):
        _, freqs, widths = channel_ms()

        with pytest.raises(ValueError, match="outside"):
            read_ms("fake.ms", freq=float(freqs[-1] + 0.6 * widths[-1]))

    def test_a_narrow_channel_gets_its_own_tolerance(self, channel_ms):
        """The band's first width says nothing about a narrow channel in the middle."""
        widths = np.array([4e6, 4e6, 1e5, 4e6, 4e6])
        _, freqs, _ = channel_ms(widths=widths)
        # Inside half of a 4 MHz channel, but far outside half of this 100 kHz one.
        asked = float(freqs[2] + 1e6)

        with pytest.raises(ValueError, match="outside"):
            read_ms("fake.ms", freq=asked)

    def test_an_explicit_channel_list_bypasses_the_range_check(self, channel_ms):
        """`chans` names channels by index; there is no frequency to be outside."""
        channel_ms()

        assert read_ms("fake.ms", chans=np.array([4]))["n_freq"] == 1


class TestReadMSNoiseAgreesWithTheBand:
    """A narrowed read must not turn a disagreeing SIGMA_SPECTRUM into a passing one."""

    def test_a_spectrum_that_misses_the_band_is_refused_even_when_narrowed(
        self, channel_ms, capsys
    ):
        """The column describes the MS's channel axis, not the slice being read.

        Validating after the narrowing let a 6-channel spectrum on 5-channel data
        pass the moment a single channel was selected -- the column and the data
        disagree about the observation either way, and selecting a channel does
        not settle which of them is right.
        """
        _, freqs, _ = channel_ms(spectrum_chans=N_FREQ_C + 1)

        ms = read_ms("fake.ms", freq=float(freqs[2]))

        out = capsys.readouterr().out
        assert "SIGMA_SPECTRUM" in out and "SIGMA" in out
        # Fell through to SIGMA, which has no channel axis.
        assert np.asarray(ms["noise"]).ndim == 1

    def test_the_same_mismatch_is_refused_on_a_full_read(self, channel_ms, capsys):
        channel_ms(spectrum_chans=N_FREQ_C + 1)

        ms = read_ms("fake.ms")

        assert "SIGMA_SPECTRUM" in capsys.readouterr().out
        assert np.asarray(ms["noise"]).ndim == 1

    def test_a_matching_spectrum_is_still_narrowed_and_used(self, channel_ms):
        _, freqs, _ = channel_ms()
        full = read_ms("fake.ms")

        one = read_ms("fake.ms", freq=float(freqs[2]))

        np.testing.assert_allclose(
            np.asarray(one["noise"]), np.asarray(full["noise"])[:, 2:3]
        )


class TestReadMSChannelSelectionExtra:

    def test_a_sigma_only_ms_still_selects(self, channel_ms):
        """SIGMA has no channel axis, so selection must leave it per baseline."""
        channel_ms(sigma_spectrum=False)

        ms = read_ms("fake.ms", freq=1.4e9)

        assert ms["n_freq"] == 1
        assert ms["noise"] is None or np.asarray(ms["noise"]).ndim == 1
