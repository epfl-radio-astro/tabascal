"""Tests for tabascal.noise — per-baseline visibility noise."""

from types import SimpleNamespace

import numpy as np
import pytest

from tabascal.noise import (
    NoUsableSigma,
    broadcast_to_vis,
    per_baseline_freq_sigma,
    per_baseline_sigma,
    read_noise_file,
    representative_sigma,
)


N_TIME, N_BL, N_FREQ = 5, 6, 3


def _sigma_column(per_bl, n_time=N_TIME, n_corr=1):
    """A SIGMA column: per-baseline values repeated over time, as an MS stores it."""

    per_bl = np.asarray(per_bl, dtype=float)

    return np.tile(per_bl, (n_time, 1)).reshape(-1, 1).repeat(n_corr, axis=1)


def _sigma_spectrum(per_bl_freq, n_time=N_TIME, n_corr=1):
    """A SIGMA_SPECTRUM column, ``(row, chan, corr)``.

    Per-(baseline, channel) values repeated over time, laid out time-major --
    row ``t * n_bl + b`` -- exactly as the visibility rows are.
    """

    per_bl_freq = np.asarray(per_bl_freq, dtype=float)
    n_bl, n_freq = per_bl_freq.shape
    rows = np.tile(per_bl_freq, (n_time, 1, 1)).reshape(n_time * n_bl, n_freq)

    return np.repeat(rows[:, :, None], n_corr, axis=2)


def _sigma_column_time(per_bl_time, n_corr=1):
    """A SIGMA column that changes over time, ``(n_bl, n_time)`` in, rows out.

    Row ``t * n_bl + b`` holds baseline ``b`` at time ``t``, the time-major
    layout the visibility rows use.
    """

    per_bl_time = np.asarray(per_bl_time, dtype=float)

    return np.repeat(per_bl_time.T.reshape(-1, 1), n_corr, axis=1)


def _sigma_spectrum_time(per_bl_freq_time, n_corr=1):
    """A SIGMA_SPECTRUM column that changes over time, ``(n_bl, n_freq, n_time)`` in."""

    arr = np.asarray(per_bl_freq_time, dtype=float)
    rows = arr.transpose(2, 0, 1).reshape(-1, arr.shape[1])

    return np.repeat(rows[:, :, None], n_corr, axis=2)


# ---------------------------------------------------------------------------
# per_baseline_sigma
# ---------------------------------------------------------------------------

class TestPerBaselineSigma:

    def test_recovers_the_per_baseline_values(self):
        per_bl = np.array([0.5, 1.0, 2.0, 1.5, 0.8, 3.0])

        out = per_baseline_sigma(_sigma_column(per_bl), N_TIME, N_BL)

        np.testing.assert_allclose(out, per_bl)

    def test_a_uniform_sigma_reproduces_the_scalar(self):
        """The simulated benchmark data is uniform, so this must be a no-op there."""
        per_bl = np.full(N_BL, 0.6496226)

        out = per_baseline_sigma(_sigma_column(per_bl), N_TIME, N_BL)

        np.testing.assert_allclose(out, 0.6496226)

    def test_a_finite_outlier_row_is_taken_at_face_value(self):
        """A positive, finite row is a measurement, whatever it looks like.

        Nothing distinguishes a corrupted row from a timestep on which the noise
        really was 1000x higher -- a passing transmitter, a dropped correlator
        block -- so it is not median-ed away: the column varies in time, and the
        time-resolved read keeps every row as written.
        """
        col = _sigma_column(np.full(N_BL, 1.0))
        col = col.reshape(N_TIME, N_BL, 1)
        col[0, :, 0] = 1000.0  # one loud timestep
        col = col.reshape(-1, 1)

        out = per_baseline_sigma(col, N_TIME, N_BL)

        assert out.shape == (N_BL, 1, N_TIME)
        np.testing.assert_allclose(out[:, 0, 0], 1000.0)
        np.testing.assert_allclose(out[:, 0, 1:], 1.0)

    def test_one_nan_row_does_not_destroy_a_baseline(self):
        """A median over a column holding a NaN is a NaN.

        Filtering only *after* the reduction throws away the four timesteps that
        were measured perfectly well, and hands a loud baseline the median of the
        quiet ones -- [10, 10, 10, 10, NaN] should be 10, not the global ~1.
        """
        per_bl = np.full(N_BL, 1.0)
        per_bl[0] = 10.0
        col = _sigma_column(per_bl).reshape(N_TIME, N_BL, 1)
        col[-1, 0, 0] = np.nan
        col = col.reshape(-1, 1)

        assert per_baseline_sigma(col, N_TIME, N_BL)[0] == pytest.approx(10.0)

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
    def test_the_measured_rows_survive_a_majority_of_bad_ones(self, bad):
        """The reduction runs over the entries that carry an estimate, only.

        With most of a baseline's rows corrupted the median of the raw column is
        itself invalid whichever way it is corrupted -- zero, negative, NaN or
        infinite -- so the baseline is filled from the others despite having been
        measured three times over.
        """
        per_bl = np.full(N_BL, 1.0)
        per_bl[0] = 10.0
        col = _sigma_column(per_bl).reshape(N_TIME, N_BL, 1)
        col[:3, 0, 0] = bad
        col = col.reshape(-1, 1)

        assert per_baseline_sigma(col, N_TIME, N_BL)[0] == pytest.approx(10.0)

    def test_a_dead_baseline_does_not_leak_a_numpy_warning(self, recwarn):
        """Dropping every timestep of a dead baseline is expected, not a fault.

        The fill message below already says which baselines had no valid value
        and what was done about it; numpy's "All-NaN slice" on top of it is noise
        about an internal choice of reduction.
        """
        per_bl = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan])

        per_baseline_sigma(_sigma_column(per_bl), N_TIME, N_BL)

        assert [w for w in recwarn if issubclass(w.category, RuntimeWarning)] == []

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
    def test_dead_baselines_take_the_median_of_the_rest(self, bad, capsys):
        """A non-positive or non-finite sigma carries no information.

        Zeroes would divide the likelihood by nothing; these baselines are
        flagged out of it anyway.
        """
        per_bl = np.array([1.0, 2.0, 3.0, 4.0, 5.0, bad])

        out = per_baseline_sigma(_sigma_column(per_bl), N_TIME, N_BL)

        assert np.isfinite(out).all() and (out > 0).all()
        assert out[-1] == pytest.approx(np.median([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert "no valid SIGMA" in capsys.readouterr().out

    def test_all_baselines_dead_is_an_error(self):
        per_bl = np.zeros(N_BL)

        with pytest.raises(ValueError, match="No baseline has a positive"):
            per_baseline_sigma(_sigma_column(per_bl), N_TIME, N_BL)

    def test_row_count_mismatch_is_an_error(self):
        with pytest.raises(ValueError, match="does not match the observation grid"):
            per_baseline_sigma(np.ones((7, 1)), N_TIME, N_BL)

    def test_accepts_a_one_dimensional_column(self):
        per_bl = np.arange(1, N_BL + 1, dtype=float)
        col = _sigma_column(per_bl).ravel()

        np.testing.assert_allclose(per_baseline_sigma(col, N_TIME, N_BL), per_bl)

    def test_selects_the_requested_correlation(self):
        per_bl = np.arange(1, N_BL + 1, dtype=float)
        col = np.stack(
            [_sigma_column(per_bl)[:, 0], _sigma_column(per_bl * 10)[:, 0]], axis=1
        )

        np.testing.assert_allclose(per_baseline_sigma(col, N_TIME, N_BL, 0), per_bl)
        np.testing.assert_allclose(
            per_baseline_sigma(col, N_TIME, N_BL, 1), per_bl * 10
        )

    def test_a_single_correlation_column_is_used_with_a_warning(self, capsys):
        """One correlation, one answer -- but the two layouts disagree, so say so.

        Some writers collapse the noise column's correlation axis, so the single
        column is still what gets used. It is not a *quiet* case though: in every
        valid single-polarisation MS the data resolves to correlation 0 too, so a
        higher index means the noise column and the POLARIZATION row the data was
        resolved against describe different layouts.
        """
        per_bl = np.arange(1, N_BL + 1, dtype=float)

        out = per_baseline_sigma(_sigma_column(per_bl), N_TIME, N_BL, corr_idx=3)

        np.testing.assert_allclose(out, per_bl)
        captured = capsys.readouterr().out
        assert "3" in captured and "SIGMA" in captured
        assert "1 correlation" in captured

    def test_the_ordinary_single_correlation_read_says_nothing(self, capsys):
        """A single-polarisation MS resolves to correlation 0: no disagreement."""
        per_bl = np.arange(1, N_BL + 1, dtype=float)

        per_baseline_sigma(_sigma_column(per_bl), N_TIME, N_BL, corr_idx=0)

        assert capsys.readouterr().out == ""

    def test_an_index_off_the_correlation_axis_is_an_error(self):
        """Silently reading correlation 0 instead would weight the fit by another
        polarisation's noise, with nothing said."""
        col = _sigma_column(np.ones(N_BL), n_corr=2)

        with pytest.raises(ValueError, match="Correlation index 3"):
            per_baseline_sigma(col, N_TIME, N_BL, corr_idx=3)

    def test_the_correlation_error_names_the_column_shape(self):
        col = _sigma_column(np.ones(N_BL), n_corr=2)

        with pytest.raises(ValueError, match=r"SIGMA.*shape \(30, 2\)"):
            per_baseline_sigma(col, N_TIME, N_BL, corr_idx=3)


# ---------------------------------------------------------------------------
# per_baseline_freq_sigma
# ---------------------------------------------------------------------------

class TestPerBaselineFreqSigma:
    """SIGMA_SPECTRUM: noise that varies over the band as well as over baselines.

    A real bandpass is not flat -- the edge channels of a subband are noisier
    than its centre -- so a per-baseline number is as wrong across frequency as a
    scalar is across baselines.
    """

    def _grid(self):
        return np.arange(1, N_BL * N_FREQ + 1, dtype=float).reshape(N_BL, N_FREQ)

    def test_recovers_the_per_baseline_channel_values(self):
        per_bl_freq = self._grid()

        out = per_baseline_freq_sigma(_sigma_spectrum(per_bl_freq), N_TIME, N_BL)

        assert out.shape == (N_BL, N_FREQ)
        np.testing.assert_allclose(out, per_bl_freq)

    def test_channel_structure_is_kept_rather_than_averaged_away(self):
        """The noisy band edge must stay noisy for the channels it belongs to."""
        per_bl_freq = np.ones((N_BL, N_FREQ))
        per_bl_freq[:, 0] = 10.0

        out = per_baseline_freq_sigma(_sigma_spectrum(per_bl_freq), N_TIME, N_BL)

        np.testing.assert_allclose(out[:, 0], 10.0)
        np.testing.assert_allclose(out[:, 1:], 1.0)

    def test_a_finite_outlier_timestep_is_taken_at_face_value(self):
        """As in SIGMA: a positive, finite row is a measurement, not a fault.

        The column says the noise on that timestep was 1000x the rest, and
        nothing here can tell that from a real one, so the time axis is kept
        rather than the row median-ed away.
        """
        col = _sigma_spectrum(np.ones((N_BL, N_FREQ)))
        col = col.reshape(N_TIME, N_BL, N_FREQ, 1)
        col[0] = 1000.0
        col = col.reshape(N_TIME * N_BL, N_FREQ, 1)

        out = per_baseline_freq_sigma(col, N_TIME, N_BL)

        assert out.shape == (N_BL, N_FREQ, N_TIME)
        np.testing.assert_allclose(out[:, :, 0], 1000.0)
        np.testing.assert_allclose(out[:, :, 1:], 1.0)

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
    def test_a_cell_keeps_the_rows_that_measured_it(self, bad):
        """Invalid rows are dropped before the reduction, not after it.

        Otherwise a cell whose noise was measured on the timesteps that worked is
        handed the median of every other cell in the column -- the one thing this
        function exists to avoid doing to a well-measured cell.
        """
        per_bl_freq = np.ones((N_BL, N_FREQ))
        per_bl_freq[2, 1] = 10.0
        col = _sigma_spectrum(per_bl_freq).reshape(N_TIME, N_BL, N_FREQ, 1)
        col[:3, 2, 1, 0] = bad
        col = col.reshape(N_TIME * N_BL, N_FREQ, 1)

        assert per_baseline_freq_sigma(col, N_TIME, N_BL)[2, 1] == pytest.approx(10.0)

    def test_a_dead_cell_does_not_leak_a_numpy_warning(self, recwarn):
        """The fill message says what happened; numpy's "All-NaN slice" adds
        nothing but an internal choice of reduction."""
        per_bl_freq = self._grid()
        per_bl_freq[2, 1] = np.nan

        per_baseline_freq_sigma(_sigma_spectrum(per_bl_freq), N_TIME, N_BL)

        assert [w for w in recwarn if issubclass(w.category, RuntimeWarning)] == []

    def test_selects_the_requested_correlation(self):
        per_bl_freq = self._grid()
        col = np.concatenate(
            [
                _sigma_spectrum(per_bl_freq),
                _sigma_spectrum(per_bl_freq * 10),
            ],
            axis=2,
        )

        np.testing.assert_allclose(
            per_baseline_freq_sigma(col, N_TIME, N_BL, 0), per_bl_freq
        )
        np.testing.assert_allclose(
            per_baseline_freq_sigma(col, N_TIME, N_BL, 1), per_bl_freq * 10
        )

    def test_an_index_off_the_correlation_axis_is_an_error(self):
        col = _sigma_spectrum(np.ones((N_BL, N_FREQ)), n_corr=2)

        with pytest.raises(ValueError, match="Correlation index 3"):
            per_baseline_freq_sigma(col, N_TIME, N_BL, corr_idx=3)

    def test_a_single_correlation_column_is_used_with_a_warning(self, capsys):
        """Used, because some writers collapse the noise correlation axis;
        warned about, because in a valid single-polarisation MS the data resolves
        to correlation 0 as well, so a higher index is the two disagreeing."""
        per_bl_freq = self._grid()

        out = per_baseline_freq_sigma(
            _sigma_spectrum(per_bl_freq), N_TIME, N_BL, corr_idx=3
        )

        np.testing.assert_allclose(out, per_bl_freq)
        captured = capsys.readouterr().out
        assert "3" in captured and "SIGMA_SPECTRUM" in captured
        assert "1 correlation" in captured

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
    def test_dead_cells_take_the_median_of_the_valid_ones(self, bad, capsys):
        """A flagged channel on one baseline carries no noise estimate.

        Per cell, not per baseline: a baseline with one dead channel keeps its
        own values on the channels that are alive.
        """
        per_bl_freq = self._grid()
        per_bl_freq[2, 1] = bad

        out = per_baseline_freq_sigma(_sigma_spectrum(per_bl_freq), N_TIME, N_BL)

        assert np.isfinite(out).all() and (out > 0).all()
        valid = np.delete(per_bl_freq.ravel(), 2 * N_FREQ + 1)
        assert out[2, 1] == pytest.approx(np.median(valid))
        np.testing.assert_allclose(out[2, 0], per_bl_freq[2, 0])
        assert "no valid SIGMA_SPECTRUM" in capsys.readouterr().out

    def test_an_entirely_invalid_column_is_signalled_for_fallback(self):
        """Not usable, but not malformed either: the caller falls back to SIGMA.

        A ValueError subclass, so a caller that only knows about ValueError still
        sees it, and the message still says what to do.
        """
        col = _sigma_spectrum(np.zeros((N_BL, N_FREQ)))

        with pytest.raises(NoUsableSigma, match="set data.noise explicitly"):
            per_baseline_freq_sigma(col, N_TIME, N_BL)

        assert issubclass(NoUsableSigma, ValueError)

    def test_row_count_mismatch_is_an_error(self):
        """A malformed column, not an empty one -- this must not be fallen back
        through: it means the reader and the MS disagree about the grid."""
        col = _sigma_spectrum(np.ones((N_BL, N_FREQ)))[:-1]

        with pytest.raises(ValueError, match="does not match the observation grid"):
            per_baseline_freq_sigma(col, N_TIME, N_BL)
        with pytest.raises(ValueError) as err:
            per_baseline_freq_sigma(col, N_TIME, N_BL)
        assert not isinstance(err.value, NoUsableSigma)

    def test_accepts_a_column_without_a_correlation_axis(self):
        per_bl_freq = self._grid()
        col = _sigma_spectrum(per_bl_freq)[:, :, 0]

        np.testing.assert_allclose(
            per_baseline_freq_sigma(col, N_TIME, N_BL), per_bl_freq
        )

    def test_a_one_dimensional_column_is_rejected(self):
        """That is a SIGMA column; it has no channel axis to spread over."""
        with pytest.raises(ValueError, match="n_row, n_chan"):
            per_baseline_freq_sigma(np.ones(N_TIME * N_BL), N_TIME, N_BL)


# ---------------------------------------------------------------------------
# Noise that varies over time
# ---------------------------------------------------------------------------

class TestTimeVaryingSigma:
    """A SIGMA that really does change over time is kept, not averaged away.

    A column constant in time is one measurement written into every row, and
    collapsing it costs nothing. A column that changes is the MS saying the noise
    changed -- a re-weighted correlator dump, a chunk of the observation the
    array was half-flagged for -- and a median over it hands every timestep a
    noise that belongs to none of them.
    """

    def _varying(self):
        """Per-baseline noise ramping over time: no two timesteps alike."""

        per_bl = np.arange(1, N_BL + 1, dtype=float)

        return per_bl[:, None] * (1.0 + np.arange(N_TIME, dtype=float))[None, :]

    def test_the_time_axis_is_kept(self):
        per_bl_time = self._varying()

        out = per_baseline_sigma(_sigma_column_time(per_bl_time), N_TIME, N_BL)

        assert out.shape == (N_BL, 1, N_TIME)
        np.testing.assert_allclose(out[:, 0, :], per_bl_time)

    def test_the_frequency_axis_is_a_length_one_placeholder(self):
        """Never a bare ``(n_bl, n_time)``.

        Whenever n_freq == n_time that shape is indistinguishable from
        ``(n_bl, n_freq)``, and every consumer would weight the visibilities by
        the wrong axis without a word. The placeholder axis says which is which.
        """
        out = per_baseline_sigma(_sigma_column_time(self._varying()), N_TIME, N_BL)

        assert out.ndim == 3 and out.shape[1] == 1

    def test_the_detection_is_said_out_loud(self, capsys):
        """The reader is deciding between two shapes; the run should say which."""
        per_baseline_sigma(_sigma_column_time(self._varying()), N_TIME, N_BL)

        captured = capsys.readouterr().out
        assert "SIGMA" in captured and "time-resolved" in captured
        assert "(n_bl, 1, n_time)" in captured

    def test_a_constant_column_collapses_exactly_as_before(self, capsys):
        """The simulated benchmark writes a uniform SIGMA: that path is untouched."""
        per_bl = np.array([0.5, 1.0, 2.0, 1.5, 0.8, 3.0])

        out = per_baseline_sigma(_sigma_column(per_bl), N_TIME, N_BL)

        assert out.shape == (N_BL,)
        np.testing.assert_allclose(out, per_bl)
        assert capsys.readouterr().out == ""

    def test_one_baseline_changing_makes_the_whole_column_time_resolved(self):
        """Detected per cell, answered column-wide: the noise is one array, so
        one baseline that changes gives every baseline a time axis."""
        per_bl_time = np.tile(
            np.arange(1, N_BL + 1, dtype=float)[:, None], (1, N_TIME)
        )
        per_bl_time[2, 3] = 99.0

        out = per_baseline_sigma(_sigma_column_time(per_bl_time), N_TIME, N_BL)

        assert out.shape == (N_BL, 1, N_TIME)
        np.testing.assert_allclose(out[:, 0, :], per_bl_time)

    def test_constancy_is_exact_equality_not_a_tolerance(self):
        """An MS writing a constant noise writes the identical value every row.

        Anything else is a column that says the noise changed, and no threshold
        can tell a corrupted row from a real change -- so none is invented, and
        a difference in the last bit is still a difference.
        """
        per_bl_time = np.tile(np.full(N_BL, 1.0)[:, None], (1, N_TIME))
        per_bl_time[0, 1] = np.nextafter(1.0, 2.0)

        out = per_baseline_sigma(_sigma_column_time(per_bl_time), N_TIME, N_BL)

        assert out.shape == (N_BL, 1, N_TIME)

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
    def test_an_unmeasured_entry_takes_its_own_timeline_median(self, bad, capsys):
        """Its own cell's median over the timesteps that measured it.

        The same philosophy as the constant path one axis in: an entry that
        measured nothing is filled from the measurements nearest it, which are
        that baseline's own, not from the median of every other baseline.
        """
        per_bl_time = self._varying()
        per_bl_time[0, 0] = bad

        out = per_baseline_sigma(_sigma_column_time(per_bl_time), N_TIME, N_BL)

        # Baseline 0 measured [_, 2, 3, 4, 5]; the gap takes their median.
        assert out[0, 0, 0] == pytest.approx(np.median([2.0, 3.0, 4.0, 5.0]))
        np.testing.assert_allclose(out[0, 0, 1:], per_bl_time[0, 1:])
        assert "no valid SIGMA" in capsys.readouterr().out

    def test_a_baseline_that_measured_nothing_takes_the_global_median(self, capsys):
        """Nothing of its own to fall back on, so the median of the cells that
        do have a timeline -- and never a zero, which divides by nothing."""
        per_bl_time = self._varying()
        per_bl_time[3, :] = 0.0

        out = per_baseline_sigma(_sigma_column_time(per_bl_time), N_TIME, N_BL)

        # Baseline b measured (b + 1) * [1..5], so its timeline median is 3(b+1);
        # the dead baseline takes the median over the rest of those.
        timelines = [3.0 * (b + 1) for b in range(N_BL) if b != 3]
        np.testing.assert_allclose(out[3, 0, :], np.median(timelines))
        assert np.isfinite(out).all() and (out > 0).all()
        assert "no valid SIGMA" in capsys.readouterr().out

    def test_a_column_with_nothing_valid_in_it_is_still_NoUsableSigma(self):
        """The fallthrough the caller reads to try the next column is unchanged:
        a column of rubbish varies in time too, and is still not a noise."""
        per_bl_time = -self._varying()

        with pytest.raises(NoUsableSigma, match="No baseline has a positive"):
            per_baseline_sigma(_sigma_column_time(per_bl_time), N_TIME, N_BL)


class TestTimeVaryingSigmaSpectrum:
    """The same over the band: ``(n_bl, n_freq, n_time)`` when the column varies."""

    def _varying(self):
        """A distinct value in every (baseline, channel, time) cell."""

        return 1.0 + np.arange(N_BL * N_FREQ * N_TIME, dtype=float).reshape(
            N_BL, N_FREQ, N_TIME
        )

    def test_the_time_axis_is_kept(self):
        per_bl_freq_time = self._varying()

        out = per_baseline_freq_sigma(
            _sigma_spectrum_time(per_bl_freq_time), N_TIME, N_BL
        )

        assert out.shape == (N_BL, N_FREQ, N_TIME)
        np.testing.assert_allclose(out, per_bl_freq_time)

    def test_the_detection_is_said_out_loud(self, capsys):
        per_baseline_freq_sigma(
            _sigma_spectrum_time(self._varying()), N_TIME, N_BL
        )

        captured = capsys.readouterr().out
        assert "SIGMA_SPECTRUM" in captured and "time-resolved" in captured
        assert "(n_bl, n_freq, n_time)" in captured

    def test_a_constant_column_collapses_exactly_as_before(self, capsys):
        per_bl_freq = np.arange(1, N_BL * N_FREQ + 1, dtype=float).reshape(
            N_BL, N_FREQ
        )

        out = per_baseline_freq_sigma(_sigma_spectrum(per_bl_freq), N_TIME, N_BL)

        assert out.shape == (N_BL, N_FREQ)
        np.testing.assert_allclose(out, per_bl_freq)
        assert capsys.readouterr().out == ""

    def test_one_changing_cell_makes_the_whole_column_time_resolved(self):
        """Mixed columns are the common case -- one channel re-weighted on one
        baseline -- and the answer is one array, so the constant cells simply
        repeat over the time axis the changing one needs."""
        per_bl_freq = np.arange(1, N_BL * N_FREQ + 1, dtype=float).reshape(
            N_BL, N_FREQ
        )
        per_bl_freq_time = np.repeat(per_bl_freq[:, :, None], N_TIME, axis=2)
        per_bl_freq_time[2, 1, 3] = 500.0

        out = per_baseline_freq_sigma(
            _sigma_spectrum_time(per_bl_freq_time), N_TIME, N_BL
        )

        assert out.shape == (N_BL, N_FREQ, N_TIME)
        np.testing.assert_allclose(out, per_bl_freq_time)

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
    def test_an_unmeasured_entry_takes_its_own_cells_timeline_median(self, bad):
        """Per cell, not per baseline and not per timestep: a cell that lost one
        timestep keeps its own noise on the ones it measured."""
        per_bl_freq_time = self._varying()
        expected = np.median(per_bl_freq_time[2, 1, 1:])
        per_bl_freq_time[2, 1, 0] = bad

        out = per_baseline_freq_sigma(
            _sigma_spectrum_time(per_bl_freq_time), N_TIME, N_BL
        )

        assert out[2, 1, 0] == pytest.approx(expected)
        np.testing.assert_allclose(out[2, 1, 1:], per_bl_freq_time[2, 1, 1:])

    def test_a_cell_that_measured_nothing_takes_the_global_median(self, capsys):
        per_bl_freq_time = self._varying()
        cell_medians = np.median(per_bl_freq_time, axis=2)
        per_bl_freq_time[2, 1, :] = 0.0
        expected = np.median(np.delete(cell_medians.ravel(), 2 * N_FREQ + 1))

        out = per_baseline_freq_sigma(
            _sigma_spectrum_time(per_bl_freq_time), N_TIME, N_BL
        )

        np.testing.assert_allclose(out[2, 1, :], expected)
        assert np.isfinite(out).all() and (out > 0).all()
        assert "no valid SIGMA_SPECTRUM" in capsys.readouterr().out

    def test_a_column_with_nothing_valid_in_it_is_still_NoUsableSigma(self):
        """What ``partition_noise`` falls through to SIGMA on."""
        with pytest.raises(NoUsableSigma, match="set data.noise explicitly"):
            per_baseline_freq_sigma(
                _sigma_spectrum_time(-self._varying()), N_TIME, N_BL
            )


# ---------------------------------------------------------------------------
# broadcast_to_vis
# ---------------------------------------------------------------------------

class TestBroadcastToVis:

    SHAPE = (N_BL, 3, 4)  # (n_bl, n_freq, n_time)

    def test_scalar_passes_through(self):
        assert broadcast_to_vis(0.5, self.SHAPE) == 0.5

    def test_per_baseline_gains_trailing_axes(self):
        noise = np.arange(1, N_BL + 1, dtype=float)

        out = broadcast_to_vis(noise, self.SHAPE)

        assert out.shape == (N_BL, 1, 1)
        assert np.broadcast_to(out, self.SHAPE).shape == self.SHAPE

    def test_each_baseline_keeps_its_own_value(self):
        """The point: baseline i must divide by noise[i], not by someone else's."""
        noise = np.arange(1, N_BL + 1, dtype=float)

        out = np.broadcast_to(broadcast_to_vis(noise, self.SHAPE), self.SHAPE)

        for i in range(N_BL):
            assert np.all(out[i] == noise[i])

    def test_wrong_length_is_an_error(self):
        with pytest.raises(ValueError, match="but the visibilities have"):
            broadcast_to_vis(np.ones(N_BL + 1), self.SHAPE)

    def test_a_full_shape_array_passes_through(self):
        noise = np.ones(self.SHAPE)

        assert broadcast_to_vis(noise, self.SHAPE).shape == self.SHAPE

    def test_a_mismatched_full_array_is_an_error(self):
        with pytest.raises(ValueError, match="cannot be broadcast"):
            broadcast_to_vis(np.ones((N_BL, 9, 9)), self.SHAPE)

    def test_per_baseline_frequency_gains_a_time_axis(self):
        noise = np.ones((N_BL, self.SHAPE[1]))

        out = broadcast_to_vis(noise, self.SHAPE)

        assert out.shape == (N_BL, self.SHAPE[1], 1)
        assert np.broadcast_to(out, self.SHAPE).shape == self.SHAPE

    def test_each_baseline_channel_cell_keeps_its_own_value(self):
        """A transposed or flattened reshape would still have the right size."""
        n_freq = self.SHAPE[1]
        noise = np.arange(1, N_BL * n_freq + 1, dtype=float).reshape(N_BL, n_freq)

        out = np.broadcast_to(broadcast_to_vis(noise, self.SHAPE), self.SHAPE)

        for i in range(N_BL):
            for f in range(n_freq):
                assert np.all(out[i, f] == noise[i, f])

    @pytest.mark.parametrize("shape", [(N_BL + 1, 3), (N_BL, 4), (3, N_BL)])
    def test_a_mismatched_baseline_frequency_array_is_an_error(self, shape):
        with pytest.raises(ValueError, match="baselines and 3 channels"):
            broadcast_to_vis(np.ones(shape), self.SHAPE)

    def test_a_time_resolved_per_baseline_noise_is_accepted(self):
        """``(n_bl, 1, n_time)``: what a time-varying SIGMA reads as."""
        noise = np.ones((N_BL, 1, self.SHAPE[2]))

        out = broadcast_to_vis(noise, self.SHAPE)

        assert out.shape == (N_BL, 1, self.SHAPE[2])
        assert np.broadcast_to(out, self.SHAPE).shape == self.SHAPE

    def test_each_baseline_timestep_keeps_its_own_value(self):
        """The point of the time axis: timestep t must divide by noise[..., t]."""
        n_time = self.SHAPE[2]
        noise = np.arange(1, N_BL * n_time + 1, dtype=float).reshape(
            N_BL, 1, n_time
        )

        out = np.broadcast_to(broadcast_to_vis(noise, self.SHAPE), self.SHAPE)

        for i in range(N_BL):
            for t in range(n_time):
                assert np.all(out[i, :, t] == noise[i, 0, t])

    def test_a_fully_resolved_noise_is_accepted(self):
        """``(n_bl, n_freq, n_time)``: a time-varying SIGMA_SPECTRUM."""
        noise = np.arange(1, np.prod(self.SHAPE) + 1, dtype=float).reshape(self.SHAPE)

        np.testing.assert_array_equal(broadcast_to_vis(noise, self.SHAPE), noise)

    @pytest.mark.parametrize(
        "shape", [(N_BL, 2, 4), (N_BL, 3, 5), (N_BL + 1, 1, 4), (3, 1, N_BL)]
    )
    def test_a_three_dimensional_noise_with_a_wrong_axis_is_an_error(self, shape):
        """Checked, not numpy's own broadcast: every axis must match the
        visibilities or be 1. ``(3, 1, N_BL)`` is the transposed grid -- the
        right size, the wrong meaning -- which numpy would refuse only by luck."""
        with pytest.raises(ValueError, match="cannot be broadcast"):
            broadcast_to_vis(np.ones(shape), self.SHAPE)

    def test_a_length_one_axis_is_a_noise_shared_along_it(self):
        """1 means "the same for every one of them", on any axis: that is what
        makes ``(n_bl, 1, n_time)`` mean per baseline and timestep."""
        noise = np.arange(1, self.SHAPE[2] + 1, dtype=float).reshape(1, 1, -1)

        out = np.broadcast_to(broadcast_to_vis(noise, self.SHAPE), self.SHAPE)

        for t in range(self.SHAPE[2]):
            assert np.all(out[:, :, t] == noise[0, 0, t])

    def test_the_three_dimensional_error_names_both_shapes(self):
        with pytest.raises(
            ValueError, match=r"shape \(6, 2, 4\).*shape \(6, 3, 4\)"
        ):
            broadcast_to_vis(np.ones((N_BL, 2, 4)), self.SHAPE)

    def test_a_two_dimensional_array_is_read_as_baseline_and_channel(self):
        """Which is why a time-resolved SIGMA carries a length-1 channel axis.

        With n_freq == n_time nothing in a ``(n_bl, n_time)`` shape says which
        axis it is, and this is the reading it gets: by channel. A reader that
        returned the bare two-dimensional form would have every timestep weighted
        by another channel's noise, silently.
        """
        square = (N_BL, 4, 4)
        noise = np.arange(1, N_BL * 4 + 1, dtype=float).reshape(N_BL, 4)

        out = np.broadcast_to(broadcast_to_vis(noise, square), square)

        for f in range(4):
            assert np.all(out[:, f, :] == noise[:, f][:, None])


# ---------------------------------------------------------------------------
# representative_sigma
# ---------------------------------------------------------------------------

class TestRepresentativeSigma:

    def test_is_the_median(self):
        assert representative_sigma([1.0, 2.0, 100.0]) == 2.0

    def test_uniform_input_returns_that_value(self):
        assert representative_sigma(np.full(5, 0.7)) == pytest.approx(0.7)

    def test_returns_a_python_float(self):
        assert isinstance(representative_sigma(np.ones(3)), float)

    def test_a_baseline_frequency_noise_reduces_over_every_cell(self):
        """The heuristics want one number whether the noise varies over the band
        or not, so the (n_bl, n_freq) case must not need its own call site."""
        sigma = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 100.0]])

        assert representative_sigma(sigma) == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# read_noise_file
# ---------------------------------------------------------------------------

class TestReadNoiseFile:

    def test_reads_sigma_bl(self, tmp_path):
        path = tmp_path / "noise.npz"
        sigma_bl = np.arange(1, N_BL + 1, dtype=float)
        np.savez(path, sigma_bl=sigma_bl)

        np.testing.assert_allclose(read_noise_file(str(path), N_BL), sigma_bl)

    def test_sigma_bl_length_is_checked(self, tmp_path):
        path = tmp_path / "noise.npz"
        np.savez(path, sigma_bl=np.ones(N_BL + 2))

        with pytest.raises(ValueError, match="but the observation has"):
            read_noise_file(str(path), N_BL)

    def test_combines_per_antenna_noise(self, tmp_path):
        path = tmp_path / "noise.npz"
        s_ant = np.array([1.0, 2.0, 3.0, 4.0])
        np.savez(path, s_ant=s_ant)
        a1, a2 = np.triu_indices(4, k=1)

        out = read_noise_file(str(path), len(a1), a1, a2)

        expected = np.sqrt(s_ant[a1] ** 2 + s_ant[a2] ** 2) / np.sqrt(2)
        np.testing.assert_allclose(out, expected)

    def test_uniform_per_antenna_noise_reproduces_itself(self, tmp_path):
        """The sqrt(2) normalisation: equal antennas give that same baseline noise."""
        path = tmp_path / "noise.npz"
        np.savez(path, s_ant=np.full(4, 0.75))
        a1, a2 = np.triu_indices(4, k=1)

        np.testing.assert_allclose(read_noise_file(str(path), len(a1), a1, a2), 0.75)

    def test_per_antenna_without_pairs_is_an_error(self, tmp_path):
        path = tmp_path / "noise.npz"
        np.savez(path, s_ant=np.ones(4))

        with pytest.raises(ValueError, match="needs the antenna pairs"):
            read_noise_file(str(path), N_BL)

    def test_unknown_keys_are_rejected_by_name(self, tmp_path):
        path = tmp_path / "noise.npz"
        np.savez(path, something_else=np.ones(3))

        with pytest.raises(ValueError, match="must carry one of"):
            read_noise_file(str(path), N_BL)

    def test_reads_sigma_bl_freq(self, tmp_path):
        path = tmp_path / "noise.npz"
        sigma = np.arange(1, N_BL * N_FREQ + 1, dtype=float).reshape(N_BL, N_FREQ)
        np.savez(path, sigma_bl_freq=sigma)

        out = read_noise_file(str(path), N_BL, n_freq=N_FREQ)

        assert out.shape == (N_BL, N_FREQ)
        np.testing.assert_allclose(out, sigma)

    def test_sigma_bl_freq_shape_is_checked(self, tmp_path):
        """Both axes: an (n_freq, n_bl) file has the right size and the wrong
        meaning, and would silently mis-weight every visibility."""
        path = tmp_path / "noise.npz"
        np.savez(path, sigma_bl_freq=np.ones((N_FREQ, N_BL)))

        with pytest.raises(ValueError, match=r"has shape \(3, 6\)"):
            read_noise_file(str(path), N_BL, n_freq=N_FREQ)

    def test_sigma_bl_freq_needs_the_channel_count(self, tmp_path):
        """Nothing to validate the frequency axis against is not a licence to
        skip validating it."""
        path = tmp_path / "noise.npz"
        np.savez(path, sigma_bl_freq=np.ones((N_BL, N_FREQ)))

        with pytest.raises(ValueError, match="needs the number of channels"):
            read_noise_file(str(path), N_BL)

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
    def test_an_invalid_sigma_bl_is_rejected_rather_than_filled(self, tmp_path, bad):
        """An override is used as given, so it cannot be quietly repaired.

        The median fill is for MS columns, where a dead baseline is a fact about
        the instrument. Here the file *is* the user's statement of the noise, and
        inventing a value for part of it would answer a question they were in the
        middle of answering themselves.
        """
        path = tmp_path / "noise.npz"
        sigma_bl = np.arange(1, N_BL + 1, dtype=float)
        sigma_bl[2] = bad
        np.savez(path, sigma_bl=sigma_bl)

        with pytest.raises(ValueError, match="sigma_bl"):
            read_noise_file(str(path), N_BL)

    def test_the_rejection_names_the_offending_count(self, tmp_path):
        path = tmp_path / "noise.npz"
        np.savez(path, sigma_bl=np.array([1.0, 0.0, 3.0, -1.0, np.nan, 6.0]))

        with pytest.raises(ValueError, match="3 of 6"):
            read_noise_file(str(path), N_BL)

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
    def test_an_invalid_sigma_bl_freq_is_rejected(self, tmp_path, bad):
        path = tmp_path / "noise.npz"
        sigma = np.ones((N_BL, N_FREQ))
        sigma[1, 1] = bad
        np.savez(path, sigma_bl_freq=sigma)

        with pytest.raises(ValueError, match="sigma_bl_freq"):
            read_noise_file(str(path), N_BL, n_freq=N_FREQ)

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
    def test_an_invalid_s_ant_is_rejected(self, tmp_path, bad):
        """Every baseline that antenna is in would inherit the value."""
        path = tmp_path / "noise.npz"
        s_ant = np.array([1.0, 2.0, 3.0, 4.0])
        s_ant[1] = bad
        np.savez(path, s_ant=s_ant)
        a1, a2 = np.triu_indices(4, k=1)

        with pytest.raises(ValueError, match="s_ant"):
            read_noise_file(str(path), len(a1), a1, a2)

    def test_an_antenna_the_observation_never_uses_is_not_policed(self, tmp_path):
        """A file may cover a whole array; only the antennas correlated here can
        mis-weight anything, so a placeholder on an unused one is not an error."""
        path = tmp_path / "noise.npz"
        s_ant = np.array([1.0, 2.0, 3.0, 0.0])  # antenna 3 is not correlated
        np.savez(path, s_ant=s_ant)
        a1, a2 = np.triu_indices(3, k=1)

        out = read_noise_file(str(path), len(a1), a1, a2)

        np.testing.assert_allclose(
            out, np.sqrt(s_ant[a1] ** 2 + s_ant[a2] ** 2) / np.sqrt(2)
        )

    def test_a_two_dimensional_sigma_bl_is_rejected_by_shape(self, tmp_path):
        """Ravelling accepts a column, or a transposed grid, whose entries are in
        an order nobody wrote down -- and mis-weights every baseline in silence."""
        path = tmp_path / "noise.npz"
        np.savez(path, sigma_bl=np.ones((N_BL, 1)))

        with pytest.raises(ValueError, match=r"sigma_bl has shape \(6, 1\)"):
            read_noise_file(str(path), N_BL)

    def test_a_two_dimensional_s_ant_is_rejected_by_shape(self, tmp_path):
        path = tmp_path / "noise.npz"
        np.savez(path, s_ant=np.ones((4, 1)))
        a1, a2 = np.triu_indices(4, k=1)

        with pytest.raises(ValueError, match=r"s_ant has shape \(4, 1\)"):
            read_noise_file(str(path), len(a1), a1, a2)

    def test_s_ant_must_cover_every_antenna_in_the_pairs(self, tmp_path):
        """Short of the highest antenna index, ``s_ant[a1]`` either wraps round to
        another antenna's noise or raises an IndexError from inside numpy."""
        path = tmp_path / "noise.npz"
        np.savez(path, s_ant=np.ones(3))
        a1, a2 = np.triu_indices(4, k=1)

        with pytest.raises(ValueError, match="s_ant"):
            read_noise_file(str(path), len(a1), a1, a2)

    @pytest.mark.parametrize(
        "key, kwargs",
        [
            ("sigma_bl_freq", {"n_freq": N_FREQ}),
            ("sigma_bl", {}),
            ("s_ant", {}),
        ],
    )
    def test_a_complex_array_is_rejected_rather_than_half_read(
        self, tmp_path, key, kwargs
    ):
        """``np.asarray(z, dtype=float64)`` keeps the real part and drops the rest.

        It does so behind a numpy ``ComplexWarning``, which is easy to miss, and
        what is left is half of what the file said -- so a file whose noise is
        complex has its key named rather than being quietly reinterpreted.
        """
        path = tmp_path / "noise.npz"
        shapes = {
            "sigma_bl_freq": (N_BL, N_FREQ),
            "sigma_bl": (N_BL,),
            "s_ant": (4,),
        }
        np.savez(path, **{key: np.full(shapes[key], 1.0 + 2.0j)})
        a1, a2 = np.triu_indices(4, k=1)

        with pytest.raises(ValueError, match=f"{key}.*complex"):
            read_noise_file(str(path), N_BL, a1, a2, **kwargs)

    def test_the_complex_rejection_does_not_leak_a_numpy_warning(
        self, tmp_path, recwarn
    ):
        """Refused before the conversion, so the ComplexWarning never fires: the
        error says the whole of it, and there is no half-read array to warn about."""
        path = tmp_path / "noise.npz"
        np.savez(path, sigma_bl=np.full(N_BL, 1.0 + 2.0j))

        with pytest.raises(ValueError):
            read_noise_file(str(path), N_BL)

        assert [w for w in recwarn if issubclass(w.category, np.exceptions.ComplexWarning)] == []

    def test_a_zero_imaginary_part_is_still_complex(self, tmp_path):
        """The dtype is the statement, not the values that happen to be in it.

        A complex array of real values is still a file that answered a different
        question, and accepting it would make the rule depend on the data.
        """
        path = tmp_path / "noise.npz"
        np.savez(path, sigma_bl=np.arange(1, N_BL + 1).astype(complex))

        with pytest.raises(ValueError, match="sigma_bl.*complex"):
            read_noise_file(str(path), N_BL)

    @pytest.mark.parametrize(
        "key, kwargs",
        [
            ("sigma_bl_freq", {"n_freq": N_FREQ}),
            ("sigma_bl", {}),
            ("s_ant", {}),
        ],
    )
    def test_a_boolean_array_is_not_a_noise(self, tmp_path, key, kwargs):
        """`np.ones(n, dtype=bool).astype(float)` is a uniform 1 Jy noise.

        The same mistake the scalar path already refuses in `data.noise: true`,
        and refused for the same reason: 1 Jy is plausible enough that a run on a
        noise nobody wrote down would go unnoticed. A file of flags is a file
        answering a different question, whichever key it is under.
        """
        path = tmp_path / "noise.npz"
        shapes = {
            "sigma_bl_freq": (N_BL, N_FREQ),
            "sigma_bl": (N_BL,),
            "s_ant": (4,),
        }
        np.savez(path, **{key: np.ones(shapes[key], dtype=bool)})
        a1, a2 = np.triu_indices(4, k=1)

        with pytest.raises(ValueError, match=f"{key}.*bool"):
            read_noise_file(str(path), N_BL, a1, a2, **kwargs)

    def test_a_string_array_is_not_a_noise(self, tmp_path):
        """`np.array(["0.7"]).astype(float)` parses the text into a number.

        A noise written as text is a file that was never a noise array -- a CSV
        read without a dtype, most likely -- and parsing it would invent one out
        of whatever the strings happened to spell.
        """
        path = tmp_path / "noise.npz"
        np.savez(path, sigma_bl=np.full(N_BL, "0.7"))

        with pytest.raises(ValueError, match="sigma_bl has dtype"):
            read_noise_file(str(path), N_BL)

    def test_an_integer_noise_file_is_still_read(self, tmp_path):
        """Only the dtypes that are not real numbers are refused: an integer
        array says exactly what it means."""
        path = tmp_path / "noise.npz"
        sigma_bl = np.arange(1, N_BL + 1)
        np.savez(path, sigma_bl=sigma_bl)

        np.testing.assert_allclose(
            read_noise_file(str(path), N_BL), sigma_bl.astype(float)
        )

    def test_the_most_specific_key_wins(self, tmp_path):
        """A file carrying both is answering the same question twice; the
        frequency-resolved answer is the one with more information in it."""
        path = tmp_path / "noise.npz"
        sigma_bl_freq = np.full((N_BL, N_FREQ), 2.0)
        np.savez(path, sigma_bl_freq=sigma_bl_freq, sigma_bl=np.full(N_BL, 9.0))

        out = read_noise_file(str(path), N_BL, n_freq=N_FREQ)

        np.testing.assert_allclose(out, sigma_bl_freq)


# ---------------------------------------------------------------------------
# The data.noise override
# ---------------------------------------------------------------------------

class TestSetNoiseOverride:
    """``data.noise``: the last word on the noise, and the only scalar fallback.

    Called on a stand-in rather than a real ``TabConfig``, which would need an MS
    on disk to construct; the method reads four attributes and writes two.
    """

    MS_ESTIMATE = object()

    def _apply(self, noise):
        from tabascal.config import TabConfig

        a1, a2 = np.triu_indices(4, k=1)
        config = SimpleNamespace(
            n_bl=len(a1),
            n_freq=N_FREQ,
            a1=a1,
            a2=a2,
            noise=self.MS_ESTIMATE,
            noise_scalar=None,
        )
        TabConfig.set_noise(config, noise)

        return config

    def test_null_keeps_the_ms_estimate(self):
        assert self._apply(None).noise is self.MS_ESTIMATE

    def test_a_scalar_applies_to_every_baseline(self):
        config = self._apply(0.7)

        assert config.noise == 0.7
        assert config.noise_scalar == pytest.approx(0.7)

    @pytest.mark.parametrize("bad", [0, 0.0, -1.5])
    def test_a_non_positive_override_is_rejected(self, bad):
        """`0` used to read as "no override" and silently keep the MS estimate.

        Zero noise is not a noise: it divides the likelihood by nothing. Whichever
        the user meant -- null, or a real value -- they should be told.
        """
        with pytest.raises(ValueError, match="data.noise"):
            self._apply(bad)

    def test_something_that_is_not_a_noise_at_all_is_rejected_by_name(self):
        """A YAML list reaching float() would say nothing about data.noise."""
        with pytest.raises(ValueError, match="data.noise"):
            self._apply([1.0, 2.0])

    @pytest.mark.parametrize("flag", [True, False])
    def test_a_boolean_is_not_a_noise(self, flag):
        """`noise: true` is YAML for a switch, and this option is not one.

        `float(True)` is 1.0, so without an explicit check the run would proceed
        on a uniform 1 Jy noise nobody wrote down -- the one failure mode worse
        than stopping, since 1 Jy is plausible enough to go unnoticed. `false`
        is the same mistake and must be reported as the same mistake, not as a
        non-positive number.
        """
        with pytest.raises(ValueError, match="neither a number nor a path"):
            self._apply(flag)

    def test_a_file_that_underflows_the_run_precision_is_rejected(
        self, tmp_path, precision
    ):
        """Validated in float64, then converted to whatever the run works in.

        Under single precision 1e-50 underflows to zero on the device, and a zero
        noise divides the likelihood by nothing, silently, for a file every value
        of which was positive and finite as written. Under double precision the
        same file is fine, so the check has to be on the converted array rather
        than on the values read.

        Kept apart from the overflow case: one array carrying both would pass this
        test on a check that caught only one of them.
        """
        path = tmp_path / "noise.npz"
        sigma_bl = np.array([1e-50, 1.0, 1.0, 1.0, 1.0, 1.0])
        np.savez(path, sigma_bl=sigma_bl)

        if precision == "single":
            with pytest.raises(ValueError, match="float32"):
                self._apply(str(path))
        else:
            config = self._apply(str(path))
            np.testing.assert_allclose(np.asarray(config.noise), sigma_bl)

    def test_a_file_that_overflows_the_run_precision_is_rejected(
        self, tmp_path, precision
    ):
        """The other half of the conversion: 1e40 becomes an infinity in float32.

        An infinite noise weights its visibilities out of the likelihood
        altogether -- the mirror image of the underflow, and just as invisible --
        so it is rejected at the same point and for the same reason. Fine in
        double precision, where the value is representable.
        """
        path = tmp_path / "noise.npz"
        sigma_bl = np.array([1.0, 1.0, 1.0, 1.0, 1e40, 1.0])
        np.savez(path, sigma_bl=sigma_bl)

        if precision == "single":
            with pytest.raises(ValueError, match="float32"):
                self._apply(str(path))
        else:
            config = self._apply(str(path))
            np.testing.assert_allclose(np.asarray(config.noise), sigma_bl)

    def test_an_ordinary_file_survives_either_precision(self, tmp_path):
        """The check must cost nothing to a noise in the units the MS uses."""
        path = tmp_path / "noise.npz"
        sigma_bl = np.arange(1, N_BL + 1, dtype=float) * 0.65
        np.savez(path, sigma_bl=sigma_bl)

        config = self._apply(str(path))

        np.testing.assert_allclose(np.asarray(config.noise), sigma_bl, rtol=1e-6)

    def test_an_npz_of_per_baseline_noise_is_read(self, tmp_path):
        path = tmp_path / "noise.npz"
        sigma_bl = np.arange(1, N_BL + 1, dtype=float)
        np.savez(path, sigma_bl=sigma_bl)

        config = self._apply(str(path))

        np.testing.assert_allclose(np.asarray(config.noise), sigma_bl)
        assert config.noise_scalar == pytest.approx(np.median(sigma_bl))

    def test_an_npz_of_per_baseline_channel_noise_is_read(self, tmp_path):
        """The channel count comes from the MS read, which has already happened."""
        path = tmp_path / "noise.npz"
        sigma = np.arange(1, N_BL * N_FREQ + 1, dtype=float).reshape(N_BL, N_FREQ)
        np.savez(path, sigma_bl_freq=sigma)

        config = self._apply(str(path))

        assert np.asarray(config.noise).shape == (N_BL, N_FREQ)
        np.testing.assert_allclose(np.asarray(config.noise), sigma)
        assert config.noise_scalar == pytest.approx(np.median(sigma))


# ---------------------------------------------------------------------------
# An MS with no usable noise column, and the override that rescues it
# ---------------------------------------------------------------------------

class TestNoiselessMS:
    """The order of the two steps is the whole of the recovery path.

    ``TabConfig`` reads the MS first and applies ``data.noise`` second, so a read
    that stopped at an unusable noise column would take the override's turn away
    and the documented recovery would be unreachable. Both real methods are
    called here, in that order, on a stand-in: a real ``TabConfig`` would want an
    MS on disk and a TLE preflight to construct, neither of which says anything
    about the noise.
    """

    def _ms_params(self):
        """What ``read_ms`` returns for a partition whose noise columns are
        unusable: everything else read, and the noise left unset."""
        a1, a2 = np.triu_indices(4, k=1)
        n_bl, n_time = len(a1), 2
        shape = (n_bl, N_FREQ, n_time)

        return {
            "ra": 0.0,
            "dec": -30.0,
            "dish_d": 2.0,
            "ants_itrf": np.zeros((4, 3)),
            "vis_obs": np.zeros(shape, dtype=complex),
            "uvw": np.zeros((n_time, n_bl, 3)),
            "flags": np.zeros(shape, dtype=bool),
            "n_ant": 4,
            "n_bl": n_bl,
            "n_time": n_time,
            "n_freq": N_FREQ,
            "n_corr": 1,
            "int_time": 2.0,
            "times": np.arange(n_time, dtype=float),
            "times_mjd": np.array([60000.0, 60000.1]),
            "times_jd": np.array([60000.0, 60000.1]) + 2400000.5,
            "time_scale": "utc",
            "chan_width": 1e5,
            "freqs": np.linspace(1e9, 1.1e9, N_FREQ),
            # partition_noise found neither column usable and said so; deciding
            # what to do about it is not the reader's business.
            "noise": None,
            "noise_scalar": None,
            "a1": a1,
            "a2": a2,
        }

    def _read(self, monkeypatch):
        from tabascal import config as config_module

        monkeypatch.setattr(
            config_module, "read_ms", lambda *args, **kwargs: self._ms_params()
        )
        config = SimpleNamespace(ms_path="never/read.ms")
        config_module.TabConfig.read_ms_params(config, None, "xx", "DATA")

        return config

    def test_the_read_leaves_the_noise_unset_instead_of_stopping(self, monkeypatch):
        """Nothing is invented -- but nothing is decided yet either."""
        config = self._read(monkeypatch)

        assert config.noise is None and config.noise_scalar is None

    def test_a_scalar_override_rescues_the_run(self, monkeypatch):
        """The documented recovery: the MS has no noise, so the user gives one."""
        from tabascal.config import TabConfig

        config = self._read(monkeypatch)
        TabConfig.set_noise(config, 0.7)

        assert config.noise == 0.7
        assert config.noise_scalar == pytest.approx(0.7)

    def test_an_npz_override_rescues_the_run(self, monkeypatch, tmp_path):
        """The out-of-band measurement, which is why an .npz is accepted at all."""
        from tabascal.config import TabConfig

        path = tmp_path / "noise.npz"
        a1, _ = np.triu_indices(4, k=1)
        sigma_bl = np.arange(1, len(a1) + 1, dtype=float)
        np.savez(path, sigma_bl=sigma_bl)

        config = self._read(monkeypatch)
        TabConfig.set_noise(config, str(path))

        np.testing.assert_allclose(np.asarray(config.noise), sigma_bl)

    def test_without_an_override_it_is_still_an_error(self, monkeypatch):
        """Deferred, not dropped: a run cannot proceed with no noise at all, and
        a made-up scale would silently re-weight the whole fit."""
        from tabascal.config import TabConfig

        config = self._read(monkeypatch)

        with pytest.raises(ValueError, match="set data.noise explicitly"):
            TabConfig.set_noise(config, None)

    def test_no_noise_can_leak_past_the_error(self, monkeypatch):
        """Whatever the model is handed, it is never a None: the error above is
        raised before anything downstream reads the attribute."""
        from tabascal.config import TabConfig

        config = self._read(monkeypatch)

        with pytest.raises(ValueError):
            TabConfig.set_noise(config, None)

        assert config.noise is None  # unchanged, and unreachable by the model


class TestInitCallsTheReadBeforeTheOverride:
    """The same ordering, pinned in the constructor that actually decides it.

    Every test above calls the two methods itself, in the order it wants, so all
    of them would still pass if ``TabConfig.__init__`` were reordered to apply
    ``data.noise`` first -- and then ``read_ms_params`` would overwrite the
    override with the MS's own estimate (or with ``None``), making the documented
    recovery unreachable and the option silently ignored. So call the real
    constructor and record what it really does.

    The heavy steps are stubbed out: the ordering is the whole subject, and a
    real construction wants an MS on disk, a TLE preflight and an RFI sampling
    estimate, none of which say anything about it.
    """

    STUBBED = [
        "read_ms_params",
        "set_noise",
        "set_flags",
        "get_orbital_elements",
        "estimate_rfi_sampling",
        "_set_freqs_times",
        "set_elevation_mask",
    ]

    def _config(self, noise=0.7):
        return {
            "model": {"components": [], "precision": "single"},
            "data": {
                "freq": 0,
                "corr": "xx",
                "data_col": "DATA",
                "noise": noise,
                "flags": False,
            },
            "rfi": {"n_int_time": 4, "n_int_freq": 1, "time_int_factor": 1.0},
            "satellites": {},
        }

    def _construct(self, monkeypatch, noise=0.7):
        """The real ``__init__``, with every step recording that it ran."""
        from tabascal.config import TabConfig

        calls = []
        # What the stubs leave behind: the attributes the rest of __init__ reads
        # from the steps that are stubbed out, so the constructor runs to the end
        # rather than stopping somewhere that would hide a reordering.
        leaves = {
            "read_ms_params": {
                "n_freq": N_FREQ,
                "times_mjd": np.array([59999.5]),
                "times_jd": np.array([2460000.0]),
                "noise": None,
                "noise_scalar": None,
                "vis_obs": np.zeros((1, N_FREQ, 1), dtype=complex),
                "flags": np.zeros((1, N_FREQ, 1), dtype=bool),
            },
            "set_noise": {"noise": 0.7, "noise_scalar": 0.7},
            "get_orbital_elements": {"n_rfi": 0},
        }

        def recorder(name):
            def method(self, *args, **kwargs):
                calls.append((name, args))
                for key, value in leaves.get(name, {}).items():
                    setattr(self, key, value)

            return method

        for name in self.STUBBED:
            monkeypatch.setattr(TabConfig, name, recorder(name))

        config = TabConfig(self._config(noise), "never/read.ms")

        return config, [name for name, _ in calls], calls

    def test_the_ms_is_read_before_the_override_is_applied(self, monkeypatch):
        _, names, _ = self._construct(monkeypatch)

        assert "read_ms_params" in names and "set_noise" in names
        assert names.index("read_ms_params") < names.index("set_noise")

    def test_the_override_is_applied_exactly_once(self, monkeypatch):
        """Applied twice, the second call would re-read a file and could disagree
        with the first; applied never, the option does nothing."""
        _, names, _ = self._construct(monkeypatch)

        assert names.count("set_noise") == 1

    def test_the_configured_value_is_what_reaches_set_noise(self, monkeypatch):
        """``data.noise``, not some other key that happens to be nearby."""
        _, _, calls = self._construct(monkeypatch, noise=0.123)

        assert dict((name, args) for name, args in calls)["set_noise"] == (0.123,)

    def test_the_noise_survives_to_the_end_of_construction(self, monkeypatch):
        """Nothing after ``set_noise`` overwrites what it resolved."""
        config, _, _ = self._construct(monkeypatch)

        assert np.asarray(config.noise) == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# reduced_chi2 with a per-baseline noise
# ---------------------------------------------------------------------------

class TestReducedChi2PerBaseline:
    """The masking hazard: `x[~flags]` flattens, so the noise must broadcast first."""

    @staticmethod
    def _chi2(pred, true, noise, flags):
        from tabascal.tab_tools import reduced_chi2
        import jax.numpy as jnp

        return float(
            reduced_chi2(jnp.asarray(pred), jnp.asarray(true),
                         jnp.asarray(noise), jnp.asarray(flags))
        )

    def test_scalar_noise_is_unchanged(self, exact_rtol):
        """Uniform data must give exactly what it always did."""
        rng = np.random.default_rng(0)
        shape = (N_BL, 3, 4)
        true = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        pred = true + 0.1
        flags = np.zeros(shape, dtype=bool)

        scalar = self._chi2(pred, true, 0.5, flags)
        uniform = self._chi2(pred, true, np.full(N_BL, 0.5), flags)

        assert scalar == pytest.approx(uniform, rel=exact_rtol)

    def test_a_uniform_baseline_frequency_noise_matches_the_scalar(self, exact_rtol):
        """The (n_bl, n_freq) path must reduce to the old answer on flat noise."""
        rng = np.random.default_rng(3)
        shape = (N_BL, 3, 4)
        true = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        pred = true + 0.1
        flags = np.zeros(shape, dtype=bool)

        scalar = self._chi2(pred, true, 0.5, flags)
        uniform = self._chi2(pred, true, np.full((N_BL, 3), 0.5), flags)

        assert scalar == pytest.approx(uniform, rel=exact_rtol)

    def test_each_baseline_divides_by_its_own_noise(self, exact_rtol):
        """Built so a mis-aligned noise gives a different, wrong answer.

        The residual must **vary per baseline** for this to bite: with the same
        residual everywhere, sum((r / noise_i)^2) is invariant under permuting
        the noise, so a shuffled pairing would score identically and the test
        would prove nothing.
        """
        shape = (N_BL, 1, 1)
        resid = np.arange(1, N_BL + 1, dtype=float).reshape(shape)
        true = np.zeros(shape, dtype=complex)
        pred = resid.astype(complex)
        noise = np.arange(1, N_BL + 1, dtype=float)
        flags = np.zeros(shape, dtype=bool)

        got = self._chi2(pred, true, noise, flags)

        # Paired correctly every ratio is 1, so the sum is exactly N / (2 N).
        expected = np.sum((resid.ravel() / noise) ** 2) / (2 * N_BL)
        assert got == pytest.approx(expected, rel=exact_rtol)
        assert got == pytest.approx(0.5, rel=exact_rtol)

        # Reversing the noise mis-pairs it, and now that is visible.
        assert self._chi2(pred, true, noise[::-1].copy(), flags) != pytest.approx(got)

    def test_alignment_survives_a_ragged_flag_mask(self, exact_rtol):
        """The regression this guards: flags remove different counts per baseline.

        Applying the noise after masking would recycle values across baselines,
        which only shows up when the mask is not uniform.
        """
        shape = (N_BL, 2, 2)
        true = np.zeros(shape, dtype=complex)
        pred = np.ones(shape, dtype=complex)
        noise = np.arange(1, N_BL + 1, dtype=float)

        flags = np.zeros(shape, dtype=bool)
        flags[0, 0, 0] = True          # drop one sample from baseline 0
        flags[3, :, :] = True          # drop baseline 3 entirely

        got = self._chi2(pred, true, noise, flags)

        kept = np.broadcast_to(noise.reshape(-1, 1, 1), shape)[~flags]
        expected = np.sum((1.0 / kept) ** 2) / (2 * kept.size)
        assert got == pytest.approx(expected, rel=exact_rtol)

    def test_baseline_frequency_alignment_survives_a_ragged_flag_mask(
        self, exact_rtol
    ):
        """The same hazard one axis further in.

        The residual varies per cell and the noise varies over both baseline and
        channel, so any reshape that transposes the two axes, or any noise applied
        after `x[~flags]` has flattened the array, gives a different number.
        """
        rng = np.random.default_rng(7)
        shape = (N_BL, 3, 4)
        true = np.zeros(shape, dtype=complex)
        pred = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        noise = np.arange(1, N_BL * 3 + 1, dtype=float).reshape(N_BL, 3)

        flags = np.zeros(shape, dtype=bool)
        flags[0, 0, 0] = True          # one sample
        flags[2, 1, :] = True          # one whole channel of one baseline
        flags[3, :, :] = True          # one whole baseline

        got = self._chi2(pred, true, noise, flags)

        full = np.broadcast_to(noise[:, :, None], shape)
        expected = np.sum(
            (np.abs(pred[~flags]) / full[~flags]) ** 2
        ) / (2 * (~flags).sum())
        assert got == pytest.approx(expected, rel=exact_rtol)

        # Transposing the noise over the two axes it spans is the mistake this
        # catches; with N_BL != n_freq it cannot even be built, so square it off.
        square = noise[:3, :3]
        assert self._chi2(
            pred[:3], true[:3], square, flags[:3]
        ) != pytest.approx(
            self._chi2(pred[:3], true[:3], square.T.copy(), flags[:3])
        )


class TestReducedChi2TimeVarying:
    """The same alignment one axis further out, for a noise that keeps time."""

    @staticmethod
    def _chi2(pred, true, noise, flags):
        from tabascal.tab_tools import reduced_chi2
        import jax.numpy as jnp

        return float(
            reduced_chi2(jnp.asarray(pred), jnp.asarray(true),
                         jnp.asarray(noise), jnp.asarray(flags))
        )

    def test_a_time_resolved_noise_aligns_per_timestep(self, exact_rtol):
        """Ragged flags and a residual that varies per timestep, so a noise
        applied after ``x[~flags]`` has flattened the array -- or paired with
        the wrong timestep -- gives a different number."""
        rng = np.random.default_rng(11)
        shape = (N_BL, 2, 3)
        true = np.zeros(shape, dtype=complex)
        pred = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        noise = (
            np.arange(1, N_BL + 1, dtype=float)[:, None]
            * np.array([1.0, 2.0, 4.0])[None, :]
        )[:, None, :]

        flags = np.zeros(shape, dtype=bool)
        flags[0, 0, 0] = True          # one sample
        flags[2, :, 1] = True          # one whole timestep of one baseline
        flags[3, :, :] = True          # one whole baseline

        got = self._chi2(pred, true, noise, flags)

        full = np.broadcast_to(noise, shape)
        expected = np.sum(
            (np.abs(pred[~flags]) / full[~flags]) ** 2
        ) / (2 * (~flags).sum())
        assert got == pytest.approx(expected, rel=exact_rtol)

        # Reversing the time axis mis-pairs it, and the varying residual makes
        # that visible -- with a flat residual the sum would be invariant.
        assert self._chi2(
            pred, true, noise[:, :, ::-1].copy(), flags
        ) != pytest.approx(got)

    def test_a_fully_resolved_noise_aligns_over_every_axis(self, exact_rtol):
        rng = np.random.default_rng(13)
        shape = (N_BL, 3, 4)
        true = np.zeros(shape, dtype=complex)
        pred = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        noise = 1.0 + np.arange(np.prod(shape), dtype=float).reshape(shape)
        flags = np.zeros(shape, dtype=bool)
        flags[1, 2, 3] = True

        got = self._chi2(pred, true, noise, flags)

        expected = np.sum(
            (np.abs(pred[~flags]) / noise[~flags]) ** 2
        ) / (2 * (~flags).sum())
        assert got == pytest.approx(expected, rel=exact_rtol)

    def test_a_time_resolved_noise_weights_by_time_when_n_freq_equals_n_time(
        self, exact_rtol
    ):
        """The ambiguity the length-1 channel axis exists to remove.

        With as many channels as timesteps the two axes are the same length, so
        nothing but the shape says which one a per-baseline-per-something noise
        belongs to. Read the SIGMA column through the real path and check the
        weighting follows time: applied along frequency instead it is a
        different number, and neither shape nor size would have said so.
        """
        n = 4
        shape = (N_BL, n, n)
        rng = np.random.default_rng(17)
        true = np.zeros(shape, dtype=complex)
        pred = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        flags = np.zeros(shape, dtype=bool)

        per_bl_time = np.arange(1, N_BL + 1, dtype=float)[:, None] * (
            1.0 + np.arange(n, dtype=float)
        )[None, :]
        noise = per_baseline_sigma(_sigma_column_time(per_bl_time), n, N_BL)

        assert noise.shape == (N_BL, 1, n)

        got = self._chi2(pred, true, noise, flags)

        by_time = np.broadcast_to(per_bl_time[:, None, :], shape)
        expected = np.sum((np.abs(pred) / by_time) ** 2) / (2 * pred.size)
        assert got == pytest.approx(expected, rel=exact_rtol)

        by_freq = per_bl_time[:, :, None]  # the same values, one axis over
        assert self._chi2(pred, true, by_freq, flags) != pytest.approx(got)


# ---------------------------------------------------------------------------
# The likelihood closure, the other broadcast-before-masking consumer
# ---------------------------------------------------------------------------

class TestLikelihoodResolvedNoise:
    """``Model.__init__`` hands the likelihood a broadcast noise; it must divide
    the sample it belongs to.

    ``gaussian`` masks with a flag array rather than by indexing, so the hazard
    is the reverse of ``reduced_chi2``'s: nothing flattens, and a noise that
    failed to broadcast would raise -- but one broadcast along the *wrong* axis
    would weight every sample by another timestep's noise and say nothing. The
    residual varies per timestep here so that mis-pairing shows up as a number.
    """

    @staticmethod
    def _nlog_like(pred, obs, noise, flags):
        import jax.numpy as jnp
        from numpyro.infer.util import log_density

        from tabascal.components.likelihood import gaussian

        args = {
            "noise": broadcast_to_vis(jnp.asarray(noise), tuple(obs.shape)),
            "flags": jnp.asarray(flags),
        }
        model = lambda: gaussian(jnp.asarray(pred), jnp.asarray(obs), args)

        return float(log_density(model, (), {}, {})[0])

    def _expected(self, pred, obs, noise, flags):
        """The masked complex Gaussian, written out over the unflagged samples."""

        sigma = np.broadcast_to(noise, pred.shape)[~flags]
        resid = (pred - obs)[~flags]
        parts = np.concatenate([resid.real, resid.imag])
        sigma = np.concatenate([sigma, sigma])

        return float(
            np.sum(-0.5 * (parts / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi)))
        )

    def test_a_time_resolved_noise_divides_its_own_timestep(self, exact_rtol):
        rng = np.random.default_rng(23)
        shape = (N_BL, 2, 3)
        obs = np.zeros(shape, dtype=complex)
        pred = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        noise = (
            np.arange(1, N_BL + 1, dtype=float)[:, None]
            * np.array([1.0, 2.0, 4.0])[None, :]
        )[:, None, :]

        flags = np.zeros(shape, dtype=bool)
        flags[0, 0, 0] = True          # one sample
        flags[2, :, 1] = True          # one whole timestep of one baseline
        flags[3, :, :] = True          # one whole baseline

        got = self._nlog_like(pred, obs, noise, flags)

        assert got == pytest.approx(
            self._expected(pred, obs, noise, flags), rel=exact_rtol
        )

        # Mis-paired over time, and the varying residual makes that visible.
        assert self._nlog_like(
            pred, obs, noise[:, :, ::-1].copy(), flags
        ) != pytest.approx(got)

    def test_a_fully_resolved_noise_divides_its_own_cell(self, exact_rtol):
        rng = np.random.default_rng(29)
        shape = (N_BL, 3, 4)
        obs = np.zeros(shape, dtype=complex)
        pred = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        noise = 1.0 + np.arange(np.prod(shape), dtype=float).reshape(shape)
        flags = np.zeros(shape, dtype=bool)
        flags[1, 2, 3] = True

        got = self._nlog_like(pred, obs, noise, flags)

        assert got == pytest.approx(
            self._expected(pred, obs, noise, flags), rel=exact_rtol
        )
