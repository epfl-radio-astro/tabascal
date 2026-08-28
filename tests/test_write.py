"""Tests for tabascal.write — per-baseline gains, the calibrated frame, weights.

Every case uses a non-unit, non-uniform gain: a unity gain cannot distinguish
``g_p conj(g_q)`` from ``|g_p|^2``, nor a calibrated column from a data-frame one.

No jax here, so the tolerances are not the ``exact_rtol`` fixture's business:
the writer works in ``complex64`` whatever precision the session is in, and the
identities below are either exact in floating point (and asserted as such) or
bounded by float64 round-off.
"""

import warnings

import numpy as np
import pytest

import tabascal.write as write_mod
from tabascal.interferometry import baseline_gains
from tabascal.write import (
    calibrated_residuals,
    calibrated_weights,
    external_baseline_gains,
    unit_bad_gains,
    count_substituted,
    warn_bad_baseline_gains,
    warn_bad_gains,
)


@pytest.fixture
def gains():
    """Per-antenna gains with distinct amplitude and phase, shape (n_ant, 1, 1)."""

    amp = np.array([0.5, 1.0, 2.0, 1.5])
    phase = np.array([0.0, 0.3, -0.7, 1.1])

    return (amp * np.exp(1j * phase)).astype(np.complex128)[:, None, None]


@pytest.fixture
def pairs():
    """Antenna pairs for a 4-antenna array, all 6 cross baselines."""

    a1, a2 = np.triu_indices(4, k=1)

    return a1, a2


# ---------------------------------------------------------------------------
# calibrated_residuals
# ---------------------------------------------------------------------------

class TestCalibratedResiduals:
    """One frame for every column: the data with all the gains divided out."""

    @pytest.fixture
    def model(self):
        rng = np.random.default_rng(0)
        shape = (6, 3, 1)
        vis_ast = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        vis_rfi = rng.normal(size=shape) + 1j * rng.normal(size=shape)

        return vis_ast, vis_rfi

    def test_the_decomposition_closes_exactly(self, model, gains, pairs):
        """The anchor: ast + rfi + total residual reconstructs the calibrated data.

        Bit-for-bit, not to a tolerance -- the total residual is formed against
        the *same* sum the two model columns are written from, so the identity
        is the one thing that fails loudly if a column moves frame.
        """
        vis_ast, vis_rfi = model
        a1, a2 = pairs
        gains_bl = baseline_gains(gains, a1, a2)

        vis_cal = (gains_bl * (vis_ast + vis_rfi) + 0.01) / gains_bl

        res = calibrated_residuals(vis_cal, vis_ast, vis_rfi)

        np.testing.assert_array_equal(vis_ast + vis_rfi + res["total"], vis_cal)

    def test_a_perfect_model_leaves_zero_residual(self, model):
        vis_ast, vis_rfi = model

        res = calibrated_residuals(vis_ast + vis_rfi, vis_ast, vis_rfi)

        np.testing.assert_allclose(res["total"], 0.0, atol=1e-12)

    def test_the_models_are_never_re_gained(self, model, gains, pairs):
        """The data-frame form is what this replaces; it must not come back."""
        vis_ast, vis_rfi = model
        a1, a2 = pairs
        gains_bl = baseline_gains(gains, a1, a2)
        vis_obs = gains_bl * (vis_ast + vis_rfi)

        res = calibrated_residuals(vis_obs / gains_bl, vis_ast, vis_rfi)

        np.testing.assert_allclose(res["total"], 0.0, atol=1e-12)
        # The superseded #134 residual, which does not close in this frame.
        assert not np.allclose(vis_obs / gains_bl - gains_bl * vis_ast, 0.0)

    def test_per_component_residuals_leave_the_other_component(self, model, gains, pairs):
        vis_ast, vis_rfi = model
        a1, a2 = pairs
        gains_bl = baseline_gains(gains, a1, a2)
        vis_cal = (gains_bl * (vis_ast + vis_rfi)) / gains_bl

        res = calibrated_residuals(vis_cal, vis_ast, vis_rfi)

        # Removing only the astronomical model leaves exactly the RFI -- and it
        # is the un-gained RFI, because that is the frame every column is in.
        np.testing.assert_allclose(res["ast"], vis_rfi, atol=1e-12)
        np.testing.assert_allclose(res["rfi"], vis_ast, atol=1e-12)

    def test_reduces_to_the_old_behaviour_under_unity_gains(self, model):
        """No change for existing UnitaryGains results, so no refs move.

        With ``g = 1`` the calibrated data *is* the data, so the calibrated
        residual and the data-frame residual it replaces are the same numbers.
        """
        vis_ast, vis_rfi = model
        data = vis_ast * 0 + 5.0

        res = calibrated_residuals(data / 1.0, vis_ast, vis_rfi)

        np.testing.assert_array_equal(res["ast"], data - vis_ast)
        np.testing.assert_array_equal(res["rfi"], data - vis_rfi)
        np.testing.assert_array_equal(res["total"], data - (vis_ast + vis_rfi))


# ---------------------------------------------------------------------------
# unit_bad_gains / warn_bad_gains
# ---------------------------------------------------------------------------

class TestUnitBadGains:
    """A zero or non-finite antenna gain is replaced by 1, not blanked."""

    @pytest.fixture
    def gain_array(self):
        """Gains shaped as the zarr stores them: (sample, ant, freq, time)."""
        rng = np.random.default_rng(7)
        shape = (2, 4, 3, 5)

        return (
            rng.normal(size=shape) + 1j * rng.normal(size=shape)
        ).astype(np.complex64)

    def test_finite_gains_are_untouched(self, gain_array):
        out, bad = unit_bad_gains(gain_array)

        np.testing.assert_array_equal(out, gain_array)
        assert not np.any(bad)

    @pytest.mark.parametrize("value", [0.0, np.nan, np.inf, -np.inf])
    def test_bad_values_become_unity(self, gain_array, value):
        gain_array = gain_array.copy()
        gain_array[1, 2, 0, 3] = value

        out, bad = unit_bad_gains(gain_array)

        assert out[1, 2, 0, 3] == 1.0
        assert bad[1, 2, 0, 3]
        assert np.count_nonzero(bad) == 1

    def test_a_nan_imaginary_part_counts_as_bad(self, gain_array):
        """isfinite on a complex value covers both parts."""
        gain_array = gain_array.copy()
        gain_array[0, 0, 0, 0] = complex(1.0, np.nan)

        out, bad = unit_bad_gains(gain_array)

        assert bad[0, 0, 0, 0] and out[0, 0, 0, 0] == 1.0

    def test_everything_stays_finite(self, gain_array):
        """The point of the substitution: nothing downstream can divide by zero."""
        gain_array = gain_array.copy()
        gain_array[:, 1] = 0.0

        out, _ = unit_bad_gains(gain_array)

        assert np.all(np.isfinite(out))
        assert not np.any(out == 0.0)

    def test_only_the_dead_antenna_is_substituted(self, gain_array):
        gain_array = gain_array.copy()
        gain_array[:, 1] = 0.0

        out, bad = unit_bad_gains(gain_array)

        assert np.all(bad[:, 1]) and not np.any(bad[:, [0, 2, 3]])
        np.testing.assert_array_equal(out[:, [0, 2, 3]], gain_array[:, [0, 2, 3]])

    def test_the_dtype_is_preserved(self, gain_array):
        gain_array = gain_array.copy()
        gain_array[0, 0, 0, 0] = 0.0

        out, _ = unit_bad_gains(gain_array)

        assert out.dtype == np.complex64

    def test_works_on_dask_arrays(self, gain_array):
        """write_results_ms passes the zarr's dask array straight in."""
        da = pytest.importorskip("dask.array")
        gain_array = gain_array.copy()
        gain_array[0, 3] = 0.0

        out, bad = unit_bad_gains(da.from_array(gain_array, chunks=(1, 2, 3, 5)))

        out, bad = np.asarray(out), np.asarray(bad)
        assert np.all(out[0, 3] == 1.0)
        assert np.count_nonzero(bad) == gain_array[0, 3].size


class TestUnitBadGainsOnBaselineGains:
    """The same helper guards the mean baseline gain, which has no antenna axis."""

    def test_a_mean_that_averages_to_zero_is_substituted(self):
        """Both samples are finite and non-zero; their mean is not."""
        gains_bl_s = np.array([1.0 + 0j, -1.0 + 0j])[:, None, None, None] * np.ones(
            (1, 6, 2, 3)
        )
        assert np.all(np.isfinite(gains_bl_s)) and not np.any(gains_bl_s == 0)

        out, bad = unit_bad_gains(gains_bl_s.mean(axis=0))

        assert out.shape == (6, 2, 3)
        assert np.all(bad) and np.all(out == 1.0)

    def test_a_healthy_mean_is_untouched(self):
        rng = np.random.default_rng(9)
        shape = (6, 2, 3)
        gains_bl = (
            rng.normal(size=shape) + 1j * rng.normal(size=shape)
        ).astype(np.complex64)

        out, bad = unit_bad_gains(gains_bl)

        np.testing.assert_array_equal(out, gains_bl)
        assert not np.any(bad)


class TestCountSubstituted:
    """The warnings need numbers, not the mask -- and lazily on dask."""

    def test_counts_and_names_antennas_on_numpy(self):
        bad = np.zeros((2, 4, 3, 5), dtype=bool)
        bad[0, 1] = True
        bad[1, 3, 0, 0] = True

        n_bad, bad_ants = count_substituted(bad, ant_axis=1)

        assert int(n_bad) == 16
        np.testing.assert_array_equal(bad_ants, [False, True, False, True])

    def test_count_alone_without_an_antenna_axis(self):
        bad = np.zeros((6, 2, 1), dtype=bool)
        bad[0] = True

        assert int(count_substituted(bad)) == 2

    def test_stays_lazy_on_dask(self):
        """Nothing full-size is materialised: the reductions are still graphs."""
        da = pytest.importorskip("dask.array")
        bad = np.zeros((2, 4, 3, 5), dtype=bool)
        bad[0, 1] = True
        lazy = da.from_array(bad, chunks=(1, 2, 3, 5))

        n_bad, bad_ants = count_substituted(lazy, ant_axis=1)

        assert hasattr(n_bad, "compute") and hasattr(bad_ants, "compute")
        assert int(n_bad.compute()) == 15
        np.testing.assert_array_equal(bad_ants.compute(), [False, True, False, False])


class TestWarnBadBaselineGains:

    def test_warns_with_the_count_and_fraction(self):
        bad = np.zeros((6, 2, 1), dtype=bool)
        bad[0] = True

        with pytest.warns(RuntimeWarning) as record:
            n_bad = warn_bad_baseline_gains(count_substituted(bad), bad.size)

        message = str(record[0].message)
        assert n_bad == 2
        assert "2 of 12" in message
        assert "mean baseline gains" in message
        # No antenna axis to name once the product has been formed.
        assert "Affected antennas" not in message

    def test_silent_when_the_mean_is_healthy(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert warn_bad_baseline_gains(0, 12) == 0


class TestWarnBadGains:

    def test_warns_with_the_count_and_the_antennas(self):
        bad = np.zeros((2, 4, 3, 5), dtype=bool)
        bad[0, 1] = True
        bad[1, 3, 0, 0] = True

        with pytest.warns(RuntimeWarning) as record:
            count, bad_ants = count_substituted(bad, ant_axis=1)
            n_bad = warn_bad_gains(count, bad.size, bad_ants)

        message = str(record[0].message)
        assert n_bad == 16
        assert "16 of 120" in message
        assert "[1, 3]" in message                  # the antennas, not the samples
        assert "set to 1" in message

    def test_silent_when_nothing_was_substituted(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert warn_bad_gains(0, 24, np.zeros(4, dtype=bool)) == 0


# ---------------------------------------------------------------------------
# external_baseline_gains
# ---------------------------------------------------------------------------

N_ANT_EXT, N_FREQ_EXT, N_TIME_EXT = 4, 2, 3


@pytest.fixture
def ext_gains():
    """Per-antenna external gains on an observation grid, non-unit and complex."""

    rng = np.random.default_rng(12)
    shape = (N_ANT_EXT, N_FREQ_EXT, N_TIME_EXT)
    amp = rng.uniform(0.3, 3.0, shape)
    phase = rng.uniform(-np.pi, np.pi, shape)

    return (amp * np.exp(1j * phase)).astype(complex)


@pytest.fixture
def placed(monkeypatch, ext_gains):
    """``external_baseline_gains`` with the table placement stubbed out.

    The placement itself is ``tabascal.gain_table``'s subject and is tested
    there; what belongs here is what the writer does with the result -- the
    baseline product, and the unity substitution for an antenna nobody solved.
    """

    captured = {}

    def _run(dead=None, **kwargs):
        gains = np.asarray(ext_gains)
        dead = np.zeros(gains.shape, dtype=bool) if dead is None else dead

        def _stub(gain_table, times, freqs, n_ant=None, verbose=True):
            captured.update(
                gain_table=gain_table, times=times, freqs=freqs, n_ant=n_ant
            )
            return np.where(dead, 1.0, gains), dead

        monkeypatch.setattr(write_mod, "gains_from_tables", _stub)

        a1, a2 = np.triu_indices(N_ANT_EXT, k=1)
        args = dict(
            gain_table=["/tables/B0"],
            times_sec=np.array([0.0, 2.0, 4.0]),
            freqs=np.array([1.0e9, 1.1e9]),
            a1=a1,
            a2=a2,
            n_ant=N_ANT_EXT,
        )
        args.update(kwargs)

        return external_baseline_gains(**args), captured, a1, a2

    return _run


class TestExternalBaselineGains:
    """The external layer of the calibrated frame, in the writer's own terms."""

    def test_is_the_baseline_product_of_the_placed_gains(self, placed, ext_gains):
        g_bl, _, a1, a2 = placed()

        np.testing.assert_allclose(g_bl, ext_gains[a1] * ext_gains[a2].conj())
        assert g_bl.shape == (len(a1), N_FREQ_EXT, N_TIME_EXT)

    def test_both_antennas_are_used(self, placed, ext_gains):
        """The ANTENNA1-twice mistake gives ``|g_p|^2``: real, and phaseless."""
        g_bl, _, a1, _ = placed()

        assert not np.allclose(g_bl, ext_gains[a1] * ext_gains[a1].conj())
        assert np.abs(np.imag(g_bl)).max() > 0

    def test_an_unsolved_antenna_makes_its_baselines_unity(self, placed, ext_gains):
        """Unity, not NaN: those visibilities are written uncalibrated, not lost."""
        dead = np.zeros(ext_gains.shape, dtype=bool)
        dead[2] = True

        g_bl, _, a1, a2 = placed(dead=dead)

        touched = (a1 == 2) | (a2 == 2)
        np.testing.assert_array_equal(g_bl[touched], 1.0)
        assert np.all(np.isfinite(g_bl))
        assert not np.any(g_bl[~touched] == 1.0)

    def test_the_grid_is_handed_to_the_placement_unchanged(self, placed):
        """The frame is only shared with the run if the grid is the run's own."""
        _, captured, _, _ = placed()

        assert captured["gain_table"] == ["/tables/B0"]
        np.testing.assert_array_equal(captured["times"], [0.0, 2.0, 4.0])
        np.testing.assert_array_equal(captured["freqs"], [1.0e9, 1.1e9])
        assert captured["n_ant"] == N_ANT_EXT


# ---------------------------------------------------------------------------
# calibrated_weights
# ---------------------------------------------------------------------------

class TestCalibratedWeights:
    """``WEIGHT_SPECTRUM = |g_total|^2 / SIGMA^2``, per channel."""

    def test_is_the_inverse_variance_of_the_calibrated_visibility(self):
        rng = np.random.default_rng(13)
        shape = (6, 2, 3)
        g = (rng.uniform(0.3, 3.0, shape) * np.exp(1j * rng.uniform(-np.pi, np.pi, shape))).astype(np.complex64)
        sigma = rng.uniform(0.1, 2.0, shape)

        weight = calibrated_weights(g, sigma)

        # sigma_cal = SIGMA / |g|, so weight = 1 / sigma_cal^2.
        sigma_cal = sigma / np.abs(g).astype(np.float64)
        np.testing.assert_allclose(weight, 1.0 / sigma_cal**2, rtol=1e-12)

    def test_a_frequency_dependent_gain_gives_a_frequency_dependent_weight(self):
        """The reason the column has to be WEIGHT_SPECTRUM and not WEIGHT."""
        g = np.array([[1.0, 2.0, 4.0]], dtype=np.complex64)
        sigma = np.ones((1, 3))

        weight = calibrated_weights(g, sigma)

        np.testing.assert_allclose(weight, [[1.0, 4.0, 16.0]])

    def test_a_unity_gain_leaves_the_ms_weight(self):
        sigma = np.array([0.5, 2.0])

        np.testing.assert_allclose(
            calibrated_weights(np.ones(2, dtype=np.complex64), sigma),
            1.0 / sigma**2,
        )

    def test_the_result_is_float64_whatever_the_gain_dtype(self):
        """Squared in float64 so the write's single cast is the only rounding."""
        weight = calibrated_weights(
            np.full(3, 2.0, dtype=np.complex64), np.full(3, 0.5)
        )

        assert weight.dtype == np.float64
        np.testing.assert_array_equal(weight, 16.0)
