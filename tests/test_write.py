"""Tests for tabascal.write — per-baseline gains and residual framing.

Every case uses a non-unit, non-uniform gain: a unity gain cannot distinguish
``g_p conj(g_q)`` from ``|g_p|^2``, nor a gained model from a raw one.
"""

import warnings

import numpy as np
import pytest

from tabascal.interferometry import baseline_gains
from tabascal.write import (
    data_frame_residuals,
    gained_model_mean,
    total_model,
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
# data_frame_residuals
# ---------------------------------------------------------------------------

class TestDataFrameResiduals:

    @pytest.fixture
    def model(self):
        rng = np.random.default_rng(0)
        shape = (6, 3, 1)
        vis_ast = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        vis_rfi = rng.normal(size=shape) + 1j * rng.normal(size=shape)

        return vis_ast, vis_rfi

    def test_the_total_residual_closes(self, model, gains, pairs):
        """vis_obs reconstructs exactly from the gained model plus its residual.

        This is the identity the bug broke: with a raw (un-gained) model
        subtracted, the gains are left behind in the residual.
        """
        vis_ast, vis_rfi = model
        a1, a2 = pairs
        gains_bl = baseline_gains(gains, a1, a2)

        vis_obs = gains_bl * (vis_ast + vis_rfi) + 0.01

        res = data_frame_residuals(
            vis_obs,
            gains_bl * vis_ast,
            gains_bl * vis_rfi,
            gains_bl * (vis_ast + vis_rfi),
        )

        np.testing.assert_allclose(
            gains_bl * (vis_ast + vis_rfi) + res["total"], vis_obs, atol=1e-12
        )

    def test_a_perfect_model_leaves_zero_residual(self, model, gains, pairs):
        vis_ast, vis_rfi = model
        a1, a2 = pairs
        gains_bl = baseline_gains(gains, a1, a2)

        vis_obs = gains_bl * (vis_ast + vis_rfi)
        res = data_frame_residuals(
            vis_obs,
            gains_bl * vis_ast,
            gains_bl * vis_rfi,
            gains_bl * (vis_ast + vis_rfi),
        )

        np.testing.assert_allclose(res["total"], 0.0, atol=1e-12)

    def test_subtracting_the_raw_model_does_not_close(self, model, gains, pairs):
        """The old behaviour, kept as a guard against reverting it."""
        vis_ast, vis_rfi = model
        a1, a2 = pairs
        gains_bl = baseline_gains(gains, a1, a2)

        vis_obs = gains_bl * (vis_ast + vis_rfi)
        old_residual = vis_obs - (vis_ast + vis_rfi)

        assert not np.allclose(old_residual, 0.0, atol=1e-8)

    def test_reduces_to_the_old_behaviour_under_unity_gains(self, model):
        """No change for existing UnitaryGains results, so no refs move."""
        vis_ast, vis_rfi = model

        res = data_frame_residuals(
            vis_ast * 0 + 5.0, vis_ast, vis_rfi, vis_ast + vis_rfi
        )

        np.testing.assert_allclose(res["ast"], 5.0 - vis_ast)
        np.testing.assert_allclose(res["rfi"], 5.0 - vis_rfi)
        np.testing.assert_allclose(res["total"], 5.0 - (vis_ast + vis_rfi))

    def test_per_component_residuals_use_the_same_gain(self, model, gains, pairs):
        vis_ast, vis_rfi = model
        a1, a2 = pairs
        gains_bl = baseline_gains(gains, a1, a2)
        vis_obs = gains_bl * (vis_ast + vis_rfi)

        res = data_frame_residuals(
            vis_obs,
            gains_bl * vis_ast,
            gains_bl * vis_rfi,
            gains_bl * (vis_ast + vis_rfi),
        )

        # Removing only the astronomical model leaves exactly the gained RFI.
        np.testing.assert_allclose(res["ast"], gains_bl * vis_rfi, atol=1e-12)
        np.testing.assert_allclose(res["rfi"], gains_bl * vis_ast, atol=1e-12)

    def test_summing_the_parts_reproduces_the_old_total(self, model, gains, pairs):
        """Passing the total explicitly changes nothing for the usual case."""
        vis_ast, vis_rfi = model
        a1, a2 = pairs
        gains_bl = baseline_gains(gains, a1, a2)
        vis_obs = gains_bl * (vis_ast + vis_rfi) + 0.01
        gained_ast, gained_rfi = gains_bl * vis_ast, gains_bl * vis_rfi

        res = data_frame_residuals(
            vis_obs, gained_ast, gained_rfi, gained_ast + gained_rfi
        )

        np.testing.assert_allclose(
            res["total"], vis_obs - (gained_ast + gained_rfi), atol=1e-12
        )

    def test_the_given_total_is_used_not_the_sum(self, model, gains, pairs):
        """The stored forward model wins: a component may gain only one term.

        See the commented-out variant in components/gains.py, where the
        astronomical term alone carries the gain.
        """
        vis_ast, vis_rfi = model
        a1, a2 = pairs
        gains_bl = baseline_gains(gains, a1, a2)
        gained_ast, gained_rfi = gains_bl * vis_ast, gains_bl * vis_rfi
        stored_total = gained_ast + vis_rfi          # RFI left un-gained
        vis_obs = stored_total

        res = data_frame_residuals(vis_obs, gained_ast, gained_rfi, stored_total)

        np.testing.assert_allclose(res["total"], 0.0, atol=1e-12)
        assert not np.allclose(
            vis_obs - (gained_ast + gained_rfi), 0.0, atol=1e-8
        )



class TestGainedModelMean:
    """E[g*m] is not E[g]E[m] once the gains and the model covary."""

    def test_forms_the_gained_model_before_reducing(self):
        # Both gain and model rise across the two samples, i.e. they covary --
        # which posterior draws from a joint fit generally do.
        gains_bl = np.array([1.0 + 0j, 2.0 + 0j])[:, None, None, None]
        model = np.array([1.0 + 0j, 3.0 + 0j])[:, None, None, None]

        correct = gained_model_mean(gains_bl, model)
        naive = gains_bl.mean(axis=0) * model.mean(axis=0)

        # E[gm] = (1*1 + 2*3) / 2 = 3.5;  E[g]E[m] = 1.5 * 2.0 = 3.0
        np.testing.assert_allclose(correct.ravel(), [3.5])
        np.testing.assert_allclose(naive.ravel(), [3.0])
        assert not np.allclose(correct, naive)

    def test_single_sample_matches_the_naive_order(self):
        rng = np.random.default_rng(4)
        shape = (1, 6, 2, 3)
        gains_bl = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        model = rng.normal(size=shape) + 1j * rng.normal(size=shape)

        np.testing.assert_allclose(
            gained_model_mean(gains_bl, model),
            gains_bl.mean(axis=0) * model.mean(axis=0),
        )

    def test_residuals_close_against_the_gained_model(self):
        """The identity the whole change exists to preserve, multi-sample."""
        rng = np.random.default_rng(5)
        shape = (3, 6, 2, 4)
        gains_bl = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        ast = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        rfi = rng.normal(size=shape) + 1j * rng.normal(size=shape)

        gained_ast = gained_model_mean(gains_bl, ast)
        gained_rfi = gained_model_mean(gains_bl, rfi)
        vis_obs = gained_ast + gained_rfi + 0.5

        res = data_frame_residuals(
            vis_obs, gained_ast, gained_rfi, gained_ast + gained_rfi
        )

        np.testing.assert_allclose(
            gained_ast + gained_rfi + res["total"], vis_obs, atol=1e-12
        )


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
# total_model
# ---------------------------------------------------------------------------

class TestTotalModel:
    """The stored forward model wins, except where the gains were substituted."""

    @pytest.fixture
    def parts(self):
        rng = np.random.default_rng(8)
        shape = (6, 2, 1)
        gained_ast = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        gained_rfi = rng.normal(size=shape) + 1j * rng.normal(size=shape)

        return gained_ast, gained_rfi

    def test_the_stored_model_is_used_where_the_gains_were_good(self, parts):
        gained_ast, gained_rfi = parts
        stored = gained_ast + gained_rfi + 7.0        # deliberately not the sum
        bad_bl = np.zeros(gained_ast.shape, dtype=bool)

        out = total_model(stored, gained_ast, gained_rfi, bad_bl)

        np.testing.assert_allclose(out, stored)

    def test_substituted_baselines_fall_back_to_the_sum(self, parts):
        """The stored value predates the substitution and still carries the zero."""
        gained_ast, gained_rfi = parts
        stored = gained_ast + gained_rfi
        stored[2] = 0.0                               # the dead antenna's baseline
        bad_bl = np.zeros(gained_ast.shape, dtype=bool)
        bad_bl[2] = True

        out = total_model(stored, gained_ast, gained_rfi, bad_bl)

        np.testing.assert_allclose(out[2], (gained_ast + gained_rfi)[2])
        np.testing.assert_allclose(out[[0, 1, 3, 4, 5]], stored[[0, 1, 3, 4, 5]])

    def test_a_non_finite_stored_value_does_not_survive(self, parts):
        gained_ast, gained_rfi = parts
        stored = gained_ast + gained_rfi
        stored[4] = np.nan
        bad_bl = np.zeros(gained_ast.shape, dtype=bool)
        bad_bl[4] = True

        out = total_model(stored, gained_ast, gained_rfi, bad_bl)

        assert np.all(np.isfinite(out))

    def test_the_choice_is_made_per_sample(self, parts):
        """One bad sample must not discard the stored model on the other.

        Reducing the mask over samples first -- the bug -- rebuilds the total
        from the two parts on every sample of the cell.
        """
        gained_ast, gained_rfi = parts
        gained_ast = np.stack([gained_ast, 2 * gained_ast])
        gained_rfi = np.stack([gained_rfi, 2 * gained_rfi])

        stored = gained_ast + gained_rfi + 5.0        # deliberately not the sum
        stored[0] = 0.0                               # sample 0 had a bad gain

        bad_bl = np.zeros(stored.shape, dtype=bool)
        bad_bl[0] = True

        out = total_model(stored, gained_ast, gained_rfi, bad_bl)

        np.testing.assert_allclose(out[0], (gained_ast + gained_rfi)[0])
        np.testing.assert_allclose(out[1], stored[1])

        # What reducing over samples first would have given.
        any_sample = np.where(
            bad_bl.any(axis=0), gained_ast + gained_rfi, stored
        )
        assert not np.allclose(out[1], any_sample[1])

    def test_works_on_dask_arrays(self, parts):
        """write_results_ms passes the zarr's dask arrays straight in."""
        da = pytest.importorskip("dask.array")
        gained_ast, gained_rfi = parts
        stored = gained_ast + gained_rfi + 3.0
        bad_bl = np.zeros(stored.shape, dtype=bool)
        bad_bl[1] = True

        out = total_model(
            da.from_array(stored, chunks=(2, 2, 1)),
            da.from_array(gained_ast, chunks=(2, 2, 1)),
            da.from_array(gained_rfi, chunks=(2, 2, 1)),
            da.from_array(bad_bl, chunks=(2, 2, 1)),
        )

        np.testing.assert_allclose(
            np.asarray(out), np.where(bad_bl, gained_ast + gained_rfi, stored)
        )

    def test_nothing_bad_means_nothing_changes(self, parts):
        gained_ast, gained_rfi = parts
        stored = gained_ast + gained_rfi

        out = total_model(
            stored, gained_ast, gained_rfi, np.zeros(stored.shape, dtype=bool)
        )

        np.testing.assert_allclose(out, gained_ast + gained_rfi)
