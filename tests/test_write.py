"""Tests for tabascal.write — per-baseline gains and residual framing.

Both behaviours here are exactly right under ``UnitaryGains`` and wrong under any
fitted gain, which is why every case uses a **non-unit, non-uniform** gain with
distinct amplitudes and phases per antenna. A unity gain cannot distinguish
``g_p conj(g_q)`` from ``|g_p|^2``, nor a gained model from a raw one.
"""

import numpy as np
import pytest

from tabascal.write import (
    _unity_gains_or_raise,
    baseline_gains,
    data_frame_residuals,
    gained_model_mean,
    mean_baseline_gains,
    read_antenna_pairs,
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
# baseline_gains
# ---------------------------------------------------------------------------

class TestBaselineGains:

    def test_matches_the_definition(self, gains, pairs):
        a1, a2 = pairs
        expected = gains[a1] * np.conj(gains[a2])

        np.testing.assert_allclose(baseline_gains(gains, a1, a2), expected)

    def test_uses_both_antennas(self, gains, pairs):
        """The regression: ANTENNA1 twice gives |g_p|^2 on every baseline."""
        a1, a2 = pairs

        wrong = gains[a1] * np.conj(gains[a1])

        assert not np.allclose(baseline_gains(gains, a1, a2), wrong)

    def test_the_wrong_form_is_real_and_positive(self, gains, pairs):
        """Why it matters: indexing a1 twice discards all phase information."""
        a1, a2 = pairs

        wrong = gains[a1] * np.conj(gains[a1])
        assert np.allclose(wrong.imag, 0.0)
        assert np.all(wrong.real > 0.0)

        # The correct gain carries a non-zero phase on these baselines.
        assert not np.allclose(baseline_gains(gains, a1, a2).imag, 0.0)

    def test_is_hermitian_under_baseline_reversal(self, gains, pairs):
        """Swapping the antenna order conjugates the baseline gain."""
        a1, a2 = pairs

        forward = baseline_gains(gains, a1, a2)
        reversed_ = baseline_gains(gains, a2, a1)

        np.testing.assert_allclose(forward, np.conj(reversed_))

    def test_unity_gains_give_unity(self, pairs):
        """Which is exactly why the bug stayed latent."""
        a1, a2 = pairs
        ones = np.ones((4, 1, 1), dtype=np.complex128)

        np.testing.assert_allclose(baseline_gains(ones, a1, a2), 1.0)

    def test_works_on_dask_arrays(self, gains, pairs):
        """write_results_ms passes dask arrays through this."""
        da = pytest.importorskip("dask.array")
        a1, a2 = pairs

        out = baseline_gains(da.from_array(gains, chunks=-1), a1, a2)

        np.testing.assert_allclose(np.asarray(out), gains[a1] * np.conj(gains[a2]))


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

        res = data_frame_residuals(vis_obs, gains_bl * vis_ast, gains_bl * vis_rfi)

        np.testing.assert_allclose(
            gains_bl * (vis_ast + vis_rfi) + res["total"], vis_obs, atol=1e-12
        )

    def test_a_perfect_model_leaves_zero_residual(self, model, gains, pairs):
        vis_ast, vis_rfi = model
        a1, a2 = pairs
        gains_bl = baseline_gains(gains, a1, a2)

        vis_obs = gains_bl * (vis_ast + vis_rfi)
        res = data_frame_residuals(vis_obs, gains_bl * vis_ast, gains_bl * vis_rfi)

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

        res = data_frame_residuals(vis_ast * 0 + 5.0, vis_ast, vis_rfi)

        np.testing.assert_allclose(res["ast"], 5.0 - vis_ast)
        np.testing.assert_allclose(res["rfi"], 5.0 - vis_rfi)
        np.testing.assert_allclose(res["total"], 5.0 - (vis_ast + vis_rfi))

    def test_per_component_residuals_use_the_same_gain(self, model, gains, pairs):
        vis_ast, vis_rfi = model
        a1, a2 = pairs
        gains_bl = baseline_gains(gains, a1, a2)
        vis_obs = gains_bl * (vis_ast + vis_rfi)

        res = data_frame_residuals(vis_obs, gains_bl * vis_ast, gains_bl * vis_rfi)

        # Removing only the astronomical model leaves exactly the gained RFI.
        np.testing.assert_allclose(res["ast"], gains_bl * vis_rfi, atol=1e-12)
        np.testing.assert_allclose(res["rfi"], gains_bl * vis_ast, atol=1e-12)


# ---------------------------------------------------------------------------
# read_antenna_pairs
# ---------------------------------------------------------------------------

class _FakeCol:
    def __init__(self, values):
        self._values = np.asarray(values)

    def __getitem__(self, idx):
        return _FakeCol(self._values[idx])

    def compute(self):
        return self._values


class _FakeMS:
    """Stand-in for the MS dataset, with distinct ANTENNA1/ANTENNA2 columns."""

    def __init__(self, a1, a2):
        self.ANTENNA1 = type("V", (), {"data": _FakeCol(a1)})()
        self.ANTENNA2 = type("V", (), {"data": _FakeCol(a2)})()


class TestReadAntennaPairs:

    def test_reads_both_antenna_columns(self):
        """The regression guard: reading ANTENNA1 twice is the original bug."""
        a1_col, a2_col = np.triu_indices(4, k=1)
        xds = _FakeMS(a1_col, a2_col)

        a1, a2 = read_antenna_pairs(xds, len(a1_col))

        np.testing.assert_array_equal(a1, a1_col)
        np.testing.assert_array_equal(a2, a2_col)
        assert not np.array_equal(a1, a2)

    def test_takes_only_the_first_baseline_set(self):
        """The columns repeat per timestep; one set of baselines is wanted."""
        a1_col, a2_col = np.triu_indices(4, k=1)
        n_bl = len(a1_col)
        xds = _FakeMS(np.tile(a1_col, 3), np.tile(a2_col, 3))

        a1, a2 = read_antenna_pairs(xds, n_bl)

        assert len(a1) == n_bl and len(a2) == n_bl
        np.testing.assert_array_equal(a2, a2_col)


# ---------------------------------------------------------------------------
# Sample-axis handling
# ---------------------------------------------------------------------------

class TestSampleAxis:
    """E[g_p conj(g_q)] is not E[g_p] conj(E[g_q]) once the gains vary.

    Every current writer of the results zarr stores exactly one sample, so this
    is latent rather than live -- but forming the product before reducing costs
    nothing and removes the trap.
    """

    def test_ant_axis_selects_the_antenna_axis(self):
        """With a leading sample axis, axis 0 is samples, not antennas."""
        rng = np.random.default_rng(1)
        gains = rng.normal(size=(2, 4, 3, 1)) + 1j * rng.normal(size=(2, 4, 3, 1))
        a1, a2 = np.triu_indices(4, k=1)

        out = baseline_gains(gains, a1, a2, ant_axis=1)

        assert out.shape == (2, len(a1), 3, 1)
        np.testing.assert_allclose(out, gains[:, a1] * np.conj(gains[:, a2]))

    def test_product_before_mean_differs_from_mean_before_product(self):
        """The distinction the reduction order makes, made concrete.

        E[XY] equals E[X]E[Y] only when X and Y are uncorrelated, so the two
        antennas have to vary *together* for the difference to appear -- both
        gains rise from 1 to 2 across the two samples here.
        """
        gains = np.array(
            [
                [1.0 + 0.0j, 1.0 + 0.0j],
                [2.0 + 0.0j, 2.0 + 0.0j],
            ]
        )[:, :, None, None]
        a1, a2 = np.array([0]), np.array([1])

        correct = baseline_gains(gains, a1, a2, ant_axis=1).mean(axis=0)
        naive = baseline_gains(gains.mean(axis=0), a1, a2)

        # E[g^2] = (1 + 4) / 2 = 2.5, but E[g]^2 = 1.5^2 = 2.25
        np.testing.assert_allclose(correct.ravel(), [2.5])
        np.testing.assert_allclose(naive.ravel(), [2.25])
        assert not np.allclose(correct, naive)

    def test_single_sample_is_unaffected(self):
        """Which is why current results are unchanged."""
        rng = np.random.default_rng(2)
        gains = rng.normal(size=(1, 4, 2, 1)) + 1j * rng.normal(size=(1, 4, 2, 1))
        a1, a2 = np.triu_indices(4, k=1)

        before = baseline_gains(gains, a1, a2, ant_axis=1).mean(axis=0)
        after = baseline_gains(gains.mean(axis=0), a1, a2)

        np.testing.assert_allclose(before, after)


# ---------------------------------------------------------------------------
# The legacy-layout unity guard
# ---------------------------------------------------------------------------

class _FakeZarr(dict):
    def __init__(self, gains=None):
        super().__init__()
        if gains is not None:
            self["gains"] = True
            self.gains = type("V", (), {"data": np.asarray(gains)})()


class TestUnityGainsGuard:

    def test_unity_gains_pass(self):
        assert _unity_gains_or_raise(_FakeZarr(np.ones((2, 4, 1, 1)))) == 1

    def test_absent_gains_pass(self):
        assert _unity_gains_or_raise(_FakeZarr()) == 1

    def test_non_unity_gains_raise(self):
        with pytest.raises(NotImplementedError, match="3-d ast_vis layout"):
            _unity_gains_or_raise(_FakeZarr(np.full((2, 4, 1, 1), 1.5)))

    def test_samples_averaging_to_unity_still_raise(self):
        """The guard tests every sample, not their mean.

        Samples of 0.9 and 1.1 average to exactly 1, so a mean-based check would
        wave through precisely the case this exists to catch.
        """
        gains = np.stack(
            [np.full((4, 1, 1), 0.9), np.full((4, 1, 1), 1.1)]
        )
        assert np.allclose(gains.mean(axis=0), 1.0)

        with pytest.raises(NotImplementedError, match="3-d ast_vis layout"):
            _unity_gains_or_raise(_FakeZarr(gains))


class TestMeanBaselineGains:
    """The reduction order, pinned where write_results_ms actually uses it."""

    def test_forms_the_product_before_reducing(self):
        gains = np.array(
            [
                [1.0 + 0.0j, 1.0 + 0.0j],
                [2.0 + 0.0j, 2.0 + 0.0j],
            ]
        )[:, :, None, None]
        a1, a2 = np.array([0]), np.array([1])

        out = mean_baseline_gains(gains, a1, a2)

        np.testing.assert_allclose(out.ravel(), [2.5])          # E[g^2]
        assert not np.allclose(out.ravel(), [2.25])             # not E[g]^2

    def test_single_sample_matches_the_naive_order(self):
        rng = np.random.default_rng(3)
        gains = rng.normal(size=(1, 4, 2, 1)) + 1j * rng.normal(size=(1, 4, 2, 1))
        a1, a2 = np.triu_indices(4, k=1)

        np.testing.assert_allclose(
            mean_baseline_gains(gains, a1, a2),
            baseline_gains(gains.mean(axis=0), a1, a2),
        )

    def test_drops_the_sample_axis(self):
        gains = np.ones((3, 4, 2, 5), dtype=complex)
        a1, a2 = np.triu_indices(4, k=1)

        assert mean_baseline_gains(gains, a1, a2).shape == (len(a1), 2, 5)


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

        res = data_frame_residuals(vis_obs, gained_ast, gained_rfi)

        np.testing.assert_allclose(
            gained_ast + gained_rfi + res["total"], vis_obs, atol=1e-12
        )
