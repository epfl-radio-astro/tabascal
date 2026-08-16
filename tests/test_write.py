"""Tests for tabascal.write — per-baseline gains and residual framing.

Both behaviours here are exactly right under ``UnitaryGains`` and wrong under any
fitted gain, which is why every case uses a **non-unit, non-uniform** gain with
distinct amplitudes and phases per antenna. A unity gain cannot distinguish
``g_p conj(g_q)`` from ``|g_p|^2``, nor a gained model from a raw one.
"""

import numpy as np
import pytest

from tabascal.write import (
    baseline_gains,
    data_frame_residuals,
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

        res = data_frame_residuals(vis_obs, vis_ast, vis_rfi, gains_bl)

        np.testing.assert_allclose(
            gains_bl * (vis_ast + vis_rfi) + res["total"], vis_obs, atol=1e-12
        )

    def test_a_perfect_model_leaves_zero_residual(self, model, gains, pairs):
        vis_ast, vis_rfi = model
        a1, a2 = pairs
        gains_bl = baseline_gains(gains, a1, a2)

        vis_obs = gains_bl * (vis_ast + vis_rfi)
        res = data_frame_residuals(vis_obs, vis_ast, vis_rfi, gains_bl)

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

        res = data_frame_residuals(vis_ast * 0 + 5.0, vis_ast, vis_rfi, 1)

        np.testing.assert_allclose(res["ast"], 5.0 - vis_ast)
        np.testing.assert_allclose(res["rfi"], 5.0 - vis_rfi)
        np.testing.assert_allclose(res["total"], 5.0 - (vis_ast + vis_rfi))

    def test_per_component_residuals_use_the_same_gain(self, model, gains, pairs):
        vis_ast, vis_rfi = model
        a1, a2 = pairs
        gains_bl = baseline_gains(gains, a1, a2)
        vis_obs = gains_bl * (vis_ast + vis_rfi)

        res = data_frame_residuals(vis_obs, vis_ast, vis_rfi, gains_bl)

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
