"""Tests for tabascal.write — per-baseline gains and residual framing.

Every case uses a non-unit, non-uniform gain: a unity gain cannot distinguish
``g_p conj(g_q)`` from ``|g_p|^2``, nor a gained model from a raw one.
"""

import warnings

import numpy as np
import pytest

from tabascal.write import (
    baseline_gains,
    data_frame_residuals,
    gained_model_mean,
    read_antenna_pairs,
    total_model,
    unit_bad_gains,
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
    """Stand-in for the MS dataset, with distinct ANTENNA1/ANTENNA2 columns.

    Carries TIME too: the MS's own baseline count is derived from it, so the
    layout checks need a whole column rather than one timestep's worth.
    """

    def __init__(self, a1, a2, times=None):
        if times is None:
            times = np.zeros(len(np.asarray(a1)))
        self.ANTENNA1 = type("V", (), {"data": _FakeCol(a1)})()
        self.ANTENNA2 = type("V", (), {"data": _FakeCol(a2)})()
        self.TIME = type("V", (), {"data": _FakeCol(times)})()


def _time_major_ms(n_ant: int = 4, n_time: int = 3):
    """A well-formed time-major fake MS and its one timestep of pairs."""

    a1_bl, a2_bl = np.triu_indices(n_ant, k=1)
    n_bl = len(a1_bl)
    times = np.repeat(np.arange(n_time, dtype=float), n_bl)

    xds = _FakeMS(np.tile(a1_bl, n_time), np.tile(a2_bl, n_time), times)

    return xds, a1_bl, a2_bl


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
        xds, a1_bl, a2_bl = _time_major_ms()
        n_bl = len(a1_bl)

        a1, a2 = read_antenna_pairs(xds, n_bl)

        assert len(a1) == n_bl and len(a2) == n_bl
        np.testing.assert_array_equal(a2, a2_bl)


class TestBaselineCountsMustMatch:
    """The results zarr and the MS have to describe the same array.

    The count is taken from the zarr, so a mismatch used to surface as an
    ordering complaint, an opaque dask chunk error, or -- when the row counts
    coincided -- silently misplaced gains.
    """

    def test_more_baselines_in_the_results_than_the_ms(self):
        xds, _, _ = _time_major_ms(n_ant=3, n_time=2)   # 3 baselines

        with pytest.raises(ValueError) as excinfo:
            read_antenna_pairs(xds, 6)

        message = str(excinfo.value)
        assert "6" in message and "3" in message
        # Not an ordering problem: this MS is perfectly time-major.
        assert "time-major" not in message

    def test_fewer_baselines_in_the_results_than_the_ms(self):
        """The case the old check passed silently."""
        xds, _, _ = _time_major_ms(n_ant=4, n_time=2)   # 6 baselines

        with pytest.raises(ValueError, match="does not belong"):
            read_antenna_pairs(xds, 3)

    def test_ragged_row_counts_are_rejected(self):
        """Rows that do not divide into whole timesteps break the reshape."""
        a1_bl, a2_bl = np.triu_indices(4, k=1)
        xds = _FakeMS(
            np.tile(a1_bl, 2)[:-1],
            np.tile(a2_bl, 2)[:-1],
            np.repeat([0.0, 1.0], 6)[:-1],
        )

        with pytest.raises(ValueError, match="whole number of baselines"):
            read_antenna_pairs(xds, 6)


class TestBaselineOrderIsFixedAcrossTimesteps:
    """The (n_time, n_bl) reshape needs the same pair sequence every timestep."""

    def test_a_permuted_timestep_is_rejected(self):
        a1_bl, a2_bl = np.triu_indices(4, k=1)
        perm = np.array([3, 1, 0, 5, 4, 2])
        a1_col = np.concatenate([a1_bl, a1_bl[perm]])
        a2_col = np.concatenate([a2_bl, a2_bl[perm]])
        xds = _FakeMS(a1_col, a2_col, np.repeat([0.0, 1.0], 6))

        with pytest.raises(ValueError, match="differs between timesteps"):
            read_antenna_pairs(xds, 6)

    def test_a_consistent_order_is_accepted(self):
        xds, a1_bl, a2_bl = _time_major_ms(n_time=4)

        a1, a2 = read_antenna_pairs(xds, len(a1_bl))

        np.testing.assert_array_equal(a1, a1_bl)
        np.testing.assert_array_equal(a2, a2_bl)


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


class TestRowOrderIsChecked:
    """Reviewer's point: `[:n_bl]` assumes a time-major MS."""

    def test_baseline_major_ordering_is_rejected(self):
        """All times of one baseline first repeats each pair across the rows."""
        a1_bl, a2_bl = np.triu_indices(4, k=1)
        n_time = 3
        xds = _FakeMS(
            np.repeat(a1_bl, n_time),
            np.repeat(a2_bl, n_time),
            np.tile(np.arange(n_time, dtype=float), len(a1_bl)),
        )

        with pytest.raises(ValueError, match="not ordered time-major"):
            read_antenna_pairs(xds, len(a1_bl))

    def test_partially_repeated_pairs_are_rejected(self):
        a1_col, a2_col = np.triu_indices(4, k=1)
        a1_col, a2_col = a1_col.copy(), a2_col.copy()
        a1_col[-1], a2_col[-1] = a1_col[0], a2_col[0]   # one duplicate pair

        with pytest.raises(ValueError, match="distinct antenna pairs"):
            read_antenna_pairs(_FakeMS(a1_col, a2_col), len(a1_col))


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


class TestWarnBadGains:

    def test_warns_with_the_count_and_the_antennas(self):
        bad = np.zeros((2, 4, 3, 5), dtype=bool)
        bad[0, 1] = True
        bad[1, 3, 0, 0] = True

        with pytest.warns(RuntimeWarning) as record:
            n_bad = warn_bad_gains(bad)

        message = str(record[0].message)
        assert n_bad == 16
        assert "16 of 120" in message
        assert "[1, 3]" in message                  # the antennas, not the samples
        assert "set to 1" in message

    def test_silent_when_nothing_was_substituted(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert warn_bad_gains(np.zeros((1, 4, 2, 3), dtype=bool)) == 0


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

    def test_nothing_bad_means_nothing_changes(self, parts):
        gained_ast, gained_rfi = parts
        stored = gained_ast + gained_rfi

        out = total_model(
            stored, gained_ast, gained_rfi, np.zeros(stored.shape, dtype=bool)
        )

        np.testing.assert_allclose(out, gained_ast + gained_rfi)
