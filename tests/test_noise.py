"""Tests for tabascal.noise — per-baseline visibility noise."""

import numpy as np
import pytest

from tabascal.noise import (
    broadcast_to_vis,
    per_baseline_sigma,
    read_noise_file,
    representative_sigma,
)


N_TIME, N_BL = 5, 6


def _sigma_column(per_bl, n_time=N_TIME, n_corr=1):
    """A SIGMA column: per-baseline values repeated over time, as an MS stores it."""

    per_bl = np.asarray(per_bl, dtype=float)

    return np.tile(per_bl, (n_time, 1)).reshape(-1, 1).repeat(n_corr, axis=1)


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

    def test_reduces_over_time_with_a_median_not_a_mean(self):
        """A few corrupted rows must not drag a baseline's estimate."""
        col = _sigma_column(np.full(N_BL, 1.0))
        col = col.reshape(N_TIME, N_BL, 1)
        col[0, :, 0] = 1000.0  # one bad timestep
        col = col.reshape(-1, 1)

        out = per_baseline_sigma(col, N_TIME, N_BL)

        np.testing.assert_allclose(out, 1.0)
        assert not np.allclose(out, col.reshape(N_TIME, N_BL)[:, 0].mean())

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

        with pytest.raises(ValueError, match="must carry either"):
            read_noise_file(str(path), N_BL)


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

    def test_scalar_noise_is_unchanged(self):
        """Uniform data must give exactly what it always did."""
        rng = np.random.default_rng(0)
        shape = (N_BL, 3, 4)
        true = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        pred = true + 0.1
        flags = np.zeros(shape, dtype=bool)

        scalar = self._chi2(pred, true, 0.5, flags)
        uniform = self._chi2(pred, true, np.full(N_BL, 0.5), flags)

        assert scalar == pytest.approx(uniform, rel=1e-12)

    def test_each_baseline_divides_by_its_own_noise(self):
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
        assert got == pytest.approx(expected, rel=1e-12)
        assert got == pytest.approx(0.5, rel=1e-12)

        # Reversing the noise mis-pairs it, and now that is visible.
        assert self._chi2(pred, true, noise[::-1].copy(), flags) != pytest.approx(got)

    def test_alignment_survives_a_ragged_flag_mask(self):
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
        assert got == pytest.approx(expected, rel=1e-12)
