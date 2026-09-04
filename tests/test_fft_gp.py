"""Tests for FFT-based Gaussian Process utilities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tabascal.fft_gp import (
    domain_k,
    domain_ss,
    fourier_cut,
    fourier_uncut,
    signal_to_latent,
    signal_to_latent_init,
    latent_to_signal_init,
    latent_to_signal,
    pad,
    pad_domain,
    pad_domain_k,
    pad_domain_specs,
    pk_cut,
    pow_spec,
    pow_spec_nd,
    supersample,
    supersample_domain,
    supersample_domain_k,
    supersample_domain_specs,
    supersample_fourier,
)
from tabascal.timing import clear_timings, disable_timings


@pytest.fixture(autouse=True)
def manage_timings():
    """Fixture to ensure a clean timing state for each test."""
    clear_timings()
    disable_timings()
    yield
    disable_timings()
    clear_timings()

class TestSupersample:
    """Test supersampling functions."""

    def test_supersample_domain_specs_1d(self):
        """Test supersampling specs for 1D domain."""
        ns = [3]
        dxs = [0.1]
        ss_factors = [2]
        n_ss, dx_ss = supersample_domain_specs(ns, dxs, ss_factors)

        assert n_ss == [6]
        assert jnp.isclose(dx_ss[0], 0.05)

    def test_supersample_domain_specs_2d(self):
        """Test supersampling specs for 2D domain."""
        ns = [1, 3]
        dxs = [200e3, 2]
        ss_factors = [2, 3]
        n_ss, dx_ss = supersample_domain_specs(ns, dxs, ss_factors)

        assert n_ss == [2, 9]
        assert jnp.isclose(dx_ss[0], 100e3)
        assert jnp.isclose(dx_ss[1], 2.0 / 3.0)

    def test_supersample_domain_specs_length_mismatch(self):
        """Test that length mismatch raises error."""
        ns = [1]
        dxs = [0.1, 0.2]
        ss_factors = [3]
        with pytest.raises(ValueError, match="Length mismatch"):
            supersample_domain_specs(ns, dxs, ss_factors)

    def test_supersample_domain_1d(self):
        """Test domain supersampling in 1D."""
        x = jnp.linspace(0, 10, 11)
        ns = [len(x)]
        dxs = [float(jnp.diff(x)[0])]
        x0s = [float(x[0])]
        ss_factors = [2]
        xs_ss = supersample_domain(ns, dxs, x0s, ss_factors)

        assert len(xs_ss) == 1
        assert len(xs_ss[0]) == 22
        assert jnp.isclose(xs_ss[0][0], x[0])
        assert jnp.isclose(xs_ss[0][-1], x[-1] + 0.5)

    def test_supersample_domain_k_1d(self):
        """Test k-domain supersampling in 1D."""
        x = jnp.linspace(0, 10, 11)
        ns = [len(x)]
        dxs = [float(jnp.diff(x)[0])]
        ss_factors = [2]
        ks_ss = supersample_domain_k(ns, dxs, ss_factors)

        assert len(ks_ss) == 1
        assert len(ks_ss[0]) == 22

    def test_supersample_signal_1d(self):
        """Test signal supersampling in 1D."""
        # Create a simple sine wave
        x = jnp.linspace(0, 2 * jnp.pi, 32)
        y = jnp.sin(x)

        # Supersample by factor of 2
        y_ss = supersample(y, [2])

        assert y_ss.shape == (64,)
        # Check that energy is preserved (up to numerical precision)
        # Supersampling should approximately double the sum of absolute values
        assert jnp.isclose(jnp.abs(y_ss).sum(), jnp.abs(y).sum() * 2, rtol=1e-3)

    def test_supersample_signal_2d(self):
        """Test signal supersampling in 2D."""
        y = jnp.ones((8, 8))
        y_ss = supersample(y, [2, 2])

        assert y_ss.shape == (16, 16)
        # For a constant signal, supersampling should preserve the constant
        assert jnp.allclose(y_ss, 1.0, rtol=1e-5)

    def test_supersample_jit_compatible(self):
        """Test that supersample is JIT-compatible."""
        @jax.jit
        def supersample_jitted(y):
            return supersample(y, [2, 2])

        y = jnp.ones((8, 8))
        y_ss = supersample_jitted(y)
        assert y_ss.shape == (16, 16)


class TestDomainK:
    """Test k-domain calculation functions."""

    def test_domain_k_1d(self):
        """Test k-domain calculation in 1D."""
        x = jnp.linspace(0, 10, 11)
        ns = [len(x)]
        dxs = [float(jnp.diff(x)[0])]
        ks = domain_k(ns, dxs)

        assert len(ks) == 1
        assert len(ks[0]) == len(x)
        # Check that k=0 is in the center (after fftshift)
        assert jnp.isclose(ks[0][len(ks[0]) // 2], 0.0, atol=1e-6)

    def test_domain_k_2d(self):
        """Test k-domain calculation in 2D."""
        x = jnp.linspace(0, 10, 11)
        y = jnp.linspace(0, 5, 6)
        ns = [len(_x) for _x in [x,y]]
        dxs = [float(jnp.diff(_x)[0]) for _x in [x,y]]
        ks = domain_k(ns, dxs)

        assert len(ks) == 2
        assert len(ks[0]) == len(x)
        assert len(ks[1]) == len(y)


class TestPadding:
    """Test padding functions."""

    def test_pad_domain_specs_validation(self):
        """Test that pad_domain_specs validates inputs."""
        x = jnp.linspace(0, 10, 11)
        ns = [len(x)]
        pad_factors = [0.5]

        # Test pad factors < 1.0
        with pytest.raises(ValueError, match="Pad factors must be >= 1.0"):
            pad_domain_specs(ns, pad_factors)

        # Test length mismatch
        with pytest.raises(ValueError, match="Length mismatch"):
            pad_domain_specs(ns, [1.5, 2.0])

    def test_pad_domain_specs_1d(self):
        """Test padding specs for 1D domain."""
        x = jnp.linspace(0, 10, 11)
        ns = [len(x)]
        n_pads = pad_domain_specs(ns, [2.0])

        assert n_pads == [5]  # (11 * (2 - 1) / 2) = 5.5 -> 5

    def test_pad_domain_1d(self):
        """Test domain padding in 1D."""
        x = jnp.linspace(0, 10, 11)
        ns = [len(x)]
        dxs = [float(jnp.diff(x)[0])]
        x0s = [float(x[0])]
        xs_pad = pad_domain(ns, dxs, x0s, [2.0])

        assert len(xs_pad) == 1
        expected_len = 11 + 2 * 5  # 21
        assert len(xs_pad[0]) == expected_len

    def test_pad_domain_k_1d(self):
        """Test k-domain padding in 1D."""
        x = jnp.linspace(0, 10, 11)
        ns = [len(x)]
        dxs = [float(jnp.diff(x)[0])]
        ks_pad = pad_domain_k(ns, dxs, [2.0])

        assert len(ks_pad) == 1
        expected_len = 11 + 2 * 5  # 21
        assert len(ks_pad[0]) == expected_len

    def test_pad_signal_1d(self):
        """Test signal padding in 1D."""
        z = jnp.ones(10)
        z_pad = pad(z, [2.0])

        assert z_pad.shape[0] > 10
        # Check that original region is preserved
        n_pad = int(10 * (2.0 - 1) / 2)
        assert jnp.allclose(z_pad[n_pad:n_pad + 10], 1.0)

    def test_pad_signal_2d(self):
        """Test signal padding in 2D."""
        z = jnp.ones((10, 10))
        z_pad = pad(z, [2.0, 2.0])

        assert z_pad.shape[0] > 10
        assert z_pad.shape[1] > 10

    def test_pad_signal_dimension_mismatch(self):
        """Test that pad validates dimension mismatch."""
        z = jnp.ones((10, 10))
        with pytest.raises(ValueError, match="Length mismatch"):
            pad(z, [2.0])

    def test_pad_jit_compatible(self):
        """Test that pad is JIT-compatible."""
        @jax.jit
        def pad_jitted(z):
            return pad(z, [2.0, 2.0])

        z = jnp.ones((10, 10))
        z_pad = pad_jitted(z)
        assert z_pad.shape[0] > 10


class TestFourierCutting:
    """Test Fourier mode cutting functions."""

    def test_pk_cut_1d(self):
        """Test power spectrum cutting in 1D."""
        # Create a power spectrum with a peak in the middle
        pk = jnp.exp(-jnp.linspace(-5, 5, 51) ** 2)
        idxs, pads = pk_cut(pk, 0.1)

        assert len(idxs) == 1
        assert len(pads) == 1
        assert isinstance(idxs[0], slice)
        # The cut region should be smaller than original
        cut_size = idxs[0].stop - idxs[0].start
        assert cut_size < len(pk)

    def test_pk_cut_2d(self):
        """Test power spectrum cutting in 2D."""
        x = jnp.linspace(-5, 5, 51)
        y = jnp.linspace(-5, 5, 51)
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        pk = jnp.exp(-(X ** 2 + Y ** 2))

        idxs, pads = pk_cut(pk, 0.1)

        assert len(idxs) == 2
        assert len(pads) == 2

    def test_fourier_cut_uncut_roundtrip(self):
        """Test that fourier_cut followed by fourier_uncut preserves shape."""
        # Create a simple signal
        x = jnp.linspace(0, 2 * jnp.pi, 32)
        y = jnp.sin(x)

        # Create a power spectrum (Gaussian in k-space)
        k = jnp.fft.fftshift(jnp.fft.fftfreq(len(x), x[1] - x[0]))
        pk = jnp.exp(-k ** 2)

        # Cut and uncut
        Y_cut = fourier_cut(pk, 0.1, y)
        Y_uncut = fourier_uncut(pk, 0.1, Y_cut)

        assert Y_uncut.shape == (len(x),)

    def test_supersample_fourier_1d(self):
        """Test Fourier-based supersampling in 1D."""
        # Create Fourier modes (shifted ordering: -k to +k)
        Y = jnp.ones(16)
        Y_ss = supersample_fourier(Y, [2])

        assert Y_ss.shape == (32,)

    def test_supersample_fourier_2d(self):
        """Test Fourier-based supersampling in 2D."""
        Y = jnp.ones((8, 8))
        Y_ss = supersample_fourier(Y, [2, 2])

        assert Y_ss.shape == (16, 16)


class TestPowerSpectrum:
    """Test power spectrum functions."""

    def test_pow_spec_1d_shape(self):
        """Test 1D power spectrum shape."""
        k = jnp.linspace(-10, 10, 101)
        pk = pow_spec(k, p0=1.0, k0=1.0, gamma=2.0)

        assert pk.shape == k.shape

    def test_pow_spec_1d_symmetry(self):
        """Test that 1D power spectrum is symmetric."""
        k = jnp.linspace(-10, 10, 101)
        pk = pow_spec(k, p0=1.0, k0=1.0, gamma=2.0)

        # Power spectrum should be symmetric around k=0
        mid = len(k) // 2
        assert jnp.allclose(pk[:mid], pk[-mid:][::-1], rtol=1e-5)

    def test_pow_spec_1d_peak_at_zero(self):
        """Test that 1D power spectrum peaks at k=0."""
        k = jnp.linspace(-10, 10, 101)
        pk = pow_spec(k, p0=1.0, k0=1.0, gamma=2.0)

        # Maximum should be at k=0
        assert jnp.argmax(pk) == len(k) // 2

    def test_pow_spec_nd_validation(self):
        """Test that pow_spec_nd validates input lengths."""
        k1 = jnp.linspace(-10, 10, 101)
        k2 = jnp.linspace(-10, 10, 101)

        with pytest.raises(ValueError, match="must have the same length"):
            pow_spec_nd([k1, k2], p0=1.0, k0s=[1.0], gammas=[2.0, 2.0])

    def test_pow_spec_nd_2d_shape(self):
        """Test 2D power spectrum shape."""
        k1 = jnp.linspace(-10, 10, 21)
        k2 = jnp.linspace(-10, 10, 31)
        pk = pow_spec_nd([k1, k2], p0=1.0, k0s=[1.0, 1.0], gammas=[2.0, 2.0])

        assert pk.shape == (21, 31)

    def test_pow_spec_nd_3d_shape(self):
        """Test 3D power spectrum shape."""
        ks = [jnp.linspace(-5, 5, 11) for _ in range(3)]
        pk = pow_spec_nd(ks, p0=1.0, k0s=[1.0, 1.0, 1.0], gammas=[2.0, 2.0, 2.0])

        assert pk.shape == (11, 11, 11)


class TestDomainSS:
    """Test supersampled domain calculation."""

    def test_domain_ss_1d(self):
        """Test supersampled domain in 1D."""
        x = jnp.linspace(0, 10, 11)
        ns = [len(x)]
        dxs = [float(jnp.diff(x)[0])]
        x0s = [float(x[0])]
        xs_ss = domain_ss(ns, dxs, x0s, ss_factors=[2], pad_factors=[1.5])

        assert len(xs_ss) == 1
        # Should have supersampled resolution
        dx_ss = jnp.diff(xs_ss[0])[0]
        dx_orig = jnp.diff(x)[0]
        assert jnp.isclose(dx_ss, dx_orig / 2, rtol=1e-4)

    def test_domain_ss_2d(self):
        """Test supersampled domain in 2D."""
        x = jnp.linspace(0, 10, 11)
        y = jnp.linspace(0, 5, 6)
        ns = [len(_x) for _x in [x, y]]
        dxs = [float(jnp.diff(_x)[0]) for _x in [x, y]]
        x0s = [float(_x[0]) for _x in [x, y]]
        xs_ss = domain_ss(ns, dxs, x0s, ss_factors=[2, 3], pad_factors=[1.5, 1.5])

        assert len(xs_ss) == 2

    @pytest.mark.requires_double
    def test_domain_ss_is_affine_in_dx_x0(self):
        """``domain_ss`` is affine in ``(dx, x0)``: the real grid equals
        ``x0 + dx * unit_grid`` where ``unit_grid`` is built with ``dx=1, x0=0``.

        ``config.TabConfig._set_freqs_times`` relies on exactly this property: it
        builds the unit grid via ``domain_ss`` (jax, f32 under single precision)
        and applies the real ``freqs[0]/chan_width`` and ``times[0]/int_time``
        offset/scale in numpy f64, because the real grids carry large magnitudes
        (freqs ~1e9, times_jd ~2.4e6) that lose all usable precision in f32. If
        this identity ever breaks, the single-precision fine grids are silently
        wrong, so guard it directly here. Checked in double precision, where
        ``domain_ss`` itself is f64-accurate at these magnitudes.
        """
        ns = [11, 6]
        dxs = [2.0e6, 3.0]        # freq-like and time-like scales
        x0s = [1.4e9, 2.4e6]      # large offsets, as in real freqs / times_jd
        ss_factors = [2, 3]
        pad_factors = [1.5, 1.5]

        real = domain_ss(ns, dxs, x0s, ss_factors, pad_factors)
        unit = domain_ss(ns, [1.0, 1.0], [0.0, 0.0], ss_factors, pad_factors)

        assert len(real) == len(unit) == 2
        for r, u, dx, x0 in zip(real, unit, dxs, x0s):
            np.testing.assert_allclose(
                np.asarray(r), x0 + dx * np.asarray(u), rtol=1e-9
            )


class TestLatentSpace:
    """Test latent space operations."""

    def test_signal_to_latent_1d(self):
        """Test latent representation extraction in 1D."""
        x = jnp.linspace(0, 10, 32)
        y = jnp.sin(2 * jnp.pi * x / 10)
        ns = [len(x)]
        dxs = [float(jnp.diff(x)[0])]

        # Use signal_to_latent_init + signal_to_latent
        idxs, pk = signal_to_latent_init(
            ns, dxs, pad_factors=[1.5], p0=1.0, k0s=[1.0], gammas=[2.0], cutoff=0.5
        )
        Y_latent = signal_to_latent(y, [1.5], idxs)

        # Latent should be smaller than original (compressed) with higher cutoff
        assert Y_latent.size <= y.size

    def test_latent_init_returns_correct_types(self):
        """Test that latent_to_signal_init returns correct data types."""
        x = jnp.linspace(0, 10, 32)
        ns = [len(x)]
        dxs = [float(jnp.diff(x)[0])]

        latent_pk, latent_ks, pads, idxs_pad_ss = latent_to_signal_init(
            ns,
            dxs,
            pad_factors=[1.5],
            ss_factors=[2],
            p0=1.0,
            k0s=[1.0],
            gammas=[2.0],
            cutoff=0.01,
        )

        # Check types
        assert isinstance(latent_pk, jnp.ndarray)
        assert isinstance(latent_ks, list)
        assert isinstance(pads, list)
        assert isinstance(idxs_pad_ss, list)
        assert all(isinstance(idx, slice) for idx in idxs_pad_ss)

    def test_latent_init_predict_roundtrip(self):
        """Test that latent_to_signal_init and latent_to_signal work together."""
        x = jnp.linspace(0, 10, 32)
        ns = [len(x)]
        dxs = [float(jnp.diff(x)[0])]

        latent_pk, latent_ks, pads, idxs_pad_ss = latent_to_signal_init(
            ns,
            dxs,
            pad_factors=[1.5],
            ss_factors=[2],
            p0=1.0,
            k0s=[1.0],
            gammas=[2.0],
            cutoff=0.01,
        )

        # Create dummy latent modes
        Y_latent = jnp.ones_like(latent_pk, dtype=complex)

        # Predict should return supersampled signal
        y_ss = latent_to_signal(Y_latent, pads, idxs_pad_ss)

        # Check that output is approximately the right size
        # (original size * supersample factor)
        assert y_ss.size >= len(x)

    def test_latent_operations_jit_compatible(self):
        """Test that latent operations are JIT-compatible."""
        x = jnp.linspace(0, 10, 32)
        ns = [len(x)]
        dxs = [float(jnp.diff(x)[0])]

        # latent_to_signal_init should work (it's called once for setup)
        latent_pk, latent_ks, pads, idxs_pad_ss = latent_to_signal_init(
            ns,
            dxs,
            pad_factors=[1.5],
            ss_factors=[2],
            p0=1.0,
            k0s=[1.0],
            gammas=[2.0],
            cutoff=0.01,
        )

        # latent_to_signal should be JIT-able
        @jax.jit
        def predict_jitted(Y_latent):
            return latent_to_signal(Y_latent, pads, idxs_pad_ss)

        Y_latent = jnp.ones_like(latent_pk, dtype=complex)
        y_ss = predict_jitted(Y_latent)
        assert y_ss.size > 0

    def test_signal_to_latent_jit_compatible(self):
        """Test that signal_to_latent_init + signal_to_latent is JIT-compatible."""
        x = jnp.linspace(0, 10, 32)
        y = jnp.sin(2 * jnp.pi * x / 10)
        ns = [len(x)]
        dxs = [float(jnp.diff(x)[0])]

        # Setup phase (not JIT-compatible, called once)
        idxs, pk = signal_to_latent_init(
            ns,
            dxs,
            pad_factors=[1.5],
            p0=1.0,
            k0s=[1.0],
            gammas=[2.0],
            cutoff=0.5,
        )

        # Apply phase (JIT-compatible)
        @jax.jit
        def compute_latent_jitted(y):
            return signal_to_latent(y, [1.5], idxs)

        Y_latent = compute_latent_jitted(y)
        assert Y_latent.size > 0

        # Verify it gives same result as non-JIT version
        Y_latent_ref = signal_to_latent(y, [1.5], idxs)
        assert jnp.allclose(Y_latent, Y_latent_ref)


class TestJAXCompatibility:
    """Test JAX compatibility of all major functions after refactoring."""

    def test_pk_cut_jit(self):
        """Test that pk_cut can be called (even though it's not fully JIT-compatible)."""
        # pk_cut returns Python slices which are inherently data-dependent
        # It's a setup function, not meant to be JIT-compiled
        # This test just verifies it works correctly outside JIT
        pk = jnp.exp(-jnp.linspace(-5, 5, 51) ** 2)
        idxs, pads = pk_cut(pk, 0.1)
        assert len(idxs) == 1
        assert len(pads) == 1
        assert isinstance(idxs[0], slice)


class TestNumericalAccuracy:
    """Test numerical accuracy of transformations."""

    def test_supersample_preserves_dc_component(self):
        """Test that supersampling preserves DC component."""
        y = jnp.ones(16)
        y_ss = supersample(y, [2])

        # DC component should be preserved
        Y_orig = jnp.fft.fftn(y, norm="forward")
        Y_ss = jnp.fft.fftn(y_ss, norm="forward")

        assert jnp.isclose(Y_orig.flatten()[0], Y_ss.flatten()[0])

    def test_fourier_cut_uncut_preserves_kept_modes(self):
        """Test that cutting and uncutting preserves kept Fourier modes."""
        # Create a signal with known Fourier content
        x = jnp.linspace(0, 2 * jnp.pi, 64)
        y = jnp.sin(x) + 0.5 * jnp.sin(3 * x)

        # Create power spectrum
        k = jnp.fft.fftshift(jnp.fft.fftfreq(len(x), x[1] - x[0]))
        pk = jnp.exp(-k ** 2 / 4)

        # Original Fourier transform
        Y_orig = jnp.fft.fftshift(jnp.fft.fftn(y, norm="forward"))

        # Cut and uncut
        Y_cut = fourier_cut(pk, 0.01, y)
        Y_uncut = fourier_uncut(pk, 0.01, Y_cut)

        # Get the indices that were kept
        idxs, _ = pk_cut(pk, 0.01)

        # Check that kept modes match
        assert jnp.allclose(Y_orig[idxs[0]], Y_cut, rtol=1e-5)

        # Check that uncut has same shape as original and kept modes match
        assert Y_uncut.shape == Y_orig.shape
        assert jnp.allclose(Y_uncut[idxs[0]], Y_orig[idxs[0]], rtol=1e-5)


class TestPkCutRefusesAnEmptySelection:
    """``pk_cut`` reports *why* nothing survived, because the advice differs.

    Before this it fell through to ``idx[i].min()`` and came back as "zero-size
    array to reduction operation min", which names neither the cutoff nor the
    cause. The four causes need four different answers, and telling someone to
    lower a cutoff that is already valid is worse than saying nothing.
    """

    def test_an_empty_grid_says_so_before_any_reduction(self):
        """Checked first: the per-axis max would otherwise raise its own
        zero-size error before the diagnosis could be reached."""
        with pytest.raises(ValueError, match="power spectrum is empty"):
            pk_cut(jnp.empty((0, 2)), 0.5)

    def test_a_cutoff_at_one_names_the_cutoff_and_says_to_lower_it(self):
        with pytest.raises(ValueError) as excinfo:
            pk_cut(jnp.ones((4, 4)), 1.0)

        assert "at or above 1" in str(excinfo.value)
        assert "Set it below 1" in str(excinfo.value)

    def test_a_spectrum_with_no_positive_power_does_not_blame_the_cutoff(self):
        """A valid cutoff and an underflowed spectrum: no cutoff can help, and
        the message has to say that rather than repeat "lower it"."""
        with pytest.raises(ValueError) as excinfo:
            pk_cut(jnp.zeros((2, 2)), 0.5)

        message = str(excinfo.value)
        assert "no positive power" in message
        assert "No cutoff can help" in message
        assert "Set it below 1" not in message

    def test_a_negative_spectrum_is_not_called_an_underflow(self):
        """`largest <= 0` covers both, but they are not the same thing."""
        with pytest.raises(ValueError, match="no positive power"):
            pk_cut(jnp.array([-2.0, -1.0]), 0.5)

    def test_a_non_finite_spectrum_is_named_as_such(self):
        """Even where the largest value is not the non-finite one."""
        with pytest.raises(ValueError) as excinfo:
            pk_cut(jnp.array([0.0, -jnp.inf]), 0.5)

        assert "not finite" in str(excinfo.value)
        assert "no positive power" not in str(excinfo.value)

    def test_a_non_finite_cutoff_is_named_as_such(self):
        with pytest.raises(ValueError, match="the cutoff is nan"):
            pk_cut(jnp.ones((2, 2)), float("nan"))
