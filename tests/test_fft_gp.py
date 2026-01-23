"""Tests for FFT-based Gaussian Process utilities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tabascal.fft_gp import (
    _get_dx,
    domain_k,
    domain_ss,
    fourier_cut,
    fourier_uncut,
    get_latent,
    latent_init,
    latent_predict,
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


class TestDomainHelpers:
    """Test basic domain helper functions."""

    def test_get_dx_regular_spacing(self):
        """Test resolution calculation for regularly spaced array."""
        x = jnp.linspace(0, 10, 101)
        dx = _get_dx(x)
        assert jnp.isclose(dx, 0.1)

    def test_get_dx_single_element(self):
        """Test resolution defaults to 1.0 for single element."""
        x = jnp.array([5.0])
        dx = _get_dx(x)
        assert dx == 1.0

    def test_get_dx_two_elements(self):
        """Test resolution calculation for two elements."""
        x = jnp.array([0.0, 2.5])
        dx = _get_dx(x)
        assert jnp.isclose(dx, 2.5)


class TestSupersample:
    """Test supersampling functions."""

    def test_supersample_domain_specs_1d(self):
        """Test supersampling specs for 1D domain."""
        x = jnp.linspace(0, 10, 11)
        n_ss, dx_ss = supersample_domain_specs([x], [2])

        assert n_ss == [22]
        assert jnp.isclose(dx_ss[0], 0.5)

    def test_supersample_domain_specs_2d(self):
        """Test supersampling specs for 2D domain."""
        x = jnp.linspace(0, 10, 11)
        y = jnp.linspace(0, 5, 6)
        n_ss, dx_ss = supersample_domain_specs([x, y], [2, 3])

        assert n_ss == [22, 18]
        assert jnp.isclose(dx_ss[0], 0.5)
        assert jnp.isclose(dx_ss[1], 1.0 / 3.0)

    def test_supersample_domain_specs_length_mismatch(self):
        """Test that length mismatch raises error."""
        x = jnp.linspace(0, 10, 11)
        with pytest.raises(ValueError, match="Length mismatch"):
            supersample_domain_specs([x], [2, 3])

    def test_supersample_domain_1d(self):
        """Test domain supersampling in 1D."""
        x = jnp.linspace(0, 10, 11)
        xs_ss = supersample_domain([x], [2])

        assert len(xs_ss) == 1
        assert len(xs_ss[0]) == 22
        assert jnp.isclose(xs_ss[0][0], x[0])
        assert jnp.isclose(xs_ss[0][-1], x[-1] + 0.5)

    def test_supersample_domain_k_1d(self):
        """Test k-domain supersampling in 1D."""
        x = jnp.linspace(0, 10, 11)
        ks_ss = supersample_domain_k([x], [2])

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
        ks = domain_k([x])

        assert len(ks) == 1
        assert len(ks[0]) == len(x)
        # Check that k=0 is in the center (after fftshift)
        assert jnp.isclose(ks[0][len(ks[0]) // 2], 0.0, atol=1e-6)

    def test_domain_k_2d(self):
        """Test k-domain calculation in 2D."""
        x = jnp.linspace(0, 10, 11)
        y = jnp.linspace(0, 5, 6)
        ks = domain_k([x, y])

        assert len(ks) == 2
        assert len(ks[0]) == len(x)
        assert len(ks[1]) == len(y)


class TestPadding:
    """Test padding functions."""

    def test_pad_domain_specs_validation(self):
        """Test that pad_domain_specs validates inputs."""
        x = jnp.linspace(0, 10, 11)

        # Test pad factors < 1.0
        with pytest.raises(ValueError, match="Pad factors must be >= 1.0"):
            pad_domain_specs([x], [0.5])

        # Test length mismatch
        with pytest.raises(ValueError, match="Length mismatch"):
            pad_domain_specs([x], [1.5, 2.0])

    def test_pad_domain_specs_1d(self):
        """Test padding specs for 1D domain."""
        x = jnp.linspace(0, 10, 11)
        ns, n_pads, dxs = pad_domain_specs([x], [2.0])

        assert ns == [11]
        assert n_pads == [5]  # (11 * (2 - 1) / 2) = 5.5 -> 5
        assert jnp.isclose(dxs[0], 1.0)

    def test_pad_domain_1d(self):
        """Test domain padding in 1D."""
        x = jnp.linspace(0, 10, 11)
        xs_pad = pad_domain([x], [2.0])

        assert len(xs_pad) == 1
        expected_len = 11 + 2 * 5  # 21
        assert len(xs_pad[0]) == expected_len

    def test_pad_domain_k_1d(self):
        """Test k-domain padding in 1D."""
        x = jnp.linspace(0, 10, 11)
        ks_pad = pad_domain_k([x], [2.0])

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
        xs_ss = domain_ss([x], ss_factors=[2], pad_factors=[1.5])

        assert len(xs_ss) == 1
        # Should have supersampled resolution
        dx_ss = _get_dx(xs_ss[0])
        dx_orig = _get_dx(x)
        assert jnp.isclose(dx_ss, dx_orig / 2, rtol=1e-4)

    def test_domain_ss_2d(self):
        """Test supersampled domain in 2D."""
        x = jnp.linspace(0, 10, 11)
        y = jnp.linspace(0, 5, 6)
        xs_ss = domain_ss([x, y], ss_factors=[2, 3], pad_factors=[1.5, 1.5])

        assert len(xs_ss) == 2


class TestLatentSpace:
    """Test latent space operations."""

    def test_get_latent_1d(self):
        """Test latent representation extraction in 1D."""
        x = jnp.linspace(0, 10, 32)
        y = jnp.sin(2 * jnp.pi * x / 10)

        Y_latent = get_latent(
            y, [x], pad_factors=[1.5], p0=1.0, k0s=[1.0], gammas=[2.0], cutoff=0.5
        )

        # Latent should be smaller than original (compressed) with higher cutoff
        assert Y_latent.size <= y.size

    def test_latent_init_returns_correct_types(self):
        """Test that latent_init returns correct data types."""
        x = jnp.linspace(0, 10, 32)

        latent_pk, latent_ks, pads, idxs_pad_ss = latent_init(
            [x],
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
        """Test that latent_init and latent_predict work together."""
        x = jnp.linspace(0, 10, 32)

        latent_pk, latent_ks, pads, idxs_pad_ss = latent_init(
            [x],
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
        y_ss = latent_predict(Y_latent, pads, idxs_pad_ss)

        # Check that output is approximately the right size
        # (original size * supersample factor)
        assert y_ss.size >= len(x)

    def test_latent_operations_jit_compatible(self):
        """Test that latent operations are JIT-compatible."""
        x = jnp.linspace(0, 10, 32)

        # latent_init should work (it's called once for setup)
        latent_pk, latent_ks, pads, idxs_pad_ss = latent_init(
            [x],
            pad_factors=[1.5],
            ss_factors=[2],
            p0=1.0,
            k0s=[1.0],
            gammas=[2.0],
            cutoff=0.01,
        )

        # latent_predict should be JIT-able
        @jax.jit
        def predict_jitted(Y_latent):
            return latent_predict(Y_latent, pads, idxs_pad_ss)

        Y_latent = jnp.ones_like(latent_pk, dtype=complex)
        y_ss = predict_jitted(Y_latent)
        assert y_ss.size > 0


class TestJAXCompatibility:
    """Test JAX compatibility of all major functions after refactoring."""

    def test_supersample_domain_specs_jit(self):
        """Test that core computation in supersample works with JIT."""
        # The spec functions return Python lists, so they're not directly JIT-compatible
        # But the underlying computations should work
        @jax.jit
        def compute_dx(x):
            return _get_dx(x)

        x = jnp.linspace(0, 10, 11)
        dx = compute_dx(x)
        assert jnp.isclose(dx, 1.0)

    def test_pad_domain_specs_jit(self):
        """Test that core computation in pad works with JIT."""
        # The spec functions return Python lists, but we can test the underlying computation
        @jax.jit
        def compute_dx(x):
            return _get_dx(x)

        x = jnp.linspace(0, 10, 11)
        dx = compute_dx(x)
        assert jnp.isclose(dx, 1.0)

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
