"""Tests for tabascal.special.cerf (single-branch Weideman complex erf).

The complex error function underlies the exact-Fresnel fringe-winding factor used
by the analytic RFI-visibility path. These tests pin its accuracy against
``scipy.special.erf`` on the production argument range (a ray |z| <= 500 at
arg +/-pi/4), the catastrophic-cancellation difference path, a bounded general
region, and the custom JVP (``d/dz erf = (2/sqrt(pi)) exp(-z^2)``).

cerf requires double precision (the large-|z| arguments overflow otherwise), so
the whole module is marked ``requires_double``.
"""

import jax
import numpy as np
import pytest
from scipy.special import erf as scipy_erf

from tabascal.special import cerf

pytestmark = pytest.mark.requires_double


def test_on_ray_accuracy():
    """|z| <= 500 on the +/-pi/4 ray (the production argument range)."""
    mags = np.logspace(np.log10(0.01), np.log10(500), 80)
    z = np.concatenate([mags * np.exp(1j * a) for a in (np.pi / 4, -np.pi / 4)])
    err = np.abs(np.asarray(cerf(z)) - scipy_erf(z))
    assert err.max() < 1e-12, f"on-ray max abs error {err.max():.2e}"


def test_general_region_accuracy():
    """Bounded general region where |erf| is not enormous."""
    re = np.linspace(-6, 6, 25)
    im = np.linspace(-6, 6, 25)
    z = (re[:, None] + 1j * im[None, :]).ravel()
    ref = scipy_erf(z)
    mask = np.abs(ref) < 10
    err = np.abs(np.asarray(cerf(z)) - ref)[mask]
    assert err.max() < 1e-10, f"general-region max abs error {err.max():.2e}"


def test_cancellation_difference():
    """erf(z1) - erf(z2) at large |z| (the heavily-washed sub-window difference)."""
    z1 = 340 * np.exp(-1j * np.pi / 4)
    z2 = 342 * np.exp(-1j * np.pi / 4)
    got = complex(cerf(z1) - cerf(z2))
    ref = complex(scipy_erf(z1) - scipy_erf(z2))
    rel = abs(got - ref) / abs(ref)
    assert rel < 1e-9, f"cancellation relerr {rel:.2e}"


def test_oddness():
    """erf is odd: erf(-z) = -erf(z)."""
    z = np.array([0.7 - 0.4j, 2.0 + 1.0j, -1.5 + 0.3j, 5.0 - 5.0j])
    np.testing.assert_allclose(np.asarray(cerf(-z)), -np.asarray(cerf(z)), atol=1e-12)


def test_real_axis_matches_scipy():
    """On the real axis cerf reduces to the real erf."""
    x = np.linspace(-4, 4, 41)
    np.testing.assert_allclose(np.asarray(cerf(x)).real, scipy_erf(x), atol=1e-12)
    np.testing.assert_allclose(np.asarray(cerf(x)).imag, 0.0, atol=1e-12)


def test_custom_jvp_matches_analytic():
    """d/dz erf = (2/sqrt(pi)) exp(-z^2), exact and entire."""
    z0 = 0.7 - 0.4j
    _, g = jax.jvp(cerf, (z0,), (1.0 + 0j,))
    expected = 2 / np.sqrt(np.pi) * np.exp(-(z0**2))
    rel = abs(complex(g) - expected) / abs(expected)
    assert rel < 1e-12, f"jvp relerr {rel:.2e}"


def test_jit_and_grad_finite():
    """cerf is JIT-able and grads flow finitely (e.g. through |erf|^2)."""
    f = jax.jit(lambda zr, zi: jnp_abs2(cerf(zr + 1j * zi)))
    val = float(f(1.2, -0.6))
    assert np.isfinite(val)
    g = jax.grad(lambda zr: jnp_abs2(cerf(zr + 1j * 0.3)))(0.9)
    assert np.isfinite(float(g))


def jnp_abs2(z):
    import jax.numpy as jnp

    return jnp.abs(z) ** 2
