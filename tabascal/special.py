"""Special functions for the analytic RFI-visibility factor.

Currently provides a single-branch complex error function :func:`cerf`, used by the
exact-Fresnel fringe-winding factor (:func:`tabascal.interferometry.F2_fresnel_jax`).
It is differentiable (custom JVP) and JIT-able, as required by the analytic visibility
path that propagates gradients to the GP inducing points.

The implementation uses Weideman's (1994) Faddeeva approximation: a single
constant-coefficient polynomial (Horner) evaluation valid across the upper half-plane,
rather than a series/asymptotic branch split (which would force every element to pay
both branches under ``jnp.where``). The coefficients are precomputed once in numpy and
baked in as constants; the JAX forward is a single Horner plus a few elementwise ops.

Faddeeva relation::

    w(Z) = exp(-Z^2) erfc(-i Z),   valid (Weideman) for Im(Z) >= 0
    =>  erfc(u) = exp(-u^2) w(i u), valid for Re(u) >= 0

We reduce a general argument to ``Re(z) >= 0`` using the oddness of ``erf``, compute
``erfc`` via ``w``, and return ``erf = 1 - erfc``.

Accuracy (float64, on the production argument ray |z| <= 500, arg +/-pi/4) is ~8e-15;
the catastrophic-cancellation path (``erf(z1) - erf(z2)`` at large |z|) is ~1e-11.

Custom JVP: ``d/dz erf(z) = (2/sqrt(pi)) exp(-z^2)`` (exact, entire).

Reference
---------
J. A. C. Weideman, "Computation of the complex error function",
SIAM J. Numer. Anal. 31 (1994) 1497-1518.
"""

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["cerf"]

# Number of Weideman terms. N=44 gives ~1e-10 on the ray in the worst case and
# ~8e-15 on the production argument range; raise for more accuracy.
_N_WEIDEMAN = 44


def _weideman_coeffs(N: int):
    """Precompute the Weideman (1994) Faddeeva coefficients.

    Returns the scale ``L`` and the ``N`` polynomial coefficients ``a`` ordered
    highest-power-first (for :func:`jax.numpy.polyval` / Horner evaluation).
    """
    M = 2 * N
    M2 = 2 * M
    k = np.arange(-M + 1, M)
    L = np.sqrt(N / np.sqrt(2.0))
    theta = k * np.pi / M
    t = L * np.tan(theta / 2.0)
    f = np.exp(-(t**2)) * (L**2 + t**2)
    f = np.append(0.0, f)
    a = np.real(np.fft.fft(np.fft.fftshift(f))) / M2
    a = np.flipud(a[1 : N + 1])  # highest-order first, for polyval/Horner
    return L, a


_L_np, _A_np = _weideman_coeffs(_N_WEIDEMAN)
_L = jnp.asarray(_L_np)
_A = jnp.asarray(_A_np)
_INV_SQRTPI = 1.0 / np.sqrt(np.pi)
_2_SQRTPI = 2.0 / np.sqrt(np.pi)


def _wofz(Z: jax.Array) -> jax.Array:
    """Weideman approximation of the Faddeeva function w(Z), valid for Im(Z) >= 0."""
    denom = _L - 1j * Z
    Zt = (_L + 1j * Z) / denom
    p = jnp.polyval(_A, Zt)  # single Horner over N constant coefficients
    return 2.0 * p / denom**2 + _INV_SQRTPI / denom


@jax.custom_jvp
def cerf(z: jax.Array) -> jax.Array:
    """Complex error function ``erf(z)`` for complex (or real) ``z``.

    Single-branch Weideman/Faddeeva evaluation, differentiable and JIT-able.

    Parameters
    ----------
    z : Array
        Complex (or real) argument(s).

    Returns
    -------
    Array
        ``erf(z)`` as a complex array.
    """
    z = z + 0j
    sgn = jnp.where(jnp.real(z) < 0, -1.0, 1.0)
    zr = sgn * z  # Re(zr) >= 0
    # erfc(zr) = exp(-zr^2) w(i zr); Im(i zr) = Re(zr) >= 0  -> Weideman valid
    erfc = jnp.exp(-(zr**2)) * _wofz(1j * zr)
    return sgn * (1.0 - erfc)


@cerf.defjvp
def _cerf_jvp(primals, tangents):
    (z,), (dz,) = primals, tangents
    return cerf(z), _2_SQRTPI * jnp.exp(-((z + 0j) ** 2)) * dz
