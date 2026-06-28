"""Tests for the analytic fringe-winding factors in tabascal.interferometry.

F1 (linear envelope, linear phase) and F2_fresnel_jax (linear envelope, quadratic
phase) are the closed forms that replace the oversample-and-average for one
sub-window of the analytic RFI-visibility path. F2q_fresnel_jax is the optional
quadratic-envelope extension.

Each factor is checked against a high-N composite-Simpson oracle of
V_bar = (1/dt) int_{-dt/2}^{+dt/2} w(s) exp(i phi(s)) ds in the regime where it is
exact, plus small-argument stability and jit/grad cleanliness.

These exercise the complex erf at large |z|, so the module requires double precision.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tabascal.interferometry import F1, F2_fresnel_jax, F2q_fresnel_jax, _S0_MAX

pytestmark = pytest.mark.requires_double

DT = 2.0  # window length (seconds)


def _tol_for(f, fdot):
    """Tight Fresnel tolerance where the exact form is exercised (|s0| <= _S0_MAX);
    envelope-floor tolerance in the large-s0 corner where the factor falls back to the
    linear-phase form (accurate there because the visibility is heavily fringe-washed)."""
    s0 = abs(f / fdot) if fdot != 0 else np.inf
    return 1e-4 if s0 <= _S0_MAX else 3e-3


def oracle(w_func, phi_func, dt=DT, N=200001):
    """High-N composite-Simpson estimate of (1/dt) int w(s) exp(i phi(s)) ds."""
    s = np.linspace(-dt / 2, dt / 2, N)
    integrand = w_func(s) * np.exp(1j * phi_func(s))
    h = (s[-1] - s[0]) / (N - 1)
    wts = np.ones(N)
    wts[1:-1:2] = 4.0
    wts[2:-1:2] = 2.0
    return (h / 3.0) * np.sum(wts * integrand) / dt


def lin_env(w0, wp):
    return lambda s: w0 + wp * s


def quad_env(w0, wp, wpp):
    return lambda s: w0 + wp * s + 0.5 * wpp * s**2


def quad_phase(phi0, f, fdot):
    return lambda s: phi0 + 2 * np.pi * f * s + np.pi * fdot * s**2


# --------------------------------------------------------------------------- #
# F1: exact for linear envelope + linear phase
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fdt", [0.0, 0.01, 0.5, 5.0, 50.0])
def test_F1_matches_oracle_linear_phase(fdt):
    phi0, w0, wp = 0.3, 1.0 - 0.2j, 0.4 + 0.1j
    f = fdt / DT
    ref = oracle(lin_env(w0, wp), quad_phase(phi0, f, 0.0))
    got = complex(F1(phi0, w0, wp, f, DT))
    assert abs(got - ref) < 1e-10, f"f*dt={fdt}: |err|={abs(got-ref):.2e}"


# --------------------------------------------------------------------------- #
# F2_fresnel_jax: exact for linear envelope + quadratic phase, any fdot*dt^2
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fdt", [0.0, 0.1, 1.0, 10.0, 50.0])
@pytest.mark.parametrize("fddt2", [1e-3, 0.05, 0.5, 2.0, 10.0])
def test_F2_fresnel_matches_oracle_quadratic_phase(fdt, fddt2):
    phi0, w0, wp = -0.5, 1.0 + 0.3j, 0.25 - 0.15j
    f = fdt / DT
    fdot = fddt2 / DT**2
    ref = oracle(lin_env(w0, wp), quad_phase(phi0, f, fdot))
    got = complex(F2_fresnel_jax(phi0, w0, wp, f, fdot, DT))
    # oracle-quadrature limited at the highest f*dt; envelope/phase both exact here.
    tol = _tol_for(f, fdot)
    assert abs(got - ref) < tol, f"f*dt={fdt}, fdot*dt^2={fddt2}: |err|={abs(got-ref):.2e}"


def test_F2_fresnel_reduces_to_F1_as_fdot_to_zero():
    phi0, w0, wp, f = 0.7, 0.9 - 0.1j, 0.3 + 0.2j, 12.0 / DT
    got = complex(F2_fresnel_jax(phi0, w0, wp, f, 0.0, DT))
    ref = complex(F1(phi0, w0, wp, f, DT))
    assert abs(got - ref) < 1e-12


@pytest.mark.parametrize("fdt", [3.0, 50.0, 800.0])
@pytest.mark.parametrize("fddt2", [1e-7, 1e-4, 1e-2])
def test_F2_fresnel_large_s0_matches_oracle(fdt, fddt2):
    """Fast-fringe / slow-chirp corner (large s0 = f/fdot, e.g. fringe-rate turning
    points): the Fresnel form is unstable there, so the factor must fall back to F1 and
    still track the oracle below the envelope floor (~7e-3 at l/dt~12)."""
    phi0, w0, wp = 0.2, 1.0 + 0.1j, 0.3 - 0.2j
    f = fdt / DT
    fdot = fddt2 / DT**2
    ref = oracle(lin_env(w0, wp), quad_phase(phi0, f, fdot))
    got = complex(F2_fresnel_jax(phi0, w0, wp, f, fdot, DT))
    assert np.isfinite(got.real) and np.isfinite(got.imag)
    assert abs(got - ref) < 3e-3, f"f*dt={fdt}, fdot*dt^2={fddt2}: |err|={abs(got-ref):.2e}"


def test_F2_fresnel_large_s0_grad_finite():
    """Gradients stay NaN-free in the large-s0 fallback corner (double-where guard)."""

    def loss(w0r):
        # f*dt=800, fddt2=1e-6 -> s0 ~ 8e5: deep in the fallback corner.
        v = F2_fresnel_jax(0.1, w0r + 0.2j, 0.3 + 0.1j, 400.0, 1e-6 / DT**2, DT)
        return jnp.abs(v) ** 2

    g = float(jax.grad(loss)(1.0))
    assert np.isfinite(g)


# --------------------------------------------------------------------------- #
# F2q_fresnel_jax: exact for quadratic envelope + quadratic phase
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fdt", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("fddt2", [0.05, 0.5, 2.0])
def test_F2q_matches_oracle_quadratic_env_phase(fdt, fddt2):
    phi0 = 0.4
    w0, wp, wpp = 1.0 - 0.2j, 0.3 + 0.1j, 0.5 - 0.4j  # w0 is the CENTRE value
    f = fdt / DT
    fdot = fddt2 / DT**2
    ref = oracle(quad_env(w0, wp, wpp), quad_phase(phi0, f, fdot))
    got = complex(F2q_fresnel_jax(phi0, w0, wp, wpp, f, fdot, DT))
    tol = _tol_for(f, fdot)
    assert abs(got - ref) < tol, f"f*dt={fdt}, fdot*dt^2={fddt2}: |err|={abs(got-ref):.2e}"


# --------------------------------------------------------------------------- #
# jit / grad cleanliness (incl. the f->0, fdot->0 Taylor/fallback branches)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("f,fdot", [(0.0, 0.0), (1e-6, 1e-13), (10.0, 1.0)])
def test_F2_fresnel_jit_grad_finite(f, fdot):
    def loss(w0r):
        v = F2_fresnel_jax(0.1, w0r + 0.2j, 0.3 + 0.1j, f, fdot, DT)
        return jnp.abs(v) ** 2

    val = float(jax.jit(loss)(1.0))
    g = float(jax.grad(loss)(1.0))
    assert np.isfinite(val) and np.isfinite(g)
