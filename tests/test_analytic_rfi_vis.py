"""Analytic moments and cell integrals against independently sampled visibilities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from tabascal.coarse_rfi_vis import (
    analytic_rfi_vis, coarse_rfi_vis, linear_phase_moments, quadratic_phase_moments,
)
from tabascal.poly_interp import fine_offsets, interp_tables, monomial_tables


def _double():
    return jax.config.x64_enabled


def _moment_reference(a, b, degree):
    return np.array([
        quad(lambda x: x**m * np.cos(a*x+b*x*x)/2, -1, 1, epsabs=2e-13, limit=2000)[0]
        + 1j*quad(lambda x: x**m * np.sin(a*x+b*x*x)/2, -1, 1, epsabs=2e-13, limit=2000)[0]
        for m in range(degree+1)
    ])


@pytest.mark.parametrize("a", [0., 1e-7, .71, 4.3, 15.2, 33.9, 40.2, 3142.7])
def test_linear_moments_in_both_recurrence_directions(a):
    got = linear_phase_moments(jnp.asarray(a), 34)
    expected = _moment_reference(a, 0, 34)
    np.testing.assert_allclose(got, expected, atol=2e-7 if not _double() else 1e-12, rtol=0)


@pytest.mark.parametrize("a,b", [
    (0., 0.), (.7, 1e-8), (3.1, .001), (4.7, .9), (3150.1, .8),
    (0., 1.1), (2.1999, 1.1), (2.2001, 1.1), (2.7499, 1.1), (2.7501, 1.1),
    (-2.1999, -1.1), (-2.2001, -1.1), (0., 3.), (5.9999, 3.), (6.0001, 3.),
])
def test_quadratic_moments_near_and_outside_the_stationary_point(a, b):
    got = quadratic_phase_moments(jnp.asarray(a), jnp.asarray(b), 4, terms=24)
    expected = _moment_reference(a, b, 4)
    np.testing.assert_allclose(got, expected, atol=2e-6 if not _double() else 2e-12, rtol=0)


def cell_inputs(turns=10.37, curvature=.11, cubic=0., n_int=65537, half_width=1):
    """Noninteger winding, frequency offsets, shifted edge stencils and a signal.

    Curvature and cubic here are turns at the cell edge, a conservative reading
    of the measured coefficients. Linear winding is turns across the full cell.
    """
    n_time = 2*half_width+1
    int_time = 2.
    freqs = np.array([1400., 1401.])  # MHz, as the operator receives them
    dnu = np.array([-.2, 0., .2])
    wf, sf = interp_tables(2, 1, dnu)
    dt = fine_offsets(n_int, int_time)
    wt, st = interp_tables(n_time, half_width, dt/int_time)
    gt, _ = monomial_tables(n_time, half_width)
    t = 2*(np.arange(n_time)-n_time//2)
    amp = np.ones((1, 2, 2, n_time), dtype=complex)
    amp[:, 0] = (1 + .12*t + (.025+.015j)*t*t)[None, :]
    amp[:, 1] = (.9 + .1j - .08j*t + .02*t*t)[None, :]
    if half_width > 1:
        amp[:, 0] += (.004*t**3 + .001j*t**4)[None, :]
        amp[:, 1] += ((.001+.001j)*t**4)[None, :]
    if half_width == 0:
        amp[:] = 1
    phase = np.zeros_like(amp.real)
    phase[:, 0] = .43
    delay = np.zeros((1, 2, n_time, 4))
    delay[:, 0, :, 0] = .12
    delay[:, 0, :, 1] = turns/(2*freqs[0])
    delay[:, 0, :, 2] = 2*curvature/freqs[0]
    delay[:, 0, :, 3] = 6*cubic/freqs[0]
    common = [jnp.asarray(x) for x in (amp, phase, delay, wf, sf)]
    tail = [jnp.asarray(x) for x in (st, dnu, freqs, np.array([0]), np.array([1]))]
    dense = common + [jnp.asarray(wt), tail[0], tail[1], jnp.asarray(dt), *tail[2:]]
    analytic = common + [jnp.asarray(gt), tail[0], tail[1], jnp.asarray(int_time), *tail[2:]]
    return analytic, dense


@pytest.mark.parametrize("turns", [1.13, 10.37, 100.23, 1000.41])
@pytest.mark.parametrize("curvature", [.001, .1, 1., 3.])
def test_cell_integral_agrees_with_dense_reference(turns, curvature):
    analytic, dense = cell_inputs(turns, curvature)
    expected = coarse_rfi_vis(*dense)
    got = jax.jit(analytic_rfi_vis)(*analytic)
    # Dense midpoint quadrature still has O(N^-2) error. The single-precision
    # reference also forms thousands of radians before exponentiation.
    np.testing.assert_allclose(got, expected, atol=2e-5 if not _double() else 2e-7, rtol=0)


def test_polynomial_amplitude_is_exact_and_constant_amplitude_is_not():
    analytic, dense = cell_inputs(37.21, .2)
    got = analytic_rfi_vis(*analytic)
    reference = coarse_rfi_vis(*dense)
    # h=0 holds each cell's centre value; all other inputs and phase are equal.
    constant = list(analytic)
    constant[5], constant[6] = (jnp.asarray(x) for x in monomial_tables(3, 0))
    wrong = analytic_rfi_vis(*constant)
    error = float(jnp.max(jnp.abs(got-reference)))
    constant_error = float(jnp.max(jnp.abs(wrong-reference)))
    assert error < (1e-6 if not _double() else 1e-8)
    assert constant_error > 1e-3
    assert constant_error > 1000*error


def test_amplitude_gradients_match_dense_jvp_and_vjp():
    analytic, dense = cell_inputs(23.41, .31, cubic=.027, n_int=8193)
    amp = analytic[0]
    call = jax.jit(lambda a: analytic_rfi_vis(a, *analytic[1:]))
    ref = lambda a: coarse_rfi_vis(a, *dense[1:])
    rng = np.random.default_rng(5)
    tangent = jnp.asarray(rng.normal(size=amp.shape)+1j*rng.normal(size=amp.shape), amp.dtype)
    tol = 2e-5 if not _double() else 3e-6
    np.testing.assert_allclose(jax.jvp(call, (amp,), (tangent,))[1], jax.jvp(ref, (amp,), (tangent,))[1], atol=tol, rtol=0)
    vis, pb = jax.vjp(call, amp)
    _, ref_pb = jax.vjp(ref, amp)
    cot = jnp.asarray(rng.normal(size=vis.shape)+1j*rng.normal(size=vis.shape), vis.dtype)
    np.testing.assert_allclose(pb(cot)[0], ref_pb(cot)[0], atol=tol, rtol=0)


@pytest.mark.parametrize("turns,curvature,cubic", [
    (2.3, .003, 1.7e-4), (13.8, .019, 8.8e-4),
    (78.8, .11, .0049), (618.37, 1.68, .027),
])
def test_cubic_omission_on_the_measured_geometry(turns, curvature, cubic):
    analytic, dense = cell_inputs(turns, curvature, cubic)
    got = analytic_rfi_vis(*analytic, cubic_terms=0)
    expected = coarse_rfi_vis(*dense)
    error = float(jnp.max(jnp.abs(got-expected)))
    # Quote error against instantaneous amplitude, not the fringe-suppressed
    # visibility. The noise is roughly 1/54 of that amplitude on this dataset.
    amplitude = float(jnp.max(jnp.abs(analytic[0][:, 0]*analytic[0][:, 1].conj())))
    assert error/amplitude < (1/54)/50, (turns, curvature, cubic, error/amplitude)
    corrected = analytic_rfi_vis(*analytic)
    assert float(jnp.max(jnp.abs(corrected-expected))) < error/20


def test_polynomial_stencil_is_near_exact_against_an_independent_integral():
    analytic, _ = cell_inputs(37.21, .2, n_int=1)
    expected = np.zeros((1, 2, 3), dtype=complex)
    for f, centre_freq in enumerate(np.asarray(analytic[9], dtype=np.float64)):
        for cell, t in enumerate([-2, 0, 2]):
            for dnu in np.asarray(analytic[7], dtype=np.float64):
                def integrand(x):
                    y = t+x
                    product = (1+.12*y+(.025+.015j)*y*y) * np.conj(.9+.1j-.08j*y+.02*y*y)
                    phase = .43 + 2*np.pi*dnu*.12 + (centre_freq+dnu)/1400*(np.pi*37.21*x + 2*np.pi*.2*x*x)
                    return product*np.exp(1j*phase)/2
                expected[0, f, cell] += (quad(lambda x: integrand(x).real, -1, 1, epsabs=1e-13)[0] + 1j*quad(lambda x: integrand(x).imag, -1, 1, epsabs=1e-13)[0])/3
    np.testing.assert_allclose(analytic_rfi_vis(*analytic), expected, atol=2e-6 if not _double() else 5e-13, rtol=0)


@pytest.mark.parametrize("half_width", [0, 1, 2])
def test_monomial_tables_reproduce_the_sampled_basis(half_width):
    n_time = 2*half_width+3
    g, starts = monomial_tables(n_time, half_width)
    x = np.array([-.91, -.4, .03, .7, .93])
    w, reference_starts = interp_tables(n_time, half_width, x/2)
    np.testing.assert_array_equal(starts, reference_starts)
    reconstructed = np.einsum('tlm,mv->tlv', g, x[None, :]**np.arange(g.shape[-1])[:, None])
    np.testing.assert_allclose(reconstructed, w, atol=2e-14, rtol=0)


@pytest.mark.parametrize("half_width", [0, 2])
def test_short_and_wide_stencils(half_width):
    analytic, dense = cell_inputs(1.13, 3., n_int=8193, half_width=half_width)
    np.testing.assert_allclose(analytic_rfi_vis(*analytic), coarse_rfi_vis(*dense), atol=2e-5 if not _double() else 3e-6, rtol=0)


def test_multiple_sources_and_reversed_baselines_keep_their_axes():
    analytic, dense = cell_inputs(14.37, .23, n_int=32769)
    for args in (analytic, dense):
        args[0] = jnp.concatenate([args[0], (.7+.2j)*args[0]], axis=0)
        for i in (1, 2):
            args[i] = jnp.concatenate([args[i], args[i]], axis=0)
        args[-2], args[-1] = jnp.array([0, 1, 0]), jnp.array([1, 0, 1])
    got, expected = analytic_rfi_vis(*analytic), coarse_rfi_vis(*dense)
    assert got.shape == (3, 2, 3)
    np.testing.assert_allclose(got, expected, atol=3e-6 if not _double() else 2e-7, rtol=0)
    np.testing.assert_allclose(got[0], got[1].conj(), atol=2e-7 if not _double() else 1e-12, rtol=0)


@pytest.mark.parametrize('turns,curvature', [(1.13, .001), (1.13, 1.68), (1.13, 3.), (10.37, 3.)])
def test_cubic_correction_is_needed_when_a_fast_baseline_has_a_slow_cell(turns, curvature):
    analytic, dense = cell_inputs(turns, curvature, cubic=.027)
    expected = coarse_rfi_vis(*dense)
    quadratic = analytic_rfi_vis(*analytic, cubic_terms=0)
    corrected = analytic_rfi_vis(*analytic)
    amplitude = float(jnp.max(jnp.abs(analytic[0][:, 0]*analytic[0][:, 1].conj())))
    # Membership uses the baseline's maximum requirement across cells and
    # sources. A cell in that fast group can still be near stationary, where
    # fringe suppression no longer protects us against dropping cubic phase.
    assert float(jnp.max(jnp.abs(quadratic-expected)))/amplitude > (1/54)/10
    assert float(jnp.max(jnp.abs(corrected-expected)))/amplitude < (2e-6 if not _double() else 1e-7)
