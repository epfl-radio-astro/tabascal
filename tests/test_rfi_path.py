"""tabascal.rfi_path: the fine-grid RFI phase from its data-grid value and path derivatives.

Checked against the definition -- ``get_rfi_phase_numpy`` on a fine grid -- on
a synthetic geometry whose times are exact, so that what is measured is the
expansion and not the jitter of a float64 Julian date: a circular orbit over a
rotating array, at a rate and a baseline that turn the fringe several times
within an integration.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from tabascal import rfi_path as rp
from tabascal.gp_interp import fine_offsets
from tabascal.interferometry import get_rfi_phase_numpy

from .components.conftest import active_precision


# ---------------------------------------------------------------------------
# A synthetic geometry with exact times
# ---------------------------------------------------------------------------

_JD0 = 2460000.5  # a real epoch, for the weight tests that want a jittered grid
_OMEGA_EARTH = 7.292e-5  # rad/s


def geometry(n_time=6, n_int_time=16, int_time=8.0, n_ant=5, n_rfi=2, baseline=8000.0):
    """Sources on circular low orbits over antennas turning with the Earth.

    Returns ``(times_jd_fine, rfi_xyz, ants_uvw, ants_xyz)`` on the fine grid
    laid out as ``TabConfig`` lays it out. The w term is a smooth sinusoid so
    that it, too, is differentiated.
    """
    times = np.arange(n_time) * int_time
    times_fine = (times[:, None] + fine_offsets(n_int_time, int_time)[None, :]).ravel()
    # Dates counted from an origin near zero rather than from a real epoch: a
    # float64 Julian date near 2.45e6 resolves ~20 us, and at tens of metres per
    # second of differential path that jitter is a degree of phase in the
    # reference itself. Here the dates are exact, so what is measured is the
    # expansion alone. (The components take the real dates and fit at the times
    # the positions were actually propagated at; see path_derivative_weights.)
    times_jd_fine = times_fine / 86400.0
    t = times_jd_fine * 86400.0

    r_orbit, r_earth = 6.8e6, 6.371e6
    omega = np.sqrt(3.986e14 / r_orbit**3)
    rfi_xyz = np.stack(
        [
            r_orbit
            * np.stack(
                [
                    np.cos(omega * t + 0.3 * k),
                    np.sin(omega * t + 0.3 * k) * np.cos(0.4 + 0.1 * k),
                    np.sin(omega * t + 0.3 * k) * np.sin(0.4 + 0.1 * k),
                ],
                axis=-1,
            )
            for k in range(n_rfi)
        ]
    )
    # Antennas along a line on the surface, turning with the Earth; the source
    # passes over them shortly after the start.
    lon0 = 0.5 * np.pi / 4 + 0.25 * np.pi + np.arange(n_ant) * baseline / (n_ant - 1) / r_earth
    lat = 0.35 + 0.02 * (np.arange(n_ant) % 2)
    lon = lon0[:, None] + _OMEGA_EARTH * t[None, :]
    ants_xyz = r_earth * np.stack(
        [np.cos(lat)[:, None] * np.cos(lon), np.cos(lat)[:, None] * np.sin(lon), np.sin(lat)[:, None] * np.ones_like(lon)],
        axis=-1,
    )
    ants_uvw = np.zeros_like(ants_xyz)
    ants_uvw[..., 2] = 50.0 * np.sin(2e-3 * t[None, :] + np.arange(n_ant)[:, None])
    return times_jd_fine, rfi_xyz, ants_uvw, ants_xyz


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def differential(phase):
    """Antenna-pair phase differences, which is what a visibility sees."""
    return phase[:, :, None] - phase[:, None, :]


# ---------------------------------------------------------------------------
# Derivative weights
# ---------------------------------------------------------------------------


def test_window_size_is_odd_and_exceeds_the_order():
    assert [rp.window_size(k) for k in range(6)] == [3, 3, 5, 5, 7, 7]
    with pytest.raises(ValueError):
        rp.window_size(-1)


@pytest.mark.parametrize("cell", [0, 1, 3, 5])
def test_the_weights_differentiate_a_polynomial_exactly_at_the_actual_times(cell):
    """A quartic sampled at jittered times is differentiated to round-off.

    Interior and edge cells alike -- the edge windows are one-sided. The times
    carry a jitter of the size a float64 Julian date has, and the polynomial is
    evaluated at the jittered times, which is what the fit has to use.
    """
    n_time, n_int_time, int_time, order = 6, 4, 8.0, 3
    times = (np.arange(n_time)[:, None] * int_time + fine_offsets(n_int_time, int_time)[None, :]).ravel()
    rng = np.random.RandomState(0)
    jd = _JD0 + times / 86400.0 + rng.uniform(-2e-10, 2e-10, times.shape)
    seconds = (jd - jd[0]) * 86400.0

    windows, centres, weights = rp.path_derivative_weights(jd, n_time, n_int_time, order)
    assert windows.shape == (n_time, 5) and weights.shape == (n_time, order + 1, 5)

    coeffs = np.array([1.0e3, 7.0, -0.5, 0.03, 0.002])  # metres, in powers of seconds
    poly = np.polynomial.Polynomial(coeffs)
    centre_time = seconds[cell * n_int_time + n_int_time // 2]
    assert windows[cell, centres[cell]] == cell * n_int_time + n_int_time // 2
    got = weights[cell] @ poly(seconds[windows[cell]])
    want = np.array([poly.deriv(k)(centre_time) for k in range(order + 1)])
    assert np.allclose(got, want, rtol=1e-7, atol=1e-9)


def test_a_short_fine_grid_caps_the_window_and_the_order():
    jd = _JD0 + np.arange(3) * 2.0 / 86400.0
    windows, centres, weights = rp.path_derivative_weights(jd, 3, 1, 3)
    assert windows.shape == (3, 3) and weights.shape == (3, 3, 3)
    assert centres.tolist() == [0, 1, 2]


def test_the_dates_have_to_be_the_fine_grid():
    with pytest.raises(ValueError, match="times_jd_fine"):
        rp.path_derivative_weights(np.arange(5.0), 3, 2, 1)


# ---------------------------------------------------------------------------
# The data-grid phase and path
# ---------------------------------------------------------------------------


def test_the_centre_phase_is_the_fine_phase_at_the_cell_centres():
    n_time, n_int_time = 6, 16
    jd, rfi_xyz, ants_uvw, ants_xyz = geometry(n_time=n_time, n_int_time=n_int_time)
    freqs = np.array([1.4e9, 1.401e9])
    windows, centres, weights = rp.path_derivative_weights(jd, n_time, n_int_time, 3)

    phase_c, path = rp.coarse_phase_and_path(rfi_xyz, ants_uvw, ants_xyz, freqs, windows, centres, weights)

    fine = get_rfi_phase_numpy(rfi_xyz, ants_uvw, ants_xyz, freqs)
    assert phase_c.shape == (2, 5, 2, n_time) and path.shape == (2, 5, n_time, 4)
    assert np.allclose(phase_c, fine[..., n_int_time // 2 :: n_int_time], atol=1e-9)
    # Differential to the array mean, at every order.
    assert np.allclose(path.mean(axis=1), 0.0, atol=1e-6 * np.abs(path).max())


@pytest.mark.requires_double
def test_the_jax_and_numpy_paths_agree():
    """One function under two array libraries, so agreement is the point of the design."""
    n_time, n_int_time = 4, 8
    jd, rfi_xyz, ants_uvw, ants_xyz = geometry(n_time=n_time, n_int_time=n_int_time, n_ant=3, n_rfi=1)
    freqs = np.array([1.4e9])
    windows, centres, weights = rp.path_derivative_weights(jd, n_time, n_int_time, 2)

    phase_np, path_np = rp.coarse_phase_and_path(rfi_xyz, ants_uvw, ants_xyz, freqs, windows, centres, weights)
    phase_jx, path_jx = rp.coarse_phase_and_path(
        jnp.asarray(rfi_xyz), jnp.asarray(ants_uvw), jnp.asarray(ants_xyz), jnp.asarray(freqs),
        windows, centres, weights, xp=jnp,
    )

    assert np.allclose(np.asarray(phase_jx), phase_np, atol=1e-8)
    assert np.allclose(np.asarray(path_jx), path_np, rtol=1e-10, atol=1e-8)
    # And it differentiates: the whole point of the jax path.
    grad = jax.grad(
        lambda x: jnp.sum(jnp.abs(rp.coarse_phase_and_path(
            x, jnp.asarray(ants_uvw), jnp.asarray(ants_xyz), jnp.asarray(freqs), windows, centres, weights, xp=jnp
        )[1]))
    )(jnp.asarray(rfi_xyz))
    assert np.all(np.isfinite(np.asarray(grad)))


# ---------------------------------------------------------------------------
# The reconstruction
# ---------------------------------------------------------------------------


def rebuild(order, n_int_freq=1, int_time=8.0, n_int_time=16):
    n_time = 6
    jd, rfi_xyz, ants_uvw, ants_xyz = geometry(n_time=n_time, n_int_time=n_int_time, int_time=int_time)
    chan_width = 1e6
    freqs = 1.4e9 + np.arange(2) * chan_width
    freqs_fine, dnu = rp.fine_frequency_terms(freqs, n_int_freq, chan_width)
    windows, centres, weights = rp.path_derivative_weights(jd, n_time, n_int_time, order)
    phase_c, path = rp.coarse_phase_and_path(rfi_xyz, ants_uvw, ants_xyz, freqs, windows, centres, weights)
    powers = rp.taylor_powers(fine_offsets(n_int_time, int_time), path.shape[-1] - 1)

    got = np.asarray(rp.fine_phase_from_path(
        jnp.asarray(phase_c), jnp.asarray(path), jnp.asarray(freqs_fine), jnp.asarray(dnu), jnp.asarray(powers)
    ))
    want = get_rfi_phase_numpy(rfi_xyz, ants_uvw, ants_xyz, freqs_fine)
    return got, want


def test_the_reconstruction_converges_on_the_fine_phase_with_the_order():
    """Within an integration the fringe turns; the expansion has to follow it.

    The differential phase, since a common per-source phase is dropped by
    design. On this pass -- 16 m/s of differential path rate on an 8 km
    baseline, 8 s integrations -- orders 0 and 1 are off by a turn, order 2 by
    a few degrees, and order 3 by a few hundredths of one, against a reference
    that is the definition on exact times. Order 4 gains nothing further.
    """
    errors = {}
    for order in range(5):
        got, want = rebuild(order)
        errors[order] = np.degrees(np.abs(wrap(differential(got) - differential(want))).max())

    assert errors[0] > 10.0 and errors[1] > 10.0
    assert errors[2] < errors[1] and errors[3] < errors[2]
    limit = 0.1 if active_precision() == "double" else 0.3
    assert errors[3] < limit and errors[4] < limit


def test_the_frequency_axis_is_exact():
    """At the centre time of a cell the time terms vanish and only the linear
    frequency term is left, which is exact: the fine-frequency phase there is
    the definition's to round-off, at any order."""
    n_int_time, n_int_freq = 4, 4
    got, want = rebuild(order=0, n_int_freq=n_int_freq, n_int_time=n_int_time)

    centre = slice(n_int_time // 2, None, n_int_time)
    err = np.abs(wrap(differential(got[..., centre]) - differential(want[..., centre]))).max()
    assert err < (1e-7 if active_precision() == "double" else 1e-3)


def test_fine_frequency_terms_lay_the_channels_out_channel_major():
    freqs_fine, dnu = rp.fine_frequency_terms([1.0e9, 1.2e9], 4, 2e8)
    assert np.allclose(dnu, np.tile(fine_offsets(4, 2e8), 2))
    assert np.allclose(freqs_fine, np.repeat([1.0e9, 1.2e9], 4) + dnu)
    # One sample per channel is the channel itself.
    freqs_fine, dnu = rp.fine_frequency_terms([1.0e9, 1.2e9], 1, 2e8)
    assert np.all(dnu == 0.0) and np.allclose(freqs_fine, [1.0e9, 1.2e9])


def test_taylor_powers():
    tau = np.array([-1.0, 0.0, 2.0])
    assert np.allclose(rp.taylor_powers(tau, 3), [[-1, 0, 2], [0.5, 0, 2], [-1 / 6, 0, 8 / 6]])
    assert rp.taylor_powers(tau, 0).shape == (0, 3)
