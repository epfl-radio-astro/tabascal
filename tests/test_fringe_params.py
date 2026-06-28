"""Tests for the host-side analytic fringe-parameter machinery in interferometry.

fit_nearfield_fringe_freq_poly_numpy / fringe_params_at_offsets / size_subwindows produce
the per-sub-window f, fdot (and the cubic-curvature-sized K) that feed the analytic
RFI-visibility path. They are G1-clean and *near-field consistent*: f is the derivative of
the same unwrapped near-field path D_p = (|ant - rfi| + w)/lambda that get_rfi_phase uses
(phi = -2*pi*D), fitted per window and differentiated analytically (never by differencing
wrapped phase). Validated here against a synthetic moving source with smooth ECI antenna
positions: the polynomial f must equal -dD/ds, fdot its derivative, and K must follow the
cubic-curvature rule.
"""

import numpy as np
import pytest

from tabascal.interferometry import (
    C,
    fit_nearfield_fringe_freq_poly_numpy,
    fringe_params_at_offsets,
    size_subwindows,
)

FREQ = 1.227e9
N_TIME = 8
INT_TIME = 2.0   # seconds per window
N_FIT = 16       # fit samples per window
N_ANT = 4


def build_times():
    """Uniform window-contiguous fit grid: N_FIT samples per coarse window."""
    n = N_TIME * N_FIT
    dt_fit = INT_TIME / N_FIT
    return (np.arange(n) + 0.5) * dt_fit - INT_TIME / 2.0  # seconds from start


def geometry(times_sec, sat_speed=7500.0):
    """Synthetic near-field geometry: ECI antennas (slowly rotating) + a fast LEO source.

    Returns rfi_xyz (1, n, 3), ants_xyz (N_ANT, n, 3), ants_w (N_ANT, n).
    """
    n = len(times_sec)
    rng = np.random.default_rng(0)
    # Antenna ECEF-ish positions, given a slow ECI rotation (Earth spin ~7.3e-5 rad/s).
    base = np.array([5109360.0, 2006852.0, -3238948.0])
    ant_off = np.concatenate([np.zeros((1, 3)), rng.uniform(-4000, 4000, (N_ANT - 1, 3))])
    ants0 = base + ant_off
    omega = 7.292e-5
    th = omega * times_sec
    cz, sz = np.cos(th), np.sin(th)
    # rotate about z (ECI spin) -> (N_ANT, n, 3)
    ax = ants0[:, 0][:, None] * cz[None, :] - ants0[:, 1][:, None] * sz[None, :]
    ay = ants0[:, 0][:, None] * sz[None, :] + ants0[:, 1][:, None] * cz[None, :]
    az = np.repeat(ants0[:, 2][:, None], n, axis=1)
    ants_xyz = np.stack([ax, ay, az], axis=-1)
    # LEO source ~550 km up, moving fast, passing near the array zenith.
    r0 = base / np.linalg.norm(base) * (np.linalg.norm(base) + 550e3)
    v = np.array([sat_speed, 0.0, 0.0])
    rfi = (r0[None, :] + v[None, :] * times_sec[:, None])[None]  # (1, n, 3)
    # fringe-stop w term: small smooth per-antenna offset.
    ants_w = 1e-3 * (ant_off[:, 0][:, None] * np.cos(0.01 * times_sec)[None, :])
    return rfi, ants_xyz, ants_w


@pytest.fixture
def fitted():
    times = build_times()
    rfi, ants_xyz, ants_w = geometry(times)
    a1, a2 = np.triu_indices(N_ANT, 1)
    coeffs = fit_nearfield_fringe_freq_poly_numpy(
        times, FREQ, rfi, ants_xyz, ants_w, a1, a2, N_TIME
    )
    return times, rfi, ants_xyz, ants_w, a1, a2, coeffs


def test_coeffs_shape(fitted):
    *_, a1, a2, coeffs = fitted
    n_bl = N_ANT * (N_ANT - 1) // 2
    assert coeffs.shape == (1, n_bl, N_TIME, 4)  # f cubic -> 4 coeffs


def test_f_equals_minus_dD_ds(fitted):
    """f at a window centre equals -dD_pq/ds of the true near-field path (finite diff)."""
    times, rfi, ants_xyz, ants_w, a1, a2, coeffs = fitted
    lam = C / FREQ
    f0, _ = fringe_params_at_offsets(coeffs, np.array([0.0]))  # (1, n_bl, N_TIME)
    # Ground-truth f from a finite difference of the near-field path at each window centre.
    win = N_TIME // 2
    ci = win * N_FIT + N_FIT // 2  # a sample near window centre
    h_idx = 1
    dt_samp = times[1] - times[0]
    D = (
        np.linalg.norm(ants_xyz[:, ci - h_idx : ci + h_idx + 1] - rfi[0, ci - h_idx : ci + h_idx + 1], axis=-1)
        + ants_w[:, ci - h_idx : ci + h_idx + 1]
    ) / lam  # (N_ANT, 3)
    for b in range(len(a1)):
        Dpq = D[a1[b]] - D[a2[b]]
        f_fd = -(Dpq[2] - Dpq[0]) / (2 * dt_samp)
        # f0 is at the exact window centre; ci is the nearest sample -> loose tol.
        np.testing.assert_allclose(f0[0, b, win], f_fd, rtol=2e-2, atol=1e-3)


def test_fdot_is_polynomial_derivative(fitted):
    *_, coeffs = fitted
    offs = np.array([-0.3, 0.0, 0.7])
    _, fdot = fringe_params_at_offsets(coeffs, offs)
    n_bl = coeffs.shape[1]
    fdot = fdot.reshape(1, n_bl, N_TIME, len(offs))
    for b in range(n_bl):
        for i in range(N_TIME):
            der = np.polyder(coeffs[0, b, i])
            np.testing.assert_allclose(fdot[0, b, i], np.polyval(der, offs), atol=1e-12)


def test_params_at_offsets_layout(fitted):
    *_, coeffs = fitted
    K = 3
    offs = -INT_TIME / 2 + (np.arange(K) + 0.5) * (INT_TIME / K)
    f, fdot = fringe_params_at_offsets(coeffs, offs)
    n_bl = coeffs.shape[1]
    assert f.shape == (1, n_bl, N_TIME * K)
    f_check = np.polyval(coeffs[0, 2, 0], offs[1])
    assert abs(f[0, 2, 1] - f_check) < 1e-9


def test_size_subwindows_rule(fitted):
    *_, coeffs = fitted
    K_loose, fddot = size_subwindows(coeffs, INT_TIME, resid_tol=1e-2)
    K_tight, _ = size_subwindows(coeffs, INT_TIME, resid_tol=1e-6)
    assert K_loose >= 1 and K_tight >= K_loose
    A = 3.3
    expected = max(1, int(np.ceil((A * fddot * INT_TIME**3 / 1e-2) ** (1 / 3))))
    assert K_loose == expected


def test_size_subwindows_flat_source_is_K1():
    """A distant, slow source (negligible chirp) needs only K=1."""
    times = build_times()
    rng = np.random.default_rng(1)
    # Far (GEO-distance), slowly drifting source -> tiny fddot.
    rfi = (np.array([4.2e7, 1e6, 1e6])[None, None]
           + np.array([5.0, 0.0, 0.0])[None, None] * times[None, :, None])
    base = np.array([5109360.0, 2006852.0, -3238948.0])
    ants0 = base + np.concatenate([np.zeros((1, 3)), rng.uniform(-50, 50, (N_ANT - 1, 3))])
    ants_xyz = np.repeat(ants0[:, None, :], len(times), axis=1)
    ants_w = np.zeros((N_ANT, len(times)))
    a1, a2 = np.triu_indices(N_ANT, 1)
    coeffs = fit_nearfield_fringe_freq_poly_numpy(times, FREQ, rfi, ants_xyz, ants_w, a1, a2, N_TIME)
    K, _ = size_subwindows(coeffs, INT_TIME, resid_tol=3e-4)
    assert K == 1
