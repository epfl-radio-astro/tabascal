"""Tests for tabascal.interferometry.calculate_fringe_frequency."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import sgp4jax
from jax import vmap

jax.config.update("jax_enable_x64", True)

from tabascal.interferometry import calculate_fringe_frequency, C, Omega_e
from tabascal.time import mjd_to_jd
from tabascal.coordinates import itrf_to_uvw_jd


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

# Four compact-array antennas (ITRF, metres)
ANTS_ITRF = jnp.array([
    [5109360.133,  2006852.586, -3238948.127],
    [5109383.997,  2006807.012, -3238949.528],
    [5109366.222,  2006839.543, -3238966.127],
    [5109340.000,  2006900.000, -3238940.000],
])

RA, DEC = 120.0, -30.0
FREQ = 1.5e9          # Hz
N_TIME = 20
# 100 seconds of observations starting near J2000
T0_MJD = 51544.5
TIMES_MJD = jnp.linspace(T0_MJD, T0_MJD + 100.0 / 86400.0, N_TIME)


def ants_u_from_jd(ants_itrf, times_mjd, ra, dec):
    """Compute the U component of antenna positions in the UVW frame."""
    times_jd = mjd_to_jd(jnp.asarray(times_mjd))
    uvw = itrf_to_uvw_jd(ants_itrf, times_jd, ra, dec)  # (n_time, n_ant, 3)
    return uvw[:, :, 0]                                   # (n_time, n_ant)


def rfi_leo(times_mjd):
    """Synthetic LEO-like trajectory: straight line in GCRF at ~700 km altitude."""
    r0 = jnp.array([5100e3, 2000e3, 4000e3])   # metres, ~700 km altitude
    v = jnp.array([7000.0, 0.0, 0.0])           # 7 km/s in GCRF x-direction
    dt = (jnp.asarray(times_mjd) - times_mjd[0]) * 86400.0
    return r0[None, :] + v[None, :] * dt[:, None]  # (n_time, 3) metres


def rfi_geostationary(times_mjd):
    """Source stationary in ECEF: fixed ITRF point converted to GCRF at each time.

    For this source s_hat_ecef is constant, so fringe_move = 0 and
    fringe_freq = -fringe_stat exactly.
    """
    r_itrf_km = jnp.array([42164.0, 0.0, 0.0])   # GEO radius at 0° longitude, km
    times_jd = mjd_to_jd(jnp.asarray(times_mjd))
    jd_whole = jnp.floor(times_jd)
    jd_frac = times_jd - jd_whole
    rfi_gcrf_km = vmap(sgp4jax.itrf_to_gcrf, (None, 0, 0))(r_itrf_km, jd_whole, jd_frac)
    return rfi_gcrf_km * 1e3   # (n_time, 3) metres


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCalculateFringeFrequency:

    def setup_method(self):
        self.ants_u = ants_u_from_jd(ANTS_ITRF, TIMES_MJD, RA, DEC)
        self.rfi = rfi_leo(TIMES_MJD)
        self.n_ant = len(ANTS_ITRF)
        self.n_bl = self.n_ant * (self.n_ant - 1) // 2

    # --- shape ---

    def test_output_shape(self):
        ff = calculate_fringe_frequency(TIMES_MJD, FREQ, self.rfi, ANTS_ITRF, self.ants_u, DEC)
        assert ff.shape == (N_TIME, self.n_bl)

    def test_output_shape_two_antennas(self):
        ants = ANTS_ITRF[:2]
        ants_u = ants_u_from_jd(ants, TIMES_MJD, RA, DEC)
        ff = calculate_fringe_frequency(TIMES_MJD, FREQ, self.rfi, ants, ants_u, DEC)
        assert ff.shape == (N_TIME, 1)

    # --- frequency scaling ---

    def test_proportional_to_frequency(self):
        ff1 = calculate_fringe_frequency(TIMES_MJD, FREQ, self.rfi, ANTS_ITRF, self.ants_u, DEC)
        ff2 = calculate_fringe_frequency(TIMES_MJD, 2 * FREQ, self.rfi, ANTS_ITRF, self.ants_u, DEC)
        np.testing.assert_allclose(ff2, 2 * ff1, rtol=1e-10)

    # --- baseline sign ---

    def test_swapping_antenna_pair_negates_fringe_frequency(self):
        # With two antennas there is one baseline.  Swapping them flips bl_ecef
        # and bl_u, which negates both fringe_move and fringe_stat.
        ants = ANTS_ITRF[:2]
        ants_u = ants_u_from_jd(ants, TIMES_MJD, RA, DEC)

        ff_fwd = calculate_fringe_frequency(TIMES_MJD, FREQ, self.rfi, ants, ants_u, DEC)
        ff_bwd = calculate_fringe_frequency(
            TIMES_MJD, FREQ, self.rfi,
            ants[::-1], ants_u[:, ::-1], DEC,
        )
        np.testing.assert_allclose(ff_bwd, -ff_fwd, rtol=1e-10)

    # --- geostationary (fringe_move = 0) ---

    def test_geostationary_source_equals_earth_rotation_term(self):
        # A source stationary in ECEF has a constant ECEF unit vector, so
        # fringe_move = 0 and fringe_freq = -fringe_stat = bl_u * Omega_e * cos(dec) / lam.
        rfi_geo = rfi_geostationary(TIMES_MJD)
        ff = calculate_fringe_frequency(TIMES_MJD, FREQ, rfi_geo, ANTS_ITRF, self.ants_u, DEC)

        a1, a2 = jnp.triu_indices(self.n_ant, 1)
        bl_u = self.ants_u[:, a1] - self.ants_u[:, a2]   # (n_time, n_bl)
        lam = C / FREQ
        expected = bl_u * Omega_e * jnp.cos(jnp.deg2rad(DEC)) / lam

        np.testing.assert_allclose(ff, expected, rtol=1e-8)

    def test_moving_source_differs_from_earth_rotation_term(self):
        # A LEO satellite has nonzero fringe_move, so fringe_freq != -fringe_stat.
        ff = calculate_fringe_frequency(TIMES_MJD, FREQ, self.rfi, ANTS_ITRF, self.ants_u, DEC)

        a1, a2 = jnp.triu_indices(self.n_ant, 1)
        bl_u = self.ants_u[:, a1] - self.ants_u[:, a2]
        lam = C / FREQ
        fringe_stat_neg = bl_u * Omega_e * jnp.cos(jnp.deg2rad(DEC)) / lam

        assert not jnp.allclose(ff, fringe_stat_neg, rtol=1e-3)
