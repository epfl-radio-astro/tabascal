"""Tests for tabascal.interferometry.calculate_fringe_frequency_numpy.

These cover the host-side (numpy) fringe-frequency calculation used during
config setup.  The numpy path works in the GAST-only ECI frame: a source's
ECI position is rotated by the Greenwich apparent sidereal angle to ECEF
internally, so synthetic sources here are constructed in that same frame.
"""

import numpy as np
import pytest

from tabascal.interferometry import (
    C,
    Omega_e,
    Rotz_numpy,
    calculate_fringe_frequency_numpy,
    itrf_to_uvw_numpy,
)
from tabascal.time import mjd_to_jd, gast_deg


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

# Four compact-array antennas (ITRF, metres)
ANTS_ITRF = np.array([
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
TIMES_MJD = np.linspace(T0_MJD, T0_MJD + 100.0 / 86400.0, N_TIME)


def ants_u_from_mjd(ants_itrf, times_mjd, ra, dec):
    """U component of antenna positions in the UVW frame (n_time, n_ant)."""
    gh0 = (gast_deg(mjd_to_jd(times_mjd)) - ra) % 360
    return itrf_to_uvw_numpy(ants_itrf, gh0, dec)[:, :, 0]


def rfi_eci_line(times_mjd):
    """Synthetic moving source: straight line in the ECI frame at ~700 km."""
    r0 = np.array([5100e3, 2000e3, 4000e3])   # metres
    v = np.array([7000.0, 0.0, 0.0])           # 7 km/s in ECI x-direction
    dt = (np.asarray(times_mjd) - times_mjd[0]) * 86400.0
    return r0[None, :] + v[None, :] * dt[:, None]  # (n_time, 3) metres


def rfi_eci_geostationary(times_mjd):
    """Source stationary in ECEF, expressed in the ECI frame.

    A fixed ECEF point rotated into ECI by the GAST at each time.  After the
    function rotates it back to ECEF its position is constant, so s_hat_ecef
    is constant, fringe_move = 0 and fringe_freq = -fringe_stat exactly.
    """
    r_ecef = np.array([42164e3, 0.0, 0.0])   # GEO radius at 0 deg longitude, metres
    gsa = gast_deg(mjd_to_jd(times_mjd))
    return np.array([Rotz_numpy(g) @ r_ecef for g in gsa])  # (n_time, 3)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCalculateFringeFrequencyNumpy:

    def setup_method(self):
        self.ants_u = ants_u_from_mjd(ANTS_ITRF, TIMES_MJD, RA, DEC)
        self.rfi = rfi_eci_line(TIMES_MJD)
        self.n_ant = len(ANTS_ITRF)
        self.n_bl = self.n_ant * (self.n_ant - 1) // 2

    # --- shape ---

    def test_output_shape(self):
        ff = calculate_fringe_frequency_numpy(TIMES_MJD, FREQ, self.rfi, ANTS_ITRF, self.ants_u, DEC)
        assert ff.shape == (N_TIME, self.n_bl)

    def test_output_shape_two_antennas(self):
        ants = ANTS_ITRF[:2]
        ants_u = ants_u_from_mjd(ants, TIMES_MJD, RA, DEC)
        ff = calculate_fringe_frequency_numpy(TIMES_MJD, FREQ, self.rfi, ants, ants_u, DEC)
        assert ff.shape == (N_TIME, 1)

    # --- frequency scaling ---

    def test_proportional_to_frequency(self):
        ff1 = calculate_fringe_frequency_numpy(TIMES_MJD, FREQ, self.rfi, ANTS_ITRF, self.ants_u, DEC)
        ff2 = calculate_fringe_frequency_numpy(TIMES_MJD, 2 * FREQ, self.rfi, ANTS_ITRF, self.ants_u, DEC)
        np.testing.assert_allclose(ff2, 2 * ff1, rtol=1e-10)

    # --- baseline sign ---

    def test_swapping_antenna_pair_negates_fringe_frequency(self):
        # With two antennas there is one baseline.  Swapping them flips bl_ecef
        # and bl_u, which negates both fringe_move and fringe_stat.
        ants = ANTS_ITRF[:2]
        ants_u = ants_u_from_mjd(ants, TIMES_MJD, RA, DEC)

        ff_fwd = calculate_fringe_frequency_numpy(TIMES_MJD, FREQ, self.rfi, ants, ants_u, DEC)
        ff_bwd = calculate_fringe_frequency_numpy(
            TIMES_MJD, FREQ, self.rfi,
            ants[::-1], ants_u[:, ::-1], DEC,
        )
        np.testing.assert_allclose(ff_bwd, -ff_fwd, rtol=1e-10)

    # --- geostationary (fringe_move = 0) ---

    def test_geostationary_source_equals_earth_rotation_term(self):
        # A source stationary in ECEF has a constant ECEF unit vector, so
        # fringe_move = 0 and fringe_freq = -fringe_stat = bl_u * Omega_e * cos(dec) / lam.
        rfi_geo = rfi_eci_geostationary(TIMES_MJD)
        ff = calculate_fringe_frequency_numpy(TIMES_MJD, FREQ, rfi_geo, ANTS_ITRF, self.ants_u, DEC)

        a1, a2 = np.triu_indices(self.n_ant, 1)
        bl_u = self.ants_u[:, a1] - self.ants_u[:, a2]   # (n_time, n_bl)
        lam = C / FREQ
        expected = bl_u * float(Omega_e) * np.cos(np.deg2rad(DEC)) / lam

        np.testing.assert_allclose(ff, expected, rtol=1e-6)

    def test_moving_source_differs_from_earth_rotation_term(self):
        # A moving source has nonzero fringe_move, so fringe_freq != -fringe_stat.
        ff = calculate_fringe_frequency_numpy(TIMES_MJD, FREQ, self.rfi, ANTS_ITRF, self.ants_u, DEC)

        a1, a2 = np.triu_indices(self.n_ant, 1)
        bl_u = self.ants_u[:, a1] - self.ants_u[:, a2]
        lam = C / FREQ
        fringe_stat_neg = bl_u * float(Omega_e) * np.cos(np.deg2rad(DEC)) / lam

        assert not np.allclose(ff, fringe_stat_neg, rtol=1e-3)
