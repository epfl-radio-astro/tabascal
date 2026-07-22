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
    get_ast_fringe_rate,
    get_divisors,
    get_strides_and_idxs,
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


# ---------------------------------------------------------------------------
# get_strides_and_idxs
#
# Groups per-baseline sampling rates into stride bins for
# RiemannVisTimeFreqVariable. The hard downstream invariant is that every
# returned stride divides max_sampling (= n_int_time, the fine-grid size), since
# each group slices the fine grid as slice(stride//2, None, stride). The binning
# must also partition every baseline into exactly one non-empty group, and must
# not collapse to a single group merely because max(samplings) is prime.
# ---------------------------------------------------------------------------

def _assert_valid_grouping(samplings, idxs, u_strides, max_sampling, min_bins):
    """Assert the structural invariants every grouping must satisfy."""
    samplings = np.asarray(samplings)
    n = samplings.size

    # max_sampling (= n_int_time) must cover the largest required sampling.
    assert max_sampling >= int(samplings.max())

    # Hard invariant: every stride divides max_sampling so the fine-grid slicing
    # stays uniform downstream.
    for s in u_strides:
        assert max_sampling % s == 0, (s, max_sampling)

    # Strides are unique, sorted, and plain ints.
    assert list(u_strides) == sorted(set(u_strides))
    assert all(isinstance(s, int) for s in u_strides)

    # One index group per stride.
    assert len(idxs) == len(u_strides)

    # Groups are non-empty and partition every baseline exactly once.
    for grp in idxs:
        assert len(grp) > 0
    concat = np.concatenate([np.asarray(g) for g in idxs])
    np.testing.assert_array_equal(np.sort(concat), np.arange(n))


class TestGetStridesAndIdxs:

    MIN_BINS, MAX_BINS = 1, 30

    @pytest.mark.parametrize("seed", range(8))
    def test_invariants_on_random_samplings(self, seed):
        # Random spreads of per-baseline sampling rates must always satisfy the
        # structural invariants regardless of where max(samplings) lands.
        rng = np.random.default_rng(seed)
        samplings = rng.integers(1, 60, size=200)
        idxs, u_strides, max_sampling = get_strides_and_idxs(
            samplings, self.MIN_BINS, self.MAX_BINS
        )
        _assert_valid_grouping(samplings, idxs, u_strides, max_sampling, self.MIN_BINS)

    def test_prime_max_does_not_collapse(self):
        # Regression: with the old divisors(max(samplings)) scheme a prime max
        # (43 -> divisors {1, 43}) collapsed every baseline onto a single stride.
        # A genuinely spread distribution must now yield more than one group.
        rng = np.random.default_rng(0)
        samplings = rng.integers(2, 44, size=300)
        samplings[0] = 43  # force a prime max
        assert int(samplings.max()) == 43

        idxs, u_strides, max_sampling = get_strides_and_idxs(
            samplings, self.MIN_BINS, self.MAX_BINS
        )
        _assert_valid_grouping(samplings, idxs, u_strides, max_sampling, self.MIN_BINS)
        assert len(u_strides) > 1

    def test_max_sampling_is_divisor_rich(self):
        # When the overshoot cap is not hit, max_sampling is bumped up to an
        # integer with at least min_divisors divisors.
        min_divisors = 4
        samplings = np.array([43])  # prime; needs bumping for richer divisors
        _, _, max_sampling = get_strides_and_idxs(
            samplings, self.MIN_BINS, self.MAX_BINS, min_divisors=min_divisors
        )
        need = max(min_divisors, self.MIN_BINS + 1)
        assert len(get_divisors(max_sampling)) >= need
        assert max_sampling >= 43

    def test_overshoot_is_bounded(self):
        # max_sampling never more than doubles max(samplings); divisor-poor
        # maxima fall back to the 2*M cap rather than searching unboundedly.
        for m in [9, 17, 43, 97]:
            _, _, max_sampling = get_strides_and_idxs(
                np.array([m]), self.MIN_BINS, self.MAX_BINS
            )
            assert m <= max_sampling <= 2 * m

    def test_all_equal_samplings_single_group(self):
        # Identical sampling on every baseline cannot be meaningfully split, so a
        # single valid group is expected (and must not raise).
        samplings = np.full(50, 10)
        idxs, u_strides, max_sampling = get_strides_and_idxs(
            samplings, self.MIN_BINS, self.MAX_BINS
        )
        _assert_valid_grouping(samplings, idxs, u_strides, max_sampling, self.MIN_BINS)
        assert len(u_strides) == 1

    def test_spread_samplings_produce_multiple_groups(self):
        # A broad spread of sampling rates should resolve into several groups.
        samplings = np.repeat(np.arange(1, 25), 8)  # 1..24, evenly populated
        idxs, u_strides, max_sampling = get_strides_and_idxs(
            samplings, self.MIN_BINS, self.MAX_BINS
        )
        _assert_valid_grouping(samplings, idxs, u_strides, max_sampling, self.MIN_BINS)
        assert len(u_strides) >= self.MIN_BINS

    def test_single_baseline(self):
        # Degenerate one-baseline case stays well defined.
        idxs, u_strides, max_sampling = get_strides_and_idxs(
            np.array([7]), self.MIN_BINS, self.MAX_BINS
        )
        _assert_valid_grouping(np.array([7]), idxs, u_strides, max_sampling, self.MIN_BINS)
        assert len(u_strides) == 1

    def test_higher_min_divisors_is_at_least_as_rich(self):
        # Raising min_divisors cannot reduce the divisor count of max_sampling.
        samplings = np.array([43])
        counts = [
            len(get_divisors(get_strides_and_idxs(
                samplings, self.MIN_BINS, self.MAX_BINS, min_divisors=r
            )[2]))
            for r in (1, 2, 4, 6)
        ]
        assert counts == sorted(counts)


# ---------------------------------------------------------------------------
# get_ast_fringe_rate
# ---------------------------------------------------------------------------

def _chord(freq, D):
    """Sky-displacement chord 2 sin(bw / 4) of a source at the beam half-angle."""
    lam = C / freq
    bw = 1.22 * lam / D
    return 2 * np.sin(bw / 4)


class TestGetAstFringeRate:
    """The maximum astronomical fringe rate f_max is

        f_max = (Omega_e / lam) |s - s0| max_t sqrt(u^2 sin^2 d
                                                    + (v sin d - w cos d)^2)

    with the sky-displacement chord |s - s0| = 2 sin(bw / 4).  These tests pin
    the two limiting geometries (pole and equator), the shape/broadcasting, and
    the monotonic dependence on baseline length and beam width.
    """

    FREQ = 1.5e9
    D = 13.5

    # --- shape ---

    def test_output_shape(self):
        rng = np.random.default_rng(0)
        n_time, n_bl = 12, 6
        uvw = rng.normal(scale=1000.0, size=(n_time, n_bl, 3))
        fr = get_ast_fringe_rate(uvw, DEC, self.FREQ, self.D)
        assert fr.shape == (n_bl,)

    def test_single_baseline_single_time(self):
        uvw = np.array([[[300.0, 400.0, 500.0]]])  # (1, 1, 3)
        fr = get_ast_fringe_rate(uvw, DEC, self.FREQ, self.D)
        assert fr.shape == (1,)

    # --- analytic limiting geometries ---

    def test_pole_reduces_to_uv_projection(self):
        # At dec = 90 deg the pole lies along w_hat, so the projection collapses
        # to the uv-plane baseline length sqrt(u^2 + v^2).
        uvw = np.array([[[300.0, 400.0, 900.0]]])  # sqrt(u^2+v^2) = 500, w ignored
        lam = C / self.FREQ
        expected = float(Omega_e) * 500.0 * _chord(self.FREQ, self.D) / lam
        fr = get_ast_fringe_rate(uvw, 90.0, self.FREQ, self.D)
        np.testing.assert_allclose(np.asarray(fr), [expected], rtol=1e-6)

    def test_equator_reduces_to_w_component(self):
        # At dec = 0 deg the pole lies along v_hat, so the projection collapses
        # to |w| and is independent of u and v.
        uvw = np.array([[[300.0, 400.0, 700.0]]])
        lam = C / self.FREQ
        expected = float(Omega_e) * 700.0 * _chord(self.FREQ, self.D) / lam
        fr = get_ast_fringe_rate(uvw, 0.0, self.FREQ, self.D)
        np.testing.assert_allclose(np.asarray(fr), [expected], rtol=1e-6)

    def test_declination_dependence_is_real(self):
        # A baseline whose uv length differs from |w| must give a different rate
        # at the pole than at the equator.
        uvw = np.array([[[300.0, 400.0, 700.0]]])  # sqrt(u^2+v^2)=500 != |w|=700
        fr_pole = float(get_ast_fringe_rate(uvw, 90.0, self.FREQ, self.D)[0])
        fr_equ = float(get_ast_fringe_rate(uvw, 0.0, self.FREQ, self.D)[0])
        assert not np.isclose(fr_pole, fr_equ, rtol=1e-3)

    # --- time reduction ---

    def test_takes_max_over_time(self):
        # The largest per-sample projection sets the rate (evaluated at the pole
        # where the projection is just sqrt(u^2 + v^2)).
        uvw = np.array([
            [[100.0, 0.0, 0.0]],
            [[500.0, 0.0, 0.0]],  # largest
            [[300.0, 0.0, 0.0]],
        ])
        lam = C / self.FREQ
        expected = float(Omega_e) * 500.0 * _chord(self.FREQ, self.D) / lam
        fr = get_ast_fringe_rate(uvw, 90.0, self.FREQ, self.D)
        np.testing.assert_allclose(np.asarray(fr), [expected], rtol=1e-6)

    # --- monotonicity / scaling ---

    def test_proportional_to_baseline_length(self):
        uvw = np.array([[[300.0, 400.0, 500.0]]])
        fr1 = float(get_ast_fringe_rate(uvw, DEC, self.FREQ, self.D)[0])
        fr2 = float(get_ast_fringe_rate(2 * uvw, DEC, self.FREQ, self.D)[0])
        np.testing.assert_allclose(fr2, 2 * fr1, rtol=1e-6)

    def test_increases_with_beam_width(self):
        # A smaller dish -> wider beam -> larger sky displacement -> higher rate.
        uvw = np.array([[[300.0, 400.0, 500.0]]])
        fr_wide = float(get_ast_fringe_rate(uvw, DEC, self.FREQ, 5.0)[0])
        fr_narrow = float(get_ast_fringe_rate(uvw, DEC, self.FREQ, 50.0)[0])
        assert fr_wide > fr_narrow > 0.0

    def test_small_angle_matches_direction_cosine(self):
        # For a narrow beam the chord 2 sin(bw / 4) agrees with the old
        # small-angle direction cosine sin(bw / 2) to high precision.
        lam = C / self.FREQ
        bw = 1.22 * lam / self.D
        np.testing.assert_allclose(_chord(self.FREQ, self.D), np.sin(bw / 2), rtol=1e-3)
