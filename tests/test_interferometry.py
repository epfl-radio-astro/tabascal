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
    fov_to_eff_diameter,
    max_ast_fringe_rate,
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
# max_ast_fringe_rate
# ---------------------------------------------------------------------------

def _brute_force_fringe_rate(uvw, dec_deg, freq, D, n_chi=4000):
    """Reference max fringe rate by maximising f = (1/lam) b.(Omega x (s - s0))
    directly over the beam azimuth and time, with no closed-form simplification.

    The source sits at the beam radius rho = 1.22 lam / D (the first null, the
    maximising offset), and we sweep the azimuth chi around s0 = w_hat.
    """
    lam = C / freq
    d = np.deg2rad(dec_deg)
    rho = 1.22 * lam / D

    nhat = np.array([0.0, np.cos(d), np.sin(d)])  # celestial pole in UVW frame
    s0 = np.array([0.0, 0.0, 1.0])

    chi = np.linspace(0.0, 2 * np.pi, n_chi)
    # source offsets on the sphere at angle rho, azimuth chi (shape (n_chi, 3))
    e = np.stack([np.cos(chi), np.sin(chi), np.zeros_like(chi)], axis=-1)
    ds = np.cos(rho) * s0 + np.sin(rho) * e - s0  # s - s0

    b = uvw  # (n_time, n_bl, 3)
    # f = (1/lam) b . (Omega x ds), Omega = Omega_e * nhat
    cross = np.cross(nhat[None, :], ds)  # (n_chi, 3)
    f = np.einsum("tbi,ci->tbc", b, cross) * float(Omega_e) / lam
    return np.max(np.abs(f), axis=(0, 2))  # max over time and azimuth -> (n_bl,)


class TestMaxAstFringeRate:
    """The exact maximum astronomical fringe rate over the beam is

        f_max = (Omega_e / lam) [ A sin(rho) + B (1 - cos(rho)) ]
        A = sqrt((v sin d - w cos d)^2 + (u sin d)^2)
        B = |u cos d|

    with the beam radius rho = 1.22 lam / D (first null): the transverse
    (sin rho) plus radial / (n - 1) curvature (1 - cos rho) couplings. The beam
    position is always maximised over; time and frequency are maximised over
    when arrays are supplied. These tests pin it against an independent
    brute-force maximisation over the sphere, the two limiting declinations, the
    presence of the radial term, the accepted input shapes, each reduction axis,
    and the monotonic dependences.
    """

    FREQ = 1.5e9
    D = 13.5

    # --- accepted shapes ---

    def test_output_shape(self):
        rng = np.random.default_rng(0)
        n_time, n_bl = 12, 6
        uvw = rng.normal(scale=1000.0, size=(n_time, n_bl, 3))
        fr = max_ast_fringe_rate(uvw, DEC, self.FREQ, self.D)
        assert fr.shape == (n_bl,)

    def test_single_baseline_single_time(self):
        uvw = np.array([[[300.0, 400.0, 500.0]]])  # (1, 1, 3)
        fr = max_ast_fringe_rate(uvw, DEC, self.FREQ, self.D)
        assert fr.shape == (1,)

    def test_scalar_sample_returns_scalar(self):
        # A single (3,) UVW sample is still maximised over the beam, and returns
        # a scalar rather than a length-1 baseline axis.
        fr = max_ast_fringe_rate(np.array([300.0, 400.0, 700.0]), DEC, self.FREQ, self.D)
        assert np.asarray(fr).shape == ()

    def test_scalar_sample_matches_observation_of_one(self):
        sample = np.array([300.0, 400.0, 700.0])
        scalar = float(max_ast_fringe_rate(sample, DEC, self.FREQ, self.D))
        obs = float(
            max_ast_fringe_rate(sample[None, None, :], DEC, self.FREQ, self.D)[0]
        )
        np.testing.assert_allclose(scalar, obs, rtol=1e-12)

    @pytest.mark.parametrize("bad", [(5, 3), (3, 5), (2, 2, 2), (4,)])
    def test_rejects_ambiguous_shapes(self, bad):
        # Only (3,) and (n_time, n_bl, 3) are accepted, so no axis is ever guessed.
        with pytest.raises(ValueError, match="uvw must have shape"):
            max_ast_fringe_rate(np.zeros(bad), DEC, self.FREQ, self.D)

    # --- reductions ---

    def test_max_over_frequency(self):
        # Supplying a band must give the largest of the per-channel rates.
        uvw = np.array([[[300.0, 400.0, 700.0]]])
        freqs = np.array([0.9e9, 1.2e9, 1.5e9])
        each = [
            float(max_ast_fringe_rate(uvw, DEC, f, self.D)[0]) for f in freqs
        ]
        band = float(max_ast_fringe_rate(uvw, DEC, freqs, self.D)[0])
        np.testing.assert_allclose(band, max(each), rtol=1e-12)

    def test_lowest_frequency_can_set_the_maximum(self):
        # The rate is not monotonic in frequency. To leading order the transverse
        # term is frequency-independent (rho ~ lam cancels the 1/lam), so the
        # radial term -- which grows linearly with lam -- decides. Differentiating
        # f(lam) shows the lowest channel wins when B > (2/3) A rho. At dec = 0,
        # A = |w| and B = |u|, so a baseline with appreciable u satisfies it.
        uvw = np.array([[[300.0, 400.0, 700.0]]])
        freqs = np.array([0.9e9, 1.5e9])
        lo = float(max_ast_fringe_rate(uvw, 0.0, freqs.min(), self.D)[0])
        hi = float(max_ast_fringe_rate(uvw, 0.0, freqs.max(), self.D)[0])
        band = float(max_ast_fringe_rate(uvw, 0.0, freqs, self.D)[0])

        assert lo > hi, "expected the lowest channel to dominate at dec = 0"
        np.testing.assert_allclose(band, lo, rtol=1e-12)

    def test_highest_frequency_sets_the_maximum_at_the_pole(self):
        # Contrast to the above: at the pole the radial term vanishes (B = 0), so
        # f = Omega_e A sin(rho) / lam falls with lam and the highest channel wins.
        uvw = np.array([[[300.0, 400.0, 700.0]]])
        freqs = np.array([0.9e9, 1.5e9])
        lo = float(max_ast_fringe_rate(uvw, 90.0, freqs.min(), self.D)[0])
        hi = float(max_ast_fringe_rate(uvw, 90.0, freqs.max(), self.D)[0])
        band = float(max_ast_fringe_rate(uvw, 90.0, freqs, self.D)[0])

        assert hi > lo, "expected the highest channel to dominate at the pole"
        np.testing.assert_allclose(band, hi, rtol=1e-12)

    def test_baselines_are_independent(self):
        # Each entry of the (n_bl,) result must equal that baseline computed alone,
        # i.e. the baseline vmap does not mix baselines.
        rng = np.random.default_rng(3)
        uvw = rng.normal(scale=800.0, size=(5, 4, 3))
        freqs = np.array([1.0e9, 1.5e9])
        together = np.asarray(max_ast_fringe_rate(uvw, DEC, freqs, self.D))
        for b in range(uvw.shape[1]):
            alone = np.asarray(
                max_ast_fringe_rate(uvw[:, b : b + 1, :], DEC, freqs, self.D)
            )
            np.testing.assert_allclose(together[b], alone[0], rtol=1e-12)

    # --- agreement with independent brute-force maximisation ---

    @pytest.mark.parametrize("dec_deg", [-60.0, -30.0, 0.0, 45.0, 90.0])
    @pytest.mark.parametrize("D", [13.5, 2.0, 0.4])  # narrow to very wide beam
    def test_matches_brute_force(self, dec_deg, D):
        rng = np.random.default_rng(1)
        uvw = rng.normal(scale=500.0, size=(8, 5, 3))
        fr = np.asarray(max_ast_fringe_rate(uvw, dec_deg, self.FREQ, D))
        ref = _brute_force_fringe_rate(uvw, dec_deg, self.FREQ, D)
        np.testing.assert_allclose(fr, ref, rtol=1e-4)

    # --- analytic limiting geometries ---

    def test_pole_reduces_to_uv_projection(self):
        # At dec = 90 deg the radial term vanishes and the projection collapses
        # to the uv-plane baseline length sqrt(u^2 + v^2); w is irrelevant.
        uvw = np.array([[[300.0, 400.0, 900.0]]])  # sqrt(u^2+v^2) = 500
        lam = C / self.FREQ
        rho = 1.22 * lam / self.D
        expected = float(Omega_e) * np.sin(rho) * 500.0 / lam
        fr = max_ast_fringe_rate(uvw, 90.0, self.FREQ, self.D)
        np.testing.assert_allclose(np.asarray(fr), [expected], rtol=1e-6)

    def test_equator_transverse_plus_radial(self):
        # At dec = 0 deg the transverse term uses |w| and the radial term uses
        # |u|: f = (Omega_e/lam)[sin(rho)|w| + (1-cos(rho))|u|].
        uvw = np.array([[[300.0, 400.0, 700.0]]])
        lam = C / self.FREQ
        rho = 1.22 * lam / self.D
        expected = (
            float(Omega_e)
            * (np.sin(rho) * 700.0 + (1 - np.cos(rho)) * 300.0)
            / lam
        )
        fr = max_ast_fringe_rate(uvw, 0.0, self.FREQ, self.D)
        np.testing.assert_allclose(np.asarray(fr), [expected], rtol=1e-6)

    def test_radial_term_is_present_off_pole(self):
        # Away from the pole a wide beam must exceed the transverse-only estimate
        # because of the (n - 1) radial contribution.
        uvw = np.array([[[600.0, 100.0, 100.0]]])
        lam = C / self.FREQ
        rho = 1.22 * lam / 0.5  # wide beam
        transverse_only = float(Omega_e) * np.sin(rho) * abs(100.0) / lam
        fr = float(max_ast_fringe_rate(uvw, 0.0, self.FREQ, 0.5)[0])
        assert fr > transverse_only

    def test_declination_dependence_is_real(self):
        uvw = np.array([[[300.0, 400.0, 700.0]]])
        fr_pole = float(max_ast_fringe_rate(uvw, 90.0, self.FREQ, self.D)[0])
        fr_equ = float(max_ast_fringe_rate(uvw, 0.0, self.FREQ, self.D)[0])
        assert not np.isclose(fr_pole, fr_equ, rtol=1e-3)

    # --- time reduction ---

    def test_takes_max_over_time(self):
        # The largest per-sample term sets the rate (evaluated at the pole where
        # the projection is just sqrt(u^2 + v^2)).
        uvw = np.array([
            [[100.0, 0.0, 0.0]],
            [[500.0, 0.0, 0.0]],  # largest
            [[300.0, 0.0, 0.0]],
        ])
        lam = C / self.FREQ
        rho = 1.22 * lam / self.D
        expected = float(Omega_e) * np.sin(rho) * 500.0 / lam
        fr = max_ast_fringe_rate(uvw, 90.0, self.FREQ, self.D)
        np.testing.assert_allclose(np.asarray(fr), [expected], rtol=1e-6)

    # --- monotonicity / scaling ---

    def test_proportional_to_baseline_length(self):
        uvw = np.array([[[300.0, 400.0, 500.0]]])
        fr1 = float(max_ast_fringe_rate(uvw, DEC, self.FREQ, self.D)[0])
        fr2 = float(max_ast_fringe_rate(2 * uvw, DEC, self.FREQ, self.D)[0])
        np.testing.assert_allclose(fr2, 2 * fr1, rtol=1e-6)

    def test_increases_with_beam_width(self):
        # A smaller dish -> wider beam -> larger sky displacement -> higher rate.
        uvw = np.array([[[300.0, 400.0, 500.0]]])
        fr_wide = float(max_ast_fringe_rate(uvw, DEC, self.FREQ, 5.0)[0])
        fr_narrow = float(max_ast_fringe_rate(uvw, DEC, self.FREQ, 50.0)[0])
        assert fr_wide > fr_narrow > 0.0

    # --- fov_deg contract ---
    #
    # These lock the user-facing meaning of the `fov_deg` config parameter: it
    # is the *full* field of view, so the maximum source offset used by the
    # fringe rate (and hence the power-spectrum knee k0) is fov_deg / 2.  The
    # 1.22 in max_ast_fringe_rate and the 2.44 in fov_to_eff_diameter must stay
    # in step; changing either alone breaks these.

    @pytest.mark.parametrize("fov_deg", [0.5, 5.0, 20.0])
    def test_fov_to_eff_diameter_gives_beam_radius_of_half_the_fov(
        self, fov_deg, exact_rtol
    ):
        D = float(fov_to_eff_diameter(fov_deg, self.FREQ))
        rho = 1.22 * (C / self.FREQ) / D  # beam radius used by max_ast_fringe_rate
        np.testing.assert_allclose(np.rad2deg(rho), fov_deg / 2, rtol=exact_rtol)

    @pytest.mark.parametrize("fov_deg", [0.5, 5.0, 20.0])
    def test_fringe_rate_from_fov_matches_explicit_half_fov_offset(
        self, fov_deg, exact_rtol
    ):
        # End-to-end: driving max_ast_fringe_rate through fov_to_eff_diameter
        # must equal evaluating the closed form with rho = fov_deg / 2.
        uvw = np.array([[[300.0, 400.0, 700.0]]])
        dec_deg = -30.0
        d = np.deg2rad(dec_deg)
        lam = C / self.FREQ
        rho = np.deg2rad(fov_deg / 2)

        u, v, w = uvw[0, 0]
        expected = (
            float(Omega_e)
            * (
                np.sin(rho) * np.sqrt((v * np.sin(d) - w * np.cos(d)) ** 2
                                      + (u * np.sin(d)) ** 2)
                + (1 - np.cos(rho)) * abs(u * np.cos(d))
            )
            / lam
        )

        D = float(fov_to_eff_diameter(fov_deg, self.FREQ))
        fr = max_ast_fringe_rate(uvw, dec_deg, self.FREQ, D)
        np.testing.assert_allclose(np.asarray(fr), [expected], rtol=exact_rtol)
