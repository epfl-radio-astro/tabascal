"""Baseline selection for the near-field matched filter (GitHub #189).

A phase-coherent filter toward a satellite only gains from a baseline whose
template phase is right. Two independent effects put a ceiling on the baseline
length over which that holds, and the shorter of the two binds:

* **TLE position error.** A transverse orbit error ``delta`` moves the apparent
  direction by ``delta / r``, which costs ``2 pi (b / lam) (delta / r)`` of phase
  on baseline ``b``. Keeping that below a radian gives

      b_tle = lam r / (2 pi delta)

* **Fringe rate inside the integration.** A satellite moving at ``v_perp``
  sweeps the baseline fringe at ``(b / lam) (v_perp / r)``. With ``n_fine``
  sub-steps per integration of length ``delta_t`` the model average itself can
  only follow ``n_fine / (2 delta_t)`` of that, so

      b_fringe = lam r n_fine / (2 delta_t v_perp)

  is where the template decoheres against its own discretisation, whatever the
  TLE quality.

Both are always in play: the three fringe inputs are required arguments, and a
caller with no fringe to worry about passes ``v_perp_m_s = 0``, which sends
``b_fringe`` to infinity and leaves the TLE ceiling as the only one that binds.
There is no half-specified call left to get wrong.

What the tests here pin is the *place the cut falls*, not that a cut happens: a
mask that flips at the wrong baseline still returns plausible booleans, still
excludes long baselines, and still "works" -- only the analytic tolerances tell
it apart from the right one. So everything below is checked against the closed
forms written out a second time in this file, and against the two worked numbers
in the issue: at 175.015 MHz and r = 567 km, a 600 m baseline tolerates a 258 m
transverse error while the full 5.3 km MWA array needs 29 m, which is why the
Cen A detection lived entirely in the 1004 baselines under 600 m and was diluted
away by the other 8176.

The mask is consumed by the jitted, GPU-resident matched-filter core, so it is
also pinned as a pure function of arrays: jittable, traceable, broadcasting, and
compiled once per shape rather than once per call.

Everything is synthetic and analytic; the ``--x64`` flag moves only the working
precision, so the comparisons go through ``exact_rtol`` rather than a literal
tolerance.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from tabascal.rfi_estimate import (
    baseline_lengths,
    coherent_baseline_mask,
    fringe_rate_coherence_length,
    tle_coherence_length,
)

C = 299792458.0  # m/s, as tabascal.interferometry.C


# ---------------------------------------------------------------------------
# The MWA Cen A case study of the issue, and the closed forms written out again
# ---------------------------------------------------------------------------

FREQ = 175.015e6  # Hz, the observing band
LAM = C / FREQ  # 1.712953 m
RANGE_TLE = 567e3  # m, slant range quoted with the TLE tolerances
RANGE_PASS = 570e3  # m, slant range quoted with the fringe rates
V_PERP = 7.1e3  # m/s, transverse speed of a Starlink pass
B_SHORT = 600.0  # m, the cut that carried the 5.6 sigma detection
B_LONG = 5300.0  # m, the longest MWA baseline in the dataset

# The fringe inputs are required, so a test that cares only about the TLE ceiling
# says so with a stationary emitter: v_perp = 0 puts b_fringe at infinity, which
# drops out of the minimum and leaves the TLE bound alone.
TLE_ONLY = {"n_fine": 40, "delta_t": 2.0, "v_perp_m_s": 0.0}


def b_tle_ref(freq, range_m, sigma):
    """``lam r / (2 pi delta)``, in host float64, owing nothing to the module."""
    return (C / np.asarray(freq)) * range_m / (2.0 * np.pi * np.asarray(sigma))


def b_fringe_ref(freq, range_m, n_fine, delta_t, v_perp):
    """``lam r n_fine / (2 delta_t v_perp)``, likewise written out."""
    return (C / np.asarray(freq)) * range_m * n_fine / (2.0 * delta_t * v_perp)


def fringe_rate(bl_len, freq, range_m, v_perp):
    """The per-baseline fringe rate ``(b / lam) (v_perp / r)`` in Hz."""
    return (np.asarray(bl_len) / (C / freq)) * (v_perp / range_m)


def assert_within(got, want, frac, what=""):
    """``got`` is within a fraction ``frac`` of the issue's rounded ``want``."""
    got = float(got)
    assert abs(got - want) <= frac * abs(want), (
        f"{what}: {got:.6g} is not within {100 * frac:g}% of the quoted {want:g}"
    )


# A small ITRF array (metres); the same five antennas the light-curve tests use.
ANTS = np.array(
    [
        [5109360.0, 2006852.0, -3238948.0],
        [5109340.0, 2006900.0, -3238900.0],
        [5109300.0, 2006800.0, -3239000.0],
        [5109420.0, 2006760.0, -3238860.0],
        [5109280.0, 2006940.0, -3239040.0],
    ]
)


class TestTleCoherenceLength:
    """b_tle = lam r / (2 pi delta): how long a baseline the orbit can steer."""

    def test_it_is_lambda_r_over_two_pi_delta(self, exact_rtol):
        got = tle_coherence_length(FREQ, RANGE_TLE, 200.0)

        np.testing.assert_allclose(
            got, b_tle_ref(FREQ, RANGE_TLE, 200.0), rtol=exact_rtol
        )

    def test_it_broadcasts_its_inputs(self, exact_rtol):
        freqs = np.array([100e6, 175.015e6, 400e6])[:, None]
        sigmas = np.array([50.0, 200.0, 1000.0])[None, :]

        got = tle_coherence_length(freqs, RANGE_TLE, sigmas)

        assert np.shape(got) == (3, 3)
        np.testing.assert_allclose(
            got, b_tle_ref(freqs, RANGE_TLE, sigmas), rtol=exact_rtol
        )

    def test_a_longer_wavelength_reaches_further(self):
        freqs = np.array([100e6, 175.015e6, 250e6, 400e6])

        got = np.asarray(tle_coherence_length(freqs, RANGE_TLE, 200.0))

        assert np.all(np.diff(got) < 0.0)

    def test_a_larger_position_error_reaches_less_far(self, exact_rtol):
        """Inverse in delta: doubling the orbit error halves the reach."""
        one = tle_coherence_length(FREQ, RANGE_TLE, 100.0)
        two = tle_coherence_length(FREQ, RANGE_TLE, 200.0)

        assert float(two) < float(one)
        np.testing.assert_allclose(2.0 * float(two), float(one), rtol=exact_rtol)

    def test_a_further_satellite_reaches_further(self, exact_rtol):
        """A given transverse error subtends a smaller angle further away."""
        near = tle_coherence_length(FREQ, 400e3, 200.0)
        far = tle_coherence_length(FREQ, 800e3, 200.0)

        np.testing.assert_allclose(2.0 * float(near), float(far), rtol=exact_rtol)

    def test_the_worked_mwa_tolerances(self, exact_rtol):
        """b_tle(delta) is symmetric in b <-> delta, so it inverts itself.

        delta_max(b) = lam r / (2 pi b) = b_tle evaluated at sigma = b, which is
        how the issue's two numbers are quoted: 600 m tolerates ~260 m of
        transverse TLE error, the full 5.3 km array needs ~29 m.
        """
        d_short = float(tle_coherence_length(FREQ, RANGE_TLE, B_SHORT))
        d_long = float(tle_coherence_length(FREQ, RANGE_TLE, B_LONG))

        np.testing.assert_allclose(
            [d_short, d_long],
            [b_tle_ref(FREQ, RANGE_TLE, B_SHORT), b_tle_ref(FREQ, RANGE_TLE, B_LONG)],
            rtol=exact_rtol,
        )
        assert_within(d_short, 260.0, 0.02, "600 m tolerance")
        assert_within(d_long, 29.0, 0.02, "5300 m tolerance")
        # And the map really is an involution: feed the tolerance back in.
        np.testing.assert_allclose(
            float(tle_coherence_length(FREQ, RANGE_TLE, d_short)),
            B_SHORT,
            rtol=exact_rtol,
        )


class TestFringeRateCoherenceLength:
    """b_fringe: where the fringe rate meets the fine-step Nyquist rate.

    Called positionally throughout, so these also pin the argument order
    ``(freq, range_m, n_fine, delta_t, v_perp_m_s)`` -- the same order the mask
    takes its fringe inputs in.
    """

    N_FINE, DELTA_T = 40, 2.0

    def test_it_is_lambda_r_n_fine_over_two_delta_t_v(self, exact_rtol):
        got = fringe_rate_coherence_length(
            FREQ, RANGE_PASS, self.N_FINE, self.DELTA_T, V_PERP
        )

        np.testing.assert_allclose(
            got,
            b_fringe_ref(FREQ, RANGE_PASS, self.N_FINE, self.DELTA_T, V_PERP),
            rtol=exact_rtol,
        )

    def test_it_broadcasts_its_inputs(self, exact_rtol):
        freqs = np.array([100e6, 175.015e6, 400e6])[:, None]
        n_fine = np.array([4, 10, 40, 100])[None, :]

        got = fringe_rate_coherence_length(
            freqs, RANGE_PASS, n_fine, self.DELTA_T, V_PERP
        )

        assert np.shape(got) == (3, 4)
        np.testing.assert_allclose(
            got,
            b_fringe_ref(freqs, RANGE_PASS, n_fine, self.DELTA_T, V_PERP),
            rtol=exact_rtol,
        )

    def test_its_fringe_rate_is_the_fine_step_nyquist_rate(self, exact_rtol):
        """The defining property: f(b_fringe) = n_fine / (2 delta_t)."""
        b_coh = float(
            fringe_rate_coherence_length(
                FREQ, RANGE_PASS, self.N_FINE, self.DELTA_T, V_PERP
            )
        )

        np.testing.assert_allclose(
            fringe_rate(b_coh, FREQ, RANGE_PASS, V_PERP),
            self.N_FINE / (2.0 * self.DELTA_T),
            rtol=exact_rtol,
        )

    def test_the_worked_starlink_fringe_rates(self):
        """~4 Hz at 600 m, ~38 Hz at 5300 m; the 10 Hz Nyquist falls between."""
        f_short = fringe_rate(B_SHORT, FREQ, RANGE_PASS, V_PERP)
        f_long = fringe_rate(B_LONG, FREQ, RANGE_PASS, V_PERP)
        nyquist = self.N_FINE / (2.0 * self.DELTA_T)

        assert abs(f_short - 4.0) < 0.5, f"600 m fringe rate {f_short:.3g} Hz"
        assert_within(f_long, 38.0, 0.02, "5300 m fringe rate")
        assert f_short < nyquist < f_long

        b_coh = float(
            fringe_rate_coherence_length(
                FREQ, RANGE_PASS, self.N_FINE, self.DELTA_T, V_PERP
            )
        )
        assert_within(b_coh, 1375.0, 0.02, "fringe coherence length")
        assert B_SHORT < b_coh < B_LONG

    def test_more_fine_steps_follow_a_longer_baseline(self, exact_rtol):
        few = float(
            fringe_rate_coherence_length(FREQ, RANGE_PASS, 4, self.DELTA_T, V_PERP)
        )
        many = float(
            fringe_rate_coherence_length(FREQ, RANGE_PASS, 40, self.DELTA_T, V_PERP)
        )

        np.testing.assert_allclose(10.0 * few, many, rtol=exact_rtol)

    def test_a_longer_integration_and_a_faster_pass_both_shorten_it(self):
        base = float(
            fringe_rate_coherence_length(
                FREQ, RANGE_PASS, self.N_FINE, self.DELTA_T, V_PERP
            )
        )
        slower_dump = float(
            fringe_rate_coherence_length(
                FREQ, RANGE_PASS, self.N_FINE, 4.0 * self.DELTA_T, V_PERP
            )
        )
        faster_sat = float(
            fringe_rate_coherence_length(
                FREQ, RANGE_PASS, self.N_FINE, self.DELTA_T, 2.0 * V_PERP
            )
        )

        assert slower_dump < base
        assert faster_sat < base


class TestCoherentBaselineMask:
    """The mask itself, with the fringe ceiling put out of the way."""

    BL = np.array([50.0, 200.0, 600.0, 1500.0, 5300.0])

    def test_a_stationary_emitter_reduces_it_to_the_tle_criterion(self, exact_rtol):
        """v_perp = 0 is how a caller asks for the TLE ceiling alone.

        The fringe inputs are required, so this is the escape hatch, and it has
        to be exact rather than merely large: b_fringe is infinite and drops out
        of the minimum, for the hard mask and the weights alike.
        """
        b_coh = b_tle_ref(FREQ, RANGE_TLE, 200.0)
        bl = np.linspace(0.0, 2.0 * b_coh, 9)

        hard = np.asarray(
            coherent_baseline_mask(self.BL, FREQ, RANGE_TLE, 200.0, **TLE_ONLY)
        )
        soft = np.asarray(
            coherent_baseline_mask(bl, FREQ, RANGE_TLE, 200.0, soft=True, **TLE_ONLY)
        )

        assert hard.dtype == np.bool_
        np.testing.assert_array_equal(hard, self.BL <= b_coh)
        np.testing.assert_allclose(
            soft, np.exp(-((bl / b_coh) ** 2)), rtol=exact_rtol
        )

    def test_the_cut_is_inclusive_at_the_coherence_length(self, exact_rtol):
        """b < b_coh and b == b_coh are coherent; b > b_coh is not.

        The comparison is against a b_coh built from the same lam, r and delta in
        the same dtype, so equality is meaningful; the two offsets are scaled off
        the working precision so the flip is resolvable in fp32 as well.
        """
        eps = 100.0 * exact_rtol
        sigma = 250.0
        b_coh = float(tle_coherence_length(FREQ, RANGE_TLE, sigma))
        bl = np.array([b_coh * (1.0 - eps), b_coh, b_coh * (1.0 + eps)])

        got = np.asarray(coherent_baseline_mask(bl, FREQ, RANGE_TLE, sigma, **TLE_ONLY))

        assert got.tolist() == [True, True, False]

    def test_the_cut_is_inclusive_at_the_fringe_ceiling_too(self, exact_rtol):
        """The same equality, on the other criterion.

        A perfect orbit puts b_tle at infinity, so the minimum is the fringe
        length bit for bit and the boundary lands exactly on it.
        """
        eps = 100.0 * exact_rtol
        n_fine, delta_t = 12, 2.0
        b_coh = float(
            fringe_rate_coherence_length(FREQ, RANGE_PASS, n_fine, delta_t, V_PERP)
        )
        bl = np.array([b_coh * (1.0 - eps), b_coh, b_coh * (1.0 + eps)])

        got = np.asarray(
            coherent_baseline_mask(
                bl, FREQ, RANGE_PASS, 0.0, n_fine, delta_t, V_PERP
            )
        )

        assert got.tolist() == [True, True, False]

    def test_six_hundred_metres_flips_at_a_two_hundred_and_fifty_eight_metre_tle(
        self, exact_rtol
    ):
        eps = 100.0 * exact_rtol
        d_max = b_tle_ref(FREQ, RANGE_TLE, B_SHORT)  # 257.6 m
        assert_within(d_max, 260.0, 0.02, "600 m tolerance")

        sigmas = np.array([d_max * (1.0 - eps), d_max * (1.0 + eps)])
        got = np.asarray(
            coherent_baseline_mask(B_SHORT, FREQ, RANGE_TLE, sigmas, **TLE_ONLY)
        )

        assert got.tolist() == [True, False]

    def test_the_full_array_flips_at_a_twenty_nine_metre_tle(self, exact_rtol):
        eps = 100.0 * exact_rtol
        d_max = b_tle_ref(FREQ, RANGE_TLE, B_LONG)  # 29.17 m
        assert_within(d_max, 29.0, 0.02, "5300 m tolerance")

        sigmas = np.array([d_max * (1.0 - eps), d_max * (1.0 + eps)])
        got = np.asarray(
            coherent_baseline_mask(B_LONG, FREQ, RANGE_TLE, sigmas, **TLE_ONLY)
        )

        assert got.tolist() == [True, False]

    def test_no_starlink_orbit_keeps_the_five_kilometre_baseline(self):
        """The case study: over the whole 0.1-1 km Starlink range, 5.3 km is out.

        The same sweep keeps 600 m in for the better half of that range, which is
        the asymmetry the Cen A detection turned on.
        """
        sigmas = np.array([100.0, 250.0, 500.0, 1000.0])
        bl = np.array([B_SHORT, B_LONG])[:, None]

        got = np.asarray(
            coherent_baseline_mask(bl, FREQ, RANGE_TLE, sigmas[None, :], **TLE_ONLY)
        )

        assert got.shape == (2, 4)
        assert not got[1].any(), "a 5.3 km baseline cannot survive a Starlink TLE"
        assert got[0].any(), "600 m survives the better Starlink TLEs"

    def test_the_mask_shrinks_as_the_orbit_gets_worse(self):
        sigmas = np.array([10.0, 50.0, 200.0, 1000.0])[:, None]

        got = np.asarray(
            coherent_baseline_mask(
                self.BL[None, :], FREQ, RANGE_TLE, sigmas, **TLE_ONLY
            )
        )

        counts = got.sum(axis=1)
        assert np.all(np.diff(counts) <= 0)
        assert counts[0] > counts[-1]


class TestBindingCriterion:
    """Two ceilings, per element the lower one is the one that matters."""

    def test_the_tle_binds_when_the_orbit_is_the_worse_of_the_two(self):
        """sigma = 500 m puts b_tle at 309 m while b_fringe is 1368 m."""
        sigma, n_fine, delta_t = 500.0, 40, 2.0
        b_tle = b_tle_ref(FREQ, RANGE_TLE, sigma)
        b_fri = b_fringe_ref(FREQ, RANGE_TLE, n_fine, delta_t, V_PERP)
        assert b_tle < B_SHORT < b_fri, "test case does not isolate the TLE bound"

        got = coherent_baseline_mask(
            B_SHORT,
            FREQ,
            RANGE_TLE,
            sigma,
            n_fine=n_fine,
            delta_t=delta_t,
            v_perp_m_s=V_PERP,
        )
        fringe_only = coherent_baseline_mask(
            B_SHORT, FREQ, RANGE_TLE, 0.0, n_fine, delta_t, V_PERP
        )

        assert not bool(got), "the TLE ceiling was ignored"
        assert bool(fringe_only), "the fringe ceiling alone would have kept it"

    def test_the_fine_step_binds_when_the_average_is_the_coarser(self):
        """sigma = 100 m puts b_tle at 1546 m, but 4 fine steps cut at 137 m."""
        sigma, n_fine, delta_t = 100.0, 4, 2.0
        b_tle = b_tle_ref(FREQ, RANGE_TLE, sigma)
        b_fri = b_fringe_ref(FREQ, RANGE_TLE, n_fine, delta_t, V_PERP)
        assert b_fri < B_SHORT < b_tle, "test case does not isolate the fringe bound"

        got = coherent_baseline_mask(
            B_SHORT,
            FREQ,
            RANGE_TLE,
            sigma,
            n_fine=n_fine,
            delta_t=delta_t,
            v_perp_m_s=V_PERP,
        )
        tle_only = coherent_baseline_mask(B_SHORT, FREQ, RANGE_TLE, sigma, **TLE_ONLY)

        assert not bool(got), "the fringe-rate ceiling was ignored"
        assert bool(tle_only), "the TLE ceiling alone would have kept it"

    def test_the_smaller_ceiling_wins_element_by_element(self):
        """A sweep in sigma where the binding criterion switches part-way.

        b_fringe is fixed at 410 m here while b_tle runs from 15 km down to
        129 m, so the fine step binds at the good end of the sweep and the orbit
        error binds at the bad end, within one call.
        """
        bl = np.geomspace(50.0, 2000.0, 6)[:, None]
        sigmas = np.array([10.0, 50.0, 100.0, 300.0, 600.0, 1200.0])[None, :]
        n_fine, delta_t = 12, 2.0

        b_tle = b_tle_ref(FREQ, RANGE_TLE, sigmas)
        b_fri = b_fringe_ref(FREQ, RANGE_TLE, n_fine, delta_t, V_PERP)
        assert (b_tle < b_fri).any() and (b_fri < b_tle).any(), "no switch in sweep"

        got = np.asarray(
            coherent_baseline_mask(
                bl,
                FREQ,
                RANGE_TLE,
                sigmas,
                n_fine=n_fine,
                delta_t=delta_t,
                v_perp_m_s=V_PERP,
            )
        )

        assert got.shape == (6, 6)
        assert got.any() and not got.all(), "a degenerate sweep proves nothing"
        np.testing.assert_array_equal(got, bl <= np.minimum(b_tle, b_fri))

    def test_adding_the_fringe_criterion_never_admits_a_baseline(self):
        bl = np.geomspace(20.0, 8000.0, 24)

        tle_only = np.asarray(
            coherent_baseline_mask(bl, FREQ, RANGE_TLE, 150.0, **TLE_ONLY)
        )
        both = np.asarray(
            coherent_baseline_mask(
                bl,
                FREQ,
                RANGE_TLE,
                150.0,
                n_fine=10,
                delta_t=2.0,
                v_perp_m_s=V_PERP,
            )
        )

        assert np.all(both <= tle_only)
        assert both.sum() < tle_only.sum(), "this case should tighten the cut"

    def test_the_fringe_arguments_are_positional_in_the_documented_order(self):
        """(..., n_fine, delta_t, v_perp_m_s): the last three of the seven."""
        bl = np.array([100.0, 600.0, 5300.0])

        positional = np.asarray(
            coherent_baseline_mask(bl, FREQ, RANGE_TLE, 150.0, 10, 2.0, V_PERP)
        )
        keyword = np.asarray(
            coherent_baseline_mask(
                bl,
                FREQ,
                RANGE_TLE,
                150.0,
                n_fine=10,
                delta_t=2.0,
                v_perp_m_s=V_PERP,
            )
        )

        np.testing.assert_array_equal(positional, keyword)


class TestMagnitudeConventions:
    """sigma and v_perp are magnitudes; a sign must not empty the mask.

    Both enter the physics through their size alone: a transverse offset to the
    left decoheres exactly as one to the right, and a satellite crossing the
    other way fringes at the same rate. A caller holding a signed component -- a
    velocity projected onto an axis, an offset measured along one -- would
    otherwise get a negative ceiling, and a negative ceiling fails quietly rather
    than loudly: ``b <= b_coh`` is simply false everywhere, which reads as "no
    baseline is coherent" and deletes the detection instead of reporting it.
    Each case below is arranged so the negated quantity is the binding one.
    """

    def test_a_negative_transverse_error_is_its_magnitude(self, exact_rtol):
        got = tle_coherence_length(FREQ, RANGE_TLE, -200.0)
        want = tle_coherence_length(FREQ, RANGE_TLE, 200.0)

        np.testing.assert_allclose(float(got), float(want), rtol=exact_rtol)

    def test_a_negative_transverse_speed_is_its_magnitude(self, exact_rtol):
        got = fringe_rate_coherence_length(FREQ, RANGE_PASS, 40, 2.0, -V_PERP)
        want = fringe_rate_coherence_length(FREQ, RANGE_PASS, 40, 2.0, V_PERP)

        np.testing.assert_allclose(float(got), float(want), rtol=exact_rtol)

    def test_the_mask_reads_a_negative_orbit_error_as_a_magnitude(self):
        """The TLE ceiling binds here, so the sign cannot hide behind the other."""
        bl = np.geomspace(20.0, 4000.0, 24)

        want = np.asarray(
            coherent_baseline_mask(bl, FREQ, RANGE_TLE, 200.0, **TLE_ONLY)
        )
        got = np.asarray(
            coherent_baseline_mask(bl, FREQ, RANGE_TLE, -200.0, **TLE_ONLY)
        )

        assert want.any() and not want.all(), "the reference cut is degenerate"
        np.testing.assert_array_equal(got, want)

    def test_the_mask_reads_a_negative_speed_as_a_magnitude(self):
        """A perfect orbit puts b_tle at infinity, leaving the fringe ceiling."""
        bl = np.geomspace(20.0, 4000.0, 24)

        want = np.asarray(
            coherent_baseline_mask(bl, FREQ, RANGE_PASS, 0.0, 12, 2.0, V_PERP)
        )
        got = np.asarray(
            coherent_baseline_mask(bl, FREQ, RANGE_PASS, 0.0, 12, 2.0, -V_PERP)
        )

        assert want.any() and not want.all(), "the reference cut is degenerate"
        np.testing.assert_array_equal(got, want)

    def test_the_weights_read_a_negative_orbit_error_as_a_magnitude(self, exact_rtol):
        """A negative b_tle would also win the minimum it has no business winning."""
        sigma, n_fine, delta_t = 100.0, 4, 2.0
        b_fri = b_fringe_ref(FREQ, RANGE_TLE, n_fine, delta_t, V_PERP)
        assert b_fri < b_tle_ref(FREQ, RANGE_TLE, sigma), "the fringe should bind"
        bl = np.linspace(0.0, 2.0 * b_fri, 9)

        want = np.asarray(
            coherent_baseline_mask(
                bl, FREQ, RANGE_TLE, sigma, n_fine, delta_t, V_PERP, soft=True
            )
        )
        got = np.asarray(
            coherent_baseline_mask(
                bl, FREQ, RANGE_TLE, -sigma, n_fine, delta_t, V_PERP, soft=True
            )
        )

        np.testing.assert_allclose(got, want, rtol=exact_rtol)

    def test_the_weights_read_a_negative_speed_as_a_magnitude(self, exact_rtol):
        """The mirror image: a negative b_fringe stealing the minimum from b_tle."""
        sigma, n_fine, delta_t = 500.0, 40, 2.0
        b_tle = b_tle_ref(FREQ, RANGE_TLE, sigma)
        assert b_tle < b_fringe_ref(FREQ, RANGE_TLE, n_fine, delta_t, V_PERP), (
            "the TLE should bind"
        )
        bl = np.linspace(0.0, 2.0 * b_tle, 9)

        want = np.asarray(
            coherent_baseline_mask(
                bl, FREQ, RANGE_TLE, sigma, n_fine, delta_t, V_PERP, soft=True
            )
        )
        got = np.asarray(
            coherent_baseline_mask(
                bl, FREQ, RANGE_TLE, sigma, n_fine, delta_t, -V_PERP, soft=True
            )
        )

        np.testing.assert_allclose(got, want, rtol=exact_rtol)


class TestSoftWeights:
    """soft=True: exp(-(b / b_coh)^2) instead of a step at b_coh."""

    SIGMA = 200.0

    def test_it_is_a_gaussian_in_baseline_length(self, exact_rtol):
        b_coh = b_tle_ref(FREQ, RANGE_TLE, self.SIGMA)
        bl = np.linspace(0.0, 2.0 * b_coh, 9)

        got = np.asarray(
            coherent_baseline_mask(bl, FREQ, RANGE_TLE, self.SIGMA, soft=True, **TLE_ONLY)
        )

        assert np.issubdtype(got.dtype, np.floating)
        np.testing.assert_allclose(got, np.exp(-((bl / b_coh) ** 2)), rtol=exact_rtol)

    def test_a_zero_baseline_is_fully_weighted(self, exact_rtol):
        got = coherent_baseline_mask(
            0.0, FREQ, RANGE_TLE, self.SIGMA, soft=True, **TLE_ONLY
        )

        np.testing.assert_allclose(float(got), 1.0, rtol=exact_rtol)

    def test_the_coherence_length_is_the_one_over_e_point(self, exact_rtol):
        b_coh = float(tle_coherence_length(FREQ, RANGE_TLE, self.SIGMA))

        got = coherent_baseline_mask(
            b_coh, FREQ, RANGE_TLE, self.SIGMA, soft=True, **TLE_ONLY
        )

        np.testing.assert_allclose(float(got), np.exp(-1.0), rtol=exact_rtol)

    def test_it_halves_at_the_coherence_length_times_root_log_two(self, exact_rtol):
        b_coh = float(tle_coherence_length(FREQ, RANGE_TLE, self.SIGMA))

        got = coherent_baseline_mask(
            b_coh * np.sqrt(np.log(2.0)),
            FREQ,
            RANGE_TLE,
            self.SIGMA,
            soft=True,
            **TLE_ONLY,
        )

        np.testing.assert_allclose(float(got), 0.5, rtol=exact_rtol)

    def test_it_decreases_monotonically_and_stays_in_the_unit_interval(self):
        b_coh = b_tle_ref(FREQ, RANGE_TLE, self.SIGMA)
        bl = np.linspace(0.0, 3.0 * b_coh, 25)

        got = np.asarray(
            coherent_baseline_mask(bl, FREQ, RANGE_TLE, self.SIGMA, soft=True, **TLE_ONLY)
        )

        assert np.all(got > 0.0) and np.all(got <= 1.0)
        assert np.all(np.diff(got) < 0.0)

    def test_the_binding_criterion_sets_the_scale(self, exact_rtol):
        """With the fine step binding, the weights fall off on b_fringe."""
        sigma, n_fine, delta_t = 100.0, 4, 2.0
        b_tle = b_tle_ref(FREQ, RANGE_TLE, sigma)
        b_fri = b_fringe_ref(FREQ, RANGE_TLE, n_fine, delta_t, V_PERP)
        assert b_fri < b_tle
        bl = np.linspace(0.0, 2.0 * b_fri, 9)

        got = np.asarray(
            coherent_baseline_mask(
                bl,
                FREQ,
                RANGE_TLE,
                sigma,
                n_fine=n_fine,
                delta_t=delta_t,
                v_perp_m_s=V_PERP,
                soft=True,
            )
        )

        np.testing.assert_allclose(got, np.exp(-((bl / b_fri) ** 2)), rtol=exact_rtol)
        assert not np.allclose(got, np.exp(-((bl / b_tle) ** 2)), rtol=1e-3)

    def test_the_soft_and_hard_cuts_agree_on_where_the_scale_is(self):
        """The half-weight point sits outside the hard mask, the 1/e point at it."""
        bl = np.geomspace(20.0, 8000.0, 40)
        hard = np.asarray(
            coherent_baseline_mask(bl, FREQ, RANGE_TLE, self.SIGMA, **TLE_ONLY)
        )
        soft = np.asarray(
            coherent_baseline_mask(bl, FREQ, RANGE_TLE, self.SIGMA, soft=True, **TLE_ONLY)
        )

        # Every baseline the hard mask keeps carries at least 1/e of its weight,
        # and everything it drops carries less.
        assert np.all(soft[hard] >= np.exp(-1.0) * (1.0 - 1e-6))
        assert np.all(soft[~hard] < np.exp(-1.0))


class TestBroadcasting:
    """The mask is a shape-polymorphic elementwise function, not a per-call cut."""

    def test_a_vector_of_baselines_with_scalar_conditions(self):
        bl = np.geomspace(20.0, 8000.0, 17)

        got = coherent_baseline_mask(bl, FREQ, RANGE_TLE, 200.0, **TLE_ONLY)

        assert np.shape(got) == (17,)

    def test_a_baseline_by_frequency_grid(self):
        bl = np.geomspace(20.0, 4000.0, 31)
        freqs = np.array([100e6, 150e6, 175.015e6, 250e6, 400e6])

        got = np.asarray(
            coherent_baseline_mask(bl[:, None], freqs, RANGE_TLE, 200.0, **TLE_ONLY)
        )

        assert got.shape == (31, 5)
        per_freq = got.sum(axis=0)
        assert np.all(np.diff(per_freq) <= 0), (
            "a higher frequency must not admit more baselines than a lower one"
        )
        assert per_freq[0] > per_freq[-1]
        np.testing.assert_array_equal(
            got, bl[:, None] <= b_tle_ref(freqs, RANGE_TLE, 200.0)
        )

    def test_a_source_by_baseline_grid(self):
        """One range and orbit error per satellite, one length per baseline."""
        bl = np.geomspace(20.0, 4000.0, 11)[None, :]
        ranges = np.array([500e3, 700e3, 1200e3])[:, None]
        sigmas = np.array([100.0, 400.0, 800.0])[:, None]

        got = np.asarray(coherent_baseline_mask(bl, FREQ, ranges, sigmas, **TLE_ONLY))

        assert got.shape == (3, 11)
        np.testing.assert_array_equal(got, bl <= b_tle_ref(FREQ, ranges, sigmas))

    def test_the_fringe_inputs_broadcast_too(self):
        bl = np.geomspace(20.0, 4000.0, 11)[None, :]
        n_fine = np.array([4, 12, 40])[:, None]

        got = np.asarray(
            coherent_baseline_mask(
                bl,
                FREQ,
                RANGE_PASS,
                150.0,
                n_fine=n_fine,
                delta_t=2.0,
                v_perp_m_s=V_PERP,
            )
        )

        assert got.shape == (3, 11)
        want = bl <= np.minimum(
            b_tle_ref(FREQ, RANGE_PASS, 150.0),
            b_fringe_ref(FREQ, RANGE_PASS, n_fine, 2.0, V_PERP),
        )
        np.testing.assert_array_equal(got, want)


class TestJaxCompatibility:
    """The matched-filter core is jitted and GPU-resident; so is this."""

    BL = jnp.geomspace(20.0, 8000.0, 16)

    def test_the_hard_mask_survives_jit(self):
        eager = np.asarray(
            coherent_baseline_mask(self.BL, FREQ, RANGE_TLE, 200.0, **TLE_ONLY)
        )

        jitted = np.asarray(
            jax.jit(coherent_baseline_mask)(
                self.BL, FREQ, RANGE_TLE, 200.0, **TLE_ONLY
            )
        )

        np.testing.assert_array_equal(jitted, eager)

    def test_the_soft_weights_survive_jit(self, exact_rtol):
        eager = np.asarray(
            coherent_baseline_mask(
                self.BL, FREQ, RANGE_TLE, 200.0, soft=True, **TLE_ONLY
            )
        )

        by_partial = np.asarray(
            jax.jit(partial(coherent_baseline_mask, soft=True))(
                self.BL, FREQ, RANGE_TLE, 200.0, **TLE_ONLY
            )
        )
        by_static = np.asarray(
            jax.jit(coherent_baseline_mask, static_argnames=("soft",))(
                self.BL, FREQ, RANGE_TLE, 200.0, soft=True, **TLE_ONLY
            )
        )

        np.testing.assert_allclose(by_partial, eager, rtol=exact_rtol)
        np.testing.assert_allclose(by_static, eager, rtol=exact_rtol)

    def test_the_fringe_inputs_may_themselves_be_traced(self):
        eager = np.asarray(
            coherent_baseline_mask(
                self.BL,
                FREQ,
                RANGE_PASS,
                150.0,
                n_fine=12,
                delta_t=2.0,
                v_perp_m_s=V_PERP,
            )
        )

        jitted = np.asarray(
            jax.jit(coherent_baseline_mask)(
                self.BL,
                jnp.asarray(FREQ),
                jnp.asarray(RANGE_PASS),
                jnp.asarray(150.0),
                jnp.asarray(12.0),
                jnp.asarray(2.0),
                jnp.asarray(V_PERP),
            )
        )

        np.testing.assert_array_equal(jitted, eager)

    def test_it_composes_inside_another_jitted_function(self):
        """The consumer's own trace: no host callback, no concrete boolean."""

        @jax.jit
        def n_coherent(bl_len, freq, range_m, sigma):
            mask = coherent_baseline_mask(bl_len, freq, range_m, sigma, **TLE_ONLY)
            return jnp.sum(jnp.where(mask, 1.0, 0.0))

        want = np.asarray(
            coherent_baseline_mask(self.BL, FREQ, RANGE_TLE, 200.0, **TLE_ONLY)
        ).sum()

        got = n_coherent(self.BL, jnp.asarray(FREQ), jnp.asarray(RANGE_TLE),
                         jnp.asarray(200.0))

        assert float(got) == float(want)

    def test_it_is_traced_once_per_shape_and_not_once_per_call(self):
        """New *values* of the same shape must hit the compilation cache."""
        n_traces = 0

        def traced(bl_len, freq, range_m, sigma):
            nonlocal n_traces
            n_traces += 1
            return coherent_baseline_mask(bl_len, freq, range_m, sigma, **TLE_ONLY)

        fn = jax.jit(traced)
        freq, rng = jnp.asarray(FREQ), jnp.asarray(RANGE_TLE)
        for sigma in (100.0, 250.0, 700.0):
            jax.block_until_ready(fn(self.BL, freq, rng, jnp.asarray(sigma)))

        assert n_traces == 1, f"recompiled {n_traces} times for one shape"

        jax.block_until_ready(
            fn(jnp.geomspace(20.0, 8000.0, 32), freq, rng, jnp.asarray(250.0))
        )

        assert n_traces == 2, "a new baseline count must trace a new program"

    def test_the_dtypes_are_boolean_and_floating(self):
        hard = coherent_baseline_mask(self.BL, FREQ, RANGE_TLE, 200.0, **TLE_ONLY)
        soft = coherent_baseline_mask(
            self.BL, FREQ, RANGE_TLE, 200.0, soft=True, **TLE_ONLY
        )

        assert hard.dtype == jnp.bool_
        assert jnp.issubdtype(soft.dtype, jnp.floating)
        # And in whichever precision the session is running in.
        assert soft.dtype == jnp.zeros(1).dtype

    def test_the_coherence_lengths_are_jittable_too(self, exact_rtol):
        freqs = jnp.asarray([100e6, 175.015e6, 400e6])

        tle = jax.jit(tle_coherence_length)(freqs, RANGE_TLE, 200.0)
        fri = jax.jit(fringe_rate_coherence_length)(
            freqs, RANGE_PASS, 40, 2.0, V_PERP
        )

        np.testing.assert_allclose(
            np.asarray(tle), b_tle_ref(np.asarray(freqs), RANGE_TLE, 200.0),
            rtol=exact_rtol,
        )
        np.testing.assert_allclose(
            np.asarray(fri),
            b_fringe_ref(np.asarray(freqs), RANGE_PASS, 40, 2.0, V_PERP),
            rtol=exact_rtol,
        )


class TestBaselineLengths:
    """The physical 3-D separation the criteria are applied to."""

    def test_it_is_the_norm_of_the_antenna_separation(self, exact_rtol):
        a1, a2 = np.triu_indices(len(ANTS), 1)
        ants = jnp.asarray(ANTS)
        # Reference in the session's own dtype: the cancellation in x[a1] - x[a2]
        # is the same on both sides, so what is left is rounding, not precision.
        host = np.asarray(ants)
        want = np.linalg.norm(host[a1] - host[a2], axis=-1)

        got = np.asarray(baseline_lengths(ants, a1, a2))

        assert got.shape == (len(a1),)
        assert np.issubdtype(got.dtype, np.floating)
        np.testing.assert_allclose(got, want, rtol=exact_rtol)

    def test_a_numpy_array_is_accepted(self, exact_rtol):
        a1, a2 = np.triu_indices(len(ANTS), 1)

        got = np.asarray(baseline_lengths(ANTS, a1, a2))
        want = np.asarray(baseline_lengths(jnp.asarray(ANTS), jnp.asarray(a1),
                                          jnp.asarray(a2)))

        np.testing.assert_allclose(got, want, rtol=exact_rtol)

    def test_autocorrelations_have_zero_length(self):
        idx = np.arange(len(ANTS))

        got = np.asarray(baseline_lengths(ANTS, idx, idx))

        np.testing.assert_array_equal(got, np.zeros(len(ANTS)))

    def test_it_does_not_care_which_antenna_comes_first(self, exact_rtol):
        a1, a2 = np.triu_indices(len(ANTS), 1)

        forward = np.asarray(baseline_lengths(ANTS, a1, a2))
        backward = np.asarray(baseline_lengths(ANTS, a2, a1))

        np.testing.assert_allclose(forward, backward, rtol=exact_rtol)

    def test_the_lengths_are_the_ones_the_mask_is_meant_to_cut_on(self):
        """End to end: lengths in, per-baseline booleans out, same shape."""
        a1, a2 = np.triu_indices(len(ANTS), 1)
        bl = baseline_lengths(ANTS, a1, a2)

        got = np.asarray(coherent_baseline_mask(bl, FREQ, RANGE_TLE, 200.0, **TLE_ONLY))

        assert got.shape == (len(a1),)
        assert got.dtype == np.bool_


class TestDegenerateInputs:
    """Where the ceiling goes to infinity, nothing may go to nan."""

    BL = np.array([0.0, 600.0, 5300.0, 1.0e5])

    def test_a_perfect_orbit_leaves_every_baseline_coherent(self):
        """delta = 0 puts b_coh at infinity, not at 0/0."""
        hard = np.asarray(
            coherent_baseline_mask(self.BL, FREQ, RANGE_TLE, 0.0, **TLE_ONLY)
        )

        assert not np.any(np.isnan(hard.astype(float)))
        assert hard.all()

    def test_a_perfect_orbit_leaves_every_weight_at_one(self, exact_rtol):
        soft = np.asarray(
            coherent_baseline_mask(self.BL, FREQ, RANGE_TLE, 0.0, soft=True, **TLE_ONLY)
        )

        assert not np.any(np.isnan(soft))
        np.testing.assert_allclose(soft, np.ones_like(soft), rtol=exact_rtol)

    def test_an_infinite_coherence_length_is_also_fine_from_the_fringe_side(self):
        """A stationary emitter has no fringe rate to outrun."""
        got = np.asarray(
            coherent_baseline_mask(
                self.BL,
                FREQ,
                RANGE_TLE,
                0.0,
                n_fine=40,
                delta_t=2.0,
                v_perp_m_s=0.0,
                soft=True,
            )
        )

        assert not np.any(np.isnan(got))
        assert np.all(got > 0.0)

    def test_a_zero_length_baseline_is_always_coherent(self, exact_rtol):
        hard = coherent_baseline_mask(0.0, FREQ, RANGE_TLE, 1.0e4, **TLE_ONLY)
        soft = coherent_baseline_mask(
            0.0, FREQ, RANGE_TLE, 1.0e4, soft=True, **TLE_ONLY
        )

        assert bool(hard)
        np.testing.assert_allclose(float(soft), 1.0, rtol=exact_rtol)

    # --- the same degeneracies, compiled, and one criterion at a time --------

    def test_under_jit_an_infinite_tle_ceiling_leaves_the_fringe_one(self, exact_rtol):
        """sigma = 0 against a real pass: min(inf, b_fringe) must be b_fringe.

        Isolating it this way is the point: with both ceilings infinite an
        implementation that mishandles the minimum still looks right, because
        everything is coherent either way.
        """
        b_fri = float(
            fringe_rate_coherence_length(FREQ, RANGE_PASS, 12, 2.0, V_PERP)
        )
        bl_soft = np.linspace(0.0, 2.0 * b_fri, 5)

        hard = np.asarray(
            jax.jit(coherent_baseline_mask)(
                self.BL, FREQ, RANGE_PASS, 0.0, 12, 2.0, V_PERP
            )
        )
        soft = np.asarray(
            jax.jit(partial(coherent_baseline_mask, soft=True))(
                bl_soft, FREQ, RANGE_PASS, 0.0, 12, 2.0, V_PERP
            )
        )

        assert hard.any() and not hard.all(), "the fringe ceiling is not binding"
        np.testing.assert_array_equal(hard, self.BL <= b_fri)
        np.testing.assert_allclose(
            soft, np.exp(-((bl_soft / b_fri) ** 2)), rtol=exact_rtol
        )

    def test_under_jit_an_infinite_fringe_ceiling_leaves_the_tle_one(self, exact_rtol):
        """v_perp = 0 against a real orbit: min(b_tle, inf) must be b_tle."""
        b_tle = float(tle_coherence_length(FREQ, RANGE_TLE, 200.0))
        bl_soft = np.linspace(0.0, 2.0 * b_tle, 5)

        hard = np.asarray(
            jax.jit(coherent_baseline_mask)(
                self.BL, FREQ, RANGE_TLE, 200.0, 40, 2.0, 0.0
            )
        )
        soft = np.asarray(
            jax.jit(partial(coherent_baseline_mask, soft=True))(
                bl_soft, FREQ, RANGE_TLE, 200.0, 40, 2.0, 0.0
            )
        )

        assert hard.any() and not hard.all(), "the TLE ceiling is not binding"
        np.testing.assert_array_equal(hard, self.BL <= b_tle)
        np.testing.assert_allclose(
            soft, np.exp(-((bl_soft / b_tle) ** 2)), rtol=exact_rtol
        )

    def test_under_jit_two_infinite_ceilings_are_still_not_a_nan(self):
        """Both zero: every baseline in, every weight exactly one.

        Exactly one, not approximately: b / inf is 0 and exp(-0) is 1 in IEEE, so
        anything else here means an epsilon or a guard crept into the division.
        """
        hard = np.asarray(
            jax.jit(coherent_baseline_mask)(self.BL, FREQ, RANGE_TLE, 0.0, 40, 2.0, 0.0)
        )
        soft = np.asarray(
            jax.jit(partial(coherent_baseline_mask, soft=True))(
                self.BL, FREQ, RANGE_TLE, 0.0, 40, 2.0, 0.0
            )
        )

        assert hard.all()
        np.testing.assert_array_equal(soft, np.ones_like(soft))
