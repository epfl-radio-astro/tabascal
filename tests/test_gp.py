"""Tests for tabascal.gp — the squared-exponential kernels and the resampling matrix.

The subject throughout is where the jitter belongs. A covariance of a grid with
ITSELF is inverted or factorised, and needs a diagonal to survive that; a
covariance BETWEEN two grids is only ever multiplied, and a diagonal added to it
is not regularisation but a bias on everything predicted through it.
"""

import jax.numpy as jnp
import pytest

from tabascal.gp import base_kernel, get_times, kernel, resampling_kernel


#: The node grid :func:`grids` builds, and the kernel it is measured against. The
#: grid is coarse enough that its covariance is well conditioned (smallest eigenvalue
#: 1.4), so every bound below is set by the jitter rather than by the inverse.
N_NODES = 8
VAR = 4.0
CORR_LENGTH = 10.0

#: The jitter the gain kernels are built with at a narrow prior.
JITTER = 1e-8


def grids(n_out):
    """A node grid and a resampling grid, offset so no point of one is a point of the other.

    What the tests turn on is the two having the same LENGTH: a cross-covariance
    between grids of equal length comes out square without either grid being the
    other, which is what used to attract the jitter.
    """

    return jnp.linspace(0.0, 100.0, N_NODES), jnp.linspace(3.0, 97.0, n_out)


def reference_resampling(x, x_, var, l, noise):
    """``K_s K^-1`` written out, with the jitter on the inverted matrix and nowhere else.

    :func:`~tabascal.gp.base_kernel` takes no jitter at all, so this is the resampling
    matrix the conditional mean is defined by, independently of anything
    :func:`~tabascal.gp.kernel` might add.
    """

    return base_kernel(x, x_, var, l) @ jnp.linalg.inv(
        base_kernel(x, x, var, l) + noise * jnp.eye(len(x))
    )


@pytest.fixture(params=[N_NODES, N_NODES + 1], ids=["equal-lengths", "differing-lengths"])
def n_out(request):
    """The two grid pairs: one that used to trip the shape test, one that never did."""
    return request.param


class TestTheCrossCovarianceCarriesNoJitter:
    """Two different grids, so nothing here is inverted and nothing needs a diagonal."""

    def test_the_kernel_is_the_covariance_and_nothing_else(self, n_out, exact_rtol):
        x, x_ = grids(n_out)

        difference = kernel(x, x_, VAR, CORR_LENGTH) - base_kernel(x, x_, VAR, CORR_LENGTH)

        assert float(jnp.max(jnp.abs(difference))) <= exact_rtol * VAR

    def test_the_resampling_matrix_is_the_conditional(self, n_out, exact_rtol):
        x, x_ = grids(n_out)

        resample = resampling_kernel(x, x_, VAR, CORR_LENGTH, JITTER)
        reference = reference_resampling(x, x_, VAR, CORR_LENGTH, JITTER)
        difference = float(jnp.max(jnp.abs(resample - reference)))

        assert difference <= exact_rtol * float(jnp.max(jnp.abs(reference)))


class TestTheInvertedCovarianceKeepsItsJitter:
    """The other half of the guarantee: the matrix that IS inverted still gets one.

    The node covariance is what ``noise`` was always for, and a squared-exponential
    Gram matrix on a node grid does not invert without it.
    """

    #: Larger than any jitter in use, so that its effect on the inverse is
    #: unmistakable against the working precision — the test asserts the gap it opens
    #: before relying on it. Only the jitter's presence is under test, not its size.
    VISIBLE_JITTER = 1e-2

    def test_the_noise_argument_reaches_the_inverse(self, n_out, exact_rtol):
        x, x_ = grids(n_out)

        jittered = reference_resampling(x, x_, VAR, CORR_LENGTH, self.VISIBLE_JITTER)
        unjittered = reference_resampling(x, x_, VAR, CORR_LENGTH, 0.0)
        tolerance = exact_rtol * float(jnp.max(jnp.abs(jittered)))
        gap = float(jnp.max(jnp.abs(jittered - unjittered)))
        # The teeth: matching one of the two references says nothing unless they are
        # further apart than the tolerance that matches them. Derived from that
        # tolerance rather than recorded, so it holds in either precision.
        assert gap > 100 * tolerance

        resample = resampling_kernel(x, x_, VAR, CORR_LENGTH, self.VISIBLE_JITTER)
        difference = float(jnp.max(jnp.abs(resample - jittered)))

        assert difference <= tolerance


class TestTheResampledMeanIsUnbiased:
    """What the spurious diagonal actually cost: the mean the resampling matrix carries."""

    def test_resampling_a_grid_onto_itself_is_the_identity(self, exact_rtol):
        """The sharpest case of the coincidence, and the one with an analytic answer.

        ``K (K + j I)^-1`` differs from the identity by ``-j (K + j I)^-1``, i.e. by at
        most ``j / (lam_min + j)`` — 7e-9 here. The diagonal on the cross term made it
        3.5e-4, five decades of bias sitting on an exactly known answer.

        The bound is analytic and computed from this run's own ``lam_min``, with a
        factor of four for the rounding of the inverse and the precision floor for
        fp32, where the jitter is below the working precision. It stays four decades
        under what the defect produced, so the headroom costs the test nothing.
        """
        x, _ = grids(N_NODES)

        resample = resampling_kernel(x, x, VAR, CORR_LENGTH, JITTER)
        lam_min = float(jnp.linalg.eigvalsh(base_kernel(x, x, VAR, CORR_LENGTH))[0])
        bound = 4 * JITTER / (lam_min + JITTER) + exact_rtol
        deviation = float(jnp.max(jnp.abs(resample - jnp.eye(N_NODES))))

        assert deviation <= bound

    def test_the_resampled_mean_is_the_conditional_mean(self, exact_rtol):
        """A function on the nodes, resampled at the length the coincidence needs."""
        x, x_ = grids(N_NODES)
        f = jnp.sin(2 * jnp.pi * x / 100.0) + 0.5

        mean = resampling_kernel(x, x_, VAR, CORR_LENGTH, JITTER) @ f
        reference = reference_resampling(x, x_, VAR, CORR_LENGTH, JITTER) @ f
        difference = float(jnp.max(jnp.abs(mean - reference)))

        assert difference <= exact_rtol * float(jnp.max(jnp.abs(f)))


class TestTheCoincidenceIsReachable:
    """It is not a contrived shape: an ordinary correlation time produces it.

    :func:`~tabascal.gp.get_times` lays down about two nodes per correlation length,
    so a correlation time of about twice the integration time makes the node grid
    exactly as long as the observation grid — and every resampling matrix built on it
    square. The grid below is the one a 1200 s, 240-sample observation gets, which is
    the shape ``gains:GPGains`` hands to :func:`~tabascal.gp.resampling_kernel`.
    """

    #: The correlation time whose node grid matches the observation grid, and its
    #: nearest neighbour that does not. Nothing physical distinguishes them.
    COINCIDENT = 10.1
    NEIGHBOUR = 10.0

    #: The jitter ``gains:GPGains`` builds a unit-variance kernel with, so the
    #: spurious diagonal was a hundred times the legitimate one where it fired.
    GAIN_JITTER = 1e-5

    def times(self):
        return jnp.linspace(0.0, 1200.0, 240)

    def test_a_short_correlation_time_matches_the_observation_grid(self):
        times = self.times()

        assert len(get_times(times, self.COINCIDENT)) == len(times)
        assert len(get_times(times, self.NEIGHBOUR)) != len(times)

    @pytest.mark.parametrize(
        "corr_time", [COINCIDENT, NEIGHBOUR], ids=["coincident", "neighbour"]
    )
    def test_the_conditional_mean_of_a_constant_is_unbiased(self, corr_time, exact_rtol):
        """The mean a constant node vector resamples to, against the jitter-free
        conditional it is defined by — both computed in this run, so nothing here
        depends on the machine.

        The node covariance at this grid is ill-conditioned (condition number ~2e8,
        two nodes per correlation length over 240 samples), which is exactly why the
        assertion is an identity against a reference rather than a bound on a
        magnitude: the magnitudes themselves are not reproducible across platforms.
        For scale only, and asserted nowhere: the distance from the constant was
        ~5e-3 to ~1.4e-2 before the fix depending on the platform, against ~2e-3 to
        ~5e-3 at the neighbouring node count.
        """
        times = self.times()
        g_times = get_times(times, corr_time)
        nodes = jnp.ones(len(g_times))

        mean = resampling_kernel(g_times, times, 1.0, corr_time, self.GAIN_JITTER) @ nodes
        reference = (
            reference_resampling(g_times, times, 1.0, corr_time, self.GAIN_JITTER) @ nodes
        )
        difference = float(jnp.max(jnp.abs(mean - reference)))

        assert difference <= exact_rtol * float(jnp.max(jnp.abs(reference)))
