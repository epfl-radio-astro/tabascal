"""tabascal.poly_interp: polynomial interpolation from the data grid to the integration grid.

The weights are a polynomial through the stencil values, so the checks are
against that definition -- the Lagrange basis on explicit nodes, exactness on
polynomials of the stencil's degree, unit weight on the coarse point -- and,
above the interpolating degree, against the prior the extra coefficients are
drawn from: the coefficient covariance is the Taylor polynomial of the prior
covariance, the interpolating degree gives the Lagrange basis back whatever
the spectrum, and the weights converge to :mod:`tabascal.gp_interp`'s as the
degree grows. The end of the file holds the two routes to a draw from the
prior.
"""

import warnings

import numpy as np
import pytest
import jax.numpy as jnp

from tabascal import gp_interp as gi
from tabascal import poly_interp as pi
from tabascal.fft_gp import latent_to_signal

from .test_gp_interp import spectrum


# ---------------------------------------------------------------------------
# The interpolating polynomial
# ---------------------------------------------------------------------------


def test_lagrange_weights_are_the_lagrange_basis():
    """Each column is the polynomial that is one at its node and zero at the others."""
    nodes = np.array([-1.0, 0.0, 1.0])
    points = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])

    W = pi.lagrange_weights(nodes, points)

    assert W.shape == (5, 3)
    for s in range(3):
        basis = np.polyfit(nodes, np.eye(3)[s], deg=2)
        assert np.allclose(W[:, s], np.polyval(basis, points), atol=1e-14)
    # The quadratic through the nodes, read at the points.
    assert np.allclose(W @ nodes**2, points**2, atol=1e-14)
    assert np.allclose(W @ np.ones(3), np.ones(5), atol=1e-14)


def test_lagrange_weights_refuse_repeated_nodes():
    with pytest.raises(ValueError, match="distinct"):
        pi.lagrange_weights([0.0, 1.0, 1.0], [0.5])
    with pytest.raises(ValueError, match="at least one"):
        pi.lagrange_weights([], [0.5])


@pytest.mark.parametrize("n_int_f, n_int_t, h", [(1, 5, 1), (2, 4, 1), (3, 3, 2), (1, 1, 1), (2, 6, 0)])
@pytest.mark.parametrize("degree", [None, "4h"])
def test_the_coarse_point_gets_unit_weight_in_every_cell(n_int_f, n_int_t, h, degree):
    """The centre sample is the coarse value, at either degree and in every cell."""
    ns, dxs = [4, 7], [1e6, 2.0]
    hs = [h if n_int_f > 1 else 0, h if n_int_t > 1 else 0]
    kwargs = {}
    if degree == "4h":
        pk, ks, _, _ = spectrum(ns, dxs, [4e6, 8.0])
        kwargs = dict(degree=4 * h, pk=pk, ks=ks, dxs=dxs)

    W = pi.polynomial_weights([n_int_f, n_int_t], hs, ns, **kwargs)

    offsets = gi.stencil_offsets(hs)
    centre = int(np.flatnonzero((offsets == 0).all(axis=1))[0])
    want = np.zeros(len(offsets))
    want[centre] = 1.0
    assert W.shape == (ns[0], ns[1], n_int_f, n_int_t, len(offsets))
    assert np.allclose(W[:, :, n_int_f // 2, n_int_t // 2], want, atol=1e-9)


def test_the_weights_reproduce_a_polynomial_of_the_stencils_degree():
    """A coarse grid that is a polynomial of degree ``2h`` per axis is interpolated exactly.

    In the interior, where every cell has its full stencil; an edge cell has
    fewer nodes and reproduces a lower degree -- the linear grid is exact
    everywhere. The fine grid is compared with the polynomial at the offsets
    :func:`gp_interp.fine_offsets` lays out, in the layout the kernels reshape.
    """
    ns, n_ints, hs = [6, 9], [2, 5], [1, 1]
    j = np.arange(ns[0])[:, None]
    k = np.arange(ns[1])[None, :]
    u_f = gi.fine_offsets(n_ints[0], 1.0)
    u_t = gi.fine_offsets(n_ints[1], 1.0)

    def fine_of(P, Q):
        jf = (j[:, :, None] + u_f[None, None, :]).reshape(-1, 1)
        kf = (k[:, :, None] + u_t[None, None, :]).reshape(1, -1)
        return P(jf) * Q(kf)

    def interpolated(coarse):
        W = pi.polynomial_weights(n_ints, hs, ns)
        stack = gi.stencil_stack(jnp.asarray(coarse), gi.stencil_offsets(hs), hs)
        return np.asarray(gi.interpolate_fine(stack, jnp.asarray(W)))

    quadratic = (lambda x: 1 + 0.3 * x - 0.2 * x**2, lambda y: 2 - 0.5 * y + 0.1 * y**2)
    got = interpolated(quadratic[0](j) * quadratic[1](k))
    want = fine_of(*quadratic)
    interior = (slice(n_ints[0], -n_ints[0]), slice(n_ints[1], -n_ints[1]))
    assert np.allclose(got[interior], want[interior], atol=1e-10)
    assert not np.allclose(got, want, atol=1e-6)  # the edge cells extrapolate linearly

    linear = (lambda x: 1 + 0.3 * x, lambda y: 2 - 0.5 * y)
    assert np.allclose(interpolated(linear[0](j) * linear[1](k)), fine_of(*linear), atol=1e-10)


def test_the_weights_are_lagrange_on_the_available_nodes():
    """Edge cells: the polynomial through whichever stencil points exist, zero on the rest."""
    W = pi.polynomial_weights([1, 4], [0, 2], [1, 8])
    u = gi.fine_offsets(4, 1.0)
    d = np.arange(-2, 3)

    assert np.allclose(W[0, 4, 0], pi.lagrange_weights(d, u))
    # Cell 1 lacks the offset -2.
    assert np.all(W[0, 1, 0][:, 0] == 0.0)
    assert np.allclose(W[0, 1, 0][:, 1:], pi.lagrange_weights(d[1:], u))
    # Cell 7 has only the offsets -2..0.
    assert np.all(W[0, 7, 0][:, 3:] == 0.0)
    assert np.allclose(W[0, 7, 0][:, :3], pi.lagrange_weights(d[:3], u))


# ---------------------------------------------------------------------------
# The prior above the interpolating degree
# ---------------------------------------------------------------------------


def test_the_coefficient_covariance_is_the_taylor_polynomial_of_the_prior_covariance():
    """``phi(u)^T cov phi(0)`` is the Taylor polynomial of ``K(u)``, closer with every degree.

    And a covariance: positive semi-definite, from the moments of a spectrum
    with power in more than one mode.
    """
    ns, dxs = [1, 16], [1e6, 2.0]
    pk, ks, _, _ = spectrum(ns, dxs, [1e9, 12.0])
    _, pk_t = gi.factorise_spectrum(pk)
    u = np.array([0.1, 0.25, 0.5])
    K_true = gi.prior_covariance_1d(pk_t, ks[1], u * dxs[1])

    errors = []
    for degree in (2, 4, 6, 8, 12):
        moments = pi.spectral_moments(pk_t, ks[1], dxs[1], 2 * degree)
        cov = pi.taylor_prior_covariance(moments, degree)
        assert cov.shape == (degree + 1, degree + 1)
        assert np.allclose(cov, cov.T)
        assert np.linalg.eigvalsh(cov).min() > -1e-12 * np.abs(cov).max()
        Phi_u = np.vander(u, degree + 1, increasing=True)
        Phi_0 = np.vander(np.zeros(1), degree + 1, increasing=True)
        errors.append(np.abs(Phi_u @ cov @ Phi_0.T - K_true[:, None]).max() / K_true.max())

    assert errors[0] < 1e-1
    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))
    assert errors[-1] < 1e-10


def test_the_moments_are_the_derivatives_of_the_covariance_at_zero_lag():
    """``M_0`` is the variance, ``M_2`` the variance of the derivative, in cell units."""
    pk_t = np.array([0.5, 0.3, 0.2])
    k_t = np.array([0.0, 0.1, -0.1])
    dx = 2.0

    M = pi.spectral_moments(pk_t, k_t, dx, 4)

    assert np.isclose(M[0], pk_t.sum())
    assert np.isclose(M[1], 2 * np.pi * dx * (0.3 * 0.1 - 0.2 * 0.1))
    assert np.isclose(M[2], (2 * np.pi * dx * 0.1) ** 2 * 0.5)
    with pytest.raises(ValueError, match="one k per spectrum entry"):
        pi.spectral_moments(pk_t, k_t[:2], dx, 2)
    with pytest.raises(ValueError, match="moments up to order 6"):
        pi.taylor_prior_covariance(M, 3)


def test_the_interpolating_degree_gives_the_lagrange_basis_whatever_the_prior():
    """With as many coefficients as nodes the prior drops out of the conditional mean."""
    nodes = np.array([-1.0, 0.0, 1.0])
    points = gi.fine_offsets(5, 1.0)
    rng = np.random.RandomState(0)
    A = rng.randn(3, 3)
    cov = A @ A.T  # any positive-definite prior on the three coefficients

    W, dropped = pi.taylor_prior_weights(nodes, points, cov)

    assert dropped == 0
    assert np.allclose(W, pi.lagrange_weights(nodes, points), atol=1e-10)


def test_the_interpolating_degree_is_the_default():
    ns, dxs = [3, 9], [1e6, 2.0]
    pk, ks, _, _ = spectrum(ns, dxs, [4e6, 8.0])

    plain = pi.polynomial_weights([2, 5], [1, 1], ns)
    at_degree = pi.polynomial_weights([2, 5], [1, 1], ns, degree=2, pk=pk, ks=ks, dxs=dxs)

    assert np.array_equal(plain, at_degree)


def test_the_weights_converge_to_the_conditional_mean_with_the_degree():
    """The Taylor covariance tends to the prior's, and the weights to ``gp_interp``'s with it."""
    ns, dxs = [1, 32], [1e6, 1.0]
    pk, ks, _, _ = spectrum(ns, dxs, [1e9, 7.5])

    for h in (1, 2):
        hs = [0, h]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            W_gp = gi.interpolation_weights(pk, ks, dxs, [1, 6], hs, ns)
        gaps = [
            np.abs(pi.polynomial_weights([1, 6], hs, ns, degree=degree, pk=pk, ks=ks, dxs=dxs) - W_gp).max()
            for degree in (2 * h, 2 * h + 8, 2 * h + 16)
        ]
        assert gaps[0] > 1e-3
        assert gaps[0] > gaps[1] > gaps[2]
        assert gaps[2] < 1e-6


def test_a_degenerate_prior_averages_the_stencil_and_warns():
    """A single kept mode: every coarse value the same draw, and the polynomial prior rank one.

    The conditional mean is the mean of whichever stencil points exist, as
    :func:`gp_interp.interpolation_weights` gives; the plain polynomial through
    equal values is that constant too, without a warning to give.
    """
    pk = np.array([[1.0]])
    ks = [np.array([0.0]), np.array([0.0])]

    with pytest.warns(RuntimeWarning, match="degenerate"):
        W = pi.polynomial_weights([1, 3], [0, 1], [2, 4], degree=3, pk=pk, ks=ks, dxs=[1.0, 1.0])

    assert np.allclose(W[0, 1], np.full(3, 1 / 3))
    assert np.allclose(W[0, 0], [0.0, 0.5, 0.5])
    assert np.allclose(W[0, 3], [0.5, 0.5, 0.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plain = pi.polynomial_weights([1, 3], [0, 1], [2, 4])
    assert np.allclose(plain.sum(axis=-1), 1.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_a_degree_below_the_stencil_is_refused():
    with pytest.raises(ValueError, match="at least 4"):
        pi.polynomial_weights([1, 3], [0, 2], [1, 8], degree=3)


def test_a_degree_above_the_stencil_needs_the_spectrum():
    with pytest.raises(ValueError, match="needs the spectrum"):
        pi.polynomial_weights([1, 3], [0, 1], [1, 8], degree=3)
    # Not on an axis with no stencil: nothing to fix there.
    W = pi.polynomial_weights([1, 1], [0, 0], [1, 8], degree=3)
    assert np.all(W == 1.0)


def test_a_spectrum_that_is_not_a_product_is_refused():
    pk, ks, _, _ = spectrum([4, 6], [1e6, 2.0], [2e6, 4.0])
    bumped = pk * (1 + 1e-3 * np.random.RandomState(0).rand(*pk.shape))

    with pytest.raises(ValueError, match="does not factorise"):
        pi.polynomial_weights([2, 3], [1, 1], [4, 6], degree=3, pk=bumped, ks=ks, dxs=[1e6, 2.0])


@pytest.mark.parametrize("hs, ns", [([-1, 1], [2, 2]), ([1, 1], [0, 2])])
def test_the_geometry_is_validated(hs, ns):
    with pytest.raises(ValueError):
        pi.polynomial_weights([2, 2], hs, ns)


# ---------------------------------------------------------------------------
# On a draw from the prior
# ---------------------------------------------------------------------------


def test_a_prior_draw_is_interpolated_to_within_its_local_scatter():
    """The polynomial against the supersampled grid, beside the conditional mean.

    The same draw as the ``gp_interp`` test of this name. The plain polynomial
    is within a factor of two of the conditional mean's error -- the conditional
    mean is the closer of the two on average over draws, by ten to thirty
    percent, but on one draw either can be by more: here the quadratic is the
    closer and the quartic the further -- and the 5-point stencil a few times
    closer than the 3-point, as there. With the
    prior's coefficients at twice the interpolating degree the polynomial is the
    conditional mean to within a few percent of that error.
    """
    ns, dxs, n_int = [1, 24], [1e6, 1.0], [1, 6]
    corr = [1e9, 8.0]
    pk, ks, pads_c, idxs_c = spectrum(ns, dxs, corr, gammas=(3, 3), ss=(1, 1))
    _, _, pads_f, idxs_f = spectrum(ns, dxs, corr, gammas=(3, 3), ss=n_int)

    rng = np.random.RandomState(1)
    latent = np.sqrt(pk) * (rng.randn(*pk.shape) + 1j * rng.randn(*pk.shape))
    coarse = np.asarray(latent_to_signal(jnp.asarray(latent), pads_c, idxs_c))
    fine = np.asarray(latent_to_signal(jnp.asarray(latent), pads_f, idxs_f))

    def error(W, h):
        stack = gi.stencil_stack(jnp.asarray(coarse), gi.stencil_offsets([0, h]), [0, h])
        got = np.asarray(gi.interpolate_fine(stack, jnp.asarray(W)))
        return np.sqrt(np.mean(np.abs(got - fine) ** 2) / np.mean(np.abs(fine) ** 2))

    def errors(h):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = error(gi.interpolation_weights(pk, ks, dxs, n_int, [0, h], ns), h)
        plain = error(pi.polynomial_weights(n_int, [0, h], ns), h)
        prior = error(
            pi.polynomial_weights(n_int, [0, h], ns, degree=4 * h, pk=pk, ks=ks, dxs=dxs), h
        )
        return gp, plain, prior

    gp_1, plain_1, prior_1 = errors(1)
    gp_2, plain_2, prior_2 = errors(2)

    assert plain_1 < 1e-2
    assert plain_2 < plain_1 / 2
    assert plain_1 < 2 * gp_1 and gp_1 < 2 * plain_1
    assert plain_2 < 2 * gp_2 and gp_2 < 2 * plain_2
    assert prior_1 < 1.05 * gp_1
    assert prior_2 < 1.05 * gp_2
