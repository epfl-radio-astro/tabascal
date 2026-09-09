"""tabascal.gp_interp: GP interpolation from the data grid to the integration grid.

The weights are the conditional mean of the signal's own prior given a block of
coarse values around each cell, so the checks here are against that definition
-- a direct conditioning written out on explicit coordinates -- and against the
grid conventions the rest of tabascal builds: the fine offsets ``TabConfig``
lays out, the coarse point every cell carries at its centre, and the layout the
visibility kernels reshape. The end of the file pins the time-blocked
visibility to the unblocked reduction, in value and gradient.
"""

import warnings

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from tabascal import gp_interp as gi
from tabascal.fft_gp import (
    domain_ss,
    knee_from_corr_scale,
    latent_to_signal,
    latent_to_signal_init,
)
from tabascal.interferometry import calculate_rfi_vis_fine

from .components.conftest import active_precision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def spectrum(ns, dxs, corr, gammas=(3, 3), cutoff=1e-9, pad=2.0, ss=(1, 1)):
    """``(pk, ks, pads, ss_idxs)`` of the RFI prior on a grid, as the components build it.

    The same call ``ComplexRFIVarAnt._compute_gp_params`` makes, with the
    supersampling factors as an argument so a fine-grid transform of the same
    latent can be built beside the coarse one.
    """
    k0s = knee_from_corr_scale(list(corr))
    pk, ks, pads, ss_idxs = latent_to_signal_init(
        list(ns), list(dxs), [pad, pad], list(ss), 1.0, k0s, list(gammas), cutoff
    )
    return np.asarray(pk, np.float64), [np.asarray(k, np.float64) for k in ks], pads, ss_idxs


def direct_covariance(pk, ks, x, y):
    """``K(x, y)`` on explicit ``(n, 2)`` coordinates, summed mode by mode.

    Independent of :func:`gp_interp.prior_covariance`: no lag bookkeeping, just
    the definition, so the two can be held to each other.
    """
    lag = x[:, None, :] - y[None, :, :]  # (nx, ny, 2)
    arg = 2 * np.pi * (
        ks[0][None, None, :, None] * lag[:, :, 0, None, None]
        + ks[1][None, None, None, :] * lag[:, :, 1, None, None]
    )
    return np.sum(pk[None, None] * np.cos(arg), axis=(2, 3))


def tols():
    return (1e-10, 1e-10) if active_precision() == "double" else (1e-4, 1e-4)


# ---------------------------------------------------------------------------
# Grid conventions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_int", [1, 2, 3, 4, 5, 8])
def test_the_fine_offsets_are_the_fine_grid_tabconfig_builds(n_int):
    """Every cell's fine samples sit where ``TabConfig._set_freqs_times`` puts them.

    That grid is ``domain_ss`` on a unit grid, scaled; the RFI phase is
    evaluated on it, so the interpolation has to land on the same samples --
    including the asymmetric placement of an even ``n_int``.
    """
    n, dx = 6, 2.0
    fine = np.asarray(domain_ss([n], [1.0], [0.0], [n_int], [2.0])[0], np.float64) * dx
    centres = np.arange(n) * dx

    offsets = fine.reshape(n, n_int) - centres[:, None]

    assert np.allclose(offsets, gi.fine_offsets(n_int, dx)[None, :], atol=1e-5)
    # And the centre sample is the coarse point, exactly in exact arithmetic.
    assert gi.fine_offsets(n_int, dx)[n_int // 2] == 0.0


def test_stencil_offsets_are_lexicographic_with_frequency_slowest():
    offsets = gi.stencil_offsets([1, 1])

    assert offsets.shape == (9, 2)
    assert offsets.tolist() == [
        [-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]
    ]
    assert gi.stencil_offsets([0, 2]).tolist() == [[0, -2], [0, -1], [0, 0], [0, 1], [0, 2]]


def test_stencil_availability_names_the_edge_cells():
    patterns, case = gi.stencil_availability(5, 1)

    assert patterns.tolist() == [[False, True, True], [True, True, False], [True, True, True]]
    assert [patterns[c].tolist() for c in case] == [
        [False, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, False],
    ]
    # An axis shorter than the stencil: the one cell lacks both neighbours.
    patterns, case = gi.stencil_availability(1, 1)
    assert patterns.tolist() == [[False, True, False]] and case.tolist() == [0]
    # No stencil at all: every cell has its own point and nothing else to lack.
    patterns, case = gi.stencil_availability(4, 0)
    assert patterns.tolist() == [[True]] and case.tolist() == [0, 0, 0, 0]


# ---------------------------------------------------------------------------
# The weights
# ---------------------------------------------------------------------------


def test_the_covariance_matches_its_definition():
    """``prior_covariance`` on lags equals the mode-by-mode sum on coordinates."""
    pk, ks, _, _ = spectrum([4, 8], [1e6, 2.0], [3e6, 6.0])
    lags_f = np.array([-1e6, 0.0, 0.5e6, 2e6])
    lags_t = np.array([-3.0, -0.4, 0.0, 2.0, 7.0])

    got = gi.prior_covariance(pk, ks, [lags_f, lags_t])

    x = np.stack(np.meshgrid(lags_f, lags_t, indexing="ij"), axis=-1).reshape(-1, 2)
    want = direct_covariance(pk, ks, x, np.zeros((1, 2)))[:, 0].reshape(len(lags_f), len(lags_t))
    assert np.allclose(got, want, rtol=1e-10, atol=1e-12 * np.abs(want).max())


@pytest.mark.parametrize("n_int_f, n_int_t, h", [(1, 5, 1), (2, 4, 1), (3, 3, 2), (1, 1, 1), (2, 6, 0)])
def test_the_coarse_point_gets_unit_weight_in_every_cell(n_int_f, n_int_t, h):
    """The centre sample is the coarse value: weight one on it, zero elsewhere.

    In every cell, edges included -- a cell may lack neighbours but never its own
    point -- and to round-off, since the fine point coincides with a stencil
    point and the conditional mean reproduces its data exactly.
    """
    ns, dxs = [4, 7], [1e6, 2.0]
    pk, ks, _, _ = spectrum(ns, dxs, [4e6, 8.0])
    hs = [h if n_int_f > 1 else 0, h if n_int_t > 1 else 0]

    W = gi.interpolation_weights(pk, ks, dxs, [n_int_f, n_int_t], hs, ns)

    offsets = gi.stencil_offsets(hs)
    centre = int(np.flatnonzero((offsets == 0).all(axis=1))[0])
    want = np.zeros(len(offsets))
    want[centre] = 1.0
    assert W.shape == (ns[0], ns[1], n_int_f, n_int_t, len(offsets))
    assert np.allclose(W[:, :, n_int_f // 2, n_int_t // 2], want, atol=1e-9)


@pytest.mark.parametrize("cell", [(0, 0), (0, 3), (2, 0), (2, 3), (3, 6), (1, 6)])
def test_the_weights_are_the_direct_conditioning_of_the_prior(cell):
    """Each cell's weights are ``K_*S K_SS^-1`` written out on explicit coordinates.

    Corner, edge and interior cells: the reference conditions on whichever
    stencil points lie inside the grid, which is what the availability patterns
    have to reproduce, and puts the fine points where ``fine_offsets`` says.
    """
    ns, dxs = [4, 7], [1e6, 2.0]
    n_ints, hs = [2, 3], [1, 1]
    # A knee near the cell width keeps the direct solve well conditioned, so the
    # comparison is a check on the bookkeeping rather than on round-off.
    pk, ks, _, _ = spectrum(ns, dxs, [1.5e6, 3.0])

    W = gi.interpolation_weights(pk, ks, dxs, n_ints, hs, ns)

    j, k = cell
    offsets = gi.stencil_offsets(hs)
    inside = [
        s
        for s, (d_f, d_t) in enumerate(offsets)
        if 0 <= j + d_f < ns[0] and 0 <= k + d_t < ns[1]
    ]
    x_S = np.array([[(j + offsets[s, 0]) * dxs[0], (k + offsets[s, 1]) * dxs[1]] for s in inside])
    of_f, of_t = gi.fine_offsets(n_ints[0], dxs[0]), gi.fine_offsets(n_ints[1], dxs[1])
    x_star = np.array(
        [[j * dxs[0] + of_f[p], k * dxs[1] + of_t[q]] for p in range(n_ints[0]) for q in range(n_ints[1])]
    )
    want = direct_covariance(pk, ks, x_star, x_S) @ np.linalg.inv(direct_covariance(pk, ks, x_S, x_S))

    got = W[j, k].reshape(-1, len(offsets))
    assert np.allclose(got[:, inside], want, atol=1e-8)
    outside = [s for s in range(len(offsets)) if s not in inside]
    assert np.all(got[:, outside] == 0.0)


def test_the_factorised_and_joint_solves_agree(monkeypatch):
    """The per-axis solve is the joint one, for a spectrum that is a product.

    The joint path is what a spectrum that does not factorise gets, so the two
    are held to each other on one that does.
    """
    ns, dxs = [5, 6], [1e6, 2.0]
    pk, ks, _, _ = spectrum(ns, dxs, [2e6, 4.0])
    assert gi.factorise_spectrum(pk) is not None

    separable = gi.interpolation_weights(pk, ks, dxs, [2, 4], [1, 1], ns)
    monkeypatch.setattr(gi, "factorise_spectrum", lambda pk, rtol=1e-5: None)
    joint = gi.interpolation_weights(pk, ks, dxs, [2, 4], [1, 1], ns)

    assert np.allclose(separable, joint, atol=1e-7)


def test_a_spectrum_that_is_not_a_product_is_seen_as_such():
    pk, _, _, _ = spectrum([4, 6], [1e6, 2.0], [2e6, 4.0])
    assert gi.factorise_spectrum(pk) is not None

    bumped = pk * (1 + 1e-3 * np.random.RandomState(0).rand(*pk.shape))
    assert gi.factorise_spectrum(bumped) is None


def test_a_degenerate_prior_averages_the_stencil_and_warns():
    """A single kept mode makes every coarse value the same draw.

    The stencil covariance then has rank one, and the conditional mean of that
    degenerate Gaussian is the mean of whichever stencil points exist -- which
    the pseudo-inverse gives, where a solve would fail.
    """
    pk = np.array([[1.0]])
    ks = [np.array([0.0]), np.array([0.0])]

    with pytest.warns(RuntimeWarning, match="degenerate"):
        W = gi.interpolation_weights(pk, ks, [1.0, 1.0], [1, 3], [0, 1], [2, 4])

    # Interior time cells: a third on each of three neighbours.
    assert np.allclose(W[0, 1], np.full(3, 1 / 3))
    # Edge cells: a half on each of the two that exist, nothing on the missing one.
    assert np.allclose(W[0, 0], [0.0, 0.5, 0.5])
    assert np.allclose(W[0, 3], [0.5, 0.5, 0.0])


def test_a_prior_draw_is_interpolated_to_within_its_local_scatter():
    """On a draw from the prior, the interpolation from the coarse points is close
    to the supersampled grid, and closer with more neighbours.

    Not exact -- three coarse values do not determine the process between them
    -- but at a correlation scale of eight cells the 3-point stencil is within a
    percent and the 5-point stencil a few times better, while a stencil of one
    point (the cell held at its centre value) is worse than either.
    """
    ns, dxs, n_int = [1, 24], [1e6, 1.0], [1, 6]
    corr = [1e9, 8.0]
    pk, ks, pads_c, idxs_c = spectrum(ns, dxs, corr, gammas=(3, 3), ss=(1, 1))
    _, _, pads_f, idxs_f = spectrum(ns, dxs, corr, gammas=(3, 3), ss=n_int)

    rng = np.random.RandomState(1)
    latent = np.sqrt(pk) * (rng.randn(*pk.shape) + 1j * rng.randn(*pk.shape))
    coarse = np.asarray(latent_to_signal(jnp.asarray(latent), pads_c, idxs_c))
    fine = np.asarray(latent_to_signal(jnp.asarray(latent), pads_f, idxs_f))
    assert coarse.shape == tuple(ns) and fine.shape == (ns[0], ns[1] * n_int[1])
    # The coarse grid is the fine one at the cell centres.
    assert np.allclose(coarse, fine[:, n_int[1] // 2 :: n_int[1]], atol=1e-6 * np.abs(fine).max())

    def error(h):
        W = gi.interpolation_weights(pk, ks, dxs, n_int, [0, h], ns)
        stack = gi.stencil_stack(jnp.asarray(coarse), gi.stencil_offsets([0, h]), [0, h])
        got = np.asarray(gi.interpolate_fine(stack, jnp.asarray(W)))
        return np.sqrt(np.mean(np.abs(got - fine) ** 2) / np.mean(np.abs(fine) ** 2))

    e0, e1, e2 = error(0), error(1), error(2)
    assert e1 < 1e-2
    assert e2 < e1 / 2 < e0 / 4


# ---------------------------------------------------------------------------
# Applying the weights
# ---------------------------------------------------------------------------


def test_interpolate_fine_has_the_layout_the_kernels_reshape():
    """``fine[..., j * n_int_f + p, k * n_int_t + q]`` is the weighted stencil sum.

    Written out as loops on a small grid, with random weights: the layout is a
    property of the application, not of what the weights are.
    """
    n_f, n_t, n_int_f, n_int_t, hs = 3, 4, 2, 3, [1, 1]
    rng = np.random.RandomState(2)
    A = rng.randn(2, n_f, n_t) + 1j * rng.randn(2, n_f, n_t)
    offsets = gi.stencil_offsets(hs)
    W = rng.randn(n_f, n_t, n_int_f, n_int_t, len(offsets))

    got = np.asarray(gi.interpolate_fine(gi.stencil_stack(jnp.asarray(A), offsets, hs), jnp.asarray(W)))

    assert got.shape == (2, n_f * n_int_f, n_t * n_int_t)
    want = np.zeros_like(got)
    for j in range(n_f):
        for k in range(n_t):
            for p in range(n_int_f):
                for q in range(n_int_t):
                    for s, (d_f, d_t) in enumerate(offsets):
                        if 0 <= j + d_f < n_f and 0 <= k + d_t < n_t:
                            want[:, j * n_int_f + p, k * n_int_t + q] += W[j, k, p, q, s] * A[:, j + d_f, k + d_t]
    atol = 1e-12 if active_precision() == "double" else 1e-4
    assert np.allclose(got, want, atol=atol * np.abs(want).max())


@pytest.mark.parametrize("time_block", [None, 1, 3, 5, 16, 100])
def test_the_time_block_changes_neither_the_value_nor_the_gradient(time_block):
    """``gp_interp_rfi_vis`` at any block size is the unblocked reduction.

    The reference is the fine-grid formula on the interpolated grid, not the
    function at another setting, so a defect every block shares would still
    show. Sizes that divide the axis, that leave a tail, and that exceed it.
    """
    n_rfi, n_ant, n_f, n_t, n_int_f, n_int_t, hs = 2, 4, 2, 16, 2, 3, [1, 1]
    a1, a2 = (a.astype("int32") for a in jnp.triu_indices(n_ant, 1))
    real = jnp.float64 if active_precision() == "double" else jnp.float32
    cplx = jnp.complex128 if real == jnp.float64 else jnp.complex64
    keys = jax.random.split(jax.random.PRNGKey(3), 5)
    A = (jax.random.normal(keys[0], (n_rfi, n_ant, n_f, n_t)) + 1j * jax.random.normal(keys[1], (n_rfi, n_ant, n_f, n_t))).astype(cplx)
    phase = jax.random.uniform(keys[2], (n_rfi, n_ant, n_f * n_int_f, n_t * n_int_t), maxval=2 * np.pi).astype(real)
    offsets = gi.stencil_offsets(hs)
    W = jax.random.normal(keys[3], (n_f, n_t, n_int_f, n_int_t, len(offsets))).astype(real)
    cotangent = (jax.random.normal(keys[4], (a1.shape[0], n_f, n_t)) + 0j).astype(cplx)

    def blocked(A, phase):
        return gi.gp_interp_rfi_vis(A, phase, W, offsets, hs, a1, a2, n_int_f, n_int_t, 3, time_block)

    def dense(A, phase):
        fine = gi.interpolate_fine(gi.stencil_stack(A, offsets, hs), W)
        vis = calculate_rfi_vis_fine(fine, phase, a1, a2)
        return jnp.mean(jnp.reshape(vis, (a1.shape[0], n_f, n_int_f, n_t, n_int_t)), axis=(2, 4))

    def value_and_grads(f):
        primal, vjp = jax.vjp(f, A, phase)
        return primal, vjp(cotangent)

    vis, (g_A, g_phase) = value_and_grads(blocked)
    ref, (r_A, r_phase) = value_and_grads(dense)

    atol, rtol = tols()
    assert vis.shape == (a1.shape[0], n_f, n_t) and vis.dtype == ref.dtype
    assert jnp.allclose(vis, ref, atol=atol * jnp.abs(ref).max(), rtol=rtol)
    assert jnp.allclose(g_A, r_A, atol=atol * jnp.abs(r_A).max(), rtol=rtol)
    assert jnp.allclose(g_phase, r_phase, atol=atol * jnp.abs(r_phase).max(), rtol=rtol)
    # And under jit, which is how the model runs it.
    assert jnp.allclose(jax.jit(blocked)(A, phase), vis, atol=atol * jnp.abs(ref).max(), rtol=rtol)
