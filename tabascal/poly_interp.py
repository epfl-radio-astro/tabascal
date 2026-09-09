"""Polynomial interpolation of a data-grid RFI signal onto the integration grid.

The counterpart of :mod:`tabascal.gp_interp` for ``rfi_vis:PolyInterpVis``: the
fine samples of a cell are read off a polynomial through the block of coarse
values around it, rather than off the conditional mean of the prior given
them. The geometry is the same -- the stencil of ``2h + 1`` coarse values per
axis, the fine offsets ``TabConfig`` lays out, the availability patterns at
the edges -- and so is everything downstream: the weights come out in the
array :func:`tabascal.gp_interp.interpolate_fine` applies, and the visibility
is the same blocked Riemann sum.

Two things are different. The interpolating polynomial through ``2h + 1``
values is unique -- degree ``2h`` per axis, the quadratic through a cell and
its two neighbours at the default stencil -- so its weights need no solve and
nothing about the prior: they are the Lagrange basis at the fine offsets, in
cell units, and they are as well conditioned for a smooth prior as for a rough
one, where the conditional mean's covariance solve loses a digit for every
power of the cell width over the correlation scale. That is the default. And it
is what the conditional mean tends to as the prior grows smooth on the scale
of the stencil -- the flat limit of a kernel interpolant is the polynomial one
-- so on a prior draw the two differ by a fraction of the interpolation error
itself; measured at a correlation time of seven or eight cells, the polynomial
is ten to thirty percent further from the supersampled grid than the
conditional mean on average over draws, and on a given draw either can be the
closer.

The prior can be brought back in above that degree. A polynomial of degree
``D > 2h`` through ``2h + 1`` values is underdetermined by ``D - 2h``
coefficients, and the process's own prior fixes them: written as a Taylor
expansion about the cell centre, the coefficients are its derivatives there,
whose covariance is the even derivatives of the prior covariance at zero lag --
the spectral moments of the spectrum the signal component samples. Conditioning
that Gaussian on the stencil values gives an interpolant that still passes
through them, and whose extra terms are the prior's expectation of them: the
slope through a cell's neighbours carries a cubic term of ``-omega^2`` times
the slope, as any band-limited function's does. It is the conditional mean of
the prior with its covariance replaced by that covariance's Taylor polynomial,
which is a covariance in its own right, and the two conditional means converge
as the degree grows. Measured at the default stencil, degree 3 already
reproduces the conditional mean's accuracy and degree 4 its weights to
``1e-4``; the 5-point stencil, whose outer lags the truncated series has to
reach, takes degree 8 and is no better than the plain polynomial below it.
Since the truncated covariance approaches the true one, so does the
conditioning of the solve -- the plain polynomial is the one that avoids it.

Units
-----
Offsets are in cells: a stencil point at ``d`` cells and a fine sample at
``(q - n_int // 2) / n_int``. The Lagrange weights are then dimensionless and
the same on every axis, and the moments are of ``2 pi k dx``, cycles per cell
scaled by ``2 pi``, so the Taylor coefficients of ``u^m`` are the derivatives
``f^(m) dx^m / m!``.
"""

from __future__ import annotations

from math import factorial
from typing import Optional, Sequence, Tuple

import numpy as np

from tabascal.gp_interp import (
    RCOND,
    _solve_weights,
    _warn_if_degenerate,
    assemble_axis_weights,
    factorise_spectrum,
    fine_offsets,
    stencil_availability,
)


# ---------------------------------------------------------------------------
# One axis: the interpolating polynomial
# ---------------------------------------------------------------------------


def lagrange_weights(nodes, points) -> np.ndarray:
    """``(n_points, n_nodes)``: the polynomial through values at ``nodes``, read at ``points``.

    The Lagrange form, weight ``s`` being ``prod_{r != s} (u - u_r) / (u_s -
    u_r)``: exact on any polynomial of degree below the number of nodes, one at
    a point that is a node and zero on the other nodes there, and formed by
    products alone. The nodes must be distinct.
    """
    nodes = np.asarray(nodes, dtype=np.float64).ravel()
    points = np.asarray(points, dtype=np.float64).ravel()
    if nodes.size == 0:
        raise ValueError("lagrange_weights needs at least one node.")
    if np.unique(nodes).size != nodes.size:
        raise ValueError(f"lagrange_weights needs distinct nodes, got {nodes.tolist()}.")
    W = np.ones((points.size, nodes.size), dtype=np.float64)
    for s, u_s in enumerate(nodes):
        for r, u_r in enumerate(nodes):
            if r != s:
                W[:, s] *= (points - u_r) / (u_s - u_r)
    return W


# ---------------------------------------------------------------------------
# One axis: the prior on the coefficients above the interpolating degree
# ---------------------------------------------------------------------------


def spectral_moments(pk_axis, k_axis, dx: float, max_order: int) -> np.ndarray:
    """``M_m = sum_k pk[k] (2 pi k dx)^m`` for ``m = 0..max_order``, one axis of the spectrum.

    The moments of the sampled spectrum in cell units. The prior covariance on
    the axis is ``K(r) = sum_k pk[k] cos(2 pi k r)`` (see
    :func:`tabascal.gp_interp.prior_covariance_1d`), so its derivatives at zero
    lag, in cells, are ``K^(2m)(0) = (-1)^m M_2m`` with the odd ones zero.
    """
    pk_axis = np.asarray(pk_axis, dtype=np.float64).ravel()
    w = 2 * np.pi * np.asarray(k_axis, dtype=np.float64).ravel() * float(dx)
    if pk_axis.shape != w.shape:
        raise ValueError(
            f"spectral_moments needs one k per spectrum entry, got {pk_axis.size} entries "
            f"and {w.size} modes."
        )
    return np.array([np.sum(pk_axis * w**m) for m in range(int(max_order) + 1)])


def taylor_prior_covariance(moments, degree: int) -> np.ndarray:
    """Prior covariance of the Taylor coefficients of the process at a cell centre.

    ``theta_m = f^(m)(0) dx^m / m!``, the coefficient of ``u^m`` for ``u`` in
    cells, for ``m = 0..degree``. For a stationary prior ``Cov(f^(i)(0),
    f^(j)(0)) = (-1)^j K^(i+j)(0)``, which :func:`spectral_moments` gives as
    ``(-1)^j (-1)^((i+j)/2) M_(i+j)`` for ``i + j`` even and zero otherwise, so
    ``moments`` has to reach order ``2 * degree``. The result is a covariance
    -- that of real random variables -- and therefore positive semi-definite;
    the polynomial kernel ``phi(u)^T cov phi(u')`` it defines is the Taylor
    polynomial of the prior covariance in ``u`` and ``u'``.
    """
    degree = int(degree)
    moments = np.asarray(moments, dtype=np.float64).ravel()
    if degree < 0:
        raise ValueError(f"degree must be non-negative, got {degree}.")
    if moments.size < 2 * degree + 1:
        raise ValueError(
            f"taylor_prior_covariance needs moments up to order {2 * degree} for degree "
            f"{degree}, got {moments.size - 1}."
        )
    cov = np.zeros((degree + 1, degree + 1), dtype=np.float64)
    for i in range(degree + 1):
        for j in range(i, degree + 1):
            if (i + j) % 2 == 0:
                cov[i, j] = cov[j, i] = (
                    (-1) ** j * (-1) ** ((i + j) // 2) * moments[i + j] / (factorial(i) * factorial(j))
                )
    return cov


def taylor_prior_weights(nodes, points, cov, rcond: float = RCOND) -> Tuple[np.ndarray, int]:
    """``(weights, dropped)``: the conditional mean under the polynomial prior.

    ``K_*S K_SS^+`` for ``K = Phi cov Phi^T`` on the ``nodes`` and ``points``,
    ``Phi`` being the powers ``u^0..u^D``: a polynomial of degree ``D`` through
    the values at the nodes, when ``D + 1`` is at least their number, whose
    coefficients beyond the interpolating ones are their prior expectation given
    the values. ``weights`` is ``(n_points, n_nodes)``. The pseudo-inverse is
    :func:`tabascal.gp_interp._solve_weights`'s, and ``dropped`` counts the
    directions it left out -- a prior with power in a single mode, say, whose
    ``cov`` is rank one.
    """
    cov = np.asarray(cov, dtype=np.float64)
    nodes = np.asarray(nodes, dtype=np.float64).ravel()
    points = np.asarray(points, dtype=np.float64).ravel()
    n_basis = cov.shape[0]
    Phi_S = np.vander(nodes, n_basis, increasing=True)
    Phi_p = np.vander(points, n_basis, increasing=True)
    K_SS = Phi_S @ cov @ Phi_S.T
    K_pS = Phi_p @ cov @ Phi_S.T
    return _solve_weights(K_SS, K_pS, np.arange(nodes.size), rcond)


# ---------------------------------------------------------------------------
# The weights on the data grid
# ---------------------------------------------------------------------------


def _axis_weights(
    n_int: int, h: int, n: int, degree: Optional[int], cov: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, int]:
    """One axis: ``(table, case, dropped)`` as :func:`tabascal.gp_interp._axis_weights` returns them.

    ``table[c, q, d + h]`` is the weight of fine sample ``q`` of a cell with
    availability pattern ``c`` on the coarse value at offset ``d``. The
    interpolating polynomial through whichever stencil points the pattern has,
    or, with ``cov`` -- the prior on the coefficients up to ``degree`` -- the
    conditional mean under it.
    """
    d = np.arange(-h, h + 1)
    u = fine_offsets(n_int, 1.0)
    patterns, case = stencil_availability(n, h)
    table = np.zeros((len(patterns), n_int, 2 * h + 1), dtype=np.float64)
    dropped = 0
    for c, pattern in enumerate(patterns):
        keep = np.flatnonzero(pattern)
        if cov is None:
            table[c][:, keep] = lagrange_weights(d[keep], u)
        else:
            table[c][:, keep], n_dropped = taylor_prior_weights(d[keep], u, cov)
            dropped = max(dropped, n_dropped)
    return table, case, dropped


def polynomial_weights(
    n_ints: Sequence[int],
    hs: Sequence[int],
    ns: Sequence[int],
    degree: Optional[int] = None,
    pk=None,
    ks: Optional[Sequence] = None,
    dxs: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Polynomial weights of every cell's fine samples on its coarse stencil.

    The array :func:`tabascal.gp_interp.interpolation_weights` returns, with the
    same meaning -- ``W[j, k, p, q, s]`` weights the coarse value at stencil
    offset ``s`` in fine sample ``(p, q)`` of cell ``(j, k)``, zero on a stencil
    point beyond the grid -- from a polynomial per axis rather than the
    conditional mean. Separable by construction: the weight on a stencil point
    is the product of the two axes' weights, as the conditional mean's is for
    the product spectra the signal components sample.

    Parameters
    ----------
    n_ints : (n_int_freq, n_int_time)
        Fine samples per cell on each axis.
    hs : (h_freq, h_time)
        Stencil half-widths on each axis.
    ns : (n_freq, n_time)
        The data-grid size, for the edge cases.
    degree
        ``None`` for the interpolating polynomial, of degree ``2h`` on each axis;
        otherwise the degree on both axes, at least ``2h`` on every axis with a
        stencil. Above ``2h`` the extra coefficients are the prior's expectation
        given the stencil values (see the module notes), which needs the
        spectrum.
    pk, ks, dxs
        The spectrum and k-grid of the sampled prior, as
        :func:`tabascal.fft_gp.latent_to_signal_init` returns them, and the
        data-grid cell widths in the units ``ks`` are conjugate to. Needed only
        above the interpolating degree, and the spectrum has to be a product
        over the axes, as every one the signal components sample is.

    Returns
    -------
    ndarray (n_freq, n_time, n_int_freq, n_int_time, n_stencil), float64
    """
    n_int_f, n_int_t = (int(n) for n in n_ints)
    h_f, h_t = (int(h) for h in hs)
    n_f, n_t = (int(n) for n in ns)
    if n_f < 1 or n_t < 1:
        raise ValueError(f"The data grid must have at least one cell per axis, got {list(ns)}.")
    if h_f < 0 or h_t < 0:
        raise ValueError(f"Stencil half-widths must be non-negative, got {list(hs)}.")

    covs = [None, None]
    if degree is not None:
        degree = int(degree)
        needed = max(2 * h_f, 2 * h_t)
        if degree < needed:
            raise ValueError(
                f"A polynomial of degree {degree} cannot pass through the {needed + 1} "
                f"stencil values on an axis with half-width {needed // 2}: the degree must "
                f"be at least {needed} for this stencil, or None for the interpolating "
                "polynomial."
            )
        extra = [degree > 2 * h and h > 0 for h in (h_f, h_t)]
        if any(extra):
            if pk is None or ks is None or dxs is None:
                raise ValueError(
                    f"A polynomial of degree {degree} through {needed + 1} stencil values "
                    "takes its extra coefficients from the prior, so it needs the "
                    "spectrum (pk, ks) and the cell widths (dxs)."
                )
            factors = factorise_spectrum(pk)
            if factors is None:
                raise ValueError(
                    "The spectrum does not factorise over the axes. The polynomial prior "
                    "is built one axis at a time, from the moments of each axis's "
                    "spectrum; every spectrum the signal components sample is a product."
                )
            for axis, (pk_axis, k_axis, dx) in enumerate(zip(factors, ks, dxs)):
                if extra[axis]:
                    moments = spectral_moments(pk_axis, k_axis, float(dx), 2 * degree)
                    covs[axis] = taylor_prior_covariance(moments, degree)

    table_f, case_f, dropped_f = _axis_weights(n_int_f, h_f, n_f, degree, covs[0])
    table_t, case_t, dropped_t = _axis_weights(n_int_t, h_t, n_t, degree, covs[1])
    _warn_if_degenerate(dropped_f, "on the frequency axis")
    _warn_if_degenerate(dropped_t, "on the time axis")
    return assemble_axis_weights((table_f, table_t), (case_f, case_t), (h_f, h_t))
