"""Gaussian-process interpolation of a data-grid RFI signal onto the integration grid.

The Fourier-domain RFI priors in :mod:`tabascal.components.rfi_signal` can put
``rfi_A`` on either grid. On the fine grid the supersampling happens inside the
transform -- the latent spectrum is zero-padded before the inverse FFT -- and
the Riemann-sum kernels then integrate that grid. This module is the other
route. The signal is produced on the data grid, one value per channel and time
step standing at the centre of the cell it belongs to, and the fine samples each
cell's integral needs are the conditional mean of the *same* Gaussian process
given a small block of coarse values around the cell. The prior covariance is
known exactly, being the inverse transform of the power spectrum the signal
component samples, so the interpolation is a closed-form solve on that block.
Its weights depend only on the fine offsets within a cell and on which
neighbours the cell has, so one set serves every interior cell and only the
cells within a stencil of an edge get their own.

The visibility is then the Riemann sum the other kernels form, over the
interpolated grid, and the fine grid can be kept to a block of time steps at a
time (:func:`gp_interp_rfi_vis`): the interpolation is local, so a block needs
only its own coarse neighbours, where the Fourier supersampling needed the whole
axis.

Conventions
-----------
Everything follows ``TabConfig._set_freqs_times``. A coarse value sits at the
centre of its cell, and the ``n_int`` fine samples of a cell of width ``dx``
sit at offsets ``(q - n_int // 2) * dx / n_int`` from that centre, so the
sample ``q = n_int // 2`` *is* the coarse point. The weight the interpolation
puts on it is therefore exactly one, and exactly zero on every other stencil
point, up to the round-off of the solve.

The covariance is that of the prior as it is sampled -- the inverse discrete
transform of the power spectrum on the padded, cut k-grid the latent lives on::

    K(x, x') = sum_k pk[k] exp(2 pi i k . (x - x'))

taken as real. The k-grid of an even-length axis carries its Nyquist mode
without a partner, which leaves the sum above with an imaginary part; the real
part is the same sum over the symmetrised spectrum, ``(pk(k) + pk(-k)) / 2``,
which is the stationary prior the spectrum is written as. The difference is
that one mode's power, and a knee inside the grid puts that at the level of the
cutoff.

The spectrum the components sample is a product of one spectrum per axis
(:func:`tabascal.fft_gp.pow_spec_nd`), so the covariance is a product of one
kernel per axis and the conditional mean on a rectangular stencil factorises
into one solve per axis. Where the spectrum factorises that is the path taken:
it is exact, and it conditions as the square root of the joint solve -- for a
smooth prior the joint 9 x 9 system loses six digits where the two 3 x 3 ones
lose three. A spectrum that does not factorise goes through the joint solve.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence, Tuple

import numpy as np
from jax import Array, checkpoint, lax
import jax.numpy as jnp

from tabascal.interferometry import calculate_rfi_vis_blocked
from tabascal.rfi_path import fine_phase_from_path


#: Eigenvalues of a stencil covariance below this fraction of its largest are
#: treated as zero. The prior then puts no power on those combinations of the
#: coarse values, and the conditional mean of a degenerate Gaussian leaves them
#: out -- a pseudo-inverse rather than a solve. A spectrum cut down to a single
#: mode on an axis is the clear case: every coarse value is then the same draw,
#: the stencil covariance has rank one, and the interpolation is their mean. A
#: smooth prior on a fine data grid approaches it by degrees, its smallest
#: eigenvalue falling as a power of the cell width over the correlation scale,
#: and below this fraction a float64 solve would be amplifying round-off.
RCOND = 1e-12


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def fine_offsets(n_int: int, dx: float) -> np.ndarray:
    """Offsets of a cell's fine samples from its centre, as ``TabConfig`` lays them out.

    ``(q - n_int // 2) * dx / n_int`` for ``q`` in ``0..n_int-1``: the sample
    ``n_int // 2`` is the centre itself. For an even ``n_int`` the samples are
    not symmetric about the centre -- the first sits on the cell's lower edge --
    which is how :func:`tabascal.fft_gp.domain_ss` builds the fine grid the RFI
    phase is evaluated on, and what the interpolation has to match.
    """
    n_int = int(n_int)
    if n_int < 1:
        raise ValueError(f"n_int must be at least 1, got {n_int}.")
    return (np.arange(n_int) - n_int // 2) * (float(dx) / n_int)


def stencil_offsets(hs: Sequence[int]) -> np.ndarray:
    """The ``(n_stencil, 2)`` integer offsets ``(d_freq, d_time)`` of the coarse stencil.

    Lexicographic: the frequency offset varies slowest. A half-width of 1 on
    both axes is the 3 x 3 block of nine coarse values around a cell.
    """
    h_f, h_t = (int(h) for h in hs)
    if h_f < 0 or h_t < 0:
        raise ValueError(f"Stencil half-widths must be non-negative, got {list(hs)}.")
    d_f, d_t = np.meshgrid(
        np.arange(-h_f, h_f + 1), np.arange(-h_t, h_t + 1), indexing="ij"
    )
    return np.stack([d_f.ravel(), d_t.ravel()], axis=-1)


def stencil_availability(n: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """Which stencil offsets exist for each cell of an axis of ``n`` cells.

    Returns ``(patterns, case)``: the distinct availability patterns, each a
    boolean row over the offsets ``-h..h``, and the pattern index of every cell.
    Interior cells share one pattern; a cell within ``h`` of an edge lacks the
    offsets that would leave the grid. An axis shorter than the stencil is just
    more of the same -- a cell can be short of neighbours on both sides.
    """
    cells = np.arange(int(n))[:, None] + np.arange(-int(h), int(h) + 1)[None, :]
    avail = (cells >= 0) & (cells < int(n))
    patterns, case = np.unique(avail, axis=0, return_inverse=True)
    return patterns, np.asarray(case).ravel()


# ---------------------------------------------------------------------------
# Covariance and weights (host side, float64)
# ---------------------------------------------------------------------------


def prior_covariance(pk, ks: Sequence, lags: Sequence) -> np.ndarray:
    """The prior covariance on an outer grid of signed lags, one lag array per axis.

    ``K[l_f, l_t] = Re sum_{k_f, k_t} pk[k_f, k_t] exp(2 pi i (k_f l_f + k_t l_t))``,
    the covariance between two points of the sampled GP separated by
    ``(l_f, l_t)`` -- ``pk`` and ``ks`` being the spectrum and k-grid of
    :func:`tabascal.fft_gp.latent_to_signal_init`, the modes the latent actually
    carries -- under the symmetrised spectrum (see the module notes on the real
    part). Two matrix products rather than a sum over the outer grid, and in
    float64 whatever precision the spectrum arrives in: the weights are a solve
    on a block of this, and the working precision would not do (see
    :data:`RCOND`).
    """
    pk = np.asarray(pk, dtype=np.float64)
    if pk.ndim != 2 or len(ks) != 2 or len(lags) != 2:
        raise ValueError(
            "prior_covariance takes a 2-D spectrum with one k array and one lag "
            f"array per axis, got pk.shape={pk.shape}, {len(ks)} k arrays and "
            f"{len(lags)} lag arrays."
        )
    phases = [
        np.exp(
            2j
            * np.pi
            * np.asarray(k, dtype=np.float64)[:, None]
            * np.asarray(lag, dtype=np.float64)[None, :]
        )
        for k, lag in zip(ks, lags)
    ]
    return np.ascontiguousarray((phases[0].T @ pk @ phases[1]).real)


def prior_covariance_1d(pk_axis, k_axis, lags) -> np.ndarray:
    """One axis of :func:`prior_covariance`: ``Re sum_k pk[k] exp(2 pi i k lag)``."""
    pk_axis = np.asarray(pk_axis, dtype=np.float64)
    k_axis = np.asarray(k_axis, dtype=np.float64)
    lags = np.asarray(lags, dtype=np.float64)
    return np.cos(2 * np.pi * lags[:, None] * k_axis[None, :]) @ pk_axis


def factorise_spectrum(pk, rtol: float = 1e-5) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """``(pk_freq, pk_time)`` with ``pk == outer(pk_freq, pk_time)``, or ``None``.

    :func:`tabascal.fft_gp.pow_spec_nd` builds the spectrum as exactly this
    outer product, so the test passes for every spectrum the signal components
    sample; the tolerance is there for a spectrum that was formed in single
    precision, whose entries carry that rounding (a few ``1e-8``) against the
    product of their factors. The factors are read off the row and column
    through the spectrum's peak, which is where a spectrum with any power at all
    is furthest from zero.
    """
    pk = np.asarray(pk, dtype=np.float64)
    if pk.ndim != 2 or pk.size == 0:
        return None
    i, j = np.unravel_index(int(np.argmax(pk)), pk.shape)
    peak = pk[i, j]
    if not np.isfinite(peak) or peak <= 0:
        return None
    pk_f = pk[:, j] / peak
    pk_t = pk[i, :]
    if not np.allclose(pk, np.outer(pk_f, pk_t), rtol=rtol, atol=0.0):
        return None
    return pk_f, pk_t


def _solve_weights(
    K_SS: np.ndarray, K_fS: np.ndarray, keep: np.ndarray, rcond: float = RCOND
) -> Tuple[np.ndarray, int]:
    """``K_*S K_SS^+`` over the stencil points ``keep``, and how many directions were dropped.

    ``K_fS`` is ``(n_fine, n_stencil)``; the result has that shape, zero on every
    stencil point not in ``keep`` -- those meet the zeros :func:`stencil_stack`
    pads the signal with. The pseudo-inverse is by eigendecomposition, the
    matrix being symmetric: eigenvalues below ``rcond`` of the largest are
    dropped (see :data:`RCOND`).
    """
    K = K_SS[np.ix_(keep, keep)]
    R = K_fS[:, keep]
    lam, U = np.linalg.eigh(K)
    if not np.isfinite(lam).all() or lam.max() <= 0.0:
        raise ValueError(
            "The prior covariance on the interpolation stencil has no positive "
            "power. Check the rfi.gp_cov correlation scales and rfi.cutoff: the "
            "spectrum the signal component sampled is empty or not finite."
        )
    ok = lam > rcond * lam.max()
    K_pinv = (U[:, ok] / lam[ok]) @ U[:, ok].T
    W = np.zeros((K_fS.shape[0], K_fS.shape[1]), dtype=np.float64)
    W[:, keep] = R @ K_pinv
    return W, int(np.count_nonzero(~ok))


def _warn_if_degenerate(dropped: int, where: str) -> None:
    if dropped:
        warnings.warn(
            f"The prior covariance on the interpolation stencil is degenerate "
            f"{where}: the prior puts no power (below {RCOND:.0e} of its largest "
            f"mode) on {dropped} combination(s) of the neighbouring coarse "
            "values, which therefore do not enter the interpolation. A smaller "
            "rfi.gp_interp_stencil, or correlation scales closer to the data "
            "grid, removes the degeneracy.",
            RuntimeWarning,
            stacklevel=3,
        )


def _axis_weights(
    pk_axis, k_axis, dx: float, n_int: int, h: int, n: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    """One axis of the factorised solve: ``(table, case, dropped)``.

    ``table[c, q, d + h]`` is the weight of fine sample ``q`` of a cell with
    availability pattern ``c`` on the coarse value at offset ``d``; ``case[j]``
    is the pattern of cell ``j``.
    """
    d = np.arange(-h, h + 1)
    C_cc = prior_covariance_1d(pk_axis, k_axis, np.arange(-2 * h, 2 * h + 1) * dx)
    K_SS = C_cc[(d[:, None] - d[None, :]) + 2 * h]
    K_fS = prior_covariance_1d(
        pk_axis, k_axis, (fine_offsets(n_int, dx)[:, None] - d[None, :] * dx).ravel()
    ).reshape(n_int, 2 * h + 1)

    patterns, case = stencil_availability(n, h)
    table = np.zeros((len(patterns), n_int, 2 * h + 1), dtype=np.float64)
    dropped = 0
    for c, pattern in enumerate(patterns):
        table[c], n_dropped = _solve_weights(K_SS, K_fS, np.flatnonzero(pattern))
        dropped = max(dropped, n_dropped)
    return table, case, dropped


def interpolation_weights(
    pk,
    ks: Sequence,
    dxs: Sequence[float],
    n_ints: Sequence[int],
    hs: Sequence[int],
    ns: Sequence[int],
) -> np.ndarray:
    """Conditional-mean weights of every cell's fine samples on its coarse stencil.

    For a cell ``(j, k)`` of the ``(n_freq, n_time)`` data grid and its fine
    sample ``(p, q)``, ``W[j, k, p, q, s]`` is the weight on the coarse value at
    stencil offset ``s`` (see :func:`stencil_offsets`), so that the interpolated
    sample is ``sum_s W[j, k, p, q, s] * A[j + d_f(s), k + d_t(s)]``. It is the
    GP conditional mean, ``K_*S K_SS^-1``, with the covariance of
    :func:`prior_covariance` on the stencil points ``S`` and the fine point
    ``*``; a stencil point beyond the grid is left out of the conditioning and
    carries a weight of zero, so a zero-padded signal can be gathered without a
    branch. Cells with the same set of neighbours share one solve, and a
    spectrum that factorises over the axes (every one the signal components
    sample) is solved one axis at a time -- see the module notes.

    Parameters
    ----------
    pk, ks
        The spectrum and k-grid of the sampled prior, as
        :func:`tabascal.fft_gp.latent_to_signal_init` returns them.
    dxs : (dx_freq, dx_time)
        The data-grid cell widths, in the units ``ks`` are conjugate to.
    n_ints : (n_int_freq, n_int_time)
        Fine samples per cell on each axis.
    hs : (h_freq, h_time)
        Stencil half-widths on each axis.
    ns : (n_freq, n_time)
        The data-grid size, for the edge cases.

    Returns
    -------
    ndarray (n_freq, n_time, n_int_freq, n_int_time, n_stencil), float64
    """
    dx_f, dx_t = (float(dx) for dx in dxs)
    n_int_f, n_int_t = (int(n) for n in n_ints)
    h_f, h_t = (int(h) for h in hs)
    n_f, n_t = (int(n) for n in ns)
    if n_f < 1 or n_t < 1:
        raise ValueError(f"The data grid must have at least one cell per axis, got {list(ns)}.")

    offsets = stencil_offsets((h_f, h_t))
    d_f, d_t = offsets[:, 0], offsets[:, 1]

    factors = factorise_spectrum(pk)
    if factors is not None:
        pk_f, pk_t = factors
        table_f, case_f, dropped_f = _axis_weights(pk_f, ks[0], dx_f, n_int_f, h_f, n_f)
        table_t, case_t, dropped_t = _axis_weights(pk_t, ks[1], dx_t, n_int_t, h_t, n_t)
        _warn_if_degenerate(dropped_f, "on the frequency axis")
        _warn_if_degenerate(dropped_t, "on the time axis")
        # (n_f, n_int_f, S) and (n_t, n_int_t, S), then the outer product.
        W_f = table_f[case_f][:, :, d_f + h_f]
        W_t = table_t[case_t][:, :, d_t + h_t]
        return np.ascontiguousarray(W_f[:, None, :, None, :] * W_t[None, :, None, :, :])

    # The joint solve, for a spectrum that is not a product over the axes.
    # Coarse-coarse lags, (d - d') * dx for d - d' in -2h..2h, and fine-coarse
    # lags, offset(p) - d * dx.
    cc_f = np.arange(-2 * h_f, 2 * h_f + 1) * dx_f
    cc_t = np.arange(-2 * h_t, 2 * h_t + 1) * dx_t
    fc_f = (fine_offsets(n_int_f, dx_f)[:, None] - np.arange(-h_f, h_f + 1)[None, :] * dx_f).ravel()
    fc_t = (fine_offsets(n_int_t, dx_t)[:, None] - np.arange(-h_t, h_t + 1)[None, :] * dx_t).ravel()
    C_cc = prior_covariance(pk, ks, [cc_f, cc_t])
    C_fc = prior_covariance(pk, ks, [fc_f, fc_t])

    # K_SS[s, s'] = K(x_s - x_s'), and K_*S[(p, q), s] = K(x_* - x_s).
    K_SS = C_cc[(d_f[:, None] - d_f[None, :]) + 2 * h_f, (d_t[:, None] - d_t[None, :]) + 2 * h_t]
    idx_f = np.arange(n_int_f)[:, None] * (2 * h_f + 1) + (d_f[None, :] + h_f)
    idx_t = np.arange(n_int_t)[:, None] * (2 * h_t + 1) + (d_t[None, :] + h_t)
    K_fS = C_fc[idx_f[:, None, :], idx_t[None, :, :]].reshape(n_int_f * n_int_t, -1)

    patterns_f, case_f = stencil_availability(n_f, h_f)
    patterns_t, case_t = stencil_availability(n_t, h_t)
    table = np.zeros(
        (len(patterns_f), len(patterns_t), n_int_f, n_int_t, len(offsets)), dtype=np.float64
    )
    dropped = 0
    for a, pattern_f in enumerate(patterns_f):
        for b, pattern_t in enumerate(patterns_t):
            keep = np.flatnonzero(pattern_f[d_f + h_f] & pattern_t[d_t + h_t])
            W, n_dropped = _solve_weights(K_SS, K_fS, keep)
            table[a, b] = W.reshape(n_int_f, n_int_t, -1)
            dropped = max(dropped, n_dropped)
    _warn_if_degenerate(dropped, "on the joint stencil")

    return np.ascontiguousarray(table[case_f[:, None], case_t[None, :]])


# ---------------------------------------------------------------------------
# Applying the weights (JAX)
# ---------------------------------------------------------------------------


def stencil_stack(A: Array, offsets: np.ndarray, hs: Sequence[int]) -> Array:
    """The coarse neighbours of every cell, stacked: ``(n_stencil, ..., n_freq, n_time)``.

    Entry ``s`` is the signal shifted by ``offsets[s]``, with zeros where the
    shift leaves the grid. Those zeros meet zero weights in
    :func:`interpolation_weights`, so the missing neighbours contribute nothing
    rather than something wrong; and a zero against a zero, not a ``where``,
    because a missing neighbour has no value to be non-finite.
    """
    h_f, h_t = (int(h) for h in hs)
    n_f, n_t = A.shape[-2:]
    pad = [(0, 0)] * (A.ndim - 2) + [(h_f, h_f), (h_t, h_t)]
    padded = jnp.pad(A, pad)
    return jnp.stack(
        [
            padded[..., h_f + d_f : h_f + d_f + n_f, h_t + d_t : h_t + d_t + n_t]
            for d_f, d_t in np.asarray(offsets)
        ]
    )


def interpolate_fine(stack: Array, weights: Array) -> Array:
    """Apply the weights to a stencil stack: ``(..., n_freq_fine, n_time_fine)``.

    ``stack`` is ``(n_stencil, ..., n_freq, n_time)`` from :func:`stencil_stack`
    and ``weights`` is ``(n_freq, n_time, n_int_freq, n_int_time, n_stencil)``
    for the same cells -- a block of the time axis of both is as good as the
    whole. The fine grid comes out in the layout the visibility kernels reshape,
    ``(n_freq, n_int_freq, n_time, n_int_time)`` flattened pairwise.
    """
    fine = jnp.einsum("s...jk,jkpqs->...jpkq", stack, weights)
    lead = fine.shape[:-4]
    n_f, n_int_f, n_t, n_int_t = fine.shape[-4:]
    return jnp.reshape(fine, lead + (n_f * n_int_f, n_t * n_int_t))


def _blocked_vis(
    stack: Array,
    weights: Array,
    phase_of,
    n_time: int,
    a1: Array,
    a2: Array,
    n_int_freq: int,
    n_int_time: int,
    baseline_block_size: Optional[int],
    time_block_size: Optional[int],
) -> Array:
    """The time-blocked Riemann sum shared by the two phase routes.

    ``phase_of(t0, n_blk)`` returns the fine-grid phase of ``n_blk`` cells from
    cell ``t0``, ``(n_rfi, n_ant, n_freq_fine, n_blk * n_int_time)``; ``t0`` is
    traced inside the scan and a Python int outside it, so it slices with
    ``dynamic_slice`` either way. See :func:`gp_interp_rfi_vis` for the rest.
    """
    n_bl = a1.shape[0]
    n_freq = stack.shape[-2]

    def vis_of(t0, n_blk):
        stack_b = lax.dynamic_slice_in_dim(stack, t0, n_blk, axis=-1)
        weights_b = lax.dynamic_slice_in_dim(weights, t0, n_blk, axis=1)
        fine = interpolate_fine(stack_b, weights_b)
        return calculate_rfi_vis_blocked(
            fine, phase_of(t0, n_blk), a1, a2, n_int_freq, n_int_time, baseline_block_size
        )

    if time_block_size is None or int(time_block_size) >= n_time:
        return vis_of(0, n_time)

    block = int(time_block_size)
    if block < 1:
        raise ValueError(f"time_block_size must be at least 1, got {time_block_size}.")
    n_full = n_time // block

    def scan_body(carry, b):
        return carry, vis_of(b * block, block)

    # prevent_cse=False is what JAX documents for a remat body under lax.scan;
    # see calculate_rfi_vis_blocked, whose inner scan is the same shape.
    _, vis_blocks = lax.scan(
        checkpoint(scan_body, prevent_cse=False), None, jnp.arange(n_full)
    )
    # (n_full, n_bl, n_freq, block) -> (n_bl, n_freq, n_full * block)
    vis = jnp.reshape(jnp.moveaxis(vis_blocks, 0, 2), (n_bl, n_freq, n_full * block))

    if n_full * block < n_time:
        t0 = n_full * block
        vis = jnp.concatenate([vis, vis_of(t0, n_time - t0)], axis=-1)

    return vis


def gp_interp_rfi_vis(
    rfi_A: Array,
    rfi_phase: Array,
    weights: Array,
    offsets: np.ndarray,
    hs: Sequence[int],
    a1: Array,
    a2: Array,
    n_int_freq: int,
    n_int_time: int,
    baseline_block_size: Optional[int],
    time_block_size: Optional[int],
) -> Array:
    """Riemann-sum RFI visibilities from a data-grid signal, a block of time steps at a time.

    ``rfi_A`` is ``(n_rfi, n_ant, n_freq, n_time)`` on the data grid and
    ``rfi_phase`` is ``(n_rfi, n_ant, n_freq_fine, n_time_fine)`` on the fine
    one; each block of ``time_block_size`` time steps is interpolated with
    :func:`interpolate_fine` and integrated with
    :func:`tabascal.interferometry.calculate_rfi_vis_blocked`, which walks the
    baselines in blocks of ``baseline_block_size`` in its turn. ``None`` for the
    time block is every time step at once, with no scan over the axis.

    The scan is under ``checkpoint`` so that a block's fine grid is recomputed in
    the backward pass rather than kept, as the baseline scan's is; without it
    the tape would stack every block's residuals and hold the whole fine grid
    after all. A last block shorter than the rest is done outside the scan,
    which has to be over blocks of one shape, rather than by padding the phase
    -- the largest array in the run -- out to a whole block.

    Returns ``(n_bl, n_freq, n_time)``.
    """
    n_int_time = int(n_int_time)
    stack = stencil_stack(rfi_A, offsets, hs)

    def phase_of(t0, n_blk):
        return lax.dynamic_slice_in_dim(rfi_phase, t0 * n_int_time, n_blk * n_int_time, axis=-1)

    return _blocked_vis(
        stack, weights, phase_of, rfi_A.shape[-1], a1, a2,
        int(n_int_freq), n_int_time, baseline_block_size, time_block_size,
    )


def gp_interp_rfi_vis_path(
    rfi_A: Array,
    rfi_phase: Array,
    rfi_path: Array,
    freqs_fine: Array,
    dnu: Array,
    powers: Array,
    weights: Array,
    offsets: np.ndarray,
    hs: Sequence[int],
    a1: Array,
    a2: Array,
    n_int_freq: int,
    n_int_time: int,
    baseline_block_size: Optional[int],
    time_block_size: Optional[int],
) -> Array:
    """:func:`gp_interp_rfi_vis` with the phase on the data grid too.

    ``rfi_phase`` is ``(n_rfi, n_ant, n_freq, n_time)``, the wrapped phase at the
    channel and cell centres, and ``rfi_path`` is ``(n_rfi, n_ant, n_time,
    order + 1)``, the differential path and its time derivatives there; each
    block's fine phase is rebuilt from them with
    :func:`tabascal.rfi_path.fine_phase_from_path` inside the scan, so neither
    the fine phase nor the fine amplitude exists beyond a block. ``freqs_fine``,
    ``dnu`` and ``powers`` are the frequency and time terms of that
    reconstruction, from :func:`tabascal.rfi_path.fine_frequency_terms` and
    :func:`tabascal.rfi_path.taylor_powers`.
    """
    stack = stencil_stack(rfi_A, offsets, hs)

    def phase_of(t0, n_blk):
        phase_b = lax.dynamic_slice_in_dim(rfi_phase, t0, n_blk, axis=-1)
        path_b = lax.dynamic_slice_in_dim(rfi_path, t0, n_blk, axis=-2)
        return fine_phase_from_path(phase_b, path_b, freqs_fine, dnu, powers)

    return _blocked_vis(
        stack, weights, phase_of, rfi_A.shape[-1], a1, a2,
        int(n_int_freq), int(n_int_time), baseline_block_size, time_block_size,
    )
