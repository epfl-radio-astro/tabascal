"""The RFI geometric phase from its data-grid value and the path's time derivatives.

The phase of a source at an antenna is ``-2 pi nu L / c`` for the geometric path
``L = |x_ant - x_src| + w`` in metres (:func:`tabascal.interferometry.get_rfi_phase`),
and the fine grid exists because ``L`` changes within an integration -- the
fringe turns. The phase is nevertheless a smooth, deterministic function of two
things, and both can be carried on the data grid:

* **Frequency.** The phase is exactly linear in ``nu`` with slope ``-2 pi L / c``.
  So the phase at a channel centre, plus ``L`` itself, gives the phase at every
  fine frequency of that channel, with no expansion at all.
* **Time.** ``L(t)`` within a cell is its Taylor series about the cell centre,
  ``L_0 + L_1 tau + L_2 tau^2 / 2 + ...``, truncated at a low order: satellite
  motion is smooth on the scale of an integration, and the derivatives are
  taken from the propagated positions themselves rather than fitted.

What is carried is therefore ``rfi_phase`` on the data grid -- the wrapped phase
at every channel centre and cell centre, which is the fine-grid phase at those
samples exactly -- and ``rfi_path``, the path and its time derivatives at the
cell centres, ``(n_rfi, n_ant, n_time, order + 1)``. The fine phase of a cell is
rebuilt from the two in :func:`fine_phase_from_path` as::

    phase(nu_f, t_i + tau) = phase(nu_j, t_i)
                             - 2 pi / c * ( (nu_f - nu_j) * L_0
                                          + nu_f * (L_1 tau + L_2 tau^2 / 2 + ...) )

**Precision.** The wrapped centre phase is computed in float64 and is small. The
path is carried *differential to the array mean* at each time, ``L - mean_ant(L)``:
a phase common to every antenna of a source cancels in ``A_p conj(A_q)``, and
the differential path is bounded by the longest baseline rather than by the
range to the satellite -- a few thousand metres against a few million, which is
what keeps the reconstruction inside single precision. Its rate is likewise
bounded by the baseline times the source's angular rate.

**Derivatives.** From the propagated positions on the fine grid the components
already build: the path is evaluated at a small window of fine samples around
each cell centre and a polynomial through them is differentiated at the centre
(:func:`path_derivative_weights`). Central where the window fits, one-sided at
the ends of the observation. The window is a few samples wide, so the cost is a
few evaluations per cell rather than one per fine sample.
"""

from __future__ import annotations

from math import factorial
from typing import Tuple

import numpy as np
from jax import Array
import jax.numpy as jnp

from tabascal.interferometry import C


def window_size(order: int) -> int:
    """Fine samples in the derivative window for a Taylor expansion of ``order``.

    An odd count, symmetric about the centre, one more than the order at least:
    the polynomial through the window has one more degree than the expansion
    keeps, so each kept derivative is central-difference accurate.
    """
    order = int(order)
    if order < 0:
        raise ValueError(f"The expansion order must be non-negative, got {order}.")
    n = order + 2
    return n if n % 2 else n + 1


def path_derivative_weights(
    times_jd_fine, n_time: int, n_int_time: int, order: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fine-sample windows and the weights that differentiate the path at each cell centre.

    Returns ``(windows, centres, weights)``: ``windows`` is ``(n_time, n_pts)``
    of fine-grid indices around each cell's centre sample, ``centres``
    ``(n_time,)`` is the position of the centre sample within its window, and
    ``weights`` is ``(n_time, order + 1, n_pts)`` with ``weights[i, k]`` the
    coefficients that turn the path at the window's samples into its ``k``-th
    time derivative at the centre, in ``m / s^k``.

    The weights are the derivatives at the centre of the polynomial through the
    window, from the inverse of a Vandermonde matrix -- interior windows give
    the classical central-difference formulas, windows against the ends of the
    observation the one-sided ones. On a fine grid shorter than the window, the
    window is the grid and the order is what that many points support.

    The sample times are taken from ``times_jd_fine``, the Julian dates the
    positions were propagated at, and not from the nominal spacing. A float64
    Julian date near 2.45e6 resolves about 20 us, so the propagated samples sit
    at times jittered by that much from the nominal grid: harmless to the
    positions, and to a phase evaluated sample by sample, but a finite
    difference divides the jitter by a power of the spacing and the derivatives
    it gives get worse with every order. The differences between the dates are
    exact in float64, so fitting at those times removes the jitter from the
    derivatives entirely.
    """
    n_time, n_int_time, order = int(n_time), int(n_int_time), int(order)
    n_fine = n_time * n_int_time
    jd = np.asarray(times_jd_fine, dtype=np.float64)
    if jd.shape != (n_fine,):
        raise ValueError(
            f"times_jd_fine has shape {jd.shape}; expected ({n_fine},) for "
            f"{n_time} time steps of {n_int_time} samples."
        )
    seconds = (jd - jd[0]) * 86400.0
    n_pts = min(window_size(order), n_fine)
    order = min(order, n_pts - 1)

    centre_idx = np.arange(n_time) * n_int_time + n_int_time // 2
    start = np.clip(centre_idx - n_pts // 2, 0, n_fine - n_pts)
    windows = start[:, None] + np.arange(n_pts)[None, :]
    centres = centre_idx - start

    weights = np.zeros((n_time, order + 1, n_pts))
    for i in range(n_time):
        offsets = seconds[windows[i]] - seconds[centre_idx[i]]
        # In units of the window's spacing, for a well-conditioned inverse.
        scale = float(np.max(np.abs(offsets))) or 1.0
        vandermonde = (offsets / scale)[:, None] ** np.arange(n_pts)[None, :]
        coefficients = np.linalg.inv(vandermonde)  # coefficients[k] gives a_k
        for k in range(order + 1):
            weights[i, k] = factorial(k) * coefficients[k] / scale**k

    return windows, centres, weights


def coarse_phase_and_path(
    rfi_xyz,
    ants_uvw,
    ants_xyz,
    freqs,
    windows,
    centres,
    weights,
    xp=np,
):
    """The data-grid phase and the differential path with its time derivatives.

    ``xp`` is ``numpy`` for the one-shot float64 setup of ``FixedOrbitCoarse``
    and ``jax.numpy`` for the differentiable ``PathCalculationRFI``; the two
    paths are the same code, which is what keeps them in agreement.

    Parameters
    ----------
    rfi_xyz : Array (n_rfi, n_time_fine, 3)
        Source positions on the fine time grid, ECI, metres.
    ants_uvw : Array (n_ant, n_time_fine, 3)
        Antenna UVW on the fine grid; only ``w`` is used, as in ``get_rfi_phase``.
    ants_xyz : Array (n_ant, n_time_fine, 3)
        Antenna positions on the fine grid, ECI, metres.
    freqs : Array (n_freq,)
        Channel centre frequencies in Hz.
    windows, centres, weights
        From :func:`path_derivative_weights`.

    Returns
    -------
    rfi_phase : Array (n_rfi, n_ant, n_freq, n_time)
        The wrapped phase at each channel centre and cell centre: what
        ``get_rfi_phase`` gives at those fine samples.
    rfi_path : Array (n_rfi, n_ant, n_time, order + 1)
        The path differential to the array mean, and its time derivatives, at
        the cell centres.
    """
    n_time, n_pts = windows.shape
    idx = xp.asarray(windows).reshape(-1)

    # L = |x_ant - x_src| + w at the window samples, (n_rfi, n_ant, n_time, n_pts).
    distances = xp.linalg.norm(
        ants_xyz[None, :, idx, :] - rfi_xyz[:, None, idx, :], axis=-1
    )
    path = distances + ants_uvw[None, :, idx, -1]
    path = path.reshape(path.shape[0], path.shape[1], n_time, n_pts)

    # The centre sample's own path gives the phase there, exactly as get_rfi_phase
    # forms it: wrapped in cycles first, so the float32 cast of the result is
    # of a number below one.
    at_centre = xp.take_along_axis(
        path, xp.asarray(centres)[None, None, :, None], axis=-1
    )[..., 0]
    fringe = (at_centre[:, :, None, :] * freqs[None, None, :, None] / C) % 1
    rfi_phase = -2.0 * xp.pi * fringe

    # Differential to the array mean, then differentiated: the weights are linear.
    path = path - xp.mean(path, axis=1, keepdims=True)
    rfi_path = xp.einsum("ratm,tkm->ratk", path, xp.asarray(weights))

    return rfi_phase, rfi_path


def fine_frequency_terms(
    freqs, n_int_freq: int, chan_width: float
) -> Tuple[np.ndarray, np.ndarray]:
    """``(freqs_fine, dnu)``: the fine frequencies and their offsets from the channel centre.

    The fine offsets are those of :func:`tabascal.gp_interp.fine_offsets`, so the
    frequencies are the ones ``TabConfig`` evaluates the fine phase at.
    """
    from tabascal.gp_interp import fine_offsets

    freqs = np.asarray(freqs, dtype=np.float64)
    dnu = np.tile(fine_offsets(n_int_freq, chan_width), len(freqs))
    freqs_fine = np.repeat(freqs, n_int_freq) + dnu
    return freqs_fine, dnu


def taylor_powers(tau, order: int) -> np.ndarray:
    """``tau^k / k!`` for ``k = 1..order``, ``(order, n_int_time)``."""
    tau = np.asarray(tau, dtype=np.float64)
    ks = np.arange(1, int(order) + 1)
    return tau[None, :] ** ks[:, None] / np.array([factorial(int(k)) for k in ks])[:, None]


def fine_phase_from_path(
    rfi_phase: Array,
    rfi_path: Array,
    freqs_fine: Array,
    dnu: Array,
    powers: Array,
) -> Array:
    """The fine-grid phase of a block of cells, from the data-grid phase and path.

    Parameters
    ----------
    rfi_phase : Array (n_rfi, n_ant, n_freq, n_blk)
        The wrapped phase at the channel and cell centres of the block.
    rfi_path : Array (n_rfi, n_ant, n_blk, order + 1)
        The differential path and its derivatives at the block's cell centres.
    freqs_fine : Array (n_freq_fine,)
        The fine frequencies, channel-major.
    dnu : Array (n_freq_fine,)
        Their offsets from the channel centres.
    powers : Array (order, n_int_time)
        :func:`taylor_powers` of the fine time offsets within a cell.

    Returns
    -------
    Array (n_rfi, n_ant, n_freq_fine, n_blk * n_int_time)
        In the layout the visibility kernels reshape.
    """
    n_rfi, n_ant, n_freq, n_blk = rfi_phase.shape
    n_freq_fine = freqs_fine.shape[0]
    n_int_freq = n_freq_fine // n_freq
    n_int_time = powers.shape[-1]

    # L(tau) - L_0 within the cell, (n_rfi, n_ant, n_blk, n_int_time).
    d_path = jnp.einsum("ratk,kq->ratq", rfi_path[..., 1:], powers)

    centre = jnp.repeat(rfi_phase, n_int_freq, axis=2)[..., None]
    across_channel = dnu[None, None, :, None, None] * rfi_path[:, :, None, :, 0, None]
    within_cell = freqs_fine[None, None, :, None, None] * d_path[:, :, None, :, :]
    phase = centre - (2.0 * jnp.pi / C) * (across_channel + within_cell)

    return jnp.reshape(phase, (n_rfi, n_ant, n_freq_fine, n_blk * n_int_time))
