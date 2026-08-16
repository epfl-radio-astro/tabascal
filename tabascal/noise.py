"""Per-baseline visibility noise.

The antennas of a real array differ in sensitivity, so the noise on a
visibility is a property of its baseline, not of the observation. Collapsing it
to one number mis-weights every point in the likelihood.

On EDA2 the per-baseline ``SIGMA`` spans a factor of ~30, so a scalar
under-weights the quietest baselines by up to ~200x in a chi-squared sum. It is
worse for anything that fits gains: the per-antenna noise **correlates** with
the per-antenna gain (measured ``sigma_a ~ amplitude_a^0.76``, R = 0.96), so a
uniform-noise likelihood cannot tell a loud antenna from a noisy one and the
fitted gain absorbs the noise structure. That is a bias in the calibration
solution, not merely a loss of efficiency.
"""

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def per_baseline_sigma(
    sigma: NDArray, n_time: int, n_bl: int, corr_idx: int = 0
) -> NDArray:
    """Per-baseline noise from an MS ``SIGMA`` column.

    ``SIGMA`` is stored per row, i.e. per (time, baseline), and is constant in
    time for a given baseline -- so it is reduced with a **median** over time
    rather than a mean, which keeps a handful of corrupted rows from dragging a
    baseline's estimate.

    Baselines whose estimate is non-positive or non-finite carry no information:
    those are the dead ones, and they are flagged out of the likelihood anyway.
    They take the median of the valid baselines rather than a zero that would
    divide the likelihood by nothing.

    Parameters
    ----------
    sigma : NDArray
        The ``SIGMA`` column, shape ``(n_row,)`` or ``(n_row, n_corr)``.
    n_time, n_bl : int
        Grid the rows are on. ``n_row`` must equal ``n_time * n_bl``.
    corr_idx : int, optional
        Correlation to select when ``sigma`` carries a correlation axis.

    Returns
    -------
    NDArray
        Per-baseline noise, shape ``(n_bl,)``, in the units of ``SIGMA``.
    """

    sigma = np.asarray(sigma, dtype=np.float64)

    if sigma.ndim == 1:
        sigma = sigma[:, None]

    expected = n_time * n_bl
    if sigma.shape[0] != expected:
        raise ValueError(
            f"SIGMA has {sigma.shape[0]} rows but n_time * n_bl = {expected}. "
            "The column does not match the observation grid."
        )

    if corr_idx >= sigma.shape[1]:
        corr_idx = 0

    per_time = sigma.reshape(n_time, n_bl, -1)[:, :, corr_idx]
    sigma_bl = np.median(per_time, axis=0)

    valid = np.isfinite(sigma_bl) & (sigma_bl > 0)
    if not valid.any():
        raise ValueError(
            "No baseline has a positive, finite SIGMA. The MS carries no usable "
            "noise estimate; set data.noise explicitly."
        )

    if not valid.all():
        n_dead = int((~valid).sum())
        print(
            f"Warning: {n_dead} of {n_bl} baselines have no valid SIGMA (non-positive "
            "or non-finite); using the median of the rest. These are normally the "
            "dead baselines, which are flagged out of the likelihood anyway."
        )
        sigma_bl = np.where(valid, sigma_bl, np.median(sigma_bl[valid]))

    return sigma_bl


def representative_sigma(sigma_bl: NDArray) -> float:
    """One number standing for a per-baseline noise.

    For the heuristics that genuinely need a scalar -- sampling rates, prior
    amplitude scales -- rather than for the likelihood, which should use the
    per-baseline values. The median is used rather than the mean so a few noisy
    baselines do not shift it.
    """

    return float(np.median(np.asarray(sigma_bl, dtype=np.float64)))


def broadcast_to_vis(noise, vis_shape: Tuple[int, ...]):
    """Noise shaped to divide a visibility array of ``vis_shape``.

    A scalar passes through; a per-baseline array gains the trailing axes it
    needs. Broadcasting has to happen **before** any flag masking, because
    ``x[~flags]`` flattens and the per-baseline values would no longer line up
    with the samples they belong to.

    Parameters
    ----------
    noise : float or array_like
        Scalar noise, or per-baseline noise of length ``vis_shape[0]``.
    vis_shape : tuple
        Shape of the visibility array, ``(n_bl, n_freq, n_time)``.
    """

    noise = np.asarray(noise) if not hasattr(noise, "ndim") else noise

    if getattr(noise, "ndim", 0) == 0:
        return noise

    if noise.ndim == 1:
        if noise.shape[0] != vis_shape[0]:
            raise ValueError(
                f"Per-baseline noise has length {noise.shape[0]} but the visibilities "
                f"have {vis_shape[0]} baselines."
            )
        return noise.reshape((-1,) + (1,) * (len(vis_shape) - 1))

    if noise.shape != vis_shape:
        raise ValueError(
            f"Noise of shape {noise.shape} cannot be broadcast onto visibilities "
            f"of shape {vis_shape}."
        )

    return noise


def read_noise_file(path: str, n_bl: int, a1: Optional[NDArray] = None,
                    a2: Optional[NDArray] = None) -> NDArray:
    """Per-baseline noise from an ``.npz``, as ``data.noise`` may point at.

    Accepts either key:

    ``sigma_bl``
        Per-baseline noise, length ``n_bl``, used as given.
    ``s_ant``
        Per-**antenna** noise, combined as ``sqrt(s_p^2 + s_q^2) / sqrt(2)`` --
        the noise on a baseline formed from two independent antennas, normalised
        so that a uniform per-antenna noise reproduces itself.

    Requires ``a1``/``a2`` for the per-antenna form, since the antenna pairs are
    what turn antenna noise into baseline noise.
    """

    with np.load(path) as npz:
        if "sigma_bl" in npz:
            sigma_bl = np.asarray(npz["sigma_bl"], dtype=np.float64).ravel()
            if sigma_bl.size != n_bl:
                raise ValueError(
                    f"{path}: sigma_bl has {sigma_bl.size} entries but the "
                    f"observation has {n_bl} baselines."
                )
            return sigma_bl

        if "s_ant" in npz:
            if a1 is None or a2 is None:
                raise ValueError(
                    f"{path} carries per-antenna noise (s_ant), which needs the "
                    "antenna pairs to form per-baseline values."
                )
            s_ant = np.asarray(npz["s_ant"], dtype=np.float64).ravel()

            return np.sqrt(s_ant[a1] ** 2 + s_ant[a2] ** 2) / np.sqrt(2.0)

        raise ValueError(
            f"{path} holds {sorted(npz.files)}. A noise .npz must carry either "
            "'sigma_bl' (per baseline) or 's_ant' (per antenna)."
        )
