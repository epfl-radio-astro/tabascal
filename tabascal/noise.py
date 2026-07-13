"""Per-baseline noise estimation for the tabascal likelihood.

tabascal's likelihood is Normal(pred, noise) with a single scalar ``noise``, i.e. it
assumes every visibility has the same uncertainty. That is not true: the antennas differ
in sensitivity, and the standard radio-interferometric noise model is per-*antenna*,

    sigma_bl = sqrt(SEFD_p * SEFD_q / (2 dnu tau))   ->   sigma_bl = s_p * s_q

with one noise amplitude ``s_p`` per antenna. Measured on EDA2 (multisat 27chan_t221-311)
the per-baseline noise spans 3.2x (p10/p90 = 2620/8471 Jy) and this rank-1 per-antenna
factorisation reproduces it to **1.4%** -- so it is the noise, not an approximation.

Estimation is **model-free and needs no calibration**: the sky and the RFI are smooth in
frequency (over this band the RFI GP fits a single frequency mode), while thermal noise is
white in frequency. Differencing adjacent channels therefore isolates the noise:

    d = (V[nu+1] - V[nu]) / sqrt(2)     ->   sigma_bl = sqrt(<|d|^2> / 2)

(the /2 converts the complex modulus to the per-component sigma the likelihood wants).
No gains, no model, no fit. It can be run on the raw column before anything else.

Do NOT parameterise the noise as a power of the gain (sigma ~ |g_p g_q|^alpha): the
antenna noise correlates with the gain (s_p ~ |g_p|^0.59, Spearman 0.86) but scatters
0.59 dex -- a factor ~4 -- about that power law. The per-antenna s_p is the right model.
"""

import numpy as np
from numpy.typing import NDArray


def estimate_baseline_noise(
    vis: NDArray,
    a1: NDArray,
    a2: NDArray,
    flags: NDArray = None,
    n_ant: int = None,
):
    """Per-baseline and per-antenna noise, from channel differences. Model-free.

    Returns
    -------
    dict with
        ``sigma_bl`` (n_bl,)  -- per-baseline noise, per real/imag component
        ``s_ant``    (n_ant,) -- per-antenna noise amplitude, sigma_bl ~ s_p * s_q
        ``sigma_fit``(n_bl,)  -- sigma_bl reconstructed from s_ant
        ``frac_err`` float    -- scatter of sigma_bl about the factorisation
    """
    vis = np.asarray(vis)
    a1, a2 = np.asarray(a1), np.asarray(a2)
    n_ant = int(max(a1.max(), a2.max())) + 1 if n_ant is None else n_ant

    if vis.shape[1] < 2:
        raise ValueError(
            "Need >= 2 channels to estimate the noise by channel-differencing."
        )

    d = (vis[:, 1:, :] - vis[:, :-1, :]) / np.sqrt(2.0)
    if flags is not None:
        f = np.asarray(flags, dtype=bool)
        keep = (~f[:, 1:, :]) & (~f[:, :-1, :])
        d = np.where(keep, d, np.nan)

    with np.errstate(invalid="ignore"):
        sigma_bl = np.sqrt(np.nanmean(np.abs(d) ** 2, axis=(1, 2)) / 2.0)

    # Rank-1 factorisation in log space: log sigma_bl = log s_p + log s_q.
    auto = a1 == a2
    m = (~auto) & np.isfinite(sigma_bl) & (sigma_bl > 0)
    rows = np.arange(int(m.sum()))
    A = np.zeros((int(m.sum()), n_ant))
    A[rows, a1[m]] += 1.0
    A[rows, a2[m]] += 1.0
    logs, *_ = np.linalg.lstsq(A, np.log(sigma_bl[m]), rcond=None)
    s_ant = np.exp(logs)

    sigma_fit = s_ant[a1] * s_ant[a2]
    frac = float(np.std(np.log(sigma_bl[m] / sigma_fit[m])))

    # Autos (and any unusable baseline) get the factorised value, which is well defined
    # everywhere; they are excluded from the likelihood by the flags in any case.
    sigma_out = np.where(np.isfinite(sigma_bl) & (sigma_bl > 0), sigma_bl, sigma_fit)

    return dict(sigma_bl=sigma_out, s_ant=s_ant, sigma_fit=sigma_fit, frac_err=frac)


def save_noise_npz(path: str, result: dict) -> None:
    np.savez(path, **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})


def load_baseline_noise(path: str, n_bl: int, a1: NDArray, a2: NDArray) -> NDArray:
    """Load a per-baseline noise array (n_bl,) from an .npz.

    Accepts either ``sigma_bl`` (n_bl,) directly, or ``s_ant`` (n_ant,) from which
    ``sigma_bl = s_p * s_q`` is rebuilt -- so a noise table measured on one subset can be
    applied to another with a different baseline ordering.
    """
    d = np.load(path)
    if "sigma_bl" in d and len(d["sigma_bl"]) == n_bl:
        return np.asarray(d["sigma_bl"], dtype=float)
    if "s_ant" in d:
        s = np.asarray(d["s_ant"], dtype=float)
        return s[np.asarray(a1)] * s[np.asarray(a2)]
    raise ValueError(
        f"{path} has neither a matching 'sigma_bl' (n_bl={n_bl}) nor an 's_ant' array."
    )
