"""Gain-invariant (closure) residual diagnostics.

Everything an antenna-based gain can do to the visibilities lives in a small subspace:
to first order a gain error ``eps_p`` produces

    r_pq = (eps_p + conj(eps_q)) * V_pq

which is ``2 n_ant - 1`` real degrees of freedom (the overall phase is unobservable)
against ``n_bl`` visibilities. The ORTHOGONAL COMPLEMENT of that subspace is exactly
what closure quantities measure: it is untouched by any antenna gain error, in the data
or in the model.

That distinction is the point. A residual can be large for two very different reasons:

* an **antenna-based error** -- a wrong gain, or a gain biased by a wrong noise model
  (which is real: fitting EDA2 with a scalar noise inflated the gain-amplitude spread
  from 1.6x to 3.3x);
* a **model error** -- the sky or RFI model is actually wrong.

Splitting the residual into those two orthogonal parts tells you which. The matched
filter cannot do this: it needs the gain in its template, so a gain error and a model
error both show up the same way.

Two functions:

* :func:`gain_subspace_split` -- the rigorous, linear version. Weighted least squares of
  the residual onto the gain subspace, then chi^2 of what is left. Preferred: it has no
  low-SNR phase bias, and the noise propagates exactly.
* :func:`closure_phase_residual` -- the classic triangle closure phase, for
  interpretability. Needs |V|/sigma >~ 3 per baseline, so on EDA2 (median per-visibility
  SNR 1.36) it must be run on frequency-averaged visibilities and restricted to the
  higher-SNR baselines.
"""

import numpy as np
from numpy.typing import NDArray


def gain_subspace_split(
    res: NDArray,
    vis_model: NDArray,
    a1: NDArray,
    a2: NDArray,
    sigma_bl: NDArray,
    flags: NDArray = None,
    n_ant: int = None,
):
    """Split a residual into the antenna-gain subspace and its (closure) complement.

    Fits ``r_pq ~ (a_p + a_q) V_pq + i (b_p - b_q) V_pq`` by weighted least squares --
    a *constant* (in freq and time) complex gain error, matching ConstGains' own freedom.
    The amplitude (a) and phase (b) blocks turn out to be exactly orthogonal, and each
    reduces to a 256x256 graph-Laplacian-like solve, so this is cheap.

    Returns
    -------
    dict
        ``rchi2_total``  reduced chi^2 of the raw residual
        ``rchi2_closure`` reduced chi^2 of the gain-immune (closure) part -- the honest
                          test of the sky/RFI model, independent of any gain error
        ``dchi2_per_dof`` chi^2 removed by the best antenna gain error, per degree of
                          freedom. ~1 = nothing an antenna gain could fix (good).
                          >> 1 = a real antenna-based error remains.
    """
    res = np.asarray(res)
    V = np.asarray(vis_model)
    a1, a2 = np.asarray(a1), np.asarray(a2)
    n_ant = int(max(a1.max(), a2.max())) + 1 if n_ant is None else n_ant

    w = np.ones(res.shape) if flags is None else (~np.asarray(flags, bool)).astype(float)
    w = w * (a1 != a2)[:, None, None]
    w = w / (np.asarray(sigma_bl)[:, None, None] ** 2)  # 1/sigma^2

    # Per-baseline aggregates over all (freq, time) cells.
    Wb = np.nansum(w * np.abs(V) ** 2, axis=(1, 2))                    # (n_bl,)
    Gb = np.nansum(w * (np.conj(V) * res).real, axis=(1, 2))
    Hb = np.nansum(w * (np.conj(V) * res).imag, axis=(1, 2))

    # Amplitude block: M[p,p] = sum_{bl ~ p} Wb ;  M[p,q] = +Wb(pq)
    # Phase block:     M[p,p] = sum_{bl ~ p} Wb ;  M[p,q] = -Wb(pq)   (singular: overall phase)
    deg = np.zeros(n_ant)
    np.add.at(deg, a1, Wb)
    np.add.at(deg, a2, Wb)
    Ma = np.diag(deg).copy()
    Mb = np.diag(deg).copy()
    np.add.at(Ma, (a1, a2), Wb)
    np.add.at(Ma, (a2, a1), Wb)
    np.add.at(Mb, (a1, a2), -Wb)
    np.add.at(Mb, (a2, a1), -Wb)

    ra = np.zeros(n_ant)
    np.add.at(ra, a1, Gb)
    np.add.at(ra, a2, Gb)
    rb = np.zeros(n_ant)
    np.add.at(rb, a1, Hb)
    np.add.at(rb, a2, -Hb)

    # dchi2 = rhs^T M^+ rhs (pseudo-inverse: the phase block has the overall-phase null space)
    xa = np.linalg.lstsq(Ma, ra, rcond=None)[0]
    xb = np.linalg.lstsq(Mb, rb, rcond=None)[0]
    dchi2 = float(ra @ xa + rb @ xb)

    chi2_total = float(np.nansum(w * np.abs(res) ** 2))
    n_data = 2.0 * float(np.nansum(w > 0))          # real + imag per visibility
    dof_gain = 2 * n_ant - 1

    return dict(
        rchi2_total=chi2_total / n_data,
        rchi2_closure=(chi2_total - dchi2) / (n_data - dof_gain),
        dchi2_per_dof=dchi2 / dof_gain,
        chi2_total=chi2_total,
        dchi2=dchi2,
        dof_gain=dof_gain,
        n_data=n_data,
    )


def closure_phase_residual(
    vis_obs: NDArray,
    vis_model: NDArray,
    a1: NDArray,
    a2: NDArray,
    sigma_bl: NDArray,
    flags: NDArray = None,
    n_tri: int = 20000,
    snr_min: float = 3.0,
    seed: int = 0,
):
    """Classic triangle closure phase of the residual: arg(V_obs) - arg(V_model).

    Closure phase ``Phi_pqr = arg(V_pq) + arg(V_qr) + arg(V_rp)`` is identically invariant
    to antenna gain phases -- they cancel round the triangle. So ``dPhi = Phi_obs -
    Phi_model`` tests the sky/RFI model with no sensitivity to any gain-phase error.

    Visibilities are averaged over frequency first: EDA2's per-visibility SNR is ~1.4, and
    closure phase is noise-dominated (and biased) below SNR ~ 3. Triangles are kept only
    when all three baselines exceed ``snr_min``.

    ``z = dPhi / sigma_Phi`` with ``sigma_Phi^2 = sum_3 (sigma_bl/|V|)^2``, so z ~ N(0,1)
    if the model is right. Returns the coverage of |z| <= 3 (null: 99.73%).
    """
    Vo, Vm = np.asarray(vis_obs), np.asarray(vis_model)
    a1, a2 = np.asarray(a1), np.asarray(a2)
    n_ant = int(max(a1.max(), a2.max())) + 1

    ok = np.ones(Vo.shape, bool) if flags is None else ~np.asarray(flags, bool)
    with np.errstate(invalid="ignore"):
        Vo_f = np.nanmean(np.where(ok, Vo, np.nan), axis=1)     # (n_bl, n_time)
        Vm_f = np.nanmean(np.where(ok, Vm, np.nan), axis=1)
    n_chan = Vo.shape[1]
    sig_f = np.asarray(sigma_bl) / np.sqrt(n_chan)              # (n_bl,)

    # baseline lookup (p<q) -> row
    idx = -np.ones((n_ant, n_ant), int)
    idx[a1, a2] = np.arange(len(a1))
    idx[a2, a1] = np.arange(len(a1))

    rng = np.random.default_rng(seed)
    tri = rng.choice(n_ant, size=(n_tri * 3, 3), replace=True)
    tri = tri[(tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])]
    tri = np.unique(np.sort(tri, axis=1), axis=0)[:n_tri]
    p, q, r = tri[:, 0], tri[:, 1], tri[:, 2]
    i_pq, i_qr, i_pr = idx[p, q], idx[q, r], idx[p, r]
    good = (i_pq >= 0) & (i_qr >= 0) & (i_pr >= 0)
    p, q, r, i_pq, i_qr, i_pr = (x[good] for x in (p, q, r, i_pq, i_qr, i_pr))

    def closure(V):   # Phi = arg(V_pq) + arg(V_qr) - arg(V_pr)   (V_rp = conj(V_pr))
        return np.angle(V[i_pq] * V[i_qr] * np.conj(V[i_pr]))

    Po, Pm = closure(Vo_f), closure(Vm_f)
    dphi = np.angle(np.exp(1j * (Po - Pm)))

    snr = np.abs(Vm_f) / sig_f[:, None]
    keep = (snr[i_pq] > snr_min) & (snr[i_qr] > snr_min) & (snr[i_pr] > snr_min)
    sig_phi = np.sqrt(
        1 / np.maximum(snr[i_pq], 1e-9) ** 2
        + 1 / np.maximum(snr[i_qr], 1e-9) ** 2
        + 1 / np.maximum(snr[i_pr], 1e-9) ** 2
    )
    z = dphi / sig_phi
    z = z[keep & np.isfinite(z)]

    return dict(
        coverage=float(np.mean(np.abs(z) <= 3.0)) if z.size else float("nan"),
        rms_z=float(np.std(z)) if z.size else float("nan"),
        n_used=int(z.size),
        n_tri=int(len(p)),
    )
