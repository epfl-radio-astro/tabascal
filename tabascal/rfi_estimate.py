"""Matched-filter light-curve extraction for RFI sources.

Given a set of satellite trajectories (currently supplied as TLEs) and the
observed visibilities, this module beam-forms the interferometer toward each
satellite and reads off its per-timestep (and per-channel) flux.  This is a
*matched filter* in visibility space: for a point source moving along a known
trajectory the RFI contribution to baseline ``(a1, a2)`` is

    V_rfi[bl] = A_a1 conj(A_a2) exp(i (phi_a1 - phi_a2))

where ``phi_a`` is the geometric phase at each antenna (``get_rfi_phase``).  The
per-baseline template is the unit-modulus steering vector
``s_bl = exp(i (phi_a1 - phi_a2))``.  The maximum-likelihood amplitude of a
signal ``s`` observed as ``V`` in white noise is ``<s, V> / <s, s>``, so the
matched-filter estimate of the source visibility at each (freq, time) is the
de-rotated, baseline-averaged data

    S_hat[f, t] = sum_bl w_bl V_bl exp(-i (phi_a1 - phi_a2)) / sum_bl w_bl .

The satellite fringe adds coherently after de-rotation while the sky and noise
add incoherently, so ``S_hat`` isolates the RFI source visibility.  Its
magnitude is a per-antenna power estimate; ``sqrt(|S_hat|)`` is the per-antenna
amplitude used to seed tabascal's RFI signal model.

Two entry points are provided:

* :func:`extract_light_curves_from_ms` -- the standalone tool.  Point it at any
  MS column and a set of NORAD IDs (or ready-made TLEs) and it returns / saves
  the light curves.  Wrapped by the ``tabascal light-curve`` CLI subcommand.
* :func:`light_curves_from_config` -- the in-process path.  Reuses the arrays a
  :class:`~tabascal.config.TabConfig` has already loaded (visibilities,
  antennas, times, TLEs) so tabascal can build an initial RFI estimate without
  touching the MS again.

Both share the pure core :func:`matched_filter_light_curves`.

Only the TLE trajectory source is implemented for now; RA/Dec and Alt/Az
pointings can be added by constructing ``rfi_xyz`` from those and feeding
:func:`rfi_phase_from_positions`.
"""

from typing import Optional

import os

import numpy as np
from numpy.typing import NDArray

from tabascal.interferometry import get_rfi_phase_numpy, itrf_to_uvw_numpy
from tabascal.components.trajectory import (
    get_satellite_positions,
    itrs_to_gcrs_sf,
    fetch_orbital_elements,
)
from tabascal.time import gast_deg, mjd_to_jd, jd_to_mjd


# ---------------------------------------------------------------------------
# Geometry: per-antenna RFI phase from a trajectory
# ---------------------------------------------------------------------------

def rfi_phase_from_positions(
    rfi_xyz: NDArray,
    ants_itrf: NDArray,
    times_jd: NDArray,
    phase_centre: dict,
    freqs: NDArray,
) -> NDArray:
    """Per-antenna geometric phase for RFI sources at known ECI positions.

    Numpy/f64 host-side computation mirroring
    :meth:`tabascal.components.trajectory.FixedOrbit._compute_rfi_phase`.

    Parameters
    ----------
    rfi_xyz : Array (n_src, n_time, 3)
        Source positions over time in the ECI (GCRF) frame in metres.
    ants_itrf : Array (n_ant, 3)
        Antenna positions in the ITRF (ECEF) frame in metres.
    times_jd : Array (n_time,)
        Observation times in Julian date.
    phase_centre : dict
        ``{"ra": <deg>, "dec": <deg>}`` phase centre of the visibilities.
    freqs : Array (n_freq,)
        Channel frequencies in Hz.

    Returns
    -------
    Array (n_src, n_ant, n_freq, n_time)
        Geometric phase at each antenna for each source.
    """
    times_jd = np.asarray(times_jd)
    freqs = np.asarray(freqs)

    gsa = gast_deg(times_jd)  # GAST in degrees (UTC convention)
    gh0 = (gsa - phase_centre["ra"]) % 360

    ants_uvw = np.transpose(
        itrf_to_uvw_numpy(ants_itrf, gh0, phase_centre["dec"]), axes=(1, 0, 2)
    )  # (n_ant, n_time, 3)
    ants_xyz = itrs_to_gcrs_sf(ants_itrf, times_jd)  # (n_ant, n_time, 3)

    return get_rfi_phase_numpy(np.asarray(rfi_xyz), ants_uvw, ants_xyz, freqs)


def rfi_phase_from_tles(
    tles: NDArray,
    ants_itrf: NDArray,
    times_jd: NDArray,
    phase_centre: dict,
    freqs: NDArray,
) -> NDArray:
    """Per-antenna geometric phase for satellites given their TLEs.

    Propagates each TLE over ``times_jd`` and defers to
    :func:`rfi_phase_from_positions`.

    Parameters
    ----------
    tles : Array (n_src, 2)
        TLE line pairs (line1, line2) per satellite.
    ants_itrf, times_jd, phase_centre, freqs
        See :func:`rfi_phase_from_positions`.

    Returns
    -------
    Array (n_src, n_ant, n_freq, n_time)
    """
    times_jd = np.asarray(times_jd)
    rfi_xyz = np.asarray(get_satellite_positions(tles, list(times_jd)))
    return rfi_phase_from_positions(rfi_xyz, ants_itrf, times_jd, phase_centre, freqs)


# ---------------------------------------------------------------------------
# The matched filter
# ---------------------------------------------------------------------------

def matched_filter_light_curves(
    vis: NDArray,
    rfi_phase: NDArray,
    a1: NDArray,
    a2: NDArray,
    flags: Optional[NDArray] = None,
    exclude_autos: bool = True,
    max_mem_gb: float = 1.0,
    return_error: bool = False,
    ant_gain: Optional[NDArray] = None,
):
    """Beam-form the data toward each source to estimate its source visibility.

    The per-baseline model of a source of visibility ``S`` is

    ``vis_bl = T_bl * S + noise``,   ``T_bl = g_a1 conj(g_a2) exp(i(phi_a1 - phi_a2))``

    with ``phi`` the geometric phase and ``g_a`` the complex per-antenna gain. The
    least-squares (inverse-variance) estimate of ``S`` is therefore

    ``S_hat[f, t] = sum_bl w_bl conj(T_bl) vis_bl / sum_bl w_bl |T_bl|^2``

    where ``w_bl`` excludes flagged (and, by default, autocorrelation) samples, and
    ``|T_bl|^2 = |g_a1|^2 |g_a2|^2``.

    Note this *down-weights* low-gain baselines. Dividing the data by the gain
    (``vis / T``) instead would up-weight exactly those noisiest baselines and
    inflate the noise -- the estimator above is the correct way to apply a gain.
    With ``ant_gain = None`` (g = 1) it reduces to the plain de-rotated baseline
    average, i.e. the original behaviour.

    Evaluated in numpy/f64 (a one-shot host-side estimate) with an inner
    time-chunk loop so the ``(n_bl, n_freq, chunk)`` template never exceeds
    ``max_mem_gb``.

    Parameters
    ----------
    vis : Array (n_bl, n_freq, n_time) complex
        Observed visibilities (any MS data column).
    rfi_phase : Array (n_src, n_ant, n_freq, n_time)
        Per-antenna geometric phase (see :func:`rfi_phase_from_positions`).
    a1, a2 : Array (n_bl,)
        Antenna indices for each baseline.
    flags : Array (n_bl, n_freq, n_time) bool, optional
        ``True`` marks samples to exclude from the average.
    exclude_autos : bool, default True
        Drop autocorrelation baselines (``a1 == a2``).
    max_mem_gb : float, default 1.0
        Approximate cap on the working-array size for the time-chunk loop.
    return_error : bool, default False
        Also return the per-cell standard error of the mean (see below).
    ant_gain : Array (n_ant,) complex, optional
        Fixed complex per-antenna gain ``g_a``. When None, unit gain.

    Returns
    -------
    Array (n_src, n_freq, n_time) complex
        Matched-filter source-visibility estimate.  ``nan`` where every
        baseline in a (freq, time) cell is flagged.
    Array (n_src, n_freq, n_time) real, optional
        Only if ``return_error``.  The standard error of the (real) flux
        estimate: ``sigma_hat / sqrt(sum_bl w |T|^2)``, the beam-former's noise
        floor.  This is the visibility-space equivalent of a dirty-image aperture
        standard deviation.
    """
    vis = np.asarray(vis)
    rfi_phase = np.asarray(rfi_phase)
    a1 = np.asarray(a1)
    a2 = np.asarray(a2)

    n_bl, n_freq, n_time = vis.shape
    n_src = rfi_phase.shape[0]
    n_ant = rfi_phase.shape[1]

    if ant_gain is None:
        g = np.ones(n_ant, dtype=np.complex128)
    else:
        g = np.asarray(ant_gain, dtype=np.complex128)
        if g.shape != (n_ant,):
            raise ValueError(f"ant_gain has shape {g.shape}, expected ({n_ant},)")

    # Per-baseline weight mask (n_bl, n_freq, n_time): 1 = keep, 0 = drop.
    w = np.ones(vis.shape, dtype=np.float64)
    if exclude_autos:
        w[a1 == a2] = 0.0
    if flags is not None:
        w = w * (~np.asarray(flags, dtype=bool))

    # |T_bl|^2 = |g_a1|^2 |g_a2|^2 -- the per-baseline template power, i.e. the
    # inverse-variance weight of that baseline's estimate of S.
    gsq = (np.abs(g[a1]) ** 2 * np.abs(g[a2]) ** 2)[:, None, None]  # (n_bl, 1, 1)
    g_bl_conj = (np.conjugate(g[a1]) * g[a2])[:, None, None]

    n_keep = np.sum(w, axis=0)                       # (n_freq, n_time) unflagged count
    den = np.sum(w * gsq, axis=0)                    # (n_freq, n_time) sum_bl w |T|^2
    safe_den = np.where(den == 0.0, np.nan, den)
    safe_gsq = np.where(gsq == 0.0, np.nan, gsq)

    # Choose a time-chunk that keeps the (n_bl, n_freq, chunk) complex template
    # within the memory budget.
    bytes_per_t = n_bl * n_freq * 16  # complex128
    time_chunk = max(1, int(max_mem_gb * 1e9 / max(bytes_per_t, 1)))

    out = np.empty((n_src, n_freq, n_time), dtype=np.complex128)
    sumsq = np.empty((n_src, n_freq, n_time), dtype=np.float64) if return_error else None

    for s in range(n_src):
        # E_a = exp(-i phi_a); geometric part of conj(T) = E_a1 * conj(E_a2)
        E = np.exp(-1.0j * rfi_phase[s])  # (n_ant, n_freq, n_time)
        for t0 in range(0, n_time, time_chunk):
            t1 = min(t0 + time_chunk, n_time)
            wc = w[:, :, t0:t1]
            vc = np.where(wc > 0.0, vis[:, :, t0:t1], 0.0)
            # conj(T_bl) = conj(g_a1) g_a2 exp(-i(phi_a1 - phi_a2))
            templ_conj = g_bl_conj * E[a1][:, :, t0:t1] * np.conjugate(E[a2][:, :, t0:t1])
            out[s, :, t0:t1] = np.sum(wc * vc * templ_conj, axis=0)
            if return_error:
                # y_bl = vis_bl / T_bl : each baseline's own (unbiased) estimate of S,
                # with variance sigma^2 / |T|^2. Accumulate sum_bl w |T|^2 Re(y)^2.
                y = vc * templ_conj / safe_gsq
                sumsq[s, :, t0:t1] = np.nansum(wc * gsq * y.real ** 2, axis=0)

    S_hat = out / safe_den[None]

    if not return_error:
        return S_hat

    # Weighted sample variance: u_bl = w |T|^2, and E[u (Re y - m)^2] = sigma_re^2, so
    #   sigma_re^2 = sum_bl u (Re y - m)^2 / (N - 1) = (sumsq - den * Re(S_hat)^2) / (N-1)
    # and the standard error of the weighted mean is sigma_re / sqrt(sum_bl u).
    # With g = 1 this is exactly std_bl(Re x) / sqrt(N), the original expression.
    N = n_keep[None]
    ss = np.maximum(sumsq - den[None] * S_hat.real ** 2, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma_re_sq = ss / (N - 1)
        err = np.sqrt(sigma_re_sq / safe_den[None])
    return S_hat, err


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def _lc_result(light_curves, norad_ids, freqs, times_mjd, data_col, corr,
               error=None, extra=None):
    times_mjd = np.asarray(times_mjd)
    if norad_ids is not None:
        titles = [str(int(n)) for n in norad_ids]
    else:
        titles = [f"src{i}" for i in range(light_curves.shape[0])]
    result = {
        "light_curves": light_curves,          # (n_src, n_freq, n_time) complex
        "norad_ids": list(norad_ids) if norad_ids is not None else None,
        "titles": titles,
        "freqs": np.asarray(freqs),
        "times_mjd": times_mjd,
        "times_sec": (times_mjd - times_mjd[0]) * 86400.0,
        "data_col": data_col,
        "corr": corr,
    }
    if error is not None:
        error = np.asarray(error)                      # (n_src, n_freq, n_time)
        result["error"] = error
        # z-statistic = coherent flux / noise floor, per (src, freq, time). The
        # de-rotated RFI is real, so Re(S_hat) carries the (residual) signal and
        # error is its standard deviation -> z ~ N(0, 1) where nothing is left.
        with np.errstate(invalid="ignore", divide="ignore"):
            result["z"] = light_curves.real / error
    if extra:
        result.update(extra)
    return result


def extract_light_curves_from_ms(
    ms_path: str,
    norad_ids: Optional[list] = None,
    tles: Optional[NDArray] = None,
    corr: str = "xx",
    data_col: str = "DATA",
    freq: Optional[float] = None,
    exclude_autos: bool = True,
    extra_tle_dir: Optional[str] = None,
    max_mem_gb: float = 1.0,
    ant_gain: Optional[NDArray] = None,
) -> dict:
    """Extract matched-filter RFI light curves from any column of an MS.

    Standalone entry point (used by the ``tabascal light-curve`` CLI).  Reads the
    requested ``data_col``, propagates the satellites' TLEs over the MS times and
    runs :func:`matched_filter_light_curves`.

    Parameters
    ----------
    ms_path : str
        Path to the Measurement Set.
    norad_ids : list[int], optional
        NORAD catalogue IDs; their TLEs are fetched (locally or via Space-Track,
        matched to the mean observation epoch).  Required unless ``tles`` given.
    tles : Array (n_src, 2), optional
        Ready-made TLE line pairs, used instead of fetching by ``norad_ids``.
    corr : str, default "xx"
        Correlation to read (``xx``/``xy``/``yx``/``yy``).
    data_col : str, default "DATA"
        MS data column to matched-filter (e.g. ``REAL_DATA``,
        ``REAL_DATA_FLUXCAL``, ``TAB_RES_DATA``).
    freq : float, optional
        If given, use only the single channel nearest this frequency (Hz).
    exclude_autos : bool, default True
        Drop autocorrelations from the beam-former.
    extra_tle_dir : str, optional
        Extra local directory searched for cached TLEs before Space-Track.
    max_mem_gb : float, default 1.0
        Memory budget for the matched-filter time-chunk loop.

    Returns
    -------
    dict
        See :func:`_lc_result`.  ``light_curves`` is ``(n_src, n_freq, n_time)``
        complex, ordered to match ``norad_ids``/``titles``.
    """
    from tabascal.tab_tools import read_ms

    ms = read_ms(ms_path, freq, None, corr, data_col)
    times_jd = mjd_to_jd(np.asarray(ms["times_mjd"]))
    phase_centre = {"ra": float(ms["ra"]), "dec": float(ms["dec"])}

    if tles is None:
        if not norad_ids:
            raise ValueError("Provide either `norad_ids` or `tles`.")
        obs_epoch_jd = float(times_jd.mean())
        _, _, norad_ids, tles = fetch_orbital_elements(
            obs_epoch_jd, list(norad_ids), extra_tle_dir=extra_tle_dir
        )

    rfi_phase = rfi_phase_from_tles(
        tles, ms["ants_itrf"], times_jd, phase_centre, ms["freqs"]
    )
    light_curves, error = matched_filter_light_curves(
        ms["vis_obs"],
        rfi_phase,
        ms["a1"],
        ms["a2"],
        flags=ms["flags"],
        exclude_autos=exclude_autos,
        max_mem_gb=max_mem_gb,
        return_error=True,
        ant_gain=ant_gain,
    )
    return _lc_result(
        light_curves, norad_ids, ms["freqs"], ms["times_mjd"], data_col, corr,
        error=error,
    )


def extract_light_curves_from_zarr(
    ms_path: str,
    zarr_path: str,
    norad_ids: Optional[list] = None,
    corr: str = "xx",
    data_col: str = "DATA",
    freq: Optional[float] = None,
    exclude_autos: bool = True,
    extra_tle_dir: Optional[str] = None,
    max_mem_gb: float = 1.0,
    ant_gain: Optional[NDArray] = None,
) -> dict:
    """Matched-filter the residual of a tabascal run, taken straight from its zarr.

    **This is the preferred way to score a run.** The MS result columns
    (``TAB_RES_DATA`` et al.) are overwritten by *every* tabascal run, so scoring off
    the MS is only valid if the columns happen to belong to the run you mean. The zarr
    is written once per run and per suffix, so it cannot be invalidated by a later run.

    The residual is formed as ``data_col - zarr.vis_obs``. The zarr's ``vis_obs`` is
    the model's own *gained* prediction (``apply_gains(gains, vis_ast + vis_rfi)``), so
    this is exactly the residual ``write_results_ms`` would write -- without the MS
    round-trip. (Verified equal to the MS path; they differ only by fp32 rounding,
    because write_results_ms forms ``g*ast + g*rfi`` while the zarr stores
    ``g*(ast + rfi)``.)

    The gain template is taken from the same zarr's fitted ``gains`` unless
    ``ant_gain`` overrides it, so the filter is automatically matched to the run being
    scored and there is no separate gain file to fall out of sync.
    """
    import xarray as xr
    from tabascal.tab_tools import read_ms

    z = xr.open_zarr(zarr_path)
    ms = read_ms(ms_path, freq, None, corr, data_col)

    model = np.asarray(z.vis_obs.isel(sample=0).data.compute())   # gains * (ast + rfi)
    res = np.asarray(ms["vis_obs"]) - model

    if ant_gain is None and "gains" in z:
        # The template must represent the TRUE instrument, not the model under test.
        # Scoring a run with its own fitted gains is circular: a run that assumed g = 1
        # then gets a unit template, which is mismatched to the real (non-unit) gain,
        # loses SNR, and cannot see its own error. That is not a fair comparison --
        # measured directly: the known-bad ConstAnt g=1 run scores -2.1pp excess with a
        # unit template (looks perfect) but +16.1pp with the correct one.
        # So: fall back to the run's gains only as a convenience, and warn loudly.
        g = np.asarray(z.gains.isel(sample=0).data.mean(axis=(1, 2)).compute())
        if not np.allclose(g, 1.0):
            ant_gain = g / np.median(np.abs(g))
            print(f"  Gain     : from the run's own zarr (|g| p10/p90 "
                  f"{np.percentile(np.abs(ant_gain), 10):.2f}/"
                  f"{np.percentile(np.abs(ant_gain), 90):.2f}, phase std "
                  f"{np.rad2deg(np.std(np.angle(ant_gain))):.1f} deg)")
            print("  WARNING  : gain taken from the run being scored. To COMPARE runs, "
                  "pass the same -g/--ant-gain to all of them.")
        else:
            print("  WARNING  : this run has unit gains, so the filter is UN-GAINED and "
                  "less sensitive (it cannot see a residual structured by the real "
                  "antenna gain). Pass -g/--ant-gain with the best known instrument "
                  "gain -- without it a bad run can score as clean.")

    times_jd = mjd_to_jd(np.asarray(ms["times_mjd"]))
    tles = [tuple(t) for t in z.attrs["tles"]] if "tles" in z.attrs else None
    if tles is None:
        if not norad_ids:
            raise ValueError(
                f"{zarr_path} has no 'tles' attr (run with data.save_rfi_A: true) -- "
                f"provide NORAD IDs so the TLEs can be fetched."
            )
        _, _, norad_ids, tles = fetch_orbital_elements(
            float(times_jd.mean()), list(norad_ids), extra_tle_dir=extra_tle_dir
        )
    elif norad_ids is None:
        norad_ids = [int(n) for n in z.norad_id.values] if "norad_id" in z else None

    rfi_phase = rfi_phase_from_tles(
        tles, ms["ants_itrf"], times_jd, {"ra": float(ms["ra"]), "dec": float(ms["dec"])},
        ms["freqs"],
    )
    light_curves, error = matched_filter_light_curves(
        res, rfi_phase, ms["a1"], ms["a2"], flags=ms["flags"],
        exclude_autos=exclude_autos, max_mem_gb=max_mem_gb, return_error=True,
        ant_gain=ant_gain,
    )
    return _lc_result(
        light_curves, norad_ids, ms["freqs"], ms["times_mjd"],
        f"{data_col} - {os.path.basename(zarr_path.rstrip('/'))}", corr, error=error,
    )


def load_ant_gain(path: str) -> NDArray:
    """Complex per-antenna gain g_a from an .npz with a 'gain' key, shape (n_ant,).

    Produced by eda2/analysis/rfi_amplitudes/export_antenna_gain.py, or by any
    tabascal run using gains:ConstGains (its fitted `gains` are in the results zarr).
    """
    g = np.asarray(np.load(os.path.abspath(path))["gain"])
    print(f"  Gain     : {path}  (|g| median {np.median(np.abs(g)):.3f}, "
          f"phase std {np.rad2deg(np.std(np.angle(g))):.1f} deg)")
    return g


def light_curves_from_config(
    tab_config,
    vis: Optional[NDArray] = None,
    exclude_autos: bool = True,
    max_mem_gb: float = 1.0,
) -> dict:
    """Matched-filter light curves from an already-loaded :class:`TabConfig`.

    In-process path for tabascal: reuses the visibilities, antenna positions,
    times and TLEs the config has loaded, so no second MS read is needed.  Use
    to build an initial estimate of the RFI signals.

    Parameters
    ----------
    tab_config : tabascal.config.TabConfig
        A configured object exposing ``vis_obs``, ``flags``, ``ants_itrf``,
        ``times_jd``, ``freqs``, ``phase_centre``, ``a1``, ``a2``, ``tles`` and
        ``norad_ids``.
    vis : Array (n_bl, n_freq, n_time), optional
        Visibilities to filter; defaults to ``tab_config.vis_obs``.
    exclude_autos : bool, default True
    max_mem_gb : float, default 1.0

    Returns
    -------
    dict
        See :func:`_lc_result`.
    """
    if vis is None:
        vis = tab_config.vis_obs
    times_jd = np.asarray(tab_config.times_jd)

    rfi_phase = rfi_phase_from_tles(
        tab_config.tles,
        tab_config.ants_itrf,
        times_jd,
        tab_config.phase_centre,
        np.asarray(tab_config.freqs),
    )
    flags = getattr(tab_config, "flags", None)
    flags = None if flags is None else np.asarray(flags)

    light_curves, error = matched_filter_light_curves(
        np.asarray(vis),
        rfi_phase,
        np.asarray(tab_config.a1),
        np.asarray(tab_config.a2),
        flags=flags,
        exclude_autos=exclude_autos,
        max_mem_gb=max_mem_gb,
        return_error=True,
    )
    return _lc_result(
        light_curves,
        tab_config.norad_ids,
        np.asarray(tab_config.freqs),
        jd_to_mjd(times_jd),
        tab_config.args["data"]["data_col"],
        tab_config.args["data"]["corr"],
        error=error,
    )


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_light_curves_npz(path: str, result: dict) -> None:
    """Save a light-curve result to ``.npz``.

    Writes the native complex matched-filter output plus a
    ``(n_src, n_time, 2)`` real ``light_curves`` array laid out like
    ``nufft-gif``'s (slot 0 = frequency-coherent ``|mean_f S_hat|``, slot 1 =
    ``max_f |S_hat|``) so the same downstream tooling (and NORAD-``titles``
    matching) can consume it.

    Parameters
    ----------
    path : str
        Output ``.npz`` path.
    result : dict
        A dict from one of the driver functions.
    """
    lc = np.asarray(result["light_curves"])           # (n_src, n_freq, n_time) complex
    mag = np.abs(lc)                                    # per-channel |S_hat|
    coherent = np.abs(np.nanmean(lc, axis=1))          # (n_src, n_time) freq-coherent
    peak = np.nanmax(mag, axis=1)                       # (n_src, n_time) max over freq
    light_curves_2 = np.stack([coherent, peak], axis=-1)  # (n_src, n_time, 2)

    freqs = np.asarray(result["freqs"], dtype=float)
    arrays = dict(
        light_curves=light_curves_2,
        light_curves_mf=lc,                            # (n_src, n_freq, n_time) complex
        titles=np.array(result["titles"]),
        norad_ids=np.array(result["norad_ids"]) if result.get("norad_ids") else np.array([]),
        chan_freq_mhz=freqs / 1e6,
        times_sec=np.asarray(result["times_sec"], dtype=float),
        times_mjd=np.asarray(result["times_mjd"], dtype=float),
        data_col=str(result["data_col"]),
        corr=str(result["corr"]),
    )
    if "error" in result:
        arrays["error"] = np.asarray(result["error"])   # (n_src, n_freq, n_time)
        arrays["z"] = np.asarray(result["z"])           # (n_src, n_freq, n_time)
    np.savez(path, **arrays)


# ---------------------------------------------------------------------------
# z-statistic (residual / floor): coverage + spectrograms
# ---------------------------------------------------------------------------

def coverage_stats(result: dict, z_crit: float = 3.0) -> dict:
    """Fraction of time-frequency cells consistent with noise, per source.

    The z-statistic ``z = Re(S_hat) / floor`` would be ~ N(0, 1) wherever nothing is
    left after subtraction, so a well-cleaned source has |z| within ``z_crit`` almost
    everywhere.  ``coverage`` is the fraction of finite (freq, time) cells with
    ``|z| <= z_crit``; ``max_z`` is the peak residual significance.

    **Compare against ``null_coverage``, not against the analytic 2*Phi(z)-1.** The
    floor is the standard error of the mean over baselines, which assumes the
    de-rotated per-baseline samples are independent.  They are not: residual sky is
    coherent across baselines, so the floor is optimistic (empirically by ~1.5x) and
    the analytic null badly over-states the expected coverage.

    ``null_coverage`` is the same statistic computed on ``Im(S_hat)/floor``.  After
    de-rotation a real source sits purely in the real part, so the imaginary part is a
    matched, source-free null carrying exactly the same noise and the same correlation
    structure.  A source is consistent with noise when its coverage is not
    significantly *below* the null; the *excess* ``null_coverage - coverage`` is the
    part attributable to a real residual.

    Parameters
    ----------
    result : dict
        A driver result carrying ``z`` (i.e. built with the error).
    z_crit : float, default 3.0
        Detection threshold; cells with |z| above it are flagged as residual.

    Returns
    -------
    dict
        ``per_source`` (title, coverage, null_coverage, excess, max_z, n_cells) and
        ``overall`` (pooled coverage, null, worst source, mean coverage, z_crit).
    """
    if "z" not in result:
        raise ValueError("result has no 'z'; build it via a driver that returns the error.")
    z = np.asarray(result["z"])
    titles = result["titles"]

    # Matched source-free null: the imaginary part of the same beam-formed estimate.
    # In memory the complex estimate is under "light_curves"; save_light_curves_npz
    # renames it to "light_curves_mf" (and puts a real, nufft-gif-shaped array under
    # "light_curves"), so accept either.
    z_null = None
    S = result.get("light_curves_mf", result.get("light_curves"))
    if S is not None and "error" in result and np.iscomplexobj(np.asarray(S)):
        with np.errstate(invalid="ignore", divide="ignore"):
            z_null = np.asarray(S).imag / np.asarray(result["error"])

    per_source, pooled_in, pooled_n, pooled_null_in = [], 0, 0, 0
    for i, title in enumerate(titles):
        zi = z[i][np.isfinite(z[i])]
        n = zi.size
        n_in = int(np.sum(np.abs(zi) <= z_crit))
        cov = (n_in / n) if n else float("nan")

        null_cov, n_null_in = float("nan"), 0
        if z_null is not None:
            zn = z_null[i][np.isfinite(z_null[i])]
            if zn.size:
                n_null_in = int(np.sum(np.abs(zn) <= z_crit))
                null_cov = n_null_in / zn.size

        per_source.append(dict(
            title=title,
            coverage=cov,
            null_coverage=null_cov,
            excess=(null_cov - cov) if np.isfinite(null_cov) else float("nan"),
            max_z=float(np.max(np.abs(zi))) if n else float("nan"),
            n_cells=int(n),
        ))
        pooled_in += n_in
        pooled_n += n
        pooled_null_in += n_null_in

    covs = [p["coverage"] for p in per_source]
    worst = min(per_source, key=lambda p: p["coverage"]) if per_source else None
    return dict(
        per_source=per_source,
        overall=dict(
            coverage=(pooled_in / pooled_n) if pooled_n else float("nan"),
            null_coverage=(pooled_null_in / pooled_n) if pooled_n and z_null is not None
            else float("nan"),
            mean_coverage=float(np.nanmean(covs)) if covs else float("nan"),
            worst_source=worst["title"] if worst else None,
            worst_coverage=worst["coverage"] if worst else float("nan"),
            z_crit=z_crit,
        ),
    )


def plot_z_spectrograms(result: dict, save_path: str, z_crit: float = 3.0,
                        vmax: Optional[float] = None) -> str:
    """Per-source spectrogram of the z-statistic (residual / floor).

    One panel per source, time on x, frequency (MHz) on y, colour = signed
    ``z = Re(S_hat) / floor`` on a diverging scale (blue = over-subtracted,
    red = under-subtracted residual).  Mirrors ``nufft-gif -mo perchan``'s
    spectrogram layout.  For single-channel data it degrades to a z-vs-time
    line plot with the +/- ``z_crit`` band shaded.

    Parameters
    ----------
    result : dict
        A driver result carrying ``z``.
    save_path : str
        Output PNG path.
    z_crit : float, default 3.0
        Marked as the +/- band edge (contour / shaded).
    vmax : float, optional
        Symmetric colour limit; defaults to ``max(2*z_crit, 99th pct |z|)``.

    Returns
    -------
    str
        ``save_path``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.asarray(result["z"])                      # (n_src, n_freq, n_time)
    titles = result["titles"]
    times = np.asarray(result["times_sec"], dtype=float)
    freqs_mhz = np.asarray(result["freqs"], dtype=float) / 1e6
    n_src, n_freq, _ = z.shape

    cov = coverage_stats(result, z_crit)["per_source"]
    if vmax is None:
        finite = z[np.isfinite(z)]
        vmax = max(2.0 * z_crit, float(np.nanpercentile(np.abs(finite), 99))) if finite.size else 2.0 * z_crit

    fig, axes = plt.subplots(n_src, 1, figsize=(10, 2.8 * n_src), squeeze=False)
    for i, ax in enumerate(axes[:, 0]):
        sub = f"cov(|z|<={z_crit:g})={cov[i]['coverage']*100:.1f}%   max|z|={cov[i]['max_z']:.1f}"
        if n_freq == 1:
            ax.plot(times, z[i, 0], color="C3", lw=0.9)
            ax.axhspan(-z_crit, z_crit, color="gray", alpha=0.2, lw=0)
            ax.axhline(0, color="0.5", lw=0.8)
            ax.set_ylabel("z = resid/floor")
        else:
            pcm = ax.pcolormesh(times, freqs_mhz, z[i], shading="nearest",
                                cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            cb = fig.colorbar(pcm, ax=ax, label="z = resid/floor")
            cb.ax.axhline(z_crit, color="k", lw=1.0, ls="--")
            cb.ax.axhline(-z_crit, color="k", lw=1.0, ls="--")
            ax.set_ylabel("Freq [MHz]")
        ax.set_title(f"{titles[i]} — {result.get('data_col', '')}   ({sub})", fontsize=9)
        ax.set_xlabel("Time [s]")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
