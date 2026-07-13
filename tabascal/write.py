from tabascal.timing import measure_runtime

from daskms import xds_from_ms, xds_from_table, xds_to_table

import os

import numpy as np

import xarray as xr
import dask.array as da
import dask


def rfi_vis_per_sat(vi_pred: dict, tab_config):
    """Per-satellite RFI visibility from the fitted forward model.

    The RFI visibility op sums over the satellite (``n_rfi``) axis, so the
    per-source contribution is recovered by evaluating the *same* forward op on
    one satellite at a time. Uses the fitted fine-grid ``rfi_A`` (amplitude) and
    the geometric ``rfi_phase`` carried in ``vi_pred`` -- so the per-source
    visibilities sum exactly back to ``vis_rfi`` -- with no re-fit and no GP
    re-evaluation (only the baseline op, which needs ``a1``/``a2`` from config).

    Returns
    -------
    vis_src : np.ndarray, complex64, shape (n_rfi, n_bl, n_freq, n_time)
    norad_ids : list[int]  -- satellite per ``src`` index (the ``n_rfi`` order)
    """
    import jax.numpy as jnp
    from tabascal.components.ffi.rfi_vis_op import RFIVisOp

    n_ant = tab_config.n_ant
    n_freq = tab_config.n_freq
    n_time = tab_config.n_time
    n_int_time = tab_config.n_int_time
    n_int_freq = tab_config.args["rfi"]["freq_int_samples"]

    op = RFIVisOp(n_ant, tab_config.a1, tab_config.a2)

    rfi_A = vi_pred["rfi_A"][0]          # (n_rfi, n_ant, n_freq_fine, n_time_fine)
    rfi_phase = vi_pred["rfi_phase"][0]
    n_rfi = rfi_A.shape[0]

    def _shape_for_op(x):
        # Mirror RiemannVisTimeFreqCalculationFFI.forward for a single satellite:
        # (1, n_ant, n_freq, n_int_freq, n_time, n_int_time) -> transpose
        # (n_ant, n_freq, n_time, n_rfi=1, n_int_freq, n_int_time).
        x = x[None].reshape(1, n_ant, n_freq, n_int_freq, n_time, n_int_time)
        return jnp.transpose(x, (1, 2, 4, 0, 3, 5))

    vis_src = np.empty((n_rfi, tab_config.n_bl, n_freq, n_time), dtype=np.complex64)
    for r in range(n_rfi):
        vis_r = op.eval(_shape_for_op(rfi_A[r]), _shape_for_op(rfi_phase[r]))
        vis_src[r] = np.asarray(vis_r).astype(np.complex64)

    return vis_src, [int(n) for n in tab_config.norad_ids]


@measure_runtime
def write_per_sat_rfi_ms(ms_path: str, results_zarr_path: str, prefix: str = "TAB_RFI_"):
    """Write each satellite's RFI visibility prediction to its own MS column.

    Reads ``rfi_vis_src`` (dims ``sample, src, bl, freq, time``) from a results
    zarr written with ``save_rfi_per_sat`` and assigns one column per satellite,
    named ``<prefix><NORAD_ID>`` (e.g. ``TAB_RFI_58126``). Re-runnable: needs only
    the zarr and the MS, no re-fit. Image a column with ``nufft-gif`` to inspect a
    single satellite's modelled RFI for astronomical-signal contamination.
    """
    xds_ms = xds_from_ms(ms_path)[0]
    xds_tab = xr.open_zarr(results_zarr_path)

    if "rfi_vis_src" not in xds_tab:
        raise ValueError(
            f"{results_zarr_path} has no 'rfi_vis_src' -- re-run tabascal with "
            f"data.save_rfi_per_sat: true to produce per-satellite RFI visibilities."
        )

    dims = ["row", "chan", "corr"]
    chunks = {k: v for k, v in xds_ms.chunks.items() if k in dims}

    src = xds_tab.rfi_vis_src.data.astype(np.complex64).mean(axis=0)  # (src, bl, freq, time)
    n_src, n_bl, n_freq, n_time = src.shape
    n_corr = 1
    norad_ids = [int(n) for n in xds_tab.norad_id.values]

    cols = []
    for i, nid in enumerate(norad_ids):
        vis = da.transpose(src[i], (2, 0, 1)).reshape(-1, n_freq, n_corr)
        col = f"{prefix}{nid}"
        xds_ms = xds_ms.assign(**{col: xr.DataArray(vis, dims=dims).chunk(chunks)})
        cols.append(col)

    col_keywords = {col: {"UNIT": "Jy"} for col in cols}
    print(f"Writing per-satellite RFI predictions to {cols} columns in MS file.")
    dask.compute(xds_to_table([xds_ms], ms_path, cols, column_keywords=col_keywords))


@measure_runtime
def write_results_ms(
    ms_path: str,
    results_zarr_path: str,
    data_col: str = "DATA",
    gain_table: str | None = None,
):
    """Write a run's results back into the MS, all in the CALIBRATED frame.

    Every column written here -- CORRECTED_DATA and the TAB_* models and residuals --
    lives in one frame: the data with all gains divided out. That is what makes the
    WEIGHT column meaningful for all of them, and it makes
    ``TAB_AST_DATA + TAB_RFI_DATA + TAB_RES_DATA == CORRECTED_DATA`` hold exactly.

    There are two gain layers and both are removed:

    * ``gain_table`` -- the external calibration (e.g. from ``flux-calibrate
      --gain-table``) that ``data.gain_table`` divided out at read time. ``data_col`` in
      the MS is still RAW, so it has to be re-applied here or the models (which were fit
      to calibrated data) would be subtracted in the wrong frame.
    * the DIE gains fitted by the model, stored in the results zarr.

    Dividing by the gain makes the noise heteroscedastic -- low-gain baselines get noisier
    -- which is exactly why WEIGHT_SPECTRUM is written alongside: sigma_cal = SIGMA /
    |g_total|, so weight_cal = |g_total|^2 / SIGMA^2. Noise-referenced metrics must read
    the weights rather than assume a single sigma.
    """
    xds_ms = xds_from_ms(ms_path)[0]
    xds_tab = xr.open_zarr(results_zarr_path)

    dims = ["row", "chan", "corr"]
    chunks = {k: v for k, v in xds_ms.chunks.items() if k in dims}

    if xds_tab.ast_vis.data.ndim == 3:
        vis_ast = xds_tab.ast_vis.data.astype(np.complex64).mean(axis=0).T.flatten()
        vis_ast = xr.DataArray(da.expand_dims(vis_ast, axis=(1, 2)), dims=dims).chunk(
            chunks
        )

        vis_rfi = xds_tab.rfi_vis.data.astype(np.complex64).mean(axis=0).T.flatten()
        vis_rfi = xr.DataArray(da.expand_dims(vis_rfi, axis=(1, 2)), dims=dims).chunk(
            chunks
        )

        gains_bl = 1.0  # this path carries no gains

    elif xds_tab.ast_vis.data.ndim == 4:
        n_sample, n_bl, n_freq, n_time = xds_tab.ast_vis.data.shape
        n_corr = 1

        vis_ast = da.transpose(
            xds_tab.ast_vis.data.astype(np.complex64).mean(axis=0), (2, 0, 1)
        ).reshape(-1, n_freq, n_corr)
        vis_ast = xr.DataArray(vis_ast, dims=dims).chunk(chunks)

        vis_rfi = da.transpose(
            xds_tab.rfi_vis.data.astype(np.complex64).mean(axis=0), (2, 0, 1)
        ).reshape(-1, n_freq, n_corr)
        vis_rfi = xr.DataArray(vis_rfi, dims=dims).chunk(chunks)

        a1 = xds_ms.ANTENNA1.data[:n_bl].compute()
        a2 = xds_ms.ANTENNA2.data[:n_bl].compute()

        gains = xds_tab.gains.data.astype(np.complex64).mean(axis=0)
        gains_bl = da.transpose(gains[a1] * da.conj(gains[a2]), (2, 0, 1)).reshape(-1, n_freq, n_corr)
        gains_bl = xr.DataArray(gains_bl, dims=dims).chunk(chunks)

    else:
        raise ValueError(
            f"Unknown data dimensions. Expected 3 or 4 but got {xds_tab.ast_vis.data.ndim}"
        )

    vis_obs = xds_ms[data_col]

    # The external calibration that data.gain_table divided out at read time. The MS's
    # data_col is still raw, so the model -- fit to calibrated data -- has to be brought
    # back together with it here.
    gains_ext_bl = _external_gains_bl(xds_ms, ms_path, gain_table, dims, chunks)

    # Both gain layers, so vis_cal is the fully calibrated (but still RFI-contaminated)
    # data, and the zarr's vis_ast / vis_rfi are already in that same frame.
    gains_tot = gains_ext_bl * gains_bl

    vis_cal = vis_obs / gains_tot

    vis_ast_res = vis_cal - vis_ast
    vis_rfi_res = vis_cal - vis_rfi
    vis_res = vis_cal - (vis_ast + vis_rfi)

    xds_ms = xds_ms.assign(CORRECTED_DATA=vis_cal)
    xds_ms = xds_ms.assign(TAB_AST_DATA=vis_ast)
    xds_ms = xds_ms.assign(TAB_RFI_DATA=vis_rfi)
    xds_ms = xds_ms.assign(TAB_AST_RES=vis_ast_res)
    xds_ms = xds_ms.assign(TAB_RFI_RES=vis_rfi_res)
    xds_ms = xds_ms.assign(TAB_RES_DATA=vis_res)

    cols = [
        "CORRECTED_DATA",
        "TAB_AST_DATA",
        "TAB_RFI_DATA",
        "TAB_AST_RES",
        "TAB_RFI_RES",
        "TAB_RES_DATA",
    ]
    col_keywords = {col: {"UNIT": "Jy"} for col in cols}

    # The weights that belong to those columns. SIGMA is the noise of the RAW data, so
    # the calibrated noise is SIGMA / |g_total| and the weight scales as |g_total|^2.
    # Written per channel because a frequency-dependent gain gives a frequency-dependent
    # weight -- which a single per-row WEIGHT cannot carry (and which CASA's
    # applycal(calwt=True) does not do either; see tabascal.caltable).
    if "SIGMA" in xds_ms:
        sigma = xds_ms.SIGMA.data[:, 0][:, None, None]          # (row, 1, 1)
        g_amp2 = da.abs(gains_tot.data if hasattr(gains_tot, "data") else gains_tot) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            weight = g_amp2 / da.where(sigma > 0, sigma, np.nan) ** 2
        weight = da.where(da.isfinite(weight), weight, 0.0).astype(np.float32)

        xds_ms = xds_ms.assign(
            WEIGHT_SPECTRUM=xr.DataArray(weight, dims=dims).chunk(chunks)
        )
        # Keep the per-row WEIGHT consistent for tools that only read it.
        xds_ms = xds_ms.assign(
            WEIGHT=xr.DataArray(weight.mean(axis=1), dims=["row", "corr"])
        )
        cols += ["WEIGHT_SPECTRUM", "WEIGHT"]

    print(f"Writing tabascal results to {cols} columns in MS file.")

    dask.compute(xds_to_table([xds_ms], ms_path, cols, column_keywords=col_keywords))

    # Export the calibration this run implies, so it can be applied by standard tooling.
    write_gain_caltable(ms_path, results_zarr_path, gain_table=gain_table)


@measure_runtime
def write_gain_caltable(
    ms_path: str,
    results_zarr_path: str,
    out_path: str | None = None,
    gain_table: str | None = None,
) -> str | None:
    """Export the DIE gains tabascal fitted as an applycal-compatible caltable.

    So a tabascal solution can be consumed by standard tooling (``applycal``, CASA,
    CARAcal/stimela) like any other calibration.

    If the run also consumed an external ``gain_table``, the emitted gains are the TOTAL
    calibration (external x fitted), so applying this one table alone reproduces
    CORRECTED_DATA. Returns the path, or None if the run fitted no gains.
    """
    from tabascal.caltable import match_gains_to_grid, read_caltable, write_caltable

    xds_tab = xr.open_zarr(results_zarr_path)
    if "gains" not in xds_tab:
        return None

    gains = np.asarray(xds_tab.gains.data.mean(axis=0))       # (n_ant, n_freq, n_time)
    if gains.ndim != 3:
        return None

    xds_ms = xds_from_ms(ms_path)[0]
    times = np.unique(np.asarray(xds_ms.TIME.data.compute(), dtype=np.float64))
    spw = xds_from_table(ms_path + "::SPECTRAL_WINDOW")[0]
    freqs = np.asarray(spw.CHAN_FREQ.data[0].compute(), dtype=float)

    if gain_table:
        g_ext = match_gains_to_grid(read_caltable(os.path.abspath(gain_table)), times, freqs)
        gains = g_ext * gains          # total calibration: external x fitted

    if out_path is None:
        out_path = os.path.splitext(results_zarr_path)[0] + ".B"

    write_caltable(out_path, gains, times, ms_path=ms_path)
    print(f"Wrote gain table: {out_path}  (apply with casatasks.applycal)")

    return out_path


def _external_gains_bl(xds_ms, ms_path: str, gain_table: str | None, dims, chunks):
    """Per-baseline product of an external caltable's gains, in MS row order.

    Returns 1.0 when there is no table, so the caller reduces to the no-gain case.
    """
    if not gain_table:
        return 1.0

    from tabascal.caltable import baseline_gains, match_gains_to_grid, read_caltable

    spw = xds_from_table(ms_path + "::SPECTRAL_WINDOW")[0]
    freqs = np.asarray(spw.CHAN_FREQ.data[0].compute(), dtype=float)

    times_all = np.asarray(xds_ms.TIME.data.compute(), dtype=np.float64)
    times = np.unique(times_all)
    n_time = len(times)
    n_row = len(times_all)
    n_bl = n_row // n_time
    n_freq = len(freqs)

    a1 = np.asarray(xds_ms.ANTENNA1.data[:n_bl].compute())
    a2 = np.asarray(xds_ms.ANTENNA2.data[:n_bl].compute())

    cal = read_caltable(os.path.abspath(gain_table))
    gains = match_gains_to_grid(cal, times, freqs)          # (n_ant, n_freq, n_time)
    g_bl = baseline_gains(gains, a1, a2)                    # (n_bl, n_freq, n_time)
    g_bl = np.where(np.isfinite(g_bl) & (g_bl != 0), g_bl, 1.0)

    # (bl, freq, time) -> MS row order, which is time-major.
    g_rows = np.transpose(g_bl, (2, 0, 1)).reshape(n_row, n_freq, 1).astype(np.complex64)
    return xr.DataArray(da.from_array(g_rows), dims=dims).chunk(chunks)


@measure_runtime 
def write_results_xds(
    vi_pred: dict, tab_config, file_path: str, overwrite: bool = True
):

    # print(vi_pred.keys())
    # print(vi_pred["rfi_vis"].shape)
    # print(vi_pred["rfi_vis"])

    # print(da.asarray(vi_pred["ast_vis"]))
    # print(da.asarray(vi_pred["gains"]))
    # print(da.asarray(vi_pred["rfi_vis"]))
    # print(da.asarray(vi_pred["vis_obs"]))
    # print(da.asarray(vi_pred["rfi_A"]))
    # print(da.asarray(args["rfi_phase"]))

    map_xds = xr.Dataset(
        data_vars={
            "rfi_vis": (["sample", "bl", "freq", "time"], da.asarray(vi_pred["vis_rfi"])),  # type: ignore
            "ast_vis": (["sample", "bl", "freq", "time"], da.asarray(vi_pred["vis_ast"])),  # type: ignore
            "gains": (["sample", "ant", "freq", "time"], da.asarray(vi_pred["gains"])),  # type: ignore
            "vis_obs": (["sample", "bl", "freq", "time"], da.asarray(vi_pred["vis_obs"])),  # type: ignore
        },
        coords={
            "time": da.asarray(tab_config.times),  # type: ignore
            "freq": da.asarray(tab_config.freqs),  # type: ignore
        },
    )
    # print(map_xds)

    # Optionally save the fitted complex per-antenna RFI amplitudes. The RFI model is
    #     vis_rfi[p, q] = sum_src rfi_A[src, p] * conj(rfi_A[src, q])
    #                     * exp(1j * (rfi_phase[src, p] - rfi_phase[src, q]))
    # so rfi_A carries the per-antenna gain/beam response toward each satellite while
    # rfi_phase is the *known* geometric (trajectory) phase. Note vis_rfi is invariant
    # under rfi_A[src] -> rfi_A[src] * exp(1j * theta) for any per-(src, freq, time)
    # theta, so only *baseline* phase differences arg(A_p) - arg(A_q) are identifiable.
    # rfi_phase stays float64: it runs to ~1e6 rad, which float32 would quantise to ~0.5 rad.
    if tab_config.args["data"].get("save_rfi_A", False) and "rfi_A" in vi_pred:
        amp_dims = ["sample", "src", "ant", "freq_fine", "time_fine"]
        map_xds = map_xds.assign(
            rfi_A=(amp_dims, da.asarray(np.asarray(vi_pred["rfi_A"]).astype(np.complex64))),
            rfi_phase=(amp_dims, da.asarray(np.asarray(vi_pred["rfi_phase"]).astype(np.float64))),
        )
        # Antenna positions and the baseline map, so the amplitudes can be related to
        # array geometry (e.g. a phase gradient across the array) without the MS.
        map_xds = map_xds.assign(
            ants_itrf=(["ant", "xyz"], np.asarray(tab_config.ants_itrf, dtype=np.float64)),
            a1=("bl", np.asarray(tab_config.a1, dtype=np.int32)),
            a2=("bl", np.asarray(tab_config.a2, dtype=np.int32)),
        )
        map_xds = map_xds.assign_coords(
            norad_id=("src", np.asarray([int(n) for n in tab_config.norad_ids])),
            time_fine=("time_fine", np.asarray(tab_config.times_fine, dtype=np.float64)),
            freq_fine=("freq_fine", np.asarray(tab_config.freqs_fine, dtype=np.float64)),
        )
        # Absolute times and the exact TLEs this run propagated, so downstream geometry
        # (e.g. fitting a satellite position offset to the fitted phases) reproduces the
        # model's own trajectory rather than re-fetching a possibly different TLE.
        map_xds = map_xds.assign(
            time_jd_fine=("time_fine", np.asarray(tab_config.times_jd_fine, dtype=np.float64)),
        )
        map_xds.attrs.update(
            n_int_time=int(tab_config.n_int_time),
            n_int_freq=int(tab_config.n_int_freq),
            tles=[[str(l1), str(l2)] for l1, l2 in np.asarray(tab_config.tles)],
            ra=float(tab_config.phase_centre["ra"]),
            dec=float(tab_config.phase_centre["dec"]),
        )

    # Optionally decompose the RFI visibility per satellite (one ``src`` slice per
    # NORAD id) so each source can be imaged on its own -- a diagnostic for
    # astronomical signal leaking into the RFI model. Off by default: it is ~n_rfi x
    # the rfi_vis storage and adds n_rfi forward-op evaluations.
    if tab_config.args["data"].get("save_rfi_per_sat", False) and "rfi_A" in vi_pred:
        vis_src, norad_ids = rfi_vis_per_sat(vi_pred, tab_config)
        map_xds = map_xds.assign(
            rfi_vis_src=(["sample", "src", "bl", "freq", "time"], da.asarray(vis_src[None]))
        )
        map_xds = map_xds.assign_coords(norad_id=("src", np.asarray(norad_ids)))

    mode = "w" if overwrite else "w-"

    map_xds.to_zarr(file_path, mode=mode)

    return map_xds