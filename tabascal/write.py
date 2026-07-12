from tabascal.timing import measure_runtime

from daskms import xds_from_ms, xds_to_table

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
def write_results_ms(ms_path: str, results_zarr_path: str, data_col: str = "DATA"):

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

    # The forward model is vis_obs = gains_bl * (vis_ast + vis_rfi), but the model
    # visibilities in the zarr are the *un-gained* vis_ast / vis_rfi. Subtracting them
    # straight off the raw data subtracts a model in the wrong frame, so with a
    # non-unit gain the "residual" is meaningless.
    #
    # Residuals are formed in the DATA frame (vis_obs - gains_bl * model), not the
    # calibrated frame (vis_obs/gains_bl - model): dividing by the gain inflates the
    # noise on low-gain baselines, which distorts any noise-referenced residual metric.
    # CORRECTED_DATA carries the calibrated data for imaging the sky.
    # All of this reduces to the old behaviour exactly when the gains are unity.
    vis_cal = vis_obs / gains_bl

    vis_ast = vis_ast * gains_bl
    vis_rfi = vis_rfi * gains_bl

    vis_ast_res = vis_obs - vis_ast
    vis_rfi_res = vis_obs - vis_rfi
    vis_res = vis_obs - (vis_ast + vis_rfi)

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

    print(f"Writing tabascal results to {cols} columns in MS file.")

    dask.compute(xds_to_table([xds_ms], ms_path, cols, column_keywords=col_keywords))


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
            # "rfi_A": (
            #     ["sample", "src", "ant", "rfi_time"],
            #     da.asarray(vi_pred["rfi_A"]),
            # ),
            # "rfi_phase": (
            #     ["src", "ant", "time_mjd_fine"],
            #     da.asarray(args["rfi_phase"]),
            # ),
        },
        coords={
            "time": da.asarray(tab_config.times),  # type: ignore
            "freq": da.asarray(tab_config.freqs),  # type: ignore
            # "rfi_time": da.asarray(args["rfi_times"]),
            # "time_mjd_fine": da.asarray(args["times_mjd_fine"]),
        },
    )
    # print(map_xds)

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