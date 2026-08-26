from tabascal.distributed import is_process_0
from tabascal.timing import measure_runtime

from daskms import xds_from_ms, xds_to_table

import numpy as np

import xarray as xr
import dask.array as da
import dask


def read_antenna_pairs(xds_ms, n_bl: int):
    """The two antenna indices of each baseline, from one timestep's rows.

    Assumes the time-major row order the reader relies on throughout
    (``reshape(n_time, n_bl)``). Checked rather than assumed: a baseline-major
    store would return ``n_bl`` rows of the same pair.
    """

    a1 = xds_ms.ANTENNA1.data[:n_bl].compute()
    a2 = xds_ms.ANTENNA2.data[:n_bl].compute()

    if len(set(zip(np.asarray(a1).tolist(), np.asarray(a2).tolist()))) != n_bl:
        raise ValueError(
            f"The first {n_bl} rows do not hold {n_bl} distinct antenna pairs, so "
            "the MS is not ordered time-major. tabascal reads visibilities as "
            "(n_time, n_bl); sort the MS by TIME before running."
        )

    return a1, a2


def baseline_gains(gains, a1, a2, ant_axis: int = 0):
    """Per-baseline gain ``g_p conj(g_q)`` from per-antenna gains.

    ``ant_axis`` names the antenna axis, so an array with a leading sample axis
    can have its product formed before the samples are reduced.
    """

    lead = (slice(None),) * ant_axis

    return gains[lead + (a1,)] * gains[lead + (a2,)].conj()


def _to_ms_column(arr, dims, chunks, n_freq, n_corr):
    """``(bl, freq, time)`` array to an MS ``(row, chan, corr)`` DataArray."""

    return xr.DataArray(
        da.transpose(arr, (2, 0, 1)).reshape(-1, n_freq, n_corr), dims=dims
    ).chunk(chunks)


def data_frame_residuals(vis_obs, gained_ast, gained_rfi):
    """``TAB_*_RES`` residuals, in the frame of the observed data.

    Takes models already multiplied by the baseline gain, since that has to
    happen per sample and only the caller still holds the sample axis.

    Data frame rather than calibrated: dividing by the gain inflates the noise on
    low-gain baselines and distorts noise-referenced metrics. Moving every column
    to one calibrated frame is #123.
    """

    return {
        "ast": vis_obs - gained_ast,
        "rfi": vis_obs - gained_rfi,
        "total": vis_obs - (gained_ast + gained_rfi),
    }


def gained_model_mean(gains_bl, model, sample_axis: int = 0):
    """Sample-mean of ``gains_bl * model``, formed per sample.

    Separate from the residual so the reduction order is pinnable.
    """

    return (gains_bl * model).mean(axis=sample_axis)


@measure_runtime
def write_results_ms(ms_path: str, results_zarr_path: str, data_col: str = "DATA"):

    # In multi-process runs only process 0 writes; the arrays involved are replicated
    # so no other rank needs to participate.
    if not is_process_0():
        return

    xds_ms = xds_from_ms(ms_path)[0]
    xds_tab = xr.open_zarr(results_zarr_path)

    dims = ["row", "chan", "corr"]
    chunks = {k: v for k, v in xds_ms.chunks.items() if k in dims}

    if xds_tab.ast_vis.data.ndim != 4:
        raise ValueError(
            f"ast_vis has {xds_tab.ast_vis.data.ndim} dimensions; expected 4, "
            "(sample, bl, freq, time)."
        )

    n_sample, n_bl, n_freq, n_time = xds_tab.ast_vis.data.shape
    n_corr = 1

    ast_vis = xds_tab.ast_vis.data.astype(np.complex64)
    rfi_vis = xds_tab.rfi_vis.data.astype(np.complex64)

    vis_ast = _to_ms_column(ast_vis.mean(axis=0), dims, chunks, n_freq, n_corr)
    vis_rfi = _to_ms_column(rfi_vis.mean(axis=0), dims, chunks, n_freq, n_corr)

    a1, a2 = read_antenna_pairs(xds_ms, n_bl)
    gains_bl_s = baseline_gains(
        xds_tab.gains.data.astype(np.complex64), a1, a2, ant_axis=1
    )

    gained_ast = _to_ms_column(
        gained_model_mean(gains_bl_s, ast_vis), dims, chunks, n_freq, n_corr
    )
    gained_rfi = _to_ms_column(
        gained_model_mean(gains_bl_s, rfi_vis), dims, chunks, n_freq, n_corr
    )
    gains_bl = _to_ms_column(
        gains_bl_s.mean(axis=0), dims, chunks, n_freq, n_corr
    )

    

    vis_obs = xds_ms[data_col]

    vis_cal = vis_obs / gains_bl

    residuals = data_frame_residuals(vis_obs, gained_ast, gained_rfi)
    vis_ast_res = residuals["ast"]
    vis_rfi_res = residuals["rfi"]
    vis_res = residuals["total"]

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

    # Only process 0 writes. Everything written below is replicated on every
    # process; per-RFI arrays (rfi_A/rfi_phase) are sharded and must not be
    # materialized here without a process_allgather.
    if not is_process_0():
        return None

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

    mode = "w" if overwrite else "w-"

    map_xds.to_zarr(file_path, mode=mode)

    return map_xds