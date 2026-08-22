from tabascal.distributed import is_process_0
from tabascal.timing import measure_runtime

from daskms import xds_from_ms, xds_to_table

import numpy as np

import xarray as xr
import dask.array as da
import dask


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

    if xds_tab.ast_vis.data.ndim == 3:
        vis_ast = xds_tab.ast_vis.data.astype(np.complex64).mean(axis=0).T.flatten()
        vis_ast = xr.DataArray(da.expand_dims(vis_ast, axis=(1, 2)), dims=dims).chunk(
            chunks
        )

        vis_rfi = xds_tab.rfi_vis.data.astype(np.complex64).mean(axis=0).T.flatten()
        vis_rfi = xr.DataArray(da.expand_dims(vis_rfi, axis=(1, 2)), dims=dims).chunk(
            chunks
        )

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
        a2 = xds_ms.ANTENNA1.data[:n_bl].compute()

        gains = xds_tab.gains.data.astype(np.complex64).mean(axis=0)
        gains_bl = da.transpose(gains[a1] * da.conj(gains[a2]), (2, 0, 1)).reshape(-1, n_freq, n_corr)
        gains_bl = xr.DataArray(gains_bl, dims=dims).chunk(chunks)

    else:
        raise ValueError(
            f"Unknown data dimensions. Expected 3 or 4 but got {xds_tab.ast_vis.data.ndim}"
        )
    
    

    vis_obs = xds_ms[data_col]
    vis_cal = vis_obs

    vis_ast_res = vis_obs - vis_ast
    vis_rfi_res = vis_obs - vis_rfi
    vis_res = vis_obs - (vis_ast + vis_rfi)
    # vis_cal = vis_obs / gains_bl

    # vis_ast_res = vis_obs - vis_ast * gains_bl
    # vis_rfi_res = vis_obs - vis_rfi * gains_bl
    # vis_res = vis_obs - (vis_ast + vis_rfi) * gains_bl

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
    # process; per-RFI arrays (rfi_A/rfi_delay_us) are sharded and must not be
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
    # print(da.asarray(args["rfi_delay_us"]))

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
            # "rfi_delay_us": (
            #     ["src", "ant", "time_mjd_fine"],
            #     da.asarray(args["rfi_delay_us"]),
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