#!/usr/bin/env python

from daskms import xds_from_ms, xds_to_table

import numpy as np

import xarray as xr
import dask.array as da
import dask

import argparse


def write_results(ms_path: str, results_zarr_path: str, data_col: str = "DATA"):

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
        n_freq = xds_tab.ast_vis.data.shape[2]
        n_corr = 1

        vis_ast = da.transpose(
            xds_tab.ast_vis.data.astype(np.complex64).mean(axis=0), (2, 0, 1)
        ).reshape(-1, n_freq, n_corr)
        vis_ast = xr.DataArray(vis_ast, dims=dims).chunk(chunks)

        vis_rfi = da.transpose(
            xds_tab.rfi_vis.data.astype(np.complex64).mean(axis=0), (2, 0, 1)
        ).reshape(-1, n_freq, n_corr)
        vis_rfi = xr.DataArray(vis_rfi, dims=dims).chunk(chunks)

    else:
        raise ValueError(
            f"Unknown data dimensions. Expected 3 or 4 but got {xds_tab.ast_vis.data.ndim}"
        )

    vis_obs = xds_ms[data_col]

    vis_ast_res = vis_obs - vis_ast
    vis_rfi_res = vis_obs - vis_rfi
    vis_res = vis_obs - vis_ast - vis_rfi

    xds_ms = xds_ms.assign(TAB_AST_DATA=vis_ast)
    xds_ms = xds_ms.assign(TAB_RFI_DATA=vis_rfi)
    xds_ms = xds_ms.assign(TAB_AST_RES=vis_ast_res)
    xds_ms = xds_ms.assign(TAB_RFI_RES=vis_rfi_res)
    xds_ms = xds_ms.assign(TAB_RES_DATA=vis_res)

    cols = [
        "TAB_AST_DATA",
        "TAB_RFI_DATA",
        "TAB_AST_RES",
        "TAB_RFI_RES",
        "TAB_RES_DATA",
    ]
    col_keywords = {col: {"UNIT": "Jy"} for col in cols}

    print(f"Writing tabascal results to {cols} columns in MS file.")

    dask.compute(xds_to_table([xds_ms], ms_path, cols, column_keywords=col_keywords))


def main():

    parser = argparse.ArgumentParser(
        description="Copy recovered data from a tabascal run and save in a MS file under the column named 'TAB_DATA'."
    )
    parser.add_argument(
        "-m", "--ms_path", required=True, help="File path to the Measurement Set."
    )
    parser.add_argument(
        "-z",
        "--results_zarr_path",
        required=True,
        help="File path to the zarr file containing results.",
    )
    parser.add_argument(
        "-d", "--data_col", default="DATA", help="Data column name. Default is DATA"
    )

    args = parser.parse_args()
    write_results(
        ms_path=args.ms_path,
        results_zarr_path=args.results_zarr_path,
        data_col=args.data_col,
    )


if __name__ == "__main__":
    main()
