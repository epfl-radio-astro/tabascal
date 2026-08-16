from tabascal.distributed import is_process_0
from tabascal.timing import measure_runtime

from daskms import xds_from_ms, xds_to_table

import numpy as np

import xarray as xr
import dask.array as da
import dask


def _unity_gains_or_raise(xds_tab, tol: float = 1e-6):
    """``1`` if the stored gains are unity, else an error naming the limitation.

    Every stored sample is tested, not their mean: a mean of 1 is not evidence of
    unity, since samples of 0.9 and 1.1 average to it. Testing the mean would let
    exactly the case this guard exists to catch pass through.
    """

    if "gains" not in xds_tab:
        return 1

    gains = np.asarray(xds_tab.gains.data)
    if np.allclose(gains, 1.0, atol=tol):
        return 1

    raise NotImplementedError(
        "This results zarr uses the 3-d ast_vis layout, which carries no baseline "
        "axis, so the per-baseline gain cannot be reconstructed and the residual "
        "columns would be written in the wrong frame. Re-run to produce the "
        "current 4-d layout, which handles fitted gains correctly."
    )


def read_antenna_pairs(xds_ms, n_bl: int):
    """The two antenna indices of each baseline, from one full set of rows.

    Read through a named function so that reading the same column twice -- which
    silently degrades the per-baseline gain to ``|g_p|^2`` -- is a testable
    mistake rather than a two-line typo.
    """

    a1 = xds_ms.ANTENNA1.data[:n_bl].compute()
    a2 = xds_ms.ANTENNA2.data[:n_bl].compute()

    return a1, a2


def baseline_gains(gains, a1, a2, ant_axis: int = 0):
    """Per-baseline gain ``g_p conj(g_q)`` from per-antenna gains.

    Both antenna indices are needed. Building this from ``ANTENNA1`` twice gives
    ``|g_p|^2`` -- real, positive, and blind to the second antenna -- which
    discards all phase and is wrong on every baseline whose two antennas differ,
    i.e. all of them. It is invisible under unitary gains, where the answer is 1
    either way.

    Works on numpy or dask arrays. ``ant_axis`` says which axis carries the
    antenna, so a stored array with a leading sample axis can have its baseline
    product formed *before* the samples are reduced -- ``E[g_p conj(g_q)]`` is
    not ``E[g_p] conj(E[g_q])`` once the gains vary.
    """

    lead = (slice(None),) * ant_axis

    return gains[lead + (a1,)] * gains[lead + (a2,)].conj()


def mean_baseline_gains(gains, a1, a2, sample_axis: int = 0, ant_axis: int = 1):
    """Sample-mean of the per-baseline gain, formed **per sample**.

    The order matters: ``E[g_p conj(g_q)]`` is not ``E[g_p] conj(E[g_q])`` unless
    the two antennas' gains are uncorrelated across samples. Reducing first and
    multiplying after is the cheaper-looking expression and the wrong one.

    Exists as a named function so that reduction order is a testable choice
    rather than an inline expression nothing can pin.
    """

    return baseline_gains(gains, a1, a2, ant_axis=ant_axis).mean(axis=sample_axis)


def data_frame_residuals(vis_obs, vis_ast, vis_rfi, gains_bl):
    """Model residuals in the frame of the observed data.

    The forward model is ``gains_bl * (vis_ast + vis_rfi)``, so a residual has to
    subtract the *gained* model. Subtracting the raw model visibilities, as the
    results zarr stores them, leaves the gains in the residual.

    Formed in the **data** frame (``vis_obs - gains_bl * model``) rather than the
    calibrated one (``vis_obs / gains_bl - model``): dividing by the gain
    inflates the noise on low-gain baselines and distorts any noise-referenced
    residual metric. Moving every column to a single calibrated frame, with the
    weights that belong to it, is issue #123.

    Returns
    -------
    dict
        ``ast``, ``rfi`` and ``total`` residuals, keyed for the ``TAB_*_RES``
        columns.
    """

    return {
        "ast": vis_obs - vis_ast * gains_bl,
        "rfi": vis_obs - vis_rfi * gains_bl,
        "total": vis_obs - (vis_ast + vis_rfi) * gains_bl,
    }


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

        # This layout carries no baseline count, so the antenna pairs cannot be
        # sliced out of the MS and the per-baseline gain cannot be formed. The
        # results writer has produced the 4-d layout since #93, so this branch is
        # only reachable for an older zarr. Rather than silently write columns in
        # the wrong frame, require the gains to be unity -- which is the only case
        # it has ever handled correctly.
        gains_bl = _unity_gains_or_raise(xds_tab)

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

        a1, a2 = read_antenna_pairs(xds_ms, n_bl)

        # Form the baseline product per sample, then reduce: E[g_p conj(g_q)] is
        # not E[g_p] conj(E[g_q]) once the gains vary across samples. Free here,
        # since every writer of this zarr stores exactly one sample.
        gains = xds_tab.gains.data.astype(np.complex64)
        gains_bl = mean_baseline_gains(gains, a1, a2)
        gains_bl = da.transpose(gains_bl, (2, 0, 1)).reshape(-1, n_freq, n_corr)
        gains_bl = xr.DataArray(gains_bl, dims=dims).chunk(chunks)

        if n_sample > 1:
            # The model visibilities above are averaged independently, so the
            # residuals below form E[g] E[m] rather than E[g m]. Equal only when
            # the gains and the model are uncorrelated across samples. Every
            # current writer stores one sample, so this is unreachable today.
            print(
                f"Warning: this results zarr holds {n_sample} samples. The gain and "
                "model visibilities are averaged separately, so the residual columns "
                "are E[g]E[model], not E[g*model]. Form the residual per sample "
                "before averaging if that difference matters."
            )

    else:
        raise ValueError(
            f"Unknown data dimensions. Expected 3 or 4 but got {xds_tab.ast_vis.data.ndim}"
        )
    
    

    vis_obs = xds_ms[data_col]

    vis_cal = vis_obs / gains_bl

    residuals = data_frame_residuals(vis_obs, vis_ast, vis_rfi, gains_bl)
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