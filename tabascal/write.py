from tabascal.distributed import is_process_0
from tabascal.timing import measure_runtime

from daskms import xds_from_ms, xds_to_table

import warnings

import numpy as np

import xarray as xr
import dask.array as da
import dask


def read_antenna_pairs(xds_ms, n_bl: int):
    """The two antenna indices of each baseline, from one timestep's rows.

    ``n_bl`` comes from the results zarr; the MS's own baseline count is derived
    the way the reader derives it (``n_row // n_unique_time``) so that a zarr
    belonging to a different MS is named as such, rather than reported as a row
    ordering problem or -- when the two row counts happen to coincide -- written
    out with every baseline's gain on the wrong rows.

    Assumes the time-major row order the reader relies on throughout
    (``reshape(n_time, n_bl)``). Checked rather than assumed: a baseline-major
    store repeats one pair across the first rows, and a per-timestep reshuffle
    keeps the row count right while breaking the reshape.
    """

    times = np.asarray(xds_ms.TIME.data.compute())
    a1_col = np.asarray(xds_ms.ANTENNA1.data.compute())
    a2_col = np.asarray(xds_ms.ANTENNA2.data.compute())

    n_row = times.shape[0]
    n_time_ms = len(np.unique(times))
    n_bl_ms, remainder = divmod(n_row, n_time_ms)

    if remainder:
        raise ValueError(
            f"The MS holds {n_row} rows over {n_time_ms} timesteps, which is not "
            "a whole number of baselines per timestep. tabascal reads "
            "visibilities as (n_time, n_bl) and cannot use this MS."
        )

    if n_bl_ms != n_bl:
        raise ValueError(
            f"The results hold {n_bl} baselines but the MS has {n_bl_ms} "
            f"({n_row} rows over {n_time_ms} timesteps). The results zarr does "
            "not belong to this measurement set, or an antenna was dropped "
            "between the run and the write."
        )

    a1 = a1_col[:n_bl]
    a2 = a2_col[:n_bl]

    if len(set(zip(a1.tolist(), a2.tolist()))) != n_bl:
        raise ValueError(
            f"The first {n_bl} rows do not hold {n_bl} distinct antenna pairs, so "
            "the MS is not ordered time-major. tabascal reads visibilities as "
            "(n_time, n_bl); sort the MS by TIME before running."
        )

    same_order = np.array_equal(
        a1_col.reshape(n_time_ms, n_bl), np.broadcast_to(a1, (n_time_ms, n_bl))
    ) and np.array_equal(
        a2_col.reshape(n_time_ms, n_bl), np.broadcast_to(a2, (n_time_ms, n_bl))
    )

    if not same_order:
        raise ValueError(
            "The baseline order differs between timesteps. tabascal reads "
            "visibilities as (n_time, n_bl) with one fixed baseline order per "
            "timestep; sort the MS by TIME, ANTENNA1, ANTENNA2 before running."
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


def data_frame_residuals(vis_obs, gained_ast, gained_rfi, gained_total):
    """``TAB_*_RES`` residuals, in the frame of the observed data.

    Takes models already multiplied by the baseline gain, since that has to
    happen per sample and only the caller still holds the sample axis.

    ``gained_total`` is passed rather than summed from the two parts: the results
    zarr stores the forward model the gains component actually produced, and a
    component is free to gain only one of the two terms.

    Data frame rather than calibrated: dividing by the gain inflates the noise on
    low-gain baselines and distorts noise-referenced metrics. Moving every column
    to one calibrated frame is #123.
    """

    return {
        "ast": vis_obs - gained_ast,
        "rfi": vis_obs - gained_rfi,
        "total": vis_obs - gained_total,
    }


def unit_bad_gains(gains):
    """Replace zero and non-finite antenna gains with 1, elementwise.

    ``GPGains`` fits an unconstrained affine GP amplitude with no positivity
    transform, and the SVI loop never checks ``isfinite``, so an unflagged dead
    antenna can be driven to zero -- and dividing ``DATA`` by a zero or NaN
    baseline gain poisons ``CORRECTED_DATA`` and every residual column that
    touches the antenna.

    Unity rather than a blank: the affected baselines are then simply
    *uncalibrated* on that antenna, which stays finite and imageable, where NaN
    would drop the data and zero would read as a real, well-calibrated value.
    Substituting on the antenna gains rather than the baseline product means
    everything derived downstream follows automatically.

    Returns ``(gains, bad)`` with ``bad`` the mask that was substituted.
    """

    bad = ~np.isfinite(gains) | (gains == 0)

    return np.where(bad, np.array(1, dtype=gains.dtype), gains), bad


def warn_bad_gains(bad) -> int:
    """Warn, naming the antennas, when any gain was substituted. Returns the count.

    Silent substitution would look like a calibration failure downstream, so the
    count and the antennas it touched are reported where they are known.
    """

    bad = np.asarray(bad)
    n_bad = int(np.count_nonzero(bad))

    if n_bad:
        ants = np.flatnonzero(bad.any(axis=(0, 2, 3))).tolist()
        warnings.warn(
            f"{n_bad} of {bad.size} antenna gain samples "
            f"({100 * n_bad / bad.size:.3g}%) were zero or non-finite and have "
            f"been set to 1. Affected antennas: {ants}. Baselines touching them "
            "are written uncalibrated on that antenna.",
            RuntimeWarning,
            stacklevel=2,
        )

    return n_bad


def total_model(stored, gained_ast, gained_rfi, bad_bl):
    """The gained total model, preferring the forward model the run stored.

    The zarr's ``vis_obs`` is what the gains component actually produced, so it
    is right even where a component gains only one of the two terms -- see the
    commented-out variant in ``components/gains.py``.

    It was formed with the *original* gains, though, so on any baseline whose
    antenna gain was pushed to unity it still carries the zero or non-finite
    value. There the model is re-derived from the two substituted parts, which
    is the same quantity everywhere the substitution did not bite.
    """

    return xr.where(bad_bl, gained_ast + gained_rfi, stored)


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

    # Before any column is built: a zarr from a different MS otherwise surfaces
    # as a dask "chunks do not add up to shape" error from the first reshape.
    a1, a2 = read_antenna_pairs(xds_ms, n_bl)

    ast_vis = xds_tab.ast_vis.data.astype(np.complex64)
    rfi_vis = xds_tab.rfi_vis.data.astype(np.complex64)

    vis_ast = _to_ms_column(ast_vis.mean(axis=0), dims, chunks, n_freq, n_corr)
    vis_rfi = _to_ms_column(rfi_vis.mean(axis=0), dims, chunks, n_freq, n_corr)

    gains, bad = unit_bad_gains(xds_tab.gains.data.astype(np.complex64))
    warn_bad_gains(bad)

    gains_bl_s = baseline_gains(gains, a1, a2, ant_axis=1)

    gained_ast = _to_ms_column(
        gained_model_mean(gains_bl_s, ast_vis), dims, chunks, n_freq, n_corr
    )
    gained_rfi = _to_ms_column(
        gained_model_mean(gains_bl_s, rfi_vis), dims, chunks, n_freq, n_corr
    )
    gains_bl = _to_ms_column(
        gains_bl_s.mean(axis=0), dims, chunks, n_freq, n_corr
    )

    # The zarr's vis_obs is the gained total the forward model produced, so the
    # total residual need not re-derive it from the two parts -- except on the
    # baselines whose gains were substituted, where the stored value predates it.
    if "vis_obs" in xds_tab:
        bad_bl = _to_ms_column(
            (bad[:, a1] | bad[:, a2]).any(axis=0), dims, chunks, n_freq, n_corr
        )
        gained_total = total_model(
            _to_ms_column(
                xds_tab.vis_obs.data.astype(np.complex64).mean(axis=0),
                dims,
                chunks,
                n_freq,
                n_corr,
            ),
            gained_ast,
            gained_rfi,
            bad_bl,
        )
    else:
        # Defensive: every current producer stores it beside the split.
        gained_total = gained_ast + gained_rfi

    vis_obs = xds_ms[data_col]

    vis_cal = vis_obs / gains_bl

    residuals = data_frame_residuals(vis_obs, gained_ast, gained_rfi, gained_total)
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