from tabascal.distributed import is_process_0
from tabascal.interferometry import baseline_gains
from tabascal.ms import ms_layout, partition_polarization, resolve_correlation
from tabascal.timing import measure_runtime

from daskms import xds_from_ms, xds_to_table

import warnings

import numpy as np

import xarray as xr
import dask.array as da
import dask


def _to_ms_column(arr, dims, chunks, n_freq, n_corr=1):
    """``(bl, freq, time)`` array to an MS ``(row, chan, corr)`` DataArray.

    ``n_corr`` is 1 here: tabascal fits one correlation, so every result starts
    life on a length-1 correlation axis and :func:`into_corr` places it on the
    MS's axis afterwards.
    """

    return xr.DataArray(
        da.transpose(arr, (2, 0, 1)).reshape(-1, n_freq, n_corr), dims=dims
    ).chunk(chunks)


def fitted_correlation(
    ms_path: str, zarr_corr, corr, n_corr: int, pol_id: int = 0
) -> int:
    """Index on the MS's correlation axis that the results belong to.

    tabascal fits one correlation. Its name comes from the ``corr`` argument if
    given, else from the ``corr`` attribute the run recorded on the results
    zarr, and is resolved to an index **by identity, not by position** -- a
    single-polarisation MS holds one correlation whatever it is, so ``yy`` is
    index 0 there.

    ``pol_id`` is the ``POLARIZATION`` row the data partition actually uses,
    the same one ``read_ms`` resolved through ``DATA_DESCRIPTION``. Row 0 is
    only a convention: a partition on another row may order its correlations
    differently, or hold fewer of them, and resolving against the wrong row
    would put the results in the wrong polarisation without a word.

    A zarr written before that attribute existed carries no name. With one
    correlation there is only one answer; with more, guessing would silently
    write the results into the wrong polarisation, so it is an error.
    """

    name = corr if corr is not None else zarr_corr

    if name is None:
        if n_corr == 1:
            return 0

        raise ValueError(
            f"The MS has {n_corr} correlations and the results zarr does not "
            "record which one was fitted -- it predates that attribute. Pass "
            "the correlation explicitly: write_results_ms(..., corr='xx'), or "
            "tab2MS -c xx."
        )

    corr_idx = resolve_correlation(ms_path, name, pol_id)

    if not 0 <= corr_idx < n_corr:
        raise ValueError(
            f"Correlation {name!r} resolves to index {corr_idx} on POLARIZATION "
            f"row {pol_id}, but the data partition has {n_corr} correlations. "
            "The MS's DATA_DESCRIPTION and POLARIZATION subtables disagree."
        )

    return corr_idx


def into_corr(col, corr_idx: int, n_corr: int, fill):
    """Place a one-correlation result on the MS's correlation axis.

    Results are ``(row, chan, 1)`` while the MS column may be ``(row, chan, 4)``.
    The fitted correlation takes the result; the others take ``fill`` -- zero for
    the model columns, and the data column itself for the data-frame columns,
    which is what "no gain applied and nothing subtracted" means there.

    Works on the raw arrays because xarray will not broadcast a length-1 ``corr``
    dimension against a length-4 one; the caller re-wraps.
    """

    if n_corr == 1:
        return col

    return np.where(np.arange(n_corr) == corr_idx, col, fill)


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
            f"{n_bad} of {bad.size} antenna gain values "
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

    Every argument keeps its sample axis, so the choice is made per sample and
    the caller averages afterwards. Reducing ``bad_bl`` over samples first would
    throw away the stored model on *every* sample of a cell because one sample
    happened to have a bad gain.
    """

    return np.where(bad_bl, gained_ast + gained_rfi, stored)


def warn_bad_baseline_gains(bad) -> int:
    """Warn when a *mean* baseline gain was substituted. Returns the count.

    Separate from the per-antenna warning because it is a separate failure. The
    per-sample gains can every one of them be finite and non-zero and still
    average to zero -- ``g_q = +1`` on one sample and ``-1`` on the next -- and
    it is the mean that ``CORRECTED_DATA`` is divided by. There is no antenna to
    name: the substitution happens after the product and after the reduction.
    """

    bad = np.asarray(bad)
    n_bad = int(np.count_nonzero(bad))

    if n_bad:
        warnings.warn(
            f"{n_bad} of {bad.size} mean baseline gains "
            f"({100 * n_bad / bad.size:.3g}%) were zero or non-finite and have "
            "been set to 1, even though the per-sample gains were not. "
            "CORRECTED_DATA equals the data in those (baseline, channel, time) "
            "cells.",
            RuntimeWarning,
            stacklevel=2,
        )

    return n_bad


def gained_model_mean(gains_bl, model, sample_axis: int = 0):
    """Sample-mean of ``gains_bl * model``, formed per sample.

    Separate from the residual so the reduction order is pinnable.
    """

    return (gains_bl * model).mean(axis=sample_axis)


@measure_runtime
def write_results_ms(
    ms_path: str,
    results_zarr_path: str,
    data_col: str = "DATA",
    corr: str | None = None,
):

    # In multi-process runs only process 0 writes; the arrays involved are replicated
    # so no other rank needs to participate.
    if not is_process_0():
        return

    xds_ms = xds_from_ms(ms_path)[0]
    xds_tab = xr.open_zarr(results_zarr_path)

    dims = ["row", "chan", "corr"]
    chunks = {k: v for k, v in xds_ms.chunks.items() if k in dims}

    # The results live on a length-1 correlation axis until they are placed on
    # the MS's, so they must not be chunked with the MS's correlation chunk.
    fit_chunks = {k: v for k, v in chunks.items() if k != "corr"}

    if xds_tab.ast_vis.data.ndim != 4:
        raise ValueError(
            f"ast_vis has {xds_tab.ast_vis.data.ndim} dimensions; expected 4, "
            "(sample, bl, freq, time)."
        )

    n_sample, n_bl, n_freq, n_time = xds_tab.ast_vis.data.shape
    n_corr = xds_ms.sizes["corr"]

    # The polarization setup this partition actually uses, which need not be
    # row 0 of POLARIZATION -- resolved the way read_ms resolves it.
    corr_idx = fitted_correlation(
        ms_path,
        xds_tab.attrs.get("corr"),
        corr,
        n_corr,
        partition_polarization(ms_path, xds_ms),
    )

    # Derived and validated in one place, shared with the reader. Before any
    # column is built: a zarr from a different MS otherwise surfaces as a dask
    # "chunks do not add up to shape" error from the first reshape.
    layout = ms_layout(xds_ms)
    a1, a2 = layout.a1, layout.a2

    # The MS's layout is the MS's business; whether the results describe *this*
    # MS is the writer's.
    if layout.n_bl != n_bl:
        raise ValueError(
            f"The results hold {n_bl} baselines but the MS has {layout.n_bl} "
            f"({layout.n_time * layout.n_bl} rows over {layout.n_time} "
            "timesteps). The results zarr does not belong to this measurement "
            "set, or an antenna was dropped between the run and the write."
        )

    ast_vis = xds_tab.ast_vis.data.astype(np.complex64)
    rfi_vis = xds_tab.rfi_vis.data.astype(np.complex64)

    vis_ast = _to_ms_column(ast_vis.mean(axis=0), dims, fit_chunks, n_freq)
    vis_rfi = _to_ms_column(rfi_vis.mean(axis=0), dims, fit_chunks, n_freq)

    gains, bad = unit_bad_gains(xds_tab.gains.data.astype(np.complex64))
    warn_bad_gains(bad)

    gains_bl_s = baseline_gains(gains, a1, a2, ant_axis=1)

    gained_ast = _to_ms_column(
        gained_model_mean(gains_bl_s, ast_vis), dims, fit_chunks, n_freq
    )
    gained_rfi = _to_ms_column(
        gained_model_mean(gains_bl_s, rfi_vis), dims, fit_chunks, n_freq
    )
    # Guarded again after the reduction: per-sample gains that are all finite
    # and non-zero can still average to zero, and it is the mean that the data
    # is divided by.
    gains_bl_mean, bad_bl_mean = unit_bad_gains(gains_bl_s.mean(axis=0))
    warn_bad_baseline_gains(bad_bl_mean)

    gains_bl = _to_ms_column(gains_bl_mean, dims, fit_chunks, n_freq)

    # The zarr's vis_obs is the gained total the forward model produced, so the
    # total residual need not re-derive it from the two parts -- except on the
    # samples whose gains were substituted, where the stored value predates it.
    # Chosen per sample, then averaged; dask shares the two products below with
    # the ones formed for the per-component columns above.
    if "vis_obs" in xds_tab:
        total_s = total_model(
            xds_tab.vis_obs.data.astype(np.complex64),
            gains_bl_s * ast_vis,
            gains_bl_s * rfi_vis,
            bad[:, a1] | bad[:, a2],
        )
    else:
        # Defensive: every current producer stores it beside the split.
        total_s = gains_bl_s * (ast_vis + rfi_vis)

    gained_total = _to_ms_column(total_s.mean(axis=0), dims, fit_chunks, n_freq)

    vis_obs = xds_ms[data_col]

    # Sliced by position rather than with .isel so no correlation coordinate can
    # come along and misalign the arithmetic below.
    vis_obs_fit = xr.DataArray(
        vis_obs.data[:, :, corr_idx : corr_idx + 1], dims=dims
    ).chunk(fit_chunks)

    vis_cal = vis_obs_fit / gains_bl

    residuals = data_frame_residuals(
        vis_obs_fit, gained_ast, gained_rfi, gained_total
    )

    def column(col, fill):
        """One result placed on the MS's correlation axis, ready to assign."""

        return xr.DataArray(
            into_corr(col.data, corr_idx, n_corr, fill), dims=dims
        ).chunk(chunks)

    # Model columns are zero on the correlations that were not fitted; the
    # data-frame columns pass the data through there, ungained and unsubtracted.
    passthrough = vis_obs.data

    xds_ms = xds_ms.assign(CORRECTED_DATA=column(vis_cal, passthrough))
    xds_ms = xds_ms.assign(TAB_AST_DATA=column(vis_ast, 0))
    xds_ms = xds_ms.assign(TAB_RFI_DATA=column(vis_rfi, 0))
    xds_ms = xds_ms.assign(TAB_AST_RES=column(residuals["ast"], passthrough))
    xds_ms = xds_ms.assign(TAB_RFI_RES=column(residuals["rfi"], passthrough))
    xds_ms = xds_ms.assign(TAB_RES_DATA=column(residuals["total"], passthrough))

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
        # Which correlation was fitted, by name. Without it the writer cannot
        # tell where the results belong on a multi-correlation MS.
        attrs={"corr": tab_config.args["data"]["corr"]},
    )
    # print(map_xds)

    mode = "w" if overwrite else "w-"

    map_xds.to_zarr(file_path, mode=mode)

    return map_xds