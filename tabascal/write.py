from tabascal.distributed import is_process_0
from tabascal.gain_table import gains_from_tables, normalise_gain_tables
from tabascal.interferometry import baseline_gains
from tabascal.ms import (
    fitted_correlation,
    grid_to_rows,
    into_corr,
    ms_layout,
    partition_noise,
    partition_polarization,
    partition_setup,
    read_time_unit,
    times_to_mjd,
)
from tabascal.noise import broadcast_to_vis
from tabascal.time import DAY_SECS
from tabascal.timing import measure_runtime

from daskms import xds_from_ms, xds_from_table, xds_to_table

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

    return xr.DataArray(grid_to_rows(arr, n_freq, n_corr), dims=dims).chunk(chunks)


def calibrated_residuals(vis_cal, vis_ast, vis_rfi):
    """``TAB_*_RES`` residuals, in the calibrated frame every column shares.

    ``vis_cal`` is the data with *all* the gains divided out -- the external
    tables' and the fitted DIE gains -- which is the frame the zarr's
    ``vis_ast``/``vis_rfi`` were fitted in, so the models are subtracted as they
    come and are never re-gained.

    The total is formed from ``vis_ast + vis_rfi``, the same sum written into
    the two model columns, which is what makes

        ``TAB_AST_DATA + TAB_RFI_DATA + TAB_RES_DATA == CORRECTED_DATA``

    an exact floating-point identity rather than an approximate one (#123) --
    exact wherever ``vis_cal - model`` is exactly representable, which complex
    arithmetic decides **per component**: Sterbenz's condition has to hold for
    the real parts and for the imaginary parts separately, each pair within a
    factor of two of the other. A residual that is small next to ``|vis_cal|`` is
    not sufficient on its own, because a cell whose real part nearly cancels can
    have a residual far larger than that component. Where the condition fails --
    such a cell, or a fit so bad that a cell is all residual -- the identity
    holds to within one float32 ulp of the visibility's magnitude instead, still
    seven orders of magnitude tighter than a column in the wrong frame, which is
    out by a gain.

    The run's own ``vis_obs`` forward model is deliberately not used: it lives in
    the gained frame, and a gains component that gains only one of the two terms
    leaves it in no single frame at all -- ``ast + rfi / g`` is not a frame
    either model column is written in.

    Calibrated rather than the data frame #134 wrote: dividing by the gain
    inflates the noise on low-gain baselines, and the answer to that is the
    ``WEIGHT_SPECTRUM`` written beside these columns -- see
    :func:`calibrated_weights` -- not a second frame no weight can describe.
    """

    return {
        "ast": vis_cal - vis_ast,
        "rfi": vis_cal - vis_rfi,
        "total": vis_cal - (vis_ast + vis_rfi),
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


def count_substituted(bad, ant_axis=None):
    """Lazy reductions of a substitution mask: the count, and which antennas.

    Reductions rather than the mask itself. The masks are the size of the gains
    -- ``(sample, ant, freq, time)`` -- or of the baseline product --
    ``(bl, freq, time)`` -- and materialising either just to count it would
    hold a full-size boolean array in memory for the sake of a number. A
    reduction on a dask array runs chunk by chunk and leaves nothing resident;
    on numpy it is simply the count.

    Returns ``bad.sum()`` alone, or ``(bad.sum(), bad.any(over every axis but
    ant_axis))`` when there is an antenna axis to name antennas from. Both are
    still lazy for dask input; the caller computes them, ideally in the same
    ``dask.compute`` as the arrays the mask feeds, so the graph runs once.
    """

    n_bad = bad.sum()

    if ant_axis is None:
        return n_bad

    axes = tuple(axis for axis in range(bad.ndim) if axis != ant_axis)

    return n_bad, bad.any(axis=axes)


def _warn_substituted(n_bad: int, size: int, what: str, detail: str) -> int:
    """Warn about ``n_bad`` of ``size`` substitutions, if any. Returns ``n_bad``.

    The two public warnings below say different things about different arrays;
    what they share is how a count becomes a percentage and a
    ``RuntimeWarning``, which is the part worth having once. Takes the computed
    numbers, not the mask: see :func:`count_substituted`.
    """

    n_bad = int(n_bad)

    if n_bad:
        warnings.warn(
            f"{n_bad} of {size} {what} ({100 * n_bad / size:.3g}%) were "
            f"zero or non-finite and have been set to 1{detail}",
            RuntimeWarning,
            stacklevel=3,
        )

    return n_bad


def warn_bad_gains(n_bad: int, size: int, bad_ants) -> int:
    """Warn, naming the antennas, when any gain was substituted. Returns the count.

    Silent substitution would look like a calibration failure downstream, so the
    count and the antennas it touched are reported where they are known.
    ``bad_ants`` is the per-antenna boolean from :func:`count_substituted`.
    """

    ants = np.flatnonzero(np.asarray(bad_ants)).tolist()

    return _warn_substituted(
        n_bad,
        size,
        "antenna gain values",
        f". Affected antennas: {ants}. Baselines touching them are written "
        "uncalibrated on that antenna.",
    )


def warn_bad_baseline_gains(n_bad: int, size: int) -> int:
    """Warn when a *mean* baseline gain was substituted. Returns the count.

    Separate from the per-antenna warning because it is a separate failure. The
    per-sample gains can every one of them be finite and non-zero and still
    average to zero -- ``g_q = +1`` on one sample and ``-1`` on the next -- and
    it is the mean that ``CORRECTED_DATA`` is divided by. There is no antenna to
    name: the substitution happens after the product and after the reduction.
    """

    return _warn_substituted(
        n_bad,
        size,
        "mean baseline gains",
        ", even though the per-sample gains were not. CORRECTED_DATA equals the "
        "data in those (baseline, channel, time) cells.",
    )


def external_baseline_gains(gain_table, times_sec, freqs, a1, a2, n_ant=None,
                            verbose: bool = True):
    """Per-baseline gains of the external calibration tables, on this grid.

    ``data.gain_table`` was divided out of the visibilities **in memory** when
    the MS was read (:meth:`TabConfig.apply_gain_table`); the MS's own data
    column is still raw. So the same gains have to be re-derived here, or the
    models -- which were fitted to the externally calibrated data -- would be
    subtracted in a frame the data is not in.

    Re-derived rather than read back off the results: the run does not record
    them. That makes the *grid* load-bearing. ``times_sec`` must be the MS's own
    ``TIME`` column in seconds on the scale it declares (never the UTC-normalised
    ``times_jd``, which is 37 s away on a TAI-declared MS) and ``freqs`` the
    partition's ``CHAN_FREQ``, exactly as the reader passed them -- otherwise
    each table is placed on a different grid and the columns quietly leave the
    frame the models are in. The placement itself, and the ordered composition
    of several tables, is :mod:`tabascal.gain_table`'s subject.

    Returns ``(n_bl, n_freq, n_time)``. A baseline no table could supply a gain
    for takes 1, the same convention as :func:`unit_bad_gains`: those
    visibilities are written *uncalibrated* rather than divided by a number
    nobody solved. The run flagged them out of the fit at read time.
    """

    gains, dead = gains_from_tables(
        gain_table, times_sec, freqs, n_ant=n_ant, verbose=verbose
    )

    a1, a2 = np.asarray(a1), np.asarray(a2)

    return np.where(dead[a1] | dead[a2], 1.0, baseline_gains(gains, a1, a2))


def calibrated_weights(gains_tot, sigma):
    """``|g_total|^2 / sigma^2`` -- the weight the calibrated columns need.

    ``sigma`` is the noise on the **raw** data, as the MS's ``SIGMA_SPECTRUM``
    or ``SIGMA`` records it. Dividing a visibility by ``g_total`` divides its
    noise by ``|g_total|`` too, so ``sigma_cal = sigma / |g_total|`` and the
    weight rises with the square. That is the answer to the objection that
    calibrating makes the noise heteroscedastic: it does, and this is the
    number that says by how much.

    Per channel, because a frequency-dependent gain gives a frequency-dependent
    weight, which a single per-row ``WEIGHT`` cannot carry -- and which CASA's
    ``applycal(calwt=True)`` does not produce either (see
    :func:`tabascal.ms.write_caltable`).

    Formed in float64 from the float32 magnitude, so the single cast the writer
    makes on the way into the column is the only rounding in it.
    """

    return abs(gains_tot).astype(np.float64) ** 2 / sigma**2


def _external_gains_column(
    ms_path, xds_ms, layout, column_keywords, gain_table, dims, chunks, n_freq
):
    """The external tables' baseline gains, as an MS ``(row, chan, 1)`` column.

    ``None`` when no table was used, so the caller divides by the fitted gains
    alone rather than by a unit array the size of the data.
    """

    paths = normalise_gain_tables(gain_table)

    if not paths:
        return None

    # The partition's own spectral window, resolved the way read_ms resolves it:
    # row 0 of SPECTRAL_WINDOW is a convention, not a guarantee.
    spw_id, _ = partition_setup(ms_path, xds_ms)
    spec = xds_from_table(ms_path + "::SPECTRAL_WINDOW", group_cols="__row__")[spw_id]
    freqs = np.asarray(spec.CHAN_FREQ.data[0].compute(), dtype=np.float64)

    if len(freqs) != n_freq:
        raise ValueError(
            f"The MS's spectral window holds {len(freqs)} channels but the "
            f"results hold {n_freq}. The gain tables would be placed on a band "
            "the results were not fitted on."
        )

    # From the ANTENNA subtable rather than from the antenna pairs in use: a
    # table solved on a master MS covers this observation in its leading rows,
    # and it is that count the reader trimmed each table to.
    n_ant = int(xds_from_table(ms_path + "::ANTENNA")[0].sizes["row"])

    # One TIME per timestep block, in seconds on the scale the column declares:
    # a caltable's TIME is a copy of this column, so the match is made against
    # the MS's own values and never against the UTC-normalised times_jd, which
    # on a TAI-declared MS is 37 s away. float64 throughout -- an MJD second
    # does not survive float32.
    times = np.asarray(
        xds_ms.TIME.data.reshape(layout.n_time, layout.n_bl)[:, 0].compute(),
        dtype=np.float64,
    )
    times_sec = times_to_mjd(times, read_time_unit(column_keywords)) * DAY_SECS

    g_bl = external_baseline_gains(
        paths, times_sec, freqs, layout.a1, layout.a2, n_ant=n_ant
    )

    return _to_ms_column(g_bl.astype(np.complex64), dims, chunks, n_freq)


def _weight_column(xds_ms, gains_tot, layout, n_freq, corr_idx, dims, chunks):
    """``WEIGHT_SPECTRUM`` for the calibrated frame, or ``None``.

    The noise comes from the MS the same way the reader takes it --
    ``SIGMA_SPECTRUM`` first, ``SIGMA`` behind it, on the *fitted* correlation
    rather than correlation 0, with the same median collapse and median fill for
    the cells that measured nothing (see :mod:`tabascal.noise`). It describes the
    raw data, which is what makes it the right numerator's denominator here.

    ``None`` when the MS carries no usable noise at all: nothing is invented, and
    the weight columns are left alone rather than filled with a made-up scale.
    """

    sigma = partition_noise(xds_ms, layout.n_time, layout.n_bl, n_freq, corr_idx)

    if sigma is None:
        print(
            "Warning: the MS partition carries no usable SIGMA_SPECTRUM or "
            "SIGMA column, so WEIGHT_SPECTRUM and WEIGHT are left untouched. "
            "The columns written are calibrated and their noise is no longer "
            "the MS's; re-weight them by |g_total|^2 before imaging."
        )

        return None

    shape = (layout.n_bl, n_freq, layout.n_time)
    # Resolved onto the visibilities before the transpose: partition_noise
    # returns a noise per baseline, per (baseline, channel), or either of those
    # per timestep as well, and only a full grid can be laid out in row order.
    sigma = np.ascontiguousarray(
        np.broadcast_to(broadcast_to_vis(sigma, shape), shape)
    )

    return calibrated_weights(gains_tot, _to_ms_column(sigma, dims, chunks, n_freq))


@measure_runtime
def write_results_ms(
    ms_path: str,
    results_zarr_path: str,
    data_col: str = "DATA",
    corr: str | None = None,
    gain_table=None,
):
    """Copy a results zarr into the Measurement Set it was fitted from.

    Every column written here -- ``CORRECTED_DATA``, ``TAB_AST_DATA``,
    ``TAB_RFI_DATA``, ``TAB_AST_RES``, ``TAB_RFI_RES`` and ``TAB_RES_DATA`` --
    lives in ONE frame: the data with **all** the gains divided out. There are
    two layers and both come off:

    * ``gain_table`` -- the external calibration ``data.gain_table`` divided out
      when the MS was read. The MS's own ``data_col`` is still raw, so it has to
      be removed here too or the models, which were fitted to calibrated data,
      would be subtracted in the wrong frame.
    * the DIE gains the model fitted, stored in the results zarr.

    That is what makes ``TAB_AST_DATA + TAB_RFI_DATA + TAB_RES_DATA ==
    CORRECTED_DATA`` hold exactly, and what lets one weight describe them all
    (#123). It reverses the data-frame residuals of #134: dividing by the gain
    does make the noise heteroscedastic, and the answer is the weight that says
    so -- ``WEIGHT_SPECTRUM = |g_total|^2 / SIGMA^2`` per channel, with
    ``WEIGHT`` its frequency mean -- rather than a second frame beside the first.

    ``corr`` names the correlation the results belong to. It is an override, not
    the normal route: ``write_results_xds`` records the fitted correlation on the
    zarr, and results carrying it need nothing here. Pass it only for a zarr
    written before that attribute existed, where a multi-correlation MS has no
    other way to know.

    ``gain_table`` is the ordered list ``data.gain_table`` named, or a single
    path. It is required whenever the run used one: without it every column is
    written a whole calibration layer away from the frame it should be in.
    """

    # In multi-process runs only process 0 writes; the arrays involved are replicated
    # so no other rank needs to participate.
    if not is_process_0():
        return

    # column_keywords for the TIME unit the gain tables are matched on, read the
    # same way read_ms reads it so the two cannot disagree about the grid.
    xds_list, column_keywords = xds_from_ms(ms_path, column_keywords=True)
    xds_ms = xds_list[0]
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
    n_bad, bad_ants = count_substituted(bad, ant_axis=1)

    gains_bl_s = baseline_gains(gains, a1, a2, ant_axis=1)

    # Guarded again after the reduction: per-sample gains that are all finite
    # and non-zero can still average to zero, and it is the mean that the data
    # is divided by.
    gains_bl_mean, bad_bl_mean = unit_bad_gains(gains_bl_s.mean(axis=0))
    n_bad_bl = count_substituted(bad_bl_mean)

    gains_bl = _to_ms_column(gains_bl_mean, dims, fit_chunks, n_freq)

    # The external calibration data.gain_table divided out at read time; the MS's
    # data_col is still raw, so it is re-derived on this observation's own grid
    # and removed together with the fitted gains.
    gains_ext = _external_gains_column(
        ms_path, xds_ms, layout, column_keywords, gain_table, dims, fit_chunks,
        n_freq,
    )

    # The one divisor the whole frame is defined by. Not separately guarded
    # against a product that underflows to zero in complex64: both factors are
    # finite and non-zero by construction, and a composed gain that cannot be
    # held in the MS's own precision is a failed calibration rather than a dead
    # antenna -- the same line the overflow case is on.
    gains_tot = gains_bl if gains_ext is None else gains_ext * gains_bl

    vis_obs = xds_ms[data_col]

    # Sliced by position rather than with .isel so no correlation coordinate can
    # come along and misalign the arithmetic below.
    vis_obs_fit = xr.DataArray(
        vis_obs.data[:, :, corr_idx : corr_idx + 1], dims=dims
    ).chunk(fit_chunks)

    vis_cal = vis_obs_fit / gains_tot

    residuals = calibrated_residuals(vis_cal, vis_ast, vis_rfi)

    weight = _weight_column(
        xds_ms, gains_tot, layout, n_freq, corr_idx, dims, fit_chunks
    )

    def column(col, fill, col_dims=None):
        """One result placed on the MS's correlation axis, ready to assign."""

        col_dims = dims if col_dims is None else col_dims

        return xr.DataArray(
            into_corr(col.data, corr_idx, n_corr, fill), dims=col_dims
        ).chunk({k: v for k, v in chunks.items() if k in col_dims})

    # Model columns are zero on the correlations that were not fitted; the data
    # columns pass the data through there, uncalibrated and unsubtracted.
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
    # Jy on the visibility columns. The weight columns are standard ones and are
    # left with whatever units the MS gives them.
    col_keywords = {col: {"UNIT": "Jy"} for col in cols}

    if weight is not None:
        # Zero on the correlations that were not fitted, matching the zeroed
        # model columns: nothing there was calibrated, so nothing there has this
        # frame's weight. Cast once, on the way into the column.
        xds_ms = xds_ms.assign(
            WEIGHT_SPECTRUM=column(weight.astype(np.float32), np.float32(0))
        )
        xds_ms = xds_ms.assign(
            WEIGHT=column(
                weight.mean(dim="chan").astype(np.float32),
                np.float32(0),
                col_dims=["row", "corr"],
            )
        )
        cols += ["WEIGHT_SPECTRUM", "WEIGHT"]

    print(f"Writing tabascal results to {cols} columns in MS file.")

    # One compute for the write and for the warning counts, so the warnings
    # follow the write: the substitution is the designed behaviour and the
    # warning a report of it, not a precondition. Warning first would cost a
    # second pass over the mask graph. The masks feed the
    # columns, so evaluating everything in one graph runs them once, chunk by
    # chunk, and nothing full-size is ever held in memory for the warnings.
    n_bad, bad_ants, n_bad_bl, _ = dask.compute(
        n_bad,
        bad_ants,
        n_bad_bl,
        xds_to_table([xds_ms], ms_path, cols, column_keywords=col_keywords),
    )

    warn_bad_gains(n_bad, bad.size, bad_ants)
    warn_bad_baseline_gains(n_bad_bl, bad_bl_mean.size)


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
            # SEAM: `time` is seconds from the start of the observation, so the
            # results carry no absolute epoch, no phase centre and no identity
            # for the MS they came from. write_results_ms therefore has to
            # re-derive both from the MS it is pointed at -- including the grid
            # the external gain tables are placed on (_external_gains_column),
            # which is only the run's own grid because the reader and the writer
            # read the same column the same way. times_jd / times_mjd, the field
            # direction and the source MS path belong here, beside `time`.
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