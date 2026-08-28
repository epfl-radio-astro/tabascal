from tabascal.distributed import (
    constrain_rfi_state,
    is_process_0,
    make_global,
    rfi_sharding,
    sharding_enabled,
)
from tabascal.gain_table import gains_from_tables, normalise_gain_tables
from tabascal.imports import import_components
from tabascal.interferometry import baseline_gains
from tabascal.ms import (
    fitted_correlation,
    grid_to_rows,
    into_corr,
    ms_layout,
    partition_noise,
    partition_polarization,
    partition_setup,
    read_time_scale,
    read_time_unit,
    remove_caltable,
    times_to_mjd,
    write_caltable,
)
from tabascal.noise import broadcast_to_vis
from tabascal.time import DAY_SECS
from tabascal.timing import measure_runtime

from daskms import xds_from_ms, xds_from_table, xds_to_table

import os
import traceback
import warnings

import numpy as np

import jax
import jax.numpy as jnp

import xarray as xr
import dask.array as da
import dask


#: Prefix of the per-satellite RFI columns; the NORAD id is appended, giving
#: ``TAB_RFI_58126``. The ``TAB_RFI_DATA`` those columns sum back to shares the
#: prefix, which is the point: they are that column, split by satellite.
RFI_PER_SAT_PREFIX = "TAB_RFI_"

#: Dimensions of the per-satellite RFI variable in a results zarr, in order.
RFI_PER_SAT_DIMS = ("sample", "src", "bl", "freq", "time")


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


def _observation_grid(ms_path, xds_ms, layout, column_keywords, n_freq):
    """``(times_sec, freqs, n_ant)`` -- the grid this run's gains live on.

    One definition of it, because two things are placed on it: the external
    tables the columns are divided by, and the calibration table written beside
    the results. Derived twice, they could be derived differently, and the two
    would then describe calibrations of two different observations.

    ``times_sec`` is one ``TIME`` per timestep block, in seconds on the scale the
    column declares -- a caltable's ``TIME`` is a copy of this column, so the
    match is made against the MS's own values and never against the
    UTC-normalised ``times_jd``, which on a TAI-declared MS is 37 s away.
    float64 throughout: an MJD second does not survive float32.
    """

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

    times = np.asarray(
        xds_ms.TIME.data.reshape(layout.n_time, layout.n_bl)[:, 0].compute(),
        dtype=np.float64,
    )
    times_sec = times_to_mjd(times, read_time_unit(column_keywords)) * DAY_SECS

    return times_sec, freqs, n_ant


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

    times_sec, freqs, n_ant = _observation_grid(
        ms_path, xds_ms, layout, column_keywords, n_freq
    )

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
    caltable_path: str | None = None,
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

    Once the columns are written, the calibration they were divided by is
    exported beside the results as a CASA table -- see
    :func:`write_gain_caltable`, which is where the two outputs deliberately
    part company over a dead gain. ``caltable_path`` puts that table somewhere
    other than the default ``<results>.B``, and is then the *only* path the
    export knows: what is written, what is refused, and what a superseded table
    is removed from all follow it, so a failed rerun cannot tidy up the default
    it displaced while leaving its own stale table standing.
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

    # A run narrowed with data.freq is a valid run whose results simply do not
    # span the MS. Which channels it holds is not the missing piece -- the zarr's
    # freq coordinate records them, since write_results_xds stores the run's own
    # tab_config.freqs. What is missing is here: nothing below maps a partial
    # band onto the MS's channel axis, and every column is built for the whole
    # of it. Refused until it does, rather than reshaped against the full band.
    if int(xds_ms.sizes["chan"]) != n_freq:
        raise ValueError(
            f"The results hold {n_freq} channels but the MS has "
            f"{int(xds_ms.sizes['chan'])}. A run narrowed with data.freq covers "
            "part of the band, and this writer does not yet use the results' "
            "freq coordinate to place a partial band on the MS's channels, so "
            "exporting one to a full-band measurement set is not yet supported."
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

    try:
        warn_bad_gains(n_bad, bad.size, bad_ants)
        warn_bad_baseline_gains(n_bad_bl, bad_bl_mean.size)
    finally:
        # The same calibration these columns were divided by, as a table standard
        # tooling can apply. Last, and additive -- but in a finally, because the
        # two warnings above can *raise*: a process filtering RuntimeWarning to an
        # error stops on them, and the columns are already written by then. The
        # export would be skipped with the previous run's table still beside the
        # new results, under the current name, describing gains that are no longer
        # there -- the stale calibration this export goes to some length to avoid
        # anywhere else. It contains its own failures (see _emit_gain_caltable),
        # and a warning it raises under such a filter arrives chained to the one
        # that was already propagating rather than replacing it.
        _emit_gain_caltable(
            ms_path, results_zarr_path, gain_table, out_path=caltable_path
        )


#: Extension of the calibration table written beside a results zarr. CASA's own
#: convention for a bandpass-shaped solution: ``<name>.B``.
CALTABLE_EXT = ".B"

#: Lines of traceback carried into a demoted export failure's warning. Enough to
#: name the frame that raised and the one that called it; a full traceback in a
#: warning is unreadable, and the message alone cannot tell a refused MS from a
#: bug in the export.
_TRACEBACK_LINES = 6


def caltable_path(results_zarr_path: str) -> str:
    """Where a results zarr's calibration table goes: the same name, ``.B``.

    One definition, because two things need it: the export, and the clean-up
    that has to find the table an export could not replace.
    """

    # normpath first: a results path with a trailing separator would otherwise
    # put the table *inside* the zarr directory.
    return os.path.splitext(os.path.normpath(results_zarr_path))[0] + CALTABLE_EXT


def _clear_failed_export(out_path: str, ms_path: str) -> str:
    """Remove a table a failed export could not replace. Returns what happened.

    A failure *before* the write -- a multi-window MS, results that do not
    describe it, an output path overlapping it -- leaves the previous run's table
    standing beside the new results, under the current name, which is the
    stale-calibration hazard :func:`_drop_superseded_caltable` exists for reached
    down the other path. The same removal is made here, through the same safe
    remover, and reported in the same warning.

    Never raises: the export has already failed, and a failure to clean up after
    it must not replace that news with its own. The returned sentence is empty
    when there was nothing there at all.
    """

    try:
        if remove_caltable(out_path, ms_path):
            return f" The superseded table at {out_path} has been removed."

        if os.path.exists(out_path):
            return (
                f" {out_path} holds something that is not a calibration table, "
                "so it has been left alone."
            )
    except Exception as err:
        return (
            f" The table at {out_path} could not be removed either: "
            f"{type(err).__name__}: {err}."
        )

    return ""


def _emit_gain_caltable(
    ms_path: str, results_zarr_path: str, gain_table, out_path: str | None = None
):
    """:func:`write_gain_caltable`, with a failure demoted to a warning.

    The export is an extra beside the columns, not a part of them, and it runs
    after they are safely written -- so a failure here has to leave the run's
    result standing rather than ending a completed job with a traceback. What
    gets demoted is everything the export can say about *this* MS or *these*
    results:

    * an MS with more than one spectral window, which the writer serves one
      partition of happily and :func:`~tabascal.ms.write_caltable` refuses,
      since a caltable files every row under one window's id;
    * an output path overlapping the MS, refused by the same guard;
    * something at the output path that is not a calibration table;
    * results whose gains do not describe the MS's antennas or timesteps;
    * a ``TIME`` column declaring a scale tabascal cannot interpret;
    * anything the filesystem refuses.

    All of those fail *before* the table is written, which leaves an earlier
    run's table standing at the destination -- so it is cleared here too, by
    :func:`_clear_failed_export`, and what happened to it is part of the warning.

    The destination is resolved here, once, and handed to both: the export and
    the clean-up after it have to be talking about the same path, or a run given
    an ``out_path`` of its own would remove the default it displaced and leave
    its own superseded table exactly where it reads as this run's solution.

    ``MemoryError`` is not demoted: that is a statement about the process rather
    than about the data, and nothing downstream could reason about a run that
    carried on past it. ``KeyboardInterrupt`` is not caught at all.

    The warning carries the exception type, its message and the tail of its
    traceback, so a bug in the export cannot hide behind the same demotion that
    exists for an MS a caltable cannot describe.
    """

    if out_path is None:
        out_path = caltable_path(results_zarr_path)

    try:
        return write_gain_caltable(
            ms_path, results_zarr_path, out_path, gain_table=gain_table
        )
    except MemoryError:
        raise
    except Exception as err:
        tail = "".join(
            traceback.format_exc().splitlines(keepends=True)[-_TRACEBACK_LINES:]
        )
        # An earlier run's table is still standing at the destination this one
        # could not write, so it goes the same way it would on a no-op export.
        cleared = _clear_failed_export(out_path, ms_path)

        warnings.warn(
            # The destination rather than the results it describes: with an
            # out_path of the caller's the table is not beside them at all, and
            # the path that failed is the one worth naming either way.
            f"The MS columns were written, but the calibration table at "
            f"{out_path} was not: {type(err).__name__}: {err}. The results at "
            f"{results_zarr_path} are unaffected; the solution is simply not "
            f"available as a table to apply.{cleared}\n{tail}",
            RuntimeWarning,
            stacklevel=2,
        )

        return None


def _drop_superseded_caltable(out_path: str, ms_path: str, why: str) -> None:
    """Remove the previous run's table when this run has none to replace it.

    A stale calibration is worse than none: it sits beside the current results
    under the current name and reads as the current solution. Returns ``None``,
    so a no-op export can ``return _drop_superseded_caltable(...)`` and say both
    things at once.

    Only ever removes a calibration table, and never one overlapping the MS --
    :func:`~tabascal.ms.remove_caltable` is where that is decided.
    """

    if remove_caltable(out_path, ms_path):
        warnings.warn(
            f"{out_path} holds a calibration table from an earlier run of these "
            f"results, and this run has none to replace it with ({why}), so it "
            "has been removed. A table left there would read as this run's "
            "solution.",
            RuntimeWarning,
            stacklevel=3,
        )

    return None


@measure_runtime
def write_gain_caltable(
    ms_path: str,
    results_zarr_path: str,
    out_path: str | None = None,
    gain_table=None,
) -> str | None:
    """Export the calibration a run implies as an ``applycal``-compatible table.

    Written beside the results as ``<results>.B``, so that a tabascal solution
    can be consumed by standard tooling -- ``applycal``, CASA, CARAcal/stimela --
    like any other calibration, rather than only as the MS columns
    :func:`write_results_ms` writes. ``out_path`` overrides that name, and once
    given it is the whole answer to "where the table goes": everything below --
    the write, the guards it is refused by, and the removal of a table an
    earlier run left -- acts on it and never on the default it displaced.

    What it holds is the **total** calibration, ``external x fitted``: the
    ordered ``gain_table`` product placed on this observation's grid, times the
    DIE gains the model fitted. One ``applycal`` of this table therefore
    reproduces ``CORRECTED_DATA`` from the MS's data column in a single step,
    which is the whole point of composing them here.

    That works because the two forms of the calibration agree. The columns are
    divided by a per-*baseline* gain and a caltable holds a per-*antenna* one,
    and::

        (g_ext_p g_fit_p) conj(g_ext_q g_fit_q)
            == (g_ext_p conj(g_ext_q)) (g_fit_p conj(g_fit_q))

    is exactly the ``gains_tot`` :func:`write_results_ms` divided by -- provided
    the sample mean commutes with the baseline product, which it does for the one
    sample a MAP run has. With several samples the columns are divided by the
    mean of ``g_p conj(g_q)`` while a per-antenna table can only carry the mean
    of ``g_p``, and the two differ by the covariance between the two antennas'
    gains.

    **A dead gain is flagged here and substituted there.** A gain that is zero or
    non-finite carries no solution, and a caltable can say so: ``FLAG`` set and
    ``CPARAM`` NaN, which is what CASA does and what a reader of this table
    expects. The MS columns cannot say it -- a NaN there is a dropped visibility
    -- so they keep :func:`unit_bad_gains`' substitution and are written
    uncalibrated on that antenna instead. The raw zarr gains are passed through
    here, not the substituted ones, and an antenna no external table could supply
    a gain for is flagged the same way. It is a deliberate divergence: each
    format gets the honest answer it is able to give.

    The table's ``TIME`` is the MS's own column and its ``MEASINFO`` declares the
    MS's own scale, never a hard-coded UTC: declared to declared, the same
    convention the external tables are matched on.

    Returns the path written, or ``None`` when there was no calibration to
    export: no ``gains`` in the results, gains that are not the ``(sample,
    antenna, channel, time)`` grid a caltable can hold, or gains that are exactly
    1 **on every sample** -- a ``UnitaryGains`` run fits none, and a table of ones
    is not a calibration. Every-sample rather than on the mean, because samples
    either side of 1 average to unity while the divisor the columns use, the mean
    of the baseline product, is a real calibration. A run that used external
    tables and fitted nothing is in the same position as a unitary one: the total
    calibration is the tables the caller already has.

    In each of those cases a table left by an *earlier* run of the same results
    is removed (:func:`_drop_superseded_caltable`), because a stale calibration
    under the current name reads as the current one. A run whose export *fails*
    is in the same position, and :func:`_clear_failed_export` does the same there.
    Only a calibration table is ever removed, and never one overlapping the MS.

    Note that ``applycal`` operates on ``DATA``. An MS whose visibilities live in
    a non-standard column cannot consume this table, whatever the table says.
    """

    # Derived before anything is read, because the no-op paths need it too: a
    # rerun that fits nothing has to remove the table the last run left here.
    if out_path is None:
        out_path = caltable_path(results_zarr_path)

    # Closed on the way out: this runs at the end of a long job, and a store left
    # open holds file handles for the rest of it.
    with xr.open_zarr(results_zarr_path) as xds_tab:
        if "gains" not in xds_tab:
            return _drop_superseded_caltable(
                out_path, ms_path, "the results carry no gains"
            )

        stored = xds_tab.gains.data
        corr = xds_tab.attrs.get("corr")

        if stored.ndim != 4:
            return _drop_superseded_caltable(
                out_path,
                ms_path,
                f"the results' gains have {stored.ndim} dimensions, not the four "
                "(sample, antenna, channel, time) a caltable can hold",
            )

        # Both reductions over the stored chunks, in one pass: a posterior is
        # (sample, antenna, channel, time) and can be far larger than the table
        # it reduces to, so neither the mean nor the unity test may materialise
        # it. Unity over the *whole* sample axis rather than over the mean --
        # what says a run fitted no gains is that every sample is 1, where a mean
        # of 1 says nothing: samples either side of it average to unity while the
        # divisor the columns use, the mean of the baseline product, is a real
        # calibration. Inside the store's context, since these are the reads.
        unitary, gains = dask.compute((stored == 1).all(), stored.mean(axis=0))

    if unitary:
        return _drop_superseded_caltable(
            out_path, ms_path, "the fitted gains are unity on every sample"
        )

    gains = np.asarray(gains)
    n_ant_fit, n_freq, n_time_fit = gains.shape

    # A second read of the MS -- the caller above has one open, but this is
    # callable on its own -- and a cheap one: the TIME column and two subtables,
    # no visibilities. daskms hands back plain datasets over its own table cache,
    # so there is nothing here to close.
    xds_list, column_keywords = xds_from_ms(ms_path, column_keywords=True)
    xds_ms = xds_list[0]
    layout = ms_layout(xds_ms)

    times_sec, freqs, n_ant = _observation_grid(
        ms_path, xds_ms, layout, column_keywords, n_freq
    )

    if (n_ant_fit, n_time_fit) != (n_ant, layout.n_time):
        raise ValueError(
            f"The results hold gains for {n_ant_fit} antennas over {n_time_fit} "
            f"timesteps, but the MS has {n_ant} antennas over {layout.n_time}. "
            "The results zarr does not belong to this measurement set."
        )

    paths = normalise_gain_tables(gain_table)

    if paths:
        # Silent: on the normal route the columns have just been written from
        # these same tables on this same grid, and their placement reported the
        # coverage. A second identical report would say nothing new.
        ext, dead = gains_from_tables(
            paths, times_sec, freqs, n_ant=n_ant, verbose=False
        )
        # NaN rather than the unity gains_from_tables substitutes: an antenna no
        # table solved has no solution to record, and NaN is what write_caltable
        # turns into a flag.
        gains = np.where(dead, np.nan, ext) * gains

    write_caltable(
        out_path,
        gains,
        times_sec,
        ms_path=ms_path,
        # The MS's own scale, since times_sec is the MS's own column: declaring
        # UTC over a TAI-declared MS moves every timestamp by the leap seconds.
        time_ref=read_time_scale(column_keywords),
        # Which correlation these gains belong to. Nothing in the caltable format
        # records it, and applying an xx solution to yx data is a silent mistake.
        keywords={} if corr is None else {"FittedCorr": str(corr)},
    )

    print(f"Wrote the total calibration to {out_path} (apply with casatasks.applycal).")

    return out_path


# ---------------------------------------------------------------------------
# Per-satellite RFI decomposition
# ---------------------------------------------------------------------------

def _rfi_vis_reference(tab_config) -> str:
    """The one ``rfi_vis`` component in the run's model list, as its reference.

    The decomposition is only the run's own RFI visibility op run one satellite
    at a time, so it has to be *that* op: a different one would answer a
    question about a model the run did not fit. Which one it was is not recorded
    anywhere but ``model.components``, and if that names none, or more than one,
    there is no answer to give rather than a guess to make.
    """

    components = (tab_config.args.get("model") or {}).get("components") or []
    refs = [
        ref
        for ref in components
        if ref.replace(":", ".").rsplit(".", 1)[0].rsplit(".", 1)[-1] == "rfi_vis"
    ]

    if len(refs) != 1:
        raise ValueError(
            f"data.save_rfi_per_sat needs exactly one rfi_vis component to "
            f"evaluate per satellite, but model.components names {len(refs)}: "
            f"{refs}. The per-satellite decomposition is the run's own RFI "
            "visibility op evaluated one source at a time, so there has to be "
            "exactly one op to evaluate."
        )

    return refs[0]


def wants_rfi_per_sat(vi_pred: dict, tab_config) -> bool:
    """Whether this run asked for a per-satellite decomposition it can make.

    Three things, and the last two are not the option's business: the run has to
    have asked (``data.save_rfi_per_sat``), the prediction has to carry the
    fitted fine grid the decomposition is made from, and there has to be at
    least one real satellite to decompose into. A satellite-free run would
    otherwise store an empty ``src`` axis, which is not a decomposition of
    anything and is a column-less export downstream.

    Evaluated on every process, and identically: the answer decides whether the
    collective evaluations in :func:`rfi_vis_per_sat` are made at all.
    """

    return bool(
        tab_config.args["data"].get("save_rfi_per_sat", False)
        and "rfi_A" in vi_pred
        and int(getattr(tab_config, "n_rfi_real", getattr(tab_config, "n_rfi", 0)))
    )


def _source_mask(r: int, n_rfi: int):
    """A boolean ``(n_rfi, 1, 1, 1)`` mask selecting source ``r`` alone.

    A **global** array under sharding, built through :func:`make_global` on the
    RFI sharding, rather than a ``jnp.arange`` comparison: the operands it is
    combined with are global arrays split over the mesh, and a process-local
    array of the full length mixed into that is the mistake the distributed
    layer exists to avoid -- every process would be describing the whole source
    axis while holding one shard of it.

    Boolean rather than 0/1, because it is used with ``jnp.where`` (see
    :func:`rfi_vis_per_sat`) and never multiplied.
    """

    mask = np.zeros((n_rfi, 1, 1, 1), dtype=bool)
    mask[r] = True

    if sharding_enabled():
        return make_global(mask, rfi_sharding())

    return jnp.asarray(mask)


def rfi_vis_per_sat(vi_pred: dict, tab_config):
    """The fitted RFI visibility, decomposed one satellite at a time.

    The RFI visibility op reduces over the satellite (``n_rfi``) axis, so each
    source's contribution is recovered by evaluating the **same** op with every
    other source's amplitude held at zero -- on the already-fitted fine-grid
    ``rfi_A`` and ``rfi_phase`` carried in ``vi_pred``. No re-fit, no GP
    re-evaluation, and no second model: the component is the run's own, rebuilt
    from ``model.components`` (see :func:`_rfi_vis_reference`).

    **The pieces sum back to ``vis_rfi`` exactly in exact arithmetic, and in
    floating point to a difference that is data-dependent and cannot be bounded
    by the coarse visibilities.** The op reduces over source *and* integration
    sample together; evaluating one source at a time re-associates that single
    reduction, and ``sum_r round(x_r) != round(sum_r x_r)``.

    What the difference is bounded by is the **fine-grid** terms, not the coarse
    values they average to -- and where the fine grid cancels, the two can be
    nothing like each other. A source whose fine samples are ``[A, -A]`` beside
    one whose are ``[1, 0]``: a kernel that sums the sources at each fine sample
    (``RiemannVis`` does, in ``calculate_rfi_vis_fine``) rounds ``A + 1`` back to
    ``A`` and averages to 0, while the per-source evaluation keeps the 1 and
    gives 0.5. The difference is then the whole of the coarse visibility. Which
    kernels it happens in is a property of their accumulation order: the FFI and
    variable kernels accumulate source-major on that input and lose nothing.

    In practice it is round-off: on fitted grids it measures ~2e-16 relative in
    fp64 and ~6e-8 in fp32, orders of magnitude below anything a wrong
    decomposition (a dropped source, a mask on the wrong axis) would produce.
    The caveat is about what can be *promised*, and the promise is qualitative.

    Zeroed rather than sliced, at ``n_rfi`` times the cost of a slice, because
    that is what keeps the RFI-axis sharding intact: the op runs inside
    ``psum_over_rfi``'s ``shard_map``, whose ``in_specs`` require the full
    ``n_rfi`` leading axis, and a one-source slice could not be split across the
    mesh at all. The alternative -- gathering the fine grids to one process --
    would materialise the largest arrays in the run on a single device, which is
    precisely what the sharding exists to avoid. Every process must therefore
    call this: the evaluations are collectives. Only process 0 keeps the result,
    since the op's output is replicated by the ``psum``.

    Padded sources are excluded: sharding pads the satellite list to a multiple
    of the device count with dark dummies, and only ``norad_ids[:n_rfi_real]``
    name a satellite. They contribute exactly zero, so dropping them leaves the
    sum-back above untouched.

    **Process 0 holds the whole result**, ``n_rfi_real`` times the size of
    ``vis_rfi``, before it is handed to the zarr -- the peak memory of the write,
    and the reason the option's cost note names a storage multiplier rather than
    a slice. Streaming each source straight into the store would remove that
    multiplier; it is not done here, because the sink would have to be created
    before the first collective and the results zarr is written after it.

    Returns ``(vis_src, norad_ids)`` with ``vis_src`` of shape ``(n_sample,
    n_rfi_real, n_bl, n_freq, n_time)`` in the model's own dtype -- ``None`` off
    process 0 -- and ``norad_ids`` the satellite behind each ``src`` index.
    """

    comp = import_components([_rfi_vis_reference(tab_config)])[0]()
    comp.setup(tab_config)
    constants = {f"{comp.prefix}/{k}": v for k, v in comp.build_constants().items()}
    forward = comp.build_forward()

    rfi_A, rfi_phase = vi_pred["rfi_A"], vi_pred["rfi_phase"]
    vis_rfi = vi_pred["vis_rfi"]
    n_sample, n_rfi = rfi_A.shape[:2]
    n_real = int(getattr(tab_config, "n_rfi_real", n_rfi))

    # One compile, reused for every source and sample: the mask is an argument
    # rather than a Python constant. Jitted so the masking fuses into the
    # reshape the op makes of rfi_A anyway, instead of leaving a second
    # full-size copy of the fine grid resident beside it.
    @jax.jit
    def one_source(mask, rfi_A, rfi_phase, vis_zero):
        # where, not a multiply by 0/1: a masked source is then exactly zero even
        # where the fitted grid is not finite, since 0 * inf and 0 * nan are nan
        # and one bad source would poison every other source's column -- the same
        # reason the elevation mask in rfi_signal is a where. The phase is masked
        # for the same reason and not only the amplitude: the kernels form
        # exp(i * phase) before multiplying by it, so a non-finite phase gives a
        # non-finite factor whatever the amplitude is.
        state = {
            "rfi_A": jnp.where(mask, rfi_A, 0),
            "rfi_phase": jnp.where(mask, rfi_phase, 0),
            "vis_rfi": vis_zero,
        }

        return forward({}, constrain_rfi_state(state, n_rfi), constants)["vis_rfi"]

    keep = is_process_0()
    vis_src, failure = None, None

    if keep:
        # Allocated before the first evaluation, and a failure here is carried to
        # the end rather than raised on the spot: the evaluations below are
        # collectives, and a rank that leaves the loop early strands every other
        # rank in the next psum until the coordinator times out. Every rank makes
        # every call; only process 0 stores what comes back.
        try:
            vis_src = np.empty(
                (n_sample, n_real) + tuple(vis_rfi.shape[1:]), dtype=vis_rfi.dtype
            )
        except Exception as err:  # pragma: no cover - an allocation this size
            failure = err

    for s in range(n_sample):
        vis_zero = jnp.zeros_like(vis_rfi[s])
        for r in range(n_real):
            vis = one_source(
                _source_mask(r, n_rfi), rfi_A[s], rfi_phase[s], vis_zero
            )

            if vis_src is not None and failure is None:
                try:
                    vis_src[s, r] = np.asarray(vis)
                except Exception as err:  # pragma: no cover - see above
                    failure = err

    # Only now, with every collective made on every rank.
    if failure is not None:
        raise failure

    return vis_src, [int(nid) for nid in tab_config.norad_ids[:n_real]]


def per_sat_sources(xds_tab, results_zarr_path: str):
    """``(rfi_vis_src, norad_ids)`` from a results zarr, or a clear ``ValueError``.

    Everything the writer assumes about the variable is checked here rather than
    left to fail later or not at all. The two silent failures are what this is
    for: an array whose axes are in another order still reshapes into MS rows,
    putting visibilities on the wrong baselines and timesteps; and a ``norad_id``
    coordinate that does not describe the ``src`` axis names the columns after
    the wrong satellites. Both would be read as a decomposition, which is exactly
    the thing this export exists to be trusted about.

    The coordinate's *length* is not checked separately: once its dimensions are
    ``("src",)`` xarray has already tied it to the ``src`` axis, and a store
    where the two disagree cannot be opened at all.
    """

    if "rfi_vis_src" not in xds_tab:
        raise ValueError(
            f"{results_zarr_path} holds no 'rfi_vis_src'. Re-run tabascal with "
            "data.save_rfi_per_sat: true to have the fit store the RFI "
            "visibility split per satellite."
        )

    var = xds_tab.rfi_vis_src

    if var.dims != RFI_PER_SAT_DIMS:
        raise ValueError(
            f"rfi_vis_src has dimensions {var.dims}; expected "
            f"{RFI_PER_SAT_DIMS}. An array on other axes, or in another order, "
            "reshapes into MS rows just as happily and puts every visibility "
            "somewhere else."
        )

    if var.sizes["src"] == 0:
        raise ValueError(
            f"{results_zarr_path} holds an empty 'src' axis: there is no "
            "satellite to write a column for. The run fitted no RFI sources."
        )

    if var.sizes["sample"] == 0:
        raise ValueError(
            f"{results_zarr_path} holds an empty 'sample' axis: there is no "
            "prediction to average. The mean over no samples is NaN, and a "
            "column of NaN would be read as a modelled visibility."
        )

    if "norad_id" not in xds_tab.coords:
        raise ValueError(
            f"{results_zarr_path} holds 'rfi_vis_src' but no 'norad_id' "
            "coordinate, so nothing says which satellite each source is. The "
            "columns are named after it and cannot be named without it."
        )

    norad_id = xds_tab.norad_id

    if norad_id.dims != ("src",):
        raise ValueError(
            f"The norad_id coordinate has dimensions {norad_id.dims}; expected "
            "('src',). It has to name one satellite per source."
        )

    if not np.issubdtype(norad_id.dtype, np.integer):
        raise ValueError(
            f"The norad_id coordinate is {norad_id.dtype}; expected an integer "
            "type. A NORAD id is an integer, and a float one would name the "
            "columns TAB_RFI_58126.0 or, worse, round to another satellite."
        )

    return var.data, [int(nid) for nid in norad_id.values]


@measure_runtime
def write_per_sat_rfi_ms(
    ms_path: str,
    results_zarr_path: str,
    prefix: str = RFI_PER_SAT_PREFIX,
    corr: str | None = None,
):
    """Write each satellite's RFI visibility prediction to its own MS column.

    Reads ``rfi_vis_src`` -- written by a run with ``data.save_rfi_per_sat:
    true`` -- back out of a results zarr and assigns one column per satellite,
    ``<prefix><NORAD id>``, e.g. ``TAB_RFI_58126``. Image one of them to see a
    single satellite's modelled RFI: a genuine satellite is a clean streak in
    exactly one per-source image, while a feature that appears in several is sky
    flux the RFI model has split across satellites -- which reduced chi^2 cannot
    see, since the split costs it nothing.

    Standalone and re-runnable: the zarr and the MS are all it needs, with no
    ``TabConfig``, no re-fit and nothing of the run left in memory. So the
    decomposition can be exported long after the fit, and re-exported.

    The columns are in the **calibrated frame**, the one frame
    :func:`write_results_ms` puts every column in, and they get there the same
    way ``TAB_RFI_DATA`` does: they are model visibilities, fitted to data with
    both gain layers already divided out, so nothing is applied to them here.
    That is what makes

        ``sum over satellites of TAB_RFI_<NORAD> == TAB_RFI_DATA``

    a statement about round-off rather than about frames. Both sides make the
    same ``complex64`` cast, in the same place -- before the sample mean, not
    after it -- and go through the same row mapping, so what this writer adds is
    float32 rounding and nothing else:

        **Given** that the zarr's ``rfi_vis`` is the exact-arithmetic sum over
        sources of ``rfi_vis_src``,

        ``|sum_r TAB_RFI_<NORAD_r> - TAB_RFI_DATA|
            <= (n_src + n_sample + 2) * ulp32(max_s sum_r |rfi_vis_src[s, r]|)``

        per component, the real and imaginary parts separately, since a
        complex64 cast rounds each of them on its own.

    The hypothesis is what makes it a bound rather than a hope, and it is a
    statement about the *zarr*, not about this writer: whether the fit's own
    ``rfi_vis`` really is that sum is :func:`rfi_vis_per_sat`'s subject, and the
    answer there is qualitative -- exact in exact arithmetic, round-off in
    practice, and unbounded in the coarse values under fine-grid cancellation.
    A results zarr carrying such a case exports columns that inherit it.

    **The scale is measured before the sample mean**, on the per-sample,
    per-source values, because that is where the rounding happens. Referencing
    it to the columns instead would be wrong wherever the samples cancel: two
    samples of ``+A`` and ``-A`` average to a column of zero while the total was
    rounded from ``A``, so the columns can be zero and the difference nowhere
    near it. The count is one ulp for each of the ``n_src`` terms cast into its
    column, one for each step of the two sample means, and one for the sum
    itself.

    It is not a bit-for-bit identity, and cannot be: ``sum_r round(x_r)`` is not
    ``round(sum_r x_r)``.

    Correlations that were not fitted are 0, matching the model columns beside
    them; ``corr`` overrides the correlation the zarr recorded, exactly as in
    :func:`write_results_ms`, and is needed only for a zarr written before that
    attribute existed.
    """

    # In multi-process runs only process 0 writes; nothing here is sharded.
    if not is_process_0():
        return

    xds_ms = xds_from_ms(ms_path)[0]
    xds_tab = xr.open_zarr(results_zarr_path)

    src, norad_ids = per_sat_sources(xds_tab, results_zarr_path)
    _, n_src, n_bl, n_freq, n_time = src.shape

    n_corr = xds_ms.sizes["corr"]

    corr_idx = fitted_correlation(
        ms_path,
        xds_tab.attrs.get("corr"),
        corr,
        n_corr,
        partition_polarization(ms_path, xds_ms),
    )

    # Validated before any column is built, for the same reason write_results_ms
    # validates there: a zarr from another MS otherwise surfaces as a dask
    # "chunks do not add up to shape" error from the first reshape.
    layout = ms_layout(xds_ms)

    if layout.n_bl != n_bl:
        raise ValueError(
            f"The results hold {n_bl} baselines but the MS has {layout.n_bl} "
            f"({layout.n_time * layout.n_bl} rows over {layout.n_time} "
            "timesteps). The results zarr does not belong to this measurement "
            "set, or an antenna was dropped between the run and the write."
        )

    if layout.n_time != n_time:
        raise ValueError(
            f"The results hold {n_time} timesteps but the MS has "
            f"{layout.n_time}. The results zarr does not belong to this "
            "measurement set."
        )

    if int(xds_ms.sizes["chan"]) != n_freq:
        raise ValueError(
            f"The results hold {n_freq} channels but the MS has "
            f"{int(xds_ms.sizes['chan'])}. A run narrowed with data.freq covers "
            "part of the band, and there is nothing here that says which part, "
            "so the columns cannot be placed."
        )

    cols = [f"{prefix}{nid}" for nid in norad_ids]

    if len(set(cols)) != n_src:
        repeated = sorted({nid for nid in norad_ids if norad_ids.count(nid) > 1})
        raise ValueError(
            f"The results hold more than one source per NORAD id {repeated}, "
            f"which would write one {prefix}<id> column per pair of sources "
            "instead of one per source -- the later source silently replacing "
            "the earlier. Fit each satellite once."
        )

    dims = ["row", "chan", "corr"]
    chunks = {k: v for k, v in xds_ms.chunks.items() if k in dims}
    # The results live on a length-1 correlation axis until into_corr places
    # them on the MS's, so they must not carry the MS's correlation chunk.
    fit_chunks = {k: v for k, v in chunks.items() if k != "corr"}

    # Cast first and averaged afterwards, which is the order write_results_ms
    # forms TAB_RFI_DATA in. Averaging first would leave the two sides of the
    # sum-back rounding at different points and a multi-sample run's columns
    # would no longer add up to the column they decompose.
    src = src.astype(np.complex64).mean(axis=0)

    for i, col in enumerate(cols):
        vis = _to_ms_column(src[i], dims, fit_chunks, n_freq)
        xds_ms = xds_ms.assign(
            **{
                col: xr.DataArray(
                    into_corr(vis.data, corr_idx, n_corr, 0), dims=dims
                ).chunk(chunks)
            }
        )

    col_keywords = {col: {"UNIT": "Jy"} for col in cols}

    print(f"Writing per-satellite RFI predictions to {cols} columns in MS file.")

    # One fused compute for every column, as the results writer does: the
    # sources share the read of the zarr and the MS's row layout.
    dask.compute(xds_to_table([xds_ms], ms_path, cols, column_keywords=col_keywords))


@measure_runtime
def write_results_xds(
    vi_pred: dict, tab_config, file_path: str, overwrite: bool = True
):

    # Optional per-satellite decomposition of the fitted RFI visibility, one
    # `src` slice per NORAD id, so each source can be imaged on its own -- a
    # diagnostic for sky signal leaking into the RFI model. Off by default: it
    # is ~n_rfi x the rfi_vis storage and n_rfi forward-op evaluations.
    #
    # Above the process-0 guard because it is a collective: the op runs inside
    # psum_over_rfi's shard_map, so every process has to reach it. The workers'
    # copy of the result is None, and they return at the guard below without
    # ever looking at it.
    per_sat = (
        rfi_vis_per_sat(vi_pred, tab_config)
        if wants_rfi_per_sat(vi_pred, tab_config)
        else None
    )

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

    if per_sat is not None:
        vis_src, norad_ids = per_sat
        map_xds = map_xds.assign(
            rfi_vis_src=(
                list(RFI_PER_SAT_DIMS),
                # One chunk per (sample, satellite), which is how it is read:
                # imaging one satellite, or writing one MS column, should not
                # pull the whole decomposition off disk to reach a slice of it.
                da.asarray(vis_src).rechunk((1, 1, -1, -1, -1)),  # type: ignore
            )
        )
        # The satellite behind each `src` index, as an int -- the name the MS
        # columns are written under, and the only thing that says which source
        # is which.
        map_xds = map_xds.assign_coords(norad_id=("src", np.asarray(norad_ids)))

    mode = "w" if overwrite else "w-"

    map_xds.to_zarr(file_path, mode=mode)

    return map_xds
