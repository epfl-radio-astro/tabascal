"""Reading Measurement Sets, and the calibration tables that go beside them.

Everything that knows the MS format lives here. Named for the format rather than
generically (``io.py``) so that a second input format becomes a sibling module
with its own name, instead of accreting into one file the way MS reading
accreted into ``tab_tools.py``.

The CASA caltable block at the end is the one place that knows the calibration
table format, so that gains can be exchanged with standard tooling (``applycal``,
CASA/CARAcal/stimela) instead of ad-hoc ``.npz`` files. It sits here because a
caltable is an MS-shaped thing that only means anything beside its MS, and is
kept a self-contained block so it can become its own module unchanged.

Its scope is the scope tabascal solves for: one spectral window and one
correlation. Both are checked rather than assumed -- a multi-window table would
label one window's gains with another's channels, and a table holding a
different Jones term per polarisation cannot be collapsed to the single gain
tabascal fits.

Gain convention (CASA's, and the only one used here)
----------------------------------------------------
The gain multiplies the *model* to give the *observed* visibility::

    V_obs[p, q] = g_p * conj(g_q) * V_true[p, q]

so calibrating divides it out, and the noise follows the data::

    V_cal      = V_obs / (g_p conj(g_q))
    sigma_cal  = sigma / |g_p conj(g_q)|
    weight_cal = weight * |g_p conj(g_q)|**2      (weight == 1 / sigma**2)

A scalar flux scale ``V_cal = k * V_obs`` is therefore the antenna-independent
gain ``g = k ** -0.5``.
"""

import os
import shutil
import warnings
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

import dask
from daskms import xds_from_ms, xds_from_table

# Re-exported as part of the caltable surface below rather than defined twice:
# g_p conj(g_q) has one definition in this codebase, and the calibration written
# to a caltable has to be the same product the model applied.
from tabascal.interferometry import baseline_gains
from tabascal.noise import (
    NoUsableSigma,
    per_baseline_freq_sigma,
    per_baseline_sigma,
    representative_sigma,
)
from tabascal.time import DAY_SECS, jd_to_datetime, mjd_to_jd, to_utc_jd
from tabascal.timing import measure_runtime


#: CASA Stokes enumeration (casacore ``Stokes.h``), for the correlations that can
#: be selected by name. The MS records which of these it holds in
#: ``POLARIZATION::CORR_TYPE``, so a correlation is identified by its code rather
#: than by where it sits on the data axis.
CORR_TYPES = {
    "i": 1, "q": 2, "u": 3, "v": 4,
    "rr": 5, "rl": 6, "lr": 7, "ll": 8,
    "xx": 9, "xy": 10, "yx": 11, "yy": 12,
}

_CORR_NAMES = {code: name for name, code in CORR_TYPES.items()}


def _corr_name(code: int) -> str:
    """A correlation code as its name, falling back to the raw code."""

    return _CORR_NAMES.get(int(code), f"<code {int(code)}>")


def resolve_data_description(ms_path: str, data_desc_id: int = 0):
    """``(spectral_window_id, polarization_id)`` for a ``DATA_DESC_ID``.

    An MS does not tie its data to row 0 of ``SPECTRAL_WINDOW`` and
    ``POLARIZATION``. It carries a ``DATA_DESC_ID`` per row, and the
    ``DATA_DESCRIPTION`` subtable maps that to the spectral window and
    polarization setups the data actually uses. Those ids are 0 in the common
    single-setup case, which is why assuming 0 usually works -- and why an MS
    with several setups would silently read another one's channel frequencies or
    correlation layout.

    ``xds_from_ms`` partitions by ``(FIELD_ID, DATA_DESC_ID)`` and records the id
    in each partition's attrs, so the caller can say which partition it is
    reading.

    Falls back to ``(0, 0)`` with a warning if ``DATA_DESCRIPTION`` cannot be
    read, which keeps a malformed store loadable.
    """

    try:
        dd = xds_from_table(ms_path + "::DATA_DESCRIPTION")[0]
        spw_ids = np.atleast_1d(np.asarray(dd.SPECTRAL_WINDOW_ID.data.compute()))
        pol_ids = np.atleast_1d(np.asarray(dd.POLARIZATION_ID.data.compute()))

        return int(spw_ids[data_desc_id]), int(pol_ids[data_desc_id])
    except Exception as err:  # pragma: no cover - depends on the MS on disk
        print(
            f"Warning: could not resolve DATA_DESC_ID {data_desc_id} through "
            f"{ms_path}::DATA_DESCRIPTION ({err}); assuming spectral window 0 and "
            "polarization 0."
        )

        return 0, 0


def resolve_correlation(ms_path: str, corr: str, pol_id: int = 0) -> int:
    """Index of ``corr`` on the MS's correlation axis.

    Resolved **by identity, not by position**: the requested correlation is
    mapped to its CASA Stokes code and located in ``POLARIZATION::CORR_TYPE``.

    A full 4-correlation MS lays its correlations out in the conventional order,
    so a fixed ``{xx: 0, xy: 1, yx: 2, yy: 3}`` table happens to work there. It
    does not generalise: an MS written with a single polarisation holds only that
    one, so its correlation axis has length 1 whatever the polarisation is, and
    ``yy`` means index 0 rather than 3. A 2-correlation (XX, YY) MS breaks the
    same table in a different way. Reading ``CORR_TYPE`` covers all three, and
    turns a request for an absent correlation into an error rather than either
    an index error or a silent read of the wrong polarisation.

    Parameters
    ----------
    ms_path : str
        Path to the Measurement Set.
    corr : str
        Correlation name, e.g. ``"xx"``. Case-insensitive.
    pol_id : int, optional
        Row of ``POLARIZATION`` describing the data being read, from
        :func:`resolve_data_description`. Defaults to 0.

    Returns
    -------
    int
        Position of ``corr`` on the data's correlation axis.

    Raises
    ------
    ValueError
        If ``corr`` is not a recognised name, or the MS does not contain it.
    """

    key = str(corr).strip().lower()
    if key not in CORR_TYPES:
        raise ValueError(
            f"Unknown correlation {corr!r}. Supported: {sorted(CORR_TYPES)}."
        )

    # Grouped per row: CORR_TYPE is a variable-shaped CASA column, so setups with
    # different NUM_CORR (a four-correlation row beside a YY-only one) cannot be
    # represented as one ungrouped dataset -- dask-ms would describe the whole
    # subtable with one exemplar row's shape and fail on any row that differs.
    # Grouping gives one dataset per row, each with its own width.
    try:
        pol_rows = xds_from_table(ms_path + "::POLARIZATION", group_cols="__row__")
        corr_type = np.atleast_1d(
            np.asarray(pol_rows[pol_id].CORR_TYPE.data.compute())
        ).ravel()
    except Exception as err:  # pragma: no cover - depends on the MS on disk
        # POLARIZATION is mandatory in the MS v2 spec, so an unreadable one means
        # a broken store. Deliberately not falling back to the conventional
        # {xx: 0, ..., yy: 3} ordering: that guess is the very thing this function
        # exists to remove, and on a single-correlation MS it returns an index off
        # the end of the axis. Better to stop than to read an unknown correlation.
        raise ValueError(
            f"Could not read {ms_path}::POLARIZATION ({err}), so the correlation "
            f"layout is unknown and {key!r} cannot be resolved. POLARIZATION is a "
            "required subtable; the MS looks incomplete."
        ) from err

    wanted = CORR_TYPES[key]
    matches = np.flatnonzero(corr_type == wanted)
    if matches.size == 0:
        available = ", ".join(_corr_name(c) for c in corr_type)
        raise ValueError(
            f"{ms_path} does not contain correlation {key!r}. It holds: {available}."
        )

    return int(matches[0])


#: Time scale assumed when an MS does not say which one its ``TIME`` column uses.
DEFAULT_TIME_SCALE = "utc"


def read_time_scale(column_keywords: dict, column: str = "TIME") -> str:
    """Time scale declared by an MS column, from its ``MEASINFO`` record.

    A Measurement Set records the scale its times are on rather than leaving it
    to convention: the ``TIME`` column carries ``MEASINFO {'type': 'epoch',
    'Ref': 'UTC'}``. ``UTC`` is overwhelmingly the common case, but it is a
    declaration to be read, not a property to be assumed -- an MS may legitimately
    declare ``TAI`` or another scale, and the difference is the accumulated leap
    seconds, 37 s since 2017, which is ~285 km along a LEO satellite's ground
    track.

    Read by :func:`read_ms`, which normalises the times it returns onto UTC, and
    by ``orbit_config``'s preflight epoch helper, which normalises the same way.

    Parameters
    ----------
    column_keywords : dict
        Per-column keyword mapping, as returned by
        ``xds_from_ms(path, column_keywords=True)[1]``.
    column : str, optional
        Column to read the scale from. Defaults to ``"TIME"``.

    Returns
    -------
    str
        The declared scale, lower-cased, or :data:`DEFAULT_TIME_SCALE` when the
        MS does not declare one.
    """

    measinfo = (column_keywords or {}).get(column, {}).get("MEASINFO", {})
    ref = measinfo.get("Ref")

    if not ref:
        print(
            f"Warning: {column} carries no MEASINFO Ref; assuming "
            f"{DEFAULT_TIME_SCALE.upper()} times."
        )
        return DEFAULT_TIME_SCALE

    return str(ref).strip().lower()


#: Units a ``TIME`` column can declare in ``QuantumUnits``, mapped to the two
#: tabascal distinguishes. casacore writes ``'s'``; the longer spellings are
#: accepted for the same reason the scale names are -- the point is to read
#: whatever the MS declares, not to insist on one spelling of it.
TIME_UNITS = {
    "s": "s", "sec": "s", "secs": "s", "second": "s", "seconds": "s",
    "d": "d", "day": "d", "days": "d",
}


def read_time_unit(column_keywords: dict, column: str = "TIME") -> Optional[str]:
    """Unit an MS column declares its times in, from its ``QuantumUnits`` keyword.

    The MS format leaves the unit of ``TIME`` to the column: casacore writes
    seconds and declares ``QuantumUnits ['s']``, but days are equally legal.
    Reading the declaration makes it authoritative and leaves
    :func:`times_to_mjd`'s heuristic as the fallback for the columns that carry
    no declaration.

    Parameters
    ----------
    column_keywords : dict
        Per-column keyword mapping, as returned by
        ``xds_from_ms(path, column_keywords=True)[1]``.
    column : str, optional
        Column to read the unit from. Defaults to ``"TIME"``.

    Returns
    -------
    str or None
        ``"s"`` or ``"d"``, or ``None`` when the MS declares nothing usable --
        which is not an error, only an absence for the caller to infer around.

    Warns
    -----
    UserWarning
        If the column declares a unit that is neither seconds nor days. An
        ignored declaration is worth saying out loud, and worth being able to
        filter and assert on, which a bare print is not.
    """

    units = (column_keywords or {}).get(column, {}).get("QuantumUnits")

    if isinstance(units, str):
        units = [units]

    if units is None or len(units) == 0:
        return None

    declared = str(units[0]).strip()
    unit = TIME_UNITS.get(declared.lower())

    if unit is None:
        warnings.warn(
            f"{column} declares QuantumUnits {declared!r}, which is neither "
            "seconds nor days; reading the unit from the times instead.",
            UserWarning,
            stacklevel=2,
        )

    return unit


#: Largest ``|MJD|`` in days that an observation could plausibly carry: MJD 1e5
#: is the year 2132, and -1e5 is 1585. Only used where there is a single
#: timestamp and so no spacing to read; see :func:`times_to_mjd`.
_MJD_DAY_LIMIT = 1e5


def times_to_mjd(times, unit: Optional[str] = None) -> np.ndarray:
    """An MS ``TIME`` column as Modified Julian Dates in days.

    tabascal works in MJD days throughout, while an MS stores ``TIME`` in the
    unit its column declares -- seconds, as casacore writes it. Pass that
    declaration (:func:`read_time_unit`) and it is honoured; pass ``None`` and
    the unit is inferred, because not every writer fills the keyword in.

    The inference reads the *spacing* of consecutive samples rather than their
    magnitude: an integration is seconds long, so a gap above 0.5 can only be
    seconds, where times stored in days step by ~1e-4. Spacing also stays
    positive for a pre-1858 epoch, whose MJD day number is negative. The
    threshold is strict, so a spacing of exactly 0.5 reads as days.

    A single-integration MS has no spacing to read, so its unit comes from
    magnitude after all: an MJD day number is at most ~1e5 in any plausible
    observing era, while the same instant in seconds is ~1e9. Strict again: a
    magnitude of exactly 1e5 reads as days.

    The classification looks at the *sorted* distinct values, and so does not
    depend on the order the times arrive in -- :func:`ms_layout` permits an MS
    whose timestep blocks do not ascend, and
    ``orbit_config.ms_integration_times_mjd`` reads the same column through
    ``np.unique``. Reading the raw leading pair instead would let those two
    classify one MS two different ways. The values come back in the order they
    were given, sorted or not: only the unit decision looks at them sorted.

    So the heuristic is one rule for both callers. A *declared* unit can still
    part them, because only :func:`read_ms` passes one: an MS whose
    ``QuantumUnits`` contradicts the spacing of the times it stores is read on
    the declaration there and on the spacing by the preflight epoch check.
    """

    times = np.asarray(times, dtype=float)

    if unit is None and times.size:
        ordered = np.unique(times)

        if ordered.size > 1:
            unit = "s" if (ordered[1] - ordered[0]) > 0.5 else "d"
        else:
            unit = "s" if abs(ordered[0]) > _MJD_DAY_LIMIT else "d"

    return times / DAY_SECS if unit == "s" else times


# ---------------------------------------------------------------------------
# Row layout
# ---------------------------------------------------------------------------

class MSLayout(NamedTuple):
    """How an MS partition's rows map onto tabascal's ``(n_time, n_bl)`` grid.

    ``a1``/``a2`` are the antenna pairs of one timestep's block, in row order.
    """

    n_time: int
    n_bl: int
    a1: np.ndarray
    a2: np.ndarray


def _materialise(*arrays):
    """Numpy values of ``arrays``, computing the dask ones in a single pass."""

    lazy = [i for i, a in enumerate(arrays) if hasattr(a, "compute")]
    values = list(arrays)

    if lazy:
        computed = dask.compute(*(arrays[i] for i in lazy))
        for i, value in zip(lazy, computed):
            values[i] = value

    return tuple(np.asarray(v) for v in values)


def ms_layout(xds) -> MSLayout:
    """Derive and validate the row layout of one MS partition.

    tabascal reads every visibility column as ``(n_time, n_bl)``, so the rows
    have to be time-major: ``n_bl`` consecutive rows holding one timestep of a
    fixed baseline sequence, repeated per timestep. That was assumed everywhere
    the reshape appears; it is checked once here instead, for the reader and the
    results writer alike.

    Three distinct ways an MS can break the reshape, each silent on its own:
    a baseline-major store repeats one pair down the first rows; a per-timestep
    reshuffle keeps the row count right while moving each baseline's data; and
    rows that cycle through baselines *and* times together satisfy both of those
    while landing every visibility on the wrong timestamp.

    Nothing the size of a column is ever held in memory. The only values read
    whole are the ``n_time`` distinct times and the first block's ``n_bl``
    antenna pairs; the checks over the full columns are reductions, computed
    together in one pass, chunk by chunk, on dask-backed input.
    """

    times = xds.TIME.data
    a1_col = xds.ANTENNA1.data
    a2_col = xds.ANTENNA2.data

    n_row = int(times.shape[0])
    (unique_times,) = _materialise(np.unique(times))
    n_time = int(unique_times.size)
    n_bl, remainder = divmod(n_row, n_time)

    if remainder:
        raise ValueError(
            f"The MS holds {n_row} rows over {n_time} timesteps, which is not "
            "a whole number of baselines per timestep. tabascal reads "
            "visibilities as (n_time, n_bl) and cannot use this MS."
        )

    a1, a2 = _materialise(a1_col[:n_bl], a2_col[:n_bl])

    if len(set(zip(a1.tolist(), a2.tolist()))) != n_bl:
        raise ValueError(
            f"The first {n_bl} rows do not hold {n_bl} distinct antenna pairs, so "
            "the MS is not ordered time-major. tabascal reads visibilities as "
            "(n_time, n_bl); sort the MS by TIME before running."
        )

    # The first block's pairs broadcast down the block axis; each block's own
    # first time broadcasts along it. Only constancy within a block is
    # required, not ascending block order: the time axis of everything tabascal
    # writes follows the same block order.
    times_2d = times.reshape(n_time, n_bl)
    same_order = (a1_col.reshape(n_time, n_bl) == a1).all() & (
        a2_col.reshape(n_time, n_bl) == a2
    ).all()
    one_time_per_block = (times_2d == times_2d[:, :1]).all()

    same_order, one_time_per_block = (
        bool(flag) for flag in _materialise(same_order, one_time_per_block)
    )

    if not same_order:
        raise ValueError(
            "The baseline order differs between timesteps. tabascal reads "
            "visibilities as (n_time, n_bl) with one fixed baseline order per "
            "timestep; sort the MS by TIME, ANTENNA1, ANTENNA2 before running."
        )

    if not one_time_per_block:
        raise ValueError(
            "The MS rows interleave timesteps within a baseline block: a block "
            f"of {n_bl} consecutive rows holds more than one TIME. tabascal "
            "reads visibilities as (n_time, n_bl), so each block must be a "
            "single timestep; sort the MS by TIME, ANTENNA1, ANTENNA2 before "
            "running."
        )

    return MSLayout(n_time=n_time, n_bl=n_bl, a1=a1, a2=a2)


def partition_setup(ms_path: str, xds) -> tuple:
    """``(spectral_window_id, polarization_id)`` for the partition ``xds``.

    ``xds_from_ms`` partitions by ``(FIELD_ID, DATA_DESC_ID)`` and records the id
    in each partition's attrs, so a partition can say which subtable rows its
    data is described by rather than assuming row 0.
    """

    return resolve_data_description(ms_path, int(xds.attrs.get("DATA_DESC_ID", 0)))


def partition_polarization(ms_path: str, xds) -> int:
    """The ``POLARIZATION`` row the partition ``xds`` uses.

    The half of :func:`partition_setup` a caller needs when it is placing
    correlations rather than reading channel frequencies.
    """

    return partition_setup(ms_path, xds)[1]


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


def _column_values(xds, name: str) -> Optional[np.ndarray]:
    """The values of a column, or ``None`` if it is not there to be read.

    "Not there" covers two cases that a noise read has to treat alike: a column
    the MS does not have, and one CASA has declared without filling any of its
    cells -- reading one of those raises rather than returning anything.
    """

    column = getattr(xds, name, None)

    if column is None:
        return None

    try:
        return np.asarray(column.data.compute())
    except Exception as err:
        print(
            f"Warning: {name} is present in the MS but could not be read ({err}); "
            "treating it as absent."
        )

        return None


def partition_noise(
    xds,
    n_time: int,
    n_bl: int,
    n_freq: int,
    corr_idx: int = 0,
    chans=None,
    n_chan_ms: Optional[int] = None,
) -> Optional[np.ndarray]:
    """The noise on one MS partition's visibilities, as resolved as the MS allows.

    Most specific column first:

    ``SIGMA_SPECTRUM`` ``(row, chan, corr)``
        Noise per (baseline, channel), shape ``(n_bl, n_freq)``. The default,
        because a bandpass is not flat and an MS that has measured that says so.
    ``SIGMA`` ``(row, corr)``
        Per-baseline noise, shape ``(n_bl,)`` -- the band-averaged version of the
        same measurement, and what most MSs carry.

    Either column keeps its time axis if it has one to keep: a column whose rows
    genuinely change over the observation comes back as ``(n_bl, n_freq,
    n_time)``, or ``(n_bl, 1, n_time)`` from ``SIGMA``. See :mod:`tabascal.noise`.

    A ``SIGMA_SPECTRUM`` that is absent, that holds no positive finite value
    anywhere (a column that was never filled in), or that describes a different
    set of channels from the ones being read is not an error: the read falls
    through to ``SIGMA``. A column that contradicts the row layout *is* an error
    -- that is the reader and the MS disagreeing about the grid, which reading
    another column instead would only bury.

    Nothing is invented: if neither column is usable this returns ``None``, after
    saying why, because a made-up noise scale silently re-weights the entire fit.
    ``None`` rather than an exception because ``data.noise`` is read *after* the
    MS -- an override is exactly the answer to an MS with no noise in it, and
    raising here would take its turn away. :meth:`TabConfig.set_noise` is where a
    still-unset noise becomes the error that stops the run.

    ``chans`` narrows ``SIGMA_SPECTRUM`` to the channels being read, and must be
    the same selection the data went through: the noise divides those
    visibilities cell by cell, so a noise left on the full band would weight
    every channel by another channel's. ``SIGMA`` has no channel axis to narrow.
    ``n_chan_ms`` is the MS's own channel count, which the column is validated
    against; it defaults to ``n_freq``, i.e. no selection was made.
    """

    # The column describes the MS's channel axis, so it is checked against the
    # whole band *before* any narrowing. Validating the narrowed column instead
    # would let a spectrum that disagrees with the data pass the moment the
    # selection happened to fit inside it -- and selecting a channel does not
    # settle which of the two is right about the observation.
    band = n_freq if n_chan_ms is None else n_chan_ms

    values = _column_values(xds, "SIGMA_SPECTRUM")

    if values is not None:
        n_chan = values.shape[1] if values.ndim > 1 else 0

        # Channels are handled exactly as read_data handles them, and must stay
        # in lockstep with it: the noise divides those visibilities, cell by
        # cell, so a column covering a different set of channels cannot weight
        # them.
        if n_chan != band:
            print(
                f"Warning: SIGMA_SPECTRUM covers {n_chan} channels but the MS has "
                f"{band}, so it cannot line up with the visibilities; falling "
                "back to the SIGMA column."
            )
        else:
            if chans is not None:
                values = values[:, np.asarray(chans)]
            try:
                return per_baseline_freq_sigma(values, n_time, n_bl, corr_idx)
            except NoUsableSigma as err:
                print(f"Warning: {err} Falling back to the SIGMA column.")

    sigma = _column_values(xds, "SIGMA")

    if sigma is None:
        print(
            "Warning: the MS partition has neither a usable SIGMA_SPECTRUM nor a "
            "readable SIGMA column, so it carries no noise estimate at all; set "
            "data.noise explicitly."
        )

        return None

    # Only the empty column is deferred. A SIGMA that contradicts the row layout
    # still raises: that is the reader and the MS disagreeing about the
    # observation, which no data.noise value makes right.
    try:
        return per_baseline_sigma(sigma, n_time, n_bl, corr_idx)
    except NoUsableSigma as err:
        print(f"Warning: {err}")

        return None


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


# ---------------------------------------------------------------------------
# Column layout: MS rows and channels vs tabascal's (bl, freq, time) grid
# ---------------------------------------------------------------------------

def rows_to_grid(col, n_time: int, n_bl: int, n_freq: int):
    """MS ``(row, chan)`` to tabascal's ``(bl, freq, time)``.

    The rows are time-major -- ``n_bl`` consecutive rows per timestep, in a fixed
    baseline order, as :func:`ms_layout` checks -- so the row axis unfolds into
    ``(n_time, n_bl)`` and the time axis then moves to the back.

    Method calls rather than ``np.``/``jnp.``/``da.`` functions, so the reader can
    pass jax arrays and the writer dask ones through the same mapping.
    """

    return col.reshape(n_time, n_bl, n_freq).transpose(1, 2, 0)


def grid_to_rows(arr, n_freq: int, n_corr: int = 1):
    """tabascal's ``(bl, freq, time)`` back to MS ``(row, chan, corr)``.

    The exact inverse of :func:`rows_to_grid`, and the reason they live next to
    each other: a transpose written once in each direction cannot drift out of
    step the way two independent ones can.
    """

    return arr.transpose(2, 0, 1).reshape(-1, n_freq, n_corr)


@measure_runtime
def read_ms(
    ms_path,
    freq: Optional[float] = None,
    chans: Optional[jax.Array] = None,
    corr: str = "xx",
    data_col: str = "DATA",
):

    xds_list, column_keywords = xds_from_ms(ms_path, column_keywords=True)
    xds = xds_list[0]

    # Which spectral window and polarization setup this partition actually uses.
    # xds_from_ms partitions by (FIELD_ID, DATA_DESC_ID) and records the id in the
    # partition attrs; DATA_DESCRIPTION maps it to the subtable rows.
    spw_id, pol_id = partition_setup(ms_path, xds)

    corr_idx = resolve_correlation(ms_path, corr, pol_id)

    time_scale = read_time_scale(column_keywords)

    xds_ant = xds_from_table(ms_path + "::ANTENNA")[0]
    # Grouped per row for the same reason as POLARIZATION above: CHAN_FREQ and
    # CHAN_WIDTH are variable-shaped, so windows with different channel counts
    # cannot share one ungrouped dataset.
    xds_spec = xds_from_table(ms_path + "::SPECTRAL_WINDOW", group_cols="__row__")
    xds_src = xds_from_table(ms_path + "::SOURCE")[0]

    ants_itrf = np.array(xds_ant.POSITION.data.compute())

    n_ant = ants_itrf.shape[0]
    layout = ms_layout(xds)
    n_time, n_bl = layout.n_time, layout.n_bl
    n_chan_ms, n_corr = xds[data_col].data.shape[1:]

    spec_row = xds_spec[spw_id]
    freqs = np.array(spec_row.CHAN_FREQ.data[0].compute())
    # Per channel, not one number for the window. A spectral window need not be
    # uniform, and the width is what says whether a frequency falls inside a
    # channel -- so one channel's width cannot answer that for another's.
    chan_widths = np.atleast_1d(np.array(spec_row.CHAN_WIDTH.data[0].compute()))
    if chan_widths.size == 1 and len(freqs) > 1:
        chan_widths = np.repeat(chan_widths, len(freqs))
    int_time = xds.INTERVAL.data[0].compute()

    times_mjd = times_to_mjd(
        np.array(xds.TIME.data.reshape(n_time, n_bl)[:, 0].compute()),
        read_time_unit(column_keywords),
    )

    # The declared scale is honoured by normalising to UTC once, here, rather
    # than by threading a scale through the trajectory maths: everything past
    # this point reads UTC Julian Dates -- skyfield through skyfield_time's
    # default, sgp4jax.itrf_to_gcrf, which has no scale concept to be told
    # otherwise, and the TLE epoch checks -- so one conversion covers all of
    # them. times_mjd stays as declared beside it: it is the MS's own column in
    # days, and orbit_config.ms_integration_times_mjd reports the same column
    # the same way, so the two remain comparable.
    times_jd = to_utc_jd(mjd_to_jd(times_mjd), time_scale)

    print(jd_to_datetime(times_jd[0]).isoformat())

    times = jnp.linspace(0, n_time * int_time, n_time, endpoint=False)

    # Which channels to read: an explicit list wins, then `freq` picks the single
    # channel nearest it, and otherwise the whole band. Held as a numpy index
    # array rather than a jax one -- it indexes dask columns, and a scalar index
    # would silently drop the channel axis it is meant to narrow.
    if chans is None and freq is not None:
        nearest = int(np.argmin(np.abs(freq - freqs)))
        offset = abs(float(freq) - float(freqs[nearest]))
        # argmin always lands on a channel, so the *request* has to be checked
        # against the band as well. Without this a frequency from another
        # subband -- or a units slip, GHz written for Hz -- reads the nearest
        # edge channel and says nothing, and everything downstream is then on a
        # channel nobody asked for.
        if offset > 0.5 * abs(float(chan_widths[nearest])):
            raise ValueError(
                f"Requested frequency {float(freq) / 1e6:.4f} MHz is outside the "
                f"band of {ms_path}: its {len(freqs)} channels run "
                f"{freqs[0] / 1e6:.4f} - {freqs[-1] / 1e6:.4f} MHz, and the "
                f"nearest centre is {freqs[nearest] / 1e6:.4f} MHz, "
                f"{offset / 1e6:.4f} MHz away -- more than half that channel's "
                f"width ({abs(float(chan_widths[nearest])) / 1e6:.4f} MHz)."
            )
        chans = nearest

    chan_sel = None if chans is None else np.atleast_1d(np.asarray(chans)).astype(int)

    if chan_sel is not None:
        if chan_sel.size == 0 or chan_sel.min() < 0 or chan_sel.max() >= n_chan_ms:
            raise ValueError(
                f"Channel selection {np.asarray(chans)} is off the {n_chan_ms} "
                f"channels of {ms_path}."
            )
        # The whole band in order is no selection at all; skipping the indexing
        # keeps the default read on the contiguous path daskms is happiest with.
        if chan_sel.size == n_chan_ms and (chan_sel == np.arange(n_chan_ms)).all():
            chan_sel = None

    if chan_sel is not None:
        freqs = freqs[chan_sel]
        chan_widths = chan_widths[chan_sel]
        print(
            f"Reading {len(freqs)} of {n_chan_ms} channels "
            f"({freqs[0] / 1e6:.3f} - {freqs[-1] / 1e6:.3f} MHz)"
        )

    n_freq = len(freqs)

    # The same selection the data goes through: a noise on the full band would
    # weight every visibility by a channel it did not come from. The MS's own
    # channel count goes with it, since that is what the column is validated
    # against -- narrowing cannot make a disagreeing column agree.
    sigma = partition_noise(
        xds, n_time, n_bl, n_freq, corr_idx, chans=chan_sel, n_chan_ms=n_chan_ms
    )

    def read_data(col_name):
        col = xds[col_name].data[:, :, corr_idx]
        if chan_sel is not None:
            col = col[:, chan_sel]
        return rows_to_grid(jnp.array(col.compute()), n_time, n_bl, n_freq)

    data = {
        **{
            key: val
            for key, val in zip(
                ["ra", "dec"], jnp.rad2deg(xds_src.DIRECTION.data[0].compute())
            )
        },
        "n_freq": n_freq,
        "n_corr": n_corr,
        "n_time": n_time,
        "n_ant": n_ant,
        "n_bl": n_bl,
        "dish_d": xds_ant.DISH_DIAMETER.data[0].compute(),
        # As the MS declares them, on the scale it declares them on.
        "times_mjd": times_mjd,
        # The same instants on UTC, which is what everything downstream reads.
        "times_jd": times_jd,
        "times": times,
        "time_scale": time_scale,
        "int_time": int_time,
        "freqs": freqs,
        # The scalar stays what it always was -- the first width being read,
        # signed as CHAN_WIDTH stores it, since the fine-grid construction steps
        # along the band with it. chan_widths is the per-channel magnitude, for
        # anything asking whether a frequency falls inside a given channel.
        "chan_width": chan_widths[0],
        "chan_widths": np.abs(chan_widths),
        "ants_itrf": ants_itrf,
        "uvw": jnp.array(xds.UVW.data.reshape(n_time, n_bl, 3).compute()),
        "vis_obs": read_data(data_col),
        "flags": read_data("FLAG"),
        # Per (baseline, channel) where the MS resolves it that far, per baseline
        # otherwise, and per timestep on top of either where the column varies in
        # time -- never collapsed to a scalar: the antennas differ in
        # sensitivity and the band is not flat, so a single number mis-weights
        # every visibility. See tabascal.noise. None when the MS carries no
        # usable noise column at all, which data.noise is then read to supply.
        "noise": None if sigma is None else jnp.asarray(sigma),
        "noise_scalar": None if sigma is None else representative_sigma(sigma),
        "a1": jnp.array(layout.a1),
        "a2": jnp.array(layout.a2),
    }

    return data


def get_observation_data_type(data_col: str):

    ast = ["DATA", "CAL_DATA", "AST_DATA", "AST_MODEL_DATA"]
    rfi = ["DATA", "CAL_DATA", "RFI_DATA", "RFI_MODEL_DATA"]
    gains = ["DATA"]

    data_type = {
        "ast": data_col in ast,
        "rfi": data_col in rfi,
        "gains": data_col in gains,
    }

    return data_type


# ---------------------------------------------------------------------------
# CASA calibration tables
#
# Self-contained: nothing above depends on anything below, and the casacore
# imports are function-local so that reading an MS through dask-ms does not pull
# python-casacore in eagerly. See the gain convention in the module docstring.
# ---------------------------------------------------------------------------

#: Subtables a caltable carries; CASA copies these straight from the MS.
_SUBTABLES = ("ANTENNA", "FIELD", "SPECTRAL_WINDOW", "OBSERVATION", "HISTORY")


def _caltable_desc():
    """Table description matching ``casatasks.gaincal``'s output."""

    from casacore.tables import makearrcoldesc, makescacoldesc, maketabdesc

    time = makescacoldesc("TIME", 0.0, valuetype="double")
    time["desc"]["keywords"] = {
        "QuantumUnits": ["s"],
        "MEASINFO": {"type": "epoch", "Ref": "UTC"},
    }
    interval = makescacoldesc("INTERVAL", 0.0, valuetype="double")
    interval["desc"]["keywords"] = {"QuantumUnits": ["s"]}

    scalars = [
        time,
        makescacoldesc("FIELD_ID", 0, valuetype="int"),
        makescacoldesc("SPECTRAL_WINDOW_ID", 0, valuetype="int"),
        makescacoldesc("ANTENNA1", 0, valuetype="int"),
        makescacoldesc("ANTENNA2", 0, valuetype="int"),
        interval,
        makescacoldesc("SCAN_NUMBER", 0, valuetype="int"),
        makescacoldesc("OBSERVATION_ID", 0, valuetype="int"),
    ]
    arrays = [
        makearrcoldesc("CPARAM", 0j, ndim=2, valuetype="complex"),
        makearrcoldesc("PARAMERR", 0.0, ndim=2, valuetype="float"),
        makearrcoldesc("FLAG", False, ndim=2, valuetype="boolean"),
        makearrcoldesc("SNR", 0.0, ndim=2, valuetype="float"),
        # CASA declares WEIGHT but leaves it unfilled; mirror that.
        makearrcoldesc("WEIGHT", 0.0, ndim=2, valuetype="float"),
    ]

    return maketabdesc(scalars + arrays)


def _path_identity(path: str):
    """``(st_dev, st_ino)`` of *path*, or ``None`` where nothing is there to stat.

    The filesystem's own answer to "which directory is this", which is the only
    one that survives the ways a single directory can be spelled.
    """

    try:
        stat = os.stat(path)
    except OSError:
        return None

    return (stat.st_dev, stat.st_ino)


def _same_directory(a: str, b: str) -> bool:
    """Whether two paths name one directory.

    Equal strings settle it; otherwise the filesystem does, because a
    case-insensitive one (APFS, NTFS) resolves ``X.ms`` and ``x.ms`` to the same
    directory while ``realpath`` hands back whichever spelling it was given. The
    strings differ there and the inode does not.

    Compared through :func:`_path_identity` rather than ``os.path.samefile``, so
    that a path disappearing between the look and the comparison answers "not
    the same directory" instead of raising from inside a validation check.
    """

    if a == b:
        return True

    a_id = _path_identity(a)

    return a_id is not None and a_id == _path_identity(b)


def _is_inside(inner: str, outer: str) -> bool:
    """Whether *inner* is *outer* or sits somewhere beneath it.

    Walks *inner* up to the root, comparing each ancestor against *outer* by
    ``(st_dev, st_ino)`` first and by name second. The inode comparison is the
    authority -- it is what catches an ancestor reached by a different spelling
    of the same directory -- and the name comparison covers the part of the walk
    with nothing on disk to stat, such as a fresh output path under an MS that
    does not exist either.

    An ancestor walk, never a string prefix test: that is what keeps
    ``/data/x.ms2`` from counting as a child of ``/data/x.ms``.
    """

    outer_id = _path_identity(outer)
    current = inner

    while True:
        if outer_id is not None and _path_identity(current) == outer_id:
            return True

        if current == outer:
            return True

        parent = os.path.dirname(current)
        if parent == current:  # reached the root
            return False

        current = parent


def _reject_overlapping_paths(path: str, ms_path: str) -> None:
    """Refuse an output path that is, contains, or sits inside the MS.

    ``overwrite=True`` removes the output outright, and the subtables are copied
    out of the MS *afterwards*, so an output that is the MS -- or an ancestor of
    it -- deletes the observation before anything is read from it. An output
    nested inside the MS is the milder form of the same mistake: it writes into
    the directories it is about to copy from.

    Decided by the filesystem rather than by comparing paths as text. One
    directory has many spellings -- a symlink, a ``..``, and on a
    case-insensitive filesystem a different case -- and a guard that trusts the
    strings hands the caller a way to delete the observation.
    """

    real_path = os.path.realpath(path)
    real_ms = os.path.realpath(ms_path)

    if _same_directory(real_path, real_ms):
        raise ValueError(
            f"path and ms_path resolve to the same directory ({real_path}). "
            "Writing the caltable there would delete the Measurement Set it is "
            "written from."
        )

    if _is_inside(real_path, real_ms):
        raise ValueError(
            f"path ({real_path}) is inside the Measurement Set at {real_ms}. A "
            "caltable written there would be writing into the subtables it "
            "copies from; put it beside the MS instead."
        )

    if _is_inside(real_ms, real_path):
        raise ValueError(
            f"path ({real_path}) would contain the Measurement Set at {real_ms}. "
            "Writing the caltable there would delete the observation before its "
            "subtables could be copied."
        )


def write_caltable(
    path: str,
    gains: NDArray,
    times: NDArray,
    ms_path: str,
    interval: float = 0.0,
    n_pol: int = 2,
    viscal: str = "B Jones",
    overwrite: bool = True,
) -> str:
    """Write an ``applycal``-compatible calibration table.

    The layout mirrors exactly what ``casatasks.gaincal`` emits (verified against
    a reference table it produced): one row per (time, antenna), time-major, with
    ``CPARAM`` of shape ``(n_chan, n_pol)``. ``B Jones`` rather than ``G Jones``
    because the gains here are frequency dependent, which a scalar G table cannot
    represent.

    Verified against CASA: ``applycal`` accepts the table, and its
    ``CORRECTED_DATA`` reproduces ``V / (g_p conj(g_q))`` to 6e-7 relative
    (float32 round-off) over 6.7e6 visibilities.

    **Do not rely on applycal to set the weights for a frequency-dependent gain.**
    ``applycal(calwt=True)`` was measured to apply a single per-row weight factor,
    constant across channels (within-row CV of the applied factor = 0.0000), even
    when ``WEIGHT_SPECTRUM`` exists -- it collapses the frequency axis rather than
    scaling each channel by its own ``|g_ch|**2``. For a channel-constant gain
    that is exact; for a frequency-dependent one it is an approximation. tabascal
    therefore computes ``WEIGHT_SPECTRUM = 1 / sigma_cal**2`` per channel itself
    when it writes results.

    A gain that is zero or non-finite carries no solution, and *both* halves of
    that are written: ``FLAG`` is set and ``CPARAM`` is NaN. A reader going by
    the flag and one going by the value have to reach the same conclusion --
    a zero left in ``CPARAM`` reads as a solution that calibrates to infinity,
    and an Inf reads as a number too.

    One spectral window only. Every row is written with
    ``SPECTRAL_WINDOW_ID = 0``, so an MS with more than one window is rejected
    rather than having one window's gains filed under another's id.

    Parameters
    ----------
    path : str
        Output caltable path.
    gains : NDArray
        ``(n_ant, n_freq, n_time)`` complex -- ``g_p``, in the
        ``V_obs = g_p conj(g_q) V_true`` convention of the module docstring.
    times : NDArray
        ``(n_time,)`` MS ``TIME`` values in seconds (MJD seconds, as in the MS).
    ms_path : str
        The MS these gains belong to; its ``ANTENNA``, ``FIELD``,
        ``SPECTRAL_WINDOW``, ``OBSERVATION`` and ``HISTORY`` subtables are copied
        into the caltable, as CASA does. Subtables it does not have are skipped.
    n_pol : int, optional
        CASA writes 2 polarisations even for a single-correlation MS, so the gain
        is duplicated across the pol axis by default.

    Returns
    -------
    str
        The caltable path.

    The gains are checked against the MS they claim to belong to before any of
    this happens, since the copied subtables are what the table's own rows index:
    ``ANTENNA1`` into ``ANTENNA``, and ``CPARAM``'s channel axis onto
    ``CHAN_FREQ``. Gains of the wrong width would produce a table that disagrees
    with the copy of the MS inside itself.

    ``path`` may not be, contain, or sit inside ``ms_path``: the output is
    removed before the subtables are copied out of the MS, so an overlapping
    path would destroy the observation. That is rejected before any of it
    happens, which is also what keeps the clean-up on the failure path from
    reaching anything but the caltable's own directory.

    Raises
    ------
    ValueError
        If the arguments do not describe one single-spectral-window solution set
        for this MS. **Every check on the caller's arguments runs before an
        existing table is removed**: ``overwrite=True`` deletes a calibration
        that took a run to produce, and a caller's mistake must not cost them
        that. An I/O failure part-way through the write cannot put the old table
        back; the partial output is then removed on a best-effort basis before
        the error is re-raised, so a half-written table can only survive a
        failure that also prevents its removal. The original exception always
        propagates -- nothing raised during the clean-up replaces it.
    FileExistsError
        If ``path`` exists and ``overwrite`` is false.
    """

    from casacore.tables import table

    gains = np.asarray(gains)
    if gains.ndim != 3:
        raise ValueError(f"gains must be (n_ant, n_freq, n_time), got {gains.shape}")
    n_ant, n_freq, n_time = gains.shape

    # Checked here rather than left to np.isfinite half-way through the write,
    # which is past the point of no return for the table being overwritten.
    if not np.issubdtype(gains.dtype, np.number):
        raise ValueError(
            f"gains must be a numeric array to be written as complex gains, got "
            f"dtype {gains.dtype}"
        )

    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size != n_time:
        raise ValueError(
            f"times must be one-dimensional of length n_time = {n_time} to match "
            f"the gains, got shape {times.shape}"
        )

    # bool first: it subclasses int, so True would otherwise be accepted as 1.
    if (
        isinstance(n_pol, (bool, np.bool_))
        or not isinstance(n_pol, (int, np.integer))
        or n_pol < 1
    ):
        raise ValueError(
            f"n_pol must be a positive integer, got {n_pol!r}. CASA writes 2 "
            "polarisations even for a single-correlation MS."
        )

    try:
        interval = float(interval)
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"interval must be a number of seconds, got {interval!r}"
        ) from err

    if not isinstance(viscal, str):
        raise ValueError(
            f"viscal must be a string naming the calibration type, got {viscal!r}"
        )

    # Checked rather than taken for its truthiness, because the truthy values are
    # the dangerous ones: overwrite="False" reads as a refusal and deletes the
    # table the caller was trying to protect.
    if not isinstance(overwrite, (bool, np.bool_)):
        raise ValueError(
            f"overwrite must be True or False, got {overwrite!r}. It decides "
            "whether an existing calibration table is deleted, so it is not "
            "taken on truthiness."
        )

    # Before anything is read from disk, let alone removed from it: the output
    # must not overlap the MS it is written from.
    _reject_overlapping_paths(path, ms_path)

    # The gains have to describe the MS whose subtables are about to be copied in
    # beside them, since the table's own rows index those copies.
    ms_spw = os.path.join(ms_path, "SPECTRAL_WINDOW")
    if os.path.exists(ms_spw):
        chan_freq = _single_spw_chan_freq(ms_spw, f"{ms_path}::SPECTRAL_WINDOW")

        if n_freq != len(chan_freq):
            raise ValueError(
                f"gains cover {n_freq} channels but {ms_path}::SPECTRAL_WINDOW "
                f"describes {len(chan_freq)}. The caltable carries a copy of that "
                "subtable, so its CPARAM channel axis would not line up with its "
                "own channel frequencies."
            )

    n_ms_ant = _ms_antenna_count(ms_path)
    if n_ms_ant is not None and n_ant != n_ms_ant:
        raise ValueError(
            f"gains cover {n_ant} antennas but {ms_path}::ANTENNA holds "
            f"{n_ms_ant} rows. The caltable carries a copy of that subtable, so "
            "its ANTENNA1 ids would not index its own antenna table."
        )

    # Everything above is validation; only now is anything on disk touched.
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(f"{path} already exists. Use overwrite=True.")
        shutil.rmtree(path)

    n_row = n_time * n_ant

    # Row order is time-major: (t0,a0), (t0,a1), ... -- matching gaincal.
    ant_idx = np.tile(np.arange(n_ant, dtype=np.int32), n_time)
    time_col = np.repeat(times, n_ant)

    # (n_ant, n_freq, n_time) -> (n_row, n_freq), then duplicated across pol.
    g_rows = np.transpose(gains, (2, 0, 1)).reshape(n_row, n_freq)
    bad = ~np.isfinite(g_rows) | (g_rows == 0)
    solved = np.where(bad, complex(np.nan, np.nan), g_rows)
    cparam = np.repeat(solved[:, :, None], n_pol, axis=2).astype(np.complex64)
    flag = np.repeat(bad[:, :, None], n_pol, axis=2)

    # Past here the old table is gone and cannot be brought back, so the one
    # guarantee left is that a failure leaves no half-written table where a valid
    # one is expected. BaseException, not Exception: a Ctrl-C mid-write is
    # exactly the case that would otherwise strand one.
    try:
        with table(path, _caltable_desc(), nrow=n_row, ack=False) as tb:
            tb.putcol("TIME", time_col)
            tb.putcol("ANTENNA1", ant_idx)
            tb.putcol("ANTENNA2", np.full(n_row, -1, dtype=np.int32))
            tb.putcol("FIELD_ID", np.zeros(n_row, dtype=np.int32))
            tb.putcol("SPECTRAL_WINDOW_ID", np.zeros(n_row, dtype=np.int32))
            tb.putcol("SCAN_NUMBER", np.zeros(n_row, dtype=np.int32))
            tb.putcol("OBSERVATION_ID", np.zeros(n_row, dtype=np.int32))
            tb.putcol("INTERVAL", np.full(n_row, interval))
            tb.putcol("CPARAM", cparam)
            tb.putcol("FLAG", flag)
            tb.putcol("PARAMERR", np.zeros((n_row, n_freq, n_pol), dtype=np.float32))
            tb.putcol("SNR", np.ones((n_row, n_freq, n_pol), dtype=np.float32))

            # CASA identifies a caltable by its table INFO record, not by its
            # keywords -- without this, applycal rejects the table with 'is not a
            # valid Calibration table'.
            tb.putinfo({"type": "Calibration", "subType": viscal, "readme": ""})

            tb.putkeyword("VisCal", viscal)
            tb.putkeyword("ParType", "Complex")
            tb.putkeyword("MSName", os.path.basename(os.path.normpath(ms_path)))
            tb.putkeyword("PolBasis", "unknown")

            for sub in _SUBTABLES:
                src = os.path.join(ms_path, sub)
                if not os.path.exists(src):
                    continue
                dst = os.path.join(path, sub)
                # Not casacore's tablecopy(): that opens the source table and
                # never closes it. Same copy, with both handles released.
                with table(src, ack=False) as source:
                    source.copy(dst).close()
                tb.putkeyword(sub, "Table: " + os.path.abspath(dst))

            tb.flush()
    except BaseException:
        # Best effort, and deliberately silent: whatever went wrong with the
        # write is what the caller needs to see, so nothing raised while clearing
        # up is allowed to take its place.
        try:
            shutil.rmtree(path, ignore_errors=True)
        except BaseException:
            pass
        raise

    return path


def _single_spw_chan_freq(spw_path: str, described: str) -> NDArray:
    """``CHAN_FREQ`` of the one spectral window in the table at *spw_path*.

    A single spectral window is the supported scope, and it is checked rather
    than assumed. A caltable's rows all carry ``SPECTRAL_WINDOW_ID = 0`` and its
    frequencies are read back from row 0, so a second window has nowhere to go:
    its gains would be filed under the first window's id, and read out against
    the first window's channels, with nothing to say they were wrong.
    """

    from casacore.tables import table

    with table(spw_path, ack=False) as spw:
        n_spw = spw.nrows()

        if n_spw != 1:
            raise ValueError(
                f"{described} holds {n_spw} spectral windows. tabascal writes and "
                "reads single-spectral-window calibration tables -- every row "
                "carries SPECTRAL_WINDOW_ID 0 and the channel frequencies come "
                "from window 0 -- so a multi-window table would silently label "
                "one window's gains with another window's channels. Split by "
                "spectral window first."
            )

        return np.asarray(spw.getcell("CHAN_FREQ", 0), dtype=float)


def _ms_antenna_count(ms_path: str) -> Optional[int]:
    """Number of rows in an MS's ``ANTENNA`` subtable, or ``None`` if it has none.

    ``None`` rather than an error: the subtables are optional, and a caltable
    written for an MS that carries none has nothing to contradict.
    """

    from casacore.tables import table

    ant_path = os.path.join(ms_path, "ANTENNA")
    if not os.path.exists(ant_path):
        return None

    with table(ant_path, ack=False) as ant:
        return int(ant.nrows())


def _caltable_freqs(path: str) -> Optional[NDArray]:
    """Channel frequencies from a caltable's own ``SPECTRAL_WINDOW``, if it has one.

    A caltable carries a copy of the MS's ``SPECTRAL_WINDOW``, which is what makes
    it self-describing: the gains can be placed on their channels without the MS
    that produced them. ``None`` when the subtable was not copied, since a table
    written for an MS that had none is still a valid caltable.
    """

    spw_path = os.path.join(path, "SPECTRAL_WINDOW")
    if not os.path.exists(spw_path):
        return None

    return _single_spw_chan_freq(spw_path, f"{path}::SPECTRAL_WINDOW")


def _collapse_pols(cparam: NDArray, flag: NDArray) -> NDArray:
    """One gain per ``(row, channel)`` from a caltable's polarisation axis.

    tabascal fits one correlation and :func:`write_caltable` duplicates that one
    solution across the pol axis, so collapsing it again is a no-op for our own
    tables. A caltable from CASA can hold a genuinely different Jones term per
    polarisation, though, and averaging those would return a gain that calibrates
    neither -- so the unflagged polarisations are *required to agree* rather than
    quietly reduced. Reading per-polarisation gains is issue #151.

    A flagged polarisation is missing, not zero: where one pol holds a solution
    and the other does not, the surviving one is the answer. Only an entry with
    no unflagged polarisation left is NaN.
    """

    valid = ~np.asarray(flag, dtype=bool)
    n_valid = valid.sum(axis=-1)

    # Every unflagged pol is compared against the first unflagged one. Where only
    # one is valid that compares it with itself; where none is, the comparison is
    # masked out entirely, so junk under a flag says nothing.
    first = np.argmax(valid, axis=-1)[..., None]
    reference = np.take_along_axis(cparam, first, axis=-1)
    disagrees = valid & ~np.isclose(cparam, reference)

    if disagrees.any():
        n_entry = int(disagrees.any(axis=-1).sum())
        raise ValueError(
            f"The caltable holds different solutions per polarisation at {n_entry} "
            "(row, channel) entries. tabascal fits a single correlation, so "
            "reading per-polarisation gains is not yet supported -- see issue "
            "#151. Select one polarisation, or split the table, before reading."
        )

    # Summed over the valid pols only. A flagged cell's CPARAM is NaN, so
    # averaging it in would erase the solution the other polarisation still holds.
    total = np.where(valid, cparam, 0).sum(axis=-1)

    return np.where(n_valid > 0, total / np.maximum(n_valid, 1), np.nan)


def read_caltable(path: str) -> dict:
    """Read a caltable written by :func:`write_caltable` (or by CASA).

    Returns a dict with ``gains`` ``(n_ant, n_freq, n_time)`` complex -- flagged
    solutions set to NaN -- plus ``times``, ``ant_idx``, ``freqs`` (``None`` if
    the table carries no ``SPECTRAL_WINDOW``) and ``viscal``.

    Reads the single-correlation, single-spectral-window tables tabascal fits.
    A table whose polarisations carry genuinely different solutions, or which
    describes more than one spectral window, is an error rather than a silent
    collapse -- see :func:`_collapse_pols` and :func:`_single_spw_chan_freq`.
    """

    from casacore.tables import table

    with table(path, ack=False) as tb:
        time_col = tb.getcol("TIME")
        ant1 = tb.getcol("ANTENNA1")
        cparam = tb.getcol("CPARAM")  # (n_row, n_freq, n_pol)
        flag = tb.getcol("FLAG")
        viscal = tb.getkeyword("VisCal") if "VisCal" in tb.keywordnames() else ""

    g = _collapse_pols(cparam, flag)  # (n_row, n_freq)

    times = np.unique(time_col)
    n_ant = int(ant1.max()) + 1
    n_time, n_freq = len(times), g.shape[1]

    gains = np.full((n_ant, n_freq, n_time), np.nan, dtype=complex)
    t_idx = np.searchsorted(times, time_col)
    gains[ant1, :, t_idx] = g

    return {
        "gains": gains,
        "times": times,
        "ant_idx": np.arange(n_ant),
        "freqs": _caltable_freqs(path),
        "viscal": viscal,
    }


def apply_gains_to_data(
    vis: NDArray,
    gains: NDArray,
    a1: NDArray,
    a2: NDArray,
    sigma: NDArray | float | None = None,
):
    """Divide the gains out of the data (and carry the noise with it).

    This is the whole convention in one place -- see the module docstring.

    ``vis`` is ``(n_bl, n_freq, n_time)``; ``gains`` is ``(n_ant, n_freq,
    n_time)``; ``sigma`` is anything broadcastable against ``vis`` (a scalar, or
    ``(n_bl, 1, 1)``).

    Returns ``(vis_cal, sigma_cal)``; ``sigma_cal`` is ``None`` if no ``sigma``
    was given. A baseline whose gain is dead -- flagged, zero or non-finite --
    has no calibrated value, and comes back as NaN in both; such visibilities
    must be flagged by the caller.
    """

    g_bl = baseline_gains(np.asarray(gains), np.asarray(a1), np.asarray(a2))

    # Every kind of dead gain is made the one NaN the caller is told to expect.
    # Left alone, a zero gain divides to Inf and an Inf gain divides to zero, and
    # both of those read downstream as ordinary numbers -- so a caller flagging
    # on isnan would keep them.
    dead = ~np.isfinite(g_bl) | (g_bl == 0)
    g_bl = np.where(dead, np.nan, g_bl)

    with np.errstate(divide="ignore", invalid="ignore"):
        vis_cal = np.asarray(vis) / g_bl
        sigma_cal = None if sigma is None else np.asarray(sigma) / np.abs(g_bl)

    return vis_cal, sigma_cal
