"""Placing externally-solved calibration tables on an observation's grid.

:mod:`tabascal.ms` knows the caltable *format*; this module knows what to do
with one. A table is solved on whatever grid the calibrator chose -- a scan's
worth of times, the master MS's channels -- and the observation being fitted
sits on its own, so the gains have to be placed on that grid before they can be
divided out.

How that is done is a choice, and the wrong choice changes the answer quietly.

Amplitude and unwrapped phase, never real and imaginary
-------------------------------------------------------
Interpolating the real and imaginary parts is the obvious implementation and is
wrong: two unit gains 60 degrees apart average to ``|g| = 0.87``, so the data
would be divided by a gain no antenna ever had and the flux scale would move by
13 %. Amplitude and phase are interpolated separately, and the phase is
*unwrapped* first: a B table winds across the band through a residual delay and
around the ``+-pi`` branch cut in time, and interpolating the stored angles
across such a step averages the two sides of the cut into a gain pointing the
wrong way.

One phase surface, not a stack of branches
-------------------------------------------
The unwrap has to be *two-dimensional*, and the phase has to stay a real
surface through both interpolation stages. Unwrapping along frequency and then
rebuilding a complex gain throws the branch away; the time stage then takes
``angle()`` again and picks a branch per channel, which tears the band by a
whole turn at whichever channel happened to cross the cut. So the phase is
unwrapped along frequency within each timestep, the timesteps are brought onto
one branch by whole turns (which is also the unwrap along time), and
``exp(i phase)`` is applied exactly once, at the end.

The shortest branch, and what that assumes
-------------------------------------------
Unwrapping recovers a phase only where the solutions *sample* it. A genuine
change of more than pi between two adjacent solved samples -- between two solved
channels, or between two solution intervals of a phase slewing faster than the
calibration cadence follows -- is not in the table at all: complex samples carry
the phase modulo 2 pi, so such a step aliases to the shorter branch and is taken
as such. A true 0 -> 1.5 pi evolution between two solutions interpolates to
-0.25 pi at the midpoint, not +0.75 pi. This is ``np.unwrap``'s assumption,
applied band-coherently rather than line by line, and **nothing in the data can
detect when it fails**: both solved intervals have to sample the phase below
half a turn, which is a requirement on the calibration that produced the table
rather than something the placement can check. The half-turn tie in
:func:`_align_timesteps` is the boundary case of exactly this ambiguity.

The interpolation is linear and separable -- frequency first, then time -- and
**extrapolation holds the edge value**: a table that does not reach the start of
the observation calibrates it with the earliest solution it does have, which is
the assumption the calibrator itself made in solving over an interval.

Flagged solutions are absences, not zeros
------------------------------------------
A flagged (NaN) or zero entry is not a support point: the interpolation bridges
across it from the solutions either side, exactly as it bridges a coordinate the
table never sampled. Only an antenna with no valid solution *anywhere* has
nothing to interpolate from; its gain is 1 and it is reported ``dead``, so the
caller can flag those visibilities rather than divide by a number nobody solved.

Every output is classified by the support it was actually built from, line by
line, and the provenance of the frequency stage is carried through the time
stage -- so a table whose solutions run along a diagonal reports the edge-holds
it really performed rather than the rectangle its two axes span.

Several tables compose on the grid, not before it
--------------------------------------------------
``data.gain_table`` takes an ordered list, and each table is placed on the
observation's grid *before* the product is formed. The two orders do not agree:
two amplitudes ramping 1 -> 3 give ``2 * 2 = 4`` half way when each is
interpolated first, and ``(1 + 9) / 2 = 5`` when the product is interpolated,
which is an artefact of fitting a quadratic with a straight line.

Times
-----
A caltable's ``TIME`` is a copy of the MS's, so it is matched against
``TabConfig.times_mjd`` -- the MS's own column, converted in unit only and still
on the scale the column declares -- and never against ``times_jd``, which the
reader has normalised to UTC (#158). Declared frame to declared frame is exact;
on a TAI-declared MS the UTC coordinate is 37 seconds away, four orders of
magnitude past the matching tolerance. The residual hazard is that a caltable's
own ``TIME`` metadata is taken on the CASA convention rather than read: the unit
is assumed to be seconds and the epoch reference assumed to be the MS's, because
a caltable is written beside its MS by tools that copy that column across. A
table whose ``TIME`` was written on another unit or another scale would be
matched wrongly, and the coverage line is what would say so -- it would report
no exact cover at all.
"""

import os
import warnings
from collections.abc import Sequence
from typing import Dict, List, NamedTuple, Optional, Union

import numpy as np
from numpy.typing import NDArray

from tabascal.ms import read_caltable


#: How close an observation time has to sit to a solved one to count as the same
#: sample, in seconds. Well below any integration, and well above the round-off
#: of an MJD second carried through a float64 day number.
TIME_ATOL = 1e-3

#: The same, for frequency, relative to the band being read. A channel centre
#: written by two different tools agrees to far better than this.
FREQ_RTOL = 1e-6

#: How each output sample got its value, worst last: the categories combine with
#: ``maximum`` as provenance is carried from one interpolation stage to the next,
#: so a sample interpolated between two edge-held ones is itself edge-held.
EXACT, INTERPOLATED, EDGE_HELD, DEAD = 0, 1, 2, 3

_TURN = 2 * np.pi


class Coverage(NamedTuple):
    """Fractions of the observation's samples by how their gain was obtained.

    ``exact`` is a solved sample sitting on the requested coordinate,
    ``interpolated`` one spanned by solutions either side, ``edge_held`` one
    beyond the last solution on the line it was placed by, and ``dead`` an
    antenna that was never solved at all. They sum to 1.
    """

    exact: float
    interpolated: float
    edge_held: float
    dead: float


class GridGains(NamedTuple):
    """One table's gains on the observation's grid.

    ``gains`` is ``(n_ant, n_freq, n_time)``; ``dead`` marks the entries no
    solution could be reached for, which carry a unity gain so that nothing
    downstream divides by a NaN. ``category`` is the same shape and holds
    :data:`EXACT` / :data:`INTERPOLATED` / :data:`EDGE_HELD` / :data:`DEAD` per
    sample -- the per-sample form of ``coverage``, which is only its histogram.
    """

    gains: NDArray
    dead: NDArray
    coverage: Coverage
    category: NDArray


def normalise_gain_tables(gain_table) -> List[str]:
    """``data.gain_table`` as an ordered list of paths that exist.

    A single path and a one-element list mean the same thing, and ``null`` and
    an empty list both mean no calibration. A path that is not there is caught
    here rather than after the MS has been read -- the tables are consumed at
    read time, so a typo would otherwise cost the whole read.
    """

    if gain_table is None:
        return []

    if isinstance(gain_table, (str, os.PathLike)):
        paths = [gain_table]
    elif isinstance(gain_table, Sequence):
        paths = list(gain_table)
    else:
        raise ValueError(
            f"data.gain_table = {gain_table!r} is neither a path to a calibration "
            "table nor an ordered list of them."
        )

    resolved = []
    for path in paths:
        if not isinstance(path, (str, os.PathLike)):
            raise ValueError(
                f"data.gain_table entry {path!r} is not a path to a calibration "
                "table."
            )
        path = os.path.abspath(os.fspath(path))
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"data.gain_table entry {path} does not exist. A calibration table "
                "is a directory, and it is read before the visibilities are, so "
                "the path is checked here."
            )
        resolved.append(path)

    return resolved


# ---------------------------------------------------------------------------
# One branch for the whole phase surface
# ---------------------------------------------------------------------------

def _groups(valid: NDArray):
    """Line indices grouped by which samples they carry.

    In practice a table shares one flagging pattern across most of its lines (or
    a handful of them), so grouping turns a per-line loop into a few vectorised
    operations over the lines that need the same thing done to them.
    """

    patterns, inverse = np.unique(valid, axis=0, return_inverse=True)
    inverse = np.reshape(inverse, -1)

    for p, pattern in enumerate(patterns):
        yield np.flatnonzero(inverse == p), pattern


def _support(pattern: NDArray, have: NDArray) -> NDArray:
    """The indices a line carries, in ascending coordinate order."""

    support = np.flatnonzero(pattern)
    support = support[np.argsort(have[support])]

    if len(support) > 1 and np.any(np.diff(have[support]) <= 0):
        raise ValueError(
            "The calibration table holds two solutions on the same coordinate, "
            "so there is no interval to interpolate over. Its times and channel "
            "frequencies must each be distinct."
        )

    return support


def _unwrap_along_last(phase: NDArray, valid: NDArray, have: NDArray) -> NDArray:
    """``np.unwrap`` along the last axis, over the samples each line carries."""

    lines = phase.reshape(-1, phase.shape[-1]).copy()
    mask = valid.reshape(-1, valid.shape[-1])

    for rows, pattern in _groups(mask):
        if pattern.sum() < 2:
            continue

        support = _support(pattern, have)
        lines[np.ix_(rows, support)] = np.unwrap(
            lines[np.ix_(rows, support)], axis=-1
        )

    return lines.reshape(phase.shape)


def _align_timesteps(phase: NDArray, valid: NDArray, times: NDArray) -> None:
    """Shift each timestep by whole turns so the band stays on one branch.

    Unwrapping along frequency leaves each timestep on the branch of its own
    first solved channel, so two timesteps can sit a whole turn apart even where
    the gain barely moved. They are brought together by the median step over the
    channels the two have in common: the band's own shape is what says where the
    branch is, and a per-channel unwrap cannot see it.

    This is also the unwrap along time -- the median step is brought into
    ``(-pi, pi]``. A step of exactly half a turn is the one genuinely ambiguous
    case and is left as written (``np.round``'s half-to-even), rather than being
    turned into a half turn of the opposite sign -- it is the boundary of the
    shortest-branch assumption the module docstring sets out, which every step
    beyond half a turn falls the other side of, silently.

    Mutates ``phase`` in place; timesteps carrying nothing are stepped over, so
    the alignment spans them the way the interpolation does.
    """

    n_ant, n_freq, _ = phase.shape
    offset = np.zeros(n_ant)
    previous = np.full(n_ant, -1)

    for t in np.argsort(times):
        here = valid[:, :, t]
        has = here.any(axis=1)

        if (previous >= 0).any():
            at = np.broadcast_to(
                np.clip(previous, 0, None)[:, None, None], (n_ant, n_freq, 1)
            )
            both = here & np.take_along_axis(valid, at, axis=2)[:, :, 0]
            both &= (previous >= 0)[:, None]

            step = np.where(
                both,
                phase[:, :, t] + offset[:, None]
                - np.take_along_axis(phase, at, axis=2)[:, :, 0],
                np.nan,
            )
            with warnings.catch_warnings():
                # A line with nothing in common with the previous timestep has
                # no step to read; it keeps the offset it already carried.
                warnings.simplefilter("ignore", RuntimeWarning)
                median = np.nanmedian(step, axis=1)

            turns = np.where(np.isnan(median), 0.0, np.round(median / _TURN))
            offset = np.where(has, offset - _TURN * turns, offset)

        phase[:, :, t] += offset[:, None]
        previous = np.where(has, t, previous)


def _unwrap_surface(gains: NDArray, valid: NDArray, times: NDArray, freqs: NDArray):
    """Amplitude and a branch-consistent unwrapped phase on the table's grid.

    Both are NaN where the table carries no solution, which is what makes them
    absences the interpolation bridges rather than values it honours.
    """

    amplitude = np.where(valid, np.abs(gains), np.nan)
    phase = np.where(valid, np.angle(gains), np.nan)

    if freqs is not None:
        phase = np.moveaxis(
            _unwrap_along_last(
                np.moveaxis(phase, 1, -1), np.moveaxis(valid, 1, -1), freqs
            ),
            -1,
            1,
        )

    _align_timesteps(phase, valid, times)

    return amplitude, phase


# ---------------------------------------------------------------------------
# Placing a surface on one axis of the observation's grid
# ---------------------------------------------------------------------------

class _Placed(NamedTuple):
    """The result of interpolating along one axis, and where it came from.

    ``lo`` and ``hi`` index the *input* axis, so a second stage can gather the
    first stage's categories at the samples it actually interpolated between;
    ``only_lo`` / ``only_hi`` mark the outputs that took one endpoint alone,
    which is every exact match and every edge-hold.
    """

    values: NDArray
    category: NDArray
    lo: NDArray
    hi: NDArray
    only_lo: NDArray
    only_hi: NDArray


def _place_axis(
    surfaces: NDArray, valid: NDArray, have: NDArray, want: NDArray, atol: float
) -> _Placed:
    """Interpolate ``(n_surface, n_line, n_have)`` onto ``want``, holding edges.

    The surfaces are amplitude and unwrapped phase, interpolated with the same
    weights because they are two halves of one gain and share one validity mask.
    """

    n_surface, n_line, _ = surfaces.shape
    n_want = len(want)

    values = np.full((n_surface, n_line, n_want), np.nan)
    category = np.full((n_line, n_want), DEAD, dtype=np.int8)
    lo_idx = np.full((n_line, n_want), -1)
    hi_idx = np.full((n_line, n_want), -1)
    only_lo = np.zeros((n_line, n_want), dtype=bool)
    only_hi = np.zeros((n_line, n_want), dtype=bool)

    for rows, pattern in _groups(valid):
        if not pattern.any():
            continue

        support = _support(pattern, have)
        xs = have[support]

        if len(xs) == 1:
            lo = hi = np.zeros(n_want, dtype=int)
            weight = np.zeros(n_want)
        else:
            hi = np.clip(np.searchsorted(xs, want), 1, len(xs) - 1)
            lo = hi - 1
            # Clipped, so a coordinate outside the solved range takes the edge
            # value rather than a straight line continued off the end of it.
            weight = np.clip((want - xs[lo]) / (xs[hi] - xs[lo]), 0.0, 1.0)

        block = surfaces[:, rows][:, :, support]
        values[:, rows] = block[..., lo] + weight * (block[..., hi] - block[..., lo])

        at_lo = np.abs(want - xs[lo]) <= atol
        at_hi = np.abs(want - xs[hi]) <= atol
        outside = (want < xs[0] - atol) | (want > xs[-1] + atol)

        category[rows] = np.where(
            outside, EDGE_HELD, np.where(at_lo | at_hi, EXACT, INTERPOLATED)
        )
        lo_idx[rows] = support[lo]
        hi_idx[rows] = support[hi]
        only_lo[rows] = at_lo | (weight <= 0.0)
        only_hi[rows] = at_hi | (weight >= 1.0)

    return _Placed(values, category, lo_idx, hi_idx, only_lo, only_hi)


def _carry(previous: NDArray, placed: _Placed) -> NDArray:
    """This stage's categories, worsened by those of the values it read.

    A sample interpolated between two edge-held ones was itself edge-held, and
    saying otherwise is how a coverage report comes to describe a rectangle the
    solutions never filled.
    """

    at_lo = np.take_along_axis(previous, np.clip(placed.lo, 0, None), axis=-1)
    at_hi = np.take_along_axis(previous, np.clip(placed.hi, 0, None), axis=-1)

    inherited = np.where(
        placed.only_lo,
        at_lo,
        np.where(placed.only_hi, at_hi, np.maximum(at_lo, at_hi)),
    )

    return np.maximum(placed.category, inherited)


def _coverage(category: NDArray) -> Coverage:
    total = category.size

    return Coverage(
        exact=float((category == EXACT).sum() / total),
        interpolated=float((category == INTERPOLATED).sum() / total),
        edge_held=float((category == EDGE_HELD).sum() / total),
        dead=float((category == DEAD).sum() / total),
    )


def interpolate_gains(
    cal: Dict,
    times,
    freqs,
    time_atol: float = TIME_ATOL,
    freq_rtol: float = FREQ_RTOL,
) -> GridGains:
    """One caltable's gains on an observation's ``(freq, time)`` grid.

    ``cal`` is what :func:`tabascal.ms.read_caltable` returns. ``times`` are in
    the table's own time convention -- MS ``TIME``, i.e. MJD seconds on the scale
    the MS declares -- and ``freqs`` in Hz.

    A table carrying no ``SPECTRAL_WINDOW`` has no frequencies to match against;
    a single solution is then broadcast across the band, but channels that
    cannot be placed are an error rather than a guess.

    See the module docstring for the interpolation semantics -- in particular
    that the phase is taken on its shortest branch, so the table's channels and
    solution intervals must each sample the phase by less than half a turn. A
    faster evolution than that is not recoverable from the solutions and is
    aliased, without anything here being able to tell.
    """

    times = np.asarray(times, dtype=float)
    freqs = np.asarray(freqs, dtype=float)

    gains = np.asarray(cal["gains"], dtype=complex)
    cal_times = np.asarray(cal["times"], dtype=float)
    cal_freqs = None if cal["freqs"] is None else np.asarray(cal["freqs"], dtype=float)

    n_ant, n_cal_freq, n_cal_time = gains.shape
    n_freq, n_time = len(freqs), len(times)

    if len(cal_times) != n_cal_time:
        raise ValueError(
            f"The calibration table holds {n_cal_time} solution times but "
            f"{len(cal_times)} time stamps."
        )

    if cal_freqs is not None and len(cal_freqs) != n_cal_freq:
        raise ValueError(
            f"The calibration table holds {n_cal_freq} solution channels but "
            f"{len(cal_freqs)} channel frequencies."
        )

    if cal_freqs is None and n_cal_freq != 1:
        raise ValueError(
            f"The calibration table has {n_cal_freq} channels but no "
            "SPECTRAL_WINDOW subtable, so there is nothing to say which "
            "frequencies they were solved on. A single-channel table is "
            "broadcast across the band; a multi-channel one cannot be placed."
        )

    # Every kind of missing solution is made the one absence the interpolation
    # bridges: a zero gain calibrates to infinity, so it is no more a solution
    # than a flagged entry is.
    valid = np.isfinite(gains) & (gains != 0)

    surfaces = np.stack(_unwrap_surface(gains, valid, cal_times, cal_freqs))

    if cal_freqs is None:
        # The table does not resolve frequency, so its solution is the answer for
        # every channel rather than an extrapolation onto them.
        on_freq = np.repeat(surfaces, n_freq, axis=2)
        cat_freq = np.where(
            np.repeat(valid, n_freq, axis=1), np.int8(EXACT), np.int8(DEAD)
        )
    else:
        # Frequency first: a B table's phase winds across the band, so it is the
        # axis whose wraps most need resolving before anything is averaged.
        placed = _place_axis(
            np.moveaxis(surfaces, 2, -1).reshape(2, -1, n_cal_freq),
            np.moveaxis(valid, 1, -1).reshape(-1, n_cal_freq),
            cal_freqs,
            freqs,
            freq_rtol * float(np.median(freqs)),
        )
        on_freq = np.moveaxis(
            placed.values.reshape(2, n_ant, n_cal_time, n_freq), -1, 2
        )
        cat_freq = np.moveaxis(
            placed.category.reshape(n_ant, n_cal_time, n_freq), -1, 1
        )

    placed = _place_axis(
        on_freq.reshape(2, -1, n_cal_time),
        (cat_freq != DEAD).reshape(-1, n_cal_time),
        cal_times,
        times,
        time_atol,
    )

    category = _carry(cat_freq.reshape(-1, n_cal_time), placed).reshape(
        n_ant, n_freq, n_time
    )
    amplitude, phase = placed.values.reshape(2, n_ant, n_freq, n_time)

    # The one place the phase becomes an angle again, once both axes are done.
    on_grid = amplitude * np.exp(1j * phase)

    dead = (category == DEAD) | ~np.isfinite(on_grid)

    return GridGains(
        np.where(dead, 1.0, on_grid), dead, _coverage(category), category
    )


def compose_gains(
    cals: Sequence[Dict],
    times,
    freqs,
    n_ant: Optional[int] = None,
    names: Optional[Sequence[str]] = None,
    verbose: bool = True,
    time_atol: float = TIME_ATOL,
    freq_rtol: float = FREQ_RTOL,
):
    """The ordered product of several tables, each placed on the grid first.

    Returns ``(gains, dead)``: the composed per-antenna gains
    ``(n_ant, n_freq, n_time)``, and the entries no table could supply a gain
    for -- unity in ``gains``, so the caller flags them rather than dividing by
    a NaN.

    Interpolating before composing is the whole point; see the module docstring.

    Composing nothing is an error rather than a unity gain: "no calibration"
    is a decision for the caller to take before it gets here, and returning
    ones would let a mis-read config calibrate with an identity nobody asked
    for.
    """

    times = np.asarray(times, dtype=float)
    freqs = np.asarray(freqs, dtype=float)

    if names is None:
        names = [f"{i + 1} of {len(cals)}" for i in range(len(cals))]

    total = None
    dead = None

    for cal, name in zip(cals, names):
        n_solved = np.shape(cal["gains"])[0]

        if n_ant is not None:
            if n_solved < n_ant:
                raise ValueError(
                    f"Calibration table {name} solves {n_solved} antennas but the "
                    f"observation has {n_ant}."
                )
            # Antenna ids index the ANTENNA subtable, so a table solved on a
            # master covers this observation in its leading rows. Trimmed before
            # the placement rather than after, so the coverage reported is the
            # coverage of *this* observation and not of the master.
            cal = {**cal, "gains": np.asarray(cal["gains"])[:n_ant]}

        on_grid = interpolate_gains(cal, times, freqs, time_atol, freq_rtol)

        total = on_grid.gains if total is None else total * on_grid.gains
        dead = on_grid.dead if dead is None else (dead | on_grid.dead)

        if verbose:
            coverage = on_grid.coverage
            print(
                f"\nGain table {name}"
                f"\n  {100 * coverage.exact:.1f} % exact, "
                f"{100 * coverage.interpolated:.1f} % interpolated, "
                f"{100 * coverage.edge_held:.1f} % edge-held, "
                f"{100 * coverage.dead:.1f} % unsolved"
            )

    if total is None:
        raise ValueError("No calibration tables were given to compose.")

    # A composed gain can still be unusable where no single table's was -- an
    # amplitude interpolated down to zero, or a product that overflowed -- and
    # dividing by it would put an infinity in the data rather than a flag.
    dead = dead | ~np.isfinite(total) | (total == 0)

    return np.where(dead, 1.0, total), dead


def gains_from_tables(
    gain_table: Union[None, str, Sequence[str]],
    times,
    freqs,
    n_ant: Optional[int] = None,
    verbose: bool = True,
):
    """:func:`compose_gains` over the tables ``data.gain_table`` names.

    ``times`` are MS ``TIME`` values in seconds, on the scale the MS declares,
    which is what a caltable's own ``TIME`` column holds.
    """

    paths = normalise_gain_tables(gain_table)

    return compose_gains(
        [read_caltable(path) for path in paths],
        times,
        freqs,
        n_ant=n_ant,
        names=paths,
        verbose=verbose,
    )
