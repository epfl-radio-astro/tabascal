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
*unwrapped* first, along each axis it is interpolated along: a B table winds
across the band through a residual delay and around the ``+-pi`` branch cut in
time, and interpolating the stored angles across such a step averages the two
sides of the cut into a gain pointing the wrong way.

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

Several tables compose on the grid, not before it
--------------------------------------------------
``data.gain_table`` takes an ordered list, and each table is placed on the
observation's grid *before* the product is formed. The two orders do not agree:
two amplitudes ramping 1 -> 3 give ``2 * 2 = 4`` half way when each is
interpolated first, and ``(1 + 9) / 2 = 5`` when the product is interpolated,
which is an artefact of fitting a quadratic with a straight line.
"""

import os
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


class Coverage(NamedTuple):
    """Fractions of the observation's samples by how their gain was obtained.

    ``exact`` is a solved sample sitting on the requested coordinate,
    ``interpolated`` one spanned by solutions either side, ``edge_held`` one
    beyond the last solution on either axis, and ``dead`` an antenna that was
    never solved at all. They sum to 1.
    """

    exact: float
    interpolated: float
    edge_held: float
    dead: float


class GridGains(NamedTuple):
    """One table's gains on the observation's grid.

    ``gains`` is ``(n_ant, n_freq, n_time)``; ``dead`` marks the entries no
    solution could be reached for, which carry a unity gain so that nothing
    downstream divides by a NaN.
    """

    gains: NDArray
    dead: NDArray
    coverage: Coverage


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


def _match(want: NDArray, have: NDArray, atol: float) -> NDArray:
    """Index in ``have`` of each ``want`` within ``atol``, or ``-1``.

    By value rather than by index, which is what lets a table solved on a master
    apply to a subset carved out of it, in whatever channel order the subset was
    written.
    """

    order = np.argsort(have)
    ordered = have[order]

    right = np.clip(np.searchsorted(ordered, want), 0, len(have) - 1)
    left = np.clip(right - 1, 0, len(have) - 1)
    nearer = np.where(
        np.abs(ordered[right] - want) <= np.abs(ordered[left] - want), right, left
    )
    idx = order[nearer]

    return np.where(np.abs(have[idx] - want) <= atol, idx, -1)


def _linear(xs: NDArray, ys: NDArray, want: NDArray) -> NDArray:
    """Linear interpolation of ``ys`` along its last axis, holding the edges.

    ``np.interp``'s behaviour, vectorised over every leading axis: the support
    points are shared by all of them, which is what makes a whole group of lines
    one array operation.
    """

    if len(xs) == 1:
        return np.repeat(ys, len(want), axis=-1)

    hi = np.clip(np.searchsorted(xs, want), 1, len(xs) - 1)
    lo = hi - 1
    # Clipped, so a coordinate outside the solved range takes the edge value
    # rather than a straight line continued off the end of the solutions.
    weight = np.clip((want - xs[lo]) / (xs[hi] - xs[lo]), 0.0, 1.0)

    return ys[..., lo] + weight * (ys[..., hi] - ys[..., lo])


def _interp_last_axis(gains: NDArray, have: NDArray, want: NDArray) -> NDArray:
    """Interpolate complex gains onto ``want`` along their last axis.

    NaN entries are absences: each line is interpolated from the samples it does
    carry, so a flagged solution is bridged rather than honoured. Lines are
    grouped by which samples they carry, since in practice a whole table shares
    one flagging pattern (or a handful) and each group is then one vectorised
    interpolation.
    """

    lines = gains.reshape(-1, gains.shape[-1])
    valid = np.isfinite(lines) & (lines != 0)

    out = np.full((lines.shape[0], len(want)), np.nan, dtype=complex)

    patterns, inverse = np.unique(valid, axis=0, return_inverse=True)
    inverse = np.reshape(inverse, -1)

    for p, pattern in enumerate(patterns):
        if not pattern.any():
            continue

        rows = np.flatnonzero(inverse == p)
        support = np.flatnonzero(pattern)
        order = np.argsort(have[support])
        support = support[order]

        xs = have[support]
        if len(xs) > 1 and np.any(np.diff(xs) <= 0):
            raise ValueError(
                "The calibration table solves two solutions on the same "
                "coordinate, so there is no interval to interpolate over. Its "
                "times and channel frequencies must each be distinct."
            )

        block = lines[np.ix_(rows, support)]
        # Unwrapped along the axis being interpolated, and only over the samples
        # actually being interpolated between: the branch cut is an artefact of
        # storing an angle, not a feature of the gain.
        amplitude = _linear(xs, np.abs(block), want)
        phase = _linear(xs, np.unwrap(np.angle(block), axis=-1), want)

        out[rows] = amplitude * np.exp(1j * phase)

    return out.reshape(gains.shape[:-1] + (len(want),))


def _classify(
    valid: NDArray,
    cal_freqs: Optional[NDArray],
    cal_times: NDArray,
    freqs: NDArray,
    times: NDArray,
    freq_atol: float,
    time_atol: float,
) -> Coverage:
    """How each sample of the observation's grid got its gain.

    Read off the table's own coordinates and validity rather than out of the
    interpolation, so the classification says something about the calibration
    that was supplied -- which is what the log line is for.
    """

    n_ant = valid.shape[0]

    if cal_freqs is None:
        # The table does not resolve frequency, so its solution is the answer
        # for every channel rather than an extrapolation onto them.
        f_idx = np.zeros(len(freqs), dtype=int)
        in_band = np.ones((n_ant, len(freqs)), dtype=bool)
    else:
        f_idx = _match(freqs, cal_freqs, freq_atol)
        in_band = _within(valid.any(axis=2), cal_freqs, freqs, freq_atol)

    t_idx = _match(times, cal_times, time_atol)
    in_span = _within(valid.any(axis=1), cal_times, times, time_atol)

    solved = valid[:, np.clip(f_idx, 0, None)][:, :, np.clip(t_idx, 0, None)]
    exact = solved & (f_idx >= 0)[None, :, None] & (t_idx >= 0)[None, None, :]

    dead = np.broadcast_to(
        ~valid.any(axis=(1, 2))[:, None, None], (n_ant, len(freqs), len(times))
    )
    edge = ~(in_band[:, :, None] & in_span[:, None, :]) & ~dead
    exact = exact & ~dead & ~edge

    total = dead.size

    return Coverage(
        exact=float(exact.sum() / total),
        interpolated=float((~exact & ~edge & ~dead).sum() / total),
        edge_held=float(edge.sum() / total),
        dead=float(dead.sum() / total),
    )


def _within(valid: NDArray, have: NDArray, want: NDArray, atol: float) -> NDArray:
    """Per antenna, whether each ``want`` lies inside its solved range."""

    # An antenna with nothing solved gets an empty range (+inf to -inf), which
    # holds no coordinate at all; it is reported dead rather than edge-held.
    low = np.min(np.where(valid, have[None], np.inf), axis=1)
    high = np.max(np.where(valid, have[None], -np.inf), axis=1)

    return (want[None] >= low[:, None] - atol) & (want[None] <= high[:, None] + atol)


def interpolate_gains(
    cal: Dict,
    times,
    freqs,
    time_atol: float = TIME_ATOL,
    freq_rtol: float = FREQ_RTOL,
) -> GridGains:
    """One caltable's gains on an observation's ``(freq, time)`` grid.

    ``cal`` is what :func:`tabascal.ms.read_caltable` returns. ``times`` are in
    the table's own time convention -- MS ``TIME``, i.e. MJD seconds -- and
    ``freqs`` in Hz.

    A table carrying no ``SPECTRAL_WINDOW`` has no frequencies to match against;
    a single solution is then broadcast across the band, but channels that
    cannot be placed are an error rather than a guess.

    See the module docstring for the interpolation semantics.
    """

    times = np.asarray(times, dtype=float)
    freqs = np.asarray(freqs, dtype=float)

    gains = np.asarray(cal["gains"], dtype=complex)
    cal_times = np.asarray(cal["times"], dtype=float)
    cal_freqs = None if cal["freqs"] is None else np.asarray(cal["freqs"], dtype=float)

    n_ant, n_cal_freq, n_cal_time = gains.shape

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

    # Every kind of missing solution is made the one absence the interpolation
    # bridges: a zero gain calibrates to infinity, so it is no more a solution
    # than a flagged entry is.
    valid = np.isfinite(gains) & (gains != 0)
    gains = np.where(valid, gains, np.nan)

    if cal_freqs is None:
        if n_cal_freq != 1:
            raise ValueError(
                f"The calibration table has {n_cal_freq} channels but no "
                "SPECTRAL_WINDOW subtable, so there is nothing to say which "
                "frequencies they were solved on. A single-channel table is "
                "broadcast across the band; a multi-channel one cannot be placed."
            )
        on_freq = np.repeat(gains, len(freqs), axis=1)
    else:
        # Frequency first: a B table's phase winds across the band, so it is the
        # axis whose wraps most need resolving before anything is averaged.
        on_freq = np.moveaxis(
            _interp_last_axis(np.moveaxis(gains, 1, -1), cal_freqs, freqs), -1, 1
        )

    on_grid = _interp_last_axis(on_freq, cal_times, times)

    freq_atol = freq_rtol * float(np.median(freqs))
    coverage = _classify(
        valid, cal_freqs, cal_times, freqs, times, freq_atol, time_atol
    )

    # Only an antenna with no solution anywhere survives both stages as a NaN:
    # one solved sample is an edge, and an edge is held.
    dead = ~np.isfinite(on_grid)

    return GridGains(np.where(dead, 1.0, on_grid), dead, coverage)


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

    ``times`` are MS ``TIME`` values in seconds (MJD seconds), which is what a
    caltable's own ``TIME`` column holds.
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
