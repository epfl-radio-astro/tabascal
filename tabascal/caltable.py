"""
caltable.py — CASA-compatible calibration tables.

The one place in tabascal that knows the CASA caltable format, so that gains can be
exchanged with standard tooling (``applycal``, CASA/CARAcal/stimela) instead of ad-hoc
``.npz`` files.

Convention (CASA's, and the only one used here)
-----------------------------------------------
The gain multiplies the *model* to give the *observed* visibility::

    V_obs[p, q] = g_p * conj(g_q) * V_true[p, q]

so calibrating divides it out, and the noise follows the data::

    V_cal   = V_obs / (g_p conj(g_q))
    sigma_cal = sigma / |g_p conj(g_q)|
    weight_cal = weight * |g_p conj(g_q)|**2      (weight == 1 / sigma**2)

A scalar flux scale ``V_cal = k * V_obs`` is therefore the antenna-independent gain
``g = k ** -0.5``.

The table layout mirrors exactly what ``casatasks.gaincal`` emits (verified against a
reference table it produced): one row per (time, antenna), with ``CPARAM`` of shape
``(n_chan, n_pol)``.  ``B Jones`` is used rather than ``G Jones`` because the gains here
are frequency dependent, which a scalar G table cannot represent.

Verified against CASA (see tests): ``applycal`` accepts a table from
:func:`write_caltable`, and its ``CORRECTED_DATA`` reproduces ``V / (g_p conj(g_q))``
to 6e-7 relative (float32 round-off) over 6.7e6 visibilities.

**Do not rely on applycal to set the weights for a frequency-dependent gain.**
``applycal(calwt=True)`` was measured to apply a single per-row weight factor, constant
across channels (within-row CV of the applied factor = 0.0000), even when
``WEIGHT_SPECTRUM`` exists -- it collapses the frequency axis rather than scaling each
channel by its own ``|g_ch|**2``. For a channel-constant gain that is exact; for a
frequency-dependent one it is an approximation. tabascal therefore computes
``WEIGHT_SPECTRUM = 1 / sigma_cal**2`` per channel itself when it writes results.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
from numpy.typing import NDArray

from casacore.tables import (
    makearrcoldesc,
    makescacoldesc,
    maketabdesc,
    table,
    tablecopy,
)

# Subtables a caltable carries; CASA copies these straight from the MS.
_SUBTABLES = ("ANTENNA", "FIELD", "SPECTRAL_WINDOW", "OBSERVATION", "HISTORY")


def _caltable_desc():
    """Table description matching casatasks.gaincal's output."""
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
    """Write an applycal-compatible calibration table.

    Parameters
    ----------
    path   : output caltable path.
    gains  : (n_ant, n_freq, n_time) complex — g_p, in the V_obs = g_p conj(g_q) V_true
             convention above.
    times  : (n_time,) MS TIME values in seconds (MJD seconds, as in the MS).
    ms_path: the MS these gains belong to; its ANTENNA/FIELD/SPECTRAL_WINDOW/OBSERVATION/
             HISTORY subtables are copied into the caltable, as CASA does.
    n_pol  : CASA writes 2 polarisations even for a single-correlation MS, so the gain is
             duplicated across the pol axis by default.

    Returns the caltable path.
    """
    gains = np.asarray(gains)
    if gains.ndim != 3:
        raise ValueError(f"gains must be (n_ant, n_freq, n_time), got {gains.shape}")
    n_ant, n_freq, n_time = gains.shape
    times = np.asarray(times, dtype=float)
    if len(times) != n_time:
        raise ValueError(f"times has {len(times)} entries but gains has {n_time} times")

    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(f"{path} already exists. Use overwrite=True.")
        shutil.rmtree(path)

    n_row = n_time * n_ant
    tb = table(path, _caltable_desc(), nrow=n_row, ack=False)

    # Row order is time-major: (t0,a0), (t0,a1), ... — matching gaincal.
    ant_idx = np.tile(np.arange(n_ant, dtype=np.int32), n_time)
    time_col = np.repeat(times, n_ant)

    # (n_ant, n_freq, n_time) -> (n_row, n_freq), then duplicated across pol.
    g_rows = np.transpose(gains, (2, 0, 1)).reshape(n_row, n_freq)
    cparam = np.repeat(g_rows[:, :, None], n_pol, axis=2).astype(np.complex64)
    # A gain that is zero or non-finite carries no solution.
    bad = ~np.isfinite(g_rows) | (g_rows == 0)
    flag = np.repeat(bad[:, :, None], n_pol, axis=2)

    tb.putcol("TIME", time_col)
    tb.putcol("ANTENNA1", ant_idx)
    tb.putcol("ANTENNA2", np.full(n_row, -1, dtype=np.int32))
    tb.putcol("FIELD_ID", np.zeros(n_row, dtype=np.int32))
    tb.putcol("SPECTRAL_WINDOW_ID", np.zeros(n_row, dtype=np.int32))
    tb.putcol("SCAN_NUMBER", np.zeros(n_row, dtype=np.int32))
    tb.putcol("OBSERVATION_ID", np.zeros(n_row, dtype=np.int32))
    tb.putcol("INTERVAL", np.full(n_row, float(interval)))
    tb.putcol("CPARAM", cparam)
    tb.putcol("FLAG", flag)
    tb.putcol("PARAMERR", np.zeros((n_row, n_freq, n_pol), dtype=np.float32))
    tb.putcol("SNR", np.ones((n_row, n_freq, n_pol), dtype=np.float32))

    # CASA identifies a caltable by its table INFO record, not by its keywords --
    # without this, applycal rejects the table with 'is not a valid Calibration table'.
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
        tablecopy(src, dst)
        tb.putkeyword(sub, "Table: " + os.path.abspath(dst))

    tb.flush()
    tb.close()
    return path


def read_caltable(path: str) -> dict:
    """Read a caltable written by :func:`write_caltable` (or by CASA).

    Returns a dict with ``gains`` (n_ant, n_freq, n_time) complex — flagged solutions set
    to NaN — plus ``times``, ``freqs`` (Hz, from the table's SPECTRAL_WINDOW; None if it
    has none), ``ant_idx`` and ``viscal``.

    ``times`` and ``freqs`` are what let a table solved on a master MS be matched onto a
    subset carved out of it — see :func:`match_gains_to_grid`.
    """
    tb = table(path, ack=False)
    time_col = tb.getcol("TIME")
    ant1 = tb.getcol("ANTENNA1")
    cparam = tb.getcol("CPARAM")          # (n_row, n_freq, n_pol)
    flag = tb.getcol("FLAG")
    viscal = tb.getkeyword("VisCal") if "VisCal" in tb.keywordnames() else ""
    tb.close()

    freqs = None
    spw_path = os.path.join(path, "SPECTRAL_WINDOW")
    if os.path.exists(spw_path):
        spw = table(spw_path, ack=False)
        freqs = np.asarray(spw.getcol("CHAN_FREQ")[0], dtype=float)
        spw.close()

    # Average the pol axis: tabascal is single-correlation, and write_caltable
    # duplicates the gain across pols, so this is a no-op for our own tables.
    g = np.where(flag, np.nan, cparam).mean(axis=-1)   # (n_row, n_freq)

    times = np.unique(time_col)
    n_ant = int(ant1.max()) + 1
    n_time, n_freq = len(times), g.shape[1]

    gains = np.full((n_ant, n_freq, n_time), np.nan, dtype=complex)
    t_idx = np.searchsorted(times, time_col)
    gains[ant1, :, t_idx] = g

    return {
        "gains": gains,
        "times": times,
        "freqs": freqs,
        "ant_idx": np.arange(n_ant),
        "viscal": viscal,
    }


def match_gains_to_grid(
    cal: dict,
    times: NDArray,
    freqs: NDArray | None = None,
    time_atol: float = 1e-3,
    freq_rtol: float = 1e-6,
) -> NDArray:
    """Select the caltable solutions that belong to a given (freq, time) grid.

    The gains are typically solved on a **master** MS and then used on a **subset**
    carved out of it, so the subset's times/frequencies are a subset of the table's.
    Matching is by value (not by index), and it is an error for a requested time or
    frequency to be missing — silently interpolating a calibration is how you end up
    with a plausible-looking wrong answer.

    Returns gains of shape ``(n_ant, len(freqs), len(times))``.
    """
    g = cal["gains"]

    t_idx = _match_axis(np.asarray(times, float), cal["times"], time_atol, "time")
    g = g[:, :, t_idx]

    if freqs is not None:
        if cal["freqs"] is None:
            raise ValueError(
                "The caltable has no SPECTRAL_WINDOW, so its channels cannot be matched "
                "to the MS frequencies."
            )
        f_want = np.asarray(freqs, float)
        f_idx = _match_axis(f_want, cal["freqs"], freq_rtol * np.median(f_want), "frequency")
        g = g[:, f_idx, :]

    return g


def _match_axis(want: NDArray, have: NDArray, atol: float, what: str) -> NDArray:
    """Index of each `want` value in `have`, erroring if any is absent."""
    order = np.argsort(have)
    pos = np.clip(np.searchsorted(have[order], want), 0, len(have) - 1)
    idx = order[pos]
    bad = np.abs(have[idx] - want) > atol
    if bad.any():
        raise ValueError(
            f"{bad.sum()} MS {what}(s) are not in the gain table "
            f"(first missing: {want[bad][0]!r}). The table must have been solved on "
            f"this MS, or on a master it was carved from."
        )
    return idx


def baseline_gains(gains: NDArray, a1: NDArray, a2: NDArray) -> NDArray:
    """Per-baseline gain product g_p conj(g_q), shape (n_bl, n_freq, n_time)."""
    gains = np.asarray(gains)
    return gains[np.asarray(a1)] * np.conj(gains[np.asarray(a2)])


def apply_gains_to_data(
    vis: NDArray,
    gains: NDArray,
    a1: NDArray,
    a2: NDArray,
    sigma: NDArray | float | None = None,
):
    """Divide the gains out of the data (and carry the noise with it).

    This is the whole convention in one place — see the module docstring.

    ``vis`` is (n_bl, n_freq, n_time); ``gains`` is (n_ant, n_freq, n_time); ``sigma`` is
    anything broadcastable against ``vis`` (a scalar, or (n_bl, 1, 1)).

    Returns ``(vis_cal, sigma_cal)``; ``sigma_cal`` is None if no ``sigma`` was given.
    A baseline with a flagged/zero gain would divide by zero, so it is returned as NaN --
    such visibilities must be flagged by the caller.
    """
    g_bl = baseline_gains(gains, a1, a2)
    with np.errstate(divide="ignore", invalid="ignore"):
        vis_cal = np.asarray(vis) / g_bl
        sigma_cal = None if sigma is None else np.asarray(sigma) / np.abs(g_bl)
    return vis_cal, sigma_cal
