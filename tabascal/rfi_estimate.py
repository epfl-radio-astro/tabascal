"""Matched-filter light-curve extraction for RFI sources.

Given a set of satellite trajectories and the observed visibilities, this module
beam-forms the interferometer toward each satellite and reads off its
per-timestep (and per-channel) flux. This is a *matched filter* in visibility
space: for a point source moving along a known trajectory the RFI contribution to
baseline ``(p, q)`` is::

    V_rfi[bl] = A_p conj(A_q) exp(i (phi_p - phi_q))

where ``phi_a`` is the geometric phase at each antenna (:func:`tabascal.interferometry.get_rfi_phase`).
The per-baseline template is the unit-modulus steering vector
``T_bl = exp(i (phi_p - phi_q))``, the data are ``V_bl = T_bl S + n_bl`` with
per-component noise ``sigma_bl``, and the maximum-likelihood
(inverse-variance-weighted) estimate of the source visibility at each
(freq, time) is the de-rotated, weighted baseline average::

    S_hat[f, t] = sum_bl w_bl conj(T_bl) V_bl / sum_bl w_bl |T_bl|^2,
    w_bl = 1 / sigma_bl^2

with variance ``1 / sum_bl w_bl |T_bl|^2``. Every template here is unit-modulus,
so the denominator is just ``D = sum_bl w_bl`` and::

    error = 1 / sqrt(D),      z = Re(S_hat) / error

``z`` is a *calibrated-frame* statistic: it reads the real part because a
de-rotated real source has no imaginary part to read, which holds only where the
antenna gain phases have been taken out. :func:`coverage_stats` reports
``|S_hat| / error`` beside it as ``amp_coverage``, against a matched Rayleigh
threshold. That magnitude is invariant to a phase *common* to every baseline --
an overall offset, or a stable phase on the source itself -- which would
otherwise turn the signal out of the real part and hide it from ``z``.

It is **not** a defence against an uncalibrated antenna gain. A gain multiplies
each baseline before the average, ``S_hat = S * sum_bl w g_p conj(g_q) / sum_bl
w``, so antenna-dependent phases decorrelate the coherent sum itself: the
estimate shrinks, and there is nothing left in its magnitude for either statistic
to find. Calibrate the phases, or accept that both numbers understate what is
there.

The satellite fringe adds coherently after de-rotation while the sky and the
noise add incoherently, so ``S_hat`` isolates the RFI source visibility. Its
magnitude is a per-antenna power estimate; ``sqrt(|S_hat|)`` is the per-antenna
amplitude used to seed tabascal's RFI signal model.

**The weights carry the calibration; the template does not.** ``w`` comes from
the noise the MS reports, resolved per baseline and per channel wherever the
column resolves it that far (``SIGMA_SPECTRUM`` / ``SIGMA``, see
:mod:`tabascal.noise`). On an uncalibrated column the source is really
``g_p conj(g_q) S``, so a unit template de-rotates the geometry but not the
gains: the per-baseline terms add with a scatter of gain phases and the coherent
sum is degraded. That loss *is* the cost of not calibrating, and it is what the
estimate should show. Putting the gain in the template and calibrating the data
while carrying the transformed noise (``WEIGHT_SPECTRUM = |g|^2 / SIGMA^2``, the
frame the ``TAB_*`` columns are written in) are the *same* estimator; what is not
allowed is calibrating and then weighting uniformly. There is no noise-vs-gain
power law anywhere here: the noise is whatever the MS says it is.

Three entry points are provided:

* :func:`light_curves_from_config` -- the in-process path. Reuses the arrays a
  :class:`~tabascal.config.TabConfig` has already loaded (visibilities, noise,
  antennas, times, orbit records) so tabascal can seed the RFI model without
  touching the MS again, and returns the curves already ordered to match
  ``satellites.norad_ids``.
* :func:`extract_light_curves_from_ms` -- the standalone tool. Point it at any MS
  column and a set of NORAD IDs and it returns / saves the light curves.
* :func:`extract_light_curves_from_zarr` -- the post-fit diagnostic. Matched
  filters the residual of a run taken straight from its results zarr.

All three share the pure core :func:`matched_filter_light_curves`.

Only the satellite-trajectory source is implemented. RA/Dec and Alt/Az pointings
can be added by constructing ``rfi_xyz`` from those and feeding
:func:`rfi_phase_from_positions`.
"""

import os
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from tabascal.components.trajectory import (
    fetch_orbital_elements,
    get_satellite_elevations,
    get_satellite_positions,
    itrs_to_gcrs_sf,
)
from tabascal.interferometry import get_rfi_phase_numpy, itrf_to_uvw_numpy
from tabascal.noise import broadcast_to_vis
from tabascal.time import gast_deg, mjd_to_jd


#: Bytes of working array per (baseline, channel, timestep) inside the
#: matched-filter loop, used to size the time chunk against ``max_mem_gb``.
#: Counts the per-baseline arrays that dominate it: the conjugated template, the
#: zeroed visibility chunk and the two temporaries of ``w * v * conj(T)`` (four
#: complex128), plus the weight chunk (one float64). The weights are counted even
#: though a per-baseline noise broadcasts from a much smaller array -- ``np.where``
#: against the flags materialises them at the chunk's full shape.
#:
#: It is a sizing heuristic for the loop, not a cap on the function: see
#: :func:`matched_filter_light_curves` for what it does not cover.
_BYTES_PER_SAMPLE = 4 * 16 + 8


# ---------------------------------------------------------------------------
# Geometry: per-antenna RFI phase from a trajectory
# ---------------------------------------------------------------------------

def rfi_phase_from_positions(
    rfi_xyz: NDArray,
    ants_itrf: NDArray,
    times_jd: NDArray,
    phase_centre: dict,
    freqs: NDArray,
) -> NDArray:
    """Per-antenna geometric phase for RFI sources at known ECI positions.

    Numpy/f64 host-side computation mirroring
    :meth:`tabascal.components.trajectory.FixedOrbit._compute_rfi_phase`, so the
    estimate is matched to the phase the forward model itself builds.

    Parameters
    ----------
    rfi_xyz : Array (n_src, n_time, 3)
        Source positions over time in the ECI (GCRF) frame, in metres.
    ants_itrf : Array (n_ant, 3)
        Antenna positions in the ITRF (ECEF) frame, in metres.
    times_jd : Array (n_time,)
        Observation times in Julian date.
    phase_centre : dict
        ``{"ra": <deg>, "dec": <deg>}`` phase centre of the visibilities.
    freqs : Array (n_freq,)
        Channel frequencies in Hz.

    Returns
    -------
    Array (n_src, n_ant, n_freq, n_time)
        Geometric phase at each antenna for each source.
    """
    times_jd = np.asarray(times_jd)
    freqs = np.asarray(freqs)

    gsa = gast_deg(times_jd)  # GAST in degrees (UTC convention)
    gh0 = (gsa - phase_centre["ra"]) % 360

    ants_uvw = np.transpose(
        itrf_to_uvw_numpy(ants_itrf, gh0, phase_centre["dec"]), axes=(1, 0, 2)
    )  # (n_ant, n_time, 3)
    ants_xyz = itrs_to_gcrs_sf(ants_itrf, times_jd)  # (n_ant, n_time, 3)

    return get_rfi_phase_numpy(np.asarray(rfi_xyz), ants_uvw, ants_xyz, freqs)


def rfi_phase_from_records(
    orbit_records: list,
    ants_itrf: NDArray,
    times_jd: NDArray,
    phase_centre: dict,
    freqs: NDArray,
) -> NDArray:
    """Per-antenna geometric phase for satellites given their orbit records.

    Propagates each record -- TLE or OMM, as resolved by :mod:`tabascal.orbit` --
    over ``times_jd`` and defers to :func:`rfi_phase_from_positions`.

    Parameters
    ----------
    orbit_records : sequence of dict, length n_src
        Orbit records, in the order the curves are wanted in.
    ants_itrf, times_jd, phase_centre, freqs
        See :func:`rfi_phase_from_positions`.

    Returns
    -------
    Array (n_src, n_ant, n_freq, n_time)
    """
    times_jd = np.asarray(times_jd)
    rfi_xyz = np.asarray(get_satellite_positions(orbit_records, list(times_jd)))

    return rfi_phase_from_positions(rfi_xyz, ants_itrf, times_jd, phase_centre, freqs)


# ---------------------------------------------------------------------------
# The matched filter
# ---------------------------------------------------------------------------

def _weight_source(noise, vis_shape: tuple) -> NDArray:
    """Inverse-variance weights ``1 / sigma^2``, broadcastable onto ``vis_shape``.

    The broadcast happens here rather than after any flag masking, for the reason
    :func:`tabascal.noise.broadcast_to_vis` gives: ``x[~flags]`` flattens, and the
    resolved values would no longer line up with the samples they belong to.
    """
    if noise is None:
        return np.ones((1, 1, 1), dtype=np.float64)

    sigma = np.asarray(
        broadcast_to_vis(np.asarray(noise, dtype=np.float64), vis_shape),
        dtype=np.float64,
    )
    # Always three-dimensional, so the loop below can slice the time axis
    # without re-deriving which axes a given noise resolves.
    sigma = sigma.reshape((1,) * (3 - sigma.ndim) + sigma.shape)

    return 1.0 / sigma**2


def _time_chunk(n_bl: int, n_freq: int, max_mem_gb: float) -> int:
    """Timesteps per block of the matched-filter loop, under ``max_mem_gb``."""
    per_t = max(n_bl * n_freq * _BYTES_PER_SAMPLE, 1)

    return max(1, int(max_mem_gb * 1e9 / per_t))


def matched_filter_light_curves(
    vis: NDArray,
    rfi_phase: NDArray,
    a1: NDArray,
    a2: NDArray,
    noise=None,
    flags: Optional[NDArray] = None,
    in_view: Optional[NDArray] = None,
    exclude_autos: bool = True,
    max_mem_gb: float = 1.0,
) -> Tuple[NDArray, NDArray]:
    """Beam-form the data toward each source to estimate its source visibility.

    The estimator and its noise floor are the ones this module's docstring
    derives. Evaluated in numpy/f64 (a one-shot host-side estimate) with an outer
    time-chunk loop sized so the ``(n_bl, n_freq, chunk)`` per-baseline arrays --
    the template, the masked visibilities, the weights and their products -- stay
    within ``max_mem_gb``.

    ``max_mem_gb`` bounds those arrays, not the function's peak. It does not
    count the per-antenna ``exp(-i phi)`` chunk or the copies fancy indexing
    makes of a partly-masked block, and it says nothing about the arrays held for
    the whole call: ``vis`` and ``rfi_phase`` as given, and the
    ``(n_src, n_freq, n_time)`` accumulator, which grows with the number of
    sources rather than with the chunk. Lowering it shrinks the loop's working
    set and nothing else.

    Parameters
    ----------
    vis : Array (n_bl, n_freq, n_time) complex
        Observed visibilities (any MS data column).
    rfi_phase : Array (n_src, n_ant, n_freq, n_time)
        Per-antenna geometric phase (see :func:`rfi_phase_from_positions`).
    a1, a2 : Array (n_bl,)
        Antenna indices of each baseline.
    noise : float or Array, optional
        Per-component noise standard deviation, as
        :class:`~tabascal.config.TabConfig` resolves it: a scalar, ``(n_bl,)``,
        ``(n_bl, n_freq)``, or a three-dimensional array whose axes match the
        visibilities or are 1. ``None`` weights every baseline equally, which
        under-weights the quiet baselines and over-weights the loud ones, and
        returns a ``nan`` error: without a sigma the weights are a shape rather
        than a variance, and ``1 / sqrt(N)`` would be asserting a noise of 1 Jy
        that nobody wrote down. The callers warn when they fall back to it.
    flags : Array (n_bl, n_freq, n_time) bool, optional
        ``True`` marks samples to exclude from the average.
    in_view : Array (n_src, n_time) bool, optional
        ``False`` marks (source, timestep) pairs the source is not up for. Those
        times are skipped entirely -- the template is never evaluated there --
        and come back as an exact zero, the same "no signal known" convention the
        forward model's elevation mask uses.
    exclude_autos : bool, default True
        Drop autocorrelation baselines (``a1 == a2``). They carry no fringe to
        de-rotate, only each antenna's own power.
    max_mem_gb : float, default 1.0
        Approximate cap on the working-array size of the time-chunk loop.

    Returns
    -------
    Array (n_src, n_freq, n_time) complex
        Matched-filter source-visibility estimate. ``nan`` where every baseline
        in a (freq, time) cell is flagged -- nothing was measured there, and a
        zero would read as a measured zero -- and exactly ``0`` where ``in_view``
        says the source was not up.
    Array (n_src, n_freq, n_time) real
        The standard error of the (real) flux estimate, ``1 / sqrt(sum_bl w)``:
        the beam-former's noise floor, the visibility-space equivalent of a
        dirty-image aperture standard deviation. ``nan`` wherever the estimate
        is not a measurement, and everywhere when ``noise`` is ``None``.
    """
    vis = np.asarray(vis)
    rfi_phase = np.asarray(rfi_phase)
    a1 = np.asarray(a1)
    a2 = np.asarray(a2)

    n_bl, n_freq, n_time = vis.shape
    n_src = rfi_phase.shape[0]

    weights = _weight_source(noise, vis.shape)
    keep_bl = np.ones((n_bl, 1, 1), dtype=bool)
    if exclude_autos:
        keep_bl = (a1 != a2)[:, None, None]

    if in_view is None:
        in_view = np.ones((n_src, n_time), dtype=bool)
    else:
        in_view = np.asarray(in_view, dtype=bool)

    num = np.zeros((n_src, n_freq, n_time), dtype=np.complex128)
    den = np.zeros((n_freq, n_time), dtype=np.float64)

    chunk = _time_chunk(n_bl, n_freq, max_mem_gb)

    for t0 in range(0, n_time, chunk):
        t1 = min(t0 + chunk, n_time)
        block = slice(t0, t1)

        # The weights only carry a time axis where the MS's noise column varies
        # over the observation; otherwise the same values serve every block.
        w = np.where(
            keep_bl, weights[:, :, block] if weights.shape[2] > 1 else weights, 0.0
        )
        if flags is not None:
            w = np.where(flags[:, :, block], 0.0, w)
        w = np.broadcast_to(w, (n_bl, n_freq, t1 - t0))

        den[:, block] = w.sum(axis=0)
        # Zeroed rather than left alone: a flagged sample may be inf or nan, and
        # 0 * nan is nan, which would poison the whole cell's sum.
        v = np.where(w > 0.0, vis[:, :, block], 0.0)

        for s in range(n_src):
            up = np.flatnonzero(in_view[s, block])
            if up.size == 0:
                continue
            # A slice while the whole block is in view -- the common case, and no
            # copy. The index array is only built where the source sets mid-block.
            read = slice(None) if up.size == t1 - t0 else up

            # conj(T_bl) = exp(-i(phi_p - phi_q)), evaluated only where the
            # source is up: `rfi_phase` at a masked time is never read.
            conj_e = np.exp(-1.0j * rfi_phase[s, :, :, block][:, :, read])
            template_conj = conj_e[a1] * np.conjugate(conj_e[a2])

            # Transposed because an advanced index alongside a slice puts its own
            # axis first, so the left-hand side is (time, freq).
            num[s, :, t0 + up] = np.sum(
                w[:, :, read] * v[:, :, read] * template_conj, axis=0
            ).T

    # A cell nothing was measured in is not a zero: it divides to nan, which the
    # light-curve reader maps back to "no signal known". Out of view is a zero,
    # the masked-signal convention, with a nan error so those cells are excluded
    # from the coverage statistic rather than counted as consistent with noise.
    safe_den = np.where(den > 0.0, den, np.nan)[None]
    visible = in_view[:, None, :]

    with np.errstate(invalid="ignore", divide="ignore"):
        light_curves = np.where(visible, num / safe_den, 0.0)
        # With no sigma the denominator is a count, not an inverse variance, and
        # 1/sqrt of it is a number in no units. Reporting nothing is the honest
        # answer; the drivers say why.
        error = (
            np.full(light_curves.shape, np.nan)
            if noise is None
            else np.where(visible, 1.0 / np.sqrt(safe_den), np.nan)
        )

    return light_curves, error


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def _lc_result(
    light_curves: NDArray,
    error: NDArray,
    norad_ids,
    freqs: NDArray,
    times_mjd: NDArray,
    data_col: str,
    corr: str,
    in_view: Optional[NDArray] = None,
) -> dict:
    """Bundle an estimate with the coordinates it is only interpretable against."""
    times_mjd = np.asarray(times_mjd, dtype=np.float64)
    norad_ids = [int(n) for n in norad_ids]

    with np.errstate(invalid="ignore", divide="ignore"):
        # z = coherent flux / noise floor, per (src, freq, time). The de-rotated
        # RFI is real, so Re(S_hat) carries the signal and `error` is its
        # standard deviation -> z ~ N(0, 1) wherever nothing is left.
        z = np.asarray(light_curves).real / np.asarray(error)

    result = {
        "light_curves": np.asarray(light_curves),  # (n_src, n_freq, n_time) complex
        "error": np.asarray(error),
        "z": z,
        "norad_ids": norad_ids,
        "titles": [str(n) for n in norad_ids],
        "freqs": np.asarray(freqs, dtype=np.float64),
        "times_mjd": times_mjd,
        "times_sec": (times_mjd - times_mjd[0]) * 86400.0,
        "in_view": None if in_view is None else np.asarray(in_view, dtype=bool),
        "data_col": data_col,
        "corr": corr,
    }

    return result


def _resolve_noise(noise, what: str):
    """The noise to weight by, or ``None`` with the loss said out loud."""
    if noise is not None:
        return noise

    print(
        f"Warning: {what} carries no usable noise estimate, so the light curves "
        "are both unweighted and unscaled. The antennas of a real array differ "
        "in sensitivity, so averaging them equally over-weights the loud "
        "baselines; and with no sigma there is no noise floor to quote, so the "
        "error and the z statistic are nan and no coverage can be computed. Set "
        "data.noise to get either back."
    )

    return None


def has_noise_scale(result: dict) -> bool:
    """Whether an estimate carries a noise floor, i.e. whether ``z`` means anything.

    False when the visibilities were filtered with no sigma to weight them by:
    the light curves are still there, but every ``error`` is ``nan`` and nothing
    downstream that divides by one has anything to say.
    """
    return bool(np.isfinite(np.asarray(result["error"])).any())


def _elevation_mask(orbit_records, times_jd, ants_itrf, min_elevation):
    """In-view mask for each record, or ``None`` when no cut is configured.

    Inclusive, as ``rfi.min_elevation`` is: the cut is the lowest elevation still
    modelled, so a sample sitting exactly on it is in view.
    """
    if min_elevation is None or len(orbit_records) == 0:
        return None

    elevations = get_satellite_elevations(orbit_records, times_jd, ants_itrf)

    return np.asarray(elevations) >= min_elevation


def _resolve_records(norad_ids, times_jd, extra_orbit_dir):
    """Orbit records for the requested IDs, through the run's own resolver.

    Goes via :func:`fetch_orbital_elements` so ``extra_orbit_dir``, the managed
    cache and SatChecker take exactly the precedence a run gives them, and the
    estimate is built from the same records the model would be.
    """
    if norad_ids is None or len(norad_ids) == 0:
        raise ValueError(
            "No satellites to filter for: give norad_ids, or a tabascal config "
            "whose satellites.norad_ids names them."
        )

    _, _, ids, records, n_real = fetch_orbital_elements(
        times_jd=np.asarray(times_jd),
        norad_ids=[int(n) for n in norad_ids],
        extra_orbit_dir=extra_orbit_dir,
    )

    # Under sharding the fetch pads the source list with duplicates of the last
    # satellite; a light curve is wanted for the real ones only.
    return [int(n) for n in ids[:n_real]], list(records[:n_real])


def light_curves_from_config(
    tab_config,
    vis: Optional[NDArray] = None,
    exclude_autos: bool = True,
    max_mem_gb: float = 1.0,
) -> dict:
    """Matched-filter light curves from an already-loaded :class:`TabConfig`.

    The in-process path: reuses the visibilities, noise, antenna positions, times
    and orbit records the config has loaded, so no second MS read is needed, and
    the curves come back ordered to match ``satellites.norad_ids`` -- no title
    matching, and no light-curve file.

    Curves are returned for the *real* satellites only. Under device sharding the
    source axis is padded with duplicates of the last satellite, and those rows
    are re-added as zeros by the seeding code that consumes this.

    The elevation cut is the run's own ``rfi.min_elevation`` mask, taken off the
    config rather than recomputed, so the estimate is masked exactly where the
    model is.

    Parameters
    ----------
    tab_config : tabascal.config.TabConfig
        A configured object exposing ``vis_obs``, ``flags``, ``noise``,
        ``ants_itrf``, ``times_jd``, ``times_mjd``, ``freqs``, ``phase_centre``,
        ``a1``, ``a2``, ``orbit_records`` and ``norad_ids``.
    vis : Array (n_bl, n_freq, n_time), optional
        Visibilities to filter; defaults to ``tab_config.vis_obs``.
    exclude_autos : bool, default True
    max_mem_gb : float, default 1.0

    Returns
    -------
    dict
        See :func:`_lc_result`.
    """
    if vis is None:
        vis = tab_config.vis_obs

    n_real = getattr(tab_config, "n_rfi_real", len(tab_config.norad_ids))
    norad_ids = [int(n) for n in tab_config.norad_ids[:n_real]]
    records = list(tab_config.orbit_records[:n_real])

    times_jd = np.asarray(tab_config.times_jd)
    rfi_phase = rfi_phase_from_records(
        records,
        np.asarray(tab_config.ants_itrf),
        times_jd,
        tab_config.phase_centre,
        np.asarray(tab_config.freqs),
    )

    # The mask the run already built, when it built one; otherwise the elevations
    # are evaluated here so a config that disabled masking for the model can
    # still ask for a masked estimate.
    rfi_mask = getattr(tab_config, "rfi_mask", None)
    if rfi_mask is None:
        in_view = _elevation_mask(
            records, times_jd, tab_config.ants_itrf,
            getattr(tab_config, "min_elevation", None),
        )
    else:
        in_view = np.asarray(rfi_mask, dtype=bool)[:n_real]

    flags = getattr(tab_config, "flags", None)
    flags = None if flags is None else np.asarray(flags)

    light_curves, error = matched_filter_light_curves(
        np.asarray(vis),
        rfi_phase,
        np.asarray(tab_config.a1),
        np.asarray(tab_config.a2),
        noise=_resolve_noise(getattr(tab_config, "noise", None), "the config"),
        flags=flags,
        in_view=in_view,
        exclude_autos=exclude_autos,
        max_mem_gb=max_mem_gb,
    )

    return _lc_result(
        light_curves,
        error,
        norad_ids,
        np.asarray(tab_config.freqs),
        np.asarray(tab_config.times_mjd),
        tab_config.args["data"]["data_col"],
        tab_config.args["data"]["corr"],
        in_view=in_view,
    )


def _read_ms(ms_path: str, freq, corr: str, data_col: str) -> dict:
    """One MS partition, read exactly as a run reads it -- noise column included."""
    from tabascal.ms import read_ms

    return read_ms(ms_path, freq, None, corr, data_col)


def _times_jd(ms: dict) -> NDArray:
    """The observation's instants on UTC, which is what the geometry reads.

    An MS declares the time scale of its ``TIME`` column, and it is not always
    UTC. :func:`tabascal.ms.read_ms` normalises whatever it finds onto UTC and
    reports that as ``times_jd``, leaving ``times_mjd`` on the declared scale so
    it stays comparable with the column itself. Rebuilding a Julian Date from
    ``times_mjd`` here would undo that: skyfield, ``sgp4jax.itrf_to_gcrf`` and
    the elevation calls all read UTC, so on a TAI-declared MS the propagation,
    the fringe and the elevation cut would every one of them be 37 s out -- some
    285 km along a LEO satellite's ground track, and nothing raises.
    """
    return np.asarray(ms["times_jd"])


def extract_light_curves_from_ms(
    ms_path: str,
    norad_ids: Optional[list] = None,
    corr: str = "xx",
    data_col: str = "DATA",
    freq: Optional[float] = None,
    exclude_autos: bool = True,
    extra_orbit_dir: Optional[str] = None,
    min_elevation: Optional[float] = 0.0,
    max_mem_gb: float = 1.0,
) -> dict:
    """Extract matched-filter RFI light curves from any column of an MS.

    The standalone entry point, used by the ``tabascal light-curve`` CLI. Reads
    the requested ``data_col`` (and the MS's own noise column, through the same
    :func:`tabascal.ms.read_ms` a run uses), propagates the satellites' orbit
    records over the MS times, and runs :func:`matched_filter_light_curves`.

    Parameters
    ----------
    ms_path : str
        Path to the Measurement Set.
    norad_ids : list[int]
        NORAD catalogue IDs; their orbit records are resolved through
        :mod:`tabascal.orbit`, with the same source precedence as a run.
    corr : str, default "xx"
        Correlation to read (``xx``/``xy``/``yx``/``yy``).
    data_col : str, default "DATA"
        MS data column to matched-filter.
    freq : float, optional
        If given, use only the single channel nearest this frequency (Hz).
    exclude_autos : bool, default True
        Drop autocorrelations from the beam-former.
    extra_orbit_dir : str, optional
        Extra local directory of orbit files, searched before the managed cache
        and SatChecker.
    min_elevation : float, optional
        Elevation in degrees below which a satellite is not filtered for. ``None``
        disables the cut.
    max_mem_gb : float, default 1.0
        Memory budget for the matched-filter time-chunk loop.

    Returns
    -------
    dict
        See :func:`_lc_result`. ``light_curves`` is ``(n_src, n_freq, n_time)``
        complex, ordered to match ``norad_ids``.
    """
    ms = _read_ms(ms_path, freq, corr, data_col)
    times_jd = _times_jd(ms)

    norad_ids, records = _resolve_records(norad_ids, times_jd, extra_orbit_dir)

    return _filter_visibilities(
        np.asarray(ms["vis_obs"]),
        ms,
        records,
        norad_ids,
        times_jd,
        data_col,
        corr,
        exclude_autos,
        min_elevation,
        max_mem_gb,
    )


def _filter_visibilities(
    vis, ms, records, norad_ids, times_jd, data_col, corr,
    exclude_autos, min_elevation, max_mem_gb,
):
    """Shared tail of the two MS-backed drivers."""
    phase_centre = {"ra": float(ms["ra"]), "dec": float(ms["dec"])}
    ants_itrf = np.asarray(ms["ants_itrf"])

    rfi_phase = rfi_phase_from_records(
        records, ants_itrf, times_jd, phase_centre, np.asarray(ms["freqs"])
    )
    in_view = _elevation_mask(records, times_jd, ants_itrf, min_elevation)

    light_curves, error = matched_filter_light_curves(
        vis,
        rfi_phase,
        np.asarray(ms["a1"]),
        np.asarray(ms["a2"]),
        noise=_resolve_noise(ms.get("noise"), "the MS partition"),
        flags=None if ms.get("flags") is None else np.asarray(ms["flags"]),
        in_view=in_view,
        exclude_autos=exclude_autos,
        max_mem_gb=max_mem_gb,
    )

    return _lc_result(
        light_curves,
        error,
        norad_ids,
        np.asarray(ms["freqs"]),
        np.asarray(ms["times_mjd"]),
        data_col,
        corr,
        in_view=in_view,
    )


def _half_channel(freqs: NDArray, chan_widths=None) -> NDArray:
    """Half the width of each channel: the radius inside which two grids agree.

    Per channel, because a spectral window need not be uniform -- a band with a
    4 MHz channel next to a 100 kHz one has no single tolerance, and the wide
    one's would accept a model channel that is nowhere near the narrow one.

    Falls back to half the spacing to the nearest neighbouring channel where no
    widths are given, which is the most the frequencies alone can say; a single
    channel with no width has neither, and is then required to match exactly.
    """
    freqs = np.asarray(freqs, dtype=np.float64)

    if chan_widths is not None:
        widths = np.abs(np.atleast_1d(np.asarray(chan_widths, dtype=np.float64)))
        if widths.size == 1:
            widths = np.repeat(widths, len(freqs))
        if widths.size == len(freqs):
            return 0.5 * widths

    if len(freqs) < 2:
        return np.zeros(len(freqs))

    gaps = np.abs(np.diff(freqs))
    # The smaller of the two neighbouring gaps, so the radius can never reach
    # into a channel that belongs to the other side.
    return 0.5 * np.minimum(
        np.concatenate([gaps[:1], gaps]), np.concatenate([gaps, gaps[-1:]])
    )


def _check_zarr_identity(xds, zarr_path: str, n_bl, times_mjd, corr=None) -> None:
    """Refuse a results store that is not this observation's.

    Matching the channels by frequency says the two grids describe the same part
    of the band. It says nothing about whether the store belongs to *this*
    observation: another pointing, another correlation or another night can carry
    the same shapes, and the residual is then two unrelated datasets differenced
    without a word.

    What can be checked is what a results zarr records today -- its baseline and
    timestep counts, the cadence of its time coordinate, and the correlation it
    was fitted on. Absolute times and an observation identity would settle it
    outright; writing those belongs with the results writer, not here.

    A store that records no ``corr`` attribute is not evidence of disagreement,
    so it passes; a store that records a different one is.
    """
    model = xds.vis_obs
    n_time = len(np.asarray(times_mjd))

    if int(model.sizes["bl"]) != int(n_bl):
        raise ValueError(
            f"{zarr_path} holds {int(model.sizes['bl'])} baselines but the "
            f"visibilities being read have {int(n_bl)}, so it is not this "
            "observation's model."
        )

    if int(model.sizes["time"]) != n_time:
        raise ValueError(
            f"{zarr_path} holds {int(model.sizes['time'])} timesteps but the "
            f"visibilities being read have {n_time}, so it is not this "
            "observation's model."
        )

    if corr is not None and xds.attrs.get("corr") not in (None, corr):
        raise ValueError(
            f"{zarr_path} was fitted on correlation {xds.attrs['corr']!r} but "
            f"{corr!r} is being read. Subtracting one correlation's model from "
            "another's visibilities is not a residual."
        )

    # Seconds from the start of the observation, which is what the results
    # writer stores. Compared as a cadence rather than as absolute times: the
    # store carries no epoch to compare against.
    if "time" not in xds.coords or n_time < 2:
        return

    model_step = float(np.mean(np.diff(np.asarray(xds["time"].values, dtype=float))))
    ms_step = float(
        np.mean(np.diff(np.asarray(times_mjd, dtype=np.float64))) * 86400.0
    )

    if not np.isclose(model_step, ms_step, rtol=1e-3, atol=1e-6):
        raise ValueError(
            f"{zarr_path} has a cadence of {model_step:.6g} s but the "
            f"visibilities being read step {ms_step:.6g} s, so the two are not "
            "the same observation however well their shapes line up."
        )


def _model_on_ms_channels(xds, freqs, zarr_path: str, chan_widths=None) -> NDArray:
    """A run's gained prediction, on the channels the visibilities were read on.

    Matched by frequency rather than by position. The two need not agree: a
    ``freq`` narrows the read to one channel while the results zarr holds the
    whole band the run fitted, and subtracting positionally would then take the
    model's channel 0 from the data's channel *n* -- the right shape, no error,
    and a residual that is mostly the difference between two channels of the
    model. Nearest match within half a channel (:func:`_half_channel`), so a
    store covering a different band is refused rather than silently differenced
    against its closest edge.

    A store with no ``freq`` coordinate cannot be matched at all; there the
    channel counts are required to agree and the axes are taken as parallel,
    which is the most that can be said about it.

    Takes the frequencies and widths themselves rather than a reader's result,
    so the in-process path can pass a :class:`TabConfig`'s and both residuals go
    through the same alignment.
    """
    model = xds.vis_obs.isel(sample=0)
    freqs = np.asarray(freqs, dtype=np.float64)

    if "freq" not in xds.coords and "freq" not in xds.variables:
        n_model = int(model.sizes["freq"])
        if n_model != len(freqs):
            raise ValueError(
                f"{zarr_path} has no 'freq' coordinate, so its {n_model} channels "
                f"can only be matched to the {len(freqs)} being read by position "
                "-- and the counts differ. Re-run tabascal to write a store that "
                "records its frequencies, or read the same channels the run fitted."
            )
        return np.asarray(model.data.compute())

    model_freqs = np.asarray(xds["freq"].values, dtype=np.float64)
    nearest = np.abs(model_freqs[None, :] - freqs[:, None]).argmin(axis=1)
    offset = np.abs(model_freqs[nearest] - freqs)
    # Inside half a channel the two grids are the same channel; outside it they
    # are different measurements and no subtraction is defined.
    tol = _half_channel(freqs, chan_widths)
    over = offset > tol

    if over.any():
        worst = int(np.argmax(offset - tol))
        raise ValueError(
            f"{zarr_path} does not cover the frequencies being read: channel "
            f"{worst} is at {freqs[worst] / 1e6:.4f} MHz and the nearest in the "
            f"store is {model_freqs[nearest[worst]] / 1e6:.4f} MHz, "
            f"{offset[worst] / 1e6:.4f} MHz away -- more than half that "
            f"channel's width ({2 * tol[worst] / 1e6:.4f} MHz). The residual "
            "would be the difference between two different channels."
        )

    return np.asarray(model.isel(freq=nearest).data.compute())


def extract_light_curves_from_zarr(
    ms_path: str,
    zarr_path: str,
    norad_ids: Optional[list] = None,
    corr: str = "xx",
    data_col: str = "DATA",
    freq: Optional[float] = None,
    exclude_autos: bool = True,
    extra_orbit_dir: Optional[str] = None,
    min_elevation: Optional[float] = 0.0,
    max_mem_gb: float = 1.0,
) -> dict:
    """Matched-filter the residual of a tabascal run, taken from its results zarr.

    **This is the way to score a run.** The MS result columns (``TAB_RES_DATA``
    et al.) are overwritten by *every* tabascal run, so scoring off the MS is only
    valid if those columns happen to belong to the run meant. The zarr is written
    once per run and per suffix, so a later run cannot invalidate it.

    The residual is formed as ``data_col - zarr.vis_obs``. The zarr's ``vis_obs``
    is the model's own *gained* prediction, ``apply_gains(gains, vis_ast +
    vis_rfi)``, so this is exactly the residual :func:`tabascal.write.write_results_ms` would
    write, without the MS round trip.

    The model is matched to the MS's channels **by frequency**, so a ``freq``
    that narrows the read to one channel still subtracts that channel's model.
    See :func:`_model_on_ms_channels`.

    ``data_col`` is the *reference* column the residual is formed against (e.g.
    ``DATA``), not a residual column. Everything else is as
    :func:`extract_light_curves_from_ms`.
    """
    import xarray as xr

    ms = _read_ms(ms_path, freq, corr, data_col)
    times_jd = _times_jd(ms)

    norad_ids, records = _resolve_records(norad_ids, times_jd, extra_orbit_dir)

    xds = xr.open_zarr(zarr_path)
    _check_zarr_identity(
        xds, zarr_path, len(np.asarray(ms["a1"])), ms["times_mjd"], corr
    )
    model = _model_on_ms_channels(
        xds, ms["freqs"], zarr_path, ms.get("chan_widths")
    )
    residual = np.asarray(ms["vis_obs"]) - model

    label = f"{data_col} - {os.path.basename(str(zarr_path).rstrip('/'))}"

    return _filter_visibilities(
        residual,
        ms,
        records,
        norad_ids,
        times_jd,
        label,
        corr,
        exclude_autos,
        min_elevation,
        max_mem_gb,
    )


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_light_curves_npz(path: str, result: dict) -> None:
    """Save a light-curve result as the ``rfi.est`` interchange format.

    The four names :func:`tabascal.components.rfi_signal.read_light_curves`
    requires -- ``light_curves`` ``(n_src, n_time, n_freq)``, ``norad_ids``,
    ``times`` (MJD) and ``freqs`` (Hz) -- so the output of a
    ``tabascal light-curve`` run can be pointed at with ``rfi.est`` unchanged.

    ``light_curves`` is the *magnitude* ``|S_hat|``, an apparent flux in Jy: the
    reader casts to float64, which would silently discard the imaginary part of a
    complex array. The native complex estimate is kept alongside it under
    ``light_curves_complex``, together with the noise floor (``error``), the
    z statistic and the in-view mask. Readers of the format ignore the extras.

    Parameters
    ----------
    path : str
        Output ``.npz`` path.
    result : dict
        A dict from one of the driver functions.
    """
    # (n_src, n_freq, n_time) -> the format's (n_src, n_time, n_freq).
    lc = np.swapaxes(np.asarray(result["light_curves"]), 1, 2)
    error = np.swapaxes(np.asarray(result["error"]), 1, 2)
    z = np.swapaxes(np.asarray(result["z"]), 1, 2)

    arrays = dict(
        light_curves=np.abs(lc),
        norad_ids=np.asarray(result["norad_ids"]),
        times=np.asarray(result["times_mjd"], dtype=np.float64),
        freqs=np.asarray(result["freqs"], dtype=np.float64),
        light_curves_complex=lc,
        error=error,
        z=z,
        data_col=str(result["data_col"]),
        corr=str(result["corr"]),
    )
    if result.get("in_view") is not None:
        arrays["in_view"] = np.asarray(result["in_view"], dtype=bool)

    np.savez(path, **arrays)


# ---------------------------------------------------------------------------
# z-statistic (residual / floor): coverage + spectrograms
# ---------------------------------------------------------------------------

def rayleigh_threshold(z_crit: float) -> float:
    """The ``|S_hat|/error`` cut enclosing the same probability as ``|z| <= z_crit``.

    Under the complex Gaussian null the real and imaginary parts of ``S_hat`` are
    independent N(0, error^2), so ``|S_hat|/error`` is Rayleigh(1) with
    ``P(R <= c) = 1 - exp(-c^2/2)``. Matching that to the two-sided normal
    probability leaves ``c = sqrt(-2 ln(erfc(z / sqrt 2)))`` -- 3.44 for the
    usual 3 sigma -- so the two coverages are read on the same scale rather than
    against thresholds that mean different things.

    The tail is taken from ``erfc`` rather than as ``1 - erf``: that subtraction
    cancels to exactly zero once ``erf`` rounds to 1, somewhere past 6 sigma, and
    the threshold came back infinite -- which marks every cell as consistent with
    noise and reports a coverage of 100% for any data at all. ``erfc`` *is* the
    two-sided tail, computed without the cancellation.
    """
    from math import erfc, log, sqrt

    tail = erfc(float(z_crit) / sqrt(2.0))

    return float("inf") if tail <= 0.0 else sqrt(-2.0 * log(tail))


def coverage_stats(result: dict, z_crit: float = 3.0) -> dict:
    """Fraction of time-frequency cells consistent with noise, per source.

    The z statistic ``z = Re(S_hat) / error`` would be ~ N(0, 1) wherever nothing
    is left after subtraction, so a well-cleaned source has ``|z|`` within
    ``z_crit`` almost everywhere. ``coverage`` is the fraction of finite
    (freq, time) cells with ``|z| <= z_crit``; ``max_z`` is the peak residual
    significance.

    **The z statistic assumes the data are phase calibrated.** ``Re(S_hat)`` is
    the whole of a de-rotated real source only when nothing else rotates it, so
    read it on a calibrated column (``CORRECTED_DATA``, the ``TAB_*`` columns, or
    a residual against a fitted model).

    ``amp_coverage`` is the same statistic on ``|S_hat|/error``, against
    :func:`rayleigh_threshold`; its null is analytic -- Rayleigh(1) -- so it
    carries no ``excess`` column. It is invariant to a rotation **common to every
    baseline**: an overall phase offset, or a stable phase on the source, turns
    ``S_hat`` as a whole, which empties ``Re(S_hat)`` and spills the source into
    the imaginary part that ``null_coverage`` is measured on -- both halves of
    that comparison then move the wrong way while the magnitude is untouched.

    It is **not** immunity to an uncalibrated antenna gain. A gain multiplies
    each baseline before the average, ``S_hat = S * sum_bl w g_p conj(g_q) /
    sum_bl w``, so antenna-dependent phases decorrelate the sum itself: the
    estimate shrinks and its magnitude with it, and *both* statistics drift
    toward "nothing here". On a raw column neither number is a detection
    threshold so much as a lower bound. The optimistic-floor caveat below applies
    to both.

    **Compare against ``null_coverage``, not against the analytic 2*Phi(z)-1.**
    The floor assumes the de-rotated per-baseline samples are independent. They
    are not: residual sky is coherent across baselines, so the floor is optimistic
    and the analytic null over-states the expected coverage. ``null_coverage`` is
    the same statistic on ``Im(S_hat)/error`` -- after de-rotation a real source
    sits purely in the real part, so the imaginary part is a matched, source-free
    null carrying the same noise and the same correlation structure. A source is
    consistent with noise when its coverage is not significantly *below* the null;
    the *excess* ``null_coverage - coverage`` is the part attributable to a real
    residual.

    Parameters
    ----------
    result : dict
        A driver result, carrying ``z``, ``light_curves`` and ``error``.
    z_crit : float, default 3.0
        Detection threshold; cells above it are flagged as residual.

    Returns
    -------
    dict
        ``per_source`` (title, coverage, null_coverage, excess, amp_coverage,
        max_z, max_amp, n_cells) and ``overall`` (pooled coverage, null,
        amp_coverage, worst source, mean, z_crit, amp_crit).
    """
    if "z" not in result:
        raise ValueError(
            "result has no 'z'; build it with one of the driver functions."
        )

    z = np.asarray(result["z"])
    titles = result["titles"]
    amp_crit = rayleigh_threshold(z_crit)

    with np.errstate(invalid="ignore", divide="ignore"):
        error = np.asarray(result["error"])
        z_null = np.asarray(result["light_curves"]).imag / error
        z_amp = np.abs(np.asarray(result["light_curves"])) / error

    per_source, pooled_in, pooled_n, pooled_null_in, pooled_amp_in = [], 0, 0, 0, 0
    for i, title in enumerate(titles):
        zi = z[i][np.isfinite(z[i])]
        n = zi.size
        n_in = int(np.sum(np.abs(zi) <= z_crit))
        cov = (n_in / n) if n else float("nan")

        zn = z_null[i][np.isfinite(z_null[i])]
        n_null_in = int(np.sum(np.abs(zn) <= z_crit))
        null_cov = (n_null_in / zn.size) if zn.size else float("nan")

        za = z_amp[i][np.isfinite(z_amp[i])]
        n_amp_in = int(np.sum(za <= amp_crit))
        amp_cov = (n_amp_in / za.size) if za.size else float("nan")

        per_source.append(
            dict(
                title=title,
                coverage=cov,
                null_coverage=null_cov,
                excess=(null_cov - cov) if np.isfinite(null_cov) else float("nan"),
                amp_coverage=amp_cov,
                max_z=float(np.max(np.abs(zi))) if n else float("nan"),
                max_amp=float(np.max(za)) if za.size else float("nan"),
                n_cells=int(n),
            )
        )
        pooled_in += n_in
        pooled_n += n
        pooled_null_in += n_null_in
        pooled_amp_in += n_amp_in

    covs = [p["coverage"] for p in per_source]
    worst = min(per_source, key=lambda p: p["coverage"]) if per_source else None

    return dict(
        per_source=per_source,
        overall=dict(
            coverage=(pooled_in / pooled_n) if pooled_n else float("nan"),
            null_coverage=(pooled_null_in / pooled_n) if pooled_n else float("nan"),
            amp_coverage=(pooled_amp_in / pooled_n) if pooled_n else float("nan"),
            mean_coverage=float(np.nanmean(covs)) if covs else float("nan"),
            worst_source=worst["title"] if worst else None,
            worst_coverage=worst["coverage"] if worst else float("nan"),
            z_crit=z_crit,
            amp_crit=amp_crit,
        ),
    )


def plot_z_spectrograms(  # pragma: no cover - matplotlib output, not unit-tested
    result: dict,
    save_path: str,
    z_crit: float = 3.0,
    vmax: Optional[float] = None,
) -> str:
    """Per-source spectrogram of the z statistic (residual / floor).

    One panel per source, time on x, frequency (MHz) on y, colour = signed
    ``z = Re(S_hat) / error`` on a diverging scale (blue = over-subtracted, red =
    under-subtracted residual). Single-channel data degrades to a z-vs-time line
    plot with the +/- ``z_crit`` band shaded.

    Returns
    -------
    str
        ``save_path``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.asarray(result["z"])  # (n_src, n_freq, n_time)
    titles = result["titles"]
    times = np.asarray(result["times_sec"], dtype=float)
    freqs_mhz = np.asarray(result["freqs"], dtype=float) / 1e6
    n_src, n_freq, _ = z.shape

    cov = coverage_stats(result, z_crit)["per_source"]
    if vmax is None:
        finite = z[np.isfinite(z)]
        vmax = (
            max(2.0 * z_crit, float(np.percentile(np.abs(finite), 99)))
            if finite.size
            else 2.0 * z_crit
        )

    fig, axes = plt.subplots(n_src, 1, figsize=(10, 2.8 * n_src), squeeze=False)
    for i, ax in enumerate(axes[:, 0]):
        sub = (
            f"cov(|z|<={z_crit:g})={cov[i]['coverage'] * 100:.1f}%   "
            f"max|z|={cov[i]['max_z']:.1f}"
        )
        if n_freq == 1:
            ax.plot(times, z[i, 0], color="C3", lw=0.9)
            ax.axhspan(-z_crit, z_crit, color="gray", alpha=0.2, lw=0)
            ax.axhline(0, color="0.5", lw=0.8)
            ax.set_ylabel("z = resid/floor")
        else:
            pcm = ax.pcolormesh(
                times, freqs_mhz, z[i], shading="nearest",
                cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            )
            cb = fig.colorbar(pcm, ax=ax, label="z = resid/floor")
            cb.ax.axhline(z_crit, color="k", lw=1.0, ls="--")
            cb.ax.axhline(-z_crit, color="k", lw=1.0, ls="--")
            ax.set_ylabel("Freq [MHz]")
        ax.set_title(f"{titles[i]} - {result.get('data_col', '')}   ({sub})", fontsize=9)
        ax.set_xlabel("Time [s]")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path
