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

Which baselines the filter should be summed over is a question of its own, and
:func:`coherent_baseline_mask` answers it from the orbit accuracy and the fringe
rate rather than from a hand-tuned cut; it is a pure array function so the jitted
tau-scan and identification search (#190/#191) can apply the same selection
inside their own traces.

**The along-track offset.** A TLE's dominant error is along-track -- kilometres
to tens of kilometres of drag mismodelling and unannounced manoeuvres -- and an
along-track error is very nearly a pure *time offset* in the trajectory. So one
scanned parameter, ``tau``, recovers the bulk of it: evaluate the orbit at
``t + tau``, build the near-field fringe model on a fine grid inside each
integration, and coherently correlate it against the data over the baselines
:func:`coherent_baseline_mask` keeps. :func:`fit_time_offset` is the whole
measurement -- horizon window, coherence cut, scan, best cell and a
decohered-antenna null for its significance -- and ``tabascal light-curve
--fit-offset`` exposes it, extracting the curves at the offset it measured and
recording that offset in the output. Its core
(:func:`near_field_fringe_model`, :func:`matched_filter_sums`,
:func:`coherence_scores`, :func:`tau_scan`) is pure ``jax.numpy`` over
fixed-shape arrays, scanning the grid with ``lax.map`` and undecorated so the
drivers own the ``jit``: one compilation covers the whole scan, and the batched
identification search of #191 ``vmap``\\ s the same function over candidates.
:func:`shift_orbit_record_epoch` is the other end of it -- an orbit record moved
by ``-tau``, which reproduces the measured trajectory through
``extra_orbit_dir`` with no further code.

**Which satellite it is.** Given a TLE snapshot and nothing else,
:func:`enumerate_candidates` screens the records down to the ones that were above
the horizon, :func:`search_candidates` runs that same scan over all of them --
``jax.vmap`` over a candidate axis, one jitted program per batch, each
candidate's horizon mask and coherence cut applied *inside* the statistic so the
shapes stay static -- and :func:`select_detections` reads the ranking against the
decohered null. ``tabascal search`` is the command, and what it emits is the
``satellites.norad_ids`` list a run needs (:func:`write_config_fragment`), the
ranking table (:func:`write_search_results`), and the light curves and shifted
orbit records of whatever it named. That is GitHub #191, which produced
STARLINK-1765 out of 2551 records on the MWA Cen A dataset.

Only the satellite-trajectory source is implemented. RA/Dec and Alt/Az pointings
can be added by constructing ``rfi_xyz`` from those and feeding
:func:`rfi_phase_from_positions`.
"""

import calendar
import os
from datetime import datetime
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from numpy.typing import NDArray

from satchecker_client.records import KIND_TLE, record_epoch_jd, record_kind
from satchecker_client.tle_parse import tle_checksum
from tabascal.components.trajectory import (
    fetch_orbital_elements,
    get_satellite_elevations,
    get_satellite_positions,
    itrs_to_gcrs_sf,
)
from tabascal.interferometry import (
    C,
    Omega_e,
    get_rfi_phase_numpy,
    itrf_to_uvw_numpy,
)
from tabascal.noise import broadcast_to_vis
from tabascal.time import (
    datetime_to_jd,
    gast_deg,
    jd_to_datetime,
    mjd_to_jd,
    to_utc_mjd,
)


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
    time_offsets_s=None,
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
    time_offsets_s : float or sequence of float, optional
        Along-track offset per source, in seconds, as measured by
        :func:`fit_time_offset`: source ``i``'s orbit is evaluated at
        ``times_jd + tau_i``. A scalar applies to every source. It moves the
        **satellite only** -- the antennas, the sidereal angle and the phase
        tracking stay at ``times_jd``, because ``tau`` is an error in the orbit,
        not in the observation's clock. Re-propagating the whole geometry at
        ``t + tau`` would rotate the Earth under the fringe tracking as well and
        give a different phase. ``None`` (the default) is the behaviour every
        existing caller has, bit for bit.

    Returns
    -------
    Array (n_src, n_ant, n_freq, n_time)
    """
    times_jd = np.asarray(times_jd)

    if time_offsets_s is None:
        rfi_xyz = np.asarray(get_satellite_positions(orbit_records, list(times_jd)))
    else:
        taus = np.atleast_1d(np.asarray(time_offsets_s, dtype=np.float64))
        if taus.size == 1:
            taus = np.repeat(taus, len(orbit_records))
        if taus.size != len(orbit_records):
            raise ValueError(
                f"time_offsets_s has {taus.size} entries for "
                f"{len(orbit_records)} sources; give one per source, or a "
                "single offset for all of them."
            )
        # One propagation per source, since each is read at its own instants.
        rfi_xyz = np.stack(
            [
                np.asarray(
                    get_satellite_positions([record], list(times_jd + tau / 86400.0))
                )[0]
                for record, tau in zip(orbit_records, taus)
            ]
        )

    return rfi_phase_from_positions(rfi_xyz, ants_itrf, times_jd, phase_centre, freqs)


# ---------------------------------------------------------------------------
# Baseline coherence: which baselines a TLE can steer
# ---------------------------------------------------------------------------

def baseline_lengths(ants_itrf: ArrayLike, a1: ArrayLike, a2: ArrayLike) -> Array:
    """Physical separation of each antenna pair, in metres.

    Both coherence criteria act on the baseline component perpendicular to the
    line of sight *to the satellite*, which changes as it crosses the sky. The
    physical length bounds that component from above in every direction, so it
    is the conservative choice and needs no per-timestep geometry. The uv
    distance is not a substitute: it is the projection toward the phase centre,
    not toward the satellite, and would admit baselines the satellite sees at
    full length.

    Parameters
    ----------
    ants_itrf : Array (n_ant, 3)
        Antenna positions in the ITRF (ECEF) frame, in metres.
    a1, a2 : Array (n_bl,)
        Antenna indices of each baseline.

    Returns
    -------
    Array (n_bl,)
        Baseline length in metres.
    """
    ants_itrf = jnp.asarray(ants_itrf)

    return jnp.linalg.norm(
        ants_itrf[jnp.asarray(a1)] - ants_itrf[jnp.asarray(a2)], axis=-1
    )


def tle_coherence_length(
    freq: ArrayLike, range_m: ArrayLike, sigma_transverse_m: ArrayLike
) -> Array:
    """Longest baseline an orbit known to ``sigma`` metres can still steer.

    A transverse position error ``delta`` at slant range ``r`` moves the
    satellite's apparent direction by ``delta / r``, which costs
    ``2 pi (b / lam) (delta / r)`` of template phase on a baseline of length
    ``b``. Holding that to one radian gives::

        b_tle = lam r / (2 pi delta)

    Only the transverse error enters. The along-track error, which is the larger
    part of a TLE's, is very nearly a pure time offset and is absorbed by the tau
    search instead of by this cut.

    The expression is symmetric in ``b`` and ``delta``, so it inverts itself: the
    same call evaluated at ``sigma_transverse_m = b`` is the largest orbit error
    the baseline ``b`` tolerates.

    Parameters
    ----------
    freq : float or Array
        Observing frequency in Hz.
    range_m : float or Array
        Slant range to the satellite in metres.
    sigma_transverse_m : float or Array
        Transverse (across the line of sight) TLE position error in metres. Taken
        as a magnitude: a caller holding a signed component -- an offset measured
        along some axis -- means its size, and a negative ceiling would be met by
        no baseline at all. Zero is a perfect orbit and returns an infinite
        coherence length.

    Returns
    -------
    Array
        Coherence length in metres, at the broadcast shape of the inputs.
    """
    lam = C / jnp.asarray(freq)
    delta = jnp.abs(jnp.asarray(sigma_transverse_m))

    return lam * jnp.asarray(range_m) / (2.0 * jnp.pi * delta)


def fringe_rate_coherence_length(
    freq: ArrayLike,
    range_m: ArrayLike,
    n_fine: ArrayLike,
    delta_t: ArrayLike,
    v_perp_m_s: ArrayLike,
) -> Array:
    """Longest baseline the model average itself can follow, in metres.

    A satellite crossing at transverse speed ``v_perp`` sweeps the baseline
    fringe at ``(b / lam) (v_perp / r)`` hertz. A model that averages ``n_fine``
    sub-steps over an integration of length ``delta_t`` samples that fringe on a
    grid of spacing ``delta_t / n_fine``, so it can follow it only up to the
    Nyquist rate of its own grid, ``n_fine / (2 delta_t)``. Equating the two::

        b_fringe = lam r n_fine / (2 delta_t v_perp)

    Past it the template decoheres inside the integration against its own
    discretisation, however good the orbit is; the cure is more fine steps or a
    shorter dump, not a better TLE.

    Parameters
    ----------
    freq, range_m : float or Array
        See :func:`tle_coherence_length`.
    n_fine : int or Array
        Fine sub-steps the model averages over per integration.
    delta_t : float or Array
        Integration (dump) time in seconds.
    v_perp_m_s : float or Array
        Satellite speed across the line of sight in m/s. Taken as a magnitude,
        for the same reason as ``sigma_transverse_m``: a pass in the other
        direction fringes at the same rate. Zero is a stationary emitter, with no
        fringe to outrun, and returns an infinite length.

    Returns
    -------
    Array
        Coherence length in metres, at the broadcast shape of the inputs.
    """
    lam = C / jnp.asarray(freq)
    v_perp = jnp.abs(jnp.asarray(v_perp_m_s))

    return (
        lam
        * jnp.asarray(range_m)
        * jnp.asarray(n_fine)
        / (2.0 * jnp.asarray(delta_t) * v_perp)
    )


def coherent_baseline_mask(
    bl_len: ArrayLike,
    freq: ArrayLike,
    range_m: ArrayLike,
    sigma_transverse_m: ArrayLike,
    n_fine: ArrayLike,
    delta_t: ArrayLike,
    v_perp_m_s: ArrayLike,
    soft: bool = False,
) -> Array:
    """Baselines a trajectory is accurate enough to beam-form with.

    Two independent effects cap the baseline over which the template phase can be
    trusted: the orbit error, at ``b_tle = lam r / (2 pi delta)`` for a radian of
    phase (:func:`tle_coherence_length`), and the fringe rate against the Nyquist
    rate of the model's own fine grid, at
    ``b_fringe = lam r n_fine / (2 delta_t v_perp)``
    (:func:`fringe_rate_coherence_length`). They are unrelated, so the smaller of
    the two binds, element by element, and a baseline is kept when
    ``b <= min(b_tle, b_fringe)``. Both are always in play: all three fringe
    inputs are required, and a stationary emitter -- or a caller who wants the
    orbit ceiling alone -- passes ``v_perp_m_s = 0.0``, which sends ``b_fringe``
    to ``+inf`` so that it drops out of the minimum. There is no half-specified
    call to get wrong.

    Beyond the binding length a baseline does not merely stop helping: it enters
    the coherent sum with an essentially random phase and dilutes the statistic
    the shorter baselines built. That is what the MWA Cen A case study measured.
    At 175 MHz and a 567 km slant range a 600 m baseline tolerates ``delta`` of
    about 258 m while the full 5.3 km array needs about 29 m, and Starlink TLEs
    carry 0.1-1 km of transverse error -- so the detection lived entirely in the
    1004 baselines under 600 m (5.6 sigma), while the phase-coherent search over
    all 9180 ranked the true satellite around 34th.

    With ``soft=True`` the step is replaced by ``exp(-(b / b_coh)^2)``, a Gaussian
    taper on the same scale: unity at zero spacing, ``1/e`` where the hard mask
    cuts. It down-weights the marginal baselines instead of discarding them,
    which is the gentler choice when ``sigma`` is itself uncertain. It is a
    *weighting*, not a wider cut: the Gaussian is never exactly zero, so the
    drivers keep the hard mask as the baseline support and apply these values
    inside it (:func:`_coherence_weights`). Reading the support off the weights
    instead would admit every baseline the array has.

    One range suffices, taken at mid-observation: ``b_coh`` depends on ``r`` only
    linearly and the slant range varies by a few tens of percent over a pass,
    which moves the cut far less than the order-of-magnitude uncertainty on
    ``delta`` does.

    Everything goes through ``jax.numpy``, so this composes inside the jitted,
    GPU-resident matched-filter core that consumes it. ``soft`` is the one static
    argument -- it selects the output dtype and so cannot be traced.

    Only the two magnitudes are made sign-safe; ``freq``, ``range_m``, ``delta_t``
    and ``n_fine`` are positive-domain quantities that are taken as given, since
    a negative frequency or dump time is a caller error rather than a convention
    to absorb.

    Parameters
    ----------
    bl_len : float or Array
        Baseline lengths in metres (see :func:`baseline_lengths`).
    freq, range_m, sigma_transverse_m : float or Array
        See :func:`tle_coherence_length`. ``sigma_transverse_m`` is read as a
        magnitude.
    n_fine, delta_t, v_perp_m_s : int, float or Array
        See :func:`fringe_rate_coherence_length`. Required, not optional:
        ``v_perp_m_s = 0.0`` is how a caller asks for the TLE ceiling alone.
        ``v_perp_m_s`` is read as a magnitude.
    soft : bool, default False
        Return Gaussian weights rather than a boolean mask.

    Returns
    -------
    Array
        Boolean mask, or float weights where ``soft``, at the broadcast shape of
        the inputs.
    """
    b_coh = jnp.minimum(
        tle_coherence_length(freq, range_m, sigma_transverse_m),
        fringe_rate_coherence_length(freq, range_m, n_fine, delta_t, v_perp_m_s),
    )

    bl_len = jnp.asarray(bl_len)

    # A perfect orbit or a stationary emitter divides by zero above and lifts
    # its own ceiling to +inf, leaving the other one to bind; with both lifted
    # b_coh is +inf and both branches carry that through to "everything is
    # coherent" without a nan. Guarding the division instead would evaluate 0/0
    # in one branch, and an epsilon would move the cut.
    if soft:
        return jnp.exp(-((bl_len / b_coh) ** 2))

    return bl_len <= b_coh


def _coherence_weights(
    bl_len: NDArray,
    mean_freq: float,
    range_m: float,
    sigma_transverse_m: float,
    n_fine: int,
    delta_t: float,
    v_fringe: float,
    soft: bool,
):
    """The baselines a satellite can be beam-formed over, and their weights.

    Two calls to :func:`coherent_baseline_mask` rather than one, because ``soft``
    says how the baselines *inside* the cut are weighted and not which ones are
    in it. The Gaussian is never exactly zero, so a support read off the soft
    weights is every baseline the array has -- an 8 km one whose fringe smears
    away inside a dump included, at a weight of ``1e-210`` that changes no answer
    and costs a whole path model apiece. Worse, whether such a weight underflows
    to zero depends on the precision the scan happens to be run in, so the
    baseline list would depend on it too.

    Inside the support the taper is bounded below by ``1/e``, since that is what
    the Gaussian is where the hard mask cuts, so nothing kept here is kept at a
    weight that cannot matter.

    Returns
    -------
    (Array (n_bl,) bool, Array (n_bl,) float64)
        The hard support, and the weight to apply to each baseline: the Gaussian
        taper where ``soft``, ones otherwise. Both are zero outside the support.
    """
    cut = dict(
        bl_len=bl_len, freq=mean_freq, range_m=range_m,
        sigma_transverse_m=sigma_transverse_m, n_fine=n_fine, delta_t=delta_t,
        v_perp_m_s=v_fringe,
    )
    support = np.asarray(coherent_baseline_mask(**cut), dtype=bool)

    if not soft:
        return support, support.astype(np.float64)

    taper = np.asarray(coherent_baseline_mask(**cut, soft=True), dtype=np.float64)

    return support, np.where(support, taper, 0.0)


# ---------------------------------------------------------------------------
# Near-field geometry on the fine grid
# ---------------------------------------------------------------------------

#: Along-track offsets :func:`fit_time_offset` scans by default, in seconds:
#: ``+-4 s`` in ``0.25 s`` steps, 33 points including 0 and both ends. Wide
#: enough for a day-old Starlink TLE (the MWA Cen A case sat at -2.25 s on a
#: TLE 1.4 h old) and fine enough to resolve the peak an array of a few hundred
#: metres gives it -- see :func:`fit_time_offset` on sizing the step.
DEFAULT_TAU_GRID = np.arange(-4.0, 4.0 + 0.125, 0.25)


def fine_time_offsets(n_fine: int, delta_t: float) -> NDArray:
    """Where inside an integration the fringe model is sampled, in seconds.

    ``n_fine`` equal sub-steps spanning one dump, taken at their **midpoints**:
    ``((k + 0.5) / n_fine - 0.5) * delta_t``. Midpoints rather than edges,
    because an edge grid samples the boundary between two integrations twice and
    biases each average by half a sub-step. ``n_fine = 1`` is then exactly the
    integration centre, which is where the forward model's own template lives --
    so the model reduces to :func:`rfi_phase_from_records`'s at one step, and the
    two cannot drift apart.

    Parameters
    ----------
    n_fine : int
        Sub-steps per integration.
    delta_t : float
        Integration (dump) time in seconds.

    Returns
    -------
    Array (n_fine,) float64
        Offsets from the integration centre, in seconds.
    """
    n_fine = int(n_fine)

    return ((np.arange(n_fine, dtype=np.float64) + 0.5) / n_fine - 0.5) * float(delta_t)


def near_field_baseline_paths(
    record,
    ants_itrf: NDArray,
    times_jd: NDArray,
    phase_centre: dict,
    a1: NDArray,
    a2: NDArray,
    n_fine: int,
    delta_t: float,
    taus_s=0.0,
) -> NDArray:
    """Per-baseline near-field path difference on the fine grid, in metres.

    The quantity the whole search is built on. Per antenna the path is the
    *spherical* one, ``|x_sat - x_a|`` -- never the plane-wave projection -- plus
    the phase-tracking term ``w_a`` the visibilities are already rotated by,
    exactly as :func:`tabascal.interferometry.get_rfi_phase_numpy` assembles it.
    The baseline quantity is the difference ``path_p - path_q``, and the fringe
    model is ``exp(-2 pi i (path_p - path_q) / lam)`` averaged over the fine axis
    (:func:`near_field_fringe_model`).

    ``taus_s`` moves the **satellite only**: the orbit is propagated to
    ``t + offset + tau`` while the antennas, the sidereal angle and ``w`` stay at
    ``t + offset``. That is what an along-track TLE error is. Shifting the whole
    geometry instead turns the Earth under the phase tracking as well, which
    moves the path difference by a tenth of a wavelength at metre wavelengths --
    a different model, and the wrong one.

    Everything here is numpy/f64 on the host, and stays f64 whatever ``--x64``
    says: an absolute path is hundreds of kilometres, which f32 resolves to tens
    of metres, and the model needs a small fraction of a wavelength. The
    *difference* handed to the jitted core is at most the array's diameter, which
    f32 does hold to well under a wavelength, so the cast happens there and not
    before.

    The antennas' fine-grid positions are computed once and shared by every
    ``tau``; only the orbit is re-propagated, in a single vectorised call over
    the flattened ``(n_tau, n_time, n_fine)`` grid.

    Sizing: the result is ``n_tau * n_bl * n_time * n_fine`` float64. For the
    MWA case (33 offsets, ~1000 coherent baselines, 27 frames, 40 sub-steps)
    that is ~290 MB, which is why the coherence cut is applied to the baseline
    list *before* the paths are built rather than after.

    Parameters
    ----------
    record : dict
        One orbit record -- TLE or OMM, as resolved by :mod:`tabascal.orbit`.
    ants_itrf : Array (n_ant, 3)
        Antenna positions in the ITRF (ECEF) frame, in metres.
    times_jd : Array (n_time,)
        Integration **centres**, as UTC Julian dates.
    phase_centre : dict
        ``{"ra": <deg>, "dec": <deg>}`` phase centre of the visibilities.
    a1, a2 : Array (n_bl,)
        Antenna indices of each baseline.
    n_fine : int
        Sub-steps per integration (see :func:`fine_time_offsets`).
    delta_t : float
        Integration time in seconds.
    taus_s : float or Array (n_tau,), default 0.0
        Along-track offsets to evaluate. A scalar still returns a grid of one,
        so the core is written once, for a grid.

    Returns
    -------
    Array (n_tau, n_bl, n_time, n_fine) float64
        Path difference ``path_p - path_q`` in metres.
    """
    taus = np.atleast_1d(np.asarray(taus_s, dtype=np.float64))
    ants_itrf = np.asarray(ants_itrf, dtype=np.float64)
    times_jd = np.asarray(times_jd, dtype=np.float64)
    a1 = np.asarray(a1)
    a2 = np.asarray(a2)
    n_fine = int(n_fine)
    n_time = len(times_jd)

    offsets = fine_time_offsets(n_fine, delta_t)
    t_fine = (times_jd[:, None] + offsets[None, :] / 86400.0).ravel()

    # The array, once. It does not move with tau.
    gh0 = (gast_deg(t_fine) - phase_centre["ra"]) % 360
    w = np.transpose(
        itrf_to_uvw_numpy(ants_itrf, gh0, phase_centre["dec"]), axes=(1, 0, 2)
    )[..., -1]  # (n_ant, n_time * n_fine)
    ants_xyz = itrs_to_gcrs_sf(ants_itrf, t_fine)  # (n_ant, n_time * n_fine, 3)

    # The orbit, once for the whole grid: propagation dominates the cost, and a
    # call per offset pays skyfield's per-call overhead n_tau times over.
    grid = (t_fine[None, :] + taus[:, None] / 86400.0).ravel()
    sat_xyz = np.asarray(get_satellite_positions([record], grid))[0].reshape(
        len(taus), len(t_fine), 3
    )

    out = np.empty((len(taus), len(a1), n_time, n_fine), dtype=np.float64)
    for i in range(len(taus)):
        # Looped rather than broadcast over tau: the (n_ant, n_time * n_fine, 3)
        # difference is the peak of this function and there is no reason to hold
        # n_tau of them at once.
        path = np.linalg.norm(ants_xyz - sat_xyz[i][None], axis=-1) + w
        out[i] = (path[a1] - path[a2]).reshape(len(a1), n_time, n_fine)

    return out


def satellite_range_and_speed(record, ants_itrf: NDArray, times_jd: NDArray):
    """Slant range and line-of-sight-crossing speed, at mid-observation.

    The two numbers :func:`coherent_baseline_mask` needs, measured from the array
    centre by finite difference over a second of the pass. Both are taken in the
    **Earth-fixed** frame, which is the one the array sits still in: what fringes
    a baseline is the rate at which the direction to the satellite sweeps across
    the antennas, and a geostationary emitter -- which hangs motionless over the
    array -- moves at three kilometres a second in the inertial frame.

    That emitter does not fringe *nothing*, though. On phase-tracked
    visibilities the ``w`` term keeps turning at the sidereal rate whatever the
    satellite does, worth a fringe of order ``Omega_e b / lam`` -- about 0.03 Hz
    on 600 m at 1.7 m, negligible beside a LEO's but not zero. This function
    returns the satellite's own transverse speed; the term the phase tracking
    adds is bounded by ``Omega_e * range_m`` and :func:`fit_time_offset` adds it
    where the fringe-rate ceiling is sized.

    The Earth-fixed direction is recovered by turning the inertial separation
    back by the sidereal angle. That leaves the precession-nutation rotation in
    it, which is constant to a part in ``1e12`` over the second differenced here
    and so cannot affect a *rate*; against the full frame transform the speed
    agrees to a part in ``1e4``.

    It is not the orbital speed. Near the horizon most of a LEO satellite's
    motion is along the line of sight and does not fringe, and its range is
    several times the overhead one -- both of which lengthen the coherence
    ceiling rather than shorten it.

    Parameters
    ----------
    record : dict
        One orbit record.
    ants_itrf : Array (n_ant, 3)
        Antenna positions in the ITRF frame, in metres; the mean is the site.
    times_jd : Array (n_time,)
        Observation times as UTC Julian dates. The middle one is used: the range
        varies by tens of percent over a pass, far less than the uncertainty on
        the orbit error the cut is set by.

    Returns
    -------
    (float, float)
        Slant range in metres, and speed across the line of sight in m/s.
    """
    times_jd = np.asarray(times_jd, dtype=np.float64)
    centre = np.mean(np.asarray(ants_itrf, dtype=np.float64), axis=0)

    step = 0.5 / 86400.0
    t_mid = float(np.mean(times_jd))
    t3 = np.array([t_mid - step, t_mid, t_mid + step])

    sep = (
        np.asarray(get_satellite_positions([record], t3))[0]
        - itrs_to_gcrs_sf(centre[None], t3)[0]
    )  # (3, 3), inertial

    theta = np.deg2rad(gast_deg(t3))
    cos, sin = np.cos(theta), np.sin(theta)
    sep = np.stack(
        [cos * sep[:, 0] + sin * sep[:, 1], -sin * sep[:, 0] + cos * sep[:, 1], sep[:, 2]],
        axis=-1,
    )

    ranges = np.linalg.norm(sep, axis=-1)
    look = sep / ranges[:, None]
    range_m = float(ranges[1])

    return range_m, float(np.linalg.norm(look[2] - look[0]) * range_m)


# ---------------------------------------------------------------------------
# The tau scan core
# ---------------------------------------------------------------------------

def near_field_fringe_model(paths: ArrayLike, freqs: ArrayLike) -> Array:
    """The per-integration fringe model: the template, averaged over its dump.

    ``mean_fine exp(-2 pi i path / lam)``. Each sub-step is a unit-modulus
    steering vector, so the average can only shrink: a baseline whose fringe
    turns a fraction of a cycle inside one integration keeps its modulus, and one
    that sweeps tens of cycles averages away to nothing. That loss is real -- it
    is in the data too -- and modelling it is what lets the normalised statistic
    of :func:`coherence_scores` stay a correlation.

    Pure ``jax.numpy`` and undecorated: the drivers own the ``jit``, and the
    batched search of #191 needs it traceable inside its own. Its working
    precision is the session's, which is why the *paths* are built in f64 on the
    host and only their (short) baseline differences arrive here.

    The intermediate is ``(..., n_bl, n_freq, n_time, n_fine)`` complex, which is
    the largest array in the scan; :func:`tau_scan` holds one offset's worth of
    it at a time.

    Parameters
    ----------
    paths : Array (..., n_bl, n_time, n_fine)
        Baseline path differences in metres (see
        :func:`near_field_baseline_paths`).
    freqs : Array (n_freq,)
        Channel frequencies in Hz.

    Returns
    -------
    Array (..., n_bl, n_freq, n_time) complex
        Fringe model, of modulus at most 1.
    """
    paths = jnp.asarray(paths)
    lam = C / jnp.asarray(freqs)

    return jnp.exp(
        -2j * jnp.pi * paths[..., None, :, :] / lam[:, None, None]
    ).mean(axis=-1)


def matched_filter_sums(vis: ArrayLike, model: ArrayLike, weights: ArrayLike):
    """The three weighted inner products the coherence statistic is built from.

    ``z = sum_bl w V conj(M)``, ``n1 = sum_bl w |V|^2``, ``n2 = sum_bl w |M|^2``,
    each summed over the baseline axis alone so the result is per channel and per
    frame: the satellite is coherent *within* an integration and the frames are
    combined afterwards, incoherently, by :func:`coherence_scores`.

    A plain weighted sum, with no flag handling of its own: a zero weight removes
    a baseline exactly, and whether a flagged sample may be ``nan`` is the
    driver's problem (:func:`fit_time_offset` zeroes those before the sums, since
    ``0 * nan`` is ``nan`` and would poison the whole cell).

    Parameters
    ----------
    vis : Array (n_bl, n_freq, n_time) complex
        Visibilities.
    model : Array (n_bl, n_freq, n_time) complex
        Fringe model on the same grid.
    weights : Array
        Anything broadcastable onto ``(n_bl, n_freq, n_time)`` -- a per-baseline
        ``(n_bl, 1, 1)`` is the common case.

    Returns
    -------
    (Array, Array, Array)
        ``(z, n1, n2)``, each ``(n_freq, n_time)``; ``z`` complex, the others
        real.
    """
    vis = jnp.asarray(vis)
    model = jnp.asarray(model)
    weights = jnp.asarray(weights)

    z = jnp.sum(weights * vis * jnp.conjugate(model), axis=0)
    n1 = jnp.sum(weights * jnp.abs(vis) ** 2, axis=0)
    n2 = jnp.sum(weights * jnp.abs(model) ** 2, axis=0)

    return z, n1, n2


def coherence_scores(z: ArrayLike, n1: ArrayLike, n2: ArrayLike, frame_mask=None):
    """The per-frame correlation and the per-channel score it combines into.

    ``r = |z| / sqrt(n1 n2)`` is a normalised correlation in ``[0, 1]``: the
    intra-dump smearing that shrinks ``|M|`` on the longer baselines divides out
    of it, so a frame's ``r`` says how well the data match the trajectory and not
    how bright the fringe was. ``z2 = sum_frames |z|^2 / (n1 n2)`` combines the
    frames incoherently -- the satellite's own phase is not modelled between
    integrations -- and is therefore bounded by the number of frames in view,
    which is what makes it comparable against a null.

    A cell nothing was measured in has ``n1 n2 = 0`` and contributes exactly
    zero, with ``r = 0`` there. Guarded with ``where`` rather than an epsilon in
    the denominator: an epsilon shifts every other cell's value to spare this
    one.

    Parameters
    ----------
    z, n1, n2 : Array (n_freq, n_time)
        The sums from :func:`matched_filter_sums`.
    frame_mask : Array (n_time,) bool or 0/1, optional
        Frames to combine. ``None`` combines all of them. It exists for the
        batched search of #191, whose shapes must stay static and so cannot slice
        the in-view window out; masking with zeros and slicing give the same
        ``z2``, so the two paths report the same detection.

    Returns
    -------
    (Array, Array)
        ``r`` ``(n_freq, n_time)`` and ``z2`` ``(n_freq,)``.
    """
    z = jnp.asarray(z)
    den = jnp.asarray(n1) * jnp.asarray(n2)
    measured = den > 0
    safe = jnp.where(measured, den, 1.0)

    r = jnp.where(measured, jnp.abs(z) / jnp.sqrt(safe), 0.0)
    per_frame = jnp.where(measured, jnp.abs(z) ** 2 / safe, 0.0)

    if frame_mask is None:
        return r, jnp.sum(per_frame, axis=-1)

    mask = jnp.asarray(frame_mask).astype(per_frame.dtype)

    return r, jnp.sum(mask * per_frame, axis=-1)


def tau_scan(
    vis: ArrayLike,
    weights: ArrayLike,
    paths: ArrayLike,
    freqs: ArrayLike,
    frame_mask=None,
    ant_offsets=None,
    a1=None,
    a2=None,
):
    """Score every along-track offset on the grid: the scan itself.

    One program for the whole grid. The offset axis is walked with
    ``jax.lax.map``, not a Python loop over jitted kernels: a loop re-enters the
    compiler per step and gives up the point of the design, which is one
    compilation and then a device-resident sweep. ``lax.map`` rather than a
    ``vmap`` over the grid for the same reason the paths are looped on the host
    -- the per-offset ``(n_bl, n_freq, n_time, n_fine)`` model is the biggest
    array in the calculation and only one of them need exist at a time.

    The function is pure and of fixed-shape arrays, so it ``vmap``\\ s over a
    leading candidate axis of ``paths``; that is the contract the multi-satellite
    search of #191 is built on.

    Parameters
    ----------
    vis : Array (n_bl, n_freq, n_time) complex
        Visibilities, already zeroed wherever their weight is.
    weights : Array
        Broadcastable onto ``vis``; see :func:`matched_filter_sums`.
    paths : Array (n_tau, n_bl, n_time, n_fine)
        Baseline path differences per offset (:func:`near_field_baseline_paths`).
    freqs : Array (n_freq,)
        Channel frequencies in Hz.
    frame_mask : Array (n_time,), optional
        See :func:`coherence_scores`.
    ant_offsets : Array (n_ant,), optional
        Per-antenna path offsets in metres, added as
        ``paths + off[a1] - off[a2]``. This is the decohered null's hook
        (:func:`decohered_null`); ``a1`` and ``a2`` are then required.
    a1, a2 : Array (n_bl,), optional
        Antenna indices, needed only with ``ant_offsets``.

    Returns
    -------
    dict
        ``z2`` ``(n_tau, n_freq)`` and ``r`` ``(n_tau, n_freq, n_time)``. ``r``
        keeps every frame, masked or not: it says what each frame did, and the
        mask decides only which are combined.
    """
    vis = jnp.asarray(vis)
    weights = jnp.asarray(weights)
    paths = jnp.asarray(paths)
    freqs = jnp.asarray(freqs)

    if ant_offsets is not None:
        if a1 is None or a2 is None:
            raise ValueError(
                "ant_offsets needs a1 and a2: a per-antenna offset only reaches "
                "a baseline through the pair it belongs to."
            )
        offsets = jnp.asarray(ant_offsets)
        delta = offsets[jnp.asarray(a1)] - offsets[jnp.asarray(a2)]
        paths = paths + delta[:, None, None]

    def one_offset(path):
        model = near_field_fringe_model(path, freqs)
        r, z2 = coherence_scores(*matched_filter_sums(vis, model, weights), frame_mask)

        return {"z2": z2, "r": r}

    return jax.lax.map(one_offset, paths)


def _null_draw_scores(vis, weights, paths_best, freqs, offsets, a1, a2, frame_mask):
    """``max``-over-channel ``z2`` for each row of ``offsets``."""

    def one_draw(offset):
        scan = tau_scan(
            vis, weights, paths_best[None], freqs,
            frame_mask=frame_mask, ant_offsets=offset, a1=a1, a2=a2,
        )

        return jnp.max(scan["z2"][0])

    return jax.lax.map(one_draw, offsets)


#: The core, compiled once per shape. Held at module level so a scan repeated
#: over satellites -- or over the candidates of #191 -- pays for compilation
#: once rather than per call.
_tau_scan_jit = jax.jit(tau_scan)
_null_draw_scores_jit = jax.jit(_null_draw_scores)


def decohered_null(
    vis: ArrayLike,
    weights: ArrayLike,
    paths_best: ArrayLike,
    freqs: ArrayLike,
    a1: ArrayLike,
    a2: ArrayLike,
    frame_mask=None,
    n_draws: int = 200,
    jitter_m: float = 50.0,
    seed: int = 0,
) -> Array:
    """What the statistic scores when the geometry is not real.

    The same sums at the same offset, with each antenna's path pushed by an
    independent ``U(0, jitter_m)``. Tens of metres are tens of wavelengths at
    metre wavelengths, so every baseline enters with an unrelated phase and the
    coherent sum collapses to an incoherent one -- which is what "no satellite
    on that trajectory" looks like, measured on *these* data, with their own
    weights, flagging, baseline set and residual sky. That is why the null is
    drawn rather than taken from a chi-squared: nothing about the real
    distribution of ``z2`` here is analytic.

    Drawn with ``jax.random`` from ``PRNGKey(seed)``, so a significance is
    reproducible; walked with ``lax.map`` rather than ``vmap`` because a batched
    draw would hold ``n_draws`` copies of the fringe model at once.

    Parameters
    ----------
    vis, weights, freqs : Array
        As :func:`tau_scan`.
    paths_best : Array (n_bl, n_time, n_fine)
        Path differences at the best offset.
    a1, a2 : Array (n_bl,)
        Antenna indices; their maximum sizes the offset vector.
    frame_mask : Array (n_time,), optional
        See :func:`coherence_scores`.
    n_draws : int, default 200
        Scrambles to draw.
    jitter_m : float, default 50.0
        Upper end of the per-antenna offset, in metres.
    seed : int, default 0
        PRNG seed.

    Returns
    -------
    Array (n_draws,)
        ``max``-over-channel ``z2`` for each draw.
    """
    a1 = np.asarray(a1)
    a2 = np.asarray(a2)
    n_ant = int(max(a1.max(initial=0), a2.max(initial=0))) + 1

    offsets = jax.random.uniform(
        jax.random.PRNGKey(int(seed)),
        (int(n_draws), n_ant),
        minval=0.0,
        maxval=float(jitter_m),
    )

    return _null_draw_scores_jit(
        vis, weights, jnp.asarray(paths_best), freqs, offsets, a1, a2, frame_mask
    )


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
# Fitting the along-track offset
# ---------------------------------------------------------------------------

def _no_offset_fit(taus, elevation, frames, n_freq, n_time, n_fine,
                   sigma_transverse_m, n_null, times_jd) -> dict:
    """The result shape for a pass there was nothing to measure.

    A satellite that never rose -- or one whose every baseline the coherence cut
    dropped -- is an answer, not a failure: the batched search will meet plenty
    of them and must not stop on one. ``tau_best`` and the significance are
    ``nan`` because nothing was measured, ``z2_best`` is 0 because that is what
    the sum over no frame comes to, and ``r_best`` is ``nan`` throughout rather
    than 0, since a zero correlation is a measurement.
    """
    return dict(
        tau_grid=taus,
        z2_tau=np.zeros((len(taus), n_freq)),
        tau_best=float("nan"),
        z2_best=0.0,
        best_chan=-1,
        best_freq=float("nan"),
        r_best=np.full((n_freq, n_time), np.nan),
        frames=frames,
        elevation=elevation,
        null=np.zeros(int(n_null)),
        null_mean=float("nan"),
        null_std=float("nan"),
        significance=float("nan"),
        n_bl_used=0,
        range_m=float("nan"),
        v_perp_m_s=float("nan"),
        b_coh=float("nan"),
        n_fine=int(n_fine),
        sigma_transverse_m=float(sigma_transverse_m),
        times_sec=(np.asarray(times_jd, dtype=np.float64) - times_jd[0]) * 86400.0,
    )


def fit_time_offset(
    vis: NDArray,
    record,
    ants_itrf: NDArray,
    times_jd: NDArray,
    phase_centre: dict,
    freqs: NDArray,
    a1: NDArray,
    a2: NDArray,
    int_time: float,
    noise=None,
    flags: Optional[NDArray] = None,
    taus_s=None,
    n_fine: int = 40,
    sigma_transverse_m: float = 300.0,
    soft_weights: bool = False,
    min_elevation: Optional[float] = 0.0,
    exclude_autos: bool = True,
    n_null: int = 200,
    null_jitter_m: float = 50.0,
    seed: int = 0,
) -> dict:
    """Measure one satellite's along-track time offset from the visibilities.

    The single-satellite search of GitHub #190, end to end: window the
    observation to the frames the satellite is up for, choose the baselines the
    orbit is accurate enough to beam-form with, score every offset on the grid,
    and calibrate the best score against a decohered-antenna null.

    The statistic, per offset, channel and frame, is the normalised correlation
    of :func:`coherence_scores` over the coherent baselines; frames are combined
    incoherently into ``z2`` per channel and the best cell is the largest of
    those over ``(tau, channel)``. Its significance is
    ``(z2_best - null_mean) / null_std``.

    **Two caveats on that significance, both deliberate.**

    It carries no trials factor. The scan maximises over the whole offset grid
    and every channel, while the null is drawn at the best offset and maximises
    over channels only -- so the number is biased high, and grows with the size
    of the grid it searched. The default threshold of 5 sigma
    (:func:`is_detection`) is calibrated against the MWA Cen A case on the
    default grid, and is a working cut rather than a false-alarm probability.

    And the step has to resolve the peak. Its half-width scales like
    ``lam r / (2 b_coh v_perp)`` -- about 0.1 s for a 600 m coherent array at
    567 km -- so an offset grid coarser than that steps over the detection. The
    0.25 s default matched the MWA curve, which decays over about +-2 s because
    the sum there is dominated by the shortest baselines; a longer coherent
    array needs a finer step, not a wider grid.

    Parameters
    ----------
    vis : Array (n_bl, n_freq, n_time) complex
        Visibilities to search.
    record : dict
        One orbit record.
    ants_itrf : Array (n_ant, 3)
        Antenna positions in the ITRF frame, in metres.
    times_jd : Array (n_time,)
        Integration centres as UTC Julian dates.
    phase_centre : dict
        ``{"ra": <deg>, "dec": <deg>}``.
    freqs : Array (n_freq,)
        Channel frequencies in Hz.
    a1, a2 : Array (n_bl,)
        Antenna indices of each baseline.
    int_time : float
        Integration (dump) time in seconds.
    noise : float or Array, optional
        Per-component noise standard deviation, in any of the shapes
        :func:`matched_filter_light_curves` accepts. ``None`` weights every
        baseline equally.
    flags : Array, optional
        ``True`` marks samples to exclude. Flagged visibilities are zeroed
        before the sums, because an MS carries ``inf`` and ``nan`` in them and
        ``0 * nan`` is ``nan``.
    taus_s : Array (n_tau,), optional
        Offsets to scan, in seconds. Defaults to :data:`DEFAULT_TAU_GRID`.
    n_fine : int, default 40
        Sub-steps per integration in the fringe model.
    sigma_transverse_m : float, default 300.0
        Transverse orbit error the coherence cut is sized by, in metres. Around
        the middle of what a Starlink TLE carries.
    soft_weights : bool, default False
        Taper the baselines inside the cut with a Gaussian on the coherence
        length instead of weighting them all equally. The set summed over is the
        hard cut either way (:func:`_coherence_weights`), so ``n_bl_used`` does
        not depend on this; what changes is how much the marginal baselines are
        allowed to say.
    min_elevation : float, optional, default 0.0
        Elevation in degrees below which the satellite is not searched for,
        inclusive, as ``rfi.min_elevation`` is. ``None`` uses every frame. The
        in-view window is *sliced* out before the scan, so nothing below the
        horizon costs anything; the core's ``frame_mask`` is the equivalent for
        callers whose shapes must stay static.
    exclude_autos : bool, default True
        Drop autocorrelations. They carry no path difference and so no fringe.
    n_null : int, default 200
        Draws in the decohered null.
    null_jitter_m : float, default 50.0
        Per-antenna scramble in the null, in metres.
    seed : int, default 0
        PRNG seed for the null.

    Returns
    -------
    dict
        ``tau_grid`` ``(n_tau,)``, ``z2_tau`` ``(n_tau, n_freq)``, ``tau_best``,
        ``z2_best``, ``best_chan``, ``best_freq``, ``r_best``
        ``(n_freq, n_time)`` on the **full** time axis with ``nan`` out of view,
        ``frames`` ``(n_time,)`` bool, ``elevation`` ``(n_time,)`` deg,
        ``times_sec`` ``(n_time,)``, ``null`` ``(n_null,)``, ``null_mean``,
        ``null_std``, ``significance``, ``n_bl_used``, ``range_m``,
        ``v_perp_m_s``, ``b_coh``, ``n_fine`` and ``sigma_transverse_m``.

        The *decision* is not among them: it needs a threshold, and a dict
        carrying one would have to guess what the caller means by a detection.
        See :func:`is_detection`.
    """
    vis = np.asarray(vis)
    ants_itrf = np.asarray(ants_itrf, dtype=np.float64)
    times_jd = np.asarray(times_jd, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    a1 = np.asarray(a1)
    a2 = np.asarray(a2)
    n_freq, n_time = vis.shape[1], vis.shape[2]

    taus = (
        DEFAULT_TAU_GRID.copy()
        if taus_s is None
        else np.atleast_1d(np.asarray(taus_s, dtype=np.float64))
    )

    elevation = np.asarray(
        get_satellite_elevations([record], times_jd, ants_itrf)
    )[0]
    frames = (
        np.ones(n_time, dtype=bool)
        if min_elevation is None
        else elevation >= min_elevation
    )
    empty = dict(
        taus=taus, elevation=elevation, frames=frames, n_freq=n_freq, n_time=n_time,
        n_fine=n_fine, sigma_transverse_m=sigma_transverse_m, n_null=n_null,
        times_jd=times_jd,
    )
    name = record.get("NORAD_CAT_ID", "?")

    if not frames.any():
        print(
            f"Warning: satellite {name} never reaches {min_elevation} degrees "
            "elevation over this observation, so there is no along-track offset "
            "to measure and no detection to report."
        )
        return _no_offset_fit(**empty)

    times_w = times_jd[frames]
    vis_w = vis[:, :, frames]

    # Weights: the noise the MS reports, zeroed on the flags and the
    # autocorrelations, tapered (or cut) by the coherence of each baseline.
    weights = _weight_source(noise, vis.shape)
    weights = weights[:, :, frames] if weights.shape[2] > 1 else weights
    weights = np.broadcast_to(weights, vis_w.shape).astype(np.float64)
    if flags is not None:
        flagged = np.broadcast_to(np.asarray(flags, dtype=bool), vis.shape)[:, :, frames]
        weights = np.where(flagged, 0.0, weights)

    non_auto = (a1 != a2) if exclude_autos else np.ones(len(a1), dtype=bool)
    bl_len = np.asarray(baseline_lengths(ants_itrf, a1, a2), dtype=np.float64)
    range_m, v_perp = satellite_range_and_speed(record, ants_itrf, times_w)
    # One frequency for the cut, at the middle of the band: b_coh scales as
    # lambda, so a band spanning a few percent moves it by a few percent, far
    # less than the order-of-magnitude uncertainty on the orbit error does.
    mean_freq = float(np.mean(freqs))
    # The phase tracking turns the w term at the sidereal rate whatever the
    # satellite does, so the fringe a baseline sees is not the satellite's
    # motion alone. That term is bounded by Omega_e * r -- some 41 m/s at
    # 570 km, 0.6 % of a LEO's transverse speed and the whole of a
    # geostationary emitter's -- and only the fringe-rate ceiling cares.
    v_fringe = v_perp + Omega_e * range_m
    support, coherence = _coherence_weights(
        bl_len, mean_freq, range_m, sigma_transverse_m, n_fine, int_time,
        v_fringe, soft_weights,
    )
    b_coh = float(
        jnp.minimum(
            tle_coherence_length(mean_freq, range_m, sigma_transverse_m),
            fringe_rate_coherence_length(
                mean_freq, range_m, n_fine, int_time, v_fringe
            ),
        )
    )

    # Baselines that contribute nothing are dropped rather than carried at zero
    # weight: the paths are the largest array in the scan and there is no point
    # building them for a baseline the cut has already answered for. The support
    # is the hard cut whether or not the weighting inside it is soft, for the
    # reason :func:`_coherence_weights` gives.
    used = np.flatnonzero(support & non_auto)
    n_bl_used = int(used.size)

    if n_bl_used == 0:
        print(
            f"Warning: no baseline is coherent for satellite {name} at "
            f"{sigma_transverse_m:g} m of transverse orbit error and a "
            f"{b_coh:.1f} m coherence length, so there is nothing to search."
        )
        return _no_offset_fit(**empty)

    vis_w = vis_w[used]
    weights = weights[used] * coherence[used][:, None, None]
    a1_used, a2_used = a1[used], a2[used]
    # A flagged visibility can be anything at all, and 0 * nan is nan.
    vis_w = np.where(weights > 0.0, vis_w, 0.0)

    paths = near_field_baseline_paths(
        record, ants_itrf, times_w, phase_centre, a1_used, a2_used, n_fine,
        int_time, taus_s=taus,
    )
    scan = _tau_scan_jit(vis_w, weights, paths, freqs)

    z2_tau = np.asarray(scan["z2"], dtype=np.float64)
    i_tau, i_chan = np.unravel_index(int(np.argmax(z2_tau)), z2_tau.shape)
    z2_best = float(z2_tau[i_tau, i_chan])

    # Back onto the full time axis, so the diagnostic shows a blank where the
    # satellite was not up rather than a dark band that reads as a measurement.
    r_best = np.full((n_freq, n_time), np.nan)
    r_best[:, frames] = np.asarray(scan["r"], dtype=np.float64)[i_tau]

    null = np.asarray(
        decohered_null(
            vis_w, weights, paths[i_tau], freqs, a1_used, a2_used,
            n_draws=n_null, jitter_m=null_jitter_m, seed=seed,
        ),
        dtype=np.float64,
    )
    null_mean, null_std = float(null.mean()), float(null.std())

    return dict(
        tau_grid=taus,
        z2_tau=z2_tau,
        tau_best=float(taus[i_tau]),
        z2_best=z2_best,
        best_chan=int(i_chan),
        best_freq=float(freqs[i_chan]),
        r_best=r_best,
        frames=frames,
        elevation=elevation,
        times_sec=(times_jd - times_jd[0]) * 86400.0,
        null=null,
        null_mean=null_mean,
        null_std=null_std,
        significance=(
            float((z2_best - null_mean) / null_std) if null_std > 0 else float("nan")
        ),
        n_bl_used=n_bl_used,
        range_m=range_m,
        v_perp_m_s=v_perp,
        b_coh=b_coh,
        n_fine=int(n_fine),
        sigma_transverse_m=float(sigma_transverse_m),
    )


def is_detection(fit: dict, threshold_sigma: float = 5.0) -> bool:
    """Whether a fit clears the null by ``threshold_sigma``.

    Kept out of :func:`fit_time_offset` on purpose: the fit measures, the caller
    decides. A pass that was never in view has a ``nan`` significance and is not
    a detection at any threshold.
    """
    significance = float(fit["significance"])

    return bool(np.isfinite(significance) and significance >= float(threshold_sigma))


def offset_fit_summary(norad_id, fit: dict, threshold_sigma: float = 5.0) -> str:
    """One line per satellite: what was measured, and whether it counts.

    ``DETECTED`` in upper case and ``not detected`` in lower, so a log can be
    grepped for the one without matching the other.
    """
    verdict = "DETECTED" if is_detection(fit, threshold_sigma) else "not detected"
    chan = fit["best_chan"]
    freq_mhz = float(fit["best_freq"]) / 1e6

    return (
        f"  {norad_id:<9} tau {fit['tau_best']:+6.2f} s  "
        f"chan {chan:>3} ({freq_mhz:9.4f} MHz)  "
        f"z2 {fit['z2_best']:.4f}  "
        f"null {fit['null_mean']:.4f} +/- {fit['null_std']:.4f}  "
        f"{fit['significance']:6.1f} sigma  {verdict}"
    )


def fit_time_offsets(
    orbit_records: list,
    norad_ids: list,
    vis: NDArray,
    ants_itrf: NDArray,
    times_jd: NDArray,
    phase_centre: dict,
    freqs: NDArray,
    a1: NDArray,
    a2: NDArray,
    int_time: float,
    threshold_sigma: float = 5.0,
    **kwargs,
) -> list:
    """Fit and report one along-track offset per satellite, in order.

    A loop over :func:`fit_time_offset` that prints
    :func:`offset_fit_summary` as each satellite is measured, so a long run says
    what it found while it is still running. Every other keyword goes straight
    to the fit.
    """
    fits = []
    for norad_id, record in zip(norad_ids, orbit_records):
        fit = fit_time_offset(
            vis, record, ants_itrf, times_jd, phase_centre, freqs, a1, a2,
            int_time, **kwargs,
        )
        print(offset_fit_summary(norad_id, fit, threshold_sigma))
        fits.append(fit)

    return fits


def _fitted_offsets(fits: list) -> list:
    """The offsets to extract curves at: the fitted one, or 0 where there is none."""
    return [
        float(fit["tau_best"]) if np.isfinite(fit["tau_best"]) else 0.0 for fit in fits
    ]


# ---------------------------------------------------------------------------
# Searching across candidate satellites
# ---------------------------------------------------------------------------

def _record_name(record, norad_id) -> str:
    """A satellite's catalogue name, or its ID where the record carries none.

    Every row of a ranking table has to be nameable, and a Space-Track export
    need not carry ``OBJECT_NAME`` -- nor need one record in a file where the
    others do, which is how a missing name arrives as a ``nan`` out of pandas.
    """
    name = record.get("OBJECT_NAME") if hasattr(record, "get") else None
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return str(int(norad_id))

    return str(name).strip() or str(int(norad_id))


def _slant_range(record, ants_itrf: NDArray, t_jd: float) -> float:
    """Distance from the array centre to the satellite at one instant, in metres.

    Taken as the separation of two positions expressed in the *same* (inertial)
    frame, which is what :func:`satellite_range_and_speed` differences. A
    distance is invariant to the frame both ends are written in, so this is the
    Earth-fixed slant range as well, without a second transform to get it.
    """
    centre = np.mean(np.asarray(ants_itrf, dtype=np.float64), axis=0)
    t = np.atleast_1d(np.float64(t_jd))
    sep = (
        np.asarray(get_satellite_positions([record], t))[0]
        - itrs_to_gcrs_sf(centre[None], t)[0]
    )

    return float(np.linalg.norm(sep[0]))


def enumerate_candidates(
    records: list,
    names: list,
    times_jd: NDArray,
    ants_itrf: NDArray,
    min_elevation: Optional[float] = 0.0,
) -> list:
    """Screen a snapshot down to the satellites worth scoring.

    Stage one of the identification search of GitHub #191, and the step that
    turns a constellation into a shortlist: on the MWA Cen A case it took 2551
    Starlink records to the 128 that were above the horizon during the 56 s
    observation. A satellite that never rose is not evidence, and scoring it
    would cost a scan.

    It also decides *which frames* each candidate is scored over. The mask is
    computed once, here, and travels with the candidate: the batched scan
    applies it inside the statistic rather than slicing, because its shapes have
    to stay static to ``vmap``.

    The screen is what keeps the shared baseline set of :func:`search_candidates`
    honest, too. The coherence ceiling grows with the slant range, so a
    satellite on the far side of the Earth -- 13 000 km away, through the ground
    -- would lift it to kilometres and readmit every long baseline. Dropping it
    here is what stops that; a search whose candidates are *all* below the
    horizon has no set to sum over and returns nothing.

    Elevations are evaluated once, at ``tau = 0``: the offsets searched for are
    seconds, which moves a LEO satellite tens of kilometres along its track and
    its elevation by a fraction of a degree. Nothing in the cut resolves that.

    Parameters
    ----------
    records : list of dict, length n_records
        Orbit records -- TLE or OMM, as resolved by :mod:`tabascal.orbit`. The
        record itself travels with the candidate rather than just its ID, since
        re-resolving later could pick up a different record for the same
        satellite.
    names : list of str, length n_records
        Catalogue names, aligned with ``records``.
    times_jd : Array (n_time,)
        Integration centres as UTC Julian dates.
    ants_itrf : Array (n_ant, 3)
        Antenna positions in the ITRF frame, in metres; the mean is the site.
    min_elevation : float, optional, default 0.0
        Elevation in degrees at or above which a satellite counts as up,
        inclusive, as ``rfi.min_elevation`` is. A satellite reaching it in at
        least one integration is kept. ``None`` keeps every record over every
        frame.

    Returns
    -------
    list of dict
        One per kept record, sorted by descending ``max_elevation``, each with
        ``norad_id``, ``name``, ``record``, ``max_elevation``, ``elevation``
        ``(n_time,)`` in degrees, ``frames`` ``(n_time,)`` bool, and ``range_m``
        -- the slant range at the frame of maximum elevation, i.e. the closest
        approach during the pass.

        ``range_m`` is for **reporting**: it says how near the satellite came,
        which is what a reader of a ranking table wants. It is not what sizes the
        coherence cut. That is the mid-window ``(range, speed)`` pair
        :func:`satellite_range_and_speed` returns for the in-view frames, one
        geometry taken at one instant, which is what :func:`fit_time_offset` uses
        and what :func:`search_candidates` reports in its per-candidate fits. The
        two are half a pass apart.
    """
    records = list(records)
    names = list(names)
    if len(records) != len(names):
        raise ValueError(
            f"{len(records)} records against {len(names)} names; every candidate "
            "has to be nameable, so the two lists are aligned."
        )
    if not records:
        return []

    times_jd = np.asarray(times_jd, dtype=np.float64)
    ants_itrf = np.asarray(ants_itrf, dtype=np.float64)
    elevations = np.asarray(get_satellite_elevations(records, times_jd, ants_itrf))

    candidates = []
    for record, name, elevation in zip(records, names, elevations):
        frames = (
            np.ones(len(times_jd), dtype=bool)
            if min_elevation is None
            else elevation >= min_elevation
        )
        if not frames.any():
            continue

        peak = int(np.argmax(elevation))
        candidates.append(
            dict(
                norad_id=int(record["NORAD_CAT_ID"]),
                name=str(name),
                record=record,
                max_elevation=float(elevation[peak]),
                elevation=elevation,
                frames=frames,
                range_m=_slant_range(record, ants_itrf, times_jd[peak]),
            )
        )

    return sorted(candidates, key=lambda candidate: -candidate["max_elevation"])


def candidates_from_orbit_dir(directory: str, times_jd: NDArray, name_filter=None):
    """Every satellite in a local snapshot directory, one record apiece.

    The ``--tle-dir`` source: a constellation export dropped in
    ``extra_orbit_dir`` format, read through the same per-ID nearest-epoch policy
    a run applies to ``extra_orbit_dir`` (:func:`tabascal.orbit._select_from_extra_dir`
    with no age ceiling) -- so a file carrying several epochs for one satellite
    contributes the record nearest the observation, rather than whichever row it
    happened to list first.

    That resolver prints a provenance line per satellite, which is the right
    thing for the handful a run configures and the wrong thing for a whole
    constellation, so only the lines reporting a *rejected* record are passed
    through. The ranking table is the output here, and it should not arrive
    under 2551 lines of bookkeeping.

    Parameters
    ----------
    directory : str
        Directory of orbit files (TLE or OMM), as ``--extra-orbit-dir`` takes.
    times_jd : Array (n_time,)
        Observation times as UTC Julian dates; their mean is the epoch records
        are chosen against.
    name_filter : str, optional
        Case-insensitive substring of ``OBJECT_NAME``, e.g. ``"STARLINK"``. A
        snapshot is usually a whole constellation plus whatever else the query
        dragged in. ``None`` keeps all. A record with no name is named by its
        ID, and so is kept only by a filter that matches the ID.

    Returns
    -------
    (list of dict, list of str, list of int)
        Records, names and NORAD IDs, in ascending ID order.
    """
    import contextlib
    import io

    from satchecker_client.cache import read_legacy_tle_records
    from tabascal.orbit import _select_from_extra_dir, observation_epoch_jd

    frame = read_legacy_tle_records(directory)
    if not len(frame):
        return [], [], []

    wanted = set()
    for value in np.asarray(frame["NORAD_CAT_ID"].values):
        try:
            wanted.add(int(value))
        except (TypeError, ValueError):
            continue
    if not wanted:
        return [], [], []

    provenance = io.StringIO()
    with contextlib.redirect_stdout(provenance):
        resolved, _ = _select_from_extra_dir(
            str(directory), wanted, observation_epoch_jd(times_jd), None
        )
    for line in provenance.getvalue().splitlines():
        if "rejected" in line:
            print(line)

    records, names, norad_ids = [], [], []
    for norad_id in sorted(resolved):
        record = dict(resolved[norad_id].record)
        name = _record_name(record, norad_id)
        if name_filter is not None and str(name_filter).lower() not in name.lower():
            continue
        records.append(record)
        names.append(name)
        norad_ids.append(int(norad_id))

    return records, names, norad_ids


def candidates_from_norad_ids(norad_ids, times_jd: NDArray, extra_orbit_dir=None):
    """Orbit records for an explicit candidate list, in the order it was asked.

    The other way into the search: ``-n``/``-np``, resolved through the run's own
    :func:`~tabascal.components.trajectory.fetch_orbital_elements`, so
    ``extra_orbit_dir``, the managed cache and SatChecker take exactly the
    precedence a run gives them and the search scores the records a run would
    model.

    Returns
    -------
    (list of dict, list of str, list of int)
        Records, names and NORAD IDs, in the requested order.
    """
    ids, records = _resolve_records(norad_ids, times_jd, extra_orbit_dir)
    names = [_record_name(record, nid) for nid, record in zip(ids, records)]

    return list(records), names, [int(n) for n in ids]


#: The batched core: one program over a leading candidate axis of the weights,
#: the paths and the frame mask, with the visibilities and the frequencies shared
#: across the batch. Held at module level and jitted once, so a sweep over a
#: constellation compiles a single time and then runs device-resident -- which is
#: the whole design, and why :func:`tau_scan` is pure and undecorated. It is that
#: same function: the search does not own a statistic of its own to disagree with
#: :func:`fit_time_offset` about.
_batched_tau_scan = jax.jit(jax.vmap(tau_scan, in_axes=(None, 0, 0, None, 0)))


def _candidate_coherence(
    candidate: dict,
    ants_itrf: NDArray,
    times_jd: NDArray,
    bl_len: NDArray,
    mean_freq: float,
    sigma_transverse_m: float,
    n_fine: int,
    int_time: float,
    soft_weights: bool,
):
    """One candidate's coherent baseline support, its weights and its geometry.

    Sized from a *single* geometry, taken at the middle of the candidate's
    in-view window -- the range and the crossing speed of one instant, from one
    :func:`satellite_range_and_speed` call, exactly as :func:`fit_time_offset`
    sizes a single satellite's, the phase-tracking term included. Mixing a range
    from one point of the pass with a speed from another gives a ceiling neither
    of them has, and would make the search disagree with the single-satellite fit
    about which baselines a satellite can be beam-formed over.

    The candidate's own ``range_m`` -- its closest approach -- is *not* used
    here. It is a reporting number, and half a pass away from this one.
    """
    frames = np.asarray(candidate["frames"], dtype=bool)
    range_m, v_perp = satellite_range_and_speed(
        candidate["record"], ants_itrf, times_jd[frames]
    )
    v_fringe = v_perp + Omega_e * range_m

    support, coherence = _coherence_weights(
        bl_len, mean_freq, range_m, sigma_transverse_m, n_fine, int_time,
        v_fringe, soft_weights,
    )
    b_coh = float(
        jnp.minimum(
            tle_coherence_length(mean_freq, range_m, sigma_transverse_m),
            fringe_rate_coherence_length(
                mean_freq, range_m, n_fine, int_time, v_fringe
            ),
        )
    )

    return support, coherence, b_coh, range_m, v_perp


def _empty_search(taus, n_freq, n_time, n_bl_used=0, b_coh_max=float("nan")) -> dict:
    """The result shape for a search with nothing to search.

    A name filter that matched nothing, an empty snapshot, a sky with nothing
    above the horizon in it, or a cut no baseline survives, is an answer about
    the observation rather than a failure; the caller decides whether to stop.
    Every key of a real result is here, so a caller can read the same fields
    either way -- all of them empty, and ``batch_size`` zero, because no batch
    was run.
    """
    return dict(
        table=[],
        candidates=[],
        norad_ids=np.zeros(0, dtype=int),
        z2_best=np.zeros(0),
        tau_best=np.zeros(0),
        best_chan=np.zeros(0, dtype=int),
        significance=np.zeros(0),
        z2_tau=np.zeros((0, len(taus), n_freq)),
        tau_grid=taus,
        frames=np.zeros((0, n_time), dtype=bool),
        n_bl_used=int(n_bl_used),
        b_coh_max=float(b_coh_max),
        batch_size=0,
        fits=[],
        median_z2=float("nan"),
    )


def _bytes_per_candidate(n_bl: int, n_freq: int, n_time: int, n_fine: int,
                         n_tau: int) -> int:
    """What one candidate of a batch costs while it is being scored, in bytes.

    Two arrays, and they are the two that grow with every dimension at once: the
    fringe model, ``(n_bl, n_freq, n_time, n_fine)`` complex, which ``lax.map``
    holds one offset at a time in whatever precision the scan is running in; and
    the path differences for the whole offset grid,
    ``(n_tau, n_bl, n_time, n_fine)`` float64, built on the host and cast on the
    device.

    Not counted: the weights (per baseline for a scalar sigma, the full
    ``(n_bl, n_freq, n_time)`` once anything is flagged), the visibilities, and
    whatever XLA keeps alive between the two. The estimate is a sizing heuristic
    for the sweep, not a cap on the function.
    """
    # The model's width follows the session's precision -- complex64 with x64
    # off, complex128 with it on -- so the budget has to be read in the precision
    # the scan is actually going to run in, not in the one it was written in.
    complex_bytes = np.dtype(jax.dtypes.canonicalize_dtype(np.complex128)).itemsize
    samples = int(n_bl) * int(n_time) * int(n_fine)

    return samples * int(n_freq) * complex_bytes + int(n_tau) * samples * 8


def _batch_for_memory(
    n_cand: int,
    n_bl: int,
    n_freq: int,
    n_time: int,
    n_fine: int,
    n_tau: int,
    batch_size: int,
    max_mem_gb,
) -> int:
    """Candidates per jitted call: what was asked for, or what will fit.

    ``batch_size`` is a wish; the observation decides. At the MWA scale the
    budget is what actually sets the batch: once candidates come near the horizon
    the coherent union reaches 7704 of the array's 9180 baselines
    (``b_coh_max`` 2880 m), one candidate over 24 channels is then some 2.1 GB,
    and a default batch of eight would ask for 17 GB -- the machine swaps long
    before the device is asked for anything.

    ``max_mem_gb = None`` is no budget at all, for a caller who has measured
    their own. At least one candidate is always scored: a budget below a single
    candidate has no smaller batch to fall back to, and refusing to run would
    leave the search unusable at exactly the scale it exists for.
    """
    n_batch = min(int(batch_size), int(n_cand))
    if max_mem_gb is None:
        return max(1, n_batch)

    per_candidate = max(
        _bytes_per_candidate(n_bl, n_freq, n_time, n_fine, n_tau), 1
    )
    affordable = int(float(max_mem_gb) * 1e9) // per_candidate

    return max(1, min(n_batch, affordable))


def search_candidates(
    vis: NDArray,
    candidates: list,
    ants_itrf: NDArray,
    times_jd: NDArray,
    phase_centre: dict,
    freqs: NDArray,
    a1: NDArray,
    a2: NDArray,
    int_time: float,
    noise=None,
    flags: Optional[NDArray] = None,
    taus_s=None,
    n_fine: int = 40,
    sigma_transverse_m: float = 300.0,
    soft_weights: bool = False,
    exclude_autos: bool = True,
    batch_size: int = 8,
    max_mem_gb: float = 4.0,
    n_null: int = 200,
    null_jitter_m: float = 50.0,
    n_null_candidates: int = 5,
    seed: int = 0,
    progress=None,
) -> dict:
    """Score every candidate satellite against the visibilities, and rank them.

    Stage two of GitHub #191: the tau scan of :func:`fit_time_offset`, run over a
    whole snapshot. It is the *same* statistic -- :func:`tau_scan` under
    ``jax.vmap``, one jitted program per batch -- because a search that
    re-derived the filter would be free to disagree with the single-satellite fit
    about what a detection is. The score per candidate is the largest ``z2`` over
    the offset grid and the channels: the emission is narrowband, so the maximum
    over channels is the right statistic and the winning channel is itself a
    deliverable, telling the user which channels to fit.

    **One baseline set, each candidate's own cut.** ``vmap`` needs static shapes,
    so the search sums over one baseline list: the **union** of the hard coherent
    sets of the candidates above the geometric horizon. A union rather than the
    farthest candidate's set, because the sets are not nested by range -- the
    fringe-rate ceiling depends on each candidate's own transverse speed, so a
    nearer, slower satellite can steer a baseline a farther, faster one cannot.
    Each candidate then applies *its own* coherence as a per-baseline weight
    inside the statistic, so another candidate's excess baselines enter at
    exactly zero and it is scored over precisely the baselines it could steer --
    the same numbers a search of that candidate alone would give. Only the
    above-horizon candidates size the union: a satellite 13 000 km away, through
    the Earth, tolerates kilometres of baseline and would otherwise readmit the
    long ones for everybody. With ``soft_weights`` the support is still the hard
    union and the taper weights the baselines inside it
    (:func:`_coherence_weights`).

    **The horizon lives inside the statistic.** Each candidate's in-view mask
    from :func:`enumerate_candidates` is passed to :func:`tau_scan` as
    ``frame_mask`` rather than slicing the arrays, so a satellite that rises or
    sets mid-observation contributes only its own frames while the batch stays
    rectangular. Masking and slicing give the same ``z2``, so the search and the
    single-satellite fit report the same detection for the same pass.

    **Sizing.** Two arrays dominate, per candidate of a batch: the fringe model,
    ``(n_bl, n_freq, n_time, n_fine)`` complex, held one offset at a time, and
    the paths for the whole grid, ``(n_tau, n_bl, n_time, n_fine)`` float64 on
    the host. ``max_mem_gb`` is a budget for their sum, and the batch actually
    run is the smaller of ``batch_size`` and what that budget affords -- reported
    back as ``batch_size``, since it is what the sweep did rather than what it
    was asked for. It is what sets the batch at real scale: on the MWA case the
    coherent union reaches 7704 of the array's 9180 baselines once candidates
    come near the horizon (``b_coh_max`` 2880 m), one candidate over 24 channels
    is then some 2.1 GB, and a batch of eight would ask for 17 GB. Not counted
    are the per-candidate weights, at whatever shape the noise and the flags
    resolve to -- per baseline for a scalar sigma, the full
    ``(n_bl, n_freq, n_time)`` once anything is flagged -- so this is a sizing
    heuristic and not a cap. A ragged last batch is padded by repeating its last
    candidate so every batch has one shape and the kernel compiles once; the
    padding rows are dropped.

    **The null is drawn for the top ``n_null_candidates`` only.** Two hundred
    extra scans per satellite over a whole constellation is the search twice
    over, spent on candidates nothing will be reported for. The rest are scored
    and ranked but carry no significance. The shortlist is taken on raw ``z2``,
    as the issue specifies -- and ``z2`` is a sum over in-view frames, so a short
    pass ranks low against a full one at the same per-frame correlation. That is
    a caveat on the shortlist, not a correction to make: comparing partial passes
    against full ones is what the score is for.

    Parameters
    ----------
    vis : Array (n_bl, n_freq, n_time) complex
        Visibilities to search.
    candidates : list of dict
        Screened candidates from :func:`enumerate_candidates`.
    ants_itrf : Array (n_ant, 3)
        Antenna positions in the ITRF frame, in metres.
    times_jd : Array (n_time,)
        Integration centres as UTC Julian dates.
    phase_centre : dict
        ``{"ra": <deg>, "dec": <deg>}``.
    freqs : Array (n_freq,)
        Channel frequencies in Hz.
    a1, a2 : Array (n_bl,)
        Antenna indices of each baseline.
    int_time : float
        Integration (dump) time in seconds.
    noise, flags, taus_s, n_fine, sigma_transverse_m, soft_weights, exclude_autos
        As :func:`fit_time_offset`, and meaning the same thing.
    n_null, null_jitter_m, seed
        The decohered null, as :func:`fit_time_offset` draws it.
    batch_size : int, default 8
        Most candidates to score per jitted call. An efficiency knob and nothing
        else: the answers do not depend on it.
    max_mem_gb : float or None, default 4.0
        Memory budget in gigabytes, which lowers the batch when ``batch_size``
        candidates would not fit. ``None`` is no budget, for a caller who has
        measured their own. See **Sizing** above for what it does and does not
        cover.
    n_null_candidates : int, default 5
        How many of the ranked candidates get a decohered null, and so a
        significance.
    progress : callable, optional
        Called ``(done, total)`` after each batch. A search over a constellation
        runs for minutes, and a command that says nothing for all of them cannot
        be told from a hung one.

    Returns
    -------
    dict
        ``table``, a list of per-candidate dicts in ranked order carrying
        ``rank``, ``norad_id``, ``name``, ``max_elevation``, ``range_m``,
        ``z2_best``, ``tau_best``, ``best_chan``, ``best_freq``, ``r_max``,
        ``significance``, ``null_mean``, ``null_std`` and ``n_frames``; the same
        columns as arrays under ``norad_ids``, ``z2_best``, ``tau_best``,
        ``best_chan`` and ``significance``; the scan curves ``z2_tau``
        ``(n_cand, n_tau, n_freq)`` on ``tau_grid``; ``frames``
        ``(n_cand, n_time)`` bool; ``n_bl_used`` and ``b_coh_max`` for the shared
        baseline set; ``batch_size``, the batch the sweep actually ran at;
        ``median_z2``; ``fits``, one dict per candidate shaped like
        :func:`fit_time_offset`'s, so :func:`plot_offset_diagnostics` and
        :func:`attach_offset_fits` take them unchanged; and ``candidates``, the
        screened candidates themselves in the same ranked order, which is where
        the orbit record of a named satellite comes from.

        A row's ``range_m`` is the candidate's closest approach, for reporting;
        the geometry the cut was sized from is the mid-window pair in
        ``fits[i]["range_m"]`` and ``fits[i]["v_perp_m_s"]``.

        The *decision* is not among them; see :func:`select_detections`.
    """
    vis = np.asarray(vis)
    ants_itrf = np.asarray(ants_itrf, dtype=np.float64)
    times_jd = np.asarray(times_jd, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    a1 = np.asarray(a1)
    a2 = np.asarray(a2)
    n_freq, n_time = vis.shape[1], vis.shape[2]

    taus = (
        DEFAULT_TAU_GRID.copy()
        if taus_s is None
        else np.atleast_1d(np.asarray(taus_s, dtype=np.float64))
    )
    candidates = list(candidates)
    n_cand = len(candidates)
    if n_cand == 0:
        return _empty_search(taus, n_freq, n_time)

    # A satellite below the geometric horizon cannot fringe the array at all, so
    # it does not get to size the baseline set: thirteen thousand kilometres away,
    # through the ground, its coherence length is kilometres and every baseline
    # the array has would be readmitted. With nothing above the horizon there is
    # no honest set to sum over, and so nothing to search.
    sizing = [i for i in range(n_cand) if candidates[i]["max_elevation"] >= 0.0]
    if not sizing:
        print(
            f"Warning: none of the {n_cand} candidates rises above the geometric "
            "horizon during this observation, so there is no baseline set any of "
            "them could be beam-formed over and nothing to search. Screening "
            "below the horizon gets a candidate past the elevation cut; it does "
            "not get one into the sum."
        )
        return _empty_search(taus, n_freq, n_time)

    # -- what each candidate can steer, and the set they are all scored over --
    non_auto = (a1 != a2) if exclude_autos else np.ones(len(a1), dtype=bool)
    bl_len = np.asarray(baseline_lengths(ants_itrf, a1, a2), dtype=np.float64)
    # One frequency for every cut, at the middle of the band, for the reason
    # fit_time_offset gives: b_coh scales as lambda and a band spanning a few
    # percent moves it far less than the orbit error does.
    mean_freq = float(np.mean(freqs))

    support, coherence, b_coh, range_mid, v_perp = [], [], [], [], []
    for candidate in candidates:
        keep, weights_bl, ceiling, distance, speed = _candidate_coherence(
            candidate, ants_itrf, times_jd, bl_len, mean_freq, sigma_transverse_m,
            n_fine, int_time, soft_weights,
        )
        support.append(keep & non_auto)
        coherence.append(weights_bl * non_auto)
        b_coh.append(ceiling)
        range_mid.append(distance)
        v_perp.append(speed)

    # The union of the hard supports, not the farthest candidate's: the sets are
    # not nested by range, since the fringe-rate ceiling turns on each
    # candidate's own transverse speed.
    union = np.zeros(len(bl_len), dtype=bool)
    for i in sizing:
        union |= support[i]
    used = np.flatnonzero(union)
    n_bl_used = int(used.size)
    b_coh_max = float(max(b_coh[i] for i in sizing))

    if n_bl_used == 0:
        # Returned here rather than scanned, as fit_time_offset returns its
        # no-fit result: a sum over no baseline is zero at every offset and
        # channel, so the argmax would pick the first cell of the grid and the
        # ranking would carry a tau and a channel nothing measured.
        print(
            f"Warning: no baseline is coherent for any candidate at "
            f"{sigma_transverse_m:g} m of transverse orbit error and a "
            f"{b_coh_max:.1f} m coherence length, so there is nothing to search."
        )
        return _empty_search(
            taus, n_freq, n_time, n_bl_used=0, b_coh_max=b_coh_max
        )

    # -- the weights and visibilities the whole batch shares --
    # Kept at whatever shape the noise resolves to rather than broadcast to the
    # visibilities': every candidate carries a copy, and a per-baseline sigma
    # need not become an (n_bl, n_freq, n_time) array to be used as one.
    weights = _weight_source(noise, vis.shape)
    weights = np.broadcast_to(weights, (len(a1),) + weights.shape[1:])[used]
    if flags is not None:
        flagged = np.broadcast_to(np.asarray(flags, dtype=bool), vis.shape)[used]
        if flagged.any():
            weights = np.where(flagged, 0.0, weights)
    # A flagged visibility can be anything at all, and 0 * nan is nan. Only the
    # shared weights are read here: a candidate's own coherence zeros multiply a
    # finite number, which is fine, and zeroing on those would need a copy of the
    # visibilities per candidate.
    vis_used = vis[used]
    vis_used = np.where(
        np.broadcast_to(weights, vis_used.shape) > 0.0, vis_used, 0.0
    )

    a1_used, a2_used = a1[used], a2[used]
    frames = np.stack([np.asarray(c["frames"], dtype=bool) for c in candidates])

    def candidate_weights(i):
        return weights * coherence[i][used][:, None, None]

    # -- the sweep --
    n_batch = _batch_for_memory(
        n_cand, n_bl_used, n_freq, n_time, n_fine, len(taus), batch_size,
        max_mem_gb,
    )
    z2_tau = np.empty((n_cand, len(taus), n_freq))
    r_best = np.full((n_cand, n_freq, n_time), np.nan)
    z2_best = np.empty(n_cand)
    tau_best = np.empty(n_cand)
    best_chan = np.empty(n_cand, dtype=int)
    r_max = np.full(n_cand, np.nan)

    paths = np.empty((n_batch, len(taus), n_bl_used, n_time, n_fine))
    for start in range(0, n_cand, n_batch):
        index = list(range(start, min(start + n_batch, n_cand)))
        for k, i in enumerate(index):
            paths[k] = near_field_baseline_paths(
                candidates[i]["record"], ants_itrf, times_jd, phase_centre,
                a1_used, a2_used, n_fine, int_time, taus_s=taus,
            )
        # A ragged last batch repeats its last candidate rather than shrinking:
        # a shorter batch is another shape and so another compilation, and the
        # padding rows are simply not read.
        padded = index + [index[-1]] * (n_batch - len(index))
        paths[len(index):] = paths[len(index) - 1]

        scan = _batched_tau_scan(
            vis_used,
            np.stack([candidate_weights(i) for i in padded]),
            paths,
            freqs,
            frames[padded].astype(np.float64),
        )

        z2 = np.asarray(scan["z2"], dtype=np.float64)
        r = np.asarray(scan["r"], dtype=np.float64)
        for k, i in enumerate(index):
            in_view = frames[i]
            i_tau, i_chan = np.unravel_index(int(np.argmax(z2[k])), z2[k].shape)
            z2_tau[i] = z2[k]
            z2_best[i] = z2[k][i_tau, i_chan]
            tau_best[i] = taus[i_tau]
            best_chan[i] = i_chan
            # Back onto the full time axis, saying nothing where nothing was
            # looked at: a zero correlation would read as a measurement.
            r_best[i][:, in_view] = r[k, i_tau][:, in_view]
            r_max[i] = np.max(r[k, i_tau, i_chan][in_view], initial=0.0)

        if progress is not None:
            progress(min(start + n_batch, n_cand), n_cand)

    # -- rank, then calibrate the top of the ranking against a null --
    order = np.argsort(-z2_best, kind="stable")
    null = [np.full(int(n_null), np.nan) for _ in range(n_cand)]
    null_mean = np.full(n_cand, np.nan)
    null_std = np.full(n_cand, np.nan)
    significance = np.full(n_cand, np.nan)

    for i in order[: max(0, int(n_null_candidates))]:
        paths_best = near_field_baseline_paths(
            candidates[i]["record"], ants_itrf, times_jd, phase_centre, a1_used,
            a2_used, n_fine, int_time, taus_s=tau_best[i],
        )[0]
        draws = np.asarray(
            decohered_null(
                vis_used, candidate_weights(i), paths_best, freqs, a1_used, a2_used,
                frame_mask=frames[i], n_draws=n_null, jitter_m=null_jitter_m,
                seed=seed,
            ),
            dtype=np.float64,
        )
        null[i] = draws
        null_mean[i] = float(draws.mean())
        null_std[i] = float(draws.std())
        if null_std[i] > 0:
            significance[i] = (z2_best[i] - null_mean[i]) / null_std[i]

    # -- the ranking, and a fit per candidate the diagnostics can read --
    times_sec = (times_jd - times_jd[0]) * 86400.0
    table, fits = [], []
    for rank, i in enumerate(order):
        candidate = candidates[i]
        table.append(
            dict(
                rank=rank,
                norad_id=int(candidate["norad_id"]),
                name=str(candidate["name"]),
                max_elevation=float(candidate["max_elevation"]),
                # The closest approach, which is what a ranking table wants to
                # report; the geometry the cut was sized from is in the fit.
                range_m=float(candidate["range_m"]),
                z2_best=float(z2_best[i]),
                tau_best=float(tau_best[i]),
                best_chan=int(best_chan[i]),
                best_freq=float(freqs[best_chan[i]]),
                r_max=float(r_max[i]),
                significance=float(significance[i]),
                null_mean=float(null_mean[i]),
                null_std=float(null_std[i]),
                n_frames=int(frames[i].sum()),
            )
        )
        fits.append(
            dict(
                tau_grid=taus,
                z2_tau=z2_tau[i],
                tau_best=float(tau_best[i]),
                z2_best=float(z2_best[i]),
                best_chan=int(best_chan[i]),
                best_freq=float(freqs[best_chan[i]]),
                r_best=r_best[i],
                frames=frames[i],
                elevation=np.asarray(candidate["elevation"], dtype=np.float64),
                times_sec=times_sec,
                null=null[i],
                null_mean=float(null_mean[i]),
                null_std=float(null_std[i]),
                significance=float(significance[i]),
                # The candidate's own count, which is what its score was summed
                # over; the search's shared set is n_bl_used at the top level.
                n_bl_used=int(support[i].sum()),
                # The mid-window geometry the cut was sized from -- what
                # fit_time_offset reports here, and not the closest approach the
                # ranking table carries.
                range_m=float(range_mid[i]),
                v_perp_m_s=float(v_perp[i]),
                b_coh=float(b_coh[i]),
                n_fine=int(n_fine),
                sigma_transverse_m=float(sigma_transverse_m),
            )
        )

    return dict(
        table=table,
        candidates=[candidates[i] for i in order],
        norad_ids=np.array([row["norad_id"] for row in table], dtype=int),
        z2_best=z2_best[order],
        tau_best=tau_best[order],
        best_chan=best_chan[order],
        significance=significance[order],
        z2_tau=z2_tau[order],
        tau_grid=taus,
        frames=frames[order],
        n_bl_used=n_bl_used,
        b_coh_max=b_coh_max,
        batch_size=int(n_batch),
        fits=fits,
        median_z2=float(np.median(z2_best)),
    )


def select_detections(
    search: dict, threshold_sigma: float = 5.0, runner_up_ratio: float = 1.5
) -> dict:
    """Decide which candidates are detections, and what to warn about.

    Ranking is not deciding. A detection is a candidate whose significance was
    *measured* and clears ``threshold_sigma``; one outside the null's shortlist
    carries ``nan``, and ``nan >= 5`` is false while ``nan < 5`` is false too --
    only one of those readings is safe to rely on. Multiple detections are simply
    all returned, in rank order: the fit accepts several satellites already.

    Two warnings, both from the issue and both about a ranking that cannot be
    read at face value:

    * a **close runner-up** -- the second candidate within ``runner_up_ratio`` of
      the first. Satellites in the same train partially match each other's
      fringes, so a winner that is not clear of the field is a result to look at
      twice rather than a satellite to name. It is a statement about the ranking
      and does not wait for a detection: two candidates level at the noise floor
      say the search could not separate them, which is worth knowing even when
      neither is named.
    * a **scan edge** -- a *detected* candidate whose best offset is the first or
      last point of the grid. The peak may be off the grid, so the offset is at
      least that large and the number reported is a floor; widening ``--tau-max``
      is the fix. Only for a detection: an undetected candidate's best offset is
      wherever noise happened to peak, and that it did so at the end of the grid
      says nothing about anything.

    Parameters
    ----------
    search : dict
        A result from :func:`search_candidates`.
    threshold_sigma : float, default 5.0
        Significance above the decohered null at which a candidate counts. It
        carries no trials factor; see :func:`fit_time_offset`.
    runner_up_ratio : float, default 1.5
        Warn when ``z2[1] >= z2[0] / runner_up_ratio``.

    Returns
    -------
    dict
        ``detected``, the qualifying rows of ``search["table"]`` in rank order,
        and ``warnings``, a list of sentences.
    """
    table = list(search["table"])
    threshold_sigma = float(threshold_sigma)
    detected = [
        row
        for row in table
        if np.isfinite(row["significance"]) and row["significance"] >= threshold_sigma
    ]

    warnings = []
    if len(table) > 1 and table[1]["z2_best"] >= table[0]["z2_best"] / float(
        runner_up_ratio
    ):
        warnings.append(
            f"close runner-up: {table[1]['norad_id']} ({table[1]['name']}) scores "
            f"{table[1]['z2_best']:.4f} against the winner "
            f"{table[0]['norad_id']} ({table[0]['name']}) at "
            f"{table[0]['z2_best']:.4f}, within {runner_up_ratio:g}x. Satellites "
            "in the same train partially match each other's fringes, so read the "
            "two scan curves before naming either."
        )

    taus = np.asarray(search["tau_grid"], dtype=np.float64)
    edges = (float(taus[0]), float(taus[-1]))
    for row in detected:
        if row["tau_best"] in edges:
            warnings.append(
                f"tau at the scan edge: {row['norad_id']} ({row['name']}) peaks "
                f"at tau = {row['tau_best']:+.3f} s, the end of the grid, so the "
                "offset is at least that large and the peak may lie past it. "
                "Widen --tau-max and run it again."
            )

    return {"detected": detected, "warnings": warnings}


# ---------------------------------------------------------------------------
# The search's artifacts
# ---------------------------------------------------------------------------

def write_search_results(
    path: str, search: dict, selection: dict, threshold_sigma: float
) -> str:
    """Save the ranking table, detections or not.

    "Nothing above the threshold" is a result about every satellite that was up,
    and the evidence for it is the table -- so this is written either way, and
    ``detected`` records which rows the selection named against the threshold it
    was named by.

    Returns
    -------
    str
        ``path``.
    """
    table = search["table"]
    detected = {row["rank"] for row in selection["detected"]}

    def column(key, dtype=np.float64):
        return np.array([row[key] for row in table], dtype=dtype)

    np.savez(
        path,
        norad_ids=column("norad_id", int),
        names=np.array([row["name"] for row in table]),
        z2_best=column("z2_best"),
        tau_best=column("tau_best"),
        best_chan=column("best_chan", int),
        best_freq=column("best_freq"),
        significance=column("significance"),
        null_mean=column("null_mean"),
        null_std=column("null_std"),
        max_elevation=column("max_elevation"),
        range_m=column("range_m"),
        r_max=column("r_max"),
        n_frames=column("n_frames", int),
        z2_tau=np.asarray(search["z2_tau"], dtype=np.float64),
        tau_grid=np.asarray(search["tau_grid"], dtype=np.float64),
        frames=np.asarray(search["frames"], dtype=bool),
        detected=np.array([row["rank"] in detected for row in table], dtype=bool),
        threshold_sigma=float(threshold_sigma),
        n_bl_used=int(search["n_bl_used"]),
        b_coh_max=float(search["b_coh_max"]),
        median_z2=float(search["median_z2"]),
    )

    return path


def write_config_fragment(path: str, selection: dict, shifted_orbit_dir=None) -> str:
    """Write the detections as a tabascal ``satellites`` section.

    The deliverable the issue asks for: the ``norad_ids`` list a run needs,
    produced from the data rather than known in advance, ready to merge into a
    config. Written even when nothing was detected -- an empty list with a line
    saying why is an artifact to point at, where a missing file leaves the reader
    to guess the run failed.

    A bare list of IDs is not auditable, so above it, per satellite and as YAML
    comments, is what it was detected on: the offset its curves must be extracted
    at, its score, its significance, and the channel -- which is what tells the
    user which channels to fit. Any warnings from :func:`select_detections` are
    written under them.

    With ``shifted_orbit_dir`` the section also points at the epoch-shifted
    records (:func:`write_shifted_orbits`) with no age ceiling, which is the
    whole point of writing them: a later run reproduces the trajectories the
    search measured, whatever SatChecker serves by then.

    Returns
    -------
    str
        ``path``.
    """
    import yaml

    detected = selection["detected"]
    lines = [
        "# tabascal satellite search: the satellites found in these visibilities.",
        "# Merge this section into a config, or point --extra-orbit-dir at the",
        "# shifted records it names.",
    ]
    if detected:
        lines.append("#")
        for row in detected:
            lines.append(
                f"#   {row['norad_id']}  {row['name']}  "
                f"tau {row['tau_best']:+.3f} s  z2 {row['z2_best']:.4f}  "
                f"{row['significance']:.2f} sigma  "
                f"chan {row['best_chan']} ({row['best_freq'] / 1e6:.4f} MHz)"
            )
    else:
        lines.append("#")
        lines.append(
            "#   No candidate cleared the detection threshold, so there is "
            "nothing to name."
        )
    for warning in selection["warnings"]:
        lines.append(f"# WARNING: {warning}")

    satellites = {"norad_ids": [int(row["norad_id"]) for row in detected]}
    if shifted_orbit_dir is not None:
        satellites["extra_orbit_dir"] = str(shifted_orbit_dir)
        satellites["extra_orbit_max_age_days"] = None

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
        # default_flow_style=None keeps the ID list on one line, as tabascal's
        # own configs write it, while the section itself stays in block style.
        fh.write(
            yaml.safe_dump(
                {"satellites": satellites}, sort_keys=False, default_flow_style=None
            )
        )

    return path


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def _lc_result(
    light_curves: NDArray,
    error: NDArray,
    norad_ids,
    freqs: NDArray,
    times_mjd_utc: NDArray,
    data_col: str,
    corr: str,
    in_view: Optional[NDArray] = None,
) -> dict:
    """Bundle an estimate with the coordinates it is only interpretable against.

    ``times_mjd_utc`` is UTC MJD, not the MS's ``TIME`` column as declared: it is
    what :func:`save_light_curves_npz` writes as the file's ``times``, and the
    interchange format states one scale so a curve can be read back against any
    measurement set covering the same pass. See :func:`tabascal.time.to_utc_mjd`.
    """
    times_mjd_utc = np.asarray(times_mjd_utc, dtype=np.float64)
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
        "times_mjd_utc": times_mjd_utc,
        "times_sec": (times_mjd_utc - times_mjd_utc[0]) * 86400.0,
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
    offset_fit: Optional[dict] = None,
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
        ``ants_itrf``, ``times_jd``, ``times_mjd``, ``time_scale``, ``freqs``,
        ``phase_centre``, ``a1``, ``a2``, ``orbit_records`` and ``norad_ids``.
    vis : Array (n_bl, n_freq, n_time), optional
        Visibilities to filter; defaults to ``tab_config.vis_obs``.
    exclude_autos : bool, default True
    max_mem_gb : float, default 1.0
    offset_fit : dict, optional
        Settings for the along-track offset search; see
        :func:`attach_offset_fits`. With it, each satellite's ``tau`` is measured
        first and the curves are extracted at the offset it found.

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
    ants_itrf = np.asarray(tab_config.ants_itrf)

    flags = getattr(tab_config, "flags", None)
    flags = None if flags is None else np.asarray(flags)

    fits = None
    if offset_fit is not None:
        fits = fit_time_offsets(
            records, norad_ids, np.asarray(vis), ants_itrf, times_jd,
            tab_config.phase_centre, np.asarray(tab_config.freqs),
            np.asarray(tab_config.a1), np.asarray(tab_config.a2),
            float(tab_config.int_time),
            noise=getattr(tab_config, "noise", None), flags=flags,
            min_elevation=getattr(tab_config, "min_elevation", None),
            exclude_autos=exclude_autos, **offset_fit,
        )

    # See _filter_visibilities: without a fit this is the call it has always been.
    offsets = {} if fits is None else {"time_offsets_s": _fitted_offsets(fits)}
    rfi_phase = rfi_phase_from_records(
        records,
        ants_itrf,
        times_jd,
        tab_config.phase_centre,
        np.asarray(tab_config.freqs),
        **offsets,
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

    result = _lc_result(
        light_curves,
        error,
        norad_ids,
        np.asarray(tab_config.freqs),
        to_utc_mjd(tab_config.times_mjd, tab_config.time_scale),
        tab_config.args["data"]["data_col"],
        tab_config.args["data"]["corr"],
        in_view=in_view,
    )

    if fits is None:
        return result

    result["orbit_records"] = records

    return attach_offset_fits(
        result, fits, offset_fit.get("threshold_sigma", 5.0)
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
    offset_fit: Optional[dict] = None,
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
    offset_fit : dict, optional
        Settings for the along-track offset search; see
        :func:`attach_offset_fits`. With it, each satellite's ``tau`` is measured
        first and the curves are extracted at the offset it found.

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
        offset_fit=offset_fit,
    )


def _filter_visibilities(
    vis, ms, records, norad_ids, times_jd, data_col, corr,
    exclude_autos, min_elevation, max_mem_gb, offset_fit=None,
):
    """Shared tail of the two MS-backed drivers."""
    phase_centre = {"ra": float(ms["ra"]), "dec": float(ms["dec"])}
    ants_itrf = np.asarray(ms["ants_itrf"])
    flags = None if ms.get("flags") is None else np.asarray(ms["flags"])

    fits = None
    if offset_fit is not None:
        fits = fit_time_offsets(
            records, norad_ids, vis, ants_itrf, times_jd, phase_centre,
            np.asarray(ms["freqs"]), np.asarray(ms["a1"]), np.asarray(ms["a2"]),
            float(ms["int_time"]), noise=ms.get("noise"), flags=flags,
            min_elevation=min_elevation, exclude_autos=exclude_autos,
            **offset_fit,
        )

    # The keyword is supplied only where an offset was actually measured, so the
    # unfitted path stays the call it has always been, argument for argument.
    offsets = {} if fits is None else {"time_offsets_s": _fitted_offsets(fits)}
    rfi_phase = rfi_phase_from_records(
        records, ants_itrf, times_jd, phase_centre, np.asarray(ms["freqs"]), **offsets
    )
    in_view = _elevation_mask(records, times_jd, ants_itrf, min_elevation)

    light_curves, error = matched_filter_light_curves(
        vis,
        rfi_phase,
        np.asarray(ms["a1"]),
        np.asarray(ms["a2"]),
        noise=_resolve_noise(ms.get("noise"), "the MS partition"),
        flags=flags,
        in_view=in_view,
        exclude_autos=exclude_autos,
        max_mem_gb=max_mem_gb,
    )

    result = _lc_result(
        light_curves,
        error,
        norad_ids,
        np.asarray(ms["freqs"]),
        to_utc_mjd(ms["times_mjd"], ms["time_scale"]),
        data_col,
        corr,
        in_view=in_view,
    )

    if fits is None:
        return result

    result["orbit_records"] = list(records)

    return attach_offset_fits(
        result, fits, offset_fit.get("threshold_sigma", 5.0)
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
    offset_fit: Optional[dict] = None,
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
        offset_fit=offset_fit,
    )


# ---------------------------------------------------------------------------
# Carrying the offset fit through a light-curve result
# ---------------------------------------------------------------------------

#: Per-source entries of a light-curve result, keyed on the first axis by the
#: source. :func:`select_sources` restricts every one of them together; anything
#: not listed here -- the frequencies, the time axis, the offset grid, the
#: threshold -- describes the observation rather than a source and is left alone.
_PER_SOURCE_KEYS = (
    "light_curves", "error", "z", "in_view", "norad_ids", "titles",
    "tau_best", "z2_tau", "z2_best", "best_chan", "significance",
    "null_mean", "null_std", "detected", "r_best", "offset_fits",
    "orbit_records",
)

#: Offset-fit arrays :func:`save_light_curves_npz` writes beside the curves.
#: ``r_best`` is not among them: it is transposed on the way out, like the
#: curves themselves.
_OFFSET_FIT_KEYS = (
    "tau_best", "tau_grid", "z2_tau", "z2_best", "best_chan", "significance",
    "null_mean", "null_std", "detected", "offset_threshold_sigma",
)


def attach_offset_fits(result: dict, fits: list, threshold_sigma: float) -> dict:
    """Add a per-satellite offset fit to a light-curve result.

    The output artifact has to record the offset the curves were measured at, or
    a later run cannot reproduce the trajectory that produced them -- so the
    scan's answers travel with the curves, stacked along the same source axis,
    and :func:`save_light_curves_npz` writes them into the ``.npz``.

    ``detected`` is decided here, once, against ``threshold_sigma``, and the
    threshold is recorded alongside it so a file says what it was judged by.

    Parameters
    ----------
    result : dict
        A light-curve result (see :func:`_lc_result`).
    fits : list of dict, length n_src
        Fits from :func:`fit_time_offset`, in the result's source order.
    threshold_sigma : float
        Significance above the null at which a fit counts as a detection.

    Returns
    -------
    dict
        A copy of ``result`` carrying ``tau_best`` ``(n_src,)``, ``tau_grid``
        ``(n_tau,)``, ``z2_tau`` ``(n_src, n_tau, n_freq)``, ``z2_best``,
        ``best_chan``, ``significance``, ``null_mean``, ``null_std``,
        ``detected`` ``(n_src,)`` bool, ``r_best`` ``(n_src, n_freq, n_time)``,
        ``offset_threshold_sigma``, and the fits themselves under
        ``offset_fits`` for the diagnostics to draw from.
    """
    out = dict(result)
    if not fits:
        return out

    out.update(
        tau_grid=np.asarray(fits[0]["tau_grid"], dtype=np.float64),
        tau_best=np.array([f["tau_best"] for f in fits], dtype=np.float64),
        z2_tau=np.stack([np.asarray(f["z2_tau"], dtype=np.float64) for f in fits]),
        z2_best=np.array([f["z2_best"] for f in fits], dtype=np.float64),
        best_chan=np.array([f["best_chan"] for f in fits], dtype=int),
        significance=np.array([f["significance"] for f in fits], dtype=np.float64),
        null_mean=np.array([f["null_mean"] for f in fits], dtype=np.float64),
        null_std=np.array([f["null_std"] for f in fits], dtype=np.float64),
        detected=np.array(
            [is_detection(f, threshold_sigma) for f in fits], dtype=bool
        ),
        r_best=np.stack([np.asarray(f["r_best"], dtype=np.float64) for f in fits]),
        offset_threshold_sigma=float(threshold_sigma),
        offset_fits=list(fits),
    )

    return out


def select_sources(result: dict, keep) -> dict:
    """A light-curve result restricted to some of its sources.

    Threshold-gated saving is a selection along the source axis, and it has to be
    made in one place: a per-source array left behind would mislabel every curve
    after the first one dropped.

    Parameters
    ----------
    result : dict
        A light-curve result, with or without an offset fit attached.
    keep : Array (n_src,) bool, or Array of int
        A mask or an index array over the sources, read the same way either way.

    Returns
    -------
    dict
        A copy carrying only the selected sources. The coordinates -- the
        frequencies, the times, the offset grid -- are not per source and come
        through untouched.
    """
    keep = np.asarray(keep)
    index = np.flatnonzero(keep) if keep.dtype == bool else keep.astype(int)

    out = dict(result)
    for key in _PER_SOURCE_KEYS:
        value = result.get(key)
        if value is None:
            continue
        out[key] = (
            value[index]
            if isinstance(value, np.ndarray)
            else [value[int(i)] for i in index]
        )

    return out


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_light_curves_npz(path: str, result: dict) -> None:
    """Save a light-curve result as the ``rfi.est`` interchange format.

    The four names :func:`tabascal.components.rfi_signal.read_light_curves`
    requires -- ``light_curves`` ``(n_src, n_time, n_freq)``, ``norad_ids``,
    ``times`` (**UTC** MJD) and ``freqs`` (Hz) -- so the output of a
    ``tabascal light-curve`` run can be pointed at with ``rfi.est`` unchanged.

    ``times`` is UTC and not the MS's ``TIME`` column as declared. The format
    states one scale so a curve stays interpretable away from the MS it was
    measured on, and the reader samples it on the same one: on a TAI-declared MS
    the declared numbers are 37 s from the instants they name, which would seed a
    later run with a satellite that brightens at the wrong times. That scale is
    stamped into the file as ``time_scale``, so a reader never has to assume it
    -- and so an untagged file, which pre-dates the stamp and may have been
    written on a declared scale, can be told apart and warned about.

    ``light_curves`` is the *magnitude* ``|S_hat|``, an apparent flux in Jy: the
    reader casts to float64, which would silently discard the imaginary part of a
    complex array. The native complex estimate is kept alongside it under
    ``light_curves_complex``, together with the noise floor (``error``), the
    z statistic and the in-view mask. Readers of the format ignore the extras.

    Where an along-track offset was fitted (:func:`attach_offset_fits`) its
    answers are written too -- ``tau_best`` above all, without which a later run
    cannot reproduce the trajectory these curves were measured on -- with
    ``r_best`` swapped into the file's ``(n_src, n_time, n_freq)`` orientation
    like the curves themselves.

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
        times=np.asarray(result["times_mjd_utc"], dtype=np.float64),
        # Stamped rather than assumed on read: without it a file written before
        # the format stated a scale is indistinguishable from one written after,
        # and a legacy file measured on a TAI-declared MS is 37 s out.
        time_scale="utc",
        freqs=np.asarray(result["freqs"], dtype=np.float64),
        light_curves_complex=lc,
        error=error,
        z=z,
        data_col=str(result["data_col"]),
        corr=str(result["corr"]),
    )
    if result.get("in_view") is not None:
        arrays["in_view"] = np.asarray(result["in_view"], dtype=bool)

    for key in _OFFSET_FIT_KEYS:
        if key in result:
            arrays[key] = np.asarray(result[key])
    if "r_best" in result:
        arrays["r_best"] = np.swapaxes(np.asarray(result["r_best"]), 1, 2)

    np.savez(path, **arrays)


# ---------------------------------------------------------------------------
# Epoch-shifted orbit records
# ---------------------------------------------------------------------------

#: Resolution of a TLE line-1 epoch field, in days: eight decimal days, 0.86 ms,
#: about 7 m along a LEO track.
_TLE_EPOCH_DECIMALS = 8


def _tle_epoch_field(epoch_jd: float):
    """``(year, day_of_year)`` as a TLE line-1 epoch field can hold them.

    The two halves of ``YYDDD.DDDDDDDD`` are one number, so they have to be
    derived from the same instant *at the same precision*. Rounding the day to
    the field's eight decimals while taking the year from the unrounded epoch
    lets them disagree within half a quantum (0.43 ms) of a New Year: the day
    rounds up to ``366.00000000`` while the year still reads the old one, and
    2025 has no day 366. That is not a date, and the parser rejects the pair
    outright -- so a file written for a later run could not be read back at all.

    Quantise first, then carry: if the rounded day reaches the year's length
    plus one it is the first instant of the next year, which is where it is
    written. One carry is always enough, since the unrounded day is inside its
    own year by construction and rounding moves it by at most half a quantum.
    """
    year = jd_to_datetime(epoch_jd).year
    day_of_year = round(
        epoch_jd - datetime_to_jd(datetime(year, 1, 1)) + 1.0, _TLE_EPOCH_DECIMALS
    )

    days_in_year = 366 if calendar.isleap(year) else 365
    if day_of_year >= days_in_year + 1.0:
        day_of_year -= days_in_year
        year += 1

    return year, day_of_year


def shift_orbit_record_epoch(record, tau_s: float) -> dict:
    """A copy of ``record`` whose epoch is moved by ``-tau_s`` seconds.

    The zero-code-change way to use a fitted offset. ``tau`` is measured as the
    time the *satellite* is evaluated at, so a positive one means the elements
    are late -- the satellite is where they say it will be ``tau`` seconds later
    -- and the record has to *become* that trajectory: propagating the shifted
    elements at ``t`` reproduces the original at ``t + tau``. Hence the minus.

    Written into a directory by :func:`write_shifted_orbits`, the result is
    picked up by ``--extra-orbit-dir`` and nothing else in tabascal has to know
    about the search at all.

    A TLE's line-1 epoch field (columns 19-32, ``YYDDD.DDDDDDDD``) is rewritten
    and the modulo-10 checksum recomputed, since a rewritten field invalidates it
    and the parser rejects a bad one -- as it should, that being how a
    single-character corruption is caught. Nothing else on either line moves. The
    field quantises to ``1e-8`` days, 0.86 ms, which is about 7 m along a LEO
    track; an OMM has no fixed-width field and keeps the epoch to the
    microsecond, so it is the format to shift where there is a choice. Either
    way the record's own ``EPOCH`` column is moved with it, since a record whose
    column disagreed with its own elements would be read one way by the age
    policy and another by the propagator.

    Parameters
    ----------
    record : dict
        A TLE or OMM orbit record.
    tau_s : float
        The measured along-track offset in seconds. Zero returns the record's own
        epoch unchanged, not a re-encoding of it.

    Returns
    -------
    dict
        A copy; the record handed in is not edited under the caller.
    """
    out = dict(record)
    epoch_jd = record_epoch_jd(record) - float(tau_s) / 86400.0

    if record_kind(record) == KIND_TLE:
        # The day of year comes off the Julian Dates rather than the datetime,
        # whose microsecond resolution would round the field's last digit and
        # leave tau = 0 a no-op only by luck.
        year, day_of_year = _tle_epoch_field(epoch_jd)
        line1 = record["TLE_LINE1"]
        body = line1[:18] + f"{year % 100:02d}{day_of_year:012.8f}" + line1[32:68]
        out["TLE_LINE1"] = body + str(tle_checksum(body))
        # The column names what the *lines* encode, not what was asked for, so
        # the quantised epoch and the text agree exactly.
        epoch_jd = datetime_to_jd(datetime(year, 1, 1)) + day_of_year - 1.0

    out["EPOCH"] = jd_to_datetime(epoch_jd).isoformat()

    return out


def write_shifted_orbits(
    directory: str,
    norad_ids,
    records: list,
    taus_s,
    filename: str = "shifted_orbits.json",
) -> str:
    """Write epoch-shifted orbit records where a later run can pick them up.

    One file in ``extra_orbit_dir`` format (:func:`tabascal.orbit.save_orbits_for_reuse`),
    so ``tabascal run --extra-orbit-dir <directory>`` reproduces the trajectories
    the search measured -- with the default unlimited age ceiling, and
    independently of what SatChecker serves by then.

    Parameters
    ----------
    directory : str
        Directory to write into; created if it does not exist.
    norad_ids : sequence of int
        Catalogue IDs, aligned with ``records``.
    records : sequence of dict
        The orbit records to shift.
    taus_s : float or sequence of float
        Offsets in seconds, one per record or one for all of them.
    filename : str, default "shifted_orbits.json"
        Name of the file inside ``directory``.

    Returns
    -------
    str
        The path written.
    """
    from tabascal.orbit import save_orbits_for_reuse

    taus = np.atleast_1d(np.asarray(taus_s, dtype=np.float64))
    if taus.size == 1:
        taus = np.repeat(taus, len(records))

    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    save_orbits_for_reuse(
        path,
        [int(n) for n in norad_ids],
        [shift_orbit_record_epoch(r, float(t)) for r, t in zip(records, taus)],
    )

    return path


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

    A source with no finite cell -- one held out of view for the whole
    observation by the elevation cut, or flagged away -- has no coverage to
    report and comes back as nan. It is still listed in ``per_source``, so the
    table says it was not measured rather than omitting it, but ``overall``
    summarises only the sources that *were*: nan is not a bad score, and a
    source nothing was measured for cannot be the worst-fitted one. With no
    source measured at all, every ``overall`` coverage metric is nan and
    ``worst_source`` is ``None``; the thresholds (``z_crit``, ``amp_crit``) are
    settings rather than measurements and stay populated.

    Returns
    -------
    dict
        ``per_source`` (title, coverage, null_coverage, excess, amp_coverage,
        max_z, max_amp, n_cells) and ``overall`` (pooled coverage, null,
        amp_coverage, worst source, mean, z_crit, amp_crit). When nothing was
        measured, ``worst_source`` is ``None`` and every coverage metric is nan,
        while ``z_crit`` and ``amp_crit`` are unchanged.
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

    # Measured sources only. A source that was never in view has no finite cell
    # and so a nan coverage, and nan loses every comparison -- `min` would keep
    # whichever nan it met first and report a source nothing was measured for as
    # the worst-fitted one, while `np.nanmean` over nothing but nans warns about
    # an empty slice. With none measured every coverage figure is nan, which is
    # what "no coverage" means; the per-source table still lists them.
    measured = [p for p in per_source if np.isfinite(p["coverage"])]
    covs = [p["coverage"] for p in measured]
    worst = min(measured, key=lambda p: p["coverage"]) if measured else None

    return dict(
        per_source=per_source,
        overall=dict(
            coverage=(pooled_in / pooled_n) if pooled_n else float("nan"),
            null_coverage=(pooled_null_in / pooled_n) if pooled_n else float("nan"),
            amp_coverage=(pooled_amp_in / pooled_n) if pooled_n else float("nan"),
            mean_coverage=float(np.mean(covs)) if covs else float("nan"),
            worst_source=worst["title"] if worst is not None else None,
            worst_coverage=worst["coverage"] if worst is not None else float("nan"),
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


def plot_offset_diagnostics(  # pragma: no cover - matplotlib output, not unit-tested
    fit: dict,
    save_path: str,
    title: Optional[str] = None,
) -> str:
    """The three panels an along-track detection is judged by eye on.

    Top, the frame-by-channel correlation ``|r|`` at the best offset, with the
    satellite's elevation over it: a real detection is a band that lights up
    while the satellite is up and goes out when it sets, on one channel, not a
    scatter of warm cells. Bottom left, the per-channel ``z2`` at that offset
    against the decohered null's mean and spread, which is the comparison the
    significance is. Bottom right, the scan itself for the best channel, with the
    other channels' range shaded behind it: a single peak that decays back into
    that band is what makes the best cell a measurement rather than the largest
    of many sidelobes.

    Parameters
    ----------
    fit : dict
        A result from :func:`fit_time_offset`.
    save_path : str
        Output PNG path.
    title : str, optional
        Prefix for the figure title, usually the NORAD ID.

    Returns
    -------
    str
        ``save_path``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = np.asarray(fit["r_best"], dtype=float)  # (n_freq, n_time)
    taus = np.asarray(fit["tau_grid"], dtype=float)
    z2_tau = np.asarray(fit["z2_tau"], dtype=float)
    elevation = np.asarray(fit["elevation"], dtype=float)
    n_freq, n_time = r.shape

    times = np.asarray(
        fit.get("times_sec", np.arange(n_time, dtype=float)), dtype=float
    )
    chan = int(fit["best_chan"])
    i_tau = int(np.argmin(np.abs(taus - fit["tau_best"]))) if np.isfinite(
        fit["tau_best"]
    ) else 0
    null_mean, null_std = float(fit["null_mean"]), float(fit["null_std"])

    fig = plt.figure(figsize=(12, 8.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.32, wspace=0.24)

    ax = fig.add_subplot(grid[0, :])
    step = float(np.mean(np.diff(times))) if n_time > 1 else 1.0
    extent = [times[0] - step / 2, times[-1] + step / 2, -0.5, n_freq - 0.5]
    image = ax.imshow(
        r, origin="lower", aspect="auto", extent=extent, cmap="inferno",
        interpolation="nearest", vmin=0.0,
    )
    bar = fig.colorbar(image, ax=ax, pad=0.09)
    bar.set_label("per-frame matched-filter correlation |r|")
    if 0 <= chan < n_freq:
        ax.axhline(chan, color="cyan", ls="--", lw=1, alpha=0.8)
    ax.set_xlabel("Time since the first frame [s]")
    ax.set_ylabel("Channel")
    twin = ax.twinx()
    twin.plot(times, elevation, color="w", lw=1.5, alpha=0.85)
    twin.set_ylabel("elevation [deg]", color="gray")
    ax.set_title(
        f"Near-field matched filter over {fit['n_bl_used']} coherent baselines "
        rf"($b \leq {fit['b_coh']:.0f}$ m), $\tau = {fit['tau_best']:+.2f}$ s"
    )

    ax = fig.add_subplot(grid[1, 0])
    channels = np.arange(n_freq)
    ax.bar(
        channels, z2_tau[i_tau],
        color=["C3" if c == chan else "C0" for c in channels],
    )
    ax.axhspan(
        null_mean - null_std, null_mean + null_std, color="gray", alpha=0.3,
        label=f"decohered null ({null_mean:.3f}$\\pm${null_std:.3f})",
    )
    ax.axhline(null_mean, color="gray", lw=1)
    ax.set_xlabel("Channel")
    ax.set_ylabel(r"$z^2$ (combined over frames)")
    ax.set_title(
        f"Per-channel score at the best offset: "
        f"{fit['significance']:.1f}$\\sigma$ above the null"
    )
    ax.legend(fontsize=8)

    ax = fig.add_subplot(grid[1, 1])
    ax.plot(taus, z2_tau[:, chan], "o-", color="C3", ms=3, label=f"chan {chan}")
    others = np.delete(z2_tau, chan, axis=1)
    if others.size:
        ax.fill_between(
            taus, others.min(axis=1), others.max(axis=1), color="C0", alpha=0.25,
            label="all other channels (range)",
        )
    ax.axvline(fit["tau_best"], color="k", ls="--", lw=1)
    ax.set_xlabel(r"along-track time offset $\tau$ [s]")
    ax.set_ylabel(r"$z^2$")
    ax.set_title(rf"Along-track scan, best $\tau = {fit['tau_best']:+.2f}$ s")
    ax.legend(fontsize=8)

    fig.suptitle(
        f"{title + ': ' if title else ''}along-track matched-filter evidence",
        y=0.99, fontsize=13,
    )
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    return save_path


def plot_candidate_ranking(  # pragma: no cover - matplotlib output, not unit-tested
    search: dict,
    selection: dict,
    save_path: str,
    title: Optional[str] = None,
) -> str:
    """The whole field, ranked: the chart a named satellite is judged against.

    One bar per candidate in rank order, detections in red, the candidate median
    as a dotted line and the winner annotated with its channel, its score and the
    runner-up's. A single NORAD ID is not evidence; a winner standing clear of a
    median drawn from every satellite that was up is. On the MWA Cen A case that
    was 0.0995 against a runner-up of 0.0523 and a median of 0.0446.

    Parameters
    ----------
    search : dict
        A result from :func:`search_candidates`.
    selection : dict
        The matching :func:`select_detections` result; its rows are highlighted.
    save_path : str
        Output PNG path.
    title : str, optional
        Prefix for the figure title.

    Returns
    -------
    str
        ``save_path``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = search["table"]
    z2 = np.asarray(search["z2_best"], dtype=float)
    detected = {row["rank"] for row in selection["detected"]}
    ranks = np.arange(len(table))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(
        ranks,
        z2,
        width=0.9,
        color=["C3" if rank in detected else "C0" for rank in ranks],
    )
    median = float(search["median_z2"])
    ax.axhline(median, color="gray", ls=":", lw=1)
    if len(table):
        ax.text(
            len(table) - 1, median * 1.08, f"median {median:.4f}", color="gray",
            fontsize=8, ha="right",
        )
        winner = table[0]
        runner_up = f", runner-up {z2[1]:.4f}" if len(table) > 1 else ""
        ax.annotate(
            f"{winner['name']} / {winner['norad_id']}\n"
            f"best chan {winner['best_chan']}, "
            rf"$z^2$={winner['z2_best']:.4f}{runner_up}",
            xy=(0, winner["z2_best"]),
            # Pinned to the top right in axes fractions rather than offset from
            # the bar in data space: the x axis is a rank count, so an offset
            # that reads well over a constellation runs off a plot of five. The
            # bars descend, so that corner is the one always free.
            xytext=(0.97, 0.92), textcoords="axes fraction", ha="right", va="top",
            arrowprops=dict(arrowstyle="->", color="C3"), color="C3", fontsize=9,
        )
    ax.set_xlabel(
        f"Candidate rank ({len(table)} above-horizon satellites, tau-scanned "
        f"near-field search over {search['n_bl_used']} coherent baselines)"
    )
    ax.set_ylabel(r"best $z^2$ (max over $\tau$ and channel)")
    ax.set_title(f"{title + ': ' if title else ''}all-candidate ranking")

    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)

    return save_path
