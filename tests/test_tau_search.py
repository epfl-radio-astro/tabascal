"""Single-satellite along-track (time-offset) matched-filter search (GitHub #190).

A TLE's dominant error is along-track: the satellite runs kilometres ahead of or
behind where the elements say, which is very nearly a pure *time offset* in the
trajectory. Scanning one parameter ``tau`` -- evaluate the orbit at ``t + tau``
-- recovers it, and is what makes a near-field matched filter work at all on a
km-class array. On the MWA Cen A dataset STARLINK-1765 sat 2.25 s (17 km) behind
a TLE 1.4 h old, and the contaminated channel went from ``z2 = 0.045`` at
``tau = 0`` to ``0.107`` at ``tau = -2.25`` s, 5.6 sigma above a decohered null.

What is pinned here is the *statistic*, not that a number comes out. A filter
that averages the right samples with the wrong template still returns plausible
correlations, and only an independent transcription of the geometry tells the
two apart. So every physical quantity is written out a second time in this file,
from the same primitives the forward model uses
(:func:`~tabascal.components.trajectory.get_satellite_positions`,
:func:`~tabascal.components.trajectory.itrs_to_gcrs_sf`,
:func:`~tabascal.interferometry.itrf_to_uvw_numpy`), and the module under test is
checked against that:

* the near-field path difference is ``|x_sat - x_p| + w_p - |x_sat - x_q| - w_q``
  -- never the plane-wave form -- with the satellite at ``t + offset + tau`` and
  the **antennas left at** ``t + offset``. Shifting the antennas too is a
  different (and wrong) model: it moves the answer by ~0.24 m of path here, well
  above every tolerance below, so the two are told apart rather than assumed;
* at ``n_fine = 1`` and ``tau = 0`` the model must be exactly the template
  :func:`~tabascal.rfi_estimate.rfi_phase_from_records` already builds, which is
  what ties the search to the forward model it feeds;
* the per-frame statistic is a normalised correlation, ``r = |z| / sqrt(n1 n2)``,
  so the intra-dump fringe smearing that shrinks ``|m|`` on the longer baselines
  cancels out of it -- and ``z2 = sum_frames |z|^2 / (n1 n2)`` is bounded by the
  number of in-view frames, which is what makes the null comparison meaningful.

The synthetic observation is an ISS-shaped TLE (the repo's own offline template)
propagated by the repo's own skyfield path, with the array centre placed at the
sub-satellite point so the pass is overhead and the near-field geometry is
strong: 8 antennas inside 250 m plus one deliberate 8 km outlier, whose
baselines the #189 coherence cut is expected to drop. A fringe at
``tau0 = +1.5 s`` is injected on one channel only, by an independent numpy model
built here. The expected answers are therefore known: ``tau_best = +1.5 s``,
``best_chan = 2``, 28 coherent non-auto baselines out of 45.

Everything is offline -- no MS, no SatChecker, no network (``block_network``
enforces it). The host-side geometry is numpy/f64 and does not move with
``--x64``; the jitted core does, so its tolerances come from ``precision``.
"""

import calendar
import os
import re
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from skyfield.framelib import itrs

from satchecker_client.records import record_epoch_jd
from satchecker_client.tle_parse import validate_tle_pair

from tabascal.components.rfi_signal import read_light_curves
from tabascal.components.trajectory import (
    _earth_satellite,
    get_satellite_elevations,
    get_satellite_positions,
    itrs_to_gcrs_sf,
)
from tabascal.interferometry import C, itrf_to_uvw_numpy
from tabascal.orbit import _select_from_extra_dir
from tabascal.rfi_estimate import (
    DEFAULT_TAU_GRID,
    _lc_result,
    attach_offset_fits,
    baseline_lengths,
    coherence_scores,
    coherent_baseline_mask,
    decohered_null,
    fine_time_offsets,
    fit_time_offset,
    is_detection,
    matched_filter_sums,
    near_field_baseline_paths,
    near_field_fringe_model,
    plot_offset_diagnostics,
    rfi_phase_from_positions,
    rfi_phase_from_records,
    satellite_range_and_speed,
    save_light_curves_npz,
    select_sources,
    shift_orbit_record_epoch,
    tau_scan,
    write_shifted_orbits,
)
from tabascal.time import gast_deg, jd_to_mjd, skyfield_time, timescale

from .tle_helpers import block_network, jd, make_omm, make_tle_record  # noqa: F401


# ---------------------------------------------------------------------------
# The synthetic observation
# ---------------------------------------------------------------------------

NORAD_ID = 25544
DECOY_ID = 27386
#: Seconds of epoch shift that puts the decoy elsewhere along its own orbit --
#: still 50-60 deg above the horizon, so it is not the elevation cut that
#: rejects it, but ~230 km from where the data's fringe was built, which no tau
#: on the grid can recover.
DECOY_EPOCH_SHIFT_S = 30.0

EPOCH_JD = jd(2026, 8, 1)
INT_TIME = 0.5
N_TIME = 20
N_FINE = 40
N_FREQ = 4
FREQS = 175e6 + 40e3 * np.arange(N_FREQ)
#: The true along-track offset injected into the visibilities, and a point of
#: the default tau grid so the scan can land on it exactly.
TAU0 = 1.5
INJECTED_CHAN = 2
SIGMA = 1.0
#: Per-baseline fringe amplitude, in units of sigma. Chosen so the per-frame
#: correlation on the injected channel lands around 0.8: high enough to be a
#: detection, low enough that the statistic is not saturated at 1.
AMP = 3.0
SIGMA_TRANSVERSE = 300.0

#: Antenna offsets (east, north) in metres from the array centre. Eight inside a
#: 250 m footprint -- every baseline shorter than the coherence length the cut
#: works out below -- and one at 8 km, whose six baselines it must drop.
ANT_OFFSETS = np.array(
    [
        [0.0, 0.0],
        [25.0, 0.0],
        [-40.0, 30.0],
        [60.0, -70.0],
        [-90.0, -60.0],
        [100.0, 85.0],
        [-20.0, 120.0],
        [70.0, 40.0],
        [8000.0, 0.0],
    ]
)
FAR_ANT = len(ANT_OFFSETS) - 1
N_ANT = len(ANT_OFFSETS)
#: Non-auto baselines that survive the coherence cut: every pair among the eight
#: core antennas. Derived from :func:`coherent_baseline_mask` in the tests too,
#: so a change in the cut shows up as a disagreement rather than as a stale
#: literal.
N_COHERENT_BL = 28

#: The TLE line-1 epoch field carries eight decimal days: 1e-8 d, 0.86 ms,
#: about 7 m along a LEO track. Nothing encoded into it is exact beyond that.
TLE_EPOCH_QUANTUM_DAYS = 1e-8
#: Two New Year instants, both in the past so the record stays inside the
#: epoch plausibility window whenever the suite is run: 2026-01-01 follows a
#: common year (2025, 365 days) and 2025-01-01 follows a leap year (2024, 366).
#: Those are the two day-of-year ceilings a rollover can trip over.
AFTER_A_COMMON_YEAR_JD = jd(2026, 1, 1)
AFTER_A_LEAP_YEAR_JD = jd(2025, 1, 1)


def _days_in_year(year):
    """365 or 366, written out rather than taken from the code under test."""
    return 366 if calendar.isleap(int(year)) else 365


#: Where TAU0 sits on the default grid.
BEST_INDEX = int(np.argmin(np.abs(DEFAULT_TAU_GRID - TAU0)))
#: A coarser grid for the fits that are not about the grid itself. It still
#: holds TAU0, so a recovery test on it means the same thing.
SHORT_GRID = np.arange(-2.0, 2.0 + 0.25, 0.5)


def _model_atol(precision):
    """Tolerance on the fringe model, set by the precision it is built in.

    The paths are host f64 either way; the model is not. Under ``--x64 false``
    an 8 km path in f32 carries ~5e-4 m of rounding, which is ~2e-3 rad of
    phase -- far below anything the physics tests turn on, and far above f64.
    """
    return 1e-9 if precision == "double" else 3e-3


def ref_fine_offsets(n_fine, delta_t):
    """Midpoint sub-step offsets, written out rather than imported."""
    return ((np.arange(n_fine) + 0.5) / n_fine - 0.5) * delta_t


def ref_paths(record, ants_itrf, times_jd, phase_centre, a1, a2, n_fine, delta_t,
              taus_s=0.0, shift_antennas=False):
    """Near-field baseline path differences, transcribed independently.

    The satellite is propagated to ``t + offset + tau``; the antennas and their
    phase-tracking ``w`` stay at ``t + offset`` unless ``shift_antennas``, which
    exists only so a test can show the two are not the same model.
    """
    taus_s = np.atleast_1d(np.asarray(taus_s, dtype=np.float64))
    offsets = ref_fine_offsets(n_fine, delta_t)
    times_jd = np.asarray(times_jd, dtype=np.float64)
    t_fine = (times_jd[:, None] + offsets[None, :] / 86400.0).ravel()

    out = np.zeros((len(taus_s), len(a1), len(times_jd), n_fine))
    for i, tau in enumerate(taus_s):
        t_ant = t_fine + (tau / 86400.0 if shift_antennas else 0.0)
        gh0 = (gast_deg(t_ant) - phase_centre["ra"]) % 360
        w = np.transpose(
            itrf_to_uvw_numpy(ants_itrf, gh0, phase_centre["dec"]), axes=(1, 0, 2)
        )[..., -1]
        ants_xyz = itrs_to_gcrs_sf(ants_itrf, t_ant)
        sat_xyz = np.asarray(
            get_satellite_positions([record], t_fine + tau / 86400.0)
        )[0]
        path = np.linalg.norm(ants_xyz - sat_xyz[None], axis=-1) + w
        out[i] = (path[a1] - path[a2]).reshape(len(a1), len(times_jd), n_fine)

    return out


def ref_model(paths, freqs=FREQS):
    """``mean_fine exp(-2 pi i path / lam)``, in numpy/f64."""
    lam = C / np.asarray(freqs, dtype=np.float64)
    phase = -2j * np.pi * np.asarray(paths)[..., None, :, :] / lam[:, None, None]

    return np.exp(phase).mean(axis=-1)


def ref_scores(vis, weights, model, frame_mask=None):
    """``(r, z2)`` written out: the estimator, with no chunking and no tricks."""
    w = np.broadcast_to(np.asarray(weights, dtype=np.float64), np.shape(vis))
    z = np.sum(w * vis * np.conjugate(model), axis=0)
    n1 = np.sum(w * np.abs(vis) ** 2, axis=0)
    n2 = np.sum(w * np.abs(model) ** 2, axis=0)
    den = n1 * n2
    safe = np.where(den > 0, den, 1.0)
    r = np.where(den > 0, np.abs(z) / np.sqrt(safe), 0.0)
    per_frame = np.where(den > 0, np.abs(z) ** 2 / safe, 0.0)
    mask = np.ones(np.shape(vis)[-1]) if frame_mask is None else np.asarray(
        frame_mask, dtype=np.float64
    )

    return r, np.sum(mask[None, :] * per_frame, axis=-1)


def ref_range_and_speed(record, ants_itrf, times_jd):
    """Slant range and line-of-sight-crossing speed, by finite difference.

    Both in the Earth-fixed frame, which is the one the array sits still in:
    ``v_perp = r |d s_hat / dt|`` with ``s_hat`` the unit vector from the array
    centre to the satellite.
    """
    t_mid = float(np.mean(np.asarray(times_jd, dtype=np.float64)))
    centre = np.mean(np.asarray(ants_itrf), axis=0)
    step = 0.5 / 86400.0
    sat = _earth_satellite(record, timescale())
    pos = sat.at(skyfield_time(np.array([t_mid - step, t_mid, t_mid + step])))
    s = pos.frame_xyz(itrs).m.T - centre[None]
    s_hat = s / np.linalg.norm(s, axis=-1, keepdims=True)
    range_m = float(np.linalg.norm(s[1]))
    v_perp = float(np.linalg.norm((s_hat[2] - s_hat[0]) / (2 * 0.5)) * range_m)

    return range_m, v_perp


def build_observation(record, times_jd, ants_itrf, phase_centre, a1, a2, seed,
                      inject=True):
    """One synthetic observation: geometry, weights, and the visibilities.

    The fringe is injected through :func:`ref_model` -- this file's own model,
    owing nothing to the module under test -- on ``INJECTED_CHAN`` only, the way
    a narrowband downlink lands in one channel.
    """
    times_jd = np.asarray(times_jd, dtype=np.float64)
    n_time = len(times_jd)
    n_bl = len(a1)

    paths = ref_paths(record, ants_itrf, times_jd, phase_centre, a1, a2, N_FINE,
                      INT_TIME, TAU0)[0]
    model = ref_model(paths)

    rng = np.random.default_rng(seed)
    shape = (n_bl, N_FREQ, n_time)
    vis = rng.normal(0, SIGMA, shape) + 1j * rng.normal(0, SIGMA, shape)
    if inject:
        vis[:, INJECTED_CHAN, :] += AMP * model[:, INJECTED_CHAN, :]

    range_m, v_perp = ref_range_and_speed(record, ants_itrf, times_jd)
    bl_len = np.asarray(baseline_lengths(ants_itrf, a1, a2))
    coherent = np.asarray(
        coherent_baseline_mask(
            bl_len, float(np.mean(FREQS)), range_m, SIGMA_TRANSVERSE, N_FINE,
            INT_TIME, v_perp,
        )
    )
    keep = coherent & (a1 != a2)
    weights = np.where(keep[:, None, None], 1.0 / SIGMA**2, 0.0)

    return SimpleNamespace(
        record=record,
        ants_itrf=ants_itrf,
        a1=a1,
        a2=a2,
        n_bl=n_bl,
        n_time=n_time,
        times_jd=times_jd,
        freqs=FREQS,
        phase_centre=phase_centre,
        int_time=INT_TIME,
        vis=vis,
        model=model,
        paths=paths,
        weights=weights,
        keep=keep,
        bl_len=bl_len,
        range_m=range_m,
        v_perp=v_perp,
        elevation=np.asarray(
            get_satellite_elevations([record], times_jd, ants_itrf)
        )[0],
    )


@pytest.fixture(scope="module")
def record():
    return make_tle_record(NORAD_ID, EPOCH_JD)


@pytest.fixture(scope="module")
def layout(record):
    """An array under the satellite: the centre is the sub-satellite point.

    Placed at the ground point below the satellite at mid-observation, so the
    pass is overhead and the direction to it swings fastest across the array --
    the geometry the near-field filter is for. The phase centre is put at the
    zenith of that site, which is where a real observation of it would point;
    it enters only through the ``w`` term.
    """
    times_jd = EPOCH_JD + np.arange(N_TIME) * INT_TIME / 86400.0
    t_ref = float(np.mean(times_jd))

    sat_itrs = (
        _earth_satellite(record, timescale())
        .at(skyfield_time(np.atleast_1d(t_ref)))
        .frame_xyz(itrs)
        .m.T[0]
    )
    centre = sat_itrs / np.linalg.norm(sat_itrs) * 6371e3
    up = centre / np.linalg.norm(centre)
    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)

    ants_itrf = (
        centre[None]
        + ANT_OFFSETS[:, :1] * east[None]
        + ANT_OFFSETS[:, 1:] * north[None]
    )
    a1, a2 = np.triu_indices(N_ANT, 0)  # autos included, as an MS lays them out
    lat = np.rad2deg(np.arcsin(centre[2] / np.linalg.norm(centre)))
    lon = np.rad2deg(np.arctan2(centre[1], centre[0]))
    phase_centre = {
        "ra": float((gast_deg(np.atleast_1d(t_ref))[0] + lon) % 360.0),
        "dec": float(lat),
    }

    return SimpleNamespace(
        ants_itrf=ants_itrf, a1=a1, a2=a2, phase_centre=phase_centre,
        times_jd=times_jd, t_ref=t_ref,
    )


@pytest.fixture(scope="module")
def obs(record, layout):
    """The observation with the tau0 fringe injected on one channel."""
    return build_observation(record, layout.times_jd, layout.ants_itrf,
                             layout.phase_centre, layout.a1, layout.a2, seed=12345)


@pytest.fixture(scope="module")
def noise_obs(record, layout):
    """The same observation with no satellite in it at all."""
    return build_observation(record, layout.times_jd, layout.ants_itrf,
                             layout.phase_centre, layout.a1, layout.a2, seed=0,
                             inject=False)


@pytest.fixture(scope="module")
def setting_obs(record, layout):
    """A window the satellite leaves partway through.

    It starts at the overhead instant, so the elevation falls monotonically from
    there, and the cut is placed numerically at the elevation halfway through
    the window. The geometric horizon would do the same job to the masking code
    while destroying the geometry the rest of the fit needs -- a grazing pass is
    ten times further away and crosses the line of sight far more slowly -- so
    the cut is put where the satellite actually is rather than at zero degrees.
    """
    times_jd = layout.t_ref + np.arange(N_TIME) * INT_TIME / 86400.0
    observation = build_observation(record, times_jd, layout.ants_itrf,
                                    layout.phase_centre, layout.a1, layout.a2,
                                    seed=4321)
    elevation = observation.elevation
    half = N_TIME // 2
    observation.min_elevation = float(0.5 * (elevation[half - 1] + elevation[half]))
    observation.frames = elevation >= observation.min_elevation

    return observation


@pytest.fixture(scope="module")
def default_fit(obs):
    """The headline fit: default grid, default null, nothing else touched."""
    return fit_time_offset(
        obs.vis, obs.record, obs.ants_itrf, obs.times_jd, obs.phase_centre,
        obs.freqs, obs.a1, obs.a2, obs.int_time, noise=SIGMA,
    )


@pytest.fixture(scope="module")
def grid_paths(obs):
    """The default grid's path differences, built once for the scan tests."""
    return ref_paths(obs.record, obs.ants_itrf, obs.times_jd, obs.phase_centre,
                     obs.a1, obs.a2, N_FINE, obs.int_time, DEFAULT_TAU_GRID)


@pytest.fixture(scope="module")
def setting_fit(setting_obs):
    """The fit on the window the satellite leaves partway through."""
    return fit_time_offset(
        setting_obs.vis, setting_obs.record, setting_obs.ants_itrf,
        setting_obs.times_jd, setting_obs.phase_centre, setting_obs.freqs,
        setting_obs.a1, setting_obs.a2, setting_obs.int_time, noise=SIGMA,
        taus_s=SHORT_GRID, min_elevation=setting_obs.min_elevation, n_null=64,
    )



# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

class TestFineTimeOffsets:
    """Where inside an integration the model is sampled."""

    def test_a_single_step_is_the_integration_centre(self):
        """n_fine = 1 must reduce to the template the forward model builds."""
        np.testing.assert_array_equal(fine_time_offsets(1, 2.0), np.array([0.0]))

    def test_they_are_the_midpoints_of_equal_sub_steps(self):
        """Midpoints, not edges: an edge grid samples one boundary twice over
        consecutive integrations and biases the average by half a step."""
        np.testing.assert_allclose(
            fine_time_offsets(4, 2.0), [-0.75, -0.25, 0.25, 0.75]
        )

    def test_they_are_symmetric_about_the_centre_and_scale_with_the_dump(self):
        offsets = fine_time_offsets(N_FINE, INT_TIME)

        assert offsets.shape == (N_FINE,)
        assert abs(float(np.sum(offsets))) < 1e-12
        assert np.all(np.abs(offsets) < INT_TIME / 2)
        np.testing.assert_allclose(np.diff(offsets), INT_TIME / N_FINE)
        np.testing.assert_allclose(
            fine_time_offsets(8, 4.0), 2.0 * fine_time_offsets(8, 2.0)
        )


class TestNearFieldBaselinePaths:
    """The path difference the whole statistic is built on."""

    def test_a_single_fine_step_is_the_forward_models_own_template(self, obs):
        """The tie to the model this search feeds.

        ``exp(-2 pi i path / lam)`` at one sub-step and zero offset has to be
        the ``exp(i(phi_p - phi_q))`` that
        :func:`rfi_phase_from_records` -- and so ``FixedOrbit`` -- already
        builds. Both sides are host f64, so the agreement is exact to rounding
        in either session precision.
        """
        paths = near_field_baseline_paths(
            obs.record, obs.ants_itrf, obs.times_jd, obs.phase_centre, obs.a1,
            obs.a2, 1, obs.int_time, taus_s=0.0,
        )
        phase = rfi_phase_from_records(
            [obs.record], obs.ants_itrf, obs.times_jd, obs.phase_centre, obs.freqs
        )
        antenna = np.exp(1j * phase[0])
        template = antenna[obs.a1] * np.conjugate(antenna[obs.a2])
        lam = C / obs.freqs
        got = np.exp(-2j * np.pi * paths[0, :, None, :, 0] / lam[None, :, None])

        np.testing.assert_allclose(got, template, atol=1e-7)

    def test_the_shape_carries_a_tau_axis_and_a_fine_axis(self, obs):
        taus = np.array([-0.5, 0.0, 0.5])
        paths = near_field_baseline_paths(
            obs.record, obs.ants_itrf, obs.times_jd, obs.phase_centre, obs.a1,
            obs.a2, N_FINE, obs.int_time, taus_s=taus,
        )

        assert paths.shape == (len(taus), obs.n_bl, obs.n_time, N_FINE)
        # Host geometry: metres of path against a 350 km range, which f32 cannot
        # hold, so this stays f64 whatever --x64 says.
        assert paths.dtype == np.float64
        # A single offset is a grid of one, not a dropped axis: the core is
        # written once, for a grid.
        scalar = near_field_baseline_paths(
            obs.record, obs.ants_itrf, obs.times_jd, obs.phase_centre, obs.a1,
            obs.a2, 4, obs.int_time, taus_s=TAU0,
        )
        assert scalar.shape == (1, obs.n_bl, obs.n_time, 4)

    def test_it_matches_an_independent_transcription(self, obs):
        """And a grid of offsets is the individual calls, stacked."""
        taus = np.array([-1.0, TAU0])
        want = ref_paths(obs.record, obs.ants_itrf, obs.times_jd,
                         obs.phase_centre, obs.a1, obs.a2, N_FINE, obs.int_time,
                         taus)

        got = near_field_baseline_paths(
            obs.record, obs.ants_itrf, obs.times_jd, obs.phase_centre, obs.a1,
            obs.a2, N_FINE, obs.int_time, taus_s=taus,
        )

        np.testing.assert_allclose(got, want, atol=1e-5)
        np.testing.assert_allclose(got[1], obs.paths, atol=1e-5)

    def test_a_positive_offset_moves_the_satellite_forward(self, obs):
        """tau is the time the *satellite* is evaluated at, and it matters.

        1.5 s is 11 km along the track, which moves the path difference by
        metres -- larger than a wavelength, which is why the scan works at all.
        """
        zero = near_field_baseline_paths(
            obs.record, obs.ants_itrf, obs.times_jd, obs.phase_centre, obs.a1,
            obs.a2, N_FINE, obs.int_time, taus_s=0.0,
        )
        core = obs.keep

        assert np.abs(obs.paths - zero[0])[core].max() > 1.0

    def test_the_antennas_stay_at_the_integration_times(self, obs):
        """Only the orbit is shifted -- the array is not re-rotated.

        Shifting the antenna frame with it is a different model: the Earth turns
        under the phase tracking and the path difference moves by ~0.2 m, which
        is a tenth of a wavelength at 175 MHz. The tolerance above (1e-5 m)
        would not survive it, so the right model is pinned rather than assumed.
        """
        both = ref_paths(obs.record, obs.ants_itrf, obs.times_jd,
                         obs.phase_centre, obs.a1, obs.a2, N_FINE, obs.int_time,
                         TAU0, shift_antennas=True)[0]

        assert np.abs(obs.paths - both)[obs.keep].max() > 0.05

    def test_an_autocorrelation_has_no_path_difference(self, obs):
        """Which is why it carries no fringe, and is dropped by default."""
        paths = near_field_baseline_paths(
            obs.record, obs.ants_itrf, obs.times_jd, obs.phase_centre, obs.a1,
            obs.a2, 4, obs.int_time, taus_s=TAU0,
        )

        assert np.abs(paths[0][obs.a1 == obs.a2]).max() < 1e-9


class TestSatelliteRangeAndSpeed:
    """The two numbers that size the coherence cut."""

    def test_the_range_is_the_slant_distance_from_the_array_centre(self, obs):
        range_m, _ = satellite_range_and_speed(
            obs.record, obs.ants_itrf, obs.times_jd
        )

        # Within the spread of the range over the window, so any sensible
        # reading of "the middle of times_jd" agrees.
        assert range_m == pytest.approx(obs.range_m, rel=0.01)
        assert 300e3 < range_m < 400e3  # an ISS-shaped orbit, seen from below

    def test_the_speed_is_the_rate_the_line_of_sight_is_crossed(self, obs):
        _, v_perp = satellite_range_and_speed(
            obs.record, obs.ants_itrf, obs.times_jd
        )

        assert v_perp == pytest.approx(obs.v_perp, rel=0.1)
        assert 5e3 < v_perp < 8e3  # nearly all of the orbital speed, overhead

    def test_a_low_pass_crosses_the_line_of_sight_more_slowly(self, record,
                                                              layout):
        """Not the orbital speed: the component across the line of sight.

        Half an orbit-quarter later the satellite is near the horizon, still
        moving at ~7.7 km/s, but most of that is now *along* the line of sight
        and does not fringe. A function returning the speed itself would pass
        the overhead test above and fail here.
        """
        times_jd = layout.t_ref + 300.0 / 86400.0 + np.arange(4) * INT_TIME / 86400.0
        elevation = get_satellite_elevations([record], times_jd, layout.ants_itrf)
        range_m, v_perp = satellite_range_and_speed(
            record, layout.ants_itrf, times_jd
        )
        ref_range, ref_v = ref_range_and_speed(record, layout.ants_itrf, times_jd)

        assert float(np.max(elevation)) < 10.0  # it really is on its way out
        assert range_m == pytest.approx(ref_range, rel=0.01)
        assert range_m > 1500e3
        assert v_perp == pytest.approx(ref_v, rel=0.1)
        assert v_perp < 0.8 * 7.7e3


# ---------------------------------------------------------------------------
# The jitted core
# ---------------------------------------------------------------------------

class TestNearFieldFringeModel:
    """The per-integration fringe model, averaged over its fine grid."""

    def test_one_fine_step_is_a_unit_modulus_steering_vector(self, obs):
        model = near_field_fringe_model(obs.paths[:, :, :1], obs.freqs)

        np.testing.assert_allclose(np.abs(np.asarray(model)), 1.0, atol=1e-5)

    def test_smearing_shrinks_the_fastest_fringing_baselines(self, obs):
        """The 8 km baseline sweeps tens of fringes inside one dump.

        Averaging them leaves almost nothing, which is the modelled loss the
        #189 fringe-rate criterion exists to keep out of the sum: past its
        ceiling a baseline decoheres against the model's own discretisation.
        """
        model = np.abs(np.asarray(near_field_fringe_model(obs.paths, obs.freqs)))
        far = ((obs.a1 == FAR_ANT) | (obs.a2 == FAR_ANT)) & (obs.a1 != obs.a2)

        assert model[far].max() < 0.1

    def test_a_short_baseline_survives_the_average(self, obs):
        """25 m turns a fraction of a fringe per dump, so it keeps its modulus.

        The average of unit-modulus samples can only shrink, never grow, so
        every baseline sits in ``[0, 1]`` and the short ones near the top.
        """
        model = np.abs(np.asarray(near_field_fringe_model(obs.paths, obs.freqs)))
        shortest = int(np.argmin(np.where(obs.a1 == obs.a2, np.inf, obs.bl_len)))

        assert obs.bl_len[shortest] == pytest.approx(25.0, abs=0.1)
        assert model[shortest].min() > 0.9
        assert model.max() <= 1.0 + 1e-5

    def test_it_matches_an_independent_numpy_model(self, obs, precision):
        got = np.asarray(near_field_fringe_model(obs.paths, obs.freqs))

        assert got.shape == (obs.n_bl, N_FREQ, obs.n_time)
        assert got.dtype == jnp.zeros(1, dtype=complex).dtype
        np.testing.assert_allclose(got, obs.model, atol=_model_atol(precision))

    def test_it_is_jittable_over_a_leading_grid_axis(self, obs, precision):
        paths = np.stack([obs.paths, obs.paths[:, :, ::-1]])
        got = np.asarray(jax.jit(near_field_fringe_model)(paths, obs.freqs))

        assert got.shape == (2, obs.n_bl, N_FREQ, obs.n_time)
        np.testing.assert_allclose(got[0], obs.model, atol=_model_atol(precision))


class TestMatchedFilterSums:
    """The three weighted sums the statistic is assembled from."""

    @staticmethod
    def _inputs(seed=3, n_bl=6, n_freq=3, n_time=4):
        rng = np.random.default_rng(seed)
        shape = (n_bl, n_freq, n_time)
        vis = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        model = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        weights = rng.uniform(0.5, 2.0, shape)
        return vis, model, weights

    def test_they_are_the_weighted_inner_products(self, exact_rtol):
        vis, model, weights = self._inputs()

        z, n1, n2 = matched_filter_sums(vis, model, weights)

        np.testing.assert_allclose(
            z, np.sum(weights * vis * np.conjugate(model), axis=0), rtol=exact_rtol
        )
        np.testing.assert_allclose(
            n1, np.sum(weights * np.abs(vis) ** 2, axis=0), rtol=exact_rtol
        )
        np.testing.assert_allclose(
            n2, np.sum(weights * np.abs(model) ** 2, axis=0), rtol=exact_rtol
        )
        assert z.shape == n1.shape == n2.shape == vis.shape[1:]

    def test_the_weights_may_broadcast_from_the_baseline_axis(self, exact_rtol):
        vis, model, weights = self._inputs()
        per_bl = weights[:, :1, :1]

        z, n1, n2 = matched_filter_sums(vis, model, per_bl)
        z_full, _, _ = matched_filter_sums(
            vis, model, np.broadcast_to(per_bl, vis.shape)
        )

        np.testing.assert_allclose(z, z_full, rtol=exact_rtol)

    def test_a_zero_weight_removes_a_baseline_from_every_sum(self, exact_rtol):
        """Flagging is expressed as a weight, so it has to be exact.

        (Whether a *flagged* sample can be nan is the driver's problem, not the
        sums': see the fit's flag handling.)
        """
        vis, model, weights = self._inputs()
        dropped = weights.copy()
        dropped[2] = 0.0

        got = matched_filter_sums(vis, model, dropped)
        want = matched_filter_sums(
            np.delete(vis, 2, axis=0), np.delete(model, 2, axis=0),
            np.delete(dropped, 2, axis=0),
        )

        for a, b in zip(got, want):
            np.testing.assert_allclose(a, b, rtol=exact_rtol)


class TestCoherenceScores:
    """The per-frame correlation and the combined per-channel score."""

    @staticmethod
    def _sums(seed=5, n_freq=3, n_time=6):
        rng = np.random.default_rng(seed)
        z = rng.normal(size=(n_freq, n_time)) + 1j * rng.normal(size=(n_freq, n_time))
        n1 = rng.uniform(1.0, 4.0, (n_freq, n_time))
        n2 = rng.uniform(1.0, 4.0, (n_freq, n_time))
        return z, n1, n2

    def test_r_is_the_normalised_correlation(self, exact_rtol):
        z, n1, n2 = self._sums()

        r, z2 = coherence_scores(z, n1, n2)

        np.testing.assert_allclose(r, np.abs(z) / np.sqrt(n1 * n2), rtol=exact_rtol)
        np.testing.assert_allclose(
            z2, np.sum(np.abs(z) ** 2 / (n1 * n2), axis=-1), rtol=exact_rtol
        )
        assert r.shape == z.shape
        assert z2.shape == z.shape[:1]

    def test_a_perfect_match_correlates_at_one(self, obs, precision):
        """r is a correlation coefficient: bounded by 1, and reached only when
        the data are the model."""
        model = near_field_fringe_model(obs.paths, obs.freqs)
        r, _ = coherence_scores(*matched_filter_sums(model, model, obs.weights))

        np.testing.assert_allclose(np.asarray(r), 1.0, atol=1e-5)

    def test_a_frame_mask_is_the_same_as_slicing_those_frames_out(self,
                                                                  exact_rtol):
        """The two horizon strategies must agree.

        The single-satellite path slices the in-view window; the batched search
        of #191 cannot -- its shapes have to stay static -- so it passes a 0/1
        mask instead. If those disagreed, the two would report different
        detections for the same pass.
        """
        z, n1, n2 = self._sums()
        mask = np.array([1, 1, 1, 0, 0, 1], dtype=bool)

        _, masked = coherence_scores(z, n1, n2, frame_mask=mask)
        _, sliced = coherence_scores(z[:, mask], n1[:, mask], n2[:, mask])

        np.testing.assert_allclose(masked, sliced, rtol=exact_rtol)

    def test_a_cell_nothing_was_measured_in_contributes_zero(self):
        """All-flagged cells make n1 n2 vanish; a nan there would spread over
        the whole channel's score."""
        z, n1, n2 = self._sums()
        z[:, 0] = 0.0
        n1[:, 0] = 0.0
        n2[:, 0] = 0.0

        r, z2 = coherence_scores(z, n1, n2)

        assert np.all(np.isfinite(np.asarray(r)))
        assert np.all(np.isfinite(np.asarray(z2)))
        np.testing.assert_allclose(np.asarray(r)[:, 0], 0.0)


class TestTauScan:
    """The scan itself: one jitted program over the whole grid_paths."""

    def test_it_recovers_the_injected_offset_on_the_injected_channel(self, obs,
                                                                     grid_paths):
        scan = tau_scan(obs.vis, obs.weights, grid_paths, obs.freqs)
        z2 = np.asarray(scan["z2"])

        assert z2.shape == (len(DEFAULT_TAU_GRID), N_FREQ)
        i_tau, i_chan = np.unravel_index(np.argmax(z2), z2.shape)
        assert DEFAULT_TAU_GRID[i_tau] == pytest.approx(TAU0, abs=0.25)
        assert i_chan == INJECTED_CHAN
        # The quiet channels never come near it: the fringe is narrowband and
        # the statistic has to say so.
        assert np.delete(z2, INJECTED_CHAN, axis=1).max() < 0.25 * z2.max()

    def test_the_scan_decays_away_from_the_peak(self, obs, grid_paths):
        """A single-peaked objective is the point: it is what makes the best
        cell a measurement rather than the largest of many sidelobes."""
        z2 = np.asarray(tau_scan(obs.vis, obs.weights, grid_paths, obs.freqs)["z2"])
        curve = z2.max(axis=1)
        peak = curve.max()

        for offset in (-3.0, 3.0):
            far = int(np.argmin(np.abs(DEFAULT_TAU_GRID - (TAU0 + offset))))
            assert curve[far] < 0.5 * peak
        at_zero = int(np.argmin(np.abs(DEFAULT_TAU_GRID)))
        assert curve[at_zero] < 0.5 * peak

    def test_r_is_a_per_frame_correlation_matching_a_transcription(self, obs,
                                                                    grid_paths,
                                                                    precision):
        window = slice(BEST_INDEX - 1, BEST_INDEX + 2)
        scan = tau_scan(obs.vis, obs.weights, grid_paths[window], obs.freqs)
        r = np.asarray(scan["r"])
        atol = 1e-8 if precision == "double" else 3e-3

        assert r.shape == (3, N_FREQ, obs.n_time)
        assert np.all((r >= 0.0) & (r <= 1.0 + 1e-5))
        assert r[1, INJECTED_CHAN].min() > 0.6
        for i, tau_index in enumerate(range(BEST_INDEX - 1, BEST_INDEX + 2)):
            want_r, want_z2 = ref_scores(obs.vis, obs.weights,
                                         ref_model(grid_paths[tau_index]))
            np.testing.assert_allclose(r[i], want_r, atol=atol)
            np.testing.assert_allclose(
                np.asarray(scan["z2"])[i], want_z2, atol=atol
            )

    def test_one_trace_covers_the_whole_grid(self, obs, grid_paths, exact_rtol):
        """The scan is a lax.map over the tau axis, not a Python loop.

        A loop over jitted kernels re-enters the compiler per step and gives up
        the whole point of the design: one compilation, then a GPU-resident
        sweep. New *values* of the same shape must hit the cache.
        """
        n_traces = 0

        def traced(vis, weights, paths, freqs):
            nonlocal n_traces
            n_traces += 1
            return tau_scan(vis, weights, paths, freqs)

        fn = jax.jit(traced)
        for scale in (1.0, 2.0, 3.0):
            jax.block_until_ready(
                fn(obs.vis * scale, obs.weights, grid_paths, obs.freqs)
            )

        assert n_traces == 1, f"recompiled {n_traces} times for one shape"

        jax.block_until_ready(fn(obs.vis, obs.weights, grid_paths[:5], obs.freqs))

        assert n_traces == 2, "a new grid_paths length must trace a new program"

        # And compiling it does not change what it computes.
        eager = tau_scan(obs.vis, obs.weights, grid_paths[:5], obs.freqs)
        jitted = jax.jit(tau_scan)(obs.vis, obs.weights, grid_paths[:5], obs.freqs)
        np.testing.assert_allclose(
            np.asarray(jitted["z2"]), np.asarray(eager["z2"]), rtol=exact_rtol
        )

    def test_it_vmaps_over_a_candidate_axis(self, obs, grid_paths, exact_rtol):
        """The contract the multi-satellite search (#191) is built on: the core
        is a pure function of fixed-shape arrays, so candidates batch."""
        stacked = np.stack([grid_paths[:4], grid_paths[4:8]])

        batched = jax.vmap(tau_scan, in_axes=(None, None, 0, None))(
            obs.vis, obs.weights, stacked, obs.freqs
        )

        assert np.asarray(batched["z2"]).shape == (2, 4, N_FREQ)
        for i, sub in enumerate((grid_paths[:4], grid_paths[4:8])):
            one = tau_scan(obs.vis, obs.weights, sub, obs.freqs)
            np.testing.assert_allclose(
                np.asarray(batched["z2"])[i], np.asarray(one["z2"]), rtol=exact_rtol
            )

    def test_zero_antenna_offsets_change_nothing(self, obs, grid_paths, exact_rtol):
        plain = tau_scan(obs.vis, obs.weights, grid_paths[:4], obs.freqs)
        offset = tau_scan(
            obs.vis, obs.weights, grid_paths[:4], obs.freqs,
            ant_offsets=np.zeros(N_ANT), a1=obs.a1, a2=obs.a2,
        )

        np.testing.assert_allclose(
            np.asarray(offset["z2"]), np.asarray(plain["z2"]), rtol=exact_rtol
        )

    def test_scrambling_the_antennas_destroys_the_peak(self, obs, grid_paths):
        """The null's mechanism, checked once on its own.

        Offsets of tens of metres are tens of wavelengths at 175 MHz, so each
        baseline enters with an unrelated phase and the coherent sum collapses
        to an incoherent one -- which is exactly what "no satellite there"
        should look like.
        """
        one_tau = grid_paths[BEST_INDEX:BEST_INDEX + 1]
        offsets = np.random.default_rng(7).uniform(0, 50.0, N_ANT)

        coherent = tau_scan(obs.vis, obs.weights, one_tau, obs.freqs)
        scrambled = tau_scan(
            obs.vis, obs.weights, one_tau, obs.freqs,
            ant_offsets=offsets, a1=obs.a1, a2=obs.a2,
        )

        assert float(np.asarray(scrambled["z2"]).max()) < 0.5 * float(
            np.asarray(coherent["z2"]).max()
        )

    def test_a_frame_mask_drops_those_frames_from_the_score(self, obs, grid_paths,
                                                            exact_rtol):
        mask = np.zeros(obs.n_time, dtype=bool)
        mask[: obs.n_time // 2] = True

        masked = tau_scan(obs.vis, obs.weights, grid_paths[:3], obs.freqs,
                          frame_mask=mask)
        sliced = tau_scan(obs.vis[:, :, mask], obs.weights,
                          grid_paths[:3][:, :, mask], obs.freqs)

        np.testing.assert_allclose(
            np.asarray(masked["z2"]), np.asarray(sliced["z2"]), rtol=exact_rtol
        )
        # r is per frame and says what each frame did; the mask decides only
        # which frames are combined, so the spectrogram keeps its full axis.
        assert np.asarray(masked["r"]).shape[-1] == obs.n_time


class TestDecoheredNull:
    """What the statistic looks like when the geometry is not real."""

    def test_it_returns_one_score_per_draw(self, obs, grid_paths):
        null = np.asarray(
            decohered_null(obs.vis, obs.weights, grid_paths[BEST_INDEX],
                           obs.freqs, obs.a1, obs.a2, n_draws=32)
        )

        assert null.shape == (32,)
        assert np.all(np.isfinite(null))
        assert np.all(null >= 0.0)

    def test_the_null_sits_far_below_the_coherent_score(self, obs, grid_paths):
        """The whole point of the comparison: a real trajectory scores where a
        scrambled one cannot reach."""
        null = np.asarray(
            decohered_null(obs.vis, obs.weights, grid_paths[BEST_INDEX],
                           obs.freqs, obs.a1, obs.a2, n_draws=200)
        )
        coherent = float(
            np.asarray(
                tau_scan(obs.vis, obs.weights, grid_paths[BEST_INDEX][None],
                         obs.freqs)["z2"]
            ).max()
        )

        assert null.mean() + 5.0 * null.std() < coherent

    def test_the_same_seed_draws_the_same_null(self, obs, grid_paths):
        """A significance that moved between two runs of the same command would
        not be a measurement."""
        kwargs = dict(n_draws=16, jitter_m=50.0)
        first = np.asarray(
            decohered_null(obs.vis, obs.weights, grid_paths[BEST_INDEX],
                           obs.freqs, obs.a1, obs.a2, seed=3, **kwargs)
        )
        again = np.asarray(
            decohered_null(obs.vis, obs.weights, grid_paths[BEST_INDEX],
                           obs.freqs, obs.a1, obs.a2, seed=3, **kwargs)
        )
        other = np.asarray(
            decohered_null(obs.vis, obs.weights, grid_paths[BEST_INDEX],
                           obs.freqs, obs.a1, obs.a2, seed=4, **kwargs)
        )

        np.testing.assert_array_equal(first, again)
        assert not np.allclose(first, other)


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------

class TestFitTimeOffset:
    """The single-satellite fit, end to end on the synthetic pass."""

    def test_it_recovers_the_injected_offset(self, default_fit):
        assert default_fit["tau_best"] == pytest.approx(TAU0, abs=0.25)
        assert default_fit["best_chan"] == INJECTED_CHAN
        assert default_fit["best_freq"] == pytest.approx(FREQS[INJECTED_CHAN])

    def test_the_default_grid_is_scanned_and_returned_per_channel(self,
                                                                   default_fit):
        """+-4 s in 0.25 s steps: wide enough for a day-old Starlink TLE, fine
        enough to resolve the peak the array's fringe rate gives it. The curve
        comes back with the answer, because the shape of the peak is how a
        detection is judged by eye."""
        grid = np.asarray(default_fit["tau_grid"])

        np.testing.assert_allclose(grid, DEFAULT_TAU_GRID)
        assert grid.shape == (33,)
        assert grid[0] == -4.0 and grid[-1] == 4.0
        assert 0.0 in grid and TAU0 in grid
        np.testing.assert_allclose(np.diff(grid), 0.25)

        z2 = np.asarray(default_fit["z2_tau"])
        assert z2.shape == (len(DEFAULT_TAU_GRID), N_FREQ)
        i_tau, i_chan = np.unravel_index(np.argmax(z2), z2.shape)
        assert np.asarray(default_fit["tau_grid"])[i_tau] == default_fit["tau_best"]
        assert i_chan == default_fit["best_chan"]
        assert default_fit["z2_best"] == pytest.approx(float(z2.max()))

    def test_the_detection_clears_the_decohered_null(self, default_fit):
        null = np.asarray(default_fit["null"])

        assert null.shape == (200,)
        assert default_fit["null_mean"] == pytest.approx(float(null.mean()))
        assert default_fit["null_std"] == pytest.approx(float(null.std()))
        assert default_fit["significance"] == pytest.approx(
            (default_fit["z2_best"] - null.mean()) / null.std(), rel=1e-6
        )
        assert default_fit["significance"] > 5.0
        # The fit measures; the threshold decides. A dict carrying the decision
        # would have to guess what the caller means by a detection.
        assert is_detection(default_fit) is True
        assert is_detection(default_fit, threshold_sigma=1e6) is False

    def test_the_spectrogram_keeps_the_full_time_axis(self, default_fit, obs):
        r = np.asarray(default_fit["r_best"])

        assert r.shape == (N_FREQ, obs.n_time)
        assert np.all(np.isfinite(r))
        assert np.all((r >= 0.0) & (r <= 1.0 + 1e-5))
        assert r[INJECTED_CHAN].min() > 0.6

    def test_an_overhead_pass_is_in_view_throughout(self, default_fit, obs):
        frames = np.asarray(default_fit["frames"])
        elevation = np.asarray(default_fit["elevation"])

        assert frames.dtype == bool
        assert frames.shape == (obs.n_time,) and frames.all()
        assert elevation.shape == (obs.n_time,)
        np.testing.assert_allclose(elevation, obs.elevation, atol=1e-6)

    def test_the_coherence_cut_drops_the_eight_kilometre_baselines(self,
                                                                   default_fit,
                                                                   obs):
        """The #189 selection, applied with the geometry the fit measured.

        With sigma_transverse = 300 m at this range the binding length is a few
        hundred metres: every core pair is inside it and every baseline to the
        far antenna is far outside, so 28 of the 36 non-auto baselines are used.
        """
        assert default_fit["n_bl_used"] == N_COHERENT_BL
        assert default_fit["n_bl_used"] == int(obs.keep.sum())
        assert default_fit["range_m"] == pytest.approx(obs.range_m, rel=0.01)
        assert default_fit["v_perp_m_s"] == pytest.approx(obs.v_perp, rel=0.1)
        assert default_fit["n_fine"] == N_FINE
        assert default_fit["sigma_transverse_m"] == SIGMA_TRANSVERSE
        # The cut is evaluated once, at the centre of the band.
        b_coh = float(
            np.minimum(
                C / np.mean(FREQS) * default_fit["range_m"]
                / (2 * np.pi * SIGMA_TRANSVERSE),
                C / np.mean(FREQS) * default_fit["range_m"] * N_FINE
                / (2 * INT_TIME * default_fit["v_perp_m_s"]),
            )
        )
        assert default_fit["b_coh"] == pytest.approx(b_coh, rel=1e-3)
        assert obs.bl_len[obs.keep].max() < default_fit["b_coh"] < 7000.0

    def test_soft_weights_find_the_same_offset(self, obs):
        """Down-weighting the marginal baselines rather than discarding them is
        the gentler cut when sigma is itself uncertain; it must not move the
        answer on data where the cut is not marginal."""
        fit = fit_time_offset(
            obs.vis, obs.record, obs.ants_itrf, obs.times_jd, obs.phase_centre,
            obs.freqs, obs.a1, obs.a2, obs.int_time, noise=SIGMA,
            taus_s=SHORT_GRID, soft_weights=True, n_null=32,
        )

        assert fit["tau_best"] == pytest.approx(TAU0, abs=0.25)
        assert fit["best_chan"] == INJECTED_CHAN
        assert fit["significance"] > 5.0

    def test_a_fully_flagged_baseline_only_costs_sensitivity(self, obs):
        """And a flagged sample never reaches the sums.

        A flagged visibility can be anything at all -- MSs carry inf and nan in
        them -- and ``0 * nan`` is nan, which would poison the whole channel's
        score rather than one baseline's contribution to it. The flagged
        samples here are nan on purpose.
        """
        flags = np.zeros(obs.vis.shape, dtype=bool)
        dropped = int(np.flatnonzero(obs.keep)[0])
        flags[dropped] = True
        vis = obs.vis.copy()
        vis[dropped] = np.nan

        fit = fit_time_offset(
            vis, obs.record, obs.ants_itrf, obs.times_jd, obs.phase_centre,
            obs.freqs, obs.a1, obs.a2, obs.int_time, noise=SIGMA, flags=flags,
            taus_s=SHORT_GRID, n_null=32,
        )

        assert fit["tau_best"] == pytest.approx(TAU0, abs=0.25)
        assert fit["best_chan"] == INJECTED_CHAN

    def test_pure_noise_does_not_clear_the_threshold(self, noise_obs):
        """The null is what stops a scan over 33 offsets and 4 channels from
        finding a satellite in every dataset it is pointed at.

        Fixed seeds throughout, so this is a statement about one realisation
        rather than about the tail of a distribution: the largest of the scan's
        cells sits a couple of sigma above its own null, not five.
        """
        fit = fit_time_offset(
            noise_obs.vis, noise_obs.record, noise_obs.ants_itrf,
            noise_obs.times_jd, noise_obs.phase_centre, noise_obs.freqs,
            noise_obs.a1, noise_obs.a2, noise_obs.int_time, noise=SIGMA,
            n_null=100,
        )

        assert fit["significance"] < 5.0
        assert is_detection(fit) is False
        assert fit["z2_best"] < 4.0


class TestHorizonMasking:
    """Frames the satellite is not up for are not evidence about it."""

    def test_the_window_is_the_frames_above_the_cut(self, setting_fit,
                                                     setting_obs):
        """And the frames outside it are nan in the spectrogram, not zero: a
        zero correlation is a measurement, and the diagnostic would show a dark
        band where nothing was ever looked at."""
        frames = np.asarray(setting_fit["frames"])

        np.testing.assert_array_equal(frames, setting_obs.frames)
        assert frames.any() and not frames.all()
        # Monotone: it sets once and does not come back inside the window.
        assert not np.any(np.diff(frames.astype(int)) > 0)

        r = np.asarray(setting_fit["r_best"])
        assert r.shape == (N_FREQ, setting_obs.n_time)
        assert np.all(np.isnan(r[:, ~frames]))
        assert np.all(np.isfinite(r[:, frames]))

    def test_the_cut_is_inclusive(self, setting_obs):
        """Same semantics as rfi.min_elevation: the cut is the lowest elevation
        still modelled, so a frame sitting exactly on it is in view."""
        exact = float(setting_obs.elevation[3])

        fit = fit_time_offset(
            setting_obs.vis, setting_obs.record, setting_obs.ants_itrf,
            setting_obs.times_jd, setting_obs.phase_centre, setting_obs.freqs,
            setting_obs.a1, setting_obs.a2, setting_obs.int_time, noise=SIGMA,
            taus_s=np.array([TAU0]), min_elevation=exact, n_null=8,
        )

        assert bool(np.asarray(fit["frames"])[3]) is True

    def test_it_still_recovers_the_offset_from_the_in_view_frames(self,
                                                                  setting_fit):
        assert setting_fit["tau_best"] == pytest.approx(TAU0, abs=0.25)
        assert setting_fit["best_chan"] == INJECTED_CHAN
        assert setting_fit["significance"] > 5.0

    def test_no_cut_keeps_every_frame(self, setting_obs):
        fit = fit_time_offset(
            setting_obs.vis, setting_obs.record, setting_obs.ants_itrf,
            setting_obs.times_jd, setting_obs.phase_centre, setting_obs.freqs,
            setting_obs.a1, setting_obs.a2, setting_obs.int_time, noise=SIGMA,
            taus_s=np.array([TAU0]), min_elevation=None, n_null=8,
        )

        assert np.asarray(fit["frames"]).all()
        assert np.all(np.isfinite(np.asarray(fit["r_best"])))

    def test_a_satellite_that_is_never_up_is_not_an_error(self, obs):
        """A pass that never happened is a result, not a failure: the batched
        search will meet plenty of them and must not stop on one."""
        fit = fit_time_offset(
            obs.vis, obs.record, obs.ants_itrf, obs.times_jd, obs.phase_centre,
            obs.freqs, obs.a1, obs.a2, obs.int_time, noise=SIGMA,
            taus_s=SHORT_GRID, min_elevation=95.0, n_null=8,
        )

        assert np.isnan(fit["tau_best"])
        assert fit["z2_best"] == 0.0
        assert np.isnan(fit["significance"])
        assert is_detection(fit) is False
        assert not np.asarray(fit["frames"]).any()
        assert np.all(np.isnan(np.asarray(fit["r_best"])))


class TestPhaseAtAFittedOffset:
    """Light curves extracted at the offset the scan measured."""

    def test_no_offset_is_the_behaviour_it_always_had(self, obs):
        """Bit-identical, not merely close: every existing caller passes None."""
        want = rfi_phase_from_records(
            [obs.record], obs.ants_itrf, obs.times_jd, obs.phase_centre, obs.freqs
        )
        got = rfi_phase_from_records(
            [obs.record], obs.ants_itrf, obs.times_jd, obs.phase_centre,
            obs.freqs, time_offsets_s=None,
        )

        assert np.array_equal(got, want)
        assert np.array_equal(
            rfi_phase_from_records(
                [obs.record], obs.ants_itrf, obs.times_jd, obs.phase_centre,
                obs.freqs, time_offsets_s=[0.0],
            ),
            want,
        )

    def test_an_offset_moves_the_satellite_and_only_the_satellite(self, obs):
        """The same convention the scan measures tau in.

        The antennas stay where the timestamps put them; only the orbit is
        re-evaluated. Re-propagating the whole geometry at ``t + tau`` is a
        different phase -- by most of a turn here -- so the two are compared
        rather than assumed equal.
        """
        shifted_sat = np.asarray(
            get_satellite_positions([obs.record], obs.times_jd + TAU0 / 86400.0)
        )
        want = rfi_phase_from_positions(
            shifted_sat, obs.ants_itrf, obs.times_jd, obs.phase_centre, obs.freqs
        )
        got = rfi_phase_from_records(
            [obs.record], obs.ants_itrf, obs.times_jd, obs.phase_centre,
            obs.freqs, time_offsets_s=[TAU0],
        )
        whole = rfi_phase_from_records(
            [obs.record], obs.ants_itrf, obs.times_jd + TAU0 / 86400.0,
            obs.phase_centre, obs.freqs,
        )

        np.testing.assert_allclose(got, want, atol=1e-9)
        assert np.abs(np.exp(1j * got) - np.exp(1j * whole)).max() > 0.1

    def test_the_offsets_are_per_source(self, obs):
        """One tau per satellite: the search fits them independently."""
        decoy = make_tle_record(DECOY_ID, EPOCH_JD)
        records = [obs.record, decoy]

        got = rfi_phase_from_records(
            records, obs.ants_itrf, obs.times_jd, obs.phase_centre, obs.freqs,
            time_offsets_s=[0.0, TAU0],
        )
        first = rfi_phase_from_records(
            [obs.record], obs.ants_itrf, obs.times_jd, obs.phase_centre, obs.freqs
        )
        second = rfi_phase_from_positions(
            np.asarray(
                get_satellite_positions([decoy], obs.times_jd + TAU0 / 86400.0)
            ),
            obs.ants_itrf, obs.times_jd, obs.phase_centre, obs.freqs,
        )

        np.testing.assert_allclose(got[0], first[0], atol=1e-9)
        np.testing.assert_allclose(got[1], second[0], atol=1e-9)


# ---------------------------------------------------------------------------
# Carrying the fit through the light-curve result
# ---------------------------------------------------------------------------

def fake_fit(tau, significance, n_tau=5, n_freq=N_FREQ, n_time=6, chan=1):
    """A fit-shaped dict, so the plumbing is tested without a second scan."""
    rng = np.random.default_rng(int(abs(tau) * 10) + 1)

    return {
        "tau_grid": np.linspace(-1.0, 1.0, n_tau),
        "z2_tau": rng.uniform(0, 1, (n_tau, n_freq)),
        "tau_best": float(tau),
        "z2_best": float(significance),
        "best_chan": int(chan),
        "best_freq": float(FREQS[chan]),
        "r_best": rng.uniform(0, 1, (n_freq, n_time)),
        "frames": np.ones(n_time, dtype=bool),
        "elevation": np.full(n_time, 45.0),
        "null": rng.uniform(0, 1, 16),
        "null_mean": 0.5,
        "null_std": 0.1,
        "significance": float(significance),
        "n_bl_used": N_COHERENT_BL,
        "range_m": 350e3,
        "v_perp_m_s": 7.4e3,
        "b_coh": 400.0,
        "n_fine": N_FINE,
        "sigma_transverse_m": SIGMA_TRANSVERSE,
    }


@pytest.fixture
def curves():
    """A two-source light-curve result, as the drivers build one."""
    n_src, n_time = 2, 6
    rng = np.random.default_rng(11)
    light = rng.normal(size=(n_src, N_FREQ, n_time)) + 1j * rng.normal(
        size=(n_src, N_FREQ, n_time)
    )
    error = np.full((n_src, N_FREQ, n_time), 0.1)

    return _lc_result(
        light, error, [NORAD_ID, DECOY_ID], FREQS,
        jd_to_mjd(EPOCH_JD) + np.arange(n_time) * INT_TIME / 86400.0,
        "DATA", "xx", in_view=np.ones((n_src, n_time), dtype=bool),
    )


class TestOffsetFitPlumbing:
    """The fit has to reach the file, or the next run cannot consume it."""

    def test_the_fit_arrays_are_attached_along_the_source_axis(self, curves):
        fits = [fake_fit(TAU0, 8.0, chan=INJECTED_CHAN), fake_fit(-0.5, 1.2)]

        result = attach_offset_fits(curves, fits, 5.0)

        np.testing.assert_allclose(result["tau_best"], [TAU0, -0.5])
        np.testing.assert_allclose(result["significance"], [8.0, 1.2])
        np.testing.assert_allclose(result["tau_grid"], fits[0]["tau_grid"])
        assert np.asarray(result["z2_tau"]).shape == (2, 5, N_FREQ)
        assert np.asarray(result["r_best"]).shape == (2, N_FREQ, 6)
        np.testing.assert_array_equal(result["best_chan"], [INJECTED_CHAN, 1])
        np.testing.assert_allclose(result["null_mean"], [0.5, 0.5])
        np.testing.assert_allclose(result["null_std"], [0.1, 0.1])
        assert result["offset_threshold_sigma"] == 5.0

    def test_the_threshold_decides_which_sources_are_detections(self, curves):
        fits = [fake_fit(TAU0, 8.0), fake_fit(-0.5, 1.2)]

        detected = np.asarray(attach_offset_fits(curves, fits, 5.0)["detected"])
        strict = np.asarray(attach_offset_fits(curves, fits, 20.0)["detected"])

        assert detected.dtype == bool
        np.testing.assert_array_equal(detected, [True, False])
        np.testing.assert_array_equal(strict, [False, False])

    def test_the_file_carries_the_fit_beside_the_curves(self, tmp_path, curves):
        """The output artifact has to record the offset it was measured at, or
        a later run cannot reproduce the trajectory that produced it."""
        result = attach_offset_fits(curves, [fake_fit(TAU0, 8.0),
                                             fake_fit(-0.5, 1.2)], 5.0)
        path = str(tmp_path / "curves.npz")

        save_light_curves_npz(path, result)

        with np.load(path, allow_pickle=False) as npz:
            np.testing.assert_allclose(npz["tau_best"], [TAU0, -0.5])
            np.testing.assert_allclose(npz["tau_grid"], result["tau_grid"])
            np.testing.assert_allclose(npz["significance"], [8.0, 1.2])
            np.testing.assert_array_equal(npz["detected"], [True, False])
            assert npz["z2_tau"].shape == (2, 5, N_FREQ)
            assert npz["z2_best"].shape == (2,)
            assert npz["best_chan"].shape == (2,)
            assert npz["null_mean"].shape == npz["null_std"].shape == (2,)
            # Swapped into the file's (n_src, n_time, n_freq) orientation, the
            # same way the curves are.
            assert npz["r_best"].shape == (2, 6, N_FREQ)
            np.testing.assert_allclose(
                npz["r_best"], np.swapaxes(result["r_best"], 1, 2)
            )

        # And it is still the rfi.est interchange format: the four required
        # names are untouched, so a later run can be seeded from it unchanged.
        read = read_light_curves(
            path, [NORAD_ID, DECOY_ID], result["times_mjd_utc"], FREQS
        )
        assert np.asarray(read).shape == (2, N_FREQ, 6)

    def test_selecting_sources_keeps_the_chosen_rows_everywhere(self, curves):
        """Threshold-gated saving is a selection along the source axis; a
        per-source array left behind would mislabel every curve after it."""
        result = attach_offset_fits(curves, [fake_fit(TAU0, 8.0),
                                             fake_fit(-0.5, 1.2)], 5.0)

        kept = select_sources(result, np.array([True, False]))

        assert kept["norad_ids"] == [NORAD_ID]
        assert kept["titles"] == [str(NORAD_ID)]
        assert np.asarray(kept["light_curves"]).shape == (1, N_FREQ, 6)
        assert np.asarray(kept["error"]).shape == (1, N_FREQ, 6)
        assert np.asarray(kept["z"]).shape == (1, N_FREQ, 6)
        assert np.asarray(kept["in_view"]).shape == (1, 6)
        np.testing.assert_allclose(kept["tau_best"], [TAU0])
        assert np.asarray(kept["z2_tau"]).shape == (1, 5, N_FREQ)
        assert np.asarray(kept["r_best"]).shape == (1, N_FREQ, 6)
        np.testing.assert_array_equal(kept["detected"], [True])
        # The coordinates are not per source and must survive intact.
        np.testing.assert_allclose(kept["freqs"], FREQS)
        np.testing.assert_allclose(kept["times_mjd_utc"], result["times_mjd_utc"])

        # Indices select the same way a mask does.
        by_index = select_sources(result, np.array([1]))
        assert by_index["norad_ids"] == [DECOY_ID]
        np.testing.assert_allclose(by_index["tau_best"], [-0.5])


# ---------------------------------------------------------------------------
# Consuming the result: an epoch-shifted orbit record
# ---------------------------------------------------------------------------

class TestShiftedOrbitRecords:
    """The zero-code-change way to use a fitted tau: move the orbit's epoch."""

    def test_a_tle_epoch_moves_by_minus_tau(self, record):
        """Minus, because the record has to *become* the trajectory the data
        showed: propagating the shifted elements at t must give what the
        original gave at t + tau.

        The ``EPOCH`` text column moves with the line, since a record whose
        column disagreed with its own line 1 would be read one way by the age
        policy and another by the propagator.
        """
        from satchecker_client.records import parse_omm_epoch_jd

        shifted = shift_orbit_record_epoch(record, TAU0)

        moved = (record_epoch_jd(shifted) - record_epoch_jd(record)) * 86400.0

        assert moved == pytest.approx(-TAU0, abs=1e-3)
        assert parse_omm_epoch_jd(shifted["EPOCH"]) == pytest.approx(
            record_epoch_jd(shifted), abs=1e-3 / 86400.0
        )

    def test_the_shifted_lines_still_validate(self, record):
        """A rewritten epoch field invalidates the modulo-10 checksum, and the
        parser rejects a bad one -- as it should, since that is how a
        single-character corruption is caught."""
        shifted = shift_orbit_record_epoch(record, -2.25)

        assert validate_tle_pair(
            shifted["TLE_LINE1"], shifted["TLE_LINE2"]
        ) == NORAD_ID

    def test_it_reproduces_the_original_at_t_plus_tau(self, record, layout):
        """The sign convention, checked where it can be seen: a positive tau
        means the TLE is late, so the satellite is where the elements say it
        will be tau seconds later."""
        shifted = shift_orbit_record_epoch(record, TAU0)

        got = np.asarray(get_satellite_positions([shifted], layout.times_jd))
        want = np.asarray(
            get_satellite_positions([record], layout.times_jd + TAU0 / 86400.0)
        )

        unshifted = np.asarray(
            get_satellite_positions([record], layout.times_jd)
        )
        # The line-1 epoch field quantises to 0.86 ms, which is ~7 m of track.
        assert np.abs(got - want).max() < 10.0
        # And it is a real shift, not a no-op: 1.5 s is 11 km of orbit.
        assert np.abs(got - unshifted).max() > 1e3
        # While tau = 0 leaves the trajectory exactly where it was.
        np.testing.assert_allclose(
            np.asarray(
                get_satellite_positions(
                    [shift_orbit_record_epoch(record, 0.0)], layout.times_jd
                )
            ),
            unshifted,
            atol=1e-3,
        )

    def test_an_omm_epoch_moves_too_and_more_exactly(self, layout):
        """An OMM has no fixed-width field to quantise into.

        What is left is the resolution of a Julian Date itself, ~40 us at this
        epoch, rather than the TLE field's 0.86 ms -- two orders of magnitude
        less along-track error, which is why this is the format to shift when
        there is a choice.
        """
        omm = make_omm(NORAD_ID, EPOCH_JD)

        shifted = shift_orbit_record_epoch(omm, TAU0)

        assert (record_epoch_jd(shifted) - record_epoch_jd(omm)) * 86400.0 == (
            pytest.approx(-TAU0, abs=1e-4)
        )
        got = np.asarray(get_satellite_positions([shifted], layout.times_jd))
        want = np.asarray(
            get_satellite_positions([omm], layout.times_jd + TAU0 / 86400.0)
        )
        assert np.abs(got - want).max() < 0.05

    def test_everything_but_the_epoch_is_left_alone(self, record):
        """It is the same satellite, with the same elements, read at another
        instant -- so nothing else in the record may move."""
        original = dict(record)

        shifted = shift_orbit_record_epoch(record, TAU0)

        assert shifted["NORAD_CAT_ID"] == record["NORAD_CAT_ID"]
        assert shifted["OBJECT_NAME"] == record["OBJECT_NAME"]
        assert shifted["TLE_LINE2"] == record["TLE_LINE2"]
        # Line 1 outside the epoch field (columns 19-32) and the checksum.
        assert shifted["TLE_LINE1"][:18] == record["TLE_LINE1"][:18]
        assert shifted["TLE_LINE1"][32:68] == record["TLE_LINE1"][32:68]
        # And the record handed in is not edited under the caller.
        assert record == original

    def test_the_written_file_resolves_as_an_extra_orbit_dir(self, tmp_path,
                                                             record, layout):
        """The point of writing it: a later run picks it up with
        --extra-orbit-dir and reproduces this trajectory, no code changes."""
        directory = str(tmp_path / "shifted")

        path = write_shifted_orbits(directory, [NORAD_ID], [record], [TAU0])

        assert os.path.exists(path)
        resolved, _ = _select_from_extra_dir(
            directory, {NORAD_ID}, layout.t_ref, None
        )
        assert NORAD_ID in resolved
        got = np.asarray(
            get_satellite_positions([resolved[NORAD_ID].record], layout.times_jd)
        )
        want = np.asarray(
            get_satellite_positions([record], layout.times_jd + TAU0 / 86400.0)
        )
        assert np.abs(got - want).max() < 10.0

    @pytest.mark.parametrize(
        "tau", [-3600.0, -1.5, -0.001, 0.0, 0.001, 1.5, 3600.0]
    )
    def test_the_shift_is_exact_for_either_sign_and_size(self, record, layout,
                                                          tau):
        """Both signs, from a millisecond to an hour.

        A negative tau -- elements running *ahead* of the satellite -- is as
        ordinary as a positive one, and the epoch has to move the other way for
        it. The equivalence is exact in the propagator, which reads only
        ``t - epoch``; what is left is the field's 1e-8 day quantisation (~7 m
        of track) and the slow drift of the frame the position comes back in,
        which an hour cannot spend more than a metre or two of.
        """
        shifted = shift_orbit_record_epoch(record, tau)

        assert validate_tle_pair(
            shifted["TLE_LINE1"], shifted["TLE_LINE2"]
        ) == NORAD_ID
        assert record_epoch_jd(shifted) == pytest.approx(
            record_epoch_jd(record) - tau / 86400.0, abs=TLE_EPOCH_QUANTUM_DAYS
        )
        got = np.asarray(get_satellite_positions([shifted], layout.times_jd))
        want = np.asarray(
            get_satellite_positions([record], layout.times_jd + tau / 86400.0)
        )
        assert np.abs(got - want).max() < 10.0


class TestShiftedEpochAcrossAYearBoundary:
    """The epoch field is ``YYDDD.DDDDDDDD``, and DDD rolls over before YY does.

    The year and the day-of-year are two halves of one number, so they have to
    come from the *same* instant at the *same* precision. Taking the year from
    the unrounded epoch while rounding the day to the field's eight decimals
    lets them disagree within half a quantum (0.43 ms) of a New Year: the day
    rounds up to ``366.00000000`` -- or ``367.00000000`` after a leap year --
    while the year still reads the old one. That is not a date (2025 has no day
    366), and the parser rejects the pair outright, so the file written for a
    later run cannot be read back at all.

    Nothing about this is exotic: elements issued in the first seconds of a year
    and a fitted offset of a couple of seconds is all it takes.
    """

    @pytest.mark.parametrize(
        "boundary_jd, offset_ms, expected_yy",
        [
            pytest.param(AFTER_A_COMMON_YEAR_JD, 0.8, 26,
                         id="just_after_a_common_year"),
            pytest.param(AFTER_A_COMMON_YEAR_JD, -0.8, 25,
                         id="just_before_a_common_year"),
            # Inside half a quantum of the boundary, where the day rounds up and
            # the year has to follow it. Either encoding of that instant is
            # right -- the new year's day 1, or the old year's last
            # representable day -- so only the legality and the decoded epoch
            # are pinned, not which of the two is chosen.
            pytest.param(AFTER_A_COMMON_YEAR_JD, -0.2, None,
                         id="rounding_edge_common_year"),
            pytest.param(AFTER_A_LEAP_YEAR_JD, 0.8, 25,
                         id="just_after_a_leap_year"),
            pytest.param(AFTER_A_LEAP_YEAR_JD, -0.2, None,
                         id="rounding_edge_leap_year"),
        ],
    )
    def test_a_shift_onto_a_boundary_stays_a_date(self, boundary_jd, offset_ms,
                                                   expected_yy):
        from satchecker_client.records import parse_omm_epoch_jd

        # Elements issued two seconds into the new year -- an ordinary epoch,
        # encodable exactly -- which the fitted offset then walks onto the
        # boundary.
        record = make_tle_record(NORAD_ID, boundary_jd + 2.0 / 86400.0)
        target_jd = boundary_jd + offset_ms * 1e-3 / 86400.0
        tau = (record_epoch_jd(record) - target_jd) * 86400.0

        shifted = shift_orbit_record_epoch(record, tau)

        line1 = shifted["TLE_LINE1"]
        year = 2000 + int(line1[18:20])
        day = float(line1[20:32])
        # What tle_epoch_jd demands, and the whole of the bug: a day-of-year its
        # own year cannot hold is not a date.
        assert 0.0 < day < _days_in_year(year) + 1.0, (
            f"encoded {line1[18:32]!r}, and {year} has {_days_in_year(year)} days"
        )
        assert validate_tle_pair(line1, shifted["TLE_LINE2"]) == NORAD_ID
        assert record_epoch_jd(shifted) == pytest.approx(
            target_jd, abs=TLE_EPOCH_QUANTUM_DAYS
        )
        # The EPOCH column and the lines must name the same instant, or the age
        # policy reads one date and the propagator another.
        assert parse_omm_epoch_jd(shifted["EPOCH"]) == pytest.approx(
            record_epoch_jd(shifted), abs=TLE_EPOCH_QUANTUM_DAYS
        )
        if expected_yy is not None:
            assert int(line1[18:20]) == expected_yy


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------

def _cli(*argv):
    from tabascal.scripts.run_tabascal import build_parser

    return build_parser().parse_args(["light-curve", *argv])


@pytest.fixture
def stub_ms(monkeypatch, obs):
    """Stand in for ``tabascal.ms.read_ms`` with the synthetic observation."""
    import tabascal.ms

    gh0 = (gast_deg(obs.times_jd) - obs.phase_centre["ra"]) % 360
    ants_uvw = itrf_to_uvw_numpy(obs.ants_itrf, gh0, obs.phase_centre["dec"])
    times_mjd = jd_to_mjd(obs.times_jd)
    data = {
        "ra": obs.phase_centre["ra"],
        "dec": obs.phase_centre["dec"],
        "ants_itrf": obs.ants_itrf,
        "times_mjd": times_mjd,
        "times_jd": obs.times_jd,
        "time_scale": "utc",
        "freqs": obs.freqs,
        "chan_width": float(obs.freqs[1] - obs.freqs[0]),
        "chan_widths": np.full(N_FREQ, float(obs.freqs[1] - obs.freqs[0])),
        "vis_obs": obs.vis,
        "n_freq": N_FREQ,
        "n_time": obs.n_time,
        "n_bl": obs.n_bl,
        "n_ant": N_ANT,
        "int_time": obs.int_time,
        "uvw": ants_uvw[:, obs.a1] - ants_uvw[:, obs.a2],
        "flags": np.zeros(obs.vis.shape, dtype=bool),
        "noise": SIGMA,
        "a1": obs.a1,
        "a2": obs.a2,
    }
    monkeypatch.setattr(tabascal.ms, "read_ms", lambda *a, **k: data)

    return data


@pytest.fixture
def stub_orbits(monkeypatch):
    """Resolve NORAD ids offline: the real satellite, and a decoy elsewhere."""
    import tabascal.rfi_estimate as mod

    def _fetch(times_jd=None, norad_ids=None, **kwargs):
        ids = [int(n) for n in norad_ids]
        records = [
            make_tle_record(
                n, EPOCH_JD + (0.0 if n == NORAD_ID else DECOY_EPOCH_SHIFT_S / 86400.0)
            )
            for n in ids
        ]
        return None, None, ids, records, len(ids)

    monkeypatch.setattr(mod, "fetch_orbital_elements", _fetch)


@pytest.fixture
def stub_config_mode(monkeypatch, obs, record, tmp_path):
    """The ``-c`` path, with the config and the TabConfig stood in for.

    A real TabConfig needs a Measurement Set and a live orbit resolution, so it
    is stubbed with the observation this file already built -- exactly the
    attributes the estimator reads off one. ``set_precision`` is stubbed too:
    it flips ``jax_enable_x64`` for the whole session, and the suite is run in
    both precisions from the outside.
    """
    import tabascal.config
    import tabascal.scripts._run_tabascal_impl as impl

    config = {
        "data": {"ms_path": str(tmp_path / "obs.ms"), "out_dir": None,
                 "data_col": "DATA", "corr": "xx", "freq": None},
        "rfi": {"min_elevation": 0},
        "satellites": {},
        "model": {"precision": "double"},
    }
    tab_config = SimpleNamespace(
        vis_obs=obs.vis,
        flags=np.zeros(obs.vis.shape, dtype=bool),
        noise=SIGMA,
        ants_itrf=obs.ants_itrf,
        times_jd=obs.times_jd,
        times_mjd=jd_to_mjd(obs.times_jd),
        time_scale="utc",
        freqs=obs.freqs,
        int_time=obs.int_time,
        phase_centre=dict(obs.phase_centre),
        a1=obs.a1,
        a2=obs.a2,
        norad_ids=[NORAD_ID],
        n_rfi_real=1,
        orbit_records=[record],
        min_elevation=0.0,
        rfi_mask=None,
        args={"data": {"data_col": "DATA", "corr": "xx"}},
    )

    monkeypatch.setattr(tabascal.config, "load_config", lambda path: config)
    monkeypatch.setattr(
        tabascal.config, "TabConfig", lambda cfg, ms_path, **kw: tab_config
    )
    monkeypatch.setattr(impl, "set_precision", lambda cfg: None)

    return config


FAST_FIT = ("--tau-max", "2.0", "--tau-step", "0.5", "--null-draws", "40")


class TestOffsetFitArguments:
    """``light-curve --fit-offset`` and the flags that shape the scan."""

    def test_the_defaults_are_the_issues_defaults(self):
        args = _cli("-ms", "obs.ms", "-n", "25544")

        assert args.fit_offset is False
        assert args.tau_max == 4.0
        assert args.tau_step == 0.25
        assert args.n_fine == 40
        assert args.sigma_transverse == 300.0
        assert args.soft_weights is False
        assert args.null_draws == 200
        assert args.null_jitter == 50.0
        assert args.threshold == 5.0
        assert args.only_detections is False
        assert args.write_shifted_tle is None

    def test_every_flag_is_accepted(self, tmp_path):
        args = _cli(
            "-ms", "obs.ms", "-n", "25544", "--fit-offset", "--tau-max", "6",
            "--tau-step", "0.5", "--n-fine", "16", "--sigma-transverse", "150",
            "--soft-weights", "--null-draws", "50", "--null-jitter", "80",
            "--threshold", "7.5", "--only-detections",
            "--write-shifted-tle", str(tmp_path),
        )

        assert args.fit_offset is True
        assert (args.tau_max, args.tau_step) == (6.0, 0.5)
        assert (args.n_fine, args.sigma_transverse) == (16, 150.0)
        assert args.soft_weights is True
        assert (args.null_draws, args.null_jitter) == (50, 80.0)
        assert args.threshold == 7.5
        assert args.only_detections is True
        assert args.write_shifted_tle == str(tmp_path)

    def test_the_flags_come_from_a_shared_builder(self):
        """#191's ``search`` subcommand registers the same options, so they are
        added by one function rather than spelled out twice."""
        import argparse

        from tabascal.scripts.rfi_estimate import add_offset_fit_arguments

        parser = argparse.ArgumentParser()
        add_offset_fit_arguments(parser)
        args = parser.parse_args(["--fit-offset", "--tau-max", "3"])

        assert args.fit_offset is True and args.tau_max == 3.0

    def test_a_scan_flag_without_the_scan_is_an_error(self, tmp_path, stub_ms,
                                                      stub_orbits):
        """Silently ignoring --tau-max would report a default-grid fit as the
        one that was asked for."""
        from tabascal.scripts.rfi_estimate import run

        with pytest.raises(SystemExit, match="--fit-offset"):
            run(_cli("-ms", str(tmp_path / "obs.ms"), "-n", "25544",
                     "--tau-max", "2.0", "-o", str(tmp_path / "c.npz")))

    def test_the_parser_module_still_imports_without_jax(self):
        """``tabascal -h`` must not pay for the run stack, so the scan's heavy
        imports stay inside the functions that need them."""
        import tabascal.scripts.rfi_estimate as script

        with open(script.__file__) as fh:
            source = fh.read()

        assert not re.search(r"(?m)^(import jax|from jax)", source)

    @pytest.mark.parametrize(
        "flag, value",
        [
            ("--tau-step", "0"),
            ("--tau-step", "-0.25"),
            ("--tau-max", "-1.0"),
            ("--n-fine", "0"),
            ("--n-fine", "-4"),
            ("--null-draws", "1"),
            ("--null-jitter", "0"),
            ("--null-jitter", "-50"),
            ("--tau-max", "nan"),
            ("--tau-step", "inf"),
            ("--sigma-transverse", "nan"),
            ("--null-jitter", "inf"),
            ("--threshold", "nan"),
        ],
    )
    def test_a_nonsense_scan_setting_is_refused_by_name(self, flag, value):
        """argparse types a value; it cannot say whether it means anything.

        Every one of these produces a scan that is not a scan -- an empty or
        infinite grid, a model with no sub-steps, a null with no spread to
        measure a significance against, a nan that quietly poisons the whole
        statistic -- and each fails somewhere far from the flag that caused it.
        Refusing it here, by name, is the difference between a typo and a
        mystery.

        A null of one draw is refused with the rest: its standard deviation is
        zero, so the significance is an infinity or a nan rather than a number.
        """
        from tabascal.scripts.rfi_estimate import resolve_offset_fit

        args = _cli("-ms", "x.ms", "-n", "1", "--fit-offset", flag, value)

        with pytest.raises(SystemExit, match=re.escape(flag)):
            resolve_offset_fit(args)

    def test_the_degenerate_but_meaningful_settings_are_allowed(self):
        """``--tau-max 0`` is a grid of one point at tau = 0 -- the honest way
        to ask for the statistic without a scan -- and ``--n-fine 1`` is the
        unsmeared template. Neither is nonsense, so neither is refused."""
        from tabascal.scripts.rfi_estimate import resolve_offset_fit

        flat = resolve_offset_fit(
            _cli("-ms", "x.ms", "-n", "1", "--fit-offset", "--tau-max", "0")
        )
        coarse = resolve_offset_fit(
            _cli("-ms", "x.ms", "-n", "1", "--fit-offset", "--n-fine", "1")
        )

        assert len(np.asarray(flat["taus_s"])) == 1
        assert float(np.asarray(flat["taus_s"])[0]) == pytest.approx(0.0)
        assert coarse["n_fine"] == 1

    @pytest.mark.parametrize(
        "tau_max, tau_step, want",
        [
            pytest.param("4", "0.25", DEFAULT_TAU_GRID, id="the_documented_grid"),
            pytest.param("4", "3", [-3.0, 0.0, 3.0],
                         id="a_step_that_does_not_divide"),
            pytest.param("0", "0.25", [0.0], id="no_half_width_at_all"),
            pytest.param("1", "0.4", [-0.8, -0.4, 0.0, 0.4, 0.8],
                         id="a_step_that_overshoots"),
        ],
    )
    def test_the_grid_is_built_from_the_step_out_to_the_half_width(
        self, tau_max, tau_step, want
    ):
        """Whole steps either side of zero, and never past ``--tau-max``.

        The scan measures a *correction*, so tau = 0 -- the trajectory as the
        elements give it -- has to be one of the points: it is the reference the
        peak is read against, and what the case study quotes its rise from
        (0.045 at tau = 0 to 0.107 at -2.25 s). Marching from ``-tau_max`` in
        steps that do not divide it loses that point entirely
        (``--tau-max 4 --tau-step 3`` gives -4, -1, 2, 5) and puts a sample
        beyond the half-width the caller asked for. Counting whole steps out
        from zero, ``n = floor(tau_max / tau_step)``, keeps the grid symmetric,
        centred and inside its own bounds, at the cost of a shorter reach when
        the step does not divide -- which is the honest reading of both flags.
        """
        from tabascal.scripts.rfi_estimate import resolve_offset_fit

        args = _cli("-ms", "x.ms", "-n", "1", "--fit-offset",
                    "--tau-max", tau_max, "--tau-step", tau_step)

        taus = np.asarray(resolve_offset_fit(args)["taus_s"])

        np.testing.assert_allclose(taus, np.asarray(want, dtype=float), atol=1e-12)
        # Symmetric about zero, with zero itself exactly on the grid.
        np.testing.assert_allclose(taus, -taus[::-1], atol=1e-12)
        assert int(np.sum(taus == 0.0)) == 1
        assert np.abs(taus).max() <= float(tau_max) + 1e-12

    def test_a_grid_of_millions_of_points_is_refused_by_name(self):
        """A step small enough to be a typo is not a scan, it is an allocation.

        The peak is a fraction of a second wide, so a grid finer than that buys
        nothing, and a microsecond step over +-4 s asks for eight million
        near-identical near-field models -- caught at the flag, before the array
        exists. A grid that is merely fine is left alone.
        """
        from tabascal.scripts.rfi_estimate import resolve_offset_fit

        with pytest.raises(SystemExit, match=re.escape("--tau-step")):
            resolve_offset_fit(
                _cli("-ms", "x.ms", "-n", "1", "--fit-offset",
                     "--tau-max", "4", "--tau-step", "1e-6")
            )

        fine = resolve_offset_fit(
            _cli("-ms", "x.ms", "-n", "1", "--fit-offset",
                 "--tau-max", "4", "--tau-step", "0.01")
        )

        assert len(np.asarray(fine["taus_s"])) == 801

    def test_the_precision_of_the_scan_can_be_named(self):
        """The scan is its own piece of numerics and gets its own switch.

        The fringe model is a phase on a path difference of a few km, which f32
        holds to ~1e-3 rad, so single precision is enough and is what the
        standalone path uses. Saying so explicitly is for the cases where it is
        not -- a long baseline, a high frequency -- and it has to be one of the
        two names tabascal already uses, not any string at all.
        """
        assert _cli("-ms", "obs.ms", "-n", "25544").precision is None

        for name in ("single", "double"):
            args = _cli("-ms", "obs.ms", "-n", "25544", "--fit-offset",
                        "--precision", name)
            assert args.precision == name

        with pytest.raises(SystemExit):
            _cli("-ms", "obs.ms", "-n", "25544", "--fit-offset",
                 "--precision", "float32")

    def test_the_standalone_scan_defaults_to_single_precision(self):
        """With ``-ms`` there is no config to ask, and tabascal's own default is
        ``model.precision: single`` -- so the scan follows it rather than
        whatever the interpreter happens to have been left in."""
        from tabascal.scripts.rfi_estimate import resolve_offset_fit

        default = resolve_offset_fit(
            _cli("-ms", "x.ms", "-n", "1", "--fit-offset")
        )
        named = resolve_offset_fit(
            _cli("-ms", "x.ms", "-n", "1", "--fit-offset", "--precision", "double")
        )

        assert default["precision"] == "single"
        assert named["precision"] == "double"


class TestCommandLine:
    """The scan as it is actually run: from parsed arguments to files on disk."""

    def test_it_fits_saves_and_reports_the_offset(self, tmp_path, stub_ms,
                                                  stub_orbits, capsys):
        """One line per satellite, ending in DETECTED or "not detected"."""
        from tabascal.scripts.rfi_estimate import run

        out = str(tmp_path / "curves.npz")
        tle_dir = str(tmp_path / "tles")

        run(_cli("-ms", str(tmp_path / "obs.ms"), "-n", str(NORAD_ID),
                 "--fit-offset", *FAST_FIT, "-o", out,
                 "--write-shifted-tle", tle_dir, "-p"))

        with np.load(out, allow_pickle=False) as npz:
            np.testing.assert_array_equal(npz["norad_ids"], [NORAD_ID])
            assert npz["tau_best"][0] == pytest.approx(TAU0, abs=0.5)
            assert bool(npz["detected"][0]) is True
            assert npz["light_curves"].shape == (1, N_TIME, N_FREQ)
            # The curves are extracted at the fitted offset, not at tau = 0:
            # the fit is not a diagnostic printed and thrown away. De-rotated
            # with the right trajectory the injected channel comes back at its
            # injected amplitude; at tau = 0 it would average incoherently to
            # the noise floor, ~sigma / sqrt(n_bl).
            assert npz["light_curves"][0, :, INJECTED_CHAN].mean() > 1.0
        printed = capsys.readouterr().out
        assert str(NORAD_ID) in printed
        assert "DETECTED" in printed
        # The scan curve, the per-channel score and the spectrogram.
        png = tmp_path / f"curves_offset_{NORAD_ID}.png"
        assert png.exists() and png.stat().st_size > 0
        # And the shifted record is written where a later run can find it.
        resolved, _ = _select_from_extra_dir(
            tle_dir, {NORAD_ID}, float(np.mean(stub_ms["times_jd"])), None
        )
        assert NORAD_ID in resolved

    def test_without_the_scan_nothing_is_fitted(self, tmp_path, stub_ms,
                                                stub_orbits):
        """--fit-offset is opt-in: the default light-curve run is unchanged."""
        from tabascal.scripts.rfi_estimate import run

        out = str(tmp_path / "curves.npz")
        run(_cli("-ms", str(tmp_path / "obs.ms"), "-n", str(NORAD_ID), "-o", out))

        with np.load(out, allow_pickle=False) as npz:
            assert "tau_best" not in npz.files
            assert npz["light_curves"].shape == (1, N_TIME, N_FREQ)

    def test_an_undetected_satellite_is_still_reported(self, tmp_path, stub_ms,
                                                       stub_orbits, capsys):
        """Both are saved without the gate, and the file says which was found.

        The decoy is the same TLE half a minute along its own orbit: still well
        above the horizon, so it is the statistic that rejects it and not the
        elevation cut. Its summary line reads "not detected" -- the marker is
        lower case, so that the upper-case DETECTED can be looked for on its
        own.
        """
        from tabascal.scripts.rfi_estimate import run

        out = str(tmp_path / "curves.npz")
        run(_cli("-ms", str(tmp_path / "obs.ms"), "-n",
                 f"{NORAD_ID},{DECOY_ID}", "--fit-offset", *FAST_FIT, "-o", out))

        with np.load(out, allow_pickle=False) as npz:
            np.testing.assert_array_equal(npz["norad_ids"], [NORAD_ID, DECOY_ID])
            np.testing.assert_array_equal(npz["detected"], [True, False])
        printed = capsys.readouterr().out
        assert str(DECOY_ID) in printed
        assert "not detected" in printed

    def test_the_gate_keeps_only_the_detections(self, tmp_path, stub_ms,
                                                stub_orbits):
        """Threshold-gated saving: a curve extracted at a tau that is not a
        detection is a curve extracted at noise."""
        from tabascal.scripts.rfi_estimate import run

        out = str(tmp_path / "curves.npz")
        tle_dir = str(tmp_path / "tles")

        run(_cli("-ms", str(tmp_path / "obs.ms"), "-n",
                 f"{NORAD_ID},{DECOY_ID}", "--fit-offset", *FAST_FIT,
                 "--only-detections", "-o", out,
                 "--write-shifted-tle", tle_dir))

        with np.load(out, allow_pickle=False) as npz:
            np.testing.assert_array_equal(npz["norad_ids"], [NORAD_ID])
            assert npz["light_curves"].shape == (1, N_TIME, N_FREQ)
        # Only the detected satellite's orbit is worth shifting.
        resolved, _ = _select_from_extra_dir(
            tle_dir, {NORAD_ID, DECOY_ID}, float(np.mean(stub_ms["times_jd"])),
            None,
        )
        assert set(resolved) == {NORAD_ID}

    @pytest.mark.parametrize(
        "flags, want", [((), "single"), (("--precision", "double"), "double")]
    )
    def test_the_standalone_run_sets_the_scans_precision(self, tmp_path, stub_ms,
                                                         stub_orbits, monkeypatch,
                                                         flags, want):
        """With ``-ms`` nothing has set a precision, so the scan sets its own.

        The setter is spied on rather than the JAX global read back: enabling
        x64 is process-wide and this suite is run in both precisions from the
        outside, so a test that asserted on the global would either fight the
        session or leak into every test after it.
        """
        import tabascal.scripts.rfi_estimate as script
        from tabascal.scripts.rfi_estimate import run

        asked = []
        monkeypatch.setattr(script, "set_precision_for_scan", asked.append)

        run(_cli("-ms", str(tmp_path / "obs.ms"), "-n", str(NORAD_ID),
                 "--fit-offset", *FAST_FIT, *flags,
                 "-o", str(tmp_path / "curves.npz")))

        assert asked == [want]

    @pytest.mark.parametrize("precision", ["double", "single"])
    def test_the_standalone_run_puts_the_process_back_as_it_found_it(
        self, tmp_path, stub_ms, stub_orbits, precision
    ):
        """Every global the scan's precision setter touches, not just x64.

        ``run`` is a function in an importable module, so a notebook or a
        pipeline can call it between two things of its own. Setting the scan's
        precision means flipping process-wide JAX flags, and
        ``set_precision`` pins ``jax_default_matmul_precision`` alongside
        ``jax_enable_x64`` -- on Ampere+ GPUs that one decides whether an f32
        matmul is really f32 or TF32, so leaving it changed silently rewrites
        the numerics of whatever the caller does next.

        The real setter is used here, unpatched, which is the only way to see
        what it leaves behind. The matmul flag is moved off the session's own
        value first so that restoring it is visible rather than coincidental,
        and both flags are put back in a finally: a test that protects the rest
        of the session must not be the thing that corrupts it.
        """
        import jax

        from tabascal.scripts.rfi_estimate import run

        session_matmul = jax.config.jax_default_matmul_precision
        was_x64 = jax.config.read("jax_enable_x64")
        marker = "float32"
        jax.config.update("jax_default_matmul_precision", marker)
        try:
            run(_cli("-ms", str(tmp_path / "obs.ms"), "-n", str(NORAD_ID),
                     "--fit-offset", *FAST_FIT, "--precision", precision,
                     "-o", str(tmp_path / "curves.npz")))

            assert jax.config.read("jax_enable_x64") == was_x64
            assert jax.config.jax_default_matmul_precision == marker
        finally:
            jax.config.update("jax_enable_x64", was_x64)
            jax.config.update("jax_default_matmul_precision", session_matmul)

    def test_the_config_run_leaves_the_precision_to_the_config(
        self, tmp_path, stub_config_mode, stub_orbits, monkeypatch
    ):
        """With ``-c`` the precision has already been set, from the run's own
        ``model.precision``, before anything is read. Setting it a second time
        from a flag nobody gave would silently overrule the config on its own
        subject -- so the scan's setter is not called at all on this path, and
        an explicit ``--precision`` remains the only way to override it.
        """
        import tabascal.scripts.rfi_estimate as script
        from tabascal.scripts.rfi_estimate import run

        asked = []
        monkeypatch.setattr(script, "set_precision_for_scan", asked.append)
        out = str(tmp_path / "curves.npz")

        run(_cli("-c", "tab.yaml", "--fit-offset", *FAST_FIT, "-o", out))

        assert asked == []
        # And the scan really did run on this path, rather than the assertion
        # above passing because nothing happened.
        with np.load(out, allow_pickle=False) as npz:
            assert npz["tau_best"][0] == pytest.approx(TAU0, abs=0.5)


class TestDiagnosticPlot:
    """The figure the issue asks for, as a file that exists."""

    def test_it_writes_a_png(self, tmp_path, default_fit):
        path = str(tmp_path / "offset.png")

        got = plot_offset_diagnostics(default_fit, path, title="25544")

        assert got == path
        assert os.path.getsize(path) > 0
