"""Cross-satellite candidate search and the ``search`` subcommand (GitHub #191).

The user story is "something is in my data and I do not know what": given a TLE
snapshot and no other prior knowledge, tabascal has to produce the
``satellites.norad_ids`` list itself. The pipeline is the one that ran by hand on
the MWA Cen A dataset -- 2551 Starlink records, 128 above the horizon during the
56 s observation, a tau-scanned near-field matched filter (#190) per candidate on
the coherent baselines (#189), and STARLINK-1765 alone at ``z2 = 0.0995`` against
a runner-up of 0.0523 and a candidate median of 0.0446.

What is pinned here is the *search*, not that a number comes out:

* the elevation screen drops a satellite that was never up, **before** anything
  is scored -- which is also what keeps the static baseline set honest, since a
  candidate 13 000 km away has a coherence length of kilometres and would let
  every long baseline back into the sum;
* the batched, ``vmap``\\ ped statistic is the *same* statistic
  :func:`~tabascal.rfi_estimate.fit_time_offset` computes one satellite at a
  time. That is checked directly, row for row, rather than assumed: a search
  that re-derived the filter would be free to disagree with the single-satellite
  fit about what a detection is;
* a satellite that rises or sets mid-observation contributes only its in-view
  frames, expressed as a 0/1 mask inside the jitted statistic (the shapes have
  to stay static to batch) -- and that must give the same score as slicing those
  frames out, which is what the single-satellite path does;
* the batch size is an efficiency knob and nothing else: 1, 2 and 4 give the
  same numbers, and a ragged last batch is padded rather than truncated.

The synthetic observation is PR #190's -- an ISS-shaped TLE propagated by the
repo's own skyfield path, with the array centre at the sub-satellite point so
the pass is overhead, eight antennas inside 250 m plus one 8 km outlier whose
baselines the #189 cut drops. Three candidates are offered to the search:

* **A** (25544), overhead at 84-90 deg, 349 km away, whose fringe is injected at
  ``tau = +1.5 s`` on channel 2;
* **B** (27386), the same orbit 30 s behind -- a train mate at 52-60 deg and
  398 km, the satellite most able to masquerade as A, and the second injected
  source (channel 0, ``tau = -1.0 s``) in the two-signal fixture;
* **C** (43013), half an orbit away and 89 deg *below* the horizon, which the
  screen must drop before it can widen the baseline set.

The expected answers are therefore known: A ranks first at ``z2 = 14.0`` and
~19 sigma, B follows at 1.2 (2 sigma) with A alone injected and at 6.0
(20 sigma) with both, pure noise reaches neither 4 in ``z2`` nor 5 sigma, and 28
of the 36 non-auto baselines are coherent for every candidate that is up.

Everything is offline -- no MS, no SatChecker, no network (``block_network``
enforces it).
"""

import os
import re
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from skyfield.framelib import itrs

from satchecker_client.records import record_epoch_jd

from tabascal.components.trajectory import (
    _earth_satellite,
    get_satellite_elevations,
    get_satellite_positions,
    itrs_to_gcrs_sf,
)
from tabascal.interferometry import C, Omega_e, itrf_to_uvw_numpy
from tabascal.orbit import _select_from_extra_dir, save_orbits_for_reuse
from tabascal.rfi_estimate import (
    _batched_tau_scan,
    baseline_lengths,
    candidates_from_norad_ids,
    candidates_from_orbit_dir,
    coherent_baseline_mask,
    enumerate_candidates,
    fit_time_offset,
    plot_candidate_ranking,
    satellite_range_and_speed,
    search_candidates,
    select_detections,
    tau_scan,
    tle_coherence_length,
    write_config_fragment,
    write_search_results,
)
from tabascal.time import gast_deg, jd_to_mjd, skyfield_time, timescale

from .tle_helpers import (  # noqa: F401
    block_network,
    jd,
    make_tle,
    make_tle_record,
    with_checksum,
)


# ---------------------------------------------------------------------------
# The synthetic observation (PR #190's, with a second and a third satellite)
# ---------------------------------------------------------------------------

EPOCH_JD = jd(2026, 8, 1)
INT_TIME = 0.5
N_TIME = 20
N_FINE = 40
N_FREQ = 4
FREQS = 175e6 + 40e3 * np.arange(N_FREQ)
SIGMA = 1.0
SIGMA_TRANSVERSE = 300.0

#: The satellite whose fringe is in the data, its train mate 30 s behind, and a
#: decoy half an orbit away that never rises.
A_ID, B_ID, C_ID = 25544, 27386, 43013
TRAIN_LAG_S = 30.0
A_NAME, B_NAME, C_NAME = "STARLINK-1765", "STARLINK-1766", "ISS (ZARYA)"

TAU_A, CHAN_A, AMP_A = 1.5, 2, 3.0
#: The second source is put on another channel and at another offset, so the two
#: detections are told apart by both of the things the search reports. Its
#: amplitude is a third of A's power, which leaves it clearly detected (20 sigma)
#: and clearly second (z2 6.0 against 14.0) -- outside the 1.5x runner-up warning
#: either way, so that warning can be exercised on its own terms.
TAU_B, CHAN_B, AMP_B = -1.0, 0, 1.0
#: The offset that lands on the last point of the grid below.
TAU_EDGE = 2.0

#: The scan grid the fixtures use: +-2 s in 0.5 s steps, which holds every
#: injected offset and puts TAU_EDGE on its last point.
SEARCH_GRID = np.arange(-2.0, 2.0 + 0.25, 0.5)
#: Three points around the injected offset, for the tests that are about the
#: geometry rather than about the shape of the peak.
LONG_GRID = np.array([1.0, TAU_A, 2.0])
#: Null draws in the fixtures. Enough for a mean and a spread; the size of the
#: null is #190's subject, not this file's.
N_NULL = 64

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
N_ANT = len(ANT_OFFSETS)
FAR_ANT = N_ANT - 1
#: Non-auto baselines inside the coherence length of every candidate that is up:
#: every pair among the eight core antennas. Derived from
#: :func:`coherent_baseline_mask` in the tests too, so a change in the cut shows
#: up as a disagreement rather than as a stale literal.
N_COHERENT_BL = 28

#: Further train mates, none of them in the data, for the batching and
#: compilation tests: 15 s behind (66-77 deg up), 15 s ahead (66-77), 60 s behind
#: (34-38). Epoch shifts, so each is the same orbit read at another point.
TRAIN = ((40015, "STARLINK-1767", 15.0), (41015, "STARLINK-1768", -15.0),
         (40060, "STARLINK-1769", 60.0))
D_ID, D_NAME = TRAIN[0][0], TRAIN[0][1]

#: A long pass, for the coherence-geometry tests: 2 s dumps -- the MWA cadence --
#: over 200 s, which A spends falling from overhead to 7 deg. Cut at the
#: half-way elevation it is in view for the first 100 s, whose middle sits
#: 496 km away against the 349 km of its closest approach.
N_TIME_LONG, INT_TIME_LONG = 100, 2.0
#: A transverse orbit error that puts the hard cut *between* two of the array's
#: baselines at one range and past both at another: 220 m at A's closest
#: approach, 262 m at B's mid-window range, 312 m at the middle of the long
#: pass's in-view window. The 239 m spacing -- the longest among the core
#: antennas, the next one out being 8 km -- is the baseline that flips, so which
#: geometry the cut is sized from is a question with an observable answer.
SIGMA_TRANSVERSE_SPLIT = 432.5
#: The core baselines inside 220 m: everything but that 239 m spacing.
N_SPLIT_BL = 27


def mean_anomaly_shifted(line2, degrees):
    """A TLE line 2 with its mean anomaly moved, and its checksum redone.

    Half a revolution puts the satellite on the far side of its own orbit --
    through the Earth from the array, ~90 deg below the horizon -- while leaving
    every other element, and so the orbit itself, exactly as it was. Substituting
    into the fixed-width field invalidates the modulo-10 checksum, which the
    parser rejects, so it is recomputed as a real TLE producer would.
    """
    anomaly = (float(line2[43:51]) + float(degrees)) % 360.0

    return with_checksum(line2[:43] + f"{anomaly:8.4f}" + line2[51:])


def ref_fine_offsets(n_fine, delta_t):
    """Midpoint sub-step offsets, written out rather than imported."""
    return ((np.arange(n_fine) + 0.5) / n_fine - 0.5) * delta_t


def ref_paths(record, ants_itrf, times_jd, phase_centre, a1, a2, n_fine, delta_t,
              taus_s=0.0):
    """Near-field baseline path differences, transcribed independently.

    The satellite is propagated to ``t + offset + tau``; the antennas and their
    phase-tracking ``w`` stay at ``t + offset``. (#190 pins that convention
    against the alternative; here it is only the source of the injected fringe.)
    """
    taus_s = np.atleast_1d(np.asarray(taus_s, dtype=np.float64))
    offsets = ref_fine_offsets(n_fine, delta_t)
    times_jd = np.asarray(times_jd, dtype=np.float64)
    t_fine = (times_jd[:, None] + offsets[None, :] / 86400.0).ravel()

    gh0 = (gast_deg(t_fine) - phase_centre["ra"]) % 360
    w = np.transpose(
        itrf_to_uvw_numpy(ants_itrf, gh0, phase_centre["dec"]), axes=(1, 0, 2)
    )[..., -1]
    ants_xyz = itrs_to_gcrs_sf(ants_itrf, t_fine)

    out = np.zeros((len(taus_s), len(a1), len(times_jd), n_fine))
    for i, tau in enumerate(taus_s):
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


def ref_slant_range(record, ants_itrf, t_jd):
    """Distance from the array centre to the satellite, in the Earth-fixed frame."""
    centre = np.mean(np.asarray(ants_itrf, dtype=np.float64), axis=0)
    position = (
        _earth_satellite(record, timescale())
        .at(skyfield_time(np.atleast_1d(np.float64(t_jd))))
        .frame_xyz(itrs)
        .m.T[0]
    )

    return float(np.linalg.norm(position - centre))


def ref_coherent(record, ants_itrf, times_jd, a1, a2,
                 sigma_transverse_m=SIGMA_TRANSVERSE, n_fine=N_FINE,
                 int_time=INT_TIME, range_m=None):
    """The #189 cut for one candidate, sized the way the single-satellite fit does.

    From the **mid-window** ``(range, v_perp)`` pair over the frames it is handed
    -- one self-consistent geometry taken at one instant, rather than a range
    from one part of the pass and a speed from another. ``range_m`` overrides the
    range alone, which is how a test asks what the cut *would* have been
    somewhere else on the pass.
    """
    range_mid, v_perp = satellite_range_and_speed(record, ants_itrf, times_jd)
    range_m = float(range_mid if range_m is None else range_m)
    bl_len = np.asarray(baseline_lengths(ants_itrf, a1, a2), dtype=np.float64)
    v_fringe = v_perp + Omega_e * range_m
    mean_freq = float(np.mean(FREQS))
    lam = C / mean_freq
    b_coh = float(
        min(
            lam * range_m / (2 * np.pi * sigma_transverse_m),
            lam * range_m * n_fine / (2 * int_time * v_fringe),
        )
    )
    mask = np.asarray(
        coherent_baseline_mask(
            bl_len, mean_freq, range_m, sigma_transverse_m, n_fine, int_time,
            v_fringe,
        )
    )

    return mask & (a1 != a2), b_coh


def build_observation(times_jd, ants_itrf, phase_centre, a1, a2, seed,
                      injections=(), int_time=INT_TIME):
    """Noise plus each injected satellite's fringe, on one channel each.

    The fringes go in through :func:`ref_model` -- this file's own model, owing
    nothing to the module under test -- the way a narrowband downlink lands in
    one channel.
    """
    times_jd = np.asarray(times_jd, dtype=np.float64)
    shape = (len(a1), N_FREQ, len(times_jd))

    rng = np.random.default_rng(seed)
    vis = rng.normal(0, SIGMA, shape) + 1j * rng.normal(0, SIGMA, shape)
    for record, tau, chan, amp in injections:
        paths = ref_paths(record, ants_itrf, times_jd, phase_centre, a1, a2,
                          N_FINE, int_time, tau)[0]
        vis[:, chan, :] += amp * ref_model(paths)[:, chan, :]

    return SimpleNamespace(
        vis=vis, times_jd=times_jd, ants_itrf=ants_itrf, a1=a1, a2=a2,
        freqs=FREQS, phase_centre=phase_centre, int_time=float(int_time),
        n_time=len(times_jd), n_bl=len(a1),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def record_a():
    return make_tle_record(A_ID, EPOCH_JD, OBJECT_NAME=A_NAME)


@pytest.fixture(scope="module")
def record_b():
    """The train mate: the same orbit, 30 s behind.

    A later epoch on the same elements is the same trajectory read earlier, so
    this is A's orbit with the satellite ~230 km back along the track -- still
    52-60 deg up, so it is the statistic and not the elevation cut that has to
    tell the two apart, and the nearest thing to a false positive a real
    constellation offers.
    """
    return make_tle_record(
        B_ID, EPOCH_JD + TRAIN_LAG_S / 86400.0, OBJECT_NAME=B_NAME
    )


@pytest.fixture(scope="module")
def record_c():
    """A satellite half a revolution away: never above the horizon here."""
    _, line2 = make_tle(C_ID, EPOCH_JD)
    record = make_tle_record(C_ID, EPOCH_JD, OBJECT_NAME=C_NAME)
    record["TLE_LINE2"] = mean_anomaly_shifted(line2, 180.0)

    return record


@pytest.fixture(scope="module")
def records(record_a, record_b, record_c):
    return [record_a, record_b, record_c]


@pytest.fixture(scope="module")
def names():
    return [A_NAME, B_NAME, C_NAME]


@pytest.fixture(scope="module")
def layout(record_a):
    """An array under the satellite: the centre is A's sub-satellite point.

    The pass is then overhead, which is where the direction to the satellite
    swings fastest across the array -- the geometry the near-field filter is
    for. The phase centre is the zenith of that site; it enters only through the
    ``w`` term.
    """
    times_jd = EPOCH_JD + np.arange(N_TIME) * INT_TIME / 86400.0
    t_ref = float(np.mean(times_jd))

    sat_itrs = (
        _earth_satellite(record_a, timescale())
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

    return SimpleNamespace(
        ants_itrf=ants_itrf, a1=a1, a2=a2, times_jd=times_jd, t_ref=t_ref,
        phase_centre={
            "ra": float((gast_deg(np.atleast_1d(t_ref))[0] + lon) % 360.0),
            "dec": float(lat),
        },
    )


@pytest.fixture(scope="module")
def obs_a(layout, record_a):
    """A's fringe alone, at ``TAU_A`` on channel 2."""
    return build_observation(
        layout.times_jd, layout.ants_itrf, layout.phase_centre, layout.a1,
        layout.a2, seed=12345,
        injections=[(record_a, TAU_A, CHAN_A, AMP_A)],
    )


@pytest.fixture(scope="module")
def obs_ab(layout, record_a, record_b):
    """Two contaminators at once, on different channels and offsets."""
    return build_observation(
        layout.times_jd, layout.ants_itrf, layout.phase_centre, layout.a1,
        layout.a2, seed=12345,
        injections=[(record_a, TAU_A, CHAN_A, AMP_A),
                    (record_b, TAU_B, CHAN_B, AMP_B)],
    )


@pytest.fixture(scope="module")
def obs_noise(layout):
    """The same observation with no satellite in it at all."""
    return build_observation(
        layout.times_jd, layout.ants_itrf, layout.phase_centre, layout.a1,
        layout.a2, seed=0,
    )


@pytest.fixture(scope="module")
def obs_edge(layout, record_a):
    """A's fringe at an offset sitting on the last point of the grid."""
    return build_observation(
        layout.times_jd, layout.ants_itrf, layout.phase_centre, layout.a1,
        layout.a2, seed=12345,
        injections=[(record_a, TAU_EDGE, CHAN_A, AMP_A)],
    )


@pytest.fixture(scope="module")
def setting_obs(layout, record_a):
    """A window A leaves partway through, with the cut at its half-way elevation.

    The window starts at the overhead instant, so the elevation falls
    monotonically from there. The geometric horizon would mask the same way while
    destroying the geometry the rest of the search needs -- a grazing pass is ten
    times further off and crosses the line of sight far more slowly -- so the cut
    is put where the satellite actually is.
    """
    times_jd = layout.t_ref + np.arange(N_TIME) * INT_TIME / 86400.0
    observation = build_observation(
        times_jd, layout.ants_itrf, layout.phase_centre, layout.a1, layout.a2,
        seed=4321, injections=[(record_a, TAU_A, CHAN_A, AMP_A)],
    )
    elevation = np.asarray(
        get_satellite_elevations([record_a], times_jd, layout.ants_itrf)
    )[0]
    half = N_TIME // 2
    observation.elevation = elevation
    observation.min_elevation = float(0.5 * (elevation[half - 1] + elevation[half]))
    observation.frames = elevation >= observation.min_elevation

    return observation


@pytest.fixture(scope="module")
def candidates(records, names, layout):
    """The three records screened: A and B are up, C never is."""
    return enumerate_candidates(
        records, names, layout.times_jd, layout.ants_itrf, min_elevation=0.0
    )


def train_mates(count):
    """``count`` further satellites of the same train, by epoch shift."""
    records = [
        make_tle_record(nid, EPOCH_JD + lag / 86400.0, OBJECT_NAME=name)
        for nid, name, lag in TRAIN[:count]
    ]

    return records, [name for _, name, _ in TRAIN[:count]]


@pytest.fixture(scope="module")
def candidates3(records, names, layout):
    """Three genuinely different candidates, for the batching tests.

    Distinct scores, so a tie cannot make two orderings both correct and hide a
    batch that dropped or reordered a row.
    """
    extra, extra_names = train_mates(1)

    return enumerate_candidates(
        records[:2] + extra, names[:2] + extra_names, layout.times_jd,
        layout.ants_itrf, min_elevation=0.0,
    )


@pytest.fixture(scope="module")
def candidates5(records, names, layout):
    """Five candidates: two full batches of two and a ragged one."""
    extra, extra_names = train_mates(3)

    return enumerate_candidates(
        records[:2] + extra, names[:2] + extra_names, layout.times_jd,
        layout.ants_itrf, min_elevation=0.0,
    )


@pytest.fixture(scope="module")
def long_pass(layout, record_a):
    """A pass followed from overhead until it is nearly down, and masked halfway.

    Two-second dumps -- the MWA cadence the case study was measured on -- over
    200 s, the window a near-field search of a LEO pass actually covers. It is
    the fixture that can tell the two candidate geometries apart: A is 349 km
    away at its closest approach, in the first frame, and 496 km away at the
    middle of the frames it is still in view for.
    """
    times_jd = layout.t_ref + np.arange(N_TIME_LONG) * INT_TIME_LONG / 86400.0
    observation = build_observation(
        times_jd, layout.ants_itrf, layout.phase_centre, layout.a1, layout.a2,
        seed=99, injections=[(record_a, TAU_A, CHAN_A, AMP_A)],
        int_time=INT_TIME_LONG,
    )
    elevation = np.asarray(
        get_satellite_elevations([record_a], times_jd, layout.ants_itrf)
    )[0]
    half = N_TIME_LONG // 2
    observation.elevation = elevation
    observation.min_elevation = float(0.5 * (elevation[half - 1] + elevation[half]))
    observation.frames = elevation >= observation.min_elevation

    return observation


def run_search(observation, candidates, **kwargs):
    """The search as every fixture runs it, on the file's own grid and null."""
    settings = dict(
        taus_s=SEARCH_GRID, n_fine=N_FINE, sigma_transverse_m=SIGMA_TRANSVERSE,
        noise=SIGMA, n_null=N_NULL,
    )
    settings.update(kwargs)

    return search_candidates(
        observation.vis, candidates, observation.ants_itrf, observation.times_jd,
        observation.phase_centre, observation.freqs, observation.a1,
        observation.a2, observation.int_time, **settings,
    )


@pytest.fixture(scope="module")
def search_a(obs_a, candidates):
    return run_search(obs_a, candidates)


@pytest.fixture(scope="module")
def search_ab(obs_ab, candidates):
    return run_search(obs_ab, candidates)


@pytest.fixture(scope="module")
def search_noise(obs_noise, candidates):
    return run_search(obs_noise, candidates)


@pytest.fixture(scope="module")
def search_edge(obs_edge, candidates):
    return run_search(obs_edge, candidates)


def single_fit(observation, record, taus_s=SEARCH_GRID, **kwargs):
    """The #190 single-satellite fit, for the "one core" comparisons."""
    settings = dict(
        noise=SIGMA, n_fine=N_FINE, sigma_transverse_m=SIGMA_TRANSVERSE
    )
    settings.update(kwargs)

    return fit_time_offset(
        observation.vis, record, observation.ants_itrf, observation.times_jd,
        observation.phase_centre, observation.freqs, observation.a1,
        observation.a2, observation.int_time, taus_s=taus_s, **settings,
    )


def bytes_per_candidate(n_bl_used, n_tau):
    """What one candidate of a batch costs, written out as the search sizes it.

    The per-offset fringe model -- ``(n_bl, n_freq, n_time, n_fine)`` complex, in
    whatever precision the scan is running in -- plus the host-side float64 path
    differences for the whole offset grid. On the MWA case the shared set runs
    out to 7704 of 9180 baselines (a candidate 2800 km away carries a 2.9 km
    coherence length), and eight of these at once is 17 GB: the machine swaps
    long before the GPU is asked for anything.
    """
    complex_bytes = jnp.zeros(1, dtype=complex).dtype.itemsize

    return (
        n_bl_used * N_FREQ * N_TIME * N_FINE * complex_bytes
        + n_tau * n_bl_used * N_TIME * N_FINE * 8
    )


def fake_search(entries, tau_grid=SEARCH_GRID):
    """A search-shaped dict from ``(z2_best, tau_best, significance)`` rows.

    The warning and selection logic is decided from the table alone, so it is
    tested on hand-built rows -- every corner of it, at no cost -- with the
    end-to-end paths checked on the real fixtures.
    """
    table = [
        dict(
            rank=i, norad_id=90000 + i, name=f"SAT-{90000 + i}",
            max_elevation=45.0, range_m=4.0e5, z2_best=float(z2),
            tau_best=float(tau), best_chan=1, best_freq=float(FREQS[1]),
            r_max=0.5, significance=float(sigma), null_mean=0.5, null_std=0.1,
            n_frames=N_TIME,
        )
        for i, (z2, tau, sigma) in enumerate(entries)
    ]

    def column(key, dtype=float):
        return np.array([row[key] for row in table], dtype=dtype)

    return dict(
        table=table,
        norad_ids=column("norad_id", int),
        z2_best=column("z2_best"),
        tau_best=column("tau_best"),
        best_chan=column("best_chan", int),
        significance=column("significance"),
        z2_tau=np.zeros((len(table), len(tau_grid), N_FREQ)),
        tau_grid=np.asarray(tau_grid, dtype=float),
        frames=np.ones((len(table), N_TIME), dtype=bool),
        n_bl_used=N_COHERENT_BL,
        b_coh_max=350.0,
        fits=[],
        median_z2=float(np.median(column("z2_best"))) if table else float("nan"),
    )


# ---------------------------------------------------------------------------
# Stage 1: which satellites are there to search for
# ---------------------------------------------------------------------------

class TestEnumerateCandidates:
    """The elevation screen: what is worth scoring, and over which frames."""

    def test_only_the_satellites_that_come_up_are_kept(self, candidates, record_a):
        """A pass that never happened is not evidence, and scoring it would cost
        a scan each -- the screen is what turns 2551 records into 128.

        The elements travel with the candidate, not just the ID: re-resolving
        them later could pick up a different record for the same satellite.
        """
        assert [c["norad_id"] for c in candidates] == [A_ID, B_ID]
        assert [c["name"] for c in candidates] == [A_NAME, B_NAME]
        assert candidates[0]["record"] is record_a

    def test_they_are_ranked_by_how_high_they_rise(self, candidates):
        elevations = [c["max_elevation"] for c in candidates]

        assert elevations == sorted(elevations, reverse=True)
        assert elevations[0] == pytest.approx(89.7, abs=1.0)
        assert elevations[1] == pytest.approx(60.4, abs=1.0)

    def test_each_candidate_carries_its_own_elevations_and_window(
        self, candidates, records, layout
    ):
        want = np.asarray(
            get_satellite_elevations(records, layout.times_jd, layout.ants_itrf)
        )

        for candidate, elevation in zip(candidates, want[:2]):
            frames = np.asarray(candidate["frames"])
            assert frames.dtype == bool
            assert frames.shape == (N_TIME,)
            assert frames.all()  # both are up throughout this window
            np.testing.assert_allclose(candidate["elevation"], elevation, atol=1e-6)
            assert candidate["max_elevation"] == pytest.approx(elevation.max())

    def test_a_setting_satellite_is_masked_from_where_it_drops(
        self, record_a, setting_obs, layout
    ):
        """The frames it is not up for are not evidence about it, and the mask
        says so once, here, for the statistic to apply without slicing."""
        [candidate] = enumerate_candidates(
            [record_a], [A_NAME], setting_obs.times_jd, layout.ants_itrf,
            min_elevation=setting_obs.min_elevation,
        )
        frames = np.asarray(candidate["frames"])

        np.testing.assert_array_equal(frames, setting_obs.frames)
        assert frames.any() and not frames.all()
        # Monotone: it sets once and does not come back inside the window.
        assert not np.any(np.diff(frames.astype(int)) > 0)

    def test_the_cut_is_inclusive(self, records, names, layout):
        """Same semantics as ``rfi.min_elevation``: the cut is the lowest
        elevation still modelled, so a satellite sitting exactly on it is up --
        for one frame, in this case, which is all "at least one integration"
        asks for."""
        elevation = np.asarray(
            get_satellite_elevations(records, layout.times_jd, layout.ants_itrf)
        )[1]
        peak = float(elevation.max())

        kept = enumerate_candidates(
            records, names, layout.times_jd, layout.ants_itrf, min_elevation=peak
        )
        above = enumerate_candidates(
            records, names, layout.times_jd, layout.ants_itrf,
            min_elevation=peak + 1e-6,
        )

        assert [c["norad_id"] for c in kept] == [A_ID, B_ID]
        frames = np.asarray(kept[1]["frames"])
        assert int(frames.sum()) == 1 and frames[int(np.argmax(elevation))]
        assert [c["norad_id"] for c in above] == [A_ID]

    def test_no_cut_at_all_keeps_even_the_satellite_below_the_horizon(
        self, records, names, layout
    ):
        kept = enumerate_candidates(
            records, names, layout.times_jd, layout.ants_itrf, min_elevation=None
        )

        assert [c["norad_id"] for c in kept] == [A_ID, B_ID, C_ID]
        assert np.asarray(kept[-1]["frames"]).all()
        assert kept[-1]["max_elevation"] < 0.0

    def test_the_range_is_the_slant_distance_at_the_highest_frame(
        self, candidates, layout
    ):
        """The closest approach, which is what a ranking table wants to report:
        how near the satellite came, once, rather than where it happened to be
        at some instant.

        It is *not* what sizes the coherence cut -- that is the mid-window
        geometry the fit uses, which the search reports separately in its fits
        (see :class:`TestCoherenceGeometry`). The two differ by half a pass.
        """
        for candidate in candidates:
            peak = int(np.argmax(candidate["elevation"]))
            want = ref_slant_range(
                candidate["record"], layout.ants_itrf, layout.times_jd[peak]
            )

            assert candidate["range_m"] == pytest.approx(want, rel=1e-3)

        assert 340e3 < candidates[0]["range_m"] < 360e3  # overhead
        assert candidates[1]["range_m"] > candidates[0]["range_m"]  # further off

    def test_nothing_to_screen_is_not_an_error(self, layout):
        """The batched search meets empty lists -- a name filter that matched
        nothing -- and must return one, not raise."""
        assert enumerate_candidates([], [], layout.times_jd, layout.ants_itrf) == []


@pytest.fixture(scope="module")
def snapshot_dir(tmp_path_factory, records):
    """A local TLE snapshot: the three satellites, and a stale copy of A.

    Two records for one satellite is the ordinary case for a constellation
    export, and the one nearest the observation has to win -- the other is two
    days old and would put A 15 000 km along its track.
    """
    directory = tmp_path_factory.mktemp("snapshot")
    stale = make_tle_record(A_ID, EPOCH_JD - 2.0, OBJECT_NAME=A_NAME)
    save_orbits_for_reuse(
        str(directory / "starlink.json"),
        [A_ID, A_ID, B_ID, C_ID],
        [records[0], stale, records[1], records[2]],
    )

    return str(directory)


class TestCandidatesFromOrbitDir:
    """Reading a snapshot directory: the ``--tle-dir`` source."""

    def test_it_returns_one_named_record_per_satellite(self, snapshot_dir, layout):
        records, names, norad_ids = candidates_from_orbit_dir(
            snapshot_dir, layout.times_jd
        )

        assert norad_ids == sorted(norad_ids)
        assert norad_ids == [A_ID, B_ID, C_ID]
        assert len(records) == len(names) == 3
        assert dict(zip(norad_ids, names)) == {
            A_ID: A_NAME, B_ID: B_NAME, C_ID: C_NAME
        }

    def test_the_record_nearest_the_observation_wins(self, snapshot_dir, layout):
        """The same per-ID nearest-epoch policy ``extra_orbit_dir`` already has,
        rather than whichever row the file happened to list first."""
        records, _, norad_ids = candidates_from_orbit_dir(
            snapshot_dir, layout.times_jd
        )
        chosen = records[norad_ids.index(A_ID)]

        assert record_epoch_jd(chosen) == pytest.approx(EPOCH_JD, abs=1e-3)

    def test_a_record_with_no_name_is_called_by_its_id(self, tmp_path, layout):
        """A ranking table has to name every row, and a Space-Track export can
        arrive without ``OBJECT_NAME``."""
        save_orbits_for_reuse(
            str(tmp_path / "anonymous.json"), [A_ID],
            [make_tle_record(A_ID, EPOCH_JD, OBJECT_NAME=None)],
        )

        _, names, norad_ids = candidates_from_orbit_dir(str(tmp_path), layout.times_jd)

        assert (names, norad_ids) == ([str(A_ID)], [A_ID])

    def test_the_name_filter_is_a_case_insensitive_substring(self, snapshot_dir,
                                                             layout):
        """A snapshot is usually a whole constellation plus whatever else the
        query dragged in; ``--name-filter STARLINK`` is how one is narrowed."""
        _, names, norad_ids = candidates_from_orbit_dir(
            snapshot_dir, layout.times_jd, name_filter="starlink"
        )

        assert norad_ids == [A_ID, B_ID]
        assert names == [A_NAME, B_NAME]
        assert candidates_from_orbit_dir(
            snapshot_dir, layout.times_jd, name_filter="NOT-A-SATELLITE"
        ) == ([], [], [])

    def test_an_empty_directory_yields_three_empty_lists(self, tmp_path, layout):
        assert candidates_from_orbit_dir(str(tmp_path), layout.times_jd) == ([], [], [])


@pytest.fixture
def stub_orbits(monkeypatch, records):
    """Resolve NORAD IDs offline, through the seam the run itself uses."""
    import tabascal.rfi_estimate as mod

    asked = {}
    by_id = {int(r["NORAD_CAT_ID"]): r for r in records}

    def _fetch(times_jd=None, norad_ids=None, **kwargs):
        asked.update(kwargs)
        ids = [int(n) for n in norad_ids]

        return None, None, ids, [by_id[n] for n in ids], len(ids)

    monkeypatch.setattr(mod, "fetch_orbital_elements", _fetch)

    return asked


class TestCandidatesFromNoradIds:
    """The explicit list: ``-n``/``-np``, through the run's own resolver."""

    def test_it_resolves_in_the_order_it_was_asked(self, stub_orbits, layout):
        records, names, norad_ids = candidates_from_norad_ids(
            [B_ID, A_ID], layout.times_jd
        )

        assert norad_ids == [B_ID, A_ID]
        assert names == [B_NAME, A_NAME]
        assert [int(r["NORAD_CAT_ID"]) for r in records] == [B_ID, A_ID]

    def test_a_local_orbit_directory_is_offered_to_the_resolver(self, stub_orbits,
                                                                layout, tmp_path):
        """So a snapshot can seed an explicit list exactly as it seeds a run."""
        candidates_from_norad_ids(
            [A_ID], layout.times_jd, extra_orbit_dir=str(tmp_path)
        )

        assert stub_orbits["extra_orbit_dir"] == str(tmp_path)


# ---------------------------------------------------------------------------
# Stage 2: scoring every candidate
# ---------------------------------------------------------------------------

class TestSearchResult:
    """The shape of what a search returns, and what it is built on."""

    def test_the_ranking_arrays_line_up_with_the_table(self, search_a):
        table = search_a["table"]

        assert [row["rank"] for row in table] == list(range(len(table)))
        np.testing.assert_array_equal(
            search_a["norad_ids"], [row["norad_id"] for row in table]
        )
        np.testing.assert_allclose(
            search_a["z2_best"], [row["z2_best"] for row in table]
        )
        np.testing.assert_allclose(
            search_a["tau_best"], [row["tau_best"] for row in table]
        )
        np.testing.assert_array_equal(
            search_a["best_chan"], [row["best_chan"] for row in table]
        )
        np.testing.assert_allclose(
            search_a["significance"], [row["significance"] for row in table]
        )
        assert search_a["median_z2"] == pytest.approx(
            float(np.median(np.asarray(search_a["z2_best"])))
        )

    def test_every_row_says_what_was_measured_and_where(self, search_a):
        """The table is the artifact the issue asks for: the log's ranking, and
        the per-satellite answer a config fragment is written from."""
        for row in search_a["table"]:
            assert set(row) >= {
                "rank", "norad_id", "name", "max_elevation", "range_m",
                "z2_best", "tau_best", "best_chan", "best_freq", "r_max",
                "significance", "null_mean", "null_std", "n_frames",
            }
            assert row["best_freq"] == pytest.approx(FREQS[row["best_chan"]])
            assert row["tau_best"] in set(SEARCH_GRID)
            assert 0.0 <= row["r_max"] <= 1.0 + 1e-5
            assert row["n_frames"] == N_TIME

    def test_the_scan_curves_come_back_for_every_candidate(self, search_a):
        """The shape of the peak is how a detection is judged by eye, and the
        four-panel report the issue asks for is drawn from it."""
        n_cand = len(search_a["table"])

        np.testing.assert_allclose(search_a["tau_grid"], SEARCH_GRID)
        assert np.asarray(search_a["z2_tau"]).shape == (
            n_cand, len(SEARCH_GRID), N_FREQ
        )
        assert np.asarray(search_a["frames"]).shape == (n_cand, N_TIME)
        assert np.asarray(search_a["frames"]).dtype == bool
        for row, z2 in zip(search_a["table"], np.asarray(search_a["z2_tau"])):
            i_tau, i_chan = np.unravel_index(int(np.argmax(z2)), z2.shape)
            assert search_a["tau_grid"][i_tau] == row["tau_best"]
            assert i_chan == row["best_chan"]
            assert row["z2_best"] == pytest.approx(float(z2.max()))

    def test_each_candidate_gets_a_fit_the_diagnostics_can_read(self, search_a):
        """Shaped like :func:`fit_time_offset`'s, so ``plot_offset_diagnostics``
        and ``attach_offset_fits`` take it unchanged and the search does not need
        a second, parallel set of plotting and saving code."""
        fits = search_a["fits"]

        assert len(fits) == len(search_a["table"])
        for fit, row in zip(fits, search_a["table"]):
            assert set(fit) >= {
                "tau_grid", "z2_tau", "tau_best", "z2_best", "best_chan",
                "best_freq", "r_best", "frames", "elevation", "null",
                "null_mean", "null_std", "significance", "n_bl_used", "range_m",
                "v_perp_m_s", "b_coh", "n_fine", "sigma_transverse_m",
            }
            assert fit["tau_best"] == row["tau_best"]
            assert fit["best_chan"] == row["best_chan"]
            assert np.asarray(fit["r_best"]).shape == (N_FREQ, N_TIME)
            assert np.asarray(fit["z2_tau"]).shape == (len(SEARCH_GRID), N_FREQ)
            assert np.asarray(fit["elevation"]).shape == (N_TIME,)
            assert fit["n_fine"] == N_FINE
            assert fit["sigma_transverse_m"] == SIGMA_TRANSVERSE
            assert fit["b_coh"] > 0.0

    def test_the_baseline_set_is_static_and_coherent_for_every_candidate(
        self, search_a, candidates, layout
    ):
        """One baseline set for the whole search, so the vmapped shapes stay
        static -- the union of the above-horizon candidates' hard supports
        (not any one candidate's set: the fringe ceiling depends on each one's
        own crossing speed, so the supports are not nested by range), with each
        candidate's own coherence applied as a weight inside.

        The satellite below the horizon is 13 000 km away and would lift the
        ceiling to kilometres, letting every 8 km baseline into the sum; that it
        does not is the screen doing its job before the geometry is sized.
        """
        bl_len = np.asarray(baseline_lengths(layout.ants_itrf, layout.a1, layout.a2))
        union = np.zeros(len(bl_len), dtype=bool)
        b_coh = []
        for candidate in candidates:
            keep, b = ref_coherent(
                candidate["record"], layout.ants_itrf,
                layout.times_jd[np.asarray(candidate["frames"])], layout.a1,
                layout.a2,
            )
            union |= keep
            b_coh.append(b)

        assert search_a["n_bl_used"] == N_COHERENT_BL
        assert search_a["n_bl_used"] == int(union.sum())
        assert search_a["b_coh_max"] == pytest.approx(max(b_coh), rel=1e-6)
        assert bl_len[union].max() < search_a["b_coh_max"] < 7000.0

    def test_the_set_is_shared_but_the_cut_is_each_candidates_own(
        self, obs_a, candidates, layout, exact_rtol
    ):
        """The *shapes* are static; the coherence is not.

        At 432.5 m of transverse error the overhead satellite can steer 220 m of
        baseline and the one half a train behind, 67 km further off, can steer
        262 m -- so the array's longest core spacing, 239 m, belongs to one of
        them and not the other. The shared set is the wider one, because
        shortening it would throw away a baseline the farther candidate can
        genuinely use; the near one is then held to its own cut by a *weight*
        rather than by a shorter array, so the batch stays rectangular. Scoring
        it beside another candidate must give exactly what scoring it alone
        gives.
        """
        settings = dict(
            taus_s=SEARCH_GRID[3:6], sigma_transverse_m=SIGMA_TRANSVERSE_SPLIT,
            n_null=4, n_null_candidates=1,
        )
        keep = [
            ref_coherent(
                candidate["record"], layout.ants_itrf,
                layout.times_jd[np.asarray(candidate["frames"])], layout.a1,
                layout.a2, sigma_transverse_m=SIGMA_TRANSVERSE_SPLIT,
            )[0]
            for candidate in candidates
        ]

        # Not vacuous: the two candidates really do disagree about the set.
        assert int(keep[0].sum()) == N_SPLIT_BL < int(keep[1].sum())

        alone = run_search(obs_a, candidates[:1], **settings)
        together = run_search(obs_a, candidates, **settings)
        rank = [row["norad_id"] for row in together["table"]].index(A_ID)

        assert alone["n_bl_used"] == int(keep[0].sum())
        assert together["n_bl_used"] == int((keep[0] | keep[1]).sum())
        np.testing.assert_allclose(
            np.asarray(together["z2_tau"])[rank], np.asarray(alone["z2_tau"])[0],
            rtol=exact_rtol,
        )


class TestCoherenceGeometry:
    """Which geometry sizes the coherent set, and what "soft" does to it."""

    def test_the_cut_is_sized_where_the_single_satellite_fit_sizes_it(
        self, long_pass, record_a, layout, precision
    ):
        """One geometry, taken at one instant, and the same one #190 takes.

        A satellite's range and its crossing speed are two halves of one
        position on the pass. Sizing the cut from the range at closest approach
        and the speed at the middle of the in-view window mixes two points of
        the orbit half a pass apart -- here 349 km against 496 km -- and gives a
        ceiling neither of them has. It also silently disagrees with
        :func:`fit_time_offset` about which baselines a satellite can be
        beam-formed over, so the same pass would be scored two ways depending on
        which command was run.

        The closest approach is still reported, in the candidate and in the
        ranking table: it says how near the satellite came, which is what a
        reader of the table wants. It is not what the statistic is built on.
        """
        [candidate] = enumerate_candidates(
            [record_a], [A_NAME], long_pass.times_jd, layout.ants_itrf,
            min_elevation=long_pass.min_elevation,
        )
        in_view = long_pass.times_jd[np.asarray(candidate["frames"])]
        geometry = dict(
            sigma_transverse_m=SIGMA_TRANSVERSE_SPLIT, int_time=INT_TIME_LONG
        )
        at_closest, _ = ref_coherent(
            record_a, layout.ants_itrf, in_view, layout.a1, layout.a2,
            range_m=candidate["range_m"], **geometry
        )
        mid_window, b_mid = ref_coherent(
            record_a, layout.ants_itrf, in_view, layout.a1, layout.a2, **geometry
        )

        # Not vacuous: at this transverse error the two geometries disagree
        # about the array's longest core baseline.
        assert int(at_closest.sum()) == N_SPLIT_BL < int(mid_window.sum())

        settings = dict(
            taus_s=LONG_GRID, sigma_transverse_m=SIGMA_TRANSVERSE_SPLIT,
            n_null=8, n_null_candidates=1,
        )
        search = run_search(long_pass, [candidate], **settings)
        fit = single_fit(
            long_pass, record_a, taus_s=LONG_GRID,
            sigma_transverse_m=SIGMA_TRANSVERSE_SPLIT,
            min_elevation=long_pass.min_elevation, n_null=8,
        )
        got = search["fits"][0]
        rtol = 1e-5 if precision == "double" else 1e-3

        assert got["range_m"] == pytest.approx(fit["range_m"], rel=1e-6)
        assert got["v_perp_m_s"] == pytest.approx(fit["v_perp_m_s"], rel=1e-6)
        assert got["b_coh"] == pytest.approx(fit["b_coh"], rel=1e-6)
        assert got["b_coh"] == pytest.approx(b_mid, rel=1e-6)
        assert got["n_bl_used"] == fit["n_bl_used"] == int(mid_window.sum())
        assert search["n_bl_used"] == int(mid_window.sum())
        np.testing.assert_allclose(
            np.asarray(search["z2_tau"])[0], np.asarray(fit["z2_tau"]), rtol=rtol
        )
        # The closest approach is what the table reports, and it is a different
        # number: the satellite ends the window 40 % further off than it began.
        assert search["table"][0]["range_m"] == pytest.approx(
            candidate["range_m"], rel=1e-9
        )
        assert candidate["range_m"] < 0.8 * got["range_m"]

    def test_a_sky_with_nothing_up_in_it_is_not_sized_from_under_the_earth(
        self, obs_a, record_c, layout, capsys
    ):
        """A satellite below the horizon cannot fringe the array at all.

        Thirteen thousand kilometres away -- through the ground -- its coherence
        length is kilometres, so sizing the shared set from it readmits every
        baseline the array has and scores a candidate over geometry that means
        nothing. Without the cut there is no honest set, so there is nothing to
        search: the result comes back empty rather than confident, and says why.
        """
        below = enumerate_candidates(
            [record_c], [C_NAME], layout.times_jd, layout.ants_itrf,
            min_elevation=None,
        )

        assert below[0]["max_elevation"] < 0.0

        search = run_search(
            obs_a, below, taus_s=SEARCH_GRID[4:5], n_null=4, n_null_candidates=1
        )

        assert search["table"] == []
        assert search["fits"] == []
        assert search["n_bl_used"] == 0
        assert np.isnan(search["b_coh_max"])
        assert np.asarray(search["z2_tau"]).shape[0] == 0
        assert "horizon" in capsys.readouterr().out

    def test_a_set_with_no_baselines_in_it_is_not_scanned(
        self, obs_a, candidates, record_a, layout, capsys
    ):
        """A cut can leave nothing, and nothing is not a small something.

        A kilometre of transverse orbit error is a satellite nobody can
        beam-form: at 349 km the phase is worth a radian across ten centimetres,
        shorter than the array's shortest spacing. Carrying on from there scans
        zero-length arrays -- an argmax over an empty axis, a null drawn from no
        baselines, a score of zero reported as a measurement -- so the search
        stops and says so, and the single-satellite fit returns the same nothing
        with the same count.
        """
        impossible = 1.0e6
        mean_freq = float(np.mean(FREQS))
        non_auto = layout.a1 != layout.a2
        bl_len = np.asarray(
            baseline_lengths(layout.ants_itrf, layout.a1, layout.a2)
        )
        ranges = [
            satellite_range_and_speed(
                candidate["record"], layout.ants_itrf,
                layout.times_jd[np.asarray(candidate["frames"])],
            )[0]
            for candidate in candidates
        ]

        # Not vacuous: even the farthest candidate, whose ceiling is the highest,
        # cannot reach the shortest baseline the array has.
        assert float(
            tle_coherence_length(mean_freq, max(ranges), impossible)
        ) < bl_len[non_auto].min()

        search = run_search(
            obs_a, candidates, taus_s=SEARCH_GRID[4:5],
            sigma_transverse_m=impossible, n_null=4, n_null_candidates=1,
        )
        fit = single_fit(
            obs_a, record_a, taus_s=SEARCH_GRID[4:5],
            sigma_transverse_m=impossible, n_null=4,
        )

        assert search["table"] == []
        assert search["fits"] == []
        assert search["n_bl_used"] == 0
        assert search["batch_size"] == 0
        assert np.asarray(search["z2_tau"]).shape[0] == 0
        assert "nothing to search" in capsys.readouterr().out
        # The two paths agree about the same emptiness.
        assert fit["n_bl_used"] == 0
        assert np.isnan(fit["tau_best"]) and np.isnan(fit["significance"])
        assert fit["z2_best"] == 0.0

    def test_a_soft_cut_keeps_the_hard_cuts_baselines(self, obs_a, record_a):
        """The taper is a taper, not a wider net.

        ``exp(-(b / b_coh)^2)`` is never exactly zero, so reading "coherent" as
        "weight above zero" hands the sum every baseline the array has -- the
        8 km one whose fringe smears away inside a dump included, at a weight of
        ``1e-210`` that changes no answer and costs a path model apiece. The
        support is the hard cut either way; softness decides how the baselines
        *inside* it are weighted, not which ones are in it.

        Sized at 432.5 m of transverse error so the first baseline outside the
        cut is a 239 m one, whose taper is ``exp(-1.18) = 0.31``: a weight no
        arithmetic can mistake for zero in any precision. (At the default cut
        the nearest outsider is 8 km, whose taper underflows to zero in fp32 and
        not in fp64 -- so a support read off the weights would depend on the
        precision the scan happened to run in.)
        """
        settings = dict(
            taus_s=SEARCH_GRID[3:6], sigma_transverse_m=SIGMA_TRANSVERSE_SPLIT,
            n_null=8,
        )
        hard = single_fit(obs_a, record_a, **settings)
        soft = single_fit(obs_a, record_a, soft_weights=True, **settings)

        assert soft["n_bl_used"] == hard["n_bl_used"] == N_SPLIT_BL
        assert soft["b_coh"] == pytest.approx(hard["b_coh"], rel=1e-9)

    def test_soft_weights_do_not_widen_the_shared_set_either(
        self, obs_a, candidates, records, layout, precision
    ):
        """And the batched search is still the single fit, taper and all.

        The shared set is the hard union -- 28 baselines, the wider of the two
        candidates' cuts -- whether or not the weighting inside it is soft. A
        support taken from the soft weights instead would be every baseline the
        array has, and would build an 8 km path model per candidate to multiply
        it by ``1e-210``.
        """
        settings = dict(
            taus_s=SEARCH_GRID[3:6], sigma_transverse_m=SIGMA_TRANSVERSE_SPLIT,
            n_null=8,
        )
        by_id = {int(record["NORAD_CAT_ID"]): record for record in records}
        rtol = 1e-5 if precision == "double" else 1e-3
        union = np.zeros(len(layout.a1), dtype=bool)
        for candidate in candidates:
            union |= ref_coherent(
                candidate["record"], layout.ants_itrf,
                layout.times_jd[np.asarray(candidate["frames"])], layout.a1,
                layout.a2, sigma_transverse_m=SIGMA_TRANSVERSE_SPLIT,
            )[0]

        hard = run_search(obs_a, candidates, n_null_candidates=1, **settings)
        soft = run_search(
            obs_a, candidates, soft_weights=True, n_null_candidates=1, **settings
        )

        assert soft["n_bl_used"] == hard["n_bl_used"] == int(union.sum())
        hard_count = {
            row["norad_id"]: fit["n_bl_used"]
            for row, fit in zip(hard["table"], hard["fits"])
        }
        for rank, row in enumerate(soft["table"]):
            fit = single_fit(
                obs_a, by_id[row["norad_id"]], soft_weights=True, **settings
            )

            assert soft["fits"][rank]["n_bl_used"] == fit["n_bl_used"]
            assert soft["fits"][rank]["n_bl_used"] == hard_count[row["norad_id"]]
            np.testing.assert_allclose(
                np.asarray(soft["z2_tau"])[rank], np.asarray(fit["z2_tau"]),
                rtol=rtol,
            )

        # The taper is applied, not merely tolerated: down-weighting the longer
        # baselines moves the score, or "soft" would be the hard cut renamed.
        assert not np.allclose(
            np.asarray(soft["z2_best"]), np.asarray(hard["z2_best"]), rtol=1e-3
        )


class TestSearchRanking:
    """What the search says about data it has seen the satellite in."""

    def test_the_injected_satellite_is_ranked_first_and_detected(self, search_a):
        winner = search_a["table"][0]

        assert winner["norad_id"] == A_ID
        assert winner["name"] == A_NAME
        assert winner["tau_best"] == pytest.approx(TAU_A, abs=0.5)
        assert winner["best_chan"] == CHAN_A
        assert winner["best_freq"] == pytest.approx(FREQS[CHAN_A])
        assert winner["significance"] > 5.0
        assert winner["r_max"] > 0.6

    def test_the_train_mate_does_not_masquerade_as_it(self, search_a):
        """The satellite 30 s along the same orbit is where a false positive
        would come from, and it is the thing the real search had to beat: 0.0995
        against a runner-up of 0.0523. Here it scores an order of magnitude
        below, and nowhere near the threshold.
        """
        winner, runner_up = search_a["table"][:2]

        assert runner_up["norad_id"] == B_ID
        assert runner_up["z2_best"] < 0.5 * winner["z2_best"]
        assert runner_up["significance"] < 5.0

    def test_two_contaminators_are_both_found_and_told_apart(self, search_ab):
        """Multiple simultaneous detections are simply all emitted; the fit
        already accepts several satellites."""
        table = search_ab["table"]

        assert [row["norad_id"] for row in table] == [A_ID, B_ID]
        assert [row["best_chan"] for row in table] == [CHAN_A, CHAN_B]
        assert table[0]["tau_best"] == pytest.approx(TAU_A, abs=0.5)
        assert table[1]["tau_best"] == pytest.approx(TAU_B, abs=0.5)
        assert min(row["significance"] for row in table) > 5.0

    def test_pure_noise_ranks_nobody(self, search_noise):
        """The null is what stops a scan over nine offsets, four channels and
        every satellite in the sky from finding one in every dataset.

        Fixed seeds throughout, so this is a statement about one realisation
        rather than about the tail of a distribution.
        """
        assert float(np.max(search_noise["z2_best"])) < 4.0
        assert float(np.nanmax(search_noise["significance"])) < 5.0

    def test_a_satellite_that_sets_is_scored_on_its_in_view_frames(
        self, setting_obs, record_a, layout
    ):
        """The mask is applied *inside* the statistic, because the batched shapes
        cannot be sliced -- and it has to give what slicing gives, or the search
        and the single-satellite fit would report different detections for the
        same pass.
        """
        screened = enumerate_candidates(
            [record_a], [A_NAME], setting_obs.times_jd, layout.ants_itrf,
            min_elevation=setting_obs.min_elevation,
        )
        search = run_search(setting_obs, screened, n_null=16)
        row = search["table"][0]
        fit = single_fit(
            setting_obs, record_a, min_elevation=setting_obs.min_elevation, n_null=8
        )

        assert row["n_frames"] == int(setting_obs.frames.sum()) < N_TIME
        assert row["z2_best"] == pytest.approx(fit["z2_best"], rel=1e-3)
        assert row["tau_best"] == fit["tau_best"] == pytest.approx(TAU_A, abs=0.5)
        # The spectrogram keeps the full time axis and says nothing where nothing
        # was looked at: a zero correlation would read as a measurement.
        r = np.asarray(search["fits"][0]["r_best"])
        assert np.all(np.isnan(r[:, ~setting_obs.frames]))
        assert np.all(np.isfinite(r[:, setting_obs.frames]))


class TestOneCoreNeverDuplicated:
    """The batched search must be the #190 statistic, not a second one."""

    @pytest.mark.parametrize("rank", [0, 1])
    def test_every_row_reproduces_the_single_satellite_scan(
        self, search_a, obs_a, records, precision, rank
    ):
        """Row for row over the whole grid, not merely at the peak.

        A filter that averaged the right samples with a slightly different
        template would still return a plausible ranking; only comparing the curve
        against the one ``fit_time_offset`` computes tells them apart.
        """
        row = search_a["table"][rank]
        record = {int(r["NORAD_CAT_ID"]): r for r in records}[row["norad_id"]]
        fit = single_fit(obs_a, record, n_null=8)
        rtol = 1e-5 if precision == "double" else 1e-3

        np.testing.assert_allclose(
            np.asarray(search_a["z2_tau"])[rank], np.asarray(fit["z2_tau"]),
            rtol=rtol,
        )
        assert row["z2_best"] == pytest.approx(fit["z2_best"], rel=rtol)
        assert row["tau_best"] == fit["tau_best"]
        assert row["best_chan"] == fit["best_chan"]

    @pytest.mark.parametrize(
        "case",
        [
            "a scalar sigma",
            "a sigma per baseline",
            "a sigma per sample",
            "flagged samples",
            "a sigma per baseline and flagged samples",
            "autocorrelations kept",
        ],
    )
    def test_it_holds_for_every_weighting_an_ms_can_hand_it(
        self, obs_a, candidates, records, precision, case
    ):
        """The weights are where the two paths could quietly diverge.

        An MS's noise arrives as a scalar, a column per baseline or a value per
        sample; its flags remove a baseline for a few frames or a whole channel
        for one; and autocorrelations are a flag away from being in the sum. The
        search carries all of that through a shared, batched array while the fit
        slices, so agreement on one shape is not agreement on the others.

        Flagged visibilities are ``nan`` on purpose: an MS carries ``inf`` and
        ``nan`` under its flags, and ``0 * nan`` is ``nan``, which would poison a
        whole channel of either path rather than one baseline's contribution.
        """
        rng = np.random.default_rng(5)
        noise, flags, exclude_autos = SIGMA, None, True
        if "per baseline" in case:
            noise = SIGMA * (1.0 + 0.25 * (np.arange(obs_a.n_bl) % 3))
        if "per sample" in case:
            noise = SIGMA * rng.uniform(0.8, 1.5, obs_a.vis.shape)
        if "flagged" in case:
            flags = np.zeros(obs_a.vis.shape, dtype=bool)
            flags[int(np.flatnonzero(obs_a.a1 != obs_a.a2)[0]), :, 3:7] = True
            flags[:, 1, 11] = True
        if case == "autocorrelations kept":
            exclude_autos = False

        vis = obs_a.vis if flags is None else np.where(flags, np.nan, obs_a.vis)
        observation = SimpleNamespace(**{**vars(obs_a), "vis": vis})
        settings = dict(
            taus_s=SEARCH_GRID[3:6], noise=noise, flags=flags,
            exclude_autos=exclude_autos,
        )
        by_id = {int(record["NORAD_CAT_ID"]): record for record in records}
        rtol = 1e-5 if precision == "double" else 1e-3

        search = run_search(
            observation, candidates, n_null=20, n_null_candidates=1, **settings
        )

        for rank, row in enumerate(search["table"]):
            fit = single_fit(
                observation, by_id[row["norad_id"]], n_null=8, **settings
            )

            assert search["fits"][rank]["n_bl_used"] == fit["n_bl_used"]
            np.testing.assert_allclose(
                np.asarray(search["z2_tau"])[rank], np.asarray(fit["z2_tau"]),
                rtol=rtol,
            )
        # Autocorrelations carry no path difference and so no fringe; kept, they
        # are more baselines in the sum, and both paths have to keep the same
        # ones.
        assert (search["n_bl_used"] > N_COHERENT_BL) is (not exclude_autos)

    @pytest.mark.parametrize("batch_size", [2, 3, 4])
    def test_the_batch_size_is_an_efficiency_knob_and_nothing_else(
        self, obs_a, candidates3, batch_size, exact_rtol
    ):
        """Including the ragged cases: three candidates in batches of two, or of
        four, must pad to a fixed shape -- so the kernel compiles once -- and
        then drop the padding, rather than truncating the last batch or
        recompiling for it.
        """
        settings = dict(taus_s=SEARCH_GRID[3:6], n_null=8, n_null_candidates=1)

        one = run_search(obs_a, candidates3, batch_size=1, **settings)
        many = run_search(obs_a, candidates3, batch_size=batch_size, **settings)

        assert len(one["table"]) == 3
        np.testing.assert_array_equal(many["norad_ids"], one["norad_ids"])
        np.testing.assert_allclose(
            np.asarray(many["z2_tau"]), np.asarray(one["z2_tau"]), rtol=exact_rtol
        )
        np.testing.assert_allclose(many["z2_best"], one["z2_best"], rtol=exact_rtol)

    def test_one_trace_covers_every_batch_of_the_same_shape(self, obs_a,
                                                             candidates, layout):
        """The contract the whole design rests on: candidates batch through one
        jitted program. A Python loop over jitted kernels re-enters the compiler
        per candidate and gives up the point -- one compilation, then a
        device-resident sweep over the whole snapshot.
        """
        keep = np.flatnonzero(
            (layout.a1 != layout.a2)
            & (layout.a1 != FAR_ANT) & (layout.a2 != FAR_ANT)
        )
        paths = np.stack(
            [
                ref_paths(candidate["record"], layout.ants_itrf, layout.times_jd,
                          layout.phase_centre, layout.a1[keep], layout.a2[keep],
                          4, INT_TIME, SEARCH_GRID[3:5])
                for candidate in candidates
            ]
        )
        weights = np.ones((len(candidates), len(keep), 1, 1))
        masks = np.ones((len(candidates), N_TIME))
        vis = obs_a.vis[keep]
        n_traces = 0

        def traced(vis, weights, paths, freqs, mask):
            nonlocal n_traces
            n_traces += 1

            return tau_scan(vis, weights, paths, freqs, mask)

        batched = jax.jit(jax.vmap(traced, in_axes=(None, 0, 0, None, 0)))
        for scale in (1.0, 2.0, 3.0):
            jax.block_until_ready(
                batched(vis * scale, weights, paths, FREQS, masks)
            )

        assert n_traces == 1, f"recompiled {n_traces} times for one batch shape"

    def test_the_production_kernel_compiles_once_for_the_whole_sweep(
        self, obs_a, candidates5
    ):
        """The same contract, measured on the kernel the search actually calls.

        Five candidates in batches of two is two full batches and a ragged one.
        Padding the last rather than shrinking it is what keeps all three to a
        single shape, and so to a single compilation: a shrinking last batch
        would trace the program a second time for every search ever run, which
        is the cost the design exists to avoid. Running the same sweep again
        must add nothing at all -- the cache is the point.
        """
        settings = dict(
            taus_s=SEARCH_GRID[4:6], batch_size=2, n_null=4, n_null_candidates=1
        )

        assert len(candidates5) == 5

        before = _batched_tau_scan._cache_size()
        run_search(obs_a, candidates5, **settings)
        after = _batched_tau_scan._cache_size()
        run_search(obs_a, candidates5, **settings)

        assert after - before <= 1, "the ragged batch traced a second program"
        assert _batched_tau_scan._cache_size() == after


class TestBatchMemoryBudget:
    """How many candidates share a batch is a memory question, not a flag.

    ``--batch-size`` says how many the sweep would *like* to score at once; what
    it can afford depends on the observation. The shared baseline set is the
    union of the above-horizon candidates' hard supports, so a search out to the
    horizon holds most of the array -- and the fringe model is that set times the channels, the frames and
    the sub-steps. The budget is what stops a default of eight from asking for
    17 GB on a real MWA snapshot.
    """

    def test_a_budget_too_small_for_two_candidates_falls_back_to_one(
        self, obs_a, candidates
    ):
        """Never to zero: a batch of one is the smallest sweep there is, and a
        budget below it is a warning about the machine rather than a reason to
        score nothing. Shrinking the batch is a decision about memory and must
        change nothing else -- the same candidates, in the same order, with the
        same scores.
        """
        settings = dict(
            taus_s=SEARCH_GRID[3:6], batch_size=8, n_null=4, n_null_candidates=1
        )
        free_calls, tight_calls = [], []

        uncapped = run_search(
            obs_a, candidates, max_mem_gb=None,
            progress=lambda done, total: free_calls.append((done, total)),
            **settings,
        )
        capped = run_search(
            obs_a, candidates, max_mem_gb=1e-9,
            progress=lambda done, total: tight_calls.append((done, total)),
            **settings,
        )
        n_cand = len(candidates)

        assert uncapped["batch_size"] == min(8, n_cand)
        assert capped["batch_size"] == 1
        # One report per batch, and either way the sweep ends on the last
        # candidate: progress is what a minutes-long search says while it runs.
        assert len(free_calls) == 1 and free_calls[-1] == (n_cand, n_cand)
        assert len(tight_calls) == n_cand and tight_calls[-1] == (n_cand, n_cand)
        np.testing.assert_array_equal(capped["norad_ids"], uncapped["norad_ids"])
        np.testing.assert_allclose(
            np.asarray(capped["z2_tau"]), np.asarray(uncapped["z2_tau"]), rtol=1e-6
        )

    def test_the_budget_admits_as_many_candidates_as_it_fit(self, obs_a,
                                                             candidates3):
        """Not one, and not the whole batch: as many as the budget pays for.

        Sized here from the search's own shared set, so the sum is the one the
        sweep would actually allocate rather than a guess at it.
        """
        taus = SEARCH_GRID[3:6]
        settings = dict(taus_s=taus, batch_size=8, n_null=4, n_null_candidates=1)

        measured = run_search(obs_a, candidates3, max_mem_gb=None, **settings)
        budget = 2.5 * bytes_per_candidate(measured["n_bl_used"], len(taus)) / 1e9
        fitted = run_search(obs_a, candidates3, max_mem_gb=budget, **settings)

        assert measured["batch_size"] == len(candidates3) == 3
        assert fitted["batch_size"] == 2
        np.testing.assert_allclose(
            np.asarray(fitted["z2_tau"]), np.asarray(measured["z2_tau"]), rtol=1e-6
        )


class TestNullAndProgress:
    """Calibrating the top of the ranking, and saying how far the sweep got."""

    def test_the_significance_is_the_score_read_against_the_null(self, search_a):
        for fit, row in zip(search_a["fits"], search_a["table"]):
            if not np.isfinite(row["significance"]):
                continue
            null = np.asarray(fit["null"])

            assert null.shape == (N_NULL,)
            assert row["null_mean"] == pytest.approx(float(null.mean()))
            assert row["null_std"] == pytest.approx(float(null.std()))
            assert row["significance"] == pytest.approx(
                (row["z2_best"] - null.mean()) / null.std(), rel=1e-6
            )

    def test_only_the_top_candidates_are_calibrated_against_a_null(self, obs_a,
                                                                    candidates):
        """The null is a few hundred extra scans *per satellite*; over a whole
        constellation that is the cost of the search twice over, spent on
        candidates nothing will be reported for. The ones below the cut are still
        scored and still ranked -- they simply carry no significance.
        """
        search = run_search(
            obs_a, candidates, taus_s=SEARCH_GRID[3:6], n_null_candidates=1, n_null=16
        )
        top, rest = search["table"][0], search["table"][1:]

        assert np.isfinite(top["significance"])
        assert np.asarray(search["fits"][0]["null"]).shape == (16,)
        for row, fit in zip(rest, search["fits"][1:]):
            assert np.isfinite(row["z2_best"])
            assert np.isnan(row["significance"])
            assert np.isnan(row["null_mean"]) and np.isnan(row["null_std"])
            # No draws were taken, so there is no null: an empty array or a
            # nan-filled one, but nothing that could be read as a measurement.
            assert not np.isfinite(np.asarray(fit["null"], dtype=float)).any()

    def test_the_sweep_reports_its_progress_to_the_last_candidate(self, obs_a,
                                                                   candidates):
        """A search over a constellation runs for minutes; a command that says
        nothing for all of them cannot be told from a hung one."""
        calls = []

        search = run_search(
            obs_a, candidates, taus_s=SEARCH_GRID[4:5], n_null=8, batch_size=1,
            progress=lambda done, total: calls.append((done, total)),
        )
        n_cand = len(search["table"])

        assert calls[-1] == (n_cand, n_cand)
        assert [done for done, _ in calls] == sorted({done for done, _ in calls})
        assert all(total == n_cand and 0 < done <= n_cand for done, total in calls)


# ---------------------------------------------------------------------------
# Stage 3: selection, warnings and the artifacts
# ---------------------------------------------------------------------------

class TestSelectDetections:
    """Ranking is not deciding: the threshold and the two warnings."""

    def test_the_threshold_is_inclusive_and_needs_a_measured_significance(self):
        """A candidate with no null drawn has a ``nan`` significance and is not a
        detection at any threshold -- ``nan >= 5`` is false, but so is
        ``nan < 5``, and only one of those readings is safe to rely on."""
        search = fake_search([(9.0, 0.5, 7.0), (5.0, 0.0, 5.0), (4.0, 0.0, float("nan"))])

        detected = select_detections(search, threshold_sigma=5.0)["detected"]

        assert [row["norad_id"] for row in detected] == [90000, 90001]
        assert select_detections(search, threshold_sigma=7.1)["detected"] == []

    def test_the_detections_come_back_in_rank_order(self, search_ab):
        detected = select_detections(search_ab)["detected"]

        assert [row["norad_id"] for row in detected] == [A_ID, B_ID]
        assert detected[0]["z2_best"] > detected[1]["z2_best"]
        assert detected == search_ab["table"]

    def test_nothing_is_detected_in_noise(self, search_noise):
        selection = select_detections(search_noise)

        assert selection["detected"] == []
        assert not any("scan edge" in w for w in selection["warnings"])

    def test_a_close_runner_up_is_warned_about(self):
        """Satellites in the same train partially match each other's fringes, so
        a winner that is not clear of the field is a result to look at twice
        rather than a satellite to name."""
        close = select_detections(fake_search([(1.0, 0.0, 9.0), (0.7, 0.0, 6.0)]))
        clear = select_detections(fake_search([(1.0, 0.0, 9.0), (0.6, 0.0, 6.0)]))

        assert sum("runner-up" in w for w in close["warnings"]) == 1
        assert not any("runner-up" in w for w in clear["warnings"])
        assert "90001" in "".join(close["warnings"])

    def test_the_runner_up_warning_does_not_wait_for_a_detection(self):
        """It is a statement about the ranking, not about the winner: two
        candidates level at the noise floor say the search could not separate
        them, which is worth knowing even when neither is named."""
        selection = select_detections(fake_search([(1.0, 0.0, 1.2), (0.9, 0.0, 1.1)]))

        assert selection["detected"] == []
        assert any("runner-up" in w for w in selection["warnings"])

    def test_the_ratio_is_the_callers(self, search_ab):
        """On the real two-satellite fixture the winner is clear at the default
        1.5x and not at 3x, so the flag really is what decides."""
        assert not any(
            "runner-up" in w for w in select_detections(search_ab)["warnings"]
        )
        assert any(
            "runner-up" in w
            for w in select_detections(search_ab, runner_up_ratio=3.0)["warnings"]
        )

    def test_a_detection_at_the_edge_of_the_scan_is_warned_about(self, search_edge,
                                                                  search_a):
        """A peak on the last grid point is a peak that may be off the grid: the
        offset is at least that large, and the number reported is a floor rather
        than a measurement. Widening ``--tau-max`` is the fix, and the run has to
        say so -- while the same satellite peaking inside the grid says nothing.
        """
        selection = select_detections(search_edge)
        edge = [w for w in selection["warnings"] if "scan edge" in w]

        assert selection["detected"][0]["norad_id"] == A_ID
        assert selection["detected"][0]["tau_best"] == pytest.approx(TAU_EDGE)
        assert len(edge) == 1
        assert str(A_ID) in edge[0]
        assert not any(
            "scan edge" in w for w in select_detections(search_a)["warnings"]
        )

    def test_only_a_detection_can_sit_at_the_edge(self):
        """An undetected candidate's best offset is where noise happened to peak;
        that it did so at the end of the grid says nothing about anything."""
        grid = SEARCH_GRID
        selection = select_detections(
            fake_search([(9.0, 0.0, 8.0), (1.0, float(grid[-1]), 1.0)])
        )

        assert not any("scan edge" in w for w in selection["warnings"])


class TestWriteSearchResults:
    """The ranking table as a file: always written, detections or not."""

    def test_it_carries_every_column_of_the_ranking(self, tmp_path, search_ab):
        selection = select_detections(search_ab)
        path = str(tmp_path / "ranking.npz")

        written = write_search_results(path, search_ab, selection, 5.0)

        assert written == path
        n_cand = len(search_ab["table"])
        with np.load(written, allow_pickle=False) as npz:
            for key in (
                "norad_ids", "names", "z2_best", "tau_best", "best_chan",
                "best_freq", "significance", "null_mean", "null_std",
                "max_elevation", "range_m", "r_max", "detected",
            ):
                assert npz[key].shape == (n_cand,), key
            assert npz["z2_tau"].shape == (n_cand, len(SEARCH_GRID), N_FREQ)
            assert npz["frames"].shape == (n_cand, N_TIME)
            np.testing.assert_allclose(npz["tau_grid"], SEARCH_GRID)
            np.testing.assert_array_equal(npz["norad_ids"], [A_ID, B_ID])
            assert list(npz["names"]) == [A_NAME, B_NAME]
            assert float(npz["threshold_sigma"]) == 5.0
            assert int(npz["n_bl_used"]) == N_COHERENT_BL
            assert float(npz["median_z2"]) == pytest.approx(search_ab["median_z2"])

    def test_the_detected_flags_are_the_selections(self, tmp_path, search_a):
        selection = select_detections(search_a)
        path = write_search_results(
            str(tmp_path / "ranking.npz"), search_a, selection, 5.0
        )

        with np.load(path, allow_pickle=False) as npz:
            assert npz["detected"].dtype == bool
            np.testing.assert_array_equal(npz["detected"], [True, False])

    def test_the_ranking_survives_a_search_that_found_nothing(self, tmp_path,
                                                              search_noise):
        """"Nothing above the threshold" is a result about 128 satellites, and
        the evidence for it is the table."""
        selection = select_detections(search_noise)
        path = write_search_results(
            str(tmp_path / "ranking.npz"), search_noise, selection, 5.0
        )

        with np.load(path, allow_pickle=False) as npz:
            assert not npz["detected"].any()
            assert npz["z2_best"].shape == (len(search_noise["table"]),)


class TestWriteConfigFragment:
    """The deliverable the issue asks for: a config fragment, ready to merge."""

    @staticmethod
    def _numbers(line):
        return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)]

    def test_it_parses_as_a_tabascal_satellites_section(self, tmp_path, search_ab):
        selection = select_detections(search_ab)
        path = str(tmp_path / "found.yaml")

        written = write_config_fragment(path, selection)
        text = open(written).read()

        assert yaml.safe_load(text)["satellites"]["norad_ids"] == [A_ID, B_ID]
        assert "extra_orbit_dir" not in yaml.safe_load(text)["satellites"]

    def test_the_shifted_records_are_named_when_there_are_some(self, tmp_path,
                                                               search_ab):
        """Pointing at them with no age ceiling is the whole point: a later run
        reproduces the trajectories the search measured, whatever SatChecker
        serves by then."""
        selection = select_detections(search_ab)

        path = write_config_fragment(
            str(tmp_path / "found.yaml"), selection,
            shifted_orbit_dir=str(tmp_path / "tles"),
        )

        satellites = yaml.safe_load(open(path))["satellites"]
        assert satellites["extra_orbit_dir"] == str(tmp_path / "tles")
        assert satellites["extra_orbit_max_age_days"] is None

    def test_each_detection_is_written_out_in_a_comment(self, tmp_path, search_ab):
        """A bare list of IDs is not auditable. Beside it, per satellite, what it
        was detected on: the offset the curves must be extracted at, the score,
        the significance, and the channel -- which is what tells the user which
        channels to fit.
        """
        selection = select_detections(search_ab)
        path = write_config_fragment(str(tmp_path / "found.yaml"), selection)
        comments = [
            line for line in open(path).read().splitlines() if line.startswith("#")
        ]

        for row in selection["detected"]:
            named = [line for line in comments if str(row["norad_id"]) in line]
            assert len(named) == 1, f"{row['norad_id']} needs one comment line"
            assert row["name"] in named[0]
            numbers = self._numbers(named[0])
            for want, tol in (
                (row["tau_best"], 5e-3),
                (row["z2_best"], 5e-3),
                (row["significance"], 5e-2),
                (row["best_chan"], 1e-9),
                (row["best_freq"] / 1e6, 5e-3),
            ):
                assert any(abs(x - want) <= tol for x in numbers), (
                    f"{want} missing from {named[0]!r}"
                )

    def test_a_search_that_found_nothing_still_writes_an_empty_list(self, tmp_path,
                                                                     search_noise):
        """So the artifact is there to point at either way, and says why it is
        empty rather than leaving the reader to guess the run failed."""
        selection = select_detections(search_noise)

        path = write_config_fragment(str(tmp_path / "found.yaml"), selection)

        text = open(path).read()
        assert yaml.safe_load(text)["satellites"]["norad_ids"] == []
        assert any(
            line.startswith("#") and "threshold" in line
            for line in text.splitlines()
        )


class TestRankingPlot:
    """The chart the issue asks for, as a file that exists."""

    def test_it_writes_a_png(self, tmp_path, search_a):
        path = str(tmp_path / "ranking.png")

        got = plot_candidate_ranking(
            search_a, select_detections(search_a), path, title="synthetic"
        )

        assert got == path
        assert os.path.getsize(path) > 0


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------

SCAN = ("--tau-max", "2", "--tau-step", "0.5", "--null-draws", "32")


def _cli(*argv):
    from tabascal.scripts.run_tabascal import build_parser

    return build_parser().parse_args(["search", *argv])


@pytest.fixture
def stub_ms(monkeypatch, layout):
    """Stand in for ``tabascal.ms.read_ms`` with one of the observations."""
    def install(observation):
        import tabascal.ms

        gh0 = (gast_deg(observation.times_jd) - layout.phase_centre["ra"]) % 360
        ants_uvw = itrf_to_uvw_numpy(
            observation.ants_itrf, gh0, layout.phase_centre["dec"]
        )
        data = {
            "ra": layout.phase_centre["ra"],
            "dec": layout.phase_centre["dec"],
            "ants_itrf": observation.ants_itrf,
            "times_mjd": jd_to_mjd(observation.times_jd),
            "times_jd": observation.times_jd,
            "time_scale": "utc",
            "freqs": observation.freqs,
            "chan_width": float(FREQS[1] - FREQS[0]),
            "chan_widths": np.full(N_FREQ, float(FREQS[1] - FREQS[0])),
            "vis_obs": observation.vis,
            "n_freq": N_FREQ,
            "n_time": observation.n_time,
            "n_bl": observation.n_bl,
            "n_ant": N_ANT,
            "int_time": observation.int_time,
            "uvw": ants_uvw[:, observation.a1] - ants_uvw[:, observation.a2],
            "flags": np.zeros(observation.vis.shape, dtype=bool),
            "noise": SIGMA,
            "a1": observation.a1,
            "a2": observation.a2,
        }
        monkeypatch.setattr(tabascal.ms, "read_ms", lambda *a, **k: data)

        return data

    return install


@pytest.fixture
def quiet_precision(monkeypatch):
    """Spy on the scan's precision switch instead of flipping it.

    Enabling x64 is process-wide and this suite is run in both precisions from
    the outside, so a test that let the real setter run would either fight the
    session or leak into every test after it. One test uses the real one, on
    purpose, to see what it leaves behind.
    """
    import tabascal.scripts.sat_search as script

    asked = []
    monkeypatch.setattr(script, "set_precision_for_scan", asked.append)

    return asked


class TestSearchArguments:
    """``tabascal search``: its own argument surface, and its own defaults."""

    def test_the_defaults_are_the_issues_defaults(self):
        """And it is its own subcommand rather than a flag on ``light-curve``:
        discovery has a different input (a snapshot, not a satellite), a
        different output (a config fragment) and different exit semantics."""
        args = _cli("-ms", "obs.ms", "--tle-dir", "tles")

        assert args.command == "search"
        assert args.ms_path == "obs.ms"
        assert args.data_col == "DATA"
        assert args.corr == "xx"
        assert args.freq is None
        assert args.name_filter is None
        assert args.extra_orbit_dir is None
        assert args.min_elevation == 0.0
        assert args.tau_max == 4.0
        # Coarser than light-curve's 0.25: the scan is run once per candidate.
        assert args.tau_step == 0.5
        assert args.n_fine == 40
        assert args.sigma_transverse == 300.0
        assert args.soft_weights is False
        assert args.null_draws == 200
        assert args.null_jitter == 50.0
        assert args.threshold == 5.0
        assert args.null_top == 5
        assert args.batch_size == 8
        assert args.max_mem_gb == 4.0
        assert args.runner_up_ratio == 1.5
        assert args.save_all is False
        assert args.output is None
        assert args.write_shifted_tle is None
        assert args.shifted_tle is True
        assert args.plot is False
        assert args.verbose is False
        assert args.precision is None

    def test_every_flag_is_accepted(self, tmp_path):
        args = _cli(
            "-ms", "obs.ms", "--tle-dir", str(tmp_path), "--name-filter", "STARLINK",
            "-dc", "CAL_DATA", "-cr", "yy", "-f", "175e6", "--min-elevation", "15",
            "--tau-max", "3", "--tau-step", "0.25", "--n-fine", "16",
            "--sigma-transverse", "150", "--soft-weights", "--null-draws", "50",
            "--null-jitter", "80", "--threshold", "7.5", "--null-top", "3",
            "--batch-size", "2", "--max-mem-gb", "0.5",
            "--runner-up-ratio", "2.0", "--save-all",
            "-o", str(tmp_path / "out"), "--write-shifted-tle", str(tmp_path),
            "-p", "-v", "--precision", "double",
        )

        assert (args.tle_dir, args.name_filter) == (str(tmp_path), "STARLINK")
        assert (args.data_col, args.corr, args.freq) == ("CAL_DATA", "yy", 175e6)
        assert (args.min_elevation, args.tau_max, args.tau_step) == (15.0, 3.0, 0.25)
        assert (args.n_fine, args.sigma_transverse) == (16, 150.0)
        assert (args.soft_weights, args.save_all, args.plot, args.verbose) == (
            True, True, True, True
        )
        assert (args.null_draws, args.null_jitter, args.threshold) == (50, 80.0, 7.5)
        assert (args.null_top, args.batch_size, args.runner_up_ratio) == (3, 2, 2.0)
        assert args.max_mem_gb == 0.5
        assert args.output == str(tmp_path / "out")
        assert args.write_shifted_tle == str(tmp_path)
        assert args.precision == "double"

    def test_the_candidates_have_to_come_from_exactly_one_place(self):
        """A snapshot and an explicit list both name the candidates, and there is
        no rule for which would win; naming neither is not a search, and neither
        is a search with no visibilities to search."""
        with pytest.raises(SystemExit):
            _cli("-ms", "obs.ms")
        with pytest.raises(SystemExit):
            _cli("-ms", "obs.ms", "--tle-dir", "tles", "-n", "25544")
        with pytest.raises(SystemExit):
            _cli("-ms", "obs.ms", "-n", "25544", "-np", "ids.txt")
        with pytest.raises(SystemExit):
            _cli("--tle-dir", "tles")

        assert _cli("-ms", "obs.ms", "-n", "25544,27386").norad_ids == "25544,27386"
        assert _cli("-ms", "obs.ms", "-np", "ids.txt").norad_path == "ids.txt"

    def test_there_is_no_way_to_turn_the_scan_off_or_gate_it_differently(self):
        """The tau scan is the search -- at tau = 0 the MWA case scored 0.045
        against a candidate median of 0.0446, which is no detection at all -- so
        there is no ``--fit-offset`` to forget. Gating is spelled the other way
        round: saving is threshold-gated by default and ``--save-all`` opens it.
        """
        for flag in ("--fit-offset", "--only-detections"):
            with pytest.raises(SystemExit):
                _cli("-ms", "obs.ms", "--tle-dir", "tles", flag)

    def test_the_shifted_records_are_written_unless_refused(self, tmp_path):
        """The config fragment points at them, so they are written by default;
        asking for both a directory and no directory is refused rather than
        silently resolved."""
        assert _cli("-ms", "o.ms", "--tle-dir", "t", "--no-shifted-tle").shifted_tle is False
        with pytest.raises(SystemExit):
            _cli("-ms", "o.ms", "--tle-dir", "t", "--no-shifted-tle",
                 "--write-shifted-tle", str(tmp_path))

    def test_the_parser_module_still_imports_without_jax(self):
        """``tabascal -h`` builds every subcommand's parser, and must not pay for
        the run stack to do it."""
        import tabascal.scripts.sat_search as script

        with open(script.__file__) as fh:
            source = fh.read()

        assert not re.search(r"(?m)^(import jax|from jax)", source)

    @pytest.mark.parametrize(
        "flag, value",
        [
            ("--tau-step", "0"),
            ("--tau-max", "-1.0"),
            ("--n-fine", "0"),
            ("--null-draws", "1"),
            ("--null-jitter", "0"),
            ("--tau-step", "inf"),
            ("--threshold", "nan"),
        ],
    )
    def test_a_nonsense_scan_setting_is_refused_by_name(self, tmp_path, flag, value):
        """The same rules, and the same messages, as ``light-curve --fit-offset``:
        one scan means one set of checks. Refused before the MS is read, so a
        typo costs nothing.
        """
        from tabascal.scripts.sat_search import run

        args = _cli("-ms", str(tmp_path / "missing.ms"), "--tle-dir", str(tmp_path),
                    flag, value)

        with pytest.raises(SystemExit, match=re.escape(flag)):
            run(args)

    def test_a_snapshot_and_an_extra_orbit_directory_are_two_sources(self, tmp_path):
        """``--extra-orbit-dir`` is where an *explicit* list is resolved from.

        With ``--tle-dir`` the snapshot already is the source, so a second
        directory can only be read for satellites the snapshot never named -- or
        quietly ignored, which is worse. Refused by name, before the MS is read.
        """
        from tabascal.scripts.sat_search import check_arguments

        with pytest.raises(SystemExit) as refused:
            check_arguments(
                _cli("-ms", "obs.ms", "--tle-dir", str(tmp_path),
                     "--extra-orbit-dir", str(tmp_path))
            )

        message = str(refused.value)
        assert "--tle-dir" in message and "--extra-orbit-dir" in message
        # Either source on its own is exactly what it always was.
        check_arguments(_cli("-ms", "obs.ms", "--tle-dir", str(tmp_path)))
        check_arguments(
            _cli("-ms", "obs.ms", "-n", str(A_ID), "--extra-orbit-dir", str(tmp_path))
        )

    @pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
    def test_a_memory_budget_that_is_not_one_is_refused_by_name(self, tmp_path,
                                                                 value):
        """A budget of nothing admits no candidate and a budget of nan admits
        none either, having lost the comparison; an infinite one is not a budget.
        Each would fail somewhere deep in the sweep, so it is refused at the
        flag, before the MS is read.
        """
        from tabascal.scripts.sat_search import run

        args = _cli("-ms", str(tmp_path / "missing.ms"), "--tle-dir", str(tmp_path),
                    "--max-mem-gb", value)

        with pytest.raises(SystemExit, match=re.escape("--max-mem-gb")):
            run(args)

    def test_the_name_filter_needs_a_snapshot_to_filter(self, tmp_path):
        """With ``-n`` the candidates are named one by one and a substring filter
        could only quietly drop some of them."""
        from tabascal.scripts.sat_search import run

        with pytest.raises(SystemExit, match="--name-filter"):
            run(_cli("-ms", str(tmp_path / "missing.ms"), "-n", str(A_ID),
                     "--name-filter", "STARLINK"))


class TestSearchCommandLine:
    """The search as it is actually run: parsed arguments to files on disk."""

    def test_it_ranks_saves_and_reports_a_detection(
        self, tmp_path, obs_a, snapshot_dir, stub_ms, quiet_precision, capsys
    ):
        """The whole command on the synthetic snapshot: three records in, one
        satellite named, and every artifact the next run needs beside it."""
        from tabascal.scripts.sat_search import run

        ms = stub_ms(obs_a)
        ms_path = str(tmp_path / "obs.ms")

        assert run(_cli("-ms", ms_path, "--tle-dir", snapshot_dir, *SCAN, "-p")) == 0

        # The default output stem sits beside the MS, named for the column read.
        stem = str(tmp_path / "sat_search" / "DATA")
        with np.load(f"{stem}_ranking.npz", allow_pickle=False) as npz:
            np.testing.assert_array_equal(npz["norad_ids"], [A_ID, B_ID])
            np.testing.assert_array_equal(npz["detected"], [True, False])
            assert npz["tau_best"][0] == pytest.approx(TAU_A, abs=0.5)
            assert npz["best_chan"][0] == CHAN_A
        # Only the detection's curves, extracted at the offset that was fitted.
        with np.load(f"{stem}_light_curves.npz", allow_pickle=False) as npz:
            np.testing.assert_array_equal(npz["norad_ids"], [A_ID])
            assert npz["light_curves"].shape == (1, N_TIME, N_FREQ)
            assert npz["tau_best"][0] == pytest.approx(TAU_A, abs=0.5)
            assert bool(npz["detected"][0]) is True
        assert yaml.safe_load(open(f"{stem}_config.yaml"))["satellites"][
            "norad_ids"
        ] == [A_ID]
        # And the shifted record is written where a later run can find it.
        resolved, _ = _select_from_extra_dir(
            f"{stem}_shifted_tles", {A_ID, B_ID}, float(np.mean(ms["times_jd"])), None
        )
        assert set(resolved) == {A_ID}
        for png in (f"{stem}_ranking.png", f"{stem}_offset_{A_ID}.png"):
            assert os.path.getsize(png) > 0

        printed = capsys.readouterr().out
        assert "DETECTED" in printed
        assert str(A_ID) in printed and str(B_ID) in printed
        assert f"{stem}_config.yaml" in printed

    def test_nothing_above_the_threshold_is_a_result_not_a_failure(
        self, tmp_path, obs_noise, snapshot_dir, stub_ms, quiet_precision, capsys
    ):
        """Exit 3, scriptably: the scan ran and found nothing, which is neither
        success nor a crash. The ranking is written either way -- it is the
        evidence for the negative -- and no curve is extracted at an offset that
        is not a detection.
        """
        from tabascal.scripts.sat_search import run

        stub_ms(obs_noise)
        stem = str(tmp_path / "out")

        code = run(_cli("-ms", str(tmp_path / "obs.ms"), "--tle-dir", snapshot_dir,
                        *SCAN, "-o", stem))

        assert code == 3
        assert os.path.exists(f"{stem}_ranking.npz")
        assert yaml.safe_load(open(f"{stem}_config.yaml"))["satellites"][
            "norad_ids"
        ] == []
        assert not os.path.exists(f"{stem}_light_curves.npz")
        assert not os.path.exists(f"{stem}_shifted_tles")
        assert "No candidate cleared the threshold" in capsys.readouterr().out

    def test_save_all_keeps_every_candidates_curve(
        self, tmp_path, obs_a, snapshot_dir, stub_ms, quiet_precision
    ):
        """Threshold-gated saving is the default because a search meets hundreds
        of candidates; ``--save-all`` is for the run where the negatives are the
        point."""
        from tabascal.scripts.sat_search import run

        stub_ms(obs_a)
        stem = str(tmp_path / "out")

        run(_cli("-ms", str(tmp_path / "obs.ms"), "--tle-dir", snapshot_dir, *SCAN,
                 "-o", stem, "--save-all"))

        with np.load(f"{stem}_light_curves.npz", allow_pickle=False) as npz:
            np.testing.assert_array_equal(npz["norad_ids"], [A_ID, B_ID])
            np.testing.assert_array_equal(npz["detected"], [True, False])

    def test_an_explicit_list_of_ids_is_searched_the_same_way(
        self, tmp_path, obs_a, stub_ms, stub_orbits, quiet_precision
    ):
        """``-n`` is the other way in: the candidates are resolved through the
        run's own orbit resolver rather than read from a snapshot."""
        from tabascal.scripts.sat_search import run

        stub_ms(obs_a)
        stem = str(tmp_path / "out")

        code = run(_cli("-ms", str(tmp_path / "obs.ms"), "-n", f"{B_ID},{A_ID}",
                        *SCAN, "-o", stem))

        assert code == 0
        with np.load(f"{stem}_ranking.npz", allow_pickle=False) as npz:
            np.testing.assert_array_equal(npz["norad_ids"], [A_ID, B_ID])
            np.testing.assert_array_equal(npz["detected"], [True, False])

    def test_no_candidate_above_the_horizon_stops_the_run(
        self, tmp_path, obs_a, record_c, stub_ms, quiet_precision
    ):
        """Before any scan: with nothing up there is no search to run, and a
        ranking of zero rows would say the search failed rather than that the
        snapshot was the wrong one."""
        from tabascal.scripts.sat_search import run

        stub_ms(obs_a)
        directory = tmp_path / "empty-sky"
        directory.mkdir()
        save_orbits_for_reuse(str(directory / "tles.json"), [C_ID], [record_c])

        with pytest.raises(SystemExit, match="(?i)horizon|elevation"):
            run(_cli("-ms", str(tmp_path / "obs.ms"), "--tle-dir", str(directory),
                     *SCAN, "-o", str(tmp_path / "out")))

    def test_a_tight_memory_budget_still_finds_the_satellite(
        self, tmp_path, obs_a, snapshot_dir, stub_ms, quiet_precision
    ):
        """The budget shrinks the batch and nothing else: the same detection,
        the same exit status, one candidate at a time."""
        from tabascal.scripts.sat_search import run

        stub_ms(obs_a)
        stem = str(tmp_path / "out")

        code = run(_cli("-ms", str(tmp_path / "obs.ms"), "--tle-dir", snapshot_dir,
                        *SCAN, "-o", stem, "--max-mem-gb", "1e-9"))

        assert code == 0
        with np.load(f"{stem}_ranking.npz", allow_pickle=False) as npz:
            np.testing.assert_array_equal(npz["norad_ids"], [A_ID, B_ID])
            np.testing.assert_array_equal(npz["detected"], [True, False])
            assert npz["tau_best"][0] == pytest.approx(TAU_A, abs=0.5)

    def test_a_snapshot_of_satellites_under_the_earth_stops_the_run(
        self, tmp_path, obs_a, record_c, stub_ms, quiet_precision
    ):
        """Lowering ``--min-elevation`` past the horizon gets a candidate through
        the screen; it does not get one into the sum.

        There is no baseline set a satellite on the far side of the Earth can be
        scored over, so the run stops exactly as it does when the screen empties
        -- rather than ranking a candidate on geometry that means nothing and
        emitting its NORAD ID for a later run to model.
        """
        from tabascal.scripts.sat_search import run

        stub_ms(obs_a)
        directory = tmp_path / "under-the-earth"
        directory.mkdir()
        save_orbits_for_reuse(str(directory / "tles.json"), [C_ID], [record_c])

        with pytest.raises(SystemExit, match="(?i)horizon"):
            run(_cli("-ms", str(tmp_path / "obs.ms"), "--tle-dir", str(directory),
                     "--min-elevation", "-90", *SCAN, "-o", str(tmp_path / "out")))

    @pytest.mark.parametrize(
        "flags, want", [((), "single"), (("--precision", "double"), "double")]
    )
    def test_the_scan_sets_its_own_precision(
        self, tmp_path, obs_a, snapshot_dir, stub_ms, quiet_precision, flags, want
    ):
        """There is no config on this path, and tabascal's own default is
        ``model.precision: single`` -- so the search follows it rather than
        whatever the interpreter happens to have been left in."""
        from tabascal.scripts.sat_search import run

        stub_ms(obs_a)

        run(_cli("-ms", str(tmp_path / "obs.ms"), "--tle-dir", snapshot_dir, *SCAN,
                 "-o", str(tmp_path / "out"), *flags))

        assert quiet_precision == [want]

    @pytest.mark.parametrize("precision", ["double", "single"])
    def test_the_run_puts_the_process_back_as_it_found_it(
        self, tmp_path, obs_noise, snapshot_dir, stub_ms, precision
    ):
        """``run`` is a function in an importable module, so a notebook or a
        pipeline can call it between two things of its own. Setting the scan's
        precision means flipping process-wide JAX flags, and
        ``jax_default_matmul_precision`` is the one that would go unnoticed: on
        Ampere+ GPUs it decides whether an f32 matmul is really f32 or TF32.

        The real setter runs here, unpatched, which is the only way to see what
        it leaves behind. The matmul flag is moved off the session's own value
        first so that restoring it is visible rather than coincidental, and both
        are put back in a finally: a test that protects the rest of the session
        must not be the thing that corrupts it.
        """
        from tabascal.scripts.sat_search import run

        stub_ms(obs_noise)
        session_matmul = jax.config.jax_default_matmul_precision
        was_x64 = jax.config.read("jax_enable_x64")
        marker = "float32"
        jax.config.update("jax_default_matmul_precision", marker)
        try:
            run(_cli("-ms", str(tmp_path / "obs.ms"), "--tle-dir", snapshot_dir,
                     *SCAN, "-o", str(tmp_path / "out"), "--precision", precision))

            assert jax.config.read("jax_enable_x64") == was_x64
            assert jax.config.jax_default_matmul_precision == marker
        finally:
            jax.config.update("jax_enable_x64", was_x64)
            jax.config.update("jax_default_matmul_precision", session_matmul)

    @pytest.mark.parametrize("code", [0, 3])
    def test_main_exits_with_the_run_status(self, monkeypatch, tmp_path, code):
        """Both halves of the contract: the top-level command dispatches
        ``search`` to this module, and the status a script reads is the one the
        search decided -- 0 for a detection, 3 for a clean nothing."""
        import tabascal.scripts.run_tabascal as top
        import tabascal.scripts.sat_search as script

        monkeypatch.setattr(script, "run", lambda args: code)
        arguments = ["-ms", "obs.ms", "--tle-dir", str(tmp_path)]

        for main, argv in (
            (script.main, ["sat-search", *arguments]),
            (top.main, ["tabascal", "search", *arguments]),
        ):
            monkeypatch.setattr("sys.argv", argv)

            with pytest.raises(SystemExit) as exit:
                main()

            assert exit.value.code == code
