from collections.abc import Mapping
from math import isfinite

from tabascal.orbit import TLEError, get_tles_by_id
from satchecker_client.records import KIND_TLE, record_elements, record_kind
from tabascal.distributed import (
    make_global,
    padded_rfi_count,
    rfi_sharding,
    sharded_rfi_zeros,
    sharding_enabled,
)
from tabascal.dist import standard_normal
from tabascal.transform import affine_transform_full
from tabascal.interferometry import get_rfi_phase, get_rfi_phase_numpy, itrf_to_uvw_numpy
from tabascal.components import Component, assert_attr_shape
from tabascal.fft_gp import (
    knee_from_corr_scale,
    latent_to_signal,
    latent_to_signal_init,
)
from tabascal.timing import measure_runtime
from tabascal.time import gast_deg, skyfield_time, timescale

import sgp4jax
from sgp4jax import WGS72 as gravity
from sgp4jax._sgp4init import sgp4init

import jax.numpy as jnp
from jax import vmap, Array
import numpy as np
from numpy.typing import NDArray

from sgp4.api import WGS72, Satrec

from skyfield.api import Distance, wgs84
from skyfield.toposlib import ITRSPosition

from skyfield.api import EarthSatellite

#: Julian Date of 1949 December 31 00:00 UT, the epoch SGP4 counts days from.
_SGP4_EPOCH_JD = 2433281.5


def _earth_satellite(record, ts):
    """A Skyfield ``EarthSatellite`` for one orbit record, whichever kind it is.

    A TLE goes through Skyfield's line parser exactly as it always has, so
    nothing about the TLE path changes. An OMM has no lines to parse — that is
    the whole point of the format — so its element set is loaded straight into an
    ``sgp4.Satrec`` via ``sgp4init``, which is the entry point the sgp4 library
    provides for precisely this. Both end up as the same propagator over the same
    model; only the way the elements are read in differs.

    Units: ``sgp4init`` wants radians and rad/min, while OMM (and tabascal's
    element columns) use degrees and rev/day.

    ``ndot`` and ``nddot`` are passed as zero. SGP4 models drag through ``bstar``
    alone and never reads them during propagation — they exist in the TLE format
    for other consumers — so dropping them in the client costs nothing here.
    """
    if record_kind(record) == KIND_TLE:
        return EarthSatellite(record["TLE_LINE1"], record["TLE_LINE2"], ts=ts)

    elements = record_elements(record)
    satrec = Satrec()
    satrec.sgp4init(
        WGS72,
        "i",  # improved mode, matching what twoline2rv uses for the TLE path
        int(record["NORAD_CAT_ID"]),
        elements["EPOCH_JD"] - _SGP4_EPOCH_JD,
        float(elements["BSTAR"]),
        0.0,  # ndot: stored by the TLE format, unused by the propagator
        0.0,  # nddot: likewise
        float(elements["ECCENTRICITY"]),
        np.deg2rad(elements["ARG_OF_PERICENTER"]),
        np.deg2rad(elements["INCLINATION"]),
        np.deg2rad(elements["MEAN_ANOMALY"]),
        elements["MEAN_MOTION"] * 2.0 * np.pi / 1440.0,  # rev/day -> rad/min
        np.deg2rad(elements["RA_OF_ASC_NODE"]),
    )
    return EarthSatellite.from_satrec(satrec, ts)


def get_satellite_positions(records: list, times_jd: list):
    """ICRS positions of satellites, by propagating their orbit records over *times_jd*.

    Parameters
    ----------
    records : sequence of dict, length n_sat
        Orbit records — TLE or OMM — as resolved by :mod:`tabascal.orbit`.
    times_jd : Array (n_time,)
        Times to calculate positions at, in Julian date.

    Returns
    -------
    Array (n_sat, n_time, 3)
        Satellite positions over time, in metres.
    """

    ts = timescale()
    sf_times = skyfield_time(times_jd)

    sat_pos = np.array(
        [
            _earth_satellite(record, ts).at(sf_times).position.km.T * 1e3
            for record in records
        ]
    )

    return sat_pos


def get_satellite_elevations(orbit_records: list, times_jd, ants_itrf) -> NDArray:
    """Topocentric elevation of each satellite, as seen from the array centre.

    Parameters
    ----------
    orbit_records : list of dict (n_sat,)
        Resolved orbit records, as returned by :func:`fetch_orbital_elements`.
        Built into propagators by :func:`_earth_satellite`, so OMM records work
        here exactly as TLE ones do -- an OMM has no lines to hand a line parser.
    times_jd : Array (n_time,)
        Times to calculate elevations at in Julian date.
    ants_itrf : Array (n_ant, 3)
        Antenna positions in ITRF, in metres. The mean is taken as the site.

    Returns
    -------
    Array (n_sat, n_time)
        Satellite elevation above the horizon, in degrees.
    """

    times_jd = np.asarray(times_jd)
    ts = timescale()
    sf_times = skyfield_time(times_jd)

    # geographic_position_of needs an ICRF position, so evaluate the (time-independent)
    # geodetic site position of the array centre at an arbitrary time
    centre_itrf = np.mean(np.asarray(ants_itrf), axis=0)
    site = wgs84.geographic_position_of(
        ITRSPosition(Distance(m=centre_itrf)).at(sf_times[0])
    )

    elevation = np.stack(
        [
            (_earth_satellite(record, ts) - site).at(sf_times).altaz()[0].degrees
            for record in orbit_records
        ]
    )

    return elevation


#: Default ``satellites.orbit_ric_std``: the 1-sigma width of the prior the SGP4
#: trajectory components put on a satellite's state at its record epoch, as
#: (radial, in-track, cross-track) position in metres followed by the same three
#: velocity components in metres per second. These are the values the components
#: carried hard-coded before the key existed, so the default changes no fit.
DEFAULT_ORBIT_RIC_STD = (7.3, 13.1, 5.4, 1.0, 1.0, 1.0)


def validate_orbit_ric_std(value):
    """``satellites.orbit_ric_std`` as six positive numbers, or a ``ValueError``.

    Returned in metres and metres per second, the units the key is written in.
    """
    if value is None:
        return np.asarray(DEFAULT_ORBIT_RIC_STD, dtype=float)

    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        arr = np.asarray([], dtype=float)

    if arr.size != 6 or not np.all(np.isfinite(arr)) or not np.all(arr > 0):
        raise ValueError(
            f"Config parameter (satellites:\n\torbit_ric_std: {value!r}) is not "
            "valid. Give six positive, finite numbers -- radial, in-track and "
            "cross-track position in metres, then the same three velocity "
            f"components in m/s -- or null for the default {list(DEFAULT_ORBIT_RIC_STD)}."
        )

    return arr


def orbit_ric_cov(config):
    """The RIC-frame prior covariance the SGP4 orbit components fit under.

    A diagonal 6x6 in the units sgp4jax works in, km and km/s, built from
    ``satellites.orbit_ric_std`` (metres, m/s) so that the config states the
    width in the units an orbit error is quoted in and the conversion happens
    once, here.

    The width is set at the *record epoch*, which is not where it is felt. SGP4
    propagates from there to the observation, so a velocity component grows into
    along-track position on the way: the position entries bound where the
    satellite was when its elements were fitted, and the velocity entries are
    what lets it be somewhere else by the time it crosses the field. A prior
    quoted for the epoch is therefore narrower at the epoch than the
    displacement it admits at the observation, and a run that needs kilometres
    of along-track freedom buys them through the velocity terms rather than the
    position ones. ``tabascal.components.trajectory.ric_prior_envelope`` reports
    what a given width is worth at given times, which is the number to set this
    key against.
    """
    std_m = validate_orbit_ric_std(
        (getattr(config, "args", None) or {}).get("satellites", {}).get("orbit_ric_std")
    )

    return jnp.diag(jnp.asarray(std_m / 1e3) ** 2)



def ric_prior_envelope(component, times_jd=None, n_draw=256, seed=0):
    """What a satellite's orbit prior is worth, in metres, where it is felt.

    ``satellites.orbit_ric_std`` is a width on the state at the *record epoch*.
    What a fit can actually do with it is the displacement that width reaches at
    the observation, after SGP4 has propagated it, and the two differ by however
    far the epoch is from the observation: a metre per second of along-track
    velocity is metres at the epoch and kilometres an hour later. This draws from
    the prior, propagates each draw, and reports the 1-sigma displacement about
    the nominal orbit in the RIC frame of the nominal state -- the number to read
    before concluding that a fit did not move because it did not want to.

    Parameters
    ----------
    component : NoDragOrbit or Orbit
        A component whose ``setup`` has run, so its prior and elements exist.
    times_jd : array-like, optional
        UTC Julian dates to evaluate at. Defaults to the component's own fine grid.
    n_draw : int
        Prior draws. The standard deviation converges as 1/sqrt(n_draw).
    seed : int
        Seed for the draws.

    Returns
    -------
    (Array, Array)
        ``(sigma_ric, speed)``: the 1-sigma (radial, in-track, cross-track)
        displacement in metres, shaped (n_rfi, n_time, 3), and the nominal
        orbital speed in m/s, shaped (n_rfi, n_time). Dividing the in-track
        column by the speed turns it into an along-track time offset in seconds,
        which is the form a trajectory timing error is usually quoted in.
    """
    times_jd = component.times_jd_fine if times_jd is None else jnp.asarray(times_jd)
    mu = np.asarray(component.mu_rfi_orbit)
    L = np.asarray(component.L_rfi_orbit)

    def propagate(elements):
        sats = component.sats_init(jnp.asarray(elements))
        xyz, vel = sgp4jax.gcrf_positions_multi_leo(sats, times_jd)
        return np.asarray(xyz) * 1e3, np.asarray(vel) * 1e3

    xyz0, vel0 = propagate(mu)
    speed = np.linalg.norm(vel0, axis=-1)

    # RIC basis of the nominal state, per satellite per time
    r_hat = xyz0 / np.linalg.norm(xyz0, axis=-1, keepdims=True)
    h = np.cross(xyz0, vel0)
    c_hat = h / np.linalg.norm(h, axis=-1, keepdims=True)
    i_hat = np.cross(c_hat, r_hat)

    rng = np.random.default_rng(seed)
    ric = np.empty((n_draw, *xyz0.shape))
    for d in range(n_draw):
        z = rng.standard_normal(mu.shape)
        xyz, _ = propagate(mu + np.einsum("sij,sj->si", L, z))
        dxyz = xyz - xyz0
        ric[d, ..., 0] = np.sum(dxyz * r_hat, axis=-1)
        ric[d, ..., 1] = np.sum(dxyz * i_hat, axis=-1)
        ric[d, ..., 2] = np.sum(dxyz * c_hat, axis=-1)

    return ric.std(axis=0), speed


#: Default ``satellites.orbit_deviation``, the prior on
#: :class:`RICDeviationGP`'s displacement. ``std`` is metres (a scalar, or one
#: per RIC axis) and ``corr_time`` seconds, with null meaning half the
#: observation -- the same thing null means for ``rfi.gp_cov.corr_time``.
DEFAULT_ORBIT_DEVIATION = {
    "std": 100.0,
    "corr_time": None,
    "gamma": 3.0,
    "cutoff": 1e-6,
    "time_pad_factor": 2.0,
}


def validate_orbit_deviation(value):
    """``satellites.orbit_deviation`` as a complete dict, or a ``ValueError``.

    ``std`` comes back as three metres, one per RIC axis, whether it was given
    as one number or three. ``corr_time`` may still be ``None``, which only the
    component can resolve -- half the observation is not known here.
    """
    given = dict(DEFAULT_ORBIT_DEVIATION) if value is None else None

    if given is None:
        if not isinstance(value, Mapping):
            raise ValueError(
                f"Config parameter (satellites:\n\torbit_deviation: {value!r}) is "
                "not valid. Give a mapping of "
                f"{sorted(DEFAULT_ORBIT_DEVIATION)}, or null for the defaults."
            )
        unknown = sorted(set(value) - set(DEFAULT_ORBIT_DEVIATION))
        if unknown:
            raise ValueError(
                f"Config parameter (satellites:\n\torbit_deviation) has unknown "
                f"key(s) {unknown}. Valid keys are {sorted(DEFAULT_ORBIT_DEVIATION)}."
            )
        given = {**DEFAULT_ORBIT_DEVIATION, **value}

    def _positive(key, allow_none=False):
        v = given[key]
        if allow_none and v is None:
            return None
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = float("nan")
        if not isfinite(v) or v <= 0:
            raise ValueError(
                f"Config parameter (satellites:\n\torbit_deviation:\n\t\t{key}: "
                f"{given[key]!r}) is not valid. Give a positive, finite number"
                f"{' or null' if allow_none else ''}."
            )
        return v

    std = np.asarray(given["std"], dtype=float).reshape(-1)
    if std.size == 1:
        std = np.repeat(std, 3)
    if std.size != 3 or not np.all(np.isfinite(std)) or not np.all(std > 0):
        raise ValueError(
            f"Config parameter (satellites:\n\torbit_deviation:\n\t\tstd: "
            f"{given['std']!r}) is not valid. Give one positive number in metres, "
            "or three -- radial, in-track, cross-track."
        )

    return {
        "std": std,
        "corr_time": _positive("corr_time", allow_none=True),
        "gamma": _positive("gamma"),
        "cutoff": _positive("cutoff"),
        "time_pad_factor": _positive("time_pad_factor"),
    }



class PhaseCalculationRFI(Component):

    requires_double = True
    required_inputs = {"rfi_xyz": ("n_rfi", "n_time_fine", 3)}
    output_shapes = {"rfi_phase": ("n_rfi", "n_ant", "n_freq_fine", "n_time_fine")}

    parameters = {}

    def setup(self, config):
        """All validation and error-prone operations here"""
        self.require_double(config)
        try:
            self.times_jd_fine = config.times_jd_fine
            self.ants_itrf = config.ants_itrf
            self.phase_centre = config.phase_centre
            self.freqs_fine = config.freqs_fine
            self.n_freq_fine = config.n_freq_fine
            self.n_rfi = config.n_rfi
            self.n_ant = config.n_ant
            self.n_time_fine = config.n_time_fine


            # Validate dimensions
            self._set_outputs()
            self._compute_ant_pos()
            self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def _compute_ant_pos(self):

        gsa = gast_deg(self.times_jd_fine)  # GAST in degrees (UTC convention)
        gh0 = (gsa - self.phase_centre["ra"]) % 360

        self.ants_xyz = vmap(vmap(sgp4jax.itrf_to_gcrf, (0, None, None), 0), (None, 0, 0), 1)(
            self.ants_itrf, 
            jnp.floor(self.times_jd_fine), 
            self.times_jd_fine - jnp.floor(self.times_jd_fine)
        )
        self.ants_uvw = jnp.transpose(
            itrf_to_uvw_numpy(self.ants_itrf, gh0, self.phase_centre["dec"]), axes=(1, 0, 2)
        )

    def _validate_dimensions(self):
        """Ensure all setup operations completed successfully"""

        ant_shape = (self.n_ant, self.n_time_fine, 3)

        assert_attr_shape(self, "ants_uvw", ant_shape)
        assert_attr_shape(self, "ants_xyz", ant_shape)
        assert_attr_shape(self, "freqs_fine", (self.n_freq_fine,))

    def build_set_params(self):

        def set_params(params):
            return params

        return set_params

    def build_constants(self):
        return {
            "ants_uvw": self.ants_uvw,
            "ants_xyz": self.ants_xyz,
            "freqs_fine": self.freqs_fine,
        }

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix

        def forward(params, state, constants):
            # Pure JAX operations only
            rfi_phase = get_rfi_phase(
                state["rfi_xyz"],
                constants[f"{prefix}/ants_uvw"],
                constants[f"{prefix}/ants_xyz"],
                constants[f"{prefix}/freqs_fine"],
            )
            state = {**state, "rfi_phase": rfi_phase}

            return state

        return forward
    
    def _set_outputs(self):

        # Fine-grid memory hog; under sharding each device only allocates its RFI shard.
        self.state_outputs = {
            "rfi_phase": sharded_rfi_zeros(
                (self.n_rfi, self.n_ant, self.n_freq_fine, self.n_time_fine), None
            ),
        }


class FixedOrbit(Component):

    required_inputs = {}  # No inputs needed
    output_shapes = {
        "rfi_xyz": ("n_rfi", "n_time_fine", 3),
        "rfi_phase": ("n_rfi", "n_ant", "n_freq_fine", "n_time_fine"),
    }

    # Add parameter specifications
    parameters = {}

    def setup(self, config):
        """All validation and error-prone operations here"""
        try:
            # Store only what's needed for forward computation
            self.orbit_records = config.orbit_records
            self.elements = config.elements
            self.epoch_jd = config.epoch_jd
            self.n_rfi = config.n_rfi
            self.n_ant = config.n_ant
            self.n_freq = config.n_freq
            self.n_time = config.n_time
            self.n_freq_fine = config.n_freq_fine
            self.n_time_fine = config.n_time_fine

            self.n_int_time = config.n_int_time
            self.n_int_freq = config.n_int_freq

            self.ants_itrf = config.ants_itrf
            self.phase_centre = config.phase_centre
            self.freqs = config.freqs
            self.times = config.times
            self.freqs_fine = config.freqs_fine
            self.times_fine = config.times_fine
            self.times_jd_fine = config.times_jd_fine

            # Do expensive setup operations once
            self._compute_rfi_phase()
            self._set_outputs()

            # Validate dimensions
            self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def build_set_params(self):

        def set_params(state):
            return state

        return set_params

    def build_constants(self):
        return {
            "rfi_xyz": self.rfi_xyz,
            "rfi_phase": self.rfi_phase,
        }

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix

        def forward(params, state, constants):
            rfi_xyz = constants[f"{prefix}/rfi_xyz"]
            rfi_phase = constants[f"{prefix}/rfi_phase"]
            return {**state, "rfi_xyz": rfi_xyz, "rfi_phase": rfi_phase}

        return forward

    def validate_and_test(self):
        """Call this before using in JIT context"""
        pass

    @measure_runtime
    def _compute_rfi_phase(self):

        self.rfi_xyz = np.asarray(
            get_satellite_positions(self.orbit_records, list(self.times_jd_fine))
        )

        self.ants_xyz = itrs_to_gcrs_sf(self.ants_itrf, self.times_jd_fine)

        # rfi_phase is one-shot setup producing a forward constant, so compute it in
        # numpy/skyfield (f64) in both precisions — faster than the jax path (no JIT
        # compile) and accurate. jnp.array casts to the active precision (f64/f32).
        gsa = gast_deg(self.times_jd_fine)  # GAST in degrees (UTC convention)
        gh0 = (gsa - self.phase_centre["ra"]) % 360

        self.ants_uvw = np.transpose(
            itrf_to_uvw_numpy(self.ants_itrf, gh0, self.phase_centre["dec"]), axes=(1, 0, 2)
        )
        # Fine-grid constant and the biggest array of this component: under sharding
        # it is created directly with the RFI-axis sharding so the full array only
        # ever exists in host numpy, never on a single device.
        rfi_phase_np = get_rfi_phase_numpy(
            self.rfi_xyz, self.ants_uvw, self.ants_xyz, self.freqs_fine
        )
        if sharding_enabled():
            dtype = jnp.zeros((), dtype=None).dtype  # match the active precision
            self.rfi_phase = make_global(rfi_phase_np.astype(dtype), rfi_sharding())
        else:
            self.rfi_phase = jnp.array(rfi_phase_np)

    def _set_outputs(self):

        self.state_outputs = {
            "rfi_xyz": self.rfi_xyz,
            "rfi_phase": self.rfi_phase,
        }

    def _validate_dimensions(self):
        """Ensure all setup operations completed successfully"""

        assert_attr_shape(self, "rfi_xyz", (self.n_rfi, self.n_time_fine, 3))
        assert_attr_shape(
            self,
            "rfi_phase",
            (self.n_rfi, self.n_ant, self.n_freq_fine, self.n_time_fine),
        )



class RICDeviationGP(Component):
    """A time-varying positional offset from a fixed orbit, fitted in the RIC frame.

    ``FixedOrbit`` propagates the orbital record and that is the satellite's
    position, exactly. ``NoDragOrbit`` and ``Orbit`` relax that by fitting the
    elements, but they still fit *an SGP4 orbit*: whatever they find, the
    satellite moves along a trajectory the model can express, and the six or
    seven numbers apply to the whole pass. This component relaxes it differently.
    It adds a smooth offset that varies over the pass -- radial, in-track and
    cross-track, each a Gaussian process in time -- to the positions the
    component before it produced, and fits that.

    The two are worth having separately because they fail differently. An
    element error is a statement about the orbit and is rigid over a pass by
    construction; if the residual wanted a satellite to be 200 m further along
    at the start of a pass and 200 m behind at the end, no element set can say
    so. Everything outside the model -- an unmodelled manoeuvre, a drag or
    solar-radiation-pressure excursion the record's epoch predates, the fact
    that the phase centre of the emission is not the centre of mass and does
    not have to sit still relative to it -- has that shape, and a fitted orbit
    absorbs it only by distorting the whole pass.

    Deliberately paired with ``FixedOrbit``: the nominal trajectory then stays
    the record's, and everything fitted here is a departure from it, which is
    the quantity worth reading. It can follow a fitted-orbit component instead,
    but then the two priors overlap on the rigid part of the deviation and
    neither term means much on its own.

    The RIC basis is built once at setup from the orbital records, which is the
    nominal trajectory, so the forward pass is a linear map from the parameters
    to a displacement: no re-propagation, no re-derivation of the frame, and a
    Jacobian the optimiser sees exactly. Emit ``rfi_xyz`` before
    ``PhaseCalculationRFI``, which recomputes the phase from whatever positions
    reach it::

        - trajectory:FixedOrbit
        - trajectory:RICDeviationGP
        - trajectory:PhaseCalculationRFI

    ``FixedOrbit`` writes an ``rfi_phase`` of its own and this component does
    not touch it, so that ordering is not optional: without the recomputation
    the deviation would move ``rfi_xyz`` and change no visibility.

    Read ``satellites.orbit_deviation``. The width is a real displacement in
    metres, unlike ``satellites.orbit_ric_std``, which is a width on the state
    at the record epoch -- here there is no propagation between where the prior
    is set and where it is felt, because the deviation is defined at the times
    observed.
    """

    # The phase is what a position moves, and the phase component it feeds is
    # double-only; a metre of an 800 km range is 1e-6 of it, which single
    # precision cannot carry through a difference of that size either.
    requires_double = True

    required_inputs = {"rfi_xyz": ("n_rfi", "n_time_fine", 3)}
    output_shapes = {"rfi_xyz": ("n_rfi", "n_time_fine", 3)}

    parameter_shapes = {
        "rfi_dev_r_base": ("n_rfi", 3, "n_k_time_dev"),
        "rfi_dev_i_base": ("n_rfi", 3, "n_k_time_dev"),
    }

    #: RIC axis order, for messages and for the per-axis width.
    AXES = ("radial", "in-track", "cross-track")

    def setup(self, config):
        """All validation and error-prone operations here"""
        self.require_double(config)
        try:
            self.n_rfi = config.n_rfi
            self.n_time = config.n_time
            self.n_time_fine = config.n_time_fine
            self.n_int_time = config.n_int_time
            self.int_time = config.int_time
            self.times_jd_fine = config.times_jd_fine
            self.orbit_records = config.orbit_records

            if self.n_time_fine < 2:
                raise ValueError(
                    "a deviation that varies over time needs more than one time "
                    f"sample; this observation has n_time_fine = {self.n_time_fine}"
                )

            self.dev_config = validate_orbit_deviation(
                (getattr(config, "args", None) or {})
                .get("satellites", {})
                .get("orbit_deviation")
            )
            if self.dev_config["corr_time"] is None:
                # Half the observation, which is what null means on the RFI
                # signal's time axis too. Resolved here rather than in the
                # validator because only the component knows how long the
                # observation is.
                self.dev_config["corr_time"] = 0.5 * self.n_time * self.int_time

            self._compute_ric_basis()
            self._compute_gp_params()
            self._compute_init_params()
            self._set_outputs()

            self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def build_set_params(self):
        n_rfi = self.n_rfi
        n_k = self.n_k_time_dev

        def set_params(params):
            params["rfi_dev_r_base"] = standard_normal("rfi_dev_r_base", (n_rfi, 3, n_k))
            params["rfi_dev_i_base"] = standard_normal("rfi_dev_i_base", (n_rfi, 3, n_k))

            return params

        return set_params

    def build_constants(self):
        return {
            "sigma_dev_k": self.sigma_dev_k,
            "ric_basis": self.ric_basis,
        }

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix
        pads = self.pads
        ss_idxs = self.ss_idxs

        def forward(params, state, constants):
            # Pure JAX operations only
            sigma_dev_k = constants[f"{prefix}/sigma_dev_k"]
            ric_basis = constants[f"{prefix}/ric_basis"]

            dev_k = sigma_dev_k * (
                params["rfi_dev_r_base"] + 1.0j * params["rfi_dev_i_base"]
            )
            # (n_rfi, 3) independent 1-D transforms, one per RIC axis per satellite
            dev = vmap(vmap(latent_to_signal, (0, None, None), 0), (0, None, None), 0)(
                dev_k, pads, ss_idxs
            )
            # Real part, not the modulus: the deviation is a signed displacement,
            # and the prior below is normalised for exactly this projection.
            dev = jnp.real(dev)

            # (n_rfi, 3 axes, n_time_fine) against (n_rfi, n_time_fine, 3 axes, 3 xyz)
            rfi_xyz = state["rfi_xyz"] + jnp.einsum("sct,stcx->stx", dev, ric_basis)

            return {**state, "rfi_xyz": rfi_xyz}

        return forward

    def validate_and_test(self):
        """Call this before using in JIT context"""
        pass

    @measure_runtime
    def _compute_ric_basis(self):
        """The RIC unit vectors of the nominal trajectory, per satellite per time.

        Built from the orbital records rather than from the incoming state, so
        the frame is the record's orbit whatever the component before this one
        did with the positions, and so it is a constant the forward pass can
        take as given.

        The velocity direction comes from a finite difference of the positions
        over the fine grid rather than from a second propagation: the frame only
        needs a direction, and the fine grid is sampled far more finely than the
        orbit turns.
        """
        xyz = np.asarray(
            get_satellite_positions(self.orbit_records, list(self.times_jd_fine))
        )                                                    # (n_rfi, n_time_fine, 3), m

        # Times in seconds for the gradient; the spacing is uniform, so any
        # positive scale gives the same unit vector, but keep it physical.
        t_s = (np.asarray(self.times_jd_fine) - np.asarray(self.times_jd_fine)[0]) * 86400.0
        vel = np.gradient(xyz, t_s, axis=1)

        def unit(v):
            n = np.linalg.norm(v, axis=-1, keepdims=True)
            # A padded dummy satellite can sit at the origin; leave its frame as
            # zeros rather than NaN. It carries no signal, so it moves nothing.
            return np.divide(v, n, out=np.zeros_like(v), where=n > 0)

        r_hat = unit(xyz)
        c_hat = unit(np.cross(xyz, vel))
        i_hat = np.cross(c_hat, r_hat)

        # (n_rfi, n_time_fine, 3 axes, 3 xyz), rows in AXES order
        self.ric_basis = jnp.asarray(np.stack([r_hat, i_hat, c_hat], axis=2))

    def _compute_gp_params(self):
        """The power spectrum of the deviation, normalised to the configured width.

        One 1-D grid over time, supersampled to the fine grid the positions live
        on, exactly as the RFI signal's time axis is. The spectrum is normalised
        so that ``sum(sigma_dev_k**2)`` is the configured variance, which -- for
        the real part of a transform of complex coefficients with independent
        unit-normal real and imaginary parts -- is the variance of the
        displacement itself, at every time. So ``std`` is the rms of the metres
        this component adds, and it stays so when the correlation time, the
        roll-off or the cutoff change: those move where the power sits, not how
        much of it there is.
        """
        std = self.dev_config["std"]                         # (3,) metres
        k0 = knee_from_corr_scale([self.dev_config["corr_time"]])

        pk, ks, self.pads, self.ss_idxs = latent_to_signal_init(
            [self.n_time],
            [self.int_time],
            [self.dev_config["time_pad_factor"]],
            [self.n_int_time],
            1.0,                       # renormalised below, so any positive scalar
            k0,
            [self.dev_config["gamma"]],
            self.dev_config["cutoff"],
        )
        self.pk = pk
        self.n_k_time_dev = int(pk.shape[0])

        unit_pk = pk / jnp.sum(pk)                           # (n_k,), sums to 1
        var = jnp.asarray(std, dtype=unit_pk.dtype) ** 2     # (3,)
        # (1 broadcast over satellites, 3 axes, n_k)
        self.sigma_dev_k = jnp.sqrt(var[None, :, None] * unit_pk[None, None, :])

        print("\nOrbit deviation specs")
        print(f"(std_R, std_I, std_C): ({std[0]:.4g}, {std[1]:.4g}, {std[2]:.4g}) m")
        print(f"(corr_time, n_k_time): ({self.dev_config['corr_time']:.4g} s, {self.n_k_time_dev})")
        print(f"(gamma, cutoff)      : ({self.dev_config['gamma']}, {self.dev_config['cutoff']:.1e})")

    def _compute_init_params(self):
        """Start on the nominal orbit.

        Zero base parameters are zero deviation, so a run begins at exactly the
        trajectory the component before this one produced and every metre it
        ends up with was asked for by the data.
        """
        zeros = jnp.zeros((self.n_rfi, 3, self.n_k_time_dev))
        self.init_params = {"rfi_dev": zeros}
        self.init_params_base = {
            "rfi_dev_r_base": zeros,
            "rfi_dev_i_base": zeros,
        }

    def _set_outputs(self):
        self.state_outputs = {
            "rfi_xyz": jnp.zeros((self.n_rfi, self.n_time_fine, 3)),
        }

    def _validate_dimensions(self):
        """Ensure all setup operations completed successfully"""

        assert_attr_shape(self, "ric_basis", (self.n_rfi, self.n_time_fine, 3, 3))
        assert_attr_shape(self, "sigma_dev_k", (1, 3, self.n_k_time_dev))

class NoDragOrbit(Component):

    requires_double = True
    required_inputs = {}  # No inputs needed
    output_shapes = {
        "rfi_xyz": ("n_rfi", "n_time_fine", 3),
        "elements": ("n_rfi", 6),  # Also output elements for downstream use
    }

    # Add parameter specifications
    parameters = {"rfi_orbit_base": ("n_rfi", 6)}

    def setup(self, config):
        """All validation and error-prone operations here"""
        self.require_double(config)
        try:
            # Store only what's needed for forward computation
            self.times_jd = config.times_jd
            self.times_jd_fine = config.times_jd_fine
            self.n_time_fine = config.n_time_fine

            self.n_rfi = config.n_rfi
            self.ric_cov = orbit_ric_cov(config)

            # Reuse the resolution the preflight check already made and enforced
            # coverage on: re-resolving here could reach a different satellite set
            # from the one the run was checked against, and would repeat the
            # provider work. Falls back to resolving when there is no preflight
            # (standalone component use and tests).
            self.elements, epoch_jd, self.norad_ids, tles = fetch_standard_orbital_elements(
                config.times_jd,
                config.norad_ids,
                extra_orbit_dir=getattr(config, "extra_orbit_dir", None),
                extra_orbit_max_age_days=getattr(config, "extra_orbit_max_age_days", None),
                resolution=getattr(config, "tle_resolution", None),
            )
            self.bstar = self.elements[:, 0]
            self.elements = self.elements[:, 1:] # Remove the bstar drag element
            self.sat_epoch = epoch_jd - 2433281.5
            self.epoch_jd_whole = jnp.floor(epoch_jd)
            self.epoch_jd_frac = epoch_jd - self.epoch_jd_whole

            # Do expensive setup operations once
            self._compute_prior_params()
            self._compute_init_params()
            self._set_outputs()

            # Validate dimensions
            self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    
    def sats_init(self, elements):

        def sat_init(sat_epoch, bstar, ecco, argpo, inclo, mo, no_kozai, nodeo, jdsatepoch, jdsatepochF):
            sat_rec = sgp4init(
                gravity, sat_epoch, 
                bstar,
                0.0, 0.0,  # ndot, nddot (fixed)
                ecco, argpo, inclo, mo, no_kozai, nodeo,
                jdsatepoch, jdsatepochF,
            )

            return sat_rec

        inclo, nodeo, ecco, argpo, mo, no_kozai = elements.T

        sats = vmap(sat_init)(
            self.sat_epoch, 
            self.bstar,
            ecco, 
            argpo, 
            inclo, 
            mo, 
            no_kozai, 
            nodeo, 
            self.epoch_jd_whole, 
            self.epoch_jd_frac
        )

        return sats

    def build_set_params(self):
        n_rfi = self.n_rfi

        def set_params(state):

            state["rfi_orbit_base"] = standard_normal("rfi_orbit_base", (n_rfi, 6))

            return state

        return set_params

    def build_constants(self):
        return {
            "times_jd_fine": self.times_jd_fine,
            "L_rfi_orbit": self.L_rfi_orbit,
            "mu_rfi_orbit": self.mu_rfi_orbit,
        }

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix
        forward_transform = self.forward_transform
        sats_init = self.sats_init

        def forward(params, state, constants):
            # Pure JAX operations only
            L_orbit = constants[f"{prefix}/L_rfi_orbit"]
            mu_orbit = constants[f"{prefix}/mu_rfi_orbit"]

            elements = forward_transform(params["rfi_orbit_base"], L_orbit, mu_orbit)

            sats = sats_init(elements)
            rfi_xyz, _ = sgp4jax.gcrf_positions_multi_leo(sats, constants[f"{prefix}/times_jd_fine"])
            rfi_xyz = rfi_xyz * 1e3

            state = {**state, "elements": elements, "rfi_xyz": rfi_xyz}

            return state

        return forward

    def validate_and_test(self):
        """Call this before using in JIT context"""
        pass

    def _compute_prior_params(self):

        sats = self.sats_init(self.elements)    
        kepler_cov = vmap(sgp4jax.cov_ric_to_elements, (None, 0, 0, 0))(self.ric_cov, sats, self.epoch_jd_whole, self.epoch_jd_frac)

        self.L_rfi_orbit = vmap(jnp.linalg.cholesky)(kepler_cov)
        self.mu_rfi_orbit = self.elements

    def _set_outputs(self):

        self.state_outputs = {
            "elements": jnp.zeros((self.n_rfi, 6)),
            "rfi_xyz": jnp.zeros((self.n_rfi, self.n_time_fine, 3)),
        }

    def forward_transform(self, base_params, L, mu):

        params = vmap(affine_transform_full)(base_params, L, mu)

        return params

    def inv_transform(self, params, L, mu):

        base_params = vmap(jnp.linalg.solve)(L, params - mu)

        return base_params

    def _compute_init_params(self):

        self.init_rfi_orbit = self.mu_rfi_orbit

        self.init_rfi_orbit_base = self.inv_transform(
            self.init_rfi_orbit, self.L_rfi_orbit, self.mu_rfi_orbit
        )

        self.init_params = {"rfi_orbit": self.init_rfi_orbit}
        self.init_params_base = {"rfi_orbit_base": self.init_rfi_orbit_base}

    def _validate_dimensions(self):
        """Ensure all setup operations completed successfully"""

        orbit_shape = (self.n_rfi, 6)

        assert_attr_shape(self, "mu_rfi_orbit", orbit_shape)
        assert_attr_shape(self, "L_rfi_orbit", (self.n_rfi, 6, 6))
        assert_attr_shape(self, "init_rfi_orbit", orbit_shape)
        assert_attr_shape(self, "init_rfi_orbit_base", orbit_shape)


class Orbit(Component):

    requires_double = True
    required_inputs = {}  # No inputs needed
    output_shapes = {
        "rfi_xyz": ("n_rfi", "n_time_fine", 3),
        "elements": ("n_rfi", 7),  # Also output elements for downstream use
    }

    # Add parameter specifications
    parameters = {"rfi_orbit_base": ("n_rfi", 7)}

    def setup(self, config):
        """All validation and error-prone operations here"""
        self.require_double(config)
        try:
            # Store only what's needed for forward computation
            self.times_jd = config.times_jd
            self.times_jd_fine = config.times_jd_fine
            self.n_time_fine = config.n_time_fine

            self.n_rfi = config.n_rfi
            self.ric_cov = orbit_ric_cov(config)

            # Reuse the resolution the preflight check already made and enforced
            # coverage on: re-resolving here could reach a different satellite set
            # from the one the run was checked against, and would repeat the
            # provider work. Falls back to resolving when there is no preflight
            # (standalone component use and tests).
            self.elements, epoch_jd, self.norad_ids, tles = fetch_standard_orbital_elements(
                config.times_jd,
                config.norad_ids,
                extra_orbit_dir=getattr(config, "extra_orbit_dir", None),
                extra_orbit_max_age_days=getattr(config, "extra_orbit_max_age_days", None),
                resolution=getattr(config, "tle_resolution", None),
            )
            self.sat_epoch = epoch_jd - 2433281.5
            self.epoch_jd_whole = jnp.floor(epoch_jd)
            self.epoch_jd_frac = epoch_jd - self.epoch_jd_whole


            # Do expensive setup operations once
            self._compute_prior_params()
            self._compute_init_params()
            self._set_outputs()

            # Validate dimensions
            self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")
    
    def sats_init(self, elements):

        def sat_init(sat_epoch, bstar, ecco, argpo, inclo, mo, no_kozai, nodeo, jdsatepoch, jdsatepochF):
            sat_rec = sgp4init(
                gravity, sat_epoch, 
                bstar,
                0.0, 0.0,  # ndot, nddot (fixed)
                ecco, argpo, inclo, mo, no_kozai, nodeo,
                jdsatepoch, jdsatepochF,
            )

            return sat_rec

        bstar, inclo, nodeo, ecco, argpo, mo, no_kozai = elements.T

        sats = vmap(sat_init)(
            self.sat_epoch, 
            bstar,
            ecco, 
            argpo, 
            inclo, 
            mo, 
            no_kozai, 
            nodeo, 
            self.epoch_jd_whole, 
            self.epoch_jd_frac
        )

        return sats

    def build_set_params(self):
        n_rfi = self.n_rfi

        def set_params(state):

            state["rfi_orbit_base"] = standard_normal("rfi_orbit_base", (n_rfi, 7))

            return state

        return set_params

    def build_constants(self):
        return {
            "times_jd_fine": self.times_jd_fine,
            "L_rfi_orbit": self.L_rfi_orbit,
            "mu_rfi_orbit": self.mu_rfi_orbit,
        }

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix
        forward_transform = self.forward_transform
        sats_init = self.sats_init

        def forward(params, state, constants):
            # Pure JAX operations only
            L_orbit = constants[f"{prefix}/L_rfi_orbit"]
            mu_orbit = constants[f"{prefix}/mu_rfi_orbit"]

            elements = forward_transform(params["rfi_orbit_base"], L_orbit, mu_orbit)

            sats = sats_init(elements)
            rfi_xyz, _ = sgp4jax.gcrf_positions_multi_leo(sats, constants[f"{prefix}/times_jd_fine"])
            rfi_xyz = rfi_xyz * 1e3

            state = {**state, "elements": elements, "rfi_xyz": rfi_xyz}

            return state

        return forward

    def validate_and_test(self):
        """Call this before using in JIT context"""
        pass

    def _compute_prior_params(self):

        sats = self.sats_init(self.elements)
        # kepler_cov shape: (n_rfi, 6, 6)
        kepler_cov = vmap(sgp4jax.cov_ric_to_elements, (None, 0, 0, 0))(self.ric_cov, sats, self.epoch_jd_whole, self.epoch_jd_frac)

        bstar_cov = 1e-6

        # Prepend a bstar row/column to each (6, 6) covariance → (n_rfi, 7, 7)
        def _prepend_bstar(cov_6x6):
            return jnp.block([
                [jnp.array([[bstar_cov]]), jnp.zeros((1, 6))],
                [jnp.zeros((6, 1)),        cov_6x6           ],
            ])

        kepler_cov = vmap(_prepend_bstar)(kepler_cov)  # (n_rfi, 7, 7)

        self.L_rfi_orbit = vmap(jnp.linalg.cholesky)(kepler_cov)
        self.mu_rfi_orbit = self.elements

    def _set_outputs(self):

        self.state_outputs = {
            "elements": jnp.zeros((self.n_rfi, 7)),
            "rfi_xyz": jnp.zeros((self.n_rfi, self.n_time_fine, 3)),
        }

    def forward_transform(self, base_params, L, mu):

        params = vmap(affine_transform_full)(base_params, L, mu)

        return params

    def inv_transform(self, params, L, mu):

        base_params = vmap(jnp.linalg.solve)(L, params - mu)

        return base_params

    def _compute_init_params(self):

        self.init_rfi_orbit = self.mu_rfi_orbit

        self.init_rfi_orbit_base = self.inv_transform(
            self.init_rfi_orbit, self.L_rfi_orbit, self.mu_rfi_orbit
        )

        self.init_params = {"rfi_orbit": self.init_rfi_orbit}
        self.init_params_base = {"rfi_orbit_base": self.init_rfi_orbit_base}

    def _validate_dimensions(self):
        """Ensure all setup operations completed successfully"""

        orbit_shape = (self.n_rfi, 7)

        assert_attr_shape(self, "mu_rfi_orbit", orbit_shape)
        assert_attr_shape(self, "L_rfi_orbit", (self.n_rfi, 7, 7))
        assert_attr_shape(self, "init_rfi_orbit", orbit_shape)
        assert_attr_shape(self, "init_rfi_orbit_base", orbit_shape)


def itrs_to_gcrs_sf(pos_itrs: NDArray, times_jd: NDArray) -> NDArray:

    # skyfield must always receive numpy (it divides by AU as a python int, which
    # overflows int32 if a jax f32 array is passed under jax_enable_x64=False).
    pos_itrs = np.asarray(pos_itrs)
    times_jd = np.asarray(times_jd)

    t_sf = skyfield_time(times_jd)

    pos_gcrs = np.stack(
        [ITRSPosition(Distance(m=pos)).at(t_sf).position.m.T for pos in pos_itrs]
    )

    return pos_gcrs


def _pad_rfi_sources(tles_df):
    """Pad the fetched TLE set to a multiple of the device count under sharding.

    The RFI axis is split evenly across devices, so when the satellite count does not
    divide, the last satellite's row is duplicated up to :func:`padded_rfi_count`.
    Padded sources are made *dark* by the RFI signal components (zero prior mean and
    zero init on their amplitude latents): the visibility contribution is quadratic in
    the amplitude, so both their signal and their gradient are exactly zero and the
    solve is unchanged. Both orbital-element fetch paths (TabConfig and the SGP4
    components' own re-fetch) go through here, so every consumer sees the same padded
    count. No-op single-device or when the count already divides.
    """
    n_pad = padded_rfi_count(len(tles_df)) - len(tles_df)
    if n_pad == 0 or len(tles_df) == 0:
        return tles_df
    import pandas as pd
    return pd.concat([tles_df, *([tles_df.iloc[[-1]]] * n_pad)], ignore_index=True)


def _orbit_records(tles_df) -> list[dict]:
    """The resolved frame as a list of raw records, one per source, in row order.

    This is what propagation and replay both consume. It used to be an
    ``(n_sat, 2)`` array of TLE line pairs, which an OMM record cannot fill —
    it has no lines, only elements. Passing the records themselves lets
    :func:`_earth_satellite` and
    :func:`tabascal.orbit.save_orbits_for_reuse` each ask the record what it is.
    """
    return tles_df.to_dict(orient="records")


#: Element columns the SGP4/Kepler propagators consume, in the order they expect.
_ELEMENT_COLUMNS = [
    "SEMIMAJOR_AXIS",
    "ECCENTRICITY",  # ecco
    "INCLINATION",  # inclo
    "RA_OF_ASC_NODE",  # nodeo
    "ARG_OF_PERICENTER",  # argpo
    "MEAN_ANOMALY",  # mo
]


def _no_satellites():
    """Empty element arrays for a model that configures no satellites.

    A satellite-free model is a legitimate configuration — ``norad_ids: []`` is
    the shipped default, and :func:`tabascal.orbit_config.model_requires_tles` is
    what rejects the case where the *model* needs TLEs but none were given. This
    path must therefore produce an empty RFI model rather than be reported as a
    resolution failure.
    """
    return (
        jnp.zeros((0, len(_ELEMENT_COLUMNS))),
        jnp.zeros((0,)),
        [],
        [],
    )


def _requested_nothing(norad_ids) -> bool:
    return norad_ids is None or not len(np.atleast_1d(np.asarray(norad_ids)))


def _require_tles(tles_df, norad_ids) -> None:
    """Validate the resolved TLEs against the requested NORAD IDs.

    Resolution is all-or-nothing, so by the time a frame reaches here every
    requested ID should be present; this is the defence in depth that stops an
    incomplete set reaching the model by another route. An empty frame would
    otherwise surface as an opaque pandas ``KeyError`` on the element columns, and
    a partial one would silently shrink the RFI model — degrading subtraction with
    no visible signal.

    Callers screen out the "nothing was requested" case first, so an empty frame
    reaching here always means a genuine failure to resolve.
    """
    requested = sorted({int(n) for n in np.atleast_1d(np.asarray(norad_ids))})
    if not len(tles_df):
        raise TLEError(
            f"No TLEs could be resolved for NORAD IDs {requested}. "
            "Check that the IDs are valid, and that either the extra TLE "
            "directory covers them or the SatChecker service is reachable."
        )
    resolved = {int(n) for n in tles_df["NORAD_CAT_ID"]}
    missing = sorted(set(requested) - resolved)
    if missing:
        raise TLEError(
            f"TLEs could not be resolved for {len(missing)} of {len(requested)} "
            f"requested satellites: NORAD IDs {missing}. TABASCAL does not "
            f"subtract an incomplete satellite model: supply their TLEs via "
            f"--extra-orbit-dir, relax satellites.remote_max_age_days "
            f"deliberately, or remove these IDs from satellites.norad_ids."
        )


def fetch_orbital_elements(
    times_jd=None,
    norad_ids=None,
    extra_orbit_dir=None,
    extra_orbit_max_age_days=None,
    resolution=None,
):
    """Orbital elements for the RFI model.

    *resolution* is the :class:`~tabascal.orbit.TLEResolution` the preflight check
    already produced; passing it is the normal path and guarantees the model is
    built from exactly the records whose coverage and ages were checked. Without
    it the satellites are resolved here instead, for callers that have no
    preflight (the components' own re-fetch, and tests).
    """
    tles_df, norad_ids = _resolved_frame(
        resolution,
        times_jd,
        norad_ids,
        extra_orbit_dir,
        extra_orbit_max_age_days,
    )
    if _requested_nothing(norad_ids):
        return (*_no_satellites(), 0)
    _require_tles(tles_df, norad_ids)
    # Real (unpadded) source count is the number of rows the fetch actually returned,
    # captured before padding. Inferring it from the padded id list (e.g. counting
    # distinct ids) is wrong when the real sources already contain a repeated NORAD id.
    n_rfi_real = len(tles_df)
    tles_df = _pad_rfi_sources(tles_df)

    elements = jnp.atleast_2d(tles_df[_ELEMENT_COLUMNS].values)
    epoch_jd = jnp.atleast_1d(tles_df["EPOCH_JD"].values)  # type: ignore
    norad_ids = list(tles_df["NORAD_CAT_ID"].values)
    orbit_records = _orbit_records(tles_df)

    return elements, epoch_jd, norad_ids, orbit_records, n_rfi_real

def _resolved_frame(
    resolution,
    times_jd,
    norad_ids,
    extra_orbit_dir,
    extra_orbit_max_age_days,
):
    """The element frame plus the ID list it must cover, from either source."""
    if resolution is not None:
        return resolution.frame(), list(resolution.requested)
    tles_df = get_tles_by_id(
        norad_ids,
        times_jd,
        extra_orbit_dir=extra_orbit_dir,
        extra_orbit_max_age_days=extra_orbit_max_age_days,
    )
    return tles_df, norad_ids


def fetch_standard_orbital_elements(
    times_jd=None,
    norad_ids=None,
    extra_orbit_dir=None,
    extra_orbit_max_age_days=None,
    resolution=None,
):
    """Orbital elements for the SGP4 propagators.

    Unlike :func:`fetch_orbital_elements` this deliberately has no empty-request
    escape: only the SGP4/Kepler trajectory components call it, and those are
    exactly the components ``model_requires_tles`` refuses to configure without
    satellites. Reaching here with nothing requested is a real failure.
    """
    tles_df, norad_ids = _resolved_frame(
        resolution,
        times_jd,
        norad_ids,
        extra_orbit_dir,
        extra_orbit_max_age_days,
    )
    _require_tles(tles_df, norad_ids)
    tles_df = _pad_rfi_sources(tles_df)

    # tles_df carries the OMM-style element columns derived locally by
    # satchecker_client.records.record_elements (degrees, rev/day, km), plus
    # NORAD_CAT_ID, EPOCH_JD, and whichever raw columns the record's kind has.

    # SGP4 MINIMUM REQUIREMENTS:
    # To propagate an orbit using SGP4, you need:
    # - EPOCH (reference time)
    # - MEAN_MOTION (revolutions/day)
    # - ECCENTRICITY (0-1)
    # - INCLINATION (degrees)
    # - RA_OF_ASC_NODE (degrees)
    # - ARG_OF_PERICENTER (degrees)
    # - MEAN_ANOMALY (degrees)
    # - BSTAR (drag term, 1/ER)
    # - NORAD_CAT_ID (for identification)

    elements = jnp.atleast_2d(
        tles_df[
            [
                "BSTAR", # bstar
                "ECCENTRICITY", # ecco
                "ARG_OF_PERICENTER", # argpo
                "INCLINATION", # inclo
                "MEAN_ANOMALY", # mo
                "MEAN_MOTION", # no_kozai
                "RA_OF_ASC_NODE", # nodeo
            ]
        ].values
    )
    rev_per_day_to_rad_per_min = 1440.0 / (2.0 * jnp.pi)
    elements = elements.at[:, 2:5].set(jnp.deg2rad(elements[:, 2:5]))
    elements = elements.at[:, -1].set(jnp.deg2rad(elements[:, -1]))
    elements = elements.at[:, -2].set(elements[:, -2] / rev_per_day_to_rad_per_min)
    # bstar, ecco, argpo, inclo, mo, no_kozai, nodeo
    # (inclo, nodeo, ecco, argpo, mo, no_kozai)
    elements = jnp.stack([
        elements[:,0], 
        elements[:,3], elements[:,6], 
        elements[:,1], elements[:,2], 
        elements[:,4], elements[:,5],
        ], axis=1
    )    
    epoch_jd = jnp.atleast_1d(tles_df["EPOCH_JD"].values)  # type: ignore
    norad_ids = list(tles_df["NORAD_CAT_ID"].values)
    orbit_records = _orbit_records(tles_df)

    return elements, epoch_jd, norad_ids, orbit_records
