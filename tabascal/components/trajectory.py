from tabascal.orbit import TLEError, get_tles_by_id
from tabascal.satchecker.records import KIND_TLE, record_elements, record_kind
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
from tabascal.fft_gp import domain_ss
from tabascal.components import Component, assert_attr_shape
from tabascal.timing import measure_runtime
from tabascal.time import gast_deg

import sgp4jax
from sgp4jax import WGS72 as gravity
from sgp4jax._sgp4init import sgp4init

import jax.numpy as jnp
from jax import vmap, Array
import numpy as np
from numpy.typing import NDArray

from sgp4.api import WGS72, Satrec

from skyfield.api import Distance, load, wgs84
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

    ts = load.timescale()
    times_jd_whole = np.floor(times_jd)
    times_jd_frac = np.array(times_jd) - times_jd_whole
    sf_times = ts._utc_jd(times_jd_whole, times_jd_frac)

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
    ts = load.timescale()
    sf_times = ts._utc_jd(np.floor(times_jd), times_jd - np.floor(times_jd))

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
        # self.ants_xyz = itrs_to_gcrs_sf(self.ants_itrf, self.times_jd_fine)
        # self.ants_xyz = jnp.transpose(itrf_to_xyz(self.ants_itrf, gsa), axes=(1, 0, 2))
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
    outputs_shapes = {
        "rfi_xyz": ("n_rfi", "n_time_fine", 3),
        "rfi_phase": ("n_rfi", "n_ant", "n_freq", "n_time_fine"),
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
            self.n_int_freq = config.args["rfi"]["freq_int_samples"]

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
            # self.elements = config.elements
            # self.epoch_jd = config.epoch_jd
            self.ric_cov = jnp.diag(jnp.array([0.73, 1.31, 0.54, 0.1, 0.1, 0.1])**2)/1e4
            # self.ric_std = config.args["satellites"]["ric_std"]

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

        # ecco, argpo, inclo, mo, no_kozai, nodeo = elements.T
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
            # self.elements = config.elements
            # self.epoch_jd = config.epoch_jd
            self.ric_cov = jnp.diag(jnp.array([0.73, 1.31, 0.54, 0.1, 0.1, 0.1])**2)/1e4
            # self.ric_std = config.args["satellites"]["ric_std"]

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

    ts = load.timescale()
    t_sf = ts._utc_jd(np.floor(times_jd), times_jd - np.floor(times_jd))

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
    # tabascal.satchecker.records.record_elements (degrees, rev/day, km), plus
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
