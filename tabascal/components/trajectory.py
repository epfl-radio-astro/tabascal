from tabsim.tle import get_tles_by_id, load_spacetrack_credentials  # type: ignore

from tabsim.jax.coordinates import (  # type: ignore
    secs_to_days,
    itrf_to_uvw,
    itrf_to_xyz,
    kepler_orbit_many,
    kepler_orbit_fisher,
    gmsa_from_jd,
    mjd_to_jd,
)
from tabascal.dist import standard_normal
from tabascal.transform import affine_transform_full
from tabascal.interferometry import get_rfi_phase
from tabascal.fft_gp import domain_ss
from tabascal.components import Component, assert_attr_shape

import sgp4jax
from sgp4jax import WGS72 as gravity
from sgp4jax._sgp4init import sgp4init

import jax.numpy as jnp
from jax import vmap, Array
import numpy as np

from skyfield.api import Distance, load
from skyfield.toposlib import ITRSPosition

from astropy.time import Time

from skyfield.api import EarthSatellite

def get_satellite_positions(tles: list, times_jd: list):
    """Calculate the ICRS positions of satellites by propagating their TLEs over the given times.

    Parameters
    ----------
    tles : Array (n_sat, 2)
        TLEs usind to propagate positions.
    times : Array (n_time,)
        Times to calculate positions at in Julian date.

    Returns
    -------
    Array (n_sat, n_time, 3)
        Satellite positions over time
    """

    ts = load.timescale()
    # sf_times = ts.ut1_jd(times_jd)
    sf_times = ts._utc_jd(np.floor(times_jd), np.array(times_jd) - np.floor(times_jd))

    sat_pos = np.array(
        [
            EarthSatellite(tle_line1, tle_line2, ts=ts).at(sf_times).position.km.T * 1e3
            for tle_line1, tle_line2 in tles
        ]
    )

    return sat_pos


class PhaseCalculationRFI(Component):

    required_inputs = {"rfi_xyz": ("n_rfi", "n_time_fine", 3)}
    output_shapes = {"rfi_phase": ("n_rfi", "n_ant", "n_freq_fine", "n_time_fine")}

    parameters = {}

    def setup(self, config):
        """All validation and error-prone operations here"""
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

        gsa = (
            Time(self.times_jd_fine, format="jd")
            .sidereal_time("mean", "greenwich")
            .hour
            * 15
        )  # type: ignore

        # gsa = gmsa_from_jd(self.times_jd_fine) % 360
        gh0 = (gsa - self.phase_centre["ra"]) % 360    

        self.ants_xyz = vmap(vmap(sgp4jax.itrf_to_gcrf, (0, None, None), 0), (None, 0, 0), 1)(
            self.ants_itrf, 
            jnp.floor(self.times_jd_fine), 
            self.times_jd_fine - jnp.floor(self.times_jd_fine)
        )
        # self.ants_xyz = itrs_to_gcrs_sf(self.ants_itrf, self.times_jd_fine)
        # self.ants_xyz = jnp.transpose(itrf_to_xyz(self.ants_itrf, gsa), axes=(1, 0, 2))
        self.ants_uvw = jnp.transpose(
            itrf_to_uvw(self.ants_itrf, gh0, self.phase_centre["dec"]), axes=(1, 0, 2)
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

        self.state_outputs = {
            "rfi_phase": jnp.zeros((self.n_rfi, self.n_ant, self.n_freq_fine, self.n_time_fine)),
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
            self.tles = config.tles
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

    def build_forward(self):
        """Return pure, JIT-compatible function"""

        def forward(params, state, constants):
            # rfi_xyz and rfi_phase already in state via state_outputs
            return state

        return forward

    def validate_and_test(self):
        """Call this before using in JIT context"""
        pass

    def _compute_rfi_phase(self):

        # from tabsim.tle import get_satellite_positions  # type: ignore

        # self.rfi_xyz = jnp.asarray(
        #     get_satellite_positions(self.tles, list(self.times_jd_fine))
        # )

        sats = sgp4jax.tles_to_satrec(self.tles)
        r_gcrf, _ = sgp4jax.gcrf_positions_multi(sats, self.times_jd_fine)

        self.rfi_xyz = r_gcrf * 1e3

        sf_rfi_xyz = jnp.asarray(
            get_satellite_positions(self.tles, list(self.times_jd_fine))
        )

        print(f"RFI Error: {jnp.sqrt(jnp.mean(jnp.sum((sf_rfi_xyz-self.rfi_xyz)**2, axis=-1))):.2e}")

        gsa = (
            Time(self.times_jd_fine, format="jd")
            .sidereal_time("mean", "greenwich")
            .hour
            * 15
        )  # type: ignore

        # self.rfi_xyz = kepler_orbit_many(
        #     self.times_jd_fine, self.epoch_jd, self.elements
        # )
        # gsa = gmsa_from_jd(self.times_jd_fine) % 360
        gh0 = (gsa - self.phase_centre["ra"]) % 360

        self.ants_xyz = vmap(vmap(sgp4jax.itrf_to_gcrf, (0, None, None), 0), (None, 0, 0), 1)(
            self.ants_itrf, 
            jnp.floor(self.times_jd_fine), 
            self.times_jd_fine - jnp.floor(self.times_jd_fine)
        )
        # self.ants_xyz = itrs_to_gcrs_sf(self.ants_itrf, self.times_jd_fine)
        sf_ants_xyz = itrs_to_gcrs_sf(self.ants_itrf, self.times_jd_fine)
        # self.ants_xyz = jnp.transpose(itrf_to_xyz(self.ants_itrf, gsa), axes=(1, 0, 2))
        self.ants_uvw = jnp.transpose(
            itrf_to_uvw(self.ants_itrf, gh0, self.phase_centre["dec"]), axes=(1, 0, 2)
        )

        print(f"Ants Error: {jnp.sqrt(jnp.mean(jnp.sum((sf_ants_xyz-self.ants_xyz)**2, axis=-1))):.2e}")

        self.rfi_phase = get_rfi_phase(
            self.rfi_xyz, self.ants_uvw, self.ants_xyz, self.freqs_fine
        )

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


class SGP4LEONoDragOrbit(Component):

    required_inputs = {}  # No inputs needed
    output_shapes = {
        "rfi_xyz": ("n_rfi", "n_time_fine", 3),
        "elements": ("n_rfi", 6),  # Also output elements for downstream use
    }

    # Add parameter specifications
    parameters = {"rfi_orbit_base": ("n_rfi", 6)}

    def setup(self, config):
        """All validation and error-prone operations here"""
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

            self.elements, epoch_jd, self.norad_ids, tles = fetch_standard_orbital_elements(jnp.mean(config.times_jd), config.norad_ids)
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
            "epoch_jd": self.epoch_jd,
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


class SGP4LEOOrbit(Component):

    required_inputs = {}  # No inputs needed
    output_shapes = {
        "rfi_xyz": ("n_rfi", "n_time_fine", 3),
        "elements": ("n_rfi", 7),  # Also output elements for downstream use
    }

    # Add parameter specifications
    parameters = {"rfi_orbit_base": ("n_rfi", 7)}

    def setup(self, config):
        """All validation and error-prone operations here"""
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

            self.elements, epoch_jd, self.norad_ids, tles = fetch_standard_orbital_elements(jnp.mean(config.times_jd), config.norad_ids)
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

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        # Pre-compute everything possible
        times_jd_fine = self.times_jd_fine
        L_orbit = self.L_rfi_orbit
        mu_orbit = self.mu_rfi_orbit
        forward_transform = self.forward_transform
        

        def forward(params, state):
            # Pure JAX operations only

            elements = forward_transform(params["rfi_orbit_base"], L_orbit, mu_orbit)
            
            sats = self.sats_init(elements)
            rfi_xyz, _ = sgp4jax.gcrf_positions_multi_leo(sats, times_jd_fine)
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

        bstar_cov = 1e-6

        kepler_cov = jnp.block([[jnp.array([[bstar_cov]]), jnp.zeros((1, 6))],
                                [jnp.zeros((6, 1)), kepler_cov]]
        )

        print(kepler_cov.shape)
        print(kepler_cov)

        self.L_rfi_orbit = vmap(jnp.linalg.cholesky)(kepler_cov)
        self.mu_rfi_orbit = self.elements

        print(self.L_rfi_orbit)
        print(self.mu_rfi_orbit)

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


def itrs_to_gcrs_sf(pos_itrs: Array, times_jd: Array) -> Array:

    ts = load.timescale()
    t_sf = ts.ut1_jd(np.array(times_jd))

    pos_gcrs = jnp.stack(
        [ITRSPosition(Distance(m=pos)).at(t_sf).position.m.T for pos in pos_itrs]
    )

    return pos_gcrs


def fetch_orbital_elements(obs_epoch_jd, norad_ids):

    st_user, st_pass = load_spacetrack_credentials()

    tles_df = get_tles_by_id(
        st_user,
        st_pass,
        norad_ids,
        obs_epoch_jd,
    )

    elements = jnp.atleast_2d(
        tles_df[
            [
                "SEMIMAJOR_AXIS", #
                "ECCENTRICITY", # ecco
                "INCLINATION", # inclo
                "RA_OF_ASC_NODE", # nodeo
                "ARG_OF_PERICENTER", # argpo
                "MEAN_ANOMALY", # mo
            ]
        ].values
    )
    epoch_jd = jnp.atleast_1d(tles_df["EPOCH_JD"].values)  # type: ignore
    norad_ids = list(tles_df["NORAD_CAT_ID"].values)
    tles = np.atleast_2d(tles_df[["TLE_LINE1", "TLE_LINE2"]].values)

    return elements, epoch_jd, norad_ids, tles

def fetch_standard_orbital_elements(obs_epoch_jd, norad_ids):

    st_user, st_pass = load_spacetrack_credentials()

    tles_df = get_tles_by_id(
        st_user,
        st_pass,
        norad_ids,
        obs_epoch_jd,
    )

    # tles_df columns
    # 'CCSDS_OMM_VERS', 'COMMENT', 'CREATION_DATE', 'ORIGINATOR',
    # 'OBJECT_NAME', 'OBJECT_ID', 'CENTER_NAME', 'REF_FRAME', 'TIME_SYSTEM',
    # 'MEAN_ELEMENT_THEORY', 'EPOCH', 'MEAN_MOTION', 'ECCENTRICITY',
    # 'INCLINATION', 'RA_OF_ASC_NODE', 'ARG_OF_PERICENTER', 'MEAN_ANOMALY',
    # 'EPHEMERIS_TYPE', 'CLASSIFICATION_TYPE', 'NORAD_CAT_ID',
    # 'ELEMENT_SET_NO', 'REV_AT_EPOCH', 'BSTAR', 'MEAN_MOTION_DOT',
    # 'MEAN_MOTION_DDOT', 'SEMIMAJOR_AXIS', 'PERIOD', 'APOAPSIS', 'PERIAPSIS',
    # 'OBJECT_TYPE', 'RCS_SIZE', 'COUNTRY_CODE', 'LAUNCH_DATE', 'SITE',
    # 'DECAY_DATE', 'FILE', 'GP_ID', 'TLE_LINE0', 'TLE_LINE1', 'TLE_LINE2',
    # 'Fetch_Timestamp', 'EPOCH_JD', 'time_diff', 'time_diff_abs'

# CCSDS_OMM_VERS                     =3.0                      
# COMMENT                            =GENERATED VIA SPACE-TRACK.ORG API
# CREATION_DATE                      =2026-03-13T03:38:44      
# ORIGINATOR                         =18 SPCS                  
# OBJECT_NAME                        =ISS (ZARYA)              
# OBJECT_ID                          =1998-067A                
# CENTER_NAME                        =EARTH                    
# REF_FRAME                          =TEME                     
# TIME_SYSTEM                        =UTC                      
# MEAN_ELEMENT_THEORY                =SGP4                     
# EPOCH                              =2026-03-12T20:51:23.157792
# MEAN_MOTION                        =15.48614629              
# ECCENTRICITY                       =0.00079238               
# INCLINATION                        =51.6324                  
# RA_OF_ASC_NODE                     =56.6367                  
# ARG_OF_PERICENTER                  =186.1410                 
# MEAN_ANOMALY                       =173.9482                 
# EPHEMERIS_TYPE                     =0                        
# CLASSIFICATION_TYPE                =U                        
# NORAD_CAT_ID                       =25544                    
# ELEMENT_SET_NO                     =999                      
# REV_AT_EPOCH                       =55682                    
# BSTAR                              =0.00021655360000         
# MEAN_MOTION_DOT                    =0.00011348               
# MEAN_MOTION_DDOT                   =0.0000000000000          
# USER_DEFINED_SEMIMAJOR_AXIS        =6798.915                 
# USER_DEFINED_PERIOD                =92.986                   
# USER_DEFINED_APOAPSIS              =426.167                  
# USER_DEFINED_PERIAPSIS             =415.393                  
# USER_DEFINED_OBJECT_TYPE           =PAYLOAD                  
# USER_DEFINED_RCS_SIZE              =LARGE                    
# USER_DEFINED_COUNTRY_CODE          =CIS                      
# USER_DEFINED_LAUNCH_DATE           =1998-11-20               
# USER_DEFINED_SITE                  =TTMTR                    
# USER_DEFINED_DECAY_DATE            =                         
# USER_DEFINED_FILE                  =5086888                  
# USER_DEFINED_GP_ID                 =315816402   
    
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
    tles = np.atleast_2d(tles_df[["TLE_LINE1", "TLE_LINE2"]].values)

    return elements, epoch_jd, norad_ids, tles


# def get_leo_pos_vel(elements):

#     sat = sgp4init(
#         gravity, sat_epoch, 
#         bstar,
#         0.0, 0.0,  # ndot, nddot (fixed)
#         ecco, argpo, inclo, mo, no_kozai, nodeo,
#         jdsatepoch, jdsatepochF,
#     )

# def satrec_from_df(df):

