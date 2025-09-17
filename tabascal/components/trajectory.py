from tabsim.config import yaml_load  # type: ignore
from tabsim.tle import get_tles_by_id  # type: ignore

from tabsim.jax.coordinates import (  # type: ignore
    itrf_to_uvw,
    itrf_to_xyz,
    kepler_orbit_many,
    kepler_orbit_fisher,
    gmsa_from_jd,
)
from tabascal.dist import standard_normal
from tabascal.transform import affine_transform_full
from tabascal.interferometry import get_rfi_phase
from tabascal.fft_gp import domain_ss

import jax.numpy as jnp
from jax import vmap
import numpy as np

from tabascal.components import Component, assert_attr_shape


class PhaseCalculationRFI(Component):

    required_inputs = {"rfi_xyz": ("n_rfi", "n_time_fine", 3)}
    output_shapes = {"rfi_phase": ("n_rfi", "n_ant", "n_freq", "n_time_fine")}

    parameters = {}

    def setup(self, config):
        """All validation and error-prone operations here"""
        try:
            self.times_jd_fine = config.times_jd_fine
            self.ants_itrf = config.ants_itrf
            self.phase_centre = config.phase_centre
            self.freqs = config.freqs
            self.n_freq = config.n_freq
            self.n_rfi = config.n_rfi
            self.n_ant = config.n_ant
            self.n_time_fine = config.n_time_fine

            # Validate dimensions
            self._compute_ant_pos()
            self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"PhaseCalculation setup failed: {e}")

    def _compute_ant_pos(self):

        gsa = gmsa_from_jd(self.times_jd_fine) % 360
        gh0 = (gsa - self.phase_centre["ra"]) % 360

        self.ants_uvw = jnp.transpose(
            itrf_to_uvw(self.ants_itrf, gh0, self.phase_centre["dec"]), axes=(1, 0, 2)
        )
        self.ants_xyz = jnp.transpose(itrf_to_xyz(self.ants_itrf, gsa), axes=(1, 0, 2))

    def _validate_dimensions(self):
        """Ensure all setup operations completed successfully"""

        ant_shape = (self.n_ant, self.n_time_fine, 3)

        assert_attr_shape(self, "ants_uvw", ant_shape)
        assert_attr_shape(self, "ants_xyz", ant_shape)
        assert_attr_shape(self, "freqs", (self.n_freq,))

    def build_set_params(self):

        def set_params(params):
            return params

        return set_params

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        # Pre-compute everything possible
        ants_uvw = self.ants_uvw
        ants_xyz = self.ants_xyz
        freqs = self.freqs

        def forward(params, state):
            # Pure JAX operations only
            rfi_phase = get_rfi_phase(state["rfi_xyz"], ants_uvw, ants_xyz, freqs)
            # [
            #     :, :, 0, :
            # ]
            state = {**state, "rfi_phase": rfi_phase}

            return state

        return forward


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
            self.n_time_fine = config.n_time_fine

            self.n_int_time = config.n_int_time
            self.n_int_freq = config.args["rfi"]["freq_int_samples"]

            self.ants_itrf = config.ants_itrf
            self.phase_centre = config.phase_centre
            self.freqs = config.freqs
            self.times = config.times

            xs = [self.freqs, self.times]
            ss_factors = [self.n_int_freq, self.n_int_time]
            pad_factors = [
                config.args["rfi"]["freq_pad_factor"],
                config.args["rfi"]["time_pad_factor"],
            ]
            self.freqs_fine, self.times_fine = domain_ss(xs, ss_factors, pad_factors)
            self.n_freq_fine = len(self.freqs_fine)

            self.times_jd_fine = config.times_jd_fine

            # Do expensive setup operations once
            self._compute_rfi_phase()
            self._set_outputs()

            # Validate dimensions
            self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"FixedOrbit setup failed: {e}")

    def build_set_params(self):

        def set_params(state):
            return state

        return set_params

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        # Pre-compute everything possible
        rfi_xyz = self.rfi_xyz
        rfi_phase = self.rfi_phase

        def forward(params, state):

            state = {**state, "rfi_xyz": rfi_xyz, "rfi_phase": rfi_phase}

            return state

        return forward

    def validate_and_test(self):
        """Call this before using in JIT context"""
        pass

    def _compute_rfi_phase(self):

        from astropy.time import Time
        from tabsim.tle import get_satellite_positions  # type: ignore

        self.rfi_xyz = jnp.asarray(
            get_satellite_positions(self.tles, list(self.times_jd_fine))
        )
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

        self.ants_xyz = jnp.transpose(itrf_to_xyz(self.ants_itrf, gsa), axes=(1, 0, 2))
        self.ants_uvw = jnp.transpose(
            itrf_to_uvw(self.ants_itrf, gh0, self.phase_centre["dec"]), axes=(1, 0, 2)
        )

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


class KeplerOrbit(Component):

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
            self.elements = config.elements
            self.epoch_jd = config.epoch_jd
            self.ric_std = config.args["satellites"]["ric_std"]

            # Do expensive setup operations once
            # self._fetch_orbital_elements()
            self._compute_prior_params()
            self._compute_init_params()
            self._set_outputs()

            # Validate dimensions
            self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"KeplerOrbit setup failed: {e}")

    def build_set_params(self):
        n_rfi = self.n_rfi

        def set_params(state):

            state["rfi_orbit_base"] = standard_normal("rfi_orbit_base", (n_rfi, 6))

            return state

        return set_params

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        # Pre-compute everything possible
        times_jd_fine = self.times_jd_fine
        epoch_jd = self.epoch_jd
        L_orbit = self.L_rfi_orbit
        mu_orbit = self.mu_rfi_orbit
        forward_transform = self.forward_transform

        def forward(params, state):
            # Pure JAX operations only

            elements = forward_transform(params["rfi_orbit_base"], L_orbit, mu_orbit)
            rfi_xyz = kepler_orbit_many(times_jd_fine, epoch_jd, elements)

            state = {**state, "elements": elements, "rfi_xyz": rfi_xyz}

            return state

        return forward

    def validate_and_test(self):
        """Call this before using in JIT context"""
        pass

    def _compute_prior_params(self):

        RIC_std = self.ric_std * jnp.array([73, 131, 54])

        F_orbit = vmap(kepler_orbit_fisher, in_axes=(None, 0, 0, None))(
            self.times_jd, self.epoch_jd, self.elements, RIC_std  # type: ignore
        )
        kepler_cov = vmap(jnp.linalg.inv)(F_orbit)

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


def fetch_orbital_elements(spacetrack_path, obs_epoch_jd, norad_ids):

    st_login = yaml_load(spacetrack_path)

    tles_df = get_tles_by_id(
        st_login["username"],
        st_login["password"],
        norad_ids,
        obs_epoch_jd,
    )

    elements = jnp.atleast_2d(
        tles_df[
            [
                "SEMIMAJOR_AXIS",
                "ECCENTRICITY",
                "INCLINATION",
                "RA_OF_ASC_NODE",
                "ARG_OF_PERICENTER",
                "MEAN_ANOMALY",
            ]
        ].values
    )
    epoch_jd = jnp.atleast_1d(tles_df["EPOCH_JD"].values)  # type: ignore
    norad_ids = list(tles_df["NORAD_CAT_ID"].values)
    tles = np.atleast_2d(tles_df[["TLE_LINE1", "TLE_LINE2"]].values)

    return elements, epoch_jd, norad_ids, tles
