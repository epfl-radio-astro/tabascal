from tabsim.config import yaml_load
from tabsim.tle import get_tles_by_id

from tabsim.jax.coordinates import (
    itrf_to_uvw,
    itrf_to_xyz,
    kepler_orbit_many,
    kepler_orbit_fisher,
    gmsa_from_jd,
)
from tabascal.dist import standard_normal
from tabascal.transform import affine_transform_full
from tabascal.interferometry import get_rfi_phase

import jax.numpy as jnp
from jax import vmap, jit
import numpy as np

from tabascal.components import Component


class PhaseCalculationRFI(Component):

    required_inputs = {"rfi_xyz": ("n_rfi", "n_time_fine", 3)}
    outputs = {"rfi_phase": ("n_rfi", "n_ant", "n_time_fine")}

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

        assert hasattr(self, "ants_uvw")
        assert self.ants_uvw.shape == (self.n_ant, self.n_time_fine, 3)

        assert hasattr(self, "ants_xyz")
        assert self.ants_uvw.shape == (self.n_ant, self.n_time_fine, 3)

        assert hasattr(self, "freqs")
        assert self.ants_uvw.shape == (self.n_freq,)

    def build_set_params(self):

        def set_params(state):
            return state

        return set_params

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        # Pre-compute everything possible
        ants_uvw = self.ants_uvw
        ants_xyz = self.ants_xyz
        freqs = self.freqs

        def forward(state):
            # Pure JAX operations only
            state["rfi_phase"] = get_rfi_phase(
                state["rfi_xyz"], ants_uvw, ants_xyz, freqs
            )
            return state

        return forward


class FixedOrbit(Component):

    required_inputs = {}  # No inputs needed
    outputs = {
        "rfi_xyz": ("n_rfi", "n_time_fine", 3),
        "elements": ("n_rfi", 6),
        "rfi_phase": ("n_rfi", "n_ant", "n_time_fine"),
    }

    # Add parameter specifications
    parameters = {}

    def setup(self, config):
        """All validation and error-prone operations here"""
        try:
            # Store only what's needed for forward computation
            self.norad_ids = config.norad_ids
            self.spacetrack_path = config.spacetrack_path
            self.n_rfi = config.n_rfi
            self.times_jd_fine = config.times_jd_fine
            self.ants_itrf = config.ants_itrf
            self.phase_centre = config.phase_centre
            self.freqs = config.freqs
            self.n_ant = config.n_ant
            self.n_time_fine = config.n_time_fine

            # Do expensive setup operations once
            self._fetch_orbital_elements()
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
        elements = self.elements
        rfi_xyz = self.rfi_xyz
        rfi_phase = self.rfi_phase

        def forward(state):
            # Pure JAX operations only
            state["elements"] = elements
            state["rfi_xyz"] = rfi_xyz
            state["rfi_phase"] = rfi_phase
            return state

        return forward

    def validate_and_test(self):
        """Call this before using in JIT context"""
        test_state = {"rfi_orbit_base": jnp.zeros((self.n_rfi, 6))}
        forward_fn = self.build_forward()

        # Test outside JIT first
        result = forward_fn(test_state)

        # Then test JIT compilation
        jitted_forward = jit(forward_fn)
        jit_result = jitted_forward(test_state)

        # Verify they match
        assert jnp.allclose(result["rfi_xyz"], jit_result["rfi_xyz"])

    def _fetch_orbital_elements(self):

        obs_epoch_jd = float(self.times_jd_fine.mean())

        self.elements, self.epoch_jd, self.norad_ids, tles = fetch_orbital_elements(
            self.spacetrack_path, obs_epoch_jd, self.norad_ids
        )

        self.n_rfi = len(self.norad_ids)

    def _compute_rfi_phase(self):

        self.rfi_xyz = kepler_orbit_many(
            self.times_jd_fine, self.epoch_jd, self.elements
        )

        gsa = gmsa_from_jd(self.times_jd_fine) % 360
        gh0 = (gsa - self.phase_centre["ra"]) % 360

        self.ants_uvw = jnp.transpose(
            itrf_to_uvw(self.ants_itrf, gh0, self.phase_centre["dec"]), axes=(1, 0, 2)
        )
        self.ants_xyz = jnp.transpose(itrf_to_xyz(self.ants_itrf, gsa), axes=(1, 0, 2))

        self.rfi_phase = get_rfi_phase(
            self.rfi_xyz, self.ants_uvw, self.ants_xyz, self.freqs
        )[:, :, 0, :]

    def _set_outputs(self):

        self.state_outputs = {
            "rfi_xyz": jnp.zeros((self.n_rfi, self.n_time_fine, 3)),
            "elements": jnp.zeros((self.n_rfi, 6)),
            "rfi_phase": jnp.zeros((self.n_rfi, self.n_ant, self.n_time_fine)),
        }

    def _validate_dimensions(self):
        """Ensure all setup operations completed successfully"""

        phase_shape = (self.n_rfi, self.n_ant, self.n_time_fine)

        assert hasattr(self, "rfi_phase")
        assert (
            self.rfi_phase.shape == phase_shape
        ), f"Invalid shape for rfi_phase. Expected {phase_shape} but got {self.rfi_phase.shape}"


class KeplerOrbit(Component):

    required_inputs = {}  # No inputs needed
    outputs = {
        "rfi_xyz": ("n_rfi", "n_time_fine", 3),
        "elements": ("n_rfi", 6),  # Also output elements for downstream use
    }

    # Add parameter specifications
    parameters = {"rfi_orbit_base": ("n_rfi", 6)}

    def setup(self, config):
        """All validation and error-prone operations here"""
        try:
            # Store only what's needed for forward computation
            self.times_jd_fine = config.times_jd_fine
            self.n_rfi = config.n_rfi
            self.n_time_fine = config.n_time_fine
            self.norad_ids = config.norad_ids
            self.spacetrack_path = config.spacetrack_path
            self.ric_std = config.ric_std

            # Do expensive setup operations once
            self._fetch_orbital_elements()
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

        def forward(state):
            # Pure JAX operations only

            state["elements"] = forward_transform(
                state["rfi_orbit_base"], L_orbit, mu_orbit
            )
            state["rfi_xyz"] = kepler_orbit_many(
                times_jd_fine, epoch_jd, state["elements"]
            )
            return state

        return forward

    def validate_and_test(self):
        """Call this before using in JIT context"""
        test_state = {"rfi_orbit_base": jnp.zeros((self.n_rfi, 6))}
        forward_fn = self.build_forward()

        # Test outside JIT first
        result = forward_fn(test_state)

        # Then test JIT compilation
        jitted_forward = jit(forward_fn)
        jit_result = jitted_forward(test_state)

        # Verify they match
        assert jnp.allclose(result["rfi_xyz"], jit_result["rfi_xyz"])

    def _fetch_orbital_elements(self):

        obs_epoch_jd = float(self.times_jd_fine.mean())

        self.elements, self.epoch_jd, self.norad_ids, tles = fetch_orbital_elements(
            self.spacetrack_path, obs_epoch_jd, self.norad_ids
        )

        self.n_rfi = len(self.norad_ids)

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
            "rfi_xyz": jnp.zeros((self.n_rfi, self.n_time_fine, 3)),
            "elements": jnp.zeros((self.n_rfi, 6)),
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

        assert hasattr(self, "mu_rfi_orbit")
        assert self.mu_rfi_orbit.shape == orbit_shape

        assert hasattr(self, "L_rfi_orbit")
        assert self.L_rfi_orbit.shape == (self.n_rfi, 6, 6)

        assert hasattr(self, "init_rfi_orbit")
        assert self.init_rfi_orbit.shape == orbit_shape

        assert hasattr(self, "init_rfi_orbit_base")
        assert self.init_rfi_orbit_base.shape == orbit_shape


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
