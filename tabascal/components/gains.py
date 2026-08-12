from tabascal.components import Component, assert_attr_shape
from tabascal.interferometry import apply_gains
from tabascal.dist import standard_normal
from tabascal.config import TabConfig
from tabascal.gp import cholesky, resampling_kernel, get_times
from tabascal.transform import affine_transform_full
from tabascal.validation import resolve_gains_defaults

import jax.numpy as jnp
from jax import vmap, Array

from typing import Dict


class BaseGPGains(Component):

    required_inputs = {
        "vis_rfi": ("n_bl", "n_freq", "n_time"), 
        "vis_ast": ("n_bl", "n_freq", "n_time")
    }
    output_shapes = {
        "gains": ("n_ant", "n_freq", "n_time"), 
        "vis_obs": ("n_bl", "n_freq", "n_time")
    }
    parameter_shapes = {}

    # Every gains key has a default -- either a constant or one derived from the
    # data by resolve_gains_defaults -- so none of them are required up front.
    required_config = ()

    def setup(self, tab_config: TabConfig):

        # Fill in the defaults that can only be derived once the MS is read.
        gains_config = resolve_gains_defaults(
            tab_config.args["gains"], tab_config.freqs, tab_config.chan_width, tab_config.times, tab_config.int_time)

        # Random seed used for random sampling such as initial parameters drawn from the prior
        self.r_seed = gains_config["r_seed"]
        # Basic shape parameters
        self.n_ant = tab_config.n_ant
        self.n_bl = tab_config.n_bl
        self.n_freq = tab_config.n_freq
        self.n_freq_fine = tab_config.n_freq_fine
        self.n_int_freq = tab_config.n_int_freq
        self.n_time = tab_config.n_time
        self.n_time_fine = tab_config.n_time_fine
        self.n_int_time = tab_config.n_int_time

        self.a1 = tab_config.a1
        self.a2 = tab_config.a2

        # Domain arrays needed to calculate Gaussian process parameters
        self.freqs = tab_config.freqs
        self.chan_width = tab_config.chan_width
        self.times = tab_config.times
        self.int_time = tab_config.int_time

        self.gp_amp_mean = gains_config["amp_mean"]
        self.gp_amp_std = gains_config["amp_std"]
        self.amp_corr_freq = gains_config["amp_corr_freq"]
        self.amp_corr_time = gains_config["amp_corr_time"]
        self.gp_phase_mean = gains_config["phase_mean"]
        self.gp_phase_std = gains_config["phase_std"]
        self.phase_corr_freq = gains_config["phase_corr_freq"]
        self.phase_corr_time = gains_config["phase_corr_time"]

    def build_set_params(self):

        def set_params(params: Dict) -> Dict:

            return params

        return set_params

    def _set_outputs(self):

        self.state_outputs = {
            "gains": jnp.ones((self.n_ant, self.n_freq, self.n_time), dtype=complex),
            "vis_obs": jnp.zeros((self.n_bl, self.n_freq, self.n_time), dtype=complex),
        }

    def _compute_gp_params(self):
        pass

    def _compute_prior_params(self):
        pass

    def _compute_init_params(self):
        pass

class UnitaryGains(BaseGPGains):

    parameters = {}

    def setup(self, tab_config: TabConfig):
        """All validation and error-prone operations here"""
        try:
            super().setup(tab_config)

            # Validate dimensions
            self._set_outputs()
            self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def _validate_dimensions(self):
        """Ensure all setup operations completed successfully"""
        pass

    def build_forward(self):
        gains = self.state_outputs["gains"]

        def forward(params: Dict, state: Dict, constants: Dict) -> Dict:
            vis_obs = state["vis_rfi"] + state["vis_ast"]
            state = {**state, "vis_obs": vis_obs, "gains": gains}
            return state

        return forward


class GPGains(BaseGPGains):

    parameters = {
        "gains_amp_induce_base": ("n_ant", "n_g_times"),
        "gains_phase_induce_base": ("n_ant-1", "n_g_times"),
    }

    def setup(self, tab_config: TabConfig):
        """All validation and error-prone operations here"""
        try:
            super().setup(tab_config)

            self._set_outputs()
            self._compute_gp_params()
            self._compute_prior_params()
            self._compute_init_params()
            # Validate dimensions
            self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")


    def build_set_params(self):

        def set_params(params):

            params["gains_amp_induce_base"] = standard_normal("gains_amp_induce_base", (self.n_ant, self.n_freq, self.n_g_times))
            params["gains_phase_induce_base"] = standard_normal("gains_phase_induce_base", (self.n_ant-1, self.n_freq, self.n_g_times))

            return params

        return set_params

    def build_constants(self):
        return {
            "resample_amp": self.resample_amp,
            "L_gains_amp": self.L_gains_amp,
            "mu_gains_amp": self.mu_gains_amp,
            "resample_phase": self.resample_phase,
            "L_gains_phase": self.L_gains_phase,
            "mu_gains_phase": self.mu_gains_phase,
        }

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix
        forward_transform = self.forward_transform
        gp_amp_mean = self.gp_amp_mean
        gp_phase_mean = self.gp_phase_mean
        a1 = self.a1
        a2 = self.a2
        n_freq = self.n_freq
        n_time = self.n_time

        def forward(params, state, constants):

            interp = lambda R, x, mu: jnp.einsum("ij,afj->afi", R, x - mu) + mu

            gains_amp_induce_base = params["gains_amp_induce_base"]
            gains_phase_induce_base = params["gains_phase_induce_base"]

            L_gains_amp = constants[f"{prefix}/L_gains_amp"]
            mu_gains_amp = constants[f"{prefix}/mu_gains_amp"]
            L_gains_phase = constants[f"{prefix}/L_gains_phase"]
            mu_gains_phase = constants[f"{prefix}/mu_gains_phase"]
            resample_amp = constants[f"{prefix}/resample_amp"]
            resample_phase = constants[f"{prefix}/resample_phase"]

            gains_amp_induce = forward_transform(gains_amp_induce_base, L_gains_amp, mu_gains_amp)
            gains_phase_induce = forward_transform(gains_phase_induce_base, L_gains_phase, mu_gains_phase)

            gains_amp = interp(resample_amp, gains_amp_induce, gp_amp_mean)
            gains_phase = jnp.concatenate([interp(resample_phase, gains_phase_induce, gp_phase_mean), jnp.zeros((1, n_freq, n_time))], axis=0)

            gains = gains_amp * jnp.exp(1.0j * gains_phase)

            vis_obs = apply_gains(gains, state["vis_rfi"] + state["vis_ast"], a1, a2)
            # vis_obs = apply_gains(gains, state["vis_rfi"], a1, a2) + state["vis_ast"]

            state = {**state, "vis_obs": vis_obs, "gains": gains}
            return state

        return forward

    def _compute_gp_params(self):

        self.g_times = get_times(self.times, min(self.amp_corr_time, self.phase_corr_time))
        self.n_g_times = len(self.g_times)

        self.resample_amp = resampling_kernel(
            self.g_times,
            self.times,
            self.gp_amp_std**2,
            self.amp_corr_time,
            1e-8,
        )
        self.resample_phase = resampling_kernel(
            self.g_times,
            self.times,
            self.gp_phase_std**2,
            self.phase_corr_time,
            1e-8,
        )

    def _compute_prior_params(self):

        self.L_gains_amp = cholesky(self.g_times, self.gp_amp_std**2, self.amp_corr_time, 1e-8)
        self.mu_gains_amp = self.gp_amp_mean * jnp.ones(
            (self.n_ant, self.n_freq, self.n_g_times)
        )

        self.L_gains_phase = cholesky(self.g_times, self.gp_phase_std**2, self.phase_corr_time, 1e-8)
        self.mu_gains_phase = self.gp_phase_mean * jnp.ones(
            (self.n_ant-1, self.n_freq, self.n_g_times)
        )

    def forward_transform(self, base_params: Array, L: Array, mu: Array) -> Array:

        affine_same_scale = lambda _base_params, _mu: affine_transform_full(_base_params, L, _mu)
        params = vmap(vmap(affine_same_scale))(base_params, mu)

        return params

    def inv_transform(self, params: Array, L: Array, mu: Array) -> Array:

        inv_affine_same_scale = lambda centred_params: jnp.linalg.solve(L, centred_params)
        base_params = vmap(vmap(inv_affine_same_scale))(params - mu)

        return base_params

    def _compute_init_params(self):

        self.init_gains_amp_induce = self.mu_gains_amp
        self.init_gains_amp_induce_base = self.inv_transform(
            self.init_gains_amp_induce, self.L_gains_amp, self.mu_gains_amp
        )

        self.init_gains_phase_induce = self.mu_gains_phase
        self.init_gains_phase_induce_base = self.inv_transform(
            self.init_gains_phase_induce, self.L_gains_phase, self.mu_gains_phase
        )

        self.init_params = {
            "gains_amp_induce": self.init_gains_amp_induce,
            "gains_phase_induce": self.init_gains_phase_induce,
        }
        self.init_params_base = {
            "gains_amp_induce_base": self.init_gains_amp_induce_base,
            "gains_phase_induce_base": self.init_gains_phase_induce_base,
        }

    def _validate_dimensions(self):
        """Ensure all setup operations completed successfully"""

        amp_shape = (self.n_ant, self.n_freq, self.n_g_times)
        phase_shape = (self.n_ant-1, self.n_freq, self.n_g_times)

        assert_attr_shape(self, "mu_gains_amp", amp_shape)
        assert_attr_shape(self, "L_gains_amp", (self.n_g_times, self.n_g_times))
        assert_attr_shape(self, "init_gains_amp_induce", amp_shape)
        assert_attr_shape(self, "init_gains_amp_induce_base", amp_shape)

        assert_attr_shape(self, "mu_gains_phase", phase_shape)
        assert_attr_shape(self, "L_gains_amp", (self.n_g_times, self.n_g_times))
        assert_attr_shape(self, "init_gains_phase_induce", phase_shape)
        assert_attr_shape(self, "init_gains_phase_induce_base", phase_shape)
