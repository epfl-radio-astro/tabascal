from jax import vmap, random, jit
import jax.numpy as jnp

from tabascal.components import Component, assert_attr_shape
from tabascal.dist import standard_normal
from tabascal.tab_tools import (
    get_ast_fringe_rate,
    pow_spec,
    get_observation_data_type,
)
from tabascal.fft_gp import latent_to_signal_init, latent_to_signal, signal_to_latent_init, signal_to_latent, pow_spec_nd
from tabascal.timing import measure_runtime

import xarray as xr


class FourierTimeAst(Component):

    required_inputs = {}  # No inputs needed
    output_shape = {
        "vis_ast": ("n_bl", "n_freq", "n_time"),
    }

    # Add parameter specifications
    parameters = {
        "ast_k_r_base": ("n_bl", "n_freq", "n_k_ast"),
        "ast_k_i_base": ("n_bl", "n_freq", "n_k_ast"),
    }

    def setup(self, config):
        """All validation and error-prone operations here"""
        try:
            # Store only what's needed for forward computation
            self.n_time = config.n_time
            self.n_bl = config.n_bl
            self.n_freq = config.n_freq
            self.int_time = config.int_time
            self.dish_d = config.dish_d
            self.uvw = config.uvw
            self.freqs = config.freqs
            self.p0 = config.args["ast"]["pow_spec"]["p0"]
            self.gamma = config.args["ast"]["pow_spec"]["gamma"]
            self.fov_deg = config.args["ast"]["pow_spec"]["fov_deg"]
            self.ast_pad_factor = config.args["ast"]["time_pad_factor"]

            # Do expensive setup operations once
            self._compute_gp_params()
            self._compute_prior_params()

            if config.args["plots"]["truth"] or config.args["ast"]["init"] == "truth":
                self._compute_true_params(
                    config.args["data"]["zarr_path"], config.args["data"]["data_col"]
                )

            self._compute_init_params(config.args["ast"]["init"])
            self._set_outputs()

            # Validate dimensions
            self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"ComplexRFI setup failed: {e}")

    def build_set_params(self):
        n_bl = self.n_bl
        n_freq = self.n_freq
        n_ast_k = self.n_ast_k

        def set_params(params):

            params["ast_k_r_base"] = standard_normal(
                "ast_k_r_base", (n_bl, n_freq, n_ast_k)
            )
            params["ast_k_i_base"] = standard_normal(
                "ast_k_i_base", (n_bl, n_freq, n_ast_k)
            )

            return params

        return set_params

    def build_constants(self):
        return {
            "sigma_ast_k": self.sigma_ast_k,
            "mu_ast_k": self.mu_ast_k,
        }

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix
        n_pad = self.n_pad
        forward_transform = self.forward_transform

        def forward(params, state, constants):
            # Pure JAX operations only
            sigma_ast_k = constants[f"{prefix}/sigma_ast_k"]
            mu_ast_k = constants[f"{prefix}/mu_ast_k"]

            ast_k_base = params["ast_k_r_base"] + 1.0j * params["ast_k_i_base"]

            ast_k = forward_transform(ast_k_base, sigma_ast_k, mu_ast_k)

            vis_ast = jnp.fft.ifft(ast_k, axis=2)[:, :, n_pad:-n_pad]

            state = {**state, "vis_ast": state["vis_ast"] + vis_ast}

            return state

        return forward

    def validate_and_test(self):
        """Call this before using in JIT context"""
        pass

    def _compute_gp_params(self):

        self.n_pad = max(int(self.ast_pad_factor * self.n_time / 2), 0)
        self.n_ast_k = self.n_time + 2 * self.n_pad

        self.k_ast = jnp.fft.fftfreq(self.n_ast_k, self.int_time)

        if self.fov_deg:
            eff_dish_d = 1.22 * 3e8 / (jnp.min(self.freqs) * jnp.deg2rad(self.fov_deg))
        else:
            eff_dish_d = self.dish_d

        self.ast_fr = vmap(get_ast_fringe_rate, (None, 0, None), (1))(
            self.uvw[:, :, :2], self.freqs, eff_dish_d
        )

    def _compute_true_params(self, zarr_path, data_col):

        xds = xr.open_zarr(zarr_path)

        data_type = get_observation_data_type(data_col)

        vis_ast = jnp.transpose(
            (
                xds.vis_ast.data[:, :, :].compute()
                if data_type["ast"]
                else jnp.zeros_like((xds.vis_ast.data[:, :, :]))
            ),
            (1, 2, 0),
        )

        vis_ast_padded = vmap(
            vmap(jnp.pad, in_axes=(0, None, None), out_axes=(0)),
            in_axes=(1, None, None),
            out_axes=(1),
        )(vis_ast, self.n_pad, "linear_ramp")
        self.true_ast_k = jnp.fft.fft(vis_ast_padded, axis=2)

        self.true_ast_k_base = self.inv_transform(
            self.true_ast_k, self.sigma_ast_k, self.mu_ast_k
        )

    def _compute_prior_params(self):

        sqrt_Pk = lambda k0: jnp.sqrt(pow_spec(self.k_ast, self.p0, k0, self.gamma))

        self.sigma_ast_k = vmap(vmap(sqrt_Pk, (0), (0)), (1), (1))(self.ast_fr)
        self.mu_ast_k = jnp.zeros((self.n_bl, self.n_freq, self.n_ast_k), dtype=complex)

    def _set_outputs(self):

        self.state_outputs = {
            "vis_ast": jnp.zeros((self.n_bl, self.n_freq, self.n_time), dtype=complex),
        }

    def forward_transform(self, base_params, sigma, mu):

        params = sigma * base_params + mu

        return params

    def inv_transform(self, params, sigma, mu):

        base_params = (params - mu) / sigma

        return base_params

    def _compute_init_params(self, init_type: str):

        if init_type == "prior":
            self.init_ast_k = self.mu_ast_k
        elif init_type == "truth":
            self.init_ast_k = self.true_ast_k
        else:
            prior_sample = random.normal(
                random.PRNGKey(1),
                (self.n_bl, self.n_freq, self.n_ast_k),
                dtype=complex,
            )
            self.init_ast_k = self.forward_transform(
                prior_sample, self.sigma_ast_k, self.mu_ast_k
            )

        self.init_ast_k_base = self.inv_transform(
            self.init_ast_k, self.sigma_ast_k, self.mu_ast_k
        )

        self.init_params = {
            "ast_k_r": self.init_ast_k.real,
            "ast_k_i": self.init_ast_k.imag,
        }
        self.init_params_base = {
            "ast_k_r_base": self.init_ast_k_base.real,
            "ast_k_i_base": self.init_ast_k_base.imag,
        }

    def _validate_dimensions(self):
        """Ensure all setup operations completed successfully"""

        ast_shape = (self.n_bl, self.n_freq, self.n_ast_k)

        assert_attr_shape(self, "mu_ast_k", ast_shape)
        assert_attr_shape(self, "sigma_ast_k", ast_shape)
        assert_attr_shape(self, "init_ast_k", ast_shape)
        assert_attr_shape(self, "init_ast_k_base", ast_shape)


class FourierTimeConstFreqAst(Component):

    required_inputs = {}  # No inputs needed
    output_shape = {
        "vis_ast": ("n_bl", 1, "n_time"),
    }

    # Add parameter specifications
    parameters = {
        "ast_k_r_base": ("n_bl", 1, "n_k_ast"),
        "ast_k_i_base": ("n_bl", 1, "n_k_ast"),
    }

    def setup(self, config):
        """All validation and error-prone operations here"""
        try:
            # Store only what's needed for forward computation
            self.n_time = config.n_time
            self.n_bl = config.n_bl
            self.n_freq = config.n_freq
            self.int_time = config.int_time
            self.dish_d = config.dish_d
            self.uvw = config.uvw
            self.freqs = config.freqs
            self.p0 = config.args["ast"]["pow_spec"]["p0"]
            self.gamma = config.args["ast"]["pow_spec"]["gamma"]
            self.fov_deg = config.args["ast"]["pow_spec"]["fov_deg"]
            self.ast_pad_factor = config.args["ast"]["time_pad_factor"]

            # Do expensive setup operations once
            self._compute_gp_params()
            self._compute_prior_params()

            if config.args["plots"]["truth"] or config.args["ast"]["init"] == "truth":
                self._compute_true_params(
                    config.args["data"]["zarr_path"], config.args["data"]["data_col"]
                )

            self._compute_init_params(config.args["ast"]["init"])
            self._set_outputs()

            # Validate dimensions
            self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"ComplexRFI setup failed: {e}")

    def build_set_params(self):
        n_bl = self.n_bl
        n_ast_k = self.n_ast_k

        def set_params(params):

            params["ast_k_r_base"] = standard_normal("ast_k_r_base", (n_bl, 1, n_ast_k))
            params["ast_k_i_base"] = standard_normal("ast_k_i_base", (n_bl, 1, n_ast_k))

            return params

        return set_params

    def build_constants(self):
        return {
            "sigma_ast_k": self.sigma_ast_k,
            "mu_ast_k": self.mu_ast_k,
        }

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix
        n_pad = self.n_pad
        forward_transform = self.forward_transform

        def forward(params, state, constants):
            # Pure JAX operations only
            sigma_ast_k = constants[f"{prefix}/sigma_ast_k"]
            mu_ast_k = constants[f"{prefix}/mu_ast_k"]

            ast_k_base = params["ast_k_r_base"] + 1.0j * params["ast_k_i_base"]

            ast_k = forward_transform(ast_k_base, sigma_ast_k, mu_ast_k)

            vis_ast = jnp.fft.ifft(ast_k, axis=2)[:, :, n_pad:-n_pad]

            state = {**state, "vis_ast": state["vis_ast"] + vis_ast}

            return state

        return forward

    def validate_and_test(self):
        """Call this before using in JIT context"""
        pass

    def _compute_gp_params(self):

        self.n_pad = max(int(self.ast_pad_factor * self.n_time / 2), 0)
        self.n_ast_k = self.n_time + 2 * self.n_pad

        self.k_ast = jnp.fft.fftfreq(self.n_ast_k, self.int_time)

        if self.fov_deg:
            eff_dish_d = 1.22 * 3e8 / (jnp.min(self.freqs) * jnp.deg2rad(self.fov_deg))
        else:
            eff_dish_d = self.dish_d

        self.ast_fr = vmap(get_ast_fringe_rate, (None, 0, None), (1))(
            self.uvw[:, :, :2], self.freqs, eff_dish_d
        )

    def _compute_true_params(self, zarr_path, data_col):

        xds = xr.open_zarr(zarr_path)

        data_type = get_observation_data_type(data_col)

        vis_ast = jnp.transpose(
            (
                xds.vis_ast.data[:, :, :].compute()
                if data_type["ast"]
                else jnp.zeros_like((xds.vis_ast.data[:, :, :]))
            ),
            (1, 2, 0),
        )

        vis_ast_padded = vmap(
            vmap(jnp.pad, in_axes=(0, None, None), out_axes=(0)),
            in_axes=(1, None, None),
            out_axes=(1),
        )(vis_ast, self.n_pad, "linear_ramp")
        self.true_ast_k = jnp.fft.fft(vis_ast_padded, axis=2)

        self.true_ast_k_base = self.inv_transform(
            self.true_ast_k, self.sigma_ast_k, self.mu_ast_k
        )

    def _compute_prior_params(self):

        sqrt_Pk = lambda k0: jnp.sqrt(pow_spec(self.k_ast, self.p0, k0, self.gamma))

        self.sigma_ast_k = vmap(vmap(sqrt_Pk, (0), (0)), (1), (1))(self.ast_fr)
        self.mu_ast_k = jnp.zeros((self.n_bl, self.n_freq, self.n_ast_k), dtype=complex)

    def _set_outputs(self):

        self.state_outputs = {
            "vis_ast": jnp.zeros((self.n_bl, self.n_freq, self.n_time), dtype=complex),
        }

    def forward_transform(self, base_params, sigma, mu):

        params = sigma * base_params + mu

        return params

    def inv_transform(self, params, sigma, mu):

        base_params = (params - mu) / sigma

        return base_params

    def _compute_init_params(self, init_type: str):

        if init_type == "prior":
            self.init_ast_k = self.mu_ast_k
        elif init_type == "truth":
            self.init_ast_k = self.true_ast_k
        else:
            prior_sample = random.normal(
                random.PRNGKey(1),
                (self.n_bl, self.n_freq, self.n_ast_k),
                dtype=complex,
            )
            self.init_ast_k = self.forward_transform(
                prior_sample, self.sigma_ast_k, self.mu_ast_k
            )

        self.init_ast_k_base = self.inv_transform(
            self.init_ast_k, self.sigma_ast_k, self.mu_ast_k
        )

        self.init_params = {
            "ast_k_r": self.init_ast_k.real,
            "ast_k_i": self.init_ast_k.imag,
        }
        self.init_params_base = {
            "ast_k_r_base": self.init_ast_k_base.real,
            "ast_k_i_base": self.init_ast_k_base.imag,
        }

    def _validate_dimensions(self):
        """Ensure all setup operations completed successfully"""

        ast_shape = (self.n_bl, self.n_freq, self.n_ast_k)

        assert_attr_shape(self, "mu_ast_k", ast_shape)
        assert_attr_shape(self, "sigma_ast_k", ast_shape)
        assert_attr_shape(self, "init_ast_k", ast_shape)
        assert_attr_shape(self, "init_ast_k_base", ast_shape)


class FourierTimeFreqAst(Component):

    required_inputs = {}  # No inputs needed
    output_shape = {
        "vis_ast": ("n_bl", "n_freq", "n_time"),
    }

    # Add parameter specifications
    parameters = {
        "ast_k_r_base": ("n_bl", "n_freq", "n_k_ast"),
        "ast_k_i_base": ("n_bl", "n_freq", "n_k_ast"),
    }

    def setup(self, config):
        """All validation and error-prone operations here"""
        try:
            # Store only what's needed for forward computation
            self.n_time = config.n_time
            self.n_bl = config.n_bl
            self.n_freq = config.n_freq
            self.int_time = config.int_time
            self.dish_d = config.dish_d
            self.uvw = config.uvw
            self.freqs = config.freqs
            self.p0 = config.args["ast"]["pow_spec"]["p0"]
            self.gamma = config.args["ast"]["pow_spec"]["gamma"]
            self.fov_deg = config.args["ast"]["pow_spec"]["fov_deg"]
            self.ast_pad_factor = config.args["ast"]["time_pad_factor"]

            # Do expensive setup operations once
            self._compute_gp_params()
            self._compute_prior_params()

            if config.args["plots"]["truth"] or config.args["ast"]["init"] == "truth":
                self._compute_true_params(
                    config.args["data"]["zarr_path"], config.args["data"]["data_col"]
                )

            self._compute_init_params(config.args["ast"]["init"])
            self._set_outputs()

            # Validate dimensions
            self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"ComplexRFI setup failed: {e}")

    def build_set_params(self):
        n_bl = self.n_bl
        n_freq = self.n_freq
        n_ast_k = self.n_ast_k

        def set_params(params):

            params["ast_k_r_base"] = standard_normal(
                "ast_k_r_base", (n_bl, n_freq, n_ast_k)
            )
            params["ast_k_i_base"] = standard_normal(
                "ast_k_i_base", (n_bl, n_freq, n_ast_k)
            )

            return params

        return set_params

    def build_constants(self):
        return {
            "sigma_ast_k": self.sigma_ast_k,
            "mu_ast_k": self.mu_ast_k,
        }

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix
        n_pad = self.n_pad
        forward_transform = self.forward_transform

        def forward(params, state, constants):
            # Pure JAX operations only
            sigma_ast_k = constants[f"{prefix}/sigma_ast_k"]
            mu_ast_k = constants[f"{prefix}/mu_ast_k"]

            ast_k_base = params["ast_k_r_base"] + 1.0j * params["ast_k_i_base"]

            ast_k = forward_transform(ast_k_base, sigma_ast_k, mu_ast_k)

            vis_ast = jnp.fft.ifft2(ast_k, axes=(1, 2))[:, :, n_pad:-n_pad]

            state = {**state, "vis_ast": state["vis_ast"] + vis_ast}

            return state

        return forward

    def validate_and_test(self):
        """Call this before using in JIT context"""
        pass

    def _compute_gp_params(self):

        self.n_pad = max(int(self.ast_pad_factor * self.n_time / 2), 0)
        self.n_ast_k = self.n_time + 2 * self.n_pad

        self.k_ast = jnp.fft.fftfreq(self.n_ast_k, self.int_time)

        if self.fov_deg:
            eff_dish_d = 1.22 * 3e8 / (jnp.min(self.freqs) * jnp.deg2rad(self.fov_deg))
        else:
            eff_dish_d = self.dish_d

        self.ast_fr = vmap(get_ast_fringe_rate, (None, 0, None), (1))(
            self.uvw[:, :, :2], self.freqs, eff_dish_d
        )

    # def _compute_true_params(self, zarr_path, data_col):

    #     xds = xr.open_zarr(zarr_path)

    #     data_type = get_observation_data_type(data_col)

    #     vis_ast = jnp.transpose(
    #         (
    #             xds.vis_ast.data[:, :, :].compute()
    #             if data_type["ast"]
    #             else jnp.zeros_like((xds.vis_ast.data[:, :, :]))
    #         ),
    #         (1, 2, 0),
    #     )

    #     vis_ast_padded = vmap(
    #         vmap(jnp.pad, in_axes=(0, None, None), out_axes=(0)),
    #         in_axes=(1, None, None),
    #         out_axes=(1),
    #     )(vis_ast, self.n_pad, "linear_ramp")
    #     self.true_ast_k = jnp.fft.fft2(vis_ast_padded, axes=(1, 2))

    #     self.true_ast_k_base = self.inv_transform(
    #         self.true_ast_k, self.sigma_ast_k, self.mu_ast_k
    #     )

    def _compute_true_params(self, zarr_path, data_col):

        xds = xr.open_zarr(zarr_path)

        data_type = get_observation_data_type(data_col)

        vis_ast = jnp.transpose(
            (
                xds.vis_ast.data[:, :, :].compute()
                if data_type["ast"]
                else jnp.zeros_like((xds.vis_ast.data[:, :, :]))
            ),
            (1, 2, 0),
        )

        vis_ast_padded = vmap(
            vmap(jnp.pad, in_axes=(0, None, None), out_axes=(0)),
            in_axes=(1, None, None),
            out_axes=(1),
        )(vis_ast, self.n_pad, "linear_ramp")
        self.true_ast_k = jnp.fft.fft(vis_ast_padded, axis=2)

        self.true_ast_k_base = self.inv_transform(
            self.true_ast_k, self.sigma_ast_k, self.mu_ast_k
        )

    def _compute_prior_params(self):

        sqrt_Pk = lambda k0: jnp.sqrt(pow_spec(self.k_ast, self.p0, k0, self.gamma))

        self.sigma_ast_k = vmap(vmap(sqrt_Pk, (0), (0)), (1), (1))(self.ast_fr)
        if self.n_freq > 1:
            self.sigma_ast_k = self.sigma_ast_k.at[:, 1:, :].set(
                self.sigma_ast_k[:, 1:, :] * 1e-6
            )
        self.mu_ast_k = jnp.zeros((self.n_bl, self.n_freq, self.n_ast_k), dtype=complex)

    def _set_outputs(self):

        self.state_outputs = {
            "vis_ast": jnp.zeros((self.n_bl, self.n_freq, self.n_time), dtype=complex),
        }

    def forward_transform(self, base_params, sigma, mu):

        params = sigma * base_params + mu

        return params

    def inv_transform(self, params, sigma, mu):

        base_params = (params - mu) / sigma

        return base_params

    def _compute_init_params(self, init_type: str):

        if init_type == "prior":
            self.init_ast_k = self.mu_ast_k
        elif init_type == "truth":
            self.init_ast_k = self.true_ast_k
        else:
            prior_sample = random.normal(
                random.PRNGKey(1),
                (self.n_bl, self.n_freq, self.n_ast_k),
                dtype=complex,
            )
            self.init_ast_k = self.forward_transform(
                prior_sample, self.sigma_ast_k, self.mu_ast_k
            )

        self.init_ast_k_base = self.inv_transform(
            self.init_ast_k, self.sigma_ast_k, self.mu_ast_k
        )

        self.init_params = {
            "ast_k_r": self.init_ast_k.real,
            "ast_k_i": self.init_ast_k.imag,
        }
        self.init_params_base = {
            "ast_k_r_base": self.init_ast_k_base.real,
            "ast_k_i_base": self.init_ast_k_base.imag,
        }

    def _validate_dimensions(self):
        """Ensure all setup operations completed successfully"""

        ast_shape = (self.n_bl, self.n_freq, self.n_ast_k)

        assert_attr_shape(self, "mu_ast_k", ast_shape)
        assert_attr_shape(self, "sigma_ast_k", ast_shape)
        assert_attr_shape(self, "init_ast_k", ast_shape)
        assert_attr_shape(self, "init_ast_k_base", ast_shape)


##############################################################################################################


class FourierTimeFreqGPAst(Component):

    required_inputs = {}  # No inputs needed
    output_shape = {
        "vis_ast": ("n_bl", "n_freq", "n_time"),
    }

    # Add parameter specifications
    parameters = {
        "ast_k_r_base": ("n_bl", "n_k_freq_ast", "n_k_time_ast"),
        "ast_k_i_base": ("n_bl", "n_k_freq_ast", "n_k_time_ast"),
    }

    def setup(self, config):
        """All validation and error-prone operations here"""
        try:
            # Store only what's needed for forward computation
            self.n_time = config.n_time
            self.n_bl = config.n_bl
            self.n_freq = config.n_freq
            self.int_time = config.int_time
            self.chan_width = config.chan_width
            self.dish_d = config.dish_d
            self.uvw = config.uvw
            self.freqs = config.freqs
            self.times = config.times

            self.p0 = config.args["ast"]["pow_spec"]["p0"]
            self.gammas = config.args["ast"]["pow_spec"]["gammas"]
            self.fov_deg = config.args["ast"]["pow_spec"]["fov_deg"]
            self.k0_freq = config.args["ast"]["pow_spec"]["k0_freq"]
            self.pk_cutoff = config.args["ast"]["pow_spec"]["cutoff"]

            self.freq_pad_factor = config.args["ast"]["freq_pad_factor"]
            self.time_pad_factor = config.args["ast"]["time_pad_factor"]

            self.xs = [self.freqs, self.times]
            self.pad_factors = [self.freq_pad_factor, self.time_pad_factor]
            self.ss_factors = [1, 1]

            # Do expensive setup operations once
            self._compute_gp_params()
            self._compute_prior_params(config.args["ast"]["mean"], config.vis_obs)

            if config.args["plots"]["truth"] or config.args["ast"]["init"] == "truth":
                self._compute_true_params(
                    config.args["data"]["zarr_path"], config.args["data"]["data_col"]
                )

            # self._compute_init_params(config.args["ast"]["init"])
            self._compute_init_params(config.args["ast"]["init"], config.vis_obs)
            self._set_outputs()

            # Validate dimensions
            self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"FourierTimeFreqGPAst setup failed: {e}")

    def build_set_params(self):
        n_bl = self.n_bl
        n_k_freq_ast = self.n_k_freq_ast
        n_k_time_ast = self.n_k_time_ast

        def set_params(params):

            params["ast_k_r_base"] = standard_normal(
                "ast_k_r_base", (n_bl, n_k_freq_ast, n_k_time_ast)
            )
            params["ast_k_i_base"] = standard_normal(
                "ast_k_i_base", (n_bl, n_k_freq_ast, n_k_time_ast)
            )

            return params

        return set_params

    def build_constants(self):
        return {
            "sigma_ast_k": self.sigma_ast_k,
            "mu_ast_k": self.mu_ast_k,
        }

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix
        pads = self.pads
        ss_idxs = self.ss_idxs
        forward_transform = self.forward_transform

        def forward(params, state, constants):
            # Pure JAX operations only
            sigma_ast_k = constants[f"{prefix}/sigma_ast_k"]
            mu_ast_k = constants[f"{prefix}/mu_ast_k"]

            ast_k_base = params["ast_k_r_base"] + 1.0j * params["ast_k_i_base"]

            ast_k = forward_transform(ast_k_base, sigma_ast_k, mu_ast_k)

            vis_ast = vmap(latent_to_signal, (0, None, None), 0)(ast_k, pads, ss_idxs)

            state = {**state, "vis_ast": state["vis_ast"] + vis_ast}

            return state

        return forward

    def validate_and_test(self):
        """Call this before using in JIT context"""
        pass

    def _compute_gp_params(self):

        if self.fov_deg:
            eff_dish_d = float(
                1.22 * 3e8 / (jnp.min(self.freqs) * jnp.deg2rad(self.fov_deg))
            )
        else:
            eff_dish_d = self.dish_d

        # self.ast_fr = vmap(get_ast_fringe_rate, (None, 0, None), (1))(
        #     self.uvw[:, :, :2], self.freqs, eff_dish_d
        # ) # Separate Fringe Rate for each baseline and frequency
        self.ast_fr = get_ast_fringe_rate(
            self.uvw[:, :, :2], self.freqs.max(), eff_dish_d
        )

        self.k0_time = self.ast_fr
        self.k0s = [self.k0_freq, self.k0_time.max()]

        ns = [self.n_freq, self.n_time]
        dxs = [self.chan_width, self.int_time]

        self.pk, self.ks, self.pads, self.ss_idxs = latent_to_signal_init(
            ns,
            dxs,
            self.pad_factors,
            self.ss_factors,
            self.p0,
            self.k0s,
            self.gammas,
            self.pk_cutoff,
        )

        # Pre-compute slicing indices for JIT-compatible latent extraction
        self.latent_idxs, _ = signal_to_latent_init(
            ns,
            dxs,
            self.pad_factors,
            self.p0,
            self.k0s,
            self.gammas,
            self.pk_cutoff,
        )

        self.signal_to_latent = lambda vis_ast: vmap(signal_to_latent, (0, None, None), 0)(vis_ast, self.pad_factors, self.latent_idxs)

        print("\nAST specs")
        print(f"(d_freq, d_time): ({dxs[0]:.3e}, {dxs[1]:.3e})")
        print(f"(n_freq, n_time): ({self.n_freq}, {self.n_time})")
        print(f"(n_k_fq, n_k_tm): {self.pk.shape}")

        self.n_k_freq_ast, self.n_k_time_ast = self.pk.shape

        sigma = lambda k0: jnp.sqrt(
            pow_spec_nd(self.ks, self.p0, [self.k0_freq, k0], self.gammas)
            / self.pk.size
        )

        self.sigma_ast_k = vmap(sigma, (0), 0)(self.k0_time)

    @measure_runtime
    def _compute_true_params(self, zarr_path, data_col):

        xds = xr.open_zarr(zarr_path)

        data_type = get_observation_data_type(data_col)

        true_vis_ast = jnp.transpose(
            (
                xds.vis_ast.data[:, :, :].compute()
                if data_type["ast"]
                else jnp.zeros_like((xds.vis_ast.data[:, :, :]))
            ),
            (1, 2, 0),
        )

        self.true_ast_k = self.signal_to_latent(true_vis_ast)
        
        self.true_ast_k_base = self.inv_transform(
            self.true_ast_k, self.sigma_ast_k, self.mu_ast_k
        )

    def _compute_prior_params(self, prior_type: str, vis_obs):

        if prior_type == "data":
            print("Using data for AST prior mean")
            self.mu_ast_k = self._compute_data_est(vis_obs)
        elif prior_type in ["zeros", 0]:
            print("Using zeros for AST prior mean")
            self.mu_ast_k = jnp.zeros(
                (self.n_bl, self.n_k_freq_ast, self.n_k_time_ast), dtype=complex
            )
        else:
            raise ValueError(f"Provided prior type: {prior_type} is not valid. Choose from (data, zeros).")

    def _set_outputs(self):

        self.state_outputs = {
            "vis_ast": jnp.zeros((self.n_bl, self.n_freq, self.n_time), dtype=complex),
        }

    def forward_transform(self, base_params, sigma, mu):

        params = sigma * base_params + mu

        return params

    def inv_transform(self, params, sigma, mu):

        base_params = (params - mu) / sigma

        return base_params
    
    def _compute_data_est(self, vis_obs):

        est_ast_k =  self.signal_to_latent(vis_obs)

        return est_ast_k

    def _compute_init_params(self, init_type: str, vis_obs):

        if init_type == "data":
            print("Using data for AST init")
            self.init_ast_k = self._compute_data_est(vis_obs)
        elif init_type == "prior":
            print("Using prior mean for AST init")
            self.init_ast_k = self.mu_ast_k
        elif init_type == "truth":
            print("Using truth for AST init")
            self.init_ast_k = self.true_ast_k
        elif init_type == "sample":
            print("Using prior sample for AST init")
            prior_sample = random.normal(
                random.PRNGKey(1),
                (self.n_bl, self.n_k_freq_ast, self.n_k_time_ast),
                dtype=complex,
            )
            self.init_ast_k = self.forward_transform(
                prior_sample, self.sigma_ast_k, self.mu_ast_k
            )
        else:
            raise ValueError(f"Provided init type: {init_type} is not valid. Choose from (data, prior, truth, sample, zeros).")

        self.init_ast_k_base = self.inv_transform(
            self.init_ast_k, self.sigma_ast_k, self.mu_ast_k
        )

        self.init_params = {
            "ast_k_r": self.init_ast_k.real,
            "ast_k_i": self.init_ast_k.imag,
        }
        self.init_params_base = {
            "ast_k_r_base": self.init_ast_k_base.real,
            "ast_k_i_base": self.init_ast_k_base.imag,
        }

    def _validate_dimensions(self):
        """Ensure all setup operations completed successfully"""

        ast_shape = (self.n_bl, self.n_k_freq_ast, self.n_k_time_ast)

        assert_attr_shape(self, "mu_ast_k", ast_shape)
        assert_attr_shape(self, "sigma_ast_k", ast_shape)
        assert_attr_shape(self, "init_ast_k", ast_shape)
        assert_attr_shape(self, "init_ast_k_base", ast_shape)


class PointSourceVisCalculation(Component):
    """No-parameter component that computes point-source visibilities via a
    direct DFT.

    Reads ``ast_radec`` (RA/Dec in radians, shape ``(n_src, 2)``) and
    ``ast_I`` (flux density, shape ``(n_src, n_freq)``) from the state and
    accumulates the result into ``vis_ast`` using the full w-projection
    visibility equation:

        V(u,v,w) = Σ_k (I_k/n_k) exp(-2πi (u l_k + v m_k + w (n_k-1)) / λ)

    For a sparse (point-source) sky the direct sum is exact, gridless and fully
    differentiable, and is unaffected by field of view or baseline length. A
    type-3 NUFFT would be cheaper only for a *dense* sky with a moderate
    space-bandwidth product; for points its internal grid scales as the cube of
    ``max|uvw|/λ · max|lmn|`` and blows up for wide fields on long baselines.
    """

    required_inputs = {
        "ast_radec": ("n_src", 2),
        "ast_I": ("n_src", "n_freq"),
    }
    output_shape = {"vis_ast": ("n_bl", "n_freq", "n_time")}
    parameters = {}

    def setup(self, config):
        try:
            self.n_bl = config.n_bl
            self.n_freq = config.n_freq
            self.n_time = config.n_time
            self.uvw = config.uvw
            self.freqs = config.freqs
            self.phase_centre_ra = jnp.deg2rad(config.phase_centre["ra"])
            self.phase_centre_dec = jnp.deg2rad(config.phase_centre["dec"])
            self._set_outputs()
        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def build_set_params(self):
        def set_params(params):
            return params
        return set_params

    def build_constants(self):
        return {
            "uvw": self.uvw,
            "freqs": self.freqs,
            "ra0": self.phase_centre_ra,
            "dec0": self.phase_centre_dec,
        }

    def build_forward(self):
        prefix = self.prefix
        n_bl = self.n_bl
        n_freq = self.n_freq
        n_time = self.n_time
        C = 299792458.0

        def forward(params, state, constants):
            uvw = constants[f"{prefix}/uvw"]      # (n_bl, n_time, 3)
            freqs = constants[f"{prefix}/freqs"]  # (n_freq,)
            ra0 = constants[f"{prefix}/ra0"]
            dec0 = constants[f"{prefix}/dec0"]

            ra = state["ast_radec"][:, 0]   # (n_src,)
            dec = state["ast_radec"][:, 1]
            I = state["ast_I"]              # (n_src, n_freq)

            dra = ra - ra0
            l = jnp.cos(dec) * jnp.sin(dra)
            m = jnp.sin(dec) * jnp.cos(dec0) - jnp.cos(dec) * jnp.sin(dec0) * jnp.cos(dra)
            n = jnp.sqrt(1.0 - l**2 - m**2)    # (n_src,)

            # Geometric path-length delay per (baseline, time, source), in metres
            lmn = jnp.stack([l, m, n - 1.0], axis=-1)         # (n_src, 3)
            tau = jnp.einsum("btx,sx->bts", uvw, lmn)         # (n_bl, n_time, n_src)
            weights = I / n[:, None]                          # (n_src, n_freq)

            # vmap over frequency channels — avoids a 4D (bl,time,src,freq) array
            def vis_at_freq(freq, w_freq):
                phase = -2.0 * jnp.pi * tau * freq / C        # (n_bl, n_time, n_src)
                fringe = jnp.exp(1.0j * phase)
                return jnp.sum(fringe * w_freq, axis=-1)      # (n_bl, n_time)

            vis = vmap(vis_at_freq)(freqs, weights.T)         # (n_freq, n_bl, n_time)
            vis_ast = vis.transpose(1, 0, 2)                  # (n_bl, n_freq, n_time)

            state = {**state, "vis_ast": state["vis_ast"] + vis_ast}
            return state

        return forward

    def _set_outputs(self):
        self.state_outputs = {
            "vis_ast": jnp.zeros((self.n_bl, self.n_freq, self.n_time), dtype=complex),
        }
