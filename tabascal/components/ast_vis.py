from jax import vmap, random
import jax.numpy as jnp

from tabascal.components import Component, assert_attr_shape
from tabascal.dist import standard_normal
from tabascal.tab_tools import (
    get_ast_fringe_rate,
    pow_spec,
    get_observation_data_type,
)
from tabascal.fft_gp import latent_init, latent_predict, get_latent, pow_spec_nd

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
            self.P0 = config.args["ast"]["pow_spec"]["P0"]
            self.gamma = config.args["ast"]["pow_spec"]["gamma"]
            self.fov_deg = config.args["ast"]["pow_spec"]["fov_deg"]
            self.ast_pad_factor = config.args["ast"]["pad_factor"]

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

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        # Pre-compute everything possible
        sigma_ast_k = self.sigma_ast_k
        mu_ast_k = self.mu_ast_k
        n_pad = self.n_pad
        forward_transform = self.forward_transform

        def forward(params, state):
            # Pure JAX operations only

            ast_k_base = params["ast_k_r_base"] + 1.0j * params["ast_k_i_base"]

            ast_k = forward_transform(ast_k_base, sigma_ast_k, mu_ast_k)

            vis_ast = jnp.fft.ifft(ast_k, axis=2)[:, :, n_pad:-n_pad]
            # vis_ast = jnp.fft.ifft(ast_k, axis=1)[:, n_pad:-n_pad]

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

        sqrt_Pk = lambda k0: jnp.sqrt(pow_spec(self.k_ast, self.P0, k0, self.gamma))

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
            self.P0 = config.args["ast"]["pow_spec"]["P0"]
            self.gamma = config.args["ast"]["pow_spec"]["gamma"]
            self.fov_deg = config.args["ast"]["pow_spec"]["fov_deg"]
            self.ast_pad_factor = config.args["ast"]["pad_factor"]

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

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        # Pre-compute everything possible
        sigma_ast_k = self.sigma_ast_k
        mu_ast_k = self.mu_ast_k
        n_pad = self.n_pad
        forward_transform = self.forward_transform

        def forward(params, state):
            # Pure JAX operations only

            ast_k_base = params["ast_k_r_base"] + 1.0j * params["ast_k_i_base"]

            ast_k = forward_transform(ast_k_base, sigma_ast_k, mu_ast_k)

            vis_ast = jnp.fft.ifft(ast_k, axis=2)[:, :, n_pad:-n_pad]
            # vis_ast = jnp.fft.ifft(ast_k, axis=1)[:, n_pad:-n_pad]

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

        sqrt_Pk = lambda k0: jnp.sqrt(pow_spec(self.k_ast, self.P0, k0, self.gamma))

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
            self.P0 = config.args["ast"]["pow_spec"]["P0"]
            self.gamma = config.args["ast"]["pow_spec"]["gamma"]
            self.fov_deg = config.args["ast"]["pow_spec"]["fov_deg"]
            self.ast_pad_factor = config.args["ast"]["pad_factor"]

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

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        # Pre-compute everything possible
        sigma_ast_k = self.sigma_ast_k
        mu_ast_k = self.mu_ast_k
        n_pad = self.n_pad
        forward_transform = self.forward_transform

        def forward(params, state):
            # Pure JAX operations only

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
        self.true_ast_k = jnp.fft.fft2(vis_ast_padded, axes=(1, 2))

        self.true_ast_k_base = self.inv_transform(
            self.true_ast_k, self.sigma_ast_k, self.mu_ast_k
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

        sqrt_Pk = lambda k0: jnp.sqrt(pow_spec(self.k_ast, self.P0, k0, self.gamma))

        self.sigma_ast_k = vmap(vmap(sqrt_Pk, (0), (0)), (1), (1))(self.ast_fr)
        if self.n_freq > 1:
            self.sigma_ast_k = self.sigma_ast_k.at[:, 1:, :].set(
                self.sigma_ast_k[:, 1:, :] * 1e-6
            )
        self.mu_ast_k = jnp.zeros((self.n_bl, self.n_freq, self.n_ast_k), dtype=complex)

        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        plt.imshow(self.sigma_ast_k[0], aspect="auto", norm=LogNorm())
        plt.colorbar()
        plt.savefig("Pk.png", format="png", dpi=300)

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

            # Do expensive setup operations once
            self._compute_gp_params()
            self._compute_prior_params()

            # if config.args["plots"]["truth"] or config.args["ast"]["init"] == "truth":
            #     self._compute_true_params(
            #         config.args["data"]["zarr_path"], config.args["data"]["data_col"]
            #     )

            self._compute_init_params(config.args["ast"]["init"])
            self._set_outputs()

            # Validate dimensions
            self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"ComplexRFI setup failed: {e}")

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

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        # Pre-compute everything possible
        sigma_ast_k = self.sigma_ast_k
        mu_ast_k = self.mu_ast_k
        pads = self.pads
        ss_idxs = self.ss_idxs
        forward_transform = self.forward_transform

        def forward(params, state):
            # Pure JAX operations only

            ast_k_base = params["ast_k_r_base"] + 1.0j * params["ast_k_i_base"]

            ast_k = forward_transform(ast_k_base, sigma_ast_k, mu_ast_k)

            vis_ast = vmap(latent_predict, (0, None, None), 0)(ast_k, pads, ss_idxs)

            state = {**state, "vis_ast": state["vis_ast"] + vis_ast}

            return state

        return forward

    def validate_and_test(self):
        """Call this before using in JIT context"""
        pass

    def _compute_gp_params(self):

        if self.fov_deg:
            eff_dish_d = 1.22 * 3e8 / (jnp.min(self.freqs) * jnp.deg2rad(self.fov_deg))
        else:
            eff_dish_d = self.dish_d

        # self.ast_fr = vmap(get_ast_fringe_rate, (None, 0, None), (1))(
        #     self.uvw[:, :, :2], self.freqs, eff_dish_d
        # ) # Separate Fringe Rate for each baseline and frequency
        self.ast_fr = get_ast_fringe_rate(
            self.uvw[:, :, :2], self.freqs.max(), eff_dish_d
        )

        self.k0_time = self.ast_fr

        xs = [self.freqs, self.times]
        pad_factors = [self.freq_pad_factor, self.time_pad_factor]
        ss_factors = [1, 1]
        k0s = [self.k0_freq, self.k0_time.max()]

        self.pk, self.ks, self.pads, self.ss_idxs = latent_init(
            xs, pad_factors, ss_factors, self.p0, k0s, self.gammas, self.pk_cutoff
        )

        dxs = tuple([float(jnp.diff(x[:2])[0]) for x in xs])

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

    def _compute_prior_params(self):

        # if self.n_freq > 1:
        #     self.sigma_ast_k = self.sigma_ast_k.at[:, 1:, :].set(
        #         self.sigma_ast_k[:, 1:, :] * 1e-6
        #     )
        self.mu_ast_k = jnp.zeros(
            (self.n_bl, self.n_k_freq_ast, self.n_k_time_ast), dtype=complex
        )

        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        plt.imshow(self.sigma_ast_k[0], aspect="auto", norm=LogNorm())
        plt.colorbar()
        plt.savefig("Pk.png", format="png", dpi=300)

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
        # elif init_type == "truth":
        #     self.init_ast_k = self.true_ast_k
        else:
            prior_sample = random.normal(
                random.PRNGKey(1),
                (self.n_bl, self.n_k_freq_ast, self.n_k_time_ast),
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

        ast_shape = (self.n_bl, self.n_k_freq_ast, self.n_k_time_ast)

        assert_attr_shape(self, "mu_ast_k", ast_shape)
        assert_attr_shape(self, "sigma_ast_k", ast_shape)
        assert_attr_shape(self, "init_ast_k", ast_shape)
        assert_attr_shape(self, "init_ast_k_base", ast_shape)
