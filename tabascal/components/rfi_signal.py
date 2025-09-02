from jax import vmap, random
import jax.numpy as jnp

from tabascal.components import Component, assert_attr_shape
from tabascal.dist import standard_normal
from tabascal.transform import affine_transform_full
from tabascal.gp import cholesky, resampling_kernel, get_times
from tabascal.tab_tools_new import get_observation_data_type

import xarray as xr


class ComplexRFI(Component):

    required_inputs = {}  # No inputs needed
    output_shapes = {
        "rfi_A": ("n_rfi", "n_ant", "n_freq", "n_time_fine"),
    }

    # Add parameter specifications
    parameter_shapes = {
        "rfi_r_induce_base": ("n_rfi", "n_ant", "n_freq", "n_rfi_time"),
        "rfi_i_induce_base": ("n_rfi", "n_ant", "n_freq", "n_rfi_time"),
    }

    def setup(self, config):
        """All validation and error-prone operations here"""
        try:
            # Store only what's needed for forward computation
            self.n_rfi = config.n_rfi
            self.n_ant = config.n_ant
            self.n_freq = config.n_freq
            self.n_time_fine = config.n_time_fine
            self.times = config.times
            self.times_fine = config.times_fine
            self.vis_obs = config.vis_obs
            self.gp_var = config.args["rfi"]["var"]
            self.gp_l = config.args["rfi"]["corr_time"]

            # Do expensive setup operations once
            self._compute_gp_params()
            self._compute_prior_params()
            self._set_outputs()

            if config.args["plots"]["truth"] or config.args["rfi"]["init"] == "truth":
                self._compute_true_params(
                    config.args["data"]["zarr_path"], config.args["data"]["data_col"]
                )

            # if config.args["rfi"]["init"] == "est":
            #     self._estimate_params(config.fringe_freqs)

            self._compute_init_params(config.args["rfi"]["init"])

            # Validate dimensions
            self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"ComplexRFI setup failed: {e}")

    def build_set_params(self):
        n_rfi = self.n_rfi
        n_ant = self.n_ant
        n_freq = self.n_freq
        n_rfi_times = self.n_rfi_times

        def set_params(params):

            params["rfi_r_induce_base"] = standard_normal(
                "rfi_r_induce_base", (n_rfi, n_ant, n_freq, n_rfi_times)
            )
            params["rfi_i_induce_base"] = standard_normal(
                "rfi_i_induce_base", (n_rfi, n_ant, n_freq, n_rfi_times)
            )

            return params

        return set_params

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        # Pre-compute everything possible
        L_rfi_A = self.L_rfi_A
        mu_rfi_A = self.mu_rfi_A
        resample_rfi = self.resample_rfi
        forward_transform = self.forward_transform

        def forward(params: dict, state: dict):
            # Pure JAX operations only

            rfi_A_induce_base = (
                params["rfi_r_induce_base"] + 1.0j * params["rfi_i_induce_base"]
            )

            rfi_A_induce = forward_transform(rfi_A_induce_base, L_rfi_A, mu_rfi_A)

            rfi_A = vmap(vmap(vmap(jnp.dot, (None, 0), 0), (None, 1), 1), (None, 2), 2)(
                resample_rfi, rfi_A_induce
            )
            # rfi_A = vmap(vmap(jnp.dot, (None, 0), 0), (None, 1), 1)(
            #     resample_rfi, rfi_A_induce
            # )
            state = {**state, "rfi_A": rfi_A}

            return state

        return forward

    def validate_and_test(self):
        """Call this before using in JIT context"""
        pass

    def _compute_gp_params(self):

        if self.gp_var is None:
            self.gp_var = jnp.max(jnp.abs(self.vis_obs))

        print(f"Using RFI var : {self.gp_var:.3e} Jy")

        if self.gp_l is None:
            self.gp_l = 1.0

        self.rfi_times = get_times(self.times, self.gp_l)
        self.n_rfi_times = len(self.rfi_times)

        self.resample_rfi = resampling_kernel(
            self.rfi_times,
            self.times_fine,
            self.gp_var,
            self.gp_l,
            1e-8,
        )

    def _set_outputs(self):

        self.state_outputs = {
            "rfi_A": jnp.zeros(
                (self.n_rfi, self.n_ant, self.n_freq, self.n_time_fine), dtype=complex
            ),
        }

    def _compute_prior_params(self):

        self.L_rfi_A = cholesky(self.rfi_times, self.gp_var, self.gp_l, 1e-8)
        self.mu_rfi_A = jnp.zeros(
            (self.n_rfi, self.n_ant, self.n_freq, self.n_rfi_times), dtype=complex
        )

    def _compute_true_params(self, zarr_path, data_col):

        xds = xr.open_zarr(zarr_path)

        data_type = get_observation_data_type(data_col)

        self.true_rfi_A_induce = jnp.transpose(
            vmap(
                vmap(
                    vmap(jnp.interp, in_axes=(None, None, 0), out_axes=(0)),
                    in_axes=(None, None, 2),
                    out_axes=(2),
                ),
                in_axes=(None, None, 3),
                out_axes=(3),
            )(
                self.rfi_times,
                xds.time_fine.data,
                (
                    xds.rfi_tle_sat_A[:, :, :, :].data.compute()
                    if data_type["rfi"]
                    else jnp.zeros_like(xds.rfi_tle_sat_A[:, :, :, :].data)
                ),
            ),
            (0, 2, 3, 1),
        )

        self.true_rfi_A_induce_base = self.inv_transform(
            self.true_rfi_A_induce, self.L_rfi_A, self.mu_rfi_A
        )

    # def _estimate_params(self, fringe_freqs):

    #     # rfi_xyz is shape (n_rfi, n_time, 3)
    #     # ants_xyz is shape (n_time, n_ant, 3)
    #     theta = angular_separation(
    #         rfi_xyz,
    #         jnp.mean(ants_xyz, axis=1, keepdims=True),
    #         ms_params["ra"],
    #         ms_params["dec"],
    #     )
    #     # theta is shape (n_rfi, n_time, n_ant)
    #     B = airy_beam(theta, ms_params["freqs"], ms_params["dish_d"])[:, :, 0, 0]
    #     # B is shape (n_rfi, n_time, n_ant, n_freq) -> (n_rfi, n_time)
    #     # fringe_freqs is shape (n_time, n_bl)
    #     bl = jnp.argmin(jnp.max(jnp.abs(fringe_freqs), axis=0))
    #     # vis_obs is shape (n_time, n_bl)
    #     rfi_amp = jnp.sqrt(
    #         jnp.max(jnp.abs(ms_params["vis_obs"][:, bl]))
    #         / jnp.max(jnp.sum(B**2, axis=0))
    #     )

    #     return B * rfi_amp

    def forward_transform(self, base_params, L, mu):

        params = vmap(
            vmap(vmap(affine_transform_full, (0, None, 0), 0), (1, None, 1), 1),
            (2, None, 2),
            2,
        )(base_params, L, mu)
        # params = vmap(vmap(affine_transform_full, (0, None, 0), 0), (1, None, 1), 1)(
        #     base_params, L, mu
        # )

        return params

    def inv_transform(self, params, L, mu):

        base_params = vmap(
            vmap(vmap(jnp.linalg.solve, (None, 0), 0), (None, 1), 1), (None, 2), 2
        )(L, params - mu)
        # base_params = vmap(vmap(jnp.linalg.solve, (None, 0), 0), (None, 1), 1)(
        #     L, params - mu
        # )

        return base_params

    def _compute_init_params(self, init_type):

        if init_type == "prior":
            print("Using prior mean for rfi_A")
            # self.init_rfi_A_induce = jnp.ones_like(self.mu_rfi_A)
            self.init_rfi_A_induce = self.mu_rfi_A
        elif init_type == "truth":
            print("Using truth for rfi_A")
            self.init_rfi_A_induce = self.true_rfi_A_induce
        else:
            print("Drawing sample from prior for rfi_A")
            prior_sample = random.normal(
                random.PRNGKey(1),
                (self.n_rfi, self.n_ant, self.n_freq, self.n_rfi_times),
                dtype=complex,
            )
            self.init_rfi_A_induce = self.forward_transform(
                prior_sample, self.L_rfi_A, self.mu_rfi_A
            )

        self.init_rfi_A_induce_base = self.inv_transform(
            self.init_rfi_A_induce, self.L_rfi_A, self.mu_rfi_A
        )

        self.init_params = {
            "rfi_r_induce": self.init_rfi_A_induce.real,
            "rfi_i_induce": self.init_rfi_A_induce.imag,
        }
        self.init_params_base = {
            "rfi_r_induce_base": self.init_rfi_A_induce_base.real,
            "rfi_i_induce_base": self.init_rfi_A_induce_base.imag,
        }

    def _validate_dimensions(self):
        """Ensure all setup operations completed successfully"""

        rfi_shape = (self.n_rfi, self.n_ant, self.n_freq, self.n_rfi_times)

        assert_attr_shape(self, "mu_rfi_A", rfi_shape)
        assert_attr_shape(self, "L_rfi_A", (self.n_rfi_times, self.n_rfi_times))
        assert_attr_shape(self, "init_rfi_A_induce", rfi_shape)
        assert_attr_shape(self, "init_rfi_A_induce_base", rfi_shape)
