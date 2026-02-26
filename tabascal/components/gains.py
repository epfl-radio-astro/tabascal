from tabascal.components import Component
from tabascal.interferometry import apply_gains
from tabascal.dist import standard_normal

import jax.numpy as jnp


class UnitaryGains(Component):

    required_inputs = {"vis_rfi": ("n_bl", "n_time"), "vis_ast": ("n_bl", "n_time")}
    output_shapes = {"gains": ("n_ant", "n_time"), "vis_obs": ("n_bl", "n_time")}

    parameters = {}

    def setup(self, config):
        """All validation and error-prone operations here"""
        try:

            self.n_ant = config.n_ant
            self.n_bl = config.n_bl
            self.n_freq = config.n_freq
            self.n_time = config.n_time

            # Validate dimensions
            self._set_outputs()
            self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"{self.__name__} setup failed: {e}")

    def _validate_dimensions(self):
        """Ensure all setup operations completed successfully"""
        pass

    def _set_outputs(self):

        self.state_outputs = {
            "gains": jnp.ones((self.n_ant, self.n_freq, self.n_time), dtype=complex),
            "vis_obs": jnp.zeros((self.n_bl, self.n_freq, self.n_time), dtype=complex),
        }

    def build_set_params(self):

        def set_params(params):
            return params

        return set_params

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        # Pre-compute everything possible
        gains = self.state_outputs["gains"]

        def forward(params, state, constants):
            # Pure JAX operations only
            vis_obs = state["vis_rfi"] + state["vis_ast"]
            state = {**state, "vis_obs": vis_obs, "gains": gains}
            return state

        return forward


# class GPGains(Component):

#     required_inputs = {"vis_rfi": ("n_bl", "n_time"), "vis_ast": ("n_bl", "n_time")}
#     outputs = {"vis_obs": ("n_bl", "n_time")}

#     parameters = {
#         "g_amp_induce_base": ("n_ant", "n_g_times"),
#         "g_phase_induce_base": ("n_ant-1", "n_g_times"),
#     }

#     def setup(self, config):
#         """All validation and error-prone operations here"""
#         try:

#             # Validate dimensions
#             self._validate_dimensions()

#         except Exception as e:
#             raise RuntimeError(f"GPGains setup failed: {e}")

#     def _validate_dimensions(self):
#         """Ensure all setup operations completed successfully"""
#         pass

#     def build_set_params(self):
#         n_ant = self.n_ant
#         n_g_times = self.n_g_times

#         def set_params(params):

#             params["g_amp_induce_base"] = standard_normal("g_amp_induce_base", (n_ant, n_g_times))
#             params["g_phase_induce_base"] = standard_normal("g_phase_induce_base", (n_ant-1, n_g_times))

#             return params

#         return set_params

#     def build_forward(self):
#         """Return pure, JIT-compatible function"""
#         # Pre-compute everything possible
#         forward_transform = self.forward_transform

#         def forward(params, state):
#             # Pure JAX operations only


#             vis_obs =
#             vis_obs = state["vis_rfi"] + state["vis_ast"]
#             state = {**state, "vis_obs": vis_obs}
#             return state

#         return forward

#     def _compute_gp_params(self):

#         if self.gp_var is None:
#             self.gp_var = jnp.max(jnp.abs(self.vis_obs))

#         if self.gp_l is None:
#             self.gp_l = 1.0

#         self.rfi_times = get_times(self.times, self.gp_l)
#         self.n_rfi_times = len(self.rfi_times)

#         self.resample_rfi = resampling_kernel(
#             self.rfi_times,
#             self.times_fine,
#             self.gp_var,
#             self.gp_l,
#             1e-8,
#         )

#     def _set_outputs(self):

#         self.state_outputs = {
#             "rfi_A": jnp.zeros(
#                 (self.n_rfi, self.n_ant, self.n_time_fine), dtype=complex
#             ),
#         }

#     def _compute_prior_params(self):

#         self.L_rfi_A = cholesky(self.rfi_times, self.gp_var, self.gp_l, 1e-8)
#         self.mu_rfi_A = jnp.zeros(
#             (self.n_rfi, self.n_ant, self.n_rfi_times), dtype=complex
#         )

#     def forward_transform(self, base_params, L, mu):

#         # params = vmap(
#         #     vmap(vmap(affine_transform_full, (0, None, 0), 0), (1, None, 1), 1),
#         #     (2, None, 2),
#         #     2,
#         # )(base_params, L, mu)
#         params = vmap(vmap(affine_transform_full, (0, None, 0), 0), (1, None, 1), 1)(
#             base_params, L, mu
#         )

#         return params

#     def inv_transform(self, params, L, mu):

#         # base_params = vmap(
#         #     vmap(vmap(jnp.linalg.solve, (None, 0), 0), (None, 1), 1), (None, 2), 2
#         # )(L, params - mu)
#         base_params = vmap(vmap(jnp.linalg.solve, (None, 0), 0), (None, 1), 1)(
#             L, params - mu
#         )

#         return base_params

#     def _compute_init_params(self):

#         self.init_rfi_A_induce = self.mu_rfi_A

#         self.init_rfi_A_induce_base = self.inv_transform(
#             self.init_rfi_A_induce, self.L_rfi_A, self.mu_rfi_A
#         )

#         self.init_params = {
#             "rfi_r_induce": self.init_rfi_A_induce.real,
#             "rfi_i_induce": self.init_rfi_A_induce.imag,
#         }
#         self.init_params_base = {
#             "rfi_r_induce_base": self.init_rfi_A_induce_base.real,
#             "rfi_i_induce_base": self.init_rfi_A_induce_base.imag,
#         }

#     def _validate_dimensions(self):
#         """Ensure all setup operations completed successfully"""

#         rfi_shape = (self.n_rfi, self.n_ant, self.n_rfi_times)

#         assert_attr_shape(self, "mu_rfi_A", rfi_shape)
#         assert_attr_shape(self, "L_rfi_A", (self.n_rfi_times, self.n_rfi_times))
#         assert_attr_shape(self, "init_rfi_A_induce", rfi_shape)
#         assert_attr_shape(self, "init_rfi_A_induce_base", rfi_shape)
