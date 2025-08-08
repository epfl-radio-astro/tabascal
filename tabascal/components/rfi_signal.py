from jax import vmap, jit
import jax.numpy as jnp

from tabascal.components import Component
from tabascal.dist import standard_normal
from tabascal.transform import affine_transform_full
from tabascal.gp import cholesky, resampling_kernel, get_times


class ComplexRFI(Component):

    required_inputs = {}  # No inputs needed
    outputs = {
        "rfi_A": ("n_rfi", "n_ant", "n_time_fine"),
    }

    # Add parameter specifications
    parameters = {
        "rfi_r_induce_base": ("n_rfi", "n_ant", "n_rfi_time"),
        "rfi_i_induce_base": ("n_rfi", "n_ant", "n_rfi_time"),
    }

    def setup(self, config):
        """All validation and error-prone operations here"""
        try:
            # Store only what's needed for forward computation
            self.n_rfi = config.n_rfi
            self.n_ant = config.n_ant
            self.n_time_fine = config.n_time_fine
            self.times = config.times
            self.times_fine = config.times_fine
            self.vis_obs = config.vis_obs
            self.gp_var = config.rfi_var
            self.gp_l = config.rfi_l

            # Do expensive setup operations once
            self._compute_gp_params()
            self._compute_prior_params()
            self._compute_init_params()
            self._set_outputs()

            # Validate dimensions
            self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"ComplexRFI setup failed: {e}")

    def build_set_params(self):
        n_rfi = self.n_rfi
        n_ant = self.n_ant
        n_rfi_times = self.n_rfi_times

        def set_params(state):

            state["rfi_r_induce_base"] = standard_normal(
                "rfi_r_induce_base", (n_rfi, n_ant, n_rfi_times)
            )
            state["rfi_i_induce_base"] = standard_normal(
                "rfi_i_induce_base", (n_rfi, n_ant, n_rfi_times)
            )

            return state

        return set_params

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        # Pre-compute everything possible
        L_rfi_A = self.L_rfi_A
        mu_rfi_A = self.mu_rfi_A
        resample_rfi = self.resample_rfi
        forward_transform = self.forward_transform

        # def forward(state):
        #     # Pure JAX operations only

        #     rfi_A_induce_base = (
        #         state["rfi_r_induce_base"] + 1.0j * state["rfi_i_induce_base"]
        #     )

        #     rfi_A_induce = forward_transform(rfi_A_induce_base, L_rfi_A, mu_rfi_A)

        #     # state["rfi_A"] = vmap(
        #     #     vmap(vmap(jnp.dot, (None, 0), 0), (None, 1), 1), (None, 2), 2
        #     # )(resample_rfi, rfi_A_induce)
        #     state["rfi_A"] = vmap(vmap(jnp.dot, (None, 0), 0), (None, 1), 1)(
        #         resample_rfi, rfi_A_induce
        #     )

        #     return state

        def forward(params, state):
            # Pure JAX operations only

            rfi_A_induce_base = (
                params["rfi_r_induce_base"] + 1.0j * params["rfi_i_induce_base"]
            )

            rfi_A_induce = forward_transform(rfi_A_induce_base, L_rfi_A, mu_rfi_A)

            # state["rfi_A"] = vmap(
            #     vmap(vmap(jnp.dot, (None, 0), 0), (None, 1), 1), (None, 2), 2
            # )(resample_rfi, rfi_A_induce)
            rfi_A = vmap(vmap(jnp.dot, (None, 0), 0), (None, 1), 1)(
                resample_rfi, rfi_A_induce
            )
            state = state._replace(rfi_A=rfi_A)

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

    def _compute_gp_params(self):

        if self.gp_var is None:
            self.gp_var = jnp.max(jnp.abs(self.vis_obs))

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
                (self.n_rfi, self.n_ant, self.n_time_fine), dtype=complex
            ),
        }

    def _compute_prior_params(self):

        self.L_rfi_A = cholesky(self.rfi_times, self.gp_var, self.gp_l, 1e-8)
        self.mu_rfi_A = jnp.zeros(
            (self.n_rfi, self.n_ant, self.n_rfi_times), dtype=complex
        )

    def forward_transform(self, base_params, L, mu):

        # params = vmap(
        #     vmap(vmap(affine_transform_full, (0, None, 0), 0), (1, None, 1), 1),
        #     (2, None, 2),
        #     2,
        # )(base_params, L, mu)
        params = vmap(vmap(affine_transform_full, (0, None, 0), 0), (1, None, 1), 1)(
            base_params, L, mu
        )

        return params

    def inv_transform(self, params, L, mu):

        # base_params = vmap(
        #     vmap(vmap(jnp.linalg.solve, (None, 0), 0), (None, 1), 1), (None, 2), 2
        # )(L, params - mu)
        base_params = vmap(vmap(jnp.linalg.solve, (None, 0), 0), (None, 1), 1)(
            L, params - mu
        )

        return base_params

    def _compute_init_params(self):

        self.init_rfi_A_induce = self.mu_rfi_A

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

        rfi_shape = (self.n_rfi, self.n_ant, self.n_rfi_times)

        assert hasattr(self, "mu_rfi_A")
        assert self.mu_rfi_A.shape == rfi_shape

        assert hasattr(self, "L_rfi_A")
        assert self.L_rfi_A.shape == (self.n_rfi_times, self.n_rfi_times)

        assert hasattr(self, "init_rfi_A_induce")
        assert self.init_rfi_A_induce.shape == rfi_shape

        assert hasattr(self, "init_rfi_A_induce_base")
        assert self.init_rfi_A_induce_base.shape == rfi_shape
