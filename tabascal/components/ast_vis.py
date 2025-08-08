from jax import vmap, jit
import jax.numpy as jnp

from tabascal.components import Component
from tabascal.dist import standard_normal
from tabascal.tab_tools import get_ast_fringe_rate, pow_spec


class FourierTimeAst(Component):

    required_inputs = {}  # No inputs needed
    outputs = {
        "vis_ast": ("n_bl", "n_time"),
    }

    # Add parameter specifications
    parameters = {
        "ast_k_r_base": ("n_bl", "n_k_ast"),
        "ast_k_i_base": ("n_bl", "n_k_ast"),
    }

    def setup(self, config):
        """All validation and error-prone operations here"""
        try:
            # Store only what's needed for forward computation
            self.n_time = config.n_time
            self.n_bl = config.n_bl
            self.int_time = config.int_time
            self.P0 = config.P0
            self.gamma = config.gamma
            self.ast_pad_factor = config.ast_pad_factor
            self.fov_deg = config.fov_deg
            self.dish_d = config.dish_d
            self.uvw = config.uvw
            self.freqs = config.freqs

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
        n_bl = self.n_bl
        n_ast_k = self.n_ast_k

        def set_params(state):

            state["ast_k_r_base"] = standard_normal("ast_k_r_base", (n_bl, n_ast_k))
            state["ast_k_i_base"] = standard_normal("ast_k_i_base", (n_bl, n_ast_k))

            return state

        return set_params

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        # Pre-compute everything possible
        sigma_ast_k = self.sigma_ast_k
        mu_ast_k = self.mu_ast_k
        n_pad = self.n_pad
        forward_transform = self.forward_transform

        # def forward(state):
        #     # Pure JAX operations only

        #     ast_k_base = state["ast_k_r_base"] + 1.0j * state["ast_k_i_base"]

        #     ast_k = forward_transform(ast_k_base, sigma_ast_k, mu_ast_k)

        #     # vis_ast = jnp.fft.ifft(ast_k, axis=1)[:, :, n_pad:-n_pad]
        #     vis_ast = jnp.fft.ifft(ast_k, axis=1)[:, n_pad:-n_pad]

        #     state["vis_ast"] = vis_ast

        #     # state = {
        #     #     **state,
        #     #     "vis_ast": state["vis_ast"] + vis_ast,
        #     # }  # instead of state["vis_ast"] = state["vis_ast"] + vis_ast

        #     return state

        def forward(params, state):
            # Pure JAX operations only

            ast_k_base = params["ast_k_r_base"] + 1.0j * params["ast_k_i_base"]

            ast_k = forward_transform(ast_k_base, sigma_ast_k, mu_ast_k)

            # vis_ast = jnp.fft.ifft(ast_k, axis=1)[:, :, n_pad:-n_pad]
            vis_ast = jnp.fft.ifft(ast_k, axis=1)[:, n_pad:-n_pad]

            state = state._replace(vis_ast=vis_ast)

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
            eff_dish_d = 1.22 * 3e8 / (self.freqs * jnp.deg2rad(self.fov_deg))
        else:
            eff_dish_d = self.dish_d

        self.ast_fr = get_ast_fringe_rate(self.uvw[:, :, :2], self.freqs, eff_dish_d)

    def _compute_prior_params(self):

        sqrt_Pk = lambda k0: jnp.sqrt(pow_spec(self.k_ast, self.P0, k0, self.gamma))

        self.sigma_ast_k = vmap(sqrt_Pk)(self.ast_fr)  # Add freq axis
        self.mu_ast_k = jnp.zeros((self.n_bl, self.n_ast_k), dtype=complex)

    def _set_outputs(self):

        self.state_outputs = {
            "vis_ast": jnp.zeros((self.n_bl, self.n_time), dtype=complex),
        }

    def forward_transform(self, base_params, sigma, mu):

        params = sigma * base_params + mu

        return params

    def inv_transform(self, params, sigma, mu):

        base_params = (params - mu) / sigma

        return base_params

    def _compute_init_params(self):

        self.init_ast_k = self.mu_ast_k
        self.init_ast_k_base = (self.init_ast_k - self.mu_ast_k) / self.sigma_ast_k

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

        ast_shape = (self.n_bl, self.n_ast_k)

        assert hasattr(self, "mu_ast_k")
        assert self.mu_ast_k.shape == ast_shape

        assert hasattr(self, "sigma_ast_k")
        assert self.sigma_ast_k.shape == ast_shape

        assert hasattr(self, "init_ast_k")
        assert self.init_ast_k.shape == ast_shape

        assert hasattr(self, "init_ast_k_base")
        assert self.init_ast_k_base.shape == ast_shape
