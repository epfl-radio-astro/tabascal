from jax import vmap, random
import jax.numpy as jnp

from tabascal.components import Component, assert_attr_shape
from tabascal.dist import standard_normal
from tabascal.interferometry import fov_to_eff_diameter, max_ast_fringe_rate
from tabascal.fft_gp import latent_to_signal_init, latent_to_signal, signal_to_latent_init, signal_to_latent, pow_spec_nd
from tabascal.timing import measure_runtime
from tabascal.truth import read_true_vis_ast


class GPVisAst(Component):

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
            self.dec = config.phase_centre["dec"]
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
            raise RuntimeError(f"GPVisAst setup failed: {e}")

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
            # fov_deg is the full field of view (diameter) out to the first null;
            # the effective diameter makes the beam radius in
            # max_ast_fringe_rate equal to fov_deg / 2.
            eff_dish_d = float(fov_to_eff_diameter(self.fov_deg, jnp.min(self.freqs)))
        else:
            eff_dish_d = self.dish_d

        # One maximum fringe rate per baseline; time and frequency are reduced
        # inside max_ast_fringe_rate.
        self.ast_fr = max_ast_fringe_rate(
            self.uvw, self.dec, self.freqs, eff_dish_d
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

        true_vis_ast = read_true_vis_ast(zarr_path, data_col)

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
