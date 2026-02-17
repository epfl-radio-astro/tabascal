from jax import vmap, random, jit, Array
import jax.numpy as jnp

from tabascal.components import Component, assert_attr_shape
from tabascal.dist import standard_normal
from tabascal.transform import affine_transform_full
from tabascal.gp import cholesky, resampling_kernel, get_times
from tabascal.tab_tools import get_observation_data_type
from tabascal.fft_gp import latent_to_signal_init, latent_to_signal, signal_to_latent_init, signal_to_latent
from tabascal.timing import measure_runtime

import xarray as xr

from typing import Tuple, Dict, Callable


def read_true_rfi_A(sim_zarr_path: str, data_col: str, times: Array) -> Array:

    xds = xr.open_zarr(sim_zarr_path)
    interp = lambda _rfi_A: jnp.interp(times, xds.time_fine.data, _rfi_A)

    data_type = get_observation_data_type(data_col)

    if data_type["rfi"]: 
        # xds.rfi_tle_sat_A is shape (n_rfi, n_time_fine, n_ant, n_freq)
        # rfi_A_fine is shape (n_rfi, n_ant, n_freq, n_time_fine)
        rfi_A_fine = jnp.transpose(jnp.array(xds.rfi_tle_sat_A.data.compute()), (0, 2, 3, 1))
        # rfi_A is shape (n_rfi, n_ant, n_freq, n_time)
        rfi_A = vmap(vmap(vmap(interp)))(rfi_A_fine)

        return rfi_A
    else:
        return jnp.zeros((xds.tle_sat_src.data[0], xds.n_ant, xds.n_freq, xds.n_time), dtype=complex)


def compute_real_space_gp_params(gp_l: float, gp_var: float, times: Array, times_fine: Array) -> Tuple[int, Array, Array]:

    gp_times = get_times(times, gp_l)
    n_gp_times = len(gp_times)

    resample_op = resampling_kernel(
        gp_times,
        times_fine,
        gp_var,
        gp_l,
        1e-8,
    )

    return n_gp_times, gp_times, resample_op



# def estimate_rfi_A(fringe_freqs):

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


class RealRFI(Component):

    required_inputs = {}  # No inputs needed
    output_shapes = {
        "rfi_A": ("n_rfi", "n_ant", "n_freq", "n_time_fine"),
    }

    # Add parameter specifications
    parameter_shapes = {
        "rfi_r_induce_base": ("n_rfi", "n_ant", "n_freq", "n_rfi_time"),
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
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def build_set_params(self) -> Callable:
        n_rfi = self.n_rfi
        n_ant = self.n_ant
        n_freq = self.n_freq
        n_rfi_times = self.n_rfi_times

        def set_params(params: Dict) -> Dict:

            params["rfi_r_induce_base"] = standard_normal(
                "rfi_r_induce_base", (n_rfi, n_ant, n_freq, n_rfi_times)
            )

            return params

        return set_params

    def build_forward(self) -> Callable:
        interp = lambda _rfi_A_induce: jnp.dot(self.resample_rfi, _rfi_A_induce)
        self.interp_multi = vmap(vmap(vmap(interp)))

        def forward(params: Dict, state: Dict) -> Dict:

            rfi_A_induce_base = params["rfi_r_induce_base"]

            rfi_A_induce = self.forward_transform(rfi_A_induce_base, self.L_rfi_A, self.mu_rfi_A)

            rfi_A = self.interp_multi(rfi_A_induce)

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

        self.n_rfi_times, self.rfi_times, sefl.resample_rfi = compute_real_space_gp_params(gp_l, gp_var, self.times, self.times_fine)

    def _set_outputs(self):

        self.state_outputs = {
            "rfi_A": jnp.zeros(
                (self.n_rfi, self.n_ant, self.n_freq, self.n_time_fine), dtype=complex
            ),
        }

    def _compute_prior_params(self):

        self.L_rfi_A = cholesky(self.rfi_times, self.gp_var, self.gp_l, 1e-8)
        self.mu_rfi_A = jnp.zeros(
            (self.n_rfi, self.n_ant, self.n_freq, self.n_rfi_times)
        )

    def _compute_true_params(self, sim_zarr_path: str, data_col: str):

        self.true_rfi_A_induce = read_true_rfi_A(sim_zarr_path, data_col, self.rfi_times).real

        self.true_rfi_A_induce_base = self.inv_transform(
            self.true_rfi_A_induce, self.L_rfi_A, self.mu_rfi_A
        )

    def forward_transform(self, base_params: Array, L: Array, mu: Array) -> Array:

        affine_same_scale = lambda _base_params, _mu: affine_transform_full(_base_params, L, _mu)

        params = vmap(vmap(vmap(affine_same_scale)))(base_params, mu)

        return params

    def inv_transform(self, params: Array, L: Array, mu: Array) -> Array:

        inv_affine_same_scale = lambda centered_params: jnp.linalg.solve(L, centred_params)

        base_params = vmap(vmap(vmap(inv_affine_same_scale)))(params - mu)

        return base_params

    def _compute_init_params(self, init_type: str):

        if init_type == "prior":
            print("Using prior mean for rfi_A")
            self.init_rfi_A_induce = self.mu_rfi_A
        elif init_type = "zeros":
            print("Using for zeros for rfi_A")
            self.init_rfi_A_induce = jnp.zeros_like(self.mu_rfi_A)
        elif init_type = "ones":
            print("Using for ones for rfi_A")
            self.init_rfi_A_induce = jnp.ones_like(self.mu_rfi_A)
        elif init_type == "truth":
            print("Using truth for rfi_A")
            self.init_rfi_A_induce = self.true_rfi_A_induce
        elif init_type == "sample":
            print("Drawing sample from prior for rfi_A")
            prior_sample = random.normal(
                random.PRNGKey(1),
                (self.n_rfi, self.n_ant, self.n_freq, self.n_rfi_times),
            )
            self.init_rfi_A_induce = self.forward_transform(
                prior_sample, self.L_rfi_A, self.mu_rfi_A
            )
        else:
            raise ValueError(f"Provided init type: {init_type} is not valid. Choose from (prior, zeros, ones, truth, sample).")

        self.init_rfi_A_induce_base = self.inv_transform(
            self.init_rfi_A_induce, self.L_rfi_A, self.mu_rfi_A
        )

        self.init_params = {
            "rfi_r_induce": self.init_rfi_A_induce,
        }
        self.init_params_base = {
            "rfi_r_induce_base": self.init_rfi_A_induce_base,
        }

    def _validate_dimensions(self):
        """Ensure all setup operations completed successfully"""

        rfi_shape = (self.n_rfi, self.n_ant, self.n_freq, self.n_rfi_times)

        assert_attr_shape(self, "mu_rfi_A", rfi_shape)
        assert_attr_shape(self, "L_rfi_A", (self.n_rfi_times, self.n_rfi_times))
        assert_attr_shape(self, "init_rfi_A_induce", rfi_shape)
        assert_attr_shape(self, "init_rfi_A_induce_base", rfi_shape)


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
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def build_set_params(self) -> Callable:
        n_rfi = self.n_rfi
        n_ant = self.n_ant
        n_freq = self.n_freq
        n_rfi_times = self.n_rfi_times

        def set_params(params: Dict) -> Dict:

            params["rfi_r_induce_base"] = standard_normal(
                "rfi_r_induce_base", (n_rfi, n_ant, n_freq, n_rfi_times)
            )
            params["rfi_i_induce_base"] = standard_normal(
                "rfi_i_induce_base", (n_rfi, n_ant, n_freq, n_rfi_times)
            )

            return params

        return set_params

    def build_forward(self) -> Callable:
        interp = lambda _rfi_A_induce: jnp.dot(self.resample_rfi, _rfi_A_induce)
        self.interp_multi = vmap(vmap(vmap(interp)))

        def forward(params: Dict, state: Dict) -> Dict:

            rfi_A_induce_base = (
                params["rfi_r_induce_base"] + 1.0j * params["rfi_i_induce_base"]
            )

            rfi_A_induce = self.forward_transform(rfi_A_induce_base, self.L_rfi_A, self.mu_rfi_A)

            rfi_A = self.interp_multi(rfi_A_induce)

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

        self.n_rfi_times, self.rfi_times, sefl.resample_rfi = compute_real_space_gp_params(gp_l, gp_var, self.times, self.times_fine)

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

    def _compute_true_params(self, sim_zarr_path, data_col):

        self.true_rfi_A_induce = read_true_rfi_A(sim_zarr_path, data_col, self.rfi_times)

        self.true_rfi_A_induce_base = self.inv_transform(
            self.true_rfi_A_induce, self.L_rfi_A, self.mu_rfi_A
        )

    def forward_transform(self, base_params: Array, L: Array, mu: Array) -> Array:

        affine_same_scale = lambda _base_params, _mu: affine_transform_full(_base_params, L, _mu)

        params = vmap(vmap(vmap(affine_same_scale)))(base_params, mu)

        return params

    def inv_transform(self, params: Array, L: Array, mu: Array) -> Array:

        inv_affine_same_scale = lambda centered_params: jnp.linalg.solve(L, centred_params)

        base_params = vmap(vmap(vmap(inv_affine_same_scale)))(params - mu)

        return base_params

    def _compute_init_params(self, init_type):

        if init_type == "prior":
            print("Using prior mean for rfi_A")
            self.init_rfi_A_induce = self.mu_rfi_A
        elif init_type = "zeros":
            print("Using for zeros for rfi_A")
            self.init_rfi_A_induce = jnp.zeros_like(self.mu_rfi_A)
        elif init_type = "ones":
            print("Using for ones for rfi_A")
            self.init_rfi_A_induce = jnp.ones_like(self.mu_rfi_A)
        elif init_type == "truth":
            print("Using truth for rfi_A")
            self.init_rfi_A_induce = self.true_rfi_A_induce
        elif init_type == "sample":
            print("Drawing sample from prior for rfi_A")
            prior_sample = random.normal(
                random.PRNGKey(1),
                (self.n_rfi, self.n_ant, self.n_freq, self.n_rfi_times),
            )
            self.init_rfi_A_induce = self.forward_transform(
                prior_sample, self.L_rfi_A, self.mu_rfi_A
            )
        else:
            raise ValueError(f"Provided init type: {init_type} is not valid. Choose from (prior, zeros, ones, truth, sample).")

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


##############################################################################################################


class FourierGPRFI(Component):

    required_inputs = {}  # No inputs needed
    output_shapes = {
        "rfi_A": ("n_rfi", "n_ant", "n_freq_fine", "n_time_fine"),
    }

    # Add parameter specifications
    parameter_shapes = {
        "rfi_r_induce_base": ("n_rfi", "n_ant", "n_freq_fine", "n_rfi_time"),
        "rfi_i_induce_base": ("n_rfi", "n_ant", "n_freq_fine", "n_rfi_time"),
    }

    def setup(self, config):
        """All validation and error-prone operations here"""
        try:
            # Store only what's needed for forward computation
            self.n_rfi = config.n_rfi
            self.n_ant = config.n_ant
            self.n_freq = config.n_freq
            self.n_time = config.n_time
            self.n_int_time = config.n_int_time
            self.n_int_freq = config.args["rfi"]["freq_int_samples"]
            self.n_freq_fine = self.n_freq * self.n_int_freq
            self.n_time_fine = self.n_time * self.n_int_time
            # self.n_time_fine = config.n_time_fine
            # self.n_freq_fine = config.n_freq_fine
            self.freqs = config.freqs
            self.times = config.times
            self.vis_obs = config.vis_obs

            self.p0 = config.args["rfi"]["pow_spec"]["p0"]
            self.k0s = config.args["rfi"]["pow_spec"]["k0s"]
            self.gammas = config.args["rfi"]["pow_spec"]["gammas"]
            self.pk_cutoff = config.args["rfi"]["pow_spec"]["cutoff"]
            self.time_pad_factor = config.args["rfi"]["time_pad_factor"]
            self.freq_pad_factor = config.args["rfi"]["freq_pad_factor"]

            self.xs = [self.freqs, self.times]
            self.pad_factors = [self.freq_pad_factor, self.time_pad_factor]
            self.ss_factors = [self.n_int_freq, self.n_int_time]

            # Do expensive setup operations once
            self._compute_gp_params()
            self._compute_prior_params(config.args["rfi"]["mean"], config.vis_obs, config.args["rfi"]["est"])
            self._set_outputs()

            if config.args["plots"]["truth"] or config.args["rfi"]["init"] == "truth":
                self._compute_true_params(
                    config.args["data"]["zarr_path"], config.args["data"]["data_col"]
                )

            # if config.args["rfi"]["init"] == "est":
            #     self._estimate_params(config.fringe_freqs)

            self._compute_init_params(config.args["rfi"]["init"], config.args["rfi"]["est"])

            # Validate dimensions
            self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def build_set_params(self):
        n_rfi = self.n_rfi
        n_ant = self.n_ant
        n_k_freq_rfi = self.n_k_freq_rfi
        n_k_time_rfi = self.n_k_time_rfi

        def set_params(params):

            params["rfi_k_r_base"] = standard_normal(
                "rfi_k_r_base", (n_rfi, n_ant, n_k_freq_rfi, n_k_time_rfi)
            )
            params["rfi_k_i_base"] = standard_normal(
                "rfi_k_i_base", (n_rfi, n_ant, n_k_freq_rfi, n_k_time_rfi)
            )

            return params

        return set_params

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        # Pre-compute everything possible
        sigma_rfi_k = self.sigma_rfi_k
        mu_rfi_k = self.mu_rfi_k
        forward_transform = self.forward_transform
        pads = self.pads
        ss_idxs = self.ss_idxs

        def forward(params: dict, state: dict):
            # Pure JAX operations only

            rfi_k_A_base = params["rfi_k_r_base"] + 1.0j * params["rfi_k_i_base"]

            rfi_k_A = forward_transform(rfi_k_A_base, sigma_rfi_k, mu_rfi_k)

            rfi_A = vmap(vmap(latent_to_signal, (0, None, None), 0), (1, None, None), 1)(
                rfi_k_A, pads, ss_idxs
            )

            state = {**state, "rfi_A": rfi_A}

            return state

        return forward

    def validate_and_test(self):
        """Call this before using in JIT context"""
        pass

    @measure_runtime
    def _compute_gp_params(self):

        # if self.gp_var is None:
        #     self.gp_var = jnp.max(jnp.abs(self.vis_obs))

        # print(f"Using RFI var : {self.gp_var:.3e} Jy")

        # if self.gp_l is None:
        #     self.gp_l = 1.0

        self.pk, self.ks, self.pads, self.ss_idxs = latent_to_signal_init(
            self.xs,
            self.pad_factors,
            self.ss_factors,
            self.p0,
            self.k0s,
            self.gammas,
            self.pk_cutoff,
        )

        # Pre-compute slicing indices for JIT-compatible latent extraction
        self.latent_idxs, _ = signal_to_latent_init(
            self.xs,
            self.pad_factors,
            self.p0,
            self.k0s,
            self.gammas,
            self.pk_cutoff,
        )

        # JIT-compiled function for efficient latent extraction
        self.signal_to_latent = lambda rfi_A: signal_to_latent(
            rfi_A,
            self.pad_factors,
            self.latent_idxs,
        )

        xs = [self.freqs, self.times]
        dxs = tuple([float(jnp.diff(x[:2])[0]) if len(x) > 1 else 1 for x in xs])

        print("\nRFI specs")
        print(f"(d_freq, d_time): ({dxs[0]:.3e}, {dxs[1]:.3e})")
        print(f"(n_freq, n_time): ({self.n_freq}, {self.n_time})")
        print(f"(n_k_fq, n_k_tm): {self.pk.shape}")

        self.n_k_freq_rfi, self.n_k_time_rfi = self.pk.shape
        self.sigma_rfi_k = jnp.sqrt(self.pk / self.pk.size)[None, None, :, :]

    def _set_outputs(self):

        self.state_outputs = {
            "rfi_A": jnp.zeros(
                (self.n_rfi, self.n_ant, self.n_freq_fine, self.n_time_fine),
                dtype=complex,
            ),
        }

    def _compute_data_est(self, vis_obs):

        est_rfi_k = self.signal_to_latent(jnp.sqrt(jnp.max(jnp.abs(vis_obs), axis=0)))[None, None, :, :] * jnp.ones((self.n_rfi, self.n_ant, 1, 1)) / self.n_rfi

        return est_rfi_k

    def _compute_prior_params(self, prior_type, vis_obs, est_path):

        if prior_type == "data":
            print("Using data for RFI prior mean")
            self.mu_rfi_k = self._compute_data_est(vis_obs)
        elif prior_type == "est": 
            print("Using provided estimate for rfi_k")
            self.mu_rfi_k = self._read_estimate(est_path)
        elif prior_type == "zeros":
            print("Using zeros for RFI prior mean")
            self.mu_rfi_k = jnp.zeros(
                (self.n_rfi, self.n_ant, self.n_k_freq_rfi, self.n_k_time_rfi), dtype=complex
            )
        else:
            raise ValueError(f"Provided prior type: {prior_type} is not valid. Choose from (data, zeros).")

    def forward_transform(self, base_params, sigma, mu):

        params = sigma * base_params + mu

        return params

    def inv_transform(self, params, sigma, mu):

        base_params = (params - mu) / sigma

        return base_params

    def _compute_true_params(self, zarr_path, data_col):

        xds = xr.open_zarr(zarr_path)

        data_type = get_observation_data_type(data_col)

        rfi_A = jnp.transpose(
            vmap(
                vmap(vmap(jnp.interp, (None, None, 0), 0), (None, None, 2), 2),
                (None, None, 3),
                3,
            )(
                self.times,
                xds.time_fine.data,
                xds.rfi_tle_sat_A.data.compute(),
            ),
            (0, 2, 3, 1),
        )

        self.true_rfi_k_A = vmap(vmap(self.signal_to_latent, (0,), 0), (1,), 1)(rfi_A)

        self.true_rfi_k_A_base = self.inv_transform(
            self.true_rfi_k_A, self.sigma_rfi_k, self.mu_rfi_k
        )

    def _read_estimate(self, est_path):

        from numpy import load

        est_rfi_A = jnp.max(jnp.sqrt(jnp.abs(jnp.array(load(est_path)[:self.n_rfi]))), axis=-1)[:, None, None, :] * jnp.ones((1, self.n_ant, self.n_freq, 1))

        return vmap(vmap(self.signal_to_latent, (0,), 0), (1,), 1)(est_rfi_A)

    def _compute_init_params(self, init_type, est_path):

        if init_type == "prior":
            print("Using prior mean for rfi_k")
            self.init_rfi_k = self.mu_rfi_k
        elif init_type == "est": 
            print("Using provided estimate for rfi_k")
            self.init_rfi_k = self._read_estimate(est_path)
        elif init_type == "truth":
            print("Using truth for rfi_A")
            self.init_rfi_k = self.true_rfi_k_A
        elif init_type == "zeros":
            print("Using zeros for rfi_k")
            ones = jnp.zeros((self.n_freq, self.n_time), dtype=complex)
            self.init_rfi_k = self.signal_to_latent(ones)[None,None,:,:] * jnp.ones((self.n_rfi, self.n_ant, 1, 1))
        elif init_type == "ones":
            print("Using ones for rfi_k")
            ones = jnp.ones((self.n_freq, self.n_time), dtype=complex)
            self.init_rfi_k = self.signal_to_latent(ones)[None,None,:,:] * jnp.ones((self.n_rfi, self.n_ant, 1, 1))
        elif init_type == "sample":
            print("Drawing sample from prior for rfi_k")
            prior_sample = random.normal(
                random.PRNGKey(1),
                (self.n_rfi, self.n_ant, self.n_k_freq_rfi, self.n_k_time_rfi),
                dtype=complex,
            )
            self.init_rfi_k = self.forward_transform(
                prior_sample, self.sigma_rfi_k, self.mu_rfi_k
            )
        else:
            raise ValueError(f"Provided init type: {init_type} is not valid. Choose from (prior, truth, zeros, ones, sample).")

        self.init_rfi_k_base = self.inv_transform(
            self.init_rfi_k, self.sigma_rfi_k, self.mu_rfi_k
        )

        self.init_params = {
            "rfi_k_r": self.init_rfi_k.real,
            "rfi_k_i": self.init_rfi_k.imag,
        }
        self.init_params_base = {
            "rfi_k_r_base": self.init_rfi_k_base.real,
            "rfi_k_i_base": self.init_rfi_k_base.imag,
        }

    def _validate_dimensions(self):
        """Ensure all setup operations completed successfully"""

        rfi_shape = (self.n_rfi, self.n_ant, self.n_k_freq_rfi, self.n_k_time_rfi)

        assert_attr_shape(self, "mu_rfi_k", rfi_shape)
        assert_attr_shape(
            self, "sigma_rfi_k", (1, 1, self.n_k_freq_rfi, self.n_k_time_rfi)
        )
        assert_attr_shape(self, "init_rfi_k", rfi_shape)
        assert_attr_shape(self, "init_rfi_k_base", rfi_shape)


class FourierGPRFIConstAnt(Component):

    required_inputs = {}  # No inputs needed
    output_shapes = {
        "rfi_A": ("n_rfi", "n_ant", "n_freq_fine", "n_time_fine"),
    }

    # Add parameter specifications
    parameter_shapes = {
        "rfi_r_induce_base": ("n_rfi", 1, "n_freq_fine", "n_rfi_time"),
        "rfi_i_induce_base": ("n_rfi", 1, "n_freq_fine", "n_rfi_time"),
    }

    def setup(self, config):
        """All validation and error-prone operations here"""
        try:
            # Store only what's needed for forward computation
            self.n_rfi = config.n_rfi
            self.n_ant = config.n_ant
            self.n_freq = config.n_freq
            self.n_time = config.n_time
            self.n_int_time = config.n_int_time
            self.n_int_freq = config.args["rfi"]["freq_int_samples"]
            self.n_freq_fine = self.n_freq * self.n_int_freq
            self.n_time_fine = self.n_time * self.n_int_time
            # self.n_time_fine = config.n_time_fine
            # self.n_freq_fine = config.n_freq_fine
            self.freqs = config.freqs
            self.times = config.times
            self.vis_obs = config.vis_obs

            self.p0 = config.args["rfi"]["pow_spec"]["p0"]
            self.k0s = config.args["rfi"]["pow_spec"]["k0s"]
            self.gammas = config.args["rfi"]["pow_spec"]["gammas"]
            self.pk_cutoff = config.args["rfi"]["pow_spec"]["cutoff"]
            self.time_pad_factor = config.args["rfi"]["time_pad_factor"]
            self.freq_pad_factor = config.args["rfi"]["freq_pad_factor"]

            self.xs = [self.freqs, self.times]
            self.pad_factors = [self.freq_pad_factor, self.time_pad_factor]
            self.ss_factors = [self.n_int_freq, self.n_int_time]

            # Do expensive setup operations once
            self._compute_gp_params()
            self._compute_prior_params(config.args["rfi"]["mean"], config.vis_obs, config.args["rfi"]["est"])
            self._set_outputs()

            if config.args["plots"]["truth"] or config.args["rfi"]["init"] == "truth":
                self._compute_true_params(
                    config.args["data"]["zarr_path"], config.args["data"]["data_col"]
                )

            # if config.args["rfi"]["init"] == "est":
            #     self._estimate_params(config.fringe_freqs)

            self._compute_init_params(config.args["rfi"]["init"], config.args["rfi"]["est"])

            # Validate dimensions
            self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def build_set_params(self):
        n_rfi = self.n_rfi
        n_ant = self.n_ant
        n_k_freq_rfi = self.n_k_freq_rfi
        n_k_time_rfi = self.n_k_time_rfi

        def set_params(params):

            params["rfi_k_r_base"] = standard_normal(
                "rfi_k_r_base", (n_rfi, n_ant, n_k_freq_rfi, n_k_time_rfi)
            )
            params["rfi_k_i_base"] = standard_normal(
                "rfi_k_i_base", (n_rfi, n_ant, n_k_freq_rfi, n_k_time_rfi)
            )

            return params

        return set_params

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        # Pre-compute everything possible
        sigma_rfi_k = self.sigma_rfi_k
        mu_rfi_k = self.mu_rfi_k
        forward_transform = self.forward_transform
        pads = self.pads
        ss_idxs = self.ss_idxs

        def forward(params: dict, state: dict):
            # Pure JAX operations only

            rfi_k_A_base = params["rfi_k_r_base"] + 1.0j * params["rfi_k_i_base"]

            rfi_k_A = forward_transform(rfi_k_A_base, sigma_rfi_k, mu_rfi_k)

            rfi_A = vmap(vmap(latent_to_signal, (0, None, None), 0), (1, None, None), 1)(
                rfi_k_A, pads, ss_idxs
            )

            state = {**state, "rfi_A": rfi_A}

            return state

        return forward

    def validate_and_test(self):
        """Call this before using in JIT context"""
        pass

    @measure_runtime
    def _compute_gp_params(self):

        # if self.gp_var is None:
        #     self.gp_var = jnp.max(jnp.abs(self.vis_obs))

        # print(f"Using RFI var : {self.gp_var:.3e} Jy")

        # if self.gp_l is None:
        #     self.gp_l = 1.0

        self.pk, self.ks, self.pads, self.ss_idxs = latent_to_signal_init(
            self.xs,
            self.pad_factors,
            self.ss_factors,
            self.p0,
            self.k0s,
            self.gammas,
            self.pk_cutoff,
        )

        # Pre-compute slicing indices for JIT-compatible latent extraction
        self.latent_idxs, _ = signal_to_latent_init(
            self.xs,
            self.pad_factors,
            self.p0,
            self.k0s,
            self.gammas,
            self.pk_cutoff,
        )

        # JIT-compiled function for efficient latent extraction
        self.signal_to_latent = lambda rfi_A: signal_to_latent(
            rfi_A,
            self.pad_factors,
            self.latent_idxs,
        )

        xs = [self.freqs, self.times]
        dxs = tuple([float(jnp.diff(x[:2])[0]) if len(x) > 1 else 1 for x in xs])

        print("\nRFI specs")
        print(f"(d_freq, d_time): ({dxs[0]:.3e}, {dxs[1]:.3e})")
        print(f"(n_freq, n_time): ({self.n_freq}, {self.n_time})")
        print(f"(n_k_fq, n_k_tm): {self.pk.shape}")

        self.n_k_freq_rfi, self.n_k_time_rfi = self.pk.shape
        self.sigma_rfi_k = jnp.sqrt(self.pk / self.pk.size)[None, None, :, :]

    def _set_outputs(self):

        self.state_outputs = {
            "rfi_A": jnp.zeros(
                (self.n_rfi, self.n_ant, self.n_freq_fine, self.n_time_fine),
                dtype=complex,
            ),
        }

    def _compute_data_est(self, vis_obs):

        est_rfi_k = self.signal_to_latent(jnp.sqrt(jnp.max(jnp.abs(vis_obs), axis=0)))[None, None, :, :] * jnp.ones((self.n_rfi, self.n_ant, 1, 1)) / self.n_rfi

        return est_rfi_k

    def _compute_prior_params(self, prior_type, vis_obs, est_path):

        if prior_type == "data":
            print("Using data for RFI prior mean")
            self.mu_rfi_k = self._compute_data_est(vis_obs)
        elif prior_type == "est": 
            print("Using provided estimate for rfi_k")
            self.mu_rfi_k = self._read_estimate(est_path)
        elif prior_type == "zeros":
            print("Using zeros for RFI prior mean")
            self.mu_rfi_k = jnp.zeros(
                (self.n_rfi, self.n_ant, self.n_k_freq_rfi, self.n_k_time_rfi), dtype=complex
            )
        else:
            raise ValueError(f"Provided prior type: {prior_type} is not valid. Choose from (data, zeros).")

    def forward_transform(self, base_params, sigma, mu):

        params = sigma * base_params + mu

        return params

    def inv_transform(self, params, sigma, mu):

        base_params = (params - mu) / sigma

        return base_params

    def _compute_true_params(self, zarr_path, data_col):

        xds = xr.open_zarr(zarr_path)

        data_type = get_observation_data_type(data_col)

        rfi_A = jnp.transpose(
            vmap(
                vmap(vmap(jnp.interp, (None, None, 0), 0), (None, None, 2), 2),
                (None, None, 3),
                3,
            )(
                self.times,
                xds.time_fine.data,
                xds.rfi_tle_sat_A.data.compute(),
            ),
            (0, 2, 3, 1),
        )

        self.true_rfi_k_A = vmap(vmap(self.signal_to_latent, (0,), 0), (1,), 1)(rfi_A)

        self.true_rfi_k_A_base = self.inv_transform(
            self.true_rfi_k_A, self.sigma_rfi_k, self.mu_rfi_k
        )

    def _read_estimate(self, est_path):

        from numpy import load

        est_rfi_A = jnp.max(jnp.sqrt(jnp.abs(jnp.array(load(est_path)[:self.n_rfi]))), axis=-1)[:, None, None, :] * jnp.ones((1, self.n_ant, self.n_freq, 1))

        return vmap(vmap(self.signal_to_latent, (0,), 0), (1,), 1)(est_rfi_A)

    def _compute_init_params(self, init_type, est_path):

        if init_type == "prior":
            print("Using prior mean for rfi_k")
            self.init_rfi_k = self.mu_rfi_k
        elif init_type == "est": 
            print("Using provided estimate for rfi_k")
            self.init_rfi_k = self._read_estimate(est_path)
        elif init_type == "truth":
            print("Using truth for rfi_A")
            self.init_rfi_k = self.true_rfi_k_A
        elif init_type == "zeros":
            print("Using zeros for rfi_k")
            ones = jnp.zeros((self.n_freq, self.n_time), dtype=complex)
            self.init_rfi_k = self.signal_to_latent(ones)[None,None,:,:] * jnp.ones((self.n_rfi, self.n_ant, 1, 1))
        elif init_type == "ones":
            print("Using ones for rfi_k")
            ones = jnp.ones((self.n_freq, self.n_time), dtype=complex)
            self.init_rfi_k = self.signal_to_latent(ones)[None,None,:,:] * jnp.ones((self.n_rfi, self.n_ant, 1, 1))
        elif init_type == "sample":
            print("Drawing sample from prior for rfi_k")
            prior_sample = random.normal(
                random.PRNGKey(1),
                (self.n_rfi, self.n_ant, self.n_k_freq_rfi, self.n_k_time_rfi),
                dtype=complex,
            )
            self.init_rfi_k = self.forward_transform(
                prior_sample, self.sigma_rfi_k, self.mu_rfi_k
            )
        else:
            raise ValueError(f"Provided init type: {init_type} is not valid. Choose from (prior, truth, zeros, ones, sample).")

        self.init_rfi_k_base = self.inv_transform(
            self.init_rfi_k, self.sigma_rfi_k, self.mu_rfi_k
        )

        self.init_params = {
            "rfi_k_r": self.init_rfi_k.real,
            "rfi_k_i": self.init_rfi_k.imag,
        }
        self.init_params_base = {
            "rfi_k_r_base": self.init_rfi_k_base.real,
            "rfi_k_i_base": self.init_rfi_k_base.imag,
        }

    def _validate_dimensions(self):
        """Ensure all setup operations completed successfully"""

        rfi_shape = (self.n_rfi, self.n_ant, self.n_k_freq_rfi, self.n_k_time_rfi)

        assert_attr_shape(self, "mu_rfi_k", rfi_shape)
        assert_attr_shape(
            self, "sigma_rfi_k", (1, 1, self.n_k_freq_rfi, self.n_k_time_rfi)
        )
        assert_attr_shape(self, "init_rfi_k", rfi_shape)
        assert_attr_shape(self, "init_rfi_k_base", rfi_shape)

