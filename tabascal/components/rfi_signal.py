from jax import vmap, random, Array
import jax.numpy as jnp

from tabascal.components import Component, assert_attr_shape
from tabascal.config import TabConfig
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

    # resample op is shape (n_time_fine, n_gp_time)
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


def rfi_signal_config_validation(rfi_config: Dict, vis_obs: Array, freqs: Array, times: Array) -> Dict:
    """Validate and set defaults of BaseGPRFI class parameters in the configuration file.

    Parameters
    ----------
    rfi_config : Dict
        RFI configuration dictionary

    Returns
    -------
    Dict
        Validated configuration dictionary with defaults set.

    Raises
    ------
    ValueError
        Raised when an invalid input is provided for one fo the configuration parameters.
    """

    extent = lambda x: float(jnp.max(x) - jnp.min(x))

    try:
        r_seed = rfi_config["r_seed"]
        gp_var = rfi_config["var"]
        gp_freq_l = rfi_config["corr_freq"]
        gp_time_l = rfi_config["corr_time"]
    except Exception as e:
        raise ValueError(f"RFI signal configuration validation failed.")

    if not r_seed: # Set Default
        rfi_config["r_seed"] = 1
    elif isinstance(r_seed, int):
        pass
    else:
        raise ValueError(f"Config parameter (rfi:\n\tr_seed: {r_seed}) is not of type int.")

    if not gp_var: # Set Default
        est_gp_var = float(jnp.max(jnp.abs(vis_obs)))
        rfi_config["var"] = est_gp_var
        print(f"Using RFI var : {est_gp_var:.3e} Jy")
    elif isinstance(gp_var, (float, int)):
        rfi_config["var"] = float(gp_var)
    else:
        raise ValueError(f"Config parameter (rfi:\n\tvar: {gp_var}) is not of type float or int.")
    
    if not gp_freq_l: # Set Default
        est_gp_freq_l = extent(freqs) / 2
        rfi_config["corr_freq"] = est_gp_freq_l
        print(f"Using RFI corr_freq : {est_gp_freq_l:.3e} Hz")
    elif isinstance(gp_freq_l, (float, int)):
        rfi_config["corr_freq"] = float(gp_freq_l)
    else:
        raise ValueError(f"Config parameter (rfi:\n\tcorr_freq: {gp_freq_l}) is not of type float or int.")
    
    if not gp_time_l: # Set Default
        est_gp_time_l = extent(times) / 2
        rfi_config["corr_freq"] = est_gp_time_l
        print(f"Using RFI corr_time : {est_gp_time_l:.3e} s")
    elif isinstance(gp_time_l, (float, int)):
        rfi_config["corr_time"] = float(gp_time_l)
    else:
        raise ValueError(f"Config parameter (rfi:\n\tcorr_time: {gp_time_l}) is not of type float or int.")    

    return rfi_config



class BaseGPRFI(Component):

    # The required state parameter needed in the forward model for this component to function
    required_inputs = {}  # No inputs needed

    # The additional state parameters included in the forward model from this component
    output_shapes = {
        "rfi_A": ("n_rfi", "n_ant", "n_freq_fine", "n_time_fine"),
    }

    # The base parameter shapes used to produce the output parameters
    parameter_shapes = {}

    def setup(self, tab_config: TabConfig):

        # Validate config and set defaults
        rfi_config = rfi_signal_config_validation(
            tab_config.args["rfi"], tab_config.vis_obs, tab_config.freqs, tab_config.times)

        # Random seed used for random sampling such as initial parameters drawn from the prior
        self.r_seed = rfi_config["r_seed"]

        # Basic shape parameters 
        self.n_rfi = tab_config.n_rfi
        self.n_ant = tab_config.n_ant
        self.n_freq = tab_config.n_freq
        self.n_freq_fine = tab_config.n_freq_fine
        self.n_int_freq = tab_config.n_int_freq
        self.n_time = tab_config.n_time
        self.n_time_fine = tab_config.n_time_fine
        self.n_int_time = tab_config.n_int_time

        # Domain arrays needed to calculate Gaussian process parameters
        self.freqs = tab_config.freqs
        self.freqs_fine = tab_config.freqs_fine
        self.chan_width = tab_config.chan_width
        self.times = tab_config.times
        self.times_fine = tab_config.times_fine
        self.int_time = tab_config.int_time

        self.gp_var = rfi_config["var"]
        self.corr_freq = rfi_config["corr_freq"]
        self.corr_time = rfi_config["corr_time"]


    def _set_outputs(self):

        self.state_outputs = {
            "rfi_A": jnp.zeros(
                (self.n_rfi, self.n_ant, self.n_freq_fine, self.n_time_fine), dtype=complex
            ),
        }

    def _compute_gp_params(self):
        pass

    def _compute_prior_params(self):
        pass

    def _compute_init_params(self):
        pass


class RealRFI(BaseGPRFI):

    required_inputs = {}  # No inputs needed
    output_shapes = {
        "rfi_A": ("n_rfi", "n_ant", "n_freq", "n_time_fine"),
    }

    # Add parameter specifications
    parameter_shapes = {
        "rfi_r_induce_base": ("n_rfi", "n_ant", "n_freq", "n_rfi_time"),
    }

    def setup(self, tab_config: TabConfig):
        """All validation and error-prone operations here"""
        try:
            super().setup(tab_config)
            self.vis_obs = tab_config.vis_obs

            # Do expensive setup operations once
            self._compute_gp_params()
            self._compute_prior_params()
            self._set_outputs()

            if tab_config.args["plots"]["truth"] or tab_config.args["rfi"]["init"] == "truth":
                self._compute_true_params(
                    tab_config.args["data"]["zarr_path"], tab_config.args["data"]["data_col"]
                )

            # if config.args["rfi"]["init"] == "est":
            #     self._estimate_params(tab_config.fringe_freqs)

            self._compute_init_params(tab_config.args["rfi"]["init"])

            # Validate dimensions
            self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def build_set_params(self) -> Callable:

        def set_params(params: Dict) -> Dict:

            params["rfi_r_induce_base"] = standard_normal(
                "rfi_r_induce_base", (self.n_rfi, self.n_ant, self.n_freq, self.n_rfi_times)
            )

            return params

        return set_params

    def build_forward(self) -> Callable:
        self.interp = lambda _rfi_A_induce: jnp.einsum("ij,safj->safi", self.resample_rfi, _rfi_A_induce)

        def forward(params: Dict, state: Dict) -> Dict:

            rfi_A_induce_base = params["rfi_r_induce_base"]
            rfi_A_induce = self.forward_transform(rfi_A_induce_base)
            rfi_A = self.interp(rfi_A_induce)

            state = {**state, "rfi_A": rfi_A}

            return state

        return forward

    def validate_and_test(self):
        """Call this before using in JIT context"""
        pass

    def _compute_gp_params(self):

        self.n_rfi_times, self.rfi_times, self.resample_rfi = compute_real_space_gp_params(self.corr_time, self.gp_var, self.times, self.times_fine)

    def _compute_prior_params(self):

        self.L_rfi_A = cholesky(self.rfi_times, self.gp_var, self.corr_time, 1e-8)
        self.mu_rfi_A = jnp.zeros(
            (self.n_rfi, self.n_ant, self.n_freq, self.n_rfi_times)
        )

    def _compute_true_params(self, sim_zarr_path: str, data_col: str):

        self.true_rfi_A_induce = read_true_rfi_A(sim_zarr_path, data_col, self.rfi_times).real
        self.true_rfi_A_induce_base = self.inv_transform(self.true_rfi_A_induce)

    def forward_transform(self, base_params: Array) -> Array:

        affine_same_scale = lambda _base_params, _mu: affine_transform_full(_base_params, self.L_rfi_A, _mu)
        params = vmap(vmap(vmap(affine_same_scale)))(base_params, self.mu_rfi_A)

        return params

    def inv_transform(self, params: Array) -> Array:

        inv_affine_same_scale = lambda centred_params: jnp.linalg.solve(self.L_rfi_A, centred_params)
        base_params = vmap(vmap(vmap(inv_affine_same_scale)))(params - self.mu_rfi_A)

        return base_params

    def _compute_init_params(self, init_type: str):

        if init_type == "prior":
            print("Using prior mean for rfi_A")
            self.init_rfi_A_induce = self.mu_rfi_A
        elif init_type in ["zeros", 0] :
            print("Using for zeros for rfi_A")
            self.init_rfi_A_induce = jnp.zeros_like(self.mu_rfi_A)
        elif init_type == "ones":
            print("Using for ones for rfi_A")
            self.init_rfi_A_induce = jnp.ones_like(self.mu_rfi_A)
        elif init_type == "truth":
            print("Using truth for rfi_A")
            self.init_rfi_A_induce = self.true_rfi_A_induce
        elif init_type == "sample":
            print("Drawing sample from prior for rfi_A")
            base_sample = random.normal(
                random.PRNGKey(self.r_seed),
                (self.n_rfi, self.n_ant, self.n_freq, self.n_rfi_times),
            )
            self.init_rfi_A_induce = self.forward_transform(base_sample)
        else:
            raise ValueError(f"Provided init type: {init_type} is not valid. Choose from (prior, zeros, ones, truth, sample).")

        self.init_rfi_A_induce_base = self.inv_transform(self.init_rfi_A_induce)

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


class ComplexRFI(BaseGPRFI):

    required_inputs = {}  # No inputs needed
    output_shapes = {
        "rfi_A": ("n_rfi", "n_ant", "n_freq", "n_time_fine"),
    }

    # Add parameter specifications
    parameter_shapes = {
        "rfi_r_induce_base": ("n_rfi", "n_ant", "n_freq", "n_rfi_time"),
        "rfi_i_induce_base": ("n_rfi", "n_ant", "n_freq", "n_rfi_time"),
    }

    def setup(self, tab_config):
        """All validation and error-prone operations here"""
        try:
            super().setup(tab_config)
            self.vis_obs = tab_config.vis_obs

            # Do expensive setup operations once
            self._compute_gp_params()
            self._compute_prior_params()
            self._set_outputs()

            if tab_config.args["plots"]["truth"] or tab_config.args["rfi"]["init"] == "truth":
                self._compute_true_params(
                    tab_config.args["data"]["zarr_path"], tab_config.args["data"]["data_col"]
                )

            # if tab_config.args["rfi"]["init"] == "est":
            #     self._estimate_params(tab_config.fringe_freqs)

            self._compute_init_params(tab_config.args["rfi"]["init"])

            # Validate dimensions
            self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def build_set_params(self) -> Callable:

        def set_params(params: Dict) -> Dict:

            shape = (self.n_rfi, self.n_ant, self.n_freq, self.n_rfi_times)

            params["rfi_r_induce_base"] = standard_normal(
                "rfi_r_induce_base", shape
            )
            params["rfi_i_induce_base"] = standard_normal(
                "rfi_i_induce_base", shape
            )

            return params

        return set_params

    def build_forward(self) -> Callable:
        self.interp = lambda _rfi_A_induce: jnp.einsum("ij,safj->safi", self.resample_rfi, _rfi_A_induce)

        def forward(params: Dict, state: Dict) -> Dict:

            rfi_A_induce_base = (
                params["rfi_r_induce_base"] + 1.0j * params["rfi_i_induce_base"]
            )
            rfi_A_induce = self.forward_transform(rfi_A_induce_base)
            rfi_A = self.interp(rfi_A_induce)

            state = {**state, "rfi_A": rfi_A}

            return state

        return forward

    def validate_and_test(self):
        """Call this before using in JIT context"""
        pass

    def _compute_gp_params(self):

        self.n_rfi_times, self.rfi_times, self.resample_rfi = compute_real_space_gp_params(self.corr_time, self.gp_var, self.times, self.times_fine)

    def _compute_prior_params(self):

        self.L_rfi_A = cholesky(self.rfi_times, self.gp_var, self.corr_time, 1e-8)
        self.mu_rfi_A = jnp.zeros(
            (self.n_rfi, self.n_ant, self.n_freq, self.n_rfi_times), dtype=complex
        )

    def _compute_true_params(self, sim_zarr_path, data_col):

        self.true_rfi_A_induce = read_true_rfi_A(sim_zarr_path, data_col, self.rfi_times)
        self.true_rfi_A_induce_base = self.inv_transform(self.true_rfi_A_induce)

    def forward_transform(self, base_params: Array) -> Array:

        affine_same_scale = lambda _base_params, _mu: affine_transform_full(_base_params, self.L_rfi_A, _mu)
        params = vmap(vmap(vmap(affine_same_scale)))(base_params, self.mu_rfi_A)

        return params

    def inv_transform(self, params: Array) -> Array:

        inv_affine_same_scale = lambda centred_params: jnp.linalg.solve(self.L_rfi_A, centred_params)
        base_params = vmap(vmap(vmap(inv_affine_same_scale)))(params - self.mu_rfi_A)

        return base_params

    def _compute_init_params(self, init_type):

        if init_type == "prior":
            print("Using prior mean for rfi_A")
            self.init_rfi_A_induce = self.mu_rfi_A
        elif init_type in ["zeros", 0]:
            print("Using for zeros for rfi_A")
            self.init_rfi_A_induce = jnp.zeros_like(self.mu_rfi_A)
        elif init_type in ["ones", 1]:
            print("Using for ones for rfi_A")
            self.init_rfi_A_induce = jnp.ones_like(self.mu_rfi_A)
        elif init_type == "truth":
            print("Using truth for rfi_A")
            self.init_rfi_A_induce = self.true_rfi_A_induce
        elif init_type == "sample":
            print("Drawing sample from prior for rfi_A")
            base_sample = random.normal(
                random.PRNGKey(self.r_seed),
                (self.n_rfi, self.n_ant, self.n_freq, self.n_rfi_times),
            )
            self.init_rfi_A_induce = self.forward_transform(
                base_sample
            )
        else:
            raise ValueError(f"Provided init type: {init_type} is not valid. Choose from (prior, zeros, ones, truth, sample).")

        self.init_rfi_A_induce_base = self.inv_transform(self.init_rfi_A_induce)

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


class FourierGPRFI(BaseGPRFI):

    required_inputs = {}  # No inputs needed
    output_shapes = {
        "rfi_A": ("n_rfi", "n_ant", "n_freq_fine", "n_time_fine"),
    }

    # Add parameter specifications
    parameter_shapes = {
        "rfi_k_r_base": ("n_rfi", "n_ant", "n_k_freq_rfi", "n_k_time_rfi"),
        "rfi_k_i_base": ("n_rfi", "n_ant", "n_k_freq_rfi", "n_k_time_rfi"),
    }

    def setup(self, tab_config):
        """All validation and error-prone operations here"""
        try:
            super().setup(tab_config)
            self.vis_obs = tab_config.vis_obs

            self.p0 = tab_config.args["rfi"]["pow_spec"]["p0"]
            self.gammas = tab_config.args["rfi"]["pow_spec"]["gammas"]
            self.pk_cutoff = tab_config.args["rfi"]["pow_spec"]["cutoff"]
            self.time_pad_factor = tab_config.args["rfi"]["time_pad_factor"]
            self.freq_pad_factor = tab_config.args["rfi"]["freq_pad_factor"]

            # Do expensive setup operations once
            self._compute_gp_params()
            self._compute_prior_params(tab_config.args["rfi"]["mean"], tab_config.vis_obs, tab_config.args["rfi"]["est"])
            self._set_outputs()

            if tab_config.args["plots"]["truth"] or tab_config.args["rfi"]["init"] == "truth":
                self._compute_true_params(
                    tab_config.args["data"]["zarr_path"], tab_config.args["data"]["data_col"]
                )

            # if tab_config.args["rfi"]["init"] == "est":
            #     self._estimate_params(tab_config.fringe_freqs)

            self._compute_init_params(tab_config.args["rfi"]["init"], tab_config.args["rfi"]["est"])

            # Validate dimensions
            self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def build_set_params(self):

        def set_params(params):

            shape = (self.n_rfi, self.n_ant, self.n_k_freq_rfi, self.n_k_time_rfi)

            params["rfi_k_r_base"] = standard_normal(
                "rfi_k_r_base", shape
            )
            params["rfi_k_i_base"] = standard_normal(
                "rfi_k_i_base", shape
            )

            return params

        return set_params

    def build_forward(self) -> Callable:

        self.latent_forward = vmap(vmap(self.latent_to_signal))

        def forward(params: Dict, state: Dict) -> Dict:

            rfi_k_A_base = params["rfi_k_r_base"] + 1.0j * params["rfi_k_i_base"]
            rfi_k_A = self.forward_transform(rfi_k_A_base)
            rfi_A = self.latent_forward(rfi_k_A)

            state = {**state, "rfi_A": rfi_A}

            return state

        return forward

    def validate_and_test(self):
        """Call this before using in JIT context"""
        pass

    @measure_runtime
    def _compute_gp_params(self):

        ns = [self.n_freq, self.n_time]
        dxs = [self.chan_width, self.int_time]
        pad_factors = [self.freq_pad_factor, self.time_pad_factor]
        k0s = 1 / (2 * jnp.pi * jnp.array([self.corr_freq, self.corr_time]))
        p0 = self.gp_var #* self.n_time * self.n_freq
        gammas = [1e2, 1e2]

        self.pk, self.ks, self.pads, self.ss_idxs = latent_to_signal_init(
            ns,
            dxs,
            pad_factors,
            [self.n_int_freq, self.n_int_time],
            p0,
            k0s,
            gammas,
            self.pk_cutoff,
        )

        self.latent_to_signal = lambda _rfi_k_A: latent_to_signal(
            _rfi_k_A, 
            self.pads, 
            self.ss_idxs
        )

        # Pre-compute slicing indices for JIT-compatible latent extraction
        self.latent_idxs, _ = signal_to_latent_init(
            ns,
            dxs,
            pad_factors,
            p0,
            k0s,
            gammas,
            self.pk_cutoff,
        )

        self.signal_to_latent = lambda _rfi_A: signal_to_latent(
            _rfi_A,
            pad_factors,
            self.latent_idxs,
        )
        
        print("\nRFI specs")
        print(f"(d_freq, d_time): ({dxs[0]:.3e}, {dxs[1]:.3e})")
        print(f"(n_freq, n_time): ({self.n_freq}, {self.n_time})")
        print(f"(n_k_fq, n_k_tm): {self.pk.shape}")

        self.n_k_freq_rfi, self.n_k_time_rfi = self.pk.shape
        self.sigma_rfi_k = jnp.sqrt(self.pk)[None, None, :, :]
        # self.sigma_rfi_k = jnp.sqrt(self.pk / self.pk.size)[None, None, :, :]

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
        elif prior_type in ["zeros", 0]:
            print("Using zeros for RFI prior mean")
            self.mu_rfi_k = jnp.zeros(
                (self.n_rfi, self.n_ant, self.n_k_freq_rfi, self.n_k_time_rfi), dtype=complex
            )
        else:
            raise ValueError(f"Provided prior type: {prior_type} is not valid. Choose from (data, zeros).")

    def forward_transform(self, base_params: Array) -> Array:

        params = self.sigma_rfi_k * base_params + self.mu_rfi_k

        return params

    def inv_transform(self, params: Array) -> Array:

        base_params = (params - self.mu_rfi_k) / self.sigma_rfi_k

        return base_params

    def _compute_true_params(self, sim_zarr_path: str, data_col: str):

        rfi_A = read_true_rfi_A(sim_zarr_path, data_col, self.times)
        self.true_rfi_k_A = vmap(vmap(self.signal_to_latent))(rfi_A)
        self.true_rfi_k_A_base = self.inv_transform(self.true_rfi_k_A)

    def _read_estimate(self, est_path):

        from numpy import load

        est_rfi_A = jnp.max(jnp.sqrt(jnp.abs(jnp.array(load(est_path)[:self.n_rfi]))), axis=-1)[:, None, None, :] * jnp.ones((1, self.n_ant, self.n_freq, 1))
        est_rfi_k_A = vmap(vmap(self.signal_to_latent))(est_rfi_A)

        return est_rfi_k_A

    def _compute_init_params(self, init_type: str, est_path: str):

        if init_type == "prior":
            print("Using prior mean for rfi_k")
            self.init_rfi_k = self.mu_rfi_k
        elif init_type == "est": 
            print("Using provided estimate for rfi_k")
            self.init_rfi_k = self._read_estimate(est_path)
        elif init_type == "truth":
            print("Using truth for rfi_A")
            self.init_rfi_k = self.true_rfi_k_A
        elif init_type in ["zeros", 0]:
            print("Using zeros for rfi_k")
            # zeros_k is shape (1, 1, n_k_freq_rfi, n_k_time_rfi)
            zeros_k = self.signal_to_latent(jnp.zeros((self.n_freq, self.n_time), dtype=complex))[None,None,:,:]
            # init_rfi_k is shape (n_rfi, n_ant, n_k_freq_rfi, n_k_time_rfi)
            self.init_rfi_k = zeros_k * jnp.ones((self.n_rfi, self.n_ant, 1, 1))
        elif init_type in ["ones", 1]:
            print("Using ones for rfi_k")
            # ones_k is shape (1, 1, n_k_freq_rfi, n_k_time_rfi)
            ones_k = self.signal_to_latent(jnp.ones((self.n_freq, self.n_time), dtype=complex))[None,None,:,:]
            # init_rfi_k is shape (n_rfi, n_ant, n_k_freq_rfi, n_k_time_rfi)
            self.init_rfi_k = ones_k * jnp.ones((self.n_rfi, self.n_ant, 1, 1))
        elif init_type == "sample":
            print("Drawing sample from prior for rfi_k")
            base_sample = random.normal(
                random.PRNGKey(self.r_seed),
                (self.n_rfi, self.n_ant, self.n_k_freq_rfi, self.n_k_time_rfi),
                dtype=complex,
            )
            self.init_rfi_k = self.forward_transform(base_sample)
        else:
            raise ValueError(f"Provided init type: {init_type} is not valid. Choose from (prior, truth, zeros, ones, sample).")

        self.init_rfi_k_base = self.inv_transform(self.init_rfi_k)

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


class FourierGPRFIConstAnt(BaseGPRFI):

    required_inputs = {}  # No inputs needed
    output_shapes = {
        "rfi_A": ("n_rfi", "n_ant", "n_freq_fine", "n_time_fine"),
    }

    # Add parameter specifications
    parameter_shapes = {
        "rfi_k_r_base": ("n_rfi", 1, "n_k_freq_rfi", "n_k_time_rfi"),
        "rfi_k_i_base": ("n_rfi", 1, "n_k_freq_rfi", "n_k_time_rfi"),
    }

    def setup(self, tab_config):
        """All validation and error-prone operations here"""
        try:
            super().setup(tab_config)
            self.vis_obs = tab_config.vis_obs

            self.p0 = tab_config.args["rfi"]["pow_spec"]["p0"]
            self.gammas = tab_config.args["rfi"]["pow_spec"]["gammas"]
            self.pk_cutoff = tab_config.args["rfi"]["pow_spec"]["cutoff"]
            self.time_pad_factor = tab_config.args["rfi"]["time_pad_factor"]
            self.freq_pad_factor = tab_config.args["rfi"]["freq_pad_factor"]

            

            # Do expensive setup operations once
            self._compute_gp_params()
            self._compute_prior_params(tab_config.args["rfi"]["mean"], tab_config.vis_obs, tab_config.args["rfi"]["est"])
            self._set_outputs()

            if tab_config.args["plots"]["truth"] or tab_config.args["rfi"]["init"] == "truth":
                self._compute_true_params(
                    tab_config.args["data"]["zarr_path"], tab_config.args["data"]["data_col"]
                )

            # if tab_config.args["rfi"]["init"] == "est":
            #     self._estimate_params(tab_config.fringe_freqs)

            self._compute_init_params(tab_config.args["rfi"]["init"], tab_config.args["rfi"]["est"])

            # Validate dimensions
            self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def build_set_params(self):

        def set_params(params):

            params["rfi_k_r_base"] = standard_normal(
                "rfi_k_r_base", (self.n_rfi, 1, self.n_k_freq_rfi, self.n_k_time_rfi)
            )
            params["rfi_k_i_base"] = standard_normal(
                "rfi_k_i_base", (self.n_rfi, 1, self.n_k_freq_rfi, self.n_k_time_rfi)
            )

            return params

        return set_params

    def build_forward(self):

        def forward(params: Dict, state: Dict) -> Dict:

            # rfi_k_A_base is shape ()
            rfi_k_A_base = params["rfi_k_r_base"] + 1.0j * params["rfi_k_i_base"]

            rfi_k_A = self.forward_transform(rfi_k_A_base)
            rfi_A = vmap(vmap(self.latent_to_signal))(rfi_k_A)
            rfi_A = rfi_A * jnp.ones((self.n_rfi, self.n_ant, self.n_freq_fine, self.n_time_fine))

            state = {**state, "rfi_A": rfi_A}

            return state

        return forward

    def validate_and_test(self):
        """Call this before using in JIT context"""
        pass

    @measure_runtime
    def _compute_gp_params(self):

        ns = [self.n_freq, self.n_time]
        dxs = [self.chan_width, self.int_time]
        pad_factors = [self.freq_pad_factor, self.time_pad_factor]
        k0s = [1.0 / self.corr_freq, 1.0 / self.corr_time]
        p0 = self.gp_var * self.n_time * self.n_freq

        self.pk, self.ks, self.pads, self.ss_idxs = latent_to_signal_init(
            ns,
            dxs,
            pad_factors,
            [self.n_int_freq, self.n_int_time],
            p0,
            k0s,
            self.gammas,
            self.pk_cutoff,
        )

        self.latent_to_signal = lambda _rfi_k_A: latent_to_signal(
            _rfi_k_A, 
            self.pads, 
            self.ss_idxs
        )

        # Pre-compute slicing indices for JIT-compatible latent extraction
        self.latent_idxs, _ = signal_to_latent_init(
            ns,
            dxs,
            pad_factors,
            p0,
            k0s,
            self.gammas,
            self.pk_cutoff,
        )

        self.signal_to_latent = lambda rfi_A: signal_to_latent(
            rfi_A,
            pad_factors,
            self.latent_idxs,
        )
        
        print("\nRFI specs")
        print(f"(d_freq, d_time): ({dxs[0]:.3e}, {dxs[1]:.3e})")
        print(f"(n_freq, n_time): ({self.n_freq}, {self.n_time})")
        print(f"(n_k_fq, n_k_tm): {self.pk.shape}")

        self.n_k_freq_rfi, self.n_k_time_rfi = self.pk.shape
        self.sigma_rfi_k = jnp.sqrt(self.pk / self.pk.size)[None, None, :, :]

    def _compute_data_est(self, vis_obs: Array) -> Array:

        # est_vis_rfi is shape (n_freq, n_time)
        # RFI antenna estimate is sqrt of average visibility amplitude on maximum baseline
        est_rfi_A = jnp.sqrt(jnp.max(jnp.abs(vis_obs / self.n_rfi), axis=0)) 
        # est_rfi_k_A is shape (n_k_freq_rfi, n_k_time_rfi)
        est_rfi_k_A = self.signal_to_latent(est_rfi_A)
        # est_rfi_k_A is now shape (n_rfi, 1, n_k_freq_rfi, n_k_time_rfi)
        est_rfi_k_A = est_rfi_k_A[None, None, :, :] * jnp.ones((self.n_rfi, 1, 1, 1)) 

        return est_rfi_k_A

    def _compute_prior_params(self, prior_type, vis_obs, est_path):

        if prior_type == "data":
            print("Using data for RFI prior mean")
            self.mu_rfi_k = self._compute_data_est(vis_obs)
        elif prior_type == "est": 
            print("Using provided estimate for rfi_k")
            self.mu_rfi_k = self._read_estimate(est_path)
        elif prior_type in ["zeros", 0]:
            print("Using zeros for RFI prior mean")
            self.mu_rfi_k = jnp.zeros(
                (self.n_rfi, 1, self.n_k_freq_rfi, self.n_k_time_rfi), dtype=complex
            )
        else:
            raise ValueError(f"Provided prior type: {prior_type} is not valid. Choose from (data, zeros).")

    def forward_transform(self, base_params: Array) -> Array:

        params = self.sigma_rfi_k * base_params + self.mu_rfi_k

        return params

    def inv_transform(self, params: Array) -> Array:

        base_params = (params - self.mu_rfi_k) / self.sigma_rfi_k

        return base_params

    def _compute_true_params(self, sim_zarr_path: str, data_col: str):

        # true_rfi_A shape goes from (n_rfi, n_ant, n_freq, n_time) -> (n_rfi, 1, n_freq, n_time)
        true_rfi_A = jnp.mean(read_true_rfi_A(sim_zarr_path, data_col, self.times), axis=1, keepdims=True)

        # true_rfi_k_A is shape (n_rfi, 1, n_k_freq_rfi, n_k_time_rfi)
        # Latent prediction is mapped over axes (0, 1)
        self.true_rfi_k_A = vmap(vmap(self.signal_to_latent))(true_rfi_A)

        self.true_rfi_k_A_base = self.inv_transform(self.true_rfi_k_A)

    def _read_estimate(self, est_path):

        from numpy import load

        est_rfi_A = jnp.max(jnp.sqrt(jnp.abs(jnp.array(load(est_path)[:self.n_rfi]))), axis=-1)[:, None, None, :] * jnp.ones((1, 1, self.n_freq, 1))

        return vmap(vmap(self.signal_to_latent))(est_rfi_A)

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
        elif init_type in ["zeros", 0]:
            print("Using zeros for rfi_k")
            ones = jnp.zeros((self.n_freq, self.n_time), dtype=complex)
            self.init_rfi_k = self.signal_to_latent(ones)[None,None,:,:] * jnp.ones((self.n_rfi, self.n_ant, 1, 1))
        elif init_type == "ones":
            print("Using ones for rfi_k")
            ones = jnp.ones((self.n_freq, self.n_time), dtype=complex)
            self.init_rfi_k = self.signal_to_latent(ones)[None,None,:,:] * jnp.ones((self.n_rfi, self.n_ant, 1, 1))
        elif init_type == "sample":
            print("Drawing sample from prior for rfi_k")
            base_sample = random.normal(
                random.PRNGKey(self.r_seed),
                (self.n_rfi, self.n_ant, self.n_k_freq_rfi, self.n_k_time_rfi),
                dtype=complex,
            )
            self.init_rfi_k = self.forward_transform(base_sample)
        else:
            raise ValueError(f"Provided init type: {init_type} is not valid. Choose from (prior, truth, zeros, ones, sample).")

        self.init_rfi_k_base = self.inv_transform(self.init_rfi_k)

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

        rfi_shape = (self.n_rfi, 1, self.n_k_freq_rfi, self.n_k_time_rfi)

        assert_attr_shape(self, "mu_rfi_k", rfi_shape)
        assert_attr_shape(
            self, "sigma_rfi_k", (1, 1, self.n_k_freq_rfi, self.n_k_time_rfi)
        )
        assert_attr_shape(self, "init_rfi_k", rfi_shape)
        assert_attr_shape(self, "init_rfi_k_base", rfi_shape)

