from abc import abstractmethod

from jax import vmap, random, Array
import jax.numpy as jnp

from tabascal.components import Component, assert_attr_shape
from tabascal.config import TabConfig
from tabascal.dist import standard_normal
from tabascal.distributed import sharded_rfi_zeros
from tabascal.transform import affine_transform_full
from tabascal.gp import cholesky, resampling_kernel, get_times
from tabascal.tab_tools import get_observation_data_type
from tabascal.fft_gp import latent_to_signal_init, latent_to_signal, signal_to_latent_init, signal_to_latent
from tabascal.timing import measure_runtime

import numpy as np
import xarray as xr

from typing import Tuple, Dict, Callable, List


def read_light_curves(est_path: str, norad_ids: List[int]) -> Array:
    """Read an RFI light curve estimate, ordered to match `norad_ids`.

    Light curve files written by `nufft-gif` label each source in `titles`, so the
    satellites are matched by NORAD ID and any non-satellite sources (e.g. Fornax A)
    are dropped. Files without `titles` fall back to the leading `n_rfi` sources,
    which is only correct when `norad_ids` is ascending.

    Returns
    -------
    Array (n_rfi, n_time, n_freq)
        Light curves, with out-of-view NaNs replaced by a zero RFI estimate.
    """

    est = np.load(est_path)
    titles = None
    if est_path.endswith(".npz"):
        titles = [str(t) for t in est["titles"]] if "titles" in est.files else None
        est = est["light_curves"]

    if titles is None:
        print(
            f"Warning: {est_path} has no source labels. Taking the first "
            f"{len(norad_ids)} light curves as the satellites, in file order."
        )
        idxs = list(range(len(norad_ids)))
    else:
        lc_idx = {int(t): i for i, t in enumerate(titles) if t.strip().isdigit()}
        missing = [n_id for n_id in norad_ids if int(n_id) not in lc_idx]
        if missing:
            raise ValueError(
                f"No light curve found in {est_path} for NORAD IDs {missing}. "
                f"It contains {sorted(lc_idx)}."
            )
        idxs = [lc_idx[int(n_id)] for n_id in norad_ids]

    # NaN (satellite out of view) -> 0 RFI estimate; keeps the init finite.
    return jnp.nan_to_num(jnp.asarray(est[idxs]))


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


def rfi_signal_config_validation(rfi_config: Dict, vis_obs: Array, freqs: Array, chan_width: float, times: Array, int_time: float) -> Dict:
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

    def extent(x, dx):
        ext = float(jnp.max(x) - jnp.min(x))
        if ext == 0.0:
            return float(dx)
        else:
            return ext

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
    elif isinstance(gp_var, (float, int)):
        rfi_config["var"] = float(gp_var)
    else:
        raise ValueError(f"Config parameter (rfi:\n\tvar: {gp_var}) is not of type float or int.")
    
    if not gp_freq_l: # Set Default
        est_gp_freq_l = extent(freqs, chan_width) / 2
        rfi_config["corr_freq"] = est_gp_freq_l
    elif isinstance(gp_freq_l, (float, int)):
        rfi_config["corr_freq"] = float(gp_freq_l)
    else:
        raise ValueError(f"Config parameter (rfi:\n\tcorr_freq: {gp_freq_l}) is not of type float or int.")
    
    if not gp_time_l: # Set Default
        est_gp_time_l = extent(times, int_time) / 2
        rfi_config["corr_time"] = est_gp_time_l
    elif isinstance(gp_time_l, (float, int)):
        rfi_config["corr_time"] = float(gp_time_l)
    else:
        raise ValueError(f"Config parameter (rfi:\n\tcorr_time: {gp_time_l}) is not of type float or int.")    
    
    print()
    print(f"Using RFI var : {rfi_config['var']:.1e} Jy")
    print(f"Using RFI corr_freq : {rfi_config['corr_freq']/1e3:.1f} kHz")
    print(f"Using RFI corr_time : {rfi_config['corr_time']:.1f} s")

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
            tab_config.args["rfi"], tab_config.vis_obs, tab_config.freqs, tab_config.chan_width, tab_config.times, tab_config.int_time)

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

        # Real (unpadded) source count. Under device sharding n_rfi is padded up to a
        # multiple of the device count with dark dummy sources; every prior mean and
        # init below zeroes rows [n_rfi_real:] so the dummies carry exactly zero
        # amplitude and zero gradient (the vis contribution is quadratic in rfi_A).
        self.n_rfi_real = getattr(tab_config, "n_rfi_real", tab_config.n_rfi)

        # Elevation mask, zeroing the RFI signal while a satellite is below the
        # horizon. None when disabled. Shape (n_rfi, n_time_fine), i.e. it covers the
        # padded dummy rows too -- they duplicate the last real satellite, so they
        # inherit its mask and are zeroed independently by masked_forward_transform.
        rfi_mask_fine = getattr(tab_config, "rfi_mask_fine", None)
        self.rfi_mask_fine = None if rfi_mask_fine is None else jnp.array(rfi_mask_fine)

        # NORAD IDs, ordered as the n_rfi axis of every RFI array. Under sharding the
        # tail entries are padding duplicates, so consumers that look ids up in an
        # external file slice to [:n_rfi_real] and zero-pad the result instead.
        self.norad_ids = tab_config.norad_ids

    @staticmethod
    def mask_signal(rfi_A: Array, mask: Array) -> Array:
        """Zero the RFI signal (n_rfi, n_ant, n_freq_fine, n_time_fine) outside the mask."""

        return rfi_A * mask[:, None, None, :]

    def _mask_dummy_rfi(self, arr: Array) -> Array:
        """Zero the padded (dark dummy) rows of an (n_rfi, ...) array; no-op unpadded."""
        return arr.at[self.n_rfi_real:].set(0)

    def _zero_pad_rfi(self, arr: Array) -> Array:
        """Zero-pad axis 0 up to the (padded) n_rfi count; no-op when already there.

        Truth/estimate sources (tab-sim zarr, estimate files) only know the real
        satellites, so their arrays arrive with n_rfi_real rows.
        """
        n_pad = self.n_rfi - arr.shape[0]
        if n_pad <= 0:
            return arr
        pad = jnp.zeros((n_pad,) + arr.shape[1:], dtype=arr.dtype)
        return jnp.concatenate([arr, pad], axis=0)

    def _set_outputs(self):

        # This placeholder is a fine-grid memory hog; under sharding each device only
        # ever allocates its own RFI shard (never the full array).
        self.state_outputs = {
            "rfi_A": sharded_rfi_zeros(
                (self.n_rfi, self.n_ant, self.n_freq_fine, self.n_time_fine), complex
            ),
        }

    def _compute_gp_params(self):
        pass

    def _compute_prior_params(self):
        pass

    def _compute_init_params(self):
        pass

    @abstractmethod
    def forward_transform(self, base_params, L, mu):
        pass

    @abstractmethod
    def inv_transform(self, params, L, mu):
        pass

    # helper function to set dummy rfi values to 0 for padding. Required for multi-device.
    def masked_forward_transform(self, base_params, L, mu):
        return self._mask_dummy_rfi(self.forward_transform(base_params, L, mu))


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

    def build_constants(self):
        constants = {
            "L_rfi_A": self.L_rfi_A,
            "mu_rfi_A": self.mu_rfi_A,
            "resample_rfi": self.resample_rfi,
        }
        if self.rfi_mask_fine is not None:
            constants["rfi_mask_fine"] = self.rfi_mask_fine.astype(self.mu_rfi_A.dtype)

        return constants

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix
        forward_transform = self.masked_forward_transform
        mask_signal = self.mask_signal
        masked = self.rfi_mask_fine is not None

        def forward(params: dict, state: dict, constants: dict):
            # Pure JAX operations only
            L_rfi_A = constants[f"{prefix}/L_rfi_A"]
            mu_rfi_A = constants[f"{prefix}/mu_rfi_A"]
            resample_rfi = constants[f"{prefix}/resample_rfi"]

            rfi_A_induce_base = params["rfi_r_induce_base"]

            rfi_A_induce = forward_transform(rfi_A_induce_base, L_rfi_A, mu_rfi_A)

            rfi_A = vmap(vmap(vmap(jnp.dot, (None, 0), 0), (None, 1), 1), (None, 2), 2)(
                resample_rfi, rfi_A_induce
            )

            if masked:
                rfi_A = mask_signal(rfi_A, constants[f"{prefix}/rfi_mask_fine"])

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

        # The zarr only knows the real satellites; zero-pad to the sharded count.
        self.true_rfi_A_induce = self._zero_pad_rfi(
            read_true_rfi_A(sim_zarr_path, data_col, self.rfi_times).real
        )
        self.true_rfi_A_induce_base = self.inv_transform(self.true_rfi_A_induce, self.L_rfi_A, self.mu_rfi_A)

    def forward_transform(self, base_params, L, mu):

        params = vmap(
            vmap(vmap(affine_transform_full, (0, None, 0), 0), (1, None, 1), 1),
            (2, None, 2),
            2,
        )(base_params, L, mu)

        return params

    def inv_transform(self, params, L, mu):

        base_params = vmap(
            vmap(vmap(jnp.linalg.solve, (None, 0), 0), (None, 1), 1), (None, 2), 2
        )(L, params - mu)

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
            self.init_rfi_A_induce = self.masked_forward_transform(base_sample, self.L_rfi_A, self.mu_rfi_A)
        else:
            raise ValueError(f"Provided init type: {init_type} is not valid. Choose from (prior, zeros, ones, truth, sample).")

        # Darkness of the padded dummy sources is enforced in the forward, by
        # masked_forward_transform -- not here. So init_rfi_A_induce (and the prior mean it
        # may come from) can carry non-zero padded rows for e.g. init="ones"; they contribute
        # exactly zero amplitude and zero gradient regardless.
        self.init_rfi_A_induce_base = self.inv_transform(self.init_rfi_A_induce, self.L_rfi_A, self.mu_rfi_A)

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

    def build_constants(self):
        constants = {
            "L_rfi_A": self.L_rfi_A,
            "mu_rfi_A": self.mu_rfi_A,
            "resample_rfi": self.resample_rfi,
        }
        if self.rfi_mask_fine is not None:
            constants["rfi_mask_fine"] = self.rfi_mask_fine.astype(self.resample_rfi.dtype)

        return constants

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix
        forward_transform = self.masked_forward_transform
        mask_signal = self.mask_signal
        masked = self.rfi_mask_fine is not None

        def forward(params: dict, state: dict, constants: dict):
            # Pure JAX operations only
            L_rfi_A = constants[f"{prefix}/L_rfi_A"]
            mu_rfi_A = constants[f"{prefix}/mu_rfi_A"]
            resample_rfi = constants[f"{prefix}/resample_rfi"]

            rfi_A_induce_base = (
                params["rfi_r_induce_base"] + 1.0j * params["rfi_i_induce_base"]
            )

            rfi_A_induce = forward_transform(rfi_A_induce_base, L_rfi_A, mu_rfi_A)

            rfi_A = vmap(vmap(vmap(jnp.dot, (None, 0), 0), (None, 1), 1), (None, 2), 2)(
                resample_rfi, rfi_A_induce
            )

            if masked:
                rfi_A = mask_signal(rfi_A, constants[f"{prefix}/rfi_mask_fine"])

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

        # The zarr only knows the real satellites; zero-pad to the sharded count.
        self.true_rfi_A_induce = self._zero_pad_rfi(
            read_true_rfi_A(sim_zarr_path, data_col, self.rfi_times)
        )
        self.true_rfi_A_induce_base = self.inv_transform(self.true_rfi_A_induce, self.L_rfi_A, self.mu_rfi_A)

    def forward_transform(self, base_params, L, mu):

        params = vmap(
            vmap(vmap(affine_transform_full, (0, None, 0), 0), (1, None, 1), 1),
            (2, None, 2),
            2,
        )(base_params, L, mu)

        return params

    def inv_transform(self, params, L, mu):

        base_params = vmap(
            vmap(vmap(jnp.linalg.solve, (None, 0), 0), (None, 1), 1), (None, 2), 2
        )(L, params - mu)

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
            self.init_rfi_A_induce = self.masked_forward_transform(
                base_sample, self.L_rfi_A, self.mu_rfi_A
            )
        else:
            raise ValueError(f"Provided init type: {init_type} is not valid. Choose from (prior, zeros, ones, truth, sample).")

        # Darkness of the padded dummy sources is enforced in the forward, by
        # masked_forward_transform -- not here. So init_rfi_A_induce (and the prior mean it
        # may come from) can carry non-zero padded rows for e.g. init="ones"; they contribute
        # exactly zero amplitude and zero gradient regardless.
        self.init_rfi_A_induce_base = self.inv_transform(self.init_rfi_A_induce, self.L_rfi_A, self.mu_rfi_A)

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

    def build_constants(self):
        constants = {
            "sigma_rfi_k": self.sigma_rfi_k,
            "mu_rfi_k": self.mu_rfi_k,
        }
        if self.rfi_mask_fine is not None:
            constants["rfi_mask_fine"] = self.rfi_mask_fine.astype(
                self.sigma_rfi_k.dtype
            )

        return constants

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix
        forward_transform = self.masked_forward_transform
        mask_signal = self.mask_signal
        masked = self.rfi_mask_fine is not None
        pads = self.pads
        ss_idxs = self.ss_idxs

        def forward(params: dict, state: dict, constants: dict):
            # Pure JAX operations only
            sigma_rfi_k = constants[f"{prefix}/sigma_rfi_k"]
            mu_rfi_k = constants[f"{prefix}/mu_rfi_k"]

            rfi_k_A_base = params["rfi_k_r_base"] + 1.0j * params["rfi_k_i_base"]

            rfi_k_A = forward_transform(rfi_k_A_base, sigma_rfi_k, mu_rfi_k)

            rfi_A = vmap(vmap(latent_to_signal, (0, None, None), 0), (1, None, None), 1)(
                rfi_k_A, pads, ss_idxs
            )

            if masked:
                rfi_A = mask_signal(rfi_A, constants[f"{prefix}/rfi_mask_fine"])

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
        # gammas = [1e2, 1e2]
        # gammas = [5, 5]
        gammas = [3, 3]
        pk_cutoff = 1e-9

        self.pk, self.ks, self.pads, self.ss_idxs = latent_to_signal_init(
            ns,
            dxs,
            pad_factors,
            [self.n_int_freq, self.n_int_time],
            p0,
            k0s,
            gammas,
            pk_cutoff,
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
            pk_cutoff,
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

        scale_norm = self.gp_var / jnp.sum(self.pk)
        self.pk = scale_norm * self.pk

        self.n_k_freq_rfi, self.n_k_time_rfi = self.pk.shape
        self.sigma_rfi_k = jnp.sqrt(self.pk)[None, None, :, :]

    def _compute_data_est(self, vis_obs):

        # Split the data estimate over the *real* sources only; padded dummies get a
        # zero mean so they stay dark.
        est_rfi_k = self.signal_to_latent(jnp.sqrt(jnp.max(jnp.abs(vis_obs), axis=0)))[None, None, :, :] * jnp.ones((self.n_rfi, self.n_ant, 1, 1)) / self.n_rfi_real

        return est_rfi_k

    def _compute_prior_params(self, prior_type, vis_obs, est_path):

        if prior_type == "data":
            print("Using data for RFI prior mean")
            self.mu_rfi_k = self._compute_data_est(vis_obs)
        elif prior_type == "est": 
            print("Using provided estimate for RFI prior mean")
            self.mu_rfi_k = self._read_estimate(est_path)
        elif prior_type in ["zeros", 0]:
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

    def _compute_true_params(self, sim_zarr_path: str, data_col: str):

        # The zarr only knows the real satellites; zero-pad to the sharded count.
        rfi_A = read_true_rfi_A(sim_zarr_path, data_col, self.times)
        self.true_rfi_k_A = self._zero_pad_rfi(vmap(vmap(self.signal_to_latent))(rfi_A))
        self.true_rfi_k_A_base = self.inv_transform(self.true_rfi_k_A, self.sigma_rfi_k, self.mu_rfi_k)

    def _read_estimate(self, est_path):

        # Only the real satellites have a light curve; the padded rows are restored as
        # zeros by _zero_pad_rfi below, so a padding duplicate is never looked up (and
        # read_light_curves' missing-id error stays about genuinely missing sources).
        light_curves = read_light_curves(est_path, self.norad_ids[:self.n_rfi_real])

        est_rfi_A = jnp.max(jnp.sqrt(jnp.abs(light_curves)), axis=-1)[:, None, None, :] * jnp.ones((1, self.n_ant, self.n_freq, 1))
        est_rfi_k_A = vmap(vmap(self.signal_to_latent))(est_rfi_A)

        return self._zero_pad_rfi(est_rfi_k_A)

    def _compute_init_params(self, init_type: str, est_path: str):

        if init_type == "prior":
            print("Using prior mean for rfi_A init")
            self.init_rfi_k = self.mu_rfi_k
        elif init_type == "est":
            print("Using provided estimate for rfi_A init")
            self.init_rfi_k = self._read_estimate(est_path)
        elif init_type == "truth":
            print("Using truth for rfi_A init")
            self.init_rfi_k = self.true_rfi_k_A
        elif init_type in ["zeros", 0]:
            print("Using zeros for rfi_A init")
            # zeros_k is shape (1, 1, n_k_freq_rfi, n_k_time_rfi)
            zeros_k = self.signal_to_latent(jnp.zeros((self.n_freq, self.n_time), dtype=complex))[None,None,:,:]
            # init_rfi_k is shape (n_rfi, n_ant, n_k_freq_rfi, n_k_time_rfi)
            self.init_rfi_k = zeros_k * jnp.ones((self.n_rfi, self.n_ant, 1, 1))
        elif init_type in ["ones", 1]:
            print("Using ones for rfi_A init")
            # ones_k is shape (1, 1, n_k_freq_rfi, n_k_time_rfi)
            ones_k = self.signal_to_latent(jnp.ones((self.n_freq, self.n_time), dtype=complex))[None,None,:,:]
            # init_rfi_k is shape (n_rfi, n_ant, n_k_freq_rfi, n_k_time_rfi)
            self.init_rfi_k = ones_k * jnp.ones((self.n_rfi, self.n_ant, 1, 1))
        elif init_type == "sample":
            print("Drawing sample from prior for rfi_A init")
            base_sample = random.normal(
                random.PRNGKey(self.r_seed),
                (self.n_rfi, self.n_ant, self.n_k_freq_rfi, self.n_k_time_rfi),
                dtype=complex,
            )
            self.init_rfi_k = self.masked_forward_transform(base_sample, self.sigma_rfi_k, self.mu_rfi_k)
        else:
            raise ValueError(f"Provided init type: {init_type} is not valid. Choose from (prior, truth, zeros, ones, sample).")

        self.init_rfi_k_base = self.inv_transform(self.init_rfi_k, self.sigma_rfi_k, self.mu_rfi_k)

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

    def build_constants(self):
        constants = {
            "sigma_rfi_k": self.sigma_rfi_k,
            "mu_rfi_k": self.mu_rfi_k,
        }
        if self.rfi_mask_fine is not None:
            constants["rfi_mask_fine"] = self.rfi_mask_fine.astype(self.sigma_rfi_k.dtype)

        return constants

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix
        forward_transform = self.masked_forward_transform
        mask_signal = self.mask_signal
        masked = self.rfi_mask_fine is not None
        pads = self.pads
        ss_idxs = self.ss_idxs
        n_rfi = self.n_rfi
        n_ant = self.n_ant
        n_freq_fine = self.n_freq_fine
        n_time_fine = self.n_time_fine

        def forward(params: dict, state: dict, constants: dict):
            # Pure JAX operations only
            sigma_rfi_k = constants[f"{prefix}/sigma_rfi_k"]
            mu_rfi_k = constants[f"{prefix}/mu_rfi_k"]

            rfi_k_A_base = params["rfi_k_r_base"] + 1.0j * params["rfi_k_i_base"]

            rfi_k_A = forward_transform(rfi_k_A_base, sigma_rfi_k, mu_rfi_k)
            rfi_A = vmap(vmap(latent_to_signal, (0, None, None), 0), (1, None, None), 1)(
                rfi_k_A, pads, ss_idxs
            )
            rfi_A = rfi_A * jnp.ones((n_rfi, n_ant, n_freq_fine, n_time_fine))

            if masked:
                rfi_A = mask_signal(rfi_A, constants[f"{prefix}/rfi_mask_fine"])

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
        p0 = self.gp_var
        gammas = [1e2, 1e2]
        pk_cutoff = 1e-6

        self.pk, self.ks, self.pads, self.ss_idxs = latent_to_signal_init(
            ns,
            dxs,
            pad_factors,
            [self.n_int_freq, self.n_int_time],
            p0,
            k0s,
            gammas,
            pk_cutoff,
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
            pk_cutoff,
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

        scale_norm = self.gp_var / jnp.sum(self.pk)
        self.pk = scale_norm * self.pk

        self.n_k_freq_rfi, self.n_k_time_rfi = self.pk.shape
        self.sigma_rfi_k = jnp.sqrt(self.pk)[None, None, :, :]

    def _compute_data_est(self, vis_obs: Array) -> Array:

        # est_vis_rfi is shape (n_freq, n_time)
        # RFI antenna estimate is sqrt of average visibility amplitude on maximum baseline.
        # Split over the *real* sources only; padded dummies get a zero mean.
        est_rfi_A = jnp.sqrt(jnp.max(jnp.abs(vis_obs / self.n_rfi_real), axis=0))
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
            print("Using provided estimate for RFI prior mean")
            self.mu_rfi_k = self._read_estimate(est_path)
        elif prior_type in ["zeros", 0]:
            print("Using zeros for RFI prior mean")
            self.mu_rfi_k = jnp.zeros(
                (self.n_rfi, 1, self.n_k_freq_rfi, self.n_k_time_rfi), dtype=complex
            )
        else:
            raise ValueError(f"Provided prior type: {prior_type} is not valid. Choose from (data, zeros).")

    def forward_transform(self, base_params, sigma, mu):

        params = sigma * base_params + mu

        return params

    def inv_transform(self, params, sigma, mu):

        base_params = (params - mu) / sigma

        return base_params

    def _compute_true_params(self, sim_zarr_path: str, data_col: str):

        # true_rfi_A shape goes from (n_rfi, n_ant, n_freq, n_time) -> (n_rfi, 1, n_freq, n_time)
        true_rfi_A = jnp.mean(read_true_rfi_A(sim_zarr_path, data_col, self.times), axis=1, keepdims=True)

        # true_rfi_k_A is shape (n_rfi, 1, n_k_freq_rfi, n_k_time_rfi)
        # Latent prediction is mapped over axes (0, 1)
        # The zarr only knows the real satellites; zero-pad to the sharded count.
        self.true_rfi_k_A = self._zero_pad_rfi(vmap(vmap(self.signal_to_latent))(true_rfi_A))

        self.true_rfi_k_A_base = self.inv_transform(self.true_rfi_k_A, self.sigma_rfi_k, self.mu_rfi_k)

    def _read_estimate(self, est_path):

        # Real satellites only; _zero_pad_rfi below restores the padded rows as zeros.
        light_curves = read_light_curves(est_path, self.norad_ids[:self.n_rfi_real])

        est_rfi_A = jnp.max(jnp.sqrt(jnp.abs(light_curves)), axis=-1)[:, None, None, :] * jnp.ones((1, 1, self.n_freq, 1))

        return self._zero_pad_rfi(vmap(vmap(self.signal_to_latent))(est_rfi_A))

    def _compute_init_params(self, init_type, est_path):

        if init_type == "prior":
            print("Using prior mean for rfi_A init")
            self.init_rfi_k = self.mu_rfi_k
        elif init_type == "est": 
            print("Using provided estimate for rfi_A init")
            self.init_rfi_k = self._read_estimate(est_path)
        elif init_type == "truth":
            print("Using truth for rfi_A int")
            self.init_rfi_k = self.true_rfi_k_A
        elif init_type in ["zeros", 0]:
            print("Using zeros for rfi_A init")
            zeros = jnp.zeros((self.n_freq, self.n_time), dtype=complex)
            self.init_rfi_k = self.signal_to_latent(zeros)[None,None,:,:] * jnp.ones((self.n_rfi, 1, 1, 1))
        elif init_type in ["ones", 1]:
            print("Using ones for rfi_A init")
            ones = jnp.ones((self.n_freq, self.n_time), dtype=complex)
            self.init_rfi_k = self.signal_to_latent(ones)[None,None,:,:] * jnp.ones((self.n_rfi, 1, 1, 1))
        elif init_type == "sample":
            print("Drawing sample from prior for rfi_A init")
            # This variant carries a singleton antenna axis: the latent is shared by
            # every antenna and only broadcast to n_ant inside the forward.
            base_sample = random.normal(
                random.PRNGKey(self.r_seed),
                (self.n_rfi, 1, self.n_k_freq_rfi, self.n_k_time_rfi),
                dtype=complex,
            )
            self.init_rfi_k = self.masked_forward_transform(base_sample, self.sigma_rfi_k, self.mu_rfi_k)
        else:
            raise ValueError(f"Provided init type: {init_type} is not valid. Choose from (prior, truth, zeros, ones, sample).")

        self.init_rfi_k_base = self.inv_transform(self.init_rfi_k, self.sigma_rfi_k, self.mu_rfi_k)

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

