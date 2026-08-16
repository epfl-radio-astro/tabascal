from abc import abstractmethod

from jax import vmap, random, Array, lax, checkpoint
import jax.numpy as jnp

from tabascal.components import Component, assert_attr_shape
from tabascal.config import TabConfig
from tabascal.dist import standard_normal
from tabascal.distributed import sharded_rfi_zeros
from tabascal.transform import affine_transform_full
from tabascal.tab_tools import get_observation_data_type
from tabascal.fft_gp import latent_to_signal_init, latent_to_signal, signal_to_latent_init, signal_to_latent
from tabascal.timing import measure_runtime

import numpy as np
from numpy.typing import NDArray
import xarray as xr

from typing import Tuple, Dict, Callable, List, Optional


#: Required contents of a light-curve file. See :func:`read_light_curves`.
_LIGHT_CURVE_VARS = ("light_curves", "norad_ids", "times", "freqs")

#: Dimensions of ``light_curves``. A store may declare them in any order.
_LIGHT_CURVE_DIMS = ("norad_ids", "times", "freqs")


def _light_curve_contents(est_path: str) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Load a light-curve file into (light_curves, norad_ids, times, freqs).

    Accepts a ``.zarr`` store (read with :func:`xarray.open_zarr`) or a ``.npz``.
    Both carry the same four arrays under the same names; the zarr form keeps
    ``norad_ids``/``times``/``freqs`` as coordinates of ``light_curves``.
    """
    if str(est_path).rstrip("/").endswith(".zarr"):
        xds = xr.open_zarr(est_path)
        available = sorted(set(xds.variables))
        missing = [
            name
            for name in _LIGHT_CURVE_VARS
            if name not in xds.variables and name not in xds.coords
        ]
        if missing:
            raise ValueError(
                f"{est_path} is missing {missing}. A light-curve zarr must hold "
                f"{list(_LIGHT_CURVE_VARS)}, with light_curves dimensioned "
                f"(norad_ids, times, freqs). It contains {available}."
            )
        # By name, not stored order: a swapped store with equal-length dims would
        # otherwise pass the shape check below and be read along the wrong axes.
        dims = tuple(xds["light_curves"].dims)
        if set(dims) != set(_LIGHT_CURVE_DIMS):
            raise ValueError(
                f"{est_path}: light_curves is dimensioned {dims}, but a "
                f"light-curve zarr must use exactly {_LIGHT_CURVE_DIMS} (in any "
                "order)."
            )
        curves = xds["light_curves"].transpose(*_LIGHT_CURVE_DIMS)

        return (
            np.asarray(curves.data),
            np.asarray(xds["norad_ids"].data),
            np.asarray(xds["times"].data),
            np.asarray(xds["freqs"].data),
        )

    npz = np.load(est_path)
    if not isinstance(npz, np.lib.npyio.NpzFile):
        raise ValueError(
            f"{est_path} is a bare .npy array. A light-curve file must be a .zarr "
            f"or a .npz holding {list(_LIGHT_CURVE_VARS)}, so that its rows can be "
            "matched to satellites by NORAD id and its samples interpolated onto "
            "the observation grid. A bare array carries none of that."
        )

    missing = [name for name in _LIGHT_CURVE_VARS if name not in npz.files]
    if missing:
        raise ValueError(
            f"{est_path} is missing {missing}. A light-curve .npz must hold "
            f"{list(_LIGHT_CURVE_VARS)}. It contains {sorted(npz.files)}."
        )
    return tuple(np.asarray(npz[name]) for name in _LIGHT_CURVE_VARS)  # type: ignore


def _as_norad_ids(labels: NDArray) -> List[Optional[int]]:
    """Labels as integer NORAD ids, with anything non-integer mapped to None.

    A file may legitimately carry named sources (nufft-gif plots e.g. "Fornax A"
    alongside the satellites); those can never match and are dropped.
    """
    ids: List[Optional[int]] = []
    for label in labels:
        text = str(label).strip()
        ids.append(int(text) if text.lstrip("-").isdigit() else None)
    return ids


def _interp_axis(values: NDArray, src: NDArray, dst: NDArray, axis: int) -> NDArray:
    """Linearly interpolate ``values`` along ``axis`` from ``src`` onto ``dst``.

    Samples beyond the ends of ``src`` are zero, matching the elevation mask's
    "no signal known" convention. A length-1 ``src`` is held constant instead.

    numpy in f64, not jax: MJD's ~6e4 day offset against ~1e-5 day spacings is
    unresolvable in f32, which collapses every sample onto one coordinate. The
    caller casts back to the working precision.
    """
    if len(src) == 1:
        return np.repeat(values, len(dst), axis=axis)

    if not np.all(np.diff(src) > 0):
        raise ValueError(
            "light-curve times and freqs must be strictly increasing; got "
            f"{src[:4]}{'...' if len(src) > 4 else ''}"
        )

    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)

    # Snap near-coincident endpoints inward, so a grid that differs from src only
    # in the last bits keeps its end samples interpolated rather than zeroed.
    tol = 1e-3 * float(np.min(np.diff(src)))
    dst = np.where((dst < src[0]) & (dst > src[0] - tol), src[0], dst)
    dst = np.where((dst > src[-1]) & (dst < src[-1] + tol), src[-1], dst)

    return np.apply_along_axis(
        lambda v: np.interp(dst, src, v, left=0.0, right=0.0), axis, values
    )


def read_light_curves(
    est_path: str, norad_ids: List[int], times_mjd: NDArray, freqs: NDArray
) -> Array:
    """Read an RFI light curve estimate onto the observation grid.

    File structure
    --------------
    A ``.zarr`` store (read with :func:`xarray.open_zarr`) or a ``.npz``, holding

    ``light_curves``
        ``(n_src, n_time, n_freq)``. One light curve per source, in the same
        units the RFI visibility amplitude is squared from.
    ``norad_ids``
        ``(n_src,)``. NORAD id labelling each row of ``light_curves``.
    ``times``
        ``(n_time,)``. Modified Julian Date, in **days**, strictly increasing.
    ``freqs``
        ``(n_freq,)``. Frequency in **Hz**, strictly increasing.

    In the zarr form the last three are coordinates of ``light_curves``.

    All four are required. The format is deliberately strict: this is the
    interchange standard between tabascal and whatever measures the light curves,
    and every loose alternative it could accept instead fails silently. Matching
    rows by position rather than by id attaches a curve to the wrong satellite
    without changing its shape; assuming the file's sampling matches the
    observation's resamples it wrongly by an unknown amount. Neither shows up as
    an error, only as a worse fit.

    Times are absolute (MJD) rather than seconds from the start of a particular
    observation, so a light curve is interpretable on its own and can be reused
    across measurement sets covering the same pass.

    Resampling
    ----------
    Light curves are interpolated linearly onto ``times_mjd`` and ``freqs``.
    Samples outside the file's coverage are zero, on either axis -- the file
    says nothing there, which is the same "no signal known" convention the
    elevation mask uses. An axis of length 1 is held constant instead, since a
    single sample carries no gradient to interpolate along.

    Partial coverage
    ----------------
    Satellites with no light curve in the file are zero, so an estimate only has
    to cover the satellites it was actually measured for rather than every
    satellite in the fit. Those are named in a warning. It is an error for *no*
    configured satellite to be found, which otherwise silently degrades the whole
    estimate to zeros.

    Returns
    -------
    Array (n_rfi, n_freq, n_time)
        Light curves on the observation grid, in ``norad_ids`` order, with
        unmatched satellites zero and NaNs replaced by zero.
    """

    curves, labels, src_times, src_freqs = _light_curve_contents(est_path)

    if curves.ndim != 3:
        raise ValueError(
            f"{est_path}: light_curves must be (n_src, n_time, n_freq), got shape "
            f"{curves.shape}."
        )
    expected = (len(labels), len(src_times), len(src_freqs))
    if curves.shape != expected:
        raise ValueError(
            f"{est_path}: light_curves has shape {curves.shape} but norad_ids, "
            f"times and freqs imply {expected}."
        )

    file_ids = _as_norad_ids(labels)

    # Rejected rather than resolved: keeping one of a repeated id would decide it
    # by file order, which is what matching by id exists to avoid.
    seen, duplicates = set(), set()
    for n_id in file_ids:
        if n_id is None:
            continue
        (duplicates if n_id in seen else seen).add(n_id)
    if duplicates:
        raise ValueError(
            f"{est_path} has more than one light curve for NORAD IDs "
            f"{sorted(duplicates)}. Each satellite must appear exactly once, so "
            "that a row is identified by its id rather than by its position."
        )

    lc_idx = {n_id: i for i, n_id in enumerate(file_ids) if n_id is not None}
    rows = [lc_idx.get(int(n_id)) for n_id in norad_ids]

    if all(row is None for row in rows):
        raise ValueError(
            f"No light curve in {est_path} matches any configured satellite. "
            f"Configured NORAD IDs are {sorted(set(int(n) for n in norad_ids))} "
            f"and the file contains {sorted(lc_idx)}."
        )

    missing = [int(n_id) for n_id, row in zip(norad_ids, rows) if row is None]
    if missing:
        print(
            f"Warning: no light curve in {est_path} for NORAD IDs "
            f"{sorted(set(missing))}; initialising those satellites at zero. "
            f"The file contains {sorted(lc_idx)}."
        )

    matched = np.asarray(curves[[r for r in rows if r is not None]], dtype=np.float64)

    # Any non-finite sample -> 0. posinf/neginf are set explicitly: nan_to_num's
    # defaults map them to the f64 extrema, which overflow back to inf in f32.
    n_inf = int(np.isinf(matched).sum())
    if n_inf:
        print(
            f"Warning: {est_path} has {n_inf} infinite light-curve samples; "
            "treating them as unmeasured (zero). Check the measurement for a "
            "divide-by-zero."
        )
    matched = np.nan_to_num(matched, nan=0.0, posinf=0.0, neginf=0.0)

    # (n_matched, n_time, n_freq) -> observation grid -> (n_matched, n_freq, n_time)
    matched = _interp_axis(matched, src_times, np.asarray(times_mjd), axis=1)
    matched = _interp_axis(matched, src_freqs, np.asarray(freqs), axis=2)
    matched = np.swapaxes(matched, 1, 2)

    out = np.zeros((len(norad_ids),) + matched.shape[1:], dtype=matched.dtype)
    out[[i for i, row in enumerate(rows) if row is not None]] = matched

    return jnp.asarray(out)


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
        # Absolute times, for resampling estimates sampled in MJD. Taken as read,
        # not recovered from times_jd, which loses ~1e-10 days on the round trip.
        self.times_mjd = np.asarray(tab_config.times_mjd)
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
        # Stored as a boolean, which is what jnp.where in the forward wants and is
        # the smallest thing to shard.
        rfi_mask_fine = getattr(tab_config, "rfi_mask_fine", None)
        self.rfi_mask_fine = (
            None if rfi_mask_fine is None else jnp.asarray(rfi_mask_fine, dtype=bool)
        )

        # Ordered as the n_rfi axis. The tail entries are sharding duplicates, so
        # consumers matching against a file slice to [:n_rfi_real] first.
        self.norad_ids = tab_config.norad_ids

    def build_mask_constants(self) -> dict:
        """Constants the signal mask needs, or ``{}`` when nothing is masked.

        Kept separate from each component's own ``build_constants`` so the
        None-check lives in one place. The mask is a constant rather than a
        closed-over array because it is indexed by ``n_rfi``: ``distributed.py``
        shards constants named in ``RFI_AXIS_NAMES`` along the source axis, and a
        captured array would instead be replicated, pulling ``rfi_A`` back to a
        full copy on every device.
        """
        if self.rfi_mask_fine is None:
            return {}
        return {"rfi_mask_fine": self.rfi_mask_fine}

    def build_masked_signal(self) -> Callable:
        """Return the signal-domain mask to apply at the end of a forward.

        Sibling of :meth:`masked_forward_transform`, which zeroes the padded dummy
        *sources* in the latent k-space. A time window cannot be expressed there:
        zeroing global Fourier modes cannot produce a time-limited signal, so the
        elevation mask has to be applied to ``rfi_A`` after ``latent_to_signal``.

        Both follow the same contract -- a base-class hook every component applies
        unconditionally, which degrades to the identity when there is nothing to
        mask (no elevation cut here, a single device there). Resolving the branch
        here rather than inside the traced function means a run without an
        elevation cut emits no mask op at all and pays nothing.

        The returned function takes any array whose leading axis is ``n_rfi`` and
        whose trailing axis is ``n_time_fine``, so a component that keeps the
        antenna axis broadcast rather than materialised can mask the smaller
        ``(n_rfi, n_freq_fine, n_time_fine)`` array before expanding it.
        """
        if self.rfi_mask_fine is None:
            return lambda rfi_A, constants: rfi_A

        prefix = self.prefix

        def masked_signal(rfi_A: Array, constants: dict) -> Array:
            mask = constants[f"{prefix}/rfi_mask_fine"]
            # (n_rfi, n_time_fine) -> (n_rfi, 1, ..., 1, n_time_fine)
            shape = (mask.shape[0], *(1,) * (rfi_A.ndim - 2), mask.shape[1])
            # where, not a multiply by 0/1: a masked sample is then exactly zero
            # even where rfi_A is non-finite, since 0 * inf and 0 * nan are nan and
            # would leak straight back through the mask. The optimiser can put
            # rfi_A somewhere non-finite transiently, and a masked sample must
            # contribute nothing regardless. Measured no more expensive than the
            # multiply, and identical in temporary allocation.
            return jnp.where(mask.reshape(shape), rfi_A, 0)

        return masked_signal

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
class ComplexRFIVarAnt(BaseGPRFI):

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
        return {
            "sigma_rfi_k": self.sigma_rfi_k,
            "mu_rfi_k": self.mu_rfi_k,
            **self.build_mask_constants(),
        }

    def build_forward(self):
        """Return pure, JIT-compatible function

        The latent-to-signal transform is scanned over antennas rather than vmapped.
        A double vmap over ``(n_rfi, n_ant)`` lowers to a single batched cuFFT of
        ``n_rfi * n_ant`` transforms on the zero-padded grid, and cuFFT sizes its plan
        work area for the whole batch. At 32 channels that reached a 12.6 GiB request
        which aborted the process from inside XLA -- a ``Check failure``, not a
        catchable Python OOM, so there was no graceful degradation. Scanning the
        antenna axis reduces that batch by ``n_ant``.

        ``checkpoint`` on the body is load-bearing rather than decorative:
        ``lax.scan`` stacks the body's residuals across iterations for reverse-mode
        AD, which would rebuild much of what the vmap was holding, so without it the
        scan fixes the cuFFT plan and not the autodiff tape.

        Measured on a 64-antenna / 32-channel / 4-satellite problem, single
        precision: peak device memory 35.80 -> 14.62 GB (2.45x) for a 4% runtime
        cost, with the optimised chi^2 unchanged to ~6 significant figures.
        """
        prefix = self.prefix
        forward_transform = self.masked_forward_transform
        masked_signal = self.build_masked_signal()
        pads = self.pads
        ss_idxs = self.ss_idxs

        # One antenna's sources at a time, so the cuFFT batch is n_rfi rather than
        # n_rfi * n_ant.
        @checkpoint
        def antenna_block(rfi_k_A_ant):
            return vmap(latent_to_signal, (0, None, None), 0)(
                rfi_k_A_ant, pads, ss_idxs
            )

        def forward(params: dict, state: dict, constants: dict):
            # Pure JAX operations only
            sigma_rfi_k = constants[f"{prefix}/sigma_rfi_k"]
            mu_rfi_k = constants[f"{prefix}/mu_rfi_k"]

            rfi_k_A_base = params["rfi_k_r_base"] + 1.0j * params["rfi_k_i_base"]

            rfi_k_A = forward_transform(rfi_k_A_base, sigma_rfi_k, mu_rfi_k)

            # lax.scan stacks along axis 0, so the antenna axis is moved there and
            # back. The leading swap is on the small latent grid; the trailing one is
            # full size and is part of the 4% measured above.
            _, rfi_A_ant_major = lax.scan(
                lambda carry, k_ant: (carry, antenna_block(k_ant)),
                None,
                jnp.swapaxes(rfi_k_A, 0, 1),
            )
            rfi_A = jnp.swapaxes(rfi_A_ant_major, 0, 1)
            rfi_A = masked_signal(rfi_A, constants)

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

        # Real satellites only; _zero_pad_rfi below restores the padded rows as
        # zeros, so a duplicate is never reported as a missing id.
        light_curves = read_light_curves(
            est_path, self.norad_ids[:self.n_rfi_real], self.times_mjd, self.freqs
        )

        # Per source, so every antenna sees the same light curve.
        est_rfi_A = jnp.broadcast_to(
            jnp.sqrt(jnp.abs(light_curves))[:, None],
            (light_curves.shape[0], self.n_ant, self.n_freq, self.n_time),
        )
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


class ComplexRFIConstAnt(BaseGPRFI):

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
        return {
            "sigma_rfi_k": self.sigma_rfi_k,
            "mu_rfi_k": self.mu_rfi_k,
            **self.build_mask_constants(),
        }

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix
        forward_transform = self.masked_forward_transform
        masked_signal = self.build_masked_signal()
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
            # The antenna axis is a singleton, so map over n_rfi only.
            rfi_A = vmap(latent_to_signal, (0, None, None), 0)(
                rfi_k_A[:, 0], pads, ss_idxs
            )
            # Masked before the broadcast: the mask does not vary over antennas, so
            # applying it here scales (n_rfi, n_freq_fine, n_time_fine) rather than
            # forcing the broadcast view below to materialise n_ant copies.
            rfi_A = masked_signal(rfi_A, constants)
            # Avoids allocating a full grid of ones and a multiply.
            rfi_A = jnp.broadcast_to(
                rfi_A[:, None], (n_rfi, n_ant, n_freq_fine, n_time_fine)
            )

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

        # Real satellites only; _zero_pad_rfi below restores the padded rows.
        light_curves = read_light_curves(
            est_path, self.norad_ids[:self.n_rfi_real], self.times_mjd, self.freqs
        )

        # Singleton antenna axis: the latent is shared and broadcast in the forward.
        est_rfi_A = jnp.sqrt(jnp.abs(light_curves))[:, None]

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

