from tabascal.imports import import_components
from tabascal.components.likelihood import gaussian
from tabascal.tab_tools import read_ms, fix_padding
from tabascal.components.trajectory import (
    fetch_orbital_elements,
    get_satellite_positions,
    itrs_to_gcrs_sf,
)
from tabascal.tle import print_spacetrack_status, preflight_tle_check
from tabascal.interferometry import (
    calculate_fringe_frequency_numpy,
    fit_nearfield_fringe_freq_poly_numpy,
    fringe_params_at_offsets,
    size_subwindows,
    get_strides_and_idxs,
    itrf_to_uvw_numpy,
)
from tabascal.fft_gp import domain_ss
from tabascal.time import secs_to_days, mjd_to_jd, jd_to_mjd, gast_deg

import jax.numpy as jnp

import numpy as np

import numpyro

from typing import Optional, Callable, Dict, List

from importlib.resources import files
import os
import re
import yaml
import collections.abc

    
def deep_update(d: Dict, u: Dict) -> Dict:
    """Recursively update a dictionary which includes subdictionaries.

    Parameters
    ----------
    d : Dict
        Base dictionary to update.
    u : Dict
        Update dictionary.

    Returns
    -------
    Dict
        Updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class _TabSafeLoader(yaml.SafeLoader):
    """SafeLoader whose float resolver also accepts bare scientific notation.

    PyYAML's stock resolver only treats a token as a float when the exponent is
    signed (``1.0e+9``); it parses ``1e9`` / ``3e3`` / ``209e3`` as *strings*. The
    config files use the bare form throughout, so add a resolver that accepts it.
    It is attached to this private subclass — not the shared ``yaml.SafeLoader`` —
    so importing tabascal does not reprogram YAML float parsing for the whole
    process. Anything that needs this behaviour must load via :func:`yaml_load`.
    """


_TabSafeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        """^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def yaml_load(path):
    with open(path) as f:
        return yaml.load(f, Loader=_TabSafeLoader)


def load_config(path: str) -> Dict:
    """Load a configuration file and populate default parameters where needed.

    Parameters
    ----------
    path : str
        Path to the yaml config file.
    
    Returns
    -------
    dict
        Configuration dictionary.
    """
    config_dir = files("tabascal").joinpath("data/config").__str__()
    tab_base_config_path = os.path.join(config_dir, "tab_config_base.yaml")
    base_config = yaml_load(tab_base_config_path)

    try:
        return deep_update(base_config, yaml_load(path))
    except Exception as e:
        raise IOError(f"Configuration file could not be loaded from {path}") from e

    
class TabConfig:
    """Configuration parameters for tabascal method"""

    def __init__(self, config: Dict, ms_path: str):

        # self.config = config
        self.args = config
        self.precision = config.get("model", {}).get("precision", "single")
        self.ms_path = ms_path
        self.spacetrack_path = config["satellites"].get("spacetrack_path")
        self.extra_tle_dir = config["satellites"].get("extra_tle_dir")

        print_spacetrack_status()
        preflight_tle_check(
            config["satellites"].get("norad_ids") or [],
            ms_path,
            extra_tle_dir=self.extra_tle_dir,
        )

        self.read_ms_params(
            config["data"]["freq"],
            config["data"]["corr"],
            config["data"]["data_col"],
        )
        self.set_noise(config["data"]["noise"])
        self.set_flags(config["data"]["flags"])
        config = fix_padding(
            config, self.n_freq
        )  # Bad solution, should be fixed in fft_gp. Issue when using a single frequency channel.

        self.get_orbital_elements(
            config["satellites"].get("norad_ids"),
            extra_tle_dir=config["satellites"].get("extra_tle_dir"),
        )

        config["rfi"]["min_time_bins"] = 1
        config["rfi"]["max_time_bins"] = 30

        self.n_int_time = config["rfi"]["n_int_time"]
        self.n_int_freq = config["rfi"]["n_int_freq"]

        self.estimate_rfi_sampling(
            config["rfi"]["time_int_factor"],
            config["rfi"]["min_time_bins"],
            config["rfi"]["max_time_bins"],
        )

        # Analytic RFI-visibility path: when selected, size the uniform sub-window count
        # K from the trajectory cubic phase-curvature and build the per-sub-window fringe
        # parameters. This replaces the oversampled time grid with an edge/centre grid
        # (built in _set_freqs_times). Default stays the oversample path.
        self.vis_method = config["rfi"].get("vis_method", "oversample")
        if self.vis_method == "analytic":
            self.setup_analytic_sampling(config["rfi"])

        self._set_freqs_times()

    def set_noise(self, noise: float):

        if noise:
            self.noise = noise

    def set_flags(self, include_flags: bool):

        if not include_flags:
            self.flags = jnp.zeros_like(self.flags, dtype=bool)

        print(f"\n{100*self.flags.mean():.1f} % Data Flagged (Not Included in Likelihood)\n")

    def read_ms_params(self, freq: float, corr: str, data_col: str):

        ms_params = read_ms(self.ms_path, freq, None, corr, data_col)

        self.phase_centre = {"ra": ms_params["ra"], "dec": ms_params["dec"]}
        self.dish_d = ms_params["dish_d"]
        self.ants_itrf = ms_params["ants_itrf"]
        self.vis_obs = ms_params["vis_obs"]
        self.uvw = ms_params["uvw"]
        self.flags = ms_params["flags"]

        self.n_ant = ms_params["n_ant"]
        self.n_bl = ms_params["n_bl"]
        self.n_time = ms_params["n_time"]
        self.n_freq = ms_params["n_freq"]
        self.n_corr = ms_params["n_corr"]

        self.int_time = ms_params["int_time"]
        self.times = np.asarray(ms_params["times"])
        self.times_jd = mjd_to_jd(ms_params["times_mjd"])

        self.chan_width = ms_params["chan_width"]
        self.freqs = np.asarray(ms_params["freqs"])

        self.noise = ms_params["noise"]
        self.a1 = ms_params["a1"]
        self.a2 = ms_params["a2"]

    def estimate_rfi_sampling(
        self, n_int_factor: float, min_time_bins: int, max_time_bins: int
    ):

        jd_minute = 1 / (24 * 60)
        times_jd_coarse = np.arange(
            self.times_jd[0], self.times_jd[-1] + jd_minute, jd_minute
        )
        # Satellite positions, GAST, antenna UVW and fringe frequencies are all
        # one-shot host-side setup, so always compute them in numpy/skyfield (f64):
        # faster than the jax path (no JIT compile) and accurate in both precisions.
        rfi_xyz = np.asarray(get_satellite_positions(self.tles, times_jd_coarse))

        gsa = gast_deg(times_jd_coarse)  # GAST in degrees (UTC convention)
        gh0 = (gsa - self.phase_centre["ra"]) % 360  # type: ignore

        ants_u = itrf_to_uvw_numpy(self.ants_itrf, gh0, self.phase_centre["dec"])[:, :, 0]

        get_fringe_freq = lambda rfi_pos: calculate_fringe_frequency_numpy(
            jd_to_mjd(times_jd_coarse),
            np.max(self.freqs),
            rfi_pos,
            self.ants_itrf,
            ants_u,
            self.phase_centre["dec"],
        )
        # fringe_freq is shape (n_rfi, n_time_coarse, n_bl)
        fringe_freq = np.array([get_fringe_freq(rfi_pos) for rfi_pos in rfi_xyz])

        self.max_rfi_vis = np.max(np.abs(self.vis_obs))
        sample_freq_bl = (
            np.pi
            * np.max(np.abs(fringe_freq), axis=(0, 1))
            * np.sqrt(self.max_rfi_vis / (6 * self.noise))
        )
        n_int_times = np.ceil(n_int_factor * self.int_time * sample_freq_bl).astype(int)

        # time_sample_idxs and time_strides are only used in RiemannVisTimeFreqVariable
        self.time_sample_idxs, self.time_strides, self.n_int_time = (
            get_strides_and_idxs(n_int_times, min_time_bins, max_time_bins)
        )

    def setup_analytic_sampling(self, rfi_config: Dict):
        """Size the analytic sub-window count K and build per-sub-window fringe params.

        Host-side (numpy/skyfield, f64) one-shot setup, mirroring ``estimate_rfi_sampling``.
        Produces the constants the ``AnalyticVisCalculation`` component consumes:

        * ``analytic_K`` / ``analytic_dt_sub`` — uniform sub-window count and width.
        * ``analytic_f`` / ``analytic_fdot`` — fringe frequency (Hz) and rate-derivative
          (Hz/s) at every sub-window centre, ``(n_rfi, n_bl, n_time*K)``, at the reference
          frequency ``max(freqs)``. They scale linearly with channel frequency (handled in
          the component).
        * ``analytic_edge_gather`` — ``(n_time, K+1)`` index of each window's edges into the
          shared global edge grid (G2: edges shared between adjacent sub-windows/windows).
        * ``analytic_freq_scale`` — ``freqs / freq_ref`` per channel.

        Assumes uniform, contiguous integration windows (spacing == int_time), the standard
        drift-scan layout also assumed by ``estimate_rfi_sampling``.
        """
        # The frequency axis must not be sub-integrated: the analytic factor is time-only.
        if self.n_int_freq != 1:
            raise ValueError(
                "analytic vis_method requires n_int_freq == 1 (time-only path); "
                f"got n_int_freq={self.n_int_freq}"
            )

        resid_tol = float(rfi_config.get("resid_tol", 3e-4))
        resid_A = float(rfi_config.get("resid_A", 3.3))
        k_max = int(rfi_config.get("max_subwindows", 64))
        n_fit = int(rfi_config.get("fit_samples", 16))

        dec = self.phase_centre["dec"]
        ra = self.phase_centre["ra"]
        freq_ref = float(np.max(self.freqs))
        n_time = self.n_time
        dt = self.int_time

        # Uniform, window-contiguous fit grid: n_fit samples per coarse window, symmetric
        # about each window centre.
        dt_fit = dt / n_fit
        n_fit_total = n_time * n_fit
        t_fit_sec = self.times[0] - dt / 2.0 + (np.arange(n_fit_total) + 0.5) * dt_fit
        times_jd_fit = self.times_jd[0] + secs_to_days(t_fit_sec)

        # Near-field geometric path inputs (ECI satellite + antenna positions, UVW w),
        # matching what get_rfi_phase / the trajectory component uses. The fringe params
        # are derived from this path so f, fdot are consistent with the geometric phase
        # phi0 the component reads (a far-field fringe rate would be inconsistent).
        rfi_xyz_fit = np.asarray(get_satellite_positions(self.tles, times_jd_fit))
        ants_xyz_fit = np.asarray(itrs_to_gcrs_sf(self.ants_itrf, times_jd_fit))
        gsa = gast_deg(times_jd_fit)
        gh0 = (gsa - ra) % 360
        # itrf_to_uvw_numpy -> (n_time, n_ant, 3); take the w component per antenna.
        ants_w_fit = itrf_to_uvw_numpy(self.ants_itrf, gh0, dec)[:, :, 2].T  # (n_ant, n_tot)

        coeffs = fit_nearfield_fringe_freq_poly_numpy(
            t_fit_sec, freq_ref, rfi_xyz_fit, ants_xyz_fit, ants_w_fit,
            np.asarray(self.a1), np.asarray(self.a2), n_time,
        )
        K, fddot_max = size_subwindows(coeffs, dt, resid_tol, resid_A, k_max)

        dt_sub = dt / K
        centre_offsets = -dt / 2.0 + (np.arange(K) + 0.5) * dt_sub
        f_ref, fdot_ref = fringe_params_at_offsets(coeffs, centre_offsets)

        self.analytic_K = int(K)
        self.analytic_dt_sub = float(dt_sub)
        self.analytic_fddot_max = float(fddot_max)
        self.analytic_f = np.asarray(f_ref)          # (n_rfi, n_bl, n_time*K)
        self.analytic_fdot = np.asarray(fdot_ref)
        self.analytic_freq_scale = np.asarray(self.freqs, dtype=np.float64) / freq_ref
        # Window i uses shared global edges [i*K .. i*K + K].
        self.analytic_edge_gather = (
            np.arange(n_time)[:, None] * K + np.arange(K + 1)[None, :]
        ).astype(np.int32)

        print(
            f"\nAnalytic RFI-vis: K={K} sub-windows (dt_sub={dt_sub:.4g}s), "
            f"max |fddot|={fddot_max:.3g} Hz/s^2, fine grid {2 * n_time * K + 1} "
            f"vs oversample {n_time * self.n_int_time} samples\n"
        )

    def _set_freqs_times(self):

        ns = [self.n_freq, self.n_time]
        ss_factors = [self.n_int_freq, self.n_int_time]
        pad_factors = [
            self.args["rfi"]["freq_pad_factor"],
            self.args["rfi"]["time_pad_factor"],
        ]
        # domain_ss is jax-based, so under jax_enable_x64=False it builds the grids
        # in f32 internally. The real grids carry large magnitudes (freqs ~1e9 Hz,
        # and times_jd_fine ~2.4e6 JD) that lose all usable precision in f32. Since
        # domain_ss is affine in (x0, dx) (output = x0 + dx * normalised_grid), build
        # the normalised grid with x0=0, dx=1 (small, f32-safe) and apply the real
        # offset/scale in numpy f64.
        unit_freqs, unit_times = domain_ss(
            ns, [1.0, 1.0], [0.0, 0.0], ss_factors, pad_factors
        )
        self.freqs_fine = self.freqs[0] + self.chan_width * np.asarray(
            unit_freqs, dtype=np.float64
        )

        if getattr(self, "vis_method", "oversample") == "analytic":
            # Edge/centre grid: interleave the shared global sub-window edges (n_time*K+1)
            # with the sub-window centres (n_time*K) into a single fine grid, so the
            # existing trajectory (phase) and GP-envelope components fill both in one pass.
            # times_fine[0::2] = edges, times_fine[1::2] = centres.
            K = self.analytic_K
            dt = self.int_time
            dt_sub = self.analytic_dt_sub
            n_e = self.n_time * K + 1
            n_c = self.n_time * K
            t0 = self.times[0] - dt / 2.0
            edges = t0 + np.arange(n_e) * dt_sub
            centres = t0 + (np.arange(n_c) + 0.5) * dt_sub
            times_fine = np.empty(n_e + n_c, dtype=np.float64)
            times_fine[0::2] = edges
            times_fine[1::2] = centres
            self.times_fine = times_fine
        else:
            self.times_fine = self.times[0] + self.int_time * np.asarray(
                unit_times, dtype=np.float64
            )

        self.n_freq_fine = len(self.freqs_fine)
        self.n_time_fine = len(self.times_fine)
        self.times_jd_fine = self.times_jd[0] + secs_to_days(self.times_fine)

    def get_orbital_elements(self, norad_ids: List[int], extra_tle_dir: Optional[str] = None):

        obs_epoch_jd = float(self.times_jd.mean())

        self.elements, self.epoch_jd, self.norad_ids, self.tles = (
            fetch_orbital_elements(obs_epoch_jd, norad_ids, extra_tle_dir=extra_tle_dir)
        )
        self.n_rfi = len(self.norad_ids)


class Model:

    def __init__(
        self,
        tab_config: TabConfig,
        component_list: List[str],
        likelihood: Callable = gaussian,
    ):

        self.noise = tab_config.noise
        self.likelihood = lambda pred, obs_data: likelihood(
            pred, obs_data, {"noise": tab_config.noise, "flags": tab_config.flags}
        )

        components = [C() for C in import_components(component_list)]
        self.components = components
        for comp in components:
            comp.setup(tab_config)

        init_params = [comp.init_params_base for comp in components]
        self.init_params = {k: v for d in init_params for k, v in d.items()}

        state = [comp.state_outputs for comp in components]
        self.state = {k: v for d in state for k, v in d.items()}

        self.constants = {}
        for comp in components:
            for key, value in comp.build_constants().items():
                self.constants[f"{comp.prefix}/{key}"] = value

        self.state["vis_ast"] = jnp.zeros_like(self.state["vis_obs"])
        self.state["vis_rfi"] = jnp.zeros_like(self.state["vis_obs"])

        self.state["rmse_ast"] = jnp.array([jnp.nan])
        self.state["rmse_rfi"] = jnp.array([jnp.nan])
        self.state["rmse_gains"] = jnp.array([jnp.nan])

        self.forward = self.build_forward()
        self.prob_model = self.build_prob_model()

    def build_forward(self):
        forwards = [comp.build_forward() for comp in self.components]

        def forward(params, state, constants):

            for sub_forward in forwards:
                state = sub_forward(params, state, constants)

            return state

        return forward

    def build_set_params(self):
        set_params_functions = [comp.build_set_params() for comp in self.components]

        def set_params():
            params = {}

            for set_params in set_params_functions:
                params = set_params(params)

            return params

        return set_params

    def build_prob_model(self):

        set_params = self.build_set_params()
        forward = self.forward
        likelihood = self.likelihood

        def prob_model(obs_data=None, state=None, constants=None):

            params = set_params()

            state = forward(params, state, constants)

            numpyro.deterministic("rfi_phase", state["rfi_phase"])
            numpyro.deterministic("rfi_A", state["rfi_A"])

            numpyro.deterministic("vis_rfi", state["vis_rfi"])
            numpyro.deterministic("vis_ast", state["vis_ast"])
            numpyro.deterministic("gains", state["gains"])
            numpyro.deterministic("vis_obs", state["vis_obs"])

            if obs_data is not None:
                likelihood(state["vis_obs"], obs_data)

            return state

        return prob_model
