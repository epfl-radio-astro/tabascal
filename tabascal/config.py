from tabascal.imports import import_components
from tabascal.components.likelihood import gaussian
from tabascal.tab_tools import read_ms, fix_padding
from tabascal.components.trajectory import fetch_orbital_elements, get_satellite_positions
from tabascal.tle import print_spacetrack_status, preflight_tle_check
from tabascal.interferometry import calculate_fringe_frequency, get_strides_and_idxs
from tabascal.fft_gp import domain_ss
from tabascal.time import secs_to_days, mjd_to_jd, jd_to_mjd
from tabascal.coordinates import itrf_to_uvw


import jax.numpy as jnp
from jax import vmap

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


loader = yaml.SafeLoader
loader.add_implicit_resolver(
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
    config = yaml.load(open(path), Loader=loader)
    return config


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
    except:
        raise IOError(f"Configuration file could not be loaded from {path}")

    
class TabConfig:
    """Configuration parameters for tabascal method"""

    def __init__(self, config: Dict, ms_path: str):

        # self.config = config
        self.args = config
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
        times_jd_coarse_whole = np.floor(times_jd_coarse)
        times_jd_coarse_frac = times_jd_coarse - times_jd_coarse_whole

        from sgp4jax._frames import _earth_orientation

        _, gast_rad = vmap(_earth_orientation)(times_jd_coarse_whole, times_jd_coarse_frac)
        gh0 = (jnp.rad2deg(gast_rad) - self.phase_centre["ra"]) % 360  # type: ignore

        ants_u = itrf_to_uvw(self.ants_itrf, gh0, self.phase_centre["dec"])[:, :, 0]

        rfi_xyz = get_satellite_positions(self.tles, times_jd_coarse)


        calc_fringe_freq = lambda _rfi_xyz: calculate_fringe_frequency(
            jd_to_mjd(times_jd_coarse),
            jnp.max(self.freqs),
            _rfi_xyz,
            self.ants_itrf,
            ants_u,
            self.phase_centre["dec"],
        )
        # fringe_freq is shape (n_rfi, n_time_coarse, n_bl)
        fringe_freq = vmap(calc_fringe_freq)(rfi_xyz)

        # # self.fringe_freqs is shape (n_rfi, n_bl)
        # self.fringe_freqs = jnp.max(jnp.abs(fringe_freq), axis=1)

        # self.max_fringe_freq = jnp.max(jnp.abs(fringe_freq))

        # self.max_rfi_vis = jnp.max(jnp.abs(self.vis_obs))

        # sample_freq = (
        #     jnp.pi
        #     * self.max_fringe_freq
        #     * jnp.sqrt(self.max_rfi_vis / (6 * self.noise))
        # )
        # self.n_int_time = int(jnp.ceil(n_int_factor * self.int_time * sample_freq))
        # self.n_int_time = max(1, self.n_int_time)

        self.max_rfi_vis = jnp.max(jnp.abs(self.vis_obs))
        sample_freq_bl = (
            jnp.pi
            * jnp.max(jnp.abs(fringe_freq), axis=(0, 1))
            * jnp.sqrt(self.max_rfi_vis / (6 * self.noise))
        )
        n_int_times = np.ceil(n_int_factor * self.int_time * sample_freq_bl).astype(int)
        # print(bl_fr.max() * bl_fr.size / jnp.sum(bl_fr))

        # time_sample_idxs and time_strides are only used in RiemannVisTimeFreqVariable
        self.time_sample_idxs, self.time_strides, self.n_int_time = (
            get_strides_and_idxs(n_int_times, min_time_bins, max_time_bins)
        )

    def _set_freqs_times(self):

        ns = [self.n_freq, self.n_time]
        dxs = [self.chan_width, self.int_time]
        x0s = [self.freqs[0], self.times[0]]
        ss_factors = [self.n_int_freq, self.n_int_time]
        pad_factors = [
            self.args["rfi"]["freq_pad_factor"],
            self.args["rfi"]["time_pad_factor"],
        ]
        self.freqs_fine, self.times_fine = domain_ss(ns, dxs, x0s, ss_factors, pad_factors)
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
