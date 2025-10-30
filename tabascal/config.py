from tabascal.imports import import_components
from tabascal.components import Component
from tabascal.components.likelihood import gaussian
from tabascal.tab_tools import read_ms, fix_padding
from tabascal.components.trajectory import fetch_orbital_elements
from tabascal.interferometry import get_strides_and_idxs

import jax.numpy as jnp
from jax import vmap, Array

import numpy as np

# from tabsim.config import JD0, yaml_load
# from tabsim.tle import get_tles_by_id
from tabsim.jax.coordinates import (
    secs_to_days,
    mjd_to_jd,
    itrf_to_uvw,
    kepler_orbit_many,
    gmsa_from_jd,
    calculate_fringe_frequency,
    jd_to_mjd,
)

# from tabsim.jax.interferometry import int_sample_times
from tabsim.dask.interferometry import int_sample_times

import numpyro.distributions as dist
import numpyro

from typing import Callable, Optional


class TabConfig:
    """Configuration parameters for tabascal method"""

    def __init__(self, config: dict, ms_path: str):

        # self.config = config
        self.ms_path = ms_path
        self.spacetrack_path = config["satellites"]["spacetrack_path"]

        self.read_ms_params(
            config["data"]["freq"],
            config["data"]["corr"],
            config["data"]["data_col"],
        )
        self.set_noise(config["data"]["noise"])
        config = fix_padding(
            config, self.n_freq
        )  # Bad solution, should be fixed in fft_gp. Issue when using a single frequency channel.

        self.get_orbital_elements(config["satellites"]["norad_ids"])

        config["rfi"]["min_time_bins"] = 1
        config["rfi"]["max_time_bins"] = 30

        self.estimate_rfi_sampling(
            config["rfi"]["time_int_factor"],
            config["rfi"]["min_time_bins"],
            config["rfi"]["max_time_bins"],
            config["rfi"]["n_int_times"],
        )

        self.args = config

    def set_noise(self, noise):

        if noise:
            self.noise = noise

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
        self.times = ms_params["times"]
        self.times_jd = mjd_to_jd(ms_params["times_mjd"])

        self.freqs = ms_params["freqs"]
        self.noise = ms_params["noise"]
        self.a1 = ms_params["a1"]
        self.a2 = ms_params["a2"]

    def estimate_rfi_sampling(
        self,
        n_int_factor: float,
        min_time_bins: int,
        max_time_bins: int,
        n_int_times: Optional[int] = None,
    ):

        if n_int_times is None:

            jd_minute = 1 / (24 * 60)
            times_jd_coarse = np.arange(
                self.times_jd[0], self.times_jd[-1] + jd_minute, jd_minute
            )

            gh0 = (gmsa_from_jd(times_jd_coarse) - self.phase_centre["ra"]) % 360  # type: ignore

            ants_u = itrf_to_uvw(self.ants_itrf, gh0, self.phase_centre["dec"])[:, :, 0]

            rfi_xyz = kepler_orbit_many(times_jd_coarse, self.epoch_jd, self.elements)  # type: ignore

            # fringe_freq is shape (n_rfi, n_time_coarse, n_bl)
            fringe_freq = vmap(
                calculate_fringe_frequency, (None, None, 0, None, None, None)
            )(
                jd_to_mjd(times_jd_coarse),  # type: ignore
                jnp.max(self.freqs),  # type: ignore
                rfi_xyz,
                self.ants_itrf,
                ants_u,
                self.phase_centre["dec"],
            )

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
            n_int_times = np.ceil(n_int_factor * self.int_time * sample_freq_bl).astype(
                int
            )
            # print(bl_fr.max() * bl_fr.size / jnp.sum(bl_fr))

        self.time_sample_idxs, self.time_strides, self.n_int_time = (
            get_strides_and_idxs(n_int_times, min_time_bins, max_time_bins)
        )

        saving = (
            np.sum(
                [i.size / s for i, s in zip(self.time_sample_idxs, self.time_strides)]
            )
            / self.n_bl
        )

        print(f"New intermediate is {100*saving:.2f} % of original size")

        self.times_fine = int_sample_times(self.times, self.n_int_time).compute()
        self.times_jd_fine = self.times_jd[0] + secs_to_days(self.times_fine)
        self.n_time_fine = len(self.times_fine)

    def get_orbital_elements(self, norad_ids: list[int]):

        obs_epoch_jd = float(self.times_jd.mean())

        self.elements, self.epoch_jd, self.norad_ids, self.tles = (
            fetch_orbital_elements(self.spacetrack_path, obs_epoch_jd, norad_ids)
        )
        self.n_rfi = len(self.norad_ids)


class Model:

    def __init__(
        self,
        config: TabConfig,
        component_list: list[str],
        likelihood: Callable = gaussian,
    ):

        self.likelihood = lambda pred, obs_data: likelihood(
            pred, obs_data, {"noise": config.noise, "flags": config.flags}
        )

        self.components = self.build_components(config, component_list)
        self.init_params = self.build_init_params()
        self.state = self.build_state(config)
        self.forward = self.build_forward()
        self.prob_model = self.build_prob_model()

    def build_components(
        self, config: TabConfig, component_list: list[str]
    ) -> list[Component]:

        components = [C() for C in import_components(component_list)]

        for comp in components:
            comp.setup(config)

        return components

    def build_init_params(self) -> dict[str, Array]:

        init_params = [comp.init_params_base for comp in self.components]
        init_params = {k: v for d in init_params for k, v in d.items()}

        return init_params

    def build_state(self, config: TabConfig) -> dict[str, Array]:

        state = [comp.state_outputs for comp in self.components]
        state = {k: v for d in state for k, v in d.items()}

        state["rfi_phase"] = jnp.zeros(
            (config.n_rfi, config.n_ant, config.n_freq, config.n_time)
        )
        state["rfi_A"] = jnp.zeros(
            (config.n_rfi, config.n_ant, config.n_freq, config.n_time)
        )
        state["vis_rfi"] = jnp.zeros_like(state["vis_obs"])
        state["vis_ast"] = jnp.zeros_like(state["vis_obs"])

        state["rmse_ast"] = jnp.array([jnp.nan])
        state["rmse_rfi"] = jnp.array([jnp.nan])
        state["rmse_gains"] = jnp.array([jnp.nan])

        return state

    def build_forward(self) -> Callable:
        forwards = [comp.build_forward() for comp in self.components]

        def forward(params, state):

            for sub_forward in forwards:
                state = sub_forward(params, state)

            return state

        return forward

    def build_set_params(self) -> Callable:
        set_params_functions = [comp.build_set_params() for comp in self.components]

        def set_params():
            params = {}

            for set_params in set_params_functions:
                params = set_params(params)

            return params

        return set_params

    def build_prob_model(self) -> Callable:

        set_params = self.build_set_params()
        forward = self.forward

        def prob_model(obs_data=None):

            params = set_params()
            state = self.state

            state = forward(params, state)

            numpyro.deterministic("rfi_phase", state["rfi_phase"])
            numpyro.deterministic("rfi_A", state["rfi_A"])

            numpyro.deterministic("vis_rfi", state["vis_rfi"])
            numpyro.deterministic("vis_ast", state["vis_ast"])
            numpyro.deterministic("gains", state["gains"])
            numpyro.deterministic("vis_obs", state["vis_obs"])

            if obs_data is not None:
                self.likelihood(state["vis_obs"], obs_data)

            return state

        return prob_model
