# from tabascal.component_functions import get_rfi_phase
# from tabascal.gp import cholesky, resampling_kernel, get_times
# from tabascal.tab_tools import (
#     # read_ms,
#     get_ast_fringe_rate,
#     estimate_sampling,
#     get_tles,
#     pow_spec,
# )

from tabascal.tab_tools_new import read_ms
from tabascal.components.trajectory import fetch_orbital_elements

from tabascal.components import Component

import jax.numpy as jnp
from jax import vmap

import numpy as np

# from tabsim.config import JD0, yaml_load
# from tabsim.tle import get_tles_by_id
from tabsim.jax.coordinates import (
    secs_to_days,
    mjd_to_jd,
    itrf_to_uvw,
    itrf_to_xyz,
    kepler_orbit_many,
    kepler_orbit_fisher,
    gmsa_from_jd,
    calculate_fringe_frequency,
    jd_to_mjd,
)

# from tabsim.jax.interferometry import int_sample_times
from tabsim.dask.interferometry import int_sample_times

import numpyro.distributions as dist
import numpyro

from typing import Callable


# class tabConfigBuilder:
#     """Configuration parameters for tabascal method"""

#     def __init__(self, config: dict, ms_path: str):

#         # self.config = config
#         self.ms_path = ms_path

#         self.read_ms_params(
#             config["data"]["freq"],
#             config["data"]["corr"],
#             config["data"]["data_col"],
#         )

#         # self.populate_model_config(config)

#     def populate_model_config(self, config):

#         components = config["enabled_components"]

#         init_params = {}

#         # Basic model config
#         model_config = {
#             "enabled_components": config["enabled_components"],
#             "n_ant": self.n_ant,
#             "n_bl": self.n_bl,
#             "n_time": self.n_time,
#             "n_freq": self.n_freq,
#             "n_corr": self.n_corr,
#             "int_time": self.int_time,
#             "times": self.times,
#             "times_jd": self.times_jd,
#             "freqs": self.freqs,
#             "noise": self.noise,
#             "a1": self.a1,
#             "a2": self.a2,
#             "vis_obs": self.vis_obs,
#         }

#         # Dynamically populate the model config dictionary

#         # RFI Phase
#         if components["rfi"]["phase"] in [
#             "fixed_phase",
#             "fixed_orbit",
#             "kepler_orbit",
#             "orbit_deviation",
#             "kepler_orbit_devation",
#         ]:
#             self.get_orbital_elements(
#                 config["satellites"]["norad_ids"],
#                 config["satellites"]["spacetrack_path"],
#             )
#             self.estimate_rfi_sampling(config["rfi"]["n_int_factor"])
#             self.set_ants_xyz_uvw()
#             model_config.update(
#                 {
#                     "epoch_jd": self.epoch_jd,
#                     "times_fine_jd": self.times_jd_fine,
#                     "ants_uvw": self.ants_uvw,
#                     "ants_xyz": self.ants_xyz,
#                 }
#             )

#         if components["rfi"]["phase"] in ["fixed_phase"]:
#             self.set_rfi_phase()
#             model_config.update({"rfi_phase": self.rfi_phase})

#         if components["rfi"]["phase"] in [
#             "fixed_orbit",
#             "orbit_deviation",
#         ]:
#             model_config.update({"kepler_elements": self.elements})

#         if components["rfi"]["phase"] in [
#             "kepler_orbit",
#             "kepler_orbit_deviation",
#         ]:
#             self.set_kepler_orbit(config["satellites"]["ric_std"])
#             model_config.update(
#                 {
#                     "mu_rfi_orbit": self.mu_rfi_orbit,
#                     "L_rfi_orbit": self.L_rfi_orbit,
#                 }
#             )
#             init_params.update({"rfi_orbit_base": self.init_rfi_orbit_base})

#         # RFI Signal
#         if components["rfi"]["signal"] in ["rfi_real", "rfi_complex"]:
#             self.set_rfi_signal(config["rfi"]["var"], config["rfi"]["corr_time"])
#             model_config.update(
#                 {
#                     "n_rfi": self.n_rfi,
#                     "n_rfi_times": self.n_rfi_times,
#                     "n_int": self.n_int_samples,
#                     "mu_rfi_A": self.mu_rfi_A,
#                     "L_rfi_A": self.L_rfi_A,
#                     "resample_rfi": self.resample_rfi,
#                 }
#             )
#         if components["rfi"]["signal"] in ["rfi_real"]:
#             init_params.update(
#                 {
#                     "rfi_A_induce_base": self.init_rfi_A_base,
#                 }
#             )
#         if components["rfi"]["signal"] in ["rfi_complex"]:
#             init_params.update(
#                 {
#                     "rfi_r_induce_base": self.init_rfi_A_base.real,
#                     "rfi_i_induce_base": self.init_rfi_A_base.imag,
#                 }
#             )

#         # RFI Vis
#         # if components["rfi"]["vis"] in ["add_rfi"]:
#         #     model_config.update({"vis_rfi_fixed": self.fixed_vis_rfi})

#         # Ast Signal
#         if components["ast"] in ["standard", "add_ast"]:
#             self.set_ast_k(
#                 int(max([config["ast"]["pad_factor"] * self.n_time // 2, 1])),
#                 config["ast"]["pow_spec"]["P0"],
#                 config["ast"]["pow_spec"]["gamma"],
#                 config["ast"]["pow_spec"]["fov_deg"],
#             )
#             model_config.update(
#                 {
#                     "n_k_ast": self.n_ast_k,
#                     "ast_pad": self.ast_pad,
#                     "mu_ast_k": self.mu_ast_k,
#                     "sigma_ast_k": self.sigma_ast_k,
#                 }
#             )
#             init_params.update(
#                 {
#                     "ast_k_r_base": self.init_ast_k_base.real,
#                     "ast_k_i_base": self.init_ast_k_base.imag,
#                 }
#             )
#         # if components["ast"] in ["add_ast"]:
#         #     model_config.update({"vis_ast_fixed": self.fixed_vis_ast})

#         if components["gains"] in ["standard"]:
#             self.set_gains(
#                 config["gains"]["amp_mean"],
#                 config["gains"]["amp_std"],
#                 config["gains"]["corr_time"] * 60,
#                 config["gains"]["phase_mean"],
#                 config["gains"]["phase_std"],
#                 config["gains"]["corr_time"] * 60,
#             )
#             model_config.update(
#                 {
#                     "n_g_times": self.n_g_times,
#                     "mu_g_amp": self.mu_g_amp,
#                     "L_g_amp": self.L_g_amp,
#                     "resample_g_amp": self.resample_g_amp,
#                     "mu_g_phase": self.mu_g_phase,
#                     "L_g_phase": self.L_g_phase,
#                     "resample_g_phase": self.resample_g_phase,
#                 }
#             )
#             init_params.update(
#                 {
#                     "g_amp_induce_base": self.init_g_amp_base,
#                     "g_phase_induce_base": self.init_g_phase_base,
#                 }
#             )

#         # if components["gains"] in ["fixed_gains"]:
#         #     model_config.update({"gains_fixed": self.gains_fixed})

#         self.model_config = model_config
#         self.init_params = init_params

#         return model_config, init_params

#     def read_ms_params(self, freq: float, corr: str, data_col: str):

#         ms_params = read_ms(self.ms_path, freq, corr, data_col)

#         self.phase_centre = {"ra": ms_params["ra"], "dec": ms_params["dec"]}
#         self.dish_d = ms_params["dish_d"]
#         self.ants_itrf = ms_params["ants_itrf"]
#         self.vis_obs = ms_params["vis_obs"][
#             :, None, :
#         ]  # Add the frequency channel into the data
#         self.uvw = ms_params["uvw"]

#         self.n_ant = ms_params["n_ant"]
#         self.n_bl = ms_params["n_bl"]
#         self.n_time = ms_params["n_time"]
#         self.n_freq = ms_params["n_freq"]
#         self.n_corr = ms_params["n_corr"]

#         self.int_time = ms_params["int_time"]
#         self.times = ms_params["times"]
#         self.times_jd = mjd_to_jd(ms_params["times_mjd"])

#         self.freqs = ms_params["freqs"]
#         self.noise = ms_params["noise"]
#         self.a1 = ms_params["a1"]
#         self.a2 = ms_params["a2"]

#     def get_orbital_elements(self, norad_ids: list[int], spacetrack_path: str):

#         obs_epoch_jd = float(self.times_jd.mean())

#         self.elements, self.epoch_jd, self.norad_ids, tles = fetch_orbital_elements(
#             spacetrack_path, obs_epoch_jd, norad_ids
#         )
#         self.n_rfi = len(self.norad_ids)

#     def set_kepler_orbit(self, ric_std: float):

#         RIC_std = ric_std * jnp.array([73, 131, 54])
#         F_orbit = vmap(kepler_orbit_fisher, in_axes=(None, 0, 0, None))(
#             self.times_jd, self.epoch_jd, self.elements, RIC_std  # type: ignore
#         )
#         kepler_cov = vmap(jnp.linalg.inv)(F_orbit)

#         self.L_rfi_orbit = vmap(jnp.linalg.cholesky)(kepler_cov)
#         self.mu_rfi_orbit = self.elements

#         self.init_rfi_orbit = self.mu_rfi_orbit
#         self.init_rfi_orbit_base = vmap(jnp.linalg.solve)(
#             self.L_rfi_orbit, self.init_rfi_orbit - self.mu_rfi_orbit
#         )
#         # self.init_rfi_orbit_base = jnp.zeros((self.n_rfi, 6))

#     def estimate_rfi_sampling(self, n_int_factor: float):

#         jd_minute = 1 / (24 * 60)
#         times_jd_coarse = jnp.arange(
#             self.times_jd[0], self.times_jd[-1] + jd_minute, jd_minute
#         )

#         gh0 = (gmsa_from_jd(times_jd_coarse) - self.phase_centre["ra"]) % 360  # type: ignore

#         ants_u = itrf_to_uvw(self.ants_itrf, gh0, self.phase_centre["dec"])[:, :, 0]

#         rfi_xyz = kepler_orbit_many(times_jd_coarse, self.epoch_jd, self.elements)

#         fringe_freq = vmap(
#             calculate_fringe_frequency, (None, None, 0, None, None, None)
#         )(
#             jd_to_mjd(times_jd_coarse),
#             self.freqs,
#             rfi_xyz,
#             self.ants_itrf,
#             ants_u,
#             self.phase_centre["dec"],
#         )
#         self.max_fringe_freq = jnp.max(jnp.abs(fringe_freq))

#         self.max_rfi_vis = jnp.max(jnp.abs(self.vis_obs))

#         sample_freq = (
#             jnp.pi
#             * self.max_fringe_freq
#             * jnp.sqrt(self.max_rfi_vis / (6 * self.noise))
#         )
#         self.n_int_samples = int(jnp.ceil(n_int_factor * self.int_time * sample_freq))

#         self.times_fine = int_sample_times(self.times, self.n_int_samples).compute()
#         self.times_jd_fine = self.times_jd[0] + secs_to_days(self.times_fine)

#     def set_ants_xyz_uvw(self):

#         gsa = gmsa_from_jd(self.times_jd_fine) % 360
#         gh0 = (gsa - self.phase_centre["ra"]) % 360

#         self.ants_uvw = jnp.transpose(
#             itrf_to_uvw(self.ants_itrf, gh0, self.phase_centre["dec"]), axes=(1, 0, 2)
#         )
#         self.ants_xyz = jnp.transpose(itrf_to_xyz(self.ants_itrf, gsa), axes=(1, 0, 2))

#     def set_rfi_phase(self):

#         rfi_xyz = kepler_orbit_many(self.times_jd_fine, self.epoch_jd, self.elements)

#         self.rfi_phase = get_rfi_phase(
#             rfi_xyz, self.ants_uvw, self.ants_xyz, self.freqs
#         )

#     def set_rfi_signal(self, gp_var: float, gp_l: float):

#         # if gp_l is None:
#         #     gp_l =

#         if gp_var is None:
#             gp_var = jnp.max(jnp.abs(self.vis_obs))

#         self.rfi_times = get_times(self.times, gp_l)
#         self.n_rfi_times = len(self.rfi_times)
#         rfi_shape = (self.n_rfi, self.n_ant, self.n_freq, self.n_rfi_times)

#         self.mu_rfi_A = jnp.zeros(rfi_shape)
#         self.L_rfi_A = cholesky(self.rfi_times, gp_var, gp_l, 1e-8)
#         self.resample_rfi = resampling_kernel(
#             self.rfi_times,
#             self.times_fine,
#             gp_var,
#             gp_l,
#             1e-8,
#         )

#         self.init_rfi_A = self.mu_rfi_A
#         self.init_rfi_A_base = vmap(
#             vmap(
#                 vmap(jnp.linalg.solve, in_axes=(None, 0), out_axes=0),
#                 in_axes=(None, 1),
#                 out_axes=1,
#             ),
#             in_axes=(None, 2),
#             out_axes=2,
#         )(self.L_rfi_A, self.init_rfi_A - self.mu_rfi_A)

#     def set_ast_k(self, ast_pad: int, P0: float, gamma: float, fov_deg: float):

#         self.ast_pad = ast_pad
#         self.n_ast_k = self.n_time + 2 * ast_pad

#         self.k_ast = jnp.fft.fftfreq(self.n_ast_k, self.int_time)

#         if fov_deg:
#             eff_dish_d = 1.22 * 3e8 / (self.freqs * jnp.deg2rad(fov_deg))
#         else:
#             eff_dish_d = self.dish_d
#         self.ast_fr = get_ast_fringe_rate(self.uvw[:, :, :2], self.freqs, eff_dish_d)

#         sqrt_Pk = lambda k0: jnp.sqrt(pow_spec(self.k_ast, P0, k0, gamma))
#         self.sigma_ast_k = vmap(sqrt_Pk)(self.ast_fr)[:, None, :] * jnp.ones(
#             (self.n_bl, self.n_freq, self.n_ast_k)
#         )  # Add freq axis
#         self.mu_ast_k = jnp.zeros((self.n_bl, self.n_freq, self.n_ast_k), dtype=complex)

#         self.init_ast_k = self.mu_ast_k
#         self.init_ast_k_base = (self.init_ast_k - self.mu_ast_k) / self.sigma_ast_k

#     def set_gains(
#         self,
#         g_amp_mu: float,
#         g_amp_std: float,
#         g_amp_l: float,
#         g_phase_mu: float,
#         g_phase_std: float,
#         g_phase_l: float,
#     ):

#         gp_l = np.min([g_amp_l, g_phase_l])
#         self.g_times = get_times(self.times, gp_l)
#         self.n_g_times = len(self.g_times)

#         self.mu_g_amp = g_amp_mu * jnp.ones((self.n_ant, self.n_freq, self.n_g_times))
#         g_amp_var = (g_amp_std / 100) ** 2
#         self.L_g_amp = cholesky(self.g_times, g_amp_var, g_amp_l, 1e-8)
#         self.resample_g_amp = resampling_kernel(
#             self.g_times,
#             self.times,
#             g_amp_var,
#             g_amp_l,
#             1e-8,
#         )

#         self.mu_g_phase = g_phase_mu * jnp.ones(
#             (self.n_ant - 1, self.n_freq, self.n_g_times)
#         )
#         g_phase_var = jnp.deg2rad(g_phase_std) ** 2
#         self.L_g_phase = cholesky(self.g_times, g_phase_var, g_phase_l, 1e-8)
#         self.resample_g_phase = resampling_kernel(
#             self.g_times,
#             self.times,
#             g_phase_var,
#             g_phase_l,
#             1e-8,
#         )

#         self.init_g_amp = self.mu_g_amp
#         self.init_g_amp_base = vmap(vmap(jnp.linalg.solve, (None, 0), 0), (None, 1), 1)(
#             self.L_g_amp, self.init_g_amp - self.mu_g_amp
#         )

#         self.init_g_phase = self.mu_g_phase
#         self.init_g_phase_base = vmap(
#             vmap(jnp.linalg.solve, (None, 0), 0), (None, 1), 1
#         )(self.L_g_phase, self.init_g_phase - self.mu_g_phase)


#####################################################################################################


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

        self.get_orbital_elements(config["satellites"]["norad_ids"])

        self.estimate_rfi_sampling(config["rfi"]["n_int_factor"])

        self.args = config

    def read_ms_params(self, freq: float, corr: str, data_col: str):

        ms_params = read_ms(self.ms_path, freq, None, corr, data_col)

        self.phase_centre = {"ra": ms_params["ra"], "dec": ms_params["dec"]}
        self.dish_d = ms_params["dish_d"]
        self.ants_itrf = ms_params["ants_itrf"]
        self.vis_obs = ms_params["vis_obs"]
        # [
        #     :, None, :
        # ]  # Add the frequency channel into the data
        self.uvw = ms_params["uvw"]

        self.n_ant = ms_params["n_ant"]
        self.n_bl = ms_params["n_bl"]
        self.n_time = ms_params["n_time"]
        self.n_freq = ms_params["n_freq"]
        self.n_corr = ms_params["n_corr"]

        self.int_time = ms_params["int_time"]
        self.times = ms_params["times"]
        self.times_jd = mjd_to_jd(ms_params["times_mjd"] / (24 * 3600))

        self.freqs = ms_params["freqs"]
        self.noise = ms_params["noise"]
        self.a1 = ms_params["a1"]
        self.a2 = ms_params["a2"]

    def estimate_rfi_sampling(self, n_int_factor: float):

        jd_minute = 1 / (24 * 60)
        times_jd_coarse = jnp.arange(
            self.times_jd[0], self.times_jd[-1] + jd_minute, jd_minute
        )

        gh0 = (gmsa_from_jd(times_jd_coarse) - self.phase_centre["ra"]) % 360  # type: ignore

        ants_u = itrf_to_uvw(self.ants_itrf, gh0, self.phase_centre["dec"])[:, :, 0]

        rfi_xyz = kepler_orbit_many(times_jd_coarse, self.epoch_jd, self.elements)

        fringe_freq = vmap(
            calculate_fringe_frequency, (None, None, 0, None, None, None)
        )(
            jd_to_mjd(times_jd_coarse),
            jnp.max(self.freqs),
            rfi_xyz,
            self.ants_itrf,
            ants_u,
            self.phase_centre["dec"],
        )

        self.fringe_freqs = jnp.max(fringe_freq, axis=1)

        self.max_fringe_freq = jnp.max(jnp.abs(fringe_freq))

        self.max_rfi_vis = jnp.max(jnp.abs(self.vis_obs))

        sample_freq = (
            jnp.pi
            * self.max_fringe_freq
            * jnp.sqrt(self.max_rfi_vis / (6 * self.noise))
        )
        self.n_int_samples = int(jnp.ceil(n_int_factor * self.int_time * sample_freq))

        self.times_fine = int_sample_times(self.times, self.n_int_samples).compute()
        self.times_jd_fine = self.times_jd[0] + secs_to_days(self.times_fine)
        self.n_time_fine = len(self.times_fine)

    def get_orbital_elements(self, norad_ids: list[int]):

        obs_epoch_jd = float(self.times_jd.mean())

        self.elements, self.epoch_jd, self.norad_ids, tles = fetch_orbital_elements(
            self.spacetrack_path, obs_epoch_jd, norad_ids
        )
        self.n_rfi = len(self.norad_ids)


from tabascal.imports import import_components
from tabascal.components.likelihood import gaussian


class Model:

    def __init__(
        self,
        config: TabConfig,
        component_list: list[str],
        likelihood: Callable = gaussian,
    ):

        self.noise = config.noise
        self.likelihood = lambda pred, obs_data: likelihood(
            pred, obs_data, {"noise": self.noise}
        )

        components = [C() for C in import_components(component_list)]
        self.components = components
        for comp in components:
            comp.setup(config)

        init_params = [comp.init_params_base for comp in components]
        self.init_params = {k: v for d in init_params for k, v in d.items()}

        state = [comp.state_outputs for comp in components]
        self.state = {k: v for d in state for k, v in d.items()}

        self.state["rmse_ast"] = jnp.array([jnp.nan])
        self.state["rmse_rfi"] = jnp.array([jnp.nan])
        self.state["rmse_gains"] = jnp.array([jnp.nan])

        self.forward = self.build_forward()
        self.prob_model = self.build_prob_model()

    def build_forward(self):
        forwards = [comp.build_forward() for comp in self.components]

        def forward(params, state):

            for sub_forward in forwards:
                state = sub_forward(params, state)

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

        def prob_model(obs_data=None):

            params = set_params()
            state = self.state

            state = forward(params, state)

            numpyro.deterministic("vis_rfi", state["vis_rfi"])
            numpyro.deterministic("vis_ast", state["vis_ast"])
            numpyro.deterministic("gains", state["gains"])
            numpyro.deterministic("vis_obs", state["vis_obs"])

            numpyro.deterministic("rmse_rfi", state["rmse_rfi"])
            numpyro.deterministic("rmse_ast", state["rmse_ast"])
            numpyro.deterministic("rmse_gains", state["rmse_gains"])

            if obs_data is not None:
                self.likelihood(state["vis_obs"], obs_data)

            return state

        return prob_model
