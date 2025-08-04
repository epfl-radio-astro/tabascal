from jax import vmap, Array, jit
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist

from tabascal.dist import standard_normal
from tabascal.transform import affine_transform_full, affine_transform_diag

from tabsim.jax.coordinates import kepler_orbit_many

# Standard functions


@jit
def get_rfi_phase(rfi_xyz: Array, ants_uvw: Array, ants_xyz: Array, freqs: Array):
    """Calculate phase at each antenna for each RFI source

    Parameters
    ----------
    rfi_xyz: Array (n_src, n_time, 3)
        Positions of the RFI sources over time in the ECI frame in metres.
    ants_uvw: Array (n_ant, n_time, 3)
        UVW coordinates of the antennas in metres. Only the w-coordinate is used as this is the phase delay for a fringe-stopping interferometer.
    ants_xyz: Array (n_ant, n_time, 3)
        Positions of the antennas over time in the ECI frame in metres.
    freqs: Array (n_freq,)
        Observation frequencies in Hz.

    Returns
    -------
    phase: Array (n_src, n_ant, n_freq, n_time)
        Phase at each antenna for each source over time.
    """
    c = 299792458.0
    lamda = c / freqs[None, None, :, None]

    distances = jnp.linalg.norm(
        ants_xyz[None, :, None, :, :] - rfi_xyz[:, None, None, :, :], axis=-1
    )
    fringe_dist = ((distances + ants_uvw[None, :, None, :, -1]) / lamda) % 1

    phases = -2.0 * jnp.pi * fringe_dist

    return phases


@jit
def calculate_rfi_vis_fine(rfi_A, rfi_phase, a1, a2):

    # rfi_A is shape (n_rfi, n_ant, n_time_fine)
    # rfi_phase is shape (n_rfi, n_ant, n_time_fine)
    # a1 and a2 are shape (n_bl,)
    # rfi_vis_fine is shape (n_bl, n_time_fine)
    vis_rfi_fine = jnp.sum(
        rfi_A[:, a1]
        * jnp.conjugate(rfi_A[:, a2])
        * jnp.exp(rfi_phase[:, a1] - rfi_phase[:, a2]),
        axis=0,
    )

    return vis_rfi_fine


@jit
def apply_gains(gains, vis, a1, a2):

    vis_obs = gains[a1] * vis * jnp.conjugate(gains)[a2]

    return vis_obs


# Computations


def calculate_total_vis(config, state):

    state["vis"] = state["vis_rfi"] + state["vis_ast"]

    return state


def apply_linear_gains(config, state):

    state = calculate_total_vis(config, state)

    a1, a2 = config["a1"], config["a2"]

    # state["vis_obs"] = (
    #     state["gains"][a1] * state["vis"] * jnp.conjugate(state["gains"][a2])
    # )

    state["vis_obs"] = apply_gains(
        state["gains"], state["vis"], config["a1"], config["a2"]
    )

    # numpyro.deterministic("vis_obs", state["vis_obs"])

    return state


def apply_unitary_gains(config, state):

    state = calculate_total_vis(config, state)

    state["vis_obs"] = state["vis"]

    # numpyro.deterministic("vis_obs", state["vis_obs"])

    return state


# Static components


def fixed_kepler_orbit(config, state):

    state["rfi_xyz"] = kepler_orbit_many(
        config["times_jd"], config["epoch_jd"], config["kepler_elements"]
    )

    return state


def fixed_rfi_phase(config, state):

    state["rfi_phase"] = config["rfi_phase"]

    # numpyro.deterministic("rfi_phase", state["rfi_phase"])

    return state


def add_fixed_vis_rfi(config, state):

    state["vis_rfi"] = state["vis_rfi"] + config["vis_rfi_fixed"]

    return state


def add_fixed_vis_ast(config, state):

    state["vis_ast"] = state["vis_ast"] + config["vis_ast_fixed"]

    return state


def apply_fixed_gains(config, state):

    state["gains"] = config["gains_fixed"]

    # numpyro.deterministic("gains", state["gains"])

    state = apply_linear_gains(config, state)

    return state


def calculate_rfi_phase(config, state):

    # rfi_xyz is shape (n_rfi, n_time_fine, 3)
    # ants_uvw is shape (n_ant, n_time_fine, 3)
    # ants_xyz is shape (n_ant, n_time_fine, 3)
    # freqs is shape (n_freq,)

    # rfi_phase shape is (n_rfi, n_ant, n_time_fine)
    state["rfi_phase"] = get_rfi_phase(
        state["rfi_xyz"], config["ants_uvw"], config["ants_xyz"], config["freqs"]
    )

    # numpyro.deterministic("rfi_phase", state["rfi_phase"])

    return state


def riemann_averaging(config, state):

    vis_rfi_fine = calculate_rfi_vis_fine(
        state["rfi_A"], state["rfi_phase"], config["a1"], config["a2"]
    )

    new_shape = (config["n_bl"], config["n_freq"], config["n_time"], config["n_int"])

    # vis_rfi_fine is shape (n_bl, n_time_fine)
    # vis_rfi is shape (n_bl, n_time)
    vis_rfi = jnp.mean(jnp.reshape(vis_rfi_fine, new_shape), axis=-1)
    state["vis_rfi"] = state["vis_rfi"] + vis_rfi

    return state


# Parameterised components


@jit
def transform_orbit(rfi_orbit_base, L_rfi_orbit, mu_rfi_orbit, times_fine_jd, epoch_jd):

    elements = vmap(affine_transform_full)(rfi_orbit_base, L_rfi_orbit, mu_rfi_orbit)

    # rfi_xyz is shape (n_rfi, n_time_fine, 3)
    rfi_xyz = kepler_orbit_many(times_fine_jd, epoch_jd, elements)

    return rfi_xyz


def include_kepler_orbit(config, state):

    rfi_orbit_base = standard_normal("rfi_orbit_base", (config["n_rfi"], 6))

    # elements = vmap(affine_transform_full)(
    #     rfi_orbit_base, config["L_rfi_orbit"], config["mu_rfi_orbit"]
    # )

    # # rfi_xyz is shape (n_rfi, n_time_fine, 3)
    # state["rfi_xyz"] = kepler_orbit_many(
    #     config["times_fine_jd"], config["epoch_jd"], elements
    # )

    state["rfi_xyz"] = transform_orbit(
        rfi_orbit_base,
        config["L_rfi_orbit"],
        config["mu_rfi_orbit"],
        config["times_fine_jd"],
        config["epoch_jd"],
    )

    return state


def include_orbit_deviation(config, state):

    base = standard_normal(
        "rfi_orbit_dev", (config["n_rfi"], config["n_orbit_dev_times"], 3)
    )

    orbit_dev = vmap(
        vmap(affine_transform_full, (0, None, 0), 0),
        (2, None, 2),
    )(base, config["L_rfi_orbit_dev"], config["mu_rfi_orbit_dev"])

    # d_rfi_xyz shape is (n_rfi, n_time_fine, 3)
    d_rfi_xyz = vmap(jnp.dot)(config["resample_orbit_dev"], orbit_dev)

    state["rfi_xyz"] = state["rfi_xyz"] + d_rfi_xyz

    return state


def include_rfi_real(config, state):

    rfi_shape = (config["n_rfi"], config["n_ant"], config["n_rfi_times"])

    rfi_A_base = standard_normal("rfi_A_induce_base", rfi_shape)

    # rfi_A_induce is shape (n_rfi, n_ant, n_rfi_time)
    rfi_A_induce = vmap(vmap(affine_transform_full, (0, None, 0), 0), (1, None, 1), 1)(
        rfi_A_base, config["L_rfi_A"], config["mu_rfi_A"]
    )

    # rfi_A is shape (n_rfi, n_ant, n_time_fine)
    state["rfi_A"] = vmap(vmap(jnp.dot, (None, 0), 0), (None, 1), 1)(
        config["resample_rfi"], rfi_A_induce
    )

    # numpyro.deterministic("rfi_A_induce", rfi_A_induce)

    return state


@jit
def transform_rfi_signal(rfi_A_base, L_rfi_A, mu_rfi_A, resample_rfi):

    rfi_A_induce = vmap(
        vmap(vmap(affine_transform_full, (0, None, 0), 0), (1, None, 1), 1),
        (2, None, 2),
        2,
    )(rfi_A_base, L_rfi_A, mu_rfi_A)

    # numpyro.deterministic("rfi_A_induce", rfi_A_induce)

    # rfi_A_induce is shape (n_rfi, n_ant, n_freq, n_time_fine)
    # resample_rfi is shape (n_time_fine, n_rfi_time)
    rfi_A = vmap(vmap(vmap(jnp.dot, (None, 0), 0), (None, 1), 1), (None, 2), 2)(
        resample_rfi, rfi_A_induce
    )

    return rfi_A


def include_rfi_complex(config, state):

    rfi_shape = (
        config["n_rfi"],
        config["n_ant"],
        config["n_freq"],
        config["n_rfi_times"],
    )

    rfi_r_base = standard_normal("rfi_r_induce_base", rfi_shape)
    rfi_i_base = standard_normal("rfi_i_induce_base", rfi_shape)

    # rfi_A_base is shape (n_rfi, n_ant, n_freq, n_rfi_time)
    rfi_A_base = rfi_r_base + 1.0j * rfi_i_base

    # # rfi_A_base is shape (n_rfi, n_ant, n_freq, n_rfi_time)
    # # mu_rfi_A is shape (n_rfi, n_ant, n_freq, n_rfi_time)
    # # L_rfi_A is shape (n_rfi_time, n_rfi_time)
    # rfi_A_induce = vmap(
    #     vmap(vmap(affine_transform_full, (0, None, 0), 0), (1, None, 1), 1),
    #     (2, None, 2),
    #     2,
    # )(rfi_A_base, config["L_rfi_A"], config["mu_rfi_A"])

    # # numpyro.deterministic("rfi_A_induce", rfi_A_induce)

    # # rfi_A_induce is shape (n_rfi, n_ant, n_freq, n_time_fine)
    # # resample_rfi is shape (n_time_fine, n_rfi_time)
    # state["rfi_A"] = vmap(
    #     vmap(vmap(jnp.dot, (None, 0), 0), (None, 1), 1), (None, 2), 2
    # )(config["resample_rfi"], rfi_A_induce)

    state["rfi_A"] = transform_rfi_signal(
        rfi_A_base, config["L_rfi_A"], config["mu_rfi_A"], config["resample_rfi"]
    )

    return state


def include_ast_k(config, state):

    ast_k_shape = (config["n_bl"], config["n_freq"], config["n_k_ast"])
    ast_pad = config["ast_pad"]

    ast_k_r_base = standard_normal("ast_k_r_base", ast_k_shape)
    ast_k_i_base = standard_normal("ast_k_i_base", ast_k_shape)

    ast_k_base = ast_k_r_base + 1.0j * ast_k_i_base

    ast_k = config["sigma_ast_k"] * ast_k_base + config["mu_ast_k"]

    # vis_ast_padded = jnp.fft.ifft2(ast_k, axes=(1, 2))
    vis_ast_padded = jnp.fft.ifft(ast_k, axis=2)

    state["vis_ast"] = state["vis_ast"] + vis_ast_padded[:, :, ast_pad:-ast_pad]

    return state


@jit
def transform_gains(
    g_amp_base,
    mu_g_amp,
    L_g_amp,
    resample_g_amp,
    g_phase_base,
    mu_g_phase,
    L_g_phase,
    resample_g_phase,
):

    n_freq = g_amp_base.shape[1]
    n_time = resample_g_amp.shape[0]

    g_amp_induce = vmap(vmap(affine_transform_full, (0, None, 0)), (1, None, 1), 1)(
        g_amp_base, L_g_amp, mu_g_amp
    )
    g_phase_induce = vmap(vmap(affine_transform_full, (0, None, 0)), (1, None, 1), 1)(
        g_phase_base, L_g_phase, mu_g_phase
    )

    g_amp = vmap(vmap(jnp.dot, (None, 0), 0), (None, 1), 1)(
        resample_g_amp, g_amp_induce
    )
    g_phase = vmap(vmap(jnp.dot, (None, 0), 0), (None, 1), 1)(
        resample_g_phase, g_phase_induce
    )
    g_phase = jnp.concatenate([g_phase, jnp.zeros((1, n_freq, n_time))], axis=0)

    gains = g_amp * jnp.exp(1.0j * g_phase)

    return gains


def include_gains_gp(config, state):

    n_ant, n_freq, n_time, n_g_times = (
        config["n_ant"],
        config["n_freq"],
        config["n_time"],
        config["n_g_times"],
    )

    g_amp_base = standard_normal("g_amp_induce_base", (n_ant, n_freq, n_g_times))
    g_phase_base = standard_normal(
        "g_phase_induce_base", (n_ant - 1, n_freq, n_g_times)
    )

    g_amp_induce = vmap(vmap(affine_transform_full, (0, None, 0)), (1, None, 1), 1)(
        g_amp_base, config["L_g_amp"], config["mu_g_amp"]
    )
    g_phase_induce = vmap(vmap(affine_transform_full, (0, None, 0)), (1, None, 1), 1)(
        g_phase_base, config["L_g_phase"], config["mu_g_phase"]
    )

    g_amp = vmap(vmap(jnp.dot, (None, 0), 0), (None, 1), 1)(
        config["resample_g_amp"], g_amp_induce
    )
    g_phase = vmap(vmap(jnp.dot, (None, 0), 0), (None, 1), 1)(
        config["resample_g_phase"], g_phase_induce
    )
    g_phase = jnp.concatenate([g_phase, jnp.zeros((1, n_freq, n_time))], axis=0)

    state["gains"] = g_amp * jnp.exp(1.0j * g_phase)

    # state["gains"] = transform_gains(
    #     g_amp_base,
    #     config["mu_g_amp"],
    #     config["L_g_amp"],
    #     config["resample_g_amp"],
    #     g_phase_base,
    #     config["mu_g_phase"],
    #     config["L_g_phase"],
    #     config["resample_g_phase"],
    # )

    state = apply_linear_gains(config, state)

    return state


component_registry = {
    #  RFI phase
    "fixed_rfi_phase": fixed_rfi_phase,
    "fixed_orbit": fixed_kepler_orbit,
    "kepler_orbit": include_kepler_orbit,
    "orbit_deviation": include_orbit_deviation,
    "calc_rfi_phase": calculate_rfi_phase,
    # RFI A
    "rfi_real": include_rfi_real,
    "rfi_complex": include_rfi_complex,
    # RFI vis
    "riemann_avg": riemann_averaging,
    "add_fixed_rfi": add_fixed_vis_rfi,
    # AST vis
    "ast_fft": include_ast_k,
    "add_fixed_ast": add_fixed_vis_ast,
    # Gains
    "unitary_gains": apply_unitary_gains,
    "fixed_gains": apply_fixed_gains,
    "gains_gp": include_gains_gp,
}

rfi_phase_options = {
    "fixed_phase": ["fixed_rfi_phase"],
    "fixed_orbit": ["fixed_orbit", "calc_rfi_phase"],
    "kepler_orbit": ["kepler_orbit", "calc_rfi_phase"],
    "orbit_deviation": ["fixed_orbit", "orbit_deviation", "calc_rfi_phase"],
    "kepler_orbit_devation": ["kepler_orbit", "orbit_deviation", "calc_rfi_phase"],
}

rfi_A_options = {
    "rfi_real": ["rfi_real"],
    "rfi_complex": ["rfi_complex"],
}

rfi_vis_options = {
    "standard": ["riemann_avg"],
    "add_rfi": ["riemann_avg", "add_fixed_rfi"],
    "fixed": ["add_fixed_rfi"],
}

ast_vis_options = {
    "standard": ["ast_fft"],
    "add_ast": ["ast_fft", "add_fixed_ast"],
    "fixed": ["add_fixed_ast"],
}

gain_options = {
    "calibrated": ["unitary_gains"],
    "fixed": ["fixed_gains"],
    "standard": ["gains_gp"],
}

# components = {
#     "rfi": {
#         "phase": "kepler_orbit",
#         "signal": "rfi_complex",
#         "vis": "standard",
#     },
#     "ast": "standard",
#     "gains" : "standard",
# }


def build_model(config):

    components = [
        *rfi_phase_options[config["enabled_components"]["rfi"]["phase"]],
        *rfi_A_options[config["enabled_components"]["rfi"]["signal"]],
        *rfi_vis_options[config["enabled_components"]["rfi"]["vis"]],
        *ast_vis_options[config["enabled_components"]["ast"]],
        *gain_options[config["enabled_components"]["gains"]],
    ]
    components = [component_registry[name] for name in components]

    def model(obs_data=None):

        shape = (config["n_bl"], config["n_freq"], config["n_time"])
        state = {
            "vis_rfi": jnp.zeros(shape, dtype=jnp.complex128),
            "vis_ast": jnp.zeros(shape, dtype=jnp.complex128),
        }

        for op in components:
            state = op(config, state)

        # for key, value in state.items():
        #     numpyro.deterministic(key, value)

        numpyro.deterministic("vis_obs", state["vis_obs"])

        if obs_data is not None:
            vis_obs_ri = jnp.stack(
                [state["vis_obs"].real, state["vis_obs"].imag], axis=0
            )
            obs_data_ri = jnp.stack([obs_data.real, obs_data.imag], axis=0)
            numpyro.sample(
                "vis_obs_ri", dist.Normal(vis_obs_ri, config["noise"]), obs=obs_data_ri
            )

        return state

    return model


# def build_model(config):

#     def model(obs_data=None):

#         shape = (config["n_bl"], config["n_freq"], config["n_time"])
#         state = {
#             "vis_rfi": jnp.zeros(shape, dtype=jnp.complex128),
#             "vis_ast": jnp.zeros(shape, dtype=jnp.complex128),
#         }

#         for name in rfi_phase_options[config["enabled_components"]["rfi"]["phase"]]:
#             fn = component_registry[name]
#             state = fn(config, state)

#         for name in rfi_A_options[config["enabled_components"]["rfi"]["signal"]]:
#             fn = component_registry[name]
#             state = fn(config, state)

#         for name in rfi_vis_options[config["enabled_components"]["rfi"]["vis"]]:
#             fn = component_registry[name]
#             state = fn(config, state)

#         for name in ast_vis_options[config["enabled_components"]["ast"]]:
#             fn = component_registry[name]
#             state = fn(config, state)

#         for name in gain_options[config["enabled_components"]["gains"]]:
#             fn = component_registry[name]
#             state = fn(config, state)

#         for key, value in state.items():
#             numpyro.deterministic(key, value)

#         if obs_data is not None:
#             vis_obs_ri = jnp.stack(
#                 [state["vis_obs"].real, state["vis_obs"].imag], axis=0
#             )
#             obs_data_ri = jnp.stack([obs_data.real, obs_data.imag], axis=0)
#             numpyro.sample(
#                 "vis_obs_ri", dist.Normal(vis_obs_ri, config["noise"]), obs=obs_data_ri
#             )

#     return model
