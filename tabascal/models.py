import jax.numpy as jnp
from jax import jit, vmap
from jax.flatten_util import ravel_pytree as flatten
from jax.tree_util import tree_map
import numpyro
import numpyro.distributions as dist
from tabascal.dist import MVN, Normal
from tabascal.vis import (
    get_ast_vis_fft,
    get_ast_vis_fft_padded,
    get_ast_vis,
    get_ast_vis1,
    get_ast_vis11,
    get_ast_vis2,
    get_ast_vis3,
    # get_rfi_vis_compressed,
    get_rfi_vis_compressed_ri,
    get_rfi_vis_full,
    get_rfi_vis_full_otf,
    get_rfi_vis_full_otf_fft,
    get_rfi_vis3,
    get_rfi_vis_fft2,
    # get_obs_vis,
    # get_obs_vis1,
    get_obs_vis_gains_all,
    get_obs_vis_gains_ast,
    get_gains,
    get_gains_mean,
    get_gains_straight,
    rmse,
    get_rfi_phase_from_orbit,
)
from tabascal.transform import affine_transform_full, affine_transform_diag

from functools import partial
from frozendict import frozendict

import numpy as np


@jit
def fixed_orbit_rfi_compressed_fft_standard_model(params, args):
    a1 = args["a1"]
    a2 = args["a2"]

    rfi_r = vmap(affine_transform_full, in_axes=(0, None, 0))(
        params["rfi_r_induce_base"], args["L_RFI"], args["mu_rfi_r"]
    )
    rfi_i = vmap(affine_transform_full, in_axes=(0, None, 0))(
        params["rfi_i_induce_base"], args["L_RFI"], args["mu_rfi_i"]
    )
    g_amp = vmap(affine_transform_full, in_axes=(0, None, 0))(
        params["g_amp_induce_base"], args["L_G_amp"], args["mu_G_amp"]
    )
    g_phase = vmap(affine_transform_full, in_axes=(0, None, 0))(
        params["g_phase_induce_base"], args["L_G_phase"], args["mu_G_phase"]
    )
    ast_k_r = vmap(affine_transform_diag, in_axes=(0, 0, 0))(
        params["ast_k_r_base"], args["sigma_ast_k"], args["mu_ast_k_r"]
    )
    ast_k_i = vmap(affine_transform_diag, in_axes=(0, 0, 0))(
        params["ast_k_i_base"], args["sigma_ast_k"], args["mu_ast_k_i"]
    )
    vis_rfi = get_rfi_vis_compressed_ri(rfi_r, rfi_i, args["rfi_kernel"], a1, a2)
    vis_ast = get_ast_vis_fft(ast_k_r, ast_k_i)
    gains = get_gains_straight(g_amp, g_phase, args["g_times"], args["times"])

    vis_obs = get_obs_vis_gains_ast(vis_ast, vis_rfi, gains, a1, a2)

    return vis_obs, (vis_rfi, vis_ast, gains)


@partial(jit, static_argnums=(1,))
def fixed_orbit_rfi_full_fft_standard_model(params, static_args, array_args):
    a1 = array_args["a1"]
    a2 = array_args["a2"]

    rfi_r = vmap(vmap(affine_transform_full, (0, None, 0), 0), (1, None, 1), 1)(
        params["rfi_r_induce_base"], array_args["L_RFI"], array_args["mu_rfi_r"]
    )
    rfi_i = vmap(vmap(affine_transform_full, (0, None, 0), 0), (1, None, 1), 1)(
        params["rfi_i_induce_base"], array_args["L_RFI"], array_args["mu_rfi_i"]
    )
    g_amp = vmap(affine_transform_full, in_axes=(0, None, 0))(
        params["g_amp_induce_base"], array_args["L_G_amp"], array_args["mu_G_amp"]
    )
    g_phase = vmap(affine_transform_full, in_axes=(0, None, 0))(
        params["g_phase_induce_base"], array_args["L_G_phase"], array_args["mu_G_phase"]
    )
    ast_k_r = vmap(affine_transform_diag, in_axes=(0, 0, 0))(
        params["ast_k_r_base"], array_args["sigma_ast_k"], array_args["mu_ast_k_r"]
    )
    ast_k_i = vmap(affine_transform_diag, in_axes=(0, 0, 0))(
        params["ast_k_i_base"], array_args["sigma_ast_k"], array_args["mu_ast_k_i"]
    )
    rfi_A = rfi_r + 1.0 * rfi_i
    vis_rfi = get_rfi_vis_full(
        rfi_A,
        static_args,
        array_args,
    )
    vis_ast = get_ast_vis_fft(ast_k_r, ast_k_i)
    gains = get_gains_straight(
        g_amp, g_phase, array_args["g_times"], array_args["times"]
    )

    vis_obs = get_obs_vis_gains_all(vis_ast, vis_rfi, gains, a1, a2)
    # vis_obs = get_obs_vis_gains_ast(vis_ast, vis_rfi, gains, a1, a2)

    return vis_obs, (vis_rfi, vis_ast, gains, rfi_A)


@partial(jit, static_argnums=(1,))
def fixed_orbit_rfi_full_fft_standard_padded_model(params, args, array_args):
    a1 = array_args["a1"]
    a2 = array_args["a2"]

    rfi_r = vmap(vmap(affine_transform_full, (0, None, 0), 0), (1, None, 1), 1)(
        params["rfi_r_induce_base"], array_args["L_RFI"], array_args["mu_rfi_r"]
    )
    rfi_i = vmap(vmap(affine_transform_full, (0, None, 0), 0), (1, None, 1), 1)(
        params["rfi_i_induce_base"], array_args["L_RFI"], array_args["mu_rfi_i"]
    )
    g_amp = vmap(affine_transform_full, in_axes=(0, None, 0))(
        params["g_amp_induce_base"], array_args["L_G_amp"], array_args["mu_G_amp"]
    )
    g_phase = vmap(affine_transform_full, in_axes=(0, None, 0))(
        params["g_phase_induce_base"], array_args["L_G_phase"], array_args["mu_G_phase"]
    )
    ast_k_r = vmap(affine_transform_diag, in_axes=(0, 0, 0))(
        params["ast_k_r_base"], array_args["sigma_ast_k"], array_args["mu_ast_k_r"]
    )
    ast_k_i = vmap(affine_transform_diag, in_axes=(0, 0, 0))(
        params["ast_k_i_base"], array_args["sigma_ast_k"], array_args["mu_ast_k_i"]
    )
    rfi_A = rfi_r + 1.0 * rfi_i
    vis_rfi = get_rfi_vis_full(
        rfi_A,
        args,
        array_args,
    )
    vis_ast = get_ast_vis_fft_padded(ast_k_r, ast_k_i, args["ast_pad"])
    gains = get_gains_straight(
        g_amp, g_phase, array_args["g_times"], array_args["times"]
    )

    vis_obs = get_obs_vis_gains_all(vis_ast, vis_rfi, gains, a1, a2)
    # vis_obs = get_obs_vis_gains_ast(vis_ast, vis_rfi, gains, a1, a2)

    return vis_obs, (vis_rfi, vis_ast, gains, rfi_A)


@jit
def fixed_orbit_rfi_full_fft_standard_model_otf(params, args):
    a1 = args["a1"]
    a2 = args["a2"]

    rfi_r = vmap(vmap(affine_transform_full, (0, None, 0), 0), (1, None, 1), 1)(
        params["rfi_r_induce_base"], args["L_RFI"], args["mu_rfi_r"]
    )
    rfi_i = vmap(vmap(affine_transform_full, (0, None, 0), 0), (1, None, 1), 1)(
        params["rfi_i_induce_base"], args["L_RFI"], args["mu_rfi_i"]
    )
    g_amp = vmap(affine_transform_full, in_axes=(0, None, 0))(
        params["g_amp_induce_base"], args["L_G_amp"], args["mu_G_amp"]
    )
    g_phase = vmap(affine_transform_full, in_axes=(0, None, 0))(
        params["g_phase_induce_base"], args["L_G_phase"], args["mu_G_phase"]
    )
    ast_k_r = vmap(affine_transform_diag, in_axes=(0, 0, 0))(
        params["ast_k_r_base"], args["sigma_ast_k"], args["mu_ast_k_r"]
    )
    ast_k_i = vmap(affine_transform_diag, in_axes=(0, 0, 0))(
        params["ast_k_i_base"], args["sigma_ast_k"], args["mu_ast_k_i"]
    )
    rfi_A = rfi_r + 1.0 * rfi_i
    vis_rfi = get_rfi_vis_full_otf(
        rfi_A,
        args,
    )
    vis_ast = get_ast_vis_fft(ast_k_r, ast_k_i)
    gains = get_gains_straight(g_amp, g_phase, args["g_times"], args["times"])

    vis_obs = get_obs_vis_gains_all(vis_ast, vis_rfi, gains, a1, a2)
    # vis_obs = get_obs_vis_gains_ast(vis_ast, vis_rfi, gains, a1, a2)

    return vis_obs, (vis_rfi, vis_ast, gains, rfi_A)


##################################################################################################


@partial(jit, static_argnums=(1,))
def fixed_orbit_rfi_full_fft_standard_model_otf_fft(params, args, array_args):
    a1 = array_args["a1"]
    a2 = array_args["a2"]

    rfi_r = vmap(vmap(affine_transform_full, (0, None, 0), 0), (1, None, 1), 1)(
        params["rfi_r_induce_base"], array_args["L_RFI"], array_args["mu_rfi_r"]
    )
    rfi_i = vmap(vmap(affine_transform_full, (0, None, 0), 0), (1, None, 1), 1)(
        params["rfi_i_induce_base"], array_args["L_RFI"], array_args["mu_rfi_i"]
    )
    g_amp = vmap(affine_transform_full, in_axes=(0, None, 0))(
        params["g_amp_induce_base"], array_args["L_G_amp"], array_args["mu_G_amp"]
    )
    g_phase = vmap(affine_transform_full, in_axes=(0, None, 0))(
        params["g_phase_induce_base"], array_args["L_G_phase"], array_args["mu_G_phase"]
    )
    ast_k_r = vmap(affine_transform_diag, in_axes=(0, 0, 0))(
        params["ast_k_r_base"], array_args["sigma_ast_k"], array_args["mu_ast_k_r"]
    )
    ast_k_i = vmap(affine_transform_diag, in_axes=(0, 0, 0))(
        params["ast_k_i_base"], array_args["sigma_ast_k"], array_args["mu_ast_k_i"]
    )
    rfi_A = rfi_r + 1.0 * rfi_i
    vis_rfi = get_rfi_vis_full_otf_fft(
        rfi_A,
        args,
        array_args,
    )
    vis_ast = get_ast_vis_fft(ast_k_r, ast_k_i)
    gains = get_gains_straight(
        g_amp, g_phase, array_args["g_times"], array_args["times"]
    )

    vis_obs = get_obs_vis_gains_all(vis_ast, vis_rfi, gains, a1, a2)
    # vis_obs = get_obs_vis_gains_ast(vis_ast, vis_rfi, gains, a1, a2)

    return vis_obs, (vis_rfi, vis_ast, gains, rfi_A)


##################################################################################################


@jit
def fixed_orbit_rfi_all_fft_standard_model(params, args):
    a1 = args["a1"]
    a2 = args["a2"]

    rfi_r = vmap(affine_transform_diag, in_axes=(0, None, 0))(
        params["rfi_r_induce_base"], args["sigma_rfi_k"], args["mu_rfi_r"]
    )
    rfi_i = vmap(affine_transform_diag, in_axes=(0, None, 0))(
        params["rfi_i_induce_base"], args["sigma_rfi_k"], args["mu_rfi_i"]
    )
    g_amp = vmap(affine_transform_full, in_axes=(0, None, 0))(
        params["g_amp_induce_base"], args["L_G_amp"], args["mu_G_amp"]
    )
    g_phase = vmap(affine_transform_full, in_axes=(0, None, 0))(
        params["g_phase_induce_base"], args["L_G_phase"], args["mu_G_phase"]
    )
    ast_k_r = vmap(affine_transform_diag, in_axes=(0, 0, 0))(
        params["ast_k_r_base"], args["sigma_ast_k"], args["mu_ast_k_r"]
    )
    ast_k_i = vmap(affine_transform_diag, in_axes=(0, 0, 0))(
        params["ast_k_i_base"], args["sigma_ast_k"], args["mu_ast_k_i"]
    )
    vis_rfi = get_rfi_vis_fft2(
        rfi_r + 1.0j * rfi_i,
        a1,
        a2,
        args["rfi_phase"],
        args["times_fine"],
        args["k_pad"],
        args["times"],
    )
    vis_ast = get_ast_vis_fft(ast_k_r, ast_k_i)
    gains = get_gains_straight(g_amp, g_phase, args["g_times"], args["times"])

    vis_obs = get_obs_vis_gains_ast(vis_ast, vis_rfi, gains, a1, a2)

    return vis_obs, (vis_rfi, vis_ast, gains)


def fixed_orbit_rfi_fft_standard(static_args, array_args, model, v_obs=None):
    rfi_shape = array_args["mu_rfi_r"].shape
    g_amp_shape = array_args["mu_G_amp"].shape
    g_phase_shape = array_args["mu_G_phase"].shape
    ast_k_shape = array_args["mu_ast_k_r"].shape

    rfi_r_base = numpyro.sample(
        "rfi_r_induce_base", dist.Normal(jnp.zeros(rfi_shape), jnp.ones(rfi_shape))
    )
    rfi_i_base = numpyro.sample(
        "rfi_i_induce_base", dist.Normal(jnp.zeros(rfi_shape), jnp.ones(rfi_shape))
    )

    g_amp_base = numpyro.sample(
        "g_amp_induce_base", dist.Normal(jnp.zeros(g_amp_shape), jnp.ones(g_amp_shape))
    )
    g_phase_base = numpyro.sample(
        "g_phase_induce_base",
        dist.Normal(jnp.zeros(g_phase_shape), jnp.ones(g_phase_shape)),
    )

    ast_k_r_base = numpyro.sample(
        "ast_k_r_base", dist.Normal(jnp.zeros(ast_k_shape), jnp.ones(ast_k_shape))
    )
    ast_k_i_base = numpyro.sample(
        "ast_k_i_base", dist.Normal(jnp.zeros(ast_k_shape), jnp.ones(ast_k_shape))
    )

    params = {
        "rfi_r_induce_base": rfi_r_base,
        "rfi_i_induce_base": rfi_i_base,
        "g_amp_induce_base": g_amp_base,
        "g_phase_induce_base": g_phase_base,
        "ast_k_r_base": ast_k_r_base,
        "ast_k_i_base": ast_k_i_base,
    }

    vis_obs, (vis_rfi, vis_ast, gains, rfi_A) = model(params, static_args, array_args)

    # numpyro.deterministic("times_mjd_fine", args["array_args"]["times_mjd_fine"])
    # numpyro.deterministic("rfi_phase", args["array_args"]["rfi_phase"])
    rfi_A = numpyro.deterministic("rfi_A", rfi_A)
    rfi_vis = numpyro.deterministic("rfi_vis", vis_rfi)
    ast_vis = numpyro.deterministic("ast_vis", vis_ast)
    gains = numpyro.deterministic("gains", gains)
    vis_obs = numpyro.deterministic("vis_obs", vis_obs)

    numpyro.deterministic(
        "rmse_ast",
        rmse(ast_vis, array_args["vis_ast_true"]) / jnp.sqrt(2),
    )
    numpyro.deterministic(
        "rmse_rfi",
        rmse(rfi_vis, array_args["vis_rfi_true"]) / jnp.sqrt(2),
    )
    numpyro.deterministic(
        "rmse_gains",
        rmse(gains, array_args["gains_true"]) / jnp.sqrt(2),
    )

    if v_obs is not None:
        return numpyro.sample(
            "obs",
            dist.Normal(
                jnp.concatenate([vis_obs.real, vis_obs.imag], axis=1),
                array_args["noise"],
            ),
            obs=v_obs,
        )


#########################################################################################################
# Orbit Parameters included
#########################################################################################################


@partial(jit, static_argnums=(1,))
def kepler_orbit_fft_padded_model(params, args, array_args):

    a1 = array_args["a1"]
    a2 = array_args["a2"]

    rfi_orbit = vmap(affine_transform_full, (0, 0, 0), 0)(
        params["rfi_orbit_base"], array_args["L_rfi_orbit"], array_args["mu_rfi_orbit"]
    )
    rfi_r = vmap(vmap(affine_transform_full, (0, None, 0), 0), (1, None, 1), 1)(
        params["rfi_r_induce_base"], array_args["L_RFI"], array_args["mu_rfi_r"]
    )
    rfi_i = vmap(vmap(affine_transform_full, (0, None, 0), 0), (1, None, 1), 1)(
        params["rfi_i_induce_base"], array_args["L_RFI"], array_args["mu_rfi_i"]
    )
    g_amp = vmap(affine_transform_full, in_axes=(0, None, 0))(
        params["g_amp_induce_base"], array_args["L_G_amp"], array_args["mu_G_amp"]
    )
    g_phase = vmap(affine_transform_full, in_axes=(0, None, 0))(
        params["g_phase_induce_base"], array_args["L_G_phase"], array_args["mu_G_phase"]
    )
    ast_k_r = vmap(affine_transform_diag, in_axes=(0, 0, 0))(
        params["ast_k_r_base"], array_args["sigma_ast_k"], array_args["mu_ast_k_r"]
    )
    ast_k_i = vmap(affine_transform_diag, in_axes=(0, 0, 0))(
        params["ast_k_i_base"], array_args["sigma_ast_k"], array_args["mu_ast_k_i"]
    )

    rfi_phase = get_rfi_phase_from_orbit(rfi_orbit, array_args)
    array_args["rfi_phase"] = rfi_phase

    rfi_A = rfi_r + 1.0 * rfi_i
    vis_rfi = get_rfi_vis_full(
        rfi_A,
        args,
        array_args,
    )
    vis_ast = get_ast_vis_fft_padded(ast_k_r, ast_k_i, args["ast_pad"])
    gains = get_gains_straight(
        g_amp, g_phase, array_args["g_times"], array_args["times"]
    )

    vis_obs = get_obs_vis_gains_all(vis_ast, vis_rfi, gains, a1, a2)
    # vis_obs = get_obs_vis_gains_ast(vis_ast, vis_rfi, gains, a1, a2)

    return vis_obs, (vis_rfi, vis_ast, gains, rfi_A, rfi_phase)


def kepler_orbit_fft(static_args, array_args, model, v_obs=None):

    # (n_rfi, n_ant, n_rfi_time)

    rfi_shape = array_args["mu_rfi_r"].shape
    # (n_rfi, 6)
    orbit_shape = (rfi_shape[0], 6)
    # (n_ant, n_g_time)
    g_amp_shape = array_args["mu_G_amp"].shape
    # (n_ant-1, n_g_time)
    g_phase_shape = array_args["mu_G_phase"].shape
    # (n_bl, n_time)
    ast_k_shape = array_args["mu_ast_k_r"].shape

    rfi_orbit_base = numpyro.sample(
        "rfi_orbit_base", dist.Normal(jnp.zeros(orbit_shape), jnp.ones(orbit_shape))
    )

    rfi_r_base = numpyro.sample(
        "rfi_r_induce_base", dist.Normal(jnp.zeros(rfi_shape), jnp.ones(rfi_shape))
    )
    rfi_i_base = numpyro.sample(
        "rfi_i_induce_base", dist.Normal(jnp.zeros(rfi_shape), jnp.ones(rfi_shape))
    )

    g_amp_base = numpyro.sample(
        "g_amp_induce_base", dist.Normal(jnp.zeros(g_amp_shape), jnp.ones(g_amp_shape))
    )
    g_phase_base = numpyro.sample(
        "g_phase_induce_base",
        dist.Normal(jnp.zeros(g_phase_shape), jnp.ones(g_phase_shape)),
    )

    ast_k_r_base = numpyro.sample(
        "ast_k_r_base", dist.Normal(jnp.zeros(ast_k_shape), jnp.ones(ast_k_shape))
    )
    ast_k_i_base = numpyro.sample(
        "ast_k_i_base", dist.Normal(jnp.zeros(ast_k_shape), jnp.ones(ast_k_shape))
    )

    params = {
        "rfi_orbit_base": rfi_orbit_base,
        "rfi_r_induce_base": rfi_r_base,
        "rfi_i_induce_base": rfi_i_base,
        "g_amp_induce_base": g_amp_base,
        "g_phase_induce_base": g_phase_base,
        "ast_k_r_base": ast_k_r_base,
        "ast_k_i_base": ast_k_i_base,
    }

    vis_obs, (vis_rfi, vis_ast, gains, rfi_A, rfi_phase) = model(
        params, static_args, array_args
    )

    # numpyro.deterministic("times_mjd_fine", args["array_args"]["times_mjd_fine"])
    numpyro.deterministic("rfi_phase", rfi_phase)
    rfi_A = numpyro.deterministic("rfi_A", rfi_A)
    rfi_vis = numpyro.deterministic("rfi_vis", vis_rfi)
    ast_vis = numpyro.deterministic("ast_vis", vis_ast)
    gains = numpyro.deterministic("gains", gains)
    vis_obs = numpyro.deterministic("vis_obs", vis_obs)

    numpyro.deterministic(
        "rmse_ast",
        rmse(ast_vis, array_args["vis_ast_true"]) / jnp.sqrt(2),
    )
    numpyro.deterministic(
        "rmse_rfi",
        rmse(rfi_vis, array_args["vis_rfi_true"]) / jnp.sqrt(2),
    )
    numpyro.deterministic(
        "rmse_gains",
        rmse(gains, array_args["gains_true"]) / jnp.sqrt(2),
    )

    if v_obs is not None:
        return numpyro.sample(
            "obs",
            dist.Normal(
                jnp.concatenate([vis_obs.real, vis_obs.imag], axis=1),
                array_args["noise"],
            ),
            obs=v_obs,
        )
