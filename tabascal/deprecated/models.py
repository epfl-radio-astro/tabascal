import jax.numpy as jnp
from jax import jit, vmap
from jax.flatten_util import ravel_pytree as flatten
from jax.tree_util import tree_map
import numpyro
import numpyro.distributions as dist
from tabascal.dist import MVN, Normal
from tabascal.deprecated.vis import (
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
)
from tabascal.transform import affine_transform_full, affine_transform_diag

from functools import partial
from frozendict import frozendict

import numpy as np


def fixed_orbit_real_rfi(args, v_obs=None):
    n_time = args["N_time"]
    n_ant = args["N_ants"]
    n_bl = args["N_bl"]
    a1 = args["a1"]
    a2 = args["a2"]

    # with numpyro.plate("N_ants", n_ant):
    #     # rfi_amp = MVN("rfi_r_induce", args["mu_rfi_r"], args["L_RFI_amp"])
    #     rfi_amp = numpyro.sample(
    #         "rfi_r_induce",
    #         dist.MultivariateNormal(
    #             loc=args["mu_rfi_r"],
    #             scale_tril=args["L_RFI_amp"] / 1e1,
    #         ),
    #     )
    #     G_amp = numpyro.sample(
    #         "g_amp_induce",
    #         dist.MultivariateNormal(
    #             loc=args["mu_G_amp"],
    #             scale_tril=args["L_G_amp"],
    #         ),
    #     )

    rfi_amp = numpyro.sample(
        "rfi_r_induce",
        dist.MultivariateNormal(
            loc=args["mu_rfi_r"],
            scale_tril=args["L_RFI_amp"] / 1e1,
        ),
    )
    G_amp = numpyro.sample(
        "g_amp_induce",
        dist.MultivariateNormal(
            loc=args["mu_G_amp"],
            scale_tril=args["L_G_amp"],
        ),
    )

    # with numpyro.plate("N_ants-1", n_ant - 1):
    #     G_phase = numpyro.sample(
    #         "g_phase_induce",
    #         dist.MultivariateNormal(
    #             loc=args["mu_G_phase"], scale_tril=args["L_G_phase"]
    #         ),
    #     )

    G_phase = numpyro.sample(
        "g_phase_induce",
        dist.MultivariateNormal(loc=args["mu_G_phase"], scale_tril=args["L_G_phase"]),
    )

    gains = numpyro.deterministic(
        "gains",
        get_gains(G_amp, G_phase, args["resample_g_amp"], args["resample_g_phase"]),
    )

    rfi_vis = numpyro.deterministic(
        "rfi_vis",
        get_rfi_vis2(
            rfi_amp,
            args["resample_rfi"],
            args["rfi_phase"],
            a1,
            a2,
            args["bl"],
            args["n_int"],
        ),
    )

    # vis_r = [
    #     MVN(f"vis_r_{i}", args["mu_vis"][i], args["L_vis"][i]) for i in range(n_bl)
    # ]
    # vis_i = [
    #     MVN(f"vis_i_{i}", args["mu_vis"][i], args["L_vis"][i]) for i in range(n_bl)
    # ]
    # ast_vis = numpyro.deterministic(
    #     "ast_vis", get_ast_vis1(vis_r, vis_i, args["resample_vis"])
    # )

    # with numpyro.plate("N_bl", n_bl):
    #     vis_r = numpyro.sample(
    #         "vis_r_induce",
    #         dist.MultivariateNormal(
    #             loc=args["mu_vis_r"], scale_tril=args["L_vis"] / 10
    #         ),
    #     )
    #     vis_i = numpyro.sample(
    #         "vis_i_induce",
    #         dist.MultivariateNormal(
    #             loc=args["mu_vis_i"], scale_tril=args["L_vis"] / 10
    #         ),
    #     )
    # vis_r = MVN("vis_r_induce", args["mu_vis_r"], args["L_vis"])
    # vis_i = MVN("vis_i_induce", args["mu_vis_i"], args["L_vis"])

    vis_r = numpyro.sample(
        "vis_r_induce",
        dist.MultivariateNormal(loc=args["mu_vis_r"], scale_tril=args["L_vis"] / 10),
    )
    vis_i = numpyro.sample(
        "vis_i_induce",
        dist.MultivariateNormal(loc=args["mu_vis_i"], scale_tril=args["L_vis"] / 10),
    )

    ast_vis = numpyro.deterministic(
        "ast_vis", get_ast_vis2(vis_r, vis_i, args["resample_vis"])
    )

    vis_obs = numpyro.deterministic(
        "vis_obs", get_obs_vis_gains_all(ast_vis, rfi_vis, gains, a1, a2)
    )

    numpyro.deterministic(
        "rmse_ast",
        rmse(ast_vis, args["vis_ast_true"]) / jnp.sqrt(2),
    )
    numpyro.deterministic(
        "rmse_rfi",
        rmse(rfi_vis, args["vis_rfi_true"]) / jnp.sqrt(2),
    )
    numpyro.deterministic(
        "rmse_gains",
        rmse(gains, args["gains_true"]) / jnp.sqrt(2),
    )

    if v_obs is not None:
        return numpyro.sample(
            "obs",
            dist.Normal(
                jnp.concatenate([vis_obs.real, vis_obs.imag], axis=1), args["noise"]
            ),
            obs=v_obs,
        )


###############################################################################


# @jit
def fixed_orbit_real_rfi_compressed(args, v_obs=None):
    n_time = args["N_time"]
    n_ant = args["N_ants"]
    n_bl = args["N_bl"]
    a1 = args["a1"]
    a2 = args["a2"]

    # with numpyro.plate("N_ants", n_ant):
    #     rfi_amp = MVN("rfi_r_induce", args["mu_rfi"], args["L_RFI_amp"])
    #     # rfi_amp = Normal("rfi_r_induce", args["mu_rfi_r"], 1e3)
    #     # G_amp = MVN("g_amp_induce", args["mu_G_amp"], args["L_G_amp"][None,:,:]*jnp.ones((n_ant,1,1)))
    #     G_amp = numpyro.sample(
    #         "g_amp_induce",
    #         dist.MultivariateNormal(
    #             loc=args["mu_G_amp"],
    #             scale_tril=args["L_G_amp"],
    #         ),
    #     )

    rfi_amp = numpyro.sample(
        "rfi_r_induce",
        dist.MultivariateNormal(
            loc=args["mu_rfi_r"],
            scale_tril=args["L_RFI_amp"],
        ),
    )
    G_amp = numpyro.sample(
        "g_amp_induce",
        dist.MultivariateNormal(
            loc=args["mu_G_amp"],
            scale_tril=args["L_G_amp"],
        ),
    )

    # with numpyro.plate("N_ants-1", n_ant - 1):
    #     G_phase = numpyro.sample(
    #         "g_phase_induce",
    #         dist.MultivariateNormal(
    #             loc=args["mu_G_phase"],
    #             scale_tril=args["L_G_phase"],
    #         ),
    #     )

    G_phase = numpyro.sample(
        "g_phase_induce",
        dist.MultivariateNormal(
            loc=args["mu_G_phase"],
            scale_tril=args["L_G_phase"],
        ),
    )

    gains = numpyro.deterministic(
        "gains",
        get_gains(G_amp, G_phase, args["resample_g_amp"], args["resample_g_phase"]),
    )

    rfi_vis = numpyro.deterministic(
        "rfi_vis", get_rfi_vis1(rfi_amp, args["rfi_kernel"], a1, a2)
    )

    # vis_r = [
    #     MVN(f"vis_r_{i}", args["mu_vis"][i], args["L_vis"][i]) for i in range(n_bl)
    # ]
    # vis_i = [
    #     MVN(f"vis_i_{i}", args["mu_vis"][i], args["L_vis"][i]) for i in range(n_bl)
    # ]
    # ast_vis = numpyro.deterministic(
    #     "ast_vis", get_ast_vis(vis_r, vis_i, args["resample_vis"])
    # )

    # with numpyro.plate("N_bl", n_bl):
    #     vis_r = MVN("vis_r_induce", args["mu_vis_r"], args["L_vis"])
    #     vis_i = MVN("vis_i_induce", args["mu_vis_i"], args["L_vis"])

    vis_r = numpyro.sample(
        "vis_r_induce",
        dist.MultivariateNormal(loc=args["mu_vis_r"], scale_tril=args["L_vis"]),
    )
    vis_i = numpyro.sample(
        "vis_i_induce",
        dist.MultivariateNormal(loc=args["mu_vis_i"], scale_tril=args["L_vis"]),
    )

    ast_vis = numpyro.deterministic(
        "ast_vis", get_ast_vis2(vis_r, vis_i, args["resample_vis"])
    )

    vis_obs = numpyro.deterministic(
        "vis_obs", get_obs_vis_gains_all(ast_vis, rfi_vis, gains, a1, a2)
    )

    numpyro.deterministic(
        "rmse_ast",
        rmse(ast_vis, args["vis_ast_true"]) / jnp.sqrt(2),
    )
    numpyro.deterministic(
        "rmse_rfi",
        rmse(rfi_vis, args["vis_rfi_true"]) / jnp.sqrt(2),
    )
    numpyro.deterministic(
        "rmse_gains",
        rmse(gains, args["gains_true"]) / jnp.sqrt(2),
    )

    if v_obs is not None:
        return numpyro.sample(
            "obs",
            dist.Normal(
                jnp.concatenate([vis_obs.real, vis_obs.imag], axis=1), args["noise"]
            ),
            obs=v_obs,
        )


###############################################################################


# @jit
def fixed_orbit_real_rfi_compressed2(args, v_obs=None):
    n_time = args["N_time"]
    n_ant = args["N_ants"]
    n_bl = args["N_bl"]
    a1 = args["a1"]
    a2 = args["a2"]

    # with numpyro.plate("N_ants", n_ant):
    #     rfi_amp = MVN("rfi_r_induce", args["mu_rfi"], args["L_RFI_amp"])
    #     # rfi_amp = Normal("rfi_r_induce", args["mu_rfi_r"], 1e3)
    #     # G_amp = MVN("g_amp_induce", args["mu_G_amp"], args["L_G_amp"][None,:,:]*jnp.ones((n_ant,1,1)))
    #     G_amp = numpyro.sample(
    #         "g_amp_induce",
    #         dist.MultivariateNormal(
    #             loc=args["mu_G_amp"],
    #             scale_tril=args["L_G_amp"],
    #         ),
    #     )

    # with numpyro.plate("N_ants-1", n_ant - 1):
    #     G_phase = numpyro.sample(
    #         "g_phase_induce",
    #         dist.MultivariateNormal(
    #             loc=args["mu_G_phase"],
    #             scale_tril=args["L_G_phase"],
    #         ),
    #     )

    # rfi_amp = numpyro.sample(
    #     "rfi_r_induce",
    #     dist.MultivariateNormal(
    #         loc=args["mu_rfi_r"],
    #         scale_tril=args["L_RFI_amp"],
    #     ),
    # )
    # G_amp = numpyro.sample(
    #     "g_amp_induce",
    #     dist.MultivariateNormal(
    #         loc=args["mu_G_amp"],
    #         scale_tril=args["L_G_amp"],
    #     ),
    # )

    # G_phase = numpyro.sample(
    #     "g_phase_induce",
    #     dist.MultivariateNormal(
    #         loc=args["mu_G_phase"],
    #         scale_tril=args["L_G_phase"],
    #     ),
    # )

    rfi_amp = jnp.stack(
        [
            MVN(f"rfi_r_induce_{i}", args["mu_rfi_r"][i], args["L_RFI_amp"])
            for i in range(len(args["mu_rfi_r"]))
        ],
        axis=0,
    )
    G_amp = jnp.stack(
        [
            MVN(f"g_amp_induce_{i}", args["mu_G_amp"][i], args["L_G_amp"])
            for i in range(len(args["mu_G_amp"]))
        ],
        axis=0,
    )
    G_phase = jnp.stack(
        [
            MVN(f"g_phase_induce_{i}", args["mu_G_phase"][i], args["L_G_phase"])
            for i in range(len(args["mu_G_phase"]))
        ],
        axis=0,
    )

    gains = numpyro.deterministic(
        "gains",
        get_gains(G_amp, G_phase, args["resample_g_amp"], args["resample_g_phase"]),
    )

    rfi_vis = numpyro.deterministic(
        "rfi_vis", get_rfi_vis1(rfi_amp, args["rfi_kernel"], a1, a2)
    )

    vis_r = jnp.stack(
        [
            MVN(f"vis_r_induce_{i}", args["mu_vis_r"][i], args["L_vis"])
            for i in range(n_bl)
        ],
        axis=0,
    )
    vis_i = jnp.stack(
        [
            MVN(f"vis_i_induce_{i}", args["mu_vis_i"][i], args["L_vis"])
            for i in range(n_bl)
        ],
        axis=0,
    )

    ast_vis = numpyro.deterministic(
        "ast_vis", get_ast_vis2(vis_r, vis_i, args["resample_vis"])
    )

    # with numpyro.plate("N_bl", n_bl):
    #     vis_r = MVN("vis_r_induce", args["mu_vis_r"], args["L_vis"])
    #     vis_i = MVN("vis_i_induce", args["mu_vis_i"], args["L_vis"])

    # vis_r = numpyro.sample(
    #     "vis_r_induce",
    #     dist.MultivariateNormal(loc=args["mu_vis_r"], scale_tril=args["L_vis"]),
    # )
    # vis_i = numpyro.sample(
    #     "vis_i_induce",
    #     dist.MultivariateNormal(loc=args["mu_vis_i"], scale_tril=args["L_vis"]),
    # )

    # ast_vis = numpyro.deterministic(
    #     "ast_vis", get_ast_vis2(vis_r, vis_i, args["resample_vis"])
    # )

    vis_obs = numpyro.deterministic(
        "vis_obs", get_obs_vis_gains_all(ast_vis, rfi_vis, gains, a1, a2)
    )

    numpyro.deterministic(
        "rmse_ast",
        rmse(ast_vis, args["vis_ast_true"]) / jnp.sqrt(2),
    )
    numpyro.deterministic(
        "rmse_rfi",
        rmse(rfi_vis, args["vis_rfi_true"]) / jnp.sqrt(2),
    )
    numpyro.deterministic(
        "rmse_gains",
        rmse(gains, args["gains_true"]) / jnp.sqrt(2),
    )

    if v_obs is not None:
        return numpyro.sample(
            "obs",
            dist.Normal(
                jnp.concatenate([vis_obs.real, vis_obs.imag], axis=1), args["noise"]
            ),
            obs=v_obs,
        )


def fixed_orbit_real_rfi_compressed3(args, v_obs=None):
    a1 = args["a1"]
    a2 = args["a2"]
    # n_time = args["N_time"]
    n_ant = args["N_ants"]
    n_bl = args["N_bl"]

    rfi_amp = jnp.array(
        tree_map(
            lambda i, mu, L: MVN(f"rfi_r_induce_{i}", mu, L),
            list(range(n_ant)),
            args["mu_rfi_r"],
            args["L_RFI_amp"],
        )
    )

    G_amp = jnp.array(
        tree_map(
            lambda i, mu, L: MVN(f"g_amp_induce_{i}", mu, L),
            list(range(n_ant)),
            args["mu_G_amp"],
            args["L_G_amp"],
        )
    )

    G_phase = jnp.array(
        tree_map(
            lambda i, mu, L: MVN(f"g_phase_induce_{i}", mu, L),
            list(range(n_ant - 1)),
            args["mu_G_phase"],
            args["L_G_phase"],
        )
    )

    vis_r = tree_map(
        lambda i, mu, L: MVN(f"vis_r_induce_{i}", mu, L),
        list(range(n_bl)),
        args["mu_vis_r"],
        args["L_vis"],
    )

    vis_i = tree_map(
        lambda i, mu, L: MVN(f"vis_i_induce_{i}", mu, L),
        list(range(n_bl)),
        args["mu_vis_i"],
        args["L_vis"],
    )

    gains = numpyro.deterministic(
        "gains",
        get_gains(G_amp, G_phase, args["resample_g_amp"], args["resample_g_phase"]),
    )

    rfi_vis = numpyro.deterministic(
        "rfi_vis", get_rfi_vis1(rfi_amp, args["rfi_kernel"], a1, a2)
    )

    ast_vis = numpyro.deterministic(
        "ast_vis", get_ast_vis1(vis_r, vis_i, args["resample_vis"])
    )

    vis_obs = numpyro.deterministic(
        "vis_obs", get_obs_vis_gains_all(ast_vis, rfi_vis, gains, a1, a2)
    )

    numpyro.deterministic(
        "rmse_ast",
        rmse(ast_vis, args["vis_ast_true"]) / jnp.sqrt(2),
    )
    numpyro.deterministic(
        "rmse_rfi",
        rmse(rfi_vis, args["vis_rfi_true"]) / jnp.sqrt(2),
    )
    numpyro.deterministic(
        "rmse_gains",
        rmse(gains, args["gains_true"]) / jnp.sqrt(2),
    )

    if v_obs is not None:
        return numpyro.sample(
            "obs",
            dist.Normal(
                jnp.concatenate([vis_obs.real, vis_obs.imag], axis=1), args["noise"]
            ),
            obs=v_obs,
        )


def fixed_orbit_real_rfi_compressed4(args, v_obs=None):
    a1 = args["a1"]
    a2 = args["a2"]
    # n_time = args["N_time"]
    n_ant = args["N_ants"]
    n_rfi_t = args["L_RFI_amp"].shape[0]
    n_g_t = args["L_G_amp"].shape[0]
    n_bl, n_vis_t = args["L_vis"].shape[:2]

    affine = lambda x, loc, scale_tril: scale_tril @ x + loc

    # rfi_r_base = numpyro.sample(
    #     "rfi_r_induce_base",
    #     dist.Normal(jnp.zeros((n_ant, n_rfi_t)), jnp.ones((n_ant, n_rfi_t))),
    # )

    rfi_r_base = MVN("rfi_r_induce_base", jnp.zeros((n_ant, n_rfi_t)), jnp.eye(n_rfi_t))

    rfi_r = numpyro.deterministic(
        "rfi_r_induce",
        vmap(affine, in_axes=(0, 0, None))(
            rfi_r_base, args["mu_rfi_r"], args["L_RFI_amp"]
        ),
    )

    g_amp_base = numpyro.sample(
        "g_amp_induce_base",
        dist.Normal(jnp.zeros((n_ant, n_g_t)), jnp.ones((n_ant, n_g_t))),
    )
    g_amp = numpyro.deterministic(
        "g_amp_induce",
        vmap(affine, in_axes=(0, 0, None))(
            g_amp_base, args["mu_G_amp"], args["L_G_amp"]
        ),
    )

    g_phase_base = numpyro.sample(
        "g_phase_induce_base",
        dist.Normal(jnp.zeros((n_ant - 1, n_g_t)), jnp.ones((n_ant - 1, n_g_t))),
    )
    g_phase = numpyro.deterministic(
        "g_phase_induce",
        vmap(affine, in_axes=(0, 0, None))(
            g_phase_base, args["mu_G_phase"], args["L_G_phase"]
        ),
    )

    vis_r_base = numpyro.sample(
        "vis_r_induce_base",
        dist.Normal(jnp.zeros((n_bl, n_vis_t)), jnp.ones((n_bl, n_vis_t))),
    )
    vis_r = numpyro.deterministic(
        "vis_r_induce",
        vmap(affine, in_axes=(0, 0, 0))(vis_r_base, args["mu_vis_r"], args["L_vis"]),
    )

    vis_i_base = numpyro.sample(
        "vis_i_induce_base",
        dist.Normal(jnp.zeros((n_bl, n_vis_t)), jnp.ones((n_bl, n_vis_t))),
    )
    vis_i = numpyro.deterministic(
        "vis_i_induce",
        vmap(affine, in_axes=(0, 0, 0))(vis_i_base, args["mu_vis_i"], args["L_vis"]),
    )

    gains = numpyro.deterministic(
        "gains",
        get_gains(g_amp, g_phase, args["resample_g_amp"], args["resample_g_phase"]),
    )

    rfi_vis = numpyro.deterministic(
        "rfi_vis", get_rfi_vis1(rfi_r, args["rfi_kernel"], a1, a2)
    )

    ast_vis = numpyro.deterministic(
        "ast_vis", get_ast_vis3(vis_r, vis_i, args["resample_vis"])
    )

    vis_obs = numpyro.deterministic(
        "vis_obs", get_obs_vis_gains_all(ast_vis, rfi_vis, gains, a1, a2)
    )

    numpyro.deterministic(
        "rmse_ast",
        rmse(ast_vis, args["vis_ast_true"]) / jnp.sqrt(2),
    )
    numpyro.deterministic(
        "rmse_rfi",
        rmse(rfi_vis, args["vis_rfi_true"]) / jnp.sqrt(2),
    )
    numpyro.deterministic(
        "rmse_gains",
        rmse(gains, args["gains_true"]) / jnp.sqrt(2),
    )

    if v_obs is not None:
        return numpyro.sample(
            "obs",
            dist.Normal(
                jnp.concatenate([vis_obs.real, vis_obs.imag], axis=1), args["noise"]
            ),
            obs=v_obs,
        )


def fixed_orbit_real_rfi_compressed5(args, v_obs=None):
    a1 = args["a1"]
    a2 = args["a2"]
    n_bl = args["N_bl"]
    bl = args["bl"]

    rfi_r = MVN("rfi_r_induce", args["mu_rfi_r"], args["L_RFI_amp"])

    g_amp = MVN("g_amp_induce", args["mu_G_amp"], args["L_G_amp"])

    g_phase = MVN("g_phase_induce", args["mu_G_phase"], args["L_G_phase"])

    vis_r = tree_map(
        lambda i, mu, L: MVN(f"vis_r_induce_{i}", mu, L),
        bl,
        args["mu_vis_r"],
        args["L_vis"],
    )

    vis_i = tree_map(
        lambda i, mu, L: MVN(f"vis_i_induce_{i}", mu, L),
        bl,
        args["mu_vis_i"],
        args["L_vis"],
    )

    gains = numpyro.deterministic(
        "gains",
        get_gains(g_amp, g_phase, args["resample_g_amp"], args["resample_g_phase"]),
    )

    rfi_vis = numpyro.deterministic(
        "rfi_vis", get_rfi_vis1(rfi_r, args["rfi_kernel"], a1, a2)
    )

    ast_vis = numpyro.deterministic(
        "ast_vis", get_ast_vis11(vis_r, vis_i, args["resample_vis"])
    )

    vis_obs = numpyro.deterministic(
        "vis_obs", get_obs_vis_gains_all(ast_vis, rfi_vis, gains, a1, a2)
    )

    numpyro.deterministic(
        "rmse_ast",
        rmse(ast_vis, args["vis_ast_true"]) / jnp.sqrt(2),
    )
    numpyro.deterministic(
        "rmse_rfi",
        rmse(rfi_vis, args["vis_rfi_true"]) / jnp.sqrt(2),
    )
    numpyro.deterministic(
        "rmse_gains",
        rmse(gains, args["gains_true"]) / jnp.sqrt(2),
    )

    if v_obs is not None:
        return numpyro.sample(
            "obs",
            dist.Normal(
                jnp.concatenate([vis_obs.real, vis_obs.imag], axis=1), args["noise"]
            ),
            obs=v_obs,
        )


def fixed_orbit_complex_rfi_compressed_fft(args, v_obs=None):
    a1 = args["a1"]
    a2 = args["a2"]

    rfi_r = MVN("rfi_r_induce", args["mu_rfi_r"], args["L_RFI"])
    rfi_i = MVN("rfi_i_induce", args["mu_rfi_i"], args["L_RFI"])

    g_amp = MVN("g_amp_induce", args["mu_G_amp"], args["L_G_amp"])

    g_phase = MVN("g_phase_induce", args["mu_G_phase"], args["L_G_phase"])

    # g_amp = args["mu_G_amp"]

    # g_phase = args["mu_G_phase"]

    ast_k_r = Normal("ast_k_r", args["mu_ast_k_r"], args["sigma_ast_k"])

    ast_k_i = Normal("ast_k_i", args["mu_ast_k_i"], args["sigma_ast_k"])

    # gains = numpyro.deterministic(
    #     "gains",
    #     get_gains(g_amp, g_phase, args["resample_g_amp"], args["resample_g_phase"]),
    # )

    gains = numpyro.deterministic(
        "gains",
        get_gains_straight(g_amp, g_phase, args["g_times"], args["times"]),
    )

    rfi_vis = numpyro.deterministic(
        "rfi_vis", get_rfi_vis(rfi_r, rfi_i, args["rfi_kernel"], a1, a2)
    )

    ast_vis = numpyro.deterministic("ast_vis", get_ast_vis_fft(ast_k_r, ast_k_i))

    vis_obs = numpyro.deterministic(
        "vis_obs", get_obs_vis_gains_all(ast_vis, rfi_vis, gains, a1, a2)
    )

    numpyro.deterministic(
        "rmse_ast",
        rmse(ast_vis, args["vis_ast_true"]) / jnp.sqrt(2),
    )
    numpyro.deterministic(
        "rmse_rfi",
        rmse(rfi_vis, args["vis_rfi_true"]) / jnp.sqrt(2),
    )
    numpyro.deterministic(
        "rmse_gains",
        rmse(gains, args["gains_true"]) / jnp.sqrt(2),
    )

    if v_obs is not None:
        return numpyro.sample(
            "obs",
            dist.Normal(
                jnp.concatenate([vis_obs.real, vis_obs.imag], axis=1), args["noise"]
            ),
            obs=v_obs,
        )


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
    rfi_A = rfi_r + 1.0j * rfi_i
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
    rfi_A = rfi_r + 1.0j * rfi_i
    vis_rfi = get_rfi_vis_full(
        rfi_A,
        args,
        array_args,
    )
    vis_ast = get_ast_vis_fft_padded(ast_k_r, ast_k_i, args["ast_pad"])
    gains = get_gains_straight(
        g_amp, g_phase, array_args["g_times"], array_args["times"]
    )

    # vis_obs = get_obs_vis_gains_all(vis_ast, vis_rfi, gains, a1, a2)
    # vis_obs = get_obs_vis_gains_ast(vis_ast, vis_rfi, gains, a1, a2)

    vis_obs = vis_ast + vis_rfi

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
    rfi_A = rfi_r + 1.0j * rfi_i
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
    rfi_A = rfi_r + 1.0j * rfi_i
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

    # static_args = {}
    # array_args = {}
    # for key, value in args.items():
    #     if hasattr(value, "shape"):
    #         array_args.update({key: value})
    #     else:
    #         static_args.update({key: value})

    # static_args = frozendict(static_args)

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


###############################################################################


# def fixed_orbit_complex_rfi_compressed(args):
#     n_time = args["N_time"]
#     n_ant = args["N_ants"]
#     n_bl = args["N_bl"]
#     a1 = args["a1"]
#     a2 = args["a2"]

#     with numpyro.plate("N_ants", n_ant):
#         rfi_amp = MVN("rfi_amp_induce", args["mu_rfi_amp"], args["L_RFI_amp"])
#         G_amp = numpyro.sample(
#             "g_amp_induce",
#             dist.MultivariateNormal(
#                 args["mu_G_amp"],
#                 args["cov_G_amp"],
#             ),
#         )

#     with numpyro.plate("N_ants-1", n_ant - 1):
#         rfi_phase = MVN("rfi_phase", args["mu_rfi_phase"], args["L_RFI_phase"])
#         G_phase = numpyro.sample(
#             "g_phase_induce",
#             dist.MultivariateNormal(args["mu_G_phase"], args["cov_G_phase"]),
#         )

#     G_amp = numpyro.deterministic("g_amp", G_amp @ args["resample_g_amp"].T)
#     G_phase = numpyro.deterministic(
#         "g_phase",
#         jnp.concatenate(
#             [G_phase @ args["resample_g_phase"].T, jnp.zeros((1, n_time))], axis=0
#         ),
#     )

#     gains = numpyro.deterministic("gains", G_amp * jnp.exp(1.0j * G_phase))

#     rfi_vis = numpyro.deterministic(
#         "rfi_vis",
#         get_rfi_vis1(
#             rfi_amp * jnp.exp(1.0j * rfi_phase), args["rfi_kernel"], a1, a2
#         ).T,
#     )

#     vis_r = [
#         MVN(f"vis_r_{i}", args["mu_vis"][i], args["L_vis"][i]) for i in range(n_bl)
#     ]
#     vis_i = [
#         MVN(f"vis_i_{i}", args["mu_vis"][i], args["L_vis"][i]) for i in range(n_bl)
#     ]

#     ast_vis = numpyro.deterministic(
#         "ast_vis", get_ast_vis(vis_r, vis_i, args["resample_vis"])
#     )

#     vis_obs = numpyro.deterministic(
#         "vis_obs", get_obs_vis_gains_all(ast_vis, rfi_vis, gains, a1, a2)
#     )

#     numpyro.deterministic(
#         "rmse_ast",
#         rmse(ast_vis, args["vis_ast_true"]) / jnp.sqrt(2),
#     )
#     numpyro.deterministic(
#         "rmse_rfi",
#         rmse(rfi_vis, args["vis_rfi_true"]) / jnp.sqrt(2),
#     )
#     numpyro.deterministic(
#         "rmse_gains",
#         rmse(gains, args["gains_true"]) / jnp.sqrt(2),
#     )

#     return numpyro.sample(
#         "obs",
#         dist.Normal(
#             jnp.concatenate([vis_obs.real, vis_obs.imag], axis=1), args["noise"]
#         ),
#         obs=args["vis_obs"],
#     )
