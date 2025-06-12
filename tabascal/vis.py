from functools import partial

import jax.numpy as jnp
from jax import jit, vmap
from jax.flatten_util import ravel_pytree as flatten
from jax.tree_util import tree_map
from jax.lax import scan, dynamic_slice
from jax import checkpoint

from tabsim.jax.interferometry import rfi_vis
from tabsim.jax.coordinates import orbit

from tabascal.gp import gp_resample_otf, gp_resample_fft


@partial(jit, static_argnums=(1,))
def averaging(x, n_avg):
    n = x.shape[0] // n_avg

    x_avg = jnp.reshape(x[:-1], (n, n_avg))
    x_end = x[n_avg::n_avg][:, None]
    x_avg = jnp.trapezoid(jnp.concatenate([x_avg, x_end], axis=1), dx=1 / n_avg, axis=1)

    return x_avg


@partial(jit, static_argnums=(1,))
def averaging0(x, n_avg):
    n = x.shape[0] // n_avg
    x_avg = jnp.mean(jnp.reshape(x, (n, n_avg)), axis=1)
    return x_avg


@partial(jit, static_argnums=(1,))
def averaging1(x, n_avg):
    n = x.shape[1] // n_avg
    x_avg = jnp.mean(jnp.reshape(x, (-1, n, n_avg)), axis=2)
    return x_avg


@jit
def averaging2(x, times, times_fine):
    n = len(times)
    n_avg = int(len(times_fine) / n)
    x_avg = jnp.mean(jnp.reshape(x, (n, n_avg, -1)), axis=1)
    return x_avg


@jit
def averaging3(x, times):
    n = len(times)
    n_avg = int(x.shape[0] / n)
    x_avg = jnp.mean(jnp.reshape(x, (n, n_avg, -1)), axis=1)
    return x_avg


@partial(jit, static_argnums=(1,))
def pad_vis(vis, length):
    return jnp.pad(vis, (0, length - vis.shape[0]))


# def get_rfi_vis_compressed(rfi_r, rfi_i, rfi_kernel, a1, a2):
@jit
def get_rfi_vis_compressed_ri(rfi_r, rfi_i, rfi_kernel, a1, a2):
    rfi_I = (rfi_r + 1.0j * rfi_i)[a1] * (rfi_r - 1.0j * rfi_i)[a2]
    vis_rfi = vmap(jnp.dot)(rfi_kernel, rfi_I)
    return vis_rfi


# def get_rfi_vis_compressed1(rfi_amp, rfi_kernel, a1, a2):
@jit
def get_rfi_vis_compressed_comp(rfi_amp, rfi_kernel, a1, a2):
    rfi_I = rfi_amp[a1] * jnp.conjugate(rfi_amp[a2])
    vis_rfi = vmap(jnp.dot, in_axes=(2, 0))(rfi_kernel, rfi_I)
    return vis_rfi


# @jit
# def get_rfi_vis_full(rfi_amp, rfi_resample, rfi_phase, a1, a2, times, times_fine):
#     rfi_amp_fine = rfi_amp @ rfi_resample.T
#     rfi_vis = (
#         rfi_amp_fine[a1]
#         * jnp.conjugate(rfi_amp_fine[a2])
#         * jnp.exp(1.0j * (rfi_phase[a1] - rfi_phase[a2]))
#     )
#     rfi_vis = averaging2(rfi_vis.T, times, times_fine).T
#     return rfi_vis


# def get_rfi_vis_full(rfi_amp, rfi_resample, rfi_phase, a1, a2, times, times_fine):
@partial(jit, static_argnums=(1,))
def get_rfi_vis_full(rfi_amp, args, array_args):
    a1, a2 = array_args["a1"], array_args["a2"]
    rfi_phase = array_args["rfi_phase"]
    # rfi_amp has shape (n_rfi, n_ant, n_time)
    rfi_amp_fine = vmap(lambda x, y: x @ y.T, in_axes=(0, None))(
        rfi_amp, array_args["resample_rfi"]
    )
    # rfi_amp_fine has shape (n_rfi, n_ant, n_time_fine)
    rfi_vis = jnp.sum(
        rfi_amp_fine[:, a1]
        * jnp.conjugate(rfi_amp_fine[:, a2])
        * jnp.exp(1.0j * (rfi_phase[:, a1] - rfi_phase[:, a2])),
        axis=0,
    )
    # rfi_vis has shape (n_bl, n_time_fine)
    rfi_vis = averaging1(rfi_vis[:, :-1], args["n_int_samples"])
    # rfi_vis = vmap(averaging, in_axes=(0, None))(rfi_vis, args["n_int_samples"])
    # rfi_vis has shape (n_bl, n_time)
    return rfi_vis


@jit
def get_rfi_vis_full_otf(rfi_amp, args):
    # rfi_amp has shape (n_rfi, n_ant, n_time)
    rfi_amp_fine = vmap(
        vmap(
            lambda y: gp_resample_otf(
                y,
                args["rfi_times"],
                args["times_fine"],
                args["rfi_var"],
                args["rfi_l"],
            ),
            in_axes=(0),
        ),
        in_axes=(0),
    )(rfi_amp)
    # rfi_amp_fine has shape (n_rfi, n_ant, n_time_fine)
    rfi_vis = jnp.sum(
        rfi_amp_fine[:, args["a1"]]
        * jnp.conjugate(rfi_amp_fine[:, args["a2"]])
        * jnp.exp(
            1.0j * (args["rfi_phase"][:, args["a1"]] - args["rfi_phase"][:, args["a2"]])
        ),
        axis=0,
    )
    # rfi_vis has shape (n_bl, n_time_fine)
    rfi_vis = averaging2(rfi_vis.T, args["times"], args["times_fine"]).T
    # rfi_vis has shape (n_bl, n_time)
    return rfi_vis


@jit
def get_rfi_vis_fine(rfi_amp_fine, rfi_phase, a1, a2):

    rfi_vis = (
        rfi_amp_fine[a1]
        * jnp.conjugate(rfi_amp_fine[a2])
        * jnp.exp(1.0j * (rfi_phase[a1] - rfi_phase[a2]))
    )

    return rfi_vis


@partial(jit, static_argnums=(4,))
def get_rfi_vis_chunk(rfi_amp_fine, rfi_phase, a1, a2, args):

    rfi_vis_fine = (
        get_rfi_vis_fine if args["n_bl_chunk"] == 1 else checkpoint(get_rfi_vis_fine)
    )
    rfi_vis = rfi_vis_fine(rfi_amp_fine, rfi_phase, a1, a2)

    # rfi_vis has shape (n_bl_chunk, n_time_fine)
    # rfi_vis = averaging1(rfi_vis, args["n_int_samples"])
    rfi_vis = vmap(averaging, in_axes=(0, None))(rfi_vis, args["n_int_samples"])
    # rfi_vis has shape (n_bl_chunk, n_time)
    return rfi_vis


@partial(jit, static_argnums=(4,))
def get_rfi_vis_fft_src(rfi_amp, rfi_phase, a1, a2, args):
    # rfi_amp has shape (n_ant, n_time)
    rfi_amp_fine = vmap(gp_resample_fft, in_axes=(0, None))(
        rfi_amp, args["n_rfi_factor"]
    )
    # rfi_amp_fine has shape (n_ant, n_time_fine)

    n_chunks = args["n_bl_chunk"]
    chunksize = args["n_bl"] // n_chunks

    a1 = jnp.reshape(a1, (n_chunks, chunksize))
    a2 = jnp.reshape(a2, (n_chunks, chunksize))

    # Checkpoint rematerializes the intermediate results when they are smaller
    # Needed for larger than memory computation
    # Scan is an alternative to vmap here

    # rfi_vis_chunk = (
    #     get_rfi_vis_chunk
    #     if n_chunks == 1
    #     else checkpoint(get_rfi_vis_chunk, static_argnums=(4,))
    # )
    rfi_vis_chunk = get_rfi_vis_chunk

    vis_rfi = vmap(rfi_vis_chunk, in_axes=(None, None, 0, 0, None))(
        rfi_amp_fine,
        rfi_phase,
        a1,
        a2,
        args,
    )

    # @partial(checkpoint, prevent_cse=False)
    # def eval_loop(carry, x):
    #     i, a1_i, a2_i = x
    #     carry = carry.at[i].set(
    #         get_rfi_vis_chunk(
    #             rfi_amp_fine,
    #             rfi_phase,
    #             a1_i,
    #             a2_i,
    #             args,
    #         )
    #     )
    #     return carry, i

    # vis_rfi, _ = scan(
    #     eval_loop,
    #     jnp.zeros((n_chunks, chunksize, args["n_time"]), dtype=jnp.complex128),
    #     (jnp.arange(n_chunks), a1, a2),
    # )

    # @checkpoint
    # def eval_loop(carry, x):
    #     a1_i, a2_i = x
    #     rfi_vis_chunk = get_rfi_vis_chunk(
    #         rfi_amp_fine,
    #         rfi_phase,
    #         a1_i,
    #         a2_i,
    #         args,
    #     )
    #     return carry, rfi_vis_chunk

    # _, vis_rfi = scan(
    #     eval_loop,
    #     None,
    #     (a1, a2),
    # )

    return jnp.reshape(vis_rfi, (args["n_bl"], args["n_time"]))


@partial(jit, static_argnums=(1,))
def get_rfi_vis_full_otf_fft(rfi_amp, args, array_args):

    # rfi_amp has shape (n_rfi, n_ant, n_time)
    @checkpoint
    def sum_loop(carry, x):
        i, rfi_amp_i, rfi_phase_i = x
        carry = carry + get_rfi_vis_fft_src(
            rfi_amp_i,
            rfi_phase_i,
            array_args["a1"],
            array_args["a2"],
            args,
        )
        return carry, i

    vis_rfi, _ = scan(
        sum_loop,
        jnp.zeros((args["n_bl"], args["n_time"]), dtype=jnp.complex128),
        (jnp.arange(args["n_rfi"]), rfi_amp, array_args["rfi_phase"]),
    )

    return vis_rfi


# @partial(jit, static_argnums=(1,))
# def get_rfi_vis_full_otf_fft(rfi_amp, args, array_args):
#     # rfi_amp has shape (n_rfi, n_ant, n_time)
#     rfi_amp_fine = vmap(
#         vmap(
#             lambda y: gp_resample_fft(
#                 y,
#                 args["n_rfi_factor"],
#             ),
#             in_axes=(0),
#         ),
#         in_axes=(0),
#     )(rfi_amp)
#     # rfi_amp_fine has shape (n_rfi, n_ant, n_time_fine)
#     # print(rfi_amp_fine.shape)
#     rfi_vis = jnp.sum(
#         rfi_amp_fine[:, array_args["a1"]]
#         * jnp.conjugate(rfi_amp_fine[:, array_args["a2"]])
#         * jnp.exp(
#             1.0j
#             * (
#                 array_args["rfi_phase"][:, array_args["a1"]]
#                 - array_args["rfi_phase"][:, array_args["a2"]]
#             )
#         ),
#         axis=0,
#     )
#     # rfi_vis has shape (n_bl, n_time_fine)
#     rfi_vis = averaging1(rfi_vis, args["n_int_samples"])
#     # rfi_vis has shape (n_bl, n_time)
#     return rfi_vis


@partial(jit, static_argnums=(6,))
def get_rfi_vis3(rfi_amp, rfi_resample, rfi_phase, a1, a2, bl, n_int):
    def get_vis(rfi_amp_fine, rfi_phase, a1, a2, bl):
        vis = averaging(rfi_amp_fine[a1] * rfi_amp_fine[a2] * rfi_phase[bl], n_int)[
            :, 0
        ]
        return vis

    rfi_amp_fine = vmap(jnp.dot, in_axes=(None, 0))(rfi_resample, rfi_amp)
    rfi_vis = vmap(get_vis, in_axes=(None, None, 0, 0, 0))(
        rfi_amp_fine, rfi_phase, a1, a2, bl
    )

    return rfi_vis


# @jit
# def calc_rfi_vis_bl(rfi_amp_fine, rfi_phase, a1, a2, times, times_fine):
#     rfi_I = rfi_amp_fine[a1] * jnp.conjugate(rfi_amp_fine[a2])
#     rfi_phasor = jnp.exp(1.0j * (rfi_phase[a1] - rfi_phase[a2]))
#     vis = averaging(rfi_I * rfi_phasor, times, times_fine)[:, 0]
#     return vis


# @jit
# def get_rfi_vis3(rfi_amp, rfi_resample, rfi_phase, a1, a2, times, times_fine):
#     rfi_amp_fine = rfi_amp @ rfi_resample.T
#     rfi_vis = vmap(calc_rfi_vis_bl, in_axes=(None, None, 0, 0, None, None))(
#         rfi_amp_fine, rfi_phase, a1, a2, times, times_fine
#     )

#     return rfi_vis


@jit
def get_rfi_vis_fft1(rfi_k, a1, a2, rfi_k_kernel):
    rfi_amp = jnp.fft.ifft(rfi_k, axis=0)
    rfi_kI = jnp.fft.fft(rfi_amp[:, a1] * rfi_amp[:, a2], axis=0)

    rfi_vis = (rfi_k_kernel * rfi_kI).sum(axis=1).T

    return rfi_vis


# @partial(jit, static_argnums=(4, 5, 6))
# def get_rfi_vis_fft2(rfi_k, a1, a2, rfi_phase, N_pad, NN, N_int_samples):
#     rfi_amp = vmap(fft_inv_even, in_axes=(1, None, None))(rfi_k, int(N_pad), int(NN)).T
#     rfi_I = rfi_amp[a1] * jnp.conjugate(rfi_amp[a2])
#     rfi_phasor = jnp.exp(1.0j * (rfi_phase[a1] - rfi_phase[a2]))
#     rfi_vis = averaging(rfi_I * rfi_phasor, int(N_int_samples))

#     return rfi_vis


@jit
def get_rfi_vis_fft2(rfi_k, a1, a2, rfi_phase, times_fine, k_pad, times):
    rfi_amp = vmap(fft_inv_even, in_axes=(0, None, None))(rfi_k, times_fine, k_pad)
    rfi_I = rfi_amp[a1] * jnp.conjugate(rfi_amp[a2])
    rfi_phasor = jnp.exp(1.0j * (rfi_phase[a1] - rfi_phase[a2]))
    rfi_vis = averaging3((rfi_I * rfi_phasor).T, times).T

    return rfi_vis


@jit
def get_ast_vis_fft(ast_k_r, ast_k_i):
    return jnp.fft.ifft(ast_k_r + 1.0j * ast_k_i, axis=1)


@partial(jit, static_argnums=(2,))
def get_ast_vis_fft_padded(ast_k_r, ast_k_i, ast_pad):
    ast_k_padded = jnp.fft.ifft(ast_k_r + 1.0j * ast_k_i, axis=1)
    return ast_k_padded[:, ast_pad:-ast_pad]


@jit
def get_ast_vis(vis_r, vis_i, resample_vis):
    n_bl = len(vis_r)
    vis_ast = jnp.array(
        [resample_vis[i] @ (vis_r[i] + 1.0j * vis_i[i]) for i in range(n_bl)]
    )
    return vis_ast


@jit
def get_ast_vis1(vis_r, vis_i, resample_vis):
    vis_ast = tree_map(lambda r, i, A: A @ (r + 1.0j * i), vis_r, vis_i, resample_vis)
    return jnp.array(vis_ast)


@jit
def get_ast_vis11(vis_r, vis_i, resample_vis):
    n_bl, _, length = resample_vis.shape
    vis_pad = tree_map(lambda r, i: pad_vis(r + 1.0j * i, length), vis_r, vis_i)
    vis_pad = flatten(vis_pad)[0].reshape(n_bl, length)
    vis_ast = vmap(jnp.dot)(resample_vis, vis_pad)
    return vis_ast


@jit
def get_ast_vis2(vis_r, vis_i, resample_vis):
    vis_ast = (vis_r + 1.0j * vis_i) @ resample_vis.T
    return vis_ast


@jit
def get_ast_vis3(vis_r, vis_i, resample_vis):
    vis_ast = vmap(lambda r, i, A: A @ (r + 1.0j * i), in_axes=(0, 0, 0))(
        vis_r, vis_i, resample_vis
    )
    return vis_ast


@jit
def get_gains(g_amp, g_phase, resample_g_amp, resample_g_phase):
    n_time = resample_g_amp.shape[0]
    # n_batch = g_amp.shape[:1] if g_amp.ndim == 3 else ()
    g_amp = g_amp @ resample_g_amp.T
    g_phase = jnp.concatenate(
        [g_phase @ resample_g_phase.T, jnp.zeros((1, n_time))], axis=-2
    )
    gains = g_amp * jnp.exp(1.0j * g_phase)
    return gains


@jit
def get_gains_straight(g_amp, g_phase, g_times, times):
    n_time = len(times)
    g_amp = vmap(jnp.interp, in_axes=(None, None, 0))(times, g_times, g_amp)
    g_phase = vmap(jnp.interp, in_axes=(None, None, 0))(times, g_times, g_phase)
    g_phase = jnp.concatenate([g_phase, jnp.zeros((1, n_time))], axis=0)
    gains = g_amp * jnp.exp(1.0j * g_phase)
    return gains


@jit
def get_gains_mean(g_amp, g_phase, resample_g_amp, resample_g_phase):
    n_time = resample_g_amp.shape[0]
    g_amp = g_amp.mean(axis=1)[:, None] * jnp.ones((1, n_time))
    g_phase = jnp.concatenate(
        [g_phase.mean(axis=1)[:, None] * jnp.ones((1, n_time)), jnp.zeros((1, n_time))],
        axis=-2,
    )
    gains = g_amp * jnp.exp(1.0j * g_phase)
    return gains


# def get_obs_vis(ast_vis, rfi_vis, gains, a1, a2):
@jit
def get_obs_vis_gains_all(ast_vis, rfi_vis, gains, a1, a2):
    vis_obs = gains[a1] * jnp.conjugate(gains[a2]) * (ast_vis + rfi_vis)
    return vis_obs


# def get_obs_vis1(ast_vis, rfi_vis, gains, a1, a2):
@jit
def get_obs_vis_gains_ast(ast_vis, rfi_vis, gains, a1, a2):
    vis_obs = gains[a1] * jnp.conjugate(gains[a2]) * ast_vis + rfi_vis
    return vis_obs


@partial(jit, static_argnums=(2,))
def rmse(x, x_true, axis=1):
    return jnp.sqrt(jnp.mean(jnp.abs(x - x_true) ** 2, axis=axis))


# @partial(jit, static_argnums=(1, 2))
# def fft_inv_even(X, n_pad, n_out):
#     n_k = len(X)
#     n_k_missing = n_out - n_k
#     factor = n_out / n_k
#     # assert n_k%2 == 0
#     X_ = X * factor
#     X_ = jnp.concatenate([X_[: n_k // 2], jnp.zeros(n_k_missing), X_[n_k // 2 :]])
#     x = jnp.fft.ifft(X_)[n_pad:-n_pad]
#     return x


@jit
def fft_inv_even(X, k_original, k_pad):
    n_k_short = len(X)
    n_k_long = len(k_pad)
    n_k_original = len(k_original)
    n_k_missing = n_k_long - n_k_short
    n_pad = (n_k_long - n_k_original) // 2
    factor = n_k_long / n_k_short
    X_ = X * factor
    X_ = jnp.concatenate(
        [X_[: n_k_short // 2], jnp.zeros(n_k_missing), X_[n_k_short // 2 :]]
    )
    x = jnp.fft.ifft(X_)[n_pad:-n_pad]
    return x


# @jit
# def fft_inv_even(X, times, times_extended):
#     n_pad = len(times_extended)
#     n_k = len(X)
#     n_k_missing = n_out - n_k
#     factor = n_out / n_k
#     # assert n_k%2 == 0
#     X_ = X * factor
#     X_ = jnp.concatenate([X_[: n_k // 2], jnp.zeros(n_k_missing), X_[n_k // 2 :]])
#     x = jnp.fft.ifft(X_)[n_pad:-n_pad]
#     return x


@jit
def construct_real_fourier(x):
    N = len(x)
    x_r = x[: N // 2 + 1] + 1.0j * jnp.concatenate(
        [jnp.zeros(1), x[N // 2 + 1 :], jnp.zeros(1)]
    )
    x_recon = jnp.concatenate([x_r[: N // 2 + 1], jnp.conjugate(x_r[-2:0:-1])])
    return x_recon


def eye(N, idx):
    x = jnp.zeros(N)
    x = x.at[idx].set(1.0)
    return x


# @jit
# def get_rfi_phase(times, rfi_orbit, ants_uvw, ants_xyz, freqs, a1, a2):
#     n_time, n_ant = ants_uvw.shape[:2]
#     n_freq = len(freqs)
#     rfi_xyz = orbit(times, *rfi_orbit)
#     distances = jnp.linalg.norm(
#         ants_xyz[None, :, :, :] - rfi_xyz[None, :, None, :], axis=-1
#     )
#     c_distances = distances - ants_uvw[None, :, :, -1]
#     app_amplitude = jnp.ones((1, n_time, n_ant, n_freq))
#     vis_rfi = rfi_vis(app_amplitude, c_distances, freqs, a1, a2)
#     return vis_rfi


@jit
def get_rfi_phase(times, rfi_orbit, ants_uvw, ants_xyz, freqs):
    c = 2.99792458e8

    rfi_xyz = orbit(times, *rfi_orbit)
    distances = jnp.linalg.norm(ants_xyz[:, :, :] - rfi_xyz[:, None, :], axis=-1)
    c_distances = distances + ants_uvw[:, :, -1]

    phases = -2.0 * jnp.pi * c_distances * freqs / c

    return phases
