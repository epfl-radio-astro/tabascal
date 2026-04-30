import jax.numpy as jnp
from functools import reduce


def base_kernel(x_in, x_out, var, l):

    x_in = jnp.atleast_2d(x_in).T
    x_out = jnp.atleast_2d(x_out).T

    chi = jnp.linalg.norm(x_in[None, :, :] - x_out[:, None, :], axis=-1) / l
    K = jnp.abs(var) * jnp.exp(-0.5 * chi**2)

    return jnp.squeeze(K)


def cholesky(x_in, var, l, noise=1e-8):

    L = jnp.linalg.cholesky(
        base_kernel(x_in, x_in, var, l) + noise * jnp.eye(x_in.shape[0])
    )

    return L


def kernel(x, x_, var, l, noise=1e-3):

    x = x[:, None] if x.ndim == 1 else x
    x_ = x_[:, None] if x_.ndim == 1 else x_
    chi = jnp.linalg.norm(x[None, :, :] - x_[:, None, :], axis=-1) / l
    cov = jnp.abs(var) * jnp.exp(-0.5 * chi**2)
    if chi.shape[0] == chi.shape[1]:
        cov += noise * jnp.eye(x.shape[0])

    return cov


def resampling_kernel(x, x_, var, l, noise=1e-3):
    K_inv = jnp.linalg.inv(kernel(x, x, var, l, noise))
    K_s = kernel(x, x_, var, l)
    return K_s @ K_inv


def get_times(times, gp_l):
    int_time = times[1] - times[0]
    t_i = times[0] - int_time / 2
    t_f = times[-1] + int_time / 2
    n_gp_times = jnp.ceil(2.0 * ((t_f - t_i) / gp_l) + 1).astype(int)

    n_gp_times = jnp.where(n_gp_times < 2, 2, n_gp_times)
    sample_times = jnp.linspace(t_i, t_f, n_gp_times)

    return sample_times

