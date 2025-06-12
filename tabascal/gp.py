import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.sparse.linalg import cg
from functools import partial, reduce

# @jit
# def kernel(x, var, l, noise=1e-5):
#     """
#     x: array (n_points, n_dim)
#     """
#     chi = jnp.linalg.norm(x[None, :, :] - x[:, None, :], axis=-1) / (l)
#     cov = jnp.abs(var) * jnp.exp(-0.5 * chi**2) + noise * jnp.eye(x.shape[0])
#     return cov

# @jit
# def inv_kernel(x, var, l):
#     return inv_sym(kernel(x, var, l))


# @jit
# def inv_kernel_vmap(x, var, l):
#     return vmap(inv_kernel, in_axes=(None, 0, 0))(x, var, l)


@jit
def base_kernel(x_in, x_out, var, l):

    x_in = jnp.atleast_2d(x_in).T
    x_out = jnp.atleast_2d(x_out).T

    chi = jnp.linalg.norm(x_in[None, :, :] - x_out[:, None, :], axis=-1) / l
    K = jnp.abs(var) * jnp.exp(-0.5 * chi**2)

    return jnp.squeeze(K)


@jit
def cholesky(x_in, var, l, noise=1e-8):

    L = jnp.linalg.cholesky(
        base_kernel(x_in, x_in, var, l) + noise * jnp.eye(x_in.shape[0])
    )

    return L


@jit
def gp_resample_otf(y_in, x_in, x_out, var, l, noise=1e-8):

    noise = noise * jnp.eye(len(y_in))
    K = lambda y: vmap(
        lambda x2, n: (base_kernel(x_in, x2, var, l) + n) @ y, in_axes=(0, 0)
    )(x_in, noise)
    y_inv = cg(K, y_in)[0]

    y_out = vmap(lambda x2: (base_kernel(x_in, x2, var, l) @ y_inv))(x_out)

    return y_out


@partial(jit, static_argnums=(1,))
def supersample(signal, factor):

    N = len(signal)

    # FFT and supersample
    X = jnp.fft.fft(signal)
    N_ss = N * factor
    X_ss = jnp.zeros(N_ss, dtype=complex)

    # Copy frequencies
    pos_freq = (N + 1) // 2
    neg_freq = (N - 1) // 2
    X_ss = X_ss.at[:pos_freq].set(X[:pos_freq])
    X_ss = X_ss.at[-neg_freq:].set(X[-neg_freq:])

    # IFFT and scale
    supersampled = jnp.fft.ifft(X_ss) * factor

    return supersampled


@partial(jit, static_argnums=(1,))
def gp_resample_fft(signal, factor):

    pad_length = len(signal) // 2
    # Pad the signal
    padded = jnp.pad(signal, pad_length, mode="linear_ramp")

    # Super sample padded signal
    supersampled = supersample(padded, factor)

    # print(supersampled.shape)

    # Remove padding
    pad_new = pad_length * factor
    signal_supersampled = supersampled[pad_new : -pad_new + 1].real

    # print(signal_supersampled.shape)

    return signal_supersampled


@jit
def kernel(x, x_, var, l, noise=1e-3):

    x = x[:, None] if x.ndim == 1 else x
    x_ = x_[:, None] if x_.ndim == 1 else x_
    chi = jnp.linalg.norm(x[None, :, :] - x_[:, None, :], axis=-1) / l
    cov = jnp.abs(var) * jnp.exp(-0.5 * chi**2)
    if chi.shape[0] == chi.shape[1]:
        cov += noise * jnp.eye(x.shape[0])

    return cov


@jit
def resampling_kernel(x, x_, var, l, noise=1e-3):
    K_inv = jnp.linalg.inv(kernel(x, x, var, l, noise))
    K_s = kernel(x, x_, var, l)
    return K_s @ K_inv


@jit
def gp_resample(y, x, x_, var, l, noise=1e-3):
    K_inv = jnp.linalg.inv(kernel(x, x, var, l, noise))
    K_s = kernel(x, x_, var, l)
    return K_s @ K_inv @ y


@jit
def l_from_uv(uv, l0=7e2, a=6e-4):
    return l0 * jnp.exp(-a * uv)


def get_times(times, gp_l):
    int_time = times[1] - times[0]
    t_i = times[0] - int_time / 2
    t_f = times[-1] + int_time / 2
    n_gp_times = jnp.ceil(2.0 * ((t_f - t_i) / gp_l) + 1).astype(int)

    n_gp_times = find_closest_factor_greater_than(len(times), n_gp_times) + 1

    n_gp_times = jnp.where(n_gp_times < 2, 2, n_gp_times)
    sample_times = jnp.linspace(t_i, t_f, n_gp_times)
    return sample_times


def get_vis_gp_params(ants_uvw, vis_ast, a1, a2, noise):

    mag_uvw = jnp.linalg.norm(ants_uvw[0, a1] - ants_uvw[0, a2], axis=-1)
    vis_var = (jnp.abs(vis_ast).max(axis=(0, 2)) + noise) ** 2
    vis_l = l_from_uv(mag_uvw, l0=5e2)

    return vis_var, vis_l


def find_factors(n):
    """Find the factor of a number n.

    Parameters:
    -----------
    n : int
        The number to find the factors of.

    Returns:
    --------
    factors : list
        The unique factors of n.
    """
    return list(
        set(
            reduce(
                list.__add__,
                ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0),
            )
        )
    )


def find_closest_factor_greater_than(N: int, n: int) -> int:
    """Find the closest factor of N to n that is greater or equal to n.

    Parameters:
    -----------
    N : int
        Number to find the factors of.
    n : int
        Number where the factor should be closest to AND greater than.

    Returns:
    --------
    n_ : int
        Factor of N that is cloest to n.
    """

    facs = jnp.sort(jnp.array(find_factors(N)))
    factor_diff = facs - n
    greater = factor_diff >= 0
    facs = facs[greater]
    factor_diff = factor_diff[greater]

    return facs[jnp.argmin(jnp.abs(factor_diff))]
