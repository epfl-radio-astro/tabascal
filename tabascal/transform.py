import jax.numpy as jnp
from jax import jit, lax
from jax.scipy.linalg import solve_triangular

from numpyro.distributions.transforms import Transform
from numpyro.distributions import constraints


@jit
def affine_transform_full(x, L, mu):
    return L @ x + mu


@jit
def affine_transform_full_inv(x, L_inv, mu):
    return L_inv @ (x - mu)


@jit
def affine_transform_diag(x, sigma, mu):
    return sigma * x + mu


@jit
def affine_transform_diag_inv(x, sigma_inv, mu):
    return sigma_inv * (x - mu)
