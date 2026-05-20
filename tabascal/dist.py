import numpyro.distributions as dist
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import ndtr, ndtri
import numpyro


def MVN(name: str, mu: Array, L: Array):
    rv = numpyro.sample(
        name,
        dist.TransformedDistribution(
            dist.Normal(jnp.zeros(mu.shape), jnp.ones(mu.shape)),
            [
                dist.transforms.LowerCholeskyAffine(mu, L),
            ],
        ),
    )
    return rv


def Normal(name: str, mu: Array, sigma: Array):
    rv = numpyro.sample(
        name,
        dist.TransformedDistribution(
            dist.Normal(jnp.zeros(mu.shape), jnp.ones(mu.shape)),
            [
                dist.transforms.AffineTransform(mu, sigma),
            ],
        ),
    )
    return rv


def standard_normal(name: str, shape: tuple):

    rv = numpyro.sample(name, dist.Normal(jnp.zeros(shape), jnp.ones(shape)))

    return rv


def gaussian_to_laplace(z: Array, scale: float, eps: float = 1e-12) -> Array:
    """Map white standard-normal samples to white zero-mean Laplace samples.

    Composes the normal CDF with the Laplace inverse-CDF, so each independent
    ``z ~ N(0, 1)`` maps to an independent ``Laplace(0, scale)`` (whiteness
    preserved). A zero-centred Laplace is a sparsity (LASSO-like) prior whose
    only parameter is the width ``scale`` (b); variance is ``2·scale²``.

    Parameters
    ----------
    z : Array
        White Gaussian base parameters.
    scale : float
        Laplace width ``b`` (> 0).
    """
    u = jnp.clip(ndtr(z), eps, 1.0 - eps)              # Phi(z) in (0, 1)
    centred = u - 0.5
    return -scale * jnp.sign(centred) * jnp.log1p(-2.0 * jnp.abs(centred))


def laplace_to_gaussian(x: Array, scale: float, eps: float = 1e-12) -> Array:
    """Inverse of :func:`gaussian_to_laplace` (Laplace(0, scale) -> N(0, 1))."""
    cdf = 0.5 + 0.5 * jnp.sign(x) * (1.0 - jnp.exp(-jnp.abs(x) / scale))
    return ndtri(jnp.clip(cdf, eps, 1.0 - eps))
