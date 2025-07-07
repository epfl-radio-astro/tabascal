import numpyro.distributions as dist
import jax.numpy as jnp
from jax import Array
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

    rv = numpyro.sample(
        name,
        dist.Normal(jnp.zeros(shape), jnp.ones(shape)),
    )

    return rv
