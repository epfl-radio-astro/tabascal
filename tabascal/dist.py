import numpyro.distributions as dist
import jax.numpy as jnp
from jax import Array
import numpyro


def standard_normal(name: str, shape: tuple):

    rv = numpyro.sample(name, dist.Normal(jnp.zeros(shape), jnp.ones(shape)))

    return rv
