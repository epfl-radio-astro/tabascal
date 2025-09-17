import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def gaussian(pred, obs, args):

    pred_ri = jnp.stack([pred.real, pred.imag], axis=0)
    obs_ri = jnp.stack([obs.real, obs.imag], axis=0)

    numpyro.sample(
        "obs",
        dist.Normal(pred_ri, args["noise"]),  # type: ignore
        obs=obs_ri,
    )
