import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from tabascal.distributed import constrain_baseline_axis


def gaussian(pred, obs, args):

    pred_ri = jnp.stack([pred.real, pred.imag], axis=0)
    obs_ri = jnp.stack([obs.real, obs.imag], axis=0)
    inv_flags = jnp.stack([~args["flags"], ~args["flags"]], axis=0)

    # Stacking put the baseline axis second, so say so: without this the
    # log-probability and its cotangents -- four arrays the size of the
    # visibilities between them -- are computed whole on every device even
    # when everything feeding them is divided. No-op unless sharding baselines.
    pred_ri, obs_ri, inv_flags = (
        constrain_baseline_axis(x, 1) for x in (pred_ri, obs_ri, inv_flags)
    )

    with numpyro.handlers.mask(mask=inv_flags):
        numpyro.sample(
            "obs",
            dist.Normal(pred_ri, args["noise"]),  # type: ignore
            obs=obs_ri,
        )
