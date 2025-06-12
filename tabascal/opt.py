from numpyro.infer import Predictive, SVI, Trace_ELBO, autoguide
import numpyro
import jax
import jax.numpy as jnp

from jax.flatten_util import ravel_pytree
from functools import partial
from numpyro.infer import log_likelihood
from numpyro.infer.util import log_density
from numpyro.optim import optax_to_numpyro

import optax
from optax.second_order import hessian_diag

from jax import jvp, vjp, jit, vmap, random
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree as flatten
from jax.scipy.sparse.linalg import cg
from functools import partial


# Taken directly from numpyro
from collections import namedtuple

SVIRunResult = namedtuple("SVIRunResult", ["params", "state", "losses"])
"""
A :func:`~collections.namedtuple` consisting of the following fields:
 - **params** - the optimized parameters.
 - **state** - the last :data:`SVIState`
 - **losses** - the losses collected at every step.
"""

H = jit(hessian_diag, static_argnums=(0,))
# H = hessian_diag


@partial(jit, static_argnums=(0,))
def fisher_vp(f, w, v):
    """Calculate the Fisher vector product. Taken from https://gebob19.github.io/natural-gradient/"""
    # J v
    _, Jv = jax.jvp(f, (w,), (v,))
    # (J v)^T J = v^T (J^T J)
    _, f_vjp = jax.vjp(f, w)
    return f_vjp(Jv)[0]


@partial(jit, static_argnums=(0,))
def fisher_diag(nll, params):
    """Calculate the diagonal of the Fisher information matrix. Taken from Optax second_order.py (hessian_diag)."""
    params_flat, unflatten = ravel_pytree(params)
    N = len(params_flat)
    vs = jnp.eye(N)
    comp = lambda v: jnp.vdot(v, ravel_pytree(fisher_vp(nll, params, unflatten(v)))[0])
    return jax.vmap(comp)(vs)


# @partial(jit, static_argnums=(0,))
def fisher_diag_inv(
    model,
    params: dict,
    kwargs_model: dict,
    kwargs_data: dict,
    likelihood_site_name: str = "obs",
):
    """Calculate the inverse of the diagonal of the Fisher information matrix.

    Parameters:
    -----------
    model: NumPyro model
        A numpyro model that is called as model(**kwargs_model, **kwargs_data).
    params: dict
        Parameter values at which to evaluate the posterior fisher information.
    kwargs_model: dict
        Keyword arguments to be passed ot the model.
    kwargs_data: dict
        Keyword arguments where the observed data is passed to the model.
    likelihood_site_name: str
        The model site name where the observed data is passed to in the form of 'obs=kwargs_data.values()'.

    Returns:
    --------
    fisher_diag_inv:
        Flattened array of the inverse diagonal fisher.
    """
    # _, unflatten = ravel_pytree(params)
    # params = jax.tree_map(lambda x: jnp.atleast_1d(x)[None, :], params)
    nlp = (
        lambda x, _, __: -1
        * log_density(model, model_args=(), model_kwargs=kwargs_model, params=x)[0]
    )
    nll = lambda x: log_likelihood(
        model, posterior_samples=x, batch_ndims=0, **kwargs_model, **kwargs_data
    )[likelihood_site_name]
    # fisher_prior = jnp.abs(H(nlp, params, None, None))
    fisher_prior = 0.0
    fisher_likelihood = fisher_diag(nll, params)

    return 1.0 / (fisher_prior + fisher_likelihood)


@partial(jit, static_argnums=(0,))
def fisher_diag_inv2(
    model,
    params: dict,
    kwargs_model: dict,
    kwargs_data: dict,
    likelihood_site_name: str = "obs",
):
    nlpost = (
        lambda x, _, __: -1
        * log_density(
            model, model_args=(), model_kwargs={**kwargs_model, **kwargs_data}, params=x
        )[0]
    )
    fisher_post = H(nlpost, params, None, None)

    return 1.0 / jnp.abs(fisher_post)


def run_svi(
    model,
    # args,
    static_args,
    array_args,
    obs,
    max_iter=1_000,
    guide_family="AutoDelta",
    init_params=None,
    epsilon=1e-3,
    key=random.PRNGKey(1),
    dual_run=True,
):
    if guide_family == "AutoDelta":
        guide = autoguide.AutoDelta(model)
    elif guide_family == "AutoDiagonalNormal":
        guide = autoguide.AutoDiagonalNormal(model)
    elif guide_family == "AutoLaplaceApproximation":
        guide = autoguide.AutoLaplaceApproximation(model)
    elif guide_family == "AutoMultivariateNormal":
        guide = autoguide.AutoMultivariateNormal(model)
    else:
        raise ValueError(f"Unknown guide_family: {guide_family}")

    # optimizer = numpyro.optim.Adam(epsilon)
    optimizer = optax_to_numpyro(optax.adabelief(epsilon))
    svi = SVI(model, guide, optimizer, Trace_ELBO())
    # svi_results = svi.run(key, max_iter, args=args, v_obs=obs, init_params=init_params)
    svi_results = svi.run(
        key,
        max_iter,
        static_args=static_args,
        array_args=array_args,
        v_obs=obs,
        init_params=init_params,
    )
    losses = svi_results.losses / obs.size
    svi_results = SVIRunResult(svi_results.params, svi_results.state, losses)

    # params = svi_results.params
    # losses = svi_results.losses
    if dual_run:
        optimizer = optax_to_numpyro(optax.adabelief(epsilon / 10))
        svi = SVI(model, guide, optimizer, Trace_ELBO())
        # svi_results = svi.run(
        #     key, max_iter, args=args, v_obs=obs, init_params=svi_results.params
        # )
        svi_results = svi.run(
            key,
            max_iter,
            static_args=static_args,
            array_args=array_args,
            v_obs=obs,
            init_params=svi_results.params,
        )
        losses = jnp.concatenate([losses, svi_results.losses / obs.size])
        svi_results = SVIRunResult(svi_results.params, svi_results.state, losses)

    return svi_results, guide


# def svi_predict(model, guide, vi_params, args, num_samples=100, key=random.PRNGKey(2)):
#     predictive = Predictive(
#         model=model, guide=guide, params=vi_params, num_samples=num_samples
#     )
#     predictions = predictive(key, args=args)

#     return predictions


def svi_predict(
    model,
    guide,
    vi_params,
    static_args,
    array_args,
    num_samples=100,
    key=random.PRNGKey(2),
):
    predictive = Predictive(
        model=model, guide=guide, params=vi_params, num_samples=num_samples
    )
    predictions = predictive(key, static_args=static_args, array_args=array_args)

    return predictions


@jit
def flatten_obs(vis_obs):
    return jnp.concatenate([vis_obs.real.flatten(), vis_obs.imag.flatten()])


@partial(jit, static_argnums=(0,))
def f_model_flat(model, params, args):
    vis_obs = model(params, args)
    vis_obs_flat = flatten_obs(vis_obs)
    return vis_obs_flat


@partial(jit, static_argnums=(0,))
def vjp_(f, x, y):
    _, vjp_fun = vjp(f, x)
    return vjp_fun(y)[0]


@partial(jit, static_argnums=(0,))
def fvp(f, x, v):
    pushforward = jvp(f, (x,), (v,))[1]
    _, vjp_fun = vjp(f, x)
    return vjp_fun(pushforward)[0]


@partial(jit, static_argnums=(0,))
def post_fvp(f, x, v):
    return tree_map(jnp.add, fvp(f, x, v), v)


@partial(jit, static_argnums=(0,))
def inv_post_fvp(f, x, v, max_iter):
    return cg(lambda v: post_fvp(f, x, v), v, maxiter=max_iter)[0]


@partial(jit, static_argnums=(0, 4, 6))
def post_samples(
    f, x, y_obs, noise_sigma, num_samples=10, key=random.PRNGKey(1), max_iter=1_000
):
    """Sample from a Gaussian approximation of the posterior distribution about the MAP point.

    Parameters
    ----------
    f : Callable
        JAX differentiable model forward mapping i.e. f(x) = y.
    x : JAX tree
        JAX tree of parameters.
    y_obs : Array
        JAX array of observed y values.
    noise_sigma : float
        Standard deviation of the noise for the Gaussian likelihood.
    num_samples : int, optional
        Number of posterior samples to draw, by default 10.
    key : PRNGKey, optional
        Pseudo-random number generator key, by default random.PRNGKey(1).
    max_iter : int, optional
        Maximum number of conjugate gradient iterations used in Fisher inversion, by default 1_000.

    Returns
    -------
    JAX tree
        JAX tree of parameter samples.
    """
    # Normalize function output such that Likelihood is N(0,1)
    f_norm = lambda x: f(x) / noise_sigma
    key, *subkeys = random.split(key, num=3)
    flat_x, unflatten = flatten(x)
    # Get Prior Samples where the prior is assumed to be dphi ~ N(0,1)
    p_samples = vmap(lambda x: unflatten(x))(
        random.normal(subkeys[0], (num_samples, flat_x.size))
    )

    # Get Likelihood noise samples where the likelihood is assumed to be deta ~ N(0,sigma_n^2)
    l_samples = random.normal(subkeys[1], (num_samples, y_obs.size))
    # Transform Likelihood samples as dtheta' = J^T Sigma^-1_n deta
    l_samples_trans = vmap(lambda y: vjp_(f_norm, x, y), in_axes=(0,))(l_samples)
    # Add transformed likelihood and prior samples
    samples = tree_map(jnp.add, l_samples_trans, p_samples)
    in_axes = (tree_map(lambda _: 0, x),)
    # Apply inverse Posterior Fisher to transformed samples
    param_deltas = vmap(
        lambda v: inv_post_fvp(f_norm, x, v, max_iter=max_iter), in_axes=in_axes
    )(samples)
    # Join samples with anti-correlated versions to increase sample size and leave mean @ MAP
    param_deltas_ = tree_map(lambda x: jnp.concatenate([x, -x], axis=0), param_deltas)

    return param_deltas_
