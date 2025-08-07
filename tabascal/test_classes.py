from tabsim.config import load_config

from tabascal.component_functions import build_model

from tabascal.tab_tools import reduced_chi2
from tabascal.opt import SVIRunResult

from tabascal.config import TabConfig, Model
from tabascal.components.trajectory import FixedOrbit
from tabascal.components.rfi_signal import ComplexRFI
from tabascal.components.rfi_vis import RiemannVisCalculation
from tabascal.components.ast_vis import FourierTimeAst
from tabascal.components.gains import UnitaryGains

from numpyro.optim import optax_to_numpyro
import optax

from jax import random, jit
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from numpyro.infer import MCMC, NUTS, Predictive, SVI, autoguide, Trace_ELBO

import numpy as np


def run_svi(
    model,
    obs_data,
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
        obs_data=obs_data,
        init_params=init_params,
    )
    losses = svi_results.losses / obs_data.size
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
            obs_data=obs_data,
            init_params=svi_results.params,
        )
        losses = jnp.concatenate([losses, svi_results.losses / obs_data.size])
        svi_results = SVIRunResult(svi_results.params, svi_results.state, losses)

    return svi_results, guide


def svi_predict(
    model,
    guide,
    vi_params,
    num_samples=100,
    key=random.PRNGKey(2),
):
    predictive = Predictive(
        model=model, guide=guide, params=vi_params, num_samples=num_samples
    )
    predictions = predictive(key)

    return predictions


def tabascal_subtraction(
    config: dict,
    ms_path: str,
):

    tab_config = TabConfig(config, ms_path)

    components = [
        FixedOrbit,
        ComplexRFI,
        RiemannVisCalculation,
        FourierTimeAst,
        UnitaryGains,
    ]

    model = Model(tab_config, components)

    forward = jit(model.build_forward())

    state = {**model.init_params, **model.state_params}

    print(jnp.sum(forward(state)["vis_obs"]))

    prob_model = model.build_prob_model()

    shapes = {key: value.shape for key, value in model.init_params.items()}
    n_params = sum([x.size for x in model.init_params.values()])
    n_data = 2 * tab_config.vis_obs.size

    print(f"Parameter shapes     : {shapes}")
    print(f"Number of parameters : {n_params}")
    print(f"Number of data points: {n_data}")

    key = random.PRNGKey(1)
    key, subkey = random.split(key)

    pred = Predictive(
        model=prob_model,
        posterior_samples=tree_map(lambda x: x[None, :], model.init_params),
        batch_ndims=1,
    )
    # with jax.checking_leaks():
    #     init_pred = pred(subkey, obs_data=model_config["vis_obs"].T)

    init_pred = pred(subkey, obs_data=tab_config.vis_obs.T)
    rchi2 = reduced_chi2(
        init_pred["vis_obs"][0], tab_config.vis_obs.T, tab_config.noise
    )
    print()
    print(f"Reduced Chi^2 @ init params : {rchi2}")

    # print(tree_map(jnp.shape, init_pred))

    if config["inference"]["opt"]:
        guides = {
            "map": "AutoDelta",
        }

        subkeys = random.split(key)

        guide_family = guides[config["opt"]["guide"]]

        vi_results, vi_guide = run_svi(
            model=prob_model,
            obs_data=tab_config.vis_obs.T,
            max_iter=config["opt"]["max_iter"],
            guide_family=guide_family,
            init_params={k + "_auto_loc": v for k, v in model.init_params.items()},
            epsilon=config["opt"]["epsilon"],
            key=subkey,
            dual_run=config["opt"]["dual_run"],
        )

        vi_params = vi_results.params
        vi_pred = svi_predict(
            model=prob_model,
            guide=vi_guide,
            vi_params=vi_params,
            num_samples=1,
            key=subkeys[1],
        )

        rchi2 = reduced_chi2(
            vi_pred["vis_obs"][0], tab_config.vis_obs.T, tab_config.noise
        )
        print()
        print(f"Reduced Chi^2 @ opt params : {rchi2}")


if __name__ == "__main__":

    import argparse
    import os

    parser = argparse.ArgumentParser(description="Apply tabascal to a simulation.")
    parser.add_argument(
        "-c", "--config", required=True, help="Path to the config file."
    )
    parser.add_argument(
        "-ms", "--ms_path", required=True, help="Path to Measurement Set."
    )
    parser.add_argument(
        "-st", "--spacetrack", help="Path to Space-Track login details."
    )
    args = parser.parse_args()
    conf_path = args.config
    spacetrack_path = args.spacetrack

    config = load_config(conf_path, config_type="tab")

    if args.spacetrack:
        config["satellites"]["spacetrack_path"] = os.path.abspath(args.spacetrack)

    tabascal_subtraction(config, args.ms_path)
