from numpyro.optim import optax_to_numpyro
import optax

import jax
import jax.numpy as jnp
from jax import random
from jax.tree_util import tree_map
from numpyro.infer import Predictive, SVI, autoguide, Trace_ELBO

from tabascal.tab_tools import reduced_chi2
from tabascal.opt import SVIRunResult
from tabascal.plot import plot_predictions
from tabascal.config import TabConfig

import os

import xarray as xr
import dask.array as da

import matplotlib.pyplot as plt
import subprocess

from datetime import datetime

from typing import Callable


def run_svi(
    prob_model: Callable,
    obs_data: jax.Array,
    max_iter=1_000,
    guide_family="AutoDelta",
    init_params=None,
    epsilon=1e-3,
    key=random.PRNGKey(1),
    dual_run=True,
):
    if guide_family == "AutoDelta":
        guide = autoguide.AutoDelta(prob_model)
    elif guide_family == "AutoDiagonalNormal":
        guide = autoguide.AutoDiagonalNormal(prob_model)
    elif guide_family == "AutoLaplaceApproximation":
        guide = autoguide.AutoLaplaceApproximation(prob_model)
    elif guide_family == "AutoMultivariateNormal":
        guide = autoguide.AutoMultivariateNormal(prob_model)
    else:
        raise ValueError(f"Unknown guide_family: {guide_family}")

    # optimizer = numpyro.optim.Adam(epsilon)
    optimizer = optax_to_numpyro(optax.adabelief(epsilon))
    svi = SVI(prob_model, guide, optimizer, Trace_ELBO())
    svi_results = svi.run(
        key,
        max_iter,
        obs_data=obs_data,
        init_params=init_params,
    )
    losses = svi_results.losses / obs_data.size
    svi_results = SVIRunResult(svi_results.params, svi_results.state, losses)

    if dual_run:
        optimizer = optax_to_numpyro(optax.adabelief(epsilon / 10))
        svi = SVI(prob_model, guide, optimizer, Trace_ELBO())

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
    prob_model: Callable,
    guide: autoguide.AutoGuide,
    vi_params: dict,
    num_samples=100,
    key=random.PRNGKey(2),
):
    predictive = Predictive(
        model=prob_model, guide=guide, params=vi_params, num_samples=num_samples
    )
    predictions = predictive(key)

    return predictions


def write_results_xds(
    vi_pred: dict, tab_config: TabConfig, file_path: str, overwrite: bool = True
):

    # print(vi_pred.keys())
    # print(vi_pred["rfi_vis"].shape)
    # print(vi_pred["rfi_vis"])

    # print(da.asarray(vi_pred["ast_vis"]))
    # print(da.asarray(vi_pred["gains"]))
    # print(da.asarray(vi_pred["rfi_vis"]))
    # print(da.asarray(vi_pred["vis_obs"]))
    # print(da.asarray(vi_pred["rfi_A"]))
    # print(da.asarray(args["rfi_phase"]))

    map_xds = xr.Dataset(
        data_vars={
            "rfi_vis": (["sample", "bl", "time"], da.asarray(vi_pred["vis_rfi"])),
            "ast_vis": (["sample", "bl", "time"], da.asarray(vi_pred["vis_ast"])),
            "gains": (["sample", "ant", "time"], da.asarray(vi_pred["gains"])),
            "vis_obs": (["sample", "bl", "time"], da.asarray(vi_pred["vis_obs"])),
            # "rfi_A": (
            #     ["sample", "src", "ant", "rfi_time"],
            #     da.asarray(vi_pred["rfi_A"]),
            # ),
            # "rfi_phase": (
            #     ["src", "ant", "time_mjd_fine"],
            #     da.asarray(args["rfi_phase"]),
            # ),
        },
        coords={
            "time": da.asarray(tab_config.times),
            # "rfi_time": da.asarray(args["rfi_times"]),
            # "time_mjd_fine": da.asarray(args["times_mjd_fine"]),
        },
    )
    # print(map_xds)

    mode = "w" if overwrite else "w-"

    map_xds.to_zarr(file_path, mode=mode)

    return map_xds


def init_predict(
    tab_config: TabConfig, prob_model: Callable, subkey: jax.Array, init_params: dict
):

    pred = Predictive(
        model=prob_model,
        posterior_samples=tree_map(lambda x: x[None, :], init_params),
        batch_ndims=1,
    )
    init_pred = pred(subkey)
    rchi2 = reduced_chi2(
        init_pred["vis_obs"][0], tab_config.vis_obs.T, tab_config.noise
    )
    print()
    print(f"Reduced Chi^2 @ init params : {rchi2}")

    return init_pred


def plot_init(
    tab_config: TabConfig, init_pred: dict, truth: dict, model_name: str, plot_dir: str
):

    start = datetime.now()
    print()
    print("Plotting Initial Parameters")
    plot_predictions(
        times=tab_config.times,
        pred=init_pred,
        truth=truth,
        type="init",
        model_name=model_name,
        max_plots=10,
        save_dir=plot_dir,
    )
    # if get_truth_conditional(config):

    #     # vi_pred keys are ['ast_vis', 'gains', 'rfi_vis', 'rmse_ast', 'rmse_gains', 'rmse_rfi', 'vis_obs']
    #     print(f"RMSE Gains      : {jnp.mean(init_pred['rmse_gains']):.5f}")
    #     print(f"RMSE RFI Vis    : {jnp.mean(init_pred['rmse_rfi']):.5f}")
    #     print(f"RMSE AST Vis    : {jnp.mean(init_pred['rmse_ast']):.5f}")

    print()
    print(f"Initial Plot Time : {datetime.now() - start}")
    print(f"{datetime.now()}")


def plot_prior(
    tab_config: TabConfig,
    prob_model: Callable,
    truth: dict,
    model_name: str,
    subkey: jax.Array,
    plot_dir: str,
):

    start = datetime.now()
    n_prior = tab_config.args["plots"]["prior_samples"]
    print()
    print(f"Plotting {n_prior:.0f} Prior Parameter Samples")
    pred = Predictive(prob_model, num_samples=n_prior)
    prior_pred = pred(subkey)
    print("Prior Samples Drawn")
    plot_predictions(
        times=tab_config.times,
        pred=prior_pred,
        truth=truth,
        type="prior",
        model_name=model_name,
        max_plots=10,
        save_dir=plot_dir,
    )
    print()
    print(f"Prior Plot Time : {datetime.now() - start}")
    print(f"{datetime.now()}")


def run_opt(
    tab_config: TabConfig,
    prob_model: Callable,
    truth: dict[str, jax.Array],
    model_name: str,
    subkeys: jax.Array,
    init_params: dict,
    plot_dir: str,
    ms_path,
    map_path,
    params_path,
):

    guides = {
        "map": "AutoDelta",
    }
    start = datetime.now()
    print()
    print("Running Optimization ...")
    guide_family = guides[tab_config.args["opt"]["guide"]]
    vi_results, vi_guide = run_svi(
        prob_model=prob_model,
        obs_data=tab_config.vis_obs.T,
        max_iter=tab_config.args["opt"]["max_iter"],
        guide_family=guide_family,
        init_params={
            **{k + "_auto_loc": v for k, v in init_params.items()},
        },
        epsilon=tab_config.args["opt"]["epsilon"],
        key=subkeys[0],
        dual_run=tab_config.args["opt"]["dual_run"],
    )
    vi_params = vi_results.params
    vi_pred = svi_predict(
        prob_model=prob_model,
        guide=vi_guide,
        vi_params=vi_params,
        num_samples=1,
        key=subkeys[1],
    )
    print()
    print(f"Optimization Run Time : {datetime.now() - start}")
    print(f"{datetime.now()}")
    start = datetime.now()

    write_results_xds(vi_pred, tab_config, map_path)
    # write_params_xds(vi_params, gp_params, ms_params, params_path, overwrite=True)

    plot_predictions(
        tab_config.times,
        pred=vi_pred,
        truth=truth,
        type=tab_config.args["opt"]["guide"],
        model_name=model_name,
        max_plots=10,
        save_dir=plot_dir,
    )
    print()
    print(f"Optimize Plot Time : {datetime.now() - start}")
    print(f"{datetime.now()}")

    # if get_truth_conditional(config):

    #     # vi_pred keys are ['ast_vis', 'gains', 'rfi_vis', 'rmse_ast', 'rmse_gains', 'rmse_rfi', 'vis_obs']
    #     print(f"RMSE Gains      : {jnp.mean(vi_pred['rmse_gains']):.5f}")
    #     print(f"RMSE RFI Vis    : {jnp.mean(vi_pred['rmse_rfi']):.5f}")
    #     print(f"RMSE AST Vis    : {jnp.mean(vi_pred['rmse_ast']):.5f}")

    rchi2 = reduced_chi2(vi_pred["vis_obs"][0], tab_config.vis_obs.T, tab_config.noise)
    print()
    print(f"Reduced Chi^2 @ opt params : {rchi2}")

    plt.semilogy(vi_results.losses)
    plt.savefig(os.path.join(plot_dir, f"{model_name}_opt_loss.pdf"), format="pdf")

    print()
    print(
        "Copying tabascal results to MS file in 'TAB_DATA' and 'TAB_RFI_DATA' columns"
    )
    print(os.path.split(map_path)[1])
    subprocess.run(
        f"tab2MS -m {ms_path} -z {map_path}", shell=True, executable="/bin/bash"
    )

    return vi_params, rchi2


def save_memory(mem_dir, mem_i):

    mem_i += 1
    jax.profiler.save_device_memory_profile(
        os.path.join(mem_dir, f"memory_{mem_i}.prof")
    )

    return mem_i


def get_observation_data_type(data_col):

    ast = ["DATA", "CAL_DATA", "AST_DATA", "AST_MODEL_DATA"]
    rfi = ["DATA", "CAL_DATA", "RFI_DATA", "RFI_MODEL_DATA"]
    gains = ["DATA"]

    data_type = {
        "ast": data_col in ast,
        "rfi": data_col in rfi,
        "gains": data_col in gains,
    }

    return data_type
