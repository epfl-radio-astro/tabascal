from numpyro.optim import optax_to_numpyro
from numpyro.infer import Predictive, SVI, autoguide, Trace_ELBO

import optax

import jax
import jax.numpy as jnp
from jax import random
from jax.tree_util import tree_map

from tabascal.opt import SVIRunResult
from tabascal.plot import plot_predictions
from tabascal.timing import measure_runtime

import os

import xarray as xr
import dask.array as da

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

from typing import Callable, Optional

from daskms import xds_from_ms, xds_from_table

from numpyro.infer import log_likelihood
from numpyro.infer.util import log_density


def nlog_like(prob_model, params, obs_data):

    nlog_l = -log_likelihood(prob_model, params, obs_data=obs_data, batch_ndims=0)[
        "obs"
    ].mean()

    return nlog_l


def nlog_post(prob_model, params, obs_data):

    nlog_p = (
        -log_density(
            prob_model,
            model_args=(obs_data,),
            model_kwargs={},
            params=params,
        )[0]
        / obs_data.size
        / 2
    )

    return nlog_p


def reduced_chi2(pred, true, noise):

    complex_types = [
        complex,
        np.complex64,
        np.complex128,
        jnp.complex64,
        jnp.complex128,
    ]
    is_complex = jnp.any(jnp.array([true.dtype == c_type for c_type in complex_types]))
    if is_complex:
        # print("Complex Data")
        norm = 2 * true.size
    else:
        norm = true.size

    rchi2 = jnp.sum((jnp.abs(pred - true) / noise) ** 2) / norm

    return rchi2


def get_ast_fringe_rate(uv, freq=1.227e9, D=13.5):

    omega = 2 * np.pi / (24 * 3600)
    lam = 3e8 / freq

    U = jnp.max(jnp.linalg.norm(uv, axis=-1), axis=0)

    bw = 1.22 * lam / D
    max_l = jnp.sin(bw / 2)

    max_fr = omega * U * max_l / lam

    return max_fr


def pow_spec(k, P0=1e7, k0=1e-3, gamma=1.0):

    k_ = k / k0
    Pk = P0 * 0.5 * (jnp.exp(-(k_**2)) + (1.0 + k_**2) ** -gamma)
    # Pk = P0 / (1.0 + k_**2) ** gamma
    # Pk = P0 * jnp.exp(-(k_**2)) # Leads to NaN values after division

    return Pk


def fix_padding(config: dict, n_freq):

    try:
        if (
            config["rfi"]["freq_pad_factor"] < 3
            and n_freq == 1
            and config["rfi"]["freq_int_samples"] > 1
        ):
            config["rfi"]["freq_pad_factor"] = 3
    except:
        print("freq_pad_factor is not defined")

    return config


@measure_runtime
def read_ms(
    ms_path,
    freq: Optional[float] = None,
    chans: Optional[jax.Array] = None,
    corr: str = "xx",
    data_col: str = "DATA",
):

    correlations = {"xx": 0, "xy": 1, "yx": 2, "yy": 3}
    corr_idx = correlations[corr]

    xds = xds_from_ms(ms_path)[0]
    xds_ant = xds_from_table(ms_path + "::ANTENNA")[0]
    xds_spec = xds_from_table(ms_path + "::SPECTRAL_WINDOW")[0]
    xds_src = xds_from_table(ms_path + "::SOURCE")[0]

    ants_itrf = jnp.array(xds_ant.POSITION.data.compute())

    n_ant = ants_itrf.shape[0]
    n_time = len(np.unique(xds.TIME.data.compute()))
    n_bl = xds.DATA.data.shape[0] // n_time
    n_freq, n_corr = xds.DATA.data.shape[1:]

    freqs = jnp.array(xds_spec.CHAN_FREQ.data[0].compute())
    int_time = xds.INTERVAL.data[0].compute()

    times_mjd = jnp.array(xds.TIME.data.reshape(n_time, n_bl)[:, 0].compute())
    if times_mjd[1] - times_mjd[0] > 0.5:
        times_mjd = times_mjd / (24 * 3600)

    # times_mjd = jnp.array(xds.TIME.data.reshape(n_time, n_bl)[:, 0].compute()) / (
    #     24 * 3600
    # )
    from astropy.time import Time

    print(Time(times_mjd[0], format="mjd").isot)

    times = jnp.linspace(0, n_time * int_time, n_time, endpoint=False)

    if chans is None:
        if freq:
            chans = jnp.argmin(jnp.abs(freq - freqs))
        else:
            chans = jnp.arange(n_freq)

    n_freq = len(chans)

    data = {
        **{
            key: val
            for key, val in zip(
                ["ra", "dec"], jnp.rad2deg(xds_src.DIRECTION.data[0].compute())
            )
        },
        "n_freq": n_freq,
        "n_corr": n_corr,
        "n_time": n_time,
        "n_ant": n_ant,
        "n_bl": n_bl,
        "dish_d": xds_ant.DISH_DIAMETER.data[0].compute(),
        "times_mjd": times_mjd,
        "times": times,
        "int_time": int_time,
        "freqs": freqs[chans],
        "ants_itrf": ants_itrf,
        "uvw": jnp.array(xds.UVW.data.reshape(n_time, n_bl, 3).compute()),
        "vis_obs": jnp.transpose(
            jnp.array(
                xds[data_col]
                .data.reshape(n_time, n_bl, n_freq, n_corr)
                .compute()[:, :, chans, corr_idx]
            ),
            (1, 2, 0),
        ),
        "noise": jnp.array(xds.SIGMA.data.mean().compute()),
        "a1": jnp.array(xds.ANTENNA1.data.reshape(n_time, n_bl)[0, :].compute()),
        "a2": jnp.array(xds.ANTENNA2.data.reshape(n_time, n_bl)[0, :].compute()),
    }

    return data


@measure_runtime
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


@measure_runtime
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


@measure_runtime 
def write_results_xds(
    vi_pred: dict, tab_config, file_path: str, overwrite: bool = True
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
            "rfi_vis": (["sample", "bl", "freq", "time"], da.asarray(vi_pred["vis_rfi"])),  # type: ignore
            "ast_vis": (["sample", "bl", "freq", "time"], da.asarray(vi_pred["vis_ast"])),  # type: ignore
            "gains": (["sample", "ant", "freq", "time"], da.asarray(vi_pred["gains"])),  # type: ignore
            "vis_obs": (["sample", "bl", "freq", "time"], da.asarray(vi_pred["vis_obs"])),  # type: ignore
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
            "time": da.asarray(tab_config.times),  # type: ignore
            "freq": da.asarray(tab_config.freqs),  # type: ignore
            # "rfi_time": da.asarray(args["rfi_times"]),
            # "time_mjd_fine": da.asarray(args["times_mjd_fine"]),
        },
    )
    # print(map_xds)

    mode = "w" if overwrite else "w-"

    map_xds.to_zarr(file_path, mode=mode)

    return map_xds


@measure_runtime
def init_predict(
    tab_config, prob_model: Callable, subkey: jax.Array, init_params: dict
):

    pred = Predictive(
        model=prob_model,
        posterior_samples=tree_map(lambda x: x[None, :], init_params),
        batch_ndims=1,
    )
    init_pred = pred(subkey)
    rchi2 = reduced_chi2(init_pred["vis_obs"][0], tab_config.vis_obs, tab_config.noise)
    print()
    print(f"Reduced Chi^2 @ init params : {rchi2}")

    return init_pred


@measure_runtime
def plot_init(tab_config, init_pred: dict, truth: dict, model_name: str, plot_dir: str):

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


@measure_runtime
def plot_prior(
    tab_config,
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


@measure_runtime
def run_opt(
    tab_config,
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
        obs_data=tab_config.vis_obs,
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

    rchi2 = reduced_chi2(vi_pred["vis_obs"][0], tab_config.vis_obs, tab_config.noise)
    print()
    print(f"Reduced Chi^2 @ opt params : {rchi2}")

    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(vi_results.losses)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Iteration")
    if vi_results.losses.min() < 0:
        ax.set_yscale("symlog")
    else:
        ax.set_yscale("log")
    plt.savefig(
        os.path.join(plot_dir, f"{model_name}_opt_loss.pdf"),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()

    print()
    print(f"Copying tabascal results to MS file from {map_path}")

    from tabascal.write import write_results

    write_results(ms_path, map_path, tab_config.args["data"]["data_col"])
    

    return vi_params, rchi2


def get_observation_data_type(data_col: str):

    ast = ["DATA", "CAL_DATA", "AST_DATA", "AST_MODEL_DATA"]
    rfi = ["DATA", "CAL_DATA", "RFI_DATA", "RFI_MODEL_DATA"]
    gains = ["DATA"]

    data_type = {
        "ast": data_col in ast,
        "rfi": data_col in rfi,
        "gains": data_col in gains,
    }

    return data_type
