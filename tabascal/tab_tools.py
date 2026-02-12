from tqdm import trange

from numpyro.optim import optax_to_numpyro
from numpyro.infer import Predictive, SVI, autoguide, Trace_ELBO

import optax

import jax
import jax.numpy as jnp
from jax import random, Array
from jax.tree_util import tree_map

from tabascal.opt import SVIRunResult
from tabascal.timing import measure_runtime
from tabascal.write import write_results_ms, write_results_xds

import numpy as np

from datetime import datetime

from typing import Callable, Optional
from functools import reduce, partial

from daskms import xds_from_ms, xds_from_table

from numpyro.infer import log_likelihood
from numpyro.infer.util import log_density


@partial(jax.jit, static_argnums=(0, 1))
def _map_step(model, optimizer, params, opt_state, state, constants, obs_data):
    """Single MAP optimization step.

    state, constants, and obs_data are explicit traced arguments, not closure captures,
    so large arrays are never embedded as XLA constants.
    """
    def neg_log_post(params):
        lp, _ = log_density(model, (obs_data,), {"state": state, "constants": constants}, params)
        return -lp / obs_data.size

    loss, grads = jax.value_and_grad(neg_log_post)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss


@measure_runtime
def nlog_like(prob_model, params, obs_data, state=None, constants=None):

    nlog_l = -log_likelihood(
        prob_model, params, obs_data=obs_data, state=state, constants=constants, batch_ndims=0
    )["obs"].mean()

    return nlog_l


@measure_runtime
def nlog_post(prob_model, params, obs_data, state=None, constants=None):

    nlog_p = (
        -log_density(
            prob_model,
            model_args=(obs_data,),
            model_kwargs={"state": state, "constants": constants},
            params=params,
        )[0]
        / obs_data.size
        / 2
    )

    return nlog_p


def reduced_chi2(pred: Array, true: Array, noise: Array, flags: Array):

    complex_types = [
        complex,
        np.complex64,
        np.complex128,
        jnp.complex64,
        jnp.complex128,
    ]
    dtype = [true.dtype == c_type for c_type in complex_types]
    is_complex = reduce(jnp.logical_or, dtype)
    if is_complex:
        norm = 2 * true[~flags].size
    else:
        norm = true[~flags].size

    rchi2 = jnp.sum((jnp.abs(pred[~flags] - true[~flags]) / noise) ** 2) / norm

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

    ants_itrf = np.array(xds_ant.POSITION.data.compute())

    n_ant = ants_itrf.shape[0]
    n_time = len(np.unique(xds.TIME.data.compute()))
    n_bl = xds[data_col].data.shape[0] // n_time
    n_freq, n_corr = xds[data_col].data.shape[1:]

    freqs = np.array(xds_spec.CHAN_FREQ.data[0].compute())
    chan_width = np.array(xds_spec.CHAN_WIDTH.data[0, 0].compute())
    int_time = xds.INTERVAL.data[0].compute()

    times_mjd = np.array(xds.TIME.data.reshape(n_time, n_bl)[:, 0].compute())
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

    print(n_freq, chans)

    read_data = lambda col_name: jnp.transpose(
        jnp.array(
            xds[col_name]
            # .data[:, chans, corr_idx].reshape(n_time, n_bl, n_freq)
            .data[:, :, corr_idx].reshape(n_time, n_bl, n_freq)
            .compute()
        ),
        (1, 2, 0),
    )

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
        "chan_width": chan_width,
        "ants_itrf": ants_itrf,
        "uvw": jnp.array(xds.UVW.data.reshape(n_time, n_bl, 3).compute()),
        "vis_obs": read_data(data_col),
        "flags": read_data("FLAG"),
        "noise": jnp.array(xds.SIGMA.data.mean().compute()),
        "a1": jnp.array(xds.ANTENNA1.data.reshape(n_time, n_bl)[0, :].compute()),
        "a2": jnp.array(xds.ANTENNA2.data.reshape(n_time, n_bl)[0, :].compute()),
        "a1": jnp.array(xds.ANTENNA1.data[:n_bl].compute()),
        "a2": jnp.array(xds.ANTENNA2.data[:n_bl].compute()),
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
    state=None,
    constants=None,
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
        state=state,
        constants=constants,
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
            state=state,
            constants=constants,
            init_params=svi_results.params,
        )
        losses = jnp.concatenate([losses, svi_results.losses / obs_data.size])
        svi_results = SVIRunResult(svi_results.params, svi_results.state, losses)

    return svi_results, guide


@measure_runtime
def run_custom_svi(
    prob_model: Callable,
    obs_data: jax.Array,
    max_iter: int = 1_000,
    init_params: dict = None,
    epsilon: float = 1e-3,
    dual_run: bool = True,
    state: dict = None,
    constants: dict = None,
) -> SVIRunResult:
    """MAP optimization that avoids capturing large state arrays as JAX constants.

    Unlike NumPyro's SVI.run(), which binds model kwargs (including state/constants)
    into a lax.scan closure causing all arrays to be lowered as XLA constants,
    this passes state and constants as explicit traced arguments to _map_step so JAX
    sees them as dynamic input buffers.

    Optimizes log p(obs | params) + log p(params) directly, equivalent to
    AutoDelta SVI for MAP estimation.
    """
    def _run_phase(params, epsilon, max_iter):
        optimizer = optax.adabelief(epsilon)
        opt_state = optimizer.init(params)
        losses = []
        window = max(max_iter // 10, 1)
        init_loss = None
        pbar = trange(max_iter)
        for i in pbar:
            params, opt_state, loss = _map_step(
                prob_model, optimizer, params, opt_state, state, constants, obs_data
            )
            loss_val = float(loss)
            losses.append(loss_val)
            if init_loss is None:
                init_loss = loss_val
            start_idx = max(0, i + 1 - window)
            avg_loss = sum(losses[start_idx:]) / len(losses[start_idx:])
            n1 = start_idx + 1
            n2 = i + 1
            pbar.set_postfix_str(
                f"init loss: {init_loss:.4f}, avg. loss [{n1}-{n2}]: {avg_loss:.4f}",
                refresh=False,
            )
        return params, losses

    params = init_params
    params, losses = _run_phase(params, epsilon, max_iter)

    if dual_run:
        params, losses2 = _run_phase(params, epsilon / 10, max_iter)
        losses = losses + losses2

    # Add _auto_loc suffix to match AutoDelta convention expected by downstream code
    params_out = {k + "_auto_loc": v for k, v in params.items()}
    return SVIRunResult(params_out, None, jnp.array(losses))


@measure_runtime
def svi_predict(
    prob_model: Callable,
    guide: autoguide.AutoGuide,
    vi_params: dict,
    num_samples=100,
    key=random.PRNGKey(2),
    state=None,
    constants=None,
):
    predictive = Predictive(
        model=prob_model, guide=guide, params=vi_params, num_samples=num_samples
    )
    predictions = predictive(key, state=state, constants=constants)

    return predictions


@measure_runtime
def init_predict(
    tab_config, prob_model: Callable, subkey: jax.Array, init_params: dict, state=None, constants=None
):

    pred = Predictive(
        model=prob_model,
        posterior_samples=tree_map(lambda x: x[None, :], init_params),
        batch_ndims=1,
    )
    init_pred = pred(subkey, state=state, constants=constants)
    rchi2 = reduced_chi2(
        init_pred["vis_obs"][0], tab_config.vis_obs, tab_config.noise, tab_config.flags
    )
    print()
    print(f"Reduced Chi^2 @ init params : {rchi2}")

    return init_pred


@measure_runtime
def run_opt(
    tab_config,
    prob_model: Callable,
    subkeys: jax.Array,
    init_params: dict,
    ms_path,
    map_path,
    params_path,
    state=None,
    constants=None,
):

    start = datetime.now()
    print()
    print("Running Optimization ...")
    vi_results = run_custom_svi(
        prob_model=prob_model,
        obs_data=tab_config.vis_obs,
        max_iter=tab_config.args["opt"]["max_iter"],
        init_params=init_params,
        epsilon=tab_config.args["opt"]["epsilon"],
        dual_run=tab_config.args["opt"]["dual_run"],
        state=state,
        constants=constants,
    )
    vi_params = vi_results.params
    # Strip _auto_loc suffix to get raw param names for Predictive
    raw_params = {k.removesuffix("_auto_loc"): v for k, v in vi_params.items()}
    vi_pred = Predictive(
        model=prob_model,
        posterior_samples=tree_map(lambda x: x[None], raw_params),
        batch_ndims=1,
    )(subkeys[1], obs_data=tab_config.vis_obs, state=state, constants=constants)
    
    write_results_xds(vi_pred, tab_config, map_path)
    # write_params_xds(vi_params, gp_params, ms_params, params_path, overwrite=True)

    # if get_truth_conditional(config):

    #     # vi_pred keys are ['ast_vis', 'gains', 'rfi_vis', 'rmse_ast', 'rmse_gains', 'rmse_rfi', 'vis_obs']
    #     print(f"RMSE Gains      : {jnp.mean(vi_pred['rmse_gains']):.5f}")
    #     print(f"RMSE RFI Vis    : {jnp.mean(vi_pred['rmse_rfi']):.5f}")
    #     print(f"RMSE AST Vis    : {jnp.mean(vi_pred['rmse_ast']):.5f}")

    print()
    print(f"Optimization Run Time : {datetime.now() - start}")
    print(f"{datetime.now()}")
    start = datetime.now()

    rchi2 = reduced_chi2(
        vi_pred["vis_obs"][0], tab_config.vis_obs, tab_config.noise, tab_config.flags
    )
    print()
    print(f"Reduced Chi^2 @ opt params : {rchi2}")

    print()
    print(f"Copying tabascal results to MS file from {map_path}")
    write_results_ms(ms_path, map_path, tab_config.args["data"]["data_col"])

    return vi_pred, vi_results.losses, vi_params, rchi2


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
