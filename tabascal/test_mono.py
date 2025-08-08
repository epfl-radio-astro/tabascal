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


from tabascal.dist import standard_normal
from jax import vmap
from tabascal.interferometry import calculate_rfi_vis_fine

import numpyro
import numpyro.distributions as dist


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
    fixed_orbit = FixedOrbit()
    fixed_orbit.setup(tab_config)
    rfi_phase = fixed_orbit.rfi_phase

    rfi_signal = ComplexRFI()
    rfi_signal.setup(tab_config)
    n_rfi = rfi_signal.n_rfi
    n_ant = rfi_signal.n_ant
    n_rfi_times = rfi_signal.n_rfi_times
    L_rfi_A = rfi_signal.L_rfi_A
    mu_rfi_A = rfi_signal.mu_rfi_A
    resample_rfi = rfi_signal.resample_rfi
    rfi_transform = rfi_signal.forward_transform

    rfi_vis = RiemannVisCalculation()
    rfi_vis.setup(tab_config)
    a1 = rfi_vis.a1
    a2 = rfi_vis.a2
    n_int = rfi_vis.n_int
    n_time = rfi_vis.n_time
    n_bl = rfi_vis.n_bl

    ast_vis = FourierTimeAst()
    ast_vis.setup(tab_config)
    n_ast_k = ast_vis.n_ast_k
    sigma_ast_k = ast_vis.sigma_ast_k
    mu_ast_k = ast_vis.mu_ast_k
    n_pad = ast_vis.n_pad
    ast_transform = ast_vis.forward_transform

    gains = UnitaryGains()
    gains.setup(tab_config)

    noise = tab_config.noise

    from tabascal.transform import affine_transform_full, affine_transform_diag
    from tabascal.vis import (
        get_ast_vis_fft_padded,
        get_rfi_vis_full_no_extra,
        averaging1,
    )

    array_args = {
        "rfi_phase": rfi_phase,
        "a1": a1,
        "a2": a2,
        "L_RFI": L_rfi_A,
        "mu_rfi_r": mu_rfi_A.real,
        "mu_rfi_i": mu_rfi_A.imag,
        "resample_rfi": resample_rfi,
        "sigma_ast_k": sigma_ast_k,
        "mu_ast_k_r": mu_ast_k.real,
        "mu_ast_k_i": mu_ast_k.imag,
    }

    from frozendict import frozendict
    from functools import partial

    args = frozendict({"n_int_samples": n_int})

    # @partial(jit, static_argnums=(1,))
    @jit
    def get_rfi_vis_full(rfi_amp):  # , args, array_args):
        # a1, a2 = array_args["a1"], array_args["a2"]
        # rfi_phase = array_args["rfi_phase"]
        # rfi_amp has shape (n_rfi, n_ant, n_time)
        rfi_amp_fine = vmap(lambda x, y: x @ y.T, in_axes=(0, None))(
            rfi_amp, resample_rfi
        )
        # rfi_amp_fine has shape (n_rfi, n_ant, n_time_fine)
        rfi_vis = jnp.sum(
            rfi_amp_fine[:, a1]
            * jnp.conjugate(rfi_amp_fine[:, a2])
            * jnp.exp(1.0j * (rfi_phase[:, a1] - rfi_phase[:, a2])),
            axis=0,
        )
        # rfi_vis has shape (n_bl, n_time_fine)
        rfi_vis = averaging1(rfi_vis, n_int)
        # rfi_vis = vmap(averaging, in_axes=(0, None))(rfi_vis, args["n_int_samples"])
        # rfi_vis has shape (n_bl, n_time)
        return rfi_vis

    @partial(jit, static_argnums=(2,))
    def get_ast_vis_fft_padded(ast_k_r, ast_k_i, ast_pad):
        ast_k_padded = jnp.fft.ifft(ast_k_r + 1.0j * ast_k_i, axis=1)
        return ast_k_padded[:, ast_pad:-ast_pad]

    @jit
    def forward_old(params):

        rfi_r = vmap(vmap(affine_transform_full, (0, None, 0), 0), (1, None, 1), 1)(
            params["rfi_r_induce_base"], array_args["L_RFI"], array_args["mu_rfi_r"]
        )
        rfi_i = vmap(vmap(affine_transform_full, (0, None, 0), 0), (1, None, 1), 1)(
            params["rfi_i_induce_base"], array_args["L_RFI"], array_args["mu_rfi_i"]
        )
        ast_k_r = vmap(affine_transform_diag, in_axes=(0, 0, 0))(
            params["ast_k_r_base"], array_args["sigma_ast_k"], array_args["mu_ast_k_r"]
        )
        ast_k_i = vmap(affine_transform_diag, in_axes=(0, 0, 0))(
            params["ast_k_i_base"], array_args["sigma_ast_k"], array_args["mu_ast_k_i"]
        )
        rfi_A = rfi_r + 1.0 * rfi_i
        # vis_rfi = get_rfi_vis_full_no_extra(
        #     rfi_A,
        #     args,
        #     array_args,
        # )
        vis_rfi = get_rfi_vis_full(rfi_A)
        vis_ast = get_ast_vis_fft_padded(ast_k_r, ast_k_i, n_pad)

        vis_obs = vis_ast + vis_rfi

        return vis_obs

    @jit
    def transform_rfi(rfi_A_induce_base, L_rfi_A, mu_rfi_A):

        rfi_A_induce = vmap(
            vmap(affine_transform_full, (0, None, 0), 0), (1, None, 1), 1
        )(rfi_A_induce_base, L_rfi_A, mu_rfi_A)

        # rfi_A = vmap(vmap(jnp.dot, (None, 0), 0), (None, 1), 1)(
        #     resample_rfi, rfi_A_induce
        # )

        return rfi_A_induce

    @jit
    def forward(rfi_A_induce_base, ast_k_base):

        rfi_A_induce = transform_rfi(
            rfi_A_induce_base.real, L_rfi_A, mu_rfi_A.real
        ) + 1.0j * transform_rfi(rfi_A_induce_base.imag, L_rfi_A, mu_rfi_A.imag)

        # rfi_A = vmap(vmap(jnp.dot, (None, 0), 0), (None, 1), 1)(
        #     resample_rfi, rfi_A_induce
        # )

        # rfi_A = transform_rfi(rfi_A_induce_base.real) + 1.0j * transform_rfi(
        #     rfi_A_induce_base.imag
        # )

        # vis_rfi_fine = calculate_rfi_vis_fine(rfi_A, rfi_phase, a1, a2)

        # # new_shape = (n_bl, n_freq, n_time, n_int)
        # new_shape = (n_bl, n_time, n_int)

        # # vis_rfi_fine is shape (n_bl, n_time_fine)
        # # vis_rfi is shape (n_bl, n_time)
        # vis_rfi = jnp.mean(jnp.reshape(vis_rfi_fine, new_shape), axis=-1)

        vis_rfi = get_rfi_vis_full(rfi_A_induce)

        # ast_k = ast_k_base * sigma_ast_k + mu_ast_k
        ast_k_r = vmap(affine_transform_diag, in_axes=(0, 0, 0))(
            ast_k_base.real, sigma_ast_k, mu_ast_k.real
        )
        ast_k_i = vmap(affine_transform_diag, in_axes=(0, 0, 0))(
            ast_k_base.imag, sigma_ast_k, mu_ast_k.imag
        )
        ast_k = ast_k_r + 1.0j * ast_k_i

        # vis_ast = jnp.fft.ifft(ast_k, axis=1)[:, :, n_pad:-n_pad]
        # vis_ast = jnp.fft.ifft(ast_k, axis=1)[:, n_pad:-n_pad]

        vis_ast = get_ast_vis_fft_padded(ast_k.real, ast_k.imag, n_pad)

        vis_obs = vis_ast + vis_rfi

        return vis_obs

    def prob_model(obs_data=None):

        rfi_r_induce_base = standard_normal(
            "rfi_r_induce_base", (n_rfi, n_ant, n_rfi_times)
        )
        rfi_i_induce_base = standard_normal(
            "rfi_i_induce_base", (n_rfi, n_ant, n_rfi_times)
        )

        ast_k_r_base = standard_normal("ast_k_r_base", (n_bl, n_ast_k))
        ast_k_i_base = standard_normal("ast_k_i_base", (n_bl, n_ast_k))

        # rfi_A_induce_base = rfi_r_induce_base + 1.0j * rfi_i_induce_base
        # ast_k_base = ast_k_r_base + 1.0j * ast_k_i_base
        # vis_obs = forward(rfi_A_induce_base, ast_k_base)

        params = {
            "rfi_r_induce_base": rfi_r_induce_base,
            "rfi_i_induce_base": rfi_i_induce_base,
            "ast_k_r_base": ast_k_r_base,
            "ast_k_i_base": ast_k_i_base,
        }
        vis_obs = forward_old(params)

        numpyro.deterministic("vis_obs", vis_obs)

        if obs_data is not None:
            vis_obs_ri = jnp.stack([vis_obs.real, vis_obs.imag], axis=0)
            obs_data_ri = jnp.stack([obs_data.real, obs_data.imag], axis=0)
            numpyro.sample(
                "vis_obs_ri",
                dist.Normal(vis_obs_ri, noise),  # type: ignore
                obs=obs_data_ri,
            )

    model = Model(tab_config, components)

    # forward = jit(model.build_forward())

    # state = {**model.init_params, **model.state_params}

    # from tabascal.config import TabascalState

    # initial_state = TabascalState.create_initial(
    #     tab_config.n_rfi,
    #     tab_config.n_bl,
    #     tab_config.n_time,
    #     tab_config.n_time_fine,
    #     tab_config.n_ant,
    # )

    # state = forward(model.init_params, initial_state)

    # print(state.vis_obs.shape)
    # print(state.vis_obs.dtype)

    # print(jnp.sum(state.vis_obs))

    # prob_model = model.build_prob_model()

    # shapes = {key: value.shape for key, value in model.init_params.items()}
    # n_params = sum([x.size for x in model.init_params.values()])
    # n_data = 2 * tab_config.vis_obs.size

    # print(f"Parameter shapes     : {shapes}")
    # print(f"Number of parameters : {n_params}")
    # print(f"Number of data points: {n_data}")

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
