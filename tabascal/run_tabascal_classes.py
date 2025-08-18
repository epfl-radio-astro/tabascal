from datetime import datetime

import shutil
import os
import sys
import yaml

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = (
    "false"  # Disable GPU Memory Preallocation
)
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = (
#     "platform"  # Enable GPU Memory allocation and deallocation on-the-fly
# )
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90" # GPU Memory Preallocation Factor

import jax
from jax import random
import jax.profiler
import jax.numpy as jnp

# jax.config.update("jax_platform_name", "cpu")

import numpy as np

from tabsim.config import Tee, load_config

# from tabascal.gp import (
#     kernel,
#     resampling_kernel,
#     cholesky,
#     find_closest_factor_greater_than,
# )
# from tabascal.models import (
#     fixed_orbit_rfi_full_fft_standard_padded_model,
#     fixed_orbit_rfi_fft_standard,
#     fixed_orbit_rfi_full_fft_standard_model,
#     fixed_orbit_rfi_full_fft_standard_model_otf,
#     fixed_orbit_rfi_full_fft_standard_model_otf_fft,
# )

# from tabascal.tab_tools import (
#     split_args,
#     read_ms,
#     get_ast_fringe_rate,
#     get_tles,
#     estimate_sampling,
#     get_rfi_phase,
#     get_gp_params,
#     calculate_estimates,
#     get_truth_conditional,
#     calculate_true_values,
#     get_prior_means,
#     pow_spec_sqrt,
#     pow_spec,
#     save_memory,
#     inv_transform,
#     get_init_params,
#     init_predict,
#     plot_init,
#     plot_truth,
#     plot_prior,
#     run_mcmc,
#     run_opt,
#     run_fisher,
#     write_results_xds,
#     write_params_xds,
#     check_antenna_and_satellite_positions,
# )

from tabascal.tab_tools_new import save_memory

from tabsim.tle import id_generator

from typing import Optional


from tabascal.config import TabConfig, Model


def tabascal_subtraction(
    config: dict,
    sim_dir: str,
    ms_path: Optional[str] = None,
    spacetrack_path: Optional[str] = None,
    norad_ids: list = [],
    suffix: str = "",
):

    if suffix:
        suffix = "_" + suffix

    run_id = id_generator()

    log_path = f"log_tab_{run_id}.txt"
    log = open(log_path, "w")
    backup = sys.stdout
    sys.stdout = Tee(sys.stdout, log)

    print()
    start_time = datetime.now()
    print(f"Start Time : {start_time}")

    key, subkey = random.split(random.PRNGKey(1))

    mem_i = 0

    model_name = "Custom"
    print(f"Model : {model_name}")
    results_name = f"{model_name}{suffix}"

    if config["data"]["sim_dir"] is None:
        config["data"]["sim_dir"] = os.path.abspath(sim_dir)
    else:
        sim_dir = os.path.abspath(config["data"]["sim_dir"])
        config["data"]["sim_dir"] = sim_dir

    config["model"]["name"] = model_name

    if sim_dir[-1] == "/":
        sim_dir = sim_dir[:-1]
    f_name = os.path.split(sim_dir)[1]

    print()
    print(f_name)
    print()

    zarr_path = os.path.join(sim_dir, f"{f_name}.zarr")
    config["data"]["zarr_path"] = zarr_path

    if not ms_path:
        ms_path = os.path.join(sim_dir, f"{f_name}.ms")
    else:
        ms_path = os.path.abspath(ms_path)

    config["data"]["ms_path"] = ms_path

    plot_dir = os.path.join(sim_dir, f"plots/{suffix[1:]}")
    results_dir = os.path.join(sim_dir, "results")
    mem_dir = os.path.join(sim_dir, "memory_profiles")

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(mem_dir, exist_ok=True)

    map_path = os.path.join(results_dir, f"map_pred_{results_name}.zarr")
    params_path = os.path.join(results_dir, f"map_params_{results_name}.zarr")
    fisher_path = os.path.join(results_dir, f"fisher_pred_{results_name}.zarr")
    mcmc_path = os.path.join(results_dir, f"mcmc_pred_{results_name}.zarr")
    init_pred_path = os.path.join(results_dir, f"init_pred_{results_name}.zarr")
    true_pred_path = os.path.join(results_dir, f"true_pred_{results_name}.zarr")

    init_params_path = os.path.join(results_dir, f"init_params_{results_name}.zarr")
    true_params_path = os.path.join(results_dir, f"true_params_{results_name}.zarr")

    tab_config = TabConfig(config, ms_path)

    model = Model(tab_config, config["model"]["components"])  # type: ignore

    prob_model = model.prob_model

    shapes = {key: value.shape for key, value in model.init_params.items()}
    n_params = sum([x.size for x in model.init_params.values()])
    n_data = 2 * tab_config.vis_obs.size

    print(f"Using {tab_config.n_int_samples} samples per time step for RFI prediction.")
    print()
    print(f"Number of Antennas   : {tab_config.n_ant: 4}")
    print(f"Number of Time Steps : {tab_config.n_time: 4}")
    print()
    print(f"Parameter shapes     : {shapes}")
    print(f"Number of parameters : {n_params}")
    print(f"Number of data points: {n_data}")

    print()
    end_start = datetime.now()
    print(f"Startup Time : {end_start - start_time}")
    print(f"{end_start}")

    mem_i = save_memory(mem_dir, mem_i)

    from tabascal.tab_tools_new import (
        write_results_xds,
        init_predict,
        plot_init,
        plot_prior,
        run_opt,
    )

    key, subkey = random.split(key)
    init_pred = init_predict(tab_config, prob_model, subkey, model.init_params)
    write_results_xds(init_pred, tab_config, init_pred_path)
    # write_params_xds(
    #     {key + "_auto_loc": value for key, value in init_params_base.items()},
    #     gp_params,
    #     ms_params,
    #     init_params_path,
    # )

    truth = {
        "vis_rfi": jnp.zeros((tab_config.n_bl, tab_config.n_time), dtype=complex),
        "vis_ast": jnp.zeros((tab_config.n_bl, tab_config.n_time), dtype=complex),
        "gains": jnp.ones((tab_config.n_ant, tab_config.n_time), dtype=complex),
    }

    if config["plots"]["init"]:
        plot_init(tab_config, init_pred, truth, model_name, plot_dir)

    ### Check and Plot Model at true parameters
    # if config["plots"]["truth"]:
    #     key, subkey = random.split(key)
    #     plot_truth(
    #         zarr_path,
    #         ms_params,
    #         static_args,
    #         array_args,
    #         model,
    #         model_name,
    #         subkey,
    #         true_params,
    #         gp_params,
    #         inv_scaling,
    #         plot_dir,
    #         true_pred_path,
    #     )

    mem_i = save_memory(mem_dir, mem_i)

    ### Check and Plot Model at prior parameters
    key, subkey = random.split(key)
    if config["plots"]["prior"]:
        plot_prior(
            tab_config,
            prob_model,
            truth,
            model_name,
            subkey,
            plot_dir,
        )

    # ### Run MCMC Inference
    # key, *subkeys = random.split(key, 3)
    # if config["inference"]["mcmc"]:
    #     mcmc = run_mcmc(
    #         ms_params,
    #         model,
    #         model_name,
    #         subkeys,
    #         static_args,
    #         array_args,
    #         init_params_base,
    #         plot_dir,
    #         mcmc_path,
    #         num_warmup=config["mcmc"]["n_warmup"],
    #         num_samples=config["mcmc"]["n_samples"],
    #         max_tree_depth=config["mcmc"]["max_tree_depth"],
    #         thin_factor=config["mcmc"]["thin_factor"],
    #     )

    # mem_i = save_memory(mem_dir, mem_i)

    ### Run Optimization
    key, *subkeys = random.split(key, 3)
    if config["inference"]["opt"]:
        vi_params, rchi2 = run_opt(
            tab_config,
            prob_model,
            truth,
            model_name,
            subkeys,
            model.init_params,
            plot_dir,
            ms_path,
            map_path,
            params_path,
        )

    mem_i = save_memory(mem_dir, mem_i)

    max_fisher_time = 30 * 60  # seconds

    # ### Run Fisher Covariance Prediction
    # key, *subkeys = random.split(key, 3)
    # if config["inference"]["fisher"] and rchi2 < 1.1 and n_int_samples < 30:
    #     from tabsim.tools import time_limit, TimeoutException

    #     try:
    #         with time_limit(max_fisher_time):
    #             run_fisher(
    #                 config,
    #                 gp_params,
    #                 ms_params,
    #                 model,
    #                 model_name,
    #                 subkeys,
    #                 vis_model,
    #                 static_args,
    #                 array_args,
    #                 vi_params,
    #                 init_params_base,
    #                 plot_dir,
    #                 fisher_path,
    #             )
    #     except TimeoutException as e:
    #         print("Timed out!")

    # print()
    # end_time = datetime.now()
    # print(f"End Time  : {end_time}")
    # print(f"Total Time : {end_time - start_time}")

    # mem_i = save_memory(mem_dir, mem_i)

    log.close()
    shutil.copy(log_path, plot_dir)
    os.remove(log_path)
    sys.stdout = backup

    with open(os.path.join(plot_dir, f"tab_config_{run_id}.yaml"), "w") as fp:
        yaml.dump(config, fp)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Apply tabascal to a simulation.")
    parser.add_argument(
        "-c", "--config", required=True, help="Path to the config file."
    )
    parser.add_argument(
        "-s", "--sim_dir", help="Path to the directory of the simulation."
    )
    parser.add_argument("-ms", "--ms_path", help="Path to Measurement Set.")
    parser.add_argument(
        "-np", "--norad_path", help="Path to text file containing NORAD IDs to include."
    )
    parser.add_argument(
        "-st", "--spacetrack", help="Path to Space-Track login details."
    )
    parser.add_argument("-sx", "--suffix", default="", help="Image name suffix.")
    args = parser.parse_args()
    sim_dir = args.sim_dir
    conf_path = args.config
    spacetrack_path = args.spacetrack
    norad_path = args.norad_path
    if sim_dir:
        norad_path = os.path.join(sim_dir, "input_data/norad_ids.yaml")

    if norad_path:
        norad_ids = [int(x) for x in np.atleast_1d(np.loadtxt(norad_path))]
    else:
        norad_ids = []

    config = load_config(conf_path, config_type="tab")

    config_st_path = config["satellites"]["spacetrack_path"]
    if spacetrack_path:
        config["satellites"]["spacetrack_path"] = os.path.abspath(spacetrack_path)
    elif config_st_path:
        config_st_path = os.path.abspath(config_st_path)
        config["satellites"]["spacetrack_path"] = config_st_path
        spacetrack_path = config_st_path

    tabascal_subtraction(
        config, sim_dir, args.ms_path, spacetrack_path, norad_ids, args.suffix
    )


if __name__ == "__main__":
    main()
