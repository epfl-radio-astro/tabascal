import os
import sys


# ---------------------------------------------------------------------------
# 'run' subcommand
# ---------------------------------------------------------------------------

def _run_cmd(args):
    from contextlib import redirect_stdout
    from datetime import datetime
    import shutil
    import yaml

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    import jax
    from jax import random
    import jax.numpy as jnp
    import numpy as np

    from tabascal.timing import measure_runtime, print_timings, enable_timings
    from tabascal.tab_tools import init_predict, run_opt, nlog_like, nlog_post
    from tabascal.config import load_config, TabConfig, Model
    from tabascal.write import write_results_xds

    @measure_runtime
    def build_model(config, ms_path):
        tab_config = TabConfig(config, ms_path)
        model = Model(tab_config, config["model"]["components"])
        return tab_config, model

    @measure_runtime
    def evaluate_init(tab_config, model, key):
        key, subkey = random.split(key)
        init_pred = init_predict(tab_config, model.prob_model, subkey, model.init_params, state=model.state, constants=model.constants)
        nlog_l = nlog_like(model.prob_model, model.init_params, tab_config.vis_obs, state=model.state, constants=model.constants)
        nlog_p = nlog_post(model.prob_model, model.init_params, tab_config.vis_obs, state=model.state, constants=model.constants)
        init_state = model.forward(model.init_params, model.state, model.constants)
        return key, init_pred, nlog_l, nlog_p, init_state

    @measure_runtime
    def tabascal_subtraction(config, sim_dir, ms_path=None, norad_ids=[], suffix="", extra_tle_dir=None):
        if suffix:
            suffix = "_" + suffix

        run_id = datetime.now().strftime("%m-%d-%YT%H:%M:%S")
        log_path = f"log_tab_{run_id}.txt"

        class _Tee:
            def __init__(self, *writers):
                self._writers = writers
            def write(self, text):
                for w in self._writers: w.write(text)
            def flush(self):
                for w in self._writers: w.flush()

        model_name = "Custom"
        results_name = f"{model_name}{suffix}"

        if sim_dir:
            config["data"]["sim_dir"] = os.path.abspath(sim_dir)
        else:
            sim_dir = os.path.abspath(config["data"]["sim_dir"])
            config["data"]["sim_dir"] = sim_dir

        config["model"]["name"] = model_name

        if sim_dir[-1] == "/":
            sim_dir = sim_dir[:-1]
        f_name = os.path.split(sim_dir)[1]

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
        init_pred_path = os.path.join(results_dir, f"init_pred_{results_name}.zarr")

        if extra_tle_dir:
            config["satellites"]["extra_tle_dir"] = extra_tle_dir

        with open(log_path, "w") as log:
            with redirect_stdout(_Tee(sys.stdout, log)):

                print()
                start_time = datetime.now()
                print(f"Start Time : {start_time}")

                key, subkey = random.split(random.PRNGKey(1))

                print(f"Model : {model_name}")
                print()
                print(f_name)
                print()

                tab_config, model = build_model(config, ms_path)

                prob_model = model.prob_model

                shapes = {key: value.shape for key, value in model.init_params.items()}
                n_params = sum([x.size for x in model.init_params.values()])
                n_data = 2 * tab_config.vis_obs.size

                print(f"Using {tab_config.n_int_time} samples per time step for RFI prediction.")
                print()
                print(f"Number of Antennas   : {tab_config.n_ant: 4}")
                print(f"Number of Time Steps : {tab_config.n_time: 4}")
                print()
                print(f"Parameter shapes     : {shapes}")
                print(f"Number of parameters : {n_params}")
                print(f"Data shape           : {tab_config.vis_obs.shape}")
                print(f"Number of data points: {n_data}")

                print()
                end_start = datetime.now()
                print(f"Startup Time : {end_start - start_time}")
                print(f"{end_start}")

                key, init_pred, nlog_l, nlog_p, init_state = evaluate_init(tab_config, model, key)
                write_results_xds(init_pred, tab_config, init_pred_path)

                print(f"log_l : {nlog_l:.3e}")
                print(f"log_p : {nlog_p:.3e}")

                truth = {
                    "vis_rfi": jnp.nan * jnp.zeros((tab_config.n_bl, tab_config.n_freq, tab_config.n_time), dtype=complex),
                    "vis_ast": jnp.nan * jnp.zeros((tab_config.n_bl, tab_config.n_freq, tab_config.n_time), dtype=complex),
                    "gains": jnp.nan * jnp.ones((tab_config.n_ant, tab_config.n_freq, tab_config.n_time), dtype=complex),
                }

                if config["plots"]["init"]:
                    from tabascal.plot import plot_init
                    plot_init(tab_config, init_pred, truth, model_name, plot_dir)

                key, subkey = random.split(key)
                if config["plots"]["prior"]:
                    from tabascal.plot import plot_prior
                    plot_prior(tab_config, prob_model, truth, model_name, subkey, plot_dir, state=model.state, constants=model.constants)

                key, *subkeys = random.split(key, 3)
                if config["inference"]["opt"] and config["opt"]["max_iter"] > 0:
                    vi_pred, losses, vi_params, rchi2 = run_opt(
                        tab_config, prob_model, subkeys, model.init_params, ms_path, map_path, params_path,
                        state=model.state, constants=model.constants,
                    )

                    if config["plots"]["opt"]:
                        from tabascal.plot import plot_opt
                        plot_opt(tab_config, vi_pred, truth, model_name, plot_dir)

                    if config["plots"]["losses"]:
                        from tabascal.plot import plot_losses
                        plot_losses(losses, model_name, plot_dir)

                    opt_params = {key.removesuffix("_auto_loc"): value for key, value in vi_params.items()}

                    nlog_l = nlog_like(prob_model, opt_params, tab_config.vis_obs, state=model.state, constants=model.constants)
                    nlog_p = nlog_post(prob_model, opt_params, tab_config.vis_obs, state=model.state, constants=model.constants)

                    print(f"log_l : {nlog_l:.3e}")
                    print(f"log_p : {nlog_p:.3e}")
                else:
                    from tabascal.write import write_results_ms
                    print(f"Copying tabascal initial values to MS file from {init_pred_path}")
                    write_results_ms(ms_path, init_pred_path, tab_config.args["data"]["data_col"])

        shutil.copy(log_path, plot_dir)
        os.remove(log_path)

        with open(os.path.join(plot_dir, f"tab_config_{run_id}.yaml"), "w") as fp:
            yaml.dump(config, fp)

    if args.timings:
        enable_timings()

    config = load_config(args.config)
    norad_ids = []

    if config.get("model", {}).get("precision", "single") == "double":
        jax.config.update("jax_enable_x64", True)

    from tabascal.tle import TLEError
    try:
        tabascal_subtraction(
            config,
            args.sim_dir,
            args.ms_path,
            norad_ids,
            args.suffix,
            extra_tle_dir=args.extra_tle_dir,
        )
    except TLEError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

    if args.timings:
        print_timings()


# ---------------------------------------------------------------------------
# 'spacetrack-login' subcommand
# ---------------------------------------------------------------------------

def _spacetrack_login_cmd(args):
    import getpass
    from tabascal.tle import save_spacetrack_credentials, spacetrack_config_path

    username = args.username or input("Space-Track username (email): ")
    password = getpass.getpass("Space-Track password: ")
    path = save_spacetrack_credentials(username, password)
    print(f"Credentials saved to: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="tabascal CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- run --
    run_parser = subparsers.add_parser("run", help="Apply tabascal to a simulation.")
    run_parser.add_argument("-c", "--config", required=True, help="Path to the config file.")
    run_parser.add_argument("-s", "--sim_dir", help="Path to the directory of the simulation.")
    run_parser.add_argument("-ms", "--ms_path", help="Path to Measurement Set.")
    run_parser.add_argument("-np", "--norad_path", help="Path to text file containing NORAD IDs to include.")
    run_parser.add_argument("-sx", "--suffix", default="", help="Image name suffix.")
    run_parser.add_argument("-t", "--timings", action="store_true", help="Enable timing measurements.")
    run_parser.add_argument(
        "--extra-tle-dir",
        dest="extra_tle_dir",
        default=None,
        metavar="DIR",
        help="Extra directory searched for cached TLEs before the managed cache and Space-Track.",
    )

    # -- spacetrack-login --
    login_parser = subparsers.add_parser(
        "spacetrack-login",
        help="Save Space-Track credentials to the user config file.",
    )
    login_parser.add_argument(
        "-u", "--username",
        default=None,
        help="Space-Track username (email address). Prompted interactively if not given.",
    )

    args = parser.parse_args()

    if args.command == "run":
        _run_cmd(args)
    elif args.command == "spacetrack-login":
        _spacetrack_login_cmd(args)


if __name__ == "__main__":
    main()
