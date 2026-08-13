import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys
import shutil
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from datetime import datetime

import yaml

import jax
from jax import random

from tabascal.timing import measure_runtime, print_timings, enable_timings
from tabascal.tab_tools import init_predict, run_opt, nlog_like, nlog_post
from tabascal.config import load_config, TabConfig, Model
from tabascal.distributed import (
    barrier,
    is_process_0,
    make_global,
    replicated_sharding,
    shard_pytree,
    sharding_enabled,
    suppress_worker_stdout,
)
from tabascal.imports import import_components
from tabascal.write import write_results_xds
from tabascal.orbit import TLEError, save_orbits_for_reuse
from tabascal.truth import require_truth, load_truth, has_truth, TruthError

import jax


class _Tee:
    def __init__(self, *writers):
        self._writers = writers

    def write(self, text):
        for w in self._writers:
            w.write(text)

    def flush(self):
        for w in self._writers:
            w.flush()


@measure_runtime
def assert_precision_supported(config):
    """Fail fast if a requested component requires double precision under single.

    Resolves the component list and checks each class's ``requires_double`` flag
    up front, so an incompatible config errors *before* the expensive
    ``TabConfig`` setup (MS read, TLE fetch) and names every offending component
    at once, rather than tripping a single component's own gate deep in the run.
    """
    model_cfg = config.get("model", {})
    if model_cfg.get("precision", "single") == "double":
        return
    classes = import_components(model_cfg.get("components", []) or [])
    offenders = sorted(
        cls.__name__ for cls in classes if getattr(cls, "requires_double", False)
    )
    if offenders:
        raise ValueError(
            "model.precision is 'single', but these components require double "
            f"precision: {', '.join(offenders)}. "
            "Set model.precision to 'double' to use them."
        )


def _print_table(headers, rows):
    """Print ``rows`` under ``headers`` as a left-aligned, column-padded table."""
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def _fmt(cells):
        # rstrip so the last column contributes no trailing whitespace.
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

    # Columns are joined by a 2-space gap, so the rule spans the padded widths plus
    # those gaps -- not sum(w + 1), which falls short by one char per extra column.
    print("=" * (sum(widths) + 2 * (len(widths) - 1)))
    print(_fmt(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(_fmt(row))


def print_devices():
    """Print a table of every *global* JAX device the run is spread over.

    Global rather than local: under multi-process only process 0 prints, and the
    model is sharded over the full mesh (:func:`rfi_mesh` spans ``jax.devices()``),
    so a local listing would understate what the job is actually using. The process
    column is what distinguishes the one-GPU-per-process layout from a
    single-process multi-GPU one at a glance.
    """
    rows = [
        (str(d), d.device_kind, str(d.process_index))
        for d in jax.devices()
    ]
    _print_table(("Device", "Kind", "Process"), rows)


def print_memory_usage():
    """Print a table of Peak memory usage across all local JAX devices.

    Peak usage is reported per device in GB. Devices whose backend exposes no
    stats (e.g. CPU) are still listed with ``n/a`` so the table accounts for
    every device.
    """
    devices = jax.local_devices()

    rows = []
    for d in devices:
        stats = d.memory_stats() or {}
        peak = stats.get("peak_bytes_in_use")
        peak_gb = f"{peak / 1e9:.3f}" if peak is not None else "n/a"
        limit = stats.get("bytes_limit")
        limit_gb = f"{limit / 1e9:.3f}" if limit is not None else "n/a"
        rows.append((str(d), peak_gb, limit_gb))

    print()
    print("Memory usage:")
    _print_table(("Device", "Peak (GB)", "Limit (GB)"), rows)


def build_model(config, ms_path):
    assert_precision_supported(config)
    require_truth(config)
    tab_config = TabConfig(config, ms_path)
    model = Model(tab_config, config["model"]["components"])
    return tab_config, model


@measure_runtime
def evaluate_init(tab_config, model, key, truth=None):
    key, subkey = random.split(key)
    init_pred = init_predict(tab_config, model.prob_model, subkey, model.init_params, state=model.state, constants=model.constants, truth=truth)
    nlog_l = nlog_like(model.prob_model, model.init_params, tab_config.vis_obs, state=model.state, constants=model.constants)
    nlog_p = nlog_post(model.prob_model, model.init_params, tab_config.vis_obs, state=model.state, constants=model.constants)
    return key, init_pred, nlog_l, nlog_p


@contextmanager
def _stdout_logger(log_path, enabled):
    if not enabled:
        yield
        return
    with open(log_path, "w") as log:
        with redirect_stdout(_Tee(sys.stdout, log)):
            yield


@dataclass
class _RunPaths:
    """Resolved output layout for a single tabascal run."""
    run_id: str
    log_path: str
    model_name: str
    f_name: str
    ms_path: str
    plot_dir: str
    map_path: str
    params_path: str
    init_pred_path: str
    used_orbits_path: str


def _resolve_paths(config, sim_dir, ms_path, suffix, extra_orbit_dir, norad_path=None):
    """Resolve the run's directory layout and write derived paths into ``config``.

    Creates the plot/results/memory directories and records the sim, zarr and MS
    paths back onto ``config``. Returns a :class:`_RunPaths` bundling everything
    the run needs so the orchestration below stays free of path arithmetic.
    """
    if suffix:
        suffix = "_" + suffix

    run_id = datetime.now().strftime("%m-%d-%YT%H:%M:%S")

    model_name = "Custom"
    results_name = f"{model_name}{suffix}"

    if sim_dir:
        sim_dir = os.path.abspath(sim_dir)
    else:
        sim_dir = os.path.abspath(config["data"]["sim_dir"])
    config["data"]["sim_dir"] = sim_dir
    config["model"]["name"] = model_name

    f_name = os.path.split(sim_dir)[1]
    config["data"]["zarr_path"] = os.path.join(sim_dir, f"{f_name}.zarr")

    if ms_path:
        ms_path = os.path.abspath(ms_path)
    else:
        ms_path = os.path.join(sim_dir, f"{f_name}.ms")
    config["data"]["ms_path"] = ms_path

    plot_dir = os.path.join(sim_dir, f"plots/{suffix[1:]}")
    results_dir = os.path.join(sim_dir, "results")
    for directory in (plot_dir, results_dir):
        os.makedirs(directory, exist_ok=True)

    if extra_orbit_dir:
        config["satellites"]["extra_orbit_dir"] = extra_orbit_dir
    if norad_path:
        # The CLI flag wins over both config keys; TabConfig's normalizer reads
        # norad_ids_path in preference to norad_ids.
        config["satellites"]["norad_ids_path"] = norad_path

    return _RunPaths(
        run_id=run_id,
        log_path=f"log_tab_{run_id}.txt",
        model_name=model_name,
        f_name=f_name,
        ms_path=ms_path,
        plot_dir=plot_dir,
        map_path=os.path.join(results_dir, f"map_pred_{results_name}.zarr"),
        params_path=os.path.join(results_dir, f"map_params_{results_name}.zarr"),
        init_pred_path=os.path.join(results_dir, f"init_pred_{results_name}.zarr"),
        used_orbits_path=os.path.join(results_dir, f"used_orbits_{results_name}.json"),
    )


def _print_run_header(model_name, f_name, start_time):
    print()
    print(f"Start Time : {start_time}")
    print(f"Model : {model_name}")
    print()
    print(f_name)
    print()


def _print_model_summary(tab_config, model, start_time):
    shapes = {key: value.shape for key, value in model.init_params.items()}
    n_params = sum(x.size for x in model.init_params.values())
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


@measure_runtime
def tabascal_subtraction(
    config, sim_dir, ms_path=None, suffix="", extra_orbit_dir=None, norad_path=None, log=True
):
    paths = _resolve_paths(config, sim_dir, ms_path, suffix, extra_orbit_dir, norad_path)
    ms_path = paths.ms_path

    with _stdout_logger(paths.log_path, log):
        start_time = datetime.now()
        key, _ = random.split(random.PRNGKey(1))

        _print_run_header(paths.model_name, paths.f_name, start_time)

        if sharding_enabled():
            print(f"Sharding RFI sources over {jax.device_count()} devices:")
        else:
            print("Running on single device:")

        print_devices()
        print()

        tab_config, model = build_model(config, ms_path)
        prob_model = model.prob_model

        # Persist the real TLEs this run resolved so it can be reproduced later.
        # Only process 0 writes the shared result path in distributed runs.
        if is_process_0():
            n_rfi_real = getattr(tab_config, "n_rfi_real", None)
            used_ids = getattr(tab_config, "norad_ids", None)
            used_records = getattr(tab_config, "orbit_records", None)
            if n_rfi_real is not None:
                used_ids = used_ids[:n_rfi_real] if used_ids is not None else None
                used_records = (
                    used_records[:n_rfi_real] if used_records is not None else None
                )
            saved_orbits = save_orbits_for_reuse(
                paths.used_orbits_path,
                used_ids,
                used_records,
            )
            if saved_orbits:
                print(f"Orbits used saved to : {saved_orbits}")
                print(
                    "  (reuse via --extra-orbit-dir to reproduce this run's "
                    "trajectory priors)"
                )

        if sharding_enabled():
            # Split every leading-RFI-axis array across the device mesh and replicate
            # the rest. _map_step takes all of these as traced jit arguments, so GSPMD
            # propagates the shardings through the whole optimization (gradients and
            # optimizer state included) from here on. (vis_obs/flags/noise were already
            # globalized in TabConfig, before Model captured them in closures.)
            model.init_params = shard_pytree(model.init_params, tab_config.n_rfi)
            model.state = shard_pytree(model.state, tab_config.n_rfi)
            model.constants = shard_pytree(model.constants, tab_config.n_rfi)

        _print_model_summary(tab_config, model, start_time)

        # Load the tab-sim ground truth once and share it across the init/opt RMSE
        # reporting and all plotting. Missing quantities come back as NaN so consumers
        # can index unconditionally; reporting skips whatever is unavailable.
        truth = load_truth(tab_config)
        if sharding_enabled():
            # Truth arrays are compared eagerly against globally-sharded predictions;
            # process-local device arrays cannot mix with global ones in multi-process.
            truth = {k: make_global(v, replicated_sharding()) for k, v in truth.items()}
        if not has_truth(truth):
            print("\nNo tab-sim truth available - skipping truth-based RMSE reporting.")

        key, init_pred, nlog_l, nlog_p = evaluate_init(tab_config, model, key, truth=truth)
        write_results_xds(init_pred, tab_config, paths.init_pred_path)

        print(f"log_l : {nlog_l:.3e}")
        print(f"log_p : {nlog_p:.3e}")

        # Plotting from precomputed (replicated) predictions is safe on process 0
        # alone. plot_prior is different: it draws fresh model samples, and under
        # multi-process any model evaluation is a collective -- running it on one
        # rank would deadlock -- so it is skipped there rather than gated.
        if config["plots"]["init"] and is_process_0():
            from tabascal.plot import plot_init
            plot_init(tab_config, init_pred, truth, paths.model_name, paths.plot_dir)

        key, subkey = random.split(key)
        if config["plots"]["prior"]:
            if jax.process_count() > 1:
                print("Skipping prior plots: not supported in multi-process runs.")
            else:
                from tabascal.plot import plot_prior
                plot_prior(tab_config, prob_model, truth, paths.model_name, subkey, paths.plot_dir, state=model.state, constants=model.constants)

        key, *subkeys = random.split(key, 3)
        if config["inference"]["opt"] and config["opt"]["max_iter"] > 0:
            vi_pred, losses, vi_params, _ = run_opt(
                tab_config, prob_model, subkeys, model.init_params, ms_path, paths.map_path, paths.params_path,
                state=model.state, constants=model.constants, truth=truth,
            )

            if config["plots"]["opt"] and is_process_0():
                from tabascal.plot import plot_opt
                plot_opt(tab_config, vi_pred, truth, paths.model_name, paths.plot_dir)

            if config["plots"]["losses"] and is_process_0():
                from tabascal.plot import plot_losses
                plot_losses(losses, paths.model_name, paths.plot_dir)

            opt_params = {key.removesuffix("_auto_loc"): value for key, value in vi_params.items()}

            nlog_l = nlog_like(prob_model, opt_params, tab_config.vis_obs, state=model.state, constants=model.constants)
            nlog_p = nlog_post(prob_model, opt_params, tab_config.vis_obs, state=model.state, constants=model.constants)

            print(f"log_l : {nlog_l:.3e}")
            print(f"log_p : {nlog_p:.3e}")
        else:
            from tabascal.write import write_results_ms
            print(f"Copying tabascal initial values to MS file from {paths.init_pred_path}")
            write_results_ms(ms_path, paths.init_pred_path, tab_config.args["data"]["data_col"])

    if log:
        shutil.copy(paths.log_path, paths.plot_dir)
        os.remove(paths.log_path)

    if is_process_0():
        with open(os.path.join(paths.plot_dir, f"tab_config_{paths.run_id}.yaml"), "w") as fp:
            yaml.dump(config, fp)

    # Workers must not tear down the distributed runtime while process 0 is still
    # writing results / running its final collectives.
    barrier("tabascal-run-end")


def set_precision(config):
    """Enable/disable JAX float64 based on ``config["model"]["precision"]``.

    sgp4jax (and potentially other jax-based libraries) call
    jax.config.update("jax_enable_x64", True) at import time, so this toggle must
    run *after* those imports to take effect — single precision has to actively
    disable x64 that they turned on. The x64-setting libraries are imported
    explicitly here so the ordering is preserved even if the module-level imports
    are refactored away. WARNING: any future jax-based dependency that flips
    jax_enable_x64 on import must be imported here, before this update, or single
    precision will silently run as double.

    Also force full-fp32 matmuls. On Ampere+ GPUs JAX defaults f32 matmuls to
    TF32 (~10-bit mantissa), which silently wrecks the visibility/GP linear
    algebra in single precision (reduced chi^2 explodes to ~1e14) while leaving
    CPU and double precision untouched — so it only bites on GPU once x64 is
    genuinely off. "highest" pins true fp32 (and is a no-op under x64).

    Returns the resulting ``jax_enable_x64`` value.
    """
    import jax  # noqa: F811 (re-imported to guarantee it's available here)
    import sgp4jax  # noqa: F401 (enables x64 on import; must precede the toggle)

    x64 = config.get("model", {}).get("precision", "single") == "double"
    jax.config.update("jax_enable_x64", x64)
    jax.config.update("jax_default_matmul_precision", "highest")
    return x64


def run(args):
    if args.timings:
        enable_timings()

    config = load_config(args.config)

    set_precision(config)

    # Workers run the identical program (multi-process collectives require it) but
    # only process 0 talks: stdout, the log file, plots and result writes are all
    # rank-0-gated. Errors still reach stderr on every process.
    try:
        with suppress_worker_stdout():
            tabascal_subtraction(
                config,
                args.sim_dir,
                args.ms_path,
                args.suffix,
                extra_orbit_dir=args.extra_orbit_dir,
                norad_path=getattr(args, "norad_path", None),
                log=getattr(args, "log", True) and is_process_0(),
            )
    except (TLEError, TruthError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

    if is_process_0():
        print_memory_usage()
        if args.timings:
            print_timings()
