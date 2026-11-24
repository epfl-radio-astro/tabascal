# Before every other import in this file: the imports below bring the JAX
# backend up, and this is read when it initialises. The helper imports nothing
# but os, so putting it above them costs nothing.
#
# It setdefaults rather than assigns. This line used to assign, and since an
# entry point had already deferred to XLA_PYTHON_CLIENT_PREALLOCATE by the time
# it imported this module, that took the user's choice back a few lines later.
# Called here as well as from the entry points so that importing this module
# directly still gets the default.
from tabascal.scripts._device_memory import default_memory_on_demand

default_memory_on_demand()

import os
import sys
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from datetime import datetime

import jax
from jax import random

from tabascal.timing import measure_runtime, print_timings, enable_timings
from tabascal.tab_tools import init_predict, run_opt, run_trace_path, nlog_like_and_post, write_ms_if_enabled
from tabascal.config import load_config, TabConfig, Model
from tabascal.schema import (
    ConfigValidationError,
    OrbitSourceConfig,
    RunConfig,
    validate_config,
)
from tabascal.distributed import (
    barrier,
    fail_together,
    is_process_0,
    make_global,
    replicated_sharding,
    shard_pytree,
    sharding_enabled,
    suppress_worker_stdout,
)
from tabascal.imports import import_components
from tabascal.jitter import jitter_init
from tabascal.opt_schedule import validate_opt
from tabascal.seeds import config_seed, describe_random_draws, run_key
from tabascal.write import check_input_column, write_results_xds
from tabascal.orbit import TLEError, save_orbits_for_reuse
from tabascal.truth import require_truth, load_truth, has_truth, TruthError
from tabascal.scripts._config_paths import config_is_unset, config_path
from tabascal.run_dir import (
    INIT_PRED,
    LOG_FILE,
    MAP_PRED,
    PLOTS_DIR,
    RESULTS_DIR,
    USED_ORBITS,
    create_run_dir,
    finish_record,
    place_outputs,
    read_config,
    start_record,
    write_config,
)

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
    if isinstance(config, RunConfig):
        precision = config.model.precision
        classes = config.model.component_classes()
    else:
        model_cfg = config.get("model", {})
        precision = model_cfg.get("precision", "single")
        raw_comps = model_cfg.get("components", []) or []
        refs = [
            next(iter(entry)) if isinstance(entry, dict) and len(entry) == 1 else entry
            for entry in raw_comps
        ]
        classes = import_components(refs)
    if precision == "double":
        return
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
    cfg = validate_config(config)
    assert_precision_supported(cfg)
    require_truth(cfg)
    tab_config = TabConfig(cfg, ms_path)
    model = Model(tab_config)
    return tab_config, model


@measure_runtime
def evaluate_init(tab_config, model, key, truth=None):
    key, subkey = random.split(key)
    init_pred = init_predict(tab_config, model.prob_model, subkey, model.init_params, state=model.state, constants=model.constants, truth=truth)
    return key, init_pred


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
    name: str
    run_dir: str
    log_path: str
    f_name: str
    ms_path: str
    plot_dir: str
    map_path: str
    init_pred_path: str
    used_orbits_path: str


def _resolve_paths(config, out_dir, ms_path, name, extra_orbit_dir, norad_path=None):
    """Resolve the run's inputs, create its directory, and write both into ``config``.

    Records the MS, truth zarr and every other input path back onto ``config``,
    absolute, and creates ``<out_dir>/runs/<name>_<time>_<token>/`` for the run
    to write into (see :mod:`tabascal.run_dir`). Returns a :class:`_RunPaths`
    bundling everything the run needs so the orchestration below stays free of
    path arithmetic.

    The MS comes from the ``-ms`` flag, else ``data.ms_path``, else the
    simulation layout ``<out_dir>/<basename>.ms`` -- the precedence
    ``rfi_estimate.resolve_ms_path`` also uses, so one config points a run and
    the light-curve extractor at the same visibilities. The config key used to
    be read and then written over by the derived path, which named a
    Measurement Set appearing nowhere in the config whenever the run was
    pointed at real data rather than a simulation (issue #207).

    ``out_dir`` is where the run's directory goes, under ``runs/``. It was
    called ``sim_dir`` and documented as the directory ``sim-vis`` creates,
    which is what it is on a simulation -- the MS and the truth zarr are looked
    for inside it -- but a real observation has no such directory, and an MS
    taken from a telescope bears no relation to the name of the folder it sits
    in. So the two are separate now: name the MS and the outputs go beside it,
    name the directory and the simulation layout is read out of it, or name
    both and neither is guessed.

    A consequence worth stating: ``-od`` does not move the MS when the config
    names one, and it no longer moves the truth zarr either -- that follows the
    MS, which is where sim-vis writes it. It moves the products, and an
    ``ms_path`` pointing outside the output directory is the whole case real
    data needs. ``-ms`` is how a run is moved onto other visibilities.

    One more, of the write-back rather than of the precedence. The resolved
    paths are recorded on ``config`` whether given or derived, and the run
    writes that config to its directory before it starts, so ``config.yaml``
    names the Measurement Set it is read on. That is what makes it a record of
    the run rather than of the intent, and it is why re-running one after
    moving the data needs ``-ms``. ``data.out_dir`` stays the parent, so a
    re-run lands beside the run it repeats.
    """

    config_ms_path = config.data.ms_path
    if ms_path:
        ms_path = os.path.abspath(ms_path)
    elif not config_is_unset(config_ms_path):
        ms_path = config_path(config_ms_path, "data.ms_path")
    else:
        ms_path = None

    config_out_dir = config.data.out_dir
    if out_dir:
        out_dir = os.path.abspath(out_dir)
    elif not config_is_unset(config_out_dir):
        out_dir = config_path(config_out_dir, "data.out_dir")
    elif ms_path:
        out_dir = os.path.dirname(ms_path)
    else:
        raise SystemExit(
            "Nothing to run on and nowhere to write. Provide -ms/--ms_path or "
            "data.ms_path to name the Measurement Set, and -od/--out_dir or "
            "data.out_dir to say where this run's directory should go, under runs/. "
            "Either implies the other: an MS alone writes beside itself, "
            "and a directory alone is read as a tab-sim simulation, whose MS "
            "and truth zarr are looked for inside it."
        )
    config.data.out_dir = out_dir

    if ms_path:
        stem, extension = os.path.splitext(os.path.basename(ms_path))
        f_name = stem if extension.lower() == ".ms" else os.path.basename(ms_path)
    else:
        f_name = os.path.basename(out_dir)
        ms_path = os.path.join(out_dir, f"{f_name}.ms")
    config.data.ms_path = ms_path

    config_truth_zarr = config.data.truth_zarr
    if not config_is_unset(config_truth_zarr):
        zarr_path = config_path(config_truth_zarr, "data.truth_zarr")
    else:
        zarr_path = os.path.join(os.path.dirname(ms_path), f"{f_name}.zarr")
    config.data.truth_zarr = zarr_path

    orbit_cfg = config.model.find_one(OrbitSourceConfig)
    if extra_orbit_dir and orbit_cfg is not None:
        orbit_cfg.extra_orbit_dir = extra_orbit_dir
    if norad_path:
        if orbit_cfg is None:
            raise SystemExit(
                "Cannot apply -np/--norad-path: model.components has no satellite "
                "trajectory component."
            )
        orbit_cfg.norad_ids_path = norad_path

    if not name:
        raise SystemExit("A run needs a name: give it with -n/--name.")

    try:
        # Resolves the inputs too, once, on process 0, for every process.
        run_dir = create_run_dir(out_dir, name, config)
    except OSError as error:
        # Reachable in a way it was not when the directory had to be given:
        # it now defaults to the MS's own, and an archive is often read-only.
        raise SystemExit(
            f"Cannot create a run directory under {out_dir}: "
            f"{error.strerror or error}. That is where this run writes. Give "
            "-od/--out_dir or data.out_dir to write somewhere else -- it "
            "defaults to the Measurement Set's own directory."
        ) from error

    place_outputs(config, run_dir)

    results_dir = os.path.join(run_dir, RESULTS_DIR)
    return _RunPaths(
        name=name,
        run_dir=run_dir,
        log_path=os.path.join(run_dir, LOG_FILE),
        f_name=f_name,
        ms_path=ms_path,
        plot_dir=os.path.join(run_dir, PLOTS_DIR),
        map_path=os.path.join(results_dir, MAP_PRED),
        init_pred_path=os.path.join(results_dir, INIT_PRED),
        used_orbits_path=os.path.join(results_dir, USED_ORBITS),
    )


def _print_run_header(run_dir, f_name, ms_path, start_time):
    """The run's identity, including which visibilities it is about to read.

    ``f_name`` is the MS's name, which used to be the output directory's and to
    determine the MS
    as well -- it was ``<out_dir>/<f_name>.ms`` and nothing else. Now that
    ``data.ms_path`` can name one anywhere, a header without it would leave a
    log unable to say where its visibilities came from, and a stale ms_path in
    a config swept over several ``-od`` directories would go unremarked.
    ``tabascal search`` names its MS for the same reason.
    """

    print()
    print(f"Start Time : {start_time}")
    print(f"Run : {run_dir}")
    print()
    print(f_name)
    print(f"MS : {ms_path}")
    print()


def _print_model_summary(tab_config, model, start_time):
    shapes = {key: value.shape for key, value in model.init_params.items()}
    n_params = sum(x.size for x in model.init_params.values())
    n_data = 2 * tab_config.vis_obs.size

    if tab_config.rfi_grid == "data":
        print("RFI visibilities integrated analytically per cell (no fine time grid).")
    else:
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
    config, out_dir, ms_path=None, name=None, extra_orbit_dir=None, norad_path=None, log=True
):
    config = validate_config(config)
    # Before the run directory exists: a config whose exports would replace
    # its own input is not a run that failed, it is one that cannot start.
    check_input_column(
        config.data.data_col,
        config.data.save_rfi_per_sat,
    )
    paths = _resolve_paths(config, out_dir, ms_path, name, extra_orbit_dir, norad_path)

    try:
        # Before any work, so a run that dies still says what it was and what it read.
        _on_process_0(
            "write the run's config.yaml and run.json",
            lambda: (write_config(paths.run_dir, config), start_record(paths.run_dir, config, config_seed(config))),
        )
        with _stdout_logger(paths.log_path, log):
            _subtract(config, paths)
        _on_process_0(
            "mark the run finished", lambda: finish_record(paths.run_dir, "finished")
        )
    except BaseException as error:
        if is_process_0():
            # Best effort, and never in place of the failure it is recording:
            # run.json may be what could not be written in the first place.
            try:
                finish_record(paths.run_dir, "failed", f"{type(error).__name__}: {error}")
            except Exception as record_error:
                print(f"\nCould not mark the run failed: {record_error}", file=sys.stderr)
        raise

    # Workers must not tear down the distributed runtime while process 0 is still
    # writing results / running its final collectives.
    barrier("tabascal-run-end")


def _on_process_0(what, work):
    """``work`` on process 0 alone, its failure raised on every process."""
    failure = None
    if is_process_0():
        try:
            work()
        except Exception as error:
            failure = error
    fail_together(failure, what)


def _record_resolved_settings(run_dir, model):
    """Write what model setup resolved (e.g. ``time_pad_factor: auto``) into ``config.yaml``."""
    saved = read_config(run_dir)
    comps = (saved.get("model") or {}).get("components")
    if not isinstance(comps, list):
        return
    changed = False
    for idx, comp in enumerate(model.components):
        if idx >= len(comps):
            break
        resolved = getattr(comp, "time_pad_factor", None)
        entry = comps[idx]
        if (
            isinstance(entry, dict)
            and len(entry) == 1
            and resolved is not None
            and resolved != "auto"
        ):
            ((ref, params),) = entry.items()
            if isinstance(params, dict) and params.get("time_pad_factor") == "auto":
                params["time_pad_factor"] = float(resolved)
                changed = True
    if changed:
        write_config(run_dir, saved)


def _subtract(config: RunConfig, paths):
    """The run itself, inside its directory: model, initial prediction, fit."""
    ms_path = paths.ms_path
    start_time = datetime.now()
    key, _ = random.split(run_key(config_seed(config)))

    _print_run_header(paths.run_dir, paths.f_name, paths.ms_path, start_time)
    print(describe_random_draws(config))
    print()
    # Here rather than in run_opt, so a bad opt key fails before the setup does.
    if config.inference.opt:
        validate_opt(config.opt, precision=config.model.precision)

    if sharding_enabled():
        print(f"Sharding RFI sources over {jax.device_count()} devices:")
    else:
        print("Running on single device:")

    print_devices()
    print()

    tab_config, model = build_model(config, ms_path)
    prob_model = model.prob_model
    _on_process_0(
        "record the resolved settings in config.yaml",
        lambda: _record_resolved_settings(paths.run_dir, model),
    )

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

    # Before sharding, so every rank perturbs the same local arrays the same way.
    model.init_params = jitter_init(
        model.init_params, config.opt.init_jitter, config_seed(config),
        n_rfi=tab_config.n_rfi, n_rfi_real=getattr(tab_config, "n_rfi_real", tab_config.n_rfi),
    )

    if sharding_enabled():
        model.init_params = shard_pytree(model.init_params, tab_config.n_rfi)
        model.state = shard_pytree(model.state, tab_config.n_rfi)
        model.constants = shard_pytree(model.constants, tab_config.n_rfi)

    _print_model_summary(tab_config, model, start_time)

    truth = load_truth(tab_config)
    if sharding_enabled():
        truth = {k: make_global(v, replicated_sharding()) for k, v in truth.items()}
    if not has_truth(truth):
        print("\nNo tab-sim truth available - skipping truth-based RMSE reporting.")

    key, init_pred = evaluate_init(tab_config, model, key, truth=truth)
    write_results_xds(init_pred, tab_config, paths.init_pred_path)
    optimise = config.inference.opt and config.opt.max_iter > 0

    if not optimise:
        nlog_l, nlog_p = nlog_like_and_post(prob_model, model.init_params, tab_config.vis_obs, state=model.state, constants=model.constants)
        print(f"log_l : {nlog_l:.3e}")
        print(f"log_p : {nlog_p:.3e}")

    if config.plots.init and is_process_0():
        from tabascal.plot import plot_init
        plot_init(tab_config, init_pred, truth, paths.name, paths.plot_dir)

    key, subkey = random.split(key)
    if config.plots.prior:
        if jax.process_count() > 1:
            print("Skipping prior plots: not supported in multi-process runs.")
        else:
            from tabascal.plot import plot_prior
            plot_prior(tab_config, prob_model, truth, paths.name, subkey, paths.plot_dir, state=model.state, constants=model.constants)

    key, *subkeys = random.split(key, 3)
    if optimise:
        vi_pred, losses, vi_params, _ = run_opt(
            tab_config, prob_model, subkeys, model.init_params, ms_path, paths.map_path,
            state=model.state, constants=model.constants, truth=truth,
        )

        if config.plots.opt and is_process_0():
            from tabascal.plot import plot_opt
            plot_opt(tab_config, vi_pred, truth, paths.name, paths.plot_dir)

        if config.plots.losses and is_process_0():
            import numpy as np
            from tabascal.plot import plot_losses
            with np.load(run_trace_path(tab_config.config.opt, paths.map_path)) as trace:
                plot_losses(dict(trace), paths.name, paths.plot_dir)

        print(f"log_p @ init : {losses[0] / 2:.3e}")
        print(f"log_p @ last step : {losses[-1] / 2:.3e}")
    else:
        write_ms_if_enabled(tab_config, ms_path, paths.init_pred_path)


def set_precision(config):
    """Enable/disable JAX float64 based on ``model.precision``."""
    import jax  # noqa: F811 (re-imported to guarantee it's available here)
    import sgp4jax  # noqa: F401 (enables x64 on import; must precede the toggle)

    if isinstance(config, RunConfig):
        precision = config.model.precision
    else:
        precision = (config.get("model") or {}).get("precision", "single")
    x64 = precision == "double"
    jax.config.update("jax_enable_x64", x64)
    jax.config.update("jax_default_matmul_precision", "highest")
    return x64


def run(args):
    if args.timings:
        enable_timings()

    try:
        config = load_config(args.config)
    except ConfigValidationError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

    if getattr(args, "skip_ms_write", False):
        config.data.skip_ms_write = True

    set_precision(config)

    try:
        with suppress_worker_stdout():
            tabascal_subtraction(
                config,
                args.out_dir,
                args.ms_path,
                args.name,
                extra_orbit_dir=args.extra_orbit_dir,
                norad_path=getattr(args, "norad_path", None),
                log=getattr(args, "log", True) and is_process_0(),
            )
    except (TLEError, TruthError, ConfigValidationError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Report even when the run dies mid-way (e.g. OOM): the peak up to
        # the failure is the number needed to diagnose it. Timings stay out of
        # here -- a timing table for a dead run misleads. (An XLA `Check
        # failure` process abort runs no `finally`, so it is still unreported.)
        if is_process_0():
            try:
                print_memory_usage()
            except Exception as e:
                # The report queries the very backend that may have just died, and
                # an exception raised in a `finally` replaces the one on its way
                # out. Never let the diagnostic destroy the failure it describes.
                print(f"\nCould not report memory usage: {e}", file=sys.stderr)

    if is_process_0() and args.timings:
        print_timings()
