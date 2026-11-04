"""One directory per ``tabascal run``, and the record that makes it reproducible.

Every run writes into a directory of its own, created for it and never reused::

    <out_dir>/runs/<name>_<YYYYmmddTHHMMSS>_<6 hex>/
        config.yaml   the resolved config, every input path absolute
        run.json      argv, host, versions, seed, input fingerprints, status
        log.txt
        plots/
        results/

``config.yaml`` and ``run.json`` are written before any work starts, so a run
that dies still says what it was. ``tab2MS``, ``tabascal rfi-per-sat`` and
``tabascal light-curve -z`` take the directory and read what they need from
these two files instead of having it repeated on their command lines.

The fingerprints are what let a later reader of the directory say whether the
inputs it names are still the ones the run read. They are compared, never
enforced: a changed input is a warning naming it.

Nothing heavy is imported at module level: the export commands' parsers come
through here, and ``tabascal -h`` must not pay for JAX.
"""

import glob
import hashlib
import json
import os
import secrets
import socket
import sys
import warnings
from datetime import datetime

import yaml

RUNS_DIR = "runs"
LATEST = "latest"
CONFIG_FILE = "config.yaml"
RECORD_FILE = "run.json"
LOG_FILE = "log.txt"
PLOTS_DIR = "plots"
RESULTS_DIR = "results"
INIT_PRED = "init_pred.zarr"
MAP_PRED = "map_pred.zarr"
USED_ORBITS = "used_orbits.json"

#: Plain files below this size are hashed; larger ones are fingerprinted by
#: size and mtime alone, which is what directories get file by file.
HASH_LIMIT = 64 * 2**20

#: The config keys that name an input file or directory, as (section, key).
#: ``data.ms_path``, ``data.out_dir`` and ``data.truth_zarr`` are resolved by
#: the run itself, with their own precedence; the rest are made absolute here.
#: Keys that also take a non-path value (``gains.init: prior``, a numeric
#: ``data.noise``, an inline ``ast.point_sources`` list) are paths only when
#: they are strings.
INPUT_KEYS = (
    ("data", "ms_path"),
    ("data", "gain_table"),
    ("data", "noise"),
    ("data", "truth_zarr"),
    ("gains", "init"),
    ("rfi", "est"),
    ("ast", "point_sources"),
    ("satellites", "extra_orbit_dir"),
    ("satellites", "norad_ids_path"),
)

#: Distributions whose versions decide what a run computes.
_VERSIONED = (
    "tabascal", "jax", "jaxlib", "numpyro", "optax", "numpy", "dask", "dask-ms",
    "xarray", "zarr", "ri_kernels", "sgp4jax", "satchecker-client",
)


# ---------------------------------------------------------------------------
# Creating the directory
# ---------------------------------------------------------------------------

def run_dir_name(name, now=None):
    """``<name>_<YYYYmmddTHHMMSS>_<6 hex>``: sortable by start, unique by token."""

    now = now or datetime.now()
    return f"{name}_{now:%Y%m%dT%H%M%S}_{secrets.token_hex(3)}"


def make_run_dir(out_dir, name, now=None):
    """Create a fresh run directory under ``<out_dir>/runs`` and return its path.

    ``os.mkdir`` without ``exist_ok``: two runs can never share a directory, and
    if the full name ever did collide this fails loudly rather than mixing them.
    """

    runs = os.path.join(out_dir, RUNS_DIR)
    os.makedirs(runs, exist_ok=True)
    path = os.path.join(runs, run_dir_name(name, now))
    os.mkdir(path)
    for sub in (PLOTS_DIR, RESULTS_DIR):
        os.mkdir(os.path.join(path, sub))

    return path


def update_latest(runs, run_dir):
    """Point ``runs/latest`` at ``run_dir``, atomically.

    A relative link, so the tree survives being moved. Made under a temporary
    name and renamed over the old one, so a reader never finds it missing. A
    failure here costs a convenience and nothing else, so it only warns.
    """

    link = os.path.join(runs, LATEST)
    tmp = os.path.join(runs, f".{LATEST}.{os.path.basename(run_dir)}")
    try:
        os.symlink(os.path.basename(run_dir), tmp)
        os.replace(tmp, link)
    except OSError as error:
        if os.path.lexists(tmp):
            os.unlink(tmp)
        warnings.warn(f"Could not update {link}: {error}", RuntimeWarning)


def create_run_dir(out_dir, name, config=None):
    """Resolve ``config``'s inputs and create the run directory, on process 0 only.

    Process 0 resolves the inputs (:func:`absolutise_inputs`) and makes the
    directory, then broadcasts both: the directory is made once however many
    processes the run has, and every process applies the very paths process 0
    resolved. Resolving per process could disagree -- another run starting in
    between moves ``runs/latest`` -- and a worker reading a different gain
    table from process 0 fails where process 0 waits. A failure on process 0
    reaches the workers as an empty payload, so they stop too.

    ``runs/latest`` moves only after the inputs were resolved, so one given
    through it -- an earlier run's ``results/`` as an orbit directory -- names
    that earlier run, not this empty one.
    """

    from tabascal.distributed import (
        RankZeroFailure,
        broadcast_bytes_from_rank0,
        is_process_0,
    )

    config = {} if config is None else config
    payload, failure = b"", None
    if is_process_0():
        try:
            absolutise_inputs(config)
            path = make_run_dir(out_dir, name)
            payload = json.dumps({"run_dir": path, "inputs": _input_values(config)}).encode()
        except Exception as error:
            failure = error

    received = broadcast_bytes_from_rank0(payload, "run-dir")
    if failure is not None:
        raise failure
    if not received:
        raise RankZeroFailure("process 0 failed to create the run directory.")

    shared = json.loads(received.decode())
    for label, value in shared["inputs"].items():
        section, key = label.split(".")
        config.setdefault(section, {})[key] = value
    if is_process_0():
        update_latest(os.path.dirname(shared["run_dir"]), shared["run_dir"])
    return shared["run_dir"]


# ---------------------------------------------------------------------------
# The config
# ---------------------------------------------------------------------------

def _is_path(value):
    return isinstance(value, (str, os.PathLike)) and str(value) != ""


def absolutise_inputs(config):
    """Resolve every input path in ``config`` in place, links included.

    Relative to the working directory, as ``data.ms_path`` and the CLI flags
    already are, so ``config.yaml`` names exactly what was read wherever it is
    re-run from. ``realpath`` rather than ``abspath``: an input named through
    ``runs/latest`` is recorded as the run it named then, not as whatever the
    link names later. ``data.gain_table`` is always recorded as a list.
    The MS and truth zarr are resolved by the run itself, with its precedence.
    """

    for section, key in INPUT_KEYS:
        block = config.get(section) or {}
        value = block.get(key)
        if (section, key) in (("data", "ms_path"), ("data", "truth_zarr")):
            continue
        if section == "gains" and value == "prior":
            continue
        if key == "gain_table" and _is_path(value):
            value = [value]
        if isinstance(value, (list, tuple)) and key == "gain_table":
            block[key] = [os.path.realpath(v) if _is_path(v) else v for v in value]
        elif _is_path(value):
            block[key] = os.path.realpath(value)


def _input_values(config):
    """``{"section.key": value}`` for the inputs :func:`absolutise_inputs` resolves."""

    return {
        f"{section}.{key}": config[section][key]
        for section, key in INPUT_KEYS
        if (section, key) not in (("data", "ms_path"), ("data", "truth_zarr"))
        and key in (config.get(section) or {})
    }


def place_outputs(config, run_dir):
    """Take a relative ``opt.trace_path`` inside the run directory."""

    # join keeps an absolute path as it is.
    trace = (config.get("opt") or {}).get("trace_path")
    if _is_path(trace):
        config["opt"]["trace_path"] = os.path.join(run_dir, trace)


def write_config(run_dir, config):
    with open(os.path.join(run_dir, CONFIG_FILE), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def read_config(run_dir):
    with open(os.path.join(run_dir, CONFIG_FILE)) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def input_paths(config):
    """``{label: path}`` for every input ``config`` names as a path."""

    paths = {}
    for section, key in INPUT_KEYS:
        value = (config.get(section) or {}).get(key)
        label = f"{section}.{key}"
        if section == "gains" and value == "prior":
            continue
        if key == "gain_table" and _is_path(value):
            value = [value]
        if isinstance(value, (list, tuple)) and key == "gain_table":
            for i, v in enumerate(value):
                if _is_path(v):
                    paths[f"{label}[{i}]"] = str(v)
        elif _is_path(value):
            paths[label] = str(value)

    return paths


def _ms_fingerprint(path):
    """What a run read from the MS, not the files it is stored in.

    ``tab2MS`` writes columns into the MS, so its file metadata changes with
    every export. The rows, the time axis, the channels, the antennas and the
    correlations do not, and they are what the results are tied to.
    """

    import numpy as np
    from daskms import xds_from_table

    digest = hashlib.sha256()
    time = np.asarray(xds_from_table(path, columns=["TIME"])[0].TIME.values)
    digest.update(time.tobytes())
    for sub, column in (("SPECTRAL_WINDOW", "CHAN_FREQ"), ("POLARIZATION", "CORR_TYPE")):
        for xds in xds_from_table(f"{path}::{sub}", columns=[column], group_cols="__row__"):
            digest.update(np.asarray(xds[column].values).tobytes())
    n_ant = int(xds_from_table(f"{path}::ANTENNA")[0].sizes["row"])

    return {"n_row": int(time.size), "n_ant": n_ant, "sha256": digest.hexdigest()}


#: Inputs that are fingerprinted by what is read from them, not by their files.
#: The orbit directory may be an earlier run's ``results/``, whose zarrs and
#: ``.B`` that run's ``tab2MS`` rewrites; only its ``*.json`` files are orbits.
_KINDS = {"data.ms_path": "ms", "satellites.extra_orbit_dir": "orbits"}


def fingerprint(path, kind=None):
    """A comparable summary of the file or directory at ``path``.

    Never raises: an input that cannot be read is the run's to report, and the
    record says only that it could not be fingerprinted.
    """

    try:
        return _fingerprint(path, kind)
    except Exception as error:
        return {"error": str(error)}


def _fingerprint(path, kind):
    if not os.path.exists(path):
        return {"exists": False}
    if kind == "ms":
        return {"ms": _ms_fingerprint(path)}
    if os.path.isdir(path):
        if kind == "orbits":
            # What the orbit resolver reads: *.json directly inside, nothing else.
            files = sorted(glob.glob(os.path.join(glob.escape(path), "*.json")))
        else:
            files = sorted(
                os.path.join(root, name)
                for root, _, names in os.walk(path)
                for name in names
            )
        digest = hashlib.sha256()
        for full in files:
            st = os.stat(full)
            digest.update(
                f"{os.path.relpath(full, path)}\0{st.st_size}\0{st.st_mtime_ns}\n".encode()
            )
        return {"files": len(files), "sha256": digest.hexdigest()}

    st = os.stat(path)
    record = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
    if st.st_size < HASH_LIMIT:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(2**20), b""):
                digest.update(block)
        record["sha256"] = digest.hexdigest()
    return record


def fingerprint_inputs(config):
    return {
        label: {"path": path, **fingerprint(path, _KINDS.get(label))}
        for label, path in input_paths(config).items()
    }


def _same(recorded, current):
    """Equal by content where both were hashed, else by every recorded field."""

    recorded = {k: v for k, v in recorded.items() if k != "path"}
    if "sha256" in recorded and "size" in recorded:
        return (recorded["sha256"], recorded["size"]) == (
            current.get("sha256"), current.get("size")
        )
    return recorded == current


def check_inputs(record, overrides=None):
    """Warn, naming each one, about inputs that differ from what the run read.

    ``overrides`` maps a label to the path a caller is using instead of the
    recorded one -- a moved MS -- and that path is the one compared.
    """

    overrides = overrides or {}
    changed = []
    for label, recorded in (record.get("inputs") or {}).items():
        path = overrides.get(label, recorded.get("path"))
        current = fingerprint(path, _KINDS.get(label))
        if not _same(recorded, current):
            changed.append(label)
            warnings.warn(
                f"{label} ({path}) is not what the run read: it has changed, "
                "moved or gone since. The results may no longer describe it.",
                RuntimeWarning,
                stacklevel=2,
            )
    return changed


# ---------------------------------------------------------------------------
# run.json
# ---------------------------------------------------------------------------

def _versions():
    from importlib.metadata import PackageNotFoundError, version

    found = {}
    for dist in _VERSIONED:
        try:
            found[dist] = version(dist)
        except PackageNotFoundError:
            pass
    return found


def write_record(run_dir, record):
    """Replace ``run.json`` atomically, so a reader never sees half of one."""

    path = os.path.join(run_dir, RECORD_FILE)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def read_record(run_dir):
    with open(os.path.join(run_dir, RECORD_FILE)) as f:
        return json.load(f)


def start_record(run_dir, config, seed):
    """Write the opening ``run.json``: status ``running``."""

    record = {
        "name": os.path.basename(run_dir),
        "status": "running",
        "started": datetime.now().isoformat(timespec="seconds"),
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "host": socket.gethostname(),
        "versions": _versions(),
        "seed": seed,
        "inputs": fingerprint_inputs(config),
    }
    write_record(run_dir, record)
    return record


def finish_record(run_dir, status, error=None):
    """Mark the run ``finished`` or ``failed``, keeping everything else."""

    record = read_record(run_dir)
    record["status"] = status
    record["ended"] = datetime.now().isoformat(timespec="seconds")
    if error is not None:
        record["error"] = error
    write_record(run_dir, record)


# ---------------------------------------------------------------------------
# Reading a run back
# ---------------------------------------------------------------------------

def is_run_dir(path):
    return os.path.isfile(os.path.join(path, RECORD_FILE))


def exported_results(run_dir, config):
    """The results zarr a run exports: ``map_pred`` if it optimised, else ``init_pred``."""

    optimised = config["inference"]["opt"] and config["opt"]["max_iter"] > 0
    path = os.path.join(run_dir, RESULTS_DIR, MAP_PRED if optimised else INIT_PRED)
    if not os.path.isdir(path):
        raise SystemExit(
            f"{path} does not exist: the run did not get as far as writing it. "
            f"Its {RECORD_FILE} and {LOG_FILE} say why."
        )
    return path


def load_run(path):
    """``(run_dir, config, record)`` for a run directory, or a message saying why not.

    ``runs/latest`` resolves to the run it points at, so what is reported and
    written names the run and not the link.
    """

    run_dir = os.path.realpath(path)
    if not is_run_dir(run_dir):
        raise SystemExit(
            f"{path} is not a tabascal run directory: it has no {RECORD_FILE}. "
            f"Give one of the directories under <out_dir>/{RUNS_DIR}/."
        )

    record = read_record(run_dir)
    if record.get("status") != "finished":
        warnings.warn(
            f"{run_dir} has status {record.get('status')!r}, not 'finished'.",
            RuntimeWarning,
            stacklevel=2,
        )

    return run_dir, read_config(run_dir), record


def override(label, given, recorded):
    """``given`` if set, warning when it differs from what the run recorded."""

    def norm(value):
        if isinstance(value, (list, tuple)):
            return [os.path.abspath(str(v)) for v in value]
        return [os.path.abspath(str(value))]

    if not given:
        return recorded
    if recorded is not None and norm(given) != norm(recorded):
        warnings.warn(
            f"{label}: using {given}, not {recorded} as the run recorded.",
            RuntimeWarning,
            stacklevel=2,
        )
    return given
