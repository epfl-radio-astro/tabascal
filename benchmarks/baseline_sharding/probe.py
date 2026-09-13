"""Measure placement, live buffers and executed MAP-step memory on a compute node.

Unlike compile-only estimates, stage snapshots include retained predictions and
runtime FFI scratch. Run each axis/scratch-budget combination in a fresh process:
JAX allocator peaks and the kernel's scratch budget are process-scoped.
"""
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import subprocess
import time

import jax
import numpy as np
import optax

from tabascal.config import load_config, TabConfig, Model
from tabascal.distributed import make_global, replicated_sharding, shard_pytree
from tabascal.scripts._run_tabascal_impl import set_precision
from tabascal.tab_tools import _map_step, init_predict, nlog_like_and_post
from tabascal.truth import load_truth


def describe(value):
    sharding = getattr(value, "sharding", None)
    return {
        "shape": list(value.shape), "dtype": str(value.dtype),
        "logical_bytes": int(value.nbytes),
        "sharding": str(sharding),
        "spec": str(getattr(sharding, "spec", "process-local/host")),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("ms")
    parser.add_argument("out")
    parser.add_argument("--orbit-dir", required=True)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--skip-init", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def emit(event, **fields):
        record = {"event": event, **fields}
        with (out / "probe.jsonl").open("a") as stream:
            stream.write(json.dumps(record, default=str) + "\n")
        print(json.dumps(record, default=str), flush=True)

    def survey(name, tree):
        for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
            if hasattr(value, "shape") and hasattr(value, "nbytes"):
                # Never call asarray here: inspecting a host constant must not
                # allocate a new device buffer and contaminate the measurement.
                emit("array", tree=name, path=jax.tree_util.keystr(path), **describe(value))

    def snapshot(stage):
        jax.block_until_ready(jax.live_arrays())
        jax.effects_barrier()
        stats = {str(d): d.memory_stats() for d in jax.local_devices()}
        live = Counter()
        buffers = {}
        for array in jax.live_arrays():
            live[(tuple(array.shape), str(array.dtype), str(array.sharding))] += 1
            for shard in array.addressable_shards:
                data = shard.data
                key = (str(shard.device), data.unsafe_buffer_pointer())
                buffers[key] = max(buffers.get(key, 0), data.nbytes)
        per_device = Counter()
        for (device, _), size in buffers.items():
            per_device[device] += size
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        emit("memory", stage=stage, stats=stats, live_buffer_bytes=dict(per_device), nvidia_smi_mib=smi)
        emit("live_arrays", stage=stage, arrays=[
            {"shape": shape, "dtype": dtype, "sharding": spec, "count": count}
            for (shape, dtype, spec), count in live.items()
        ])

    config = load_config(args.config)
    set_precision(config)
    ms = Path(args.ms).resolve()
    zarr = ms.with_suffix(".zarr")
    config["data"].update(ms_path=str(ms), truth_zarr=str(zarr), zarr_path=str(zarr), out_dir=str(out))
    config["satellites"]["extra_orbit_dir"] = args.orbit_dir
    emit("environment", axis=os.environ.get("TABASCAL_SHARD_AXIS", "source"),
         scratch_mb=os.environ.get("RI_KERNELS_INTERP_SCRATCH_MB"),
         jax=jax.__version__, devices=[str(d) for d in jax.devices()],
         preallocate=os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"))
    tc = TabConfig(config, str(ms))
    snapshot("config")
    # Stage setup separately: the full-width encoding of truth/data happens
    # before any optimizer executable exists and can set the run's peak.
    from tabascal.imports import import_components
    originals = []
    for cls in import_components(config["model"]["components"]):
        original = cls.setup
        def measured_setup(self, config, original=original, name=cls.__name__):
            result = original(self, config)
            snapshot("setup/" + name)
            return result
        originals.append((cls, original))
        cls.setup = measured_setup
    try:
        model = Model(tc, config["model"]["components"])
    finally:
        for cls, original in originals:
            cls.setup = original
    for comp in model.components:
        survey("component/" + comp.prefix + "/state_outputs", comp.state_outputs)
    snapshot("model_before_placement")
    for name in ("init_params", "state", "constants"):
        survey("before/" + name, getattr(model, name))
        setattr(model, name, shard_pytree(getattr(model, name), tc.n_rfi, tc.n_bl))
        survey("after/" + name, getattr(model, name))
    survey("observations", {"obs": tc.vis_obs, "noise": tc.noise, "flags": tc.flags})
    snapshot("model_after_placement")
    truth = load_truth(tc)
    truth = {k: make_global(v, replicated_sharding()) for k, v in truth.items()}
    survey("truth", truth)
    snapshot("truth")
    if not args.skip_init:
        # Retain this result for comparison with the reference probe. The runner
        # now releases it after export/plotting, before optimization.
        import tabascal.tab_tools as tab_tools
        original_metrics = tab_tools.print_truth_metrics
        def measured_metrics(*values, **options):
            snapshot("before_truth_metrics")
            result = original_metrics(*values, **options)
            snapshot("after_truth_metrics")
            return result
        tab_tools.print_truth_metrics = measured_metrics
        try:
            init_pred = init_predict(tc, model.prob_model, jax.random.PRNGKey(1), model.init_params,
                                     state=model.state, constants=model.constants, truth=truth)
        finally:
            tab_tools.print_truth_metrics = original_metrics
        jax.block_until_ready(init_pred)
        survey("init_prediction", init_pred)
        snapshot("init_prediction")
        values = nlog_like_and_post(model.prob_model, model.init_params, tc.vis_obs,
                                   state=model.state, constants=model.constants)
        jax.block_until_ready(values)
        snapshot("init_diagnostics")

    optimizer = optax.adabelief(config["opt"]["epsilon"])
    params = model.init_params
    opt_state = optimizer.init(params)
    survey("optimizer", opt_state)
    compiled = _map_step.lower(model.prob_model, optimizer, params, opt_state,
                               model.state, model.constants, tc.vis_obs).compile()
    memory = compiled.memory_analysis()
    emit("compiled_memory", **{name: getattr(memory, name) for name in (
        "argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes")})
    (out / "step.hlo.txt").write_text(compiled.as_text())
    emit("compiled_shardings", inputs=str(compiled.input_shardings), outputs=str(compiled.output_shardings))
    snapshot("compiled")
    for i in range(args.steps):
        start = time.perf_counter()
        params, opt_state, loss = compiled(params, opt_state, model.state, model.constants, tc.vis_obs)
        jax.block_until_ready((params, opt_state, loss))
        emit("step", index=i, seconds=time.perf_counter() - start, loss=float(loss))
        if i == 0 or i == args.steps - 1:
            snapshot(f"step_{i}")
    if not args.skip_init:
        del init_pred
    del truth
    gc.collect()
    snapshot("released_predictions_and_truth")
    emit("done")


if __name__ == "__main__":
    main()
