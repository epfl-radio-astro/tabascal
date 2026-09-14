# Baseline sharding memory audit

Run this on a compute node, not on a laptop or a cluster login node. The probe
uses the same model and explicit parameter/state/constant placement as the
production runner, then compiles **and executes** `_map_step`.

The older `shard_mem.py` probe from job 4656875 stopped before the runner's
`shard_pytree` calls and never executed the compiled step. Its XLA memory report
is useful, but it does not account for initialization predictions retained by
the runner or allocations requested by the FFI kernels at execution time.

`probe.py` writes:

- `probe.jsonl`: every state, parameter, constant, observation, optimizer and
  prediction leaf with its actual sharding; stage memory statistics; physical
  live-array buffer totals deduplicated by device pointer; timed, synchronized
  optimizer steps and their losses.
- `step.hlo.txt`: the optimized executable, including collective operations.

Stage snapshots distinguish configuration, model construction, placement,
truth loading, initial prediction, initial diagnostics, compilation and step
execution. The last snapshot releases predictions and truth, making their
retained memory visible separately from the optimizer's footprint. Live array
buffers do not include internal XLA temporary allocations or FFI scratch.
Allocator peaks are cumulative within a process; a stage's unchanged peak does
not mean that stage allocated nothing. `nvidia-smi` memory is recorded separately
because reserved pool size is not the same as `peak_bytes_in_use`.

`probe.sbatch` runs the GPU equivalence tests first and records the loaded kernel
binary's SHA-256. Every case uses that same binary and the same configuration.
The current C++ sources are unchanged by the Python seam and placement changes,
so this script reuses the installed binary; rebuild it before using this script
to compare a C++ change. Both `eval_with_indices` and
`analytic_eval_with_indices` and both GPU libraries are required by the guard.

The matrix compares source and baseline sharding at an 8192 MiB scratch budget,
then baseline sharding at 1024 and 256 MiB. Each case is a fresh process because
the kernel reads its scratch budget once. The shipped analytic crossover stays
`null`; this benchmark's configuration explicitly sets `quadrature_limit: 0`.
The probe does not write the input Measurement Set or prediction products.

After syncing the checkout, submit from the cluster's benchmark directory:

```sh
cd ~/pasc/chris/poly-interp-hyb
sbatch ~/pasc/chris/dft-gp-ast/benchmarks/baseline_sharding/probe.sbatch
```

The first executed step includes runtime initialization. Use subsequent step
times for a steady-state comparison, then verify any candidate with the full
102-iteration CLI benchmark, `--skip-ms-write`, and the reference reduced chi^2.
Only then attempt the 512-station 32- and 64-channel fits.

`candidate.sbatch` isolates the local sky-scan change against both axes and two
scratch budgets. `memory.sbatch` adds bounded initialization, masked diagnostic
reductions and the shorter prediction lifetime, then verifies both axes at
8 channels before attempting baseline-sharded 32- and 64-channel fits. These
scripts use separate frozen checkout paths to avoid changing a running job's
code. All CLI cases use 102 iterations and `--skip-ms-write`.

`state.sbatch` additionally places retained component placeholders during model
assembly. It records component output layouts as well as the model layouts,
requires 33 GPU tests with no skips, and repeats the complete channel matrix.
