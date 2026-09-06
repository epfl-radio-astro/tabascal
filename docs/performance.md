# Performance regression checks

TABASCAL's runtime and memory use are guarded by
[ReFrame](https://reframe-hpc.readthedocs.io/) checks in
[`ci/reframe/tabascal_perf_check.py`](https://github.com/epfl-radio-astro/tabascal/blob/main/ci/reframe/tabascal_perf_check.py).
The checks themselves are system-generic — they declare a `generic:default`
partition and run anywhere ReFrame does, a dev machine included (see [Running
the checks directly](#running-the-checks-directly)); it is the *reference
values* that belong to a node type. In this project's CI they run on, and their
references were measured on, the CSCS Daint GH200 partition defined in
[`ci/cscs.yml`](https://github.com/epfl-radio-astro/tabascal/blob/main/ci/cscs.yml).

They complement the [pipeline tests](pipeline_tests.md): the pipeline tests
catch a change in *what* TABASCAL infers, these catch a change in *how long it
takes*.

## What is checked

Two test classes, each parameterised over `variant` (`Riemann`, `RiemannFFI`)
and `precision` (`single`, `double`):

- **`TabascalPerfCheck`** — single GPU.
- **`TabascalMultiGpuPerfCheck`** — every GPU on the node. It unsets the
  `CUDA_VISIBLE_DEVICES=0` default that the rest of the pipeline pins, so it
  exercises the sharded solve.

The metrics compared against references are `total_runtime`,
`optimizer_runtime` and `memory_usage`. Metrics without a reference entry are
reported but not asserted.

The `Riemann` references describe a deliberate trade rather than a target: that
component bounds its peak memory by scanning the baseline axis and recomputing
each block in the backward pass (see
[`baseline_block_size`](config.md#rfi-signal)), which on the GH200 references
above is roughly an order of magnitude less memory for an optimiser step 1.2 to
1.7 times longer. A run that comes out faster and hungrier than these numbers has
probably lost the scan, not gained a speed-up.

References are keyed by `(variant, precision)` and are only meaningful on the
node type they were measured on. Sanity therefore requires the run to have used
exactly the expected number of GPUs of exactly the expected kind (`GH200`) —
anything else fails rather than being silently compared against timings it
cannot be compared against. That gate is on the *comparison*, not on running the
checks: elsewhere, run them with `-S strict_check=0` (below) to get the metrics
reported without being asserted.

## Reproducing a regression locally

When CI reports a regression, reproduce it on a smaller dataset with 8 antennas.
This is fast enough to simulate and run on any dev machine, so you do not need
GH200 access to bisect the cause:

```bash
# Generate an 8-antenna simulation from the standard 96A config
sim-vis -c ci/reframe/data/sim_target_96A.yaml -a 8

# Run tabascal with timing output against the generated dataset
tabascal run -c ci/reframe/data/tab_target.yaml -od ci/reframe/data/data/pnt_src_obs_08A_090T-0000-0890_001I_001F-1.500e+08-1.500e+08_050PAST_000GAST_000EAST_32SAT_0GRD_1.0e+00RFI -t
```

The `-t` flag prints a per-function timing table identical to the CI output.

The absolute numbers from an 8-antenna run on a laptop are not comparable to the
GH200 references — use it to find *which* function got slower, then confirm the
magnitude on the real configuration.

## Running the checks directly

With the `dev` environment installed (see [Developer install](installation.md)),
ReFrame is available as `reframe`:

```bash
reframe -C ci/reframe/settings.py \
        -c ci/reframe/tabascal_perf_check.py \
        --system=generic:default \
        --exec-policy=serial \
        --run --performance-report
```

Add `-S strict_check=0` to report the metrics without failing on the reference
comparison, which is what the nightly cron job does.

## Updating the references

A reference change and a regression look identical in the diff, so treat the
references the same way as the [pipeline test
references](pipeline_tests.md#re-recording-the-references): when a change
intentionally alters performance, re-measure rather than widening the
tolerances, and **record in the commit message why the numbers moved**, with the
before/after values.

The comments in `tabascal_perf_check.py` are part of that record. For example
the current values note that #103 replaced the real-space
`rfi_signal:ComplexRFI` with the scanned Fourier `ComplexRFIVarAnt`, a
deliberate accuracy-for-runtime trade (see issue #107) that made the optimiser
slower where the RFI-signal component dominates the step and slightly faster
where RFI-vis does. Those older, faster numbers describe a component that no
longer exists and must not be "restored".

## Benchmark history

The nightly cron run converts the ReFrame report with
[`ci/reframe/reframe_to_bmf.py`](https://github.com/epfl-radio-astro/tabascal/blob/main/ci/reframe/reframe_to_bmf.py)
and uploads it to [bencher.dev](https://bencher.dev/) under the `tabascal`
project, testbed `cscs-daint-gh200`. The `gpu_mode` variable keeps the
single- and multi-GPU series apart: anything other than `single` is appended to
the benchmark name, so the existing single-GPU series keep their historical
names.
