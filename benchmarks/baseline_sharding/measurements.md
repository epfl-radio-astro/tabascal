# Verified result

**512 antennas at both 32 and 64 channels now complete on four GH200s.**
Job **4657092** completed with exit code **0:0**, including all 102 optimizer
steps, final predictions, zarr export and truth diagnostics for every case.

All cases use `GPVisAst`, analytic-only RFI (`quadrature_limit: 0`),
`RI_KERNELS_INTERP_SCRATCH_MB=1024`, and `--skip-ms-write`. The shipped analytic
crossover remains `null`. Enable the baseline route with
`TABASCAL_SHARD_AXIS=baseline`.

| Channels | Axis | Total | Optimization | Master peak (GB) | Max worker peak (GB) | Final chi² |
|---|---|---:|---:|---:|---:|---:|
| 8 | source | 251.88 s | 132.82 s | 33.911 | 28.664 | 83.42597198486328 |
| 8 | baseline | 261.88 s | 140.97 s | 13.703 | 12.578 | 83.42597198486328 |
| 32 | baseline | 922.23 s | 550.30 s | 48.360 | 47.336 | 84.39815521240234 |
| 64 | baseline | 1.87 ks | 1.13 ks | 94.590 | 93.857 | 84.91548156738281 |

The profiler rounds durations over 1000 seconds to kiloseconds. Peaks are
allocator `peak_bytes_in_use`, in decimal GB, within a 96.905 GB pool per GPU;
reserved memory reported by `nvidia-smi` is separate. The 64-channel run fits
with about **2.315 GB** below that pool limit on its limiting device, so it has
little margin for larger configurations or a larger scratch setting.

At 8 channels the exact reference chi² is preserved on both axes. Against the
handoff baseline, total time fell from 350.88 to 261.88 s and the limiting peak
from 25.186 to 13.703 GB. Optimization remains **6.1% slower than source
sharding**, despite removal of all baseline all-gathers; that gap is not claimed
as resolved. The antenna-grouping construction in `antenna_grouping.md` attains
an optimal common antenna width of 308 for this complete four-worker graph,
but is not enabled in production and has not been runtime-benchmarked.

The fixes keep the sky scan and transpose local, bound initialization FFT
batches, replace diagnostic boolean compaction with masked reductions, place
retained component placeholders, and release initial predictions before the
optimizer. The existing kernel seam required no further C++ changes.

The flat FFI pack/unpack order also agrees: eight index fields plus positions
per group; four shared tables; three tables and an optional antenna index per
group; then three signal arrays. The corresponding spec sequence follows the
same loops. No ordering mismatch was found; this remains positional code.

Validation included the 2895-pass full mini suite (42 expected skips),
187 initialization/diagnostic cases in each precision, eight four-device sky
and placement cases in each precision, and **33 GPU cases with zero skips**.
The final integer-input normalization guard passed all 165 diagnostic/noise
cases in each precision and leaves the benchmark's float32 path unchanged.

All four final configs have SHA-256
`7464b811f030bbbf40bdcc53f6912969188e557379039569887855e421010e7c`.
The GPU binary SHA-256 is
`a46447d0afcd61194c1a8c9441d4326ef0f3cb05c622e9a1dbeabcc98274550f`.
The frozen checkout is `~/pasc/chris/baseline-state-candidate`; results are
`~/pasc/chris/poly-interp-hyb/runs/state_cli_*_4657092`.
Local copies of raw logs, probe/HLO output and GPU test XML are in
`/private/tmp/tabascal-baseline-state-4657092`.

## Measurement history

## Reference audit: job 4656919

Four GH200s, 512 antennas, 8 channels, 150 integrations, analytic-only RFI,
`GPVisAst` sky, tabascal `eae6aa4`, kernel Python `248abf3`.
GPU binary SHA-256:
`a46447d0afcd61194c1a8c9441d4326ef0f3cb05c622e9a1dbeabcc98274550f`.
Each case is a fresh process. Times are the median of steps 1–5 after one
warmup step, not a replacement for the 102-iteration CLI benchmark.

| Axis | Scratch budget (MiB) | Step (s) | Worker peak (GB) |
|---|---:|---:|---:|
| source | 8192 | 1.2833 | 25.968 |
| baseline | 8192 | 1.3978 | 18.001 |
| baseline | 1024 | 1.3894 | 15.145 |
| baseline | 256 | 1.4384 | 15.145 |

These peaks include the probe's initialization and diagnostics, but not the
CLI's final prediction/export. Allocator peaks are not reserved GPU memory:
`nvidia-smi` reports approximately 94,000 MiB per device with 95% preallocation.

The baseline 8192 MiB case decomposes as follows:

- After placement: worker live allocations 2.416 GB.
- After truth loading: 4.933 GB.
- After initial prediction and diagnostics: 6.241 GB, peak 15.145 GB.
- Compiled optimizer: 1.321 GB arguments, 0.169 GB outputs, 3.334 GB temporaries.
- During the first executed step: live allocations 6.434 GB, peak 17.934 GB.
- After six steps: peak 18.001 GB.
- Releasing retained initialization predictions and truth: live allocations
  fall from 6.444 GB to 2.619 GB.

The analytic transpose's per-cell scratch formula in
`src/rfi_analytic_kernel_gpu.cu` is `sizeof(complex) * n_ant * n_rfi * n_offsets *
n_coefficients * (1 + ceil(n_ant / 32))`. At this shape, its full-cell allocation
is 8.022 GB. Together with the measured XLA temporaries, this accounts for the
execution-time rise; lowering the scratch budget confirms that it is not an
additional 13 GB of persistent replicated arrays. At 1024 MiB, initialization
sets the worker's peak instead of the optimizer step.

The optimized baseline HLO also contains:

- Three sky latent all-gathers into `[131028,203]` arrays (one real, two complex).
- A visibility-cotangent all-gather into `c64[131028,8,150]`, approximately
  1.258 GB, at `transpose(jvp())/reshape`.
- An RFI signal-gradient all-reduce of `c64[32,512,8,150]`. This reduction is
  required because the source/antenna signal is shared across baseline workers;
  it is distinct from the avoidable sky gathers.

The sky gathers come from padding and reshaping a global scan. The candidate
runs the same scan inside `map_over_baselines`, making both its padding and
its transpose local. Six four-device CPU tests on mini check values, gradients,
and the absence of all-gathers, for FFT and DFT components and divisible and
partial scan blocks.

The reference audit's GPU pytest invocation skipped all 15 tests because the
top-level conftest sets `CUDA_VISIBLE_DEVICES=0` when unset. This is not accepted
as correctness evidence. The scripts now explicitly expose devices `0,1,2,3`
and reject any skips. Candidate job 4657037 includes that corrected guard.

The initial full mini suite passed: 2885 passed, 36 skipped. Its prebuilt CPU
library does not supply the analytic kernel, so the four-GPU guard remains
required.

## Local sky scan: job 4657037

The corrected four-GPU guard passed all 21 cases without skips. At the
production shape, the optimized step has **no all-gather or collective-permute**
operations. The remaining collectives are the shared RFI amplitude gradient
and three scalar reductions. XLA temporaries fall from 3.334 GB to 1.526 GB
per device. This isolates the resharding cost from FFI scratch.

Additional stage snapshots locate the remaining initialization peaks:

- `GPVisAst.setup`: master peak 19.545 GB, largest allocation 5.023 GB
  (four full visibility arrays). Its signal-to-latent encoding vmaps the padded
  FFT over every baseline before the blocked forward exists.
- Before truth metrics: worker peak 12.503 GB, largest allocation 1.884 GB
  (three int32 index vectors for every visibility cell). Reduced chi-squared
  compacts arrays with a three-dimensional boolean mask.
- After truth metrics: worker peak 15.172 GB. The same indexing pattern appears
  in the truth error moments.

The next candidate bounds initialization FFT batches at the existing 64 MiB
sky budget, replaces diagnostic boolean compaction with fused masked reductions,
and releases saved initialization predictions before optimization. The probe
retains those predictions for comparison; the full CLI measures the lifetime
change. The shipped analytic crossover remains unchanged.

Initialization and diagnostic regressions passed 187 tests in each precision
on mini, including ragged masks, excluded NaNs/zero noise, all-flagged data,
resolved noise, FFT/DFT encoders, and batch tails. Job 4657071 verifies these
changes on GPUs and then runs the 8/32/64-channel CLI matrix.

Completed 102-step CLI rows from job 4657037 (same binary):

| Axis | Scratch (MiB) | Total (s) | Optimization (s) | Master peak (GB) | Worker peak (GB) | Final chi² |
|---|---:|---:|---:|---:|---:|---:|
| source | 8192 | 297.03 | 132.84 | 37.798 | 32.461 | 83.42597198486328 |
| baseline | 8192 | 309.53 | 141.05 | 24.665 | 16.689 | 83.42597198486328 |
| baseline | 1024 | 310.10 | 141.51 | 24.665 | 16.689 | 83.42597198486328 |

The local sky scan removes the observed gathers, but baseline optimization is
still slower than source optimization. Baseline workers materialize all RFI
sources on the group antenna axis; eliminating gathers alone does not eliminate
that redundant per-antenna kernel work. An antenna-aware baseline permutation
would be a separate change requiring an end-to-end ordering and runtime check.

With the initialization/diagnostic fixes, the complete mini suite passed:
2895 passed, 42 skipped, 29 existing write-metadata warnings. The additional
six skips are the four-device sky tests in this one-device full-suite run;
those tests passed separately on four CPU devices and in the four-GPU guard.

## Bounded initialization and diagnostics: job 4657071

All 31 four-GPU regression cases passed without skips. At 512A/8ch, baseline
axis, 1024 MiB scratch:

| Stage | Master peak (GB) | Worker peak (GB) |
|---|---:|---:|
| GPVisAst setup | 5.888 | 1.571 |
| Model after placement | 10.848 | 2.544 |
| Truth loaded | 12.290 | 6.184 |
| Before truth metrics | 13.446 | 8.116 |
| After truth metrics | 13.446 | 8.116 |
| Initial log-likelihood/posterior diagnostics | 15.004 | 9.670 |
| Six optimizer steps | 15.004 | 9.670 |

The largest allocation is now 1.526 GB (compiled optimizer temporaries),
versus the original 5.023 GB padded initialization array. The truth-metric
stage no longer raises either device's peak.

The live-array inventory also shows three full complex visibility cubes on
device 0 after model placement. The RFI, sky and gain components retain these
zero placeholders in `state_outputs`; replacing only the assembled model's
state leaves those original references alive. The next candidate places
component outputs immediately after setup, before assembly and forward
closures retain them. Four-device tests cover both divisible and indivisible
baseline counts and passed in both precisions. Job 4657092 measures this change
and runs the final channel matrix.

Completed CLI rows from job 4657071:

| Axis | Scratch (MiB) | Total (s) | Optimization (s) | Master peak (GB) | Max worker peak (GB) | Final chi² |
|---|---:|---:|---:|---:|---:|---:|
| source | 1024 | 255.50 | 133.82 | 34.104 | 28.770 | 83.42597198486328 |
| baseline | 1024 | 263.37 | 140.99 | 16.729 | 11.818 | 83.42597198486328 |

This job was deliberately stopped after the 8-channel pair when the retained
placeholder candidate was available, rather than repeating wider-channel runs
on superseded code. Its 32-channel run was interrupted, not an OOM result.

## Retained component placement: job 4657092

All 33 four-GPU cases passed without skips. The 8-channel probe master peak
fell from 15.004 GB to 11.796 GB. Worker peak rose from 9.670 GB to 10.276 GB
because retained placeholders now occupy their proper shards on all devices
instead of whole cubes on device 0. Live allocations after releasing probe
predictions/truth are 4.247 GB on the master and 3.142 GB on workers, versus
7.425 GB and 2.634 GB before this change. This confirms the retained-buffer
attribution; it is not a further optimizer resharding change.

Final candidate, completed 8-channel CLI rows:

| Axis | Scratch (MiB) | Total (s) | Optimization (s) | Master peak (GB) | Max worker peak (GB) | Final chi² |
|---|---:|---:|---:|---:|---:|---:|
| source | 1024 | 251.88 | 132.82 | 33.911 | 28.664 | 83.42597198486328 |
| baseline | 1024 | 261.88 | 140.97 | 13.703 | 12.578 | 83.42597198486328 |

Both axes use the same frozen checkout and binary on the same node. Baseline
optimization remains 6.1% slower than source optimization; this work has not
eliminated that gap. Baseline total time is 25.4% below the handoff's 350.88 s,
and its limiting-device peak is 45.6% below 25.186 GB. Both wider-channel runs completed successfully; the final table above records their results.

The completed 512A/32ch run used 922.23 s total and 550.30 s in optimization,
with peaks of 48.360 GB on device 0 and 47.336 GB on workers. Final chi² was
84.39815521240234. This includes all 102 steps, final prediction, export and
truth diagnostics.

A final scalar-normalization regression also checks integer-valued data: the
count uses at least float32 rather than narrowing to the data's integer dtype.
All 165 diagnostic/noise tests passed in each precision after that guard. It
does not change the float32 normalization used by the frozen GPU runs.
