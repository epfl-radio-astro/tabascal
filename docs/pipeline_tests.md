# Pipeline tests

`tests/test_tabascal_pipeline.py` runs the whole of `run_tabascal` end to end and
compares the result against recorded reference values. These are the tests that
catch a change in what tabascal actually infers, as opposed to the unit tests,
which check individual functions.

Each case runs `run_tabascal.py` in a **subprocess** against a simulation
downloaded from the `epfl-radio-astro/rfi-simulations` HuggingFace dataset, then
parses the run's stdout.

```bash
pixi run -e dev pytest tests/test_tabascal_pipeline.py            # double precision
pixi run -e dev pytest tests/test_tabascal_pipeline.py --x64 false # single precision
```

`--x64` selects the precision requested of the subprocess (it is written into the
generated config as `model.precision`). Cases marked `requires_double` — the
SGP4/phase trajectory components, which need fp64 for orbit propagation — are
skipped under `--x64 false`.

## What is asserted

Each case is a `PipelineTestConfig` carrying its references:

- **`chi2_ref`** — the reduced chi² at the optimisation point. A scalar is
  asserted at 1% relative tolerance; a `(lo, hi)` tuple is asserted as inclusive
  bounds. One value covers **both** precisions — see "Which architecture to
  record on" below.
- **`metrics_ref`** — optional truth-based metrics at the `opt` point, as
  `{quantity: {metric: ref}}` where quantity is `ast`, `rfi` or `gains`. Only the
  metrics listed are checked, so a case can assert just what it cares about. Each
  value takes the same scalar-or-`(lo, hi)` form as `chi2_ref`.

The truth metrics come from `print_truth_metrics` (`tabascal/tab_tools.py`). The
two normally worth asserting are:

- **`NRMSE(noise)`** — the residual against the thermal-noise floor. This is the
  science-meaningful yardstick (below 1 means the residual is sub-noise) and the
  most architecture-stable normalisation. **Lower is better.**
- **`bias_significance`** — the coherent component of the error, in sigma:

  ```
  bias_significance = sqrt(2 * N_eff) * |ME| / RMSE
  ```

  where `|ME|` is the magnitude of the mean error and `N_eff` is the
  *correlation-deflated* effective sample size (`_effective_sample_size`). It is
  a "no significant coherent bias" guard — it catches e.g. RFI leaking
  systematically into the recovered sky — not a tight value to match.

### Reading bias_significance correctly

`bias_significance` is a **detectability** measure, not an error magnitude, and it
does not move with `NRMSE(noise)`. Note that `RMSE` is in the *denominator* and
`N_eff` is inside the square root, so the significance can rise while the fit
strictly improves.

This happened when the astronomical fringe rate was corrected (`RiemannVis` case,
double precision):

| | before | after | |
|---|---|---|---|
| ast RMSE | 1.989e-01 | 1.700e-01 | −14.5% (better) |
| ast \|ME\| (bias) | 1.205e-02 | 1.217e-02 | +1.0% (flat) |
| ast `N_eff` | 54 | 113 | +109% |
| ast `bias_significance` | 0.6 | 1.1 | +83% |

The total error dropped 14.5% and the coherent bias barely moved, yet the
significance nearly doubled. Both surviving factors push the same way: `N_eff`
more than doubled, so `sqrt(2 * N_eff)` grew ~44%, and the shrinking `RMSE` in
the denominator added another ~17%. The same bias measured over more
effectively-independent samples, against a smaller residual, is simply resolved
more sharply. A rising significance is only a problem if `|ME|` itself is
growing, or if it approaches the bound — check the absolute bias before
concluding anything from the sigma value alone.

Beware of rounding too: the significance is printed to one decimal, so the `rfi`
value in that same change reads as a jump from 0.1 to 0.2 when the underlying
values are 0.106 → 0.232.

## Re-recording the references

When a change intentionally alters what the model infers, the recorded values no
longer describe it and must be re-recorded. **Do not widen the tolerances to make
the existing numbers fit** — that silently discards the regression signal.

Run with `--record-refs` to skip the assertions and print the measured values
instead. This is the point of the flag: you do not have to make the assertions
pass, or hand-edit the file, just to find out what the new values are.

```bash
pixi run -e dev pytest tests/test_tabascal_pipeline.py --record-refs -s
pixi run -e dev pytest tests/test_tabascal_pipeline.py --record-refs -s --x64 false
```

`-s` is required — the values are printed, and pytest captures stdout without it.
For one case:

```bash
pixi run -e dev pytest "tests/test_tabascal_pipeline.py::test_pipeline[GPGains]" \
    --record-refs -s
```

Output per case, shaped to be read straight across into the config literal and
its arch table:

```
--- measured: GPGains [double] ---
    chi2_ref=0.8874142592424018,
    ast: {'NRMSE(noise)': 0.2617, 'NRMSE(signal)': 0.09797, 'RMSE': 0.17, 'bias_significance': 1.1}
    rfi: {'NRMSE(noise)': 0.4279, 'NRMSE(signal)': 0.0253, 'RMSE': 0.278, 'bias_significance': 0.2}
    gains: {'NRMSE(signal)': 0.0005328, 'RMSE': 0.0005328, 'bias_significance': 1.1}
```

Then:

1. Paste the `chi2_ref` scalar. Record both precisions and check they agree
   before pasting either — see below.
2. Move any `metrics_ref` bound whose measured value now falls outside it, keeping
   roughly the existing relative width. Leave the bounds that still hold — a
   change that only moves some of them is more credible than one that moves all.
3. Update the arch table comment in that case, so the next person can see what was
   measured where.
4. **Record in the commit message why the values moved**, with the before/after
   numbers. A reference change is indistinguishable from a regression without it.

### Filling in the GPU and x86 arch rows

The `dev` environment is CPU-only; the CUDA jaxlib lives in `cuda12-dev`. To fill
in a GPU row, and to pin the CPU row to the CPU backend on a machine that also
has a GPU:

```bash
# GPU rows
pixi run -e cuda12-dev pytest tests/test_tabascal_pipeline.py::test_pipeline \
    --record-refs -s
pixi run -e cuda12-dev pytest tests/test_tabascal_pipeline.py::test_pipeline \
    --record-refs -s --x64 false

# CPU rows on a GPU machine — otherwise jax picks the GPU
JAX_PLATFORMS=cpu pixi run -e dev pytest tests/test_tabascal_pipeline.py::test_pipeline \
    --record-refs -s
```

Target `::test_pipeline` explicitly rather than `-k test_pipeline`: the latter
also selects `test_pipeline_sharded_equivalence` and `test_pipeline_multiprocess`,
which `--record-refs` does not apply to and which take several minutes.

`conftest.py` pins `CUDA_VISIBLE_DEVICES=0`, so a GPU run is single-device and
does not exercise the sharded solve. Three cases are `requires_double` and are
skipped under `--x64 false`, so a single-precision run records three cases, not
six.

### Which architecture to record on

Any of them. Both precisions and all three architectures currently agree well
inside the 1% tolerance, which is why each case carries **one** scalar rather
than a per-precision split.

Measured across ARM (Apple silicon), x86 CPU and an NVIDIA GPU, in both
precisions:

- **Double is architecture-stable** to 5.5e-8 relative in the worst case (the
  two SGP4 orbit cases, where the propagation amplifies rounding; the non-orbit
  cases agree to ~1.7e-10), and the printed truth metrics agree to every digit
  shown.
- **fp32 agrees with fp64** to 2.4e-5 relative on `chi2`, and to the printed
  precision on every truth metric. The offset is the *same* 2.4e-5 on ARM, x86
  and CUDA alike — a precision effect, not an architecture one.

So a reference recorded on any platform in either precision is canonical for all
of them. Prefer the CI/x86 value when you have it, since that is what gates the
PR.

This was **not** true of the pre-#103 real-space GP model, where fp32
convergence rate differed markedly between architectures (ARM reached chi2 ~0.92
in 100 iterations while x86 was still at ~1.13, and the rfi residual nearly
doubled between GPU and x86). That is what the per-precision `(lo, hi)` bounds
existed to absorb. The Fourier model removed the need for them. If you
reintroduce a component whose fp32 convergence is architecture-dependent, that
split has to come back — measure both precisions on at least two architectures
before collapsing a reference to a scalar.

If a double value differs between two machines by much more than 1e-6, suspect a
stale reference rather than an architecture difference. That is what the earlier
"ARM runs ~0.7% high" comments in this file turned out to be: the references had
drifted while staying inside the 1% tolerance, so nothing failed until a real
change pushed them over.

## Adding a case

Append a `pytest.param(PipelineTestConfig(...), id="...")` to the relevant list
(`trajectory_configs`, `rfi_vis_configs`, `gains_configs`, …); they are
concatenated into `all_configs`. Give it an explicit `id` — that id is the test
name and the handle for `-k`. Record its references with `--record-refs` as above.
