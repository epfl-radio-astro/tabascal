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

- **`chi2_ref`** — the reduced chi² at the optimisation point, per precision.
  A scalar is asserted at 1% relative tolerance; a `(lo, hi)` tuple is asserted
  as inclusive bounds.
- **`metrics_ref`** — optional truth-based metrics at the `opt` point, as
  `{precision: {quantity: {metric: ref}}}` where quantity is `ast`, `rfi` or
  `gains`. Only the metrics listed are checked, so a case can assert just what it
  cares about.

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

This happened when the astronomical fringe rate was corrected:

| | before | after | |
|---|---|---|---|
| ast RMSE | 1.902e-01 | 1.631e-01 | −14% (better) |
| ast \|ME\| (bias) | 1.280e-02 | 1.171e-02 | −8.5% (better) |
| ast `N_eff` | 51 | 112 | +120% |
| ast `bias_significance` | 0.7 | 1.1 | +57% |

Both the total error and the coherent bias got *smaller*; the significance rose
because `N_eff` more than doubled, so `sqrt(2 * N_eff)` grew ~48% and dominated.
A smaller residual measured over more effectively-independent samples resolves
the same bias more sharply. A rising significance is only a problem if `|ME|`
itself is growing, or if it approaches the bound — check the absolute bias before
concluding anything from the sigma value alone.

Beware of rounding too: the significance is printed to one decimal, so the `rfi`
value "increasing" from 0.2 to 0.3 in that same change was really 0.249 → 0.260.

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
    chi2_ref={"double": 0.9150190232947091},
    arch table:  ast NRMSE(noise) 0.251 ast sig 1.1 | rfi NRMSE(noise) 0.386 rfi sig 0.2 | gains RMSE 3.9e-04 | chi2 0.915
    ast: {'NRMSE(noise)': 0.2514, ...}
```

Then:

1. Paste the `chi2_ref` scalar for the precision you ran.
2. Move any `metrics_ref` bound whose measured value now falls outside it, keeping
   roughly the existing relative width. Leave the bounds that still hold — a
   change that only moves some of them is more credible than one that moves all.
3. Update the arch table comment in that case, so the next person can see what was
   measured where.
4. **Record in the commit message why the values moved**, with the before/after
   numbers. A reference change is indistinguishable from a regression without it.

### Which architecture to record on

Double-precision values are architecture-stable: measured ARM (Apple silicon) and
CI/x86 values agree to better than 1e-8 relative, so a double reference recorded
on either is canonical for both. Prefer the CI/x86 value when you have it, since
that is what gates the PR.

Single precision is **not** architecture-stable — fp32 convergence rate differs
markedly between ARM, x86 and GPU — which is why those references are wide
`(lo, hi)` bounds rather than scalars.

If a double value differs between two machines by much more than 1e-8, suspect a
stale reference rather than an architecture difference. That is what the earlier
"ARM runs ~0.7% high" comments in this file turned out to be: the references had
drifted while staying inside the 1% tolerance, so nothing failed until a real
change pushed them over.

## Adding a case

Append a `pytest.param(PipelineTestConfig(...), id="...")` to the relevant list
(`trajectory_configs`, `rfi_vis_configs`, `gains_configs`, …); they are
concatenated into `all_configs`. Give it an explicit `id` — that id is the test
name and the handle for `-k`. Record its references with `--record-refs` as above.
