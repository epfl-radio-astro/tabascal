# Output from TABASCAL

When you run TABASCAL on a dataset, a few directories are created in the same location as the Measurement Set (MS). These directories are populated with memory profiling files, diagnostic plots, and the estimates for the different components. An example of the directory structure with descriptions is shown below

```text
parent_directory/
|   └── dataset.ms                              # Original MS file with new columns added
|   ├── memory_profiles/
|       └── memory_1.prof                       # Memory profile for model initialisation
|       └── memory_2.prof                       # Memory profile for prediction of initial parameters
|       └── memory_3.prof                       # Memory profile for prediction of prior distribution
|   ├── plots/
|       └── Custom_init_ast_vis_imag.pdf        # Astronomical visibility prediction of initial parameters
|       └── Custom_init_gains_amp_phase.pdf     # Gain prediction of initial parameters
|       └── Custom_init_rfi_vis_imag.pdf        # RFI visibility prediction of initial parameters
|       └── Custom_map_ast_vis_imag.pdf         # Astronomical visibility prediction of optimised parameters
|       └── Custom_map_gains_amp_phase.pdf      # Gain prediction of optimised parameters
|       └── Custom_map_rfi_vis_imag.pdf         # RFI visibility prediction of optimised parameters
|       └── Custom_opt_loss.pdf                 # Optimiser loss over the optimisation period
|       └── Custom_prior_ast_vis_imag.pdf       # Astronomical visibility prediction of prior distribution
|       └── Custom_prior_gains_amp_phase.pdf    # Gain prediction of prior distribution
|       └── Custom_prior_rfi_vis_imag.pdf       # RFI visibility prediction of prior distribution
|   ├── results/
|       └── init_pred_Custom.zarr               # Initial parameter prediction values
|       └── init_pred_Custom.B                  # The calibration the initial values imply
|       └── map_pred_Custom.zarr                # Optimised parameter prediction values
|       └── map_pred_Custom.B                   # The calibration it implies, as a CASA table
```

## Results `.zarr` files

The results of the run are saved into a `.zarr` file inside the `results` directory as shown above. The contents of this file can be accessed using [Xarray](https://xarray.com/) as follows

```python
import xarray as xr
result_fp = ""
xds = xr.open_zarr(result_fp)
```

The output of this above shows the following dataset structure.

```text
<xarray.Dataset> Size: 178kB
Dimensions:  (sample: 1, bl: 28, freq: 1, time: 120, ant: 8)
Coordinates:
  * freq     (freq) float64 8B 1.227e+09
  * time     (time) float64 960B 0.0 2.0 4.0 6.0 8.0 ... 232.0 234.0 236.0 238.0
Dimensions without coordinates: sample, bl, ant
Data variables:
    ast_vis  (sample, bl, freq, time) complex128 54kB dask.array<chunksize=(1, 28, 1, 120), meta=np.ndarray>
    gains    (sample, ant, freq, time) complex128 15kB dask.array<chunksize=(1, 8, 1, 120), meta=np.ndarray>
    rfi_vis  (sample, bl, freq, time) complex128 54kB dask.array<chunksize=(1, 28, 1, 120), meta=np.ndarray>
    vis_obs  (sample, bl, freq, time) complex128 54kB dask.array<chunksize=(1, 28, 1, 120), meta=np.ndarray>
```

This shows that we have a dataset with 1 sample, 28 baselines, 1 frequency channel, 120 time steps and 8 antennas. We have just 1 sample because this is the prediction from the maximum a posteriori (MAP) estimate which is a single point. The same prediction datasets are also available for the initial parameters and the prior distribution.

The dataset contains the predictions for the astronomical visibilities, `ast_vis`, the gains, `gains`, the RFI visibilities, `rfi_vis`, and finally the observed visibility prediction, `vis_obs`. It also carries a `corr` attribute naming the correlation the run fitted, which is what tells the MS writer where the results belong.

A run with [`data.save_rfi_per_sat: true`](config.md) stores one more variable, `rfi_vis_src (sample, src, bl, freq, time)`, with a `norad_id` coordinate on the `src` axis: the same `rfi_vis` split into its per-satellite contributions. It is `n_rfi` times the size of `rfi_vis` on disk, which is why it is opt-in — but only one `(sample, satellite)` block of it is ever in memory, since each is written into the store as it is evaluated.

## MS file columns

The results from tabascal shown above are also copied into the MS file used. Six visibility columns are written: the standard `CORRECTED_DATA`, and five custom non-standard `TAB_*` columns, plus the two weight columns described below. `DATA` and `MODEL_DATA` are left untouched.

**Every column is written in one frame: the data with all the gains divided out.** There are two gain layers and both come off:

* the **external** calibration named by [`data.gain_table`](config.md), which is divided out of the visibilities when the MS is read. The MS's own data column is still raw, so the writer has to remove it again — pass the same tables to `tab2MS -gt` (see below), or the columns land a whole calibration layer away from the frame the models are in;
* the **DIE gains the model fitted**, stored in the results `.zarr`.

Below, `g_p` and `g_q` are the fitted gain predictions for the two antennas of a baseline, `g_ext` the per-baseline product of the external tables' gains, and

```text
g_total = g_ext × g_p g_q*
```

`DATA` stands for whichever column the run was given (usually `DATA`).

| Column | Contents | Frame |
| --- | --- | --- |
| `CORRECTED_DATA` | `DATA / g_total` — the calibrated data | Calibrated |
| `TAB_AST_DATA` | The astronomical visibility prediction | Calibrated |
| `TAB_RFI_DATA` | The RFI visibility prediction | Calibrated |
| `TAB_AST_RES` | `CORRECTED_DATA - ast` — should contain only the RFI signal and noise | Calibrated |
| `TAB_RFI_RES` | `CORRECTED_DATA - rfi` — should contain only the astronomical signal and noise | Calibrated |
| `TAB_RES_DATA` | `CORRECTED_DATA - (ast + rfi)` — should contain only noise | Calibrated |

One frame means the decomposition closes:

```text
TAB_AST_DATA + TAB_RFI_DATA + TAB_RES_DATA == CORRECTED_DATA
```

The identity is **exact** — the same floating-point numbers, not merely to a tolerance — wherever Sterbenz's condition holds **per component**: for the real parts and for the imaginary parts *separately*, the model's component and the calibrated data's component must lie within a factor of two of one another, which makes `data − model` exactly representable and the sum reconstruct `data` bit for bit.

That is a condition on the components, not on the magnitudes. A residual that is small next to `|data|` is not enough on its own: a cell whose real part very nearly cancels while its imaginary part is large can have a residual that dwarfs its own real part, and there the identity holds only to within **one `float32` ulp of the visibility's magnitude**. The same bound applies on a fit so bad that a cell is all residual.

Either way it is worth checking after any change to the writer: a column written in the wrong frame is wrong by a *gain*, which is seven orders of magnitude above an ulp, so this closes loudly.

The total model subtracted by `TAB_RES_DATA` is the sum of the two model columns, not the `vis_obs` forward model stored in the results `.zarr`. That stored model is in the *gained* frame, and a gains component that gains only one of its two terms leaves it in no single frame at all — `ast + rfi / g` is not a frame either model column is written in.

### Weights

Dividing by the gain makes the noise heteroscedastic: a low-gain baseline is noisier once calibrated. That is why the calibrated frame is usable — the writer also writes the weights that belong to it:

| Column | Contents |
| --- | --- |
| `WEIGHT_SPECTRUM` | `\|g_total\|² / SIGMA²`, per (row, channel) |
| `WEIGHT` | the frequency mean of `WEIGHT_SPECTRUM` |

`SIGMA` here is the noise of the **raw** data as the MS records it, read exactly as the run reads it: `SIGMA_SPECTRUM` where the MS carries a usable one, `SIGMA` behind it, on the correlation that was fitted rather than correlation 0, with the same per-cell median collapse over time and median fill for cells that measured nothing (see the `tabascal.noise` module). Calibrating divides the noise by `|g_total|`, so `sigma_cal = SIGMA / |g_total|` and the weight rises with the square.

`WEIGHT_SPECTRUM` rather than `WEIGHT` alone because a frequency-dependent gain gives a frequency-dependent weight, which a single per-row number cannot carry — and which CASA's `applycal(calwt=True)` does not produce either: it was measured to apply one per-row factor, constant across channels, even when `WEIGHT_SPECTRUM` exists.

Both columns are **overwritten**, and whatever they held before is gone — if the MS carried weights from some other source (a re-weighting task, a hand-tuned scheme), copy them elsewhere first. Dividing the new `WEIGHT_SPECTRUM` by `|g_total|²` recovers `1 / SIGMA²`, the uncalibrated inverse-variance weighting implied by the MS's own noise column, and nothing else. Where a gain was substituted with 1 (see below) the weight is simply `1 / SIGMA²`, which is correct: those visibilities were written uncalibrated.

If the MS carries no usable noise column at all, nothing is invented: a warning is printed and the two weight columns are left exactly as they were, while the visibility columns are still written.

### Correlations

TABASCAL fits a single correlation, named by `data.corr` in the configuration (default `xx`) and recorded in the results `.zarr`. The results are written into that correlation of the MS only, resolved by identity against `POLARIZATION::CORR_TYPE` rather than by position — on the `POLARIZATION` row that the data partition's `DATA_DESC_ID` points to, exactly as the run resolved it when reading — a single-polarisation MS holds one correlation whatever it is, so `yy` is index 0 there.

On the other correlations of a multi-correlation MS:

* `TAB_AST_DATA` and `TAB_RFI_DATA` — the model columns — are **0**;
* `WEIGHT_SPECTRUM` and `WEIGHT` are **0** too, matching them: nothing there was calibrated, so nothing there has this frame's weight;
* `CORRECTED_DATA`, `TAB_AST_RES`, `TAB_RFI_RES` and `TAB_RES_DATA` carry the **data column passed through unchanged**: no gain applied and nothing subtracted, which is the honest value for a correlation that was never modelled. The closure identity still holds there, trivially.

A results `.zarr` written before the correlation was recorded does not say where it belongs. On a single-correlation MS there is only one answer; on a wider one, writing raises a `ValueError` rather than guessing. Pass the correlation explicitly in that case:

```bash
tab2MS -m path/to/file.ms -z path/to/results.zarr -c xx
```

### Writing results for a run that used a gain table

The external calibration is applied when the MS is read, so the MS's data column on disk is still raw and the writer has to remove that layer itself. `tabascal` does this automatically — the run hands its own `data.gain_table` list to the writer. Running `tab2MS` by hand, name the same tables, in the same order:

```bash
tab2MS -m path/to/file.ms -z path/to/results.zarr -gt path/to/flux.B0 -gt path/to/phase.G0
tab2MS -m path/to/file.ms -z path/to/results.zarr -gt path/to/flux.B0,path/to/phase.G0
```

Repeat `-gt` or give a comma-separated list; the two forms are equivalent and both mirror `data.gain_table`'s ordered list, since the tables compose in order. The gains are placed on this observation's grid exactly as the reader placed them — the MS's own `TIME` column in the unit and on the scale it declares, and the channel frequencies of the partition's spectral window — so the columns land in the frame the models were fitted in. Omitting the flag for a run that used a table writes every column, and the weights, one calibration layer out.

A visibility no table could supply a gain for is written *uncalibrated on that layer* rather than blanked, the same convention as a dead fitted gain.

### Multiple samples

The predictions above are averaged over the `sample` axis of the results `.zarr`. The baseline gain is formed per sample and averaged afterwards, so the divisor is the mean of `g_p g_q*` and not the product of the two mean gains; the model columns are the sample means of `ast` and `rfi`. For a MAP run there is only one sample and the distinction does not arise, but for a posterior with several samples the two orders give different answers whenever the two antennas' gains covary. Because every column shares one divisor, that choice reaches all of them.

### Per-satellite RFI columns

`TAB_RFI_DATA` is the RFI model summed over every satellite the run fitted. A run with [`data.save_rfi_per_sat: true`](config.md) also stores that sum's parts, and they can be written into the MS as one column per satellite — `TAB_RFI_<NORAD id>`, e.g. `TAB_RFI_58126`:

```bash
tabascal rfi-per-sat -m path/to/file.ms -z path/to/results/map_pred_Custom.zarr
```

`tab2MS-persat` is the same tool under the name that matches `tab2MS`, and `-p` changes the `TAB_RFI_` prefix. It reads `rfi_vis_src` back out of the zarr and needs nothing else: no re-fit, no configuration, and no gain tables — so the export can be made long after the run, and re-made. A zarr from a run that did not store the decomposition is an error saying which option produces one.

**What they are for.** Image one column at a time. A real satellite is a clean streak in exactly one per-source image; a feature that shows up in several is astronomical flux the RFI model has split across satellites. Reduced chi² cannot see that — a fixed total split differently between sources costs it nothing — so the per-source images are the diagnostic for it.

The columns are in the same **calibrated frame** as every other column above, and they get there the same way `TAB_RFI_DATA` does: they are model visibilities, fitted to data whose gain layers were already divided out, so nothing is applied to them. Correlations that were not fitted are `0`, matching the model columns. The correlation comes off the zarr, and `-c` overrides it for a zarr written before that was recorded, exactly as for `tab2MS`.

They decompose `TAB_RFI_DATA`:

```text
sum over satellites of TAB_RFI_<NORAD> == TAB_RFI_DATA
```

**exactly in exact arithmetic, and not bit for bit in floating point.** Two separate things stand between the two sides, and only the first of them can be given a bound.

**The writer's rounding.** Both sides make the same `complex64` cast in the same place (before the sample mean, not after it) and go through the same row mapping, but the per-source columns are rounded individually and their sum is not the rounded sum. *Given* that the zarr's `rfi_vis` is the exact-arithmetic sum over sources of `rfi_vis_src`, this contributes at most

```text
|Σ_r TAB_RFI_<NORAD_r> − TAB_RFI_DATA|
    ≤ (n_src + n_sample + 2) × ulp32( max_s Σ_r |rfi_vis_src[s, r]| )
```

per component — the real and imaginary parts separately, since a `complex64` cast rounds each of them on its own — counting one ulp for each of the `n_src` terms cast into its column, one for each step of the two sample means, and one for the sum itself.

**The scale is the per-sample, per-source values, taken before the sample mean**, because that is where the rounding happens. Referencing it to the columns instead would not be a bound at all: two samples of `+A` and `−A` average to a column of exactly zero while the total was rounded from `A`, so the columns can be zero and the difference nowhere near it. For a MAP run — one sample — the two readings coincide, but the guarantee is the one above.

**The decomposition's own re-association**, which is the hypothesis above and has *no* bound in these coarse values. The RFI-visibility op reduces over source *and* integration sample together; evaluating it one satellite at a time splits that single reduction into per-source partial sums. The difference is bounded by the **fine-grid** terms behind each visibility, not by the coarse values they average to, and where the fine grid cancels the two are nothing like each other: a source whose fine samples are `[A, −A]` beside one whose are `[1, 0]` gives a joint evaluation of exactly `0` and per-source values summing to exactly `0.5` — a difference equal to the whole coarse visibility. Whether it happens at all depends on the kernel's accumulation order (`RiemannVis` sums the sources at each fine sample and loses the `1`; the FFI and variable kernels accumulate source-major on that input and lose nothing).

On fitted grids it is round-off: measured at ~2e-16 relative in double precision and ~6e-8 in single. Treat the sum-back as an exact identity that floating point perturbs, rather than as a guarantee with a number on it, and read a large discrepancy as what it usually is — a dropped source or a column in the wrong frame, which are orders of magnitude larger again — before suspecting cancellation.

A satellite is named by its NORAD id, so a run that somehow fitted the same satellite twice is refused rather than writing one column per *pair* of sources. Sources the sharding padded the list with are not written at all: they carry no signal and name no satellite.

Note that a run narrowed to one channel with `data.freq` is refused here too — the results then cover part of the MS's band, and nothing in them says which part.

### Zero and non-finite gains

An unflagged dead antenna can be driven to a zero or non-finite gain by the fit. Any such antenna gain is replaced by 1 before anything is derived from it, and a `RuntimeWarning` is raised naming the affected antennas and the number of gain values substituted. Baselines touching such an antenna are then written *uncalibrated on that antenna* — the other antenna's gain is still applied — rather than being blanked. The same substitution is applied once more to the *mean* baseline gain that `CORRECTED_DATA` is divided by, since per-sample gains that are all finite and non-zero can still average to zero; that case raises its own `RuntimeWarning`. Both warnings are emitted *after* the columns have been written: their counts come out of the same single pass that writes the data, so nothing is evaluated twice and nothing full-size is held in memory — which also means that a process promoting `RuntimeWarning` to an error will find the MS already written when it raises. The gain handling never introduces a `NaN` or `inf` from a zero or non-finite gain: no column is divided by, or multiplied with, one. It does not guard against a gain that is finite but so large that the `complex64` baseline product `g_p g_q*` overflows, nor against a `g_total` whose two finite, non-zero layers underflow to zero when they are multiplied together — a calibration of that size is a failed fit, not a dead antenna, and it is written as it comes. Values that were already non-finite are written as they are — whatever the data column holds, including on correlations that were not fitted, which pass through unchanged, and any model value that the fit itself left non-finite in `ast_vis` or `rfi_vis`. The `vis_obs` stored in the results `.zarr` is not read at all, so a zero gain that it was formed with has nothing to leak into.

### When the results do not match the MS

Writing raises a `ValueError`, before any column is built, if:

* the results `.zarr` holds a different number of baselines than the MS — the results belong to another MS, or an antenna was dropped between the run and the write;
* the results `.zarr` holds a different number of channels than the MS. A run narrowed with [`data.freq`](config.md) covers part of the MS's band. The results do record which part — the `freq` coordinate holds the channel frequencies the run was fitted on — but the writer does not yet use it to place a partial band on the MS's channel axis, so writing a narrowed run back to a full-band MS is not yet supported, the initial-prediction export included;
* the MS rows are not ordered time-major, so the first `n_bl` rows do not hold `n_bl` distinct antenna pairs;
* the baseline order differs between timesteps, which the `(n_time, n_bl)` reshape tabascal reads visibilities with cannot represent;
* the rows interleave timesteps within a block of `n_bl` rows, which reshapes cleanly but puts every visibility on the wrong timestamp.

Sorting the MS by `TIME`, `ANTENNA1`, `ANTENNA2` resolves the last three.

## Calibration table

Every run that fits gains also writes the calibration it implies as a CASA calibration table, beside the results `.zarr` and named after it — `map_pred_Custom.zarr` gives `map_pred_Custom.B`. It is a `B Jones` table, one row per (time, antenna) with `CPARAM` of shape (channel, polarisation), exactly the layout `casatasks.gaincal` produces, so a tabascal solution can be consumed by standard tooling like any other calibration rather than only as the columns above.

Every results `.zarr` the run writes gets its own table, named after it. A configuration that writes the initial prediction as well as the optimised one therefore leaves an `init_pred_Custom.B` beside `init_pred_Custom.zarr`: the calibration those initial values imply, which is a real if uninteresting calibration and is never confused with the fitted one.

### Writing the table somewhere else

`tab2MS -o` (`--caltable-path`) names the output, for a pipeline that wants the calibration under a name of its own rather than the results':

```bash
tab2MS -m path/to/file.ms -z path/to/results.zarr -o path/to/cal/tabascal.B
```

The path given is the **only** one the export uses: it writes there, every rule below is applied to it, and a table an earlier run left *there* is what a rerun with nothing to export removes. The default `<results>.B` is not written, read or removed.

There is no configuration key for it, so the automatic export at the end of a run always names its table after the results `.zarr` it describes. That pairing is what the stale-table rules below are stated in terms of — a table left by an earlier run of *these* results — and a path fixed in a configuration, which is reused across runs, would break it: every run would write over one table, and a rerun that fitted no gains would remove one belonging to another run's results. Re-export with `tab2MS -o` when the table has to live elsewhere.

The table's `TIME` column is a copy of the MS's, and its `MEASINFO` record declares the scale **the MS declares** rather than assuming UTC. Declared to declared, the same convention the external gain tables are matched on: relabelling a TAI-declared observation as UTC would move every timestamp by the accumulated leap seconds — 37 s since 2017 — for anything that reads the declaration.

The table holds the **total** calibration — the external tables placed on this observation's grid, times the DIE gains the model fitted:

```text
G_p = g_ext_p × g_p
```

so one application of this one table takes `DATA` to `CORRECTED_DATA`:

```python
from casatasks import applycal
applycal(vis="dataset.ms", gaintable=["results/map_pred_Custom.B"])
```

That works because the two forms of the calibration agree: the columns are divided by a per-*baseline* gain and the table carries a per-*antenna* one, and `(g_ext_p g_p) conj(g_ext_q g_q)` is `(g_ext_p conj(g_ext_q)) (g_p conj(g_q))` — the very `g_total` the columns were divided by.

**`applycal` operates on `DATA`.** An MS whose visibilities live in a non-standard column cannot consume the table, whatever the table says — it fails with `Error in Calibrater::correct` regardless of the calibration given.

The table records which correlation was fitted in a `FittedCorr` keyword, since nothing in the caltable format says so and applying an `xx` solution to `yx` data is a silent mistake.

### Flagged in the table, substituted in the columns

A gain that is zero or non-finite carries no solution, and the two outputs give the honest answer each is able to give — deliberately different answers:

* the **table** flags it: `FLAG` set and `CPARAM` `NaN`, which is what CASA does with an unsolved antenna and what a reader of the table expects. The raw gains from the `.zarr` are used here, not the substituted ones, and an antenna no external table could supply a gain for is flagged the same way;
* the **columns** substitute 1, as described above, because a `NaN` there is a dropped visibility rather than a statement about the calibration.

So an antenna the fit killed is *missing* from the table and *uncalibrated* in the columns. Applying the table to that MS therefore flags those visibilities where the columns kept them.

### Several samples

The table carries the mean of the per-antenna gains over the `sample` axis. Where the columns are divided by the mean of `g_p g_q*`, a per-antenna table can only hold the mean of `g_p`, and the two differ by the covariance between the two antennas' gains. For a MAP run — one sample — they are identical.

### When no table is written

Nothing is written when there is no calibration to export: results with no `gains`, gains that are not the four-dimensional `(sample, antenna, channel, time)` grid a caltable can hold, or gains that are exactly 1 **on every sample**. A `UnitaryGains` run stores ones and fits nothing, and a run that used external tables while fitting nothing is in the same position — the total calibration is then the tables the caller already has.

Every-sample rather than on the mean: samples either side of 1 average to exactly 1 while the divisor the columns use — the mean of the baseline *product* — is a real calibration, so testing the mean alone would throw one away.

**A table from a previous run of the same results is removed** in that case, with a `RuntimeWarning` saying so. A stale calibration sitting under the current name beside the current results reads as the current solution, which is a bad thing to be wrong about.

Only a calibration table is ever removed: the path has to be a casacore table — casacore's own structural check, not just the marker files — whose INFO record declares `Type = Calibration`, and it may not be, contain, or sit inside the MS. A directory of your own that happens to be named `<results>.B` — or named by `-o` — is left exactly as it is, and so is a table too damaged for casacore to open. The same rule applies to *writing*: the export replaces a previous calibration table at that path and refuses anything else, rather than deleting it.

### If the export fails

The export is the last thing the writer does and is additive to it. Everything it can say about *this* MS or *these* results is reported as a `RuntimeWarning` — carrying the exception type, its message and the tail of its traceback — while the columns, which were already written, stand:

* an MS with more than one spectral window, which the writer serves one partition of while a caltable can only file rows under one window's id;
* an output path overlapping the MS — results written *inside* the MS put the table there too, and `-o` can name a path inside it outright;
* something at the output path that is not a calibration table;
* results whose gains do not describe the MS's antennas or timesteps;
* a `TIME` column declaring a time scale tabascal cannot interpret;
* anything the filesystem refuses.

**A table from an earlier run is removed here too**, and the warning says so. A run whose export fails is in the same position as one that has nothing to export: the table it could not replace would otherwise stand beside the new results as if it were this run's answer. It is removed under exactly the rules above — so a directory of your own in the way is reported and left alone.

A `MemoryError` is not demoted — that is a statement about the process, not about the data — and `Ctrl-C` stops the run as it always would.

The export also runs when the gain warnings above are promoted to errors (`python -W error::RuntimeWarning`, or a `warnings.simplefilter` in a calling script). Those warnings are raised after the columns are written, so a process that turns them into exceptions would otherwise stop with the columns updated and the previous run's table still beside them; the export runs in a `finally` instead. The exception still propagates, and if the export needs to warn under the same filter its warning arrives *chained* to the first one rather than replacing it — both reports survive in the traceback.
