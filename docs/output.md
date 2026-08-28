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
|       └── map_pred_Custom.zarr                # Optimised parameter prediction values
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

exactly — the same floating-point numbers, not merely to a tolerance, wherever the residual is no larger than the data (which is what a fitted model gives; on a fit so bad that a cell is all residual the identity holds only to a `float32` ulp). It is worth checking after any change to the writer: a column written in the wrong frame is wrong by a *gain*, so this closes loudly.

The total model subtracted by `TAB_RES_DATA` is the sum of the two model columns, not the `vis_obs` forward model stored in the results `.zarr`. That stored model is in the *gained* frame, and a gains component that gains only one of its two terms leaves it in no single frame at all — `ast + rfi / g` is not a frame either model column is written in.

### Weights

Dividing by the gain makes the noise heteroscedastic: a low-gain baseline is noisier once calibrated. That is why the calibrated frame is usable — the writer also writes the weights that belong to it:

| Column | Contents |
| --- | --- |
| `WEIGHT_SPECTRUM` | `\|g_total\|² / SIGMA²`, per (row, channel) |
| `WEIGHT` | the frequency mean of `WEIGHT_SPECTRUM` |

`SIGMA` here is the noise of the **raw** data as the MS records it, read exactly as the run reads it: `SIGMA_SPECTRUM` where the MS carries a usable one, `SIGMA` behind it, on the correlation that was fitted rather than correlation 0, with the same per-cell median collapse over time and median fill for cells that measured nothing (see the `tabascal.noise` module). Calibrating divides the noise by `|g_total|`, so `sigma_cal = SIGMA / |g_total|` and the weight rises with the square.

`WEIGHT_SPECTRUM` rather than `WEIGHT` alone because a frequency-dependent gain gives a frequency-dependent weight, which a single per-row number cannot carry — and which CASA's `applycal(calwt=True)` does not produce either: it was measured to apply one per-row factor, constant across channels, even when `WEIGHT_SPECTRUM` exists.

Both columns are **overwritten**. The MS's original weights are recoverable as `WEIGHT_SPECTRUM / |g_total|²`. Where a gain was substituted with 1 (see below) the weight is simply `1 / SIGMA²`, which is correct: those visibilities were written uncalibrated.

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

### Zero and non-finite gains

An unflagged dead antenna can be driven to a zero or non-finite gain by the fit. Any such antenna gain is replaced by 1 before anything is derived from it, and a `RuntimeWarning` is raised naming the affected antennas and the number of gain values substituted. Baselines touching such an antenna are then written *uncalibrated on that antenna* — the other antenna's gain is still applied — rather than being blanked. The same substitution is applied once more to the *mean* baseline gain that `CORRECTED_DATA` is divided by, since per-sample gains that are all finite and non-zero can still average to zero; that case raises its own `RuntimeWarning`. Both warnings are emitted *after* the columns have been written: their counts come out of the same single pass that writes the data, so nothing is evaluated twice and nothing full-size is held in memory — which also means that a process promoting `RuntimeWarning` to an error will find the MS already written when it raises. The gain handling never introduces a `NaN` or `inf` from a zero or non-finite gain: no column is divided by, or multiplied with, one. It does not guard against a gain that is finite but so large that the `complex64` baseline product `g_p g_q*` overflows, nor against a `g_total` whose two finite, non-zero layers underflow to zero when they are multiplied together — a calibration of that size is a failed fit, not a dead antenna, and it is written as it comes. Values that were already non-finite are written as they are — whatever the data column holds, including on correlations that were not fitted, which pass through unchanged, and any model value that the fit itself left non-finite in `ast_vis` or `rfi_vis`. The `vis_obs` stored in the results `.zarr` is not read at all, so a zero gain that it was formed with has nothing to leak into.

### When the results do not match the MS

Writing raises a `ValueError`, before any column is built, if:

* the results `.zarr` holds a different number of baselines than the MS — the results belong to another MS, or an antenna was dropped between the run and the write;
* the MS rows are not ordered time-major, so the first `n_bl` rows do not hold `n_bl` distinct antenna pairs;
* the baseline order differs between timesteps, which the `(n_time, n_bl)` reshape tabascal reads visibilities with cannot represent;
* the rows interleave timesteps within a block of `n_bl` rows, which reshapes cleanly but puts every visibility on the wrong timestamp.

Sorting the MS by `TIME`, `ANTENNA1`, `ANTENNA2` resolves the last three.
