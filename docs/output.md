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

The results from tabascal shown above are also copied into the MS file used. Six columns are written: the standard `CORRECTED_DATA`, and five custom non-standard `TAB_*` columns. `DATA` and `MODEL_DATA` are left untouched.

Below, `g_p` and `g_q` are the gain predictions for the two antennas of a baseline, so `g_p g_q*` is the per-baseline gain, and `DATA` stands for whichever column the run was given (usually `DATA`).

| Column | Contents | Frame |
| --- | --- | --- |
| `CORRECTED_DATA` | `DATA / (g_p g_q*)` — the calibrated data | Calibrated |
| `TAB_AST_DATA` | The astronomical visibility prediction | Calibrated (no gains applied) |
| `TAB_RFI_DATA` | The RFI visibility prediction | Calibrated (no gains applied) |
| `TAB_AST_RES` | `DATA - g_p g_q* × ast` — should contain only the RFI signal and noise | Data |
| `TAB_RFI_RES` | `DATA - g_p g_q* × rfi` — should contain only the astronomical signal and noise | Data |
| `TAB_RES_DATA` | `DATA - vis_obs` — should contain only noise | Data |

`TAB_RES_DATA` subtracts the `vis_obs` prediction stored in the results `.zarr`, which is the full forward model the gains component produced, rather than re-deriving it as `g_p g_q* × (ast + rfi)`.

### Correlations

TABASCAL fits a single correlation, named by `data.corr` in the configuration (default `xx`) and recorded in the results `.zarr`. The results are written into that correlation of the MS only, resolved by identity against `POLARIZATION::CORR_TYPE` rather than by position — on the `POLARIZATION` row that the data partition's `DATA_DESC_ID` points to, exactly as the run resolved it when reading — a single-polarisation MS holds one correlation whatever it is, so `yy` is index 0 there.

On the other correlations of a multi-correlation MS:

* `TAB_AST_DATA` and `TAB_RFI_DATA` — the model columns — are **0**;
* `CORRECTED_DATA`, `TAB_AST_RES`, `TAB_RFI_RES` and `TAB_RES_DATA` — the data-frame columns — carry the **data column passed through unchanged**: no gain applied and nothing subtracted, which is the honest value for a correlation that was never modelled.

A results `.zarr` written before the correlation was recorded does not say where it belongs. On a single-correlation MS there is only one answer; on a wider one, writing raises a `ValueError` rather than guessing. Pass the correlation explicitly in that case:

```bash
tab2MS -m path/to/file.ms -z path/to/results.zarr -c xx
```

### Why the residuals are in the data frame

The three residual columns are formed in the frame of the observed data, `DATA - gains × model`, rather than in the calibrated frame, `DATA / gains - model`. Dividing by the gain inflates the noise on low-gain baselines, which distorts any noise-referenced metric computed from the residual. Moving every column into a single calibrated frame, together with the `WEIGHT_SPECTRUM` that belongs to it, is tracked in [issue #123](https://github.com/epfl-radio-astro/tabascal/issues/123).

### Multiple samples

The predictions above are averaged over the `sample` axis of the results `.zarr`. Products are formed per sample and averaged afterwards: the written gain is the mean of `g_p g_q*`, not the product of the two mean gains, and `TAB_AST_RES` subtracts the mean of `g_p g_q* × ast`. For a MAP run there is only one sample and the distinction does not arise, but for a posterior with several samples the two orders give different answers whenever the quantities covary.

### Zero and non-finite gains

An unflagged dead antenna can be driven to a zero or non-finite gain by the fit. Any such antenna gain is replaced by 1 before anything is derived from it, and a `RuntimeWarning` is raised naming the affected antennas and the number of gain samples substituted. Baselines touching such an antenna are then written *uncalibrated on that antenna* — the other antenna's gain is still applied — rather than being blanked. The same substitution is applied once more to the *mean* baseline gain that `CORRECTED_DATA` is divided by, since per-sample gains that are all finite and non-zero can still average to zero; that case raises its own `RuntimeWarning`. Nothing is ever written as `NaN` or `inf`, so every column stays finite and imageable. The `vis_obs` stored in the results `.zarr` predates the substitution, so wherever it was substituted `TAB_RES_DATA` falls back to re-deriving the total as `g_p g_q* × ast + g_p g_q* × rfi` from the substituted gains. That choice is made per sample, so one bad sample does not discard the stored model on the others.

### When the results do not match the MS

Writing raises a `ValueError`, before any column is built, if:

* the results `.zarr` holds a different number of baselines than the MS — the results belong to another MS, or an antenna was dropped between the run and the write;
* the MS rows are not ordered time-major, so the first `n_bl` rows do not hold `n_bl` distinct antenna pairs;
* the baseline order differs between timesteps, which the `(n_time, n_bl)` reshape tabascal reads visibilities with cannot represent;
* the rows interleave timesteps within a block of `n_bl` rows, which reshapes cleanly but puts every visibility on the wrong timestamp.

Sorting the MS by `TIME`, `ANTENNA1`, `ANTENNA2` resolves the last three.
