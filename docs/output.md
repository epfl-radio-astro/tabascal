# Output from TABASCAL

When you run TABASCAL on a dataset, a few directories are created in the same location as the Mesurement Set (MS) located. These directories are populated with memory profiling files, diagnostic plots, and the estimates for the different components. An example of the directory structure with descriptions is shown below

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

This shows that we have a dataset with 1 sample, 28 baselines, 1 frequency channel, 120 time steps and 8 antennas. We have just 1 sample becuase this is the prediction from the maximum a posteriori (MAP) estimate which is a single point. The same prediction datasets are also available for the initial parameters and the prior distribution.

The dataset contains the predictions for the astronomical visibilities, `ast_vis`, the gains, `gains`, the RFI visibilities, `rfi_vis`, and finally the observed visibility prediction, `vis_obs`. 

## Custom MS file columns

The results from tabascal shown above are also copied into the MS file used. Only custom non-standard columns are added and the original standard columns (`DATA`, `MODEL_DATA`, `CORRECTED_DATA`) are left untouched. The following non-standrd columns are added

* `TAB_AST_DATA`: The calibrated, astronomical visibility prediction.
* `TAB_RFI_DATA`: The calibratde RFI visibility prediction.
* `TAB_AST_RES`: The used data column (usually `DATA`) with the astronomical prediction subtracted. This should only contain the RFI signal and noise.
* `TAB_RFI_RES`: The used data column (usually `DATA`) with the RFI prediction subtracted. This should only contain the astronomical signal and noise.
* `TAB_RES_DATA`: The used data column with both the astronomical and RFI predictions subtracted. This should only contain noise.