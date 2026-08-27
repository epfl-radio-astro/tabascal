# Configuration File

TABASCAL makes use of a configuration file to fully define the model used, what satellites to include and what inference to do. The configuration file is in YAML format using only the most basic constructs and types. The configuration file is broken up into multiple sections. Each section will be described below.

## Model

The forward model used by TABASCAL is modular and configurable directly in the `model` section of the configuration file. An example of this section is shown below.

```yaml
model:
  name: Custom
  components:
    # RFI Phase
    - trajectory:KeplerOrbit
    - trajectory:PhaseCalculationRFI
    # RFI Signal
    - rfi_signal:ComplexRFIVarAnt
    # RFI Visibilty
    - rfi_vis:RiemannVis
    # Astronomical Visibility
    - ast_vis:GPVisAst
    # Gains
    - gains:UnitaryGains
```

The components should be given in order of dependency. For example, `trajectory:KeplerOrbit` is specified before `trajectory:PhaseCalculationRFI` because the later depends on the output of the former. Each component is a class which defines the parameters (if any), their initialisation, their prior distribution, and its own forward model for the component. The component modules are located in [`tabascal/components/`](https://github.com/epfl-radio-astro/tabascal/tree/main/tabascal/components). The model component given in the configuration file should use the module name and then the class name. For example the component `trajectory:KeplerOrbit` is a class named {class}`~tabascal.components.trajectory.KeplerOrbit` that resides in the module file [`tabascal/components/trajectory.py`](https://github.com/epfl-radio-astro/tabascal/blob/main/tabascal/components/trajectory.py)

### Precision

TABASCAL runs in **single precision (fp32) by default**. It is set in the same
`model` section:

```yaml
model:
  precision: single   # default; or "double" for fp64
```

* `single` (default). Halves device-memory use (~2×), which raises the largest
  problem size that fits on a GPU. On GPUs with first-class fp64 (e.g.
  Hopper/GH200) it is **not** faster in wall-clock — the win is memory capacity,
  not speed.
* `double`. Required by some components, and recommended when fitting satellite
  trajectories, as the differentiable orbit models need fp64 accuracy.

The following components run in **double precision only** and raise a clear
error under `single`, so set `model.precision: double` to use them:

* `trajectory:PhaseCalculationRFI`
* `trajectory:NoDragOrbit`
* `trajectory:Orbit`

Both `rfi_vis` kernels (`RiemannVis` and the FFI `RiemannVisFFI`, see [RFI-visibility
kernels](kernels.md)) and the GP astronomical and gains components run in either
precision.

## Data

The `data` section of the configuration file includes only a few options to select the data to use. An exhaustive example is given below.

```yaml
data:
  sim_dir:
  ms_path: path/to/ms_file.ms
  data_col: DATA
  freq: 0
  corr: xx
  noise:
```

* `sim_dir`: Simulation directory created when using `sim-vis` to simulate a dataset. This can also be given at runtime of `tabascal` with the `-s` flag.
* `ms_path`: Path to the Measurement Set (MS) to run on. This can also be given at runtime of `tabascal` with the `-ms` flag.
* `data_col`: The data column within the MS file to use as the observed data. Default is `DATA` but can be any column that exists in the MS file.
* `freq`: This is the frequency channel to run on. Default is to run on all frequency channels.
* `corr`: This is the correlation product to run on, default `xx`. It is matched against the MS's `POLARIZATION::CORR_TYPE` **by identity, not by position**, so it names the correlation you want rather than an axis index: `yy` selects YY whether the MS holds all four correlations or only that one. Linear (`xx`, `xy`, `yx`, `yy`), circular (`rr`, `rl`, `lr`, `ll`) and Stokes (`i`, `q`, `u`, `v`) names are accepted. Requesting a correlation the MS does not hold is an error naming what it does hold.
* `noise`: The per-visibility noise in Jy. **Leave it null to use the MS's own noise columns**, which are read *per baseline*, and *per channel* where the MS resolves them that far, rather than averaged to one number: the antennas of a real array differ in sensitivity and a bandpass is not flat, so a single value mis-weights every visibility. On EDA2 the per-baseline `SIGMA` spans a factor of ~30, so a scalar under-weights the quietest baselines by up to ~200x. It matters most when fitting gains, because the per-antenna noise correlates with the per-antenna gain (measured `sigma_a ~ amplitude_a^0.76`, R = 0.96) — a uniform-noise likelihood cannot tell a loud antenna from a noisy one, so the fitted gain absorbs the noise structure.

  With `noise: null` the noise is resolved from the MS, most specific column first:

  | column | shape read | constant in time | varying in time |
  |---|---|---|---|
  | `SIGMA_SPECTRUM` | `(row, chan, corr)` | `(n_bl, n_freq)` — per baseline **and** channel | `(n_bl, n_freq, n_time)` |
  | `SIGMA` | `(row, corr)` | `(n_bl,)` — per baseline | `(n_bl, 1, n_time)` — per baseline **and** timestep |

  **The time axis is kept only when it carries something.** Each column is read cell by cell — per baseline for `SIGMA`, per (baseline, channel) for `SIGMA_SPECTRUM` — and a cell whose rows are all bit-identical is one measurement written into every row: it collapses to that value, so an MS with a uniform `SIGMA` weights exactly as it always did. If any cell's rows differ, the column is saying the noise changed over the observation, and the whole column is kept time-resolved. There is no tolerance and no threshold: an MS that writes a constant noise writes the identical value in every row, and a column that varies is taken at **face value**, because nothing here can tell a corrupted row from a timestep on which the noise really was different — both are a positive, finite number the MS wrote down. One odd-but-finite row is therefore enough to make a column vary, and it is kept rather than median-ed away. Validity masking applies either way: a non-positive or non-finite entry is never a measurement. The read prints one line when it finds time variation, saying the noise is being kept time-resolved. A time-resolved `SIGMA` is `(n_bl, 1, n_time)` and never `(n_bl, n_time)` — whenever the observation has as many channels as timesteps, nothing in that shape says which axis is which, and every consumer would weight the visibilities by the wrong one in silence.

  Where a column does collapse, it collapses with a median per cell, over the rows that carry a value — non-positive and non-finite rows are dropped before the median, not after it, so a few invalid rows cannot cost a baseline the rows that measured it properly. Cells left with no valid value at all — dead baselines, flagged channels — take the median of the valid cells rather than a zero that would divide the likelihood by nothing, with a warning saying how many. In the time-resolved case the same fill runs one axis further out: an entry that measured nothing takes its own cell's median over the timesteps that did, and only a cell that measured nothing at all falls back on the median over the cells that did. Such cells are normally flagged, and a flagged cell is dropped from the likelihood whatever noise it was given; the fill itself does not read the flags, so an *unflagged* cell with no valid noise is included at the median. Take the warning as saying which cells to check rather than as a guarantee they carry no weight. `SIGMA_SPECTRUM` is optional in the MS format: if it is absent, unreadable, covers a different set of channels from the ones being read, or holds no positive finite value anywhere — a column that was never filled in — the read falls through to `SIGMA` with a warning. If neither column is usable nothing is invented — a noise scale invented on the spot would silently re-weight the entire fit — so the read warns and leaves the noise unset. Setting `noise` is then what supplies it, and is how such an MS is run; left null, the run stops with an error naming this option.

  A scalar may still be given, applying to every visibility. `0` is rejected rather than treated as "no override" — zero noise divides the likelihood by nothing, so whichever was meant, say it. `true`/`false` are rejected too: YAML booleans reach `float()` as 1.0 and 0.0, and a run on a uniform 1 Jy noise nobody asked for is plausible enough to go unnoticed. A path to an `.npz` is also accepted, for a noise measured out of band; it must carry one of, most specific first:

  | key | shape | meaning |
  |---|---|---|
  | `sigma_bl_freq` | `(n_bl, n_freq)` | per-baseline, per-channel noise, used as given |
  | `sigma_bl` | `(n_bl,)` | per-baseline noise, used as given |
  | `s_ant` | `(n_ant,)` | per-antenna noise, combined as `sqrt(s_p^2 + s_q^2) / sqrt(2)` |

  `sigma_bl` and `s_ant` must be one-dimensional, `s_ant` must cover every antenna the observation's baselines are formed from, and the values read must be real, positive and finite: every entry of `sigma_bl_freq` or `sigma_bl`, and, for `s_ant`, every antenna this observation actually correlates. A file of per-antenna noise may legitimately cover a whole array, and an entry for an antenna these baselines never use cannot mis-weight anything, so those entries are deliberately not policed. A complex array is rejected rather than read as its real part, and so are boolean and string arrays — only an integer or floating-point array is read as a noise, since `astype(float)` would otherwise turn a file of flags into a uniform 1 Jy and parse a file of text into whatever the strings spell. An override is used exactly as given, so it is *not* repaired the way an MS column is: a file carrying an entry that is not a noise is rejected, naming the key and how many entries offend, rather than having part of the user's own answer filled in for them. The median fill above applies to the MS columns only. No override is time-resolved — there is no key for a time axis, and a scalar, per-baseline or per-(baseline, channel) override applies to every timestep — so an MS whose noise genuinely varies over the observation is best read from its own columns.

  The values are checked again after conversion to the precision the run works in (`model.precision`). Under single precision a value like `1e-50` underflows to zero — which would divide the likelihood by nothing — and `1e40` overflows to infinity, so a file that is valid as written but not at the run's precision is an error naming the dtype, not a silent re-weighting.

## Plots

The `plots` section defines which plots of predictions to create. These produced plots are saved in the `plots/` directory alongside the MS file TABASCAL was run on. These are defined as follows

```yaml
plots:
  init: True
  truth: False
  prior: True
  prior_samples: 100
```

The above configuration will plot the predictions for the initial parameters and the prior distributions. To plot the prediction from the prior, 100 samples will be drawn from the prior and then pushed through the forward model to get a predicition. The `truth` plots can only be included if TABASCAL is being run on a simulation created with `sim-vis`.

## Inference

The `inference` section defines the type of inference that will be done. An example is given below.

```yaml
inference:
  opt: True
  mcmc: False
  fisher: False
```

* `opt`: Optimisation will be done to find the maximum a posteriori (MAP) point.
* `mcmc`: Markov Chain Monte Carlo (MCMC) will be run to draw samples from the posterior after some number of warmup iterations. If `opt` is `True` then the MCMC chains will be initialised from the MAP point, otherwise initialisation will be done according to the definitions in the appropriate sections.
* `fisher`: A Laplace approximation of the covariance will be performed about the MAP point.

## Optimisation

The `opt` section gives the parameters for the optimiser. An example is given below.

```yaml
opt:
  epsilon: 1e-1
  max_iter: 10000
  dual_run: True
  trace_path: null
```

* `epsilon`: This is the optimiser step size
* `max_iter`: This is the number of iterations to run the optimiser for in each optimisation run.
* `dual_run`: This determines whether the optimiser will be run a second time starting from where the previous run left off but with an `epsilon` that is 10x smaller.
* `trace_path`: Path of an `.npz` file to record the optimiser's convergence to, or `null` (the default) to record nothing. See [Optimiser trace](#optimiser-trace).

### Optimiser trace

Two models cannot be compared on their loss curves. The loss is a negative log *joint*, so its prior term scales with the latent dimension of whichever parameterisation is running — a Fourier basis with 123 k-modes and 76 inducing times for the same Gaussian process do not put their losses on a common scale. And loss *per iteration* hides the cost of an iteration, so a model that converges in fewer but more expensive steps looks better than it is.

Setting `trace_path` records, once per optimiser iteration, the quantities that can be compared: the wall-clock time the iteration finished, and the metrics fixed by the data rather than by the parameterisation. The metrics are read out of the same forward pass as the loss, so they cost a few elementwise reductions rather than a second evaluation of the model, and the run is otherwise unchanged. With `trace_path` left `null` nothing is recorded and the optimiser takes the same compiled path it always did.

Set the `TAB_LOSS_TRACE` environment variable to override `trace_path` for a single run, for tracing a config that should not be edited. Under multiple processes every process traces — they all evaluate the same model and must run the same program — and one file is written, by the first process.

The file is written once, at the end of the run, and holds one array per key, each of length `max_iter` (or `2 * max_iter` with `dual_run`) with one entry per iteration:

| Key | Present | Meaning |
|---|---|---|
| `loss` | always | The optimiser's own loss, the negative log joint divided by `vis_obs.size` |
| `time_s` | always | Seconds from the start of the first iteration to the end of this one, measured after the device sync that reads the loss — so it bounds work completed, not work dispatched |
| `chi2` | always | Reduced chi-squared of the observed visibilities, flag-masked and weighted by the resolved noise, exactly as the value printed at init and opt |
| `vis_ast_nrmse` | with truth | RMSE of the recovered astronomical visibilities against the simulation truth, over the representative noise |
| `vis_rfi_nrmse` | with truth | RMSE of the recovered RFI visibilities against the simulation truth, over the representative noise |

The `nrmse` keys need a truth to score against, so they appear only on a dataset simulated with `sim-vis`. Every entry is recorded at the parameters that produced that iteration's gradient, i.e. *before* its update — so the first entry is the value at the initialisation, and the last is one update behind the reported optimum.

## Fisher

This section gives the parameters for the Laplace approximation. An example is given below.

```yaml
fisher:
  n_samples: 1
  max_cg_iter: 10_000
```

The covariance approximation is not performed in the traditional way of evaluating the negastive inverse Hessian. Rather, the Gaussian approximation of the posterior is sampled around the MAP point. The inverse of the posterior covariance is implicitly defined and then applied to samples from $\mathcal{N}(\boldsymbol{0}, \boldsymbol{\Sigma}^{-1})$. Therefore, the number of samples is defined in `n_samples` and when applying the inverse covariance to the samples the conjugate gradient method is used when `max_cg_iter` defines the number of iteration used in the conjugate gradient method. Increasing both of these values leads to a greater computational load but also improves the accuracy of the resulting posterior samples.

## Astronomical Signal

The `ast` section defines the prior distrbution, intialisation and forward model parameters for the astronomical signal. An example is given below.

```yaml
ast:
  init: prior
  mean: 0
  freq_pad_factor: 2.0
  time_pad_factor: 2.0
  pow_spec:
    p0: 3e3
    k0_freq: 1
    fov_deg: 5
    gammas: [5, 5]
    cutoff: 1e-6
```

* `init`: This gives the initialisation of the parameters. In the sample above `prior` is given so then the parameters will be initialised with the mean of the prior distribution. Other options include `est` to estimate the best initialisation, `sample` to draw a sample from the prior distribution, and `truth` to initialise at the true values. `truth` is only possible when running on a dataset simulated with `sim-vis`.
* `mean`: This is the mean value of the prior distribution.
* `freq_pad_factor`: This defines the size of the padding used when modelling the signal in the Fourier domain. The signal is modelled in the Fourier domain where periodicity is assumed on some interval. If `freq_pad_factor: 1.0` is given then the interval is the interval of the data itself and will lead to periodic solutions.
* `time_pad_factor`: This defines the padding used in the time axis of the signal. It is the time axis equivalent to `freq_pad_factor`.
* `pow_spec`: This is the section that defines the prior covariance of the signal. The signal is modelled in the Fourier domain so the prior covariance is given by the power spectrum of the signal.

The parameters for the power spectrum are defined as

* `p0`: Mean power of the signal.
* `k0_freq`: Inverse correlation scale along the frequency axis.
* `fov_deg`: The field of view in degrees used to set the maximum astronomical fringe rate (the knee `k0` of the time-axis power spectrum). It is the *full* field of view, i.e. the angular diameter out to the first null of the primary beam; the maximum source offset from the phase centre is `fov_deg / 2`. When omitted, it defaults to the primary-beam field of view of the telescope, `2 * 1.22 * lambda / D` (null-to-null), from the dish diameter `D` and frequency read from the MS file.
* `gammas`: The rate of drop off in the power spectrum. As $\gamma \rightarrow \infty$, the power spectrum tends to a Gaussian with width given by `k0_freq` in the frequency axis and inferred from `fov_deg` in the time axis.
* `cutoff`: This is the relative cutoff for Fourier components. The power spectrum is calculated and then Fourier components, where the power spectrum value is less than `p0 * cutoff`, are removed and not modelled. This reduces the number of parameters to fit.


## RFI signal

The `rfi` section defines the prior distribution over the RFI signal. An example of this section is given below.

```yaml
rfi:
  init: sample
  mean: 0
  min_elevation: 0
  freq_pad_factor: 2.0
  time_pad_factor: 2.0
  freq_int_samples: 1
  time_int_factor: 1
  pow_spec:
    p0: 1e3
    k0s: [1e0, 1e-2]
    gammas: [5, 5]
    cutoff: 1e-6
```

All parameters in this section that overlap with those of the `ast` section have the same definition. The only additional parameters are

* `min_elevation`: Elevation in degrees below which a satellite's RFI signal is held at zero, so it is only modelled while it is up. The default is `0`, which masks a satellite exactly while it is below the geometric horizon. Set it to `null` to disable masking entirely and model every satellite over the whole observation.

  While a satellite is below the horizon it contributes no signal, but an unmasked model still carries a full set of free parameters for it over those times. Those parameters have no signal of their own to constrain them, so they are free to absorb signal that belongs elsewhere — the astronomical sky, or another RFI source — to the extent that the RFI signal prior admits it and the fringe rates overlap. Masking removes the parameters rather than relying on the fit to leave them alone. This is why `0` rather than `null` is the default: a satellite below the horizon is not a modelling choice, it is simply not there.

  Each satellite gets its own in-view window, evaluated on the observation time grid and expanded over each integration, so an integration is never partially masked. Setup fails if a satellite is never above the cut, since it would then be modelled nowhere.

  Raising the cut above `0` additionally excludes the low-elevation part of each pass, where the fringe rate is lowest and the overlap with other components is therefore greatest. How far to raise it is observation-dependent and is not currently calibrated, so no value above `0` is recommended here. Note that masking is about which parameters exist, not about subtraction quality, and reduced $\chi^2$ is largely insensitive to it — judge the effect on the recovered sky model.

* `freq_int_samples`: This is the amount of over-sampling in the frequency domain that is used and then averaged back down to the data sampling rate. It therefore determines the number of samples per frequency channel that are used in the averaging to correctly calculate the fringe-winding loss (band-smearing). Band-smearing can be caused by both the phase variation over the channel width due to the geometric phase as well as the intrinsic signal of the RFI sources.
* `time_int_factor`: In the time axis the number of integration samples needed to accurately model fringe-winding loss (time-smearing) is calculated based solely on the fringe rate due to the movement of the RFI source as well as the signal to noise ratio with

$$N^T_\text{int} \geq  \pi \nu_F \Delta t \sqrt{\frac{\lvert V^\text{RFI}_\text{inst} \rvert}{6 \sigma_n}}$$

where $N^T_\text{int}$ is the number of integration samples used per time step, $\Delta t$ is the integration time for a single sample, $\nu_F$ is the fringe frequency of the source due to its movement, $\lvert V^\text{RFI}_\text{inst} \rvert$ is the instantaneous RFI visibility amplitude, and $\sigma_n$ is the visibility noise of a single data point. This parameter (`time_int_factor`) determines the factor by which to increase this oversampling.

### RFI light curve estimates

`rfi.est` points at a measured light curve file, used by `init: est` and `mean: est` to seed the RFI signal. This is the interchange format between tabascal and whatever measures the light curves, so it is deliberately strict.

The file is either a **`.zarr` store** (read with `xarray.open_zarr`) or a **`.npz`**, and must contain all four of

| name | shape | contents |
|---|---|---|
| `light_curves` | `(n_src, n_time, n_freq)` | apparent flux per source, in **Jy** |
| `norad_ids` | `(n_src,)` | NORAD id of each row of `light_curves` |
| `times` | `(n_time,)` | Modified Julian Date, in **days**, strictly increasing |
| `freqs` | `(n_freq,)` | frequency in **Hz**, strictly increasing |

In the zarr form the last three are coordinates of `light_curves`, whose dimensions must be exactly `norad_ids`, `times` and `freqs` — declared in any order, since they are identified by name and transposed on read. A minimal writer:

```python
import numpy as np, xarray as xr

xr.Dataset(
    {"light_curves": (("norad_ids", "times", "freqs"), curves)},
    coords={"norad_ids": np.array([25544, 27386]), "times": times_mjd, "freqs": freqs_hz},
).to_zarr("light_curves.zarr")
```

**Rows are matched to satellites by NORAD id, never by position**, so the order of sources in the file does not have to match `satellites.norad_ids`. **Samples are interpolated onto the observation's own time and frequency grid**, so the file's sampling does not have to match the observation either.

Both are strict because their failure modes are silent. A light curve attached to the wrong satellite still has the right shape and still optimises — it just seeds the prior from another satellite. A file whose sampling is assumed rather than declared is resampled wrongly by an unknown amount. Neither surfaces as an error, only as a worse fit, so a file that cannot state which satellite and which sample times it describes is rejected rather than guessed at.

Times are absolute (MJD) rather than seconds from the start of a particular observation, so a light curve is interpretable on its own and can be reused across measurement sets covering the same pass.

`light_curves` is a **flux in Jy**, not the modelled amplitude `rfi_A`. The RFI visibility is quadratic in `rfi_A` ($V^\text{RFI}_{pq} = A_p A_q^* e^{i\Delta\phi}$), so `rfi_A` carries units of $\sqrt{\text{Jy}}$ and the estimate is seeded with $\sqrt{\lvert \text{light\_curves} \rvert}$. Supplying an amplitude where a flux is expected is squared away silently, so the value is wrong rather than the shape — give the flux the source would show in the visibilities, on the same scale as `rfi.var`.

Some further details:

* Each satellite must appear **exactly once**. A repeated NORAD id has no single answer to which row belongs to it, and resolving that by file order is the thing id-matching exists to avoid, so it is rejected — merge the passes or drop one before using the file as an estimate.
* Labels that are not integer NORAD ids never match a satellite and are dropped, so a file may carry named sources (e.g. `Fornax A`) alongside the satellites without filtering beforehand, and those may repeat freely.
* **Samples outside the file's coverage are zero**, on either axis — the file says nothing there, which is the same "no signal known" convention the elevation mask uses. An axis of length 1 is held constant instead, since a single sample carries no gradient to interpolate along; a single-frequency light curve therefore applies across the whole band rather than being zeroed outside it.
* **The file does not have to cover every satellite in the fit.** Satellites with no light curve are initialised at zero and named in a warning, so light curves can be measured for a subset — the bright or well-characterised sources — while the rest are still modelled and fitted, just without an informative starting point. It is an error only if *no* configured satellite is found, which would otherwise silently reduce the whole estimate to zeros.

## Satellites

The `satellites` section determines which satelites to include in the model and the prior distribution to use for there trajectories. An example is given below.

```yaml
satellites:
  norad_ids: [20452, 38833, 45854]
  norad_ids_path: null
  extra_orbit_dir: null
  extra_orbit_max_age_days: null
  remote_max_age_days: 3
  cache_reuse_max_age_days: 1
  ric_std: 1e2
```

* `norad_ids`: List of the NORAD IDs of the satellites to include. TABASCAL requests the record whose epoch is closest to the observation from the [IAU CPS SatChecker](https://satchecker.cps.iau.org/) service (via the [satchecker-client](https://satchecker-client.readthedocs.io/) package) — its `get-nearest-omm` endpoint for observations from 2026-07-12 onwards and `get-nearest-tle` before that, falling back to the other archive if the first has nothing acceptable. Cache misses run concurrently with a bounded five-worker pool; no account or credentials are required. **Every ID listed here must resolve to an acceptable record**: otherwise preflight stops before reading the visibilities and names each failure. TABASCAL never silently drops a configured satellite from the RFI model.

  An empty list (or `null`) is valid only for a model that does not use a satellite trajectory component — a stationary-RFI or astronomical-only run. If the `model.components` list includes one that consumes orbital records (`FixedOrbit`, `KeplerOrbit`, `Orbit`, `NoDragOrbit`), configuring no IDs is a configuration error rather than a run that models nothing. Either separator form of a component reference is recognised, so `trajectory:FixedOrbit` and `trajectory.FixedOrbit` behave identically here.
* `norad_ids_path`: Optional path to a text file of NORAD IDs, one per line; blank lines and `#` comments are ignored and malformed lines are reported with their line number. When set it takes precedence over `norad_ids`, and the `-np/--norad-path` CLI flag takes precedence over both.
* `extra_orbit_dir`: Optional path to an additional directory of local orbit files, searched **per NORAD ID before** the managed cache and SatChecker. Every `*.json` file in the directory is considered; files must be pandas-oriented JSON tables carrying either `NORAD_CAT_ID`, `TLE_LINE1` and `TLE_LINE2`, or `NORAD_CAT_ID`, `EPOCH` and the seven OMM element columns. The kind is inferred, so a Space-Track `gp`/`gp_history` export drops in unconverted. For each requested satellite the valid record whose epoch is closest to the observation is chosen — by epoch distance, regardless of format — and, if it is accepted (see `extra_orbit_max_age_days`), it wins outright and no service call is made for that satellite. Files that cannot be read or lack either required column set are skipped. Records that fail validation are rejected, allowing that satellite to fall through to the managed cache and SatChecker. Legacy date-named files and the bundled Space-Track fixtures remain supported. This directory can be given at runtime with the `--extra-orbit-dir` flag. (The `ORBIT_CACHE_DIR` environment variable is a different thing: it relocates where the *managed cache* is stored, and is not an additional source of records.)
* `extra_orbit_max_age_days`: Maximum allowed absolute difference, in days, between an `extra_orbit_dir` record's epoch and the observation epoch. `null` (default) applies no age limit, preserving exact replay of `used_orbits_*.json`; `0` accepts an epoch match within TLE precision. A rejected local record falls through to the managed cache and SatChecker. The age comes from the record itself — line 1 for a TLE, the `EPOCH` field for an OMM — not from the filename or modification time.
* `remote_max_age_days`: Hard ceiling, in days, on how far a SatChecker or managed-cache record may be from the observation. A TLE's epoch is re-derived locally from line 1; an OMM has no lines, so its `EPOCH` field is used after being range-checked. Every accepted remote record's source, provider, endpoint, signed offset and absolute age is logged. `null` explicitly removes the ceiling.

  This ceiling is also what makes the endpoint fallback work. Neither SatChecker endpoint reports that it has nothing near the epoch requested — `get-nearest-omm` answers a pre-2026 request with its earliest record — so an over-age response is the signal that the record belongs to the other archive, and the other endpoint is then asked.

  **The default of `3` is provisional.** It is a hard backstop against obviously unsuitable remote records — for one observation, SatChecker's per-satellite fallback silently returned records ~31 days old, worth ~9,663 km of ISS position error — and *not* a claim that a three-day-old element set gives adequate positional accuracy. The calibrated, observation-specific suitability policy that should replace it is tracked in [issue #101](https://github.com/epfl-radio-astro/tabascal/issues/101); it may end up rejecting records younger than three days for some orbits and baselines, or accepting older ones where independently justified.
* `cache_reuse_max_age_days`: Request-avoidance threshold for the per-NORAD cache (default `1`). A cached record this close to the observation avoids a request. An older cached record triggers an exact-epoch nearest lookup — including against the fallback archive, since holding a stale record is not the same as the archive having answered — but remains an offline fallback if it is within `remote_max_age_days`. A response replaces it only when strictly closer to the observation. `null` always reuses the nearest acceptable cached record. When both limits are set, this value must not exceed the hard ceiling.
* `ric_std`: The error in the orbital elements is not provided as part of the element sets. When estimated positions are analysed the error is calculated in a local reference frame of the satellite. This is the radial, in-track, and cross-track (RIC) frame. This parameter gives a factor by which to scale the RIC covariance that is stored internally which is taken from a paper where the average errors are calculated.

## Gains
