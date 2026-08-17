# Configuration File

TABASCAL makes use of a configuration file to fully define the model used, what satellites to include and what inference to do. The configuration file is in YAML format using only the most basic constructs and types. The configuration file is broken up into multiple sections. Each section will be described below.

## Validation

Every parameter is declared in the code that reads it — the model components in
their own `config_params`, and the parameters read outside any component (`model`,
`data`, `plots`, `inference`, `opt`, `satellites`, and the RFI sampling grid) on
{class}`~tabascal.config.TabConfig`. Those declarations are the schema: there is
no separate list of key names to fall out of sync with them, and the defaults
below come from them rather than from a packaged base config.

The file is checked against the declarations at the start of every run, before
the Measurement Set is read or any orbital records are fetched. Every problem is
reported together, so a config with four mistakes takes one run to fix:

```text
Error: invalid configuration in tab_target.yaml: 3 problems found

  ast.pow_spec.p0: required, but not set (mean power of the astronomical signal)
  gains.corr_time: unknown key (did you mean 'gains.amp_corr_time'?)
  opt.max_iter: expected an integer >= 0, got 'many'
```

Because only the components you selected contribute, **an unrecognised key is an
error**, not something quietly ignored — including one that belongs to a
component you have not put in `model.components`, which is reported as such.
Anything you leave out takes the default given below, and writing a key with no
value (`corr:` with nothing after it) is the same as leaving it out. The
exceptions are the few parameters where `null` means "off" rather than "unset",
noted where they appear.

Run `tabascal check-config -c your_config.yaml` to check a file and print the
resolved configuration, defaults included, without running anything.

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

## Data

The `data` section of the configuration file includes only a few options to select the data to use. An exhaustive example is given below.

```yaml
data:
  sim_dir:
  ms_path: path/to/ms_file.ms
  data_col: DATA
  freq:
  corr: xx
  noise:
  flags: False
```

* `sim_dir`: Simulation directory created when using `sim-vis` to simulate a dataset. This can also be given at runtime of `tabascal` with the `-s` flag.
* `ms_path`: Path to the Measurement Set (MS) to run on. This can also be given at runtime of `tabascal` with the `-ms` flag.
* `data_col`: The data column within the MS file to use as the observed data. Default is `DATA` but can be any column that exists in the MS file.
* `freq`: A frequency in Hz. The single nearest channel to it is modelled; leaving it unset (the default) models every frequency channel.
* `corr`: This is the correlation product to run on, default `xx`. It is matched against the MS's `POLARIZATION::CORR_TYPE` **by identity, not by position**, so it names the correlation you want rather than an axis index: `yy` selects YY whether the MS holds all four correlations or only that one. Linear (`xx`, `xy`, `yx`, `yy`), circular (`rr`, `rl`, `lr`, `ll`) and Stokes (`i`, `q`, `u`, `v`) names are accepted. Requesting a correlation the MS does not hold is an error naming what it does hold.
* `noise`: This is the per visibility data point noise in Jy. It is assumed that the data is independent and identitically distributed with Gaussian noise. Unset (the default) reads it from the Measurement Set.
* `flags`: Whether to include the Measurement Set's flags in the likelihood. Default `False`.

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
```

* `opt`: Optimisation will be done to find the maximum a posteriori (MAP) point. Default `True`. With `opt: False` (or `opt.max_iter: 0`) TABASCAL writes the prediction from the initial parameters instead of fitting.

MCMC sampling and the Fisher/Laplace covariance approximation are not currently
implemented, so `inference.mcmc`, `inference.fisher` and the `fisher` section are
no longer accepted.

## Optimisation

The `opt` section gives the parameters for the optimiser. An example is given below.

```yaml
opt:
  epsilon: 1e-1
  max_iter: 10000
  dual_run: True
```

* `epsilon`: This is the optimiser step size. Default `1e-2`.
* `max_iter`: This is the number of iterations to run the optimiser for in each optimisation run. Default `500`.
* `dual_run`: This determines whether the optimiser will be run a second time starting from where the previous run left off but with an `epsilon` that is 10x smaller. Default `True`.
* `guide`: The type of optimisation to run. Only `map` is implemented, which is the default.

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

* `init`: This gives the initialisation of the parameters. In the sample above `prior` is given so then the parameters will be initialised with the mean of the prior distribution. Other options are `data` to estimate the initialisation from the observed visibilities, `sample` to draw a sample from the prior distribution (the default), and `truth` to initialise at the true values. `truth` is only possible when running on a dataset simulated with `sim-vis`.
* `mean`: This is the mean value of the prior distribution: `0` (the default, equivalently `zeros`) or `data` to estimate it from the observed visibilities.
* `freq_pad_factor`: This defines the size of the padding used when modelling the signal in the Fourier domain. The signal is modelled in the Fourier domain where periodicity is assumed on some interval. If `freq_pad_factor: 1.0` is given then the interval is the interval of the data itself and will lead to periodic solutions. Default `2`.
* `time_pad_factor`: This defines the padding used in the time axis of the signal. It is the time axis equivalent to `freq_pad_factor`. Default `2`.
* `pow_spec`: This is the section that defines the prior covariance of the signal. The signal is modelled in the Fourier domain so the prior covariance is given by the power spectrum of the signal.

The parameters for the power spectrum are defined as

* `p0`: Mean power of the signal. Required.
* `k0_freq`: Inverse correlation scale along the frequency axis. Required.
* `fov_deg`: The field of view in degrees used to set the maximum astronomical fringe rate (the knee `k0` of the time-axis power spectrum). It is the *full* field of view, i.e. the angular diameter out to the first null of the primary beam; the maximum source offset from the phase centre is `fov_deg / 2`. When omitted, it defaults to the primary-beam field of view of the telescope, `2 * 1.22 * lambda / D` (null-to-null), from the dish diameter `D` and frequency read from the MS file.
* `gammas`: The rate of drop off in the power spectrum, one per axis as `[freq, time]`. As $\gamma \rightarrow \infty$, the power spectrum tends to a Gaussian with width given by `k0_freq` in the frequency axis and inferred from `fov_deg` in the time axis. Required.
* `cutoff`: This is the relative cutoff for Fourier components. The power spectrum is calculated and then Fourier components, where the power spectrum value is less than `p0 * cutoff`, are removed and not modelled. This reduces the number of parameters to fit. Default `1e-6`.


## RFI signal

The `rfi` section defines the prior distribution over the RFI signal. An example of this section is given below.

```yaml
rfi:
  init: sample
  mean: 0
  var:
  corr_freq: 1e6
  corr_time: 24
  min_elevation: 0
  freq_pad_factor: 2.0
  time_pad_factor: 2.0
  freq_int_samples: 1
  time_int_factor: 1
```

The `init`, `mean` and padding parameters have the same meaning as in the `ast`
section, except that `init` also accepts `est` (seed from a measured light curve,
see below), `zeros` and `ones`, and `mean` also accepts `est`. The RFI prior is a
Gaussian process specified by a variance and two correlation lengths rather than
by a power spectrum, so the additional parameters are

* `var`: Variance of the RFI signal in Jy. Unset (the default) estimates it from the data, as the largest observed visibility amplitude.
* `corr_freq`: Correlation bandwidth of the RFI signal in Hz, default `1e6`. Set it to `null` to derive it from the data instead, as half the bandwidth of the observation.
* `corr_time`: Correlation time of the RFI signal in seconds, default `24`. `null` derives it from the data, as half the duration of the observation.
* `r_seed`: Seed for the prior samples drawn by `init: sample`. Default `123`.

* `min_elevation`: Elevation in degrees below which a satellite's RFI signal is held at zero, so it is only modelled while it is up. The default is `0`, which masks a satellite exactly while it is below the geometric horizon. Set it to `null` to disable masking entirely and model every satellite over the whole observation.

  While a satellite is below the horizon it contributes no signal, but an unmasked model still carries a full set of free parameters for it over those times. Those parameters have no signal of their own to constrain them, so they are free to absorb signal that belongs elsewhere — the astronomical sky, or another RFI source — to the extent that the RFI signal prior admits it and the fringe rates overlap. Masking removes the parameters rather than relying on the fit to leave them alone. This is why `0` rather than `null` is the default: a satellite below the horizon is not a modelling choice, it is simply not there.

  Each satellite gets its own in-view window, evaluated on the observation time grid and expanded over each integration, so an integration is never partially masked. Setup fails if a satellite is never above the cut, since it would then be modelled nowhere.

  Raising the cut above `0` additionally excludes the low-elevation part of each pass, where the fringe rate is lowest and the overlap with other components is therefore greatest. How far to raise it is observation-dependent and is not currently calibrated, so no value above `0` is recommended here. Note that masking is about which parameters exist, not about subtraction quality, and reduced $\chi^2$ is largely insensitive to it — judge the effect on the recovered sky model.

* `freq_int_samples`: This is the amount of over-sampling in the frequency domain that is used and then averaged back down to the data sampling rate. It therefore determines the number of samples per frequency channel that are used in the averaging to correctly calculate the fringe-winding loss (band-smearing). Band-smearing can be caused by both the phase variation over the channel width due to the geometric phase as well as the intrinsic signal of the RFI sources.
* `time_int_factor`: In the time axis the number of integration samples needed to accurately model fringe-winding loss (time-smearing) is calculated based solely on the fringe rate due to the movement of the RFI source as well as the signal to noise ratio with

$$N^T_\text{int} \geq  \pi \nu_F \Delta t \sqrt{\frac{\lvert V^\text{RFI}_\text{inst} \rvert}{6 \sigma_n}}$$

where $N^T_\text{int}$ is the number of integration samples used per time step, $\Delta t$ is the integration time for a single sample, $\nu_F$ is the fringe frequency of the source due to its movement, $\lvert V^\text{RFI}_\text{inst} \rvert$ is the instantaneous RFI visibility amplitude, and $\sigma_n$ is the visibility noise of a single data point. This parameter (`time_int_factor`) determines the factor by which to increase this oversampling. Default `1`.

* `min_time_bins`, `max_time_bins`: Bounds on the number of stride groups used when the per-baseline sampling rates above are binned, which only the variable-sampling RFI visibility components (`RiemannVisVariable`, `RiemannVisVariableFFI`) use. Defaults `1` and `30`.

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
```

* `norad_ids`: List of the NORAD IDs of the satellites to include. TABASCAL requests the record whose epoch is closest to the observation from the [IAU CPS SatChecker](https://satchecker.cps.iau.org/) service — its `get-nearest-omm` endpoint for observations from 2026-07-12 onwards and `get-nearest-tle` before that, falling back to the other archive if the first has nothing acceptable. Cache misses run concurrently with a bounded five-worker pool; no account or credentials are required. **Every ID listed here must resolve to an acceptable record**: otherwise preflight stops before reading the visibilities and names each failure. TABASCAL never silently drops a configured satellite from the RFI model.

  An empty list (or `null`) is valid only for a model that does not use a satellite trajectory component — a stationary-RFI or astronomical-only run. If the `model.components` list includes one that consumes orbital records (`FixedOrbit`, `KeplerOrbit`, `Orbit`, `NoDragOrbit`), configuring no IDs is a configuration error rather than a run that models nothing. Either separator form of a component reference is recognised, so `trajectory:FixedOrbit` and `trajectory.FixedOrbit` behave identically here.
* `norad_ids_path`: Optional path to a text file of NORAD IDs, one per line; blank lines and `#` comments are ignored and malformed lines are reported with their line number. When set it takes precedence over `norad_ids`, and the `-np/--norad-path` CLI flag takes precedence over both.
* `extra_orbit_dir`: Optional path to an additional directory of local orbit files, searched **per NORAD ID before** the managed cache and SatChecker. Every `*.json` file in the directory is considered; files must be pandas-oriented JSON tables carrying either `NORAD_CAT_ID`, `TLE_LINE1` and `TLE_LINE2`, or `NORAD_CAT_ID`, `EPOCH` and the seven OMM element columns. The kind is inferred, so a Space-Track `gp`/`gp_history` export drops in unconverted. For each requested satellite the valid record whose epoch is closest to the observation is chosen — by epoch distance, regardless of format — and, if it is accepted (see `extra_orbit_max_age_days`), it wins outright and no service call is made for that satellite. Files that cannot be read or lack either required column set are skipped. Records that fail validation are rejected, allowing that satellite to fall through to the managed cache and SatChecker. Legacy date-named files and the bundled Space-Track fixtures remain supported. This directory can be given at runtime with the `--extra-orbit-dir` flag. (The `ORBIT_CACHE_DIR` environment variable is a different thing: it relocates where the *managed cache* is stored, and is not an additional source of records.)
* `extra_orbit_max_age_days`: Maximum allowed absolute difference, in days, between an `extra_orbit_dir` record's epoch and the observation epoch. `null` (default) applies no age limit, preserving exact replay of `used_orbits_*.json`; `0` accepts an epoch match within TLE precision. A rejected local record falls through to the managed cache and SatChecker. The age comes from the record itself — line 1 for a TLE, the `EPOCH` field for an OMM — not from the filename or modification time.
* `remote_max_age_days`: Hard ceiling, in days, on how far a SatChecker or managed-cache record may be from the observation. A TLE's epoch is re-derived locally from line 1; an OMM has no lines, so its `EPOCH` field is used after being range-checked. Every accepted remote record's source, provider, endpoint, signed offset and absolute age is logged. `null` explicitly removes the ceiling.

  This ceiling is also what makes the endpoint fallback work. Neither SatChecker endpoint reports that it has nothing near the epoch requested — `get-nearest-omm` answers a pre-2026 request with its earliest record — so an over-age response is the signal that the record belongs to the other archive, and the other endpoint is then asked.

  **The default of `3` is provisional.** It is a hard backstop against obviously unsuitable remote records — for one observation, SatChecker's per-satellite fallback silently returned records ~31 days old, worth ~9,663 km of ISS position error — and *not* a claim that a three-day-old element set gives adequate positional accuracy. The calibrated, observation-specific suitability policy that should replace it is tracked in [issue #101](https://github.com/epfl-radio-astro/tabascal/issues/101); it may end up rejecting records younger than three days for some orbits and baselines, or accepting older ones where independently justified.
* `cache_reuse_max_age_days`: Request-avoidance threshold for the per-NORAD cache (default `1`). A cached record this close to the observation avoids a request. An older cached record triggers an exact-epoch nearest lookup — including against the fallback archive, since holding a stale record is not the same as the archive having answered — but remains an offline fallback if it is within `remote_max_age_days`. A response replaces it only when strictly closer to the observation. `null` always reuses the nearest acceptable cached record. When both limits are set, this value must not exceed the hard ceiling.

## Gains

The `gains` section defines the prior distribution over the antenna gains. It is
read by the gain components (`gains:UnitaryGains`, `gains:GPGains`); with
`UnitaryGains` the gains are held at unity and the priors below are not fitted.
An example is given below.

```yaml
gains:
  amp_mean: 1.0
  phase_mean: 0.0
  amp_std: 1.0        # %
  phase_std: 1.0      # degrees
  amp_corr_freq:
  amp_corr_time:
  phase_corr_freq:
  phase_corr_time:
  r_seed: 123
```

The amplitude and phase of the gains are each modelled as a Gaussian process over
frequency and time, so each has a mean, a standard deviation and a correlation
length per axis.

* `amp_mean`: Mean of the prior over the gain amplitudes. Default `1.0`.
* `phase_mean`: Mean of the prior over the gain phases, **in radians**. Default `0.0`.
* `amp_std`: Standard deviation of the prior over the gain amplitudes, **as a percentage** of `amp_mean`. Unset defaults to 1 %.
* `phase_std`: Standard deviation of the prior over the gain phases, **in degrees**. Unset defaults to 1 degree.
* `amp_corr_freq`, `phase_corr_freq`: Correlation bandwidth in Hz. Unset defaults to the bandwidth of the observation, i.e. no variation across the band beyond the mean. A single-channel observation has no bandwidth to measure, so the channel width is used instead.
* `amp_corr_time`, `phase_corr_time`: Correlation time in seconds. Unset defaults to the duration of the observation, with the integration time standing in for a single-integration observation.
* `r_seed`: Seed for the prior samples used to initialise the gains. Default `123`.

`0` is a legitimate value for the means and standard deviations and is honoured
as given — it means no variation about the mean, not "use the default".
