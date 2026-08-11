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
    - rfi_signal:FourierGPRFI
    # RFI Visibilty
    - rfi_vis:RiemannVisTimeFreqCalculation
    # Astronomical Visibility
    - ast_vis:FourierTimeFreqGPAst
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
  freq: 0
  corr: xx
  noise: 
```

* `sim_dir`: Simulation directory created when using `sim-vis` to simulate a dataset. This can also be given at runtime of `tabascal` with the `-s` flag.
* `ms_path`: Path to the Measurement Set (MS) to run on. This can also be given at runtime of `tabascal` with the `-ms` flag.
* `data_col`: The data column within the MS file to use as the observed data. Default is `DATA` but can be any column that exists in the MS file.
* `freq`: This is the frequency channel to run on. Default is to run on all frequency channels.
* `corr`: This is the correlation product to run on. Default is `xx` or the first correlation product along the `corr` axis in the MS file.
* `noise`: This is the per visibility data point noise in Jy. It is assumed that the data is independent and identitically distributed with Gaussian noise.

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
```

* `epsilon`: This is the optimiser step size
* `max_iter`: This is the number of iterations to run the optimiser for in each optimisation run.
* `dual_run`: This determines whether the optimiser will be run a second time starting from where the previous run left off but with an `epsilon` that is 10x smaller.

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
* `fov_deg`: The field of view in degrees of the acceptable fringe rate range. The default value for this is calculated from the expected field of view of the telescope based on the dish diameter and frequency as read form the MS file.
* `gammas`: The rate of drop off in the power spectrum. As $\gamma \rightarrow \infty$, the power spectrum tends to a Gaussian with width given by `k0_freq` in the frequency axis and inferred from `fov_deg` in the itme axis.
* `cutoff`: This is the relative cutoff for Fourier components. The power spectrum is calculated and then Fourier components, where the power spectrum value is less than `p0 * cutoff`, are removed and not modelled. This reduces the number of parameters to fit.   


## RFI signal

The `rfi` section defines the prior distribution over the RFI signal. An example of this section is given below.

```yaml
rfi:
  init: sample
  mean: 0
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

* `freq_int_samples`: This is the amount of over-sampling in the frequency domain that is used and then averaged back down to the data sampling rate. It therefore determines the number of samples per frequency channel that are used in the averaging to correctly calculate the fringe-winding loss (band-smearing). Band-smearing can be caused by both the phase variation over the channel width due to the geometric phase as well as the intrinsic signal of the RFI sources.
* `time_int_factor`: In the time axis the number of integration samples needed to accurately model fringe-winding loss (time-smearing) is calculated based solely on the fringe rate due to the movement of the RFI source as well as the signal to noise ratio with
  
$$N^T_\text{int} \geq  \pi \nu_F \Delta t \sqrt{\frac{\lvert V^\text{RFI}_\text{inst} \rvert}{6 \sigma_n}}$$

where $N^T_\text{int}$ is the number of integration samples used per time step, $\Delta t$ is the integration time for a single sample, $\nu_F$ is the fringe frequency of the source due to its movement, $\lvert V^\text{RFI}_\text{inst} \rvert$ is the instantaneous RFI visibility amplitude, and $\sigma_n$ is the visibility noise of a single data point. This parameter (`time_int_factor`) determines the factor by which to increase this oversampling. 

## Satellites

The `satellites` section determines which satelites to include in the model and the prior distribution to use for there trajectories. An example is given below.

```yaml
satellites:
  norad_ids: [20452, 38833, 45854]
  extra_tle_dir: null
  extra_tle_max_age_days: null
  tle_catalogue_interval_hours: 2
  ric_std: 1e2
```

* `norad_ids`: List of the NORAD IDs of the satellites to include. Their orbital elements (TLEs) are fetched automatically from the [IAU CPS SatChecker](https://satchecker.cps.iau.org/) service — no account or credentials are required — and cached locally for reuse. If the service reports no bulk catalogue at the observation epoch (its catalogue has a data horizon, so very recent observations can fall beyond it), each satellite is fetched individually instead; the retrieved TLE may then be somewhat older than the observation, and its epoch is reported in the run log. See [Two-Line Elements (TLEs)](tles.md) for the full caching behaviour, how to reproduce a run's TLEs exactly, and how to supply TLEs manually when the service cannot.
* `extra_tle_dir`: Optional path to an additional directory of local TLE files, searched **per NORAD ID before** the managed cache and SatChecker. Every `*.json` file in the directory is considered; files must be pandas-oriented JSON tables with `NORAD_CAT_ID`, `TLE_LINE1`, and `TLE_LINE2` columns. For each requested satellite the valid record whose TLE epoch is closest to the observation is chosen and, if it is accepted (see `extra_tle_max_age_days`), it wins outright and no service call is made for that satellite. Files that cannot be read or lack the required columns are skipped. Records with malformed TLE lines or a NORAD ID that does not match the ID encoded in both TLE lines are rejected, allowing that satellite to fall through to the managed cache and SatChecker. Legacy date-named files and the bundled Space-Track fixtures remain supported. The managed cache location can also be overridden with the `TLE_CACHE_DIR` environment variable, and this directory can be given at runtime with the `--extra-tle-dir` flag.
* `extra_tle_max_age_days`: Maximum allowed absolute difference, in days, between an `extra_tle_dir` TLE's epoch and the observation epoch. `null` (default) applies no age limit — preserving the previous behaviour and the bundled fixtures. `0` accepts only a record at the exact observation epoch — "exact" meaning within one TLE epoch quantum (~2.6 ms, the ~0.9 ms resolution of a TLE line-1 epoch plus floating-point conversion slack), so a record whose epoch matches to TLE precision is still accepted. A record that is too old (or too far in the future) is rejected and that satellite falls through to the managed catalogue / SatChecker. Negative values are a configuration error. The limit applies only to `extra_tle_dir`, and the age is measured from the TLE line-1 epoch (not the filename or file modification time).
* `tle_catalogue_interval_hours`: Width, in hours, of the fixed UTC bucket used to reuse a managed catalogue snapshot (default `2`; minimum one second, or `1/3600` hour). Buckets are globally anchored at the Unix epoch and a request is served by the SatChecker catalogue nearest each bucket's midpoint, so the reused snapshot depends only on the request and this width — never on what is already cached. **This is a deliberate approximation**: with the default two-hour bucket the catalogue epoch differs from the exact observation epoch by at most one hour, so a returned record is nearest to the canonical bucket epoch, *not* necessarily nearest to the exact observation. Two observations less than one bucket-width apart can still straddle a boundary and use different snapshots.
* `ric_std`: The error in the orbital elements are not provided as part of the TLEs. When TLE estimated positions are analysed the error is calculated in a local reference frame of the satellite. This is the radial, in-track, and cross-track (RIC) frame. This parameter gives a factor by which to scale the RIC covariance that is stored internally which is taken from a paper where the average errors are calculated.  

## Gains
