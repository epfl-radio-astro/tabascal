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

The `ast` section defines the prior distribution, initialisation and forward model parameters for the astronomical signal. There are two ways to model the sky, which may be combined:

* a **visibility-domain Gaussian process** that models `vis_ast` directly (used by the `ast_vis:FourierTime*Ast` components), configured by the flat `ast.{init, mean, pow_spec, freq_pad_factor, time_pad_factor}` keys documented immediately below; and
* **sky-domain** point and image models (used by the `ast_signal:*` components together with `ast_vis:PointSourceVisCalculation` / `ast_vis:ImageVisCalculation`), configured by the `ast.grid` and `ast.signals` keys documented below under **Dense and point sky models**.

The visibility-domain GP is configured as follows.

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

### Dense and point sky models

The sky-domain components model the sky as a **point catalogue** or a **dense image** and are configured per component under `ast.signals`, with an optional shared image grid under `ast.grid`. An example is given below.

```yaml
ast:
  grid:                              # shared image grid + wgridder plan (image skies only)
    fov_deg: 5.0
    n_pix: 256
    epsilon: 1e-6
  signals:                          # one entry per sky-signal component, keyed by class name
    FixedPointSky:
      init: {type: from_catalogue, fmt: zarr}
    PointSky:
      init: {type: from_catalogue, fmt: bbs, path: model.bbs}
      start: truth
      prior: {laplace_width: 1.0}
    ImageSky:
      init: {type: zeros}
      prior:
        mean: {type: zeros}
        pow_spec: {p0: 1.0, k0_freq: 1.0, k0_lm: 300.0, gamma_freq: 2.0, gamma_lm: 2.0, cutoff: 1e-6, mu: -2.0}
```

* `grid`: Builds the shared cosine image grid and reusable wgridder plan used by the image-based components (`FixedImageSky`, `ImageSky`, `ImageVisCalculation`). `fov_deg` is the field of view in degrees, `n_pix` the number of pixels per side, and `epsilon` the wgridder accuracy. Omit it (or leave `null`) when no image component is used.
* `signals`: A map keyed by the **component class name**. Each entry configures that component, most importantly its sky **source** under `init` (and, for learnable components, its `prior`).

#### Sky sources

A *source spec* is a small dict with an explicit `type`, resolved by {func}`~tabascal.sky_sources.resolve_sky_source`. The same source can act as a fixed sky, a learnable initialisation, or a learnable prior mean. The supported types are

* `zeros`: an empty sky (a zero image / empty catalogue); useful as a neutral prior mean.
* `from_catalogue`: a point catalogue. `fmt` is `zarr` (a tabsim simulation catalogue) or `bbs` (a WSClean/DP3 component list). `path` is the file; if omitted it falls back to `data.zarr_path`.
* `from_fits`: a FITS image (2-D continuum or a spectral cube interpolated onto the model channels). The spatial grid must match `ast.grid`. A Jy/beam image is converted to Jy/pixel from the header beam (`BMAJ`/`BMIN`/`CDELT`); set `unit` to override the header `BUNIT`. `hdu` selects the HDU (default 0).
* `from_ms`: a MeasurementSet visibility column, rendered to an image as its (adjoint) dirty image. `column` selects the column and is independent of `data.data_col`.

Only Stokes I is supported. A source spec may carry a `stokes` field, but anything other than `[I]` is currently rejected.

#### Per-component keys

* `init`: the source seeding the component. For the fixed components (`FixedPointSky`, `FixedImageSky`) it *is* the sky; for the learnable components it sets the starting point of the parameters.
* `start` (`PointSky` only): the flux-parameter initialisation — `sample` (default), `zeros`, or `truth` (the catalogue flux). The source positions are always taken from `init`.
* `prior.laplace_width` (`PointSky`): the width of the zero-mean Laplace (sparsity) prior on the per-source fluxes.
* `prior.pow_spec` (`ImageSky`): the power-spectrum prior of the log-sky Gaussian random field — `p0`, `k0_freq`, `k0_lm`, `gamma_freq`, `gamma_lm`, `cutoff`, and an optional `mu` (log-sky offset) and `freq_pad_factor`/`lm_pad_factor`.
* `prior.mean` (`ImageSky`): a source spec (rendered to an image and used as the GRF mean) or `{type: zeros}` (the default). This lets an external sky model centre the prior.

Pair each signal component with the matching visibility calculator in the `model.components` list — a point sky with `ast_vis:PointSourceVisCalculation`, an image sky with `ast_vis:ImageVisCalculation`. See the [astronomical signal](components/ast_signal.md) and [astronomical visibility](components/ast_vis.md) component pages for details.


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
  spacetrack_path: spacetrack_login.yaml
  tle_offset: 0
  ric_std: 1e2
```

* `norad_ids`: List of the NORAD IDs of the satellites to include.
* `spacetrack_path`: Path to the Space-Track login details for collecting the orbital elements. This can also be given at runtime of the `tabascal` script with the `-st` flag.
* `tle_offset`: The orbital elements collected from Space-Track are called two-line element sets (TLE). They are updated regularly as the associated model becomes inaccurate far away from the measurement time and satellites also perform orbital manoeuvres to avoid collisions. By default the TLEs collected are the closest measurement to the observation time. This forms the mean of the prior distribution. The offset given by this parameter is in days where a negative value collects TLEs from times prior to the observation and positive values lead to TLEs from after the observation.
* `ric_std`: The error in the orbital elements are not provided as part of the TLEs. When TLE estimated positions are analysed the error is calculated in a local reference frame of the satellite. This is the radial, in-track, and cross-track (RIC) frame. This parameter gives a factor by which to scale the RIC covariance that is stored internally which is taken from a paper where the average errors are calculated.  

## Gains