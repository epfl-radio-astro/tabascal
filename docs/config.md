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
  gain_table:
  save_rfi_per_sat: false
```

* `sim_dir`: Simulation directory created when using `sim-vis` to simulate a dataset. This can also be given at runtime of `tabascal` with the `-s` flag.
* `ms_path`: Path to the Measurement Set (MS) to run on. This can also be given at runtime of `tabascal` with the `-ms` flag.
* `data_col`: The data column within the MS file to use as the observed data. Default is `DATA` but can be any column that exists in the MS file.
* `freq`: This is the frequency channel to run on. Default is to run on all frequency channels.
* `corr`: This is the correlation product to run on, default `xx`. It is matched against the MS's `POLARIZATION::CORR_TYPE` **by identity, not by position**, so it names the correlation you want rather than an axis index: `yy` selects YY whether the MS holds all four correlations or only that one. Linear (`xx`, `xy`, `yx`, `yy`), circular (`rr`, `rl`, `lr`, `ll`) and Stokes (`i`, `q`, `u`, `v`) names are accepted. Requesting a correlation the MS does not hold is an error naming what it does hold.
* `freq`: A single frequency in Hz to read instead of the whole band; the nearest channel is used. The request must fall inside the band — a frequency more than half a channel beyond the nearest centre is an error rather than a silent read of the edge channel, since `argmin` always returns a channel and a units slip would otherwise pass unnoticed. `null` (the default) reads every channel. `SIGMA_SPECTRUM` is narrowed to the same channel, so the noise cannot come back on a channel the visibilities did not.
* `noise`: The per-visibility noise in Jy. **Leave it null to use the MS's own noise columns**, which are read *per baseline*, and *per channel* where the MS resolves them that far, rather than averaged to one number: the antennas of a real array differ in sensitivity and a bandpass is not flat, so a single value mis-weights every visibility. On EDA2 the per-baseline `SIGMA` spans a factor of ~30, so a scalar under-weights the quietest baselines by up to ~200x. It matters most when fitting gains, because the per-antenna noise correlates with the per-antenna gain (measured `sigma_a ~ amplitude_a^0.76`, R = 0.96) — a uniform-noise likelihood cannot tell a loud antenna from a noisy one, so the fitted gain absorbs the noise structure.

  With `noise: null` the noise is resolved from the MS, most specific column first:

  | column | shape read | constant in time | varying in time |
  |---|---|---|---|
  | `SIGMA_SPECTRUM` | `(row, chan, corr)` | `(n_bl, n_freq)` — per baseline **and** channel | `(n_bl, n_freq, n_time)` |
  | `SIGMA` | `(row, corr)` | `(n_bl,)` — per baseline | `(n_bl, 1, n_time)` — per baseline **and** timestep |

  The column is checked against the MS's **whole** channel axis before any single-channel selection is applied: a `SIGMA_SPECTRUM` that covers a different number of channels from `DATA` is a disagreement about the observation, and narrowing both to one channel would let it pass without settling which is right.

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

  **`noise` is in the frame of `data_col`, i.e. before `gain_table` is applied.** With a gain table, leave it null and let the MS's `SIGMA` carry the noise rather than pre-scaling it by hand — the table is divided out of the noise as well as the data, so a value given here would be scaled a second time.

* `gain_table`: Path to a CASA calibration table — from `gaincal`, `flux-calibrate`, or anything else that writes one — or an ordered list of them. The gains are divided out of the visibilities **and the noise** once, when the MS is read:

  ```
  vis_obs   = DATA  / (g_p conj(g_q))
  sigma_cal = SIGMA / |g_p conj(g_q)|
  ```

  so everything downstream — priors, the RFI and astronomical models, chi², and the results written back to the MS — lives in one frame, the calibrated one, and the gains are applied once instead of on every forward pass. Carrying the noise with the data is the point of using a table rather than scaling the data by hand: get it wrong and chi² is off by `|g|²`. This replaces the manual "scale `noise` by k and `ast.pow_spec.p0` by k²" workaround.

  The division happens in memory, so `data_col` on disk is untouched and still raw. The results writer therefore removes the same layer again when it writes the `TAB_*` columns and the weights; the run hands it this list automatically, and `tab2MS` takes it as `-gt` — see [Output](output.md).

  A table is solved on whatever `(frequency, time)` grid the calibrator chose, so it is placed on the observation's grid first. Matching is **by value**, not by index — a time within 1 ms and a channel within 1e-6 of the band centre frequency count as the same sample — which is what lets a table solved on a master MS apply to a subset carved out of it, in whatever channel order the subset was written. The times matched are the MS's own `TIME` column on the scale it declares (`times_mjd`), not the UTC-normalised `times_jd`, because a caltable's `TIME` is a copy of the MS's: declared frame to declared frame is exact, where the UTC coordinate of a TAI-declared MS is 37 s away.

  Where the grids do not coincide the gains are interpolated **linearly in amplitude and unwrapped phase, never in real and imaginary parts**: two unit gains 60° apart average to `|g| = 0.87` in real/imag, so the data would be divided by a gain no antenna ever had and the flux scale would move by 13 %. The phase is unwrapped in **two dimensions** — along frequency within each timestep, and then the timesteps onto one another by whole turns — and stays a real surface through both interpolation stages, `exp(i·phase)` being applied once at the end. A `B` table winds across the band through a residual delay and around the ±π branch cut in time, so interpolating the stored angles across such a step averages the two sides of the cut into a gain pointing the wrong way; rebuilding a complex gain between the two stages loses the branch just as badly, and tears the band by a whole turn at whichever channel crossed the cut. **Beyond the solved range the edge value is held**: a table that does not reach the start of the observation calibrates it with the earliest solution it does have.

  **The table has to sample the phase below half a turn, and nothing here can check that it did.** Unwrapping recovers a phase only where the solutions sample it: a genuine change of more than π between two adjacent solved samples — between two solved channels, or between two solution intervals of a phase slewing faster than the calibration cadence follows — is simply not in the table, because complex solutions carry the phase modulo 2π. Such a step aliases to the shorter branch and is taken as such: a true `0 → 1.5π` evolution between two solutions interpolates to `−0.25π` half way, not `+0.75π`. This is the same assumption `np.unwrap` makes, applied coherently across the band rather than channel by channel, and it is undetectable from the data — it is a requirement on the calibration that produced the table (solve finely enough in both axes), not something the interpolation can warn about. A step of exactly π is the boundary case, and is left as written rather than turned into a half turn of the opposite sign.

  A flagged or zero solution is an *absence*, not a value: the interpolation bridges across it from the solutions either side, exactly as it bridges a coordinate the table never sampled. Only an antenna with no valid solution anywhere has nothing to interpolate from — its gain is 1 and its visibilities are flagged, **even when `flags: false`**, since a visibility nobody calibrated is not data.

  With a list, **each table is interpolated onto the observation's grid and only then are they composed**, `g_total = Π gᵢ` in the order given. The two orders disagree: two amplitudes ramping 1 → 3 give `2 × 2 = 4` half way when each is interpolated first, and `(1 + 9) / 2 = 5` when the product is interpolated, which is an artefact of fitting a quadratic with a straight line.

  Each table prints one coverage line as it is read, giving the fractions of the observation's samples whose gain was taken exactly from a solution, interpolated between solutions, held from an edge, or left unsolved — a table that turns out to cover the observation mostly by extrapolation says so rather than being applied in silence. Each sample is classified by the support it was actually built from, so a table whose solutions run along a diagonal reports the edge-holds it really performed rather than the rectangle its two axes span. The coverage line is also what catches a table whose `TIME` was written on some other unit or scale, which is assumed rather than read: it would report no exact cover at all.

* `save_rfi_per_sat`: Also store the fitted RFI visibility **split per satellite** in the results `.zarr`, as `rfi_vis_src (sample, src, bl, freq, time)` with a `norad_id` coordinate naming the satellite behind each `src`. Default `false`.

  It is a diagnostic for astronomical signal leaking into the RFI model: a genuine satellite is a clean streak in exactly one per-source image, while a feature that appears in several is sky flux the model has split across satellites — which reduced chi² is blind to, since the split costs it nothing. Write the sources into MS columns with [`tabascal rfi-per-sat`](output.md) and image one at a time.

  Off by default because it is not free. The stored array is `n_rfi` times the size of `rfi_vis`, and filling it costs `n_rfi` extra evaluations of the run's own RFI-visibility op — one per satellite, each over the whole source axis with the other satellites' amplitudes held at zero, which is what keeps the evaluation inside the RFI-axis sharding rather than gathering the fine grids onto one device. That is a few forward passes' worth of work at the end of a fit, not a second fit, but it is `n_rfi` × the *storage* forever.

  **The same multiplier applies to memory on the writing process**, which assembles the whole decomposition before handing it to the zarr: budget `n_rfi × sizeof(rfi_vis)` on rank 0 at the end of a run, on top of what the fit already holds. On disk it is chunked one satellite per chunk, so reading or imaging a single source does not pull the rest of them in.

  Padded sources are not stored: under sharding the satellite list is padded to a multiple of the device count with dark dummies, and only the real satellites get a `src` slice. The sources sum back to `rfi_vis` — exactly in exact arithmetic, and to within round-off in floating point, since splitting the op's single reduction over (source, integration sample) into per-source partial sums re-associates it.

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

### A fixed sky of discrete sources

The GP sky above is flexible: it has per-baseline freedom, so a free per-antenna gain can be absorbed into it and is a flat direction of the likelihood. A source with a *known* position, flux and shape is rigid, and so anchors the gain. That sky is configured with `ast.point_sources` and needs two components, {class}`~tabascal.components.ast_signal.FixedDiscreteSky` to read the catalogue and {class}`~tabascal.components.ast_vis.DiscreteSkyVis` to turn it into visibilities:

```yaml
model:
  components:
    - ast_signal:FixedDiscreteSky   # must come before DiscreteSkyVis
    - ast_vis:GPVisAst              # optional
    - ast_vis:DiscreteSkyVis
    - gains:UnitaryGains

ast:
  point_sources:
    - {name: Fornax A, ra: 50.6738, dec: -37.2083, I: 750.0, ref_freq_mhz: 154.0, alpha: -0.77}
  source_block_size: 128
```

`FixedDiscreteSky` writes the sky into the model state and `DiscreteSkyVis` reads it, so **`FixedDiscreteSky` must be listed first**. `DiscreteSkyVis` *accumulates* into `vis_ast` rather than assigning it, so it composes with `ast_vis:GPVisAst` — with both listed, `vis_ast` is the GP plus the fixed sources, and since `vis_ast` is zeroed before the components run the two may be given in either order. Neither component has any free parameters.

Sources are either the inline list above or a path to an OSKAR sky model file:

```yaml
ast:
  point_sources: /path/to/sky.osm
```

The inline form requires `ra` and `dec` in degrees and `I` in Jy, and optionally takes `name`, `ref_freq_mhz`, `alpha`, `Q`, `U`, `V`, `rm`, `fwhm_major_arcsec`, `fwhm_minor_arcsec` and `position_angle_deg`. That list is exhaustive and case-sensitive: any other field is an error rather than being ignored, since a typo would otherwise change the source in silence — `fwhm_maj` for `fwhm_major_arcsec` would leave a Gaussian modelled as a point. An optional field may be given as `null` to mean "unset", but any other unreadable value (`alpha: ""`, `ref_freq_mhz: false`) is an error rather than a fallback to the default. The file form is the OSKAR sky model, which is what [Karabo](https://github.com/i4Ds/Karabo-Pipeline) emits: one source per line, fields separated by whitespace and/or commas, `#` starts a comment, and blank lines are skipped.

`I` is the *integrated* flux of the source, and that is what appears on a zero-length baseline in any direction — there is no `1/n` applied to it. (The measurement equation carries the sky *brightness* as `B/n`, because the solid-angle element is `dl dm / n`; for a discrete source of integrated flux `S` the brightness is `S` times a delta function of solid angle, and the two factors of `n` cancel exactly.)

| # | Column | Units | Meaning |
|---|---|---|---|
| 1 | RA | deg | Right ascension |
| 2 | Dec | deg | Declination |
| 3 | I | Jy | Stokes I at the reference frequency |
| 4 | Q | Jy | Stokes Q |
| 5 | U | Jy | Stokes U |
| 6 | V | Jy | Stokes V |
| 7 | Reference frequency | Hz | Frequency at which the Stokes fluxes are quoted |
| 8 | Spectral index | — | `alpha` in `I(nu) = I (nu / nu_ref)**alpha` |
| 9 | Rotation measure | rad/m² | |
| 10 | FWHM major | arcsec | Gaussian major axis; `0` for a point source |
| 11 | FWHM minor | arcsec | Gaussian minor axis; `0` for a point source |
| 12 | Position angle | deg | Major-axis position angle, north through east |

A row need not carry all twelve fields, but only three lengths are meaningful, and they follow OSKAR's own fixed-format reader:

* **3 to 9 columns** — the leading columns above, with the rest defaulting to zero. `ra dec I` is a valid flat-spectrum point source.
* **11 columns** — the *legacy* layout: columns 1-8 as above, then FWHM major, FWHM minor and position angle, with no rotation measure (it defaults to zero). This is not the 12-column layout with one field missing; read as though it were, the major axis lands in the rotation-measure column and a perfectly ordinary Gaussian is rejected as polarised.
* **12 columns** — the full modern layout.

**10 columns, or 13 and more, are rejected**: 10 is a half-specified shape (a major axis with no minor axis or position angle) and 13+ is not the format.

Two points about the columns themselves:

* **Polarisation is parsed but not modelled.** A non-zero Stokes Q, U or V, or a non-zero rotation measure, is *rejected* with an error naming the source, rather than being silently dropped. The whole format is accepted so that modelling polarisation later ([issue #151](https://github.com/epfl-radio-astro/tabascal/issues/151)) widens what these values mean rather than changing what the file may contain.
* **A spectral index needs a reference frequency.** A non-zero spectral index with no positive reference frequency is an error, since falling back to a flat spectrum would put the source at the wrong flux in every channel with nothing in the output to say so. A zero spectral index is a flat spectrum and needs no reference frequency.

A source with a non-zero FWHM is an elliptical Gaussian, which multiplies the point-source visibility by the uv-plane envelope

$$G(u,v) = \exp\left(-\frac{\pi^2}{4\ln 2}\left(a^2 u'^2 + b^2 v'^2\right)\right)$$

for FWHM $a$ (major) and $b$ (minor) in radians, where $(u', v')$ is the baseline in wavelengths rotated into the source frame, $u' = u\sin\phi + v\cos\phi$ along the major axis and $v' = u\cos\phi - v\sin\phi$ along the minor axis. The position angle $\phi$ follows the radio convention, measured from north (the $m$ axis) through east (the $l$ axis), so `position_angle_deg: 0` puts the major axis north-south and a north-south baseline is the one that resolves the source out. A zero FWHM gives $G = 1$ exactly, so points and Gaussians are the same code path and the same component.

A source more than 90 degrees from the phase centre is modelled as given, not rejected — the w term is computed exactly over the whole sphere — but it raises a warning naming the sources, since in practice it means a swapped or mis-signed coordinate rather than a real field.

* `source_block_size`: The number of sources `DiscreteSkyVis` handles per step of its scan over the catalogue, a whole number defaulting to `128`. The geometric delay array is `(n_bl, n_time, n_src)`, which for a large catalogue is the biggest array in the model; the scan replaces `n_src` in that shape with `source_block_size`, at the cost of recomputing each block in the backward pass. It is purely a memory strategy — the result does not depend on it.

Note that the catalogue fluxes are in the same scale as the data the model is fit to. With data calibrated to Jy these are physical Jy; without that, the data are in raw correlator units and a Jy catalogue flux is meaningless.


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

All parameters in this section that overlap with those of the `ast` section have the same definition, except that `init` and `mean` accept one more value:

* `init` / `mean`: `matched-filter` (alias `mf`) estimates the per-satellite light curves directly from the visibilities the run has already loaded, by matched-filtering them against the known satellite trajectory phase, and seeds the RFI amplitude with them. It is the same seed as `est` without the file: no imaging step, no `rfi.est`, and no matching of light curves to satellites by name, since the estimator is handed `satellites.norad_ids` and returns the curves in that order. See [Estimating the light curves from the data](#estimating-the-light-curves-from-the-data).

The only additional parameters are

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

### Estimating the light curves from the data

`rfi.init: matched-filter` (alias `mf`) needs no file at all: it measures the light curves from the visibilities the run has already loaded. For a satellite on a known trajectory the RFI contribution to baseline $(p, q)$ is $A_p A_q^* e^{i(\phi_p - \phi_q)}$ with $\phi$ the geometric phase, so the unit-modulus template $T_{pq} = e^{i(\phi_p - \phi_q)}$ de-rotates it. The maximum-likelihood estimate of the source visibility at each channel and timestep is the inverse-variance-weighted, de-rotated baseline average

$$\hat{S}[f, t] = \frac{\sum_{pq} w_{pq} T_{pq}^{*} V_{pq}}{\sum_{pq} w_{pq}}, \qquad w_{pq} = \frac{1}{\sigma_{pq}^2},$$

with standard error $1/\sqrt{\sum w}$. The satellite's fringe adds coherently after de-rotation while the sky and the noise do not, so $\hat{S}$ isolates the satellite, and $\sqrt{\lvert \hat{S} \rvert}$ is the per-antenna amplitude the model is seeded with — the same quantity `est` reads out of a file.

The weights are the run's own noise, resolved per baseline and per channel as far as the MS resolves it (see `data.noise`); the template carries no gain. Uniform weights on an array whose antennas differ in sensitivity would over-weight the loud baselines. With no usable noise anywhere the curves are still measured, but they are then both unweighted and *unscaled*: the error and the z statistic come back as NaN rather than as a floor derived from a noise nobody stated. Flagged samples and autocorrelations are excluded, a satellite is not filtered for while it is below `rfi.min_elevation` (those times seed at zero, exactly as the elevation mask holds them there), and a channel and timestep where every baseline is flagged seeds at zero rather than at a measured value.

The same estimate is available as a standalone tool, `tabascal light-curve`, which writes it in the interchange format above so it can seed a later run through `rfi.est`. See [Usage](usage.md#extracting-rfi-light-curves).

The z statistic reported alongside the curves reads `Re(S_hat)` and so assumes
the column is phase calibrated. The magnitude statistic reported beside it is
robust to a phase common to every baseline, but not to an uncalibrated antenna
gain, which decorrelates the coherent sum and shrinks the estimate itself — see
[Usage](usage.md#extracting-rfi-light-curves).

Two limitations are worth knowing. The estimator assumes the satellite is exactly where its orbit record says it is: a position error scatters the per-baseline phases and costs coherence, which shows as an under-estimated flux rather than as an error. And it does not filter the astronomical signal out first, so a bright source in the field contributes to $\hat{S}$ wherever its fringe rate overlaps the satellite's.

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

The `gains` section defines the prior over the antenna gains and how they are initialised. An example is given below.

```yaml
gains:
  init: prior
  amp_mean: 1.0
  amp_std: 10
  phase_mean: 0.0
  phase_std: 30
  amp_corr_time: 180
  phase_corr_time: 180
  ref_ant: null
  fix_flux_scale: true
```

* `init`: How the gain parameters are initialised. `prior` (the default) starts at the prior mean. `gains:ConstGains` additionally accepts a path to a previously measured gain — see [A constant gain per antenna](#a-constant-gain-per-antenna).
* `amp_mean`: The centre of the prior over the gain amplitude. It must be positive and finite. `gains:GPGains` takes it as the mean of a Gaussian; `gains:ConstGains` fits the log amplitude, so there it is the **median** of a lognormal — the centre in log space — and it reads it only when `fix_flux_scale` is `false`. See below.
* `amp_std`: The standard deviation of the prior over the gain amplitude, **as a percentage** of `amp_mean`. `amp_std: 10` with `amp_mean: 1.0` is a 10 % spread.
* `phase_mean`: The mean of the prior over the gain phase, in **radians**.
* `phase_std`: The standard deviation of the prior over the gain phase, in **degrees**.
* `amp_corr_freq`, `amp_corr_time`, `phase_corr_freq`, `phase_corr_time`: The correlation lengths of the gain Gaussian process, in Hz and in seconds. They are read by `gains:GPGains` only, and each defaults to the extent of the observation along that axis, i.e. to a gain that varies smoothly across the whole run.
* `ref_ant`, `fix_flux_scale`: Read by `gains:ConstGains` only; see below.

### A constant gain per antenna

{class}`~tabascal.components.gains.ConstGains` fits **one complex gain per antenna, constant over time and frequency** — the static direction-independent gain the array is known to have:

$$V^\text{OBS}_{pq} = g_p g_q^* \left( V^\text{AST}_{pq} + V^\text{RFI}_{pq} \right)$$

It adds only $2 n_\text{ant} - 2$ parameters, and unlike a *fixed* gain it is constrained by the data.

```yaml
model:
  components:
    - trajectory:FixedOrbit
    - rfi_signal:ComplexRFIConstAnt   # not ComplexRFIVarAnt — see below
    - rfi_vis:RiemannVis
    - ast_signal:FixedDiscreteSky
    - ast_vis:DiscreteSkyVis
    - gains:ConstGains

gains:
  ref_ant: null
  fix_flux_scale: true
```

**A gain is only identifiable against a model term it cannot deform**, which is what the component list above is for ([issue #124](https://github.com/epfl-radio-astro/tabascal/issues/124)):

* **Pair it with `rfi_signal:ComplexRFIConstAnt`.** With the per-antenna RFI model `ComplexRFIVarAnt` the RFI amplitude $A_p$ is already free per antenna, so $g_p A_p (g_q A_q)^*$ is unchanged by $g_p \rightarrow c_p g_p$ together with $A_p \rightarrow A_p / c_p$: the gain is an exact flat direction of the RFI term and only the astronomical model constrains it. Setup **warns** when the two are combined rather than refusing — the pairing is a modelling rule, not a hard error — and names the flat direction.
* **The astronomical GP absorbs a gain the same way.** `ast_vis:GPVisAst` has per-baseline freedom, so a gain solved against it alone is a reparametrisation of an already-free `vis_ast`. A rigid sky — `ast_signal:FixedDiscreteSky` with `ast_vis:DiscreteSkyVis` — is what anchors the gain.

**The gauge.** The gain is purely *relative*: it carries no absolute flux scale and no absolute phase, and both are removed by construction rather than fitted.

* The overall **phase** is unobservable, so `ref_ant`'s phase is pinned to exactly 0 and the other $n_\text{ant} - 1$ phases are free.
* The overall **amplitude** is degenerate with the RFI source amplitude and the astronomical amplitude, so the log amplitudes are carried by $n_\text{ant} - 1$ parameters on an orthonormal basis of the zero-sum subspace, giving $\sum_p \log |g_p| = 0$, i.e. a geometric mean $|g|$ of exactly 1. Left free it simply drifts — in one earlier run it settled at a median $|g|$ of 0.70, with the sky model absorbing the reciprocal — which is a nuisance direction that buys nothing and slows convergence.

Both directions are removed from the **parameters**, not just from the value they map to. Writing $n_\text{ant}$ amplitude parameters and subtracting their mean would give the same gains and the same prior, but would leave the all-ones direction of that latent space invisible to every visibility: flat in the likelihood however much data there is, curved only by the prior. Such a coordinate wrecks the conditioning of the optimisation and makes a likelihood-only Fisher matrix singular, so there is no such coordinate.

The prior on $|g_p|$ is **lognormal**: `amp_std` is used as the standard deviation of $\log|g|$, which agrees with a fractional spread to first order and keeps the gain positive by construction. `amp_mean` is the **median** of that prior — its centre in log space, and the value the fit starts at — rather than its arithmetic mean, which is the slightly larger $\texttt{amp\_mean} \cdot e^{\sigma^2/2}$ for $\sigma$ the log-space spread. And it is that **only when the flux scale is free**: under the zero-sum gauge the geometric mean of $|g|$ is 1 by construction and `amp_mean` merely sets the scale that `amp_std`'s percentage is taken of, so a non-unit `amp_mean` with `fix_flux_scale: true` raises a warning saying so. A value that is not positive and finite is an error rather than a default — `amp_mean: 0` used to be read as "unset" and silently become 1.0.

* `ref_ant`: The antenna whose phase is pinned to 0. `null` (the default) selects the first antenna with any unflagged data. An antenna every one of whose baselines is flagged everywhere is not constrained by any visibility, so it cannot be the reference the others are measured against, and naming one explicitly is an error rather than a silently unpinned fit.

  **One reference pins one connected group.** A phase is measured relative to another antenna's along a chain of baselines that carry data, so if the unflagged baselines split the array into groups that share no baseline, pinning `ref_ant` in one group leaves every other group's overall phase unconstrained. Setup computes the connected components of the unflagged baseline graph and **stops**, naming the groups, rather than fitting a model with a flat direction in it. Fit the groups separately, or flag the smaller ones out of the run. Heavy flagging is fine as long as what survives still connects the array.
* `fix_flux_scale`: Whether to keep the zero-sum log-amplitude constraint above. `true` is the default. `false` frees the overall amplitude and is accepted **only** with `ast_signal:FixedDiscreteSky` in `model.components`, since a fixed-flux sky is the one thing in the model that can set the scale; without it the run stops at setup with the degeneracy spelled out. Nothing else changes: the phase reference is still pinned either way.

The absolute flux scale is assumed to be set by the data — e.g. by a `REAL_DATA_FLUXCAL` column, or by the fixed sky above — and if it ever needs fitting it belongs in a separate scalar component rather than in the gain.

**Initialising at a measured gain.** `gains.init` optionally starts the fit at a gain measured elsewhere, which is a much better starting point than the prior mean:

```yaml
gains:
  init: /path/to/gains.npz     # or /path/to/caltable.B
```

* An **`.npz`** carrying the per-antenna gain under the key `gain`, shape `(n_ant,)` complex.
* A **calibration table**, read with {func}`~tabascal.ms.read_caltable` — one tabascal wrote with `tab2MS`, or one from CASA. A caltable is resolved over frequency and time and `ConstGains` is not, so it is reduced to the median $|g|$ and the mean phase direction over each antenna's valid samples, with a warning naming the largest deviation when the solutions actually do vary. Flagged solutions are dropped rather than counted as zero, and an antenna the table has no solution for at all falls back to unit gain, with a warning.

Either way the gain is **projected into the gauge** rather than taken as given: the mean log amplitude is subtracted (unless `fix_flux_scale: false`) and the phases are referenced to `ref_ant`. Projecting an already-projected gain changes nothing, so the same file can be handed back to a second run. A zero or non-finite gain is an error — the fit is in log amplitude, which such a value has no value at.
