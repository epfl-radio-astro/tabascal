# Trajectory Components

## Data grid - {class}`~tabascal.components.trajectory.FixedOrbit`

The fixed-orbit trajectory of {class}`~tabascal.components.trajectory.FixedOrbitFine`, written once per data cell for {class}`~tabascal.components.rfi_vis.AnalyticVis`: the phase at each channel and cell centre, and the geometric delay relative to the array mean with its first `rfi.path_order` time derivatives (`rfi_delay_poly_us`). See [Analytic RFI visibilities](../analytic_rfi_vis.md).

## Fine grid - {class}`~tabascal.components.trajectory.OrbitDeviationFine`

A Gaussian-process offset of each satellite from the orbit before it, for when the orbit records place a satellite less accurately than the phase needs. It reads the upstream `rfi_xyz`, in the GCRS, and writes

$$
\mathbf{x}(t) = \mathbf{x}_\mathrm{orbit}(t) + R(t)\,\boldsymbol{\delta}(t),
$$

where $\boldsymbol{\delta} = (\delta_R, \delta_I, \delta_C)$ is the offset along the satellite's radial, in-track and cross-track (RIC) axes and $R(t)$ the rotation from those axes to the GCRS: radial along $\mathbf{r}$, cross-track along $\mathbf{r}\times\mathbf{v}$, and in-track completing the right-handed set. The offset itself is put in the model state as `rfi_orbit_dev` `(n_rfi, n_time_fine, 3)`. It is not among the quantities a run writes to its results zarr.

It goes between an orbit and the phase calculation:

```yaml
model:
  components:
    - trajectory:FixedOrbitFine
    - trajectory:OrbitDeviationFine
    - trajectory:PhaseCalculationRFIFine
    - rfi_signal:ComplexRFIVarAntFine
    - rfi_vis:RiemannVisFine
    - ast_vis:GPVisAst
    - gains:UnitaryGains
```

and works the same way after a learnable orbit, `trajectory:NoDragOrbitFine` or `trajectory:OrbitFine`. The gradient then reaches the orbit's elements through the base positions and the basis they set, as well as the deviation's own parameters. `FixedOrbitFine` also writes a phase, which is of the undeviated orbit. The component therefore declares `rfi_phase` stale, and a list that reads it again without a `PhaseCalculationRFIFine` in between is refused when the model is assembled. Like the phase calculation, it requires `model.precision: double`.

**The velocity** that orients the basis is a central difference of the positions (one-sided at the ends), so the component works after any orbit, including the fixed ones, which output no velocity. Only the orbital plane enters: in-track is cross-track times radial, and a difference of positions on the orbit lies in the plane they span, so the difference's own error along the track turns no axis. What remains is the plane's own turn over a fine-grid step, small for any orbit sampled finely enough for its phase. The basis only defines the coordinates the prior is written in, and never moves the base orbit, which a zero deviation reproduces exactly.

**The prior** on each axis is the Fourier Gaussian process of the RFI amplitude ([`tabascal.fft_gp`](../concepts/gp.md)) in its real signal space: one Hartley coefficient per time mode, on the data-grid time axis padded by `orbit_deviation.time_pad_factor` and supersampled onto the fine grid. Each axis's spectrum is normalised so that the deviation has standard deviation `orbit_deviation.std` metres on that axis at every time. The three axes share one latent grid, the modes their shortest `corr_time` keeps above `orbit_deviation.cutoff`; an axis with a longer one gives the extra modes less power, but generally not none. The parameter is `rfi_orbit_dev_base` `(n_rfi, 3, n_k_time)`, standard normal, and starts at zero. See [Orbit deviation](../config.md#orbit-deviation) for the keys.

There is no data-grid variant. The data-grid route's `trajectory:FixedOrbit` hands {class}`~tabascal.components.rfi_vis.AnalyticVis` a delay polynomial it fits once, in NumPy, from fixed positions. Nothing on that route recomputes it from a state `rfi_xyz`, so a deviation there would first need a differentiable data-grid phase calculation.
