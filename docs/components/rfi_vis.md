# RFI Visibility Calculation Components

The RFI visibility component turns the per-antenna RFI signal (`rfi_A`) and the geometric phase of each source at each antenna (`rfi_phase`) into the RFI visibility on every baseline, integrated over each channel and time step of the data. The integral is a Riemann sum over a fine grid of `n_int_freq x n_int_time` samples per data cell: the phase turns within an integration, and the sum is what averages the fringe.

## Riemann sums over a fine-grid signal - {class}`~tabascal.components.rfi_vis.RiemannVis`, {class}`~tabascal.components.rfi_vis.RiemannVisFFI`

These read `rfi_A` on the fine grid, which the Fourier-domain signal components supersample onto by zero-padding their latent spectrum. `RiemannVis` is the pure-JAX reference, scanned over blocks of baselines (`rfi.baseline_block_size`); `RiemannVisFFI` is the same integral through the compiled kernels of `ri_kernels` (see [RFI-visibility kernels](../kernels.md)). {class}`~tabascal.components.rfi_vis.RiemannVisVariable` and {class}`~tabascal.components.rfi_vis.RiemannVisVariableFFI` sample each baseline only as finely as its fringe rate needs.

## Interpolating from the data grid - {class}`~tabascal.components.rfi_vis.GPInterpVis`

`GPInterpVis` reads `rfi_A` on the data grid instead -- one value per channel and time step, at the centre of its cell -- from {class}`~tabascal.components.rfi_signal.ComplexRFIVarAntCoarse` or {class}`~tabascal.components.rfi_signal.ComplexRFIConstAntCoarse`, and forms the fine samples itself, as the conditional mean of the signal's Gaussian process given the block of coarse values around each cell (`rfi.gp_interp_stencil` cells on every side; the default `1` is the 3 x 3 block, nine values and a 9 x 9 covariance). The covariance is the prior's own: the inverse transform of the spectrum the signal component samples, which it leaves on the configuration for this component to read. The weights are one solve at setup, shared by every cell with the same neighbours, and the visibility is then the same Riemann sum `RiemannVis` forms, through the same blocked kernel. See {mod}`tabascal.gp_interp`.

```yaml
model:
  components:
    - trajectory:FixedOrbitCoarse
    - rfi_signal:ComplexRFIVarAntCoarse
    - rfi_vis:GPInterpVis
    - ast_vis:GPVisAst
    - gains:UnitaryGains
```

The phase can arrive on either grid. From `trajectory:FixedOrbit` or `trajectory:PhaseCalculationRFI` it is the fine grid, sliced per block. From {class}`~tabascal.components.trajectory.FixedOrbitCoarse` or {class}`~tabascal.components.trajectory.PathCalculationRFI` it is the data grid -- the wrapped phase at the channel and cell centres -- beside `rfi_path`, the geometric path and its first `rfi.path_order` time derivatives, and each block's fine phase is rebuilt from the two inside the scan: exactly across frequency, where the phase is linear in it, and by the Taylor series across time. See [Trajectory components](trajectory.md) and {mod}`tabascal.rfi_path`. With both the signal and the phase on the data grid, as in the list above, neither fine grid exists anywhere but inside a block.

What this buys is locality. The Fourier supersampling needs the whole axis at once, so the fine grid exists in full between the signal and visibility components. Here the state the optimiser carries between them is the data grid, and the fine grid -- the amplitude, and with a data-grid phase the phase too -- can be formed a block of time steps at a time (`rfi.time_block_size`), under the same recompute-in-reverse `checkpoint` the baseline scan uses; the two blockings compose. An axis with a single integration sample is left out of the stencil whatever the setting, since its one fine sample is the coarse value itself and the product form of the prior puts every weight on it already -- the default configuration, which supersamples time alone, interpolates along time only, from three neighbouring time steps.

What it costs is exactness against the supersampled grid. The interpolation is the conditional mean given a block of neighbours rather than the whole grid, so it differs from the Fourier supersampling of the same coarse values by the process's posterior scatter within a cell. On a draw from the prior at a correlation time of eight integrations that is about `1e-3` of the signal's rms with the 3-point time stencil, and a few times less with the 5-point one. A prior with structure below the cell width cannot be interpolated from the cell centres by either route: that is a prior rougher than the data grid, not a property of the interpolation.

Pair it with a data-grid signal component. The state key is the one the fine-grid components write too, so the assembly check cannot tell the two apart; the component refuses a fine-grid `rfi_A` by shape instead, when the model is first traced, naming the components that would have been right.
