# RFI Signal Components

The RFI signal component models the complex-valued signal of each RFI source, at each antenna, over time and frequency. Therefore, this signal captures the combination of the intrinsic signal of the RFI source as well as the direction dependent effects such as the primary beam and the ionosphere. 

<!-- The effect of the primary beam and the ionosphere are expected to vary more smoothly over time and frequency compared to the instrinsic signal. Therefore, the level of correlation in the signal over time and frequency is typically limited on the smooth side by these direction dependent factors. The frequency axis of the signal  -->

<!-- The combined signal should be  -->

## Fourier-domian - {class}`~tabascal.components.rfi_signal.ComplexRFIVarAnt`, {class}`~tabascal.components.rfi_signal.ComplexRFIConstAnt`

## Data-grid variants - {class}`~tabascal.components.rfi_signal.ComplexRFIVarAntCoarse`, {class}`~tabascal.components.rfi_signal.ComplexRFIConstAntCoarse`

The same two priors, written on the data grid rather than the fine integration grid: one value per channel and time step, at the centre of the cell it stands for. Latent, spectrum, parameters and initialisation are their parents' -- the transform simply leaves out the supersampling, so where the two grids overlap they agree exactly. They exist for {class}`~tabascal.components.rfi_vis.GPInterpVis`, which forms the fine samples the visibility integral needs from this grid under the same Gaussian process; each leaves the spectrum it samples on the configuration for that component to read. The Riemann-sum kernels read the fine grid and refuse a data-grid `rfi_A` by shape.
