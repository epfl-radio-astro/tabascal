# RFI Signal Components

The RFI signal component models the complex-valued signal of each RFI source, at each antenna, over time and frequency. Therefore, this signal captures the combination of the intrinsic signal of the RFI source as well as the direction dependent effects such as the primary beam and the ionosphere. 

<!-- The effect of the primary beam and the ionosphere are expected to vary more smoothly over time and frequency compared to the instrinsic signal. Therefore, the level of correlation in the signal over time and frequency is typically limited on the smooth side by these direction dependent factors. The frequency axis of the signal  -->

<!-- The combined signal should be  -->

## Fourier-domian - {class}`~tabascal.components.rfi_signal.ComplexRFIVarAnt`, {class}`~tabascal.components.rfi_signal.ComplexRFIConstAnt`