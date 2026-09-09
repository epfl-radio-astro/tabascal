# Trajectory Components

The trajectory components put each RFI source's position over the observation in the model state as `rfi_xyz`, and turn it into the geometric phase of the source at every antenna, `rfi_phase`, which the RFI visibility component multiplies the signal by. The phase is `-2 pi nu L / c` for the path `L = |x_ant - x_src| + w` in metres, and it changes within an integration -- the fringe turns -- which is why the visibility is integrated over a fine grid of `n_int_time` samples per time step.

## A fixed orbit - {class}`~tabascal.components.trajectory.FixedOrbit`

The orbit records the satellites section resolves are propagated once, at setup, over the fine time grid, and the phase is computed from the positions there, in float64, and carried as a constant. Nothing about the trajectory is fitted.

## A fitted orbit - {class}`~tabascal.components.trajectory.Orbit`, {class}`~tabascal.components.trajectory.NoDragOrbit`, {class}`~tabascal.components.trajectory.PhaseCalculationRFI`

The orbital elements are parameters, with a prior about the resolved record, and the positions are propagated inside the forward pass so that the fit can move them. `PhaseCalculationRFI` then computes the phase from `rfi_xyz` on the fine grid, differentiably. Double precision, since the positions are millions of metres and the phase turns on centimetres.

## The phase on the data grid - {class}`~tabascal.components.trajectory.FixedOrbitCoarse`, {class}`~tabascal.components.trajectory.PathCalculationRFI`

The fine-grid phase, `(n_rfi, n_ant, n_freq_fine, n_time_fine)`, is the largest array a run holds. These two write it on the data grid instead, beside what rebuilds it: `rfi_phase` at the channel and cell centres only, which is the fine-grid phase at exactly those samples, and `rfi_path`, the path `L` and its first `rfi.path_order` time derivatives at each cell centre, differential to the array mean. {class}`~tabascal.components.rfi_vis.GPInterpVis` (and {class}`~tabascal.components.rfi_vis.PolyInterpVis`, which shares its forward) rebuilds each block's fine phase from the two inside its scan -- exactly across frequency, where the phase is linear in it with slope `-2 pi L / c`, and by the Taylor series across time -- so the fine phase never exists beyond a block. See {mod}`tabascal.rfi_path`.

The derivatives come from the propagated positions themselves: the path is evaluated at a window of a few fine samples around each cell centre and a polynomial through them is differentiated there. The fit is at the sample times the positions were actually propagated at, because a float64 Julian date resolves about 20 microseconds and a finite difference on the nominal grid would amplify that jitter with every order. At the default order of 3 the rebuilt differential phase agrees with the fine grid's to a hundredth of a degree on a low-Earth-orbit pass over a 450 m array, and to what that jitter allows on longer baselines.

`FixedOrbitCoarse` does this once at setup, in float64, from the same propagation `FixedOrbit` makes. `PathCalculationRFI` does it in the forward pass from the `rfi_xyz` an orbit component writes, in JAX, so the derivatives with respect to the elements flow through; its cost is a few path evaluations per cell rather than one per fine sample, and it never forms the fine phase at all. Both keep `rfi_xyz` on the fine time grid, which carries no antenna or frequency axis and is what the window samples.

Nothing but `GPInterpVis` and `PolyInterpVis` reads a data-grid phase, so these pair with one of them and a data-grid signal component; the fine-grid kernels refuse the shape.
