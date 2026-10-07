# Trajectory Components

## Data grid - {class}`~tabascal.components.trajectory.FixedOrbit`

The fixed-orbit trajectory of {class}`~tabascal.components.trajectory.FixedOrbitFine`, written once per data cell for {class}`~tabascal.components.rfi_vis.AnalyticVis`: the phase at each channel and cell centre, and the geometric delay relative to the array mean with its first `rfi.path_order` time derivatives (`rfi_delay_poly_us`). See [Analytic RFI visibilities](../analytic_rfi_vis.md).
