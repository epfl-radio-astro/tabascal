# RFI Visibility Calculation Components

## Data grid - {class}`~tabascal.components.rfi_vis.AnalyticVis`

Integrates each data cell in closed form from the data-grid signal, phase and delay polynomial, through the compiled `RFIAnalyticVisOp` of `ri-kernels`; no fine time grid is formed. It replaces the whole fine-grid chain, not only the `RiemannVis*Fine` component. See [Analytic RFI visibilities](../analytic_rfi_vis.md).
