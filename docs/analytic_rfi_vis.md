# Analytic RFI visibilities

An RFI visibility is the average over each data cell -- one channel, one
integration -- of the product of two antennas' signals and the fringe between
them. TABASCAL evaluates that average on one of two grids:

- **The fine grid** (the `*Fine` components): the trajectory and the signal are
  written at `n_int_freq * n_int_time` samples per cell and
  {class}`~tabascal.components.rfi_vis.RiemannVisFine` (or one of its kernels)
  takes their mean. `n_int_time` is estimated from the fringe rate, and climbs
  steeply with the array's extent.
- **The data grid**: the trajectory and the signal are written once per cell,
  and {class}`~tabascal.components.rfi_vis.AnalyticVis` integrates each cell in
  closed form through the compiled `RFIAnalyticVisOp` of
  [`ri-kernels`](kernels.md) (0.2.2 or later). No fine time grid is formed, so
  `rfi.time_int_factor`, `rfi.min_time_bins` and `rfi.max_time_bins` are ignored
  and the fringe-rate estimate that sizes it is skipped.

```yaml
model:
  components:
    - trajectory:FixedOrbit
    - rfi_signal:ComplexRFIVarAnt   # or rfi_signal:ComplexRFIConstAnt
    - rfi_vis:AnalyticVis
    - ast_vis:GPVisAst
    - gains:UnitaryGains
```

The three RFI components go together: a data-grid component next to a fine-grid
one is refused when the model is assembled, naming the key and both grids.

## State

| Component | Writes | Shape |
|---|---|---|
| `trajectory:FixedOrbit` | `rfi_xyz` | `(n_rfi, n_time, 3)` |
| | `rfi_phase` | `(n_rfi, n_ant, n_freq, n_time)` |
| | `rfi_delay_poly_us` | `(n_rfi, n_ant, n_time, path_order + 1)` |
| `rfi_signal:ComplexRFIVarAnt` / `ComplexRFIConstAnt` | `rfi_A` | `(n_rfi, n_ant, n_freq, n_time)` |
| `rfi_vis:AnalyticVis` | `vis_rfi` (added to) | `(n_bl, n_freq, n_time)` |

## The model inside a cell

**Phase.** `FixedOrbit` propagates the source at `rfi.path_nodes` nodes across
each cell, forms the geometric delay -- range plus the antenna's `w`, over `c`
-- in float64, and writes two things: the phase at the channel and cell centre,
reduced to a turn in float64, and the delay *relative to the array mean* with
its first `rfi.path_order` time derivatives, from a least-squares polynomial
through the nodes. A term common to every antenna cancels in a baseline's
phase, and what is left is small enough for float32 to carry across a cell.
Delays are in microseconds and frequencies in MHz, so their product is cycles.
The kernel keeps derivatives up to the third, so `path_order` is at most 3.

**Signal.** Before masking, the data-grid signal is the fine-grid signal at each
cell's own sample: the same latent, prior and parameters as its `Fine` twin. Inside a cell
it is the polynomial through the `2 * rfi.analytic.stencil + 1` nearest cells on
each axis (Lagrange weights; shifted inwards at the observation's edges). The
elevation mask is not applied to it, so the prediction-state `rfi_A` is unmasked. Instead
each source's visibility is masked per cell after interpolation, and a visible
cell's stencil reads its hidden neighbours' signal, which is continuous across
the horizon. A satellite that sets mid-cell is still all or nothing per cell.
Under a mask each source is its own kernel call, and ri-kernels 0.2.2 runs a
single source's CPU transpose on one thread, so a masked gradient is several
times slower there than an unmasked one.

**Integral.** The time integral of the product is closed form; the frequency
average is the same `n_int_freq`-point rule as the fine grid. The signal and the
phase are differentiated; the delay and the tables are constants. With
`FixedOrbit` the phase is constant too, and only the signal-only kernels run.

## Options

`rfi.path_order`, `rfi.path_nodes` and `rfi.analytic.*`; see
[the configuration reference](config.md). The defaults (`stencil: 1`,
`segments: 2`, `terms: 6`, `cubic_terms: 3`) are the measured speed/accuracy
trade; `segments: 4, terms: 16` agrees with a dense Riemann sum of the same
integral to ~1e-5.

## Limits

- No fitted orbit on this grid: `NoDragOrbit`, `Orbit` and
  `PhaseCalculationRFI` exist only as `*Fine`, since the kernel stops the
  gradient on the delay.
- `data.save_rfi_per_sat` is refused; it re-evaluates fine-grid state.
- Each `(a1, a2)` baseline may appear once (reversed pairs and autocorrelations
  are fine).

The fine-grid route stays as the reference: the tests hold the analytic route to
it and to a dense Riemann sum of the integral `AnalyticVis` evaluates.
