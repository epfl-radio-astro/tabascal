# The data-grid RFI route and its kernel boundary

The fine-grid route models the RFI signal and phase on a grid of
`n_int_freq x n_int_time` samples inside every data cell
({class}`~tabascal.components.trajectory.FixedOrbit`,
{class}`~tabascal.components.rfi_signal.ComplexRFIVarAnt`,
{class}`~tabascal.components.rfi_vis.RiemannVisFFI`), and that grid is the
largest thing in the run: `n_rfi * n_ant * n_freq_fine * n_time_fine` complex
samples of state, held for the reverse pass. The data-grid route carries
neither on the fine grid. Both live on the data grid, and the fine samples of
each cell are rebuilt inside the visibility calculation, one time cell at a
time, and discarded.

```yaml
model:
  components:
    - trajectory:FixedOrbitCoarse
    - rfi_signal:ComplexRFIVarAntCoarse
    - rfi_vis:PolyInterpVis
    - ast_vis:GPVisAst
    - gains:UnitaryGains
```

The route exists as a **reference**: the pure-JAX
{func}`~tabascal.coarse_rfi_vis.coarse_rfi_vis` is written to be replaced by a
compiled CPU and GPU kernel, and everything on this page is about where that
boundary is and what crosses it. The three components are deliberately the
smallest things that produce its inputs.

## The three components

{class}`~tabascal.components.rfi_signal.ComplexRFIVarAntCoarse` is
`ComplexRFIVarAnt` with the supersampling left out: the same latent Fourier
modes, prior, parameters and initialisation, and an inverse transform that
lands on the data grid, `rfi_A` of shape `(n_rfi, n_ant, n_freq, n_time)`.
The value it gives a cell is the fine-grid signal at that cell's own sample,
so the two components agree exactly where the grids meet.

{class}`~tabascal.components.trajectory.FixedOrbitCoarse` is `FixedOrbit`
with the phase written at the channel and cell centres only, `rfi_phase` of
shape `(n_rfi, n_ant, n_freq, n_time)`, plus what rebuilds it in between:
`rfi_path` of shape `(n_rfi, n_ant, n_time, rfi.path_order + 1)`, the path
from the source to each antenna (its range plus the antenna's `w`) and its
first `rfi.path_order` time derivatives at each cell centre, from a
least-squares polynomial through the cell's fine samples. Both are float64
host-side constants, as `FixedOrbit`'s phase is.

{class}`~tabascal.components.rfi_vis.PolyInterpVis` builds two small tables
at setup and calls the one function. The tables are the weights that turn the
`2h + 1` coarse samples nearest to a cell into that cell's fine samples, on
each axis (`h` is `rfi.poly_interp_stencil`, default 1), and where each cell's
stencil starts. They are the Lagrange weights of the polynomial through the
stencil: nothing is solved, nothing is read from the prior, and the same
polynomial is evaluated off-centre at the edges rather than a shorter one
fitted. Away from the edges every cell's table is the same.

## The kernel boundary

{func}`tabascal.coarse_rfi_vis.coarse_rfi_vis` is the whole of it. Its module
docstring is the specification; in short, everything comes in on the data
grid and the data-grid visibilities go out:

| input | shape | |
|---|---|---|
| `rfi_A` | `(n_rfi, n_ant, n_freq, n_time)` complex | the signal on the data grid, **the only differentiated input** |
| `rfi_phase` | `(n_rfi, n_ant, n_freq, n_time)` | the phase at the channel and cell centre, reduced to one turn |
| `rfi_path` | `(n_rfi, n_ant, n_time, n_path)` | the path (m) and its time derivatives (m/s^k) at the cell centre |
| `w_freq`, `start_freq` | `(n_freq, n_sf, n_int_freq)`, `(n_freq,)` | interpolation weights across each channel, and each stencil's first channel |
| `w_time`, `start_time` | `(n_time, n_st, n_int_time)`, `(n_time,)` | the same across each cell |
| `dnu`, `dt` | `(n_int_freq,)`, `(n_int_time,)` | the fine offsets from the channel and cell centres (Hz, s) |
| `freqs` | `(n_freq,)` | channel centres (Hz) |
| `a1`, `a2` | `(n_bl,)` | the antennas of each baseline |
| **output** `vis_rfi` | `(n_bl, n_freq, n_time)` complex | |

For one cell `(f, t)` and one of its fine samples `(u, v)`:

```
A[r, a](u, v)   = sum_k sum_l w_freq[f, k, u] w_time[t, l, v] rfi_A[r, a, start_freq[f] + k, start_time[t] + l]
dL[r, a](v)     = sum_{k >= 1} rfi_path[r, a, t, k] dt[v]^k / k!
phi[r, a](u, v) = rfi_phase[r, a, f, t] - (2 pi / c) ((freqs[f] + dnu[u]) dL[r, a](v) + dnu[u] rfi_path[r, a, t, 0])
S[r, a](u, v)   = A[r, a](u, v) exp(i phi[r, a](u, v))
vis_rfi[b, f, t] = mean_{u, v} sum_r S[r, a1[b]](u, v) conj(S[r, a2[b]](u, v))
```

The last line is the fine-grid Riemann sum's integrand
({func}`~tabascal.interferometry.calculate_rfi_vis_fine`, averaged over each
cell). The lines before it replace reading `A` and `phi` from fine-grid state:
the signal is the separable interpolation of the nearest coarse samples, the
phase is exact across the channel (linear in frequency) and a Taylor series
across the cell.

Three things about that boundary are the point of the design:

- **The weights are data.** The kernel does not know they are polynomial. The
  conditional mean of the Gaussian-process prior, a windowed sinc, or any
  other linear interpolant is a different table fed to the same kernel, and
  the stencil width is a table dimension, not a kernel constant.
- **Only `rfi_A` carries a gradient.** The phase comes from a fixed orbit and
  the rest are tables, so the kernel needs a JVP and a VJP with respect to one
  input. Both are structural: the result is bilinear in the fine samples `S`,
  and `S` is linear in `rfi_A`. The JVP is
  `B(dS, S) + B(S, dS)` with `dS` the tangent pushed through the same
  interpolation and phase factor. The VJP scatters the visibility cotangent to
  the fine samples of each baseline's two antennas, each weighted by the other
  antenna's sample, exactly as the fine-grid kernel's transpose does, then
  multiplies by the conjugate phase factor and pushes back through the weight
  tables: a stencil-sized scatter-add onto the data grid. Nothing of the fine
  grid survives either.
- **Precision is arranged, not hoped for.** The unreduced phase
  `2 pi freqs L / c` is of order a million turns, which float32 cannot hold to
  a fraction of a turn. `rfi_phase` carries it reduced, computed in float64 on
  the host; the kernel adds only the change across the cell and across the
  channel, a few hundred radians at most. A kernel must not rebuild
  `rfi_phase` from `rfi_path[..., 0]`.

A compiled kernel is validated as `ri_kernels` is: value, JVP and VJP against
this function. `tests/test_coarse_rfi_vis.py` holds the function itself to the
fine-grid sum on the fine grid it forms, and its derivatives to finite
differences; `tests/components/test_coarse_components.py` holds the three
components to their fine-grid twins on one configuration.

## What the reference is and is not

The reference forms one time cell at a time, `(n_bl, n_rfi, n_freq,
n_int_freq, n_int_time)` complex, under `jax.checkpoint`, so the reverse pass
recomputes the cell rather than keeping it. That bounds its memory at the
sizes the fine-grid route runs at. It is not fast: it gathers per baseline and
recomputes every antenna's fine samples for every baseline that antenna is on.
A kernel would stage each antenna's fine samples once per cell and reuse them
across its baselines, which is the reuse a compiled kernel exists to get.

The fine sample count is a quadrature order for the phase, not something the
signal needs: the signal is `2h + 1` numbers per cell on each axis whatever
the count, and the count is set by how fast the fringe winds across a cell.
Because the interpolant is evaluated inside the kernel, the fine samples need
not be uniform. `dt` and `dnu` are inputs, and a kernel fed Gauss-Legendre
nodes and their weights in place of the uniform mean would integrate the same
cell with fewer samples. The reference does not do this; it keeps the uniform
grid the fine-grid route has, so the two can be compared sample for sample.

## Accuracy

Interpolation error relative to the Fourier supersampling depends on the
signal's correlation time in integrations; at a correlation time of eight
integrations the 3-point stencil is at the 1e-3 level and the 5-point one
(`rfi.poly_interp_stencil: 2`) below 1e-4. Where the correlation time
approaches the integration time no stencil recovers what the data grid never
held, and the fine-grid route is the one to use.

The rebuilt phase is measured against `FixedOrbit`'s fine grid by the
component tests, as the phase *difference* a baseline sees: the third-order
series is within a few thousandths of a degree at 8 s integrations on the
test geometry, the first-order one a few tenths of a degree, and the fit is
insensitive to the float64 jitter of the propagation times, which is common
to every antenna at a sample and cancels in the difference.

The pipeline test `test_pipeline[PolyInterpVis]` runs the route on the
8-antenna simulation with the base defaults, against the same truth as the
`RiemannVis` case: the optimum is 1.4e-6 from that case's in reduced chi^2,
the truth metrics are identical to the printed precision, and the fp32 offset
is the same 2.3e-5. The references are recorded beside that case's in
`tests/test_tabascal_pipeline.py`.

## Configuration

Two keys, both in the `rfi` section and both with base defaults:
[`poly_interp_stencil`](config.md#rfi-signal) and
[`path_order`](config.md#rfi-signal). `data.save_rfi_per_sat` is refused by
`PolyInterpVis`: the per-satellite split re-evaluates the visibility op on the
fine-grid state, which this route does not carry.
