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
`rfi_delay_poly_us` of shape `(n_rfi, n_ant, n_time, rfi.path_order + 1)`, the
geometric delay from the source to each antenna (its range plus the antenna's
`w`, over `c`, in microseconds, relative to the array mean, with the sign
that makes the phase `2 pi f tau` as the fine-grid route's `rfi_delay_us` has
it) and its first `rfi.path_order` time derivatives at each cell centre, from
a least-squares polynomial through the cell's fine samples. Both are float64
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
| `rfi_A` | `(n_rfi, n_ant, n_freq, n_time)` complex | the signal on the data grid; differentiated, as are the phase and the delay |
| `rfi_phase` | `(n_rfi, n_ant, n_freq, n_time)` | the phase at the channel and cell centre, reduced to one turn |
| `rfi_delay_poly_us` | `(n_rfi, n_ant, n_time, n_path)` | the geometric delay relative to the array mean (microseconds) and its time derivatives (microseconds per second^k) at the cell centre |
| `w_freq`, `start_freq` | `(n_freq, n_sf, n_int_freq)`, `(n_freq,)` | interpolation weights across each channel, and each stencil's first channel |
| `w_time`, `start_time` | `(n_time, n_st, n_int_time)`, `(n_time,)` | the same across each cell |
| `dnu_mhz`, `dt` | `(n_int_freq,)`, `(n_int_time,)` | the fine offsets from the channel and cell centres (MHz, s) |
| `freqs_mhz` | `(n_freq,)` | channel centres (MHz) |
| `a1`, `a2` | `(n_bl,)` | the antennas of each baseline |
| **output** `vis_rfi` | `(n_bl, n_freq, n_time)` complex | |

For one cell `(f, t)` and one of its fine samples `(u, v)`:

```
A[r, a](u, v)   = sum_k sum_l w_freq[f, k, u] w_time[t, l, v] rfi_A[r, a, start_freq[f] + k, start_time[t] + l]
dtau[r, a](v)   = sum_{k >= 1} rfi_delay_poly_us[r, a, t, k] dt[v]^k / k!
phi[r, a](u, v) = rfi_phase[r, a, f, t] + 2 pi ((freqs_mhz[f] + dnu_mhz[u]) dtau[r, a](v) + dnu_mhz[u] rfi_delay_poly_us[r, a, t, 0])
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
- **Derivatives are structural.** The result is bilinear in the fine samples
  `S`, `S` is linear in `rfi_A`, and `S` changes as `i S` per unit of phase,
  so every derivative the kernel needs is a small step from the products it
  already forms. The JVP is `B(dS, S) + B(S, dS)` with `dS` the signal
  tangent pushed through the same interpolation and phase factor, plus
  `i dphi S` for a phase or delay tangent, `dphi` being the phase formula on
  the tangents. The VJP scatters the visibility cotangent to the fine samples
  of each baseline's two antennas, each weighted by the other antenna's
  sample, exactly as the fine-grid kernel's transpose does: the signal's
  cotangent is that factor times the conjugate phase factor pushed back
  through the weight tables, a stencil-sized scatter-add onto the data grid;
  the phase's is the imaginary part of the sample times the same factor,
  summed over the cell's samples, and the delay's is that weighted by the
  phase's derivative in each polynomial coefficient. Nothing of the fine
  grid survives. The compiled operator carries two kernel pairs, one for the
  signal alone and one with the phase and delay as well, and its JVP rule
  picks the first whenever the phase and delay tangents are symbolic zeros:
  a fixed orbit computes no phase derivative, a fitted one gets it without
  a switch.
- **Precision is arranged, not hoped for.** The unreduced phase
  `2 pi f tau` is of order a million turns, which float32 cannot hold to a
  fraction of a turn. `rfi_phase` carries it reduced, computed in float64 on
  the host; the kernel adds only the change across the cell and across the
  channel, in MHz times microseconds, which is cycles. A kernel must not
  rebuild `rfi_phase` from `rfi_delay_poly_us[..., 0]`. And that change is
  small only because the delay is relative to the array mean: a term common
  to every antenna cancels in a baseline's phase difference, so
  `FixedOrbitCoarse` subtracts the mean over antennas at each fine sample
  before fitting, leaving microseconds and tens of nanoseconds per second
  where the full delay would change by 1e4 wavelengths across a cell, beyond
  float32's reach. This is the convention of the fine-grid route's
  `rfi_delay_us` (PR #144).

A compiled kernel is validated as `ri_kernels` is: value, JVP and VJP against
this function. `tests/test_coarse_rfi_vis.py` holds the function itself to the
fine-grid sum on the fine grid it forms, and its derivatives to finite
differences; `tests/components/test_coarse_components.py` holds the three
components to their fine-grid twins on one configuration.

## The compiled operator

`rfi_vis:PolyInterpVisFFI` is the same component with the one function
replaced by `ri_kernels.jax_api.RFIInterpVisOp`: CPU and GPU kernels carrying
the primal, the JVP and the transpose, from the `interp-vis` branch of
[ri-kernels](https://github.com/epfl-radio-astro/ri-kernels). The operator's
inputs are the reference function's, with the antenna axis first (the
component transposes the three data-grid arrays on the way in, data-grid
sized), and its tests hold it to a JAX transcription of that function in
value, forward mode and reverse mode. A release of `ri_kernels` without the
operator is refused at setup.

```yaml
model:
  components:
    - trajectory:FixedOrbitCoarse
    - rfi_signal:ComplexRFIVarAntCoarse
    - rfi_vis:PolyInterpVisFFI
    - ast_vis:GPVisAst
    - gains:UnitaryGains
```

The GPU kernels first materialise the fine samples of a chunk of time cells,
once per cell and antenna, with contiguous reads of the data-grid arrays: a
kernel that rebuilds them where they are used gathers a dozen scattered
values per sample and repeats that once per tile pair the antenna sits in,
and that traffic, not the products, was the cost. The antennas are then cut
into tiles of 32, and a block works one unordered pair of tiles on one cell:
per source and chunk of samples it loads the two tiles' samples into shared
memory, contiguous runs, and forms the tile pair's baseline products from
there, each pair once whatever orderings the baseline list holds. The
transpose is the same block structure as the forward's mirror: from the tile
pair's matrix of cotangent weights it forms each tile's cotangent samples as
two small dense products, turns them by the antennas' phase factors and
pushes them through the stencil weights into a per-tile-pair partial, which
a gather sums per antenna in a fixed order. Every element is owned by one
thread: no atomics, and the result is deterministic. Shared memory stays
within the 48 KB every device offers whatever the stencil, and the scratch
(samples, phase factors, partials) is chunked over time cells to at most a
few hundred MB. In single precision the phase change across a cell,
`2 pi freqs L_1 dt / c`, is of order 1e4 rad per antenna at orbital range
rates and rounds at ~1e-3 rad, in the kernel as in the pure-JAX reference;
the operator's tests hold the single-precision kernels to the float64
reference at that level.

The CPU kernels are the same idea without the tiles. A task takes a run of
time cells (the transpose, one source and a run of output cells), and for
each cell, channel and source it builds the fine samples of every antenna
once; a baseline is then a dot product of its two antennas' samples in the
forward, or a scatter of its cotangent onto them in the transpose. The
per-cell work is vectorised with [Highway](https://github.com/google/highway)
and dispatched at run time to whatever the machine supports, as the fine-grid
kernels are. The fine samples are the only axis long enough to vectorise: the
cell's interpolation weights, channel offsets and time-offset powers are
built once per cell, the per-antenna and per-source quantities are broadcast
scalars against them, and the sample buffers are split into real and
imaginary parts and padded to whole vectors, so a load is a load rather than
a load and a shuffle and no loop needs a scalar tail. In the transpose every
task writes a disjoint slab of each output, recomputing the few cells on
either side of its run that its stencils reach, so the result is
deterministic with one parallel pass and no accumulation across tasks.


Measured on the SKA-Low scaling simulations (150 integrations of 2 s, 32
satellites, `rfi.time_int_factor: 1`, so 37, 59 and 174 fine samples per
integration at 64, 128 and 256 antennas), all three routes from one checkout
and one environment:

| 1 GH200, 100 iterations, single precision | RiemannVisFFI (fine grid) | PolyInterpVis (pure JAX) | PolyInterpVisFFI (staged kernels) |
|---|---|---|---|
| 8 ch, 64 A: optimiser | 15.8 s | 40.3 s | 8.4 s |
| 8 ch, 64 A: peak memory | 6.14 GB | 0.64 GB | 0.77 GB |
| 8 ch, 128 A: optimiser | 59.0 s | 270.9 s | 28.6 s |
| 8 ch, 128 A: peak memory | 19.8 GB | 2.37 GB | 2.37 GB |
| 8 ch, 256 A: optimiser | out of memory | (not run) | 205 s |
| 8 ch, 256 A: peak memory | 92.6 GB requested | | 8.87 GB |

The operator keeps the data-grid route's memory, a tenth of the fine-grid
kernel's, at twice that kernel's speed and five to nine times the pure-JAX
reference's. All three reach the same optimum. The operator's own forward
and VJP at these sizes (32 sources, 8 channels, 150 integrations) take 7 and
16 ms at 64 antennas and 32 and 75 ms at 128 on the GH200, and 1.4 and 2.5 ms
at 64 antennas on a GTX 1060; the first staged kernels, which rebuilt the
samples in place, took 22 and 70 ms at 64 antennas on the GH200. Those VJPs
are the signal-only ones a fixed orbit binds; the full VJP with the phase and
delay cotangents, which a fitted trajectory would bind, takes 23 and 109 ms
at 64 and 128 antennas on the GH200 and 3.8 ms at 64 on the GTX 1060.

On CPU, where some groups will run this, the same three routes on the
64-antenna simulation (20 optimiser iterations, single precision):

| 8 ch, 64 A, 20 iterations | RiemannVisFFI (fine grid) | PolyInterpVis (pure JAX) | PolyInterpVisFFI (operator) |
|---|---|---|---|
| Grace, 72 cores (Neoverse-V2) | 143.7 s | 139.2 s | 17.7 s |
| Apple M4, 10 cores | 134.7 s | 203.2 s | 16.3 s |
| Intel i5-8400, 6 cores (AVX2) | 352.2 s | 593.7 s | 53.0 s |

At 128 antennas on the Grace CPU the same 20 iterations take 533.4 s on the
fine-grid kernel against 50.2 s on the operator, so the ratio widens with the
array as the fine grid does: it was 8.1 at 64 antennas and is 10.6 here.

Before this arrangement the operator's CPU kernels rebuilt each antenna's
samples once per baseline and were not vectorised, and the same runs took
562 s on the M4 and 1770 s on the i5: the operator was several times slower
than the fine-grid kernel on CPU where it is now some eight times faster.
Against the GPU, the gap the operator has to make up is far smaller than the
fine-grid route's. On one Grace-Hopper node, the same 20 iterations take
6.1 s on the GH200 against 17.7 s on the node's own 72 CPU cores, where the
fine-grid route takes 5.5 s against 143.7 s; per optimiser iteration, once
the compilation both share is discounted, the GH200 is about 24 times the
CPU. A GTX 1060 runs the operator in 15.3 s against 53.0 s on the six-core
i5 beside it, and cannot run the fine-grid kernel at all at this size, which
asks for more memory than the card has.


## Variable sampling per baseline

`rfi_vis:PolyInterpVisVariable` and `rfi_vis:PolyInterpVisVariableFFI` are the
data-grid twins of `RiemannVisVariable` and its FFI form: the baselines are
grouped by the fringe rate they need to resolve (`rfi.min_time_bins`,
`rfi.max_time_bins`, the same estimate `TabConfig` makes for the fine-grid
components), and a group with stride `s` integrates every `s`-th fine sample
of the cell. On the data grid that needs little: the tables are per fine offset,
so a group is the same function, or the same operator, called on the group's
baselines with the rows of every `s`-th offset of the time tables. The
variable sampling is a difference in inputs only.

Whether it pays is another matter. A stride per baseline inside the kernel
was built and measured against per-group calls and against full sampling, on
a GTX 1060 and a GH200, at 64 and 128 antennas with two stride distributions:
neither variable route beat full sampling. The products a stride saves are
the cheap part of the kernels once the samples are materialised, so
in-kernel striding costs more bookkeeping than it saves, and per-group calls
rebuild the samples once per group. The stride machinery was taken out of
the kernels again; the per-group component stays as the way to sample
variably should a cheaper sample build make it worthwhile.

| 1 GH200, 100 iterations, single precision, variable sampling | RiemannVisVariableFFI (fine grid) | PolyInterpVisVariable (pure JAX) | PolyInterpVisVariableFFI (operator per group) | PolyInterpVisFFI (full sampling) |
|---|---|---|---|---|
| 8 ch, 64 A: optimiser | 11.9 s | 27.9 s | 8.7 s | 8.4 s |
| 8 ch, 128 A: optimiser | 41.8 s | 115.0 s | 41.3 s | 28.6 s |

The fine-grid and pure-JAX routes gain from the coarser sampling of the slow
baselines because their cost is in the samples; the operator's is not, and
its per-group form pays for the samples rebuilt per group. (The divisor-rich
fine grid the grouping needs also has slightly more samples: 40 and 60
against 37 and 59.)

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
