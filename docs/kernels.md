# RFI-visibility kernels

The {class}`~tabascal.components.rfi_vis.RiemannVisFFI` and
{class}`~tabascal.components.rfi_vis.RiemannVisVariableFFI` components call
compiled kernels instead of the pure-JAX implementation used by
{class}`~tabascal.components.rfi_vis.RiemannVis`.

Those kernels ship in the separate
[`ri-kernels`](https://github.com/epfl-radio-astro/ri-kernels) package, a plain
runtime dependency of TABASCAL — **nothing is built from the TABASCAL
repository**, and no compiler or CUDA toolkit is needed to install TABASCAL
itself.

## Installation

The CPU kernel comes with `ri_kernels` itself, so it is already present in a
standard install.

The GPU kernel ships as an add-on wheel (`ri_kernels_cuda12` /
`ri_kernels_cuda13`, Linux only). It is pulled in by TABASCAL's `cuda12` /
`cuda13` extras, or can be installed on its own:

```bash
pip install "ri_kernels[cuda12]"
```

AMD GPUs using ROCm are supported, but may require the `ri-kernels` package to
be compiled from source.

## Enabling them

Select the component in the `model` section of the [configuration
file](config.md):

```yaml
model:
  components:
    - rfi_vis:RiemannVisFFI
```

## Memory

The Riemann sum is a reduction over a fine grid of shape `(n_bl, n_rfi,
n_freq_fine, n_time_fine)`, which is `n_rfi * n_int_freq * n_int_time` times the
size of the visibilities it produces. The compiled kernels never form it: gather,
multiply, sum over sources and the average back onto the data grid happen in one
pass, and their transpose rule recomputes the same terms from the per-antenna
inputs, so nothing of that size is kept for the backward pass either.

{class}`~tabascal.components.rfi_vis.RiemannVis` reaches the same bound by
scanning the baseline axis under `jax.checkpoint`: each step forms the fine grid
for `rfi.baseline_block_size` baselines only, averages it onto the data grid, and
lets the backward pass recompute it. The block size is the memory/recomputation
trade and does not change the result — see
[`baseline_block_size`](config.md#rfi-signal). It remains the slower of the two
components, and what it holds still grows with the number of baselines, since the
visibilities do; what the scan removes is the fine grid's baseline axis, which is
the term that carried `n_rfi * n_int_freq * n_int_time` with it.

On the reference workload of the [performance checks](performance.md) — 96
antennas on one GH200 — that is a peak of 0.70 GB in single precision against
8.68 GB without the scan, near the compiled kernel's own 0.56 GB, in exchange for
an optimiser step 1.6 times longer. The recomputation is what is being paid for
the memory. `rfi.baseline_block_size` moves the trade in either direction: a
larger block is fewer scan steps and a larger fine grid. The compiled kernel pays
neither cost and remains the faster choice where it is available.

## Precision

The kernels are compiled for both single and double precision and run in
whichever the config selects, so they impose no constraint of their own — see
[Precision](config.md#precision).
