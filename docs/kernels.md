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

## What they compute

Both components call the *delay* kernel, `RFIDelayVisOp`. It takes the
per-antenna RFI amplitudes on the fine time/frequency grid together with the
compact geometric delays `rfi_delay_us` — shaped `(n_rfi, n_ant, n_time_fine)`,
with no frequency axis — and the fine frequency grid in MHz, and forms the
baseline phase `2π f (τ₁ − τ₂)` inside the kernel. Nothing the size of the
former per-frequency phase array, `(n_rfi, n_ant, n_freq_fine, n_time_fine)`,
is ever materialised; the delay input is smaller than it by a factor of
`n_freq × freq_int_samples`.

The delays are computed once, in double precision, by the trajectory
component and centred across antennas per source and time sample before
being cast to the run's precision. Only delay differences enter a baseline,
so this is exact, and it is what keeps single precision usable: the kernel's
float32 path is specified for arrays up to about 10 km across (`|Δτ| ≲ 33 μs`),
where the worst-case phase resolution at 1 GHz is about 0.025 rad. Use double
precision for arrays approaching 100 km.

## Installation

The CPU kernel comes with `ri_kernels` itself, so it is already present in a
standard install. The delay kernels need `ri_kernels >= 0.3.0`; with an older
release the FFI components fail at setup with a message saying so, while the
pure-JAX `RiemannVis` keeps working.

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
    - rfi_vis: RiemannVisFFI
```

## Precision

The kernels are compiled for both single and double precision and run in
whichever the config selects, so they impose no constraint of their own — see
[Precision](config.md#precision).
