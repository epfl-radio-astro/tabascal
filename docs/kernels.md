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
    - rfi_vis: RiemannVisFFI
```

## Precision

The kernels are compiled for both single and double precision and run in
whichever the config selects, so they impose no constraint of their own — see
[Precision](config.md#precision).
