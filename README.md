# tabascal

Trajectory-based RFI subtraction for radio interferometric data.

# Getting started with pixi

[pixi](https://pixi.sh) is the recommended way to install and develop tabascal. It manages both conda and PyPI dependencies and creates isolated environments automatically.

## Install pixi

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

## Available environments

| Environment  | Platform          | Description                           |
|--------------|-------------------|---------------------------------------|
| `default`    | linux-*, macos-*  | CPU-only runtime environment          |
| `dev`        | linux-*, macos-*  | CPU + testing and documentation tools |
| `cuda12`     | linux-*           | NVIDIA GPU (CUDA 12) runtime          |
| `cuda12-dev` | linux-*           | NVIDIA GPU + testing and docs         |

## Install an environment

```bash
# CPU (default)
pixi install

# Development (includes pytest, sphinx, etc.)
pixi install -e dev

# NVIDIA GPU — installs CPU-only; the CUDA FFI kernel must be built separately
pixi install -e cuda12
pixi run -e cuda12 build-ffi-cuda     # see "Build the FFI shared libraries"
```

`pixi install` builds only the CPU FFI library; the GPU kernel needs the extra
`build-ffi-cuda` step (explained under "Build the FFI shared libraries"). Verify
any environment with `pixi run -e <env> check-install`.

## Open a shell in the environment

```bash
pixi shell          # default environment
pixi shell -e dev   # dev environment
```

# Installing without pixi

pixi is recommended, but tabascal can also be installed with plain `pip` or into
a conda/mamba environment.

Prerequisites:
- **Python 3.10–3.13.** The build pins `jax==0.6.0` for the minimum FFI ABI, and
  that jaxlib has no wheels for Python 3.14.
- **A C++20-capable compiler** (GCC ≥10, Clang ≥10, or Apple Clang ≥13). The
  FFI kernels are compiled from source during install (CMake and ninja are
  fetched automatically). GPU builds additionally need the CUDA 12 or 13
  toolkit (`nvcc`) or the ROCm/HIP toolchain on `PATH`.
- **`python-casacore`** is required at runtime but is *not* a pip dependency.
  It is pip-installable on **linux-x86_64**; on **macOS and linux-aarch64**,
  installing via conda is strongly recommended because `python-casacore` is
  difficult to build from source on those platforms (see the conda/mamba
  section).

`TABASCAL_CUDA` / `TABASCAL_ROCM` select which FFI kernel is compiled
(`libtabascal_cuda.so` / `libtabascal_hip.so`); with neither set, only the CPU
library (`libtabascal.so`) is built. GPU kernels target Ampere + Hopper (CUDA
`sm_80`, `sm_90`) and MI200/MI300 + RDNA2/3 (HIP `gfx90a`, `gfx942`, `gfx1030`,
`gfx1100`) by default — override `CMAKE_CUDA_ARCHITECTURES` /
`CMAKE_HIP_ARCHITECTURES` (e.g. via `CMAKE_ARGS`) to target other GPUs.

## pip

Run these from a clone of the repository (`pip install .`), or replace `.` with
`git+https://github.com/epfl-radio-astro/tabascal.git`.

```bash
# CPU
pip install .

# NVIDIA GPU — requires the CUDA toolkit (nvcc) on PATH.
TABASCAL_CUDA=1 pip install ".[cuda12]"     # or ".[cuda13]" for CUDA 13

# AMD GPU (ROCm) — first install a ROCm-compatible jax/jaxlib by following the
# JAX install guide: https://docs.jax.dev/en/latest/installation.html
# then build the ROCm kernels (requires the ROCm/HIP toolchain on PATH):
TABASCAL_ROCM=1 pip install .
```

## conda / mamba

Use conda to provide `python-casacore` (and a C++ compiler), then pip-install
tabascal into the activated environment:

```bash
mamba create -n tabascal -c conda-forge "python>=3.10,<3.14" python-casacore cxx-compiler
mamba activate tabascal

# CPU
pip install .

# NVIDIA GPU — also bring the CUDA toolchain into the environment:
mamba install -c conda-forge cuda-nvcc cuda-cudart-dev
TABASCAL_CUDA=1 pip install ".[cuda12]"
```

On macOS and linux-aarch64 this conda route is the recommended way to install tabascal.

# Build the FFI shared libraries

The `RiemannVisTimeFreqCalculationFFI` component requires a compiled shared
library: `libtabascal.so` on CPU, or `libtabascal_cuda.so` for NVIDIA GPUs
(`RiemannVisTimeFreqCalculationFFI` only runs in double precision — set
`model.precision: double`). `pixi install` builds the **CPU** library
automatically.

The **GPU** kernel is *not* built by a plain `pixi install`: pixi/uv build the
editable `tabascal` package once and share that single (CPU) build across every
environment, so the cuda12 env inherits the CPU-only library. Build the CUDA
kernel explicitly after installing:

```bash
pixi install -e cuda12                # base env (gets the shared CPU build)
pixi run -e cuda12 build-ffi-cuda     # force the cuda12-specific CUDA build
```

`build-ffi-cuda` (`pixi reinstall -e cuda12 tabascal`) rebuilds `tabascal` in the
cuda12 environment, where `TABASCAL_CUDA=1` is active, producing
`libtabascal_cuda.so`. Use the same task to rebuild after editing the C++/CUDA
kernels (`pixi run -e dev build-ffi` for the CPU library).

Verify an environment is wired correctly — the FFI libraries load and the
CPU/GPU kernels actually execute — with:

```bash
pixi run -e cuda12 check-install      # or -e default / -e dev
```

# Precision

tabascal runs in **single precision (fp32) by default**. Set it in the config:

```yaml
model:
  precision: single   # default; or "double" for fp64
```

- **`single` (default).** Halves device-memory use (~2×), which raises the
  largest problem size that fits on a GPU. On GPUs with first-class fp64 (e.g.
  Hopper/GH200) it is **not** faster in wall-clock — the win is memory capacity,
  not speed.
- **`double`.** Required by some components, and recommended when fitting
  satellite trajectories (the differentiable orbit models need fp64 accuracy).

The following components run in **double precision only** and raise a clear error
under `single` (set `model.precision: double` to use them):

- `trajectory:PhaseCalculationRFI`
- `trajectory:SGP4LEONoDragOrbit`
- `trajectory:SGP4LEOOrbit`
- `rfi_vis:RiemannVisTimeFreqCalculationFFI` (the FFI kernel is compiled for
  complex128)

The non-FFI `rfi_vis:RiemannVisTimeFreqCalculation` and the GP astronomical/gains
components run in either precision.

# Developer

## Running tests

```bash
pixi run -e dev test               # all tests
pixi run -e dev test-components    # component tests only
```

To run a single file or test, open a dev shell first:

```bash
pixi shell -e dev
pytest tests/components/test_gains.py
pytest tests/components/test_gains.py::TestGPGains::test_forward_output_shapes
```

SGP4 component tests use a bundled TLE cache (`tabascal/data/tles/`) and run without Space-Track credentials. 

## Build the documentation

```bash
pixi run -e dev docs-build
```

After building, open `docs/_build/html/index.html` in a browser.

## Debugging a performance regression locally

When a performance regression is detected in CI, reproduce it on a smaller dataset with 8
antennas (faster to simulate, runs on any dev machine):

```bash
# Generate an 8-antenna simulation from the standard 96A config
sim-vis -c ci/reframe/data/sim_target_96A.yaml -a 8

# Run tabascal with timing output against the generated dataset
tabascal -c ci/reframe/data/tab_target.yaml \
         -s ci/reframe/data/data/pnt_src_obs_08A_090T-0000-0890_001I_001F-1.500e+08-1.500e+08_050PAST_000GAST_000EAST_32SAT_0GRD_1.0e+00RFI \
         -t
```

The `-t` flag prints a per-function timing table identical to the CI output.
