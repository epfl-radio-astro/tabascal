# tabascal

Trajectory-based RFI subtraction for radio interferometric data.

# Getting started with pixi

[pixi](https://pixi.sh) is the recommended way to install and develop tabascal. It manages both conda and PyPI dependencies and creates isolated environments automatically.

## Install pixi

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

## Available environments

| Environment      | Platform          | Description                           |
|------------------|-------------------|---------------------------------------|
| `default`        | linux-*, macos-*  | CPU-only runtime environment          |
| `dev`            | linux-*, macos-*  | CPU + testing and documentation tools |
| `cuda12`         | linux-*           | NVIDIA GPU (CUDA 12) runtime          |
| `cuda12-dev`     | linux-*           | NVIDIA GPU + testing and docs         |
| `cuda12-compile` | linux-*           | NVIDIA GPU + compilation toolchain    |

## Install an environment

```bash
# CPU (default)
pixi install

# Development (includes pytest, sphinx, etc.)
pixi install -e dev

# NVIDIA GPU
pixi install -e cuda12
```

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
- **A C++ compiler.** The FFI kernels are compiled from source during install
  (CMake and ninja are fetched automatically). GPU builds additionally need the
  CUDA toolkit (`nvcc`) or the ROCm/HIP toolchain on `PATH`.
- **`python-casacore`** is required at runtime but is *not* a pip dependency. On
  Linux it can be pip-installed; on **macOS installing via conda is strongly
  recommended** because `python-casacore` is difficult to build from source there
  (see the conda/mamba section).

`TABASCAL_CUDA` / `TABASCAL_ROCM` select which FFI kernel is compiled
(`libtabascal_cuda.so` / `libtabascal_hip.so`); with neither set, only the CPU
library (`libtabascal.so`) is built.

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

On macOS this conda route is the recommended way to install tabascal.

# Build the FFI shared libraries

The `RiemannVisTimeFreqCalculationFFI` component requires a compiled shared
library (`libtabascal.so`, or `libtabascal_cuda.so` for NVIDIA GPUs). These are
built automatically by CMake/scikit-build-core when the environment is created
(`pixi install`). To rebuild after editing the C++/CUDA kernels:

```bash
pixi run -e dev build-ffi                    # CPU (libtabascal.so)
pixi run -e cuda12-compile build-ffi-cuda    # NVIDIA GPU (libtabascal_cuda.so)
```

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
