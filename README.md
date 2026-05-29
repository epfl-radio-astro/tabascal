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
