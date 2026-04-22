# tabascal

Trajectory-based RFI subtraction for radio interferometric data.

# Getting started with pixi

[pixi](https://pixi.sh) is the recommended way to install and develop tabascal. It manages both conda and PyPI dependencies and creates isolated environments automatically.

## Install pixi

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

## Available environments

| Environment  | Platform             | Description                           |
|--------------|----------------------|---------------------------------------|
| `default`    | any                  | CPU-only runtime environment          |
| `dev`        | any                  | CPU + testing and documentation tools |
| `cuda12`     | linux-64, aarch64    | NVIDIA GPU (CUDA 12) runtime          |
| `cuda12-dev` | linux-64, aarch64    | NVIDIA GPU + testing and docs         |
| `rocm`       | linux-64             | AMD GPU (ROCm) runtime                |
| `rocm-dev`   | linux-64             | AMD GPU + testing and docs            |

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

The `RiemannVisTimeFreqCalculationFFI` component requires a compiled shared library. Build it with:

```bash
pixi run build-ffi        # CPU (tabascal.so)
pixi run build-ffi-cuda   # NVIDIA GPU (tabascal_cuda.so)
pixi run build-ffi-hip    # AMD GPU / ROCm (tabascal_hip.so)
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

SGP4 component tests use a bundled TLE cache (`tabascal/data/tles/`) and run without Space-Track credentials. Pipeline tests that use `SGP4LEONoDragOrbit` or `SGP4LEOOrbit` require credentials and are skipped automatically when none are found.

## Test inventory

`tests/TEST_INVENTORY.md` is a generated file — do not edit it by hand. Regenerate it after adding or modifying tests:

```bash
pixi run -e dev test-inventory
```

The script (`scripts/generate_test_inventory.py`) collects test IDs via `pytest --collect-only` and reads descriptions from each test function's docstring (first line only). The output is a Markdown table per test class with each test name linked to its source line.

When adding tests, give every test function a one-line docstring — this becomes the Description column in the inventory. New test files must be added to `FILE_ORDER` in `scripts/generate_test_inventory.py`.

```python
def test_forward_output_shape(self):
    """Forward pass produces rfi_phase with shape (n_rfi, n_ant, n_freq_fine, n_time_fine)."""
    ...
```

## Build the documentation

```bash
pixi run -e dev docs-build
```

After building, open `docs/_build/html/index.html` in a browser.
