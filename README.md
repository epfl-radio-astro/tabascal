# tabascal
New home for tabascal with all private code included.

# Getting started with pixi

[pixi](https://pixi.sh) is the recommended way to install and develop tabascal. It manages both conda and PyPI dependencies and creates isolated environments automatically.

## Install pixi

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

## Available environments

| Environment  | Platform             | Description                          |
|--------------|----------------------|--------------------------------------|
| `default`    | any                  | CPU-only runtime environment         |
| `dev`        | any                  | CPU + testing and documentation tools |
| `cuda12`     | linux-64, aarch64    | NVIDIA GPU (CUDA 12) runtime         |
| `cuda12-dev` | linux-64, aarch64    | NVIDIA GPU + testing and docs        |
| `rocm`       | linux-64             | AMD GPU (ROCm) runtime               |
| `rocm-dev`   | linux-64             | AMD GPU + testing and docs           |

## Install an environment

```bash
# CPU (default)
pixi install

# Development (includes pytest, sphinx, etc.)
pixi install -e dev

# NVIDIA GPU
pixi install -e cuda12
```

## Run tasks

```bash
# Run all tests
pixi run test

# Run component tests only
pixi run test-components

# Compile the CPU FFI shared library (required for RiemannVisTimeFreqCalculationFFI)
pixi run build-ffi

# Compile the CUDA FFI shared library
pixi run build-ffi-cuda

# Compile the HIP/ROCm FFI shared library
pixi run build-ffi-hip

# Build the documentation
pixi run docs-build
```

Tasks run in the `default` environment by default. Pass `-e <env>` to target a specific environment:

```bash
pixi run -e dev test
```

## Open a shell in the environment

```bash
pixi shell        # default environment
pixi shell -e dev # dev environment
```

# Documentation

## Build the docs

```bash
pixi run docs-build
```

After building, open `docs/_build/html/index.html` in a browser.
