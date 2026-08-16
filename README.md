# tabascal

[![Tests](https://github.com/epfl-radio-astro/tabascal/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/epfl-radio-astro/tabascal/actions/workflows/test.yaml)
[![Documentation Status](https://readthedocs.org/projects/tabascal/badge/?version=latest)](https://tabascal.readthedocs.io/en/latest/)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/epfl-radio-astro/tabascal/badges/coverage.json)](https://github.com/epfl-radio-astro/tabascal/actions/workflows/test.yaml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/epfl-radio-astro/tabascal/blob/main/pyproject.toml)
[![MNRAS 2023](https://img.shields.io/badge/MNRAS%202023-10.1093%2Fmnras%2Fstad1979-blue)](https://doi.org/10.1093/mnras/stad1979)
[![A&A 2025](https://img.shields.io/badge/A%26A%202025-10.1051%2F0004--6361%2F202554596-blue)](https://doi.org/10.1051/0004-6361/202554596)

Trajectory-based RFI subtraction for radio interferometric data.

![Starlink subtraction on EDA2 data](docs/images/eda2_starlink_subtraction.svg)

A result on real **EDA2** data: a 151 MHz observation (XX) crossed by several Starlink satellites. The satellite trails dominating the field (a) are gone after subtraction (b), leaving the inferred sky (c). Reversing the split shows what was removed — subtracting the sky instead (e) isolates the trails, which the model reconstructs as the satellite signal (f). The final residual (d) is noise-like apart from the marked features. Note the two rows use different flux scales: the satellite signal is a few hundred Jy/beam against a sky spanning roughly -400 to 1000 Jy/beam.

## Citing tabascal

- Finlay, Bassett & Kunz (2023), *Trajectory-based RFI subtraction and calibration for radio interferometry*, MNRAS — [10.1093/mnras/stad1979](https://doi.org/10.1093/mnras/stad1979)
- Finlay, Bassett & Kunz (2025), *TABASCAL: Removing multi-satellite interference from radio interferometry observations*, A&A — [10.1051/0004-6361/202554596](https://doi.org/10.1051/0004-6361/202554596)

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

tabascal is pure Python — no compiler or CUDA toolkit is needed to install it.

Prerequisites:
- **Python 3.10–3.13.** 3.14 has not been validated against the pinned
  jax/jaxlib versions.
- **`python-casacore`** is required at runtime but is *not* a pip dependency.
  It is pip-installable on **linux-x86_64**; on **macOS and linux-aarch64**,
  installing via conda is strongly recommended because `python-casacore` is
  difficult to build from source on those platforms (see the conda/mamba
  section).

## pip

Run these from a clone of the repository (`pip install .`), or replace `.` with
`git+https://github.com/epfl-radio-astro/tabascal.git`.

```bash
# CPU
pip install .

# NVIDIA GPU (Linux only)
pip install ".[cuda12]"     # or ".[cuda13]" for CUDA 13
```

## conda / mamba

Use conda to provide `python-casacore`, then pip-install tabascal into the
activated environment:

```bash
mamba create -n tabascal -c conda-forge "python>=3.10,<3.14" python-casacore
mamba activate tabascal

# CPU
pip install .

# NVIDIA GPU (Linux only)
pip install ".[cuda12]"
```

On macOS and linux-aarch64 this conda route is the recommended way to install tabascal.

# RFI-visibility kernels

The `RiemannVisFFI` and `RiemannVisVariableFFI` components call compiled
kernels that ship in the separate
[`ri-kernels`](https://github.com/epfl-radio-astro/ri-kernels) package, a plain
runtime dependency of tabascal — nothing is built from this repository. The CPU
kernel comes with `ri_kernels` itself; the GPU kernel ships as an add-on wheel
(`ri_kernels_cuda12` / `ri_kernels_cuda13`, Linux only) pulled in by tabascal's
`cuda12` / `cuda13` extras, or installable on its own:

```bash
pip install "ri_kernels[cuda12]"
```

The kernels are compiled for both single and double precision and run in
whichever the config selects. Enable them in the config with:

```yaml
components:
  - rfi_vis: RiemannVisFFI
```

Note: AMD GPUs using ROCm are supported, but may require the "ri-kernels" package
to be compiled from source.

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
- `trajectory:NoDragOrbit`
- `trajectory:Orbit`

Both `rfi_vis` kernels (`RiemannVis` and the FFI `RiemannVisFFI`) and the GP
astronomical/gains components run in either precision.

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

The documentation is also published on Read the Docs, one version per release tag
plus `latest` from `main`, selectable from the version flyout on every page.
`docs/readthedocs.md` covers the version scheme, how to publish a
release or an extra branch, and the warnings-as-errors build that CI and Read the
Docs both run.

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
