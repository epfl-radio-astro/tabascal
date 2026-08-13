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

The `RiemannVisTimeFreqCalculationFFI` and `RiemannVisTimeFreqVariableFFI`
components call compiled kernels that ship in the separate
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
  - rfi_vis: RiemannVisTimeFreqCalculationFFI
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
- `trajectory:SGP4LEONoDragOrbit`
- `trajectory:SGP4LEOOrbit`

Both `rfi_vis` kernels (`RiemannVisTimeFreqCalculation` and the FFI
`RiemannVisTimeFreqCalculationFFI`) and the GP astronomical/gains components run
in either precision.

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
