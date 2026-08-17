# Developer Install

This page covers setting up a development environment. If you only want to *use*
TABASCAL, follow the [installation instructions in the usage
guide](usage.md#installation) instead.

## pixi (recommended for development)

[pixi](https://pixi.sh) is the recommended way to develop TABASCAL. It manages
both conda and PyPI dependencies and creates isolated environments
automatically, so it resolves `python-casacore` for you rather than leaving it
as a manual conda step.

### Install pixi

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

### Available environments

| Environment  | Platform          | Description                           |
|--------------|-------------------|---------------------------------------|
| `default`    | linux-*, macos-*  | CPU-only runtime environment          |
| `dev`        | linux-*, macos-*  | CPU + testing and documentation tools |
| `cuda12`     | linux-*           | NVIDIA GPU (CUDA 12) runtime          |
| `cuda12-dev` | linux-*           | NVIDIA GPU + testing and docs         |

### Install an environment

```bash
git clone git@github.com:epfl-radio-astro/tabascal.git
cd tabascal

# CPU (default)
pixi install

# Development (includes pytest, sphinx, etc.)
pixi install -e dev

# NVIDIA GPU
pixi install -e cuda12
```

### Open a shell in the environment

```bash
pixi shell          # default environment
pixi shell -e dev   # dev environment
```

## Editable install without pixi

An editable install into an existing environment also works. Provide
`python-casacore` first, as described in the [usage
guide](usage.md#installation):

```bash
git clone git@github.com:epfl-radio-astro/tabascal.git
pip install -e ./tabascal[dev]
```

The `dev` extra pulls in the `test` and `docs` extras along with `pre-commit`
and `reframe-hpc`.

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

SGP4 component tests use a bundled TLE cache (`tabascal/data/tles/`) and run
without Space-Track credentials.

The end-to-end pipeline tests, which compare inference results against recorded
reference values, are described in [Pipeline tests](pipeline_tests.md).

## Building the documentation

```bash
pixi run -e dev docs-build
```

After building, open `docs/_build/html/index.html` in a browser.

Without pixi, run `sphinx-build` directly from the base directory of the
repository:

```bash
sphinx-build -b html docs docs/_build/html
```

The documentation is also published on Read the Docs, one version per release
tag plus `latest` from `main`, selectable from the version flyout on every page.
[Read the Docs setup](readthedocs.md) covers the version scheme, how to publish
a release or an extra branch, and the warnings-as-errors build that CI and Read
the Docs both run.
