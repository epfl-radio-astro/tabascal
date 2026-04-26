#!/bin/sh
# Run check_chi2_refs.py preferring the pixi dev environment, falling back to
# an active mamba/conda or venv environment if pixi is not available.
if command -v pixi > /dev/null 2>&1; then
    pixi run -e dev python ci/check_chi2_refs.py
elif [ -n "$CONDA_PREFIX" ] || [ -n "$VIRTUAL_ENV" ]; then
    python ci/check_chi2_refs.py
else
    echo "ERROR: no Python environment found (install pixi or activate a conda/venv)" >&2
    exit 1
fi
