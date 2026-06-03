"""Top-level pytest configuration for tabascal tests.

Controls JAX float64 (double precision) globally via the ``--x64`` flag so the
in-process unit tests can be run in either precision. Defaults to ``true`` to
preserve the historical behaviour. Set ``--x64 false`` to exercise the tests in
single precision. (Pipeline tests run ``run_tabascal`` in a subprocess and pick
their precision from the config, so they are unaffected by this flag.)

Some components only work in double precision (the SGP4/phase trajectory
components, which need fp64 for the orbit propagation). Mark their tests with
``@pytest.mark.requires_double``; they are skipped under ``--x64 false`` instead
of erroring on the components' precision gate. (The FFI RFI-vis kernel is built
for both complex64 and complex128, so it is exercised in either precision.)
"""

import jax
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--x64",
        action="store",
        default="true",
        choices=("true", "false"),
        help="Enable JAX float64 (double precision) in tests. Default: true.",
    )


def pytest_configure(config):
    # sgp4jax enables x64 at import time. Import it here so that one-time side
    # effect fires now; our setting below then wins, and later (cached) imports
    # of sgp4jax during collection/tests won't re-enable it.
    import sgp4jax  # noqa: F401

    jax.config.update("jax_enable_x64", config.getoption("--x64") == "true")
    # Mirror run_tabascal's set_precision: pin true fp32 matmuls. On Ampere+ GPUs
    # JAX defaults f32 matmuls to TF32 (~10-bit mantissa), which wrecks the
    # single-precision linear algebra; "highest" forces real fp32 (no-op under
    # x64). Without this the tests would not match the production precision.
    jax.config.update("jax_default_matmul_precision", "highest")
    config.addinivalue_line(
        "markers",
        "requires_double: test only runs under double precision (skipped with --x64 false)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip ``requires_double`` tests when the session runs in single precision."""
    if config.getoption("--x64") == "true":
        return
    skip_single = pytest.mark.skip(reason="requires double precision; not run under --x64 false")
    for item in items:
        if "requires_double" in item.keywords:
            item.add_marker(skip_single)


@pytest.fixture(scope="session")
def precision(pytestconfig):
    """Session precision string ('double'/'single') from the --x64 flag."""
    return "double" if pytestconfig.getoption("--x64") == "true" else "single"
