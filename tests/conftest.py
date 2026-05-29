"""Top-level pytest configuration for tabascal tests.

Controls JAX float64 (double precision) globally via the ``--x64`` flag so the
in-process unit tests can be run in either precision. Defaults to ``true`` to
preserve the historical behaviour. Set ``--x64 false`` to exercise the tests in
single precision. (Pipeline tests run ``run_tabascal`` in a subprocess and pick
their precision from the config, so they are unaffected by this flag.)
"""

import jax


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
