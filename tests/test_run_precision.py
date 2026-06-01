"""Guard the single-precision (fp32) runtime toggle.

``run_tabascal``'s precision is driven by ``model.precision`` in the config via
``set_precision``. This switch has been silently dropped by a refactor before
(see PR #66), so assert it actually flips ``jax_enable_x64`` both ways.
"""

import jax
import pytest

from tabascal.scripts._run_tabascal_impl import set_precision


@pytest.fixture(autouse=True)
def restore_x64():
    """Preserve the global jax_enable_x64 flag across these tests."""
    original = jax.config.read("jax_enable_x64")
    yield
    jax.config.update("jax_enable_x64", original)


@pytest.mark.parametrize(
    "precision, expected_x64",
    [
        ("single", False),
        ("double", True),
    ],
)
def test_set_precision_toggles_x64(precision, expected_x64):
    config = {"model": {"precision": precision}}
    returned = set_precision(config)
    assert returned is expected_x64
    assert jax.config.read("jax_enable_x64") is expected_x64


def test_set_precision_defaults_to_single():
    """Missing precision (or model) defaults to single precision (x64 off)."""
    assert set_precision({"model": {}}) is False
    assert jax.config.read("jax_enable_x64") is False
    assert set_precision({}) is False
    assert jax.config.read("jax_enable_x64") is False


def test_single_disables_x64_after_sgp4jax_enabled_it():
    """sgp4jax turns x64 on at import; single precision must turn it back off."""
    import sgp4jax  # noqa: F401 (side effect: enables x64)

    jax.config.update("jax_enable_x64", True)
    assert set_precision({"model": {"precision": "single"}}) is False
    assert jax.config.read("jax_enable_x64") is False
