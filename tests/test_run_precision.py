"""Guard the single-precision (fp32) runtime toggle.

``run_tabascal``'s precision is driven by ``model.precision`` in the config via
``set_precision``. This switch has been silently dropped by a refactor before
(see PR #66), so assert it actually flips ``jax_enable_x64`` both ways.
"""

import jax
import pytest

from tabascal.scripts._run_tabascal_impl import (
    assert_precision_supported,
    set_precision,
)
from tabascal.validation import resolve_components


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


# ---------------------------------------------------------------------------
# assert_precision_supported — fast-fail preflight before TabConfig setup
# ---------------------------------------------------------------------------

# The differentiable orbit/phase trajectory components are the genuinely
# double-only ones (two, so the "every offender" test can prove it lists all).
# The FFI RFI-vis kernel is built for both precisions, so it is the single-ok
# representative here — which also guards against it regressing to double-only.
_DOUBLE_ONLY = ["trajectory:SGP4LEOOrbit", "trajectory:PhaseCalculationRFI"]
_SINGLE_OK = ["rfi_vis:RiemannVisTimeFreqCalculationFFI"]


def preflight(config):
    """Run the check the way ``check_components`` does: on resolved classes.

    The classes are resolved once per run and shared between this check, the
    per-component config validation and ``Model``.
    """
    return assert_precision_supported(config, resolve_components(config))


def test_preflight_single_rejects_double_only_component():
    """A single-precision config using a double-only component raises, naming it."""
    config = {"model": {"precision": "single", "components": _DOUBLE_ONLY[:1] + _SINGLE_OK}}
    with pytest.raises(ValueError, match="SGP4LEOOrbit"):
        preflight(config)


def test_preflight_reports_every_offender():
    """The error lists all offending components, not just the first."""
    config = {"model": {"precision": "single", "components": _DOUBLE_ONLY}}
    with pytest.raises(ValueError) as exc:
        preflight(config)
    msg = str(exc.value)
    assert "SGP4LEOOrbit" in msg and "PhaseCalculationRFI" in msg


def test_preflight_single_allows_single_capable_components():
    """Single precision with only single-capable components passes."""
    preflight({"model": {"precision": "single", "components": _SINGLE_OK}})


def test_preflight_double_allows_double_only_component():
    """Double precision allows the double-only components."""
    preflight({"model": {"precision": "double", "components": _DOUBLE_ONLY}})


def test_preflight_defaults_and_empty_components_are_noops():
    """Missing precision defaults to single; empty/absent components never raise."""
    preflight({"model": {"components": _SINGLE_OK}})
    preflight({"model": {}})
    preflight({})
