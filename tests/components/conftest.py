"""Shared helpers for component-level tests."""

import jax
import jax.numpy as jnp


def active_precision():
    """The model.precision string matching the live ``jax_enable_x64`` setting.

    Config builders default to this so a mock config's ``precision`` always agrees
    with the precision JAX is actually running in (driven by the ``--x64`` flag).
    """
    return "double" if jax.config.read("jax_enable_x64") else "single"


def make_constants(comp):
    return {f"{comp.prefix}/{k}": v for k, v in comp.build_constants().items()}


def assert_transform_roundtrip(comp, base, L, mu, atol=None):
    """Check both directions of a forward/inverse affine-Cholesky transform.

    The default tolerance is precision-aware. The round-trip ``solve(L, L @ base)``
    loses precision in proportion to ``cond(L)``; in single precision the residual
    is O(1e-4) (vs O(1e-13) in double), so a fp64-calibrated atol would spuriously
    fail. ``matmul_precision="highest"`` does not help here — the loss is in the
    triangular solve, not a TF32 matmul.
    """
    if atol is None:
        atol = 1e-6 if active_precision() == "double" else 1e-3
    transformed = comp.forward_transform(base, L, mu)
    assert jnp.allclose(comp.inv_transform(transformed, L, mu), base, atol=atol)
    base2 = comp.inv_transform(base, L, mu)
    assert jnp.allclose(comp.forward_transform(base2, L, mu), base, atol=atol)
