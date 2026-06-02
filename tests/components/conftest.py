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


def assert_transform_roundtrip(comp, base, L, mu, atol=1e-6):
    """Check both directions of a forward/inverse affine-Cholesky transform."""
    transformed = comp.forward_transform(base, L, mu)
    assert jnp.allclose(comp.inv_transform(transformed, L, mu), base, atol=atol)
    base2 = comp.inv_transform(base, L, mu)
    assert jnp.allclose(comp.forward_transform(base2, L, mu), base, atol=atol)
