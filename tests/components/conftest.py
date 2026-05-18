"""Shared helpers for component-level tests."""

import jax
import jax.numpy as jnp


def make_constants(comp):
    return {f"{comp.prefix}/{k}": v for k, v in comp.build_constants().items()}


def assert_transform_roundtrip(comp, base, L, mu, atol=1e-6):
    """Check both directions of a forward/inverse affine-Cholesky transform."""
    transformed = comp.forward_transform(base, L, mu)
    assert jnp.allclose(comp.inv_transform(transformed, L, mu), base, atol=atol)
    base2 = comp.inv_transform(base, L, mu)
    assert jnp.allclose(comp.forward_transform(base2, L, mu), base, atol=atol)
