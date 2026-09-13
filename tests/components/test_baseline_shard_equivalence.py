"""Sharding the baseline axis must compute what one device computes.

Each device is handed the whole per-antenna signal and only its own rows of
the visibility array, and nothing is summed across devices afterwards. What
that has to produce is exactly what a single device produces, and this is the
test that says so -- through the real component, on four of them.

The device count is fixed before jax starts, so these run only when the suite
was launched with ``XLA_FLAGS=--xla_force_host_platform_device_count=4``.
"""
from types import SimpleNamespace

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import tabascal.distributed as dist
from tabascal.components import rfi_vis
from tabascal.components.rfi_vis import PolyInterpVisVariableFFI, eval_with_indices
from .test_poly_interp_vis_variable import _component_call

pytestmark = [
    pytest.mark.skipif(
        jax.device_count() < 4,
        reason="needs four devices: XLA_FLAGS=--xla_force_host_platform_device_count=4",
    ),
    pytest.mark.skipif(
        eval_with_indices is None or rfi_vis.RFIInterpVisOp is None,
        reason="needs ri_kernels from interp-shardable",
    ),
]


def _case(n_ant=8, n_bl=8, requirements=None):
    """A case whose baseline count divides over four devices.

    ``requirements`` splits the baselines into groups with different sampling,
    so the per-device counts come out uneven and the ghost padding is
    exercised rather than skipped.
    """
    pairs = np.array([(i, j) for i in range(n_ant) for j in range(i + 1, n_ant)])[:n_bl]
    if requirements is None:
        requirements = np.where(np.arange(n_bl) < 3, 30, 2)
    cfg = SimpleNamespace(
        n_ant=n_ant, n_bl=n_bl, n_freq=3, n_time=4, n_int_freq=3, n_int_time=31,
        a1=pairs[:, 0].astype(np.int32), a2=pairs[:, 1].astype(np.int32),
        int_time=2., chan_width=1e6, freqs=np.array([1.4e9, 1.401e9, 1.402e9]),
        args={"rfi": {}}, rfi_time_requirements=np.asarray(requirements),
    )
    rng = np.random.default_rng(7)
    shape = (2, n_ant, cfg.n_freq, cfg.n_time)
    state = {
        "rfi_A": jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape)),
        "rfi_phase": jnp.asarray(rng.normal(size=shape)),
        "rfi_delay_poly_us": jnp.asarray(rng.normal(scale=1e-5, size=(2, n_ant, cfg.n_time, 3))),
        "vis_rfi": jnp.ones((n_bl, cfg.n_freq, cfg.n_time), dtype=complex),
    }
    return cfg, state


def _unsharded(monkeypatch, cfg, state):
    """One device's answer: no map, no padding, no ghosts."""
    monkeypatch.setattr(rfi_vis, "sharding_baselines", lambda: False)
    monkeypatch.setattr(rfi_vis, "psum_over_rfi", lambda fn: fn)
    call, comp = _component_call(PolyInterpVisVariableFFI, cfg, state)
    return np.asarray(jax.jit(call)(state["rfi_A"])), comp


def _baseline_sharded(monkeypatch, cfg, state):
    # Undo whatever the reference patched: monkeypatch lasts to the end of the
    # test, so leaving it in place would build the unsharded component again
    # and compare it with itself.
    monkeypatch.setattr(rfi_vis, "sharding_baselines", dist.sharding_baselines)
    monkeypatch.setattr(rfi_vis, "psum_over_rfi", dist.psum_over_rfi)
    monkeypatch.setenv("TABASCAL_SHARD_AXIS", "baseline")
    dist.baseline_mesh.cache_clear()
    call, comp = _component_call(PolyInterpVisVariableFFI, cfg, state)
    # If this is missing the sharded path was not taken and any agreement below
    # would be vacuous.
    assert hasattr(comp, "_device_groups"), "the sharded path was not built"
    return np.asarray(jax.jit(call)(state["rfi_A"])), comp


def test_four_devices_compute_what_one_computes(monkeypatch):
    cfg, state = _case()
    want, _ = _unsharded(monkeypatch, cfg, state)
    got, comp = _baseline_sharded(monkeypatch, cfg, state)
    assert got.shape == want.shape == (cfg.n_bl, cfg.n_freq, cfg.n_time)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-8)


def test_the_ghosts_were_real_and_changed_nothing(monkeypatch):
    """The padding has to be exercised, not merely available.

    A ghost landing on a real row would be wrong in a way no shape check
    catches, so assert the split actually needed padding and that the answer
    still matches the unpadded one.
    """
    cfg, state = _case()
    want, _ = _unsharded(monkeypatch, cfg, state)
    got, comp = _baseline_sharded(monkeypatch, cfg, state)
    padded = [
        shard.n_padded - shard.n_real
        for shards in comp._device_groups for shard in shards
    ]
    assert sum(padded) > 0, f"no ghosts were needed: {padded}"
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-8)


def test_an_even_split_needs_no_ghosts_and_still_agrees(monkeypatch):
    """The other half of the same claim: padding is absent when unnecessary."""
    cfg, state = _case(requirements=np.full(8, 30))     # one group, 8 over 4
    want, _ = _unsharded(monkeypatch, cfg, state)
    got, comp = _baseline_sharded(monkeypatch, cfg, state)
    padded = [
        shard.n_padded - shard.n_real
        for shards in comp._device_groups for shard in shards
    ]
    assert sum(padded) == 0, f"unexpected ghosts: {padded}"
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-8)
