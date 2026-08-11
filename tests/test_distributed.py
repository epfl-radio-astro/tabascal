"""Unit tests for the RFI-axis sharding helpers in tabascal.distributed.

Two regimes:

- In-process: the test session runs on a single CPU device, where every sharding
  helper must be an exact no-op / passthrough. This is the regime all other unit
  tests and single-GPU production runs live in, so identity here is load-bearing.
- Subprocess: multi-device behaviour needs ``XLA_FLAGS=--xla_force_host_platform_
  device_count=N`` set *before* jax initializes, so those checks run a small script
  in a fresh interpreter and assert inside it.
"""

import os
import subprocess
import sys
import textwrap

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from tabascal import distributed as dist


# ---------------------------------------------------------------------------
# Single-device (in-process): everything must be a no-op
# ---------------------------------------------------------------------------

def test_sharding_disabled_single_device():
    assert jax.device_count() == 1
    assert not dist.sharding_enabled()


def test_padded_rfi_count_identity_single_device():
    for n in (1, 3, 32):
        assert dist.padded_rfi_count(n) == n


def test_shard_pytree_identity_single_device():
    tree = {"rfi_A": jnp.zeros((3, 2)), "other": jnp.ones(4)}
    assert dist.shard_pytree(tree, 3) is tree


def test_sharded_rfi_zeros_single_device():
    z = dist.sharded_rfi_zeros((3, 2), complex)
    assert z.shape == (3, 2)
    assert z.dtype == jnp.zeros((1,), dtype=complex).dtype
    np.testing.assert_array_equal(np.asarray(z), 0)


def test_constrain_rfi_state_identity_single_device():
    state = {"rfi_A": jnp.zeros((3, 2))}
    assert dist.constrain_rfi_state(state, 3) is state


def test_psum_over_rfi_passthrough_single_device():
    def local_vis(a, p):
        return jnp.sum(a * p, axis=0)

    assert dist.psum_over_rfi(local_vis) is local_vis


def test_barrier_and_rank0_first_single_process():
    dist.barrier("test")  # must not raise or block
    ran = []
    with dist.rank0_first("test"):
        ran.append(True)
    assert ran == [True]


def test_is_process_0_single_process():
    assert dist.is_process_0()
    dist.print0("", end="")  # rank-0 print; must not raise


def test_process_count_single_process():
    assert dist.process_count() == 1


def test_broadcast_bytes_single_process_is_the_payload():
    assert dist.broadcast_bytes_from_rank0(b"resolved-tles", "tle-fetch") == b"resolved-tles"
    assert dist.broadcast_bytes_from_rank0(None, "tle-fetch") == b""


# ---------------------------------------------------------------------------
# Name/shape matching rules
# ---------------------------------------------------------------------------

def test_wants_rfi_axis_matching():
    n_rfi = 3
    # plain param key
    assert dist._wants_rfi_axis("rfi_k_r_base", np.zeros((3, 4)), n_rfi)
    # prefixed constant key matches on the segment after the last "/"
    assert dist._wants_rfi_axis("_c/FourierGPRFI/mu_rfi_k", np.zeros((3, 4)), n_rfi)
    # right name, wrong leading dim -> replicated
    assert not dist._wants_rfi_axis("rfi_A", np.zeros((5, 4)), n_rfi)
    # L_rfi_A is (n_rfi_times, n_rfi_times) and deliberately not in the name list,
    # even when n_rfi_times happens to equal n_rfi
    assert not dist._wants_rfi_axis("_c/ComplexRFI/L_rfi_A", np.zeros((3, 3)), n_rfi)
    # non-RFI names never shard
    assert not dist._wants_rfi_axis("ast_k_r_base", np.zeros((3, 4)), n_rfi)
    # scalars never shard
    assert not dist._wants_rfi_axis("rfi_A", np.float64(0.0), n_rfi)


# ---------------------------------------------------------------------------
# Multi-device behaviour (subprocess with 4 fake CPU devices)
# ---------------------------------------------------------------------------

_MULTI_DEVICE_SCRIPT = textwrap.dedent(
    """
    import numpy as np
    import jax

    jax.config.update("jax_enable_x64", True)  # keep f64 host arrays exact on device

    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    from tabascal import distributed as dist

    assert jax.device_count() == 4
    assert dist.sharding_enabled()

    # padding: up to the next multiple of the device count
    assert dist.padded_rfi_count(3) == 4
    assert dist.padded_rfi_count(4) == 4
    assert dist.padded_rfi_count(5) == 8

    # shard_pytree: rfi-axis leaves split, others replicated, values preserved
    n_rfi = 4
    rng = np.random.default_rng(0)
    tree = {
        "rfi_k_r_base": rng.normal(size=(n_rfi, 2, 3)),
        "_c/FourierGPRFI/mu_rfi_k": rng.normal(size=(n_rfi, 2, 3)),
        "_c/ComplexRFI/L_rfi_A": rng.normal(size=(4, 4)),   # name-excluded
        "ast_k_r_base": rng.normal(size=(6, 3)),
    }
    sharded = dist.shard_pytree(tree, n_rfi)
    for key, val in tree.items():
        np.testing.assert_array_equal(np.asarray(sharded[key]), val)
    assert sharded["rfi_k_r_base"].sharding.spec == P("rfi")
    assert sharded["_c/FourierGPRFI/mu_rfi_k"].sharding.spec == P("rfi")
    assert sharded["_c/ComplexRFI/L_rfi_A"].sharding.spec == P()
    assert sharded["ast_k_r_base"].sharding.spec == P()

    # pre-sharded leaves pass through untouched
    again = dist.shard_pytree(sharded, n_rfi)
    assert again["rfi_k_r_base"] is sharded["rfi_k_r_base"]

    # sharded_rfi_zeros: correct global shape/value, one shard per device
    z = dist.sharded_rfi_zeros((n_rfi, 5), complex)
    assert z.shape == (n_rfi, 5)
    assert z.sharding.spec == P("rfi")
    np.testing.assert_array_equal(np.asarray(z), 0)
    assert all(s.data.shape == (1, 5) for s in z.addressable_shards)

    # psum_over_rfi: matches the unsharded computation exactly
    A = rng.normal(size=(n_rfi, 3, 6))
    Ph = rng.normal(size=(n_rfi, 3, 6))

    def local_vis(a, p):
        # toy per-source contribution with the source axis summed locally
        return jnp.sum(a * jnp.exp(1.0j * p), axis=0)

    expect = np.asarray(local_vis(jnp.asarray(A), jnp.asarray(Ph)))
    A_s = dist.make_global(A, dist.rfi_sharding())
    Ph_s = dist.make_global(Ph, dist.rfi_sharding())
    got = dist.psum_over_rfi(local_vis)(A_s, Ph_s)
    assert got.sharding.spec == P()
    np.testing.assert_allclose(np.asarray(got), expect, rtol=1e-12)

    # ... and under jit + grad the result and gradient shardings hold
    def loss(a):
        return jnp.abs(dist.psum_over_rfi(local_vis)(a, Ph_s)).sum()

    g = jax.jit(jax.grad(loss))(A_s)
    assert g.shape == A.shape
    assert g.sharding.spec == P("rfi")

    # constrain_rfi_state inside jit keeps the rfi sharding
    def f(state):
        state = {"rfi_A": state["rfi_A"] * 2.0}
        return dist.constrain_rfi_state(state, n_rfi)

    out = jax.jit(f)({"rfi_A": A_s})
    assert out["rfi_A"].sharding.spec == P("rfi")

    print("MULTI_DEVICE_OK")
    """
)


def test_multi_device_helpers_subprocess():
    env = dict(os.environ)
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
    ).strip()
    env["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [sys.executable, "-c", _MULTI_DEVICE_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "MULTI_DEVICE_OK" in result.stdout


# ---------------------------------------------------------------------------
# rank0_first control flow: process 0 must release the barrier even on error
# ---------------------------------------------------------------------------
#
# The real cross-process behaviour needs working multi-host collectives (see
# test_pipeline_multiprocess), which are unavailable in this single-host CPU test
# session. These patch rank0_first into its multi-process branch and assert the
# barrier ordering directly -- in particular that a raise inside process 0's block
# still hits the barrier, which is what wakes the workers instead of deadlocking
# them at their entry barrier until the coordinator timeout.


def _patch_multiprocess(monkeypatch, *, is_rank0: bool):
    """Force rank0_first's multi-process branch and record barrier calls."""
    calls = []
    monkeypatch.setattr(dist.jax, "process_count", lambda: 2)
    monkeypatch.setattr(dist, "is_process_0", lambda: is_rank0)
    monkeypatch.setattr(dist, "barrier", lambda name: calls.append(name))
    return calls


def test_rank0_first_releases_barrier_on_error(monkeypatch):
    calls = _patch_multiprocess(monkeypatch, is_rank0=True)

    with pytest.raises(RuntimeError):
        with dist.rank0_first("boom"):
            raise RuntimeError("process 0 failed inside the block")

    # The finally released the barrier despite the raise; workers are not stranded.
    assert calls == ["rank0_first:boom"]


def test_rank0_first_process0_barrier_after_success(monkeypatch):
    calls = _patch_multiprocess(monkeypatch, is_rank0=True)

    order = []
    with dist.rank0_first("ok"):
        order.append("body")
        assert calls == []  # barrier fires only after the block completes
    order.append("after")

    assert order == ["body", "after"]
    assert calls == ["rank0_first:ok"]


def test_rank0_first_worker_waits_before_body(monkeypatch):
    calls = _patch_multiprocess(monkeypatch, is_rank0=False)

    order = []
    with dist.rank0_first("w"):
        # Worker barriered on entry, so the resource is already in place.
        assert calls == ["rank0_first:w"]
        order.append("body")

    assert order == ["body"]
