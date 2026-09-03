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


def test_broadcast_bytes_casts_a_widened_collective_result(monkeypatch):
    """JAX implements broadcast as a sum, which widens uint8 reductions."""
    payload = b'{"ok":true}'
    calls = 0

    def broadcast(value):
        nonlocal calls
        calls += 1
        if calls == 1:
            return np.asarray([len(payload)], dtype=np.int64)
        # This is the dtype returned when JAX reduces the uint8 payload. Calling
        # .tobytes() on it directly would insert three NULs after every byte.
        return np.frombuffer(payload, dtype=np.uint8).astype(np.uint32)

    monkeypatch.setattr(dist.jax, "process_count", lambda: 2)
    monkeypatch.setattr(dist, "is_process_0", lambda: True)
    from jax.experimental import multihost_utils
    monkeypatch.setattr(multihost_utils, "broadcast_one_to_all", broadcast)

    assert dist.broadcast_bytes_from_rank0(payload, "tle-fetch") == payload


# ---------------------------------------------------------------------------
# Name/shape matching rules
# ---------------------------------------------------------------------------

def test_wants_rfi_axis_matching():
    n_rfi = 3
    # plain param key
    assert dist._wants_rfi_axis("rfi_k_r_base", np.zeros((3, 4)), n_rfi)
    # prefixed constant key matches on the segment after the last "/"
    assert dist._wants_rfi_axis("_c/ComplexRFIVarAnt/mu_rfi_k", np.zeros((3, 4)), n_rfi)
    # right name, wrong leading dim -> replicated
    assert not dist._wants_rfi_axis("rfi_A", np.zeros((5, 4)), n_rfi)
    # A component constant that is not in the name list stays replicated, even when
    # its leading dimension happens to equal n_rfi
    assert not dist._wants_rfi_axis("_c/ConstGains/amp_basis", np.zeros((3, 2)), n_rfi)
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
    n_bl = 8
    rng = np.random.default_rng(0)
    tree = {
        "rfi_k_r_base": rng.normal(size=(n_rfi, 2, 3)),
        "_c/ComplexRFIVarAnt/mu_rfi_k": rng.normal(size=(n_rfi, 2, 3)),
        "_c/ConstGains/amp_basis": rng.normal(size=(4, 3)),  # name-excluded
        "ast_k_r_base": rng.normal(size=(n_bl, 3)),
    }
    sharded = dist.shard_pytree(tree, n_rfi)
    for key, val in tree.items():
        np.testing.assert_array_equal(np.asarray(sharded[key]), val)
    assert sharded["rfi_k_r_base"].sharding.spec == P("dev")
    assert sharded["_c/ComplexRFIVarAnt/mu_rfi_k"].sharding.spec == P("dev")
    assert sharded["_c/ConstGains/amp_basis"].sharding.spec == P()
    # no n_bl given: the baseline axis is not sharded at all
    assert sharded["ast_k_r_base"].sharding.spec == P()

    # ... and with it, the astronomical latents split along the baseline axis
    with_bl = dist.shard_pytree(tree, n_rfi, n_bl)
    for key, val in tree.items():
        np.testing.assert_array_equal(np.asarray(with_bl[key]), val)
    assert with_bl["ast_k_r_base"].sharding.spec == P("dev")
    assert with_bl["rfi_k_r_base"].sharding.spec == P("dev")
    assert with_bl["_c/ConstGains/amp_basis"].sharding.spec == P()

    # a baseline count that does not divide the mesh stays replicated: there is
    # nothing to pad a baseline axis with
    assert dist.baselines_shardable(n_bl)
    assert not dist.baselines_shardable(6)
    ragged = {"ast_k_r_base": rng.normal(size=(6, 3))}
    assert dist.shard_pytree(ragged, n_rfi, 6)["ast_k_r_base"].sharding.spec == P()

    # pre-sharded leaves pass through untouched
    again = dist.shard_pytree(sharded, n_rfi)
    assert again["rfi_k_r_base"] is sharded["rfi_k_r_base"]

    # sharded_rfi_zeros: correct global shape/value, one shard per device
    z = dist.sharded_rfi_zeros((n_rfi, 5), complex)
    assert z.shape == (n_rfi, 5)
    assert z.sharding.spec == P("dev")
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
    assert g.sharding.spec == P("dev")

    # map_over_baselines: no collective, the result stays split, and value and
    # gradient match the unsharded computation
    K = rng.normal(size=(n_bl, 2, 3))
    S = rng.normal(size=(n_bl, 2, 3))
    M = rng.normal(size=(n_bl, 2, 3))

    def local_ast(k, s, m):
        # toy per-baseline transform: elementwise, then a reduction that keeps
        # the baseline axis, as the real one does
        return jnp.sum(s * k + m, axis=(1, 2))

    expect = np.asarray(local_ast(jnp.asarray(K), jnp.asarray(S), jnp.asarray(M)))
    K_s = dist.make_global(K, dist.bl_sharding())
    S_s = dist.make_global(S, dist.bl_sharding())
    M_s = dist.make_global(M, dist.bl_sharding())
    got = dist.map_over_baselines(local_ast, n_bl)(K_s, S_s, M_s)
    assert got.sharding.spec == P("dev")
    np.testing.assert_allclose(np.asarray(got), expect, rtol=1e-12)

    def ast_loss(k):
        return jnp.sum(dist.map_over_baselines(local_ast, n_bl)(k, S_s, M_s) ** 2)

    def plain_loss(k):
        return jnp.sum(local_ast(k, jnp.asarray(S), jnp.asarray(M)) ** 2)

    g_ast = jax.jit(jax.grad(ast_loss))(K_s)
    assert g_ast.shape == K.shape
    assert g_ast.sharding.spec == P("dev")
    # The values too, not only the shape and the sharding: the transpose of a
    # shard_map is its own rule, and a wrong one keeps both of those.
    np.testing.assert_allclose(
        np.asarray(g_ast), np.asarray(jax.grad(plain_loss)(jnp.asarray(K))), rtol=1e-12
    )

    # a ragged baseline count falls back to running the body unsharded
    assert dist.map_over_baselines(local_ast, 6) is local_ast

    # constrain_rfi_state inside jit keeps the rfi sharding
    def f(state):
        state = {"rfi_A": state["rfi_A"] * 2.0}
        return dist.constrain_rfi_state(state, n_rfi)

    out = jax.jit(f)({"rfi_A": A_s})
    assert out["rfi_A"].sharding.spec == P("dev")

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


# ---------------------------------------------------------------------------
# fail_together: work only process 0 does must not end only process 0
# ---------------------------------------------------------------------------
#
# Same constraint as the rank0_first tests above -- real multi-host collectives
# are unavailable in this session -- so these force the multi-process branch and
# stand in for the broadcast. What they pin is the decision each process takes
# from what the broadcast tells it, which is what keeps a worker out of a
# collective process 0 has already unwound past.
#
# Waking the workers is *not* enough here, which is why this exists beside
# rank0_first: a woken worker carries on into the next model evaluation, and
# there is no missing resource for it to fail on -- it simply waits.


def _patch_broadcast(monkeypatch, *, rank0_failed: bool):
    """Force the multi-process branch; the broadcast reports rank 0's verdict."""

    from jax.experimental import multihost_utils

    sent = []

    def broadcast(value):
        sent.append(np.asarray(value).tolist())
        return np.array([1 if rank0_failed else 0], dtype=np.int32)

    monkeypatch.setattr(dist.jax, "process_count", lambda: 2)
    monkeypatch.setattr(multihost_utils, "broadcast_one_to_all", broadcast)

    return sent


class TestFailTogether:

    def test_single_process_just_raises_what_it_was_given(self):
        with pytest.raises(OSError, match="no space"):
            dist.fail_together(OSError("no space left"), "write the results")

        assert dist.fail_together(None, "write the results") is None

    def test_a_worker_stops_when_process_0_failed(self, monkeypatch):
        """The whole point: this process had no error of its own to find."""

        _patch_broadcast(monkeypatch, rank0_failed=True)

        with pytest.raises(dist.RankZeroFailure, match="write the results"):
            dist.fail_together(None, "write the results")

    def test_a_worker_carries_on_when_process_0_did_not(self, monkeypatch):
        _patch_broadcast(monkeypatch, rank0_failed=False)

        assert dist.fail_together(None, "write the results") is None

    def test_process_0_raises_the_error_it_actually_hit(self, monkeypatch):
        """Not a RankZeroFailure: the run's report has to name the real cause."""

        sent = _patch_broadcast(monkeypatch, rank0_failed=True)

        with pytest.raises(OSError, match="no space"):
            dist.fail_together(OSError("no space left"), "write the results")

        # And it told the others before raising -- a process that raised first
        # would leave every other one waiting in the broadcast it never made.
        assert sent == [[1]]

    def test_the_others_are_told_even_when_nothing_failed(self, monkeypatch):
        """The collective is unconditional, or the two sides cannot meet at all."""

        sent = _patch_broadcast(monkeypatch, rank0_failed=False)

        dist.fail_together(None, "write the results")

        assert sent == [[0]]


# ---------------------------------------------------------------------------
# Resolution broadcast: elements must be identical on every rank
# ---------------------------------------------------------------------------

class TestResolutionWireRoundTrip:
    """The single guard on the wire-format hazard.

    Process 0 resolves the satellites and broadcasts the accepted records; every
    other rank rebuilds the element frame from what it receives. A TLE survives
    that hop trivially — the two lines are text, and every rank re-parses the
    same 69 characters with the same parser. An OMM record has no lines, so the
    element *values* themselves make the trip, and a lossy projection would
    leave the ranks holding subtly different trajectory priors. Nothing would
    raise; the run would just be wrong. Hence exact equality below, never
    ``approx``.
    """

    def _round_trip(self, record, obs_epoch_jd):
        import json

        from tabascal import orbit as tle

        resolution = tle.TLEResolution([int(record["NORAD_CAT_ID"])], obs_epoch_jd, 3)
        tle._accept_remote(
            {int(record["NORAD_CAT_ID"]): dict(record)},
            "test",
            obs_epoch_jd,
            3,
            resolution.resolved,
            resolution.rejected,
        )
        assert resolution.complete, "the fixture record was not accepted"
        wire = json.loads(json.dumps(tle._resolution_to_wire(resolution)))
        return resolution, tle._resolution_from_wire(wire)

    @pytest.mark.parametrize("kind", ["tle", "omm"])
    def test_elements_survive_the_broadcast_exactly(self, kind):
        from .tle_helpers import jd, make_record

        obs = jd(2026, 8, 1)
        record = make_record(kind, 25544, obs)
        rank0, worker = self._round_trip(record, obs)

        # The wire deliberately carries a subset of the columns — workers need
        # what the elements are derived from, not the provenance. What must not
        # differ is any number the model is built from.
        before, after = rank0.frame(), worker.frame()
        for column in (
            "SEMIMAJOR_AXIS",
            "ECCENTRICITY",
            "INCLINATION",
            "RA_OF_ASC_NODE",
            "ARG_OF_PERICENTER",
            "MEAN_ANOMALY",
            "MEAN_MOTION",
            "BSTAR",
            "EPOCH_JD",
        ):
            assert before[column].tolist() == after[column].tolist(), (
                f"{kind}: {column} diverged between rank 0 and the worker"
            )

    @pytest.mark.parametrize("kind", ["tle", "omm"])
    def test_epochs_and_offsets_survive_the_broadcast_exactly(self, kind):
        from .tle_helpers import jd, make_record

        obs = jd(2026, 8, 1)
        rank0, worker = self._round_trip(make_record(kind, 25544, obs), obs)

        assert worker.obs_epoch_jd == rank0.obs_epoch_jd
        assert worker.resolved[25544].epoch_jd == rank0.resolved[25544].epoch_jd
        assert worker.resolved[25544].offset_days == rank0.resolved[25544].offset_days

    def test_omm_elements_cross_the_wire_as_numbers_not_strings(self):
        # A string projection would round-trip today and mislead the next reader
        # into extending it to a field where it does not. Pin the types.
        from tabascal import orbit as tle

        from .tle_helpers import jd, make_omm

        wired = tle._wire_record(make_omm(25544, jd(2026, 8, 1)))
        assert wired["RECORD_KIND"] == "omm"
        for column in (
            "INCLINATION",
            "RA_OF_ASC_NODE",
            "ECCENTRICITY",
            "ARG_OF_PERICENTER",
            "MEAN_ANOMALY",
            "MEAN_MOTION",
            "BSTAR",
        ):
            assert isinstance(wired[column], float), column
        assert isinstance(wired["EPOCH"], str)
        assert "TLE_LINE1" not in wired

    def test_tle_projection_is_unchanged(self):
        # The TLE path predates this and must not have been disturbed.
        from tabascal import orbit as tle

        from .tle_helpers import jd, make_tle_record

        record = make_tle_record(25544, jd(2026, 8, 1))
        wired = tle._wire_record(record)
        assert wired["RECORD_KIND"] == "tle"
        assert wired["TLE_LINE1"] == record["TLE_LINE1"]
        assert wired["TLE_LINE2"] == record["TLE_LINE2"]
        assert wired["NORAD_CAT_ID"] == 25544
        assert not any(column.startswith("MEAN_") for column in wired)

    def test_a_worst_case_float_survives_the_hop(self):
        # repr() is the shortest round-tripping representation in Python 3, but
        # only if the value goes through the float path. Prove it with values
        # whose decimal expansion is long.
        import json

        from tabascal import orbit as tle

        from .tle_helpers import jd, make_omm

        awkward = {
            "INCLINATION": 51.641600000000004,
            "MEAN_MOTION": 15.721253910000001,
            "ECCENTRICITY": 0.0006703000000000001,
        }
        record = make_omm(25544, jd(2026, 8, 1), **awkward)
        wired = json.loads(json.dumps(tle._wire_record(record)))
        for column, value in awkward.items():
            assert wired[column] == value


class TestResolveSharedFailure:
    """Process 0 must always enter the collective, however it failed.

    Every other rank is already blocked in ``broadcast_one_to_all`` waiting for
    it. A failure that escapes before the broadcast does not fail the run — it
    hangs it, until the coordinator times out.
    """

    def _rank0_of_two(self, monkeypatch):
        monkeypatch.setattr(dist, "process_count", lambda: 2)
        monkeypatch.setattr(dist, "is_process_0", lambda: True)
        sent = []

        def broadcast(payload, label):
            sent.append(payload)
            return payload

        monkeypatch.setattr(dist, "broadcast_bytes_from_rank0", broadcast)
        return sent

    def test_a_resolution_failure_is_broadcast_not_raised_early(self, monkeypatch):
        import json

        from tabascal import orbit as tle

        sent = self._rank0_of_two(monkeypatch)

        def resolve():
            raise tle.TLEError("no coverage")

        with pytest.raises(tle.TLEError, match="no coverage"):
            tle.resolve_shared(resolve)
        assert sent, "process 0 skipped the broadcast every other rank waits on"
        assert json.loads(sent[0])["ok"] is False

    def test_a_serialisation_failure_is_broadcast_too(self, monkeypatch):
        import json

        from tabascal import orbit as tle

        sent = self._rank0_of_two(monkeypatch)
        # A value that resolves fine and then cannot be encoded: the failure
        # lands on json.dumps, which must be inside the guard with everything
        # else, not after it.
        monkeypatch.setattr(
            tle, "_resolution_to_wire", lambda resolution: {"ok": True, "x": object()}
        )

        with pytest.raises(tle.TLEError, match="TypeError"):
            tle.resolve_shared(lambda: None)
        assert sent, "process 0 skipped the broadcast every other rank waits on"
        assert json.loads(sent[0])["ok"] is False
