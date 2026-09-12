"""Independent group quadratures and antenna compaction, including the transpose."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tabascal.coarse_rfi_vis import coarse_rfi_vis
from tabascal.components.rfi_signal import ComplexRFIVarAntCoarse
from tabascal.components.rfi_vis import (
    PolyInterpVis, PolyInterpVisVariable, PolyInterpVisVariableFFI, RFIInterpVisOp,
)
from tabascal.components.trajectory import FixedOrbitCoarse
from tabascal.poly_interp import fine_offsets, interp_tables

from .conftest import active_precision, make_constants
from .test_coarse_components import make_config


def make_variable_config(**kwargs):
    cfg = make_config(n_int_time=31, **kwargs)
    cfg.rfi_time_requirements = np.full(cfg.n_bl, 3)
    cfg.rfi_time_requirements[-1] = 31
    return cfg


def _route(vis_cls, cfg):
    comps = [FixedOrbitCoarse(), ComplexRFIVarAntCoarse(), vis_cls()]
    for comp in comps:
        comp.setup(cfg)
    constants = {k: v for comp in comps for k, v in make_constants(comp).items()}
    forwards = [comp.build_forward() for comp in comps]

    def vis(params):
        state = {"vis_rfi": jnp.zeros((cfg.n_bl, cfg.n_freq, cfg.n_time), dtype=complex)}
        for forward in forwards:
            state = forward(params, state, constants)
        return state["vis_rfi"]

    return vis, comps


def _assert_close(got, expected):
    rtol, atol = (1e-4, 1e-4) if active_precision() == "single" else (1e-8, 1e-10)
    np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol * max(1., float(jnp.abs(expected).max())))


def _compact_case():
    # Both groups need antenna 2, both must gather, and antennas 1, 3 and 5
    # never occur in a baseline. The MS order puts the fast baseline first.
    cfg = SimpleNamespace(
        n_ant=6, n_bl=2, n_freq=3, n_time=4, n_int_freq=3, n_int_time=31,
        a1=np.array([2, 0], dtype=np.int32), a2=np.array([4, 2], dtype=np.int32),
        int_time=2., chan_width=1e6, freqs=np.array([1.4e9, 1.401e9, 1.402e9]),
        args={"rfi": {}}, rfi_time_requirements=np.array([30, 2]),
    )
    rng = np.random.default_rng(7)
    shape = (2, cfg.n_ant, cfg.n_freq, cfg.n_time)
    state = {
        "rfi_A": jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape)),
        "rfi_phase": jnp.asarray(rng.normal(size=shape)),
        "rfi_delay_poly_us": jnp.asarray(rng.normal(scale=1e-5, size=(2, cfg.n_ant, cfg.n_time, 3))),
        "vis_rfi": jnp.ones((cfg.n_bl, cfg.n_freq, cfg.n_time), dtype=complex),
    }
    return cfg, state


def _component_call(cls, cfg, state):
    comp = cls()
    comp.setup(cfg)
    forward, constants = comp.build_forward(), make_constants(comp)

    def call(amp):
        return forward({}, {**state, "rfi_A": amp}, constants)["vis_rfi"]

    return call, comp


def _uncompacted_call(cfg, state, groups):
    """An independent full-antenna call per group, at freshly built offsets."""
    dnu = fine_offsets(cfg.n_int_freq, cfg.chan_width)
    wf, sf = interp_tables(cfg.n_freq, 1, dnu / cfg.chan_width)
    tables = []
    for group in groups:
        dt = fine_offsets(group.n_g, cfg.int_time)
        wt, st = interp_tables(cfg.n_time, 1, dt / cfg.int_time)
        idx = group.baseline_indices
        tables.append((idx, tuple(jnp.asarray(x) for x in (
            wf, sf, wt, st, dnu / 1e6, dt, cfg.freqs / 1e6, cfg.a1[idx], cfg.a2[idx],
        ))))

    def call(amp):
        vis = state["vis_rfi"]
        for idx, table in tables:
            group_vis = coarse_rfi_vis(amp, state["rfi_phase"], state["rfi_delay_poly_us"], *table)
            vis = vis.at[idx].add(group_vis)
        return vis

    return call


class _ReferenceOp:
    """Exercise the FFI wrapper on CPU even without the compiled operator.

    This tests the constructor contract and antenna-first input layout; it
    does not claim to test the compiled kernel, which has separate tests below.
    """
    def __init__(self, n_ant, a1, a2):
        self.n_ant, self.a1, self.a2 = n_ant, a1, a2
        assert max(a1.max(), a2.max()) < n_ant

    def eval(self, amp, phase, delay, *tables):
        assert amp.shape[0] == phase.shape[0] == delay.shape[0] == self.n_ant
        return coarse_rfi_vis(
            *(jnp.swapaxes(x, 0, 1) for x in (amp, phase, delay)),
            *tables, jnp.asarray(self.a1), jnp.asarray(self.a2),
        )


@pytest.fixture(params=[PolyInterpVisVariable, PolyInterpVisVariableFFI])
def implementation(request, monkeypatch):
    if request.param is PolyInterpVisVariableFFI:
        monkeypatch.setattr("tabascal.components.rfi_vis.RFIInterpVisOp", _ReferenceOp)
    return request.param


def test_independent_tables_and_remapped_antennas(implementation):
    cfg, state = _compact_case()
    call, comp = _component_call(implementation, cfg, state)
    assert [g.n_g for g in comp.groups] == [3, 31]
    assert comp.identity_antennas == [False, False]
    constants = comp.build_constants()
    assert "w_time" not in constants  # no shared maximum-size table to slice
    for i, group in enumerate(comp.groups):
        assert constants[f"dt_{i}"].shape == (group.n_g,)
        np.testing.assert_allclose(constants[f"dt_{i}"], fine_offsets(group.n_g, cfg.int_time))
        np.testing.assert_array_equal(group.antennas[group.a1], cfg.a1[group.baseline_indices])
        np.testing.assert_array_equal(group.antennas[group.a2], cfg.a2[group.baseline_indices])
    expected = _uncompacted_call(cfg, state, comp.groups)(state["rfi_A"])
    _assert_close(call(state["rfi_A"]), expected)
    _assert_close(jax.jit(call)(state["rfi_A"]), expected)


def test_shared_antenna_cotangents_accumulate_from_both_groups(implementation):
    cfg, state = _compact_case()
    call, comp = _component_call(implementation, cfg, state)
    reference = _uncompacted_call(cfg, state, comp.groups)
    amp = state["rfi_A"]
    rng = np.random.default_rng(14)
    tangent = jnp.asarray(rng.normal(size=amp.shape) + 1j * rng.normal(size=amp.shape), amp.dtype)
    _assert_close(jax.jvp(call, (amp,), (tangent,))[1], jax.jvp(reference, (amp,), (tangent,))[1])
    vis, pullback = jax.vjp(jax.jit(call), amp)
    _, expected_pullback = jax.vjp(reference, amp)
    cot = jnp.asarray(rng.normal(size=vis.shape) + 1j * rng.normal(size=vis.shape), vis.dtype)
    (got,) = pullback(cot)
    (expected,) = expected_pullback(cot)
    _assert_close(got, expected)
    contributions = []
    for group in comp.groups:
        group_cot = jnp.zeros_like(cot).at[group.baseline_indices].set(cot[group.baseline_indices])
        (contribution,) = expected_pullback(group_cot)
        assert float(jnp.abs(contribution[:, 2]).max()) > 0.1
        contributions.append(contribution)
    _assert_close(got[:, 2], (contributions[0] + contributions[1])[:, 2])
    assert float(jnp.abs(got[:, 2] - contributions[0][:, 2]).max()) > 0.1
    assert float(jnp.abs(got[:, 2] - contributions[1][:, 2]).max()) > 0.1
    np.testing.assert_array_equal(got[:, [1, 3, 5]], 0)


def test_identity_antenna_map_bypasses_the_gather(implementation, monkeypatch):
    cfg, state = _compact_case()
    cfg.n_ant = 3
    cfg.a1, cfg.a2 = np.array([0, 1]), np.array([1, 2])
    cfg.args["rfi"]["poly_time_sampling"] = {"max_groups": 1}
    state = {k: v[:, :3] if k != "vis_rfi" else v for k, v in state.items()}
    call, comp = _component_call(implementation, cfg, state)
    assert comp.identity_antennas == [True]
    take = jnp.take

    def checked_take(a, indices, axis=None, **kwargs):
        assert axis != 1, "an identity antenna map must not gather"
        return take(a, indices, axis=axis, **kwargs)

    monkeypatch.setattr(jnp, "take", checked_take)
    call(state["rfi_A"])


def test_one_group_matches_the_non_variable_route(implementation):
    cfg, state = _compact_case()
    cfg.args["rfi"]["poly_time_sampling"] = {"max_groups": 1}
    call, comp = _component_call(implementation, cfg, state)
    ref, _ = _component_call(PolyInterpVis, cfg, state)
    assert len(comp.groups) == 1
    _assert_close(call(state["rfi_A"]), ref(state["rfi_A"]))


def test_gradient_reaches_the_latent_parameters():
    cfg = make_variable_config()
    var, comps = _route(PolyInterpVisVariable, cfg)
    assert len(comps[2].groups) == 2
    grads = jax.grad(lambda p: jnp.sum(jnp.abs(var(p)) ** 2))(comps[1].init_params_base)
    assert all(bool(jnp.all(jnp.isfinite(g))) and float(jnp.abs(g).max()) > 0 for g in grads.values())


@pytest.mark.parametrize("options", [{"max_groups": 0}, {"split_at": 2.5}, {"unknown": {}}, None])
def test_invalid_config_is_refused_at_setup(options):
    cfg, _ = _compact_case()
    cfg.args["rfi"]["poly_time_sampling"] = options
    with pytest.raises(RuntimeError, match="rfi.poly_time_sampling"):
        PolyInterpVisVariable().setup(cfg)


@pytest.mark.skipif(RFIInterpVisOp is None, reason="the installed ri_kernels has no RFIInterpVisOp")
class TestCompiledPolyInterpVisVariableFFI:
    @pytest.fixture(autouse=True)
    def _needs_the_library(self):
        from ri_kernels.jax_api.rfi_interp_vis_op import _TAB_LIB_INTERP
        if _TAB_LIB_INTERP is None:
            pytest.skip("the RFI interp FFI library is not built")

    def test_value_jvp_and_shared_antenna_vjp(self):
        cfg, state = _compact_case()
        call, _ = _component_call(PolyInterpVisVariableFFI, cfg, state)
        reference, _ = _component_call(PolyInterpVisVariable, cfg, state)
        amp = state["rfi_A"]
        rng = np.random.default_rng(3)
        tangent = jnp.asarray(rng.normal(size=amp.shape) + 1j * rng.normal(size=amp.shape), amp.dtype)
        expected, expected_tangent = jax.jvp(reference, (amp,), (tangent,))
        got, got_tangent = jax.jvp(jax.jit(call), (amp,), (tangent,))
        _assert_close(got, expected)
        _assert_close(got_tangent, expected_tangent)
        _, ref_pb = jax.vjp(reference, amp)
        _, got_pb = jax.vjp(jax.jit(call), amp)
        cot = jnp.asarray(rng.normal(size=got.shape) + 1j * rng.normal(size=got.shape), got.dtype)
        _assert_close(got_pb(cot)[0], ref_pb(cot)[0])


def test_group_quadrature_matches_the_analytic_visibility(implementation):
    cfg, state = _compact_case()
    cfg.int_time = 1.
    cfg.n_int_freq = 1
    snr = 10000
    cycles = np.array([0.31, 0.02])
    cfg.rfi_time_requirements = np.ceil(np.pi * cycles * np.sqrt(snr / 6))
    cfg.n_int_time = 41
    # Unit signals and linear delays leave just a constant-rate fringe. The
    # shared antenna is stationary; the endpoints on either side produce the
    # two positive baseline rates. Each frequency has its own analytic sinc.
    state["rfi_A"] = jnp.ones_like(state["rfi_A"][:1])
    state["rfi_phase"] = jnp.zeros_like(state["rfi_phase"][:1])
    delay = np.zeros((1, cfg.n_ant, cfg.n_time, 2))
    delay[0, 0, :, 1] = cycles[1] / (cfg.freqs.max() / 1e6)
    delay[0, 4, :, 1] = -cycles[0] / (cfg.freqs.max() / 1e6)
    state["rfi_delay_poly_us"] = jnp.asarray(delay)
    state["vis_rfi"] = jnp.zeros_like(state["vis_rfi"])
    call, comp = _component_call(implementation, cfg, state)
    assert [g.n_g for g in comp.groups] == [3, 41]
    expected = np.broadcast_to(np.sinc(cycles[:, None, None] * cfg.freqs[None, :, None] / cfg.freqs.max()), state["vis_rfi"].shape)
    assert float(jnp.abs(call(state["rfi_A"]) - expected).max()) < 1 / snr
