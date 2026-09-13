"""Compiled hybrid values and amplitude derivatives, plus CPU wiring coverage."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tabascal.coarse_rfi_vis import analytic_rfi_vis
from tabascal.components import rfi_vis
from .test_poly_interp_vis_hybrid import _case
from .test_poly_interp_vis_variable import _ReferenceOp, _component_call, _assert_close


class _AnalyticReferenceOp(_ReferenceOp):
    def eval(self, amp, phase, delay, *tables, segments, terms, cubic_terms):
        assert amp.shape[0] == phase.shape[0] == delay.shape[0] == self.n_ant
        assert tables[5].ndim == 0  # scalar duration in the dt slot
        return analytic_rfi_vis(
            *(jnp.swapaxes(x, 0, 1) for x in (amp, phase, delay)),
            *tables, jnp.asarray(self.a1), jnp.asarray(self.a2),
            segments=segments, terms=terms, cubic_terms=cubic_terms,
        )


@pytest.fixture(params=[False, True], ids=["float32", "float64"])
def working_precision(request):
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", request.param)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.fixture(params=["wiring", "compiled"])
def operators(request, monkeypatch, working_precision):
    if request.param == "wiring":
        monkeypatch.setattr(rfi_vis, "RFIInterpVisOp", _ReferenceOp)
        monkeypatch.setattr(rfi_vis, "RFIAnalyticVisOp", _AnalyticReferenceOp)
    else:
        if rfi_vis.RFIInterpVisOp is None or rfi_vis.RFIAnalyticVisOp is None:
            pytest.skip("requires ri_kernels from interp-analytic")
        from ri_kernels.jax_api import rfi_interp_vis_op as quadrature
        from ri_kernels.jax_api import rfi_analytic_vis_op as analytic
        suffix = "_GPU" if jax.default_backend() == "gpu" else ""
        if (getattr(quadrature, "_TAB_LIB_INTERP" + suffix) is None or
                getattr(analytic, "_TAB_LIB_ANALYTIC" + suffix) is None):
            pytest.skip("compiled quadrature and analytic libraries are not built for this backend")


def _assert_derivatives(call, reference, state, comp):
    amp = state["rfi_A"]
    rng = np.random.default_rng(42)
    tangent = jnp.asarray(rng.normal(size=amp.shape) + 1j*rng.normal(size=amp.shape), amp.dtype)
    expected, expected_dot = jax.jvp(jax.jit(reference), (amp,), (tangent,))
    got, got_dot = jax.jvp(jax.jit(call), (amp,), (tangent,))
    _assert_close(got, expected)
    _assert_close(got_dot, expected_dot)
    _, pb = jax.vjp(jax.jit(call), amp)
    _, ref_pb = jax.vjp(jax.jit(reference), amp)
    cot = jnp.asarray(rng.normal(size=got.shape) + 1j*rng.normal(size=got.shape), got.dtype)
    gradient = pb(cot)[0]
    _assert_close(gradient, ref_pb(cot)[0])
    contributions = []
    for group in comp.groups:
        group_cot = jnp.zeros_like(cot).at[group.baseline_indices].set(cot[group.baseline_indices])
        contribution = pb(group_cot)[0]
        _assert_close(contribution, ref_pb(group_cot)[0])
        assert float(jnp.abs(contribution[:, 2]).max()) > .1
        contributions.append(contribution)
    _assert_close(gradient[:, 2], sum(contributions)[:, 2])
    np.testing.assert_array_equal(gradient[:, [1, 3, 5]], 0)


@pytest.mark.parametrize("limit", [0, 10, 100000])
@pytest.mark.parametrize("options", [{}, {"segments": 3, "terms": 12, "cubic_terms": 0}])
def test_value_jvp_and_shared_antenna_vjp(operators, limit, options):
    cfg, state = _case(limit)
    cfg.args["rfi"]["poly_analytic"].update(options)
    # Include a nonzero cubic derivative to exercise forwarding cubic_terms.
    state["rfi_delay_poly_us"] = jnp.concatenate((
        state["rfi_delay_poly_us"], jnp.full((2, cfg.n_ant, cfg.n_time, 1), 1e-6),
    ), axis=-1).at[:, 4, :, 3].set(2e-6)
    call, comp = _component_call(rfi_vis.PolyInterpVisHybridFFI, cfg, state)
    ref, _ = _component_call(rfi_vis.PolyInterpVisHybrid, cfg, state)
    assert comp.analytic_groups == ({0: [True], 10: [False, True], 100000: [False]}[limit])
    assert len(comp._ops) == len(comp.groups)
    _assert_derivatives(call, ref, state, comp)


def test_all_quadrature_is_plain_ffi_identity(operators):
    cfg, state = _case(100000)
    call, comp = _component_call(rfi_vis.PolyInterpVisHybridFFI, cfg, state)
    reference, _ = _component_call(rfi_vis.PolyInterpVisFFI, cfg, state)
    assert comp.analytic_groups == [False]
    assert comp.groups[0].n_g == cfg.n_int_time
    _assert_derivatives(call, reference, state, comp)


@pytest.mark.parametrize("limit,unused", [(0, "RFIInterpVisOp"), (100000, "RFIAnalyticVisOp")])
def test_empty_group_never_constructs_or_calls_an_operator(monkeypatch, working_precision, limit, unused):
    monkeypatch.setattr(rfi_vis, "RFIInterpVisOp", _ReferenceOp)
    monkeypatch.setattr(rfi_vis, "RFIAnalyticVisOp", _AnalyticReferenceOp)

    def forbidden(*args, **kwargs):
        pytest.fail("constructed an operator for an empty group")

    monkeypatch.setattr(rfi_vis, unused, forbidden)
    cfg, state = _case(limit)
    call, comp = _component_call(rfi_vis.PolyInterpVisHybridFFI, cfg, state)
    assert len(comp._ops) == 1
    assert bool(jnp.all(jnp.isfinite(jax.jit(call)(state["rfi_A"]))))
    # An unavailable unused operator must also leave the single-group route usable.
    monkeypatch.setattr(rfi_vis, unused, None)
    _component_call(rfi_vis.PolyInterpVisHybridFFI, cfg, state)


@pytest.mark.parametrize("missing", ["RFIInterpVisOp", "RFIAnalyticVisOp"])
def test_missing_required_operator_fails_at_setup(monkeypatch, missing):
    monkeypatch.setattr(rfi_vis, "RFIInterpVisOp", _ReferenceOp)
    monkeypatch.setattr(rfi_vis, "RFIAnalyticVisOp", _AnalyticReferenceOp)
    monkeypatch.setattr(rfi_vis, missing, None)
    cfg, state = _case()
    with pytest.raises(RuntimeError, match=f"{missing}.*interp-analytic"):
        _component_call(rfi_vis.PolyInterpVisHybridFFI, cfg, state)
