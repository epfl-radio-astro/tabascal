"""The hybrid route shares group maps, accumulation and the amplitude transpose."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tabascal.components.rfi_vis import PolyInterpVisHybrid
from .test_poly_interp_vis_variable import _compact_case, _component_call, _uncompacted_call


def _case(limit=10):
    cfg, state = _compact_case()
    cfg.args["rfi"]["poly_analytic"] = {"quadrature_limit": limit}
    return cfg, state


@pytest.mark.parametrize("x64,limit", [(False, 166), (True, 56)])
@pytest.mark.parametrize("options", [{}, {"quadrature_limit": None}, {"segments": 4, "terms": 16}])
def test_default_cut_uses_measured_vjp_crossover(x64, limit, options):
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", x64)
    try:
        cfg, state = _case()
        cfg.args["rfi"]["poly_analytic"] = options
        cfg.rfi_time_requirements = np.array([limit, limit-1])
        _, comp = _component_call(PolyInterpVisHybrid, cfg, state)
        assert comp.quadrature_limit == limit
        assert (comp.segments, comp.terms, comp.cubic_terms) == (
            options.get("segments", 2), options.get("terms", 6), 3,
        )
        assert comp.analytic_groups == [False, True]
        assert [g.n_g for g in comp.groups] == [limit-1, limit+1]
        constants = comp.build_constants()
        assert constants['w_time_0'].shape[-1] == limit-1
        assert constants['w_time_1'].shape[-1] == 3
        assert constants['dt_1'].ndim == 0
        assert comp.identity_antennas == [False, False]
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("limit,analytic", [(0, True), (100000, False)])
def test_either_group_can_be_empty(limit, analytic):
    cfg, state = _case(limit)
    call, comp = _component_call(PolyInterpVisHybrid, cfg, state)
    assert comp.analytic_groups == [analytic]
    assert len(comp.groups) == 1
    assert jnp.all(jnp.isfinite(jax.jit(call)(state['rfi_A'])))


def test_hybrid_value_and_shared_antenna_gradients_match_quadrature():
    cfg, state = _case()
    call, comp = _component_call(PolyInterpVisHybrid, cfg, state)
    assert comp.analytic_groups == [False, True]
    # The slow group retains its actual quadrature; only the fast reference
    # is densely sampled. Both references carry every antenna independently.
    reference_groups = [replace(g, n_g=8193) if analytic else g for g, analytic in zip(comp.groups, comp.analytic_groups)]
    ref = _uncompacted_call(cfg, state, reference_groups)
    amp = state['rfi_A']
    tol = 5e-6 if not jax.config.x64_enabled else 2e-7
    np.testing.assert_allclose(jax.jit(call)(amp), ref(amp), atol=tol, rtol=0)
    rng = np.random.default_rng(4)
    tangent = jnp.asarray(rng.normal(size=amp.shape)+1j*rng.normal(size=amp.shape), amp.dtype)
    np.testing.assert_allclose(jax.jvp(call, (amp,), (tangent,))[1], jax.jvp(ref, (amp,), (tangent,))[1], atol=tol, rtol=0)
    vis, pb = jax.vjp(jax.jit(call), amp)
    _, ref_pb = jax.vjp(ref, amp)
    cot = jnp.asarray(rng.normal(size=vis.shape)+1j*rng.normal(size=vis.shape), vis.dtype)
    got, expected = pb(cot)[0], ref_pb(cot)[0]
    np.testing.assert_allclose(got, expected, atol=tol, rtol=0)
    contributions = []
    for group in comp.groups:
        group_cot = jnp.zeros_like(cot).at[group.baseline_indices].set(cot[group.baseline_indices])
        contribution = ref_pb(group_cot)[0]
        assert float(jnp.abs(contribution[:, 2]).max()) > .1
        contributions.append(contribution)
    np.testing.assert_allclose(got[:, 2], (contributions[0]+contributions[1])[:, 2], atol=tol, rtol=0)
    np.testing.assert_array_equal(got[:, [1, 3, 5]], 0)


@pytest.mark.parametrize("options", [
    {"quadrature_limit": -1}, {"quadrature_limit": True}, {"quadrature_limit": 2.5},
    {"segments": 0}, {"segments": False}, {"terms": 0}, {"terms": 1.1}, {"unknown": 1}, {"cubic_terms": -1}, {"cubic_terms": True},
])
def test_invalid_settings_are_refused(options):
    cfg, state = _case()
    cfg.args['rfi']['poly_analytic'] = options
    with pytest.raises(RuntimeError, match='rfi.poly_analytic'):
        _component_call(PolyInterpVisHybrid, cfg, state)
