"""PolyInterpVisFFI against PolyInterpVis: the compiled operator against the
reference function, in value, forward mode and reverse mode, on the data-grid
route's own inputs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tabascal.components.rfi_signal import ComplexRFIVarAntCoarse
from tabascal.components.rfi_vis import PolyInterpVis, PolyInterpVisFFI, RFIInterpVisOp
from tabascal.components.trajectory import FixedOrbitCoarse

from .conftest import active_precision, make_constants
from .test_coarse_components import make_config, run_route

if RFIInterpVisOp is None:
    pytest.skip("the installed ri_kernels has no RFIInterpVisOp", allow_module_level=True)
else:
    from ri_kernels.jax_api.rfi_interp_vis_op import _TAB_LIB_INTERP

    if _TAB_LIB_INTERP is None:
        pytest.skip("the RFI interp FFI library is not built", allow_module_level=True)


def _tols():
    # A different summation order and sincos than the reference: 1e-9 apart in
    # double. In single the phase change across a cell, ~1e4 rad per antenna
    # at orbital range rates, rounds at ~1e-3 rad in both, in different places:
    # see the operator's own tests, which hold it to the float64 reference.
    return (5e-3, 5e-3) if active_precision() == "single" else (1e-7, 1e-9)


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

    return vis, comps[1].init_params_base


@pytest.mark.parametrize("n_int_freq, half_width", [(1, 1), (3, 1), (1, 2), (1, 0)])
def test_value_matches_the_reference(n_int_freq, half_width):
    cfg = make_config(n_ant=5, n_int_time=6, n_int_freq=n_int_freq, rfi_args={"poly_interp_stencil": half_width})
    ref, params = _route(PolyInterpVis, cfg)
    ffi, _ = _route(PolyInterpVisFFI, cfg)
    rtol, atol = _tols()
    expected = ref(params)
    np.testing.assert_allclose(ffi(params), expected, rtol=rtol, atol=atol * float(jnp.abs(expected).max()))
    np.testing.assert_allclose(jax.jit(ffi)(params), expected, rtol=rtol, atol=atol * float(jnp.abs(expected).max()))


def test_jvp_matches_the_reference():
    cfg = make_config(n_ant=5, n_int_time=6)
    ref, params = _route(PolyInterpVis, cfg)
    ffi, _ = _route(PolyInterpVisFFI, cfg)
    tangents = jax.tree.map(lambda p: jnp.asarray(np.random.default_rng(1).normal(size=p.shape), p.dtype), params)
    _, expected = jax.jvp(ref, (params,), (tangents,))
    _, got = jax.jvp(ffi, (params,), (tangents,))
    rtol, atol = _tols()
    np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol * float(jnp.abs(expected).max()))


def test_vjp_matches_the_reference():
    cfg = make_config(n_ant=5, n_int_time=6)
    ref, params = _route(PolyInterpVis, cfg)
    ffi, _ = _route(PolyInterpVisFFI, cfg)
    vis, ref_pullback = jax.vjp(ref, params)
    _, ffi_pullback = jax.vjp(ffi, params)
    rng = np.random.default_rng(2)
    cot = jnp.asarray(rng.normal(size=vis.shape) + 1j * rng.normal(size=vis.shape), vis.dtype)
    (expected,) = ref_pullback(cot)
    (got,) = ffi_pullback(cot)
    rtol, atol = _tols()
    for name in expected:
        scale = float(jnp.abs(expected[name]).max())
        np.testing.assert_allclose(got[name], expected[name], rtol=rtol, atol=atol * scale, err_msg=name)


def test_the_component_is_the_reference_with_the_operator():
    cfg = make_config()
    comp = PolyInterpVisFFI()
    comp.setup(cfg)
    ref = PolyInterpVis()
    ref.setup(cfg)
    for name in ("w_time", "start_time", "w_freq", "start_freq", "dt", "dnu_mhz", "freqs_mhz"):
        np.testing.assert_array_equal(getattr(comp, name), getattr(ref, name))
    assert comp.required_inputs == ref.required_inputs


def _route_from_trajectory(vis_cls, cfg):
    """The route as a function of the trajectory's outputs: the phase and the
    delay polynomial become inputs, as they would be with a fitted orbit."""
    comps = [FixedOrbitCoarse(), ComplexRFIVarAntCoarse(), vis_cls()]
    for comp in comps:
        comp.setup(cfg)
    constants = {k: v for comp in comps for k, v in make_constants(comp).items()}
    forwards = [comp.build_forward() for comp in comps]
    params = comps[1].init_params_base
    state0 = forwards[0](params, {"vis_rfi": jnp.zeros((cfg.n_bl, cfg.n_freq, cfg.n_time), dtype=complex)}, constants)

    def vis(rfi_phase, rfi_delay):
        state = {**state0, "rfi_phase": rfi_phase, "rfi_delay_poly_us": rfi_delay}
        for forward in forwards[1:]:
            state = forward(params, state, constants)
        return state["vis_rfi"]

    return vis, (state0["rfi_phase"], state0["rfi_delay_poly_us"])


def test_phase_and_delay_derivatives_match_the_reference():
    """With the trajectory's outputs as inputs, the operator's JVP and VJP with
    respect to the phase and the delay polynomial agree with the reference's,
    which differentiates through them as plain JAX."""
    cfg = make_config(n_ant=5, n_int_time=6)
    ref, (phase, delay) = _route_from_trajectory(PolyInterpVis, cfg)
    ffi, _ = _route_from_trajectory(PolyInterpVisFFI, cfg)
    rng = np.random.default_rng(3)
    phase_dot = jnp.asarray(rng.normal(size=phase.shape), phase.dtype)
    delay_dot = jnp.asarray(rng.normal(size=delay.shape) * np.abs(np.asarray(delay)).mean(axis=(0, 1, 2)), delay.dtype)
    rtol, atol = _tols()
    vis, exp_t = jax.jvp(ref, (phase, delay), (phase_dot, delay_dot))
    _, got_t = jax.jvp(ffi, (phase, delay), (phase_dot, delay_dot))
    np.testing.assert_allclose(got_t, exp_t, rtol=rtol, atol=atol * float(jnp.abs(exp_t).max()))
    cot = jnp.asarray(rng.normal(size=vis.shape) + 1j * rng.normal(size=vis.shape), vis.dtype)
    _, ref_pb = jax.vjp(ref, phase, delay)
    _, ffi_pb = jax.vjp(ffi, phase, delay)
    for name, got, expected in zip(("rfi_phase", "rfi_delay_poly_us"), ffi_pb(cot), ref_pb(cot)):
        scale = float(jnp.abs(expected).max())
        np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol * scale, err_msg=name)
