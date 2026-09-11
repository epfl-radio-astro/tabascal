"""PolyInterpVisVariable and PolyInterpVisVariableFFI: the data-grid route
with the fine sampling set per baseline group, against PolyInterpVis on the
same configuration."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tabascal.coarse_rfi_vis import coarse_rfi_vis
from tabascal.components.rfi_signal import ComplexRFIVarAntCoarse
from tabascal.components.rfi_vis import (
    PolyInterpVis,
    PolyInterpVisVariable,
    PolyInterpVisVariableFFI,
    RFIInterpVisOp,
)
from tabascal.components.trajectory import FixedOrbitCoarse
from tabascal.interferometry import get_strides_and_idxs

from .conftest import active_precision, make_constants
from .test_coarse_components import make_config


def make_variable_config(n_int_time=12, **kwargs):
    """The route's mock config with the baselines split into stride groups
    the way TabConfig.estimate_rfi_sampling does, from made-up sampling rates."""
    cfg = make_config(n_int_time=n_int_time, **kwargs)
    rng = np.random.default_rng(3)
    samplings = rng.integers(2, n_int_time + 1, size=cfg.n_bl)
    samplings[0] = n_int_time  # one group needs the whole grid
    idxs, strides, n_int = get_strides_and_idxs(samplings, min_bins=2, max_bins=4, min_divisors=1)
    assert n_int == n_int_time, (n_int, n_int_time)
    cfg.time_sample_idxs = [np.asarray(i, dtype=np.int32) for i in idxs]
    cfg.time_strides = [int(s) for s in strides]
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


def _tols():
    return (1e-4, 1e-4) if active_precision() == "single" else (1e-9, 1e-11)


class TestPolyInterpVisVariable:

    def test_groups_come_from_the_config(self):
        cfg = make_variable_config()
        comp = PolyInterpVisVariable()
        comp.setup(cfg)
        assert len(comp.time_strides) == len(cfg.time_strides) >= 2
        assert sorted(np.concatenate(comp.time_sample_idxs)) == list(range(cfg.n_bl))
        constants = make_constants(comp)
        for i, s in enumerate(comp.time_strides):
            assert constants[f"{comp.prefix}/dt_{i}"].shape == (len(range(s // 2, 12, s)),)
            assert constants[f"{comp.prefix}/w_time_{i}"].shape[-1] == constants[f"{comp.prefix}/dt_{i}"].shape[0]

    def test_a_stride_one_group_is_the_full_route(self):
        cfg = make_variable_config()
        full, _ = _route(PolyInterpVis, cfg)
        var, comps = _route(PolyInterpVisVariable, cfg)
        params = comps[1].init_params_base
        vis_full, vis_var = full(params), var(params)
        rtol, atol = _tols()
        scale = float(jnp.abs(vis_full).max())
        for idx, s in zip(comps[2].time_sample_idxs, comps[2].time_strides):
            if s == 1:
                np.testing.assert_allclose(vis_var[idx], vis_full[idx], rtol=rtol, atol=atol * scale)

    def test_a_strided_group_is_the_function_on_every_sth_offset(self):
        """What the group computes is the reference on the subsampled tables:
        the coarser quadrature, nothing else changes."""
        cfg = make_variable_config()
        var, comps = _route(PolyInterpVisVariable, cfg)
        params = comps[1].init_params_base
        vis_var = var(params)
        traj, sig, vis = comps
        constants = {k: v for comp in comps for k, v in make_constants(comp).items()}
        state = {"vis_rfi": jnp.zeros((cfg.n_bl, cfg.n_freq, cfg.n_time), dtype=complex)}
        for comp in comps[:2]:
            state = comp.build_forward()(params, state, constants)
        c = lambda name: constants[f"{vis.prefix}/{name}"]
        rtol, atol = _tols()
        for i, (idx, s) in enumerate(zip(vis.time_sample_idxs, vis.time_strides)):
            expected = coarse_rfi_vis(
                state["rfi_A"], state["rfi_phase"], state["rfi_delay_poly_us"],
                c("w_freq"), c("start_freq"), jnp.asarray(vis.w_time[:, :, s // 2 :: s]), c("start_time"),
                c("dnu_mhz"), jnp.asarray(vis.dt[s // 2 :: s]), c("freqs_mhz"),
                c("a1")[idx], c("a2")[idx],
            )
            np.testing.assert_allclose(vis_var[idx], expected, rtol=rtol, atol=atol * float(jnp.abs(expected).max()))

    def test_a_slow_fringe_configuration_survives_its_strides(self):
        """With strides that suit the fringe rates -- here every baseline slow
        enough for its stride, at 150 MHz on 0.1 s cells -- the groups stay
        close to the full sampling. (The strides are the estimator's
        business; this checks that a coarser quadrature is all a stride is.)"""
        cfg = make_config(n_ant=5, n_int_time=12, int_time=0.1, chan_width=1e5)
        cfg.freqs = 1.5e8 + 1e5 * np.arange(cfg.n_freq, dtype=np.float64)
        cfg.freqs_fine = cfg.freqs.copy()
        # stride by baseline length: the shortest third at 3, the middle at 2
        itrf = np.asarray(cfg.ants_itrf)
        length = np.linalg.norm(itrf[np.asarray(cfg.a1)] - itrf[np.asarray(cfg.a2)], axis=-1)
        order = np.argsort(length)
        n = len(order)
        groups = [order[: n // 3], order[n // 3 : 2 * n // 3], order[2 * n // 3 :]]
        cfg.time_sample_idxs = [np.asarray(g, dtype=np.int32) for g in groups if len(g)]
        cfg.time_strides = [s for g, s in zip(groups, (3, 2, 1)) if len(g)]
        full, _ = _route(PolyInterpVis, cfg)
        var, comps = _route(PolyInterpVisVariable, cfg)
        params = comps[1].init_params_base
        vis_full, vis_var = full(params), var(params)
        err = float(jnp.abs(vis_var - vis_full).max()) / float(jnp.abs(vis_full).max())
        assert err < 5e-2, err

    def test_the_gradient_reaches_the_latent_parameters(self):
        cfg = make_variable_config()
        var, comps = _route(PolyInterpVisVariable, cfg)
        grads = jax.grad(lambda p: jnp.sum(jnp.abs(var(p)) ** 2))(comps[1].init_params_base)
        assert all(bool(jnp.all(jnp.isfinite(g))) and float(jnp.abs(g).max()) > 0 for g in grads.values())


@pytest.mark.skipif(RFIInterpVisOp is None, reason="the installed ri_kernels has no RFIInterpVisOp")
class TestPolyInterpVisVariableFFI:
    """One stride-free operator call per group, tables cut to the group."""

    @pytest.fixture(autouse=True)
    def _needs_the_library(self):
        from ri_kernels.jax_api.rfi_interp_vis_op import _TAB_LIB_INTERP

        if _TAB_LIB_INTERP is None:
            pytest.skip("the RFI interp FFI library is not built")

    def _tols(self):
        return (5e-3, 5e-3) if active_precision() == "single" else (1e-7, 1e-9)

    def test_value_matches_the_pure_jax_variant(self):
        cfg = make_variable_config(n_ant=5)
        ref, comps = _route(PolyInterpVisVariable, cfg)
        groups, _ = _route(PolyInterpVisVariableFFI, cfg)
        params = comps[1].init_params_base
        expected = ref(params)
        rtol, atol = self._tols()
        np.testing.assert_allclose(groups(params), expected, rtol=rtol, atol=atol * float(jnp.abs(expected).max()))
        np.testing.assert_allclose(jax.jit(groups)(params), expected, rtol=rtol, atol=atol * float(jnp.abs(expected).max()))

    def test_jvp_and_vjp_match_the_pure_jax_variant(self):
        cfg = make_variable_config(n_ant=5)
        ref, comps = _route(PolyInterpVisVariable, cfg)
        groups, _ = _route(PolyInterpVisVariableFFI, cfg)
        params = comps[1].init_params_base
        rng = np.random.default_rng(1)
        tangents = jax.tree.map(lambda p: jnp.asarray(rng.normal(size=p.shape), p.dtype), params)
        _, exp_t = jax.jvp(ref, (params,), (tangents,))
        _, got_t = jax.jvp(groups, (params,), (tangents,))
        rtol, atol = self._tols()
        np.testing.assert_allclose(got_t, exp_t, rtol=rtol, atol=atol * float(jnp.abs(exp_t).max()))
        vis, ref_pb = jax.vjp(ref, params)
        _, got_pb = jax.vjp(groups, params)
        cot = jnp.asarray(rng.normal(size=vis.shape) + 1j * rng.normal(size=vis.shape), vis.dtype)
        (exp_g,) = ref_pb(cot)
        (got_g,) = got_pb(cot)
        for name in exp_g:
            np.testing.assert_allclose(got_g[name], exp_g[name], rtol=rtol, atol=atol * float(jnp.abs(exp_g[name]).max()), err_msg=name)
