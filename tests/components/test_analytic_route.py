"""The data-grid route -- FixedOrbit, ComplexRFIVarAnt/ConstAnt, AnalyticVis -- against
its Fine twins and against a dense Riemann sum of the integral AnalyticVis evaluates."""

from math import factorial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tabascal.components import ComponentOrderError
from tabascal.components.rfi_signal import (
    BaseGPRFI,
    ComplexRFIConstAnt,
    ComplexRFIConstAntFine,
    ComplexRFIVarAnt,
    ComplexRFIVarAntFine,
)
from tabascal.components.rfi_vis import AnalyticVis, RiemannVisFine
from tabascal.components.trajectory import FixedOrbit, FixedOrbitFine
from tabascal.interferometry import calculate_rfi_vis_blocked
from tabascal.poly_interp import fine_offsets, interp_tables

from .conftest import active_precision, make_constants
from .test_component_order import check
from .test_rfi_signal import make_rfi_config
from .test_trajectory import _EPOCH_JD, make_trajectory_config

ANALYTIC = {"stencil": 1, "segments": 2, "terms": 6, "cubic_terms": 3, "scratch_mb": 256}
PRECISE = {"segments": 4, "terms": 16, "cubic_terms": 3}
DATA = [FixedOrbit, ComplexRFIVarAnt, AnalyticVis]
SIGNALS = [
    pytest.param(ComplexRFIVarAnt, ComplexRFIVarAntFine, id="VarAnt"),
    pytest.param(ComplexRFIConstAnt, ComplexRFIConstAntFine, id="ConstAnt"),
]


@pytest.fixture(params=[False, True], ids=["float32", "float64"])
def working_precision(request):
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", request.param)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def make_config(
    n_ant=4, n_rfi=1, n_freq=2, n_time=6, n_int_time=5, n_int_freq=1,
    int_time=8.0, chan_width=1e6, corr_time=200.0, rfi_args=None, rfi_mask=None,
):
    """A mock TabConfig for both routes, with the fine grid where TabConfig puts it."""
    traj = make_trajectory_config(
        n_ant=n_ant, n_rfi=n_rfi, n_freq=n_freq, n_time=n_time,
        n_int_time=n_int_time, n_int_freq=n_int_freq,
    )
    sig = make_rfi_config(
        n_rfi=n_rfi, n_rfi_real=n_rfi, n_ant=n_ant, n_freq=n_freq, n_time=n_time,
        n_int_freq=n_int_freq, n_int_time=n_int_time, corr_time=corr_time, corr_freq=1e9,
        init="sample",
    )
    times = int_time * np.arange(n_time, dtype=np.float64)
    freqs = 1.4e9 + chan_width * np.arange(n_freq, dtype=np.float64)
    a1, a2 = np.triu_indices(n_ant, 1)

    cfg = SimpleNamespace(**{**vars(sig), **vars(traj)})
    cfg.args = sig.args  # not the trajectory stub's empty rfi section
    # float64 whatever the session precision, as TabConfig keeps them.
    cfg.times, cfg.freqs = times, freqs
    cfg.times_fine = (times[:, None] + fine_offsets(n_int_time, int_time)).ravel()
    cfg.freqs_fine = (freqs[:, None] + fine_offsets(n_int_freq, chan_width)).ravel()
    cfg.times_jd = _EPOCH_JD + times / 86400.0
    cfg.times_jd_fine = _EPOCH_JD + cfg.times_fine / 86400.0
    cfg.int_time, cfg.chan_width = int_time, chan_width
    cfg.a1, cfg.a2, cfg.n_bl = a1.astype("int32"), a2.astype("int32"), len(a1)
    cfg.rfi_mask = rfi_mask
    cfg.rfi_mask_fine = None if rfi_mask is None else np.repeat(rfi_mask, n_int_time, axis=-1)
    cfg.args["rfi"].update({
        "baseline_block_size": None, "path_order": 3, "path_nodes": None,
        "analytic": dict(ANALYTIC), **(rfi_args or {}),
    })
    return cfg


def run_route(components, cfg):
    """Set ``components`` up on ``cfg``; return their chained forward and the signal's init."""
    comps = [cls() for cls in components]
    for comp in comps:
        comp.setup(cfg)
    constants = {k: v for comp in comps for k, v in make_constants(comp).items()}
    forwards = [comp.build_forward() for comp in comps]

    def run(params):
        state = {"vis_rfi": jnp.zeros((cfg.n_bl, cfg.n_freq, cfg.n_time), dtype=complex)}
        for forward in forwards:
            state = forward(params, state, constants)
        return state

    signals = [c for c in comps if isinstance(c, BaseGPRFI)]
    return run, (signals[0].init_params_base if signals else {})


def trajectory_state(cls, cfg):
    comp = cls()
    comp.setup(cfg)
    return comp.build_forward()({}, {}, make_constants(comp))


def fine_phase(phase, delay, cfg, n_int_time):
    """The phase at ``n_int_time`` samples per cell from the centre phase and the
    delay's Taylor series, ``(r, a, f, n_int_freq, t, n_int_time)``."""
    dt = jnp.asarray(fine_offsets(n_int_time, cfg.int_time), phase.dtype)
    dnu = fine_offsets(cfg.n_int_freq, cfg.chan_width) / 1e6
    nu = jnp.asarray(np.asarray(cfg.freqs)[:, None] / 1e6 + dnu, phase.dtype)
    dnu = jnp.asarray(dnu, phase.dtype)
    dtau = sum(delay[..., k, None] * dt**k / factorial(k) for k in range(1, delay.shape[-1]))
    # MHz times microseconds is cycles.
    cycles = nu[:, :, None, None] * dtau[:, :, None, None] + dnu[:, None, None] * delay[:, :, None, None, :, :1]
    return phase[:, :, :, None, :, None] + 2 * jnp.pi * cycles


def fine_amp(amp, cfg, h, n_int_time):
    """The signal at ``n_int_time`` samples per cell through the ``2h + 1`` stencil."""
    wt, st = interp_tables(cfg.n_time, h, fine_offsets(n_int_time, cfg.int_time) / cfg.int_time)
    wf, sf = interp_tables(cfg.n_freq, h, fine_offsets(cfg.n_int_freq, cfg.chan_width) / cfg.chan_width)
    stencil = amp[:, :, sf[:, None] + np.arange(wf.shape[1])][..., st[:, None] + np.arange(wt.shape[1])]
    real = amp.real.dtype
    return jnp.einsum("rafktl,fku,tlv->rafutv", stencil, jnp.asarray(wf, real), jnp.asarray(wt, real))


def dense_vis(amp, phase, delay, cfg, h, n=2001):
    """Midpoint rule at ``n`` samples per cell of AnalyticVis's time integral."""
    shape = amp.shape[:2] + (cfg.n_freq * cfg.n_int_freq, cfg.n_time * n)
    A = fine_amp(amp, cfg, h, n).reshape(shape)
    P = fine_phase(phase, delay, cfg, n).reshape(shape)
    return calculate_rfi_vis_blocked(A, P, jnp.asarray(cfg.a1), jnp.asarray(cfg.a2), cfg.n_int_freq, n, None)


def op_case(n_int_freq=3, stencil=1, **options):
    """Two sources on six antennas (three unused), a reversed pair and an autocorrelation.

    Across a cell the delay winds up to 1.5 turns linearly, 0.2 quadratically, 0.05 cubically.
    """
    cfg = SimpleNamespace(
        n_ant=6, n_freq=3, n_time=5, n_int_freq=n_int_freq, int_time=2.0, chan_width=1e6,
        freqs=1.4e9 + 1e6 * np.arange(3.0),
        a1=np.array([2, 0, 4, 1], np.int32), a2=np.array([4, 2, 2, 1], np.int32),
        args={"rfi": {"analytic": {**ANALYTIC, "stencil": stencil, **options}}, "data": {}},
    )
    cfg.n_bl = len(cfg.a1)
    rng = np.random.default_rng(7)
    shape = (2, cfg.n_ant, cfg.n_freq, cfg.n_time)
    half = cfg.int_time / 2
    delay = [rng.uniform(-5.0, 5.0, shape[:2] + (cfg.n_time,))]  # us
    for k, turns in ((1, 1.5), (2, 0.2), (3, 0.05)):
        scale = turns * factorial(k) / (cfg.freqs[0] / 1e6 * half**k)  # us/s^k
        delay.append(rng.uniform(-scale, scale, shape[:2] + (cfg.n_time,)))
    state = {
        "rfi_A": jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape)),
        "rfi_phase": jnp.asarray(rng.uniform(0, 2 * np.pi, shape)),
        "rfi_delay_poly_us": jnp.asarray(np.stack(delay, -1)),
        "vis_rfi": jnp.asarray(rng.normal(size=(cfg.n_bl, 3, 5)) + 1j * rng.normal(size=(cfg.n_bl, 3, 5))),
    }
    return cfg, state


def analytic_call(cfg, state):
    comp = AnalyticVis()
    comp.setup(cfg)
    forward, constants = comp.build_forward(), make_constants(comp)
    return lambda amp, phase: forward({}, {**state, "rfi_A": amp, "rfi_phase": phase}, constants)["vis_rfi"]


def rel_err(got, expected):
    return float(jnp.abs(got - expected).max()) / float(jnp.abs(expected).max())


class TestFixedOrbit:

    @pytest.mark.parametrize("order", [0, 3])
    def test_writes_the_data_grid(self, order):
        state = trajectory_state(FixedOrbit, make_config(rfi_args={"path_order": order}))
        assert {k: v.shape for k, v in state.items()} == {
            "rfi_xyz": (1, 6, 3), "rfi_phase": (1, 4, 2, 6), "rfi_delay_poly_us": (1, 4, 6, order + 1),
        }

    def test_the_node_fit_is_the_all_samples_fit(self):
        """The default 13 nodes give the delay a 41-node fit does, at 41 samples per cell."""
        cfg = make_config()
        default = trajectory_state(FixedOrbit, cfg)["rfi_delay_poly_us"]
        cfg.args["rfi"]["path_nodes"] = 41
        full = trajectory_state(FixedOrbit, cfg)["rfi_delay_poly_us"]
        dt = fine_offsets(41, cfg.int_time)
        basis = dt[:, None] ** np.arange(4) / np.array([factorial(k) for k in range(4)])
        # 1e-6 us is 0.3 mm of path.
        assert np.abs((np.asarray(default) - np.asarray(full)) @ basis.T).max() < 1e-6

    def test_the_centre_phase_is_the_fine_grid_phase(self):
        """As a baseline sees it: the absolute phase carries ~20 us of float64 jitter
        in the propagation times, common to every antenna."""
        cfg = make_config(n_int_time=5)
        fine = np.asarray(trajectory_state(FixedOrbitFine, cfg)["rfi_phase"])[..., 2::5]
        state = trajectory_state(FixedOrbit, cfg)
        # relative to the array mean: no antenna carries the range to the satellite
        assert float(jnp.abs(state["rfi_delay_poly_us"][..., 0]).max()) < 20.0
        data, a1, a2 = np.asarray(state["rfi_phase"]), cfg.a1, cfg.a2
        d = (data[:, a1] - data[:, a2]) - (fine[:, a1] - fine[:, a2])
        d = (d + np.pi) % (2 * np.pi) - np.pi
        assert np.abs(d).max() < (1e-4 if active_precision() == "double" else 1e-2)

    @pytest.mark.parametrize("n_int_time, n_int_freq", [(5, 1), (6, 1), (11, 3)])
    def test_rebuilds_the_fine_grid_differential_phase(self, n_int_time, n_int_freq):
        cfg = make_config(n_ant=5, n_int_time=n_int_time, n_int_freq=n_int_freq)
        fine = np.asarray(trajectory_state(FixedOrbitFine, cfg)["rfi_phase"])
        a1, a2 = cfg.a1, cfg.a2
        errors = {}
        for order in (1, 3):
            cfg.args["rfi"]["path_order"] = order
            state = trajectory_state(FixedOrbit, cfg)
            phase = np.asarray(fine_phase(state["rfi_phase"], state["rfi_delay_poly_us"], cfg, n_int_time))
            phase = phase.reshape(fine.shape)
            d = (phase[:, a1] - phase[:, a2]) - (fine[:, a1] - fine[:, a2])
            errors[order] = np.degrees(np.abs((d + np.pi) % (2 * np.pi) - np.pi).max())
        assert errors[3] < (0.05 if active_precision() == "double" else 0.5), errors
        assert errors[1] > errors[3]

    @pytest.mark.parametrize("args, key", [
        *[({"path_order": v}, "path_order") for v in (-1, 1.5, True, 4)],
        *[({"path_nodes": v}, "path_nodes") for v in (3, 0, 2.5, True)],
    ])
    def test_a_bad_path_setting_is_refused(self, args, key):
        with pytest.raises(RuntimeError, match=rf"rfi\.{key}"):
            FixedOrbit().setup(make_config(rfi_args=args))


class TestDataGridSignal:

    @pytest.mark.parametrize("data_cls, fine_cls", SIGNALS)
    def test_is_the_fine_signal_at_each_cells_own_sample(self, data_cls, fine_cls):
        cfg = make_config(n_int_time=5, n_int_freq=3)
        data, fine = data_cls(), fine_cls()
        data.setup(cfg)
        fine.setup(cfg)
        params = fine.init_params_base
        assert {k: v.shape for k, v in data.init_params_base.items()} == {k: v.shape for k, v in params.items()}
        for name in ("mu_rfi_k", "sigma_rfi_k"):
            np.testing.assert_array_equal(getattr(data, name), getattr(fine, name))
        A = data.build_forward()(params, {}, make_constants(data))["rfi_A"]
        A_fine = fine.build_forward()(params, {}, make_constants(fine))["rfi_A"]
        assert A.shape == (1, 4, 2, 6)
        tol = 1e-10 if active_precision() == "double" else 1e-4
        np.testing.assert_allclose(A, A_fine[:, :, 1::3, 2::5], rtol=tol, atol=tol)

    @pytest.mark.parametrize("data_cls", [ComplexRFIVarAnt, ComplexRFIConstAnt])
    def test_the_elevation_mask_is_on_the_data_grid(self, data_cls):
        mask = np.ones((1, 6), dtype=bool)
        mask[:, 4:] = False
        comp = data_cls()
        comp.setup(make_config(rfi_mask=mask))
        A = comp.build_forward()(comp.init_params_base, {}, make_constants(comp))["rfi_A"]
        assert bool(jnp.all(A[..., 4:] == 0)) and bool(jnp.all(A[..., :4] != 0))


class TestAnalyticVis:

    @pytest.mark.parametrize("n_int_freq, stencil, options", [(3, 1, PRECISE), (1, 2, PRECISE), (3, 1, {})])
    def test_is_the_dense_riemann_sum(self, working_precision, n_int_freq, stencil, options):
        """Added to the incoming vis_rfi; the reversed pair is the conjugate."""
        cfg, state = op_case(n_int_freq, stencil, **options)
        got = analytic_call(cfg, state)(state["rfi_A"], state["rfi_phase"]) - state["vis_rfi"]
        expected = dense_vis(state["rfi_A"], state["rfi_phase"], state["rfi_delay_poly_us"], cfg, stencil)
        # The default expansion is the measured speed/accuracy trade; the dense
        # sum's own midpoint error is ~2e-6 here.
        tol = (1e-5 if jax.config.x64_enabled else 1e-4) if options else 1e-4
        assert rel_err(got, expected) < tol
        eps = 1e-12 if jax.config.x64_enabled else 1e-5
        np.testing.assert_allclose(got[2], jnp.conj(got[0]), rtol=0, atol=eps * float(jnp.abs(got).max()))

    def test_derivatives_are_the_dense_riemann_sums(self, working_precision):
        """With respect to the signal and the phase, forward and reverse."""
        cfg, state = op_case(**PRECISE)
        call = analytic_call(cfg, state)
        dense = lambda amp, phase: dense_vis(amp, phase, state["rfi_delay_poly_us"], cfg, 1)
        primals = (state["rfi_A"], state["rfi_phase"])
        rng = np.random.default_rng(1)
        amp, phase = primals
        tangents = (
            jnp.asarray(rng.normal(size=amp.shape) + 1j * rng.normal(size=amp.shape), amp.dtype),
            jnp.asarray(rng.normal(size=phase.shape), phase.dtype),
        )
        tol = 1e-5 if jax.config.x64_enabled else 1e-4
        assert rel_err(jax.jvp(call, primals, tangents)[1], jax.jvp(dense, primals, tangents)[1]) < tol
        vis, pullback = jax.vjp(call, *primals)
        _, dense_pullback = jax.vjp(dense, *primals)
        cot = jnp.asarray(rng.normal(size=vis.shape) + 1j * rng.normal(size=vis.shape), vis.dtype)
        for got, expected in zip(pullback(cot), dense_pullback(cot)):
            assert rel_err(got, expected) < tol

    @pytest.mark.parametrize("signals, n_int_freq, gp", [
        pytest.param((ComplexRFIVarAntFine, ComplexRFIVarAnt), 1, None, id="VarAnt-1"),
        pytest.param((ComplexRFIVarAntFine, ComplexRFIVarAnt), 3, None, id="VarAnt-3"),
        # ConstAnt's own prior keeps only the DC mode here: a constant signal, which
        # every stencil interpolates exactly. VarAnt's gammas and cutoff let it vary.
        pytest.param((ComplexRFIConstAntFine, ComplexRFIConstAnt), 3, ([3, 3], 1e-9), id="ConstAnt-3"),
    ])
    def test_approximates_the_fine_grid_route(self, signals, n_int_freq, gp):
        """The same latent parameters through both routes, and closer than holding each
        cell's value (stencil 0). Cells wind ~4 turns: 101 samples keep the fine route's
        own Riemann error (6e-3 at 31) below the signal interpolation's (9e-4)."""
        cfg = make_config(n_ant=5, n_int_time=101, n_int_freq=n_int_freq, corr_time=80.0)
        if gp is not None:
            cfg.args["rfi"]["gp_cov"]["gammas"], cfg.args["rfi"]["cutoff"] = gp
        fine, params = run_route([FixedOrbitFine, signals[0], RiemannVisFine], cfg)
        expected = fine(params)["vis_rfi"]
        err = rel_err(run_route([FixedOrbit, signals[1], AnalyticVis], cfg)[0](params)["vis_rfi"], expected)
        cfg.args["rfi"]["analytic"]["stencil"] = 0
        err_held = rel_err(run_route([FixedOrbit, signals[1], AnalyticVis], cfg)[0](params)["vis_rfi"], expected)
        assert err < 1e-2 and err < err_held / 3, (err, err_held)

    def test_the_route_does_not_read_the_fine_grid(self):
        """One fine sample per cell, as TabConfig sets on this route, changes nothing:
        in particular the path fit keeps its cubic."""
        s1, s41 = [run(p) for run, p in (run_route(DATA, make_config(n_int_time=n)) for n in (1, 41))]
        assert s1["rfi_delay_poly_us"].shape[-1] == 4
        for key in s1:
            np.testing.assert_array_equal(s1[key], s41[key], err_msg=key)

    def test_the_gradient_reaches_the_latent_signal_through_the_signal_only_kernel(self):
        """The fixed orbit's phase is a constant, so its tangent is a symbolic zero."""
        run, params = run_route(DATA, make_config())
        loss = lambda p: jnp.sum(jnp.abs(run(p)["vis_rfi"]) ** 2)
        grads = jax.grad(loss)(params)
        assert set(grads) == {"rfi_k_r_base", "rfi_k_i_base"}
        assert all(bool(jnp.all(jnp.isfinite(g))) and float(jnp.abs(g).max()) > 0 for g in grads.values())
        jaxpr = str(jax.make_jaxpr(jax.grad(loss))(params))
        assert "rfi_analytic_transpose_op" in jaxpr
        assert "rfi_analytic_full_transpose_op" not in jaxpr

    @pytest.mark.parametrize("key, value", [
        ("stencil", -1), ("stencil", 5), ("stencil", 1.5), ("segments", 0), ("terms", 33),
        ("terms", True), ("cubic_terms", 9), ("scratch_mb", 0), ("stencl", 1),
    ])
    def test_a_bad_analytic_option_is_refused(self, key, value):
        cfg, _ = op_case(**{key: value})
        with pytest.raises(RuntimeError, match=rf"rfi\.analytic\.{key}"):
            AnalyticVis().setup(cfg)

    @pytest.mark.parametrize("attrs, data, match", [
        ({}, {"save_rfi_per_sat": True}, "save_rfi_per_sat"),
        ({"a1": np.array([0, 0], np.int32), "a2": np.array([1, 1], np.int32), "n_bl": 2}, {}, r"\(0, 1\)"),
    ])
    def test_the_per_satellite_split_and_a_repeated_baseline_are_refused(self, attrs, data, match):
        cfg, _ = op_case()
        vars(cfg).update(attrs)
        cfg.args["data"].update(data)
        with pytest.raises(RuntimeError, match=match):
            AnalyticVis().setup(cfg)


class TestComponentOrder:

    @pytest.mark.parametrize("signal", ["rfi_signal:ComplexRFIVarAnt", "rfi_signal:ComplexRFIConstAnt"])
    def test_the_data_grid_chain_assembles(self, signal):
        check(["trajectory:FixedOrbit", signal, "rfi_vis:AnalyticVis", "ast_vis:GPVisAst", "gains:UnitaryGains"])

    @pytest.mark.parametrize("refs, consumer, key", [
        (["trajectory:FixedOrbit", "rfi_signal:ComplexRFIVarAnt", "rfi_vis:RiemannVisFine"], "rfi_vis:RiemannVisFine", "rfi_phase"),
        (["trajectory:FixedOrbitFine", "rfi_signal:ComplexRFIConstAnt", "rfi_vis:RiemannVisFine"], "rfi_vis:RiemannVisFine", "rfi_A"),
        (["trajectory:FixedOrbit", "rfi_signal:ComplexRFIVarAntFine", "rfi_vis:AnalyticVis"], "rfi_vis:AnalyticVis", "rfi_A"),
        (["trajectory:FixedOrbit", "trajectory:PhaseCalculationRFIFine"], "trajectory:PhaseCalculationRFIFine", "rfi_xyz"),
    ])
    def test_a_grid_mismatch_is_refused_at_assembly(self, refs, consumer, key):
        with pytest.raises(ComponentOrderError) as excinfo:
            check(refs)
        message = str(excinfo.value)
        for part in (consumer, key, "fine grid", "data grid"):
            assert part in message, (part, message)

    def test_a_missing_producer_is_offered_on_the_right_grid(self):
        with pytest.raises(ComponentOrderError) as excinfo:
            check(["rfi_signal:ComplexRFIVarAnt", "rfi_vis:AnalyticVis"])
        message = str(excinfo.value)
        assert "'trajectory:FixedOrbit'" in message
        assert "FixedOrbitFine" not in message and "PhaseCalculationRFIFine" not in message
        with pytest.raises(ComponentOrderError, match="'rfi_signal:ComplexRFIVarAnt'.*-- add one"):
            check(["trajectory:FixedOrbit", "rfi_vis:AnalyticVis", "rfi_signal:ComplexRFIVarAntFine"])
