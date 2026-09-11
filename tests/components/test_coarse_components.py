"""The data-grid route's components: FixedOrbitCoarse, ComplexRFIVarAntCoarse
and PolyInterpVis, against their fine-grid twins on the same configuration."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tabascal.components.rfi_signal import ComplexRFIVarAnt, ComplexRFIVarAntCoarse
from tabascal.components.rfi_vis import PolyInterpVis, RiemannVis
from tabascal.components.trajectory import FixedOrbit, FixedOrbitCoarse
from tabascal.coarse_rfi_vis import fine_phase
from tabascal.poly_interp import fine_offsets

from .conftest import active_precision, make_constants
from .test_rfi_signal import make_rfi_config
from .test_trajectory import _EPOCH_JD, make_trajectory_config


def _offsets(n_int, spacing):
    return (np.arange(n_int) - n_int // 2) * spacing / n_int


def make_config(
    n_ant=4, n_rfi=1, n_freq=2, n_time=6, n_int_time=5, n_int_freq=1,
    int_time=8.0, chan_width=1e6, corr_time=200.0, rfi_args=None, rfi_mask=None,
):
    """A mock TabConfig for the whole route, fine and data grid alike.

    The trajectory and signal builders each stub what their components read;
    this joins them and lays the fine grid out as ``TabConfig`` does -- every
    cell's fine samples at the same offsets from its centre -- which is what the
    data-grid components rely on and the other builders' ``linspace`` is not.
    """
    traj = make_trajectory_config(
        n_ant=n_ant, n_rfi=n_rfi, n_freq=n_freq, n_time=n_time,
        n_int_time=n_int_time, n_int_freq=n_int_freq,
    )
    sig = make_rfi_config(
        n_rfi=n_rfi, n_rfi_real=n_rfi, n_ant=n_ant, n_freq=n_freq, n_time=n_time,
        n_int_freq=n_int_freq, n_int_time=n_int_time, corr_time=corr_time, corr_freq=1e9,
        init="sample",  # a signal, not the zero prior mean
    )
    times = int_time * np.arange(n_time, dtype=np.float64)
    freqs = 1.4e9 + chan_width * np.arange(n_freq, dtype=np.float64)
    dt, dnu = _offsets(n_int_time, int_time), _offsets(n_int_freq, chan_width)
    a1, a2 = jnp.triu_indices(n_ant, 1)

    cfg = SimpleNamespace(**{**vars(sig), **vars(traj)})
    cfg.args = sig.args  # the trajectory stub's empty rfi section must not win
    # numpy float64 whatever the session precision, as TabConfig keeps them: a
    # Julian date in float32 resolves a quarter of a day, and the fine grid
    # only means anything if every sample has its own time.
    cfg.times = times
    cfg.times_fine = (times[:, None] + dt).ravel()
    cfg.times_jd = _EPOCH_JD + times / 86400.0
    cfg.times_jd_fine = _EPOCH_JD + cfg.times_fine / 86400.0
    cfg.freqs = freqs
    cfg.freqs_fine = (freqs[:, None] + dnu).ravel()
    cfg.int_time, cfg.chan_width = int_time, chan_width
    cfg.a1, cfg.a2, cfg.n_bl = a1.astype("int32"), a2.astype("int32"), len(a1)
    cfg.rfi_mask = rfi_mask
    cfg.rfi_mask_fine = None if rfi_mask is None else np.repeat(rfi_mask, n_int_time, axis=-1)
    cfg.args["rfi"].update({"baseline_block_size": None, **(rfi_args or {})})
    return cfg


def run_route(components, cfg, params=None):
    """Set up ``components`` on ``cfg`` and run their forwards in order."""
    comps = [cls() for cls in components]
    for comp in comps:
        comp.setup(cfg)
    constants = {k: v for comp in comps for k, v in make_constants(comp).items()}
    state = {"vis_rfi": jnp.zeros((cfg.n_bl, cfg.n_freq, cfg.n_time), dtype=complex)}
    signal = next(c for c in comps if isinstance(c, ComplexRFIVarAnt))
    params = signal.init_params_base if params is None else params
    for comp in comps:
        state = comp.build_forward()(params, state, constants)
    return state, comps


FINE = [FixedOrbit, ComplexRFIVarAnt, RiemannVis]
COARSE = [FixedOrbitCoarse, ComplexRFIVarAntCoarse, PolyInterpVis]


def _fine_layout(x, n_freq, n_int_freq, n_time, n_int_time):
    """(n_rfi, n_ant, n_freq, n_int_freq, n_int_time, n_time) -> the fine-grid layout."""
    return np.transpose(x, (0, 1, 2, 3, 5, 4)).reshape(x.shape[0], x.shape[1], n_freq * n_int_freq, n_time * n_int_time)


class TestFixedOrbitCoarse:

    def test_output_shapes(self):
        cfg = make_config()
        comp = FixedOrbitCoarse()
        comp.setup(cfg)
        assert comp.rfi_phase.shape == (1, 4, 2, 6)
        assert comp.rfi_delay_poly_us.shape == (1, 4, 6, 4)
        assert comp.rfi_xyz.shape == (1, cfg.n_time_fine, 3)
        state = comp.build_forward()({}, {}, make_constants(comp))
        assert set(state) == {"rfi_xyz", "rfi_phase", "rfi_delay_poly_us"}

    def test_the_phase_is_the_fine_grid_phase_at_the_cell_centre(self):
        """With an odd sample count one fine sample is the cell centre itself,
        where FixedOrbit's phase and this one must agree -- as the difference a
        baseline sees. The absolute phase of one antenna carries the float64
        jitter of the propagation times (~20 us, so ~0.3 m of path, a turn or
        more), which is common to every antenna at a sample and cancels here."""
        cfg = make_config(n_int_time=5)
        fine, comp = FixedOrbit(), FixedOrbitCoarse()
        fine.setup(cfg)
        comp.setup(cfg)
        assert float(comp.rfi_phase.min()) >= 0.0 and float(comp.rfi_phase.max()) <= 2 * np.pi
        # relative to the array mean: no antenna carries the range to the satellite
        assert float(jnp.abs(comp.rfi_delay_poly_us[..., 0]).max()) < 20.0  # microseconds
        a1, a2 = np.asarray(cfg.a1), np.asarray(cfg.a2)
        centre = np.asarray(fine.rfi_phase)[..., 5 // 2 :: 5]
        coarse = np.asarray(comp.rfi_phase)
        d = (coarse[:, a1] - coarse[:, a2]) - (centre[:, a1] - centre[:, a2])
        d = (d + np.pi) % (2 * np.pi) - np.pi
        assert np.abs(d).max() < (1e-4 if active_precision() == "double" else 1e-2)

    @pytest.mark.parametrize("n_int_time, n_int_freq", [(5, 1), (6, 1), (11, 3)])
    def test_rebuilds_the_fine_grid_differential_phase(self, n_int_time, n_int_freq):
        """The phase difference a baseline sees, against FixedOrbit's fine grid."""
        cfg = make_config(n_ant=5, n_int_time=n_int_time, n_int_freq=n_int_freq, int_time=8.0)
        fine = FixedOrbit()
        fine.setup(cfg)
        errors = {}
        for order in (1, 3):
            cfg.args["rfi"]["path_order"] = order
            comp = FixedOrbitCoarse()
            comp.setup(cfg)
            dt = jnp.asarray(fine_offsets(n_int_time, cfg.int_time))
            dnu_mhz = jnp.asarray(fine_offsets(n_int_freq, cfg.chan_width) / 1e6)
            freqs_mhz = jnp.asarray(np.asarray(cfg.freqs) / 1e6)
            phase = np.stack(
                [fine_phase(comp.rfi_phase[..., t], comp.rfi_delay_poly_us[:, :, t], freqs_mhz, dnu_mhz, dt) for t in range(cfg.n_time)],
                -1,
            )
            phase = _fine_layout(phase, cfg.n_freq, n_int_freq, cfg.n_time, n_int_time)
            a1, a2 = np.asarray(cfg.a1), np.asarray(cfg.a2)
            d = (phase[:, a1] - phase[:, a2]) - (np.asarray(fine.rfi_phase)[:, a1] - np.asarray(fine.rfi_phase)[:, a2])
            errors[order] = np.degrees(np.abs((d + np.pi) % (2 * np.pi) - np.pi).max())
        tol = 0.05 if active_precision() == "double" else 0.5
        assert errors[3] < tol, errors
        assert errors[1] > errors[3]

    def test_a_bad_path_order_is_refused(self):
        with pytest.raises(RuntimeError, match="rfi.path_order"):
            FixedOrbitCoarse().setup(make_config(rfi_args={"path_order": -1}))


class TestComplexRFIVarAntCoarse:

    def test_lands_on_the_data_grid_with_the_same_parameters(self):
        cfg = make_config(n_int_time=5, n_int_freq=3)
        fine, coarse = ComplexRFIVarAnt(), ComplexRFIVarAntCoarse()
        fine.setup(cfg)
        coarse.setup(cfg)
        assert coarse.state_outputs["rfi_A"].shape == (1, 4, 2, 6)
        assert coarse.init_params_base["rfi_k_r_base"].shape == fine.init_params_base["rfi_k_r_base"].shape
        for name in ("mu_rfi_k", "sigma_rfi_k"):
            np.testing.assert_array_equal(getattr(coarse, name), getattr(fine, name))

    def test_is_the_fine_signal_at_each_cells_own_sample(self):
        cfg = make_config(n_int_time=5, n_int_freq=3)
        fine, coarse = ComplexRFIVarAnt(), ComplexRFIVarAntCoarse()
        fine.setup(cfg)
        coarse.setup(cfg)
        params = fine.init_params_base
        A_fine = fine.build_forward()(params, {}, make_constants(fine))["rfi_A"]
        A_coarse = coarse.build_forward()(params, {}, make_constants(coarse))["rfi_A"]
        assert A_coarse.shape == (1, 4, 2, 6)
        rtol = 1e-10 if active_precision() == "double" else 1e-4
        np.testing.assert_allclose(A_coarse, A_fine[:, :, 3 // 2 :: 3, 5 // 2 :: 5], rtol=rtol, atol=rtol)

    def test_the_elevation_mask_is_on_the_data_grid(self):
        mask = np.ones((1, 6), dtype=bool)
        mask[:, 4:] = False
        cfg = make_config(n_int_time=5, rfi_mask=mask)
        coarse = ComplexRFIVarAntCoarse()
        coarse.setup(cfg)
        constants = make_constants(coarse)
        assert constants[f"{coarse.prefix}/rfi_mask_fine"].shape == (1, 6)
        A = coarse.build_forward()(coarse.init_params_base, {}, constants)["rfi_A"]
        assert bool(jnp.all(A[..., 4:] == 0)) and bool(jnp.all(A[..., :4] != 0))


class TestPolyInterpVis:

    def test_reads_the_data_grid_and_writes_the_visibilities(self):
        cfg = make_config()
        state, comps = run_route(COARSE, cfg)
        assert state["rfi_A"].shape == (1, 4, 2, 6)
        assert state["vis_rfi"].shape == (cfg.n_bl, 2, 6)
        assert bool(jnp.all(jnp.isfinite(state["vis_rfi"])))
        vis = comps[-1]
        assert vis.w_time.shape == (6, 3, 5) and vis.w_freq.shape == (2, 1, 1)

    @pytest.mark.parametrize("n_int_freq", [1, 3])
    def test_approximates_the_fine_grid_route(self, n_int_freq):
        """Same latent parameters through both routes: the data-grid one lands
        within the interpolation error of the fine-grid one, and closer than
        holding each cell's value across it."""
        cfg = make_config(n_ant=5, n_int_time=7, n_int_freq=n_int_freq, corr_time=200.0)
        vis_fine = run_route(FINE, cfg)[0]["vis_rfi"]
        vis_coarse = run_route(COARSE, cfg)[0]["vis_rfi"]
        cfg.args["rfi"]["poly_interp_stencil"] = 0
        vis_held = run_route(COARSE, cfg)[0]["vis_rfi"]
        scale = float(jnp.abs(vis_fine).max())
        err = float(jnp.abs(vis_coarse - vis_fine).max()) / scale
        err_held = float(jnp.abs(vis_held - vis_fine).max()) / scale
        assert err < 1e-2, (err, err_held)
        assert err < err_held / 3, (err, err_held)

    def test_one_sample_per_cell_reproduces_the_fine_grid_route(self):
        cfg = make_config(n_int_time=1, n_int_freq=1)
        vis_fine = run_route(FINE, cfg)[0]["vis_rfi"]
        vis_coarse = run_route(COARSE, cfg)[0]["vis_rfi"]
        # To the round-off of the reduced phase: the two routes reduce a
        # 1e7-turn phase to one turn, and one ulp of that is 1e-8 rad.
        rtol = 1e-7 if active_precision() == "double" else 1e-4
        np.testing.assert_allclose(vis_coarse, vis_fine, rtol=rtol, atol=rtol * float(jnp.abs(vis_fine).max()))

    def test_the_gradient_reaches_the_latent_parameters(self):
        cfg = make_config()
        _, comps = run_route(COARSE, cfg)
        constants = {k: v for comp in comps for k, v in make_constants(comp).items()}
        forwards = [comp.build_forward() for comp in comps]

        def loss(params):
            state = {"vis_rfi": jnp.zeros((cfg.n_bl, cfg.n_freq, cfg.n_time), dtype=complex)}
            for forward in forwards:
                state = forward(params, state, constants)
            return jnp.sum(jnp.abs(state["vis_rfi"]) ** 2)

        grads = jax.grad(loss)(comps[1].init_params_base)
        assert set(grads) == {"rfi_k_r_base", "rfi_k_i_base"}
        assert all(bool(jnp.all(jnp.isfinite(g))) and float(jnp.abs(g).max()) > 0 for g in grads.values())

    def test_a_bad_stencil_is_refused(self):
        with pytest.raises(RuntimeError, match="rfi.poly_interp_stencil"):
            PolyInterpVis().setup(make_config(rfi_args={"poly_interp_stencil": 1.5}))

    def test_the_per_satellite_split_is_refused(self):
        cfg = make_config()
        cfg.args["data"]["save_rfi_per_sat"] = True
        with pytest.raises(RuntimeError, match="save_rfi_per_sat"):
            PolyInterpVis().setup(cfg)
