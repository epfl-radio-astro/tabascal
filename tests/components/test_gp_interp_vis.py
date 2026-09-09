"""rfi_vis:GPInterpVis, paired with the data-grid signal components.

The component reads ``rfi_A`` on the data grid and integrates the fine grid it
interpolates from that under the signal's own prior. The checks are of the
wiring -- that it is the Riemann sum over the interpolated grid, that the
time-block scan changes nothing, that the pairing is enforced -- and of the
one thing that is not exact: how close the data-grid route comes to the
fine-grid route on a draw from the prior, and that more neighbours bring it
closer.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from tabascal.components import validate_component_order
from tabascal.components.rfi_signal import (
    ComplexRFIConstAnt,
    ComplexRFIConstAntCoarse,
    ComplexRFIVarAnt,
    ComplexRFIVarAntCoarse,
)
from tabascal.components.rfi_vis import GPInterpVis, RiemannVis
from tabascal.config import BASE_STATE_KEYS
from tabascal.gp_interp import interpolate_fine, stencil_stack
from tabascal.write import rfi_vis_per_sat

from .conftest import active_precision, make_constants
from .test_rfi_signal import make_rfi_config, random_params

PAIRS = [
    (ComplexRFIVarAnt, ComplexRFIVarAntCoarse),
    (ComplexRFIConstAnt, ComplexRFIConstAntCoarse),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_config(
    n_ant=4,
    n_rfi=2,
    n_freq=4,
    n_time=16,
    n_int_freq=1,
    n_int_time=5,
    corr_freq=2e7,
    corr_time=60.0,
    **rfi_keys,
):
    """The signal components' mock config, with the baselines the vis components read.

    The mock's time axis spans 120 s in ``n_time`` steps, so at 16 steps the
    default correlation time of 60 s is seven and a half integrations -- a
    prior the 3-point stencil interpolates to a few parts in a thousand.
    """
    config = make_rfi_config(
        n_rfi=n_rfi,
        n_rfi_real=n_rfi,
        n_ant=n_ant,
        n_freq=n_freq,
        n_time=n_time,
        n_int_freq=n_int_freq,
        n_int_time=n_int_time,
        corr_freq=corr_freq,
        corr_time=corr_time,
    )
    a1, a2 = jnp.triu_indices(n_ant, 1)
    config.a1, config.a2 = a1.astype("int32"), a2.astype("int32")
    config.n_bl = int(a1.shape[0])
    config.args["rfi"].update(rfi_keys)
    return config


def signal(cls, config, params=None):
    """Set a signal component up on ``config`` -- which leaves the spectrum on it -- and run it."""
    comp = cls()
    comp.setup(config)
    if params is None:
        params = random_params(comp, seed=3)
    rfi_A = comp.build_forward()(params, {}, make_constants(comp))["rfi_A"]
    return comp, rfi_A, params


def phase_on(config, dtype):
    shape = (config.n_rfi, config.n_ant, config.n_freq_fine, config.n_time_fine)
    return jax.random.uniform(jax.random.PRNGKey(1), shape, maxval=2 * np.pi).astype(dtype)


def vis_state(config, rfi_A, rfi_phase):
    return {
        "rfi_A": rfi_A,
        "rfi_phase": rfi_phase,
        "vis_rfi": jnp.zeros((config.n_bl, config.n_freq, config.n_time), dtype=rfi_A.dtype),
    }


def run_vis(comp, state):
    return comp.build_forward()({}, state, make_constants(comp))["vis_rfi"]


def rel_rms(got, want):
    return float(jnp.sqrt(jnp.mean(jnp.abs(got - want) ** 2) / jnp.mean(jnp.abs(want) ** 2)))


def tols():
    return (1e-10, 1e-10) if active_precision() == "double" else (1e-4, 1e-4)


# ---------------------------------------------------------------------------
# Setup and pairing
# ---------------------------------------------------------------------------


def test_setup_needs_the_spectrum_a_signal_component_leaves():
    """Without a signal component set up first there is no covariance to interpolate under."""
    config = make_config()
    assert not hasattr(config, "rfi_prior_spectrum")

    with pytest.raises(RuntimeError, match="ComplexRFIVarAntCoarse"):
        GPInterpVis().setup(config)


@pytest.mark.parametrize("fine_cls, coarse_cls", PAIRS)
def test_a_fine_grid_signal_is_refused_by_shape(fine_cls, coarse_cls):
    """The state key is shared with the fine-grid components, so the shape has to say.

    The assembly check passes the pairing -- both write ``rfi_A`` -- and the
    forward names the components that would have been right.
    """
    config = make_config()
    _, rfi_A_fine, _ = signal(fine_cls, config)
    comp = GPInterpVis()
    comp.setup(config)

    with pytest.raises(ValueError, match="data grid"):
        run_vis(comp, vis_state(config, rfi_A_fine, phase_on(config, rfi_A_fine.real.dtype)))


def test_the_documented_pairing_assembles():
    validate_component_order(
        [
            C()
            for C in (
                __import__("tabascal.imports", fromlist=["import_components"]).import_components(
                    [
                        "trajectory:FixedOrbit",
                        "rfi_signal:ComplexRFIVarAntCoarse",
                        "rfi_vis:GPInterpVis",
                        "ast_vis:GPVisAst",
                        "gains:UnitaryGains",
                    ]
                )
            )
        ],
        BASE_STATE_KEYS,
    )


@pytest.mark.parametrize(
    "key, value",
    [
        ("gp_interp_stencil", -1),
        ("gp_interp_stencil", 1.5),
        ("gp_interp_stencil", True),
        ("gp_interp_stencil", "1"),
        ("gp_interp_stencil", None),
        ("time_block_size", 0),
        ("time_block_size", 2.5),
        ("time_block_size", False),
        ("baseline_block_size", 0),
    ],
)
def test_the_keys_are_validated_by_name(key, value):
    config = make_config(**{key: value})
    signal(ComplexRFIVarAntCoarse, config)

    with pytest.raises(RuntimeError, match=key):
        GPInterpVis().setup(config)


@pytest.mark.parametrize("n_int_freq, stencil, want", [(1, 2, [0, 2]), (2, 2, [2, 2]), (1, 1, [0, 1]), (3, 0, [0, 0])])
def test_an_axis_with_one_sample_per_cell_gets_no_stencil(n_int_freq, stencil, want):
    """Its one fine sample is the coarse value, on which the prior puts every weight anyway."""
    config = make_config(n_int_freq=n_int_freq, n_int_time=5, gp_interp_stencil=stencil)
    signal(ComplexRFIVarAntCoarse, config)
    comp = GPInterpVis()
    comp.setup(config)

    assert comp.stencil == want
    n_stencil = (2 * want[0] + 1) * (2 * want[1] + 1)
    assert comp.weights.shape == (config.n_freq, config.n_time, n_int_freq, 5, n_stencil)


# ---------------------------------------------------------------------------
# What it computes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fine_cls, coarse_cls", PAIRS)
@pytest.mark.parametrize("n_int_freq", [1, 2])
def test_it_is_the_riemann_sum_over_the_interpolated_grid(fine_cls, coarse_cls, n_int_freq):
    """``RiemannVis`` on the grid the weights interpolate is what the component returns."""
    config = make_config(n_int_freq=n_int_freq)
    _, rfi_A, _ = signal(coarse_cls, config)
    phase = phase_on(config, rfi_A.real.dtype)
    comp = GPInterpVis()
    comp.setup(config)
    reference = RiemannVis()
    reference.setup(config)

    got = run_vis(comp, vis_state(config, rfi_A, phase))
    fine = interpolate_fine(stencil_stack(rfi_A, comp.offsets, comp.stencil), jnp.asarray(comp.weights))
    want = run_vis(reference, vis_state(config, fine, phase))

    atol, rtol = tols()
    assert got.shape == (config.n_bl, config.n_freq, config.n_time)
    assert got.dtype == want.dtype
    assert jnp.allclose(got, want, atol=atol * jnp.abs(want).max(), rtol=rtol)


@pytest.mark.parametrize("fine_cls, coarse_cls", PAIRS)
def test_it_agrees_with_the_fine_grid_route_on_a_prior_draw(fine_cls, coarse_cls):
    """Coarse signal + interpolation against fine signal + Riemann sum, same latent.

    The two are the same model evaluated two ways, and differ by the process's
    scatter within a cell given its neighbours: small at a correlation time of
    seven integrations, and smaller with the 5-point stencil than the 3-point.
    """
    config_fine, config_1, config_2 = make_config(), make_config(), make_config(gp_interp_stencil=2)
    _, rfi_A_fine, params = signal(fine_cls, config_fine)
    _, rfi_A_1, _ = signal(coarse_cls, config_1, params)
    _, rfi_A_2, _ = signal(coarse_cls, config_2, params)
    assert jnp.allclose(rfi_A_1, rfi_A_2)
    phase = phase_on(config_fine, rfi_A_fine.real.dtype)

    reference = RiemannVis()
    reference.setup(config_fine)
    want = run_vis(reference, vis_state(config_fine, rfi_A_fine, phase))

    def coarse_route(config, rfi_A):
        comp = GPInterpVis()
        comp.setup(config)
        return run_vis(comp, vis_state(config, rfi_A, phase))

    err_1 = rel_rms(coarse_route(config_1, rfi_A_1), want)
    err_2 = rel_rms(coarse_route(config_2, rfi_A_2), want)

    assert err_1 < 1e-2
    assert err_2 < err_1 / 2


@pytest.mark.parametrize("time_block", [None, 1, 3, 5, 16, 64])
def test_the_time_block_size_changes_neither_the_value_nor_the_gradient(time_block):
    """Blocks that divide the axis, leave a tail, and exceed it, against no scan."""
    config = make_config(n_int_freq=2, time_block_size=time_block)
    _, rfi_A, _ = signal(ComplexRFIVarAntCoarse, config)
    state = vis_state(config, rfi_A, phase_on(config, rfi_A.real.dtype))
    cotangent = jax.random.normal(jax.random.PRNGKey(7), state["vis_rfi"].shape).astype(state["vis_rfi"].dtype)

    def value_and_grads(comp):
        constants = make_constants(comp)
        forward = comp.build_forward()
        primal, vjp = jax.vjp(lambda s: forward({}, s, constants)["vis_rfi"], state)
        (grads,) = vjp(cotangent)
        return primal, grads

    blocked = GPInterpVis()
    blocked.setup(config)
    config.args["rfi"]["time_block_size"] = None
    whole = GPInterpVis()
    whole.setup(config)

    vis, grads = value_and_grads(blocked)
    ref, ref_grads = value_and_grads(whole)

    atol, rtol = tols()
    assert jnp.allclose(vis, ref, atol=atol * jnp.abs(ref).max(), rtol=rtol)
    for key in ("rfi_A", "rfi_phase"):
        assert grads[key].dtype == ref_grads[key].dtype
        assert jnp.allclose(grads[key], ref_grads[key], atol=atol * jnp.abs(ref_grads[key]).max(), rtol=rtol)


def test_the_forward_is_jit_compatible():
    config = make_config(n_int_freq=2, time_block_size=3)
    _, rfi_A, _ = signal(ComplexRFIVarAntCoarse, config)
    state = vis_state(config, rfi_A, phase_on(config, rfi_A.real.dtype))
    comp = GPInterpVis()
    comp.setup(config)
    constants = make_constants(comp)
    forward = comp.build_forward()

    eager = forward({}, state, constants)["vis_rfi"]
    jitted = jax.jit(forward)({}, state, constants)["vis_rfi"]

    atol, rtol = tols()
    assert jnp.allclose(eager, jitted, atol=atol * jnp.abs(eager).max(), rtol=rtol)


def test_the_forward_preserves_other_state_keys():
    config = make_config()
    _, rfi_A, _ = signal(ComplexRFIVarAntCoarse, config)
    state = {**vis_state(config, rfi_A, phase_on(config, rfi_A.real.dtype)), "some_extra_key": jnp.array(42.0)}
    comp = GPInterpVis()
    comp.setup(config)

    out = comp.build_forward()({}, state, make_constants(comp))

    assert out["some_extra_key"] == 42.0
    assert out["rfi_A"] is state["rfi_A"]


# ---------------------------------------------------------------------------
# Downstream: the per-satellite decomposition runs the component's own forward
# ---------------------------------------------------------------------------


def test_the_per_satellite_decomposition_sums_back_to_the_total():
    """``data.save_rfi_per_sat`` evaluates the run's rfi_vis op one source at a time.

    It rebuilds the component from ``model.components`` on the run's config --
    where the signal component has left the spectrum -- and hands it the fitted
    ``rfi_A``, which for this component is the data grid. The pieces have to
    sum back to the total.
    """
    config = make_config()
    config.args["model"] = {
        "components": ["rfi_signal:ComplexRFIVarAntCoarse", "rfi_vis:GPInterpVis"]
    }
    _, rfi_A, _ = signal(ComplexRFIVarAntCoarse, config)
    phase = phase_on(config, rfi_A.real.dtype)
    comp = GPInterpVis()
    comp.setup(config)
    total = run_vis(comp, vis_state(config, rfi_A, phase))

    vi_pred = {"rfi_A": rfi_A[None], "rfi_phase": phase[None], "vis_rfi": total[None]}
    vis_src, norad_ids = rfi_vis_per_sat(vi_pred, config)

    atol, rtol = tols()
    assert vis_src.shape == (1, config.n_rfi, config.n_bl, config.n_freq, config.n_time)
    assert list(norad_ids) == [int(n) for n in config.norad_ids]
    assert jnp.allclose(vis_src.sum(axis=1)[0], total, atol=atol * jnp.abs(total).max(), rtol=rtol)


# ---------------------------------------------------------------------------
# The phase on the data grid, with the path's derivatives
# ---------------------------------------------------------------------------

from tabascal.gp_interp import fine_offsets
from tabascal.rfi_path import fine_frequency_terms, fine_phase_from_path, taylor_powers


def path_state(config, rfi_A, order=3, seed=11):
    """A data-grid phase and path of realistic size, and the fine phase they rebuild to.

    Random rather than propagated: the wiring is what is under test here, and
    the reconstruction itself is held to the definition in test_rfi_path.
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), 2)
    shape = (config.n_rfi, config.n_ant, config.n_time)
    phase_c = jax.random.uniform(keys[0], (config.n_rfi, config.n_ant, config.n_freq, config.n_time), minval=-np.pi, maxval=np.pi)
    # Path, rate, acceleration, jerk of a low orbit on a kilometre baseline.
    scales = jnp.asarray([1e3, 3.0, 0.03, 3e-3][: order + 1])
    path = jax.random.normal(keys[1], shape + (order + 1,)) * scales
    path = path - path.mean(axis=1, keepdims=True)
    real = rfi_A.real.dtype
    phase_c, path = phase_c.astype(real), path.astype(real)

    freqs_fine, dnu = fine_frequency_terms(config.freqs, config.n_int_freq, config.chan_width)
    powers = taylor_powers(fine_offsets(config.n_int_time, config.int_time), order)
    fine = fine_phase_from_path(phase_c, path, jnp.asarray(freqs_fine, real), jnp.asarray(dnu, real), jnp.asarray(powers, real))
    return phase_c, path, fine


@pytest.mark.parametrize("n_int_freq", [1, 2])
def test_the_data_grid_phase_route_is_the_fine_phase_route_on_the_rebuilt_phase(n_int_freq):
    """With the fine phase being what the path rebuilds, the two routes are one computation."""
    config = make_config(n_int_freq=n_int_freq)
    _, rfi_A, _ = signal(ComplexRFIVarAntCoarse, config)
    phase_c, path, fine = path_state(config, rfi_A)
    comp = GPInterpVis()
    comp.setup(config)

    from_fine = run_vis(comp, vis_state(config, rfi_A, fine))
    from_path = run_vis(comp, {**vis_state(config, rfi_A, phase_c), "rfi_path": path})

    atol, rtol = tols()
    assert jnp.allclose(from_path, from_fine, atol=atol * jnp.abs(from_fine).max(), rtol=rtol)


def test_a_data_grid_phase_without_the_path_is_refused():
    config = make_config()
    _, rfi_A, _ = signal(ComplexRFIVarAntCoarse, config)
    phase_c, _, _ = path_state(config, rfi_A)
    comp = GPInterpVis()
    comp.setup(config)

    with pytest.raises(ValueError, match="rfi_path"):
        run_vis(comp, vis_state(config, rfi_A, phase_c))


def test_a_phase_on_neither_grid_is_refused():
    config = make_config()
    _, rfi_A, _ = signal(ComplexRFIVarAntCoarse, config)
    comp = GPInterpVis()
    comp.setup(config)
    odd = jnp.zeros((config.n_rfi, config.n_ant, config.n_freq, config.n_time + 1), dtype=rfi_A.real.dtype)

    with pytest.raises(ValueError, match="fine grid"):
        run_vis(comp, vis_state(config, rfi_A, odd))


@pytest.mark.parametrize("time_block", [None, 1, 3, 5, 16])
def test_the_time_block_changes_nothing_on_the_data_grid_phase_route(time_block):
    """Value and gradient with respect to the amplitude, the phase and the path."""
    config = make_config(n_int_freq=2, time_block_size=time_block)
    _, rfi_A, _ = signal(ComplexRFIVarAntCoarse, config)
    phase_c, path, _ = path_state(config, rfi_A)
    state = {**vis_state(config, rfi_A, phase_c), "rfi_path": path}
    cotangent = jax.random.normal(jax.random.PRNGKey(7), state["vis_rfi"].shape).astype(state["vis_rfi"].dtype)

    def value_and_grads(comp):
        constants = make_constants(comp)
        forward = comp.build_forward()
        primal, vjp = jax.vjp(lambda s: forward({}, s, constants)["vis_rfi"], state)
        (grads,) = vjp(cotangent)
        return primal, grads

    blocked = GPInterpVis()
    blocked.setup(config)
    config.args["rfi"]["time_block_size"] = None
    whole = GPInterpVis()
    whole.setup(config)

    vis, grads = value_and_grads(blocked)
    ref, ref_grads = value_and_grads(whole)

    atol, rtol = tols()
    assert jnp.allclose(vis, ref, atol=atol * jnp.abs(ref).max(), rtol=rtol)
    for key in ("rfi_A", "rfi_phase", "rfi_path"):
        assert jnp.allclose(grads[key], ref_grads[key], atol=atol * jnp.abs(ref_grads[key]).max(), rtol=rtol)


def test_the_per_satellite_decomposition_carries_the_path():
    config = make_config()
    config.args["model"] = {
        "components": ["rfi_signal:ComplexRFIVarAntCoarse", "rfi_vis:GPInterpVis"]
    }
    _, rfi_A, _ = signal(ComplexRFIVarAntCoarse, config)
    phase_c, path, _ = path_state(config, rfi_A)
    comp = GPInterpVis()
    comp.setup(config)
    total = run_vis(comp, {**vis_state(config, rfi_A, phase_c), "rfi_path": path})

    vi_pred = {"rfi_A": rfi_A[None], "rfi_phase": phase_c[None], "rfi_path": path[None], "vis_rfi": total[None]}
    vis_src, _ = rfi_vis_per_sat(vi_pred, config)

    atol, rtol = tols()
    assert jnp.allclose(vis_src.sum(axis=1)[0], total, atol=atol * jnp.abs(total).max(), rtol=rtol)


def test_the_fully_data_grid_pairing_assembles():
    from tabascal.imports import import_components

    validate_component_order(
        [
            C()
            for C in import_components(
                [
                    "trajectory:FixedOrbitCoarse",
                    "rfi_signal:ComplexRFIConstAntCoarse",
                    "rfi_vis:GPInterpVis",
                    "ast_vis:GPVisAst",
                    "gains:UnitaryGains",
                ]
            )
        ],
        BASE_STATE_KEYS,
    )
    validate_component_order(
        [
            C()
            for C in import_components(
                [
                    "trajectory:Orbit",
                    "trajectory:PathCalculationRFI",
                    "rfi_signal:ComplexRFIVarAntCoarse",
                    "rfi_vis:GPInterpVis",
                    "gains:UnitaryGains",
                ]
            )
        ],
        BASE_STATE_KEYS,
    )
