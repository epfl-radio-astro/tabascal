"""rfi_vis:PolyInterpVis, the data-grid route with a polynomial through the stencil.

It inherits everything but the weights from ``GPInterpVis``, so the checks are
of what is its own: that the weights are the polynomial's and need no prior,
that the degree key is validated and, above the interpolating degree, reads
the prior, that the forward names the component, and how the route compares
with the fine-grid one and with ``GPInterpVis`` on a draw from the prior.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from tabascal.components import validate_component_order
from tabascal.components.rfi_signal import ComplexRFIVarAnt, ComplexRFIVarAntCoarse
from tabascal.components.rfi_vis import GPInterpVis, PolyInterpVis, RiemannVis
from tabascal.config import BASE_STATE_KEYS
from tabascal.gp_interp import interpolate_fine, stencil_stack
from tabascal.imports import import_components
from tabascal.poly_interp import polynomial_weights

from .conftest import make_constants
from .test_gp_interp_vis import make_config, phase_on, rel_rms, run_vis, signal, tols, vis_state


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def test_the_default_weights_are_the_polynomials_and_need_no_prior():
    """No signal component set up, no spectrum on the config: the polynomial needs neither."""
    config = make_config(n_int_freq=2)
    assert not hasattr(config, "rfi_prior_spectrum")

    comp = PolyInterpVis()
    comp.setup(config)

    assert comp.stencil == [1, 1]
    want = polynomial_weights([2, 5], [1, 1], [config.n_freq, config.n_time])
    assert np.array_equal(comp.weights, want)


def test_a_degree_above_the_stencil_reads_the_prior():
    """Above the interpolating degree the extra coefficients come from the signal's spectrum."""
    config = make_config(poly_interp_degree=4)
    with pytest.raises(RuntimeError, match="ComplexRFIVarAntCoarse"):
        PolyInterpVis().setup(config)

    signal(ComplexRFIVarAntCoarse, config)
    comp = PolyInterpVis()
    comp.setup(config)
    spectrum = config.rfi_prior_spectrum
    want = polynomial_weights(
        [1, 5],
        [0, 1],
        [config.n_freq, config.n_time],
        degree=4,
        pk=spectrum["pk"],
        ks=spectrum["ks"],
        dxs=[config.chan_width, config.int_time],
    )
    assert np.array_equal(comp.weights, want)


@pytest.mark.parametrize(
    "key, value",
    [
        ("poly_interp_degree", 1),
        ("poly_interp_degree", -2),
        ("poly_interp_degree", 2.5),
        ("poly_interp_degree", True),
        ("poly_interp_degree", "2"),
        ("gp_interp_stencil", None),
        ("time_block_size", 0),
    ],
)
def test_the_keys_are_validated_by_name(key, value):
    config = make_config(**{key: value})
    with pytest.raises(RuntimeError, match=key):
        PolyInterpVis().setup(config)


def test_the_interpolating_degree_is_the_default():
    plain = PolyInterpVis()
    plain.setup(make_config())
    at_degree = PolyInterpVis()
    at_degree.setup(make_config(poly_interp_degree=2))

    assert np.array_equal(plain.weights, at_degree.weights)


def test_the_documented_pairing_assembles():
    validate_component_order(
        [
            C()
            for C in import_components(
                [
                    "trajectory:FixedOrbitCoarse",
                    "rfi_signal:ComplexRFIVarAntCoarse",
                    "rfi_vis:PolyInterpVis",
                    "ast_vis:GPVisAst",
                    "gains:UnitaryGains",
                ]
            )
        ],
        BASE_STATE_KEYS,
    )


# ---------------------------------------------------------------------------
# What it computes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_int_freq", [1, 2])
def test_it_is_the_riemann_sum_over_the_interpolated_grid(n_int_freq):
    config = make_config(n_int_freq=n_int_freq)
    _, rfi_A, _ = signal(ComplexRFIVarAntCoarse, config)
    phase = phase_on(config, rfi_A.real.dtype)
    comp = PolyInterpVis()
    comp.setup(config)
    reference = RiemannVis()
    reference.setup(config)

    got = run_vis(comp, vis_state(config, rfi_A, phase))
    fine = interpolate_fine(stencil_stack(rfi_A, comp.offsets, comp.stencil), jnp.asarray(comp.weights))
    want = run_vis(reference, vis_state(config, fine, phase))

    atol, rtol = tols()
    assert got.shape == (config.n_bl, config.n_freq, config.n_time)
    assert jnp.allclose(got, want, atol=atol * jnp.abs(want).max(), rtol=rtol)


def test_it_agrees_with_the_fine_grid_route_on_a_prior_draw():
    """Coarse signal + polynomial against fine signal + Riemann sum, same latent.

    Within a percent at a correlation time of seven integrations, and a few
    times closer with the 5-point stencil; and within a factor of two of the
    conditional mean, which is closer on average over draws by a fraction of
    the error -- the polynomial is its smooth limit -- but not on every draw.
    """
    config_fine, config_1, config_2 = make_config(), make_config(), make_config(gp_interp_stencil=2)
    _, rfi_A_fine, params = signal(ComplexRFIVarAnt, config_fine)
    _, rfi_A_1, _ = signal(ComplexRFIVarAntCoarse, config_1, params)
    _, rfi_A_2, _ = signal(ComplexRFIVarAntCoarse, config_2, params)
    phase = phase_on(config_fine, rfi_A_fine.real.dtype)

    reference = RiemannVis()
    reference.setup(config_fine)
    want = run_vis(reference, vis_state(config_fine, rfi_A_fine, phase))

    def coarse_route(cls, config, rfi_A):
        comp = cls()
        comp.setup(config)
        return run_vis(comp, vis_state(config, rfi_A, phase))

    err_1 = rel_rms(coarse_route(PolyInterpVis, config_1, rfi_A_1), want)
    err_2 = rel_rms(coarse_route(PolyInterpVis, config_2, rfi_A_2), want)
    err_gp = rel_rms(coarse_route(GPInterpVis, config_1, rfi_A_1), want)

    assert err_1 < 1e-2
    assert err_2 < err_1 / 2
    assert err_1 < 2 * err_gp and err_gp < 2 * err_1


def test_the_prior_degree_brings_it_to_the_conditional_mean():
    """At twice the interpolating degree the polynomial's weights are ``GPInterpVis``'s."""
    config = make_config(poly_interp_degree=4)
    _, rfi_A, _ = signal(ComplexRFIVarAntCoarse, config)
    phase = phase_on(config, rfi_A.real.dtype)
    poly = PolyInterpVis()
    poly.setup(config)
    gp = GPInterpVis()
    gp.setup(config)

    assert np.abs(poly.weights - gp.weights).max() < 1e-3
    got = run_vis(poly, vis_state(config, rfi_A, phase))
    want = run_vis(gp, vis_state(config, rfi_A, phase))
    assert rel_rms(got, want) < 1e-3


def test_the_time_block_and_jit_change_nothing():
    """The inherited forward under the subclass's constants: blocked and jitted against neither."""
    config = make_config(n_int_freq=2, time_block_size=3)
    _, rfi_A, _ = signal(ComplexRFIVarAntCoarse, config)
    state = vis_state(config, rfi_A, phase_on(config, rfi_A.real.dtype))
    blocked = PolyInterpVis()
    blocked.setup(config)
    config.args["rfi"]["time_block_size"] = None
    whole = PolyInterpVis()
    whole.setup(config)

    ref = run_vis(whole, state)
    got = run_vis(blocked, state)
    jitted = jax.jit(blocked.build_forward())({}, state, make_constants(blocked))["vis_rfi"]

    atol, rtol = tols()
    assert jnp.allclose(got, ref, atol=atol * jnp.abs(ref).max(), rtol=rtol)
    assert jnp.allclose(jitted, ref, atol=atol * jnp.abs(ref).max(), rtol=rtol)


def test_the_errors_name_the_component():
    config = make_config()
    _, rfi_A_fine, _ = signal(ComplexRFIVarAnt, config)
    comp = PolyInterpVis()
    comp.setup(config)

    with pytest.raises(ValueError, match="PolyInterpVis reads rfi_A on the data grid"):
        run_vis(comp, vis_state(config, rfi_A_fine, phase_on(config, rfi_A_fine.real.dtype)))


def test_the_data_grid_phase_route_is_the_fine_phase_route_on_the_rebuilt_phase():
    """The inherited phase reconstruction, under the subclass: one computation either way."""
    from .test_gp_interp_vis import path_state

    config = make_config(n_int_freq=2)
    _, rfi_A, _ = signal(ComplexRFIVarAntCoarse, config)
    phase_c, path, fine = path_state(config, rfi_A)
    comp = PolyInterpVis()
    comp.setup(config)

    from_fine = run_vis(comp, vis_state(config, rfi_A, fine))
    from_path = run_vis(comp, {**vis_state(config, rfi_A, phase_c), "rfi_path": path})

    atol, rtol = tols()
    assert jnp.allclose(from_path, from_fine, atol=atol * jnp.abs(from_fine).max(), rtol=rtol)
