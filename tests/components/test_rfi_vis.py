import pytest
from types import SimpleNamespace
from tabascal.components.rfi_vis import *
import jax.numpy as jnp
import jax

from .conftest import active_precision, make_constants


def create_config(n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq, precision=None):
    a1, a2 = jnp.triu_indices(n_ant, 1)
    a1 = a1.astype('int32')
    a2 = a2.astype('int32')
    return SimpleNamespace(
        n_ant=n_ant,
        n_rfi=n_rfi,
        n_time=n_time,
        n_freq=n_freq,
        n_int_time=n_int_time,
        n_bl=a1.shape[0],
        a1=a1,
        a2=a2,
        precision=precision or active_precision(),
        args={"rfi": {"freq_int_samples": n_int_freq}},
    )


def create_state(config, rand_vis_rfi = False, r_key = 42):
    n_int_freq = config.args["rfi"]["freq_int_samples"]
    input_shape = (config.n_rfi, config.n_ant, config.n_freq * n_int_freq, config.n_time * config.n_int_time)
    output_shape = (config.n_bl, config.n_freq, config.n_time)
    rfi_phase = jax.random.uniform(jax.random.PRNGKey(r_key), input_shape)
    rfi_amp = jax.random.normal(
        jax.random.PRNGKey(r_key + 2), input_shape) + 1.0j * jax.random.normal(jax.random.PRNGKey(r_key + 3), input_shape)

    state = {"rfi_A": rfi_amp, "rfi_phase": rfi_phase, "vis_rfi": jnp.zeros(output_shape, dtype=rfi_amp.dtype)}
    if rand_vis_rfi:
        state["vis_rfi"] = jax.random.normal(
        jax.random.PRNGKey(r_key + 4), output_shape) + 1.0j * jax.random.normal(jax.random.PRNGKey(r_key + 5), output_shape)
    else:
        state["vis_rfi"] = jnp.zeros(output_shape, dtype = rfi_amp.dtype)

    return state


test_sizes = [(1, 1, 1, 1, 1, 1), (4, 5, 6, 7, 8, 9), (64, 20, 16, 12, 4, 2)]

@pytest.mark.requires_double
@pytest.mark.parametrize("n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq", test_sizes)
def test_ffi(n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq):
    """FFI and reference Riemann kernels produce identical vis_rfi outputs."""
    config = create_config(n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq)

    def compute_vis_rfi(impl):
        state = create_state(config, False, 42)
        impl.setup(config)
        return impl.build_forward()({}, state, make_constants(impl))["vis_rfi"]

    ref_result = compute_vis_rfi(RiemannVisTimeFreqCalculation())
    ffi_result = compute_vis_rfi(RiemannVisTimeFreqCalculationFFI())


    assert jnp.allclose(ref_result, ffi_result)

@pytest.mark.requires_double
@pytest.mark.parametrize("n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq", test_sizes)
def test_ffi_jvp(n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq):
    """Forward-mode Jacobian-vector products of FFI and reference kernels match."""
    config = create_config(n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq)


    def compue_jvp(impl):
        state = create_state(config, False, r_key=42)
        tangents_state = create_state(config, False, r_key=50)
        impl.setup(config)
        constants = make_constants(impl)
        _, tangents = jax.jvp(
            lambda s: impl.build_forward()({}, s, constants),
            (state,), (tangents_state,)
        )
        return tangents["vis_rfi"]

    ref_result = compue_jvp(RiemannVisTimeFreqCalculation())
    ffi_result = compue_jvp(RiemannVisTimeFreqCalculationFFI())

    assert jnp.allclose(ref_result, ffi_result)


@pytest.mark.requires_double
@pytest.mark.parametrize("n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq", test_sizes)
def test_ffi_vjp(n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq):
    """Reverse-mode VJP gradients w.r.t. rfi_A and rfi_phase match between FFI and reference."""
    config = create_config(n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq)


    def compue_vjp(impl):
        input_state = create_state(config, False, r_key=42)
        vjp_state = create_state(config, True, r_key=50)
        impl.setup(config)
        constants = make_constants(impl)
        primal_state, vjp_func = jax.vjp(
            lambda s: impl.build_forward()({}, s, constants), input_state
        )

        (output_state,) = vjp_func(vjp_state)

        return output_state



    ref_state = compue_vjp(RiemannVisTimeFreqCalculation())
    ffi_state = compue_vjp(RiemannVisTimeFreqCalculationFFI())

    assert jnp.allclose(ref_state["rfi_A"], ffi_state["rfi_A"])
    assert jnp.allclose(ref_state["rfi_phase"], ffi_state["rfi_phase"])
