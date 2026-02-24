import pytest
from tabascal.components.rfi_vis import *
from collections import namedtuple
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)

def create_config(n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq):
    a1, a2 = jnp.triu_indices(n_ant, 1)
    a1 = a1.astype('int32')
    a2 = a2.astype('int32')

    n_bl = a1.shape[0]
    config = namedtuple("config", ["n_ant", "n_rfi", "n_time", "n_freq", "n_int_time", "n_int_freq", "n_bl","a1","a2"])
    config.n_ant = n_ant
    config.n_rfi = n_rfi
    config.n_time = n_time
    config.n_freq = n_freq
    config.n_int_time = n_int_time
    config.args = {"rfi": {"freq_int_samples": n_int_freq}}
    config.n_bl = n_bl
    config.a1 = a1
    config.a2 = a2
    return config


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

@pytest.mark.parametrize("n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq", test_sizes)
def test_ffi(n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq):
    config = create_config(n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq)

    def compute_vis_rfi(impl):
        state = create_state(config, False, 42)
        impl.setup(config)
        prefix = impl.prefix
        for key, value in impl.build_constants().items():
            state[f"{prefix}/{key}"] = value
        return impl.build_forward()({}, state)["vis_rfi"]

    ref_result = compute_vis_rfi(RiemannVisTimeFreqCalculation())
    ffi_result = compute_vis_rfi(RiemannVisTimeFreqCalculationFFI())


    assert jnp.allclose(ref_result, ffi_result)

@pytest.mark.parametrize("n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq", test_sizes)
def test_ffi_jvp(n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq):

    config = create_config(n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq)


    def compue_jvp(impl):
        state = create_state(config, False, r_key = 42)
        tangents_state = create_state(config, False, r_key = 50)
        impl.setup(config)
        prefix = impl.prefix
        for key, value in impl.build_constants().items():
            state[f"{prefix}/{key}"] = value
            if jnp.issubdtype(value.dtype, jnp.integer) or jnp.issubdtype(value.dtype, jnp.bool_):
                tangents_state[f"{prefix}/{key}"] = jnp.zeros(value.shape, dtype=jax.dtypes.float0)
            else:
                tangents_state[f"{prefix}/{key}"] = jnp.zeros_like(value)
        _, tangents = jax.jvp(impl.build_forward(), ({}, state), ({}, tangents_state))

        return tangents["vis_rfi"]

    ref_result = compue_jvp(RiemannVisTimeFreqCalculation())
    ffi_result = compue_jvp(RiemannVisTimeFreqCalculationFFI())

    assert jnp.allclose(ref_result, ffi_result)


@pytest.mark.parametrize("n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq", test_sizes)
def test_ffi_vjp(n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq):

    config = create_config(n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq)


    def compue_vjp(impl):
        input_state = create_state(config, False, r_key = 42)
        vjp_state = create_state(config, True, r_key = 50)
        impl.setup(config)
        prefix = impl.prefix
        for key, value in impl.build_constants().items():
            input_state[f"{prefix}/{key}"] = value
            vjp_state[f"{prefix}/{key}"] = jnp.zeros_like(value)
        primal_state, vjp_func = jax.vjp(impl.build_forward(), {}, input_state)

        _, output_state = vjp_func(vjp_state)

        return output_state



    ref_state = compue_vjp(RiemannVisTimeFreqCalculation())
    ffi_state = compue_vjp(RiemannVisTimeFreqCalculationFFI())

    assert jnp.allclose(ref_state["rfi_A"], ffi_state["rfi_A"])
    assert jnp.allclose(ref_state["rfi_phase"], ffi_state["rfi_phase"])
