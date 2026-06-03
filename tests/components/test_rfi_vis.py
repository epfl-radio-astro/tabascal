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


def create_state(config, rand_vis_rfi=False, r_key=42, real_dtype=jnp.float64):
    complex_dtype = jnp.complex64 if real_dtype == jnp.float32 else jnp.complex128
    n_int_freq = config.args["rfi"]["freq_int_samples"]
    input_shape = (config.n_rfi, config.n_ant, config.n_freq * n_int_freq, config.n_time * config.n_int_time)
    output_shape = (config.n_bl, config.n_freq, config.n_time)
    rfi_phase = jax.random.uniform(jax.random.PRNGKey(r_key), input_shape).astype(real_dtype)
    rfi_amp = (
        jax.random.normal(jax.random.PRNGKey(r_key + 2), input_shape)
        + 1.0j * jax.random.normal(jax.random.PRNGKey(r_key + 3), input_shape)
    ).astype(complex_dtype)

    state = {"rfi_A": rfi_amp, "rfi_phase": rfi_phase, "vis_rfi": jnp.zeros(output_shape, dtype=complex_dtype)}
    if rand_vis_rfi:
        state["vis_rfi"] = (
            jax.random.normal(jax.random.PRNGKey(r_key + 4), output_shape)
            + 1.0j * jax.random.normal(jax.random.PRNGKey(r_key + 5), output_shape)
        ).astype(complex_dtype)
    else:
        state["vis_rfi"] = jnp.zeros(output_shape, dtype=complex_dtype)

    return state


# First case uses 2 antennas so it exercises a real baseline rather than the
# degenerate zero-baseline configuration.
test_sizes = [(2, 1, 1, 1, 1, 1), (4, 5, 6, 7, 8, 9), (64, 20, 16, 12, 4, 2)]

# The FFI kernel runs in whichever precision the inputs carry. float32 is always
# exercised; float64 only when JAX has x64 enabled (--x64 true), since otherwise
# a float64 request is silently downcast to float32 and the case would be a
# redundant float32 run held to a fp64 tolerance.
test_dtypes = [
    jnp.float32,
    pytest.param(
        jnp.float64,
        marks=pytest.mark.skipif(
            not jax.config.read("jax_enable_x64"),
            reason="float64 requires --x64 true",
        ),
    ),
]


def _tols(dtype):
    return (1e-4, 1e-4) if dtype == jnp.float32 else (1e-8, 1e-8)


@pytest.mark.parametrize("n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq", test_sizes)
@pytest.mark.parametrize("real_dtype", test_dtypes)
def test_ffi(n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq, real_dtype):
    """FFI and reference Riemann kernels produce identical vis_rfi outputs."""
    config = create_config(n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq)

    def compute_vis_rfi(impl):
        state = create_state(config, False, 42, real_dtype=real_dtype)
        impl.setup(config)
        return impl.build_forward()({}, state, make_constants(impl))["vis_rfi"]

    ref_result = compute_vis_rfi(RiemannVisTimeFreqCalculation())
    ffi_result = compute_vis_rfi(RiemannVisTimeFreqCalculationFFI())

    atol, rtol = _tols(real_dtype)
    assert ffi_result.dtype == ref_result.dtype
    assert jnp.allclose(ref_result, ffi_result, atol=atol, rtol=rtol)


@pytest.mark.parametrize("n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq", test_sizes)
@pytest.mark.parametrize("real_dtype", test_dtypes)
def test_ffi_jvp(n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq, real_dtype):
    """Forward-mode Jacobian-vector products of FFI and reference kernels match."""
    config = create_config(n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq)


    def compue_jvp(impl):
        state = create_state(config, False, r_key=42, real_dtype=real_dtype)
        tangents_state = create_state(config, False, r_key=50, real_dtype=real_dtype)
        impl.setup(config)
        constants = make_constants(impl)
        _, tangents = jax.jvp(
            lambda s: impl.build_forward()({}, s, constants),
            (state,), (tangents_state,)
        )
        return tangents["vis_rfi"]

    ref_result = compue_jvp(RiemannVisTimeFreqCalculation())
    ffi_result = compue_jvp(RiemannVisTimeFreqCalculationFFI())

    atol, rtol = _tols(real_dtype)
    assert ffi_result.dtype == ref_result.dtype
    assert jnp.allclose(ref_result, ffi_result, atol=atol, rtol=rtol)


@pytest.mark.parametrize("n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq", test_sizes)
@pytest.mark.parametrize("real_dtype", test_dtypes)
def test_ffi_vjp(n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq, real_dtype):
    """Reverse-mode VJP gradients w.r.t. rfi_A and rfi_phase match between FFI and reference."""
    config = create_config(n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq)


    def compue_vjp(impl):
        input_state = create_state(config, False, r_key=42, real_dtype=real_dtype)
        vjp_state = create_state(config, True, r_key=50, real_dtype=real_dtype)
        impl.setup(config)
        constants = make_constants(impl)
        primal_state, vjp_func = jax.vjp(
            lambda s: impl.build_forward()({}, s, constants), input_state
        )

        (output_state,) = vjp_func(vjp_state)

        return output_state



    ref_state = compue_vjp(RiemannVisTimeFreqCalculation())
    ffi_state = compue_vjp(RiemannVisTimeFreqCalculationFFI())

    atol, rtol = _tols(real_dtype)
    assert ffi_state["rfi_A"].dtype == ref_state["rfi_A"].dtype
    assert ffi_state["rfi_phase"].dtype == ref_state["rfi_phase"].dtype
    assert jnp.allclose(ref_state["rfi_A"], ffi_state["rfi_A"], atol=atol, rtol=rtol)
    assert jnp.allclose(ref_state["rfi_phase"], ffi_state["rfi_phase"], atol=atol, rtol=rtol)


@pytest.mark.requires_double
def test_mixed_precision_rejected():
    """Mismatched amp/phase precision is rejected at the lowering boundary.

    Needs x64 enabled to construct a genuine float64 phase array; under
    ``--x64 false`` the float64 request downcasts to float32 and there is no
    mismatch to reject.
    """
    config = create_config(4, 2, 3, 3, 2, 2)
    n_int_freq = config.args["rfi"]["freq_int_samples"]
    input_shape = (config.n_rfi, config.n_ant, config.n_freq * n_int_freq, config.n_time * config.n_int_time)
    rfi_amp = (
        jax.random.normal(jax.random.PRNGKey(0), input_shape)
        + 1.0j * jax.random.normal(jax.random.PRNGKey(1), input_shape)
    ).astype(jnp.complex64)
    rfi_phase = jax.random.uniform(jax.random.PRNGKey(2), input_shape).astype(jnp.float64)

    new_shape = (config.n_rfi, config.n_ant, config.n_freq, n_int_freq, config.n_time, config.n_int_time)
    rfi_amp = jnp.transpose(rfi_amp.reshape(new_shape), (1, 2, 4, 0, 3, 5))
    rfi_phase = jnp.transpose(rfi_phase.reshape(new_shape), (1, 2, 4, 0, 3, 5))

    op = RFIVisOp(config.n_ant, config.a1, config.a2)
    with pytest.raises(TypeError):
        op.eval(rfi_amp, rfi_phase)
