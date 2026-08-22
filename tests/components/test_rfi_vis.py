import pytest
from types import SimpleNamespace
from tabascal.components.rfi_vis import *
import jax.numpy as jnp
import jax
import numpy as np

from ri_kernels.jax_api import RFIDelayVisOp
from tabascal.interferometry import get_divisors
from .conftest import active_precision, make_constants


# A MeerKAT-like L-band grid: 1.227 GHz start, 209 kHz channels, with the
# n_int_freq fine samples of a channel stored consecutively (as TabConfig's
# domain_ss grid does), which is the (n_freq, n_int_freq) split the kernels use.
_START_FREQ_HZ = 1.227e9
_CHAN_WIDTH_HZ = 209e3

# Centred per-antenna delays are at baseline scale: +-16.7 us is a 5 km radius,
# half the 10 km the single-precision delay kernel is specified for.
_MAX_DELAY_US = 16.7


def create_config(n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq, precision=None):
    a1, a2 = jnp.triu_indices(n_ant, 1)
    a1 = a1.astype('int32')
    a2 = a2.astype('int32')
    freqs_fine = _START_FREQ_HZ + _CHAN_WIDTH_HZ * (
        np.arange(n_freq * n_int_freq, dtype=np.float64) / n_int_freq
    )
    return SimpleNamespace(
        n_ant=n_ant,
        n_rfi=n_rfi,
        n_time=n_time,
        n_freq=n_freq,
        n_int_time=n_int_time,
        n_bl=a1.shape[0],
        a1=a1,
        a2=a2,
        freqs_fine=freqs_fine,
        precision=precision or active_precision(),
        args={"rfi": {"freq_int_samples": n_int_freq}},
    )


def create_state(config, rand_vis_rfi=False, r_key=42, real_dtype=jnp.float64):
    complex_dtype = jnp.complex64 if real_dtype == jnp.float32 else jnp.complex128
    n_int_freq = config.args["rfi"]["freq_int_samples"]
    input_shape = (config.n_rfi, config.n_ant, config.n_freq * n_int_freq, config.n_time * config.n_int_time)
    delay_shape = (config.n_rfi, config.n_ant, config.n_time * config.n_int_time)
    output_shape = (config.n_bl, config.n_freq, config.n_time)
    rfi_delay_us = jax.random.uniform(
        jax.random.PRNGKey(r_key), delay_shape, minval=-_MAX_DELAY_US, maxval=_MAX_DELAY_US
    ).astype(real_dtype)
    rfi_amp = (
        jax.random.normal(jax.random.PRNGKey(r_key + 2), input_shape)
        + 1.0j * jax.random.normal(jax.random.PRNGKey(r_key + 3), input_shape)
    ).astype(complex_dtype)

    state = {"rfi_A": rfi_amp, "rfi_delay_us": rfi_delay_us, "vis_rfi": jnp.zeros(output_shape, dtype=complex_dtype)}
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


def _session_dtype():
    """Real dtype matching the session precision (driven by the --x64 flag).

    The FFI kernel runs in whichever precision its inputs carry, and the suite
    fixes one precision per run via --x64. So the comparison tests below use fp32
    under --x64 false and fp64 under --x64 true, each with matching tolerances,
    rather than exercising both dtypes in a single session (a float64 request
    would silently downcast to float32 under --x64 false anyway).
    """
    return jnp.float64 if active_precision() == "double" else jnp.float32


def _tols(dtype):
    return (1e-4, 1e-4) if dtype == jnp.float32 else (1e-8, 1e-8)


def _assert_close(ref, got, dtype):
    """Elementwise closeness with the absolute tolerance scaled to the array.

    The RFI visibility and its tangents are sums over sources and fine samples
    of terms that are individually large (the delay enters through an angular
    frequency of ~7.7e3 rad/us), so an element can be much smaller than the
    terms that cancel to produce it. An absolute tolerance fixed at ``rtol`` of
    the *largest* element, rather than of the element itself, is the right
    yardstick for that rounding; it is what ri_kernels' own tests use too.
    """
    atol, rtol = _tols(dtype)
    scale = max(float(jnp.max(jnp.abs(ref))), 1.0)
    assert jnp.allclose(ref, got, atol=atol * scale, rtol=rtol)


@pytest.mark.parametrize("n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq", test_sizes)
def test_ffi(n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq):
    """FFI and reference Riemann kernels produce identical vis_rfi outputs."""
    config = create_config(n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq)
    real_dtype = _session_dtype()

    def compute_vis_rfi(impl):
        state = create_state(config, False, 42, real_dtype=real_dtype)
        impl.setup(config)
        return impl.build_forward()({}, state, make_constants(impl))["vis_rfi"]

    ref_result = compute_vis_rfi(RiemannVis())
    ffi_result = compute_vis_rfi(RiemannVisFFI())
    assert ffi_result.dtype == ref_result.dtype
    _assert_close(ref_result, ffi_result, real_dtype)


@pytest.mark.parametrize("n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq", test_sizes)
def test_ffi_jvp(n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq):
    """Forward-mode Jacobian-vector products of FFI and reference kernels match."""
    config = create_config(n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq)
    real_dtype = _session_dtype()

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

    ref_result = compue_jvp(RiemannVis())
    ffi_result = compue_jvp(RiemannVisFFI())
    assert ffi_result.dtype == ref_result.dtype
    _assert_close(ref_result, ffi_result, real_dtype)


@pytest.mark.parametrize("n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq", test_sizes)
def test_ffi_vjp(n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq):
    """Reverse-mode VJP gradients w.r.t. rfi_A and rfi_delay_us match between FFI and reference."""
    config = create_config(n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq)
    real_dtype = _session_dtype()

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



    ref_state = compue_vjp(RiemannVis())
    ffi_state = compue_vjp(RiemannVisFFI())
    assert ffi_state["rfi_A"].dtype == ref_state["rfi_A"].dtype
    assert ffi_state["rfi_delay_us"].dtype == ref_state["rfi_delay_us"].dtype
    _assert_close(ref_state["rfi_A"], ffi_state["rfi_A"], real_dtype)
    _assert_close(ref_state["rfi_delay_us"], ffi_state["rfi_delay_us"], real_dtype)


@pytest.mark.requires_double
def test_mixed_precision_rejected():
    """Mismatched amp/delay precision is rejected at the lowering boundary.

    Needs x64 enabled to construct a genuine float64 delay array; under
    ``--x64 false`` the float64 request downcasts to float32 and there is no
    mismatch to reject.
    """
    config = create_config(4, 2, 3, 3, 2, 2)
    n_int_freq = config.args["rfi"]["freq_int_samples"]
    input_shape = (config.n_rfi, config.n_ant, config.n_freq * n_int_freq, config.n_time * config.n_int_time)
    delay_shape = (config.n_rfi, config.n_ant, config.n_time * config.n_int_time)
    rfi_amp = (
        jax.random.normal(jax.random.PRNGKey(0), input_shape)
        + 1.0j * jax.random.normal(jax.random.PRNGKey(1), input_shape)
    ).astype(jnp.complex64)
    rfi_delay_us = jax.random.uniform(jax.random.PRNGKey(2), delay_shape).astype(jnp.float64)
    freqs_mhz = jnp.asarray(config.freqs_fine / 1e6, dtype=jnp.float64).reshape(config.n_freq, n_int_freq)

    amp_shape = (config.n_rfi, config.n_ant, config.n_freq, n_int_freq, config.n_time, config.n_int_time)
    rfi_amp = jnp.transpose(rfi_amp.reshape(amp_shape), (1, 2, 4, 0, 3, 5))
    delay_shape = (config.n_rfi, config.n_ant, config.n_time, config.n_int_time)
    rfi_delay_us = jnp.transpose(rfi_delay_us.reshape(delay_shape), (1, 2, 0, 3))

    op = RFIDelayVisOp(config.n_ant, config.a1, config.a2)
    with pytest.raises(TypeError):
        op.eval(rfi_amp, rfi_delay_us, freqs_mhz)


# ---------------------------------------------------------------------------
# Variable (per-baseline) time sampling
#
# RiemannVisVariable and RiemannVisVariableFFI split the baselines into groups,
# each integrated over a coarser time stride. The config
# carries this grouping as ``time_sample_idxs`` (the baseline indices in each
# group) and ``time_strides`` (the matching integration-time stride per group).
# ---------------------------------------------------------------------------


def make_variable_groups(n_bl, candidate_strides):
    """Partition ``n_bl`` baselines across ``candidate_strides`` (round-robin).

    Round-robin assignment guarantees every baseline lands in exactly one group
    (so ``vis_rfi`` is fully written, never left uninitialised) and that no group
    is empty, which keeps the FFI operator's index precomputation well defined.

    Returns ``(idxs, strides)`` where ``idxs[i]`` are the int32 baseline indices
    for group ``i`` and ``strides[i]`` its integration-time stride.
    """
    assign = np.array(
        [candidate_strides[i % len(candidate_strides)] for i in range(n_bl)]
    )
    u_strides = sorted({int(s) for s in assign})
    idxs = [np.where(assign == s)[0].astype("int32") for s in u_strides]
    return idxs, u_strides


def make_variable_config(
    n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq, strides, precision=None
):
    config = create_config(
        n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq, precision=precision
    )
    idxs, u_strides = make_variable_groups(config.n_bl, strides)
    config.time_sample_idxs = idxs
    config.time_strides = u_strides
    return config


@pytest.mark.parametrize("n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq", test_sizes)
def test_variable_single_group_matches_reference(
    n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq
):
    """One group spanning all baselines at stride 1 == full-resolution reference.

    With a single stride-1 group every integration sample is kept and averaged,
    so the variable kernel must reproduce the dense RiemannVis exactly
    (up to floating-point rounding).
    """
    config = make_variable_config(
        n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq, strides=[1]
    )
    real_dtype = _session_dtype()

    def compute_vis_rfi(impl):
        state = create_state(config, False, 42, real_dtype=real_dtype)
        impl.setup(config)
        return impl.build_forward()({}, state, make_constants(impl))["vis_rfi"]

    ref_result = compute_vis_rfi(RiemannVis())
    var_result = compute_vis_rfi(RiemannVisVariable())
    assert var_result.shape == ref_result.shape
    assert var_result.dtype == ref_result.dtype
    _assert_close(ref_result, var_result, real_dtype)


@pytest.mark.parametrize("n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq", test_sizes)
def test_variable_ffi_single_group_matches_reference(
    n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq
):
    """The FFI variable kernel matches the dense FFI reference at stride 1."""
    config = make_variable_config(
        n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq, strides=[1]
    )
    real_dtype = _session_dtype()

    def compute_vis_rfi(impl):
        state = create_state(config, False, 42, real_dtype=real_dtype)
        impl.setup(config)
        return impl.build_forward()({}, state, make_constants(impl))["vis_rfi"]

    ref_result = compute_vis_rfi(RiemannVisFFI())
    var_result = compute_vis_rfi(RiemannVisVariableFFI())
    assert var_result.shape == ref_result.shape
    assert var_result.dtype == ref_result.dtype
    _assert_close(ref_result, var_result, real_dtype)


@pytest.mark.parametrize("n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq", test_sizes)
def test_variable_ffi_matches_variable(
    n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq
):
    """FFI and pure-JAX variable kernels agree under genuine multi-group striding.

    The strides are the divisors of n_int_time, so each group's subsample slice
    divides the integration axis evenly and both implementations index the same
    samples.
    """
    strides = [int(s) for s in get_divisors(n_int_time)]
    config = make_variable_config(
        n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq, strides=strides
    )
    real_dtype = _session_dtype()

    def compute_vis_rfi(impl):
        state = create_state(config, False, 42, real_dtype=real_dtype)
        impl.setup(config)
        return impl.build_forward()({}, state, make_constants(impl))["vis_rfi"]

    ref_result = compute_vis_rfi(RiemannVisVariable())
    ffi_result = compute_vis_rfi(RiemannVisVariableFFI())
    assert ffi_result.shape == (config.n_bl, config.n_freq, config.n_time)
    assert ffi_result.dtype == ref_result.dtype
    _assert_close(ref_result, ffi_result, real_dtype)


@pytest.mark.parametrize("n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq", test_sizes)
def test_variable_ffi_matches_variable_jvp(
    n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq
):
    """Forward-mode JVPs of both variable kernels match (FFI custom JVP vs autodiff)."""
    strides = [int(s) for s in get_divisors(n_int_time)]
    config = make_variable_config(
        n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq, strides=strides
    )
    real_dtype = _session_dtype()

    def compute_jvp(impl):
        state = create_state(config, False, r_key=42, real_dtype=real_dtype)
        tangents_state = create_state(config, False, r_key=50, real_dtype=real_dtype)
        impl.setup(config)
        constants = make_constants(impl)
        _, tangents = jax.jvp(
            lambda s: impl.build_forward()({}, s, constants),
            (state,), (tangents_state,)
        )
        return tangents["vis_rfi"]

    ref_result = compute_jvp(RiemannVisVariable())
    ffi_result = compute_jvp(RiemannVisVariableFFI())
    assert ffi_result.dtype == ref_result.dtype
    _assert_close(ref_result, ffi_result, real_dtype)


@pytest.mark.parametrize("n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq", test_sizes)
def test_variable_ffi_matches_variable_vjp(
    n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq
):
    """Reverse-mode VJP gradients w.r.t. rfi_A and rfi_delay_us match between kernels."""
    strides = [int(s) for s in get_divisors(n_int_time)]
    config = make_variable_config(
        n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq, strides=strides
    )
    real_dtype = _session_dtype()

    def compute_vjp(impl):
        input_state = create_state(config, False, r_key=42, real_dtype=real_dtype)
        vjp_state = create_state(config, True, r_key=50, real_dtype=real_dtype)
        impl.setup(config)
        constants = make_constants(impl)
        _, vjp_func = jax.vjp(
            lambda s: impl.build_forward()({}, s, constants), input_state
        )
        (output_state,) = vjp_func(vjp_state)
        return output_state

    ref_state = compute_vjp(RiemannVisVariable())
    ffi_state = compute_vjp(RiemannVisVariableFFI())
    assert ffi_state["rfi_A"].dtype == ref_state["rfi_A"].dtype
    assert ffi_state["rfi_delay_us"].dtype == ref_state["rfi_delay_us"].dtype
    _assert_close(ref_state["rfi_A"], ffi_state["rfi_A"], real_dtype)
    _assert_close(ref_state["rfi_delay_us"], ffi_state["rfi_delay_us"], real_dtype)


@pytest.mark.parametrize(
    "Impl", [RiemannVisVariable, RiemannVisVariableFFI]
)
def test_variable_accumulates_into_state(Impl):
    """forward adds vis_rfi onto the incoming state rather than overwriting it."""
    config = make_variable_config(64, 20, 16, 12, 4, 2, strides=[1, 2, 4])
    real_dtype = _session_dtype()

    impl = Impl()
    impl.setup(config)
    constants = make_constants(impl)
    forward = impl.build_forward()

    zero_state = create_state(config, rand_vis_rfi=False, r_key=42, real_dtype=real_dtype)
    seeded_state = create_state(config, rand_vis_rfi=True, r_key=42, real_dtype=real_dtype)

    from_zero = forward({}, zero_state, constants)["vis_rfi"]
    from_seed = forward({}, seeded_state, constants)["vis_rfi"]

    # The two runs share rfi_A/rfi_delay_us (same r_key), so the only difference is
    # the pre-existing vis_rfi that forward should add on top.
    expected = from_zero + seeded_state["vis_rfi"]
    _assert_close(from_seed, expected, real_dtype)
