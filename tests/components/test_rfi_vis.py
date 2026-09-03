import pytest
from types import SimpleNamespace
from tabascal.components.rfi_vis import *
import jax.numpy as jnp
import jax
import numpy as np

from ri_kernels.jax_api import RFIVisOp
from tabascal.interferometry import (
    calculate_rfi_vis_blocked,
    calculate_rfi_vis_fine,
    get_divisors,
)
from .conftest import active_precision, make_constants


def create_config(
    n_ant,
    n_rfi,
    n_time,
    n_freq,
    n_int_time,
    n_int_freq,
    precision=None,
    rfi_args=None,
):
    a1, a2 = jnp.triu_indices(n_ant, 1)
    a1 = a1.astype('int32')
    a2 = a2.astype('int32')
    return SimpleNamespace(
        n_ant=n_ant,
        n_rfi=n_rfi,
        n_time=n_time,
        n_freq=n_freq,
        n_int_time=n_int_time,
        n_int_freq=n_int_freq,
        n_bl=a1.shape[0],
        a1=a1,
        a2=a2,
        precision=precision or active_precision(),
        # Deliberately empty by default: the integration sample counts are read
        # off the bound TabConfig attributes, never out of the raw config dict.
        # ``rfi_args`` is for the keys that genuinely do live in the config dict,
        # i.e. rfi.baseline_block_size.
        args={"rfi": dict(rfi_args or {})},
    )


def create_state(config, rand_vis_rfi=False, r_key=42, real_dtype=jnp.float64):
    complex_dtype = jnp.complex64 if real_dtype == jnp.float32 else jnp.complex128
    n_int_freq = config.n_int_freq
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

    atol, rtol = _tols(real_dtype)
    assert ffi_result.dtype == ref_result.dtype
    assert jnp.allclose(ref_result, ffi_result, atol=atol, rtol=rtol)


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

    atol, rtol = _tols(real_dtype)
    assert ffi_result.dtype == ref_result.dtype
    assert jnp.allclose(ref_result, ffi_result, atol=atol, rtol=rtol)


@pytest.mark.parametrize("n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq", test_sizes)
def test_ffi_vjp(n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq):
    """Reverse-mode VJP gradients w.r.t. rfi_A and rfi_phase match between FFI and reference."""
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
    n_int_freq = config.n_int_freq
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


# ---------------------------------------------------------------------------
# Blocked baseline scan
#
# RiemannVis walks the baseline axis in blocks of rfi.baseline_block_size under
# jax.checkpoint, so the (n_bl, n_rfi, n_freq_fine, n_time_fine) fine grid the
# Riemann sum is built from is never formed for every baseline at once, and is
# recomputed in the backward pass rather than kept. Baselines are independent, so
# the block size is a memory strategy and nothing else: the tests below pin that
# it changes neither the value nor the gradient, and that the fine grid really
# does stay off the tape.
# ---------------------------------------------------------------------------


def dense_vis_rfi(state, config):
    """The unblocked reduction: the whole fine grid, then the fine->coarse mean.

    What ``RiemannVis`` computed before the scan, written out here so the blocked
    kernel is held to the formula rather than only to itself.
    """
    vis_fine = calculate_rfi_vis_fine(
        state["rfi_A"], state["rfi_phase"], config.a1, config.a2
    )
    new_shape = (
        config.n_bl,
        config.n_freq,
        config.n_int_freq,
        config.n_time,
        config.n_int_time,
    )
    return jnp.mean(jnp.reshape(vis_fine, new_shape), axis=(-3, -1))


def reverse_pass_residual_bytes(f, *primals):
    """The bytes the forward pass leaves behind for the reverse pass.

    ``jax.linearize`` runs the forward pass once and returns the linear map that
    the backward pass transposes; everything the forward pass saved for it is
    closed into that map as a constant. Summing those constants is therefore a
    direct measurement of the tape, without reference to any backend's allocator.
    """
    _, f_lin = jax.linearize(f, *primals)
    tangents = jax.tree.map(jnp.zeros_like, primals)
    jaxpr = jax.make_jaxpr(f_lin)(*tangents)

    return sum(c.size * c.dtype.itemsize for c in jaxpr.consts)


def state_bytes(state):
    return sum(x.size * x.dtype.itemsize for x in state.values())


# n_bl for these sizes is 1, 6 and 2016, so the block sizes below span "one
# baseline at a time", blocks that do and do not divide n_bl, blocks larger than
# the whole axis, and None -- a single block, which is the setting that keeps the
# checkpoint and drops the scan.
block_sizes = [1, 5, 128, 4096, None]


@pytest.mark.parametrize("n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq", test_sizes)
@pytest.mark.parametrize("block_size", block_sizes)
def test_blocked_kernel_matches_the_unblocked_reduction(
    n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq, block_size
):
    """The scanned kernel reproduces the fine-grid formula, at any block size.

    Including block sizes that leave a ragged last block, whose padding baselines
    are antenna 0 against itself and must be dropped rather than summed in.
    """
    config = create_config(n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq)
    real_dtype = _session_dtype()
    state = create_state(config, False, 42, real_dtype=real_dtype)

    blocked = calculate_rfi_vis_blocked(
        state["rfi_A"],
        state["rfi_phase"],
        config.a1,
        config.a2,
        n_int_freq,
        n_int_time,
        block_size,
    )
    dense = dense_vis_rfi(state, config)

    atol, rtol = _tols(real_dtype)
    assert blocked.shape == (config.n_bl, n_freq, n_time)
    assert blocked.dtype == dense.dtype
    assert jnp.allclose(blocked, dense, atol=atol, rtol=rtol)


@pytest.mark.parametrize("block_size", block_sizes)
def test_the_block_size_changes_neither_the_value_nor_the_gradient(block_size):
    """The component's output and its VJP match the unblocked reduction's.

    The reference is the fine-grid formula, not the same component at some other
    block size: a defect that every block shares -- a systematically dropped
    conjugate, a mis-shaped reduction -- would agree with itself at every setting
    and only disagree with the formula. The gradient case is not redundant
    either: ``checkpoint`` changes how the tape is built, so a padding baseline
    reaching the result, or a residual captured across steps, shows in reverse
    mode alone.
    """
    sizes = (64, 4, 8, 4, 4, 2)
    real_dtype = _session_dtype()
    config = create_config(*sizes, rfi_args={"baseline_block_size": block_size})
    state = create_state(config, False, 42, real_dtype=real_dtype)
    cotangent = create_state(config, True, 50, real_dtype=real_dtype)["vis_rfi"]

    impl = RiemannVis()
    impl.setup(config)
    constants = make_constants(impl)
    forward = impl.build_forward()

    def value_and_grads(f):
        primal, vjp = jax.vjp(f, state)
        (grads,) = vjp(cotangent)

        return primal, grads

    vis, grads = value_and_grads(lambda s: forward({}, s, constants)["vis_rfi"])
    ref_vis, ref_grads = value_and_grads(lambda s: dense_vis_rfi(s, config))

    atol, rtol = _tols(real_dtype)
    assert jnp.allclose(vis, ref_vis, atol=atol, rtol=rtol)
    for key in ("rfi_A", "rfi_phase"):
        assert grads[key].dtype == ref_grads[key].dtype
        assert jnp.allclose(grads[key], ref_grads[key], atol=atol, rtol=rtol)


def test_the_fine_grid_is_not_kept_for_the_reverse_pass():
    """The tape holds the inputs and the result, not the fine grid.

    The point of the change, measured rather than asserted structurally: the
    unblocked reduction saves intermediates of shape
    ``(n_bl, n_rfi, n_freq_fine, n_time_fine)``, and the blocked kernel saves
    nothing carrying both the baseline axis and the fine grid. What it does keep
    is a transposed copy of the per-antenna inputs, sized by ``n_ant``.
    """
    block_size = 16
    config = create_config(64, 4, 8, 4, 4, 2, rfi_args={"baseline_block_size": block_size})
    real_dtype = _session_dtype()
    state = create_state(config, False, 42, real_dtype=real_dtype)

    impl = RiemannVis()
    impl.setup(config)
    constants = make_constants(impl)
    forward = impl.build_forward()

    blocked_bytes = reverse_pass_residual_bytes(
        lambda s: forward({}, s, constants)["vis_rfi"], state
    )
    dense_bytes = reverse_pass_residual_bytes(lambda s: dense_vis_rfi(s, config), state)

    # The transposed inputs, and whatever the compiler keeps beside them. The
    # bound is the state the caller already holds -- both fine-grid inputs plus a
    # vis_rfi -- with a factor of two for headroom rather than as a measurement:
    # the blocked residual sits at roughly one copy of the two inputs.
    assert blocked_bytes < 2 * state_bytes(state)
    # And the comparison the assertion above is only meaningful against: the
    # unblocked form of the same reduction really does keep the fine grid, which
    # here is n_rfi * n_int_freq * n_int_time = 32 times the visibilities.
    assert dense_bytes > 8 * blocked_bytes


def test_the_default_block_size_is_used_when_the_config_does_not_set_one():
    """A config predating the key still builds, on the base default of 128."""
    config = create_config(4, 2, 3, 5, 4, 3)
    assert config.args["rfi"] == {}

    impl = RiemannVis()
    impl.setup(config)

    assert impl.baseline_block_size == 128


def test_a_null_block_size_is_one_block_over_every_baseline():
    """null is accepted, kept as None, and equals a block that spans the axis.

    The two are the same computation -- a block size at or above ``n_bl`` clamps
    to it -- so this pins the spelling, which is the point of the setting: it
    says "no scan" without the config having to know how many baselines there
    are.
    """
    sizes = (64, 4, 8, 4, 4, 2)
    real_dtype = _session_dtype()
    config = create_config(*sizes, rfi_args={"baseline_block_size": None})
    state = create_state(config, False, 42, real_dtype=real_dtype)

    impl = RiemannVis()
    impl.setup(config)
    assert impl.baseline_block_size is None

    vis = impl.build_forward()({}, state, make_constants(impl))["vis_rfi"]
    whole_axis = calculate_rfi_vis_blocked(
        state["rfi_A"],
        state["rfi_phase"],
        config.a1,
        config.a2,
        config.n_int_freq,
        config.n_int_time,
        config.n_bl,
    )

    assert jnp.array_equal(vis, whole_axis)


def test_a_null_block_size_bounds_the_tape_but_not_the_peak():
    """What null is for, and what it gives up, in one measurement.

    The checkpoint is still there with no scan around it, so the fine grid stays
    off the tape -- the residuals match a small block's. What it no longer does
    is bound the array itself: the backward pass rebuilds every baseline at once
    instead of one block at a time. Only the first half is measurable here, the
    peak being the compiler's business, so the second is left to
    ``docs/kernels.md`` and the reference numbers.
    """
    sizes = (64, 4, 8, 4, 4, 2)
    real_dtype = _session_dtype()

    def residual_bytes(block):
        config = create_config(*sizes, rfi_args={"baseline_block_size": block})
        state = create_state(config, False, 42, real_dtype=real_dtype)
        impl = RiemannVis()
        impl.setup(config)
        constants = make_constants(impl)
        forward = impl.build_forward()

        return reverse_pass_residual_bytes(
            lambda s: forward({}, s, constants)["vis_rfi"], state
        ), state

    null_bytes, state = residual_bytes(None)
    block_bytes, _ = residual_bytes(16)

    assert null_bytes < 2 * state_bytes(state)
    assert null_bytes == block_bytes


@pytest.mark.parametrize(
    "block_size", [0, -1, 1.5, True, "128", float("inf"), float("nan")]
)
def test_a_baseline_block_size_that_is_not_a_positive_whole_number_is_rejected(
    block_size,
):
    """Rejected in setup, by name, rather than silently rounded or ignored.

    ``.inf`` and ``.nan`` are in the list because yaml can spell them and because
    they are the two that reach ``int()`` before any comparison does. ``None`` is
    deliberately not in it: null is a setting, covered below.
    """
    config = create_config(4, 2, 3, 5, 4, 3, rfi_args={"baseline_block_size": block_size})

    with pytest.raises(RuntimeError, match="baseline_block_size"):
        RiemannVis().setup(config)


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


@pytest.mark.parametrize(
    "Impl", [RiemannVis, RiemannVisFFI, RiemannVisVariable, RiemannVisVariableFFI]
)
def test_integration_sample_counts_come_from_the_bound_config(Impl):
    """``n_int_freq`` and ``n_int_time`` are read off the TabConfig, symmetrically.

    The frequency count used to be read straight out of ``config.args["rfi"]``
    under a second spelling, which both duplicated the bound attribute the rest
    of the model uses and made a config that never set that spelling fail in
    setup. Reading the bound attribute is what keeps the fine grid these
    components reshape and the one ``TabConfig`` built the same grid.
    """

    config = make_variable_config(4, 2, 3, 5, 4, 3, strides=[1, 2])
    assert config.args["rfi"] == {}

    impl = Impl()
    impl.setup(config)

    assert impl.n_int_freq == config.n_int_freq == 3
    assert impl.n_int_time == config.n_int_time == 4


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

    atol, rtol = _tols(real_dtype)
    assert var_result.shape == ref_result.shape
    assert var_result.dtype == ref_result.dtype
    assert jnp.allclose(ref_result, var_result, atol=atol, rtol=rtol)


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

    atol, rtol = _tols(real_dtype)
    assert var_result.shape == ref_result.shape
    assert var_result.dtype == ref_result.dtype
    assert jnp.allclose(ref_result, var_result, atol=atol, rtol=rtol)


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

    atol, rtol = _tols(real_dtype)
    assert ffi_result.shape == (config.n_bl, config.n_freq, config.n_time)
    assert ffi_result.dtype == ref_result.dtype
    assert jnp.allclose(ref_result, ffi_result, atol=atol, rtol=rtol)


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

    atol, rtol = _tols(real_dtype)
    assert ffi_result.dtype == ref_result.dtype
    assert jnp.allclose(ref_result, ffi_result, atol=atol, rtol=rtol)


@pytest.mark.parametrize("n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq", test_sizes)
def test_variable_ffi_matches_variable_vjp(
    n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq
):
    """Reverse-mode VJP gradients w.r.t. rfi_A and rfi_phase match between kernels."""
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

    atol, rtol = _tols(real_dtype)
    assert ffi_state["rfi_A"].dtype == ref_state["rfi_A"].dtype
    assert ffi_state["rfi_phase"].dtype == ref_state["rfi_phase"].dtype
    assert jnp.allclose(ref_state["rfi_A"], ffi_state["rfi_A"], atol=atol, rtol=rtol)
    assert jnp.allclose(ref_state["rfi_phase"], ffi_state["rfi_phase"], atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "Impl", [RiemannVisVariable, RiemannVisVariableFFI]
)
def test_variable_accumulates_into_state(Impl):
    """forward adds vis_rfi onto the incoming state rather than overwriting it."""
    config = make_variable_config(64, 20, 16, 12, 4, 2, strides=[1, 2, 4])
    real_dtype = _session_dtype()
    atol, rtol = _tols(real_dtype)

    impl = Impl()
    impl.setup(config)
    constants = make_constants(impl)
    forward = impl.build_forward()

    zero_state = create_state(config, rand_vis_rfi=False, r_key=42, real_dtype=real_dtype)
    seeded_state = create_state(config, rand_vis_rfi=True, r_key=42, real_dtype=real_dtype)

    from_zero = forward({}, zero_state, constants)["vis_rfi"]
    from_seed = forward({}, seeded_state, constants)["vis_rfi"]

    # The two runs share rfi_A/rfi_phase (same r_key), so the only difference is
    # the pre-existing vis_rfi that forward should add on top.
    expected = from_zero + seeded_state["vis_rfi"]
    assert jnp.allclose(from_seed, expected, atol=atol, rtol=rtol)
