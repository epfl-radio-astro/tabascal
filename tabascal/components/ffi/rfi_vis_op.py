"""
Custom JAX primitives for RFI visibility calculation using FFI.
"""

import ctypes
import os
from functools import partial

import jax
import jax.numpy as jnp
from jax.core import ShapedArray
from jax.extend import core
from jax.interpreters import ad, mlir, xla

class RFIVisOp:
    """
    Operator for computing RFI visibility using JAX FFI with precomputed indices.
    
    This class encapsulates the antenna baseline indexing logic required for efficient
    RFI visibility calculations. It precomputes sorted indices and search positions
    for both antenna arrays (a1 and a2) to enable fast lookups in the FFI kernel.
    """
    
    def __init__(self, n_ant, a1, a2):
        """
        Initialize the RFI visibility operator with antenna baseline information.
        
        Args:
            n_ant: Number of antennas in the array.
            a1: Array of first antenna indices for each baseline (shape: [n_baselines]).
            a2: Array of second antenna indices for each baseline (shape: [n_baselines]).
        
        The initialization precomputes:
            - a1_sorter, a2_sorter: Indices that would sort the a1 and a2 arrays
            - a1_start, a2_start: Starting positions for each antenna in the sorted arrays
        """
        self.a1 = a1
        self.a2 = a2
        self.a1_sorter = jnp.argsort(a1).astype(a1.dtype)
        self.a2_sorter = jnp.argsort(a2).astype(a1.dtype)

        v = jnp.arange(0, n_ant, dtype=a1.dtype)
        self.a1_start = jnp.searchsorted(a1, v, sorter=self.a1_sorter)
        self.a2_start = jnp.searchsorted(a2, v, sorter=self.a2_sorter)
    
    def eval(self, rfi_amp_fine, rfi_phase):
        """
        Evaluate the RFI visibility for given RFI amplitudes and phases.
        
        Args:
            rfi_amp_fine: Fine-grained RFI amplitude array with shape
                         (n_rfi, n_ant, n_freq, n_int_freq, n_time, n_int_time).
            rfi_phase: RFI phase array with shape matching rfi_amp_fine.
        
        Returns:
            Array of RFI visibilities with shape (n_baselines, n_freq, n_time).
        """
        return rfi_vis_op.bind(
            self.a1, self.a1_sorter, self.a1_start,
            self.a2, self.a2_sorter, self.a2_start,
            rfi_amp_fine, rfi_phase
        )


def prepare_indices(n_ant, a1, a2):
    """
    Prepare indices for sorted usage in the C++ kernel.

    Args:
        n_ant: Number of antennas.
        a1: Antenna 1 indices.
        a2: Antenna 2 indices.

    Returns:
        A tuple (a1_sorter, a1_start, a2_sorter, a2_start) suitable for the kernel.
    """
    a1_sorter = jnp.argsort(a1).astype(a1.dtype)
    a2_sorter = jnp.argsort(a2).astype(a1.dtype)

    v = jnp.arange(0, n_ant, dtype=a1.dtype)
    a1_start = jnp.searchsorted(a1, v, sorter=a1_sorter)
    a2_start = jnp.searchsorted(a2, v, sorter=a2_sorter)

    return (a1_sorter, a1_start, a2_sorter, a2_start)


_DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def _load_library(name):
    lib_path = os.path.join(_DIR_PATH, name)
    if os.path.exists(lib_path):
        return ctypes.cdll.LoadLibrary(lib_path)
    return None


_TAB_LIB = _load_library("tabascal.so")
_TAB_LIB_GPU = _load_library("tabascal_gpu.so")

if _TAB_LIB:
    jax.ffi.register_ffi_target(
        "calc_rfi", jax.ffi.pycapsule(_TAB_LIB.calc_rfi_vis_cpu), platform="cpu"
    )
    jax.ffi.register_ffi_target(
        "calc_rfi_jvp", jax.ffi.pycapsule(_TAB_LIB.calc_rfi_jvp_cpu), platform="cpu"
    )
    jax.ffi.register_ffi_target(
        "calc_rfi_transpose",
        jax.ffi.pycapsule(_TAB_LIB.calc_rfi_transpose_cpu),
        platform="cpu",
    )

if _TAB_LIB_GPU:
    jax.ffi.register_ffi_target(
        "calc_rfi_gpu", jax.ffi.pycapsule(_TAB_LIB_GPU.calc_rfi_vis_gpu), platform="gpu"
    )
    jax.ffi.register_ffi_target(
        "calc_rfi_jvp_gpu",
        jax.ffi.pycapsule(_TAB_LIB_GPU.calc_rfi_jvp_gpu),
        platform="gpu",
    )
    jax.ffi.register_ffi_target(
        "calc_rfi_transpose_gpu",
        jax.ffi.pycapsule(_TAB_LIB_GPU.calc_rfi_transpose_gpu),
        platform="gpu",
    )


def _check_tab_lib():
    if _TAB_LIB is None:
        raise RuntimeError(
            f"FFI selected, but tabascal.so not found! "
            f"Compilation required, check included makefile at {_DIR_PATH}"
        )


def _check_tab_lib_gpu():
    if _TAB_LIB_GPU is None:
        raise RuntimeError(
            f"FFI selected, but tabascal_gpu.so not found! "
            f"Compilation required, check included makefile at {_DIR_PATH}"
        )


# --- Transpose Op ---

rfi_transpose_op = core.Primitive("rfi_transpose_op")
rfi_transpose_op.def_impl(partial(xla.apply_primitive, rfi_transpose_op))
rfi_transpose_op.multiple_results = True


def rfi_transpose_abstract(
    a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine, rfi_phase, g
):
    """
    Abstract evaluation for the RFI transpose operation.
    """
    # Shapes are implicitly checked by JAX mechanisms or assumed correct from lowering
    t1 = ShapedArray(rfi_amp_fine.shape, rfi_amp_fine.dtype)
    t2 = ShapedArray(rfi_phase.shape, rfi_phase.dtype)
    return (t1, t2)


rfi_transpose_op.def_abstract_eval(rfi_transpose_abstract)


def rfi_transpose_lowering_cpu(
    ctx, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine, rfi_phase, g
):
    _check_tab_lib()
    res = jax.ffi.ffi_lowering("calc_rfi_transpose")
    return res(
        ctx,
        a1,
        a1_sorter,
        a1_start,
        a2,
        a2_sorter,
        a2_start,
        rfi_amp_fine,
        rfi_phase,
        g,
    )


mlir.register_lowering(rfi_transpose_op, rfi_transpose_lowering_cpu, platform="cpu")


def rfi_transpose_lowering_gpu(
    ctx, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine, rfi_phase, g
):
    _check_tab_lib_gpu()
    res = jax.ffi.ffi_lowering("calc_rfi_transpose_gpu")
    return res(
        ctx,
        a1,
        a1_sorter,
        a1_start,
        a2,
        a2_sorter,
        a2_start,
        rfi_amp_fine,
        rfi_phase,
        g,
    )


mlir.register_lowering(rfi_transpose_op, rfi_transpose_lowering_gpu, platform="gpu")


# --- JVP Op ---

rfi_jvp_op = core.Primitive("rfi_jvp_op")
rfi_jvp_op.def_impl(partial(xla.apply_primitive, rfi_jvp_op))


def rfi_jvp_abstract(
    a1,
    a1_sorter,
    a1_start,
    a2,
    a2_sorter,
    a2_start,
    rfi_amp_fine,
    rfi_amp_fine_grad,
    rfi_phase,
    rfi_phase_grad,
):
    """
    Abstract evaluation for the RFI JVP operation.
    """
    # rfi_amp_fine and rfi_phase shape is
    # (n_rfi, n_ant, n_freq, n_int_freq, n_time, n_int_time)
    n_time = rfi_amp_fine.shape[4]
    n_freq = rfi_amp_fine.shape[2]
    # n_bl = a1.shape[0] # unused
    return ShapedArray([a1.shape[0], n_freq, n_time], rfi_amp_fine.dtype)


rfi_jvp_op.def_abstract_eval(rfi_jvp_abstract)


def rfi_jvp_lowering_cpu(
    ctx,
    a1,
    a1_sorter,
    a1_start,
    a2,
    a2_sorter,
    a2_start,
    rfi_amp_fine,
    rfi_amp_fine_grad,
    rfi_phase,
    rfi_phase_grad,
):
    _check_tab_lib()
    res = jax.ffi.ffi_lowering("calc_rfi_jvp")
    return [
        res(
            ctx,
            a1,
            a1_sorter,
            a1_start,
            a2,
            a2_sorter,
            a2_start,
            rfi_amp_fine,
            rfi_amp_fine_grad,
            rfi_phase,
            rfi_phase_grad,
        )
    ]


mlir.register_lowering(rfi_jvp_op, rfi_jvp_lowering_cpu, platform="cpu")


def rfi_jvp_lowering_gpu(
    ctx,
    a1,
    a1_sorter,
    a1_start,
    a2,
    a2_sorter,
    a2_start,
    rfi_amp_fine,
    rfi_amp_fine_grad,
    rfi_phase,
    rfi_phase_grad,
):
    _check_tab_lib_gpu()
    res = jax.ffi.ffi_lowering("calc_rfi_jvp_gpu")
    return [
        res(
            ctx,
            a1,
            a1_sorter,
            a1_start,
            a2,
            a2_sorter,
            a2_start,
            rfi_amp_fine,
            rfi_amp_fine_grad,
            rfi_phase,
            rfi_phase_grad,
        )
    ]


mlir.register_lowering(rfi_jvp_op, rfi_jvp_lowering_gpu, platform="gpu")


def rfi_jvp_transpose(
    g,
    a1,
    a1_sorter,
    a1_start,
    a2,
    a2_sorter,
    a2_start,
    rfi_amp_fine,
    rfi_amp_fine_grad,
    rfi_phase,
    rfi_phase_grad,
):
    """
    Transpose rule for the RFI JVP operation.
    """
    t1, t2 = rfi_transpose_op.bind(
        a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine, rfi_phase, g
    )

    return None, None, None, None, None, None, t1, t1, t2, t2


ad.primitive_transposes[rfi_jvp_op] = rfi_jvp_transpose


# --- Vis Op ---

rfi_vis_op = core.Primitive("rfi_vis_op")
rfi_vis_op.def_impl(partial(xla.apply_primitive, rfi_vis_op))


def rfi_vis_abstract(
    a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine, rfi_phase
):
    """
    Abstract evaluation for the RFI visibility operation.
    """
    n_time = rfi_amp_fine.shape[4]
    n_freq = rfi_amp_fine.shape[2]
    return ShapedArray([a1.shape[0], n_freq, n_time], rfi_amp_fine.dtype)


rfi_vis_op.def_abstract_eval(rfi_vis_abstract)


def rfi_vis_lowering_cpu(
    ctx, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine, rfi_phase
):
    _check_tab_lib()
    res = jax.ffi.ffi_lowering("calc_rfi")
    return [
        res(
            ctx,
            a1,
            a1_sorter,
            a1_start,
            a2,
            a2_sorter,
            a2_start,
            rfi_amp_fine,
            rfi_phase,
        )
    ]


mlir.register_lowering(rfi_vis_op, rfi_vis_lowering_cpu, platform="cpu")


def rfi_vis_lowering_gpu(
    ctx, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine, rfi_phase
):
    _check_tab_lib_gpu()
    res = jax.ffi.ffi_lowering("calc_rfi_gpu")
    return [
        res(
            ctx,
            a1,
            a1_sorter,
            a1_start,
            a2,
            a2_sorter,
            a2_start,
            rfi_amp_fine,
            rfi_phase,
        )
    ]


mlir.register_lowering(rfi_vis_op, rfi_vis_lowering_gpu, platform="gpu")


def rfi_vis_jvp(args, tangents):
    """
    JVP rule for the RFI visibility operation.
    """
    a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine, rfi_phase = args
    _, _, _, _, _, _, rfi_amp_fine_dot, rfi_phase_dot = tangents

    if isinstance(rfi_amp_fine_dot, ad.Zero):
        rfi_amp_fine_dot = jnp.zeros(rfi_amp_fine.shape, rfi_amp_fine.dtype)
    if isinstance(rfi_phase_dot, ad.Zero):
        rfi_phase_dot = jnp.zeros(rfi_phase.shape, rfi_phase.dtype)

    grad = rfi_jvp_op.bind(
        a1,
        a1_sorter,
        a1_start,
        a2,
        a2_sorter,
        a2_start,
        rfi_amp_fine,
        rfi_amp_fine_dot,
        rfi_phase,
        rfi_phase_dot,
    )

    return (
        rfi_vis_op.bind(
            a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine, rfi_phase
        ),
        grad,
    )


ad.primitive_jvps[rfi_vis_op] = rfi_vis_jvp
