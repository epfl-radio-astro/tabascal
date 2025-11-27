import jax
import ctypes
from functools import partial
import os
from jax.extend import core
from jax.interpreters import mlir, ad, xla
from jax.core import ShapedArray
import jax.numpy as jnp

dir_path = os.path.dirname(os.path.realpath(__file__))

tab_lib_path = f"{dir_path}/tabascal.so"
tab_lib_gpu_path = f"{dir_path}/tabascal_gpu.so"

if os.path.exists(tab_lib_path):
  tab_lib = ctypes.cdll.LoadLibrary(tab_lib_path)
  jax.ffi.register_ffi_target(
      "calc_rfi", jax.ffi.pycapsule(tab_lib.calc_rfi_vis_cpu), platform="cpu")
  jax.ffi.register_ffi_target(
      "calc_rfi_jvp", jax.ffi.pycapsule(tab_lib.calc_rfi_jvp_cpu), platform="cpu")
  jax.ffi.register_ffi_target(
      "calc_rfi_transpose", jax.ffi.pycapsule(tab_lib.calc_rfi_transpose_cpu), platform="cpu")
else:
  tab_lib = None

if os.path.exists(tab_lib_gpu_path):
  tab_lib_gpu = ctypes.cdll.LoadLibrary(tab_lib_gpu_path)
  jax.ffi.register_ffi_target(
      "calc_rfi_gpu", jax.ffi.pycapsule(tab_lib_gpu.calc_rfi_vis_gpu), platform="gpu")
  jax.ffi.register_ffi_target(
      "calc_rfi_jvp_gpu", jax.ffi.pycapsule(tab_lib_gpu.calc_rfi_jvp_gpu), platform="gpu")
  jax.ffi.register_ffi_target(
      "calc_rfi_transpose_gpu", jax.ffi.pycapsule(tab_lib_gpu.calc_rfi_transpose_gpu), platform="gpu")
else:
  tab_lib_gpu = None


def check_tab_lib():
  if tab_lib is None:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    raise RuntimeError(
        f"FFI selected, but tabascal.so not found! Compilation required, check included makefile at {dir_path}")

def check_tab_lib_gpu():
  if tab_lib_gpu is None:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    raise RuntimeError(
        f"FFI selected, but tabascal_gpu.so not found! Compilation required, check included makefile at {dir_path}")


rfi_transpose_op = core.Primitive("rfi_transpose_op")
rfi_transpose_op.def_impl(partial(xla.apply_primitive, rfi_transpose_op))
rfi_transpose_op.multiple_results=True

def rfi_transpose_abstract(a1, a2, rfi_amp_fine, rfi_phase, g):
    n_time = rfi_amp_fine.shape[4]
    n_freq = rfi_amp_fine.shape[2]
    n_bl = a1.shape[0]

    t1 = ShapedArray(rfi_amp_fine.shape, rfi_amp_fine.dtype)
    t2 = ShapedArray(rfi_phase.shape, rfi_phase.dtype)
    return (t1, t2)

rfi_transpose_op.def_abstract_eval(rfi_transpose_abstract)

def rfi_transpose_lowering_cpu(ctx, a1, a2, rfi_amp_fine, rfi_phase, g):
    check_tab_lib()
    res = jax.ffi.ffi_lowering("calc_rfi_transpose")
    return res(ctx, a1, a2, rfi_amp_fine, rfi_phase, g)

mlir.register_lowering(rfi_transpose_op, rfi_transpose_lowering_cpu, platform='cpu')

def rfi_transpose_lowering_gpu(ctx, a1, a2, rfi_amp_fine, rfi_phase, g):
    check_tab_lib_gpu()
    res = jax.ffi.ffi_lowering("calc_rfi_transpose_gpu")
    return res(ctx, a1, a2, rfi_amp_fine, rfi_phase, g)

mlir.register_lowering(rfi_transpose_op, rfi_transpose_lowering_gpu, platform='gpu')


rfi_jvp_op = core.Primitive("rfi_jvp_op")
rfi_jvp_op.def_impl(partial(xla.apply_primitive, rfi_jvp_op))

def rfi_jvp_abstract(a1, a2, rfi_amp_fine, rfi_amp_fine_grad, rfi_phase, rfi_phase_grad):
    # rfi_amp_fine and rfi_phase shape is
    # (n_rfi, n_ant, n_freq, n_int_freq, n_time, n_int_time)
    n_time = rfi_amp_fine.shape[4]
    n_freq = rfi_amp_fine.shape[2]
    n_bl = a1.shape[0]
    return ShapedArray([a1.shape[0], n_freq, n_time], rfi_amp_fine.dtype)

rfi_jvp_op.def_abstract_eval(rfi_jvp_abstract)

def rfi_jvp_lowering_cpu(ctx, a1, a2, rfi_amp_fine, rfi_amp_fine_grad, rfi_phase, rfi_phase_grad):
    check_tab_lib()
    res = jax.ffi.ffi_lowering("calc_rfi_jvp")
    return [res(ctx, a1, a2, rfi_amp_fine, rfi_amp_fine_grad, rfi_phase, rfi_phase_grad)]

mlir.register_lowering(rfi_jvp_op, rfi_jvp_lowering_cpu, platform='cpu')

def rfi_jvp_lowering_gpu(ctx, a1, a2, rfi_amp_fine, rfi_amp_fine_grad, rfi_phase, rfi_phase_grad):
    check_tab_lib_gpu()
    res = jax.ffi.ffi_lowering("calc_rfi_jvp_gpu")
    return [res(ctx, a1, a2, rfi_amp_fine, rfi_amp_fine_grad, rfi_phase, rfi_phase_grad)]

mlir.register_lowering(rfi_jvp_op, rfi_jvp_lowering_gpu, platform='gpu')


def rfi_jvp_transpose(g, a1, a2, rfi_amp_fine, rfi_amp_fine_grad, rfi_phase, rfi_phase_grad):
  t1, t2 = rfi_transpose_op.bind(a1, a2, rfi_amp_fine, rfi_phase, g)

  return None, None, t1, t1, t2, t2

ad.primitive_transposes[rfi_jvp_op] = rfi_jvp_transpose


rfi_vis_op = core.Primitive("rfi_vis_op")
rfi_vis_op.def_impl(partial(xla.apply_primitive, rfi_vis_op))

def rfi_vis_abstract(a1, a2, rfi_amp_fine, rfi_phase):
    # rfi_amp_fine and rfi_phase shape is
    # (n_rfi, n_ant, n_freq, n_int_freq, n_time, n_int_time)
    n_time = rfi_amp_fine.shape[4]
    n_freq = rfi_amp_fine.shape[2]
    n_bl = a1.shape[0]
    return ShapedArray([a1.shape[0], n_freq, n_time], rfi_amp_fine.dtype)

rfi_vis_op.def_abstract_eval(rfi_vis_abstract)

def rfi_vis_lowering_cpu(ctx, a1, a2, rfi_amp_fine, rfi_phase):
    check_tab_lib()
    res = jax.ffi.ffi_lowering("calc_rfi")
    return [res(ctx, a1, a2, rfi_amp_fine, rfi_phase)]


mlir.register_lowering(rfi_vis_op, rfi_vis_lowering_cpu, platform='cpu')

def rfi_vis_lowering_gpu(ctx, a1, a2, rfi_amp_fine, rfi_phase):
    check_tab_lib_gpu()
    res = jax.ffi.ffi_lowering("calc_rfi_gpu")
    return [res(ctx, a1, a2, rfi_amp_fine, rfi_phase)]


mlir.register_lowering(rfi_vis_op, rfi_vis_lowering_gpu, platform='gpu')

def rfi_vis_jvp(args, tangents):
  a1, a2, rfi_amp_fine, rfi_phase = args
  a1_dot, a2_dot, rfi_amp_fine_dot, rfi_phase_dot = tangents

  if type(rfi_amp_fine_dot) is ad.Zero:
      rfi_amp_fine_dot = jnp.zeros(rfi_amp_fine.shape, rfi_amp_fine.dtype)
  if type(rfi_phase_dot) is ad.Zero:
      rfi_phase_dot = jnp.zeros(rfi_phase.shape, rfi_phase.dtype)


  grad = rfi_jvp_op.bind(a1, a2, rfi_amp_fine, rfi_amp_fine_dot, rfi_phase, rfi_phase_dot)

  return rfi_vis_op.bind(a1, a2, rfi_amp_fine, rfi_phase), grad

ad.primitive_jvps[rfi_vis_op] = rfi_vis_jvp
