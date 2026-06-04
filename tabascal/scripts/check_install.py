"""Self-check for a tabascal environment.

Verifies, for the active environment:
  1. the FFI shared libraries that the loader can find (libtabascal.so and the
     GPU variant libtabascal_cuda.so / libtabascal_hip.so), and
  2. that the RFI-visibility FFI kernel actually *executes* on every device JAX
     sees (CPU always; CUDA/ROCm GPU when present).

Running the kernel — not just loading the library — is the real test: on a GPU
device the op lowers through the GPU FFI path, so a missing/!broken GPU kernel
fails loudly here instead of deep inside a run.

Exit status is non-zero if a required kernel is missing or fails, so this is
usable as a CI / smoke gate. The common failure is a GPU env whose CUDA kernel
was never built (uv shares the CPU editable build across envs); the fix it
prints is ``pixi run -e <env> build-ffi-cuda``.
"""

import os
import sys


def _small_inputs():
    import jax.numpy as jnp

    n_ant = 2
    # Baseline antenna indices are int32 in the model (see TabConfig / the
    # component tests); the FFI handler is decoded against that width.
    a1 = jnp.array([0], dtype=jnp.int32)
    a2 = jnp.array([1], dtype=jnp.int32)
    # (n_rfi, n_ant, n_freq, n_int_freq, n_time, n_int_time)
    shape = (1, n_ant, 1, 1, 1, 1)
    amp = jnp.ones(shape, dtype=complex)
    phase = jnp.zeros(shape)
    return n_ant, a1, a2, amp, phase


def _run_on_device(dev):
    """Execute the RFI-vis FFI kernel on ``dev``; return (ok, detail)."""
    import jax
    from tabascal.components.ffi.rfi_vis_op import RFIVisOp

    n_ant, a1, a2, amp, phase = _small_inputs()
    try:
        with jax.default_device(dev):
            op = RFIVisOp(n_ant, a1, a2)
            out = jax.jit(op.eval)(amp, phase)
            out.block_until_ready()
        import numpy as np

        if not np.all(np.isfinite(np.asarray(out))):
            return False, "kernel ran but returned non-finite values"
        return True, f"output shape {tuple(out.shape)}"
    except Exception as e:  # noqa: BLE001 - report any failure verbatim
        msg = str(e).strip().splitlines()
        return False, msg[0] if msg else type(e).__name__


def main():
    import jax

    # The FFI kernels are compiled for complex128 (std::complex<double>), so the
    # kernel-execution check must run in double precision regardless of the env's
    # default. This is an availability check, not a precision check.
    jax.config.update("jax_enable_x64", True)

    from tabascal.components.ffi import rfi_vis_op as ffi

    print("tabascal install check")
    print("======================")
    print(f"python   : {sys.version.split()[0]}")
    print(f"jax      : {jax.__version__}")
    print(f"x64      : {jax.config.read('jax_enable_x64')} (forced on; FFI kernels are double-only)")

    devices = jax.devices()
    platforms = sorted({d.platform for d in devices})
    print(f"devices  : {devices}")
    print(f"platforms: {', '.join(platforms)}")
    print()

    def _first_existing(name):
        for d in ffi._candidate_dirs():
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
        return None

    print("FFI libraries (as resolved by the loader):")
    cpu_path = _first_existing("libtabascal.so")
    gpu_path = _first_existing(ffi._TAB_LIB_GPU_NAME)
    print(f"  libtabascal.so        : {'LOADED  ' + cpu_path if ffi._TAB_LIB else 'MISSING'}")
    print(f"  {ffi._TAB_LIB_GPU_NAME:<22}: "
          f"{'LOADED  ' + gpu_path if ffi._TAB_LIB_GPU else 'MISSING'}")
    print()

    has_gpu = any(p in ("gpu", "cuda", "rocm") for p in platforms)

    print("Kernel execution:")
    failures = []
    for dev in devices:
        ok, detail = _run_on_device(dev)
        status = "OK  " if ok else "FAIL"
        print(f"  [{dev.platform:<4}] calc_rfi : {status} - {detail}")
        if not ok:
            failures.append(dev.platform)
    print()

    if failures:
        print(f"RESULT: FAIL - kernel did not run on: {', '.join(failures)}")
        if any(p in ("gpu", "cuda", "rocm") for p in failures):
            print("  GPU kernel missing/broken. Build it for this env with:")
            print("    pixi run -e <env> build-ffi-cuda   # e.g. -e cuda12")
        return 1

    if has_gpu:
        print("RESULT: PASS - CPU and GPU FFI kernels available and working.")
    else:
        print("RESULT: PASS - CPU FFI kernel available and working "
              "(no GPU device detected in this env).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
