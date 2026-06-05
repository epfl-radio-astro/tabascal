#!/usr/bin/env python3
"""Benchmark the RFI visibility kernel: memory and runtime across baseline batching.

Compares the pure-JAX :func:`tabascal.interferometry.calculate_rfi_vis` against the
C++ FFI kernel, sweeping the ``batch_size`` knob that trades peak memory for speed:

  * ``vmap``      -> batch_size=None (single vmap over all baselines)
  * ``chunk<N>``  -> batch_size=N   (jax.lax.map blocks of N baselines)
  * ``scan``      -> batch_size=1   (per-baseline scan, minimum memory)
  * ``ffi``       -> the C++ kernel (reference for value and for the memory we want)
  * ``legacy``    -> the old gather-based path (materialises (n_bl, n_rfi, ...))

Two memory numbers are reported:
  * ``temp_MiB`` -- XLA's compile-time scratch estimate
    (``compiled.memory_analysis().temp_size_in_bytes``). Backend-accurate and
    independent of run order, so it is the metric to compare. On CPU XLA fuses
    the gather+reduce and this is ~0 for every variant; on GPU the legacy/vmap
    paths show the blow-up while chunk/scan stay bounded.
  * ``peak_MiB`` -- live device high-water mark
    (``device.memory_stats()['peak_bytes_in_use']``). This is a *process*
    high-water mark, so in all-variants mode it only ever rises. To get a clean
    per-variant peak, run one variant per process with ``--only`` (see below).

Usage
-----
    # All variants, all presets (CPU or GPU):
    pixi run python3 benchmarks/bench_rfi_vis.py

    # One preset, custom batch sweep:
    pixi run python3 benchmarks/bench_rfi_vis.py --preset large --batch-sizes 128 512 2048

    # Clean isolated peak GPU memory for a single variant (run once per variant):
    for v in legacy vmap chunk512 scan ffi; do
        pixi run python3 benchmarks/bench_rfi_vis.py --preset large --only $v
    done

    # Double precision:
    pixi run python3 benchmarks/bench_rfi_vis.py --x64
"""
import argparse
import time

import jax


# Problem-size presets: (n_ant, n_rfi, n_freq, n_time, n_int_freq, n_int_time).
# n_bl = n_ant*(n_ant-1)/2. The legacy/vmap intermediate scales as
# n_bl * n_rfi * (n_freq*n_int_freq) * (n_time*n_int_time) complex elements.
PRESETS = {
    "small": (64, 20, 16, 12, 4, 2),
    "mid": (64, 10, 64, 30, 4, 4),
    "large": (128, 8, 64, 60, 4, 4),
    "huge": (256, 8, 64, 60, 4, 4),
}


def make_inputs(n_ant, n_rfi, n_freq, n_time, n_int_freq, n_int_time, rdt, cdt, seed=0):
    import jax.numpy as jnp

    a1, a2 = jnp.triu_indices(n_ant, 1)
    a1, a2 = a1.astype("int32"), a2.astype("int32")
    shape = (n_rfi, n_ant, n_freq * n_int_freq, n_time * n_int_time)
    phase = jax.random.uniform(jax.random.PRNGKey(seed), shape).astype(rdt)
    amp = (
        jax.random.normal(jax.random.PRNGKey(seed + 1), shape)
        + 1j * jax.random.normal(jax.random.PRNGKey(seed + 2), shape)
    ).astype(cdt)
    return a1, a2, amp, phase


def to_contiguous(amp, phase, n_rfi, n_ant, n_freq, n_int_freq, n_time, n_int_time):
    """(n_rfi, n_ant, n_f_fine, n_t_fine) -> (n_ant, n_freq, n_time, n_rfi, n_int_freq, n_int_time)."""
    import jax.numpy as jnp

    new = (n_rfi, n_ant, n_freq, n_int_freq, n_time, n_int_time)
    perm = (1, 2, 4, 0, 3, 5)
    return (
        jnp.transpose(amp.reshape(new), perm),
        jnp.transpose(phase.reshape(new), perm),
    )


def legacy_kernel(amp, phase, a1, a2, n_freq, n_int_freq, n_time, n_int_time):
    """The pre-refactor gather-based path, on the original (n_rfi, n_ant, ...) layout."""
    import jax.numpy as jnp

    amp_ = jnp.swapaxes(amp, 0, 1)
    phase_ = jnp.swapaxes(phase, 0, 1)
    vis_fine = jnp.sum(
        amp_[a1] * jnp.conjugate(amp_[a2]) * jnp.exp(1j * (phase_[a1] - phase_[a2])),
        axis=1,
    )
    n_bl = a1.shape[0]
    return jnp.mean(
        jnp.reshape(vis_fine, (n_bl, n_freq, n_int_freq, n_time, n_int_time)),
        axis=(-3, -1),
    )


def temp_mib(compiled):
    try:
        return compiled.memory_analysis().temp_size_in_bytes / 2**20
    except Exception:
        return float("nan")


def peak_mib(device):
    try:
        return device.memory_stats().get("peak_bytes_in_use", float("nan")) / 2**20
    except Exception:
        return float("nan")


def timeit(fn, args, reps):
    out = jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / reps, out


def build_variants(amp, phase, amp_c, phase_c, a1, a2, dims, batch_sizes):
    """Map variant name -> (compiled_fn, call_args)."""
    import jax.numpy as jnp

    from tabascal.components.ffi.rfi_vis_op import RFIVisOp
    from tabascal.interferometry import calculate_rfi_vis

    n_ant, n_rfi, n_freq, n_time, n_int_freq, n_int_time = dims
    op = RFIVisOp(n_ant, a1, a2)

    variants = {}

    f = jax.jit(lambda A, P, x, y: legacy_kernel(A, P, x, y, n_freq, n_int_freq, n_time, n_int_time))
    variants["legacy"] = (f.lower(amp, phase, a1, a2).compile(), (amp, phase, a1, a2))

    f = jax.jit(lambda A, P, x, y: calculate_rfi_vis(A, P, x, y, None))
    variants["vmap"] = (f.lower(amp_c, phase_c, a1, a2).compile(), (amp_c, phase_c, a1, a2))

    for bs in batch_sizes:
        f = jax.jit(lambda A, P, x, y, _bs=bs: calculate_rfi_vis(A, P, x, y, _bs))
        variants[f"chunk{bs}"] = (f.lower(amp_c, phase_c, a1, a2).compile(), (amp_c, phase_c, a1, a2))

    f = jax.jit(lambda A, P, x, y: calculate_rfi_vis(A, P, x, y, 1))
    variants["scan"] = (f.lower(amp_c, phase_c, a1, a2).compile(), (amp_c, phase_c, a1, a2))

    f = jax.jit(lambda A, P: op.eval(A, P))
    variants["ffi"] = (f.lower(amp_c, phase_c).compile(), (amp_c, phase_c))

    return variants


def run_preset(name, dims, batch_sizes, reps, only, rdt, cdt, device):
    import jax.numpy as jnp

    n_ant, n_rfi, n_freq, n_time, n_int_freq, n_int_time = dims
    a1, a2, amp, phase = make_inputs(*dims, rdt, cdt)
    amp_c, phase_c = to_contiguous(amp, phase, n_rfi, n_ant, n_freq, n_int_freq, n_time, n_int_time)
    n_bl = int(a1.shape[0])

    intermediate = n_bl * n_rfi * (n_freq * n_int_freq) * (n_time * n_int_time)
    bytes_per = 16 if cdt == jnp.complex128 else 8
    print(
        f"\n=== {name}: n_ant={n_ant} n_bl={n_bl} n_rfi={n_rfi} "
        f"n_freq={n_freq}x{n_int_freq} n_time={n_time}x{n_int_time} | {rdt.__name__} ===\n"
        f"    legacy/vmap intermediate ~= {intermediate * bytes_per / 2**20:.0f} MiB if not fused"
    )

    variants = build_variants(amp, phase, amp_c, phase_c, a1, a2, dims, batch_sizes)

    if only is not None:
        if only not in variants:
            raise SystemExit(f"--only {only!r} not in {list(variants)}")
        variants = {only: variants[only]}

    ref = None if only else jax.block_until_ready(variants["ffi"][0](*variants["ffi"][1]))

    print(f"{'impl':10} {'temp_MiB':>10} {'peak_MiB':>10} {'time_ms':>10} {'max_err':>10}")
    for vname, (compiled, args) in variants.items():
        t, out = timeit(compiled, args, reps)
        err = "-" if ref is None else f"{float(jnp.max(jnp.abs(out - ref))):.2e}"
        print(f"{vname:10} {temp_mib(compiled):10.2f} {peak_mib(device):10.2f} {t * 1e3:10.3f} {err:>10}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=list(PRESETS) + ["all"], default="all")
    p.add_argument("--size", help="custom size 'n_ant,n_rfi,n_freq,n_time,n_int_freq,n_int_time' (overrides --preset)")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[256, 512, 1024])
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--only", help="run a single variant (e.g. legacy, vmap, chunk512, scan, ffi) for a clean isolated peak")
    p.add_argument("--x64", action="store_true", help="run in double precision")
    args = p.parse_args()

    jax.config.update("jax_enable_x64", args.x64)
    import jax.numpy as jnp

    rdt = jnp.float64 if args.x64 else jnp.float32
    cdt = jnp.complex128 if args.x64 else jnp.complex64
    device = jax.devices()[0]
    print(f"jax {jax.__version__}  x64={args.x64}  device={device}  ({device.platform})")

    if args.size:
        sizes = {"custom": tuple(int(x) for x in args.size.split(","))}
    elif args.preset == "all":
        sizes = PRESETS
    else:
        sizes = {args.preset: PRESETS[args.preset]}

    for name, dims in sizes.items():
        run_preset(name, dims, args.batch_sizes, args.reps, args.only, rdt, cdt, device)


if __name__ == "__main__":
    main()
