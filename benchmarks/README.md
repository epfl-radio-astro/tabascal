# Benchmarks

Standalone performance/memory scripts. Not part of the test suite — run them by hand.

## `bench_rfi_vis.py`

Benchmarks the RFI visibility kernel
([`tabascal.interferometry.calculate_rfi_vis`](../tabascal/interferometry.py)) and the
C++ FFI kernel, sweeping the `batch_size` knob that trades peak memory for speed
over baselines.

### Variants compared

| Variant | `batch_size` | What it is |
|---------|--------------|------------|
| `legacy` | — | The pre-refactor gather path; materialises `(n_bl, n_rfi, …)`. |
| `vmap` | `None` | Single `vmap` over all baselines (the default). |
| `chunk<N>` | `N` | `jax.lax.map` over blocks of `N` baselines. |
| `scan` | `1` | Per-baseline scan (minimum memory). |
| `ffi` | — | The C++ FFI kernel (reference for value and memory). |

### What it reports

- **`temp_MiB`** — XLA's compile-time scratch estimate
  (`compiled.memory_analysis().temp_size_in_bytes`). Backend-accurate and
  independent of run order, so this is the metric to compare. On CPU XLA fuses
  the gather+reduce and it is ~0 for every variant; on GPU the `legacy`/`vmap`
  paths show the blow-up while `chunk`/`scan` stay bounded.
- **`peak_MiB`** — live device high-water mark
  (`device.memory_stats()['peak_bytes_in_use']`). Populates on GPU; `nan` on CPU.
  It is a *process* high-water mark, so in all-variants mode it only ever rises —
  use `--only` (below) for a clean per-variant peak.
- **`time_ms`** — mean wall-clock per call.
- **`max_err`** — max abs difference vs the FFI kernel.

It also prints the theoretical un-fused intermediate size per preset, so you can
sanity-check the memory numbers.

### Usage

```bash
# Sweep all variants across all presets (CPU or GPU):
pixi run python3 benchmarks/bench_rfi_vis.py

# One preset, custom batch sweep (use this to pick rfi.rfi_vis_batch_size):
pixi run python3 benchmarks/bench_rfi_vis.py --preset large --batch-sizes 128 512 2048

# Clean isolated peak device memory — one variant per process. peak_MiB is a
# process high-water mark, so this is the accurate way to measure it on GPU:
for v in legacy vmap chunk512 scan ffi; do
    pixi run python3 benchmarks/bench_rfi_vis.py --preset large --only $v
done

# Double precision:
pixi run python3 benchmarks/bench_rfi_vis.py --x64
```

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--preset {small,mid,large,huge,all}` | `all` | Problem-size preset. `huge` is 256 antennas (32640 baselines). |
| `--size n_ant,n_rfi,n_freq,n_time,n_int_freq,n_int_time` | — | Custom shape (overrides `--preset`). |
| `--batch-sizes N [N …]` | `256 512 1024` | Block sizes to benchmark as `chunk<N>`. |
| `--reps N` | `20` | Timed repetitions per variant. |
| `--only NAME` | — | Run a single variant (e.g. `legacy`, `vmap`, `chunk512`, `scan`, `ffi`) for a clean isolated peak. |
| `--x64` | off | Run in double precision. |

### Interpreting the results

- The `legacy` row is the before; `vmap`/`chunk`/`scan` are the after. On GPU,
  compare their `temp_MiB`/`peak_MiB` to quantify the memory saving.
- The `chunk<N>` sweep tells you what to set
  [`rfi.rfi_vis_batch_size`](../tabascal/data/config/tab_config_base.yaml) to: the
  smallest `N` that fits your GPU memory budget while staying close to `vmap`/`ffi`
  on `time_ms`.
- `ffi` is the speed/memory target the pure-JAX path is chasing.
