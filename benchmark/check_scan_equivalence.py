"""Verify FourierGPRFIScan reproduces FourierGPRFI exactly.

The scan/remat variant is a pure implementation change, so its forward output and
its gradients must match the vmap version to floating-point noise. Runs small enough
to sit on a CPU, since the point is correctness rather than scale.

    pixi run -e dev python3 benchmark/check_scan_equivalence.py
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from types import SimpleNamespace

from tabascal.components.rfi_signal import FourierGPRFI, FourierGPRFIScan


def make_rfi_config(n_rfi=3, n_ant=4, n_freq=4, n_time=8, n_int_time=2, n_int_freq=1):
    """Minimal stand-in for TabConfig, mirroring tests/components/test_rfi_signal.py.

    Inlined rather than imported: that test module uses package-relative imports and
    is not loadable as a plain script.
    """
    freqs = jnp.linspace(1.4e9, 1.41e9, n_freq)
    times = jnp.linspace(0.0, 120.0, n_time)
    n_freq_fine, n_time_fine = n_freq * n_int_freq, n_time * n_int_time
    n_bl = n_ant * (n_ant - 1) // 2

    return SimpleNamespace(
        n_rfi=n_rfi, n_rfi_real=n_rfi, n_ant=n_ant, n_freq=n_freq, n_time=n_time,
        n_freq_fine=n_freq_fine, n_time_fine=n_time_fine,
        n_int_freq=n_int_freq, n_int_time=n_int_time,
        freqs=freqs, freqs_fine=jnp.linspace(freqs[0], freqs[-1], n_freq_fine),
        chan_width=float(freqs[1] - freqs[0]),
        times=times, times_fine=jnp.linspace(times[0], times[-1], n_time_fine),
        int_time=float(times[1] - times[0]),
        vis_obs=jnp.ones((n_bl, n_freq, n_time), dtype=complex),
        args={
            "rfi": {
                "r_seed": 1, "var": 1.0, "corr_freq": 5e6, "corr_time": 60.0,
                "init": "prior", "mean": "zeros", "est": None,
                "time_pad_factor": 2, "freq_pad_factor": 2,
            },
            "plots": {"truth": False},
            "data": {"zarr_path": None, "data_col": "DATA"},
        },
    )


def constants_for(comp):
    return {f"{comp.prefix}/{k}": v for k, v in comp.build_constants().items()}


def main() -> None:
    kwargs = dict(n_rfi=3, n_ant=4, n_freq=4, n_time=8, n_int_time=2)

    ref, scan = FourierGPRFI(), FourierGPRFIScan()
    ref.setup(make_rfi_config(**kwargs))
    scan.setup(make_rfi_config(**kwargs))

    assert set(ref.parameter_shapes) == set(scan.parameter_shapes), (
        "parameter sets differ -- the subclass changed the model, not just the forward"
    )

    keys = jax.random.split(jax.random.PRNGKey(0), len(ref.init_params_base))
    params = {
        name: jax.random.normal(key, ref.init_params_base[name].shape)
        for name, key in zip(sorted(ref.init_params_base), keys)
    }

    out_ref = ref.build_forward()(params, {}, constants_for(ref))["rfi_A"]
    out_scan = scan.build_forward()(params, {}, constants_for(scan))["rfi_A"]

    print(f"shapes: vmap {out_ref.shape}  scan {out_scan.shape}")
    assert out_ref.shape == out_scan.shape, "output shapes differ"
    max_abs = float(jnp.max(jnp.abs(out_ref - out_scan)))
    scale = float(jnp.max(jnp.abs(out_ref)))
    print(f"forward: max |diff| = {max_abs:.3e}  (max |value| = {scale:.3e})")
    assert max_abs <= 1e-10 * max(scale, 1.0), "forward outputs disagree"

    # Gradients matter as much as the forward: checkpoint changes how the tape is
    # built, so a bug there would only show up in reverse mode.
    def loss(p, comp):
        out = comp.build_forward()(p, {}, constants_for(comp))["rfi_A"]
        return jnp.sum(jnp.abs(out) ** 2)

    g_ref = jax.grad(lambda p: loss(p, ref))(params)
    g_scan = jax.grad(lambda p: loss(p, scan))(params)
    for name in sorted(g_ref):
        d = float(jnp.max(jnp.abs(g_ref[name] - g_scan[name])))
        s = float(jnp.max(jnp.abs(g_ref[name])))
        print(f"grad {name}: max |diff| = {d:.3e}  (max |grad| = {s:.3e})")
        assert d <= 1e-8 * max(s, 1.0), f"gradients disagree for {name}"

    print("\nOK: FourierGPRFIScan matches FourierGPRFI in value and gradient.")


if __name__ == "__main__":
    main()
