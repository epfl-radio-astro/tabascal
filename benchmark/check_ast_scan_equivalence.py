"""Verify FourierTimeFreqGPAstScan reproduces FourierTimeFreqGPAst exactly.

Pure implementation change, so forward values and gradients must match to
floating-point noise.

    pixi run -e dev python3 benchmark/check_ast_scan_equivalence.py
"""

import jax

jax.config.update("jax_enable_x64", True)

from types import SimpleNamespace

import jax.numpy as jnp

from tabascal.components.ast_vis import (
    FourierTimeFreqGPAst,
    FourierTimeFreqGPAstScan,
)


def make_ast_config(n_ant=8, n_freq=8, n_time=12):
    n_bl = n_ant * (n_ant - 1) // 2
    freqs = jnp.linspace(1.4e8, 1.5e8, n_freq)
    times = jnp.linspace(0.0, 120.0, n_time)
    key = jax.random.PRNGKey(0)
    # (n_time, n_bl, 3): get_ast_fringe_rate maxes over axis 0 to get per-baseline U.
    uvw = jax.random.normal(key, (n_time, n_bl, 3)) * 200.0

    return SimpleNamespace(
        n_time=n_time, n_bl=n_bl, n_freq=n_freq,
        int_time=float(times[1] - times[0]),
        chan_width=float(freqs[1] - freqs[0]),
        dish_d=35.0, uvw=uvw, freqs=freqs, times=times,
        vis_obs=jnp.ones((n_bl, n_freq, n_time), dtype=complex),
        args={
            "ast": {
                "init": "prior", "mean": 0,
                "pow_spec": {
                    "p0": 3e3, "k0_freq": 1, "fov_deg": 5,
                    "gammas": [5, 5], "cutoff": 1e-6,
                },
                "freq_pad_factor": 2.0, "time_pad_factor": 2.0,
            },
            "plots": {"truth": False},
            "data": {"zarr_path": None, "data_col": "DATA"},
        },
    )


def constants_for(comp):
    return {f"{comp.prefix}/{k}": v for k, v in comp.build_constants().items()}


def main() -> None:
    ref = FourierTimeFreqGPAst()
    ref.setup(make_ast_config())

    # Copy the parent's setup state so the two differ only in build_forward; this
    # rules out a config difference masquerading as agreement (or disagreement).
    scan = FourierTimeFreqGPAstScan()
    scan.__dict__.update(ref.__dict__)
    # Force several blocks: at the default max_block_size a small test would fit in
    # one block, which is just the vmap path again and would test nothing.
    scan.max_block_size = 4

    keys = jax.random.split(jax.random.PRNGKey(3), 2)
    shape = ref.init_params_base["ast_k_r_base"].shape
    params = {
        "ast_k_r_base": jax.random.normal(keys[0], shape),
        "ast_k_i_base": jax.random.normal(keys[1], shape),
    }
    zeros = {"vis_ast": jnp.zeros((ref.n_bl, ref.n_freq, ref.n_time), dtype=complex)}

    out_ref = ref.build_forward()(params, zeros, constants_for(ref))["vis_ast"]
    out_scan = scan.build_forward()(params, zeros, constants_for(scan))["vis_ast"]

    print(f"shapes: vmap {out_ref.shape}  scan {out_scan.shape}")
    assert out_ref.shape == out_scan.shape
    d = float(jnp.max(jnp.abs(out_ref - out_scan)))
    s = float(jnp.max(jnp.abs(out_ref)))
    print(f"forward: max |diff| = {d:.3e}  (max |value| = {s:.3e})")
    assert d <= 1e-10 * max(s, 1.0), "forward outputs disagree"

    def loss(p, comp):
        out = comp.build_forward()(p, zeros, constants_for(comp))["vis_ast"]
        return jnp.sum(jnp.abs(out) ** 2)

    g_ref = jax.grad(lambda p: loss(p, ref))(params)
    g_scan = jax.grad(lambda p: loss(p, scan))(params)
    for name in sorted(g_ref):
        gd = float(jnp.max(jnp.abs(g_ref[name] - g_scan[name])))
        gs = float(jnp.max(jnp.abs(g_ref[name])))
        print(f"grad {name}: max |diff| = {gd:.3e}  (max |grad| = {gs:.3e})")
        assert gd <= 1e-8 * max(gs, 1.0), f"gradients disagree for {name}"

    print("\nOK: FourierTimeFreqGPAstScan matches in value and gradient.")


if __name__ == "__main__":
    main()
