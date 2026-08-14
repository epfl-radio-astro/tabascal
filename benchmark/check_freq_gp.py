"""Check ComplexRFITimeFreq against ComplexRFI.

This is a model change, not an implementation one, so the two are *not* expected to
agree. What must hold:

1. the transform pair inverts exactly (truth/init go through inv_transform);
2. the marginal prior variance is unchanged, so the two are comparable at the same
   config rather than one being implicitly scaled;
3. frequency correlation is actually imposed -- the point of the exercise;
4. rfi_A comes out on the same grid as ComplexRFI.

    pixi run -e dev python3 benchmark/check_freq_gp.py
"""

import jax

jax.config.update("jax_enable_x64", True)

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from tabascal.components.rfi_signal import ComplexRFI, ComplexRFITimeFreq


def make_rfi_config(n_rfi=3, n_ant=4, n_freq=16, n_time=8, n_int_time=2,
                    corr_freq=None, corr_time=60.0, var=1.0):
    freqs = jnp.linspace(1.4e9, 1.41e9, n_freq)
    times = jnp.linspace(0.0, 120.0, n_time)
    n_bl = n_ant * (n_ant - 1) // 2
    return SimpleNamespace(
        n_rfi=n_rfi, n_rfi_real=n_rfi, n_ant=n_ant, n_freq=n_freq, n_time=n_time,
        n_freq_fine=n_freq, n_time_fine=n_time * n_int_time,
        n_int_freq=1, n_int_time=n_int_time,
        freqs=freqs, freqs_fine=freqs, chan_width=float(freqs[1] - freqs[0]),
        times=times, times_fine=jnp.linspace(times[0], times[-1], n_time * n_int_time),
        int_time=float(times[1] - times[0]),
        vis_obs=jnp.ones((n_bl, n_freq, n_time), dtype=complex),
        args={
            "rfi": {
                "r_seed": 1, "var": var, "corr_freq": corr_freq,
                "corr_time": corr_time, "init": "prior", "mean": "zeros", "est": None,
                "time_pad_factor": 2, "freq_pad_factor": 2,
            },
            "plots": {"truth": False},
            "data": {"zarr_path": None, "data_col": "DATA"},
        },
    )


def constants_for(comp):
    return {f"{comp.prefix}/{k}": v for k, v in comp.build_constants().items()}


def prior_draws(comp, n_draw=400, seed=0):
    """rfi_A from n_draw prior draws, shape (n_draw,) + rfi_A.shape."""
    forward = comp.build_forward()
    consts = constants_for(comp)
    shape = comp.init_params_base["rfi_r_induce_base"].shape
    out = []
    for i in range(n_draw):
        k1, k2 = jax.random.split(jax.random.PRNGKey(seed + i))
        params = {
            "rfi_r_induce_base": jax.random.normal(k1, shape),
            "rfi_i_induce_base": jax.random.normal(k2, shape),
        }
        out.append(forward(params, {}, consts)["rfi_A"])
    return jnp.stack(out)


def main() -> None:
    # corr_freq of a quarter of the band: strong but not degenerate correlation.
    band = 1.41e9 - 1.4e9
    cfg = dict(n_rfi=3, n_ant=4, n_freq=16, n_time=8, n_int_time=2,
               corr_freq=band / 4, corr_time=60.0, var=1.0)

    ref = ComplexRFI()
    ref.setup(make_rfi_config(**cfg))
    tf = ComplexRFITimeFreq()
    tf.setup(make_rfi_config(**cfg))

    print(f"\nComplexRFI          induce shape {ref._induce_shape}")
    print(f"ComplexRFITimeFreq  induce shape {tf._induce_shape}")
    n_ref = int(np.prod(ref._induce_shape))
    n_tf = int(np.prod(tf._induce_shape))
    print(f"parameters (per real/imag): {n_ref} -> {n_tf}  ({n_ref / n_tf:.1f}x fewer)")

    # 1. transform round-trip
    z = jax.random.normal(jax.random.PRNGKey(1), tf._induce_shape) + 1.0j * jax.random.normal(
        jax.random.PRNGKey(2), tf._induce_shape
    )
    rt = tf.inv_transform(tf.forward_transform(z, tf.L_rfi_A, tf.mu_rfi_A),
                          tf.L_rfi_A, tf.mu_rfi_A)
    err = float(jnp.max(jnp.abs(rt - z)))
    print(f"\n1. transform round-trip: max |diff| = {err:.3e}")
    assert err < 1e-8, "forward/inv transform are not inverses"

    # 2 + 3 + 4. prior draws
    d_ref = prior_draws(ref)
    d_tf = prior_draws(tf)
    print(f"\n4. rfi_A shape: ComplexRFI {d_ref.shape[1:]}  TimeFreq {d_tf.shape[1:]}")
    assert d_ref.shape == d_tf.shape, "rfi_A grids differ"

    v_ref = float(jnp.mean(jnp.abs(d_ref) ** 2))
    v_tf = float(jnp.mean(jnp.abs(d_tf) ** 2))
    print(f"\n2. marginal prior variance: ComplexRFI {v_ref:.4g}  TimeFreq {v_tf:.4g}"
          f"  (ratio {v_tf / v_ref:.3f})")
    assert 0.5 < v_tf / v_ref < 2.0, (
        "marginal variance changed materially -- the Kronecker factors are not "
        "sharing gp_var as intended"
    )

    # 3. correlation between neighbouring frequency channels, averaged over draws.
    def freq_corr(d, lag=1):
        # d: (n_draw, n_rfi, n_ant, n_freq, n_time_fine)
        x = d - jnp.mean(d, axis=0, keepdims=True)
        a, b = x[:, :, :, :-lag, :], x[:, :, :, lag:, :]
        num = jnp.mean(jnp.real(a * jnp.conj(b)))
        den = jnp.sqrt(jnp.mean(jnp.abs(a) ** 2) * jnp.mean(jnp.abs(b) ** 2))
        return float(num / den)

    c_ref, c_tf = freq_corr(d_ref), freq_corr(d_tf)
    print(f"\n3. adjacent-channel correlation: ComplexRFI {c_ref:+.3f}  "
          f"TimeFreq {c_tf:+.3f}")
    assert abs(c_ref) < 0.15, "ComplexRFI unexpectedly shows frequency correlation"
    assert c_tf > 0.8, "TimeFreq does not impose strong frequency correlation"

    print("\nOK: inverts exactly, variance preserved, frequency correlation imposed.")


if __name__ == "__main__":
    main()
