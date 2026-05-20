import pytest
from types import SimpleNamespace

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from tabascal.components.ast_vis import PointSourceVisCalculation
from .conftest import make_constants

C = 299792458.0


def make_config(n_ant, n_src, n_time, n_freq, ra0_deg=0.0, dec0_deg=45.0,
                uvw_scale=1e3):
    a1, a2 = jnp.triu_indices(n_ant, 1)
    n_bl = a1.shape[0]
    key = jax.random.PRNGKey(0)
    uvw = jax.random.normal(key, (n_bl, n_time, 3)) * uvw_scale  # metres
    freqs = jnp.linspace(1.4e9, 1.5e9, n_freq)
    return SimpleNamespace(
        n_ant=n_ant,
        n_src=n_src,
        n_bl=n_bl,
        n_time=n_time,
        n_freq=n_freq,
        uvw=uvw,
        freqs=freqs,
        phase_centre={"ra": ra0_deg, "dec": dec0_deg},
    )


def make_state(config, n_src, seed=0):
    key = jax.random.PRNGKey(seed)
    return {
        "ast_radec": jax.random.uniform(key, (n_src, 2)) * 1e-3,  # near phase centre
        "ast_I": jax.random.normal(jax.random.PRNGKey(seed + 1), (n_src, config.n_freq)),
        "vis_ast": jnp.zeros((config.n_bl, config.n_freq, config.n_time), dtype=complex),
    }


# ── helpers ──────────────────────────────────────────────────────────────────

def run_forward(config, state):
    comp = PointSourceVisCalculation()
    comp.setup(config)
    return comp.build_forward()({}, state, make_constants(comp))


def manual_dft(radec, I, uvw, freqs, ra0, dec0):
    """Independent reference: V(u,v,w) = Σ_k (I_k/n_k) exp(-2πi (u l + v m + w(n-1))/λ)."""
    ra, dec = radec[:, 0], radec[:, 1]
    dra = ra - ra0
    l = jnp.cos(dec) * jnp.sin(dra)
    m = jnp.sin(dec) * jnp.cos(dec0) - jnp.cos(dec) * jnp.sin(dec0) * jnp.cos(dra)
    n = jnp.sqrt(1.0 - l**2 - m**2)

    dot = (
        uvw[:, :, 0:1] * l[None, None, :]
        + uvw[:, :, 1:2] * m[None, None, :]
        + uvw[:, :, 2:3] * (n - 1.0)[None, None, :]
    )  # (n_bl, n_time, n_src)

    lam = C / freqs
    phase = -2.0 * jnp.pi * dot[:, :, :, None] / lam[None, None, None, :]
    weights = (I / n[:, None])[None, None, :, :]
    return jnp.sum(weights * jnp.exp(1.0j * phase), axis=2).transpose(0, 2, 1)


# ── tests ─────────────────────────────────────────────────────────────────────

test_sizes = [(3, 1, 2, 1), (4, 5, 6, 3)]


@pytest.mark.parametrize("n_ant, n_src, n_time, n_freq", test_sizes)
def test_output_shape(n_ant, n_src, n_time, n_freq):
    config = make_config(n_ant, n_src, n_time, n_freq)
    state = make_state(config, n_src)
    out = run_forward(config, state)
    assert out["vis_ast"].shape == (config.n_bl, config.n_freq, config.n_time)


@pytest.mark.parametrize("n_ant, n_freq", [(3, 1), (5, 4)])
def test_source_at_phase_centre(n_ant, n_freq):
    """A point source exactly at the phase centre should give V = I for every baseline/time."""
    n_time = 3
    ra0, dec0 = 12.0, -30.0
    config = make_config(n_ant, 1, n_time, n_freq, ra0_deg=ra0, dec0_deg=dec0)

    I = jnp.ones((1, n_freq)) * 2.5
    state = {
        "ast_radec": jnp.array([[jnp.deg2rad(ra0), jnp.deg2rad(dec0)]]),
        "ast_I": I,
        "vis_ast": jnp.zeros((config.n_bl, n_freq, n_time), dtype=complex),
    }

    out = run_forward(config, state)
    expected = jnp.broadcast_to(I[0, :, None], (n_freq, n_time))  # (n_freq, n_time)
    for bl in range(config.n_bl):
        assert jnp.allclose(out["vis_ast"][bl], expected, atol=1e-6), (
            f"baseline {bl}: max error {jnp.abs(out['vis_ast'][bl] - expected).max()}"
        )


@pytest.mark.parametrize("n_ant, n_src, n_time, n_freq", test_sizes)
def test_accumulates_into_vis_ast(n_ant, n_src, n_time, n_freq):
    """The component adds to an existing vis_ast rather than overwriting it."""
    config = make_config(n_ant, n_src, n_time, n_freq)
    state = make_state(config, n_src)

    out_zero = run_forward(config, state)
    increment = out_zero["vis_ast"]

    state_nonzero = {**state, "vis_ast": increment}
    out_twice = run_forward(config, state_nonzero)

    assert jnp.allclose(out_twice["vis_ast"], 2.0 * increment, atol=1e-10)


@pytest.mark.parametrize("n_ant, n_src, n_time, n_freq", test_sizes)
def test_jvp_ast_I(n_ant, n_src, n_time, n_freq):
    """Forward-mode JVP through ast_I should not raise and should be non-trivially non-zero."""
    config = make_config(n_ant, n_src, n_time, n_freq)
    state = make_state(config, n_src)

    comp = PointSourceVisCalculation()
    comp.setup(config)
    constants = make_constants(comp)
    fwd = comp.build_forward()

    tangent = {k: jnp.zeros_like(v) for k, v in state.items()}
    tangent["ast_I"] = jnp.ones_like(state["ast_I"])

    _, jvp_out = jax.jvp(lambda s: fwd({}, s, constants), (state,), (tangent,))
    assert jvp_out["vis_ast"].shape == (config.n_bl, config.n_freq, config.n_time)
    assert jnp.any(jnp.abs(jvp_out["vis_ast"]) > 0)


@pytest.mark.parametrize("n_ant, n_src, n_time, n_freq", test_sizes)
def test_vjp_ast_I(n_ant, n_src, n_time, n_freq):
    """Reverse-mode VJP should propagate gradients back to ast_I."""
    config = make_config(n_ant, n_src, n_time, n_freq)
    state = make_state(config, n_src)

    comp = PointSourceVisCalculation()
    comp.setup(config)
    constants = make_constants(comp)
    fwd = comp.build_forward()

    _, vjp_fn = jax.vjp(lambda s: fwd({}, s, constants), state)
    cotangent = {
        "vis_ast": jnp.ones((config.n_bl, config.n_freq, config.n_time), dtype=complex),
        "ast_radec": jnp.zeros_like(state["ast_radec"]),
        "ast_I": jnp.zeros_like(state["ast_I"]),
    }
    (grad_state,) = vjp_fn(cotangent)
    assert grad_state["ast_I"].shape == state["ast_I"].shape
    assert jnp.any(jnp.abs(grad_state["ast_I"]) > 0)


def test_against_manual_dft():
    """3 off-centre sources, 3-antenna array: component matches a direct DFT sum."""
    ra0 = jnp.deg2rad(30.0)
    dec0 = jnp.deg2rad(45.0)

    radec = jnp.array([
        [ra0 + 0.01,  dec0 + 0.005],
        [ra0 - 0.02,  dec0 + 0.015],
        [ra0 + 0.005, dec0 - 0.010],
    ])

    n_freq = 2
    n_time = 2

    I = jnp.array([
        [1.5, 2.0],
        [0.7, 1.1],
        [3.2, 0.4],
    ])  # (n_src, n_freq)

    uvw = jnp.array([
        [[ 100.0,  200.0,  10.0], [ 120.0,  210.0,  12.0]],
        [[ 300.0, -150.0,  -5.0], [ 310.0, -140.0,  -4.0]],
        [[ 200.0, -350.0, -15.0], [ 190.0, -350.0, -16.0]],
    ])  # (n_bl=3, n_time=2, 3)

    freqs = jnp.array([1.4e9, 1.5e9])

    config = SimpleNamespace(
        n_ant=3, n_src=3, n_bl=3, n_time=n_time, n_freq=n_freq,
        uvw=uvw, freqs=freqs,
        phase_centre={"ra": jnp.rad2deg(ra0), "dec": jnp.rad2deg(dec0)},
    )
    state = {
        "ast_radec": radec,
        "ast_I": I,
        "vis_ast": jnp.zeros((3, n_freq, n_time), dtype=complex),
    }

    dft_vis = run_forward(config, state)["vis_ast"]
    expected = manual_dft(radec, I, uvw, freqs, ra0, dec0)

    assert jnp.allclose(dft_vis, expected, atol=1e-6)


def test_wide_field_long_baselines():
    """Sources 30°+ off-axis with realistic (km-scale) baselines compute finitely
    and match the DFT — the case the type-3 NUFFT could not handle."""
    n_freq, n_time = 2, 3
    ra0_deg, dec0_deg = 30.0, 0.0
    ra0, dec0 = jnp.deg2rad(ra0_deg), jnp.deg2rad(dec0_deg)

    # Sources spanning ±40° in RA and ±35° in Dec around the phase centre
    key = jax.random.PRNGKey(11)
    k_ra, k_dec, k_I = jax.random.split(key, 3)
    ra = ra0 + jnp.deg2rad(jax.random.uniform(k_ra, (8,), minval=-40.0, maxval=40.0))
    dec = dec0 + jnp.deg2rad(jax.random.uniform(k_dec, (8,), minval=-35.0, maxval=35.0))
    radec = jnp.stack([ra, dec], axis=-1)
    I = jnp.abs(jax.random.normal(k_I, (8, n_freq))) + 0.1

    # Realistic baselines that would OOM a 3D type-3 NUFFT (1 km std)
    config = make_config(n_ant=6, n_src=8, n_time=n_time, n_freq=n_freq,
                         ra0_deg=ra0_deg, dec0_deg=dec0_deg, uvw_scale=1e3)
    state = {
        "ast_radec": radec,
        "ast_I": I,
        "vis_ast": jnp.zeros((config.n_bl, n_freq, n_time), dtype=complex),
    }

    vis = run_forward(config, state)["vis_ast"]
    assert jnp.all(jnp.isfinite(vis))

    expected = manual_dft(radec, I, config.uvw, config.freqs, ra0, dec0)
    assert jnp.allclose(vis, expected, atol=1e-6)
