"""Tests for tabascal.components.ast_vis.ImageVisCalculation (dense-sky wgridder)
and its parity with PointSourceVisCalculation."""

import warnings

import pytest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from tabascal.components.ast_vis import ImageVisCalculation, PointSourceVisCalculation
from .conftest import make_constants


# ── helpers ──────────────────────────────────────────────────────────────────

def make_config(n_ant=4, n_time=3, n_freq=2, fov_deg=8.0, n_pix=128,
                epsilon=1e-9, uvw_scale=20.0, ra0=0.0, dec0=0.0, seed=0):
    # Defaults are well-sampled (~3.6 pixels/beam) so functional tests do not
    # trip the grid-sampling guard; the guard tests below set their own configs.
    a1, a2 = jnp.triu_indices(n_ant, 1)
    n_bl = a1.shape[0]
    uvw = jax.random.normal(jax.random.PRNGKey(seed), (n_bl, n_time, 3)) * uvw_scale
    freqs = jnp.linspace(1.4e9, 1.5e9, n_freq)
    return SimpleNamespace(
        n_ant=n_ant, n_bl=n_bl, n_time=n_time, n_freq=n_freq,
        uvw=uvw, freqs=freqs,
        phase_centre={"ra": ra0, "dec": dec0},
        args={"ast": {"image": {"fov_deg": fov_deg, "n_pix": n_pix,
                                "epsilon": epsilon}}},
    )


def on_grid_sources(config, pixels, fluxes):
    """Build an ast_image with bright pixels, and the matching point-source
    (ra, dec, flux) at the exact pixel centres. Assumes phase centre (0, 0)."""
    n_pix = config.args["ast"]["image"]["n_pix"]
    pixsize = np.deg2rad(config.args["ast"]["image"]["fov_deg"]) / n_pix
    image = np.zeros((config.n_freq, n_pix, n_pix))
    ls, ms = [], []
    for (a, b), f in zip(pixels, fluxes):
        image[:, a, b] = f
        ls.append((a - n_pix / 2) * pixsize)
        ms.append((b - n_pix / 2) * pixsize)
    ls, ms = np.array(ls), np.array(ms)
    # Invert direction cosines to (ra, dec) for phase centre (0, 0):
    #   m = sin(dec);  l = cos(dec) sin(ra)
    dec = np.arcsin(ms)
    ra = np.arcsin(ls / np.sqrt(1.0 - ms**2))
    radec = jnp.array(np.stack([ra, dec], axis=-1))
    return jnp.asarray(image), radec, jnp.asarray(fluxes)


def run_image(config, image):
    comp = ImageVisCalculation()
    comp.setup(config)
    state = {"ast_image": image,
             "vis_ast": jnp.zeros((config.n_bl, config.n_freq, config.n_time), complex)}
    return comp.build_forward()({}, state, make_constants(comp))


# ── functional tests (well-sampled configs, no sampling warnings) ─────────────

@pytest.mark.parametrize("n_ant, n_time, n_freq", [(3, 1, 1), (4, 3, 2)])
def test_output_shape(n_ant, n_time, n_freq):
    config = make_config(n_ant=n_ant, n_time=n_time, n_freq=n_freq)
    n_pix = config.args["ast"]["image"]["n_pix"]
    image = jnp.zeros((n_freq, n_pix, n_pix))
    out = run_image(config, image)
    assert out["vis_ast"].shape == (config.n_bl, n_freq, n_time)


def test_central_pixel_gives_flux():
    """A single pixel at the phase centre (l=m=0, n=1) gives V = I everywhere."""
    n_freq = 2
    config = make_config(n_freq=n_freq)
    n_pix = config.args["ast"]["image"]["n_pix"]
    image = np.zeros((n_freq, n_pix, n_pix))
    flux = np.array([2.5, 1.7])
    image[:, n_pix // 2, n_pix // 2] = flux
    out = run_image(config, jnp.asarray(image))["vis_ast"]   # (n_bl, n_freq, n_time)
    expected = jnp.broadcast_to(jnp.asarray(flux)[None, :, None], out.shape)
    assert jnp.allclose(out, expected, atol=1e-6)


def test_matches_point_source():
    """Dense wgridder visibilities match the point-source DFT for on-grid sources,
    to within the requested wgridder accuracy (~10·epsilon)."""
    config = make_config()
    pixels = [(94, 44), (34, 79), (104, 84)]      # off-centre, on-grid (n_pix=128)
    fluxes = np.array([[1.0, 0.8], [0.5, 1.2], [1.3, 0.4]])
    image, radec, I = on_grid_sources(config, pixels, fluxes)

    dense = run_image(config, image)["vis_ast"]

    pv = PointSourceVisCalculation()
    pv.setup(config)
    sstate = {"ast_radec": radec, "ast_I": I,
              "vis_ast": jnp.zeros((config.n_bl, config.n_freq, config.n_time), complex)}
    sparse = pv.build_forward()({}, sstate, make_constants(pv))["vis_ast"]

    rel_err = jnp.linalg.norm(dense - sparse) / jnp.linalg.norm(sparse)
    assert rel_err < 1e-6, f"dense/sparse rel_err {rel_err:.2e}"


def test_accumulates_into_vis_ast():
    """The component adds to an existing vis_ast rather than overwriting it."""
    config = make_config()
    image, _, _ = on_grid_sources(config, [(94, 44), (34, 79)],
                                  np.array([[1.0, 0.8], [0.6, 1.1]]))
    increment = run_image(config, image)["vis_ast"]

    comp = ImageVisCalculation()
    comp.setup(config)
    state = {"ast_image": image, "vis_ast": increment}
    out = comp.build_forward()({}, state, make_constants(comp))["vis_ast"]
    assert jnp.allclose(out, 2.0 * increment, atol=1e-10)


def test_jvp_ast_image():
    """Forward-mode JVP through ast_image is finite and non-trivially non-zero."""
    config = make_config()
    image, _, _ = on_grid_sources(config, [(94, 44)], np.array([[1.0, 0.8]]))
    comp = ImageVisCalculation()
    comp.setup(config)
    constants = make_constants(comp)
    fwd = comp.build_forward()

    state = {"ast_image": image,
             "vis_ast": jnp.zeros((config.n_bl, config.n_freq, config.n_time), complex)}
    tangent = {k: jnp.zeros_like(v) for k, v in state.items()}
    tangent["ast_image"] = jnp.ones_like(image)

    _, jvp_out = jax.jvp(lambda s: fwd({}, s, constants), (state,), (tangent,))
    assert jnp.all(jnp.isfinite(jvp_out["vis_ast"]))
    assert jnp.any(jnp.abs(jvp_out["vis_ast"]) > 0)


def test_vjp_ast_image():
    """Reverse-mode VJP propagates finite, non-zero gradients back to ast_image."""
    config = make_config()
    image, _, _ = on_grid_sources(config, [(94, 44)], np.array([[1.0, 0.8]]))
    comp = ImageVisCalculation()
    comp.setup(config)
    constants = make_constants(comp)
    fwd = comp.build_forward()

    state = {"ast_image": image,
             "vis_ast": jnp.zeros((config.n_bl, config.n_freq, config.n_time), complex)}
    _, vjp_fn = jax.vjp(lambda s: fwd({}, s, constants), state)
    cotangent = {
        "ast_image": jnp.zeros_like(image),
        "vis_ast": jnp.ones((config.n_bl, config.n_freq, config.n_time), complex),
    }
    (grad_state,) = vjp_fn(cotangent)
    assert grad_state["ast_image"].shape == image.shape
    assert jnp.all(jnp.isfinite(grad_state["ast_image"]))
    assert jnp.any(jnp.abs(grad_state["ast_image"]) > 0)


# ── grid-sampling guard tests (warn, never raise) ─────────────────────────────

def test_no_warning_when_well_sampled():
    """The default config is well-sampled; setup emits no sampling warnings."""
    config = make_config()
    with warnings.catch_warnings():
        warnings.simplefilter("error")        # any warning would raise here
        ImageVisCalculation().setup(config)


def test_warns_under_resolution():
    """A coarse grid against long baselines warns about aliasing (but does not raise)."""
    config = make_config(n_pix=64, uvw_scale=200.0)
    with pytest.warns(UserWarning, match="under-samples"):
        ImageVisCalculation().setup(config)


def test_warns_over_resolution():
    """A very fine grid against short baselines warns about a wasteful grid."""
    config = make_config(n_pix=1024, uvw_scale=2.0)
    with pytest.warns(UserWarning, match="over-samples"):
        ImageVisCalculation().setup(config)


def test_warns_horizon_clip():
    """A field wide enough that grid corners pass the horizon warns about zeroed pixels."""
    config = make_config(fov_deg=120.0)
    with pytest.warns(UserWarning, match="horizon"):
        ImageVisCalculation().setup(config)
