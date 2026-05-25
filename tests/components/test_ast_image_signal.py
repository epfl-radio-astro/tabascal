"""Tests for tabascal.components.ast_signal.FixedImageSky and ImageSky on the
ast.signals / ast.grid config schema (sources resolved via tabascal.sky_sources)."""

import warnings
from types import SimpleNamespace

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

jax.config.update("jax_enable_x64", True)

from tabascal.components.ast_signal import FixedImageSky, ImageSky
from tabascal.components.ast_vis import ImageVisCalculation, PointSourceVisCalculation
from tabascal.imaging import make_image_plan
from .conftest import make_constants


# ── helpers ──────────────────────────────────────────────────────────────────

def make_config(init=None, n_ant=4, n_time=3, n_freq=2, fov_deg=8.0,
                n_pix=128, epsilon=1e-9, uvw_scale=20.0, ra0=0.0, dec0=0.0,
                zarr_path=None, seed=0):
    """Config for FixedImageSky. ``init`` is an ast.signals.FixedImageSky source
    spec (or None to leave the block out)."""
    a1, a2 = jnp.triu_indices(n_ant, 1)
    n_bl = a1.shape[0]
    uvw = jax.random.normal(jax.random.PRNGKey(seed), (n_bl, n_time, 3)) * uvw_scale
    freqs = jnp.linspace(1.4e9, 1.5e9, n_freq)
    signals = {} if init is None else {"FixedImageSky": {"init": init}}
    config = SimpleNamespace(
        n_ant=n_ant, n_bl=n_bl, n_time=n_time, n_freq=n_freq,
        uvw=uvw, freqs=freqs,
        phase_centre={"ra": ra0, "dec": dec0},
        args={"ast": {"grid": {"fov_deg": fov_deg, "n_pix": n_pix, "epsilon": epsilon},
                      "signals": signals},
              "data": {"zarr_path": zarr_path, "data_col": "DATA"}},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config.image_grid = make_image_plan(uvw, freqs, fov_deg, n_pix, epsilon)
    return config


def pixels_to_radec(pixels, n_pix, pixsize):
    """On-grid pixel centres -> (ra, dec) for phase centre (0, 0)."""
    ls = np.array([(a - n_pix / 2) * pixsize for a, b in pixels])
    ms = np.array([(b - n_pix / 2) * pixsize for a, b in pixels])
    dec = np.arcsin(ms)
    ra = np.arcsin(ls / np.sqrt(1.0 - ms**2))
    return np.stack([ra, dec], axis=-1), ls, ms


def write_catalogue_zarr(path, radec_deg, flux_per_freq, n_time, freqs):
    """tabsim-style point catalogue: ast_p_radec (deg), ast_p_I (src, time, freq)."""
    n_src, n_freq = flux_per_freq.shape
    flux_cube = np.repeat(flux_per_freq[:, None, :], n_time, axis=1)
    ds = xr.Dataset(
        {
            "ast_p_radec": (["ast_p_src", "radec"], np.asarray(radec_deg)),
            "ast_p_I": (["ast_p_src", "time", "freq"], flux_cube),
        },
        coords={"freq": np.asarray(freqs)},
    )
    ds.to_zarr(str(path), mode="w")


def _catalogue_init(zarr_path):
    return {"type": "from_catalogue", "fmt": "zarr", "path": str(zarr_path)}


def _fits_init(path):
    return {"type": "from_fits", "path": str(path)}


# ── catalogue path: parity with PointSourceVisCalculation ─────────────────────

def test_catalogue_pipeline_matches_point_source(tmp_path):
    """FixedImageSky (rasterised catalogue) -> ImageVisCalculation matches the
    point-source DFT for sources placed at pixel centres."""
    n_pix, n_freq = 128, 2
    config = make_config(n_pix=n_pix, n_freq=n_freq)
    pixsize = config.image_grid.pixsize

    pixels = [(94, 44), (34, 79), (104, 84)]
    flux = np.array([[1.0, 0.8], [0.5, 1.2], [1.3, 0.4]])
    radec, _, _ = pixels_to_radec(pixels, n_pix, pixsize)

    zarr_path = tmp_path / "cat.zarr"
    write_catalogue_zarr(zarr_path, np.rad2deg(radec), flux, config.n_time, config.freqs)
    config.args["ast"]["signals"]["FixedImageSky"] = {"init": _catalogue_init(zarr_path)}

    fis = FixedImageSky()
    iv = ImageVisCalculation()
    fis.setup(config)
    iv.setup(config)
    state = {**fis.state_outputs, **iv.state_outputs}
    constants = {**make_constants(fis), **make_constants(iv)}
    state = fis.build_forward()({}, state, constants)
    state = iv.build_forward()({}, state, constants)
    dense = state["vis_ast"]

    pv = PointSourceVisCalculation()
    pv.setup(config)
    sstate = {"ast_radec": jnp.asarray(radec), "ast_I": jnp.asarray(flux),
              "vis_ast": jnp.zeros((config.n_bl, n_freq, config.n_time), complex)}
    sparse = pv.build_forward()({}, sstate, make_constants(pv))["vis_ast"]

    rel_err = jnp.linalg.norm(dense - sparse) / jnp.linalg.norm(sparse)
    assert rel_err < 1e-6, f"dense/sparse rel_err {rel_err:.2e}"


def test_catalogue_continuum_broadcasts(tmp_path):
    """A single-channel catalogue is broadcast across all model frequencies."""
    config = make_config(n_pix=64, n_freq=3)
    pixsize = config.image_grid.pixsize
    radec, _, _ = pixels_to_radec([(40, 28)], 64, pixsize)
    flux = np.array([[2.0]])                       # one channel
    zarr_path = tmp_path / "cont.zarr"
    write_catalogue_zarr(zarr_path, np.rad2deg(radec), flux, config.n_time,
                         np.array([1.227e9]))
    config.args["ast"]["signals"]["FixedImageSky"] = {"init": _catalogue_init(zarr_path)}

    fis = FixedImageSky()
    fis.setup(config)
    img = np.asarray(fis.ast_image)
    assert img.shape == (3, 64, 64)
    assert np.allclose(img[:, 40, 28], 2.0)        # same flux every channel


# ── FITS path ─────────────────────────────────────────────────────────────────

def test_fits_continuum_image(tmp_path):
    """A 2-D FITS image loads and broadcasts across frequency."""
    from astropy.io import fits

    n_pix, n_freq = 64, 2
    data = np.zeros((n_pix, n_pix))
    data[20, 30] = 1.5
    data[40, 10] = 0.7
    path = tmp_path / "sky.fits"
    fits.PrimaryHDU(data).writeto(path)

    config = make_config(init=_fits_init(path), n_pix=n_pix, n_freq=n_freq)
    fis = FixedImageSky()
    fis.setup(config)
    img = np.asarray(fis.ast_image)
    assert img.shape == (n_freq, n_pix, n_pix)
    assert np.allclose(img[0], data) and np.allclose(img[1], data)


def test_fits_jy_per_beam_conversion(tmp_path):
    """A Jy/beam FITS image is converted to Jy/pixel using BMAJ/BMIN/CDELT."""
    from astropy.io import fits

    n_pix = 64
    fov_deg = 8.0
    cdelt = fov_deg / n_pix                          # 0.125 deg/pixel
    bmaj = bmin = 0.25                               # restoring beam FWHM, deg
    data = np.zeros((n_pix, n_pix))
    data[20, 30] = 1.0                               # 1 Jy/beam
    hdr = fits.Header()
    hdr["BUNIT"] = "Jy/beam"
    hdr["BMAJ"], hdr["BMIN"] = bmaj, bmin
    hdr["CDELT1"], hdr["CDELT2"] = -cdelt, cdelt
    path = tmp_path / "beam.fits"
    fits.PrimaryHDU(data, hdr).writeto(path)

    config = make_config(init=_fits_init(path), n_pix=n_pix, n_freq=2, fov_deg=fov_deg)
    fis = FixedImageSky()
    fis.setup(config)

    beam_area = (np.pi / (4.0 * np.log(2.0))) * bmaj * bmin
    factor = cdelt**2 / beam_area
    assert np.isclose(np.asarray(fis.ast_image)[0, 20, 30], factor)


def test_fits_jy_per_beam_missing_beam_errors(tmp_path):
    """Jy/beam without a beam in the header cannot be converted -> error."""
    from astropy.io import fits

    hdr = fits.Header()
    hdr["BUNIT"] = "Jy/beam"
    path = tmp_path / "nobeam.fits"
    fits.PrimaryHDU(np.zeros((64, 64)), hdr).writeto(path)
    config = make_config(init=_fits_init(path), n_pix=64)
    with pytest.raises(RuntimeError, match="Jy/beam"):
        FixedImageSky().setup(config)


def test_fits_grid_mismatch_errors(tmp_path):
    """A FITS image whose spatial grid differs from the config grid errors."""
    from astropy.io import fits

    path = tmp_path / "wrong.fits"
    fits.PrimaryHDU(np.zeros((66, 66))).writeto(path)
    config = make_config(init=_fits_init(path), n_pix=64)
    with pytest.raises(RuntimeError, match="does not match"):
        FixedImageSky().setup(config)


# ── error handling ─────────────────────────────────────────────────────────────

def test_error_on_missing_source(tmp_path):
    """No FixedImageSky source and no data.zarr_path fallback -> clear error."""
    config = make_config(init=None, zarr_path=None)
    with pytest.raises(RuntimeError, match="path"):
        FixedImageSky().setup(config)


def test_error_without_image_grid(tmp_path):
    config = make_config(init=_fits_init(tmp_path / "x.fits"))
    config.image_grid = None
    with pytest.raises(RuntimeError, match="image grid"):
        FixedImageSky().setup(config)


def test_error_on_unknown_source_type():
    config = make_config(init={"type": "bogus"})
    with pytest.raises(RuntimeError, match="Unknown sky-source"):
        FixedImageSky().setup(config)


# ── ImageSky (learnable Gaussian random-field sky) ────────────────────────────

_INIT_SPECS = {
    "zeros": {"type": "zeros"},
    "prior": {"type": "prior"},
    "sample": {"type": "sample"},
    "data": {"type": "from_ms"},          # dirty image of the data column
    "bogus": {"type": "bogus"},
}


def make_sky_config(n_ant=4, n_time=3, n_freq=2, fov_deg=8.0, n_pix=64,
                    epsilon=1e-7, uvw_scale=15.0, mu=-2.0, init="sample",
                    chan_width=1e6, with_vis_obs=False, seed=0):
    a1, a2 = jnp.triu_indices(n_ant, 1)
    n_bl = a1.shape[0]
    uvw = jax.random.normal(jax.random.PRNGKey(seed), (n_bl, n_time, 3)) * uvw_scale
    freqs = jnp.linspace(1.4e9, 1.5e9, n_freq)
    pow_spec = {"p0": 1.0, "k0_freq": 1e-6, "k0_lm": 300.0,
                "gamma_freq": 2.0, "gamma_lm": 2.0, "cutoff": 0.0, "mu": mu}
    init_spec = _INIT_SPECS.get(init, {"type": init})
    config = SimpleNamespace(
        n_ant=n_ant, n_bl=n_bl, n_time=n_time, n_freq=n_freq,
        uvw=uvw, freqs=freqs, chan_width=chan_width,
        phase_centre={"ra": 0.0, "dec": 0.0},
        args={"ast": {"grid": {"fov_deg": fov_deg, "n_pix": n_pix, "epsilon": epsilon},
                      "signals": {"ImageSky": {
                          "init": init_spec,
                          "prior": {"mean": {"type": "zeros"}, "pow_spec": pow_spec}}}},
              "data": {"zarr_path": None, "data_col": "DATA"}},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config.image_grid = make_image_plan(uvw, freqs, fov_deg, n_pix, epsilon)
    if with_vis_obs:
        k = jax.random.PRNGKey(seed + 9)
        vis = (jax.random.normal(k, (n_bl, n_freq, n_time))
               + 1j * jax.random.normal(jax.random.PRNGKey(seed + 10),
                                        (n_bl, n_freq, n_time)))
        # Mirror real MS data: single precision (complex64). The data init must
        # cope with this against the float64 (x64) wgridder plan.
        config.vis_obs = vis.astype(jnp.complex64)
    return config


def run_sky(config):
    sky = ImageSky()
    sky.setup(config)
    out = sky.build_forward()(sky.init_params_base, {}, make_constants(sky))
    return sky, out


def test_imagesky_shape_and_finite():
    config = make_sky_config(n_pix=64, n_freq=2)
    sky, out = run_sky(config)
    img = out["ast_image"]
    assert img.shape == (config.n_freq, 64, 64)
    assert jnp.all(jnp.isfinite(img))


def test_imagesky_zeros_init_is_flat():
    """zeros init + zeros prior mean -> base params zero -> I = 0 everywhere
    (the field is linear, so the prior-mean flux is zero)."""
    config = make_sky_config(init="zeros")
    sky, out = run_sky(config)
    assert jnp.allclose(out["ast_image"], 0.0, atol=1e-8)


@pytest.mark.parametrize("init", ["zeros", "prior", "sample", "data"])
def test_imagesky_init_paths(init):
    config = make_sky_config(init=init, with_vis_obs=(init == "data"))
    sky, out = run_sky(config)
    assert jnp.all(jnp.isfinite(out["ast_image"]))
    # base params live on the latent Fourier grid
    assert sky.init_params_base["image_k_r_base"].shape == sky.pk.shape


def test_imagesky_param_and_prior_shapes():
    config = make_sky_config()
    sky = ImageSky()
    sky.setup(config)
    k_shape = sky.pk.shape
    assert sky.sigma_image_k.shape == k_shape
    assert sky.mu_image_k.shape == k_shape
    assert sky.init_params_base["image_k_i_base"].shape == k_shape


def test_imagesky_prior_mean_from_source(tmp_path):
    """A source prior mean (a FITS image) sets the GRF latent mean: with zeros
    init the sky reproduces that image."""
    from astropy.io import fits

    n_pix, n_freq = 64, 2
    config = make_sky_config(n_pix=n_pix, n_freq=n_freq, init="zeros")
    img = np.full((n_pix, n_pix), 0.05)
    img[30, 18] = 1.2
    path = tmp_path / "mean.fits"
    fits.PrimaryHDU(img).writeto(path)
    sig = config.args["ast"]["signals"]["ImageSky"]
    sig["prior"]["mean"] = {"type": "from_fits", "path": str(path)}

    sky, out = run_sky(config)
    # Round-trips through latent -> signal (band-limited), so allow modest error.
    rel = jnp.linalg.norm(out["ast_image"][0] - img) / jnp.linalg.norm(img)
    assert rel < 1e-2, f"prior-mean reconstruction rel_err {rel:.2e}"


def test_imagesky_pipeline_to_vis():
    config = make_sky_config(n_pix=64, n_freq=2)
    sky = ImageSky()
    iv = ImageVisCalculation()
    sky.setup(config)
    iv.setup(config)
    state = {**sky.state_outputs, **iv.state_outputs}
    constants = {**make_constants(sky), **make_constants(iv)}
    params = sky.init_params_base
    state = sky.build_forward()(params, state, constants)
    state = iv.build_forward()(params, state, constants)
    vis = state["vis_ast"]
    assert vis.shape == (config.n_bl, config.n_freq, config.n_time)
    assert jnp.all(jnp.isfinite(vis)) and jnp.any(jnp.abs(vis) > 0)


def test_imagesky_differentiable():
    """JVP and VJP through the base params are finite and non-trivially non-zero."""
    config = make_sky_config()
    sky = ImageSky()
    sky.setup(config)
    constants = make_constants(sky)
    fwd = sky.build_forward()
    params = sky.init_params_base

    def f(p):
        return fwd(p, {}, constants)["ast_image"]

    tangent = {k: jnp.ones_like(v) for k, v in params.items()}
    _, jvp_out = jax.jvp(f, (params,), (tangent,))
    assert jnp.all(jnp.isfinite(jvp_out)) and jnp.any(jnp.abs(jvp_out) > 0)

    _, vjp_fn = jax.vjp(f, params)
    (grad,) = vjp_fn(jnp.ones((config.n_freq, sky.n_pix, sky.n_pix)))
    assert grad["image_k_r_base"].shape == sky.pk.shape
    assert jnp.all(jnp.isfinite(grad["image_k_r_base"]))
    assert jnp.any(jnp.abs(grad["image_k_r_base"]) > 0)


def test_imagesky_invalid_init_errors():
    config = make_sky_config(init="bogus")
    with pytest.raises(RuntimeError, match="Unknown sky-source"):
        ImageSky().setup(config)


def test_imagesky_data_requires_vis_obs():
    config = make_sky_config(init="data", with_vis_obs=False)
    with pytest.raises(RuntimeError, match="vis_obs"):
        ImageSky().setup(config)


def test_imagesky_data_init_single_channel_complex64():
    """Regression: data init (dirty image) with single-channel complex64 vis_obs
    against the float64 wgridder plan must not hit a precision mismatch."""
    config = make_sky_config(n_freq=1, init="data", with_vis_obs=True)
    sky, out = run_sky(config)
    assert out["ast_image"].shape == (1, sky.n_pix, sky.n_pix)
    assert jnp.all(jnp.isfinite(out["ast_image"]))
