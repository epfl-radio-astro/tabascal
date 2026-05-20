"""Tests for tabascal.components.ast_signal.FixedImageSky (dense fixed sky)."""

import warnings
from types import SimpleNamespace

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

jax.config.update("jax_enable_x64", True)

from tabascal.components.ast_signal import FixedImageSky
from tabascal.components.ast_vis import ImageVisCalculation, PointSourceVisCalculation
from tabascal.imaging import make_image_plan
from .conftest import make_constants


# ── helpers ──────────────────────────────────────────────────────────────────

def make_config(fixed_path=None, n_ant=4, n_time=3, n_freq=2, fov_deg=8.0,
                n_pix=128, epsilon=1e-9, uvw_scale=20.0, ra0=0.0, dec0=0.0, seed=0):
    a1, a2 = jnp.triu_indices(n_ant, 1)
    n_bl = a1.shape[0]
    uvw = jax.random.normal(jax.random.PRNGKey(seed), (n_bl, n_time, 3)) * uvw_scale
    freqs = jnp.linspace(1.4e9, 1.5e9, n_freq)
    image_args = {"fov_deg": fov_deg, "n_pix": n_pix, "epsilon": epsilon}
    if fixed_path is not None:
        image_args["fixed_path"] = str(fixed_path)
    config = SimpleNamespace(
        n_ant=n_ant, n_bl=n_bl, n_time=n_time, n_freq=n_freq,
        uvw=uvw, freqs=freqs,
        phase_centre={"ra": ra0, "dec": dec0},
        args={"ast": {"image": image_args}},
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
    config.args["ast"]["image"]["fixed_path"] = str(zarr_path)

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
    config.args["ast"]["image"]["fixed_path"] = str(zarr_path)

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

    config = make_config(fixed_path=path, n_pix=n_pix, n_freq=n_freq)
    fis = FixedImageSky()
    fis.setup(config)
    img = np.asarray(fis.ast_image)
    assert img.shape == (n_freq, n_pix, n_pix)
    assert np.allclose(img[0], data) and np.allclose(img[1], data)


def test_fits_grid_mismatch_errors(tmp_path):
    """A FITS image whose spatial grid differs from the config grid errors."""
    from astropy.io import fits

    path = tmp_path / "wrong.fits"
    fits.PrimaryHDU(np.zeros((66, 66))).writeto(path)
    config = make_config(fixed_path=path, n_pix=64)
    with pytest.raises(RuntimeError, match="does not match"):
        FixedImageSky().setup(config)


# ── error handling ─────────────────────────────────────────────────────────────

def test_error_on_missing_fixed_path():
    config = make_config(fixed_path=None)
    with pytest.raises(RuntimeError, match="fixed_path"):
        FixedImageSky().setup(config)


def test_error_without_image_grid(tmp_path):
    config = make_config(fixed_path=tmp_path / "x.fits")
    config.image_grid = None
    with pytest.raises(RuntimeError, match="image grid"):
        FixedImageSky().setup(config)


def test_error_on_unsupported_extension():
    config = make_config(fixed_path="sky.txt")
    with pytest.raises(RuntimeError, match="Unsupported"):
        FixedImageSky().setup(config)
