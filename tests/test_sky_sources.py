"""Unit tests for the sky-source loader (tabascal.sky_sources)."""

import warnings
from types import SimpleNamespace

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

jax.config.update("jax_enable_x64", True)

from tabascal.imaging import make_image_plan
from tabascal.sky_sources import (
    resolve_sky_source,
    warn_if_large_catalogue,
    ZerosSource,
    CatalogueSource,
    FitsSource,
    MSSource,
    DFT_SOURCE_WARN_THRESHOLD,
)


def make_config(n_freq=2, ra0=150.0, dec0=-20.0, zarr_path=None, vis_obs=None):
    return SimpleNamespace(
        n_freq=n_freq,
        freqs=np.linspace(1.4e9, 1.5e9, n_freq),
        phase_centre={"ra": ra0, "dec": dec0},
        ms_path=None,
        vis_obs=vis_obs,
        args={"data": {"zarr_path": zarr_path, "data_col": "DATA",
                       "freq": None, "corr": "xx"}},
    )


def make_grid(n_bl=6, n_time=3, n_freq=2, n_pix=64, fov_deg=8.0, seed=0):
    uvw = jax.random.normal(jax.random.PRNGKey(seed), (n_bl, n_time, 3)) * 20.0
    freqs = jnp.linspace(1.4e9, 1.5e9, n_freq)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return make_image_plan(uvw, freqs, fov_deg, n_pix, 1e-7), uvw, freqs


def write_zarr(path, radec_deg, flux_per_freq, n_time, freqs):
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


# ── resolve_sky_source dispatch + validation ──────────────────────────────────

def test_unknown_type_errors():
    with pytest.raises(ValueError, match="Unknown sky-source type"):
        resolve_sky_source({"type": "nope"}, make_config())


def test_spec_must_be_typed_dict():
    with pytest.raises(ValueError, match="must be a dict with a 'type'"):
        resolve_sky_source({"path": "x"}, make_config())


def test_non_stokes_i_rejected():
    with pytest.raises(NotImplementedError, match="Stokes I"):
        resolve_sky_source({"type": "zeros", "stokes": ["I", "Q"]}, make_config())


def test_from_catalogue_requires_path():
    with pytest.raises(ValueError, match="requires a 'path'"):
        resolve_sky_source({"type": "from_catalogue", "fmt": "zarr"}, make_config())


def test_from_catalogue_falls_back_to_data_zarr(tmp_path):
    zp = tmp_path / "cat.zarr"
    write_zarr(zp, [[150.0, -20.0]], np.array([[1.0, 1.0]]), 2, np.linspace(1.4e9, 1.5e9, 2))
    cfg = make_config(zarr_path=str(zp))
    src = resolve_sky_source({"type": "from_catalogue", "fmt": "zarr"}, cfg)
    radec, flux = src.catalogue()
    assert radec.shape == (1, 2) and flux.shape == (1, 2)


# ── ZerosSource ───────────────────────────────────────────────────────────────

def test_zeros_source_image_and_catalogue():
    cfg = make_config(n_freq=3)
    grid, *_ = make_grid(n_freq=3, n_pix=32)
    src = ZerosSource(cfg)
    img = src.image(grid)
    assert img.shape == (3, 32, 32) and jnp.all(img == 0)
    radec, flux = src.catalogue()
    assert radec.shape == (0, 2) and flux.shape == (0, 3)

    with pytest.raises(NotImplementedError, match="visibilities"):
        src.visibilities()


# ── CatalogueSource (zarr) ────────────────────────────────────────────────────

def test_catalogue_zarr_catalogue_and_rasterise(tmp_path):
    n_freq = 2
    cfg = make_config(n_freq=n_freq, ra0=0.0, dec0=0.0)
    grid, uvw, freqs = make_grid(n_freq=n_freq, n_pix=64, fov_deg=8.0)
    pixsize = grid.pixsize

    # One source at a known on-grid pixel for phase centre (0, 0).
    a, b = 40, 28
    l = (a - 64 / 2) * pixsize
    m = (b - 64 / 2) * pixsize
    dec = np.arcsin(m)
    ra = np.arcsin(l / np.sqrt(1 - m**2))
    zp = tmp_path / "cat.zarr"
    write_zarr(zp, [[np.rad2deg(ra), np.rad2deg(dec)]], np.array([[2.0, 3.0]]),
               2, np.asarray(cfg.freqs))

    src = CatalogueSource(cfg, str(zp), fmt="zarr")
    radec, flux = src.catalogue()
    assert radec.shape == (1, 2)
    assert jnp.allclose(radec[0], jnp.array([ra, dec]), atol=1e-9)
    assert jnp.allclose(flux[0], jnp.array([2.0, 3.0]))

    image = src.image(grid)
    assert image.shape == (n_freq, 64, 64)
    assert jnp.allclose(image[:, a, b], jnp.array([2.0, 3.0]))
    assert jnp.isclose(image.sum(), 5.0)               # only that pixel is lit


def test_catalogue_zarr_missing_variable_errors(tmp_path):
    ds = xr.Dataset({"dummy": (["x"], np.array([1.0]))})
    zp = str(tmp_path / "empty.zarr")
    ds.to_zarr(zp, mode="w")
    with pytest.raises(ValueError, match="ast_p_radec"):
        CatalogueSource(make_config(), zp, fmt="zarr")


def test_catalogue_bad_fmt_errors(tmp_path):
    with pytest.raises(ValueError, match="Unsupported catalogue fmt"):
        CatalogueSource(make_config(zarr_path="x"), "x", fmt="csv")


# ── CatalogueSource (BBS) ─────────────────────────────────────────────────────

def test_bbs_wsclean_style_with_spectral_index(tmp_path):
    """WSClean header + sexagesimal coords + logarithmic spectral index."""
    bbs = tmp_path / "model.bbs"
    bbs.write_text(
        "# (Name, Type, Ra, Dec, I, SpectralIndex, LogarithmicSI, "
        "ReferenceFrequency='150000000') = format\n"
        "s0, POINT, 10:00:00.0, -20.00.00.0, 2.0, [-0.8], true, \n"
    )
    freqs = np.array([1.4e9, 1.5e9])
    cfg = make_config(n_freq=2)
    cfg.freqs = freqs
    src = CatalogueSource(cfg, str(bbs), fmt="bbs")
    radec, flux = src.catalogue()

    assert jnp.allclose(jnp.rad2deg(radec[0]), jnp.array([150.0, -20.0]), atol=1e-6)
    expected = 2.0 * (freqs / 150e6) ** (-0.8)
    assert jnp.allclose(flux[0], expected, rtol=1e-6)


def test_bbs_dp3_style_decimal_degrees_flat_spectrum(tmp_path):
    bbs = tmp_path / "dp3.bbs"
    bbs.write_text(
        "FORMAT = Name, Type, Ra, Dec, I\n"
        "a, POINT, 150.0, -20.0, 1.5\n"
        "b, POINT, 151.0, -19.0, 0.5\n"
    )
    cfg = make_config(n_freq=3)
    src = CatalogueSource(cfg, str(bbs), fmt="bbs")
    radec, flux = src.catalogue()
    assert radec.shape == (2, 2)
    assert jnp.allclose(jnp.rad2deg(radec), jnp.array([[150.0, -20.0], [151.0, -19.0]]))
    # No spectral index -> flat across the 3 channels.
    assert jnp.allclose(flux[0], 1.5) and jnp.allclose(flux[1], 0.5)


def test_bbs_missing_format_errors(tmp_path):
    bbs = tmp_path / "noheader.bbs"
    bbs.write_text("a, POINT, 150.0, -20.0, 1.0\n")
    with pytest.raises(ValueError, match="no 'format' header"):
        CatalogueSource(make_config(), str(bbs), fmt="bbs")


# ── FitsSource ────────────────────────────────────────────────────────────────

def test_fits_source_image_and_no_catalogue(tmp_path):
    from astropy.io import fits

    n_pix, n_freq = 64, 2
    data = np.zeros((n_pix, n_pix))
    data[10, 20] = 0.9
    path = tmp_path / "img.fits"
    fits.PrimaryHDU(data).writeto(path)

    cfg = make_config(n_freq=n_freq)
    grid, *_ = make_grid(n_freq=n_freq, n_pix=n_pix)
    src = FitsSource(cfg, str(path))
    image = src.image(grid)
    assert image.shape == (n_freq, n_pix, n_pix)
    assert jnp.allclose(image[0], data) and jnp.allclose(image[1], data)

    with pytest.raises(NotImplementedError, match="point catalogue"):
        src.catalogue()


# ── MSSource ──────────────────────────────────────────────────────────────────

def test_ms_source_dirty_image_and_visibilities():
    n_bl, n_time, n_freq, n_pix = 6, 3, 2, 64
    grid, uvw, freqs = make_grid(n_bl=n_bl, n_time=n_time, n_freq=n_freq, n_pix=n_pix)
    vis = (jax.random.normal(jax.random.PRNGKey(1), (n_bl, n_freq, n_time))
           + 1j * jax.random.normal(jax.random.PRNGKey(2), (n_bl, n_freq, n_time))
           ).astype(jnp.complex64)
    cfg = make_config(n_freq=n_freq, vis_obs=vis)

    src = MSSource(cfg)                          # column defaults to data_col
    assert src.visibilities() is vis
    dirty = src.image(grid)
    assert dirty.shape == (n_freq, n_pix, n_pix)
    assert jnp.all(jnp.isfinite(dirty))


def test_ms_source_missing_vis_obs_errors():
    cfg = make_config(vis_obs=None)
    with pytest.raises(ValueError, match="vis_obs is not set"):
        MSSource(cfg).visibilities()


# ── large-catalogue DFT warning ───────────────────────────────────────────────

def test_warn_if_large_catalogue():
    with pytest.warns(UserWarning, match="direct-DFT"):
        warn_if_large_catalogue(DFT_SOURCE_WARN_THRESHOLD + 1, "test")


def test_no_warn_for_small_catalogue():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_large_catalogue(10, "test")
