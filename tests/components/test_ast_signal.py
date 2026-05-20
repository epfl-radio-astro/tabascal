"""Tests for tabascal.components.ast_signal.FixedPointSky and its pipeline
with PointSourceVisCalculation."""

import pytest
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

jax.config.update("jax_enable_x64", True)

from tabascal.components.ast_signal import FixedPointSky
from tabascal.components.ast_vis import PointSourceVisCalculation
from .conftest import make_constants


# ── zarr fixture helpers ──────────────────────────────────────────────────────

def write_sky_zarr(path: Path, n_src: int, n_time: int, n_freq: int, seed: int = 0):
    """Write a minimal tabsim-style zarr with point-source variables."""
    rng = np.random.default_rng(seed)
    # ast_p_radec: (n_src, 2) in degrees — sources span a wide FoV
    radec_deg = rng.uniform(low=[0.0, -30.0], high=[60.0, 30.0], size=(n_src, 2))
    # ast_p_I: (n_src, n_time, n_freq) positive fluxes in Jy
    I = np.abs(rng.normal(size=(n_src, n_time, n_freq))) + 0.1

    ds = xr.Dataset(
        {
            "ast_p_radec": (["ast_p_src", "radec"], radec_deg),
            "ast_p_I":     (["ast_p_src", "time", "freq"], I),
        }
    )
    zarr_path = str(path / "sky.zarr")
    ds.to_zarr(zarr_path, mode="w")
    return zarr_path, radec_deg, I


def make_config(zarr_path: str, n_ant: int, n_freq: int, n_time: int,
                ra0_deg: float = 30.0, dec0_deg: float = 0.0):
    a1, a2 = jnp.triu_indices(n_ant, 1)
    n_bl = a1.shape[0]
    key = jax.random.PRNGKey(7)
    # 1 km-scale baselines — the direct DFT handles any baseline length and
    # field of view, so wide-FoV sources pose no problem
    uvw = jax.random.normal(key, (n_bl, n_time, 3)) * 1e3
    freqs = jnp.linspace(1.4e9, 1.5e9, n_freq)
    return SimpleNamespace(
        n_ant=n_ant, n_bl=n_bl, n_freq=n_freq, n_time=n_time,
        uvw=uvw, freqs=freqs,
        phase_centre={"ra": ra0_deg, "dec": dec0_deg},
        args={"data": {"zarr_path": zarr_path}},
    )


# ── FixedPointSky unit tests ──────────────────────────────────────────────────

class TestFixedPointSky:

    def test_state_output_shapes(self, tmp_path):
        n_src, n_time, n_freq = 5, 3, 4
        zarr_path, _, _ = write_sky_zarr(tmp_path, n_src, n_time, n_freq)
        config = make_config(zarr_path, n_ant=3, n_freq=n_freq, n_time=2)

        comp = FixedPointSky()
        comp.setup(config)

        assert comp.state_outputs["ast_radec"].shape == (n_src, 2)
        assert comp.state_outputs["ast_I"].shape == (n_src, n_freq)

    def test_radec_converted_to_radians(self, tmp_path):
        n_src, n_time, n_freq = 3, 2, 2
        zarr_path, radec_deg, _ = write_sky_zarr(tmp_path, n_src, n_time, n_freq)
        config = make_config(zarr_path, n_ant=3, n_freq=n_freq, n_time=2)

        comp = FixedPointSky()
        comp.setup(config)

        expected = jnp.deg2rad(jnp.array(radec_deg))
        assert jnp.allclose(comp.ast_radec, expected, atol=1e-12)

    def test_flux_averaged_over_time(self, tmp_path):
        n_src, n_time, n_freq = 4, 6, 3
        zarr_path, _, I = write_sky_zarr(tmp_path, n_src, n_time, n_freq)
        config = make_config(zarr_path, n_ant=3, n_freq=n_freq, n_time=2)

        comp = FixedPointSky()
        comp.setup(config)

        expected = jnp.array(np.mean(I, axis=1))  # mean over time axis
        assert jnp.allclose(comp.ast_I, expected, atol=1e-6)

    def test_forward_writes_to_state(self, tmp_path):
        n_src, n_time, n_freq = 3, 2, 2
        zarr_path, _, _ = write_sky_zarr(tmp_path, n_src, n_time, n_freq)
        config = make_config(zarr_path, n_ant=3, n_freq=n_freq, n_time=2)

        comp = FixedPointSky()
        comp.setup(config)
        state = {}
        out = comp.build_forward()({}, state, make_constants(comp))

        assert "ast_radec" in out
        assert "ast_I" in out
        assert out["ast_radec"].shape == (n_src, 2)
        assert out["ast_I"].shape == (n_src, n_freq)

    def test_error_on_missing_zarr_path(self, tmp_path):
        config = make_config(None, n_ant=3, n_freq=2, n_time=2)
        with pytest.raises(RuntimeError, match="zarr_path"):
            FixedPointSky().setup(config)

    def test_error_on_missing_ast_p_radec(self, tmp_path):
        # Zarr exists but has no point-source variables
        ds = xr.Dataset({"dummy": (["x"], np.array([1.0]))})
        zarr_path = str(tmp_path / "empty.zarr")
        ds.to_zarr(zarr_path, mode="w")

        config = make_config(zarr_path, n_ant=3, n_freq=2, n_time=2)
        with pytest.raises(RuntimeError, match="ast_p_radec"):
            FixedPointSky().setup(config)


# ── Pipeline test: FixedPointSky → PointSourceVisCalculation ─────────────────

class TestPointSkyPipeline:

    def _run_pipeline(self, tmp_path, n_src, n_ant, n_freq, n_time, seed=0):
        ra0_deg, dec0_deg = 30.0, 0.0
        zarr_path, _, _ = write_sky_zarr(tmp_path, n_src, n_time, n_freq, seed)
        config = make_config(zarr_path, n_ant, n_freq, n_time, ra0_deg, dec0_deg)

        sky = FixedPointSky()
        vis = PointSourceVisCalculation()
        sky.setup(config)
        vis.setup(config)

        state = {**sky.state_outputs, **vis.state_outputs}
        constants = {**make_constants(sky), **make_constants(vis)}

        state = sky.build_forward()({}, state, constants)
        state = vis.build_forward()({}, state, constants)
        return state, sky, config

    @pytest.mark.parametrize("n_src,n_ant,n_freq,n_time", [
        (1, 3, 1, 1),
        (5, 4, 3, 6),
    ])
    def test_vis_ast_output_shape(self, tmp_path, n_src, n_ant, n_freq, n_time):
        state, _, config = self._run_pipeline(tmp_path, n_src, n_ant, n_freq, n_time)
        assert state["vis_ast"].shape == (config.n_bl, n_freq, n_time)

    def test_pipeline_matches_manual_dft(self, tmp_path):
        """Visibilities from the pipeline match a direct DFT over the loaded sources."""
        C = 299792458.0
        n_src, n_ant, n_freq, n_time = 3, 3, 2, 2
        ra0_deg, dec0_deg = 30.0, 0.0
        ra0 = jnp.deg2rad(ra0_deg)
        dec0 = jnp.deg2rad(dec0_deg)

        state, sky, config = self._run_pipeline(tmp_path, n_src, n_ant, n_freq, n_time, seed=42)
        nufft_vis = state["vis_ast"]

        # Reproduce the DFT manually using the values loaded by FixedPointSky
        radec = sky.ast_radec      # (n_src, 2) radians
        I = sky.ast_I              # (n_src, n_freq)
        uvw = config.uvw           # (n_bl, n_time, 3)
        freqs = config.freqs       # (n_freq,)

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
        dft_vis = jnp.sum(weights * jnp.exp(1.0j * phase), axis=2).transpose(0, 2, 1)

        assert jnp.allclose(nufft_vis, dft_vis, atol=1e-6)


# ── Laplace sparsity transform + learnable PointSky ───────────────────────────

from tabascal.components.ast_signal import PointSky
from tabascal.dist import gaussian_to_laplace, laplace_to_gaussian


def test_gaussian_to_laplace_stats():
    """White N(0,1) -> white Laplace(0, b): mean 0, variance 2 b^2."""
    z = jax.random.normal(jax.random.PRNGKey(0), (200_000,))
    for b in (0.5, 2.0):
        x = np.asarray(gaussian_to_laplace(z, b))
        assert abs(x.mean()) < 0.02
        assert np.isclose(x.var(), 2 * b**2, rtol=0.05)


def test_laplace_gaussian_roundtrip():
    z = jax.random.normal(jax.random.PRNGKey(1), (1000,))
    z2 = laplace_to_gaussian(gaussian_to_laplace(z, 1.3), 1.3)
    assert jnp.allclose(z2, z, atol=1e-8)


def write_point_catalogue(path: Path, radec_deg, flux_cube, freqs):
    ds = xr.Dataset(
        {
            "ast_p_radec": (["ast_p_src", "radec"], np.asarray(radec_deg)),
            "ast_p_I": (["ast_p_src", "time", "freq"], np.asarray(flux_cube)),
        },
        coords={"freq": np.asarray(freqs)},
    )
    zp = str(path / "point.zarr")
    ds.to_zarr(zp, mode="w")
    return zp


def make_point_config(zarr_path, width=1.0, init="sample", n_ant=4, n_freq=2,
                      n_time=2, freqs=None):
    a1, a2 = jnp.triu_indices(n_ant, 1)
    n_bl = a1.shape[0]
    uvw = jax.random.normal(jax.random.PRNGKey(3), (n_bl, n_time, 3)) * 100.0
    freqs = jnp.linspace(1.4e9, 1.5e9, n_freq) if freqs is None else jnp.asarray(freqs)
    return SimpleNamespace(
        n_ant=n_ant, n_bl=n_bl, n_freq=n_freq, n_time=n_time,
        uvw=uvw, freqs=freqs, phase_centre={"ra": 27.0, "dec": -30.0},
        args={"data": {"zarr_path": zarr_path},
              "ast": {"point": {"laplace_width": width, "init": init}}},
    )


class TestPointSky:

    def _catalogue(self, tmp_path, n_src=5, n_time=2, n_freq=2):
        freqs = np.linspace(1.4e9, 1.5e9, n_freq)
        rng = np.random.default_rng(0)
        radec = rng.uniform([20.0, -32.0], [34.0, -28.0], size=(n_src, 2))
        flux = np.abs(rng.normal(size=(n_src, n_time, n_freq))) + 0.1
        zp = write_point_catalogue(tmp_path, radec, flux, freqs)
        return zp, radec, flux, freqs

    def test_shapes(self, tmp_path):
        zp, radec, _, freqs = self._catalogue(tmp_path)
        config = make_point_config(zp, freqs=freqs)
        ps = PointSky()
        ps.setup(config)
        out = ps.build_forward()(ps.init_params_base, {}, make_constants(ps))
        assert out["ast_radec"].shape == (radec.shape[0], 2)
        assert out["ast_I"].shape == (radec.shape[0], config.n_freq)
        assert jnp.all(jnp.isfinite(out["ast_I"]))

    def test_zeros_init_all_off(self, tmp_path):
        zp, _, _, freqs = self._catalogue(tmp_path)
        config = make_point_config(zp, init="zeros", freqs=freqs)
        ps = PointSky()
        ps.setup(config)
        out = ps.build_forward()(ps.init_params_base, {}, make_constants(ps))
        assert jnp.allclose(out["ast_I"], 0.0, atol=1e-12)

    def test_truth_init_recovers_flux(self, tmp_path):
        zp, _, flux, freqs = self._catalogue(tmp_path)
        config = make_point_config(zp, init="truth", width=1.0, freqs=freqs)
        ps = PointSky()
        ps.setup(config)
        out = ps.build_forward()(ps.init_params_base, {}, make_constants(ps))
        assert jnp.allclose(out["ast_I"], flux.mean(axis=1), atol=1e-6)

    @pytest.mark.parametrize("init", ["zeros", "sample", "truth"])
    def test_init_paths(self, tmp_path, init):
        zp, _, _, freqs = self._catalogue(tmp_path)
        config = make_point_config(zp, init=init, freqs=freqs)
        ps = PointSky()
        ps.setup(config)
        assert ps.init_params_base["ast_I_base"].shape == (ps.n_src, config.n_freq)

    def test_pipeline_to_vis(self, tmp_path):
        zp, radec, _, freqs = self._catalogue(tmp_path)
        config = make_point_config(zp, freqs=freqs)
        ps = PointSky()
        pv = PointSourceVisCalculation()
        ps.setup(config)
        pv.setup(config)
        state = {**ps.state_outputs, **pv.state_outputs,
                 "vis_ast": jnp.zeros((config.n_bl, config.n_freq, config.n_time), complex)}
        constants = {**make_constants(ps), **make_constants(pv)}
        params = ps.init_params_base
        state = ps.build_forward()(params, state, constants)
        state = pv.build_forward()(params, state, constants)
        assert state["vis_ast"].shape == (config.n_bl, config.n_freq, config.n_time)
        assert jnp.all(jnp.isfinite(state["vis_ast"]))

    def test_differentiable(self, tmp_path):
        zp, _, _, freqs = self._catalogue(tmp_path)
        config = make_point_config(zp, freqs=freqs)
        ps = PointSky()
        ps.setup(config)
        constants = make_constants(ps)
        fwd = ps.build_forward()
        params = ps.init_params_base

        def f(p):
            return fwd(p, {}, constants)["ast_I"]

        _, vjp_fn = jax.vjp(f, params)
        (grad,) = vjp_fn(jnp.ones((ps.n_src, config.n_freq)))
        assert grad["ast_I_base"].shape == (ps.n_src, config.n_freq)
        assert jnp.all(jnp.isfinite(grad["ast_I_base"]))
        assert jnp.any(jnp.abs(grad["ast_I_base"]) > 0)

    def test_invalid_init_errors(self, tmp_path):
        zp, _, _, freqs = self._catalogue(tmp_path)
        config = make_point_config(zp, init="bogus", freqs=freqs)
        with pytest.raises(RuntimeError, match="init type"):
            PointSky().setup(config)
