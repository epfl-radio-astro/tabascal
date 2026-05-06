"""Tests for tabascal.time and tabascal.coordinates."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from tabascal.time import secs_to_days, days_to_secs, jd_to_mjd, mjd_to_jd
from tabascal.coordinates import gcrf_to_uvw, itrf_to_uvw, itrf_to_uvw_jd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Three antenna positions roughly resembling a compact array (ITRF, metres)
ANTS_ITRF = jnp.array([
    [5109360.133,  2006852.586, -3238948.127],
    [5109383.997,  2006807.012, -3238949.528],
    [5109366.222,  2006839.543, -3238966.127],
])

# A test epoch close to J2000
TIMES_JD = jnp.array([2451545.0, 2451545.01, 2451545.02])

RA, DEC = 120.0, -30.0


# ---------------------------------------------------------------------------
# tabascal.time
# ---------------------------------------------------------------------------

class TestTime:

    def test_secs_to_days_roundtrip(self):
        secs = 12345.6
        assert days_to_secs(secs_to_days(secs)) == pytest.approx(secs)

    def test_secs_to_days_scaling(self):
        assert secs_to_days(86400.0) == pytest.approx(1.0)
        assert days_to_secs(1.0) == pytest.approx(86400.0)

    def test_jd_to_mjd_roundtrip(self):
        jd = 2451545.0
        assert mjd_to_jd(jd_to_mjd(jd)) == pytest.approx(jd)

    def test_jd_to_mjd_known_value(self):
        # J2000.0 is JD 2451545.0 = MJD 51544.5
        assert jd_to_mjd(2451545.0) == pytest.approx(51544.5)

    def test_mjd_to_jd_known_value(self):
        assert mjd_to_jd(51544.5) == pytest.approx(2451545.0)

    def test_offset_is_2400000_5(self):
        jd = 2459000.0
        assert jd_to_mjd(jd) == pytest.approx(jd - 2400000.5)
        assert mjd_to_jd(jd - 2400000.5) == pytest.approx(jd)

    def test_array_inputs(self):
        jds = np.array([2451545.0, 2451546.0, 2451547.0])
        mjds = jd_to_mjd(jds)
        assert np.allclose(mjd_to_jd(mjds), jds)


# ---------------------------------------------------------------------------
# tabascal.coordinates — gcrf_to_uvw
# ---------------------------------------------------------------------------

class TestGcrfToUvw:

    def test_phase_centre_direction_maps_to_w_axis(self):
        # The unit vector pointing toward the phase centre should map to (0, 0, 1)
        ra_r = jnp.deg2rad(jnp.array(RA))
        dec_r = jnp.deg2rad(jnp.array(DEC))
        source_gcrf = jnp.array([
            jnp.cos(dec_r) * jnp.cos(ra_r),
            jnp.cos(dec_r) * jnp.sin(ra_r),
            jnp.sin(dec_r),
        ])
        uvw = gcrf_to_uvw(source_gcrf, RA, DEC)
        assert float(uvw[2]) == pytest.approx(1.0, abs=1e-12)
        assert float(uvw[0]) == pytest.approx(0.0, abs=1e-12)
        assert float(uvw[1]) == pytest.approx(0.0, abs=1e-12)

    def test_rotation_preserves_norm(self):
        rng = np.random.default_rng(0)
        vecs = jnp.array(rng.standard_normal((10, 3)))
        uvw = gcrf_to_uvw(vecs, RA, DEC)
        np.testing.assert_allclose(
            jnp.linalg.norm(uvw, axis=-1),
            jnp.linalg.norm(vecs, axis=-1),
            atol=1e-12,
        )

    def test_rotation_matrix_is_orthonormal(self):
        # Build R implicitly: apply gcrf_to_uvw to the identity columns
        I = jnp.eye(3)
        R = gcrf_to_uvw(I, RA, DEC)  # rows of R are the UVW axes in GCRF
        np.testing.assert_allclose(R @ R.T, jnp.eye(3), atol=1e-12)

    def test_vector_input_shape(self):
        v = jnp.ones(3)
        assert gcrf_to_uvw(v, RA, DEC).shape == (3,)

    def test_2d_batch_shape(self):
        vecs = jnp.ones((5, 3))
        assert gcrf_to_uvw(vecs, RA, DEC).shape == (5, 3)

    def test_3d_batch_shape(self):
        vecs = jnp.ones((4, 6, 3))
        assert gcrf_to_uvw(vecs, RA, DEC).shape == (4, 6, 3)

    def test_known_value_ra0_dec0(self):
        # At ra=0, dec=0: w-hat is (1,0,0), v-hat is (0,0,1), u-hat is (0,1,0)
        x_hat = jnp.array([1.0, 0.0, 0.0])
        y_hat = jnp.array([0.0, 1.0, 0.0])
        z_hat = jnp.array([0.0, 0.0, 1.0])
        assert float(gcrf_to_uvw(x_hat, 0.0, 0.0)[2]) == pytest.approx(1.0, abs=1e-12)  # w
        assert float(gcrf_to_uvw(y_hat, 0.0, 0.0)[0]) == pytest.approx(1.0, abs=1e-12)  # u
        assert float(gcrf_to_uvw(z_hat, 0.0, 0.0)[1]) == pytest.approx(1.0, abs=1e-12)  # v


# ---------------------------------------------------------------------------
# tabascal.coordinates — itrf_to_uvw
# ---------------------------------------------------------------------------

class TestItrfToUvw:

    def test_output_shape(self):
        h0 = jnp.array([10.0, 20.0, 30.0])
        uvw = itrf_to_uvw(ANTS_ITRF, h0, DEC)
        assert uvw.shape == (3, 3, 3)  # (n_time, n_ant, 3)

    def test_first_antenna_baseline_is_zero(self):
        h0 = jnp.array([0.0, 45.0])
        uvw = itrf_to_uvw(ANTS_ITRF, h0, DEC)
        np.testing.assert_allclose(uvw[:, 0, :], 0.0, atol=1e-8)

    def test_single_time(self):
        uvw = itrf_to_uvw(ANTS_ITRF, jnp.array([0.0]), DEC)
        assert uvw.shape == (1, 3, 3)

    def test_uvw_changes_with_hour_angle(self):
        uvw_0 = itrf_to_uvw(ANTS_ITRF, jnp.array([0.0]), DEC)
        uvw_45 = itrf_to_uvw(ANTS_ITRF, jnp.array([45.0]), DEC)
        assert not jnp.allclose(uvw_0, uvw_45)


# ---------------------------------------------------------------------------
# tabascal.coordinates — itrf_to_uvw_jd
# ---------------------------------------------------------------------------

class TestItrfToUvwJd:

    def test_output_shape(self):
        uvw = itrf_to_uvw_jd(ANTS_ITRF, TIMES_JD, RA, DEC)
        assert uvw.shape == (3, 3, 3)  # (n_time, n_ant, 3)

    def test_first_antenna_baseline_is_zero(self):
        uvw = itrf_to_uvw_jd(ANTS_ITRF, TIMES_JD, RA, DEC)
        np.testing.assert_allclose(uvw[:, 0, :], 0.0, atol=1e-8)

    def test_single_time(self):
        uvw = itrf_to_uvw_jd(ANTS_ITRF, jnp.array([2451545.0]), RA, DEC)
        assert uvw.shape == (1, 3, 3)

    def test_uvw_changes_with_time(self):
        t0 = jnp.array([2451545.0])
        t1 = jnp.array([2451545.01])  # ~14 minutes later
        uvw_0 = itrf_to_uvw_jd(ANTS_ITRF, t0, RA, DEC)
        uvw_1 = itrf_to_uvw_jd(ANTS_ITRF, t1, RA, DEC)
        assert not jnp.allclose(uvw_0, uvw_1)

    def test_baseline_norm_preserved_across_times(self):
        # The ITRF → GCRF → UVW chain is a rotation so baseline norms are constant
        uvw = itrf_to_uvw_jd(ANTS_ITRF, TIMES_JD, RA, DEC)
        norms = jnp.linalg.norm(uvw, axis=-1)  # (n_time, n_ant)
        # Variance across times should be negligible
        assert float(norms.std(axis=0).max()) < 1e-6

    def test_agrees_with_itrf_to_uvw_given_gast(self):
        # Cross-check: itrf_to_uvw_jd should agree with itrf_to_uvw when the
        # hour angle is derived from the same GAST used internally by sgp4jax.
        from sgp4jax._frames import _earth_orientation
        from jax import vmap as jvmap

        jd = 2451545.0
        jd_arr = jnp.array([jd])
        jd_whole = jnp.floor(jd_arr)
        jd_frac = jd_arr - jd_whole

        _, gast_rad = jvmap(_earth_orientation)(jd_whole, jd_frac)
        gast_deg = float(jnp.rad2deg(gast_rad[0]))
        h0 = (gast_deg - RA) % 360

        uvw_h0 = itrf_to_uvw(ANTS_ITRF, jnp.array([h0]), DEC)
        uvw_jd = itrf_to_uvw_jd(ANTS_ITRF, jd_arr, RA, DEC)

        # itrf_to_uvw applies only Rz(GAST); itrf_to_uvw_jd applies the full
        # precession/nutation matrix on top.  The difference is ~arcseconds,
        # which for a ~50 m baseline amounts to a few mm.
        np.testing.assert_allclose(uvw_h0, uvw_jd, atol=0.01)
