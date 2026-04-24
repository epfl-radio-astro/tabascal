"""Tests for tabascal.components.trajectory — FixedOrbit, PhaseCalculationRFI,
SGP4LEONoDragOrbit, and SGP4LEOOrbit.

Space-Track-dependent tests are skipped automatically when credentials are
not configured on the current machine.
"""

import pytest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro

jax.config.update("jax_enable_x64", True)

from tabascal.components.trajectory import FixedOrbit, PhaseCalculationRFI
from tabascal.interferometry import get_rfi_phase


def make_constants(comp):
    return {f"{comp.prefix}/{k}": v for k, v in comp.build_constants().items()}


# ---------------------------------------------------------------------------
# Space-Track credential detection
# ---------------------------------------------------------------------------

def _has_spacetrack_credentials() -> bool:
    try:
        from tabascal.tle import load_spacetrack_credentials
        user, passwd = load_spacetrack_credentials()
        return user is not None and passwd is not None
    except Exception:
        return False


requires_spacetrack = pytest.mark.skipif(
    not _has_spacetrack_credentials(),
    reason="Space-Track credentials not configured — skipping",
)


# ---------------------------------------------------------------------------
# Common TLE strings (ISS, epoch 2008-09-20)
# These are public TLEs that require no Space-Track account to use.
# ---------------------------------------------------------------------------

_ISS_TLE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
_ISS_TLE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

# A second LEO TLE for multi-satellite tests (Envisat, epoch ~2008)
_ENVISAT_TLE1 = "1 27386U 02009A   08264.20891862  .00000135  00000-0  87244-4 0  3620"
_ENVISAT_TLE2 = "2 27386  98.1745 271.4088 0001247  86.4921 273.6404 14.37834818337895"

# Epoch JD for the ISS TLE above: year 2008 + day 264.51782528
_EPOCH_JD = 2454467.5 + 264.51782528   # ≈ 2454732.018


# ---------------------------------------------------------------------------
# Mock config helpers
# ---------------------------------------------------------------------------

# Approximate MeerKAT antenna positions in ITRF (metres).
# Extended with synthetic offsets so tests with n_ant > 6 still work.
_MEERKAT_ITRF_BASE = jnp.array([
    [5109360.133, 2006852.586, -3238948.127],
    [5109396.146, 2006858.412, -3238845.401],
    [5109433.965, 2006880.659, -3238746.399],
    [5109453.087, 2006888.908, -3238686.003],
    [5109476.343, 2006904.234, -3238609.113],
    [5109502.877, 2006921.456, -3238524.891],
])


def _build_ants_itrf(n_ant: int) -> jnp.ndarray:
    """Return n_ant ITRF positions, extending the base array with synthetic offsets."""
    if n_ant <= len(_MEERKAT_ITRF_BASE):
        return _MEERKAT_ITRF_BASE[:n_ant]
    extra = jnp.stack([
        _MEERKAT_ITRF_BASE[-1] + jnp.array([i * 60.0, i * 20.0, 0.0])
        for i in range(1, n_ant - len(_MEERKAT_ITRF_BASE) + 1)
    ])
    return jnp.concatenate([_MEERKAT_ITRF_BASE, extra], axis=0)


_MEERKAT_ITRF = _MEERKAT_ITRF_BASE  # kept for backward compat inside this module

_PHASE_CENTRE = {"ra": 21.44417, "dec": -30.71278}


def make_trajectory_config(
    n_ant=4,
    n_rfi=1,
    n_freq=2,
    n_time=4,
    n_int_time=2,
    n_int_freq=1,
    tles=None,
    epoch_jd=None,
):
    """Build a minimal mock TabConfig for trajectory components."""
    n_freq_fine = n_freq * n_int_freq
    n_time_fine = n_time * n_int_time
    ep = _EPOCH_JD if epoch_jd is None else epoch_jd

    times_jd_fine = jnp.linspace(ep, ep + n_time_fine * 8.0 / 86400, n_time_fine)
    freqs_fine = jnp.linspace(1.4e9, 1.41e9, n_freq_fine)
    freqs = jnp.linspace(1.4e9, 1.41e9, n_freq)
    times = jnp.linspace(0.0, n_time * 8.0, n_time)
    times_fine = jnp.linspace(0.0, n_time * 8.0, n_time_fine)

    if tles is None:
        tles = np.array([[_ISS_TLE1, _ISS_TLE2]] * n_rfi)

    return SimpleNamespace(
        n_ant=n_ant,
        n_rfi=n_rfi,
        n_freq=n_freq,
        n_time=n_time,
        n_freq_fine=n_freq_fine,
        n_time_fine=n_time_fine,
        n_int_time=n_int_time,
        n_int_freq=n_int_freq,
        tles=tles,
        elements=jnp.zeros((n_rfi, 6)),  # placeholder — not used by FixedOrbit forward
        epoch_jd=jnp.full((n_rfi,), ep),
        times_jd=jnp.linspace(ep, ep + n_time * 8.0 / 86400, n_time),
        times_jd_fine=times_jd_fine,
        ants_itrf=_build_ants_itrf(n_ant),
        phase_centre=_PHASE_CENTRE,
        freqs=freqs,
        freqs_fine=freqs_fine,
        times=times,
        times_fine=times_fine,
        args={"rfi": {"freq_int_samples": n_int_freq}},
    )


# ---------------------------------------------------------------------------
# PhaseCalculationRFI
# ---------------------------------------------------------------------------

class TestPhaseCalculationRFI:

    def test_setup_succeeds(self):
        """Component initialises without error with a default mock config."""
        cfg = make_trajectory_config()
        comp = PhaseCalculationRFI()
        comp.setup(cfg)

    def test_setup_validates_dimensions(self):
        """If the config is self-consistent, _validate_dimensions must not raise."""
        cfg = make_trajectory_config(n_ant=4, n_rfi=2, n_freq=3, n_time=6, n_int_time=2)
        comp = PhaseCalculationRFI()
        comp.setup(cfg)  # internally calls _validate_dimensions
        assert comp.ants_uvw.shape == (cfg.n_ant, cfg.n_time_fine, 3)
        assert comp.ants_xyz.shape == (cfg.n_ant, cfg.n_time_fine, 3)

    def test_set_params_is_identity(self):
        """build_set_params returns a no-op pass-through."""
        cfg = make_trajectory_config()
        comp = PhaseCalculationRFI()
        comp.setup(cfg)
        sentinel = {"foo": jnp.array(1.0)}
        out = comp.build_set_params()(sentinel)
        assert out is sentinel

    def test_forward_output_shape(self):
        """Forward pass produces rfi_phase with shape (n_rfi, n_ant, n_freq_fine, n_time_fine)."""
        n_ant, n_rfi, n_freq, n_time, n_int_time = 4, 2, 3, 5, 2
        cfg = make_trajectory_config(
            n_ant=n_ant, n_rfi=n_rfi, n_freq=n_freq,
            n_time=n_time, n_int_time=n_int_time,
        )
        comp = PhaseCalculationRFI()
        comp.setup(cfg)

        n_time_fine = n_time * n_int_time
        n_freq_fine = n_freq
        rfi_xyz = jnp.zeros((n_rfi, n_time_fine, 3)) + jnp.array([7e6, 0.0, 0.0])
        state = {"rfi_xyz": rfi_xyz}

        out = comp.build_forward()({}, state, make_constants(comp))

        assert "rfi_phase" in out
        assert out["rfi_phase"].shape == (n_rfi, n_ant, n_freq_fine, n_time_fine)

    def test_forward_phase_is_finite(self):
        """All phase values should be finite for a realistic satellite position."""
        n_ant, n_rfi = 4, 1
        cfg = make_trajectory_config(n_ant=n_ant, n_rfi=n_rfi)
        comp = PhaseCalculationRFI()
        comp.setup(cfg)

        # ISS-like position: ~400 km altitude, in the GCRF frame
        rfi_xyz = jnp.broadcast_to(
            jnp.array([[6.8e6, 0.0, 0.0]]),
            (n_rfi, cfg.n_time_fine, 3),
        )
        out = comp.build_forward()({}, {"rfi_xyz": rfi_xyz}, make_constants(comp))
        assert jnp.all(jnp.isfinite(out["rfi_phase"]))

    def test_forward_phase_varies_across_antennas(self):
        """Different antennas should see different phase delays."""
        n_ant, n_rfi = 4, 1
        cfg = make_trajectory_config(n_ant=n_ant, n_rfi=n_rfi)
        comp = PhaseCalculationRFI()
        comp.setup(cfg)

        rfi_xyz = jnp.broadcast_to(
            jnp.array([[6.8e6, 0.0, 0.0]]),
            (n_rfi, cfg.n_time_fine, 3),
        )
        out = comp.build_forward()({}, {"rfi_xyz": rfi_xyz}, make_constants(comp))
        phase = out["rfi_phase"]  # (n_rfi, n_ant, n_freq_fine, n_time_fine)
        # Not all antennas should have identical phases
        assert not jnp.allclose(phase[0, 0], phase[0, 1])

    def test_forward_preserves_rfi_xyz_in_state(self):
        """Forward pass copies rfi_xyz through to the output state unchanged."""
        cfg = make_trajectory_config()
        comp = PhaseCalculationRFI()
        comp.setup(cfg)

        rfi_xyz = jnp.zeros((cfg.n_rfi, cfg.n_time_fine, 3)) + 6.8e6
        state = {"rfi_xyz": rfi_xyz}
        out = comp.build_forward()({}, state, make_constants(comp))

        assert "rfi_xyz" in out
        assert jnp.array_equal(out["rfi_xyz"], rfi_xyz)

    @pytest.mark.parametrize("n_ant,n_rfi,n_freq,n_time,n_int_time", [
        (2, 1, 1, 2, 1),
        (4, 2, 4, 8, 2),
        (8, 3, 2, 6, 3),
    ])
    def test_parametric_sizes(self, n_ant, n_rfi, n_freq, n_time, n_int_time):
        """Output shape and finiteness verified across a range of dimension combinations."""
        cfg = make_trajectory_config(
            n_ant=n_ant, n_rfi=n_rfi, n_freq=n_freq,
            n_time=n_time, n_int_time=n_int_time,
        )
        comp = PhaseCalculationRFI()
        comp.setup(cfg)

        n_time_fine = n_time * n_int_time
        rfi_xyz = jnp.zeros((n_rfi, n_time_fine, 3)) + jnp.array([6.8e6, 0.0, 0.0])
        out = comp.build_forward()({}, {"rfi_xyz": rfi_xyz}, make_constants(comp))

        assert out["rfi_phase"].shape == (n_rfi, n_ant, n_freq, n_time_fine)
        assert jnp.all(jnp.isfinite(out["rfi_phase"]))

    # Low-level: _compute_ant_pos

    def test_compute_ant_pos_xyz_earth_radius(self):
        """ants_xyz (GCRF) should be at Earth's surface radius (~6.37e6 m)."""
        cfg = make_trajectory_config(n_ant=4)
        comp = PhaseCalculationRFI()
        comp.setup(cfg)
        radii = jnp.linalg.norm(comp.ants_xyz, axis=-1)
        assert jnp.all(radii > 6.35e6), "Antenna radius below Earth surface"
        assert jnp.all(radii < 6.40e6), "Antenna radius too large for ground-based telescope"

    def test_compute_ant_pos_uvw_shape_and_finite(self):
        """ants_uvw must have shape (n_ant, n_time_fine, 3) with all finite values."""
        n_ant, n_time, n_int_time = 4, 4, 2
        cfg = make_trajectory_config(n_ant=n_ant, n_time=n_time, n_int_time=n_int_time)
        comp = PhaseCalculationRFI()
        comp.setup(cfg)
        assert comp.ants_uvw.shape == (n_ant, n_time * n_int_time, 3)
        assert jnp.all(jnp.isfinite(comp.ants_uvw))

    def test_compute_ant_pos_distinct_across_antennas(self):
        """Different antennas must have distinct GCRF positions."""
        cfg = make_trajectory_config(n_ant=4)
        comp = PhaseCalculationRFI()
        comp.setup(cfg)
        assert not jnp.allclose(comp.ants_xyz[0, 0], comp.ants_xyz[1, 0])


# ---------------------------------------------------------------------------
# FixedOrbit
# ---------------------------------------------------------------------------

class TestFixedOrbit:

    def test_setup_succeeds(self):
        """Component propagates the TLE orbit and pre-computes phase without error."""
        cfg = make_trajectory_config(n_rfi=1)
        comp = FixedOrbit()
        comp.setup(cfg)

    def test_rfi_xyz_shape(self):
        """Pre-computed satellite positions stored at setup have shape (n_rfi, n_time_fine, 3)."""
        n_rfi, n_time, n_int_time = 1, 4, 2
        cfg = make_trajectory_config(n_rfi=n_rfi, n_time=n_time, n_int_time=n_int_time)
        comp = FixedOrbit()
        comp.setup(cfg)
        n_time_fine = n_time * n_int_time
        assert comp.rfi_xyz.shape == (n_rfi, n_time_fine, 3)

    def test_rfi_phase_shape(self):
        """Pre-computed phase stored at setup has shape (n_rfi, n_ant, n_freq, n_time_fine)."""
        n_rfi, n_ant, n_freq, n_time, n_int_time = 1, 4, 2, 4, 2
        cfg = make_trajectory_config(
            n_rfi=n_rfi, n_ant=n_ant, n_freq=n_freq,
            n_time=n_time, n_int_time=n_int_time,
        )
        comp = FixedOrbit()
        comp.setup(cfg)
        n_time_fine = n_time * n_int_time
        assert comp.rfi_phase.shape == (n_rfi, n_ant, n_freq, n_time_fine)

    def test_rfi_xyz_nonzero(self):
        """Propagated satellite positions must be non-zero (orbit was computed)."""
        cfg = make_trajectory_config(n_rfi=1)
        comp = FixedOrbit()
        comp.setup(cfg)
        assert not jnp.allclose(comp.rfi_xyz, 0.0)

    def test_rfi_xyz_altitude_reasonable(self):
        """ISS is at ~400 km altitude — distance from Earth's centre ≈ 6.8e6 m."""
        cfg = make_trajectory_config(n_rfi=1)
        comp = FixedOrbit()
        comp.setup(cfg)
        radii = jnp.linalg.norm(comp.rfi_xyz[0], axis=-1)
        assert jnp.all(radii > 6.0e6)
        assert jnp.all(radii < 8.0e6)

    def test_rfi_phase_finite(self):
        """All pre-computed phase values are finite."""
        cfg = make_trajectory_config(n_rfi=1)
        comp = FixedOrbit()
        comp.setup(cfg)
        assert jnp.all(jnp.isfinite(comp.rfi_phase))

    def test_forward_adds_rfi_xyz_and_phase_to_state(self):
        """Forward pass inserts rfi_xyz and rfi_phase into the state dict."""
        cfg = make_trajectory_config(n_rfi=1)
        comp = FixedOrbit()
        comp.setup(cfg)
        state = {}
        out = comp.build_forward()({}, state, make_constants(comp))
        assert "rfi_xyz" in out
        assert "rfi_phase" in out

    def test_forward_output_matches_precomputed(self):
        """Forward pass must return the same pre-computed arrays stored at setup time."""
        cfg = make_trajectory_config(n_rfi=1)
        comp = FixedOrbit()
        comp.setup(cfg)
        out = comp.build_forward()({}, {}, make_constants(comp))
        assert jnp.array_equal(out["rfi_xyz"], comp.rfi_xyz)
        assert jnp.array_equal(out["rfi_phase"], comp.rfi_phase)

    def test_forward_is_deterministic(self):
        """Calling build_forward twice with the same input gives the same result."""
        cfg = make_trajectory_config(n_rfi=1)
        comp = FixedOrbit()
        comp.setup(cfg)
        constants = make_constants(comp)
        fwd = comp.build_forward()
        out1 = fwd({}, {}, constants)
        out2 = fwd({}, {}, constants)
        assert jnp.array_equal(out1["rfi_xyz"], out2["rfi_xyz"])

    def test_two_satellites_shape(self):
        """Two distinct TLEs produce position and phase arrays of the correct shape."""
        n_rfi = 2
        tles = np.array([
            [_ISS_TLE1, _ISS_TLE2],
            [_ENVISAT_TLE1, _ENVISAT_TLE2],
        ])
        n_ant, n_freq, n_time, n_int_time = 4, 2, 4, 2
        cfg = make_trajectory_config(
            n_rfi=n_rfi, n_ant=n_ant, n_freq=n_freq,
            n_time=n_time, n_int_time=n_int_time,
            tles=tles,
        )
        comp = FixedOrbit()
        comp.setup(cfg)
        n_time_fine = n_time * n_int_time
        assert comp.rfi_xyz.shape == (n_rfi, n_time_fine, 3)
        assert comp.rfi_phase.shape == (n_rfi, n_ant, n_freq, n_time_fine)

    def test_two_satellites_have_different_positions(self):
        """Two distinct TLEs must propagate to distinct positions."""
        tles = np.array([
            [_ISS_TLE1, _ISS_TLE2],
            [_ENVISAT_TLE1, _ENVISAT_TLE2],
        ])
        cfg = make_trajectory_config(n_rfi=2, tles=tles)
        comp = FixedOrbit()
        comp.setup(cfg)
        assert not jnp.allclose(comp.rfi_xyz[0], comp.rfi_xyz[1])

    def test_build_set_params_is_identity(self):
        """FixedOrbit.build_set_params returns a pass-through with no side effects."""
        cfg = make_trajectory_config(n_rfi=1)
        comp = FixedOrbit()
        comp.setup(cfg)

        sentinel = {"foo": jnp.array(1.0)}
        set_params = comp.build_set_params()
        out = set_params(sentinel)

        assert out is sentinel

    # Low-level: _compute_rfi_phase

    def test_compute_rfi_phase_consistent_with_get_rfi_phase(self):
        """Phase stored at setup must equal get_rfi_phase called with the same arrays."""
        cfg = make_trajectory_config(n_rfi=1, n_ant=4, n_freq=2, n_time=4, n_int_time=2)
        comp = FixedOrbit()
        comp.setup(cfg)
        expected = get_rfi_phase(comp.rfi_xyz, comp.ants_uvw, comp.ants_xyz, comp.freqs_fine)
        assert jnp.allclose(comp.rfi_phase, expected)

    def test_compute_rfi_phase_xyz_is_satellite_altitude(self):
        """rfi_xyz computed during _compute_rfi_phase should be at LEO altitude."""
        cfg = make_trajectory_config(n_rfi=1)
        comp = FixedOrbit()
        comp.setup(cfg)
        radii = jnp.linalg.norm(comp.rfi_xyz[0], axis=-1)
        assert jnp.all(radii > 6.0e6), "Satellite altitude below LEO"
        assert jnp.all(radii < 8.0e6), "Satellite altitude above expected LEO range"


# ---------------------------------------------------------------------------
# Bundled TLE constants (NAVSTAR 18 / 67, cached 2023-02-21)
# These match tabascal/data/tles/2023-02-21-navstar.json so no Space-Track
# call is needed — get_tles_by_id finds the file from cache.
# ---------------------------------------------------------------------------

# JD of 2023-02-21 13:55:04.589 UTC — must match the date prefix of the bundled file.
_BUNDLED_TLE_EPOCH_JD = 2459997.079914223  # 2023-02-21 13:55:04.589 UTC => GMSA = 0
_BUNDLED_NORAD_IDS = [20452, 38833]


# ---------------------------------------------------------------------------
# SGP4LEONoDragOrbit
# ---------------------------------------------------------------------------

class TestSGP4LEONoDragOrbit:

    def _make_config(self, n_ant=4, n_freq=2, n_time=4, n_int_time=2, n_int_freq=1):
        # Use the bundled TLE epoch so get_tles_by_id hits the repo cache file
        # and never contacts Space-Track.
        epoch_jd = _BUNDLED_TLE_EPOCH_JD
        n_rfi = len(_BUNDLED_NORAD_IDS)
        n_time_fine = n_time * n_int_time
        n_freq_fine = n_freq * n_int_freq

        times_jd = jnp.linspace(epoch_jd, epoch_jd + n_time * 8.0 / 86400, n_time)
        times_jd_fine = jnp.linspace(epoch_jd, epoch_jd + n_time_fine * 8.0 / 86400, n_time_fine)

        return SimpleNamespace(
            n_ant=n_ant,
            n_rfi=n_rfi,
            n_freq=n_freq,
            n_time=n_time,
            n_freq_fine=n_freq_fine,
            n_time_fine=n_time_fine,
            n_int_time=n_int_time,
            n_int_freq=n_int_freq,
            tles=None,
            elements=jnp.zeros((n_rfi, 6)),
            epoch_jd=jnp.full((n_rfi,), epoch_jd),
            times_jd=times_jd,
            times_jd_fine=times_jd_fine,
            ants_itrf=_build_ants_itrf(n_ant),
            phase_centre=_PHASE_CENTRE,
            freqs=jnp.linspace(1.4e9, 1.41e9, n_freq),
            freqs_fine=jnp.linspace(1.4e9, 1.41e9, n_freq_fine),
            times=jnp.linspace(0.0, n_time * 8.0, n_time),
            times_fine=jnp.linspace(0.0, n_time_fine * 8.0, n_time_fine),
            norad_ids=_BUNDLED_NORAD_IDS,
            args={"rfi": {"freq_int_samples": n_int_freq}},
        )

    def test_setup_succeeds(self):
        """Component loads TLEs from the repo cache and initialises without error."""
        from tabascal.components.trajectory import SGP4LEONoDragOrbit
        cfg = self._make_config()
        comp = SGP4LEONoDragOrbit()
        comp.setup(cfg)

    def test_rfi_xyz_shape(self):
        """Initial state_outputs['rfi_xyz'] placeholder has shape (n_rfi, n_time_fine, 3)."""
        from tabascal.components.trajectory import SGP4LEONoDragOrbit
        cfg = self._make_config()
        comp = SGP4LEONoDragOrbit()
        comp.setup(cfg)
        assert comp.state_outputs["rfi_xyz"].shape == (cfg.n_rfi, cfg.n_time_fine, 3)

    def test_init_params_base_shape(self):
        """Initial base orbit parameters have shape (n_rfi, 6) — bstar excluded from learnable params."""
        from tabascal.components.trajectory import SGP4LEONoDragOrbit
        cfg = self._make_config()
        comp = SGP4LEONoDragOrbit()
        comp.setup(cfg)
        assert comp.init_params_base["rfi_orbit_base"].shape == (cfg.n_rfi, 6)

    def test_prior_covariance_positive_definite(self):
        """L_rfi_orbit must be lower-triangular with positive diagonal."""
        from tabascal.components.trajectory import SGP4LEONoDragOrbit
        cfg = self._make_config()
        comp = SGP4LEONoDragOrbit()
        comp.setup(cfg)
        for i in range(cfg.n_rfi):
            diag = jnp.diag(comp.L_rfi_orbit[i])
            assert jnp.all(diag > 0), f"Cholesky diagonal not positive for satellite {i}"

    def test_forward_output_shapes(self):
        """Forward pass produces rfi_xyz (n_rfi, n_time_fine, 3) and elements (n_rfi, 6)."""
        from tabascal.components.trajectory import SGP4LEONoDragOrbit
        cfg = self._make_config()
        comp = SGP4LEONoDragOrbit()
        comp.setup(cfg)

        params = {"rfi_orbit_base": comp.init_params_base["rfi_orbit_base"]}
        out = comp.build_forward()(params, {}, make_constants(comp))

        assert out["rfi_xyz"].shape == (cfg.n_rfi, cfg.n_time_fine, 3)
        assert out["elements"].shape == (cfg.n_rfi, 6)

    def test_forward_rfi_xyz_finite(self):
        """SGP4-propagated satellite positions from the forward pass are all finite."""
        from tabascal.components.trajectory import SGP4LEONoDragOrbit
        cfg = self._make_config()
        comp = SGP4LEONoDragOrbit()
        comp.setup(cfg)

        params = {"rfi_orbit_base": comp.init_params_base["rfi_orbit_base"]}
        out = comp.build_forward()(params, {}, make_constants(comp))

        assert jnp.all(jnp.isfinite(out["rfi_xyz"]))

    def test_build_set_params_samples_correct_shapes(self):
        """build_set_params must sample rfi_orbit_base with shape (n_rfi, 6) inside a NumPyro trace."""
        from tabascal.components.trajectory import SGP4LEONoDragOrbit
        cfg = self._make_config()
        comp = SGP4LEONoDragOrbit()
        comp.setup(cfg)

        set_params = comp.build_set_params()
        with numpyro.handlers.seed(rng_seed=0):
            params = set_params({})

        assert "rfi_orbit_base" in params
        assert params["rfi_orbit_base"].shape == (cfg.n_rfi, 6)

    def test_forward_transform_roundtrip(self):
        """inv_transform(forward_transform(x)) == x."""
        from tabascal.components.trajectory import SGP4LEONoDragOrbit
        cfg = self._make_config()
        comp = SGP4LEONoDragOrbit()
        comp.setup(cfg)

        base = jax.random.normal(jax.random.PRNGKey(0), (cfg.n_rfi, 6))
        transformed = comp.forward_transform(base, comp.L_rfi_orbit, comp.mu_rfi_orbit)
        recovered = comp.inv_transform(transformed, comp.L_rfi_orbit, comp.mu_rfi_orbit)

        assert jnp.allclose(recovered, base, atol=1e-6)

    def test_inv_transform_roundtrip(self):
        """forward_transform(inv_transform(x)) == x."""
        from tabascal.components.trajectory import SGP4LEONoDragOrbit
        cfg = self._make_config()
        comp = SGP4LEONoDragOrbit()
        comp.setup(cfg)

        params = comp.mu_rfi_orbit + jax.random.normal(jax.random.PRNGKey(1), (cfg.n_rfi, 6)) * 0.01
        base = comp.inv_transform(params, comp.L_rfi_orbit, comp.mu_rfi_orbit)
        recovered = comp.forward_transform(base, comp.L_rfi_orbit, comp.mu_rfi_orbit)

        assert jnp.allclose(recovered, params, atol=1e-6)

    # Low-level: sats_init

    def test_sats_init_direct_call(self):
        """sats_init called directly with comp.elements must return without error."""
        from tabascal.components.trajectory import SGP4LEONoDragOrbit
        cfg = self._make_config()
        comp = SGP4LEONoDragOrbit()
        comp.setup(cfg)
        sats = comp.sats_init(comp.elements)
        assert sats is not None

    def test_sats_init_produces_valid_positions(self):
        """sats_init output propagated via sgp4jax must yield finite LEO positions."""
        import sgp4jax
        from tabascal.components.trajectory import SGP4LEONoDragOrbit
        cfg = self._make_config()
        comp = SGP4LEONoDragOrbit()
        comp.setup(cfg)
        sats = comp.sats_init(comp.elements)
        positions, _ = sgp4jax.gcrf_positions_multi_leo(sats, comp.times_jd_fine)
        assert positions.shape == (cfg.n_rfi, cfg.n_time_fine, 3)
        assert jnp.all(jnp.isfinite(positions))


# ---------------------------------------------------------------------------
# SGP4LEOOrbit
# ---------------------------------------------------------------------------

class TestSGP4LEOOrbit:

    def _make_config(self, n_ant=4, n_freq=2, n_time=4, n_int_time=2, n_int_freq=1):
        epoch_jd = _BUNDLED_TLE_EPOCH_JD
        n_rfi = len(_BUNDLED_NORAD_IDS)
        n_time_fine = n_time * n_int_time
        n_freq_fine = n_freq * n_int_freq

        times_jd = jnp.linspace(epoch_jd, epoch_jd + n_time * 8.0 / 86400, n_time)
        times_jd_fine = jnp.linspace(epoch_jd, epoch_jd + n_time_fine * 8.0 / 86400, n_time_fine)

        return SimpleNamespace(
            n_ant=n_ant,
            n_rfi=n_rfi,
            n_freq=n_freq,
            n_time=n_time,
            n_freq_fine=n_freq_fine,
            n_time_fine=n_time_fine,
            n_int_time=n_int_time,
            n_int_freq=n_int_freq,
            tles=None,
            elements=jnp.zeros((n_rfi, 7)),
            epoch_jd=jnp.full((n_rfi,), epoch_jd),
            times_jd=times_jd,
            times_jd_fine=times_jd_fine,
            ants_itrf=_build_ants_itrf(n_ant),
            phase_centre=_PHASE_CENTRE,
            freqs=jnp.linspace(1.4e9, 1.41e9, n_freq),
            freqs_fine=jnp.linspace(1.4e9, 1.41e9, n_freq_fine),
            times=jnp.linspace(0.0, n_time * 8.0, n_time),
            times_fine=jnp.linspace(0.0, n_time_fine * 8.0, n_time_fine),
            norad_ids=_BUNDLED_NORAD_IDS,
            args={"rfi": {"freq_int_samples": n_int_freq}},
        )

    def test_setup_succeeds(self):
        """Component loads TLEs from the repo cache and initialises without error."""
        from tabascal.components.trajectory import SGP4LEOOrbit
        cfg = self._make_config()
        comp = SGP4LEOOrbit()
        comp.setup(cfg)

    def test_rfi_xyz_shape(self):
        """Initial state_outputs['rfi_xyz'] placeholder has shape (n_rfi, n_time_fine, 3)."""
        from tabascal.components.trajectory import SGP4LEOOrbit
        cfg = self._make_config()
        comp = SGP4LEOOrbit()
        comp.setup(cfg)
        assert comp.state_outputs["rfi_xyz"].shape == (cfg.n_rfi, cfg.n_time_fine, 3)

    def test_init_params_base_shape(self):
        """SGP4LEOOrbit has 7 orbit parameters (includes bstar)."""
        from tabascal.components.trajectory import SGP4LEOOrbit
        cfg = self._make_config()
        comp = SGP4LEOOrbit()
        comp.setup(cfg)
        assert comp.init_params_base["rfi_orbit_base"].shape == (cfg.n_rfi, 7)

    def test_prior_covariance_positive_definite(self):
        """Cholesky factor L_rfi_orbit (7x7) has a positive diagonal for every satellite."""
        from tabascal.components.trajectory import SGP4LEOOrbit
        cfg = self._make_config()
        comp = SGP4LEOOrbit()
        comp.setup(cfg)
        for i in range(cfg.n_rfi):
            diag = jnp.diag(comp.L_rfi_orbit[i])
            assert jnp.all(diag > 0), f"Cholesky diagonal not positive for satellite {i}"

    def test_forward_output_shapes(self):
        """Forward pass produces rfi_xyz (n_rfi, n_time_fine, 3) and elements (n_rfi, 7)."""
        from tabascal.components.trajectory import SGP4LEOOrbit
        cfg = self._make_config()
        comp = SGP4LEOOrbit()
        comp.setup(cfg)

        params = {"rfi_orbit_base": comp.init_params_base["rfi_orbit_base"]}
        out = comp.build_forward()(params, {}, make_constants(comp))

        assert out["rfi_xyz"].shape == (cfg.n_rfi, cfg.n_time_fine, 3)
        assert out["elements"].shape == (cfg.n_rfi, 7)

    def test_forward_rfi_xyz_finite(self):
        """SGP4-propagated satellite positions from the forward pass are all finite."""
        from tabascal.components.trajectory import SGP4LEOOrbit
        cfg = self._make_config()
        comp = SGP4LEOOrbit()
        comp.setup(cfg)

        params = {"rfi_orbit_base": comp.init_params_base["rfi_orbit_base"]}
        out = comp.build_forward()(params, {}, make_constants(comp))

        assert jnp.all(jnp.isfinite(out["rfi_xyz"]))

    def test_build_set_params_samples_correct_shapes(self):
        """build_set_params must sample rfi_orbit_base with shape (n_rfi, 7) inside a NumPyro trace."""
        from tabascal.components.trajectory import SGP4LEOOrbit
        cfg = self._make_config()
        comp = SGP4LEOOrbit()
        comp.setup(cfg)

        set_params = comp.build_set_params()
        with numpyro.handlers.seed(rng_seed=0):
            params = set_params({})

        assert "rfi_orbit_base" in params
        assert params["rfi_orbit_base"].shape == (cfg.n_rfi, 7)

    def test_forward_transform_roundtrip(self):
        """inv_transform(forward_transform(x)) == x."""
        from tabascal.components.trajectory import SGP4LEOOrbit
        cfg = self._make_config()
        comp = SGP4LEOOrbit()
        comp.setup(cfg)

        base = jax.random.normal(jax.random.PRNGKey(0), (cfg.n_rfi, 7))
        transformed = comp.forward_transform(base, comp.L_rfi_orbit, comp.mu_rfi_orbit)
        recovered = comp.inv_transform(transformed, comp.L_rfi_orbit, comp.mu_rfi_orbit)

        assert jnp.allclose(recovered, base, atol=1e-6)

    def test_inv_transform_roundtrip(self):
        """forward_transform(inv_transform(x)) == x."""
        from tabascal.components.trajectory import SGP4LEOOrbit
        cfg = self._make_config()
        comp = SGP4LEOOrbit()
        comp.setup(cfg)

        params = comp.mu_rfi_orbit + jax.random.normal(jax.random.PRNGKey(1), (cfg.n_rfi, 7)) * 0.01
        base = comp.inv_transform(params, comp.L_rfi_orbit, comp.mu_rfi_orbit)
        recovered = comp.forward_transform(base, comp.L_rfi_orbit, comp.mu_rfi_orbit)

        assert jnp.allclose(recovered, params, atol=1e-6)

    # Low-level: sats_init

    def test_sats_init_direct_call(self):
        """sats_init called directly with comp.elements must return without error."""
        from tabascal.components.trajectory import SGP4LEOOrbit
        cfg = self._make_config()
        comp = SGP4LEOOrbit()
        comp.setup(cfg)
        sats = comp.sats_init(comp.elements)
        assert sats is not None

    def test_sats_init_produces_valid_positions(self):
        """sats_init output propagated via sgp4jax must yield finite LEO positions."""
        import sgp4jax
        from tabascal.components.trajectory import SGP4LEOOrbit
        cfg = self._make_config()
        comp = SGP4LEOOrbit()
        comp.setup(cfg)
        sats = comp.sats_init(comp.elements)
        positions, _ = sgp4jax.gcrf_positions_multi_leo(sats, comp.times_jd_fine)
        assert positions.shape == (cfg.n_rfi, cfg.n_time_fine, 3)
        assert jnp.all(jnp.isfinite(positions))
