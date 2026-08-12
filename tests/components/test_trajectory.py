"""Tests for tabascal.components.trajectory — FixedOrbit, PhaseCalculationRFI,
SGP4LEONoDragOrbit, and SGP4LEOOrbit.

TLEs are sourced from IAU CPS SatChecker (no credentials). The SGP4 orbit tests
run fully offline against the bundled TLE cache under tabascal/data/tles/.
"""

import pytest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro

from tabascal.components.trajectory import FixedOrbit, PhaseCalculationRFI
from tabascal.interferometry import get_rfi_phase, get_rfi_phase_numpy

from .conftest import active_precision, make_constants, assert_transform_roundtrip


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
    precision=None,
):
    """Build a minimal mock TabConfig for trajectory components."""
    precision = precision or active_precision()
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
        precision=precision,
        args={"rfi": {"freq_int_samples": n_int_freq}},
    )


# ---------------------------------------------------------------------------
# PhaseCalculationRFI
# ---------------------------------------------------------------------------

@pytest.mark.requires_double
class TestPhaseCalculationRFI:

    def test_setup_validates_dimensions(self):
        """If the config is self-consistent, _validate_dimensions must not raise."""
        cfg = make_trajectory_config(n_ant=4, n_rfi=2, n_freq=3, n_time=6, n_int_time=2)
        comp = PhaseCalculationRFI()
        comp.setup(cfg)
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

    def test_compute_ant_pos_xyz_earth_radius(self):
        """ants_xyz (GCRF) should be at Earth's surface radius (~6.37e6 m)."""
        cfg = make_trajectory_config(n_ant=4)
        comp = PhaseCalculationRFI()
        comp.setup(cfg)
        radii = jnp.linalg.norm(comp.ants_xyz, axis=-1)
        assert jnp.all(radii > 6.35e6), "Antenna radius below Earth surface"
        assert jnp.all(radii < 6.40e6), "Antenna radius too large for ground-based telescope"

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
        out = comp.build_forward()({}, {}, make_constants(comp))
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
        assert comp.build_set_params()(sentinel) is sentinel

    def test_compute_rfi_phase_consistent_with_get_rfi_phase(self):
        """Phase stored at setup must equal get_rfi_phase called with the same arrays."""
        cfg = make_trajectory_config(n_rfi=1, n_ant=4, n_freq=2, n_time=4, n_int_time=2)
        comp = FixedOrbit()
        comp.setup(cfg)
        expected = get_rfi_phase(comp.rfi_xyz, comp.ants_uvw, comp.ants_xyz, comp.freqs_fine)
        assert jnp.allclose(comp.rfi_phase, expected)

    def test_single_precision_uses_numpy_phase(self):
        """Under single precision, rfi_phase is precomputed in numpy and matches get_rfi_phase_numpy."""
        cfg = make_trajectory_config(
            n_rfi=1, n_ant=4, n_freq=2, n_time=4, n_int_time=2, precision="single"
        )
        comp = FixedOrbit()
        comp.setup(cfg)
        assert comp.rfi_phase.shape == (cfg.n_rfi, cfg.n_ant, cfg.n_freq_fine, cfg.n_time_fine)
        assert jnp.all(jnp.isfinite(comp.rfi_phase))
        expected = get_rfi_phase_numpy(
            comp.rfi_xyz, comp.ants_uvw, comp.ants_xyz, comp.freqs_fine
        )
        assert jnp.allclose(comp.rfi_phase, expected)


# ---------------------------------------------------------------------------
# Bundled TLE constants (NAVSTAR 18 / 67, epoch 2023-02-21)
# These match tabascal/data/tles/2023-02-21-navstar.json, which the mock config
# below passes as extra_tle_dir — the highest-precedence source — so both IDs
# resolve locally and no SatChecker call is made.
# ---------------------------------------------------------------------------

# JD of 2023-02-21 13:55:04.589 UTC — must match the date prefix of the bundled file.
_BUNDLED_TLE_EPOCH_JD = 2459997.079914223  # 2023-02-21 13:55:04.589 UTC => GMSA = 0
_BUNDLED_NORAD_IDS = [20452, 38833]


def _make_sgp4_config(n_params, n_ant=4, n_freq=2, n_time=4, n_int_time=2, n_int_freq=1, precision=None):
    """Build a mock TabConfig for SGP4 orbit components using the bundled TLE cache."""
    precision = precision or active_precision()
    from importlib.resources import files as _res_files
    _bundled_tle_dir = str(_res_files("tabascal").joinpath("data/tles"))

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
        elements=jnp.zeros((n_rfi, n_params)),
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
        extra_tle_dir=_bundled_tle_dir,
        precision=precision,
        args={"rfi": {"freq_int_samples": n_int_freq}},
    )


# ---------------------------------------------------------------------------
# Element fetchers — clear error when no TLE resolves
# ---------------------------------------------------------------------------

class TestFetchOrbitalElementsEmpty:
    """When no requested NORAD ID resolves to a TLE, the fetchers must raise a
    clear TLEError naming the IDs — not an opaque pandas KeyError."""

    def _patch_empty(self, monkeypatch):
        import pandas as pd
        from tabascal.components import trajectory as traj_mod
        monkeypatch.setattr(traj_mod, "get_tles_by_id", lambda *a, **k: pd.DataFrame())

    def test_fetch_orbital_elements_raises_tle_error(self, monkeypatch):
        from tabascal.components.trajectory import fetch_orbital_elements
        from tabascal.tle import TLEError
        self._patch_empty(monkeypatch)
        with pytest.raises(TLEError, match=r"No TLEs could be resolved.*99999"):
            fetch_orbital_elements(2460000.0, [99999])

    def test_fetch_standard_orbital_elements_raises_tle_error(self, monkeypatch):
        from tabascal.components.trajectory import fetch_standard_orbital_elements
        from tabascal.tle import TLEError
        self._patch_empty(monkeypatch)
        with pytest.raises(TLEError, match="No TLEs could be resolved"):
            fetch_standard_orbital_elements(2460000.0, [99999])


class TestFetchOrbitalElementsNoSatellites:
    """Configuring no satellites is not a resolution failure.

    ``norad_ids: []`` is the shipped default and TabConfig calls the fetcher
    unconditionally, so a model with no TLE trajectory component must build an
    empty RFI model. Treating the empty request as "no TLEs could be resolved"
    made every satellite-free configuration unrunnable.
    """

    def test_empty_request_yields_an_empty_rfi_model(self):
        from tabascal.components.trajectory import fetch_orbital_elements
        from tabascal.tle import TLEResolution

        resolution = TLEResolution(
            requested=[], obs_epoch_jd=float("nan"), remote_max_age_days=3.0
        )
        elements, epoch_jd, norad_ids, tles, n_rfi_real = fetch_orbital_elements(
            resolution=resolution
        )
        assert elements.shape == (0, 6)
        assert epoch_jd.shape == (0,)
        assert norad_ids == []
        assert tles.shape == (0, 2)
        assert n_rfi_real == 0

    def test_empty_request_without_a_preflight_resolution(self, monkeypatch):
        import pandas as pd
        from tabascal.components import trajectory as traj_mod

        monkeypatch.setattr(traj_mod, "get_tles_by_id", lambda *a, **k: pd.DataFrame())
        _, _, norad_ids, _, n_rfi_real = traj_mod.fetch_orbital_elements(
            2460000.0, []
        )
        assert norad_ids == []
        assert n_rfi_real == 0


class TestFetchOrbitalElementsPartial:
    """A partial resolution must stop the run, naming the excluded NORAD IDs.

    A satellite quietly dropped from the RFI model degrades subtraction with no
    signal in the output, so the element fetchers refuse to build a model from a
    subset of what was configured — even though resolution itself already
    enforces complete coverage upstream.
    """

    def _patch_partial(self, monkeypatch, resolved_ids):
        from tabascal.components import trajectory as traj_mod
        from tabascal.tle import _add_parsed_elements
        from ..tle_helpers import jd, make_catalogue_df

        df = _add_parsed_elements(
            make_catalogue_df([(nid, jd(2023, 2, 21, 13)) for nid in resolved_ids])
        )
        monkeypatch.setattr(traj_mod, "get_tles_by_id", lambda *a, **k: df)

    def test_partial_resolution_raises_and_names_missing_ids(self, monkeypatch):
        from tabascal.components.trajectory import fetch_orbital_elements
        from tabascal.tle import TLEError

        self._patch_partial(monkeypatch, [25544])
        with pytest.raises(TLEError, match=r"NORAD IDs \[99999\]"):
            fetch_orbital_elements(2460000.0, [25544, 99999])

    def test_full_resolution_proceeds(self, monkeypatch):
        from tabascal.components.trajectory import fetch_orbital_elements

        self._patch_partial(monkeypatch, [25544, 38833])
        _, _, norad_ids, _, n_rfi_real = fetch_orbital_elements(
            2460000.0, [25544, 38833]
        )
        assert sorted(int(n) for n in norad_ids) == [25544, 38833]
        assert n_rfi_real == 2

    def test_preflight_resolution_is_reused_without_refetching(self, monkeypatch):
        # The normal path: TabConfig hands the fetchers the resolution preflight
        # already made, so no provider work happens here at all.
        from tabascal.components import trajectory as traj_mod
        from tabascal.components.trajectory import fetch_orbital_elements
        from tabascal import tle
        from ..tle_helpers import jd, make_catalogue_df

        def boom(*args, **kwargs):
            raise AssertionError("the preflight resolution must be reused as-is")

        monkeypatch.setattr(traj_mod, "get_tles_by_id", boom)

        epoch = jd(2023, 2, 21, 13)
        records = make_catalogue_df([(25544, epoch), (38833, epoch)])
        resolution = tle.TLEResolution(
            requested=[25544, 38833],
            obs_epoch_jd=epoch,
            remote_max_age_days=3.0,
            resolved={
                int(row["NORAD_CAT_ID"]): tle.ResolvedTLE(
                    norad_id=int(row["NORAD_CAT_ID"]),
                    record=row.to_dict(),
                    source="managed per-satellite cache",
                    provider="test",
                    epoch_jd=epoch,
                    offset_days=0.0,
                )
                for _, row in records.iterrows()
            },
        )

        _, _, norad_ids, _, n_rfi_real = fetch_orbital_elements(resolution=resolution)
        assert [int(n) for n in norad_ids] == [25544, 38833]
        assert n_rfi_real == 2


# ---------------------------------------------------------------------------
# SGP4LEONoDragOrbit and SGP4LEOOrbit — merged parametrized class
# SGP4LEONoDragOrbit: n_params=6 (bstar excluded from learnable params)
# SGP4LEOOrbit:       n_params=7 (bstar included)
# ---------------------------------------------------------------------------

@pytest.mark.requires_double
@pytest.mark.parametrize("orbit_cls,n_params", [
    pytest.param("SGP4LEONoDragOrbit", 6, id="SGP4LEONoDragOrbit"),
    pytest.param("SGP4LEOOrbit", 7, id="SGP4LEOOrbit"),
])
class TestSGP4LEOOrbit:

    def _get_cls(self, orbit_cls):
        from tabascal.components import trajectory as traj_mod
        return getattr(traj_mod, orbit_cls)

    def test_rfi_xyz_shape(self, orbit_cls, n_params):
        """Initial state_outputs['rfi_xyz'] placeholder has shape (n_rfi, n_time_fine, 3)."""
        cls = self._get_cls(orbit_cls)
        cfg = _make_sgp4_config(n_params)
        comp = cls()
        comp.setup(cfg)
        assert comp.state_outputs["rfi_xyz"].shape == (cfg.n_rfi, cfg.n_time_fine, 3)

    def test_init_params_base_shape(self, orbit_cls, n_params):
        """Initial base orbit parameters have shape (n_rfi, n_params)."""
        cls = self._get_cls(orbit_cls)
        cfg = _make_sgp4_config(n_params)
        comp = cls()
        comp.setup(cfg)
        assert comp.init_params_base["rfi_orbit_base"].shape == (cfg.n_rfi, n_params)

    def test_prior_covariance_positive_definite(self, orbit_cls, n_params):
        """L_rfi_orbit must be lower-triangular with positive diagonal."""
        cls = self._get_cls(orbit_cls)
        cfg = _make_sgp4_config(n_params)
        comp = cls()
        comp.setup(cfg)
        for i in range(cfg.n_rfi):
            diag = jnp.diag(comp.L_rfi_orbit[i])
            assert jnp.all(diag > 0), f"Cholesky diagonal not positive for satellite {i}"


# ---------------------------------------------------------------------------
# Component.require_double — the shared double-precision gate
# ---------------------------------------------------------------------------

def test_require_double_gate():
    """``Component.require_double`` raises for a ``requires_double`` component run
    in any non-double precision, and is a no-op otherwise.

    Reads ``config.precision`` / the ``requires_double`` flag (not the live
    ``jax_enable_x64``), so it behaves the same in either test precision. This
    exercises the raise path that the ``requires_double``-marked component tests
    skip.
    """
    from tabascal.components import Component

    class _NeedsDouble(Component):
        requires_double = True

        def setup(self, config):  # pragma: no cover - not called
            ...

        def build_forward(self):  # pragma: no cover - not called
            return lambda params, state, constants: state

    class _AnyPrecision(_NeedsDouble):
        requires_double = False

    comp = _NeedsDouble()
    comp.require_double(SimpleNamespace(precision="double"))  # must not raise
    for bad in ("single", "half", ""):
        with pytest.raises(ValueError, match="requires double precision"):
            comp.require_double(SimpleNamespace(precision=bad))

    # A component that does not require double is never gated.
    _AnyPrecision().require_double(SimpleNamespace(precision="single"))

    def test_forward_output_shapes(self, orbit_cls, n_params):
        """Forward pass produces rfi_xyz (n_rfi, n_time_fine, 3) and elements (n_rfi, n_params)."""
        cls = self._get_cls(orbit_cls)
        cfg = _make_sgp4_config(n_params)
        comp = cls()
        comp.setup(cfg)

        params = {"rfi_orbit_base": comp.init_params_base["rfi_orbit_base"]}
        out = comp.build_forward()(params, {}, make_constants(comp))

        assert out["rfi_xyz"].shape == (cfg.n_rfi, cfg.n_time_fine, 3)
        assert out["elements"].shape == (cfg.n_rfi, n_params)

    def test_forward_rfi_xyz_finite(self, orbit_cls, n_params):
        """SGP4-propagated satellite positions from the forward pass are all finite."""
        cls = self._get_cls(orbit_cls)
        cfg = _make_sgp4_config(n_params)
        comp = cls()
        comp.setup(cfg)

        params = {"rfi_orbit_base": comp.init_params_base["rfi_orbit_base"]}
        out = comp.build_forward()(params, {}, make_constants(comp))

        assert jnp.all(jnp.isfinite(out["rfi_xyz"]))

    def test_build_set_params_samples_correct_shapes(self, orbit_cls, n_params):
        """build_set_params must sample rfi_orbit_base with shape (n_rfi, n_params) inside a NumPyro trace."""
        cls = self._get_cls(orbit_cls)
        cfg = _make_sgp4_config(n_params)
        comp = cls()
        comp.setup(cfg)

        set_params = comp.build_set_params()
        with numpyro.handlers.seed(rng_seed=0):
            params = set_params({})

        assert "rfi_orbit_base" in params
        assert params["rfi_orbit_base"].shape == (cfg.n_rfi, n_params)

    def test_forward_transform_roundtrips(self, orbit_cls, n_params):
        """inv_transform(forward_transform(x)) == x and forward_transform(inv_transform(x)) == x."""
        cls = self._get_cls(orbit_cls)
        cfg = _make_sgp4_config(n_params)
        comp = cls()
        comp.setup(cfg)

        base = jax.random.normal(jax.random.PRNGKey(0), (cfg.n_rfi, n_params))
        assert_transform_roundtrip(comp, base, comp.L_rfi_orbit, comp.mu_rfi_orbit)

    def test_sats_init_produces_valid_positions(self, orbit_cls, n_params):
        """sats_init output propagated via sgp4jax must yield finite LEO positions."""
        import sgp4jax
        cls = self._get_cls(orbit_cls)
        cfg = _make_sgp4_config(n_params)
        comp = cls()
        comp.setup(cfg)
        sats = comp.sats_init(comp.elements)
        positions, _ = sgp4jax.gcrf_positions_multi_leo(sats, comp.times_jd_fine)
        assert positions.shape == (cfg.n_rfi, cfg.n_time_fine, 3)
        assert jnp.all(jnp.isfinite(positions))
