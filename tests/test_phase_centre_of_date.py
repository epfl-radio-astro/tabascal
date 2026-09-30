"""Phase tracking must follow the J2000 phase centre to the date of observation (#252)."""

import jax.numpy as jnp
import numpy as np
import pytest

from tabascal import rfi_estimate
from tabascal.components import trajectory
from tabascal.components.trajectory import PHASE_TRACKING, FixedOrbitFine, PhaseCalculationRFIFine
from tabascal.interferometry import C
from tests.components.conftest import make_constants
from tests.components.test_trajectory import make_trajectory_config

R = 1e12  # far field: curvature across the array is ~1e-5 m
F = 1e6  # 300 m wavelength, so the wrapped phase is unambiguous
LAM = C / F
PC = {"ra": 0.0, "dec": -20.0}  # J2000, as read from a Measurement Set
TIMES = 2440587.5 + (1568538661.0 + np.array([0.0, 300.0, 600.0])) / 86400.0  # 2019-09-15


def _ants():
    """MeerKAT m000 plus tangent-plane offsets spanning ~7 km."""
    r0 = np.array([5109360.133, 2006852.586, -3238948.127])
    up = r0 / np.linalg.norm(r0)
    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    en = [(0, 0), (40, 100), (-300, 500), (1200, -800), (-2500, -1500), (3000, 2500)]
    return np.array([r0 + e * east + n * north for e, n in en])


ANTS = _ants()
BASELINE = np.linalg.norm(ANTS - ANTS[0], axis=-1)[:, None]  # (n_ant, 1)


def _unit(ra, dec):
    ra, dec = np.deg2rad(ra), np.deg2rad(dec)
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def _phase_to_path(phase):
    cyc = -(phase - phase[:1]) / (2 * np.pi)
    return (cyc - np.round(cyc)) * LAM


def _config(pc):
    cfg = make_trajectory_config(n_ant=len(ANTS), n_freq=1, n_time=len(TIMES), n_int_time=1)
    cfg.times_jd_fine, cfg.ants_itrf, cfg.freqs_fine, cfg.phase_centre = TIMES, ANTS, np.array([F]), pc
    return cfg


def _sat(t, s):
    """Far along ``s`` from the reference antenna, so geocentric parallax cancels."""
    return (trajectory.itrs_to_gcrs_sf(ANTS[:1], np.asarray(t))[0] + R * s)[None]


def _fixed_sat(monkeypatch, module, s):
    monkeypatch.setattr(module, "get_satellite_positions", lambda recs, t: _sat(t, s))


def _positions(monkeypatch, s, pc):
    phase = rfi_estimate.rfi_phase_from_positions(_sat(TIMES, s), ANTS, TIMES, pc, [F])
    return _phase_to_path(phase[0, :, 0])


def _path_grid(monkeypatch, s, pc):
    _fixed_sat(monkeypatch, rfi_estimate, s)
    n = len(ANTS)
    out = rfi_estimate.near_field_baseline_paths(
        None, ANTS, TIMES, pc, np.arange(n), np.zeros(n, int), n_fine=2, delta_t=8.0
    )
    return out[0].reshape(n, -1)


def _fixed_orbit(monkeypatch, s, pc):
    _fixed_sat(monkeypatch, trajectory, s)
    comp = FixedOrbitFine()
    comp.setup(_config(pc))
    return _phase_to_path(np.asarray(comp.rfi_phase)[0, :, 0])


def _phase_calc(monkeypatch, s, pc):
    comp = PhaseCalculationRFIFine()
    comp.setup(_config(pc))
    out = comp.build_forward()({}, {"rfi_xyz": jnp.asarray(_sat(TIMES, s))}, make_constants(comp))
    return _phase_to_path(np.asarray(out["rfi_phase"])[0, :, 0])


SITES = pytest.mark.parametrize("site", [
    pytest.param(_positions, id="rfi_phase_from_positions"),
    pytest.param(_path_grid, id="antenna_path_grid"),
    pytest.param(_fixed_orbit, id="FixedOrbitFine"),
    pytest.param(_phase_calc, id="PhaseCalculationRFIFine", marks=pytest.mark.requires_double),
])


def _assert_tracked(resid, per_metre, label):
    worst = np.max(np.abs(resid[1:]) / BASELINE[1:])
    assert np.all(np.abs(resid) <= per_metre * BASELINE + 1e-3), (
        f"{label}: worst residual {worst * 1e3:.4f} mm/m (max {np.max(np.abs(resid)):.4f} m), "
        f"allowed {per_metre * 1e3:.4f} mm/m"
    )


@SITES
def test_phase_centre_source_is_fringe_stopped(site, monkeypatch):
    """A far source at the J2000 phase centre has a constant path across the array."""
    resid = site(monkeypatch, _unit(PC["ra"], PC["dec"]), dict(PC))
    _assert_tracked(resid, 0.25e-3, "J2000 direction")


@SITES
def test_tracking_includes_annual_aberration(site, monkeypatch):
    """The tracked direction is the aberrated (apparent) J2000 phase centre."""
    erfa = pytest.importorskip("erfa")
    pvh, pvb = erfa.epv00(TIMES[1], 0.0)
    v = pvb[1] / erfa.DC
    s_app = erfa.ab(_unit(PC["ra"], PC["dec"]), v, np.linalg.norm(pvh[0]), np.sqrt(1 - v @ v))
    resid = site(monkeypatch, s_app, dict(PC))
    _assert_tracked(resid, np.deg2rad(0.5 / 3600), "aberrated direction")


@SITES
def test_phase_centre_argument_stays_j2000(site, monkeypatch):
    """Callers keep passing J2000 and the dict they pass is left untouched."""
    pc = dict(PC)
    site(monkeypatch, _unit(PC["ra"], PC["dec"]), pc)
    assert pc == PC


def test_tabsim_phase_tracking_selects_the_j2000_convention():
    """``tabsim.phase_tracking: j2000`` is the old GAST - RA_J2000 term exactly (#253)."""
    from tabascal.config import normalise_tabsim_config
    from tabascal.interferometry import itrf_to_uvw_numpy
    from tabascal.time import gast_deg

    old = np.transpose(
        itrf_to_uvw_numpy(ANTS, (gast_deg(TIMES) - PC["ra"]) % 360, PC["dec"]), axes=(1, 0, 2)
    )
    legacy = normalise_tabsim_config({"tabsim": {"phase_tracking": "J2000"}})
    default = normalise_tabsim_config({"tabsim": None})

    np.testing.assert_array_equal(
        trajectory.tracking_uvw(ANTS, TIMES, {**PC, "tracking": legacy["phase_tracking"]}), old
    )
    assert default["phase_tracking"] == "apparent"
    assert np.abs(trajectory.tracking_uvw(ANTS, TIMES, PC) - old).max() > 1.0


@pytest.mark.parametrize("config, want", [
    ({}, "apparent"),
    ({"tabsim": None}, "apparent"),
    ({"tabsim": {}}, "apparent"),
    ({"tabsim": {"phase_tracking": "J2000"}}, "j2000"),
    *[({"tabsim": v}, "mapping") for v in (False, 0, "", [])],
    *[({"tabsim": {"phase_tracking": v}}, "must be one of") for v in (None, False, 0, "", [], "icrs")],
    ({"tabsim": {"phase_traking": "j2000"}}, "no key"),
])
def test_the_tabsim_section_is_validated_as_given(config, want):
    """Only an absent or null section or key defaults; anything else is checked."""
    from tabascal.config import normalise_tabsim_config

    if want in PHASE_TRACKING:
        assert normalise_tabsim_config(config)["phase_tracking"] == want
    else:
        with pytest.raises(ValueError, match=want):
            normalise_tabsim_config(config)
