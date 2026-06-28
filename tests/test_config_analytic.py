"""Tests for the analytic RFI-visibility config wiring in TabConfig.

setup_analytic_sampling sizes K and builds the per-sub-window fringe parameters;
_set_freqs_times builds the interleaved edge/centre fine grid. These run host-side and
depend on the satellite-trajectory / ECI-position helpers, which are monkeypatched here
to synthetic geometry so the test needs no network or TLEs. The checks pin the grid
layout (edges at even indices, centres at odd), the edge-gather indexing, and the
consistency of the fringe-parameter shapes that the AnalyticVisCalculation component
consumes.
"""

import numpy as np
import pytest

import tabascal.config as config_mod
from tabascal.config import TabConfig
from tabascal.time import mjd_to_jd

pytestmark = pytest.mark.requires_double

N_ANT = 5
N_TIME = 6
INT_TIME = 2.0
FREQ = 1.5e9
T0_MJD = 51544.5

_BASE = np.array([5109360.0, 2006852.0, -3238948.0])
_rng = np.random.default_rng(3)
_ANTS_ITRF = _BASE + np.concatenate([np.zeros((1, 3)), _rng.uniform(-4000, 4000, (N_ANT - 1, 3))])


def _make_config(monkeypatch, **rfi_overrides):
    """Build a bare TabConfig with just the attributes the analytic setup needs, and
    monkeypatch the trajectory/position helpers to synthetic (LEO-like) geometry."""
    cfg = TabConfig.__new__(TabConfig)
    cfg.phase_centre = {"ra": 120.0, "dec": -30.0}
    cfg.freqs = np.array([FREQ])
    cfg.chan_width = 1e6
    cfg.n_freq = 1
    cfg.n_time = N_TIME
    cfg.int_time = INT_TIME
    times = np.arange(N_TIME) * INT_TIME
    cfg.times = times
    cfg.times_jd = mjd_to_jd(T0_MJD + times / 86400.0)
    cfg.ants_itrf = _ANTS_ITRF
    a1, a2 = np.triu_indices(N_ANT, 1)
    cfg.a1, cfg.a2 = a1, a2
    cfg.n_bl = len(a1)
    cfg.tles = [("dummy1", "dummy1b")]  # not used (positions monkeypatched)
    cfg.n_int_freq = 1
    cfg.n_int_time = 30
    cfg.vis_method = "analytic"
    cfg.args = {"rfi": {"freq_pad_factor": 2, "time_pad_factor": 2}}

    r0 = _BASE / np.linalg.norm(_BASE) * (np.linalg.norm(_BASE) + 550e3)
    v = np.array([7500.0, 0.0, 0.0])

    def fake_sat(tles, times_jd):
        t = (np.asarray(times_jd) - cfg.times_jd[0]) * 86400.0
        return (r0[None, :] + v[None, :] * t[:, None])[None]  # (1, n, 3)

    def fake_ants_eci(ants_itrf, times_jd):
        # slow ECI rotation of the ITRF antennas (n_ant, n, 3)
        t = (np.asarray(times_jd) - cfg.times_jd[0]) * 86400.0
        th = 7.292e-5 * t
        cz, sz = np.cos(th), np.sin(th)
        ax = ants_itrf[:, 0][:, None] * cz[None, :] - ants_itrf[:, 1][:, None] * sz[None, :]
        ay = ants_itrf[:, 0][:, None] * sz[None, :] + ants_itrf[:, 1][:, None] * cz[None, :]
        az = np.repeat(ants_itrf[:, 2][:, None], len(t), axis=1)
        return np.stack([ax, ay, az], axis=-1)

    monkeypatch.setattr(config_mod, "get_satellite_positions", fake_sat)
    monkeypatch.setattr(config_mod, "itrs_to_gcrs_sf", fake_ants_eci)

    rfi_config = {"vis_method": "analytic", **rfi_overrides}
    cfg.setup_analytic_sampling(rfi_config)
    cfg._set_freqs_times()
    return cfg


def test_analytic_setup_shapes(monkeypatch):
    cfg = _make_config(monkeypatch)
    K = cfg.analytic_K
    assert K >= 1
    assert cfg.analytic_f.shape == (1, cfg.n_bl, N_TIME * K)
    assert cfg.analytic_fdot.shape == (1, cfg.n_bl, N_TIME * K)
    assert cfg.analytic_edge_gather.shape == (N_TIME, K + 1)
    assert cfg.analytic_dt_sub == pytest.approx(INT_TIME / K)


def test_interleaved_grid_layout(monkeypatch):
    cfg = _make_config(monkeypatch)
    K = cfg.analytic_K
    assert cfg.n_time_fine == 2 * N_TIME * K + 1
    edges = cfg.times_fine[0::2]
    centres = cfg.times_fine[1::2]
    assert len(edges) == N_TIME * K + 1
    assert len(centres) == N_TIME * K
    # uniform sub-window spacing; centres sit at edge midpoints.
    dt_sub = cfg.analytic_dt_sub
    np.testing.assert_allclose(np.diff(edges), dt_sub, rtol=1e-9)
    np.testing.assert_allclose(centres, 0.5 * (edges[1:] + edges[:-1]), rtol=1e-9, atol=1e-9)


def test_edge_gather_indexes_shared_edges(monkeypatch):
    cfg = _make_config(monkeypatch)
    K = cfg.analytic_K
    eg = cfg.analytic_edge_gather
    # window i uses edges [iK .. iK+K]; right edge of i == left edge of i+1 (shared).
    assert eg[0, 0] == 0 and eg[-1, -1] == N_TIME * K
    for i in range(N_TIME - 1):
        assert eg[i, -1] == eg[i + 1, 0]


def test_K_tightens_with_tolerance(monkeypatch):
    cfg_loose = _make_config(monkeypatch, resid_tol=1e-2)
    cfg_tight = _make_config(monkeypatch, resid_tol=1e-6)
    assert cfg_tight.analytic_K >= cfg_loose.analytic_K


def test_rejects_freq_subintegration(monkeypatch):
    with pytest.raises(ValueError, match="n_int_freq"):
        # n_int_freq != 1 must be rejected (time-only path).
        cfg = TabConfig.__new__(TabConfig)
        cfg.n_int_freq = 2
        cfg.setup_analytic_sampling({"vis_method": "analytic"})
