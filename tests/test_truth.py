"""Tests for tabascal.truth -- the unified tab-sim ground-truth loader.

Builds a tiny in-memory simulation ``.zarr`` (no MS, no network) and checks that truth
discovery, the fail-fast preflight, and the aligned loader behave as expected, plus the
RMSE reporting helpers in tab_tools.
"""

from types import SimpleNamespace

import numpy as np
import xarray as xr
import jax.numpy as jnp
import pytest

from tabascal import truth
from tabascal.truth import (
    TruthError,
    available_truth,
    require_truth,
    load_truth,
    read_true_vis_ast,
    has_truth,
)
from tabascal.tab_tools import rmse, print_truth_metrics


N_TIME, N_BL, N_FREQ, N_ANT = 4, 3, 2, 5


def _write_sim_zarr(path, *, vis_ast=True, vis_rfi=True, gains=False):
    """Write a minimal sim zarr. Variables stored (n_time, n_bl, n_freq) like tab-sim."""
    rng = np.random.default_rng(0)
    data = {}
    if vis_ast:
        data["vis_ast"] = (("time", "bl", "freq"), rng.standard_normal((N_TIME, N_BL, N_FREQ)) + 1j)
    if vis_rfi:
        data["vis_rfi"] = (("time", "bl", "freq"), rng.standard_normal((N_TIME, N_BL, N_FREQ)) + 2j)
    if gains:
        data["gains_ants"] = (("time", "ant", "freq"), rng.standard_normal((N_TIME, N_ANT, N_FREQ)) + 1j)
    xr.Dataset(data).to_zarr(path, mode="w")
    return str(path)


def _config(zarr_path, **overrides):
    cfg = {
        "data": {"zarr_path": zarr_path, "data_col": "DATA"},
        "ast": {"init": "sample"},
        "rfi": {"init": "sample"},
        "plots": {"truth": False},
    }
    for section, vals in overrides.items():
        cfg.setdefault(section, {}).update(vals)
    return cfg


def _tab_config(zarr_path):
    return SimpleNamespace(
        n_bl=N_BL, n_freq=N_FREQ, n_time=N_TIME, n_ant=N_ANT,
        noise=1.0, flags=jnp.zeros((N_BL, N_FREQ, N_TIME), dtype=bool),
        args=_config(zarr_path),
    )


def test_available_truth_reports_present_variables(tmp_path):
    zp = _write_sim_zarr(tmp_path / "sim.zarr", vis_ast=True, vis_rfi=True, gains=True)
    have = available_truth(_config(zp))
    assert have == {"vis_ast": True, "vis_rfi": True, "gains": True, "rfi_A": False}


def test_available_truth_missing_zarr_is_all_false():
    have = available_truth(_config("/nonexistent/sim.zarr"))
    assert have == {"vis_ast": False, "vis_rfi": False, "gains": False, "rfi_A": False}


def test_read_true_vis_ast_is_baseline_freq_time(tmp_path):
    zp = _write_sim_zarr(tmp_path / "sim.zarr")
    vis = read_true_vis_ast(zp)
    assert vis.shape == (N_BL, N_FREQ, N_TIME)


def test_read_true_vis_ast_zeroed_when_data_col_excludes_ast(tmp_path):
    zp = _write_sim_zarr(tmp_path / "sim.zarr")
    # RFI_DATA contains rfi but not ast -> ast truth zeroed for init-at-truth.
    vis = read_true_vis_ast(zp, data_col="RFI_DATA")
    assert jnp.all(vis == 0)


def test_load_truth_fills_missing_with_nan(tmp_path):
    zp = _write_sim_zarr(tmp_path / "sim.zarr", vis_ast=True, vis_rfi=False, gains=False)
    tc = _tab_config(zp)
    t = load_truth(tc)
    assert t["vis_ast"].shape == (N_BL, N_FREQ, N_TIME)
    assert not jnp.any(jnp.isnan(t["vis_ast"]))
    assert jnp.all(jnp.isnan(t["vis_rfi"]))   # absent -> NaN placeholder
    assert jnp.all(jnp.isnan(t["gains"]))
    assert has_truth(t)


def test_load_truth_no_zarr_returns_all_nan():
    tc = _tab_config("/nonexistent/sim.zarr")
    t = load_truth(tc)
    assert set(t) == {"vis_ast", "vis_rfi", "gains"}
    assert all(jnp.all(jnp.isnan(v)) for v in t.values())
    assert not has_truth(t)


def test_require_truth_noop_when_not_requested(tmp_path):
    zp = _write_sim_zarr(tmp_path / "sim.zarr")
    require_truth(_config(zp))  # ast/rfi init=sample, plots.truth False -> no requirement


def test_require_truth_raises_when_zarr_absent():
    cfg = _config("/nonexistent/sim.zarr", ast={"init": "truth"})
    with pytest.raises(TruthError, match="ast.init: truth"):
        require_truth(cfg)


def test_require_truth_raises_when_required_var_missing(tmp_path):
    zp = _write_sim_zarr(tmp_path / "sim.zarr", vis_ast=False, vis_rfi=True)
    cfg = _config(zp, ast={"init": "truth"})
    with pytest.raises(TruthError, match="vis_ast"):
        require_truth(cfg)


def test_require_truth_passes_when_required_var_present(tmp_path):
    zp = _write_sim_zarr(tmp_path / "sim.zarr", vis_ast=True)
    require_truth(_config(zp, ast={"init": "truth"}))


def test_rmse_zero_for_identical_and_flag_masked():
    a = jnp.array([1.0 + 1j, 2.0, 3.0])
    assert float(rmse(a, a)) == pytest.approx(0.0)
    # flag masking matches reduced_chi2's ~flags convention
    pred = jnp.array([1.0, 99.0, 3.0])
    true = jnp.array([1.0, 0.0, 3.0])
    flags = jnp.array([False, True, False])
    assert float(rmse(pred, true, flags)) == pytest.approx(0.0)


def test_print_truth_metrics_dynamic(tmp_path, capsys):
    zp = _write_sim_zarr(tmp_path / "sim.zarr", vis_ast=True, vis_rfi=False)
    tc = _tab_config(zp)
    t = load_truth(tc)
    pred = {
        "vis_ast": load_truth(tc)["vis_ast"][None],   # perfect prediction -> RMSE 0
        "vis_rfi": jnp.zeros((1, N_BL, N_FREQ, N_TIME), dtype=complex),
        "gains": jnp.zeros((1, N_ANT, N_FREQ, N_TIME), dtype=complex),
    }
    print_truth_metrics(pred, t, tc, "init")
    out = capsys.readouterr().out
    assert "Truth metrics @ init params" in out
    assert "Ast. Vis" in out          # available
    assert "RFI Vis" not in out       # NaN truth -> skipped (dynamic)
    assert "Gains" not in out
