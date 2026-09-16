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
from tabascal.tab_tools import (
    rmse,
    print_truth_metrics,
    _effective_sample_size,
    _integrated_autocorr_time,
)


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
        noise=1.0, noise_scalar=1.0,
        flags=jnp.zeros((N_BL, N_FREQ, N_TIME), dtype=bool),
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


def test_effective_sample_size_tracks_correlation():
    """N_eff deflates from N as the residual becomes more correlated.

    The bias significance divides by sqrt(N_eff), so this is what stops a strongly
    time-correlated residual from looking like a real bias. Three regimes:
    uncorrelated ~ N, constant-in-time ~ N / n_time, fully constant -> 1.
    """
    rng = np.random.default_rng(0)
    n_row, n_freq, n_time = 20, 1, 200
    N = n_row * n_freq * n_time

    white = (rng.standard_normal((n_row, n_freq, n_time))
             + 1j * rng.standard_normal((n_row, n_freq, n_time)))
    n_white = _effective_sample_size(white)
    assert n_white > 0.4 * N            # ~independent -> close to N

    # Same value repeated along time (rows still independent) -> deflated by ~n_time.
    per_row = (rng.standard_normal((n_row, n_freq, 1))
               + 1j * rng.standard_normal((n_row, n_freq, 1)))
    constant_in_time = np.broadcast_to(per_row, (n_row, n_freq, n_time)).copy()
    n_corr = _effective_sample_size(constant_in_time)
    assert n_corr < n_white / 10        # strongly deflated
    assert 5 < n_corr < 80              # ~ n_row (= N / n_time)

    # A single constant offset everywhere is fully coherent -> one effective sample.
    assert _effective_sample_size(np.full((n_row, n_freq, n_time), 1.0 + 1j)) == 1.0


def test_effective_sample_size_matches_the_explicit_gram():
    """The column-sum shortcut equals the full correlation matrix it replaces.

    ``N_eff_row`` wants only the *total* of the row correlation matrix, and that
    total factors through the column sums, so the matrix is never formed. It would
    be n_bl x n_bl -- tens of gigabytes at the sizes this runs at -- so the identity
    is what makes the metric affordable. Check it against the explicit Gram at a
    size where forming one is still cheap.
    """
    rng = np.random.default_rng(3)
    for shape in ((37, 3, 11), (9, 1, 5), (64, 2, 2)):
        y = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
        n_row = shape[0]

        # Reproduce the pre-image of the shortcut exactly as the function builds it.
        yc = y - y.mean()
        yr = yc.reshape(n_row, -1)
        yr = yr - yr.mean(axis=1, keepdims=True)
        nrm = np.sqrt(np.sum(np.abs(yr) ** 2, axis=1))
        good = nrm > 0
        yg = yr[good] / nrm[good][:, None]

        explicit = (yg @ yg.conj().T).real.sum()
        shortcut = float(np.sum(np.abs(yg.sum(axis=0)) ** 2))
        assert np.isclose(shortcut, explicit, rtol=1e-9, atol=0)

        # And through the public function, against a reference built the old way.
        neff_row = good.sum() ** 2 / explicit
        expected = min(
            max(
                (shape[2] / _integrated_autocorr_time(yc, 2))
                * (shape[1] / _integrated_autocorr_time(yc, 1))
                * neff_row,
                1.0,
            ),
            y.size,
        )
        assert np.isclose(_effective_sample_size(y), expected, rtol=1e-9, atol=0)


def test_effective_sample_size_rows_that_cancel_count_as_independent():
    """Rows that cancel exactly leave nothing to divide the row count by.

    With rows ``r`` and ``-r`` the column sums of the normalised rows are zero, so
    ``n_row^2 / total`` has no finite value. The guard counts the good rows as
    independent instead, so the time and frequency deflation still applies rather
    than the result jumping to the ``N`` cap.
    """
    rng = np.random.default_rng(5)
    # Integer-valued with zero mean, so every centring step is exact and the two
    # normalised rows are exact negatives: the total is zero, not merely small.
    # A walk rather than white steps, so the time axis carries correlation and the
    # result sits below the N cap -- where dividing by zero would have landed it.
    walk = np.cumsum(rng.integers(-2, 3, size=(2, 20)).astype(float), axis=1)
    parts = np.concatenate([walk, -walk[:, ::-1]], axis=1)
    series = (parts[0] + 1j * parts[1])[None, None, :]
    y = np.concatenate([series, -series])

    tau_time = _integrated_autocorr_time(y, 2)
    assert tau_time > 1
    expected = (y.shape[2] / tau_time) * 1.0 * 2
    neff = _effective_sample_size(y)
    assert neff < y.size
    assert np.isclose(neff, expected, rtol=1e-12, atol=0)


def test_autocorr_matches_explicit_lags():
    """FFT lags preserve the overlap normalisation and first-negative window."""
    rng = np.random.default_rng(42)
    for shape in ((7, 3, 11), (2, 5, 150), (4, 1, 3)):
        for complex_data in (False, True):
            arr = rng.normal(size=shape)
            if complex_data:
                arr = arr + 1j * rng.normal(size=shape)
            for data in (arr, np.cumsum(arr, axis=-1), np.ones(shape), np.zeros(shape)):
                for axis in range(3):
                    n = shape[axis]
                    m = np.moveaxis(data, axis, -1).reshape(-1, n)
                    g0 = np.mean(np.sum(np.abs(m) ** 2, axis=1))
                    expected = 1.0
                    if n >= 4 and g0 > 0:
                        for k in range(1, n):
                            rho = np.mean(np.sum(m[:, :n-k] * np.conj(m[:, k:]), axis=1).real) / g0
                            if rho <= 0:
                                break
                            expected += 2 * (1 - k / n) * rho
                    np.testing.assert_allclose(_integrated_autocorr_time(data, axis), expected, rtol=1e-12)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_autocorr_matches_long_correlated_window(dtype):
    """Correlated residuals exercise many lags before the Sokal window closes.

    White noise can stop at lag one, hiding the cost this FFT replaces. A smooth
    complex oscillation has zero time mean and stays positively correlated for
    roughly a quarter of the 150-sample series, in either production precision.
    """
    n = 150
    rng = np.random.default_rng(17)
    amplitude = rng.normal(size=(4, 3, 1)) + 1j * rng.normal(size=(4, 3, 1))
    residual = (amplitude * np.exp(2j * np.pi * np.arange(n) / n)).astype(dtype)
    residual -= residual.mean()
    m = residual.reshape(-1, n)
    g0 = np.mean(np.sum(np.abs(m) ** 2, axis=1))
    expected = 1.0
    positive_lags = 0
    for k in range(1, n):
        rho = np.mean(np.sum(m[:, :n-k] * np.conj(m[:, k:]), axis=1).real) / g0
        if rho <= 0:
            break
        positive_lags += 1
        expected += 2 * (1 - k / n) * rho

    # Pin the long window as well as the result, so this cannot silently become
    # another early-exit white-noise test when the residual fixture changes.
    assert n // 5 < positive_lags < n // 3
    assert expected > 20
    tolerance = 1e-6 if dtype == np.complex64 else 1e-12
    np.testing.assert_allclose(
        _integrated_autocorr_time(residual, axis=2), expected, rtol=tolerance,
    )


def test_autocorr_single_precision_does_not_drift_with_row_count():
    """The lag power is averaged over every row, and float32 drifts with the count.

    Production averages over ~1e6 baseline-channel rows, where a float32 running
    sum moves tau by ~1e-4 and with it the printed N_eff. Accumulating in float64
    keeps complex64 input within float32 rounding of the same series in complex128.
    """
    rng = np.random.default_rng(11)
    shape = (200_000, 1, 8)
    series = np.cumsum(rng.standard_normal(shape) + 1j * rng.standard_normal(shape), axis=-1)
    reference = _integrated_autocorr_time(series, axis=2)
    single = _integrated_autocorr_time(series.astype(np.complex64), axis=2)
    np.testing.assert_allclose(single, reference, rtol=1e-6)
