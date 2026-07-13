"""Tests for the CASA-compatible calibration tables in :mod:`tabascal.caltable`.

The end-to-end check that CASA's ``applycal`` accepts these tables (and reproduces
``V / (g_p conj(g_q))`` from them) needs an MS and casatasks, so it lives with the
pipeline verification. What is locked down here is the format CASA keys off and the
gain convention, which is what silently breaks if either is changed.
"""

import numpy as np
import pytest

from casacore.tables import table

from tabascal.caltable import (
    apply_gains_to_data,
    baseline_gains,
    match_gains_to_grid,
    read_caltable,
    write_caltable,
)


@pytest.fixture
def gains():
    rng = np.random.default_rng(0)
    n_ant, n_freq, n_time = 6, 4, 3
    amp = rng.uniform(0.3, 3.0, (n_ant, n_freq, n_time))
    phase = rng.uniform(-np.pi, np.pi, (n_ant, n_freq, n_time))
    return (amp * np.exp(1j * phase)).astype(complex)


def test_roundtrip(tmp_path, gains):
    path = str(tmp_path / "test.B")
    times = np.array([1.0, 2.0, 3.0]) * 1e9
    write_caltable(path, gains, times, ms_path=str(tmp_path / "fake.ms"))

    out = read_caltable(path)
    assert np.allclose(out["gains"], gains, rtol=1e-5, atol=1e-6)
    assert np.allclose(out["times"], times)
    assert out["viscal"] == "B Jones"


def test_casa_format(tmp_path, gains):
    """CASA identifies a caltable by its INFO record; applycal rejects it otherwise."""
    path = str(tmp_path / "test.B")
    times = np.arange(gains.shape[2], dtype=float)
    write_caltable(path, gains, times, ms_path=str(tmp_path / "fake.ms"))

    tb = table(path, ack=False)
    assert tb.info()["type"] == "Calibration"
    assert tb.info()["subType"] == "B Jones"
    assert tb.getkeyword("ParType") == "Complex"

    n_ant, n_freq, n_time = gains.shape
    assert tb.nrows() == n_ant * n_time
    # CASA writes 2 pols even for a single-correlation MS.
    assert tb.getcell("CPARAM", 0).shape == (n_freq, 2)
    # One row per (time, antenna), with no second antenna.
    assert set(np.unique(tb.getcol("ANTENNA2"))) == {-1}
    assert sorted(np.unique(tb.getcol("ANTENNA1"))) == list(range(n_ant))
    tb.close()


def test_flagged_gains_roundtrip_as_nan(tmp_path, gains):
    g = gains.copy()
    g[2, :, 1] = 0.0                      # a dead antenna at one time
    path = str(tmp_path / "flagged.B")
    write_caltable(path, g, np.arange(g.shape[2], dtype=float),
                   ms_path=str(tmp_path / "fake.ms"))

    out = read_caltable(path)["gains"]
    assert np.all(np.isnan(out[2, :, 1]))
    assert np.allclose(out[0], g[0], rtol=1e-5, atol=1e-6)


def test_gain_convention(gains):
    """V_obs = g_p conj(g_q) V_true, so calibrating divides that out exactly."""
    n_ant, n_freq, n_time = gains.shape
    a1 = np.array([0, 0, 1, 2])
    a2 = np.array([1, 2, 2, 3])

    rng = np.random.default_rng(1)
    vis_true = (rng.normal(size=(len(a1), n_freq, n_time))
                + 1j * rng.normal(size=(len(a1), n_freq, n_time)))

    g_bl = baseline_gains(gains, a1, a2)
    vis_obs = g_bl * vis_true                       # forward: corrupt with the gains
    vis_cal, _ = apply_gains_to_data(vis_obs, gains, a1, a2)

    assert np.allclose(vis_cal, vis_true, rtol=1e-8, atol=1e-10)


def test_noise_and_weight_transform(gains):
    """sigma follows the data: sigma_cal = sigma / |g|, i.e. weight_cal = weight |g|^2."""
    a1, a2 = np.array([0, 1]), np.array([1, 2])
    sigma = np.array([2.0, 5.0])[:, None, None]

    g_bl = baseline_gains(gains, a1, a2)
    _, sigma_cal = apply_gains_to_data(np.ones((2, 4, 3), complex), gains, a1, a2, sigma)

    assert np.allclose(sigma_cal, sigma / np.abs(g_bl))
    weight, weight_cal = 1.0 / sigma**2, 1.0 / sigma_cal**2
    assert np.allclose(weight_cal, weight * np.abs(g_bl) ** 2)


def test_match_subset_grid(gains):
    """A table solved on a master must select correctly onto a subset carved from it."""
    cal = {
        "gains": gains,                                   # (6 ant, 4 freq, 3 time)
        "times": np.array([10.0, 20.0, 30.0]),
        "freqs": np.array([1e8, 2e8, 3e8, 4e8]),
    }
    # A subset: the middle time, two of the four channels — deliberately out of order.
    got = match_gains_to_grid(cal, times=[20.0], freqs=[3e8, 1e8])

    assert got.shape == (6, 2, 1)
    assert np.allclose(got[:, 0, 0], gains[:, 2, 1])      # 3e8 -> channel 2
    assert np.allclose(got[:, 1, 0], gains[:, 0, 1])      # 1e8 -> channel 0


def test_match_rejects_missing_sample(gains):
    """Silently interpolating a missing solution is how you get a plausible wrong answer."""
    cal = {
        "gains": gains,
        "times": np.array([10.0, 20.0, 30.0]),
        "freqs": np.array([1e8, 2e8, 3e8, 4e8]),
    }
    with pytest.raises(ValueError, match="time"):
        match_gains_to_grid(cal, times=[25.0], freqs=[1e8])
    with pytest.raises(ValueError, match="frequency"):
        match_gains_to_grid(cal, times=[10.0], freqs=[5e8])


def test_scalar_flux_scale_is_g_k_minus_half():
    """flux-calibrate's V_cal = k V_obs is the antenna-independent gain g = k**-0.5."""
    k = 1700.0
    n_ant = 4
    g = np.full((n_ant, 1, 1), k**-0.5, dtype=complex)
    a1, a2 = np.array([0, 1]), np.array([1, 2])

    vis = np.ones((2, 1, 1), dtype=complex)
    vis_cal, _ = apply_gains_to_data(vis, g, a1, a2)

    assert np.allclose(vis_cal, k * vis)
