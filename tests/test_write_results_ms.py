"""End-to-end tests for ``write_results_ms``.

The unit tests in ``test_write.py`` pin the helpers; nothing pinned the writer
that wires them together, so reverting it to mean-then-multiply -- or to the
original ANTENNA1-twice gain -- failed no test. These run the real writer over a
real results zarr and an in-memory stand-in for the measurement set, and compare
every written column against an independent numpy reference.

Every case uses non-unit, non-uniform complex gains: unity gains cannot tell
``g_p conj(g_q)`` from ``|g_p|^2``, nor a gained model from a raw one.
"""

import warnings

import numpy as np
import pytest

import dask.array as da
import xarray as xr

import tabascal.write as write_mod
from tabascal.write import write_results_ms


N_ANT = 4
N_TIME = 3
N_FREQ = 2
N_CORR = 1

A1_BL, A2_BL = np.triu_indices(N_ANT, k=1)
N_BL = len(A1_BL)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _to_ms(arr):
    """``(bl, freq, time)`` to ``(row, chan, corr)``, time-major -- the reference.

    Written out longhand rather than reusing ``_to_ms_column`` so the test does
    not inherit a mistake from the code it checks.
    """

    arr = np.asarray(arr)
    out = np.empty((N_TIME * N_BL, N_FREQ, N_CORR), dtype=arr.dtype)

    for t in range(N_TIME):
        for b in range(N_BL):
            out[t * N_BL + b, :, 0] = arr[b, :, t]

    return out


def _model(n_sample: int, seed: int = 0):
    """Gains, astronomical and RFI models that covary across samples.

    ``E[g m]`` differs from ``E[g] E[m]`` only when the two covary, so a straight
    per-sample-independent draw would let a mean-then-multiply writer pass. Both
    the gain and the models are scaled by the sample index here.
    """

    rng = np.random.default_rng(seed)

    amp = np.array([0.5, 1.0, 2.0, 1.5])
    phase = np.array([0.0, 0.3, -0.7, 1.1])
    gain_1 = (amp * np.exp(1j * phase))[:, None, None] * np.ones((1, N_FREQ, N_TIME))

    shape = (N_BL, N_FREQ, N_TIME)
    ast_1 = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    rfi_1 = 10 * (rng.normal(size=shape) + 1j * rng.normal(size=shape))

    scale = 1.0 + np.arange(n_sample)
    gains = gain_1[None] * scale[:, None, None, None]
    ast = ast_1[None] * scale[:, None, None, None]
    rfi = rfi_1[None] * (scale**2)[:, None, None, None]

    return gains, ast, rfi


def _baseline_gains(gains):
    """``g_p conj(g_q)`` per sample, the reference form."""

    return gains[:, A1_BL] * np.conj(gains[:, A2_BL])


def _write_zarr(tmp_path, gains, ast, rfi, name="results.zarr", vis_obs=None):
    """A results zarr in the layout ``write_results_xds`` produces."""

    if vis_obs is None:
        vis_obs = _baseline_gains(gains) * (ast + rfi)

    n_bl = ast.shape[1]

    xds = xr.Dataset(
        data_vars={
            "ast_vis": (["sample", "bl", "freq", "time"], da.asarray(ast)),
            "rfi_vis": (["sample", "bl", "freq", "time"], da.asarray(rfi)),
            "vis_obs": (["sample", "bl", "freq", "time"], da.asarray(vis_obs)),
            "gains": (["sample", "ant", "freq", "time"], da.asarray(gains)),
        },
        coords={
            "time": np.arange(N_TIME, dtype=float),
            "freq": np.linspace(1.0e9, 1.1e9, N_FREQ),
            "bl": np.arange(n_bl),
        },
    )

    path = str(tmp_path / name)
    xds.to_zarr(path, mode="w")

    return path


def _fake_ms(data):
    """An MS-like dataset: dask-backed, time-major rows, one correlation."""

    row_chunk = N_BL

    return xr.Dataset(
        data_vars={
            "DATA": (
                ["row", "chan", "corr"],
                da.from_array(data, chunks=(row_chunk, N_FREQ, N_CORR)),
            ),
            "ANTENNA1": (
                ["row"], da.from_array(np.tile(A1_BL, N_TIME), chunks=row_chunk)
            ),
            "ANTENNA2": (
                ["row"], da.from_array(np.tile(A2_BL, N_TIME), chunks=row_chunk)
            ),
            "TIME": (
                ["row"],
                da.from_array(
                    np.repeat(np.arange(N_TIME, dtype=float), N_BL), chunks=row_chunk
                ),
            ),
        }
    )


def _observed(model_ms, seed: int = 7, noise_frac: float = 0.05):
    """MS ``DATA``: the gained model plus noise, stored complex64 as an MS is.

    The noise is scaled to the model, not a fixed 0.01: the writer works in
    complex64, so a residual that is orders of magnitude below the visibilities
    it is differenced from measures float32 cancellation rather than the writer.
    """

    rng = np.random.default_rng(seed)
    scale = float(np.abs(model_ms).max())
    noise = noise_frac * scale * (
        rng.normal(size=model_ms.shape) + 1j * rng.normal(size=model_ms.shape)
    )

    return (model_ms + noise).astype(np.complex64)


def _tolerances(data):
    """complex64 round-off, referenced to the magnitude being differenced."""

    return dict(rtol=1e-5, atol=1e-5 * float(np.abs(data).max()))


COLS = [
    "CORRECTED_DATA",
    "TAB_AST_DATA",
    "TAB_RFI_DATA",
    "TAB_AST_RES",
    "TAB_RFI_RES",
    "TAB_RES_DATA",
]


@pytest.fixture
def run_writer(monkeypatch):
    """Run ``write_results_ms`` against an in-memory MS, capturing the columns.

    ``xds_from_ms``/``xds_to_table`` are the only casacore-facing calls, so
    replacing them keeps the whole writer under test without an MS on disk.
    """

    captured = {}

    def _run(xds_ms, zarr_path):
        monkeypatch.setattr(write_mod, "xds_from_ms", lambda path: [xds_ms])

        def _capture(datasets, path, cols, column_keywords=None):
            captured["xds"] = datasets[0]
            captured["cols"] = list(cols)
            captured["keywords"] = column_keywords
            return []

        monkeypatch.setattr(write_mod, "xds_to_table", _capture)

        write_results_ms("unused.ms", zarr_path)

        values = {
            col: np.asarray(captured["xds"][col].data) for col in captured["cols"]
        }

        return values, captured

    return _run


# ---------------------------------------------------------------------------
# Column values
# ---------------------------------------------------------------------------

class TestWrittenColumns:

    @pytest.mark.parametrize("n_sample", [1, 2])
    def test_every_column_matches_the_reference(self, tmp_path, run_writer, n_sample):
        gains, ast, rfi = _model(n_sample)
        gains_bl = _baseline_gains(gains)
        vis_obs_zarr = gains_bl * (ast + rfi)

        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        data = _observed(_to_ms(vis_obs_zarr.mean(axis=0)))

        values, captured = run_writer(_fake_ms(data), zarr_path)

        assert captured["cols"] == COLS

        mean_gains_bl = _to_ms(gains_bl.mean(axis=0))
        gained_ast = _to_ms((gains_bl * ast).mean(axis=0))
        gained_rfi = _to_ms((gains_bl * rfi).mean(axis=0))
        gained_total = _to_ms(vis_obs_zarr.mean(axis=0))

        kw = _tolerances(data)
        np.testing.assert_allclose(
            values["CORRECTED_DATA"], data / mean_gains_bl, **kw
        )
        np.testing.assert_allclose(values["TAB_AST_DATA"], _to_ms(ast.mean(0)), **kw)
        np.testing.assert_allclose(values["TAB_RFI_DATA"], _to_ms(rfi.mean(0)), **kw)
        np.testing.assert_allclose(values["TAB_AST_RES"], data - gained_ast, **kw)
        np.testing.assert_allclose(values["TAB_RFI_RES"], data - gained_rfi, **kw)
        np.testing.assert_allclose(values["TAB_RES_DATA"], data - gained_total, **kw)

    def test_the_columns_are_written_in_the_ms_frame(self, tmp_path, run_writer):
        """Shape and dims, not just values: the transpose is easy to get wrong."""
        gains, ast, rfi = _model(1)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        xds_ms = _fake_ms(np.zeros((N_TIME * N_BL, N_FREQ, N_CORR), np.complex64))

        values, captured = run_writer(xds_ms, zarr_path)

        for col in COLS:
            assert captured["xds"][col].dims == ("row", "chan", "corr")
            assert values[col].shape == (N_TIME * N_BL, N_FREQ, N_CORR)

        assert captured["keywords"] == {col: {"UNIT": "Jy"} for col in COLS}

    def test_the_total_residual_closes_on_a_perfect_model(self, tmp_path, run_writer):
        """With DATA equal to the stored forward model, nothing is left over."""
        gains, ast, rfi = _model(1)
        vis_obs_zarr = _baseline_gains(gains) * (ast + rfi)

        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        data = _to_ms(vis_obs_zarr.mean(axis=0)).astype(np.complex64)

        values, _ = run_writer(_fake_ms(data), zarr_path)

        np.testing.assert_allclose(
            values["TAB_RES_DATA"], 0.0, **_tolerances(data)
        )


class TestReductionOrder:
    """``E[g m]`` is formed per sample, not from the two sample means."""

    def test_mean_then_multiply_would_disagree(self, tmp_path, run_writer):
        gains, ast, rfi = _model(2)
        gains_bl = _baseline_gains(gains)

        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        data = _observed(_to_ms((gains_bl * (ast + rfi)).mean(axis=0)))

        values, _ = run_writer(_fake_ms(data), zarr_path)

        naive_ast = _to_ms(gains_bl.mean(axis=0) * ast.mean(axis=0))
        naive_gain = _to_ms(
            gains.mean(axis=0)[A1_BL] * np.conj(gains.mean(axis=0)[A2_BL])
        )

        assert not np.allclose(values["TAB_AST_RES"], data - naive_ast, rtol=1e-3)
        assert not np.allclose(
            values["CORRECTED_DATA"], data / naive_gain, rtol=1e-3
        )


class TestBothAntennasAreUsed:
    """The original bug: ANTENNA1 indexed twice, giving ``|g_p|^2``."""

    def test_the_calibrated_column_is_not_the_autocorrelation_form(
        self, tmp_path, run_writer
    ):
        gains, ast, rfi = _model(1)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        data = _observed(_to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)))

        values, _ = run_writer(_fake_ms(data), zarr_path)

        wrong_gain = _to_ms((gains[:, A1_BL] * np.conj(gains[:, A1_BL])).mean(axis=0))

        assert not np.allclose(values["CORRECTED_DATA"], data / wrong_gain, rtol=1e-3)
        # The wrong form is real and positive; the right one carries phase.
        assert not np.allclose(np.imag(values["CORRECTED_DATA"]), np.imag(data))


class TestTheStoredModelIsPreferred:
    """The zarr's vis_obs is the forward model, which need not be the sum."""

    def test_a_gains_component_that_gains_only_one_term(self, tmp_path, run_writer):
        """Re-deriving the total from the two parts would be wrong here.

        See the commented-out variant in components/gains.py, where the
        astronomical term carries the gain and the RFI does not.
        """
        gains, ast, rfi = _model(1)
        gains_bl = _baseline_gains(gains)

        stored = gains_bl * ast + rfi
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi, vis_obs=stored)
        data = _observed(_to_ms(stored.mean(axis=0)))

        values, _ = run_writer(_fake_ms(data), zarr_path)

        np.testing.assert_allclose(
            values["TAB_RES_DATA"],
            data - _to_ms(stored.mean(axis=0)),
            **_tolerances(data),
        )

        summed = _to_ms((gains_bl * (ast + rfi)).mean(axis=0))
        assert not np.allclose(values["TAB_RES_DATA"], data - summed, rtol=1e-3)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

class TestBaselineCountGuard:

    def test_a_zarr_from_another_ms_is_rejected(self, tmp_path, run_writer):
        """Three baselines in the results, six in the MS."""
        gains, ast, rfi = _model(1)
        gains = gains[:, :3]
        ast, rfi = ast[:, :3], rfi[:, :3]
        vis_obs = np.zeros_like(ast)

        zarr_path = _write_zarr(tmp_path, gains, ast, rfi, vis_obs=vis_obs)
        xds_ms = _fake_ms(np.zeros((N_TIME * N_BL, N_FREQ, N_CORR), np.complex64))

        with pytest.raises(ValueError, match="does not belong"):
            run_writer(xds_ms, zarr_path)


def _substitute(gains):
    """The unity substitution, written out as the reference expects it."""

    return np.where(~np.isfinite(gains) | (gains == 0), 1.0, gains)


class TestBadGainsAreSubstituted:
    """A dead antenna is pushed to unity, so nothing is blanked."""

    @pytest.mark.parametrize("value", [0.0, np.nan])
    def test_a_dead_antenna_leaves_every_column_finite(
        self, tmp_path, run_writer, value
    ):
        gains, ast, rfi = _model(2)
        gains = gains.copy()
        gains[0, 2] = value                     # antenna 2, on the first sample

        # The zarr's vis_obs is what the run stored: formed with the bad gain.
        stored = _baseline_gains(gains) * (ast + rfi)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi, vis_obs=stored)

        gains_bl = _baseline_gains(_substitute(gains))
        data = _observed(_to_ms((gains_bl * (ast + rfi)).mean(axis=0)))

        with pytest.warns(RuntimeWarning, match=r"Affected antennas: \[2\]"):
            values, _ = run_writer(_fake_ms(data), zarr_path)

        for col in COLS:
            assert np.all(np.isfinite(values[col])), col

        kw = _tolerances(data)
        gained_ast = _to_ms((gains_bl * ast).mean(axis=0))
        gained_rfi = _to_ms((gains_bl * rfi).mean(axis=0))

        np.testing.assert_allclose(
            values["CORRECTED_DATA"], data / _to_ms(gains_bl.mean(axis=0)), **kw
        )
        np.testing.assert_allclose(values["TAB_AST_RES"], data - gained_ast, **kw)
        np.testing.assert_allclose(values["TAB_RFI_RES"], data - gained_rfi, **kw)

        # The stored total still carries the bad gain on the touched baselines,
        # so the total residual is re-derived there and stored elsewhere.
        touched = np.tile((A1_BL == 2) | (A2_BL == 2), N_TIME)
        np.testing.assert_allclose(
            values["TAB_RES_DATA"][touched],
            (data - (gained_ast + gained_rfi))[touched],
            **kw,
        )
        np.testing.assert_allclose(
            values["TAB_RES_DATA"][~touched],
            (data - _to_ms(stored.mean(axis=0)))[~touched],
            **kw,
        )

    def test_the_other_antennas_gain_is_still_applied(self, tmp_path, run_writer):
        """Uncalibrated on the dead antenna only, not on the whole baseline."""
        gains, ast, rfi = _model(1)
        gains = gains.copy()
        gains[:, 2] = 0.0

        stored = _baseline_gains(gains) * (ast + rfi)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi, vis_obs=stored)

        gains_sub = _substitute(gains)
        data = _observed(_to_ms((_baseline_gains(gains_sub) * (ast + rfi)).mean(0)))

        with pytest.warns(RuntimeWarning):
            values, _ = run_writer(_fake_ms(data), zarr_path)

        # On baseline (0, 2) the divisor is g_0 * conj(1), not g_0 * conj(g_2)
        # and not 1: antenna 0 is still calibrated.
        bl = int(np.flatnonzero((A1_BL == 0) & (A2_BL == 2))[0])
        kw = _tolerances(data)

        for t in range(N_TIME):
            row = t * N_BL + bl
            np.testing.assert_allclose(
                values["CORRECTED_DATA"][row, :, 0],
                data[row, :, 0] / gains_sub[0, 0, :, t],
                **kw,
            )

        assert not np.allclose(
            values["CORRECTED_DATA"][bl :: N_BL], data[bl :: N_BL], rtol=1e-3
        )

    def test_the_stored_total_is_not_used_where_the_gain_was_bad(
        self, tmp_path, run_writer
    ):
        """Without the fallback the zero in the stored model leaks into the residual."""
        gains, ast, rfi = _model(1)
        gains = gains.copy()
        gains[:, 2] = 0.0

        stored = _baseline_gains(gains) * (ast + rfi)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi, vis_obs=stored)

        gains_bl = _baseline_gains(_substitute(gains))
        data = _observed(_to_ms((gains_bl * (ast + rfi)).mean(axis=0)))

        with pytest.warns(RuntimeWarning):
            values, _ = run_writer(_fake_ms(data), zarr_path)

        touched = np.tile((A1_BL == 2) | (A2_BL == 2), N_TIME)
        naive = (data - _to_ms(stored.mean(axis=0)))[touched]

        # The stored model is zero on those baselines, so the naive residual is
        # just the data -- an obviously wrong, and obviously different, answer.
        assert not np.allclose(values["TAB_RES_DATA"][touched], naive, rtol=1e-3)

    def test_one_bad_sample_does_not_discard_the_stored_model_on_the_other(
        self, tmp_path, run_writer
    ):
        """The fallback is per sample, not per cell."""
        gains, ast, rfi = _model(2)
        gains = gains.copy()
        gains[0, 2] = 0.0                    # antenna 2 dead on sample 0 only

        gains_bl = _baseline_gains(_substitute(gains))

        # A one-term-gained forward model, so the stored total differs from the
        # sum of the two gained parts and which one was used is visible.
        stored = _baseline_gains(gains) * ast + rfi
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi, vis_obs=stored)
        data = _observed(_to_ms((gains_bl * (ast + rfi)).mean(axis=0)))

        with pytest.warns(RuntimeWarning):
            values, _ = run_writer(_fake_ms(data), zarr_path)

        touched = (A1_BL == 2) | (A2_BL == 2)
        bad_bl_s = np.zeros(ast.shape, dtype=bool)
        bad_bl_s[0, touched] = True

        expected = np.where(bad_bl_s, gains_bl * (ast + rfi), stored).mean(axis=0)
        np.testing.assert_allclose(
            values["TAB_RES_DATA"],
            data - _to_ms(expected),
            **_tolerances(data),
        )

        # Reducing the mask over samples first rebuilds the total on *both*
        # samples of the touched cells, throwing away the stored sample 1.
        any_sample = np.where(
            bad_bl_s.any(axis=0), gains_bl * (ast + rfi), stored
        ).mean(axis=0)
        rows = np.tile(touched, N_TIME)
        assert not np.allclose(
            values["TAB_RES_DATA"][rows],
            (data - _to_ms(any_sample))[rows],
            rtol=1e-3,
        )

    def test_finite_gains_raise_no_warning(self, tmp_path, run_writer):
        gains, ast, rfi = _model(1)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        data = _observed(_to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)))

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            values, _ = run_writer(_fake_ms(data), zarr_path)

        assert np.all(np.isfinite(values["CORRECTED_DATA"]))
