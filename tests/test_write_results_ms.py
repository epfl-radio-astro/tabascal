"""End-to-end tests for ``write_results_ms``.

The unit tests in ``test_write.py`` pin the helpers; nothing pinned the writer
that wires them together, so reverting it to mean-then-multiply -- or to the
original ANTENNA1-twice gain -- failed no test. These run the real writer over a
real results zarr and an in-memory stand-in for the measurement set, and compare
every written column against an independent numpy reference.

Every case uses non-unit, non-uniform complex gains: unity gains cannot tell
``g_p conj(g_q)`` from ``|g_p|^2``, nor a gained model from a raw one.

``write_results_xds`` is covered here too, for the one thing the writer reads
back out of it: which correlation the run fitted.
"""

import warnings

import numpy as np
import pytest

import dask.array as da
import xarray as xr

import tabascal.write as write_mod
from tabascal.write import write_results_ms, write_results_xds


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
    out = np.empty((N_TIME * N_BL, N_FREQ, 1), dtype=arr.dtype)

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


def _write_zarr(
    tmp_path, gains, ast, rfi, name="results.zarr", vis_obs=None, corr=None
):
    """A results zarr in the layout ``write_results_xds`` produces.

    ``corr`` is the fitted-correlation attribute; ``None`` leaves it off, which
    is what a zarr written before the attribute existed looks like.
    """

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
        attrs={} if corr is None else {"corr": corr},
    )

    path = str(tmp_path / name)
    xds.to_zarr(path, mode="w")

    return path


def _fake_ms(data):
    """An MS-like dataset: dask-backed, time-major rows.

    The correlation axis is however wide the ``DATA`` given here is, so the same
    fake covers a single-correlation MS and a full four-correlation one.
    """

    row_chunk = N_BL
    n_corr = data.shape[2]

    return xr.Dataset(
        data_vars={
            "DATA": (
                ["row", "chan", "corr"],
                da.from_array(data, chunks=(row_chunk, N_FREQ, n_corr)),
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


def _spread(fit, n_corr: int, corr_idx: int, seed: int = 11):
    """A full MS ``DATA`` column: the fitted correlation, junk on the others.

    The other correlations carry values of their own so that "passed through
    unchanged" is a real assertion rather than a comparison of zeros.
    """

    if n_corr == 1:
        return fit

    rng = np.random.default_rng(seed)
    shape = (fit.shape[0], fit.shape[1], n_corr)
    scale = float(np.abs(fit).max())
    other = scale * (rng.normal(size=shape) + 1j * rng.normal(size=shape))

    data = other.astype(fit.dtype)
    data[:, :, corr_idx] = fit[:, :, 0]

    return data


def _fit(col, corr_idx: int = 0):
    """The fitted correlation's slice of a written column, as ``(row, chan, 1)``."""

    return col[:, :, corr_idx : corr_idx + 1]


def _others(col, corr_idx: int):
    """Every correlation of a written column except the fitted one."""

    keep = [c for c in range(col.shape[2]) if c != corr_idx]

    return col[:, :, keep]


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

    def _run(xds_ms, zarr_path, *, corr=None, corr_idx=0):
        monkeypatch.setattr(write_mod, "xds_from_ms", lambda path: [xds_ms])

        def _capture(datasets, path, cols, column_keywords=None):
            captured["xds"] = datasets[0]
            captured["cols"] = list(cols)
            captured["keywords"] = column_keywords
            return []

        def _resolve(ms_path, name, pol_id=0):
            """Stand in for the casacore-backed resolver in tabascal.ms."""
            captured["resolved"] = name
            return corr_idx

        monkeypatch.setattr(write_mod, "xds_to_table", _capture)
        monkeypatch.setattr(write_mod, "resolve_correlation", _resolve)

        write_results_ms("unused.ms", zarr_path, corr=corr)

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
    @pytest.mark.parametrize(
        "n_corr, corr_idx", [(1, 0), (4, 2)], ids=["one_corr", "four_corr"]
    )
    def test_every_column_matches_the_reference(
        self, tmp_path, run_writer, n_sample, n_corr, corr_idx
    ):
        gains, ast, rfi = _model(n_sample)
        gains_bl = _baseline_gains(gains)
        vis_obs_zarr = gains_bl * (ast + rfi)

        zarr_path = _write_zarr(tmp_path, gains, ast, rfi, corr="yx")
        fit = _observed(_to_ms(vis_obs_zarr.mean(axis=0)))
        data = _spread(fit, n_corr, corr_idx)

        values, captured = run_writer(_fake_ms(data), zarr_path, corr_idx=corr_idx)

        assert captured["cols"] == COLS
        assert captured["resolved"] == "yx"          # resolved by name, not index

        mean_gains_bl = _to_ms(gains_bl.mean(axis=0))
        gained_ast = _to_ms((gains_bl * ast).mean(axis=0))
        gained_rfi = _to_ms((gains_bl * rfi).mean(axis=0))
        gained_total = _to_ms(vis_obs_zarr.mean(axis=0))

        kw = _tolerances(fit)
        got = {col: _fit(values[col], corr_idx) for col in COLS}

        np.testing.assert_allclose(got["CORRECTED_DATA"], fit / mean_gains_bl, **kw)
        np.testing.assert_allclose(got["TAB_AST_DATA"], _to_ms(ast.mean(0)), **kw)
        np.testing.assert_allclose(got["TAB_RFI_DATA"], _to_ms(rfi.mean(0)), **kw)
        np.testing.assert_allclose(got["TAB_AST_RES"], fit - gained_ast, **kw)
        np.testing.assert_allclose(got["TAB_RFI_RES"], fit - gained_rfi, **kw)
        np.testing.assert_allclose(got["TAB_RES_DATA"], fit - gained_total, **kw)

        for col in COLS:
            assert values[col].shape == (N_TIME * N_BL, N_FREQ, n_corr)

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


class TestTheMeanBaselineGainIsGuarded:
    """Every per-sample gain can be healthy and their mean still be zero."""

    @pytest.fixture
    def flipped_gains(self):
        """Antenna 3's gain flips sign between the two samples.

        Every baseline touching it then averages to exactly zero, while every
        individual sample is finite and non-zero -- so the per-antenna guard
        sees nothing to do.
        """
        amp = np.array([0.5, 1.0, 2.0, 1.0])
        phase = np.array([0.0, 0.3, -0.7, 0.0])
        one = (amp * np.exp(1j * phase))[:, None, None] * np.ones((1, N_FREQ, N_TIME))

        gains = np.stack([one, one.copy()])
        gains[1, 3] *= -1

        return gains

    def test_the_calibrated_data_stays_finite(
        self, tmp_path, run_writer, flipped_gains
    ):
        gains = flipped_gains
        _, ast, rfi = _model(2)

        assert np.all(np.isfinite(gains)) and not np.any(gains == 0)

        stored = _baseline_gains(gains) * (ast + rfi)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi, vis_obs=stored)
        data = _observed(_to_ms(stored.mean(axis=0)))

        with pytest.warns(RuntimeWarning, match="mean baseline gains") as record:
            values, _ = run_writer(_fake_ms(data), zarr_path)

        # The per-antenna guard has nothing to say here.
        assert not any("Affected antennas" in str(w.message) for w in record)

        for col in COLS:
            assert np.all(np.isfinite(values[col])), col

        kw = _tolerances(data)
        touched = np.tile((A1_BL == 3) | (A2_BL == 3), N_TIME)

        # Divided by 1 there: CORRECTED_DATA is the data, uncalibrated.
        np.testing.assert_allclose(
            values["CORRECTED_DATA"][touched], data[touched], **kw
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            calibrated = data / _to_ms(_baseline_gains(gains).mean(axis=0))

        np.testing.assert_allclose(
            values["CORRECTED_DATA"][~touched], calibrated[~touched], **kw
        )

    def test_without_the_guard_the_column_would_be_infinite(self, flipped_gains):
        """Why it matters: the mean divisor is exactly zero on those baselines."""
        gains_bl = _baseline_gains(flipped_gains).mean(axis=0)
        touched = (A1_BL == 3) | (A2_BL == 3)

        assert np.all(gains_bl[touched] == 0)
        assert np.all(gains_bl[~touched] != 0)


class TestMultipleCorrelations:
    """tabascal fits one correlation; the others must survive untouched."""

    N_CORR = 4
    CORR_IDX = 2

    @pytest.fixture
    def run(self, tmp_path, run_writer):
        """A four-correlation MS whose fitted correlation sits at index 2."""

        gains, ast, rfi = _model(1)
        gains_bl = _baseline_gains(gains)
        stored = gains_bl * (ast + rfi)

        fit = _observed(_to_ms(stored.mean(axis=0)))
        data = _spread(fit, self.N_CORR, self.CORR_IDX)

        def _go(*, corr=None, zarr_corr="yx", n_corr=None, corr_idx=None):
            n = self.N_CORR if n_corr is None else n_corr
            idx = self.CORR_IDX if corr_idx is None else corr_idx
            zarr_path = _write_zarr(tmp_path, gains, ast, rfi, corr=zarr_corr)
            ms_data = data if n == self.N_CORR else _spread(fit, n, idx)

            return _fake_ms(ms_data), zarr_path, ms_data, run_writer, corr, idx

        return _go

    def test_model_columns_are_zero_on_the_other_correlations(self, run):
        xds_ms, zarr_path, _, run_writer, corr, idx = run()

        values, _ = run_writer(xds_ms, zarr_path, corr=corr, corr_idx=idx)

        for col in ["TAB_AST_DATA", "TAB_RFI_DATA"]:
            np.testing.assert_array_equal(_others(values[col], idx), 0.0)
            assert not np.allclose(_fit(values[col], idx), 0.0)

    def test_data_frame_columns_pass_the_data_through(self, run):
        """Ungained and unsubtracted, which is what those columns mean there."""
        xds_ms, zarr_path, ms_data, run_writer, corr, idx = run()

        values, _ = run_writer(xds_ms, zarr_path, corr=corr, corr_idx=idx)

        for col in ["CORRECTED_DATA", "TAB_AST_RES", "TAB_RFI_RES", "TAB_RES_DATA"]:
            np.testing.assert_array_equal(
                _others(values[col], idx), _others(ms_data, idx)
            )
            # And the fitted correlation is *not* the raw data.
            assert not np.allclose(
                _fit(values[col], idx), _fit(ms_data, idx), rtol=1e-3
            )

    def test_the_dtype_of_the_written_columns_is_unchanged(self, run):
        xds_ms, zarr_path, ms_data, run_writer, corr, idx = run()

        values, _ = run_writer(xds_ms, zarr_path, corr=corr, corr_idx=idx)

        for col in COLS:
            assert values[col].dtype == ms_data.dtype

    def test_the_correlation_comes_from_the_zarr_attribute(self, run):
        xds_ms, zarr_path, _, run_writer, corr, idx = run()

        _, captured = run_writer(xds_ms, zarr_path, corr=corr, corr_idx=idx)

        assert captured["resolved"] == "yx"

    def test_an_explicit_argument_overrides_the_zarr(self, run):
        xds_ms, zarr_path, _, run_writer, _, idx = run(zarr_corr="xx")

        _, captured = run_writer(xds_ms, zarr_path, corr="yy", corr_idx=idx)

        assert captured["resolved"] == "yy"

    def test_a_zarr_without_the_attribute_is_rejected(self, run):
        """An older zarr does not say which correlation it belongs to."""
        xds_ms, zarr_path, _, run_writer, _, idx = run(zarr_corr=None)

        with pytest.raises(ValueError, match="does not record which one"):
            run_writer(xds_ms, zarr_path, corr_idx=idx)

    def test_a_zarr_without_the_attribute_is_fine_on_a_one_corr_ms(self, run):
        """There is only one answer, so nothing has to be guessed."""
        xds_ms, zarr_path, _, run_writer, _, idx = run(
            zarr_corr=None, n_corr=1, corr_idx=0
        )

        values, captured = run_writer(xds_ms, zarr_path, corr_idx=0)

        assert "resolved" not in captured        # never had to resolve a name
        assert values["CORRECTED_DATA"].shape[2] == 1


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


# ---------------------------------------------------------------------------
# The other end of the round trip
# ---------------------------------------------------------------------------

class TestWriteResultsXdsRecordsTheCorrelation:
    """The writer can only place the results if the run said where they go."""

    @pytest.fixture
    def tab_config(self):
        """The three attributes write_results_xds reads off the config."""

        class _Config:
            args = {"data": {"corr": "yx"}}
            times = np.arange(N_TIME, dtype=float)
            freqs = np.linspace(1.0e9, 1.1e9, N_FREQ)

        return _Config()

    @pytest.fixture
    def vi_pred(self):
        gains, ast, rfi = _model(1)

        return {
            "vis_ast": ast,
            "vis_rfi": rfi,
            "gains": gains,
            "vis_obs": _baseline_gains(gains) * (ast + rfi),
        }

    def test_the_correlation_is_recorded_as_an_attribute(
        self, tmp_path, monkeypatch, tab_config, vi_pred
    ):
        monkeypatch.setattr(write_mod, "is_process_0", lambda: True)
        path = str(tmp_path / "map_pred.zarr")

        write_results_xds(vi_pred, tab_config, path)

        assert xr.open_zarr(path).attrs["corr"] == "yx"

    def test_the_writer_reads_it_back(
        self, tmp_path, monkeypatch, run_writer, tab_config, vi_pred
    ):
        """The round trip: no corr argument needed on a four-correlation MS."""
        monkeypatch.setattr(write_mod, "is_process_0", lambda: True)
        path = str(tmp_path / "map_pred.zarr")
        write_results_xds(vi_pred, tab_config, path)

        fit = _observed(_to_ms(vi_pred["vis_obs"].mean(axis=0)))
        data = _spread(fit, 4, 2)

        _, captured = run_writer(_fake_ms(data), path, corr_idx=2)

        assert captured["resolved"] == "yx"
