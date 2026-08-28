"""End-to-end tests for ``write_results_ms``.

The unit tests in ``test_write.py`` pin the helpers; nothing pinned the writer
that wires them together, so reverting it to mean-then-multiply -- or to the
original ANTENNA1-twice gain -- failed no test. These run the real writer over a
real results zarr and an in-memory stand-in for the measurement set, and compare
every written column against an independent numpy reference.

Every case uses non-unit, non-uniform complex gains: unity gains cannot tell
``g_p conj(g_q)`` from ``|g_p|^2``, nor a calibrated column from a data-frame one.

Every column is written in ONE frame -- the data with both gain layers, the
external table's and the fitted DIE gains, divided out (#123) -- so the anchor
assertion here is the closure identity ``TAB_AST_DATA + TAB_RFI_DATA +
TAB_RES_DATA == CORRECTED_DATA``, which fails loudly if any column moves frame.

The calibration table the writer exports beside the results is covered here as
well, since what makes it right is that it reproduces the columns written in the
same call -- and what makes it interesting is where it deliberately does not:
a gain the fit killed is flagged in the table and substituted with 1 in the
columns. Those cases need a real MS on disk to copy subtables out of
(``ms_skeleton``) and are skipped where casacore is not installed.

``write_results_xds`` is covered here too, for the one thing the writer reads
back out of it: which correlation the run fitted.

No jax runs here: the writer works in ``complex64`` in either session
precision, so the tolerances come from float32 round-off (``_tolerances``) and
not from the ``exact_rtol`` fixture, whose fp64 bound the writer could not meet.
"""

import os
import warnings

import numpy as np
import pytest

import dask.array as da
import xarray as xr

import tabascal.ms as ms_mod
import tabascal.write as write_mod
from tabascal.ms import read_caltable
from tabascal.write import write_results_ms, write_results_xds


N_ANT = 4
N_TIME = 3
N_FREQ = 2
N_CORR = 1

A1_BL, A2_BL = np.triu_indices(N_ANT, k=1)
N_BL = len(A1_BL)

#: Per-baseline noise for the fake MS's ``SIGMA``. Non-uniform, so a weight
#: built from the wrong axis -- or from one number for the whole array -- shows.
SIGMA_BL = np.array([0.3, 0.5, 0.7, 1.1, 1.3, 1.7])

#: Per-channel factors on top of it for ``SIGMA_SPECTRUM``: a bandpass is not
#: flat, and a weight that ignores the channel axis reproduces neither.
SIGMA_CHAN = np.array([1.0, 2.5])


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
    tmp_path,
    gains,
    ast,
    rfi,
    name="results.zarr",
    vis_obs=None,
    corr=None,
    sample_chunk=None,
):
    """A results zarr in the layout ``write_results_xds`` produces.

    ``corr`` is the fitted-correlation attribute; ``None`` leaves it off, which
    is what a zarr written before the attribute existed looks like.

    ``sample_chunk`` stores the arrays chunked along the sample axis, as a
    posterior with many samples is: the export reduces them chunk by chunk and
    must never hold the whole four-dimensional array.
    """

    if vis_obs is None:
        vis_obs = _baseline_gains(gains) * (ast + rfi)

    n_bl = ast.shape[1]
    # Taken from the arrays rather than fixed at the module's counts, so a run
    # narrowed with data.freq -- fewer channels in the results than in the MS --
    # is built by the same helper as a full-band one.
    n_freq = ast.shape[2]

    xds = xr.Dataset(
        data_vars={
            "ast_vis": (["sample", "bl", "freq", "time"], da.asarray(ast)),
            "rfi_vis": (["sample", "bl", "freq", "time"], da.asarray(rfi)),
            "vis_obs": (["sample", "bl", "freq", "time"], da.asarray(vis_obs)),
            "gains": (["sample", "ant", "freq", "time"], da.asarray(gains)),
        },
        coords={
            "time": np.arange(N_TIME, dtype=float),
            "freq": np.linspace(1.0e9, 1.1e9, N_FREQ)[:n_freq],
            "bl": np.arange(n_bl),
        },
        attrs={} if corr is None else {"corr": corr},
    )

    if sample_chunk is not None:
        xds = xds.chunk({"sample": sample_chunk})

    path = str(tmp_path / name)
    xds.to_zarr(path, mode="w")

    return path


def _sigma_column(n_row, n_corr, per_chan: bool):
    """A noise column for the fake MS, differing per correlation.

    Each correlation carries its own noise, so reading the fitted one rather
    than correlation 0 is an assertion rather than a coincidence.
    """

    scale = 1.0 + np.arange(n_corr, dtype=float)
    # resize rather than tile: the baseline-count guard's MS has a row count
    # that is not a multiple of N_BL, and it must reach the guard, not a
    # shape error from this column.
    rows = np.resize(SIGMA_BL, n_row)

    if per_chan:
        return rows[:, None, None] * SIGMA_CHAN[None, :, None] * scale

    return rows[:, None] * scale


def _sigma_grid(corr_idx: int, per_chan: bool = False):
    """The noise the writer must have read, as ``(bl, freq, time)``.

    The reference, written out from the same two constants the column is built
    from rather than by re-reading it: ``SIGMA`` resolves per baseline and
    ``SIGMA_SPECTRUM`` per (baseline, channel), and both are constant in time
    here, so the time axis is a broadcast.
    """

    sigma = SIGMA_BL[:, None] * (SIGMA_CHAN if per_chan else np.ones(N_FREQ))

    return np.broadcast_to(
        (sigma * (1.0 + corr_idx))[:, :, None], (N_BL, N_FREQ, N_TIME)
    )


def _fake_ms(data, a1=None, a2=None, times=None, noise="sigma"):
    """An MS-like dataset: dask-backed, time-major rows.

    The correlation axis is however wide the ``DATA`` given here is, so the same
    fake covers a single-correlation MS and a full four-correlation one. The row
    columns default to the standard four-antenna layout.

    ``noise`` picks the noise column(s) present: ``"sigma"`` (the default, what
    most MSs carry), ``"sigma_spectrum"``, ``"both"`` -- where the frequency
    resolved column must win -- or ``None`` for an MS with no noise at all.
    """

    row_chunk = N_BL
    n_row, _, n_corr = data.shape
    a1 = np.tile(A1_BL, N_TIME) if a1 is None else np.asarray(a1)
    a2 = np.tile(A2_BL, N_TIME) if a2 is None else np.asarray(a2)
    if times is None:
        times = np.repeat(np.arange(N_TIME, dtype=float), N_BL)

    data_vars = {
        "DATA": (
            ["row", "chan", "corr"],
            da.from_array(data, chunks=(row_chunk, N_FREQ, n_corr)),
        ),
        "ANTENNA1": (["row"], da.from_array(a1, chunks=row_chunk)),
        "ANTENNA2": (["row"], da.from_array(a2, chunks=row_chunk)),
        "TIME": (["row"], da.from_array(np.asarray(times), chunks=row_chunk)),
    }

    if noise in ("sigma", "both"):
        data_vars["SIGMA"] = (
            ["row", "corr"],
            da.from_array(_sigma_column(n_row, n_corr, False), chunks=(row_chunk, n_corr)),
        )
    if noise in ("sigma_spectrum", "both"):
        data_vars["SIGMA_SPECTRUM"] = (
            ["row", "chan", "corr"],
            da.from_array(
                _sigma_column(n_row, n_corr, True), chunks=(row_chunk, N_FREQ, n_corr)
            ),
        )

    return xr.Dataset(data_vars=data_vars)


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

#: Written after the six visibility columns, when the MS carries a noise column.
WEIGHT_COLS = ["WEIGHT_SPECTRUM", "WEIGHT"]

#: The channel frequencies the fake ``SPECTRAL_WINDOW`` declares.
FREQS = np.linspace(1.0e9, 1.1e9, N_FREQ)


@pytest.fixture
def run_writer(monkeypatch, tmp_path):
    """Run ``write_results_ms`` against an in-memory MS, capturing the columns.

    ``xds_from_ms``/``xds_from_table``/``xds_to_table`` are the only
    casacore-facing calls, so replacing them keeps the whole writer under test
    without an MS on disk.

    ``ext_gains`` supplies the external calibration the run was fitted with, in
    place of a table on disk: the *placement* of a real caltable onto the
    observation's grid is ``tabascal.gain_table``'s subject and is tested there,
    while what belongs here is the frame the writer puts the columns in and the
    grid it asks for the gains on -- which is captured and asserted.
    """

    captured = {}

    def _run(
        xds_ms,
        zarr_path,
        *,
        corr=None,
        corr_idx=0,
        pol_id=0,
        ext_gains=None,
        ext_dead=None,
        gain_table=None,
        n_ant=N_ANT,
        spw_id=0,
        ms_path="unused.ms",
        emit=False,
        time_scale=None,
    ):
        # What the MS's TIME column declares. The scale is left off unless a case
        # is about it, which is what an MS that declares only a unit looks like.
        keywords = {"TIME": {"QuantumUnits": ["s"]}}

        if time_scale is not None:
            keywords["TIME"]["MEASINFO"] = {"type": "epoch", "Ref": time_scale}

        def _from_ms(path, column_keywords=False):
            return ([xds_ms], keywords) if column_keywords else [xds_ms]

        monkeypatch.setattr(write_mod, "xds_from_ms", _from_ms)

        def _describe(ms_path, data_desc_id=0):
            """Stand in for DATA_DESCRIPTION: records the id, returns pol_id."""
            captured["data_desc_id"] = data_desc_id
            return spw_id, pol_id

        monkeypatch.setattr(ms_mod, "resolve_data_description", _describe)

        def _subtable(path, group_cols=None):
            """Stand in for the SPECTRAL_WINDOW and ANTENNA subtables."""
            if path.endswith("::SPECTRAL_WINDOW"):
                # One dataset per row, as read_ms groups them: the partition's
                # window is picked by its id rather than being assumed to be row 0.
                return [
                    xr.Dataset(
                        {"CHAN_FREQ": (["row", "chan"], da.from_array(FREQS[None]))}
                    )
                    for _ in range(spw_id + 1)
                ]
            if path.endswith("::ANTENNA"):
                return [
                    xr.Dataset(
                        {
                            "POSITION": (
                                ["row", "xyz"],
                                da.from_array(np.zeros((n_ant, 3))),
                            )
                        }
                    )
                ]
            raise AssertionError(f"unexpected subtable read: {path}")

        monkeypatch.setattr(write_mod, "xds_from_table", _subtable)

        if ext_gains is not None:
            dead = (
                np.zeros(np.shape(ext_gains), dtype=bool)
                if ext_dead is None
                else ext_dead
            )

            def _placed(paths, times, freqs, n_ant=None, verbose=True):
                captured["ext"] = {
                    "gain_table": list(paths),
                    "times": np.asarray(times),
                    "freqs": np.asarray(freqs),
                    "n_ant": n_ant,
                }
                return np.where(dead, 1.0, ext_gains), dead

            monkeypatch.setattr(write_mod, "gains_from_tables", _placed)

            if gain_table is None:
                # A real path, so normalise_gain_tables does its own job.
                table = tmp_path / "flux.B0"
                table.mkdir(exist_ok=True)
                gain_table = str(table)

        def _capture(datasets, path, cols, column_keywords=None):
            captured["xds"] = datasets[0]
            captured["cols"] = list(cols)
            captured["keywords"] = column_keywords
            return []

        def _resolve(ms_path, name, pol_id=0):
            """Stand in for the casacore-backed resolver in tabascal.ms."""
            captured["resolved"] = name
            captured["pol_id"] = pol_id
            return corr_idx

        monkeypatch.setattr(write_mod, "xds_to_table", _capture)
        monkeypatch.setattr(ms_mod, "resolve_correlation", _resolve)

        if not emit:
            # The caltable export runs at the end of every write and needs a real
            # MS on disk to copy subtables out of, which these in-memory cases do
            # not have. Stubbed by default so the column assertions stay about the
            # columns; the export has its own tests, which pass ``emit=True``.
            def _no_export(ms, zarr, out_path=None, gain_table=None):
                captured["caltable"] = {
                    "ms_path": ms,
                    "results_zarr_path": zarr,
                    "out_path": out_path,
                    "gain_table": gain_table,
                }
                return None

            monkeypatch.setattr(write_mod, "write_gain_caltable", _no_export)

        write_results_ms(ms_path, zarr_path, corr=corr, gain_table=gain_table)

        values = {
            col: np.asarray(captured["xds"][col].data) for col in captured["cols"]
        }

        return values, captured

    return _run


def _ms_skeleton(tmp_path, name="skeleton.ms", n_spw=1):
    """A real on-disk MS holding only the subtables a caltable copies out of one.

    The writer itself still runs against the in-memory stand-in; it is
    ``write_caltable`` that needs a real table on disk, because the caltable
    carries a copy of the MS's ``ANTENNA`` and ``SPECTRAL_WINDOW``. The two
    describe the same observation -- ``N_ANT`` antennas on ``FREQS`` -- which is
    what the export validates itself against before it writes anything.

    ``n_spw`` above 1 makes an MS the writer serves one partition of happily and
    a caltable cannot describe at all.
    """

    tables = pytest.importorskip("casacore.tables")

    path = str(tmp_path / name)

    main = tables.table(
        path,
        tables.maketabdesc([tables.makescacoldesc("TIME", 0.0, valuetype="double")]),
        nrow=1,
        ack=False,
    )

    ant = tables.table(
        os.path.join(path, "ANTENNA"),
        tables.maketabdesc(
            [tables.makearrcoldesc("POSITION", 0.0, ndim=1, valuetype="double")]
        ),
        nrow=N_ANT,
        ack=False,
    )
    ant.putcol("POSITION", np.zeros((N_ANT, 3)))
    ant.close()

    spw = tables.table(
        os.path.join(path, "SPECTRAL_WINDOW"),
        tables.maketabdesc(
            [tables.makearrcoldesc("CHAN_FREQ", 0.0, ndim=1, valuetype="double")]
        ),
        nrow=n_spw,
        ack=False,
    )
    spw.putcol("CHAN_FREQ", np.tile(FREQS, (n_spw, 1)))
    spw.close()

    main.close()

    return path


@pytest.fixture
def ms_skeleton(tmp_path):
    """A real single-spectral-window MS to write caltables from."""

    return _ms_skeleton(tmp_path)


def _caltable_path(zarr_path: str) -> str:
    """Where the export puts its table: the results path with a ``.B`` extension."""

    return os.path.splitext(zarr_path)[0] + ".B"


def _uniform_gains(n_sample: int, value=1.0):
    """``UnitaryGains``-shaped gains: the same value on every antenna."""

    return np.full((n_sample, N_ANT, N_FREQ, N_TIME), value, dtype=np.complex128)


def _fitted_gains_bl(gains):
    """The mean baseline gain the writer divides by, in the writer's precision.

    The zarr's gains are cast to ``complex64`` *before* the baseline product and
    the sample mean, so a reference that casts afterwards differs in the last
    bit -- which matters wherever a column is asserted exactly.
    """

    g = np.asarray(gains).astype(np.complex64)

    return _to_ms((g[:, A1_BL] * g[:, A2_BL].conj()).mean(axis=0))


def _near(model_ms, seed: int = 7, frac: float = 0.02):
    """MS ``DATA``: the model with a small per-component perturbation.

    The closure identity is exact floating point only where the residual is
    small next to the data -- the regime a fitted model is in, and the one #123
    verified end-to-end on EDA2. ``_observed`` scales its noise to the *largest*
    visibility in the array, which is what makes residual tolerances meaningful
    but swamps the smallest cells; there ``s + (c - s)`` genuinely rounds to a
    neighbouring float and the identity holds only to a ulp.

    Perturbed per component rather than by a complex factor: a multiplicative
    complex perturbation mixes the real and imaginary parts, so a cell whose
    real part is far smaller than its imaginary part would be moved by more than
    itself.
    """

    rng = np.random.default_rng(seed)
    shape = model_ms.shape

    return (
        model_ms.real * (1 + frac * rng.normal(size=shape))
        + 1j * model_ms.imag * (1 + frac * rng.normal(size=shape))
    ).astype(np.complex64)


def _ext_gains(seed: int = 21):
    """Per-antenna external gains, as a caltable would place them on the grid."""

    rng = np.random.default_rng(seed)
    shape = (N_ANT, N_FREQ, N_TIME)
    amp = rng.uniform(0.4, 2.5, shape)
    phase = rng.uniform(-np.pi, np.pi, shape)

    return (amp * np.exp(1j * phase)).astype(complex)


def _ext_bl(gains):
    """``g_p conj(g_q)`` of the external gains, as ``(row, chan, 1)`` complex64."""

    return _to_ms(
        (gains[A1_BL] * gains[A2_BL].conj()).astype(np.complex64)
    )


def _real_gains(n_sample: int = 1):
    """Real, non-unit, non-uniform fitted gains: a divisor that acts per component.

    Complex division mixes the real and imaginary parts, so with a complex gain
    the calibrated data's real part is not a scaled copy of the model's real
    part and Sterbenz's condition cannot be *arranged* per component -- only
    hoped for. Dividing by a real positive gain is two independent real
    divisions, which is what lets the exactness precondition be constructed and
    then checked. Still non-unit and antenna-dependent, so it is a real divisor
    and not a no-op.
    """

    amp = np.array([0.5, 1.0, 2.0, 1.5])
    one = np.broadcast_to(amp[:, None, None], (N_ANT, N_FREQ, N_TIME))

    return np.broadcast_to(one, (n_sample,) + one.shape).astype(np.complex128)


def _real_ext_gains(seed: int = 23):
    """External gains that are real and positive, for the same reason."""

    rng = np.random.default_rng(seed)

    return rng.uniform(0.4, 2.5, (N_ANT, N_FREQ, N_TIME)).astype(complex)


def _at_most_one_ulp(got, want):
    """``got`` within one float32 ulp of ``want``, per visibility.

    Referenced to the visibility's *magnitude*, which is the honest bound: the
    error in ``model + (data - model)`` is set by the size of the residual that
    was rounded, and the residual is a property of the complex number, not of
    whichever component happens to be small.
    """

    tol = np.spacing(np.abs(np.asarray(want)).astype(np.float32))
    worst = np.abs(np.asarray(got) - np.asarray(want))

    assert np.all(worst <= tol), (
        f"max |got - want| = {worst.max():.3e}, worst allowance {tol.max():.3e}"
    )


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

        assert captured["cols"] == COLS + WEIGHT_COLS
        assert captured["resolved"] == "yx"          # resolved by name, not index

        mean_gains_bl = _to_ms(gains_bl.mean(axis=0))
        model_ast = _to_ms(ast.mean(0))
        model_rfi = _to_ms(rfi.mean(0))
        # Every column in one frame: the data with the gains divided out.
        vis_cal = fit / mean_gains_bl

        kw = _tolerances(fit)
        got = {col: _fit(values[col], corr_idx) for col in COLS}

        np.testing.assert_allclose(got["CORRECTED_DATA"], vis_cal, **kw)
        np.testing.assert_allclose(got["TAB_AST_DATA"], model_ast, **kw)
        np.testing.assert_allclose(got["TAB_RFI_DATA"], model_rfi, **kw)
        np.testing.assert_allclose(got["TAB_AST_RES"], vis_cal - model_ast, **kw)
        np.testing.assert_allclose(got["TAB_RFI_RES"], vis_cal - model_rfi, **kw)
        np.testing.assert_allclose(
            got["TAB_RES_DATA"], vis_cal - (model_ast + model_rfi), **kw
        )

        for col in COLS:
            assert values[col].shape == (N_TIME * N_BL, N_FREQ, n_corr)

    def test_the_columns_are_written_in_the_ms_frame(self, tmp_path, run_writer):
        """Shape and dims, not just values: the transpose is easy to get wrong."""
        gains, ast, rfi = _model(1)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        xds_ms = _fake_ms(np.zeros((N_TIME * N_BL, N_FREQ, N_CORR), np.complex64))

        values, captured = run_writer(xds_ms, zarr_path)

        for col in COLS + ["WEIGHT_SPECTRUM"]:
            assert captured["xds"][col].dims == ("row", "chan", "corr")
            assert values[col].shape == (N_TIME * N_BL, N_FREQ, N_CORR)

        assert captured["xds"]["WEIGHT"].dims == ("row", "corr")
        assert values["WEIGHT"].shape == (N_TIME * N_BL, N_CORR)

        # Jy on the visibility columns only; WEIGHT and WEIGHT_SPECTRUM are
        # standard columns and are left with the units the MS gives them.
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


# ---------------------------------------------------------------------------
# The calibrated frame
# ---------------------------------------------------------------------------

class TestClosureIdentity:
    """``TAB_AST_DATA + TAB_RFI_DATA + TAB_RES_DATA == CORRECTED_DATA``.

    #123's anchor check, and the reason it is worth having: it is floating-point
    arithmetic on the written columns themselves, so it fails loudly if any one
    of them is written in a frame the others are not.

    It is *bit exact* where Sterbenz's condition holds per component -- the real
    parts within a factor of two of one another and the imaginary parts likewise
    -- and within one float32 ulp of the visibility's magnitude otherwise. The
    two cases are tested separately, because a residual that is small next to
    ``|data|`` does not establish the condition for a cell whose real part
    nearly cancels, and a test asserting exactness on random complex data is
    passing on its seed.
    """

    @staticmethod
    def _closed(values):
        return (
            values["TAB_AST_DATA"] + values["TAB_RFI_DATA"] + values["TAB_RES_DATA"]
        )

    @pytest.mark.parametrize("external", [False, True], ids=["no_table", "table"])
    @pytest.mark.parametrize("unitary", [False, True], ids=["fitted", "unitary"])
    @pytest.mark.parametrize(
        "n_corr, corr_idx", [(1, 0), (4, 2)], ids=["one_corr", "four_corr"]
    )
    def test_it_is_bit_exact_where_the_condition_holds(
        self, tmp_path, run_writer, external, unitary, n_corr, corr_idx
    ):
        """Constructed so the per-component condition provably holds, then checked."""
        _, ast, rfi = _model(1)
        gains = _uniform_gains(1) if unitary else _real_gains(1)
        ext = _real_ext_gains() if external else None

        stored = _baseline_gains(gains) * (ast + rfi)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi, vis_obs=stored, corr="yx")

        # The divisor the writer will use: real and positive, so the complex
        # division it performs is two independent real divisions.
        g_total = _to_ms(_baseline_gains(gains).mean(axis=0)).real
        if external:
            g_total = g_total * _to_ms(ext[A1_BL] * ext[A2_BL].conj()).real

        model = _to_ms((ast + rfi).mean(axis=0))
        rng = np.random.default_rng(3)

        # Each component of the data is its own model component times g_total,
        # perturbed by at most 20 %. The calibrated data's component then sits
        # within a factor of two of the model's, which is Sterbenz's condition.
        def perturbed(component):
            return g_total * component * (1 + 0.2 * rng.uniform(-1, 1, model.shape))

        fit = (perturbed(model.real) + 1j * perturbed(model.imag)).astype(np.complex64)
        data = _spread(fit, n_corr, corr_idx)

        values, _ = run_writer(
            _fake_ms(data), zarr_path, corr_idx=corr_idx, ext_gains=ext
        )

        calibrated = _fit(values["CORRECTED_DATA"], corr_idx)
        summed = _fit(values["TAB_AST_DATA"] + values["TAB_RFI_DATA"], corr_idx)

        # The precondition, checked rather than assumed: constructing it wrongly
        # would leave the exactness below passing by luck, which is exactly the
        # failure mode this test exists to remove.
        for part in (np.real, np.imag):
            model_part, data_part = part(summed), part(calibrated)
            assert np.all(model_part != 0)
            ratio = data_part / model_part
            assert np.all((ratio >= 0.5) & (ratio <= 2.0)), ratio

        np.testing.assert_array_equal(self._closed(values), values["CORRECTED_DATA"])

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize("external", [False, True], ids=["no_table", "table"])
    @pytest.mark.parametrize("unitary", [False, True], ids=["fitted", "unitary"])
    def test_a_good_fit_closes_to_within_one_ulp(
        self, tmp_path, run_writer, seed, external, unitary
    ):
        """Complex gains and a small residual: the general case, over five seeds.

        Nothing here arranges the per-component condition -- a complex divisor
        mixes the components -- so one ulp of the magnitude is the bound that is
        actually available, and it is asserted rather than exactness, which would
        be a property of the seed.
        """
        _, ast, rfi = _model(1)
        gains = _uniform_gains(1) if unitary else _model(1)[0]
        ext = _ext_gains(seed=21 + seed) if external else None

        stored = _baseline_gains(gains) * (ast + rfi)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi, vis_obs=stored, corr="yx")

        gained = stored.mean(axis=0)
        if external:
            gained = gained * (ext[A1_BL] * ext[A2_BL].conj())
        data = _near(_to_ms(gained), seed=seed)

        values, _ = run_writer(_fake_ms(data), zarr_path, ext_gains=ext)

        _at_most_one_ulp(self._closed(values), values["CORRECTED_DATA"])

    def test_a_bad_fit_still_closes_to_float32_round_off(self, tmp_path, run_writer):
        """Where a cell is all residual the bound widens, but only to round-off.

        ``s + (c - s)`` reconstructs ``c`` exactly only when ``c - s`` is exact.
        Here the residual is the size of the visibility, so the subtraction
        rounds and the sum lands a few float32 steps away. It is worth saying
        out loud that this is the whole of the degradation: a column in the wrong
        frame is off by the gain, not by round-off.
        """
        gains, ast, rfi = _model(1)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi, corr="yx")
        # Noise scaled to the loudest visibility, so the quietest cells are all
        # residual.
        data = _observed(_to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)))

        values, _ = run_writer(_fake_ms(data), zarr_path)

        np.testing.assert_allclose(
            self._closed(values),
            values["CORRECTED_DATA"],
            rtol=0,
            atol=1e-6 * float(np.abs(values["CORRECTED_DATA"]).max()),
        )

    def test_it_closes_on_the_unfitted_correlations_too(self, tmp_path, run_writer):
        """The models are zero there and the data passes through, so it must."""
        gains, ast, rfi = _model(1)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi, corr="yx")
        fit = _observed(_to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)))
        data = _spread(fit, 4, 2)

        values, _ = run_writer(_fake_ms(data), zarr_path, corr_idx=2)

        closed = (
            values["TAB_AST_DATA"] + values["TAB_RFI_DATA"] + values["TAB_RES_DATA"]
        )

        np.testing.assert_array_equal(
            _others(closed, 2), _others(values["CORRECTED_DATA"], 2)
        )
        np.testing.assert_array_equal(_others(closed, 2), _others(data, 2))


class TestBothGainLayersComeOff:
    """``CORRECTED_DATA = V / (g_ext g_fit)``: the external table and the fit."""

    def test_the_external_and_fitted_gains_are_both_divided_out(
        self, tmp_path, run_writer
    ):
        gains, ast, rfi = _model(1)
        ext = _ext_gains()

        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        ext_bl = _ext_bl(ext)
        fit = _observed(
            _to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)) * ext_bl
        )

        values, _ = run_writer(_fake_ms(fit), zarr_path, ext_gains=ext)

        gains_tot = ext_bl * _to_ms(_baseline_gains(gains).mean(axis=0))

        np.testing.assert_allclose(
            values["CORRECTED_DATA"], fit / gains_tot, **_tolerances(fit)
        )
        # Removing only the fitted layer leaves the whole external gain behind.
        assert not np.allclose(
            values["CORRECTED_DATA"],
            fit / _to_ms(_baseline_gains(gains).mean(axis=0)),
            rtol=1e-3,
        )

    def test_the_models_stay_where_the_run_left_them(self, tmp_path, run_writer):
        """The zarr's models are already in the fully calibrated frame."""
        gains, ast, rfi = _model(1)
        ext = _ext_gains()

        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        fit = _observed(
            _to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)) * _ext_bl(ext)
        )

        values, _ = run_writer(_fake_ms(fit), zarr_path, ext_gains=ext)

        kw = _tolerances(fit)
        np.testing.assert_allclose(values["TAB_AST_DATA"], _to_ms(ast.mean(0)), **kw)
        np.testing.assert_allclose(values["TAB_RFI_DATA"], _to_ms(rfi.mean(0)), **kw)

    def test_the_residuals_are_against_the_calibrated_data(self, tmp_path, run_writer):
        gains, ast, rfi = _model(1)
        ext = _ext_gains()

        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        ext_bl = _ext_bl(ext)
        fit = _observed(
            _to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)) * ext_bl
        )

        values, _ = run_writer(_fake_ms(fit), zarr_path, ext_gains=ext)

        vis_cal = fit / (ext_bl * _to_ms(_baseline_gains(gains).mean(axis=0)))
        kw = _tolerances(fit)

        np.testing.assert_allclose(
            values["TAB_AST_RES"], vis_cal - _to_ms(ast.mean(0)), **kw
        )
        np.testing.assert_allclose(
            values["TAB_RFI_RES"], vis_cal - _to_ms(rfi.mean(0)), **kw
        )
        # The superseded #134 residual: the data with a re-gained model taken off.
        assert not np.allclose(
            values["TAB_AST_RES"],
            fit - _to_ms((_baseline_gains(gains) * ast).mean(axis=0)),
            rtol=1e-3,
        )

    def test_the_table_is_placed_on_the_runs_own_grid(self, tmp_path, run_writer):
        """Same times, channels and antenna count the reader used, or another frame.

        The external gains are re-derived here from the table rather than read
        off the results, so they are only the *same* gains if they are asked for
        on the same grid.
        """
        gains, ast, rfi = _model(1)
        ext = _ext_gains()
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        fit = _observed(_to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)))

        table = tmp_path / "second.B0"
        table.mkdir()

        _, captured = run_writer(
            _fake_ms(fit), zarr_path, ext_gains=ext, gain_table=[str(table)]
        )

        assert captured["ext"]["gain_table"] == [str(table)]
        # The MS TIME column, one value per timestep block, in seconds.
        np.testing.assert_array_equal(
            captured["ext"]["times"], np.arange(N_TIME, dtype=float)
        )
        np.testing.assert_array_equal(captured["ext"]["freqs"], FREQS)
        assert captured["ext"]["n_ant"] == N_ANT

    def test_an_unsolved_antenna_is_left_uncalibrated_not_blanked(
        self, tmp_path, run_writer
    ):
        gains, ast, rfi = _model(1)
        ext = _ext_gains()
        dead = np.zeros(ext.shape, dtype=bool)
        dead[1] = True

        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        fit = _observed(_to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)))

        values, _ = run_writer(
            _fake_ms(fit), zarr_path, ext_gains=ext, ext_dead=dead
        )

        assert np.all(np.isfinite(values["CORRECTED_DATA"]))

        # On a baseline touching antenna 1 only the fitted gain comes off.
        touched = np.tile((A1_BL == 1) | (A2_BL == 1), N_TIME)
        fitted_only = fit / _to_ms(_baseline_gains(gains).mean(axis=0))
        np.testing.assert_allclose(
            values["CORRECTED_DATA"][touched],
            fitted_only[touched],
            **_tolerances(fit),
        )

    def test_without_a_table_only_the_fitted_gains_come_off(
        self, tmp_path, run_writer
    ):
        gains, ast, rfi = _model(1)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        fit = _observed(_to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)))

        values, captured = run_writer(_fake_ms(fit), zarr_path)

        assert "ext" not in captured
        np.testing.assert_allclose(
            values["CORRECTED_DATA"],
            fit / _to_ms(_baseline_gains(gains).mean(axis=0)),
            **_tolerances(fit),
        )


class TestUnityGainsReproduceTheOldBehaviour:
    """The regression guard: unity gains and no table must move nothing.

    Under unity gains the calibrated frame and the data frame are the same
    numbers, so the columns are compared against the *superseded* expressions --
    bit for bit, since dividing by and multiplying with an exact ``1 + 0j`` is
    exact. This is what keeps the pipeline references from moving.
    """

    @pytest.fixture
    def written(self, tmp_path, run_writer):
        _, ast, rfi = _model(1)
        gains = _uniform_gains(1)
        stored = _baseline_gains(gains) * (ast + rfi)

        zarr_path = _write_zarr(tmp_path, gains, ast, rfi, vis_obs=stored)
        data = _observed(_to_ms(stored.mean(axis=0)))

        values, _ = run_writer(_fake_ms(data), zarr_path)

        # complex64, the precision the writer casts the models to: the point is
        # that the *values* did not move, not that the dtype did.
        return (
            values,
            data,
            _to_ms(ast.mean(0)).astype(np.complex64),
            _to_ms(rfi.mean(0)).astype(np.complex64),
        )

    def test_corrected_data_is_the_data(self, written):
        values, data, _, _ = written

        np.testing.assert_array_equal(values["CORRECTED_DATA"], data)

    def test_the_residuals_are_the_data_frame_residuals(self, written):
        values, data, ast, rfi = written

        np.testing.assert_array_equal(values["TAB_AST_RES"], data - ast)
        np.testing.assert_array_equal(values["TAB_RFI_RES"], data - rfi)
        np.testing.assert_array_equal(values["TAB_RES_DATA"], data - (ast + rfi))

    def test_the_model_columns_are_the_models(self, written):
        values, _, ast, rfi = written

        np.testing.assert_array_equal(values["TAB_AST_DATA"], ast)
        np.testing.assert_array_equal(values["TAB_RFI_DATA"], rfi)

    def test_the_weights_are_the_ms_inverse_variance(self, written):
        """``|1|^2 / SIGMA^2``: the MS's own weights, unchanged by calibration."""
        values, _, _, _ = written

        expected = (1.0 / _sigma_grid(0) ** 2).astype(np.float32)

        np.testing.assert_array_equal(
            values["WEIGHT_SPECTRUM"], _to_ms(expected)
        )


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

class TestWeights:
    """``WEIGHT_SPECTRUM = |g_total|^2 / SIGMA^2``, and ``WEIGHT`` its band mean."""

    def _reference(self, gains_tot_ms, sigma_grid):
        """The weight, written out longhand in the order the writer forms it."""

        sigma = _to_ms(sigma_grid.astype(np.float64))

        return np.abs(gains_tot_ms).astype(np.float64) ** 2 / sigma**2

    @pytest.fixture
    def run(self, tmp_path, run_writer):
        def _go(*, noise="sigma", external=False, n_corr=1, corr_idx=0):
            gains, ast, rfi = _model(1)
            ext = _ext_gains() if external else None

            zarr_path = _write_zarr(tmp_path, gains, ast, rfi, corr="yx")
            gained = (_baseline_gains(gains) * (ast + rfi)).mean(axis=0)
            model_ms = _to_ms(gained)
            if external:
                model_ms = model_ms * _ext_bl(ext)
            data = _spread(_observed(model_ms), n_corr, corr_idx)

            values, captured = run_writer(
                _fake_ms(data, noise=noise),
                zarr_path,
                corr_idx=corr_idx,
                ext_gains=ext,
            )

            gains_tot = _fitted_gains_bl(gains)
            if external:
                gains_tot = _ext_bl(ext) * gains_tot

            return values, captured, gains_tot

        return _go

    def test_weight_spectrum_is_the_calibrated_inverse_variance(self, run):
        values, _, gains_tot = run()

        expected = self._reference(gains_tot, _sigma_grid(0)).astype(np.float32)

        np.testing.assert_array_equal(values["WEIGHT_SPECTRUM"], expected)
        assert values["WEIGHT_SPECTRUM"].dtype == np.float32

    def test_weight_is_the_frequency_mean_of_the_spectrum(self, run):
        values, _, gains_tot = run()

        expected = self._reference(gains_tot, _sigma_grid(0))

        np.testing.assert_array_equal(
            values["WEIGHT"], expected.mean(axis=1).astype(np.float32)
        )
        # And it really is the mean of what was written, to float32 round-off.
        np.testing.assert_allclose(
            values["WEIGHT"], values["WEIGHT_SPECTRUM"].mean(axis=1), rtol=1e-6
        )

    def test_the_external_gain_raises_the_weight(self, run):
        """Both layers of the divisor, or the weight describes another frame."""
        with_table, _, gains_tot = run(external=True)
        without, _, fitted_only = run()

        np.testing.assert_array_equal(
            with_table["WEIGHT_SPECTRUM"],
            self._reference(gains_tot, _sigma_grid(0)).astype(np.float32),
        )
        assert not np.allclose(
            with_table["WEIGHT_SPECTRUM"], without["WEIGHT_SPECTRUM"], rtol=1e-3
        )

    def test_sigma_spectrum_is_preferred_over_sigma(self, run):
        """A bandpass is not flat; the frequency-resolved column says so."""
        both, _, gains_tot = run(noise="both")

        np.testing.assert_array_equal(
            both["WEIGHT_SPECTRUM"],
            self._reference(gains_tot, _sigma_grid(0, per_chan=True)).astype(
                np.float32
            ),
        )
        # The channel factors differ, so the two columns cannot coincide.
        flat, _, _ = run(noise="sigma")
        assert not np.allclose(both["WEIGHT_SPECTRUM"], flat["WEIGHT_SPECTRUM"])

    def test_the_noise_is_read_on_the_fitted_correlation(self, run):
        """SIGMA[:, 0] would weight the fit by another polarisation's noise."""
        values, _, gains_tot = run(n_corr=4, corr_idx=2)

        expected = self._reference(gains_tot, _sigma_grid(2)).astype(np.float32)

        np.testing.assert_array_equal(
            _fit(values["WEIGHT_SPECTRUM"], 2), expected
        )
        assert not np.allclose(
            _fit(values["WEIGHT_SPECTRUM"], 2),
            self._reference(gains_tot, _sigma_grid(0)).astype(np.float32),
        )

    def test_unfitted_correlations_carry_no_weight(self, run):
        """Zero, matching the zeroed model columns: nothing was fitted there."""
        values, _, _ = run(n_corr=4, corr_idx=2)

        keep = [c for c in range(4) if c != 2]

        np.testing.assert_array_equal(_others(values["WEIGHT_SPECTRUM"], 2), 0.0)
        np.testing.assert_array_equal(values["WEIGHT"][:, keep], 0.0)
        assert not np.allclose(_fit(values["WEIGHT_SPECTRUM"], 2), 0.0)
        assert not np.allclose(values["WEIGHT"][:, 2], 0.0)

    def test_an_ms_with_no_noise_column_gets_no_weights(self, tmp_path, run_writer):
        """Nothing is invented: the weights are simply not written."""
        gains, ast, rfi = _model(1)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        data = _observed(_to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)))

        values, captured = run_writer(_fake_ms(data, noise=None), zarr_path)

        assert captured["cols"] == COLS
        assert "WEIGHT_SPECTRUM" not in values


class TestReductionOrder:
    """``E[g_p conj(g_q)]`` is formed per sample, not from the two mean gains."""

    def test_mean_then_multiply_would_disagree(self, tmp_path, run_writer):
        gains, ast, rfi = _model(2)
        gains_bl = _baseline_gains(gains)

        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        data = _observed(_to_ms((gains_bl * (ast + rfi)).mean(axis=0)))

        values, _ = run_writer(_fake_ms(data), zarr_path)

        naive_gain = _to_ms(
            gains.mean(axis=0)[A1_BL] * np.conj(gains.mean(axis=0)[A2_BL])
        )

        np.testing.assert_allclose(
            values["CORRECTED_DATA"],
            data / _to_ms(gains_bl.mean(axis=0)),
            **_tolerances(data),
        )
        assert not np.allclose(
            values["CORRECTED_DATA"], data / naive_gain, rtol=1e-3
        )
        # One divisor for the whole frame, so the same reduction order reaches
        # every residual column too.
        assert not np.allclose(
            values["TAB_AST_RES"],
            data / naive_gain - _to_ms(ast.mean(axis=0)),
            rtol=1e-3,
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


class TestTheTotalModelIsTheSumOfTheWrittenModels:
    """In one frame there is one total: ``TAB_AST_DATA + TAB_RFI_DATA``.

    The zarr's ``vis_obs`` is the forward model in the *gained* frame, and a
    component that gains only one of the two terms leaves it in no single frame
    at all -- ``ast + rfi / g``, which neither model column is written in. So
    the calibrated total is the sum of the two calibrated model columns, which
    is also what makes the closure identity exact rather than approximate.
    """

    def test_a_gains_component_that_gains_only_one_term(self, tmp_path, run_writer):
        """See the commented-out variant in components/gains.py."""
        gains, ast, rfi = _model(1)
        gains_bl = _baseline_gains(gains)

        stored = gains_bl * ast + rfi
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi, vis_obs=stored)
        # Data consistent with the calibrated-frame model -- the frame the two
        # written model columns are in -- so the closure identity is exact and
        # the stored one-term-gained total is visibly not what was subtracted.
        data = _near(_to_ms((gains_bl * (ast + rfi)).mean(axis=0)))

        values, _ = run_writer(_fake_ms(data), zarr_path)

        vis_cal = data / _to_ms(gains_bl.mean(axis=0))

        np.testing.assert_allclose(
            values["TAB_RES_DATA"],
            vis_cal - (_to_ms(ast.mean(axis=0)) + _to_ms(rfi.mean(axis=0))),
            **_tolerances(data),
        )
        _at_most_one_ulp(
            values["TAB_AST_DATA"] + values["TAB_RFI_DATA"] + values["TAB_RES_DATA"],
            values["CORRECTED_DATA"],
        )

        # The stored forward model brought into the calibrated frame is a
        # different quantity here, and the decomposition would not close on it.
        stored_cal = _to_ms((stored / gains_bl).mean(axis=0))
        assert not np.allclose(
            values["TAB_RES_DATA"], vis_cal - stored_cal, rtol=1e-3
        )


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

    def test_the_data_columns_pass_the_data_through(self, run):
        """Uncalibrated and unsubtracted, which is what those columns mean there."""
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

    def test_the_name_is_resolved_on_the_partition_polarization_row(self, run):
        """The partition's DATA_DESC_ID picks the POLARIZATION row, not row 0.

        Resolving against row 0 would place the results in the wrong
        polarisation on any MS whose partition uses another row.
        """
        xds_ms, zarr_path, _, run_writer, corr, idx = run()
        xds_ms = xds_ms.assign_attrs(DATA_DESC_ID=1)

        _, captured = run_writer(
            xds_ms, zarr_path, corr=corr, corr_idx=idx, pol_id=2
        )

        assert captured["data_desc_id"] == 1
        assert captured["pol_id"] == 2

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
    """Whether the results describe *this* MS is the writer's question.

    The MS's own layout is validated in ``tabascal.ms``; only the comparison
    against the results lives here.
    """

    def test_fewer_baselines_in_the_results_than_the_ms(self, tmp_path, run_writer):
        """Three baselines in the results, six in the MS."""
        gains, ast, rfi = _model(1)
        gains = gains[:, :3]
        ast, rfi = ast[:, :3], rfi[:, :3]
        vis_obs = np.zeros_like(ast)

        zarr_path = _write_zarr(tmp_path, gains, ast, rfi, vis_obs=vis_obs)
        xds_ms = _fake_ms(np.zeros((N_TIME * N_BL, N_FREQ, N_CORR), np.complex64))

        with pytest.raises(ValueError, match="does not belong"):
            run_writer(xds_ms, zarr_path)

    def test_more_baselines_in_the_results_than_the_ms(self, tmp_path, run_writer):
        """The MS is perfectly time-major; it is simply a different array."""
        gains, ast, rfi = _model(1)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)

        # Three antennas, so three baselines against the results' six.
        keep = np.flatnonzero((A1_BL < 3) & (A2_BL < 3))
        rows = np.concatenate([keep + t * N_BL for t in range(N_TIME)])
        xds_ms = _fake_ms(
            np.zeros((len(rows), N_FREQ, N_CORR), np.complex64),
            a1=np.tile(A1_BL[keep], N_TIME),
            a2=np.tile(A2_BL[keep], N_TIME),
            times=np.repeat(np.arange(N_TIME, dtype=float), len(keep)),
        )

        with pytest.raises(ValueError) as excinfo:
            run_writer(xds_ms, zarr_path)

        message = str(excinfo.value)
        # The guard's own diagnosis, naming both counts -- not dask's downstream
        # "chunks do not add up to shape", which this would otherwise reach.
        assert "does not belong" in message
        assert "6 baselines" in message and "has 3" in message
        assert "time-major" not in message


class TestChannelCountGuard:
    """A run narrowed with ``data.freq`` reaches a refusal, not a reshape.

    The results then cover part of the MS's band. Which part is not what is
    missing -- the zarr's ``freq`` coordinate records the fitted channels -- the
    writer simply has no path from a partial band onto the MS's channel axis.
    Placing one is future work; until then it is refused, and refused where
    every other mismatch is: before the first column is built, rather than
    deep inside dask with a chunk-arithmetic error naming neither count.
    """

    def _narrowed(self, tmp_path):
        """A results zarr on one channel of the two-channel fake MS."""

        gains, ast, rfi = _model(1)

        return _write_zarr(
            tmp_path, gains[:, :, :1], ast[:, :, :1], rfi[:, :, :1]
        )

    def test_fewer_channels_in_the_results_than_the_ms(self, tmp_path, run_writer):
        zarr_path = self._narrowed(tmp_path)
        xds_ms = _fake_ms(np.zeros((N_TIME * N_BL, N_FREQ, N_CORR), np.complex64))

        with pytest.raises(ValueError) as excinfo:
            run_writer(xds_ms, zarr_path)

        message = str(excinfo.value)
        # Both counts, and the limitation named as one: an export that is not
        # supported yet reads differently from a zarr that is simply wrong.
        assert "1 channels" in message and "has 2" in message
        assert "data.freq" in message
        assert "not yet supported" in message
        # And named accurately. The zarr does say which channels were fitted;
        # the writer is what does not yet place them, and a message blaming the
        # results would send a reader looking for a coordinate that is there.
        assert "freq coordinate" in message

    def test_no_column_is_built_and_the_ms_is_untouched(
        self, tmp_path, monkeypatch, run_writer
    ):
        """Nothing is computed and nothing is written: the guard is first."""

        def _no_columns(*args, **kwargs):
            raise AssertionError("a column was built before the guard ran")

        monkeypatch.setattr(write_mod, "_to_ms_column", _no_columns)

        zarr_path = self._narrowed(tmp_path)
        data = (
            np.arange(N_TIME * N_BL * N_FREQ * N_CORR)
            .reshape(N_TIME * N_BL, N_FREQ, N_CORR)
            .astype(np.complex64)
        )
        xds_ms = _fake_ms(data)
        before = set(xds_ms.data_vars)

        with pytest.raises(ValueError, match="not yet supported"):
            run_writer(xds_ms, zarr_path)

        assert set(xds_ms.data_vars) == before
        np.testing.assert_array_equal(np.asarray(xds_ms["DATA"].data), data)


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
        model_ast = _to_ms(ast.mean(axis=0))
        model_rfi = _to_ms(rfi.mean(axis=0))
        # The substituted gain is the divisor, so the whole frame follows it.
        vis_cal = data / _to_ms(gains_bl.mean(axis=0))

        np.testing.assert_allclose(values["CORRECTED_DATA"], vis_cal, **kw)
        np.testing.assert_allclose(values["TAB_AST_RES"], vis_cal - model_ast, **kw)
        np.testing.assert_allclose(values["TAB_RFI_RES"], vis_cal - model_rfi, **kw)
        np.testing.assert_allclose(
            values["TAB_RES_DATA"], vis_cal - (model_ast + model_rfi), **kw
        )

        # The zero the run stored in vis_obs cannot reach any column.
        assert not np.allclose(
            values["TAB_RES_DATA"], vis_cal - _to_ms(stored.mean(axis=0)), rtol=1e-3
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

    def test_the_stored_total_never_reaches_a_column(self, tmp_path, run_writer):
        """The stored model carries the zero gain; the written columns must not.

        The calibrated total is the sum of the two model columns, so a dead
        antenna's zero in the run's own ``vis_obs`` has nothing to leak into --
        the guard the superseded per-sample fallback used to provide.
        """
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
        vis_cal = data / _to_ms(gains_bl.mean(axis=0))

        np.testing.assert_allclose(
            values["TAB_RES_DATA"],
            vis_cal - (_to_ms(ast.mean(axis=0)) + _to_ms(rfi.mean(axis=0))),
            **_tolerances(data),
        )

        # The stored model is zero on those baselines, so a residual formed
        # against it would just be the calibrated data.
        assert not np.allclose(
            values["TAB_RES_DATA"][touched],
            (vis_cal - _to_ms(stored.mean(axis=0)))[touched],
            rtol=1e-3,
        )
        assert np.all(np.isfinite(values["TAB_RES_DATA"]))

    def test_the_weights_follow_the_substituted_gain(self, tmp_path, run_writer):
        """The weight describes the frame that was written, substitution and all."""
        gains, ast, rfi = _model(1)
        gains = gains.copy()
        gains[:, 2] = 0.0

        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        gains_bl = _baseline_gains(_substitute(gains))
        data = _observed(_to_ms((gains_bl * (ast + rfi)).mean(axis=0)))

        with pytest.warns(RuntimeWarning):
            values, _ = run_writer(_fake_ms(data), zarr_path)

        expected = (
            np.abs(_to_ms(gains_bl.mean(axis=0).astype(np.complex64))).astype(
                np.float64
            )
            ** 2
            / _to_ms(_sigma_grid(0).astype(np.float64)) ** 2
        ).astype(np.float32)

        np.testing.assert_array_equal(values["WEIGHT_SPECTRUM"], expected)
        assert np.all(np.isfinite(values["WEIGHT_SPECTRUM"]))

    def test_the_columns_are_written_before_the_warning(self, tmp_path, monkeypatch):
        """The counts come out of the write's own compute, so the warning follows it.

        Pinned because it is observable: promoting RuntimeWarning to an error
        raises after the MS has been written, not instead of writing it. Wired
        by hand rather than through run_writer so the write call itself can be
        seen even though the warning escapes as an exception.
        """
        import tabascal.ms as ms_mod

        gains, ast, rfi = _model(1)
        gains[0, 2] = 0.0
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        data = _observed(_to_ms((_baseline_gains(_substitute(gains)) * (ast + rfi)).mean(axis=0)))
        xds_ms = _fake_ms(data)

        written = []
        monkeypatch.setattr(
            write_mod,
            "xds_from_ms",
            lambda path, column_keywords=False: (
                ([xds_ms], {}) if column_keywords else [xds_ms]
            ),
        )
        monkeypatch.setattr(
            write_mod, "xds_to_table",
            lambda datasets, path, cols, column_keywords=None: written.append(cols) or [],
        )
        monkeypatch.setattr(ms_mod, "resolve_data_description", lambda ms_path, ddid=0: (0, 0))
        monkeypatch.setattr(ms_mod, "resolve_correlation", lambda ms_path, name, pol_id=0: 0)
        # The export runs after this warning whatever the warning does, and there
        # is no MS on disk here for it to run against; it is the subject of its
        # own tests above, not of this one.
        monkeypatch.setattr(write_mod, "write_gain_caltable", lambda *a, **kw: None)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with pytest.raises(RuntimeWarning, match="Affected antennas"):
                write_results_ms("unused.ms", zarr_path)

        assert written, "the columns were handed to xds_to_table before the warning was raised"

    def test_finite_gains_raise_no_warning(self, tmp_path, run_writer):
        gains, ast, rfi = _model(1)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        data = _observed(_to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)))

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            values, _ = run_writer(_fake_ms(data), zarr_path)

        assert np.all(np.isfinite(values["CORRECTED_DATA"]))


# ---------------------------------------------------------------------------
# The calibration table written beside the results
# ---------------------------------------------------------------------------

class TestTheExportRuns:
    """It is the last thing the writer does, and it cannot undo the rest."""

    def test_it_is_handed_the_ms_the_results_and_the_same_tables(
        self, tmp_path, run_writer
    ):
        """The same MS, the same zarr and the same external tables, in order.

        The export composes the total calibration itself, so it has to be given
        the layer the columns were divided by; a different list would emit a
        table that is not the calibration the columns are in the frame of.
        """
        gains, ast, rfi = _model(1)
        ext = _ext_gains()
        table = tmp_path / "flux.B0"
        table.mkdir()

        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        data = _observed(
            _to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)) * _ext_bl(ext)
        )

        _, captured = run_writer(
            _fake_ms(data),
            zarr_path,
            ext_gains=ext,
            gain_table=[str(table)],
            ms_path="real.ms",
        )

        assert captured["caltable"] == {
            "ms_path": "real.ms",
            "results_zarr_path": zarr_path,
            "out_path": None,
            "gain_table": [str(table)],
        }

    def _columns_are_intact(self, values, gains, data):
        """Every column written, and written correctly, export or no export."""

        assert set(COLS).issubset(values)
        np.testing.assert_allclose(
            values["CORRECTED_DATA"],
            data / _to_ms(_baseline_gains(gains).mean(axis=0)),
            **_tolerances(data),
        )

    def test_a_multi_window_ms_warns_and_keeps_the_columns(self, tmp_path, run_writer):
        """The real failure this demotion exists for.

        The writer serves one partition of a multi-spectral-window MS happily,
        and ``write_caltable`` refuses the MS outright, because a caltable files
        every row under one window's id. A completed run must not end in a
        traceback over a table it could not have written.
        """
        ms_path = _ms_skeleton(tmp_path, name="two_windows.ms", n_spw=2)

        gains, ast, rfi = _model(1)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        data = _observed(_to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)))

        with pytest.warns(RuntimeWarning, match="spectral window"):
            values, _ = run_writer(
                _fake_ms(data), zarr_path, ms_path=ms_path, emit=True
            )

        self._columns_are_intact(values, gains, data)
        assert not os.path.exists(_caltable_path(zarr_path))

    def test_an_output_inside_the_ms_warns_and_leaves_the_ms_alone(
        self, tmp_path, run_writer, ms_skeleton
    ):
        """#157's overlap guard fires, and the MS it protects is still there.

        Results written inside the MS put the table there too, which is the
        milder half of the guard: the export would be writing into the very
        subtables it copies from.
        """
        gains, ast, rfi = _model(1)
        zarr_path = _write_zarr(
            tmp_path, gains, ast, rfi, name=os.path.join("skeleton.ms", "results.zarr")
        )
        data = _observed(_to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)))

        with pytest.warns(RuntimeWarning, match="inside the Measurement Set"):
            values, _ = run_writer(
                _fake_ms(data), zarr_path, ms_path=ms_skeleton, emit=True
            )

        self._columns_are_intact(values, gains, data)
        # The observation the guard is about, still readable.
        tables = pytest.importorskip("casacore.tables")
        with tables.table(os.path.join(ms_skeleton, "ANTENNA"), ack=False) as ant:
            assert ant.nrows() == N_ANT

    def test_a_failed_export_removes_the_superseded_table(
        self, tmp_path, run_writer, ms_skeleton
    ):
        """A failure before the write is another way to leave a stale table.

        The no-op paths remove the previous run's table because it would read as
        this run's solution; a run whose export *fails* is in exactly that
        position, and the table it could not replace is removed there too.
        """
        gains, ast, rfi = _model(1)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        data = _observed(_to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)))
        run_writer(_fake_ms(data), zarr_path, ms_path=ms_skeleton, emit=True)

        assert os.path.exists(_caltable_path(zarr_path))

        # The same results, re-fitted, on an MS no caltable can describe.
        two_windows = _ms_skeleton(tmp_path, name="two_windows.ms", n_spw=2)
        again = _write_zarr(tmp_path, 2 * gains, ast, rfi)

        with pytest.warns(RuntimeWarning) as record:
            run_writer(_fake_ms(data), again, ms_path=two_windows, emit=True)

        # By category, not by position: the skeleton MS declares no MEASINFO Ref,
        # so the reader's own UserWarning shares the record with this one.
        message = str(record.pop(RuntimeWarning).message)

        assert "spectral window" in message
        assert "superseded" in message
        assert not os.path.exists(_caltable_path(zarr_path))

    def test_a_directory_in_the_way_is_neither_written_nor_removed(
        self, tmp_path, run_writer, ms_skeleton
    ):
        """The export names its own output; it does not own everything so named.

        ``results.zarr`` gives ``results.B``, and if a directory of the caller's
        is already there it is left exactly as it is -- neither overwritten by
        the write nor swept up by the clean-up afterwards.
        """
        gains, ast, rfi = _model(1)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        data = _observed(_to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)))

        in_the_way = _caltable_path(zarr_path)
        os.makedirs(in_the_way)
        sentinel = os.path.join(in_the_way, "mine.txt")
        with open(sentinel, "w") as f:
            f.write("keep me")

        with pytest.warns(RuntimeWarning, match="not a calibration table"):
            values, _ = run_writer(
                _fake_ms(data), zarr_path, ms_path=ms_skeleton, emit=True
            )

        self._columns_are_intact(values, gains, data)

        with open(sentinel) as f:
            assert f.read() == "keep me"

    def test_it_runs_even_when_the_gain_warnings_are_errors(
        self, tmp_path, run_writer, ms_skeleton
    ):
        """A filter promoting ``RuntimeWarning`` must not strand the old table.

        The bad-gain warnings are raised after the columns are written, so a
        process that promotes them to errors stops there -- with the columns
        already updated and the previous run's table still beside them, under the
        current name, describing gains that are no longer in the results. The
        export runs in a ``finally`` for exactly that: the error still comes out,
        and the table on disk is this run's.
        """
        gains, ast, rfi = _model(1)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        data = _observed(_to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)))
        run_writer(_fake_ms(data), zarr_path, ms_path=ms_skeleton, emit=True)

        stale = read_caltable(_caltable_path(zarr_path))["gains"]
        assert np.all(np.isfinite(stale))

        # Re-fitted, and this time the fit killed an antenna.
        dead = gains.copy()
        dead[:, 2] = 0.0
        again = _write_zarr(tmp_path, dead, ast, rfi)
        data = _observed(_to_ms((_baseline_gains(_substitute(dead)) * (ast + rfi)).mean(axis=0)))

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)

            with pytest.raises(RuntimeWarning, match="Affected antennas"):
                run_writer(_fake_ms(data), again, ms_path=ms_skeleton, emit=True)

        # The table beside the results is the new one: antenna 2 has no solution.
        current = read_caltable(_caltable_path(zarr_path))["gains"]

        assert np.all(np.isnan(current[2]))
        assert np.all(np.isfinite(current[[0, 1, 3]]))

    def test_two_warnings_in_flight_keep_the_first(
        self, tmp_path, run_writer, ms_skeleton
    ):
        """Bad gains *and* a failed export, both under an error filter.

        The export still clears the table it could not replace -- that happens
        before its own warning -- and the warning it then raises arrives chained
        to the bad-gain error rather than in place of it, so neither report is
        lost.
        """
        gains, ast, rfi = _model(1)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        data = _observed(_to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)))
        run_writer(_fake_ms(data), zarr_path, ms_path=ms_skeleton, emit=True)

        assert os.path.exists(_caltable_path(zarr_path))

        two_windows = _ms_skeleton(tmp_path, name="two_windows.ms", n_spw=2)
        dead = gains.copy()
        dead[:, 2] = 0.0
        again = _write_zarr(tmp_path, dead, ast, rfi)
        data = _observed(_to_ms((_baseline_gains(_substitute(dead)) * (ast + rfi)).mean(axis=0)))

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)

            with pytest.raises(RuntimeWarning) as raised:
                run_writer(_fake_ms(data), again, ms_path=two_windows, emit=True)

        chain, error = [], raised.value
        while error is not None:
            chain.append(str(error))
            error = error.__context__

        assert any("spectral window" in link for link in chain)
        assert any("Affected antennas" in link for link in chain)
        # And the table that described the previous fit is gone even here: the
        # clean-up runs before the export's own warning does.
        assert not os.path.exists(_caltable_path(zarr_path))

    def test_a_memory_error_is_not_demoted(self, tmp_path, run_writer, monkeypatch):
        """Running out of memory is the machine's problem, not the export's.

        Everything else the export can raise is a statement about this MS or
        these results; ``MemoryError`` is a statement about the process, and
        swallowing it would let the run carry on in a state nothing else here
        can reason about.
        """

        def _oom(*args, **kwargs):
            raise MemoryError("out of memory")

        monkeypatch.setattr(write_mod, "write_gain_caltable", _oom)

        gains, ast, rfi = _model(1)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        data = _observed(_to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)))

        with pytest.raises(MemoryError):
            run_writer(_fake_ms(data), zarr_path, emit=True)

    def test_the_warning_names_the_error_and_where_it_came_from(
        self, tmp_path, run_writer, monkeypatch
    ):
        """A demoted failure has to stay diagnosable: type, message, traceback tail."""

        def _boom(*args, **kwargs):
            raise ZeroDivisionError("a bug, not a bad MS")

        monkeypatch.setattr(write_mod, "write_gain_caltable", _boom)

        gains, ast, rfi = _model(1)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        data = _observed(_to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)))

        with pytest.warns(RuntimeWarning) as record:
            run_writer(_fake_ms(data), zarr_path, emit=True)

        message = str(record[0].message)

        assert "ZeroDivisionError" in message
        assert "a bug, not a bad MS" in message
        assert "_boom" in message  # the traceback tail, not just the exception


class TestTheEmittedCaltable:
    """What the table holds, and what applying it does.

    The claim the export exists for is that **one** application of this table
    takes the MS's data column to the ``CORRECTED_DATA`` written beside it, so
    that is asserted directly, against the columns the same run wrote.
    """

    def test_it_lands_beside_the_results_on_the_ms_own_grid(
        self, tmp_path, run_writer, ms_skeleton
    ):
        gains, ast, rfi = _model(1)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        data = _observed(_to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)))

        run_writer(_fake_ms(data), zarr_path, ms_path=ms_skeleton, emit=True)

        read = read_caltable(_caltable_path(zarr_path))
        mean = gains.mean(axis=0)

        np.testing.assert_allclose(
            read["gains"], mean, rtol=1e-5, atol=1e-5 * np.abs(mean).max()
        )
        # The MS's own grid: its TIME column in seconds, and its channels.
        np.testing.assert_allclose(read["times"], np.arange(N_TIME, dtype=float), atol=1e-9)
        np.testing.assert_allclose(read["freqs"], FREQS)
        # Frequency dependent, so B and not G.
        assert read["viscal"] == "B Jones"

    def test_the_total_calibration_is_external_times_fitted(
        self, tmp_path, run_writer, ms_skeleton
    ):
        gains, ast, rfi = _model(1)
        ext = _ext_gains()
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        data = _observed(
            _to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)) * _ext_bl(ext)
        )

        run_writer(
            _fake_ms(data), zarr_path, ext_gains=ext, ms_path=ms_skeleton, emit=True
        )

        read = read_caltable(_caltable_path(zarr_path))
        total = ext * gains.mean(axis=0)

        np.testing.assert_allclose(
            read["gains"], total, rtol=1e-5, atol=1e-5 * np.abs(total).max()
        )
        # The fitted layer on its own is a different calibration.
        assert not np.allclose(read["gains"], gains.mean(axis=0), rtol=1e-3)

    def test_the_table_reproduces_the_calibrated_column(
        self, tmp_path, run_writer, ms_skeleton
    ):
        """``g_p^tot conj(g_q^tot)`` is the very divisor the columns were written with.

        This is tabascal's own composition arithmetic, checked at complex64
        round-off -- not a test of CASA. That ``applycal`` accepts a table of this
        shape and applies it as ``V / (g_p conj(g_q))`` is
        :func:`~tabascal.ms.write_caltable`'s claim, verified against CASA where
        that function is documented.

        The table carries per-*antenna* gains while the writer divides by a
        per-*baseline* product, and the two agree because the composition is the
        same either way::

            (g_ext_p g_fit_p) conj(g_ext_q g_fit_q)
                == (g_ext_p conj(g_ext_q)) (g_fit_p conj(g_fit_q))

        which is what makes one ``applycal`` enough. The sample mean commutes
        with the product here because there is one sample, as there is in a MAP
        run; see the several-samples case below for what a table can carry when
        there is more than one.
        """
        gains, ast, rfi = _model(1)
        ext = _ext_gains()
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        ext_bl = _ext_bl(ext)
        data = _observed(
            _to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)) * ext_bl
        )

        values, _ = run_writer(
            _fake_ms(data), zarr_path, ext_gains=ext, ms_path=ms_skeleton, emit=True
        )

        g = read_caltable(_caltable_path(zarr_path))["gains"].astype(np.complex64)
        g_bl = _to_ms(g[A1_BL] * g[A2_BL].conj())

        np.testing.assert_allclose(
            g_bl, ext_bl * _fitted_gains_bl(gains), **_tolerances(g_bl)
        )
        np.testing.assert_allclose(
            values["CORRECTED_DATA"] * g_bl, data, **_tolerances(data)
        )

    def test_a_dead_gain_is_flagged_in_the_table_and_unity_in_the_column(
        self, tmp_path, run_writer, ms_skeleton
    ):
        """The one divergence, both halves of it in a single run.

        A gain the fit drove to zero carries no solution: the table says so, with
        ``FLAG`` set and ``CPARAM`` NaN, which is what CASA does with an unsolved
        antenna. The columns cannot say it -- a blank column is a dropped
        visibility -- so they keep #134's unity substitution and are written
        uncalibrated on that antenna instead.
        """
        tables = pytest.importorskip("casacore.tables")

        gains, ast, rfi = _model(1)
        gains = gains.copy()
        gains[:, 2] = 0.0

        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        gains_sub = _substitute(gains)
        data = _observed(_to_ms((_baseline_gains(gains_sub) * (ast + rfi)).mean(axis=0)))

        with pytest.warns(RuntimeWarning):
            values, _ = run_writer(
                _fake_ms(data), zarr_path, ms_path=ms_skeleton, emit=True
            )

        out = _caltable_path(zarr_path)
        read = read_caltable(out)

        assert np.all(np.isnan(read["gains"][2]))
        assert np.all(np.isfinite(read["gains"][[0, 1, 3]]))

        with tables.table(out, ack=False) as tb:
            flag = tb.getcol("FLAG")
            ant1 = tb.getcol("ANTENNA1")

        assert np.all(flag[ant1 == 2])
        assert not np.any(flag[ant1 != 2])

        # The columns, in the same run: unity, so baseline (0, 2) is still
        # calibrated on antenna 0 and nothing is blanked.
        bl = int(np.flatnonzero((A1_BL == 0) & (A2_BL == 2))[0])

        for t in range(N_TIME):
            row = t * N_BL + bl
            np.testing.assert_allclose(
                values["CORRECTED_DATA"][row, :, 0],
                data[row, :, 0] / gains_sub[0, 0, :, t],
                **_tolerances(data),
            )

        assert np.all(np.isfinite(values["CORRECTED_DATA"]))

    def test_an_antenna_no_table_solved_is_flagged_too(
        self, tmp_path, run_writer, ms_skeleton
    ):
        """The same divergence on the external layer.

        ``gains_from_tables`` reports an antenna it could not place a gain for
        and hands back unity so the columns can be written uncalibrated on it.
        The table has somewhere better to put that: a flag.
        """
        gains, ast, rfi = _model(1)
        ext = _ext_gains()
        dead = np.zeros(ext.shape, dtype=bool)
        dead[1] = True

        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        data = _observed(_to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)))

        values, _ = run_writer(
            _fake_ms(data),
            zarr_path,
            ext_gains=ext,
            ext_dead=dead,
            ms_path=ms_skeleton,
            emit=True,
        )

        read = read_caltable(_caltable_path(zarr_path))

        assert np.all(np.isnan(read["gains"][1]))
        assert np.all(np.isfinite(read["gains"][[0, 2, 3]]))
        assert np.all(np.isfinite(values["CORRECTED_DATA"]))

    def test_the_fitted_correlation_is_recorded(
        self, tmp_path, run_writer, ms_skeleton
    ):
        """A ``yx``-fitted table has to be identifiable as one.

        Nothing in the caltable format says which correlation a single-solution
        table belongs to, and applying an ``xx`` solution to ``yx`` data is a
        silent mistake, so the run's own answer is carried as a keyword.
        """
        tables = pytest.importorskip("casacore.tables")

        gains, ast, rfi = _model(1)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi, corr="yx")
        data = _observed(_to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)))

        run_writer(_fake_ms(data), zarr_path, ms_path=ms_skeleton, emit=True)

        with tables.table(_caltable_path(zarr_path), ack=False) as tb:
            assert tb.getkeyword("FittedCorr") == "yx"

    def test_several_samples_carry_the_mean_of_the_per_antenna_gains(
        self, tmp_path, run_writer, ms_skeleton
    ):
        """All a per-antenna table can carry, and not quite the columns' divisor.

        The columns divide by the mean of ``g_p conj(g_q)``; a caltable holds one
        gain per antenna, so it can only carry the mean of ``g_p``. The two part
        company exactly when the two antennas' gains covary across samples, which
        a MAP run -- one sample -- never does.
        """
        gains, ast, rfi = _model(2)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        data = _observed(_to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)))

        run_writer(_fake_ms(data), zarr_path, ms_path=ms_skeleton, emit=True)

        read = read_caltable(_caltable_path(zarr_path))
        mean = gains.mean(axis=0)

        np.testing.assert_allclose(
            read["gains"], mean, rtol=1e-5, atol=1e-5 * np.abs(mean).max()
        )

        g = read["gains"].astype(np.complex64)
        assert not np.allclose(
            _to_ms(g[A1_BL] * g[A2_BL].conj()), _fitted_gains_bl(gains), rtol=1e-3
        )

    def test_the_time_scale_is_the_ms_own(self, tmp_path, run_writer, ms_skeleton):
        """A TAI-declared MS gives a TAI-declared table, with the same numbers.

        The caltable's ``TIME`` is a copy of the MS's column, so relabelling it
        UTC would move every timestamp by the leap seconds -- 37 s since 2017 --
        for anything that reads the declaration. Declared to declared, which is
        also the convention the gain tables are matched on.
        """
        tables = pytest.importorskip("casacore.tables")

        gains, ast, rfi = _model(1)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        data = _observed(_to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)))

        run_writer(
            _fake_ms(data), zarr_path, ms_path=ms_skeleton, emit=True, time_scale="TAI"
        )

        with tables.table(_caltable_path(zarr_path), ack=False) as tb:
            assert tb.getcolkeyword("TIME", "MEASINFO")["Ref"] == "TAI"
            assert np.array_equal(
                tb.getcol("TIME"), np.repeat(np.arange(N_TIME, dtype=float), N_ANT)
            )

    def test_the_table_is_named_after_the_results_it_came_from(
        self, tmp_path, run_writer, ms_skeleton
    ):
        """Including the initial prediction, which gets a table of its own.

        A run writes results twice under some configurations -- the initial
        parameters and the optimised ones -- and each export is named after the
        zarr it was made from, so the two never overwrite one another and it is
        always clear which solution a table is.
        """
        gains, ast, rfi = _model(1)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi, name="init_pred_Custom.zarr")
        data = _observed(_to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)))

        run_writer(_fake_ms(data), zarr_path, ms_path=ms_skeleton, emit=True)

        assert os.path.exists(str(tmp_path / "init_pred_Custom.B"))

    def test_a_chunked_posterior_is_reduced_chunk_by_chunk(
        self, tmp_path, run_writer, ms_skeleton
    ):
        """The sample axis is reduced by dask rather than materialised whole.

        A posterior is ``(sample, antenna, channel, time)`` and can be far larger
        than the table it reduces to, so both the unity check and the sample mean
        run over the stored chunks. The memory that is *not* used is not
        something a test can see; what is asserted is that the chunked path
        reaches the same answer as the contiguous one.
        """
        gains, ast, rfi = _model(3)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi, sample_chunk=1)
        data = _observed(_to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)))

        # One chunk per sample on disk, or the reduction is over a single chunk
        # and proves nothing.
        assert xr.open_zarr(zarr_path).gains.chunksizes["sample"] == (1, 1, 1)

        run_writer(_fake_ms(data), zarr_path, ms_path=ms_skeleton, emit=True)

        read = read_caltable(_caltable_path(zarr_path))
        mean = gains.mean(axis=0)

        np.testing.assert_allclose(
            read["gains"], mean, rtol=1e-5, atol=1e-5 * np.abs(mean).max()
        )

    def test_a_second_write_replaces_the_table(
        self, tmp_path, run_writer, ms_skeleton
    ):
        """Re-running over the same results overwrites its own table."""
        gains, ast, rfi = _model(1)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        data = _observed(_to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)))
        run_writer(_fake_ms(data), zarr_path, ms_path=ms_skeleton, emit=True)

        again = _write_zarr(tmp_path, 2 * gains, ast, rfi)
        run_writer(_fake_ms(data), again, ms_path=ms_skeleton, emit=True)

        read = read_caltable(_caltable_path(zarr_path))
        mean = 2 * gains.mean(axis=0)

        np.testing.assert_allclose(
            read["gains"], mean, rtol=1e-5, atol=1e-5 * np.abs(mean).max()
        )


class TestNothingIsExportedWithoutAFit:
    """No fitted gains, no calibration to export -- and no stale table left."""

    def test_a_unitary_run_leaves_no_table_beside_its_results(
        self, tmp_path, run_writer, ms_skeleton
    ):
        """``UnitaryGains`` stores ones, which is not a calibration."""
        _, ast, rfi = _model(1)
        gains = _uniform_gains(1)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        data = _observed(_to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)))

        run_writer(_fake_ms(data), zarr_path, ms_path=ms_skeleton, emit=True)

        assert not os.path.exists(_caltable_path(zarr_path))

    def test_a_rerun_that_fits_nothing_removes_the_superseded_table(
        self, tmp_path, run_writer, ms_skeleton
    ):
        """The stale table is the dangerous one: it reads as this run's answer.

        Re-running the same results path with a model that fits no gains has no
        table to overwrite the old one with, so the old one is removed and the
        removal is reported.
        """
        gains, ast, rfi = _model(1)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        data = _observed(_to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)))
        run_writer(_fake_ms(data), zarr_path, ms_path=ms_skeleton, emit=True)

        assert os.path.exists(_caltable_path(zarr_path))

        unity = _uniform_gains(1)
        again = _write_zarr(tmp_path, unity, ast, rfi)
        data = _observed(_to_ms((_baseline_gains(unity) * (ast + rfi)).mean(axis=0)))

        with pytest.warns(RuntimeWarning, match="earlier run"):
            run_writer(_fake_ms(data), again, ms_path=ms_skeleton, emit=True)

        assert not os.path.exists(_caltable_path(zarr_path))

    def test_nothing_to_remove_is_silent(self, tmp_path, run_writer, ms_skeleton):
        """The usual unitary run: no table there, nothing to say about it."""
        _, ast, rfi = _model(1)
        gains = _uniform_gains(1)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        data = _observed(_to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)))

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            run_writer(_fake_ms(data), zarr_path, ms_path=ms_skeleton, emit=True)

    def test_a_chunked_unitary_posterior_exports_nothing(
        self, tmp_path, run_writer, ms_skeleton
    ):
        """The unity test is a chunked reduction and gives the chunked answer."""
        _, ast, rfi = _model(3)
        gains = _uniform_gains(3)
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi, sample_chunk=1)
        data = _observed(_to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)))

        run_writer(_fake_ms(data), zarr_path, ms_path=ms_skeleton, emit=True)

        assert not os.path.exists(_caltable_path(zarr_path))

    def test_samples_that_average_to_unity_are_still_a_calibration(
        self, tmp_path, run_writer, ms_skeleton
    ):
        """Only gains that are unity on *every* sample say the run fitted none.

        Two samples either side of 1 average to exactly 1 while the divisor the
        columns use -- the mean of the baseline *product* -- is nothing of the
        kind, so a check on the mean alone would throw away a real calibration.
        """
        _, ast, rfi = _model(2)
        gains = np.concatenate([_uniform_gains(1, 0.5), _uniform_gains(1, 1.5)])
        zarr_path = _write_zarr(tmp_path, gains, ast, rfi)
        data = _observed(_to_ms((_baseline_gains(gains) * (ast + rfi)).mean(axis=0)))

        run_writer(_fake_ms(data), zarr_path, ms_path=ms_skeleton, emit=True)

        read = read_caltable(_caltable_path(zarr_path))

        assert np.all(gains.mean(axis=0) == 1)
        np.testing.assert_allclose(read["gains"], 1.0, rtol=1e-5, atol=1e-6)

    def test_a_zarr_without_gains_exports_nothing(self, tmp_path, ms_skeleton):
        _, ast, rfi = _model(1)
        path = str(tmp_path / "no_gains.zarr")
        xr.Dataset(
            data_vars={
                "ast_vis": (["sample", "bl", "freq", "time"], da.asarray(ast)),
                "rfi_vis": (["sample", "bl", "freq", "time"], da.asarray(rfi)),
            }
        ).to_zarr(path, mode="w")

        assert write_mod.write_gain_caltable(ms_skeleton, path) is None
        assert not os.path.exists(_caltable_path(path))

    def test_gains_that_are_not_a_grid_export_nothing(self, tmp_path, ms_skeleton):
        """Not three-dimensional after the sample mean: nothing a caltable holds."""
        path = str(tmp_path / "flat.zarr")
        xr.Dataset(
            data_vars={
                "gains": (
                    ["sample", "ant"],
                    da.asarray(np.full((1, N_ANT), 2.0, dtype=complex)),
                )
            }
        ).to_zarr(path, mode="w")

        assert write_mod.write_gain_caltable(ms_skeleton, path) is None
        assert not os.path.exists(_caltable_path(path))


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
