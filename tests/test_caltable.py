"""Tests for the CASA calibration tables in :mod:`tabascal.ms`.

The end-to-end check that CASA's ``applycal`` accepts these tables (and
reproduces ``V / (g_p conj(g_q))`` from them) needs casatasks, so it lives with
the pipeline verification. What is locked down here is the format CASA keys off
and the gain convention, which is what silently breaks if either is changed.

No jax here, so the tolerances are not the ``exact_rtol`` fixture's business:
they are set by the table format itself. ``CPARAM`` is complex64 whatever
precision the run is in, so a write/read round-trip is a float32 comparison in
either session; the pure-numpy identities are float64 in both.
"""

import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

# The whole module is casacore-facing; skip cleanly where it is not installed.
tables = pytest.importorskip("casacore.tables")

from casacore.tables import (  # noqa: E402
    makearrcoldesc,
    makescacoldesc,
    maketabdesc,
    table,
)

from tabascal.ms import (  # noqa: E402
    apply_gains_to_data,
    baseline_gains,
    read_caltable,
    write_caltable,
)


N_ANT, N_FREQ, N_TIME = 6, 4, 3
N_ROW = N_ANT * N_TIME

#: Channel frequencies the minimal MS below declares, in Hz.
FREQS = 1e9 + 1e6 * np.arange(N_FREQ, dtype=float)


@pytest.fixture
def gains():
    """Non-unit, non-uniform complex gains.

    Unity gains cannot tell ``g_p conj(g_q)`` from ``|g_p|**2``, nor a
    frequency-dependent gain from a scalar one.
    """

    rng = np.random.default_rng(0)
    amp = rng.uniform(0.3, 3.0, (N_ANT, N_FREQ, N_TIME))
    phase = rng.uniform(-np.pi, np.pi, (N_ANT, N_FREQ, N_TIME))

    return (amp * np.exp(1j * phase)).astype(complex)


# ---------------------------------------------------------------------------
# A minimal MS to copy subtables out of
# ---------------------------------------------------------------------------

def _write_spw(path: str, freq_rows) -> None:
    """A ``SPECTRAL_WINDOW`` subtable with one row per entry of *freq_rows*."""

    freq_rows = np.asarray(freq_rows, dtype=float)
    spw = table(
        path,
        maketabdesc(
            [
                makearrcoldesc("CHAN_FREQ", 0.0, ndim=1, valuetype="double"),
                makescacoldesc("NUM_CHAN", 0, valuetype="int"),
            ]
        ),
        nrow=len(freq_rows),
        ack=False,
    )
    spw.putcol("CHAN_FREQ", freq_rows)
    spw.putcol("NUM_CHAN", np.full(len(freq_rows), freq_rows.shape[1], dtype=np.int32))
    spw.close()


def _minimal_ms(path: str, n_ant: int = N_ANT, n_spw: int = 1) -> str:
    """An MS carrying only the subtables a caltable copies out of one.

    Enough of an MS for :func:`write_caltable`: it reads no main-table column,
    only the spectral window count, and ``tablecopy``s the subtables it finds.
    Written here rather than taken from the other MS tests because those build
    in-memory xarray stand-ins, and copying a subtable needs a real table on
    disk.

    All five subtables a caltable carries are present, so their copy and
    keyword registration are exercised; ``FIELD``/``OBSERVATION``/``HISTORY``
    are as small as a valid table can be, since nothing reads their contents.
    """

    main = table(
        path,
        maketabdesc([makescacoldesc("TIME", 0.0, valuetype="double")]),
        nrow=1,
        ack=False,
    )

    ant = table(
        os.path.join(path, "ANTENNA"),
        maketabdesc([makearrcoldesc("POSITION", 0.0, ndim=1, valuetype="double")]),
        nrow=n_ant,
        ack=False,
    )
    ant.putcol("POSITION", np.arange(3 * n_ant, dtype=float).reshape(n_ant, 3))
    ant.close()

    _write_spw(
        os.path.join(path, "SPECTRAL_WINDOW"),
        np.tile(FREQS, (n_spw, 1)) + 1e8 * np.arange(n_spw)[:, None],
    )

    for sub, column, nrow in (
        ("FIELD", "NAME", 1),
        ("OBSERVATION", "TELESCOPE_NAME", 1),
        ("HISTORY", "MESSAGE", 0),
    ):
        sub_tb = table(
            os.path.join(path, sub),
            maketabdesc([makescacoldesc(column, "", valuetype="string")]),
            nrow=nrow,
            ack=False,
        )
        sub_tb.close()

    main.close()

    return path


@pytest.fixture
def ms_path(tmp_path):
    """A minimal on-disk MS to copy subtables from."""

    return _minimal_ms(str(tmp_path / "minimal.ms"))


@pytest.fixture
def case_insensitive_fs(tmp_path):
    """Skip unless this filesystem treats ``X`` and ``x`` as one name.

    Probed rather than inferred from the platform: macOS is case-insensitive by
    default but can be formatted either way, and a Linux box can mount a
    case-insensitive volume. On a case-sensitive filesystem a case variant is
    genuinely a different directory, so there is nothing for these to test.
    """

    probe = tmp_path / "CaseProbe"
    probe.mkdir()
    insensitive = (tmp_path / "caseprobe").exists()
    probe.rmdir()

    if not insensitive:
        pytest.skip(
            "case-sensitive filesystem: a case variant is a different directory"
        )


def _raw(path: str, column: str):
    """A column straight out of the caltable, with no interpretation applied."""

    with table(path, ack=False) as tb:
        return tb.getcol(column)


def _overwrite_pols(path: str, cparam, flag) -> None:
    """Replace a caltable's per-polarisation values, standing in for CASA.

    ``write_caltable`` only ever duplicates one solution across the pol axis, so
    a table whose polarisations genuinely differ has to be built by hand.
    """

    with table(path, readonly=False, ack=False) as tb:
        tb.putcol("CPARAM", np.asarray(cparam, dtype=np.complex64))
        tb.putcol("FLAG", np.asarray(flag, dtype=bool))
        tb.flush()


# ---------------------------------------------------------------------------
# The format CASA keys off
# ---------------------------------------------------------------------------

class TestCasaFormat:

    def test_casa_format(self, tmp_path, gains, ms_path):
        """CASA identifies a caltable by its INFO record; applycal rejects it otherwise."""

        path = str(tmp_path / "test.B")
        times = np.arange(N_TIME, dtype=float)
        write_caltable(path, gains, times, ms_path=ms_path)

        with table(path, ack=False) as tb:
            assert tb.info()["type"] == "Calibration"
            assert tb.info()["subType"] == "B Jones"
            assert tb.getkeyword("ParType") == "Complex"
            assert tb.getkeyword("VisCal") == "B Jones"
            assert tb.getkeyword("PolBasis") == "unknown"

            assert tb.nrows() == N_ROW
            # CASA writes 2 pols even for a single-correlation MS.
            assert tb.getcell("CPARAM", 0).shape == (N_FREQ, 2)
            # One row per (time, antenna), with no second antenna.
            assert set(np.unique(tb.getcol("ANTENNA2"))) == {-1}
            assert sorted(np.unique(tb.getcol("ANTENNA1"))) == list(range(N_ANT))

    def test_cparam_is_complex64(self, tmp_path, gains, ms_path):
        """The format's own dtype, whatever precision the gains were solved in."""

        path = str(tmp_path / "test.B")
        write_caltable(path, gains, np.arange(N_TIME, dtype=float), ms_path=ms_path)

        assert _raw(path, "CPARAM").dtype == np.complex64

    def test_the_solution_columns_are_present_and_shaped(
        self, tmp_path, gains, ms_path
    ):
        """CPARAM/PARAMERR/FLAG/SNR are filled; WEIGHT is declared and left empty."""

        path = str(tmp_path / "test.B")
        write_caltable(path, gains, np.arange(N_TIME, dtype=float), ms_path=ms_path)

        with table(path, ack=False) as tb:
            for column in ("CPARAM", "PARAMERR", "FLAG", "SNR", "WEIGHT"):
                assert column in tb.colnames()

            for column in ("CPARAM", "PARAMERR", "FLAG", "SNR"):
                assert tb.getcol(column).shape == (N_ROW, N_FREQ, 2)

            # CASA declares WEIGHT but leaves it unfilled; mirror that.
            assert not tb.iscelldefined("WEIGHT", 0)

    def test_the_subtables_are_copied_from_the_ms(self, tmp_path, gains, ms_path):
        """A caltable carries its own copy of the MS's description subtables."""

        path = str(tmp_path / "test.B")
        write_caltable(path, gains, np.arange(N_TIME, dtype=float), ms_path=ms_path)

        with table(path, ack=False) as tb:
            for sub in ("ANTENNA", "FIELD", "SPECTRAL_WINDOW", "OBSERVATION", "HISTORY"):
                assert os.path.exists(os.path.join(path, sub))
                # Registered by absolute path, as CASA registers them.
                assert tb.getkeyword(sub) == "Table: " + os.path.abspath(
                    os.path.join(path, sub)
                )

            assert tb.getkeyword("MSName") == "minimal.ms"

        with table(os.path.join(path, "ANTENNA"), ack=False) as ant:
            assert ant.nrows() == N_ANT

    def test_caller_keywords_ride_along(self, tmp_path, gains, ms_path):
        """Extra keywords are written beside the table's own, not instead of them.

        What a solver knows about its solution and the format has no field for --
        the correlation tabascal fitted, say -- goes in a keyword, so the table
        still says it when it is read back somewhere else.
        """

        path = str(tmp_path / "test.B")
        write_caltable(
            path,
            gains,
            np.arange(N_TIME, dtype=float),
            ms_path=ms_path,
            keywords={"FittedCorr": "yx"},
        )

        with table(path, ack=False) as tb:
            assert tb.getkeyword("FittedCorr") == "yx"
            # The keywords CASA identifies the table by are untouched.
            assert tb.getkeyword("VisCal") == "B Jones"
            assert tb.getkeyword("ANTENNA").startswith("Table: ")

    def test_a_missing_subtable_is_not_an_error(self, tmp_path, gains):
        """The subtables are copied if they are there; none of them is required."""

        path = str(tmp_path / "test.B")
        write_caltable(
            path, gains, np.arange(N_TIME, dtype=float),
            ms_path=str(tmp_path / "fake.ms"),
        )

        assert not os.path.exists(os.path.join(path, "ANTENNA"))
        assert read_caltable(path)["freqs"] is None

    def test_the_row_order_is_time_major(self, tmp_path, gains, ms_path):
        """(t0, a0), (t0, a1), ... -- the order gaincal writes, and applycal reads."""

        path = str(tmp_path / "test.B")
        times = np.array([10.0, 20.0, 30.0])
        write_caltable(path, gains, times, ms_path=ms_path)

        assert np.array_equal(
            _raw(path, "ANTENNA1"), np.tile(np.arange(N_ANT), N_TIME)
        )
        assert np.array_equal(_raw(path, "TIME"), np.repeat(times, N_ANT))


# ---------------------------------------------------------------------------
# Dead solutions
# ---------------------------------------------------------------------------

class TestFlagging:
    """A gain that is zero or non-finite carries no solution.

    The contract is both halves at once: ``CPARAM`` NaN *and* ``FLAG`` true. A
    reader that trusts the flag and a reader that trusts the value have to reach
    the same conclusion -- a zero left in CPARAM reads as a real solution that
    calibrates to infinity, and an Inf left there reads as a real one too.
    """

    @pytest.mark.parametrize("bad_value", [0.0, np.inf, -np.inf, np.nan])
    def test_a_dead_gain_is_written_as_nan_and_flagged(
        self, tmp_path, gains, ms_path, bad_value
    ):
        g = gains.copy()
        g[2, 1, 0] = bad_value
        path = str(tmp_path / "dead.B")
        write_caltable(path, g, np.arange(N_TIME, dtype=float), ms_path=ms_path)

        # Row 2 is (t0, a2) in the time-major layout; channel 1, both pols.
        cparam, flag = _raw(path, "CPARAM"), _raw(path, "FLAG")
        dead = cparam[2, 1]

        assert flag[2, 1].all()
        # Both components, checked apart: np.isnan on a complex is true when
        # either half is NaN, so it would pass on a real-only NaN too.
        assert np.all(np.isnan(dead.real))
        assert np.all(np.isnan(dead.imag))

        if not np.isnan(bad_value):
            # The raw value is gone, not merely flagged. Only meaningful for the
            # values that would otherwise have survived into the column -- NaN
            # compares equal to nothing, so this says nothing about that case,
            # where the two assertions above carry the weight instead.
            assert not np.any(dead == np.complex64(bad_value))

        # Its neighbours are untouched.
        assert not flag[2, 0].any()
        assert np.all(np.isfinite(cparam[2, 0]))

    def test_flagged_gains_roundtrip_as_nan(self, tmp_path, gains, ms_path):
        g = gains.copy()
        g[2, :, 1] = 0.0  # a dead antenna at one time
        path = str(tmp_path / "flagged.B")
        write_caltable(path, g, np.arange(N_TIME, dtype=float), ms_path=ms_path)

        out = read_caltable(path)["gains"]
        assert np.all(np.isnan(out[2, :, 1]))
        assert np.allclose(out[0], g[0], rtol=1e-5, atol=1e-6)

    def test_non_finite_gains_are_flagged(self, tmp_path, gains, ms_path):
        """A gain that is not finite carries no solution either."""

        g = gains.copy()
        g[3, 1, 0] = np.nan
        path = str(tmp_path / "nonfinite.B")
        write_caltable(path, g, np.arange(N_TIME, dtype=float), ms_path=ms_path)

        flag = _raw(path, "FLAG")  # (n_row, n_freq, n_pol), time-major rows

        assert flag[3, 1].all()  # antenna 3, channel 1, first timestep
        assert not flag[3, 0].any()
        assert np.all(np.isnan(read_caltable(path)["gains"][3, 1, 0]))


# ---------------------------------------------------------------------------
# Polarisation
# ---------------------------------------------------------------------------

class TestPolarisations:
    """tabascal solves one correlation; a caltable has room for two.

    ``write_caltable`` duplicates its single solution across the pol axis, so
    collapsing that axis on the way back in is a no-op for our own tables. A
    table from CASA can hold genuinely different Jones terms per polarisation,
    and averaging those would return a gain that calibrates neither.
    """

    def _cal(self, tmp_path, gains, ms_path):
        path = str(tmp_path / "pol.B")
        write_caltable(path, gains, np.arange(N_TIME, dtype=float), ms_path=ms_path)

        return path

    def test_duplicated_polarisations_collapse_silently(
        self, tmp_path, gains, ms_path
    ):
        """Our own tables: the two pols are the same number, so this is a no-op."""

        path = self._cal(tmp_path, gains, ms_path)

        assert np.allclose(read_caltable(path)["gains"], gains, rtol=1e-5, atol=1e-6)

    def test_disagreeing_polarisations_are_rejected(self, tmp_path, gains, ms_path):
        """The bug: averaging XX and YY returns a gain that calibrates neither."""

        path = self._cal(tmp_path, gains, ms_path)
        cparam = _raw(path, "CPARAM")
        cparam[:, :, 1] *= 2.0  # a genuinely different second Jones term

        _overwrite_pols(path, cparam, np.zeros_like(cparam, dtype=bool))

        with pytest.raises(ValueError, match="polarisation"):
            read_caltable(path)

    def test_the_rejection_points_at_the_polarisation_issue(
        self, tmp_path, gains, ms_path
    ):
        """Said as a limitation with somewhere to go, not as a corrupt-table error."""

        path = self._cal(tmp_path, gains, ms_path)
        cparam = _raw(path, "CPARAM")
        cparam[0, 0, 1] *= 3.0

        _overwrite_pols(path, cparam, np.zeros_like(cparam, dtype=bool))

        with pytest.raises(ValueError, match="#151"):
            read_caltable(path)

    def test_a_flagged_polarisation_falls_back_to_the_other(
        self, tmp_path, gains, ms_path
    ):
        """A flagged pol is missing, not zero: the surviving solution is the answer."""

        path = self._cal(tmp_path, gains, ms_path)
        cparam = _raw(path, "CPARAM")
        flag = np.zeros_like(cparam, dtype=bool)

        # Pol 1 is dead at (row 0, chan 0); pol 0 still has the solution.
        flag[0, 0, 1] = True
        cparam[0, 0, 1] = np.nan
        _overwrite_pols(path, cparam, flag)

        out = read_caltable(path)["gains"]

        # Row 0 is (t0, a0). The surviving pol is returned, not NaN.
        assert np.isfinite(out[0, 0, 0])
        assert np.allclose(out[0, 0, 0], gains[0, 0, 0], rtol=1e-5, atol=1e-6)

    def test_both_polarisations_flagged_is_still_nan(self, tmp_path, gains, ms_path):
        """No surviving solution means no solution."""

        path = self._cal(tmp_path, gains, ms_path)
        cparam = _raw(path, "CPARAM")
        flag = np.zeros_like(cparam, dtype=bool)
        flag[0, 0, :] = True
        cparam[0, 0, :] = np.nan
        _overwrite_pols(path, cparam, flag)

        assert np.isnan(read_caltable(path)["gains"][0, 0, 0])

    def test_a_disagreement_under_a_flag_is_not_a_disagreement(
        self, tmp_path, gains, ms_path
    ):
        """Only unflagged polarisations have to agree; a dead one holds anything."""

        path = self._cal(tmp_path, gains, ms_path)
        cparam = _raw(path, "CPARAM")
        flag = np.zeros_like(cparam, dtype=bool)
        flag[0, 0, 1] = True
        cparam[0, 0, 1] = 12345.0  # junk, but flagged, so it says nothing
        _overwrite_pols(path, cparam, flag)

        assert np.allclose(
            read_caltable(path)["gains"][0, 0, 0], gains[0, 0, 0],
            rtol=1e-5, atol=1e-6,
        )


# ---------------------------------------------------------------------------
# Spectral windows
# ---------------------------------------------------------------------------

class TestSingleSpectralWindow:
    """One spectral window is the supported scope, and it is checked.

    Every row is written with ``SPECTRAL_WINDOW_ID = 0`` and the frequencies are
    read from row 0, so a multi-window MS would silently get one window's gains
    labelled with another window's channels.
    """

    def test_write_rejects_a_multi_window_ms(self, tmp_path, gains):
        ms = _minimal_ms(str(tmp_path / "two_spw.ms"), n_spw=2)

        with pytest.raises(ValueError, match="spectral window"):
            write_caltable(
                str(tmp_path / "test.B"), gains,
                np.arange(N_TIME, dtype=float), ms_path=ms,
            )

    def test_the_rejection_happens_before_anything_is_written(self, tmp_path, gains):
        ms = _minimal_ms(str(tmp_path / "two_spw.ms"), n_spw=2)
        path = str(tmp_path / "test.B")

        with pytest.raises(ValueError):
            write_caltable(path, gains, np.arange(N_TIME, dtype=float), ms_path=ms)

        assert not os.path.exists(path)

    def test_read_rejects_a_multi_window_caltable(self, tmp_path, gains, ms_path):
        """A CASA table can carry several windows even though ours cannot."""

        path = str(tmp_path / "test.B")
        write_caltable(path, gains, np.arange(N_TIME, dtype=float), ms_path=ms_path)

        # Swap the copied one-window subtable for a two-window one.
        spw_path = os.path.join(path, "SPECTRAL_WINDOW")
        shutil.rmtree(spw_path)
        _write_spw(spw_path, np.tile(FREQS, (2, 1)))

        with pytest.raises(ValueError, match="spectral window"):
            read_caltable(path)


# ---------------------------------------------------------------------------
# Round-tripping
# ---------------------------------------------------------------------------

class TestRoundtrip:

    def test_roundtrip(self, tmp_path, gains, ms_path):
        path = str(tmp_path / "test.B")
        times = np.array([1.0, 2.0, 3.0]) * 1e9
        write_caltable(path, gains, times, ms_path=ms_path)

        out = read_caltable(path)
        assert np.allclose(out["gains"], gains, rtol=1e-5, atol=1e-6)
        assert np.allclose(out["times"], times)
        assert np.array_equal(out["ant_idx"], np.arange(N_ANT))
        assert out["viscal"] == "B Jones"

    def test_freqs_come_back_from_the_caltables_own_spectral_window(
        self, tmp_path, gains, ms_path
    ):
        """The copied SPECTRAL_WINDOW is what makes the read self-describing."""

        path = str(tmp_path / "test.B")
        write_caltable(path, gains, np.arange(N_TIME, dtype=float), ms_path=ms_path)

        assert np.allclose(read_caltable(path)["freqs"], FREQS)

    def test_write_read_write_is_idempotent(self, tmp_path, gains, ms_path):
        """A second pass through the format changes nothing: complex64 is a fixpoint."""

        first = str(tmp_path / "first.B")
        second = str(tmp_path / "second.B")
        times = np.arange(N_TIME, dtype=float)

        write_caltable(first, gains, times, ms_path=ms_path)
        out_1 = read_caltable(first)
        write_caltable(second, out_1["gains"], out_1["times"], ms_path=ms_path)
        out_2 = read_caltable(second)

        np.testing.assert_array_equal(out_2["gains"], out_1["gains"])
        np.testing.assert_array_equal(out_2["times"], out_1["times"])
        np.testing.assert_array_equal(out_2["freqs"], out_1["freqs"])
        assert out_2["viscal"] == out_1["viscal"]


# ---------------------------------------------------------------------------
# Validation, and what a rejected call leaves behind
# ---------------------------------------------------------------------------

#: Every way a caller can get ``write_caltable``'s arguments wrong, as an
#: override of the good call built in :meth:`TestValidation._call`, paired with
#: what the error has to say. Each of these must be caught before the existing
#: table is removed, so the same list drives both halves of the guarantee.
BAD_CALLS = [
    pytest.param(
        {"gains": np.ones((N_ANT, N_FREQ), dtype=complex)},
        "n_ant, n_freq, n_time",
        id="gains-too-few-axes",
    ),
    pytest.param(
        {"gains": np.ones((N_ANT, N_FREQ, N_TIME, 1), dtype=complex)},
        "n_ant, n_freq, n_time",
        id="gains-too-many-axes",
    ),
    pytest.param(
        {"gains": np.full((N_ANT, N_FREQ, N_TIME), "x")},
        "numeric",
        id="gains-not-numeric",
    ),
    pytest.param(
        {"gains": np.ones((N_ANT, N_FREQ + 1, N_TIME), dtype=complex)},
        "channel",
        id="gains-channels-vs-ms",
    ),
    pytest.param(
        {"gains": np.ones((N_ANT + 1, N_FREQ, N_TIME), dtype=complex)},
        "antenna",
        id="gains-antennas-vs-ms",
    ),
    pytest.param(
        {"times": np.arange(N_TIME + 1, dtype=float)}, "times", id="times-length"
    ),
    pytest.param(
        {"times": np.zeros((N_TIME, 2), dtype=float)}, "times", id="times-shape"
    ),
    pytest.param({"n_pol": 0}, "n_pol", id="n_pol-zero"),
    pytest.param({"n_pol": -1}, "n_pol", id="n_pol-negative"),
    pytest.param({"n_pol": 2.5}, "n_pol", id="n_pol-fractional"),
    pytest.param({"n_pol": "2"}, "n_pol", id="n_pol-string"),
    pytest.param({"n_pol": None}, "n_pol", id="n_pol-none"),
    # bool is a subclass of int, so True would otherwise pass as n_pol = 1.
    pytest.param({"n_pol": True}, "n_pol", id="n_pol-bool"),
    pytest.param({"interval": "soon"}, "interval", id="interval-not-a-number"),
    pytest.param({"viscal": 3}, "viscal", id="viscal-not-a-string"),
    # "False" is a truthy string: taken as a flag it would delete the table the
    # caller was trying to protect.
    pytest.param({"overwrite": "False"}, "overwrite", id="overwrite-string"),
    pytest.param({"overwrite": 1}, "overwrite", id="overwrite-int"),
    pytest.param({"overwrite": None}, "overwrite", id="overwrite-none"),
    pytest.param({"keywords": "FittedCorr"}, "keywords", id="keywords-not-a-mapping"),
    pytest.param({"keywords": {5: "yx"}}, "keywords", id="keywords-name-not-a-string"),
    # The table's own keywords are how CASA identifies it and reaches its
    # subtables; a caller overwriting one would produce a table CASA rejects.
    pytest.param({"keywords": {"VisCal": "G Jones"}}, "VisCal", id="keywords-viscal"),
    pytest.param({"keywords": {"ANTENNA": "elsewhere"}}, "ANTENNA", id="keywords-subtable"),
]


class TestValidation:
    """Nothing is deleted until the arguments are known to be good.

    ``overwrite=True`` removes the existing table, so any check that runs after
    it turns a caller's mistake into the loss of a good calibration -- including
    the checks that only fail deep in the write, like a gains array of strings
    reaching ``np.isfinite`` or an ``interval`` that is not a number.
    """

    def _call(self, path, gains, ms_path, overrides):
        kwargs = {
            "gains": gains,
            "times": np.arange(N_TIME, dtype=float),
            "ms_path": ms_path,
        }
        kwargs.update(overrides)

        return write_caltable(path, **kwargs)

    def test_an_existing_table_is_not_overwritten_unasked(
        self, tmp_path, gains, ms_path
    ):
        path = str(tmp_path / "test.B")
        times = np.arange(N_TIME, dtype=float)
        write_caltable(path, gains, times, ms_path=ms_path)

        with pytest.raises(FileExistsError):
            write_caltable(path, gains, times, ms_path=ms_path, overwrite=False)

        # Still readable: the refusal left the table alone.
        assert np.allclose(read_caltable(path)["gains"], gains, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("overrides, message", BAD_CALLS)
    def test_a_bad_call_is_rejected_before_anything_is_written(
        self, tmp_path, gains, ms_path, overrides, message
    ):
        path = str(tmp_path / "test.B")

        with pytest.raises(ValueError, match=message):
            self._call(path, gains, ms_path, overrides)

        assert not os.path.exists(path)

    @pytest.mark.parametrize("overrides, message", BAD_CALLS)
    def test_a_bad_call_leaves_an_existing_table_intact(
        self, tmp_path, gains, ms_path, overrides, message
    ):
        """The whole point of validating first: ``overwrite=True`` must not
        destroy a good calibration on the way to raising."""

        path = str(tmp_path / "test.B")
        write_caltable(path, gains, np.arange(N_TIME, dtype=float), ms_path=ms_path)

        # overwrite first, so a case that is *about* a bad overwrite keeps its
        # own value rather than having it replaced by the default here.
        with pytest.raises(ValueError, match=message):
            self._call(path, gains, ms_path, {"overwrite": True, **overrides})

        assert np.allclose(read_caltable(path)["gains"], gains, rtol=1e-5, atol=1e-6)


class TestSourceMsConsistency:
    """The gains have to describe the MS whose subtables get copied in beside them.

    A caltable carries the MS's own ``ANTENNA`` and ``SPECTRAL_WINDOW``, and its
    rows index them: ``ANTENNA1`` into the antenna table, ``CPARAM``'s channel
    axis onto ``CHAN_FREQ``. Gains of the wrong width produce a table that
    disagrees with the copy of the MS inside itself, which nothing downstream can
    detect.
    """

    def test_the_channel_count_must_match_the_spectral_window(
        self, tmp_path, gains, ms_path
    ):
        wrong = np.ones((N_ANT, N_FREQ + 2, N_TIME), dtype=complex)

        with pytest.raises(ValueError, match="channel") as err:
            write_caltable(
                str(tmp_path / "test.B"), wrong,
                np.arange(N_TIME, dtype=float), ms_path=ms_path,
            )

        # Both numbers, so the caller can see which end is wrong.
        assert str(N_FREQ + 2) in str(err.value) and str(N_FREQ) in str(err.value)

    def test_the_antenna_count_must_match_the_antenna_table(
        self, tmp_path, gains, ms_path
    ):
        wrong = np.ones((N_ANT + 2, N_FREQ, N_TIME), dtype=complex)

        with pytest.raises(ValueError, match="antenna") as err:
            write_caltable(
                str(tmp_path / "test.B"), wrong,
                np.arange(N_TIME, dtype=float), ms_path=ms_path,
            )

        assert str(N_ANT + 2) in str(err.value) and str(N_ANT) in str(err.value)

    def test_an_ms_without_the_subtables_cannot_contradict_anything(
        self, tmp_path, gains
    ):
        """Nothing to check against is not a mismatch: the subtables are optional."""

        path = str(tmp_path / "test.B")
        write_caltable(
            path, gains, np.arange(N_TIME, dtype=float),
            ms_path=str(tmp_path / "fake.ms"),
        )

        assert os.path.exists(path)


class TestOutputDoesNotOverlapTheMs:
    """The output must not be, contain, or sit inside the MS it is written from.

    ``overwrite=True`` removes the output path outright, and the subtables are
    copied out of the MS *afterwards* -- so an output that is the MS, or contains
    it, deletes the observation before reading it. The comparison is on resolved
    paths, because a symlink and a ``..`` are two spellings of one directory and
    a plain string prefix would call ``x.ms2`` a child of ``x.ms``.
    """

    def _ms_still_reads(self, ms_path):
        """The MS is openable and its subtables still hold what they held."""

        with table(ms_path, ack=False) as ms:
            assert ms.nrows() == 1

        with table(os.path.join(ms_path, "ANTENNA"), ack=False) as ant:
            assert ant.nrows() == N_ANT

        with table(os.path.join(ms_path, "SPECTRAL_WINDOW"), ack=False) as spw:
            assert np.allclose(spw.getcell("CHAN_FREQ", 0), FREQS)

    def test_the_output_may_not_be_the_ms_itself(self, tmp_path, gains, ms_path):
        with pytest.raises(ValueError, match="same directory"):
            write_caltable(
                ms_path, gains, np.arange(N_TIME, dtype=float), ms_path=ms_path
            )

        self._ms_still_reads(ms_path)

    def test_the_output_may_not_be_a_symlink_to_the_ms(
        self, tmp_path, gains, ms_path
    ):
        """Why the comparison is on realpaths and not on the strings given."""

        link = str(tmp_path / "link.B")
        os.symlink(ms_path, link)

        with pytest.raises(ValueError, match="same directory"):
            write_caltable(
                link, gains, np.arange(N_TIME, dtype=float), ms_path=ms_path
            )

        self._ms_still_reads(ms_path)

    def test_the_output_may_not_sit_inside_the_ms(self, tmp_path, gains, ms_path):
        inside = os.path.join(ms_path, "cal.B")

        with pytest.raises(ValueError, match="inside the Measurement Set"):
            write_caltable(
                inside, gains, np.arange(N_TIME, dtype=float), ms_path=ms_path
            )

        self._ms_still_reads(ms_path)

    def test_the_output_may_not_contain_the_ms(self, tmp_path, gains):
        """The one that costs the observation: rmtree of an ancestor of the MS."""

        outer = str(tmp_path / "outer")
        os.makedirs(outer)
        ms = _minimal_ms(os.path.join(outer, "inner.ms"))

        with pytest.raises(ValueError, match="would contain the Measurement Set"):
            write_caltable(outer, gains, np.arange(N_TIME, dtype=float), ms_path=ms)

        assert os.path.exists(outer)
        self._ms_still_reads(ms)

    def test_a_relative_spelling_of_the_same_directory_is_caught(
        self, tmp_path, gains, ms_path
    ):
        """``..`` makes a second spelling that a string comparison would miss."""

        detour = os.path.join(ms_path, "..", os.path.basename(ms_path))

        with pytest.raises(ValueError, match="same directory"):
            write_caltable(
                detour, gains, np.arange(N_TIME, dtype=float), ms_path=ms_path
            )

        self._ms_still_reads(ms_path)

    def test_a_sibling_whose_name_extends_the_ms_name_is_fine(self, tmp_path, gains):
        """``x.ms2`` is not inside ``x.ms``; a prefix test would say it was."""

        ms = _minimal_ms(str(tmp_path / "x.ms"))
        path = str(tmp_path / "x.ms2")

        write_caltable(path, gains, np.arange(N_TIME, dtype=float), ms_path=ms)

        assert np.allclose(read_caltable(path)["gains"], gains, rtol=1e-5, atol=1e-6)
        self._ms_still_reads(ms)

    # -- Case-variant aliases -------------------------------------------------
    #
    # On a case-insensitive filesystem (APFS, NTFS) "X.ms" and "x.ms" are one
    # directory, but realpath returns whichever spelling it was handed, so the
    # resolved strings still differ. A guard comparing those strings sees two
    # unrelated paths and lets the caller delete the MS. Only the filesystem can
    # settle it, so the guard asks it -- and so do these.

    def test_a_case_variant_of_the_ms_is_the_same_directory(
        self, tmp_path, gains, case_insensitive_fs
    ):
        ms = _minimal_ms(str(tmp_path / "x.ms"))
        alias = str(tmp_path / "X.ms")

        with pytest.raises(ValueError, match="same directory"):
            write_caltable(
                alias, gains, np.arange(N_TIME, dtype=float), ms_path=ms
            )

        self._ms_still_reads(ms)

    def test_a_case_variant_ancestor_still_contains_the_ms(
        self, tmp_path, gains, case_insensitive_fs
    ):
        """The costly one: rmtree of an ancestor reached by a different spelling."""

        outer = str(tmp_path / "outer")
        os.makedirs(outer)
        ms = _minimal_ms(os.path.join(outer, "inner.ms"))

        with pytest.raises(ValueError, match="would contain the Measurement Set"):
            write_caltable(
                str(tmp_path / "OUTER"), gains,
                np.arange(N_TIME, dtype=float), ms_path=ms,
            )

        assert os.path.exists(outer)
        self._ms_still_reads(ms)

    def test_a_case_variant_parent_puts_the_output_inside_the_ms(
        self, tmp_path, gains, case_insensitive_fs
    ):
        ms = _minimal_ms(str(tmp_path / "x.ms"))
        inside = os.path.join(str(tmp_path / "X.ms"), "cal.B")

        with pytest.raises(ValueError, match="inside the Measurement Set"):
            write_caltable(
                inside, gains, np.arange(N_TIME, dtype=float), ms_path=ms
            )

        self._ms_still_reads(ms)


class TestPartialWrites:
    """A write that fails part-way leaves nothing behind.

    The two halves of the guarantee are different: a caller's mistake is caught
    before the old table is removed and costs nothing, but an I/O failure after
    that point cannot bring the old table back. What it must not do is leave a
    half-written table sitting where a valid one is expected.
    """

    # KeyboardInterrupt as well as an ordinary error: a Ctrl-C mid-write is the
    # likeliest way to strand a partial table, and it is not an Exception, so
    # narrowing the cleanup to `except Exception` would pass every other check
    # here while leaving exactly that case broken.
    @pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
    def test_a_failure_mid_write_leaves_no_partial_table(
        self, tmp_path, gains, ms_path, monkeypatch, failure
    ):
        import casacore.tables

        path = str(tmp_path / "test.B")
        real_table = casacore.tables.table

        def flaky(name, *args, **kwargs):
            # Once the output exists the validation reads are done, so this is
            # the subtable copy: the first genuinely mid-write step.
            if os.path.exists(path) and str(name).startswith(ms_path):
                raise failure("the disk went away")

            return real_table(name, *args, **kwargs)

        monkeypatch.setattr(casacore.tables, "table", flaky)

        with pytest.raises(failure, match="disk"):
            write_caltable(path, gains, np.arange(N_TIME, dtype=float), ms_path=ms_path)

        assert not os.path.exists(path)

    def test_a_cleanup_failure_does_not_replace_the_original_error(
        self, tmp_path, gains, ms_path, monkeypatch
    ):
        """Removal is best effort; the error that caused it always propagates."""

        import casacore.tables

        path = str(tmp_path / "test.B")
        real_table = casacore.tables.table

        def flaky(name, *args, **kwargs):
            if os.path.exists(path) and str(name).startswith(ms_path):
                raise RuntimeError("the disk went away")

            return real_table(name, *args, **kwargs)

        def unremovable(*args, **kwargs):
            raise OSError("the directory is not going anywhere either")

        monkeypatch.setattr(casacore.tables, "table", flaky)
        monkeypatch.setattr(shutil, "rmtree", unremovable)

        # The write's failure, not the cleanup's.
        with pytest.raises(RuntimeError, match="disk"):
            write_caltable(path, gains, np.arange(N_TIME, dtype=float), ms_path=ms_path)


# ---------------------------------------------------------------------------
# The convention
# ---------------------------------------------------------------------------

class TestGainConvention:

    def test_gain_convention(self, gains):
        """V_obs = g_p conj(g_q) V_true, so calibrating divides that out exactly."""

        a1 = np.array([0, 0, 1, 2])
        a2 = np.array([1, 2, 2, 3])

        rng = np.random.default_rng(1)
        shape = (len(a1), N_FREQ, N_TIME)
        vis_true = rng.normal(size=shape) + 1j * rng.normal(size=shape)

        g_bl = baseline_gains(gains, a1, a2)
        vis_obs = g_bl * vis_true  # forward: corrupt with the gains
        vis_cal, _ = apply_gains_to_data(vis_obs, gains, a1, a2)

        assert np.allclose(vis_cal, vis_true, rtol=1e-8, atol=1e-10)

    def test_baseline_gains_conjugates_the_second_antenna(self, gains):
        """g_p conj(g_q), not conj(g_p) g_q and not |g|**2."""

        a1, a2 = np.array([0, 1]), np.array([1, 2])
        g_bl = baseline_gains(gains, a1, a2)

        assert np.allclose(g_bl, gains[a1] * np.conj(gains[a2]))
        assert not np.allclose(g_bl, np.conj(gains[a1]) * gains[a2])

    def test_noise_and_weight_transform(self, gains):
        """sigma follows the data: sigma_cal = sigma / |g|, i.e. weight_cal = weight |g|^2."""

        a1, a2 = np.array([0, 1]), np.array([1, 2])
        sigma = np.array([2.0, 5.0])[:, None, None]

        g_bl = baseline_gains(gains, a1, a2)
        _, sigma_cal = apply_gains_to_data(
            np.ones((2, N_FREQ, N_TIME), complex), gains, a1, a2, sigma
        )

        assert np.allclose(sigma_cal, sigma / np.abs(g_bl))
        weight, weight_cal = 1.0 / sigma**2, 1.0 / sigma_cal**2
        assert np.allclose(weight_cal, weight * np.abs(g_bl) ** 2)

    def test_no_sigma_means_no_calibrated_sigma(self, gains):
        a1, a2 = np.array([0, 1]), np.array([1, 2])
        _, sigma_cal = apply_gains_to_data(
            np.ones((2, N_FREQ, N_TIME), complex), gains, a1, a2
        )

        assert sigma_cal is None

    @pytest.mark.parametrize("dead", [0.0, np.nan, np.inf])
    def test_a_dead_gain_calibrates_to_nan(self, gains, dead):
        """Zero divides to Inf, which reads as a real number downstream.

        The docstring promises NaN for every dead gain, so all of them have to
        arrive as NaN -- a caller flagging on ``isnan`` would keep the Inf.
        """

        g = gains.copy()
        g[1] = dead
        a1, a2 = np.array([0, 0]), np.array([1, 2])
        sigma = np.array([2.0, 5.0])[:, None, None]

        vis_cal, sigma_cal = apply_gains_to_data(
            np.ones((2, N_FREQ, N_TIME), complex), g, a1, a2, sigma
        )

        assert np.all(np.isnan(vis_cal[0]))
        assert np.all(np.isnan(sigma_cal[0]))
        assert np.all(np.isfinite(vis_cal[1]))
        assert np.all(np.isfinite(sigma_cal[1]))

    def test_scalar_flux_scale_is_g_k_minus_half(self):
        """flux-calibrate's V_cal = k V_obs is the antenna-independent gain g = k**-0.5."""

        k = 1700.0
        g = np.full((4, 1, 1), k**-0.5, dtype=complex)
        a1, a2 = np.array([0, 1]), np.array([1, 2])

        vis = np.ones((2, 1, 1), dtype=complex)
        vis_cal, _ = apply_gains_to_data(vis, g, a1, a2)

        assert np.allclose(vis_cal, k * vis)


# ---------------------------------------------------------------------------
# Import cost
# ---------------------------------------------------------------------------

def test_importing_ms_does_not_pull_casacore_in():
    """``ms.py``'s module level stays dask-ms only; casacore is function-local.

    In a subprocess because this session has already imported casacore.tables at
    the top of this file, so ``sys.modules`` here says nothing.
    """

    code = (
        "import sys; import tabascal.ms; "
        "sys.exit(1 if 'casacore.tables' in sys.modules else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )

    assert result.returncode == 0, (
        f"importing tabascal.ms pulled casacore.tables in\n{result.stderr}"
    )
