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


def _minimal_ms(path: str, n_ant: int = N_ANT, freqs=FREQS) -> str:
    """An MS carrying only the subtables a caltable copies out of one.

    Enough of an MS for :func:`write_caltable`: it reads no main-table column,
    only ``tablecopy``s the subtables it finds. Written here rather than taken
    from the other MS tests because those build in-memory xarray stand-ins, and
    ``tablecopy`` needs a real table on disk.
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

    spw = table(
        os.path.join(path, "SPECTRAL_WINDOW"),
        maketabdesc(
            [
                makearrcoldesc("CHAN_FREQ", 0.0, ndim=1, valuetype="double"),
                makescacoldesc("NUM_CHAN", 0, valuetype="int"),
            ]
        ),
        nrow=1,
        ack=False,
    )
    spw.putcol("CHAN_FREQ", np.asarray(freqs, dtype=float)[None])
    spw.putcol("NUM_CHAN", np.array([len(freqs)], dtype=np.int32))
    spw.close()

    main.close()

    return path


@pytest.fixture
def ms_path(tmp_path):
    """A minimal on-disk MS to copy subtables from."""

    return _minimal_ms(str(tmp_path / "minimal.ms"))


# ---------------------------------------------------------------------------
# The format CASA keys off
# ---------------------------------------------------------------------------

class TestCasaFormat:

    def test_casa_format(self, tmp_path, gains, ms_path):
        """CASA identifies a caltable by its INFO record; applycal rejects it otherwise."""

        path = str(tmp_path / "test.B")
        times = np.arange(N_TIME, dtype=float)
        write_caltable(path, gains, times, ms_path=ms_path)

        tb = table(path, ack=False)
        assert tb.info()["type"] == "Calibration"
        assert tb.info()["subType"] == "B Jones"
        assert tb.getkeyword("ParType") == "Complex"

        assert tb.nrows() == N_ANT * N_TIME
        # CASA writes 2 pols even for a single-correlation MS.
        assert tb.getcell("CPARAM", 0).shape == (N_FREQ, 2)
        # One row per (time, antenna), with no second antenna.
        assert set(np.unique(tb.getcol("ANTENNA2"))) == {-1}
        assert sorted(np.unique(tb.getcol("ANTENNA1"))) == list(range(N_ANT))
        tb.close()

    def test_the_subtables_are_copied_from_the_ms(self, tmp_path, gains, ms_path):
        """A caltable carries its own copy of the MS's description subtables."""

        path = str(tmp_path / "test.B")
        write_caltable(path, gains, np.arange(N_TIME, dtype=float), ms_path=ms_path)

        tb = table(path, ack=False)
        for sub in ("ANTENNA", "SPECTRAL_WINDOW"):
            assert os.path.exists(os.path.join(path, sub))
            # Registered by absolute path, as CASA registers them.
            assert tb.getkeyword(sub) == "Table: " + os.path.abspath(
                os.path.join(path, sub)
            )

        assert tb.getkeyword("MSName") == "minimal.ms"
        tb.close()

        ant = table(os.path.join(path, "ANTENNA"), ack=False)
        assert ant.nrows() == N_ANT
        ant.close()

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

        tb = table(path, ack=False)
        assert np.array_equal(tb.getcol("ANTENNA1"), np.tile(np.arange(N_ANT), N_TIME))
        assert np.array_equal(tb.getcol("TIME"), np.repeat(times, N_ANT))
        tb.close()


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

        tb = table(path, ack=False)
        flag = tb.getcol("FLAG")  # (n_row, n_freq, n_pol), time-major rows
        tb.close()

        assert flag[3, 1].all()  # antenna 3, channel 1, first timestep
        assert not flag[3, 0].any()
        assert np.all(np.isnan(read_caltable(path)["gains"][3, 1, 0]))

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

    @pytest.mark.parametrize(
        "shape, message",
        [
            ((N_ANT, N_FREQ), "n_ant, n_freq, n_time"),
            ((N_ANT, N_FREQ, N_TIME, 1), "n_ant, n_freq, n_time"),
        ],
    )
    def test_gains_must_carry_all_three_axes(self, tmp_path, shape, message):
        with pytest.raises(ValueError, match=message):
            write_caltable(
                str(tmp_path / "test.B"),
                np.ones(shape, dtype=complex),
                np.arange(N_TIME, dtype=float),
                ms_path=str(tmp_path / "fake.ms"),
            )

    def test_times_must_match_the_gains_time_axis(self, tmp_path, gains):
        with pytest.raises(ValueError, match="times"):
            write_caltable(
                str(tmp_path / "test.B"),
                gains,
                np.arange(N_TIME + 1, dtype=float),
                ms_path=str(tmp_path / "fake.ms"),
            )


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

    def test_a_flagged_gain_calibrates_to_nan(self, gains):
        """Dividing by a dead antenna's gain is a NaN for the caller to flag."""

        g = gains.copy()
        g[1] = np.nan
        a1, a2 = np.array([0, 0]), np.array([1, 2])

        vis_cal, _ = apply_gains_to_data(
            np.ones((2, N_FREQ, N_TIME), complex), g, a1, a2
        )

        assert np.all(np.isnan(vis_cal[0]))
        assert np.all(np.isfinite(vis_cal[1]))

    def test_scalar_flux_scale_is_g_k_minus_half(self):
        """flux-calibrate's V_cal = k V_obs is the antenna-independent gain g = k**-0.5."""

        k = 1700.0
        g = np.full((4, 1, 1), k**-0.5, dtype=complex)
        a1, a2 = np.array([0, 1]), np.array([1, 2])

        vis = np.ones((2, 1, 1), dtype=complex)
        vis_cal, _ = apply_gains_to_data(vis, g, a1, a2)

        assert np.allclose(vis_cal, k * vis)
