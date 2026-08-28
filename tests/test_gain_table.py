"""Tests for consuming externally-solved gain tables (``data.gain_table``).

The interpolation semantics are the subject: a calibration solved on one
(time, frequency) grid has to be placed on the MS's grid, and *how* that is done
is a choice that quietly changes the answer. Interpolating the real and
imaginary parts is the obvious implementation and the wrong one -- two unit
gains 60 degrees apart average to ``|g| = 0.87``, so the data would be scaled by
a gain the instrument never had. The tests below pin amplitude and unwrapped
phase instead, and the 60-degree case is written out explicitly.

Most of them work on the dict :func:`tabascal.ms.read_caltable` returns, so they
need no casacore; the ones that go through a real table on disk take it from
:func:`tabascal.ms.write_caltable` and skip where casacore is missing.
"""

import os

import jax.numpy as jnp
import numpy as np
import pytest

from tabascal.gain_table import (
    Coverage,
    compose_gains,
    gains_from_tables,
    interpolate_gains,
    normalise_gain_tables,
)


DAY_SECS = 86400.0


def _cal(gains, times, freqs):
    """The dict :func:`read_caltable` returns, without a table on disk."""

    return {
        "gains": np.asarray(gains, dtype=complex),
        "times": None if times is None else np.asarray(times, dtype=float),
        "freqs": None if freqs is None else np.asarray(freqs, dtype=float),
        "ant_idx": np.arange(np.shape(gains)[0]),
        "viscal": "B Jones",
    }


def _one_ant(values, times, freqs):
    """A single-antenna table from a ``(n_freq, n_time)`` block of gains."""

    return _cal(np.asarray(values, dtype=complex)[None], times, freqs)


# ---------------------------------------------------------------------------
# Amplitude and unwrapped phase, never real and imaginary
# ---------------------------------------------------------------------------

class TestAmplitudeAndPhase:

    def test_two_unit_gains_60_degrees_apart_keep_unit_amplitude(self):
        """The case that decides the whole scheme.

        Real/imaginary interpolation of ``exp(+-i pi/6)`` gives ``cos(pi/6) =
        0.866``, i.e. a 13 % flux error invented by the interpolator. Amplitude
        and phase keep the unit gain the instrument actually had.
        """

        g = np.exp(1j * np.array([-np.pi / 6, np.pi / 6]))
        cal = _one_ant(g[None, :], times=[0.0, 10.0], freqs=[1e9])

        got = interpolate_gains(cal, times=[5.0], freqs=[1e9]).gains

        assert np.abs(got[0, 0, 0]) == pytest.approx(1.0, abs=1e-12)
        assert np.angle(got[0, 0, 0]) == pytest.approx(0.0, abs=1e-12)
        # The implementation it is NOT: (g0 + g1) / 2.
        assert np.abs(g.mean()) == pytest.approx(np.cos(np.pi / 6), abs=1e-12)

    def test_amplitude_is_linear_between_solutions(self):
        cal = _one_ant([[1.0 + 0j, 3.0 + 0j]], times=[0.0, 10.0], freqs=[1e9])

        got = interpolate_gains(cal, times=[0.0, 2.5, 5.0, 10.0], freqs=[1e9]).gains

        assert np.allclose(np.abs(got[0, 0]), [1.0, 1.5, 2.0, 3.0])

    def test_phase_unwraps_across_a_wrap_in_time(self):
        """Two phases either side of +-pi interpolate the short way round."""

        phases = np.array([175.0, -175.0]) * np.pi / 180
        cal = _one_ant(np.exp(1j * phases)[None, :], times=[0.0, 10.0], freqs=[1e9])

        got = interpolate_gains(cal, times=[5.0], freqs=[1e9]).gains

        # 180 deg, not the 0 deg a naive mean of the wrapped phases gives.
        assert abs(np.angle(got[0, 0, 0])) == pytest.approx(np.pi, abs=1e-12)
        assert np.abs(got[0, 0, 0]) == pytest.approx(1.0, abs=1e-12)

    def test_phase_unwraps_across_a_wrap_in_freq(self):
        """A residual delay winds the phase across the band; B tables wrap in freq."""

        freqs = np.array([1e9, 1.1e9, 1.2e9])
        # A residual delay winding 0.8 turns across the band: the phase passes
        # through +-pi between the second and third channels, so the stored
        # angles are 0, 0.8 pi, -0.4 pi and only unwrapping makes them a ramp.
        phases = np.array([0.0, 0.8, 1.6]) * np.pi
        cal = _one_ant(np.exp(1j * phases)[:, None], times=[0.0], freqs=freqs)

        got = interpolate_gains(cal, times=[0.0], freqs=[1.05e9, 1.15e9]).gains

        assert np.angle(got[0, 0, 0]) == pytest.approx(0.4 * np.pi, abs=1e-9)
        # 1.2 pi, wrapped back into (-pi, pi]. Interpolating the *stored* angles
        # would give 0.2 pi here -- a gain rotated by a whole radian.
        assert np.angle(got[0, 1, 0]) == pytest.approx(-0.8 * np.pi, abs=1e-9)

    def test_the_two_axes_share_one_phase_branch(self):
        """The phase is one surface, not a stack of independently-branched lines.

        Three channels whose phase passes through +-pi between the two solved
        times. Reconstructing a complex gain after the frequency stage throws
        the unwrap branch away, and the time stage then picks a branch per
        channel -- tearing the band by a whole turn at whichever channel crossed
        the cut.
        """

        block = np.exp(1j * np.pi * np.array([[0.0, 0.85], [0.0, 0.95], [0.0, 1.05]]))
        cal = _one_ant(block, times=[0.0, 10.0], freqs=[1e9, 1.1e9, 1.2e9])

        got = interpolate_gains(cal, times=[5.0], freqs=[1e9, 1.1e9, 1.2e9]).gains

        assert np.allclose(
            np.angle(got[0, :, 0]), np.pi * np.array([0.425, 0.475, 0.525])
        )

    def test_a_half_turn_step_leaves_the_band_continuous(self):
        """The reproduction case: a median step of exactly half a turn.

        The two channels step 0.9 pi and 1.1 pi over the same interval, so the
        band's own shape is what says they belong on one branch -- a per-channel
        unwrap has nothing to go on and flips the second by 2 pi.
        """

        block = np.exp(1j * np.pi * np.array([[0.0, 0.9], [0.0, 1.1]]))
        cal = _one_ant(block, times=[0.0, 10.0], freqs=[1e9, 1.2e9])

        got = interpolate_gains(cal, times=[5.0], freqs=[1e9, 1.1e9, 1.2e9]).gains

        assert np.allclose(
            np.angle(got[0, :, 0]), np.pi * np.array([0.45, 0.50, 0.55])
        )

    def test_extrapolation_holds_the_edge_value_in_both_axes(self):
        block = np.array([[1.0 + 0j, 2.0 + 0j], [4.0 + 0j, 8.0 + 0j]])
        cal = _one_ant(block, times=[0.0, 10.0], freqs=[1e9, 2e9])

        got = interpolate_gains(
            cal, times=[-100.0, 0.0, 10.0, 110.0], freqs=[0.5e9, 1e9, 2e9, 3e9]
        ).gains

        # Beyond either end of either axis the nearest solved corner is held.
        assert got[0, 0, 0] == pytest.approx(1.0)
        assert got[0, -1, 0] == pytest.approx(4.0)
        assert got[0, 0, -1] == pytest.approx(2.0)
        assert got[0, -1, -1] == pytest.approx(8.0)

    def test_a_single_solution_applies_everywhere(self):
        """One solved sample is an edge in both axes, so it is held over the grid."""

        cal = _one_ant([[2.0 + 2.0j]], times=[5.0], freqs=[1.5e9])

        got = interpolate_gains(cal, times=[0.0, 5.0, 10.0], freqs=[1e9, 2e9]).gains

        assert np.allclose(got, 2.0 + 2.0j)

    def test_a_table_without_frequencies_is_broadcast_across_the_band(self):
        cal = _one_ant([[1.0 + 1.0j, 2.0 + 2.0j]], times=[0.0, 10.0], freqs=None)

        got = interpolate_gains(cal, times=[0.0, 10.0], freqs=[1e9, 2e9, 3e9]).gains

        assert got.shape == (1, 3, 2)
        assert np.allclose(got[0, :, 0], 1.0 + 1.0j)
        assert np.allclose(got[0, :, 1], 2.0 + 2.0j)

    def test_a_multichannel_table_without_frequencies_is_an_error(self):
        cal = _one_ant([[1.0 + 0j], [2.0 + 0j]], times=[0.0], freqs=None)

        with pytest.raises(ValueError, match="SPECTRAL_WINDOW"):
            interpolate_gains(cal, times=[0.0], freqs=[1e9, 2e9])


# ---------------------------------------------------------------------------
# Flagged solutions are bridged, not honoured
# ---------------------------------------------------------------------------

class TestBridging:

    def test_a_flagged_middle_solution_is_bridged(self):
        """A NaN is an absence, so the interpolation spans it from either side."""

        cal = _one_ant(
            [[1.0 + 0j, np.nan, 3.0 + 0j]], times=[0.0, 5.0, 10.0], freqs=[1e9]
        )

        got = interpolate_gains(cal, times=[2.5, 5.0, 7.5], freqs=[1e9]).gains

        assert np.allclose(np.abs(got[0, 0]), [1.5, 2.0, 2.5])
        assert not interpolate_gains(cal, times=[5.0], freqs=[1e9]).dead.any()

    def test_a_zero_solution_is_not_a_support_point(self):
        """A zero gain calibrates to infinity; it is an absence, like a NaN."""

        cal = _one_ant([[1.0 + 0j, 0.0 + 0j, 3.0 + 0j]], times=[0, 5, 10], freqs=[1e9])

        got = interpolate_gains(cal, times=[5.0], freqs=[1e9]).gains

        assert np.abs(got[0, 0, 0]) == pytest.approx(2.0)

    def test_a_flagged_channel_is_bridged_across_frequency(self):
        cal = _one_ant(
            np.array([[1.0 + 0j], [np.nan], [3.0 + 0j]]),
            times=[0.0],
            freqs=[1e9, 2e9, 3e9],
        )

        got = interpolate_gains(cal, times=[0.0], freqs=[1e9, 2e9, 3e9]).gains

        assert np.allclose(np.abs(got[0, :, 0]), [1.0, 2.0, 3.0])

    def test_a_time_with_no_valid_channel_is_bridged(self):
        """A wholly flagged timestep leaves the other times to span it."""

        block = np.array(
            [[1.0 + 0j, np.nan, 3.0 + 0j], [1.0 + 0j, np.nan, 3.0 + 0j]]
        )
        cal = _one_ant(block, times=[0.0, 5.0, 10.0], freqs=[1e9, 2e9])

        got = interpolate_gains(cal, times=[5.0], freqs=[1e9, 2e9]).gains

        assert np.allclose(np.abs(got[0, :, 0]), 2.0)

    def test_an_antenna_with_no_valid_solution_anywhere_is_dead(self):
        """Unity gain, and reported so the caller can flag it."""

        gains = np.ones((2, 1, 2), dtype=complex)
        gains[1] = np.nan
        cal = _cal(gains, times=[0.0, 10.0], freqs=[1e9])

        got = interpolate_gains(cal, times=[0.0, 5.0, 10.0], freqs=[1e9])

        assert not got.dead[0].any()
        assert got.dead[1].all()
        assert np.all(got.gains[1] == 1.0)
        assert np.isfinite(got.gains).all()

    def test_an_antenna_solved_on_one_channel_only_covers_the_band(self):
        """Edge-hold in frequency: a partly-solved antenna is not a dead one."""

        gains = np.full((1, 3, 1), np.nan, dtype=complex)
        gains[0, 1, 0] = 2.0
        cal = _cal(gains, times=[0.0], freqs=[1e9, 2e9, 3e9])

        got = interpolate_gains(cal, times=[0.0], freqs=[1e9, 2e9, 3e9])

        assert not got.dead.any()
        assert np.allclose(got.gains[0, :, 0], 2.0)


# ---------------------------------------------------------------------------
# Coverage reporting
# ---------------------------------------------------------------------------

class TestCoverage:

    def test_a_table_on_the_ms_grid_is_covered_exactly(self):
        cal = _one_ant([[1.0 + 0j, 2.0 + 0j]], times=[0.0, 10.0], freqs=[1e9])

        cov = interpolate_gains(cal, times=[0.0, 10.0], freqs=[1e9]).coverage

        assert cov == Coverage(exact=1.0, interpolated=0.0, edge_held=0.0, dead=0.0)

    def test_the_three_classes_are_reported_separately(self):
        cal = _one_ant([[1.0 + 0j, 2.0 + 0j]], times=[0.0, 10.0], freqs=[1e9])

        # One solved time, one between them, two beyond either end.
        cov = interpolate_gains(
            cal, times=[-5.0, 0.0, 5.0, 15.0], freqs=[1e9]
        ).coverage

        assert cov.exact == pytest.approx(0.25)
        assert cov.interpolated == pytest.approx(0.25)
        assert cov.edge_held == pytest.approx(0.5)
        assert cov.exact + cov.interpolated + cov.edge_held + cov.dead == 1.0

    def test_a_flagged_solution_on_the_grid_is_not_an_exact_cover(self):
        """The MS sample sits on a solved coordinate, but nothing was solved there."""

        cal = _one_ant(
            [[1.0 + 0j, np.nan, 3.0 + 0j]], times=[0.0, 5.0, 10.0], freqs=[1e9]
        )

        cov = interpolate_gains(cal, times=[5.0], freqs=[1e9]).coverage

        assert cov == Coverage(
            exact=0.0, interpolated=1.0, edge_held=0.0, dead=0.0
        )

    def test_a_non_rectangular_support_is_not_reported_as_a_rectangle(self):
        """Two solutions on a diagonal cover their own corners and hold an edge
        for the other two.

        An envelope taken over the two axes separately would call both of those
        interpolated: each sits inside the antenna's solved range on *each*
        axis, and inside nothing at all on the line it was actually placed by.
        """

        gains = np.full((1, 2, 2), np.nan, dtype=complex)
        gains[0, 0, 0] = 1.0
        gains[0, 1, 1] = 2.0
        cal = _cal(gains, times=[0.0, 10.0], freqs=[1e9, 2e9])

        cov = interpolate_gains(cal, times=[0.0, 10.0], freqs=[1e9, 2e9]).coverage

        assert cov.exact == pytest.approx(0.5)
        assert cov.edge_held == pytest.approx(0.5)
        assert cov.interpolated == 0.0
        assert cov.dead == 0.0

    def test_a_dead_antenna_is_reported_as_dead(self):
        gains = np.ones((2, 1, 1), dtype=complex)
        gains[1] = np.nan
        cal = _cal(gains, times=[0.0], freqs=[1e9])

        cov = interpolate_gains(cal, times=[0.0], freqs=[1e9]).coverage

        assert cov.dead == pytest.approx(0.5)
        assert cov.exact == pytest.approx(0.5)

    def test_one_coverage_line_is_printed_per_table(self, capsys):
        cal = _one_ant([[1.0 + 0j, 2.0 + 0j]], times=[0.0, 10.0], freqs=[1e9])

        compose_gains([cal, cal], times=[0.0, 10.0], freqs=[1e9], n_ant=1)

        out = capsys.readouterr().out
        assert out.count("exact") == 2
        assert out.count("edge-held") == 2


# ---------------------------------------------------------------------------
# A table solved on a master, applied to a subset carved out of it
# ---------------------------------------------------------------------------

class TestSubsetGrid:

    def test_a_master_table_covers_a_carved_out_subset_exactly(self):
        rng = np.random.default_rng(3)
        gains = rng.normal(size=(4, 4, 3)) + 1j * rng.normal(size=(4, 4, 3))
        table_freqs = np.array([1e8, 2e8, 3e8, 4e8])
        table_times = np.array([10.0, 20.0, 30.0])
        cal = _cal(gains, table_times, table_freqs)

        # The middle time and two of the four channels, deliberately out of order.
        got = interpolate_gains(cal, times=[20.0], freqs=[3e8, 1e8])

        assert got.gains.shape == (4, 2, 1)
        assert np.allclose(got.gains[:, 0, 0], gains[:, 2, 1])
        assert np.allclose(got.gains[:, 1, 0], gains[:, 0, 1])
        assert got.coverage.exact == 1.0

    def test_matching_is_by_value_within_the_tolerances(self):
        """Round-tripped coordinates differ in the last bits and still match."""

        cal = _one_ant([[1.0 + 0j]], times=[1e9], freqs=[1.4e9])

        got = interpolate_gains(cal, times=[1e9 + 5e-4], freqs=[1.4e9 * (1 + 5e-7)])

        assert got.coverage.exact == 1.0

    def test_a_time_outside_the_solved_range_is_edge_held_not_an_error(self):
        """The daint draft raised here; the locked semantics hold the edge."""

        cal = _one_ant([[1.0 + 0j, 2.0 + 0j]], times=[10.0, 20.0], freqs=[1e9])

        got = interpolate_gains(cal, times=[25.0], freqs=[1e9])

        assert got.gains[0, 0, 0] == pytest.approx(2.0)
        assert got.coverage.edge_held == 1.0


# ---------------------------------------------------------------------------
# Several tables: interpolate each, then compose
# ---------------------------------------------------------------------------

class TestComposition:

    def _ramp(self, times):
        """Amplitude ramping 1 -> 3 over ``times``, so a product is not linear."""

        return _one_ant([[1.0 + 0j, 3.0 + 0j]], times=times, freqs=[1e9])

    def test_each_table_is_interpolated_before_they_are_composed(self):
        """Composition and interpolation do not commute -- the order is the API.

        Two amplitudes ramping 1 -> 3 give ``2 * 2 = 4`` half way when each is
        interpolated first, and ``(1*1 + 3*3) / 2 = 5`` when the product is
        formed on the table grid and interpolated afterwards. 4 is the gain each
        antenna had; 5 is an artefact of interpolating a quadratic linearly.
        """

        cal = self._ramp([0.0, 10.0])

        gains, _ = compose_gains([cal, cal], times=[5.0], freqs=[1e9], n_ant=1)

        assert np.abs(gains[0, 0, 0]) == pytest.approx(4.0)

    def test_the_composition_is_the_product_of_the_interpolations(self):
        a = _one_ant([[1.0 + 0j, 3.0 + 0j]], times=[0.0, 10.0], freqs=[1e9])
        b = _one_ant([[np.exp(1j), 2.0 * np.exp(-1j)]], times=[0.0, 5.0], freqs=[2e9])
        times, freqs = [2.0, 7.0], [1e9, 1.5e9]

        gains, _ = compose_gains([a, b], times=times, freqs=freqs, n_ant=1)

        expected = (
            interpolate_gains(a, times, freqs).gains
            * interpolate_gains(b, times, freqs).gains
        )
        assert np.allclose(gains, expected)

    def test_tables_on_different_grids_compose(self):
        a = _one_ant([[1.0 + 0j, 3.0 + 0j]], times=[0.0, 10.0], freqs=[1e9])
        b = _one_ant(
            [[1.0 + 0j, 2.0 + 0j, 4.0 + 0j]], times=[0.0, 5.0, 10.0], freqs=[1e9]
        )

        gains, _ = compose_gains([a, b], times=[5.0], freqs=[1e9], n_ant=1)

        assert np.abs(gains[0, 0, 0]) == pytest.approx(2.0 * 2.0)

    def test_a_dead_antenna_in_any_table_is_dead_in_the_composition(self):
        good = _cal(np.ones((2, 1, 1)), times=[0.0], freqs=[1e9])
        gains = np.ones((2, 1, 1), dtype=complex)
        gains[1] = np.nan
        partly_dead = _cal(gains, times=[0.0], freqs=[1e9])

        composed, dead = compose_gains(
            [good, partly_dead], times=[0.0], freqs=[1e9], n_ant=2
        )

        assert not dead[0].any()
        assert dead[1].all()
        assert np.all(composed[1] == 1.0)

    def test_a_table_with_too_few_antennas_is_an_error(self):
        cal = _cal(np.ones((2, 1, 1)), times=[0.0], freqs=[1e9])

        with pytest.raises(ValueError, match="antenna"):
            compose_gains([cal], times=[0.0], freqs=[1e9], n_ant=4)

    def test_a_master_tables_extra_antennas_are_dropped(self, capsys):
        """Antenna ids index the ANTENNA subtable, so a master's leading rows
        are this observation's -- and the coverage reported is this
        observation's, not the master's."""

        solved = 2.0 * np.ones((5, 1, 1), dtype=complex)
        solved[3:] = np.nan  # antennas the master has and this observation does not
        cal = _cal(solved, times=[0.0], freqs=[1e9])

        gains, dead = compose_gains([cal], times=[0.0], freqs=[1e9], n_ant=3)

        assert gains.shape == (3, 1, 1)
        assert not dead.any()
        assert "0.0 % unsolved" in capsys.readouterr().out

    def test_two_solutions_on_one_coordinate_are_an_error(self):
        cal = _one_ant([[1.0 + 0j, 2.0 + 0j]], times=[5.0, 5.0], freqs=[1e9])

        with pytest.raises(ValueError, match="same"):
            compose_gains([cal], times=[5.0], freqs=[1e9], n_ant=1)


class TestNormalisation:

    def test_a_single_path_is_a_list_of_one(self, tmp_path):
        path = tmp_path / "cal.B"
        path.mkdir()

        assert normalise_gain_tables(str(path)) == [str(path)]

    def test_a_list_is_kept_in_order(self, tmp_path):
        first, second = tmp_path / "a.B", tmp_path / "b.B"
        first.mkdir()
        second.mkdir()

        got = normalise_gain_tables([str(second), str(first)])

        assert got == [str(second), str(first)]

    def test_nothing_configured_is_no_tables(self):
        assert normalise_gain_tables(None) == []
        assert normalise_gain_tables([]) == []

    def test_a_missing_table_is_named(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="gain_table"):
            normalise_gain_tables(str(tmp_path / "absent.B"))


# ---------------------------------------------------------------------------
# TabConfig.apply_gain_table
# ---------------------------------------------------------------------------

N_ANT, N_FREQ, N_TIME = 6, 4, 5
FREQS = 1e9 + 1e6 * np.arange(N_FREQ, dtype=float)
#: MJD days; an integration of 2 s, which the caltable carries as MJD seconds.
TIMES_MJD = 60000.0 + 2.0 * np.arange(N_TIME) / DAY_SECS

A1, A2 = np.triu_indices(N_ANT, k=1)
N_BL = len(A1)


@pytest.fixture
def casacore():
    return pytest.importorskip("casacore.tables")


def _minimal_ms(path: str, n_ant: int = N_ANT) -> str:
    """The subtables :func:`write_caltable` copies, and nothing else.

    ``write_caltable`` reads no main-table column, checks the antenna count and
    the channel count, and copies whichever subtables it finds -- so this is
    enough of an MS to write a caltable beside.
    """

    from casacore.tables import (
        makearrcoldesc,
        makescacoldesc,
        maketabdesc,
        table,
    )

    main = table(
        path,
        maketabdesc([makescacoldesc("TIME", 0.0, valuetype="double")]),
        nrow=1,
        ack=False,
    )
    main.close()

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
    spw.putcol("CHAN_FREQ", FREQS[None])
    spw.putcol("NUM_CHAN", np.array([N_FREQ], dtype=np.int32))
    spw.close()

    return path


def _write_table(tmp_path, gains, name="cal.B", times_mjd=TIMES_MJD):
    """A caltable on disk holding ``gains``, beside a minimal MS."""

    from tabascal.ms import write_caltable

    ms_path = str(tmp_path / "obs.ms")
    if not os.path.exists(ms_path):
        _minimal_ms(ms_path)

    return write_caltable(
        str(tmp_path / name),
        np.asarray(gains, dtype=complex),
        np.asarray(times_mjd) * DAY_SECS,
        ms_path,
    )


def _tab_config(vis_obs, noise, flags=None):
    """A ``TabConfig`` carrying only what ``apply_gain_table`` reads.

    The constructor wants an MS, a TLE preflight and an RFI sampling estimate,
    none of which say anything about applying a gain table, so the method is
    exercised on an instance holding exactly the attributes it consumes.
    """

    from tabascal.config import TabConfig

    config = TabConfig.__new__(TabConfig)
    config.n_ant, config.n_bl = N_ANT, N_BL
    config.n_freq, config.n_time = N_FREQ, N_TIME
    config.times_mjd = TIMES_MJD
    config.freqs = FREQS
    config.a1, config.a2 = A1, A2
    config.vis_obs = jnp.asarray(vis_obs)
    config.noise = noise
    config.noise_scalar = None if noise is None else 1.0
    config.flags = (
        jnp.zeros(np.shape(vis_obs), dtype=bool) if flags is None else jnp.asarray(flags)
    )

    return config


@pytest.fixture
def gains():
    """Non-unit, antenna-dependent gains, constant over frequency and time.

    Constant so that a per-baseline noise stays per-baseline through the
    calibration, which is what the noise-shape tests want to vary on their own.
    """

    rng = np.random.default_rng(0)
    amp = rng.uniform(0.5, 2.0, (N_ANT, 1, 1))
    phase = rng.uniform(-np.pi, np.pi, (N_ANT, 1, 1))

    return np.broadcast_to(amp * np.exp(1j * phase), (N_ANT, N_FREQ, N_TIME)).copy()


@pytest.fixture
def vis_obs():
    rng = np.random.default_rng(1)
    shape = (N_BL, N_FREQ, N_TIME)

    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(complex)


class TestApplyGainTable:

    def test_nothing_configured_leaves_the_data_alone(self, vis_obs):
        config = _tab_config(vis_obs, noise=0.5)
        # As the read left it, at whatever precision the run works in: the
        # subject is that nothing touched it, not what it holds.
        before = np.asarray(config.vis_obs)

        config.apply_gain_table(None)

        assert config.gain_table == []
        assert config.gain_flags is None
        assert np.array_equal(np.asarray(config.vis_obs), before)
        assert config.noise == 0.5

    def test_the_data_is_divided_by_the_baseline_gain(
        self, casacore, tmp_path, gains, vis_obs, exact_rtol
    ):
        from tabascal.interferometry import baseline_gains
        from tabascal.ms import read_caltable

        path = _write_table(tmp_path, gains)
        config = _tab_config(vis_obs, noise=0.5)

        config.apply_gain_table(path)

        # Against the gains as the table stores them (CPARAM is complex64), so
        # the two sides differ only by the run's own precision.
        g_bl = baseline_gains(read_caltable(path)["gains"], A1, A2)
        expected = jnp.asarray(vis_obs) / jnp.asarray(g_bl)
        assert np.allclose(
            np.asarray(config.vis_obs), np.asarray(expected), rtol=exact_rtol
        )

    def test_a_perfect_model_has_unit_chi2_in_the_calibrated_frame(
        self, casacore, tmp_path, gains
    ):
        """The point of carrying the noise with the data: get it wrong and the
        chi^2 is out by |g|^2."""

        from tabascal.tab_tools import reduced_chi2

        rng = np.random.default_rng(4)
        shape = (N_BL, N_FREQ, N_TIME)
        sigma = 0.3
        model = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        vis_cal = model + sigma * (rng.normal(size=shape) + 1j * rng.normal(size=shape))

        g_bl = np.asarray(gains)[A1] * np.asarray(gains)[A2].conj()
        path = _write_table(tmp_path, gains)
        # The MS's own noise is in the frame of its data: sigma * |g|.
        config = _tab_config(vis_cal * g_bl, noise=sigma * np.abs(g_bl))

        config.apply_gain_table(path)
        config.set_flags(False)

        chi2 = reduced_chi2(
            jnp.asarray(model), config.vis_obs, config.noise, config.flags
        )
        assert float(chi2) == pytest.approx(1.0, rel=0.15)

    @pytest.mark.parametrize("shape", ["scalar", "bl", "bl_freq", "bl_freq_time"])
    def test_every_noise_shape_ends_broadcast_and_divided(
        self, casacore, tmp_path, gains, vis_obs, shape, exact_rtol
    ):
        from tabascal.interferometry import baseline_gains
        from tabascal.ms import read_caltable
        from tabascal.noise import broadcast_to_vis, representative_sigma

        rng = np.random.default_rng(5)
        noise = {
            "scalar": 0.4,
            "bl": rng.uniform(0.1, 0.9, N_BL),
            "bl_freq": rng.uniform(0.1, 0.9, (N_BL, N_FREQ)),
            "bl_freq_time": rng.uniform(0.1, 0.9, (N_BL, N_FREQ, N_TIME)),
        }[shape]

        path = _write_table(tmp_path, gains)
        config = _tab_config(vis_obs, noise=noise)

        config.apply_gain_table(path)

        # Against the gains as the table stores them (CPARAM is complex64), so
        # the two sides differ only by the run's own precision.
        g_bl = baseline_gains(read_caltable(path)["gains"], A1, A2)
        expected = np.asarray(
            jnp.asarray(broadcast_to_vis(noise, (N_BL, N_FREQ, N_TIME)))
            / jnp.abs(jnp.asarray(g_bl))
        )

        assert config.noise.shape == (N_BL, N_FREQ, N_TIME)
        assert np.allclose(np.asarray(config.noise), expected, rtol=exact_rtol)
        assert config.noise_scalar == pytest.approx(
            representative_sigma(config.noise), rel=exact_rtol
        )

    def test_a_noiseless_ms_stays_noiseless(self, casacore, tmp_path, gains, vis_obs):
        """A gain table cannot rescue an MS with no noise -- set_noise decides that."""

        path = _write_table(tmp_path, gains)
        config = _tab_config(vis_obs, noise=None)

        config.apply_gain_table(path)

        assert config.noise is None
        assert config.noise_scalar is None

    def test_an_unsolved_antenna_is_flagged_even_with_flags_off(
        self, casacore, tmp_path, gains, vis_obs
    ):
        dead = np.asarray(gains).copy()
        dead[2] = np.nan
        path = _write_table(tmp_path, dead)
        config = _tab_config(vis_obs, noise=0.5)

        config.apply_gain_table(path)
        config.set_flags(False)

        touches_2 = (A1 == 2) | (A2 == 2)
        flags = np.asarray(config.flags)
        assert flags[touches_2].all()
        assert not flags[~touches_2].any()
        # Unity gain rather than a NaN, so nothing downstream sees an infinity.
        assert np.isfinite(np.asarray(config.vis_obs)).all()
        assert np.allclose(
            np.asarray(config.vis_obs)[touches_2], vis_obs[touches_2], atol=1e-6
        )

    def test_the_gain_flags_survive_flags_being_read_from_the_ms(
        self, casacore, tmp_path, gains, vis_obs
    ):
        dead = np.asarray(gains).copy()
        dead[2] = np.nan
        ms_flags = np.zeros((N_BL, N_FREQ, N_TIME), dtype=bool)
        ms_flags[0, 0, 0] = True

        path = _write_table(tmp_path, dead)
        config = _tab_config(vis_obs, noise=0.5, flags=ms_flags)

        config.apply_gain_table(path)
        config.set_flags(True)

        flags = np.asarray(config.flags)
        assert flags[0, 0, 0]
        assert flags[(A1 == 2) | (A2 == 2)].all()

    def test_the_ms_times_round_trip_through_the_writer(
        self, casacore, tmp_path, gains, vis_obs, capsys
    ):
        """The caltable's TIME is MS TIME in seconds; the config holds MJD days."""

        path = _write_table(tmp_path, gains)
        config = _tab_config(vis_obs, noise=0.5)

        config.apply_gain_table(path)

        out = capsys.readouterr().out
        assert "100.0 % exact" in out

    def test_an_ordered_list_composes_the_tables(
        self, casacore, tmp_path, gains, vis_obs, exact_rtol
    ):
        from tabascal.interferometry import baseline_gains
        from tabascal.ms import read_caltable

        first = _write_table(tmp_path, gains, name="first.B")
        second = _write_table(tmp_path, 2.0 * gains, name="second.B")

        config = _tab_config(vis_obs, noise=0.5)
        config.apply_gain_table([first, second])

        g_bl = baseline_gains(
            read_caltable(first)["gains"] * read_caltable(second)["gains"], A1, A2
        )
        expected = jnp.asarray(vis_obs) / jnp.asarray(g_bl)
        assert np.allclose(
            np.asarray(config.vis_obs), np.asarray(expected), rtol=exact_rtol
        )
        assert config.gain_table == [os.path.abspath(first), os.path.abspath(second)]


class TestDeclaredTimeScale:
    """A caltable's TIME is matched on the MS's own numbers, not on UTC.

    Since #158 the reader keeps two time coordinates: ``times_mjd``, the MS's
    ``TIME`` column converted in unit only and still on the scale the column
    declares, and ``times_jd``, the instants those numbers name normalised to
    UTC. A caltable's ``TIME`` is a copy of the MS's, so the match is
    declared-frame to declared-frame -- exact, and free of any leap-second
    question. Matching on ``times_jd`` instead would be 37 seconds out on a
    TAI-declared MS, which is four orders of magnitude past the tolerance.

    The relationship between the two is taken from ``time.to_utc_jd``, which is
    the function ``read_ms`` itself builds ``times_jd`` with; the reader's end
    of it is pinned by ``test_ms.py::TestDeclaredTimeScale``.
    """

    #: Leap seconds at the epoch TIMES_MJD sits on; they have stood at 37 since
    #: 2017, and a future one changes the offset from then on, never this one.
    LEAP_SECS = 37.0

    def _tai_config(self, vis_obs):
        """A config as ``read_ms`` leaves one for a TAI-declared MS."""

        from tabascal.time import mjd_to_jd, to_utc_jd

        config = _tab_config(vis_obs, noise=0.5)
        config.times_jd = to_utc_jd(mjd_to_jd(config.times_mjd), "tai")

        return config

    def test_the_two_coordinates_really_do_differ(self, vis_obs):
        """Otherwise the tests below would pass on either choice."""

        from tabascal.time import jd_to_mjd, mjd_to_jd

        config = self._tai_config(vis_obs)
        drift = (mjd_to_jd(config.times_mjd) - config.times_jd) * DAY_SECS

        assert np.allclose(drift, self.LEAP_SECS, atol=1e-3)
        assert np.abs(jd_to_mjd(config.times_jd) - config.times_mjd).max() > 0

    def test_the_table_matches_the_declared_times_exactly(
        self, casacore, tmp_path, gains, vis_obs, capsys
    ):
        path = _write_table(tmp_path, gains)
        config = self._tai_config(vis_obs)

        config.apply_gain_table(path)

        assert "100.0 % exact" in capsys.readouterr().out

    def test_matching_on_the_utc_times_would_miss_every_sample(
        self, casacore, tmp_path, gains, vis_obs
    ):
        """What the 37 seconds cost, stated as behaviour rather than arithmetic."""

        from tabascal.time import jd_to_mjd

        from tabascal.ms import read_caltable

        path = _write_table(tmp_path, gains)
        config = self._tai_config(vis_obs)

        coverage = interpolate_gains(
            read_caltable(path), jd_to_mjd(config.times_jd) * DAY_SECS, config.freqs
        ).coverage

        # Every sample lands 37 s past the last solved one, so the whole
        # observation would be calibrated by holding one edge.
        assert coverage.exact == 0.0
        assert coverage.edge_held == 1.0


# ---------------------------------------------------------------------------
# Where apply_gain_table sits in TabConfig.__init__
# ---------------------------------------------------------------------------

class TestInitOrdering:
    """The slot is the whole feature.

    Applied before ``set_noise`` the override would be scaled twice, or not at
    all; applied after ``set_flags`` the gain flags would never reach the flags;
    applied after the sharding block ``vis_obs`` would already be a global array
    and the local numpy division could not touch it. Every other test calls the
    method itself, so all of them would pass on any of those. Follows the
    pattern of ``test_noise.py::TestInitCallsTheReadBeforeTheOverride``.
    """

    STUBBED = [
        "read_ms_params",
        "set_noise",
        "apply_gain_table",
        "set_flags",
        "get_orbital_elements",
        "estimate_rfi_sampling",
        "_set_freqs_times",
        "set_elevation_mask",
    ]

    def _construct(self, monkeypatch, gain_table="cal.B", key=True, calls=None):
        from tabascal.config import TabConfig

        calls = [] if calls is None else calls
        leaves = {
            "read_ms_params": {
                "n_freq": N_FREQ,
                "times_jd": np.array([2460000.0]),
                "noise": None,
                "noise_scalar": None,
                "vis_obs": np.zeros((1, N_FREQ, 1), dtype=complex),
                "flags": np.zeros((1, N_FREQ, 1), dtype=bool),
            },
            "set_noise": {"noise": 0.7, "noise_scalar": 0.7},
            "get_orbital_elements": {"n_rfi": 0},
        }

        def recorder(name):
            def method(self, *args, **kwargs):
                calls.append((name, args))
                for key, value in leaves.get(name, {}).items():
                    setattr(self, key, value)

            return method

        for name in self.STUBBED:
            monkeypatch.setattr(TabConfig, name, recorder(name))

        config = {
            "model": {"components": [], "precision": "single"},
            "data": {
                "freq": 0,
                "corr": "xx",
                "data_col": "DATA",
                "noise": 0.7,
                "flags": False,
            },
            "rfi": {"n_int_time": 4, "n_int_freq": 1, "time_int_factor": 1.0},
            "satellites": {},
        }
        if key:
            config["data"]["gain_table"] = gain_table

        TabConfig(config, "never/read.ms")

        return [name for name, _ in calls], calls

    def test_the_table_is_applied_after_the_noise_and_before_the_flags(
        self, monkeypatch
    ):
        names, _ = self._construct(monkeypatch)

        assert names.index("set_noise") < names.index("apply_gain_table")
        assert names.index("apply_gain_table") < names.index("set_flags")

    def test_it_is_applied_before_the_arrays_are_made_global(self, monkeypatch):
        """Under sharding ``vis_obs`` becomes a global array at the end of the
        constructor, and a process-local division cannot touch one."""

        from tabascal import config as config_module

        calls = []

        def make_global(array, sharding):
            calls.append(("make_global", ()))
            return array

        monkeypatch.setattr(config_module, "sharding_enabled", lambda: True)
        monkeypatch.setattr(config_module, "replicated_sharding", lambda: None)
        monkeypatch.setattr(config_module, "make_global", make_global)

        names, _ = self._construct(monkeypatch, calls=calls)

        assert "make_global" in names
        assert names.index("apply_gain_table") < names.index("make_global")

    def test_it_is_applied_exactly_once(self, monkeypatch):
        names, _ = self._construct(monkeypatch)

        assert names.count("apply_gain_table") == 1

    def test_the_configured_value_is_what_reaches_it(self, monkeypatch):
        _, calls = self._construct(monkeypatch, gain_table=["a.B", "b.B"])

        applied = [args for name, args in calls if name == "apply_gain_table"]
        assert applied == [(["a.B", "b.B"],)]

    def test_a_config_without_the_key_still_runs(self, monkeypatch):
        """The base config supplies it, but a config dict written before it must
        not break."""

        names, calls = self._construct(monkeypatch, key=False)

        assert "apply_gain_table" in names
        assert [args for name, args in calls if name == "apply_gain_table"] == [(None,)]
