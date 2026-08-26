"""Tests for tabascal.ms — row layout, correlation resolution, time scales."""

import numpy as np
import pytest

from tabascal.ms import (
    CORR_TYPES,
    DEFAULT_TIME_SCALE,
    fitted_correlation,
    grid_to_rows,
    into_corr,
    ms_layout,
    partition_polarization,
    partition_setup,
    read_time_scale,
    rows_to_grid,
    resolve_correlation,
    resolve_data_description,
)


class _FakePol:
    """One grouped POLARIZATION row.

    dask-ms with ``group_cols="__row__"`` yields one dataset per row, each
    keeping a leading row axis of length 1 -- so CORR_TYPE is (1, n_corr) and
    rows of differing width never have to share a shape.
    """

    def __init__(self, corr_type):
        self.CORR_TYPE = _FakeVar(np.asarray([corr_type]))


class _FakeVar:
    def __init__(self, values):
        self.data = values if hasattr(values, "compute") else _FakeData(values)


class _FakeData:
    def __init__(self, values):
        self._values = values

    def __getitem__(self, idx):
        return _FakeData(self._values[idx])

    def compute(self):
        return self._values


@pytest.fixture
def polarization(monkeypatch):
    """Patch the POLARIZATION read with a chosen CORR_TYPE row."""

    def _install(*rows):
        def fake_xds_from_table(path, group_cols=None):
            assert path.endswith("::POLARIZATION")
            assert group_cols == "__row__", "rows must be grouped, see variable shapes"
            return [_FakePol(r) for r in rows]

        monkeypatch.setattr("tabascal.ms.xds_from_table", fake_xds_from_table)

    return _install


# ---------------------------------------------------------------------------
# The case this exists for: a single-correlation MS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("corr", ["xx", "yy", "ll", "i"])
def test_single_correlation_ms_resolves_to_index_zero(polarization, corr):
    """An MS holding one polarisation has it at index 0, whatever it is.

    The conventional {xx: 0, ..., yy: 3} table returns 3 for 'yy', which is off
    the end of a length-1 correlation axis.
    """
    polarization([CORR_TYPES[corr]])

    assert resolve_correlation("fake.ms", corr) == 0


def test_single_correlation_ms_rejects_a_correlation_it_does_not_hold(polarization):
    """Asking for XX on a YY-only MS is an error, not a silent read of YY."""
    polarization([CORR_TYPES["yy"]])

    with pytest.raises(ValueError, match="does not contain correlation 'xx'"):
        resolve_correlation("fake.ms", "xx")


def test_the_error_names_what_the_ms_actually_holds(polarization):
    polarization([CORR_TYPES["yy"]])

    with pytest.raises(ValueError, match="It holds: yy"):
        resolve_correlation("fake.ms", "xx")


# ---------------------------------------------------------------------------
# The other layouts the positional table gets wrong
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "corr, expected", [("xx", 0), ("xy", 1), ("yx", 2), ("yy", 3)]
)
def test_full_four_correlation_ms_matches_the_conventional_order(
    polarization, corr, expected
):
    """The case the old positional table handled; unchanged."""
    polarization([CORR_TYPES[c] for c in ("xx", "xy", "yx", "yy")])

    assert resolve_correlation("fake.ms", corr) == expected


@pytest.mark.parametrize("corr, expected", [("xx", 0), ("yy", 1)])
def test_two_correlation_ms(polarization, corr, expected):
    """An (XX, YY) MS puts YY at index 1, where the positional table says 3."""
    polarization([CORR_TYPES["xx"], CORR_TYPES["yy"]])

    assert resolve_correlation("fake.ms", corr) == expected


def test_non_conventional_ordering_is_followed(polarization):
    """The axis order is read, not assumed."""
    polarization([CORR_TYPES["yy"], CORR_TYPES["xx"]])

    assert resolve_correlation("fake.ms", "xx") == 1
    assert resolve_correlation("fake.ms", "yy") == 0


def test_circular_correlations(polarization):
    polarization([CORR_TYPES[c] for c in ("rr", "rl", "lr", "ll")])

    assert resolve_correlation("fake.ms", "ll") == 3


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def test_correlation_name_is_case_insensitive(polarization):
    polarization([CORR_TYPES["yy"]])

    assert resolve_correlation("fake.ms", "YY") == 0


def test_unknown_correlation_name_is_rejected(polarization):
    polarization([CORR_TYPES["xx"]])

    with pytest.raises(ValueError, match="Unknown correlation 'zz'"):
        resolve_correlation("fake.ms", "zz")


def test_unreadable_polarization_raises(monkeypatch):
    """No positional fallback: guessing is the bug this function removes.

    POLARIZATION is a required subtable. Falling back to the conventional
    {xx: 0, ..., yy: 3} ordering would return 3 for a single-correlation MS --
    off the end of its axis, and exactly the failure #128 is about.
    """

    def broken(path, group_cols=None):
        raise RuntimeError("no such table")

    monkeypatch.setattr("tabascal.ms.xds_from_table", broken)

    with pytest.raises(ValueError, match="correlation layout is unknown"):
        resolve_correlation("fake.ms", "yy")


# ---------------------------------------------------------------------------
# Time scale, from the TIME column's MEASINFO record
# ---------------------------------------------------------------------------

def _keywords(ref=None, column="TIME"):
    """Column keywords as dask-ms returns them, optionally declaring a scale."""

    measinfo = {"type": "epoch"}
    if ref is not None:
        measinfo["Ref"] = ref

    return {column: {"QuantumUnits": ["s"], "MEASINFO": measinfo}}


def test_reads_the_declared_scale():
    assert read_time_scale(_keywords("UTC")) == "utc"


@pytest.mark.parametrize("ref", ["UTC", "TAI", "TT", "UT1", "TDB"])
def test_every_declared_scale_is_returned_lowercased(ref):
    assert read_time_scale(_keywords(ref)) == ref.lower()


def test_a_non_utc_scale_is_reported_as_declared():
    """Not silently coerced to UTC -- the caller decides what to do about it."""
    assert read_time_scale(_keywords("TAI")) == "tai"


def test_missing_measinfo_ref_falls_back_with_a_warning(capsys):
    assert read_time_scale(_keywords(None)) == DEFAULT_TIME_SCALE
    assert "no MEASINFO Ref" in capsys.readouterr().out


def test_missing_column_falls_back_with_a_warning(capsys):
    assert read_time_scale({}) == DEFAULT_TIME_SCALE
    assert "no MEASINFO Ref" in capsys.readouterr().out


def test_none_keywords_fall_back():
    assert read_time_scale(None) == DEFAULT_TIME_SCALE


def test_a_different_column_can_be_read():
    keywords = _keywords("TAI", column="TIME_CENTROID")

    assert read_time_scale(keywords, column="TIME_CENTROID") == "tai"


# ---------------------------------------------------------------------------
# DATA_DESCRIPTION: which subtable rows the data actually uses
# ---------------------------------------------------------------------------

class _FakeDataDesc:
    def __init__(self, spw_ids, pol_ids):
        self.SPECTRAL_WINDOW_ID = _FakeVar(np.asarray(spw_ids))
        self.POLARIZATION_ID = _FakeVar(np.asarray(pol_ids))


@pytest.fixture
def data_description(monkeypatch):
    """Patch the DATA_DESCRIPTION read with a chosen id mapping."""

    def _install(spw_ids, pol_ids):
        def fake_xds_from_table(path, group_cols=None):
            assert path.endswith("::DATA_DESCRIPTION")
            return [_FakeDataDesc(spw_ids, pol_ids)]

        monkeypatch.setattr("tabascal.ms.xds_from_table", fake_xds_from_table)

    return _install


class TestResolveDataDescription:
    """An MS does not tie its data to row 0 of SPECTRAL_WINDOW / POLARIZATION."""

    def test_single_setup_resolves_to_zero(self, data_description):
        """The common case, and why assuming 0 usually works."""
        data_description([0], [0])

        assert resolve_data_description("fake.ms", 0) == (0, 0)

    def test_a_later_data_desc_id_selects_its_own_rows(self, data_description):
        """The bug: hardcoding row 0 reads another setup's configuration."""
        data_description([0, 1, 2], [0, 1, 1])

        assert resolve_data_description("fake.ms", 1) == (1, 1)
        assert resolve_data_description("fake.ms", 2) == (2, 1)

    def test_spw_and_pol_ids_are_independent(self, data_description):
        """Several windows can share one polarization setup, and vice versa."""
        data_description([3, 4], [1, 1])

        assert resolve_data_description("fake.ms", 0) == (3, 1)
        assert resolve_data_description("fake.ms", 1) == (4, 1)

    def test_unreadable_table_falls_back_with_a_warning(self, monkeypatch, capsys):
        def broken(path, group_cols=None):
            raise RuntimeError("no such table")

        monkeypatch.setattr("tabascal.ms.xds_from_table", broken)

        assert resolve_data_description("fake.ms", 1) == (0, 0)
        assert "Warning" in capsys.readouterr().out


class TestCorrelationUsesTheRightPolarizationRow:

    def test_pol_id_selects_the_row(self, polarization):
        """Row 1 holds YY only; row 0 holds the full four."""
        polarization(
            [CORR_TYPES[c] for c in ("xx", "xy", "yx", "yy")],
            [CORR_TYPES["yy"]],
        )

        assert resolve_correlation("fake.ms", "yy", pol_id=0) == 3
        assert resolve_correlation("fake.ms", "yy", pol_id=1) == 0

    def test_reading_row_zero_would_accept_an_absent_correlation(self, polarization):
        """The failure the fix prevents: row 0 holds XX, the data's row does not."""
        polarization([CORR_TYPES["xx"]], [CORR_TYPES["yy"]])

        assert resolve_correlation("fake.ms", "xx", pol_id=0) == 0
        with pytest.raises(ValueError, match="does not contain correlation 'xx'"):
            resolve_correlation("fake.ms", "xx", pol_id=1)

    def test_defaults_to_row_zero(self, polarization):
        polarization([CORR_TYPES["xx"]], [CORR_TYPES["yy"]])

        assert resolve_correlation("fake.ms", "xx") == 0


class TestHeterogeneousPolarizationRows:
    """Rows of differing NUM_CORR: the case an ungrouped read cannot represent.

    CORR_TYPE is a variable-shaped CASA column. Read ungrouped, dask-ms describes
    the whole subtable with one exemplar row's width and fails on any row that
    differs -- so a four-correlation setup beside a YY-only one breaks precisely
    the single-correlation support #128 adds.
    """

    def test_wide_row_beside_narrow_row(self, polarization):
        polarization(
            [CORR_TYPES[c] for c in ("xx", "xy", "yx", "yy")],   # 4 wide
            [CORR_TYPES["yy"]],                                   # 1 wide
        )

        assert resolve_correlation("fake.ms", "yy", pol_id=0) == 3
        assert resolve_correlation("fake.ms", "yy", pol_id=1) == 0

    def test_narrow_row_first(self, polarization):
        """Order must not matter; neither row is the exemplar."""
        polarization(
            [CORR_TYPES["yy"]],
            [CORR_TYPES[c] for c in ("xx", "xy", "yx", "yy")],
        )

        assert resolve_correlation("fake.ms", "yy", pol_id=0) == 0
        assert resolve_correlation("fake.ms", "yy", pol_id=1) == 3

    def test_two_correlation_row_beside_four(self, polarization):
        polarization(
            [CORR_TYPES[c] for c in ("xx", "xy", "yx", "yy")],
            [CORR_TYPES["xx"], CORR_TYPES["yy"]],
        )

        assert resolve_correlation("fake.ms", "yy", pol_id=1) == 1
        with pytest.raises(ValueError, match="does not contain correlation 'xy'"):
            resolve_correlation("fake.ms", "xy", pol_id=1)


# ---------------------------------------------------------------------------
# Row layout: the (n_time, n_bl) reshape every reader and writer relies on
# ---------------------------------------------------------------------------

class _FakeRows:
    """An MS partition stripped to the three columns ms_layout reads.

    dask-backed, as the daskms dataset is, in row chunks that deliberately do
    not divide the baseline count: the layout checks must reshape across chunk
    boundaries the way they will on a real MS.
    """

    ROW_CHUNK = 5

    def __init__(self, a1, a2, times=None, attrs=None):
        import dask.array as da

        a1, a2 = np.asarray(a1), np.asarray(a2)
        if times is None:
            times = np.zeros(len(a1))

        column = lambda values: _FakeVar(da.from_array(values, chunks=self.ROW_CHUNK))
        self.ANTENNA1 = column(a1)
        self.ANTENNA2 = column(a2)
        self.TIME = column(np.asarray(times))
        self.attrs = {} if attrs is None else attrs


def _time_major(n_ant: int = 4, n_time: int = 3):
    """A well-formed time-major partition and its one block of pairs."""

    a1_bl, a2_bl = np.triu_indices(n_ant, k=1)
    n_bl = len(a1_bl)
    times = np.repeat(np.arange(n_time, dtype=float), n_bl)

    return _FakeRows(np.tile(a1_bl, n_time), np.tile(a2_bl, n_time), times), a1_bl, a2_bl


class TestMSLayout:
    """The four facts the reshape needs, derived once for reader and writer."""

    def test_derives_the_grid_and_the_antenna_pairs(self):
        xds, a1_bl, a2_bl = _time_major(n_ant=4, n_time=3)

        layout = ms_layout(xds)

        assert (layout.n_time, layout.n_bl) == (3, 6)
        np.testing.assert_array_equal(layout.a1, a1_bl)
        np.testing.assert_array_equal(layout.a2, a2_bl)

    def test_reads_both_antenna_columns(self):
        """The regression guard: reading ANTENNA1 twice was the original bug."""
        a1_col, a2_col = np.triu_indices(4, k=1)

        layout = ms_layout(_FakeRows(a1_col, a2_col))

        np.testing.assert_array_equal(layout.a1, a1_col)
        np.testing.assert_array_equal(layout.a2, a2_col)
        assert not np.array_equal(layout.a1, layout.a2)

    def test_takes_only_the_first_baseline_block(self):
        """The columns repeat per timestep; one block of baselines is wanted."""
        xds, a1_bl, a2_bl = _time_major(n_time=3)

        layout = ms_layout(xds)

        assert len(layout.a1) == len(a1_bl) and len(layout.a2) == len(a2_bl)
        np.testing.assert_array_equal(layout.a2, a2_bl)

    def test_ragged_row_counts_are_rejected(self):
        """Rows that do not divide into whole timesteps break the reshape."""
        a1_bl, a2_bl = np.triu_indices(4, k=1)
        xds = _FakeRows(
            np.tile(a1_bl, 2)[:-1],
            np.tile(a2_bl, 2)[:-1],
            np.repeat([0.0, 1.0], 6)[:-1],
        )

        with pytest.raises(ValueError, match="whole number of baselines"):
            ms_layout(xds)

    def test_baseline_major_ordering_is_rejected(self):
        """All times of one baseline first repeats each pair down the rows."""
        a1_bl, a2_bl = np.triu_indices(4, k=1)
        n_time = 3
        xds = _FakeRows(
            np.repeat(a1_bl, n_time),
            np.repeat(a2_bl, n_time),
            np.tile(np.arange(n_time, dtype=float), len(a1_bl)),
        )

        with pytest.raises(ValueError, match="not ordered time-major"):
            ms_layout(xds)

    def test_partially_repeated_pairs_are_rejected(self):
        a1_col, a2_col = np.triu_indices(4, k=1)
        a1_col, a2_col = a1_col.copy(), a2_col.copy()
        a1_col[-1], a2_col[-1] = a1_col[0], a2_col[0]   # one duplicate pair

        with pytest.raises(ValueError, match="distinct antenna pairs"):
            ms_layout(_FakeRows(a1_col, a2_col))

    def test_a_permuted_timestep_is_rejected(self):
        a1_bl, a2_bl = np.triu_indices(4, k=1)
        perm = np.array([3, 1, 0, 5, 4, 2])
        xds = _FakeRows(
            np.concatenate([a1_bl, a1_bl[perm]]),
            np.concatenate([a2_bl, a2_bl[perm]]),
            np.repeat([0.0, 1.0], 6),
        )

        with pytest.raises(ValueError, match="differs between timesteps"):
            ms_layout(xds)

    def test_interleaved_timesteps_are_rejected(self):
        """Pairs repeat per block and the rows reshape cleanly -- and every
        visibility still lands on the wrong timestamp."""
        a1_bl, a2_bl = np.triu_indices(3, k=1)          # 3 baselines
        xds = _FakeRows(
            np.tile(a1_bl, 2),
            np.tile(a2_bl, 2),
            np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        )

        with pytest.raises(ValueError, match="interleave timesteps"):
            ms_layout(xds)

    def test_blocks_out_of_ascending_time_order_are_accepted(self):
        """Only block-constancy matters: the time axis follows the blocks."""
        a1_bl, a2_bl = np.triu_indices(4, k=1)
        n_bl = len(a1_bl)
        xds = _FakeRows(
            np.tile(a1_bl, 3), np.tile(a2_bl, 3), np.repeat([2.0, 0.0, 1.0], n_bl)
        )

        layout = ms_layout(xds)

        assert layout.n_time == 3
        np.testing.assert_array_equal(layout.a1, a1_bl)


    def test_never_holds_a_full_column_in_memory(self):
        """Nothing larger than a chunk, or the n_time unique times, is ever a
        task result: the checks over the full columns are reductions.
        """
        import dask
        from dask.callbacks import Callback

        xds, _, _ = _time_major(n_ant=6, n_time=4)   # 60 rows, chunks of 5
        n_row = xds.TIME.data.shape[0]
        largest = []

        class Largest(Callback):
            def _posttask(self, key, result, dsk, state, worker_id):
                largest.append(int(getattr(result, "size", 0)))

        with dask.config.set(scheduler="synchronous"), Largest():
            layout = ms_layout(xds)

        assert layout.n_bl == 15 and layout.n_time == 4
        assert max(largest) < n_row
        assert max(largest) <= max(_FakeRows.ROW_CHUNK, layout.n_bl, layout.n_time) * 2

# ---------------------------------------------------------------------------
# Which subtable rows a partition uses
# ---------------------------------------------------------------------------

class TestPartitionSetup:
    """The partition names its own DATA_DESC_ID; row 0 is only a convention."""

    def test_the_partitions_id_is_resolved(self, data_description):
        data_description([0, 1], [0, 2])
        xds, _, _ = _time_major()
        xds.attrs = {"DATA_DESC_ID": 1}

        assert partition_setup("fake.ms", xds) == (1, 2)
        assert partition_polarization("fake.ms", xds) == 2

    def test_a_partition_without_the_attribute_falls_back_to_zero(
        self, data_description
    ):
        data_description([3], [4])
        xds, _, _ = _time_major()

        assert partition_setup("fake.ms", xds) == (3, 4)
        assert partition_polarization("fake.ms", xds) == 4


# ---------------------------------------------------------------------------
# Row/channel <-> (bl, freq, time)
# ---------------------------------------------------------------------------

class TestRowGridMapping:
    """The reshape the reader and the writer each used to spell out."""

    @pytest.fixture
    def grid(self):
        """A ``(bl, freq, time)`` array with every element distinguishable."""
        rng = np.random.default_rng(21)
        shape = (6, 3, 4)                       # 6 baselines, 3 chans, 4 times

        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    def test_round_trips(self, grid):
        n_bl, n_freq, n_time = grid.shape

        rows = grid_to_rows(grid, n_freq)
        back = rows_to_grid(rows[:, :, 0], n_time, n_bl, n_freq)

        np.testing.assert_array_equal(back, grid)

    def test_rows_are_time_major(self, grid):
        """Row ``t * n_bl + b`` holds baseline ``b`` at time ``t``."""
        n_bl, n_freq, n_time = grid.shape
        rows = grid_to_rows(grid, n_freq)

        assert rows.shape == (n_time * n_bl, n_freq, 1)
        for t in range(n_time):
            for b in range(n_bl):
                np.testing.assert_array_equal(
                    rows[t * n_bl + b, :, 0], grid[b, :, t]
                )

    def test_rows_to_grid_reproduces_the_readers_old_expression(self, grid):
        """The inline reshape/transpose read_ms used to carry."""
        n_bl, n_freq, n_time = grid.shape
        col = grid_to_rows(grid, n_freq)[:, :, 0]

        expected = np.transpose(col.reshape(n_time, n_bl, n_freq), (1, 2, 0))

        np.testing.assert_array_equal(rows_to_grid(col, n_time, n_bl, n_freq), expected)

    def test_grid_to_rows_reproduces_the_writers_old_expression(self, grid):
        """The inline transpose/reshape _to_ms_column used to carry."""
        n_freq = grid.shape[1]

        expected = np.transpose(grid, (2, 0, 1)).reshape(-1, n_freq, 1)

        np.testing.assert_array_equal(grid_to_rows(grid, n_freq), expected)

    def test_a_wider_correlation_axis_is_kept(self, grid):
        n_freq = grid.shape[1]

        assert grid_to_rows(grid, n_freq, n_corr=1).shape[2] == 1

    def test_works_on_dask_and_jax_arrays(self, grid, exact_rtol):
        """One mapping for the writer's dask arrays and the reader's jax ones."""
        da = pytest.importorskip("dask.array")
        jnp = pytest.importorskip("jax.numpy")
        n_bl, n_freq, n_time = grid.shape

        expected = grid_to_rows(grid, n_freq)

        np.testing.assert_allclose(
            np.asarray(grid_to_rows(da.from_array(grid, chunks=(3, 3, 2)), n_freq)),
            expected,
        )
        np.testing.assert_allclose(
            np.asarray(grid_to_rows(jnp.asarray(grid), n_freq)), expected, rtol=exact_rtol
        )


# ---------------------------------------------------------------------------
# Placing one fitted correlation on the MS's correlation axis
# ---------------------------------------------------------------------------

class TestIntoCorr:
    """One fitted correlation placed on a wider MS correlation axis."""

    @pytest.fixture
    def col(self):
        """A result on a length-1 correlation axis, as the writer builds it."""
        return np.arange(6, dtype=np.complex64).reshape(3, 2, 1) + 1.0

    def test_a_single_correlation_ms_is_left_alone(self, col):
        out = into_corr(col, 0, 1, 0)

        assert out is col

    def test_the_result_lands_on_the_fitted_correlation(self, col):
        out = into_corr(col, 2, 4, 0)

        assert out.shape == (3, 2, 4)
        np.testing.assert_array_equal(out[:, :, 2:3], col)

    def test_a_scalar_fill_covers_the_others(self, col):
        out = into_corr(col, 2, 4, 0)

        np.testing.assert_array_equal(out[:, :, [0, 1, 3]], 0.0)

    def test_an_array_fill_passes_its_own_values_through(self, col):
        """The data-frame columns keep the data on the correlations not fitted."""
        fill = (100 + np.arange(24)).astype(np.complex64).reshape(3, 2, 4)

        out = into_corr(col, 2, 4, fill)

        np.testing.assert_array_equal(out[:, :, [0, 1, 3]], fill[:, :, [0, 1, 3]])
        np.testing.assert_array_equal(out[:, :, 2:3], col)

    def test_the_dtype_is_preserved(self, col):
        assert into_corr(col, 2, 4, 0).dtype == np.complex64

    def test_works_on_dask_arrays(self, col):
        """write_results_ms passes dask arrays through this."""
        da = pytest.importorskip("dask.array")
        fill = (100 + np.arange(24)).astype(np.complex64).reshape(3, 2, 4)

        out = into_corr(
            da.from_array(col, chunks=(3, 2, 1)),
            2,
            4,
            da.from_array(fill, chunks=(3, 2, 4)),
        )

        np.testing.assert_array_equal(
            np.asarray(out), into_corr(col, 2, 4, fill)
        )


class TestFittedCorrelation:
    """Which correlation the results belong to, resolved by name."""

    @pytest.fixture
    def resolver(self, monkeypatch):
        """Stand in for the casacore-backed resolver, recording the name."""
        seen = {}

        def _resolve(ms_path, name, pol_id=0):
            seen["name"] = name
            seen["pol_id"] = pol_id
            return 3

        monkeypatch.setattr("tabascal.ms.resolve_correlation", _resolve)

        return seen

    def test_the_argument_is_resolved_by_name(self, resolver):
        assert fitted_correlation("ms", None, "yy", 4) == 3
        assert resolver["name"] == "yy"

    def test_the_zarr_attribute_is_used_when_no_argument_is_given(self, resolver):
        assert fitted_correlation("ms", "xy", None, 4) == 3
        assert resolver["name"] == "xy"

    def test_the_argument_wins_over_the_attribute(self, resolver):
        fitted_correlation("ms", "xx", "yy", 4)

        assert resolver["name"] == "yy"

    def test_a_single_correlation_ms_needs_no_name(self, resolver):
        """One correlation, one answer -- and nothing to resolve."""
        assert fitted_correlation("ms", None, None, 1) == 0
        assert "name" not in resolver

    def test_a_nameless_zarr_on_a_wide_ms_is_rejected(self, resolver):
        """Guessing would write the results into the wrong polarisation."""
        with pytest.raises(ValueError, match="does not record which one"):
            fitted_correlation("ms", None, None, 4)

    def test_the_error_says_how_to_fix_it(self, resolver):
        with pytest.raises(ValueError, match="tab2MS -c xx"):
            fitted_correlation("ms", None, None, 2)

    def test_the_partition_polarization_row_is_forwarded(self, resolver):
        """Row 0 is a convention, not where this partition's data lives."""
        fitted_correlation("ms", None, "yy", 4, pol_id=2)

        assert resolver["pol_id"] == 2

    def test_the_row_defaults_to_zero(self, resolver):
        fitted_correlation("ms", None, "yy", 4)

        assert resolver["pol_id"] == 0

    def test_an_index_off_the_partition_axis_is_rejected(self, resolver):
        """A wider POLARIZATION row than the data: index 3 on a 2-corr axis.

        Without this, into_corr would match nothing and silently write zero
        models and untouched data everywhere.
        """
        with pytest.raises(ValueError, match="resolves to index 3"):
            fitted_correlation("ms", None, "yy", 2)


# ---------------------------------------------------------------------------
# Backwards compatibility for the move out of tab_tools
# ---------------------------------------------------------------------------

class TestMovedNamesStayImportable:
    """read_ms and get_observation_data_type moved to tabascal.ms.

    The old import path keeps working so the move does not break callers that
    predate it, but warns so it does not become the permanent home.
    """

    @pytest.mark.parametrize("name", ["read_ms", "get_observation_data_type"])
    def test_old_import_path_still_resolves(self, name):
        import tabascal.ms
        import tabascal.tab_tools

        with pytest.warns(DeprecationWarning, match="moved to tabascal.ms"):
            moved = getattr(tabascal.tab_tools, name)

        assert moved is getattr(tabascal.ms, name)

    def test_unknown_attribute_still_raises_attribute_error(self):
        import tabascal.tab_tools

        with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
            tabascal.tab_tools.nonexistent

    def test_no_warning_when_importing_from_the_new_home(self, recwarn):
        import importlib

        importlib.reload(importlib.import_module("tabascal.ms"))

        assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]
