"""Tests for tabascal.ms — correlation resolution and time-scale reading."""

import numpy as np
import pytest

from tabascal.ms import (
    CORR_TYPES,
    DEFAULT_TIME_SCALE,
    read_time_scale,
    resolve_correlation,
    resolve_data_description,
)


class _FakePol:
    """Stand-in for the POLARIZATION subtable's xarray dataset.

    ``rows`` is one CORR_TYPE list per POLARIZATION row.
    """

    def __init__(self, *rows):
        self.CORR_TYPE = _FakeVar(np.asarray(rows, dtype=object))


class _FakeVar:
    def __init__(self, values):
        self.data = _FakeData(values)


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
        def fake_xds_from_table(path):
            assert path.endswith("::POLARIZATION")
            return [_FakePol(*rows)]

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


def test_unreadable_polarization_falls_back_with_a_warning(monkeypatch, capsys):
    """POLARIZATION is mandatory, so a failure is unusual -- but stay readable."""

    def broken(path):
        raise RuntimeError("no such table")

    monkeypatch.setattr("tabascal.ms.xds_from_table", broken)

    assert resolve_correlation("fake.ms", "yy") == 3
    assert "Warning" in capsys.readouterr().out


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
        def fake_xds_from_table(path):
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
        def broken(path):
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
