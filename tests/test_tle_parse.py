"""Tests for the shared TLE parser (:mod:`tabascal.satchecker.tle_parse`).

This module is the single parser used by both cache validation and orbital
element extraction, so these tests pin the identifier decoding (including
Alpha-5) and the guarantee that validation and parsing agree.
"""

import pytest

from tabascal.satchecker.tle_parse import (
    decode_norad_id,
    parse_tle_elements,
    tle_checksum,
    validate_tle_pair,
)

from .tle_helpers import jd, make_tle, with_checksum

_EPOCH = jd(2023, 2, 21, 13)
# Letters used by Alpha-5, in value order: I and O are excluded to avoid
# confusion with 1 and 0, so A=10 ... H=17, J=18 ... N=22, P=23 ... Z=33.
_ALPHA5_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"


class TestDecodeNoradId:

    @pytest.mark.parametrize(
        "field,expected",
        [("25544", 25544), ("00001", 1), ("99999", 99999),
         ("A0000", 100000), ("E8493", 148493), ("Z9999", 339999)],
    )
    def test_known_identifiers(self, field, expected):
        assert decode_norad_id(field) == expected

    def test_every_alpha5_letter_maps_to_its_spec_value(self):
        for i, ch in enumerate(_ALPHA5_LETTERS):
            assert decode_norad_id(f"{ch}1234") == (10 + i) * 10_000 + 1234

    @pytest.mark.parametrize("field", ["I1234", "O1234", "!1234", "e8493", "", "   "])
    def test_invalid_identifiers_raise(self, field):
        with pytest.raises(ValueError):
            decode_norad_id(field)


class TestValidateTlePair:

    def test_valid_pair_returns_decoded_id(self):
        assert validate_tle_pair(*make_tle(25544, _EPOCH)) == 25544

    def test_alpha5_pair_is_valid(self):
        l1, l2 = make_tle(25544, _EPOCH)
        assert validate_tle_pair(
            with_checksum("1 E8493" + l1[7:]), with_checksum("2 E8493" + l2[7:])
        ) == 148493

    def test_disagreeing_identifiers_raise(self):
        l1, _ = make_tle(25544, _EPOCH)
        _, l2 = make_tle(38833, _EPOCH)
        with pytest.raises(ValueError, match="identifiers disagree"):
            validate_tle_pair(l1, l2)

    @pytest.mark.parametrize("line", ["", "not a tle", 12345, None])
    def test_malformed_lines_raise(self, line):
        l1, l2 = make_tle(25544, _EPOCH)
        with pytest.raises(ValueError):
            validate_tle_pair(line, l2)

    @pytest.mark.parametrize(
        "start,stop",
        [(8, 16), (17, 25), (26, 33), (34, 42), (43, 51), (52, 63)],
        ids=["inclination", "raan", "eccentricity", "arg_pericenter",
             "mean_anomaly", "mean_motion"],
    )
    def test_validation_covers_every_field_the_parser_reads(self, start, stop):
        # Whatever parse_tle_elements consumes, validate_tle_pair must reject
        # when malformed — the two can never drift apart.
        l1, l2 = make_tle(25544, _EPOCH)
        bad2 = l2[:start] + "A" * (stop - start) + l2[stop:]
        with pytest.raises(ValueError):
            validate_tle_pair(l1, bad2)
        with pytest.raises(ValueError):
            parse_tle_elements(l1, bad2)

    def test_malformed_bstar_rejected(self):
        l1, l2 = make_tle(25544, _EPOCH)
        with pytest.raises(ValueError):
            validate_tle_pair(with_checksum(l1[:53] + "ABCDEFGH" + l1[61:]), l2)

    def test_out_of_range_epoch_day_rejected(self):
        l1, l2 = make_tle(25544, _EPOCH)
        with pytest.raises(ValueError, match="epoch day out of range"):
            validate_tle_pair(with_checksum(l1[:20] + "999.00000000" + l1[32:]), l2)

    @pytest.mark.parametrize(
        "start,stop,value",
        [(8, 16, "nan"), (17, 25, "inf"), (52, 63, "inf")],
        ids=["inclination_nan", "raan_inf", "mean_motion_inf"],
    )
    def test_non_finite_elements_are_rejected(self, start, stop, value):
        # Checksum-corrected, so this proves the *element* validation rejects it
        # rather than the line integrity check getting there first.
        l1, l2 = make_tle(25544, _EPOCH)
        bad2 = with_checksum(l2[:start] + value.rjust(stop - start) + l2[stop:])
        with pytest.raises(ValueError, match="non-finite"):
            validate_tle_pair(l1, bad2)

    @pytest.mark.parametrize("mean_motion", [" 0.00000000", "-1.00000000"])
    def test_non_positive_mean_motion_is_rejected(self, mean_motion):
        l1, l2 = make_tle(25544, _EPOCH)
        bad2 = with_checksum(l2[:52] + mean_motion + l2[63:])
        with pytest.raises(ValueError, match="must be positive"):
            validate_tle_pair(l1, bad2)


# ---------------------------------------------------------------------------
# Line width and checksum
# ---------------------------------------------------------------------------

class TestLineIntegrity:
    """The checksum is what makes single-character corruption detectable.

    Without it a flipped digit inside a fixed-width numeric field parses cleanly,
    stays in range, and silently shifts the modelled trajectory.
    """

    def _pair(self):
        return make_tle(25544, _EPOCH)

    def test_valid_pair_is_accepted(self):
        l1, l2 = self._pair()
        assert validate_tle_pair(l1, l2) == 25544

    def test_trailing_whitespace_is_tolerated(self):
        # Files routinely carry it; it is not part of the fixed-width columns.
        l1, l2 = self._pair()
        assert validate_tle_pair(l1 + "  \n", l2 + "\r\n") == 25544

    @pytest.mark.parametrize("line", [1, 2])
    def test_short_line_is_rejected(self, line):
        pair = list(self._pair())
        pair[line - 1] = pair[line - 1][:-1]
        with pytest.raises(ValueError, match="69 characters"):
            validate_tle_pair(*pair)

    @pytest.mark.parametrize("line", [1, 2])
    def test_long_line_is_rejected(self, line):
        pair = list(self._pair())
        pair[line - 1] = pair[line - 1] + "0"
        with pytest.raises(ValueError, match="69 characters"):
            validate_tle_pair(*pair)

    @pytest.mark.parametrize("line", [1, 2])
    def test_wrong_checksum_is_rejected(self, line):
        pair = list(self._pair())
        bad = str((int(pair[line - 1][68]) + 1) % 10)
        pair[line - 1] = pair[line - 1][:68] + bad
        with pytest.raises(ValueError, match="checksum mismatch"):
            validate_tle_pair(*pair)

    @pytest.mark.parametrize("line", [1, 2])
    def test_non_digit_checksum_column_is_rejected(self, line):
        pair = list(self._pair())
        pair[line - 1] = pair[line - 1][:68] + "X"
        with pytest.raises(ValueError, match="checksum column"):
            validate_tle_pair(*pair)

    def test_single_character_corruption_in_a_numeric_field_is_caught(self):
        # The motivating case: an inclination digit flipped in transit. The value
        # still parses and stays in range, so only the checksum can catch it.
        l1, l2 = self._pair()
        corrupted = l2[:9] + ("6" if l2[9] != "6" else "7") + l2[10:]
        assert parse_tle_elements(l1, corrupted)["INCLINATION"] != \
            parse_tle_elements(l1, l2)["INCLINATION"]      # it would have changed the orbit
        with pytest.raises(ValueError, match="checksum mismatch"):
            validate_tle_pair(l1, corrupted)

    def test_checksum_counts_minus_signs_as_one(self):
        # The rule that distinguishes a TLE checksum from a plain digit sum.
        assert tle_checksum("-" * 68) == 68 % 10
        assert tle_checksum("1" * 68) == 68 % 10
        assert tle_checksum(" " * 68) == 0

    def test_real_bundled_records_satisfy_both_checks(self):
        # Guards the enforcement against the data we actually ship: every bundled
        # Space-Track record must remain acceptable.
        from importlib.resources import files as _files
        from tabascal.satchecker import read_legacy_tle_records

        df = read_legacy_tle_records(str(_files("tabascal").joinpath("data/tles")))
        assert len(df) > 200
        for _, row in df.iterrows():
            validate_tle_pair(row["TLE_LINE1"], row["TLE_LINE2"])
