"""Tests for the shared TLE parser (:mod:`tabascal.satchecker.tle_parse`).

This module is the single parser used by both cache validation and orbital
element extraction, so these tests pin the identifier decoding (including
Alpha-5) and the guarantee that validation and parsing agree.
"""

import pytest

from tabascal.satchecker.tle_parse import (
    decode_norad_id,
    parse_tle_elements,
    validate_tle_pair,
)

from .tle_helpers import jd, make_tle

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
        assert validate_tle_pair("1 E8493" + l1[7:], "2 E8493" + l2[7:]) == 148493

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
            validate_tle_pair(l1[:53] + "ABCDEFGH" + l1[61:], l2)

    def test_out_of_range_epoch_day_rejected(self):
        l1, l2 = make_tle(25544, _EPOCH)
        with pytest.raises(ValueError):
            validate_tle_pair(l1[:20] + "999.00000000" + l1[32:], l2)
