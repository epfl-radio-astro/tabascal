"""Kind dispatch: what a record is, when it is usable, and what it means.

These are the tests that let the policy layer stay format-blind. If
``record_kind`` mis-identifies a record, or the two kinds disagree about the
elements they describe for one satellite, everything above them is resolving
the wrong orbit with no visible error — so the equivalence between kinds is
asserted directly rather than inferred from higher-level behaviour.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tabascal.satchecker.records import (
    KIND_OMM,
    KIND_TLE,
    RecordKindError,
    record_elements,
    record_epoch_jd,
    record_kind,
    validate_record,
)
from tabascal.time import datetime_to_jd, jd_to_datetime

from .tle_helpers import (  # noqa: F401  block_network is an autouse fixture
    block_network,
    both_kinds,
    jd,
    make_omm,
    make_record,
    make_tle,
    make_tle_record,
)


_EPOCH = jd(2026, 8, 1)


# ---------------------------------------------------------------------------
# Kind
# ---------------------------------------------------------------------------

class TestRecordKind:

    @both_kinds
    def test_explicit_kind_is_used(self, kind):
        assert record_kind(make_record(kind, 25544, _EPOCH)) == kind

    def test_tle_is_inferred_from_its_lines(self):
        record = make_tle_record(25544, _EPOCH)
        del record["RECORD_KIND"]
        assert record_kind(record) == KIND_TLE

    def test_omm_is_inferred_from_its_element_columns(self):
        record = make_omm(25544, _EPOCH)
        del record["RECORD_KIND"]
        assert record_kind(record) == KIND_OMM

    def test_spacetrack_gp_json_without_a_kind_field_resolves(self):
        # docs/orbits.md promises a Space-Track `gp`/`gp_history` export can be
        # dropped into extra_orbit_dir unconverted. That JSON carries TLE lines
        # and element columns but no kind field, and the lines are what we
        # validate against, so it must infer as a TLE rather than an OMM.
        record = make_tle_record(25544, _EPOCH)
        del record["RECORD_KIND"]
        record.update(
            {k: v for k, v in make_omm(25544, _EPOCH).items() if k != "RECORD_KIND"}
        )
        assert record_kind(record) == KIND_TLE
        assert validate_record(record) == 25544

    def test_unknown_kind_is_rejected_rather_than_guessed(self):
        record = make_omm(25544, _EPOCH, RECORD_KIND="ephemeris")
        with pytest.raises(RecordKindError, match="unknown RECORD_KIND"):
            record_kind(record)

    def test_a_record_that_is_neither_is_rejected(self):
        with pytest.raises(RecordKindError, match="neither TLE lines nor"):
            record_kind({"NORAD_CAT_ID": 25544, "OBJECT_NAME": "ISS"})

    def test_a_partial_omm_element_set_is_not_an_omm(self):
        record = make_omm(25544, _EPOCH)
        del record["RECORD_KIND"]
        del record["MEAN_MOTION"]
        with pytest.raises(RecordKindError):
            record_kind(record)

    @pytest.mark.parametrize("empty", [None, float("nan"), "", "   "])
    def test_a_present_but_empty_line_does_not_make_it_a_tle(self, empty):
        # A frame that has been concatenated with TLE rows carries the line
        # columns on every row, filled with nulls. Treating those as a TLE would
        # send an OMM record into the line parser.
        record = make_omm(25544, _EPOCH)
        del record["RECORD_KIND"]
        record["TLE_LINE1"] = empty
        record["TLE_LINE2"] = empty
        assert record_kind(record) == KIND_OMM


# ---------------------------------------------------------------------------
# Epoch
# ---------------------------------------------------------------------------

class TestRecordEpoch:

    @both_kinds
    def test_epoch_round_trips(self, kind):
        # The TLE line field is quantised to 1e-8 day (~0.9 ms); OMM is exact.
        record = make_record(kind, 25544, _EPOCH)
        assert record_epoch_jd(record) == pytest.approx(_EPOCH, abs=1e-7)

    def test_the_two_kinds_agree_on_one_satellites_epoch(self):
        tle = record_epoch_jd(make_tle_record(25544, _EPOCH))
        omm = record_epoch_jd(make_omm(25544, _EPOCH))
        assert abs(tle - omm) < 1e-7

    @pytest.mark.parametrize(
        "text",
        [
            "2026-08-01T00:00:00",
            "2026-08-01T00:00:00Z",
            "2026-08-01T00:00:00z",
            "2026-08-01T00:00:00+00:00",
            "2026-08-01T02:00:00+02:00",
        ],
        ids=["naive", "zulu", "lowercase-zulu", "offset", "shifted-offset"],
    )
    def test_iso8601_spellings_all_reach_the_same_instant(self, text):
        record = make_omm(25544, _EPOCH, EPOCH=text)
        assert record_epoch_jd(record) == pytest.approx(jd(2026, 8, 1))

    @pytest.mark.parametrize(
        "bad", ["", "   ", "not a date", "2026-13-01T00:00:00", 12345.6]
    )
    def test_unparseable_omm_epochs_are_rejected(self, bad):
        with pytest.raises(ValueError):
            record_epoch_jd(make_omm(25544, _EPOCH, EPOCH=bad))

    def test_a_missing_omm_epoch_is_rejected(self):
        record = make_omm(25544, _EPOCH)
        del record["EPOCH"]
        with pytest.raises(ValueError, match="missing EPOCH"):
            record_epoch_jd(record)

    def test_a_pre_sputnik_epoch_is_rejected(self):
        record = make_omm(25544, _EPOCH, EPOCH="1901-01-01T00:00:00")
        with pytest.raises(ValueError, match="plausible window"):
            record_epoch_jd(record)

    def test_an_epoch_years_in_the_future_is_rejected(self):
        far = datetime.now(timezone.utc) + timedelta(days=800)
        record = make_omm(25544, _EPOCH, EPOCH=far.replace(tzinfo=None).isoformat())
        with pytest.raises(ValueError, match="plausible window"):
            record_epoch_jd(record)

    def test_a_slightly_future_epoch_is_accepted(self):
        # Element sets are legitimately published a little ahead of their epoch.
        soon = datetime.now(timezone.utc) + timedelta(days=7)
        record = make_omm(25544, _EPOCH, EPOCH=soon.replace(tzinfo=None).isoformat())
        assert record_epoch_jd(record) == pytest.approx(datetime_to_jd(soon), abs=1e-6)

    def test_a_tle_epoch_is_re_derived_and_not_read_from_the_epoch_column(self):
        # The EPOCH column is a provider field; line 1 is the authority. A record
        # whose two disagree must follow the lines.
        record = make_tle_record(25544, _EPOCH, EPOCH="1999-01-01T00:00:00")
        assert record_epoch_jd(record) == pytest.approx(_EPOCH, abs=1e-7)


# ---------------------------------------------------------------------------
# Elements
# ---------------------------------------------------------------------------

class TestRecordElements:

    def test_both_kinds_describe_the_same_orbit(self):
        tle = record_elements(make_tle_record(25544, _EPOCH))
        omm = record_elements(make_omm(25544, _EPOCH))
        assert set(tle) == set(omm)
        for column in tle:
            if column == "EPOCH_JD":
                assert abs(tle[column] - omm[column]) < 1e-7
            else:
                assert tle[column] == pytest.approx(omm[column])

    @both_kinds
    def test_column_order_is_identical_across_kinds(self, kind):
        # _add_parsed_elements builds a DataFrame from these dicts, so a
        # kind-dependent key order would produce kind-dependent column order.
        assert list(record_elements(make_record(kind, 25544, _EPOCH))) == [
            "INCLINATION",
            "RA_OF_ASC_NODE",
            "ECCENTRICITY",
            "ARG_OF_PERICENTER",
            "MEAN_ANOMALY",
            "MEAN_MOTION",
            "BSTAR",
            "SEMIMAJOR_AXIS",
            "EPOCH_JD",
        ]

    def test_a_providers_semimajor_axis_is_not_trusted(self):
        # OMM may carry SEMIMAJOR_AXIS; we recompute it from the mean motion so
        # the two kinds cannot disagree about a satellite they both describe.
        record = make_omm(25544, _EPOCH, SEMIMAJOR_AXIS=1.0)
        assert record_elements(record)["SEMIMAJOR_AXIS"] > 6000.0

    @pytest.mark.parametrize(
        "column,value",
        [
            ("INCLINATION", 181.0),
            ("INCLINATION", float("nan")),
            ("RA_OF_ASC_NODE", 360.0),
            ("ECCENTRICITY", 1.0),
            ("ARG_OF_PERICENTER", -1.0),
            ("MEAN_ANOMALY", float("inf")),
            ("MEAN_MOTION", 0.0),
            ("MEAN_MOTION", -1.0),
            ("BSTAR", float("nan")),
        ],
    )
    def test_out_of_range_omm_elements_are_rejected(self, column, value):
        # OMM has no checksum, so these bounds are the only defence against a
        # corrupted element that would otherwise parse cleanly.
        with pytest.raises(ValueError):
            record_elements(make_omm(25544, _EPOCH, **{column: value}))

    def test_an_unreadable_omm_element_is_rejected(self):
        with pytest.raises(ValueError):
            record_elements(make_omm(25544, _EPOCH, MEAN_MOTION="not a number"))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidateRecord:

    @both_kinds
    def test_a_good_record_returns_its_norad_id(self, kind):
        assert validate_record(make_record(kind, 25544, _EPOCH)) == 25544

    def test_a_tle_id_comes_from_the_lines_not_the_row(self):
        # This is the cross-check OMM cannot offer: a row filed under the wrong
        # satellite is caught because the lines carry their own identifier.
        record = make_tle_record(25544, _EPOCH, NORAD_CAT_ID=38833)
        assert validate_record(record) == 25544

    def test_a_corrupted_tle_checksum_is_rejected(self):
        line1, line2 = make_tle(25544, _EPOCH)
        record = make_tle_record(25544, _EPOCH, TLE_LINE1=line1[:68] + "9")
        with pytest.raises(ValueError, match="checksum"):
            validate_record(record)

    @pytest.mark.parametrize("bad", ["abc", float("nan"), float("inf"), 25544.5, None])
    def test_an_unusable_omm_norad_id_is_rejected(self, bad):
        with pytest.raises(ValueError):
            validate_record(make_omm(25544, _EPOCH, NORAD_CAT_ID=bad))

    def test_an_omm_with_a_bad_epoch_does_not_validate(self):
        # The epoch window is part of validation, not only of epoch derivation:
        # get-nearest-omm answers a pre-handover request with its earliest
        # record instead of reporting that it has none, so a wrong epoch is the
        # failure this format is prone to.
        record = make_omm(25544, _EPOCH, EPOCH="1899-01-01T00:00:00")
        with pytest.raises(ValueError, match="plausible window"):
            validate_record(record)

    def test_an_omm_id_check_is_vacuous_by_construction(self):
        # Documenting the asymmetry as a test: there is only one identifier in
        # an OMM record, so validate_record can only hand back what it was told.
        record = make_omm(25544, _EPOCH, NORAD_CAT_ID=38833)
        assert validate_record(record) == 38833

    def test_a_datetime_epoch_object_is_accepted(self):
        record = make_omm(25544, _EPOCH, EPOCH=jd_to_datetime(_EPOCH))
        assert record_epoch_jd(record) == pytest.approx(_EPOCH, abs=1e-6)
