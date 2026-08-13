"""Kind dispatch over the two orbital-record formats SatChecker serves.

SatChecker's TLE archive is frozen and its OMM archive starts where the TLE one
stops (see :mod:`tabascal.satchecker.client` for the handover date), so tabascal
has to carry both formats at once. Rather than thread a format flag through the
policy layer, records stay plain ``dict``-alikes and everything format-specific
is answered here:

``record_kind``
    Which format a record is in.
``record_epoch_jd``
    Its epoch as a UTC Julian Date — the number every age policy is measured
    against.
``record_elements``
    Its OMM-style orbital elements, in the units the trajectory components
    expect.
``validate_record``
    Whether it is usable at all, returning the NORAD catalogue ID it belongs to.

Everything above the seams — source precedence, the age ceiling, the
strictly-fresher incumbency rule — works off ``epoch_jd`` and an opaque record,
and does not change.

Validation is not symmetric between the two kinds, and pretending otherwise
would be the dangerous move. A TLE carries a modulo-10 checksum over each line
and a second copy of the satellite identifier, so single-character corruption is
detectable and a record can be caught belonging to a different satellite than
the row claims. OMM has neither: one ``NORAD_CAT_ID`` field with nothing to
check it against, and element values that parse cleanly whatever they say.
There is a third loss too — :func:`tabascal.satchecker.tle_parse.tle_epoch_jd`
exists so a provider's own epoch field is never trusted, and for OMM there are
no lines to re-derive it from.

What is done instead, for OMM only: every element goes through the shared
:func:`~tabascal.satchecker.tle_parse.validate_elements` range and finiteness
checks, and the epoch must parse as ISO 8601 and land inside an absolute
plausibility window. That window is not cosmetic. ``get-nearest-omm`` answers a
pre-handover request with its *earliest* record rather than reporting that it
has none, so a wrong epoch is exactly the failure this format is prone to.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from ._time import datetime_to_jd
from .tle_parse import (
    ELEMENT_FIELDS,
    parse_tle_elements,
    semimajor_axis_km,
    tle_epoch_jd,
    validate_elements,
    validate_tle_pair,
)


KIND_TLE = "tle"
KIND_OMM = "omm"

#: Column naming the format explicitly. Optional — see :func:`record_kind`.
KIND_FIELD = "RECORD_KIND"

#: The two lines that make a TLE record.
TLE_LINE_COLUMNS = ("TLE_LINE1", "TLE_LINE2")

#: Element columns an OMM record carries directly, rather than encoding in lines.
OMM_ELEMENT_COLUMNS = tuple(column for column, _ in ELEMENT_FIELDS)

#: Identity and provenance columns both kinds carry.
COMMON_COLUMNS = ("NORAD_CAT_ID", "OBJECT_NAME", "DATA_SOURCE", "DATE_COLLECTED")

#: Sputnik 1. Nothing with an earlier epoch is a real orbital element set.
_EARLIEST_PLAUSIBLE_EPOCH_JD = datetime_to_jd(datetime(1957, 1, 1))

#: How far past *now* an epoch may sit before it is treated as wrong rather than
#: predicted. Propagated element sets are published slightly ahead of time; a
#: year ahead is not that.
_FUTURE_EPOCH_TOLERANCE_DAYS = 366.0


class RecordKindError(ValueError):
    """A record's format could not be determined, or is not one we handle."""


# ---------------------------------------------------------------------------
# Field presence
# ---------------------------------------------------------------------------

def _missing(value) -> bool:
    """True for the several ways a column can be absent in these frames.

    Records arrive from JSON (``None``), from pandas (``nan`` in an object
    column, from a concat that widened the columns), and from user files, so
    "the column exists" is not the same question as "the column has a value".
    """
    if value is None:
        return True
    try:
        if value != value:  # NaN
            return True
    except (TypeError, ValueError):
        return False
    return isinstance(value, str) and not value.strip()


def _has(record, column: str) -> bool:
    try:
        value = record[column]
    except (KeyError, IndexError, TypeError):
        return False
    return not _missing(value)


def _get(record, column: str, context: str):
    if not _has(record, column):
        raise ValueError(f"{context} is missing {column}")
    return record[column]


# ---------------------------------------------------------------------------
# Kind
# ---------------------------------------------------------------------------

def record_kind(record) -> str:
    """Which format *record* is in.

    An explicit ``RECORD_KIND`` wins. Inferring the rest is not a nicety: the
    documented ``extra_orbit_dir`` contract is that a Space-Track ``gp`` /
    ``gp_history`` JSON export can be dropped in unconverted, and that JSON
    carries no kind field. So a record with TLE lines is a TLE, a record with
    the element columns is an OMM, and anything else is rejected rather than
    guessed at.
    """
    if _has(record, KIND_FIELD):
        kind = str(record[KIND_FIELD]).strip().lower()
        if kind not in (KIND_TLE, KIND_OMM):
            raise RecordKindError(f"unknown {KIND_FIELD} {record[KIND_FIELD]!r}")
        return kind
    if all(_has(record, column) for column in TLE_LINE_COLUMNS):
        return KIND_TLE
    if all(_has(record, column) for column in OMM_ELEMENT_COLUMNS):
        return KIND_OMM
    raise RecordKindError(
        "record carries neither TLE lines nor a complete set of OMM element "
        f"columns, and no {KIND_FIELD} field to say which it is"
    )


# ---------------------------------------------------------------------------
# Epoch
# ---------------------------------------------------------------------------

def parse_omm_epoch_jd(value) -> float:
    """UTC Julian Date of an OMM ``EPOCH`` field.

    ISO 8601, with or without a trailing ``Z`` or a UTC offset. The absolute
    plausibility window is checked here rather than at the call site so no path
    can reach the age policy with an epoch that is not a date at all.
    """
    if isinstance(value, datetime):
        stamp = value
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("OMM epoch is empty")
        # Python 3.10's fromisoformat does not accept the military 'Z' suffix.
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            stamp = datetime.fromisoformat(text)
        except ValueError as e:
            raise ValueError(f"OMM epoch is not ISO 8601: {value!r} ({e})") from e
    epoch_jd = datetime_to_jd(stamp)
    if not math.isfinite(epoch_jd):
        raise ValueError(f"OMM epoch is not finite: {value!r}")
    latest = datetime_to_jd(datetime.now(timezone.utc)) + _FUTURE_EPOCH_TOLERANCE_DAYS
    if not _EARLIEST_PLAUSIBLE_EPOCH_JD <= epoch_jd <= latest:
        raise ValueError(
            f"OMM epoch {stamp.isoformat()} is outside the plausible window "
            f"(1957-01-01 to {_FUTURE_EPOCH_TOLERANCE_DAYS:g} days from now)"
        )
    return epoch_jd


def record_epoch_jd(record) -> float:
    """UTC Julian Date of *record*'s epoch, whichever kind it is.

    For a TLE this re-derives the epoch from line 1 rather than reading any
    ``EPOCH`` column that happens to be present — that is deliberate and
    long-standing. For an OMM there is nothing to re-derive it from, so the
    provider's field is parsed and range-checked instead.
    """
    kind = record_kind(record)
    if kind == KIND_TLE:
        return tle_epoch_jd(_get(record, "TLE_LINE1", "TLE record"))
    return parse_omm_epoch_jd(_get(record, "EPOCH", "OMM record"))


# ---------------------------------------------------------------------------
# Elements
# ---------------------------------------------------------------------------

def record_elements(record) -> dict:
    """OMM-style orbital elements for *record*, in the components' units.

    Angles in degrees, mean motion in rev/day, semi-major axis in km. Both kinds
    return the same keys in the same order, and both derive ``SEMIMAJOR_AXIS``
    from the mean motion rather than taking a provider's own value, so the two
    paths cannot disagree on a satellite they both describe.
    """
    kind = record_kind(record)
    if kind == KIND_TLE:
        return parse_tle_elements(
            _get(record, "TLE_LINE1", "TLE record"),
            _get(record, "TLE_LINE2", "TLE record"),
        )
    elements = {
        column: float(_get(record, column, "OMM record"))
        for column in OMM_ELEMENT_COLUMNS
    }
    validate_elements(elements, "OMM")
    elements["SEMIMAJOR_AXIS"] = semimajor_axis_km(elements["MEAN_MOTION"])
    elements["EPOCH_JD"] = parse_omm_epoch_jd(_get(record, "EPOCH", "OMM record"))
    return elements


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def norad_id_of(record, context: str = "record") -> int:
    """The record's own ``NORAD_CAT_ID``, as a positive finite integer.

    A fractional ID would truncate to a *different* satellite and an infinity
    raises inside the numeric casts downstream, so both are rejected here.
    """
    raw = _get(record, "NORAD_CAT_ID", context)
    try:
        value = float(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{context} has a non-numeric NORAD_CAT_ID {raw!r}") from e
    if not math.isfinite(value):
        raise ValueError(f"{context} has a non-finite NORAD_CAT_ID {raw!r}")
    if value != round(value):
        raise ValueError(f"{context} has a non-integer NORAD_CAT_ID {raw!r}")
    return int(round(value))


def validate_record(record) -> int:
    """Fully validate *record*; return the NORAD catalogue ID it belongs to.

    For a TLE the returned ID is decoded from the *lines*, so a caller comparing
    it against the row's own ``NORAD_CAT_ID`` catches a record filed under the
    wrong satellite. For an OMM there is only one identifier in the record, so
    the same comparison is vacuous — the check does not exist to be made. That
    asymmetry is real and is documented rather than papered over; callers make
    the comparison unconditionally because it costs nothing and is meaningful
    for exactly the kind that can support it.

    Raises ``ValueError`` on any problem, which callers treat as "reject this
    record and try another source".
    """
    kind = record_kind(record)
    if kind == KIND_TLE:
        return validate_tle_pair(
            _get(record, "TLE_LINE1", "TLE record"),
            _get(record, "TLE_LINE2", "TLE record"),
        )
    norad_id = norad_id_of(record, "OMM record")
    record_elements(record)  # runs validate_elements and the epoch window
    return norad_id
