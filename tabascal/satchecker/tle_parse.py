"""TLE line parsing shared by cache validation and element extraction.

One parser, two consumers: :mod:`tabascal.tle` derives OMM-style orbital
elements through it, and :mod:`tabascal.satchecker.cache` validates envelopes
through it — so anything the element parser consumes is, by construction,
exactly what validation exercises. Imports only the standard library and
:mod:`tabascal.time`.

Satellite identifiers use the Alpha-5 scheme where needed: catalogue numbers
above 99999 encode the leading digits as a letter (``E8493`` -> 148493; the
letters I and O are excluded to avoid confusion with 1 and 0).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

from tabascal.time import datetime_to_jd


_MU_KM3_S2 = 398600.4418  # Earth gravitational parameter, km^3/s^2

_ALPHA5_EXCLUDED = {"I", "O"}


def decode_norad_id(field: str) -> int:
    """Decode a 5-character TLE satellite field, including Alpha-5 identifiers.

    Plain digits decode directly (``"25544"`` -> 25544). Alpha-5 fields carry a
    leading letter worth 10-33 (A-H, J-N, P-Z; I and O are excluded), so
    ``"E8493"`` -> 148493 and ``"Z9999"`` -> 339999. Raises ``ValueError`` for
    anything else.
    """
    s = str(field).strip()
    if not s:
        raise ValueError("empty satellite identifier field")
    if s.isdigit():
        return int(s)
    head, tail = s[0], s[1:]
    if head.isalpha() and head.isupper() and head not in _ALPHA5_EXCLUDED and tail.isdigit():
        value = ord(head) - 55  # A -> 10
        if head > "I":
            value -= 1
        if head > "O":
            value -= 1
        return value * 10_000 + int(tail)
    raise ValueError(f"invalid satellite identifier field {field!r}")


def _parse_exp_field(field: str) -> float:
    """Parse a TLE exponential field (e.g. '-11606-4' -> -0.11606e-4)."""
    s = field.strip()
    if not s or s in ("+00000-0", "00000-0", "00000+0"):
        return 0.0
    sign = 1.0
    if s[0] in "+-":
        sign = -1.0 if s[0] == "-" else 1.0
        s = s[1:]
    mantissa = s[:-2].replace(" ", "")
    exponent = int(s[-2:])
    if not mantissa:
        return 0.0
    return sign * float("0." + mantissa) * (10.0 ** exponent)


def tle_epoch_jd(line1: str) -> float:
    """UTC Julian Date of a TLE epoch (line 1 columns 19-32)."""
    epoch_year = int(line1[18:20])
    epoch_day = float(line1[20:32])
    if not 0.0 < epoch_day < 367.0:
        raise ValueError(f"TLE epoch day out of range: {epoch_day}")
    year = 2000 + epoch_year if epoch_year < 57 else 1900 + epoch_year
    dt = datetime(year, 1, 1) + timedelta(days=epoch_day - 1.0)
    return datetime_to_jd(dt)


def parse_tle_elements(line1: str, line2: str) -> dict:
    """Derive OMM-style orbital elements from a TLE pair.

    Angles are in degrees, mean motion in rev/day and the semi-major axis in km
    — matching the units Space-Track's OMM reported, so downstream consumers are
    unchanged. ``SEMIMAJOR_AXIS`` is computed from the mean motion via Kepler's
    third law (reproduces the Space-Track OMM value). Raises ``ValueError`` (or
    ``ZeroDivisionError`` for a zero mean motion) on malformed fields.
    """
    inclination = float(line2[8:16])
    raan = float(line2[17:25])
    eccentricity = float("0." + line2[26:33].strip())
    arg_pericenter = float(line2[34:42])
    mean_anomaly = float(line2[43:51])
    mean_motion = float(line2[52:63])  # rev/day
    bstar = _parse_exp_field(line1[53:61])

    parsed_fields = {
        "inclination": inclination,
        "RAAN": raan,
        "eccentricity": eccentricity,
        "argument of pericenter": arg_pericenter,
        "mean anomaly": mean_anomaly,
        "mean motion": mean_motion,
        "BSTAR": bstar,
    }
    non_finite = [name for name, value in parsed_fields.items() if not math.isfinite(value)]
    if non_finite:
        raise ValueError(f"TLE has non-finite fields: {', '.join(non_finite)}")
    if not 0.0 <= inclination <= 180.0:
        raise ValueError(f"TLE inclination out of range: {inclination}")
    for name, value in (
        ("RAAN", raan),
        ("argument of pericenter", arg_pericenter),
        ("mean anomaly", mean_anomaly),
    ):
        if not 0.0 <= value < 360.0:
            raise ValueError(f"TLE {name} out of range: {value}")
    if not 0.0 <= eccentricity < 1.0:
        raise ValueError(f"TLE eccentricity out of range: {eccentricity}")
    if mean_motion <= 0.0:
        raise ValueError(f"TLE mean motion must be positive, got {mean_motion}")

    n_rad_s = mean_motion * 2.0 * math.pi / 86400.0
    semimajor_axis = (_MU_KM3_S2 / n_rad_s ** 2) ** (1.0 / 3.0)

    return {
        "INCLINATION": inclination,
        "RA_OF_ASC_NODE": raan,
        "ECCENTRICITY": eccentricity,
        "ARG_OF_PERICENTER": arg_pericenter,
        "MEAN_ANOMALY": mean_anomaly,
        "MEAN_MOTION": mean_motion,
        "BSTAR": bstar,
        "SEMIMAJOR_AXIS": semimajor_axis,
        "EPOCH_JD": tle_epoch_jd(line1),
    }


def validate_tle_pair(line1, line2) -> int:
    """Fully validate a TLE pair; return its decoded NORAD catalogue ID.

    Runs the *same* parser downstream element extraction uses, so every consumed
    field (epoch, inclination, RAAN, eccentricity, argument of pericenter, mean
    anomaly, mean motion, BSTAR) must parse. Also decodes the satellite
    identifier embedded in both lines (Alpha-5 aware) and requires them to
    agree. Raises ``ValueError`` on any problem.
    """
    if not (isinstance(line1, str) and isinstance(line2, str)):
        raise ValueError("TLE lines must be strings")
    if not line1.startswith("1 ") or not line2.startswith("2 "):
        raise ValueError("TLE lines must start with '1 ' and '2 '")
    id1 = decode_norad_id(line1[2:7])
    id2 = decode_norad_id(line2[2:7])
    if id1 != id2:
        raise ValueError(f"TLE line identifiers disagree: {id1} vs {id2}")
    try:
        parse_tle_elements(line1, line2)
    except ZeroDivisionError as e:
        raise ValueError("TLE mean motion is zero") from e
    return id1
