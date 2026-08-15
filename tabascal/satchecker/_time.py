"""Small UTC Julian-date conversions used internally by SatChecker.

Kept here instead of importing :mod:`tabascal.time` so this package can be
lifted into a standalone distribution without pulling in TABASCAL, NumPy, or
Skyfield.
"""

from datetime import datetime, timedelta, timezone


DAY_SECONDS = 86_400.0
UNIX_EPOCH = datetime(1970, 1, 1)
UNIX_EPOCH_JD = 2440587.5


def datetime_to_jd(value: datetime) -> float:
    """Convert a naive/UTC datetime to a UTC Julian Date."""
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return UNIX_EPOCH_JD + (value - UNIX_EPOCH).total_seconds() / DAY_SECONDS


def jd_to_datetime(value: float) -> datetime:
    """Convert a UTC Julian Date to a naive UTC datetime."""
    return UNIX_EPOCH + timedelta(days=float(value) - UNIX_EPOCH_JD)
