from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
from skyfield.api import load

DAY_SECS = 24 * 3600.0  # Seconds in a day

_UNIX_EPOCH = datetime(1970, 1, 1)  # naive, UTC
_UNIX_EPOCH_JD = 2440587.5          # Julian Date of 1970-01-01T00:00:00 UTC

def secs_to_days(seconds):

    return seconds / DAY_SECS


def days_to_secs(days):

    return days * DAY_SECS


def jd_to_mjd(jd):

    mjd = jd - 2400000.5

    return mjd


def mjd_to_jd(mjd):

    jd = mjd + 2400000.5

    return jd


def jd_to_datetime(jd):
    """UTC Julian Date → naive (UTC) :class:`datetime.datetime`.

    Civil-time conversion treating UTC as a uniform day count (no leap-second
    handling), which is all that is needed for TLE epoch dates and timestamps.
    """

    return _UNIX_EPOCH + timedelta(days=float(jd) - _UNIX_EPOCH_JD)


def datetime_to_jd(dt):
    """Naive (UTC) :class:`datetime.datetime` → UTC Julian Date.

    Inverse of :func:`jd_to_datetime`. A timezone-aware datetime is accepted and
    treated as UTC.
    """

    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)

    return _UNIX_EPOCH_JD + (dt - _UNIX_EPOCH).total_seconds() / DAY_SECS


@lru_cache(maxsize=1)
def timescale():
    """The skyfield timescale, built once and reused."""

    return load.timescale()


def skyfield_time(times_jd):
    """UTC Julian Dates → :class:`skyfield.timelib.Time`.

    The single entry point for turning observation times into skyfield times, so
    the two decisions below are made once rather than at each call site.

    Measurement Set times follow the UTC convention (the ``TIME`` column carries
    ``MEASINFO Ref: UTC``), so the Julian Dates are interpreted as UTC — skyfield
    then applies the UT1-UTC offset internally. Reading them as UT1 instead (i.e.
    ``ts.ut1_jd``) shifts every epoch by DUT1, dragging satellite positions along
    their track by up to ~0.9 s of motion.

    The Julian Date is split into whole and fractional parts before being handed
    to skyfield to preserve full f64 precision: a JD's ~2.5e6 day magnitude
    leaves f64 only ~5e-10 days of resolution on the value as a whole.

    Uses skyfield's private ``_utc_jd``, which is why ``pyproject.toml`` pins
    ``skyfield>=1.49,<2``. Keeping it to this one call site means the pin
    protects a single line.

    Parameters
    ----------
    times_jd : array_like
        Observation times as UTC Julian Dates.

    Returns
    -------
    skyfield.timelib.Time
        The same times, on the UTC scale.
    """

    times_jd = np.asarray(times_jd, dtype=float)
    jd_whole = np.floor(times_jd)
    jd_frac = times_jd - jd_whole

    return timescale()._utc_jd(jd_whole, jd_frac)


def gast_deg(times_jd):
    """Greenwich Apparent Sidereal Time, in degrees, for UTC Julian Dates.

    The *apparent* (not mean) sidereal angle is returned, i.e. it includes the
    equation of the equinoxes, so this is GAST and not GMST.

    Parameters
    ----------
    times_jd : array_like
        Observation times as UTC Julian Dates.

    Returns
    -------
    np.ndarray
        GAST in degrees.
    """

    return np.asarray(skyfield_time(times_jd).gast) * 15.0  # GAST hours → degrees
