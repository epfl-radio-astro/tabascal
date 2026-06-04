from datetime import datetime, timedelta

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


def gast_deg(times_jd):
    """Greenwich Apparent Sidereal Time, in degrees, for UTC Julian Dates.

    Measurement Set times follow the UTC convention, so the input Julian Dates
    are interpreted as UTC (not UT1) — skyfield then applies the UT1-UTC offset
    internally. The *apparent* (not mean) sidereal angle is returned, i.e. it
    includes the equation of the equinoxes, so this is GAST and not GMST.

    The Julian Date is split into whole and fractional parts before being
    handed to skyfield to preserve full f64 precision.

    Parameters
    ----------
    times_jd : array_like
        Observation times as UTC Julian Dates.

    Returns
    -------
    np.ndarray
        GAST in degrees.
    """

    times_jd = np.asarray(times_jd, dtype=float)
    jd_whole = np.floor(times_jd)
    jd_frac = times_jd - jd_whole

    ts = load.timescale()
    t_sf = ts._utc_jd(jd_whole, jd_frac)

    return np.asarray(t_sf.gast) * 15.0  # GAST hours → degrees
