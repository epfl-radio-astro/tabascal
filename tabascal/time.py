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


#: Time scales that can be named in a Measurement Set's ``TIME`` column
#: ``MEASINFO`` record, mapped to the :class:`skyfield.timelib.Timescale`
#: constructor that interprets a Julian Date on that scale.
#:
#: casacore names several of these more than once, and the name it writes is not
#: always the one an outsider would reach for: its canonical spelling of
#: Terrestrial Time is ``TDT``, with ``TT`` and ``ET`` as synonyms, and TAI is
#: also spelled ``IAT``. All spellings are accepted, since the point is to
#: forward whatever the MS declares.
TIME_SCALES = {
    "utc": "_utc_jd",
    "tai": "tai_jd",
    "iat": "tai_jd",
    "tdt": "tt_jd",
    "tt": "tt_jd",
    "et": "tt_jd",
    "tdb": "tdb_jd",
    "ut1": "ut1_jd",
    "ut": "ut1_jd",
}

#: Epoch references casacore can name that tabascal deliberately does not accept:
#: the sidereal angles, which are not a scale an observation timestamp is on, and
#: the relativistic scales, for which skyfield offers no constructor. Named so the
#: error can say "not supported" rather than implying a typo.
_UNSUPPORTED_SCALES = {
    "last": "local apparent sidereal time",
    "lmst": "local mean sidereal time",
    "gmst1": "Greenwich mean sidereal time",
    "gmst": "Greenwich mean sidereal time",
    "gast": "Greenwich apparent sidereal time",
    "ut2": "UT2",
    "tcg": "geocentric coordinate time",
    "tcb": "barycentric coordinate time",
}

#: Scales whose skyfield constructor takes only a single Julian Date, so the
#: whole/fraction split cannot be carried through to it.
_UNSPLIT_SCALES = frozenset({"ut1", "ut"})


def skyfield_time(times_jd, scale: str = "utc"):
    """Julian Dates on a named time scale → :class:`skyfield.timelib.Time`.

    The single entry point for turning observation times into skyfield times, so
    the decisions below are made once rather than at each call site.

    **The scale is not cosmetic.** A Julian Date is a number until a scale says
    what it counts. Reading a UTC epoch as UT1 shifts it by DUT1 (up to ~0.9 s),
    dragging a satellite along its track by the distance it covers in that time;
    reading it as TAI shifts it by the accumulated leap seconds, currently 37 s.
    Neither produces an error — only a wrong position.

    ``scale`` defaults to ``"utc"`` because that is what a Measurement Set's
    ``TIME`` column almost always declares (``MEASINFO Ref: UTC``). It is a
    default, not an assumption: an MS may declare ``TAI`` or another scale, and
    callers reading one should pass what it says rather than relying on this.

    The Julian Date is split into whole and fractional parts before being handed
    to skyfield, to preserve full f64 precision: a JD's ~2.5e6 day magnitude
    leaves f64 only ~5e-10 days of resolution on the value as a whole. ``ut1`` is
    the exception — skyfield's ``ut1_jd`` takes no fraction argument, so that one
    scale is passed the recombined Julian Date and keeps only ~5e-10 days
    (~40 us) of resolution.

    For ``utc`` this uses skyfield's private ``_utc_jd``, which is why
    ``pyproject.toml`` pins ``skyfield>=1.49,<2``. Keeping it to this one call
    site means the pin protects a single line.

    Parameters
    ----------
    times_jd : array_like
        Observation times as Julian Dates on ``scale``.
    scale : str, optional
        Time scale the Julian Dates are on, as named in an MS ``MEASINFO``
        record. One of :data:`TIME_SCALES`; case-insensitive. Defaults to
        ``"utc"``.

    Returns
    -------
    skyfield.timelib.Time
        The same times, read on ``scale``.

    Raises
    ------
    ValueError
        If ``scale`` is not one tabascal can interpret.
    """

    key = str(scale).strip().lower()
    if key not in TIME_SCALES:
        if key in _UNSUPPORTED_SCALES:
            raise ValueError(
                f"Time scale {scale!r} ({_UNSUPPORTED_SCALES[key]}) is a valid "
                "Measurement Set epoch reference, but tabascal cannot interpret "
                "observation times on it. Supported: "
                f"{sorted(TIME_SCALES)}."
            )
        raise ValueError(
            f"Unsupported time scale {scale!r}. Supported: {sorted(TIME_SCALES)}."
        )

    times_jd = np.asarray(times_jd, dtype=float)
    jd_whole = np.floor(times_jd)
    jd_frac = times_jd - jd_whole

    constructor = getattr(timescale(), TIME_SCALES[key])
    if key in _UNSPLIT_SCALES:
        return constructor(times_jd)

    return constructor(jd_whole, jd_frac)


def gast_deg(times_jd, scale: str = "utc"):
    """Greenwich Apparent Sidereal Time, in degrees, for Julian Dates.

    The *apparent* (not mean) sidereal angle is returned, i.e. it includes the
    equation of the equinoxes, so this is GAST and not GMST.

    Parameters
    ----------
    times_jd : array_like
        Observation times as Julian Dates on ``scale``.
    scale : str, optional
        Time scale the Julian Dates are on; see :func:`skyfield_time`.

    Returns
    -------
    np.ndarray
        GAST in degrees.
    """

    gast = skyfield_time(times_jd, scale).gast

    return np.asarray(gast) * 15.0  # GAST hours → degrees
