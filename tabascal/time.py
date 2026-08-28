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


def utc_offset_days(times_jd, scale: str = "utc"):
    """Days to add to a Julian Date on ``scale`` to name the same instant on UTC.

    The offset is what separates the scales -- the leap seconds for TAI, those
    plus 32.184 s for TT, DUT1 for UT1 -- so it is small, at most a minute or so
    of days, and it is computed **per sample**: an observation straddling a leap
    second is offset by 36 s on one side of it and 37 s on the other.

    It is read out of the whole/fraction pair skyfield already holds, rather than
    by differencing two Julian Dates or by reconstructing a UTC date from an
    accessor. Both of those would work at the Julian Date's own ~2.5e6 magnitude,
    where f64 resolves only ~40 us; the whole days cancel exactly and the
    fractions are O(1), so the offset itself comes out exact to picoseconds. The
    Julian Date it is then added to still resolves only to that ~40 us -- see
    :func:`to_utc_jd`.

    Parameters
    ----------
    times_jd : array_like
        Julian Dates as the source declares them, on ``scale``.
    scale : str, optional
        Time scale the Julian Dates are on; see :func:`skyfield_time`.

    Returns
    -------
    np.ndarray
        The offset in days, one per input time. Exactly zero for ``utc``.
    """

    times_jd = np.asarray(times_jd, dtype=float)

    if str(scale).strip().lower() == "utc":
        return np.zeros_like(times_jd)

    declared = skyfield_time(times_jd, scale)
    as_utc = skyfield_time(times_jd, "utc")

    return (declared.whole - as_utc.whole) + (
        declared.tt_fraction - as_utc.tt_fraction
    )


def to_utc_jd(times_jd, scale: str = "utc"):
    """Julian Dates on a named scale as the same instants on UTC.

    tabascal reads times on whatever scale their source declares and works in
    UTC everywhere after that: :func:`skyfield_time` defaults to it, the epoch
    checks compare against it, and ``sgp4jax.itrf_to_gcrf`` has no scale concept
    to be told anything else. Converting once, where the times are read, puts all
    of them on the instant the source actually named without a ``scale``
    argument threaded through any of them.

    ``utc`` returns the input unchanged -- bit-identical, not merely close, since
    no arithmetic is done at all. The other scales have :func:`utc_offset_days`
    added to the day fraction, with the whole day carried across separately.

    That does **not** make the answer picosecond-accurate. The return value is a
    single f64 Julian Date, and near 2.5e6 days those are spaced ~40 us apart:
    that is the floor on any JD, before or after conversion, and no arrangement
    of the arithmetic beats it. What the split buys is that the conversion costs
    nothing *beyond* that floor -- the offset itself is exact to picoseconds,
    being a difference of O(1) fractions, and recombining rounds once. Rebuilding
    the UTC date from a skyfield accessor would spend the floor a second time.

    Parameters
    ----------
    times_jd : array_like
        Julian Dates as the source declares them, on ``scale``.
    scale : str, optional
        Time scale the Julian Dates are on; see :func:`skyfield_time`. Defaults
        to ``"utc"``, which is a no-op.

    Returns
    -------
    np.ndarray
        The same instants, as UTC Julian Dates.

    Raises
    ------
    ValueError
        If ``scale`` is not one tabascal can interpret.
    """

    times_jd = np.asarray(times_jd, dtype=float)

    if str(scale).strip().lower() == "utc":
        return times_jd

    whole = np.floor(times_jd)

    return whole + ((times_jd - whole) + utc_offset_days(times_jd, scale))


def to_utc_mjd(times_mjd, scale: str = "utc"):
    """Modified Julian Dates on a named scale as the same instants on UTC.

    The MJD counterpart of :func:`to_utc_jd`, for the day numbers that leave an
    observation and are compared against another source's -- a light-curve
    estimate's time axis, say. An MJD is a number until a scale says what it
    counts, so a day number written on whatever an MS happened to declare cannot
    be matched against one from anywhere else; UTC is the scale tabascal states
    for those, and this is what puts a declared column on it.

    ``utc`` returns the input unchanged -- bit-identical, not merely close, since
    no arithmetic is done at all. The other scales have :func:`utc_offset_days`
    added.

    The offset is added at MJD magnitude rather than by routing through
    :func:`to_utc_jd`: an MJD near 6e4 is spaced ~1.3 us apart in f64, where a
    Julian Date near 2.5e6 is spaced ~40 us apart, so the round trip out to a JD
    and back would round twice at the coarser magnitude -- on a UTC MS, for a
    conversion that is not even needed. The offset *lookup* still goes through a
    Julian Date, which costs nothing: it selects which leap-second era the time
    falls in, and 40 us only changes that within 40 us of a leap second.

    Parameters
    ----------
    times_mjd : array_like
        Modified Julian Dates as the source declares them, on ``scale``.
    scale : str, optional
        Time scale the day numbers are on; see :func:`skyfield_time`. Defaults
        to ``"utc"``, which is a no-op.

    Returns
    -------
    np.ndarray
        The same instants, as UTC Modified Julian Dates.

    Raises
    ------
    ValueError
        If ``scale`` is not one tabascal can interpret.
    """

    times_mjd = np.asarray(times_mjd, dtype=float)

    if str(scale).strip().lower() == "utc":
        return times_mjd

    return times_mjd + utc_offset_days(mjd_to_jd(times_mjd), scale)


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
