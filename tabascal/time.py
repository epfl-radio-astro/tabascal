DAY_SECS = 24 * 3600.0  # Seconds in a day

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


def gmsa_from_jd(jd: float) -> float:
    """Get the Greenwich Mean Sidereal Angle in degrees from the Julian Day (UT1).
    Calculated using https://aa.usno.navy.mil/faq/GAST

    Parameters
    ----------
    jd : float
        Julian Day (UT1).

    Returns
    -------
    float
        Greenwich Mean Sidereal Angle in degrees.
    """

    gmst_hours = 18.697375 + 24.065709824279 * (jd - 2451545.0)

    gmsa = gmst_hours * 15

    return gmsa