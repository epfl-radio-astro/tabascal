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
