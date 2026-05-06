import sgp4jax

from jax import vmap, Array
import jax.numpy as jnp

def itrf_to_uvw(itrf: Array, h0: Array, dec: float) -> Array:
    """
    Calculate uvw coordinates from ITRF/ECEF coordinates,
    source hour angle and declination. Use the Greenwich hour
    angle when using true ITRF coordinates such as those produced
    with 'enu_to_itrf' or provided in an MS file. Use local hour angle when using local 'xyz'
    coordinates as defined in most radio interferometry textbooks
    or those produced with 'enu_to_xyz_local'.

    Parameters
    ----------
    ITRF: Array (n_ant, 3)
        Antenna positions in the ITRF frame in units of metres.
    h0: Array (n_time,)
        The hour angle of the target in decimal degrees.
    dec: float
        The declination of the target in decimal degrees.

    Returns
    -------
    uvw: Array (n_time, n_ant, 3)
        The uvw coordinates of the antennas for a given observer
        location, time and target (ra,dec).
    """

    itrf = jnp.atleast_2d(itrf)
    itrf = itrf - itrf[0, None, :]

    h0 = jnp.deg2rad(jnp.atleast_1d(h0))
    dec = jnp.deg2rad(jnp.asarray(dec))  # type: ignore
    ones = jnp.ones_like(h0)

    R = jnp.array(
        [
            [jnp.sin(h0), jnp.cos(h0), jnp.zeros_like(h0)],
            [
                -jnp.sin(dec) * jnp.cos(h0),
                jnp.sin(dec) * jnp.sin(h0),
                jnp.cos(dec) * ones,
            ],
            [
                jnp.cos(dec) * jnp.cos(h0),
                -jnp.cos(dec) * jnp.sin(h0),
                jnp.sin(dec) * ones,
            ],
        ]
    )

    uvw = jnp.einsum("ijt,aj->tai", R, itrf)

    return uvw


def gcrf_to_uvw(gcrf: Array, ra: float, dec: float) -> Array:
    """Apply the GCRF → UVW rotation for a given phase centre.

    Parameters
    ----------
    gcrf : Array (..., 3)
        Positions or baselines in the GCRF frame, any units.
    ra : float
        Phase centre right ascension in degrees.
    dec : float
        Phase centre declination in degrees.

    Returns
    -------
    Array (..., 3)
        UVW coordinates in the same units as *gcrf*.
    """
    ra_r = jnp.deg2rad(jnp.asarray(ra))
    dec_r = jnp.deg2rad(jnp.asarray(dec))
    R = jnp.array([
        [-jnp.sin(ra_r),                          jnp.cos(ra_r),                        0.0           ],
        [-jnp.sin(dec_r) * jnp.cos(ra_r), -jnp.sin(dec_r) * jnp.sin(ra_r), jnp.cos(dec_r)],
        [ jnp.cos(dec_r) * jnp.cos(ra_r),  jnp.cos(dec_r) * jnp.sin(ra_r), jnp.sin(dec_r)],
    ])
    return jnp.asarray(gcrf) @ R.T


def itrf_to_uvw_jd(ants_itrf: Array, times_jd: Array, ra: float, dec: float) -> Array:
    """Calculate UVW coordinates from ITRF antenna positions and Julian dates.

    Converts ITRF positions to GCRF using the full IAU-2006 precession /
    IAU-2000A nutation model (sgp4jax), subtracts the first antenna to form
    baselines, then applies :func:`gcrf_to_uvw`.

    Parameters
    ----------
    ants_itrf : Array (n_ant, 3)
        Antenna positions in the ITRF frame in metres.
    times_jd : Array (n_time,)
        Observation times in Julian Date.
    ra : float
        Phase centre right ascension in degrees.
    dec : float
        Phase centre declination in degrees.

    Returns
    -------
    Array (n_time, n_ant, 3)
        UVW coordinates in metres.
    """
    ants_itrf = jnp.atleast_2d(ants_itrf)
    times_jd = jnp.atleast_1d(jnp.asarray(times_jd))

    jd_whole = jnp.floor(times_jd)
    jd_frac = times_jd - jd_whole

    # ITRF → GCRF for every (antenna, time) pair; shape (n_ant, n_time, 3)
    ants_gcrf = vmap(vmap(sgp4jax.itrf_to_gcrf, (0, None, None), 0), (None, 0, 0), 1)(
        ants_itrf, jd_whole, jd_frac
    )

    # Baselines relative to the first antenna
    ants_gcrf = ants_gcrf - ants_gcrf[0:1]

    # (n_ant, n_time, 3) → (n_time, n_ant, 3)
    return jnp.transpose(gcrf_to_uvw(ants_gcrf, ra, dec), axes=(1, 0, 2))
