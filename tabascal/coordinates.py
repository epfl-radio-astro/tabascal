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


def xyz_to_itrf(xyz: Array, gsa: Array) -> Array:
    """Transform coordinates from the ECI frame to the ITRF (ECEF) frame that is fixed with the Earth.

    Parameters
    ----------
    xyz : Array (n_time, 3)
        ECI coordinates in metres.
    gsa : Array (n_time,)
        Greenwich sidereal time in degrees.

    Returns
    -------
    Array (n_time, 3)
        ITRF (ECEF) coordinates in metres.
    """

    xyz = jnp.atleast_2d(xyz)
    gsa = jnp.atleast_1d(gsa)
    eci_to_ecef = lambda eci, gsa: Rotz(-gsa) @ eci
    itrf = vmap(eci_to_ecef, in_axes=(0, 0))(xyz, gsa)

    return itrf


def Rotz(theta: float) -> Array:
    """
    Define a rotation matrix about the 'z-axis' by an angle theta, in degrees.

    Parameters
    ----------
    theta: float
        Rotation angle in degrees.

    Returns
    -------
    R: ndarray (3, 3)
        Rotation matrix.
    """
    theta = jnp.asarray(theta).flatten()[0]
    c = jnp.cos(jnp.deg2rad(theta))
    s = jnp.sin(jnp.deg2rad(theta))
    Rz = jnp.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    return Rz