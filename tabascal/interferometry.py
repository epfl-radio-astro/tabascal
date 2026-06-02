from tabascal.time import mjd_to_jd, gast_deg

from jax import jit, Array
import jax.numpy as jnp
from functools import partial
import numpy as np
from numpy.typing import NDArray


T_s = 86164.0905  # Sidereal day in seconds
Omega_e = 2 * jnp.pi / T_s  # Earth rotation rate in rad/s
C = 299792458.0  # Speed of light in m/s

def get_rfi_phase_numpy(
    rfi_xyz: NDArray, ants_uvw: NDArray, ants_xyz: NDArray, freqs: NDArray
) -> NDArray:
    """Calculate phase at each antenna for each RFI source

    Parameters
    ----------
    rfi_xyz: Array (n_src, n_time, 3)
        Positions of the RFI sources over time in the ECI frame in metres.
    ants_uvw: Array (n_ant, n_time, 3)
        UVW coordinates of the antennas in metres. Only the w-coordinate is used as this is the phase delay for a fringe-stopping interferometer.
    ants_xyz: Array (n_ant, n_time, 3)
        Positions of the antennas over time in the ECI frame in metres.
    freqs: Array (n_freq,)
        Observation frequencies in Hz.

    Returns
    -------
    phase: Array (n_src, n_ant, n_freq, n_time)
        Phase at each antenna for each source over time.
    """
    c = 299792458.0
    lamda = c / freqs[None, None, :, None]

    distances = np.linalg.norm(
        ants_xyz[None, :, None, :, :] - rfi_xyz[:, None, None, :, :], axis=-1
    )
    fringe_dist = ((distances + ants_uvw[None, :, None, :, -1]) / lamda) % 1

    phases = -2.0 * np.pi * fringe_dist

    return phases


def itrf_to_uvw_numpy(itrf: NDArray, h0: NDArray, dec: float) -> NDArray:
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

    itrf = np.atleast_2d(itrf)
    itrf = itrf - itrf[0, None, :]

    h0 = np.deg2rad(np.atleast_1d(h0))
    dec = np.deg2rad(np.asarray(dec))  # type: ignore
    ones = np.ones_like(h0)

    R = np.array(
        [
            [np.sin(h0), np.cos(h0), np.zeros_like(h0)],
            [
                -np.sin(dec) * np.cos(h0),
                np.sin(dec) * np.sin(h0),
                np.cos(dec) * ones,
            ],
            [
                np.cos(dec) * np.cos(h0),
                -np.cos(dec) * np.sin(h0),
                np.sin(dec) * ones,
            ],
        ]
    )

    uvw = np.einsum("ijt,aj->tai", R, itrf)

    return uvw


def Rotz_numpy(theta: float) -> NDArray:
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
    theta = np.asarray(theta).flatten()[0]
    c = np.cos(np.deg2rad(theta))
    s = np.sin(np.deg2rad(theta))
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    return Rz


def xyz_to_itrf_numpy(xyz: NDArray, gsa: NDArray) -> NDArray:
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

    xyz = np.atleast_2d(xyz)
    gsa = np.atleast_1d(gsa)
    itrf = np.array([Rotz_numpy(-g) @ x for x, g in zip(xyz, gsa)])

    return itrf


def calculate_fringe_frequency_numpy(
    times_mjd: NDArray,
    freq: float,
    rfi_xyz: NDArray,
    ants_itrf: NDArray,
    ants_u: NDArray,
    dec: float,
) -> NDArray:
    """Calculate the fringe frequency of an RFI source.

    Parameters
    ----------
    times_mjd : NDArray (n_time,)
        Times are which the RFI and antenna positions are given in Modified Julian Date.
    freq : float
        Observational frequency in Hz.
    rfi_xyz : NDArray (n_time, 3)
        Position of the RFI source in the ECI frame in metres.
    ants_itrf : NDArray (n_ant, 3)
        Antenna positions in the ITRF (ECEF) frame in metres.
    ants_u : NDArray (n_time, n_ant)
        U component of the antennas in UVW frame in metres.
    dec : float
        Phase centre declination in degrees.

    Returns
    -------
    Array (n_time, n_bl)
        Fringe frequencies on each baseline.
    """

    lam = C / freq
    gsa = gast_deg(mjd_to_jd(times_mjd))  # GAST in degrees (UTC convention)
    times = (times_mjd - times_mjd[0]) * 24 * 3600

    r_ecef = xyz_to_itrf_numpy(rfi_xyz, gsa)  # type: ignore
    s_ecef = r_ecef - np.mean(ants_itrf, axis=0)
    s_hat_ecef = s_ecef / np.linalg.norm(s_ecef, axis=-1, keepdims=True)
    s_hat_dot = np.gradient(s_hat_ecef, np.diff(times[:2])[0], axis=0)

    a1, a2 = np.triu_indices(len(ants_itrf), 1)
    bl_ecef = ants_itrf[a1] - ants_itrf[a2]
    bl_u = ants_u[:, a1] - ants_u[:, a2]

    fringe_move = np.einsum("bi,ti->tb", bl_ecef, s_hat_dot) / lam
    fringe_stat = -bl_u * Omega_e * np.cos(np.deg2rad(dec)) / lam
    fringe_freq = fringe_move - fringe_stat

    return fringe_freq


def get_rfi_phase(
    rfi_xyz: Array, ants_uvw: Array, ants_xyz: Array, freqs: Array
) -> Array:
    """Calculate phase at each antenna for each RFI source

    Parameters
    ----------
    rfi_xyz: Array (n_src, n_time, 3)
        Positions of the RFI sources over time in the ECI frame in metres.
    ants_uvw: Array (n_ant, n_time, 3)
        UVW coordinates of the antennas in metres. Only the w-coordinate is used as this is the phase delay for a fringe-stopping interferometer.
    ants_xyz: Array (n_ant, n_time, 3)
        Positions of the antennas over time in the ECI frame in metres.
    freqs: Array (n_freq,)
        Observation frequencies in Hz.

    Returns
    -------
    phase: Array (n_src, n_ant, n_freq, n_time)
        Phase at each antenna for each source over time.
    """
    c = 299792458.0
    lamda = c / freqs[None, None, :, None]

    distances = jnp.linalg.norm(
        ants_xyz[None, :, None, :, :] - rfi_xyz[:, None, None, :, :], axis=-1
    )
    fringe_dist = ((distances + ants_uvw[None, :, None, :, -1]) / lamda) % 1

    phases = -2.0 * jnp.pi * fringe_dist

    return phases


def calculate_rfi_vis_fine(
    rfi_A: Array, rfi_phase: Array, a1: Array, a2: Array
) -> Array:
    """Calculates the visibility across baselines from the complex antenna signals and geometric phase delays at each antenna.

    Parameters
    ----------
    rfi_A : Array (n_rfi, n_ant, ...)
        The complex-valued signal at each antennna.
    rfi_phase : Array (n_rfi, n_ant, ...)
        The geometric phase delay at each antenna.
    a1 : Array (n_bl,)
        The antenna index for antenna 1 in a baseline.
    a2 : Array (n_bl,)
        The antenna index for antenna 2 in a baseline.

    Returns
    -------
    Array (n_bl, ...)
        The visibilities on each baseline.
    """

    # rfi_A is shape (n_rfi, n_ant, ...)
    # rfi_phase is shape (n_rfi, n_ant, ...)
    # a1 and a2 are shape (n_bl,)
    # rfi_vis_fine is shape (n_bl, ...)

    # Workaround for bug in jax>=0.5.3
    rfi_A_ = jnp.swapaxes(rfi_A, 0, 1)
    rfi_phase_ = jnp.swapaxes(rfi_phase, 0, 1)

    vis_rfi_fine = jnp.sum(
        rfi_A_[a1]
        * jnp.conjugate(rfi_A_[a2])
        * jnp.exp(1.0j * (rfi_phase_[a1] - rfi_phase_[a2])),
        axis=1,
    )

    return vis_rfi_fine


@partial(jit, static_argnames=("freq_stride", "time_stride"))
def calculate_rfi_vis_variable(
    rfi_A: Array,
    rfi_phase: Array,
    a1: Array,
    a2: Array,
    freq_stride: int,
    time_stride: int,
) -> Array:

    # rfi_A is shape (n_ant, n_freq, n_int_freq, n_time, n_int_time)
    # rfi_phase is shape (n_ant, n_freq, n_int_freq, n_time, n_int_time)
    # a1 and a2 are shape (n_bl_grp,)
    # vis_rfi is shape (n_bl_grp, n_freq, n_time)

    idx = [
        slice(None),  # all antennas
        slice(None),  # all rfi sources
        slice(None),  # all frequency channels
        slice(freq_stride // 2, None, freq_stride),  # limited frequency samples
        slice(None),  # all time steps
        slice(time_stride // 2, None, time_stride),  # limited time samples
    ]

    rfi_A = rfi_A[tuple(idx)]
    rfi_phase = rfi_phase[tuple(idx)]

    vis_rfi = jnp.sum(
        jnp.mean(
            rfi_A[a1]
            * jnp.conjugate(rfi_A[a2])
            * jnp.exp(1.0j * (rfi_phase[a1] - rfi_phase[a2])),
            axis=(3, 5),
        ),
        axis=1,
    )

    return vis_rfi


def apply_gains(gains: Array, vis: Array, a1: Array, a2: Array) -> Array:

    vis_obs = gains[a1] * vis * jnp.conjugate(gains)[a2]

    return vis_obs


#########################################################################


def get_divisors(n: int) -> NDArray:
    """Get all divisors of n in ascending order.

    Parameters
    ----------
    n : int
        The value to get the divisors of.

    Returns
    -------
    NDArray
        The divisors of n.
    """

    divisors = []

    for i in range(1, int(np.sqrt(n)) + 1):
        if n % i == 0:
            divisors += [i, n // i]

    return np.unique(divisors)


def get_sampling_bins(min_sampling: int, min_bins: int, max_bins: int) -> NDArray:
    """Get a set of set of divisible samplings where the largest sampling is greater than min_sampling.

    Parameters
    ----------
    min_sampling : int
        The minimum sampling required at the top end of the range.
    min_bins : int
        The minimum number of sampling bins.
    max_bins : int
        The maximum number of sampling bins. If more divisble sampling than desired are found, then the largest are returned.

    Returns
    -------
    NDArray
        The set of sampling bins that are all divisors of the largest sampling.
    """

    assert min_bins < max_bins, "min_bins must be smaller than max_bins"

    i = 0
    divisors = get_divisors(min_sampling + i)[-max_bins:]

    while len(divisors) < min_bins:
        i += 1
        divisors = get_divisors(min_sampling + i)[-max_bins:]

    return divisors


def round_up_to_nearest(original: NDArray, roundings: NDArray) -> NDArray:
    """Round up values to the nearest values in the roundings array.

    Parameters
    ----------
    original : NDArray
        The values to round up to the nearest.
    roundings : NDArray
        The array of values to roound up to.

    Returns
    -------
    NDArray
        The rounded values.
    """

    roundings = np.unique(roundings)

    indices = np.searchsorted(roundings, original, side="left")

    rounded = roundings[indices]

    return rounded


def get_strides_and_idxs(
    samplings: NDArray, min_bins: int, max_bins: int
) -> tuple[list, list[int], int]:
    """Calculate the binned indices, strides, and maximum sampling from an array of random sampling rates.

    Parameters
    ----------
    samplings : NDArray
        The sampling rates.
    min_bins : int
        The minimum number of sampling bins.
    max_bins : int
        The maximum number of sampling bins.

    Returns
    -------
    tuple[list, list[int], int]
        The indices from the samplings array that fall into each stride bin, the binned strides, and the maximum sampling rate which is divisible by all strides.
    """

    divisors = get_sampling_bins(np.max(samplings), min_bins, max_bins)

    max_sampling = max(divisors)

    rounded_samplings = round_up_to_nearest(samplings, divisors)

    strides = [max_sampling // i for i in rounded_samplings]
    u_strides = [int(x) for x in np.unique(strides)]
    idxs = [np.where(np.array(strides) == i)[0] for i in u_strides]

    return idxs, u_strides, max_sampling
