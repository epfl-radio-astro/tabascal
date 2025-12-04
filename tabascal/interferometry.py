from jax import jit, vmap, Array
import jax.numpy as jnp
from functools import partial
import numpy as np
from numpy.typing import NDArray


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
        slice(None),  # all frequency channels
        slice(freq_stride // 2, None, freq_stride),  # limited frequency samples
        slice(None),  # all time steps
        slice(time_stride // 2, None, time_stride),  # limited time samples
    ]

    rfi_A = rfi_A[*idx]
    rfi_phase = rfi_phase[*idx]

    vis_rfi = jnp.mean(
        rfi_A[a1]
        * jnp.conjugate(rfi_A[a2])
        * jnp.exp(1.0j * (rfi_phase[a1] - rfi_phase[a2])),
        axis=(2, 4),
    )

    return vis_rfi


def apply_gains(gains, vis, a1, a2):

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
