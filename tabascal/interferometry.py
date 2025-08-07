from jax import jit, vmap, Array
import jax.numpy as jnp


@jit
def get_rfi_phase(rfi_xyz: Array, ants_uvw: Array, ants_xyz: Array, freqs: Array):
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


# @jit
def calculate_rfi_vis_fine(rfi_A, rfi_phase, a1, a2):

    # rfi_A is shape (n_rfi, n_ant, n_time_fine)
    # rfi_phase is shape (n_rfi, n_ant, n_time_fine)
    # a1 and a2 are shape (n_bl,)
    # rfi_vis_fine is shape (n_bl, n_time_fine)
    vis_rfi_fine = jnp.sum(
        rfi_A[:, a1]
        * jnp.conjugate(rfi_A[:, a2])
        * jnp.exp(rfi_phase[:, a1] - rfi_phase[:, a2]),
        axis=0,
    )

    return vis_rfi_fine


@jit
def apply_gains(gains, vis, a1, a2):

    vis_obs = gains[a1] * vis * jnp.conjugate(gains)[a2]

    return vis_obs
