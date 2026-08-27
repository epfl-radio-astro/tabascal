from tabascal.time import mjd_to_jd, gast_deg

from jax import jit, lax, Array
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

    numpy twin of the jax :func:`get_rfi_phase` — same formula, kept in numpy/f64
    for the host-side one-shot setup in ``FixedOrbit`` (large magnitudes need f64).
    The two must stay in sync; their equivalence is checked by
    ``tests/components/test_trajectory.py::TestFixedOrbit::test_compute_rfi_phase_consistent_with_get_rfi_phase``.

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


def fov_to_eff_diameter(fov_deg: float, freq: float) -> Array:
    """Effective dish diameter reproducing a given field of view.

    ``fov_deg`` is the full field of view (angular diameter), so the maximum
    source offset from the phase centre is ``fov_deg / 2``. Since
    :func:`max_ast_fringe_rate` uses a beam radius ``rho = 1.22 lam / D``, the
    diameter that gives ``rho = fov_deg / 2`` is ``2 * 1.22 lam / fov``.

    Parameters
    ----------
    fov_deg : float
        Full field of view (diameter) in degrees.
    freq : float
        Observational frequency in Hz. Use the lowest frequency of the band for
        the widest beam.

    Returns
    -------
    Array
        Effective dish diameter in metres.
    """
    return 2.44 * C / (freq * jnp.deg2rad(fov_deg))


def _pole_projections(uvw: Array, d: Array) -> tuple[Array, Array]:
    """Baseline projections the two beam offsets couple to.

    Frequency-independent half of :func:`max_ast_fringe_rate`; see that function
    for the derivation. ``d`` is the declination in *radians*. Both outputs have
    the shape of ``uvw`` with its trailing length-3 axis dropped.
    """
    u, v, w = uvw[..., 0], uvw[..., 1], uvw[..., 2]

    # A: baseline component perpendicular to the celestial pole, which the
    # transverse (l, m) offset couples to.
    A = jnp.sqrt((v * jnp.sin(d) - w * jnp.cos(d)) ** 2 + (u * jnp.sin(d)) ** 2)
    # B: the component the radial (n - 1) offset couples to.
    B = jnp.abs(u * jnp.cos(d))

    return A, B


def _beam_couplings(freq: Array, D: float) -> tuple[Array, Array]:
    """Coefficients of the transverse and radial terms at one frequency.

    Baseline-independent half of :func:`max_ast_fringe_rate`: the rate at the
    beam edge is ``A * g_transverse + B * g_radial`` with ``A, B`` from
    :func:`_pole_projections`.
    """
    lam = C / freq
    # Beam radius: the largest angular offset of a source from the phase centre,
    # taken to the first null of a uniformly illuminated aperture.
    rho = 1.22 * lam / D

    # 1 - cos(rho) is evaluated as the equivalent 2 sin^2(rho / 2): for the small
    # rho of a typical beam, cos(rho) is within rounding of 1, so the subtraction
    # cancels catastrophically (~1e-3 relative error in fp32 at rho = 0.25 deg,
    # against ~1e-7 for the half-angle form). tabascal defaults to fp32, so the
    # stable form matters here.
    return Omega_e * jnp.sin(rho) / lam, Omega_e * 2 * jnp.sin(rho / 2) ** 2 / lam


@jit
def _max_over_time_and_freq(uvw: Array, d: Array, freqs: Array, D: float) -> Array:
    """Max fringe rate per baseline over time and frequency, without a 3D temporary.

    ``uvw`` is ``(n_time, n_bl, 3)``, ``d`` the declination in radians and
    ``freqs`` a ``(n_freq,)`` array; the result is ``(n_bl,)``. The frequency
    axis is consumed by a :func:`~jax.lax.scan` whose carry is the per-baseline
    running maximum, so the largest array ever formed is the ``(n_time, n_bl)``
    slice for one channel, rather than the ``(n_time, n_bl, n_freq)`` block a
    vectorised max would materialise -- 13 GB for a full MeerKAT band.
    """
    A, B = _pole_projections(uvw, d)  # both (n_time, n_bl)

    def step(best: Array, freq: Array) -> tuple[Array, None]:
        g_transverse, g_radial = _beam_couplings(freq, D)
        rate = A * g_transverse + B * g_radial  # (n_time, n_bl)
        return jnp.maximum(best, jnp.max(rate, axis=0)), None

    n_bl = uvw.shape[1]
    best, _ = lax.scan(step, jnp.full((n_bl,), -jnp.inf, A.dtype), freqs)

    return best


def max_ast_fringe_rate(
    uvw: Array, dec: float, freq: float | Array, D: float
) -> Array:
    """Maximum astronomical fringe rate of a source within the primary beam.

    The fringe-stopped visibility phase is phi = (2 pi / lam) b . (s - s0), where
    s - s0 is the offset of the source direction from the phase centre. Because
    the sky and the baseline rotate rigidly with respect to each other at the
    Earth rotation rate Omega, the fringe rate is f = (1 / lam) b . (Omega x
    (s - s0)). In the UVW frame the celestial pole lies along
    n_hat = cos(d) v_hat + sin(d) w_hat, and a source at angular offset r,
    azimuth chi has s - s0 = (l, m, n - 1) with
    (l, m, n) = (sin(r) cos(chi), sin(r) sin(chi), cos(r)). Expanding
    b . (Omega x (s - s0)) and maximising over chi gives

        f_max(r) = (Omega_e / lam) [ A sin(r) + B (1 - cos(r)) ]
        A = sqrt((v sin d - w cos d)^2 + (u sin d)^2)
        B = |u cos d|

    The two terms couple to different baseline projections:

    * Transverse (A) - the offset perpendicular to the line of sight, magnitude
      sin(r), against the baseline component perpendicular to the celestial
      pole. This is the dominant, first-order contribution.
    * Radial (B) - the (n - 1) curvature of the celestial sphere, magnitude
      1 - cos(r), against u cos(d). Second order in r, and zero at the pole
      (d = 90 deg), so it only matters for wide fields away from the pole.

    **What is maximised.** The beam position is *always* maximised over: over the
    azimuth chi (above) and over the offset r within the beam (below). Time and
    frequency are maximised over only when arrays are supplied for them -- a
    single UVW sample or a scalar frequency is used as given.

    **Why the beam maximum is at its edge.** Differentiating the expression above,

        df_max/dr = (Omega_e / lam) [ A cos(r) + B sin(r) ]

    Since A, B >= 0, this is non-negative for 0 <= r <= pi/2, so f_max increases
    out to the beam radius and the maximum over the beam is attained at r = rho.
    Realistic primary beams satisfy rho <= 90 deg, so this covers every practical
    case; note that monotonicity is *not* claimed unconditionally, as cos(r) turns
    negative beyond pi/2.

    The beam radius is taken to the first null, rho = 1.22 lam / D, rather than
    the half-power point: this rate is used as the knee k0 of the astronomical
    power-spectrum prior (the width of its Gaussian roll-off), so it should cover
    the largest offset that still contributes appreciable flux -- underestimating
    it suppresses genuine fast-fringe power in the prior. Use
    :func:`fov_to_eff_diameter` to obtain ``D`` from a configured field of view.

    Parameters
    ----------
    uvw : Array (3,) or (n_time, n_bl, 3)
        Either a single baseline sample, or baselines over an observation. No
        other shape is accepted, so that the axes are never ambiguous.
    dec : float
        Phase centre declination in degrees.
    freq : float or Array (n_freq,)
        Observational frequency in Hz. An array is maximised over.
    D : float
        Dish diameter (or effective diameter for a given field of view) in metres.

    Returns
    -------
    Array
        A scalar for ``(3,)`` input, or one maximum fringe rate per baseline,
        shape ``(n_bl,)``, for ``(n_time, n_bl, 3)`` input. Time and frequency
        are reduced away as they are traversed, so neither the returned array nor
        any temporary is ever ``(n_time, n_bl, n_freq)``.
    """
    uvw = jnp.asarray(uvw)
    freqs = jnp.atleast_1d(jnp.asarray(freq))
    d = jnp.deg2rad(dec)

    # A single sample is the degenerate n_time = n_bl = 1 observation, so it goes
    # through the same reduction rather than a second implementation of it.
    if uvw.ndim == 1 and uvw.shape[0] == 3:
        return _max_over_time_and_freq(uvw[None, None, :], d, freqs, D)[0]

    if uvw.ndim == 3 and uvw.shape[-1] == 3:
        return _max_over_time_and_freq(uvw, d, freqs, D)

    raise ValueError(
        "uvw must have shape (3,) for a single sample or (n_time, n_bl, 3) for "
        f"an observation, got {uvw.shape}"
    )


def get_rfi_phase(
    rfi_xyz: Array, ants_uvw: Array, ants_xyz: Array, freqs: Array
) -> Array:
    """Calculate phase at each antenna for each RFI source

    jax (differentiable) version, used by ``PhaseCalculationRFI`` (double-only).
    Keep in sync with the numpy twin :func:`get_rfi_phase_numpy`.


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


def baseline_gains(gains: Array, a1: Array, a2: Array, ant_axis: int = 0) -> Array:
    """Per-baseline gain ``g_p conj(g_q)`` from per-antenna gains.

    ``ant_axis`` names the antenna axis, so an array carrying a leading sample
    axis can have the product formed per sample, before the samples are reduced:
    ``E[g_p conj(g_q)]`` is not ``E[g_p] conj(E[g_q])`` once the two antennas
    covary.

    Written with plain indexing and ``.conj()`` rather than ``jnp`` calls so that
    the one definition serves both users -- the model jits it on jax arrays, and
    the results writer applies it to dask arrays straight out of a zarr.
    """

    lead = (slice(None),) * ant_axis

    return gains[lead + (a1,)] * gains[lead + (a2,)].conj()


def apply_gains(gains: Array, vis: Array, a1: Array, a2: Array) -> Array:
    """Apply per-antenna gains to per-baseline visibilities.

    The same product as :func:`baseline_gains`, which is the one definition of
    the convention and what the tests hold this to. It is multiplied in the
    order ``g_p * vis * conj(g_q)`` rather than ``(g_p conj(g_q)) * vis``,
    though: floating point is not associative, and with ``complex64`` gains far
    from unity the baseline product can overflow where ``g_p * vis`` first
    does not. This order is also the one every reference result was produced
    with, so the model's output is unchanged to the bit.
    """

    return gains[a1] * vis * gains[a2].conj()


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
    # Values above max(roundings) yield index == len(roundings); clip so they
    # round down to the largest available value instead of raising IndexError.
    indices = np.minimum(indices, len(roundings) - 1)

    rounded = roundings[indices]

    return rounded


def get_strides_and_idxs(
    samplings: NDArray, min_bins: int, max_bins: int, min_divisors: int = 8
) -> tuple[list, list[int], int]:
    """Calculate the binned indices, strides, and maximum sampling from an array of random sampling rates.

    The sampling rates are grouped into stride bins for ``RiemannVisVariable``.
    Each returned stride must divide ``max_sampling`` (which becomes ``n_int_time``,
    the fine-grid size) so that the per-group fine-grid slicing stays uniform.

    The naive approach of keying the bins off ``divisors(max(samplings))`` is
    fragile: when ``max(samplings)`` is prime (e.g. 43) its only divisors are
    ``{1, 43}``, so every baseline rounds onto a single stride and the whole
    observation collapses into one group. To avoid this we (1) bump
    ``max_sampling`` up to a divisor-RICH integer ``>= max(samplings)`` so there
    are enough candidate strides to separate the distribution, and (2) place the
    bin levels at quantiles of the actual sampling distribution (snapped to
    divisors) so each group is non-empty and balanced by baseline count.

    Parameters
    ----------
    samplings : NDArray
        The sampling rates.
    min_bins : int
        The minimum number of sampling bins.
    max_bins : int
        The maximum number of sampling bins.
    min_divisors : int, optional
        Require at least this many divisors of ``max_sampling`` so the candidate
        strides are dense enough to separate the samplings. Higher values give
        more, better-separated groups at the cost of a larger ``max_sampling``
        (hence larger fine grid). Default 8.

    Returns
    -------
    tuple[list, list[int], int]
        The indices from the samplings array that fall into each stride bin,
        the binned strides, and the maximum sampling rate which is divisible by all strides.
    """

    samplings = np.asarray(samplings)
    max_samp = int(np.max(samplings))

    # 1. Fine-grid size >= max(samplings), chosen to be divisor-rich so the
    #    candidate strides (its divisors) are dense enough to separate the
    #    sampling distribution. Bounded overshoot keeps the fine grid in check.
    need = max(min_divisors, min_bins + 1)
    max_sampling = max_samp
    while len(get_divisors(max_sampling)) < need and max_sampling < 2 * max_samp:
        max_sampling += 1
    divisors = get_divisors(max_sampling)

    # 2. Place up to max_bins bin levels at quantiles of the sampling
    #    distribution, snapped up to divisors of max_sampling. The top divisor is
    #    always included so it covers max(samplings).
    n_levels = max(min(max_bins, len(divisors)), min_bins)
    targets = np.quantile(samplings, np.linspace(0.0, 1.0, n_levels))
    levels = np.unique(round_up_to_nearest(targets, divisors))
    levels = np.unique(np.append(levels, max_sampling))

    # 3. stride = max_sampling // rounded level; each stride divides max_sampling.
    rounded_samplings = round_up_to_nearest(samplings, levels)
    strides = max_sampling // rounded_samplings
    u_strides = [int(x) for x in np.unique(strides)]
    idxs = [np.where(strides == i)[0] for i in u_strides]

    return idxs, u_strides, max_sampling
