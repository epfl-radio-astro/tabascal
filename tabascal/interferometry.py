from tabascal.time import mjd_to_jd, gast_deg
from tabascal.special import cerf

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


def fit_nearfield_fringe_freq_poly_numpy(
    times_sec: NDArray,
    freq_ref: float,
    rfi_xyz: NDArray,
    ants_xyz: NDArray,
    ants_w: NDArray,
    a1: NDArray,
    a2: NDArray,
    n_time: int,
    deg_path: int = 4,
) -> NDArray:
    """Per-window polynomial of the per-baseline fringe frequency f(s), from the
    *near-field* geometric path (consistent with :func:`get_rfi_phase`).

    This is the G1-clean source of the analytic fringe parameters (f, fdot, fddot) used
    by the analytic RFI-visibility path. The geometric phase the trajectory component
    produces is the near-field path ``phi = -2*pi*D`` with
    ``D_p(s) = (|ant_p(s) - rfi(s)| + w_p(s)) / lambda`` (cycles). The local fringe
    frequency must be the derivative of *that* path, ``f = -dD/ds`` — NOT the far-field
    plane-wave :func:`calculate_fringe_frequency_numpy` (which uses a single array-centre
    source direction for all baselines and disagrees by ~1% for LEO + km baselines, i.e.
    several radians of phase per sub-window on long baselines). We therefore fit the
    smooth, unwrapped per-antenna path ``D_p(s)`` per window and differentiate
    analytically; differencing the wrapped phase would alias (G1).

    Parameters
    ----------
    times_sec : NDArray (n_time*n_fit,)
        Seconds from the start, on a uniform window-contiguous fit grid with
        ``n_fit = len(times_sec) // n_time`` samples per coarse integration window.
    freq_ref : float
        Reference frequency in Hz (use the band edge maximising the fringe rate). f scales
        linearly with channel frequency, so a single reference suffices.
    rfi_xyz : NDArray (n_rfi, n_time*n_fit, 3)
        Satellite positions in the ECI frame in metres, on the fit grid.
    ants_xyz : NDArray (n_ant, n_time*n_fit, 3)
        Antenna positions in the ECI frame in metres, on the fit grid.
    ants_w : NDArray (n_ant, n_time*n_fit)
        W component (fringe-stop term) of the antennas in the UVW frame in metres.
    a1, a2 : NDArray (n_bl,)
        Antenna indices for each baseline.
    n_time : int
        Number of coarse integration windows.
    deg_path : int
        Polynomial degree for the per-window path fit. 4 makes f cubic and fddot linear in
        s (the leading neglected phase term), matching the K-sizing model.

    Returns
    -------
    NDArray (n_rfi, n_bl, n_time, deg_path)
        Polynomial coefficients of the baseline fringe frequency f(s) = -dD_pq/ds per
        window in s = t - t_c, highest power first (the ``np.polyval`` convention).
    """
    n_rfi = rfi_xyz.shape[0]
    n_ant = ants_xyz.shape[0]
    n_fit_total = len(times_sec)
    n_fit = n_fit_total // n_time
    if n_fit * n_time != n_fit_total:
        raise ValueError("len(times_sec) must be an integer multiple of n_time")
    if n_fit < deg_path + 1:
        raise ValueError(f"need >= {deg_path + 1} fit samples per window, got {n_fit}")

    n_bl = len(a1)
    lam = C / freq_ref
    # Unwrapped near-field fringe distance D_p [cycles] per (rfi, ant).
    diff = ants_xyz[None, :, :, :] - rfi_xyz[:, None, :, :]  # (n_rfi, n_ant, n_tot, 3)
    dist = np.linalg.norm(diff, axis=-1)
    D = (dist + ants_w[None, :, :]) / lam  # (n_rfi, n_ant, n_tot)

    # Fit the per-baseline difference D_pq = D_p - D_q, not the per-antenna D_p: D_p is
    # ~10^10 cycles (full path length), so a degree-4 fit of it loses precision in the
    # high-order coefficients (the noise floor swamps fddot). The difference D_pq is only
    # ~10^3-10^4 cycles (geometric delay), so its fit is well conditioned.
    D_pq = D[:, a1, :] - D[:, a2, :]  # (n_rfi, n_bl, n_tot)

    s_win = times_sec.reshape(n_time, n_fit)
    s_win = s_win - s_win.mean(axis=1, keepdims=True)  # offset from window centre
    Dw = D_pq.reshape(n_rfi, n_bl, n_time, n_fit)

    coeffs_D = np.empty((n_rfi, n_bl, n_time, deg_path + 1))
    for i in range(n_time):
        y = Dw[:, :, i, :].reshape(n_rfi * n_bl, n_fit).T  # (n_fit, n_rfi*n_bl)
        c = np.polyfit(s_win[i], y, deg_path)              # (deg_path+1, n_rfi*n_bl)
        coeffs_D[:, :, i, :] = c.T.reshape(n_rfi, n_bl, deg_path + 1)

    # f_pq(s) = -dD_pq/ds (baseline fringe-frequency polynomial).
    coeffs_f = -coeffs_D[..., :-1] * np.arange(deg_path, 0, -1)  # polyder, highest first
    return coeffs_f


def _polyval_nder(coeffs: NDArray, offsets: NDArray, nder: int) -> NDArray:
    """Evaluate a polynomial and its derivatives up to order ``nder`` at ``offsets``.

    Parameters
    ----------
    coeffs : NDArray (..., deg+1)
        Polynomial coefficients, highest power first (``np.polyval`` convention).
    offsets : NDArray (K,)
        Points at which to evaluate (sub-window centre offsets within a window).
    nder : int
        Highest derivative order to return.

    Returns
    -------
    list of NDArray (..., K)
        ``[p(offsets), p'(offsets), ..., p^{(nder)}(offsets)]``.
    """
    out = []
    c = coeffs
    for _ in range(nder + 1):
        deg = c.shape[-1] - 1
        powers = np.arange(deg, -1, -1)
        S = offsets[:, None] ** powers[None, :]  # (K, deg+1)
        out.append(np.einsum("...j,kj->...k", c, S))
        # Derivative coefficients (highest power first): c[j] * (deg - j).
        if deg > 0:
            c = c[..., :-1] * powers[:-1]
        else:
            c = np.zeros_like(c)
    return out


def fringe_params_at_offsets(
    coeffs: NDArray, offsets: NDArray
) -> tuple[NDArray, NDArray]:
    """Evaluate fringe frequency f and rate-derivative fdot at sub-window centres.

    Parameters
    ----------
    coeffs : NDArray (n_rfi, n_bl, n_time, deg+1)
        Per-window fringe-frequency polynomials from :func:`fit_fringe_freq_poly_numpy`.
    offsets : NDArray (K,)
        Sub-window centre offsets within a window, s = t - t_c.

    Returns
    -------
    f, fdot : NDArray (n_rfi, n_bl, n_time*K)
        Fringe frequency (Hz) and its time derivative (Hz/s) at every sub-window centre,
        flattened over (window, sub-window) to match the centre grid layout.
    """
    f, fdot = _polyval_nder(coeffs, np.asarray(offsets, dtype=float), 1)
    # (n_rfi, n_bl, n_time, K) -> (n_rfi, n_bl, n_time*K)
    new_shape = coeffs.shape[:2] + (coeffs.shape[2] * len(offsets),)
    return f.reshape(new_shape), fdot.reshape(new_shape)


def size_subwindows(
    coeffs: NDArray,
    int_time: float,
    resid_tol: float,
    A: float = 3.3,
    k_max: int = 64,
) -> tuple[int, float]:
    """Choose a uniform sub-window count K from the cubic phase-curvature budget.

    The exact-Fresnel factor is exact to quadratic phase; the leading neglected term is
    the cubic, whose per-sub-window residual scales as ``A * fddot * (dt/K)^3``. Setting
    that below the relative tolerance ``resid_tol`` gives the science-floor-tied,
    washing-aware sizing rule (HANDOVER 11):

        K = ceil( (A * fddot_max * dt^3 / resid_tol)^(1/3) )

    with the measured cubic-error slope ``A ~ 3.3``. fddot_max is the maximum of
    ``|d^2 f / ds^2|`` over all windows / sources / baselines (evaluated at window edges,
    where the linear-in-s cubic-fit fddot is largest).

    Parameters
    ----------
    coeffs : NDArray (n_rfi, n_bl, n_time, deg+1)
        Per-window fringe-frequency polynomials.
    int_time : float
        Coarse integration window length dt (seconds).
    resid_tol : float
        Target relative residual epsilon (set from the science floor).
    A : float
        Cubic-error slope (measured ~3.3).
    k_max : int
        Cap on K (guards against pathological fits / non-physical inputs).

    Returns
    -------
    K : int
        Uniform sub-window count (>= 1).
    fddot_max : float
        The maximum |fddot| that set K (Hz/s^2), for diagnostics.
    """
    L = int_time / 2.0
    # fddot is linear in s for a cubic fit; its window extreme is at s = +/- L.
    _, _, fddot = _polyval_nder(coeffs, np.array([-L, L]), 2)
    fddot_max = float(np.max(np.abs(fddot)))
    K = int(np.ceil((A * fddot_max * int_time**3 / resid_tol) ** (1.0 / 3.0)))
    K = max(1, min(K, k_max))
    return K, fddot_max


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


#########################################################################
# Analytic fringe-winding factors
#
# Closed forms for the time-averaged visibility over one (sub-)window:
#
#     V_bar = (1/dt) * integral_{-dt/2}^{+dt/2} w(s) * exp(i*phi(s)) ds
#
# with a slow complex envelope w(s) (modelled linearly through its two window-edge
# values) and a fast deterministic phase
#
#     phi(s) = phi0 + 2*pi*f*s + pi*fdot*s^2 + (pi/3)*fddot*s^3 + ...
#
# F1 is exact for linear envelope + linear phase; F2_fresnel_jax is exact for linear
# envelope + quadratic phase (any |fdot*dt^2|, via the complex error function). These
# replace the oversample-and-average for one sub-window in the analytic RFI-vis path;
# the residual is set by the neglected cubic phase term (sized by the sub-window count
# K) and the envelope linearisation. All pure JAX, JIT-able and differentiable (the
# complex erf carries an exact custom JVP, see tabascal.special.cerf).
#
# Controlling dimensionless numbers: x = pi*f*dt (sinc regime) and fdot*dt^2 (accel).
#########################################################################

# Threshold below which the small-argument Taylor branches are used (G3: avoids
# catastrophic cancellation in the closed forms as x->0 or fdot->0).
_SMALL = 1e-3

# Fresnel-form validity guards (G3). The exact-Fresnel factor loses accuracy in two
# corners, both of which occur when the phase stationary point s0 = f/fdot sits far
# outside the window (fast fringe / slow chirp, e.g. near a fringe-rate turning point):
#   * the moment recursion I1 = pre*J1 - s0*I0 cancels catastrophically for large |s0|;
#   * the complex-erf argument kappa*(s0 +/- L) leaves the validated range |z| <~ 500.
# In exactly that corner the quadratic phase term is negligible over the window (or the
# visibility is heavily fringe-washed), so the linear-phase factor F1 is accurate AND
# numerically stable. We therefore fall back to F1 there. _S0_MAX is set from a direct
# accuracy scan vs a high-N oracle (F2 < 3e-4, below the envelope floor, for |s0| <=
# _S0_MAX; F1 takes over beyond, where it is at or below that error).
_S0_MAX = 200.0
_ZARG_MAX = 500.0


def _g1(x: Array) -> Array:
    """(sin x - x cos x) / (2 x^2), with the x->0 limit handled.

    Series: x/6 - x^3/60 + x^5/1680 - ... . This is the amplitude-slope x fringe
    coupling term of the linear-envelope factor.
    """
    x2 = x * x
    exact = (jnp.sin(x) - x * jnp.cos(x)) / (2.0 * jnp.where(x2 == 0, 1.0, x2))
    taylor = x / 6.0 - x * x2 / 60.0 + x * x2 * x2 / 1680.0
    return jnp.where(jnp.abs(x) < _SMALL, taylor, exact)


def _m2_over_dt3(x: Array) -> Array:
    """M2(f, dt) / dt^3 as a function of x = pi*f*dt, where
    M2 = int_{-dt/2}^{+dt/2} s^2 exp(i 2 pi f s) ds  (real).

    Closed form M2 = (dt^3/4) h(x) with h(x) = sin x/x + 2 cos x/x^2 - 2 sin x/x^3;
    limit x->0 is h = 1/3 - x^2/10 + ... so M2 -> dt^3/12. Returns h(x)/4.
    Used by the quadratic-envelope factor F2q_fresnel_jax.
    """
    x2 = x * x
    xs = jnp.where(x2 == 0, 1.0, x)
    h_exact = jnp.sin(xs) / xs + 2.0 * jnp.cos(xs) / xs**2 - 2.0 * jnp.sin(xs) / xs**3
    h_taylor = 1.0 / 3.0 - x2 / 10.0 + x2 * x2 / 168.0
    h = jnp.where(jnp.abs(x) < _SMALL, h_taylor, h_exact)
    return h / 4.0


def F1(phi0: Array, w0: Array, wp: Array, f: Array, dt: float) -> Array:
    """Linear-envelope, linear-phase fringe-winding factor (exact in that regime).

    V = exp(i phi0) [ w0 sinc(f dt) + i wp dt (sin x - x cos x)/(2 x^2) ],  x = pi f dt.

    This is the fdot -> 0 (no fringe acceleration) limit of :func:`F2_fresnel_jax`.

    Parameters
    ----------
    phi0 : Array
        Phase at the sub-window centre (radians).
    w0 : Array
        Envelope value at the sub-window centre (complex), = (w_plus + w_minus)/2.
    wp : Array
        Envelope slope dw/ds (complex), = (w_plus - w_minus)/dt.
    f : Array
        Fringe frequency at the sub-window centre (Hz).
    dt : float
        Sub-window length (seconds).

    Returns
    -------
    Array
        The time-averaged visibility over the sub-window (complex).
    """
    x = jnp.pi * f * dt
    return jnp.exp(1j * phi0) * (w0 * jnp.sinc(f * dt) + 1j * wp * dt * _g1(x))


def F2_fresnel_jax(
    phi0: Array, w0: Array, wp: Array, f: Array, fdot: Array, dt: float
) -> Array:
    """Exact linear-envelope, quadratic-phase fringe-winding factor.

    Closed form of V_bar = (1/dt) int w(s) exp(i phi(s)) ds for a linear envelope
    w(s) = w0 + wp*s and a quadratic phase phi(s) = phi0 + 2 pi f s + pi fdot s^2,
    via the complex error function (complete-the-square -> Fresnel integral). Exact
    for *any* f*dt and fdot*dt^2; the only residual is the neglected cubic phase term,
    controlled by sub-windowing (the analytic RFI-vis path picks K so fddot*dt^3 stays
    below tolerance per sub-window). Falls back to :func:`F1` where |fdot| ~ 0.

    Parameters
    ----------
    phi0 : Array
        Phase at the sub-window centre (radians).
    w0, wp : Array
        Envelope centre value and slope (complex); w0=(w_plus+w_minus)/2,
        wp=(w_plus-w_minus)/dt.
    f, fdot : Array
        Fringe frequency (Hz) and fringe-rate derivative (Hz/s) at the centre.
    dt : float
        Sub-window length (seconds).

    Returns
    -------
    Array
        The time-averaged visibility over the sub-window (complex).
    """
    a = 2.0 * jnp.pi * f
    b = jnp.pi * fdot
    L = dt / 2.0
    b_safe = jnp.where(jnp.abs(b) < 1e-12, 1.0, b)
    s0 = a / (2.0 * b_safe)
    pre = jnp.exp(-1j * a**2 / (4.0 * b_safe))
    kappa = jnp.sqrt(-1j * b_safe + 0j)
    tp = L + s0
    tm = -L + s0
    # Fall back to the linear-phase form when fdot ~ 0 (F1 exact) or in the large-s0 /
    # large-erf-argument corner where the Fresnel form is numerically unstable.
    zmax = jnp.maximum(jnp.abs(kappa * tp), jnp.abs(kappa * tm))
    use_lin = (jnp.abs(b) < 1e-12) | (jnp.abs(s0) > _S0_MAX) | (zmax > _ZARG_MAX)
    # Feed benign arguments into the discarded Fresnel branch so it never produces
    # NaN/Inf that would poison gradients through the jnp.where (G5).
    tp_s = jnp.where(use_lin, 0.0, tp)
    tm_s = jnp.where(use_lin, 0.0, tm)
    s0_s = jnp.where(use_lin, 0.0, s0)
    I0 = (
        pre
        * (jnp.sqrt(jnp.pi) / (2.0 * kappa))
        * (cerf(kappa * tp_s) - cerf(kappa * tm_s))
    )
    J1 = (jnp.exp(1j * b_safe * tp_s**2) - jnp.exp(1j * b_safe * tm_s**2)) / (
        2j * b_safe
    )
    I1 = pre * J1 - s0_s * I0
    V = jnp.exp(1j * phi0) * (w0 * I0 + wp * I1) / dt
    return jnp.where(use_lin, F1(phi0, w0, wp, f, dt), V)


def F2q_fresnel_jax(
    phi0: Array,
    w0: Array,
    wp: Array,
    wpp: Array,
    f: Array,
    fdot: Array,
    dt: float,
) -> Array:
    """Exact quadratic-envelope, quadratic-phase fringe-winding factor (optional).

    Promotes the envelope model from linear to quadratic w(s)=w0+wp*s+0.5*wpp*s^2,
    adding the second-moment integral I2 to :func:`F2_fresnel_jax`. This drops the
    envelope-linearisation floor a further order ((dt/l)^2 -> (dt/l)^3 with a fringe),
    at the cost of one extra envelope sample per antenna (edge + centre). Here w0 is
    the sub-window-CENTRE envelope value (not the edge average).

    Caveat: I2 has a large-s0 cancellation (large f, small fdot); fall back to the
    perturbative form there (same trigger as the fdot -> 0 branch). Falls back to a
    quadratic-envelope / linear-phase form where |fdot| ~ 0.
    """
    a = 2.0 * jnp.pi * f
    b = jnp.pi * fdot
    L = dt / 2.0
    b_safe = jnp.where(jnp.abs(b) < 1e-12, 1.0, b)
    s0 = a / (2.0 * b_safe)
    pre = jnp.exp(-1j * a**2 / (4.0 * b_safe))
    kappa = jnp.sqrt(-1j * b_safe + 0j)
    tp = L + s0
    tm = -L + s0
    # Fall back to the quadratic-envelope / linear-phase form when fdot ~ 0 or in the
    # large-s0 corner (here I2's cancellation is even sharper than I1's). Feed benign
    # arguments into the discarded Fresnel branch to keep gradients NaN-free (G5).
    zmax = jnp.maximum(jnp.abs(kappa * tp), jnp.abs(kappa * tm))
    use_lin = (jnp.abs(b) < 1e-12) | (jnp.abs(s0) > _S0_MAX) | (zmax > _ZARG_MAX)
    tp_s = jnp.where(use_lin, 0.0, tp)
    tm_s = jnp.where(use_lin, 0.0, tm)
    s0_s = jnp.where(use_lin, 0.0, s0)
    ep = jnp.exp(1j * b_safe * tp_s**2)
    em = jnp.exp(1j * b_safe * tm_s**2)
    J0 = (jnp.sqrt(jnp.pi) / (2.0 * kappa)) * (cerf(kappa * tp_s) - cerf(kappa * tm_s))
    J1 = (ep - em) / (2j * b_safe)
    J2 = (tp_s * ep - tm_s * em) / (2j * b_safe) - J0 / (2j * b_safe)
    I0 = pre * J0
    I1 = pre * (J1 - s0_s * J0)
    I2 = pre * (J2 - 2.0 * s0_s * J1 + s0_s**2 * J0)
    V = jnp.exp(1j * phi0) * (w0 * I0 + wp * I1 + 0.5 * wpp * I2) / dt
    x = jnp.pi * f * dt
    V_lin = jnp.exp(1j * phi0) * (
        w0 * jnp.sinc(f * dt)
        + 1j * wp * dt * _g1(x)
        + 0.5 * wpp * dt**2 * _m2_over_dt3(x)
    )
    return jnp.where(use_lin, V_lin, V)


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
