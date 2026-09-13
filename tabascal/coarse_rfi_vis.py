"""RFI visibilities from the data grid: the function a compiled kernel replaces.

:func:`coarse_rfi_vis` is the whole of the boundary. Everything comes in on the
data grid, the data-grid visibilities go out, and the fine grid -- the
``n_int_freq x n_int_time`` samples inside each data cell that the fine-grid
route (``trajectory:FixedOrbit``, ``rfi_signal:ComplexRFIVarAnt``,
``rfi_vis:RiemannVisFFI``) carries as model state -- exists only inside it, one
time cell at a time. A kernel implementing this function, with its JVP and VJP
with respect to ``rfi_A``, drops into
:class:`tabascal.components.rfi_vis.PolyInterpVis` exactly where ``ri_kernels``
drops into ``RiemannVisFFI``. Nothing else in the route needs to change: the
components that produce the inputs run on the host at setup or are a plain
inverse FFT, and are the same for a compiled kernel as for this reference.

:func:`analytic_rfi_vis` is the pure-JAX alternative for fast baselines. It
uses the same coarse inputs and frequency quadrature, but integrates the
amplitude polynomial against a quadratic phase, with a cubic correction. Its time
table holds monomial coefficients rather than fine-sample weights, so the
work has no Nyquist floor and does not grow with fringe winding.

Inputs
------
Per source ``r``, antenna ``a``, channel ``f`` and time cell ``t``. Only
``rfi_A`` carries a gradient; every other input is a constant of the run.

- ``rfi_A`` ``(n_rfi, n_ant, n_freq, n_time)``, complex: the RFI signal on
  the data grid.
- ``rfi_phase`` ``(n_rfi, n_ant, n_freq, n_time)``: the phase at the channel
  and cell centre, reduced to one turn.
- ``rfi_delay_poly_us`` ``(n_rfi, n_ant, n_time, n_path)``: the geometric delay
  ``tau`` (us) and its time derivatives ``tau_k`` (us/s^k) at the cell centre,
  relative to the array mean (see below). The delay is ``-(range + w) / c``, so
  the phase is ``2 pi f tau``: the fine-grid route's ``rfi_delay_us`` convention.
- ``w_freq`` ``(n_freq, n_sf, n_int_freq)`` and ``start_freq`` ``(n_freq,)``:
  interpolation weights across each channel, and the first channel of each
  channel's stencil.
- ``w_time`` ``(n_time, n_st, n_int_time)`` and ``start_time`` ``(n_time,)``:
  the same across each cell.
- ``dnu_mhz`` ``(n_int_freq,)`` and ``dt`` ``(n_int_time,)``: the fine offsets
  from the channel centre (MHz) and the cell centre (s).
- ``freqs_mhz`` ``(n_freq,)``: channel centres (MHz). MHz times microseconds
  is cycles, so the phase is formed without a scaling constant.
- ``a1``, ``a2`` ``(n_bl,)``: the two antennas of each baseline.

Output: ``vis_rfi`` ``(n_bl, n_freq, n_time)``, complex.

The computation
---------------
For one cell ``(f, t)`` and one of its fine samples ``(u, v)``, with ``c`` the
speed of light::

    signal   A[r, a](u, v) = sum_k sum_l  w_freq[f, k, u] w_time[t, l, v]
                                          rfi_A[r, a, start_freq[f] + k, start_time[t] + l]
    delay    dtau[r, a](v) = sum_{k >= 1} rfi_delay_poly_us[r, a, t, k] dt[v]^k / k!
    phase    phi[r, a](u, v) = rfi_phase[r, a, f, t]
                               + 2 pi ( (freqs_mhz[f] + dnu_mhz[u]) dtau[r, a](v)
                                        + dnu_mhz[u] rfi_delay_poly_us[r, a, t, 0] )
    sample   S[r, a](u, v) = A[r, a](u, v) exp(i phi[r, a](u, v))
    result   vis_rfi[b, f, t] = mean_{u, v} sum_r S[r, a1[b]](u, v) conj(S[r, a2[b]](u, v))

The last line is the integrand of the fine-grid Riemann sum
(:func:`tabascal.interferometry.calculate_rfi_vis_fine`, averaged over each
cell as :func:`~tabascal.interferometry.calculate_rfi_vis_blocked` does);
the three lines before it are what replace reading ``A`` and ``phi`` from the
fine-grid state. The signal interpolates the ``2h + 1`` nearest coarse samples
on each axis; the phase is exact across the channel (linear in frequency) and
a Taylor series across the cell.

The weights are data. :func:`tabascal.poly_interp.interp_tables` fills them
with the polynomial through the stencil; the conditional mean of a Gaussian
process prior, or any other linear interpolant, is a different table and the
same kernel.

Precision
---------
The phase is arranged so that a single-precision kernel never forms a large
number and then reduces it. The unreduced phase ``2 pi freqs L / c`` is of
order a million turns, which float32 cannot hold to a fraction of a turn;
``rfi_phase`` carries it reduced, computed in float64 on the host. The kernel
adds to it only the change across the cell and across the channel. Do not
rebuild ``rfi_phase`` from ``rfi_delay_poly_us[..., 0]``
inside a kernel.

Derivatives
-----------
``rfi_A`` is the only differentiated input; ``rfi_phase`` and ``rfi_delay_poly_us``
come from a fixed orbit and the rest are tables. The result is bilinear in the
fine samples ``S``, and ``S`` is linear in ``rfi_A`` (the interpolation is
linear and the phase factor is a constant), so both derivatives are structural:

- JVP: with ``dS`` the tangent of ``rfi_A`` pushed through the same
  interpolation and phase factor,
  ``dvis[b] = mean sum_r ( dS[r, a1[b]] conj(S[r, a2[b]]) + S[r, a1[b]] conj(dS[r, a2[b]]) )``.
- VJP: the cotangent of ``vis_rfi`` is scattered to the fine samples of each
  baseline's two antennas, each weighted by the other antenna's sample -- the
  fine-grid kernel's own transpose -- then multiplied by the conjugate phase
  factor and pushed back through the weight tables, which is a stencil-sized
  scatter-add onto the data grid, and nothing of the fine grid survives it.

Sign and conjugation conventions are JAX's for complex inputs; the reference
for both is JAX's derivative of this function, and
``tests/test_coarse_rfi_vis.py`` holds that to finite differences. A kernel is
validated the way ``ri_kernels`` is: value, JVP and VJP against this function.

Memory
------
The reference forms one time cell at a time, ``(n_bl, n_rfi, n_freq,
n_int_freq, n_int_time)`` complex, under ``jax.checkpoint`` so the reverse
pass recomputes the cell rather than keeping it. That is what makes it usable
at the sizes the fine-grid route runs at, not what makes it fast: it gathers
per baseline and recomputes each antenna's samples for every baseline it is
on. A kernel would stage each antenna's fine samples once per cell.
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

def fine_signal(
    rfi_A: Array, w_freq: Array, start_freq: Array, w_time: Array, start_time: Array
) -> Array:
    """The fine samples of one time cell, for every source, antenna and channel.

    ``w_time`` ``(n_st, n_int_time)`` and ``start_time`` (a scalar) are the
    cell's own row of the tables; the frequency tables are whole, since every
    channel of the cell is interpolated at once.

    Returns ``(n_rfi, n_ant, n_freq, n_int_freq, n_int_time)``.
    """
    n_sf, n_st = w_freq.shape[1], w_time.shape[0]

    # The stencil: the n_st cells around this one, then the n_sf channels
    # around each channel.
    idx_time = start_time + jnp.arange(n_st)  # (n_st,)
    idx_freq = start_freq[:, None] + jnp.arange(n_sf)  # (n_freq, n_sf)
    stencil = jnp.take(rfi_A, idx_time, axis=3)  # (n_rfi, n_ant, n_freq, n_st)
    stencil = jnp.take(stencil, idx_freq, axis=2)  # (n_rfi, n_ant, n_freq, n_sf, n_st)

    # Separable weights: one table per axis.
    return jnp.einsum("rafkl,fku,lv->rafuv", stencil, w_freq, w_time)


def fine_phase(
    rfi_phase: Array, rfi_delay: Array, freqs_mhz: Array, dnu_mhz: Array, dt: Array
) -> Array:
    """The fine phase of one time cell, for every source, antenna and channel.

    ``rfi_phase`` ``(n_rfi, n_ant, n_freq)`` and ``rfi_delay`` ``(n_rfi, n_ant,
    n_path)`` are the cell's own slices; the delay in microseconds and its
    derivatives in microseconds per second^k, the frequencies in MHz.

    Returns ``(n_rfi, n_ant, n_freq, n_int_freq, n_int_time)``.
    """
    # The delay's change across the cell, from its Taylor series at the centre.
    d_tau = jnp.zeros(rfi_delay.shape[:-1] + dt.shape, dtype=dt.dtype)
    for k in range(1, rfi_delay.shape[-1]):
        d_tau = d_tau + rfi_delay[..., k, None] * dt**k / math.factorial(k)
    # (n_rfi, n_ant, n_int_time)

    nu = freqs_mhz[:, None] + dnu_mhz[None, :]  # (n_freq, n_int_freq), MHz
    across_cell = nu[None, None, :, :, None] * d_tau[:, :, None, None, :]
    across_channel = (
        dnu_mhz[None, None, None, :, None] * rfi_delay[..., 0][:, :, None, None, None]
    )
    # MHz times microseconds is cycles.
    return rfi_phase[..., None, None] + 2.0 * jnp.pi * (across_cell + across_channel)


def cell_vis(S: Array, a1: Array, a2: Array) -> Array:
    """The cell's visibilities from its fine samples, ``(n_bl, n_freq)``.

    ``S`` is ``(n_rfi, n_ant, n_freq, n_int_freq, n_int_time)``: the summed
    product over sources, averaged over the cell's fine samples.
    """
    S = jnp.swapaxes(S, 0, 1)  # antenna axis first, for the baseline gather
    product = S[a1] * jnp.conj(S[a2])  # (n_bl, n_rfi, n_freq, n_int_freq, n_int_time)
    return jnp.mean(jnp.sum(product, axis=1), axis=(-2, -1))


def coarse_rfi_vis(
    rfi_A: Array,
    rfi_phase: Array,
    rfi_delay: Array,
    w_freq: Array,
    start_freq: Array,
    w_time: Array,
    start_time: Array,
    dnu_mhz: Array,
    dt: Array,
    freqs_mhz: Array,
    a1: Array,
    a2: Array,
) -> Array:
    """The data-grid RFI visibilities, ``(n_bl, n_freq, n_time)``.

    The reference implementation of the function described in the module
    docstring, and the boundary a compiled kernel replaces. One time cell per
    step of a ``lax.map``, each cell under ``jax.checkpoint``.
    """
    n_time = rfi_A.shape[-1]

    # prevent_cse=False is what JAX documents for a remat body under lax.scan.
    @functools.partial(jax.checkpoint, prevent_cse=False)
    def one_cell(t):
        A = fine_signal(rfi_A, w_freq, start_freq, w_time[t], start_time[t])
        phase = fine_phase(rfi_phase[..., t], rfi_delay[:, :, t], freqs_mhz, dnu_mhz, dt)
        return cell_vis(A * jnp.exp(1.0j * phase), a1, a2)

    vis = lax.map(one_cell, jnp.arange(n_time))  # (n_time, n_bl, n_freq)
    return jnp.moveaxis(vis, 0, -1)


def linear_phase_moments(a: Array, degree: int) -> Array:
    """The moments ``mean_{[-1,1]} x**m exp(i*a*x)``, through ``degree``.

    Integration by parts divides by the large winding, so upward recurrence
    is stable only while ``m <= |a|``. Above that point we run the same
    identity downwards from a zero tail, 64 orders beyond the last requested
    moment. The unwanted solution then contracts by ``|a|/m`` at every step.
    This also supplies the zero-frequency limit without a division by zero or
    a cancellation-prone Taylor series at moderate winding.
    """
    a = jnp.asarray(a)
    positive, negative = jnp.exp(1j * a), jnp.exp(-1j * a)
    safe_a = jnp.where(jnp.abs(a) >= 1, a, 1)
    zero = jnp.sinc(a / jnp.pi).astype(positive.dtype)

    def upward_step(last, m):
        sign = jnp.where(m % 2 == 0, 1, -1)
        value = ((positive - sign * negative) / 2 - m * last) / (1j * safe_a)
        value = jnp.where(m <= jnp.abs(a), value, 0)
        return value, value

    _, upward = lax.scan(upward_step, zero, jnp.arange(1, degree + 1))
    upward = jnp.concatenate((zero[..., None], jnp.moveaxis(upward, 0, -1)), axis=-1)

    # Large a uses the upward result throughout. Giving the unused downward
    # branch a benign argument keeps masked overflows out of differentiation.
    down_a = jnp.where(jnp.abs(a) <= degree, a, 0)
    ep, em = jnp.exp(1j * down_a), jnp.exp(-1j * down_a)
    values = jnp.zeros(a.shape + (degree + 1,), dtype=positive.dtype)

    def step(carry, m):
        last, values = carry
        previous = ((ep - (-1.)**m * em) / 2 - 1j * down_a * last) / m
        values = lax.cond(m <= degree + 1, lambda v: v.at[..., m - 1].set(previous), lambda v: v, values)
        return (previous, values), None

    (_, downward), _ = lax.scan(
        step, (jnp.zeros_like(positive), values), jnp.arange(degree + 64, 0, -1), unroll=1,
    )
    return jnp.where(jnp.arange(degree + 1) <= jnp.abs(a)[..., None], upward, downward)


def quadratic_phase_moments(a: Array, b: Array, degree: int, terms: int = 16) -> Array:
    """Normalised moments on [-1, 1] of ``exp(i*(a*x+b*x*x))``.

    Outside or beyond the neighbourhood of the stationary point, expand the
    curvature about linear-phase moments. The caller splits the cell first:
    with ``|b| <= 1`` per piece, 16 terms leave less than 2e-13 absolute
    remainder. Near the stationary point the Fresnel seed and its upward
    recurrence are stable, except at small b where division by b is itself
    ill-conditioned. There the same convergent series supplies the continuous
    limit instead.
    """
    from jax.scipy.special import fresnel

    a, b = jnp.broadcast_arrays(a, b)
    stationary = (jnp.abs(a) <= 2.5 * jnp.abs(b)) & (jnp.abs(b) >= 1)
    series_a, series_b = jnp.where(stationary, 0., a), jnp.where(stationary, 0., b)
    linear = linear_phase_moments(series_a, degree + 2 * (terms - 1))
    series = jnp.zeros_like(linear[..., :degree + 1])
    coefficient = jnp.ones_like(series_a, dtype=linear.dtype)

    def series_step(carry, k):
        series, coefficient = carry
        window = lax.dynamic_slice_in_dim(linear, 2*k, degree + 1, axis=-1)
        series = series + coefficient[..., None] * window
        coefficient = coefficient * (1j * series_b) / (k + 1)
        return (series, coefficient), None

    (series, _), _ = lax.scan(series_step, (series, coefficient), jnp.arange(terms), unroll=1)

    # Complete the square only near the stationary point: doing so at large
    # winding subtracts almost equal Fresnel values with huge phase arguments.
    fa, fb = jnp.where(stationary, a, 0.), jnp.where(stationary, b, 1.)
    scale = jnp.sqrt(2 * jnp.abs(fb) / jnp.pi)
    shift = fa / (2 * fb)
    sp, cp = fresnel(scale * (1 + shift))
    sm, cm = fresnel(scale * (-1 + shift))
    zero = jnp.exp(-1j * fa * shift / 2) * ((cp - cm) + 1j * jnp.sign(fb) * (sp - sm)) / (2 * scale)
    ep, em = jnp.exp(1j * (fa + fb)), jnp.exp(1j * (-fa + fb))

    def moment_step(carry, m):
        before_last, last = carry
        previous = (m - 1) * before_last
        sign = jnp.where((m - 1) % 2 == 0, 1, -1)
        value = ((ep - sign * em) / 2 - previous - 1j * fa * last) / (2j * fb)
        return (last, value), value

    _, moments = lax.scan(moment_step, (jnp.zeros_like(zero), zero), jnp.arange(1, degree + 1))
    moments = jnp.concatenate((zero[..., None], jnp.moveaxis(moments, 0, -1)), axis=-1)
    return jnp.where(stationary[..., None], moments, series)


def analytic_rfi_vis(
    rfi_A: Array, rfi_phase: Array, rfi_delay: Array,
    w_freq: Array, start_freq: Array, g_time: Array, start_time: Array,
    dnu_mhz: Array, int_time: Array, freqs_mhz: Array, a1: Array, a2: Array,
    *, segments: int = 2, terms: int = 6, cubic_terms: int = 3,
) -> Array:
    """Integrate the amplitude polynomial against the quadratic delay phase.

    ``g_time[t,l,m]`` is the Lagrange basis in powers of x = 2*tau/T. The
    frequency contraction is unchanged, so finite channel integration retains
    exactly the reference's frequency offsets. Time integration is analytic:
    multiply the antenna polynomials by convolution, then contract against
    phase moments. The quadratic phase is integrated analytically and residual
    cubic phase is expanded on each piece. Three terms suffice at the measured
    cubic coefficients; zero deliberately drops cubic phase for comparison.
    Derivatives above order three are omitted.

    Equal pieces keep curvature small without imposing a Nyquist sample count.
    Translation of both the amplitude and phase is exact, including the
    constant phase of each piece. The working arrays carry polynomial degree,
    not fringe winding. Only the data-grid amplitude is differentiated.
    """
    amp_degree = g_time.shape[-1] - 1
    degree = 2 * amp_degree
    n_cubic = max(1, cubic_terms)
    moment_degree = degree + 3 * (n_cubic - 1)
    radius = 1. / segments
    nu = freqs_mhz[:, None] + dnu_mhz[None, :]
    orders = jnp.arange(degree + 1)
    binomial = [
        [math.comb(m, j) if j <= m else 0 for j in range(degree + 1)]
        for m in range(degree + 1)
    ]
    radius_powers = np.power(radius, np.arange(degree + 1))

    @functools.partial(jax.checkpoint, prevent_cse=False)
    def one_cell(t):
        amplitude = fine_signal(rfi_A, w_freq, start_freq, g_time[t], start_time[t])
        p, q = amplitude[:, a1], amplitude[:, a2].conj()
        if amp_degree == 0:
            product = p * q
        else:
            # Each row holds q[m-j], with zeros outside its polynomial. Contract
            # all output degrees together, retaining full float32 precision on GPU.
            indices = orders[:, None] - jnp.arange(amp_degree + 1)
            shifted_q = jnp.where(
                (indices >= 0) & (indices <= amp_degree),
                q[..., jnp.clip(indices, 0, amp_degree)], 0,
            )
            product = jnp.einsum("...j,...mj->...m", p, shifted_q, precision=lax.Precision.HIGHEST)
        delay_cell, phase_cell = rfi_delay[:, :, t], rfi_phase[..., t]
        delay = delay_cell[:, a1] - delay_cell[:, a2]
        phi = phase_cell[:, a1] - phase_cell[:, a2]
        phi0 = phi[..., None] + 2 * jnp.pi * dnu_mhz * delay[..., 0, None, None]
        d1 = delay[..., 1] if rfi_delay.shape[-1] > 1 else jnp.zeros_like(delay[..., 0])
        d2 = delay[..., 2] if rfi_delay.shape[-1] > 2 else jnp.zeros_like(delay[..., 0])
        a = 2 * jnp.pi * nu * d1[..., None, None] * (int_time / 2)
        b = jnp.pi * nu * d2[..., None, None] * (int_time / 2)**2
        d3 = delay[..., 3] if rfi_delay.shape[-1] > 3 and cubic_terms else jnp.zeros_like(delay[..., 0])
        c = (jnp.pi / 3) * nu * d3[..., None, None] * (int_time / 2)**3

        def piece(total, i):
            centre = -1 + (2 * i + 1) * radius
            # x = centre + radius*y. Keeping coefficients dimensionless avoids
            # powers of a seconds-valued T in the moment recurrence.
            # Form all powers together, then sum each translated coefficient
            # from left to right in the source degree.
            centre_powers = centre**orders
            combinations = jnp.asarray(binomial, dtype=centre.dtype)
            radii = jnp.asarray(radius_powers, dtype=centre.dtype)

            def translate(shifted, m):
                power = centre_powers[jnp.maximum(m - orders, 0)]
                value = combinations[m] * power * radii * product[..., m, None]
                return jnp.where(orders <= m, shifted + value, shifted), None

            shifted, _ = lax.scan(translate, jnp.zeros_like(product), orders, unroll=1)
            moments = quadratic_phase_moments(
                (a + 2*b*centre + 3*c*centre**2)*radius,
                (b + 3*c*centre)*radius**2, moment_degree, terms,
            )
            # Translate the cubic exactly too. Only the residual c*r^3*y^3
            # needs expansion; its constant, linear and quadratic parts are
            # already in the phase and moments. The residual coefficient
            # falls with the cube of the piece width.
            coefficient = jnp.ones_like(a, dtype=product.dtype)
            integral = jnp.zeros_like(coefficient)

            def cubic_step(carry, k):
                integral, coefficient = carry
                window = lax.dynamic_slice_in_dim(moments, 3*k, degree + 1, axis=-1)
                integral = integral + coefficient * jnp.sum(shifted * window, axis=-1)
                coefficient = coefficient * (1j * c * radius**3) / (k + 1)
                return (integral, coefficient), None

            (integral, _), _ = lax.scan(
                cubic_step, (integral, coefficient), jnp.arange(n_cubic), unroll=1,
            )
            phase = phi0 + a*centre + b*centre**2 + c*centre**3
            return total + jnp.exp(1j * phase) * integral / segments, None

        result, _ = lax.scan(
            piece, jnp.zeros(product.shape[:-1], dtype=product.dtype), jnp.arange(segments), unroll=1,
        )
        return jnp.sum(jnp.mean(result, axis=-1), axis=0)

    return jnp.moveaxis(lax.map(one_cell, jnp.arange(rfi_A.shape[-1])), 0, -1)
