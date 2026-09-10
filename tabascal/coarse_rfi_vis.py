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

Inputs
------
Per source ``r``, antenna ``a``, channel ``f`` and time cell ``t``. Only
``rfi_A`` carries a gradient; every other input is a constant of the run.

- ``rfi_A`` ``(n_rfi, n_ant, n_freq, n_time)``, complex: the RFI signal on
  the data grid.
- ``rfi_phase`` ``(n_rfi, n_ant, n_freq, n_time)``: the phase at the channel
  and cell centre, reduced to one turn.
- ``rfi_path`` ``(n_rfi, n_ant, n_time, n_path)``: the path ``L`` (m) and its
  time derivatives ``L_k`` (m/s^k) at the cell centre.
- ``w_freq`` ``(n_freq, n_sf, n_int_freq)`` and ``start_freq`` ``(n_freq,)``:
  interpolation weights across each channel, and the first channel of each
  channel's stencil.
- ``w_time`` ``(n_time, n_st, n_int_time)`` and ``start_time`` ``(n_time,)``:
  the same across each cell.
- ``dnu`` ``(n_int_freq,)`` and ``dt`` ``(n_int_time,)``: the fine offsets from
  the channel centre (Hz) and the cell centre (s).
- ``freqs`` ``(n_freq,)``: channel centres (Hz).
- ``a1``, ``a2`` ``(n_bl,)``: the two antennas of each baseline.

Output: ``vis_rfi`` ``(n_bl, n_freq, n_time)``, complex.

The computation
---------------
For one cell ``(f, t)`` and one of its fine samples ``(u, v)``, with ``c`` the
speed of light::

    signal   A[r, a](u, v) = sum_k sum_l  w_freq[f, k, u] w_time[t, l, v]
                                          rfi_A[r, a, start_freq[f] + k, start_time[t] + l]
    path     dL[r, a](v)   = sum_{k >= 1} rfi_path[r, a, t, k] dt[v]^k / k!
    phase    phi[r, a](u, v) = rfi_phase[r, a, f, t]
                               - (2 pi / c) ( (freqs[f] + dnu[u]) dL[r, a](v)
                                              + dnu[u] rfi_path[r, a, t, 0] )
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
adds to it only the change across the cell and across the channel, at most a
few hundred radians. Do not rebuild ``rfi_phase`` from ``rfi_path[..., 0]``
inside a kernel.

Derivatives
-----------
``rfi_A`` is the only differentiated input; ``rfi_phase`` and ``rfi_path``
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
from jax import Array, lax

#: Speed of light in m/s, as :mod:`tabascal.interferometry` uses it.
C_LIGHT = 299792458.0


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
    rfi_phase: Array, rfi_path: Array, freqs: Array, dnu: Array, dt: Array
) -> Array:
    """The fine phase of one time cell, for every source, antenna and channel.

    ``rfi_phase`` ``(n_rfi, n_ant, n_freq)`` and ``rfi_path`` ``(n_rfi, n_ant,
    n_path)`` are the cell's own slices.

    Returns ``(n_rfi, n_ant, n_freq, n_int_freq, n_int_time)``.
    """
    # The path's change across the cell, from its Taylor series at the centre.
    d_path = jnp.zeros(rfi_path.shape[:-1] + dt.shape, dtype=dt.dtype)
    for k in range(1, rfi_path.shape[-1]):
        d_path = d_path + rfi_path[..., k, None] * dt**k / math.factorial(k)
    # (n_rfi, n_ant, n_int_time)

    nu = freqs[:, None] + dnu[None, :]  # (n_freq, n_int_freq)
    across_cell = nu[None, None, :, :, None] * d_path[:, :, None, None, :]
    across_channel = (
        dnu[None, None, None, :, None] * rfi_path[..., 0][:, :, None, None, None]
    )
    return rfi_phase[..., None, None] - (2.0 * jnp.pi / C_LIGHT) * (
        across_cell + across_channel
    )


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
    rfi_path: Array,
    w_freq: Array,
    start_freq: Array,
    w_time: Array,
    start_time: Array,
    dnu: Array,
    dt: Array,
    freqs: Array,
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
        phase = fine_phase(rfi_phase[..., t], rfi_path[:, :, t], freqs, dnu, dt)
        return cell_vis(A * jnp.exp(1.0j * phase), a1, a2)

    vis = lax.map(one_cell, jnp.arange(n_time))  # (n_time, n_bl, n_freq)
    return jnp.moveaxis(vis, 0, -1)
