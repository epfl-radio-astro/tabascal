"""Tests for tabascal.coarse_rfi_vis: the kernel boundary of the data-grid route.

The function is held to the fine-grid Riemann sum on the fine grid it forms,
and its derivatives to finite differences. A compiled kernel replacing it is
validated the same way, against this function.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from tabascal.coarse_rfi_vis import cell_vis, coarse_rfi_vis, fine_phase, fine_signal
from tabascal.interferometry import calculate_rfi_vis_blocked, calculate_rfi_vis_fine
from tabascal.poly_interp import interp_tables

from .components.conftest import active_precision


def _offsets(n_int, spacing):
    return (np.arange(n_int) - n_int // 2) * spacing / n_int


def make_inputs(n_rfi=2, n_ant=4, n_freq=3, n_time=6, n_int_freq=2, n_int_time=5, half_width=1, seed=0):
    """Random data-grid inputs with the tables a config of that shape would build."""
    rng = np.random.default_rng(seed)
    int_time, chan_width = 2.0, 1e4
    dt = _offsets(n_int_time, int_time)
    dnu = _offsets(n_int_freq, chan_width)
    freqs = 1.5e8 + chan_width * np.arange(n_freq)
    dnu_mhz, freqs_mhz = dnu / 1e6, freqs / 1e6
    w_time, start_time = interp_tables(n_time, half_width, dt / int_time)
    w_freq, start_freq = interp_tables(n_freq, half_width, dnu / chan_width)
    a1, a2 = np.triu_indices(n_ant, 1)

    shape = (n_rfi, n_ant, n_freq, n_time)
    rfi_A = rng.normal(size=shape) + 1.0j * rng.normal(size=shape)
    rfi_phase = rng.uniform(-2 * np.pi, 0.0, size=shape)
    # The delay relative to the array mean, in microseconds, as FixedOrbitCoarse
    # writes it: a kilometre-scale array against a LEO satellite.
    rfi_delay = np.stack(
        [
            rng.normal(0.0, 1.7, shape[:2] + (n_time,)),  # us
            rng.normal(0.0, 0.07, shape[:2] + (n_time,)),  # us/s
            rng.normal(0.0, 3e-4, shape[:2] + (n_time,)),  # us/s^2
            rng.normal(0.0, 3e-6, shape[:2] + (n_time,)),  # us/s^3
        ],
        axis=-1,
    )
    arrays = (rfi_A, rfi_phase, rfi_delay, w_freq, start_freq, w_time, start_time, dnu_mhz, dt, freqs_mhz, a1, a2)
    return [jnp.asarray(x) for x in arrays]


def fine_grid(args):
    """The fine amplitude and phase the kernel forms, assembled whole.

    Returned in the fine-grid components' layout, ``(n_rfi, n_ant, n_freq_fine,
    n_time_fine)``, so the fine-grid functions can be fed with them.
    """
    rfi_A, rfi_phase, rfi_delay, w_freq, start_freq, w_time, start_time, dnu, dt, freqs = args[:10]
    n_rfi, n_ant, n_freq, n_time = rfi_A.shape
    n_int_freq, n_int_time = len(dnu), len(dt)
    A = np.stack([fine_signal(rfi_A, w_freq, start_freq, w_time[t], start_time[t]) for t in range(n_time)], -1)
    phase = np.stack([fine_phase(rfi_phase[..., t], rfi_delay[:, :, t], freqs, dnu, dt) for t in range(n_time)], -1)

    def flat(x):  # (n_rfi, n_ant, n_freq, n_int_freq, n_int_time, n_time) -> fine-grid layout
        return jnp.asarray(np.transpose(x, (0, 1, 2, 3, 5, 4)).reshape(n_rfi, n_ant, n_freq * n_int_freq, n_time * n_int_time))

    return flat(A), flat(phase)


def _rtol():
    return 1e-4 if active_precision() == "single" else 1e-10


class TestValue:

    @pytest.mark.parametrize(
        "sizes",
        [
            dict(),
            dict(n_int_freq=1, n_int_time=1),
            dict(n_int_freq=1, n_int_time=4, half_width=0),
            dict(n_freq=1, n_time=7, n_int_time=6, half_width=2),
            dict(n_ant=2, n_rfi=1, n_freq=2),
        ],
    )
    def test_matches_the_fine_grid_riemann_sum_on_the_grid_it_forms(self, sizes):
        args = make_inputs(**sizes)
        A_fine, phase_fine = fine_grid(args)
        n_int_freq, n_int_time = len(args[7]), len(args[8])
        expected = calculate_rfi_vis_blocked(A_fine, phase_fine, args[10], args[11], n_int_freq, n_int_time, None)
        got = coarse_rfi_vis(*args)
        assert got.shape == expected.shape
        np.testing.assert_allclose(got, expected, rtol=_rtol(), atol=_rtol() * float(jnp.abs(expected).max()))

    def test_one_sample_per_cell_is_the_data_grid_product(self):
        """With no fine grid there is nothing to rebuild: the reduced phase and
        the coarse signal go straight into the fine-grid product."""
        args = make_inputs(n_int_freq=1, n_int_time=1)
        rfi_A, rfi_phase = args[0], args[1]
        expected = calculate_rfi_vis_fine(rfi_A, rfi_phase, args[10], args[11])
        np.testing.assert_allclose(coarse_rfi_vis(*args), expected, rtol=_rtol(), atol=_rtol())

    def test_the_phase_across_the_cell_is_the_taylor_series(self):
        args = make_inputs(n_int_freq=1)
        rfi_phase, rfi_delay, dt, freqs_mhz = args[1], args[2], args[8], args[9]
        got = fine_phase(rfi_phase[..., 2], rfi_delay[:, :, 2], freqs_mhz, args[7], dt)
        tau = rfi_delay[:, :, 2]
        d_tau = tau[..., 1, None] * dt + tau[..., 2, None] * dt**2 / 2 + tau[..., 3, None] * dt**3 / 6
        # MHz times microseconds is cycles
        expected = rfi_phase[..., 2][..., None, None] + 2 * jnp.pi * freqs_mhz[None, None, :, None, None] * d_tau[:, :, None, None, :]
        np.testing.assert_allclose(got, expected, rtol=_rtol(), atol=_rtol())

    def test_the_phase_across_the_channel_is_linear_in_frequency(self):
        args = make_inputs(n_int_time=1, n_int_freq=3)
        rfi_phase, rfi_delay, dnu_mhz, freqs_mhz = args[1], args[2], args[7], args[9]
        got = fine_phase(rfi_phase[..., 0], rfi_delay[:, :, 0], freqs_mhz, dnu_mhz, args[8])
        # no change across the cell (dt = 0): only the channel's own slope, 2 pi tau per MHz
        slope = 2 * jnp.pi * rfi_delay[:, :, 0, 0]  # (n_rfi, n_ant), per MHz
        expected = rfi_phase[..., 0][..., None, None] + slope[:, :, None, None, None] * dnu_mhz[None, None, None, :, None]
        np.testing.assert_allclose(got, expected, rtol=_rtol(), atol=_rtol())

    def test_cell_vis_is_the_summed_product_averaged_over_the_cell(self):
        args = make_inputs()
        S = jnp.exp(1.0j * fine_phase(args[1][..., 1], args[2][:, :, 1], args[9], args[7], args[8]))
        a1, a2 = args[10], args[11]
        expected = jnp.mean(jnp.sum(S[:, a1] * jnp.conj(S[:, a2]), axis=0), axis=(-2, -1))
        np.testing.assert_allclose(cell_vis(S, a1, a2), expected, rtol=_rtol(), atol=_rtol())


@pytest.mark.requires_double
class TestDerivatives:
    """Only rfi_A is differentiated; the reference for a kernel's JVP and VJP."""

    def test_jvp_and_vjp_against_finite_differences(self):
        args = make_inputs()
        f = lambda rfi_A: coarse_rfi_vis(rfi_A, *args[1:])
        check_grads(f, (args[0],), order=1, modes=("fwd", "rev"), atol=1e-6, rtol=1e-6)

    def test_the_jvp_is_the_bilinear_form(self):
        """d vis = B(dS, S) + B(S, dS), with dS the tangent pushed through the
        same interpolation and phase factor: the structure a kernel's JVP has."""
        args = make_inputs(n_int_freq=1)
        rfi_A = args[0]
        tangent = jnp.asarray(np.random.default_rng(1).normal(size=rfi_A.shape) + 0.0j)
        _, got = jax.jvp(lambda A: coarse_rfi_vis(A, *args[1:]), (rfi_A,), (tangent,))
        vis = lambda A, B: _bilinear(A, B, args)
        expected = vis(tangent, rfi_A) + vis(rfi_A, tangent)
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)

    def test_the_vjp_reaches_the_data_grid_and_nothing_else(self):
        args = make_inputs()
        f = lambda rfi_A: coarse_rfi_vis(rfi_A, *args[1:])
        vis, vjp = jax.vjp(f, args[0])
        (cot,) = vjp(jnp.ones_like(vis))
        assert cot.shape == args[0].shape
        assert bool(jnp.all(jnp.isfinite(cot)))
        assert float(jnp.abs(cot).max()) > 0


def _bilinear(A, B, args):
    """mean sum_r S_A[a1] conj(S_B[a2]) with S_A, S_B built from A and B."""
    rfi_phase, rfi_delay, w_freq, start_freq, w_time, start_time, dnu, dt, freqs, a1, a2 = args[1:]
    n_time = A.shape[-1]
    out = []
    for t in range(n_time):
        phase = fine_phase(rfi_phase[..., t], rfi_delay[:, :, t], freqs, dnu, dt)
        S_A = fine_signal(A, w_freq, start_freq, w_time[t], start_time[t]) * jnp.exp(1.0j * phase)
        S_B = fine_signal(B, w_freq, start_freq, w_time[t], start_time[t]) * jnp.exp(1.0j * phase)
        out.append(jnp.mean(jnp.sum(S_A[:, a1] * jnp.conj(S_B[:, a2]), axis=0), axis=(-2, -1)))
    return jnp.stack(out, -1)
