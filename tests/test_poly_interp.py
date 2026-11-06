"""The setup-time tables of the analytic RFI route (tabascal.poly_interp)."""

from math import factorial
from types import SimpleNamespace

import numpy as np
import pytest

from tabascal.config import TabConfig
from tabascal.poly_interp import fine_offsets, fit_path, interp_tables, lagrange_basis, monomial_tables


def test_the_lagrange_basis_is_the_identity_at_its_nodes_and_sums_to_one():
    nodes = np.arange(5.0)
    np.testing.assert_allclose(lagrange_basis(nodes, nodes), np.eye(5), atol=1e-14)
    np.testing.assert_allclose(lagrange_basis(nodes, np.linspace(-1.0, 6.0, 37)).sum(axis=0), 1.0, atol=1e-12)


@pytest.mark.parametrize("half_width", [0, 1, 2])
@pytest.mark.parametrize("n_int", [1, 4, 5])
def test_interp_tables_reproduce_a_polynomial_of_the_stencil_degree(half_width, n_int):
    """Everywhere, the edge cells included: their stencil is shifted, not shortened."""
    n_cells, offsets = 8, fine_offsets(n_int, 1.0)
    weights, start = interp_tables(n_cells, half_width, offsets)
    assert weights.shape == (n_cells, 2 * half_width + 1, n_int) and start.dtype == np.int32
    poly = np.polynomial.Polynomial(0.5 ** np.arange(2 * half_width + 1))
    stencil = poly(np.arange(n_cells, dtype=float))[start[:, None] + np.arange(2 * half_width + 1)]
    expected = poly(np.arange(n_cells)[:, None] + offsets)
    np.testing.assert_allclose(np.einsum("ckv,ck->cv", weights, stencil), expected, atol=1e-12)


def test_a_short_axis_takes_the_widest_stencil_it_holds():
    assert interp_tables(2, 2, fine_offsets(3, 1.0))[0].shape == (2, 1, 3)
    assert interp_tables(4, 2, fine_offsets(3, 1.0))[0].shape == (4, 3, 3)


def test_fit_path_recovers_the_derivatives_of_a_cubic():
    offsets = fine_offsets(7, 2.0)
    coeffs = np.array([1.0e6, 7.0e3, 5.0, 0.3])  # m, m/s, m/s^2, m/s^3
    path = sum(c * offsets**k / factorial(k) for k, c in enumerate(coeffs))
    got = fit_path(np.tile(path, (2, 3, 1)), offsets, 3)
    # float64 least squares of a 1e6 m path: its round-off, not the fit, sets rtol.
    np.testing.assert_allclose(got, np.broadcast_to(coeffs, (2, 3, 4)), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("n_int_freq, n_int_time", [(1, 1), (3, 4), (2, 5)])
def test_fine_offsets_are_where_tabconfig_puts_the_fine_grid(n_int_freq, n_int_time):
    n_freq, n_time = 3, 6
    cfg = SimpleNamespace(
        n_freq=n_freq, n_time=n_time, n_int_freq=n_int_freq, n_int_time=n_int_time,
        freqs=1.4e9 + 1e6 * np.arange(n_freq), chan_width=1e6,
        times=2.0 * np.arange(n_time), int_time=2.0,
        times_jd=2460000.5 + 2.0 * np.arange(n_time) / 86400.0,
        args={"rfi": {"freq_pad_factor": 2, "time_pad_factor": 2}},
    )
    TabConfig._set_freqs_times(cfg)
    # The unit grid is built in the run's precision: float32 resolves 1e-6 of a cell here.
    np.testing.assert_allclose(cfg.times_fine, (cfg.times[:, None] + fine_offsets(n_int_time, 2.0)).ravel(), rtol=0, atol=2e-6)
    np.testing.assert_allclose(cfg.freqs_fine, (cfg.freqs[:, None] + fine_offsets(n_int_freq, 1e6)).ravel(), rtol=0, atol=1.0)


@pytest.mark.parametrize("half_width", [0, 1, 2])
def test_monomial_tables_are_the_interp_tables_in_x(half_width):
    """Coefficients in x = 2 dt / T of the same Lagrange basis, edge stencils included."""
    n_time = 2 * half_width + 3
    g, start = monomial_tables(n_time, half_width)
    x = np.array([-0.91, -0.4, 0.03, 0.7, 0.93])
    w, expected_start = interp_tables(n_time, half_width, x / 2)
    np.testing.assert_array_equal(start, expected_start)
    np.testing.assert_allclose(np.einsum("tlm,mv->tlv", g, x ** np.arange(g.shape[-1])[:, None]), w, atol=2e-14)
