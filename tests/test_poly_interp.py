"""Tests for tabascal.poly_interp: the setup-time tables of the data-grid route."""

from math import factorial

import numpy as np
import pytest

from tabascal.fft_gp import domain_ss
from tabascal.poly_interp import fine_offsets, fit_path, interp_tables, lagrange_basis


def _offsets(n_int, spacing=1.0):
    """Fine offsets as TabConfig lays them out: (v - n_int // 2) * spacing / n_int."""
    return (np.arange(n_int) - n_int // 2) * spacing / n_int


class TestLagrangeBasis:

    def test_is_the_identity_at_the_nodes(self):
        nodes = np.arange(5.0)
        np.testing.assert_allclose(lagrange_basis(nodes, nodes), np.eye(5), atol=1e-14)

    def test_sums_to_one_everywhere(self):
        x = np.linspace(-1.0, 5.0, 37)
        np.testing.assert_allclose(lagrange_basis(np.arange(4.0), x).sum(axis=0), 1.0, atol=1e-13)

    @pytest.mark.parametrize("n_nodes", [1, 2, 3, 5])
    def test_reproduces_every_polynomial_of_the_stencil_degree(self, n_nodes):
        nodes = np.arange(n_nodes, dtype=float)
        x = np.linspace(-0.7, n_nodes - 0.3, 23)
        coeffs = np.arange(1, n_nodes + 1, dtype=float)  # degree n_nodes - 1
        poly = np.polynomial.Polynomial(coeffs)
        np.testing.assert_allclose(poly(nodes) @ lagrange_basis(nodes, x), poly(x), atol=1e-11)


class TestInterpTables:

    def test_shapes_and_starts(self):
        weights, start = interp_tables(7, 1, _offsets(4))
        assert weights.shape == (7, 3, 4)
        assert start.dtype == np.int32
        # centred away from the edges, shifted inwards at them
        np.testing.assert_array_equal(start, [0, 0, 1, 2, 3, 4, 4])

    def test_interior_cells_share_one_table(self):
        weights, _ = interp_tables(9, 2, _offsets(6))
        for c in range(2, 7):
            np.testing.assert_array_equal(weights[c], weights[2])

    @pytest.mark.parametrize("half_width", [0, 1, 2])
    @pytest.mark.parametrize("n_int", [1, 4, 5])
    def test_reproduces_a_polynomial_of_the_stencil_degree_everywhere(self, half_width, n_int):
        n_cells = 8
        offsets = _offsets(n_int)
        weights, start = interp_tables(n_cells, half_width, offsets)
        coeffs = 0.5 ** np.arange(2 * half_width + 1)
        poly = np.polynomial.Polynomial(coeffs)
        coarse = poly(np.arange(n_cells, dtype=float))
        stencil = coarse[start[:, None] + np.arange(2 * half_width + 1)]
        fine = np.einsum("ckv,ck->cv", weights, stencil)
        expected = poly(np.arange(n_cells)[:, None] + offsets[None, :])
        # the edge cells included: their stencil is shifted, not shortened
        np.testing.assert_allclose(fine, expected, atol=1e-12)

    def test_one_sample_per_cell_is_the_identity(self):
        weights, start = interp_tables(6, 1, np.zeros(1))
        fine = np.einsum("ckv,ck->cv", weights, np.arange(6.0)[start[:, None] + np.arange(3)])
        np.testing.assert_allclose(fine[:, 0], np.arange(6.0), atol=1e-14)

    def test_a_short_axis_takes_the_widest_stencil_it_holds(self):
        weights, start = interp_tables(2, 3, _offsets(3))
        assert weights.shape == (2, 1, 3)  # one cell: its value across it
        np.testing.assert_array_equal(weights, 1.0)
        np.testing.assert_array_equal(start, [0, 1])
        weights, _ = interp_tables(4, 3, _offsets(3))
        assert weights.shape == (4, 3, 3)  # 4 cells hold a 3-point stencil, not 7

    def test_no_cells_is_refused(self):
        with pytest.raises(ValueError, match="at least one cell"):
            interp_tables(0, 1, _offsets(3))


class TestFitPath:

    def test_recovers_the_derivatives_of_a_cubic(self):
        offsets = _offsets(7, spacing=2.0)
        coeffs = np.array([1.0e6, 7.0e3, 5.0, 0.3])  # m, m/s, m/s^2, m/s^3
        path = sum(c * offsets**k / factorial(k) for k, c in enumerate(coeffs))
        got = fit_path(np.tile(path, (2, 3, 1)), offsets, 3)
        assert got.shape == (2, 3, 4)
        # To ~1e-7 of the constant term: the fit is a float64 least squares
        # of a path of 1e6 m, and the derivatives inherit that scale's round-off.
        np.testing.assert_allclose(got, np.broadcast_to(coeffs, got.shape), rtol=1e-6, atol=1e-6)

    def test_the_degree_is_capped_by_the_sample_count(self):
        offsets = _offsets(2)
        got = fit_path(np.ones((5, 2)), offsets, 3)
        assert got.shape == (5, 2)  # a line through two points
        got = fit_path(np.full((5, 1), 3.0), np.zeros(1), 3)
        assert got.shape == (5, 1)  # one sample: the value and nothing else
        np.testing.assert_array_equal(got[:, 0], 3.0)

    def test_the_constant_term_is_the_value_at_the_centre(self):
        offsets = _offsets(5)
        path = 2.0 + offsets + 0.1 * offsets**2
        np.testing.assert_allclose(fit_path(path[None], offsets, 2)[0, 0], 2.0, atol=1e-12)


class TestFineOffsets:

    def test_is_the_layout_tabconfig_builds(self):
        """The offsets are what domain_ss puts the fine samples at: n_int per
        cell in cell order, the data-grid sample at index n_int // 2."""
        for n_cells, n_int in [(6, 1), (6, 4), (6, 5), (2, 3), (7, 2)]:
            (unit,) = domain_ss([n_cells], [1.0], [0.0], [n_int], [2.0])
            unit = np.asarray(unit, dtype=np.float64)
            expected = (np.arange(n_cells)[:, None] + fine_offsets(n_int, 1.0)).ravel()
            np.testing.assert_allclose(unit, expected, atol=1e-6)

    def test_scales_with_the_spacing(self):
        np.testing.assert_allclose(fine_offsets(4, 8.0), [-4.0, -2.0, 0.0, 2.0])
        np.testing.assert_allclose(fine_offsets(5, 2.0), [-0.8, -0.4, 0.0, 0.4, 0.8])

    def test_one_sample_per_cell_gives_a_zero_offset(self):
        np.testing.assert_array_equal(fine_offsets(1, 8.0), [0.0])
