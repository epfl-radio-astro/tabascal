"""Setup-time tables for the data-grid RFI route.

Everything here runs once, on the host, in float64 numpy, and produces the
*constant* inputs of :func:`tabascal.coarse_rfi_vis.coarse_rfi_vis`:

- :func:`interp_tables`: for each data cell on an axis, the weights that turn
  the ``2h + 1`` coarse samples nearest to it into its fine samples, and where
  that stencil starts. These are the polynomial (Lagrange) weights. The kernel
  takes the tables as data, so any other linear interpolant -- the conditional
  mean of a Gaussian process, a windowed sinc -- is the same kernel fed a
  different table.
- :func:`fit_path`: the polynomial in time through each cell's propagated delay
  samples, from whose coefficients the kernel rebuilds the fine phase.
- :func:`fine_offsets`: where a cell's fine samples sit relative to its own
  data-grid sample.
"""

from math import factorial

import numpy as np
from numpy.typing import NDArray


def lagrange_basis(nodes: NDArray, x: NDArray) -> NDArray:
    """Lagrange basis through ``nodes`` evaluated at ``x``, as ``(n_nodes, n_x)``.

    Row ``k`` is the polynomial of degree ``n_nodes - 1`` that is 1 at
    ``nodes[k]`` and 0 at every other node, so ``values @ basis`` is the
    interpolating polynomial through ``(nodes, values)`` sampled at ``x``.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    basis = np.ones((len(nodes), len(x)))
    for k, x_k in enumerate(nodes):
        for j, x_j in enumerate(nodes):
            if j != k:
                basis[k] *= (x - x_j) / (x_k - x_j)
    return basis


def interp_tables(
    n_cells: int, half_width: int, offsets: NDArray
) -> tuple[NDArray, NDArray]:
    """Per-cell interpolation weights and stencil starts for one axis.

    Parameters
    ----------
    n_cells : int
        Number of data cells on the axis.
    half_width : int
        Stencil half-width ``h``: ``2h + 1`` cells enter each cell's
        interpolant, which is the polynomial of degree ``2h`` through them.
        Reduced to what the axis can hold when it has fewer cells than that.
    offsets : Array (n_int,)
        Positions of a cell's fine samples relative to its centre, in units of
        the cell spacing (see :func:`fine_offsets`).

    Returns
    -------
    weights : Array (n_cells, n_stencil, n_int)
    start : Array (n_cells,) of int
        ``fine[c, v] = sum_k weights[c, k, v] * coarse[start[c] + k]``.

    Notes
    -----
    The stencil of cell ``c`` is the ``2h + 1`` cells nearest to it: centred on
    ``c`` away from the edges, and at the edges shifted inwards, where the
    polynomial through those same nearest cells is evaluated off-centre rather
    than a shorter one fitted. Away from the edges every row of ``weights`` is
    the same table; a kernel may exploit that, the reference does not.
    """
    if n_cells < 1:
        raise ValueError(f"interp_tables needs at least one cell, got {n_cells}")
    half_width = min(int(half_width), (n_cells - 1) // 2)
    n_stencil = 2 * half_width + 1
    offsets = np.asarray(offsets, dtype=np.float64)

    start = np.clip(np.arange(n_cells) - half_width, 0, n_cells - n_stencil)
    nodes = np.arange(n_stencil)
    weights = np.stack(
        [lagrange_basis(nodes, (c - s) + offsets) for c, s in enumerate(start)]
    )
    return weights, start.astype(np.int32)


def fit_path(path_fine: NDArray, offsets: NDArray, order: int) -> NDArray:
    """Least-squares polynomial through each cell's fine path samples.

    Parameters
    ----------
    path_fine : Array (..., n_time, n_int)
        The path -- or delay, or any quantity -- at the fine samples of each
        cell, in whatever unit the caller keeps it in.
    offsets : Array (n_int,)
        Offsets of the fine samples from the cell centre, in seconds.
    order : int
        Degree of the polynomial. Reduced to ``n_int - 1`` when a cell has
        fewer samples than a higher degree needs; at one sample per cell that
        is the path at the centre and nothing else.

    Returns
    -------
    Array (..., n_time, order + 1)
        Coefficients ``L_k`` with ``path(t_c + d) ~ sum_k L_k d^k / k!``: the
        quantity and its first ``order`` time derivatives at each cell centre,
        in its unit per second^k.
    """
    path_fine = np.asarray(path_fine, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    order = min(int(order), path_fine.shape[-1] - 1)
    ks = np.arange(order + 1)
    design = offsets[:, None] ** ks / np.array([factorial(k) for k in ks])
    return path_fine @ np.linalg.pinv(design).T


def fine_offsets(n_int: int, spacing: float) -> NDArray:
    """Offsets of a cell's fine samples from its data-grid sample, ``(n_int,)``.

    ``TabConfig`` lays a fine axis out with ``n_int`` samples per cell, in
    cell order, ``spacing / n_int`` apart, with the cell's own data-grid sample
    at index ``n_int // 2`` of its block (the crop in
    :func:`tabascal.fft_gp.latent_to_signal_init`), so the offsets are
    ``(v - n_int // 2) * spacing / n_int``. They are formed here from the count
    and the spacing rather than read off the config's fine grid: that grid is
    built from a unit grid in the run's precision, and in single precision its
    samples jitter by 1e-4 of a cell, which is metres of path at orbital speed.
    ``tests/test_poly_interp.py`` holds this to the layout that function makes.
    """
    return (np.arange(n_int) - n_int // 2) * (spacing / n_int)
