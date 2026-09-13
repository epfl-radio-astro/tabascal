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
- :func:`poly_time_groups`: the baseline partition and compact antenna maps
  that decide which independent time tables each variable call needs.
"""

from dataclasses import dataclass
from math import factorial

import numpy as np
from numpy.typing import NDArray


def poly_sample_counts(requirements: NDArray) -> NDArray:
    """Round fringe-rate requirements up to positive odd quadrature counts.

    The estimate bounds midpoint quadrature error. With ``fine_offsets`` an
    even count instead shifts the grid half a sample to the left, introducing
    a first-order phase error that rounding up alone does not control. Odd
    counts keep that convention and sample the midpoints of equal sub-intervals.
    """
    requirements = np.asarray(requirements, dtype=np.float64)
    if (
        requirements.ndim != 1
        or not np.all(np.isfinite(requirements))
        or np.any(requirements < 0)
    ):
        raise ValueError("sampling requirements must be a finite, non-negative vector")
    counts = np.maximum(1, np.ceil(requirements)).astype(np.int64)
    return counts + (counts % 2 == 0)


@dataclass(frozen=True)
class PolyTimeGroup:
    """Host-side baseline membership and compact antenna indices for one call."""

    baseline_indices: NDArray
    n_g: int
    antennas: NDArray
    a1: NDArray
    a2: NDArray


def make_poly_time_group(requirements, a1, a2, indices) -> PolyTimeGroup:
    """Materialise a chosen partition with the same compact maps on every route."""
    a1, a2 = np.asarray(a1), np.asarray(a2)
    idx = np.sort(indices).astype(np.int32)
    antennas = np.unique(np.concatenate((a1[idx], a2[idx]))).astype(np.int32)
    return PolyTimeGroup(
        idx, int(poly_sample_counts(np.asarray(requirements)[idx]).max()), antennas,
        np.searchsorted(antennas, a1[idx]).astype(np.int32),
        np.searchsorted(antennas, a2[idx]).astype(np.int32),
    )


def split_group_over_devices(group: PolyTimeGroup, n_dev: int) -> tuple[PolyTimeGroup, ...]:
    """Divide one group's baselines evenly over devices, keeping its antennas.

    Each device computes a share of the group's baselines and none of anyone
    else's, so nothing has to be summed across devices -- unlike sharding the
    source axis, where every device computes every baseline for a few sources
    and the partial visibilities must be added back together.

    The antenna set is deliberately *not* recompacted per device, which is what
    separates this from :func:`make_poly_time_group`. Every device keeps the
    whole group's antennas, so the per-antenna signal is identical on all of
    them and enters the map replicated; recompacting would give each device a
    different antenna count, and ``shard_map`` runs one program with one set of
    shapes. The signal is small -- 0.24 GB at 512 stations against 1.256 GB for
    a single visibility array -- so replicating it costs far less than the
    visibilities it lets us divide.

    ``a1`` and ``a2`` stay indices into the group's compact antenna axis, and
    ``baseline_indices`` stay indices into the *global* visibility array, so a
    device knows where its own results belong.

    Raises when the count does not divide: ``shard_map`` needs one shape for
    every device, and a silent remainder would drop baselines.
    """
    if isinstance(n_dev, bool) or not isinstance(n_dev, (int, np.integer)) or n_dev < 1:
        raise ValueError("n_dev must be a positive whole number")
    n_bl = len(group.baseline_indices)
    if n_bl % n_dev:
        raise ValueError(
            f"cannot split {n_bl} baselines evenly over {n_dev} devices "
            f"({n_bl % n_dev} left over). Every device must take the same "
            "number for shard_map to run one program over them."
        )
    per = n_bl // n_dev
    return tuple(
        PolyTimeGroup(
            group.baseline_indices[d * per:(d + 1) * per],
            group.n_g,
            group.antennas,
            group.a1[d * per:(d + 1) * per],
            group.a2[d * per:(d + 1) * per],
        )
        for d in range(n_dev)
    )


@dataclass(frozen=True)
class DeviceGroup:
    """One device's share of a group, shaped identically on every device.

    ``a1``/``a2`` index the group's antenna axis extended by
    ``n_ghost_antennas`` dark antennas, the same count on every device.
    ``n_real`` output rows are real baselines; ghosts scatter to a spare row
    that is discarded before returning the visibility array. ``positions`` says where the real rows belong in the device's own
    block of it.
    """

    positions: NDArray
    a1: NDArray
    a2: NDArray
    n_real: int
    n_ghost_antennas: int = 0

    @property
    def n_padded(self) -> int:
        """Rows the operator is asked for, real and ghost, the same on every device."""
        return len(self.a1)


def device_groups(
    group: PolyTimeGroup, n_dev: int, n_bl_total: int, offset: int = 0,
) -> tuple[DeviceGroup, ...]:
    """Split a group by which device owns each baseline, padding to one shape.

    With ``block = ceil(n_bl_total / n_dev)``, device ``d`` owns the
    contiguous range ``[d*block, (d+1)*block)`` of the
    visibility array in the order the data already has, so nothing is
    reordered: the observed visibilities, the flags and the noise are simply
    sharded along the axis they already have, and every array downstream keeps
    that split all the way to the likelihood. A device therefore never needs
    anyone else's visibility rows. Likelihood scalars and reverse-mode
    gradients of shared per-antenna inputs still need reductions. For
    indivisible totals the last block extends past the real array; the caller
    trims those rows and uses replicated placement at component boundaries.

    Which of a group's baselines land on a device is then whatever the data
    ordering puts there, so the counts differ between devices. ``shard_map``
    runs one program, so they are padded up to a common count with dark ghost
    baselines: distinct pairs against extra antennas carrying no signal,
    which collide with no real baseline, stay distinct from each other, and
    contribute neither visibility nor gradient -- the same device
    :func:`~tabascal.distributed.padded_rfi_count` uses on the source axis.

    ``positions`` are local to the owning device's block, so a group writes
    into the device's own rows, and are padded to the same length as ``a1``
    with ``block`` -- one past the block's last row. The caller gives its local
    visibility array that one extra row, lets the ghosts land in it and drops
    it, so the scatter has a fixed shape and a ghost can never overwrite a real
    baseline. ``offset`` is where this group's share starts within that block,
    since several groups share it.
    """
    if isinstance(n_dev, bool) or not isinstance(n_dev, (int, np.integer)) or n_dev < 1:
        raise ValueError("n_dev must be a positive whole number")
    if n_bl_total < 1:
        raise ValueError("n_bl_total must be positive")
    block = (n_bl_total + n_dev - 1) // n_dev
    idx = np.asarray(group.baseline_indices)
    owner = idx // block
    shares = [idx[owner == d] for d in range(n_dev)]
    mine_a1 = [np.asarray(group.a1)[owner == d] for d in range(n_dev)]
    mine_a2 = [np.asarray(group.a2)[owner == d] for d in range(n_dev)]

    per = max((len(s) for s in shares), default=0)
    ghost = len(group.antennas)
    max_pad = per - min((len(s) for s in shares), default=0)
    # Each extra dark antenna supplies `ghost` distinct real/dark pairs.
    # Size this from ownership imbalance, which can be much larger than n_ant.
    n_ghost = (max_pad + ghost - 1) // ghost if max_pad else 0

    out = []
    for d in range(n_dev):
        real = len(shares[d])
        pad = per - real
        ghost_rows = np.arange(pad)
        a1 = np.concatenate((
            mine_a1[d], (ghost_rows % ghost).astype(np.asarray(group.a1).dtype),
        ))
        a2 = np.concatenate((
            mine_a2[d], (ghost + ghost_rows // ghost).astype(np.asarray(group.a2).dtype),
        ))
        # Ghost rows are scattered to `block`, the spare row the caller adds
        # and discards, so every device scatters the same number of rows.
        rows = (shares[d] - d * block).astype(np.int32) + offset
        out.append(DeviceGroup(
            np.concatenate((rows, np.full(pad, block, dtype=np.int32))), a1, a2, real, n_ghost,
        ))
    return tuple(out)


def poly_time_groups(
    requirements: NDArray, a1: NDArray, a2: NDArray,
    *, max_groups: int = 2, split_at: int | None = None,
) -> tuple[PolyTimeGroup, ...]:
    """Choose one or two groups by the work of materialising antenna samples.

    Each group's count is the largest rounded requirement of its baselines.
    Search every threshold between distinct requirements, scoring a partition by
    ``sum(n_g * len(antennas_g))``. The source, channel and cell dimensions
    multiply every candidate equally, so they need not enter the score. An
    antenna shared by the two groups is materialised twice and counted twice.
    This is work, not a runtime prediction for either implementation.

    ``split_at`` restricts the search to requirements at or below that threshold
    versus requirements above it. An empty side or a split that does not strictly
    improve on one group falls back to one group; ties favour fewer groups.
    Equal two-group scores choose the lowest threshold deterministically.

    Sorting once and accumulating prefix/suffix antenna incidence gives all
    candidate antenna counts without constructing a baseline mask per split.
    Only the winning partition is materialised into static group records.
    """
    if (
        isinstance(max_groups, bool)
        or not isinstance(max_groups, (int, np.integer))
        or max_groups < 1
    ):
        raise ValueError("rfi.poly_time_sampling.max_groups must be a positive whole number")
    if split_at is not None and (
        isinstance(split_at, bool) or not isinstance(split_at, (int, np.integer)) or split_at < 1
    ):
        raise ValueError("rfi.poly_time_sampling.split_at must be null or a positive whole count")
    counts = poly_sample_counts(requirements)
    a1, a2 = np.asarray(a1), np.asarray(a2)
    if any(
        a.shape != counts.shape
        or not np.issubdtype(a.dtype, np.integer)
        or np.any(a < 0)
        for a in (a1, a2)
    ):
        raise ValueError("a1 and a2 must be non-negative integer vectors matching requirements")
    if not len(counts):
        return ()

    # Compress the labels for incidence storage too: an MS may omit antennas,
    # and the largest label need not describe the size of the observed array.
    antennas, inverse = np.unique(np.concatenate((a1, a2)), return_inverse=True)
    endpoints = np.stack((inverse[:len(a1)], inverse[len(a1):]), axis=1)
    # Distinct requirements can round to the same odd count. Their boundary
    # still matters: moving a baseline can add antennas to one group without
    # removing them from the other, even though the low group's count stays.
    requirements = np.asarray(requirements, dtype=np.float64)
    order = np.argsort(requirements, kind="stable")
    sorted_requirements = requirements[order]
    sorted_counts = counts[order]
    best_score = int(sorted_counts[-1]) * len(antennas)
    best_cut = None
    if max_groups == 2:
        def incidence(sequence):
            seen = np.zeros(len(antennas), dtype=bool)
            totals = np.empty(len(counts), dtype=np.int64)
            total = 0
            for i, b in enumerate(sequence):
                for ant in endpoints[b]:
                    if not seen[ant]:
                        seen[ant] = True
                        total += 1
                totals[i] = total
            return totals

        prefix = incidence(order)
        suffix = incidence(order[::-1])[::-1]
        cuts = np.flatnonzero(sorted_requirements[:-1] != sorted_requirements[1:]) + 1
        if split_at is not None:
            cuts = [int(np.searchsorted(sorted_requirements, split_at, side="right"))]
        for cut in cuts:
            if cut == 0 or cut == len(counts):
                continue
            score = (
                int(sorted_counts[cut - 1]) * int(prefix[cut - 1])
                + int(sorted_counts[-1]) * int(suffix[cut])
            )
            if score < best_score:
                best_score, best_cut = score, cut

    if max_groups > 2:
        # More than two groups are cut at equal baseline counts rather than
        # searched. Products pay the MEAN of the groups' resolutions and
        # materialising pays their SUM, so the penalty grows linearly in the
        # group count while the saving approaches the per-baseline ideal, and
        # the two only meet at n_bl / n_ant groups -- 255 at 512 stations.
        # Where to stop is a question for measurement, not for this search.
        edges = [int(round(i * len(counts) / max_groups)) for i in range(max_groups + 1)]
        partitions = tuple(order[lo:hi] for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo)
    else:
        partitions = (
            (np.arange(len(counts)),) if best_cut is None
            else (order[:best_cut], order[best_cut:])
        )
    return tuple(make_poly_time_group(requirements, a1, a2, idx) for idx in partitions)


def monomial_tables(n_cells: int, half_width: int) -> tuple[NDArray, NDArray]:
    """Lagrange coefficients in x = 2*tau/T, including the shifted edge stencils.

    These occupy the time-table slot of ``fine_signal``: the identical stencil
    contraction now yields polynomial coefficients rather than sampled values.
    No Vandermonde fit is needed; multiplying each basis's linear factors gives
    its monomials directly on the host in double precision.
    """
    _, starts = interp_tables(n_cells, half_width, np.zeros(1))
    degree = 2 * min(half_width, (n_cells - 1) // 2)
    coefficients = np.zeros((n_cells, degree + 1, degree + 1))
    for cell, start in enumerate(starts):
        nodes = 2 * (np.arange(degree + 1) + start - cell)
        for k, node in enumerate(nodes):
            polynomial = np.polynomial.Polynomial([1.])
            for j, other in enumerate(nodes):
                if j != k:
                    polynomial *= np.polynomial.Polynomial([-other, 1.]) / (node - other)
            coefficients[cell, k, :len(polynomial.coef)] = polynomial.coef
    return coefficients, starts


def analytic_sampling_cut(half_width: int, segments: int, terms: int, cubic_terms: int = 3) -> int:
    """A conservative operation-count crossover, in quadrature samples.

    For product degree P=4h and J cubic terms, moments reach D=P+3(J-1),
    and the linear recurrence reaches M=D+2(K-1). Per piece it takes M upward
    and M+64 downward stages, K(D+1) curvature and J(P+1) cubic contractions,
    and (2h+1)^2 amplitude products. Counting each as one sample
    is deliberately conservative: a quadrature sample also interpolates and
    exponentiates. This is an arithmetic diagnostic, not a runtime model. The hybrid
    components use measured precision-dependent VJP crossovers instead.
    """
    product_degree = 4 * half_width
    n_cubic = max(1, cubic_terms)
    degree = product_degree + 3 * (n_cubic - 1)
    moment_degree = degree + 2 * (terms - 1)
    return segments * (
        2 * moment_degree + 64 + terms * (degree + 1)
        + n_cubic * (product_degree + 1) + (2 * half_width + 1)**2
    )


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
