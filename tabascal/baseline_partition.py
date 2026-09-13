"""Deterministic host-side antenna reuse plans; not enabled in the runner."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BaselinePartition:
    """Packed rows map to input rows; inverse_indices restores flattened output.

    Antennas are global IDs padded with -1: gather only where antenna_mask is
    true, and zero the remaining slots. a1/a2 address these local tables and
    preserve the input orientation (no visibility conjugation is required).
    """
    baseline_indices: np.ndarray
    inverse_indices: np.ndarray
    antennas: np.ndarray
    antenna_mask: np.ndarray
    a1: np.ndarray
    a2: np.ndarray


def optimal_four_worker_partition(a1, a2, *, n_devices: int) -> BaselinePartition:
    """Partition a complete cross-correlation graph equally over four workers.

    Pass the configured baseline mesh width, not total source×baseline devices.
    Accept only sizes attaining the ceil(3*n/5) common antenna-width lower bound
    with zero ghost baselines. Unsupported sizes/topologies raise explicitly.
    Sorted antenna IDs, lexicographic pairs and fixed remainder assignment make
    ownership independent of input row order and orientation. This is a setup
    planner; callers must apply its permutation consistently to baseline state.
    """
    if isinstance(n_devices, (bool, np.bool_)) or not isinstance(n_devices, (int, np.integer)) or n_devices != 4:
        raise ValueError("this optimal construction requires four baseline devices")
    a1, a2 = np.asarray(a1), np.asarray(a2)
    if (a1.ndim != 1 or a2.shape != a1.shape
            or a1.dtype.kind not in "iu" or a2.dtype.kind not in "iu"
            or np.any(a1 < 0) or np.any(a2 < 0)):
        raise ValueError("endpoints must be equally sized non-negative integer vectors")
    if np.any(a1 > np.iinfo(np.int64).max) or np.any(a2 > np.iinfo(np.int64).max):
        raise ValueError("antenna IDs must fit signed int64")
    a1, a2 = a1.astype(np.int64), a2.astype(np.int64)
    ids = np.union1d(a1, a2)
    n = len(ids)
    if n < 5 or len(a1) != n * (n - 1) // 2 or np.any(a1 == a2):
        raise ValueError("a complete cross-correlation graph without autocorrelations is required")
    x, y = np.searchsorted(ids, a1), np.searchsorted(ids, a2)
    keys = np.minimum(x, y) * n + np.maximum(x, y)
    if len(np.unique(keys)) != len(keys):
        raise ValueError("duplicate baseline, including reversed duplicates")
    s = n // 5
    d = n - 3 * s
    if len(a1) % 4 or max(3 * s, s + d) != (3 * n + 4) // 5:
        raise ValueError("this antenna count does not admit the balanced optimal construction")
    capacity = len(a1) // 4
    extra = capacity - 3 * s * s
    inner = s * (s - 1) // 2
    take = np.array([extra // 3 + (i < extra % 3) for i in range(3)])
    dd_take = capacity - s * d - (inner - take)
    if extra < 0 or np.any(take > inner) or np.any(dd_take < 0):
        raise ValueError("this antenna count does not admit the balanced optimal construction")
    groups = [np.arange(i * s, (i + 1) * s) for i in range(3)] + [np.arange(3 * s, n)]

    def within(g):
        i, j = np.triu_indices(len(g), 1)
        return g[i] * n + g[j]

    def cross(g, h):
        return (g[:, None] * n + h[None, :]).ravel()

    *inside, dd = [within(g) for g in groups]
    worker_keys = [np.concatenate([
        cross(groups[0], groups[1]), cross(groups[0], groups[2]),
        cross(groups[1], groups[2]),
        *[inside[i][:take[i]] for i in range(3)],
    ])]
    offset = 0
    for i in range(3):
        end = offset + int(dd_take[i])
        worker_keys.append(np.concatenate([
            cross(groups[i], groups[3]), inside[i][take[i]:], dd[offset:end],
        ]))
        offset = end
    assert offset == len(dd)
    order = np.argsort(keys)
    rows = np.stack([order[np.searchsorted(keys[order], k)] for k in worker_keys])
    width = (3 * n + 4) // 5
    antennas = np.full((4, width), -1, dtype=np.int64)
    local_a1, local_a2 = [], []
    mask = np.zeros((4, width), dtype=bool)
    for i, row in enumerate(rows):
        used = np.union1d(a1[row], a2[row])
        antennas[i, :len(used)] = used
        mask[i, :len(used)] = True
        local_a1.append(np.searchsorted(used, a1[row]))
        local_a2.append(np.searchsorted(used, a2[row]))
    return BaselinePartition(rows, np.argsort(rows.ravel()), antennas, mask,
                             np.stack(local_a1), np.stack(local_a2))
