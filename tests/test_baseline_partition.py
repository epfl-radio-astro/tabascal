"""Coverage and ordering contracts for the optional host-side grouping plan."""
import numpy as np
import pytest

from tabascal.baseline_partition import optimal_four_worker_partition


@pytest.mark.parametrize("n", [16, 17, 32, 40, 512])
def test_complete_balanced_round_trip(n):
    a1, a2 = np.triu_indices(n, 1)
    plan = optimal_four_worker_partition(a1, a2, n_devices=4)
    np.testing.assert_array_equal(np.sort(plan.baseline_indices.ravel()), np.arange(len(a1)))
    assert plan.baseline_indices.shape == (4, len(a1) // 4)
    assert plan.antennas.shape == (4, (3 * n + 4) // 5)
    for endpoints, local in [(a1, plan.a1), (a2, plan.a2)]:
        packed = np.take_along_axis(plan.antennas, local, axis=1)
        np.testing.assert_array_equal(packed.ravel()[plan.inverse_indices], endpoints)
    assert np.all(plan.antenna_mask[np.arange(4)[:, None], plan.a1])
    assert np.all(plan.antenna_mask[np.arange(4)[:, None], plan.a2])
    if n == 512:
        np.testing.assert_array_equal(plan.antenna_mask.sum(axis=1), [306, 308, 308, 308])


def test_deterministic_under_row_permutation_and_reversed_orientation():
    a1, a2 = np.triu_indices(512, 1)
    a1, a2 = a1 * 7 + 11, a2 * 7 + 11
    reference = optimal_four_worker_partition(a1, a2, n_devices=4)
    order = np.random.default_rng(42).permutation(len(a1))
    b1, b2 = a1[order].copy(), a2[order].copy()
    b1[::2], b2[::2] = b2[::2].copy(), b1[::2].copy()
    plan = optimal_four_worker_partition(b1, b2, n_devices=4)
    np.testing.assert_array_equal(order[plan.baseline_indices], reference.baseline_indices)
    np.testing.assert_array_equal(plan.antennas, reference.antennas)
    for endpoints, local in [(b1, plan.a1), (b2, plan.a2)]:
        np.testing.assert_array_equal(np.take_along_axis(plan.antennas, local, axis=1), endpoints[plan.baseline_indices])


@pytest.mark.parametrize("devices", [1, 2, 8, True, 4.0])
def test_unsupported_device_counts(devices):
    a1, a2 = np.triu_indices(32, 1)
    with pytest.raises(ValueError, match="four baseline devices"):
        optimal_four_worker_partition(a1, a2, n_devices=devices)


def test_invalid_graphs():
    a1, a2 = np.triu_indices(32, 1)
    for x, y in [(a1[:-1], a2[:-1]), (a1, a1), (a1.astype(float), a2), (-a1, a2)]:
        with pytest.raises(ValueError):
            optimal_four_worker_partition(x, y, n_devices=4)
    a1[0], a2[0] = a2[1], a1[1]
    with pytest.raises(ValueError, match="duplicate"):
        optimal_four_worker_partition(a1, a2, n_devices=4)


@pytest.mark.parametrize("n", [5, 8, 18, 33])
def test_unsupported_sizes(n):
    a1, a2 = np.triu_indices(n, 1)
    with pytest.raises(ValueError, match="balanced optimal"):
        optimal_four_worker_partition(a1, a2, n_devices=4)
