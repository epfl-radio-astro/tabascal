"""The host partition search and the quadrature accuracy its counts promise."""

import numpy as np
import pytest

from tabascal.poly_interp import fine_offsets, poly_sample_counts, poly_time_groups


def _score(groups):
    return sum(g.n_g * len(g.antennas) for g in groups)


def _brute_force(requirements, a1, a2):
    counts = poly_sample_counts(requirements)

    def score(partition):
        return sum(int(counts[idx].max()) * len(np.unique(np.r_[a1[idx], a2[idx]])) for idx in partition)

    best = [np.arange(len(counts))]
    for threshold in np.unique(requirements)[:-1]:
        candidate = [np.flatnonzero(requirements <= threshold), np.flatnonzero(requirements > threshold)]
        if score(candidate) < score(best):
            best = candidate
    return best, score(best)


class TestPolyTimeGroups:
    def test_exact_search_matches_exhaustive_partitions(self):
        rng = np.random.default_rng(45)
        for _ in range(100):
            # Repeated baselines, auto-correlations and missing antenna labels
            # also exercise incidence: an endpoint is present, however often
            # it occurs, and the order in the MS must survive the partition.
            a1, a2 = rng.choice([0, 2, 5, 8, 10], (2, 30))
            requirements = rng.choice([0, 2.1, 3, 8, 18, 43], 30)
            expected, score = _brute_force(requirements, a1, a2)
            groups = poly_time_groups(requirements, a1, a2)
            assert _score(groups) == score
            assert len(groups) == len(expected)
            for group, idx in zip(groups, expected):
                np.testing.assert_array_equal(group.baseline_indices, idx)
                np.testing.assert_array_equal(group.antennas[group.a1], a1[idx])
                np.testing.assert_array_equal(group.antennas[group.a2], a2[idx])
                assert np.all(np.diff(group.antennas) > 0)
                assert group.n_g >= requirements[idx].max()
                assert group.n_g % 2 == 1

    def test_antenna_overlap_can_make_a_baseline_saving_more_expensive(self):
        a1 = np.array([0, 0, 1])
        a2 = np.array([1, 2, 2])
        groups = poly_time_groups([3, 3, 5], a1, a2)
        # The tempting split saves baseline products (11 versus 15), but
        # materialises 3*3 + 5*2 = 19 antenna samples versus 5*3 = 15.
        assert len(groups) == 1
        assert _score(groups) == 15

    def test_a_useful_split_counts_shared_antennas_twice(self):
        groups = poly_time_groups([2, 10], np.array([0, 2]), np.array([2, 3]))
        assert [g.n_g for g in groups] == [3, 11]
        assert _score(groups) == 28 < 33
        np.testing.assert_array_equal(groups[0].antennas, [0, 2])
        np.testing.assert_array_equal(groups[1].antennas, [2, 3])

    def test_distinct_requirements_that_round_alike_still_offer_a_split(self):
        groups = poly_time_groups(
            [2, 9, 9, 3, 9, 3], np.array([1, 4, 1, 1, 2, 1]), np.array([0, 1, 3, 1, 2, 3]),
        )
        # Splitting after 2 costs 42; splitting after 3 or using one group
        # costs 45, although both 2 and 3 round to the same quadrature count.
        assert _score(groups) == 42
        np.testing.assert_array_equal(groups[0].baseline_indices, [0])

    def test_ties_choose_one_group(self):
        groups = poly_time_groups([3, 3, 9], np.array([0, 1, 0]), np.array([2, 2, 1]))
        assert len(groups) == 1  # 3*3 + 9*2 == 9*3

    def test_equal_two_group_scores_choose_the_lowest_threshold(self):
        groups = poly_time_groups([3, 5, 7], np.array([0, 2, 4]), np.array([1, 3, 5]))
        assert _score(groups) == 34
        np.testing.assert_array_equal(groups[0].baseline_indices, [0])

    @pytest.mark.parametrize("requirements", [[0, 0], [17, 17], [42.1, 43]])
    def test_equal_requirements_and_singleton_fallback(self, requirements):
        groups = poly_time_groups(requirements, np.array([1, 3]), np.array([2, 4]))
        assert len(groups) == 1
        assert groups[0].n_g == poly_sample_counts(requirements).max()

    def test_max_groups_and_explicit_threshold(self):
        requirements = [3, 11, 31]
        a1, a2 = np.array([0, 2, 4]), np.array([1, 3, 5])
        assert len(poly_time_groups(requirements, a1, a2, max_groups=1)) == 1
        auto = poly_time_groups(requirements, a1, a2)
        assert [g.n_g for g in auto] == [11, 31]
        fixed = poly_time_groups(requirements, a1, a2, split_at=3)
        assert [g.n_g for g in fixed] == [3, 31]
        for threshold in [1, 31, 100]:
            assert len(poly_time_groups(requirements, a1, a2, split_at=threshold)) == 1
        assert len(poly_time_groups(requirements, a1, a2, max_groups=1, split_at=3)) == 1

    def test_an_explicit_unhelpful_split_still_falls_back(self):
        assert len(poly_time_groups([3, 5], np.array([0, 0]), np.array([1, 1]), split_at=3)) == 1

    @pytest.mark.parametrize("options", [
        {"max_groups": n} for n in (0, 3, True, 2.0, None)
    ] + [{"split_at": n} for n in (0, -1, True, 2.5, np.nan, "auto")])
    def test_bad_options_are_refused(self, options):
        with pytest.raises(ValueError, match="rfi.poly_time_sampling"):
            poly_time_groups([3], np.array([0]), np.array([1]), **options)

    @pytest.mark.parametrize("requirements", [[-1], [np.inf], [np.nan], [[3]]])
    def test_bad_requirements_are_refused(self, requirements):
        with pytest.raises(ValueError, match="requirements"):
            poly_time_groups(requirements, np.array([0]), np.array([1]))

    def test_empty_baseline_set(self):
        assert poly_time_groups([], np.array([], dtype=int), np.array([], dtype=int)) == ()


class TestQuadratureAccuracy:
    @pytest.mark.parametrize("snr", [100, 10000])
    def test_rounded_group_counts_resolve_the_analytic_fringe_integral(self, snr):
        cycles = np.linspace(0.01, 8, 100)
        requirements = np.pi * cycles * np.sqrt(snr / 6)
        counts = poly_sample_counts(requirements)
        # The integral over the centred unit cell is sinc(cycles), including
        # its sign beyond a fringe null. Comparing complex values catches the
        # even grid's phase bias; comparing magnitudes would miss it.
        for fringe, count in zip(cycles, counts):
            sampled = np.exp(2j * np.pi * fringe * fine_offsets(count, 1)).mean()
            assert abs(sampled - np.sinc(fringe)) < 1 / snr

    def test_ceiling_alone_fails_for_an_even_count(self):
        snr, cycles = 10000, 0.31
        requirement = np.pi * cycles * np.sqrt(snr / 6)
        even = int(np.ceil(requirement))
        assert even == 40
        odd = int(poly_sample_counts([requirement])[0])
        assert odd == 41
        exact = np.sinc(cycles)
        errors = [abs(np.exp(2j * np.pi * cycles * fine_offsets(n, 1)).mean() - exact) for n in (even, odd)]
        assert errors[0] > 200 / snr
        assert errors[1] < 1 / snr
