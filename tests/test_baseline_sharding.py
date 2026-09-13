"""Sharding the baseline axis rather than the source axis.

The source mesh leaves every visibility-shaped array replicated -- its ``psum``
is declared ``out_specs=P()`` -- so four devices divided 0.24 GB of a 36 GB
footprint and the limiting device got *bigger*. Everything that is actually
large carries a baseline axis and no source axis, so that is the axis to split.

The device count has to be fixed before jax initialises, so the multi-device
cases run in a subprocess with four host devices rather than in this one.
"""
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from tabascal import distributed


def run_on_four_devices(body: str):
    """Execute ``body`` in a fresh interpreter that sees four host devices."""
    script = textwrap.dedent(
        """
        import os
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
        import jax
        jax.config.update("jax_platforms", "cpu")
        import numpy as np
        import jax.numpy as jnp
        from jax.sharding import PartitionSpec as P
        from tabascal.distributed import (
            baseline_mesh, baselines_divide, map_over_baselines, sharding_enabled,
        )
        assert jax.device_count() == 4, jax.device_count()
        """
    ) + textwrap.dedent(body)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    return out.stdout


class TestSingleDevice:
    """With one device the map must vanish, exactly as the source route does."""

    def test_the_map_is_the_function_itself(self):
        fn = lambda x: x * 2
        assert distributed.map_over_baselines(fn, in_specs=(None,)) is fn

    def test_nothing_divides_without_sharding(self):
        # One device is not sharding, whatever the count divides into.
        assert distributed.baselines_divide(130816) is False


class TestFourDevices:
    def test_the_mesh_axis_is_named_for_baselines(self):
        assert "bl" in run_on_four_devices(
            """
            print(baseline_mesh().axis_names[0])
            """
        )

    def test_the_mesh_is_cached_so_jit_does_not_recompile(self):
        # A fresh Mesh compares unequal and every sharding built from it with it.
        assert "True" in run_on_four_devices(
            """
            print(baseline_mesh() is baseline_mesh())
            """
        )

    @pytest.mark.parametrize("n_bl,expected", [(130816, True), (73536, True),
                                               (32640, True), (130817, False)])
    def test_divisibility_is_reported_not_assumed(self, n_bl, expected):
        assert str(expected) in run_on_four_devices(
            f"""
            print(baselines_divide({n_bl}))
            """
        )

    def test_each_device_computes_its_own_baselines_with_no_reduction(self):
        """The point of the exercise: the output is split, not summed.

        The per-baseline argument arrives already divided and the per-antenna
        one whole, which is the split the RFI visibility needs -- every device
        holds the full signal and produces only the baselines it owns.
        """
        out = run_on_four_devices(
            """
            n_bl = 16
            a1 = jnp.arange(n_bl)          # per baseline: divided
            signal = jnp.arange(5.0)       # per antenna: whole on every device

            def local(a1_local, signal_whole):
                # Each shard sees a quarter of the baselines and all the signal.
                assert a1_local.shape == (n_bl // 4,), a1_local.shape
                assert signal_whole.shape == (5,), signal_whole.shape
                return a1_local * signal_whole.sum()

            mapped = map_over_baselines(local, in_specs=(P("bl"), P()))
            got = jax.jit(mapped)(a1, signal)
            want = a1 * signal.sum()
            assert got.shape == (n_bl,), got.shape
            assert jnp.allclose(got, want), (got, want)
            print("OK")
            """
        )
        assert "OK" in out

    def test_device_local_index_values_differ_from_a_global_slice(self):
        """The correctness trap: sorters must be rebuilt, not sliced.

        ``a1_sorter`` is an argsort of the *local* ``a1`` and ``pair_index``
        maps a pair to a *local* baseline number. Slicing the global arrays
        gives indices that point outside the shard, and silently -- nothing
        raises, the visibilities are just wrong. Stack per-device arrays instead.
        """
        rng = np.random.default_rng(0)
        a1 = rng.integers(0, 8, size=16)
        per_device = np.stack([np.argsort(a1[i * 4:(i + 1) * 4]) for i in range(4)])
        sliced = np.argsort(a1).reshape(4, 4)
        # Were these the same, slicing the global sorter would be safe. It is not.
        assert not np.array_equal(per_device, sliced)
        assert per_device.max() < 4  # local indices stay inside their own shard


class TestSplittingAGroupOverDevices:
    """Each device takes a share of the baselines and keeps the whole antenna set."""

    @staticmethod
    def a_group(n_ant=8):
        from tabascal.poly_interp import make_poly_time_group
        a1, a2 = np.triu_indices(n_ant, k=1)
        req = np.full(len(a1), 9.0)
        return make_poly_time_group(req, a1, a2, np.arange(len(a1)))

    def test_every_baseline_lands_on_exactly_one_device(self):
        from tabascal.poly_interp import split_group_over_devices
        group = self.a_group()                      # 28 baselines over 8 antennas
        shards = split_group_over_devices(group, 4)
        assert len(shards) == 4
        seen = np.concatenate([s.baseline_indices for s in shards])
        # A partition: nothing computed twice, nothing dropped.
        assert np.array_equal(np.sort(seen), np.sort(group.baseline_indices))
        assert len(np.unique(seen)) == len(seen)

    def test_the_shards_are_the_same_shape(self):
        from tabascal.poly_interp import split_group_over_devices
        shards = split_group_over_devices(self.a_group(), 4)
        # shard_map runs one program: differing shapes would not compile.
        assert len({len(s.baseline_indices) for s in shards}) == 1
        assert len({len(s.a1) for s in shards}) == 1

    def test_the_antenna_axis_is_shared_not_recompacted(self):
        """The signal must be identical on every device, so it enters replicated."""
        from tabascal.poly_interp import split_group_over_devices
        group = self.a_group()
        for shard in split_group_over_devices(group, 4):
            assert np.array_equal(shard.antennas, group.antennas)
            assert shard.n_g == group.n_g

    def test_endpoints_follow_their_own_baselines(self):
        from tabascal.poly_interp import split_group_over_devices
        group = self.a_group()
        shards = split_group_over_devices(group, 4)
        per = len(group.baseline_indices) // 4
        for d, shard in enumerate(shards):
            want = slice(d * per, (d + 1) * per)
            assert np.array_equal(shard.a1, group.a1[want])
            assert np.array_equal(shard.a2, group.a2[want])

    def test_a_remainder_is_refused_rather_than_dropped(self):
        from tabascal.poly_interp import split_group_over_devices
        group = self.a_group()                      # 28 baselines
        with pytest.raises(ValueError, match="left over"):
            split_group_over_devices(group, 8)      # 28 / 8 leaves 4
        # The message must name the numbers, so a config error is actionable.
        with pytest.raises(ValueError, match="28 baselines evenly over 3"):
            split_group_over_devices(group, 3)

    @pytest.mark.parametrize("bad", [0, -1, True, 2.0])
    def test_a_nonsensical_device_count_is_refused(self, bad):
        from tabascal.poly_interp import split_group_over_devices
        with pytest.raises(ValueError, match="positive whole number"):
            split_group_over_devices(self.a_group(), bad)

    def test_one_device_is_the_group_itself(self):
        from tabascal.poly_interp import split_group_over_devices
        group = self.a_group()
        only, = split_group_over_devices(group, 1)
        assert np.array_equal(only.baseline_indices, group.baseline_indices)
        assert np.array_equal(only.a1, group.a1)


class TestGhostPaddedDeviceGroups:
    """Splitting by owning device, padded so every device carries one shape."""

    @staticmethod
    def a_group(n_ant=12, keep=None):
        from tabascal.poly_interp import make_poly_time_group
        a1, a2 = np.triu_indices(n_ant, k=1)
        req = np.full(len(a1), 9.0)
        idx = np.arange(len(a1)) if keep is None else keep
        return make_poly_time_group(req, a1, a2, idx)

    def test_every_device_gets_the_same_shape(self):
        from tabascal.poly_interp import device_groups
        group = self.a_group()                       # 66 baselines, does not divide by 4
        shards = device_groups(group, 4, n_bl_total=68)
        assert len({len(s.a1) for s in shards}) == 1
        assert len({len(s.a2) for s in shards}) == 1

    def test_a_baseline_goes_to_the_device_that_owns_its_row(self):
        """The split follows the data's own ordering; nothing is reordered."""
        from tabascal.poly_interp import device_groups
        # A group holding only some baselines, spread unevenly over the blocks.
        keep = np.array([0, 1, 2, 3, 20, 40, 41, 60, 61, 62])
        group = self.a_group(keep=keep)
        shards = device_groups(group, 4, n_bl_total=68)   # blocks of 17
        expected = [np.sum(keep // 17 == d) for d in range(4)]
        assert [s.n_real for s in shards] == expected
        # Padded to the largest share, which is the first block's four.
        assert all(len(s.a1) == max(expected) for s in shards)

    def test_positions_are_local_to_the_owning_block(self):
        from tabascal.poly_interp import device_groups
        keep = np.array([0, 1, 2, 3, 20, 40, 41, 60, 61, 62])
        group = self.a_group(keep=keep)
        shards = device_groups(group, 4, n_bl_total=68)
        for d, shard in enumerate(shards):
            mine = keep[keep // 17 == d]
            # A device writes into its own rows, counted from its block's start.
            assert np.array_equal(shard.positions, mine - d * 17)
            assert np.all(shard.positions < 17)

    def test_the_real_baselines_are_a_partition(self):
        from tabascal.poly_interp import device_groups
        group = self.a_group()
        shards = device_groups(group, 4, n_bl_total=68)
        assert sum(s.n_real for s in shards) == len(group.baseline_indices)
        rebuilt = np.concatenate([s.a1[: s.n_real] for s in shards])
        assert np.array_equal(rebuilt, group.a1)

    def test_ghosts_pair_with_an_antenna_the_group_does_not_have(self):
        """A ghost pair cannot collide with a real one, which the operator forbids."""
        from tabascal.poly_interp import device_groups
        group = self.a_group()
        ghost = len(group.antennas)
        for shard in device_groups(group, 4, n_bl_total=68):
            ghosts_1, ghosts_2 = shard.a1[shard.n_real:], shard.a2[shard.n_real:]
            assert np.all(ghosts_2 == ghost)
            assert np.all(ghosts_1 < ghost)
            pairs = np.stack([shard.a1, shard.a2], axis=1)
            assert len(np.unique(pairs, axis=0)) == len(pairs)

    def test_an_even_group_needs_no_ghosts(self):
        from tabascal.poly_interp import device_groups
        group = self.a_group(n_ant=8)                # 28 baselines, divides by 4
        for shard in device_groups(group, 4, n_bl_total=28):
            assert shard.n_real == len(shard.a1) == 7

    def test_an_indivisible_total_is_refused(self):
        from tabascal.poly_interp import device_groups
        group = self.a_group()
        with pytest.raises(ValueError, match="left over"):
            device_groups(group, 4, n_bl_total=66)

    def test_one_device_leaves_the_group_untouched(self):
        from tabascal.poly_interp import device_groups
        group = self.a_group()
        only, = device_groups(group, 1, n_bl_total=66)
        assert only.n_real == len(group.baseline_indices)
        assert np.array_equal(only.a1, group.a1)
