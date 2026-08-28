"""Tests for TabConfig.set_elevation_mask — building the RFI elevation mask.

The elevation *computation* is covered by
``tests/components/test_trajectory.py::TestSatelliteElevations``; what is pinned
here is the masking logic built on top of it, which is otherwise only exercised
indirectly through components that take a ready-made mask.

``get_satellite_elevations`` is stubbed so the elevations are exact and the
boundary cases are reachable at all — a propagated orbit never lands precisely
on a cutoff.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from tabascal.config import TabConfig


def build_mask(elevations, min_elevation, n_int_time=1, monkeypatch=None,
               require_in_view=True):
    """Run set_elevation_mask against a fixed elevation array."""
    elevations = np.asarray(elevations, dtype=float)
    n_rfi, n_time = elevations.shape

    cfg = SimpleNamespace(
        n_rfi=n_rfi,
        n_int_time=n_int_time,
        norad_ids=[40000 + i for i in range(n_rfi)],
        orbit_records=[{} for _ in range(n_rfi)],
        times_jd=2460000.5 + np.arange(n_time) / 86400.0,
        ants_itrf=np.zeros((3, 3)),
    )
    monkeypatch.setattr(
        "tabascal.config.get_satellite_elevations",
        lambda *args, **kwargs: elevations,
    )
    TabConfig.set_elevation_mask(cfg, min_elevation, require_in_view=require_in_view)
    return cfg


class TestElevationMaskBoundary:
    """min_elevation is the lowest elevation still modelled, so the cut is inclusive."""

    def test_a_sample_exactly_on_the_cut_is_kept(self, monkeypatch):
        # The option masks elevations *below* the cut, so equality is in view.
        cfg = build_mask([[-5.0, 0.0, 5.0]], 0.0, monkeypatch=monkeypatch)
        np.testing.assert_array_equal(cfg.rfi_mask[0], [False, True, True])

    def test_a_nonzero_cut_is_inclusive_too(self, monkeypatch):
        cfg = build_mask([[9.9, 10.0, 10.1]], 10.0, monkeypatch=monkeypatch)
        np.testing.assert_array_equal(cfg.rfi_mask[0], [False, True, True])

    def test_a_pass_peaking_exactly_at_the_cut_is_not_rejected(self, monkeypatch):
        """A strict comparison would call this satellite 'never above' and raise."""
        cfg = build_mask([[-10.0, 20.0, -10.0]], 20.0, monkeypatch=monkeypatch)
        assert cfg.rfi_mask[0].sum() == 1


class TestElevationMaskBehaviour:

    def test_none_disables_masking(self, monkeypatch):
        cfg = build_mask([[10.0, 20.0]], None, monkeypatch=monkeypatch)
        assert cfg.rfi_mask_fine is None
        assert cfg.rfi_mask is None
        assert cfg.rfi_elevation is None

    def test_each_satellite_gets_its_own_window(self, monkeypatch):
        cfg = build_mask(
            [[30.0, 30.0, -1.0], [-1.0, 30.0, 30.0]], 0.0, monkeypatch=monkeypatch
        )
        np.testing.assert_array_equal(cfg.rfi_mask[0], [True, True, False])
        np.testing.assert_array_equal(cfg.rfi_mask[1], [False, True, True])

    @pytest.mark.parametrize("n_int_time", [1, 2, 5])
    def test_the_mask_is_expanded_over_whole_integrations(self, monkeypatch, n_int_time):
        """An integration is either fully modelled or fully masked, never split."""
        cfg = build_mask(
            [[30.0, -1.0, 30.0]], 0.0, n_int_time=n_int_time, monkeypatch=monkeypatch
        )
        assert cfg.rfi_mask_fine.shape == (1, 3 * n_int_time)
        fine = cfg.rfi_mask_fine[0].reshape(3, n_int_time)
        for integration in fine:
            assert integration.all() or not integration.any()
        np.testing.assert_array_equal(fine[:, 0], [True, False, True])

    def test_a_satellite_never_above_the_cut_raises(self, monkeypatch):
        with pytest.raises(ValueError, match="never above"):
            build_mask([[10.0, 20.0], [-5.0, -1.0]], 30.0, monkeypatch=monkeypatch)

    def test_the_error_names_the_offending_satellite(self, monkeypatch):
        with pytest.raises(ValueError) as excinfo:
            build_mask([[50.0, 50.0], [-5.0, -1.0]], 30.0, monkeypatch=monkeypatch)
        assert "40001" in str(excinfo.value)


class TestRequireInView:
    """``require_in_view=False`` measures what is there instead of stopping.

    A fully-masked satellite has no signal for inference to fit, so the default
    is to stop. The matched-filter light-curve extractor is not inference: it
    reports a zero curve for a satellite that never rose, which is an answer, and
    stopping instead would make it impossible to measure the satellites that
    *were* up without first editing them out of the config.
    """

    def test_a_never_visible_satellite_is_tolerated(self, monkeypatch):
        cfg = build_mask(
            [[50.0, 50.0], [-5.0, -1.0]], 30.0,
            monkeypatch=monkeypatch, require_in_view=False,
        )
        np.testing.assert_array_equal(cfg.rfi_mask[0], [True, True])
        assert not cfg.rfi_mask[1].any()

    def test_the_tolerated_satellite_is_still_named(self, monkeypatch, capsys):
        """Silently modelling nothing for a configured satellite is the failure."""
        build_mask(
            [[50.0, 50.0], [-5.0, -1.0]], 30.0,
            monkeypatch=monkeypatch, require_in_view=False,
        )
        out = capsys.readouterr().out
        assert "40001" in out and "never above" in out

    def test_the_default_still_raises(self, monkeypatch):
        with pytest.raises(ValueError, match="never above"):
            build_mask([[50.0, 50.0], [-5.0, -1.0]], 30.0, monkeypatch=monkeypatch)
