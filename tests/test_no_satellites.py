"""A model that configures no satellites must build, not fail somewhere downstream.

``norad_ids: []`` is the shipped default in ``tab_config_base.yaml``, and
``TabConfig`` runs the RFI setup unconditionally, so every step it touches has to
cope with an empty satellite set. Returning empty element arrays from the fetch
is only half of it: the RFI sampling estimate then has to survive them too.
"""

import numpy as np
from types import SimpleNamespace

from tabascal.components.trajectory import fetch_orbital_elements
from tabascal.config import TabConfig
from tabascal.tle import TLEResolution


def _empty_resolution() -> TLEResolution:
    return TLEResolution(
        requested=[], obs_epoch_jd=float("nan"), remote_max_age_days=3.0
    )


def test_element_fetch_yields_an_empty_rfi_model():
    elements, epoch_jd, norad_ids, tles, n_rfi_real = fetch_orbital_elements(
        resolution=_empty_resolution()
    )
    assert elements.shape == (0, 6)
    assert epoch_jd.shape == (0,)
    assert norad_ids == []
    assert tles.shape == (0, 2)
    assert n_rfi_real == 0


def test_rfi_sampling_estimate_survives_zero_satellites():
    """The step that used to fail with an opaque AxisError, far from the cause.

    With no satellites ``get_satellite_positions`` returns an empty array, so the
    fringe frequencies collapse to 1-D and the ``axis=(0, 1)`` reduction that
    derives the sampling rate cannot run. There is nothing to sample, so the
    sampling estimate has to be skipped rather than computed from nothing.
    """
    config = SimpleNamespace(n_rfi=0, n_bl=45, vis_obs=np.ones((4, 3), dtype=complex))

    TabConfig.estimate_rfi_sampling(config, 1.0, 1, 30, min_divisors=1)

    assert config.n_int_time >= 1
    assert len(config.time_strides) >= 1
    # One stride group covering every baseline: no satellite splits them apart.
    assert len(config.time_sample_idxs) == 1


def test_rfi_sampling_estimate_does_not_touch_the_satellite_path():
    """The guard must return before any satellite-dependent attribute is read.

    ``vis_obs`` and ``n_bl`` are all it may use; reaching for ``tles``, ``freqs``
    or the antenna geometry would mean the empty-model path is still running the
    satellite computation.
    """
    config = SimpleNamespace(n_rfi=0, n_bl=8, vis_obs=np.ones((2, 2), dtype=complex))

    TabConfig.estimate_rfi_sampling(config, 1.0, 1, 30, min_divisors=1)

    for attribute in ("tles", "freqs", "ants_itrf", "times_jd", "phase_centre"):
        assert not hasattr(config, attribute), (
            f"the satellite-free path read {attribute!r}; it should have returned first"
        )
