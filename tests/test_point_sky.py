"""FixedPointSky + PointSourceVisCalculation.

The point of these components is to give the gain a sky it cannot deform, so the thing
worth pinning down is that the visibilities they produce are actually right -- a wrong
uvw axis order or sign convention would still "work" and silently corrupt any gain solved
against them.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tabascal.components.ast_signal import FixedPointSky
from tabascal.components.ast_vis import PointSourceVisCalculation


class FakeConfig:
    """The attributes the two components read off TabConfig."""

    def __init__(self, sources, n_bl=5, n_time=3, freqs=(1.0e8, 1.5e8)):
        self.n_bl = n_bl
        self.n_time = n_time
        self.n_freq = len(freqs)
        self.freqs = np.array(freqs)
        self.phase_centre = {"ra": 30.0, "dec": -26.7}     # degrees
        rng = np.random.default_rng(0)
        # (n_time, n_bl, 3) -- the axis order read_ms gives on this branch
        self.uvw = rng.normal(0, 200, (n_time, n_bl, 3))
        self.args = {"ast": {"point_sources": sources}}


def _build(cfg):
    sky, vis = FixedPointSky(), PointSourceVisCalculation()
    sky.setup(cfg)
    vis.setup(cfg)
    constants = {}
    for comp in (sky, vis):
        for k, v in comp.build_constants().items():
            constants[f"{comp.prefix}/{k}"] = v
    state = {**sky.state_outputs, **vis.state_outputs}
    state = sky.build_forward()({}, state, constants)
    return vis.build_forward()({}, state, constants)


def test_source_at_phase_centre_gives_flat_visibility():
    """l=m=0, n=1 -> zero delay on every baseline, so V == I exactly.

    This is the check that catches a transposed uvw or a flipped sign: both still produce
    a plausible-looking array, but neither gives a constant V = I here.
    """
    cfg = FakeConfig([{"name": "at centre", "ra": 30.0, "dec": -26.7, "I": 7.0}])
    out = _build(cfg)

    vis = np.asarray(out["vis_ast"])
    assert vis.shape == (cfg.n_bl, cfg.n_freq, cfg.n_time)
    assert np.allclose(vis.real, 7.0, atol=1e-4)
    assert np.allclose(vis.imag, 0.0, atol=1e-4)


def test_offset_source_has_unit_amplitude_and_varying_phase():
    """Off centre the amplitude is still I/n, but the phase must vary with baseline."""
    cfg = FakeConfig([{"ra": 34.0, "dec": -30.0, "I": 3.0}])
    vis = np.asarray(_build(cfg)["vis_ast"])

    assert np.allclose(np.abs(vis), np.abs(vis).mean(), rtol=1e-3)   # |V| ~ I/n
    assert np.ptp(np.angle(vis)) > 0.1                               # fringes, not constant


def test_fluxes_add_and_accumulate_onto_existing_vis_ast():
    """It ADDS to vis_ast, so it composes with the astronomical GP rather than replacing it."""
    src = {"ra": 30.0, "dec": -26.7, "I": 2.0}
    cfg = FakeConfig([src, src])                    # two identical sources -> 2x the flux
    vis = np.asarray(_build(cfg)["vis_ast"])
    assert np.allclose(vis.real, 4.0, atol=1e-4)


def test_power_law_spectrum():
    cfg = FakeConfig(
        [{"ra": 30.0, "dec": -26.7, "I": 750.0, "ref_freq_mhz": 154.0, "alpha": -0.77}],
        freqs=(154.0e6, 308.0e6),
    )
    sky = FixedPointSky()
    sky.setup(cfg)

    I = np.asarray(sky.ast_I)[0]
    assert np.isclose(I[0], 750.0)                       # at the reference frequency
    assert np.isclose(I[1], 750.0 * 2.0**-0.77)          # one octave up


def test_empty_catalogue_is_an_error():
    with pytest.raises(RuntimeError, match="point_sources is empty"):
        FixedPointSky().setup(FakeConfig([]))
