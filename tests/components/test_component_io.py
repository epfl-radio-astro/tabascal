"""Declared-vs-runtime I/O consistency for components.

Each component declares its state I/O (``reads`` / ``writes`` / ``accumulates``,
key -> symbolic shape). The dependency resolver trusts those declarations, so
this module checks they match what each ``forward`` actually does at runtime:
the set of keys read/written/accumulated, and that the array under each key has
the declared shape (symbolic dims resolved to the invocation's concrete sizes).

It reuses the per-component config/state builders from the sibling test modules
so no MeasurementSet or TLE data is needed. Components needing network/TLE data
(trajectory, rfi_signal) are covered by the end-to-end pipeline tests instead.
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from tabascal.components.ast_vis import (
    ImageVisCalculation,
    PointSourceVisCalculation,
)
from tabascal.components.ast_signal import FixedPointSky, PointSky
from tabascal.components.gains import UnitaryGains
from tabascal.components.rfi_vis import RiemannVisTimeFreqCalculation

from .conftest import assert_declared_io, make_constants
from . import test_ast_vis_point_source as tps
from . import test_ast_image_vis as tiv
from . import test_ast_signal as tas
from . import test_rfi_vis as trv
from . import test_gains as tg


def test_point_source_vis_calculation_io():
    n_ant, n_src, n_time, n_freq = 4, 5, 6, 3
    config = tps.make_config(n_ant, n_src, n_time, n_freq)
    state = tps.make_state(config, n_src)
    comp = PointSourceVisCalculation()
    comp.setup(config)
    dims = {"n_src": n_src, "n_freq": n_freq, "n_bl": config.n_bl,
            "n_time": n_time}
    assert_declared_io(comp, {}, state, make_constants(comp), dims)


def test_image_vis_calculation_io():
    config = tiv.make_config(n_ant=4, n_time=3, n_freq=2)
    n_pix = config.args["ast"]["image"]["n_pix"]
    state = {
        "ast_image": jnp.zeros((config.n_freq, n_pix, n_pix)),
        "vis_ast": jnp.zeros((config.n_bl, config.n_freq, config.n_time), complex),
    }
    comp = ImageVisCalculation()
    comp.setup(config)
    dims = {"n_freq": config.n_freq, "n_l": n_pix, "n_m": n_pix,
            "n_bl": config.n_bl, "n_time": config.n_time}
    assert_declared_io(comp, {}, state, make_constants(comp), dims)


def test_fixed_point_sky_io(tmp_path):
    n_src, n_time, n_freq = 5, 3, 4
    zarr_path, _, _ = tas.write_sky_zarr(tmp_path, n_src, n_time, n_freq)
    config = tas.make_config(zarr_path, n_ant=3, n_freq=n_freq, n_time=2)
    comp = FixedPointSky()
    comp.setup(config)
    dims = {"n_src": n_src, "n_freq": n_freq}
    assert_declared_io(comp, {}, {}, make_constants(comp), dims)


def test_point_sky_io(tmp_path):
    n_src, n_time, n_freq = 5, 2, 2
    freqs = np.linspace(1.4e9, 1.5e9, n_freq)
    rng = np.random.default_rng(0)
    radec = rng.uniform([20.0, -32.0], [34.0, -28.0], size=(n_src, 2))
    flux = np.abs(rng.normal(size=(n_src, n_time, n_freq))) + 0.1
    zarr_path = tas.write_point_catalogue(tmp_path, radec, flux, freqs)
    config = tas.make_point_config(zarr_path, freqs=freqs, n_freq=n_freq)
    comp = PointSky()
    comp.setup(config)
    dims = {"n_src": n_src, "n_freq": n_freq}
    assert_declared_io(comp, comp.init_params_base, {}, make_constants(comp), dims)


def test_riemann_vis_time_freq_calculation_io():
    n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq = 4, 5, 6, 7, 8, 9
    config = trv.create_config(n_ant, n_rfi, n_time, n_freq, n_int_time, n_int_freq)
    state = trv.create_state(config, rand_vis_rfi=False)
    comp = RiemannVisTimeFreqCalculation()
    comp.setup(config)
    dims = {
        "n_rfi": n_rfi, "n_ant": n_ant,
        "n_freq_fine": n_freq * n_int_freq, "n_time_fine": n_time * n_int_time,
        "n_bl": config.n_bl, "n_freq": n_freq, "n_time": n_time,
    }
    assert_declared_io(comp, {}, state, make_constants(comp), dims)


def test_unitary_gains_io():
    n_ant, n_freq, n_time = 4, 2, 6
    config = tg.make_gains_config(n_ant=n_ant, n_freq=n_freq, n_time=n_time)
    state = tg.make_vis_state(n_ant, n_freq, n_time)
    comp = UnitaryGains()
    comp.setup(config)
    dims = {"n_bl": config.n_bl, "n_freq": n_freq, "n_time": n_time, "n_ant": n_ant}
    assert_declared_io(comp, {}, state, make_constants(comp), dims)
