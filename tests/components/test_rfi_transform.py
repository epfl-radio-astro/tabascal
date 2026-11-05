"""``rfi.transform`` -- the Fourier RFI signals with and without the padded grid.

The DFT is held to the FFT in value and in gradient on every Fourier signal
component, on the fine grid and on the data grid. The matrices themselves are
held to the pad, inverse FFT and crop chain in ``test_ast_transform``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tabascal.components.rfi_signal import (
    ComplexRFIConstAnt,
    ComplexRFIConstAntFine,
    ComplexRFIVarAnt,
    ComplexRFIVarAntFine,
    rfi_signal_config_validation,
)

from .conftest import active_precision, make_constants
from .test_rfi_signal import make_rfi_config, random_params

# (n_time, n_int_time). The data-grid signals do not supersample, so they take
# an even and an odd padded length; (7, 2) is an odd one supersampled evenly.
CASES = [
    *[(cls, n, k) for cls in (ComplexRFIVarAntFine, ComplexRFIConstAntFine) for n, k in [(12, 1), (12, 3), (7, 2)]],
    *[(cls, n, 1) for cls in (ComplexRFIVarAnt, ComplexRFIConstAnt) for n in (12, 7)],
]


def _setup(cls, transform, n_time, n_int_time):
    config = make_rfi_config(n_freq=5, n_time=n_time, n_int_time=n_int_time)
    config.args["rfi"]["transform"] = transform
    comp = cls()
    comp.setup(config)
    return comp


def _rtol():
    return 1e-10 if active_precision() == "double" else 2e-4


@pytest.mark.parametrize("cls, n_time, n_int_time", CASES)
def test_dft_matches_fft_in_value_and_gradient(cls, n_time, n_int_time):
    fft, dft = _setup(cls, "fft", n_time, n_int_time), _setup(cls, "dft", n_time, n_int_time)
    assert dft.init_params_base.keys() == fft.init_params_base.keys()
    params = random_params(fft)

    def run(comp):
        forward, constants = comp.build_forward(), make_constants(comp)
        rfi_A = lambda p: forward(p, {}, constants)["rfi_A"]
        loss = lambda p: jnp.sum(jnp.abs(rfi_A(p)) ** 2)
        return rfi_A(params), jax.grad(loss)(params)

    (a_fft, g_fft), (a_dft, g_dft) = run(fft), run(dft)
    scale = float(jnp.max(jnp.abs(a_fft)))
    np.testing.assert_allclose(a_dft, a_fft, rtol=_rtol(), atol=_rtol() * scale)
    for name in g_fft:
        g_scale = float(jnp.max(jnp.abs(g_fft[name])))
        np.testing.assert_allclose(
            g_dft[name], g_fft[name], rtol=_rtol(), atol=_rtol() * g_scale, err_msg=name
        )


@pytest.mark.parametrize("given, resolved", [(None, "fft"), ("fft", "fft"), ("dft", "dft")])
def test_transform_key_defaults_to_fft(given, resolved):
    cfg = {"gp_cov": {"std": 1.0, "corr_freq": 1e6, "corr_time": 10.0}}
    if given is not None:
        cfg["transform"] = given
    out = rfi_signal_config_validation(
        cfg, jnp.ones((3, 4, 8)), jnp.linspace(1.4e9, 1.41e9, 4), 1e6, jnp.linspace(0, 70, 8), 10.0
    )
    assert out["transform"] == resolved


@pytest.mark.parametrize("bad", ["nufft", False, 0, ""])
def test_unknown_transform_is_refused(bad):
    cfg = {"gp_cov": {"std": 1.0, "corr_freq": 1e6, "corr_time": 10.0}, "transform": bad}
    with pytest.raises(ValueError, match="rfi.transform"):
        rfi_signal_config_validation(
            cfg, jnp.ones((3, 4, 8)), jnp.linspace(1.4e9, 1.41e9, 4), 1e6, jnp.linspace(0, 70, 8), 10.0
        )
