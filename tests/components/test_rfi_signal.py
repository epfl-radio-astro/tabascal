"""Tests for tabascal.components.rfi_signal — the GP RFI-signal components.

The central contract here is the device-sharding one: the RFI axis is padded up to a
multiple of the device count with duplicate "dark dummy" satellites, and those dummies
must contribute exactly zero amplitude and exactly zero gradient. That is enforced by
``BaseGPRFI.masked_forward_transform`` inside every component's forward, so it holds for
arbitrary parameters — not just the init values. ``TestDummySourcesStayDark`` pins it down.

Padding needs no extra devices: it is driven purely by ``n_rfi_real < n_rfi`` on the
config, and the mask is device-independent. Only ``test_multi_device_padded_sources_dark``
spends a subprocess on the real sharded placeholders.
"""

import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

import jax
from jax import vmap
import jax.numpy as jnp
import numpy as np
import numpyro

from tabascal.components.rfi_signal import (
    ComplexRFIVarAnt,
    ComplexRFIConstAnt,
    read_light_curves,
    rfi_signal_config_validation,
)
from tabascal.fft_gp import latent_to_signal

from .conftest import active_precision, assert_transform_roundtrip, make_constants

# The in-memory MS rig from the reader's own tests, whose TIME column declares
# its scale through a MEASINFO record the way casacore does. Imported rather
# than restated so the scale under test is the one read_ms actually reads.
from ..test_ms import _keywords, run_reader  # noqa: F401 -- run_reader is a fixture


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_RFI, N_RFI_REAL, N_ANT, N_FREQ, N_TIME = 4, 3, 3, 4, 8

# Arbitrary but fixed epoch for the mock observation's absolute time grid.
_TEST_EPOCH_JD = 2460000.5
_TEST_EPOCH_MJD = _TEST_EPOCH_JD - 2400000.5

ALL_CLASSES = [ComplexRFIVarAnt, ComplexRFIConstAnt]
FOURIER_CLASSES = [ComplexRFIVarAnt, ComplexRFIConstAnt]

# Init modes every class accepts. "truth" is excluded throughout: it goes through
# read_true_rfi_A, which needs a real simulation .zarr store.
COMMON_INITS = ["prior", "zeros", "ones", "sample"]


def _norad_ids(n_rfi, n_rfi_real):
    """NORAD ids laid out as TabConfig builds them: real sources, then padding.

    Device sharding pads the satellite list to a multiple of the device count by
    repeating the *last real* satellite, so the tail entries are duplicates of it --
    which is exactly why consumers that look ids up in an external file must slice to
    ``[:n_rfi_real]`` rather than trust the full list to be unique.
    """
    ids = [40000 + i for i in range(n_rfi_real)]
    return ids + [ids[-1]] * (n_rfi - n_rfi_real)


def make_rfi_config(
    n_rfi=N_RFI,
    n_rfi_real=N_RFI_REAL,
    n_ant=N_ANT,
    n_freq=N_FREQ,
    n_time=N_TIME,
    n_int_freq=1,
    n_int_time=1,
    init="prior",
    mean="zeros",
    est=None,
    r_seed=1,
    var=1.0,
    corr_freq=5e6,
    corr_time=60.0,
    pad_factor=2,
    with_n_rfi_real=True,
    rfi_mask_fine=None,
    time_scale="utc",
):
    """Build a minimal mock TabConfig for the RFI-signal components.

    A real TabConfig needs a Measurement Set, Space-Track credentials and skyfield, so
    the component tests in this package stub it with a SimpleNamespace instead.

    ``pad_factor`` defaults to 2 to match ``tab_config_base.yaml`` — small pad factors
    combined with supersampling crop the fine grid down to a zero-sized axis.
    """
    freqs = jnp.linspace(1.4e9, 1.41e9, n_freq)
    chan_width = float(freqs[1] - freqs[0]) if n_freq > 1 else 1e6
    times = jnp.linspace(0.0, 120.0, n_time)
    int_time = float(times[1] - times[0]) if n_time > 1 else 8.0

    n_freq_fine, n_time_fine = n_freq * n_int_freq, n_time * n_int_time
    n_bl = n_ant * (n_ant - 1) // 2

    config = SimpleNamespace(
        norad_ids=_norad_ids(n_rfi, n_rfi_real),
        rfi_mask_fine=rfi_mask_fine,
        n_rfi=n_rfi,
        n_ant=n_ant,
        n_freq=n_freq,
        n_time=n_time,
        n_freq_fine=n_freq_fine,
        n_time_fine=n_time_fine,
        n_int_freq=n_int_freq,
        n_int_time=n_int_time,
        freqs=freqs,
        freqs_fine=jnp.linspace(freqs[0], freqs[-1], n_freq_fine),
        chan_width=chan_width,
        times=times,
        # Absolute grid the light-curve estimate is interpolated against. MJD is
        # carried alongside JD rather than derived from it; see read_ms_params.
        # ``time_scale`` is what the MS's TIME column declared, and is what puts
        # times_mjd onto the UTC the estimate file's own axis is on.
        times_mjd=_TEST_EPOCH_MJD + np.linspace(0.0, 120.0, n_time, dtype=np.float64) / 86400.0,
        times_jd=_TEST_EPOCH_JD + np.linspace(0.0, 120.0, n_time, dtype=np.float64) / 86400.0,
        time_scale=time_scale,
        times_fine=jnp.linspace(times[0], times[-1], n_time_fine),
        int_time=int_time,
        vis_obs=jnp.ones((n_bl, n_freq, n_time), dtype=complex),
        args={
            "rfi": {
                "r_seed": r_seed,
                "var": var,
                "corr_freq": corr_freq,
                "corr_time": corr_time,
                "init": init,
                "mean": mean,
                "est": est,
                "time_pad_factor": pad_factor,
                "freq_pad_factor": pad_factor,
            },
            "plots": {"truth": False},
            "data": {"zarr_path": None, "data_col": "DATA"},
        },
    )
    # Omitted entirely (rather than set to None) so the getattr fallback in
    # BaseGPRFI.setup is exercised as it would be on an unpadded TabConfig.
    if with_n_rfi_real:
        config.n_rfi_real = n_rfi_real

    return config


def setup_component(cls, **kwargs):
    """Construct and set up a component on a mock config."""
    comp = cls()
    comp.setup(make_rfi_config(**kwargs))
    return comp


def random_params(comp, seed=0):
    """Random values shaped like the component's base parameters.

    Deliberately *not* the init values: the darkness guarantee has to hold for whatever
    the optimiser wanders into, not just the starting point.
    """
    names = sorted(comp.init_params_base)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(names))
    return {
        name: jax.random.normal(key, comp.init_params_base[name].shape)
        for name, key in zip(names, keys)
    }


def run_forward(comp, params=None, state=None):
    """Run the component's forward pass and return the rfi_A output."""
    if params is None:
        params = comp.init_params_base
    forward = comp.build_forward()
    return forward(params, {} if state is None else state, make_constants(comp))["rfi_A"]


def make_est_file(tmp_path, n_rfi_real=N_RFI_REAL, n_time=N_TIME, n_freq=N_FREQ):
    """Write the light-curve .npz that the `est` prior/init modes read.

    Covers the mock observation's own time and frequency span, so the
    interpolation in read_light_curves is an identity rather than an edge case.
    """
    path = tmp_path / "rfi_est.npz"
    rng = np.random.RandomState(0)
    # f64 throughout: MJD's ~6e4 day offset against second-scale spacing is below
    # f32 resolution, and jnp under --x64 false would collapse every sample onto
    # the same coordinate. The helper has to be as careful as the reader.
    times = np.linspace(0.0, 120.0, N_TIME, dtype=np.float64)
    np.savez(
        path,
        light_curves=np.abs(rng.randn(n_rfi_real, n_time, n_freq)),
        norad_ids=np.array(_norad_ids(n_rfi_real, n_rfi_real)),
        times=_TEST_EPOCH_MJD + times / 86400.0,
        freqs=np.linspace(1.4e9, 1.41e9, n_freq, dtype=np.float64),
        time_scale="utc",
    )
    return str(path)


def latent_attr(comp):
    """Name of the class's latent init attribute (real-space vs Fourier families)."""
    return "init_rfi_k" if hasattr(comp, "init_rfi_k") else "init_rfi_A_induce"


def transform_args(comp):
    """(scale, mean) positional args for the class's forward/inv transform pair."""
    if hasattr(comp, "sigma_rfi_k"):
        return comp.sigma_rfi_k, comp.mu_rfi_k
    return comp.L_rfi_A, comp.mu_rfi_A


def tol():
    """Precision-aware tolerance for comparisons that are not exact-zero."""
    return 1e-8 if active_precision() == "double" else 1e-4


# ---------------------------------------------------------------------------
# rfi_signal_config_validation
# ---------------------------------------------------------------------------

class TestRfiSignalConfigValidation:

    @staticmethod
    def _grid(n_freq=4, n_time=8):
        return (
            jnp.linspace(1.4e9, 1.41e9, n_freq),
            jnp.linspace(0.0, 120.0, n_time),
        )

    def test_null_values_get_defaults(self):
        """All-None config picks up defaults derived from the data and the observation grid."""
        freqs, times = self._grid()
        vis_obs = 3.0 * jnp.ones((6, 4, 8), dtype=complex)
        cfg = {"r_seed": None, "var": None, "corr_freq": None, "corr_time": None}

        result = rfi_signal_config_validation(cfg, vis_obs, freqs, 1e6, times, 8.0)

        assert result["r_seed"] == 1
        assert result["var"] == pytest.approx(3.0)  # max |vis_obs|
        assert result["corr_freq"] == pytest.approx(float(freqs[-1] - freqs[0]) / 2)
        assert result["corr_time"] == pytest.approx(float(times[-1] - times[0]) / 2)

    def test_null_corr_time_does_not_clobber_corr_freq(self):
        """A defaulted corr_time must be written to corr_time, not over corr_freq.

        Regression test: the defaulting branch used to assign the *time* extent to the
        corr_freq key, wiping out the frequency default and leaving corr_time as None.
        """
        freqs, times = self._grid()
        vis_obs = jnp.ones((6, 4, 8), dtype=complex)
        cfg = {"r_seed": 1, "var": 1.0, "corr_freq": None, "corr_time": None}

        result = rfi_signal_config_validation(cfg, vis_obs, freqs, 1e6, times, 8.0)

        freq_extent, time_extent = float(freqs[-1] - freqs[0]), float(times[-1] - times[0])
        assert result["corr_time"] is not None
        assert result["corr_time"] == pytest.approx(time_extent / 2)
        assert result["corr_freq"] == pytest.approx(freq_extent / 2)
        # The two defaults live on wildly different scales; catching a swap matters.
        assert result["corr_freq"] != pytest.approx(result["corr_time"])

    def test_explicit_values_preserved_as_floats(self):
        """Explicit numeric values survive validation and are coerced to float."""
        freqs, times = self._grid()
        vis_obs = jnp.ones((6, 4, 8), dtype=complex)
        cfg = {"r_seed": 42, "var": 7, "corr_freq": 5e6, "corr_time": 60}

        result = rfi_signal_config_validation(cfg, vis_obs, freqs, 1e6, times, 8.0)

        assert result["r_seed"] == 42
        assert isinstance(result["var"], float) and result["var"] == pytest.approx(7.0)
        assert result["corr_freq"] == pytest.approx(5e6)
        assert isinstance(result["corr_time"], float)
        assert result["corr_time"] == pytest.approx(60.0)

    def test_missing_key_raises(self):
        """A config missing one of the four required keys raises ValueError."""
        freqs, times = self._grid()
        vis_obs = jnp.ones((6, 4, 8), dtype=complex)
        cfg = {"r_seed": 1, "var": 1.0, "corr_freq": 5e6}  # no corr_time

        with pytest.raises(ValueError):
            rfi_signal_config_validation(cfg, vis_obs, freqs, 1e6, times, 8.0)

    @pytest.mark.parametrize("key", ["r_seed", "var", "corr_freq", "corr_time"])
    def test_non_numeric_value_raises(self, key):
        """A non-numeric value for any tunable raises ValueError."""
        freqs, times = self._grid()
        vis_obs = jnp.ones((6, 4, 8), dtype=complex)
        cfg = {"r_seed": 1, "var": 1.0, "corr_freq": 5e6, "corr_time": 60.0}
        cfg[key] = "not a number"

        with pytest.raises(ValueError):
            rfi_signal_config_validation(cfg, vis_obs, freqs, 1e6, times, 8.0)

    def test_single_channel_single_integration_defaults(self):
        """With a zero-extent grid the defaults fall back to the step sizes, not zero."""
        freqs, times = jnp.array([1.4e9]), jnp.array([0.0])
        vis_obs = jnp.ones((6, 1, 1), dtype=complex)
        cfg = {"r_seed": None, "var": None, "corr_freq": None, "corr_time": None}

        result = rfi_signal_config_validation(cfg, vis_obs, freqs, 1e6, times, 8.0)

        assert result["corr_freq"] == pytest.approx(1e6 / 2)
        assert result["corr_time"] == pytest.approx(8.0 / 2)


# ---------------------------------------------------------------------------
# BaseGPRFI padding helpers
# ---------------------------------------------------------------------------

class TestPaddingHelpers:
    """The device-sharding helpers on BaseGPRFI, driven through a concrete subclass."""

    def test_mask_dummy_rfi_zeroes_only_padded_rows(self):
        comp = setup_component(ComplexRFIVarAnt)
        arr = jnp.arange(N_RFI * 2 * 3, dtype=float).reshape(N_RFI, 2, 3) + 1.0

        masked = comp._mask_dummy_rfi(arr)

        assert jnp.array_equal(masked[:N_RFI_REAL], arr[:N_RFI_REAL])
        assert jnp.all(masked[N_RFI_REAL:] == 0)

    def test_mask_dummy_rfi_is_noop_when_unpadded(self):
        comp = setup_component(ComplexRFIVarAnt, n_rfi=N_RFI, n_rfi_real=N_RFI)
        arr = jnp.ones((N_RFI, 2, 3))

        assert jnp.array_equal(comp._mask_dummy_rfi(arr), arr)

    @pytest.mark.parametrize("dtype", [float, complex])
    def test_zero_pad_rfi_grows_and_zeroes(self, dtype):
        """A truth/estimate array with only the real sources is padded with exact zeros."""
        comp = setup_component(ComplexRFIVarAnt)
        arr = jnp.ones((N_RFI_REAL, 2, 3), dtype=dtype)

        padded = comp._zero_pad_rfi(arr)

        assert padded.shape == (N_RFI, 2, 3)
        assert padded.dtype == arr.dtype
        assert jnp.array_equal(padded[:N_RFI_REAL], arr)
        assert jnp.all(padded[N_RFI_REAL:] == 0)

    def test_zero_pad_rfi_is_identity_when_already_full(self):
        comp = setup_component(ComplexRFIVarAnt)
        arr = jnp.ones((N_RFI, 2, 3))

        assert comp._zero_pad_rfi(arr) is arr

    def test_n_rfi_real_defaults_to_n_rfi_when_config_lacks_it(self):
        """An unpadded TabConfig has no n_rfi_real; the mask must then be a no-op."""
        comp = setup_component(ComplexRFIVarAnt, with_n_rfi_real=False)

        assert comp.n_rfi_real == comp.n_rfi
        arr = jnp.ones((N_RFI, 2, 3))
        assert jnp.array_equal(comp._mask_dummy_rfi(arr), arr)

    @pytest.mark.parametrize("cls", ALL_CLASSES)
    def test_masked_forward_transform_equals_mask_of_forward(self, cls):
        comp = setup_component(cls)
        scale, mu = transform_args(comp)
        base = jnp.ones(comp.init_params_base[sorted(comp.init_params_base)[0]].shape)

        assert jnp.allclose(
            comp.masked_forward_transform(base, scale, mu),
            comp._mask_dummy_rfi(comp.forward_transform(base, scale, mu)),
        )


# ---------------------------------------------------------------------------
# Shared component contract
# ---------------------------------------------------------------------------

class TestComponentContract:

    @pytest.mark.parametrize("cls", ALL_CLASSES)
    @pytest.mark.parametrize("init", COMMON_INITS)
    def test_setup_succeeds_for_every_init_mode(self, cls, init):
        """setup() and its internal _validate_dimensions pass for all shared init modes."""
        comp = setup_component(cls, init=init)
        assert comp.init_params_base

    @pytest.mark.parametrize("cls", FOURIER_CLASSES)
    @pytest.mark.parametrize("mean", ["zeros", "data"])
    def test_setup_succeeds_for_every_prior_mean(self, cls, mean):
        comp = setup_component(cls, mean=mean)
        assert comp.mu_rfi_k.shape[0] == N_RFI

    @pytest.mark.parametrize("cls", ALL_CLASSES)
    def test_state_output_placeholder(self, cls):
        """The rfi_A placeholder covers the full fine grid and is complex."""
        comp = setup_component(cls)
        placeholder = comp.state_outputs["rfi_A"]

        assert placeholder.shape == (N_RFI, N_ANT, N_FREQ, N_TIME)
        assert jnp.issubdtype(placeholder.dtype, jnp.complexfloating)

    @pytest.mark.parametrize(
        "cls,expected",
        [
            (ComplexRFIVarAnt, {"sigma_rfi_k", "mu_rfi_k"}),
            (ComplexRFIConstAnt, {"sigma_rfi_k", "mu_rfi_k"}),
        ],
    )
    def test_build_constants_keys(self, cls, expected):
        """build_constants supplies exactly the constants build_forward reads back."""
        comp = setup_component(cls)
        assert set(comp.build_constants()) == expected

    @pytest.mark.parametrize("cls", ALL_CLASSES)
    def test_build_set_params_shapes(self, cls):
        """set_params samples every declared base parameter at the right shape."""
        comp = setup_component(cls)
        set_params = comp.build_set_params()

        # standard_normal is numpyro.sample, so a seed handler is mandatory.
        with numpyro.handlers.seed(rng_seed=0):
            params = set_params({})

        assert set(params) == set(comp.parameter_shapes)
        for name, value in params.items():
            assert value.shape == comp.init_params_base[name].shape

    @pytest.mark.parametrize("cls", ALL_CLASSES)
    def test_init_params_base_matches_declared_parameters(self, cls):
        comp = setup_component(cls)
        assert set(comp.init_params_base) == set(comp.parameter_shapes)

    @pytest.mark.parametrize("cls", ALL_CLASSES)
    def test_forward_output(self, cls):
        """rfi_A comes out on the observation grid and is finite."""
        comp = setup_component(cls)
        rfi_A = run_forward(comp, random_params(comp))

        assert rfi_A.shape == (N_RFI, N_ANT, N_FREQ, N_TIME)
        assert jnp.all(jnp.isfinite(jnp.abs(rfi_A)))

    @pytest.mark.parametrize(
        "cls,complex_valued",
        [
            # All surviving components model a complex amplitude. Note the
            # narrows the dtype when it overwrites it.
            (ComplexRFIVarAnt, True),
            (ComplexRFIConstAnt, True),
        ],
    )
    def test_forward_output_dtype(self, cls, complex_valued):
        comp = setup_component(cls)
        rfi_A = run_forward(comp, random_params(comp))

        assert jnp.issubdtype(rfi_A.dtype, jnp.complexfloating) == complex_valued

    @pytest.mark.parametrize("cls", ALL_CLASSES)
    def test_forward_preserves_other_state_keys(self, cls):
        comp = setup_component(cls)
        state = {"some_extra_key": jnp.array(42.0)}
        out = comp.build_forward()(comp.init_params_base, state, make_constants(comp))

        assert "some_extra_key" in out

    @pytest.mark.parametrize("cls", ALL_CLASSES)
    def test_forward_is_jit_compatible(self, cls):
        """build_forward claims purity; jit must compile and agree with eager."""
        comp = setup_component(cls)
        params = random_params(comp)
        constants = make_constants(comp)
        forward = comp.build_forward()

        eager = forward(params, {}, constants)["rfi_A"]
        jitted = jax.jit(forward)(params, {}, constants)["rfi_A"]

        assert jnp.allclose(eager, jitted, atol=tol())

    @pytest.mark.parametrize("cls", ALL_CLASSES)
    def test_invalid_init_type_raises(self, cls):
        """Subclass setup wraps failures, so the surfaced error is a RuntimeError."""
        with pytest.raises(RuntimeError, match="not_an_init_type"):
            setup_component(cls, init="not_an_init_type")

    @pytest.mark.parametrize("cls", FOURIER_CLASSES)
    def test_invalid_prior_mean_raises(self, cls):
        with pytest.raises(RuntimeError, match="not_a_prior_type"):
            setup_component(cls, mean="not_a_prior_type")

    @pytest.mark.parametrize("cls", FOURIER_CLASSES)
    def test_supersampled_fine_grid(self, cls):
        """With n_int_time > 1 the forward returns the supersampled time grid."""
        comp = setup_component(cls, n_int_time=2)
        rfi_A = run_forward(comp, random_params(comp))

        assert rfi_A.shape == (N_RFI, N_ANT, N_FREQ, 2 * N_TIME)


# ---------------------------------------------------------------------------
# The padding contract
# ---------------------------------------------------------------------------

class TestDummySourcesStayDark:
    """Padded dummy satellites must carry zero amplitude and zero gradient.

    These are the assertions that stop a future refactor from quietly undoing the
    masking in ``masked_forward_transform``.
    """

    @pytest.mark.parametrize("cls", ALL_CLASSES)
    @pytest.mark.parametrize("init", COMMON_INITS)
    def test_forward_is_dark_for_arbitrary_params(self, cls, init):
        """rfi_A is exactly zero on the padded rows, whatever the parameters are."""
        comp = setup_component(cls, init=init)
        rfi_A = run_forward(comp, random_params(comp))

        assert jnp.all(rfi_A[N_RFI_REAL:] == 0)
        # ...and the real sources are not accidentally zeroed too.
        assert jnp.max(jnp.abs(rfi_A[:N_RFI_REAL])) > 0

    @pytest.mark.parametrize("cls", ALL_CLASSES)
    @pytest.mark.parametrize("mean", ["zeros", "data"])
    def test_forward_is_dark_for_every_prior_mean(self, cls, mean):
        """A non-zero prior mean on the padded rows must still produce a dark output."""
        comp = setup_component(cls, mean=mean)
        rfi_A = run_forward(comp, random_params(comp))

        assert jnp.all(rfi_A[N_RFI_REAL:] == 0)

    @pytest.mark.parametrize("cls", ALL_CLASSES)
    def test_gradient_vanishes_on_padded_rows(self, cls):
        """Dummy sources receive exactly zero gradient, so the optimiser cannot light them."""
        comp = setup_component(cls, init="sample")
        constants = make_constants(comp)
        forward = comp.build_forward()

        def loss(params):
            return jnp.sum(jnp.abs(forward(params, {}, constants)["rfi_A"]) ** 2)

        grads = jax.grad(loss)(random_params(comp))

        for name, grad in grads.items():
            assert jnp.all(grad[N_RFI_REAL:] == 0), f"{name} has gradient on padded rows"
            assert jnp.max(jnp.abs(grad[:N_RFI_REAL])) > 0, f"{name} has no real gradient"

    @pytest.mark.parametrize("cls", ALL_CLASSES)
    def test_padding_does_not_change_the_real_sources(self, cls):
        """Adding dark dummies leaves the real sources' visibility contribution alone.

        The unit-level analogue of test_pipeline_sharded_equivalence.
        """
        padded = setup_component(cls, n_rfi=N_RFI, n_rfi_real=N_RFI_REAL, init="sample")
        unpadded = setup_component(cls, n_rfi=N_RFI_REAL, n_rfi_real=N_RFI_REAL, init="sample")

        # Same parameters for the real sources; the padded run gets zeros for the dummies.
        shared = random_params(unpadded)
        padded_params = {
            name: jnp.concatenate(
                [value, jnp.zeros((N_RFI - N_RFI_REAL,) + value.shape[1:], value.dtype)]
            )
            for name, value in shared.items()
        }

        padded_out = run_forward(padded, padded_params)
        unpadded_out = run_forward(unpadded, shared)

        assert jnp.allclose(padded_out[:N_RFI_REAL], unpadded_out, atol=tol())

    @pytest.mark.parametrize("cls", FOURIER_CLASSES)
    @pytest.mark.parametrize("mode", ["init", "mean"])
    def test_estimate_file_is_zero_padded(self, cls, mode, tmp_path):
        """_read_estimate loads only the real sources and zero-pads to the sharded count."""
        est = make_est_file(tmp_path)
        kwargs = {"est": est}
        kwargs["init" if mode == "init" else "mean"] = "est"
        comp = setup_component(cls, **kwargs)

        latent = getattr(comp, latent_attr(comp)) if mode == "init" else comp.mu_rfi_k

        assert latent.shape[0] == N_RFI
        assert jnp.all(latent[N_RFI_REAL:] == 0)
        assert jnp.max(jnp.abs(latent[:N_RFI_REAL])) > 0

    def test_data_estimate_splits_over_real_sources_only(self):
        """The data-derived prior mean divides by n_rfi_real, not the padded n_rfi."""
        padded = setup_component(ComplexRFIVarAnt, n_rfi=N_RFI, n_rfi_real=N_RFI_REAL, mean="data")
        unpadded = setup_component(
            ComplexRFIVarAnt, n_rfi=N_RFI_REAL, n_rfi_real=N_RFI_REAL, mean="data"
        )

        # Same per-source share regardless of how many dummies were appended.
        assert jnp.allclose(padded.mu_rfi_k[:N_RFI_REAL], unpadded.mu_rfi_k, atol=tol())


# ---------------------------------------------------------------------------
# Elevation mask
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Seeding from a matched-filter estimate
# ---------------------------------------------------------------------------

class TestMatchedFilterSeeding:
    """``rfi.init``/``rfi.mean: matched-filter`` seeds from the data in memory.

    The imaging route measures the light curves out of band, writes them to a
    file and points ``rfi.est`` at it. ``matched-filter`` estimates the *same*
    curves from the visibilities the run has already loaded, so the two paths
    must agree on the same data -- that equivalence is what says the new one is a
    drop-in for the old, and it is the assertion these tests are built around.
    """

    @staticmethod
    def _estimate(n_rfi_real=N_RFI_REAL, n_freq=N_FREQ, n_time=N_TIME):
        """A light-curve result from the real estimator, not a hand-written dict."""
        from tabascal.rfi_estimate import _lc_result, matched_filter_light_curves

        rng = np.random.RandomState(0)
        n_ant = 4
        a1, a2 = np.triu_indices(n_ant, 1)
        phase = rng.uniform(-np.pi, np.pi, (n_rfi_real, n_ant, n_freq, n_time))
        E = np.exp(1j * phase)
        curves = np.abs(rng.randn(n_rfi_real, n_freq, n_time)) + 0.5
        vis = sum(
            (E[s][a1] * np.conj(E[s][a2])) * curves[s] for s in range(n_rfi_real)
        )
        lc, err = matched_filter_light_curves(vis, phase, a1, a2)

        return _lc_result(
            lc,
            err,
            _norad_ids(n_rfi_real, n_rfi_real),
            np.linspace(1.4e9, 1.41e9, n_freq, dtype=np.float64),
            _TEST_EPOCH_MJD + np.linspace(0.0, 120.0, n_time, dtype=np.float64) / 86400.0,
            "DATA",
            "xx",
        )

    @pytest.fixture
    def estimate(self, monkeypatch, tmp_path):
        """The same estimate reachable both ways: in memory, and through a file.

        ``light_curves_from_config`` is stubbed rather than run: the estimator
        itself is covered by ``tests/test_rfi_estimate.py``, and a real one here
        would need an MS and a SatChecker resolution behind the mock config.
        """
        from tabascal.rfi_estimate import save_light_curves_npz
        import tabascal.rfi_estimate as mod

        result = self._estimate()
        calls = []
        monkeypatch.setattr(
            mod,
            "light_curves_from_config",
            lambda config, **kwargs: (calls.append(config), result)[1],
        )
        path = str(tmp_path / "mf.npz")
        save_light_curves_npz(path, result)

        return SimpleNamespace(result=result, path=path, calls=calls)

    @pytest.mark.parametrize("cls", FOURIER_CLASSES)
    @pytest.mark.parametrize("mode", ["init", "mean"])
    def test_it_agrees_with_the_same_curves_read_from_a_file(
        self, cls, mode, estimate, exact_rtol
    ):
        """The verification the imaging-free path rests on: same data, same seed."""
        key = "init" if mode == "init" else "mean"
        filtered = setup_component(cls, **{key: "matched-filter"})
        from_file = setup_component(cls, est=estimate.path, **{key: "est"})

        attr = latent_attr(filtered) if mode == "init" else "mu_rfi_k"
        got, expected = getattr(filtered, attr), getattr(from_file, attr)

        assert jnp.max(jnp.abs(expected)) > 0
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(expected),
            rtol=exact_rtol, atol=float(jnp.max(jnp.abs(expected))) * exact_rtol,
        )

    @pytest.mark.parametrize("cls", FOURIER_CLASSES)
    @pytest.mark.parametrize("alias", ["matched-filter", "mf"])
    def test_the_alias_selects_the_same_estimate(self, cls, alias, estimate):
        long_form = setup_component(cls, init="matched-filter")
        aliased = setup_component(cls, init=alias)

        np.testing.assert_array_equal(
            np.asarray(aliased.init_rfi_k), np.asarray(long_form.init_rfi_k)
        )

    @pytest.mark.parametrize("cls", FOURIER_CLASSES)
    @pytest.mark.parametrize("mode", ["init", "mean"])
    def test_the_padded_sources_stay_dark(self, cls, mode, estimate):
        """The estimator only knows the real satellites; the dummies stay at zero."""
        key = "init" if mode == "init" else "mean"
        comp = setup_component(cls, **{key: "matched-filter"})

        latent = getattr(comp, latent_attr(comp)) if mode == "init" else comp.mu_rfi_k

        assert latent.shape[0] == N_RFI
        assert jnp.all(latent[N_RFI_REAL:] == 0)
        assert jnp.max(jnp.abs(latent[:N_RFI_REAL])) > 0

    def test_the_antenna_axis_follows_the_model(self, estimate):
        """VarAnt carries one amplitude per antenna; ConstAnt shares one."""
        assert setup_component(ComplexRFIVarAnt, init="mf").init_rfi_k.shape[1] == N_ANT
        assert setup_component(ComplexRFIConstAnt, init="mf").init_rfi_k.shape[1] == 1

    def test_the_run_config_is_what_gets_filtered(self, estimate):
        """The estimate is made from this run's own visibilities, not a file."""
        config = make_rfi_config(init="matched-filter")
        comp = ComplexRFIVarAnt()
        comp.setup(config)

        assert estimate.calls and estimate.calls[0] is config

    @pytest.mark.parametrize("cls", FOURIER_CLASSES)
    def test_an_unmeasured_sample_seeds_at_zero(self, cls, monkeypatch):
        """A fully flagged cell comes back nan, and nan must not reach the latent."""
        import tabascal.rfi_estimate as mod

        result = self._estimate()
        result["light_curves"][0, 0, 0] = np.nan
        monkeypatch.setattr(
            mod, "light_curves_from_config", lambda config, **kwargs: result
        )

        comp = setup_component(cls, init="matched-filter")

        assert jnp.all(jnp.isfinite(comp.init_rfi_k))

    @pytest.mark.parametrize("cls", FOURIER_CLASSES)
    def test_no_estimate_file_is_needed(self, cls, estimate):
        """`rfi.est` stays null: that is the point of the option."""
        comp = setup_component(cls, init="matched-filter", est=None)

        assert jnp.max(jnp.abs(comp.init_rfi_k)) > 0

    @pytest.mark.parametrize("cls", FOURIER_CLASSES)
    @pytest.mark.parametrize("key", ["init", "mean"])
    def test_the_option_is_named_in_the_error_for_a_bad_value(self, cls, key):
        with pytest.raises(RuntimeError, match="matched-filter"):
            setup_component(cls, **{key: "nonsense"})


def elevation_mask(n_rfi=N_RFI, n_time_fine=N_TIME, down_from=N_TIME // 2):
    """A mask putting source 0 below the cut from ``down_from`` on; the rest always up.

    Shaped like ``TabConfig.rfi_mask_fine`` — (n_rfi, n_time_fine), already expanded
    over each integration so a sample is never partially masked.
    """
    mask = np.ones((n_rfi, n_time_fine), dtype=bool)
    mask[0, down_from:] = False
    return mask


class TestElevationMask:
    """rfi.min_elevation zeroes a satellite's signal while it is below the cut.

    The mask is applied in the forward, so — like the padding mask it composes with —
    it holds for arbitrary parameters, not just the init values.
    """

    @pytest.mark.parametrize("cls", ALL_CLASSES)
    def test_masked_samples_are_exactly_zero(self, cls):
        comp = setup_component(cls, rfi_mask_fine=elevation_mask())
        rfi_A = run_forward(comp, random_params(comp))

        # Exactly zero, not merely small: the mask multiplies the signal by 0.
        assert jnp.max(jnp.abs(rfi_A[0, :, :, N_TIME // 2:])) == 0.0

    @pytest.mark.parametrize("cls", ALL_CLASSES)
    def test_in_view_samples_are_untouched(self, cls):
        """Masking is confined to the out-of-view samples of the masked source."""
        masked = setup_component(cls, rfi_mask_fine=elevation_mask())
        plain = setup_component(cls)
        params = random_params(plain)

        got, want = run_forward(masked, params), run_forward(plain, params)

        assert jnp.allclose(got[0, :, :, : N_TIME // 2], want[0, :, :, : N_TIME // 2])
        # Sources that never drop below the cut are unaffected everywhere.
        assert jnp.allclose(got[1:N_RFI_REAL], want[1:N_RFI_REAL])

    @pytest.mark.parametrize("cls", ALL_CLASSES)
    def test_no_mask_leaves_the_forward_unchanged(self, cls):
        """min_elevation: null must be a true no-op, not a multiply by ones."""
        comp = setup_component(cls)

        assert comp.rfi_mask_fine is None
        assert "rfi_mask_fine" not in comp.build_constants()

    @pytest.mark.parametrize("cls", ALL_CLASSES)
    def test_mask_composes_with_dummy_padding(self, cls):
        """Both masks apply: padded rows stay dark and the masked source still cuts."""
        comp = setup_component(cls, rfi_mask_fine=elevation_mask())
        rfi_A = run_forward(comp, random_params(comp))

        assert jnp.all(rfi_A[N_RFI_REAL:] == 0), "padded dummy sources are not dark"
        assert jnp.max(jnp.abs(rfi_A[0, :, :, N_TIME // 2:])) == 0.0
        # The masking must not have swallowed the signal wholesale.
        assert jnp.max(jnp.abs(rfi_A[:N_RFI_REAL])) > 0

    @pytest.mark.parametrize("cls", ALL_CLASSES)
    def test_mask_is_expanded_over_integrations(self, cls):
        """A mask on the fine grid masks whole integrations, never part of one."""
        n_int_time = 2
        comp = setup_component(
            cls,
            n_int_time=n_int_time,
            rfi_mask_fine=elevation_mask(n_time_fine=N_TIME * n_int_time,
                                         down_from=N_TIME),
        )
        rfi_A = run_forward(comp, random_params(comp))

        assert rfi_A.shape[-1] == N_TIME * n_int_time
        assert jnp.max(jnp.abs(rfi_A[0, :, :, N_TIME:])) == 0.0
        assert jnp.max(jnp.abs(rfi_A[0, :, :, :N_TIME])) > 0


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class TestTransforms:

    @pytest.mark.parametrize("cls", FOURIER_CLASSES)
    def test_fourier_roundtrip(self, cls):
        """The Fourier transform pair is a plain scale-and-shift, so it inverts exactly."""
        comp = setup_component(cls, mean="data")
        shape = comp.init_params_base["rfi_k_r_base"].shape
        k1, k2 = jax.random.split(jax.random.PRNGKey(42))
        base = jax.random.normal(k1, shape) + 1j * jax.random.normal(k2, shape)

        assert_transform_roundtrip(comp, base, comp.sigma_rfi_k, comp.mu_rfi_k, atol=tol())


# ---------------------------------------------------------------------------
# Scanned latent-to-signal transform
# ---------------------------------------------------------------------------

class TestFourierScanTransform:
    """ComplexRFIVarAnt scans the antenna axis instead of vmapping it.

    That is a pure implementation change made to bound the cuFFT plan work area (a
    batched transform over n_rfi * n_ant asked for 12.6 GiB at 32 channels and
    aborted inside XLA). These pin the result to the vmap semantics it replaced, in
    both value and gradient -- ``checkpoint`` alters how the tape is built, so a bug
    there would surface only in reverse mode.
    """

    @staticmethod
    def _vmap_reference(comp, params):
        """rfi_A computed the way the pre-scan implementation did."""
        rfi_k_A_base = params["rfi_k_r_base"] + 1.0j * params["rfi_k_i_base"]
        rfi_k_A = comp.masked_forward_transform(
            rfi_k_A_base, comp.sigma_rfi_k, comp.mu_rfi_k
        )
        return vmap(
            vmap(latent_to_signal, (0, None, None), 0), (1, None, None), 1
        )(rfi_k_A, comp.pads, comp.ss_idxs)

    def test_matches_vmap_reference(self):
        comp = setup_component(ComplexRFIVarAnt)
        params = random_params(comp)

        got = run_forward(comp, params)
        expected = self._vmap_reference(comp, params)

        assert got.shape == expected.shape
        assert jnp.allclose(got, expected, atol=tol(), rtol=tol())

    def test_gradients_match_vmap_reference(self):
        comp = setup_component(ComplexRFIVarAnt)
        params = random_params(comp)
        constants = make_constants(comp)

        def scanned(p):
            out = comp.build_forward()(p, {}, constants)["rfi_A"]
            return jnp.sum(jnp.abs(out) ** 2)

        def reference(p):
            return jnp.sum(jnp.abs(self._vmap_reference(comp, p)) ** 2)

        g_scan = jax.grad(scanned)(params)
        g_ref = jax.grad(reference)(params)

        for name in g_ref:
            assert jnp.allclose(g_scan[name], g_ref[name], atol=tol(), rtol=tol()), (
                f"gradient mismatch for {name}"
            )

    def test_const_ant_broadcast_matches_full_grid(self):
        """The shared-antenna signal is identical across antennas after broadcast."""
        comp = setup_component(ComplexRFIConstAnt)
        rfi_A = run_forward(comp, random_params(comp))

        assert rfi_A.shape == (N_RFI, N_ANT, N_FREQ, N_TIME)
        # Every antenna carries the same signal -- that is what "ConstAnt" means, and
        # it is the property the broadcast has to preserve.
        for ant in range(1, N_ANT):
            assert jnp.array_equal(rfi_A[:, 0], rfi_A[:, ant])


# ---------------------------------------------------------------------------
# Class-specific behaviour
# ---------------------------------------------------------------------------

class TestClassSpecifics:

    @pytest.mark.parametrize("cls", FOURIER_CLASSES)
    def test_fourier_init_params_split_real_and_imaginary(self, cls):
        comp = setup_component(cls)
        assert set(comp.init_params) == {"rfi_k_r", "rfi_k_i"}

    def test_const_ant_has_singleton_antenna_axis(self):
        """ComplexRFIConstAnt shares one latent across antennas."""
        comp = setup_component(ComplexRFIConstAnt, init="sample")

        assert comp.mu_rfi_k.shape == (N_RFI, 1, comp.n_k_freq_rfi, comp.n_k_time_rfi)
        for value in comp.init_params_base.values():
            assert value.shape == (N_RFI, 1, comp.n_k_freq_rfi, comp.n_k_time_rfi)

    @pytest.mark.parametrize("init", COMMON_INITS)
    def test_const_ant_forward_is_identical_across_antennas(self, init):
        """The defining property: every antenna sees the same RFI amplitude."""
        comp = setup_component(ComplexRFIConstAnt, init=init)
        rfi_A = run_forward(comp, random_params(comp))

        assert rfi_A.shape == (N_RFI, N_ANT, N_FREQ, N_TIME)
        assert jnp.allclose(rfi_A, rfi_A[:, :1], atol=tol())

    def test_variable_ant_forward_differs_across_antennas(self):
        """ComplexRFIVarAnt, by contrast, gives each antenna its own amplitude."""
        comp = setup_component(ComplexRFIVarAnt, init="sample")
        rfi_A = run_forward(comp, random_params(comp))

        assert not jnp.allclose(rfi_A[:N_RFI_REAL], rfi_A[:N_RFI_REAL, :1])


# ---------------------------------------------------------------------------
# Multi-device
# ---------------------------------------------------------------------------

_MULTI_DEVICE_SCRIPT = textwrap.dedent(
    """
    import jax
    import jax.numpy as jnp
    from types import SimpleNamespace

    from tabascal.components.rfi_signal import ComplexRFIVarAnt

    assert jax.device_count() == 2, jax.device_count()

    n_rfi, n_rfi_real, n_ant, n_freq, n_time = 4, 3, 3, 4, 8
    freqs = jnp.linspace(1.4e9, 1.41e9, n_freq)
    times = jnp.linspace(0.0, 120.0, n_time)

    config = SimpleNamespace(
        norad_ids=[40000, 40001, 40002, 40002],
        n_rfi=n_rfi, n_rfi_real=n_rfi_real, n_ant=n_ant, n_freq=n_freq, n_time=n_time,
        n_freq_fine=n_freq, n_time_fine=n_time, n_int_freq=1, n_int_time=1,
        freqs=freqs, freqs_fine=freqs, chan_width=float(freqs[1] - freqs[0]),
        times=times, times_fine=times, int_time=float(times[1] - times[0]),
    times_jd=2460000.5 + times / 86400.0,
    times_mjd=60000.0 + times / 86400.0,
    time_scale="utc",
        vis_obs=jnp.ones((3, n_freq, n_time), dtype=complex),
        args={
            "rfi": {"r_seed": 1, "var": 1.0, "corr_freq": 5e6, "corr_time": 60.0,
                    "init": "sample", "mean": "zeros", "est": None,
                    "time_pad_factor": 2, "freq_pad_factor": 2},
            "plots": {"truth": False},
            "data": {"zarr_path": None, "data_col": "DATA"},
        },
    )

    comp = ComplexRFIVarAnt()
    comp.setup(config)

    # The placeholder is allocated per-shard, never as a full single-device array.
    placeholder = comp.state_outputs["rfi_A"]
    assert placeholder.shape == (n_rfi, n_ant, n_freq, n_time), placeholder.shape
    assert len(placeholder.sharding.device_set) == 2, placeholder.sharding

    constants = {f"{comp.prefix}/{k}": v for k, v in comp.build_constants().items()}
    keys = jax.random.split(jax.random.PRNGKey(0), len(comp.init_params_base))
    params = {
        name: jax.random.normal(key, comp.init_params_base[name].shape)
        for name, key in zip(sorted(comp.init_params_base), keys)
    }

    rfi_A = comp.build_forward()(params, {}, constants)["rfi_A"]
    assert jnp.all(rfi_A[n_rfi_real:] == 0), "padded sources are not dark"
    assert jnp.max(jnp.abs(rfi_A[:n_rfi_real])) > 0, "real sources are dark"

    print("MULTI_DEVICE_OK")
    """
)


def test_multi_device_padded_sources_dark():
    """Padded sources stay dark with the RFI axis genuinely split across devices."""
    env = {
        **os.environ,
        "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
        "JAX_PLATFORMS": "cpu",
    }
    env.pop("CUDA_VISIBLE_DEVICES", None)

    result = subprocess.run(
        [sys.executable, "-c", _MULTI_DEVICE_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
    )

    assert "MULTI_DEVICE_OK" in result.stdout, (
        f"returncode={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# read_light_curves
# ---------------------------------------------------------------------------

class TestReadLightCurves:
    """The light-curve interchange format: strict, id-matched, and resampled.

    Every loose alternative this could accept instead fails silently. Matching
    rows by position attaches a curve to the wrong satellite without changing its
    shape; assuming the file's sampling matches the observation resamples it
    wrongly by an unknown amount. Neither shows up as an error, only as a worse
    fit — so the assertions here are what distinguish correct from plausible.
    """

    MJD0 = 60000.0
    F0, F1 = 1.0e9, 1.1e9

    def _times(self, n=5):
        return self.MJD0 + np.arange(n) / 86400.0

    def _freqs(self, n=3):
        return np.linspace(self.F0, self.F1, n)

    def _npz(self, tmp_path, labels, times=None, freqs=None, curves=None,
             name="est.npz", time_scale="utc", **extra):
        """A light-curve .npz. ``time_scale=None`` writes an untagged legacy file."""
        times = self._times() if times is None else times
        freqs = self._freqs() if freqs is None else freqs
        if curves is None:
            # Source i is a constant i+1, so a row's value identifies its source.
            curves = np.stack(
                [np.full((len(times), len(freqs)), i + 1.0) for i in range(len(labels))]
            )
        path = tmp_path / name
        fields = {"light_curves": curves, "norad_ids": np.array(labels),
                  "times": np.asarray(times), "freqs": np.asarray(freqs)}
        if time_scale is not None:
            fields["time_scale"] = time_scale
        fields.update(extra)
        np.savez(path, **fields)
        return str(path)

    def _zarr(self, tmp_path, labels, times=None, freqs=None, curves=None,
              time_scale="utc"):
        import xarray as xr

        times = self._times() if times is None else times
        freqs = self._freqs() if freqs is None else freqs
        if curves is None:
            curves = np.stack(
                [np.full((len(times), len(freqs)), i + 1.0) for i in range(len(labels))]
            )
        path = str(tmp_path / "est.zarr")
        xr.Dataset(
            {"light_curves": (("norad_ids", "times", "freqs"), curves)},
            coords={"norad_ids": np.array(labels), "times": np.asarray(times),
                    "freqs": np.asarray(freqs)},
            attrs={} if time_scale is None else {"time_scale": time_scale},
        ).to_zarr(path)
        return path

    # --- id matching ---

    def test_rows_are_reordered_to_match_norad_ids(self, tmp_path):
        path = self._npz(tmp_path, [300, 100, 200])
        out = np.asarray(
            read_light_curves(path, [200, 300, 100], self._times(), self._freqs())
        )
        assert out.shape == (3, 3, 5)          # (n_rfi, n_freq, n_time)
        assert out[0].max() == 3.0             # 200 -> file row 2
        assert out[1].max() == 1.0             # 300 -> file row 0
        assert out[2].max() == 2.0             # 100 -> file row 1

    def test_non_satellite_labels_are_dropped(self, tmp_path):
        path = self._npz(tmp_path, ["100", "Fornax A", "200"])
        out = np.asarray(
            read_light_curves(path, [100, 200], self._times(), self._freqs())
        )
        assert out[0].max() == 1.0 and out[1].max() == 3.0

    def test_a_file_matching_nothing_raises(self, tmp_path):
        path = self._npz(tmp_path, [100, 200])
        with pytest.raises(ValueError, match="matches any configured satellite"):
            read_light_curves(path, [900, 901], self._times(), self._freqs())

    # --- partial coverage ---

    def test_unmatched_satellites_are_zero(self, tmp_path, capsys):
        path = self._npz(tmp_path, [100, 300])
        out = np.asarray(
            read_light_curves(path, [100, 200, 300], self._times(), self._freqs())
        )
        assert out[0].max() == 1.0
        assert np.all(out[1] == 0.0)
        assert out[2].max() == 2.0
        warning = capsys.readouterr().out
        assert "200" in warning and "zero" in warning

    def test_zero_filling_leaves_the_matched_rows_intact(self, tmp_path):
        path = self._npz(tmp_path, [300, 100])
        full = np.asarray(
            read_light_curves(path, [100, 300], self._times(), self._freqs())
        )
        partial = np.asarray(
            read_light_curves(path, [100, 999, 300], self._times(), self._freqs())
        )
        np.testing.assert_array_equal(partial[0], full[0])
        np.testing.assert_array_equal(partial[2], full[1])
        assert np.all(partial[1] == 0.0)

    # --- resampling ---

    def test_an_exact_grid_is_returned_unchanged(self, tmp_path):
        times, freqs = self._times(), self._freqs()
        curves = np.arange(1 * 5 * 3, dtype=float).reshape(1, 5, 3)
        path = self._npz(tmp_path, [100], curves=curves)

        out = np.asarray(read_light_curves(path, [100], times, freqs))
        # Returned as (n_rfi, n_freq, n_time), so the source is transposed.
        np.testing.assert_allclose(out[0], curves[0].T, rtol=1e-6)

    def test_time_is_interpolated_onto_the_observation_grid(self, tmp_path):
        # A ramp in time: interpolating halfway between samples must give halves.
        src_times = self.MJD0 + np.arange(3) / 86400.0
        curves = np.array([[[0.0], [10.0], [20.0]]])          # (1, 3, 1)
        path = self._npz(tmp_path, [100], times=src_times, freqs=[self.F0],
                         curves=curves)

        dst = self.MJD0 + np.array([0.0, 0.5, 1.0, 1.5, 2.0]) / 86400.0
        out = np.asarray(read_light_curves(path, [100], dst, [self.F0]))
        np.testing.assert_allclose(out[0, 0], [0.0, 5.0, 10.0, 15.0, 20.0], rtol=1e-5)

    def test_frequency_is_interpolated_too(self, tmp_path):
        curves = np.array([[[0.0, 20.0]]])                    # (1, 1, 2)
        path = self._npz(tmp_path, [100], times=[self.MJD0],
                         freqs=[1.0e9, 1.2e9], curves=curves)

        out = np.asarray(
            read_light_curves(path, [100], [self.MJD0], [1.0e9, 1.1e9, 1.2e9])
        )
        np.testing.assert_allclose(out[0, :, 0], [0.0, 10.0, 20.0], rtol=1e-5)

    def test_samples_outside_the_files_coverage_are_zero(self, tmp_path):
        src_times = self.MJD0 + np.arange(3) / 86400.0
        path = self._npz(tmp_path, [100], times=src_times, freqs=[self.F0],
                         curves=np.full((1, 3, 1), 7.0))

        # Ask either side of the measured window as well as inside it.
        dst = self.MJD0 + np.array([-5.0, 1.0, 9.0]) / 86400.0
        out = np.asarray(read_light_curves(path, [100], dst, [self.F0]))
        assert out[0, 0, 0] == 0.0        # before the file starts
        assert out[0, 0, 1] == 7.0        # inside
        assert out[0, 0, 2] == 0.0        # after it ends

    def test_out_of_band_frequencies_are_zero(self, tmp_path):
        path = self._npz(tmp_path, [100], freqs=[1.0e9, 1.1e9],
                         curves=np.full((1, 5, 2), 4.0))
        out = np.asarray(
            read_light_curves(path, [100], self._times(), [0.5e9, 1.05e9, 2.0e9])
        )
        assert np.all(out[0, 0] == 0.0)
        assert np.all(out[0, 1] == 4.0)
        assert np.all(out[0, 2] == 0.0)

    def test_a_single_sample_axis_is_held_constant(self, tmp_path):
        """One sample carries no gradient, so it is a value, not a zero-width band."""
        path = self._npz(tmp_path, [100], freqs=[1.05e9],
                         curves=np.full((1, 5, 1), 3.0))
        out = np.asarray(
            read_light_curves(path, [100], self._times(), [0.5e9, 1.05e9, 2.0e9])
        )
        assert np.all(out[0] == 3.0), "a single-frequency curve was zeroed out of band"

    def test_nans_become_a_zero_estimate(self, tmp_path):
        curves = np.ones((1, 5, 3))
        curves[0, :2] = np.nan
        path = self._npz(tmp_path, [100], curves=curves)

        out = np.asarray(read_light_curves(path, [100], self._times(), self._freqs()))
        assert np.all(np.isfinite(out))
        assert np.all(out[0, :, :2] == 0.0)

    def test_non_monotonic_sampling_raises(self, tmp_path):
        times = self.MJD0 + np.array([0.0, 2.0, 1.0]) / 86400.0
        path = self._npz(tmp_path, [100], times=times,
                         curves=np.ones((1, 3, 3)))
        with pytest.raises(ValueError, match="strictly increasing"):
            read_light_curves(path, [100], self._times(), self._freqs())

    # --- the scale the file states its times are on ---

    def test_a_file_stating_utc_reads_without_comment(self, tmp_path, recwarn):
        """The stamp the current writer leaves; nothing to say about it."""
        path = self._npz(tmp_path, [100])

        read_light_curves(path, [100], self._times(), self._freqs())

        assert not [w for w in recwarn if issubclass(w.category, UserWarning)]

    @pytest.mark.parametrize(
        "declared", ["UTC", " utc ", b"utc", b"UTC", np.array(["utc"])]
    )
    def test_the_stated_scale_is_read_leniently(self, tmp_path, declared, recwarn):
        """Case, surrounding space and storage type are not a different scale.

        A stamp comes back as whatever the writer stored: npz keeps a scalar as
        a 0-d array and a byte string as ``|S``, and a zarr attribute may be
        either. Compared with a bare ``str()`` those render as ``"b'utc'"`` or
        ``"['utc']"`` and a perfectly good UTC file is refused as being on a
        scale nobody wrote.
        """
        path = self._npz(tmp_path, [100], time_scale=declared)

        read_light_curves(path, [100], self._times(), self._freqs())

        assert not [w for w in recwarn if issubclass(w.category, UserWarning)]

    @pytest.mark.parametrize("declared", ["tai", "tt", "ut1", b"tai"])
    def test_a_file_on_another_scale_is_rejected_by_name(self, tmp_path, declared):
        """Not converted here: the reader cannot know what a third party meant.

        A file that states TAI is self-describing enough to be fixed at the
        source, and converting it silently would make the reader the second
        place the format's scale is decided. Named in the message as text, so a
        byte-string stamp reads as ``'tai'`` rather than as ``"b'tai'"``.
        """
        path = self._npz(tmp_path, [100], time_scale=declared)

        with pytest.raises(ValueError, match="time_scale") as excinfo:
            read_light_curves(path, [100], self._times(), self._freqs())
        message = str(excinfo.value)
        expected = declared.decode() if isinstance(declared, bytes) else declared
        assert f"'{expected}'" in message
        assert "UTC" in message or "utc" in message

    def test_an_untagged_file_warns_that_it_is_being_read_as_utc(self, tmp_path):
        """Legacy files pre-date the stamp, and one of them may be 37 s out.

        Before the scale was stated, `tabascal light-curve` wrote whatever the
        measurement set's TIME column declared. One written from a UTC MS -- the
        common case -- is already right; one written from a TAI MS is offset by
        the leap seconds and cannot be told apart from it. A warning, not an
        error: refusing every legacy file would break the runs that are correct.
        """
        path = self._npz(tmp_path, [100], time_scale=None)

        with pytest.warns(UserWarning, match="time_scale") as record:
            out = np.asarray(
                read_light_curves(path, [100], self._times(), self._freqs())
            )

        message = str(record[0].message)
        assert "UTC" in message
        assert "regenerate" in message or "regenerated" in message
        # And it still reads, on the assumption it just stated.
        assert out[0].max() == 1.0

    def test_an_untagged_zarr_warns_too(self, tmp_path):
        """Both branches carry the tag, so both must notice its absence."""
        path = self._zarr(tmp_path, [100], time_scale=None)

        with pytest.warns(UserWarning, match="time_scale"):
            read_light_curves(path, [100], self._times(), self._freqs())

    def test_a_zarr_attribute_states_the_scale(self, tmp_path, recwarn):
        """The zarr form carries it as a store attribute, not as a variable."""
        path = self._zarr(tmp_path, [100])

        read_light_curves(path, [100], self._times(), self._freqs())

        assert not [w for w in recwarn if issubclass(w.category, UserWarning)]

    def test_a_zarr_on_another_scale_is_rejected_too(self, tmp_path):
        path = self._zarr(tmp_path, [100], time_scale="tai")

        with pytest.raises(ValueError, match="time_scale"):
            read_light_curves(path, [100], self._times(), self._freqs())

    # --- zarr ---

    def test_a_zarr_store_reads_like_an_npz(self, tmp_path):
        curves = np.arange(2 * 5 * 3, dtype=float).reshape(2, 5, 3)
        zarr_path = self._zarr(tmp_path, [100, 200], curves=curves)
        npz_path = self._npz(tmp_path, [100, 200], curves=curves)

        args = ([200, 100], self._times(), self._freqs())
        np.testing.assert_allclose(
            np.asarray(read_light_curves(zarr_path, *args)),
            np.asarray(read_light_curves(npz_path, *args)),
            rtol=1e-6,
        )

    def test_a_zarr_missing_a_variable_raises(self, tmp_path):
        import xarray as xr

        path = str(tmp_path / "bad.zarr")
        xr.Dataset(
            {"light_curves": (("norad_ids", "times"), np.ones((2, 5)))},
            coords={"norad_ids": np.array([100, 200]), "times": self._times()},
        ).to_zarr(path)
        with pytest.raises(ValueError, match="missing"):
            read_light_curves(path, [100], self._times(), self._freqs())

    def test_the_zarr_store_is_closed_after_reading(self, tmp_path, monkeypatch):
        """A read store holds its backing files open until it is closed.

        Harmless for a plain directory store on POSIX, but a zipped or
        consolidated store keeps a real file handle, and the reader is called
        once per run — inside long-lived processes that also open measurement
        sets. The npz branch was closed in #174; this is the same leak.
        """
        import xarray as xr

        from tabascal.components import rfi_signal

        path = self._zarr(tmp_path, [100, 200])
        opened = []
        real_open_zarr = xr.open_zarr

        def spy(*args, **kwargs):
            xds = real_open_zarr(*args, **kwargs)
            opened.append(xds)
            return xds

        monkeypatch.setattr(rfi_signal.xr, "open_zarr", spy)
        read_light_curves(path, [100, 200], self._times(), self._freqs())

        assert opened, "the zarr branch did not open a store"
        # xarray drops the store's close callable when the Dataset is closed, so
        # a still-set _close is an unclosed store. Checked against a live one so
        # the probe cannot pass by simply never being set.
        with real_open_zarr(path) as control:
            assert control._close is not None, "xarray no longer sets _close"
        assert all(xds._close is None for xds in opened), "the zarr store was left open"

    # --- strictness ---

    @pytest.mark.parametrize("drop", ["norad_ids", "times", "freqs"])
    def test_an_npz_missing_a_required_array_raises(self, tmp_path, drop):
        fields = {
            "light_curves": np.ones((2, 5, 3)),
            "norad_ids": np.array([100, 200]),
            "times": self._times(),
            "freqs": self._freqs(),
        }
        del fields[drop]
        path = tmp_path / "est.npz"
        np.savez(path, **fields)

        with pytest.raises(ValueError, match="missing") as excinfo:
            read_light_curves(str(path), [100], self._times(), self._freqs())
        assert drop in str(excinfo.value)

    def test_a_bare_npy_is_rejected(self, tmp_path):
        """The old positional format is no longer accepted."""
        path = tmp_path / "est.npy"
        np.save(path, np.ones((2, 5, 3)))
        with pytest.raises(ValueError, match="bare .npy"):
            read_light_curves(str(path), [100, 200], self._times(), self._freqs())

    def test_a_complex_light_curve_array_is_rejected(self, tmp_path):
        """Casting to f64 would keep Re(S) and drop Im(S), with only a warning.

        A complex file is the plausible mistake — it is what the matched filter
        natively produces — and the truncation is not visible in the result: the
        curve stays the right shape and is merely wrong by the phase, which for
        an uncalibrated column is most of the signal.
        """
        curves = np.full((1, 5, 3), 3.0 + 4.0j)
        path = self._npz(tmp_path, [100], curves=curves)

        with pytest.raises(ValueError, match="complex") as excinfo:
            read_light_curves(path, [100], self._times(), self._freqs())
        message = str(excinfo.value)
        assert "light_curves" in message
        assert "light_curves_complex" in message, "the error does not name the fix"

    def test_a_complex_zarr_is_rejected_too(self, tmp_path):
        """Both branches load the same array, so both must reject it."""
        curves = np.full((1, 5, 3), 3.0 + 4.0j)
        path = self._zarr(tmp_path, [100], curves=curves)

        with pytest.raises(ValueError, match="complex"):
            read_light_curves(path, [100], self._times(), self._freqs())

    def test_the_magnitude_is_read_with_the_complex_estimate_riding_along(
        self, tmp_path
    ):
        """The rfi.est format save_light_curves_npz writes: |S| plus the extra."""
        complex_curves = np.full((1, 5, 3), 3.0 + 4.0j)
        path = self._npz(
            tmp_path, [100], curves=np.abs(complex_curves),
            light_curves_complex=complex_curves,
        )

        out = np.asarray(read_light_curves(path, [100], self._times(), self._freqs()))
        assert np.allclose(out, 5.0), "the magnitude was not read, or the extra was"

    def test_a_shape_disagreement_raises(self, tmp_path):
        path = self._npz(tmp_path, [100, 200], curves=np.ones((2, 4, 3)))
        with pytest.raises(ValueError, match="imply"):
            read_light_curves(path, [100], self._times(), self._freqs())

    def test_an_npz_named_npy_is_still_read_as_an_npz(self, tmp_path):
        """Detection is by what np.load returned, not by the file extension."""
        import os

        written = self._npz(tmp_path, [100, 200])
        path = str(tmp_path / "est.npy")
        os.rename(written, path)

        out = np.asarray(
            read_light_curves(path, [200, 100], self._times(), self._freqs())
        )
        assert out[0].max() == 2.0 and out[1].max() == 1.0

    def test_realistic_mjd_sampling_survives_single_precision(self, tmp_path):
        """MJD spacing is far below f32 resolution, so interpolation must be f64.

        A real observation samples seconds (~1e-5 days) against an MJD offset of
        ~6e4 days. In f32 every sample collapses onto the same coordinate and the
        interpolation silently returns a constant instead of the light curve —
        which looks entirely plausible downstream. tabascal defaults to f32, so
        this is the configuration that matters.
        """
        n = 64
        src_times = 60123.45 + np.arange(n) * 2.0 / 86400.0   # 2 s cadence
        ramp = np.linspace(0.0, 100.0, n)
        path = self._npz(tmp_path, [100], times=src_times, freqs=[self.F0],
                         curves=ramp.reshape(1, n, 1))

        out = np.asarray(read_light_curves(path, [100], src_times, [self.F0]))
        np.testing.assert_allclose(out[0, 0], ramp, rtol=1e-5)
        assert np.ptp(out[0, 0]) > 99.0, "the time axis collapsed to a constant"

    # --- boundary precision and axis order (from Codex review of #116) ---

    def test_an_mjd_grid_keeps_its_endpoints(self, tmp_path):
        """Endpoints must survive a grid that is only bit-for-bit *nearly* the same.

        Two grids meant to be identical rarely are to the last bit. Without a
        boundary tolerance the first or last destination sample falls just
        outside the source range and np.interp replaces a measured endpoint with
        zero — silently, since every other sample is fine.
        """
        n = 64
        src = 60123.45 + np.arange(n) * 2.0 / 86400.0
        path = self._npz(tmp_path, [100], times=src, freqs=[self.F0],
                         curves=np.full((1, n, 1), 5.0))

        # Perturb by ~1e-10 days, the scale of a JD round trip, outwards at both ends.
        dst = src.copy()
        dst[0] -= 2.5e-10
        dst[-1] += 2.5e-10

        out = np.asarray(read_light_curves(path, [100], dst, [self.F0]))
        assert out[0, 0, 0] == 5.0, "first sample was zeroed by a sub-microsecond shift"
        assert out[0, 0, -1] == 5.0, "last sample was zeroed by a sub-microsecond shift"

    def test_a_genuinely_out_of_range_sample_is_still_zero(self, tmp_path):
        """The tolerance must not swallow a sample that is really outside."""
        n = 8
        src = 60123.45 + np.arange(n) * 2.0 / 86400.0
        path = self._npz(tmp_path, [100], times=src, freqs=[self.F0],
                         curves=np.full((1, n, 1), 5.0))

        # One full sample beyond each end, far outside any tolerance.
        dst = np.array([src[0] - 2.0 / 86400.0, src[3], src[-1] + 2.0 / 86400.0])
        out = np.asarray(read_light_curves(path, [100], dst, [self.F0]))
        assert out[0, 0, 0] == 0.0 and out[0, 0, 2] == 0.0
        assert out[0, 0, 1] == 5.0

    def test_a_zarr_with_reordered_dimensions_is_transposed(self, tmp_path):
        """Axes are identified by name, so a differently-ordered store still reads.

        With equal-length axes a raw .data read would pass the shape check and
        silently interpret time as frequency.
        """
        import xarray as xr

        n = 3  # every axis the same length, so shape checking cannot catch a swap
        labels = [100, 200, 300]
        times = self.MJD0 + np.arange(n) / 86400.0
        freqs = np.linspace(self.F0, self.F1, n)
        curves = np.arange(n ** 3, dtype=float).reshape(n, n, n)

        attrs = {"time_scale": "utc"}
        canonical = str(tmp_path / "canonical.zarr")
        xr.Dataset(
            {"light_curves": (("norad_ids", "times", "freqs"), curves)},
            coords={"norad_ids": np.array(labels), "times": times, "freqs": freqs},
            attrs=attrs,
        ).to_zarr(canonical)

        swapped = str(tmp_path / "swapped.zarr")
        xr.Dataset(
            {"light_curves": (("times", "freqs", "norad_ids"),
                              np.transpose(curves, (1, 2, 0)))},
            coords={"norad_ids": np.array(labels), "times": times, "freqs": freqs},
            attrs=attrs,
        ).to_zarr(swapped)

        args = (labels, times, freqs)
        np.testing.assert_allclose(
            np.asarray(read_light_curves(swapped, *args)),
            np.asarray(read_light_curves(canonical, *args)),
            rtol=1e-9,
        )

    def test_a_zarr_with_wrong_dimension_names_raises(self, tmp_path):
        import xarray as xr

        path = str(tmp_path / "bad_dims.zarr")
        xr.Dataset(
            {"light_curves": (("norad_ids", "times", "channel"), np.ones((2, 5, 3)))},
            coords={"norad_ids": np.array([100, 200]), "times": self._times(),
                    "freqs": ("channel", self._freqs())},
        ).to_zarr(path)
        with pytest.raises(ValueError, match="dimensioned"):
            read_light_curves(path, [100], self._times(), self._freqs())

    def test_duplicate_ids_in_the_file_raise(self, tmp_path):
        """Two rows for one satellite has no answer that is not file order."""
        path = self._npz(tmp_path, [100, 200, 100])
        with pytest.raises(ValueError, match="more than one light curve") as excinfo:
            read_light_curves(path, [100, 200], self._times(), self._freqs())
        assert "100" in str(excinfo.value)

    def test_repeated_non_satellite_labels_are_still_fine(self, tmp_path):
        """Only integer ids identify a row, so named sources may repeat freely."""
        path = self._npz(tmp_path, ["Fornax A", "100", "Fornax A"])
        out = np.asarray(
            read_light_curves(path, [100], self._times(), self._freqs())
        )
        assert out[0].max() == 2.0

    def test_a_duplicated_configured_satellite_is_not_a_file_error(self, tmp_path):
        """Sharding pads the satellite list by repeating one; the file is still valid."""
        path = self._npz(tmp_path, [100, 200])
        out = np.asarray(
            read_light_curves(path, [100, 200, 200], self._times(), self._freqs())
        )
        assert out.shape[0] == 3
        np.testing.assert_array_equal(out[1], out[2])

    def test_infinite_samples_become_zero_not_float_max(self, tmp_path, capsys):
        """nan_to_num's default maps inf onto the f64 extrema, which is not finite.

        1.8e308 overflows back to inf in f32 — the default working precision —
        and survives the later sqrt, so an estimate meant to be finite would
        poison the prior mean or the init with inf.
        """
        curves = np.ones((1, 5, 3))
        curves[0, 1, 0] = np.inf
        curves[0, 2, 1] = -np.inf
        path = self._npz(tmp_path, [100], curves=curves)

        out = np.asarray(read_light_curves(path, [100], self._times(), self._freqs()))
        assert np.all(np.isfinite(out))
        # Well below f32 max, i.e. genuinely zeroed rather than clipped to a huge float.
        assert np.abs(out).max() <= 1.0
        assert np.isfinite(np.sqrt(np.abs(out))).all()
        assert "infinite" in capsys.readouterr().out

    def test_infinities_survive_the_single_precision_cast(self, tmp_path):
        """The end-to-end shape of the bug: inf in the file -> inf in the estimate."""
        curves = np.ones((1, 5, 3))
        curves[0, 0, 0] = np.inf
        path = self._npz(tmp_path, [100], curves=curves)

        out = read_light_curves(path, [100], self._times(), self._freqs())
        assert jnp.isfinite(jnp.sqrt(jnp.abs(out))).all()

    def test_nan_is_not_reported_as_an_infinity(self, tmp_path, capsys):
        """NaN is the documented out-of-view marker, so it must not warn."""
        curves = np.ones((1, 5, 3))
        curves[0, :2] = np.nan
        path = self._npz(tmp_path, [100], curves=curves)

        read_light_curves(path, [100], self._times(), self._freqs())
        assert "infinite" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Which time axis the estimate is sampled on
# ---------------------------------------------------------------------------

class TestTheEstimateIsSampledOnUtc:
    """A light-curve file's ``times`` are UTC MJD, so the sampling has to be too.

    The format is an interchange one, reusable across measurement sets covering
    the same pass, which only works if its time axis names an *instant* rather
    than whatever number some MS happened to store. That instant is on UTC, the
    scale everything past ``read_ms`` already works on.

    An MS is free to declare something else in its ``TIME`` column's ``MEASINFO``
    record, and ``read_ms`` deliberately leaves ``times_mjd`` on the declared
    scale so it stays comparable with the column itself. Sampling the estimate
    there instead reads the file 37 s from where it was measured on a
    TAI-declared MS -- silently, since a light curve resampled off by a few
    integrations is still the right shape and still optimises.

    The MS comes through the in-memory rig from ``tests.test_ms``, so the scale
    is declared the way casacore declares it rather than asserted onto a stub.
    """

    #: 2025-01-01T00:00:00 UTC as an MJD, when TAI - UTC was 37 s.
    MJD = 60676.0
    LEAP_SECS = 37.0

    #: 25 integrations 8 s apart: long enough that 37 s is a few timesteps inside
    #: the observation rather than more than the whole of it.
    N_TIME, STEP = 25, 8.0

    #: Half-width of the spike, and the half-width of its flat top, in seconds.
    #: The plateau is wider than any rounding in the time axis and far narrower
    #: than one integration, so the peak sample is exactly 1.0 when it lands on
    #: the feature and exactly 0.0 one integration away.
    WIDTH, PLATEAU = 8.0, 1.0

    F0 = 1.405e9

    def _ms_times(self):
        """The MS ``TIME`` column, in seconds, on whatever scale it declares."""

        return (self.MJD + np.arange(self.N_TIME) * self.STEP / 86400.0) * 86400.0

    @property
    def _peak_step(self):
        """Timestep the spike is centred on."""

        return self.N_TIME // 2

    def _peak_mjd_utc(self, shift_secs):
        """UTC MJD of that timestep, given what the declaration costs it.

        Stated as "the declared number, less the leap seconds" rather than read
        off the conversion under test, so the expectation is independent of it.
        """

        declared = self.MJD + self._peak_step * self.STEP / 86400.0

        return declared + shift_secs / 86400.0

    def _spike_file(self, tmp_path, norad_ids, peak_mjd_utc):
        """A curve that is zero except for one narrow spike at ``peak_mjd_utc``.

        Sampled every second -- an order finer than the observation -- so where
        the spike lands in the result is set by the coordinate the reader asks
        for and not by the file's own resolution. A single-frequency axis, which
        the reader holds constant across the band, keeps frequency out of it.
        """

        secs = np.arange(-120.0, 120.0 + 1.0, 1.0)
        curve = np.clip(
            (self.WIDTH - np.abs(secs)) / (self.WIDTH - self.PLATEAU), 0.0, 1.0
        )
        path = str(tmp_path / "spike.npz")
        np.savez(
            path,
            light_curves=np.tile(curve[None, :, None], (len(norad_ids), 1, 1)),
            norad_ids=np.asarray(norad_ids),
            times=peak_mjd_utc + secs / 86400.0,
            freqs=np.array([self.F0]),
            time_scale="utc",
        )

        return path

    def _component(self, data):
        """A component set up on the times and scale a real MS read produced."""

        config = make_rfi_config(n_time=self.N_TIME)
        config.times_mjd = np.asarray(data["times_mjd"])
        config.times_jd = np.asarray(data["times_jd"])
        config.time_scale = data["time_scale"]

        comp = ComplexRFIVarAnt()
        comp.setup(config)

        return comp, config

    @staticmethod
    def _sample(comp, path, times_mjd):
        """The one satellite's light curve, read onto ``times_mjd``."""

        ids = comp.norad_ids[: comp.n_rfi_real]

        return np.asarray(read_light_curves(path, ids, times_mjd, comp.freqs))[0, 0]

    def test_a_tai_ms_samples_the_estimate_where_utc_says(self, run_reader, tmp_path):
        """The acceptance check: the spike lands on the integration that saw it.

        The file's peak is at the UTC instant of one integration. Sampled on
        UTC, that integration reads the peak and its neighbours read zero.
        Sampled on the declared TAI numbers the whole curve slides 37 s, which is
        4.6 integrations -- so the peak lands nowhere near the timestep that
        measured it, and the run is seeded with a satellite that brightens at the
        wrong time.
        """
        data = run_reader(self._ms_times(), keywords=_keywords("TAI"))
        comp, config = self._component(data)
        path = self._spike_file(
            tmp_path,
            comp.norad_ids[: comp.n_rfi_real],
            self._peak_mjd_utc(-self.LEAP_SECS),
        )

        curve = self._sample(comp, path, comp.est_times_mjd)

        peak = self._peak_step
        assert curve.argmax() == peak
        assert curve[peak] == 1.0
        assert curve[peak - 1] == 0.0 and curve[peak + 1] == 0.0

        # And what sampling the declared column instead would have done: the
        # feature moves by the leap seconds, i.e. round(37 / 8) = 5 integrations.
        declared = self._sample(comp, path, config.times_mjd)
        assert declared.argmax() == peak - round(self.LEAP_SECS / self.STEP)
        assert declared[peak] == 0.0

    def test_the_component_reads_the_estimate_on_utc(self, run_reader):
        """The axis itself, to the sub-microsecond an MJD resolves to at f64."""
        data = run_reader(self._ms_times(), keywords=_keywords("TAI"))
        comp, config = self._component(data)

        np.testing.assert_allclose(
            comp.est_times_mjd,
            np.asarray(config.times_mjd) - self.LEAP_SECS / 86400.0,
            rtol=0,
            atol=1e-9,  # days, i.e. ~90 us
        )

    def test_a_utc_ms_is_sampled_exactly_where_it_always_was(
        self, run_reader, tmp_path
    ):
        """Behaviour-neutral for the common case: bit-identical, not merely close.

        A UTC-declared MS needs no conversion, and must get none -- routing its
        column out to a Julian Date and back would round it at the coarser ~2.5e6
        magnitude and move every sample by ~10 us for nothing.
        """
        data = run_reader(self._ms_times(), keywords=_keywords("UTC"))
        comp, config = self._component(data)
        path = self._spike_file(
            tmp_path, comp.norad_ids[: comp.n_rfi_real], self._peak_mjd_utc(0.0)
        )

        np.testing.assert_array_equal(comp.est_times_mjd, config.times_mjd)
        np.testing.assert_array_equal(
            self._sample(comp, path, comp.est_times_mjd),
            self._sample(comp, path, config.times_mjd),
        )

    def test_a_utc_ms_still_finds_the_feature(self, run_reader, tmp_path):
        """The control for the TAI case: with nothing to shift, nothing shifts."""
        data = run_reader(self._ms_times(), keywords=_keywords("UTC"))
        comp, _ = self._component(data)
        path = self._spike_file(
            tmp_path, comp.norad_ids[: comp.n_rfi_real], self._peak_mjd_utc(0.0)
        )

        curve = self._sample(comp, path, comp.est_times_mjd)

        assert curve.argmax() == self._peak_step
        assert curve[self._peak_step] == 1.0


class TestMaskIsolatesNonFiniteSamples:
    """A masked sample must be exactly zero, whatever rfi_A holds there.

    The mask is a select, not a multiply by 0/1: ``0 * inf`` and ``0 * nan`` are
    both nan, so a multiply lets a non-finite value leak straight back through
    the mask it is supposed to remove. The optimiser can put rfi_A somewhere
    non-finite transiently, and one nan in the state poisons every gradient
    downstream of it.
    """

    @pytest.mark.parametrize("cls", ALL_CLASSES)
    def test_masked_samples_are_zero_even_when_the_signal_is_not_finite(self, cls):
        comp = setup_component(cls, rfi_mask_fine=elevation_mask())
        constants = make_constants(comp)
        mask = np.asarray(comp.rfi_mask_fine)

        # Stand in for a forward that has wandered somewhere non-finite.
        rfi_A = jnp.full((N_RFI, N_ANT, N_FREQ, N_TIME), jnp.inf, dtype=complex)
        masked = np.asarray(comp.build_masked_signal()(rfi_A, constants))

        out_of_view = ~mask[:, None, None, :] & np.ones_like(masked, dtype=bool)
        assert np.all(masked[out_of_view] == 0), "a masked sample is not exactly zero"

    @pytest.mark.parametrize("cls", ALL_CLASSES)
    def test_a_nan_outside_the_window_does_not_reach_the_kept_samples(self, cls):
        comp = setup_component(cls, rfi_mask_fine=elevation_mask())
        constants = make_constants(comp)
        mask = np.asarray(comp.rfi_mask_fine)

        rfi_A = jnp.ones((N_RFI, N_ANT, N_FREQ, N_TIME), dtype=complex)
        rfi_A = rfi_A.at[0, :, :, ~mask[0]].set(jnp.nan)

        masked = np.asarray(comp.build_masked_signal()(rfi_A, constants))
        assert np.all(np.isfinite(masked)), "a masked nan survived the mask"
        assert np.all(masked[0, :, :, mask[0]] == 1.0)


# ---------------------------------------------------------------------------
# rfi.pow_spec
# ---------------------------------------------------------------------------


def setup_with_pow_spec(cls, pow_spec, n_freq=8, n_time=16, **kwargs):
    """Set a component up with an ``rfi.pow_spec`` block, or without one."""
    config = make_rfi_config(
        n_rfi=2, n_rfi_real=2, n_ant=3, n_freq=n_freq, n_time=n_time, **kwargs
    )
    if pow_spec is not _ABSENT:
        config.args["rfi"]["pow_spec"] = pow_spec

    comp = cls()
    comp.setup(config)

    return comp


def pow_spec_error(cls, pow_spec):
    """The message from a rejected ``rfi.pow_spec``, as one string."""
    with pytest.raises(RuntimeError) as excinfo:
        setup_with_pow_spec(cls, pow_spec)

    return str(excinfo.value)


#: Distinguishes "no block at all" from an explicit ``pow_spec: null``, which are
#: different config files and must behave the same.
_ABSENT = object()


class TestPowSpecIsRead:
    """``rfi.pow_spec`` was in the shipped configs and read by nothing (#111).

    The tests that matter here are the ones that would fail if it went back to
    being ignored: setting a key has to move the latent dimension, which is what
    the cutoff and the roll-off exponent between them decide.
    """

    @pytest.mark.parametrize("absent", [_ABSENT, None])
    @pytest.mark.parametrize("cls", FOURIER_CLASSES)
    def test_no_block_means_the_components_own_default(self, cls, absent):
        """A config predating the key runs exactly as it did."""
        comp = setup_with_pow_spec(cls, absent)

        assert comp.gp_pow_spec() == (
            [float(g) for g in cls.default_gammas],
            float(cls.default_pk_cutoff),
        )

    def test_the_two_components_keep_their_own_defaults(self):
        """Preserved rather than unified: making them agree is a model change."""
        assert ComplexRFIVarAnt.default_gammas != ComplexRFIConstAnt.default_gammas
        assert (
            ComplexRFIVarAnt.default_pk_cutoff != ComplexRFIConstAnt.default_pk_cutoff
        )

    @pytest.mark.parametrize("cls", FOURIER_CLASSES)
    def test_a_cutoff_that_bites_drops_k_modes(self, cls):
        """The key is live: a coarser cutoff leaves fewer fitted parameters."""
        kept = setup_with_pow_spec(cls, {"cutoff": 1e-9})
        cut = setup_with_pow_spec(cls, {"cutoff": 1e-2})

        assert cut.n_k_freq_rfi < kept.n_k_freq_rfi
        assert cut.n_k_time_rfi < kept.n_k_time_rfi

    @pytest.mark.parametrize("cls", FOURIER_CLASSES)
    def test_a_steeper_roll_off_drops_k_modes(self, cls):
        """The other half of the same statement, for gammas."""
        shallow = setup_with_pow_spec(cls, {"gammas": [3, 3], "cutoff": 1e-9})
        steep = setup_with_pow_spec(cls, {"gammas": [8, 8], "cutoff": 1e-9})

        assert steep.n_k_freq_rfi < shallow.n_k_freq_rfi
        assert steep.n_k_time_rfi < shallow.n_k_time_rfi

    @pytest.mark.parametrize("cls", FOURIER_CLASSES)
    def test_one_key_given_leaves_the_other_on_its_default(self, cls):
        comp = setup_with_pow_spec(cls, {"cutoff": 1e-4})

        assert comp.gp_pow_spec() == ([float(g) for g in cls.default_gammas], 1e-4)

    @pytest.mark.parametrize("key", ["p0", "k0s"])
    def test_the_derived_keys_are_refused_by_name(self, key):
        """They were in the old blocks and are not settings; saying so beats ignoring."""
        message = pow_spec_error(ComplexRFIVarAnt, {key: 1.0})

        assert f"rfi.pow_spec.{key} is not a setting" in message

    def test_an_unknown_key_is_refused_by_name(self):
        """Asserted against the offending-key list, not the whole message: every
        message ends "It takes ['gammas', 'cutoff']", so a bare `"gamma" in
        message` passes for a validator that names nothing."""
        message = pow_spec_error(ComplexRFIVarAnt, {"gamma": 3})

        assert "no key(s) ['gamma']" in message

    @pytest.mark.parametrize(
        "gammas", [3, "3", [3], [3, 3, 3], [0, 3], [-1, 3], [True, 3], [float("inf"), 3], [float("nan"), 3]]
    )
    def test_gammas_that_are_not_a_pair_of_positive_numbers_are_refused(self, gammas):
        assert "gammas" in pow_spec_error(ComplexRFIVarAnt, {"gammas": gammas})

    @pytest.mark.parametrize(
        "cutoff", [0, -1e-6, "1e-6", True, float("inf"), float("nan"), [1e-6]]
    )
    def test_a_cutoff_that_is_not_a_positive_number_is_refused(self, cutoff):
        assert "cutoff" in pow_spec_error(ComplexRFIVarAnt, {"cutoff": cutoff})

    def test_a_pow_spec_that_is_not_a_mapping_is_refused(self):
        assert "pow_spec" in pow_spec_error(ComplexRFIVarAnt, [3, 3])

    @pytest.mark.parametrize("cutoff", [1.0, 2.0])
    def test_a_cutoff_that_cuts_everything_is_refused(self, cutoff):
        """It is relative to the largest mode and the comparison is strict, so 1
        keeps nothing. Left to fft_gp it surfaces as "zero-size array to reduction
        operation min", which names neither the key nor the reason."""
        assert "cutoff" in pow_spec_error(ComplexRFIVarAnt, {"cutoff": cutoff})

    def test_a_cutoff_just_below_one_is_still_accepted(self):
        """The bound is at 1, not near it. Asserted as "fewer modes than the
        default, and at least one", rather than an exact shape, which is a
        property of this fixture's grid and not of the bound."""
        comp = setup_with_pow_spec(ComplexRFIVarAnt, {"cutoff": 0.999999})
        default = setup_with_pow_spec(ComplexRFIVarAnt, _ABSENT)

        assert 1 <= comp.n_k_freq_rfi < default.n_k_freq_rfi
        assert 1 <= comp.n_k_time_rfi < default.n_k_time_rfi

    def test_a_cutoff_that_only_empties_the_grid_in_the_working_precision(self):
        """The config bound cannot catch this one, so pk_cut is the backstop.

        0.99999999 is below 1 as a Python float, so ``cutoff < 1`` accepts it,
        and it is exactly 1.0 as a float32 -- so in single precision every mode
        is cut. The value has to be the near-1 one and the grid has to be
        float32, or the test is about the case the config validator already
        rejects. pk_cut is called directly because that is where the backstop is;
        through a component the error arrives wrapped in RuntimeError.
        """
        from tabascal.fft_gp import pk_cut

        near_one = 0.99999999
        assert near_one < 1.0 and np.float32(near_one) == np.float32(1.0)

        with pytest.raises(ValueError, match="no Fourier modes"):
            pk_cut(jnp.ones((4, 4), dtype=jnp.float32), near_one)

    @pytest.mark.requires_double
    def test_the_same_cutoff_is_harmless_in_double_precision(self):
        """The other half of the one above: it is a precision effect, not a bound.

        Marked ``requires_double`` rather than asking for float64 inline --
        under ``--x64 false`` jax truncates a requested float64 to float32 with
        only a warning, so an unguarded version of this asserts the opposite of
        what it says and fails in single precision, which is how it reached CI.
        """
        from tabascal.fft_gp import pk_cut

        idxs, _ = pk_cut(jnp.ones((4, 4), dtype=jnp.float64), 0.99999999)

        assert all(idx.stop > idx.start for idx in idxs)

    @pytest.mark.parametrize(
        "gammas", [np.array([3.0, 3.0]), (3, 3), [np.float32(3), np.float64(3)]]
    )
    def test_an_ordered_pair_is_accepted_however_it_is_spelled(self, gammas):
        """A config assembled in Python carries numpy; only *unordered* pairs are
        the problem, and those are refused by name."""
        comp = setup_with_pow_spec(ComplexRFIVarAnt, {"gammas": gammas})

        assert comp.gp_pow_spec()[0] == [3.0, 3.0]

    def test_gammas_given_as_something_keyed_rather_than_ordered_is_refused(self):
        """Keyed things are refused whatever shape they take.

        A pandas Series is the realistic instance -- keyed, 1-D, length two, so
        every shape-based check accepts it and reads its values in index order,
        silently swapping the axes if the user wrote time first. pandas is not a
        declared dependency, so the contract is pinned here with a stand-in of
        the same shape and confirmed against the real thing below.
        """

        class KeyedPair:
            """len 2, indexable, iterable -- and keyed, which is the point."""

            def __init__(self, **entries):
                self._entries = dict(entries)

            def keys(self):
                return self._entries.keys()

            def __len__(self):
                return len(self._entries)

            def __iter__(self):
                return iter(self._entries.values())

            def __getitem__(self, key):
                return list(self._entries.values())[key]

        keyed = KeyedPair(time=3.0, freq=4.0)
        assert len(keyed) == 2 and keyed[0] == 3.0

        assert "gammas" in pow_spec_error(ComplexRFIVarAnt, {"gammas": keyed})

    def test_a_pandas_series_is_refused(self):
        """The real instance of the case above, when pandas is installed."""
        pd = pytest.importorskip("pandas")
        series = pd.Series({"time": 3.0, "freq": 4.0})

        assert "gammas" in pow_spec_error(ComplexRFIVarAnt, {"gammas": series})

    @pytest.mark.parametrize("gammas", [{3: None, 4: None}, {3, 4}])
    def test_gammas_given_as_an_unordered_pair_are_refused(self, gammas):
        """A mapping or a set of length two passes len() and iterates to its keys,
        in an order that means nothing -- but the two entries name the frequency
        and time axes, in that order."""
        assert "gammas" in pow_spec_error(ComplexRFIVarAnt, {"gammas": gammas})

    def test_unknown_keys_that_are_not_all_strings_still_name_themselves(self):
        """YAML keys need not be strings, and sorting a mixed set raises from
        inside the validator instead of saying which key is wrong."""
        message = pow_spec_error(ComplexRFIVarAnt, {1: 2, "gamma": 3})

        assert "'gamma'" in message and "1" in message
        assert "not supported between instances" not in message

    def test_the_base_config_ships_the_key_unset(self):
        """So that upgrading changes nothing until someone sets a value."""
        from tabascal.config import yaml_load
        from importlib.resources import files

        base = yaml_load(
            str(files("tabascal.data.config").joinpath("tab_config_base.yaml"))
        )

        assert base["rfi"]["pow_spec"] == {"gammas": None, "cutoff": None}
