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
    RealRFIVarAnt,
    compute_real_space_gp_params,
    rfi_signal_config_validation,
)
from tabascal.fft_gp import latent_to_signal
from tabascal.gp import base_kernel, get_times

from .conftest import active_precision, assert_transform_roundtrip, make_constants


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_RFI, N_RFI_REAL, N_ANT, N_FREQ, N_TIME = 4, 3, 3, 4, 8

ALL_CLASSES = [RealRFIVarAnt, ComplexRFIVarAnt, ComplexRFIConstAnt]
REAL_SPACE_CLASSES = [RealRFIVarAnt]
FOURIER_CLASSES = [ComplexRFIVarAnt, ComplexRFIConstAnt]

# Init modes every class accepts. "truth" is excluded throughout: it goes through
# read_true_rfi_A, which needs a real simulation .zarr store.
COMMON_INITS = ["prior", "zeros", "ones", "sample"]


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


def make_est_file(tmp_path, n_rfi_real=N_RFI_REAL, n_time=N_TIME):
    """Write the .npy RFI estimate that the `est` prior/init modes read."""
    path = tmp_path / "rfi_est.npy"
    rng = np.random.RandomState(0)
    np.save(path, np.abs(rng.randn(n_rfi_real, n_time, 2)))
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
# compute_real_space_gp_params
# ---------------------------------------------------------------------------

class TestComputeRealSpaceGPParams:

    def test_shapes_and_count(self):
        """n_gp_times follows get_times, and the resample operator maps GP -> fine grid."""
        times = jnp.linspace(0.0, 120.0, 8)
        times_fine = jnp.linspace(0.0, 120.0, 16)
        corr_time = 24.0

        n_gp_times, gp_times, resample_op = compute_real_space_gp_params(
            corr_time, 1.0, times, times_fine
        )

        assert n_gp_times == len(get_times(times, corr_time))
        assert gp_times.shape == (n_gp_times,)
        assert resample_op.shape == (len(times_fine), n_gp_times)

    @pytest.mark.requires_double
    @pytest.mark.parametrize("signal_fn", [jnp.ones_like, lambda t: jnp.sin(t / 24.0)])
    def test_resampling_interpolates_onto_the_fine_grid(self, signal_fn):
        """The operator interpolates a GP-grid signal onto the fine grid.

        Loose tolerance on purpose: this is GP interpolation with a 1e-8 nugget, not an
        exact reconstruction, and the assertion only needs to catch a broken operator.

        Double precision only: ``resampling_kernel`` inverts the GP covariance with a
        1e-8 nugget, which in fp32 is ill-conditioned enough that the operator picks up
        O(0.5) ripples and interpolates visibly badly.
        """
        times = jnp.linspace(0.0, 120.0, 8)
        times_fine = jnp.linspace(0.0, 120.0, 16)
        corr_time = 24.0

        _, gp_times, resample_op = compute_real_space_gp_params(
            corr_time, 1.0, times, times_fine
        )

        assert jnp.allclose(
            resample_op @ signal_fn(gp_times), signal_fn(times_fine), atol=1e-2
        )


# ---------------------------------------------------------------------------
# BaseGPRFI padding helpers
# ---------------------------------------------------------------------------

class TestPaddingHelpers:
    """The device-sharding helpers on BaseGPRFI, driven through a concrete subclass."""

    def test_mask_dummy_rfi_zeroes_only_padded_rows(self):
        comp = setup_component(RealRFIVarAnt)
        arr = jnp.arange(N_RFI * 2 * 3, dtype=float).reshape(N_RFI, 2, 3) + 1.0

        masked = comp._mask_dummy_rfi(arr)

        assert jnp.array_equal(masked[:N_RFI_REAL], arr[:N_RFI_REAL])
        assert jnp.all(masked[N_RFI_REAL:] == 0)

    def test_mask_dummy_rfi_is_noop_when_unpadded(self):
        comp = setup_component(RealRFIVarAnt, n_rfi=N_RFI, n_rfi_real=N_RFI)
        arr = jnp.ones((N_RFI, 2, 3))

        assert jnp.array_equal(comp._mask_dummy_rfi(arr), arr)

    @pytest.mark.parametrize("dtype", [float, complex])
    def test_zero_pad_rfi_grows_and_zeroes(self, dtype):
        """A truth/estimate array with only the real sources is padded with exact zeros."""
        comp = setup_component(RealRFIVarAnt)
        arr = jnp.ones((N_RFI_REAL, 2, 3), dtype=dtype)

        padded = comp._zero_pad_rfi(arr)

        assert padded.shape == (N_RFI, 2, 3)
        assert padded.dtype == arr.dtype
        assert jnp.array_equal(padded[:N_RFI_REAL], arr)
        assert jnp.all(padded[N_RFI_REAL:] == 0)

    def test_zero_pad_rfi_is_identity_when_already_full(self):
        comp = setup_component(RealRFIVarAnt)
        arr = jnp.ones((N_RFI, 2, 3))

        assert comp._zero_pad_rfi(arr) is arr

    def test_n_rfi_real_defaults_to_n_rfi_when_config_lacks_it(self):
        """An unpadded TabConfig has no n_rfi_real; the mask must then be a no-op."""
        comp = setup_component(RealRFIVarAnt, with_n_rfi_real=False)

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
            (RealRFIVarAnt, {"L_rfi_A", "mu_rfi_A", "resample_rfi"}),
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
            # RealRFIVarAnt models a real-valued amplitude; the others are complex. Note the
            # state_outputs placeholder is always complex, so for RealRFIVarAnt the forward
            # narrows the dtype when it overwrites it.
            (RealRFIVarAnt, False),
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
# Transforms
# ---------------------------------------------------------------------------

class TestTransforms:

    def test_real_rfi_roundtrip(self):
        comp = setup_component(RealRFIVarAnt)
        shape = comp.init_params_base["rfi_r_induce_base"].shape
        base = jax.random.normal(jax.random.PRNGKey(42), shape)

        assert_transform_roundtrip(comp, base, comp.L_rfi_A, comp.mu_rfi_A)

    @pytest.mark.parametrize("cls", FOURIER_CLASSES)
    def test_fourier_roundtrip(self, cls):
        """The Fourier transform pair is a plain scale-and-shift, so it inverts exactly."""
        comp = setup_component(cls, mean="data")
        shape = comp.init_params_base["rfi_k_r_base"].shape
        k1, k2 = jax.random.split(jax.random.PRNGKey(42))
        base = jax.random.normal(k1, shape) + 1j * jax.random.normal(k2, shape)

        assert_transform_roundtrip(comp, base, comp.sigma_rfi_k, comp.mu_rfi_k, atol=tol())

    @pytest.mark.parametrize("cls", REAL_SPACE_CLASSES)
    def test_cholesky_factor_reconstructs_the_kernel(self, cls):
        """L is lower-triangular and L @ L.T recovers the GP covariance."""
        comp = setup_component(cls)
        L = comp.L_rfi_A

        assert L.shape == (comp.n_rfi_times, comp.n_rfi_times)
        assert jnp.allclose(L, jnp.tril(L))

        expected = base_kernel(
            comp.rfi_times, comp.rfi_times, comp.gp_var, comp.corr_time
        ) + 1e-8 * jnp.eye(comp.n_rfi_times)
        assert jnp.allclose(L @ L.T, expected, atol=1e-5)

    @pytest.mark.parametrize("cls", FOURIER_CLASSES)
    def test_sigma_is_broadcastable_over_sources_and_antennas(self, cls):
        """The Fourier scale is shared by every source and antenna."""
        comp = setup_component(cls)
        assert comp.sigma_rfi_k.shape == (1, 1, comp.n_k_freq_rfi, comp.n_k_time_rfi)
        assert jnp.all(comp.sigma_rfi_k > 0)


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

    def test_real_rfi_init_params_are_real_only(self):
        comp = setup_component(RealRFIVarAnt)
        assert set(comp.init_params) == {"rfi_r_induce"}

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

    @pytest.mark.parametrize("cls", REAL_SPACE_CLASSES)
    def test_real_space_gp_grid_attributes(self, cls):
        comp = setup_component(cls)

        assert comp.rfi_times.shape == (comp.n_rfi_times,)
        assert comp.resample_rfi.shape == (N_TIME, comp.n_rfi_times)
        assert comp.mu_rfi_A.shape == (N_RFI, N_ANT, N_FREQ, comp.n_rfi_times)


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
        n_rfi=n_rfi, n_rfi_real=n_rfi_real, n_ant=n_ant, n_freq=n_freq, n_time=n_time,
        n_freq_fine=n_freq, n_time_fine=n_time, n_int_freq=1, n_int_time=1,
        freqs=freqs, freqs_fine=freqs, chan_width=float(freqs[1] - freqs[0]),
        times=times, times_fine=times, int_time=float(times[1] - times[0]),
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
