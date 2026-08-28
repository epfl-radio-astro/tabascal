"""Tests for tabascal.components.gains — UnitaryGains, GPGains, and config validation."""

import pytest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro

from tabascal.components.gains import (
    UnitaryGains,
    GPGains,
    gains_config_validation,
    validate_gain_scales,
)

from .conftest import make_constants, assert_transform_roundtrip


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_gains_config(
    n_ant=4,
    n_freq=4,
    n_time=8,
    amp_mean=1.0,
    amp_std=1.0,
    phase_mean=0.0,
    phase_std=1.0,
    amp_corr_freq=None,
    amp_corr_time=None,
    phase_corr_freq=None,
    phase_corr_time=None,
):
    """Build a minimal mock TabConfig for gains components."""
    a1, a2 = jnp.triu_indices(n_ant, 1)
    n_bl = len(a1)
    freqs = jnp.linspace(1.4e9, 1.41e9, n_freq)
    chan_width = float(freqs[1] - freqs[0]) if n_freq > 1 else 1e6
    times = jnp.linspace(0.0, 120.0, n_time)
    int_time = float(times[1] - times[0]) if n_time > 1 else 8.0

    return SimpleNamespace(
        n_ant=n_ant,
        n_bl=n_bl,
        n_freq=n_freq,
        n_time=n_time,
        n_freq_fine=n_freq,
        n_time_fine=n_time,
        n_int_time=1,
        n_int_freq=1,
        freqs=freqs,
        chan_width=chan_width,
        times=times,
        int_time=int_time,
        a1=a1.astype("int32"),
        a2=a2.astype("int32"),
        args={
            "gains": {
                "r_seed": 123,
                "amp_mean": amp_mean,
                "amp_std": amp_std,
                "amp_corr_freq": amp_corr_freq,
                "amp_corr_time": amp_corr_time,
                "phase_mean": phase_mean,
                "phase_std": phase_std,
                "phase_corr_freq": phase_corr_freq,
                "phase_corr_time": phase_corr_time,
            }
        },
    )


def make_vis_state(n_ant, n_freq, n_time, rng_key=0):
    """Build a minimal state dict with vis_rfi and vis_ast."""
    a1, a2 = jnp.triu_indices(n_ant, 1)
    n_bl = len(a1)
    key = jax.random.PRNGKey(rng_key)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    shape = (n_bl, n_freq, n_time)
    vis_rfi = jax.random.normal(k1, shape) + 1j * jax.random.normal(k2, shape)
    vis_ast = jax.random.normal(k3, shape) + 1j * jax.random.normal(k4, shape)
    return {"vis_rfi": vis_rfi, "vis_ast": vis_ast}


# ---------------------------------------------------------------------------
# gains_config_validation
# ---------------------------------------------------------------------------

class TestGainsConfigValidation:

    def test_null_values_get_defaults(self):
        """None corr_time / corr_freq values are replaced with defaults derived from the observation grid."""
        freqs = jnp.linspace(1.4e9, 1.41e9, 4)
        times = jnp.linspace(0.0, 120.0, 8)
        cfg = {
            "r_seed": None,
            "amp_mean": None,
            "amp_std": None,
            "amp_corr_freq": None,
            "amp_corr_time": None,
            "phase_mean": None,
            "phase_std": None,
            "phase_corr_freq": None,
            "phase_corr_time": None,
        }
        result = gains_config_validation(cfg, freqs, 1e6, times, 8.0)

        assert result["r_seed"] == 2
        assert result["amp_mean"] == pytest.approx(1.0)
        assert result["amp_std"] == pytest.approx(0.01)  # 1% of amp_mean=1.0
        assert result["phase_mean"] == pytest.approx(0.0)
        assert result["amp_corr_time"] > 0
        assert result["phase_corr_time"] > 0

    def test_explicit_values_stored_correctly(self):
        """Non-null correlation times and frequencies are stored unchanged on the component."""
        freqs = jnp.linspace(1.4e9, 1.41e9, 4)
        times = jnp.linspace(0.0, 120.0, 8)
        cfg = {
            "r_seed": 42,
            "amp_mean": 2.0,
            "amp_std": 5.0,   # percent
            "amp_corr_freq": 5e6,
            "amp_corr_time": 60.0,
            "phase_mean": 0.1,
            "phase_std": 2.0,  # degrees
            "phase_corr_freq": 5e6,
            "phase_corr_time": 30.0,
        }
        result = gains_config_validation(cfg, freqs, 1e6, times, 8.0)

        assert result["r_seed"] == 42
        assert result["amp_mean"] == pytest.approx(2.0)
        assert result["amp_std"] == pytest.approx(5.0 / 100 * 2.0)
        assert result["amp_corr_freq"] == pytest.approx(5e6)
        assert result["amp_corr_time"] == pytest.approx(60.0)
        assert result["phase_mean"] == pytest.approx(0.1)
        assert result["phase_std"] == pytest.approx(float(jnp.deg2rad(2.0)))
        assert result["phase_corr_time"] == pytest.approx(30.0)

    def test_invalid_amp_mean_type_raises(self):
        """A non-numeric amp_mean raises ValueError during gains_config_validation."""
        freqs = jnp.linspace(1.4e9, 1.41e9, 4)
        times = jnp.linspace(0.0, 120.0, 8)
        cfg = {
            "r_seed": 1,
            "amp_mean": "bad",
            "amp_std": None,
            "amp_corr_freq": None,
            "amp_corr_time": None,
            "phase_mean": None,
            "phase_std": None,
            "phase_corr_freq": None,
            "phase_corr_time": None,
        }
        with pytest.raises(ValueError):
            gains_config_validation(cfg, freqs, 1e6, times, 8.0)

    def test_invalid_phase_std_type_raises(self):
        """A non-numeric phase_std raises ValueError during gains_config_validation."""
        freqs = jnp.linspace(1.4e9, 1.41e9, 4)
        times = jnp.linspace(0.0, 120.0, 8)
        cfg = {
            "r_seed": 1,
            "amp_mean": 1.0,
            "amp_std": None,
            "amp_corr_freq": None,
            "amp_corr_time": None,
            "phase_mean": None,
            "phase_std": "bad",
            "phase_corr_freq": None,
            "phase_corr_time": None,
        }
        with pytest.raises(ValueError):
            gains_config_validation(cfg, freqs, 1e6, times, 8.0)

    def test_single_freq_single_time_defaults(self):
        """Single channel/integration — corr lengths should default to step size."""
        freqs = jnp.array([1.4e9])
        times = jnp.array([0.0])
        cfg = {k: None for k in [
            "r_seed", "amp_mean", "amp_std", "amp_corr_freq", "amp_corr_time",
            "phase_mean", "phase_std", "phase_corr_freq", "phase_corr_time",
        ]}
        result = gains_config_validation(cfg, freqs, 1e6, times, 8.0)
        assert result["amp_corr_freq"] > 0
        assert result["amp_corr_time"] > 0


# ---------------------------------------------------------------------------
# None means "unset", 0 means 0
# ---------------------------------------------------------------------------

SCALE_KEYS = ("r_seed", "amp_mean", "amp_std", "phase_mean", "phase_std")
CORR_KEYS = ("amp_corr_freq", "amp_corr_time", "phase_corr_freq", "phase_corr_time")

#: Every value that is not a width: zero included, since a zero-width prior is a
#: degenerate Normal rather than an unset one.
UNUSABLE = [0, 0.0, -1.0, float("nan"), float("inf"), -float("inf")]


def scale_config(**overrides):
    """The keys :func:`validate_gain_scales` reads, all unset unless overridden."""

    return {**{key: None for key in SCALE_KEYS}, **overrides}


def full_config(**overrides):
    """The keys :func:`gains_config_validation` reads, all unset unless overridden."""

    return {**{key: None for key in SCALE_KEYS + CORR_KEYS}, **overrides}


def validate_full(cfg, freqs=None, times=None, chan_width=1e6, int_time=8.0):
    freqs = jnp.linspace(1.4e9, 1.41e9, 4) if freqs is None else freqs
    times = jnp.linspace(0.0, 120.0, 8) if times is None else times

    return gains_config_validation(cfg, freqs, chan_width, times, int_time)


class TestSeed:

    def test_an_explicit_zero_seed_is_honoured(self):
        """0 is a seed like any other; ``not r_seed`` used to read it as "unset".

        The substitution was not harmless — seed 0 and the default seed 2 drive
        different draws — so a config asking for 0 quietly got someone else's random
        numbers.
        """
        assert validate_gain_scales(scale_config(r_seed=0))["r_seed"] == 0
        assert not jnp.allclose(
            jax.random.normal(jax.random.PRNGKey(0), (8,)),
            jax.random.normal(jax.random.PRNGKey(2), (8,)),
        )

    def test_an_unset_seed_still_defaults(self):
        assert validate_gain_scales(scale_config())["r_seed"] == 2
        assert validate_full(full_config())["r_seed"] == 2

    def test_a_non_integer_seed_is_an_error(self):
        with pytest.raises(ValueError, match="r_seed"):
            validate_gain_scales(scale_config(r_seed=1.5))


class TestPriorWidths:

    @pytest.mark.parametrize("key", ["amp_std", "phase_std"])
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_an_unusable_width_is_an_error(self, key, value):
        """A zero width is a degenerate prior, not an unset one.

        It pins every gain to the mean and gives the fit nothing to move, so it is
        named rather than silently replaced by the default width.
        """
        with pytest.raises(ValueError, match=key):
            validate_gain_scales(scale_config(**{key: value}))

        with pytest.raises(ValueError, match=key):
            validate_full(full_config(**{key: value}))

    def test_unset_widths_still_default(self):
        cfg = validate_gain_scales(scale_config())

        assert cfg["amp_std"] == pytest.approx(0.01)  # 1 % of the default amp_mean
        assert cfg["phase_std"] == pytest.approx(float(np.deg2rad(1)))

    def test_explicit_widths_keep_their_units(self):
        cfg = validate_gain_scales(scale_config(amp_mean=2.0, amp_std=5.0, phase_std=2.0))

        assert cfg["amp_std"] == pytest.approx(5.0 / 100 * 2.0)
        assert cfg["phase_std"] == pytest.approx(float(np.deg2rad(2.0)))

    def test_the_percentage_is_floated_before_it_is_divided(self):
        """``float(std) / 100``, not ``std / 100``: the two orders are not the same.

        An int too large to be represented exactly as a float is rounded by
        ``float()`` but divided exactly by ``/``, so the order of the two decides the
        last bit. Pinned because the conversion moved into a closure and the arithmetic
        of an accepted value must not shift.
        """
        std = 9007199254740993  # 2**53 + 1, the first int a float cannot hold
        cfg = validate_gain_scales(scale_config(amp_mean=1.0, amp_std=std))

        assert cfg["amp_std"] == float(std) / 100 * 1.0
        assert cfg["amp_std"] != std / 100 * 1.0

    @pytest.mark.parametrize("key", ["amp_std", "phase_std"])
    def test_a_non_numeric_width_is_an_error(self, key):
        with pytest.raises(ValueError, match=key):
            validate_gain_scales(scale_config(**{key: "bad"}))


class TestPhaseMean:

    def test_the_default_phase_mean_is_zero(self):
        """Reading 0 honestly costs nothing here: the default it replaced is also 0.

        Which is what makes this key the one behaviour-neutral member of the group —
        worth pinning, since it is the only reason the old ``not phase_mean`` read
        never bit anyone.
        """
        assert validate_gain_scales(scale_config())["phase_mean"] == 0.0
        assert validate_gain_scales(scale_config(phase_mean=0))["phase_mean"] == 0.0
        assert validate_gain_scales(scale_config(phase_mean=0.0))["phase_mean"] == 0.0

    def test_a_non_zero_phase_mean_is_kept(self):
        assert validate_gain_scales(scale_config(phase_mean=-0.3))["phase_mean"] == pytest.approx(-0.3)

    def test_a_non_numeric_phase_mean_is_an_error(self):
        with pytest.raises(ValueError, match="phase_mean"):
            validate_gain_scales(scale_config(phase_mean="bad"))


class TestCorrelationLengths:

    @pytest.mark.parametrize("key", CORR_KEYS)
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_an_unusable_correlation_length_is_an_error(self, key, value):
        """A zero correlation length is a degenerate kernel, not an unset one."""
        with pytest.raises(ValueError, match=key):
            validate_full(full_config(**{key: value}))

    @pytest.mark.parametrize("key", CORR_KEYS)
    def test_a_non_numeric_correlation_length_is_an_error(self, key):
        with pytest.raises(ValueError, match=key):
            validate_full(full_config(**{key: "bad"}))

    def test_unset_correlation_lengths_still_come_from_the_observation(self):
        """The documented default: the extent of the observation along that axis."""
        freqs = jnp.linspace(1.4e9, 1.41e9, 4)
        times = jnp.linspace(0.0, 120.0, 8)
        cfg = validate_full(full_config(), freqs=freqs, times=times)

        assert cfg["amp_corr_freq"] == pytest.approx(1e7)
        assert cfg["phase_corr_freq"] == pytest.approx(1e7)
        assert cfg["amp_corr_time"] == pytest.approx(120.0)
        assert cfg["phase_corr_time"] == pytest.approx(120.0)


class TestMissingKeysAreNamed:

    @pytest.mark.parametrize("key", SCALE_KEYS)
    def test_validate_gain_scales_names_the_missing_key(self, key):
        cfg = scale_config()
        del cfg[key]

        with pytest.raises(ValueError, match=key):
            validate_gain_scales(cfg)

    @pytest.mark.parametrize("key", SCALE_KEYS + CORR_KEYS)
    def test_gains_config_validation_names_the_missing_key(self, key):
        cfg = full_config()
        del cfg[key]
        original = dict(cfg)

        with pytest.raises(ValueError, match=key):
            validate_full(cfg)

        # Still nothing half-normalised behind the failure.
        assert cfg == original

    def test_the_original_key_error_is_chained(self):
        cfg = full_config()
        del cfg["phase_corr_time"]

        with pytest.raises(ValueError) as excinfo:
            validate_full(cfg)

        assert isinstance(excinfo.value.__cause__, KeyError)


# ---------------------------------------------------------------------------
# UnitaryGains
# ---------------------------------------------------------------------------

class TestUnitaryGains:

    def test_state_outputs_shapes(self):
        """state_outputs['gains'] placeholder has shape (n_ant, n_freq, n_time)."""
        n_ant, n_freq, n_time = 4, 3, 6
        cfg = make_gains_config(n_ant=n_ant, n_freq=n_freq, n_time=n_time)
        comp = UnitaryGains()
        comp.setup(cfg)
        a1, a2 = jnp.triu_indices(n_ant, 1)
        n_bl = len(a1)
        assert comp.state_outputs["gains"].shape == (n_ant, n_freq, n_time)
        assert comp.state_outputs["vis_obs"].shape == (n_bl, n_freq, n_time)

    def test_no_learnable_params(self):
        """init_params_base is empty — UnitaryGains has no free parameters."""
        cfg = make_gains_config()
        comp = UnitaryGains()
        comp.setup(cfg)
        assert comp.init_params_base == {}

    def test_forward_vis_obs_equals_sum(self):
        """UnitaryGains applies no actual gains: vis_obs = vis_rfi + vis_ast."""
        n_ant, n_freq, n_time = 4, 2, 6
        cfg = make_gains_config(n_ant=n_ant, n_freq=n_freq, n_time=n_time)
        comp = UnitaryGains()
        comp.setup(cfg)

        state = make_vis_state(n_ant, n_freq, n_time)
        out = comp.build_forward()({}, state, make_constants(comp))

        expected = state["vis_rfi"] + state["vis_ast"]
        assert jnp.allclose(out["vis_obs"], expected)

    def test_forward_preserves_other_state_keys(self):
        """Forward does not drop pre-existing keys in the state dict."""
        cfg = make_gains_config()
        comp = UnitaryGains()
        comp.setup(cfg)
        state = make_vis_state(cfg.n_ant, cfg.n_freq, cfg.n_time)
        state["some_extra_key"] = jnp.array(42.0)
        out = comp.build_forward()({}, state, make_constants(comp))
        assert "some_extra_key" in out


# ---------------------------------------------------------------------------
# GPGains
# ---------------------------------------------------------------------------

class TestGPGains:

    def test_prior_params_shapes(self):
        """Prior mean and Cholesky L arrays have shapes consistent with the GP parameterisation."""
        n_ant, n_freq, n_time = 4, 3, 8
        cfg = make_gains_config(
            n_ant=n_ant, n_freq=n_freq, n_time=n_time,
            amp_corr_time=60.0, phase_corr_time=60.0,
        )
        comp = GPGains()
        comp.setup(cfg)

        n_g = comp.n_g_times
        assert comp.L_gains_amp.shape == (n_g, n_g)
        assert comp.L_gains_phase.shape == (n_g, n_g)
        assert comp.mu_gains_amp.shape == (n_ant, n_freq, n_g)
        assert comp.mu_gains_phase.shape == (n_ant - 1, n_freq, n_g)

    def test_init_params_base_shapes(self):
        """Initial base parameter arrays match the GP amplitude and phase grid sizes."""
        n_ant, n_freq, n_time = 4, 3, 8
        cfg = make_gains_config(
            n_ant=n_ant, n_freq=n_freq, n_time=n_time,
            amp_corr_time=60.0, phase_corr_time=60.0,
        )
        comp = GPGains()
        comp.setup(cfg)

        n_g = comp.n_g_times
        assert "gains_amp_induce_base" in comp.init_params_base
        assert "gains_phase_induce_base" in comp.init_params_base
        assert comp.init_params_base["gains_amp_induce_base"].shape == (n_ant, n_freq, n_g)
        assert comp.init_params_base["gains_phase_induce_base"].shape == (n_ant - 1, n_freq, n_g)

    def test_forward_output_shapes(self):
        """Forward produces gains (n_ant, n_freq, n_time) and vis_obs (n_bl, n_freq, n_time)."""
        n_ant, n_freq, n_time = 4, 2, 8
        cfg = make_gains_config(
            n_ant=n_ant, n_freq=n_freq, n_time=n_time,
            amp_corr_time=60.0, phase_corr_time=60.0,
        )
        comp = GPGains()
        comp.setup(cfg)

        a1, _ = jnp.triu_indices(n_ant, 1)
        n_bl = len(a1)

        state = make_vis_state(n_ant, n_freq, n_time)
        params = {
            "gains_amp_induce_base": comp.init_params_base["gains_amp_induce_base"],
            "gains_phase_induce_base": comp.init_params_base["gains_phase_induce_base"],
        }
        out = comp.build_forward()(params, state, make_constants(comp))

        assert out["gains"].shape == (n_ant, n_freq, n_time)
        assert out["vis_obs"].shape == (n_bl, n_freq, n_time)

    def test_forward_gains_at_prior_mean_amplitude(self):
        """At init params (prior mean), gain amplitudes should be close to amp_mean."""
        n_ant, n_freq, n_time = 4, 2, 8
        amp_mean = 1.5
        cfg = make_gains_config(
            n_ant=n_ant, n_freq=n_freq, n_time=n_time,
            amp_mean=amp_mean, amp_std=0.1,
            amp_corr_time=60.0, phase_corr_time=60.0,
        )
        comp = GPGains()
        comp.setup(cfg)

        state = make_vis_state(n_ant, n_freq, n_time)
        params = {
            "gains_amp_induce_base": comp.init_params_base["gains_amp_induce_base"],
            "gains_phase_induce_base": comp.init_params_base["gains_phase_induce_base"],
        }
        out = comp.build_forward()(params, state, make_constants(comp))

        gain_amps = jnp.abs(out["gains"])
        assert jnp.allclose(gain_amps, amp_mean, atol=1e-3), (
            f"Expected gain amplitude ≈ {amp_mean}, got mean {float(gain_amps.mean()):.4f}"
        )

    def test_forward_gains_last_antenna_phase_zero(self):
        """Last antenna phase is fixed to zero (reference antenna)."""
        n_ant, n_freq, n_time = 4, 2, 8
        cfg = make_gains_config(
            n_ant=n_ant, n_freq=n_freq, n_time=n_time,
            phase_std=1.0, amp_corr_time=60.0, phase_corr_time=60.0,
        )
        comp = GPGains()
        comp.setup(cfg)

        state = make_vis_state(n_ant, n_freq, n_time)
        params = {
            "gains_amp_induce_base": comp.init_params_base["gains_amp_induce_base"],
            "gains_phase_induce_base": comp.init_params_base["gains_phase_induce_base"],
        }
        out = comp.build_forward()(params, state, make_constants(comp))

        last_phase = jnp.angle(out["gains"][-1])  # (n_freq, n_time)
        assert jnp.allclose(last_phase, 0.0, atol=1e-6)

    def test_forward_output_is_complex(self):
        """gains and vis_obs from the forward pass are complex-valued."""
        cfg = make_gains_config(amp_corr_time=60.0, phase_corr_time=60.0)
        comp = GPGains()
        comp.setup(cfg)

        state = make_vis_state(cfg.n_ant, cfg.n_freq, cfg.n_time)
        params = {
            "gains_amp_induce_base": comp.init_params_base["gains_amp_induce_base"],
            "gains_phase_induce_base": comp.init_params_base["gains_phase_induce_base"],
        }
        out = comp.build_forward()(params, state, make_constants(comp))

        assert jnp.issubdtype(out["gains"].dtype, jnp.complexfloating)
        assert jnp.issubdtype(out["vis_obs"].dtype, jnp.complexfloating)

    @pytest.mark.parametrize("n_ant,n_freq,n_time", [
        (2, 1, 4),
        (5, 3, 12),
        (8, 4, 16),
    ])
    def test_setup_and_forward_various_sizes(self, n_ant, n_freq, n_time):
        """Setup and forward succeed end-to-end for the given (n_ant, n_freq, n_time)."""
        cfg = make_gains_config(
            n_ant=n_ant, n_freq=n_freq, n_time=n_time,
            amp_corr_time=60.0, phase_corr_time=60.0,
        )
        comp = GPGains()
        comp.setup(cfg)

        a1, _ = jnp.triu_indices(n_ant, 1)
        n_bl = len(a1)
        state = make_vis_state(n_ant, n_freq, n_time)
        params = {
            "gains_amp_induce_base": comp.init_params_base["gains_amp_induce_base"],
            "gains_phase_induce_base": comp.init_params_base["gains_phase_induce_base"],
        }
        out = comp.build_forward()(params, state, make_constants(comp))

        assert out["vis_obs"].shape == (n_bl, n_freq, n_time)
        assert out["gains"].shape == (n_ant, n_freq, n_time)
        assert jnp.all(jnp.isfinite(jnp.abs(out["gains"])))
        assert jnp.all(jnp.isfinite(jnp.abs(out["vis_obs"])))

    def test_build_set_params_samples_correct_shapes(self):
        """build_set_params must produce correctly shaped samples inside a NumPyro trace."""
        n_ant, n_freq, n_time = 4, 3, 8
        cfg = make_gains_config(
            n_ant=n_ant, n_freq=n_freq, n_time=n_time,
            amp_corr_time=60.0, phase_corr_time=60.0,
        )
        comp = GPGains()
        comp.setup(cfg)
        n_g = comp.n_g_times

        set_params = comp.build_set_params()
        with numpyro.handlers.seed(rng_seed=0):
            params = set_params({})

        assert "gains_amp_induce_base" in params
        assert "gains_phase_induce_base" in params
        assert params["gains_amp_induce_base"].shape == (n_ant, n_freq, n_g)
        assert params["gains_phase_induce_base"].shape == (n_ant - 1, n_freq, n_g)

    def test_forward_transform_roundtrips(self):
        """inv_transform(forward_transform(x)) == x and vice versa."""
        n_ant, n_freq, n_time = 4, 3, 8
        cfg = make_gains_config(
            n_ant=n_ant, n_freq=n_freq, n_time=n_time,
            amp_corr_time=60.0, phase_corr_time=60.0,
        )
        comp = GPGains()
        comp.setup(cfg)
        n_g = comp.n_g_times

        base = jax.random.normal(jax.random.PRNGKey(42), (n_ant, n_freq, n_g))
        assert_transform_roundtrip(comp, base, comp.L_gains_amp, comp.mu_gains_amp)
