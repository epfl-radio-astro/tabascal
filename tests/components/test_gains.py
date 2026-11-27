"""Tests for tabascal.components.gains — UnitaryGains and the gain prior scales.

``ConstGains``, the one gain component that fits anything, has its own module
(``test_const_gains.py``); what lives here is the validation of the prior it
reads (:class:`ConstGainsConfig` and its :meth:`~ConstGainsConfig.prior_scales`),
and the component that reads none of it.
"""

import pytest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from tabascal.components.gains import ConstGainsConfig, UnitaryGains

from .conftest import make_constants


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
# None means "unset", 0 means 0
# ---------------------------------------------------------------------------

#: Every value that is not a width: zero included, since a zero-width prior is a
#: degenerate Normal rather than an unset one.
UNUSABLE = [0, 0.0, -1.0, float("nan"), float("inf"), -float("inf")]

#: What an unset width resolves to, in the units the config writes it in — a
#: percentage of ``amp_mean`` and degrees. Restated here rather than imported from
#: :mod:`tabascal.components.gains`, so these are the numbers the base config
#: comments and docs/config.md quote and not whatever the module happens to hold.
DEFAULT_AMP_STD_PERCENT = 20.0
DEFAULT_PHASE_STD_DEGREES = 180.0


def scales(**options):
    """The prior scales ``ConstGains`` runs with, for these options."""
    return ConstGainsConfig(**options).prior_scales()


def test_an_old_r_seed_is_refused_as_an_unknown_option():
    """``r_seed`` was retired for ``inference.seed``; a config still carrying it
    under the component is told so, rather than having it silently ignored."""
    with pytest.raises(ValueError, match="r_seed"):
        ConstGainsConfig.model_validate({"r_seed": "x"})


class TestPriorWidths:

    @pytest.mark.parametrize("key", ["amp_std_pct", "phase_std_deg"])
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_an_unusable_width_is_an_error(self, key, value):
        """A zero width is a degenerate prior, not an unset one.

        It pins every gain to the mean and gives the fit nothing to move, so it is
        named rather than silently replaced by the default width.
        """
        with pytest.raises(ValueError, match=key):
            ConstGainsConfig(**{key: value})

    def test_unset_widths_still_default(self):
        cfg = scales()

        assert cfg["amp_std"] == pytest.approx(
            DEFAULT_AMP_STD_PERCENT / 100
        )  # of the default amp_mean
        assert cfg["phase_std"] == pytest.approx(
            float(jnp.deg2rad(DEFAULT_PHASE_STD_DEGREES))
        )

    def test_the_default_amp_width_is_a_percentage_of_amp_mean(self):
        """The default is a *fraction* of amp_mean, exactly as an explicit value is."""
        cfg = scales(amp_mean=2.5)

        assert cfg["amp_std"] == pytest.approx(DEFAULT_AMP_STD_PERCENT / 100 * 2.5)

    @pytest.mark.parametrize(
        "key, written",
        [("amp_std_pct", DEFAULT_AMP_STD_PERCENT), ("phase_std_deg", DEFAULT_PHASE_STD_DEGREES)],
    )
    def test_an_unset_width_resolves_exactly_as_writing_the_default_out_does(
        self, key, written
    ):
        """The documented default and the value written into a config are one number.

        Which is what makes the documentation checkable: a config that leaves
        ``amp_std_pct`` out is the config that says ``amp_std_pct: 20``, bit for bit, so
        the two conversion paths cannot drift apart unnoticed. ``null`` is not a
        third spelling of it.
        """
        scale = {"amp_std_pct": "amp_std", "phase_std_deg": "phase_std"}[key]
        assert scales()[scale] == scales(**{key: written})[scale]
        with pytest.raises(ValueError, match=f"{key}: null"):
            scales(**{key: None})

    def test_explicit_widths_keep_their_units(self):
        cfg = scales(amp_mean=2.0, amp_std_pct=5.0, phase_std_deg=2.0)

        assert cfg["amp_std"] == pytest.approx(5.0 / 100 * 2.0)
        assert cfg["phase_std"] == pytest.approx(float(np.deg2rad(2.0)))

    def test_the_percentage_is_floated_before_it_is_divided(self):
        """``float(std) / 100``, not ``std_jy / 100``: the two orders are not the same.

        An int too large to be represented exactly as a float is rounded by
        ``float()`` but divided exactly by ``/``, so the order of the two decides the
        last bit. Pinned because the conversion moved into a closure and the arithmetic
        of an accepted value must not shift.
        """
        std = 9007199254740993  # 2**53 + 1, the first int a float cannot hold
        cfg = scales(amp_mean=1.0, amp_std_pct=std)

        assert cfg["amp_std"] == float(std) / 100 * 1.0
        assert cfg["amp_std"] != std / 100 * 1.0

    @pytest.mark.parametrize("key", ["amp_std_pct", "phase_std_deg"])
    def test_a_non_numeric_width_is_an_error(self, key):
        with pytest.raises(ValueError, match=key):
            ConstGainsConfig(**{key: "bad"})


class TestTheDefaultPhasePriorIsEffectivelyUniform:
    """The unset ``phase_std_deg`` has to say nothing about the phase, and let the fit move it.

    The half of this that a component can be driven through -- draws from the prior
    covering the circle -- lives in ``test_const_gains.py``, beside the component
    that reads the width. What is checked here is the width itself.

    Two separate properties, both of which the old 1-degree default failed:

    * the prior the *model* sees is on an angle, so what matters is the WRAPPED
      normal. Its density is
      ``p(t) = (1 + 2 sum_k exp(-(k sigma)^2 / 2) cos(k t)) / 2 pi``, flat to within
      ``2 exp(-sigma^2 / 2)`` — 1.4 % at sigma = 180 degrees;
    * the parameterisation is non-centred (``phase = mean + sigma z`` with
      ``z ~ N(0, 1)``), so ``sigma`` also scales the map from the fitted coordinate to
      the phase. Under a per-coordinate optimiser the step in ``z`` is set by
      ``opt.epsilon`` whatever the gradient, so the phase moves by ``sigma * epsilon``
      per iteration: at 1 degree and the default ``epsilon`` of 1e-2 that is 0.01
      degrees an iteration, and 500 iterations cannot cross a radian.
    """

    #: Coarse enough that a genuinely flat prior passes on sampling noise alone, and
    #: far too tight for anything concentrated: a 1-degree prior puts every draw in
    #: one bin.
    N_BINS = 12
    MAX_BIN_RATIO = 1.5

    def wrapped_ripple(self, sigma: float) -> float:
        """Peak relative deviation of the wrapped normal from uniform, at t = 0."""

        return 2 * sum(np.exp(-0.5 * (k * sigma) ** 2) for k in range(1, 200))

    def test_the_wrapped_default_prior_is_flat_over_the_circle(self):
        default = scales()["phase_std"]

        assert self.wrapped_ripple(default) < 0.02
        # The value it replaces is not remotely flat, which is the point.
        assert self.wrapped_ripple(float(jnp.deg2rad(1))) > 1.0


class TestPhaseMean:

    def test_the_default_phase_mean_is_zero(self):
        """Reading 0 honestly costs nothing here: the default it replaced is also 0.

        Which is what makes this key the one behaviour-neutral member of the group —
        worth pinning, since it is the only reason the old ``not phase_mean`` read
        never bit anyone.
        """
        assert scales()["phase_mean"] == 0.0
        assert scales(phase_mean_deg=0)["phase_mean"] == 0.0
        assert scales(phase_mean_deg=0.0)["phase_mean"] == 0.0

    def test_a_non_zero_phase_mean_is_kept_and_read_in_degrees(self):
        assert scales(phase_mean_deg=-30.0)["phase_mean"] == pytest.approx(np.deg2rad(-30.0))

    def test_a_non_numeric_phase_mean_is_an_error(self):
        with pytest.raises(ValueError, match="phase_mean"):
            ConstGainsConfig(phase_mean_deg="bad")


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

    def test_it_reads_nothing_from_any_config_section(self):
        cfg = make_gains_config()
        comp = UnitaryGains()
        comp.setup(cfg)

        state = make_vis_state(cfg.n_ant, cfg.n_freq, cfg.n_time)
        out = comp.build_forward()({}, state, make_constants(comp))

        assert jnp.allclose(out["gains"], 1.0 + 0j)
        assert jnp.allclose(out["vis_obs"], state["vis_rfi"] + state["vis_ast"])

    def test_forward_preserves_other_state_keys(self):
        """Forward does not drop pre-existing keys in the state dict."""
        cfg = make_gains_config()
        comp = UnitaryGains()
        comp.setup(cfg)
        state = make_vis_state(cfg.n_ant, cfg.n_freq, cfg.n_time)
        state["some_extra_key"] = jnp.array(42.0)
        out = comp.build_forward()({}, state, make_constants(comp))
        assert "some_extra_key" in out
