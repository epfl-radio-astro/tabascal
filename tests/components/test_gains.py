"""Tests for tabascal.components.gains — UnitaryGains, GPGains, and config validation."""

import re

import pytest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro

from tabascal.components.gains import (
    _GP_JITTER_FLOOR,
    UnitaryGains,
    GPGains,
    gains_config_validation,
    gp_jitter,
    validate_gain_scales,
)
from tabascal.gp import base_kernel

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
        assert result["amp_std"] == pytest.approx(0.2)  # 20 % of amp_mean=1.0
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

#: What an unset width resolves to, in the units the config writes it in — a
#: percentage of ``amp_mean`` and degrees. Restated here rather than imported from
#: :mod:`tabascal.components.gains`, so these are the numbers the base config
#: comments and docs/config.md quote and not whatever the module happens to hold.
DEFAULT_AMP_STD_PERCENT = 20.0
DEFAULT_PHASE_STD_DEGREES = 180.0


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

        assert cfg["amp_std"] == pytest.approx(
            DEFAULT_AMP_STD_PERCENT / 100
        )  # of the default amp_mean
        assert cfg["phase_std"] == pytest.approx(
            float(jnp.deg2rad(DEFAULT_PHASE_STD_DEGREES))
        )

    def test_the_default_amp_width_is_a_percentage_of_amp_mean(self):
        """The default is a *fraction* of amp_mean, exactly as an explicit value is."""
        cfg = validate_gain_scales(scale_config(amp_mean=2.5))

        assert cfg["amp_std"] == pytest.approx(DEFAULT_AMP_STD_PERCENT / 100 * 2.5)

    @pytest.mark.parametrize(
        "key, written",
        [("amp_std", DEFAULT_AMP_STD_PERCENT), ("phase_std", DEFAULT_PHASE_STD_DEGREES)],
    )
    def test_an_unset_width_resolves_exactly_as_writing_the_default_out_does(
        self, key, written
    ):
        """The documented default and the value written into a config are one number.

        Which is what makes the documentation checkable: a config that says
        ``amp_std: 20`` is the config that says ``amp_std: null``, bit for bit, so
        the two conversion paths cannot drift apart unnoticed.
        """
        assert validate_gain_scales(scale_config())[key] == validate_gain_scales(
            scale_config(**{key: written})
        )[key]

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


class TestTheDefaultPhasePriorIsEffectivelyUniform:
    """The unset ``phase_std`` has to say nothing about the phase, and let the fit move it.

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
        default = validate_gain_scales(scale_config())["phase_std"]

        assert self.wrapped_ripple(default) < 0.02
        # The value it replaces is not remotely flat, which is the point.
        assert self.wrapped_ripple(float(jnp.deg2rad(1))) > 1.0

    def draw_phases(self, comp, n_draws=24):
        """Gain phases drawn through the component's own prior path.

        The last antenna's phase is pinned to zero by :meth:`GPGains.build_forward`
        (the overall phase is unobservable), so it is dropped: it is a gauge choice
        rather than a draw from the prior.
        """

        forward = comp.build_forward()
        constants = make_constants(comp)
        state = make_vis_state(comp.n_ant, comp.n_freq, comp.n_time)
        shapes = {
            "gains_amp_induce_base": (comp.n_ant, comp.n_freq, comp.n_g_times),
            "gains_phase_induce_base": (comp.n_ant - 1, comp.n_freq, comp.n_g_times),
        }

        phases = []
        for seed in range(n_draws):
            keys = jax.random.split(jax.random.PRNGKey(seed), 2)
            params = {
                name: jax.random.normal(key, shape)
                for key, (name, shape) in zip(keys, shapes.items())
            }
            phases.append(jnp.angle(forward(params, state, constants)["gains"][:-1]))

        return np.asarray(jnp.concatenate([p.ravel() for p in phases]))

    def make_comp(self, phase_std):
        """A GPGains over enough antennas for the draws to be worth histogramming."""

        cfg = make_gains_config(
            n_ant=33, n_freq=4, n_time=8, amp_std=None, phase_std=phase_std
        )
        comp = GPGains()
        comp.setup(cfg)

        return comp

    def test_the_default_prior_covers_the_circle(self):
        comp = self.make_comp(phase_std=None)

        # The Cholesky of the phase kernel is taken at sigma^2, and its jitter is
        # absolute: a wide prior makes 1e-8 relatively smaller. Checked here because
        # it is the one part of the widening that could fail numerically, and it
        # would fail in single precision first.
        assert bool(jnp.all(jnp.isfinite(comp.L_gains_phase)))

        phases = self.draw_phases(comp)
        counts, _ = np.histogram(phases, bins=self.N_BINS, range=(-np.pi, np.pi))

        assert counts.min() > 0
        assert counts.max() / counts.min() < self.MAX_BIN_RATIO
        # Mean resultant length: exp(-sigma^2/2) for a wrapped normal, so ~7e-3 at
        # 180 degrees against ~1 for any prior worth calling concentrated.
        assert abs(np.mean(np.exp(1j * phases))) < 0.1

    def test_a_one_degree_prior_does_not(self):
        """The teeth of the test above: the default it replaces fails both bounds."""

        phases = self.draw_phases(self.make_comp(phase_std=1.0))
        counts, _ = np.histogram(phases, bins=self.N_BINS, range=(-np.pi, np.pi))

        assert counts.min() == 0
        assert abs(np.mean(np.exp(1j * phases))) > 0.9


#: Correlation times, in seconds, against the 1200 s observation
#: :meth:`TestTheGPStaysFactorisable.make_comp` builds. The node grid is
#: ``~2 * extent / corr_time`` wide, so these run from 4 nodes to 243 -- and every row
#: from 60 s down returned a NaN Cholesky in single precision at the widened default
#: widths, before the jitter was tied to the kernel variance.
GP_CORR_TIMES = [1200.0, 600.0, 300.0, 120.0, 60.0, 30.0, 20.0, 10.0]


class TestTheGPStaysFactorisable:
    """A wide prior must not cost the Gaussian process its Cholesky.

    ``gp.cholesky`` adds its jitter ABSOLUTELY, after the kernel has been scaled by
    the prior variance, so a fixed 1e-8 regularises a squared-exponential Gram matrix
    by ``1e-8 / var`` in relative terms -- 3.3e-5 at the old 1-degree phase width and
    1e-9 at 180 degrees, which is below fp32's ~1.2e-7 of precision. Widening the
    default without tying the jitter to the variance therefore turned the *default*
    configuration into a NaN Cholesky in single precision, and a NaN
    ``init_params_base`` behind it, for any correlation time short against the
    observation.

    The worst case is the ASYMMETRIC one: the node grid is built from the shorter of
    the two correlation times while each kernel keeps its own length scale, so a short
    ``phase_corr_time`` puts the long-length-scale AMPLITUDE kernel on a fine grid.
    Both sides are covered below.
    """

    def make_comp(self, phase_corr_time, amp_corr_time=None, amp_std=None, phase_std=None):
        """GPGains over a 1200 s observation, at the default widths unless overridden."""

        cfg = make_gains_config(
            n_ant=4, n_freq=2, n_time=240, amp_std=amp_std, phase_std=phase_std,
            amp_corr_time=amp_corr_time, phase_corr_time=phase_corr_time,
        )
        cfg.times = jnp.linspace(0.0, 1200.0, 240)
        cfg.int_time = float(cfg.times[1] - cfg.times[0])
        comp = GPGains()
        comp.setup(cfg)

        return comp

    def assert_usable(self, comp):
        for name in (
            "L_gains_amp", "L_gains_phase", "resample_amp", "resample_phase",
        ):
            assert bool(jnp.all(jnp.isfinite(getattr(comp, name)))), name
        for name, value in comp.init_params_base.items():
            assert bool(jnp.all(jnp.isfinite(value))), name

    @pytest.mark.parametrize("corr_time", GP_CORR_TIMES)
    def test_the_defaults_factorise_at_every_correlation_time(self, corr_time):
        """Both correlation times short together: the node grid is fine for both."""

        self.assert_usable(self.make_comp(corr_time, amp_corr_time=corr_time))

    @pytest.mark.parametrize("corr_time", GP_CORR_TIMES)
    def test_the_defaults_factorise_with_a_short_phase_correlation_time(self, corr_time):
        """The asymmetric case: ``amp_corr_time`` null, so the amplitude kernel is
        the smooth one sitting on the grid the phase asked for."""

        self.assert_usable(self.make_comp(corr_time, amp_corr_time=None))


class TestTheJitterIsRelativeAboveTheFloor:
    """Tied to the kernel variance, but never below the absolute value it always had.

    ``chol(var (SE + R I)) = sqrt(var) chol(SE + R I)``, so a jitter proportional to
    the variance makes the factorisation scale-free -- whether it succeeds depends on
    the node grid and ``R``, not on the units the prior is written in. The floor is
    what keeps every prior narrow enough for 1e-8 to dominate regularised exactly as
    it was before, which is what leaves the pipeline references where they are.
    """

    def test_the_floor_applies_to_narrow_priors(self):
        assert gp_jitter(1e-4) == 1e-8
        assert gp_jitter(0.0) == 1e-8

    def test_the_jitter_is_relative_to_a_wide_prior(self):
        assert gp_jitter(9.87) == pytest.approx(9.87e-5)

    def test_the_hinge_is_where_the_two_meet(self):
        hinge = 1e-8 / 1e-5
        assert gp_jitter(hinge) == pytest.approx(1e-8)
        assert gp_jitter(10 * hinge) == pytest.approx(1e-7)

    @pytest.mark.parametrize(
        "std, label",
        [(1.0 / 100 * 1.0, "amp_std: 1 %"), (float(jnp.deg2rad(1.0)), "phase_std: 1 deg")],
    )
    def test_the_recorded_pipeline_widths_are_bit_identical(self, std, label):
        """The widths the GPGains pipeline case writes out must factorise unchanged.

        The jitter is the ONLY argument this change touches, and at these widths it is
        still exactly the 1e-8 the kernels were always handed — so every call is the
        call it was before, on any grid, and the recorded chi2 cannot move. Asserting
        that one number is the whole of the guarantee; comparing two ``cholesky``
        calls that differ only in an argument just shown to be equal would prove
        nothing beyond it.
        """
        assert gp_jitter(std**2) == _GP_JITTER_FLOOR == 1e-8, label


class TestTheJitterStaysOffTheResamplingMatrix:
    """The gain resampling matrix is a CROSS-covariance and must carry no jitter.

    It is never inverted or factorised — it multiplies the node values on their way
    to the observation grid — so a diagonal added to it is not regularisation but a
    bias on every gain the run reports. ``gp.resampling_kernel`` used to add one
    whenever the matrix came out square, which the node grid and the observation grid
    do whenever they happen to have the same length: ``gp.get_times`` lays down about
    two nodes per correlation length, so a correlation time of about twice the
    integration time is enough. That is what the 1200 s, 240-sample observation below
    gets at 10.1 s, and the jitter it attracted was an ABSOLUTE 1e-3 — a hundred times
    the jitter the node covariance is regularised with at these widths.

    See ``tests/test_gp.py`` for the kernels themselves; this is the reachability.
    """

    CORR_TIME = 10.1

    def make_comp(self):
        cfg = make_gains_config(
            n_ant=4, n_freq=2, n_time=240,
            amp_corr_time=self.CORR_TIME, phase_corr_time=self.CORR_TIME,
        )
        cfg.times = jnp.linspace(0.0, 1200.0, 240)
        cfg.int_time = float(cfg.times[1] - cfg.times[0])
        comp = GPGains()
        comp.setup(cfg)

        return comp

    def test_the_node_grid_can_be_as_long_as_the_observation(self):
        comp = self.make_comp()

        assert comp.n_g_times == len(comp.times)

    @pytest.mark.parametrize("name, std_attr", [
        ("resample_amp", "gp_amp_std"), ("resample_phase", "gp_phase_std"),
    ])
    def test_the_resampling_matrices_are_the_bare_conditional(self, name, std_attr, exact_rtol):
        comp = self.make_comp()
        var = getattr(comp, std_attr) ** 2
        # K_s K^-1, with the jitter on the inverted matrix alone. base_kernel takes no
        # jitter at all, so the reference cannot inherit the defect under test.
        reference = base_kernel(comp.g_times, comp.times, var, self.CORR_TIME) @ jnp.linalg.inv(
            base_kernel(comp.g_times, comp.g_times, var, self.CORR_TIME)
            + gp_jitter(var) * jnp.eye(comp.n_g_times)
        )

        difference = jnp.max(jnp.abs(getattr(comp, name) - reference))

        assert float(difference) <= exact_rtol * float(jnp.max(jnp.abs(reference)))


class TestAnUnfactorisableGPIsNamed:
    """Past the jitter's reach the run stops with an error, rather than reporting NaN.

    The jitter buys a wide margin, not an unlimited one: a correlation length short
    enough against the observation puts hundreds of nodes on the grid and the
    squared-exponential Gram matrix goes singular faster than any reasonable jitter
    regularises it. That is a real limit of the working precision, so it is named --
    where before it was a NaN Cholesky that flowed silently into the initial
    parameters and out through every gain the run reported.
    """

    #: The attribution clause, as opposed to the sentence listing both key names to
    #: explain where the grid comes from. Matching the latter would pass whatever the
    #: error blamed, which is exactly the bug this class exists to catch.
    ATTRIBUTION = r"so gains\.{} is asking"

    def over_resolve(self, short_key, precision):
        """A GP whose two correlation times are far enough apart to be unfactorisable.

        Whichever key is given the short time sets the node grid, so that is the key
        the error has to name -- lengthening the other one, which is already null and
        therefore maximal, could not coarsen the grid by a single node.
        """
        if precision == "double":
            pytest.skip("double precision factorises every grid this test can build")

        other = "amp_corr_time" if short_key == "phase_corr_time" else "phase_corr_time"

        return TestTheGPStaysFactorisable().make_comp(**{short_key: 2.0, other: None})

    @pytest.mark.parametrize("short_key", ["phase_corr_time", "amp_corr_time"])
    def test_an_over_resolved_gp_blames_the_key_that_set_the_grid(
        self, short_key, precision
    ):
        """Both directions of the asymmetry: the error names the SHORTER key.

        Parameterised over which side is short because the amplitude and phase
        kernels are checked separately and each used to name its own key -- so on an
        asymmetric grid the failure of one kernel would blame a correlation time that
        was not responsible for the grid and could not be lengthened any further.
        """
        with pytest.raises(RuntimeError, match=self.ATTRIBUTION.format(short_key)):
            self.over_resolve(short_key, precision)

    @pytest.mark.parametrize("short_key", ["phase_corr_time", "amp_corr_time"])
    def test_the_error_does_not_blame_the_maximal_key(self, short_key, precision):
        """The remedy has to be actionable: never "lengthen" an already-null key."""
        other = "amp_corr_time" if short_key == "phase_corr_time" else "phase_corr_time"

        with pytest.raises(RuntimeError) as excinfo:
            self.over_resolve(short_key, precision)

        assert not re.search(self.ATTRIBUTION.format(other), str(excinfo.value))

    def test_the_error_says_what_to_do_about_it(self, precision):
        with pytest.raises(RuntimeError, match="too finely resolved") as excinfo:
            self.over_resolve("phase_corr_time", precision)

        message = str(excinfo.value)
        assert "single precision" in message
        assert "model.precision: double" in message
        assert "ConstGains" in message

    def test_an_equally_short_pair_names_both(self, precision):
        """With the two correlation times equal, both keys set the grid.

        A symmetric grid is the one arrangement that does not fail, so this pins the
        naming rather than the failure: lengthening only one of an equal pair leaves
        the other setting the same grid, so both have to be named.
        """
        if precision == "double":
            pytest.skip("double precision factorises every grid this test can build")

        comp = TestTheGPStaysFactorisable().make_comp(600.0, amp_corr_time=600.0)

        assert comp.grid_corr_keys == ["amp_corr_time", "phase_corr_time"]


class TestResolvedValuesArePrinted:
    """The run's own record of what it used, in the units the config writes.

    ``amp_std`` is stored as a fraction of ``amp_mean`` and ``phase_std`` in radians,
    so both are converted back on the way out; the defaults have to survive that
    round trip and read as the numbers the documentation quotes.
    """

    def test_the_defaults_print_in_config_units(self, capsys):
        validate_full(full_config())
        out = capsys.readouterr().out

        assert f"Using Gains amplitude std : {DEFAULT_AMP_STD_PERCENT:.1f} %" in out
        assert f"Using Gains phase std : {DEFAULT_PHASE_STD_DEGREES:.1f} degrees" in out

    def test_explicit_values_print_as_written(self, capsys):
        validate_full(full_config(amp_mean=2.0, amp_std=7.0, phase_std=45.0))
        out = capsys.readouterr().out

        assert "Using Gains amplitude std : 7.0 %" in out
        assert "Using Gains phase std : 45.0 degrees" in out


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
