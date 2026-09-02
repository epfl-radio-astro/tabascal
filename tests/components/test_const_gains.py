"""ConstGains — one complex direction-independent gain per antenna.

The component is a gauge as much as a model: the overall amplitude and the overall
phase of a per-antenna gain are not observable, so they are removed by construction
rather than fitted. What is worth locking down is therefore that the gauge actually
holds (zero-sum log amplitude, reference-antenna phase pinned), that anything read in
from outside is *projected* into it rather than taken as given, and that the two
identifiability rules of issue #124 — a rigid sky before the flux scale may be freed,
and a warning when the RFI model already carries per-antenna freedom — are enforced
where the model is assembled rather than left to the fit to discover.
"""

import importlib.util
import os
import warnings
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import optax
import pytest

from tabascal.components.gains import ConstGains, validate_gain_scales

from .conftest import make_constants


requires_casacore = pytest.mark.skipif(
    importlib.util.find_spec("casacore") is None,
    reason="python-casacore is not installed",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_flags(n_ant, n_freq, n_time, flagged_ants=()):
    """Flags (n_bl, n_freq, n_time) with every baseline of *flagged_ants* flagged."""
    a1, a2 = np.triu_indices(n_ant, 1)
    flags = np.zeros((len(a1), n_freq, n_time), dtype=bool)
    for ant in flagged_ants:
        flags[(a1 == ant) | (a2 == ant)] = True
    return flags


def make_const_gains_config(
    n_ant=5,
    n_freq=3,
    n_time=4,
    amp_mean=1.0,
    amp_std=10.0,      # percent, as the config gives it
    phase_mean=0.0,
    phase_std=30.0,    # degrees, as the config gives it
    init="prior",
    ref_ant=None,
    fix_flux_scale=True,
    components=("gains:ConstGains",),
    flagged_ants=(),
    flags=None,
):
    """A minimal mock TabConfig carrying everything ConstGains reads."""
    a1, a2 = np.triu_indices(n_ant, 1)
    if flags is None:
        flags = make_flags(n_ant, n_freq, n_time, flagged_ants)

    return SimpleNamespace(
        n_ant=n_ant,
        n_bl=len(a1),
        n_freq=n_freq,
        n_time=n_time,
        a1=jnp.asarray(a1),
        a2=jnp.asarray(a2),
        flags=jnp.asarray(np.asarray(flags)),
        args={
            "gains": {
                "r_seed": 123,
                "amp_mean": amp_mean,
                "amp_std": amp_std,
                "phase_mean": phase_mean,
                "phase_std": phase_std,
                "init": init,
                "ref_ant": ref_ant,
                "fix_flux_scale": fix_flux_scale,
            },
            "model": {"components": list(components)},
        },
    )


def make_vis_state(n_ant, n_freq, n_time, seed=0):
    a1, _ = np.triu_indices(n_ant, 1)
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 4)
    shape = (len(a1), n_freq, n_time)
    return {
        "vis_rfi": jax.random.normal(keys[0], shape) + 1j * jax.random.normal(keys[1], shape),
        "vis_ast": jax.random.normal(keys[2], shape) + 1j * jax.random.normal(keys[3], shape),
    }


def random_params(comp, seed=0, scale=1.0):
    """A random draw of the base parameters, as the prior would give them.

    Shapes are taken from the component, since the amplitude block is
    ``n_ant - 1`` long under the zero-sum gauge and ``n_ant`` long without it.
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), 2)
    return {
        name: scale * jax.random.normal(key, comp.init_params_base[name].shape)
        for name, key in zip(("gains_amp_base", "gains_phase_base"), keys)
    }


def helmert_basis(n):
    """A textbook Helmert contrast basis, built independently of the component.

    ``n - 1`` orthonormal columns, each summing to zero, so they span exactly the
    zero-sum subspace of R^n.
    """
    H = np.zeros((n, n - 1))
    for k in range(1, n):
        H[:k, k - 1] = 1.0 / np.sqrt(k * (k + 1))
        H[k, k - 1] = -k / np.sqrt(k * (k + 1))
    return H


def run_forward(comp, params, state=None, seed=0):
    if state is None:
        state = make_vis_state(comp.n_ant, comp.n_freq, comp.n_time, seed=seed)
    return comp.build_forward()(params, state, make_constants(comp))


def gauge_project(g, ref_ant, fix_flux_scale=True):
    """The gauge ConstGains works in, computed independently of the component."""
    g = np.asarray(g, dtype=complex)
    scale = np.exp(np.log(np.abs(g)).mean()) if fix_flux_scale else 1.0
    return g / scale * np.exp(-1j * np.angle(g[ref_ant]))


def write_test_caltable(path, gains, times=None):
    """The minimal caltable :func:`tabascal.ms.read_caltable` reads.

    ``TIME``/``ANTENNA1``/``CPARAM``/``FLAG`` only — no subtables, which
    ``read_caltable`` treats as a table that simply does not name its channels.
    """
    from casacore.tables import (
        makearrcoldesc,
        makescacoldesc,
        maketabdesc,
        table,
    )

    gains = np.asarray(gains, dtype=complex)
    n_ant, n_freq, n_time = gains.shape
    if times is None:
        times = 5e9 + 10.0 * np.arange(n_time, dtype=float)

    n_row = n_ant * n_time
    g_rows = np.transpose(gains, (2, 0, 1)).reshape(n_row, n_freq)
    bad = ~np.isfinite(g_rows) | (g_rows == 0)
    solved = np.where(bad, complex(np.nan, np.nan), g_rows)

    desc = maketabdesc(
        [
            makescacoldesc("TIME", 0.0, valuetype="double"),
            makescacoldesc("ANTENNA1", 0, valuetype="int"),
            makearrcoldesc("CPARAM", 0j, ndim=2, valuetype="complex"),
            makearrcoldesc("FLAG", False, ndim=2, valuetype="boolean"),
        ]
    )
    with table(path, desc, nrow=n_row, ack=False) as tb:
        tb.putcol("TIME", np.repeat(np.asarray(times, dtype=float), n_ant))
        tb.putcol("ANTENNA1", np.tile(np.arange(n_ant, dtype=np.int32), n_time))
        tb.putcol("CPARAM", np.repeat(solved[:, :, None], 2, axis=2).astype(np.complex64))
        tb.putcol("FLAG", np.repeat(bad[:, :, None], 2, axis=2))

    return path


# ---------------------------------------------------------------------------
# The gauge
# ---------------------------------------------------------------------------


class TestGauge:

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_log_amplitude_is_zero_sum(self, seed, exact_rtol):
        """mean(log|g|) is zero for any parameter draw: the flux scale is not fitted."""
        comp = ConstGains()
        comp.setup(make_const_gains_config(n_ant=7))

        out = run_forward(comp, random_params(comp, seed=seed))
        mean_log_amp = jnp.mean(jnp.log(jnp.abs(out["gains"][:, 0, 0])))

        assert abs(float(mean_log_amp)) < exact_rtol

    @pytest.mark.parametrize("ref_ant", [0, 3, 6])
    def test_reference_antenna_phase_is_exactly_zero(self, ref_ant):
        """The reference gain is exactly real and positive — not merely close to it."""
        comp = ConstGains()
        comp.setup(make_const_gains_config(n_ant=7, ref_ant=ref_ant))

        out = run_forward(comp, random_params(comp, seed=3))
        g_ref = out["gains"][ref_ant]

        assert jnp.all(jnp.imag(g_ref) == 0)
        assert jnp.all(jnp.angle(g_ref) == 0)
        assert jnp.all(jnp.real(g_ref) > 0)

    def test_free_antennas_are_not_pinned(self):
        """Only the reference antenna is pinned: the rest carry a non-zero phase."""
        comp = ConstGains()
        comp.setup(make_const_gains_config(n_ant=5, ref_ant=2))

        out = run_forward(comp, random_params(comp, seed=4))
        phase = np.angle(np.asarray(out["gains"][:, 0, 0]))

        assert np.count_nonzero(phase) == 4

    def test_amplitude_spread_survives_the_gauge(self):
        """The zero-sum constraint removes the overall scale, not the relative amplitudes."""
        comp = ConstGains()
        comp.setup(make_const_gains_config(n_ant=6))

        amp = np.abs(np.asarray(run_forward(comp, random_params(comp, seed=5))["gains"][:, 0, 0]))

        assert amp.std() > 1e-3


# ---------------------------------------------------------------------------
# Shapes and the forward model
# ---------------------------------------------------------------------------


class TestForward:

    def test_output_shapes(self):
        n_ant, n_freq, n_time = 5, 3, 4
        comp = ConstGains()
        comp.setup(make_const_gains_config(n_ant=n_ant, n_freq=n_freq, n_time=n_time))

        out = run_forward(comp, random_params(comp))

        assert out["gains"].shape == (n_ant, n_freq, n_time)
        assert out["vis_obs"].shape == (comp.n_bl, n_freq, n_time)
        assert jnp.issubdtype(out["gains"].dtype, jnp.complexfloating)
        assert jnp.issubdtype(out["vis_obs"].dtype, jnp.complexfloating)

    def test_gains_are_constant_over_frequency_and_time(self):
        """Exactly constant — the whole point of the component."""
        comp = ConstGains()
        comp.setup(make_const_gains_config(n_ant=5, n_freq=3, n_time=4))

        gains = run_forward(comp, random_params(comp, seed=6))["gains"]

        assert jnp.all(gains == gains[:, :1, :1])

    def test_vis_obs_is_the_gain_product(self, exact_rtol):
        """vis_obs = g_p conj(g_q) (vis_ast + vis_rfi), against an independent reference."""
        comp = ConstGains()
        cfg = make_const_gains_config(n_ant=5, n_freq=2, n_time=3, ref_ant=1)
        comp.setup(cfg)

        params = random_params(comp, seed=7)
        state = make_vis_state(comp.n_ant, comp.n_freq, comp.n_time)
        out = run_forward(comp, params, state)

        # Rebuilt from the parameters in numpy, not read back out of the component.
        log_amp = comp.log_amp_std * (
            helmert_basis(comp.n_ant) @ np.asarray(params["gains_amp_base"])
        )
        free = comp.gp_phase_mean + comp.gp_phase_std * np.asarray(params["gains_phase_base"])
        phase = np.insert(free, comp.ref_ant, 0.0)
        g = np.exp(log_amp) * np.exp(1j * phase)

        a1, a2 = np.asarray(cfg.a1), np.asarray(cfg.a2)
        vis = np.asarray(state["vis_rfi"]) + np.asarray(state["vis_ast"])
        expected = g[a1][:, None, None] * vis * g[a2][:, None, None].conj()

        assert jnp.allclose(out["vis_obs"], expected, rtol=exact_rtol, atol=exact_rtol)

    def test_forward_preserves_other_state_keys(self):
        comp = ConstGains()
        comp.setup(make_const_gains_config())
        state = make_vis_state(comp.n_ant, comp.n_freq, comp.n_time)
        state["some_extra_key"] = jnp.array(42.0)

        assert "some_extra_key" in run_forward(comp, random_params(comp), state)

    def test_state_outputs_shapes(self):
        comp = ConstGains()
        cfg = make_const_gains_config(n_ant=4, n_freq=2, n_time=5)
        comp.setup(cfg)

        assert comp.state_outputs["gains"].shape == (4, 2, 5)
        assert comp.state_outputs["vis_obs"].shape == (cfg.n_bl, 2, 5)

    def test_set_params_shapes_in_a_numpyro_trace(self):
        comp = ConstGains()
        comp.setup(make_const_gains_config(n_ant=6))

        with numpyro.handlers.seed(rng_seed=0):
            params = comp.build_set_params()({})

        assert params["gains_amp_base"].shape == (5,)
        assert params["gains_phase_base"].shape == (5,)


class TestLatentParameterisation:
    """The latent space carries the degrees of freedom the data can see, and no more.

    A latent coordinate the forward model does not depend on is flat in the
    likelihood however much data there is, so it is only ever curved by the prior:
    it wrecks the conditioning of the optimisation and turns any likelihood-only
    Fisher calculation singular. Both removed directions are therefore removed from
    the *parameters*, not just from the value they map to.
    """

    def test_parameter_count_is_two_n_ant_minus_two(self):
        """n_ant - 1 amplitudes and n_ant - 1 phases: the gauge costs one of each."""
        comp = ConstGains()
        comp.setup(make_const_gains_config(n_ant=8))

        n_params = sum(int(np.size(v)) for v in comp.init_params_base.values())

        assert comp.init_params_base["gains_amp_base"].shape == (7,)
        assert comp.init_params_base["gains_phase_base"].shape == (7,)
        assert n_params == 2 * 8 - 2

    def test_freeing_the_flux_scale_restores_the_amplitude_parameter(self):
        comp = ConstGains()
        comp.setup(
            make_const_gains_config(
                n_ant=8,
                fix_flux_scale=False,
                components=("ast_signal:FixedDiscreteSky", "gains:ConstGains"),
            )
        )

        n_params = sum(int(np.size(v)) for v in comp.init_params_base.values())

        assert comp.init_params_base["gains_amp_base"].shape == (8,)
        assert n_params == 2 * 8 - 1

    def test_the_amplitude_basis_is_orthonormal_and_zero_sum(self, exact_rtol):
        n_ant = 7
        comp = ConstGains()
        comp.setup(make_const_gains_config(n_ant=n_ant))

        H = np.asarray(comp.amp_basis)

        assert H.shape == (n_ant, n_ant - 1)
        assert np.allclose(H.T @ H, np.eye(n_ant - 1), rtol=exact_rtol, atol=1e-6)
        assert np.allclose(H.sum(axis=0), 0.0, rtol=exact_rtol, atol=1e-6)

    def test_the_implied_amplitude_prior_is_unchanged(self, exact_rtol):
        """The reparameterisation does not move the prior on log|g|.

        The n_ant-parameter construction was ``log_amp = sigma (I - J/n) z`` with
        ``z ~ N(0, I_n)``, whose covariance is ``sigma^2 (I - J/n)``. The
        (n_ant - 1)-parameter one is ``log_amp = sigma H u`` with ``u ~ N(0, I_{n-1})``
        and ``H`` orthonormal, whose covariance is ``sigma^2 H H^T`` — the same
        projector, with no flat latent direction left over.
        """
        n_ant = 7
        comp = ConstGains()
        comp.setup(make_const_gains_config(n_ant=n_ant))

        H = np.asarray(comp.amp_basis)
        projector = np.eye(n_ant) - np.ones((n_ant, n_ant)) / n_ant

        old = comp.log_amp_std**2 * (projector @ projector.T)
        new = comp.log_amp_std**2 * (H @ H.T)

        assert np.allclose(new, old, rtol=exact_rtol, atol=1e-8)

    @pytest.mark.parametrize(
        "fix_flux_scale,components",
        [
            (True, ("gains:ConstGains",)),
            (False, ("ast_signal:FixedDiscreteSky", "gains:ConstGains")),
        ],
    )
    def test_no_latent_amplitude_direction_is_flat(self, fix_flux_scale, components):
        """d log|g| / d(amplitude parameters) has full column rank.

        A rank deficit is exactly a direction of the latent space the observed
        visibilities cannot see. With the n_ant-parameter zero-sum construction the
        Jacobian was (n_ant, n_ant) of rank n_ant - 1 — the all-ones direction.
        """
        n_ant = 6
        comp = ConstGains()
        comp.setup(
            make_const_gains_config(
                n_ant=n_ant, fix_flux_scale=fix_flux_scale, components=components
            )
        )

        params = random_params(comp, seed=11)
        state = make_vis_state(n_ant, comp.n_freq, comp.n_time)
        constants = make_constants(comp)
        forward = comp.build_forward()

        def log_amp(amp_base):
            gains = forward({**params, "gains_amp_base": amp_base}, state, constants)
            return jnp.log(jnp.abs(gains["gains"][:, 0, 0]))

        jac = np.asarray(jax.jacfwd(log_amp)(params["gains_amp_base"]))

        assert jac.shape == (n_ant, params["gains_amp_base"].shape[0])
        assert np.linalg.matrix_rank(jac) == jac.shape[1]

    def test_no_latent_phase_direction_is_flat(self):
        """The same statement for the phase block, which pins by omission already."""
        n_ant = 6
        comp = ConstGains()
        comp.setup(make_const_gains_config(n_ant=n_ant, ref_ant=2))

        params = random_params(comp, seed=12)
        state = make_vis_state(n_ant, comp.n_freq, comp.n_time)
        constants = make_constants(comp)
        forward = comp.build_forward()

        def phase(phase_base):
            gains = forward({**params, "gains_phase_base": phase_base}, state, constants)
            return jnp.angle(gains["gains"][:, 0, 0])

        jac = np.asarray(jax.jacfwd(phase)(params["gains_phase_base"]))

        assert jac.shape == (n_ant, n_ant - 1)
        assert np.linalg.matrix_rank(jac) == n_ant - 1


# ---------------------------------------------------------------------------
# The reference antenna
# ---------------------------------------------------------------------------


class TestRefAnt:

    def test_default_is_the_first_antenna(self):
        comp = ConstGains()
        comp.setup(make_const_gains_config(n_ant=5))

        assert comp.ref_ant == 0

    def test_default_skips_a_fully_flagged_antenna(self):
        """A dead antenna cannot carry the phase reference, so the default steps past it."""
        comp = ConstGains()
        comp.setup(make_const_gains_config(n_ant=5, flagged_ants=(0, 1)))

        assert comp.ref_ant == 2

    def test_explicit_reference_is_honoured(self):
        comp = ConstGains()
        comp.setup(make_const_gains_config(n_ant=5, ref_ant=3))

        assert comp.ref_ant == 3
        out = run_forward(comp, random_params(comp, seed=8))
        assert jnp.all(jnp.angle(out["gains"][3]) == 0)

    def test_fully_flagged_explicit_reference_is_an_error(self):
        with pytest.raises(RuntimeError, match="ref_ant"):
            ConstGains().setup(
                make_const_gains_config(n_ant=5, ref_ant=1, flagged_ants=(1,))
            )

    def test_out_of_range_reference_is_an_error(self):
        with pytest.raises(RuntimeError, match="ref_ant"):
            ConstGains().setup(make_const_gains_config(n_ant=5, ref_ant=5))

    def test_non_integer_reference_is_an_error(self):
        with pytest.raises(RuntimeError, match="ref_ant"):
            ConstGains().setup(make_const_gains_config(n_ant=5, ref_ant="ant3"))

    def test_all_antennas_flagged_is_an_error(self):
        with pytest.raises(RuntimeError, match="fully flagged"):
            ConstGains().setup(
                make_const_gains_config(n_ant=4, flagged_ants=(0, 1, 2, 3))
            )

    def test_an_observation_with_no_baselines_is_an_error(self):
        """A single antenna correlates with nothing, so no gain is measured at all."""
        with pytest.raises(RuntimeError, match="no baselines"):
            ConstGains().setup(
                make_const_gains_config(n_ant=1, flags=np.zeros((0, 3, 4), dtype=bool))
            )


# ---------------------------------------------------------------------------
# Connectivity of the unflagged baselines
# ---------------------------------------------------------------------------


def keep_only_baselines(n_ant, n_freq, n_time, keep):
    """Flags leaving only the (p, q) baselines in *keep* unflagged."""
    a1, a2 = np.triu_indices(n_ant, 1)
    flags = np.ones((len(a1), n_freq, n_time), dtype=bool)
    for p, q in keep:
        flags[(a1 == p) & (a2 == q)] = False
    return flags


class TestConnectivity:
    """One phase reference only pins one connected group of antennas.

    A gain phase is measured relative to another antenna's, through a baseline that
    carries data. Two groups joined by no chain of unflagged baselines have no common
    reference at all, so pinning one leaves the other's overall phase flat — which
    participation alone (every antenna having *some* data) does not catch.
    """

    def test_two_disconnected_groups_are_an_error(self):
        flags = keep_only_baselines(4, 2, 2, [(0, 1), (2, 3)])

        with pytest.raises(RuntimeError, match="no baseline"):
            ConstGains().setup(make_const_gains_config(n_ant=4, n_freq=2, n_time=2, flags=flags))

    def test_the_error_names_the_groups(self):
        flags = keep_only_baselines(5, 2, 2, [(0, 1), (2, 3), (2, 4), (3, 4)])

        with pytest.raises(RuntimeError) as excinfo:
            ConstGains().setup(make_const_gains_config(n_ant=5, n_freq=2, n_time=2, flags=flags))

        message = str(excinfo.value)
        assert "[0, 1]" in message
        assert "[2, 3, 4]" in message

    def test_a_heavily_flagged_but_connected_array_is_fine(self):
        """A bare spanning chain is still one group, and is accepted."""
        flags = keep_only_baselines(5, 2, 2, [(0, 1), (1, 2), (2, 3), (3, 4)])

        comp = ConstGains()
        comp.setup(make_const_gains_config(n_ant=5, n_freq=2, n_time=2, flags=flags))

        assert comp.ref_ant == 0

    def test_an_isolated_flagged_antenna_does_not_count_as_a_group(self):
        """Antenna 0 has no data at all — it is skipped, not called a second group."""
        comp = ConstGains()
        comp.setup(make_const_gains_config(n_ant=5, flagged_ants=(0,)))

        assert comp.ref_ant == 1


# ---------------------------------------------------------------------------
# The liftable amplitude constraint
# ---------------------------------------------------------------------------


class TestFixFluxScale:

    def test_default_is_constrained(self):
        comp = ConstGains()
        comp.setup(make_const_gains_config())

        assert comp.fix_flux_scale is True

    def test_free_flux_scale_without_a_fixed_sky_is_an_error(self):
        with pytest.raises(RuntimeError, match="fix_flux_scale"):
            ConstGains().setup(make_const_gains_config(fix_flux_scale=False))

    def test_the_error_explains_the_degeneracy(self):
        with pytest.raises(RuntimeError, match="FixedDiscreteSky"):
            ConstGains().setup(make_const_gains_config(fix_flux_scale=False))

    def test_free_flux_scale_with_a_fixed_sky_is_allowed(self):
        comp = ConstGains()
        comp.setup(
            make_const_gains_config(
                fix_flux_scale=False,
                components=(
                    "ast_signal:FixedDiscreteSky",
                    "ast_vis:DiscreteSkyVis",
                    "gains:ConstGains",
                ),
            )
        )

        assert comp.fix_flux_scale is False

    def test_free_flux_scale_leaves_the_mean_log_amplitude_free(self):
        """Without the constraint the overall scale is a fitted quantity again."""
        comp = ConstGains()
        comp.setup(
            make_const_gains_config(
                n_ant=7,
                fix_flux_scale=False,
                components=("ast_signal:FixedDiscreteSky", "gains:ConstGains"),
            )
        )

        params = random_params(comp, seed=9)
        out = run_forward(comp, params)
        mean_log_amp = float(jnp.mean(jnp.log(jnp.abs(out["gains"][:, 0, 0]))))

        # The same draw is zero-sum under the default gauge, so a non-zero mean here
        # is the constraint being absent rather than a property of the draw.
        assert abs(mean_log_amp) > 1e-3

    def test_a_non_boolean_flag_is_an_error(self):
        with pytest.raises(RuntimeError, match="fix_flux_scale"):
            ConstGains().setup(make_const_gains_config(fix_flux_scale="yes"))

    def test_null_means_the_default(self):
        comp = ConstGains()
        comp.setup(make_const_gains_config(fix_flux_scale=None))

        assert comp.fix_flux_scale is True


# ---------------------------------------------------------------------------
# What amp_mean means in each mode
# ---------------------------------------------------------------------------


class TestAmpMean:

    def test_the_fixed_gauge_starts_at_unit_gain_whatever_amp_mean_says(self, exact_rtol):
        """Under the zero-sum gauge the geometric mean of |g| is 1 by construction."""
        comp = ConstGains()
        with pytest.warns(UserWarning, match="amp_mean"):
            comp.setup(make_const_gains_config(n_ant=5, amp_mean=2.0))

        amp = np.abs(np.asarray(run_forward(comp, comp.init_params_base)["gains"][:, 0, 0]))

        assert np.allclose(amp, 1.0, rtol=exact_rtol, atol=exact_rtol)

    def test_a_non_unit_amp_mean_warns_under_the_fixed_gauge(self):
        with pytest.warns(UserWarning, match="fix_flux_scale"):
            ConstGains().setup(make_const_gains_config(amp_mean=1.4))

    def test_a_unit_amp_mean_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            ConstGains().setup(make_const_gains_config(amp_mean=1.0))

    def test_the_free_scale_starts_at_amp_mean(self, exact_rtol):
        """With the scale free, amp_mean is the prior mean it actually starts at."""
        comp = ConstGains()
        comp.setup(
            make_const_gains_config(
                n_ant=5,
                amp_mean=2.0,
                fix_flux_scale=False,
                components=("ast_signal:FixedDiscreteSky", "gains:ConstGains"),
            )
        )

        amp = np.abs(np.asarray(run_forward(comp, comp.init_params_base)["gains"][:, 0, 0]))

        assert np.allclose(amp, 2.0, rtol=exact_rtol, atol=exact_rtol)

    def test_the_free_scale_prior_is_lognormal_about_amp_mean(self, exact_rtol):
        """A one-sigma draw sits at amp_mean * exp(amp_std), the log-space spread."""
        comp = ConstGains()
        comp.setup(
            make_const_gains_config(
                n_ant=5,
                amp_mean=2.0,
                amp_std=10.0,  # percent
                fix_flux_scale=False,
                components=("ast_signal:FixedDiscreteSky", "gains:ConstGains"),
            )
        )

        params = {
            "gains_amp_base": jnp.ones(5),
            "gains_phase_base": jnp.zeros(4),
        }
        amp = np.abs(np.asarray(run_forward(comp, params)["gains"][:, 0, 0]))

        assert np.allclose(amp, 2.0 * np.exp(0.1), rtol=exact_rtol, atol=exact_rtol)

    @pytest.mark.parametrize(
        "amp_mean", [-1.0, 0.0, 0, float("nan"), float("inf"), -float("inf")]
    )
    def test_an_unusable_amp_mean_is_an_error(self, amp_mean):
        """Zero included: it used to be read as "unset" and silently become 1.0.

        Every use of amp_mean divides by it or takes its logarithm, so a value that
        is not positive and finite is a mistake worth naming rather than a default
        worth guessing.
        """
        with pytest.raises(RuntimeError, match="amp_mean"):
            ConstGains().setup(make_const_gains_config(amp_mean=amp_mean))

    def test_an_unset_amp_mean_still_defaults(self):
        comp = ConstGains()
        comp.setup(make_const_gains_config(amp_mean=None))

        assert comp.gp_amp_mean == 1.0


# ---------------------------------------------------------------------------
# The rest of the scale keys, read the same way
# ---------------------------------------------------------------------------


class TestScaleKeysReachTheComponent:
    """What :func:`validate_gain_scales` decides, ``ConstGains`` is the one that uses.

    The unit tests of the reads themselves live in ``test_gains.py``; these check the
    decision survives the trip through setup, whose ``except`` wraps a ``ValueError``
    into a ``RuntimeError``.
    """

    def test_a_zero_seed_is_the_seed_it_says(self):
        comp = ConstGains()
        cfg = make_const_gains_config()
        cfg.args["gains"]["r_seed"] = 0
        comp.setup(cfg)

        assert comp.r_seed == 0

    @pytest.mark.parametrize("width", [0, 0.0, -1.0, float("nan"), float("inf")])
    @pytest.mark.parametrize("key", ["amp_std", "phase_std"])
    def test_an_unusable_prior_width_is_an_error(self, key, width):
        """A zero width used to be read as "unset" and silently become the default.

        ``phase_std`` divides the initial phase in :meth:`_compute_init_params` and
        both widths scale a standard normal, so a degenerate width is a mistake worth
        naming rather than a default worth guessing.
        """
        cfg = make_const_gains_config()
        cfg.args["gains"][key] = width

        with pytest.raises(RuntimeError, match=key):
            ConstGains().setup(cfg)

    @pytest.mark.parametrize("key", ["amp_std", "phase_std"])
    def test_an_unset_prior_width_still_defaults(self, key):
        cfg = make_const_gains_config()
        cfg.args["gains"][key] = None
        comp = ConstGains()
        comp.setup(cfg)

        expected = {"amp_std": 0.2, "phase_std": float(jnp.deg2rad(180))}[key]

        assert getattr(comp, {"amp_std": "gp_amp_std", "phase_std": "gp_phase_std"}[key]) == pytest.approx(expected)

    def test_the_default_phase_width_puts_a_whole_turn_inside_one_sigma(self, tmp_path):
        """180 degrees is half a turn, so any measured phase starts within |z| <= 1.

        ``phase_std`` divides the measured phase in :meth:`_compute_init_params`, so
        the width decides where a gain read in from outside starts the fit: under the
        1-degree default an ordinary phase started it ~100 sigma from the prior mean,
        in a coordinate the optimiser then had to walk back one ``opt.epsilon`` step
        at a time. Driven through the real measured-gain branch rather than through a
        re-implementation of its arithmetic, so it is the shipped path that is pinned.
        """
        n_ant = 6
        rng = np.random.default_rng(0)
        # Phases spread over the whole circle, including near +/-pi.
        g = np.exp(1j * np.linspace(-np.pi + 1e-3, np.pi, n_ant))
        path = tmp_path / "gain.npz"
        np.savez(path, gain=g)

        comp = ConstGains()
        comp.setup(
            make_const_gains_config(n_ant=n_ant, phase_std=None, init=str(path))
        )

        base = np.asarray(comp.init_params_base["gains_phase_base"])

        assert np.all(np.abs(base) <= 1.0)
        # And the width really is what bounds it: the old default puts the same gain
        # far outside the prior.
        narrow = ConstGains()
        narrow.setup(
            make_const_gains_config(n_ant=n_ant, phase_std=1.0, init=str(path))
        )

        assert np.max(np.abs(np.asarray(narrow.init_params_base["gains_phase_base"]))) > 100


class TestTheDefaultPhasePriorCoversTheCircle:
    """Drawn through the component, the unset ``phase_std`` says nothing about the phase.

    The width itself is checked analytically in ``test_gains.py`` (the wrapped normal
    at half a turn is flat to 1.4 %). What is checked here is that the component's own
    prior path delivers that: the parameterisation is non-centred
    (``phase = mean + sigma z``, ``z ~ N(0, 1)``) and the phase is then wrapped by
    ``exp(1j phase)``, so a standard normal draw has to come out spread over the whole
    circle rather than concentrated anywhere on it.

    One time--frequency sample per draw: the gain is constant over both, so the extra
    grid would only duplicate each phase.
    """

    #: Coarse enough that a genuinely flat prior passes on sampling noise alone, and
    #: far too tight for anything concentrated: a 1-degree prior puts every draw in
    #: one bin.
    N_BINS = 12
    MAX_BIN_RATIO = 1.5

    N_ANT = 33
    N_DRAWS = 512

    def draw_phases(self, phase_std):
        """Gain phases from ``N_DRAWS`` prior draws, through the component's forward.

        The reference antenna's phase is pinned to zero (the overall phase is not
        observable), so it is dropped: a gauge choice rather than a draw.
        """

        comp = ConstGains()
        comp.setup(
            make_const_gains_config(
                n_ant=self.N_ANT, n_freq=1, n_time=1, phase_std=phase_std, ref_ant=0
            )
        )

        forward = comp.build_forward()
        state = make_vis_state(self.N_ANT, 1, 1)
        constants = make_constants(comp)
        keys = jax.random.split(jax.random.PRNGKey(0), 2 * self.N_DRAWS)

        def one_draw(amp_key, phase_key):
            params = {
                "gains_amp_base": jax.random.normal(
                    amp_key, comp.init_params_base["gains_amp_base"].shape
                ),
                "gains_phase_base": jax.random.normal(
                    phase_key, comp.init_params_base["gains_phase_base"].shape
                ),
            }
            gains = forward(params, state, constants)["gains"][:, 0, 0]
            return jnp.angle(gains[1:])  # ref_ant is antenna 0

        phases = jax.vmap(one_draw)(keys[: self.N_DRAWS], keys[self.N_DRAWS :])

        return np.asarray(phases).ravel()

    def test_the_default_prior_covers_the_circle(self):
        phases = self.draw_phases(phase_std=None)
        counts, _ = np.histogram(phases, bins=self.N_BINS, range=(-np.pi, np.pi))

        assert counts.min() > 0
        assert counts.max() / counts.min() < self.MAX_BIN_RATIO
        # Mean resultant length: exp(-sigma^2/2) for a wrapped normal, so ~7e-3 at
        # 180 degrees against ~1 for any prior worth calling concentrated.
        assert abs(np.mean(np.exp(1j * phases))) < 0.1

    def test_a_one_degree_prior_does_not(self):
        """The teeth of the test above: the default it replaces fails both bounds."""

        phases = self.draw_phases(phase_std=1.0)
        counts, _ = np.histogram(phases, bins=self.N_BINS, range=(-np.pi, np.pi))

        assert counts.min() == 0
        assert abs(np.mean(np.exp(1j * phases))) > 0.9


# ---------------------------------------------------------------------------
# The RFI degeneracy warning
# ---------------------------------------------------------------------------


class TestDegeneracyWarning:

    def test_warns_with_a_per_antenna_rfi_amplitude(self):
        with pytest.warns(UserWarning, match="ComplexRFIVarAnt"):
            ConstGains().setup(
                make_const_gains_config(
                    components=("rfi_signal:ComplexRFIVarAnt", "gains:ConstGains")
                )
            )

    def test_the_warning_names_the_flat_direction_and_the_issue(self):
        with pytest.warns(UserWarning) as record:
            ConstGains().setup(
                make_const_gains_config(
                    components=("rfi_signal:ComplexRFIVarAnt", "gains:ConstGains")
                )
            )

        message = str(record[0].message)
        assert "ComplexRFIConstAnt" in message
        assert "#124" in message

    def test_does_not_warn_with_a_constant_antenna_rfi_amplitude(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            ConstGains().setup(
                make_const_gains_config(
                    components=("rfi_signal:ComplexRFIConstAnt", "gains:ConstGains")
                )
            )

    def test_the_warning_is_not_an_error(self):
        """Degenerate is not forbidden — the fit still runs."""
        comp = ConstGains()
        with pytest.warns(UserWarning):
            comp.setup(
                make_const_gains_config(
                    components=("rfi_signal:ComplexRFIVarAnt", "gains:ConstGains")
                )
            )

        assert run_forward(comp, random_params(comp))["gains"].shape[0] == comp.n_ant


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:

    def test_prior_init_is_unit_gain(self, exact_rtol):
        comp = ConstGains()
        comp.setup(make_const_gains_config(n_ant=5, init="prior"))

        out = run_forward(comp, comp.init_params_base)

        assert jnp.allclose(out["gains"], 1.0 + 0.0j, rtol=exact_rtol, atol=exact_rtol)

    def test_prior_init_sits_at_the_phase_prior_mean(self, exact_rtol):
        """The prior mean of the phase is phase_mean, and the reference is still pinned."""
        comp = ConstGains()
        comp.setup(make_const_gains_config(n_ant=5, phase_mean=0.25, ref_ant=2))

        phase = np.angle(np.asarray(run_forward(comp, comp.init_params_base)["gains"][:, 0, 0]))

        assert phase[2] == 0.0
        assert np.allclose(np.delete(phase, 2), 0.25, rtol=exact_rtol, atol=exact_rtol)

    def test_npz_init_lands_in_the_gauge(self, tmp_path, exact_rtol):
        n_ant = 6
        rng = np.random.default_rng(0)
        g = rng.uniform(0.4, 2.5, n_ant) * np.exp(1j * rng.uniform(-2.0, 2.0, n_ant))
        path = tmp_path / "gain.npz"
        np.savez(path, gain=g)

        comp = ConstGains()
        comp.setup(make_const_gains_config(n_ant=n_ant, init=str(path), ref_ant=2))

        fitted = np.asarray(run_forward(comp, comp.init_params_base)["gains"][:, 0, 0])

        assert np.allclose(fitted, gauge_project(g, 2), rtol=exact_rtol, atol=exact_rtol)

    def test_projecting_an_in_gauge_init_is_a_no_op(self, exact_rtol, tmp_path):
        """A gain already in the gauge comes back unchanged, so the projection is idempotent."""
        n_ant = 6
        rng = np.random.default_rng(1)
        g = gauge_project(
            rng.uniform(0.4, 2.5, n_ant) * np.exp(1j * rng.uniform(-2.0, 2.0, n_ant)), 0
        )
        path = tmp_path / "gain.npz"
        np.savez(path, gain=g)

        comp = ConstGains()
        comp.setup(make_const_gains_config(n_ant=n_ant, init=str(path)))

        fitted = np.asarray(run_forward(comp, comp.init_params_base)["gains"][:, 0, 0])

        assert np.allclose(fitted, g, rtol=exact_rtol, atol=exact_rtol)

    def test_npz_init_keeps_the_flux_scale_when_it_is_free(self, tmp_path, exact_rtol):
        """With fix_flux_scale off the measured overall amplitude is not thrown away."""
        n_ant = 5
        rng = np.random.default_rng(2)
        g = 3.0 * rng.uniform(0.8, 1.2, n_ant) * np.exp(1j * rng.uniform(-1.0, 1.0, n_ant))
        path = tmp_path / "gain.npz"
        np.savez(path, gain=g)

        comp = ConstGains()
        comp.setup(
            make_const_gains_config(
                n_ant=n_ant,
                init=str(path),
                fix_flux_scale=False,
                components=("ast_signal:FixedDiscreteSky", "gains:ConstGains"),
            )
        )

        fitted = np.asarray(run_forward(comp, comp.init_params_base)["gains"][:, 0, 0])

        assert np.allclose(
            fitted, gauge_project(g, 0, fix_flux_scale=False), rtol=exact_rtol, atol=exact_rtol
        )

    def test_npz_with_the_wrong_shape_is_an_error(self, tmp_path):
        path = tmp_path / "gain.npz"
        np.savez(path, gain=np.ones(3, dtype=complex))

        with pytest.raises(RuntimeError, match=r"\(3,\)"):
            ConstGains().setup(make_const_gains_config(n_ant=5, init=str(path)))

    def test_npz_without_a_gain_key_is_an_error(self, tmp_path):
        path = tmp_path / "gain.npz"
        np.savez(path, gains=np.ones(5, dtype=complex))

        with pytest.raises(RuntimeError, match="gain"):
            ConstGains().setup(make_const_gains_config(n_ant=5, init=str(path)))

    def test_a_zero_gain_is_an_error(self, tmp_path):
        """log|g| of a zero gain is -inf; a dead antenna is not an initialisation."""
        path = tmp_path / "gain.npz"
        g = np.ones(5, dtype=complex)
        g[2] = 0.0
        np.savez(path, gain=g)

        with pytest.raises(RuntimeError, match="zero"):
            ConstGains().setup(make_const_gains_config(n_ant=5, init=str(path)))

    def test_an_unusable_init_string_is_an_error(self):
        with pytest.raises(RuntimeError, match="init"):
            ConstGains().setup(make_const_gains_config(init="truth"))


# ---------------------------------------------------------------------------
# Initialisation from a calibration table
# ---------------------------------------------------------------------------


@requires_casacore
class TestCaltableInit:

    def test_a_constant_caltable_lands_in_the_gauge(self, tmp_path, exact_rtol):
        n_ant, n_freq, n_time = 6, 4, 3
        rng = np.random.default_rng(3)
        g = rng.uniform(0.4, 2.5, n_ant) * np.exp(1j * rng.uniform(-2.0, 2.0, n_ant))
        # complex64 on disk, so the round trip is a float32 comparison whatever
        # precision the session runs in.
        g = g.astype(np.complex64).astype(complex)
        path = write_test_caltable(
            str(tmp_path / "cal.B"), np.broadcast_to(g[:, None, None], (n_ant, n_freq, n_time))
        )

        comp = ConstGains()
        comp.setup(make_const_gains_config(n_ant=n_ant, init=path, ref_ant=1))

        fitted = np.asarray(run_forward(comp, comp.init_params_base)["gains"][:, 0, 0])

        assert np.allclose(fitted, gauge_project(g, 1), rtol=max(exact_rtol, 1e-6), atol=1e-6)

    def test_a_varying_caltable_warns_and_reduces(self, tmp_path):
        n_ant, n_freq, n_time = 5, 3, 4
        rng = np.random.default_rng(4)
        g = rng.uniform(0.5, 2.0, n_ant) * np.exp(1j * rng.uniform(-1.0, 1.0, n_ant))
        varying = g[:, None, None] * (1 + 0.2 * rng.standard_normal((n_ant, n_freq, n_time)))
        path = write_test_caltable(str(tmp_path / "cal.B"), varying)

        comp = ConstGains()
        with pytest.warns(UserWarning, match="varies"):
            comp.setup(make_const_gains_config(n_ant=n_ant, init=path))

        fitted = np.asarray(run_forward(comp, comp.init_params_base)["gains"][:, 0, 0])
        expected = gauge_project(
            np.median(np.abs(varying), axis=(1, 2))
            * np.exp(1j * np.angle(np.mean(varying / np.abs(varying), axis=(1, 2)))),
            0,
        )

        assert np.allclose(fitted, expected, rtol=1e-5, atol=1e-6)

    def test_a_constant_caltable_does_not_warn(self, tmp_path):
        n_ant, n_freq, n_time = 4, 2, 2
        g = np.array([1.0, 1.5, 0.7, 2.0], dtype=complex) * np.exp(
            1j * np.array([0.0, 0.3, -0.4, 1.0])
        )
        path = write_test_caltable(
            str(tmp_path / "cal.B"), np.broadcast_to(g[:, None, None], (n_ant, n_freq, n_time))
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            ConstGains().setup(make_const_gains_config(n_ant=n_ant, init=path))

    def test_flagged_samples_are_dropped_from_the_reduction(self, tmp_path):
        """A flagged solution is missing, not zero — the surviving samples set the gain."""
        n_ant, n_freq, n_time = 4, 3, 2
        g = np.array([1.0, 1.5, 0.7, 2.0], dtype=complex)
        table_gains = np.broadcast_to(g[:, None, None], (n_ant, n_freq, n_time)).copy()
        table_gains[1, 0, :] = 0.0  # written flagged by write_test_caltable
        path = write_test_caltable(str(tmp_path / "cal.B"), table_gains)

        comp = ConstGains()
        comp.setup(make_const_gains_config(n_ant=n_ant, init=path))

        fitted = np.asarray(run_forward(comp, comp.init_params_base)["gains"][:, 0, 0])

        assert np.allclose(fitted, gauge_project(g, 0), rtol=1e-5, atol=1e-6)

    def test_a_fully_flagged_antenna_falls_back_to_unit_gain(self, tmp_path):
        n_ant, n_freq, n_time = 4, 2, 2
        g = np.array([1.0, 1.5, 0.7, 2.0], dtype=complex)
        table_gains = np.broadcast_to(g[:, None, None], (n_ant, n_freq, n_time)).copy()
        table_gains[2] = 0.0
        path = write_test_caltable(str(tmp_path / "cal.B"), table_gains)

        comp = ConstGains()
        with pytest.warns(UserWarning, match="no solution"):
            comp.setup(make_const_gains_config(n_ant=n_ant, init=path))

        expected = g.copy()
        expected[2] = 1.0
        fitted = np.asarray(run_forward(comp, comp.init_params_base)["gains"][:, 0, 0])

        assert np.allclose(fitted, gauge_project(expected, 0), rtol=1e-5, atol=1e-6)

    def test_an_antenna_count_mismatch_is_an_error(self, tmp_path):
        path = write_test_caltable(str(tmp_path / "cal.B"), np.ones((3, 2, 2), dtype=complex))

        with pytest.raises(RuntimeError, match="3"):
            ConstGains().setup(make_const_gains_config(n_ant=5, init=path))


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


def test_component_is_importable_by_config_string():
    from tabascal.imports import import_components

    assert import_components(["gains:ConstGains"]) == [ConstGains]


def test_base_config_carries_the_new_keys():
    from importlib.resources import files

    from tabascal.config import yaml_load

    base = yaml_load(
        os.path.join(
            str(files("tabascal").joinpath("data/config")), "tab_config_base.yaml"
        )
    )

    assert base["gains"]["ref_ant"] is None
    assert base["gains"]["fix_flux_scale"] is True


def test_the_correlation_lengths_are_not_read_here():
    """The gain correlation lengths went with ``gains:GPGains`` (#129).

    A config carrying them is stopped at load by
    :func:`~tabascal.config.check_removed_keys`; what is pinned here is the other
    half of that, namely that nothing on the ``ConstGains`` path would look for one.
    """
    cfg = make_const_gains_config()

    assert not [key for key in cfg.args["gains"] if key.endswith(("_corr_freq", "_corr_time"))]

    comp = ConstGains()
    comp.setup(cfg)  # no correlation length in the config, and none missed


def test_validate_gain_scales_converts_percent_and_degrees():
    """The scales reach the component in the units the model works in."""
    cfg = validate_gain_scales(
        {
            "r_seed": None,
            "amp_mean": 2.0,
            "amp_std": 5.0,
            "phase_mean": None,
            "phase_std": 2.0,
        }
    )

    assert cfg["r_seed"] == 2
    assert cfg["amp_std"] == pytest.approx(5.0 / 100 * 2.0)
    assert cfg["phase_mean"] == pytest.approx(0.0)
    assert cfg["phase_std"] == pytest.approx(float(np.deg2rad(2.0)))


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_map_fit_recovers_known_constant_gains(precision):
    """A gain fitted against a *rigid* sky is recovered, up to the gauge.

    ``vis_ast`` is held fixed here, which is the identifiability condition of issue
    #124 in its simplest form: a sky the gain cannot deform. The truth is compared
    after gauge projection, since the overall amplitude and phase are not observable
    and the model does not carry them.
    """
    n_ant, n_freq, n_time = 6, 2, 3
    rng = np.random.default_rng(0)

    comp = ConstGains()
    comp.setup(make_const_gains_config(n_ant=n_ant, n_freq=n_freq, n_time=n_time, ref_ant=0))

    g_true = gauge_project(
        np.exp(rng.normal(0, 0.15, n_ant)) * np.exp(1j * rng.uniform(-0.6, 0.6, n_ant)), 0
    )

    a1, a2 = np.triu_indices(n_ant, 1)
    vis_ast = rng.normal(0, 1, (len(a1), n_freq, n_time)) + 1j * rng.normal(
        0, 1, (len(a1), n_freq, n_time)
    )
    data = jnp.asarray(
        g_true[a1][:, None, None] * vis_ast * g_true[a2][:, None, None].conj()
    )

    state = {
        "vis_ast": jnp.asarray(vis_ast),
        "vis_rfi": jnp.zeros_like(jnp.asarray(vis_ast)),
    }
    forward = comp.build_forward()
    constants = make_constants(comp)

    def loss(params):
        residual = forward(params, state, constants)["vis_obs"] - data
        return jnp.mean(jnp.abs(residual) ** 2)

    # Started away from the truth so convergence is doing the work, not the init.
    params = random_params(comp, seed=0, scale=0.5)

    optimiser = optax.adam(1e-2)
    opt_state = optimiser.init(params)

    @jax.jit
    def step(carry, _):
        params, opt_state = carry
        value, grads = jax.value_and_grad(loss)(params)
        updates, opt_state = optimiser.update(grads, opt_state)
        return (optax.apply_updates(params, updates), opt_state), value

    (params, _), losses = jax.lax.scan(step, (params, opt_state), None, length=4000)

    g_fit = np.asarray(forward(params, state, constants)["gains"][:, 0, 0])

    atol = 1e-5 if precision == "double" else 1e-3
    assert float(losses[-1]) < atol
    assert np.allclose(g_fit, g_true, rtol=0, atol=atol), (
        f"max |g_fit - g_true| = {np.abs(g_fit - g_true).max():.2e}"
    )
