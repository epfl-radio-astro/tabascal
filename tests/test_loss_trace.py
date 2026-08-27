"""The optional optimiser trace: activation, schema, and what it costs when off.

``run_custom_svi`` can dump a per-iteration record of the loss, the wall-clock
time it was reached, and the metrics that are fixed by the data rather than by
the parameterisation. Three things are worth locking in:

- **Off is off.** With no trace path the optimiser must take the original
  ``_map_step`` path and produce bit-identical parameters, so a diagnostic can
  never be blamed for a production result.
- **The numbers are the same numbers.** The traced ``chi2`` must equal
  :func:`~tabascal.tab_tools.reduced_chi2` and the traced NRMSEs must equal
  :func:`~tabascal.tab_tools.rmse` over the representative noise -- including
  with a resolved, per-(baseline, channel) noise, which a scalar division
  broadcasts onto the wrong axis of ``(n_bl, n_freq, n_time)``.
- **Nothing large is baked in.** ``run_custom_svi`` exists to keep the big
  arrays out of the compiled program as constants; the traced path must not
  put them back.
"""

import os
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
import pytest
from jax import random
from numpyro.infer.util import log_density

from tabascal import tab_tools
from tabascal.components.likelihood import gaussian
from tabascal.config import yaml_load
from tabascal.noise import broadcast_to_vis, representative_sigma
from tabascal.tab_tools import (
    build_trace_metrics,
    loss_trace_path,
    reduced_chi2,
    rmse,
    run_custom_svi,
)

# (n_bl, n_freq, n_time). Small enough to compile in a moment, big enough that
# every axis has a different length -- a noise broadcast onto the wrong one is
# then a shape error rather than a silently plausible number.
SHAPE = (6, 4, 5)

TRACE_ENV = "TAB_LOSS_TRACE"


def _toy_problem(noise, seed: int = 0, shape=SHAPE):
    """A toy MAP problem shaped like the real one, with a truth to score against.

    ``vis_ast`` is the only thing fitted; ``vis_rfi`` is handed in as a constant
    so the model still writes both deterministic sites the trace reads. The
    likelihood is the production :func:`gaussian`, so the flag masking and noise
    broadcasting are the ones the pipeline uses.
    """

    keys = random.split(random.PRNGKey(seed), 5)

    def _complex(key):
        r, i = random.normal(key, (2,) + shape)
        return jax.lax.complex(r, i)

    vis_ast_true = _complex(keys[0])
    vis_rfi_true = _complex(keys[1])
    vis_obs = vis_ast_true + vis_rfi_true + 0.1 * _complex(keys[2])

    # A few flagged samples, so a metric that ignored the mask would disagree.
    flags = random.uniform(keys[3], shape) < 0.1

    noise_vis = broadcast_to_vis(jnp.asarray(noise), shape)
    state = {"noise": noise_vis, "flags": flags}
    constants = {"vis_rfi": vis_rfi_true}

    def prob_model(obs_data=None, state=None, constants=None):
        a_r = numpyro.sample("a_r", dist.Normal(0.0, 1.0).expand(shape).to_event(3))
        a_i = numpyro.sample("a_i", dist.Normal(0.0, 1.0).expand(shape).to_event(3))
        vis_ast = numpyro.deterministic("vis_ast", jax.lax.complex(a_r, a_i))
        vis_rfi = numpyro.deterministic("vis_rfi", constants["vis_rfi"])
        vis = numpyro.deterministic("vis_obs", vis_ast + vis_rfi)
        if obs_data is not None:
            gaussian(vis, obs_data, {"noise": state["noise"], "flags": state["flags"]})
        return {"vis_obs": vis}

    # Deliberately away from the truth: a chi^2 of exactly 1 would agree with a
    # wrong normalisation as readily as with the right one.
    init = random.normal(keys[4], (2,) + shape)
    init_params = {"a_r": init[0], "a_i": init[1]}

    tab_config = SimpleNamespace(
        vis_obs=vis_obs,
        flags=flags,
        noise=jnp.asarray(noise),
        noise_scalar=representative_sigma(np.asarray(noise)),
    )
    truth = {"vis_ast": vis_ast_true, "vis_rfi": vis_rfi_true}

    return SimpleNamespace(
        tab_config=tab_config,
        prob_model=prob_model,
        init_params=init_params,
        state=state,
        constants=constants,
        truth=truth,
        vis_obs=vis_obs,
        flags=flags,
    )


def _const_size(node) -> int:
    """Elements held as constants by a jaxpr, its nested ``jit`` jaxprs included.

    An array closed over by a jitted function does not appear on the *outer*
    jaxpr: the call lowers to a single ``jit`` equation and the constant sits on
    the jaxpr inside it. Counting only the outer consts would report zero for
    exactly the mistake this is looking for, so the walk descends.
    """

    total, stack = 0, [node]
    while stack:
        item = stack.pop()
        consts = getattr(item, "consts", None)
        if consts is not None:
            total += sum(int(np.size(c)) for c in consts)
        jaxpr = getattr(item, "jaxpr", item)
        for eqn in getattr(jaxpr, "eqns", []):
            for param in eqn.params.values():
                values = param if isinstance(param, (list, tuple)) else (param,)
                stack += [
                    v for v in values if hasattr(v, "eqns") or hasattr(v, "consts")
                ]
    return total


def _init_trace(toy):
    """The model trace at the initial parameters -- what iteration 0 scores."""

    _, model_trace = log_density(
        toy.prob_model,
        (toy.vis_obs,),
        {"state": toy.state, "constants": toy.constants},
        toy.init_params,
    )
    return model_trace


def _run(toy, tmp_path, max_iter=3, with_truth=True, trace=True, path=None):
    """Run the toy optimiser, returning the path the trace was asked for at.

    ``trace=False`` is the untraced call as ``run_opt`` makes it: no path *and*
    no metrics, since the path is what decides whether metrics are built at all.
    """

    path = path or str(tmp_path / "trace.npz")
    metrics = (
        build_trace_metrics(toy.tab_config, toy.truth if with_truth else None)
        if trace
        else None
    )
    run_custom_svi(
        prob_model=toy.prob_model,
        obs_data=toy.vis_obs,
        max_iter=max_iter,
        init_params=toy.init_params,
        epsilon=1e-2,
        dual_run=False,
        state=toy.state,
        constants=toy.constants,
        trace_path=path if trace else None,
        metrics=metrics,
    )
    return path


@pytest.fixture(autouse=True)
def _no_trace_env(monkeypatch):
    """The env override must not leak in from the developer's shell."""

    monkeypatch.delenv(TRACE_ENV, raising=False)


class TestTraceDisabled:
    """With no trace path the production path must be untouched."""

    def test_optimiser_output_is_bit_identical_to_the_plain_step(self, tmp_path):
        toy = _toy_problem(0.1)
        max_iter, epsilon = 3, 1e-2

        # The reference: _map_step driven by hand, exactly as the untraced
        # optimiser drives it.
        optimizer = optax.adabelief(epsilon)
        params = toy.init_params
        opt_state = optimizer.init(params)
        ref_losses = []
        for _ in range(max_iter):
            params, opt_state, loss = tab_tools._map_step(
                toy.prob_model, optimizer, params, opt_state,
                toy.state, toy.constants, toy.vis_obs,
            )
            ref_losses.append(float(loss))

        results = run_custom_svi(
            prob_model=toy.prob_model,
            obs_data=toy.vis_obs,
            max_iter=max_iter,
            init_params=toy.init_params,
            epsilon=epsilon,
            dual_run=False,
            state=toy.state,
            constants=toy.constants,
        )

        assert list(np.asarray(results.losses)) == ref_losses
        for key, ref in params.items():
            got = results.params[key + "_auto_loc"]
            assert np.array_equal(np.asarray(got), np.asarray(ref))

    def test_metrics_step_is_never_compiled(self, monkeypatch, tmp_path):
        toy = _toy_problem(0.1)
        calls = {"plain": 0, "metrics": 0}
        plain = tab_tools._map_step

        def counting_plain(*args, **kwargs):
            calls["plain"] += 1
            return plain(*args, **kwargs)

        def forbidden(*args, **kwargs):
            calls["metrics"] += 1
            raise AssertionError("the traced step ran with the trace disabled")

        monkeypatch.setattr(tab_tools, "_map_step", counting_plain)
        monkeypatch.setattr(tab_tools, "_map_step_metrics", forbidden)

        run_custom_svi(
            prob_model=toy.prob_model,
            obs_data=toy.vis_obs,
            max_iter=3,
            init_params=toy.init_params,
            epsilon=1e-2,
            dual_run=False,
            state=toy.state,
            constants=toy.constants,
        )

        assert calls == {"plain": 3, "metrics": 0}
        assert list(tmp_path.iterdir()) == []

    def test_no_file_is_written_without_a_path(self, tmp_path):
        toy = _toy_problem(0.1)
        _run(toy, tmp_path, trace=False)

        assert list(tmp_path.iterdir()) == []

    def test_metrics_without_a_path_are_ignored(self, monkeypatch, tmp_path):
        """The path decides, not the metrics.

        Otherwise a caller that built metrics and left the path off would
        silently compile and run the traced step -- paying for diagnostics on
        every iteration of a production run and then throwing them away.
        """
        toy = _toy_problem(0.1)

        def forbidden(*args, **kwargs):
            raise AssertionError("the traced step ran with no trace path")

        monkeypatch.setattr(tab_tools, "_map_step_metrics", forbidden)

        results = run_custom_svi(
            prob_model=toy.prob_model,
            obs_data=toy.vis_obs,
            max_iter=3,
            init_params=toy.init_params,
            epsilon=1e-2,
            dual_run=False,
            state=toy.state,
            constants=toy.constants,
            trace_path=None,
            metrics=build_trace_metrics(toy.tab_config, toy.truth),
        )

        assert len(results.losses) == 3
        assert list(tmp_path.iterdir()) == []

    def test_no_timestamps_are_taken(self, monkeypatch):
        """"Free when unset" means no clock reads and no per-iteration list."""

        toy = _toy_problem(0.1)

        def forbidden():
            raise AssertionError("the wall clock was read with the trace disabled")

        monkeypatch.setattr(tab_tools, "perf_counter", forbidden)

        run_custom_svi(
            prob_model=toy.prob_model,
            obs_data=toy.vis_obs,
            max_iter=3,
            init_params=toy.init_params,
            epsilon=1e-2,
            dual_run=False,
            state=toy.state,
            constants=toy.constants,
        )


class TestTraceContract:
    """``trace_path`` is what turns the trace on, and it promises a schema."""

    def test_a_path_without_metrics_is_refused(self, tmp_path):
        """A trace with no chi^2 is not the documented file.

        Failing at the call is the only place this can be said: by the time the
        writer runs, the whole optimisation has already happened and refusing
        then would throw the run away.
        """
        toy = _toy_problem(0.1)

        with pytest.raises(ValueError, match="metrics"):
            run_custom_svi(
                prob_model=toy.prob_model,
                obs_data=toy.vis_obs,
                max_iter=3,
                init_params=toy.init_params,
                epsilon=1e-2,
                dual_run=False,
                state=toy.state,
                constants=toy.constants,
                trace_path=str(tmp_path / "trace.npz"),
                metrics=None,
            )

        assert list(tmp_path.iterdir()) == []

    def test_the_refusal_happens_before_any_optimisation(self, monkeypatch, tmp_path):
        toy = _toy_problem(0.1)

        def forbidden(*args, **kwargs):
            raise AssertionError("the optimiser ran before the contract was checked")

        monkeypatch.setattr(tab_tools, "_map_step", forbidden)
        monkeypatch.setattr(tab_tools, "_map_step_metrics", forbidden)

        with pytest.raises(ValueError, match="metrics"):
            run_custom_svi(
                prob_model=toy.prob_model,
                obs_data=toy.vis_obs,
                max_iter=3,
                init_params=toy.init_params,
                epsilon=1e-2,
                dual_run=False,
                state=toy.state,
                constants=toy.constants,
                trace_path=str(tmp_path / "trace.npz"),
            )


class TestWriter:
    """Where the file goes, and what it does to what is already there."""

    def test_an_existing_trace_is_overwritten(self, tmp_path):
        path = tmp_path / "trace.npz"
        np.savez(path, stale=np.arange(99.0))
        toy = _toy_problem(0.1)

        _run(toy, tmp_path, path=str(path))

        with np.load(path) as npz:
            assert "stale" not in npz.files
            assert npz["chi2"].shape == (3,)

    def test_a_missing_parent_directory_is_created(self, tmp_path):
        """A trace configured into a results directory must not lose the run.

        The file is written at the very end of the optimisation, so a missing
        directory there would kill a run that has already done all of its work.
        """
        toy = _toy_problem(0.1)
        path = tmp_path / "results" / "run-1" / "trace.npz"

        _run(toy, tmp_path, path=str(path))

        with np.load(path) as npz:
            assert npz["loss"].shape == (3,)


class TestTraceEnabled:
    """What the ``.npz`` carries, and that its numbers are the reported ones."""

    def test_schema_lengths_and_monotonic_time(self, tmp_path):
        toy = _toy_problem(0.1)
        max_iter = 3

        with np.load(_run(toy, tmp_path, max_iter=max_iter)) as npz:
            assert set(npz.files) == {
                "loss", "time_s", "chi2", "vis_ast_nrmse", "vis_rfi_nrmse",
            }
            assert all(npz[key].shape == (max_iter,) for key in npz.files)
            time_s = npz["time_s"]

        assert np.all(np.diff(time_s) >= 0)
        assert time_s[0] >= 0

    def test_truth_free_run_drops_only_the_nrmse_keys(self, tmp_path):
        toy = _toy_problem(0.1)

        with np.load(_run(toy, tmp_path, with_truth=False)) as npz:
            assert set(npz.files) == {"loss", "time_s", "chi2"}

    def test_chi2_at_init_matches_reduced_chi2(self, tmp_path, exact_rtol):
        toy = _toy_problem(0.1)
        expected = reduced_chi2(
            _init_trace(toy)["vis_obs"]["value"],
            toy.tab_config.vis_obs,
            toy.tab_config.noise,
            toy.tab_config.flags,
        )

        with np.load(_run(toy, tmp_path)) as npz:
            np.testing.assert_allclose(npz["chi2"][0], float(expected), rtol=exact_rtol)

    @pytest.mark.parametrize("key", ["vis_ast", "vis_rfi"])
    def test_nrmse_at_init_matches_rmse_over_the_representative_noise(
        self, tmp_path, exact_rtol, key
    ):
        toy = _toy_problem(0.1)
        expected = float(
            rmse(_init_trace(toy)[key]["value"], toy.truth[key], toy.flags)
        ) / toy.tab_config.noise_scalar

        with np.load(_run(toy, tmp_path)) as npz:
            np.testing.assert_allclose(
                npz[f"{key}_nrmse"][0], expected, rtol=exact_rtol
            )


class TestResolvedNoise:
    """A per-(baseline, channel) noise must weight the axis it belongs to.

    The regression case from #138: ``vis_obs`` is ``(n_bl, n_freq, n_time)``, so
    dividing it by an ``(n_bl, n_freq)`` noise broadcasts the channel axis onto
    the *time* axis and the baseline axis onto the *channel* axis. It only
    raises when the lengths happen to disagree, so on a real observation it
    would come out as a plausible, wrong chi^2.
    """

    @staticmethod
    def _noise(shape):
        # Spread over a factor of ~30, as EDA2's per-baseline SIGMA is: a
        # mis-aligned broadcast then moves the chi^2 by a lot, not a little.
        rng = np.random.default_rng(7)
        return 0.05 * np.exp(rng.uniform(0.0, np.log(30.0), shape))

    def test_chi2_matches_reduced_chi2_with_per_baseline_freq_noise(
        self, tmp_path, exact_rtol
    ):
        toy = _toy_problem(self._noise(SHAPE[:2]))
        assert toy.tab_config.noise.shape == SHAPE[:2]

        expected = reduced_chi2(
            _init_trace(toy)["vis_obs"]["value"],
            toy.tab_config.vis_obs,
            toy.tab_config.noise,
            toy.tab_config.flags,
        )

        with np.load(_run(toy, tmp_path)) as npz:
            np.testing.assert_allclose(npz["chi2"][0], float(expected), rtol=exact_rtol)

    def test_chi2_matches_reduced_chi2_when_the_wrong_axis_would_also_fit(
        self, tmp_path, exact_rtol
    ):
        """The silent case, with as many baselines as timesteps.

        A per-baseline noise divided in without being broadcast first lines its
        one axis up with the *last* axis of the visibilities. Usually the lengths
        disagree and it raises; when the observation happens to have as many
        baselines as timesteps it does not, and every visibility is weighted by
        another baseline's noise with nothing said.
        """
        shape = (SHAPE[2], SHAPE[1], SHAPE[2])
        toy = _toy_problem(self._noise((shape[0],)), shape=shape)

        expected = reduced_chi2(
            _init_trace(toy)["vis_obs"]["value"],
            toy.tab_config.vis_obs,
            toy.tab_config.noise,
            toy.tab_config.flags,
        )

        with np.load(_run(toy, tmp_path)) as npz:
            np.testing.assert_allclose(npz["chi2"][0], float(expected), rtol=exact_rtol)

    def test_a_scalar_noise_would_not_have_caught_this(self, exact_rtol):
        """The resolved and collapsed chi^2 differ, so the tests above have teeth."""

        toy = _toy_problem(self._noise(SHAPE[:2]))
        pred = _init_trace(toy)["vis_obs"]["value"]
        resolved = float(
            reduced_chi2(pred, toy.vis_obs, toy.tab_config.noise, toy.flags)
        )
        collapsed = float(
            reduced_chi2(pred, toy.vis_obs, toy.tab_config.noise_scalar, toy.flags)
        )

        assert not np.isclose(resolved, collapsed, rtol=1e-2)


class TestNoConstantsBakedIn:
    """The traced step must not undo ``run_custom_svi``'s reason for existing."""

    def test_tracing_adds_no_large_constants_to_the_compiled_program(self):
        toy = _toy_problem(0.1)
        optimizer = optax.adabelief(1e-2)
        opt_state = optimizer.init(toy.init_params)
        metrics_fn, metrics_data = build_trace_metrics(toy.tab_config, toy.truth)

        plain = jax.make_jaxpr(
            lambda p, o, s, c, d: tab_tools._map_step(
                toy.prob_model, optimizer, p, o, s, c, d
            )
        )(toy.init_params, opt_state, toy.state, toy.constants, toy.vis_obs)

        traced = jax.make_jaxpr(
            lambda p, o, s, c, d, m: tab_tools._map_step_metrics(
                toy.prob_model, optimizer, metrics_fn, p, o, s, c, d, m
            )
        )(
            toy.init_params, opt_state, toy.state, toy.constants, toy.vis_obs,
            metrics_data,
        )

        # vis_obs, the noise, the mask and each truth array are all obs-sized, so
        # closure-capturing any one of them would blow this bound.
        assert _const_size(traced) - _const_size(plain) < toy.vis_obs.size

    def test_the_metrics_read_their_argument_not_a_closure(self):
        """Feeding different data through gives different numbers."""

        toy = _toy_problem(0.1)
        metrics_fn, metrics_data = build_trace_metrics(toy.tab_config, toy.truth)
        model_trace = _init_trace(toy)

        shifted = dict(metrics_data, vis_obs=metrics_data["vis_obs"] + 1.0)

        assert not np.isclose(
            float(metrics_fn(model_trace, metrics_data)["chi2"]),
            float(metrics_fn(model_trace, shifted)["chi2"]),
        )


class TestActivation:
    """``opt.trace_path`` turns it on; ``TAB_LOSS_TRACE`` overrides it."""

    def test_off_by_default(self):
        assert loss_trace_path(None) is None

    def test_config_key_turns_it_on(self):
        assert loss_trace_path("from_config.npz") == "from_config.npz"

    def test_env_var_overrides_the_config(self, monkeypatch):
        monkeypatch.setenv(TRACE_ENV, "from_env.npz")

        assert loss_trace_path("from_config.npz") == "from_env.npz"

    def test_env_var_works_without_a_config_key(self, monkeypatch):
        monkeypatch.setenv(TRACE_ENV, "from_env.npz")

        assert loss_trace_path(None) == "from_env.npz"

    def test_empty_env_var_falls_through_to_the_config(self, monkeypatch):
        monkeypatch.setenv(TRACE_ENV, "")

        assert loss_trace_path("from_config.npz") == "from_config.npz"

    def test_base_config_ships_the_key_disabled(self):
        from importlib.resources import files

        base = yaml_load(
            os.path.join(
                str(files("tabascal").joinpath("data/config")),
                "tab_config_base.yaml",
            )
        )

        assert base["opt"]["trace_path"] is None


class TestRankGating:
    """Every rank traces; only rank 0 writes.

    The model evaluation carries collectives under sharding, so every process
    has to execute the same compiled program. Gating *activation* by rank would
    put rank 0 in `_map_step_metrics` and every other rank in `_map_step`, and
    the first collective inside the traced step would hang the run -- a
    deadlock that only appears the day someone traces a distributed job. So the
    rank decides nothing about what is computed, only about what is written.
    """

    @pytest.mark.parametrize("is_rank_0", [True, False])
    @pytest.mark.parametrize("source", ["config", "env"])
    def test_every_rank_resolves_the_same_path(self, monkeypatch, is_rank_0, source):
        monkeypatch.setattr(tab_tools, "is_process_0", lambda: is_rank_0)
        if source == "env":
            monkeypatch.setenv(TRACE_ENV, "trace.npz")
            configured = None
        else:
            configured = "trace.npz"

        assert loss_trace_path(configured) == "trace.npz"

    @pytest.mark.parametrize("is_rank_0", [True, False])
    def test_every_rank_runs_the_traced_step_and_reads_the_clock(
        self, monkeypatch, tmp_path, is_rank_0
    ):
        monkeypatch.setattr(tab_tools, "is_process_0", lambda: is_rank_0)
        toy = _toy_problem(0.1)
        max_iter = 3
        calls = {"metrics": 0, "clock": 0}
        traced, clock = tab_tools._map_step_metrics, tab_tools.perf_counter

        def counting_traced(*args, **kwargs):
            calls["metrics"] += 1
            return traced(*args, **kwargs)

        def counting_clock():
            calls["clock"] += 1
            return clock()

        def forbidden(*args, **kwargs):
            raise AssertionError("a rank took the untraced step while tracing")

        monkeypatch.setattr(tab_tools, "_map_step_metrics", counting_traced)
        monkeypatch.setattr(tab_tools, "perf_counter", counting_clock)
        monkeypatch.setattr(tab_tools, "_map_step", forbidden)

        _run(toy, tmp_path, max_iter=max_iter)

        assert calls["metrics"] == max_iter
        # One read to start the clock, one at the end of each iteration.
        assert calls["clock"] == max_iter + 1

    def test_only_rank_0_writes(self, monkeypatch, tmp_path):
        monkeypatch.setattr(tab_tools, "is_process_0", lambda: False)
        toy = _toy_problem(0.1)

        _run(toy, tmp_path)

        assert list(tmp_path.iterdir()) == []

    def test_a_non_zero_rank_does_not_create_the_directory_either(
        self, monkeypatch, tmp_path
    ):
        """The makedirs has to sit inside the gate, not before it."""

        monkeypatch.setattr(tab_tools, "is_process_0", lambda: False)
        toy = _toy_problem(0.1)

        _run(toy, tmp_path, path=str(tmp_path / "results" / "trace.npz"))

        assert not (tmp_path / "results").exists()
