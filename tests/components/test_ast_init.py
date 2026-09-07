"""``ast.init`` and ``ast.mean`` -- the options GPVisAst accepts, and what they mean.

An option list is the only map a user has to these two keys, and every copy of it
was wrong in a different direction. ``_compute_init_params`` offered ``zeros`` in
its own error text without implementing it, so following the code's own
suggestion raised the same error again; the base config advertised ``est`` and
``truth_mean``, neither of which has ever been a handler, and listed neither
``data`` nor the default ``sample``.

So the tests below pin the three copies to each other: every option the error
text offers has to construct, the base config's comment has to name exactly the
options the code takes, and ``zeros`` has to mean what it says -- an initial
astronomical visibility of zero, through the forward the model actually runs.
"""

import re
from importlib.resources import files

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from tabascal.components.ast_vis import GPVisAst

from tests.test_base_config import base_args, make_ast_config
from .conftest import make_constants


#: Every value ``ast.init`` accepts. ``0`` is deliberately not among them: it is
#: an alias only of ``ast.mean``, whose default it is.
INIT_OPTIONS = ("data", "prior", "truth", "sample", "zeros")

#: Every value ``ast.mean`` accepts, ``0`` included -- it is the default, so the
#: alias is not optional.
MEAN_OPTIONS = ("data", "zeros", 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ast_config(tmp_path, *, seed=1, **overrides):
    """A mock TabConfig whose ``ast`` section is the base config plus overrides.

    ``vis_obs`` is non-trivial, unlike the zeros of :func:`make_ast_config`: with
    zero data the ``data``, ``zeros`` and (at ``mean: 0``) ``prior`` options all
    coincide, so nothing here would be able to tell them apart.
    """
    args = base_args(tmp_path)
    args["ast"].update(overrides)

    config = make_ast_config(args)

    rng = np.random.default_rng(seed)
    shape = (config.n_bl, config.n_freq, config.n_time)
    config.vis_obs = jnp.asarray(
        rng.normal(scale=10.0, size=shape) + 1j * rng.normal(scale=10.0, size=shape)
    )

    return config


def with_truth(config, tmp_path, seed=2):
    """Write a tab-sim style zarr of true ``vis_ast`` and point the config at it."""
    rng = np.random.default_rng(seed)
    shape = (config.n_time, config.n_bl, config.n_freq)
    vis_ast = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    path = tmp_path / "sim.zarr"
    xr.Dataset({"vis_ast": (("time", "bl", "freq"), vis_ast)}).to_zarr(path, mode="w")
    config.args["data"]["zarr_path"] = str(path)

    return config


def setup_ast(config):
    comp = GPVisAst()
    comp.setup(config)
    return comp


def forward_vis_ast(comp):
    """``vis_ast`` at the initial parameters, through the real forward function."""
    state = comp.build_forward()(
        comp.init_params_base, dict(comp.state_outputs), make_constants(comp)
    )
    return np.asarray(state["vis_ast"])


def setup_error(config):
    """The message of the error ``GPVisAst.setup`` raises for this config."""
    with pytest.raises(RuntimeError) as excinfo:
        setup_ast(config)
    return str(excinfo.value)


def offered_options(message):
    """The options an error message offers, from its ``Choose from (...)`` list."""
    listed = re.search(r"Choose from \(([^)]*)\)", message)
    assert listed, f"error message offers no option list: {message}"
    return tuple(option.strip() for option in listed.group(1).split(","))


def base_config_comment(key):
    """The trailing ``#`` comment on ``key`` in the ``ast`` block of the base config."""
    path = files("tabascal").joinpath("data/config/tab_config_base.yaml")
    lines = path.read_text().splitlines()

    start = lines.index("ast:")
    for line in lines[start + 1 :]:
        if line and not line.startswith(" "):
            break
        match = re.match(rf"  {key}:[^#]*#(.*)", line)
        if match:
            return match.group(1)

    raise AssertionError(f"no 'ast: {key}:' line found in the base config")


# ---------------------------------------------------------------------------
# zeros
# ---------------------------------------------------------------------------


def test_zeros_init_is_exactly_the_zero_latent(tmp_path):
    """``zeros`` encodes a zero signal, and that is the zero latent exactly.

    The handler puts zero visibilities through the same ``signal_to_latent`` path
    the ``data`` init uses rather than writing the zero latent directly, so this
    is the claim that the encoding is linear and unbiased rather than an
    assumption about it. ``mean: data`` makes the prior mean non-zero, so a
    handler that quietly fell back to the prior would fail here.
    """
    comp = setup_ast(ast_config(tmp_path, init="zeros", mean="data"))

    assert jnp.any(comp.mu_ast_k != 0)
    assert jnp.all(comp.init_ast_k == 0)


def test_zeros_init_gives_zero_visibilities_through_the_forward(tmp_path, exact_rtol):
    """The optimisation starts from an identically zero sky, not merely a small one.

    The latent zero is exact, but the parameters actually handed to the forward
    are ``(0 - mu) / sigma``, so the forward reconstructs zero by cancelling
    ``mu`` against itself. What survives is rounding, measured against the scale
    of the sky the ``data`` init would have started from.
    """
    zeros = forward_vis_ast(setup_ast(ast_config(tmp_path, init="zeros", mean="data")))
    data = forward_vis_ast(setup_ast(ast_config(tmp_path, init="data", mean="data")))

    assert np.max(np.abs(data)) > 0
    assert np.max(np.abs(zeros)) <= exact_rtol * np.max(np.abs(data))


def test_zeros_init_is_not_the_prior_mean(tmp_path):
    """``zeros`` and ``prior`` part company as soon as the prior mean is the data.

    They agree at the default ``mean: 0``, which is exactly why the difference
    has to be pinned somewhere the mean is something else -- otherwise ``zeros``
    could be an alias for ``prior`` and nothing would notice.
    """
    zeros = forward_vis_ast(setup_ast(ast_config(tmp_path, init="zeros", mean="data")))
    prior = forward_vis_ast(setup_ast(ast_config(tmp_path, init="prior", mean="data")))

    assert np.max(np.abs(prior)) > 0
    assert np.max(np.abs(prior - zeros)) > 0


# ---------------------------------------------------------------------------
# The option lists
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("init", INIT_OPTIONS)
def test_every_accepted_init_option_constructs(tmp_path, init):
    config = with_truth(ast_config(tmp_path, init=init), tmp_path)

    comp = setup_ast(config)

    latent_shape = (config.n_bl, comp.n_k_freq_ast, comp.n_k_time_ast)
    assert comp.init_params_base["ast_k_r_base"].shape == latent_shape
    assert np.all(np.isfinite(comp.init_params_base["ast_k_r_base"]))
    assert np.all(np.isfinite(comp.init_params_base["ast_k_i_base"]))


@pytest.mark.parametrize("mean", MEAN_OPTIONS)
def test_every_accepted_mean_option_constructs(tmp_path, mean):
    config = ast_config(tmp_path, mean=mean)

    comp = setup_ast(config)

    assert comp.mu_ast_k.shape == (config.n_bl, comp.n_k_freq_ast, comp.n_k_time_ast)


def test_the_init_error_offers_exactly_the_options_that_work(tmp_path):
    """The bug this file exists for: an error text advertising an option that raises.

    ``est`` was one of the two the base config recommended, so it is what a user
    following the old documentation would have written.
    """
    message = setup_error(ast_config(tmp_path, init="est"))

    assert offered_options(message) == INIT_OPTIONS
    for option in offered_options(message):
        setup_ast(with_truth(ast_config(tmp_path, init=option), tmp_path))


def test_the_mean_error_offers_exactly_the_options_that_work(tmp_path):
    message = setup_error(ast_config(tmp_path, mean="est"))

    assert offered_options(message) == tuple(str(o) for o in MEAN_OPTIONS)
    for option in offered_options(message):
        # `0` is a number in the config, not the string the error text prints.
        value = int(option) if option.isdigit() else option
        setup_ast(ast_config(tmp_path, mean=value))


@pytest.mark.parametrize("init", ["est", "truth_mean"])
def test_the_options_the_base_config_used_to_advertise_now_raise(tmp_path, init):
    message = setup_error(ast_config(tmp_path, init=init))

    assert f"Provided init type: {init} is not valid" in message


def test_the_base_config_comments_name_exactly_the_accepted_options():
    """The base config is what a user copies, so its comment is documentation.

    Checked against the same lists the tests above construct with, in both
    directions: an option named there that the code rejects sends the user round
    the loop this task closed, and an option the code takes but the comment omits
    is a feature nobody can find.
    """
    init_comment = base_config_comment("init")
    assert set(re.findall(r"'([^']+)'", init_comment)) == set(INIT_OPTIONS)

    mean_comment = base_config_comment("mean")
    assert set(re.findall(r"'([^']+)'", mean_comment)) == {"data", "zeros"}
    assert re.search(r"(?<![\w.])0(?![\w.])", mean_comment), (
        f"ast.mean accepts 0 but the base config comment does not say so: {mean_comment}"
    )


# ---------------------------------------------------------------------------
# mean
# ---------------------------------------------------------------------------


def test_ast_mean_zero_and_zeros_are_the_same_prior(tmp_path):
    """``0`` and ``zeros`` are one option under two spellings, not two options."""
    numeric = setup_ast(ast_config(tmp_path, mean=0, init="sample"))
    named = setup_ast(ast_config(tmp_path, mean="zeros", init="sample"))

    np.testing.assert_array_equal(numeric.mu_ast_k, named.mu_ast_k)
    for key, value in numeric.init_params_base.items():
        np.testing.assert_array_equal(value, named.init_params_base[key])


# ---------------------------------------------------------------------------
# The baseline block scan
#
# latent_to_signal pads the latent grid up to the padded k-grid, transforms and
# crops back, so vmapping it over every baseline holds (n_bl, n_freq_pad,
# n_time_pad) several times over -- padding the crop then throws away. The
# component walks the baseline axis in blocks of ast.baseline_block_size
# instead. Baselines are independent, so the block changes memory and nothing
# else; these tests are what holds that.
# ---------------------------------------------------------------------------

#: 6 baselines at n_ant=4, so these span one baseline per step, blocks that do
#: and do not divide the axis, a block wider than it, and null (a single block).
BLOCK_SIZES = [1, 4, 6, 128, None]


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
def test_the_baseline_block_size_changes_neither_value_nor_gradient(
    tmp_path, block_size
):
    """The scanned transform reproduces the unblocked one, block for block.

    Value and gradient both: the scan carries the affine transform into its body
    and pads its last block, either of which could go wrong in a way that only
    reverse mode would show.
    """

    def value_and_grad(block):
        comp = setup_ast(ast_config(tmp_path, baseline_block_size=block))
        constants = make_constants(comp)
        forward = comp.build_forward()
        state = dict(comp.state_outputs)

        def loss(params):
            vis = forward(params, state, constants)["vis_ast"]
            return jnp.sum(jnp.abs(vis) ** 2)

        params = comp.init_params_base
        vis = forward(params, state, constants)["vis_ast"]

        return np.asarray(vis), jax.grad(loss)(params)

    # The reference is the whole axis in one step, i.e. the vmap this replaced.
    ref_vis, ref_grads = value_and_grad(None)
    vis, grads = value_and_grad(block_size)

    np.testing.assert_allclose(vis, ref_vis, rtol=1e-12, atol=1e-12)
    for key in ref_grads:
        np.testing.assert_allclose(
            np.asarray(grads[key]), np.asarray(ref_grads[key]), rtol=1e-12, atol=1e-12
        )


def test_the_default_baseline_block_size_is_sized_from_the_padded_grid(tmp_path):
    """A config predating the key still builds, on the base default of ``auto``.

    The point of ``auto`` is that a block which does not bind is pure overhead,
    so on a grid this small it has to come out as a single step over the whole
    axis rather than as some fixed count.
    """

    args = base_args(tmp_path)
    del args["ast"]["baseline_block_size"]

    config = make_ast_config(args)
    comp = GPVisAst()
    comp.setup(config)

    assert comp.baseline_block_size_setting == "auto"
    assert comp.baseline_block_size == config.n_bl


def test_auto_blocks_a_padded_grid_that_does_not_fit_the_budget(tmp_path):
    """And binds where the grid is large, which is the case it exists for."""

    comp = setup_ast(ast_config(tmp_path, baseline_block_size="auto"))
    padded = 1
    for dim, (lo, hi) in zip((comp.n_k_freq_ast, comp.n_k_time_ast), comp.pads):
        padded *= dim + lo + hi

    # The same arithmetic the component does, against a budget cut so far that
    # the grid cannot fit: the block has to fall well short of the axis.
    comp._BLOCK_BUDGET_BYTES = 8 * padded
    comp._resolve_baseline_block_size()

    assert 1 <= comp.baseline_block_size < comp.n_bl


@pytest.mark.parametrize(
    "block_size", [0, -1, 1.5, True, "128", float("inf"), float("nan")]
)
def test_a_baseline_block_size_that_is_not_a_positive_whole_number_is_rejected(
    tmp_path, block_size
):
    """Rejected in setup, by name. ``None`` is not here: null is a setting."""

    message = setup_error(ast_config(tmp_path, baseline_block_size=block_size))

    assert "baseline_block_size" in message


# ---------------------------------------------------------------------------
# ast.pow_spec
# ---------------------------------------------------------------------------


def pow_spec_config(tmp_path, **overrides):
    """A config whose ``ast.pow_spec`` block carries ``overrides``."""
    config = ast_config(tmp_path)
    config.args["ast"]["pow_spec"] = {
        **config.args["ast"]["pow_spec"],
        **overrides,
    }
    return config


class TestThePriorAmplitudeIsTheWidthItClaims:
    """``std`` is the width of the prior, in Jy, and nothing else moves it.

    The spectrum is normalised to it, so the configured number is the realised
    width -- not its square, not its square root, not a constant times it. That
    is what lets the guidance be "read the amplitude off a clean channel and
    put it here", and what these tests pin.
    """

    def _realised_std(self, tmp_path, draws=400, seed=0, **overrides):
        """The width of the prior these settings actually produce, sampled.

        Through the component's real ``build_forward``, so the pad, the scan
        over the baseline axis and the crop are all in the path -- a helper
        that redid the transform by hand would not have caught a block sized
        wrongly or a per-baseline sigma paired with the wrong baseline. And
        over *every* baseline, not baseline 0, since the normalisation is per
        baseline and one of them tells you nothing about the rest.

        The latent is drawn the way :meth:`GPVisAst.build_set_params` draws it
        -- a real and an imaginary standard normal -- and not as one complex
        normal. They are not the same distribution: JAX's complex normal is
        circularly symmetric with ``E|z|^2 = 1``, the model's carries 2, and a
        check that draws the first while measuring a complex width gets the
        right answer from two errors of ``sqrt(2)`` cancelling.
        """
        from jax import random

        comp = setup_ast(pow_spec_config(tmp_path, **overrides))
        forward = comp.build_forward()
        shape = (comp.n_bl, comp.n_k_freq_ast, comp.n_k_time_ast)

        total, count = 0.0, 0
        for draw in range(draws):
            keys = random.split(random.PRNGKey(seed + draw), 2)
            params = {
                "ast_k_r_base": random.normal(keys[0], shape),
                "ast_k_i_base": random.normal(keys[1], shape),
            }
            vis = forward(params, dict(comp.state_outputs), make_constants(comp))[
                "vis_ast"
            ]
            total += float(jnp.sum(jnp.abs(vis) ** 2))
            count += vis.size

        # rms|V|, which is what `std` is defined as -- the same quantity a user
        # reads off a clean channel.
        return float(np.sqrt(total / count))

    def test_the_configured_number_is_the_prior_width(self, tmp_path):
        """Not its square, not its square root, and not times a constant.

        ``rms|V|`` specifically, which is the quantity the guidance names: read
        the amplitude off a channel with no RFI in it and put that number here.
        """
        assert self._realised_std(tmp_path, std=30.0) == pytest.approx(30.0, rel=0.02)

    def test_it_is_the_complex_width_and_not_the_real_part(self, tmp_path):
        """The factor of sqrt(2) that the first version of this got wrong.

        The two differ by exactly that, so a prior normalised per component
        would sit here at 30 / sqrt(2); asserting both ends pins which one the
        key means and would catch the factor going missing again.
        """
        from jax import random, vmap
        from tabascal.fft_gp import latent_to_signal

        comp = setup_ast(pow_spec_config(tmp_path, std=30.0))
        sigma = comp.sigma_ast_k[0]
        keys = random.split(random.PRNGKey(0), 2)
        shape = (6000, *sigma.shape)
        base = random.normal(keys[0], shape) + 1j * random.normal(keys[1], shape)
        vis = vmap(latent_to_signal, (0, None, None), 0)(
            sigma * base, comp.pads, comp.ss_idxs
        )

        assert float(jnp.sqrt(jnp.mean(jnp.abs(vis) ** 2))) == pytest.approx(30.0, rel=0.02)
        assert float(jnp.std(vis.real)) == pytest.approx(30.0 / jnp.sqrt(2.0), rel=0.02)

    @pytest.mark.parametrize(
        "overrides",
        [
            # Both of these actually drop modes on this fixture: 0.5 keeps 40
            # of 128 and 0.1 keeps 120. A cutoff below about 0.02 keeps all
            # of them here, so the 1e-3/1e-9 pair this started with was
            # re-running the baseline case without varying the one thing it is
            # here to vary.
            {"cutoff": 0.5},
            {"cutoff": 0.1},
            {"gammas": [3.0, 3.0]},
            {"gammas": [8.0, 8.0]},
            {"fov_deg": 2.0},
        ],
        ids=["cutoff keeps 40/128", "cutoff keeps 120/128", "shallow gammas", "steep gammas", "narrow fov"],
    )
    def test_the_width_survives_the_other_settings(self, tmp_path, overrides):
        """Each of these changes which modes are fitted or how they are weighted.
        None of them is a statement about how bright the sky is, so none of
        them may move the width of the prior on it.
        """
        assert self._realised_std(tmp_path, std=30.0, **overrides) == pytest.approx(
            30.0, rel=0.02
        )

    def test_every_baseline_gets_the_configured_width(self, tmp_path):
        """Each baseline's knee is its own maximum fringe rate.

        Normalising per baseline is what makes ``std`` the width everywhere
        rather than on an average baseline: unnormalised, the shipped
        configuration spans a factor of 1.23 across its baselines.
        """
        comp = setup_ast(pow_spec_config(tmp_path, std=30.0))

        # The realised variance, which is twice the sum of the mode variances:
        # the latent carries E|z|^2 = 2. Asserting the sum alone would pin the
        # arithmetic without saying what it is for.
        realised = 2 * jnp.sum(comp.sigma_ast_k**2, axis=(1, 2))

        assert np.allclose(np.asarray(realised), 30.0**2, rtol=1e-5)

    @pytest.mark.skipif(
        jax.config.jax_enable_x64,
        reason="float64 holds every std the validator accepts, so the guard "
        "cannot fire: any finite positive float is representable there",
    )
    @pytest.mark.parametrize("std", [1e-50, 1e40], ids=["underflows", "overflows"])
    def test_a_std_the_precision_cannot_hold_is_refused(self, tmp_path, std):
        """Finite and positive is not the same as representable.

        The validator takes any finite positive float, but sigma is built in
        the run's own precision. In float32 these flush to zero and to
        infinity, and the first thing that divides by sigma -- encoding the
        initial sky -- then yields non-finite parameters, with setup already
        past and nothing left pointing at std. Single precision only, because
        float64 holds anything the validator lets through.
        """
        message = setup_error(pow_spec_config(tmp_path, std=std))

        assert "ast.pow_spec.std" in message
        assert "representable" in message

    def test_data_measures_the_width_per_baseline(self, tmp_path, capsys):
        """``std: data`` is not an estimate of the width -- it is the width.

        ``std`` is defined as rms|V|, and that is exactly what this measures,
        so the number the data gives is the number the prior gets, per
        baseline, with no conversion in between.
        """
        config = pow_spec_config(tmp_path, std="data")
        comp = setup_ast(config)

        expected = np.sqrt(
            np.mean(np.abs(np.asarray(config.vis_obs)) ** 2, axis=(1, 2))
        )
        realised = np.sqrt(np.asarray(2 * jnp.sum(comp.sigma_ast_k**2, axis=(1, 2))))

        assert np.allclose(realised, expected, rtol=1e-5)
        assert realised.shape == (comp.n_bl,)
        # One width per baseline, and they differ -- otherwise this would pass
        # against a scalar too.
        assert realised.std() > 0

    def test_data_ignores_what_is_flagged(self, tmp_path):
        """Which is the whole reason to take the mask rather than the array.

        Half the samples are given an amplitude ten times the rest and then
        flagged; the width has to come back as the unflagged half alone.
        """
        config = pow_spec_config(tmp_path, std="data")
        vis = np.asarray(config.vis_obs).copy()
        flags = np.zeros(vis.shape, dtype=bool)
        flags[:, :, ::2] = True
        vis[:, :, ::2] *= 10.0
        config.vis_obs = jnp.asarray(vis)
        config.flags = jnp.asarray(flags)

        comp = setup_ast(config)

        kept = np.sqrt(np.mean(np.abs(vis[:, :, 1::2]) ** 2, axis=(1, 2)))
        realised = np.sqrt(np.asarray(2 * jnp.sum(comp.sigma_ast_k**2, axis=(1, 2))))

        assert np.allclose(realised, kept, rtol=1e-5)

    def test_data_says_so_when_nothing_is_flagged(self, tmp_path, capsys):
        """Because then it is measuring the RFI as well as the sky.

        On the shipped 8A simulation, whose RFI is unflagged because modelling
        it is the job, this returns about 11 Jy against a true sky of 1.7. The
        estimate is only the sky where the contamination has been flagged, and
        nothing else in the run will say so.
        """
        setup_ast(pow_spec_config(tmp_path, std="data"))

        printed = capsys.readouterr().out
        assert "none is flagged" in printed
        assert "RFI" in printed

    def test_data_with_everything_flagged_is_refused(self, tmp_path):
        """There is nothing to measure, and a zero width is not a prior."""
        config = pow_spec_config(tmp_path, std="data")
        config.flags = jnp.ones(jnp.shape(config.vis_obs), dtype=bool)

        message = setup_error(config)

        assert "nothing to measure" in message

    def test_a_word_other_than_data_is_refused(self, tmp_path):
        """`data` is the only word; anything else is a typo, not a setting."""
        message = setup_error(pow_spec_config(tmp_path, std="truth"))

        assert "ast.pow_spec.std" in message
        assert "'data'" in message


class TestTheFrequencyKneeIsAskedForAsABandwidth:
    """``corr_freq`` is a correlation bandwidth in Hz, not a knee.

    The knee it sets is a delay -- the frequency axis transforms to
    ``fftfreq(n_freq, chan_width)``, whose units are inverse Hz -- and a delay
    is not a quantity anyone has intuition for at a glance. A bandwidth is, and
    it is the spelling ``rfi.corr_freq`` already uses.
    """

    def test_the_knee_is_the_reciprocal_of_the_bandwidth(self, tmp_path):
        """Pinned by a test rather than inferred from the name.

        The same conversion ``rfi_signal`` makes from ``rfi.corr_freq``, and
        the reason the two sections can be read side by side.
        """
        comp = setup_ast(pow_spec_config(tmp_path, corr_freq=1e6))

        assert comp.k0_freq == pytest.approx(1 / (2 * np.pi * 1e6))

    def test_it_is_the_conversion_the_rfi_prior_uses(self, tmp_path):
        """One conversion, called from both, rather than two that agree today.

        The astronomical and RFI priors disagreeing about what their frequency
        knee meant is the whole of GitHub #117, and it happened because the
        same arithmetic was written out in both places.
        """
        from tabascal.fft_gp import knee_from_corr_scale

        comp = setup_ast(pow_spec_config(tmp_path, corr_freq=1e6))

        assert comp.k0_freq == knee_from_corr_scale(1e6)

    def test_null_is_no_roll_off_along_the_frequency_axis(self, tmp_path):
        """Which is what the shipped default has always done, said out loud.

        The power spectrum tends to ``p0`` as the knee grows, so an infinite
        knee keeps every delay mode and prefers none.
        """
        comp = setup_ast(pow_spec_config(tmp_path, corr_freq=None))

        assert comp.k0_freq == float("inf")

    def test_null_reproduces_the_default_it_replaces(self):
        """The rename moves no result, and this is the measurement saying so.

        On a single channel the two are bit-identical -- the only delay mode is
        zero -- and on a wide band they differ by less than single precision's
        epsilon, so no shipped configuration and no reference moves.
        """
        from tabascal.fft_gp import knee_from_corr_scale, latent_to_signal_init

        def pk(k0_freq, n_freq):
            spectrum, *_ = latent_to_signal_init(
                [n_freq, 120], [209e3, 2.0], [2.0, 2.0], [1, 1],
                3e3, [k0_freq, 1e-3], [5.0, 5.0], 1e-6,
            )
            return np.asarray(spectrum)

        # Through the helper, not a literal infinity: the claim is about what
        # `corr_freq: null` does. Comparing the two alone would be satisfied by
        # a null path that had simply become the old default, so the flatness
        # it is supposed to produce is asserted first and separately.
        unset = knee_from_corr_scale(None)

        flat = pk(unset, 32)
        assert np.allclose(flat, flat[0, :][None, :], rtol=1e-12), (
            "an unset corr_freq must leave the frequency axis flat"
        )

        assert np.array_equal(pk(1.0, 1), pk(unset, 1))

        was, now = pk(1.0, 32), flat
        assert was.shape == now.shape
        assert np.max(np.abs(was - now) / now) < np.finfo(np.float32).eps


class TestAstPowSpecIsValidated:
    """``ast.pow_spec`` went straight to the Fourier machinery unchecked.

    A negative exponent, a string, or a cutoff of 1 -- which cuts every mode --
    surfaced from inside ``fft_gp`` with a message about array shapes, if it
    surfaced at all. The RFI prior is checked by the same validator, so these
    tests are the astronomical half of one contract rather than a second one.
    """

    def test_the_shipped_defaults_still_set_up(self, tmp_path):
        """The base config's own block has to pass its own validation."""
        comp = setup_ast(pow_spec_config(tmp_path))

        assert comp.n_k_freq_ast >= 1 and comp.n_k_time_ast >= 1

    @pytest.mark.parametrize("key", ["std", "corr_freq", "cutoff"])
    @pytest.mark.parametrize("value", [0, -1, "3e3", True, float("inf"), float("nan")])
    def test_a_scalar_key_that_is_not_a_positive_number_is_refused(
        self, tmp_path, key, value
    ):
        message = setup_error(pow_spec_config(tmp_path, **{key: value}))

        assert f"ast.pow_spec.{key}" in message

    @pytest.mark.parametrize("cutoff", [1.0, 2.0])
    def test_a_cutoff_that_cuts_every_mode_is_refused(self, tmp_path, cutoff):
        """Relative to the largest mode on each axis, and the comparison is
        strict, so 1 leaves nothing to fit. Unchecked it reached fft_gp and came
        back as a zero-size reduction."""
        message = setup_error(pow_spec_config(tmp_path, cutoff=cutoff))

        assert "ast.pow_spec.cutoff" in message
        assert "below 1" in message

    @pytest.mark.parametrize(
        "gammas", [5, "55", [5], [5, 5, 5], [-1, 5], [0, 5], {5: None, 6: None}, {5, 6}]
    )
    def test_gammas_that_are_not_an_ordered_pair_of_positives_are_refused(
        self, tmp_path, gammas
    ):
        assert "gammas" in setup_error(pow_spec_config(tmp_path, gammas=gammas))

    @pytest.mark.parametrize("gammas", [(5, 5), np.array([5.0, 5.0])])
    def test_an_ordered_pair_is_accepted_however_it_is_spelled(self, tmp_path, gammas):
        comp = setup_ast(pow_spec_config(tmp_path, gammas=gammas))

        assert comp.gammas == [5.0, 5.0]

    def test_an_unknown_key_is_refused_by_name(self, tmp_path):
        """`gamma` for `gammas` is the mistake this catches.

        Asserted against the offending-key list rather than the whole message:
        every message ends "It takes [... 'gammas' ...]", so a bare
        `"gamma" in message` passes for a validator that names nothing. The same
        trap was fixed on the RFI side and then walked into again here.
        """
        message = setup_error(pow_spec_config(tmp_path, gamma=5))

        assert "no key(s) ['gamma']" in message

    def test_the_two_knees_may_be_null_and_the_others_may_not(self, tmp_path):
        """And they mean different things by it.

        null fov_deg is the telescope's own beam; null corr_freq is no roll-off
        along the frequency axis at all. The rest have no such fallback, and an
        unset one is a config that cannot be run rather than a default to
        invent."""
        setup_ast(pow_spec_config(tmp_path, fov_deg=None))
        setup_ast(pow_spec_config(tmp_path, corr_freq=None))

        for key in ("std", "gammas", "cutoff"):
            message = setup_error(pow_spec_config(tmp_path, **{key: None}))
            assert f"ast.pow_spec.{key}" in message
            assert "required" in message

    def test_the_validation_is_the_one_the_rfi_prior_uses(self):
        """One contract, not two: the sections differ in which keys are live."""
        from tabascal.components.ast_vis import validate_pow_spec as ast_validator
        from tabascal.components.rfi_signal import validate_pow_spec as rfi_validator

        assert ast_validator is rfi_validator
