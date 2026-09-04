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

    @pytest.mark.parametrize("key", ["p0", "k0_freq", "cutoff"])
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
        """`gamma` for `gammas` is the mistake this catches."""
        message = setup_error(pow_spec_config(tmp_path, gamma=5))

        assert "gamma" in message

    def test_fov_deg_may_be_null_and_the_others_may_not(self, tmp_path):
        """null fov_deg means the telescope's own beam; the rest have no such
        fallback, and an unset one is a config that cannot be run rather than a
        default to invent."""
        setup_ast(pow_spec_config(tmp_path, fov_deg=None))

        for key in ("p0", "k0_freq", "gammas", "cutoff"):
            message = setup_error(pow_spec_config(tmp_path, **{key: None}))
            assert f"ast.pow_spec.{key}" in message
            assert "required" in message

    def test_the_validation_is_the_one_the_rfi_prior_uses(self):
        """One contract, not two: the sections differ in which keys are live."""
        from tabascal.components.ast_vis import validate_pow_spec as ast_validator
        from tabascal.components.rfi_signal import validate_pow_spec as rfi_validator

        assert ast_validator is rfi_validator
