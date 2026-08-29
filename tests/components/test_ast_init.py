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
