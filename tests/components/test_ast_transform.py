"""``ast.transform: dft`` -- the astronomical GP with the padded grid never formed.

The surviving Fourier modes reach the visibilities through a matrix per axis
rather than through a zero-pad, a full inverse FFT of the padded grid and a
crop (``ast.transform: fft``). The DFT is therefore held to the FFT in value
and in gradient, not to a tolerance of its own, and the transform itself is
held to the chain it stands for.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tabascal.components.ast_vis import GPVisAst, time_pad_for_fringe_rate
from tabascal.fft_gp import (
    latent_to_signal,
    latent_to_signal_dft,
    latent_to_signal_dft_init,
    latent_to_signal_init,
    pad_domain_k,
)

from .conftest import active_precision, make_constants
from tests.components.test_ast_init import ast_config


def _tols():
    """The two routes sum the same terms in a different order.

    In double that is round-off on a sum of a few hundred modes. In single the
    FFT's pairwise summation and the matrix product's accumulation differ by
    more than either does from the exact answer, so the comparison is loosened
    rather than the DFT being called wrong: the double case is what pins the
    values.
    """
    return (1e-10, 1e-10) if active_precision() == "double" else (2e-4, 2e-4)


# --- the transform ---------------------------------------------------------


@pytest.mark.parametrize(
    "ns, pad_factors, ss_factors",
    [
        ([6, 10], [2.0, 2.0], [1, 1]),
        ([4, 7], [2.0, 2.0], [1, 1]),
        ([1, 12], [2.0, 2.0], [1, 1]),  # a single channel
        ([5, 8], [1.5, 3.0], [1, 1]),  # padding that is not the default
        ([4, 6], [2.0, 2.0], [1, 2]),  # supersampled, as the RFI path is
    ],
)
def test_the_matrices_are_the_pad_ifft_crop_chain(ns, pad_factors, ss_factors):
    dxs = [1.3e5, 2.0]
    k0s, gammas, cutoff = [1 / 3.0e5, 1 / 40.0], [2.0, 2.0], 1e-3

    pk, _, pads, ss_idxs = latent_to_signal_init(
        ns, dxs, pad_factors, ss_factors, 1.0, k0s, gammas, cutoff
    )
    pk_dft, _, mats, first_axis = latent_to_signal_dft_init(
        ns, dxs, pad_factors, ss_factors, 1.0, k0s, gammas, cutoff
    )
    assert pk_dft.shape == pk.shape

    rng = np.random.default_rng(0)
    Y = jnp.asarray(
        rng.normal(size=pk.shape) + 1j * rng.normal(size=pk.shape),
        dtype=jnp.result_type(jnp.complex64, jnp.zeros(0, float).dtype),
    )
    expected = latent_to_signal(Y, pads, ss_idxs)
    got = latent_to_signal_dft(Y, mats, first_axis)

    assert got.shape == expected.shape
    rtol, atol = _tols()
    scale = float(jnp.max(jnp.abs(expected)))
    np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol * max(scale, 1.0))


def test_both_contraction_orders_agree():
    """``first_axis`` is a memory choice, not a numerical one."""
    ns, dxs = [6, 10], [1.3e5, 2.0]
    pk, _, mats, _ = latent_to_signal_dft_init(
        ns, dxs, [2.0, 2.0], [1, 1], 1.0, [1 / 3.0e5, 1 / 40.0], [2.0, 2.0], 1e-3
    )
    rng = np.random.default_rng(1)
    Y = jnp.asarray(rng.normal(size=pk.shape) + 1j * rng.normal(size=pk.shape))

    rtol, atol = _tols()
    np.testing.assert_allclose(
        latent_to_signal_dft(Y, mats, 0), latent_to_signal_dft(Y, mats, 1),
        rtol=rtol, atol=atol,
    )


def test_the_matrices_do_not_depend_on_the_working_precision():
    """The phase is an exact integer ratio, so single precision keeps it.

    A float product would carry up to N/2 cycles into the rounding before the
    exponential ever saw it, which is the reason the phase is reduced modulo
    the grid length first.
    """
    ns, dxs = [6, 64], [1.3e5, 2.0]
    args = (ns, dxs, [2.0, 2.0], [1, 1], 1.0, [1 / 3.0e5, 1 / 40.0], [2.0, 2.0], 1e-3)
    *_, mats, _ = latent_to_signal_dft_init(*args)
    for m in mats:
        # Every entry is on the unit circle, to the precision it is stored in.
        np.testing.assert_allclose(np.abs(np.asarray(m)), 1.0, rtol=0, atol=1e-6)


# --- the component ---------------------------------------------------------


def _route(transform, config):
    config.args["ast"]["transform"] = transform
    comp = GPVisAst()
    comp.setup(config)
    forward = comp.build_forward()
    constants = make_constants(comp)
    state = dict(comp.state_outputs)

    def vis(params):
        return forward(params, state, constants)["vis_ast"]

    return comp, vis


@pytest.mark.parametrize("block_size", [None, 2])
def test_the_dft_matches_the_fft(tmp_path, block_size):
    config = ast_config(tmp_path, baseline_block_size=block_size)
    ref_comp, ref_vis = _route("fft", config)
    dft_comp, dft_vis = _route("dft", ast_config(tmp_path, baseline_block_size=block_size))

    # The same prior and the same modes: only the transform differs.
    assert dft_comp.pk.shape == ref_comp.pk.shape
    np.testing.assert_array_equal(
        np.asarray(dft_comp.init_params_base["ast_k_r_base"]),
        np.asarray(ref_comp.init_params_base["ast_k_r_base"]),
    )

    expected = np.asarray(ref_vis(ref_comp.init_params_base))
    got = np.asarray(dft_vis(dft_comp.init_params_base))

    rtol, atol = _tols()
    np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol * max(np.abs(expected).max(), 1.0))


def test_the_dft_gradients_match_the_fft(tmp_path):
    """Reverse mode too: the transpose of a matrix product is not the transpose
    of a pad, a shift and an inverse transform, even where the values agree."""
    config = ast_config(tmp_path)

    def value_and_grad(transform):
        comp, vis = _route(transform, ast_config(tmp_path))

        def loss(params):
            return jnp.sum(jnp.abs(vis(params)) ** 2)

        return jax.grad(loss)(comp.init_params_base)

    ref = value_and_grad("fft")
    got = value_and_grad("dft")

    rtol, atol = _tols()
    for key in ref:
        scale = max(float(np.abs(np.asarray(ref[key])).max()), 1.0)
        np.testing.assert_allclose(
            np.asarray(got[key]), np.asarray(ref[key]), rtol=rtol, atol=atol * scale,
            err_msg=key,
        )


def test_the_dft_has_no_fft_in_its_compiled_program(tmp_path):
    """The point of the transform: no inverse FFT, so no padded grid to hold.

    Read off the compiled program rather than timed. The FFT is compiled
    beside it, so the check is that the two differ in this, not that
    some name happens to be absent.
    """
    fft_comp, fft_vis = _route("fft", ast_config(tmp_path))
    dft_comp, dft_vis = _route("dft", ast_config(tmp_path))

    fft_text = jax.jit(fft_vis).lower(fft_comp.init_params_base).compile().as_text()
    dft_text = jax.jit(dft_vis).lower(dft_comp.init_params_base).compile().as_text()

    # ``fft_type`` marks the HLO operation itself; the bare word also appears
    # in the source-file metadata of both, this module's name among it.
    assert "fft_type" in fft_text
    assert "fft_type" not in dft_text


def test_auto_blocking_is_a_single_step_under_the_dft(tmp_path):
    """``auto`` exists to bound the padded grid. This transform builds none,
    and the scan's own stack of outputs costs memory, so blocking it is worse
    on both counts and ``auto`` should not do it."""
    comp, _ = _route("dft", ast_config(tmp_path, baseline_block_size="auto"))

    assert comp.baseline_block_size == comp.n_bl


def test_an_explicit_block_is_still_honoured(tmp_path):
    """The way to trade the room back if another part of a model wants it."""
    comp, vis = _route("dft", ast_config(tmp_path, baseline_block_size=2))
    assert comp.baseline_block_size == 2

    single, single_vis = _route("dft", ast_config(tmp_path, baseline_block_size=None))
    rtol, atol = _tols()
    np.testing.assert_allclose(
        np.asarray(vis(comp.init_params_base)),
        np.asarray(single_vis(single.init_params_base)),
        rtol=rtol, atol=atol,
    )


def test_one_pair_of_matrices_serves_every_baseline(tmp_path):
    """The transform is the same for every baseline, so the matrices carry no
    baseline axis and do not grow with the array.

    The modes differ per baseline, the map that turns them into visibilities
    does not: it is set by the grid and the surviving modes alone. A change
    that gave the matrices a baseline axis would cost ``n_bl`` times their
    size and would not be visible in any value this file checks.
    """
    comp, _ = _route("dft", ast_config(tmp_path))

    n_out = [comp.n_freq, comp.n_time]
    n_k = [comp.n_k_freq_ast, comp.n_k_time_ast]
    for axis, mat in enumerate(comp.dft_mats):
        assert mat.ndim == 2, "a baseline axis would make this rank three"
        assert mat.shape == (n_k[axis], n_out[axis])

    # And the transform broadcasts them rather than mapping over them.
    bigger, _ = _route("dft", ast_config(tmp_path))
    for small, large in zip(comp.dft_mats, bigger.dft_mats):
        assert small.shape == large.shape


# --- wide time padding --------------------------------------------------------


def _random_latent(comp, seed=3):
    """A latent away from the init, so a zero start cannot make two routes agree."""
    rng = np.random.default_rng(seed)
    return {
        k: jnp.asarray(rng.normal(size=v.shape), dtype=v.dtype)
        for k, v in comp.init_params_base.items()
    }


@pytest.mark.parametrize("time_pad_factor", [2.0, 8.0, 16.0])
def test_the_dft_matches_the_fft_at_wide_time_padding(tmp_path, time_pad_factor):
    """A time_pad_factor of 8 removes a start/end-of-pass excess in long passes
    and quadruples the padded grid, which is the case this component is for. The modes it keeps
    grow with the padding, so 2 alone does not cover the matrices it builds there."""
    ref_comp, ref_vis = _route("fft", ast_config(tmp_path, time_pad_factor=time_pad_factor))
    dft_comp, dft_vis = _route("dft", ast_config(tmp_path, time_pad_factor=time_pad_factor))

    assert dft_comp.pk.shape == ref_comp.pk.shape
    if time_pad_factor > 2:
        narrow, _ = _route("fft", ast_config(tmp_path, time_pad_factor=2.0))
        assert ref_comp.n_k_time_ast > narrow.n_k_time_ast

    params = _random_latent(ref_comp)
    expected = np.asarray(ref_vis(params))
    got = np.asarray(dft_vis(params))

    rtol, atol = _tols()
    np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol * max(np.abs(expected).max(), 1.0))


@pytest.mark.parametrize("time_pad_factor", [2.0, 8.0, 16.0])
def test_the_dft_gradients_match_the_fft_at_wide_time_padding(tmp_path, time_pad_factor):
    ref_comp, ref_vis = _route("fft", ast_config(tmp_path, time_pad_factor=time_pad_factor))
    _, dft_vis = _route("dft", ast_config(tmp_path, time_pad_factor=time_pad_factor))
    params = _random_latent(ref_comp)
    # A loss with data in it, so the gradient is not a multiple of the value.
    target = np.asarray(ref_vis(_random_latent(ref_comp, seed=4)))

    def grad(vis):
        return jax.grad(lambda p: jnp.sum(jnp.abs(vis(p) - target) ** 2))(params)

    ref, got = grad(ref_vis), grad(dft_vis)

    rtol, atol = _tols()
    for key in ref:
        scale = max(float(np.abs(np.asarray(ref[key])).max()), 1.0)
        np.testing.assert_allclose(
            np.asarray(got[key]), np.asarray(ref[key]), rtol=rtol, atol=atol * scale,
            err_msg=key,
        )


# --- time padding from the fringe rate ----------------------------------------


def _first_time_mode(n_time, int_time, time_pad_factor):
    ks = np.asarray(pad_domain_k([n_time], [int_time], [time_pad_factor])[0])
    return np.abs(ks[ks != 0]).min()


@pytest.mark.parametrize("n_time, int_time, rate", [(8, 17.1, 1e-3), (77, 8.0, 3e-5), (150, 2.0, 1e-2)])
def test_the_padding_is_the_least_that_resolves_the_fringe_rate(n_time, int_time, rate):
    f = time_pad_for_fringe_rate(n_time, int_time, rate)
    assert _first_time_mode(n_time, int_time, f) <= rate
    if f > 1:
        # One cell a side fewer and the first mode is above the rate again.
        assert _first_time_mode(n_time, int_time, f - 2 / n_time) > rate


def _auto(tmp_path, fov_deg):
    config = ast_config(tmp_path, time_pad_factor="auto")
    config.args["ast"]["gp_cov"]["fov_deg"] = fov_deg
    comp = GPVisAst()
    comp.setup(config)
    return comp, config


@pytest.mark.parametrize("fov_deg", [None, 0.05])
def test_auto_reaches_the_slowest_baseline_and_no_less_than_the_floor(tmp_path, fov_deg):
    comp, config = _auto(tmp_path, fov_deg)
    needed = time_pad_for_fringe_rate(comp.n_time, comp.int_time, comp.ast_fr.min())

    assert comp.time_pad_factor == max(2.0, needed)
    assert config.args["ast"]["time_pad_factor"] == comp.time_pad_factor
    assert _first_time_mode(comp.n_time, comp.int_time, comp.time_pad_factor) <= comp.ast_fr.min()
    if fov_deg is not None:
        # Narrow enough that the floor is not what decided it.
        assert needed > 2.0


@pytest.mark.parametrize("bad", ["fast", 0.5, True, float("nan")])
def test_a_time_pad_factor_that_is_neither_a_factor_nor_auto_is_refused(tmp_path, bad):
    with pytest.raises(RuntimeError, match="time_pad_factor"):
        GPVisAst().setup(ast_config(tmp_path, time_pad_factor=bad))


def test_the_transform_defaults_to_the_dft_and_refuses_anything_else(tmp_path):
    comp = GPVisAst()
    comp.setup(ast_config(tmp_path))
    assert comp.transform == "dft"

    with pytest.raises(RuntimeError, match="ast.transform"):
        GPVisAst().setup(ast_config(tmp_path, transform="nufft"))
