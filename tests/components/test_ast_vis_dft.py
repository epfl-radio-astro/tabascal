"""``GPVisAstDFT`` -- the astronomical GP with the padded grid never formed.

The component is :class:`GPVisAst` with one step replaced: the surviving
Fourier modes reach the visibilities through a matrix per axis rather than
through a zero-pad, a full inverse FFT of the padded grid and a crop. It is
therefore held to the FFT component in value and in gradient, not to a
tolerance of its own, and the transform itself is held to the chain it stands
for.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tabascal.components.ast_vis import GPVisAst, GPVisAstDFT
from tabascal.fft_gp import (
    latent_to_signal,
    latent_to_signal_dft,
    latent_to_signal_dft_init,
    latent_to_signal_init,
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


def _route(cls, config):
    comp = cls()
    comp.setup(config)
    forward = comp.build_forward()
    constants = make_constants(comp)
    state = dict(comp.state_outputs)

    def vis(params):
        return forward(params, state, constants)["vis_ast"]

    return comp, vis


@pytest.mark.parametrize("block_size", [None, 2])
def test_the_component_matches_the_fft_component(tmp_path, block_size):
    config = ast_config(tmp_path, baseline_block_size=block_size)
    ref_comp, ref_vis = _route(GPVisAst, config)
    dft_comp, dft_vis = _route(GPVisAstDFT, ast_config(tmp_path, baseline_block_size=block_size))

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


def test_the_gradients_match_the_fft_component(tmp_path):
    """Reverse mode too: the transpose of a matrix product is not the transpose
    of a pad, a shift and an inverse transform, even where the values agree."""
    config = ast_config(tmp_path)

    def value_and_grad(cls):
        comp, vis = _route(cls, ast_config(tmp_path))

        def loss(params):
            return jnp.sum(jnp.abs(vis(params)) ** 2)

        return jax.grad(loss)(comp.init_params_base)

    ref = value_and_grad(GPVisAst)
    got = value_and_grad(GPVisAstDFT)

    rtol, atol = _tols()
    for key in ref:
        scale = max(float(np.abs(np.asarray(ref[key])).max()), 1.0)
        np.testing.assert_allclose(
            np.asarray(got[key]), np.asarray(ref[key]), rtol=rtol, atol=atol * scale,
            err_msg=key,
        )


def test_the_dft_component_has_no_transform_in_its_compiled_program(tmp_path):
    """The point of the component: no inverse FFT, so no padded grid to hold.

    Read off the compiled program rather than timed. The FFT component is
    compiled beside it, so the check is that the two differ in this, not that
    some name happens to be absent.
    """
    fft_comp, fft_vis = _route(GPVisAst, ast_config(tmp_path))
    dft_comp, dft_vis = _route(GPVisAstDFT, ast_config(tmp_path))

    fft_text = jax.jit(fft_vis).lower(fft_comp.init_params_base).compile().as_text()
    dft_text = jax.jit(dft_vis).lower(dft_comp.init_params_base).compile().as_text()

    # ``fft_type`` marks the HLO operation itself; the bare word also appears
    # in the source-file metadata of both, this module's name among it.
    assert "fft_type" in fft_text
    assert "fft_type" not in dft_text


def test_auto_blocking_is_a_single_step(tmp_path):
    """``auto`` exists to bound the padded grid. This transform builds none,
    and the scan's own stack of outputs costs memory, so blocking it is worse
    on both counts and ``auto`` should not do it."""
    comp, _ = _route(GPVisAstDFT, ast_config(tmp_path, baseline_block_size="auto"))

    assert comp.baseline_block_size == comp.n_bl


def test_an_explicit_block_is_still_honoured(tmp_path):
    """The way to trade the room back if another part of a model wants it."""
    comp, vis = _route(GPVisAstDFT, ast_config(tmp_path, baseline_block_size=2))
    assert comp.baseline_block_size == 2

    single, single_vis = _route(GPVisAstDFT, ast_config(tmp_path, baseline_block_size=None))
    rtol, atol = _tols()
    np.testing.assert_allclose(
        np.asarray(vis(comp.init_params_base)),
        np.asarray(single_vis(single.init_params_base)),
        rtol=rtol, atol=atol,
    )
