"""Initialization must bound padded FFTs as well as the forward transform."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tabascal.components import ast_vis
from tests.test_base_config import base_args, make_ast_config


@pytest.mark.parametrize('cls', [ast_vis.GPVisAst, ast_vis.GPVisAstDFT])
@pytest.mark.parametrize('budget', [1, 20000, 2**30])
def test_blocked_encoder_matches_full_vmap(tmp_path, monkeypatch, cls, budget):
    monkeypatch.setattr(cls, '_BLOCK_BUDGET_BYTES', budget)
    args = base_args(tmp_path)
    args['ast'].update(init='sample')
    comp = cls()
    comp.setup(make_ast_config(args, n_ant=8))
    shape = (comp.n_bl, comp.n_freq, comp.n_time)
    rng = np.random.default_rng(31)
    vis = jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape))
    expected = jax.vmap(ast_vis.signal_to_latent, (0, None, None))(
        vis, comp.pad_factors, comp.latent_idxs)
    actual = comp.signal_to_latent(vis)
    tol = 2e-5 if vis.dtype == jnp.complex64 else 1e-10
    np.testing.assert_allclose(actual, expected, atol=tol, rtol=tol)
