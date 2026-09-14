"""A local sky scan must not gather all baselines in its forward or transpose."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tabascal.components import ast_vis
from tabascal.distributed import baseline_sharding, shard_pytree
from tests.test_base_config import base_args, make_ast_config
from .conftest import make_constants

pytestmark = pytest.mark.skipif(jax.device_count() != 4, reason="requires four devices")


@pytest.mark.parametrize("cls", [ast_vis.GPVisAst, ast_vis.GPVisAstDFT])
@pytest.mark.parametrize("block", [3, 7, None])
def test_sky_values_and_gradients_stay_local(tmp_path, monkeypatch, cls, block):
    # 28 baselines / 4 = 7 rows each: block=3 needs local tail padding, while
    # block=7 fits exactly. Both exercise the reshape that gathered at 512A.
    args = base_args(tmp_path)
    args["ast"].update(baseline_block_size=block, init="sample")
    comp = cls()
    comp.setup(make_ast_config(args, n_ant=8))
    constants = make_constants(comp)
    state = comp.state_outputs
    params = comp.init_params_base

    with monkeypatch.context() as patch:
        patch.setattr(ast_vis, "sharding_baselines", lambda: False)
        reference = comp.build_forward()
    monkeypatch.setenv("TABASCAL_SHARD_AXIS", "baseline")
    sharded = comp.build_forward()
    placed = [shard_pytree(tree, 1, comp.n_bl) for tree in (params, constants, state)]

    def loss(forward):
        def evaluate(p, c, s):
            vis = forward(p, s, c)["vis_ast"]
            return jnp.sum(jnp.abs(vis) ** 2), vis
        return jax.value_and_grad(evaluate, has_aux=True)

    want = jax.jit(loss(reference))(params, constants, state)
    compiled = jax.jit(loss(sharded)).lower(*placed).compile()
    got = compiled(*placed)
    tolerance = 2e-5 if params["ast_k_r_base"].dtype == jnp.float32 else 1e-10
    for actual, expected in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    assert got[0][1].sharding == baseline_sharding()
    assert all(x.sharding == baseline_sharding() for x in got[1].values())
    # A scalar loss reduction is expected; gathering the sky arrays is not.
    assert "all-gather" not in compiled.as_text().lower()
    assert "shard_map" in str(jax.make_jaxpr(sharded)(*placed[:1], placed[2], placed[1]))
