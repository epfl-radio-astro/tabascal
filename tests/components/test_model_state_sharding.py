"""Retained component placeholders must follow the model's baseline layout."""
import jax
import numpy as np
import pytest

from tabascal.config import Model
from tabascal.distributed import baseline_sharding, replicated_sharding
from tests.test_base_config import base_args, make_ast_config

pytestmark = pytest.mark.skipif(jax.device_count() != 4, reason='requires four devices')


@pytest.mark.parametrize('n_ant', [8, 6])
def test_component_visibility_placeholders_are_placed(tmp_path, monkeypatch, n_ant):
    monkeypatch.setenv('TABASCAL_SHARD_AXIS', 'baseline')
    args = base_args(tmp_path)
    args['ast']['init'] = 'sample'
    cfg = make_ast_config(args, n_ant=n_ant)
    cfg.n_rfi, cfg.noise = 0, 1.
    model = Model(cfg, ['ast_vis:GPVisAst', 'gains:UnitaryGains'])
    wanted = baseline_sharding() if cfg.n_bl % 4 == 0 else replicated_sharding()
    for comp in model.components:
        for key, array in comp.state_outputs.items():
            if key.startswith('vis_'):
                assert array.sharding == wanted
                np.testing.assert_array_equal(array, 0)
    for key in ('vis_ast', 'vis_rfi', 'vis_obs'):
        assert model.state[key].sharding == wanted
    # Assembly shares the component's observation placeholder; there is no
    # second full observation cube kept only by the component object.
    assert model.state['vis_obs'] is model.components[-1].state_outputs['vis_obs']
