"""The paired diagnostic preserves both existing reductions with one model call."""
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from tabascal.tab_tools import nlog_like, nlog_post, nlog_like_and_post


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("scale", [1., 2.])
def test_paired_diagnostics(masked, scale):
    calls = []
    def model(obs_data, state=None, constants=None):
        calls.append(1)
        loc = numpyro.sample("loc", dist.Normal(0, 1))
        mask = jnp.array([True, False, True]) if masked else True
        # Flags go through the mask handler, as components/likelihood.py applies them.
        with numpyro.handlers.scale(scale=scale), numpyro.handlers.mask(mask=mask):
            numpyro.sample("obs", dist.Normal(loc + state, constants), obs=obs_data)
    params = {"loc": jnp.array(0.4)}
    data = jnp.array([1., 2., 3.])
    kwargs = dict(state=0.2, constants=1.3)
    expected = (nlog_like(model, params, data, **kwargs), nlog_post(model, params, data, **kwargs))
    calls.clear()
    actual = nlog_like_and_post(model, params, data, **kwargs)
    assert len(calls) == 1
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
