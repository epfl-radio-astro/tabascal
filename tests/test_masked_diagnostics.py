"""Masked reductions avoid dynamic index arrays and ignore excluded bad data."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tabascal.tab_tools import reduced_chi2, _truth_reductions


def test_integer_data_does_not_narrow_the_sample_count():
    true = jnp.zeros((200, 3, 5), dtype=jnp.int8)
    pred = jnp.ones_like(true)
    flags = jnp.zeros_like(true, dtype=bool)
    np.testing.assert_allclose(reduced_chi2(pred, true, jnp.asarray(1.), flags), 1.)


@pytest.mark.parametrize('complex_data', [False, True])
@pytest.mark.parametrize('all_flagged', [False, True])
def test_masked_moments(complex_data, all_flagged):
    rng = np.random.default_rng(4)
    true = rng.normal(size=(7, 3, 5))
    pred = true + rng.normal(size=true.shape)
    if complex_data:
        true = true + 1j * rng.normal(size=true.shape)
        pred = pred + 1j * rng.normal(size=true.shape)
    flags = rng.random(true.shape) < .3
    if all_flagged:
        flags[:] = True
    true[flags] = np.nan
    pred[flags] = np.inf
    noise = np.broadcast_to(np.array([1., 2., 4.])[None, :, None], true.shape).copy()
    noise[flags] = 0
    with np.errstate(invalid='ignore', divide='ignore'):
        diff = (pred - true)[~flags]
        count = diff.size
        want = np.sum((np.abs(diff) / noise[~flags])**2) / (count * (2 if complex_data else 1))
        moments = (np.sqrt(np.sum(np.abs(diff)**2) / count),
                   np.abs(np.sum(diff) / count),
                   np.sqrt(np.sum(np.abs(true[~flags])**2) / count))
    arrays = tuple(map(jnp.asarray, (pred, true, noise, flags)))
    np.testing.assert_allclose(reduced_chi2(*arrays), want, rtol=2e-6, equal_nan=True)
    got = _truth_reductions(arrays[0], arrays[1], arrays[3])
    np.testing.assert_allclose(got, moments, rtol=2e-6, equal_nan=True)
    # Dynamic flags must lower directly; boolean compaction cannot do so.
    text = str(jax.make_jaxpr(reduced_chi2)(*arrays))
    assert 'nonzero' not in text
