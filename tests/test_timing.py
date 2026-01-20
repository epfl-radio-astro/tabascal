"""Tests for timing utility functions."""

import jax.numpy as jnp
import pytest

from tabascal.timing import measure_runtime


def test_block_until_ready_simple_array():
    """Test the decorator with a simple JAX array."""
    @measure_runtime
    def add(x, y):
        return x + y
    
    a = jnp.array([1, 2, 3])
    b = jnp.array([4, 5, 6])
    result = add(a, b)
    
    assert jnp.allclose(result, jnp.array([5, 7, 9]))


def test_block_until_ready_list():
    """Test the decorator with lists of JAX arrays."""
    @measure_runtime
    def sum_list(arrays):
        return [x * 2 for x in arrays]
    
    inputs = [jnp.array([1, 2]), jnp.array([3, 4])]
    results = sum_list(inputs)
    
    assert len(results) == 2
    assert jnp.allclose(results[0], jnp.array([2, 4]))
    assert jnp.allclose(results[1], jnp.array([6, 8]))


def test_block_until_ready_tuple():
    """Test the decorator with tuples of JAX arrays."""
    @measure_runtime
    def process_tuple(data):
        a, b = data
        return (a * 2, b + 1)
    
    inputs = (jnp.array([1, 2]), jnp.array([3, 4]))
    result_a, result_b = process_tuple(inputs)
    
    assert jnp.allclose(result_a, jnp.array([2, 4]))
    assert jnp.allclose(result_b, jnp.array([4, 5]))


def test_block_until_ready_dict():
    """Test the decorator with dictionaries of JAX arrays."""
    @measure_runtime
    def process_dict(data):
        return {
            'doubled': data['x'] * 2,
            'incremented': data['y'] + 1
        }
    
    inputs = {
        'x': jnp.array([1, 2, 3]),
        'y': jnp.array([4, 5, 6])
    }
    results = process_dict(inputs)
    
    assert jnp.allclose(results['doubled'], jnp.array([2, 4, 6]))
    assert jnp.allclose(results['incremented'], jnp.array([5, 6, 7]))


def test_block_until_ready_nested():
    """Test the decorator with nested structures."""
    @measure_runtime
    def nested_operation(data):
        return {
            'arrays': [data['arrays'][0] * 2, data['arrays'][1] + 1],
            'scalar': data['scalar']
        }
    
    inputs = {
        'arrays': [jnp.array([1, 2]), jnp.array([3, 4])],
        'scalar': 42
    }
    results = nested_operation(inputs)
    
    assert len(results['arrays']) == 2
    assert jnp.allclose(results['arrays'][0], jnp.array([2, 4]))
    assert jnp.allclose(results['arrays'][1], jnp.array([4, 5]))
    assert results['scalar'] == 42


def test_block_until_ready_mixed_types():
    """Test the decorator with mixed types (arrays and non-arrays)."""
    @measure_runtime
    def mixed_function(arr, scalar, text):
        return arr * scalar, text.upper()
    
    result_arr, result_text = mixed_function(jnp.array([1, 2, 3]), 2, "hello")
    
    assert jnp.allclose(result_arr, jnp.array([2, 4, 6]))
    assert result_text == "HELLO"
