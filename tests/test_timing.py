"""Tests for timing utility functions."""

import jax.numpy as jnp
import pytest
from tabascal.timing import (
    clear_timings,
    disable_timings,
    enable_timings,
    get_timings,
    measure_runtime,
    timer,
)


@pytest.fixture(autouse=True)
def manage_timings():
    """Fixture to ensure a clean timing state for each test."""
    clear_timings()
    disable_timings()
    yield
    disable_timings()
    clear_timings()


def test_timing_collection_enabled():
    """Test that timings are collected when enabled."""
    enable_timings()

    @measure_runtime
    def add(x, y):
        return x + y

    a = jnp.array([1, 2, 3])
    b = jnp.array([4, 5, 6])
    result = add(a, b)

    timings = get_timings()
    func_name = "test_timing_collection_enabled.<locals>.add"
    assert func_name in timings
    assert len(timings[func_name].timings) == 1
    assert jnp.allclose(result, jnp.array([5, 7, 9]))


def test_timing_collection_disabled():
    """Test that timings are NOT collected when disabled."""
    disable_timings()

    @measure_runtime
    def add(x, y):
        return x + y

    a = jnp.array([1, 2, 3])
    b = jnp.array([4, 5, 6])
    result = add(a, b)

    timings = get_timings()
    assert "add" not in timings
    assert jnp.allclose(result, jnp.array([5, 7, 9]))


def test_hierarchical_timing():
    """Test that hierarchical timings are correctly captured."""
    enable_timings()

    @measure_runtime
    def child(x):
        return x * 2

    @measure_runtime
    def parent(x):
        return child(x) + child(x)

    result = parent(jnp.array([1, 2]))

    timings = get_timings()
    parent_name = "test_hierarchical_timing.<locals>.parent"
    child_name = "test_hierarchical_timing.<locals>.child"
    assert parent_name in timings
    assert child_name in timings[parent_name].children
    assert len(timings[parent_name].timings) == 1
    assert len(timings[parent_name].children[child_name].timings) == 2
    assert jnp.allclose(result, jnp.array([4, 8]))


def test_timer_context_manager():
    """Test the manual timer context manager."""
    enable_timings()

    with timer("manual_block"):
        x = jnp.arange(10).sum()

    timings = get_timings()
    assert "manual_block" in timings
    assert len(timings["manual_block"].timings) == 1
    assert int(x) == 45


def test_measure_runtime_data_structures():
    """Test that measure_runtime works with various data structures."""
    enable_timings()

    @measure_runtime
    def process_data(data):
        return {
            "list": [x * 2 for x in data["list"]],
            "tuple": (data["tuple"][0] + 1,),
            "dict": {"inner": data["dict"]["inner"] / 2},
        }

    inputs = {
        "list": [jnp.array([1, 2])],
        "tuple": (jnp.array([10, 20]),),
        "dict": {"inner": jnp.array([4, 8])},
    }

    results = process_data(inputs)
    timings = get_timings()

    func_name = "test_measure_runtime_data_structures.<locals>.process_data"
    assert func_name in timings
    assert jnp.allclose(results["list"][0], jnp.array([2, 4]))
    assert jnp.allclose(results["tuple"][0], jnp.array([11, 21]))
    assert jnp.allclose(results["dict"]["inner"], jnp.array([2, 4]))


def test_measure_runtime_mixed_types():
    """Test the decorator with mixed JAX and non-JAX types."""
    enable_timings()

    @measure_runtime
    def mixed_function(arr, scalar, text):
        return arr * scalar, text.upper()

    result_arr, result_text = mixed_function(jnp.array([1, 2, 3]), 2, "hello")

    timings = get_timings()
    func_name = "test_measure_runtime_mixed_types.<locals>.mixed_function"
    assert func_name in timings
    assert jnp.allclose(result_arr, jnp.array([2, 4, 6]))
    assert result_text == "HELLO"
