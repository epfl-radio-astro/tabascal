"""Timing and profiling utilities for tabascal."""

import functools
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

# Type variable for generic function signatures
F = TypeVar("F", bound=Callable[..., Any])

@dataclass
class TimingNode:
    """A node in the hierarchical timing tree."""

    timings: List[float] = field(default_factory=list)
    children: Dict[str, "TimingNode"] = field(default_factory=dict)

    @property
    def total_time(self) -> float:
        """Total time spent in this node across all calls."""
        return sum(self.timings) if self.timings else 0.0


class TimingManager:
    """Manages the hierarchical timing state and collection."""

    def __init__(self):
        self.enabled = False
        self.roots: Dict[str, TimingNode] = {}
        self.current_node: Optional[TimingNode] = None
        self._node_stack: List[Optional[TimingNode]] = []

    def enable(self):
        """Enable timing collection."""
        self.enabled = True

    def disable(self):
        """Disable timing collection."""
        self.enabled = False

    def clear(self):
        """Clear all collected timing data."""
        self.roots.clear()
        self.current_node = None
        self._node_stack = []

    @contextmanager
    def scope(self, name: str):
        """Context manager for a named timing scope."""
        if not self.enabled:
            yield
            return

        # Skip if in a JIT context (tracer present)
        if isinstance(jnp.array(0) + 1, jax.core.Tracer):
            print(f"WARNING: Timing used in JIT context with name '{name}'. Skipping timing.")
            yield
            return

        # Initialize or get the appropriate node
        if self.current_node is None:
            node = self.roots.setdefault(name, TimingNode())
        else:
            node = self.current_node.children.setdefault(name, TimingNode())

        # Push current node to stack and descend
        self._node_stack.append(self.current_node)
        self.current_node = node

        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.current_node.timings.append(elapsed)
            # Ascend to previous node
            self.current_node = self._node_stack.pop()


# Global singleton manager
_MANAGER = TimingManager()


def enable_timings():
    """Enable global timing collection for tabascal."""
    _MANAGER.enable()


def disable_timings():
    """Disable global timing collection for tabascal."""
    _MANAGER.disable()


def clear_timings():
    """Clear all collected timing data."""
    _MANAGER.clear()


def get_timings() -> Dict[str, TimingNode]:
    """Return a copy of the collected timing roots."""
    return _MANAGER.roots.copy()


def _block_until_ready(obj: Any) -> None:
    """
    Synchronize JAX computations on the given object.
    
    Uses JAX's block_until_ready which handles Pytrees automatically.
    """
    try:
        jax.block_until_ready(obj)
    except (AttributeError, TypeError):
        # Fallback if the object doesn't support synchronization
        pass


def measure_runtime(func: F) -> F:
    """
    Decorator for measuring function runtime with JAX synchronization.
    
    Ensures accurate timing by blocking on JAX arrays in inputs and outputs.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not _MANAGER.enabled:
            return func(*args, **kwargs)

        # Skip if in a JIT context
        if isinstance(jnp.array(0) + 1, jax.core.Tracer):
            print(f"WARNING: Timing used in JIT context with function name '{func.__qualname__}'. Skipping timing.")
            return func(*args, **kwargs)

        # Synchronize inputs
        _block_until_ready(args)
        _block_until_ready(kwargs)

        with _MANAGER.scope(func.__qualname__):
            result = func(*args, **kwargs)
            # Synchronize output
            _block_until_ready(result)
            return result

    return wrapper


@contextmanager
def timer(name: str):
    """
    Context manager for manual timing of a block of code.
    
    Example:
        with timer("data_loading"):
            data = load_dataset()
    """
    with _MANAGER.scope(name):
        yield


def _convert_time_to_str(t: float) -> str:
    """Convert time in seconds to a human-readable string with SI units."""
    if t < 1e-6:
        return f"{t * 1e9:.2f} ns"
    if t < 1e-3:
        return f"{t * 1e6:.2f} us"
    if t < 1:
        return f"{t * 1e3:.2f} ms"
    if t < 1e3:
        return f"{t:.2f} s "
    if t < 1e6:
        return f"{t / 1e3:.2f} ks"
    return f"{t / 1e6:.2f} Ms"


def print_timings():
    """Print a summary of collected timings in a hierarchical tree format."""
    roots = _MANAGER.roots
    if not roots:
        print("No timings collected. Use enable_timings() to start collecting.")
        return

    # Table configuration
    COL_WIDTHS = {
        "name": 50,
        "calls": 8,
        "metric": 12,
        "rel": 10,
        "glob": 10,
    }
    # Total width including padding spaces
    TOTAL_WIDTH = (
        COL_WIDTHS["name"]
        + COL_WIDTHS["calls"]
        + COL_WIDTHS["rel"]
        + COL_WIDTHS["glob"]
        + 3 * COL_WIDTHS["metric"]
        + 8
    )

    print("\nRuntime Statistics:")
    print("=" * TOTAL_WIDTH)

    # Print table header
    header = (
        f"{'Function':<{COL_WIDTHS['name']}} "
        f"{'Calls':>{COL_WIDTHS['calls']}} "
        f"{'Total':>{COL_WIDTHS['metric']}} "
        f"{'Glob (%)':>{COL_WIDTHS['glob']}} "
        f"{'Rel (%)':>{COL_WIDTHS['rel']}} "
        f"{'Mean':>{COL_WIDTHS['metric']}} "
        f"{'Std':>{COL_WIDTHS['metric']}} "
    )
    print(header)
    print("-" * TOTAL_WIDTH)

    def _print_node(
        name: str,
        node: TimingNode,
        depth: int = 0,
        parent_total: float = 0.0,
        root_total: float = 0.0,
    ):
        """Recursively print a timing node and its children."""
        if not node.timings:
            return

        times = np.array(node.timings)
        total = times.sum()

        # Calculate percentages
        rel_pct = (total / parent_total * 100) if parent_total > 0 else 100.0
        glob_pct = (total / root_total * 100) if root_total > 0 else 100.0

        indent = "  " * depth
        func_display = f"{indent}{name}"

        row = (
            f"{func_display:<{COL_WIDTHS['name']}} "
            f"{len(times):>{COL_WIDTHS['calls']}} "
            f"{_convert_time_to_str(total):>{COL_WIDTHS['metric']}} "
            f"{glob_pct:>9.1f}% "
            f"{rel_pct:>9.1f}% "
            f"{_convert_time_to_str(times.mean()):>{COL_WIDTHS['metric']}} "
            f"{_convert_time_to_str(times.std()):>{COL_WIDTHS['metric']}} "
        )
        print(row)

        # Print children recursively
        for child_name, child_node in sorted(node.children.items()):
            _print_node(
                child_name,
                child_node,
                depth + 1,
                parent_total=total,
                root_total=root_total,
            )

    # Print each top-level function and its tree
    for i, (func_name, node) in enumerate(sorted(roots.items())):
        if i > 0:
            print()
        _print_node(
            func_name,
            node,
            depth=0,
            parent_total=node.total_time,
            root_total=node.total_time,
        )

    print("=" * TOTAL_WIDTH)


__all__ = [
    "enable_timings",
    "disable_timings",
    "clear_timings",
    "get_timings",
    "measure_runtime",
    "timer",
    "print_timings",
    "TimingNode",
]
