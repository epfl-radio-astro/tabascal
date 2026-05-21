"""Shared helpers for component-level tests."""

import jax
import jax.numpy as jnp


def make_constants(comp):
    return {f"{comp.prefix}/{k}": v for k, v in comp.build_constants().items()}


def assert_transform_roundtrip(comp, base, L, mu, atol=1e-6):
    """Check both directions of a forward/inverse affine-Cholesky transform."""
    transformed = comp.forward_transform(base, L, mu)
    assert jnp.allclose(comp.inv_transform(transformed, L, mu), base, atol=atol)
    base2 = comp.inv_transform(base, L, mu)
    assert jnp.allclose(comp.forward_transform(base2, L, mu), base, atol=atol)


class _ReadTracingDict(dict):
    """A ``dict`` that records keys fetched via ``__getitem__``.

    A forward's ``{**state, "k": v}`` splat (and ``dict(state)``) use the
    C-level fast path for ``dict`` subclasses and do NOT call ``__getitem__``,
    so only an explicit ``state["k"]`` registers as a read. This lets us
    recover what a component's forward actually consumes from the state.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read_keys = set()

    def __getitem__(self, key):
        self.read_keys.add(key)
        return super().__getitem__(key)


def trace_forward_io(comp, params, state, constants):
    """Run ``comp``'s forward and classify its actual state I/O at runtime.

    Returns a dict with ``reads`` (consumed but not produced), ``writes``
    (produced without depending on a prior value), ``accumulates`` (read then
    re-written), and ``out`` (the resulting state).
    """
    snapshot = dict(state)
    tracer = _ReadTracingDict(state)
    out = comp.build_forward()(params, tracer, constants)
    reads = set(tracer.read_keys)
    produced = {k for k in out if k not in snapshot or out[k] is not snapshot[k]}
    return {
        "reads": reads - produced,
        "writes": produced - reads,
        "accumulates": produced & reads,
        "out": out,
    }


def assert_declared_io(comp, params, state, constants, dims):
    """Assert a component's declared reads/writes/accumulates (keys *and* shapes)
    match what its forward does at runtime.

    ``dims`` maps every symbolic dimension name used in the component's declared
    shapes to a concrete int for this invocation.
    """
    name = type(comp).__name__
    traced = trace_forward_io(comp, params, state, constants)

    for kind in ("reads", "writes", "accumulates"):
        declared = set(getattr(comp, kind))
        assert traced[kind] == declared, (
            f"{name}: declared {kind} {declared} != runtime {traced[kind]}"
        )

    def resolve(shape):
        return tuple(dims[d] if isinstance(d, str) else d for d in shape)

    out = traced["out"]
    for key, shape in comp.reads.items():
        assert state[key].shape == resolve(shape), (
            f"{name}: read '{key}' runtime shape {state[key].shape} "
            f"!= declared {resolve(shape)}"
        )
    for key, shape in {**comp.writes, **comp.accumulates}.items():
        assert out[key].shape == resolve(shape), (
            f"{name}: output '{key}' runtime shape {out[key].shape} "
            f"!= declared {resolve(shape)}"
        )
