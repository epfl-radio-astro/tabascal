"""Unit tests for the config-time component dependency resolver
(``tabascal.config.validate_component_dependencies``).

These exercise the resolver in isolation with lightweight stub components, so
no MeasurementSet, JAX arrays, or component setup is required.
"""

import pytest

from tabascal.config import (
    _MODEL_SEED_SHAPES,
    validate_component_dependencies,
)


def comp(name, reads=None, writes=None, accumulates=None):
    """A minimal stub component carrying only the resolver's I/O declarations.

    ``type(name, ...)`` gives each stub a distinct ``__name__`` so error
    messages (which the resolver builds from ``type(comp).__name__``) are
    realistic.
    """
    return type(
        name,
        (),
        {
            "reads": reads or {},
            "writes": writes or {},
            "accumulates": accumulates or {},
        },
    )()


# Concrete dims so symbolic shapes resolve; n_freq == n_freq_fine here unless a
# test overrides it (mirrors the n_int_freq == 1 case).
DIMS = {
    "n_rfi": 5, "n_ant": 4, "n_bl": 6,
    "n_freq": 7, "n_freq_fine": 7,
    "n_time": 8, "n_time_fine": 16,
}

VIS = ("n_bl", "n_freq", "n_time")
RFI4 = ("n_rfi", "n_ant", "n_freq_fine", "n_time_fine")


def valid_stack():
    """A correctly ordered RFI + astronomy + gains stack (mirrors the e2e cases)."""
    return [
        comp("Orbit", writes={"rfi_xyz": ("n_rfi", "n_time_fine", 3),
                              "rfi_phase": RFI4}),
        comp("PhaseCalc", reads={"rfi_xyz": ("n_rfi", "n_time_fine", 3)},
             writes={"rfi_phase": RFI4}),
        comp("RFISignal", writes={"rfi_A": RFI4}),
        comp("RFIVis", reads={"rfi_phase": RFI4, "rfi_A": RFI4},
             accumulates={"vis_rfi": VIS}),
        comp("Sky", writes={"ast_radec": ("n_src", 2), "ast_I": ("n_src", "n_freq")}),
        comp("AstVis", reads={"ast_radec": ("n_src", 2), "ast_I": ("n_src", "n_freq")},
             accumulates={"vis_ast": VIS}),
        comp("Gains", reads={"vis_rfi": VIS, "vis_ast": VIS},
             writes={"vis_obs": VIS, "gains": ("n_ant", "n_freq", "n_time")}),
    ]


def test_valid_stack_passes():
    validate_component_dependencies(valid_stack(), _MODEL_SEED_SHAPES, DIMS)


def test_seeded_accumulators_interleave_freely():
    """Two vis_ast accumulators with no explicit writer pass — vis_ast is seeded."""
    stack = [
        comp("AstA", accumulates={"vis_ast": VIS}),
        comp("AstB", accumulates={"vis_ast": VIS}),
        comp("Gains", reads={"vis_rfi": VIS, "vis_ast": VIS},
             writes={"vis_obs": VIS}),
    ]
    validate_component_dependencies(stack, _MODEL_SEED_SHAPES, DIMS)


def test_consumer_before_producer_is_ordering_error():
    stack = [
        comp("AstVis", reads={"ast_radec": ("n_src", 2)}, accumulates={"vis_ast": VIS}),
        comp("Sky", writes={"ast_radec": ("n_src", 2)}),
    ]
    with pytest.raises(ValueError, match="ordering error") as e:
        validate_component_dependencies(stack, _MODEL_SEED_SHAPES, DIMS)
    assert "ast_radec" in str(e.value)
    assert "Sky" in str(e.value)


def test_missing_producer_is_unresolved_dependency():
    stack = [
        comp("AstVis", reads={"ast_radec": ("n_src", 2)}, accumulates={"vis_ast": VIS}),
    ]
    with pytest.raises(ValueError, match="Unresolved dependency") as e:
        validate_component_dependencies(stack, _MODEL_SEED_SHAPES, DIMS)
    assert "ast_radec" in str(e.value)


def test_accumulate_into_unseeded_unproduced_key_errors():
    stack = [comp("Weird", accumulates={"mystery": VIS})]
    with pytest.raises(ValueError, match="Unresolved dependency") as e:
        validate_component_dependencies(stack, _MODEL_SEED_SHAPES, DIMS)
    assert "accumulates into" in str(e.value)


def test_shape_mismatch_errors():
    stack = [
        comp("Sky", writes={"ast_I": ("n_src", "n_freq")}),
        comp("AstVis", reads={"ast_I": ("n_src", "n_time")},  # wrong dim
             accumulates={"vis_ast": VIS}),
    ]
    with pytest.raises(ValueError, match="Shape mismatch") as e:
        validate_component_dependencies(stack, _MODEL_SEED_SHAPES, DIMS)
    assert "ast_I" in str(e.value)


def test_concrete_resolution_allows_equal_dims():
    """n_freq vs n_freq_fine differ symbolically but are equal concretely when
    n_int_freq == 1, so the producer/consumer edge is compatible."""
    stack = [
        comp("RFISignalCoarse", writes={"rfi_A": ("n_rfi", "n_ant", "n_freq", "n_time_fine")}),
        comp("RFIVisFine", reads={"rfi_A": RFI4}, accumulates={"vis_rfi": VIS}),
    ]
    dims_equal = {**DIMS, "n_freq": 7, "n_freq_fine": 7}
    validate_component_dependencies(stack, _MODEL_SEED_SHAPES, dims_equal)


def test_concrete_resolution_flags_unequal_dims():
    """The same edge is a genuine mismatch when n_freq != n_freq_fine
    (n_int_freq > 1)."""
    stack = [
        comp("RFISignalCoarse", writes={"rfi_A": ("n_rfi", "n_ant", "n_freq", "n_time_fine")}),
        comp("RFIVisFine", reads={"rfi_A": RFI4}, accumulates={"vis_rfi": VIS}),
    ]
    dims_unequal = {**DIMS, "n_freq": 7, "n_freq_fine": 14}
    with pytest.raises(ValueError, match="Shape mismatch"):
        validate_component_dependencies(stack, _MODEL_SEED_SHAPES, dims_unequal)
