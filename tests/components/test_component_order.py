"""The ``model.components`` list is checked for dependency order at assembly.

Each component declares the state keys it reads (``required_inputs``) and the
keys it writes (``output_shapes``, realised as ``state_outputs`` once it has been
set up). Those declarations used to be decorative: ``Model`` never checked them,
so a list missing a component -- or holding the right components in the wrong
order -- only failed once the forward pass was traced, as a ``KeyError`` on a
state key raised from inside JAX with no mention of which component wanted it.

These tests pin the assembly-time check instead: the error names the component,
the missing key, and what produces it, and says whether that producer is absent
or merely listed too late.
"""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tabascal.components import (
    ComponentOrderError,
    in_tree_components,
    state_key_producers,
    validate_component_order,
)
from tabascal.config import BASE_STATE_KEYS
from tabascal.imports import import_components

REPO_ROOT = Path(__file__).resolve().parents[2]


def build(refs):
    """The components of a config list, constructed but not set up.

    ``Model`` builds them exactly like this before calling ``setup``; the order
    check runs on the declarations alone, so it needs no observation.
    """
    return [C() for C in import_components(refs)]


def check(refs):
    validate_component_order(build(refs), BASE_STATE_KEYS)


FULL_RFI = [
    "trajectory:FixedOrbit",
    "rfi_signal:ComplexRFIVarAnt",
    "rfi_vis:RiemannVis",
]


class TestMissingProducer:
    """A component whose input nothing in the list produces."""

    def test_discrete_sky_vis_without_fixed_discrete_sky(self):
        """The motivating case: the sky reader left out of the list.

        ``DiscreteSkyVis`` reads the catalogue ``FixedDiscreteSky`` puts in the
        state. Without it the run used to die tracing the forward pass on
        ``state["ast_radec"]``.
        """
        with pytest.raises(ComponentOrderError) as excinfo:
            check(FULL_RFI + ["ast_vis:DiscreteSkyVis", "gains:UnitaryGains"])

        message = str(excinfo.value)
        assert "DiscreteSkyVis" in message
        assert "ast_radec" in message
        # Names the component that would supply it, and that it has to be added.
        assert "ast_signal:FixedDiscreteSky" in message
        assert "add one before" in message
        assert "wrong order" not in message

    def test_rfi_vis_without_an_rfi_signal_component(self):
        with pytest.raises(ComponentOrderError) as excinfo:
            check(
                [
                    "trajectory:FixedOrbit",
                    "rfi_vis:RiemannVis",
                    "ast_vis:GPVisAst",
                    "gains:UnitaryGains",
                ]
            )

        message = str(excinfo.value)
        assert "RiemannVis" in message
        assert "rfi_A" in message
        # Every in-tree producer of the key is offered, not just one.
        assert "rfi_signal:ComplexRFIVarAnt" in message
        assert "rfi_signal:ComplexRFIConstAnt" in message

    def test_phase_calculation_without_a_trajectory(self):
        with pytest.raises(ComponentOrderError) as excinfo:
            check(
                [
                    "trajectory:PhaseCalculationRFI",
                    "rfi_signal:ComplexRFIVarAnt",
                    "rfi_vis:RiemannVis",
                    "ast_vis:GPVisAst",
                    "gains:UnitaryGains",
                ]
            )

        message = str(excinfo.value)
        assert "PhaseCalculationRFI" in message
        assert "rfi_xyz" in message
        assert "trajectory:FixedOrbit" in message


class TestWrongOrder:
    """The right components, listed in the wrong order."""

    def test_discrete_sky_components_reversed(self):
        with pytest.raises(ComponentOrderError) as excinfo:
            check(
                FULL_RFI
                + [
                    "ast_vis:DiscreteSkyVis",
                    "ast_signal:FixedDiscreteSky",
                    "gains:UnitaryGains",
                ]
            )

        message = str(excinfo.value)
        assert "DiscreteSkyVis" in message
        assert "ast_radec" in message
        assert "ast_signal:FixedDiscreteSky" in message
        # The distinguishing part: the producer is present, just too late.
        assert "listed after" in message
        assert "wrong order" in message
        assert "add one before" not in message

    def test_trajectory_after_the_phase_calculation(self):
        with pytest.raises(ComponentOrderError) as excinfo:
            check(
                [
                    "trajectory:PhaseCalculationRFI",
                    "trajectory:FixedOrbit",
                    "rfi_signal:ComplexRFIVarAnt",
                    "rfi_vis:RiemannVis",
                    "ast_vis:GPVisAst",
                    "gains:UnitaryGains",
                ]
            )

        message = str(excinfo.value)
        assert "PhaseCalculationRFI" in message
        assert "rfi_xyz" in message
        assert "listed after" in message


class TestValidLists:
    """Working combinations must keep working -- no false positives."""

    @pytest.mark.parametrize(
        "refs",
        [
            pytest.param(
                FULL_RFI + ["ast_vis:GPVisAst", "gains:UnitaryGains"], id="default"
            ),
            pytest.param(
                [
                    "trajectory:FixedOrbit",
                    "trajectory:PhaseCalculationRFI",
                    "rfi_signal:ComplexRFIVarAnt",
                    "rfi_vis:RiemannVisFFI",
                    "ast_vis:GPVisAst",
                    "gains:GPGains",
                ],
                id="fitted-phase",
            ),
            pytest.param(
                [
                    "trajectory:Orbit",
                    "trajectory:PhaseCalculationRFI",
                    "rfi_signal:ComplexRFIConstAnt",
                    "rfi_vis:RiemannVisVariable",
                    "ast_signal:FixedDiscreteSky",
                    "ast_vis:DiscreteSkyVis",
                    "gains:ConstGains",
                ],
                id="const-gains",
            ),
            pytest.param(
                [
                    "trajectory:NoDragOrbit",
                    "trajectory:PhaseCalculationRFI",
                    "rfi_signal:ComplexRFIVarAnt",
                    "rfi_vis:RiemannVisVariableFFI",
                    "ast_signal:FixedDiscreteSky",
                    "ast_vis:GPVisAst",
                    "ast_vis:DiscreteSkyVis",
                    "gains:UnitaryGains",
                ],
                id="gp-plus-fixed-sky",
            ),
        ],
    )
    def test_documented_combinations_assemble(self, refs):
        check(refs)

    def test_the_accumulators_come_from_the_model(self):
        """``vis_ast``/``vis_rfi`` are zeroed by ``Model``, not by a component.

        A visibility component may therefore be the first thing in the list and
        still find the accumulator it adds into.
        """
        check(["ast_vis:GPVisAst"])


def _config_component_lists():
    """Every ``model.components`` list shipped in the repo, as (label, refs)."""
    found = []
    for directory in ("examples", "tests/data", "ci/reframe/data"):
        for path in sorted((REPO_ROOT / directory).glob("*.yaml")):
            config = yaml.safe_load(path.read_text()) or {}
            refs = (config.get("model") or {}).get("components")
            if refs:
                found.append((str(path.relative_to(REPO_ROOT)), refs))
    return found


def _doc_component_lists():
    """The same, from the fenced YAML blocks of the documentation."""
    found = []
    for path in sorted((REPO_ROOT / "docs").glob("*.md")):
        blocks = re.findall(r"^```yaml\n(.*?)^```", path.read_text(), re.M | re.S)
        for i, block in enumerate(blocks):
            try:
                config = yaml.safe_load(block) or {}
            except yaml.YAMLError:
                continue
            if not isinstance(config, dict):
                continue
            refs = (config.get("model") or {}).get("components")
            if refs:
                found.append((f"{path.relative_to(REPO_ROOT)}#{i}", refs))
    return found


ALL_SHIPPED_LISTS = _config_component_lists() + _doc_component_lists()


def _is_whole_model(refs):
    """Whether a list is a whole model rather than an illustrative fragment.

    A whole model has to produce ``vis_obs``, the visibility ``Model`` hands to
    the likelihood, so a list with a gains component is one and a snippet showing
    a single component in isolation is not. Only a whole model can be expected to
    satisfy every component's inputs.
    """
    known = in_tree_components()
    return any(
        "vis_obs" in getattr(known.get(str(ref).replace(".", ":")), "output_shapes", {})
        for ref in refs
    )


WHOLE_MODEL_LISTS = [
    (label, refs) for label, refs in ALL_SHIPPED_LISTS if _is_whole_model(refs)
]


def test_the_shipped_lists_were_found():
    """Guard the collection above: a silent zero would make the next tests vacuous."""
    assert len(_config_component_lists()) >= 3
    assert len(_doc_component_lists()) >= 2
    assert len(WHOLE_MODEL_LISTS) >= 4


@pytest.mark.parametrize(
    "refs", [pytest.param(refs, id=label) for label, refs in ALL_SHIPPED_LISTS]
)
def test_every_shipped_component_reference_resolves(refs):
    """Every component named in a shipped config or doc is a real class.

    Fragments included: a snippet whose reference cannot be imported is as broken
    as a whole model whose components are in the wrong order, and the YAML has to
    be written ``module:Class`` with no space, or it parses as a mapping and never
    reaches the importer as a reference at all.
    """
    build(refs)


@pytest.mark.parametrize(
    "refs", [pytest.param(refs, id=label) for label, refs in WHOLE_MODEL_LISTS]
)
def test_every_shipped_model_assembles(refs):
    """Every example config and documented whole model in the repo passes the check."""
    check(refs)


def test_model_makes_the_check_before_it_sets_anything_up():
    """The wiring itself: ``Model`` rejects the list, not just the helper.

    The check runs before the components are set up, so the config is never
    touched -- which is the point of doing it there: nothing has fetched a TLE or
    built a GP by the time the list is refused. A stub config with only the two
    attributes ``Model`` reads first is therefore enough, and its emptiness is
    what proves the ordering came first.
    """
    from tabascal.config import Model

    with pytest.raises(ComponentOrderError, match="ast_radec"):
        Model(
            SimpleNamespace(noise=1.0, n_rfi=0),
            FULL_RFI + ["ast_vis:DiscreteSkyVis", "gains:UnitaryGains"],
        )


def test_every_required_input_has_an_in_tree_producer():
    """No component may declare an input nothing can supply.

    The check is only as good as the declarations, so this pins the other side of
    them: a typo in a ``required_inputs`` key, or an ``output_shapes`` that has
    drifted from what the component actually writes, shows up here rather than as
    a config that can never be made to validate.
    """
    producers = state_key_producers()
    for ref, cls in in_tree_components().items():
        for key in cls.required_inputs:
            assert key in producers or key in BASE_STATE_KEYS, (
                f"{ref} requires '{key}', which no in-tree component produces "
                "and which Model does not put in the state."
            )
