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

import importlib
import pkgutil
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from tabascal.components import (
    Component,
    ComponentOrderError,
    in_tree_components,
    is_class,
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
                    "gains:UnitaryGains",
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

    def test_an_empty_component_list_validates_vacuously(self):
        """Nothing to chain, so nothing to reject.

        A model with no components is refused later, on its own terms -- the
        point here is that the order walk has no opinion about it, so the
        harvest can carry an empty list without the check inventing a failure.
        """
        validate_component_order([], BASE_STATE_KEYS)

    def test_the_accumulators_come_from_the_model(self):
        """``vis_ast``/``vis_rfi`` are zeroed by ``Model``, not by a component.

        A visibility component may therefore be the first thing in the list and
        still find the accumulator it adds into.
        """
        check(["ast_vis:GPVisAst"])


# Where a shipped ``model.components`` list can live. Searched recursively: a
# config in a subdirectory is as shipped as one at the top.
CONFIG_ROOTS = ("examples", "tests/data", "ci/reframe/data")
DOC_ROOT = "docs"

#: A fenced YAML block in Markdown, however it is indented and however its info
#: string is spelled. Three or more backticks open it (four are how a block that
#: itself contains a fence is written), ``yaml``/``yml`` in any case must be the
#: first word of the info string but may be followed by attributes, and it closes
#: on a run at least as long at the same indent -- with nothing after it but
#: horizontal whitespace, so a line that merely *starts* with backticks is not a
#: closer and cannot truncate the block.
_YAML_FENCE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<ticks>`{3,})[ \t]*ya?ml(?![^ \t\n])[^\n]*\n"
    r"(?P<body>.*?)"
    r"^(?P=indent)(?P=ticks)`*[ \t]*$",
    re.M | re.S | re.I,
)

#: Whether *unparseable* text looked like it was meant to carry a component
#: list. Deciding that lexically is only defensible when there is no parse to
#: consult: it reads a ``components:`` key inside a block scalar or a comment as
#: real, and misses one written quoted or in flow style. Both are tolerable in a
#: best-effort screen over text that is already known to be broken -- the worst
#: case is a broken document reported with a slightly wrong reason, or not
#: reported. It must never be asked about text that parsed.
_COMPONENTS_KEY = re.compile(r"^[ \t]*components[ \t]*:", re.M)

#: Every place the harvest below must find a list, and how many it must find
#: there. Pinned exactly rather than as a lower bound so that a shipped example
#: disappearing from the harvest -- a fence that stops matching, a file that
#: moves -- fails here instead of quietly shrinking the coverage.
EXPECTED_LOCATIONS = {
    "ci/reframe/data/tab_target.yaml": 1,
    "examples/tab_target.yaml": 1,
    "tests/data/tab_target.yaml": 1,
    "docs/config.md": 3,
    "docs/kernels.md": 1,
}

_UPDATE_HINT = (
    "Update EXPECTED_LOCATIONS in this file when you add, move or remove a "
    "shipped config or a documented model.components example."
)


def _fenced_yaml_blocks(text):
    """The YAML blocks of a Markdown document, dedented to their fence."""
    for match in _YAML_FENCE.finditer(text):
        indent, body = match.group("indent"), match.group("body")
        if indent:
            body = "\n".join(
                line[len(indent) :] if line.startswith(indent) else line
                for line in body.splitlines()
            )
        yield body


def _component_lists(label, text, broken):
    """The ``model.components`` lists in one YAML document.

    A document meant to carry a component list but not readable as one is
    recorded in ``broken`` rather than skipped: silently dropping it would hide
    exactly the breakage this harvest exists to catch.

    "Meant to" is decided from the parsed structure whenever there *is* one --
    a ``components`` key under ``model``, or one at the root, which is malformed
    since every shipped list in this repo sits under ``model:``. Text is only
    consulted when the parse failed, where there is nothing else to go on;
    judging a document that parsed by its text would condemn perfectly good YAML
    whose prose happens to contain a ``components:`` line.
    """
    try:
        config = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        if _COMPONENTS_KEY.search(text):
            broken.append(f"{label}: not valid YAML ({exc.__class__.__name__})")
        return []

    if not isinstance(config, dict):
        return []

    model = config.get("model")
    if not (isinstance(model, dict) and "components" in model):
        if "components" in config:
            broken.append(
                f"{label}: carries a root-level components: "
                "but no model.components mapping"
            )
        return []

    refs = model["components"]
    if not isinstance(refs, list):
        broken.append(f"{label}: 'model.components' is not a list")
        return []
    # An explicitly empty list is harvested like any other, so that it shows up
    # in the location counts instead of vanishing between the two.
    return [refs]


def _shipped_component_lists():
    """Every shipped ``model.components`` list, as (label, refs), plus breakages."""
    found, broken = [], []

    for root in CONFIG_ROOTS:
        # Enumerate everything and match on the lowered suffix: a glob pattern is
        # itself case-sensitive on a case-sensitive filesystem, so filtering a
        # `*.y*ml` glob would never get the chance to see a CONFIG.YAML.
        paths = (REPO_ROOT / root).rglob("*")
        for path in sorted(p for p in paths if p.suffix.lower() in (".yaml", ".yml")):
            label = str(path.relative_to(REPO_ROOT))
            for refs in _component_lists(label, path.read_text(), broken):
                found.append((label, refs))

    for path in sorted((REPO_ROOT / DOC_ROOT).rglob("*.md")):
        label = str(path.relative_to(REPO_ROOT))
        lists = []
        for i, block in enumerate(_fenced_yaml_blocks(path.read_text())):
            lists += _component_lists(f"{label} block {i}", block, broken)
        found += [(f"{label}#{i}", refs) for i, refs in enumerate(lists)]

    return found, broken


ALL_SHIPPED_LISTS, BROKEN_SHIPPED_LISTS = _shipped_component_lists()


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


class TestTheHarvestDecision:
    """What counts as a component list, a breakage, and neither.

    The harvest above is the only thing standing between a broken shipped
    example and a green CI run, so its judgement is pinned here directly rather
    than only through whatever happens to be in the repo today.
    """

    def read(self, text):
        broken = []
        return _component_lists("doc", text, broken), broken

    def test_a_model_components_list_is_harvested(self):
        found, broken = self.read("model:\n  components:\n    - gains:UnitaryGains\n")
        assert found == [["gains:UnitaryGains"]]
        assert broken == []

    def test_an_explicitly_empty_list_is_harvested_not_dropped(self):
        """``components: []`` is a real, visible list, not an absent one.

        Discarding it would put it in neither the harvest nor the breakages, so
        it would slip past the exact-location counts -- the one thing those
        counts exist to make impossible.
        """
        found, broken = self.read("model:\n  components: []\n")
        assert found == [[]]
        assert broken == []

    def test_prose_mentioning_components_is_not_condemned(self):
        """A ``components:`` line inside a block scalar is content, not a key.

        The document parses, and the parse says there is no component list
        anywhere in it. Reaching for the text instead would fail a file that is
        perfectly well formed.
        """
        found, broken = self.read("note: |\n  components: only prose\n")
        assert found == []
        assert broken == []

    def test_a_root_level_components_list_is_a_breakage(self):
        found, broken = self.read("components:\n  - gains:UnitaryGains\n")
        assert found == []
        assert broken == ["doc: carries a root-level components: but no model.components mapping"]

    def test_a_non_list_components_value_is_a_breakage(self):
        found, broken = self.read("model:\n  components: gains:UnitaryGains\n")
        assert found == []
        assert broken == ["doc: 'model.components' is not a list"]

    def test_unparseable_text_that_looks_like_a_list_is_a_breakage(self):
        found, broken = self.read("model:\n  components:\n   - a\n  - b\n")
        assert found == []
        assert broken == ["doc: not valid YAML (ParserError)"]

    def test_unparseable_text_that_looks_like_nothing_is_ignored(self):
        found, broken = self.read("data:\n   - a\n  - b\n")
        assert (found, broken) == ([], [])


def test_no_shipped_yaml_carrying_a_component_list_is_unreadable():
    """A shipped example that cannot be read is a failure, never a skip."""
    assert not BROKEN_SHIPPED_LISTS, (
        "shipped YAML carrying a 'components:' key could not be read as a "
        f"model.components list:\n  " + "\n  ".join(BROKEN_SHIPPED_LISTS)
    )


def test_the_harvest_found_every_expected_location():
    """The harvest itself, pinned to an exact set of places and counts.

    The tests below are only as good as what this collects, and a harvest that
    quietly stops seeing a file proves nothing while still passing. Pinning the
    locations means a fence that stops matching, a config that moves, or an
    example that is deleted fails here.
    """
    counts = {}
    for label, _ in ALL_SHIPPED_LISTS:
        counts[label.split("#")[0]] = counts.get(label.split("#")[0], 0) + 1
    assert counts == EXPECTED_LOCATIONS, _UPDATE_HINT


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


class TestTheComponentScan:
    """Discovery has to survive everything a module namespace holds.

    The scan reads ``vars(module)``, which is full of things that are not
    classes -- type aliases, constants, functions -- and ``issubclass`` raises
    on any of them. The guard cannot be ``isinstance(obj, type)``: up to Python
    3.10 a PEP 585 alias proxies ``__class__`` to its origin and passes it, so
    that spelling discovered nothing locally on 3.13 and crashed collection on
    the 3.10 floor, where ``numpy.typing.NDArray`` is such an alias and is
    imported by three of the scanned modules.
    """

    #: What ``numpy.typing.NDArray`` is on the versions that broke.
    alias = np.ndarray[Any, np.dtype[Any]]

    def test_a_generic_alias_is_not_a_class(self):
        assert not is_class(self.alias)

    def test_letting_one_through_is_what_breaks_collection(self):
        """Why the guard has to hold: the call it guards raises, on every version."""
        with pytest.raises(TypeError):
            issubclass(self.alias, Component)

    def test_real_classes_are_still_classes(self):
        for cls in (int, Component, *in_tree_components().values()):
            assert is_class(cls)

    def test_the_scanned_modules_really_do_hold_non_classes(self):
        """The precondition, so this stays a regression test and not a tautology.

        If the modules stopped importing anything but classes the guard would
        pass vacuously and the next `NDArray`-shaped import would break the
        floor again with nothing to catch it.
        """
        package = importlib.import_module("tabascal.components")
        non_classes = [
            f"{info.name}:{name}"
            for info in pkgutil.iter_modules(package.__path__)
            for name, obj in vars(
                importlib.import_module(f"tabascal.components.{info.name}")
            ).items()
            if not is_class(obj)
        ]
        assert non_classes
        assert in_tree_components()


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
