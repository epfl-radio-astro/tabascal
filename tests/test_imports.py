"""Resolving ``model.components`` references, and what a stale one has to say.

PR #106 renamed most components and deleted six of them. There are no aliases
and none are coming: a config written before it is expected to fail, and the
user is expected to edit it. What is pinned here is that the failure explains
itself -- it names the reference that failed, says what to write instead when
the name is one of the known old ones, lists what the module actually offers,
and points at the migration table. What it used to say named neither the rename
nor the valid options: only that the class was not in the module.
"""

import re
import sys
import textwrap
from pathlib import Path

import pytest

from tabascal.components import in_tree_components
from tabascal.components.rfi_signal import ComplexRFIConstAnt, ComplexRFIVarAnt
from tabascal.components.rfi_vis import RiemannVis
from tabascal.imports import (
    MIGRATION_DOCS,
    REMOVED_COMPONENTS,
    RENAMED_COMPONENTS,
    import_components,
)


#: The page the failures point at, read once.
CONFIG_DOCS = (Path(__file__).resolve().parents[1] / "docs" / "config.md").read_text()

#: Just the section the pointer names -- from its heading to the next one at the
#: same level -- so that a name is checked where the user is sent, not anywhere
#: on a long page.
MIGRATION_SECTION = CONFIG_DOCS.split("### Renamed and removed components\n", 1)[
    -1
].split("\n### ", 1)[0]


def message(*refs):
    """The error text from trying to import ``refs``."""
    with pytest.raises(ImportError) as excinfo:
        import_components(list(refs))
    return str(excinfo.value)


class TestTheHappyPath:
    """Current names keep resolving, in every spelling the importer accepts."""

    def test_a_current_reference_resolves(self):
        assert import_components(["rfi_signal:ComplexRFIVarAnt"]) == [ComplexRFIVarAnt]

    def test_the_dotted_spelling_resolves_the_same(self):
        assert import_components(["rfi_signal.ComplexRFIVarAnt"]) == [ComplexRFIVarAnt]

    def test_a_fully_qualified_reference_resolves(self):
        """The base package is a convenience, not a requirement."""
        assert import_components(["tabascal.components.rfi_vis:RiemannVis"]) == [
            RiemannVis
        ]

    def test_a_whole_current_model_resolves(self):
        refs = [
            "trajectory:FixedOrbit",
            "rfi_signal:ComplexRFIVarAnt",
            "rfi_vis:RiemannVis",
            "ast_vis:GPVisAst",
            "gains:UnitaryGains",
        ]
        assert [cls.__name__ for cls in import_components(refs)] == [
            ref.split(":")[1] for ref in refs
        ]

    @pytest.mark.parametrize("ref", sorted(in_tree_components()))
    def test_every_in_tree_component_resolves_by_its_reference(self, ref):
        assert import_components([ref]) == [in_tree_components()[ref]]


class TestAnUnknownClassInAKnownModule:
    """The generic failure: a name the module does not have, stale or misspelt."""

    ref = "rfi_signal:NoSuchComponent"

    def test_the_requested_reference_is_named(self):
        text = message(self.ref)
        assert self.ref in text or "NoSuchComponent" in text
        assert "rfi_signal" in text

    def test_what_the_module_does_offer_is_listed(self):
        """Introspected, so the list cannot drift from the module."""
        text = message(self.ref)
        offered = [
            name
            for name in (ComplexRFIVarAnt.__name__, ComplexRFIConstAnt.__name__)
            if name in text
        ]
        assert len(offered) >= 2

    def test_an_abstract_base_is_not_offered(self):
        """``BaseGPRFI`` cannot be listed in a config, so suggesting it misleads."""
        assert "BaseGPRFI" not in message(self.ref)

    def test_the_migration_table_is_pointed_at(self):
        assert MIGRATION_DOCS in message(self.ref)

    def test_an_unknown_name_gets_no_invented_replacement(self):
        """Only the known old names carry a rename note."""
        text = message(self.ref)
        assert "was renamed to" not in text
        assert "was deleted" not in text


class TestTheKnownOldNames:
    """Every name PR #106 changed says what it became."""

    @pytest.mark.parametrize("old,new", sorted(RENAMED_COMPONENTS.items()))
    def test_a_renamed_component_names_its_replacement(self, old, new):
        module = new.split(":")[0]
        text = message(f"{module}:{old}")
        assert old in text
        assert new in text
        assert "was renamed to" in text

    @pytest.mark.parametrize("old,nearest", sorted(REMOVED_COMPONENTS.items()))
    def test_a_deleted_component_says_so_and_names_the_nearest(self, old, nearest):
        module = nearest.split(":")[0]
        text = message(f"{module}:{old}")
        assert old in text
        assert nearest in text
        assert "was deleted" in text
        assert "no successor" in text

    @pytest.mark.parametrize(
        "old,new",
        sorted({**RENAMED_COMPONENTS, **REMOVED_COMPONENTS}.items()),
    )
    def test_a_known_old_name_still_fails(self, old, new):
        """The map is message-only. Naming the replacement is not resolving it."""
        module = new.split(":")[0]
        with pytest.raises(ImportError):
            import_components([f"{module}:{old}"])

    @pytest.mark.parametrize(
        "replacement",
        sorted(set(RENAMED_COMPONENTS.values()) | set(REMOVED_COMPONENTS.values())),
    )
    def test_every_replacement_is_a_real_component(self, replacement):
        """A later rename must not leave the migration table pointing at nothing."""
        assert replacement in in_tree_components()

    def test_the_rows_of_the_documented_table_are_covered(self):
        """The names the maps have to carry, whatever else is added to them."""
        documented = {
            "FourierGPRFI",
            "FourierGPRFIConstAnt",
            "RiemannVisTimeFreqCalculation",
            "RiemannVisTimeFreqCalculationFFI",
            "FourierTimeFreqGPAst",
            "SGP4LEONoDragOrbit",
            "ComplexRFI",
            "RealRFI",
        }
        assert documented <= set(RENAMED_COMPONENTS) | set(REMOVED_COMPONENTS)


class TestTheDocumentedTable:
    """The importer and the migration table have to say the same thing.

    Every failure points the user at that table, so a name the importer knows
    about but the table does not -- or a replacement the two spell differently --
    sends them to a page that cannot answer the question they arrived with.
    """

    @pytest.mark.parametrize(
        "old,new", sorted({**RENAMED_COMPONENTS, **REMOVED_COMPONENTS}.items())
    )
    def test_every_name_the_importer_knows_is_documented(self, old, new):
        # Whole words, inside the section the error points at: 'ComplexRFI' is a
        # prefix of a current class name and occurs all over the page, so a
        # substring search anywhere in the file passes without the row existing.
        assert re.search(rf"\b{re.escape(old)}\b", MIGRATION_SECTION)
        assert re.search(rf"\b{re.escape(new)}\b", MIGRATION_SECTION)

    def test_the_pointer_lands_on_the_section_that_holds_it(self):
        """The anchor in the URL is the heading, slugified. Keep them together."""
        anchor = MIGRATION_DOCS.split("#")[1].replace("-", " ")
        assert f"### {anchor.capitalize()}\n" in CONFIG_DOCS

    def test_the_section_was_actually_found(self):
        """Guards the two above: an empty slice would pass nothing, silently."""
        assert "ComplexRFIVarAnt" in MIGRATION_SECTION
        assert len(MIGRATION_SECTION) < len(CONFIG_DOCS)


class TestAnUnknownModule:
    """A module that is not there gets the same treatment as a missing class."""

    ref = "rfi_signals:ComplexRFIVarAnt"

    def test_the_module_is_named(self):
        text = message(self.ref)
        assert "rfi_signals" in text

    def test_the_modules_that_do_exist_are_listed(self):
        text = message(self.ref)
        assert "rfi_signal" in text
        assert "trajectory" in text

    def test_the_migration_table_is_pointed_at(self):
        assert MIGRATION_DOCS in message(self.ref)

    def test_a_stale_class_in_an_unknown_module_still_names_its_replacement(self):
        text = message("rfi_signals:FourierGPRFI")
        assert "rfi_signal:ComplexRFIVarAnt" in text


class TestAModuleThatCannotBeImported:
    """A module whose *own* import fails is not a missing module.

    Reporting it as one sends the user hunting for a typo in a name that is
    spelt correctly; the thing to fix is inside the module.
    """

    @pytest.fixture
    def module_body(self):
        """A module that dies on import for the commonest reason: a missing dep."""
        return "import a_dependency_that_is_not_installed  # noqa: F401\n"

    @pytest.fixture
    def loose(self, tmp_path, monkeypatch, module_body):
        """The broken module as a top-level module, imported with no base package."""
        (tmp_path / "brokencomponent.py").write_text(module_body)
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delitem(sys.modules, "brokencomponent", raising=False)
        return {"refs": ["brokencomponent:Thing"], "base_package": None}

    @pytest.fixture
    def packaged(self, tmp_path, monkeypatch, module_body):
        """The same module inside a package, reached the way a config reaches one.

        The default path resolves against ``base_package`` first, so this is the
        shape that actually runs in production; the loose one only exercises the
        fallback.
        """
        package = tmp_path / "brokenpackage"
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "brokencomponent.py").write_text(module_body)
        monkeypatch.syspath_prepend(str(tmp_path))
        for name in ("brokenpackage", "brokenpackage.brokencomponent"):
            monkeypatch.delitem(sys.modules, name, raising=False)
        return {"refs": ["brokencomponent:Thing"], "base_package": "brokenpackage"}

    @pytest.mark.parametrize("shape", ["loose", "packaged"])
    def test_the_missing_dependency_is_reported_not_the_module(self, shape, request):
        case = request.getfixturevalue(shape)
        with pytest.raises(ImportError) as excinfo:
            import_components(case["refs"], base_package=case["base_package"])
        text = str(excinfo.value)
        assert "brokencomponent" in text
        assert "a_dependency_that_is_not_installed" in text
        assert not re.search(r"there is no module 'brokencomponent'", text)

    @pytest.mark.parametrize(
        "module_body",
        [
            pytest.param("raise ImportError('boom')\n", id="ImportError"),
            pytest.param("raise ValueError('boom')\n", id="ValueError"),
            pytest.param(
                "raise ModuleNotFoundError('boom')\n", id="no-name-on-the-error"
            ),
        ],
    )
    def test_any_import_time_failure_still_names_the_reference(self, loose):
        """Not just ``ModuleNotFoundError``: whatever the module raises.

        Without this the exception reached the caller as its own bare text, with
        no reference attached, so a list of components said only 'boom'.
        """
        with pytest.raises(ImportError) as excinfo:
            import_components(loose["refs"], base_package=None)
        text = str(excinfo.value)
        assert "brokencomponent:Thing" in text
        assert "could not be imported" in text
        assert "boom" in text


class TestTheReportItself:
    """One raise for the whole list, and a reference that is not a reference."""

    def test_every_bad_reference_is_reported_not_just_the_first(self):
        text = message("rfi_signal:FourierGPRFI", "ast_vis:FourierTimeFreqGPAst")
        assert "ComplexRFIVarAnt" in text
        assert "GPVisAst" in text

    def test_a_good_reference_beside_a_bad_one_does_not_rescue_the_call(self):
        with pytest.raises(ImportError):
            import_components(["rfi_signal:ComplexRFIVarAnt", "rfi_signal:Nope"])

    def test_a_reference_with_no_module_part_is_rejected(self):
        assert "not a valid" in message("ComplexRFIVarAnt")

    def test_resolving_to_something_that_is_not_a_class_is_rejected(self):
        text = message("rfi_signal:jnp")
        assert "not a class" in text
