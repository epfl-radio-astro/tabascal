"""Resolving ``model.components`` references, and what a bad one has to say.

There are no aliases: a reference naming something no module defines is a
config to edit, and the run stops. What is pinned here is that the failure
explains itself rather than saying only that a class was not found -- it names
the reference that failed, distinguishes a missing module from a missing class
from a module whose own imports are broken, and lists what is actually on
offer at whichever of those levels went wrong.
"""

import re
import sys

import pytest

from tabascal.components import in_tree_components
from tabascal.components.rfi_signal import ComplexRFIConstAnt, ComplexRFIVarAnt
from tabascal.components.rfi_vis import RiemannVis
from tabascal.imports import (
    import_components,
)


#: The page the failures point at, read once.

#: Just the section the pointer names -- from its heading to the next one at the
#: same level -- so that a name is checked where the user is sent, not anywhere
#: on a long page.
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
