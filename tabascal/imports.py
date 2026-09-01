from __future__ import annotations
import importlib
import inspect
import pkgutil
from typing import Iterable, List, Type

#: Where the migration table lives. Quoted in every failure to resolve a
#: component reference, since a stale config is the likeliest reason for one.
MIGRATION_DOCS = (
    "https://tabascal.readthedocs.io/en/latest/config.html"
    "#renamed-and-removed-components"
)

#: Components renamed in PR #106, as ``old class name -> current reference``.
#: Message-only: nothing here is ever resolved, so an old config still fails --
#: the map exists so that the failure can say what to write instead. Keyed on the
#: class name alone because that is what identifies the component; the module a
#: config happens to name it under does not have to be right for the hint to be.
RENAMED_COMPONENTS = {
    "FourierGPRFI": "rfi_signal:ComplexRFIVarAnt",
    "FourierGPRFIConstAnt": "rfi_signal:ComplexRFIConstAnt",
    "FourierTimeFreqGPAst": "ast_vis:GPVisAst",
    "RiemannVisTimeFreqCalculation": "rfi_vis:RiemannVis",
    "RiemannVisTimeFreqCalculationFFI": "rfi_vis:RiemannVisFFI",
    "RiemannVisTimeFreqVariable": "rfi_vis:RiemannVisVariable",
    "RiemannVisTimeFreqVariableFFI": "rfi_vis:RiemannVisVariableFFI",
    "SGP4LEONoDragOrbit": "trajectory:NoDragOrbit",
    "SGP4LEOOrbit": "trajectory:Orbit",
}

#: Components deleted in PR #106, as ``old class name -> nearest current
#: reference``. Nearest, not equivalent: none of these has a drop-in successor,
#: so a config using one has a modelling decision to make, not a substitution.
REMOVED_COMPONENTS = {
    "ComplexRFI": "rfi_signal:ComplexRFIVarAnt",
    "FourierTimeAst": "ast_vis:GPVisAst",
    "FourierTimeConstFreqAst": "ast_vis:GPVisAst",
    "FourierTimeFreqAst": "ast_vis:GPVisAst",
    "RealRFI": "rfi_signal:ComplexRFIVarAnt",
    "RiemannVisCalculation": "rfi_vis:RiemannVis",
}


def _migration_note(cls_name: str) -> str:
    """What became of ``cls_name``, if it is one of the names PR #106 changed."""
    if cls_name in RENAMED_COMPONENTS:
        return (
            f"'{cls_name}' was renamed to '{RENAMED_COMPONENTS[cls_name]}' in #106; "
            "there are no aliases, so update the config by hand."
        )
    if cls_name in REMOVED_COMPONENTS:
        return (
            f"'{cls_name}' was deleted in #106 with no successor; the nearest "
            f"current component is '{REMOVED_COMPONENTS[cls_name]}'."
        )
    return ""


def _is_class(obj) -> bool:
    """Whether ``obj`` really is a class, on every supported Python.

    The same guard as :func:`tabascal.components.is_class` -- ``isinstance(obj,
    type)`` is not enough, since up to Python 3.10 a PEP 585 alias proxies
    ``__class__`` to its origin -- kept local so that describing a module that
    failed to import does not itself import the component package.
    """
    return issubclass(type(obj), type)


def _component_base():
    """The base class a listable component has to subclass, if it can be had.

    Imported here rather than at module scope so that describing a reference
    that did not resolve does not itself drag in the component package, and jax
    with it, for a caller pointed at some other ``base_package``.
    """
    try:
        from tabascal.components import Component

        return Component
    except Exception:
        return None


def _offered_classes(module) -> List[str]:
    """The class names a module offers a config, by introspection.

    The same test :func:`tabascal.components.in_tree_components` applies, so the
    two cannot disagree about what is listable: a component, public, defined in
    the module itself -- a re-export belongs to the module it was written in --
    and concrete, since an abstract base cannot go in a config and offering one
    as an alternative would only mislead.
    """
    base = _component_base()
    return sorted(
        name
        for name, obj in vars(module).items()
        if not name.startswith("_")
        and _is_class(obj)
        and (base is None or issubclass(obj, base))
        and getattr(obj, "__module__", None) == module.__name__
        and not inspect.isabstract(obj)
    )


def _offered_modules(base_package: str | None) -> List[str]:
    """The module names ``base_package`` offers a config, by introspection.

    Best effort: the listing is a nicety, so a package that cannot be scanned
    leaves it out rather than replacing the real error with its own.
    """
    if not base_package:
        return []
    try:
        package = importlib.import_module(base_package)
        return sorted(info.name for info in pkgutil.iter_modules(package.__path__))
    except Exception:
        return []


def _sentence(text: str) -> str:
    """A sentence, or nothing at all, ready to be concatenated."""
    return f"{text} " if text else ""


def _listing(label: str, names: List[str]) -> str:
    return _sentence(f"{label}: {', '.join(names)}." if names else "")


def _pointer() -> str:
    return f"Renamed and removed components: {MIGRATION_DOCS}"


def _missing_class_message(ref: str, cls_name: str, module) -> str:
    return (
        f"'{ref}': module '{module.__name__}' has no class '{cls_name}'. "
        + _sentence(_migration_note(cls_name))
        + _listing("It defines", _offered_classes(module))
        + _pointer()
    )


def _broken_module_message(ref: str, module_name: str, exc: Exception) -> str:
    return (
        f"'{ref}': module '{module_name}' could not be imported: "
        f"{exc.__class__.__name__}: {exc}."
    )


def _missing_module_message(
    ref: str, mod_path: str, cls_name: str, base_package: str | None
) -> str:
    where = f" in '{base_package}', and none at top level" if base_package else ""
    return (
        f"'{ref}': there is no module '{mod_path}'{where}. "
        + _sentence(_migration_note(cls_name))
        + _listing(f"'{base_package}' holds", _offered_modules(base_package))
        + _pointer()
    )


def _absent(module_name: str, exc: ModuleNotFoundError) -> bool:
    """Whether ``exc`` says ``module_name`` is missing, rather than a dependency.

    A module that imports something uninstalled raises the same exception type as
    one that does not exist, and the two need opposite fixes. ``exc.name`` is the
    module that was not found: it is the one asked for (or a package above it)
    only in the first case. Without a name there is nothing to attribute it to,
    so it is not treated as absent -- the failure is then reported in full, which
    is true either way, rather than as a module that does not exist.
    """
    return exc.name is not None and (
        module_name == exc.name or module_name.startswith(f"{exc.name}.")
    )


def _import_module(ref: str, mod_path: str, cls_name: str, base_package: str | None):
    """Import ``mod_path``, relative to ``base_package`` first if there is one."""
    candidates = ([f"{base_package}.{mod_path}"] if base_package else []) + [mod_path]
    for name in candidates:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as e:
            if _absent(name, e):
                continue
            # The module is there; something it imports is not. Say that, rather
            # than sending the user to look for a typo in a name that is spelt
            # correctly.
            raise ImportError(_broken_module_message(ref, name, e)) from e
        except Exception as e:
            # Anything else the module does on import -- a failing top-level
            # call, a bad type annotation -- is its own failure and not a
            # missing name, so it is reported the same way rather than reaching
            # the caller as a bare exception with no reference attached.
            raise ImportError(_broken_module_message(ref, name, e)) from e
    raise ImportError(_missing_module_message(ref, mod_path, cls_name, base_package))


def import_components(
    class_paths: Iterable[str],
    *,
    base_package: str | None = "tabascal.components",
) -> List[Type]:
    """
    Import classes given strings like 'module:Class' or 'module.Class'.

    By default, resolves modules relative to 'tabascal.components'
    (e.g., 'foo:Foo' -> 'tabascal.components.foo.Foo') and then falls
    back to absolute imports if that fails.

    A reference that does not resolve raises, always: the components renamed and
    deleted in PR #106 have no aliases, so a config predating it is broken and
    has to be edited. The failure carries what is needed to do that -- the
    current name where the old one is known, what the module does offer, and the
    migration table.
    """
    classes: List[Type] = []
    errors: list[str] = []

    def _split(ref: str) -> tuple[str, str]:
        norm = ref.replace(":", ".")
        try:
            mod, cls = norm.rsplit(".", 1)
        except ValueError:
            raise ImportError(
                f"'{ref}' is not a valid 'module.Class' or 'module:Class' reference."
            )
        return mod, cls

    for ref in class_paths:
        try:
            mod_path, cls_name = _split(ref)

            module = _import_module(ref, mod_path, cls_name, base_package)

            try:
                cls = getattr(module, cls_name)
            except AttributeError as e:
                raise ImportError(
                    _missing_class_message(ref, cls_name, module)
                ) from e

            if not isinstance(cls, type):
                raise ImportError(
                    f"Resolved '{ref}' to '{module.__name__}.{cls_name}', "
                    f"but it is not a class (got {type(cls).__name__})."
                )

            classes.append(cls)

        except Exception as e:
            errors.append(str(e))

    if errors:
        raise ImportError(
            "Failed to import one or more classes:\n  - " + "\n  - ".join(errors)
        )

    return classes
