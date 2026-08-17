"""Declarative schema for the tabascal config YAML (the ``-c`` option).

Every configuration parameter is declared as data on the class that reads it: the
model components declare what they need in their own ``config_params``, and the
parameters read outside any component (``model``, ``data``, ``plots``, ``opt``,
``satellites`` and the RFI sampling grid) are declared on
:class:`tabascal.config.TabConfig`. There is no packaged base config to merge
into and no second copy of the key names to keep in sync -- the declaration next
to the reader *is* the schema.

:func:`collect_params` merges the declarations for one run (``TabConfig`` plus the
classes named in ``model.components``) and :func:`validate_config` checks the
user's file against that merged set, applying defaults and collecting **every**
problem into a single :class:`ConfigError`. Since only the *selected* components
contribute, a key belonging to a component that is not in the model is reported
as such rather than silently ignored.

This mirrors the fail-fast convention already used by
:func:`tabascal.truth.require_truth` and
:func:`tabascal.orbit_config.normalise_tle_config`: validate before anything
expensive (the MS read, the TLE preflight) and name every failure at once. It
deliberately stays a small, purpose-built checker rather than a general schema
language -- no new dependency, no JAX import, pure Python.

Semantic checks that need more than a type and a range stay where they are:
:func:`~tabascal.orbit_config.normalise_tle_config` still owns the cross-field
rules for ``satellites``, and the parameters marked :data:`FROM_DATA` are
resolved by their component from the measurement set, which does not exist yet at
validation time.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml


class ConfigError(ValueError):
    """The config YAML is malformed, incomplete, or fails validation.

    Subclasses :class:`ValueError` so it reads as what it is -- a bad argument --
    and is caught by the CLI alongside ``TLEError``/``TruthError`` and printed as
    a message rather than a traceback.
    """


class _Sentinel:
    """A named marker used as a ``Param.default`` that is not a value."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return self.name


#: The parameter has no default and must appear in the config file.
REQUIRED = _Sentinel("<required>")

#: The parameter defaults to a value derived from the measurement set data (the
#: frequency/time extent, the observed visibility amplitude, the dish diameter).
#: It cannot be resolved here -- validation leaves it as ``None`` and the owning
#: component fills it in during ``setup``, once the data has been read.
FROM_DATA = _Sentinel("<derived from the data>")


@dataclass(frozen=True)
class Param:
    """Specification for a single config parameter.

    Attributes
    ----------
    types
        Accepted Python types. ``bool`` never satisfies a numeric parameter even
        though ``isinstance(True, int)`` is True in Python; declare
        ``types=(bool,)`` for a genuine flag.
    default
        The value used when the key is absent or ``null``. :data:`REQUIRED` makes
        the key mandatory; :data:`FROM_DATA` resolves to ``None`` for the owning
        component to fill in from the data.
    choices
        Permitted values, combined with ``types`` as an *or*: a value passes if it
        is one of ``choices`` **or** an instance of ``types``. That is what lets
        ``ast.mean`` be either a number or ``"data"``. For a plain enum leave
        ``types`` empty, otherwise ``types=(str,)`` would let any string through.
    item
        Element type(s) for a list-valued parameter.
    gt, ge, lt, le
        Bounds for a numeric parameter.
    null_ok
        An explicit ``null`` in the config is a *meaningful value* and is kept as
        ``None`` instead of being replaced by ``default``. Needed for the handful
        of parameters where null means "off" rather than "unset" --
        ``rfi.min_elevation: null`` disables elevation masking, and
        ``satellites.remote_max_age_days: null`` removes the age ceiling -- which
        would otherwise be impossible to express, since the default is not null.
    doc
        One line describing the parameter. Shown by ``tabascal check-config`` and
        appended to this parameter's error message.
    """

    types: Tuple[type, ...] = ()
    default: Any = REQUIRED
    choices: Tuple[Any, ...] = ()
    item: Optional[Tuple[type, ...]] = None
    gt: Optional[float] = None
    ge: Optional[float] = None
    lt: Optional[float] = None
    le: Optional[float] = None
    null_ok: bool = False
    doc: str = ""

    @property
    def required(self) -> bool:
        return self.default is REQUIRED


# ---------------------------------------------------------------------------
# Value checking
# ---------------------------------------------------------------------------

_TYPE_NAMES = {
    bool: "a boolean",
    int: "an integer",
    float: "a float",
    str: "a string",
    list: "a list",
    dict: "a mapping",
}

_TYPE_PLURALS = {
    bool: "booleans",
    int: "integers",
    float: "floats",
    str: "strings",
}


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _isinstance(value: Any, types: Tuple[type, ...]) -> bool:
    """``isinstance`` that never lets a bool satisfy a numeric type."""
    if _is_bool(value) and bool not in types:
        return False
    return isinstance(value, types)


def _equal(value: Any, choice: Any) -> bool:
    """Equality that keeps ``True``/``1`` and ``False``/``0`` distinct."""
    if _is_bool(value) != _is_bool(choice):
        return False
    return value == choice


def _type_names(types: Tuple[type, ...]) -> str:
    if set(types) == {int, float}:
        return "a number"
    return " or ".join(_TYPE_NAMES.get(t, t.__name__) for t in types)


def _item_names(types: Tuple[type, ...]) -> str:
    if set(types) == {int, float}:
        return "numbers"
    return " or ".join(_TYPE_PLURALS.get(t, f"{t.__name__}s") for t in types)


def _expected(param: Param) -> str:
    """Human-readable description of what this parameter accepts."""
    parts: List[str] = []
    if param.types:
        described = _type_names(param.types)
        if param.item is not None:
            described += f" of {_item_names(param.item)}"
        parts.append(described)
    if param.choices:
        parts.append("one of " + ", ".join(repr(c) for c in param.choices))
    expected = " or ".join(parts) if parts else "a value"

    bounds = [
        (name, bound)
        for name, bound in (
            (">", param.gt), (">=", param.ge), ("<", param.lt), ("<=", param.le)
        )
        if bound is not None
    ]
    if bounds:
        expected += " " + " and ".join(f"{op} {bound:g}" for op, bound in bounds)
    return expected


def _check_value(path: str, value: Any, param: Param) -> List[str]:
    """Return the problems with *value* under *param* (empty when it is fine)."""
    ok = False
    if param.choices and any(_equal(value, choice) for choice in param.choices):
        ok = True
    elif param.types and _isinstance(value, param.types):
        ok = True
    if not ok:
        return [f"{path}: expected {_expected(param)}, got {value!r}"]

    problems: List[str] = []
    if param.item is not None and isinstance(value, list):
        bad = [item for item in value if not _isinstance(item, param.item)]
        if bad:
            problems.append(
                f"{path}: expected {_expected(param)}, but "
                + ", ".join(repr(item) for item in bad[:3])
                + " is not"
            )
    if not _is_bool(value) and isinstance(value, (int, float)):
        for op, bound, fails in (
            (">", param.gt, lambda v, b: v <= b),
            (">=", param.ge, lambda v, b: v < b),
            ("<", param.lt, lambda v, b: v >= b),
            ("<=", param.le, lambda v, b: v > b),
        ):
            if bound is not None and fails(value, bound):
                problems.append(f"{path}: must be {op} {bound:g}, got {value!r}")
    return problems


# ---------------------------------------------------------------------------
# Collecting declarations
# ---------------------------------------------------------------------------

def _declared(owner: type) -> Dict[str, Param]:
    """The parameters *owner* declares, with its bases' declarations merged in.

    Walks the MRO base-first so a subclass extends -- and may override -- what its
    base declared, which is how ``ComplexRFIVarAnt`` adds the padding factors to
    ``BaseGPRFI``'s set.
    """
    declared: Dict[str, Param] = {}
    for klass in reversed(getattr(owner, "__mro__", (owner,))):
        declared.update(vars(klass).get("config_params", {}))
    return declared


def collect_params(*owners: type) -> Dict[str, Param]:
    """Merge the ``config_params`` of every *owner* into one path -> Param map.

    Owners are ``TabConfig`` and the component classes named in
    ``model.components``. Two owners declaring the same path is normal and
    expected -- every ``rfi_vis`` component reads ``rfi.freq_int_samples`` -- as
    long as they declare it identically. Declaring it *differently* is a bug in
    the declarations, not in the user's config, so it raises immediately.
    """
    merged: Dict[str, Param] = {}
    sources: Dict[str, str] = {}
    for owner in owners:
        name = getattr(owner, "__name__", repr(owner))
        for path, param in _declared(owner).items():
            previous = merged.get(path)
            if previous is not None and previous != param:
                raise ConfigError(
                    f"Conflicting declarations for config parameter '{path}': "
                    f"{sources[path]} and {name} declare it differently. "
                    "This is a bug in the component declarations."
                )
            merged[path] = param
            sources.setdefault(path, name)
    return merged


def component_param_owners() -> Dict[str, str]:
    """Map every config path declared anywhere in ``tabascal.components``.

    Used only to improve the message for an unrecognised key: a key that a real
    component declares, but which no *selected* component reads, is a modelling
    mistake ("you configured the astronomical prior but there is no astronomical
    component") rather than a typo, and is worth saying so.
    """
    import importlib
    import pkgutil
    import types

    import tabascal.components as package
    from tabascal.components import Component

    owners: Dict[str, str] = {}
    for info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{package.__name__}.{info.name}")
        for name, obj in vars(module).items():
            # Generic aliases such as numpy's ``NDArray`` are imported into the
            # component modules, and on Python < 3.11 they pass ``isinstance(obj,
            # type)`` while still blowing up ``issubclass``, so skip them first.
            if isinstance(obj, types.GenericAlias):
                continue
            if (
                isinstance(obj, type)
                and issubclass(obj, Component)
                and obj is not Component
                and obj.__module__ == module.__name__
            ):
                for path in _declared(obj):
                    owners.setdefault(path, f"{info.name}:{name}")
    return owners


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _tree(params: Mapping[str, Param]) -> Dict[str, Any]:
    """Nest a flat ``{"ast.pow_spec.p0": Param}`` map into a dict of dicts."""
    root: Dict[str, Any] = {}
    for path, param in params.items():
        node = root
        parts = path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(
                    f"'{path}' is declared both as a section and as a parameter."
                )
        node[parts[-1]] = param
    return root


def _as_mapping(value: Any) -> Optional[Mapping]:
    """A YAML section, with an empty ``section:`` read as an empty mapping."""
    if value is None:
        return {}
    return value if isinstance(value, Mapping) else None


def _resolve(
    tree: Mapping[str, Any], config: Mapping, prefix: str, problems: List[str]
) -> Dict[str, Any]:
    """Build the resolved config for one level of the schema tree."""
    resolved: Dict[str, Any] = {}
    for key, node in tree.items():
        path = f"{prefix}{key}"
        value = config.get(key)

        if isinstance(node, dict):
            section = _as_mapping(value)
            if section is None:
                problems.append(f"{path}: expected a mapping, got {value!r}")
                section = {}
            resolved[key] = _resolve(node, section, f"{path}.", problems)
            continue

        if key not in config or value is None:
            if node.required:
                problems.append(f"{path}: required, but not set" + _hint(node))
            elif node.null_ok and key in config:
                # An explicit null that means something -- see Param.null_ok.
                resolved[key] = None
            else:
                resolved[key] = None if node.default is FROM_DATA else node.default
            continue

        value_problems = _check_value(path, value, node)
        if value_problems:
            problems.extend(problem + _hint(node) for problem in value_problems)
        resolved[key] = value
    return resolved


def _hint(param: Param) -> str:
    return f" ({param.doc})" if param.doc else ""


def _unknown(
    tree: Mapping[str, Any],
    config: Mapping,
    prefix: str,
    known: Mapping[str, Param],
    elsewhere: Mapping[str, str],
    problems: List[str],
) -> None:
    """Report every key of *config* that the schema tree does not declare."""
    for key, value in config.items():
        path = f"{prefix}{key}"
        node = tree.get(key)
        if node is None:
            problems.append(f"{path}: unknown key{_suggestion(path, known, elsewhere)}")
        elif isinstance(node, dict):
            section = _as_mapping(value)
            if section:
                _unknown(node, section, f"{path}.", known, elsewhere, problems)


def _split(path: str) -> Tuple[str, str]:
    """``"ast.pow_spec.p0"`` -> ``("ast.pow_spec", "p0")``."""
    parent, _, leaf = path.rpartition(".")
    return parent, leaf


def _suggestion(
    path: str, known: Mapping[str, Param], elsewhere: Mapping[str, str]
) -> str:
    """A ``did you mean``/``declared by`` hint for an unrecognised key."""
    owner = _owner(path, elsewhere)
    if owner is not None:
        return (
            f" -- declared by {owner}, which is not in model.components; "
            "add the component or remove the key"
        )

    candidates = list(known) + list(elsewhere)
    parent, leaf = _split(path)
    siblings = [c for c in candidates if _split(c)[0] == parent]

    # A key in the same section is the likeliest thing that was meant, so look
    # there first -- a misspelling ('cont'), then a case slip ('P0' for 'p0').
    # Only then widen to the same leaf name in another section, which is what
    # finds a key written under the wrong heading, and finally to the whole path.
    match = (
        _closest(leaf, siblings, exact_first=True)
        or _closest(leaf, candidates, exact_first=True, fuzzy=False)
        or _closest(path, candidates)
    )
    return f" (did you mean '{match}'?)" if match else ""


def _owner(path: str, elsewhere: Mapping[str, str]) -> Optional[str]:
    """Which unselected component declares *path*, or anything beneath it."""
    if path in elsewhere:
        return elsewhere[path]
    owners = sorted({o for p, o in elsewhere.items() if p.startswith(f"{path}.")})
    return ", ".join(owners) if owners else None


def _closest(
    name: str,
    candidates: Sequence[str],
    *,
    exact_first: bool = False,
    fuzzy: bool = True,
) -> Optional[str]:
    """The candidate path whose leaf (or whole path) best matches *name*."""
    by_key = {c if not exact_first else _split(c)[1]: c for c in reversed(candidates)}
    if exact_first:
        for key, candidate in by_key.items():
            if key.lower() == name.lower():
                return candidate
    if not fuzzy:
        return None
    matches = difflib.get_close_matches(name, list(by_key), n=1, cutoff=0.75)
    return by_key[matches[0]] if matches else None


def validate_config(
    config: Optional[Mapping],
    params: Mapping[str, Param],
    *,
    source: Optional[str] = None,
    strict: bool = True,
    elsewhere: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Validate *config* against *params* and return it with defaults applied.

    Every problem is collected and raised together as one :class:`ConfigError`, so
    a config with four mistakes is fixed in one pass rather than four runs.

    Parameters
    ----------
    config
        The parsed YAML mapping. ``None`` (an empty file) is treated as empty.
    params
        Declared parameters, from :func:`collect_params`.
    source
        Path of the config file, used in the error message.
    strict
        Reject keys that *params* does not declare. Set ``False`` for the
        bootstrap pass in :func:`tabascal.config.load_config`, which validates
        only the handful of keys needed before the components can be imported and
        must leave the rest of the file alone.
    elsewhere
        Paths declared by components that are not part of this model, from
        :func:`component_param_owners`. Consulted only to explain an unknown key;
        defaults to computing it lazily when one is found.

    Returns
    -------
    dict
        A new config with every declared parameter present, so downstream
        ``config["section"]["key"]`` reads never raise ``KeyError``. Under
        ``strict`` the result is built in declaration order and contains nothing
        else; otherwise the undeclared parts of *config* are carried through.
    """
    mapping = _as_mapping(config)
    if mapping is None:
        raise ConfigError(
            _prefix(source) + f"expected a mapping of sections, got {config!r}"
        )

    tree = _tree(params)
    problems: List[str] = []
    resolved = _resolve(tree, mapping, "", problems)

    if strict:
        unknown_problems: List[str] = []
        _unknown(tree, mapping, "", params, elsewhere or {}, unknown_problems)
        if unknown_problems and elsewhere is None:
            # Only worth importing every component module to explain a key that
            # actually turned out to be unrecognised.
            unknown_problems = []
            _unknown(
                tree, mapping, "", params, component_param_owners(), unknown_problems
            )
        problems.extend(unknown_problems)
    else:
        resolved = _merge(mapping, resolved)

    if problems:
        raise ConfigError(
            _prefix(source)
            + f"{len(problems)} problem{'s' if len(problems) > 1 else ''} found\n\n  "
            + "\n  ".join(sorted(problems))
            + "\n"
        )
    return resolved


def _merge(config: Mapping, resolved: Mapping) -> Dict[str, Any]:
    """Overlay the *resolved* subset back onto the original config.

    Used by the non-strict (bootstrap) pass, which validates a couple of keys and
    must return the rest of the file untouched -- including any sibling of a key
    it did resolve, at whatever depth.
    """
    merged = dict(config)
    for key, value in resolved.items():
        section = _as_mapping(config.get(key))
        if isinstance(value, dict) and section:
            merged[key] = _merge(section, value)
        else:
            merged[key] = value
    return merged


def _prefix(source: Optional[str]) -> str:
    return f"invalid configuration in {source}: " if source else "invalid configuration: "


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_config(config: Mapping) -> str:
    """The resolved config as YAML, in declaration order."""
    return yaml.dump(dict(config), sort_keys=False, default_flow_style=False)
