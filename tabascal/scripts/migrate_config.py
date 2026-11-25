"""Convert a run config from the sectioned layout to per-component options.

The old layout listed the model as ``model.components`` and put every
component's options in the shared ``ast``, ``rfi`` and ``gains`` sections::

    model:
      components: [rfi_signal:ComplexRFIVarAnt, rfi_vis:AnalyticVis, ...]
      precision: single
    rfi:
      init: truth
      gp_cov: {corr_time: 24}
      n_int_freq: 1

The new layout keeps ``model`` and ``satellites`` (alongside ``data``,
``plots``, ``opt``, and the top-level ``seed``) and moves each component's
options directly under its entry in ``model.components``::

    model:
      precision: single
      components:
        - rfi_signal.ComplexRFIVarAnt: {init: truth, corr_time_s: 24}
        - rfi_vis.AnalyticVis: {n_int_freq: 1}

Each option moves onto the component that reads it via the ``legacy_keys``
declared on each component's ``Config`` schema; an option no listed component
reads did nothing in the old layout either, and is dropped with a note saying
so. The remaining top-level sections are copied as they are. YAML comments are
not carried over.

Usage::

    tabascal migrate-config old.yaml -o new.yaml
"""

from __future__ import annotations

from collections.abc import Mapping
import sys
from typing import Any, Dict, List, Tuple

#: Sections whose options moved onto the components.
_COMPONENT_SECTIONS = ("ast", "rfi", "gains")

_ABSENT = object()


def _lookup(mapping: Mapping, path: Tuple[str, ...]) -> Any:
    node: Any = mapping
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return _ABSENT
        node = node[key]
    return node


def _leaves(section: Mapping, path: Tuple[str, ...] = ()):
    """Every key path in ``section`` that holds a value rather than a non-empty block."""
    for key, value in section.items():
        if isinstance(value, Mapping) and value:
            yield from _leaves(value, path + (key,))
        else:
            yield path + (key,)


def _migrate_null(config_cls, field: str, value: Any) -> Any:
    """What a legacy value of ``field`` is now: itself, a named default, or ``_ABSENT``.

    A legacy ``null`` meant the default on most options. Where the default is
    computed it now has a name (``legacy_null_value``); where ``null`` is no
    longer a setting at all, the key is left out so the default applies.
    """
    from tabascal.schema import field_allows_none

    if value is not None:
        return value
    if field in config_cls.legacy_null_value:
        return config_cls.legacy_null_value[field]
    field_info = config_cls.model_fields.get(field)
    if field_info is not None and not field_allows_none(field_info):
        return _ABSENT
    return value


def migrate(old: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Return ``(new_config, notes)`` for a config in the sectioned layout.

    ``old`` is the user's file as written, not merged over any defaults, so the
    result names only what the user set. The notes say what was dropped and why.
    """
    from tabascal.imports import canonical_component_ref, import_components
    from tabascal.schema import EmptyComponentConfig, TOP_LEVEL_SECTIONS

    old = dict(old or {})
    notes: List[str] = []
    raw_model = old.pop("model", None) or {}
    if not isinstance(raw_model, Mapping):
        raise ValueError(f"model must be a mapping, got {raw_model!r}")

    for key in raw_model:
        if key not in ("components", "precision"):
            notes.append(
                f"model.{key} dropped: the model section takes only components and precision."
            )

    used = set()
    components: List[Any] = []
    for entry in raw_model.get("components") or []:
        if isinstance(entry, Mapping) and len(entry) == 1:
            ((ref, existing_opts),) = entry.items()
            existing_opts = dict(existing_opts) if isinstance(existing_opts, Mapping) else {}
        else:
            ref = entry
            existing_opts = {}

        (cls,) = import_components([ref])
        config_cls = getattr(cls, "Config", EmptyComponentConfig)
        options: Dict[str, Any] = dict(existing_opts)
        for field, path in config_cls.legacy_keys.items():
            value = _lookup(old, path)
            if value is _ABSENT:
                if field in config_cls.legacy_absent_value and field not in options:
                    options[field] = config_cls.legacy_absent_value[field]
                    default = config_cls.model_fields[field].default
                    notes.append(
                        f"{'.'.join(path)} was unset, which meant {options[field]!r}; "
                        f"written out, since the default is now {default!r}."
                    )
                continue
            used.add(path)
            value = _migrate_null(config_cls, field, value)
            if value is _ABSENT:
                continue
            if value is not None and field in config_cls.legacy_transforms:
                value = config_cls.legacy_transforms[field](value)
            options.setdefault(field, value)
        name = canonical_component_ref(str(ref), cls)
        notes.extend(f"{name}.{note}" for note in config_cls.legacy_cleanup(options))
        components.append({name: options} if options else name)

    for section in _COMPONENT_SECTIONS:
        block = old.pop(section, None)
        if not isinstance(block, Mapping):
            continue
        for path in _leaves(block, (section,)):
            if path in used:
                continue
            dotted = ".".join(path)
            if path[-1] == "r_seed":
                notes.append(
                    f"{dotted} dropped: it had no effect; every draw comes from seed."
                )
            else:
                notes.append(
                    f"{dotted} dropped: no component in model.components reads it."
                )

    new_model: Dict[str, Any] = {}
    if raw_model.get("precision") is not None:
        new_model["precision"] = raw_model["precision"]
    new_model["components"] = components

    new: Dict[str, Any] = {"model": new_model}
    inference = old.pop("inference", None)
    if isinstance(inference, Mapping):
        if inference.get("seed") is not None:
            new["seed"] = inference["seed"]
        if inference.get("opt") is False:
            old.setdefault("opt", {})
            old["opt"] = {**(old["opt"] or {}), "max_iter": 0}
            notes.append("inference.opt: false is now opt.max_iter: 0, which runs no fit.")
        for key in set(inference) - {"seed", "opt"}:
            notes.append(f"inference.{key} dropped: the inference section takes no other option.")
    for name, section in old.items():
        section_cls = TOP_LEVEL_SECTIONS.get(name)
        if section_cls is not None and isinstance(section, Mapping):
            renamed = {section_cls.legacy_renames.get(key, key): value for key, value in section.items()}
            migrated_section: Dict[str, Any] = {}
            for key, value in renamed.items():
                value = _migrate_null(section_cls, key, value)
                if value is _ABSENT:
                    continue
                if value is not None and key in section_cls.legacy_transforms:
                    value = section_cls.legacy_transforms[key](value)
                migrated_section[key] = value
            section = migrated_section
            notes.extend(f"{name}.{note}" for note in section_cls.legacy_cleanup(section))
        new[name] = section
    return new, notes


def build_parser(parser=None):
    import argparse

    if parser is None:
        parser = argparse.ArgumentParser(
            prog="tabascal migrate-config",
            description=(
                "Convert a run config from the sectioned layout "
                "(ast/rfi/gains sections) to per-component options under model.components."
            ),
        )
    parser.add_argument("config", help="The config file to convert.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Where to write the converted config. Printed to stdout when omitted.",
    )
    return parser


def run(args) -> int:
    import yaml

    from tabascal.config import yaml_load
    from tabascal.schema import ConfigValidationError, validate_config

    new, notes = migrate(yaml_load(args.config))
    text = yaml.safe_dump(new, sort_keys=False, default_flow_style=False)

    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text)

    for note in notes:
        print(f"note: {note}", file=sys.stderr)

    try:
        validate_config(new)
    except ConfigValidationError as err:
        print(f"The converted config does not validate:\n{err}", file=sys.stderr)
        return 1
    return 0


def main():
    sys.exit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
