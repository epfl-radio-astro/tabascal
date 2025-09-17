from __future__ import annotations
import importlib
from typing import Iterable, List, Type


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

            module = None
            if base_package:
                try:
                    module = importlib.import_module(f"{base_package}.{mod_path}")
                except ModuleNotFoundError:
                    # fall back to absolute import (supports 'tabascal.components.foo:Foo' too)
                    pass
            if module is None:
                module = importlib.import_module(mod_path)

            try:
                cls = getattr(module, cls_name)
            except AttributeError as e:
                raise ImportError(
                    f"Class '{cls_name}' not found in module '{module.__name__}' "
                    f"(from reference '{ref}')."
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
