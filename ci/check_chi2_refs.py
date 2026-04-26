#!/usr/bin/env python3
"""Pre-commit check: all PipelineTestConfig instances must have chi2_ref set.

Run manually:
    pixi run python ci/check_chi2_refs.py

Or automatically via pre-commit (see .pre-commit-config.yaml).
"""
import ast
import sys
from pathlib import Path

TARGET = Path("tests/test_tabascal_pipeline.py")


def _chi2_ref_missing(call: ast.Call) -> bool:
    chi2_kw = next((kw for kw in call.keywords if kw.arg == "chi2_ref"), None)
    return chi2_kw is None or (
        isinstance(chi2_kw.value, ast.Constant) and chi2_kw.value.value is None
    )


def _is_pipeline_config_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "PipelineTestConfig"
    )


def check(path: Path) -> tuple[list[str], list[str]]:
    tree = ast.parse(path.read_text())
    errors = []
    ids = []
    for node in ast.walk(tree):
        # Look for pytest.param(PipelineTestConfig(...), id="...")
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "param"
        ):
            continue
        config_call = next((a for a in node.args if _is_pipeline_config_call(a)), None)
        if config_call is None or not _chi2_ref_missing(config_call):
            continue
        id_kw = next((kw for kw in node.keywords if kw.arg == "id"), None)
        label = (
            id_kw.value.value
            if id_kw and isinstance(id_kw.value, ast.Constant)
            else f"line {config_call.lineno}"
        )
        errors.append(f"  {label} (line {config_call.lineno})")
        ids.append(label)
    return errors, ids


errors, ids = check(TARGET)
if errors:
    print(f"ERROR: unpopulated chi2_ref in {TARGET}:")
    for e in errors:
        print(e)
    print("\nRun each test to obtain its chi2_ref value, then set it before committing:")
    for test_id in ids:
        print(f"  pixi run -e dev pytest tests/test_tabascal_pipeline.py -k \"{test_id}\" -s 2>&1 | grep 'Chi^2'")
    sys.exit(1)

print(f"OK: all PipelineTestConfig chi2_ref values are set in {TARGET}")

