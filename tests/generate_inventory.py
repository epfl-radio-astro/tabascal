#!/usr/bin/env python3
"""
Generate tests/TEST_INVENTORY.md from test file docstrings.

Run from the tabascal/ directory:
    python tests/generate_inventory.py

Each test function's one-line docstring becomes the Description column.
Parametrised variants share their function's docstring.
File-level and class-level notes live in FILE_NOTES / CLASS_NOTES below;
update those when the intent behind a test file or class changes.
"""

import ast
import re
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

TESTS_DIR = Path(__file__).parent

# Ordered list of test files to include in the inventory.
FILE_ORDER = [
    "components/test_trajectory.py",
    "components/test_gains.py",
    "components/test_rfi_vis.py",
    "test_fft_gp.py",
    "test_timing.py",
    "test_tabascal_pipeline.py",
]

# Prose paragraph(s) inserted after each file's `##` heading.
FILE_NOTES: dict[str, str] = {
    "components/test_trajectory.py": (
        "No Space-Track credentials are required for any test in this file. "
        "`TestSGP4LEONoDragOrbit` and `TestSGP4LEOOrbit` use a bundled TLE cache file "
        "(`tabascal/data/tles/2026-04-15-bundled.json`, NORAD IDs 20452 / 38833) and a "
        "fixed observation epoch matching that file's date prefix, so `get_tles_by_id` "
        "reads from disk and never contacts the Space-Track API."
    ),
    "components/test_rfi_vis.py": (
        "Tests for `RiemannVisTimeFreqCalculation` and `RiemannVisTimeFreqCalculationFFI`. "
        "Each parametrized over three size tuples: `(1,1,1,1,1,1)`, `(4,5,6,7,8,9)`, "
        "`(64,20,16,12,4,2)`."
    ),
    "test_tabascal_pipeline.py": (
        "End-to-end integration tests. Each test invokes `run_tabascal.py` as a subprocess, "
        "checks `returncode == 0`, and validates the `Reduced Chi^2 @ opt params` value "
        "printed to stdout.\n\n"
        "`TabConfig` fetches TLEs using the mean observation epoch from the MS file. The "
        "simulation data (from HuggingFace) uses the tabsim default epoch of 2023-02-21; "
        "the repo ships `tabascal/data/tles/2023-02-21-HMZGLE.json` containing NORAD IDs "
        "20452, 38833, and 45854 (all three listed in `tests/data/tab_target.yaml`), so the "
        "`TabConfig` TLE lookup is always satisfied from disk. Tests that additionally use "
        "`SGP4LEONoDragOrbit` or `SGP4LEOOrbit` call `fetch_standard_orbital_elements` a "
        "second time during component `setup`; that second call is also satisfied by the same "
        "bundled cache file — but only when the component's observation epoch likewise falls "
        "on 2023-02-21. Tests marked **requires Space-Track** are skipped automatically when "
        "`tabascal.tle.load_spacetrack_credentials` returns `(None, None)`."
    ),
}

# Prose paragraph inserted after each class's `###` heading, keyed by class name.
CLASS_NOTES: dict[str, str] = {
    "TestSGP4LEONoDragOrbit": "Uses the bundled TLE cache; no Space-Track credentials required.",
    "TestSGP4LEOOrbit": "Uses the bundled TLE cache; no Space-Track credentials required.",
}


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

@dataclass
class FuncInfo:
    lineno: int
    docstring: str  # first line of docstring, or empty string


def parse_test_file(path: Path) -> dict[Optional[str], dict[str, FuncInfo]]:
    """
    Return {class_name_or_None: {func_name: FuncInfo}} for a test file.
    Only direct class methods and top-level functions starting with 'test_' are included.
    """
    tree = ast.parse(path.read_text())
    result: dict[Optional[str], dict[str, FuncInfo]] = {None: {}}

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods: dict[str, FuncInfo] = {}
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test_"):
                    raw = ast.get_docstring(child) or ""
                    first_line = raw.strip().splitlines()[0] if raw.strip() else ""
                    methods[child.name] = FuncInfo(child.lineno, first_line)
            result[node.name] = methods
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            raw = ast.get_docstring(node) or ""
            first_line = raw.strip().splitlines()[0] if raw.strip() else ""
            result[None][node.name] = FuncInfo(node.lineno, first_line)

    return result


# ---------------------------------------------------------------------------
# pytest collection helpers
# ---------------------------------------------------------------------------

def collect_test_ids(rel_path: str) -> list[str]:
    """
    Run pytest --collect-only on one file and return node IDs.
    IDs are returned with rel_path as the file prefix (i.e. relative to tests/).
    """
    project_root = TESTS_DIR.parent
    full_rel = f"tests/{rel_path}"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "--no-header", full_rel],
        capture_output=True,
        text=True,
        cwd=project_root,
    )
    ids = []
    prefix = full_rel.replace("\\", "/")
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith(prefix) and "::" in line:
            # Strip the "tests/" prefix so IDs are relative to tests/ for link building
            ids.append(line[len("tests/"):])
    return ids


def parse_test_id(test_id: str):
    """
    Parse a pytest node ID (relative to tests/) into
    (rel_file, class_name_or_None, func_name, param_suffix_or_None).
    Returns None if the ID cannot be parsed.
    """
    parts = test_id.split("::")
    if len(parts) == 2:
        rel_file, func_part = parts
        class_name = None
    elif len(parts) == 3:
        rel_file, class_name, func_part = parts
    else:
        return None

    m = re.match(r"^(\w+)(\[.*\])?$", func_part)
    if not m:
        return None

    return rel_file, class_name, m.group(1), m.group(2)  # param_suffix includes brackets


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------

def make_link(rel_file: str, func_name: str, param_suffix: Optional[str], lineno: int) -> str:
    display = f"`{func_name}{param_suffix}`" if param_suffix else f"`{func_name}`"
    return f"[{display}]({rel_file}#L{lineno})"


def generate() -> str:
    lines: list[str] = ["# Test Inventory\n"]

    for rel_file in FILE_ORDER:
        path = TESTS_DIR / rel_file
        ast_info = parse_test_file(path)
        test_ids = collect_test_ids(rel_file)

        lines.append(f"## `tests/{rel_file}`\n")
        if rel_file in FILE_NOTES:
            lines.append(FILE_NOTES[rel_file] + "\n")

        # Group collected IDs by class, preserving order
        classes: OrderedDict[Optional[str], list[tuple[str, Optional[str]]]] = OrderedDict()
        for test_id in test_ids:
            parsed = parse_test_id(test_id)
            if parsed is None:
                continue
            _, class_name, func_name, param_suffix = parsed
            classes.setdefault(class_name, []).append((func_name, param_suffix))

        for class_name, func_list in classes.items():
            if class_name:
                lines.append(f"\n### {class_name}\n")
                if class_name in CLASS_NOTES:
                    lines.append(CLASS_NOTES[class_name] + "\n")

            lines.append("\n| Test name | Description |")
            lines.append("|-----------|-------------|")

            for func_name, param_suffix in func_list:
                func_info = (ast_info.get(class_name) or {}).get(func_name)
                if func_info:
                    description = func_info.docstring or "*(no docstring)*"
                    lineno = func_info.lineno
                else:
                    description = "*(no docstring)*"
                    lineno = 0

                link = make_link(rel_file, func_name, param_suffix, lineno)
                lines.append(f"| {link} | {description} |")

        lines.append("\n---\n")

    # Remove trailing separator
    while lines and lines[-1].strip() in ("---", ""):
        lines.pop()

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    content = generate()
    out = TESTS_DIR / "TEST_INVENTORY.md"
    out.write_text(content)
    print(f"Written {out}")
