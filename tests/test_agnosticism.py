"""The committed tree names no telescope, observation or cluster of its own.

TABASCAL is written for any MS-based interferometric observation, and the
sources, docs, examples and tests are meant to read that way. A number measured
on one array is worth quoting as a measurement; quoting it as a fact about *the*
array turns a general tool into that array's tool, and a machine the authors
happen to develop on is not a property of the software at all. Review catches
most of that; a grep catches the rest, which is what this is.

``ci/`` is deliberately outside the scanned roots. Those files configure real
infrastructure -- a GitLab CI runner, a ReFrame system name, a bencher testbed
-- where the machine name *is* the identifier and renaming it would break the
job rather than generalise it. ``docs/performance.md`` documents that same
infrastructure and is allowed for the same reason; it is the only exception.
"""

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).parent.parent

# Everything that ships or is published. ``ci/`` is excluded on purpose -- see
# the module docstring.
_ROOTS = ("tabascal", "docs", "tests", "examples", "README.md", "pyproject.toml")

# Build artefacts and caches that live inside the roots but are not the tree.
_SKIP_DIRS = frozenset({"__pycache__", ".pytest_cache", "_build", ".ipynb_checkpoints"})

_FORBIDDEN = (
    # A telescope some of this work was validated against. "On <array> SIGMA
    # spans a factor of ~30" is a measurement and reads fine without the name.
    re.compile(r"(?<![A-Za-z0-9])eda[-_ ]?2(?![0-9])", re.IGNORECASE),
    # The cluster and the computing centre the project benchmarks on. Facts
    # about a machine, not about TABASCAL.
    re.compile(r"(?<![A-Za-z])daint(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])cscs(?![A-Za-z])", re.IGNORECASE),
)

# Paths relative to the repository root that may name them anyway. Keep this
# short, and say why for every entry.
_ALLOWED = {
    # Documents the ReFrame performance checks, which run on one specific
    # partition under one specific CI project: the names are the identifiers
    # of `ci/cscs.yml` and of the bencher testbed the history is stored under.
    "docs/performance.md",
}

# This file has to spell the tokens out to look for them.
_SELF = Path(__file__).relative_to(_REPO).as_posix()


def scanned_files(repo=_REPO):
    """Every file under the published roots, minus caches and the allowlist."""

    for root in _ROOTS:
        path = repo / root
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(p for p in path.rglob("*") if p.is_file())
        else:  # pragma: no cover - the roots are all tracked
            continue
        for file in candidates:
            rel = file.relative_to(repo).as_posix()
            if set(file.relative_to(repo).parts) & _SKIP_DIRS:
                continue
            if rel in _ALLOWED or rel == _SELF:
                continue
            yield rel, file


def _hits(text, rel):
    for lineno, line in enumerate(text.splitlines(), start=1):
        for pattern in _FORBIDDEN:
            if pattern.search(line):
                yield f"{rel}:{lineno}: {line.strip()[:120]}"
                break


def test_the_roots_are_all_present():
    """A typo in ``_ROOTS`` would make the sweep pass by scanning nothing."""

    for root in _ROOTS:
        assert (_REPO / root).exists(), f"{root} is not in the repository"

    assert any(True for _ in scanned_files()), "the scan found no files at all"


def test_no_file_contents_name_a_telescope_or_a_cluster():
    found = []
    for rel, file in scanned_files():
        text = file.read_text(encoding="utf-8", errors="replace")
        found.extend(_hits(text, rel))

    assert not found, (
        "TABASCAL's tree stays telescope- and observation-agnostic; these lines "
        "name a specific array, cluster or computing centre. Rewrite them "
        "generically (quote the measurement, not the array), or -- if the name "
        "is a load-bearing identifier -- add the file to _ALLOWED with a "
        "reason:\n  " + "\n  ".join(found)
    )


def test_no_file_paths_name_a_telescope_or_a_cluster():
    """A generic caption over ``images/<array>_result.svg`` is still a hit."""

    found = [
        rel
        for rel, _ in scanned_files()
        if any(pattern.search(rel) for pattern in _FORBIDDEN)
    ]

    assert not found, "these paths name a specific array or cluster:\n  " + "\n  ".join(
        found
    )


@pytest.mark.parametrize(
    "text, expected",
    [
        ("EDA2", True),
        ("eda2's per-baseline SIGMA", True),
        ("eda-2", True),
        ("the EDA 2 array", True),
        ("images/eda2_starlink.svg", True),
        ("daint:gpu", True),
        ("Piz Daint", True),
        ("cscs-daint-gh200", True),
        # Not hits: the tokens have to stand on their own.
        ("Alameda 2000", False),
        ("edam", False),
        ("the median absolute deviation", False),
        ("wideband", False),
    ],
)
def test_the_patterns_match_the_names_and_nothing_near_them(text, expected):
    assert any(pattern.search(text) for pattern in _FORBIDDEN) is expected
