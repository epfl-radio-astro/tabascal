"""The committed tree names no telescope, observation or cluster of its own.

TABASCAL is written for any MS-based interferometric observation, and the
sources, docs, examples and tests are meant to read that way. A number measured
on one array is worth quoting as a measurement; quoting it as a fact about *the*
array turns a general tool into that array's tool, and a machine the authors
happen to develop on is not a property of the software at all. Review catches
most of that; a grep catches the rest, which is what this is.

Two standing rules, both encoded below rather than left to judgement:

* **Generic first, instrument in parentheses.** A page that shows one
  instrument's actual data may name it -- "real data from a low-frequency
  aperture array (EDA2)" -- because the reader is entitled to know what they are
  looking at. The generic description leads and the name goes in brackets after
  it; bare "on EDA2" prose still fails, on the showcase pages as everywhere else.
* **A machine name may be an identifier.** ``ci/`` and ``docs/performance.md``
  configure and document real infrastructure -- a GitLab runner, a ReFrame
  system, a bencher testbed -- where the cluster name *is* the identifier and
  renaming it breaks the job rather than generalising it. They are exempt from
  the cluster names only; the array name is still forbidden there.

The scan is over what ``git ls-files`` reports, not over the working tree: an
untracked scratch file is not something a reviewer can be asked to rewrite, and
a new top-level directory should be covered the day it is committed rather than
the day someone remembers to add it here.
"""

import re
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).parent.parent

# The array some of this work was validated against. "On <array> SIGMA spans a
# factor of ~30" is a measurement and reads fine without the name.
_ARRAY = re.compile(r"(?<![A-Za-z0-9])eda[-_ ]?2(?![0-9])", re.IGNORECASE)

# The cluster and the computing centre the project benchmarks on. Facts about a
# machine, not about TABASCAL. The lookbehind keeps a base64 blob -- a
# regenerated SVG, an embedded asset -- from reading as a name.
_CLUSTER = (
    re.compile(r"(?<![A-Za-z0-9])daint(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9])cscs(?![A-Za-z])", re.IGNORECASE),
)

# Pages carrying a figure made from one instrument's actual data, which may
# therefore name it in parentheses after the generic description. Keep this
# short: it is for content that *shows* the instrument's data, not for prose
# that merely mentions it.
_ARRAY_IN_PARENTHESES_OK = {
    "README.md",
    "docs/example.md",
}

# Where the cluster names are load-bearing identifiers rather than references.
_CLUSTER_EXEMPT_PREFIXES = (
    # The CI pipeline, the ReFrame system definition and the performance
    # references keyed by partition: `daint:gpu`, `.container-runner-daint-gh200`,
    # `$CSCS_REGISTRY_PATH`.
    "ci/",
)
_CLUSTER_EXEMPT_FILES = {
    # Documents the above, including the bencher.dev testbed `cscs-daint-gh200`
    # the benchmark history is stored under.
    "docs/performance.md",
}

_PARENTHESISED = re.compile(r"\(([^()]*)\)")

# This file has to spell the names out to look for them.
_SELF = Path(__file__).relative_to(_REPO).as_posix()


def tracked_files(repo=_REPO):
    """Every file git tracks, as repository-relative posix paths."""

    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "-z"],
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as e:  # pragma: no cover
        pytest.skip(f"not a git checkout, so there is no committed tree to scan: {e}")
    return [rel for rel in out.decode().split("\0") if rel and rel != _SELF]


def cluster_patterns_for(rel):
    """The cluster names forbidden in this file, which may be none of them."""

    if rel in _CLUSTER_EXEMPT_FILES or rel.startswith(_CLUSTER_EXEMPT_PREFIXES):
        return ()
    return _CLUSTER


def _array_offends(line, rel):
    """Is the array named here in a way the generic-first rule does not allow?"""

    allowed = (
        [m.span(1) for m in _PARENTHESISED.finditer(line)]
        if rel in _ARRAY_IN_PARENTHESES_OK
        else []
    )
    return any(
        not any(lo <= m.start() and m.end() <= hi for lo, hi in allowed)
        for m in _ARRAY.finditer(line)
    )


def _hits(text, rel):
    cluster = cluster_patterns_for(rel)
    for lineno, line in enumerate(text.splitlines(), start=1):
        if _array_offends(line, rel) or any(p.search(line) for p in cluster):
            yield f"{rel}:{lineno}: {line.strip()[:120]}"


def test_the_scan_actually_reaches_the_tree():
    """A broken ``git ls-files`` would let the sweep pass by scanning nothing."""

    tracked = tracked_files()

    assert len(tracked) > 50, "the scan found almost no files"
    for expected in ("README.md", "tabascal/noise.py", "docs/config.md"):
        assert expected in tracked, f"{expected} is not in the scan"


def test_no_file_contents_name_a_telescope_or_a_cluster():
    found = []
    for rel in tracked_files():
        raw = (_REPO / rel).read_bytes()
        if b"\0" in raw:  # a binary asset, not prose
            continue
        found.extend(_hits(raw.decode("utf-8", errors="replace"), rel))

    assert not found, (
        "TABASCAL's tree stays telescope- and observation-agnostic; these lines "
        "name a specific array, cluster or computing centre. Lead with the "
        "generic description and quote the measurement, not the array. A page "
        "showing one instrument's actual data may name it in parentheses after "
        "the generic phrase; a load-bearing machine identifier is exempted by "
        "file, with a reason:\n  " + "\n  ".join(found)
    )


def test_no_file_paths_name_a_telescope_or_a_cluster():
    """A generic caption over ``images/<array>_result.svg`` is still a hit.

    Paths get no parenthesis rule -- there is nothing in a filename for the
    generic description to lead with.
    """

    found = [
        rel
        for rel in tracked_files()
        if _ARRAY.search(rel) or any(p.search(rel) for p in cluster_patterns_for(rel))
    ]

    assert not found, "these paths name a specific array or cluster:\n  " + "\n  ".join(
        found
    )


@pytest.mark.parametrize(
    "text, expected",
    [
        ("EDA2", True),
        ("eda2's", True),
        ("eda-2", True),
        ("eda 2", True),
        ("images/eda2_starlink.svg", True),
        ("daint:gpu", True),
        ("cscs-daint-gh200", True),
        # Not hits: a name has to stand on its own, and a character before one
        # makes it a fragment -- a longer word, or a base64 chunk.
        ("Alameda 2000", False),
        ("edam", False),
        ("7daint", False),
        ("9cscs", False),
        ("the median absolute deviation", False),
        ("wideband", False),
    ],
)
def test_the_patterns_match_the_names_and_nothing_near_them(text, expected):
    assert any(p.search(text) for p in (_ARRAY, *_CLUSTER)) is expected


def test_the_cluster_exemption_does_not_extend_to_the_array_name():
    """``ci/`` may name the machine it runs on. It may not name the array."""

    for rel in ("ci/cscs.yml", "ci/reframe/data/tab_target.yaml", "docs/performance.md"):
        assert list(_hits("runs on daint:gpu at CSCS", rel)) == []
        assert list(_hits("recorded on EDA2", rel)) != []


class TestTheGenericFirstRule:
    """The showcase pages may bracket the instrument, not lead with it."""

    def test_a_parenthesised_instrument_is_allowed_where_the_data_is_shown(self):
        line = "real data from a low-frequency aperture array (EDA2): 151 MHz"

        assert list(_hits(line, "README.md")) == []
        assert list(_hits(line, "docs/example.md")) == []

    def test_bare_prose_still_fails_on_those_same_pages(self):
        assert list(_hits("A result on real EDA2 data", "README.md")) != []
        assert list(_hits("the EDA2 observation", "docs/example.md")) != []

    def test_the_allowance_does_not_leak_to_other_files(self):
        line = "measured on a low-frequency array (EDA2)"

        assert list(_hits(line, "docs/config.md")) != []
        assert list(_hits(line, "tabascal/noise.py")) != []

    def test_the_showcase_pages_are_still_held_to_the_cluster_names(self):
        assert list(_hits("benchmarked on daint", "README.md")) != []
