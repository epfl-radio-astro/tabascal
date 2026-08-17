"""Sphinx configuration.

Built locally with ``pixi run -e dev docs-build`` and on Read the Docs from
``.readthedocs.yaml``. See docs/readthedocs.md for how versions are published.
"""

import os
import sys
from datetime import date
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

sys.path.insert(0, os.path.abspath(".."))

project = "tabascal"
author = "Chris Finlay"
copyright = f"2025-{date.today():%Y}, {author}"

# The version shown in the sidebar and page titles. Release builds are made from
# a vX.Y.Z tag, so the installed package metadata is the source of truth. Branch
# builds (`latest` from main, plus any other activated branch) report the version
# in pyproject.toml, which is the release being worked towards rather than one
# that exists -- so fall back to the Read the Docs version name to keep those
# pages distinguishable from a real release.
try:
    release = package_version("tabascal")
except PackageNotFoundError:  # docs built against an uninstalled source tree
    release = os.environ.get("READTHEDOCS_VERSION", "unknown")
else:
    rtd_version = os.environ.get("READTHEDOCS_VERSION", "")
    if rtd_version and os.environ.get("READTHEDOCS_VERSION_TYPE") == "branch":
        release = f"{release} ({rtd_version})"
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.graphviz",
]
# Allow markdown files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
# Optional: enable some MyST extras (tables, figures, math, etc.)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
]
# Generate anchors for headings up to level 3 so cross-page links of the form
# `[text](other.md#a-heading)` resolve. Without this they build as broken
# references, which fails the -W builds that CI and Read the Docs both run.
myst_heading_anchors = 3
templates_path = ["_templates"]
exclude_patterns = ["_build"]
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,  # keeps sidebar fixed
}

# Read the Docs serves several versions of these docs at once. It injects the
# version selector and the "you are reading an old version" notification through
# its addons script; the flag below tells the theme to make room for them, and
# the canonical URL points search engines at the default version rather than at
# whichever version they crawled.
if os.environ.get("READTHEDOCS") == "True":
    html_context = {"READTHEDOCS": True}
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
