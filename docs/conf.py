import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "tabascal"
author = "Chris Finlay"
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
templates_path = ["_templates"]
exclude_patterns = []
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,  # keeps sidebar fixed
}
