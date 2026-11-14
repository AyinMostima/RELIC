"""Sphinx configuration for RELIC documentation.

This repository uses Markdown (MyST) and reStructuredText sources.
"""

from __future__ import annotations

import os

project = "RELIC"
author = "RELIC Contributors"
copyright = "2026, RELIC Contributors"
html_title = "RELIC Documentation"
language = "en"

extensions = [
    "myst_parser",
    "sphinxcontrib.mermaid",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

root_doc = "index"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

exclude_patterns = [
    "_build",
    # Internal execution artifacts are not part of the architecture site.
    "RELIC_IPHS_CONTINUATION_PROMPT.md",
    "RELIC_REPAIR_MASTER_PLAN_V2.md",
    "phase8_scientific_iphs_standardization_plan.md",
    "three_papers_framework_summary.md",
    "typed_cochain_morphism_enhancement_plan.md",
    # The public site has its own root document and Sphinx configuration.
    "public/**",
]

suppress_warnings = [
    "misc.highlighting_failure",
    "myst.header",
]

html_theme = "sphinx_rtd_theme"
html_search_language = "en"
html_copy_source = False
html_show_sourcelink = False
html_use_index = False

# MyST configuration
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "tasklist",
]
myst_heading_anchors = 3

# Mermaid is rendered through the existing Windows mmdc command-script owner.
# SVG keeps architecture labels and status strokes inspectable in built docs.
mermaid_output_format = "svg"
mermaid_cmd = os.environ.get(
    "RELIC_MERMAID_CMD",
    "mmdc.cmd" if os.name == "nt" else "mmdc",
)
mermaid_cmd_shell = "True" if os.name == "nt" else "False"
