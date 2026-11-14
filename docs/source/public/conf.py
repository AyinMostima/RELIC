"""Sphinx configuration for the public RELIC documentation."""

from __future__ import annotations

import os
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility for sphinx_env.
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[3]
with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
    release = tomllib.load(stream)["project"]["version"]

project = "RELIC"
author = "RELIC Contributors"
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
]
if os.environ.get("RELIC_ENABLE_SPHINX_LLM") == "1":
    extensions.append("sphinx_llm.txt")

root_doc = "index"
language = os.environ.get("RELIC_DOC_LANGUAGE", "en")
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
exclude_patterns = []

locale_dirs = [str(Path(__file__).resolve().parents[2] / "locale")]
gettext_compact = False
gettext_allow_fuzzy_translations = False

html_theme = "sphinx_rtd_theme"
html_title = f"RELIC {release} Documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["language_switch.js"]
html_copy_source = False
html_show_sourcelink = False
html_use_index = False
html_domain_indices = False
html_search_language = None
html_last_updated_fmt = None
html_show_sphinx = False
html_show_copyright = False

html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
}

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
]
myst_heading_anchors = 3

# sphinx-llm is enabled only by docs\build_rag_docs.ps1. The sequential
# Markdown build keeps the HTML-equivalent corpus deterministic on Windows.
llms_txt_description = (
    "RELIC biogeochemical simulation architecture, scientific theory, "
    "class reference, and operational workflows."
    if language == "en"
    else "RELIC 生物地球化学模拟架构、科学理论、类参考与操作流程。"
)
llms_txt_build_parallel = False
llms_txt_suffix_mode = "file-suffix"
llms_txt_full_build = True
markdown_flavor = "github"
markdown_uri_doc_suffix = ".html.md"
