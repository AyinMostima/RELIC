"""Validate sphinx-llm output against the published RELIC HTML site."""

from __future__ import annotations

import argparse
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit


LANGUAGES = ("en", "zh")
CJK_PATTERN = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]")
HAN_PATTERN = re.compile(r"[\u3400-\u9fff]")
LINK_PATTERN = re.compile(r"(?<!!)\]\(([^)]+)\)")
LINK_TARGET_PATTERN = re.compile(r"(?<=\]\()[^)\r\n]+(?=\))")
SITEMAP_LINK_PATTERN = re.compile(r"(?<!!)\]\(([^)]+\.html\.md)\)")
IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]+\)")
FENCE_PATTERN = re.compile(r"^\s*(?P<fence>`{3,}|~{3,})")
TABLE_SEPARATOR_CELL_PATTERN = re.compile(r"^\s*:?-{3,}:?\s*$")
BANNED_PATTERN = re.compile(
    r"pytest|test[\\/]|sphinx-build|doctree|linkcheck|"
    r"phase8_scientific_iphs_standardization_plan|"
    r"typed_cochain_morphism_enhancement_plan",
    re.IGNORECASE,
)


def _relative_files(root: Path, pattern: str) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob(pattern)
        if path.is_file()
    }


def _split_markdown_pipe_row(line: str) -> list[str] | None:
    stripped = line.strip()
    if len(stripped) < 2 or not stripped.startswith("|") or not stripped.endswith("|"):
        return None
    cells: list[str] = []
    current: list[str] = []
    backslash_run = 0
    for character in stripped[1:-1]:
        if character == "|" and backslash_run % 2 == 0:
            cells.append("".join(current))
            current = []
            backslash_run = 0
            continue
        current.append(character)
        backslash_run = backslash_run + 1 if character == "\\" else 0
    cells.append("".join(current))
    return cells


def _markdown_tables(text: str, source: Path) -> list[list[list[str]]]:
    lines = text.splitlines()
    tables: list[list[list[str]]] = []
    active_fence_character: str | None = None
    active_fence_length = 0
    index = 0
    while index < len(lines):
        line = lines[index]
        fence_match = FENCE_PATTERN.match(line)
        if active_fence_character is not None:
            if (
                fence_match is not None
                and fence_match.group("fence")[0] == active_fence_character
                and len(fence_match.group("fence")) >= active_fence_length
            ):
                active_fence_character = None
                active_fence_length = 0
            index += 1
            continue
        if fence_match is not None:
            fence = fence_match.group("fence")
            active_fence_character = fence[0]
            active_fence_length = len(fence)
            index += 1
            continue
        if index + 1 >= len(lines):
            break

        header = _split_markdown_pipe_row(line)
        separator = _split_markdown_pipe_row(lines[index + 1])
        if (
            header is None
            or separator is None
            or not separator
            or not all(
                TABLE_SEPARATOR_CELL_PATTERN.fullmatch(cell)
                for cell in separator
            )
        ):
            index += 1
            continue
        if len(header) != len(separator):
            raise ValueError(
                f"Markdown table header width mismatch in {source}:{index + 1}; "
                f"header={len(header)} separator={len(separator)}"
            )

        expected_width = len(separator)
        rows = [header]
        row_index = index + 2
        while row_index < len(lines):
            row = _split_markdown_pipe_row(lines[row_index])
            if row is None:
                break
            if len(row) != expected_width:
                raise ValueError(
                    f"Markdown table width mismatch in {source}:{row_index + 1}; "
                    f"expected={expected_width} actual={len(row)}"
                )
            rows.append(row)
            row_index += 1
        tables.append(rows)
        index = row_index
    return tables


class _HtmlTableParser(HTMLParser):
    def __init__(self, source: Path) -> None:
        super().__init__(convert_charrefs=True)
        self.source = source
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        if tag == "table":
            if self._table is not None:
                raise ValueError(f"Nested HTML table in {self.source}")
            self._table = []
        elif tag == "tr" and self._table is not None:
            if self._row is not None:
                raise ValueError(f"Nested HTML table row in {self.source}")
            self._row = []
        elif tag in {"th", "td"} and self._row is not None:
            if self._cell is not None:
                raise ValueError(f"Nested HTML table cell in {self.source}")
            colspan = int(attributes.get("colspan") or "1")
            rowspan = int(attributes.get("rowspan") or "1")
            if colspan != 1 or rowspan != 1:
                raise ValueError(
                    f"RAG pipe tables cannot preserve span attributes in "
                    f"{self.source}: colspan={colspan} rowspan={rowspan}"
                )
            self._cell = []
        elif self._cell is not None and tag in {"br", "div", "li", "p"}:
            self._cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"} and self._cell is not None:
            assert self._row is not None
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if not self._row:
                raise ValueError(f"Empty HTML table row in {self.source}")
            assert self._table is not None
            self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            if self._row is not None or self._cell is not None:
                raise ValueError(f"Unclosed HTML table structure in {self.source}")
            if not self._table:
                raise ValueError(f"Empty HTML table in {self.source}")
            self.tables.append(self._table)
            self._table = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def validate_closed(self) -> None:
        if self._table is not None or self._row is not None or self._cell is not None:
            raise ValueError(f"Unclosed HTML table structure in {self.source}")


def _html_tables(path: Path) -> list[list[list[str]]]:
    parser = _HtmlTableParser(path)
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    parser.validate_closed()
    return parser.tables


def _validate_page_tables(
    html_path: Path,
    markdown_path: Path,
    markdown_text: str,
) -> dict[str, int]:
    html_tables = _html_tables(html_path)
    markdown_tables = _markdown_tables(markdown_text, markdown_path)
    if len(html_tables) != len(markdown_tables):
        raise ValueError(
            f"HTML/RAG table count mismatch for {markdown_path}; "
            f"html={len(html_tables)} rag={len(markdown_tables)}"
        )

    row_count = 0
    cell_count = 0
    encoded_pipe_count = 0
    for table_index, (html_table, markdown_table) in enumerate(
        zip(html_tables, markdown_tables, strict=True),
        start=1,
    ):
        if len(html_table) != len(markdown_table):
            raise ValueError(
                f"HTML/RAG row count mismatch for {markdown_path}, "
                f"table={table_index}; html={len(html_table)} "
                f"rag={len(markdown_table)}"
            )
        for row_index, (html_row, markdown_row) in enumerate(
            zip(html_table, markdown_table, strict=True),
            start=1,
        ):
            if len(html_row) != len(markdown_row):
                raise ValueError(
                    f"HTML/RAG cell count mismatch for {markdown_path}, "
                    f"table={table_index}, row={row_index}; "
                    f"html={len(html_row)} rag={len(markdown_row)}"
                )
            row_count += 1
            cell_count += len(markdown_row)
            for cell_index, (html_cell, markdown_cell) in enumerate(
                zip(html_row, markdown_row, strict=True),
                start=1,
            ):
                if not markdown_cell.strip():
                    raise ValueError(
                        f"Empty RAG table cell in {markdown_path}, "
                        f"table={table_index}, row={row_index}, cell={cell_index}"
                    )
                if r"\|" in markdown_cell:
                    raise ValueError(
                        f"Non-portable escaped pipe in RAG table cell: "
                        f"{markdown_path}, table={table_index}, "
                        f"row={row_index}, cell={cell_index}"
                    )
                decoded_markdown_cell = unescape(markdown_cell)
                if decoded_markdown_cell.count("|") != html_cell.count("|"):
                    raise ValueError(
                        f"HTML/RAG data-pipe mismatch for {markdown_path}, "
                        f"table={table_index}, row={row_index}, cell={cell_index}"
                    )
                encoded_pipe_count += markdown_cell.count("&#124;")
    return {
        "tables": len(markdown_tables),
        "rows": row_count,
        "cells": cell_count,
        "encoded_pipes": encoded_pipe_count,
    }


def _validate_local_links(
    page_path: Path,
    text: str,
    language_root: Path,
) -> None:
    if IMAGE_PATTERN.search(text):
        raise ValueError(f"Unresolved image link in RAG text: {page_path}")
    for raw_target in LINK_PATTERN.findall(text):
        if "\\" in raw_target:
            raise ValueError(
                f"Non-portable backslash in RAG link: {page_path}: {raw_target}"
            )
        target = raw_target.strip().strip("<>")
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or target.startswith("#"):
            continue
        local_path = unquote(parsed.path)
        if not local_path:
            continue
        candidate = (
            language_root / local_path.lstrip("/")
            if local_path.startswith("/")
            else page_path.parent / local_path
        ).resolve()
        language_resolved = language_root.resolve()
        if language_resolved not in candidate.parents and candidate != language_resolved:
            raise ValueError(f"RAG link escapes language root: {page_path}: {target}")
        if not candidate.is_file():
            raise ValueError(f"Broken local RAG link: {page_path}: {target}")


def _rewrite_links_for_corpus_root(
    page_path: Path,
    text: str,
    language_root: Path,
) -> str:
    language_resolved = language_root.resolve()

    def replace_target(match: re.Match[str]) -> str:
        raw_target = match.group(0)
        wrapped = raw_target.startswith("<") and raw_target.endswith(">")
        target = raw_target[1:-1] if wrapped else raw_target
        parsed = urlsplit(target)
        if (
            parsed.scheme
            or parsed.netloc
            or target.startswith("#")
            or not parsed.path
        ):
            return raw_target
        local_path = unquote(parsed.path)
        candidate = (
            language_resolved / local_path.lstrip("/")
            if local_path.startswith("/")
            else page_path.parent / local_path
        ).resolve()
        if (
            language_resolved not in candidate.parents
            and candidate != language_resolved
        ):
            raise ValueError(
                f"RAG link escapes language root: {page_path}: {target}"
            )
        portable_target = candidate.relative_to(language_resolved).as_posix()
        if parsed.query:
            portable_target += f"?{parsed.query}"
        if parsed.fragment:
            portable_target += f"#{parsed.fragment}"
        return f"<{portable_target}>" if wrapped else portable_target

    return LINK_TARGET_PATTERN.sub(replace_target, text)


def _validate_language(
    html_root: Path,
    rag_root: Path,
    language: str,
) -> dict[str, int]:
    html_language_root = html_root / language
    rag_language_root = rag_root / language
    if not html_language_root.is_dir():
        raise ValueError(f"Missing HTML language directory: {html_language_root}")
    if not rag_language_root.is_dir():
        raise ValueError(f"Missing RAG language directory: {rag_language_root}")

    html_pages = _relative_files(html_language_root, "*.html")
    html_pages.discard("search.html")
    rag_pages = {
        relative.removesuffix(".md")
        for relative in _relative_files(rag_language_root, "*.html.md")
    }
    if html_pages != rag_pages:
        missing = sorted(html_pages - rag_pages)
        unexpected = sorted(rag_pages - html_pages)
        raise ValueError(
            f"{language} RAG page mismatch; missing={missing}, "
            f"unexpected={unexpected}"
        )

    sitemap_path = rag_language_root / "llms.txt"
    full_path = rag_language_root / "llms-full.txt"
    for required in (sitemap_path, full_path):
        if not required.is_file() or required.stat().st_size == 0:
            raise ValueError(f"Missing or empty RAG artifact: {required}")

    sitemap = sitemap_path.read_text(encoding="utf-8")
    full_text = full_path.read_text(encoding="utf-8")
    _validate_local_links(sitemap_path, sitemap, rag_language_root)
    if "\\" in "".join(LINK_PATTERN.findall(full_text)):
        raise ValueError(f"Non-portable backslash in {language} llms-full.txt link")
    if IMAGE_PATTERN.search(full_text):
        raise ValueError(f"Unresolved image link in {language} llms-full.txt")
    _validate_local_links(full_path, full_text, rag_language_root)
    sitemap_pages = {
        target
        for target in SITEMAP_LINK_PATTERN.findall(sitemap)
    }
    expected_markdown_pages = {
        f"{relative}.md"
        for relative in html_pages
    }
    if sitemap_pages != expected_markdown_pages:
        missing = sorted(expected_markdown_pages - sitemap_pages)
        unexpected = sorted(sitemap_pages - expected_markdown_pages)
        raise ValueError(
            f"{language} llms.txt mismatch; missing={missing}, "
            f"unexpected={unexpected}"
        )

    ordered_pages = sorted(
        expected_markdown_pages,
        key=lambda relative: (relative != "index.html.md", relative),
    )
    expected_full_parts: list[str] = []
    table_totals = {
        "tables": 0,
        "rows": 0,
        "cells": 0,
        "encoded_pipes": 0,
    }
    for relative in ordered_pages:
        page_path = rag_language_root / Path(relative)
        page_text = page_path.read_text(encoding="utf-8")
        if not page_text.strip():
            raise ValueError(f"Empty RAG page: {page_path}")
        if BANNED_PATTERN.search(page_text):
            raise ValueError(f"Internal build or test content in RAG page: {relative}")
        if language == "en" and CJK_PATTERN.search(page_text):
            raise ValueError(f"Non-English script in English RAG page: {relative}")
        if language == "zh" and not HAN_PATTERN.search(page_text):
            raise ValueError(f"Chinese text missing from RAG page: {relative}")
        _validate_local_links(page_path, page_text, rag_language_root)
        html_page_path = html_language_root / Path(relative.removesuffix(".md"))
        page_table_stats = _validate_page_tables(
            html_page_path,
            page_path,
            page_text,
        )
        for key in table_totals:
            table_totals[key] += page_table_stats[key]
        corpus_text = _rewrite_links_for_corpus_root(
            page_path,
            page_text.strip(),
            rag_language_root,
        )
        expected_full_parts.append(f"# {relative}\n\n{corpus_text}\n\n")

    expected_full = "".join(expected_full_parts)
    if full_text != expected_full:
        raise ValueError(
            f"{language} llms-full.txt is not the deterministic, "
            "path-rebased concatenation of all RAG pages"
        )

    if BANNED_PATTERN.search(sitemap) or BANNED_PATTERN.search(full_text):
        raise ValueError(f"Internal build or test content in {language} RAG corpus")
    return {"pages": len(html_pages), **table_totals}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("html_root", type=Path)
    parser.add_argument("rag_root", type=Path)
    args = parser.parse_args()

    language_stats = {
        language: _validate_language(args.html_root, args.rag_root, language)
        for language in LANGUAGES
    }
    for key in ("pages", "tables", "rows", "cells", "encoded_pipes"):
        values = {stats[key] for stats in language_stats.values()}
        if len(values) != 1:
            raise ValueError(
                f"RAG language {key} counts differ: {language_stats}"
            )

    stats = language_stats["en"]
    print(
        "[OK] RAG corpus validated: languages=2 "
        f"pages_per_language={stats['pages']} "
        f"tables_per_language={stats['tables']} "
        f"rows_per_language={stats['rows']} "
        f"cells_per_language={stats['cells']} "
        f"encoded_pipes_per_language={stats['encoded_pipes']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
