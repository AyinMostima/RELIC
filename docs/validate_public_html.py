"""Validate the staged RELIC bilingual HTML publication."""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

from bs4 import BeautifulSoup, NavigableString, Tag


LANGUAGES = ("en", "zh")
PUBLIC_IMAGE_REFERENCES = {
    Path("architecture.html"): (
        "relic_architecture_overview_en.jpg",
        "relic_four_scientific_abstractions.png",
        "relic_five_variable_roles.png",
    ),
    Path("orchestration.html"): (
        "relic_port_passport_and_contracts.png",
    ),
}
SCIENCE_GOVERNING_TITLES = (
    "governing equations and canonical operator",
    "控制方程与 canonical operator",
)
RAW_FORMULA_API_PAGES = {
    Path("api/parameters_contracts.html"),
    Path("api/scientific_structure_extensions.html"),
}
MATH_ERROR_SELECTOR = (
    "mjx-merror, merror, .mjx-merror, .MathJax_Error, "
    ".math-error, [data-mjx-error]"
)
RAW_MATH_DELIMITER_PATTERN = re.compile(
    r"\\(?:\(|\)|\[|\])"
    r"|\$\$"
    r"|(?<!\$)\$(?=\S)[^$\n]+(?<=\S)\$(?!\$)"
)
RAW_TEX_COMMAND_PATTERN = re.compile(
    r"\\(?:"
    r"begin|end|frac|dfrac|tfrac|sum|prod|int|dot|ddot|partial|"
    r"nabla|Delta|mathcal|mathrm|mathbf|mathsf|operatorname|"
    r"qquad|quad|succeq|geq|leq|times|cdot"
    r")\b"
)
API_RAW_FORMULA_PATTERN = re.compile(
    r"\\(?:dot|Delta|sum|iota|mathsf|qquad|succeq|geq)\b"
    r"|[A-Za-z]_\{[^}]+\}[^=\n]*="
)
MAPPING_SECTION_PREFIXES = (
    "governing-equations-and-canonical-operator",
    "state-advancement-conservation-and-open-boundaries",
)
RAW_MAPPING_FORMULA_PATTERN = re.compile(
    r"(?<![<>=:])=(?!=)"
    r"|(?<![A-Za-z0-9_\\])d[A-Za-z][A-Za-z0-9_,]*/"
    r"d[A-Za-z][A-Za-z0-9_]*"
    r"|\.[T](?![A-Za-z])"
    r"|\^[T](?![A-Za-z])"
    r"|\*"
    r"|(?<!\\)\|[A-Za-z][A-Za-z0-9_]*\|"
)
NONSTANDARD_MATH_PATTERNS = (
    (
        re.compile(
            r"(?<![A-Za-z0-9_\\])d[A-Za-z][A-Za-z0-9_,]*/"
            r"d[A-Za-z][A-Za-z0-9_]*"
        ),
        "ASCII derivative",
    ),
    (
        re.compile(
            r"(?<!\\)(?<!\\left)\|[^|\r\n]+?(?<!\\right)\|"
        ),
        "ASCII absolute-value delimiter",
    ),
    (re.compile(r"\\\|"), "noncanonical norm delimiter"),
    (re.compile(r"\.[T](?![A-Za-z])"), "ASCII transpose"),
    (re.compile(r"\^[T](?![A-Za-z])"), "ungrouped transpose"),
    (re.compile(r"\\sqrt\s*\("), "ungrouped square-root argument"),
    (re.compile(r"\\text\{d[^{}]*/d[^{}]*\}"), "textified derivative"),
    (re.compile(r"\\mathrm\{dot\}"), "textified dot product"),
    (
        re.compile(
            r"\^\s*(?:\{[^{}]*\}|\\[A-Za-z]+|[A-Za-z0-9])\s*\^"
        ),
        "double exponent",
    ),
    (re.compile(r"[A-Za-z{}_\s]+\\cdot\s*\^"), "missing exponent base"),
    (re.compile(r"[A-Za-z]_\{\\mathrm\{dot\}"), "textified time derivative"),
)


def _positive_span(cell: Tag, attribute: str) -> int:
    raw_value = cell.get(attribute, "1")
    try:
        value = int(str(raw_value))
    except ValueError as exc:
        raise ValueError(
            f"Invalid {attribute}={raw_value!r} in a table cell."
        ) from exc
    if value < 1:
        raise ValueError(f"{attribute} must be positive, received {value}.")
    return value


def _table_shape(
    table: Tag,
    source: Path,
    index: int,
) -> tuple[tuple[tuple[str, int, int], ...], ...]:
    active_rowspans: dict[int, int] = {}
    row_widths: list[int] = []
    row_signatures: list[tuple[tuple[str, int, int], ...]] = []
    rows = table.find_all("tr")
    if not rows:
        raise ValueError(f"{source}: table {index} contains no rows.")

    for row_index, row in enumerate(rows, start=1):
        occupied = set(active_rowspans)
        next_rowspans = {
            column: remaining - 1
            for column, remaining in active_rowspans.items()
            if remaining > 1
        }
        cells = row.find_all(["th", "td"], recursive=False)
        if not cells:
            raise ValueError(
                f"{source}: table {index} row {row_index} contains no cells."
            )

        cursor = 0
        row_signature: list[tuple[str, int, int]] = []
        for cell in cells:
            while cursor in occupied:
                cursor += 1
            colspan = _positive_span(cell, "colspan")
            rowspan = _positive_span(cell, "rowspan")
            row_signature.append((cell.name, colspan, rowspan))
            columns = range(cursor, cursor + colspan)
            if any(column in occupied for column in columns):
                raise ValueError(
                    f"{source}: table {index} row {row_index} has "
                    "overlapping spans."
                )
            text = cell.get_text(" ", strip=True)
            if not text:
                raise ValueError(
                    f"{source}: table {index} row {row_index} has "
                    "an empty cell."
                )
            for column in columns:
                occupied.add(column)
                if rowspan > 1:
                    next_rowspans[column] = rowspan - 1
            cursor += colspan

        width = max(occupied, default=-1) + 1
        if occupied != set(range(width)):
            raise ValueError(
                f"{source}: table {index} row {row_index} has a column gap."
            )
        row_widths.append(width)
        row_signatures.append(tuple(row_signature))
        active_rowspans = next_rowspans

    if active_rowspans:
        raise ValueError(f"{source}: table {index} has a dangling rowspan.")
    if len(set(row_widths)) != 1:
        raise ValueError(
            f"{source}: table {index} has inconsistent logical widths "
            f"{row_widths}."
        )
    return tuple(row_signatures)


def _page_shapes(
    path: Path,
) -> tuple[tuple[tuple[tuple[str, int, int], ...], ...], ...]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    return tuple(
        _table_shape(table, path, index)
        for index, table in enumerate(soup.find_all("table"), start=1)
    )


def _math_profile(soup: BeautifulSoup, source: Path) -> tuple[int, int, int]:
    inline_count = 0
    display_count = 0
    for node in soup.select(".math"):
        math_text = node.get_text(strip=True)
        for pattern, label in NONSTANDARD_MATH_PATTERNS:
            if pattern.search(math_text):
                snippet = " ".join(math_text.split())[:120]
                raise ValueError(
                    f"{source}: {label} remains in a math container: "
                    f"{snippet!r}."
                )
        if (
            node.name == "span"
            and math_text.startswith(r"\(")
            and math_text.endswith(r"\)")
        ):
            inline_count += 1
        elif (
            node.name == "div"
            and math_text.startswith(r"\[")
            and math_text.endswith(r"\]")
        ):
            display_count += 1
        else:
            snippet = " ".join(math_text.split())[:120]
            raise ValueError(
                f"{source}: malformed math container <{node.name}>: "
                f"{snippet!r}."
            )
    return inline_count + display_count, inline_count, display_count


def _text_is_inside_literal_or_math(text: NavigableString) -> bool:
    for parent in text.parents:
        if not isinstance(parent, Tag):
            continue
        if parent.name in {"code", "pre", "script", "style"}:
            return True
        if "math" in parent.get("class", ()):
            return True
    return False


def _plain_text_outside_literal_or_math(container: Tag) -> str:
    return " ".join(
        str(text)
        for text in container.find_all(string=True)
        if not _text_is_inside_literal_or_math(text)
    )


def _validate_math_rendering(
    soup: BeautifulSoup,
    source: Path,
    relative: Path,
) -> tuple[int, int, int]:
    error_nodes = soup.select(MATH_ERROR_SELECTOR)
    if error_nodes:
        raise ValueError(
            f"{source}: found {len(error_nodes)} MathJax error node(s)."
        )

    raw_leakages: list[str] = []
    for text in soup.find_all(string=True):
        if _text_is_inside_literal_or_math(text):
            continue
        value = str(text)
        if (
            RAW_MATH_DELIMITER_PATTERN.search(value)
            or RAW_TEX_COMMAND_PATTERN.search(value)
        ):
            raw_leakages.append(" ".join(value.split())[:120])
    if raw_leakages:
        raise ValueError(
            f"{source}: raw TeX leaked outside a math container: "
            f"{raw_leakages[0]!r}."
        )

    if relative.parts and relative.parts[0] == "science":
        for section in soup.find_all("section"):
            section_id = str(section.get("id", ""))
            if any(
                section_id.startswith(prefix)
                for prefix in MAPPING_SECTION_PREFIXES
            ):
                for cell in section.select("table td"):
                    plain_text = _plain_text_outside_literal_or_math(cell)
                    match = RAW_MAPPING_FORMULA_PATTERN.search(plain_text)
                    if match:
                        snippet = " ".join(plain_text.split())[:120]
                        raise ValueError(
                            f"{source}: scientific mapping table contains "
                            f"raw formula notation {match.group(0)!r}: "
                            f"{snippet!r}."
                        )
            heading = section.find(
                ["h1", "h2", "h3", "h4", "h5", "h6"],
                recursive=False,
            )
            if heading is None:
                continue
            heading_text = " ".join(heading.get_text(" ", strip=True).split())
            if not any(
                title in heading_text.lower()
                for title in SCIENCE_GOVERNING_TITLES
            ):
                continue
            raw_blocks = list(section.find_all("pre"))
            raw_blocks.extend(
                block
                for block in section.select("div.highlight-text")
                if block.find("pre") is None
            )
            if raw_blocks:
                raise ValueError(
                    f"{source}: governing-equation section {heading_text!r} "
                    f"contains {len(raw_blocks)} raw text/pre block(s); "
                    "use Sphinx math nodes."
                )

    if relative in RAW_FORMULA_API_PAGES:
        for block in soup.find_all("pre"):
            formula_text = block.get_text("\n", strip=True)
            if API_RAW_FORMULA_PATTERN.search(formula_text):
                snippet = " ".join(formula_text.split())[:120]
                raise ValueError(
                    f"{source}: API formula remains in a raw pre block: "
                    f"{snippet!r}."
                )

    return _math_profile(soup, source)


def _page_math_profile(path: Path, relative: Path) -> tuple[int, int, int]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    return _validate_math_rendering(soup, path, relative)


def _page_math_structure(path: Path) -> tuple[str, ...]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    signatures: list[str] = []
    for node in soup.select(".math"):
        math_text = node.get_text(strip=True)
        if math_text.startswith((r"\(", r"\[")) and math_text.endswith(
            (r"\)", r"\]")
        ):
            math_text = math_text[2:-2]
        math_text = re.sub(
            r"\\text\{[^{}]*\}",
            r"\\text{TEXT}",
            math_text,
        )
        math_text = math_text.translate(
            str.maketrans({"。": ".", "，": ",", "：": ":"})
        )
        signatures.append(re.sub(r"\s+", "", math_text))
    return tuple(signatures)


def _relative_pages(root: Path) -> list[Path]:
    return sorted(
        path.relative_to(root)
        for path in root.rglob("*.html")
        if path.name != "search.html"
    )


def _validate_public_images(
    roots: dict[str, Path],
    picture_dir: Path,
) -> None:
    source_hashes: dict[str, str] = {}
    has_picture_dir = picture_dir.is_dir()
    for image_names in PUBLIC_IMAGE_REFERENCES.values():
        for image_name in image_names:
            if has_picture_dir:
                source_image = picture_dir / image_name
            else:
                source_image = roots["en"] / "_images" / image_name
            if not source_image.is_file() or source_image.stat().st_size == 0:
                raise FileNotFoundError(f"Missing source public image: {source_image}")
            source_hashes[image_name] = hashlib.sha256(
                source_image.read_bytes()
            ).hexdigest()

    for language, root in roots.items():
        for page, image_names in PUBLIC_IMAGE_REFERENCES.items():
            page_path = root / page
            if not page_path.is_file():
                raise FileNotFoundError(f"Missing public page for {language}: {page_path}")
            soup = BeautifulSoup(page_path.read_text(encoding="utf-8"), "html.parser")
            for image_name in image_names:
                image = root / "_images" / image_name
                if not image.is_file() or image.stat().st_size == 0:
                    raise FileNotFoundError(
                        f"Missing public image for {language}: {image}"
                    )
                actual_hash = hashlib.sha256(image.read_bytes()).hexdigest()
                if actual_hash != source_hashes[image_name]:
                    raise ValueError(
                        f"Public image hash differs from the source for {language}: "
                        f"{image_name}."
                    )
                image_references = [
                    node
                    for node in soup.find_all("img")
                    if (
                        not (parsed_src := urlsplit(str(node.get("src", "")))).scheme
                        and not parsed_src.netloc
                        and parsed_src.path == f"_images/{image_name}"
                    )
                ]
                if len(image_references) != 1:
                    raise ValueError(
                        f"{page_path}: expected one public image reference for "
                        f"{image_name}, found {len(image_references)}."
                    )


def validate(stage_root: Path) -> None:
    roots = {language: stage_root / language for language in LANGUAGES}
    for language, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"Missing language root: {root}")
        switch_script = root / "_static" / "language_switch.js"
        if not switch_script.is_file():
            raise FileNotFoundError(
                f"Missing language switch script: {switch_script}"
            )
        script_text = switch_script.read_text(encoding="utf-8")
        for token in (
            "RELICLanguageTarget",
            "relic-language-switch",
            'currentLanguage === "en" ? "zh" : "en"',
        ):
            if token not in script_text:
                raise ValueError(
                    f"{switch_script}: missing switch contract {token!r}."
                )

    page_sets = {
        language: _relative_pages(root)
        for language, root in roots.items()
    }
    if page_sets["en"] != page_sets["zh"]:
        raise ValueError("English and Chinese content-page sets differ.")

    table_count = 0
    math_count = 0
    for relative in page_sets["en"]:
        en_path = roots["en"] / relative
        zh_path = roots["zh"] / relative
        en_shapes = _page_shapes(en_path)
        zh_shapes = _page_shapes(zh_path)
        if en_shapes != zh_shapes:
            raise ValueError(
                f"English and Chinese table structures differ in {relative}."
            )
        table_count += len(en_shapes)
        en_math_profile = _page_math_profile(en_path, relative)
        zh_math_profile = _page_math_profile(zh_path, relative)
        if en_math_profile != zh_math_profile:
            raise ValueError(
                f"English and Chinese math-node counts differ in {relative}: "
                f"en={en_math_profile}, zh={zh_math_profile}."
            )
        en_math_structure = _page_math_structure(en_path)
        zh_math_structure = _page_math_structure(zh_path)
        en_math_multiset = Counter(en_math_structure)
        zh_math_multiset = Counter(zh_math_structure)
        if en_math_multiset != zh_math_multiset:
            missing = list((en_math_multiset - zh_math_multiset).elements())
            unexpected = list(
                (zh_math_multiset - en_math_multiset).elements()
            )
            raise ValueError(
                f"English and Chinese math structures differ in {relative}: "
                f"missing_from_zh={missing[0][:120] if missing else None!r}, "
                f"unexpected_in_zh="
                f"{unexpected[0][:120] if unexpected else None!r}."
            )
        math_count += en_math_profile[0]

    picture_dir = Path(__file__).resolve().parent.parent / "Picture"
    _validate_public_images(roots, picture_dir)

    print(
        "[OK] Validated "
        f"{len(page_sets['en'])} bilingual content pages and "
        f"{table_count} table pairs with "
        f"{math_count} math nodes per language."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate staged RELIC bilingual HTML."
    )
    parser.add_argument("stage_root", type=Path)
    arguments = parser.parse_args()
    validate(arguments.stage_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
