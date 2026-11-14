"""Text file I/O in relic declares its encoding explicitly.

Configuration, parameter and data text files are UTF-8. Reading them with the
platform default codec (GBK on a Chinese-locale Windows host) fails on a UTF-8
BOM or on non-ASCII text, so every text-mode ``open`` / ``Path.open`` /
``read_text`` / ``write_text`` call in relic passes ``encoding``: reads use
``utf-8-sig`` (UTF-8 that also accepts a leading BOM), writes use ``utf-8``.
Binary modes carry no encoding and are exempt.
"""

from __future__ import annotations

import ast
from pathlib import Path

RELIC_ROOT = Path(__file__).resolve().parents[2] / "relic"
READ_ENCODING = "utf-8-sig"
WRITE_ENCODING = "utf-8"
# ``<owner>.open(...)`` calls on these owners are not text file opens.
NON_FILE_OPEN_OWNERS = {
    "os", "webbrowser", "gzip", "bz2", "lzma", "zipfile", "tarfile",
    "h5py", "Image", "xr", "xarray", "netCDF4", "tempfile",
}


def _mode_node(call: ast.Call) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == "mode":
            return keyword.value
    func = call.func
    if isinstance(func, ast.Name) or (
        isinstance(func, ast.Attribute) and ast.unparse(func.value) in {"io", "codecs"}
    ):
        return call.args[1] if len(call.args) >= 2 else None
    return call.args[0] if call.args else None


def _text_io_calls(tree: ast.AST):
    """Yield (call, direction) for each text-mode file call; direction is read or write."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in {"read_text", "write_text"}:
            yield node, "read" if func.attr == "read_text" else "write"
            continue
        is_open = isinstance(func, ast.Name) and func.id == "open"
        if isinstance(func, ast.Attribute) and func.attr == "open":
            is_open = ast.unparse(func.value) not in NON_FILE_OPEN_OWNERS
        if not is_open:
            continue
        mode = _mode_node(node)
        if mode is None:
            yield node, "read"
        elif isinstance(mode, ast.Constant) and isinstance(mode.value, str):
            if "b" in mode.value:
                continue
            yield node, "write" if set(mode.value) & set("wax+") else "read"
        else:
            yield node, "unknown"


def _declared_encoding(call: ast.Call):
    for keyword in call.keywords:
        if keyword.arg == "encoding":
            value = keyword.value
            return value.value if isinstance(value, ast.Constant) else ast.unparse(value)
    return None


def test_relic_text_file_io_declares_utf8_encoding():
    offenders = []
    for path in sorted(RELIC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for call, direction in _text_io_calls(tree):
            encoding = _declared_encoding(call)
            expected = {
                "read": {READ_ENCODING},
                "write": {WRITE_ENCODING},
                "unknown": {READ_ENCODING, WRITE_ENCODING},
            }[direction]
            if encoding not in expected:
                offenders.append(
                    f"{path.relative_to(RELIC_ROOT.parent)}:{call.lineno}: "
                    f"{direction} encoding={encoding!r}: {ast.unparse(call)[:90]}"
                )
    assert offenders == [], "\n".join(offenders)
