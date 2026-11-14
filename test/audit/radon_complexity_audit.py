"""Normalize radon complexity statistics into a single deterministic summary."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "analysis_results" / "radon_cc.json"
OUTPUT_PATH = REPO_ROOT / "analysis_results" / "radon_complexity_summary.json"

MONITORED_SYMBOLS = {
    "relic\\core\\simulator\\execution_group.py::execute_group": (
        "relic\\core\\simulator\\execution_group.py",
        "execute_group",
    ),
    "relic\\core\\functor\\functor_base.py::get_parameter_value": (
        "relic\\core\\functor\\functor_base.py",
        "get_parameter_value",
    ),
}


def _generate_radon_json(path: Path) -> None:
    cmd = [sys.executable, "-m", "radon", "cc", "-j", "relic"]
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "radon complexity scan failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(completed.stdout, encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    encodings = ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be")
    for encoding in encodings:
        try:
            return json.loads(path.read_text(encoding=encoding))
        except Exception:
            continue
    raise RuntimeError(f"Unable to parse JSON file with supported encodings: {path}")


def _iter_nested_entities(entity: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for key in ("closures", "methods"):
        children = entity.get(key, [])
        if not isinstance(children, list):
            continue
        for child in children:
            if isinstance(child, dict):
                yield child
                yield from _iter_nested_entities(child)


def _iter_entities(
    payload: Dict[str, Any],
    include_nested: bool,
) -> Iterable[Dict[str, Any]]:
    for file_path, entities in payload.items():
        if not isinstance(entities, list):
            continue
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            record = dict(entity)
            record["file"] = file_path
            yield record
            if include_nested:
                for child in _iter_nested_entities(entity):
                    child_record = dict(child)
                    child_record["file"] = file_path
                    yield child_record


def _to_symbol(entity: Dict[str, Any]) -> str:
    name = str(entity.get("name", "") or "")
    class_name = str(entity.get("classname", "") or "")
    if class_name:
        return f"{class_name}.{name}"
    return name


def _to_hotspot(entity: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "file": str(entity.get("file", "")),
        "symbol": _to_symbol(entity),
        "complexity": int(entity.get("complexity", 0) or 0),
        "line": int(entity.get("lineno", 0) or 0),
        "entity_type": str(entity.get("type", "")),
    }


def _entity_key(entity: Dict[str, Any]) -> Tuple[str, str, str, int, str, int]:
    return (
        str(entity.get("file", "")),
        str(entity.get("name", "")),
        str(entity.get("classname", "")),
        int(entity.get("lineno", 0) or 0),
        str(entity.get("type", "")),
        int(entity.get("complexity", 0) or 0),
    )


def _dedupe_entities(entities: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for entity in entities:
        key = _entity_key(entity)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entity)
    return deduped


def _count_at_least(entities: Iterable[Dict[str, Any]], threshold: int) -> int:
    count = 0
    for entity in entities:
        if int(entity.get("complexity", 0) or 0) > threshold:
            count += 1
    return count


def _sort_hotspots(entities: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted = [_to_hotspot(entity) for entity in entities]
    converted.sort(
        key=lambda item: (
            -int(item["complexity"]),
            str(item["file"]).lower(),
            str(item["symbol"]).lower(),
            int(item["line"]),
        )
    )
    return converted


def _pick_monitored_hotspots(hotspots: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    tracked: Dict[str, Dict[str, Any]] = {}
    for label, (target_file, target_name) in MONITORED_SYMBOLS.items():
        match = next(
            (
                h
                for h in hotspots
                if h["file"] == target_file and h["symbol"].split(".")[-1] == target_name
            ),
            None,
        )
        tracked[label] = {
            "present": match is not None,
            "complexity": int(match["complexity"]) if match else None,
            "line": int(match["line"]) if match else None,
        }
    return tracked


def main() -> None:
    try:
        payload = _read_json(INPUT_PATH)
    except RuntimeError:
        _generate_radon_json(INPUT_PATH)
        payload = _read_json(INPUT_PATH)

    top_level_entities_raw = list(_iter_entities(payload, include_nested=False))
    flat_entities_raw = list(_iter_entities(payload, include_nested=True))
    top_level_entities = _dedupe_entities(top_level_entities_raw)
    flat_entities = _dedupe_entities(flat_entities_raw)
    hotspots_flat = _sort_hotspots(flat_entities)

    summary = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(INPUT_PATH.relative_to(REPO_ROOT)).replace("/", "\\"),
        "metrics": {
            "files_scanned": len(payload),
            "entries_top_level_raw": len(top_level_entities_raw),
            "entries_flat_raw": len(flat_entities_raw),
            "entries_top_level": len(top_level_entities),
            "entries_flat": len(flat_entities),
            "complexity_gt15_top_level": _count_at_least(top_level_entities, threshold=15),
            "complexity_gt15_flat": _count_at_least(flat_entities, threshold=15),
            "complexity_gt30_flat": _count_at_least(flat_entities, threshold=30),
        },
        "top_hotspots_flat": hotspots_flat[:35],
        "monitored_hotspots": _pick_monitored_hotspots(hotspots_flat),
    }

    OUTPUT_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[OK] summary={OUTPUT_PATH}")
    print(
        "[OK] gt15_top={top} gt15_flat={flat} gt30_flat={gt30}".format(
            top=summary["metrics"]["complexity_gt15_top_level"],
            flat=summary["metrics"]["complexity_gt15_flat"],
            gt30=summary["metrics"]["complexity_gt30_flat"],
        )
    )


if __name__ == "__main__":
    main()
