"""
Plan visualization tool for scaling manifests.

Reads a scaling manifest and outputs a simple Markdown summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, cast

import yaml


def _load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text) or {}
    else:
        payload = json.loads(text)
    if isinstance(payload, dict):
        return cast(Dict[str, Any], payload)
    raise ValueError(f"Manifest root must be a mapping, got {type(payload).__name__}")


def _render_markdown(manifest: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Scaling Plan Visualization")
    lines.append("")
    manifest_hash = manifest.get("hash", "unknown")
    lines.append(f"Manifest hash: {manifest_hash}")
    lines.append("")

    levels = manifest.get("levels") or []
    lines.append("## Levels")
    if not levels:
        lines.append("- No levels defined.")
    else:
        for level in levels:
            name = level.get("name", "unknown") if isinstance(level, dict) else str(level)
            lines.append(f"- {name}")
    lines.append("")

    steps = manifest.get("steps") or []
    lines.append("## Steps")
    if not steps:
        lines.append("- No steps defined.")
    else:
        for idx, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                lines.append(f"- Step {idx}: {step}")
                continue
            src = step.get("source", "unknown")
            tgt = step.get("target", "unknown")
            method = step.get("method", "unknown")
            ratio = step.get("time_ratio")
            ratio_text = f" time_ratio={ratio}" if ratio is not None else ""
            lines.append(f"- Step {idx}: {src} -> {tgt} ({method}){ratio_text}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a scaling plan manifest.")
    parser.add_argument("--input", required=True, help="Path to scaling manifest (yaml/json).")
    parser.add_argument("--output", help="Optional output markdown path.")
    args = parser.parse_args()

    manifest = _load_manifest(Path(args.input))
    report = _render_markdown(manifest)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
