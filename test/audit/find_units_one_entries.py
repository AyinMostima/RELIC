from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
YAML_ROOT = ROOT / "relic" / "parameters" / "database" / "relic"
OUT = ROOT / "analysis_results" / "units_one_entries_v3.json"


def main() -> None:
    entries = []
    for path in sorted(YAML_ROOT.rglob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        params = data.get("parameters", {})
        if not isinstance(params, dict):
            continue
        for key, spec in params.items():
            if not isinstance(spec, dict):
                continue
            units = spec.get("units")
            if units == "1":
                entries.append(
                    {
                        "file": str(path.relative_to(ROOT)).replace("/", "\\"),
                        "key": key,
                        "units": units,
                        "dimension": spec.get("dimension"),
                    }
                )
    OUT.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] file:", OUT)
    print("[OK] count:", len(entries))
    print(json.dumps(entries[:50], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
