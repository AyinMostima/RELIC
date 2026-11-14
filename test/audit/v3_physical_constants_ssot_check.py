import json
from collections import defaultdict
from pathlib import Path

import yaml


def main():
    root = Path("relic/parameters/database/relic")
    refs = defaultdict(list)
    for y in root.rglob("*.yaml"):
        data = yaml.safe_load(y.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        params = data.get("parameters", {})
        if not isinstance(params, dict):
            continue
        for key in params.keys():
            if key.startswith("relic.physical.constants.") or key.startswith("physical.constants."):
                refs[key].append(str(y).replace("\\", "/"))

    violations = []
    for key, files in refs.items():
        in_ssot = [f for f in files if f.endswith("relic/physical/constants.yaml")]
        if not in_ssot or len(files) > 1:
            violations.append({"key": key, "files": files})

    out = Path("test/audit/v3_physical_constants_ssot_check.json")
    out.write_text(json.dumps(violations, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK]", out.as_posix())
    print("violations", len(violations))
    for v in violations[:20]:
        print(v)


if __name__ == "__main__":
    main()
