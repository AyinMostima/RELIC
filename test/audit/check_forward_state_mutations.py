from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "analysis_results" / "deep_scientific_audit_v3_report.json"
OUT_PATH = ROOT / "analysis_results" / "forward_state_mutations_v3.json"


def main() -> None:
    data = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    findings = []
    for row in data.get("python_scans", []):
        file_path = row["file"]
        full_path = ROOT / file_path
        text = full_path.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue

        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            bridge = None
            for b in cls.bases:
                btxt = ast.unparse(b)
                if btxt.split(".")[-1] in {
                    "PhysicalFieldFunctor",
                    "BiologicalProcessFunctor",
                    "TransportMechanismFunctor",
                    "ScaleCouplingFunctor",
                }:
                    bridge = btxt.split(".")[-1]
            if not bridge:
                continue
            for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"]:
                for sub in ast.walk(fn):
                    if isinstance(sub, ast.Assign):
                        for target in sub.targets:
                            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == "state":
                                findings.append(
                                    {
                                        "file": file_path,
                                        "class": cls.name,
                                        "bridge": bridge,
                                        "line": sub.lineno,
                                        "snippet": ast.unparse(sub)[:200],
                                    }
                                )
                    if isinstance(sub, ast.AugAssign):
                        target = sub.target
                        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == "state":
                            findings.append(
                                {
                                    "file": file_path,
                                    "class": cls.name,
                                    "bridge": bridge,
                                    "line": sub.lineno,
                                    "snippet": ast.unparse(sub)[:200],
                                }
                            )
    OUT_PATH.write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] findings:", OUT_PATH)
    print("[OK] count:", len(findings))
    print(json.dumps(findings[:40], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
