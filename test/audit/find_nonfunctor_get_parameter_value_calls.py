from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SEARCH_ROOTS = [ROOT / "relic" / "scientific", ROOT / "relic" / "coupling"]
OUT = ROOT / "analysis_results" / "nonfunctor_get_parameter_value_calls_v3.json"

FUNCTOR_BASE_NAMES = {
    "PhysicalFieldFunctor",
    "BiologicalProcessFunctor",
    "TransportMechanismFunctor",
    "ScaleCouplingFunctor",
    "FunctorBase",
}


def class_base_names(cls: ast.ClassDef) -> set[str]:
    names: set[str] = set()
    for b in cls.bases:
        text = ast.unparse(b)
        names.add(text.split(".")[-1])
    return names


def main() -> None:
    findings = []
    for root in SEARCH_ROOTS:
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
                bases = class_base_names(cls)
                is_functor = bool(FUNCTOR_BASE_NAMES & bases)
                for call in [n for n in ast.walk(cls) if isinstance(n, ast.Call)]:
                    fn = ast.unparse(call.func)
                    if fn.endswith("get_parameter_value") and not is_functor:
                        findings.append(
                            {
                                "file": str(path.relative_to(ROOT)).replace("/", "\\"),
                                "class": cls.name,
                                "line": call.lineno,
                                "call": fn,
                                "bases": sorted(bases),
                            }
                        )
    OUT.write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] file:", OUT)
    print("[OK] count:", len(findings))
    print(json.dumps(findings[:80], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
