from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SEARCH_ROOTS = [ROOT / "relic" / "scientific", ROOT / "relic" / "coupling"]
OUT = ROOT / "analysis_results" / "nonfunctor_missing_accessor_v3.json"

FUNCTOR_BASE_NAMES = {
    "PhysicalFieldFunctor",
    "BiologicalProcessFunctor",
    "TransportMechanismFunctor",
    "ScaleCouplingFunctor",
    "FunctorBase",
}


def class_base_names(cls: ast.ClassDef) -> set[str]:
    return {ast.unparse(b).split(".")[-1] for b in cls.bases}


def has_method(cls: ast.ClassDef, name: str) -> bool:
    return any(isinstance(n, ast.FunctionDef) and n.name == name for n in cls.body)


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
                if FUNCTOR_BASE_NAMES & bases:
                    continue
                defines_accessor = has_method(cls, "get_parameter_value")
                for call in [n for n in ast.walk(cls) if isinstance(n, ast.Call)]:
                    fn = ast.unparse(call.func)
                    if fn.endswith("get_parameter_value"):
                        if not defines_accessor:
                            findings.append(
                                {
                                    "file": str(path.relative_to(ROOT)).replace("/", "\\"),
                                    "class": cls.name,
                                    "line": call.lineno,
                                    "call": fn,
                                    "bases": sorted(bases),
                                    "defines_get_parameter_value": defines_accessor,
                                }
                            )
    OUT.write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] file:", OUT)
    print("[OK] count:", len(findings))
    print(json.dumps(findings[:60], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
