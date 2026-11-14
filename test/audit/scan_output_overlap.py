import ast
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path("relic/scientific")
WHITELIST_PATH = Path("relic/core/simulator/output_overlap_whitelist.json")
OUT_DIR = Path("analysis_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
rows = []

for py in ROOT.rglob("*.py"):
    src = py.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        continue

    for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
        bases = []
        for b in cls.bases:
            if isinstance(b, ast.Name):
                bases.append(b.id)
            elif isinstance(b, ast.Attribute):
                bases.append(b.attr)
            else:
                bases.append("")
        if not any("Functor" in b for b in bases):
            continue

        for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]:
            for call in ast.walk(fn):
                if not isinstance(call, ast.Call):
                    continue
                name = ""
                if isinstance(call.func, ast.Name):
                    name = call.func.id
                elif isinstance(call.func, ast.Attribute):
                    name = call.func.attr
                if name != "FunctorMetadata":
                    continue

                out_vars = []
                in_vars = []
                spatial_scale = None
                variable_kind = None

                for kw in call.keywords:
                    if kw.arg == "output_variables" and isinstance(kw.value, (ast.List, ast.Tuple)):
                        for elt in kw.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                out_vars.append(elt.value)
                    if kw.arg == "input_variables" and isinstance(kw.value, (ast.List, ast.Tuple)):
                        for elt in kw.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                in_vars.append(elt.value)
                    if kw.arg == "spatial_scale":
                        spatial_scale = ast.unparse(kw.value) if hasattr(ast, "unparse") else None
                    if kw.arg == "variable_kind":
                        variable_kind = ast.unparse(kw.value) if hasattr(ast, "unparse") else None

                rows.append(
                    {
                        "file": str(py).replace("/", "\\"),
                        "class": cls.name,
                        "line": cls.lineno,
                        "outputs": out_vars,
                        "inputs": in_vars,
                        "spatial_scale": spatial_scale,
                        "variable_kind": variable_kind,
                    }
                )

owners = defaultdict(list)
for row in rows:
    for out in row["outputs"]:
        if out.startswith("diagnostics."):
            continue
        owners[out].append(f"{row['class']}@{row['file']}:{row['line']}")

dups = {k: v for k, v in owners.items() if len(v) > 1}
try:
    whitelist_raw = json.loads(WHITELIST_PATH.read_text(encoding="utf-8"))
except Exception:
    whitelist_raw = {}
whitelist = whitelist_raw if isinstance(whitelist_raw, dict) else {}
governed = {k: v for k, v in dups.items() if k in whitelist}
ungoverned = {k: v for k, v in dups.items() if k not in whitelist}

result = {
    "functors": rows,
    "duplicate_outputs": dups,
    "governed_duplicate_outputs": governed,
    "ungoverned_duplicate_outputs": ungoverned,
    "duplicate_count": len(dups),
    "governed_duplicate_count": len(governed),
    "ungoverned_duplicate_count": len(ungoverned),
    "effective_conflict_count": len(ungoverned),
}
(OUT_DIR / "output_overlap_scan.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"functors={len(rows)}")
print(f"duplicate_outputs={len(dups)}")
print(f"governed_duplicate_outputs={len(governed)}")
print(f"ungoverned_duplicate_outputs={len(ungoverned)}")
print(f"effective_conflict_count={len(ungoverned)}")
for key in sorted(ungoverned.keys())[:40]:
    print(f"{key} => {' | '.join(ungoverned[key])}")
