import ast
import json
from collections import defaultdict
from pathlib import Path


BRIDGES = {
    "PhysicalFieldFunctor",
    "BiologicalProcessFunctor",
    "TransportMechanismFunctor",
    "ScaleCouplingFunctor",
}


def base_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def main():
    outputs = defaultdict(list)
    roots = [Path("relic/scientific"), Path("relic/coupling")]

    for root in roots:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8-sig", errors="ignore")
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue

            for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
                bases = {base_name(b) for b in cls.bases}
                if not (bases & BRIDGES):
                    continue

                for node in ast.walk(cls):
                    if not isinstance(node, ast.Call):
                        continue
                    fn = node.func.id if isinstance(node.func, ast.Name) else (
                        node.func.attr if isinstance(node.func, ast.Attribute) else None
                    )
                    if fn != "FunctorMetadata":
                        continue
                    for kw in node.keywords:
                        if kw.arg != "output_variables":
                            continue
                        if not isinstance(kw.value, ast.List):
                            continue
                        for elt in kw.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                outputs[elt.value].append(
                                    {
                                        "file": str(path).replace("\\", "/"),
                                        "class": cls.name,
                                        "metadata_line": node.lineno,
                                    }
                                )
                    break

    overlaps = {
        var: writers
        for var, writers in outputs.items()
        if len(writers) > 1 and not var.startswith("diagnostics.")
    }
    overlaps = dict(sorted(overlaps.items(), key=lambda kv: len(kv[1]), reverse=True))

    out_path = Path("test/audit/v3_output_overlap_details.json")
    out_path.write_text(json.dumps(overlaps, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[OK]", out_path.as_posix())
    print("overlap_vars", len(overlaps))
    for var, writers in list(overlaps.items())[:15]:
        print(var, len(writers))


if __name__ == "__main__":
    main()
