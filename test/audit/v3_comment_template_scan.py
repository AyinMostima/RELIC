import ast
import json
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
    missing_template = []
    checked = 0

    for path in Path("relic/scientific").rglob("*.py"):
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue

        has_template = (
            "# References:" in text
            and "# Mathematical expression:" in text
            and "# Parameter meanings:" in text
        )

        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            bases = {base_name(b) for b in cls.bases}
            if not (bases & BRIDGES):
                continue
            checked += 1
            if not has_template:
                missing_template.append(
                    {
                        "file": str(path).replace("\\", "/"),
                        "class": cls.name,
                        "line": cls.lineno,
                    }
                )

    out = Path("test/audit/v3_comment_template_scan.json")
    out.write_text(json.dumps({"checked": checked, "missing": missing_template}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK]", out.as_posix())
    print("checked", checked)
    print("missing", len(missing_template))
    for item in missing_template[:40]:
        print(item)


if __name__ == "__main__":
    main()
