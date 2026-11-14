from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "analysis_results" / "deep_scientific_audit_v3_report.json"
OUT_CSV = ROOT / "analysis_results" / "functor_inventory_v3.csv"
OUT_JSON = ROOT / "analysis_results" / "functor_findings_v3.json"


def main() -> None:
    data = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    rows = []
    findings = {
        "missing_super_init": [],
        "bad_forward_signature": [],
        "metadata_empty_io": [],
        "state_direct_write_files": [],
        "torch_tensor_constant_hits": [],
    }

    for f in data.get("python_scans", []):
        file_path = f["file"]
        if file_path.startswith("relic\\scientific") and f.get("state_direct_write_hits"):
            findings["state_direct_write_files"].append(
                {"file": file_path, "hits": f.get("state_direct_write_hits", [])[:10]}
            )
        if file_path.startswith("relic\\scientific") and f.get("torch_tensor_constant_hits"):
            findings["torch_tensor_constant_hits"].append(
                {"file": file_path, "hits": f.get("torch_tensor_constant_hits", [])[:10]}
            )
        for c in f.get("classes", []):
            if not c.get("bridge_base"):
                continue
            dep = ";".join(c.get("dependencies", []))
            metadata = c.get("metadata_calls", [])
            spatial = ""
            variable_kind = ""
            execution_phase = ""
            input_vars = ""
            output_vars = ""
            for m in metadata:
                kw = m.get("keywords", {})
                spatial = str(kw.get("spatial_scale", spatial))
                variable_kind = str(kw.get("variable_kind", variable_kind))
                execution_phase = str(kw.get("execution_phase", execution_phase))
                if isinstance(kw.get("input_variables"), list):
                    input_vars = ";".join(kw.get("input_variables", []))
                if isinstance(kw.get("output_variables"), list):
                    output_vars = ";".join(kw.get("output_variables", []))
                if isinstance(kw.get("input_variables"), list) and isinstance(kw.get("output_variables"), list):
                    if len(kw["input_variables"]) == 0 or len(kw["output_variables"]) == 0:
                        findings["metadata_empty_io"].append(
                            {
                                "file": file_path,
                                "class": c["name"],
                                "line": m.get("lineno"),
                                "input_len": len(kw["input_variables"]),
                                "output_len": len(kw["output_variables"]),
                            }
                        )

            if not c.get("has_super_init"):
                findings["missing_super_init"].append(
                    {"file": file_path, "class": c["name"], "line": c.get("lineno")}
                )

            sig = c.get("forward_signature")
            if sig:
                args = sig.get("args", [])
                ok1 = args[:5] == ["self", "state", "time", "dt"]
                ok2 = args[:6] == ["self", "state", "environment", "time", "dt"]
                if not (ok1 or ok2):
                    findings["bad_forward_signature"].append(
                        {"file": file_path, "class": c["name"], "line": sig.get("lineno"), "args": args}
                    )

            rows.append(
                {
                    "file_path": file_path,
                    "class_name": c["name"],
                    "abstraction": c.get("abstraction") or "",
                    "execution_phase": execution_phase,
                    "dependencies": dep,
                    "parameter_namespace": c.get("default_parameter_namespace") or "",
                    "spatial_scale": spatial,
                    "variable_kind": variable_kind,
                    "input_variables": input_vars,
                    "output_variables": output_vars,
                    "forward_signature": ",".join(c.get("forward_signature", {}).get("args", [])),
                }
            )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "file_path",
                "class_name",
                "abstraction",
                "execution_phase",
                "dependencies",
                "parameter_namespace",
                "spatial_scale",
                "variable_kind",
                "input_variables",
                "output_variables",
                "forward_signature",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    OUT_JSON.write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] inventory:", OUT_CSV)
    print("[OK] findings:", OUT_JSON)
    print(json.dumps({k: len(v) for k, v in findings.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
