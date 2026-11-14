from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "analysis_results" / "deep_scientific_audit_v3_report.json"
OUT_PATH = ROOT / "analysis_results" / "deep_scientific_audit_v3_summary.json"


def main() -> None:
    data = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    py = data["python_scans"]
    ys = data["yaml_scans"]

    classes = [(r["file"], c) for r in py for c in r.get("classes", [])]
    functors = [(f, c) for f, c in classes if c.get("bridge_base")]

    missing_meta = [(f, c["name"], c["lineno"]) for f, c in functors if not c.get("metadata_calls")]
    missing_super = [(f, c["name"], c["lineno"]) for f, c in functors if not c.get("has_super_init")]
    missing_forward = [(f, c["name"], c["lineno"]) for f, c in functors if not c.get("has_forward")]
    bad_forward = [
        (f, c["name"], c["forward_signature"])
        for f, c in functors
        if c.get("forward_signature")
        and c["forward_signature"]["args"][:5] != ["self", "state", "time", "dt"]
        and c["forward_signature"]["args"][:6] != ["self", "state", "environment", "time", "dt"]
    ]
    legacy_compute = [
        (f, c["name"], c["lineno"], c["has_compute_rate"], c["has_compute_flux"])
        for f, c in functors
        if c.get("abstraction") in ("BiologicalProcess", "TransportMechanism")
        and (c["has_compute_rate"] or c["has_compute_flux"])
    ]

    sci_get_global = [r["file"] for r in py if r["file"].startswith("relic\\scientific") and r.get("get_global_manager_call")]
    manager_getpv = [r["file"] for r in py if r.get("manager_get_parameter_value_call")]
    category_theory_imports = [r["file"] for r in py if r.get("category_theory_import")]
    state_direct_write_files = [r["file"] for r in py if r.get("state_direct_write_hits") and r["file"].startswith("relic\\scientific")]

    meta_missing_scale = []
    meta_unknown_scale = []
    meta_missing_kind = []
    meta_unknown_kind = []
    meta_empty_io = []
    for f, c in functors:
        for m in c.get("metadata_calls", []):
            kw = m.get("keywords", {})
            if "spatial_scale" not in kw:
                meta_missing_scale.append((f, c["name"], m["lineno"]))
            elif "UNKNOWN" in str(kw.get("spatial_scale")):
                meta_unknown_scale.append((f, c["name"], m["lineno"], kw.get("spatial_scale")))
            if "variable_kind" not in kw:
                meta_missing_kind.append((f, c["name"], m["lineno"]))
            elif "UNKNOWN" in str(kw.get("variable_kind")):
                meta_unknown_kind.append((f, c["name"], m["lineno"], kw.get("variable_kind")))
            if isinstance(kw.get("input_variables"), list) and isinstance(kw.get("output_variables"), list):
                if len(kw["input_variables"]) == 0 or len(kw["output_variables"]) == 0:
                    meta_empty_io.append((f, c["name"], m["lineno"], len(kw["input_variables"]), len(kw["output_variables"])))

    yaml_missing = []
    yaml_empty_ref = []
    yaml_dim = []
    units1_files = []
    frac_unit_files = []
    for y in ys:
        if y.get("uses_units_one"):
            units1_files.append(y["file"])
        if y.get("has_fraction_units"):
            frac_unit_files.append(y["file"])
        for prm in y.get("parameters", []):
            if prm.get("missing_fields"):
                yaml_missing.append((y["file"], prm["key"], prm["missing_fields"]))
            if prm.get("empty_references"):
                yaml_empty_ref.append((y["file"], prm["key"]))
            if prm.get("dimension_mismatch"):
                yaml_dim.append((y["file"], prm["key"], prm["dimension_mismatch"]))

    param_default_hits = sum(len(r.get("parameter_default_numeric_hits", [])) for r in py)
    state_default_hits = sum(len(r.get("state_get_default_constant_hits", [])) for r in py)
    tensor_const_hits = sum(len(r.get("torch_tensor_constant_hits", [])) for r in py)

    summary = {
        "functor_count": len(functors),
        "missing_metadata_count": len(missing_meta),
        "missing_super_init_count": len(missing_super),
        "missing_forward_count": len(missing_forward),
        "bad_forward_signature_count": len(bad_forward),
        "legacy_compute_method_count": len(legacy_compute),
        "scientific_get_global_manager_count": len(sci_get_global),
        "manager_get_parameter_value_call_count": len(manager_getpv),
        "category_theory_import_count": len(category_theory_imports),
        "parameter_default_numeric_hits": param_default_hits,
        "state_get_default_constant_hits": state_default_hits,
        "torch_tensor_constant_hits": tensor_const_hits,
        "scientific_state_direct_write_file_count": len(state_direct_write_files),
        "metadata_missing_spatial_scale_count": len(meta_missing_scale),
        "metadata_unknown_spatial_scale_count": len(meta_unknown_scale),
        "metadata_missing_variable_kind_count": len(meta_missing_kind),
        "metadata_unknown_variable_kind_count": len(meta_unknown_kind),
        "metadata_empty_io_count": len(meta_empty_io),
        "yaml_missing_fields_count": len(yaml_missing),
        "yaml_empty_references_count": len(yaml_empty_ref),
        "yaml_dimension_mismatch_count": len(yaml_dim),
        "yaml_units_one_file_count": len(units1_files),
        "yaml_fraction_unit_file_count": len(frac_unit_files),
        "output_overlap_count": len(data.get("output_overlaps", {})),
        "theory_missing_candidate_count": len(data.get("theory_missing_candidates", [])),
        "exact_orphan_parameter_key_count": len(data.get("exact_orphan_parameter_keys", [])),
    }

    samples = {
        "legacy_compute": legacy_compute[:40],
        "scientific_get_global_manager": sci_get_global[:40],
        "metadata_missing_spatial_scale": meta_missing_scale[:40],
        "metadata_missing_variable_kind": meta_missing_kind[:40],
        "metadata_unknown_spatial_scale": meta_unknown_scale[:40],
        "metadata_unknown_variable_kind": meta_unknown_kind[:40],
        "yaml_dimension_mismatch": yaml_dim[:40],
        "output_overlaps": list(data.get("output_overlaps", {}).items())[:30],
        "theory_missing_candidates": data.get("theory_missing_candidates", [])[:60],
        "state_direct_write_files": state_direct_write_files[:60],
        "missing_forward": missing_forward[:40],
        "bad_forward_signature": bad_forward[:40],
    }

    payload = {"summary": summary, "samples": samples}
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] summary file:", OUT_PATH)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
