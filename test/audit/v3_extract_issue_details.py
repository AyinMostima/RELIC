import json
from pathlib import Path


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_issue_block(issues: dict, key: str, limit: int = 20):
    items = issues.get(key, [])
    print(f"{key}: {len(items)}")
    for item in items[:limit]:
        print(item)


def main():
    static_scan = load_json("test/audit_static_scan_v3.json")
    contract_scan = load_json("test/audit_functor_contract_scan_v3.json")

    print_section("STATIC_SCAN_COUNTS")
    print(static_scan.get("counts", {}))
    print("functor_classes:", len(static_scan.get("functor_classes", [])))
    print("issue_keys:", sorted(static_scan.get("issues", {}).keys()))

    print_section("CONTRACT_SCAN_COUNTS")
    print("class_count:", contract_scan.get("class_count", 0))
    print("issue_keys:", sorted(contract_scan.get("issues", {}).keys()))

    static_issues = static_scan.get("issues", {})
    contract_issues = contract_scan.get("issues", {})

    print_section("HIGH_SIGNAL_STATIC_ISSUES")
    for key in [
        "scientific_direct_global_manager",
        "torch_tensor_literal",
        "state_get_numeric_default",
        "parameter_value_numeric_default",
        "yaml_dimension_unit_mismatch",
        "yaml_missing_required_fields",
        "yaml_empty_references",
        "physical_constant_outside_ssot",
        "category_theory_runtime_import",
    ]:
        print_issue_block(static_issues, key, limit=30)

    print_section("HIGH_SIGNAL_CONTRACT_ISSUES")
    for key in [
        "metadata_missing_fields",
        "missing_get_dependencies_method",
        "scientific_get_global_manager_get",
        "forward_signature_noncanonical",
        "missing_forward",
        "metadata_spatial_unknown",
        "metadata_variable_kind_unknown",
        "metadata_empty_io",
    ]:
        print_issue_block(contract_issues, key, limit=30)


if __name__ == "__main__":
    main()
