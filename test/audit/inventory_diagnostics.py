import json
from collections import defaultdict, Counter


def main() -> None:
    with open("analysis_results/scientific_functor_inventory.json", "r", encoding="utf-8") as f:
        inventory = json.load(f)
    with open("analysis_results/scientific_audit_issues.json", "r", encoding="utf-8") as f:
        issues = json.load(f)

    print(f"functor_count={len(inventory)}")
    print("issue_counts:")
    for k, v in Counter(i["type"] for i in issues).most_common():
        print(f"  {k}: {v}")

    print("missing_bridge_count:")
    missing_bridge = [c for c in inventory if c.get("bridge") is None]
    print(f"  {len(missing_bridge)}")
    for item in missing_bridge[:40]:
        print(f"    {item['class']} @ {item['file']}:{item['line']}")

    overlaps = defaultdict(list)
    for c in inventory:
        md = c.get("metadata_fields") or {}
        outs = md.get("output_variables")
        if isinstance(outs, list):
            for o in outs:
                overlaps[str(o)].append(c)

    print("output_overlaps_non_diagnostics:")
    found = False
    for var, producers in sorted(overlaps.items()):
        if var.startswith("diagnostics."):
            continue
        if len(producers) > 1:
            found = True
            print(f"  {var}: {len(producers)}")
            for p in producers[:10]:
                print(f"    {p['class']} @ {p['file']}:{p['line']}")
    if not found:
        print("  none")


if __name__ == "__main__":
    main()
