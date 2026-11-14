from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = PROJECT_ROOT / "analysis_results" / "scientific_functor_inventory.json"


def main() -> None:
    data = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    expected = ["self", "state", "time", "dt"]

    nonstandard = []
    for row in data:
        sig = row.get("forward_signature", "")
        if not sig:
            continue
        args_str = sig[sig.find("(") + 1 : sig.rfind(")")]
        args = [x.strip() for x in args_str.split(",") if x.strip()]
        if args[:4] != expected:
            nonstandard.append(
                {
                    "file": row["file_path"],
                    "class": row["class_name"],
                    "signature": sig,
                }
            )

    print(f"[OK] nonstandard_forward={len(nonstandard)}")
    for item in nonstandard:
        print(f"{item['file']} | {item['class']} | {item['signature']}")


if __name__ == "__main__":
    main()
