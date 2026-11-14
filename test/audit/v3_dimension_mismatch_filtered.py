import json
from pathlib import Path

import yaml


UNIT_DIM_MAP = {
    "s-1": "frequency",
    "day-1": "frequency",
    "d-1": "frequency",
    "yr-1": "frequency",
    "j mol-1": "energy",
    "w m-2": "energy_flux",
    "kg m-2 s-1": "mass_flux",
    "m3 m-3": "volume_fraction",
    "m2 m-2": "area_ratio",
    "dimensionless": "dimensionless",
    "umol m-2 s-1": "molar_flux",
    "s": "time",
    "day": "time",
    "yr": "time",
    "k": "temperature",
    "degc": "temperature",
    "j kg-1 k-1": "specific_heat_capacity",
}

SYSTEM_CATEGORY_PREFIX = (
    "configuration",
    "computational",
    "numerical",
    "validation",
    "unit_conversion",
    "conversion_factor",
)


def is_system_category(cat: str) -> bool:
    c = (cat or "").strip().lower()
    return c.startswith(SYSTEM_CATEGORY_PREFIX)


def main():
    root = Path("relic/parameters/database/relic")
    mismatches = []
    for y in root.rglob("*.yaml"):
        data = yaml.safe_load(y.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        params = data.get("parameters", {})
        if not isinstance(params, dict):
            continue
        for key, spec in params.items():
            if not isinstance(spec, dict):
                continue
            units = str(spec.get("units", "")).strip().lower()
            dim = str(spec.get("dimension", "")).strip()
            cat = str(spec.get("category", "")).strip()
            if is_system_category(cat):
                continue
            expected = UNIT_DIM_MAP.get(units)
            if expected and dim and dim != expected:
                mismatches.append(
                    {
                        "file": str(y).replace("\\", "/"),
                        "parameter": key,
                        "category": cat,
                        "units": spec.get("units"),
                        "dimension": dim,
                        "expected": expected,
                    }
                )

    out = Path("test/audit/v3_dimension_mismatch_filtered.json")
    out.write_text(json.dumps(mismatches, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK]", out.as_posix())
    print("count", len(mismatches))
    for item in mismatches[:40]:
        print(item)


if __name__ == "__main__":
    main()
