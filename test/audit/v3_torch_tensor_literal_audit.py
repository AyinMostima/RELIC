"""
AST audit for suspicious torch.tensor numeric literals in scientific modules.

This script focuses on high-risk physical constants that must come from SSOT
(`relic/parameters/database/relic/physical/constants.yaml`) instead of inline
numeric literals inside `torch.tensor(...)`.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCIENTIFIC_ROOT = REPO_ROOT / "relic" / "scientific"
OUTPUT_JSON = REPO_ROOT / "test" / "audit" / "v3_torch_tensor_literal_audit.json"

# value -> semantic name
HIGH_RISK_CONSTANTS = {
    0.41: "von_karman_constant",
    9.81: "gravity",
    273.15: "celsius_to_kelvin_offset",
    287.058: "dry_air_gas_constant",
    1005.0: "specific_heat_air_constant_pressure",
    2450000.0: "latent_heat_vaporization_reference",
    6371000.0: "earth_radius",
}


def _is_torch_tensor_call(node: ast.Call) -> bool:
    if not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr != "tensor":
        return False
    value = node.func.value
    return isinstance(value, ast.Name) and value.id == "torch"


def _extract_numeric_literal(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _extract_numeric_literal(node.operand)
        if inner is not None:
            return -inner
    return None


def _matches_high_risk_constant(value: float) -> tuple[float, str] | None:
    for constant_value, constant_name in HIGH_RISK_CONSTANTS.items():
        tolerance = max(1e-12, abs(constant_value) * 1e-12)
        if abs(value - constant_value) <= tolerance:
            return constant_value, constant_name
    return None


def collect_suspicious_literals() -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path in SCIENTIFIC_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not _is_torch_tensor_call(node):
                continue
            if not node.args:
                continue

            literal_value = _extract_numeric_literal(node.args[0])
            if literal_value is None:
                continue

            hit = _matches_high_risk_constant(literal_value)
            if hit is None:
                continue
            constant_value, constant_name = hit
            findings.append(
                {
                    "file": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "line": node.lineno,
                    "literal": literal_value,
                    "constant_name": constant_name,
                    "expected_source": "get_physical_constant(...) or parameter manager SSOT",
                    "matched_constant": constant_value,
                }
            )
    return findings


def write_report(findings: list[dict[str, Any]]) -> None:
    OUTPUT_JSON.write_text(
        json.dumps(findings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    findings = collect_suspicious_literals()
    write_report(findings)
    print("[OK]", OUTPUT_JSON.as_posix())
    print("suspicious_count", len(findings))
    for item in findings[:50]:
        print(item)
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
