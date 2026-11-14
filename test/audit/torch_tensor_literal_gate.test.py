"""
Audit gate for high-risk physical constants hardcoded in torch.tensor(...).
"""

from __future__ import annotations

from test.audit.v3_torch_tensor_literal_audit import collect_suspicious_literals


def test_no_high_risk_physical_constant_literals_in_torch_tensor_calls() -> None:
    """
    References:
    - RELIC architecture_overview.md (Physical Constants SSOT policy).
    - CODATA and standard atmospheric constants policy in parameter SSOT.

    Mathematical expression:
    - N/A (static AST gate).

    Parameter meanings:
    - literal: numeric value passed directly to torch.tensor(...).
    - constant_name: physical constant category that must come from SSOT.
    """
    findings = collect_suspicious_literals()
    assert not findings, (
        "Found high-risk physical constants hardcoded in torch.tensor(...). "
        "Use get_physical_constant(...) or parameter manager SSOT instead. "
        f"Examples: {findings[:10]}"
    )
