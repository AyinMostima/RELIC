"""IPHS expression helpers for FunctorMetadata.

References:
    - Ramirez and Le Gorrec style irreversible port-Hamiltonian systems:
      x_dot = J e_E + M e_S + G u + N r.
    - GENERIC irreversible thermodynamics: skew reversible structure,
      symmetric positive semidefinite irreversible structure, and
      nonnegative entropy production.
    - RELIC-IPHS-DEC plan: typed cochains, Hodge or Onsager mobility,
      ports, and ParameterManager-backed structural coefficients.

Mathematical expression:
    Functor metadata exposes only a non-strict discrete IPHS embedding template:
        x_dot = J_h e_E + M_h e_S + G_h u_h + N_h r_h
    The helper does not infer energy or entropy functionals, their gradients,
    operator properties, GENERIC degeneracy, or entropy-production closure.
    Those strict properties require certificate and executable runtime evidence.

Parameter meanings:
    Metadata parameter keys identify ParameterManager entries that provide
    material, kinetic, metric, Hodge, Onsager, port, or validation
    coefficients. This module never resolves parameter values directly.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

IPHS_MARKER = "Canonical IPHS:"


def ensure_iphs_equivalent_expression(
    base_expression: str,
    *,
    category: Any,
    variable_kind: Any = None,
    parameters: Mapping[str, Any] | None = None,
    conservation_laws: Sequence[str] | None = None,
    input_variables: Sequence[str] | None = None,
    output_variables: Sequence[str] | None = None,
    functor_name: str,
) -> str:
    """Return an idempotent canonical IPHS metadata expression.

    References:
        Irreversible port-Hamiltonian and GENERIC formulations require a
        skew reversible operator, a positive semidefinite dissipative
        operator, and nonnegative entropy production for irreversible terms.

    Mathematical expression:
        x_dot = J_h e_E + M_h e_S + G_h u_h + N_h r_h.

    Parameter meanings:
        Parameter keys listed in FunctorMetadata.parameters are treated as
        ParameterManager-backed coefficients for the IPHS operators and
        closure laws. No parameter value is hardcoded or resolved here.
    """
    base = "" if base_expression is None else str(base_expression).strip()
    if not base:
        return base
    if IPHS_MARKER in base:
        return base

    category_value = _category_value(category)
    variable_kind_value = _category_value(variable_kind)
    expression_role = _expression_role(
        variable_kind_value=variable_kind_value,
    )
    parameter_keys = _format_sequence(sorted(str(key) for key in (parameters or {})))
    input_keys = _format_sequence(input_variables or ())
    output_keys = _format_sequence(output_variables or ())
    law_keys = _format_sequence(conservation_laws or ())
    template = _category_template(category_value, expression_role)
    owner = functor_name.strip() or "unnamed_functor"
    structural_block = _structural_block(expression_role)

    return (
        f"{IPHS_MARKER}\n"
        f"  owner: {owner}\n"
        f"{structural_block}"
        f"  category mapping: {template}\n"
        f"  inputs: {input_keys}\n"
        f"  outputs: {output_keys}\n"
        f"  conserved quantities: {law_keys}\n"
        "  parameter access: theta is resolved only through "
        f"ParameterManager-backed metadata.parameters keys: {parameter_keys}\n"
        "  equivalence evidence: literature formulas are stored only in theory "
        "documents, equivalence certificates, and baseline tests"
    )


def _category_value(category: Any) -> str:
    if category is None:
        return ""
    value = getattr(category, "value", category)
    text = str(value).strip().lower()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text


def _expression_role(
    *,
    variable_kind_value: str,
) -> str:
    if variable_kind_value == "diagnostic":
        return "observation"
    return "state_advancement"


def _structural_block(expression_role: str) -> str:
    if expression_role == "observation":
        return (
            "  state embedding: x_h labels declared diagnostic inputs; "
            "typed cochain construction requires certificate or runtime evidence\n"
            "  observation: y_h = O_h x_h + B_h u_h\n"
            "  dynamics: diagnostic-only; no state advance is performed by this Functor\n"
            "  structure: O_h and B_h are non-strict mapping placeholders; "
            "typed chain-map properties are not inferred by metadata\n"
            "  entropy: diagnostic metadata makes no entropy-production claim\n"
        )
    return (
        "  state embedding: x_h labels declared process coordinates; "
        "typed cochain construction requires certificate or runtime evidence\n"
        "  efforts: energy and entropy gradients are not inferred by metadata\n"
        "  dynamics: x_dot = J_h e_E + M_h e_S + G_h u_h + N_h r_h(x_h, theta); "
        "non-strict embedding template; not executable proof\n"
        "  structure: operator properties and strict GENERIC degeneracy are not inferred "
        "by metadata\n"
        "  entropy: entropy-production closure is not inferred by metadata\n"
    )


def _format_sequence(values: Sequence[Any], *, max_items: int = 12) -> str:
    items = [str(value) for value in values if str(value)]
    if not items:
        return "none declared"
    if len(items) <= max_items:
        return ", ".join(items)
    visible = ", ".join(items[:max_items])
    return f"{visible}, and {len(items) - max_items} more"


def _category_template(category_value: str, expression_role: str = "state_advancement") -> str:
    if expression_role == "observation":
        return (
            "observation or diagnostic processes use y_h = O_h x_h + B_h u_h "
            "as a typed cochain map; they expose coupling signals without an "
            "independent state-advance law"
        )
    templates = {
        "transport_mechanism": (
            "passive transport uses X_h = -D_h phi_h, F_h = L_h X_h, "
            "x_dot = -D_h F_h + G_h u_h + s_h with L_h = L_h.T >= 0"
        ),
        "biological_process": (
            "reaction or growth terms use x_dot = N_h r_h(x_h, theta) "
            "+ M_h e_S + G_h u_h; strict use requires declared stoichiometry "
            "and entropy evidence"
        ),
        "physical_field": (
            "field diagnostics or balances contribute efforts, ports, or "
            "constraint audits without bypassing state-manager commits"
        ),
        "scale_coupling": (
            "cross-scale transfer uses conservative chain maps R_h with "
            "R_storage D_source ~= D_target R_flux when fluxes are mapped"
        ),
    }
    return templates.get(
        category_value,
        "generic process contribution mapped into IPHS ports, sources, or diagnostics",
    )


__all__ = ["IPHS_MARKER", "ensure_iphs_equivalent_expression"]
