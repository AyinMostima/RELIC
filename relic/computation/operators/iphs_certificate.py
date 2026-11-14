"""IPHS equivalence certificates for scientific process replacement."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import inspect
import json
import keyword
import math
from pathlib import Path
from typing import Any, Callable, cast, get_args, Mapping, Optional

import yaml

from relic.parameters.spec import has_complete_parameter_metadata

from .iphs import IPHSFormulation

ALLOWED_CANONICAL_ROLES = {
    "transport",
    "reaction",
    "field",
    "scale_coupling",
    "port",
    "observation",
}

ALLOWED_THEORY_DOCUMENT_STATUSES = {"created", "updated", "already_current"}

ALLOWED_EQUIVALENCE_TYPES = {
    "exact",
    "thermodynamic_embedding",
    "port_equivalent",
    "diagnostic",
    "reaction_network",
}

IPHS_LEVEL_DEFINITIONS = {
    0: {
        "name": "ordinary_empirical",
        "description": "Ordinary empirical process with no IPHS structural claim.",
    },
    1: {
        "name": "iphs_compatible",
        "description": "Conservation, dissipation, port, or DEC metadata are compatible with IPHS.",
    },
    2: {
        "name": "iphs_wrapped",
        "description": "Canonical runtime uses an IPHS operator around an existing tendency.",
    },
    3: {
        "name": "iphs_structured",
        "description": "Forms, ports, Hodge, J/M/G/N, or assembly evidence are declared.",
    },
    4: {
        "name": "strict_iphs",
        "description": "Dissipative pH dynamics use (J-R)e_H+Gu with a closed-port dH audit.",
    },
    5: {
        "name": "strict_iphs_generic",
        "description": (
            "GENERIC dynamics require L e_E+M e_S, double degeneracy, and an "
            "executable Jacobi proof; the tensor-only proof path remains [N]."
        ),
    },
}

IPHS_LEVEL_REQUIRED_EVIDENCE = {
    1: ("classification", "compatible_basis"),
    2: ("runtime_replacement", "canonical_operator"),
    3: ("structural_contribution", "assembly_participation"),
    4: (
        "storage_functional",
        "effort_definition",
        "dissipation_audit",
        "port_power_audit",
    ),
    5: (
        "energy_entropy_pair",
        "generic_degeneracy",
        "entropy_production",
    ),
}

MAX_IPHS_LEVEL_BY_EQUIVALENCE_TYPE = {
    "diagnostic": 1,
}

MAX_IPHS_LEVEL_BY_ROLE = {
    "observation": 1,
}


def _tuple_of_text(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _parse_iphs_level(value: Any, *, allow_names: bool = True) -> int | None:
    if not allow_names and type(value) is not int:
        return None
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_")
        if normalized.startswith("level_"):
            normalized = normalized.split("level_", 1)[1]
        for level, definition in IPHS_LEVEL_DEFINITIONS.items():
            if normalized in {str(level), str(definition["name"]).lower()}:
                return level
    return None


def resolve_iphs_level_requirement(
    declarations: Mapping[str, Mapping[str, Any]],
    *,
    strict_iphs: bool = False,
    canonical_role: str = "",
) -> tuple[int | None, list[tuple[str, Any]]]:
    """Resolve one runtime requirement without coupling content strictness.

    Scopes label caller-provided mappings for existing error adapters. Metadata
    levels require exact integers; named certificate levels remain a separate
    input grammar. Descriptive levels 0-3 add no runtime floor, positive
    required_level values do, and explicit strict IPHS requests retain their
    role-appropriate minimum. No scientific evidence is created here.
    """
    if type(strict_iphs) is not bool:
        raise TypeError("strict_iphs must be bool.")
    levels: list[int] = []
    invalid: list[tuple[str, Any]] = []
    for scope, fields in declarations.items():
        for key in ("required_level", "iphs_level"):
            if key not in fields:
                continue
            value = fields[key]
            level = _parse_iphs_level(value, allow_names=False)
            if level is None or level not in IPHS_LEVEL_DEFINITIONS:
                invalid.append((f"{scope}.{key}", value))
            elif key == "required_level" or level >= 4:
                levels.append(level)
    if invalid:
        return None, invalid
    required = max(levels, default=0)
    role_cap = MAX_IPHS_LEVEL_BY_ROLE.get(str(canonical_role).strip().lower())
    if strict_iphs and (role_cap is None or role_cap >= 4):
        required = max(4, required)
    return (required if required > 0 else None), []


def _iphs_level_name(level: int | None) -> str:
    if level in IPHS_LEVEL_DEFINITIONS:
        return str(IPHS_LEVEL_DEFINITIONS[int(level)]["name"])
    return "invalid"


def _evidence_is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        return bool(value) and all(
            isinstance(key, str)
            and bool(key.strip())
            and _evidence_is_present(nested)
            for key, nested in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return bool(value) and all(_evidence_is_present(item) for item in value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized not in {"", "false", "none", "null", "0"}
    if isinstance(value, (int, float)):
        return not isinstance(value, bool) and math.isfinite(float(value))
    return True


def _required_evidence_keys(level: int) -> tuple[str, ...]:
    keys: list[str] = []
    for threshold in sorted(IPHS_LEVEL_REQUIRED_EVIDENCE):
        if level >= threshold:
            keys.extend(IPHS_LEVEL_REQUIRED_EVIDENCE[threshold])
    return tuple(keys)


def _mapping_text_contains(mapping: Mapping[str, Any], *tokens: str) -> bool:
    text = " ".join(str(key) + " " + str(value) for key, value in mapping.items())
    normalized = text.lower().replace(" ", "").replace("*", "")
    return all(token.lower().replace(" ", "").replace("*", "") in normalized for token in tokens)


def _normalized_formula(value: Any) -> str:
    return (
        _text(value)
        .lower()
        .replace(" ", "")
        .replace("*", "")
        .replace("^", "")
    )


def _validate_formulation_structure(
    *,
    level: int,
    iphs_expression: str,
    operator_map: Mapping[str, Any],
    state_advancement_map: Mapping[str, Any],
) -> list[str]:
    """Reject explicit dPH/GENERIC declaration contradictions."""

    issues: list[str] = []
    operator_text = _normalized_formula(
        " ".join(f"{key}={value}" for key, value in operator_map.items())
    )
    effort_text = _normalized_formula(
        state_advancement_map.get("effort_variable")
    )
    advancer_text = _normalized_formula(
        state_advancement_map.get("state_advancer")
    )
    expression_text = _normalized_formula(iphs_expression)
    if level == 4:
        required_tokens = ("j_h", "r_h", "e_h")
        if (
            not all(token in expression_text for token in required_tokens)
            or not all(token in advancer_text for token in required_tokens)
            or "j_h" not in operator_text
            or "r_h" not in operator_text
            or "e_h" not in effort_text
            or "m_h" in operator_text
            or "m_h" in advancer_text
            or "e_s" in effort_text
            or "e_s" in advancer_text
        ):
            issues.append(
                "Level 4 formulation structure must declare (J_h-R_h)e_H+G_h*u_h "
                "with J_h/R_h/e_H and without GENERIC M_h/e_S dissipation"
            )
    elif level == 5:
        reversible_present = "l_h" in operator_text or "j_h" in operator_text
        expression_reversible = "l_h" in expression_text or "j_h" in expression_text
        advancer_reversible = "l_h" in advancer_text or "j_h" in advancer_text
        if (
            not reversible_present
            or "m_h" not in operator_text
            or not expression_reversible
            or "m_h" not in expression_text
            or "e_e" not in expression_text
            or "e_s" not in expression_text
            or not advancer_reversible
            or "m_h" not in advancer_text
            or "e_e" not in advancer_text
            or "e_s" not in advancer_text
            or "r_h" in operator_text
            or "r_h" in advancer_text
            or "e_h" in effort_text
            or "e_h" in advancer_text
        ):
            issues.append(
                "Level 5 formulation structure must declare L_h*e_E+M_h*e_S+G_h*u_h "
                "with L_h(or J_h)/M_h/e_E/e_S and without dissipative-pH R_h/e_H"
            )
    return issues


_STRICT_EVIDENCE_REQUIRED_FIELDS = {
    "storage_functional": (
        "runtime_owner",
        "value_key",
        "state_keys",
        "units",
        "theory_anchor",
        "validation_test",
    ),
    "effort_definition": (
        "runtime_owner",
        "value_key",
        "derivative_of",
        "method",
        "validation_test",
    ),
    "dissipation_audit": (
        "runtime_owner",
        "operator_owner",
        "operator_kind",
        "value_key",
        "balance_key",
        "inequality",
        "tolerance_parameter_key",
        "validation_test",
    ),
    "port_power_audit": (
        "runtime_owner",
        "balance_key",
        "sign_convention",
        "tolerance_parameter_key",
        "validation_test",
    ),
    "energy_entropy_pair": (
        "energy_owner",
        "entropy_owner",
        "energy_key",
        "entropy_key",
        "validation_test",
    ),
    "generic_degeneracy": (
        "structure_owner",
        "j_entropy_residual_key",
        "m_energy_residual_key",
        "reversible_operator_state_dependence",
        "jacobi_identity_basis",
        "tolerance_parameter_key",
        "validation_test",
    ),
    "entropy_production": (
        "runtime_owner",
        "value_key",
        "lower_bound",
        "tolerance_parameter_key",
        "validation_test",
    ),
}

_RUNTIME_OWNER_FIELDS = {
    "runtime_owner",
    "operator_owner",
    "energy_owner",
    "entropy_owner",
    "structure_owner",
}

_FORBIDDEN_RUNTIME_EVIDENCE = {
    "assigned_zero",
    "constant_zero",
    "hardcoded_zero",
    "zeros_like",
    "true",
    "false",
}


def _normalized_path(value: Any) -> str:
    return str(value or "").strip().replace("/", "\\").lower()


_SOURCE_TREE_ROOT = Path(__file__).resolve().parents[3]


def _source_tree_path(value: Any) -> Path | None:
    """Resolve repository-relative evidence without leaving the source tree."""
    try:
        file_part = str(value or "").partition("::")[0].strip().replace("\\", "/")
        if not file_part:
            return None
        path = Path(file_part)
        if path.is_absolute():
            return None
        source_root = _SOURCE_TREE_ROOT.resolve()
        candidate = (source_root / path).resolve()
        candidate.relative_to(source_root)
    except (OSError, RuntimeError, ValueError):
        return None
    return candidate


def _source_tree_file_exists(value: Any) -> bool:
    """Return whether evidence resolves to a file inside the source checkout."""
    path = _source_tree_path(value)
    if path is None:
        return False
    try:
        return path.is_file()
    except (OSError, RuntimeError, ValueError):
        return False


_UNRESOLVED = object()


def _resolve_dotted_object(dotted: str) -> Any:
    """Import the longest module prefix of a dotted path and walk the remaining attributes."""
    parts = dotted.split(".")
    for split_index in range(len(parts), 0, -1):
        module_name = ".".join(parts[:split_index])
        try:
            value: Any = importlib.import_module(module_name)
        except (ImportError, ValueError):
            continue
        for attribute in parts[split_index:]:
            if not hasattr(value, attribute):
                return _UNRESOLVED
            value = getattr(value, attribute)
        return value
    return _UNRESOLVED


def _runtime_owner_resolves(owner: str) -> bool:
    value = _resolve_dotted_object(owner)
    return value is not _UNRESOLVED and callable(value)


def _is_json_scalar(value: Any) -> bool:
    """True for a JSON scalar: null, bool, finite number, or string."""
    if value is None or isinstance(value, (bool, str)):
        return True
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _validate_constructor_selection(
    selection: Mapping[str, Any],
    *,
    process_id: str,
    iphs_level: int | None,
) -> list[str]:
    """Check a constructor-scoped runtime-point selection against the process class.

    A non-empty selection marks the certificate as a constructor-scoped runtime
    point: it maps constructor keywords of the class named by ``process_id`` to
    JSON scalar values. It is a Level 4/5 runtime-point claim only; the main
    process certificate declares no selection.
    """
    issues: list[str] = []
    if not selection:
        return issues
    if iphs_level not in (4, 5):
        issues.append(
            "constructor_selection marks a constructor-scoped runtime-point "
            "certificate and requires iphs_level 4 or 5"
        )
    for key, value in selection.items():
        if not isinstance(key, str) or not key.isidentifier() or keyword.iskeyword(key):
            issues.append(
                f"constructor_selection key {key!r} must be a Python keyword identifier"
            )
        if not _is_json_scalar(value):
            issues.append(
                f"constructor_selection value for {key!r} must be a JSON scalar "
                "(null, bool, finite number, or string)"
            )
    process_class = _resolve_dotted_object(process_id) if process_id else _UNRESOLVED
    if not isinstance(process_class, type):
        issues.append(
            f"constructor_selection requires process_id to resolve to an importable "
            f"process class; got {process_id!r}"
        )
        return issues
    keywords = {
        name
        for name, parameter in inspect.signature(process_class).parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    for key in selection:
        if isinstance(key, str) and key not in keywords:
            issues.append(
                f"constructor_selection key {key!r} is not a constructor keyword of "
                f"{process_class.__module__}.{process_class.__qualname__}"
            )
    return issues


def _constructor_selection_payload(value: Any) -> dict[str, Any]:
    """Load the optional constructor_selection mapping without dropping a malformed value."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(
            "constructor_selection must be a mapping of constructor keyword to JSON "
            f"scalar; got {type(value).__name__}"
        )
    return dict(value)


_CANONICAL_STRICT_OPERATOR_OWNERS = {
    "relic.computation.operators.iphs.IPHSStructureSpec.canonical_drift",
    "relic.computation.operators.iphs.IPHSStructureSpec.validate",
    "relic.computation.operators.iphs.advance_reaction_network",
    "relic.computation.operators.iphs.assemble_iphs_structure",
}


def _runtime_owner_binds_claimed_process(
    owner: str,
    process_id: str,
    *,
    allow_canonical_operator: bool,
) -> bool:
    """Require executable evidence to bind the claimed process or canonical operator."""

    if not _runtime_owner_resolves(owner):
        return False
    process_prefix = f"{process_id}." if process_id else ""
    if process_prefix and owner.startswith(process_prefix):
        return True
    return allow_canonical_operator and owner in _CANONICAL_STRICT_OPERATOR_OWNERS


def _validation_test_is_bound(
    validation_test: Any,
    declared_tests: tuple[str, ...],
) -> bool:
    normalized = _normalized_path(validation_test)
    if not normalized or normalized not in {_normalized_path(item) for item in declared_tests}:
        return False
    return _source_tree_file_exists(validation_test)


def _contains_forbidden_runtime_placeholder(value: Any) -> str:
    if isinstance(value, Mapping):
        for nested in value.values():
            match = _contains_forbidden_runtime_placeholder(nested)
            if match:
                return match
        return ""
    if isinstance(value, (list, tuple, set)):
        for nested in value:
            match = _contains_forbidden_runtime_placeholder(nested)
            if match:
                return match
        return ""
    normalized = str(value).strip().lower().replace("(", "_").replace(")", "")
    for token in _FORBIDDEN_RUNTIME_EVIDENCE:
        if token in normalized:
            return token
    return ""


def _strict_evidence_names(level: int) -> tuple[str, ...]:
    names = list(IPHS_LEVEL_REQUIRED_EVIDENCE[4])
    if level >= 5:
        names.extend(IPHS_LEVEL_REQUIRED_EVIDENCE[5])
    return tuple(names)


def _strict_tolerance_parameter_keys(
    *,
    level: int,
    evidence: Mapping[str, Any],
) -> tuple[str, ...]:
    keys = {
        _text(entry.get("tolerance_parameter_key"))
        for evidence_name in _strict_evidence_names(level)
        if isinstance((entry := evidence.get(evidence_name)), Mapping)
        and _text(entry.get("tolerance_parameter_key"))
    }
    return tuple(sorted(keys))


def _validate_strict_level_evidence(
    *,
    level: int,
    evidence: Mapping[str, Any],
    validation_tests: tuple[str, ...],
    interface_variables: set[str],
    process_id: str,
    parameter_resolver: Callable[[str], Any],
) -> list[str]:
    issues: list[str] = []
    for evidence_name in _strict_evidence_names(level):
        raw_entry = evidence.get(evidence_name)
        if not isinstance(raw_entry, Mapping):
            issues.append(
                f"{evidence_name} must be a structured runtime binding, not prose"
            )
            continue
        entry = dict(raw_entry)
        required_fields = _STRICT_EVIDENCE_REQUIRED_FIELDS[evidence_name]
        missing_fields: list[str] = []
        for field_name in required_fields:
            field_value = entry.get(field_name)
            if (
                field_name not in entry
                or field_value is None
                or (isinstance(field_value, str) and not field_value.strip())
                or (
                    isinstance(field_value, (list, tuple, set, dict))
                    and not field_value
                )
            ):
                missing_fields.append(field_name)
        if missing_fields:
            issues.append(
                f"{evidence_name} structured runtime binding must define: "
                + ",".join(missing_fields)
            )
            continue

        placeholder = _contains_forbidden_runtime_placeholder(entry)
        if placeholder:
            issues.append(
                f"{evidence_name} contains forbidden runtime placeholder '{placeholder}'"
            )

        for owner_field in _RUNTIME_OWNER_FIELDS.intersection(entry):
            owner = _text(entry.get(owner_field))
            if not _runtime_owner_binds_claimed_process(
                owner,
                process_id,
                allow_canonical_operator=True,
            ):
                issues.append(
                    f"{evidence_name}.{owner_field} must be bound to the claimed process "
                    "or canonical IPHS operator"
                )

        validation_test = entry.get("validation_test")
        if not _validation_test_is_bound(validation_test, validation_tests):
            issues.append(
                f"{evidence_name}.validation_test must be declared and exist"
            )

        if evidence_name == "storage_functional":
            state_keys = set(_tuple_of_text(entry.get("state_keys")))
            if not state_keys or not state_keys.issubset(interface_variables):
                issues.append(
                    "storage_functional.state_keys must bind declared interface variables"
                )
        elif evidence_name == "effort_definition":
            if _text(entry.get("derivative_of")) != "storage_functional":
                issues.append("effort_definition.derivative_of must be storage_functional")
            if _text(entry.get("method")).lower() not in {"analytic", "autograd"}:
                issues.append("effort_definition.method must be analytic or autograd")
        elif evidence_name == "port_power_audit":
            if _text(entry.get("sign_convention")).lower() not in {
                "into_system",
                "out_of_system",
            }:
                issues.append(
                    "port_power_audit.sign_convention must be into_system or out_of_system"
                )
        elif evidence_name == "dissipation_audit":
            if level == 4:
                if _text(entry.get("operator_kind")) != "resistance_R":
                    issues.append(
                        "Level 4 dissipation_audit.operator_kind must be resistance_R"
                    )
                if _text(entry.get("value_key")) != "energy_dissipation":
                    issues.append(
                        "Level 4 dissipation_audit.value_key must be energy_dissipation"
                    )
                if _text(entry.get("balance_key")) != "closed_port_energy_rate":
                    issues.append(
                        "Level 4 dissipation_audit.balance_key must be closed_port_energy_rate"
                    )
                if _text(entry.get("inequality")) != "dH_dt<=port_power":
                    issues.append(
                        "Level 4 dissipation_audit.inequality must be dH_dt<=port_power"
                    )
            else:
                if _text(entry.get("operator_kind")) != "mobility_M":
                    issues.append(
                        "Level 5 dissipation_audit.operator_kind must be mobility_M"
                    )
                if _text(entry.get("value_key")) != "entropy_production":
                    issues.append(
                        "Level 5 dissipation_audit.value_key must be entropy_production"
                    )
                if _text(entry.get("inequality")) != "dS_dt>=port_entropy_flow":
                    issues.append(
                        "Level 5 dissipation_audit.inequality must be "
                        "dS_dt>=port_entropy_flow"
                    )
        elif evidence_name == "generic_degeneracy":
            dependence = _text(
                entry.get("reversible_operator_state_dependence")
            )
            if dependence not in {"constant", "state_dependent", "zero_field"}:
                issues.append(
                    "generic_degeneracy.reversible_operator_state_dependence must be "
                    "constant, state_dependent, or zero_field"
                )
            elif dependence == "zero_field":
                if _text(entry.get("jacobi_identity_basis")) != "canonical_zero_field":
                    issues.append(
                        "zero_field generic_degeneracy.jacobi_identity_basis must be "
                        "canonical_zero_field, the executable basis verified by "
                        "IPHSStructureSpec.validate"
                    )
            elif dependence == "constant":
                if _text(entry.get("jacobi_identity_basis")) != "constant_field_exact":
                    issues.append(
                        "constant generic_degeneracy.jacobi_identity_basis must be "
                        "constant_field_exact, the executable basis produced by "
                        "IPHSStructureSpec.validate for poisson_operator constant_field; "
                        "text jacobi_identity_basis is not Level 5 evidence"
                    )
            elif dependence == "state_dependent":
                if _text(entry.get("jacobi_identity_basis")) != "sampled_domain_jacobi_residual":
                    issues.append(
                        "state-dependent generic_degeneracy.jacobi_identity_basis must be "
                        "sampled_domain_jacobi_residual, the executable basis produced by "
                        "IPHSStructureSpec.validate for poisson_operator state_field; "
                        "text jacobi_identity_basis is not Level 5 evidence"
                    )
                else:
                    sample_count = entry.get("jacobi_sample_count")
                    tolerance = entry.get("jacobi_tolerance")
                    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
                        issues.append(
                            "state-dependent generic_degeneracy.jacobi_sample_count must be a "
                            "positive integer number of declared domain samples"
                        )
                    if not _is_finite_number(tolerance) or float(tolerance) <= 0.0:
                        issues.append(
                            "state-dependent generic_degeneracy.jacobi_tolerance must be a "
                            "positive finite residual tolerance"
                        )
        elif evidence_name == "entropy_production":
            lower_bound = entry.get("lower_bound")
            if isinstance(lower_bound, bool) or not isinstance(lower_bound, (int, float)):
                issues.append("entropy_production.lower_bound must be numeric zero")
            elif float(lower_bound) != 0.0:
                issues.append("entropy_production.lower_bound must be zero")

        tolerance_key = _text(entry.get("tolerance_parameter_key"))
        if tolerance_key:
            # The resolver (ParameterManager.get_parameter) returns None for
            # a missing key, which is reported below; a resolver failure
            # propagates and is never reported as incomplete metadata.
            parameter_spec = parameter_resolver(tolerance_key)
            if not has_complete_parameter_metadata(parameter_spec):
                issues.append(
                    f"{evidence_name}.tolerance_parameter_key '{tolerance_key}' "
                    "must resolve to complete ParameterManager metadata"
                )
    if len(_strict_tolerance_parameter_keys(level=level, evidence=evidence)) > 1:
        issues.append(
            "strict IPHS evidence must use a single tolerance_parameter_key "
            "across all balance, dissipation, degeneracy, and entropy audits"
        )
    return issues


_RUNTIME_POINT_STATUS = "runtime_point_verified"
_RUNTIME_POINT_SCOPE = "runtime_point"
_GOVERNED_EXECUTION_ENTRY = "FunctorBase.__call__"


def _validate_process_bound_runtime_audit(
    *,
    level: int,
    formulation: str,
    output_variables: tuple[str, ...],
    audit: Any,
) -> list[str]:
    """Check one process-bound runtime-point audit against the certificate.

    References:
      - Grmela and Ottinger (1997), GENERIC degeneracy and entropy production.
      - van der Schaft and Jeltsema (2014), dissipative port-Hamiltonian audits.
    Mathematical expression:
      the audit is the dict returned by IPHSStructureSpec.validate for
      required_level >= 4 with bound process potentials; the certificate
      accepts it only when runtime_verified_level >= level, the formulation
      matches, sigma >= 0, reaction closure rates are consistent, and a
      governed execution witness names the committed state outputs.
    Parameter meanings:
      level: certificate iphs_level; formulation: certificate formulation;
      output_variables: certificate outputs whose non-diagnostic members must
      appear in the committed-variable witness.
    Units: audit values are dimensionless residuals or the process units
      declared by the structure; no conversion is performed here.
    Domain/constraints: audit must be a mapping produced by the canonical
      operator on the actual process; text placeholders are not evidence.
    Sign convention: entropy production and reaction entropy rate must be
      non-negative; the closed reaction energy rate must vanish.
    """

    prefix = "runtime_point_audit"
    if not isinstance(audit, Mapping):
        return [f"{prefix} must be the mapping returned by IPHSStructureSpec.validate"]
    issues: list[str] = []
    verified_level = audit.get("runtime_verified_level")
    if (
        isinstance(verified_level, bool)
        or not isinstance(verified_level, int)
        or verified_level < level
    ):
        issues.append(
            f"{prefix}.runtime_verified_level must be an integer >= {level}; "
            f"got {verified_level!r}"
        )
    required_level = audit.get("required_level")
    if (
        isinstance(required_level, bool)
        or not isinstance(required_level, int)
        or required_level < level
    ):
        issues.append(
            f"{prefix}.required_level must be an integer >= {level}; got {required_level!r}"
        )
    if _text(audit.get("process_level_status")) != _RUNTIME_POINT_STATUS:
        issues.append(
            f"{prefix}.process_level_status must be {_RUNTIME_POINT_STATUS}; "
            f"got {audit.get('process_level_status')!r}"
        )
    if _text(audit.get("evidence_scope")) != _RUNTIME_POINT_SCOPE:
        issues.append(
            f"{prefix}.evidence_scope must be {_RUNTIME_POINT_SCOPE}; "
            f"got {audit.get('evidence_scope')!r}"
        )
    if _text(audit.get("formulation")) != _text(formulation):
        issues.append(
            f"{prefix}.formulation must equal the certificate formulation "
            f"{formulation!r}; got {audit.get('formulation')!r}"
        )
    if level >= 5:
        issues.extend(_audit_level_five_terms(prefix, audit))
    if level == 4:
        issues.extend(_audit_level_four_terms(prefix, audit))
    witness = audit.get("governed_execution")
    if not isinstance(witness, Mapping):
        issues.append(
            f"{prefix}.governed_execution witness mapping is required "
            "(entry, functor, committed_variables, step)"
        )
        return issues
    issues.extend(_audit_governed_witness(prefix, witness, output_variables))
    return issues


def _is_finite_number(value: Any) -> bool:
    """True for a finite int or float that is not a bool."""
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


_EXECUTABLE_JACOBI_BASES = frozenset(
    {"canonical_zero_field", "constant_field_exact", "sampled_domain_jacobi_residual"}
)


def _audit_level_five_terms(prefix: str, audit: Mapping[str, Any]) -> list[str]:
    """Level 5 GENERIC terms: executable Jacobi basis, degeneracy residuals, entropy rates.

    The Jacobi basis is canonical_zero_field, constant_field_exact, or the
    sampled_domain_jacobi_residual of a state field. The sampled basis is
    finite-point evidence: the declared samples plus the executed runtime point,
    whose skew defect and Jacobi residual the runtime audit must report within
    the same tolerance.
    """
    issues: list[str] = []
    basis = _text(audit.get("jacobi_identity_basis"))
    if audit.get("jacobi_identity_verified") is not True or basis not in _EXECUTABLE_JACOBI_BASES:
        issues.append(
            f"{prefix}.jacobi_identity_basis must be one of "
            f"{sorted(_EXECUTABLE_JACOBI_BASES)} with jacobi_identity_verified for Level 5"
        )
    elif basis == "sampled_domain_jacobi_residual":
        residual = audit.get("jacobi_max_residual")
        tolerance = audit.get("jacobi_tolerance")
        count = audit.get("jacobi_sample_count")
        tolerance_valid = _is_finite_number(tolerance)
        if not _is_finite_number(residual) or not tolerance_valid or float(residual) > float(tolerance):
            issues.append(
                f"{prefix}.jacobi_max_residual must be a finite number within jacobi_tolerance"
            )
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            issues.append(f"{prefix}.jacobi_sample_count must be a positive integer")
        point_residual = audit.get("runtime_point_jacobi_residual")
        if (
            not _is_finite_number(point_residual)
            or not tolerance_valid
            or float(point_residual) > float(tolerance)
        ):
            issues.append(
                f"{prefix}.runtime_point_jacobi_residual must be a finite number within "
                "jacobi_tolerance at the executed runtime point"
            )
        if not _is_finite_number(audit.get("runtime_point_skew_residual")):
            issues.append(f"{prefix}.runtime_point_skew_residual must be a finite number")
    for key in ("J_entropy_gradient_residual", "M_energy_gradient_residual"):
        if not _is_finite_number(audit.get(key)):
            issues.append(f"{prefix}.{key} must be a finite number")
    for key in ("entropy_production", "reaction_entropy_rate"):
        if key not in audit:
            if key == "entropy_production":
                issues.append(f"{prefix}.{key} is required")
            continue
        value = audit.get(key)
        if not _is_finite_number(value):
            issues.append(f"{prefix}.{key} must be a finite number")
        elif float(value) < 0.0:
            issues.append(f"{prefix}.{key} must be non-negative; got {value!r}")
    if "reaction_energy_rate" in audit and not _is_finite_number(audit.get("reaction_energy_rate")):
        issues.append(f"{prefix}.reaction_energy_rate must be a finite number")
    return issues


def _audit_level_four_terms(prefix: str, audit: Mapping[str, Any]) -> list[str]:
    """Level 4 dissipative pH terms: storage kind and reaction energy rate sign."""
    issues: list[str] = []
    storage_kind = audit.get("storage_kind")
    if storage_kind is not None and _text(storage_kind) not in {"total_energy", "free_energy"}:
        issues.append(
            f"{prefix}.storage_kind must be total_energy or free_energy; got {storage_kind!r}"
        )
    if "reaction_energy_rate" in audit:
        value = audit.get("reaction_energy_rate")
        if not _is_finite_number(value):
            issues.append(f"{prefix}.reaction_energy_rate must be a finite number")
        elif storage_kind is None:
            issues.append(f"{prefix}.storage_kind is required with reaction_energy_rate")
        elif _text(storage_kind) == "total_energy" and float(value) != 0.0:
            issues.append(
                f"{prefix}.reaction_energy_rate must vanish for a total_energy storage; got {value!r}"
            )
        elif _text(storage_kind) == "free_energy" and float(value) > 0.0:
            issues.append(
                f"{prefix}.reaction_energy_rate must be non-positive for a free_energy storage; got {value!r}"
            )
    return issues


def _audit_governed_witness(
    prefix: str, witness: Mapping[str, Any], output_variables: tuple[str, ...]
) -> list[str]:
    """Governed execution witness: entry point, functor, committed outputs, step."""
    issues: list[str] = []
    if _text(witness.get("entry")) != _GOVERNED_EXECUTION_ENTRY:
        issues.append(
            f"{prefix}.governed_execution.entry must be {_GOVERNED_EXECUTION_ENTRY}; "
            f"got {witness.get('entry')!r}"
        )
    if not _text(witness.get("functor")):
        issues.append(f"{prefix}.governed_execution.functor must name the executed Functor")
    committed = set(_tuple_of_text(witness.get("committed_variables")))
    required_outputs = {
        name for name in output_variables if not str(name).startswith("diagnostics.")
    }
    if not committed or not required_outputs.issubset(committed):
        issues.append(
            f"{prefix}.governed_execution.committed_variables must cover the certified "
            f"state outputs {sorted(required_outputs)}"
        )
    step = witness.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        issues.append(
            f"{prefix}.governed_execution.step must be a positive accepted step count; "
            f"got {step!r}"
        )
    return issues


def _validate_strict_runtime_replacement(
    runtime_replacement_map: Mapping[str, Any],
    validation_tests: tuple[str, ...],
    process_id: str,
) -> list[str]:
    issues: list[str] = []
    legacy_never_existed = runtime_replacement_map.get("legacy_never_existed", False)
    if not isinstance(legacy_never_existed, bool):
        issues.append("runtime_replacement_map.legacy_never_existed must be bool")
        legacy_never_existed = False
    required_fields = (
        "runtime_uses_iphs",
        "legacy_removed",
        "runtime_owner",
        "removed_legacy_symbols",
        "validation_test",
    )
    missing = [
        key
        for key in required_fields
        if not (key == "removed_legacy_symbols" and legacy_never_existed)
        and (
            key not in runtime_replacement_map
            or runtime_replacement_map.get(key) is None
            or runtime_replacement_map.get(key) == ""
            or runtime_replacement_map.get(key) == ()
            or runtime_replacement_map.get(key) == []
        )
    ]
    if missing:
        issues.append(
            "strict runtime_replacement_map must define executable bindings: "
            + ",".join(missing)
        )
        return issues
    runtime_uses_iphs = runtime_replacement_map.get("runtime_uses_iphs")
    legacy_removed = runtime_replacement_map.get("legacy_removed")
    if (
        not isinstance(runtime_uses_iphs, bool)
        or not isinstance(legacy_removed, bool)
        or not runtime_uses_iphs
        or not legacy_removed
    ):
        issues.append(
            "strict runtime_replacement_map must confirm boolean "
            "runtime_uses_iphs and legacy_removed"
        )
    removed_symbols = runtime_replacement_map.get("removed_legacy_symbols")
    if legacy_never_existed and removed_symbols in (None, [], ()):
        removed_symbols = ()
    if not isinstance(removed_symbols, (list, tuple)):
        issues.append(
            "runtime_replacement_map.removed_legacy_symbols must be a sequence "
            "of unique exact strings"
        )
    else:
        normalized_symbols = tuple(removed_symbols)
        if (
            any(
                not isinstance(symbol, str)
                or not symbol
                or symbol != symbol.strip()
                for symbol in normalized_symbols
            )
            or len(set(normalized_symbols)) != len(normalized_symbols)
        ):
            issues.append(
                "runtime_replacement_map.removed_legacy_symbols must contain "
                "unique non-empty exact strings"
            )
    owner = _text(runtime_replacement_map.get("runtime_owner"))
    if not _runtime_owner_binds_claimed_process(
        owner,
        process_id,
        allow_canonical_operator=False,
    ):
        issues.append(
            "runtime_replacement_map.runtime_owner must be bound to the claimed process"
        )
    if not _validation_test_is_bound(
        runtime_replacement_map.get("validation_test"),
        validation_tests,
    ):
        issues.append(
            "runtime_replacement_map.validation_test must be declared and exist"
        )
    return issues


@dataclass(frozen=True)
class IPHSEquivalenceCertificate:
    """Certificate that a scientific process is replaced by an IPHS form.

    References:
      - van der Schaft and Maschke (1995), port-Hamiltonian systems.
      - Grmela and Ottinger (1997), GENERIC formalism.
      - Hairer, Lubich, and Wanner (2006), structure-preserving integration.
    Mathematical expressions:
      legacy_f(x, theta, t) == canonical_f(x, theta, t), where
      dissipative PH uses x_dot = (J_h - R_h)e_H + G_ext u + N_h r_h,
      and GENERIC uses x_dot = L_h e_E + M_h e_S + G_ext u + N_h r_h.
    Parameter meanings:
      parameter_keys: ParameterManager keys used by both forms.
      input_variables/output_variables: runtime interface contract.
      composition_ports: declared ports used for module composition.
      J_h or L_h: skew reversible operator; R_h: positive semidefinite loss.
      M_h: positive semidefinite GENERIC entropy mobility; G_ext: external ports.
      constructor_selection: optional constructor keyword -> JSON scalar mapping;
        a non-empty mapping marks a constructor-scoped Level 4/5 runtime-point
        certificate of the process class, never the main process certificate.
    """

    module_name: str
    legacy_expression: str
    iphs_expression: str
    input_variables: tuple[str, ...]
    output_variables: tuple[str, ...]
    parameter_keys: tuple[str, ...]
    parameter_free: bool = False
    formulation: IPHSFormulation = "unspecified"
    process_id: str = ""
    canonical_role: str = ""
    iphs_level: int | None = None
    iphs_level_evidence: Mapping[str, Any] = field(default_factory=dict)
    theory_document: str = ""
    theory_document_status: str = ""
    literature_formula: str = ""
    state_map: Mapping[str, Any] = field(default_factory=dict)
    parameter_map: Mapping[str, Any] = field(default_factory=dict)
    operator_map: Mapping[str, Any] = field(default_factory=dict)
    state_advancement_map: Mapping[str, Any] = field(default_factory=dict)
    unit_map: Mapping[str, Any] = field(default_factory=dict)
    input_output_map: Mapping[str, Any] = field(default_factory=dict)
    composition_ports: tuple[str, ...] = field(default_factory=tuple)
    references: tuple[str, ...] = field(default_factory=tuple)
    validation_tests: tuple[str, ...] = field(default_factory=tuple)
    runtime_replacement_map: Mapping[str, Any] = field(default_factory=dict)
    parameter_management_map: Mapping[str, Any] = field(default_factory=dict)
    composition_management_map: Mapping[str, Any] = field(default_factory=dict)
    dependency_interconnection_map: Mapping[str, Any] = field(default_factory=dict)
    overflow_routing_map: Mapping[str, Any] = field(default_factory=dict)
    conservation_map: Mapping[str, Any] = field(default_factory=dict)
    entropy_audit: Mapping[str, Any] = field(default_factory=dict)
    stochastic_sdae_map: Mapping[str, Any] = field(default_factory=dict)
    noise_covariance_map: Mapping[str, Any] = field(default_factory=dict)
    conservative_noise_flux_map: Mapping[str, Any] = field(default_factory=dict)
    descriptor_constraint_map: Mapping[str, Any] = field(default_factory=dict)
    jump_reset_map: Mapping[str, Any] = field(default_factory=dict)
    stochastic_reproducibility_map: Mapping[str, Any] = field(default_factory=dict)
    equivalence_type: str = ""
    interface_map: Mapping[str, Any] = field(default_factory=dict)
    certificate_version: str = "1.0"
    constructor_selection: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "IPHSEquivalenceCertificate":
        """Build a certificate from a JSON/YAML mapping."""

        interface_map = _mapping(payload.get("interface_map"))
        input_variables = _tuple_of_text(
            payload.get("input_variables")
            or interface_map.get("input_variables")
            or payload.get("legacy_inputs")
        )
        output_variables = _tuple_of_text(
            payload.get("output_variables")
            or interface_map.get("output_variables")
            or payload.get("legacy_outputs")
        )
        parameter_keys = _tuple_of_text(
            payload.get("parameter_keys")
            or payload.get("iphs_parameter_keys")
            or payload.get("parameters")
        )
        return cls(
            module_name=_text(
                payload.get("module_name")
                or payload.get("name")
                or payload.get("process_id")
            ),
            legacy_expression=_text(
                payload.get("legacy_expression")
                or payload.get("legacy_formula")
            ),
            iphs_expression=_text(
                payload.get("iphs_expression")
                or payload.get("iphs_formula")
            ),
            input_variables=input_variables,
            output_variables=output_variables,
            parameter_keys=parameter_keys,
            parameter_free=bool(payload.get("parameter_free", False)),
            formulation=cast(
                IPHSFormulation,
                _text(payload.get("formulation")) or "unspecified",
            ),
            process_id=_text(payload.get("process_id") or payload.get("module_name")),
            canonical_role=_text(payload.get("canonical_role")),
            iphs_level=_parse_iphs_level(
                payload.get("iphs_level")
                if "iphs_level" in payload
                else payload.get("iphs_standardization_level")
            ),
            iphs_level_evidence=_mapping(payload.get("iphs_level_evidence")),
            theory_document=_text(payload.get("theory_document")),
            theory_document_status=_text(payload.get("theory_document_status")),
            literature_formula=_text(payload.get("literature_formula")),
            state_map=_mapping(payload.get("state_map")),
            parameter_map=_mapping(payload.get("parameter_map")),
            operator_map=_mapping(payload.get("operator_map")),
            state_advancement_map=_mapping(payload.get("state_advancement_map")),
            unit_map=_mapping(payload.get("unit_map")),
            input_output_map=_mapping(payload.get("input_output_map")),
            composition_ports=_tuple_of_text(
                payload.get("composition_ports")
                or payload.get("ports")
                or _mapping(payload.get("composition_management_map")).get("ports")
            ),
            references=_tuple_of_text(payload.get("references")),
            validation_tests=_tuple_of_text(
                payload.get("validation_tests") or payload.get("baseline_tests")
            ),
            runtime_replacement_map=_mapping(payload.get("runtime_replacement_map")),
            parameter_management_map=_mapping(payload.get("parameter_management_map")),
            composition_management_map=_mapping(payload.get("composition_management_map")),
            dependency_interconnection_map=_mapping(payload.get("dependency_interconnection_map")),
            overflow_routing_map=_mapping(payload.get("overflow_routing_map")),
            conservation_map=_mapping(payload.get("conservation_map")),
            entropy_audit=_mapping(payload.get("entropy_audit")),
            stochastic_sdae_map=_mapping(payload.get("stochastic_sdae_map")),
            noise_covariance_map=_mapping(payload.get("noise_covariance_map")),
            conservative_noise_flux_map=_mapping(payload.get("conservative_noise_flux_map")),
            descriptor_constraint_map=_mapping(payload.get("descriptor_constraint_map")),
            jump_reset_map=_mapping(payload.get("jump_reset_map")),
            stochastic_reproducibility_map=_mapping(
                payload.get("stochastic_reproducibility_map")
            ),
            equivalence_type=_text(payload.get("equivalence_type")),
            interface_map=interface_map,
            certificate_version=str(payload.get("certificate_version", "1.0")).strip() or "1.0",
            constructor_selection=_constructor_selection_payload(
                payload.get("constructor_selection")
            ),
        )

    def strict_tolerance_parameter_keys(self) -> tuple[str, ...]:
        """Return the unique strict-audit tolerance keys declared by this certificate."""

        if self.iphs_level is None or self.iphs_level < 4:
            return ()
        return _strict_tolerance_parameter_keys(
            level=self.iphs_level,
            evidence=self.iphs_level_evidence,
        )

    def declares_fail_closed_nonimplementation(self) -> bool:
        """Return whether this certificate declares an exact unavailable boundary.

        This predicate classifies executable availability independently from
        certificate validity. A true result remains non-implementation evidence;
        it does not make an intentionally incomplete level-0 certificate valid.
        """

        replacement_entrypoint = _text(
            self.runtime_replacement_map.get("replacement_entrypoint")
        ).lower()
        return (
            self.iphs_level == 0
            and self.runtime_replacement_map.get("operational_claim") is False
            and self.runtime_replacement_map.get("runtime_uses_iphs") is False
            and self.runtime_replacement_map.get("legacy_removed") is False
            and replacement_entrypoint.startswith("none_")
            and not self.input_variables
            and not self.output_variables
            and not self.composition_ports
            and self.iphs_expression.lstrip().startswith("[N]")
        )

    def validate(
        self,
        *,
        strict: bool = True,
        required_level: int | None = None,
        parameter_resolver: Optional[Callable[[str], Any]] = None,
        runtime_point_audit: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Validate certificate completeness and runtime evidence level.

        ``strict`` controls full certificate-map validation. ``required_level`` is
        a caller-owned runtime acceptance threshold, independent of content
        strictness. A zero floor adds no runtime evidence obligation; positive
        floors compare against executable evidence, never the declared level.
        The certificate remains the sole level owner, and its own Level 4/5
        claims retain every corresponding scientific and runtime requirement.
        """

        issues: list[str] = []
        runtime_evidence_issues: list[str] = []
        accepted_required_level: int | None = None
        runtime_binding = "descriptive"
        if runtime_point_audit is not None and (
            self.iphs_level is None or self.iphs_level < 4
        ):
            issues.append(
                "runtime_point_audit is accepted only for Level 4 or 5 certificates"
            )
        if required_level is not None:
            accepted_required_level, invalid_levels = resolve_iphs_level_requirement(
                {"caller": {"required_level": required_level}},
            )
            if invalid_levels:
                issues.append(
                    "required_level must be an integer IPHS level from 0 through 5"
                )
        if not self.module_name:
            issues.append("module_name is required")
        if not self.process_id:
            issues.append("process_id is required")
        if self.canonical_role not in ALLOWED_CANONICAL_ROLES:
            issues.append(
                "canonical_role must be one of: "
                + ",".join(sorted(ALLOWED_CANONICAL_ROLES))
            )
        if self.iphs_level is None:
            issues.append("iphs_level is required")
        elif self.iphs_level not in IPHS_LEVEL_DEFINITIONS:
            issues.append(
                "iphs_level must be one of: "
                + ",".join(str(level) for level in sorted(IPHS_LEVEL_DEFINITIONS))
            )
        else:
            max_by_role = MAX_IPHS_LEVEL_BY_ROLE.get(self.canonical_role)
            if max_by_role is not None and self.iphs_level > max_by_role:
                issues.append(
                    f"canonical_role '{self.canonical_role}' cannot claim iphs_level > {max_by_role}"
                )
            max_by_equivalence = MAX_IPHS_LEVEL_BY_EQUIVALENCE_TYPE.get(
                self.equivalence_type
            )
            if max_by_equivalence is not None and self.iphs_level > max_by_equivalence:
                issues.append(
                    f"equivalence_type '{self.equivalence_type}' cannot claim iphs_level > {max_by_equivalence}"
                )
            missing_evidence = [
                key
                for key in _required_evidence_keys(self.iphs_level)
                if not _evidence_is_present(self.iphs_level_evidence.get(key))
            ]
            if missing_evidence:
                issues.append(
                    "iphs_level_evidence must define: "
                    + ",".join(missing_evidence)
                )
        if self.formulation not in set(get_args(IPHSFormulation)):
            issues.append(
                "formulation must be unspecified, dissipative_ph, or generic"
            )
        if self.iphs_level == 4 and self.formulation != "dissipative_ph":
            issues.append("Level 4 requires formulation='dissipative_ph'")
        if self.iphs_level == 5 and self.formulation != "generic":
            issues.append("Level 5 requires formulation='generic'")
        if not self.theory_document:
            issues.append("theory_document is required")
        elif not _source_tree_file_exists(self.theory_document):
            issues.append("theory_document must exist in the current source tree")
        if self.theory_document_status not in ALLOWED_THEORY_DOCUMENT_STATUSES:
            issues.append(
                "theory_document_status must be created, updated, or already_current"
            )
        if not self.legacy_expression:
            issues.append("legacy_formula or legacy_expression is required")
        if not self.literature_formula:
            issues.append("literature_formula is required")
        if not self.iphs_expression:
            issues.append("iphs_formula or iphs_expression is required")
        if not self.input_variables:
            issues.append("input_variables are required")
        if not self.output_variables:
            issues.append("output_variables are required")
        if not self.parameter_keys and not self.parameter_free:
            issues.append("parameter_keys are required")
        if self.parameter_free and self.parameter_keys:
            issues.append("parameter_free certificates cannot declare parameter_keys")
        if not self.references:
            issues.append("references are required")
        if not self.validation_tests:
            issues.append("baseline_tests or validation_tests are required")
        elif any(
            not _source_tree_file_exists(test_path)
            for test_path in self.validation_tests
        ):
            issues.append("validation_tests must exist in the current source tree")
        if self.equivalence_type not in ALLOWED_EQUIVALENCE_TYPES:
            issues.append(
                "equivalence_type must be one of: "
                + ",".join(sorted(ALLOWED_EQUIVALENCE_TYPES))
            )
        if not isinstance(self.constructor_selection, Mapping):
            issues.append(
                "constructor_selection must be a mapping of constructor keyword to JSON scalar"
            )
        else:
            issues.extend(
                _validate_constructor_selection(
                    self.constructor_selection,
                    process_id=self.process_id,
                    iphs_level=self.iphs_level,
                )
            )

        if self.iphs_level is not None and self.iphs_level >= 4:
            structured_evidence = all(
                isinstance(self.iphs_level_evidence.get(name), Mapping)
                for name in _strict_evidence_names(self.iphs_level)
            )
            if structured_evidence and not callable(parameter_resolver):
                from relic.parameters import get_global_manager

                parameter_resolver = get_global_manager().get_parameter
            evidence_resolver = (
                parameter_resolver if callable(parameter_resolver) else lambda key: None
            )
            runtime_evidence_issues.extend(
                _validate_strict_level_evidence(
                    level=self.iphs_level,
                    evidence=self.iphs_level_evidence,
                    validation_tests=self.validation_tests,
                    interface_variables=set(self.input_variables).union(self.output_variables),
                    process_id=self.process_id,
                    parameter_resolver=evidence_resolver,
                )
            )
            runtime_evidence_issues.extend(
                _validate_strict_runtime_replacement(
                    self.runtime_replacement_map,
                    self.validation_tests,
                    self.process_id,
                )
            )
            runtime_evidence_issues.extend(
                _validate_formulation_structure(
                    level=self.iphs_level,
                    iphs_expression=self.iphs_expression,
                    operator_map=self.operator_map,
                    state_advancement_map=self.state_advancement_map,
                )
            )
            if runtime_point_audit is None:
                runtime_evidence_issues.append(
                    "process_bound_runtime_unavailable [N]: callable paths, mapping "
                    "claims, and existing tests do not prove governed forward execution, "
                    "canonical IPHS replacement, or StateManager commit"
                )
            else:
                witness_issues = _validate_process_bound_runtime_audit(
                    level=self.iphs_level,
                    formulation=self.formulation,
                    output_variables=self.output_variables,
                    audit=runtime_point_audit,
                )
                if witness_issues:
                    runtime_evidence_issues.extend(witness_issues)
                else:
                    runtime_binding = "process_bound"
            if structured_evidence:
                for key in self.parameter_keys:
                    if "*" in key:
                        runtime_evidence_issues.append(
                            f"strict certificate parameter key '{key}' cannot be a wildcard"
                        )
                        continue
                    # None (missing key) is an issue; a resolver failure propagates.
                    parameter_spec = evidence_resolver(key)
                    if not has_complete_parameter_metadata(parameter_spec):
                        runtime_evidence_issues.append(
                            f"strict certificate parameter key '{key}' must resolve to "
                            "complete ParameterManager metadata"
                        )
            issues.extend(runtime_evidence_issues)

        runtime_uses_iphs = self.runtime_replacement_map.get("runtime_uses_iphs")
        legacy_removed = self.runtime_replacement_map.get("legacy_removed")
        if (
            self.iphs_level is not None
            and self.iphs_level < 4
            and isinstance(runtime_uses_iphs, bool)
            and isinstance(legacy_removed, bool)
            and (runtime_uses_iphs or legacy_removed)
        ):
            issues.append(
                "non-strict certificates cannot self-attest strict runtime "
                "replacement; declare runtime_uses_iphs=false and "
                "legacy_removed=false"
            )

        legacy_inputs = set(_tuple_of_text(self.interface_map.get("legacy_inputs")))
        iphs_inputs = set(_tuple_of_text(self.interface_map.get("iphs_inputs")))
        if legacy_inputs and iphs_inputs and legacy_inputs != iphs_inputs:
            issues.append("legacy_inputs and iphs_inputs must match")

        legacy_outputs = set(_tuple_of_text(self.interface_map.get("legacy_outputs")))
        iphs_outputs = set(_tuple_of_text(self.interface_map.get("iphs_outputs")))
        if legacy_outputs and iphs_outputs and legacy_outputs != iphs_outputs:
            issues.append("legacy_outputs and iphs_outputs must match")

        if strict:
            required_maps = {
                "state_map": self.state_map,
                "operator_map": self.operator_map,
                "state_advancement_map": self.state_advancement_map,
                "unit_map": self.unit_map,
                "input_output_map": self.input_output_map,
                "interface_map": self.interface_map,
                "dependency_interconnection_map": self.dependency_interconnection_map,
                "overflow_routing_map": self.overflow_routing_map,
                "conservation_map": self.conservation_map,
                "entropy_audit": self.entropy_audit,
            }
            if not self.parameter_free:
                required_maps["parameter_map"] = self.parameter_map
            for name, mapping in required_maps.items():
                if not mapping:
                    issues.append(f"{name} is required")
                elif not _evidence_is_present(mapping):
                    issues.append(f"{name} contains empty evidence")

            if (
                self.iphs_level is not None
                and self.iphs_level < 4
                and (
                    not isinstance(runtime_uses_iphs, bool)
                    or not isinstance(legacy_removed, bool)
                )
            ):
                issues.append(
                    "runtime_replacement_map must define boolean "
                    "runtime_uses_iphs and legacy_removed"
                )

            required_state_advancement_keys = {
                "potential_basis",
                "effort_variable",
                "state_advancer",
                "runtime_usage",
            }
            missing_state_advancement = sorted(
                key
                for key in required_state_advancement_keys
                if not _text(self.state_advancement_map.get(key))
            )
            if missing_state_advancement:
                issues.append(
                    "state_advancement_map must define: "
                    + ",".join(missing_state_advancement)
                )

            if self.canonical_role == "reaction":
                if not _mapping_text_contains(self.operator_map, "N_h", "r_h"):
                    issues.append(
                        "reaction operator_map must declare explicit N_h and r_h usage"
                    )
                state_advancer = _text(self.state_advancement_map.get("state_advancer"))
                if not _mapping_text_contains({"state_advancer": state_advancer}, "N_h", "r_h"):
                    issues.append(
                        "reaction state_advancement_map.state_advancer must use explicit N_h r_h"
                    )

            parameter_sources = self.parameter_management_map
            if not self.parameter_free and not parameter_sources:
                issues.append("parameter_management_map is required")
            elif not self.parameter_free:
                missing_parameters = [
                    key for key in self.parameter_keys if key not in parameter_sources
                ]
                if missing_parameters:
                    issues.append(
                        "parameter_management_map must cover all parameter_keys: "
                        + ",".join(missing_parameters)
                    )
                for key in self.parameter_keys:
                    entry = parameter_sources.get(key)
                    if isinstance(entry, Mapping):
                        source = str(entry.get("source", entry.get("manager", ""))).lower()
                    else:
                        source = str(entry).lower()
                    if not source or "parameter" not in source:
                        issues.append(f"parameter '{key}' must be managed by ParameterManager")

            composition_ports_valid = bool(self.composition_ports) and all(
                isinstance(port, str)
                and bool(port)
                and port == port.strip()
                for port in self.composition_ports
            )
            composition_map_valid = _evidence_is_present(
                self.composition_management_map
            )
            composition_ok = composition_ports_valid or composition_map_valid
            if not composition_ok:
                issues.append(
                    "material composition evidence requires non-empty exact ports "
                    "or a recursively non-empty composition_management_map"
                )

            stochastic_declared = bool(
                self.stochastic_sdae_map
                or self.noise_covariance_map
                or self.conservative_noise_flux_map
                or self.descriptor_constraint_map
                or self.jump_reset_map
                or self.stochastic_reproducibility_map
            )
            if stochastic_declared:
                if not self.stochastic_sdae_map:
                    issues.append("stochastic_sdae_map is required when stochastic/SDAE maps are declared")
                declared_requirements = {
                    "noise_covariance_map": bool(
                        self.stochastic_sdae_map.get("requires_noise_covariance", False)
                        or self.stochastic_sdae_map.get("declares_fdt_noise", False)
                    ),
                    "conservative_noise_flux_map": bool(
                        self.stochastic_sdae_map.get("requires_conservative_noise_flux", False)
                    ),
                    "descriptor_constraint_map": bool(
                        self.stochastic_sdae_map.get("requires_descriptor_constraint", False)
                    ),
                    "jump_reset_map": bool(
                        self.stochastic_sdae_map.get("requires_jump_reset", False)
                    ),
                    "stochastic_reproducibility_map": bool(
                        self.stochastic_sdae_map.get("declares_stochastic_terms", False)
                        or self.stochastic_sdae_map.get("requires_jump_reset", False)
                    ),
                }
                map_values = {
                    "noise_covariance_map": self.noise_covariance_map,
                    "conservative_noise_flux_map": self.conservative_noise_flux_map,
                    "descriptor_constraint_map": self.descriptor_constraint_map,
                    "jump_reset_map": self.jump_reset_map,
                    "stochastic_reproducibility_map": self.stochastic_reproducibility_map,
                }
                for name, required in declared_requirements.items():
                    if required and not map_values[name]:
                        issues.append(f"{name} is required by stochastic_sdae_map")
                owner = _text(
                    self.stochastic_sdae_map.get(
                        "runtime_owner",
                        self.stochastic_sdae_map.get("stochastic_runtime_owner", ""),
                    )
                ).replace("/", "\\").lower()
                if owner and not (
                    owner.startswith("relic\\stochastic")
                    or owner.startswith("relic.stochastic")
                    or owner == "relicsimulator.run_ensemble"
                ):
                    issues.append("stochastic_sdae_map runtime_owner must be an existing RELIC stochastic owner")

        if self.iphs_level is not None and self.iphs_level >= 4:
            runtime_evidence_status = (
                "verified" if not runtime_evidence_issues else "invalid"
            )
            runtime_verified_level = (
                self.iphs_level if not runtime_evidence_issues else 0
            )
        else:
            runtime_evidence_status = "descriptive_only"
            runtime_verified_level = 0

        if (
            accepted_required_level is not None
            and runtime_verified_level < accepted_required_level
        ):
            issues.append(
                "required runtime-verified IPHS level "
                f"{accepted_required_level} was not met; verified level is "
                f"{runtime_verified_level}"
            )

        return {
            "passed": not issues,
            "module_name": self.module_name,
            "process_id": self.process_id,
            "canonical_role": self.canonical_role,
            "iphs_level": self.iphs_level,
            "iphs_level_name": _iphs_level_name(self.iphs_level),
            "formulation": self.formulation,
            "runtime_verified_level": runtime_verified_level,
            "runtime_evidence_status": runtime_evidence_status,
            "runtime_binding": runtime_binding,
            "constructor_selection": (
                dict(self.constructor_selection)
                if isinstance(self.constructor_selection, Mapping)
                else self.constructor_selection
            ),
            "runtime_evidence_issue_count": len(runtime_evidence_issues),
            "iphs_level_evidence_count": len(self.iphs_level_evidence),
            "theory_document_status": self.theory_document_status,
            "issue_count": len(issues),
            "issues": issues,
            "input_count": len(self.input_variables),
            "output_count": len(self.output_variables),
            "parameter_count": len(self.parameter_keys),
            "parameter_free": self.parameter_free,
            "composition_port_count": len(self.composition_ports),
            "reference_count": len(self.references),
            "validation_test_count": len(self.validation_tests),
            "strict_map_count": sum(
                1
                for mapping in (
                    self.state_map,
                    self.parameter_map,
                    self.operator_map,
                    self.state_advancement_map,
                    self.unit_map,
                    self.input_output_map,
                    self.interface_map,
                    self.dependency_interconnection_map,
                    self.overflow_routing_map,
                    self.conservation_map,
                    self.entropy_audit,
                    self.stochastic_sdae_map,
                    self.noise_covariance_map,
                    self.conservative_noise_flux_map,
                    self.descriptor_constraint_map,
                    self.jump_reset_map,
                    self.stochastic_reproducibility_map,
                    self.iphs_level_evidence,
                )
                if mapping
            ),
        }


def load_iphs_equivalence_certificate(source: Mapping[str, Any] | str | Path) -> IPHSEquivalenceCertificate:
    """Load an IPHS equivalence certificate from a mapping, JSON, or YAML file."""

    if isinstance(source, Mapping):
        payload = dict(source)
    else:
        path = Path(source)
        text = path.read_text(encoding="utf-8-sig")
        if path.suffix.lower() == ".json":
            payload = json.loads(text)
        else:
            payload = yaml.safe_load(text) or {}
    if not isinstance(payload, Mapping):
        raise TypeError("IPHS equivalence certificate must load to a mapping.")
    return IPHSEquivalenceCertificate.from_mapping(payload)


__all__ = [
    "IPHS_LEVEL_DEFINITIONS",
    "IPHSEquivalenceCertificate",
    "load_iphs_equivalence_certificate",
]
