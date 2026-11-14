"""
Graph validation and connection checks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from relic.computation.operators.iphs_certificate import (
    ALLOWED_CANONICAL_ROLES,
    resolve_iphs_level_requirement,
)
from relic.core.simulator.dependency_ops import (
    _VALID_EXECUTION_GROUPS,
    _invalid_execution_group_message,
    _is_valid_execution_group,
)
from relic.core.simulator.output_overlap_governance import (
    OVERLAP_POLICY_MUTUALLY_EXCLUSIVE,
    get_output_overlap_policy,
)
from relic.data.standards.dimension_check import check_port_dimension_consistency
from relic.data.standards.dimensionality import dimensionality_from_unit
from relic.data.standards.unit_converter import (
    ConversionContext,
    UnitConverter,
    unit_converter as canonical_unit_converter,
)

from .graph_ir import GraphIR
from .node_registry import NodeRegistry
from .port_spec import PORT_SUPPORT_FIELDS, PortSpec
from .ssot_catalog import SSOTCatalog


@dataclass
class GraphValidationIssue:
    code: str
    message: str
    node_id: Optional[str] = None
    port: Optional[str] = None
    expected: Optional[str] = None
    got: Optional[str] = None
    hint: Optional[str] = None


ValidationIssue = GraphValidationIssue


@dataclass
class ConnectionDecision:
    ok: bool
    errors: List[GraphValidationIssue] = field(default_factory=list)
    warnings: List[GraphValidationIssue] = field(default_factory=list)
    requires_adapter: bool = False
    adapter_type: Optional[str] = None


@dataclass
class ValidationReport:
    errors: List[GraphValidationIssue] = field(default_factory=list)
    warnings: List[GraphValidationIssue] = field(default_factory=list)
    assertion_audit: Dict[str, Any] = field(default_factory=dict)


def _get_context_value(context: Optional[ConversionContext], key: str) -> Optional[Any]:
    if context is None:
        return None
    if isinstance(context, dict):
        return context.get(key)
    return getattr(context, key, None)


def _normalize_text(value: Optional[str]) -> Optional[str]:
    if not value or not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized.lower() if normalized else None


def _normalize_required_local_standard_name(
    value: Any,
    *,
    port_label: str,
) -> str:
    """Validate local port identity before case-insensitive comparison."""

    if not isinstance(value, str):
        raise TypeError(
            f"{port_label} standard_name must be an exact non-empty string"
        )
    if not value or value != value.strip():
        raise ValueError(
            f"{port_label} standard_name must be an exact non-empty string"
        )
    return value.lower()


def _add_error(decision: ConnectionDecision, issue: GraphValidationIssue) -> None:
    decision.ok = False
    decision.errors.append(issue)


def _add_strict_error_or_warning(
    decision: ConnectionDecision,
    issue: GraphValidationIssue,
    *,
    strict: bool,
) -> bool:
    if strict:
        _add_error(decision, issue)
        return True
    decision.warnings.append(issue)
    return False


def _validate_optional_tag_match(
    decision: ConnectionDecision,
    *,
    source_value: Optional[str],
    target_value: Optional[str],
    code: str,
    message: str,
    strict: bool,
) -> bool:
    if source_value and target_value and source_value != target_value:
        issue = GraphValidationIssue(
            code=code,
            message=message,
            expected=target_value,
            got=source_value,
        )
        if _add_strict_error_or_warning(decision, issue, strict=strict):
            return False
    return True


def _validate_datatype_match(
    decision: ConnectionDecision,
    *,
    source: PortSpec,
    target: PortSpec,
) -> bool:
    if source.datatype and target.datatype and source.datatype != target.datatype:
        _add_error(
            decision,
            GraphValidationIssue(
                code="datatype_mismatch",
                message="Datatype mismatch between ports.",
                expected=target.datatype,
                got=source.datatype,
            ),
        )
        return False
    return True


def _validate_units_compatibility(
    decision: ConnectionDecision,
    *,
    source: PortSpec,
    target: PortSpec,
    unit_converter: UnitConverter,
    context: Optional[ConversionContext],
    strict: bool,
) -> bool:
    if not source.units or not target.units:
        issue = GraphValidationIssue(
            code="missing_units",
            message="Units missing for connection.",
            expected=target.units or "defined units",
            got=source.units or "undefined units",
            hint="Define units in SSOT catalogs for both ports.",
        )
        if _add_strict_error_or_warning(decision, issue, strict=strict):
            return False

    if source.units and target.units:
        unsupported_units = [
            unit
            for unit in (source.units, target.units)
            if not unit_converter.can_convert(unit, unit, "generic")
        ]
        if unsupported_units:
            issue = GraphValidationIssue(
                code="unsupported_units",
                message="Port connection declares unsupported unit labels.",
                expected="unit labels recognized by the canonical dimensionality owner",
                got=", ".join(sorted(set(unsupported_units))),
                hint="Use a supported canonical SI-token unit expression.",
            )
            if _add_strict_error_or_warning(decision, issue, strict=strict):
                return False

    if source.units and target.units and source.units != target.units:
        if unit_converter.can_convert(source.units, target.units, source.standard_name):
            decision.requires_adapter = True
            decision.adapter_type = "unit_cast"
        elif unit_converter.can_convert(source.units, target.units, "generic"):
            decision.requires_adapter = True
            decision.adapter_type = "unit_cast"
        else:
            _add_error(
                decision,
                GraphValidationIssue(
                    code="unit_inconsistent",
                    message="Units are not convertible.",
                    expected=target.units,
                    got=source.units,
                    hint="Add conversion rule or adapter in UnitConverter.",
                ),
            )
            return False

        if decision.requires_adapter:
            # Trial conversion: UnitConverter.convert raises ValueError for an
            # unsupported pair or insufficient context, which becomes a
            # missing_context connection error. Infrastructure failures
            # (parameter or rule loading) are not a context problem and
            # propagate.
            try:
                unit_converter.convert(
                    1.0,
                    variable_name=source.standard_name or "generic",
                    source_units=source.units,
                    target_units=target.units,
                    context=context,
                )
            except ValueError as exc:
                _add_error(
                    decision,
                    GraphValidationIssue(
                        code="missing_context",
                        message=f"Conversion requires context: {exc}",
                        expected="conversion context",
                        got="missing context",
                        hint="Provide required context in graph or adapter.",
                    ),
                )
                return False

    return True


def _ensure_semantic_adapter(decision: ConnectionDecision) -> None:
    if decision.requires_adapter and decision.adapter_type != "semantic_adapter":
        decision.warnings.append(
            GraphValidationIssue(
                code="multiple_adapters_required",
                message="Connection requires unit and semantic adapters.",
                expected="unit_cast + semantic_adapter",
                got=decision.adapter_type,
            )
        )
    decision.requires_adapter = True
    decision.adapter_type = "semantic_adapter"


def _extract_semantic_fields(source: PortSpec, target: PortSpec) -> tuple[str, str, str, str]:
    source_sem = (source.semantics or "").strip().lower()
    target_sem = (target.semantics or "").strip().lower()
    target_mode = (target.update_mode or "").strip().lower()
    target_write_mode = (target.write_mode or "").strip().lower()
    if not target_write_mode and target_mode in {"immutable", "derived"}:
        target_write_mode = target_mode
    return source_sem, target_sem, target_mode, target_write_mode


def _validate_semantics(
    decision: ConnectionDecision,
    *,
    source_sem: str,
    target_sem: str,
    target_mode: str,
    target_write_mode: str,
) -> bool:
    if source_sem and target_sem:
        if target_sem in {"state", "diagnostic"}:
            if source_sem == "flux":
                _ensure_semantic_adapter(decision)
            elif source_sem not in {"state", "diagnostic"}:
                _add_error(
                    decision,
                    GraphValidationIssue(
                        code="semantics_mismatch",
                        message="State/diagnostic input cannot accept this semantics.",
                        expected="state or diagnostic",
                        got=source_sem,
                    ),
                )
                return False
        elif target_sem == "increment":
            if source_sem == "flux":
                _ensure_semantic_adapter(decision)
            elif source_sem != "increment":
                _add_error(
                    decision,
                    GraphValidationIssue(
                        code="semantics_mismatch",
                        message="Increment input cannot accept this semantics.",
                        expected="increment",
                        got=source_sem,
                    ),
                )
                return False
        elif target_sem == "flux":
            if source_sem != "flux":
                _add_error(
                    decision,
                    GraphValidationIssue(
                        code="semantics_mismatch",
                        message="Flux input must connect to flux output.",
                        expected="flux",
                        got=source_sem,
                    ),
                )
                return False
        elif target_sem == "parameter":
            if source_sem != "parameter":
                _add_error(
                    decision,
                    GraphValidationIssue(
                        code="semantics_mismatch",
                        message="Parameter input must connect to parameter output.",
                        expected="parameter",
                        got=source_sem,
                    ),
                )
                return False

    if target_mode == "replace" and source_sem == "flux":
        _ensure_semantic_adapter(decision)

    if target_write_mode == "immutable" and source_sem and source_sem != "parameter":
        _add_error(
            decision,
            GraphValidationIssue(
                code="immutable_target",
                message="Immutable target cannot accept non-parameter semantics.",
                expected="parameter",
                got=source_sem,
            ),
        )
        return False

    if target_write_mode == "derived":
        decision.warnings.append(
            GraphValidationIssue(
                code="derived_target",
                message="Derived target is diagnostic and should not accept updates.",
                expected="diagnostic-only",
                got=source_sem or "unspecified",
            )
        )

    return True


def _validate_context_requirements(
    decision: ConnectionDecision,
    *,
    target: PortSpec,
    context: Optional[ConversionContext],
    strict: bool,
) -> bool:
    if target.context_requirements:
        missing = [key for key in target.context_requirements if _get_context_value(context, key) is None]
        if missing:
            issue = GraphValidationIssue(
                code="missing_context_requirements",
                message="Target port requires context values that are missing.",
                expected=", ".join(missing),
                got="missing",
                hint="Provide required context in ConversionContext or graph config.",
            )
            if _add_strict_error_or_warning(decision, issue, strict=strict):
                return False
    return True


_REQUIRED_STRUCTURAL_PORT_FIELDS = (
    "form_degree",
    "primality",
    "extensivity",
    "complex_id",
)


def _has_structural_port_metadata(port: PortSpec) -> bool:
    return any(getattr(port, field_name, None) is not None for field_name in _REQUIRED_STRUCTURAL_PORT_FIELDS)


def _validate_structural_port_contract(
    decision: ConnectionDecision,
    *,
    source: PortSpec,
    target: PortSpec,
    strict: bool,
) -> bool:
    source_structural = _has_structural_port_metadata(source)
    target_structural = _has_structural_port_metadata(target)
    if not source_structural and not target_structural:
        return True

    missing_source = [
        field_name
        for field_name in _REQUIRED_STRUCTURAL_PORT_FIELDS
        if getattr(source, field_name, None) is None
    ]
    missing_target = [
        field_name
        for field_name in _REQUIRED_STRUCTURAL_PORT_FIELDS
        if getattr(target, field_name, None) is None
    ]
    if missing_source or missing_target:
        issue = GraphValidationIssue(
            code="missing_structural_port_metadata",
            message="Structural ports require complete DEC form metadata.",
            expected=", ".join(_REQUIRED_STRUCTURAL_PORT_FIELDS),
            got=(
                f"source missing {missing_source or 'none'}; "
                f"target missing {missing_target or 'none'}"
            ),
            hint="Declare form metadata or use a validated chain-map adapter between complete spaces.",
        )
        return not _add_strict_error_or_warning(decision, issue, strict=strict)

    mismatched_fields = [
        field_name
        for field_name in _REQUIRED_STRUCTURAL_PORT_FIELDS
        if getattr(source, field_name) != getattr(target, field_name)
    ]
    optional_mismatches = [
        field_name
        for field_name in ("orientation_convention", "physical_domain")
        if getattr(source, field_name, None)
        and getattr(target, field_name, None)
        and _normalize_text(getattr(source, field_name))
        != _normalize_text(getattr(target, field_name))
    ]
    mismatched_fields.extend(optional_mismatches)
    if not mismatched_fields:
        return True

    for field_name in mismatched_fields:
        issue = GraphValidationIssue(
            code=f"{field_name}_mismatch",
            message=f"Structural port {field_name} mismatch requires an explicit chain-map adapter node.",
            expected=str(getattr(target, field_name, None)),
            got=str(getattr(source, field_name, None)),
            hint=(
                "Connect through an adapter whose output declares the target PortSpec, "
                "then validate that adapter with the existing chain_map_check assertion."
            ),
        )
        _add_strict_error_or_warning(decision, issue, strict=strict)
    return decision.ok


def _validate_port_support_contract(
    decision: ConnectionDecision,
    *,
    source: PortSpec,
    target: PortSpec,
    strict: bool,
) -> bool:
    one_sided_fields = [
        field_name
        for field_name in PORT_SUPPORT_FIELDS
        if source.is_support_declared(field_name)
        != target.is_support_declared(field_name)
    ]
    if one_sided_fields:
        issue = GraphValidationIssue(
            code="missing_port_support_metadata",
            message=(
                "Port support metadata is declared on only one side of the "
                "connection."
            ),
            expected=str(
                {
                    field_name: getattr(target, field_name, None)
                    for field_name in one_sided_fields
                }
            ),
            got=str(
                {
                    field_name: getattr(source, field_name, None)
                    for field_name in one_sided_fields
                }
            ),
            hint=(
                "Declare matching support metadata or insert an explicit "
                "regrid, aggregation, temporal-resampling, vertical-remap, "
                "or area-normalization adapter node."
            ),
        )
        _add_strict_error_or_warning(decision, issue, strict=strict)

    mismatched_fields = [
        field_name
        for field_name in PORT_SUPPORT_FIELDS
        if source.is_support_declared(field_name)
        and target.is_support_declared(field_name)
        and getattr(source, field_name) != getattr(target, field_name)
    ]
    for field_name in mismatched_fields:
        issue = GraphValidationIssue(
            code=f"{field_name}_mismatch",
            message=(
                f"Port {field_name} mismatch requires an explicit support "
                "transformation adapter node."
            ),
            expected=str(getattr(target, field_name)),
            got=str(getattr(source, field_name)),
            hint=(
                "Insert a declared conservative regrid, aggregation, "
                "temporal-resampling, vertical-remap, or area-normalization "
                "adapter and validate its output PortSpec."
            ),
        )
        _add_strict_error_or_warning(decision, issue, strict=strict)
    return decision.ok


def can_connect(
    source: PortSpec,
    target: PortSpec,
    unit_converter: UnitConverter,
    context: Optional[ConversionContext] = None,
    strict: bool = True,
) -> ConnectionDecision:
    decision = ConnectionDecision(ok=True)

    if not _validate_datatype_match(decision, source=source, target=target):
        return decision

    source_quantity = _normalize_text(source.quantity)
    target_quantity = _normalize_text(target.quantity)
    if not _validate_optional_tag_match(
        decision,
        source_value=source_quantity,
        target_value=target_quantity,
        code="quantity_mismatch",
        message="Quantity mismatch between ports.",
        strict=strict,
    ):
        return decision

    source_element = _normalize_text(source.element)
    target_element = _normalize_text(target.element)
    if not _validate_optional_tag_match(
        decision,
        source_value=source_element,
        target_value=target_element,
        code="element_mismatch",
        message="Element tag mismatch between ports.",
        strict=strict,
    ):
        return decision

    source_local = _normalize_required_local_standard_name(
        source.standard_name,
        port_label="Source port",
    )
    target_local = _normalize_required_local_standard_name(
        target.standard_name,
        port_label="Target port",
    )
    source_cf = SSOTCatalog._normalize_optional_cf_standard_name(
        source.cf_standard_name,
        source="Source port",
    )
    target_cf = SSOTCatalog._normalize_optional_cf_standard_name(
        target.cf_standard_name,
        source="Target port",
    )

    if source_cf is not None and target_cf is not None:
        if not _validate_optional_tag_match(
            decision,
            source_value=source_cf,
            target_value=target_cf,
            code="cf_standard_name_mismatch",
            message="CF standard name mismatch between ports.",
            strict=strict,
        ):
            return decision
    elif source_local != target_local:
        issue = GraphValidationIssue(
            code="standard_name_mismatch",
            message="RELIC local standard-name identity mismatch between ports.",
            expected=target.standard_name,
            got=source.standard_name,
        )
        if _add_strict_error_or_warning(decision, issue, strict=strict):
            return decision

    if not _validate_units_compatibility(
        decision,
        source=source,
        target=target,
        unit_converter=unit_converter,
        context=context,
        strict=strict,
    ):
        return decision

    source_sem, target_sem, target_mode, target_write_mode = _extract_semantic_fields(source, target)
    if not _validate_semantics(
        decision,
        source_sem=source_sem,
        target_sem=target_sem,
        target_mode=target_mode,
        target_write_mode=target_write_mode,
    ):
        return decision

    if not _validate_context_requirements(
        decision,
        target=target,
        context=context,
        strict=strict,
    ):
        return decision

    if not _validate_structural_port_contract(
        decision,
        source=source,
        target=target,
        strict=strict,
    ):
        return decision

    if not _validate_port_support_contract(
        decision,
        source=source,
        target=target,
        strict=strict,
    ):
        return decision

    return decision


def _default_assertion_audit() -> Dict[str, Any]:
    return {
        "present": False,
        "valid": True,
        "phi": None,
        "op": None,
        "scope": None,
        "op_resolved": False,
        "issues": [],
    }


def _validate_assertion_fields(
    *,
    phi: Any,
    op: Any,
    scope: Any,
    context_payload: Any,
) -> List[GraphValidationIssue]:
    issues: List[GraphValidationIssue] = []
    if not isinstance(phi, str) or not phi.strip():
        issues.append(
            GraphValidationIssue(
                code="assertion_missing_phi",
                message="Graph assertion requires non-empty phi.",
                hint="Set assertion.phi to a well-formed predicate string.",
            )
        )
    if not isinstance(op, str) or not op.strip():
        issues.append(
            GraphValidationIssue(
                code="assertion_missing_op",
                message="Graph assertion requires non-empty op.",
                hint="Set assertion.op to an executable protocol or node identifier.",
            )
        )
    if scope is None:
        issues.append(
            GraphValidationIssue(
                code="assertion_missing_scope",
                message="Graph assertion requires scope.",
                hint="Set assertion.scope to a non-empty string or object.",
            )
        )
    elif isinstance(scope, str):
        if not scope.strip():
            issues.append(
                GraphValidationIssue(
                    code="assertion_invalid_scope",
                    message="Graph assertion scope string must be non-empty.",
                )
            )
    elif isinstance(scope, dict):
        if not scope:
            issues.append(
                GraphValidationIssue(
                    code="assertion_invalid_scope",
                    message="Graph assertion scope object must be non-empty.",
                )
            )
    else:
        issues.append(
            GraphValidationIssue(
                code="assertion_invalid_scope",
                message="Graph assertion scope must be string or object.",
            )
        )

    if context_payload is not None and not isinstance(context_payload, dict):
        issues.append(
            GraphValidationIssue(
                code="assertion_invalid_context",
                message="Graph assertion context must be a mapping when provided.",
            )
        )
    return issues


def _is_registry_protocol_match(
    graph: GraphIR,
    registry: NodeRegistry,
    normalized_op: str,
) -> bool:
    for node_spec in graph.nodes.values():
        definition = registry.get_definition(node_spec.class_type)
        if definition is None or not isinstance(definition.metadata, dict):
            continue
        protocols = definition.metadata.get("op_protocols") or definition.metadata.get("supported_ops")
        if not isinstance(protocols, list):
            continue
        if normalized_op in {str(item).strip() for item in protocols}:
            return True
    return False


_STRUCTURAL_ASSERTION_OPS = {
    "cochain_complex_check",
    "hodge_psd_check",
    "iphs_structure_check",
    "chain_map_check",
    "entropy_step_check",
    "port_power_units_check",
    "port_matching_check",
    "overflow_routing_check",
    "equivalence_certificate_check",
    "reaction_network_check",
    "stochastic_sdae_check",
    "state_domain_check",
    "observation_diagnostic_check",
}

_STRUCTURAL_ASSERTION_AUDIT_KEYS = {
    "cochain_complex_check": "dec_topology_audit",
    "hodge_psd_check": "hodge_audit",
    "iphs_structure_check": "iphs_audit",
    "chain_map_check": "naturality_audit",
    "entropy_step_check": "entropy_audit",
    "port_power_units_check": "port_power_audit",
    "port_matching_check": "port_matching_audit",
    "overflow_routing_check": "overflow_routing_audit",
    "equivalence_certificate_check": "equivalence_certificate_audit",
    "reaction_network_check": "reaction_network_audit",
    "stochastic_sdae_check": "stochastic_sdae_audit",
    "state_domain_check": "state_domain_audit",
    "observation_diagnostic_check": "observation_audit",
}

_STRICT_DYNAMIC_ASSERTION_OPS = {
    "cochain_complex_check",
    "hodge_psd_check",
    "iphs_structure_check",
    "entropy_step_check",
    "port_power_units_check",
    "port_matching_check",
    "overflow_routing_check",
    "equivalence_certificate_check",
    "state_domain_check",
}


def _is_strict_structural_graph(graph_metadata: Dict[str, Any]) -> bool:
    return bool(
        graph_metadata.get("strict_structural_assertions")
        or _is_strict_iphs_graph(graph_metadata)
    )


def _is_strict_iphs_graph(graph_metadata: Dict[str, Any]) -> bool:
    """Identify an advanced claim without promoting ordinary structural checks."""
    required_level, _ = _required_iphs_level(
        {}, strict=False, graph_metadata=graph_metadata,
    )
    return bool(
        graph_metadata.get("phase8_strict")
        or required_level is not None and required_level >= 4
    )


def _normalized_canonical_role(graph_metadata: Dict[str, Any]) -> str:
    value = graph_metadata.get("canonical_role")
    return value.strip().lower() if isinstance(value, str) else ""


def _strict_canonical_role_issues(
    graph_metadata: Dict[str, Any],
) -> list[GraphValidationIssue]:
    if not _is_strict_structural_graph(graph_metadata):
        return []
    role = _normalized_canonical_role(graph_metadata)
    if not role:
        if not _is_strict_iphs_graph(graph_metadata) and "canonical_role" not in graph_metadata:
            return []
        return [
            GraphValidationIssue(
                code="strict_canonical_role_required",
                message="Strict structural graphs must declare canonical_role explicitly.",
                expected=", ".join(sorted(ALLOWED_CANONICAL_ROLES)),
                hint=(
                    "Use the certificate-owned scientific role taxonomy; declare "
                    "stochastic and hybrid semantics through extension metadata."
                ),
            )
        ]
    if role not in ALLOWED_CANONICAL_ROLES:
        return [
            GraphValidationIssue(
                code="strict_canonical_role_invalid",
                message="Strict graph canonical_role is not certificate-owned.",
                expected=", ".join(sorted(ALLOWED_CANONICAL_ROLES)),
                got=role,
                hint=(
                    "Keep transport/reaction/field/scale_coupling/port/observation "
                    "as the scientific role and declare stochastic or hybrid semantics "
                    "orthogonally."
                ),
            )
        ]
    return []


def _is_structural_assertion_op(normalized_op: str) -> bool:
    return normalized_op in _STRUCTURAL_ASSERTION_OPS


def _required_structural_assertion_ops(graph_metadata: Dict[str, Any]) -> set[str]:
    required: set[str] = set()
    explicit = graph_metadata.get("required_structural_assertions", ())
    if isinstance(explicit, str):
        explicit = (explicit,)
    if isinstance(explicit, (list, tuple, set)):
        required.update(str(item).strip() for item in explicit if str(item).strip())

    required_level, _ = _required_iphs_level(
        {}, strict=False, graph_metadata=graph_metadata,
    )
    if required_level is not None:
        required.add("equivalence_certificate_check")
    if _required_stochastic_semantics(graph_metadata):
        required.add("stochastic_sdae_check")
    if graph_metadata.get("required_observation_kinds"):
        required.add("observation_diagnostic_check")

    strict = _is_strict_iphs_graph(graph_metadata)
    if not strict:
        return required

    role = _normalized_canonical_role(graph_metadata)
    if role == "observation":
        required.update(
            {
                "equivalence_certificate_check",
                "state_domain_check",
                "observation_diagnostic_check",
            }
        )
        return required

    required.update(_STRICT_DYNAMIC_ASSERTION_OPS)
    if role == "reaction":
        required.add("reaction_network_check")
    if role == "scale_coupling":
        required.add("chain_map_check")
    explicit_semantics = graph_metadata.get("required_stochastic_semantics", ())
    if isinstance(explicit_semantics, str):
        explicit_semantics = (explicit_semantics,)
    has_explicit_semantics = isinstance(explicit_semantics, (list, tuple, set)) and any(
        str(item).strip() for item in explicit_semantics
    )
    if has_explicit_semantics or any(
        bool(graph_metadata.get(key))
        for key in (
            "stochastic_extension",
            "jump_extension",
            "descriptor_extension",
            "ale_extension",
            "memory_extension",
            "topology_change_extension",
        )
    ):
        required.add("stochastic_sdae_check")
    return required


def _required_stochastic_semantics(graph_metadata: Dict[str, Any]) -> set[str]:
    required: set[str] = set()
    explicit = graph_metadata.get("required_stochastic_semantics", ())
    if isinstance(explicit, str):
        explicit = (explicit,)
    if isinstance(explicit, (list, tuple, set)):
        required.update(str(item).strip() for item in explicit if str(item).strip())

    extension_markers = {
        "stochastic_extension": "continuous_sde",
        "jump_extension": "marked_jump",
        "memory_extension": "memory",
        "ale_extension": "ale",
        "descriptor_extension": "descriptor_dae",
        "topology_change_extension": "topology_change",
    }
    for key, semantic in extension_markers.items():
        if bool(graph_metadata.get(key)):
            required.add(semantic)
    return required


def _required_observation_kinds(graph_metadata: Dict[str, Any]) -> set[str]:
    required: set[str] = set()
    explicit = graph_metadata.get("required_observation_kinds", ())
    if isinstance(explicit, str):
        explicit = (explicit,)
    if isinstance(explicit, (list, tuple, set)):
        required.update(str(item).strip() for item in explicit if str(item).strip())
    role = _normalized_canonical_role(graph_metadata)
    if role == "observation":
        required.add("observation")
    return required


def _observation_spec_node_bindings(
    graph: GraphIR,
    scope: Any,
    context_payload: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Bind strict observation declarations to exact GraphIR node ports."""

    def _specs(value: Any) -> tuple[Any, ...]:
        if value is None:
            return ()
        if isinstance(value, (list, tuple)):
            return tuple(value)
        return (value,)

    specs = (
        *_specs(context_payload.get("observation_specs")),
        *_specs(context_payload.get("diagnostic_specs")),
    )
    spec_names: list[str] = []
    for spec in specs:
        name = spec.get("name") if isinstance(spec, dict) else getattr(spec, "name", None)
        if not isinstance(name, str) or not name.strip():
            return {}
        spec_names.append(name)

    if isinstance(scope, str):
        node_ids = {name: scope for name in spec_names}
    elif isinstance(scope, dict) and set(scope) == {"node_id"}:
        node_ids = {name: scope.get("node_id") for name in spec_names}
    elif isinstance(scope, dict) and set(scope) == {"spec_nodes"}:
        declared = scope.get("spec_nodes")
        if not isinstance(declared, dict) or set(declared) != set(spec_names):
            return {}
        node_ids = {name: declared.get(name) for name in spec_names}
    else:
        return {}

    bindings: Dict[str, Dict[str, Any]] = {}
    for spec_name, node_id in node_ids.items():
        node = graph.nodes.get(node_id) if isinstance(node_id, str) else None
        bindings[spec_name] = {
            "node_id": node_id,
            "input_ports": tuple(node.inputs) if node is not None else (),
            "output_ports": tuple(node.outputs or ()) if node is not None else (),
        }
    return bindings


def _missing_required_assertion_issues(
    graph_metadata: Dict[str, Any],
    observed_ops: set[str],
) -> list[GraphValidationIssue]:
    required_ops = _required_structural_assertion_ops(graph_metadata)
    issues: list[GraphValidationIssue] = []
    for missing_op in sorted(required_ops.difference(observed_ops)):
        issues.append(
            GraphValidationIssue(
                code="missing_required_structural_assertion",
                message=f"Required structural assertion '{missing_op}' is missing.",
                expected=missing_op,
                hint="Declare and execute the existing structural assertion op.",
            )
        )
    return issues


def _resolve_assertion_op(
    graph: GraphIR,
    registry: NodeRegistry,
    ssot_catalog: Optional[SSOTCatalog],
    normalized_op: str,
) -> bool:
    if not normalized_op:
        return False
    node_match = normalized_op in graph.nodes
    class_match = any(node.class_type == normalized_op for node in graph.nodes.values())
    ssot_match = ssot_catalog is not None and ssot_catalog.get(normalized_op) is not None
    registry_match = _is_registry_protocol_match(graph, registry, normalized_op)
    structural_match = _is_structural_assertion_op(normalized_op)
    return node_match or class_match or ssot_match or registry_match or structural_match


def _validation_contract_identities(result: Any) -> list[dict[str, Any]]:
    identities: dict[str, dict[str, Any]] = {}
    for detail in list(getattr(result, "details", ()) or ()):
        metadata = getattr(detail, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            continue
        identity = metadata.get("contract_identity")
        if not isinstance(identity, dict) or not identity:
            continue
        canonical_text = json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        identities[canonical_text] = json.loads(canonical_text)
    return [identities[key] for key in sorted(identities)]


def _serialize_validation_result(result: Any) -> Dict[str, Any]:
    return {
        "passed": bool(getattr(result, "passed", False)),
        "level": str(getattr(getattr(result, "level", None), "value", getattr(result, "level", ""))),
        "message": str(getattr(result, "message", "")),
        "scope": str(getattr(result, "scope", "")),
        "tags": list(getattr(result, "tags", []) or []),
        "metrics": dict(getattr(result, "metrics", {}) or {}),
        "contract_identities": _validation_contract_identities(result),
    }


def _iphs_assembly_port_payloads(graph: GraphIR) -> tuple[Any, ...]:
    graph_metadata = graph.metadata if isinstance(graph.metadata, dict) else {}
    assembly = graph_metadata.get("iphs_assembly")
    if not isinstance(assembly, dict):
        return ()
    contributions: list[Any] = list(assembly.get("contributions", ()) or ())
    node_ids_payload = assembly.get("contribution_node_ids")
    collect_all = bool(assembly.get("collect_node_contributions", False))
    if collect_all or node_ids_payload is not None:
        if node_ids_payload is None:
            node_ids = sorted(graph.nodes)
        elif isinstance(node_ids_payload, (list, tuple, set)):
            node_ids = [str(item) for item in node_ids_payload]
        else:
            node_ids = []
        for node_id in node_ids:
            node = graph.nodes.get(node_id)
            if node is None:
                continue
            node_contribution = node.config.get("iphs_contribution")
            if node_contribution is not None:
                contributions.append(node_contribution)
    ports: list[Any] = []
    top_level_ports = assembly.get("ports")
    if top_level_ports is not None:
        if isinstance(top_level_ports, (list, tuple)):
            ports.extend(top_level_ports)
        else:
            ports.append(top_level_ports)
    for contribution in contributions:
        contribution_ports = (
            contribution.get("ports")
            if isinstance(contribution, dict)
            else getattr(contribution, "ports", None)
        )
        if contribution_ports is None:
            continue
        if isinstance(contribution_ports, (list, tuple)):
            ports.extend(contribution_ports)
        else:
            ports.append(contribution_ports)
    return tuple(ports)


def _bind_port_assertion_to_iphs_assembly(
    *,
    normalized_op: str,
    graph: GraphIR,
    graph_strict: bool,
    structural_result: Dict[str, Any],
) -> list[GraphValidationIssue]:
    if normalized_op not in {"port_power_units_check", "port_matching_check"}:
        return []
    graph_metadata = graph.metadata if isinstance(graph.metadata, dict) else {}
    if not isinstance(graph_metadata.get("iphs_assembly"), dict):
        return []
    assembly_ports = _iphs_assembly_port_payloads(graph)
    if not assembly_ports:
        structural_result["iphs_assembly_contract_bound"] = None
        structural_result["iphs_assembly_contract_binding_reason"] = (
            "no_declared_iphs_ports"
        )
        return []
    if normalized_op == "port_power_units_check":
        from relic.validation.framework.rules.port_power import (
            validate_port_power_payload,
        )

        assembly_validation = validate_port_power_payload(
            {"ports": list(assembly_ports)}
        )
    else:
        from relic.validation.framework.rules.port_matching import (
            validate_port_matching_payload,
        )

        assembly_validation = validate_port_matching_payload(
            {"ports": list(assembly_ports), "strict": graph_strict}
        )
    assembly_result = _serialize_validation_result(assembly_validation)
    structural_result["iphs_assembly_contract_validation"] = assembly_result
    assertion_identities = structural_result.get("contract_identities", [])
    assembly_identities = assembly_result.get("contract_identities", [])
    identities_match = bool(assertion_identities) and (
        assertion_identities == assembly_identities
    )
    structural_result["iphs_assembly_contract_bound"] = identities_match
    if not bool(assembly_result.get("passed", False)):
        return [
            GraphValidationIssue(
                code="iphs_assembly_port_contract_invalid",
                message=(
                    "IPHS assembly ports fail the existing port validation owner."
                ),
                got=normalized_op,
                hint=(
                    "Correct the ports in graph.metadata.iphs_assembly; do not "
                    "validate a decoy assertion payload."
                ),
            )
        ]
    if not identities_match:
        return [
            GraphValidationIssue(
                code="port_assertion_iphs_binding_mismatch",
                message=(
                    "Port assertion contract identities do not match the actual "
                    "IPHS assembly ports."
                ),
                expected=json.dumps(
                    assembly_identities,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                got=json.dumps(
                    assertion_identities,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                hint=(
                    "Run the existing port validator on the same PortSpecIPHS "
                    "objects declared by iphs_assembly."
                ),
            )
        ]
    return []


def _required_iphs_level(
    context_payload: Dict[str, Any],
    *,
    strict: bool,
    canonical_role: str = "",
    graph_metadata: Optional[Dict[str, Any]] = None,
) -> tuple[Optional[int], list[GraphValidationIssue]]:
    """Resolve one monotone runtime threshold for validator and compiler.

    Descriptive iphs_level values 0-3 do not request runtime certification.
    Positive required_level values do; zero adds no runtime floor.
    Levels 4-5 are advanced claims regardless of their declaration location.
    """

    metadata = graph_metadata if isinstance(graph_metadata, dict) else {}
    assembly = metadata.get("iphs_assembly")
    assembly = assembly if isinstance(assembly, dict) else {}
    strict_assembly = assembly.get("strict") is True
    required_level, invalid_levels = resolve_iphs_level_requirement(
        {
            "context": context_payload,
            "graph.metadata": metadata,
            "graph.metadata.iphs_assembly": assembly,
        },
        strict_iphs=strict or strict_assembly,
        canonical_role="" if strict_assembly else canonical_role,
    )
    issues = [
        GraphValidationIssue(
            code="invalid_iphs_level_threshold",
            message=(
                "Graph and assertion IPHS acceptance levels must be exact "
                "integers from 0 through 5."
            ),
            expected=f"{source}=integer in 0..5",
            got=f"{source}={value!r}",
            hint=(
                "Remove the field when no threshold is declared; explicit "
                "None, bool, float, and string values are invalid."
            ),
        )
        for source, value in invalid_levels
    ]
    return required_level, issues


def _bind_certificate_identity_to_graph(
    *,
    graph: GraphIR,
    registry: NodeRegistry,
    scope: Any,
    required_level: Optional[int],
    structural_result: Dict[str, Any],
) -> list[GraphValidationIssue]:
    """Bind certificate declarations to one canonical Graph node identity."""

    structural_result.update(
        {
            "certificate_graph_identity_bound": False,
            "resolved_node_id": None,
            "resolved_runtime_class": None,
            "identity_evidence_scope": "compile_time_identity_only",
        }
    )
    issues: list[GraphValidationIssue] = []
    graph_metadata = graph.metadata if isinstance(graph.metadata, dict) else {}
    identities = structural_result.get("contract_identities", [])
    identity = (
        identities[0]
        if len(identities) == 1 and isinstance(identities[0], dict)
        else None
    )

    if identity is not None:
        expected_role = _normalized_canonical_role(graph_metadata)
        certificate_role = identity.get("canonical_role")
        if expected_role and certificate_role != expected_role:
            issues.append(
                GraphValidationIssue(
                    code="equivalence_certificate_role_mismatch",
                    message=(
                        "GraphIR canonical_role must match the canonical role in "
                        "its IPHS equivalence certificate."
                    ),
                    expected=expected_role,
                    got=str(certificate_role or "missing"),
                    hint=(
                        "Correct the graph declaration or bind the certificate "
                        "for this process."
                    ),
                )
            )
        if "formulation" in graph_metadata:
            graph_formulation = graph_metadata.get("formulation")
            certificate_formulation = identity.get("formulation")
            if certificate_formulation != graph_formulation:
                issues.append(
                    GraphValidationIssue(
                        code="equivalence_certificate_formulation_mismatch",
                        message=(
                            "Explicit GraphIR formulation must match the "
                            "certificate formulation exactly."
                        ),
                        expected=str(graph_formulation),
                        got=str(certificate_formulation),
                    )
                )
        if "iphs_level" in graph_metadata:
            graph_level = graph_metadata.get("iphs_level")
            certificate_level = identity.get("declared_iphs_level")
            if certificate_level != graph_level:
                issues.append(
                    GraphValidationIssue(
                        code="equivalence_certificate_declared_level_mismatch",
                        message=(
                            "Explicit GraphIR iphs_level must match the certificate "
                            "declared level exactly."
                        ),
                        expected=str(graph_level),
                        got=str(certificate_level),
                    )
                )

    if required_level is None or required_level < 4:
        return issues

    if isinstance(scope, str) and scope.strip() and scope in graph.nodes:
        node_id = scope
    elif (
        isinstance(scope, dict)
        and set(scope) == {"node_id"}
        and isinstance(scope.get("node_id"), str)
        and bool(scope["node_id"].strip())
    ):
        node_id = scope["node_id"]
    else:
        issues.append(
            GraphValidationIssue(
                code="equivalence_certificate_scope_unbound",
                message=(
                    "Level 4/5 equivalence certificate scope must identify one "
                    "exact GraphIR node."
                ),
                expected="node id string or exact {'node_id': '<node id>'}",
                got=repr(scope),
                hint=(
                    "Use the scientific process node id, not a role, phase label, "
                    "or observation spec mapping."
                ),
            )
        )
        return issues

    node = graph.nodes.get(node_id)
    structural_result["resolved_node_id"] = node_id
    if node is None:
        issues.append(
            GraphValidationIssue(
                code="equivalence_certificate_node_unresolved",
                message="Equivalence certificate scope references an unknown GraphIR node.",
                expected="existing GraphIR node id",
                got=node_id,
            )
        )
        return issues
    if not node.class_type.startswith("Functor:"):
        issues.append(
            GraphValidationIssue(
                code="equivalence_certificate_node_not_process",
                message=(
                    "Level 4/5 certificates must bind to a registered scientific "
                    "Functor process, not a state or forcing boundary."
                ),
                expected="Functor:<canonical dotted runtime class>",
                got=node.class_type,
            )
        )
        return issues
    if registry.get_definition(node.class_type) is None:
        issues.append(
            GraphValidationIssue(
                code="equivalence_certificate_node_unresolved",
                message=(
                    "Certificate-scoped Functor node has no authoritative "
                    "NodeRegistry definition."
                ),
                expected="registered NodeDefinition with canonical runtime_class",
                got=node.class_type,
            )
        )
        return issues
    # NodeRegistry.resolve_class_type reports an unresolvable class through
    # ImportError, TypeError, or ValueError; that is an unresolved-node issue
    # carrying the cause. Other errors are resolver defects and propagate.
    try:
        resolved_class = registry.resolve_class_type(node.class_type)
    except (ImportError, TypeError, ValueError) as exc:
        issues.append(
            GraphValidationIssue(
                code="equivalence_certificate_node_unresolved",
                message=(
                    "NodeRegistry could not resolve the certificate-scoped "
                    "Functor runtime class."
                ),
                expected="NodeRegistry-resolvable runtime class",
                got=f"{node.class_type}: {exc}",
            )
        )
        return issues
    if not isinstance(resolved_class, type):
        issues.append(
            GraphValidationIssue(
                code="equivalence_certificate_node_unresolved",
                message=(
                    "NodeRegistry did not resolve the certificate-scoped node "
                    "to a runtime class."
                ),
                expected="runtime class",
                got=node.class_type,
            )
        )
        return issues

    runtime_class = f"{resolved_class.__module__}.{resolved_class.__qualname__}"
    structural_result["resolved_runtime_class"] = runtime_class
    if identity is None:
        issues.append(
            GraphValidationIssue(
                code="equivalence_certificate_identity_unavailable",
                message=(
                    "Equivalence rule did not provide exactly one certificate "
                    "contract identity for Graph binding."
                ),
                expected="one structural contract_identity",
                got=repr(identities),
            )
        )
        return issues
    process_id = identity.get("process_id")
    if process_id != runtime_class:
        issues.append(
            GraphValidationIssue(
                code="equivalence_certificate_process_identity_mismatch",
                message=(
                    "Certificate process_id does not match the NodeRegistry-resolved "
                    "Graph process identity."
                ),
                expected=runtime_class,
                got=str(process_id or "missing"),
                hint=(
                    "Bind the certificate for the scoped process; do not import or "
                    "substitute an unrelated same-role process."
                ),
            )
        )
    if not issues:
        structural_result["certificate_graph_identity_bound"] = True
    return issues


def _structural_parameter_resolver(
    context_payload: Dict[str, Any],
    *,
    strict: bool,
) -> Optional[Any]:
    """Resolve the existing ParameterManager access path for structural rules."""

    parameter_resolver = context_payload.get("parameter_resolver")
    parameter_manager = context_payload.get("parameter_manager")
    if not callable(parameter_resolver) and parameter_manager is not None:
        candidate = getattr(parameter_manager, "get_parameter", None)
        if callable(candidate):
            parameter_resolver = candidate
    if strict and not callable(parameter_resolver):
        from relic.parameters import get_global_manager

        parameter_resolver = get_global_manager().get_parameter
    return parameter_resolver if callable(parameter_resolver) else None


def _run_structural_assertion(
    normalized_op: str,
    context_payload: Dict[str, Any],
    *,
    strict: bool = False,
    canonical_role: str = "",
    required_iphs_level: Optional[int] = None,
    require_affinity: bool = False,
) -> Optional[Any]:
    if normalized_op == "cochain_complex_check":
        from relic.validation.framework.rules.dec_complex import (
            resolve_dec_complex_tolerance,
            validate_dec_complex_bundle,
        )

        bundle = context_payload.get("dec_complex_bundle") or context_payload.get("bundle")
        tol = resolve_dec_complex_tolerance(context_payload)
        return validate_dec_complex_bundle(bundle, tol=tol, strict=strict)
    if normalized_op == "hodge_psd_check":
        from relic.validation.framework.rules.hodge import validate_hodge_spec

        spec = context_payload.get("hodge_spec") or context_payload.get("spec")
        return validate_hodge_spec(
            spec,
            passive=True,
            strict=strict,
            parameter_resolver=_structural_parameter_resolver(
                context_payload,
                strict=strict,
            ),
        )
    if normalized_op == "iphs_structure_check":
        from relic.validation.framework.rules.iphs import validate_iphs_spec

        spec = context_payload.get("iphs_spec") or context_payload.get("spec")
        return validate_iphs_spec(spec, required_level=required_iphs_level)
    if normalized_op == "chain_map_check":
        from relic.validation.framework.rules.chain_map import validate_chain_map_payload

        return validate_chain_map_payload(context_payload)
    if normalized_op == "entropy_step_check":
        from relic.validation.framework.rules.entropy_step import validate_entropy_step_payload

        return validate_entropy_step_payload(
            context_payload,
            strict=strict,
            parameter_resolver=_structural_parameter_resolver(
                context_payload,
                strict=strict,
            ),
        )
    if normalized_op == "port_power_units_check":
        from relic.validation.framework.rules.port_power import validate_port_power_payload

        return validate_port_power_payload(context_payload)
    if normalized_op == "port_matching_check":
        from relic.validation.framework.rules.port_matching import validate_port_matching_payload

        effective_payload = dict(context_payload)
        if strict:
            effective_payload["strict"] = True
        return validate_port_matching_payload(effective_payload)
    if normalized_op == "overflow_routing_check":
        from relic.validation.framework.rules.overflow_routing import validate_overflow_routing_payload

        return validate_overflow_routing_payload(context_payload, strict=strict)
    if normalized_op == "equivalence_certificate_check":
        from relic.validation.framework.rules.equivalence_certificate import (
            validate_equivalence_certificate_payload,
        )

        certificate = (
            context_payload.get("certificate")
            or context_payload.get("iphs_equivalence_certificate")
            or context_payload
        )
        return validate_equivalence_certificate_payload(
            certificate,
            strict=strict,
            required_level=required_iphs_level,
        )
    if normalized_op == "reaction_network_check":
        from relic.validation.framework.rules.reaction_network import validate_reaction_network_payload

        return validate_reaction_network_payload(
            context_payload, strict=strict, require_affinity=require_affinity,
        )
    if normalized_op == "stochastic_sdae_check":
        from relic.validation.framework.rules.stochastic_sdae import validate_stochastic_sdae_payload

        effective_payload = dict(context_payload)
        if strict:
            effective_payload["require_declared_structure"] = True
        required_semantics = effective_payload.get("required_semantics", ())
        return validate_stochastic_sdae_payload(
            effective_payload,
            required_semantics=required_semantics,
        )
    if normalized_op == "state_domain_check":
        from relic.validation.framework.rules.state_domain import validate_state_domain_payload

        state = context_payload.get("state")
        constraints = context_payload.get("state_domain_constraints")
        return validate_state_domain_payload(
            state if isinstance(state, dict) else {},
            constraints,
            strict=strict,
            required_state_vars=context_payload.get("required_state_vars"),
        )
    if normalized_op == "observation_diagnostic_check":
        from relic.validation.framework.rules.observation_diagnostic import (
            validate_observation_diagnostic_payload,
        )

        observations = context_payload.get("observation_specs") or ()
        diagnostics = context_payload.get("diagnostic_specs") or ()
        return validate_observation_diagnostic_payload(
            observations,
            diagnostics,
            strict=strict,
            required_kinds=context_payload.get("required_observation_kinds", ()),
            available_state_vars=context_payload.get("available_state_vars", ()),
            required_observation_state_vars=context_payload.get(
                "required_observation_state_vars", ()
            ),
            spec_node_bindings=context_payload.get("spec_node_bindings"),
        )
    return None


def _iter_assertion_payloads(assertion_payload: Any) -> list[Dict[str, Any]]:
    if isinstance(assertion_payload, dict):
        for key in ("assertions", "items", "checks"):
            values = assertion_payload.get(key)
            if isinstance(values, list):
                return [dict(item) for item in values if isinstance(item, dict)]
        return [assertion_payload]
    if isinstance(assertion_payload, list):
        return [dict(item) for item in assertion_payload if isinstance(item, dict)]
    return []


def _strict_entropy_tolerance_binding_issues(
    assertion_items: list[Dict[str, Any]],
    graph_metadata: Dict[str, Any],
) -> list[GraphValidationIssue]:
    """Bind the strict entropy audit to the certificate-owned tolerance key."""

    if not _is_strict_iphs_graph(graph_metadata):
        return []
    contexts: dict[str, Dict[str, Any]] = {}
    for item in assertion_items:
        op = str(item.get("op", "")).strip()
        context = item.get("context")
        if op in {"entropy_step_check", "equivalence_certificate_check"} and isinstance(
            context,
            dict,
        ):
            contexts.setdefault(op, context)
    entropy_context = contexts.get("entropy_step_check")
    certificate_context = contexts.get("equivalence_certificate_check")
    if entropy_context is None or certificate_context is None:
        return []

    from relic.computation.operators.iphs_certificate import (
        IPHSEquivalenceCertificate,
        load_iphs_equivalence_certificate,
    )

    source = (
        certificate_context.get("certificate")
        or certificate_context.get("iphs_equivalence_certificate")
        or certificate_context
    )
    # A certificate that cannot be read (OSError) or parsed (ValueError,
    # TypeError from load_iphs_equivalence_certificate) leaves the strict
    # tolerance binding unverifiable: that is reported, never skipped.
    try:
        certificate = (
            source
            if isinstance(source, IPHSEquivalenceCertificate)
            else load_iphs_equivalence_certificate(source)
        )
    except (OSError, TypeError, ValueError) as exc:
        return [
            GraphValidationIssue(
                code="strict_entropy_tolerance_binding_unverifiable",
                message=(
                    "Strict entropy tolerance binding cannot be checked because the "
                    "IPHS equivalence certificate could not be loaded."
                ),
                expected="loadable IPHSEquivalenceCertificate",
                got=f"{type(exc).__name__}: {exc}",
                hint="Provide a readable certificate mapping or JSON/YAML file.",
            )
        ]
    certificate_keys = certificate.strict_tolerance_parameter_keys()
    if len(certificate_keys) != 1:
        # Without exactly one certificate-owned strict tolerance key the
        # binding cannot be checked: reported, never skipped.
        return [
            GraphValidationIssue(
                code="strict_entropy_tolerance_binding_unverifiable",
                message=(
                    "Strict entropy tolerance binding requires the IPHS equivalence "
                    "certificate to own exactly one strict tolerance parameter key."
                ),
                expected="exactly one certificate strict tolerance_parameter_key",
                got=repr(list(certificate_keys)),
                hint="Declare one shared tolerance_parameter_key in the certificate evidence.",
            )
        ]
    expected_key = certificate_keys[0]
    raw_entropy_key = entropy_context.get("tolerance_parameter_key")
    entropy_key = raw_entropy_key.strip() if isinstance(raw_entropy_key, str) else ""
    if entropy_key == expected_key:
        return []
    return [
        GraphValidationIssue(
            code="strict_entropy_tolerance_binding_mismatch",
            message=(
                "Strict entropy and IPHS certificate audits must resolve the same "
                "ParameterManager tolerance key."
            ),
            expected=expected_key,
            got=entropy_key or "missing",
            hint="Use the certificate-owned tolerance_parameter_key for entropy_step_check.",
        )
    ]


def _strict_reaction_iphs_closure_issues(
    assertion_items: list[Dict[str, Any]],
    graph_metadata: Dict[str, Any],
) -> list[GraphValidationIssue]:
    """Fail closed until one IPHS spec owns the executable reaction closure."""

    if not _is_strict_iphs_graph(graph_metadata):
        return []
    if _normalized_canonical_role(graph_metadata) != "reaction":
        return []
    observed_ops = {
        str(item.get("op", "")).strip()
        for item in assertion_items
        if isinstance(item, dict)
    }
    if not {"iphs_structure_check", "reaction_network_check"}.issubset(observed_ops):
        return []
    return [
        GraphValidationIssue(
            code="strict_reaction_iphs_closure_not_implemented",
            message=(
                "Strict reaction IPHS evidence requires one executable structure "
                "to bind N, r, affinity convention, effort, and the resulting "
                "energy or entropy balance; independent passing assertions cannot "
                "establish that closure [N]."
            ),
            expected=(
                "one IPHSStructureSpec-owned executable reaction closure witness"
            ),
            got="independent iphs_structure_check and reaction_network_check payloads",
            hint=(
                "Keep the process at IPHS Level 3 or lower until the existing IPHS "
                "owner computes and audits the reaction contribution without double counting."
            ),
        )
    ]


def _validate_single_assertion_payload(
    assertion_payload: Dict[str, Any],
    graph: GraphIR,
    registry: NodeRegistry,
    ssot_catalog: Optional[SSOTCatalog],
) -> tuple[Dict[str, Any], List[GraphValidationIssue]]:
    audit = _default_assertion_audit()
    audit["present"] = True
    phi = assertion_payload.get("phi")
    op = assertion_payload.get("op")
    scope = assertion_payload.get("scope")
    context_payload = assertion_payload.get("context")
    audit_key = assertion_payload.get("audit_key")
    issues = _validate_assertion_fields(
        phi=phi,
        op=op,
        scope=scope,
        context_payload=context_payload,
    )

    op_resolved = False
    normalized_op = str(op).strip() if isinstance(op, str) else ""
    if normalized_op:
        op_resolved = _resolve_assertion_op(graph, registry, ssot_catalog, normalized_op)
        if not op_resolved:
            issues.append(
                GraphValidationIssue(
                    code="assertion_op_unresolvable",
                    message="Graph assertion op cannot be resolved to node, class, SSOT entry, or node metadata protocol.",
                    got=normalized_op,
                    hint="Use an existing node id/class_type or register op_protocols in NodeDefinition metadata.",
                )
            )
    canonical_audit_key = _STRUCTURAL_ASSERTION_AUDIT_KEYS.get(normalized_op)
    if canonical_audit_key is not None:
        if audit_key is None:
            audit_key = canonical_audit_key
        elif audit_key != canonical_audit_key:
            issues.append(
                GraphValidationIssue(
                    code="assertion_audit_key_mismatch",
                    message=(
                        f"Structural assertion '{normalized_op}' must use its canonical audit key."
                    ),
                    expected=canonical_audit_key,
                    got=str(audit_key),
                )
            )
    structural_result: Optional[Dict[str, Any]] = None
    if op_resolved and _is_structural_assertion_op(normalized_op):
        if not isinstance(context_payload, dict):
            issues.append(
                GraphValidationIssue(
                    code="structural_assertion_missing_context",
                    message="Structural assertion requires assertion.context mapping.",
                    got=type(context_payload).__name__,
                )
            )
        else:
            graph_metadata = graph.metadata if isinstance(graph.metadata, dict) else {}
            graph_strict = _is_strict_structural_graph(graph_metadata)
            strict_iphs = _is_strict_iphs_graph(graph_metadata)
            effective_context = dict(context_payload)
            if graph_strict:
                effective_context["strict"] = True
                if normalized_op == "stochastic_sdae_check":
                    payload_semantics = effective_context.get("required_semantics", ())
                    if isinstance(payload_semantics, str):
                        payload_semantics = (payload_semantics,)
                    combined_semantics = {
                        str(item).strip()
                        for item in payload_semantics
                        if str(item).strip()
                    }
                    combined_semantics.update(
                        _required_stochastic_semantics(graph_metadata)
                    )
                    effective_context["required_semantics"] = sorted(
                        combined_semantics
                    )
                if normalized_op == "state_domain_check":
                    effective_context["required_state_vars"] = (
                        graph.required_initial_state_keys()
                    )
                if normalized_op == "observation_diagnostic_check":
                    payload_kinds = effective_context.get(
                        "required_observation_kinds", ()
                    )
                    if isinstance(payload_kinds, str):
                        payload_kinds = (payload_kinds,)
                    combined_kinds = {
                        str(item).strip()
                        for item in payload_kinds
                        if str(item).strip()
                    }
                    combined_kinds.update(
                        _required_observation_kinds(graph_metadata)
                    )
                    effective_context["required_observation_kinds"] = sorted(
                        combined_kinds
                    )
                    spec_bindings = _observation_spec_node_bindings(
                        graph,
                        scope,
                        effective_context,
                    )
                    effective_context["spec_node_bindings"] = spec_bindings
                    scoped_variables: set[str] = set()
                    for binding in spec_bindings.values():
                        scoped_variables.update(
                            item
                            for item in (
                                *binding.get("input_ports", ()),
                                *binding.get("output_ports", ()),
                            )
                            if isinstance(item, str) and item
                        )
                    effective_context["available_state_vars"] = sorted(scoped_variables)
            required_iphs_level: Optional[int] = None
            level_issues: list[GraphValidationIssue] = []
            require_affinity = strict_iphs
            if normalized_op == "reaction_network_check" and "require_affinity" in effective_context:
                declared_requirement = effective_context["require_affinity"]
                if type(declared_requirement) is not bool:
                    level_issues.append(GraphValidationIssue(
                        code="invalid_reaction_affinity_requirement",
                        message="An explicit require_affinity declaration must be bool.",
                        got=repr(declared_requirement),
                    ))
                    issues.extend(level_issues)
                else:
                    require_affinity = require_affinity or declared_requirement
            if normalized_op in {
                "iphs_structure_check",
                "equivalence_certificate_check",
            }:
                required_iphs_level, level_issues = _required_iphs_level(
                    effective_context,
                    strict=strict_iphs,
                    canonical_role=_normalized_canonical_role(graph_metadata),
                    graph_metadata=graph_metadata,
                )
                issues.extend(level_issues)
            validation_result = None
            if not level_issues:
                validation_result = _run_structural_assertion(
                    normalized_op,
                    effective_context,
                    strict=graph_strict,
                    canonical_role=_normalized_canonical_role(graph_metadata),
                    required_iphs_level=required_iphs_level,
                    require_affinity=require_affinity,
                )
            if validation_result is not None:
                structural_result = _serialize_validation_result(validation_result)
                if normalized_op == "equivalence_certificate_check":
                    issues.extend(
                        _bind_certificate_identity_to_graph(
                            graph=graph,
                            registry=registry,
                            scope=scope,
                            required_level=required_iphs_level,
                            structural_result=structural_result,
                        )
                    )
                issues.extend(
                    _bind_port_assertion_to_iphs_assembly(
                        normalized_op=normalized_op,
                        graph=graph,
                        graph_strict=graph_strict,
                        structural_result=structural_result,
                    )
                )
                if not bool(structural_result.get("passed", False)):
                    issues.append(
                        GraphValidationIssue(
                            code="structural_assertion_failed",
                            message=str(structural_result.get("message", "")),
                            got=normalized_op,
                            hint="Inspect assertion_audit.structural_result for details.",
                        )
                    )

    audit.update(
        {
            "phi": phi if isinstance(phi, str) else None,
            "op": normalized_op or None,
            "scope": scope,
            "audit_key": audit_key if isinstance(audit_key, str) else None,
            "op_resolved": op_resolved,
            "valid": len(issues) == 0,
            "issues": [{"code": issue.code, "message": issue.message} for issue in issues],
            "structural_result": structural_result,
        }
    )
    return audit, issues


def _validate_assertion_payload(
    graph: GraphIR,
    registry: NodeRegistry,
    ssot_catalog: Optional[SSOTCatalog],
) -> tuple[Dict[str, Any], List[GraphValidationIssue]]:
    audit = _default_assertion_audit()
    graph_metadata = graph.metadata if isinstance(graph.metadata, dict) else {}
    role_issues = _strict_canonical_role_issues(graph_metadata)
    assertion_payload = (
        getattr(graph, "assertion", {}) or {}
    ) or graph_metadata.get("structural_assertions") or graph_metadata.get("mandatory_structural_assertions") or {}
    _, level_issues = _required_iphs_level(
        {}, strict=False, graph_metadata=graph_metadata,
    )
    if level_issues:
        issues = [*role_issues, *level_issues]
        audit["present"] = bool(assertion_payload)
        audit["valid"] = False
        audit["issues"] = [
            {"code": item.code, "message": item.message} for item in issues
        ]
        return audit, issues
    if not assertion_payload:
        if _required_structural_assertion_ops(graph_metadata):
            issue = GraphValidationIssue(
                code="structural_assertions_required",
                message="Requested scientific validation requires structural assertions.",
                hint="Use GraphIR.assertion or graph.metadata.structural_assertions with existing assertion ops.",
            )
            audit["valid"] = False
            audit["required"] = True
            issues = [*role_issues, issue]
            audit["issues"] = [
                {"code": item.code, "message": item.message} for item in issues
            ]
            return audit, issues
        audit["valid"] = not role_issues
        return audit, role_issues

    assertion_items = _iter_assertion_payloads(assertion_payload)
    if not assertion_items:
        issue = GraphValidationIssue(
            code="assertion_invalid_payload",
            message="Graph assertion must be a mapping or a list of mappings.",
            hint="Use {'phi': str, 'op': str, 'scope': str|dict}.",
        )
        audit["present"] = True
        audit["valid"] = False
        issues = [*role_issues, issue]
        audit["issues"] = [
            {"code": item.code, "message": item.message} for item in issues
        ]
        return audit, issues

    graph_metadata = graph.metadata if isinstance(graph.metadata, dict) else {}
    if len(assertion_items) == 1:
        item_audit, item_issues = _validate_single_assertion_payload(
            assertion_items[0], graph, registry, ssot_catalog
        )
        item_op = str(assertion_items[0].get("op", "")).strip()
        required_issues = _missing_required_assertion_issues(
            graph_metadata,
            {item_op} if item_op else set(),
        )
        binding_issues = _strict_entropy_tolerance_binding_issues(
            assertion_items,
            graph_metadata,
        )
        binding_issues.extend(
            _strict_reaction_iphs_closure_issues(assertion_items, graph_metadata)
        )
        if required_issues or binding_issues:
            item_issues.extend(required_issues)
            item_issues.extend(binding_issues)
            item_audit["valid"] = False
            item_audit["required_ops"] = sorted(
                _required_structural_assertion_ops(graph_metadata)
            )
            item_audit["issues"] = [
                {"code": issue.code, "message": issue.message} for issue in item_issues
            ]
        if role_issues:
            item_issues.extend(role_issues)
            item_audit["valid"] = False
            item_audit["issues"] = [
                {"code": issue.code, "message": issue.message} for issue in item_issues
            ]
        return item_audit, item_issues

    all_issues: list[GraphValidationIssue] = list(role_issues)
    results: list[Dict[str, Any]] = []
    seen_structural_ops: set[str] = set()
    for item in assertion_items:
        item_op = str(item.get("op", "")).strip()
        if item_op in _STRUCTURAL_ASSERTION_OPS:
            if item_op in seen_structural_ops:
                all_issues.append(
                    GraphValidationIssue(
                        code="duplicate_structural_assertion",
                        message=f"Structural assertion '{item_op}' is declared more than once.",
                        got=item_op,
                    )
                )
            seen_structural_ops.add(item_op)
        item_audit, item_issues = _validate_single_assertion_payload(
            item,
            graph,
            registry,
            ssot_catalog,
        )
        results.append(item_audit)
        all_issues.extend(item_issues)

    required_issues = _missing_required_assertion_issues(graph_metadata, seen_structural_ops)
    all_issues.extend(required_issues)
    all_issues.extend(
        _strict_entropy_tolerance_binding_issues(assertion_items, graph_metadata)
    )
    all_issues.extend(
        _strict_reaction_iphs_closure_issues(assertion_items, graph_metadata)
    )

    audit["present"] = True
    audit["valid"] = not all_issues
    audit["op_resolved"] = all(bool(item.get("op_resolved", False)) for item in results)
    audit["results"] = results
    audit["required_ops"] = sorted(_required_structural_assertion_ops(graph_metadata))
    audit["issues"] = [
        {"code": issue.code, "message": issue.message} for issue in all_issues
    ]
    return audit, all_issues


_EXECUTION_ONLY_CONFIG_FIELDS = frozenset(
    {"execution_group", "iphs_contribution"}
)


def _validate_execution_groups(graph: GraphIR) -> List[GraphValidationIssue]:
    """Errors for node ``config.execution_group`` values that registration rejects.

    Strict and non-strict validation alike: register_functor raises for such a
    value, so validation is never more permissive than runtime. The predicate
    and the message are the simulator's own
    (``dependency_ops._is_valid_execution_group`` and
    ``_invalid_execution_group_message``).
    """
    errors: List[GraphValidationIssue] = []
    expected = (
        f"{', '.join(sorted(_VALID_EXECUTION_GROUPS))}, or omitted/None"
    )
    for node_id, node in graph.nodes.items():
        config = node.config or {}
        if "execution_group" not in config:
            continue
        execution_group = config["execution_group"]
        if execution_group is None or _is_valid_execution_group(execution_group):
            continue
        errors.append(
            GraphValidationIssue(
                code="invalid_execution_group",
                message=_invalid_execution_group_message(
                    execution_group,
                    source=f"config.execution_group of node '{node_id}'",
                ),
                node_id=node_id,
                expected=expected,
                got=repr(execution_group),
                hint=(
                    "Use fast, medium, or slow, or omit the field so that "
                    "registration resolves the group from the Functor's "
                    "declarations."
                ),
            )
        )
    return errors


def _validate_registered_node_contracts(
    graph: GraphIR,
    registry: NodeRegistry,
    *,
    strict: bool,
) -> tuple[List[GraphValidationIssue], List[GraphValidationIssue]]:
    errors: List[GraphValidationIssue] = []
    warnings: List[GraphValidationIssue] = []
    for node_id, node in graph.nodes.items():
        definition = registry.get_definition(node.class_type)
        if definition is None:
            continue

        declared_inputs = set(node.inputs)
        definition_inputs = set(definition.inputs)
        for input_name in sorted(declared_inputs - definition_inputs):
            issue = GraphValidationIssue(
                code="input_not_in_registered_definition",
                message=(
                    f"Node input '{input_name}' is not present in the "
                    "authoritative NodeDefinition."
                ),
                node_id=node_id,
                port=input_name,
            )
            (errors if strict else warnings).append(issue)

        declared_outputs = set(node.outputs)
        definition_outputs = set(definition.outputs)
        for output_name in sorted(declared_outputs - definition_outputs):
            issue = GraphValidationIssue(
                code="output_not_in_registered_definition",
                message=(
                    f"Node output '{output_name}' is not present in the "
                    "authoritative NodeDefinition."
                ),
                node_id=node_id,
                port=output_name,
            )
            (errors if strict else warnings).append(issue)

        for input_name, port_spec in definition.inputs.items():
            payload = node.inputs.get(input_name)
            if payload is None:
                if port_spec.required:
                    issue = GraphValidationIssue(
                        code="missing_required_input_binding",
                        message=(
                            f"Required input '{input_name}' has no executable "
                            "GraphIR binding."
                        ),
                        node_id=node_id,
                        port=input_name,
                        hint=(
                            "Bind it with inputs.<port>.link or the same-name "
                            "inputs.<port>.state declaration."
                        ),
                    )
                    (errors if strict else warnings).append(issue)
                continue
            if not isinstance(payload, dict):
                issue = GraphValidationIssue(
                    code="unsupported_input_binding",
                    message="GraphIR input binding must be a mapping.",
                    node_id=node_id,
                    port=input_name,
                    got=type(payload).__name__,
                )
                (errors if strict else warnings).append(issue)
                continue
            binding_keys = {
                key for key in ("link", "state") if key in payload
            }
            if len(binding_keys) != 1 or len(payload) != 1:
                issue = GraphValidationIssue(
                    code="unsupported_input_binding",
                    message=(
                        "GraphIR input must declare exactly one executable "
                        "link or state binding."
                    ),
                    node_id=node_id,
                    port=input_name,
                    got=str(sorted(payload)),
                )
                (errors if strict else warnings).append(issue)
                continue
            if "link" in binding_keys:
                link_payload = payload["link"]
                if (
                    not isinstance(link_payload, (list, tuple))
                    or len(link_payload) != 2
                    or not all(str(item).strip() for item in link_payload)
                ):
                    issue = GraphValidationIssue(
                        code="unsupported_input_binding",
                        message=(
                            "GraphIR link binding must contain source node "
                            "and source port."
                        ),
                        node_id=node_id,
                        port=input_name,
                    )
                    (errors if strict else warnings).append(issue)
                continue
            state_name = payload["state"]
            if not isinstance(state_name, str) or not state_name.strip():
                issue = GraphValidationIssue(
                    code="unsupported_input_binding",
                    message="GraphIR state binding must be a non-empty name.",
                    node_id=node_id,
                    port=input_name,
                )
                (errors if strict else warnings).append(issue)
            elif state_name != input_name:
                issue = GraphValidationIssue(
                    code="state_input_alias_not_executable",
                    message=(
                        "GraphExecutor reads flat StateManager keys and does "
                        "not implement state-input aliases."
                    ),
                    node_id=node_id,
                    port=input_name,
                    expected=input_name,
                    got=state_name,
                    hint=(
                        "Use the same state and input name or add an explicit "
                        "executable adapter node."
                    ),
                )
                (errors if strict else warnings).append(issue)

        adapter_contract = definition.metadata.get("adapter_contract")
        if isinstance(adapter_contract, dict):
            graph_contract = {
                key: value
                for key, value in node.config.items()
                if key not in _EXECUTION_ONLY_CONFIG_FIELDS
            }
            if graph_contract != adapter_contract:
                issue = GraphValidationIssue(
                    code="adapter_contract_mismatch",
                    message=(
                        "Adapter GraphIR config differs from its authoritative "
                        "NodeRegistry contract."
                    ),
                    node_id=node_id,
                    expected=str(adapter_contract),
                    got=str(graph_contract),
                    hint=(
                        "Rebuild the selected-profile registry from the same "
                        "frozen adapter configuration."
                    ),
                )
                (errors if strict else warnings).append(issue)
        config_contract = definition.metadata.get("config_contract")
        if isinstance(config_contract, dict):
            graph_contract = {
                key: value
                for key, value in node.config.items()
                if key not in _EXECUTION_ONLY_CONFIG_FIELDS
            }
            if graph_contract != config_contract:
                issue = GraphValidationIssue(
                    code="node_config_contract_mismatch",
                    message=(
                        "GraphIR config differs from its authoritative "
                        "NodeRegistry constructor contract."
                    ),
                    node_id=node_id,
                    expected=str(config_contract),
                    got=str(graph_contract),
                    hint=(
                        "Build the selected registry and GraphIR from the same "
                        "frozen constructor configuration."
                    ),
                )
                (errors if strict else warnings).append(issue)
    return errors, warnings


def _bound_unit_cast_endpoints(
    graph: GraphIR,
    registry: NodeRegistry,
    unit_converter: UnitConverter,
) -> tuple[set[tuple[str, str, str]], List[GraphValidationIssue]]:
    """Resolve ports proven to be adjacent to one executable unit_cast node."""
    bound: set[tuple[str, str, str]] = set()
    issues: List[GraphValidationIssue] = []
    for adapter_node_id, adapter_node in graph.nodes.items():
        definition = registry.get_definition(adapter_node.class_type)
        if definition is None:
            continue
        if definition.metadata.get("runtime_class") != (
            "relic.orchestration.adapters.ExecutableAdapterFunctor"
        ):
            continue
        contract = definition.metadata.get("adapter_contract")
        if not isinstance(contract, dict) or contract.get("adapter_type") != "unit_cast":
            continue
        graph_contract = {
            key: value
            for key, value in adapter_node.config.items()
            if key not in _EXECUTION_ONLY_CONFIG_FIELDS
        }
        if graph_contract != contract:
            continue
        source_node_id = str(contract.get("source_node", "")).strip()
        source_port_name = str(contract.get("source_port", "")).strip()
        target_port_name = str(contract.get("target_port", "")).strip()
        variable_name = str(contract.get("variable_name", "")).strip()
        source_units = str(contract.get("source_units", "")).strip()
        target_units = str(contract.get("target_units", "")).strip()
        if not all(
            (
                source_node_id,
                source_port_name,
                target_port_name,
                variable_name,
                source_units,
                target_units,
            )
        ):
            continue
        adapter_input = definition.inputs.get(source_port_name)
        adapter_output = definition.outputs.get(target_port_name)
        if (
            adapter_input is None
            or adapter_output is None
            or str(adapter_input.units or "").strip() != source_units
            or str(adapter_output.units or "").strip() != target_units
            or adapter_input.standard_name != variable_name
            or adapter_output.standard_name != variable_name
        ):
            continue
        try:
            conversion_supported = unit_converter.can_convert(
                source_units,
                target_units,
                variable_name,
            )
            if conversion_supported:
                unit_converter.convert(
                    1.0,
                    variable_name=variable_name,
                    source_units=source_units,
                    target_units=target_units,
                    context=None,
                )
        except ValueError:
            # UnitConverter.convert rejects an unsupported pair or a missing
            # conversion context with ValueError: reported as not executable
            # below. Converter infrastructure failures propagate.
            conversion_supported = False
        if not conversion_supported:
            issues.append(
                GraphValidationIssue(
                    code="unit_cast_not_executable",
                    message=(
                        "Declared unit_cast is not executable through the canonical "
                        "UnitConverter without an undeclared conversion context."
                    ),
                    node_id=adapter_node_id,
                    expected=(
                        f"supported {variable_name}: {source_units} -> {target_units}"
                    ),
                    got="unsupported conversion contract",
                    hint=(
                        "Use a registered canonical conversion or declare the "
                        "existing adapter context inputs required by that conversion."
                    ),
                )
            )
            continue
        matching_inputs = [
            link
            for link in graph.links
            if link.source.node == source_node_id
            and link.source.port == source_port_name
            and link.target.node == adapter_node_id
            and link.target.port == source_port_name
        ]
        if len(matching_inputs) != 1:
            continue
        source_node = graph.nodes.get(source_node_id)
        if source_node is None:
            continue
        source_definition = registry.get_definition(source_node.class_type)
        if source_definition is None:
            continue
        source_port = source_definition.outputs.get(source_port_name)
        if (
            source_port is None
            or str(source_port.units or "").strip() != source_units
            or source_port.standard_name != variable_name
        ):
            continue
        bound.update(
            {
                (source_node_id, "output", source_port_name),
                (adapter_node_id, "input", source_port_name),
                (adapter_node_id, "output", target_port_name),
            }
        )
        for link in graph.links:
            if (
                link.source.node != adapter_node_id
                or link.source.port != target_port_name
            ):
                continue
            target_node = graph.nodes.get(link.target.node)
            if target_node is None:
                continue
            target_definition = registry.get_definition(target_node.class_type)
            if target_definition is None:
                continue
            target_port = target_definition.inputs.get(link.target.port)
            if (
                target_port is not None
                and str(target_port.units or "").strip() == target_units
                and target_port.standard_name == adapter_output.standard_name
            ):
                bound.add((link.target.node, "input", link.target.port))
    return bound, issues


def _validate_declared_port_unit_contracts(
    graph: GraphIR,
    registry: NodeRegistry,
    *,
    strict: bool,
    ssot_catalog: Optional[SSOTCatalog],
    unit_converter: UnitConverter,
) -> tuple[List[GraphValidationIssue], List[GraphValidationIssue]]:
    """Validate every Graph-declared authoritative port, including unlinked ports."""
    errors: List[GraphValidationIssue] = []
    warnings: List[GraphValidationIssue] = []
    bound_unit_cast_endpoints, unit_cast_issues = _bound_unit_cast_endpoints(
        graph,
        registry,
        unit_converter,
    )
    (errors if strict else warnings).extend(unit_cast_issues)
    for node_id, node in graph.nodes.items():
        definition = registry.get_definition(node.class_type)
        if definition is None:
            continue
        declared_ports = (
            ("input", node.inputs.keys(), definition.inputs),
            ("output", node.outputs, definition.outputs),
        )
        for direction, port_names, authoritative_ports in declared_ports:
            for port_name in sorted(port_names):
                port = authoritative_ports.get(port_name)
                if port is None:
                    continue
                units = str(port.units or "").strip()
                if not units or dimensionality_from_unit(units) is None:
                    issue = GraphValidationIssue(
                        code="unsupported_port_units",
                        message=(
                            f"Declared {direction} port '{port_name}' has missing or "
                            "unsupported units."
                        ),
                        node_id=node_id,
                        port=port_name,
                        expected=(
                            "unit label recognized by the canonical dimensionality owner"
                        ),
                        got=units or "missing",
                        hint="Use a supported canonical SI-token unit expression.",
                    )
                    (errors if strict else warnings).append(issue)
                    continue
                if ssot_catalog is None:
                    continue
                for message in check_port_dimension_consistency([port], ssot_catalog):
                    issue = GraphValidationIssue(
                        code="dimension_inconsistent",
                        message=message,
                        node_id=node_id,
                        port=port_name,
                        hint="Align port units with SSOT dimensionality metadata.",
                    )
                    (errors if strict else warnings).append(issue)
                entry = ssot_catalog.get(port.name) or ssot_catalog.get_by_standard_name(
                    port.standard_name
                )
                canonical_units = str(getattr(entry, "units", "") or "").strip()
                if not canonical_units or units == canonical_units:
                    continue
                if (node_id, direction, port_name) in bound_unit_cast_endpoints:
                    continue
                issue = GraphValidationIssue(
                    code="unit_scale_or_offset_unbound",
                    message=(
                        f"Declared {direction} port '{port_name}' does not use "
                        "the exact SSOT unit and has no executable unit_cast binding."
                    ),
                    node_id=node_id,
                    port=port_name,
                    expected=canonical_units,
                    got=units,
                    hint=(
                        "Use the exact SSOT unit or insert the existing executable "
                        "unit_cast adapter with bound source_units and target_units."
                    ),
                )
                (errors if strict else warnings).append(issue)
    return errors, warnings


def _is_publication_profile(graph: GraphIR) -> bool:
    profile_id = graph.metadata.get("profile_id")
    return isinstance(profile_id, str) and bool(profile_id.strip())


def _validate_publication_port_support_completeness(
    graph: GraphIR,
    registry: NodeRegistry,
    *,
    strict: bool,
) -> tuple[List[GraphValidationIssue], List[GraphValidationIssue]]:
    errors: List[GraphValidationIssue] = []
    warnings: List[GraphValidationIssue] = []
    if not _is_publication_profile(graph):
        return errors, warnings

    for node_id, node in graph.nodes.items():
        definition = registry.get_definition(node.class_type)
        if definition is None:
            if node.class_type.startswith("Functor:"):
                continue
            definition = registry.build_from_node_spec(
                node.class_type,
                input_names=node.inputs.keys(),
                output_names=node.outputs,
            )

        declared_ports = (
            ("input", node.inputs.keys(), definition.inputs),
            ("output", node.outputs, definition.outputs),
        )
        for direction, port_names, authoritative_ports in declared_ports:
            for port_name in port_names:
                port = authoritative_ports.get(port_name)
                if port is None:
                    continue
                missing_fields = port.missing_support_declarations()
                if missing_fields:
                    issue = GraphValidationIssue(
                        code="publication_port_support_incomplete",
                        message=(
                            f"Publication {direction} port requires explicit "
                            "shape and support declarations."
                        ),
                        node_id=node_id,
                        port=port_name,
                        expected=", ".join(PORT_SUPPORT_FIELDS),
                        got=", ".join(missing_fields),
                        hint=(
                            "Complete the canonical SCM variable contract; "
                            "explicit null is allowed for categorical support "
                            "fields when scientifically appropriate."
                        ),
                    )
                    (errors if strict else warnings).append(issue)

                if "shape" in missing_fields:
                    continue
                shape = port.shape
                dimensions = (
                    shape.get("dimensions")
                    if isinstance(shape, dict)
                    else None
                )
                valid_dimensions = isinstance(
                    dimensions,
                    (list, tuple),
                ) and all(
                    isinstance(dimension, str) and bool(dimension.strip())
                    for dimension in dimensions
                )
                if not valid_dimensions:
                    issue = GraphValidationIssue(
                        code="publication_port_shape_invalid",
                        message=(
                            f"Publication {direction} port shape must declare "
                            "a dimensions sequence of non-empty names."
                        ),
                        node_id=node_id,
                        port=port_name,
                        expected='{"dimensions": ["cell"]} or {"dimensions": []}',
                        got=str(shape),
                        hint=(
                            "Declare the canonical tensor rank and named "
                            "dimensions in the existing SCM variable contract."
                        ),
                    )
                    (errors if strict else warnings).append(issue)
    return errors, warnings


def build_audit_summary(
    report: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    errors = list(getattr(report, "errors", []) or [])
    warnings = list(getattr(report, "warnings", []) or [])
    assertion_audit = getattr(report, "assertion_audit", {}) or {}
    if not isinstance(assertion_audit, dict):
        assertion_audit = {}
    if not assertion_audit:
        assertion_audit = _default_assertion_audit()

    semantic_error_codes = {
        "semantics_mismatch",
        "quantity_mismatch",
        "element_mismatch",
        "standard_name_mismatch",
        "cf_standard_name_mismatch",
        "unit_inconsistent",
        "missing_units",
        "missing_context",
        "datatype_mismatch",
        "publication_port_support_incomplete",
        "publication_port_shape_invalid",
    }
    semantic_issues = errors + warnings
    semantic_ok = not any(
        getattr(issue, "code", "") in semantic_error_codes
        for issue in semantic_issues
    )

    meta = metadata if isinstance(metadata, dict) else {}
    topology_signature = meta.get("topology_signature", {})
    if not isinstance(topology_signature, dict):
        topology_signature = {}
    conservation_vectors = topology_signature.get("conservation_vectors", [])
    if not isinstance(conservation_vectors, list):
        conservation_vectors = []

    audit_summary = {
        "syntax_ok": len(errors) == 0,
        "semantic_ok": semantic_ok,
        "assertion_ok": bool(assertion_audit.get("valid", True)),
        "assertion_audit": assertion_audit,
        "conservation_audit": {
            "source": "topology_signature.left_null_space",
            "enabled": bool(topology_signature),
            "rank": topology_signature.get("rank"),
            "nullity": topology_signature.get("nullity"),
            "betti_number": topology_signature.get("betti_number"),
            "conservation_vector_count": len(conservation_vectors),
        },
        "validation_error_count": len(errors),
        "validation_warning_count": len(warnings),
    }
    audit_key = assertion_audit.get("audit_key")
    structural_result = assertion_audit.get("structural_result")
    if isinstance(audit_key, str) and audit_key and isinstance(structural_result, dict):
        audit_summary[audit_key] = dict(structural_result)
    return audit_summary


def validate_graph(
    graph: GraphIR,
    registry: NodeRegistry,
    unit_converter: Optional[UnitConverter] = None,
    strict: bool = True,
    context: Optional[ConversionContext] = None,
    ssot_catalog: Optional[SSOTCatalog] = None,
) -> ValidationReport:
    report = ValidationReport()
    if unit_converter is None:
        unit_converter = canonical_unit_converter
    # The canonical SSOTCatalog; a catalog that cannot be loaded (for example
    # conflicting declarations) raises instead of silently disabling the
    # catalog-backed dimension checks.
    dimension_catalog = ssot_catalog
    if dimension_catalog is None:
        dimension_catalog = SSOTCatalog.from_parameter_manager()
    assertion_audit, assertion_issues = _validate_assertion_payload(
        graph=graph,
        registry=registry,
        ssot_catalog=dimension_catalog,
    )
    report.assertion_audit = assertion_audit
    report.errors.extend(assertion_issues)
    declared_links = sorted(
        (
            link.source.node,
            link.source.port,
            link.target.node,
            link.target.port,
        )
        for link in graph.links
    )
    input_links = sorted(
        (
            link.source.node,
            link.source.port,
            link.target.node,
            link.target.port,
        )
        for link in graph._infer_links_from_inputs()
    )
    if declared_links != input_links:
        issue = GraphValidationIssue(
            code="link_declaration_mismatch",
            message=(
                "Graph links and node input link declarations do not describe "
                "the same connections."
            ),
            expected=str(input_links),
            got=str(declared_links),
            hint="Make links and inputs[*].link identical before strict validation.",
        )
        if strict:
            report.errors.append(issue)
        else:
            report.warnings.append(issue)

    report.errors.extend(_validate_execution_groups(graph))

    missing_functor_definitions = set()
    if strict:
        for node_id, node in graph.nodes.items():
            if (
                node.class_type.startswith("Functor:")
                and registry.get_definition(node.class_type) is None
            ):
                missing_functor_definitions.add(node.class_type)
                report.errors.append(
                    GraphValidationIssue(
                        code="unregistered_functor_definition",
                        message=(
                            f"Functor node '{node_id}' has no authoritative "
                            "NodeRegistry definition."
                        ),
                        node_id=node_id,
                        hint=(
                            "Register the real Functor instance before strict "
                            "validation; GraphIR declarations are not registry truth."
                        ),
                    )
                )
    contract_errors, contract_warnings = _validate_registered_node_contracts(
        graph,
        registry,
        strict=strict,
    )
    report.errors.extend(contract_errors)
    report.warnings.extend(contract_warnings)
    unit_errors, unit_warnings = _validate_declared_port_unit_contracts(
        graph,
        registry,
        strict=strict,
        ssot_catalog=dimension_catalog,
        unit_converter=unit_converter,
    )
    report.errors.extend(unit_errors)
    report.warnings.extend(unit_warnings)
    support_errors, support_warnings = (
        _validate_publication_port_support_completeness(
            graph,
            registry,
            strict=strict,
        )
    )
    report.errors.extend(support_errors)
    report.warnings.extend(support_warnings)
    output_writers: Dict[str, List[str]] = {}
    for node_id, node in graph.nodes.items():
        output_names = set(node.outputs)
        definition = registry.get_definition(node.class_type)
        if definition is not None:
            output_names.update(definition.outputs.keys())
        for output_name in output_names:
            if output_name.startswith("diagnostics."):
                continue
            output_writers.setdefault(output_name, []).append(node_id)
    for output_name, writer_nodes in sorted(output_writers.items()):
        if len(writer_nodes) < 2:
            continue
        if get_output_overlap_policy(output_name) != OVERLAP_POLICY_MUTUALLY_EXCLUSIVE:
            continue
        report.errors.append(
            GraphValidationIssue(
                code="mutually_exclusive_output_overlap",
                message=(
                    f"Output '{output_name}' is mutually exclusive but is declared by "
                    f"multiple graph nodes: {sorted(writer_nodes)}. Select one closure."
                ),
                port=output_name,
                hint="Remove all but one alternative closure before compile_graph.",
            )
        )
    for link in graph.links:
        source_node_id = link.source.node
        target_node_id = link.target.node
        source_node = graph.nodes.get(source_node_id)
        target_node = graph.nodes.get(target_node_id)

        if source_node is None:
            report.errors.append(
                GraphValidationIssue(
                    code="missing_node",
                    message=f"Source node '{source_node_id}' not found.",
                    node_id=source_node_id,
                )
            )
            continue
        if target_node is None:
            report.errors.append(
                GraphValidationIssue(
                    code="missing_node",
                    message=f"Target node '{target_node_id}' not found.",
                    node_id=target_node_id,
                )
            )
            continue

        declaration_error = False
        if link.source.port not in source_node.outputs:
            report.errors.append(
                GraphValidationIssue(
                    code="undeclared_source_output",
                    message=(
                        f"Link source port '{link.source.port}' is not declared "
                        f"by node '{source_node_id}'."
                    ),
                    node_id=source_node_id,
                    port=link.source.port,
                    hint="Declare the source port in the node outputs list.",
                )
            )
            declaration_error = True
        if link.target.port not in target_node.inputs:
            report.errors.append(
                GraphValidationIssue(
                    code="undeclared_target_input",
                    message=(
                        f"Link target port '{link.target.port}' is not declared "
                        f"by node '{target_node_id}'."
                    ),
                    node_id=target_node_id,
                    port=link.target.port,
                    hint="Declare the target port in the node inputs mapping.",
                )
            )
            declaration_error = True
        if declaration_error:
            continue

        source_def = registry.get_definition(source_node.class_type)
        if source_def is None:
            if strict and source_node.class_type in missing_functor_definitions:
                continue
            source_def = registry.build_from_node_spec(
                source_node.class_type,
                input_names=source_node.inputs.keys(),
                output_names=source_node.outputs,
            )
        target_def = registry.get_definition(target_node.class_type)
        if target_def is None:
            if strict and target_node.class_type in missing_functor_definitions:
                continue
            target_def = registry.build_from_node_spec(
                target_node.class_type,
                input_names=target_node.inputs.keys(),
                output_names=target_node.outputs,
            )

        source_port = source_def.outputs.get(link.source.port)
        if source_port is None and not strict:
            source_port = registry.build_from_node_spec(
                source_node.class_type,
                input_names=[],
                output_names=[link.source.port],
            ).outputs.get(link.source.port)
        target_port = target_def.inputs.get(link.target.port)
        if target_port is None and not strict:
            target_port = registry.build_from_node_spec(
                target_node.class_type,
                input_names=[link.target.port],
                output_names=[],
            ).inputs.get(link.target.port)

        if source_port is None or target_port is None:
            report.errors.append(
                GraphValidationIssue(
                    code="missing_port",
                    message="Port not defined for link.",
                    node_id=target_node_id,
                    port=link.target.port,
                )
            )
            continue

        source_group = source_node.config.get("execution_group")
        target_group = target_node.config.get("execution_group")
        target_metadata = (
            target_def.metadata
            if isinstance(target_def.metadata, dict)
            else {}
        )
        adapter_contract = target_metadata.get("adapter_contract")
        registered_temporal_boundary = (
            isinstance(adapter_contract, dict)
            and adapter_contract.get("adapter_type") == "temporal_resample"
        )
        if (
            _is_valid_execution_group(source_group)
            and _is_valid_execution_group(target_group)
            and source_group != target_group
            and not registered_temporal_boundary
        ):
            group_issue = GraphValidationIssue(
                code="cross_execution_group_raw_dependency",
                message=(
                    "Direct GraphIR dependency crosses execution groups "
                    "without an explicit temporal adapter: "
                    f"'{source_node_id}.{link.source.port}' "
                    f"({source_group}) -> "
                    f"'{target_node_id}.{link.target.port}' "
                    f"({target_group})."
                ),
                node_id=target_node_id,
                port=link.target.port,
                expected=(
                    "same execution_group or registered "
                    "temporal_resample adapter"
                ),
                got=f"{source_group} -> {target_group}",
                hint=(
                    "Insert the existing executable temporal_resample "
                    "adapter at the cadence boundary or align "
                    "execution_group values; do not rely on runtime "
                    "stale-read warnings or auto-promotion."
                ),
            )
            if strict:
                report.errors.append(group_issue)
            else:
                report.warnings.append(group_issue)

        decision = can_connect(
            source_port,
            target_port,
            unit_converter=unit_converter,
            context=context,
            strict=strict,
        )
        for err in decision.errors:
            err.node_id = target_node_id
            err.port = link.target.port
            report.errors.append(err)
        for warn in decision.warnings:
            warn.node_id = target_node_id
            warn.port = link.target.port
            report.warnings.append(warn)

        if decision.requires_adapter and decision.adapter_type:
            adapter_issue = GraphValidationIssue(
                code="adapter_required",
                message="Adapter required for connection.",
                node_id=target_node_id,
                port=link.target.port,
                expected=target_port.units,
                got=source_port.units,
                hint=f"Insert adapter: {decision.adapter_type}",
            )
            explicit_adapters = (
                _normalize_text(str(graph.metadata.get("adapters", "")))
                == "explicit"
            )
            if strict and explicit_adapters:
                report.errors.append(
                    GraphValidationIssue(
                        code="unlowered_explicit_adapter_requirement",
                        message=(
                            "Graph declares explicit adapters but this direct "
                            "connection still requires an adapter."
                        ),
                        node_id=target_node_id,
                        port=link.target.port,
                        expected=decision.adapter_type,
                        hint=(
                            "Insert the executable adapter node so both "
                            "resulting links are independently compatible."
                        ),
                    )
                )
            else:
                report.warnings.append(adapter_issue)

    return report


