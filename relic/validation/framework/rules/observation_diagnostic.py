"""Validation rules for observation operators and diagnostic outputs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ....computation.operators.dec.forms import DiagnosticSpec, ObservationOperatorSpec
from ....data.standards.dimensionality import dimensionality_from_unit
from ..decorators import validated_rule
from ..models import (
    ValidationCode,
    ValidationContext,
    ValidationDetail,
    ValidationLevel,
    ValidationResultModel,
    ValidationResults,
)


def _iter_specs(payload: Any) -> Iterable[Any]:
    if payload is None:
        return ()
    if isinstance(payload, (list, tuple)):
        return payload
    return (payload,)


def _normalize_spec_container(
    payload: Any,
    *,
    label: str,
    accepted_type: type,
) -> tuple[tuple[Any, ...] | None, str | None]:
    if payload is None:
        return (), None
    if not isinstance(payload, (list, tuple)):
        return None, f"{label} must be a list or tuple."
    specs = tuple(payload)
    for index, spec in enumerate(specs):
        if not isinstance(spec, (Mapping, accepted_type)):
            return (
                None,
                f"{label}[{index}] must be {accepted_type.__name__} or Mapping.",
            )
    return specs, None


def _get(spec: Any, key: str, default: Any = None) -> Any:
    if isinstance(spec, Mapping):
        return spec.get(key, default)
    return getattr(spec, key, default)


def _metadata(spec: Any) -> Mapping[str, Any]:
    metadata = _get(spec, "metadata", {})
    return metadata if isinstance(metadata, Mapping) else {}


def _failure(name: str, message: str) -> ValidationResultModel:
    return ValidationResultModel(
        level=ValidationLevel.ERROR,
        code=ValidationCode.GENERIC_VALIDATION_FAILED,
        message=message,
        passed=False,
        layer="L1",
        scope="observation_diagnostic_structure",
        tags=["observation", "diagnostic", "structural", "assimilation"],
        details=[ValidationDetail(variable_name=name, value=message)],
    )


def _strict_identities(value: Any, *, label: str) -> tuple[set[str] | None, str | None]:
    identities: list[str] = []
    for item in _iter_specs(value):
        if not isinstance(item, str) or not item or item != item.strip():
            return None, f"{label} must contain non-empty exact string identities."
        identities.append(item)
    if len(identities) != len(set(identities)):
        return None, f"{label} must not contain duplicate identities."
    return set(identities), None


def _diagnostic_output_names(spec: Any) -> tuple[str, ...]:
    explicit = _get(spec, "output_name")
    if explicit is not None:
        return (explicit.strip(),) if isinstance(explicit, str) and explicit.strip() else ()
    outputs = _metadata(spec).get("outputs", ())
    return tuple(
        item.strip()
        for item in _iter_specs(outputs)
        if isinstance(item, str) and item.strip()
    )


def validate_observation_diagnostic_payload(
    observations: Any,
    diagnostics: Any,
    *,
    strict: bool = False,
    required_kinds: Iterable[str] = (),
    available_state_vars: Iterable[str] = (),
    required_observation_state_vars: Iterable[str] = (),
    spec_node_bindings: Mapping[str, Mapping[str, Any]] | None = None,
) -> ValidationResultModel:
    """Validate structural observation mappings and read-only diagnostics.

    Evidence boundary:
      Graph bindings prove declared node/port identity only. Runtime forward
      execution, operator provenance, support correctness, and assimilation remain [N].
    """

    if not isinstance(strict, bool):
        return _failure("strict", "strict must be bool.")
    observation_specs, observation_error = _normalize_spec_container(
        observations,
        label="observations",
        accepted_type=ObservationOperatorSpec,
    )
    if observation_error:
        return _failure("observations", observation_error)
    diagnostic_specs, diagnostic_error = _normalize_spec_container(
        diagnostics,
        label="diagnostics",
        accepted_type=DiagnosticSpec,
    )
    if diagnostic_error:
        return _failure("diagnostics", diagnostic_error)
    assert observation_specs is not None
    assert diagnostic_specs is not None
    if strict:
        required_identities, required_error = _strict_identities(
            required_kinds,
            label="required_kinds",
        )
        if required_error:
            return _failure("required_kinds", required_error)
        required = {
            item.lower() for item in (required_identities or set())
        }
    else:
        required = {
            str(item).strip().lower()
            for item in _iter_specs(required_kinds)
            if str(item).strip()
        }
    allowed_kinds = {"observation", "diagnostic"}
    invalid_kinds = sorted(required.difference(allowed_kinds))
    if invalid_kinds:
        return _failure(
            "required_kinds",
            "Observation contract contains unrecognized required kinds: "
            + ",".join(invalid_kinds),
        )
    if strict and not observation_specs and not diagnostic_specs:
        return _failure(
            "observation_diagnostic",
            "Strict observation/diagnostic validation requires a non-empty payload.",
        )
    if "observation" in required and not observation_specs:
        return _failure(
            "observation",
            "Strict observation declaration count must be greater than zero.",
        )
    if "diagnostic" in required and not diagnostic_specs:
        return _failure(
            "diagnostic",
            "Strict diagnostic declaration count must be greater than zero.",
        )

    checked = 0
    all_specs = (*observation_specs, *diagnostic_specs)
    spec_names: list[str] = []
    for spec in all_specs:
        raw_name = _get(spec, "name")
        if strict and (
            not isinstance(raw_name, str)
            or not raw_name
            or raw_name != raw_name.strip()
        ):
            return _failure(
                "spec_name",
                "Strict observation/diagnostic specs require non-empty exact names.",
            )
        spec_names.append(str(raw_name or ""))
    if strict and len(spec_names) != len(set(spec_names)):
        return _failure(
            "spec_name",
            "Strict observation/diagnostic spec names must be unique.",
        )

    verified_bindings: dict[str, tuple[str, set[str], set[str]]] = {}
    node_binding_views: dict[str, tuple[set[str], set[str]]] = {}
    if strict and spec_node_bindings is not None:
        if not isinstance(spec_node_bindings, Mapping):
            return _failure(
                "spec_node_bindings",
                "Strict spec_node_bindings must be a Mapping.",
            )
        binding_keys, binding_error = _strict_identities(
            tuple(spec_node_bindings),
            label="spec_node_bindings keys",
        )
        if binding_error:
            return _failure("spec_node_bindings", binding_error)
        expected_keys = set(spec_names)
        if binding_keys != expected_keys:
            missing = sorted(expected_keys.difference(binding_keys or set()))
            extra = sorted((binding_keys or set()).difference(expected_keys))
            return _failure(
                "spec_node_bindings",
                "Strict graph bindings must cover exactly the declared specs; "
                f"missing={missing}, extra={extra}.",
            )
        for spec_name in spec_names:
            binding = spec_node_bindings.get(spec_name)
            if not isinstance(binding, Mapping):
                return _failure(
                    spec_name,
                    f"Graph binding for '{spec_name}' must be a mapping.",
                )
            node_id = binding.get("node_id")
            if (
                not isinstance(node_id, str)
                or not node_id
                or node_id != node_id.strip()
            ):
                return _failure(
                    spec_name,
                    f"Graph binding for '{spec_name}' requires an exact node_id.",
                )
            input_ports, input_error = _strict_identities(
                binding.get("input_ports", ()),
                label=f"Graph node '{node_id}' input ports",
            )
            if input_error:
                return _failure(spec_name, input_error)
            output_ports, output_error = _strict_identities(
                binding.get("output_ports", ()),
                label=f"Graph node '{node_id}' output ports",
            )
            if output_error:
                return _failure(spec_name, output_error)
            if not output_ports:
                return _failure(
                    spec_name,
                    f"Graph node '{node_id}' output ports must be non-empty.",
                )
            input_set = input_ports or set()
            output_set = output_ports or set()
            previous_view = node_binding_views.get(node_id)
            if previous_view is not None and previous_view != (
                input_set,
                output_set,
            ):
                return _failure(
                    spec_name,
                    f"Graph specs sharing node '{node_id}' require consistent "
                    "input/output port sets.",
                )
            node_binding_views[node_id] = (input_set, output_set)
            verified_bindings[spec_name] = (node_id, input_set, output_set)

    if strict:
        available, available_error = _strict_identities(
            available_state_vars,
            label="available_state_vars",
        )
        if available_error:
            return _failure("available_state_vars", available_error)
        required_observation_states, coverage_error = _strict_identities(
            required_observation_state_vars,
            label="required_observation_state_vars",
        )
        if coverage_error:
            return _failure("required_observation_state_vars", coverage_error)
        available = available or set()
        required_observation_states = required_observation_states or set()
    else:
        available = {
            str(item).strip()
            for item in _iter_specs(available_state_vars)
            if str(item).strip()
        }
        required_observation_states = {
            str(item).strip()
            for item in _iter_specs(required_observation_state_vars)
            if str(item).strip()
        }
    covered_observation_states: set[str] = set()
    output_names: set[str] = set()
    declared_output_names = {
        raw_output.strip()
        for spec in observation_specs
        for raw_output in (_get(spec, "output_name", ""),)
        if isinstance(raw_output, str) and raw_output.strip()
    }
    for spec in diagnostic_specs:
        declared_output_names.update(_diagnostic_output_names(spec))
    allowed_assimilation_modes = {"conservative_projection", "port_accounting"}
    declared_outputs_by_node: dict[str, set[str]] = {}
    for spec in observation_specs:
        name = _get(spec, "name", "observation")
        checked += 1
        raw_metadata = _get(spec, "metadata", {})
        if strict and not isinstance(raw_metadata, Mapping):
            return _failure(
                name,
                f"Observation '{name}' metadata must be a Mapping.",
            )
        conservative_required = _get(
            spec,
            "conservative_projection_required",
            False,
        )
        if strict and not isinstance(conservative_required, bool):
            return _failure(
                name,
                f"Observation '{name}' conservative_projection_required must be bool.",
            )
        if strict:
            input_state_vars, input_error = _strict_identities(
                _get(spec, "input_state_vars"),
                label=f"Observation '{name}' input_state_vars",
            )
            if input_error:
                return _failure(name, input_error)
            input_state_vars = input_state_vars or set()
        else:
            input_state_vars = {
                str(item).strip()
                for item in _iter_specs(_get(spec, "input_state_vars"))
                if str(item).strip()
            }
        if not input_state_vars:
            return _failure(name, f"Observation '{name}' must declare input_state_vars.")
        if strict and not available:
            return _failure(
                name,
                f"Observation '{name}' requires available_state_vars coverage evidence.",
            )
        unavailable = sorted(input_state_vars.difference(available)) if strict else []
        if unavailable:
            return _failure(
                name,
                f"Observation '{name}' references unavailable state variables: "
                + ",".join(unavailable),
            )
        self_references = sorted(input_state_vars.intersection(declared_output_names))
        if strict and self_references:
            return _failure(
                name,
                f"Observation '{name}' cannot use its own output or another "
                "declared observation/diagnostic output as state input: "
                + ",".join(self_references),
            )
        covered_observation_states.update(input_state_vars)
        for field_name in (
            "output_name",
            "output_units",
            "support",
            "averaging_window",
            "footprint_description",
            "error_model",
        ):
            field_value = _get(spec, field_name)
            if strict and (
                not isinstance(field_value, str)
                or not field_value
                or field_value != field_value.strip()
            ):
                return _failure(
                    name,
                    f"Observation '{name}' {field_name} must be a non-empty exact string.",
                )
            if not field_value:
                return _failure(
                    name,
                    f"Observation '{name}' must declare {field_name}.",
                )
        output_name = _get(spec, "output_name", "")
        output_units = _get(spec, "output_units", "")
        if strict and dimensionality_from_unit(output_units) is None:
            return _failure(
                name,
                f"Observation '{name}' requires parseable output_units.",
            )
        if strict and output_name in output_names:
            return _failure(
                name,
                f"Observation output_name '{output_name}' must be unique.",
            )
        output_names.add(output_name)
        if strict and name in verified_bindings:
            node_id, node_inputs, node_outputs = verified_bindings[name]
            if input_state_vars != node_inputs:
                return _failure(
                    name,
                    f"Observation '{name}' input_state_vars must equal its scoped "
                    f"Graph node input ports; declared={sorted(input_state_vars)}, "
                    f"scoped={sorted(node_inputs)}.",
                )
            if output_name not in node_outputs:
                return _failure(
                    name,
                    f"Observation '{name}' output_name must belong to its scoped "
                    f"Graph node output ports; output={output_name!r}, "
                    f"scoped={sorted(node_outputs)}.",
                )
            declared_outputs_by_node.setdefault(node_id, set()).add(output_name)
        mode = _metadata(spec).get("assimilation_mode")
        if strict and mode is not None and mode not in allowed_assimilation_modes:
            return _failure(
                name,
                f"Observation '{name}' assimilation_mode must be conservative_projection "
                "or port_accounting when declared.",
            )
        if conservative_required is True:
            if mode not in allowed_assimilation_modes:
                return _failure(
                    name,
                    f"Observation '{name}' requires conservative projection or port accounting.",
                )

    for spec in diagnostic_specs:
        name = _get(spec, "name", "diagnostic")
        checked += 1
        raw_metadata = _get(spec, "metadata", {})
        if strict and not isinstance(raw_metadata, Mapping):
            return _failure(
                name,
                f"Diagnostic '{name}' metadata must be a Mapping.",
            )
        writes_state = _get(spec, "writes_state", False)
        if strict and not isinstance(writes_state, bool):
            return _failure(
                name,
                f"Diagnostic '{name}' writes_state must be bool.",
            )
        feedback_port = _get(spec, "feedback_port")
        if strict and feedback_port is not None and (
            not isinstance(feedback_port, str)
            or not feedback_port
            or feedback_port != feedback_port.strip()
        ):
            return _failure(
                name,
                f"Diagnostic '{name}' feedback_port must be an exact string when declared.",
            )
        if strict:
            input_state_vars, input_error = _strict_identities(
                _get(spec, "input_state_vars"),
                label=f"Diagnostic '{name}' input_state_vars",
            )
            if input_error:
                return _failure(name, input_error)
            input_state_vars = input_state_vars or set()
        else:
            input_state_vars = {
                str(item).strip()
                for item in _iter_specs(_get(spec, "input_state_vars"))
                if str(item).strip()
            }
        if not input_state_vars:
            return _failure(name, f"Diagnostic '{name}' must declare input_state_vars.")
        if strict and not available:
            return _failure(
                name,
                f"Diagnostic '{name}' requires available_state_vars coverage evidence.",
            )
        unavailable = sorted(input_state_vars.difference(available)) if strict else []
        if unavailable:
            return _failure(
                name,
                f"Diagnostic '{name}' references unavailable state variables: "
                + ",".join(unavailable),
            )
        self_references = sorted(input_state_vars.intersection(declared_output_names))
        if strict and self_references:
            return _failure(
                name,
                f"Diagnostic '{name}' cannot use its own output or another "
                "declared observation/diagnostic output as state input: "
                + ",".join(self_references),
            )
        if strict:
            for field_name in ("output_units", "diagnostic_kind"):
                field_value = _get(spec, field_name)
                if (
                    not isinstance(field_value, str)
                    or not field_value
                    or field_value != field_value.strip()
                ):
                    return _failure(
                        name,
                        f"Diagnostic '{name}' {field_name} must be a non-empty exact string.",
                    )
            explicit_output = _get(spec, "output_name")
            if explicit_output is not None and (
                not isinstance(explicit_output, str)
                or not explicit_output
                or explicit_output != explicit_output.strip()
            ):
                return _failure(
                    name,
                    f"Diagnostic '{name}' output_name must be a non-empty exact string.",
                )
            if explicit_output is None and "outputs" in raw_metadata:
                _, output_identity_error = _strict_identities(
                    raw_metadata.get("outputs"),
                    label=f"Diagnostic '{name}' metadata.outputs",
                )
                if output_identity_error:
                    return _failure(name, output_identity_error)
            diagnostic_outputs = _diagnostic_output_names(spec)
            if not diagnostic_outputs:
                return _failure(
                    name,
                    f"Diagnostic '{name}' must declare output_name or metadata.outputs.",
                )
            output_units = _get(spec, "output_units", "")
            if dimensionality_from_unit(output_units) is None:
                return _failure(
                    name,
                    f"Diagnostic '{name}' requires parseable output_units.",
                )
            for output_name in diagnostic_outputs:
                if output_name in output_names:
                    return _failure(
                        name,
                        f"Diagnostic output_name '{output_name}' must be unique.",
                    )
                output_names.add(output_name)
            if name in verified_bindings:
                node_id, node_inputs, node_outputs = verified_bindings[name]
                unavailable_inputs = sorted(
                    input_state_vars.difference(node_inputs.union(node_outputs))
                )
                if unavailable_inputs:
                    return _failure(
                        name,
                        f"Diagnostic '{name}' inputs must belong to its scoped Graph "
                        f"node input or output ports: {unavailable_inputs}.",
                    )
                outside_outputs = sorted(set(diagnostic_outputs).difference(node_outputs))
                if outside_outputs:
                    return _failure(
                        name,
                        f"Diagnostic '{name}' outputs must belong to its scoped Graph "
                        f"node output ports: {outside_outputs}.",
                    )
                declared_outputs_by_node.setdefault(node_id, set()).update(
                    diagnostic_outputs
                )
                if feedback_port is not None and feedback_port not in node_inputs.union(
                    node_outputs
                ):
                    return _failure(
                        name,
                        f"Diagnostic '{name}' feedback_port must belong to its scoped "
                        "Graph node ports.",
                    )
            if writes_state is True:
                return _failure(
                    name,
                    f"Diagnostic '{name}' strict diagnostics are read-only; "
                    "feedback_port cannot authorize state writes.",
                )
        if writes_state is True and not feedback_port:
            return _failure(
                name,
                f"Diagnostic '{name}' writes state without a feedback port.",
            )

    if strict and spec_node_bindings is None:
        return _failure(
            "spec_node_bindings",
            "Strict observation/diagnostic spec_node_bindings are required; "
            "runtime observation execution remains [N].",
        )
    if strict:
        for node_id, (_, scoped_outputs) in node_binding_views.items():
            declared_outputs = declared_outputs_by_node.get(node_id, set())
            if declared_outputs != scoped_outputs:
                return _failure(
                    node_id,
                    f"Graph node '{node_id}' declared output union must exactly equal "
                    f"scoped output ports; declared={sorted(declared_outputs)}, "
                    f"scoped={sorted(scoped_outputs)}.",
                )

    missing_coverage = sorted(
        required_observation_states.difference(covered_observation_states)
    )
    if strict and missing_coverage:
        return _failure(
            "observation_coverage",
            "Strict observation coverage is missing state variables: "
            + ",".join(missing_coverage),
        )

    return ValidationResultModel(
        level=ValidationLevel.INFO,
        code=ValidationCode.GENERIC_VALIDATION_OK,
        message="Observation and diagnostic structural contracts passed.",
        passed=True,
        layer="L1",
        scope="observation_diagnostic_structure",
        tags=["observation", "diagnostic", "structural", "assimilation"],
        metrics={
            "checked_specs": checked,
            "observation_count": len(observation_specs),
            "diagnostic_count": len(diagnostic_specs),
            "covered_observation_state_vars": len(covered_observation_states),
            "graph_binding_verified": int(strict and spec_node_bindings is not None),
            "required_observation_state_count": len(required_observation_states),
        },
    )


@validated_rule(
    "observation_diagnostic_structure",
    "L1",
    name="Observation and diagnostic structure",
    description="Validate observation operators, conservative assimilation declarations, and diagnostic read-only behavior.",
    domains=["computation", "data"],
    tags=["observation", "diagnostic", "structural", "assimilation"],
    references=[
        "RELIC-IPHS-DEC Typed Cochain Morphism Enhancement Plan",
        "biogeochemical systems engineering principles: observation is not state",
    ],
)
def validate_observation_diagnostic_structure(context: ValidationContext) -> ValidationResults:
    """Validate observation and diagnostic specs from ValidationContext metadata."""

    metadata = context.metadata if isinstance(context.metadata, dict) else {}
    singular_aliases = sorted(
        key for key in ("observation_spec", "diagnostic_spec") if key in metadata
    )
    if singular_aliases:
        return [
            _failure(
                "observation_diagnostic",
                "Use plural canonical observation_specs/diagnostic_specs only; "
                f"noncanonical={singular_aliases}.",
            )
        ]
    observations = metadata.get("observation_specs", ())
    diagnostics = metadata.get("diagnostic_specs", ())
    available_state_vars = metadata.get("available_state_vars")
    if available_state_vars is None:
        available_state_vars = tuple(context.state) if isinstance(context.state, dict) else ()
    return [
        validate_observation_diagnostic_payload(
            observations,
            diagnostics,
            strict=metadata.get("strict") is True,
            required_kinds=_iter_specs(
                metadata.get("required_observation_kinds")
            ),
            available_state_vars=_iter_specs(available_state_vars),
            required_observation_state_vars=_iter_specs(
                metadata.get("required_observation_state_vars")
            ),
            spec_node_bindings=metadata.get("spec_node_bindings"),
        )
    ]


__all__ = [
    "validate_observation_diagnostic_payload",
    "validate_observation_diagnostic_structure",
]
