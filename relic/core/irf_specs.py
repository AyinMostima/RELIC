"""
IRF InitSpecs/RunSpecs loader and validator for RELIC.

This module provides optional, file-driven contracts that validate
parameter units/bounds, state variable metadata, and update mode rules
before execution. When disabled, existing behavior is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable, cast
import logging
import yaml

from relic.parameters.ops.unit_helpers import normalize_unit_string as _normalize_unit_string

logger = logging.getLogger(__name__)

ALLOWED_UPDATE_MODES = {"accumulate", "replace", "flux"}
ALLOWED_SEMANTIC_MODES = {"mutable", "immutable", "derived"}


@dataclass
class ParameterRequirement:
    key: str
    units: Optional[str] = None
    bounds: Optional[Tuple[Optional[float], Optional[float]]] = None
    required: bool = True
    requires_grad: Optional[bool] = None


@dataclass
class StateVariableRequirement:
    name: str
    units: Optional[str] = None
    bounds: Optional[Tuple[Optional[float], Optional[float]]] = None


@dataclass
class FunctorRequirement:
    name: str
    execution_phase: Optional[str] = None


@dataclass
class InitSpecs:
    parameters: List[ParameterRequirement] = field(default_factory=list)
    state_variables: List[StateVariableRequirement] = field(default_factory=list)
    functors: List[FunctorRequirement] = field(default_factory=list)


@dataclass
class RunSpecs:
    update_rules: Dict[str, str] = field(default_factory=dict)
    default_mode: str = "replace"
    semantic_rules: Dict[str, str] = field(default_factory=dict)
    default_semantic: str = "mutable"
    strict: bool = False
    warnings: List[str] = field(default_factory=list)

    def resolve_update_mode(self, variable: str, requested_mode: str) -> str:
        mode = str(requested_mode).lower()
        if mode not in ALLOWED_UPDATE_MODES:
            msg = f"RunSpecs: invalid update mode '{requested_mode}' for {variable}"
            if self.strict:
                raise ValueError(msg)
            self.warnings.append(msg)
            mode = self.default_mode
        allowed = self.update_rules.get(variable)
        if mode in {"accumulate", "flux"} and allowed != mode:
            msg = f"RunSpecs: {mode} not allowed for {variable}; using {self.default_mode}"
            if self.strict:
                raise ValueError(msg)
            self.warnings.append(msg)
            return self.default_mode
        return mode

    def resolve_semantic_mode(self, variable: str, requested_mode: Optional[str]) -> str:
        mode = str(requested_mode).lower() if requested_mode else self.default_semantic
        if mode not in ALLOWED_SEMANTIC_MODES:
            msg = f"RunSpecs: invalid semantic mode '{requested_mode}' for {variable}"
            if self.strict:
                raise ValueError(msg)
            self.warnings.append(msg)
            mode = self.default_semantic
        allowed = self.semantic_rules.get(variable)
        if allowed and mode != allowed:
            msg = f"RunSpecs: semantic mode '{mode}' not allowed for {variable}; using {allowed}"
            if self.strict:
                raise ValueError(msg)
            self.warnings.append(msg)
            mode = allowed
        return mode


@dataclass
class IrfSpecs:
    schema_version: str
    strict: bool = False
    init_specs: InitSpecs = field(default_factory=InitSpecs)
    run_specs: RunSpecs = field(default_factory=RunSpecs)


@dataclass
class IrfValidationReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def raise_if_strict(self, strict: bool) -> None:
        if strict and self.errors:
            raise ValueError("IRF validation failed: " + "; ".join(self.errors))


def _normalize_units(units: Optional[str]) -> Optional[str]:
    if units is None:
        return None
    return cast(Optional[str], _normalize_unit_string(units))


def _parse_bounds(raw: Any) -> Optional[Tuple[Optional[float], Optional[float]]]:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        lo, hi = raw
        lo_val = float(lo) if lo is not None else None
        hi_val = float(hi) if hi is not None else None
        return (lo_val, hi_val)
    raise ValueError(f"Invalid bounds format: {raw}")


def _parse_parameters(payload: Iterable[Dict[str, Any]]) -> List[ParameterRequirement]:
    params: List[ParameterRequirement] = []
    for entry in payload:
        params.append(
            ParameterRequirement(
                key=str(entry.get("key", "")).strip(),
                units=entry.get("units"),
                bounds=_parse_bounds(entry.get("bounds")) if "bounds" in entry else None,
                required=bool(entry.get("required", True)),
                requires_grad=entry.get("requires_grad"),
            )
        )
    return params


def _parse_state_variables(payload: Iterable[Dict[str, Any]]) -> List[StateVariableRequirement]:
    vars_out: List[StateVariableRequirement] = []
    for entry in payload:
        vars_out.append(
            StateVariableRequirement(
                name=str(entry.get("name", "")).strip(),
                units=entry.get("units"),
                bounds=_parse_bounds(entry.get("bounds")) if "bounds" in entry else None,
            )
        )
    return vars_out


def _parse_functors(payload: Iterable[Dict[str, Any]]) -> List[FunctorRequirement]:
    functors: List[FunctorRequirement] = []
    for entry in payload:
        functors.append(
            FunctorRequirement(
                name=str(entry.get("name", "")).strip(),
                execution_phase=entry.get("execution_phase"),
            )
        )
    return functors


def _parse_run_specs(payload: Dict[str, Any], strict: bool) -> RunSpecs:
    update_rules_raw = payload.get("update_rules", {})
    update_rules: Dict[str, str] = {}
    if isinstance(update_rules_raw, dict):
        for key, value in update_rules_raw.items():
            update_rules[str(key)] = str(value).lower()
    elif isinstance(update_rules_raw, list):
        for entry in update_rules_raw:
            variable = str(entry.get("variable", "")).strip()
            mode = str(entry.get("mode", "")).lower()
            if variable:
                update_rules[variable] = mode
    default_mode = str(payload.get("default_mode", "replace")).lower()
    if default_mode not in ALLOWED_UPDATE_MODES:
        default_mode = "replace"
    semantic_rules_raw = payload.get("semantic_rules", {})
    semantic_rules: Dict[str, str] = {}
    if isinstance(semantic_rules_raw, dict):
        for key, value in semantic_rules_raw.items():
            semantic_rules[str(key)] = str(value).lower()
    elif isinstance(semantic_rules_raw, list):
        for entry in semantic_rules_raw:
            variable = str(entry.get("variable", "")).strip()
            mode = str(entry.get("mode", "")).lower()
            if variable:
                semantic_rules[variable] = mode
    default_semantic = str(payload.get("default_semantic", "mutable")).lower()
    if default_semantic not in ALLOWED_SEMANTIC_MODES:
        default_semantic = "mutable"
    run_specs = RunSpecs(
        update_rules=update_rules,
        default_mode=default_mode,
        semantic_rules=semantic_rules,
        default_semantic=default_semantic,
        strict=bool(payload.get("strict", strict)),
    )
    return run_specs


def load_irf_specs(path: Path) -> IrfSpecs:
    if not path.exists():
        raise FileNotFoundError(f"IRF specs file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    schema_version = str(data.get("schema_version", "1.0.0"))
    strict = bool(data.get("strict", False))
    init_payload = data.get("init_specs", {}) or {}
    run_payload = data.get("run_specs", {}) or {}

    init_specs = InitSpecs(
        parameters=_parse_parameters(init_payload.get("parameters", []) or []),
        state_variables=_parse_state_variables(init_payload.get("state_variables", []) or []),
        functors=_parse_functors(init_payload.get("functors", []) or []),
    )
    run_specs = _parse_run_specs(run_payload, strict=strict)

    return IrfSpecs(
        schema_version=schema_version,
        strict=strict,
        init_specs=init_specs,
        run_specs=run_specs,
    )


def _get_state_schema(state_manager: Any) -> Any:
    schema = getattr(state_manager, "schema", None)
    if callable(schema):
        schema = schema()
    return schema


def _validate_run_specs_defaults(run_specs: RunSpecs, report: IrfValidationReport) -> None:
    if run_specs.default_mode not in ALLOWED_UPDATE_MODES:
        report.errors.append("RunSpecs.default_mode must be 'accumulate', 'replace', or 'flux'")
    if run_specs.default_semantic not in ALLOWED_SEMANTIC_MODES:
        report.errors.append(
            "RunSpecs.default_semantic must be one of 'mutable', 'immutable', 'derived'"
        )


def _validate_update_rules(
    update_rules: Dict[str, str],
    state_manager: Any,
    report: IrfValidationReport,
) -> None:
    for name, mode in update_rules.items():
        if mode not in ALLOWED_UPDATE_MODES:
            report.errors.append(f"RunSpecs.update_rules[{name}] has invalid mode '{mode}'")
        if state_manager is not None:
            schema = _get_state_schema(state_manager)
            if isinstance(schema, dict) and name not in schema:
                report.warnings.append(
                    f"RunSpecs.update_rules refers to unknown state variable '{name}'"
                )


def _validate_semantic_rules(
    semantic_rules: Dict[str, str],
    state_manager: Any,
    report: IrfValidationReport,
) -> None:
    for name, mode in semantic_rules.items():
        if mode not in ALLOWED_SEMANTIC_MODES:
            report.errors.append(f"RunSpecs.semantic_rules[{name}] has invalid mode '{mode}'")
        if state_manager is not None:
            schema = _get_state_schema(state_manager)
            if isinstance(schema, dict) and name not in schema:
                report.warnings.append(
                    f"RunSpecs.semantic_rules refers to unknown state variable '{name}'"
                )


def _validate_init_specs_availability(
    init_specs: InitSpecs,
    parameter_manager: Any,
    state_manager: Any,
    report: IrfValidationReport,
) -> None:
    if init_specs.parameters and parameter_manager is None:
        report.errors.append("InitSpecs.parameters provided but ParameterManager is unavailable")
    if init_specs.state_variables and state_manager is None:
        report.errors.append("InitSpecs.state_variables provided but StateManager is unavailable")


def _validate_parameter_requirements(
    parameter_requirements: Iterable[ParameterRequirement],
    parameter_manager: Any,
    report: IrfValidationReport,
) -> None:
    for parameter_req in parameter_requirements:
        if not parameter_req.key:
            report.errors.append("InitSpecs.parameters entry missing key")
            continue
        if parameter_manager is None:
            continue
        spec = parameter_manager.get_parameter(parameter_req.key)
        if spec is None:
            if parameter_req.required:
                report.errors.append(f"InitSpecs parameter missing: {parameter_req.key}")
            else:
                report.warnings.append(
                    f"InitSpecs optional parameter missing: {parameter_req.key}"
                )
            continue
        expected_units = _normalize_units(parameter_req.units)
        actual_units = _normalize_units(getattr(spec, "units", None))
        if expected_units and actual_units and expected_units != actual_units:
            report.errors.append(
                f"InitSpecs parameter units mismatch for {parameter_req.key}: {actual_units} != {expected_units}"
            )
        if parameter_req.bounds is not None:
            actual_bounds = getattr(spec, "bounds", None)
            if actual_bounds is None:
                report.errors.append(
                    f"InitSpecs parameter bounds missing for {parameter_req.key}"
                )
            else:
                actual_pair = _parse_bounds(actual_bounds)
                if actual_pair != parameter_req.bounds:
                    report.errors.append(
                        f"InitSpecs parameter bounds mismatch for {parameter_req.key}: {actual_pair} != {parameter_req.bounds}"
                    )
        if parameter_req.requires_grad is not None:
            if bool(getattr(spec, "requires_grad", False)) != bool(parameter_req.requires_grad):
                report.warnings.append(
                    f"InitSpecs requires_grad mismatch for {parameter_req.key}: "
                    f"{getattr(spec, 'requires_grad', False)} != {parameter_req.requires_grad}"
                )
            if (
                parameter_req.requires_grad
                and parameter_req.key not in getattr(parameter_manager, "nn_parameters", {})
            ):
                report.warnings.append(
                    f"InitSpecs requires_grad missing nn.Parameter for {parameter_req.key}"
                )


def _validate_state_variable_requirements(
    state_requirements: Iterable[StateVariableRequirement],
    state_manager: Any,
    report: IrfValidationReport,
) -> None:
    for state_req in state_requirements:
        if not state_req.name:
            report.errors.append("InitSpecs.state_variables entry missing name")
            continue
        if state_manager is None:
            continue
        schema = _get_state_schema(state_manager)
        if not isinstance(schema, dict) or state_req.name not in schema:
            report.errors.append(f"InitSpecs state variable missing: {state_req.name}")
            continue
        spec = schema[state_req.name]
        expected_units = _normalize_units(state_req.units)
        actual_units = _normalize_units(getattr(spec, "units", None))
        if expected_units and actual_units and expected_units != actual_units:
            report.errors.append(
                f"InitSpecs state units mismatch for {state_req.name}: {actual_units} != {expected_units}"
            )
        if state_req.bounds is not None:
            actual_bounds = getattr(spec, "bounds", None)
            if actual_bounds is None:
                report.errors.append(f"InitSpecs state bounds missing for {state_req.name}")
            else:
                actual_pair = _parse_bounds(actual_bounds)
                if actual_pair != state_req.bounds:
                    report.errors.append(
                        f"InitSpecs state bounds mismatch for {state_req.name}: {actual_pair} != {state_req.bounds}"
                    )


def _validate_functor_requirements(
    functor_requirements: Iterable[FunctorRequirement],
    functors: Optional[Dict[str, Any]],
    report: IrfValidationReport,
) -> None:
    for functor_req in functor_requirements:
        if not functor_req.name:
            report.errors.append("InitSpecs.functors entry missing name")
            continue
        if functors is None:
            report.warnings.append(
                f"InitSpecs functor not checked (missing registry): {functor_req.name}"
            )
            continue
        functor = functors.get(functor_req.name)
        if functor is None:
            report.errors.append(f"InitSpecs functor missing: {functor_req.name}")
            continue
        if functor_req.execution_phase:
            metadata = getattr(functor, "metadata", None)
            phase = getattr(metadata, "execution_phase", None)
            if phase is not None:
                phase_value = phase.value if hasattr(phase, "value") else str(phase)
                if str(functor_req.execution_phase) != phase_value:
                    report.errors.append(
                        f"InitSpecs functor phase mismatch for {functor_req.name}: {phase_value} != {functor_req.execution_phase}"
                    )
            else:
                report.warnings.append(
                    f"InitSpecs functor missing metadata: {functor_req.name}"
                )


def validate_irf_specs(
    irf_specs: IrfSpecs,
    parameter_manager: Any = None,
    state_manager: Any = None,
    functors: Optional[Dict[str, Any]] = None,
    strict: Optional[bool] = None,
) -> IrfValidationReport:
    report = IrfValidationReport()
    enforce_strict = irf_specs.strict if strict is None else strict

    _validate_run_specs_defaults(irf_specs.run_specs, report)
    _validate_update_rules(irf_specs.run_specs.update_rules, state_manager, report)
    _validate_semantic_rules(irf_specs.run_specs.semantic_rules, state_manager, report)
    _validate_init_specs_availability(
        irf_specs.init_specs,
        parameter_manager,
        state_manager,
        report,
    )
    _validate_parameter_requirements(irf_specs.init_specs.parameters, parameter_manager, report)
    _validate_state_variable_requirements(irf_specs.init_specs.state_variables, state_manager, report)
    _validate_functor_requirements(irf_specs.init_specs.functors, functors, report)

    report.raise_if_strict(enforce_strict)
    return report
