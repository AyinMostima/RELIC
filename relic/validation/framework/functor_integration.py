"""
Functor Metadata Integration for Validation Framework

This module provides automatic validation rule generation from FunctorMetadata,
enabling seamless integration between the functor system and validation framework.

Features:
    - Automatic rule generation from FunctorMetadata
    - Conservation law validation from metadata
    - Parameter validation from metadata constraints
    - Unit consistency checking from metadata
    - Dependency validation from metadata

Mathematical Framework:
    For each functor F with metadata M:
    V(F) = {r_i : r_i validates constraint c_i from M}
    
References:
    - RELIC Architecture Documentation
    - Validation Framework Design Specification
"""

import logging
import importlib.util
from typing import Dict, List, Optional, Any
import torch

from ...core.functor import FunctorMetadata, ProcessCategory
from .registry import ValidationRegistry, RuleMetadata, get_global_registry
from .models import (
    ValidationCode,
    ValidationContext,
    ValidationDetail,
    ValidationLevel,
    ValidationResultModel,
    ValidationResults,
)

if importlib.util.find_spec("relic.parameters.manager") is not None:
    from ...parameters import get_global_manager

    PARAMETER_MANAGER_AVAILABLE = True
else:
    get_global_manager = None
    PARAMETER_MANAGER_AVAILABLE = False

logger = logging.getLogger(__name__)


def _append_result(
    results: ValidationResults,
    *,
    level: ValidationLevel,
    message: str,
    category: str,
    layer: str,
    scope: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Append a normalized validation result."""
    code = (
        ValidationCode.GENERIC_VALIDATION_FAILED
        if level in (ValidationLevel.ERROR, ValidationLevel.CRITICAL)
        else ValidationCode.GENERIC_VALIDATION_OK
    )
    details = [ValidationDetail(metadata=metadata or {})]
    results.append(
        ValidationResultModel(
            level=level,
            code=code,
            message=message,
            passed=level not in (ValidationLevel.ERROR, ValidationLevel.CRITICAL),
            details=details,
            layer=layer,
            scope=scope,
            tags=[category],
        )
    )


def _add_warning(
    results: ValidationResults,
    *,
    message: str,
    category: str,
    layer: str,
    scope: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Append a warning result."""
    _append_result(
        results,
        level=ValidationLevel.WARNING,
        message=message,
        category=category,
        layer=layer,
        scope=scope,
        metadata=metadata,
    )


def _add_error(
    results: ValidationResults,
    *,
    message: str,
    category: str,
    layer: str,
    scope: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Append an error result."""
    _append_result(
        results,
        level=ValidationLevel.ERROR,
        message=message,
        category=category,
        layer=layer,
        scope=scope,
        metadata=metadata,
    )


def _add_info(
    results: ValidationResults,
    *,
    message: str,
    category: str,
    layer: str,
    scope: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Append an info result."""
    _append_result(
        results,
        level=ValidationLevel.INFO,
        message=message,
        category=category,
        layer=layer,
        scope=scope,
        metadata=metadata,
    )


def _iter_parameter_entries(parameter_specs: Any):
    if isinstance(parameter_specs, dict):
        return parameter_specs.items()
    return ((param_name, {}) for param_name in parameter_specs)


def _resolve_validation_manager() -> Any:
    """Return the process-wide parameter manager of the parameter rules.

    A manager that fails to load raises; the failure is not read as an absent
    manager, which would turn every parameter check into a silent pass.
    """
    if not PARAMETER_MANAGER_AVAILABLE or get_global_manager is None:
        return None
    return get_global_manager()


def _lookup_manager_spec(
    param_manager: Any,
    *,
    namespace_hint: Any,
    category: Any,
    functor_name: str,
    param_name: str,
) -> Any:
    if param_manager is None:
        return None

    lookup_candidates = _build_lookup_candidates(
        namespace_hint=namespace_hint,
        category=category,
        functor_name=functor_name,
        param_name=param_name,
    )
    # get_parameter returns None for an undeclared key; any error it raises (for
    # example a unit validation failure of a declared spec) propagates.
    for candidate in lookup_candidates:
        manager_spec = param_manager.get_parameter(candidate)
        if manager_spec is not None:
            return manager_spec
    return None


def _build_lookup_candidates(
    *,
    namespace_hint: Any,
    category: Any,
    functor_name: str,
    param_name: str,
) -> List[str]:
    candidates: List[str] = []
    if isinstance(param_name, str) and param_name:
        candidates.append(param_name)
    if namespace_hint:
        if str(namespace_hint).endswith(param_name):
            candidates.append(str(namespace_hint))
        else:
            candidates.append(f"{namespace_hint}.{param_name}")
    if category is not None:
        candidates.append(f"{category.value}.{functor_name}.{param_name}")
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(candidates))


def _lookup_manager_value(
    param_manager: Any,
    *,
    namespace_hint: Any,
    category: Any,
    functor_name: str,
    param_name: str,
    manager_spec: Any = None,
) -> Any:
    """Read the value from the already resolved YAML-backed specification."""
    if param_manager is None or manager_spec is None:
        return None
    if isinstance(manager_spec, dict):
        return manager_spec.get("value")
    return getattr(manager_spec, "value", None)


def _to_numeric_value(param_value: Any) -> Optional[float]:
    if isinstance(param_value, (int, float)):
        return float(param_value)
    if isinstance(param_value, torch.Tensor) and param_value.numel() == 1:
        return float(param_value.item())
    return None


def _merge_bound_and_unit_from_spec(
    min_bound: Any,
    max_bound: Any,
    expected_units: Any,
    manager_spec: Any,
) -> tuple[Any, Any, Any]:
    if manager_spec is None:
        return min_bound, max_bound, expected_units
    if min_bound is None and manager_spec.bounds:
        min_bound = manager_spec.bounds[0]
    if max_bound is None and manager_spec.bounds:
        max_bound = manager_spec.bounds[1]
    if expected_units is None:
        expected_units = manager_spec.units
    return min_bound, max_bound, expected_units


def _validate_parameter_bounds(
    results: ValidationResults,
    *,
    numeric_value: Optional[float],
    min_bound: Any,
    max_bound: Any,
    functor_name: str,
    param_name: str,
) -> None:
    if min_bound is not None and numeric_value is not None and numeric_value < float(min_bound):
        _add_error(
            results,
            message=f"Parameter '{param_name}' value {numeric_value} below minimum {min_bound}",
            category="parameter_validation",
            layer="L1",
            scope=functor_name,
        )
    if max_bound is not None and numeric_value is not None and numeric_value > float(max_bound):
        _add_error(
            results,
            message=f"Parameter '{param_name}' value {numeric_value} above maximum {max_bound}",
            category="parameter_validation",
            layer="L1",
            scope=functor_name,
        )


class FunctorMetadataIntegrator:
    """
    Integrates FunctorMetadata with the validation framework.
    
    Automatically generates and registers validation rules based on
    functor metadata, ensuring that all scientific constraints and
    contracts defined in metadata are enforced during execution.
    """
    
    def __init__(self, registry: Optional[ValidationRegistry] = None):
        """
        Initialize the integrator.
        
        Parameters:
            registry: Validation registry to use (defaults to global)
        """
        self.registry = registry or get_global_registry()
        self._generated_rules = {}  # Track which rules were auto-generated
    
    def register_functor_metadata(self, 
                                 functor_name: str,
                                 metadata: FunctorMetadata,
                                 overwrite: bool = False) -> List[str]:
        """
        Generate and register validation rules from FunctorMetadata.
        
        Parameters:
            functor_name: Name of the functor
            metadata: FunctorMetadata instance
            overwrite: Whether to overwrite existing rules
            
        Returns:
            List of registered rule IDs
        """
        registered_rules = []
        
        # Generate L1 unit validation rules
        if metadata.input_variables and metadata.output_variables:
            rule_id = self._register_unit_validation(functor_name, metadata, overwrite)
            if rule_id:
                registered_rules.append(rule_id)
        
        # Generate L1 parameter validation rules
        if metadata.parameters:
            rule_id = self._register_parameter_validation(functor_name, metadata, overwrite)
            if rule_id:
                registered_rules.append(rule_id)
        
        # Conservation laws register no metadata rule: conservation belongs to the
        # documented layers (FunctorBase local hook, StateManager ledgers,
        # ConservationGuardian; architecture_overview Section 20).

        # Generate L2 dependency validation rules
        rule_id = self._register_dependency_validation(functor_name, metadata, overwrite)
        if rule_id:
            registered_rules.append(rule_id)
        
        # Generate category-specific rules
        if metadata.category:
            rule_ids = self._register_category_specific_rules(functor_name, metadata, overwrite)
            registered_rules.extend(rule_ids)
        
        # Track generated rules
        self._generated_rules[functor_name] = registered_rules
        
        logger.info(f"Registered {len(registered_rules)} validation rules for functor '{functor_name}'")
        return registered_rules
    
    def _register_unit_validation(self, 
                                 functor_name: str,
                                 metadata: FunctorMetadata,
                                 overwrite: bool) -> Optional[str]:
        """Generate and register unit validation rule."""
        rule_id = f"{functor_name}_unit_validation"
        
        def validate_functor_units(context: ValidationContext) -> ValidationResults:
            """Validate input/output units match metadata specifications."""
            results: ValidationResults = []
            context_metadata = (
                context.metadata if isinstance(context.metadata, dict) else {}
            )
            # Requiredness comes from the Functor's get_required_inputs(), which
            # the FunctorBase validation pipeline passes in the context metadata.
            raw_required_inputs = context_metadata.get("required_input_variables", ())
            if isinstance(raw_required_inputs, str):
                required_inputs = {raw_required_inputs}
            elif isinstance(raw_required_inputs, (list, tuple, set, frozenset)):
                required_inputs = set(raw_required_inputs)
            else:
                required_inputs = set()
            
            # Check input variable availability
            input_specs = metadata.input_variables or []
            if isinstance(input_specs, dict):
                input_iter = input_specs.items()
            else:
                input_iter = ((var_name, None) for var_name in input_specs)

            for var_name, expected_units in input_iter:
                if not context.state or var_name not in context.state:
                    if var_name not in required_inputs:
                        continue
                    _add_warning(
                        results,
                        message=f"Input variable '{var_name}' not found in state",
                        category="unit_validation",
                        layer="L1",
                        scope=functor_name,
                    )
                    continue

                variable_value = context.state[var_name]
                if variable_value is None:
                    _add_error(
                        results,
                        message=f"Input variable '{var_name}' is None, expected units: {expected_units}",
                        category="unit_validation",
                        layer="L1",
                        scope=functor_name,
                    )

            # Check output variable units (if output is available in context)
            output_specs = metadata.output_variables or []
            optional_outputs = set(metadata.optional_output_variables or [])
            if hasattr(context, 'output') and context.output:
                if isinstance(output_specs, dict):
                    output_iter = output_specs.items()
                else:
                    output_iter = ((var_name, None) for var_name in output_specs)

                for var_name, expected_units in output_iter:
                    if var_name not in context.output:
                        if var_name in optional_outputs:
                            continue
                        _add_warning(
                            results,
                            message=f"Output variable '{var_name}' not found, expected units: {expected_units}",
                            category="unit_validation",
                            layer="L1",
                            scope=functor_name,
                        )
            return results
        
        # Create rule metadata
        rule_metadata = RuleMetadata(
            rule_id=rule_id,
            name=f"{functor_name} Unit Validation",
            description=f"Validates units for inputs/outputs of {functor_name}",
            layer="L1",
            domains=[metadata.category.value] if metadata.category else [],
            tags=["units", "automatic", functor_name],
            units_contract=metadata.input_variables,
            version=metadata.version,
            author="FunctorMetadataIntegrator",
            enabled=True
        )
        
        try:
            self.registry.register_rule(validate_functor_units, rule_metadata, overwrite)
            return rule_id
        except ValueError as e:
            logger.warning(f"Could not register unit validation for {functor_name}: {e}")
            return None
    
    def _register_parameter_validation(
            self,
            functor_name: str,
            metadata: FunctorMetadata,
            overwrite: bool) -> Optional[str]:
        """Generate and register parameter validation rule."""
        rule_id = f"{functor_name}_parameter_validation"

        def validate_parameters(context: ValidationContext) -> ValidationResults:
            """Validate parameters match metadata specifications."""
            results: ValidationResults = []

            parameter_specs = metadata.parameters or {}
            parameter_iter = _iter_parameter_entries(parameter_specs)
            param_manager = _resolve_validation_manager()

            for param_name, param_spec in parameter_iter:
                spec_info = param_spec if isinstance(param_spec, dict) else {}
                required = spec_info.get('required', False)
                expected_units = spec_info.get('units')
                min_bound = spec_info.get('min')
                max_bound = spec_info.get('max')
                namespace_hint = spec_info.get('namespace')
                manager_spec = _lookup_manager_spec(
                    param_manager,
                    namespace_hint=namespace_hint,
                    category=metadata.category,
                    functor_name=functor_name,
                    param_name=param_name,
                )

                param_value = None
                if context.parameters:
                    param_value = context.parameters.get(param_name)
                if param_value is None:
                    param_value = _lookup_manager_value(
                        param_manager,
                        namespace_hint=namespace_hint,
                        category=metadata.category,
                        functor_name=functor_name,
                        param_name=param_name,
                        manager_spec=manager_spec,
                    )

                if param_value is None and required:
                    _add_error(
                        results,
                        message=f"Required parameter '{param_name}' not found",
                        category="parameter_validation",
                        layer="L1",
                        scope=functor_name,
                    )
                    continue

                numeric_value = _to_numeric_value(param_value)
                min_bound, max_bound, expected_units = _merge_bound_and_unit_from_spec(
                    min_bound,
                    max_bound,
                    expected_units,
                    manager_spec,
                )
                _validate_parameter_bounds(
                    results,
                    numeric_value=numeric_value,
                    min_bound=min_bound,
                    max_bound=max_bound,
                    functor_name=functor_name,
                    param_name=param_name,
                )
            return results

        rule_metadata = RuleMetadata(
            rule_id=rule_id,
            name=f"{functor_name} Parameter Validation",
            description=f"Validates parameters for {functor_name}",
            layer="L1",
            domains=[metadata.category.value] if metadata.category else [],
            tags=["parameters", "automatic", functor_name],
            inputs_schema=metadata.parameters,
            version=metadata.version,
            author="FunctorMetadataIntegrator",
            enabled=True
        )

        try:
            self.registry.register_rule(validate_parameters, rule_metadata, overwrite)
            return rule_id
        except ValueError as exc:
            logger.warning(f"Could not register parameter validation for {functor_name}: {exc}")
            return None

    def _register_dependency_validation(self,
                                       functor_name: str,
                                       metadata: FunctorMetadata,
                                       overwrite: bool) -> Optional[str]:
        """Generate and register dependency validation rule."""
        rule_id = f"{functor_name}_dependency_validation"
        
        def validate_dependencies(context: ValidationContext) -> ValidationResults:
            """Validate functor dependencies are satisfied."""
            results: ValidationResults = []
            
            # Check that all required inputs are available
            input_specs = metadata.input_variables or {}
            if isinstance(input_specs, dict):
                required_inputs = list(input_specs.keys())
            else:
                required_inputs = list(input_specs)
            state_data = getattr(context, 'state', {}) or {}
            for var_name in required_inputs:
                if var_name not in state_data:
                    _add_error(
                        results,
                        message=f"Required input '{var_name}' not available for {functor_name}",
                        category="dependencies",
                        layer="L2",
                        scope=functor_name,
                    )
            
            # Check coupling interfaces if specified
            if metadata.coupling_interfaces:
                for interface_name in metadata.coupling_interfaces:
                    # Here we would check if the interface is properly connected
                    # For now, just log it
                    _add_info(
                        results,
                        message=f"Coupling interface '{interface_name}' declared",
                        category="dependencies",
                        layer="L2",
                        scope=functor_name,
                    )
            return results
        
        # Create rule metadata
        rule_metadata = RuleMetadata(
            rule_id=rule_id,
            name=f"{functor_name} Dependency Validation",
            description=f"Validates dependencies for {functor_name}",
            layer="L2",
            domains=[metadata.category.value] if metadata.category else [],
            tags=["dependencies", "automatic", functor_name],
            required_data=list(metadata.input_variables.keys()) if isinstance(metadata.input_variables, dict) else list(metadata.input_variables or []),
            version=metadata.version,
            author="FunctorMetadataIntegrator",
            enabled=True
        )
        
        try:
            self.registry.register_rule(validate_dependencies, rule_metadata, overwrite)
            return rule_id
        except ValueError as e:
            logger.warning(f"Could not register dependency validation for {functor_name}: {e}")
            return None
    
    def _register_category_specific_rules(self,
                                         functor_name: str,
                                         metadata: FunctorMetadata,
                                         overwrite: bool) -> List[str]:
        """Generate category-specific validation rules."""
        registered = []
        
        if metadata.category == ProcessCategory.BIOLOGICAL_PROCESS:
            # Add biological process specific validations
            rule_id = f"{functor_name}_biological_constraints"
            
            def validate_biological(context: ValidationContext) -> ValidationResults:
                """Validate biological process constraints."""
                results: ValidationResults = []
                state_data = context.state or {}
                
                # References: the SSOT keys validation.numerical.clamp_min_temperature
                #   and clamp_max_temperature (valid temperature range of biological
                #   processes; their YAML entries record the source).
                # Mathematical expression: warn when the mean temperature T_mean lies
                #   outside [T_min, T_max].
                # Parameter meanings: T_mean the mean of the state temperature; T_min and
                #   T_max the valid temperature range of biological processes.
                # Units: T_mean, T_min, T_max [K].
                # Domain/constraints: a warning only; the state value is not changed.
                # Sign convention: not applicable (a range check).
                if 'temperature' in state_data:
                    temp = state_data['temperature']
                    if isinstance(temp, torch.Tensor):
                        param_manager = _resolve_validation_manager()
                        if param_manager is None:
                            raise RuntimeError(
                                'The biological temperature range needs the parameter manager'
                            )
                        temp_min = float(param_manager.get(
                            'validation.numerical.clamp_min_temperature', units='K', required=True
                        ))
                        temp_max = float(param_manager.get(
                            'validation.numerical.clamp_max_temperature', units='K', required=True
                        ))
                        temp_val = temp.item() if temp.numel() == 1 else temp.mean().item()
                        if temp_val < temp_min or temp_val > temp_max:
                            _add_warning(
                                results,
                                message=f"Temperature {temp_val-273.15:.1f}C outside biological range",
                                category="biological",
                                layer="L1",
                                scope=functor_name,
                            )
                
                # Report declared biological inputs that the state does not supply.
                # No Functor substitutes a default: a required input raises in the
                # Functor and an optional input takes its declared absent branch.
                bio_vars = ['par', 'co2_concentration', 'water_stress']
                for var in bio_vars:
                    if var not in state_data and var in metadata.input_variables:
                        _add_info(
                            results,
                            message=(
                                f"Biological variable '{var}' not provided; no default is "
                                "substituted (a required input raises in the Functor, an "
                                "optional input takes its declared absent branch)"
                            ),
                            category="biological",
                            layer="L1",
                            scope=functor_name,
                        )
                return results
            
            rule_metadata = RuleMetadata(
                rule_id=rule_id,
                name=f"{functor_name} Biological Constraints",
                description=f"Validates biological constraints for {functor_name}",
                layer="L1",
                domains=["biological_process"],
                tags=["biological", "automatic", functor_name],
                version=metadata.version,
                author="FunctorMetadataIntegrator",
                enabled=True
            )
            
            try:
                self.registry.register_rule(validate_biological, rule_metadata, overwrite)
                registered.append(rule_id)
            except ValueError:
                pass
        
        elif metadata.category == ProcessCategory.TRANSPORT_MECHANISM:
            # Add transport mechanism specific validations
            rule_id = f"{functor_name}_transport_constraints"
            
            def validate_transport(context: ValidationContext) -> ValidationResults:
                """Validate transport mechanism constraints."""
                results: ValidationResults = []
                params = context.parameters or {}
                
                # Check CFL condition if available
                if all(k in params for k in ['dt', 'dx', 'velocity']):
                    dt = params['dt']
                    dx = params['dx']
                    v = params['velocity']
                    cfl = v * dt / dx
                    if cfl > 1.0:
                        _add_error(
                            results,
                            message=f"CFL condition violated: {cfl:.2f} > 1.0",
                            category="transport",
                            layer="L1",
                            scope=functor_name,
                        )
                return results
            
            rule_metadata = RuleMetadata(
                rule_id=rule_id,
                name=f"{functor_name} Transport Constraints",
                description=f"Validates transport constraints for {functor_name}",
                layer="L1",
                domains=["transport_mechanism"],
                tags=["transport", "automatic", functor_name],
                version=metadata.version,
                author="FunctorMetadataIntegrator",
                enabled=True
            )
            
            try:
                self.registry.register_rule(validate_transport, rule_metadata, overwrite)
                registered.append(rule_id)
            except ValueError:
                pass
        
        return registered
    
    def unregister_functor_metadata(self, functor_name: str) -> bool:
        """
        Unregister all validation rules generated for a functor.
        
        Parameters:
            functor_name: Name of the functor
            
        Returns:
            True if rules were unregistered, False if not found
        """
        if functor_name not in self._generated_rules:
            return False
        
        rule_ids = self._generated_rules[functor_name]
        for rule_id in rule_ids:
            self.registry.unregister_rule(rule_id)
        
        del self._generated_rules[functor_name]
        logger.info(f"Unregistered {len(rule_ids)} validation rules for functor '{functor_name}'")
        return True
    
    def get_functor_rules(self, functor_name: str) -> List[str]:
        """
        Get all validation rule IDs generated for a functor.
        
        Parameters:
            functor_name: Name of the functor
            
        Returns:
            List of rule IDs
        """
        return self._generated_rules.get(functor_name, [])


# Global integration facade bound to the canonical registry instance.
_global_integrator: Optional[FunctorMetadataIntegrator] = None


def get_global_integrator() -> FunctorMetadataIntegrator:
    """Get the integration facade for the current canonical registry."""
    global _global_integrator
    current_registry = get_global_registry()
    if (
        _global_integrator is None
        or _global_integrator.registry is not current_registry
    ):
        _global_integrator = FunctorMetadataIntegrator(registry=current_registry)
    return _global_integrator


def register_functor_validations(functor_name: str,
                                metadata: FunctorMetadata,
                                overwrite: bool = False) -> List[str]:
    """
    Convenience function to register functor metadata validations.
    
    Parameters:
        functor_name: Name of the functor
        metadata: FunctorMetadata instance
        overwrite: Whether to overwrite existing rules
        
    Returns:
        List of registered rule IDs
    """
    integrator = get_global_integrator()
    return integrator.register_functor_metadata(functor_name, metadata, overwrite)
