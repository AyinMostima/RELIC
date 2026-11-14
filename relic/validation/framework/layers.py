"""
Validation Layer Definitions and Orchestration

This module defines the four-tier validation architecture (L1-L4) used
throughout RELIC for systematic validation from unit tests to scientific
validation against observational data.

Validation Layers:
    L1_UNIT: Parameter specs, units, distributions, scoring rules
    L2_INTEGRATION: Interface contracts, conservation, coupling validation
    L3_BENCHMARK: Comparison with established models/baselines  
    L4_SCIENTIFIC: Validation against real observational data

Mathematical Framework:
    Each layer l in {L1, L2, L3, L4} contains validation rules R_l:
    Validation(system) = _{l} {r(system) : r in R_l}
    
    Dependency order: L1 -> L2 -> L3 -> L4 (higher layers depend on lower layers)
    
References:
    - RELIC Core Development Guidelines
    - Validation framework specification
    - Scientific validation best practices
"""

from typing import Dict, List, Callable, Any
from dataclasses import dataclass
from enum import Enum


class ValidationLayer(str, Enum):
    """Enumeration of validation layers with their identifiers."""
    L1_UNIT = "L1"
    L2_INTEGRATION = "L2" 
    L3_BENCHMARK = "L3"
    L4_SCIENTIFIC = "L4"


@dataclass
class LayerMetadata:
    """
    Metadata describing a validation layer.
    
    Fields:
        name: Human-readable layer name
        description: Detailed description of the layer's purpose
        dependencies: List of layers that must pass before this one
        domains: Applicable scientific domains  
        typical_rules: Examples of validation rules in this layer
        execution_strategy: How rules in this layer are executed
        failure_handling: What happens when rules fail
    """
    name: str
    description: str
    dependencies: List[str]
    domains: List[str]
    typical_rules: List[str]
    execution_strategy: str
    failure_handling: str


# Layer constant definitions
L1_UNIT = ValidationLayer.L1_UNIT.value
L2_INTEGRATION = ValidationLayer.L2_INTEGRATION.value  
L3_BENCHMARK = ValidationLayer.L3_BENCHMARK.value
L4_SCIENTIFIC = ValidationLayer.L4_SCIENTIFIC.value


# Layer metadata registry
LAYER_METADATA = {
    L1_UNIT: LayerMetadata(
        name="Unit Validation",
        description=(
            "Validates individual components in isolation: parameter specifications, "
            "physical unit consistency, distribution parameters, bounds checking, "
            "and probabilistic scoring rules (CRPS, PIT, Brier scores)."
        ),
        dependencies=[],
        domains=[
            "parameters", "stochastic", "units", "cf_conventions", 
            "scoring_rules", "distributions"
        ],
        typical_rules=[
            "parameter_bounds_check",
            "unit_conversion_validity", 
            "cf_standard_name_syntax",
            "distribution_parameter_consistency",
            "crps_score_validation",
            "pit_uniformity_test",
            "brier_score_computation",
            "quantile_score_validation"
        ],
        execution_strategy="parallel",
        failure_handling="collect_and_continue"
    ),
    
    L2_INTEGRATION: LayerMetadata(
        name="Integration Validation", 
        description=(
            "Validates interactions between components: interface contracts between "
            "domains, mass/energy conservation laws, coupling validation, and "
            "cross-domain variable consistency."
        ),
        dependencies=[L1_UNIT],
        domains=[
            "biosphere_atmosphere", "soil_vegetation", "water_carbon",
            "hydrosphere_interfaces", "coupling_contracts", "conservation"
        ],
        typical_rules=[
            "water_mass_conservation",
            "carbon_mass_conservation", 
            "energy_balance_validation",
            "interface_variable_consistency",
            "coupling_contract_compliance",
            "stress_factor_bounds_check",
            "cf_cross_domain_consistency"
        ],
        execution_strategy="dependency_ordered",
        failure_handling="fail_fast_on_critical"
    ),
    
    L3_BENCHMARK: LayerMetadata(
        name="Benchmark Validation",
        description=(
            "Compares model outputs against established benchmarks: reference "
            "datasets, mature model outputs (LPJ-GUESS, CENTURY), and standardized "
            "test cases from literature."
        ),
        dependencies=[L1_UNIT, L2_INTEGRATION],
        domains=[
            "benchmarks", "reference_models", "test_cases",
            "literature_comparisons", "model_intercomparison"
        ],
        typical_rules=[
            "lpj_guess_comparison",
            "century_carbon_benchmark",
            "fluxnet_gpp_comparison", 
            "standardized_test_cases",
            "literature_validation_cases",
            "model_intercomparison_metrics"
        ],
        execution_strategy="sequential_with_data_dependency",
        failure_handling="collect_and_warn"
    ),
    
    L4_SCIENTIFIC: LayerMetadata(
        name="Scientific Validation",
        description=(
            "Validates against real observational data: FluxNet observations, "
            "remote sensing data, field measurements, and scientific consistency "
            "checks against established ecological principles."
        ),
        dependencies=[L1_UNIT, L2_INTEGRATION, L3_BENCHMARK],
        domains=[
            "fluxnet", "remote_sensing", "field_observations", 
            "ecological_principles", "phenology", "climate_response"
        ],
        typical_rules=[
            "fluxnet_tower_validation",
            "modis_lai_comparison",
            "phenology_timing_validation",
            "climate_response_validation",
            "ecosystem_service_validation",
            "carbon_cycle_realism_check"
        ],
        execution_strategy="data_driven_sequential", 
        failure_handling="scientific_review_required"
    )
}


# SSOT Section 20 wording for a rule whose whole data domain is absent.
_DOMAIN_ABSENT = "not applicable (domain absent)"


class LayerExecutionStrategy:
    """
    Execution strategies for different validation layers.
    
    Defines how validation rules within a layer should be executed,
    considering dependencies, resource requirements, and failure handling.
    """
    
    @staticmethod
    def parallel(rules: List[Callable], context: Any) -> List[Any]:
        """
        Execute rules in parallel (suitable for L1).
        
        Parameters:
            rules: List of validation rule functions
            context: Validation context
            
        Returns:
            List of validation results
        """
        # Simplified parallel execution - in production would use threading/multiprocessing
        results = []
        for rule in rules:
            try:
                result = rule(context)
                if isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)
            except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
                # Convert exceptions to validation results
                from .models import ValidationResultModel, ValidationLevel, ValidationCode
                error_result = ValidationResultModel(
                    level=ValidationLevel.ERROR,
                    code=ValidationCode.GENERIC_VALIDATION_FAILED,
                    message=f"Rule execution failed: {str(e)}",
                    layer=context.execution_layer or "unknown",
                    scope=getattr(rule, '__name__', 'unknown_rule'),
                    tags=["execution_error"]
                )
                results.append(error_result)
        return results
    
    @staticmethod
    def dependency_ordered(rules: List[Callable], context: Any) -> List[Any]:
        """
        Execute rules respecting dependencies (suitable for L2).
        
        Parameters:
            rules: List of validation rule functions  
            context: Validation context
            
        Returns:
            List of validation results
        """
        # For now, execute sequentially - in production would build dependency graph
        results = []
        critical_failure = False
        
        for rule in rules:
            if critical_failure:
                break
                
            try:
                result = rule(context)
                if isinstance(result, list):
                    results.extend(result)
                    # Check for critical failures
                    if any(r.level.value == "CRITICAL" for r in result):
                        critical_failure = True
                else:
                    results.append(result)
                    if result.level.value == "CRITICAL":
                        critical_failure = True
            except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
                from .models import ValidationResultModel, ValidationLevel, ValidationCode
                error_result = ValidationResultModel(
                    level=ValidationLevel.CRITICAL,
                    code=ValidationCode.GENERIC_VALIDATION_FAILED,
                    message=f"Critical rule failure: {str(e)}",
                    layer=context.execution_layer or "unknown", 
                    scope=getattr(rule, '__name__', 'unknown_rule'),
                    tags=["execution_error", "critical"]
                )
                results.append(error_result)
                critical_failure = True
                
        return results
    
    @staticmethod  
    def sequential_with_data_dependency(rules: List[Callable], context: Any) -> List[Any]:
        """
        Execute rules sequentially with data dependencies (suitable for L3).
        
        Parameters:
            rules: List of validation rule functions
            context: Validation context
            
        Returns:
            List of validation results
        """
        # Sequential execution with data availability checks
        results = []
        
        for rule in rules:
            # Required-data contract (SSOT Section 20 applicability rule): when
            # none of the rule's required data is present its domain is absent
            # and the rule is not applicable (WARNING, stated as such); when
            # only part of it is present the rule is not evaluable, which is a
            # failed result (ERROR, passed=False) naming the missing keys.
            if hasattr(rule, '_required_data'):
                required_data = list(rule._required_data)
                available = context.metadata or {}
                missing = [key for key in required_data if key not in available]
                if missing:
                    from .models import ValidationResultModel, ValidationLevel, ValidationCode
                    rule_name = getattr(rule, '__name__', 'unknown_rule')
                    if len(missing) == len(required_data):
                        results.append(ValidationResultModel(
                            level=ValidationLevel.WARNING,
                            code=ValidationCode.MISSING_REQUIRED_DATA,
                            message=(
                                f"{rule_name}: {_DOMAIN_ABSENT}; the context carries none of "
                                f"{required_data}, no check was performed"
                            ),
                            layer=context.execution_layer or "unknown",
                            scope=rule_name,
                            tags=["data_dependency", "not_applicable"]
                        ))
                    else:
                        results.append(ValidationResultModel(
                            level=ValidationLevel.ERROR,
                            code=ValidationCode.MISSING_REQUIRED_DATA,
                            message=(
                                f"{rule_name} is not evaluable: required data {missing} "
                                "is missing from the validation context"
                            ),
                            passed=False,
                            layer=context.execution_layer or "unknown",
                            scope=rule_name,
                            tags=["data_dependency", "not_evaluable"]
                        ))
                    continue
            
            try:
                result = rule(context)
                if isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)
            except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
                # A benchmark rule may itself report a benchmark mismatch as a
                # WARNING result; a rule that crashes produced no benchmark
                # evidence, so the execution failure is an ERROR (never passing).
                from .models import ValidationResultModel, ValidationLevel, ValidationCode
                error_result = ValidationResultModel(
                    level=ValidationLevel.ERROR,
                    code=ValidationCode.GENERIC_VALIDATION_FAILED,
                    message=f"Benchmark validation rule execution failed: {str(e)}",
                    layer=context.execution_layer or "unknown",
                    scope=getattr(rule, '__name__', 'unknown_rule'),
                    tags=["execution_error", "benchmark"]
                )
                results.append(error_result)
                
        return results
    
    @staticmethod
    def data_driven_sequential(rules: List[Callable], context: Any) -> List[Any]:
        """
        Execute rules based on data availability (suitable for L4).
        
        Parameters:
            rules: List of validation rule functions
            context: Validation context
            
        Returns:
            List of validation results
        """
        # Data-driven execution for scientific validation
        results = []
        
        for rule in rules:
            try:
                result = rule(context)
                if isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)
            except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
                # L4 failures require scientific review
                from .models import ValidationResultModel, ValidationLevel, ValidationCode
                error_result = ValidationResultModel(
                    level=ValidationLevel.ERROR,
                    code=ValidationCode.GENERIC_VALIDATION_FAILED,
                    message=f"Scientific validation requires review: {str(e)}",
                    layer=context.execution_layer or "unknown",
                    scope=getattr(rule, '__name__', 'unknown_rule'),
                    tags=["execution_error", "scientific_review"]
                )
                results.append(error_result)
                
        return results


def get_layer_metadata(layer: str) -> LayerMetadata:
    """Get metadata for a validation layer."""
    if layer not in LAYER_METADATA:
        raise ValueError(f"Unknown validation layer: {layer}")
    return LAYER_METADATA[layer]


def get_execution_strategy(layer: str) -> Callable:
    """Get the execution strategy function for a layer."""
    metadata = get_layer_metadata(layer)
    strategy_name = metadata.execution_strategy
    
    strategy_map = {
        "parallel": LayerExecutionStrategy.parallel,
        "dependency_ordered": LayerExecutionStrategy.dependency_ordered,
        "sequential_with_data_dependency": LayerExecutionStrategy.sequential_with_data_dependency,
        "data_driven_sequential": LayerExecutionStrategy.data_driven_sequential
    }
    
    if strategy_name not in strategy_map:
        raise ValueError(f"Unknown execution strategy: {strategy_name}")
    
    return strategy_map[strategy_name]


def validate_layer_dependencies(layer: str) -> List[str]:
    """
    Get the list of layers that must be validated before the given layer.
    
    Parameters:
        layer: Target validation layer
        
    Returns:
        List of dependency layers in execution order
    """
    metadata = get_layer_metadata(layer)
    return list(metadata.dependencies)


def get_all_layers() -> List[str]:
    """Get all available validation layers in dependency order."""
    return [L1_UNIT, L2_INTEGRATION, L3_BENCHMARK, L4_SCIENTIFIC]


def get_layers_for_domains(domains: List[str]) -> Dict[str, List[str]]:
    """
    Get validation layers that apply to specific domains.
    
    Parameters:
        domains: List of domain names to filter by
        
    Returns:
        Dictionary mapping layer -> applicable domains
    """
    result = {}
    unique_domains = list(dict.fromkeys(domains))
    
    for layer, metadata in LAYER_METADATA.items():
        applicable_domains = [
            domain for domain in unique_domains if domain in metadata.domains
        ]
        if applicable_domains:
            result[layer] = applicable_domains
    
    return result


# Default layer execution order
DEFAULT_LAYER_ORDER = [L1_UNIT, L2_INTEGRATION, L3_BENCHMARK, L4_SCIENTIFIC]

# Layer filtering shortcuts
CORE_LAYERS = [L1_UNIT, L2_INTEGRATION]  # Essential for CI
BENCHMARK_LAYERS = [L3_BENCHMARK]  # For model comparison  
SCIENTIFIC_LAYERS = [L4_SCIENTIFIC]  # For research validation
