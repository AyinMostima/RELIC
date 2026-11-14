"""
Validation Engine

This module provides the core execution engine for the unified validation
framework, orchestrating validation rule execution across all layers (L1-L4)
with support for dependency resolution, parallel execution, and result aggregation.

Key Features:
    - Layer-based execution (L1-L4) with dependency resolution
    - Domain and tag filtering for targeted validation
    - Parallel and sequential execution strategies
    - Result aggregation and report generation
    - Resource management and error handling
    - Progress tracking and performance metrics

Mathematical Framework:
    Engine E executes ruleset R with context C:
    E(R, C) = {r(C) : r in R} where rules are dependency-ordered
    
    For layer L with rules R_L:
    ValidationResult(L, C) = Aggregate(ExecutionStrategy_L(R_L, C))
    
References:
    - Validation framework specification
    - Execution orchestration patterns
    - Performance optimization strategies
"""

from typing import Dict, List, Optional, Union, Any
import importlib
import logging
import time
import threading
from dataclasses import dataclass
from datetime import datetime

from .models import (
    ValidationResultModel, ValidationReportModel, ValidationLevel, 
    ValidationCode, ValidationContext, ValidationResults
)
from .registry import ValidationRegistry, get_global_registry
from .layers import (
    get_execution_strategy, get_all_layers, DEFAULT_LAYER_ORDER
)

logger = logging.getLogger(__name__)


@dataclass
class ExecutionMetrics:
    """Metrics collected during validation execution.

    The engine sets every field explicitly: the counters start at 0, and
    end_time / execution_time_seconds stay None until the run is finalized,
    which happens before the report is generated.
    """
    start_time: datetime
    end_time: Optional[datetime]
    total_rules_executed: int
    rules_passed: int
    rules_failed: int
    execution_time_seconds: Optional[float]
    layer_execution_times: Dict[str, float]


class ValidationEngine:
    """
    Core validation execution engine.
    
    Orchestrates the execution of validation rules across all layers,
    providing filtering, dependency resolution, and result aggregation.
    
    The engine supports multiple execution modes:
    - Single layer validation
    - Multi-layer validation with dependencies
    - Domain-specific validation
    - Tag-based filtering
    
    Example:
        >>> engine = ValidationEngine()
        >>> # Execute L1 and L2 validation for biosphere domain
        >>> results = engine.run(layers=["L1", "L2"], domains=["biosphere"])
        >>> print(f"Found {len(results)} validation issues")
        >>> 
        >>> # Generate comprehensive report
        >>> report = engine.generate_report(results)
        >>> print(report.summary)
    """
    
    def __init__(self,
                 registry: Optional[ValidationRegistry] = None):
        """
        Initialize validation engine.

        Rules run sequentially in the calling thread; the engine has no
        worker pool and no per-rule timeout.

        Parameters:
            registry: Validation rule registry (uses global if None)
        """
        self.registry = registry or get_global_registry()
        self._ensure_core_rules_registered()

        # Execution state
        self._current_context: Optional[ValidationContext] = None
        self._execution_metrics: Optional[ExecutionMetrics] = None
        self._lock = threading.Lock()
        
        logger.info(f"ValidationEngine initialized with {len(self.registry._rules)} registered rules")

    def _ensure_core_rules_registered(self) -> None:
        """
        Ensure framework core rules are present after global-registry resets.

        Notes
        -----
        In test workflows, ``reset_global_registry()`` creates a fresh registry
        while rule modules remain imported in ``sys.modules``. A plain import
        does not rerun decorators, so we reload rule modules when core rules
        are missing.
        """
        global_registry = get_global_registry()
        if self.registry is not global_registry:
            return

        required_core_rules = {
            "water_mass_conservation",
            "dec_complex_structure",
            "hodge_passivity_structure",
            "iphs_structure",
            "chain_map_structure",
            "entropy_step",
            "port_power_units",
            "port_matching",
            "overflow_routing",
            "equivalence_certificate",
            "reaction_network_structure",
            "stochastic_sdae_structure",
            "state_domain_structure",
            "observation_diagnostic_structure",
        }
        if required_core_rules.issubset(self.registry._rules):
            return

        from . import rules as rules_pkg

        for module_name in (
            "cross_domain",
            "stochastic_scores",
            "parameters",
            "conservation",
            "coupling_contracts",
            "dec_complex",
            "hodge",
            "iphs",
            "chain_map",
            "state_domain",
            "observation_diagnostic",
            "entropy_step",
            "port_power",
            "port_matching",
            "overflow_routing",
            "equivalence_certificate",
            "reaction_network",
            "stochastic_sdae",
        ):
            module = getattr(rules_pkg, module_name, None)
            if module is not None:
                importlib.reload(module)

        importlib.reload(rules_pkg)

    def run(self,
           layers: Optional[Union[str, List[str]]] = None,
           domains: Optional[List[str]] = None,
           tags: Optional[List[str]] = None,
           enabled_only: bool = True,
           context: Optional[ValidationContext] = None,
           fail_fast: bool = False,
           include_metrics: bool = True,
        **kwargs) -> ValidationReportModel:
        """
        Execute validation rules and return comprehensive report.
        
        Parameters:
            layers: Validation layer(s) to execute (default: all layers)
            domains: Scientific domains to include
            tags: Tags to filter rules by
            enabled_only: Only execute enabled rules
            context: Validation context (created if None)
            fail_fast: Stop execution on first critical error
            include_metrics: Include execution metrics in report
            
        Returns:
            ValidationReportModel: Comprehensive validation report
            
        Raises:
            ValueError: If invalid layers specified
            RuntimeError: If critical validation failures with fail_fast=True
        """
            
        # Normalize layers
        if layers is None:
            target_layers = DEFAULT_LAYER_ORDER
        elif isinstance(layers, str):
            target_layers = [layers]
        else:
            target_layers = layers
        
        # Validate layers
        for layer in target_layers:
            if layer not in get_all_layers():
                raise ValueError(f"Invalid validation layer: {layer}")
        
        # Create execution context
        if context is None:
            context = ValidationContext(
                domain_filter=domains,
                metadata={"execution_timestamp": datetime.now()}
            )
        
        # Initialize metrics
        metrics = ExecutionMetrics(
            start_time=datetime.now(),
            end_time=None,
            total_rules_executed=0,
            rules_passed=0,
            rules_failed=0,
            execution_time_seconds=None,
            layer_execution_times={},
        )
        
        try:
            with self._lock:
                self._current_context = context
                self._execution_metrics = metrics
            
            logger.info(f"Starting validation execution: layers={target_layers}, domains={domains}, tags={tags}")
            
            # Execute validation by layer
            all_results = []
            critical_failure = False
            
            for layer in target_layers:
                if critical_failure and fail_fast:
                    logger.warning(f"Skipping layer {layer} due to critical failure with fail_fast=True")
                    break
                
                layer_start_time = time.time()
                
                try:
                    # Get rules for this layer
                    layer_results = self._execute_layer(
                        layer=layer,
                        domains=domains,
                        tags=tags,
                        enabled_only=enabled_only,
                        context=context
                    )
                    
                    all_results.extend(layer_results)
                    
                    # Update metrics
                    layer_execution_time = time.time() - layer_start_time
                    metrics.layer_execution_times[layer] = layer_execution_time
                    metrics.total_rules_executed += len([r for r in layer_results if r.scope])
                    
                    # Check for critical failures
                    layer_critical_count = len([r for r in layer_results if r.level == ValidationLevel.CRITICAL])
                    if layer_critical_count > 0:
                        critical_failure = True
                        logger.error(f"Layer {layer} produced {layer_critical_count} critical failures")
                    
                    logger.info(f"Layer {layer} completed: {len(layer_results)} results in {layer_execution_time:.2f}s")
                    
                except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
                    logger.error(f"Layer {layer} execution failed: {str(e)}")
                    
                    # Create error result for layer failure
                    error_result = ValidationResultModel(
                        level=ValidationLevel.CRITICAL,
                        code=ValidationCode.GENERIC_VALIDATION_FAILED,
                        message=f"Layer {layer} execution failed: {str(e)}",
                        layer=layer,
                        scope="validation_engine",
                        tags=["engine_error", "layer_failure"]
                    )
                    all_results.append(error_result)
                    critical_failure = True
                    
                    if fail_fast:
                        break
            
            # Finalize the metrics first: the report copies their values.
            metrics.end_time = datetime.now()
            metrics.execution_time_seconds = (metrics.end_time - metrics.start_time).total_seconds()
            for result in all_results:
                if result.level in [ValidationLevel.INFO, ValidationLevel.WARNING]:
                    metrics.rules_passed += 1
                else:
                    metrics.rules_failed += 1

            # Generate comprehensive report
            report = self.generate_report(all_results, metrics if include_metrics else None)

            logger.info(f"Validation execution completed: {len(all_results)} results in {metrics.execution_time_seconds:.2f}s")
            
            # Check for critical failures with fail_fast
            if fail_fast and critical_failure:
                raise RuntimeError("Critical validation failures detected with fail_fast=True")
            
            return report
            
        finally:
            with self._lock:
                self._current_context = None
                self._execution_metrics = None
    
    def _execute_layer(self,
                      layer: str,
                      domains: Optional[List[str]],
                      tags: Optional[List[str]],
                      enabled_only: bool,
                      context: ValidationContext) -> ValidationResults:
        """
        Execute all validation rules for a specific layer.
        
        Parameters:
            layer: Validation layer to execute
            domains: Domain filter
            tags: Tag filter  
            enabled_only: Only execute enabled rules
            context: Validation context
            
        Returns:
            List of validation results from all rules in the layer
        """
        # Build filter criteria
        filters = {"layer": layer}
        if domains:
            filters["domains"] = domains
        if tags:
            filters["tags"] = tags  
        if enabled_only:
            filters["enabled"] = True
        
        # Get matching rules
        rule_ids = self.registry.list_rules(**filters)
        
        if not rule_ids:
            logger.info(f"No rules found for layer {layer} with specified filters")
            return []
        
        # Resolve dependencies and order rules
        try:
            ordered_rule_ids = self.registry.resolve_dependencies(rule_ids)
        except ValueError as e:
            logger.error(f"Dependency resolution failed for layer {layer}: {str(e)}")
            # Create error result
            error_result = ValidationResultModel(
                level=ValidationLevel.CRITICAL,
                code=ValidationCode.GENERIC_VALIDATION_FAILED,
                message=f"Dependency resolution failed: {str(e)}",
                layer=layer,
                scope="validation_engine",
                tags=["dependency_error"]
            )
            return [error_result]
        
        logger.debug(f"Executing {len(ordered_rule_ids)} rules for layer {layer}: {ordered_rule_ids}")
        
        # Get execution strategy for layer
        execution_strategy = get_execution_strategy(layer)
        
        # Get rule functions
        rule_functions = []
        for rule_id in ordered_rule_ids:
            rule_func = self.registry.get_rule(rule_id)
            if rule_func:
                rule_functions.append(rule_func)
            else:
                logger.warning(f"Rule {rule_id} not found in registry")
        
        # Set layer context
        context.execution_layer = layer
        
        # Execute rules using layer strategy
        try:
            results = execution_strategy(rule_functions, context)
            return results if isinstance(results, list) else [results]
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            logger.error(f"Rule execution strategy failed for layer {layer}: {str(e)}")
            error_result = ValidationResultModel(
                level=ValidationLevel.CRITICAL,
                code=ValidationCode.GENERIC_VALIDATION_FAILED,
                message=f"Execution strategy failed: {str(e)}",
                layer=layer,
                scope="validation_engine",
                tags=["execution_error"]
            )
            return [error_result]
    
    def generate_report(self, 
                       results: ValidationResults,
                       metrics: Optional[ExecutionMetrics] = None) -> ValidationReportModel:
        """
        Generate comprehensive validation report from results.
        
        Parameters:
            results: List of validation results
            metrics: Execution metrics (optional)
            
        Returns:
            ValidationReportModel: Structured report with summaries
        """
        report = ValidationReportModel(results=results)
        
        # Add execution metadata
        if metrics:
            report.execution_metadata = {
                "execution_time_seconds": metrics.execution_time_seconds,
                "total_rules_executed": metrics.total_rules_executed,
                "rules_passed": metrics.rules_passed,
                "rules_failed": metrics.rules_failed,
                "layer_execution_times": metrics.layer_execution_times,
                "start_time": metrics.start_time.isoformat(),
                "end_time": metrics.end_time.isoformat() if metrics.end_time else None
            }
        
        # Generate summaries (handled by ValidationReportModel._update_summary())
        report._update_summary()
        
        return report
    def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution status and metrics."""
        with self._lock:
            if self._current_context is None:
                return {"status": "idle"}
            
            return {
                "status": "running",
                "current_layer": self._current_context.execution_layer,
                "start_time": self._execution_metrics.start_time.isoformat() if self._execution_metrics else None,
                "rules_executed": self._execution_metrics.total_rules_executed if self._execution_metrics else 0
            }
    
    def stop_execution(self) -> bool:
        """Attempt to stop current execution (best effort)."""
        # Note: This is a simplified implementation
        # In production, would need more sophisticated cancellation
        with self._lock:
            if self._current_context is not None:
                logger.warning("Execution stop requested - setting stop flag")
                if self._current_context.metadata is None:
                    self._current_context.metadata = {}
                self._current_context.metadata["stop_requested"] = True
                return True
        return False
    
    def get_registry_info(self) -> Dict[str, Any]:
        """Get information about the validation registry."""
        return {
            "registry_statistics": self.registry.get_rule_statistics(),
            "registry_integrity": self.registry.validate_registry_integrity(),
            "available_layers": get_all_layers(),
        }


# Convenience functions for common validation scenarios

def validate_layer(layer: str, 
                  domains: Optional[List[str]] = None,
                  context: Optional[ValidationContext] = None) -> ValidationReportModel:
    """
    Convenience function to validate a single layer.
    
    Parameters:
        layer: Validation layer (L1/L2/L3/L4)
        domains: Scientific domains to include
        context: Validation context
        
    Returns:
        ValidationReportModel: Validation report for the layer
    """
    engine = ValidationEngine()
    return engine.run(layers=[layer], domains=domains, context=context)


def validate_domains(domains: List[str],
                    layers: Optional[List[str]] = None,
                    context: Optional[ValidationContext] = None) -> ValidationReportModel:
    """
    Convenience function to validate specific domains.
    
    Parameters:
        domains: Scientific domains to validate
        layers: Validation layers to execute
        context: Validation context
        
    Returns:
        ValidationReportModel: Validation report for the domains
    """
    engine = ValidationEngine()
    return engine.run(layers=layers, domains=domains, context=context)


def validate_all(context: Optional[ValidationContext] = None,
                fail_fast: bool = False) -> ValidationReportModel:
    """
    Convenience function to run complete validation across all layers.
    
    Parameters:
        context: Validation context
        fail_fast: Stop on first critical error
        
    Returns:
        ValidationReportModel: Comprehensive validation report
    """
    engine = ValidationEngine()
    return engine.run(context=context, fail_fast=fail_fast)
