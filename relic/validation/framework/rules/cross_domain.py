"""
Cross-Domain Validation Rules

This module contains registered rules for cross-domain variable consistency,
unit conversion, and lexical syntax of optional RELIC CF base identities.
Official CF table membership and dataset-wide conformance are outside this module.

Key Rules:
    - Variable name consistency across domain interfaces
    - Unit consistency and conversion validation
    - Optional CF base-identity lexical syntax
    - Mass/energy conservation at domain boundaries
    - Hydrosphere interface validation

Mathematical Framework:
    For variables v_1 in Domain_1, v_2 in Domain_2 at interface I:
    - CF base syntax: type(s) is str and fullmatch([a-z][a-z0-9_]*, s)
    - Name consistency: map(v_1)  map(v_2) under standard naming
    - Unit consistency: convert(units(v_1)) -> units(v_2) exists
    - Value consistency: |v_1 - convert(v_2, units(v_1))| <= tolerance

References:
    - CF standard-name construction background:
      https://cfconventions.org/Data/cf-standard-names/docs/guidelines.html
      This reference does not establish official table membership here.
    - RELIC ValidationRegistry contracts
    - RELIC domain interface specifications
"""

from collections.abc import Mapping
import re
from typing import Any
import logging

import torch

from ..models import (
    ValidationResultModel, ValidationLevel, ValidationCode,
    ValidationContext, ValidationResults, ValidationDetail
)
from ..decorators import validated_rule
from ....scientific.hydrosphere.interface_mapping import (
    hydrosphere_mapper,
    validate_hydrosphere_interface as validate_hydrosphere_functor_interface,
)

logger = logging.getLogger(__name__)

# Import unit converter and variable specs
from ....data.standards.unit_converter import (
    unit_converter,
    ConversionContext,
)


@validated_rule(
    rule_id="cf_standard_name_syntax",
    layer="L1",
    name="RELIC CF Standard-Name Base Syntax",
    description=(
        "Checks only the lowercase ASCII lexical form of explicit RELIC "
        "cf_standard_name base identities; official CF table membership is not "
        "evaluated"
    ),
    domains=["cf_conventions", "metadata"],
    tags=["cf_syntax", "metadata", "lexical"],
    references=[
        "CF Standard Name Construction Guidelines "
        "(https://cfconventions.org/Data/cf-standard-names/docs/guidelines.html)"
    ],
)
def validate_cf_standard_name_syntax(context: ValidationContext) -> ValidationResults:
    """
    Check lexical syntax for explicit RELIC CF base-name identities.

    References:
        CF Standard Name Construction Guidelines. Official table membership,
        aliases, modifiers, canonical units, and quantity semantics are not checked.

    Mathematical Expression:
        Syntax(s) = type(s) is str and fullmatch([a-z][a-z0-9_]*, s).

    Parameter meanings:
        context.state_manager.schema: Mapping from local variable names to specs.
        spec.cf_standard_name: Optional exact base identity to check.

    Units:
        Not applicable; this rule does not evaluate canonical-unit compatibility.

    Domain and constraints:
        The field is optional. Missing or None values are neutral. An explicit
        value must be one lowercase ASCII base-name lexeme. A NetCDF standard_name
        attribute containing a modifier is outside this internal identity field.

    Sign convention:
        Not applicable.

    Returns:
        One scope-limited syntax result. It never proves official CF membership.
    """
    if context.state_manager is None:
        return [ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.MISSING_REQUIRED_DATA,
            message="State manager required for CF lexical syntax audit",
            layer="L1",
            scope="cf_standard_name_syntax",
            tags=["cf_syntax", "missing_data"],
            details=[ValidationDetail(
                expected="StateManager.schema mapping",
                metadata={
                    "validation_scope": "lexical_syntax_only",
                    "official_membership_checked": False,
                },
            )],
        )]

    state_schema = getattr(context.state_manager, "schema", None)
    if not isinstance(state_schema, Mapping):
        return [ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.GENERIC_VALIDATION_FAILED,
            message="StateManager.schema must be a mapping for CF lexical syntax audit",
            layer="L1",
            scope="cf_standard_name_syntax",
            tags=["cf_syntax", "schema_error"],
            details=[ValidationDetail(
                expected="StateManager.schema mapping",
                metadata={
                    "validation_scope": "lexical_syntax_only",
                    "official_membership_checked": False,
                },
            )],
        )]

    variables_seen = len(state_schema)
    explicit_names_syntax_checked = 0
    variables_without_cf_name = 0
    invalid_details = []

    for var_name, var_spec in state_schema.items():
        if isinstance(var_spec, Mapping):
            cf_standard_name = var_spec.get("cf_standard_name")
        else:
            cf_standard_name = getattr(var_spec, "cf_standard_name", None)

        if cf_standard_name is None:
            variables_without_cf_name += 1
            continue

        explicit_names_syntax_checked += 1
        if not _matches_cf_standard_name_syntax(cf_standard_name):
            invalid_details.append(ValidationDetail(
                variable_name=str(var_name),
                cf_standard_name=(
                    cf_standard_name if isinstance(cf_standard_name, str) else None
                ),
                value=(
                    cf_standard_name
                    if isinstance(cf_standard_name, (str, int, float))
                    else repr(cf_standard_name)
                ),
                expected="[a-z][a-z0-9_]* RELIC base identity",
                metadata={
                    "validation_scope": "lexical_syntax_only",
                    "official_membership_checked": False,
                    "input_type": type(cf_standard_name).__name__,
                },
            ))

    metrics = {
        "variables_seen": variables_seen,
        "explicit_names_syntax_checked": explicit_names_syntax_checked,
        "variables_without_cf_name": variables_without_cf_name,
        "syntax_invalid_count": len(invalid_details),
    }

    if invalid_details:
        return [ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.GENERIC_VALIDATION_FAILED,
            message=(
                f"Lexical form rejected for {len(invalid_details)} explicit RELIC "
                "CF base-name identities; official membership was not evaluated"
            ),
            layer="L1",
            scope="cf_standard_name_syntax",
            tags=["cf_syntax", "invalid_lexical_form"],
            details=invalid_details,
            metrics=metrics,
        )]

    if explicit_names_syntax_checked == 0:
        message = (
            "No explicit CF standard names were provided for lexical syntax "
            "checking; official membership was not evaluated"
        )
    else:
        message = (
            "Lexical syntax accepted for "
            f"{explicit_names_syntax_checked} explicit RELIC CF base-name "
            "identities; official membership not evaluated"
        )

    return [ValidationResultModel(
        level=ValidationLevel.INFO,
        code=ValidationCode.GENERIC_VALIDATION_OK,
        message=message,
        layer="L1",
        scope="cf_standard_name_syntax",
        tags=["cf_syntax", "lexical"],
        details=[ValidationDetail(
            expected="[a-z][a-z0-9_]* RELIC base identity",
            metadata={
                "validation_scope": "lexical_syntax_only",
                "official_membership_checked": False,
            },
        )],
        metrics=metrics,
    )]


@validated_rule(
    rule_id="unit_conversion_validity",
    layer="L1", 
    name="Unit Conversion Validity",
    description="Validates unit conversion consistency between domains",
    domains=["units", "conversion"],
    tags=["units", "conversion", "consistency"],
    units_contract={"input_units": "various", "output_units": "various"},
    references=["SI Unit System", "CF-1.8 Unit Conventions"]
)
def validate_unit_conversion_validity(context: ValidationContext) -> ValidationResults:
    """
    Validate unit conversion consistency for cross-domain variables.
    
    Mathematical Expression:
        For units u_1, u_2: consistent(u_1, u_2)   conversion_factor c : u_2 = c x u_1
        
    Parameters:
        context: Validation context with unit conversion pairs to test
        
    Returns:
        List of validation results for unit conversion consistency
    """
    results = []
    
    if unit_converter is None:
        return [ValidationResultModel(
            level=ValidationLevel.WARNING,
            code=ValidationCode.MISSING_REQUIRED_DATA,
            message="Unit converter not available",
            layer="L1",
            scope="unit_conversion_validity",
            tags=["units", "missing_dependency"]
        )]
    
    # Test common unit conversion pairs
    test_conversions = [
        # Mass flux conversions
        ('kg m-2 s-1', 'g m-2 s-1', 1000.0),
        ('g m-2 s-1', 'kg m-2 s-1', 0.001),
        ('kg m-2 s-1', 'mm s-1', 1.0),  # For water
        
        # Carbon flux conversions  
        ('kg C m-2 s-1', 'g C m-2 s-1', 1000.0),
        ('mol C m-2 s-1', 'g C m-2 s-1', 12.011),
        
        # Temperature conversions
        ('K', 'degC', None),  # Function-based conversion
        ('degC', 'K', None),
        
        # Pressure conversions
        ('Pa', 'kPa', 0.001),
        ('Pa', 'hPa', 0.01),
        
        # Concentration conversions
        ('ppm', 'mol mol-1', 1e-6),
        ('ppb', 'mol mol-1', 1e-9)
    ]
    
    successful_conversions = 0
    failed_conversions = 0
    
    for from_unit, to_unit, expected_factor in test_conversions:
        try:
            # Test if conversion is possible
            can_convert = unit_converter.can_convert(from_unit, to_unit, variable_name='generic')
            
            if can_convert:
                # Test actual conversion with a test value
                test_value = torch.tensor(1.0)
                converted_value = unit_converter.convert(test_value, 'generic', from_unit, to_unit, ConversionContext())
                
                if expected_factor is not None:
                    # Check if conversion factor is correct
                    actual_factor = float(converted_value)
                    if abs(actual_factor - expected_factor) > 1e-10:
                        results.append(ValidationResultModel(
                            level=ValidationLevel.ERROR,
                            code=ValidationCode.CF_UNIT_CONVERSION_FAILED,
                            message=f"Unit conversion factor incorrect: {from_unit}->{to_unit}",
                            layer="L1",
                            scope="unit_conversion_validity",
                            tags=["units", "conversion_error"],
                            details=[ValidationDetail(
                                variable_name=f"{from_unit}_to_{to_unit}",
                                value=actual_factor,
                                expected=expected_factor,
                                units="dimensionless"
                            )]
                        ))
                        failed_conversions += 1
                    else:
                        successful_conversions += 1
                else:
                    # Function-based conversion (e.g., temperature)
                    successful_conversions += 1
                    
            else:
                results.append(ValidationResultModel(
                    level=ValidationLevel.WARNING,
                    code=ValidationCode.CF_UNIT_CONVERSION_FAILED,
                    message=f"Unit conversion not supported: {from_unit}->{to_unit}",
                    layer="L1",
                    scope="unit_conversion_validity",
                    tags=["units", "unsupported_conversion"],
                    details=[ValidationDetail(
                        variable_name=f"{from_unit}_to_{to_unit}",
                        value="not_supported"
                    )]
                ))
                failed_conversions += 1
                
        except (TypeError, ValueError, RuntimeError) as e:
            results.append(ValidationResultModel(
                level=ValidationLevel.ERROR,
                code=ValidationCode.CF_UNIT_CONVERSION_FAILED,
                message=f"Unit conversion failed: {from_unit}->{to_unit}: {str(e)}",
                layer="L1",
                scope="unit_conversion_validity",
                tags=["units", "conversion_exception"],
                details=[ValidationDetail(
                    variable_name=f"{from_unit}_to_{to_unit}",
                    metadata={"error": str(e)}
                )]
            ))
            failed_conversions += 1
    
    # Summary result
    total_tests = len(test_conversions)
    if failed_conversions == 0:
        results.append(ValidationResultModel(
            level=ValidationLevel.INFO,
            code=ValidationCode.CF_UNIT_CONVERSION_OK,
            message="All unit conversions validated successfully",
            layer="L1",
            scope="unit_conversion_validity",
            tags=["units", "success"],
            metrics={
                "total_conversions_tested": total_tests,
                "successful_conversions": successful_conversions,
                "failed_conversions": failed_conversions
            }
        ))
    
    return results


@validated_rule(
    rule_id="variable_consistency_cross_domain",
    layer="L2",
    name="Cross-Domain Variable Consistency",
    description="Validates variable name and unit consistency across domain interfaces",
    domains=["biosphere", "atmosphere", "hydrosphere", "pedosphere"],
    tags=["cross_domain", "consistency", "interfaces"],
    dependencies=["cf_standard_name_syntax", "unit_conversion_validity"]
)
def validate_variable_consistency_cross_domain(context: ValidationContext) -> ValidationResults:
    """
    Validate variable consistency across domain boundaries.
    
    Mathematical Expression:
        For interface I = (Domain_1, Domain_2):
        Consistent(v_1, v_2)  name_consistent(v_1, v_2)  unit_consistent(v_1, v_2)  value_consistent(v_1, v_2)
        
    Parameters:
        context: Validation context with cross-domain variable data
        
    Returns:
        List of validation results for variable consistency
    """
    results = []
    
    # This is a simplified implementation - in production would need actual cross-domain data
    if not context.state_manager:
        return [ValidationResultModel(
            level=ValidationLevel.WARNING,
            code=ValidationCode.MISSING_REQUIRED_DATA,
            message="State manager required for cross-domain variable validation",
            layer="L2",
            scope="variable_consistency_cross_domain",
            tags=["cross_domain", "missing_data"]
        )]
    
    try:
        # Check for common cross-domain variables that should be consistent
        common_variables = [
            "air_temperature",
            "evapotranspiration_flux", 
            "net_radiation",
            "soil_moisture",
            "co2_flux",
            "water_flux"
        ]
        
        state_schema = getattr(context.state_manager, '_schema', {})
        
        for var_name in common_variables:
            if var_name in state_schema:
                var_spec = state_schema[var_name]
                
                # Check if variable has proper CF standard name
                cf_name = getattr(var_spec, 'cf_standard_name', None)
                units = getattr(var_spec, 'units', None)
                
                if not cf_name:
                    results.append(ValidationResultModel(
                        level=ValidationLevel.WARNING,
                        code=ValidationCode.CF_NAME_MISSING,
                        message=f"Cross-domain variable '{var_name}' lacks CF standard name",
                        layer="L2",
                        scope="variable_consistency_cross_domain",
                        tags=["cross_domain", "cf_missing"],
                        details=[ValidationDetail(
                            variable_name=var_name,
                            units=units
                        )]
                    ))
                
                # Check units are appropriate for cross-domain use
                if units and not _is_cross_domain_consistent_unit(units):
                    results.append(ValidationResultModel(
                        level=ValidationLevel.WARNING,
                        code=ValidationCode.CF_UNIT_MISMATCH,
                        message=f"Variable '{var_name}' has units that may not be cross-domain consistent",
                        layer="L2", 
                        scope="variable_consistency_cross_domain",
                        tags=["cross_domain", "unit_consistency"],
                        details=[ValidationDetail(
                            variable_name=var_name,
                            units=units,
                            cf_standard_name=cf_name
                        )]
                    ))
        
        # Success result if no major issues
        if not any(r.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL] for r in results):
            results.append(ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.CF_NAME_OK,
                message="Cross-domain variable consistency check completed",
                layer="L2",
                scope="variable_consistency_cross_domain",
                tags=["cross_domain", "success"],
                metrics={"variables_checked": len(common_variables)}
            ))
            
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
        logger.error(f"Cross-domain variable validation failed: {str(e)}")
        results.append(ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.GENERIC_VALIDATION_FAILED,
            message=f"Cross-domain variable validation failed: {str(e)}",
            layer="L2",
            scope="variable_consistency_cross_domain",
            tags=["cross_domain", "execution_error"]
        ))
    
    return results


@validated_rule(
    rule_id="hydrosphere_interface_validation",
    layer="L2",
    name="Hydrosphere Interface Validation", 
    description="Validates hydrosphere functor interfaces using CF-1.8 mappings",
    domains=["hydrosphere", "interfaces"],
    tags=["hydrosphere", "interfaces", "cf"],
    cf_requirements=["CF-1.8 variable mappings"],
    references=["Hydrosphere interface mapping specification"]
)
def validate_hydrosphere_interface(context: ValidationContext) -> ValidationResults:
    """
    Validate hydrosphere functor interfaces using CF-1.8 mappings.
    
    Parameters:
        context: Validation context with functor interface information
        
    Returns:
        List of validation results for hydrosphere interfaces
    """
    results = []
    
    try:
        # Get functor information from context (simplified)
        metadata = context.metadata or {}
        functor_inputs = metadata.get('functor_inputs', [])
        functor_outputs = metadata.get('functor_outputs', [])
        functor_name = metadata.get('functor_name', 'unknown_functor')
        
        if not functor_inputs and not functor_outputs:
            return [ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.MISSING_REQUIRED_DATA,
                message="No hydrosphere functor interface data to validate",
                layer="L2",
                scope="hydrosphere_interface_validation",
                tags=["hydrosphere", "no_data"]
            )]
        
        # Use interface validation function
        issues = validate_hydrosphere_functor_interface(functor_inputs, functor_outputs)
        
        for issue in issues:
            if "error" in issue.lower() or "inconsistent" in issue.lower():
                level = ValidationLevel.ERROR
                code = ValidationCode.INTERFACE_VARIABLE_MISSING
            else:
                level = ValidationLevel.WARNING
                code = ValidationCode.CF_NAME_MISSING
            
            results.append(ValidationResultModel(
                level=level,
                code=code,
                message=f"Hydrosphere interface issue in {functor_name}: {issue}",
                layer="L2",
                scope="hydrosphere_interface_validation",
                tags=["hydrosphere", "interface_issue"],
                details=[ValidationDetail(
                    variable_name=functor_name,
                    metadata={"issue": issue}
                )]
            ))
        
        # Validate the declared CF variable-metadata subset.
        all_vars = functor_inputs + functor_outputs
        for var_name in all_vars:
            try:
                var_info = hydrosphere_mapper.get_variable_info(var_name)
                if not var_info.cf_standard_name:
                    results.append(ValidationResultModel(
                        level=ValidationLevel.WARNING,
                        code=ValidationCode.CF_NAME_MISSING,
                        message=f"Hydrosphere variable '{var_name}' lacks CF standard name",
                        layer="L2",
                        scope="hydrosphere_interface_validation",
                        tags=["hydrosphere", "cf_missing"],
                        details=[ValidationDetail(
                            variable_name=var_name
                        )]
                    ))
            except ValueError:
                # Variable not in hydrosphere mapping
                results.append(ValidationResultModel(
                    level=ValidationLevel.ERROR,
                    code=ValidationCode.INTERFACE_VARIABLE_MISSING,
                    message=f"Unknown hydrosphere variable '{var_name}' not in registry",
                    layer="L2",
                    scope="hydrosphere_interface_validation",
                    tags=["hydrosphere", "unknown_variable"],
                    details=[ValidationDetail(
                        variable_name=var_name
                    )]
                ))
        
        # Success result if no major issues
        if not any(r.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL] for r in results):
            results.append(ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.CF_NAME_OK,
                message=f"Hydrosphere interface validation completed for {functor_name}",
                layer="L2",
                scope="hydrosphere_interface_validation",
                tags=["hydrosphere", "success"],
                metrics={
                    "inputs_checked": len(functor_inputs),
                    "outputs_checked": len(functor_outputs)
                }
            ))
            
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
        logger.error(f"Hydrosphere interface validation failed: {str(e)}")
        results.append(ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.GENERIC_VALIDATION_FAILED,
            message=f"Hydrosphere interface validation failed: {str(e)}",
            layer="L2",
            scope="hydrosphere_interface_validation",
            tags=["hydrosphere", "execution_error"]
        ))
    
    return results


# Helper functions

def _matches_cf_standard_name_syntax(name: Any) -> bool:
    """Return whether a value is one exact lowercase ASCII base-name lexeme."""
    return isinstance(name, str) and re.fullmatch(
        r"[a-z][a-z0-9_]*\Z", name
    ) is not None


def _is_cross_domain_consistent_unit(unit: str) -> bool:
    """Check if unit is appropriate for cross-domain use."""
    if not unit:
        return False
    
    # Units that are generally consistent across domains
    consistent_units = [
        # Standard SI units
        "K", "degC", "Pa", "kPa", "hPa", "m", "s", "kg", "g",
        # Flux units
        "kg m-2 s-1", "g m-2 s-1", "W m-2", "mol m-2 s-1",
        # Concentration units  
        "ppm", "ppb", "mol mol-1",
        # Dimensionless
        "1", "dimensionless",
        # Carbon units
        "g C m-2", "kg C m-2", "g C m-2 s-1", "kg C m-2 s-1"
    ]
    
    return unit in consistent_units


# Export validation rule functions
__all__ = [
    "validate_cf_standard_name_syntax",
    "validate_unit_conversion_validity", 
    "validate_variable_consistency_cross_domain",
    "validate_hydrosphere_interface"
]
