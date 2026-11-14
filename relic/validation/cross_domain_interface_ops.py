"""Interface-focused validation helpers for CrossDomainValidator."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple
import warnings

import torch
from torch import Tensor

from ..scientific.hydrosphere.interface_mapping import hydrosphere_mapper
from .framework.rules.cross_domain import (
    validate_hydrosphere_interface as validate_hydrosphere_rule,
    validate_variable_consistency_cross_domain,
)

logger = logging.getLogger(__name__)


def validate_variable_consistency_impl(
    validator,
    source_vars: Dict[str, Tuple[Tensor, str]],
    target_vars: Dict[str, Tuple[Tensor, str]],
    source_domain: Any,
    target_domain: Any,
) -> bool:
    """Validate variable consistency across domain interfaces."""
    validator.validation_errors = []
    validator.validation_warnings = []

    contract = validator._find_contract(source_domain, target_domain)
    if not contract:
        validator.validation_warnings.append(
            f"No interface contract found for {source_domain.value} -> {target_domain.value}"
        )

    context = validator.ValidationContext(
        state_manager=validator.state_manager,
        metadata={
            "source_domain": source_domain.value,
            "target_domain": target_domain.value,
            "source_variables": list(source_vars.keys()),
            "target_variables": list(target_vars.keys()),
        },
    )
    validator._merge_framework_results(
        validate_variable_consistency_cross_domain(context)
    )

    for var_name in source_vars:
        if var_name not in target_vars:
            continue
        source_val, source_unit = source_vars[var_name]
        target_val, target_unit = target_vars[var_name]

        if source_unit != target_unit:
            if validator.unit_converter.can_convert(source_unit, target_unit, variable_name=var_name):
                validator.validation_warnings.append(
                    f"Variable '{var_name}': units differ but convertible "
                    f"({source_unit} -> {target_unit})"
                )
            else:
                validator.validation_errors.append(
                    f"Variable '{var_name}': inconsistent units "
                    f"({source_unit} vs {target_unit})"
                )

        if var_name in validator.variable_registry:
            spec = validator.variable_registry[var_name]
            if spec.bounds:
                min_val, max_val = spec.bounds
                if torch.any(source_val < min_val) or torch.any(source_val > max_val):
                    validator.validation_warnings.append(
                        f"Variable '{var_name}' in {source_domain.value}: "
                        f"values outside expected bounds [{min_val}, {max_val}]"
                    )

    if validator.validation_errors:
        for error in validator.validation_errors:
            logger.error("Cross-domain validation error: %s", error)
        return False

    if validator.validation_warnings:
        for warning in validator.validation_warnings:
            warnings.warn(warning)

    return True


def validate_mass_conservation_impl(
    validator,
    inputs: Dict[str, Tuple[Tensor, str]],
    outputs: Dict[str, Tuple[Tensor, str]],
    storage_change: Optional[Tuple[Tensor, str]],
    element: str,
) -> bool:
    """Validate elemental mass conservation using framework conservation rule."""
    from .framework.rules.conservation import (
        evaluate_element_mass_balance,
    )

    is_conserved, diagnostics = evaluate_element_mass_balance(
        inputs=inputs,
        outputs=outputs,
        storage_change=storage_change,
        element=element,
        unit_converter=validator.unit_converter,
    )

    for issue in diagnostics.get("errors", []):
        validator.validation_errors.append(issue)

    if not is_conserved and not diagnostics.get("errors"):
        validator.validation_errors.append(
            f"{element.capitalize()} conservation violated: "
            f"input={diagnostics['total_input']:.6f}, "
            f"output={diagnostics['total_output']:.6f} "
            f"({diagnostics['target_unit']}), "
            f"relative error={diagnostics['relative_error']:.2e}"
        )

    return is_conserved


def validate_hydrosphere_interfaces_impl(
    validator,
    functor_inputs,
    functor_outputs,
    functor_name: str,
) -> bool:
    """Validate hydrosphere interfaces via framework and registry checks."""
    validator.validation_errors = []
    validator.validation_warnings = []

    context = validator.ValidationContext(
        state_manager=validator.state_manager,
        metadata={
            "functor_inputs": functor_inputs,
            "functor_outputs": functor_outputs,
            "functor_name": functor_name,
        },
    )
    validator._merge_framework_results(validate_hydrosphere_rule(context))

    all_vars = functor_inputs + functor_outputs
    for var_name in all_vars:
        try:
            var_info = hydrosphere_mapper.get_variable_info(var_name)
            if not var_info.cf_standard_name:
                validator.validation_warnings.append(
                    f"{functor_name}: Variable '{var_name}' lacks CF standard name"
                )
        except ValueError:
            if var_name not in validator.variable_registry:
                validator.validation_errors.append(
                    f"{functor_name}: Unknown variable '{var_name}' not in any registry"
                )

    if validator.validation_errors:
        for error in validator.validation_errors:
            logger.error("Cross-domain validation error: %s", error)
        return False

    if validator.validation_warnings:
        for warning in validator.validation_warnings:
            warnings.warn(warning)

    return True


__all__ = [
    "validate_variable_consistency_impl",
    "validate_mass_conservation_impl",
    "validate_hydrosphere_interfaces_impl",
]
