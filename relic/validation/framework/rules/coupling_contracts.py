"""
Coupling Contract Validation Rules

This module contains validation rules for inter-domain coupling contracts,
ensuring that interfaces between different scientific domains (atmosphere,
biosphere, hydrosphere, pedosphere) maintain consistency and physical validity.

Key Coupling Contracts:
    - Biosphere-Atmosphere coupling interfaces
    - Soil-Vegetation coupling interfaces  
    - Water-Carbon coupling interfaces
    - Stress factor bounds validation
    - Interface variable consistency
    - Coupling conservation laws

Mathematical Framework:
    For coupling C between domains D_1 and D_2:
    Interface(C) = {v in D_1 : v is required by D_2} union {v in D_2 : v is required by D_1}
    
    Contract validation:
    - Type consistency: type(v_D_1) = type(v_D_2) for shared variables v
    - Unit consistency: units(v_D_1) consistent with units(v_D_2)
    - Range consistency: range(v_D_1) intersection range(v_D_2) != 
    - Conservation: suminputs = sumoutputs for conserved quantities

References:
    - RELIC coupling specification
    - Domain interface requirements
    - Inter-domain conservation principles
    - Coupling functor contracts
"""

import logging

import torch
from torch import Tensor

from ..models import (
    ValidationResultModel, ValidationLevel, ValidationCode,
    ValidationContext, ValidationResults, ValidationDetail
)
from ..decorators import validated_rule
from .conservation import domain_absent_result
from ....coupling.core.conservation_checker import reduction_rounding_bound

logger = logging.getLogger(__name__)

# Import coupling components
from ....data.standards.unit_converter import unit_converter
from ...cross_domain_coupling_ops import (
    check_energy_conservation_impl,
    check_stress_factor_bounds_impl,
    check_water_conservation_impl,
)

HAS_COUPLING_SYSTEM = True


@validated_rule(
    rule_id="biosphere_atmosphere_coupling_validation",
    layer="L2",
    name="Biosphere-Atmosphere Coupling Validation",
    description="Validates coupling contracts between biosphere and atmosphere domains",
    domains=["biosphere", "atmosphere", "coupling"],
    tags=["coupling", "biosphere_atmosphere", "interface"],
    dependencies=["water_mass_conservation", "energy_balance_conservation"],
    references=["Biosphere-Atmosphere coupling specification"]
)
def validate_biosphere_atmosphere_coupling(context: ValidationContext) -> ValidationResults:
    """
    Validate biosphere-atmosphere coupling contracts.
    
    Mathematical Expression:
        Biosphere-atmosphere interface requires:
        - Energy balance: Rn = H + LE + G
        - Water balance: ET_atm = T_bio + E_soil  
        - CO2 exchange: NEE_atm = -GPP_bio + Ra_bio + Rh_soil
        
    Parameters:
        context: Validation context with coupling data
        
    Returns:
        List of validation results for biosphere-atmosphere coupling
    """
    results = []
    
    try:
        # Extract coupling data from context
        metadata = context.metadata or {}
        coupling_data = metadata.get('biosphere_atmosphere_coupling', {})
        
        if not coupling_data:
            return [domain_absent_result(
                "Biosphere-atmosphere coupling",
                "biosphere_atmosphere_coupling_validation",
                ["coupling", "biosphere_atmosphere"],
                "biosphere_atmosphere_coupling",
            )]
        
        # Extract interface variables
        input_state = coupling_data.get('input_state', {})
        output_fluxes = coupling_data.get('output_fluxes', {})
        
        # Validate required variables are present
        required_inputs = [
            'net_radiation', 'air_temperature', 'vapor_pressure_deficit',
            'atmospheric_pressure', 'wind_speed'
        ]
        required_outputs = [
            'evapotranspiration_flux', 'sensible_heat_flux', 'latent_heat_flux',
            'co2_flux', 'transpiration_rate'
        ]
        
        missing_inputs = [var for var in required_inputs if var not in input_state]
        missing_outputs = [var for var in required_outputs if var not in output_fluxes]
        
        if missing_inputs:
            results.append(ValidationResultModel(
                level=ValidationLevel.ERROR,
                code=ValidationCode.INTERFACE_VARIABLE_MISSING,
                message=f"Missing required input variables: {', '.join(missing_inputs)}",
                layer="L2",
                scope="biosphere_atmosphere_coupling_validation",
                tags=["coupling", "biosphere_atmosphere", "missing_inputs"],
                details=[ValidationDetail(
                    variable_name="missing_inputs",
                    value=', '.join(missing_inputs),
                    expected=', '.join(required_inputs)
                )]
            ))
        
        if missing_outputs:
            results.append(ValidationResultModel(
                level=ValidationLevel.ERROR,
                code=ValidationCode.INTERFACE_VARIABLE_MISSING,
                message=f"Missing required output variables: {', '.join(missing_outputs)}",
                layer="L2",
                scope="biosphere_atmosphere_coupling_validation",
                tags=["coupling", "biosphere_atmosphere", "missing_outputs"],
                details=[ValidationDetail(
                    variable_name="missing_outputs",
                    value=', '.join(missing_outputs),
                    expected=', '.join(required_outputs)
                )]
            ))
        
        # Validate coupling-specific conservation laws
        conservation_checks = []
        
        # Reuse the canonical runtime contract so rules cannot drift from execution.
        water_check = check_water_conservation_impl(
            "biosphere_atmosphere",
            input_state,
            output_fluxes,
        )
        conservation_checks.append({
            'name': 'water_balance',
            'passed': water_check['passed'],
            'evaluable': water_check['checked'],
            'residual': water_check['relative_residual'],
            'tolerance': water_check['tolerance'],
            'metrics': (
                {'water_balance_residual': water_check['relative_residual']}
                if water_check['relative_residual'] is not None
                else {}
            ),
            'message': water_check['message'] or "Water balance satisfied",
        })

        # 2. Energy balance consistency
        energy_check = check_energy_conservation_impl(input_state, output_fluxes)
        conservation_checks.append({
            'name': 'energy_balance',
            'passed': energy_check['passed'],
            'evaluable': energy_check['checked'],
            'residual': energy_check['relative_residual'],
            'tolerance': energy_check['tolerance'],
            'metrics': (
                {'energy_balance_residual': energy_check['relative_residual']}
                if energy_check['relative_residual'] is not None
                else {}
            ),
            'message': energy_check['message'] or "Energy balance satisfied",
        })

        # 3. CO2 flux diagnostic. Flux direction is state-dependent, but values
        # must be finite and non-empty before they can be reported.
        if 'co2_flux' in output_fluxes:
            co2_flux = torch.as_tensor(output_fluxes['co2_flux'], dtype=torch.float64)
            co2_valid = co2_flux.numel() > 0 and bool(
                torch.all(torch.isfinite(co2_flux))
            )
            co2_mean = (
                float(co2_flux.mean().detach().cpu().item())
                if co2_valid
                else float("nan")
            )
            conservation_checks.append({
                'name': 'co2_flux_finite',
                'passed': co2_valid,
                'evaluable': True,
                # A finiteness check: it has no residual and no tolerance.
                'residual': None,
                'tolerance': None,
                'metrics': {'co2_flux_mean': co2_mean} if co2_valid else {},
                'message': (
                    f"CO2 flux mean: {co2_mean:.6e} g C m-2 s-1"
                    if co2_valid
                    else "CO2 flux must be finite and non-empty"
                ),
            })
        
        # Report conservation check results; the water and energy residuals are the relative
        # residuals their budget checks compared with the budget keys.
        for check in conservation_checks:
            tolerance = check["tolerance"]
            if not check['passed']:
                verdict = "violation" if check['evaluable'] else "not evaluable"
                results.append(ValidationResultModel(
                    level=ValidationLevel.ERROR,
                    code=(
                        ValidationCode.COUPLING_CONTRACT_VIOLATION
                        if check['evaluable']
                        else ValidationCode.MISSING_REQUIRED_DATA
                    ),
                    message=f"Biosphere-atmosphere coupling {verdict} ({check['name']}): {check['message']}",
                    layer="L2",
                    scope="biosphere_atmosphere_coupling_validation",
                    tags=["coupling", "biosphere_atmosphere", check['name']],
                    details=[ValidationDetail(
                        variable_name=check['name'],
                        value=check['residual'],
                        threshold=tolerance
                    )]
                ))
            else:
                results.append(ValidationResultModel(
                    level=ValidationLevel.INFO,
                    code=ValidationCode.COUPLING_CONTRACT_OK,
                    message=f"Biosphere-atmosphere coupling check passed ({check['name']}): {check['message']}",
                    layer="L2",
                    scope="biosphere_atmosphere_coupling_validation",
                    tags=["coupling", "biosphere_atmosphere", check['name'], "success"],
                    metrics=check['metrics']
                ))
        
    # Boundary of this rule (RULES 2): a term that is not tensor-convertible, not
    # finite or not shape-compatible, a refused tolerance override or a failed unit
    # conversion (TypeError, ValueError, RuntimeError) is reported as one explicit
    # failed result; verdicts appended before the failure are dropped, so no result
    # of this rule can read as success. Context-schema and configuration faults
    # (KeyError, AttributeError) propagate to the validated_rule boundary.
    except (TypeError, ValueError, RuntimeError) as e:
        logger.error(f"Biosphere-atmosphere coupling validation failed: {str(e)}")
        return [ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.GENERIC_VALIDATION_FAILED,
            message=f"Biosphere-atmosphere coupling validation failed: {str(e)}",
            passed=False,
            layer="L2",
            scope="biosphere_atmosphere_coupling_validation",
            tags=["coupling", "biosphere_atmosphere", "execution_error"]
        )]
    
    return results


@validated_rule(
    rule_id="soil_vegetation_coupling_validation",
    layer="L2", 
    name="Soil-Vegetation Coupling Validation",
    description="Validates coupling contracts between soil and vegetation domains",
    domains=["pedosphere", "biosphere", "coupling"],
    tags=["coupling", "soil_vegetation", "interface"],
    references=["Soil-Vegetation coupling specification"]
)
def validate_soil_vegetation_coupling(context: ValidationContext) -> ValidationResults:
    """
    Validate soil-vegetation coupling contracts.
    
    Mathematical Expression:
        Soil-vegetation interface requires:
        - Water uptake: sum(root_uptake_i x Delta z_i) = effective_transpiration, with
          Delta z_i = z_i - z_(i-1) from the layer-bottom depths z (z_0 = 0)
        - Nutrient balance: N_uptake <= N_available
        - Root distribution: sumroot_fraction_i = 1
        
    Parameters:
        context: Validation context with coupling data
        
    Returns:
        List of validation results for soil-vegetation coupling
    """
    results = []
    
    try:
        # Extract coupling data from context
        metadata = context.metadata or {}
        coupling_data = metadata.get('soil_vegetation_coupling', {})
        
        if not coupling_data:
            return [domain_absent_result(
                "Soil-vegetation coupling",
                "soil_vegetation_coupling_validation",
                ["coupling", "soil_vegetation"],
                "soil_vegetation_coupling",
            )]
        
        input_state = coupling_data.get('input_state', {})
        output_fluxes = coupling_data.get('output_fluxes', {})
        
        # Reuse the canonical layered uptake contract and its fail-closed checks.
        water_check = check_water_conservation_impl(
            "soil_vegetation",
            input_state,
            output_fluxes,
        )
        if water_check["passed"]:
            results.append(ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.COUPLING_CONTRACT_OK,
                message=(
                    "Soil-vegetation water balance satisfied: "
                    f"residual = {water_check['residual']:.6e}"
                ),
                layer="L2",
                scope="soil_vegetation_coupling_validation",
                tags=["coupling", "soil_vegetation", "water_balance"],
                metrics={"water_balance_residual": water_check["residual"]},
            ))
        else:
            results.append(ValidationResultModel(
                level=ValidationLevel.ERROR,
                code=(
                    ValidationCode.COUPLING_CONTRACT_VIOLATION
                    if water_check["checked"]
                    else ValidationCode.MISSING_REQUIRED_DATA
                ),
                message=(
                    "Soil-vegetation water balance "
                    f"{'violation' if water_check['checked'] else 'not evaluable'}: "
                    f"{water_check['message']}"
                ),
                layer="L2",
                scope="soil_vegetation_coupling_validation",
                tags=["coupling", "soil_vegetation", "water_balance"],
                details=[ValidationDetail(
                    variable_name="water_uptake_relative_residual",
                    value=water_check["relative_residual"],
                    threshold=water_check["tolerance"],
                    units="dimensionless",
                )],
            ))
        
        # Validate root fraction normalization
        if 'root_fraction_profile' in output_fluxes:
            root_fractions = output_fluxes['root_fraction_profile']
            if not isinstance(root_fractions, Tensor):
                root_fractions = torch.as_tensor(root_fractions, dtype=torch.float64)
            elif not root_fractions.dtype.is_floating_point:
                root_fractions = root_fractions.to(torch.float64)

            if root_fractions.numel() == 0 or not bool(
                torch.all(torch.isfinite(root_fractions))
            ):
                results.append(ValidationResultModel(
                    level=ValidationLevel.ERROR,
                    code=ValidationCode.COUPLING_CONTRACT_VIOLATION,
                    message="Root fractions must be finite and non-empty",
                    layer="L2",
                    scope="soil_vegetation_coupling_validation",
                    tags=[
                        "coupling",
                        "soil_vegetation",
                        "root_fraction",
                        "invalid_values",
                    ],
                ))
            else:
                # References:
                #   Jackson et al. (1996), Oecologia 108:389-411 (root fractions of a profile);
                #   Higham (2002), Accuracy and Stability of Numerical Algorithms, section 4.2
                #   (reduction_rounding_bound, the rounding term of ledger_acceptance).
                # Mathematical expression:
                #   s = sum_i f_i; the profile is normalized iff
                #   |s - 1| <= gamma_K(u) * (sum_i |f_i| + 1) with K = 2 (n + 1), and f_i >= 0.
                # Parameter meanings:
                #   f_i: root fraction of soil layer i; n: number of layers; u: unit roundoff of the
                #   dtype of f (binary64 for Python numbers); gamma_K: reduction_rounding_bound.
                # Units: f_i, s and the bound are dimensionless [1].
                # Domain/constraints: finite, non-empty fractions. The RELIC root profile
                #   (SoilVegetationCoupling.compute_root_distribution) is divided by its depth
                #   integral, so sum_i f_i = 1 holds exactly and only the rounding of that
                #   normalization and of this sum is admitted; a truncated profile is reported
                #   whatever the size of its deficit. No tolerance parameter or literal applies.
                # Sign convention: f_i >= 0 (negative fractions are reported below).
                fraction_sum = torch.sum(root_fractions)
                deviation = float(torch.abs(fraction_sum - 1.0))
                rounding_bound = reduction_rounding_bound(
                    root_fractions.dtype, 2 * (root_fractions.numel() + 1)
                ) * (float(torch.sum(torch.abs(root_fractions))) + 1.0)

                if deviation > rounding_bound:
                    results.append(ValidationResultModel(
                        level=ValidationLevel.WARNING,
                        code=ValidationCode.COUPLING_CONTRACT_VIOLATION,
                        message=(
                            f"Root fractions do not sum to 1: sum = {float(fraction_sum):.6f}, "
                            f"deviation {deviation:.3e} exceeds the rounding bound "
                            f"{rounding_bound:.3e}"
                        ),
                        layer="L2",
                        scope="soil_vegetation_coupling_validation",
                        tags=["coupling", "soil_vegetation", "root_fraction"],
                        details=[ValidationDetail(
                            variable_name="root_fraction_sum",
                            value=float(fraction_sum),
                            expected=1.0,
                            threshold=rounding_bound,
                            units="dimensionless"
                        )]
                    ))
            
                # Check for negative root fractions
                if torch.any(root_fractions < 0):
                    results.append(ValidationResultModel(
                        level=ValidationLevel.ERROR,
                        code=ValidationCode.COUPLING_CONTRACT_VIOLATION,
                        message="Negative root fractions detected",
                        layer="L2",
                        scope="soil_vegetation_coupling_validation",
                        tags=["coupling", "soil_vegetation", "negative_fraction"]
                    ))
        
        # Validate nutrient uptake feasibility
        if all(var in output_fluxes for var in ['nitrogen_uptake', 'phosphorus_uptake']):
            n_uptake = output_fluxes['nitrogen_uptake']
            p_uptake = output_fluxes['phosphorus_uptake']
            
            # Check for negative uptake
            nutrient_warnings = []
            if torch.any(torch.tensor(n_uptake) < 0):
                nutrient_warnings.append("negative nitrogen uptake")
            if torch.any(torch.tensor(p_uptake) < 0):
                nutrient_warnings.append("negative phosphorus uptake")
            
            if nutrient_warnings:
                results.append(ValidationResultModel(
                    level=ValidationLevel.WARNING,
                    code=ValidationCode.COUPLING_CONTRACT_VIOLATION,
                    message=f"Unrealistic nutrient uptake: {', '.join(nutrient_warnings)}",
                    layer="L2",
                    scope="soil_vegetation_coupling_validation",
                    tags=["coupling", "soil_vegetation", "nutrient_uptake"]
                ))
        
        # Summary result if no major issues
        if not any(r.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL] for r in results):
            results.append(ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.COUPLING_CONTRACT_OK,
                message="Soil-vegetation coupling validation completed successfully",
                layer="L2",
                scope="soil_vegetation_coupling_validation",
                tags=["coupling", "soil_vegetation", "success"],
                metrics={"coupling_checks_passed": True}
            ))
        
    # Boundary of this rule (RULES 2): a term that is not tensor-convertible, not
    # finite or not shape-compatible, a refused tolerance override or a failed unit
    # conversion (TypeError, ValueError, RuntimeError) is reported as one explicit
    # failed result; verdicts appended before the failure are dropped, so no result
    # of this rule can read as success. Context-schema and configuration faults
    # (KeyError, AttributeError) propagate to the validated_rule boundary.
    except (TypeError, ValueError, RuntimeError) as e:
        logger.error(f"Soil-vegetation coupling validation failed: {str(e)}")
        return [ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.GENERIC_VALIDATION_FAILED,
            message=f"Soil-vegetation coupling validation failed: {str(e)}",
            passed=False,
            layer="L2",
            scope="soil_vegetation_coupling_validation",
            tags=["coupling", "soil_vegetation", "execution_error"]
        )]
    
    return results


@validated_rule(
    rule_id="stress_factor_bounds_validation",
    layer="L2",
    name="Stress Factor Bounds Validation",
    description="Validates that all stress factors are within [0, 1] bounds",
    domains=["coupling", "stress_factors"],
    tags=["coupling", "stress_factors", "bounds"],
    references=["Stress factor specification: all factors in [0, 1]"]
)
def validate_stress_factor_bounds(context: ValidationContext) -> ValidationResults:
    """
    Validate stress factor bounds.
    
    Mathematical Expression:
        For all stress factors s_i: 0 <= s_i <= 1
        where s_i represents various environmental stress limitations
        
    Parameters:
        context: Validation context with stress factor data
        
    Returns:
        List of validation results for stress factor bounds
    """
    results = []
    
    try:
        # Extract stress factor data from context
        metadata = context.metadata or {}
        stress_factors = metadata.get('stress_factors', {})
        
        if not stress_factors:
            return [ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.MISSING_REQUIRED_DATA,
                message="No stress factor data available for bounds validation",
                layer="L2",
                scope="stress_factor_bounds_validation",
                tags=["coupling", "stress_factors", "no_data"]
            )]
        
        # Reuse the canonical runtime bounds contract, including finite and
        # non-empty checks, so the framework rule cannot report false success.
        bounds_check = check_stress_factor_bounds_impl(stress_factors)
        total_factors = len(stress_factors)

        if not bounds_check["checked"]:
            results.append(ValidationResultModel(
                level=ValidationLevel.WARNING,
                code=ValidationCode.MISSING_REQUIRED_DATA,
                message="No recognized stress factors found for validation",
                layer="L2",
                scope="stress_factor_bounds_validation",
                tags=["coupling", "stress_factors", "none_found"]
            ))
        elif bounds_check["passed"]:
            results.append(ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.STRESS_FACTOR_OK,
                message=f"All {total_factors} stress factors within bounds [0, 1]",
                layer="L2",
                scope="stress_factor_bounds_validation",
                tags=["coupling", "stress_factors", "success"],
                metrics={
                    "total_stress_factors": total_factors,
                    "bounds_violations": 0,
                }
            ))
        else:
            for violation in bounds_check["violations"]:
                results.append(ValidationResultModel(
                    level=ValidationLevel.ERROR,
                    code=ValidationCode.STRESS_FACTOR_OUT_OF_BOUNDS,
                    message=f"Stress factor contract violation: {violation}",
                    layer="L2",
                    scope="stress_factor_bounds_validation",
                    tags=["coupling", "stress_factors", "bounds_violation"],
                ))
            results.append(ValidationResultModel(
                level=ValidationLevel.ERROR,
                code=ValidationCode.STRESS_FACTOR_OUT_OF_BOUNDS,
                message=(
                    f"{len(bounds_check['violations'])}/{total_factors} "
                    "stress factor checks violate the contract"
                ),
                layer="L2",
                scope="stress_factor_bounds_validation",
                tags=["coupling", "stress_factors", "summary"],
                metrics={
                    "total_stress_factors": total_factors,
                    "bounds_violations": len(bounds_check["violations"]),
                    "violation_rate": (
                        len(bounds_check["violations"]) / max(total_factors, 1)
                    ),
                }
            ))
        
    # Boundary of this rule (RULES 2): a term that is not tensor-convertible, not
    # finite or not shape-compatible, a refused tolerance override or a failed unit
    # conversion (TypeError, ValueError, RuntimeError) is reported as one explicit
    # failed result; verdicts appended before the failure are dropped, so no result
    # of this rule can read as success. Context-schema and configuration faults
    # (KeyError, AttributeError) propagate to the validated_rule boundary.
    except (TypeError, ValueError, RuntimeError) as e:
        logger.error(f"Stress factor bounds validation failed: {str(e)}")
        return [ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.GENERIC_VALIDATION_FAILED,
            message=f"Stress factor bounds validation failed: {str(e)}",
            passed=False,
            layer="L2",
            scope="stress_factor_bounds_validation",
            tags=["coupling", "stress_factors", "execution_error"]
        )]
    
    return results


@validated_rule(
    rule_id="interface_type_consistency_validation",
    layer="L2",
    name="Interface Type Consistency Validation",
    description="Validates type consistency across coupling interfaces",
    domains=["coupling", "interfaces"],
    tags=["coupling", "interfaces", "type_consistency"],
    references=["Domain interface type specifications"]
)
def validate_interface_type_consistency(context: ValidationContext) -> ValidationResults:
    """
    Validate type consistency across coupling interfaces.
    
    Parameters:
        context: Validation context with interface data
        
    Returns:
        List of validation results for interface type consistency
    """
    results = []
    
    try:
        # Extract interface specifications from context
        metadata = context.metadata or {}
        interfaces = metadata.get('coupling_interfaces', {})
        
        if not interfaces:
            return [ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.MISSING_REQUIRED_DATA,
                message="No coupling interface data available for type validation",
                layer="L2",
                scope="interface_type_consistency_validation",
                tags=["coupling", "interfaces", "no_data"]
            )]
        
        contract_mismatches = 0
        total_interfaces = 0
        checked_shared_variables = 0
        unchecked_interfaces = 0
        
        for interface_name, interface_spec in interfaces.items():
            total_interfaces += 1
            
            # Extract source and target variable specifications
            source_vars = interface_spec.get('source_variables', {})
            target_vars = interface_spec.get('target_variables', {})
            
            # Check shared variables for type consistency
            shared_vars = set(source_vars.keys()) & set(target_vars.keys())

            if not shared_vars:
                unchecked_interfaces += 1
                results.append(ValidationResultModel(
                    level=ValidationLevel.WARNING,
                    code=ValidationCode.MISSING_REQUIRED_DATA,
                    message=(
                        f"No shared variables found for coupling interface "
                        f"'{interface_name}'"
                    ),
                    layer="L2",
                    scope="interface_type_consistency_validation",
                    tags=["coupling", "interfaces", "no_shared_variables"],
                ))
                continue

            checked_shared_variables += len(shared_vars)
            
            for var_name in sorted(shared_vars):
                source_spec = source_vars[var_name]
                target_spec = target_vars[var_name]

                # A shared variable is compared only on declared dtype, shape and units; an
                # undeclared field is not replaced by a default (such as float32, a scalar
                # shape or dimensionless units).
                undeclared = sorted(
                    f"{side}.{field}"
                    for side, spec in (("source", source_spec), ("target", target_spec))
                    for field in ("dtype", "shape", "units")
                    if field not in spec
                )
                if undeclared:
                    contract_mismatches += 1
                    results.append(ValidationResultModel(
                        level=ValidationLevel.ERROR,
                        code=ValidationCode.MISSING_REQUIRED_DATA,
                        message=(
                            f"Interface {interface_name} variable '{var_name}' is not evaluable: "
                            f"undeclared {', '.join(undeclared)}"
                        ),
                        passed=False,
                        layer="L2",
                        scope="interface_type_consistency_validation",
                        tags=["coupling", "interfaces", "undeclared_contract"],
                        details=[ValidationDetail(
                            variable_name=var_name,
                            value=None,
                            metadata={"interface": interface_name, "undeclared": undeclared},
                        )],
                    ))
                    continue

                # Check data type consistency
                source_type = source_spec['dtype']
                target_type = target_spec['dtype']

                if source_type != target_type:
                    contract_mismatches += 1
                    results.append(ValidationResultModel(
                        level=ValidationLevel.WARNING,
                        code=ValidationCode.INTERFACE_TYPE_MISMATCH,
                        message=f"Type mismatch in {interface_name} for '{var_name}': {source_type} -> {target_type}",
                        layer="L2",
                        scope="interface_type_consistency_validation",
                        tags=["coupling", "interfaces", "type_mismatch"],
                        details=[ValidationDetail(
                            variable_name=var_name,
                            value=f"{source_type} -> {target_type}",
                            expected="consistent types",
                            metadata={
                                "interface": interface_name,
                                "source_type": source_type,
                                "target_type": target_type
                            }
                        )]
                    ))

                # Check shape consistency (an empty shape declares a scalar)
                source_shape = list(source_spec['shape'])
                target_shape = list(target_spec['shape'])

                if source_shape != target_shape:
                    contract_mismatches += 1
                    results.append(ValidationResultModel(
                        level=ValidationLevel.ERROR,
                        code=ValidationCode.INTERFACE_TYPE_MISMATCH,
                        message=f"Shape mismatch in {interface_name} for '{var_name}': {source_shape} -> {target_shape}",
                        layer="L2",
                        scope="interface_type_consistency_validation",
                        tags=["coupling", "interfaces", "shape_mismatch"]
                    ))

                # Check units consistency
                source_units = source_spec['units']
                target_units = target_spec['units']

                if source_units != target_units:
                    # Check if units are convertible
                    if not unit_converter.can_convert(source_units, target_units, variable_name=var_name):
                        contract_mismatches += 1
                        results.append(ValidationResultModel(
                            level=ValidationLevel.ERROR,
                            code=ValidationCode.CF_UNIT_MISMATCH,
                            message=f"inconsistent units in {interface_name} for '{var_name}': {source_units} vs {target_units}",
                            layer="L2",
                            scope="interface_type_consistency_validation",
                            tags=["coupling", "interfaces", "unit_mismatch"]
                        ))
                    else:
                        results.append(ValidationResultModel(
                            level=ValidationLevel.INFO,
                            code=ValidationCode.CF_UNIT_MISMATCH,
                            message=f"Unit conversion required in {interface_name} for '{var_name}': {source_units} -> {target_units}",
                            layer="L2",
                            scope="interface_type_consistency_validation",
                            tags=["coupling", "interfaces", "unit_conversion"]
                        ))

        # Summary result
        if total_interfaces == 0:
            results.append(ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.MISSING_REQUIRED_DATA,
                message="No coupling interfaces found for type consistency validation",
                layer="L2",
                scope="interface_type_consistency_validation",
                tags=["coupling", "interfaces", "none_found"]
            ))
        elif (
            contract_mismatches == 0
            and checked_shared_variables > 0
            and unchecked_interfaces == 0
        ):
            results.append(ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.INTERFACE_TYPE_OK,
                message=f"Type consistency validated for {total_interfaces} interfaces",
                layer="L2",
                scope="interface_type_consistency_validation",
                tags=["coupling", "interfaces", "success"],
                metrics={
                    "total_interfaces": total_interfaces,
                    "checked_shared_variables": checked_shared_variables,
                    "contract_mismatches": contract_mismatches,
                }
            ))
        
    # Boundary of this rule (RULES 2): a term that is not tensor-convertible, not
    # finite or not shape-compatible, a refused tolerance override or a failed unit
    # conversion (TypeError, ValueError, RuntimeError) is reported as one explicit
    # failed result; verdicts appended before the failure are dropped, so no result
    # of this rule can read as success. Context-schema and configuration faults
    # (KeyError, AttributeError) propagate to the validated_rule boundary.
    except (TypeError, ValueError, RuntimeError) as e:
        logger.error(f"Interface type consistency validation failed: {str(e)}")
        return [ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.GENERIC_VALIDATION_FAILED,
            message=f"Interface type consistency validation failed: {str(e)}",
            passed=False,
            layer="L2",
            scope="interface_type_consistency_validation",
            tags=["coupling", "interfaces", "execution_error"]
        )]
    
    return results


# Export validation rule functions
__all__ = [
    "validate_biosphere_atmosphere_coupling",
    "validate_soil_vegetation_coupling",
    "validate_stress_factor_bounds", 
    "validate_interface_type_consistency"
]
