"""
Biosphere PFT Parameter Validation System

This module provides comprehensive validation of Plant Functional Type (PFT)
parameters to ensure scientific consistency, physical realism, and numerical
stability for ecosystem model simulations.

Features
--------
- Range validation against literature values
- Cross-parameter consistency checking
- Temperature response validation  
- Allocation constraint enforcement
- BVOC emission factor verification
- Ozone sensitivity parameter validation
- Warning system for unusual but acceptable values

Validation Levels
-----------------
- STRICT: Enforce tight literature-based ranges, raise errors
- MODERATE: Allow broader ranges, issue warnings
- PERMISSIVE: Only check for physically impossible values

References
----------
- Kattge, J. et al. (2011). TRY - a global database of plant traits
- Wullschleger, S.D. (1993). Biochemical limitations to carbon assimilation
- Wright, I.J. et al. (2004). The worldwide leaf economics spectrum
- Rogers, A. et al. (2017). A roadmap for improving photosynthesis representation
- Guenther, A.B. et al. (2012). MEGAN2.1 emission factors
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

# Import PFT types
from relic.scientific.biosphere.pft.pft_types import (
    PFTDefinition, 
    PhotosynthesisPathway
)

logger = logging.getLogger(__name__)


class PFTValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"          # Tight literature ranges, errors on violations
    MODERATE = "moderate"      # Broader ranges, warnings for violations  
    PERMISSIVE = "permissive"  # Only physical impossibilities cause errors


@dataclass 
class PFTValidationResult:
    """Result of PFT parameter validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    parameter_checks: Dict[str, bool]
    
    def add_error(self, message: str) -> None:
        """Add validation error."""
        self.errors.append(message)
        self.is_valid = False
        
    def add_warning(self, message: str) -> None:
        """Add validation warning.""" 
        self.warnings.append(message)
    
    def merge(self, other: 'PFTValidationResult') -> None:
        """Merge another validation result."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.parameter_checks.update(other.parameter_checks)
        if not other.is_valid:
            self.is_valid = False


# Backward-compatible aliases. New code should use the PFT-specific names to
# avoid confusion with the framework-wide validation result and severity types.
ValidationLevel = PFTValidationLevel
ValidationResult = PFTValidationResult


@dataclass
class ParameterRange:
    """Valid range for a parameter with contextual information."""
    min_value: float
    max_value: float
    typical_min: float
    typical_max: float
    units: str
    description: str
    references: List[str]
    
    def check_value(self, value: float, validation_level: ValidationLevel) -> Tuple[bool, Optional[str]]:
        """
        Check if value is within acceptable range.
        
        Returns
        -------
        Tuple[bool, Optional[str]]
            (is_acceptable, error_or_warning_message)
        """
        if value < self.min_value or value > self.max_value:
            return False, f"Value {value} outside physical range [{self.min_value}, {self.max_value}] {self.units}"
        
        if validation_level == ValidationLevel.STRICT:
            if value < self.typical_min or value > self.typical_max:
                return False, f"Value {value} outside typical range [{self.typical_min}, {self.typical_max}] {self.units}"
        
        if validation_level == ValidationLevel.MODERATE:
            if value < self.typical_min or value > self.typical_max:
                return True, f"Value {value} outside typical range [{self.typical_min}, {self.typical_max}] {self.units}"
        
        return True, None


class BiospherePFTValidator:
    """
    Comprehensive PFT parameter validator for biosphere processes.
    
    This class provides methods to validate all aspects of PFT parameter
    definitions, ensuring scientific consistency and numerical stability.
    
    Features
    --------
    - Literature-based parameter range validation
    - Cross-parameter consistency checking
    - Pathway-specific validation (C3/C4/CAM)
    - Temperature response curve validation
    - Allocation constraint enforcement
    - BVOC emission factor verification
    
    Examples
    --------
    >>> validator = BiospherePFTValidator(ValidationLevel.MODERATE)
    >>> result = validator.validate_pft(pft_definition)
    >>> if result.is_valid:
    ...     print("PFT parameters are valid")
    >>> else:
    ...     print(f"Validation errors: {result.errors}")
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        """
        Initialize PFT validator.
        
        Parameters
        ----------
        validation_level : ValidationLevel
            Strictness level for validation
        """
        self.validation_level = validation_level
        self._parameter_ranges = self._initialize_parameter_ranges()
        self._consistency_rules = self._initialize_consistency_rules()
    
    def _initialize_parameter_ranges(self) -> Dict[str, ParameterRange]:
        """Initialize parameter ranges based on literature values."""
        ranges = {}
        
        # Photosynthesis parameters
        ranges['vcmax25'] = ParameterRange(
            min_value=5.0, max_value=300.0,
            typical_min=20.0, typical_max=150.0,
            units="mumol m^2 s^-1",
            description="Maximum carboxylation rate at 25degC",
            references=["Wullschleger 1993", "Kattge et al. 2011"]
        )
        
        ranges['jmax25'] = ParameterRange(
            min_value=10.0, max_value=600.0, 
            typical_min=40.0, typical_max=350.0,
            units="mumol m^2 s^-1", 
            description="Maximum electron transport rate at 25degC",
            references=["Wullschleger 1993", "Medlyn et al. 2002"]
        )
        
        ranges['rd25'] = ParameterRange(
            min_value=0.0, max_value=20.0,
            typical_min=0.5, typical_max=5.0,
            units="mumol m^2 s^-1",
            description="Dark respiration rate at 25degC", 
            references=["Atkin et al. 2015"]
        )
        
        # Activation energies (J mol^-1)
        ranges['vcmax_ea'] = ParameterRange(
            min_value=40000.0, max_value=90000.0,
            typical_min=58000.0, typical_max=75000.0,
            units="J mol^-1",
            description="Vcmax activation energy",
            references=["Bernacchi et al. 2001", "Kattge & Knorr 2007"]
        )
        
        ranges['jmax_ea'] = ParameterRange(
            min_value=25000.0, max_value=60000.0,
            typical_min=35000.0, typical_max=50000.0,
            units="J mol^-1",
            description="Jmax activation energy", 
            references=["Bernacchi et al. 2001"]
        )
        
        # Light response parameters
        ranges['alpha'] = ParameterRange(
            min_value=0.1, max_value=0.4,
            typical_min=0.2, typical_max=0.3,
            units="mol e mol^-1 photon",
            description="Quantum yield of electron transport",
            references=["Bjrkman & Demmig 1987"]
        )
        
        ranges['theta'] = ParameterRange(
            min_value=0.5, max_value=1.0,
            typical_min=0.65, typical_max=0.85,
            units="dimensionless",
            description="Light response curvature parameter",
            references=["Marshall & Biscoe 1980"]
        )
        
        # Stomatal parameters
        ranges['g0'] = ParameterRange(
            min_value=0.0, max_value=0.1,
            typical_min=0.005, typical_max=0.03,
            units="mol m^2 s^-1",
            description="Minimum stomatal conductance",
            references=["Ball et al. 1987", "Medlyn et al. 2011"]
        )
        
        ranges['g1'] = ParameterRange(
            min_value=2.0, max_value=20.0,
            typical_min=4.0, typical_max=12.0,
            units="dimensionless",
            description="Stomatal slope parameter",
            references=["Medlyn et al. 2011"]
        )
        
        # Ozone parameters
        ranges['o3_sensitivity'] = ParameterRange(
            min_value=0.0, max_value=3.0,
            typical_min=0.5, typical_max=2.0,
            units="dimensionless",
            description="Ozone sensitivity factor",
            references=["Mills et al. 2011", "Emberson et al. 2000"]
        )
        
        ranges['o3_threshold'] = ParameterRange(
            min_value=20.0, max_value=80.0,
            typical_min=35.0, typical_max=50.0,
            units="ppb",
            description="Ozone damage threshold",
            references=["Fuhrer et al. 1997"]
        )
        
        # Phenology parameters
        ranges['gdd_leaf_on'] = ParameterRange(
            min_value=0.0, max_value=1000.0,
            typical_min=50.0, typical_max=500.0,
            units="degree-days",
            description="GDD threshold for leaf onset",
            references=["White et al. 1997"]
        )
        
        ranges['specific_leaf_area'] = ParameterRange(
            min_value=2.0, max_value=80.0,
            typical_min=5.0, typical_max=50.0,
            units="m^2 kg^-1",
            description="Specific leaf area",
            references=["Wright et al. 2004", "Poorter et al. 2009"]
        )
        
        ranges['leaf_longevity'] = ParameterRange(
            min_value=30.0, max_value=2000.0,
            typical_min=60.0, typical_max=1500.0,
            units="days",
            description="Average leaf longevity",
            references=["Wright et al. 2004"]
        )
        
        # Allocation parameters
        ranges['f_leaf'] = ParameterRange(
            min_value=0.05, max_value=0.8,
            typical_min=0.15, typical_max=0.6,
            units="dimensionless",
            description="Leaf allocation fraction",
            references=["Poorter et al. 2012"]
        )
        
        ranges['f_stem'] = ParameterRange(
            min_value=0.05, max_value=0.7,
            typical_min=0.1, typical_max=0.5,
            units="dimensionless",
            description="Stem allocation fraction",
            references=["Poorter et al. 2012"]
        )
        
        ranges['f_root'] = ParameterRange(
            min_value=0.1, max_value=0.8,
            typical_min=0.2, typical_max=0.6,
            units="dimensionless",
            description="Root allocation fraction", 
            references=["Poorter et al. 2012"]
        )
        
        # Respiration parameters
        ranges['maint_resp_leaf'] = ParameterRange(
            min_value=0.005, max_value=0.1,
            typical_min=0.01, typical_max=0.03,
            units="g C g^-1 C day-1",
            description="Leaf maintenance respiration coefficient",
            references=["Ryan 1991", "Atkin et al. 2007"]
        )
        
        ranges['q10_maint_resp'] = ParameterRange(
            min_value=1.5, max_value=3.5,
            typical_min=1.8, typical_max=2.5,
            units="dimensionless",
            description="Q10 temperature coefficient for maintenance respiration",
            references=["Atkin & Tjoelker 2003"]
        )
        
        # BVOC emission factors  
        ranges['ef_isoprene'] = ParameterRange(
            min_value=0.0, max_value=100.0,
            typical_min=0.0, typical_max=50.0,
            units="mug g^-1 h^-1",
            description="Isoprene emission factor",
            references=["Guenther et al. 2012"]
        )
        
        ranges['ef_monoterpene'] = ParameterRange(
            min_value=0.0, max_value=50.0,
            typical_min=0.0, typical_max=20.0,
            units="mug g^-1 h^-1",
            description="Monoterpene emission factor",
            references=["Guenther et al. 2012"]
        )
        
        # Stress parameters
        ranges['water_stress_threshold'] = ParameterRange(
            min_value=-10.0, max_value=0.0,
            typical_min=-5.0, typical_max=-0.5,
            units="MPa",
            description="Water stress threshold",
            references=["Bartlett et al. 2012"]
        )
        
        return ranges
    
    def _initialize_consistency_rules(self) -> List[Dict[str, Any]]:
        """Initialize cross-parameter consistency rules."""
        rules = []
        
        # Jmax/Vcmax ratio rules
        rules.append({
            'name': 'jmax_vcmax_ratio_c3',
            'description': 'Jmax/Vcmax ratio for C3 plants',
            'condition': lambda pft: pft.photosynthesis.pathway == PhotosynthesisPathway.C3,
            'check': lambda pft: 1.0 <= pft.photosynthesis.jmax25 / pft.photosynthesis.vcmax25 <= 3.0,
            'message': 'C3 Jmax/Vcmax ratio should be between 1.0 and 3.0',
            'level': ValidationLevel.MODERATE
        })
        
        rules.append({
            'name': 'jmax_vcmax_ratio_c4',
            'description': 'Jmax/Vcmax ratio for C4 plants', 
            'condition': lambda pft: pft.photosynthesis.pathway == PhotosynthesisPathway.C4,
            'check': lambda pft: 0.5 <= pft.photosynthesis.jmax25 / pft.photosynthesis.vcmax25 <= 2.0,
            'message': 'C4 Jmax/Vcmax ratio should be between 0.5 and 2.0',
            'level': ValidationLevel.MODERATE
        })
        
        # Dark respiration ratio
        rules.append({
            'name': 'rd_vcmax_ratio',
            'description': 'Rd/Vcmax ratio check',
            'condition': lambda pft: True,
            'check': lambda pft: pft.photosynthesis.rd25 / pft.photosynthesis.vcmax25 <= 0.15,
            'message': 'Rd25/Vcmax25 ratio should not exceed 0.15',
            'level': ValidationLevel.MODERATE
        })
        
        # Allocation fraction sum
        rules.append({
            'name': 'allocation_sum',
            'description': 'Allocation fractions must sum to 1.0',
            'condition': lambda pft: True,
            'check': lambda pft: abs(pft.allocation.f_leaf + pft.allocation.f_stem + pft.allocation.f_root - 1.0) <= 0.01,
            'message': 'Leaf + stem + root allocation fractions must sum to 1.0',
            'level': ValidationLevel.STRICT
        })
        
        # Temperature range consistency
        rules.append({
            'name': 'temperature_order',
            'description': 'Temperature thresholds must be ordered correctly',
            'condition': lambda pft: True,
            'check': lambda pft: (pft.phenology.temp_growth_min < pft.phenology.temp_growth_opt < pft.phenology.temp_growth_max),
            'message': 'Growth temperatures must be ordered: min < opt < max',
            'level': ValidationLevel.STRICT
        })
        
        # GDD consistency
        rules.append({
            'name': 'gdd_order',
            'description': 'GDD thresholds must be ordered correctly',
            'condition': lambda pft: True,
            'check': lambda pft: pft.phenology.gdd_leaf_on <= pft.phenology.gdd_leaf_off,
            'message': 'GDD leaf on must be <= GDD leaf off',
            'level': ValidationLevel.STRICT
        })
        
        # SLA vs leaf longevity tradeoff
        rules.append({
            'name': 'sla_longevity_tradeoff',
            'description': 'SLA and leaf longevity tradeoff',
            'condition': lambda pft: True,
            'check': lambda pft: not (pft.phenology.specific_leaf_area > 30 and pft.phenology.leaf_longevity > 500),
            'message': 'High SLA (>30) typically not consistent with long leaf longevity (>500 days)',
            'level': ValidationLevel.MODERATE
        })
        
        # Ozone sensitivity for different pathways
        rules.append({
            'name': 'c4_ozone_sensitivity',
            'description': 'C4 plants typically less ozone sensitive',
            'condition': lambda pft: pft.photosynthesis.pathway == PhotosynthesisPathway.C4,
            'check': lambda pft: pft.photosynthesis.o3_sensitivity <= 1.2,
            'message': 'C4 plants typically have ozone sensitivity <= 1.2',
            'level': ValidationLevel.MODERATE
        })
        
        # BVOC emission mutual exclusivity
        rules.append({
            'name': 'isoprene_monoterpene_tradeoff',
            'description': 'High isoprene emitters typically low monoterpene',
            'condition': lambda pft: pft.bvoc.ef_isoprene > 10.0,
            'check': lambda pft: pft.bvoc.ef_monoterpene < 5.0,
            'message': 'High isoprene emitters (>10) typically have low monoterpene emission (<5)',
            'level': ValidationLevel.MODERATE
        })
        
        return rules
    
    def validate_pft(self, pft: PFTDefinition) -> ValidationResult:
        """
        Perform comprehensive validation of PFT parameters.
        
        Parameters
        ----------
        pft : PFTDefinition
            PFT definition to validate
            
        Returns
        -------
        ValidationResult
            Comprehensive validation results
        """
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            parameter_checks={}
        )
        
        # Validate individual parameters
        self._validate_parameter_ranges(pft, result)
        
        # Validate cross-parameter consistency
        self._validate_consistency_rules(pft, result)
        
        # Validate pathway-specific constraints
        self._validate_pathway_constraints(pft, result)
        
        # Validate temperature response curves
        self._validate_temperature_responses(pft, result)
        
        # Validate numerical stability
        self._validate_numerical_stability(pft, result)
        
        return result
    
    def _validate_parameter_ranges(self, pft: PFTDefinition, result: ValidationResult) -> None:
        """Validate individual parameter ranges."""
        
        # Photosynthesis parameters
        params_to_check = [
            ('photosynthesis.vcmax25', pft.photosynthesis.vcmax25),
            ('photosynthesis.jmax25', pft.photosynthesis.jmax25),
            ('photosynthesis.rd25', pft.photosynthesis.rd25),
            ('photosynthesis.vcmax_ea', pft.photosynthesis.vcmax_ea),
            ('photosynthesis.jmax_ea', pft.photosynthesis.jmax_ea),
            ('photosynthesis.alpha', pft.photosynthesis.alpha),
            ('photosynthesis.theta', pft.photosynthesis.theta),
            ('photosynthesis.g0', pft.photosynthesis.g0),
            ('photosynthesis.g1', pft.photosynthesis.g1),
            ('photosynthesis.o3_sensitivity', pft.photosynthesis.o3_sensitivity),
            ('photosynthesis.o3_threshold', pft.photosynthesis.o3_threshold),
            
            # Phenology parameters
            ('phenology.gdd_leaf_on', pft.phenology.gdd_leaf_on),
            ('phenology.specific_leaf_area', pft.phenology.specific_leaf_area),
            ('phenology.leaf_longevity', pft.phenology.leaf_longevity),
            
            # Allocation parameters
            ('allocation.f_leaf', pft.allocation.f_leaf),
            ('allocation.f_stem', pft.allocation.f_stem),
            ('allocation.f_root', pft.allocation.f_root),
            
            # Respiration parameters
            ('respiration.maint_resp_leaf', pft.respiration.maint_resp_leaf),
            ('respiration.q10_maint_resp', pft.respiration.q10_maint_resp),
            
            # BVOC parameters
            ('bvoc.ef_isoprene', pft.bvoc.ef_isoprene),
            ('bvoc.ef_monoterpene', pft.bvoc.ef_monoterpene),
            
            # Stress parameters
            ('stress.water_stress_threshold', pft.stress.water_stress_threshold),
        ]
        
        for param_name, param_value in params_to_check:
            # Extract base parameter name for range lookup
            base_name = param_name.split('.')[-1]
            
            if base_name in self._parameter_ranges:
                param_range = self._parameter_ranges[base_name]
                is_acceptable, message = param_range.check_value(param_value, self.validation_level)
                
                result.parameter_checks[param_name] = is_acceptable
                
                if not is_acceptable:
                    result.add_error(f"{param_name}: {message}")
                elif message:  # Warning case
                    result.add_warning(f"{param_name}: {message}")
    
    def _validate_consistency_rules(self, pft: PFTDefinition, result: ValidationResult) -> None:
        """Validate cross-parameter consistency rules."""
        
        for rule in self._consistency_rules:
            if rule['condition'](pft):  # Check if rule applies to this PFT
                rule_passes = rule['check'](pft)
                rule_name = rule['name']
                
                result.parameter_checks[rule_name] = rule_passes
                
                if not rule_passes:
                    if rule['level'] == ValidationLevel.STRICT or self.validation_level == ValidationLevel.STRICT:
                        result.add_error(f"Consistency rule '{rule_name}': {rule['message']}")
                    else:
                        result.add_warning(f"Consistency rule '{rule_name}': {rule['message']}")
    
    def _validate_pathway_constraints(self, pft: PFTDefinition, result: ValidationResult) -> None:
        """Validate pathway-specific constraints."""
        
        pathway = pft.photosynthesis.pathway
        
        if pathway == PhotosynthesisPathway.C4:
            # C4 plants should have higher temperature optimums
            if pft.phenology.temp_growth_opt < 25.0:
                result.add_warning("C4 plants typically have growth optimum >= 25degC")
            
            # C4 plants should have higher GDD requirements
            if pft.phenology.gdd_leaf_on < 100.0:
                result.add_warning("C4 plants typically require more heat units for leaf onset")
            
            # C4 plants should have different allocation patterns
            if pft.allocation.f_leaf > 0.4:
                result.add_warning("C4 plants typically have lower leaf allocation (<0.4)")
        
        elif pathway == PhotosynthesisPathway.C3:
            # C3 plants should have reasonable temperature ranges
            if pft.phenology.temp_growth_opt > 30.0:
                result.add_warning("C3 plants typically have growth optimum <= 30degC")
    
    def _validate_temperature_responses(self, pft: PFTDefinition, result: ValidationResult) -> None:
        """Validate temperature response curves for physical realism."""
        
        # Check activation energy ratios
        ea_ratio = pft.photosynthesis.jmax_ea / pft.photosynthesis.vcmax_ea
        if not 0.5 <= ea_ratio <= 1.2:
            result.add_warning(f"Jmax/Vcmax activation energy ratio {ea_ratio:.2f} outside typical range [0.5, 1.2]")
        
        # Check for reasonable deactivation energies
        if hasattr(pft.photosynthesis, 'vcmax_hd'):
            if pft.photosynthesis.vcmax_hd < 100000.0:
                result.add_warning("Vcmax deactivation energy unusually low (<100 kJ mol^-1)")
        
        # Validate Q10 values are reasonable
        if not 1.5 <= pft.respiration.q10_maint_resp <= 3.5:
            result.add_warning(f"Maintenance respiration Q10 {pft.respiration.q10_maint_resp:.2f} outside typical range [1.5, 3.5]")
    
    def _validate_numerical_stability(self, pft: PFTDefinition, result: ValidationResult) -> None:
        """Check for parameter combinations that might cause numerical issues."""
        
        # Check for zero or very small values that might cause division issues
        if pft.photosynthesis.vcmax25 < 1.0:
            result.add_error("Vcmax25 too small, may cause numerical instability")
        
        if pft.photosynthesis.alpha < 0.01:
            result.add_error("Alpha (quantum yield) too small, may cause numerical issues")
        
        if pft.photosynthesis.g0 < 0.001 and pft.photosynthesis.g1 < 1.0:
            result.add_warning("Very low stomatal conductance parameters may cause convergence issues")
        
        # Check allocation fractions for extreme values
        min_allocation = min(pft.allocation.f_leaf, pft.allocation.f_stem, pft.allocation.f_root)
        if min_allocation < 0.05:
            result.add_warning("Very low allocation fraction (<0.05) may cause numerical issues")
        
        # Check temperature range spans
        temp_range = pft.phenology.temp_growth_max - pft.phenology.temp_growth_min
        if temp_range < 10.0:
            result.add_warning("Very narrow growth temperature range may limit model flexibility")
        
        if temp_range > 60.0:
            result.add_warning("Very wide growth temperature range may be unrealistic")
    
    def validate_pft_collection(self, pfts: Dict[str, PFTDefinition]) -> Dict[str, ValidationResult]:
        """
        Validate a collection of PFTs.
        
        Parameters
        ----------
        pfts : Dict[str, PFTDefinition]
            Dictionary of PFT name to definition
            
        Returns
        -------
        Dict[str, ValidationResult]
            Dictionary of PFT name to validation result
        """
        results = {}

        for pft_name, pft_def in pfts.items():
            # A PFT whose parameter data cannot be evaluated (a None or
            # non-numeric value: TypeError; an invalid numeric value: ValueError)
            # gets a failed result. Any other error comes from the validator
            # implementation and propagates.
            try:
                results[pft_name] = self.validate_pft(pft_def)
            except (TypeError, ValueError) as e:
                results[pft_name] = ValidationResult(False, [f"Validation exception: {e}"], [], {})
                logger.error(f"Exception validating PFT '{pft_name}': {e}")
                continue
            logger.info(f"Validated PFT '{pft_name}': {'PASS' if results[pft_name].is_valid else 'FAIL'}")
        
        return results
    
    def generate_validation_report(self, results: Dict[str, ValidationResult]) -> str:
        """
        Generate a comprehensive validation report.
        
        Parameters
        ----------
        results : Dict[str, ValidationResult]
            Validation results for multiple PFTs
            
        Returns
        -------
        str
            Formatted validation report
        """
        report_lines = []
        report_lines.append("PFT Parameter Validation Report")
        report_lines.append("=" * 50)
        report_lines.append(f"Validation Level: {self.validation_level.value.upper()}")
        report_lines.append(f"Total PFTs: {len(results)}")
        
        valid_count = sum(1 for r in results.values() if r.is_valid)
        report_lines.append(f"Valid PFTs: {valid_count}/{len(results)}")
        report_lines.append("")
        
        # Summary statistics
        total_errors = sum(len(r.errors) for r in results.values())
        total_warnings = sum(len(r.warnings) for r in results.values())
        report_lines.append(f"Total Errors: {total_errors}")
        report_lines.append(f"Total Warnings: {total_warnings}")
        report_lines.append("")
        
        # Individual PFT results
        for pft_name, result in results.items():
            report_lines.append(f"PFT: {pft_name}")
            report_lines.append("-" * 30)
            report_lines.append(f"Status: {'VALID' if result.is_valid else 'INVALID'}")
            
            if result.errors:
                report_lines.append("Errors:")
                for error in result.errors:
                    report_lines.append(f"  - {error}")
            
            if result.warnings:
                report_lines.append("Warnings:")
                for warning in result.warnings:
                    report_lines.append(f"  - {warning}")
            
            report_lines.append("")
        
        return "\n".join(report_lines)


# Convenience functions
def validate_pft(pft: PFTDefinition, 
                validation_level: ValidationLevel = ValidationLevel.MODERATE) -> ValidationResult:
    """
    Validate a single PFT definition.
    
    Parameters
    ----------
    pft : PFTDefinition
        PFT to validate
    validation_level : ValidationLevel
        Validation strictness level
        
    Returns
    -------
    ValidationResult
        Validation results
    """
    validator = BiospherePFTValidator(validation_level)
    return validator.validate_pft(pft)


def quick_validate_pft(pft: PFTDefinition) -> bool:
    """
    Quick validation check returning only pass/fail.
    
    Parameters
    ---------- 
    pft : PFTDefinition
        PFT to validate
        
    Returns
    -------
    bool
        True if valid, False otherwise
    """
    result = validate_pft(pft, ValidationLevel.PERMISSIVE)
    return result.is_valid


# Export main classes and functions
__all__ = [
    'ValidationLevel',
    'ValidationResult',
    'ParameterRange',
    'BiospherePFTValidator',
    'validate_pft',
    'quick_validate_pft'
]
