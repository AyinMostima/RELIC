"""
Central Model Builder for Multi-Scale Configuration Management

This module implements the central builder pattern to solve redundancy and 
inconsistency issues in the relic/models directory. It provides unified
metadata specifications, deduplication, conflict detection, and bounded CF metadata.

Key Components:
    - ScaleConfig: Data structure for scale-specific configurations
    - VariableSpec: Standardized variable specification
    - CentralModelBuilder: Main builder class with deduplication logic
    - ConflictDetector: Validation and conflict resolution
    - CFComplianceChecker: CF-1.8 standard compliance validation

Architecture Principle:
    Based on "Composition over Inheritance" pattern from RHESSys modeling
    framework, this builder dynamically composes scale configurations without
    cumulative inheritance chains.

Mathematical Expression:
    Effective configuration for scale s:
    F_eff(s) = union_{i<=s} F_delta(i) with deduplication by functor name
    V_eff(s) = oplus_{i<=s} V_delta(i) with conflict detection and resolution

References:
    - Tague, C.L. & Band, L.E. (2004). RHESSys: Regional Hydro-Ecologic Simulation System - an
      object-oriented approach to spatially distributed modeling of carbon, water, and nutrient
      cycling. Earth Interactions, 8(19), 1-42.
    - CF Metadata Conventions v1.8 (2017)
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Set
from enum import Enum
from dataclasses import dataclass, field
import logging
import warnings


# Import models modules
from relic.models import (
    individual, patch, hillslope, landscape, basin, regional
)
from relic.core.constants import get_scale_hierarchy
from relic.coupling.scales.types import SpatialQuantityBasis, resolve_spatial_basis

logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """
    Enumeration of spatial aggregation methods for multi-scale modeling.
    
    References:
        - RHESSys aggregation methods (Tague & Band, 2004)
        - CF cell_methods conventions v1.8
    """
    SUM = "sum"                           # Extensive variables: total across units
    MEAN = "mean"                         # Simple arithmetic mean
    AREA_WEIGHTED_MEAN = "area_weighted_mean"    # Weighted by area: Sigma(X_i * A_i) / Sigma(A_i)
    VOLUME_WEIGHTED_MEAN = "volume_weighted_mean" # Weighted by volume (3D fields)
    MAX = "max"                           # Maximum value
    MIN = "min"                           # Minimum value
    STANDARD_DEVIATION = "standard_deviation"    # Spatial variability
    MODE = "mode"                         # Most frequent value (categorical variables)
    EFFECTIVE_ROUGHNESS_LENGTH = "effective_roughness_length"  # Drag-coefficient blending (CMAQ Eq. 12-37')


@dataclass
class VariableSpec:
    """
    Variable specification with bounded CF metadata.

    Parameters:
        units: SI or CF-recommended units (e.g., "kg C m-2 s-1")
        standard_name: CF standard name (e.g., "gross_primary_productivity_of_carbon"), or None
            when the quantity has no CF standard name (no name is invented)
        spatial_basis: Explicit spatial representation (SpatialQuantityBasis owned by
            relic.coupling.scales.types: intensive, cell_total, area_density, categorical);
            a bare "extensive" is rejected as ambiguous
        description: Human-readable description
        aggregation: Internal aggregation method for spatial upscaling
        cell_methods: CF cell_methods string (optional, auto-generated if None)
        extras: Additional metadata (bounds, range, conserved flag, etc.)

    Mathematical Expression:
        intensive or area density to a parent density: X_agg = Sigma(X_i * A_i) / Sigma(A_i)
        cell total to a parent amount: X_total = Sigma(X_i)
        The basis/method pair is validated at construction (_validate_basis_method); a parent
        amount of an area density is a different variable with the cell_total basis.

    References:
        - CF Metadata Conventions v1.8 Section 7
        - Rew, R. & Davis, G. (1990). NetCDF user guide
    """
    units: str
    standard_name: Optional[str]
    spatial_basis: SpatialQuantityBasis
    description: str
    aggregation: AggregationMethod = AggregationMethod.AREA_WEIGHTED_MEAN
    cell_methods: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Parse the basis, validate the basis/method pair and generate cell_methods."""
        self.spatial_basis = SpatialQuantityBasis.parse(self.spatial_basis)
        if isinstance(self.aggregation, str):
            self.aggregation = AggregationMethod(self.aggregation)
        self._validate_basis_method()

        # cell_methods always follow the aggregation method and the basis.
        self.cell_methods = self._generate_cell_methods()

    def _validate_basis_method(self) -> None:
        """Raise unless the aggregation method is the conservative one of the declared basis."""
        # References: docs/source/theory/scale_coupling.md section 2.3.2; CF-1.8 section 7.3
        #   (cell_methods "area: sum" for amounts, "area: mean" for densities).
        # Mathematical expression: cell_total parent amount Y_j = sum_{i in j} X_i (SUM);
        #   area_density parent density y_j = sum_{i in j} x_i A_i / sum_{i in j} A_i
        #   (AREA_WEIGHTED_MEAN); categorical parent class = area-weighted mode (MODE).
        # Parameter meanings: X_i cell total; x_i area density; A_i cell area.
        # Units: A [m2]; X, Y amounts; x, y amounts m-2 (the density keeps its units).
        # Domain/constraints: any other method for an amount representation is not conservative
        #   and any arithmetic on class codes is meaningless, so it raises instead of being
        #   corrected silently; intensive fields accept every method (a sum only warns).
        # Sign convention: not applicable (method selection only).
        name = self.standard_name
        method = self.aggregation
        if self.spatial_basis is SpatialQuantityBasis.CELL_TOTAL and method is not AggregationMethod.SUM:
            raise ValueError(
                f"VariableSpec standard_name '{name}' is a cell_total; its conservative aggregation is SUM, got "
                f"{method.value}."
            )
        if self.spatial_basis is SpatialQuantityBasis.AREA_DENSITY and method is not AggregationMethod.AREA_WEIGHTED_MEAN:
            raise ValueError(
                f"VariableSpec standard_name '{name}' is an area_density; its parent value is the area-weighted mean "
                f"(AREA_WEIGHTED_MEAN), got {method.value}; a parent amount is a cell_total variable."
            )
        if self.spatial_basis is SpatialQuantityBasis.CATEGORICAL and method is not AggregationMethod.MODE:
            raise ValueError(
                f"VariableSpec standard_name '{name}' is categorical; its aggregation is MODE, got {method.value}."
            )
        if self.spatial_basis is SpatialQuantityBasis.INTENSIVE and method is AggregationMethod.SUM:
            logger.warning(
                "Variable '%s' is intensive but uses sum aggregation; AREA_WEIGHTED_MEAN keeps it "
                "physically consistent.",
                name,
            )
    
    def _generate_cell_methods(self) -> str:
        """
        Generate bounded cell_methods metadata from an aggregation method.
        
        Returns:
            CF cell_methods string (e.g., "area: mean time: mean")
        """
        method_map = {
            AggregationMethod.SUM: "sum",
            AggregationMethod.MEAN: "mean", 
            AggregationMethod.AREA_WEIGHTED_MEAN: "mean",
            AggregationMethod.VOLUME_WEIGHTED_MEAN: "mean",
            AggregationMethod.MAX: "maximum",
            AggregationMethod.MIN: "minimum",
            AggregationMethod.STANDARD_DEVIATION: "standard_deviation",
            AggregationMethod.MODE: "mode",
            AggregationMethod.EFFECTIVE_ROUGHNESS_LENGTH: "mean",
        }

        cf_method = method_map[self.aggregation]
        
        # Build cell_methods string
        methods = []
        
        # Add spatial dimension methods
        if self.aggregation is AggregationMethod.AREA_WEIGHTED_MEAN:
            methods.append(f"area: {cf_method}")
        elif self.aggregation == AggregationMethod.VOLUME_WEIGHTED_MEAN:
            methods.append(f"volume: {cf_method}")
        elif self.aggregation in [AggregationMethod.SUM, AggregationMethod.MEAN]:
            # For non-weighted aggregation, still specify dimension
            methods.append(f"area: {cf_method}")
        elif self.aggregation == AggregationMethod.MODE:
            # For categorical variables
            methods.append("area: mode")
        elif self.aggregation == AggregationMethod.STANDARD_DEVIATION:
            methods.append("area: standard_deviation")
        elif self.aggregation in [AggregationMethod.MAX, AggregationMethod.MIN]:
            methods.append(f"area: {cf_method}")
        elif self.aggregation == AggregationMethod.EFFECTIVE_ROUGHNESS_LENGTH:
            # CF has no dedicated method; the comment names the blending rule.
            methods.append(
                "area: mean (comment: effective roughness length from linearly additive drag "
                "coefficients at the blending height)"
            )
        
        # Add time dimension for time-varying variables
        # Skip for categorical/mode or point-in-time measurements
        if self.aggregation not in [AggregationMethod.MODE, 
                                   AggregationMethod.STANDARD_DEVIATION]:
            # Check if it's a categorical variable by examining the spatial basis
            if self.spatial_basis is not SpatialQuantityBasis.CATEGORICAL:
                methods.append("time: mean")
        elif self.spatial_basis is SpatialQuantityBasis.CATEGORICAL:
            methods.append("time: point")
        
        return " ".join(methods) if methods else "time: mean"


@dataclass 
class ScaleConfig:
    """
    Scale-specific configuration with delta (incremental) definitions.
    
    This data structure stores only the incremental changes introduced at each
    scale, following the "delta configuration" pattern to eliminate redundancy.
    
    Parameters:
        name: Scale name (e.g., "patch", "landscape")
        delta_functors: List of functor names unique to this scale
        delta_variables: Variables defined/modified at this scale
        coordinate_vars: Coordinate variable definitions
        coupling: Scale coupling configuration
        constants: Scale-specific constants
    
    Mathematical Expression:
        Total configuration at scale s: Config(s) = oplus_{i<=s} ScaleConfig(i)
        Where oplus represents conflict-resolved merge operation
    """
    name: str
    delta_functors: List[str] = field(default_factory=list)
    delta_variables: Dict[str, VariableSpec] = field(default_factory=dict)
    coordinate_vars: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    coupling: Dict[str, Any] = field(default_factory=dict)
    constants: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConflictReport:
    """
    Report of variable conflicts between scales with resolution suggestions.
    
    Parameters:
        conflicts: Dict mapping variable names to conflict details
        warnings: List of non-critical warnings
        errors: List of critical errors requiring resolution
    """
    conflicts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def has_errors(self) -> bool:
        """Check if report contains critical errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if report contains warnings."""
        return len(self.warnings) > 0


@dataclass
class ValidationReport:
    """
    Scoped metadata and unit validation report.
    
    Parameters:
        valid: Overall validation status
        cf_errors: CF standard compliance errors
        unit_errors: Unit consistency errors
        suggestions: Improvement suggestions
    """
    valid: bool = True
    cf_errors: List[str] = field(default_factory=list)
    unit_errors: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class CentralModelBuilder:
    """
    Central builder for multi-scale model configurations.
    
    This class implements the central builder pattern to eliminate redundancy
    in the relic/models directory by consuming delta configurations and
    generating effective (deduplicated, conflict-resolved) configurations.
    
    Key Features:
        - Functor deduplication with stable ordering
        - Variable conflict detection and resolution
        - Single-variable standard-name/unit validation
        - Integration with StateManager variable specifications
    
    Mathematical Expression:
        Effective functors: F_eff(s) = deduplicate(?_{iinhierarchy(s)} F_delta(i))
        Effective variables: V_eff(s) = resolve_conflicts(oplus_{iinhierarchy(s)} V_delta(i))
    
    References:
        - Tague, C.L. & Band, L.E. (2004). Earth Interactions, 8(19), 1-42 (RHESSys hierarchy).
    """
    
    def __init__(self, allow_overrides: Optional[Dict[str, List[str]]] = None):
        """
        Initialize central model builder.

        Parameters:
            allow_overrides: Dict mapping variable names to lists of fields
                           that can be overridden without error
        """
        self.allow_overrides = allow_overrides or {}
        self.scale_hierarchy = get_scale_hierarchy()
        self._scale_configs: Dict[str, ScaleConfig] = {}
        self._load_existing_configurations()
    
    def _load_existing_configurations(self):
        """
        Load existing scale configurations from relic/models modules.
        
        Converts the current cumulative inheritance pattern into delta
        configurations for each scale.
        """
        logger.info("Loading existing scale configurations from relic/models")
        
        # Define scale modules mapping
        scale_modules = {
            "individual": individual,
            "patch": patch,
            "hillslope": hillslope,
            "landscape": landscape,
            "basin": basin,
            "regional": regional
        }
        
        previous_functors: Set[str] = set()
        previous_variables: Set[str] = set()
        
        for scale_name in self.scale_hierarchy:
            if scale_name not in scale_modules:
                logger.warning(f"Scale module not found: {scale_name}")
                continue
                
            module = scale_modules[scale_name]
            
            # New functors of this scale in their declared LAYER_FUNCTORS order (a set
            # difference would order them by string hash, which varies with PYTHONHASHSEED).
            current_functors = list(dict.fromkeys(getattr(module, 'LAYER_FUNCTORS', [])))
            delta_functors = [name for name in current_functors if name not in previous_functors]
            
            # Extract variables (convert to VariableSpec format)
            current_vars = getattr(module, 'EXPOSED_VARIABLES', {})
            delta_variables = {}
            
            for var_name, var_config in current_vars.items():
                if var_name not in previous_variables:
                    # Convert variable config to VariableSpec
                    var_spec = self._convert_variable_spec(var_name, var_config)
                    delta_variables[var_name] = var_spec
            
            # Extract other configurations
            coordinate_vars = getattr(module, 'COORDINATE_VARIABLES', {})
            coupling = getattr(module, 'SCALE_COUPLING', {})
            
            # Create ScaleConfig for this scale
            config = ScaleConfig(
                name=scale_name,
                delta_functors=delta_functors,
                delta_variables=delta_variables,
                coordinate_vars=coordinate_vars,
                coupling=coupling
            )
            
            self._scale_configs[scale_name] = config
            
            # Update sets for next iteration
            previous_functors.update(current_functors)
            previous_variables.update(current_vars.keys())
            
            logger.debug(f"Loaded {scale_name}: {len(delta_functors)} functors, "
                        f"{len(delta_variables)} variables")
    
    def _convert_variable_spec(self, name: str, variable_config: Dict[str, Any]) -> VariableSpec:
        """
        Convert variable configuration to VariableSpec.
        
        Parameters:
            name: Variable name
            variable_config: Variable configuration dict
            
        Returns:
            VariableSpec instance with normalized fields
        """
        # Extract core fields
        units = variable_config.get("units", '1')
        # A variable without a CF standard name has none (None); no name is invented.
        standard_name = variable_config.get("standard_name")
        description = variable_config.get("description", f'{name} variable')

        # Explicit spatial basis from spatial_basis / variable_type (shared resolver; an
        # undeclared variable or a bare "extensive" raises instead of defaulting to intensive)
        spatial_basis = resolve_spatial_basis(name, variable_config)

        # Infer aggregation method from configuration formats
        aggregation = self._infer_aggregation_method(name, variable_config, spatial_basis)

        # Extract cell_methods (may be overridden by auto-generation)
        cell_methods = variable_config.get("cell_methods", None)

        # Collect extra metadata
        extras = {}
        for key, value in variable_config.items():
            if key not in ['units', 'standard_name', 'variable_type', 'spatial_basis', 'description',
                          'cell_methods', 'aggregation_method']:
                extras[key] = value

        # VariableSpec validates the basis/method pair (the single validation place); the error
        # is re-raised with the catalog variable name.
        try:
            return VariableSpec(
                units=units,
                standard_name=standard_name,
                spatial_basis=spatial_basis,
                description=description,
                aggregation=aggregation,
                cell_methods=cell_methods,
                extras=extras,
            )
        except ValueError as exc:
            raise ValueError(f"Variable '{name}': {exc}") from exc

    def _infer_aggregation_method(
        self,
        name: str,
        variable_config: Dict[str, Any],
        spatial_basis: SpatialQuantityBasis,
    ) -> AggregationMethod:
        """
        Infer aggregation method from configuration and the declared spatial basis.

        Parameters:
            name: Variable name (used in error messages)
            variable_config: Variable configuration
            spatial_basis: Declared SpatialQuantityBasis of the variable

        Returns:
            AggregationMethod enum value

        Raises:
            ValueError: If an explicit aggregation_method is not an AggregationMethod value.
        """
        # An explicit aggregation_method must be one of the accepted values; an unknown
        # string is rejected instead of silently falling back to cell_methods or the basis.
        if 'aggregation_method' in variable_config:
            method_str = variable_config["aggregation_method"]
            accepted = [method.value for method in AggregationMethod]
            if method_str not in accepted:
                raise ValueError(
                    f"Variable '{name}' declares aggregation_method {method_str!r}, which is not "
                    f"an accepted value; accepted values are {accepted}."
                )
            return AggregationMethod(method_str)

        # Parse cell_methods to look for SPATIAL aggregation hints
        # Note: "time: mean" alone doesn't specify spatial aggregation
        cell_methods = variable_config.get("cell_methods", '')
        
        # Check for explicit area/volume aggregation in cell_methods ("area: sum" is the sum of
        # cell totals; VariableSpec rejects it for any other basis)
        if 'area: sum' in cell_methods:
            return AggregationMethod.SUM
        elif 'area: mean' in cell_methods:
            return AggregationMethod.AREA_WEIGHTED_MEAN
        elif 'volume: mean' in cell_methods:
            return AggregationMethod.VOLUME_WEIGHTED_MEAN
        elif 'area: maximum' in cell_methods:
            return AggregationMethod.MAX
        elif 'area: minimum' in cell_methods:
            return AggregationMethod.MIN
        elif 'area: standard_deviation' in cell_methods:
            return AggregationMethod.STANDARD_DEVIATION
        elif 'point' in cell_methods and spatial_basis is SpatialQuantityBasis.CATEGORICAL:
            return AggregationMethod.MODE

        # Without an explicit spatial aggregation the conservative method of the declared
        # basis applies: cell totals are summed, area densities (and intensive fields) are
        # averaged over the parent area, categorical fields take the mode.
        if spatial_basis is SpatialQuantityBasis.CELL_TOTAL:
            return AggregationMethod.SUM
        if spatial_basis is SpatialQuantityBasis.CATEGORICAL:
            return AggregationMethod.MODE
        return AggregationMethod.AREA_WEIGHTED_MEAN
    
    def get_effective_functors(self, scale_name: str, include_lower: bool = True) -> List[str]:
        """
        Get deduplicated and ordered list of effective functors for a scale.
        
        Parameters:
            scale_name: Target scale name
            include_lower: Whether to include functors from lower scales
            
        Returns:
            Deduplicated list of functor names in stable order
            
        Mathematical Expression:
            F_eff(s) = deduplicate(?_{iinhierarchy(s)} F_delta(i))
            where hierarchy(s) = {scales from individual to s}
        """
        if scale_name not in self.scale_hierarchy:
            raise ValueError(f"Unknown scale: {scale_name}")
        
        if not include_lower:
            # Return only this scale's delta functors
            config = self._scale_configs.get(scale_name)
            return config.delta_functors if config else []
        
        # Collect functors from hierarchy
        effective_functors = []
        seen_functors = set()
        
        # Get scales up to and including target scale
        target_idx = self.scale_hierarchy.index(scale_name)
        relevant_scales = self.scale_hierarchy[:target_idx + 1]
        
        for scale in relevant_scales:
            config = self._scale_configs.get(scale)
            if config:
                for functor in config.delta_functors:
                    if functor not in seen_functors:
                        effective_functors.append(functor)
                        seen_functors.add(functor)
        
        total_functors = sum(len(self._scale_configs[s].delta_functors) 
                            for s in relevant_scales if s in self._scale_configs)
        logger.debug(f"Effective functors for {scale_name}: {len(effective_functors)} "
                    f"(deduplicated from {total_functors} total)")
        
        return effective_functors
    
    def get_effective_variables(self, 
                              scale_name: str, 
                              include_lower: bool = True,
                              allow_overrides: Optional[Dict[str, List[str]]] = None) -> Dict[str, VariableSpec]:
        """
        Get conflict-resolved effective variables for a scale.
        
        Parameters:
            scale_name: Target scale name
            include_lower: Whether to include variables from lower scales
            allow_overrides: Additional override permissions for this call
            
        Returns:
            Dict of variable names to VariableSpec instances
            
        Raises:
            ValueError: If critical conflicts are detected
            
        Mathematical Expression:
            V_eff(s) = resolve_conflicts(oplus_{iinhierarchy(s)} V_delta(i))
        """
        if scale_name not in self.scale_hierarchy:
            raise ValueError(f"Unknown scale: {scale_name}")
        
        if not include_lower:
            config = self._scale_configs.get(scale_name)
            return config.delta_variables if config else {}
        
        # Merge allow_overrides
        merged_overrides = dict(self.allow_overrides)
        if allow_overrides:
            merged_overrides.update(allow_overrides)
        
        # Collect variables from hierarchy
        effective_variables: Dict[str, VariableSpec] = {}
        target_idx = self.scale_hierarchy.index(scale_name)
        relevant_scales = self.scale_hierarchy[:target_idx + 1]
        
        conflict_report = ConflictReport()
        
        for scale in relevant_scales:
            config = self._scale_configs.get(scale)
            if not config:
                continue
                
            for var_name, var_spec in config.delta_variables.items():
                if var_name in effective_variables:
                    # Conflict detected - check if override is allowed
                    existing_spec = effective_variables[var_name]
                    conflicts = self._detect_variable_conflicts(
                        var_name, existing_spec, var_spec, merged_overrides
                    )
                    
                    if conflicts['critical']:
                        conflict_report.errors.append(
                            f"Critical conflict for variable '{var_name}' between scales: {conflicts['details']}"
                        )
                    else:
                        if conflicts['warnings']:
                            conflict_report.warnings.extend(conflicts['warnings'])
                        # Override with newer definition
                        effective_variables[var_name] = var_spec
                        logger.debug(f"Variable '{var_name}' overridden in scale '{scale}'")
                else:
                    effective_variables[var_name] = var_spec
        
        # Report conflicts
        if conflict_report.has_errors():
            error_msg = "Critical variable conflicts detected:\n" + "\n".join(conflict_report.errors)
            raise ValueError(error_msg)
        
        if conflict_report.has_warnings():
            for warning in conflict_report.warnings:
                warnings.warn(f"Variable conflict: {warning}", UserWarning)
        
        logger.debug(f"Effective variables for {scale_name}: {len(effective_variables)}")
        
        return effective_variables
    
    def _detect_variable_conflicts(self, 
                                  var_name: str,
                                  existing: VariableSpec, 
                                  new: VariableSpec,
                                  allow_overrides: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Detect conflicts between variable specifications.
        
        Parameters:
            var_name: Variable name
            existing: Existing VariableSpec
            new: New VariableSpec to merge
            allow_overrides: Fields that can be overridden
            
        Returns:
            Dict with conflict analysis results
        """
        conflicts = {
            'critical': False,
            'warnings': [],
            'details': {}
        }
        
        # Core fields that must match (unless explicitly allowed)
        core_fields = ['units', 'standard_name', 'spatial_basis']
        allowed_overrides = allow_overrides.get(var_name, [])
        
        for field_item in core_fields:
            existing_val = getattr(existing, field_item, None)
            new_val = getattr(new, field_item, None)
            
            if existing_val != new_val:
                if field_item in allowed_overrides:
                    conflicts['warnings'].append(
                        f"Field '{field_item}' overridden for variable '{var_name}': "
                        f"{existing_val} -> {new_val}"
                    )
                else:
                    conflicts['critical'] = True
                    conflicts['details'][field_item] = {
                        'existing': existing_val,
                        'new': new_val
                    }
        
        # Non-core fields (soft conflicts - warnings only)
        soft_fields = ['description', 'aggregation', 'cell_methods']
        for soft_field in soft_fields:
            existing_val = getattr(existing, soft_field, None)
            new_val = getattr(new, soft_field, None)
            
            if existing_val != new_val:
                conflicts['warnings'].append(
                    f"Non-critical field '{soft_field}' changed for variable '{var_name}': "
                    f"{existing_val} -> {new_val}"
                )
        
        return conflicts
    
    def check_conflicts(self, scale_name: str) -> ConflictReport:
        """
        Check for conflicts in variable definitions up to specified scale.
        
        Parameters:
            scale_name: Target scale to check
            
        Returns:
            ConflictReport with detailed conflict analysis
        """
        report = ConflictReport()
        
        try:
            # Attempt to get effective variables (will raise if critical conflicts)
            _ = self.get_effective_variables(scale_name, include_lower=True)
            logger.info(f"No critical conflicts detected for scale '{scale_name}'")
        except ValueError as e:
            report.errors.append(str(e))
        
        return report
    
    def validate_cf_and_units(self, variables: Dict[str, VariableSpec]) -> ValidationReport:
        """
        Check standard names against the CF standard name table and units with cf_units.

        Each variable goes through CFConventions.validate_cf_compliance (the CF owner,
        relic/data/standards/cf_conventions.py): the standard_name must be an official table
        entry or alias, the units must be a UDUNITS string convertible to the canonical units
        of that name. A name outside the table is a cf_error; an unrecognised or incompatible
        unit is a unit_error. A standard_name of None means the variable has no CF standard
        name: the name check is skipped and the units must still be a UDUNITS string.

        Parameters:
            variables: Dict of variables to validate

        Returns:
            ValidationReport with validation results
        """
        from relic.data.standards.cf_conventions import CFConventions

        report = ValidationReport()
        for var_name, var_spec in variables.items():
            metadata = {"units": var_spec.units}
            if var_spec.standard_name is not None:
                metadata["standard_name"] = var_spec.standard_name
            result = CFConventions.validate_cf_compliance(metadata)
            for issue in result.issues:
                message = (
                    f"Variable '{var_name}' (standard_name '{var_spec.standard_name}', units "
                    f"'{var_spec.units}'): {issue}"
                )
                if result.membership_checked or issue.startswith("Unrecognized UDUNITS"):
                    report.unit_errors.append(message)
                else:
                    report.cf_errors.append(message)

        report.valid = len(report.cf_errors) == 0 and len(report.unit_errors) == 0
        if not report.valid:
            logger.warning(f"CF/Units validation failed: {len(report.cf_errors)} CF errors, "
                          f"{len(report.unit_errors)} unit errors")
        return report

    def get_scale_config(self, scale_name: str) -> Optional[ScaleConfig]:
        """Get ScaleConfig for a specific scale."""
        return self._scale_configs.get(scale_name)
    
    def list_scales(self) -> List[str]:
        """List all available scales."""
        return list(self._scale_configs.keys())


# Factory function for easy instantiation
def create_model_builder(allow_overrides: Optional[Dict[str, List[str]]] = None) -> CentralModelBuilder:
    """
    Factory function to create CentralModelBuilder instance.
    
    Parameters:
        allow_overrides: Dict mapping variable names to overrideable fields
        
    Returns:
        Configured CentralModelBuilder instance
    """
    return CentralModelBuilder(allow_overrides=allow_overrides)


# Example usage and testing functions
def _test_basic_functionality():
    """Basic functionality test for the central model builder."""
    builder = create_model_builder()
    
    # Test get_effective_functors
    print("Testing get_effective_functors...")
    individual_functors = builder.get_effective_functors("individual", include_lower=False)
    patch_functors = builder.get_effective_functors("patch", include_lower=True)
    print(f"Individual functors: {len(individual_functors)}")
    print(f"Patch effective functors: {len(patch_functors)}")
    
    # Test get_effective_variables
    print("\nTesting get_effective_variables...")
    patch_vars = builder.get_effective_variables("patch", include_lower=True)
    print(f"Patch effective variables: {len(patch_vars)}")
    
    # Test conflict checking
    print("\nTesting conflict detection...")
    conflicts = builder.check_conflicts("landscape")
    print(f"Conflicts found: {conflicts.has_errors()}")
    
    # Test CF validation
    print("\nTesting CF validation...")
    validation = builder.validate_cf_and_units(patch_vars)
    print(f"CF validation passed: {validation.valid}")
    
    print("Basic functionality test completed.")


if __name__ == "__main__":
    _test_basic_functionality()
