"""
Centralized category alias mapping for the parameter system.

Provides a single source of truth that converts category labels into the
canonical ``ParameterCategory`` enumeration. This
ensures that parameter loading, validation, and auditing all interpret
category metadata consistently, eliminating ``Unknown category`` warnings
and duplicated alias logic across the codebase.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from .types import ParameterCategory

# fmt: off  # Keep mapping compact for readability
CATEGORY_ALIAS_MAP: Dict[str, ParameterCategory] = {
    # Core category labels
    'physical_parameter': ParameterCategory.PHYSICAL,
    'biological_parameter': ParameterCategory.BIOPHYSICAL,
    'environmental_parameter': ParameterCategory.ENVIRONMENTAL,
    'model_parameter': ParameterCategory.MODEL_PARAMETER,
    'calibration_parameter': ParameterCategory.CALIBRATION,
    'derived_parameter': ParameterCategory.DERIVED_PARAMETER,
    'physical_constant': ParameterCategory.PHYSICAL_CONSTANT,
    'environmental_variable': ParameterCategory.ENVIRONMENTAL_VARIABLE,

    # Namespace-derived shortcuts
    'physical': ParameterCategory.PHYSICAL,
    'biosphere': ParameterCategory.BIOPHYSICAL,
    'atmosphere': ParameterCategory.PHYSICAL,
    'hydrosphere': ParameterCategory.PHYSICAL,
    'pedosphere': ParameterCategory.PHYSICAL,
    'biosphere.process': ParameterCategory.BIOPHYSICAL,
    'transport': ParameterCategory.PHYSICAL,
    'relic.transport': ParameterCategory.PHYSICAL,
    'transport_property': ParameterCategory.PHYSICAL,
    'anthroposphere': ParameterCategory.ENVIRONMENTAL,
    'anthro_management': ParameterCategory.MANAGEMENT,
    'irrigation_parameter': ParameterCategory.MANAGEMENT,
    'grazing_parameter': ParameterCategory.MANAGEMENT,
    'land_use_mapping': ParameterCategory.AGGREGATION,
    'emission_inventory': ParameterCategory.PROCESS_COUPLING,
    'emission_sector_share': ParameterCategory.MODEL_PARAMETER,

    # Numerical/numerics related
    'numerical_parameter': ParameterCategory.COMPUTATIONAL,
    'computational_parameter': ParameterCategory.COMPUTATIONAL,
    'numerical_method': ParameterCategory.COMPUTATIONAL,
    'numerical_scaling': ParameterCategory.COMPUTATIONAL,
    'numerical_stability': ParameterCategory.COMPUTATIONAL,
    'numerical_threshold': ParameterCategory.COMPUTATIONAL,
    'numerical_tolerance': ParameterCategory.COMPUTATIONAL,
    'performance': ParameterCategory.COMPUTATIONAL,
    'iteration_limit': ParameterCategory.COMPUTATIONAL,
    'convergence_criterion': ParameterCategory.COMPUTATIONAL,
    'numerical': ParameterCategory.NUMERICAL,
    'numerical_constant': ParameterCategory.NUMERICAL,
    'numerical_default': ParameterCategory.NUMERICAL_default,

    # Model / empirical constants
    'empirical_constant': ParameterCategory.MODEL_PARAMETER,
    'reference_condition': ParameterCategory.MODEL_PARAMETER,
    'empirical_threshold': ParameterCategory.MODEL_PARAMETER,
    'default_value': ParameterCategory.MODEL_PARAMETER,
    'model_constant': ParameterCategory.MODEL_PARAMETER,
    'process_timescale': ParameterCategory.MODEL_PARAMETER,
    'environmental_driver': ParameterCategory.MODEL_PARAMETER,
    'conversion_factor': ParameterCategory.MODEL_PARAMETER,
    'general': ParameterCategory.MODEL_PARAMETER,
    'conservation_check': ParameterCategory.MODEL_PARAMETER,
    'conservation': ParameterCategory.CONSERVATION_CHECK,
    'validation_parameter': ParameterCategory.MODEL_PARAMETER,
    'emission_factor': ParameterCategory.MODEL_PARAMETER,
    'coupling_parameter': ParameterCategory.PROCESS_COUPLING,

    # Model configuration / discretisation
    'spatial_allocation': ParameterCategory.MODEL_CONFIGURATION,
    'spatial_configuration': ParameterCategory.MODEL_CONFIGURATION,
    'spatial_resolution': ParameterCategory.MODEL_CONFIGURATION,
    'temporal_resolution': ParameterCategory.MODEL_CONFIGURATION,
    'temporal_flag': ParameterCategory.MODEL_CONFIGURATION,
    'temporal_constant': ParameterCategory.MODEL_CONFIGURATION,
    'model_option': ParameterCategory.MODEL_CONFIGURATION,
    'model_switch': ParameterCategory.MODEL_CONFIGURATION,
    'configuration': ParameterCategory.MODEL_CONFIGURATION,
    'configuration_parameter': ParameterCategory.MODEL_CONFIGURATION,
    'configuration_flag': ParameterCategory.MODEL_CONFIGURATION,
    'control_flag': ParameterCategory.MODEL_CONFIGURATION,
    'capability_switch': ParameterCategory.MODEL_CONFIGURATION,
    'discretization_parameter': ParameterCategory.MODEL_CONFIGURATION,
    'grid_layout': ParameterCategory.MODEL_CONFIGURATION,
    'quality_control': ParameterCategory.VALIDATION_PARAMETER,
    'data_quality': ParameterCategory.DATA_QUALITY,
    'data_sources': ParameterCategory.MODEL_CONFIGURATION,
    'diagnostic_configuration': ParameterCategory.MODEL_CONFIGURATION,
    'runtime_configuration': ParameterCategory.MODEL_CONFIGURATION,
    'configuration_path': ParameterCategory.MODEL_CONFIGURATION,

    # Physical constants & references
    'environmental_constant': ParameterCategory.PHYSICAL_CONSTANT,
    'chemical_constant': ParameterCategory.PHYSICAL_CONSTANT,
    'gas_property': ParameterCategory.PHYSICAL_CONSTANT,
    'normalization_constant': ParameterCategory.PHYSICAL_CONSTANT,
    'reference_parameter': ParameterCategory.PHYSICAL_CONSTANT,
    'reference_ratio': ParameterCategory.PHYSICAL_CONSTANT,
    'reference_constant': ParameterCategory.PHYSICAL_CONSTANT,

    # Biological / PFT shortcuts
    'broadleaf_deciduous_temperate': ParameterCategory.BIOPHYSICAL,
    'c3_grass_temperate': ParameterCategory.BIOPHYSICAL,
    'c4_grass_temperate': ParameterCategory.BIOPHYSICAL,
    'needleleaf_evergreen_temperate': ParameterCategory.BIOPHYSICAL,
    'shrub_deciduous': ParameterCategory.BIOPHYSICAL,

    # Aggregation / weighting
    'aggregation_parameter': ParameterCategory.AGGREGATION,
    'weighting_parameter': ParameterCategory.AGGREGATION,
    'quality_index': ParameterCategory.AGGREGATION,
    'litter_composition': ParameterCategory.AGGREGATION,

    # Management / defaults
    'management_parameter': ParameterCategory.MANAGEMENT,
    'operational_parameter': ParameterCategory.MANAGEMENT,
    'management_objective': ParameterCategory.MANAGEMENT,
    'environmental_default': ParameterCategory.MANAGEMENT,
    'grid_default': ParameterCategory.MODEL_CONFIGURATION,
    'diagnostic_default': ParameterCategory.VALIDATION_PARAMETER,
    'stress_modifier': ParameterCategory.SCALING,
    'stress_indicator': ParameterCategory.SCALING,
    'scaling_factor': ParameterCategory.SCALING,
    'efficiency_definition': ParameterCategory.MODEL_PARAMETER,

    # Scaling & stoichiometry
    'nitrogen_partitioning': ParameterCategory.SCALING,
    'phosphorus_partitioning': ParameterCategory.SCALING,
    'uptake_parameter': ParameterCategory.SCALING,
    'root_distribution': ParameterCategory.SCALING,
    'immobilization': ParameterCategory.SCALING,

    # Atmospheric dynamics
    'turbulence_parameter': ParameterCategory.ATMOSPHERIC,
    'mixing_length_parameter': ParameterCategory.ATMOSPHERIC,
    'atmospheric_parameter': ParameterCategory.ATMOSPHERIC,
    'surface_parameter': ParameterCategory.ATMOSPHERIC,
    'canopy_parameter': ParameterCategory.ATMOSPHERIC,
    'stability_parameter': ParameterCategory.ATMOSPHERIC,

    # Isotope processes
    'isotope_standard': ParameterCategory.ISOTOPE,
    'fractionation_parameter': ParameterCategory.ISOTOPE,
    'temperature_parameter': ParameterCategory.ISOTOPE,
    'tracking_parameter': ParameterCategory.ISOTOPE,

    # Miscellaneous scientific domains
    'geochemical_parameter': ParameterCategory.GEOCHEMICAL,

    # Newly introduced atmospheric transport labels
    'atmospheric_state': ParameterCategory.MODEL_PARAMETER,
    'flux_parameter': ParameterCategory.MODEL_PARAMETER,
    'surface_property': ParameterCategory.MODEL_PARAMETER,
    'resistance_parameter': ParameterCategory.MODEL_PARAMETER,

    # Domain-specific labels seen in biosphere YAML
    'mechanistic_parameter': ParameterCategory.MODEL_PARAMETER,
    'crop_parameter': ParameterCategory.BIOPHYSICAL,
    'vegetation_descriptor': ParameterCategory.BIOPHYSICAL,
    'morphological_parameter': ParameterCategory.BIOPHYSICAL,
    'turnover_parameter': ParameterCategory.BIOPHYSICAL,
    'gsi_parameter': ParameterCategory.BIOPHYSICAL,
    'gdd_parameter': ParameterCategory.BIOPHYSICAL,
    'phenology_calendar': ParameterCategory.MODEL_CONFIGURATION,
    'seasonal_profile': ParameterCategory.MODEL_CONFIGURATION,
    'temporal_profile': ParameterCategory.MODEL_CONFIGURATION,
    'composition_parameter': ParameterCategory.MODEL_PARAMETER,
    'state_fraction': ParameterCategory.MODEL_PARAMETER,
    'transition_rate': ParameterCategory.MODEL_PARAMETER,
    'land_surface_modifier': ParameterCategory.MODEL_PARAMETER,
    'biogeochemical_modifier': ParameterCategory.MODEL_PARAMETER,
    'hydrological_modifier': ParameterCategory.MODEL_PARAMETER,
    'radiative_modifier': ParameterCategory.MODEL_PARAMETER,
    
    # Added aliases
    'interface_mapping': ParameterCategory.PROCESS_COUPLING,
    'biogeochemical_parameter': ParameterCategory.BIOPHYSICAL,
    'radiative_parameter': ParameterCategory.ATMOSPHERIC,
    'allocation_parameter': ParameterCategory.MODEL_CONFIGURATION,
    'conservation_flag': ParameterCategory.VALIDATION_PARAMETER,

    # RELIC-IPHS-DEC structural labels. These remain aliases or tags and do
    # not introduce a parallel ParameterCategory enum branch.
    'hodge': ParameterCategory.PHYSICAL,
    'hodge_star': ParameterCategory.PHYSICAL,
    'onsager': ParameterCategory.PHYSICAL,
    'mobility': ParameterCategory.PHYSICAL,
    'metric': ParameterCategory.MODEL_CONFIGURATION,
    'dec_metric': ParameterCategory.MODEL_CONFIGURATION,
    'iphs': ParameterCategory.PROCESS_COUPLING,
    'port': ParameterCategory.PROCESS_COUPLING,
    'structural_parameter': ParameterCategory.MODEL_PARAMETER,
    'structural_tolerance': ParameterCategory.VALIDATION_PARAMETER,
    'regime_number': ParameterCategory.VALIDATION_PARAMETER,
    'thermodynamic_parameter': ParameterCategory.PHYSICAL,
}
# fmt: on


def normalize_category(
    value: str,
    *,
    logger: Optional[logging.Logger] = None,
) -> ParameterCategory:
    """
    Translate a category string into a canonical :class:`ParameterCategory`.

    Parameters
    ----------
    value:
        Raw category string from YAML/database.
    logger:
        Optional logger used to report a recognized alias translation.
    """

    if not isinstance(value, str):
        raise TypeError("Parameter category must be a string")

    key = value.strip().lower()

    # Normalize spelling consistently for canonical values and declared aliases.
    try:
        return ParameterCategory(key)
    except ValueError as exc:
        if key not in CATEGORY_ALIAS_MAP:
            raise ValueError(f"Unknown parameter category: {value!r}") from exc

    if logger:
        logger.debug("Mapped category alias '%s' -> '%s'", value, CATEGORY_ALIAS_MAP[key].value)
    return CATEGORY_ALIAS_MAP[key]
