"""
Unit Conversion System for RELIC

This module provides explicit numerical conversion between scientific units.
Official CF standard-name membership and canonical-unit lookup are owned by
``CFConventions`` and are not inferred here. Explicit rules in
``UNIT_CONVERSIONS`` are tried first; a pair without a rule is converted only
when it is a pure same-dimension UDUNITS scale (the unit relation
``CFConventions`` uses), so a CF-consistent label reaches its canonical unit.

Mathematical Framework:
    Soil moisture conversions:
    - Volume fraction to mass: kg/m^2 = (m^3/m^3) * depth(m) * rho_water(kg/m^3)
    - Mass to volume fraction: m^3/m^3 = (kg/m^2) / (depth(m) * rho_water(kg/m^3))

References:
    - CF Conventions Version 1.8: http://cfconventions.org/
    - Hillel, D. (1998). Environmental Soil Physics
    - Brady, N.C. & Weil, R.R. (2008). The Nature and Properties of Soils
"""

import math

import torch
from torch import Tensor
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import yaml
from cf_units import Unit

logger = logging.getLogger(__name__)


@dataclass 
class ConversionContext:
    """Context information required for unit conversions."""
    soil_depth: Optional[Union[float, Tensor]] = None
    water_density: Optional[Union[float, Tensor]] = None
    bulk_density: Optional[float] = None
    temperature: Optional[Union[float, Tensor]] = None
    pressure: Optional[Union[float, Tensor]] = None
    allow_identity: bool = False  # Enable only when identity conversions are explicitly desired
    allow_defaults: bool = False  # Defaults require explicit opt-in


class UnitConverter:
    """
    Unit converter for explicit physical-unit transformations.
    
    Handles complex conversions like soil moisture units while maintaining
    scientific accuracy and data conservation.

    Variable Name Normalization:
        Callers pass canonical scientific variable names. The converter maps
        recognized canonical names to conversion categories before dispatch-
        table lookup, eliminating duplicate conversion formulas.
    """

    _ELEMENT_BUDGET_UNIT_SPECS = {
        "carbon": (
            "g C m-2",
            "g C m-2 s-1",
            "carbon_amount",
            "carbon_flux",
        ),
        "nitrogen": (
            "g N m-2",
            "g N m-2 s-1",
            "nitrogen_amount",
            "nitrogen_flux",
        ),
        "phosphorus": (
            "g P m-2",
            "g P m-2 s-1",
            "phosphorus_amount",
            "phosphorus_flux",
        ),
        "water": (
            "kg m-2",
            "kg m-2 s-1",
            "water_amount",
            "water_flux",
        ),
    }

    # ------------------------------------------------------------------ #
    # Variable name normalization: alias -> canonical name               #
    # All aliases are lowercased before lookup.                          #
    # ------------------------------------------------------------------ #
    VARIABLE_ALIASES: Dict[str, str] = {
        # Carbon flux aliases -> 'carbon_flux'
        'gross_primary_productivity': 'carbon_flux',
        'gross_primary_productivity_of_carbon': 'carbon_flux',
        'carbon_respiration_flux': 'carbon_flux',
        'net_primary_productivity': 'carbon_flux',
        'net_primary_productivity_of_carbon': 'carbon_flux',
        'ecosystem_respiration': 'carbon_flux',
        'heterotrophic_respiration': 'carbon_flux',
        'autotrophic_respiration': 'carbon_flux',
        'leaf_respiration': 'carbon_flux',
        'nee': 'carbon_flux',
        'net_ecosystem_exchange': 'carbon_flux',
        'leaf_litterfall_carbon': 'carbon_flux',
        'froot_litterfall_carbon': 'carbon_flux',

        # Water flux aliases -> 'water_flux'
        'evapotranspiration_flux': 'water_flux',
        'transpiration_flux': 'water_flux',
        'evaporation_flux': 'water_flux',
        'water_evaporation_flux': 'water_flux',
        'effective_transpiration': 'water_flux',
        'runoff_flux': 'water_flux',
        'surface_runoff_flux': 'water_flux',
        'subsurface_runoff_flux': 'water_flux',
        'baseflow_flux': 'water_flux',
        'interflow_flux': 'water_flux',
        'infiltration_flux': 'water_flux',
        'drainage_flux': 'water_flux',
        'capillary_rise_flux': 'water_flux',
        'snow_melt_flux': 'water_flux',
        'groundwater_pumping': 'water_flux',
        'lateral_inflow': 'water_flux',
        'irrigation_flux': 'water_flux',
        'sublimation_flux': 'water_flux',
        'snowpack_net_mass_flux': 'water_flux',
        'water_sublimation_flux': 'water_flux',
        'irrigation_demand_flux': 'water_flux',
        'total_irrigation_flux': 'water_flux',

        # No alias into 'soil_moisture_content': that category is the areal mass
        # content of soil water [kg m-2] (CF soil_moisture_content). The volumetric
        # fraction [m3 m-3] (soil_water_content, volumetric_water_content, ...) is
        # another quantity; it becomes a mass content only as the declared
        # soil_moisture_content conversion with an explicit layer thickness.
        # latent_heat_flux [W m-2] is an energy flux, not a water mass flux, and
        # 'precipitation' names an amount or a flux (see CATALOG_ALIASES); neither
        # has a water-flux alias.

        # Temperature aliases -> 'air_temperature'
        'surface_temperature': 'air_temperature',
        'soil_temperature': 'air_temperature',
        'leaf_temperature': 'air_temperature',
        'canopy_temperature': 'air_temperature',
        'temperature': 'air_temperature',

        # Pressure aliases -> 'pressure'
        'air_pressure': 'pressure',
        'surface_pressure': 'pressure',
        'atmospheric_pressure': 'pressure',

        # CO2 concentration aliases -> 'co2_concentration'
        'atmospheric_mole_concentration_of_carbon_dioxide': 'co2_concentration',
        'co2_partial_pressure': 'co2_concentration',

        # Precipitation aliases -> 'precipitation_flux'
        'rainfall_flux': 'precipitation_flux',
        'snowfall_flux': 'precipitation_flux',
        'rain_flux': 'precipitation_flux',
        'snowfall': 'precipitation_flux',

        # Canonical nitrogen pools and rates -> element-specific conversion classes
        'organic_nitrogen': 'nitrogen_amount',
        'microbial_nitrogen': 'nitrogen_amount',
        'nh4_nitrogen': 'nitrogen_amount',
        'no3_nitrogen': 'nitrogen_amount',
        'leaf_nitrogen': 'nitrogen_amount',
        'froot_nitrogen': 'nitrogen_amount',
        'plant_nitrogen_pool': 'nitrogen_amount',
        'plant_n_storage': 'nitrogen_amount',
        'mineralization_flux': 'nitrogen_flux',
        'gross_mineralization_flux': 'nitrogen_flux',
        'immobilization_flux': 'nitrogen_flux',
        'microbial_death_flux': 'nitrogen_flux',
        'nitrification_flux': 'nitrogen_flux',
        'denitrification_flux': 'nitrogen_flux',
        'fertilizer_n_flux': 'nitrogen_flux',
        'leaf_litterfall_nitrogen': 'nitrogen_flux',
        'froot_litterfall_nitrogen': 'nitrogen_flux',
        'nitrogen_retranslocation': 'nitrogen_flux',

    }

    # ------------------------------------------------------------------ #
    # Dispatch table: (canonical_variable, source_units, target_units)    #
    #   -> method_name                                                    #
    # After normalization, lookup uses only canonical variable names.     #
    # 'generic' is the universal default dispatch key for unrecognized variables.  #
    # ------------------------------------------------------------------ #
    UNIT_CONVERSIONS = {
        # Soil moisture conversions
        ('soil_moisture_content', 'm3 m-3', 'kg m-2'): 'volumetric_to_mass_soil_moisture',
        ('soil_moisture_content', 'kg m-2', 'm3 m-3'): 'mass_to_volumetric_soil_moisture',

        # Temperature conversions
        ('air_temperature', 'celsius', 'K'): 'celsius_to_kelvin',
        ('air_temperature', 'C', 'K'): 'celsius_to_kelvin',
        ('air_temperature', 'degC', 'K'): 'celsius_to_kelvin',
        ('air_temperature', 'K', 'celsius'): 'kelvin_to_celsius',
        ('air_temperature', 'K', 'C'): 'kelvin_to_celsius',
        ('air_temperature', 'K', 'degC'): 'kelvin_to_celsius',
        ('generic', 'celsius', 'K'): 'celsius_to_kelvin',
        ('generic', 'C', 'K'): 'celsius_to_kelvin',
        ('generic', 'degC', 'K'): 'celsius_to_kelvin',
        ('generic', 'K', 'celsius'): 'kelvin_to_celsius',
        ('generic', 'K', 'C'): 'kelvin_to_celsius',
        ('generic', 'K', 'degC'): 'kelvin_to_celsius',

        # Pressure conversions
        ('pressure', 'hPa', 'Pa'): 'hpa_to_pa',
        ('pressure', 'Pa', 'hPa'): 'pa_to_hpa',
        ('pressure', 'Pa', 'kPa'): 'pa_to_kpa',
        ('pressure', 'kPa', 'Pa'): 'kpa_to_pa',
        ('generic', 'hPa', 'Pa'): 'hpa_to_pa',
        ('generic', 'Pa', 'hPa'): 'pa_to_hpa',
        ('generic', 'Pa', 'kPa'): 'pa_to_kpa',
        ('generic', 'kPa', 'Pa'): 'kpa_to_pa',

        # Gas concentration conversions
        ('co2_concentration', 'ppm', 'mol m-3'): 'ppm_to_molar_concentration',
        ('co2_concentration', 'mol m-3', 'ppm'): 'molar_concentration_to_ppm',
        ('co2_concentration', 'umol mol-1', 'Pa'): 'co2_ppm_to_pa',
        ('co2_concentration', 'Pa', 'umol mol-1'): 'co2_pa_to_ppm',
        ('co2_concentration', 'mol mol-1', 'Pa'): 'co2_mole_fraction_to_pa',
        ('co2_concentration', 'Pa', 'mol mol-1'): 'co2_pa_to_mole_fraction',
        ('co2_concentration', 'ppm', 'Pa'): 'co2_ppm_to_pa',
        ('co2_concentration', 'Pa', 'ppm'): 'co2_pa_to_ppm',

        # Mass conversions
        ('generic', 'g', 'kg'): 'gram_to_kilogram',
        ('generic', 'kg', 'g'): 'kilogram_to_gram',

        # Time conversions
        ('generic', 's', 'hour'): 'second_to_hour',
        ('generic', 'hour', 's'): 'hour_to_second',
        ('generic', 'day', 's'): 'day_to_second',
        ('generic', 's', 'day'): 'second_to_day',

        # Carbon flux conversions (canonical name: carbon_flux)
        ('carbon_flux', 'g m-2 s-1', 'umol m-2 s-1'): 'carbon_flux_g_to_umol',
        ('carbon_flux', 'umol m-2 s-1', 'g m-2 s-1'): 'carbon_flux_umol_to_g',
        ('carbon_flux', 'g C m-2 day-1', 'kg C m-2 s-1'): 'carbon_flux_g_day_to_kg_s',
        ('carbon_flux', 'kg C m-2 s-1', 'g C m-2 day-1'): 'carbon_flux_kg_s_to_g_day',
        ('carbon_flux', 'g C m-2 s-1', 'kg C m-2 s-1'): 'gram_to_kilogram',
        ('carbon_flux', 'kg C m-2 s-1', 'g C m-2 s-1'): 'kilogram_to_gram',
        ('carbon_flux', 'kg C m-2 s-1', 'umol m-2 s-1'): 'carbon_flux_kg_to_umol',
        ('carbon_flux', 'umol m-2 s-1', 'kg C m-2 s-1'): 'carbon_flux_umol_to_kg',
        # Element-annotated mass <-> amount pairs (g C m-2 <-> mol C m-2, kg N m-2 s-1
        # <-> umol N m-2 s-1, ...) use the single rule _element_mass_amount_factor.
        ('generic', 'g C m-2 day-1', 'kg C m-2 s-1'): 'carbon_flux_g_day_to_kg_s',
        ('generic', 'kg C m-2 s-1', 'g C m-2 day-1'): 'carbon_flux_kg_s_to_g_day',
        ('generic', 'g C m-2 s-1', 'kg C m-2 s-1'): 'gram_to_kilogram',
        ('generic', 'kg C m-2 s-1', 'g C m-2 s-1'): 'kilogram_to_gram',
        ('generic', 'g C m-2', 'kg C m-2'): 'gram_to_kilogram',
        ('generic', 'kg C m-2', 'g C m-2'): 'kilogram_to_gram',
        ('generic', 'g P m-2', 'kg P m-2'): 'gram_to_kilogram',
        ('generic', 'kg P m-2', 'g P m-2'): 'kilogram_to_gram',
        ('generic', 'g P m-2 s-1', 'kg P m-2 s-1'): 'gram_to_kilogram',
        ('generic', 'kg P m-2 s-1', 'g P m-2 s-1'): 'kilogram_to_gram',
        ('generic', 'mol m-2 s-1', 'umol m-2 s-1'): 'mol_to_umol_flux',
        ('generic', 'umol m-2 s-1', 'mol m-2 s-1'): 'umol_to_mol_flux',

        # Element-specific nitrogen mass scaling
        ('nitrogen_amount', 'kg N m-2', 'g N m-2'): 'kilogram_to_gram',
        ('nitrogen_amount', 'g N m-2', 'kg N m-2'): 'gram_to_kilogram',
        ('nitrogen_flux', 'kg N m-2 s-1', 'g N m-2 s-1'): 'kilogram_to_gram',
        ('nitrogen_flux', 'g N m-2 s-1', 'kg N m-2 s-1'): 'gram_to_kilogram',

        # Water flux conversions (canonical name: water_flux)
        ('water_flux', 'mol m-2 s-1', 'kg m-2 s-1'): 'water_flux_mol_to_kg',
        ('water_flux', 'kg m-2 s-1', 'mol m-2 s-1'): 'water_flux_kg_to_mol',
        ('water_flux', 'mol H2O m-2 s-1', 'kg m-2 s-1'): 'water_flux_mol_to_kg',
        ('water_flux', 'kg m-2 s-1', 'mol H2O m-2 s-1'): 'water_flux_kg_to_mol',

        # Unit harmonization and identity conversions
        ('generic', 'mol O3 m-2', 'mol O3 m^-2'): 'identity',
        ('generic', 'mol O3 m^-2', 'mol O3 m-2'): 'identity',
        ('generic', 'mol e- mol-1 photon', 'dimensionless'): 'identity',
        ('generic', 'dimensionless', 'mol e- mol-1 photon'): 'identity',
        ('generic', 'dimensionless', '1'): 'identity',
        ('generic', '1', 'dimensionless'): 'identity',
        ('generic', 'g mol-1', 'g/mol'): 'identity',
        ('generic', 'g/mol', 'g mol-1'): 'identity',
        ('generic', 'kg mol-1', 'kg/mol'): 'identity',
        ('generic', 'kg/mol', 'kg mol-1'): 'identity',
        ('generic', 'ppm', 'umol mol-1'): 'identity',
        ('generic', 'umol mol-1', 'ppm'): 'identity',
        ('generic', 'ppm', 'mol mol-1'): 'ppm_to_mole_fraction',
        ('generic', 'mol mol-1', 'ppm'): 'mole_fraction_to_ppm',
        ('generic', 'ppb', 'mol mol-1'): 'ppb_to_mole_fraction',
        ('generic', 'mol mol-1', 'ppb'): 'mole_fraction_to_ppb',
        ('molar_flux', 'mol m-2 s-1', 'umol m-2 s-1'): 'mol_to_umol_flux',
        ('molar_flux', 'umol m-2 s-1', 'mol m-2 s-1'): 'umol_to_mol_flux',
        ('generic', 'mol m-3', 'umol m-3'): 'mol_to_umol_flux',
        ('generic', 'umol m-3', 'mol m-3'): 'umol_to_mol_flux',

        # Length conversions
        ('generic', 'm', 'mm'): 'meter_to_millimeter',
        ('generic', 'mm', 'm'): 'millimeter_to_meter',
        ('generic', 'm', 'cm'): 'meter_to_centimeter',
        ('generic', 'cm', 'm'): 'centimeter_to_meter',
        ('generic', 'm', 'km'): 'meter_to_kilometer',
        ('generic', 'km', 'm'): 'kilometer_to_meter',

        # External model specific conversions
        ('external_model', 'Ly day-1', 'W m-2'): 'external_radiation_langley_to_wm2',
        ('external_model', 'cal cm-2 day-1', 'W m-2'): 'external_radiation_cal_to_wm2',

        # Conductance conversions share dimensions but retain distinct semantics.
        ('stomatal_conductance', 'mol m-2 s-1', 'm s-1'): 'conductance_mol_to_ms',
        ('stomatal_conductance', 'm s-1', 'mol m-2 s-1'): 'conductance_ms_to_mol',
        ('canopy_conductance', 'mol m-2 s-1', 'm s-1'): 'conductance_mol_to_ms',
        ('canopy_conductance', 'm s-1', 'mol m-2 s-1'): 'conductance_ms_to_mol',
        ('boundary_layer_conductance', 'mol m-2 s-1', 'm s-1'): 'conductance_mol_to_ms',
        ('boundary_layer_conductance', 'm s-1', 'mol m-2 s-1'): 'conductance_ms_to_mol',
    }
    
    # Liquid-water-equivalent (lwe) family: canonical categories whose depth form is,
    # by declaration, the thickness of liquid water (CF standard-name table v94:
    # "lwe" means liquid water equivalent). CF names starting with "lwe_" belong to
    # the family as well. A unit pair alone never selects this rule.
    _LIQUID_WATER_EQUIVALENT_CATEGORIES = frozenset(
        {"water_flux", "water_amount", "precipitation_flux"}
    )

    def __init__(self):
        self._parameter_manager_cache = None
        self._cross_dimension_rules = None

    @classmethod
    def get_element_budget_units(cls, element: str) -> Dict[str, str]:
        """Return the canonical unit basis for one elemental budget."""
        key = str(element).strip().lower()
        try:
            storage, flux, amount_variable, flux_variable = (
                cls._ELEMENT_BUDGET_UNIT_SPECS[key]
            )
        except KeyError as exc:
            raise ValueError(
                f"Unsupported conservation element '{element}'"
            ) from exc
        return {
            "storage": storage,
            "flux": flux,
            "amount_variable": amount_variable,
            "flux_variable": flux_variable,
        }

    @staticmethod
    def _get_physical_constant(name: str) -> float:
        """Resolve one constant through the canonical physical-constant API."""
        from ...utils.physical_constants import get_physical_constant

        return float(get_physical_constant(name))

    def _resolve_molecular_weight(self, parameter_key: str) -> float:
        """Resolve a molecular weight from context or the physical SSOT."""
        constant_path = parameter_key
        if not constant_path.startswith("relic.physical.constants."):
            constant_path = f"relic.physical.constants.{constant_path}"
        try:
            return self._get_physical_constant(constant_path)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Failed to resolve molecular weight for {parameter_key}: {exc}") from exc

    def _resolve_physical_constant(
        self,
        parameter_key: str,
        *,
        units: Optional[str] = None,
    ) -> float:
        """Resolve physical constants through the canonical SSOT API."""
        constant_path = parameter_key
        if not constant_path.startswith("relic.physical.constants."):
            constant_path = f"relic.physical.constants.{constant_path}"
        try:
            return self._get_physical_constant(constant_path)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Failed to resolve physical constant for {parameter_key}: {exc}"
            ) from exc

    @staticmethod
    def _find_repo_root(start_path: Path) -> Optional[Path]:
        """Find the repository root by locating the relic directory."""
        for candidate in [start_path] + list(start_path.parents):
            if (candidate / "relic").is_dir():
                return candidate
        return None

    def _load_cross_dimension_rules(self) -> List[Dict[str, Any]]:
        """Load cross-dimension authorization rules from YAML."""
        root = self._find_repo_root(Path(__file__).resolve())
        if root is None:
            raise RuntimeError(
                "Unable to locate repository root for cross-dimension rules."
            )
        rules_path = root / "relic" / "parameters" / "schemas" / "cross_dimension_rules.yaml"
        if not rules_path.exists():
            raise FileNotFoundError(
                f"Cross-dimension rules file does not exist: {rules_path}"
            )
        try:
            data = yaml.safe_load(rules_path.read_text(encoding="utf-8-sig")) or {}
        except (OSError, ValueError, yaml.YAMLError) as exc:
            raise RuntimeError(
                f"Failed to load cross-dimension rules from {rules_path}: {exc}"
            ) from exc
        rules = data.get("rules", [])
        if not isinstance(rules, list):
            raise ValueError("Cross-dimension schema rules must be a list.")
        return rules

    def _get_cross_dimension_rules(self) -> List[Dict[str, Any]]:
        if self._cross_dimension_rules is None:
            self._cross_dimension_rules = self._load_cross_dimension_rules()
        return self._cross_dimension_rules

    def _match_cross_dimension_rule(
        self,
        variable_name: str,
        source_units: str,
        target_units: str,
    ) -> Optional[Dict[str, Any]]:
        rules = self._get_cross_dimension_rules()
        var_name = (variable_name or "").strip().lower()
        normalized_name = self._normalize_variable_name(var_name).strip().lower()
        candidate_names = {var_name, normalized_name}
        for rule in rules:
            if rule.get("lhs") != source_units or rule.get("rhs") != target_units:
                continue
            variables = rule.get("variables") or []
            if variables:
                variables = [str(v).strip().lower() for v in variables]
                if not candidate_names.intersection(variables):
                    continue
            return rule
        return None

    @staticmethod
    def _missing_context_fields(
        context: ConversionContext, required: Dict[str, Any]
    ) -> List[str]:
        missing = []
        for field, unit in required.items():
            if getattr(context, field, None) is None:
                if unit:
                    missing.append(f"{field} ({unit})")
                else:
                    missing.append(str(field))
        return missing

    def _is_cross_dimension(self, source_units: str, target_units: str) -> bool:
        source_dim = self.get_dimension(source_units)
        target_dim = self.get_dimension(target_units)
        if source_dim and target_dim and source_dim != target_dim:
            return True
        return False

    @staticmethod
    def _is_temperature_affine_pair(source_units: str, target_units: str) -> bool:
        """
        Return True for authorized affine temperature unit pairs.

        Notes:
        - Keep case-sensitive 'C' (Coulomb) handling in get_dimension().
        - This short-circuit applies only when the selected conversion method
          is explicitly a temperature affine transform.
        """
        pair = (source_units.strip(), target_units.strip())
        allowed_pairs = {
            ("K", "C"),
            ("K", "degC"),
            ("K", "celsius"),
            ("C", "K"),
            ("degC", "K"),
            ("celsius", "K"),
        }
        return pair in allowed_pairs

    def _enforce_cross_dimension_rules(
        self,
        variable_name: str,
        source_units: str,
        target_units: str,
        context: ConversionContext,
        method_name: str,
    ) -> None:
        # Temperature affine transforms are authorized even when
        # get_dimension("C") resolves to electric_charge.
        if method_name in {"celsius_to_kelvin", "kelvin_to_celsius"}:
            if self._is_temperature_affine_pair(source_units, target_units):
                return

        rule = self._match_cross_dimension_rule(variable_name, source_units, target_units)
        if rule:
            required = rule.get("require_context") or {}
            missing = self._missing_context_fields(context, required)
            if missing and rule.get("forbid_if_missing_context", False):
                missing_str = ", ".join(missing)
                raise ValueError(
                    "Missing required conversion context: "
                    f"{missing_str} for {source_units} -> {target_units}"
                )
            expected_transform = rule.get("transform")
            if expected_transform and expected_transform != method_name:
                logger.warning(
                    "Cross-dimension rule '%s' expects '%s' but using '%s'",
                    rule.get("name", "unnamed"),
                    expected_transform,
                    method_name,
                )
            return

        if self._is_cross_dimension(source_units, target_units):
            raise ValueError(
                "Cross-dimension conversion not authorized: "
                f"{variable_name} {source_units} -> {target_units}. "
                "Add rule to cross_dimension_rules.yaml"
            )

    def _apply_context_defaults(self, context: ConversionContext) -> ConversionContext:
        if not getattr(context, "allow_defaults", True):
            return context

        if context.soil_depth is None:
            if self._parameter_manager_cache is None:
                from ...parameters import get_global_manager

                self._parameter_manager_cache = get_global_manager()
            pm = self._parameter_manager_cache
            if pm is None:
                raise RuntimeError(
                    "ParameterManager is required for conversion context defaults."
                )

            from ...parameters.namespaces import ParameterNamespace, resolve_namespace

            conv_ns = resolve_namespace(ParameterNamespace.CONVERSION_CONTEXT)
            value = pm.get(
                "soil_depth",
                namespace=conv_ns,
                units="m",
                expected_type=float,
            )
            if value is not None:
                context.soil_depth = float(value)
        if context.temperature is None:
            context.temperature = self._get_physical_constant(
                "reference_temperature"
            )
        if context.pressure is None:
            context.pressure = self._get_physical_constant(
                "reference_pressure"
            )
        if context.water_density is None:
            context.water_density = self._get_physical_constant(
                "water_density"
            )

        return context

    def _hydrate_context(self, context: Optional[ConversionContext]) -> ConversionContext:
        return self._apply_context_defaults(context or ConversionContext())
    
    @classmethod
    def _normalize_variable_name(cls, variable_name: str) -> str:
        """
        Normalize variable name to its canonical form using VARIABLE_ALIASES.

        This eliminates the need for duplicate dispatch table entries.
        If the name is not found in VARIABLE_ALIASES, it is returned as-is.
        """
        if not variable_name:
            return 'generic'
        canonical = cls.VARIABLE_ALIASES.get(variable_name.lower())
        return canonical if canonical is not None else variable_name

    def can_convert(
        self,
        from_unit: str,
        to_unit: str,
        variable_name: str = 'generic'
    ) -> bool:
        """
        Check if conversion between units is supported.
        
        Parameters:
            from_unit : str
                Source unit specification
            to_unit : str
                Target unit specification
            variable_name : str
                CF standard name or variable identifier (default: 'generic')
                
        Returns:
            bool: True if conversion is supported, False otherwise
        """
        # Identity is valid only for labels recognized by the canonical parser.
        if from_unit == to_unit:
            from .dimensionality import dimensionality_from_unit

            return dimensionality_from_unit(from_unit) is not None
        
        # Normalize variable name before lookup
        normalized = self._normalize_variable_name(variable_name)
        conversion_key = (normalized, from_unit, to_unit)
        if conversion_key in self.UNIT_CONVERSIONS:
            return True
        if ('generic', from_unit, to_unit) in self.UNIT_CONVERSIONS:
            return True
        if (
            self._declares_liquid_water_equivalent(variable_name)
            and self._liquid_water_equivalent_factor(from_unit, to_unit) is not None
        ):
            return True
        if self._element_mass_amount_factor(from_unit, to_unit) is not None:
            return True
        if self._element_label_factor(variable_name, from_unit, to_unit) is not None:
            return True
        return self._same_dimension_scale_factor(from_unit, to_unit) is not None

    @classmethod
    def _declares_liquid_water_equivalent(cls, variable_name: Optional[str]) -> bool:
        """True when the variable is declared a liquid-water quantity (by name family)."""
        if not variable_name:
            return False
        name = variable_name.strip().lower()
        return (
            name.startswith("lwe_")
            or cls._normalize_variable_name(name) in cls._LIQUID_WATER_EQUIVALENT_CATEGORIES
        )

    def _liquid_water_equivalent_factor(
        self, source_units: str, target_units: str
    ) -> Optional[float]:
        """Return f with target = f * source for a liquid-water depth <-> areal mass pair.

        References: CF Standard Name Table v94 ("lwe" means liquid water equivalent;
            lwe_thickness_of_X_amount "means the vertical extent of a layer of liquid
            water having the same mass per unit area"); UDUNITS-2 for the label scales.
        Mathematical expression: F = rho_w * d (depth or depth rate to areal mass or
            mass flux) and d = F / rho_w (reverse); f = rho_w * U(1; source * kg m-3 ->
            target) or f = U(1; source / kg m-3 -> target) / rho_w.
        Parameter meanings: d liquid-water-equivalent depth (rate); F areal water mass
            (flux); rho_w density of liquid water from the physical-constant SSOT
            relic.physical.constants.water.density_liquid; U the UDUNITS scale factor.
        Units: d [m] or [m s-1] (any length or length-per-time label, e.g. mm day-1,
            mm s-1, mm hr-1); F [kg m-2] or [kg m-2 s-1] (any matching label);
            rho_w [kg m-3]; f carries target/source.
        Domain/constraints: only for variables declared liquid water by name family
            (_declares_liquid_water_equivalent); rho_w is the SSOT constant, never a
            state-varying density; both labels must parse with the RELIC dimensionality
            owner and UDUNITS, differ exactly by the dimension of density, and have zero
            offset; otherwise None.
        Sign convention: rho_w > 0, so the sign of the flux is preserved.
        """
        from .dimensionality import dimensionality_from_unit

        source_dimension = dimensionality_from_unit(source_units)
        target_dimension = dimensionality_from_unit(target_units)
        if source_dimension is None or target_dimension is None:
            return None
        try:
            source = Unit(source_units)
            target = Unit(target_units)
            density_unit = Unit("kg m-3")
        except (TypeError, ValueError):
            return None
        if source.is_unknown() or target.is_unknown():
            return None
        rho_w = self._get_physical_constant("relic.physical.constants.water.density_liquid")
        for combined, scale in ((source * density_unit, rho_w), (source / density_unit, 1.0 / rho_w)):
            if not combined.is_convertible(target):
                continue
            if float(combined.convert(0.0, target)) != 0.0:
                return None
            factor = float(combined.convert(1.0, target)) * scale
            if not math.isfinite(factor) or factor <= 0.0:
                return None
            return factor
        return None

    @staticmethod
    def _same_dimension_scale_factor(source_units: str, target_units: str) -> Optional[float]:
        """Return the exact linear factor of a same-dimension UDUNITS pair.

        References: UDUNITS-2 (Unidata), the unit engine CFConventions uses for
            the standard-name/canonical-unit relation; CF Conventions 1.13,
            Section 3.1; BIPM SI Brochure, 9th edition (2019), Section 2.3.
        Mathematical expression: v_target = f * v_source with
            f = U(1; source -> target) and U(0; source -> target) = 0.
        Parameter meanings: f is the pure scale factor between two units of
            one physical dimension (for example day-1 -> s-1, g -> kg, hPa -> kPa).
        Units: f is dimensionless; v_source and v_target carry the labels.
        Domain/constraints: both labels must parse with the RELIC dimensionality
            owner to the same SI vector and with UDUNITS as convertible units;
            affine pairs (non-zero offset, such as degF -> K), different
            dimensions, element-annotated versus plain labels (UDUNITS reads
            "C" as coulomb), and unparseable labels return None, so such pairs
            still need an explicit rule in UNIT_CONVERSIONS.
        Sign convention: a positive factor preserves the sign of the value.
        """
        from fractions import Fraction
        from .dimensionality import (
            _normalize_unit_expression,
            _TOKEN_PATTERN,
            dimensionality_from_unit,
        )

        source_dimension = dimensionality_from_unit(source_units)
        target_dimension = dimensionality_from_unit(target_units)
        if source_dimension is None or target_dimension is None:
            return None
        if not source_dimension.is_compatible(target_dimension):
            return None
        tokens_by_unit = [
            _normalize_unit_expression(label).split()
            for label in (source_units, target_units)
        ]
        denominators = [
            Fraction(match.group("power")).denominator
            for tokens in tokens_by_unit for token in tokens
            if (match := _TOKEN_PATTERN.fullmatch(token)) is not None
            and match.group("power") is not None
        ]
        power_multiplier = math.lcm(*denominators) if denominators else 1
        coefficient_ratio = Fraction(1)
        if power_multiplier > 1:
            # References: BIPM SI Brochure (2019), derived units and prefix scaling;
            #   UDUNITS-2 integer-power syntax; RELIC dimensionality's canonical tokens.
            # Mathematical expression: f=(a_source/a_target)*[U(1; source^q -> target^q)]^(1/q),
            #   q=lcm(denominators of the rational exponents of both unit expressions).
            # Parameter meanings: f scale; a numerical coefficient; q integer lifting power; U UDUNITS.
            # Units: f is the ratio of source to target units; q is dimensionless.
            # Domain/constraints: same SI vector; positive linear scales only, no affine units.
            #   UDUNITS can misread Pa0.5 as a dimensionless number, so never parse the
            #   fractional spelling directly. Shared tokens also preserve compound units.
            #   Numerical coefficients are cancelled exactly before scaling, never raised to q.
            # Sign convention: positive f preserves the sign and gradient of the input value.
            lifted = []
            coefficients = []
            for tokens in tokens_by_unit:
                pieces = []
                numerical_coefficient = Fraction(1)
                divide_next = False
                for token in tokens:
                    if token == "/":
                        divide_next = True
                        continue
                    sign = -1 if divide_next else 1
                    divide_next = False
                    match = _TOKEN_PATTERN.fullmatch(token)
                    if match is None:
                        try:
                            coefficient = Fraction(token)
                            finite_coefficient = math.isfinite(float(coefficient))
                        except (ValueError, OverflowError):
                            return None
                        if coefficient <= 0 or not finite_coefficient:
                            return None
                        numerical_coefficient *= coefficient ** sign
                        continue
                    symbol = match.group("symbol")
                    exponent = Fraction(match.group("power") or "1") * sign * power_multiplier
                    try:
                        base = Unit(symbol)
                        kelvin = Unit("K")
                    except (TypeError, ValueError):
                        return None
                    if base.is_convertible(kelvin) and float(base.convert(0.0, kelvin)) != 0.0:
                        return None
                    pieces.append(symbol + str(int(exponent)))
                lifted.append(" ".join(pieces))
                coefficients.append(numerical_coefficient)
            source_units, target_units = lifted
            coefficient_ratio = coefficients[0] / coefficients[1]
        try:
            source = Unit(source_units)
            target = Unit(target_units)
        except (TypeError, ValueError):
            return None
        if source.is_unknown() or target.is_unknown():
            return None
        if not source.is_convertible(target):
            return None
        if float(source.convert(0.0, target)) != 0.0:
            return None
        factor = float(source.convert(1.0, target))
        if not math.isfinite(factor) or factor <= 0.0:
            return None
        try:
            factor = float(coefficient_ratio * Fraction(factor ** (1.0 / power_multiplier)))
        except OverflowError:
            return None
        return factor if math.isfinite(factor) and factor > 0.0 else None

    @staticmethod
    def _element_label_pair(
        variable_name: Optional[str], source_units: str, target_units: str
    ) -> Optional[Tuple[str, str, Tuple[str, ...], str]]:
        """Split a (labelled, plain) unit pair whose CF name states an element basis.

        Returns (labelled, plain, labels, basis) when exactly one side carries
        element labels and variable_name is a CF standard name with an element
        basis (CFConventions.element_basis_of_standard_name); otherwise None.
        """
        from .cf_conventions import CFConventions
        from .dimensionality import element_annotations_from_unit

        source_labels = element_annotations_from_unit(source_units)
        target_labels = element_annotations_from_unit(target_units)
        if source_labels is None or target_labels is None or bool(source_labels) == bool(target_labels):
            return None
        if not variable_name or not CFConventions.is_valid_standard_name(variable_name):
            return None
        basis = CFConventions.element_basis_of_standard_name(variable_name)
        if basis is None:
            return None
        if source_labels:
            return source_units, target_units, source_labels, basis
        return target_units, source_units, target_labels, basis

    def _element_label_mismatch(
        self, variable_name: Optional[str], source_units: str, target_units: str
    ) -> Optional[str]:
        """Error message when the element label contradicts the element the CF name states."""
        pair = self._element_label_pair(variable_name, source_units, target_units)
        if pair is None:
            return None
        labelled, _, labels, basis = pair
        if set(labels) == {basis}:
            return None
        from .cf_conventions import CFConventions

        return (
            f"element label {sorted(set(labels))} of {labelled!r} contradicts the "
            f"{CFConventions._ELEMENT_WORDS[basis]} basis stated by the CF standard name {variable_name!r}"
        )

    def _element_label_factor(
        self, variable_name: Optional[str], source_units: str, target_units: str
    ) -> Optional[float]:
        """Return f with target = f * source when only the element label differs.

        References: CF Standard Name guidelines (the "expressed_as" construction and
            the element basis of "<element>_mass_..." and "..._mass_<q>_of_<element>"
            names: the canonical units exclude the element, which the name states);
            UDUNITS-2 for the scale of the remaining unit; BIPM SI Brochure, 9th
            edition (2019), Section 2.3.
        Mathematical expression: v_target = f * v_source with f = U(1; strip(source)
            -> target) for a labelled source, or f = U(1; source -> strip(target)) for
            a labelled target, where strip removes the element labels.
        Parameter meanings: strip(u) = dimensionality.unit_without_element_annotations(u);
            U the pure scale factor of _same_dimension_scale_factor; the element label
            is the one dimensionality.element_annotations_from_unit reports and the
            basis is CFConventions.element_basis_of_standard_name(variable_name).
        Units: f is dimensionless; e.g. kg C m-2 s-1 -> kg m-2 s-1 gives f = 1 and
            g C m-2 s-1 -> kg m-2 s-1 gives f = 1e-3.
        Domain/constraints: exactly one side carries element labels, all labels are
            the one element the CF name states, the labelled unit is equivalent to the
            name (CFConventions.are_standard_name_units_equivalent) and the stripped
            and plain units have the same SI vector; a contradicting element is an
            error (_element_label_mismatch); any other pair returns None.
        Sign convention: f > 0, so the sign of the value (and of a flux) is kept.
        """
        pair = self._element_label_pair(variable_name, source_units, target_units)
        if pair is None:
            return None
        labelled, plain, labels, basis = pair
        if set(labels) != {basis}:
            return None
        from .cf_conventions import CFConventions
        from .dimensionality import unit_without_element_annotations

        if not CFConventions.are_standard_name_units_equivalent(variable_name, labelled):
            return None
        stripped = unit_without_element_annotations(labelled)
        if labelled == source_units:
            return self._same_dimension_scale_factor(stripped, plain)
        return self._same_dimension_scale_factor(plain, stripped)

    def get_dimension(self, unit: str) -> Optional[str]:
        """
        Return the canonical physical dimension for a unit string.

        This method is unified with the canonical dimension taxonomy used
        by the parameter system (see architecture_overview.md section 6.7.2).
        It is used for cross-dimension safety checks in convert().
        """
        from .dimensionality import dimension_category_from_unit

        return dimension_category_from_unit(unit)
        
    def convert(
        self,
        data: Union[Tensor, float, int, 'np.ndarray'],
        variable_name: str,
        source_units: str,
        target_units: str,
        context: Optional[ConversionContext] = None
    ) -> Union[Tensor, float, int, np.ndarray]:
        """
        Convert data between units with conservation validation.

        Supports Tensor, float, int and numpy arrays. Identity conversion preserves
        the original object. Arithmetic conversion preserves floating array precision;
        integer data is promoted and Python scalar results are floats.

        Parameters:
            data : Tensor, float, int, or numpy.ndarray
                Data to convert (auto-normalized to Tensor internally)
            variable_name : str
                CF standard name or variable identifier
            source_units : str
                Source unit specification
            target_units : str
                Target unit specification
            context : ConversionContext
                Additional context for conversion (e.g., soil depth)

        Returns:
            Tensor, float, int or ndarray: Converted values without integer truncation.

        Raises:
            ValueError: If conversion is not supported or context is insufficient
        """
        # Validate the container before identity handling, without narrowing precision.
        is_scalar = isinstance(data, (int, float))
        is_numpy = isinstance(data, np.ndarray)
        if not isinstance(data, (int, float, np.ndarray, Tensor)):
            raise TypeError(
                f"Unsupported data type for unit conversion: {type(data)}. "
                f"Expected Tensor, float, int, or numpy.ndarray"
            )

        # Fast path: no numerical conversion is needed, but the unit label must
        # still be recognized by the canonical dimensionality owner.
        if source_units == target_units:
            from .dimensionality import dimensionality_from_unit

            if dimensionality_from_unit(source_units) is None:
                raise ValueError(
                    "Unknown or unsupported unit label for identity conversion: "
                    f"{source_units!r}"
                )
            return data

        # Normalize variable name before lookup
        normalized = self._normalize_variable_name(variable_name)
        conversion_key = (normalized, source_units, target_units)

        if conversion_key not in self.UNIT_CONVERSIONS:
            resolution_keys = [
                ('generic', source_units, target_units),
                ('molar_flux', source_units, target_units),
            ]
            for key in resolution_keys:
                if key in self.UNIT_CONVERSIONS:
                    conversion_key = key
                    break
            else:
                if self._declares_liquid_water_equivalent(variable_name):
                    factor = self._liquid_water_equivalent_factor(source_units, target_units)
                    if factor is not None:
                        scaled = self._as_float_tensor(data) * factor
                        return self._format_output(scaled, is_scalar, is_numpy)
                element_factor = self._element_mass_amount_factor(source_units, target_units)
                if element_factor is not None:
                    scaled = self._as_float_tensor(data) * element_factor
                    return self._format_output(scaled, is_scalar, is_numpy)
                # An element label the CF name states is removed (or added) at the
                # scale of the remaining unit; a contradicting element is an error.
                mismatch = self._element_label_mismatch(variable_name, source_units, target_units)
                if mismatch is not None:
                    raise ValueError(mismatch)
                label_factor = self._element_label_factor(variable_name, source_units, target_units)
                if label_factor is not None:
                    scaled = self._as_float_tensor(data) * label_factor
                    return self._format_output(scaled, is_scalar, is_numpy)
                # Same-dimension UDUNITS pairs (for example a CF-consistent
                # label and its canonical unit) scale by one exact factor; the
                # rule and its limits are documented on _same_dimension_scale_factor.
                factor = self._same_dimension_scale_factor(source_units, target_units)
                if factor is None:
                    raise ValueError(
                        f"Unsupported conversion: {variable_name} (normalized: {normalized}) "
                        f"from {source_units} to {target_units}"
                    )
                scaled = self._as_float_tensor(data) * factor
                return self._format_output(scaled, is_scalar, is_numpy)

        method_name = self.UNIT_CONVERSIONS[conversion_key]
        method = getattr(self, method_name)

        # Apply conversion
        hydrated_context = self._hydrate_context(context)
        self._enforce_cross_dimension_rules(
            variable_name,
            source_units,
            target_units,
            hydrated_context,
            method_name,
        )
        if method_name == "identity":
            return method(data, hydrated_context)

        converted_data = method(self._as_float_tensor(data), hydrated_context)

        # Skip conservation validation for unit conversions
        # (conservation is handled by the conversion methods themselves)

        return self._format_output(converted_data, is_scalar, is_numpy)

    @staticmethod
    def _as_float_tensor(data: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        """Return a floating tensor view of the input without narrowing precision."""
        if isinstance(data, (int, float)):
            return torch.tensor(data, dtype=torch.float64)
        if isinstance(data, np.ndarray):
            array = data
            if array.dtype.kind in "biu":
                array = array.astype(np.float64)
            elif not array.dtype.isnative:
                array = array.astype(array.dtype.newbyteorder("="))
            if any(stride < 0 for stride in array.strides) or not array.flags.writeable:
                array = array.copy(order="C")
            return torch.from_numpy(array)
        if not data.is_floating_point() and not torch.is_complex(data):
            return data.to(dtype=torch.float64)
        return data

    def _format_output(
        self,
        data: Tensor,
        is_scalar: bool,
        is_numpy: bool
    ) -> Union[Tensor, float, int, np.ndarray]:
        """Restore the input container without truncating fractional results."""

        if is_scalar:
            return float(data.item())
        elif is_numpy:
            # Return numpy array
            return data.detach().cpu().numpy()
        else:
            # Return tensor
            return data

    def external_radiation_langley_to_wm2(self, data: Tensor, context: ConversionContext) -> Tensor:
        """Convert Langley day-1 to W m-2 for external model data."""
        # References: WMO unit definitions; 1 Langley = 1 cal cm-2.
        # Mathematical expression: Q = L * (1 / j_to_cal) * 1e4 / t_day.
        # Parameter meanings: L is Langley per day, j_to_cal converts joules to calories.
        # Units: L [cal cm-2 day-1], Q [J m-2 s-1] = [W m-2].
        # Open route: unit conversion only; no radiative energy is created or removed.
        joule_to_calorie = self._get_physical_constant(
            "relic.physical.constants.conversions.j_to_cal"
        )
        seconds_per_day = self._get_physical_constant("seconds_per_day")
        return data * (1.0 / joule_to_calorie) * 1.0e4 / seconds_per_day

    def external_radiation_cal_to_wm2(self, data: Tensor, context: ConversionContext) -> Tensor:
        """Convert cal cm-2 day-1 to W m-2 for external model data."""
        return self.external_radiation_langley_to_wm2(data, context)

    @staticmethod
    def _conductance_input_tensor(data: Tensor) -> Tensor:
        """Validate a physical gas conductance without changing its graph."""
        if not bool(torch.all(torch.isfinite(data)).item()):
            raise ValueError("conductance must contain only finite values")
        if bool(torch.any(data < 0).item()):
            raise ValueError("conductance must be non-negative")
        return data

    @staticmethod
    def _conductance_context_tensor(
        value: Union[float, Tensor],
        reference: Tensor,
        field_name: str,
    ) -> Tensor:
        """Place one thermodynamic context value beside the conductance tensor."""
        if isinstance(value, Tensor):
            tensor = value.to(device=reference.device, dtype=reference.dtype)
        else:
            tensor = torch.as_tensor(
                value,
                device=reference.device,
                dtype=reference.dtype,
            )
        if not bool(torch.all(torch.isfinite(tensor)).item()):
            raise ValueError(f"{field_name} must contain only finite values")
        if bool(torch.any(tensor <= 0).item()):
            raise ValueError(f"{field_name} must be greater than zero")
        return tensor

    def _conductance_thermodynamic_context(
        self,
        data: Tensor,
        context: ConversionContext,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return validated conductance, temperature, and pressure tensors."""
        if context.temperature is None:
            raise ValueError("temperature is required for conductance conversion")
        if context.pressure is None:
            raise ValueError("pressure is required for conductance conversion")
        conductance = self._conductance_input_tensor(data)
        temperature = self._conductance_context_tensor(
            context.temperature,
            conductance,
            "temperature",
        )
        pressure = self._conductance_context_tensor(
            context.pressure,
            conductance,
            "pressure",
        )
        return conductance, temperature, pressure

    # References: Gerlein-Safdi et al. (2018), Eq. 13; Jones (1992).
    # Mathematical expression: g_ms = g_mol * R * T / P.
    # Parameter meanings: g_mol [mol m-2 s-1], R [J mol-1 K-1], T [K], P [Pa].
    # Units: g_ms [m s-1]. This conversion has no mass or energy source.
    def conductance_mol_to_ms(self, data: Tensor, context: ConversionContext) -> Tensor:
        """Convert gas conductance from mol m-2 s-1 to m s-1."""
        conductance, temperature, pressure = (
            self._conductance_thermodynamic_context(data, context)
        )
        R = self._resolve_physical_constant("universal.gas_constant", units="J mol-1 K-1")
        return conductance * R * temperature / pressure

    # References: Gerlein-Safdi et al. (2018), Eq. 13; Jones (1992).
    # Mathematical expression: g_mol = g_ms * P / (R * T).
    # Parameter meanings: g_ms [m s-1], R [J mol-1 K-1], T [K], P [Pa].
    # Units: g_mol [mol m-2 s-1]. This conversion has no mass or energy source.
    def conductance_ms_to_mol(self, data: Tensor, context: ConversionContext) -> Tensor:
        """Convert gas conductance from m s-1 to mol m-2 s-1."""
        conductance, temperature, pressure = (
            self._conductance_thermodynamic_context(data, context)
        )
        R = self._resolve_physical_constant("universal.gas_constant", units="J mol-1 K-1")
        return conductance * pressure / (R * temperature)

    def volumetric_to_mass_soil_moisture(
        self, 
        volumetric_content: Tensor, 
        context: ConversionContext
    ) -> Tensor:
        """
        Convert volumetric soil moisture (m^3/m^3) to mass (kg/m^2).
        
        Formula: mass_content = volumetric_content * soil_depth * water_density
        """
        # References: Hillel (1998), Environmental Soil Physics; CF-1.8.
        # Mathematical expression: m_w = theta * L * rho_w (elementwise).
        # Parameter meanings: theta is volumetric content, L is layer depth, rho_w is water density.
        # Units: theta [m3 m-3], L [m], rho_w [kg m-3], m_w [kg m-2].
        # Domain/constraints: L and rho_w are scalars or tensors broadcastable to theta
        # (per-cell context, never averaged); both are required, none is defaulted here.
        # Sign convention: the sign of theta carries through unchanged (positive L, rho_w).
        # Open route: unit conversion only; no water enters or leaves the ledger.
        if context.soil_depth is None:
            raise ValueError("Soil depth required for volumetric to mass conversion")
        if context.water_density is None:
            raise ValueError("Water density required for volumetric to mass conversion")
            
        return volumetric_content * context.soil_depth * context.water_density
    
    def mass_to_volumetric_soil_moisture(
        self, 
        mass_content: Tensor, 
        context: ConversionContext
    ) -> Tensor:
        """
        Convert mass soil moisture (kg/m^2) to volumetric (m^3/m^3).
        
        Formula: volumetric_content = mass_content / (soil_depth * water_density)
        """
        # References: Hillel (1998), Environmental Soil Physics; CF-1.8.
        # Mathematical expression: theta = m_w / (L * rho_w) (elementwise).
        # Parameter meanings: m_w is areal water mass, L is layer depth, rho_w is water density.
        # Units: m_w [kg m-2], L [m], rho_w [kg m-3], theta [m3 m-3].
        # Domain/constraints: L and rho_w are scalars or tensors broadcastable to m_w
        # (per-cell context, never averaged); both are required, none is defaulted here.
        # Sign convention: the sign of m_w carries through unchanged (positive L, rho_w).
        # Open route: unit conversion only; no water enters or leaves the ledger.
        if context.soil_depth is None:
            raise ValueError("Soil depth required for mass to volumetric conversion")
        if context.water_density is None:
            raise ValueError("Water density required for mass to volumetric conversion")
            
        volumetric_content = mass_content / (context.soil_depth * context.water_density)

        return volumetric_content

    @staticmethod
    def soil_moisture_m3m3_to_kgm2(
        volumetric_content: Tensor,
        soil_depth: Union[float, Tensor],
        water_density: Optional[Union[float, Tensor]] = None,
    ) -> Tensor:
        """
        Convenience wrapper: volumetric soil moisture [m3 m-3] -> mass [kg m-2].
        soil_depth and water_density are required (scalar or broadcastable tensor);
        no default is applied (ConversionContext.allow_defaults is False).
        """
        converter = unit_converter
        ctx = ConversionContext(soil_depth=soil_depth, water_density=water_density)
        ctx = converter._apply_context_defaults(ctx)
        return converter.volumetric_to_mass_soil_moisture(volumetric_content, ctx)

    @staticmethod
    def soil_moisture_kgm2_to_m3m3(
        mass_content: Tensor,
        soil_depth: Union[float, Tensor],
        water_density: Optional[Union[float, Tensor]] = None,
    ) -> Tensor:
        """
        Convenience wrapper: mass soil moisture [kg m-2] -> volumetric [m3 m-3].
        soil_depth and water_density are required (scalar or broadcastable tensor);
        no default is applied (ConversionContext.allow_defaults is False).
        """
        converter = unit_converter
        ctx = ConversionContext(soil_depth=soil_depth, water_density=water_density)
        ctx = converter._apply_context_defaults(ctx)
        return converter.mass_to_volumetric_soil_moisture(mass_content, ctx)
    
    # References: SI Brochure, 9th edition, temperature scales.
    # Mathematical expression: T_K = T_C + Delta_CK.
    # Parameter meanings: Delta_CK is the Celsius-to-Kelvin offset from the physical SSOT.
    # Units: T_C [degC], Delta_CK [K], T_K [K].
    # Open route: affine unit conversion only; no energy enters or leaves the ledger.
    def celsius_to_kelvin(self, temp_celsius: Tensor, context: ConversionContext) -> Tensor:
        """Convert temperature from Celsius to Kelvin."""
        offset = self._get_physical_constant("celsius_to_kelvin")
        return temp_celsius + offset
    
    def kelvin_to_celsius(self, temp_kelvin: Tensor, context: ConversionContext) -> Tensor:
        """Convert temperature from Kelvin to Celsius."""
        offset = self._get_physical_constant("celsius_to_kelvin")
        return temp_kelvin - offset
    
    def hpa_to_pa(self, pressure_hpa: Tensor, context: ConversionContext) -> Tensor:
        """Convert pressure from hPa to Pa."""
        return pressure_hpa * 100.0
    
    def pa_to_hpa(self, pressure_pa: Tensor, context: ConversionContext) -> Tensor:
        """Convert pressure from Pa to hPa."""
        return pressure_pa / 100.0
    
    # References: Ideal gas law; CF-1.8.
    # Mathematical expression: C = (x * P) / (R * T), x = ppm * 1e-6.
    # Parameter meanings: C [mol m-3], x [mol mol-1], P [Pa], T [K], R [J mol-1 K-1].
    def ppm_to_molar_concentration(
        self,
        concentration_ppm: Tensor,
        context: ConversionContext,
    ) -> Tensor:
        """Convert gas concentration from ppm to mol/m^3."""
        if context.temperature is None or context.pressure is None:
            raise ValueError("Temperature and pressure required for ppm to molar conversion")

        R = self._resolve_physical_constant("universal.gas_constant", units="J mol-1 K-1")
        mole_fraction = concentration_ppm * 1e-6
        return (mole_fraction * context.pressure) / (R * context.temperature)
    
    # References: Ideal gas law; CF-1.8.
    # Mathematical expression: x = C * R * T / P; ppm = x * 1e6.
    # Parameter meanings: C [mol m-3], x [mol mol-1], P [Pa], T [K], R [J mol-1 K-1].
    def molar_concentration_to_ppm(
        self,
        molar_conc: Tensor,
        context: ConversionContext,
    ) -> Tensor:
        """Convert gas concentration from mol/m^3 to ppm."""
        if context.temperature is None or context.pressure is None:
            raise ValueError("Temperature and pressure required for molar to ppm conversion")

        R = self._resolve_physical_constant("universal.gas_constant", units="J mol-1 K-1")
        mole_fraction = (molar_conc * R * context.temperature) / context.pressure
        return mole_fraction * 1e6

    def ppm_to_mole_fraction(self, concentration_ppm: Tensor, context: ConversionContext) -> Tensor:
        """
        Convert concentration from ppm to mol mol-1 (mole fraction).

        Mathematical Expression:
            x = ppm * 1e-6

        Parameters:
            concentration_ppm : Tensor
                Concentration in ppm
            context : ConversionContext
                Conversion context (not used for this conversion)

        Returns:
            Tensor: Mole fraction in mol mol-1

        References:
            - CF-1.8 Convention
            - SI Brochure (ppm definition)
        """
        return concentration_ppm * self._resolve_physical_constant("conversions.ppm_to_fraction")

    def mole_fraction_to_ppm(self, mole_fraction: Tensor, context: ConversionContext) -> Tensor:
        """
        Convert concentration from mol mol-1 (mole fraction) to ppm.

        Mathematical Expression:
            ppm = x * 1e6

        Parameters:
            mole_fraction : Tensor
                Mole fraction in mol mol-1
            context : ConversionContext
                Conversion context (not used for this conversion)

        Returns:
            Tensor: Concentration in ppm

        References:
            - CF-1.8 Convention
            - SI Brochure (ppm definition)
        """
        return mole_fraction / self._resolve_physical_constant("conversions.ppm_to_fraction")

    def ppb_to_mole_fraction(self, concentration_ppb: Tensor, context: ConversionContext) -> Tensor:
        """Convert parts per billion to mole fraction.

        References: BIPM SI Brochure, 9th edition (2019).
        Mathematical expression: x = concentration_ppb * ppb_to_fraction.
        Parameter meanings: ppb_to_fraction is the existing physical SSOT scale.
        Units: ppb to mol mol-1; no density or temperature dependence.
        Domain/constraints: numeric values, retaining input dtype/device/autograd.
        Sign convention: positive scale preserves the input sign.
        """
        return concentration_ppb * self._resolve_physical_constant("conversions.ppb_to_fraction")

    def mole_fraction_to_ppb(self, mole_fraction: Tensor, context: ConversionContext) -> Tensor:
        """Convert mole fraction to parts per billion.

        References: BIPM SI Brochure, 9th edition (2019).
        Mathematical expression: concentration_ppb = x / ppb_to_fraction.
        Parameter meanings: ppb_to_fraction is the same SSOT scale as the inverse.
        Units: mol mol-1 to ppb; no density or temperature dependence.
        Domain/constraints: numeric values, retaining input dtype/device/autograd.
        Sign convention: positive scale preserves the input sign.
        """
        return mole_fraction / self._resolve_physical_constant("conversions.ppb_to_fraction")

    def co2_ppm_to_pa(self, concentration_ppm: Tensor, context: ConversionContext) -> Tensor:
        """
        Convert CO2 concentration from ppm to partial pressure in Pa using ambient pressure.
        """
        if context.pressure is None:
            raise ValueError("Pressure is required in ConversionContext for ppm to Pa conversion.")
        fraction = concentration_ppm * 1e-6
        return fraction * context.pressure

    def co2_pa_to_ppm(self, partial_pressure: Tensor, context: ConversionContext) -> Tensor:
        """
        Convert CO2 partial pressure from Pa to ppm using ambient pressure.
        """
        if context.pressure is None:
            raise ValueError("Pressure is required in ConversionContext for Pa to ppm conversion.")
        fraction = partial_pressure / context.pressure
        return fraction * 1e6

    def co2_mole_fraction_to_pa(self, mole_fraction: Tensor, context: ConversionContext) -> Tensor:
        """
        Convert CO2 mole fraction (mol mol-1) to partial pressure in Pa using ambient pressure.
        """
        if context.pressure is None:
            raise ValueError("Pressure is required in ConversionContext for mole fraction to Pa conversion.")
        return mole_fraction * context.pressure

    def co2_pa_to_mole_fraction(self, partial_pressure: Tensor, context: ConversionContext) -> Tensor:
        """
        Convert CO2 partial pressure in Pa to mole fraction (mol mol-1) using ambient pressure.
        """
        if context.pressure is None:
            raise ValueError("Pressure is required in ConversionContext for Pa to mole fraction conversion.")
        return partial_pressure / context.pressure

    # Basic unit conversion methods
    def meter_to_millimeter(self, data: Tensor, context: ConversionContext) -> Tensor:
        """Convert meters to millimeters."""
        return data * 1000.0

    def millimeter_to_meter(self, data: Tensor, context: ConversionContext) -> Tensor:
        """Convert millimeters to meters."""
        return data / 1000.0

    def meter_to_centimeter(self, data: Tensor, context: ConversionContext) -> Tensor:
        """Convert meters to centimeters."""
        return data * 100.0

    def centimeter_to_meter(self, data: Tensor, context: ConversionContext) -> Tensor:
        """Convert centimeters to meters."""
        return data / 100.0

    def meter_to_kilometer(self, data: Tensor, context: ConversionContext) -> Tensor:
        """Convert meters to kilometers."""
        return data / 1000.0

    def kilometer_to_meter(self, data: Tensor, context: ConversionContext) -> Tensor:
        """Convert kilometers to meters."""
        return data * 1000.0

    def identity(
        self, data: Union[Tensor, float, int, np.ndarray], context: ConversionContext,
    ) -> Union[Tensor, float, int, np.ndarray]:
        """No-op conversion used for unit alias harmonization."""
        if not context.allow_identity:
            raise ValueError(
                "Identity unit conversion requires explicit "
                "ConversionContext(allow_identity=True)."
            )
        return data

    # Mass conversion methods
    def gram_to_kilogram(self, data: Tensor, context: ConversionContext) -> Tensor:
        """
        Convert mass from grams to kilograms.
        
        Mathematical Expression:
            mass(kg) = mass(g) / 1000
        
        Parameters:
            data : Tensor
                Mass in grams
            context : ConversionContext
                Conversion context (not used for this conversion)
        
        Returns:
            Tensor: Mass in kilograms
        """
        return data / 1000.0

    def kilogram_to_gram(self, data: Tensor, context: ConversionContext) -> Tensor:
        """
        Convert mass from kilograms to grams.
        
        Mathematical Expression:
            mass(g) = mass(kg) * 1000
        
        Parameters:
            data : Tensor
                Mass in kilograms
            context : ConversionContext
                Conversion context (not used for this conversion)
        
        Returns:
            Tensor: Mass in grams
        """
        return data * 1000.0

    # Pressure conversion methods
    def pa_to_kpa(self, data: Tensor, context: ConversionContext) -> Tensor:
        """
        Convert pressure from Pascals to kiloPascals.
        
        Mathematical Expression:
            pressure(kPa) = pressure(Pa) / 1000
        
        Parameters:
            data : Tensor
                Pressure in Pascals
            context : ConversionContext
                Conversion context (not used for this conversion)
        
        Returns:
            Tensor: Pressure in kiloPascals
        """
        return data / 1000.0

    def kpa_to_pa(self, data: Tensor, context: ConversionContext) -> Tensor:
        """
        Convert pressure from kiloPascals to Pascals.
        
        Mathematical Expression:
            pressure(Pa) = pressure(kPa) * 1000
        
        Parameters:
            data : Tensor
                Pressure in kiloPascals
            context : ConversionContext
                Conversion context (not used for this conversion)
        
        Returns:
            Tensor: Pressure in Pascals
        """
        return data * 1000.0

    # Time conversion methods
    def second_to_hour(self, data: Tensor, context: ConversionContext) -> Tensor:
        """
        Convert time from seconds to hours.
        
        Mathematical Expression:
            time(hour) = time(s) / 3600
        
        Parameters:
            data : Tensor
                Time in seconds
            context : ConversionContext
                Conversion context (not used for this conversion)
        
        Returns:
            Tensor: Time in hours
        """
        seconds_per_hour = self._get_physical_constant("seconds_per_hour")
        return data / seconds_per_hour

    def hour_to_second(self, data: Tensor, context: ConversionContext) -> Tensor:
        """
        Convert time from hours to seconds.
        
        Mathematical Expression:
            time(s) = time(hour) * 3600
        
        Parameters:
            data : Tensor
                Time in hours
            context : ConversionContext
                Conversion context (not used for this conversion)
        
        Returns:
            Tensor: Time in seconds
        """
        seconds_per_hour = self._get_physical_constant("seconds_per_hour")
        return data * seconds_per_hour

    def day_to_second(self, data: Tensor, context: ConversionContext) -> Tensor:
        """
        Convert time from days to seconds.
        
        Mathematical Expression:
            time(s) = time(day) * 86400
        
        Parameters:
            data : Tensor
                Time in days
            context : ConversionContext
                Conversion context (not used for this conversion)
        
        Returns:
            Tensor: Time in seconds
        
        References:
            1 day = 24 hours = 24 * 3600 s = 86400 s
        """
        seconds_per_day = self._get_physical_constant("seconds_per_day")
        return data * seconds_per_day

    def second_to_day(self, data: Tensor, context: ConversionContext) -> Tensor:
        """
        Convert time from seconds to days.
        
        Mathematical Expression:
            time(day) = time(s) / 86400
        
        Parameters:
            data : Tensor
                Time in seconds
            context : ConversionContext
                Conversion context (not used for this conversion)
        
        Returns:
            Tensor: Time in days
        
        References:
            1 day = 86400 seconds
        """
        seconds_per_day = self._get_physical_constant("seconds_per_day")
        return data / seconds_per_day

    # Carbon flux conversion methods (critical for biosphere modeling)
    def carbon_flux_g_to_umol(self, data: Tensor, context: ConversionContext) -> Tensor:
        """
        Convert carbon flux from g C m-2 s-1 to umol C m-2 s-1.
        
        Mathematical Expression:
            flux(umol C m-2 s-1) = flux(g C m-2 s-1) * (1e6 umol/mol) / (M_C g/mol)
            where M_C = 12.011 g/mol (molar mass of carbon)
        
        Physical Meaning:
            Converts mass-based carbon flux to molar flux units commonly used
            in photosynthesis and respiration measurements.
        
        Parameters:
            data : Tensor
                Carbon flux in g C m-2 s-1
            context : ConversionContext
                Conversion context (not used for this conversion)
        
        Returns:
            Tensor: Carbon flux in umol C m-2 s-1
        
        References:
            - Farquhar et al. (1980). Planta, 149(1), 78-90.
            - Von Caemmerer & Farquhar (1981). Planta, 153(4), 376-387.
        
        Example:
            1 g C m-2 s-1 = (1 g / 12.011 g/mol) * 1e6 umol/mol
                         = 83,263.5 umol C m-2 s-1
        """
        mw_carbon = self._resolve_molecular_weight("molecular_weights.carbon")
        return data * (1e6 / mw_carbon)

    def carbon_flux_umol_to_g(self, data: Tensor, context: ConversionContext) -> Tensor:
        """
        Convert carbon flux from umol C m-2 s-1 to g C m-2 s-1.
        
        Mathematical Expression:
            flux(g C m-2 s-1) = flux(umol C m-2 s-1) * (M_C g/mol) / (1e6 umol/mol)
            where M_C = 12.011 g/mol (molar mass of carbon)
        
        Physical Meaning:
            Converts molar-based carbon flux back to mass flux units for
            carbon balance calculations and ecosystem modeling.
        
        Parameters:
            data : Tensor
                Carbon flux in umol C m-2 s-1
            context : ConversionContext
                Conversion context (not used for this conversion)
        
        Returns:
            Tensor: Carbon flux in g C m-2 s-1
        
        References:
            - Farquhar et al. (1980). Planta, 149(1), 78-90.
            - Von Caemmerer & Farquhar (1981). Planta, 153(4), 376-387.
        
        Example:
            83,263.5 umol C m-2 s-1 = (83263.5 umol * 12.011 g/mol) / 1e6 umol/mol
                                    = 1.0 g C m-2 s-1
        """
        mw_carbon = self._resolve_molecular_weight("molecular_weights.carbon")
        return data * (mw_carbon / 1e6)

    def carbon_flux_kg_to_umol(self, data: Tensor, context: ConversionContext) -> Tensor:
        """
        Convert carbon flux from kg C m-2 s-1 to umol C m-2 s-1 using ParameterManager weights.
        """
        grams = self.kilogram_to_gram(data, context)
        return self.carbon_flux_g_to_umol(grams, context)

    def carbon_flux_umol_to_kg(self, data: Tensor, context: ConversionContext) -> Tensor:
        """
        Convert carbon flux from umol C m-2 s-1 to kg C m-2 s-1 using ParameterManager weights.
        """
        grams = self.carbon_flux_umol_to_g(data, context)
        return self.gram_to_kilogram(grams, context)

    def carbon_flux_g_day_to_kg_s(self, data: Tensor, context: ConversionContext) -> Tensor:
        """
        Convert carbon flux from g C m-2 day-1 to kg C m-2 s-1.

        Mathematical Expression:
            F_kg_s = F_g_day / (1000 * 86400)

        Parameters:
            data : Tensor
                Carbon flux in g C m-2 day-1
            context : ConversionContext
                Conversion context (not used for this conversion)

        Returns:
            Tensor: Carbon flux in kg C m-2 s-1

        References:
            - CF-1.8 Convention
            - SI Brochure (day to second)
        """
        seconds_per_day = self._get_physical_constant("seconds_per_day")
        return data / (1000.0 * seconds_per_day)

    def carbon_flux_kg_s_to_g_day(self, data: Tensor, context: ConversionContext) -> Tensor:
        """
        Convert carbon flux from kg C m-2 s-1 to g C m-2 day-1.

        Mathematical Expression:
            F_g_day = F_kg_s * (1000 * 86400)

        Parameters:
            data : Tensor
                Carbon flux in kg C m-2 s-1
            context : ConversionContext
                Conversion context (not used for this conversion)

        Returns:
            Tensor: Carbon flux in g C m-2 day-1

        References:
            - CF-1.8 Convention
            - SI Brochure (day to second)
        """
        seconds_per_day = self._get_physical_constant("seconds_per_day")
        return data * (1000.0 * seconds_per_day)

    @staticmethod
    def _element_atomic_molar_mass(element: str) -> float:
        """Resolve one elemental molar mass from the physical-constant SSOT."""
        key = str(element).strip().lower()
        if key == "carbon":
            return UnitConverter._get_physical_constant(
                "relic.physical.constants.molecular_weights.carbon"
            )
        if key == "nitrogen":
            return UnitConverter._get_physical_constant(
                "relic.physical.constants.molecular_weights.nitrogen"
            )
        if key == "phosphorus":
            return UnitConverter._get_physical_constant(
                "relic.physical.constants.molecular_weights.phosphorus"
            )
        raise ValueError(f"Unsupported elemental molar mass '{element}'")

    def _element_mass_amount_factor(self, source_units: str, target_units: str) -> Optional[float]:
        """Return f with target = f * source for an element mass <-> element amount pair.

        References: IUPAC Commission on Isotopic Abundances and Atomic Weights
            (standard atomic weights, the physical-constant SSOT values
            relic.physical.constants.molecular_weights.<element>); UDUNITS-2 for the
            prefix and remaining-unit scales; BIPM SI Brochure, 9th edition, 2.3.
        Mathematical expression: n_E = m_E / M_E (mass to amount) and m_E = n_E M_E
            (amount to mass); f = U(1; m / [M_E] -> n) / M_E or f = U(1; n * [M_E] -> m) * M_E,
            where m and n are the labels without the element token.
        Parameter meanings: m_E element mass (areal or flux form); n_E element amount;
            M_E molar mass of element E from the SSOT; [M_E] its SSOT unit (g mol-1);
            U the UDUNITS scale factor of the element-free labels.
        Units: any mass prefix (ug, mg, g, kg, Mg, ...) and amount prefix (umol, mmol,
            mol, ...) with the same remaining unit dimension, e.g. kg C m-2 <->
            mol C m-2 or g N m-2 s-1 <-> umol N m-2 s-1.
        Domain/constraints: both labels carry exactly one element token (C, N or P,
            from _ELEMENT_BUDGET_UNIT_SPECS), the same element, and differ exactly by
            the dimension of molar mass; mass <-> mass or amount <-> amount pairs,
            different elements and unannotated labels return None.
        Sign convention: M_E > 0, so the sign of the value (and of a flux) is kept.
        """
        from .dimensionality import element_annotations_from_unit, unit_without_element_annotations

        symbols = {
            element_annotations_from_unit(spec[0])[0]: element
            for element, spec in self._ELEMENT_BUDGET_UNIT_SPECS.items()
            if element_annotations_from_unit(spec[0])
        }

        def split(units: str):
            # The element label and the label-free unit come from the one
            # dimensionality rule (a C, N or P directly after a mass or amount unit).
            labels = element_annotations_from_unit(units)
            if not labels or len(labels) != 1 or labels[0] not in symbols:
                return None, None
            return symbols[labels[0]], unit_without_element_annotations(units)

        source_element, source_plain = split(source_units)
        target_element, target_plain = split(target_units)
        if source_element is None or source_element != target_element:
            return None
        from ...parameters import get_global_manager

        constant_key = f"relic.physical.constants.molecular_weights.{source_element}"
        molar_mass = self._element_atomic_molar_mass(source_element)
        molar_mass_unit = Unit(get_global_manager().get_parameter(constant_key).units)
        try:
            source = Unit(source_plain)
            target = Unit(target_plain)
        except (TypeError, ValueError):
            return None
        for combined, scale in ((source / molar_mass_unit, 1.0 / molar_mass), (source * molar_mass_unit, molar_mass)):
            if not combined.is_convertible(target):
                continue
            if float(combined.convert(0.0, target)) != 0.0:
                return None
            factor = float(combined.convert(1.0, target)) * scale
            if not math.isfinite(factor) or factor <= 0.0:
                return None
            return factor
        return None

    def water_flux_mol_to_kg(self, data: Tensor, context: ConversionContext) -> Tensor:
        """
        Convert water flux from mol H2O m-2 s-1 to kg H2O m-2 s-1.
        """
        mw_water = self._resolve_molecular_weight("water.molecular_weight")
        return data * (mw_water * 1e-3)

    def water_flux_kg_to_mol(self, data: Tensor, context: ConversionContext) -> Tensor:
        """
        Convert water flux from kg H2O m-2 s-1 to mol H2O m-2 s-1.
        """
        mw_water = self._resolve_molecular_weight("water.molecular_weight")
        return data / (mw_water * 1e-3)

    def mol_to_umol_flux(self, data: Tensor, context: ConversionContext) -> Tensor:
        """
        Convert molar flux from mol m-2 s-1 to umol m-2 s-1.

        References:
            1 mol = 1e6 umol (SI Brochure, 9th Edition)
        """
        return data * 1e6

    def umol_to_mol_flux(self, data: Tensor, context: ConversionContext) -> Tensor:
        """
        Convert molar flux from umol m-2 s-1 to mol m-2 s-1.

        References:
            1 umol = 1e-6 mol (SI Brochure, 9th Edition)
        """
        return data / 1e6

    def get_conversion_context(
        self,
        state: Dict[str, Tensor],
        *,
        layer_thickness_variable: Optional[str] = None,
        layer_thickness: Optional[Union[float, Tensor]] = None,
        temperature: Optional[Union[float, Tensor]] = None,
        pressure: Optional[Union[float, Tensor]] = None,
    ) -> ConversionContext:
        """
        Build a conversion context from declared state fields and explicit arguments.

        State fields are stored as given (tensor context, no reduction): the
        conversions broadcast them elementwise against the converted data and
        stay differentiable with respect to them. A field that is absent stays
        None, and the conversion that needs it raises.

        The layer thickness ``soil_depth`` is never looked up by a fixed state
        name: it is the thickness of the layer that the converted variable
        describes, so it comes from that variable's declaration.

        Parameters:
            state : Dict[str, Tensor]
                Model state dictionary
            layer_thickness_variable : Optional[str]
                Name of the state variable that the declaration of the variable
                being converted names as its layer thickness [m] (for example
                ``"root_zone_depth"`` for a declared root-zone mean); read
                pointwise from ``state``
            layer_thickness : Optional[float | Tensor]
                Explicit layer thickness [m] supplied by the caller; kept as given
            temperature : Optional[float | Tensor]
                Temperature [K] the conversion needs when it is not the air
                temperature (for example a leaf or soil temperature); without it
                the state field ``air_temperature`` is read
            pressure : Optional[float | Tensor]
                Pressure [Pa] the conversion needs when it is not the surface
                air pressure; without it the state field ``atmospheric_pressure``
                is read

        Returns:
            ConversionContext with extracted values
        """
        context = ConversionContext()

        # References: Hillel (1998), Environmental Soil Physics; CF-1.8.
        # Mathematical expression: m_w = theta * L * rho_w (elementwise), consumed by
        #   volumetric_to_mass_soil_moisture / mass_to_volumetric_soil_moisture.
        # Parameter meanings: L is the thickness of the layer that theta describes.
        # Units: L [m].
        # Domain/constraints: L comes only from the converted variable's declaration
        #   (layer_thickness_variable, read pointwise) or an explicit argument; exactly
        #   one source may be given; no fixed state name (soil_depth, root_zone_depth)
        #   is substituted and no value is averaged. Absent L stays None.
        # Sign convention: not applicable (L is a positive thickness).
        if layer_thickness_variable is not None and layer_thickness is not None:
            raise ValueError(
                "Give either layer_thickness_variable or layer_thickness, not both"
            )
        if layer_thickness_variable is not None:
            if not isinstance(layer_thickness_variable, str) or not layer_thickness_variable.strip():
                raise ValueError(
                    "layer_thickness_variable must be a non-empty state variable name, "
                    f"got {layer_thickness_variable!r}"
                )
            if layer_thickness_variable not in state:
                raise ValueError(
                    f"declared layer thickness variable {layer_thickness_variable!r} "
                    "is not in the state"
                )
            context.soil_depth = state[layer_thickness_variable]
        elif layer_thickness is not None:
            if isinstance(layer_thickness, Tensor):
                context.soil_depth = layer_thickness
            elif isinstance(layer_thickness, (int, float)) and not isinstance(layer_thickness, bool):
                context.soil_depth = float(layer_thickness)
            else:
                raise ValueError(
                    "layer_thickness must be a tensor or a real number, "
                    f"got {type(layer_thickness).__name__}"
                )

        # Thermodynamic fields, kept elementwise and differentiable. Each quantity is
        # read from its canonical contract name only (air_temperature,
        # atmospheric_pressure); aliases such as air_pressure are resolved by the
        # ContractRegistry before a state reaches the converter, and no other name
        # is tried. A conversion that needs a different temperature or pressure
        # (leaf, soil, ...) receives it explicitly from the caller.
        if temperature is not None:
            context.temperature = temperature
        elif 'air_temperature' in state:
            context.temperature = state['air_temperature']

        if pressure is not None:
            context.pressure = pressure
        elif 'atmospheric_pressure' in state:
            context.pressure = state['atmospheric_pressure']

        # Water density may be state-derived, for example from temperature or salinity;
        # it is kept per cell. Standard water density remains available only through
        # explicit SSOT opt-in.
        if 'water_density' in state:
            context.water_density = state['water_density']

        return self._apply_context_defaults(context)


# Global converter instance for easy access
unit_converter = UnitConverter()
