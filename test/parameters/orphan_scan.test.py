from __future__ import annotations

import ast
import re
import warnings
from pathlib import Path
from typing import Dict, Set

import yaml


ROOT = Path(__file__).resolve().parents[2]
PEDO_PARAM_ROOT = ROOT / "relic" / "parameters" / "database" / "relic" / "pedosphere"
SCAN_DIRS = [ROOT / "relic", ROOT / "test", ROOT / "docs"]
DYNAMIC_REFERENCED_KEYS = {
    # Used via dynamic namespace alias self._state_defaults_namespace in
    # relic/scientific/pedosphere/decomposition/microbial_enzyme.py.
    "pedosphere.state_defaults.available_som_c",
}


def _iter_pedosphere_keys() -> Dict[str, str]:
    entries: Dict[str, str] = {}
    for yaml_path in PEDO_PARAM_ROOT.rglob("*.yaml"):
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        namespace = str(payload.get("namespace", "")).strip()
        params = payload.get("parameters", {})
        if not isinstance(params, dict):
            continue
        for raw_key in params.keys():
            key = str(raw_key)
            full_key = key if (not namespace or key.startswith(namespace + ".")) else f"{namespace}.{key}"
            entries[full_key] = str(yaml_path.relative_to(ROOT)).replace("/", "\\")
    return entries


def _extract_ast_parameter_usage(py_file: Path) -> Set[str]:
    usage: Set[str] = set()
    try:
        source = py_file.read_text(encoding="utf-8-sig")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            tree = ast.parse(source)
    except Exception:
        return usage

    call_names = {
        "get_parameter_value",
        "require_param",
        "require_scalar",
        "_param",
        "_scalar_param",
        "get",
    }

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = None
        if isinstance(node.func, ast.Attribute):
            call_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            call_name = node.func.id
        if call_name not in call_names:
            continue

        key = None
        namespace = None
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            key = node.args[0].value

        if call_name in {"_param", "_scalar_param"} and len(node.args) >= 2:
            arg0 = node.args[0]
            arg1 = node.args[1]
            if (
                isinstance(arg0, ast.Constant)
                and isinstance(arg0.value, str)
                and isinstance(arg1, ast.Constant)
                and isinstance(arg1.value, str)
            ):
                namespace = arg0.value
                key = arg1.value

        for kw in node.keywords:
            if kw.arg in {"name", "key"} and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                key = kw.value.value
            if kw.arg in {"namespace", "ns"} and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                namespace = kw.value.value

        if not key:
            continue
        if key.startswith("pedosphere."):
            usage.add(key)
            continue
        if namespace and namespace.startswith("pedosphere."):
            usage.add(key if key.startswith(namespace + ".") else f"{namespace}.{key}")
            continue
        if call_name == "get" and key.startswith("pedosphere."):
            usage.add(key)

    return usage


def _extract_textual_parameter_usage(text: str) -> Set[str]:
    return set(re.findall(r"\bpedosphere\.[a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)+\b", text))


def test_pedosphere_parameter_database_has_no_orphan_keys():
    pedosphere_keys = _iter_pedosphere_keys()
    used_keys: Set[str] = set()

    for scan_dir in SCAN_DIRS:
        for path in scan_dir.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix in {".py", ".md", ".yaml", ".yml", ".json", ".toml", ".ini", ".txt"}:
                try:
                    text = path.read_text(encoding="utf-8")
                except Exception:
                    continue
                used_keys.update(_extract_textual_parameter_usage(text))
            if suffix == ".py":
                used_keys.update(_extract_ast_parameter_usage(path))

    used_keys.update(DYNAMIC_REFERENCED_KEYS)

    orphan_keys = sorted([key for key in pedosphere_keys if key not in used_keys])
    assert not orphan_keys, (
        "Found orphan pedosphere parameter keys:\n"
        + "\n".join(f"{key} ({pedosphere_keys[key]})" for key in orphan_keys)
    )


# Parameters removed because no relic code reads them, statically or through a
# dynamically built key: the relative-humidity default of the atmosphere state
# defaults and the land-use parameter modifiers whose only reader,
# LandUseTransitions._calculate_parameter_modifiers, was removed.
REMOVED_ORPHAN_KEYS = (
    # Unread management objective targets and emission policy inputs (item 124).
    "anthroposphere.management.coordinator.objectives.water_use_efficiency.target_value",
    "anthroposphere.management.coordinator.objectives.nutrient_use_efficiency.target_value",
    "anthroposphere.management.coordinator.objectives.soil_carbon_sequestration.target_value",
    "anthroposphere.management.coordinator.objectives.economic_return.target_value",
    "anthroposphere.management.coordinator.objectives.biodiversity_index.target_value",
    "anthroposphere.emissions.policy_inputs.emission_tax",
    "anthroposphere.emissions.policy_inputs.subsidy_rate",
    "anthroposphere.emissions.policy_inputs.conservation_policy",
    # Emission temperature response entries without a reader; the q10 key they
    # belonged to was already removed (MA-21 follow-up F1 add-on).
    "anthroposphere.emissions.temperature_response.q10_interval",
    "anthroposphere.emissions.temperature_response.reference_temperature",
    "anthroposphere.emissions.temperature_response.min_factor",
    "anthroposphere.emissions.temperature_response.max_factor",
    # Emission keys without any reader (static full keys, namespace-relative keys and the
    # keys built by SectorAllocation.required_parameters and SpeciesMapper were checked):
    # the 25 C baseline normalized the removed temperature response above (its read twin
    # is pedosphere.decomposition.param_25); SectorAllocation takes one fixed spatial
    # proxy per sector and reads only sector_allocation.proxy_floor and
    # .cropland_class_index, so the proxy mixing weights (the nested set and its flat
    # gdp_weight duplicate), the proxy decay scales and the allocation-method switch
    # entered no equation; the anthroposphere.inventory copies of the weights were
    # already removed as dead inventory keys.
    "anthroposphere.emissions.param_25_baseline",
    "anthroposphere.emissions.param_25.sector_allocation.decay_scales.gdp",
    "anthroposphere.emissions.param_25.sector_allocation.decay_scales.population",
    "anthroposphere.emissions.param_25.sector_allocation.gdp_weight",
    "anthroposphere.emissions.param_25.sector_allocation.landuse_weight",
    "anthroposphere.emissions.param_25.sector_allocation.population_weight",
    "anthroposphere.emissions.gdp_weight",
    "anthroposphere.emissions.allocation_method",
    # Emission policy-response sensitivities and limits without any reader (MA-21 C5).
    "anthroposphere.emissions.policy_response.carbon_tax_sensitivity",
    "anthroposphere.emissions.policy_response.subsidy_sensitivity",
    "anthroposphere.emissions.policy_response.conservation_sensitivity",
    "anthroposphere.emissions.policy_response.min_factor",
    "anthroposphere.emissions.policy_response.max_factor",
    # Runtime counters and a flux value migrated into the parameter database by a
    # hardcoded-value fixer (MA-21 C9 round 11): the step count is RELICSimulator.step_count,
    # the solver counters are no parameters, and the soil heat flux is a state input of
    # the energy-balance check; no reader requested any of them.
    "general_parameters.step_count",
    "general_parameters.function_evaluations",
    "general_parameters.solve_count",
    "general_parameters.soil_heat_flux",
    # The irrigation quota is scenario state irrigation_remaining_quota (item 119).
    "anthroposphere.management.annual_water_quota",
    "anthroposphere.management.irrigation.annual_water_quota",
    # Generic irrigation efficiency: every valid irrigation_type (1 to 4) selects one
    # of the four system efficiencies, so no cell ever used the generic value; the
    # system efficiencies are the learnable parameters of IrrigationManagement.
    "anthroposphere.management.irrigation.irrigation_efficiency",
    # FireManagement holds no tensor state (item 118): days_since_fire is the owned
    # state input and the suppression buffer had no reader.
    "anthroposphere.management.fire.days_since_fire_initial",
    "anthroposphere.management.fire.suppression_resources_default",
    "atmosphere.state_defaults.humidity",
    "atmosphere.state_defaults.gpp",
    # Former runtime substitutes of missing inputs (item 83).
    "atmosphere.state_defaults.temperature",
    "atmosphere.state_defaults.soil_temperature",
    # Runtime default of the retired ozone_concentration input (item 86).
    "biosphere.stress.ozone.default_ozone_concentration",
    # State defaults without a reader (item 87): environmental_forcing derives the
    # deficit from T and RH, and the respiration Functors require the maintenance
    # flux as state (its umol CO2 label also contradicted the kg C contract).
    "atmosphere.state_defaults.vapor_pressure_deficit",
    "atmosphere.state_defaults.maintenance_respiration_total",
    # Runtime substitutes of missing inputs (item 88): the advection-diffusion
    # winds and the tillage soil moisture and soil carbon stock.
    "atmosphere.state_defaults.wind_u_component",
    "atmosphere.state_defaults.wind_v_component",
    "atmosphere.state_defaults.wind_w_component",
    "anthroposphere.management.soil_moisture",
    "anthroposphere.management.soil_carbon",
    # The C3 Farquhar path has no triose phosphate utilization limit (item 90).
    "biosphere.photosynthesis.tpu25",
    # Runtime substitutes of missing inputs (item 93): fire litter and the ignition
    # threshold used as a fuel moisture state, the tillage elapsed-time state, and
    # the atmospheric advection-diffusion background source.
    "anthroposphere.management.litter_mass_default",
    "anthroposphere.management.fire.fuel_moisture_threshold",
    "anthroposphere.management.tillage.default_days_since_tillage",
    # Tillage rate core (item 96d): the substitute of a missing intensity control
    # and the coefficient of the unpublished rate-core porosity response.
    "anthroposphere.management.tillage.default_intensity",
    "anthroposphere.management.tillage.porosity_intensity_coefficient",
    # Harvest (item 96): the reference state of the removed property query and the
    # crop-stage substitute of a missing crop_maturity.
    "anthroposphere.management.aboveground_biomass_default",
    "anthroposphere.management.belowground_biomass_default",
    "anthroposphere.management.harvest.grain_moisture_content",
    "anthroposphere.management.irrigation.crop_stage_default",
    "atmosphere.transport.air_temperature_default",
    # Farquhar (item 97): the absent Vcmax scaling is the identity 1, not a parameter.
    "biosphere.photosynthesis.default_vcmax_scaling",
    # Keys read only by the removed fertilization integration example (item 98).
    "anthroposphere.management.fertilization.ph_factor_min",
    "anthroposphere.management.fertilization.ph_factor_max",
    "anthroposphere.management.fertilization.n2o_temp_factor_min",
    "anthroposphere.management.fertilization.n2o_temp_factor_max",
    "anthroposphere.management.fertilization.moisture_factor_min",
    "anthroposphere.management.fertilization.moisture_factor_max",
    "anthroposphere.management.fertilization.urea_hydrolysis_base_coefficient",
    "anthroposphere.management.fertilization.urea_hydrolysis_temperature_coefficient",
    "anthroposphere.management.fertilization.urea_hydrolysis_temperature_offset",
    "anthroposphere.management.fertilization.k_no3_to_no2",
    "anthroposphere.management.fertilization.k_no2_to_no",
    "anthroposphere.management.fertilization.k_no_to_n2o",
    "anthroposphere.management.fertilization.k_n2o_to_n2",
    "anthroposphere.management.fertilization.soc_baseline",
    "anthroposphere.management.fertilization.soil_moisture_default",
    "anthroposphere.management.fertilization.soil_temperature_default",
    "anthroposphere.management.fertilization.soil_ph_default",
    "anthroposphere.management.fertilization.soil_organic_carbon_default",
    "anthroposphere.management.fertilization.soil_nitrogen_default",
    "anthroposphere.management.fertilization.soil_phosphorus_default",
    "anthroposphere.management.fertilization.soil_no3_default",
    "anthroposphere.management.fertilization.cold_threshold",
    "anthroposphere.management.fertilization.heat_threshold",
    "anthroposphere.management.fertilization.cold_reduction_factor",
    "anthroposphere.management.fertilization.heat_reduction_factor",
    "anthroposphere.management.fertilization.efficiency_broadcast",
    "anthroposphere.management.fertilization.q10_base",
    "anthroposphere.management.fertilization.q10_n2o",
    "anthroposphere.management.fertilization.q10_denitrification",
    "anthroposphere.management.fertilization.reference_temperature",
    # Nominal coordinator stand-ins replaced by state inputs or removed (items 96, 99).
    "anthroposphere.management.et_demand_default",
    "anthroposphere.management.irrigation.et_demand_default",
    "anthroposphere.management.irrigation.field_capacity_default",
    "anthroposphere.management.irrigation.wilting_point_default",
    "anthroposphere.management.coordinator.priority_weights.environmental_conditions",
    "anthroposphere.management.harvest.default_expected_yield",
    # Tillage keys without a reader after the single-accessor and single-flux fix (item 104).
    "anthroposphere.management.tillage.carbon_loss_rate",
    "anthroposphere.management.tillage.carbon_loss_intensity_coefficient",
    "anthroposphere.management.tillage.mixing_efficiency",
    "anthroposphere.management.tillage.conventional_soil_mixing",
    "anthroposphere.management.tillage.reduced_soil_mixing",
    "anthroposphere.management.tillage.no_till_soil_mixing",
    "anthroposphere.management.tillage.conventional_residue_incorporation",
    "anthroposphere.management.tillage.reduced_residue_incorporation",
    "anthroposphere.management.tillage.no_till_residue_incorporation",
    "anthroposphere.management.tillage.base_efficiency",
    "anthroposphere.management.tillage.tillage_depth",
    "anthroposphere.management.tillage.no_till_depth",
    "anthroposphere.management.tillage.residue_incorporation",
    # Harvest keys without any reader (item 108).
    "anthroposphere.management.harvest.index_wheat",
    "anthroposphere.management.harvest.index_maize",
    "anthroposphere.management.harvest.index_rice",
    "anthroposphere.management.harvest.index_soybean",
    "anthroposphere.management.harvest.index_default",
    "anthroposphere.management.harvest.residue_retention_fraction",
    "anthroposphere.management.harvest.residue_burning_fraction",
    "anthroposphere.management.harvest.straw_moisture_content",
    "anthroposphere.management.harvest.storage_loss_fraction",
    "anthroposphere.management.harvest.root_turnover_rate",
    "anthroposphere.management.harvest.harvest_index",
    # Harvest keys read only by the removed event-less partition or declared without a
    # reader (item 105).
    "anthroposphere.management.harvest.harvest_type_factors",
    "anthroposphere.management.harvest.grain_moisture_threshold",
    "anthroposphere.management.harvest.base_efficiency",
    "anthroposphere.management.harvest.residue_removal_fraction",
    "anthroposphere.management.harvest.default_c_content",
    "anthroposphere.management.harvest.default_cn_ratio",
    "anthroposphere.management.harvest.default_root_shoot_ratio",
    "anthroposphere.management.harvest.default_residue_quality",
    "anthroposphere.management.harvest.cereal_c3_harvest_index",
    "anthroposphere.management.harvest.cereal_c3_root_shoot_ratio",
    "anthroposphere.management.harvest.cereal_c3_residue_quality",
    "anthroposphere.management.harvest.cereal_c4_harvest_index",
    "anthroposphere.management.harvest.cereal_c4_root_shoot_ratio",
    "anthroposphere.management.harvest.cereal_c4_residue_quality",
    "anthroposphere.management.harvest.legume_harvest_index",
    "anthroposphere.management.harvest.legume_root_shoot_ratio",
    "anthroposphere.management.harvest.legume_residue_quality",
    "anthroposphere.management.harvest.root_tuber_harvest_index",
    "anthroposphere.management.harvest.root_tuber_root_shoot_ratio",
    "anthroposphere.management.harvest.root_tuber_residue_quality",
    "anthroposphere.management.harvest.forage_harvest_index",
    "anthroposphere.management.harvest.forage_root_shoot_ratio",
    "anthroposphere.management.harvest.forage_residue_quality",
    "anthroposphere.management.harvest.forest_harvest_index",
    "anthroposphere.management.harvest.forest_root_shoot_ratio",
    "anthroposphere.management.harvest.forest_residue_quality",
    "atmosphere.transport.source_term_default",
    "anthroposphere.land_use.parameter_modifiers.forest_lai_modifier",
    "anthroposphere.land_use.parameter_modifiers.crop_lai_modifier",
    "anthroposphere.land_use.parameter_modifiers.urban_lai_modifier",
    "anthroposphere.land_use.parameter_modifiers.forest_albedo_modifier",
    "anthroposphere.land_use.parameter_modifiers.crop_albedo_modifier",
    "anthroposphere.land_use.parameter_modifiers.urban_albedo_modifier",
    "anthroposphere.land_use.parameter_modifiers.forest_root_depth_modifier",
    "anthroposphere.land_use.parameter_modifiers.crop_root_depth_modifier",
    "anthroposphere.land_use.parameter_modifiers.urban_root_depth_modifier",
    # Phenology environment defaults (D33): runtime substitutes of missing forcing
    # read only by the removed EnvironmentNormalizer, its copy of the Celsius offset
    # (the SSOT is relic.physical.constants.temperature.celsius_to_kelvin) and the
    # JIT and UQ switches that PhenologyBase read but never implemented.
    "biosphere.phenology.defaults.air_temperature",
    "biosphere.phenology.defaults.soil_moisture",
    "biosphere.phenology.defaults.daylength",
    "biosphere.phenology.defaults.solar_radiation",
    "biosphere.phenology.defaults.temperature_min",
    "biosphere.phenology.defaults.temperature_max",
    "biosphere.phenology.defaults.vapor_pressure_deficit",
    "biosphere.phenology.defaults.relative_humidity",
    "biosphere.phenology.defaults.soil_water_potential",
    "biosphere.phenology.defaults.photosynthetic_photon_flux",
    "biosphere.phenology.defaults.day_of_year",
    "biosphere.phenology.defaults.simulation_time",
    "biosphere.phenology.defaults.time_step",
    "biosphere.phenology.defaults.leaf_area_index",
    "biosphere.phenology.computational.celsius_to_kelvin",
    "biosphere.phenology.computational.use_jit",
    "biosphere.phenology.computational.enable_uq",
    # Phenology computational keys that no reader uses (MA-21 12.4): the former
    # substitute defaults of missing inputs and unused numerical helpers.
    "biosphere.phenology.computational.default_day_of_year",
    "biosphere.phenology.computational.default_temperature",
    "biosphere.phenology.computational.default_leaf_carbon",
    "biosphere.phenology.computational.default_froot_carbon",
    "biosphere.phenology.computational.default_leaf_nitrogen",
    "biosphere.phenology.computational.default_froot_nitrogen",
    "biosphere.phenology.computational.inactive_growth_value",
    "biosphere.phenology.computational.fraction_param",
    "biosphere.phenology.computational.kelvin_detection_threshold",
    # Read only by the removed phenology constants module (D33), whose species and
    # biome tables had no production caller.
    "biosphere.phenology.memory.temperature_memory_days",
    "biosphere.phenology.stress.cold_stress_base",
    "biosphere.phenology.stress.photoperiod_sensitivity_default",
    "biosphere.phenology.turnover.n_retrans_efficiency_needles",
    # Biosphere state defaults (D33): their only reader was the removed
    # EnvironmentNormalizer, which collected the namespace dynamically.
    "biosphere.defaults.water_stress",
    "biosphere.defaults.temperature_stress",
    "biosphere.defaults.ozone_stress",
    "biosphere.defaults.co2_intercellular_historical",
    "biosphere.defaults.co2_intercellular_modern",
    "biosphere.defaults.temperature",
    "biosphere.defaults.vapor_pressure_deficit",
    "biosphere.defaults.leaf_area_index",
    "biosphere.defaults.vcmax_scaling",
    # Isotope defaults (D34, D37): the ci/ca ratio, the substrate quality and source
    # and the BVOC compound are inputs of the call, not substituted defaults; the
    # carbon-13 VPD multiplier and its scale are removed; the pathway and the oxygen-18
    # kinetic reductions are read under their parameter names.
    "isotopes.fractionation.carbon.ci_ca_ratio_default",
    "isotopes.fractionation.carbon.vpd_sensitivity",
    "isotopes.fractionation.carbon.vpd_scale",
    "isotopes.fractionation.carbon.decomposition.substrate_quality_default",
    "isotopes.fractionation.carbon.bvoc.substrate_source_default",
    "isotopes.fractionation.carbon.bvoc.default_type",
    "isotopes.fractionation.carbon.photosynthesis_type_default",
    "isotopes.fractionation.oxygen.evaporation.soil_kinetic_reduction_default",
    "isotopes.fractionation.oxygen.evaporation.canopy_kinetic_reduction_default",
    # Carbon-13 photosynthesis (MA-21 12.3): the Q10 and PAR multipliers of the
    # discrimination and the C4 temperature multiplier have no literature basis
    # (Farquhar et al. 1982, 1989; Farquhar 1983); ci/ca is validated on [0, 1]
    # instead of being clipped to a numerical-stability range.
    "isotopes.fractionation.carbon.ci_ca_ratio_min",
    "isotopes.fractionation.carbon.ci_ca_ratio_max",
    "isotopes.fractionation.carbon.temperature_sensitivity",
    "isotopes.fractionation.carbon.temperature_sensitivity_threshold",
    "isotopes.fractionation.carbon.c3_temp_correction_scale",
    "isotopes.fractionation.carbon.light_sensitivity",
    "isotopes.fractionation.carbon.light_par_saturation",
    "isotopes.fractionation.carbon.c4_optimum_temperature",
    "isotopes.fractionation.carbon.c4_temp_correction_below",
    "isotopes.fractionation.carbon.c4_temp_correction_above",
    # Phenology parameters that no phenology model reads (D33): each model reads
    # only the common fields and its own; vegetation type, latitude, wood
    # turnover, leaf C:N, the stress thresholds and the limitation switches
    # entered no equation (the SLA stays: StructuralGrowth reads it).
    "biosphere.phenology.common.vegetation_type",
    "biosphere.phenology.common.latitude",
    "biosphere.phenology.turnover.wood_turnover_rate",
    "biosphere.phenology.morphology.leaf_c_n_ratio",
    "biosphere.phenology.stress.temp_stress_threshold",
    "biosphere.phenology.stress.photoperiod_sensitivity",
    "biosphere.phenology.stress.frost_tolerance",
    "biosphere.phenology.config.enable_nitrogen_limitation",
    "biosphere.phenology.config.enable_water_limitation",
    "biosphere.phenology.config.enable_temperature_limitation",
    "biosphere.phenology.presets.deciduous_tree_gsi.vegetation_type",
    "biosphere.phenology.presets.deciduous_tree_gsi.sla",
    "biosphere.phenology.presets.deciduous_tree_gsi.wood_turnover_rate",
    "biosphere.phenology.presets.evergreen_tree_static.vegetation_type",
    "biosphere.phenology.presets.evergreen_tree_static.sla",
    "biosphere.phenology.presets.evergreen_tree_static.wood_turnover_rate",
    "biosphere.phenology.presets.grassland_gdd.vegetation_type",
    "biosphere.phenology.presets.grassland_gdd.sla",
    "biosphere.phenology.presets.grassland_gdd.wood_turnover_rate",
    "biosphere.phenology.presets.winter_wheat_crop.vegetation_type",
    "biosphere.phenology.presets.winter_wheat_crop.sla",
    "biosphere.phenology.presets.winter_wheat_crop.wood_turnover_rate",
    "biosphere.phenology.presets.maize_crop.vegetation_type",
    "biosphere.phenology.presets.maize_crop.sla",
    "biosphere.phenology.presets.maize_crop.wood_turnover_rate",
    # coupling.variables.<name>.bounds is read only by register_coupling_variables, for the
    # outputs it registers (boundary-layer conductance and the soil-vegetation outputs); the
    # bounds of other names had no reader (MA-21 items 186 and 195).
    "coupling.variables.stomatal_conductance.bounds",
    "coupling.variables.canopy_conductance.bounds",
    "coupling.variables.evapotranspiration_flux.bounds",
    "coupling.variables.sensible_heat_flux.bounds",
    "coupling.variables.latent_heat_flux.bounds",
    "coupling.variables.heterotrophic_respiration.bounds",
    "coupling.variables.npp.bounds",
    "coupling.variables.transpiration_rate.bounds",
    "coupling.variables.canopy_temperature.bounds",
    "coupling.variables.intercellular_co2.bounds",
    "coupling.variables.water_use_efficiency.bounds",
    "coupling.variables.gpp_water_stress_factor.bounds",
    "coupling.variables.gpp_vpd_stress_factor.bounds",
    "coupling.variables.autotrophic_respiration_modifier.bounds",
    "coupling.variables.heterotrophic_respiration_modifier.bounds",
    "coupling.variables.decomposition_rate_modifier.bounds",
    "coupling.variables.overall_water_stress.bounds",
    "coupling.variables.soil_respiration_moisture_factor.bounds",
    "coupling.variables.soil_respiration_temperature_factor.bounds",
    "coupling.variables.autotrophic_respiration.bounds",
    "coupling.variables.growth_rate_leaf.bounds",
    "coupling.variables.growth_rate_stem.bounds",
    "coupling.variables.growth_rate_root.bounds",
    # Second conservation tolerances of the nitrification and denitrification YAML
    # (MA-21 item 54.2): no code read them; the tolerances of the conservation checks
    # come only from validation\conservation.yaml (validation.conservation.*).
    "pedosphere.denitrification.mass_conservation_tolerance",
    "pedosphere.nitrification.mass_conservation_tolerance",
    # The volumetric-moisture oxygen threshold and its inverse slope, superseded by the
    # water-filled pore space optimum pedosphere.physical.wfps_nitrification_optimum
    # (MA-21 item 53.1).
    "pedosphere.nitrification.moisture_threshold",
    "pedosphere.nitrification.oxygen_limitation_factor",
    # Nitrification and denitrification entries without any reader in relic, tests,
    # examples, tutorials, documentation generators or certificates (MA-21 follow-up F1);
    # vmax_nh4 was a duplicate of vmax_nitrification.
    "pedosphere.denitrification.carbon_limitation_factor",
    "pedosphere.denitrification.drought_stress_factor",
    "pedosphere.denitrification.oxygen_inhibition_threshold",
    "pedosphere.denitrification.oxygen_inhibition_factor",
    "pedosphere.denitrification.nh4_inhibition_threshold",
    "pedosphere.denitrification.nh4_inhibition_factor",
    "pedosphere.denitrification.sulfate_competition_factor",
    "pedosphere.denitrification.min_carbon_pool",
    "pedosphere.denitrification.rate_clipping_threshold",
    "pedosphere.denitrification.temperature_minimum",
    "pedosphere.denitrification.temperature_maximum",
    "pedosphere.denitrification.extreme_temperature_factor",
    "pedosphere.nitrification.vmax_nh4",
    "pedosphere.nitrification.n2o_moisture_effect",
    "pedosphere.nitrification.n2o_ph_effect",
    "pedosphere.nitrification.drought_stress_threshold",
    "pedosphere.nitrification.waterlog_stress_threshold",
    "pedosphere.nitrification.substrate_limitation_factor",
    "pedosphere.nitrification.minimum_substrate_factor",
    "pedosphere.nitrification.nh3_inhibition_threshold",
    "pedosphere.nitrification.nh3_inhibition_factor",
    "pedosphere.nitrification.salinity_inhibition_threshold",
    "pedosphere.nitrification.salinity_inhibition_factor",
    "pedosphere.nitrification.min_nh4_pool",
    "pedosphere.nitrification.rate_clipping_threshold",
    # The volumetric-moisture anaerobic threshold of denitrification, superseded by the
    # water-filled pore space threshold pedosphere.physical.wfps_denitrification_threshold
    # (MA-21 follow-up F2).
    "pedosphere.denitrification.anaerobic_threshold",
    # Denitrification drought threshold: never read (the WFPS anaerobic factor already
    # reduces the rate to its floor in dry soil); the coupling threshold of the same leaf
    # name, coupling.scales.water_carbon.drought_threshold, is a different parameter.
    "pedosphere.denitrification.drought_threshold",
    # Atmospheric chemistry photolysis and OH closures (MA-21 photolysis/OH round): the
    # empirical Q10 of J_NO2 (the NO2 quantum yield is 1.00 at 300-398 nm at 248 and 298 K;
    # replaced by atmosphere.chemistry.photolysis_temperature_mode), its reference temperature
    # (now atmosphere.chemistry.photolysis.reference_temperature), the clear-sky shortwave of
    # the removed zenith estimate (cos_zenith is required), J(O3) = 0.1 J_NO2 (replaced by
    # the MCM J1/J4 ratio), the O(1D) branching ratio (replaced by the stoichiometric yield
    # atmosphere.chemistry.o1d_water_oh_yield) and the OH clamp and loss floor (removed).
    "atmosphere.chemistry.photolysis_temperature_q10",
    "atmosphere.chemistry.reference_temperature",
    "atmosphere.chemistry.clear_sky_shortwave_max",
    "atmosphere.chemistry.oh_o3_photolysis_factor",
    "atmosphere.chemistry.oh_o1d_branching_ratio",
    "atmosphere.chemistry.default_oh_concentration",
    "atmosphere.chemistry.oh_loss_rate_min",
    "atmosphere.chemistry.oh_lower_multiplier",
    "atmosphere.chemistry.oh_upper_multiplier",
    # Reduced J_NO2 proxy J0 (SW/SW_ref) cos(chi): it counted the solar geometry twice
    # (SW already contains cos(chi)); replaced by the Trebs et al. (2009) irradiance
    # relation atmosphere.chemistry.j_no2_irradiance_{linear,quadratic}_coefficient and
    # atmosphere.chemistry.uv_a_surface_albedo, which applies no zenith factor or clamp.
    "atmosphere.chemistry.j_no2_reference",
    "atmosphere.chemistry.reference_solar_radiation",
    "atmosphere.chemistry.radiation_factor_min",
    "atmosphere.chemistry.radiation_factor_max",
    "atmosphere.chemistry.cos_zenith_min",
    "atmosphere.chemistry.cos_zenith_max",
)


def test_removed_orphan_parameters_have_no_database_entry():
    from relic.parameters import get_global_manager

    manager = get_global_manager()
    present = [key for key in REMOVED_ORPHAN_KEYS if manager.get(key, required=False) is not None]
    assert present == []


def test_environment_defaults_namespace_is_retired() -> None:
    """No pedosphere Functor substitutes an environment default any more: the inputs
    are required, so pedosphere/environment_defaults.yaml and its namespace are gone."""
    root = Path(__file__).resolve().parents[2] / "relic" / "parameters" / "database" / "relic" / "pedosphere"
    assert not (root / "environment_defaults.yaml").exists()
    from relic.parameters import get_global_manager

    manager = get_global_manager()
    for key in (
        "pedosphere.environment_defaults.soil_temperature",
        "pedosphere.environment_defaults.soil_ph",
        "pedosphere.environment_defaults.labile_carbon",
        "pedosphere.environment_defaults.microbial_biomass",
        "pedosphere.nutrient_cycling.soil_depth_default",
        "pedosphere.physical.bulk_density_default",
        "pedosphere.stoichiometry.carbon_inputs_default",
    ):
        assert manager.get_parameter(key) is None, key


def test_fertilization_integration_example_module_is_removed():
    """The fertilization integration example (runtime soil-default filler, printing
    demonstration and validation-case table) had no production caller."""
    import importlib.util

    assert importlib.util.find_spec("relic.parameters.integration") is None


def test_readerless_soil_tortuosity_is_removed() -> None:
    """transport.diffusion.soil_tortuosity had no reader; it and its basename
    allowance are gone (the hydrosphere.transport copy went in round 35)."""
    from relic.parameters import get_global_manager

    manager = get_global_manager()
    assert manager.get_parameter("transport.diffusion.soil_tortuosity") is None
    allowed = manager.get("validation.numerical.audit.basename_duplicates_allow_basenames", required=True)
    assert "soil_tortuosity" not in list(allowed)
