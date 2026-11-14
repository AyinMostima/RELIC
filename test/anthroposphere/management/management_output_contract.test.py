from relic.scientific.anthroposphere.management.fire_management import FireManagement
from relic.scientific.anthroposphere.management.harvest import HarvestManagement


def test_fire_management_metadata_declares_runtime_outputs():
    metadata_outputs = set(FireManagement(config={"seed": 0}).metadata.output_variables)
    required_outputs = {
        "burned_fraction",
        "co2_emission",
        "co_emission",
        "fire_ch4_emission",
        "n2o_emission",
        "ash_deposition",
        "days_since_fire",
        "fire_fuel_carbon_input",
        "carbon_loss",
        "combustion_flux",
        "diagnostics.fire.events_today",
        "diagnostics.fire.event_support_seconds",
        "diagnostics.fire.carbon_availability_residual",
        "diagnostics.fire_management.canonical_iphs_runtime",
    }
    assert required_outputs.issubset(metadata_outputs)
    assert "cumulative_burned_area" not in metadata_outputs
    assert "cumulative_emissions" not in metadata_outputs


def test_harvest_management_metadata_declares_runtime_outputs():
    metadata_outputs = set(HarvestManagement().metadata.output_variables)
    required_outputs = {
        "harvest_flux",
        "harvest_biomass_carbon_input",
        "harvest_carbon_export_flux",
        "residue_c_flux",
        "harvest_biomass_nitrogen_input",
        "harvest_nitrogen_export_flux",
        "residue_n_flux",
        "delta_residue_cover",
        "diagnostics.harvest.events_today",
        "diagnostics.harvest.biomass_partition_residual",
        "diagnostics.harvest.carbon_ledger_residual",
        "diagnostics.harvest.nitrogen_ledger_residual",
        "diagnostics.harvest_management.free_energy",
        "diagnostics.harvest_management.chemical_potential_norm",
        "diagnostics.harvest_management.canonical_iphs_runtime",
    }
    assert metadata_outputs == required_outputs
