"""
Audit gate for state single-owner policy and NO3 budget closure.
"""

from __future__ import annotations

from pathlib import Path

import torch

from relic.contracts.contract_loader import ContractRegistry
from relic.scientific.atmosphere.environmental_forcing import EnvironmentalForcingFunctor
from relic.scientific.pedosphere.nutrient_cycling.nitrogen import NitrogenCycling


def _tensor(value: float) -> torch.Tensor:
    return torch.tensor(float(value), dtype=torch.float32)


def test_solute_transport_scm_matches_local_ledger_and_full_state_semantics() -> None:
    """Solute ports must not masquerade as additive areal element pools."""

    contract_dir = Path("relic/contracts/variables/pedosphere")
    paths = {
        name: contract_dir / f"{name}.yaml"
        for name in (
            "nitrogen_leaching_flux",
            "carbon_leaching_flux",
            "no3_concentration",
            "doc_concentration",
            "surface_no3_concentration",
            "surface_doc_concentration",
        )
    }
    for path in paths.values():
        assert path.is_file(), f"Missing SCM contract: {path}"
        assert "REVIEW_NEEDED" not in path.read_text(encoding="utf-8")

    registry = ContractRegistry(strict=True)
    for path in paths.values():
        registry.load_contract_file(path)

    writer = "SoluteAdvectionDiffusion1D"
    for name, unit in (
        ("no3_concentration", "g N m-3"),
        ("doc_concentration", "g C m-3"),
    ):
        contract = registry.get_contract(name)
        assert contract is not None
        assert contract.kind == "state"
        assert contract.udunits == unit
        assert contract.owner == writer
        assert contract.aliases == []
        assert contract.incrementable is False
        assert contract.accumulates is None
        assert contract.writers == []
        assert registry.can_write(name, writer)
        assert contract.raw["provenance"]["runtime_role"].endswith(
            "full_state_concentration"
        )

    for name, unit in (
        ("surface_no3_concentration", "g N m-3"),
        ("surface_doc_concentration", "g C m-3"),
    ):
        contract = registry.get_contract(name)
        assert contract is not None
        assert contract.kind == "forcing"
        assert contract.udunits == unit
        assert contract.owner is None
        assert contract.writers == []
        assert contract.incrementable is False
        assert contract.accumulates is None
        assert contract.raw["provenance"]["runtime_writer"] is None

    carbon_flux = registry.get_contract("carbon_leaching_flux")
    assert carbon_flux is not None
    assert carbon_flux.kind == "flux"
    assert carbon_flux.udunits == "g C m-2 s-1"
    assert carbon_flux.owner == writer
    assert carbon_flux.incrementable is False
    assert carbon_flux.accumulates is None
    assert carbon_flux.tendency_writers == [writer]
    assert carbon_flux.flux_out == [writer]

    nitrogen_flux = registry.get_contract("nitrogen_leaching_flux")
    assert nitrogen_flux is not None
    assert nitrogen_flux.kind == "flux"
    assert nitrogen_flux.udunits == "g N m-2 s-1"
    assert nitrogen_flux.owner is None
    assert nitrogen_flux.incrementable is False
    assert nitrogen_flux.accumulates is None
    assert set(nitrogen_flux.tendency_writers) == {
        "nitrogen_leaching",
        writer,
    }
    assert set(nitrogen_flux.flux_out) == {
        "nitrogen_leaching",
        writer,
    }


def test_transport_open_boundary_ports_have_exact_scm_contracts() -> None:
    """Bind the external ghost state and signed face flux to one owner chain."""

    contract_dir = Path("relic/contracts/variables/atmosphere")
    paths = {
        "tracer_concentration_boundary_state": (
            contract_dir / "tracer_concentration_boundary_state.yaml"
        ),
        "tracer_concentration_advective_boundary_flux": (
            contract_dir / "tracer_concentration_advective_boundary_flux.yaml"
        ),
        "tracer_concentration_diffusive_boundary_flux": (
            contract_dir / "tracer_concentration_diffusive_boundary_flux.yaml"
        ),
    }
    for path in paths.values():
        assert path.is_file(), f"Missing SCM contract: {path}"
        assert "REVIEW_NEEDED" not in path.read_text(encoding="utf-8")

    registry = ContractRegistry(strict=True)
    for path in paths.values():
        registry.load_contract_file(path)

    boundary_state = registry.get_contract("tracer_concentration_boundary_state")
    assert boundary_state is not None
    assert boundary_state.kind == "forcing"
    assert boundary_state.udunits == "kg m-3"
    assert boundary_state.owner is None
    assert boundary_state.aliases == []
    assert boundary_state.incrementable is False
    assert boundary_state.writers == []
    assert boundary_state.raw["provenance"]["runtime_writer"] is None

    boundary_flux = registry.get_contract(
        "tracer_concentration_advective_boundary_flux"
    )
    assert boundary_flux is not None
    assert boundary_flux.kind == "flux"
    assert boundary_flux.udunits == "kg m-2 s-1"
    assert boundary_flux.owner == "linear_advection"
    assert boundary_flux.aliases == []
    assert boundary_flux.incrementable is False
    assert boundary_flux.tendency_writers == ["linear_advection"]
    assert boundary_flux.flux_in == []
    assert boundary_flux.flux_out == ["linear_advection"]
    assert boundary_flux.raw["provenance"]["face_order"] == [
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "z_min",
        "z_max",
    ]

    diffusive_flux = registry.get_contract(
        "tracer_concentration_diffusive_boundary_flux"
    )
    assert diffusive_flux is not None
    assert diffusive_flux.kind == "diagnostic"
    assert diffusive_flux.udunits == "kg m-2 s-1"
    assert diffusive_flux.owner == "fickian_diffusion"
    assert diffusive_flux.aliases == []
    assert diffusive_flux.incrementable is False
    assert diffusive_flux.tendency_writers == []
    assert diffusive_flux.flux_in == []
    assert diffusive_flux.flux_out == []
    assert diffusive_flux.raw["provenance"]["sign_convention"] == (
        "positive_outflow_negative_inflow"
    )


def test_transport_diagnostics_are_nonincrementable_observations() -> None:
    paths = {
        "diagnostics.diffusion_stability": Path(
            "relic/contracts/variables/transport/diagnostics_diffusion_stability.yaml"
        ),
        "diagnostics.conservation_error": Path(
            "relic/contracts/variables/atmosphere/diagnostics_conservation_error.yaml"
        ),
    }
    for path in paths.values():
        assert path.is_file()
        assert "REVIEW_NEEDED" not in path.read_text(encoding="utf-8")

    registry = ContractRegistry(strict=True)
    for path in paths.values():
        registry.load_contract_file(path)

    stability = registry.get_contract("diagnostics.diffusion_stability")
    assert stability is not None
    assert stability.kind == "diagnostic"
    assert stability.owner is None
    assert stability.incrementable is False
    assert set(stability.tendency_writers) == {
        "fickian_diffusion",
        "advection_diffusion",
    }

    conservation = registry.get_contract("diagnostics.conservation_error")
    assert conservation is not None
    assert conservation.kind == "diagnostic"
    assert conservation.owner is None
    assert conservation.incrementable is False
    assert set(conservation.tendency_writers) == {
        "linear_advection",
        "fickian_diffusion",
        "advection_diffusion",
        "atmospheric_advection_diffusion",
    }


def test_phenology_scm_contracts_reject_stale_aggregator_ownership() -> None:
    """Phenology state contracts must match compile-time owner selection."""

    contract_paths = {
        "leaf_area_index": Path(
            "relic/contracts/variables/biosphere/biosphere_leaf_area_index.yaml"
        ),
        "leaf_carbon": Path(
            "relic/contracts/variables/biosphere/biosphere_leaf_carbon.yaml"
        ),
        "biosphere.storage_carbon": Path(
            "relic/contracts/variables/biosphere/biosphere_storage_carbon.yaml"
        ),
        "biosphere.litter_carbon": Path(
            "relic/contracts/variables/biosphere/biosphere_litter_carbon.yaml"
        ),
    }
    for path in contract_paths.values():
        assert "REVIEW_NEEDED" not in path.read_text(encoding="utf-8")

    registry = ContractRegistry(strict=True)
    for path in contract_paths.values():
        registry.load_contract_file(path)

    selected_owner_candidates = {
        "static_phenology",
        "gdd_phenology",
        "gsi_phenology",
        "crop_phenology",
    }
    # CESM TLAI maps to leaf_area_index: the CESM adapter is its one external alternative writer.
    external_writers = {"leaf_area_index": {"CESMAdapter"}, "leaf_carbon": set()}
    for variable in ("leaf_area_index", "leaf_carbon"):
        contract = registry.get_contract(variable)
        assert contract is not None
        assert contract.owner is None
        assert contract.aliases == []
        assert set(contract.writers) == selected_owner_candidates | external_writers[variable]
        for writer in selected_owner_candidates:
            assert registry.can_write(variable, writer)
        assert not registry.can_write(variable, "biosphere_phenology_functor")
        assert not registry.can_write(variable, "unknown_phenology_writer")

    leaf_carbon = registry.get_contract("leaf_carbon")
    assert leaf_carbon is not None
    assert leaf_carbon.incrementable is True
    assert leaf_carbon.udunits == "kg C m-2"
    assert set(leaf_carbon.increment_writers) == selected_owner_candidates

    leaf_area_index = registry.get_contract("leaf_area_index")
    assert leaf_area_index is not None
    assert leaf_area_index.incrementable is False
    assert leaf_area_index.udunits == "m2 m-2"
    assert set(leaf_area_index.tendency_writers) == selected_owner_candidates | {"CESMAdapter"}

    for variable in ("biosphere.storage_carbon", "biosphere.litter_carbon"):
        contract = registry.get_contract(variable)
        assert contract is not None
        assert contract.owner is None
        assert contract.writers == []
        assert contract.incrementable is False
        assert contract.raw["provenance"]["runtime_role"] == (
            "optional_externally_supplied_state"
        )
        assert contract.raw["provenance"]["runtime_writer"] is None


def test_gdd_temporal_state_contracts_have_one_full_state_writer() -> None:
    """GDD history must be checkpointed full state owned by GDDPhenology."""

    temporal_units = {
        "phenology_gdd_accumulated": "degC day",
        "phenology_gdd_previous_day_of_year": "day",
    }
    registry = ContractRegistry(strict=True)
    for variable in temporal_units:
        path = Path(f"relic/contracts/variables/biosphere/{variable}.yaml")
        assert path.is_file(), f"Missing SCM contract: {path}"
        assert "REVIEW_NEEDED" not in path.read_text(encoding="utf-8")
        registry.load_contract_file(path)

    for variable, unit in temporal_units.items():
        contract = registry.get_contract(variable)
        assert contract is not None
        assert contract.kind == "state"
        assert contract.udunits == unit
        assert contract.owner == "gdd_phenology"
        assert contract.aliases == []
        assert contract.incrementable is False
        assert contract.increment_writers == []
        assert contract.tendency_writers == ["gdd_phenology"]
        assert contract.writers == ["gdd_phenology"]
        assert registry.can_write(variable, "gdd_phenology")
        assert not registry.can_increment(variable, "gdd_phenology")
        assert not registry.can_write(variable, "static_phenology")
        assert contract.raw["provenance"]["runtime_writer"] == "gdd_phenology"


def test_crop_temporal_state_contracts_have_one_full_state_writer() -> None:
    """Crop history must be checkpointed full state owned by CropPhenology."""

    temporal_units = {
        "phenology_crop_accumulated_phu": "degC day",
        "phenology_crop_is_sown": "dimensionless",
        "phenology_crop_is_harvested": "dimensionless",
        "phenology_crop_vernalization_days": "day",
    }
    registry = ContractRegistry(strict=True)
    for variable in temporal_units:
        path = Path(f"relic/contracts/variables/biosphere/{variable}.yaml")
        assert path.is_file(), f"Missing SCM contract: {path}"
        assert "REVIEW_NEEDED" not in path.read_text(encoding="utf-8")
        registry.load_contract_file(path)

    for variable, unit in temporal_units.items():
        contract = registry.get_contract(variable)
        assert contract is not None
        assert contract.kind == "state"
        assert contract.udunits == unit
        assert contract.owner == "crop_phenology"
        assert contract.aliases == []
        assert contract.incrementable is False
        assert contract.increment_writers == []
        assert contract.tendency_writers == ["crop_phenology"]
        assert contract.writers == ["crop_phenology"]
        assert registry.can_write(variable, "crop_phenology")
        assert not registry.can_increment(variable, "crop_phenology")
        assert not registry.can_write(variable, "gdd_phenology")
        provenance = contract.raw["provenance"]
        assert provenance["runtime_writer"] == "crop_phenology"
        assert provenance["canonical_key"] == variable

    for flag in ("phenology_crop_is_sown", "phenology_crop_is_harvested"):
        numerics = registry.get_contract(flag).raw["numerics"]
        assert numerics["lower_bound"] == 0.0
        assert numerics["upper_bound"] == 1.0


def test_phenology_cn_ledger_and_control_ports_have_exact_scm_contracts() -> None:
    """Bind every retained phenology ledger/control port to its runtime mode."""

    writers = {
        "static_phenology",
        "gdd_phenology",
        "gsi_phenology",
        "crop_phenology",
    }
    state_units = {
        "froot_carbon": "kg C m-2",
        "leaf_nitrogen": "kg N m-2",
        "froot_nitrogen": "kg N m-2",
        "plant_nitrogen_pool": "kg N m-2",
    }
    control_units = {
        "phenology_phase": "dimensionless",
        "phenology_senescence_factor": "dimensionless",
    }
    flux_units = {
        "leaf_litterfall_carbon": "kg C m-2 s-1",
        "froot_litterfall_carbon": "kg C m-2 s-1",
        "leaf_litterfall_nitrogen": "kg N m-2 s-1",
        "froot_litterfall_nitrogen": "kg N m-2 s-1",
        "nitrogen_retranslocation": "kg N m-2 s-1",
    }
    registry = ContractRegistry(strict=True)
    for redundant in ("leaf_turnover_flux", "froot_turnover_flux"):
        assert not Path(
            f"relic/contracts/variables/biosphere/{redundant}.yaml"
        ).exists()
    for variable in [*state_units, *control_units, *flux_units, "harvest_index"]:
        path = Path(f"relic/contracts/variables/biosphere/{variable}.yaml")
        assert path.is_file(), f"Missing SCM contract: {path}"
        registry.load_contract_file(path)

    for variable, unit in state_units.items():
        contract = registry.get_contract(variable)
        assert contract is not None
        assert contract.kind == "state"
        assert contract.udunits == unit
        assert contract.owner is None
        assert contract.aliases == []
        assert contract.incrementable is True
        assert set(contract.increment_writers) == writers
        assert contract.tendency_writers == []

    for variable, unit in {**control_units, **flux_units}.items():
        contract = registry.get_contract(variable)
        assert contract is not None
        expected_kind = "diagnostic" if variable in control_units else "flux"
        assert contract.kind == expected_kind
        assert contract.udunits == unit
        assert contract.owner is None
        assert contract.aliases == []
        assert contract.incrementable is False
        assert contract.increment_writers == []
        assert set(contract.tendency_writers) == writers

    boundary_domains = {
        "leaf_litterfall_carbon": "carbon.total",
        "froot_litterfall_carbon": "carbon.total",
        "leaf_litterfall_nitrogen": "nitrogen.total",
        "froot_litterfall_nitrogen": "nitrogen.total",
    }
    for variable, domain in boundary_domains.items():
        contract = registry.get_contract(variable)
        assert contract is not None
        assert contract.accumulates == domain
        assert contract.flux_in == []
        assert set(contract.flux_out) == writers

    retranslocation = registry.get_contract("nitrogen_retranslocation")
    assert retranslocation is not None
    assert retranslocation.accumulates is None
    assert retranslocation.flux_in == []
    assert retranslocation.flux_out == []

    harvest = registry.get_contract("harvest_index")
    assert harvest is not None
    assert harvest.kind == "diagnostic"
    assert harvest.udunits == "dimensionless"
    assert harvest.owner == "crop_phenology"
    assert harvest.aliases == []
    assert harvest.incrementable is False
    assert harvest.increment_writers == []
    assert harvest.tendency_writers == ["crop_phenology"]


def test_litterfall_interface_target_ports_have_one_cn_chain_map_owner() -> None:
    """Bind target-side litter C/N input ports to the unique chain-map writer."""

    expected = {
        "metabolic_litter_input": (
            "g C m-2 s-1",
            "carbon.total",
            ["first_order_decomposition", "pools_matrix"],
        ),
        "structural_litter_input": (
            "g C m-2 s-1",
            "carbon.total",
            ["first_order_decomposition", "pools_matrix"],
        ),
        "organic_nitrogen_input": (
            "g N m-2 s-1",
            "nitrogen.total",
            ["mineralization"],
        ),
    }
    registry = ContractRegistry(strict=True)
    for variable in expected:
        path = Path(f"relic/contracts/variables/pedosphere/{variable}.yaml")
        assert path.is_file(), f"Missing SCM contract: {path}"
        registry.load_contract_file(path)

    for variable, (unit, domain, consumers) in expected.items():
        contract = registry.get_contract(variable)
        assert contract is not None
        assert contract.kind == "flux"
        assert contract.udunits == unit
        assert contract.owner == "litterfall_interface_mapper"
        assert contract.aliases == []
        assert contract.incrementable is False
        assert contract.accumulates == domain
        assert contract.increment_writers == []
        assert contract.tendency_writers == ["litterfall_interface_mapper"]
        assert contract.flux_in == ["litterfall_interface_mapper"]
        assert contract.flux_out == consumers
        assert registry.can_write(variable, "litterfall_interface_mapper")
        assert not registry.can_write(variable, "first_order_decomposition")
        assert not registry.can_write(variable, "mineralization")
        provenance = contract.raw["provenance"]
        assert provenance["runtime_writer"] == "litterfall_interface_mapper"


def test_atmospheric_co2_contract_is_external_canonical_forcing() -> None:
    """Atmospheric CO2 must not be owned by the pedosphere gas functor."""

    path = Path("relic/contracts/variables/atmosphere/co2_concentration.yaml")
    assert "REVIEW_NEEDED" not in path.read_text(encoding="utf-8")

    registry = ContractRegistry(strict=True)
    registry.load_contract_file(path)

    contract = registry.get_contract("atmospheric_co2_ppm")
    assert contract is not None
    assert registry.get_contract("co2_concentration") is None
    assert contract.kind == "forcing"
    assert contract.owner is None
    assert contract.writers == []
    assert not registry.can_write(
        "atmospheric_co2_ppm",
        EnvironmentalForcingFunctor().metadata.name,
    )
    assert not registry.can_write("atmospheric_co2_ppm", "GasDiffusion1D")
    assert contract.incrementable is False
    assert contract.raw["provenance"]["runtime_role"] == (
        "external_atmospheric_co2_forcing"
    )
    assert contract.raw["provenance"]["runtime_writer"] is None


def test_vpd_contract_has_one_derived_forcing_writer_in_pa() -> None:
    """Canonical VPD is derived by EnvironmentalForcing, never accumulated."""

    path = Path("relic/contracts/variables/hydrosphere/vapor_pressure_deficit.yaml")
    assert "REVIEW_NEEDED" not in path.read_text(encoding="utf-8")

    registry = ContractRegistry(strict=True)
    registry.load_contract_file(path)

    contract = registry.get_contract("vapor_pressure_deficit")
    owner_name = EnvironmentalForcingFunctor().metadata.name
    assert contract is not None
    assert contract.kind == "forcing"
    assert contract.udunits == "Pa"
    assert contract.owner == owner_name
    assert contract.aliases == []
    assert contract.incrementable is False
    assert contract.increment_writers == []
    assert contract.tendency_writers == [owner_name]
    assert registry.can_write("vapor_pressure_deficit", owner_name)
    assert not registry.can_increment("vapor_pressure_deficit", owner_name)
    assert contract.raw["provenance"]["runtime_writer"] == owner_name


def test_decomposition_flux_contracts_authorize_only_selected_closure() -> None:
    """Alternative decomposers may write one canonical flux set, never transport."""
    paths = {
        "carbon_decomposition_rate": Path(
            "relic/contracts/variables/pedosphere/decomposition_rate.yaml"
        ),
        "heterotrophic_respiration": Path(
            "relic/contracts/variables/pedosphere/heterotrophic_respiration.yaml"
        ),
    }
    registry = ContractRegistry(strict=True)
    for path in paths.values():
        assert "REVIEW_NEEDED" not in path.read_text(encoding="utf-8")
        registry.load_contract_file(path)

    expected_writers = {
        "carbon_decomposition_rate": {
            "first_order_decomposition",
            "pools_matrix",
        },
        "heterotrophic_respiration": {
            "first_order_decomposition",
            "pools_matrix",
            "microbial_enzyme_decomposition",
        },
    }
    for variable, variable_writers in expected_writers.items():
        contract = registry.get_contract(variable)
        assert contract is not None
        assert contract.kind == "flux"
        assert contract.owner is None
        assert contract.udunits == "g C m-2 s-1"
        assert contract.incrementable is False
        assert set(contract.writers) == variable_writers
        assert set(contract.raw["provenance"]["runtime_writers"]) == variable_writers
        for writer in variable_writers:
            assert registry.can_write(variable, writer)
        assert not registry.can_write(variable, "GasDiffusion1D")
    assert not registry.can_write(
        "carbon_decomposition_rate",
        "microbial_enzyme_decomposition",
    )


def test_microbial_enzyme_unique_carbon_states_have_one_increment_writer() -> None:
    """Microbial substrate and biomass states use one explicit ACCUMULATE owner."""
    registry = ContractRegistry(strict=True)
    paths = {
        "available_som_c": Path(
            "relic/contracts/variables/pedosphere/available_som_c.yaml"
        ),
        "microbial_biomass_c": Path(
            "relic/contracts/variables/pedosphere/microbial_biomass_c.yaml"
        ),
    }
    for path in paths.values():
        assert "REVIEW_NEEDED" not in path.read_text(encoding="utf-8")
        registry.load_contract_file(path)

    for variable in paths:
        contract = registry.get_contract(variable)
        assert contract is not None
        assert contract.kind == "state"
        assert contract.owner == "microbial_enzyme_decomposition"
        assert contract.udunits == "g C m-2"
        assert contract.incrementable is True
        assert contract.writers == ["microbial_enzyme_decomposition"]
        assert registry.can_increment(variable, "microbial_enzyme_decomposition")
        assert not registry.can_increment(variable, "first_order_decomposition")


def test_phenology_control_contracts_match_real_replace_writers() -> None:
    """Phenology controls are observations, not accumulable prognostic states."""
    registry = ContractRegistry(strict=True)
    registry.load_contract_file(
        Path("relic/contracts/variables/biosphere/phenological_factor.yaml")
    )
    registry.load_contract_file(
        Path("relic/contracts/variables/biosphere/growth_active.yaml")
    )

    factor = registry.get_contract("phenological_factor")
    active = registry.get_contract("diagnostics.phenology.growth_active")
    assert factor is not None
    assert active is not None
    expected_factor_writers = {
        "static_phenology",
        "gdd_phenology",
        "gsi_phenology",
        "crop_phenology",
    }
    assert factor.kind == "diagnostic"
    assert factor.owner is None
    assert factor.udunits == "dimensionless"
    assert factor.incrementable is False
    assert set(factor.writers) == expected_factor_writers
    assert set(factor.raw["provenance"]["runtime_writers"]) == expected_factor_writers
    assert not registry.can_write("phenological_factor", "phenology_growth")

    assert active.kind == "diagnostic"
    assert active.owner == "phenology_growth"
    assert active.udunits == "dimensionless"
    assert active.incrementable is False
    assert active.writers == ["phenology_growth"]
    writer = active.raw["writers"]["tendencies"][0]
    assert writer["provides"] == ["diagnostics.phenology.growth_active"]
    assert writer["unit"] == active.udunits
    assert active.raw["provenance"]["runtime_writer"] == "phenology_growth"


def test_nutrient_stress_control_contract_matches_replace_writer() -> None:
    """The public N/P control has one non-incrementable runtime writer."""
    registry = ContractRegistry(strict=True)
    registry.load_contract_file(
        Path("relic/contracts/variables/biosphere/nutrient_stress_factor.yaml")
    )

    factor = registry.get_contract("nutrient_stress_factor")
    assert factor is not None
    assert factor.kind == "diagnostic"
    assert factor.owner == "nutrient_stress"
    assert factor.udunits == "dimensionless"
    assert factor.incrementable is False
    assert factor.writers == ["nutrient_stress"]
    writer = factor.raw["writers"]["tendencies"][0]
    assert writer["provides"] == ["nutrient_stress_factor"]
    assert writer["unit"] == factor.udunits
    assert factor.raw["provenance"]["runtime_writer"] == "nutrient_stress"


def test_gas_surface_flux_contracts_have_transport_owner_and_units() -> None:
    """GasDiffusion1D owns boundary fluxes, not reaction source fluxes."""
    expected = {
        "co2_surface_flux": "g C m-2 s-1",
        "n2o_surface_flux": "g N m-2 s-1",
    }
    registry = ContractRegistry(strict=True)
    for variable, units in expected.items():
        path = Path(f"relic/contracts/variables/pedosphere/{variable}.yaml")
        assert "REVIEW_NEEDED" not in path.read_text(encoding="utf-8")
        registry.load_contract_file(path)
        contract = registry.get_contract(variable)
        assert contract is not None
        assert contract.kind == "flux"
        assert contract.owner == "GasDiffusion1D"
        assert contract.udunits == units
        assert contract.incrementable is False
        assert contract.writers == ["GasDiffusion1D"]


def test_gas_profile_transport_contracts_match_update_semantics() -> None:
    """Gas profiles are instantaneous fluxes and concentrations are state increments."""
    expected_flux_units = {
        "co2_flux_profile": "g C m-2 s-1",
        "n2o_flux_profile": "g N m-2 s-1",
    }
    expected_state_units = {
        "soil_co2_concentration": "g C m-3",
        "n2o_concentration": "g N m-3",
    }
    registry = ContractRegistry(strict=True)

    for variable, units in expected_flux_units.items():
        path = Path(f"relic/contracts/variables/pedosphere/{variable}.yaml")
        assert "REVIEW_NEEDED" not in path.read_text(encoding="utf-8")
        registry.load_contract_file(path)
        contract = registry.get_contract(variable)
        assert contract is not None
        assert contract.kind == "flux"
        assert contract.owner == "GasDiffusion1D"
        assert contract.udunits == units
        assert contract.incrementable is False
        assert contract.increment_writers == []
        assert contract.tendency_writers == ["GasDiffusion1D"]

    for variable, units in expected_state_units.items():
        path = Path(f"relic/contracts/variables/pedosphere/{variable}.yaml")
        assert "REVIEW_NEEDED" not in path.read_text(encoding="utf-8")
        registry.load_contract_file(path)
        contract = registry.get_contract(variable)
        assert contract is not None
        assert contract.kind == "state"
        assert contract.owner == "GasDiffusion1D"
        assert contract.udunits == units
        assert contract.incrementable is True
        assert contract.increment_writers == ["GasDiffusion1D"]
        assert contract.tendency_writers == []
        assert registry.can_increment(variable, "GasDiffusion1D")


def test_pedosphere_interface_ports_are_instantaneous_water_fluxes() -> None:
    """The declared coupling map writes three non-incrementable water ports."""
    variables = ("leaching_flux", "lateral_flow", "surface_loss")
    writer = "pedosphere_interface_mapper"
    registry = ContractRegistry(strict=True)

    for variable in variables:
        path = Path(f"relic/contracts/variables/pedosphere/{variable}.yaml")
        assert "REVIEW_NEEDED" not in path.read_text(encoding="utf-8")
        registry.load_contract_file(path)
        contract = registry.get_contract(variable)
        assert contract is not None
        assert contract.kind == "flux"
        assert contract.owner == writer
        assert contract.udunits == "kg m-2 s-1"
        assert contract.incrementable is False
        assert contract.increment_writers == []
        assert contract.tendency_writers == [writer]
        assert registry.can_write(variable, writer)
        assert not registry.can_increment(variable, writer)


def test_soil_vegetation_flux_ports_have_exact_scm_contracts() -> None:
    """Soil-vegetation coupling owns two instantaneous water-flux ports."""

    expected_units = {
        "root_uptake_profile": "kg m-3 s-1",
        "effective_transpiration": "kg m-2 s-1",
    }
    registry = ContractRegistry(strict=True)
    for variable, unit in expected_units.items():
        path = Path(f"relic/contracts/variables/hydrosphere/{variable}.yaml")
        assert path.is_file(), f"Missing SCM contract: {path}"
        assert "REVIEW_NEEDED" not in path.read_text(encoding="utf-8")
        registry.load_contract_file(path)

        contract = registry.get_contract(variable)
        assert contract is not None
        assert contract.kind == "flux"
        assert contract.udunits == unit
        assert contract.owner == "soil_vegetation"
        assert contract.aliases == []
        assert contract.incrementable is False
        assert contract.increment_writers == []
        assert contract.tendency_writers == ["soil_vegetation"]
        assert registry.can_write(variable, "soil_vegetation")
        assert not registry.can_write(variable, "richards_soil_moisture")
        assert contract.raw["provenance"]["runtime_writer"] == "soil_vegetation"


def test_no3_nh4_contract_declares_real_increment_writers() -> None:
    """
    References:
    - docs/source/architecture_overview.md (ContractRegistry ownership policy).
    - LeVeque (2002), finite-volume conservation bookkeeping.

    Mathematical expression:
    - S^(n+1) = S^n + sum_i(delta_i) for incrementable state variables.

    Parameter meanings:
    - owner: null because the pools have multiple additive process writers.
    - increment_writers: permitted contributors to ACCUMULATE updates.
    """
    registry = ContractRegistry(strict=True)
    registry.load_contracts_dir(Path("relic/contracts/variables"))

    no3 = registry.get_contract("no3_nitrogen")
    nh4 = registry.get_contract("nh4_nitrogen")
    assert no3 is not None
    assert nh4 is not None

    assert "REVIEW_NEEDED" not in Path(
        "relic/contracts/variables/pedosphere/no3_nitrogen.yaml"
    ).read_text(encoding="utf-8")
    assert "REVIEW_NEEDED" not in Path(
        "relic/contracts/variables/pedosphere/nh4_nitrogen.yaml"
    ).read_text(encoding="utf-8")
    assert no3.owner is None
    assert nh4.owner is None

    allowed_no3 = {
        "nitrification",
        "denitrification",
        "nitrogen_leaching",
        "root_nutrient_uptake",
    }
    for writer in allowed_no3:
        assert registry.can_increment("no3_nitrogen", writer), writer
    assert not registry.can_increment("no3_nitrogen", "unauthorized_no3_writer")

    allowed_nh4 = {
        "NitrogenCycling",
        "mineralization",
        "nitrification",
        "root_nutrient_uptake",
    }
    for writer in allowed_nh4:
        assert registry.can_increment("nh4_nitrogen", writer), writer
    assert not registry.can_increment("nh4_nitrogen", "unauthorized_nh4_writer")
    assert set(nh4.flux_in) == {"mineralization", "NitrogenCycling"}
    assert set(nh4.flux_out) == {
        "mineralization",
        "nitrification",
        "root_nutrient_uptake",
    }


def test_nitrogen_contracts_bind_runtime_functor_names_and_units() -> None:
    """Every nitrogen state, flux, and diagnostic must name its real writer."""

    registry = ContractRegistry()
    registry.load_contracts_dir(Path("relic/contracts/variables"))
    expected = {
        "organic_nitrogen": ("mineralization", "g N m-2", True),
        "microbial_nitrogen": ("mineralization", "g N m-2", True),
        "plant_n_storage": ("root_nutrient_uptake", "mol N m-2", True),
        "plant_p_storage": ("root_nutrient_uptake", "g P m-2", True),
        "mineralization_flux": ("mineralization", "g N m-2 s-1", False),
        "gross_mineralization_flux": ("mineralization", "g N m-2 s-1", False),
        "immobilization_flux": ("mineralization", "g N m-2 s-1", False),
        "microbial_death_flux": ("mineralization", "g N m-2 s-1", False),
        "nitrification_flux": ("nitrification", "g N m-2 s-1", False),
        "denitrification_flux": ("denitrification", "g N m-2 s-1", False),
        "nitrogen_uptake_rate": (
            "root_nutrient_uptake",
            "mol N m-2 s-1",
            False,
        ),
        "phosphorus_uptake_rate": (
            "root_nutrient_uptake",
            "g P m-2 s-1",
            False,
        ),
        "no3_uptake_by_layer": (
            "root_nutrient_uptake",
            "mol N m-2 s-1",
            False,
        ),
        "nh4_uptake_by_layer": (
            "root_nutrient_uptake",
            "mol N m-2 s-1",
            False,
        ),
        "po4_uptake_by_layer": (
            "root_nutrient_uptake",
            "g P m-2 s-1",
            False,
        ),
        "fertilizer_n_flux": (
            "fertilization_management",
            "kg N m-2 s-1",
            False,
        ),
        "fertilizer_p_flux": (
            "fertilization_management",
            "g P m-2 s-1",
            False,
        ),
    }

    for variable, (owner, units, incrementable) in expected.items():
        contract = registry.get_contract(variable)
        assert contract is not None, variable
        domain = (
            "anthroposphere"
            if variable.startswith("fertilizer_")
            else "pedosphere"
        )
        source = Path("relic/contracts/variables") / domain / f"{variable}.yaml"
        assert "REVIEW_NEEDED" not in source.read_text(encoding="utf-8")
        assert contract.owner == owner, variable
        assert contract.udunits == units, variable
        assert contract.incrementable is incrementable, variable
        assert registry.can_write(variable, owner), variable
        assert not registry.can_write(variable, "NitrogenCycling"), variable

    nitrogen_leaching = registry.get_contract("nitrogen_leaching_flux")
    assert nitrogen_leaching is not None
    assert nitrogen_leaching.kind == "flux"
    assert nitrogen_leaching.owner is None
    assert nitrogen_leaching.udunits == "g N m-2 s-1"
    assert nitrogen_leaching.incrementable is False
    assert set(nitrogen_leaching.tendency_writers) == {
        "nitrogen_leaching",
        "SoluteAdvectionDiffusion1D",
    }
    assert registry.can_write("nitrogen_leaching_flux", "nitrogen_leaching")
    assert registry.can_write(
        "nitrogen_leaching_flux", "SoluteAdvectionDiffusion1D"
    )
    assert not registry.can_write("nitrogen_leaching_flux", "NitrogenCycling")

    available_p = registry.get_contract("available_phosphorus")
    assert available_p is not None
    assert available_p.kind == "state"
    assert available_p.owner is None
    assert available_p.udunits == "g P m-2"
    assert available_p.incrementable is True
    assert available_p.accumulates == "phosphorus.total"
    assert set(available_p.increment_writers) == {
        "fertilization_management",
        "root_nutrient_uptake",
    }
    assert available_p.tendency_writers == []
    assert registry.can_increment(
        "available_phosphorus", "fertilization_management"
    )
    assert registry.can_increment(
        "available_phosphorus", "root_nutrient_uptake"
    )
    assert not registry.can_increment("available_phosphorus", "NitrogenCycling")

    for variable in ("organic_nitrogen", "microbial_nitrogen"):
        contract = registry.get_contract(variable)
        assert contract is not None, variable
        assert contract.kind == "state", variable
        assert contract.accumulates == "nitrogen.total", variable
        assert contract.increment_writers == ["mineralization"], variable
        assert contract.tendency_writers == [], variable
        assert registry.can_increment(variable, "mineralization"), variable
        assert not registry.can_increment(variable, "NitrogenCycling"), variable

    n2o = registry.get_contract("n2o_flux")
    assert n2o is not None
    assert n2o.kind == "flux"
    assert n2o.owner is None
    assert n2o.udunits == "g N m-2 s-1"
    assert n2o.incrementable is True
    assert n2o.accumulates is None
    assert set(n2o.writers) == {"nitrification", "denitrification"}
    assert registry.can_increment("n2o_flux", "nitrification")
    assert registry.can_increment("n2o_flux", "denitrification")
    assert not registry.can_increment("n2o_flux", "GasDiffusion1D")
    assert not registry.can_increment("n2o_flux", "NitrogenCycling")
    for writer in n2o.raw["writers"]["increments"]:
        assert writer["provides"] == ["n2o_flux"]
        assert writer["unit"] == n2o.udunits


def test_nitrogen_cycling_reports_no3_budget_without_rewriting_sources() -> None:
    """
    References:
    - Parton et al. (1987) nitrogen transformation bookkeeping.
    - Li et al. (1992), process-form nitrogen loss pathways.

    Mathematical expression:
    - delta_NO3 = delta_nitrification + delta_denitrification + delta_leaching + delta_root_uptake.

    Parameter meanings:
    - process pool outputs: per-step storage increments [g N m-2].
    """
    nitrogen = NitrogenCycling(
        config={},
        enable_mineralization=False,
        enable_nitrification=True,
        enable_denitrification=True,
        enable_leaching=True,
        enable_root_uptake=True,
    )

    nitrification_delta_no3 = _tensor(0.45)
    denitrification_delta_no3 = _tensor(-0.20)
    leaching_delta_no3 = _tensor(-0.10)
    root_delta_no3 = _tensor(-0.05)
    nitrification_delta_nh4 = _tensor(-0.50)
    root_delta_nh4 = _tensor(-0.03)

    subprocess_outputs = {
        "nitrification": {
            "nh4_nitrogen": nitrification_delta_nh4,
            "no3_nitrogen": nitrification_delta_no3,
            "nitrification_flux": _tensor(0.50),
        },
        "denitrification": {
            "no3_nitrogen": denitrification_delta_no3,
            "denitrification_flux": _tensor(0.20),
        },
        "nitrogen_leaching": {
            "no3_nitrogen": leaching_delta_no3,
            "nitrogen_leaching_flux": _tensor(0.10),
        },
        "root_nutrient_uptake": {
            "no3_nitrogen": root_delta_no3,
            "nh4_nitrogen": root_delta_nh4,
            "nitrogen_uptake_rate": _tensor(0.03),
        },
    }

    state = {
        "nh4_nitrogen": _tensor(5.0),
        "no3_nitrogen": _tensor(10.0),
    }

    output = nitrogen.forward(
        state=state,
        time=0.0,
        dt=1.0,
        subprocess_outputs=subprocess_outputs,
    )

    expected_delta_no3 = (
        nitrification_delta_no3
        + denitrification_delta_no3
        + leaching_delta_no3
        + root_delta_no3
    )
    expected_delta_nh4 = nitrification_delta_nh4 + root_delta_nh4

    assert torch.allclose(
        output["diagnostics.nitrogen_cycling.expected_delta_no3_nitrogen"],
        expected_delta_no3,
    )
    assert torch.allclose(
        output["diagnostics.nitrogen_cycling.expected_delta_nh4_nitrogen"],
        expected_delta_nh4,
    )
    assert "no3_nitrogen" not in output
    assert "nh4_nitrogen" not in output
    for flux_key in (
        "nitrification_flux",
        "denitrification_flux",
        "nitrogen_leaching_flux",
        "nitrogen_uptake_rate",
    ):
        assert flux_key not in output
