from pathlib import Path

import jsonschema
import yaml

from relic.contracts.contract_loader import ContractRegistry


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "relic" / "contracts" / "schemas" / "scm_variable_schema.yaml"
NITRIFICATION_PATH = ROOT / "relic" / "contracts" / "variables" / "pedosphere" / "nitrification_flux.yaml"
BIOSPHERE_CONTRACTS = ROOT / "relic" / "contracts" / "variables" / "biosphere"
OVERLAP_POLICY_PATH = (
    ROOT / "relic" / "core" / "simulator" / "output_overlap_whitelist.json"
)


def test_contract_schema_kind_supports_flux_and_forcing() -> None:
    schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    enum_values = schema["properties"]["kind"]["enum"]
    assert "flux" in enum_values
    assert "forcing" in enum_values


def test_contract_schema_freezes_publication_support_vocabulary() -> None:
    schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    support = schema["properties"]["support"]

    assert set(support["required"]) == {
        "spatial_kind",
        "area_basis",
        "temporal_basis",
        "vertical_basis",
    }
    assert support["additionalProperties"] is False
    for field_name in (
        "spatial_kind",
        "area_basis",
        "temporal_basis",
        "vertical_basis",
    ):
        field_schema = support["properties"][field_name]
        assert set(field_schema["type"]) == {"string", "null"}
        assert None in field_schema["enum"]
    assert set(support["properties"]["spatial_kind"]["enum"]) - {None} == {
        "leaf",
        "canopy",
        "grid_cell",
        "footprint",
        "tower_height",
        "vertical_column",
    }
    assert set(support["properties"]["area_basis"]["enum"]) - {None} == {
        "leaf_area",
        "ground_area",
        "grid_area",
    }
    assert set(support["properties"]["temporal_basis"]["enum"]) - {None} == {
        "instantaneous",
        "interval_mean",
        "interval_integral",
    }
    assert set(support["properties"]["vertical_basis"]["enum"]) - {None} == {
        "none",
        "height_level",
        "layer_integral",
    }


def test_contract_schema_accepts_explicit_not_applicable_support() -> None:
    schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    payload = {
        "variable": "air_temperature",
        "kind": "forcing",
        "dimension": {"udunits": "K"},
        "support": {
            "spatial_kind": "grid_cell",
            "area_basis": None,
            "temporal_basis": "instantaneous",
            "vertical_basis": "none",
        },
    }

    jsonschema.validate(instance=payload, schema=schema)


def test_g2_contracts_freeze_shape_and_support_semantics() -> None:
    expected = {
        "air_temperature": ("grid_cell", None, "instantaneous", "none"),
        "relative_humidity": ("grid_cell", None, "instantaneous", "none"),
        "surface_albedo": ("grid_cell", None, "instantaneous", "none"),
        "ppfd": ("grid_cell", "ground_area", "instantaneous", "none"),
        "atmospheric_co2_ppm": (
            "grid_cell",
            None,
            "instantaneous",
            "none",
        ),
        "vapor_pressure_deficit": (
            "grid_cell",
            None,
            "instantaneous",
            "none",
        ),
        "gross_photosynthesis": (
            "canopy",
            "ground_area",
            "instantaneous",
            "none",
        ),
        "electron_transport_rate": (
            "canopy",
            "ground_area",
            "instantaneous",
            "none",
        ),
        "leaf_area_index": (
            "canopy",
            "ground_area",
            "instantaneous",
            "none",
        ),
        "isoprene_emission": (
            "grid_cell",
            "ground_area",
            "instantaneous",
            "none",
        ),
        "bvoc_carbon_allocation_flux": (
            "grid_cell",
            "ground_area",
            "instantaneous",
            "none",
        ),
        "gross_primary_productivity": (
            "grid_cell",
            "ground_area",
            "instantaneous",
            "none",
        ),
        "maintenance_respiration_total": (
            "grid_cell",
            "ground_area",
            "instantaneous",
            "none",
        ),
        "growth_respiration_total": (
            "grid_cell",
            "ground_area",
            "instantaneous",
            "none",
        ),
        "total_autotrophic_respiration": (
            "grid_cell",
            "ground_area",
            "instantaneous",
            "none",
        ),
        "net_primary_productivity": (
            "grid_cell",
            "ground_area",
            "instantaneous",
            "none",
        ),
        "net_radiation": (
            "grid_cell",
            "ground_area",
            "instantaneous",
            "none",
        ),
        "evapotranspiration_flux": (
            "grid_cell",
            "ground_area",
            "instantaneous",
            "none",
        ),
        "soil_moisture": ("grid_cell", None, "instantaneous", "none"),
        "precipitation_flux": (
            "grid_cell",
            "ground_area",
            "instantaneous",
            "none",
        ),
        "surface_runoff_flux": (
            "grid_cell",
            "ground_area",
            "instantaneous",
            "none",
        ),
        "drainage_flux": (
            "grid_cell",
            "ground_area",
            "instantaneous",
            "none",
        ),
        "incoming_shortwave_potential": (
            "grid_cell",
            None,
            "instantaneous",
            "none",
        ),
        "cos_zenith": ("grid_cell", None, "instantaneous", "none"),
        "surface_temperature": (
            "grid_cell",
            None,
            "instantaneous",
            "none",
        ),
        "surface_emissivity": (
            "grid_cell",
            None,
            "instantaneous",
            "none",
        ),
        "cloud_fraction": ("grid_cell", None, "instantaneous", "none"),
        "atmospheric_pressure": (
            "grid_cell",
            None,
            "instantaneous",
            "none",
        ),
        "footprint_weights": (
            "footprint",
            "grid_area",
            "interval_mean",
            "none",
        ),
    }
    registry = ContractRegistry(strict=True)
    registry.load_contracts_dir(ROOT / "relic" / "contracts" / "variables")

    for variable, support_values in expected.items():
        contract = registry.get_contract(variable)
        assert contract is not None, variable
        support = contract.raw.get("support")
        assert isinstance(support, dict), variable
        assert tuple(
            support[field_name]
            for field_name in (
                "spatial_kind",
                "area_basis",
                "temporal_basis",
                "vertical_basis",
            )
        ) == support_values
        assert support["shape"] == {"dimensions": ["cell"]}


def test_nitrification_flux_kind_is_valid_against_schema_enum() -> None:
    schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    enum_values = set(schema["properties"]["kind"]["enum"])
    payload = yaml.safe_load(NITRIFICATION_PATH.read_text(encoding="utf-8"))
    assert payload["kind"] in enum_values


def test_photosynthesis_contracts_encode_units_and_canonical_owners() -> None:
    expected = {
        "gross_photosynthesis": ("flux", "umol CO2 m-2 s-1", "farquhar_photosynthesis"),
        "net_photosynthesis": ("flux", "umol CO2 m-2 s-1", "farquhar_photosynthesis"),
        "gross_primary_productivity": ("flux", "kg C m-2 s-1", None),
        "ecosystem_respiration": ("flux", "kg C m-2 s-1", "simple_photosynthesis"),
        "net_ecosystem_exchange": ("flux", "kg C m-2 s-1", "simple_photosynthesis"),
        "ozone_stress_factor": ("diagnostic", "dimensionless", "ozone_stress"),
    }

    for variable, (kind, units, owner) in expected.items():
        path = BIOSPHERE_CONTRACTS / f"{variable}.yaml"
        text = path.read_text(encoding="utf-8")
        payload = yaml.safe_load(text)
        assert "REVIEW_NEEDED" not in text
        assert payload["kind"] == kind
        assert payload["dimension"]["udunits"] == units
        assert payload["owner"] == owner
        assert payload["incrementable"] is False

    gpp = yaml.safe_load(
        (BIOSPHERE_CONTRACTS / "gross_primary_productivity.yaml").read_text(
            encoding="utf-8"
        )
    )
    writers = gpp["writers"]["tendencies"]
    assert {entry["name"] for entry in writers} == {
        "farquhar_photosynthesis",
        "simple_photosynthesis",
    }
    assert all(entry["unit"] == "kg C m-2 s-1" for entry in writers)
    assert all(entry["provides"] == ["gross_primary_productivity"] for entry in writers)


def test_photosynthesis_overlap_policy_contains_only_real_alternatives() -> None:
    import json

    policies = json.loads(OVERLAP_POLICY_PATH.read_text(encoding="utf-8"))
    assert policies["gross_primary_productivity"]["policy"] == "mutually_exclusive"
    assert "gross_photosynthesis" not in policies
    assert "net_photosynthesis" not in policies
    assert "ozone_stress_factor" not in policies


def test_biomass_growth_ports_close_the_structural_carbon_domain() -> None:
    expected_bvoc_writers = {
        "megan_bvoc_model",
        "mechanistic_isoprene_synthesis",
        "mechanistic_monoterpene_synthesis",
        "sesquiterpene_synthesis",
    }
    bvoc_contract = yaml.safe_load(
        (BIOSPHERE_CONTRACTS / "bvoc_isoprene_carbon_loss.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert {
        entry["name"] for entry in bvoc_contract["writers"]["tendencies"]
    } == expected_bvoc_writers
    assert set(bvoc_contract["flux_contract"]["allowed_out"]) == expected_bvoc_writers

    for variable in ("growth_rate_leaf", "growth_rate_stem", "growth_rate_root"):
        contract = yaml.safe_load(
            (BIOSPHERE_CONTRACTS / f"{variable}.yaml").read_text(
                encoding="utf-8"
            )
        )
        assert contract["kind"] == "flux"
        assert contract["dimension"]["udunits"] == "kg C m-2 s-1"
        assert contract["accumulates"] == "carbon.total"
        assert contract["flux_contract"]["allowed_in"] == ["biomass_allocation"]
        assert contract["flux_contract"]["allowed_out"] == []


def test_runtime_unbound_generic_isotope_claim_contracts_are_absent() -> None:
    isotope_contracts = ROOT / "relic" / "contracts" / "variables" / "isotopes"
    forbidden = {
        "isotope_flux": isotope_contracts / "isotope_flux.yaml",
        "fractionation_factor": isotope_contracts / "fractionation_factor.yaml",
    }

    for path in forbidden.values():
        assert not path.exists(), path

    registry = ContractRegistry(strict=True)
    registry.load_contracts_dir(ROOT / "relic" / "contracts" / "variables")
    for variable in forbidden:
        assert registry.get_contract(variable) is None
