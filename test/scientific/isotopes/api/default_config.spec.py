from pathlib import Path
import warnings

import pytest

from relic.contracts.contract_loader import ContractRegistry
from relic.scientific.isotopes import get_default_config
from relic.scientific.isotopes.config import (
    IsotopeConfig,
    IsotopeSpecies,
    IsotopeSpeciesConfig,
)
from relic.scientific.isotopes.config import multi_isotope_models

IsotopeStandard = multi_isotope_models.IsotopeStandard


def test_default_config_is_explicitly_runtime_unbound() -> None:
    config = get_default_config()

    assert isinstance(config, IsotopeConfig)
    assert set(config.species_configs) == {
        IsotopeSpecies.CARBON_13,
        IsotopeSpecies.NITROGEN_15,
        IsotopeSpecies.OXYGEN_18,
    }
    assert config.get_enabled_species() == set()
    assert config.tracked_species == set()
    assert config.enabled is False
    assert config.unified_output.variables == {}

    expected_blockers = {
        IsotopeSpecies.CARBON_13: "canonical_parent_selection_unbound",
        IsotopeSpecies.NITROGEN_15: "nuclide_mass_conversion_unbound",
        IsotopeSpecies.OXYGEN_18: "missing_17O_closure",
    }
    for species, blocker in expected_blockers.items():
        species_config = config.species_configs[species]
        assert species_config.enabled is False
        assert blocker in species_config.runtime_blocker

        payload = species_config.model_dump()
        payload["enabled"] = True
        with pytest.raises(ValueError, match=blocker):
            IsotopeSpeciesConfig.model_validate(payload)


def test_default_config_emits_no_false_runtime_enablement_warnings() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        get_default_config()

    assert [str(item.message) for item in caught] == []


def test_nitrogen_candidate_uses_canonical_parent_state_contracts() -> None:
    config = get_default_config()
    nitrogen = config.species_configs[IsotopeSpecies.NITROGEN_15]
    expected_parents = {
        "organic_nitrogen",
        "nh4_nitrogen",
        "no3_nitrogen",
    }

    assert set(nitrogen.pools) == expected_parents
    registry = ContractRegistry(strict=True)
    registry.load_contracts_dir(Path("relic/contracts/variables"))

    for pool_name, pool in nitrogen.pools.items():
        contract = registry.get_contract(pool.state_variable)
        assert contract is not None, pool_name
        assert contract.kind == "state"
        assert pool.units == contract.udunits


def test_carbon_candidate_pool_units_match_contracted_parents() -> None:
    """A pool declares the unit of the StateManager parent it names; where the
    parent has a variable contract, that is the contract unit. The candidate's
    amount-of-substance basis is reached only through the declared, still
    unbound mass-to-amount conversion (runtime blocker)."""
    config = get_default_config()
    carbon = config.species_configs[IsotopeSpecies.CARBON_13]
    registry = ContractRegistry(strict=True)
    registry.load_contracts_dir(Path("relic/contracts/variables"))

    contracted = 0
    for pool_name, pool in carbon.pools.items():
        contract = registry.get_contract(pool.state_variable)
        if contract is None:
            continue
        contracted += 1
        assert pool.units == contract.udunits, pool_name
    assert contracted >= 2
    assert "parent_mass_to_amount_conversion_unbound" in carbon.runtime_blocker


@pytest.mark.parametrize(
    "pool_name,parent",
    [("root_biomass", "root_biomass_c"), ("leaf_biomass", "leaf_biomass_c")],
)
def test_carbon_biomass_pools_have_contracted_carbon_basis_parents(
    pool_name: str, parent: str
) -> None:
    """A carbon isotope ratio needs a carbon-basis parent: the root and leaf
    pools bind root_biomass_c / leaf_biomass_c (state, kg C m-2, written by
    biomass_allocation), not dry-matter or uncontracted biomass variables."""
    config = get_default_config()
    carbon = config.species_configs[IsotopeSpecies.CARBON_13]
    pool = carbon.pools[pool_name]
    registry = ContractRegistry(strict=True)
    registry.load_contracts_dir(Path("relic/contracts/variables"))
    contract = registry.get_contract(pool.state_variable)

    assert pool.state_variable == parent
    assert contract is not None
    assert contract.kind == "state"
    assert contract.udunits == pool.units == "kg C m-2"
    assert carbon.enabled is False
    assert (
        "canonical_parent_selection_unbound(atmospheric_co2,soil_organic_carbon)"
        in carbon.runtime_blocker
    )


def test_species_config_declares_ratio_and_component_amount_basis() -> None:
    config = get_default_config()
    expected_bases = {
        IsotopeSpecies.CARBON_13: "amount_of_substance",
        IsotopeSpecies.NITROGEN_15: "element_mass",
        IsotopeSpecies.OXYGEN_18: "amount_of_substance",
    }

    for species, species_config in config.species_configs.items():
        fields = type(species_config).model_fields
        assert "natural_abundance" not in fields
        assert species_config.ratio_convention == (
            "heavy_to_light_amount_ratio"
        )
        assert species_config.component_amount_basis == expected_bases[species]
        assert species_config.standard_heavy_to_light_ratio > 0.0


def test_runtime_enablement_fields_reject_normal_mutation_and_copy() -> None:
    config = get_default_config()
    oxygen = config.species_configs[IsotopeSpecies.OXYGEN_18]
    oxygen_before = oxygen.model_dump(mode="python", round_trip=True)
    config_before = config.model_dump(mode="python", round_trip=True)

    with pytest.raises(ValueError, match="runtime_unbound"):
        oxygen.enabled = True
    assert oxygen.model_dump(mode="python", round_trip=True) == oxygen_before

    with pytest.raises(ValueError, match="runtime_unbound"):
        oxygen.model_copy(update={"enabled": True})
    with pytest.raises(ValueError, match="runtime_unbound"):
        oxygen.copy(update={"enabled": True})

    with pytest.raises(ValueError, match="runtime_unbound"):
        config.enabled = True
    with pytest.raises(ValueError, match="runtime disabled"):
        config.tracked_species = {IsotopeSpecies.OXYGEN_18}
    with pytest.raises(AttributeError):
        config.tracked_species.add(IsotopeSpecies.OXYGEN_18)
    with pytest.raises(ValueError, match="runtime disabled"):
        config.model_copy(
            update={"tracked_species": {IsotopeSpecies.OXYGEN_18}}
        )
    with pytest.raises(ValueError, match="runtime_unbound"):
        config.copy(update={"enabled": True})

    assert config.model_dump(mode="python", round_trip=True) == config_before


def test_candidate_identity_and_standard_fields_reject_mutation() -> None:
    config = get_default_config()
    oxygen = config.species_configs[IsotopeSpecies.OXYGEN_18]
    oxygen_before = oxygen.model_dump(mode="python", round_trip=True)
    config_before = config.model_dump(mode="python", round_trip=True)

    for field_name, value in (
        ("species", IsotopeSpecies.NITROGEN_15),
        ("standard_heavy_to_light_ratio", 0.5),
        ("component_amount_basis", "element_mass"),
        ("runtime_blocker", "cleared"),
    ):
        with pytest.raises(ValueError, match="immutable"):
            setattr(oxygen, field_name, value)
        assert oxygen.model_dump(mode="python", round_trip=True) == oxygen_before

    with pytest.raises(ValueError, match="immutable"):
        config.species_configs = {
            IsotopeSpecies.CARBON_13: config.species_configs[
                IsotopeSpecies.NITROGEN_15
            ]
        }
    assert config.model_dump(mode="python", round_trip=True) == config_before


def test_runtime_disabled_candidates_expose_no_enabled_process_payload() -> None:
    config = get_default_config()

    for species_config in config.species_configs.values():
        assert species_config.delta_calculation is False
        assert all(
            process.enabled is False
            for process in species_config.get_all_processes().values()
        )
    assert config.cross_constraints.mass_balance_check is False
    assert config.cross_constraints.thermodynamic_consistency is False
    assert config.cross_constraints.stoichiometric_constraints is False


def test_air_standard_uses_ratio_parameter_not_abundance(monkeypatch) -> None:
    requested = []
    sentinels = {
        "isotopes.standards.carbon_13.pdb_ratio": 0.011,
        "isotopes.standards.nitrogen_15.air_ratio": 0.004,
        "isotopes.standards.nitrogen_15.abundance": 0.400,
        "isotopes.standards.oxygen_18.vsmow_ratio": 0.002,
    }

    class _ParameterManager:
        def get(self, key, **_kwargs):
            requested.append(key)
            return sentinels[key]

    monkeypatch.setattr(
        multi_isotope_models,
        "get_global_manager",
        lambda: _ParameterManager(),
    )
    first = multi_isotope_models.get_isotope_standards()
    sentinels["isotopes.standards.nitrogen_15.air_ratio"] = 0.005
    second = multi_isotope_models.get_isotope_standards()

    assert first[IsotopeStandard.AIR_N2] == 0.004
    assert second[IsotopeStandard.AIR_N2] == 0.005
    assert "isotopes.standards.nitrogen_15.air_ratio" in requested
    assert "isotopes.standards.nitrogen_15.abundance" not in requested


def test_default_output_variables_leave_the_parameter_value_unchanged() -> None:
    """The per-variable compression and fill value are set on the configuration built
    from isotopes.n15.output.variables_default, never on the stored parameter value."""
    import copy

    from relic.parameters import get_global_manager
    from relic.scientific.isotopes.config.models import _default_output_variables

    manager = get_global_manager()
    before = copy.deepcopy(manager.get("isotopes.n15.output.variables_default", required=True))
    _default_output_variables()
    assert manager.get("isotopes.n15.output.variables_default", required=True) == before
