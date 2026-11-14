"""L1 tests for the canonical reduced-instantaneous MEGAN BVOC functor.

The tests intentionally distinguish the RELIC reduced instantaneous response
from the full MEGAN2.1 canopy-history algorithm. They enforce canonical input
ports, explicit StateManager update semantics, molar atmospheric exchange
ports, carbon accounting, and the Guenther et al. (2012) CO2 response.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
import warnings

import pytest
import torch

from relic.contracts.contract_loader import ContractRegistry
from relic.core.abstractions import BiologicalProcess
from relic.core.functor.functor_exceptions import ExecutionError
from relic.core.simulator.execution_group import _clean_updates_and_collect_diagnostics
from relic.core.state import StateManager, StateVariableSpec
from relic.core.state.types import UpdateItem, UpdateMode
from relic.data.standards.unit_converter import unit_converter
from relic.parameters import get_global_manager
from relic.scientific.biosphere.bvoc.megan_model import (
    CouplingStrategy,
    EmissionPathway,
    MEGANModel,
)


SPECIES = ("isoprene", "monoterpenes", "sesquiterpenes", "ovocs")
PROJECT_ROOT = Path(__file__).resolve().parents[4]


class _StateUpdateRecorder:
    """Capture accepted StateManager update provenance for assertions."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record_state_update(
        self,
        step: int,
        variable: str,
        pre_value: torch.Tensor,
        post_value: torch.Tensor,
        functor_increments: Dict[str, torch.Tensor],
        unit: str,
    ) -> None:
        self.events.append(
            {
                "step": step,
                "variable": variable,
                "pre_value": pre_value,
                "post_value": post_value,
                "functor_increments": functor_increments,
                "unit": unit,
            }
        )


def _scalar(value: Any) -> float:
    """Return a scalar value for exact parameter comparisons."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().reshape(-1)[0].item())
    return float(value)


def _state(
    *,
    air_temperature: float = 303.15,
    ppfd: float = 1000.0,
    leaf_area_index: float = 1.0,
    atmospheric_co2_ppm: float = 400.0,
    net_photosynthesis: float | None = None,
    stress: bool = False,
    shape: tuple[int, ...] = (),
) -> Dict[str, torch.Tensor]:
    """Build a canonical StateManager-facing MEGAN input mapping."""

    def tensor(value: float) -> torch.Tensor:
        if shape:
            return torch.full(shape, value, dtype=torch.float32)
        return torch.tensor(value, dtype=torch.float32)

    state = {
        "air_temperature": tensor(air_temperature),
        "ppfd": tensor(ppfd),
        "leaf_area_index": tensor(leaf_area_index),
        "atmospheric_co2_ppm": tensor(atmospheric_co2_ppm),
    }
    if net_photosynthesis is not None:
        state["net_photosynthesis"] = tensor(net_photosynthesis)
    if stress:
        state.update(
            {
                "water_stress_factor": tensor(0.8),
                "temperature_stress_factor": tensor(0.9),
                "ozone_stress_factor": tensor(0.95),
            }
        )
    return state


def _values(output: Dict[str, UpdateItem]) -> Dict[str, torch.Tensor]:
    """Unwrap an output only after checking its explicit runtime semantics."""
    values: Dict[str, torch.Tensor] = {}
    for key, item in output.items():
        assert isinstance(item, UpdateItem), key
        assert item.mode is UpdateMode.REPLACE, key
        assert item.per_timestep is False, key
        assert item.source_functor is not None, key
        assert isinstance(item.value, torch.Tensor), key
        values[key] = item.value
    return values


def _forward_values(
    model: MEGANModel,
    state: Dict[str, torch.Tensor],
    *,
    dt: float = 60.0,
) -> Dict[str, torch.Tensor]:
    return _values(model.forward(state, time=0.0, dt=dt))


class TestMEGANEnums:
    def test_coupling_strategy_values_are_stable(self):
        assert {member.value for member in CouplingStrategy} == {
            "simultaneous",
            "sequential",
            "independent",
        }

    def test_emission_pathway_values_are_stable(self):
        assert {member.value for member in EmissionPathway} == {
            "standard",
            "photosynthesis",
            "hybrid",
        }


class TestMEGANInitialization:
    def test_default_configuration_is_explicit_and_non_strict(self):
        model = MEGANModel()
        assert model.name == "megan_bvoc_model"
        assert model.emission_pathway is EmissionPathway.HYBRID
        assert model.coupling_strategy is CouplingStrategy.SEQUENTIAL
        assert model.use_stress_functors is False
        assert model.enabled_compounds == list(SPECIES)
        assert model.device.type == "cpu"

    def test_references_are_registered_bibliographic_records(self):
        """Declared and module references match registered records: the enclosure
        protocol paper is Niinemets et al. (2011) Biogeosciences 8, 2209-2246, the
        Monson et al. global-change review is the 2007 Phil. Trans. R. Soc. A article,
        and the Monson et al. (2012) short citations resolve to the New Phytologist
        isoprene-modelling review."""
        import relic.scientific.biosphere.bvoc.megan_model as module

        declared = " | ".join(MEGANModel().metadata.references)
        assert "Biogeosciences, 8, 2209-2246" in declared
        assert "1233-1277" not in declared
        header = module.__doc__
        assert "367, 1022-1035" not in header
        assert "Philosophical Transactions of the Royal Society A 365, 1677-1695" in header
        assert "New Phytologist 195, 541-559" in header

    def test_enabled_compounds_are_instance_specific(self):
        model = MEGANModel(
            config={"enabled_compounds": ["isoprene", "ovocs"]},
            emission_pathway=EmissionPathway.STANDARD,
        )
        assert model.enabled_compounds == ["isoprene", "ovocs"]
        assert set(model.metadata.output_variables) == set(model.get_provides())
        assert "monoterpenes_emission" not in model.get_provides()

    @pytest.mark.parametrize(
        "enabled",
        [[], ["unknown"], ["isoprene", "unknown"], "isoprene"],
    )
    def test_invalid_or_empty_enabled_compounds_fail_fast(self, enabled):
        with pytest.raises((TypeError, ValueError), match="enabled_compounds"):
            MEGANModel(config={"enabled_compounds": enabled})

    def test_feature_config_cannot_override_scientific_parameters(self):
        manager = get_global_manager()
        model = MEGANModel(
            config={
                "emission_factors": {"isoprene": 999999.0},
                "light_response": {"alpha": 999999.0},
            }
        )
        assert model.isoprene_ef.item() == pytest.approx(
            _scalar(manager.get("biosphere.bvoc.emission_factors.isoprene"))
        )
        assert model.alpha_light.item() == pytest.approx(
            _scalar(manager.get("biosphere.bvoc.megan.alpha_light"))
        )

    def test_explicit_device_places_managed_buffers(self):
        model = MEGANModel(device=torch.device("cpu"))
        assert model.device.type == "cpu"
        assert model.isoprene_ef.device.type == "cpu"
        assert model.micrograms_to_grams.device.type == "cpu"


class TestMEGANCanonicalInterface:
    def test_explicit_state_ppfd_requires_no_radiation_functor(self):
        model = MEGANModel(
            config={"ppfd_source": "state", "enabled_compounds": ["isoprene"]},
            emission_pathway=EmissionPathway.STANDARD,
        )
        assert model.get_dependencies() == []
        assert model.get_required_inputs() == [
            "air_temperature", "ppfd", "leaf_area_index", "atmospheric_co2_ppm"
        ]
        assert torch.isfinite(model(_state(), time=0.0, dt=1800.0)["isoprene_emission"].value)
        missing = _state()
        del missing["ppfd"]
        with pytest.raises(ExecutionError, match="ppfd") as caught:
            model(missing, time=0.0, dt=1800.0)
        assert isinstance(caught.value.__cause__, ValueError)

    def test_state_ppfd_keeps_other_actual_and_declared_producers(self):
        model = MEGANModel(
            config={"ppfd_source": "state", "use_stress_functors": True,
                    "dependencies": ["site_driver"]},
            emission_pathway=EmissionPathway.PHOTOSYNTHESIS,
        )
        assert model.get_dependencies() == [
            "farquhar_photosynthesis", "water_stress", "thermal_stress", "ozone_stress", "site_driver"
        ]

    @pytest.mark.parametrize("source", [None, False, "", "automatic", "measured"])
    def test_ppfd_source_rejects_undeclared_provider_modes(self, source):
        with pytest.raises(ValueError, match="ppfd_source"):
            MEGANModel(config={"ppfd_source": source}, emission_pathway=EmissionPathway.STANDARD)

    def test_standard_dependencies_follow_actual_producers(self):
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        assert model.get_dependencies() == ["atmospheric_radiation"]

    @pytest.mark.parametrize(
        "pathway", [EmissionPathway.PHOTOSYNTHESIS, EmissionPathway.HYBRID]
    )
    def test_photosynthesis_path_dependencies_include_farquhar(self, pathway):
        model = MEGANModel(emission_pathway=pathway)
        assert model.get_dependencies() == [
            "atmospheric_radiation",
            "farquhar_photosynthesis",
        ]

    def test_stress_dependencies_exist_only_when_enabled(self):
        model = MEGANModel(
            config={"use_stress_functors": True},
            emission_pathway=EmissionPathway.STANDARD,
        )
        assert model.get_dependencies() == [
            "atmospheric_radiation",
            "water_stress",
            "thermal_stress",
            "ozone_stress",
        ]

    def test_required_inputs_match_metadata_exactly(self):
        standard = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        assert standard.get_required_inputs() == [
            "air_temperature",
            "ppfd",
            "leaf_area_index",
            "atmospheric_co2_ppm",
        ]
        hybrid = MEGANModel(
            config={"use_stress_functors": True},
            emission_pathway=EmissionPathway.HYBRID,
        )
        assert hybrid.get_required_inputs() == [
            "air_temperature",
            "ppfd",
            "leaf_area_index",
            "atmospheric_co2_ppm",
            "net_photosynthesis",
            "water_stress_factor",
            "temperature_stress_factor",
            "ozone_stress_factor",
        ]
        assert hybrid.metadata.input_variables == hybrid.get_required_inputs()

    @pytest.mark.parametrize(
        ("canonical", "legacy"),
        [
            ("air_temperature", "temperature"),
            ("ppfd", "par"),
            ("leaf_area_index", "foliar_density"),
            ("atmospheric_co2_ppm", "co2_intercellular"),
        ],
    )
    def test_legacy_input_aliases_are_not_runtime_fallbacks(self, canonical, legacy):
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        state = _state()
        state[legacy] = state.pop(canonical)
        with pytest.raises(ValueError, match=canonical):
            model.forward(state, time=0.0, dt=60.0)

    @pytest.mark.parametrize(
        "pathway", [EmissionPathway.PHOTOSYNTHESIS, EmissionPathway.HYBRID]
    )
    def test_selected_photosynthesis_path_requires_net_photosynthesis(self, pathway):
        model = MEGANModel(emission_pathway=pathway)
        with pytest.raises(ValueError, match="net_photosynthesis"):
            model.forward(_state(), time=0.0, dt=60.0)

    def test_enabled_stress_path_requires_each_canonical_factor(self):
        model = MEGANModel(
            config={"use_stress_functors": True},
            emission_pathway=EmissionPathway.STANDARD,
        )
        for key in (
            "water_stress_factor",
            "temperature_stress_factor",
            "ozone_stress_factor",
        ):
            state = _state(stress=True)
            state.pop(key)
            with pytest.raises(ValueError, match=key):
                model.forward(state, time=0.0, dt=60.0)

    def test_declared_stress_factor_set_defines_inputs_and_dependencies(self):
        model = MEGANModel(
            config={
                "use_stress_functors": True,
                "stress_factors": ["temperature", "water"],
            },
            emission_pathway=EmissionPathway.STANDARD,
        )
        assert model.get_required_inputs() == [
            "air_temperature",
            "ppfd",
            "leaf_area_index",
            "atmospheric_co2_ppm",
            "water_stress_factor",
            "temperature_stress_factor",
        ]
        assert model.metadata.input_variables == model.get_required_inputs()
        assert model.get_dependencies() == [
            "atmospheric_radiation",
            "water_stress",
            "thermal_stress",
        ]

    def test_declared_stress_factor_is_required_and_undeclared_one_is_not_read(self):
        model = MEGANModel(
            config={"use_stress_functors": True, "stress_factors": ["water"]},
            emission_pathway=EmissionPathway.STANDARD,
        )
        state = _state(stress=True)
        full = _forward_values(model, state)
        for undeclared in ("temperature_stress_factor", "ozone_stress_factor"):
            state.pop(undeclared)
        reduced = _forward_values(model, state)
        for key in full:
            assert torch.equal(full[key], reduced[key]), key
        state.pop("water_stress_factor")
        with pytest.raises(ValueError, match="water_stress_factor"):
            model.forward(state, time=0.0, dt=60.0)

    @pytest.mark.parametrize(
        ("config", "error"),
        [
            ({"stress_factors": ["water"]}, ValueError),
            ({"use_stress_functors": True, "stress_factors": []}, TypeError),
            ({"use_stress_functors": True, "stress_factors": "water"}, TypeError),
            (
                {"use_stress_functors": True, "stress_factors": ["water", "water"]},
                ValueError,
            ),
            (
                {"use_stress_functors": True, "stress_factors": ["nutrient"]},
                ValueError,
            ),
        ],
    )
    def test_invalid_stress_factor_declarations_are_refused(self, config, error):
        with pytest.raises(error, match="stress_factors"):
            MEGANModel(config=config, emission_pathway=EmissionPathway.STANDARD)


class TestMEGANOutputContract:
    def test_default_output_set_matches_instance_metadata_exactly(self):
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        output = model.forward(_state(), time=0.0, dt=60.0)
        expected = {
            *(f"{compound}_emission" for compound in SPECIES),
            "total_bvoc_emission",
            "bvoc_carbon_allocation_flux",
            "diagnostics.bvoc.megan.environmental_stress_factor",
            "diagnostics.bvoc.megan.co2_inhibition_factor",
            *(f"diagnostics.bvoc.megan.{compound}.gamma_temperature" for compound in SPECIES),
            *(f"diagnostics.bvoc.megan.{compound}.gamma_light" for compound in SPECIES),
        }
        assert set(output) == expected
        assert set(model.metadata.output_variables) == expected
        assert set(model.get_provides()) == expected
        assert "coupling_factor" not in output
        assert "environmental_stress_factor" not in output
        assert "co2_inhibition_factor" not in output
        assert "total_bvoc_carbon_flux" not in output
        assert model.metadata.conservation_laws == {
            "carbon_boundary": {"quantity": "carbon_boundary"}
        }

    def test_all_outputs_have_explicit_replace_semantics_and_units(self):
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        output = model.forward(_state(), time=0.0, dt=60.0)
        for key, item in output.items():
            assert isinstance(item, UpdateItem)
            assert item.mode is UpdateMode.REPLACE
            assert item.per_timestep is False
            assert item.source_functor == model.name
            assert item.element is None
            if key.endswith("_emission"):
                assert item.unit == "mol m-2 s-1"
            elif key == "bvoc_carbon_allocation_flux":
                assert item.unit == "kg C m-2 s-1"
            else:
                assert item.unit == "dimensionless"

    def test_custom_compound_output_set_is_exact(self):
        model = MEGANModel(
            config={"enabled_compounds": ["monoterpenes"]},
            emission_pathway=EmissionPathway.STANDARD,
        )
        output = model.forward(_state(), time=0.0, dt=60.0)
        assert set(output) == {
            "monoterpenes_emission",
            "total_bvoc_emission",
            "bvoc_carbon_allocation_flux",
            "diagnostics.bvoc.megan.environmental_stress_factor",
            "diagnostics.bvoc.megan.monoterpenes.gamma_temperature",
            "diagnostics.bvoc.megan.monoterpenes.gamma_light",
        }
        assert set(output) == set(model.metadata.output_variables)

    def test_grid_shape_device_and_dtype_are_preserved(self):
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        values = _forward_values(model, _state(shape=(2, 3)))
        for value in values.values():
            assert value.shape == (2, 3)
            assert value.device == model.device
            assert value.dtype == torch.float32

    def test_synchronized_ports_commit_through_contract_registry_and_state_manager(self):
        """Canonical synchronized ports must retain units, writer, and open-route semantics."""
        model = MEGANModel(
            config={
                "enabled_compounds": ["isoprene", "monoterpenes", "ovocs"]
            },
            emission_pathway=EmissionPathway.STANDARD,
        )
        contract_names = (
            "isoprene_emission",
            "monoterpenes_emission",
            "ovocs_emission",
            "total_bvoc_emission",
            "bvoc_carbon_allocation_flux",
        )
        registry = ContractRegistry(strict=True)
        contract_root = PROJECT_ROOT / "relic" / "contracts" / "variables" / "biosphere"
        for name in contract_names:
            contract_path = (
                contract_root / "bvoc_isoprene_carbon_loss.yaml"
                if name == "bvoc_carbon_allocation_flux"
                else contract_root / f"{name}.yaml"
            )
            registry.load_contract_file(contract_path)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            output = model.forward(_state(), time=0.0, dt=60.0)
            manager = StateManager(
                schema={},
                autoload_contract_registry=False,
                device="cpu",
            )
            manager._contract_registry = registry
            recorder = _StateUpdateRecorder()
            manager.set_provenance_recorder(recorder)
            diagnostics = {}
            state_updates = _clean_updates_and_collect_diagnostics(
                SimpleNamespace(state_manager=manager), model.name, model, output, diagnostics,
            )
            for key, item in state_updates.items():
                manager.register_variable(
                    StateVariableSpec(
                        name=key,
                        units=item.unit or "dimensionless",
                        default=torch.zeros_like(item.value),
                    )
                )
            sources = {key: model.name for key in state_updates}
            increments = {
                key: {model.name: item.value}
                for key, item in state_updates.items()
            }
            manager.apply_updates(
                state_updates,
                dt=60.0,
                check_conservation=True,
                update_sources=sources,
                update_increments=increments,
            )

        assert caught == []
        assert manager.device.type == "cpu"
        assert model.device.type == "cpu"
        for key, item in output.items():
            assert item.source_functor == model.name
            assert item.element is None
            if key in state_updates:
                assert torch.equal(manager.get_state(key), item.value)
            else:
                assert key not in manager.get_state()
                assert diagnostics[key] is item
                diagnostic_name = f"{model.name}.{key.removeprefix('diagnostics.')}"
                assert manager.conservation_history[-1].custom_checks[diagnostic_name] == pytest.approx(item.value.item())
        for name in contract_names:
            contract = registry.get_contract(name)
            assert contract is not None
            assert contract.udunits == output[name].unit
            assert registry.can_write(name, model.name)
        assert set(manager.get_conservation_ledger().get("carbon", [])) & set(output) == set()
        assert {event["variable"] for event in recorder.events} == set(state_updates)
        assert all(
            set(event["functor_increments"]) == {model.name}
            for event in recorder.events
        )


class TestMEGANScientificNumerics:
    def test_open_carbon_boundary_checker_matches_enabled_species(self):
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        state = _state()
        output = model.forward(state, time=0.0, dt=60.0)

        residuals = model.check_conservation_laws(state, output)

        assert residuals["carbon_boundary"] == pytest.approx(0.0, abs=1.0e-15)
        tampered = dict(output)
        allocation = output["bvoc_carbon_allocation_flux"].value
        tampered["bvoc_carbon_allocation_flux"] = (
            allocation + torch.full_like(allocation, 1.0e-9)
        )
        assert model.check_conservation_laws(
            state,
            tampered,
        )["carbon_boundary"] == pytest.approx(1.0e-9, rel=1.0e-5)

    @pytest.mark.parametrize(
        "missing_key",
        [
            "isoprene_emission",
            "monoterpenes_emission",
            "sesquiterpenes_emission",
            "ovocs_emission",
            "bvoc_carbon_allocation_flux",
        ],
    )
    def test_open_carbon_boundary_checker_requires_complete_outputs(
        self,
        missing_key,
    ):
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        state = _state()
        output = model.forward(state, time=0.0, dt=60.0)
        malformed = dict(output)
        malformed.pop(missing_key)

        with pytest.raises(KeyError, match=missing_key):
            model.check_conservation_laws(state, malformed)

    @pytest.mark.parametrize(
        ("invalid_value", "expected_exception"),
        [
            ("not-a-tensor", TypeError),
            (torch.empty(0), ValueError),
            (torch.tensor(float("nan")), ValueError),
        ],
    )
    def test_open_carbon_boundary_checker_rejects_invalid_values(
        self,
        invalid_value,
        expected_exception,
    ):
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        state = _state()
        output = model.forward(state, time=0.0, dt=60.0)
        malformed = dict(output)
        malformed["bvoc_carbon_allocation_flux"] = invalid_value

        with pytest.raises(expected_exception):
            model.check_conservation_laws(state, malformed)

    def test_open_carbon_boundary_checker_rejects_shape_mismatch(self):
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        state = _state()
        output = model.forward(state, time=0.0, dt=60.0)
        malformed = dict(output)
        malformed["bvoc_carbon_allocation_flux"] = torch.ones(2)

        with pytest.raises(ValueError, match="identical shapes"):
            model.check_conservation_laws(state, malformed)

    def test_open_carbon_boundary_checker_supports_one_enabled_compound(self):
        model = MEGANModel(
            config={"enabled_compounds": ["monoterpenes"]},
            emission_pathway=EmissionPathway.STANDARD,
        )
        state = _state()
        output = model.forward(state, time=0.0, dt=60.0)

        assert model.check_conservation_laws(state, output) == {
            "carbon_boundary": 0.0
        }

    def test_standard_mass_to_molar_conversion_and_carbon_ledger(self):
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        state = _state()
        values = _forward_values(model, state)
        manager = get_global_manager()
        foliar_density = state["leaf_area_index"] * _scalar(
            manager.get("biosphere.bvoc.foliar_density_factor")
        )
        from relic.utils.physical_constants import get_physical_constant

        seconds_per_hour = float(get_physical_constant("seconds_per_hour"))
        micrograms_to_grams = _scalar(
            manager.get("relic.physical.constants.conversions.micrograms_to_grams")
        )
        expected_total_molar = torch.zeros_like(state["air_temperature"])
        expected_carbon_g = torch.zeros_like(state["air_temperature"])
        for compound in SPECIES:
            ef = _scalar(manager.get(f"biosphere.bvoc.emission_factors.{compound}"))
            gamma_t = model._get_temperature_factor(compound, state["air_temperature"])
            gamma_p = model._get_light_factor(compound, state["ppfd"])
            mass_ug = ef * gamma_t * gamma_p * foliar_density / seconds_per_hour
            if compound == "isoprene":
                mass_ug = mass_ug * model._compute_co2_inhibition_factor(
                    state["atmospheric_co2_ppm"]
                )
            mw = model.molecular_weights[compound]
            expected_molar = mass_ug * micrograms_to_grams / mw
            assert torch.allclose(
                values[f"{compound}_emission"], expected_molar, rtol=1e-6, atol=1e-12
            )
            expected_total_molar = expected_total_molar + expected_molar
            expected_carbon_g = expected_carbon_g + (
                expected_molar * mw * model._get_carbon_fraction(compound)
            )
        assert torch.allclose(
            values["total_bvoc_emission"], expected_total_molar, rtol=1e-6, atol=1e-12
        )
        expected_carbon_kg = unit_converter.convert(
            expected_carbon_g,
            variable_name="carbon_flux",
            source_units="g C m-2 s-1",
            target_units="kg C m-2 s-1",
        )
        assert torch.allclose(
            values["bvoc_carbon_allocation_flux"],
            expected_carbon_kg,
            rtol=1e-6,
            atol=1e-15,
        )
        assert torch.all(values["bvoc_carbon_allocation_flux"] >= 0)

    def test_governed_standard_savedpoint_matches_independent_double_precision(self):
        """BE-Vie source row 25611 exposed Eq. (8) reciprocal cancellation.

        Check returned float32 ports against an independent scalar-double oracle,
        using the same represented input and coefficient values, not shared laws.
        """
        import math

        model = MEGANModel(
            config={"ppfd_source": "state", "enabled_compounds": ["isoprene"]},
            emission_pathway=EmissionPathway.STANDARD,
        )
        state = _state(
            air_temperature=300.48999999999995,
            ppfd=1785.44,
            leaf_area_index=5.899923839606345,
            atmospheric_co2_ppm=387.45,
        )
        saved_inputs = {key: value.clone() for key, value in state.items()}
        values = _values(model(state, time=0.0, dt=1800.0))
        manager = get_global_manager()
        temperature, ppfd = _scalar(state["air_temperature"]), _scalar(state["ppfd"])
        lai, ambient_co2 = _scalar(state["leaf_area_index"]), _scalar(state["atmospheric_co2_ppm"])
        topt, gas_constant = _scalar(model.topt), _scalar(model.gas_constant)
        eopt, ct1, ct2 = _scalar(model.eopt_isoprene), _scalar(model.ct1_iso), _scalar(model.ct2)
        alpha, cl = _scalar(model.alpha_light), _scalar(model.cl_light)
        co2 = {
            name: _scalar(manager.get(f"biosphere.bvoc.co2_inhibition.{name}"))
            for name in ("ismax", "c_star", "hill_exponent", "intercellular_to_ambient_ratio")
        }
        foliar_factor = _scalar(manager.get("biosphere.bvoc.foliar_density_factor"))
        molecular_weight = _scalar(torch.tensor(model.molecular_weights["isoprene"], dtype=torch.float32))
        carbon_fraction = _scalar(model._get_carbon_fraction("isoprene"))
        # References: Guenther et al. (2012), Eqs. (4), (8), (14); RELIC reduced
        #   dry-mass closure in docs/source/theory/bvoc_emissions.md; BIPM SI Brochure.
        # Mathematical expression: x=(1/Topt-1/T)/R;
        #   gamma_T=Eopt*CT2*exp(CT1*x)/(CT2-CT1*(1-exp(CT2*x)));
        #   gamma_P=alpha*CL*Q/sqrt(1+alpha^2*Q^2); Ci=r*Ca;
        #   gamma_C=ISmax-ISmax*Ci^h/(Cstar^h+Ci^h);
        #   F=epsilon*gamma_T*gamma_P*LAI*kf/th*kug*gamma_C/MW;
        #   FC=F*MW*fC*1e-3.
        # Parameter meanings: T leaf-temperature proxy; Topt optimum temperature;
        #   R molar gas constant; CT1/CT2 activation/deactivation energies;
        #   Eopt optimum activity; Q PPFD; alpha/CL light-response coefficients;
        #   Ca ambient CO2; Ci intercellular proxy; r ratio; ISmax/Cstar/h Hill
        #   coefficients; epsilon emission potential; LAI leaf area index;
        #   kf dry-leaf mass per leaf area; th seconds per hour; kug ug-to-g
        #   conversion; MW molecular weight; fC carbon mass fraction.
        # Units: T/Topt [K]; R [J mol-1 K-1]; CT1/CT2 [J mol-1];
        #   Q [umol photons m-2 s-1]; alpha [m2 s umol-1]; Ca/Ci/Cstar [ppm];
        #   epsilon [ug g-1 DW h-1]; kf [g DW m-2]; th [s h-1]; kug [g ug-1];
        #   MW [g mol-1]; FC [kg C m-2 s-1]; F [mol m-2 s-1]; other factors [1].
        # Domain/constraints: One finite admitted observed-input savedpoint;
        #   fixed history, generic EF, no stress; oracle uses represented float32
        #   inputs/coefficients in scalar double arithmetic and no production law.
        # Sign convention: F and FC are outward nonnegative emission magnitudes.
        x = (1.0 / topt - 1.0 / temperature) / gas_constant
        gamma_t = eopt * ct2 * math.exp(ct1 * x) / (ct2 - ct1 * (1.0 - math.exp(ct2 * x)))
        gamma_p = alpha * cl * ppfd / math.sqrt(1.0 + alpha * alpha * ppfd * ppfd)
        ci_power = (co2["intercellular_to_ambient_ratio"] * ambient_co2) ** co2["hill_exponent"]
        gamma_c = co2["ismax"] - co2["ismax"] * ci_power / (co2["c_star"] ** co2["hill_exponent"] + ci_power)
        expected_species = (
            _scalar(model.isoprene_ef) * gamma_t * gamma_p * lai * foliar_factor
            / _scalar(model.seconds_per_hour) * gamma_c
            * _scalar(model.micrograms_to_grams) / molecular_weight
        )
        expected_carbon = expected_species * molecular_weight * carbon_fraction * 1.0e-3
        assert abs(_scalar(values["isoprene_emission"]) - expected_species) <= 1.0e-12 + 1.0e-6 * abs(expected_species)
        assert abs(_scalar(values["bvoc_carbon_allocation_flux"]) - expected_carbon) <= 1.0e-15 + 1.0e-6 * abs(expected_carbon)
        for value in values.values():
            assert value.dtype is torch.float32
            assert value.device == state["air_temperature"].device
        for key in state:
            assert torch.equal(state[key], saved_inputs[key])

    def test_guenther_2012_equation_14_is_exact(self):
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        manager = get_global_manager()
        ambient = torch.tensor([300.0, 400.0, 800.0], dtype=torch.float32)
        ismax = _scalar(manager.get("biosphere.bvoc.co2_inhibition.ismax"))
        c_star = _scalar(manager.get("biosphere.bvoc.co2_inhibition.c_star"))
        hill = _scalar(manager.get("biosphere.bvoc.co2_inhibition.hill_exponent"))
        ratio = _scalar(
            manager.get("biosphere.bvoc.co2_inhibition.intercellular_to_ambient_ratio")
        )
        ci = ratio * ambient
        expected = ismax - ismax * ci.pow(hill) / (c_star**hill + ci.pow(hill))
        actual = model._compute_co2_inhibition_factor(ambient)
        assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-7)
        assert actual[0] > actual[1] > actual[2]

    def test_co2_factor_float64_uses_declared_float32_parameters_continuously(self):
        """Pin the declared precision contract of Guenther et al. (2012) Eq. (14).

        The four CO2 response keys are declared float32 in the parameter YAML,
        so a float64 evaluation inherits the float32-rounded parameter values
        (ISmax = 1.343999981880188, h = 1.461400032043457, r_Ci =
        0.699999988079071, Cstar = 585 exactly). The float64 path must equal an
        80-digit reference built from those rounded values to 5e-12, and RELIC
        must stay continuous at 400 ppm, where the official MEGAN2.1
        GAMMA_CO2 (gamma_etc.f) returns 1.0 instead of the formula value.
        """
        from decimal import Decimal, getcontext

        getcontext().prec = 80
        manager = get_global_manager()
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        names = ("ismax", "c_star", "hill_exponent", "intercellular_to_ambient_ratio")
        rounded = {}
        for name in names:
            spec = manager.get_parameter(f"biosphere.bvoc.co2_inhibition.{name}")
            assert spec.dtype == "float32"
            held = model.get_parameter_value(
                name, namespace="biosphere.bvoc.co2_inhibition", required=True
            )
            assert held.dtype is torch.float32
            as_float32 = torch.tensor(float(spec.value), dtype=torch.float32).item()
            assert _scalar(held) == as_float32
            rounded[name] = Decimal(as_float32)

        # References: Guenther et al. (2012), Eq. (14); MEGAN2.1 gamma_etc.f GAMMA_CO2.
        # Mathematical expression: Ci=r_Ci*Ca; gamma=ISmax-ISmax*Ci^h/(Cstar^h+Ci^h).
        # Parameter meanings: Ca ambient CO2; r_Ci intercellular ratio; h Hill
        #   exponent; Cstar response scale; ISmax maximum activity factor.
        # Units: Ca,Ci,Cstar [ppm]; ISmax,r_Ci,h,gamma [1].
        # Domain/constraints: Ca >= 0; Ci^h is 0 at Ca = 0; 80 decimal digits.
        # Sign convention: gamma decreases with increasing Ca.
        def reference(ca: float) -> Decimal:
            ci = rounded["intercellular_to_ambient_ratio"] * Decimal(ca)
            if ci == 0:
                return rounded["ismax"]
            h = rounded["hill_exponent"]
            ci_power = (h * ci.ln()).exp()
            c_star_power = (h * rounded["c_star"].ln()).exp()
            return rounded["ismax"] - rounded["ismax"] * ci_power / (c_star_power + ci_power)

        ambient = [0.0, 280.0, 400.0, 585.0, 1000.0]
        actual = model._compute_co2_inhibition_factor(
            torch.tensor(ambient, dtype=torch.float64)
        )
        assert actual.dtype is torch.float64
        for ca, value in zip(ambient, actual.tolist()):
            expected = reference(ca)
            assert abs(Decimal(value) - expected) <= Decimal("5e-12") * abs(expected), ca

        # No CO2 == 400 ppm special case: the factor at 400 ppm is the formula
        # value (about 1.0024714), and neighbours 1e-6 ppm away differ only by
        # the local slope (|d gamma / d Ca| < 1e-3 ppm-1 here).
        near = model._compute_co2_inhibition_factor(
            torch.tensor([400.0 - 1.0e-6, 400.0, 400.0 + 1.0e-6], dtype=torch.float64)
        ).detach()
        assert abs(near[1].item() - 1.0) > 2.0e-3
        assert torch.all(near[:-1] > near[1:])
        assert float((near[1:] - near[:-1]).abs().max()) < 1.0e-8

    def test_lai_scaling_is_linear(self):
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        low = _forward_values(model, _state(leaf_area_index=1.0))
        high = _forward_values(model, _state(leaf_area_index=2.0))
        for compound in SPECIES:
            ratio = high[f"{compound}_emission"] / low[f"{compound}_emission"]
            assert ratio.item() == pytest.approx(2.0, rel=1e-6)

    def test_temperature_and_light_response_shapes_are_physical(self):
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        temperatures = torch.tensor([283.15, 298.15, 313.15, 328.15])
        isoprene_t = model._get_temperature_factor("isoprene", temperatures)
        monoterpene_t = model._get_temperature_factor("monoterpenes", temperatures)
        assert isoprene_t[0] < isoprene_t[1] < isoprene_t[2]
        assert isoprene_t[2] > isoprene_t[3]
        assert torch.all(monoterpene_t[1:] > monoterpene_t[:-1])
        ppfd = torch.tensor([0.0, 100.0, 500.0, 1000.0, 2000.0])
        isoprene_p = model._get_light_factor("isoprene", ppfd)
        assert torch.all(isoprene_p[1:] > isoprene_p[:-1])

    def test_activity_laws_are_the_shared_module_functions(self):
        """MEGANModel evaluates Guenther et al. (2012) Eqs. (4) and (8) through the
        module-level laws that the BVOC response helper reuses (the Eq. (7) blend of
        the partly light-dependent classes is tested below)."""
        from relic.scientific.biosphere.bvoc.megan_model import (
            megan_ldf_temperature_activity,
            megan_light_dependent_activity,
        )

        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        temperatures = torch.tensor([283.15, 298.15, 313.15, 328.15], dtype=torch.float32)
        ppfd = torch.tensor([0.0, 100.0, 500.0, 1000.0, 2000.0], dtype=torch.float32)
        assert torch.allclose(
            model._get_light_factor("isoprene", ppfd),
            megan_light_dependent_activity(ppfd, model.alpha_light, model.cl_light),
            rtol=1e-7,
            atol=0.0,
        )
        assert torch.allclose(
            model._get_temperature_factor("isoprene", temperatures),
            megan_ldf_temperature_activity(
                temperatures,
                model.eopt_isoprene,
                model.topt,
                model.ct1_iso,
                model.ct2,
                model.gas_constant,
            ),
            rtol=1e-7,
            atol=0.0,
        )

    def test_eq11_law_documents_the_standard_temperature_pairing(self):
        """The Eq. (11) law states that T_s must share the reference temperature of
        the emission factors it multiplies (RELIC 303.15 K dry-mass factors versus
        the MEGAN2.1 T_s = 297 K with its own area factors)."""
        import inspect

        from relic.scientific.biosphere.bvoc.megan_model import (
            megan_lif_temperature_activity,
        )

        source = inspect.getsource(megan_lif_temperature_activity)
        domain = source.split("# Domain/constraints:", 1)[1].split("# Sign convention:", 1)[0]
        for fragment in ("303.15 K", "297 K", "reference temperature", "must not be mixed"):
            assert fragment in domain, fragment

    def test_temperature_activity_is_the_megan21_eq7_blend_of_one_table4_row(self):
        """Guenther et al. (2012) Eq. (7): gamma_T,i = (1 - LDF_i) gamma_T,LIF,i
        + LDF_i gamma_T,LDF,i, with beta_i, LDF_i, C_T1,i and C_eo,i (E_opt at the
        fixed standard history) from one Table 4 row: Isoprene (LDF 1, C_T1 95 kJ
        mol-1, C_eo 2), Other Monoterpenes (beta 0.1 K-1, LDF 0.4, C_T1 80, C_eo 1.83)
        and Other Sesquiterpenes (beta 0.17, LDF 0.5, C_T1 130, C_eo 2.37). T_opt and
        C_T2 are class independent. MEGANModel evaluates the shared law."""
        from relic.scientific.biosphere.bvoc.megan_model import (
            megan_ldf_temperature_activity,
            megan_lif_temperature_activity,
            megan_temperature_activity,
        )

        manager = get_global_manager()
        rows = {
            "isoprene": (None, None, 95000.0, 2.0),
            "monoterpenes": (0.4, 0.1, 80000.0, 1.83),
            "sesquiterpenes": (0.5, 0.17, 130000.0, 2.37),
        }
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        temperatures = torch.tensor(
            [288.15, 298.15, 303.15, 308.15, 313.15], dtype=torch.float64
        )

        def like(value):
            return torch.as_tensor(value, dtype=temperatures.dtype)

        topt, ct2 = like(model.topt), like(model.ct2)
        gas_constant, standard = like(model.gas_constant), like(model.temp_std)
        for compound, (ldf, beta, ct1, ceo) in rows.items():
            namespace = "biosphere.bvoc.megan"
            assert _scalar(manager.get(f"{namespace}.ct1_{compound}")) == pytest.approx(ct1)
            assert _scalar(manager.get(f"{namespace}.eopt_{compound}")) == pytest.approx(ceo)
            if ldf is not None:
                assert _scalar(manager.get(f"{namespace}.ldf_{compound}")) == pytest.approx(ldf)
                assert _scalar(manager.get(f"{namespace}.beta_{compound}")) == pytest.approx(beta)
            light_dependent = megan_ldf_temperature_activity(
                temperatures, like(ceo), topt, like(ct1), ct2, gas_constant
            )
            if ldf is None:
                expected = light_dependent
            else:
                expected = (1.0 - ldf) * megan_lif_temperature_activity(
                    temperatures, like(beta), standard
                ) + ldf * light_dependent
            shared = megan_temperature_activity(
                temperatures,
                None if ldf is None else like(ldf),
                None if beta is None else like(beta),
                standard,
                like(ceo),
                topt,
                like(ct1),
                ct2,
                gas_constant,
            )
            assert torch.allclose(shared, expected, rtol=1e-12, atol=0.0), compound
            executed = model._get_temperature_factor(compound, temperatures)
            assert torch.allclose(
                executed.to(torch.float64), expected, rtol=1e-6, atol=0.0
            ), compound

    def test_table4_temperature_parameters_are_declared(self):
        """The Eq. (8) parameters of the monoterpene and sesquiterpene rows are
        declared in metadata and certificate next to the class-independent T_opt
        and C_T2."""
        from relic.computation.operators.iphs_certificate import (
            load_iphs_equivalence_certificate,
        )

        keys = {
            f"biosphere.bvoc.megan.{name}_{compound}"
            for name in ("ct1", "eopt")
            for compound in ("monoterpenes", "sesquiterpenes")
        }
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        assert keys <= set(model.metadata.parameters)
        certificate = load_iphs_equivalence_certificate(
            PROJECT_ROOT
            / "docs"
            / "source"
            / "theory"
            / "iphs_certificates"
            / "biosphere_bvoc_megan_model_megan_model.yaml"
        )
        assert certificate.validate()["passed"] is True
        assert keys <= set(certificate.parameter_keys)
        scope = certificate.runtime_replacement_map["scientific_scope"]
        assert "Eq. (7)" not in scope

    def test_species_carbon_stoichiometry_is_self_consistent(self):
        """For every species the carbon mass fraction equals n_C A_r(C) / MW, so the
        molar port, the carbon number and the carbon allocation proxy agree."""
        from relic.utils.physical_constants import get_physical_constant

        carbon = float(get_physical_constant("relic.physical.constants.molecular_weights.carbon"))
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        for compound in SPECIES:
            ratio = (
                model.molecular_weights[compound]
                * float(model._get_carbon_fraction(compound))
                / (model.carbon_atoms[compound] * carbon)
            )
            assert ratio == pytest.approx(1.0, abs=1.0e-3), compound

    def test_photosynthesis_pathway_reports_the_allocated_carbon(self):
        """In the photosynthesis-coupled closure the reported carbon of each species
        equals the allocated photosynthate carbon A f_i gamma_T,i (times gamma_CO2
        for isoprene); the former OVOC set (100 g mol-1, 6 C, f_C 0.6) returned 0.833."""
        from relic.utils.physical_constants import get_physical_constant

        carbon = float(get_physical_constant("relic.physical.constants.molecular_weights.carbon"))
        model = MEGANModel(emission_pathway=EmissionPathway.PHOTOSYNTHESIS)
        state = _state(net_photosynthesis=12.0)
        values = _forward_values(model, state)
        for compound in SPECIES:
            allocated_mol_c = (
                state["net_photosynthesis"]
                * getattr(model, f"f_carbon_{compound}")
                * model._get_temperature_factor(compound, state["air_temperature"])
                * 1.0e-6
            )
            if compound == "isoprene":
                allocated_mol_c = allocated_mol_c * model._compute_co2_inhibition_factor(
                    state["atmospheric_co2_ppm"]
                )
            reported_g_c = (
                values[f"{compound}_emission"]
                * model.molecular_weights[compound]
                * model._get_carbon_fraction(compound)
            )
            ratio = (reported_g_c / (allocated_mol_c * carbon)).reshape(-1)[0].item()
            assert ratio == pytest.approx(1.0, abs=1.0e-3), compound

    def test_ovoc_activities_are_the_emission_weighted_component_mixture(self):
        """The lumped OVOC class is methanol, acetone and acetaldehyde (the bidirectional
        VOC class of Guenther et al. 2012 Table 1). Each activity factor is the
        emission-weighted mixture sum_j w_j gamma_j of the component factors, w_j the
        Table 6 global emission shares (99.6, 43.7 and 20.7 Tg yr-1) and every
        coefficient of component j from its own Table 4 row: Methanol (beta 0.08 K-1,
        LDF 0.8, C_T1 60 kJ mol-1, C_eo 1.6), Acetone (0.1, 0.2, 80, 1.83) and
        Bidirectional VOC (0.13, 0.8, 95, 2). No lumped beta, LDF, C_T1 or C_eo is
        registered."""
        from relic.scientific.biosphere.bvoc.megan_model import (
            megan_light_activity,
            megan_temperature_activity,
        )

        manager = get_global_manager()
        for retired in ("biosphere.bvoc.megan.ldf_ovocs", "biosphere.bvoc.megan.beta_ovocs"):
            assert manager.get_parameter(retired) is None
        components = {
            "methanol": (9.96e13, 0.08, 0.8, 60000.0, 1.6),
            "acetone": (4.37e13, 0.1, 0.2, 80000.0, 1.83),
            "acetaldehyde": (2.07e13, 0.13, 0.8, 95000.0, 2.0),
        }
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        temperatures = torch.tensor([283.15, 293.15, 303.15, 313.15], dtype=torch.float64)
        ppfd = torch.tensor([0.0, 500.0, 1500.0], dtype=torch.float64)

        def like(value, reference):
            return torch.as_tensor(value, dtype=reference.dtype)

        total = sum(entry[0] for entry in components.values())
        expected_t = torch.zeros_like(temperatures)
        expected_p = torch.zeros_like(ppfd)
        for name, (emission, beta, ldf, ct1, ceo) in components.items():
            prefix = f"biosphere.bvoc.megan.ovoc.{name}"
            assert _scalar(manager.get(f"{prefix}.global_emission")) == pytest.approx(emission)
            assert _scalar(manager.get(f"{prefix}.beta")) == pytest.approx(beta)
            assert _scalar(manager.get(f"{prefix}.ldf")) == pytest.approx(ldf)
            assert _scalar(manager.get(f"{prefix}.ct1")) == pytest.approx(ct1)
            assert _scalar(manager.get(f"{prefix}.eopt")) == pytest.approx(ceo)
            weight = emission / total
            expected_t = expected_t + weight * megan_temperature_activity(
                temperatures,
                like(ldf, temperatures),
                like(beta, temperatures),
                like(model.temp_std, temperatures),
                like(ceo, temperatures),
                like(model.topt, temperatures),
                like(ct1, temperatures),
                like(model.ct2, temperatures),
                like(model.gas_constant, temperatures),
            )
            expected_p = expected_p + weight * megan_light_activity(
                ppfd, like(model.alpha_light, ppfd), like(model.cl_light, ppfd), like(ldf, ppfd)
            )
        assert torch.allclose(
            model._get_temperature_factor("ovocs", temperatures).to(torch.float64),
            expected_t,
            rtol=1e-6,
            atol=0.0,
        )
        light = model._get_light_factor("ovocs", ppfd).to(torch.float64)
        assert torch.allclose(light, expected_p, rtol=1e-6, atol=0.0)
        # In darkness only the light-independent shares remain: 1 - sum_j w_j LDF_j.
        dark = 1.0 - sum(entry[0] / total * entry[2] for entry in components.values())
        assert light[0].item() == pytest.approx(dark, rel=1e-6)

    def test_fixed_standard_history_of_eq8_is_stated(self):
        """Guenther et al. (2012) Eqs. (9)-(10): T_opt = 313 + 0.6 (T240 - T_s) and
        E_opt = C_eo exp(0.05 (T24 - T_s)) exp(0.05 (T240 - T_s)). RELIC carries no
        24 h or 240 h leaf-temperature history, so T_opt and E_opt are fixed at the
        standard history T24 = T240 = T_s = 297 K (E_opt = C_eo, T_opt = 313 K); the
        simplification is stated in the metadata and in the certificate."""
        from relic.computation.operators.iphs_certificate import (
            load_iphs_equivalence_certificate,
        )

        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        for text in (
            model.metadata.description,
            model.metadata.parameters["biosphere.bvoc.megan.eopt_isoprene"],
        ):
            assert "T24 = T240" in text and "standard history" in text
        certificate = load_iphs_equivalence_certificate(
            PROJECT_ROOT
            / "docs"
            / "source"
            / "theory"
            / "iphs_certificates"
            / "biosphere_bvoc_megan_model_megan_model.yaml"
        )
        scope = certificate.runtime_replacement_map["scientific_scope"]
        assert "T24 = T240 = T_s = 297 K" in scope and "E_opt = C_eo" in scope

    def test_ovoc_factor_mixing_approximation_is_documented(self):
        """E_OVOC multiplies the component-mixed light and temperature factors instead of
        mixing the component products sum_j w_j gamma_P,j gamma_T,j; the neglected
        covariance term and its size are stated in the metadata and the certificate."""
        from relic.computation.operators.iphs_certificate import (
            load_iphs_equivalence_certificate,
        )

        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        # The runtime mathematical_expression is the framework's canonical IPHS text; the
        # closure states the approximation in its description.
        description = model.metadata.description
        assert "sum_j w_j" in description and "covariance" in description
        certificate = load_iphs_equivalence_certificate(
            PROJECT_ROOT
            / "docs"
            / "source"
            / "theory"
            / "iphs_certificates"
            / "biosphere_bvoc_megan_model_megan_model.yaml"
        )
        statement = str(certificate.runtime_replacement_map["lumped_ovoc_mixture"])
        for fragment in ("sum_j w_j", "covariance", "Table 6", "Table 4"):
            assert fragment in statement, fragment

    def test_non_isoprene_light_activity_follows_megan21_eq3(self):
        """Monoterpenes and sesquiterpenes use Guenther et al. (2012) Eq. (3) with the
        registered light-dependent fraction (0.4, the representative 'Other
        Monoterpenes' value, and 0.5 for every sesquiterpene class) and the shared
        Eq. (4) law, so the sesquiterpene light activity equals the one that
        SesquiterpeneSynthesis executes through the response helper."""
        from relic.scientific.biosphere.bvoc.megan_model import (
            megan_light_activity,
            megan_light_dependent_activity,
        )
        from relic.scientific.biosphere.bvoc.responses import BVOCLightResponse

        manager = get_global_manager()
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        ppfd = torch.tensor([0.0, 100.0, 500.0, 1000.0, 1500.0], dtype=torch.float32)
        gamma_ldf = megan_light_dependent_activity(ppfd, model.alpha_light, model.cl_light)
        for compound, key, expected_ldf in (
            ("monoterpenes", "biosphere.bvoc.megan.ldf_monoterpenes", 0.4),
            ("sesquiterpenes", "biosphere.bvoc.megan.ldf_sesquiterpenes", 0.5),
        ):
            ldf = _scalar(manager.get(key))
            assert ldf == pytest.approx(expected_ldf)
            gamma = model._get_light_factor(compound, ppfd)
            assert torch.allclose(gamma, (1.0 - ldf) + ldf * gamma_ldf, rtol=1e-6, atol=0.0)
            assert gamma[0].item() == pytest.approx(1.0 - ldf)
            assert torch.allclose(
                gamma,
                megan_light_activity(
                    ppfd,
                    model.alpha_light,
                    model.cl_light,
                    torch.tensor(ldf, dtype=torch.float32),
                ),
                rtol=1e-7,
                atol=0.0,
            )
        assert torch.allclose(
            model._get_light_factor("sesquiterpenes", ppfd),
            BVOCLightResponse.get_compound_light_response("sesquiterpenes", ppfd),
            rtol=1e-6,
            atol=0.0,
        )

    def test_light_dependent_fractions_are_declared_parameters(self):
        """The executed LDF parameters are declared in metadata and certificate."""
        from relic.computation.operators.iphs_certificate import (
            load_iphs_equivalence_certificate,
        )

        keys = {
            "biosphere.bvoc.megan.ldf_monoterpenes",
            "biosphere.bvoc.megan.ldf_sesquiterpenes",
            "biosphere.bvoc.megan.ovoc.methanol.ldf",
            "biosphere.bvoc.megan.ovoc.acetone.ldf",
            "biosphere.bvoc.megan.ovoc.acetaldehyde.ldf",
        }
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        assert keys <= set(model.metadata.parameters)
        certificate = load_iphs_equivalence_certificate(
            PROJECT_ROOT
            / "docs"
            / "source"
            / "theory"
            / "iphs_certificates"
            / "biosphere_bvoc_megan_model_megan_model.yaml"
        )
        assert certificate.validate()["passed"] is True
        assert keys <= set(certificate.parameter_keys)

    def test_photosynthesis_path_scales_with_required_net_assimilation(self):
        model = MEGANModel(emission_pathway=EmissionPathway.PHOTOSYNTHESIS)
        low = _forward_values(model, _state(net_photosynthesis=10.0))
        high = _forward_values(model, _state(net_photosynthesis=20.0))
        for compound in SPECIES:
            ratio = high[f"{compound}_emission"] / low[f"{compound}_emission"]
            assert ratio.item() == pytest.approx(2.0, rel=1e-6)

    @pytest.mark.parametrize("net_photosynthesis", [-2.0, 0.0, 10.0])
    def test_photosynthesis_carbon_floor_residual_is_explicit(
        self,
        net_photosynthesis,
    ):
        model = MEGANModel(emission_pathway=EmissionPathway.PHOTOSYNTHESIS)
        values = _forward_values(
            model,
            _state(net_photosynthesis=net_photosynthesis),
        )
        key = "diagnostics.bvoc.megan.photosynthesis_carbon_floor_residual"
        floor = _scalar(
            get_global_manager().get("validation.numerical.carbon_flux_min")
        )
        expected = max(net_photosynthesis, floor) - net_photosynthesis

        assert key in model.metadata.output_variables
        assert values[key].item() == pytest.approx(expected, abs=1.0e-12)
        if net_photosynthesis <= floor:
            for compound in SPECIES:
                assert values[f"{compound}_emission"].item() == 0.0

    def test_standard_path_does_not_claim_photosynthesis_floor_diagnostic(self):
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        key = "diagnostics.bvoc.megan.photosynthesis_carbon_floor_residual"

        assert key not in model.metadata.output_variables
        assert key not in model.forward(_state(), time=0.0, dt=60.0)

    def test_hybrid_is_declared_weighted_combination(self):
        state = _state(net_photosynthesis=20.0)
        standard = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        photosynthesis = MEGANModel(emission_pathway=EmissionPathway.PHOTOSYNTHESIS)
        hybrid = MEGANModel(emission_pathway=EmissionPathway.HYBRID)
        standard_values = _forward_values(standard, state)
        photo_values = _forward_values(photosynthesis, state)
        hybrid_values = _forward_values(hybrid, state)
        weight = hybrid.hybrid_coupling_weight
        for compound in SPECIES:
            key = f"{compound}_emission"
            expected = weight * photo_values[key] + (1.0 - weight) * standard_values[key]
            assert torch.allclose(hybrid_values[key], expected, rtol=1e-6, atol=1e-12)

    def test_stress_product_is_explicit(self):
        model = MEGANModel(
            config={"use_stress_functors": True},
            emission_pathway=EmissionPathway.STANDARD,
        )
        unstressed_model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        stressed = _forward_values(model, _state(stress=True))
        unstressed = _forward_values(unstressed_model, _state())
        expected_factor = 0.8 * 0.9 * 0.95
        diagnostic = stressed["diagnostics.bvoc.megan.environmental_stress_factor"]
        assert diagnostic.item() == pytest.approx(expected_factor, rel=1e-6)
        for compound in SPECIES:
            ratio = stressed[f"{compound}_emission"] / unstressed[f"{compound}_emission"]
            assert ratio.item() == pytest.approx(expected_factor, rel=1e-6)

    def test_stress_product_covers_only_the_declared_factors(self):
        # References: Guenther et al. (2012), reduced closure of megan_model.py.
        # Mathematical expression: E_i = E_0,i prod_{k in S} f_k for the declared
        #   set S of stress controls; f_stress is that product.
        # Parameter meanings: E_0,i the unstressed emission of compound i, f_k a
        #   declared control (water 0.8, temperature 0.9 here).
        # Units: E_i, E_0,i [mol m-2 s-1]; f_k, f_stress [1].
        # Domain/constraints: f_k in [0, 1]; the ozone control of the state
        #   (0.95) is outside S and is no term of the product.
        # Sign convention: emission is positive upward.
        model = MEGANModel(
            config={
                "use_stress_functors": True,
                "stress_factors": ["water", "temperature"],
            },
            emission_pathway=EmissionPathway.STANDARD,
        )
        unstressed_model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        stressed = _forward_values(model, _state(stress=True))
        unstressed = _forward_values(unstressed_model, _state())
        expected_factor = 0.8 * 0.9
        diagnostic = stressed["diagnostics.bvoc.megan.environmental_stress_factor"]
        assert diagnostic.item() == pytest.approx(expected_factor, rel=1e-6)
        for compound in SPECIES:
            ratio = stressed[f"{compound}_emission"] / unstressed[f"{compound}_emission"]
            assert ratio.item() == pytest.approx(expected_factor, rel=1e-6)

    def test_fluxes_are_instantaneous_and_dt_independent(self):
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        one = _forward_values(model, _state(), dt=1.0)
        day = _forward_values(model, _state(), dt=86400.0)
        for key in one:
            assert torch.equal(one[key], day[key])


class TestMEGANValidation:
    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("air_temperature", float("nan")),
            ("air_temperature", float("inf")),
            ("ppfd", -1.0),
            ("ppfd", float("nan")),
            ("leaf_area_index", -1.0),
            ("leaf_area_index", float("inf")),
            ("atmospheric_co2_ppm", 0.0),
            ("atmospheric_co2_ppm", float("nan")),
        ],
    )
    def test_invalid_required_inputs_fail_fast(self, key, value):
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        state = _state()
        state[key] = torch.tensor(value)
        with pytest.raises(ValueError, match=key):
            model.forward(state, time=0.0, dt=60.0)

    def test_nonfinite_required_net_photosynthesis_fails_fast(self):
        model = MEGANModel(emission_pathway=EmissionPathway.HYBRID)
        with pytest.raises(ValueError, match="net_photosynthesis"):
            model.forward(
                _state(net_photosynthesis=float("nan")), time=0.0, dt=60.0
            )

    def test_parameter_sensitivity_uses_canonical_forward(self):
        """The base BiologicalProcess sensitivity (governed call, member replaced through
        functional_call) is the one path; MEGANModel keeps no environment override."""
        assert "parameter_sensitivity" not in MEGANModel.__dict__
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        # temp_std is a non-learnable buffer; the learnable light coefficient is the
        # held manager binding and is refused here (manager calibration path).
        result = model.parameter_sensitivity(
            "temp_std", _state(), time=0.0, dt=60.0, target_output="total_bvoc_emission"
        )
        assert torch.isfinite(result).all()
        assert torch.any(result != 0.0)
        with pytest.raises(TypeError, match="calibration path"):
            model.parameter_sensitivity(
                "alpha_light", _state(), time=0.0, dt=60.0, target_output="total_bvoc_emission"
            )


class TestMEGANArchitecture:
    def test_biological_process_bridge_contract(self):
        model = MEGANModel()
        assert isinstance(model, BiologicalProcess)
        assert model.process_name == model.name
        assert callable(model.forward)

    def test_reduced_closure_metadata_is_evidence_bounded(self):
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        expression = model.metadata.mathematical_expression
        method_contract = MEGANModel._compute_standard_emission.__doc__ or ""

        assert "reduced instantaneous" in model.metadata.description.lower()
        assert expression.startswith("Canonical IPHS:")
        assert "reduced RELIC dry-mass-normalized" in method_contract
        assert "rho_f = LAI * k_f" in method_contract
        assert "/ seconds_per_hour" in method_contract
        assert not hasattr(model, "PHASE8_RUNTIME_REPLACEMENT_STATUS")

    def test_module_exports_are_minimal_and_stable(self):
        from relic.scientific.biosphere.bvoc import megan_model

        assert megan_model.__all__ == [
            "MEGANModel",
            "CouplingStrategy",
            "EmissionPathway",
        ]

    def test_package_contract_bounds_owner_and_public_units(self):
        import relic.scientific.biosphere.bvoc as bvoc

        package_contract = bvoc.__doc__ or ""
        assert "reduced instantaneous" in package_contract.lower()
        assert "ContractRegistry" in package_contract
        assert "leaf_area_index" in package_contract
        assert "mol m-2 s-1" in package_contract
        assert "full MEGAN2.1 canopy-history algorithm remains [N]" in package_contract

        forbidden_claims = (
            "Authoritative MEGAN",
            "single source of truth for BVOC emission calculations",
            "All species-specific modules -> unified MEGANModel",
            "enable_gpu",
            "'foliar_density'",
            "Traditional MEGAN",
        )
        assert not any(claim in package_contract for claim in forbidden_claims)

def _ovoc_covariance(model, ppfd_value, celsius):
    """Relative error of the lumped OVOC product of mixed factors against the mixture of
    component products, with the executed laws and coefficients (float64)."""
    from relic.scientific.biosphere.bvoc import megan_model as mm

    like = torch.zeros((), dtype=torch.float64)

    def lookup(key, units):
        return model._table4_value(key, units, like)

    def as64(name):
        return getattr(model, name).to(torch.float64)

    rows = mm.MEGAN_TABLE4_ROWS["ovocs"]
    emissions = [lookup(row.global_emission, "g yr-1") for row in rows]
    weights = [emission / sum(emissions) for emission in emissions]
    ppfd = torch.tensor(float(ppfd_value), dtype=torch.float64)
    temperature = torch.tensor(273.15 + celsius, dtype=torch.float64)
    light = [
        mm.megan_light_activity(
            ppfd, as64("alpha_light"), as64("cl_light"), lookup(row.ldf, "dimensionless")
        )
        for row in rows
    ]
    heat = [
        mm.megan_temperature_activity(
            temperature,
            lookup(row.ldf, "dimensionless"),
            lookup(row.beta, "K-1"),
            as64("temp_std"),
            lookup(row.eopt, "dimensionless"),
            as64("topt"),
            lookup(row.ct1, "J mol-1"),
            as64("ct2"),
            as64("gas_constant"),
        )
        for row in rows
    ]
    mixed = sum(w * p for w, p in zip(weights, light)) * sum(w * t for w, t in zip(weights, heat))
    exact = sum(w * p * t for w, p, t in zip(weights, light, heat))
    return float((mixed / exact - 1.0).detach())


class TestMEGANFormulaNames:
    """12.5, 12.7, 12.8: names, values and statements follow the executed MEGAN2.1 laws."""

    def test_no_bridge_temperature_label_is_passed(self):
        """12.16: no algorithm reads the BiologicalProcess temperature_response or
        ph_dependence labels; the executed law is the per-class Eq. (7) activity, so
        MEGANModel passes neither and the class-free temperature_function refuses."""
        import inspect

        from relic.scientific.biosphere.bvoc import megan_model

        source = inspect.getsource(megan_model)
        assert "temperature_response=" not in source
        assert "ph_dependence=" not in source
        with pytest.raises(RuntimeError):
            MEGANModel().temperature_function(torch.tensor([300.0]))

    def test_class_independent_eq8_parameters_carry_formula_names(self):
        model = MEGANModel()
        keys = set(model.metadata.parameters)
        assert {"biosphere.bvoc.megan.topt", "biosphere.bvoc.megan.ct2"} <= keys
        # The former isoprene-suffixed names of the class-independent Eq. (8) parameters
        # are gone, with no alias.
        retired_keys = {'biosphere.bvoc.megan.topt_isoprene', 'biosphere.bvoc.megan.ct2_isoprene'}
        assert not retired_keys & keys
        # Both are learnable database coefficients, held as the manager bindings of
        # their formula-named keys.
        assert model.manager_binding_key(model.topt) == "biosphere.bvoc.megan.topt"
        assert model.manager_binding_key(model.ct2) == "biosphere.bvoc.megan.ct2"
        assert not hasattr(model, "topt" + "_iso") and not hasattr(model, "ct2" + "_iso")
        manager = get_global_manager()
        for key in retired_keys:
            assert manager.get(key, required=False) is None

    def test_topt_is_eq9_at_the_standard_history(self):
        """T_opt = 313 + 0.6 (T240 - 297) = 313 K at T240 = T_s = 297 K (G12 Eq. 9)."""
        manager = get_global_manager()
        assert float(manager.get("biosphere.bvoc.megan.topt", units="K")) == 313.0
        model = MEGANModel()
        eopt = float(manager.get("biosphere.bvoc.megan.eopt_isoprene"))
        gamma = model._get_temperature_factor(
            "isoprene", torch.tensor([313.0], dtype=torch.float64)
        )
        assert gamma.item() == pytest.approx(eopt, rel=1.0e-6)

    def test_metadata_quotes_the_executed_ovoc_covariance(self):
        model = MEGANModel()
        description = model.metadata.description
        for celsius in (0, 20, 40):
            value = 100.0 * _ovoc_covariance(model, 0, celsius)
            assert f"{value:+.1f}% at {celsius} degC" in description, (celsius, value)
        temperatures = [0.5 * step for step in range(0, 81)]
        worst_500 = max(
            abs(_ovoc_covariance(model, ppfd, celsius))
            for ppfd in range(500, 2001, 100)
            for celsius in temperatures
        )
        assert f"at most {100.0 * worst_500:.2f}%" in description
        worst_750 = max(
            abs(_ovoc_covariance(model, ppfd, celsius))
            for ppfd in range(750, 2001, 125)
            for celsius in temperatures
        )
        assert worst_750 <= 0.002
        assert "below 0.2% at PPFD >= 500" not in description
        # The 0.2% bound holds from 750 to 2000 umol m-2 s-1 only; the error keeps
        # rising with PPFD (0.205% at 2200), so the statement quotes 2500.
        worst_2500 = max(
            abs(_ovoc_covariance(model, 2500, celsius)) for celsius in temperatures
        )
        assert worst_2500 > 0.002
        bounded = (
            "at most 0.2% for PPFD 750-2000 umol m-2 s-1 "
            f"({100.0 * worst_2500:.2f}% at 2500)"
        )
        assert bounded in description
        assert "PPFD >= 750" not in description
        from relic.computation.operators.iphs_certificate import (
            load_iphs_equivalence_certificate,
        )

        certificate = load_iphs_equivalence_certificate(
            PROJECT_ROOT
            / "docs"
            / "source"
            / "theory"
            / "iphs_certificates"
            / "biosphere_bvoc_megan_model_megan_model.yaml"
        )
        statement = str(certificate.runtime_replacement_map["lumped_ovoc_mixture"])
        assert bounded in statement and "PPFD >= 750" not in statement

    @pytest.mark.parametrize(
        "pathway",
        [
            EmissionPathway.STANDARD,
            EmissionPathway.PHOTOSYNTHESIS,
            EmissionPathway.HYBRID,
        ],
    )
    def test_governed_forward_preserves_float64_across_pathways_and_stress_factors(
        self,
        pathway: EmissionPathway,
    ):
        """Governed MEGAN evaluation must preserve float64 canonical inputs and carbon boundary checks."""
        model = MEGANModel(
            config={
                "ppfd_source": "state",
                "use_stress_functors": True,
                "stress_factors": ["water", "temperature", "ozone"],
            },
            emission_pathway=pathway,
        )
        state = {
            key: value.to(dtype=torch.float64)
            for key, value in _state(
                net_photosynthesis=15.0 if pathway != EmissionPathway.STANDARD else None,
                stress=True,
                shape=(2, 2),
            ).items()
        }
        output = model(state, time=0.0, dt=1800.0)
        values = _values(output)
        for key, tensor in values.items():
            assert tensor.dtype is torch.float64, key
            assert tensor.shape == (2, 2), key
            assert tensor.device == model.device, key
        residuals = model.check_conservation_laws(state, output)
        assert residuals["carbon_boundary"] == pytest.approx(0.0, abs=1.0e-18)

    def test_captured_201907_midpoint_witness_in_float64_eliminates_binary32_discontinuity(
        self,
    ):
        """Captured 2019-07-19 parent (106, 28) binary32 midpoint witness resolves within record tolerance in float64."""
        import math
        import struct

        # References: Guenther et al. (2012), Eqs. (4), (8), (14); IEEE 754-2019
        #   roundTiesToEven midpoint rounding; Goldberg (1991), ACM Comput. Surv. 23, 5-48.
        # Mathematical expression: T_ref = 0x1.29498f0000000p+8 (exact binary32 midpoint);
        #   T_prod = 0x1.29498efffffffp+8 (one binary64 step below);
        #   B_J = F(T_J, P_J, LAI_J, Ca) * C_J * dt;
        #   allowed_J = (atol + rtol * |F_ref|) * C_J * dt.
        # Parameter meanings: T_ref/T_prod scalar-fsum and production-remap coarse
        #   temperatures [K]; P_J coarse PPFD [umol photons m-2 s-1]; LAI_J coarse LAI
        #   [m2 m-2]; Ca ambient CO2 [ppm]; C_J parent covered area [m2]; dt day length [s].
        # Units: T [K]; P [umol photons m-2 s-1]; LAI [m2 m-2]; Ca [ppm]; C_J [m2];
        #   dt [s]; B_J, allowed_J [kg C].
        prod_t64 = float.fromhex("0x1.29498efffffffp+8")
        ref_t64 = float.fromhex("0x1.29498f0000000p+8")
        prod_p64 = float.fromhex("0x1.74d1d691a0ecbp+8")
        ref_p64 = float.fromhex("0x1.74d1d691a0ec9p+8")
        prod_l64 = float.fromhex("0x1.af69f03c8192bp+1")
        ref_l64 = float.fromhex("0x1.af69f03c8192dp+1")
        co2_64 = float.fromhex("0x1.98e147ae147aep+8")
        covered_area_m2 = 319710866.819234
        dt_day = 86400.0

        prod_bits32 = struct.unpack(">I", struct.pack(">f", prod_t64))[0]
        ref_bits32 = struct.unpack(">I", struct.pack(">f", ref_t64))[0]
        assert prod_bits32 == 0x4394A4C7
        assert ref_bits32 == 0x4394A4C8

        model = MEGANModel(
            config={"ppfd_source": "state", "enabled_compounds": ["isoprene"]},
            emission_pathway=EmissionPathway.STANDARD,
        )
        state_f32 = {
            "air_temperature": torch.tensor([prod_t64], dtype=torch.float32),
            "ppfd": torch.tensor([prod_p64], dtype=torch.float32),
            "leaf_area_index": torch.tensor([prod_l64], dtype=torch.float32),
            "atmospheric_co2_ppm": torch.tensor([co2_64], dtype=torch.float32),
        }
        state_f64 = {
            "air_temperature": torch.tensor([prod_t64], dtype=torch.float64),
            "ppfd": torch.tensor([prod_p64], dtype=torch.float64),
            "leaf_area_index": torch.tensor([prod_l64], dtype=torch.float64),
            "atmospheric_co2_ppm": torch.tensor([co2_64], dtype=torch.float64),
        }
        out_f32 = _values(model(state_f32, time=0.0, dt=dt_day))
        out_f64 = _values(model(state_f64, time=0.0, dt=dt_day))
        assert out_f64["bvoc_carbon_allocation_flux"].dtype is torch.float64

        manager = get_global_manager()
        topt, gas_constant = _scalar(model.topt), _scalar(model.gas_constant)
        eopt, ct1, ct2 = _scalar(model.eopt_isoprene), _scalar(model.ct1_iso), _scalar(model.ct2)
        alpha, cl = _scalar(model.alpha_light), _scalar(model.cl_light)
        co2 = {
            name: float(torch.tensor(_scalar(manager.get(f"biosphere.bvoc.co2_inhibition.{name}")), dtype=torch.float32).item())
            for name in ("ismax", "c_star", "hill_exponent", "intercellular_to_ambient_ratio")
        }
        foliar_factor = float(torch.tensor(_scalar(manager.get("biosphere.bvoc.foliar_density_factor")), dtype=torch.float32).item())
        molecular_weight = float(torch.tensor(model.molecular_weights["isoprene"], dtype=torch.float32).item())
        carbon_fraction = _scalar(model._get_carbon_fraction("isoprene"))

        def oracle_carbon_flux(temp_val: float, ppfd_val: float, lai_val: float, co2_val: float) -> float:
            x = (1.0 / topt - 1.0 / temp_val) / gas_constant
            gamma_t = eopt * ct2 * math.exp(ct1 * x) / (ct2 - ct1 * (1.0 - math.exp(ct2 * x)))
            gamma_p = alpha * cl * ppfd_val / math.sqrt(1.0 + alpha * alpha * ppfd_val * ppfd_val)
            ci_power = (co2["intercellular_to_ambient_ratio"] * co2_val) ** co2["hill_exponent"]
            gamma_c = co2["ismax"] - co2["ismax"] * ci_power / (co2["c_star"] ** co2["hill_exponent"] + ci_power)
            molar = (
                _scalar(model.isoprene_ef) * gamma_t * gamma_p * lai_val * foliar_factor
                / _scalar(model.seconds_per_hour) * gamma_c
                * _scalar(model.micrograms_to_grams) / molecular_weight
            )
            return molar * molecular_weight * carbon_fraction * 1.0e-3

        ref_flux_f32_quantized = oracle_carbon_flux(
            float(torch.tensor(ref_t64, dtype=torch.float32).item()),
            float(torch.tensor(ref_p64, dtype=torch.float32).item()),
            float(torch.tensor(ref_l64, dtype=torch.float32).item()),
            float(torch.tensor(co2_64, dtype=torch.float32).item()),
        )
        allowed_f32 = (1.0e-15 + 1.0e-6 * abs(ref_flux_f32_quantized)) * covered_area_m2 * dt_day
        saved_f32_amount = float(out_f32["bvoc_carbon_allocation_flux"].item()) * covered_area_m2 * dt_day
        expected_f32_amount = ref_flux_f32_quantized * covered_area_m2 * dt_day
        assert abs(saved_f32_amount - expected_f32_amount) > allowed_f32

        ref_flux_f64 = oracle_carbon_flux(ref_t64, ref_p64, ref_l64, co2_64)
        allowed_f64 = (1.0e-15 + 1.0e-6 * abs(ref_flux_f64)) * covered_area_m2 * dt_day
        saved_f64_amount = float(out_f64["bvoc_carbon_allocation_flux"].item()) * covered_area_m2 * dt_day
        expected_f64_amount = ref_flux_f64 * covered_area_m2 * dt_day
        assert abs(saved_f64_amount - expected_f64_amount) <= allowed_f64

    def test_mismatched_or_nonfloating_input_and_conservation_dtypes_fail_fast(self):
        """Mixed floating input dtypes, integer inputs, and mixed conservation dtypes must raise TypeError."""
        model = MEGANModel(
            config={"ppfd_source": "state", "enabled_compounds": ["isoprene"]},
            emission_pathway=EmissionPathway.STANDARD,
        )
        mixed_state = _state()
        mixed_state["air_temperature"] = mixed_state["air_temperature"].to(dtype=torch.float64)
        with pytest.raises(TypeError, match="dtype"):
            model.forward(mixed_state, time=0.0, dt=60.0)

        int_state = _state()
        int_state["air_temperature"] = torch.tensor(300, dtype=torch.int64)
        with pytest.raises(TypeError, match="dtype"):
            model.forward(int_state, time=0.0, dt=60.0)

        state_f64 = {k: v.to(dtype=torch.float64) for k, v in _state().items()}
        output_f64 = model.forward(state_f64, time=0.0, dt=60.0)
        mixed_output = dict(output_f64)
        mixed_output["bvoc_carbon_allocation_flux"] = output_f64[
            "bvoc_carbon_allocation_flux"
        ].value.to(dtype=torch.float32)
        with pytest.raises(TypeError, match="dtype"):
            model.check_conservation_laws(state_f64, mixed_output)

