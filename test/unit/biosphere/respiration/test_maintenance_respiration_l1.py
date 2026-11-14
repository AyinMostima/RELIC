"""L1 tests for the canonical maintenance-respiration carbon flux."""

from __future__ import annotations

import math
import warnings

import pytest
import torch

from relic.contracts.contract_loader import ContractRegistry
from relic.core.simulator import RELICSimulator
from relic.core.state import StateManager, UpdateItem, UpdateMode
from relic.scientific.biosphere.respiration.maintenance_respiration import (
    MaintenanceRespiration,
)


REQUIRED_INPUTS = [
    "air_temperature",
    "biomass_leaf",
    "biomass_stem",
    "biomass_root",
    "soluble_sugar",
    "soil_water_potential",
]
OPTIONAL_INPUTS = ["growing_temperature"]
OUTPUTS = [
    "maintenance_respiration_total",
    "diagnostics.maintenance_respiration.leaf_flux",
    "diagnostics.maintenance_respiration.stem_flux",
    "diagnostics.maintenance_respiration.root_flux",
    "diagnostics.maintenance_respiration.leaf_rate_coefficient",
    "diagnostics.maintenance_respiration.substrate_factor",
    "diagnostics.maintenance_respiration.water_stress_factor",
    "diagnostics.maintenance_respiration.growth_temperature",
]
FLUX_OUTPUTS = OUTPUTS[:4]


def _state(**overrides: torch.Tensor) -> dict[str, torch.Tensor]:
    state = {
        "air_temperature": torch.tensor(298.15),
        "biomass_leaf": torch.tensor(200.0),
        "biomass_stem": torch.tensor(500.0),
        "biomass_root": torch.tensor(150.0),
        "soluble_sugar": torch.tensor(50.0),
        "soil_water_potential": torch.tensor(-0.5),
        "growing_temperature": torch.tensor(298.15),
    }
    state.update(overrides)
    return state


def _value(output: dict[str, UpdateItem], key: str) -> torch.Tensor:
    item = output[key]
    assert isinstance(item, UpdateItem)
    assert isinstance(item.value, torch.Tensor)
    return item.value


@pytest.fixture
def maintenance_respiration() -> MaintenanceRespiration:
    return MaintenanceRespiration(enable_acclimation=False)


class TestMaintenanceRespirationL1:
    """Verify the single kg-C maintenance-respiration contract."""

    def test_initialization_and_metadata(
        self,
        maintenance_respiration: MaintenanceRespiration,
    ) -> None:
        assert isinstance(maintenance_respiration, torch.nn.Module)
        assert maintenance_respiration.pft_name is None
        assert maintenance_respiration.metadata.input_variables == (
            REQUIRED_INPUTS + OPTIONAL_INPUTS
        )
        assert maintenance_respiration.optional_input_variables == OPTIONAL_INPUTS
        assert maintenance_respiration.get_required_inputs() == REQUIRED_INPUTS
        assert maintenance_respiration.metadata.output_variables == OUTPUTS
        assert maintenance_respiration.get_provides() == OUTPUTS
        assert maintenance_respiration.get_dependencies() == []
        assert not hasattr(maintenance_respiration, "_compute_rate_core")

    def test_forward_exact_contract_and_fixed_numerics(
        self,
        maintenance_respiration: MaintenanceRespiration,
    ) -> None:
        output = maintenance_respiration.forward(_state(), time=0.0, dt=3600.0)

        assert list(output) == OUTPUTS
        assert set(output) == set(maintenance_respiration.metadata.output_variables)
        for key, item in output.items():
            assert isinstance(item, UpdateItem), key
            assert item.mode is UpdateMode.REPLACE
            assert item.per_timestep is False
            assert item.source_functor == "maintenance_respiration"

        expected = {
            "diagnostics.maintenance_respiration.leaf_flux": 2.0666666666666667e-08,
            "diagnostics.maintenance_respiration.stem_flux": 5.166666666666667e-09,
            "diagnostics.maintenance_respiration.root_flux": 1.0330357142857143e-08,
            "maintenance_respiration_total": 3.6163690476190476e-08,
            "diagnostics.maintenance_respiration.leaf_rate_coefficient": 1.736e-07,
            "diagnostics.maintenance_respiration.substrate_factor": 5.0 / 6.0,
            "diagnostics.maintenance_respiration.water_stress_factor": 5.0 / 7.0,
            "diagnostics.maintenance_respiration.growth_temperature": 298.15,
        }
        for key, expected_value in expected.items():
            torch.testing.assert_close(
                _value(output, key),
                torch.tensor(expected_value),
                rtol=2.0e-6,
                atol=1.0e-14,
            )

        component_sum = sum(_value(output, key) for key in FLUX_OUTPUTS[1:])
        torch.testing.assert_close(
            _value(output, "maintenance_respiration_total"),
            component_sum,
            rtol=1.0e-6,
            atol=1.0e-14,
        )

        expected_units = {
            "maintenance_respiration_total": "kg C m-2 s-1",
            "diagnostics.maintenance_respiration.leaf_flux": "kg C m-2 s-1",
            "diagnostics.maintenance_respiration.stem_flux": "kg C m-2 s-1",
            "diagnostics.maintenance_respiration.root_flux": "kg C m-2 s-1",
            "diagnostics.maintenance_respiration.leaf_rate_coefficient": "s-1",
            "diagnostics.maintenance_respiration.substrate_factor": "1",
            "diagnostics.maintenance_respiration.water_stress_factor": "1",
            "diagnostics.maintenance_respiration.growth_temperature": "K",
        }
        assert {key: item.unit for key, item in output.items()} == expected_units

    def test_nonincrementable_flux_commits_through_state_manager(
        self,
        maintenance_respiration: MaintenanceRespiration,
    ) -> None:
        """Route the instantaneous carbon flux without a storage-element hint."""
        output = maintenance_respiration.forward(_state(), time=0.0, dt=3600.0)
        key = "maintenance_respiration_total"
        item = output[key]

        manager = StateManager(device="cpu")
        registry = manager._contract_registry
        assert isinstance(registry, ContractRegistry)
        contract = registry.get_contract(key)
        assert contract is not None
        assert contract.kind == "flux"
        assert contract.incrementable is False
        assert contract.raw["accumulates"] is None

        manager.set_state({key: torch.tensor(0.0)}, validate=False)
        committed = manager.apply_updates(
            {key: item},
            dt=3600.0,
            check_conservation=False,
            update_sources={key: maintenance_respiration.name},
        )

        assert item.element is None
        torch.testing.assert_close(committed[key], item.value, rtol=0.0, atol=0.0)

    def test_strict_simulator_accepts_external_state_without_fake_dependencies(
        self,
    ) -> None:
        """Route external inputs through StateManager and retain one writer."""
        simulator = RELICSimulator(
            {
                "device": "cpu",
                "time_step": 3600.0,
                "max_time": 3600.0,
                "conservation_checks": False,
                "strict_dependencies": True,
                "strict_scale_detection": False,
            }
        )
        simulator.initialize_state(_state())
        simulator.register_functor(
            MaintenanceRespiration(enable_acclimation=False),
            execution_group="fast",
        )

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            updates = simulator.step(dt=3600.0)

        assert captured == []
        actual = updates["maintenance_respiration_total"]
        expected = torch.tensor(
            3.6163690476190476e-08,
            dtype=actual.dtype,
            device=actual.device,
        )
        torch.testing.assert_close(
            updates["maintenance_respiration_total"],
            expected,
            rtol=2.0e-6,
            atol=1.0e-14,
        )
        torch.testing.assert_close(
            simulator.state_manager.get_state("maintenance_respiration_total"),
            expected,
            rtol=2.0e-6,
            atol=1.0e-14,
        )
        writers = {
            item["functor"]
            for item in simulator.get_variable_provenance(
                "maintenance_respiration_total"
            )
            if item.get("role") in {"writer", "final_writer"}
        }
        assert writers == {"maintenance_respiration"}
        assert simulator.get_effective_device("maintenance_respiration").type == "cpu"

    def test_tissue_specific_q10_response(
        self,
        maintenance_respiration: MaintenanceRespiration,
    ) -> None:
        at_25 = maintenance_respiration.forward(_state(), 0.0, 1.0)
        at_35 = maintenance_respiration.forward(
            _state(air_temperature=torch.tensor(308.15)),
            0.0,
            1.0,
        )
        at_15 = maintenance_respiration.forward(
            _state(air_temperature=torch.tensor(288.15)),
            0.0,
            1.0,
        )

        expected_q10 = {
            "diagnostics.maintenance_respiration.leaf_flux": 2.0,
            "diagnostics.maintenance_respiration.stem_flux": 1.8,
            "diagnostics.maintenance_respiration.root_flux": 2.2,
        }
        for key, q10 in expected_q10.items():
            torch.testing.assert_close(
                _value(at_35, key) / _value(at_25, key),
                torch.tensor(q10),
                rtol=2.0e-6,
                atol=1.0e-7,
            )
            torch.testing.assert_close(
                _value(at_15, key) / _value(at_25, key),
                torch.tensor(1.0 / q10),
                rtol=2.0e-6,
                atol=1.0e-7,
            )

    def test_tissue_coefficients_control_component_order(
        self,
        maintenance_respiration: MaintenanceRespiration,
    ) -> None:
        equal_biomass = _state(
            biomass_leaf=torch.tensor(200.0),
            biomass_stem=torch.tensor(200.0),
            biomass_root=torch.tensor(200.0),
        )
        output = maintenance_respiration.forward(equal_biomass, 0.0, 1.0)

        leaf = _value(output, "diagnostics.maintenance_respiration.leaf_flux")
        stem = _value(output, "diagnostics.maintenance_respiration.stem_flux")
        root = _value(output, "diagnostics.maintenance_respiration.root_flux")
        assert leaf > root > stem

    def test_water_stress_has_no_hidden_carbon_floor(
        self,
        maintenance_respiration: MaintenanceRespiration,
    ) -> None:
        unstressed = maintenance_respiration.forward(
            _state(soil_water_potential=torch.tensor(-0.1)),
            0.0,
            1.0,
        )
        moderate = maintenance_respiration.forward(
            _state(soil_water_potential=torch.tensor(-0.8)),
            0.0,
            1.0,
        )
        wilting = maintenance_respiration.forward(
            _state(soil_water_potential=torch.tensor(-1.5)),
            0.0,
            1.0,
        )

        total_key = "maintenance_respiration_total"
        assert _value(unstressed, total_key) > _value(moderate, total_key)
        assert _value(moderate, total_key) > _value(wilting, total_key)
        torch.testing.assert_close(
            _value(wilting, total_key),
            torch.tensor(0.0),
            rtol=0.0,
            atol=0.0,
        )

    def test_substrate_response_and_zero_substrate_closure(
        self,
        maintenance_respiration: MaintenanceRespiration,
    ) -> None:
        high = maintenance_respiration.forward(
            _state(soluble_sugar=torch.tensor(100.0)),
            0.0,
            1.0,
        )
        low = maintenance_respiration.forward(
            _state(soluble_sugar=torch.tensor(1.0)),
            0.0,
            1.0,
        )
        zero = maintenance_respiration.forward(
            _state(soluble_sugar=torch.tensor(0.0)),
            0.0,
            1.0,
        )

        total_key = "maintenance_respiration_total"
        assert _value(high, total_key) > _value(low, total_key)
        assert _value(low, total_key) > _value(zero, total_key)
        torch.testing.assert_close(
            _value(zero, total_key),
            torch.tensor(0.0),
            rtol=0.0,
            atol=0.0,
        )

    def test_explicit_growth_temperature_controls_acclimation(self) -> None:
        model = MaintenanceRespiration(enable_acclimation=True)
        cool = model.forward(
            _state(growing_temperature=torch.tensor(288.15)),
            0.0,
            1.0,
        )
        warm = model.forward(
            _state(growing_temperature=torch.tensor(308.15)),
            0.0,
            1.0,
        )

        ratio = _value(cool, "maintenance_respiration_total") / _value(
            warm,
            "maintenance_respiration_total",
        )
        torch.testing.assert_close(
            ratio,
            torch.tensor(math.exp(1.0)),
            rtol=2.0e-6,
            atol=1.0e-7,
        )

    def test_rate_outputs_are_dt_invariant_and_replace_only(
        self,
        maintenance_respiration: MaintenanceRespiration,
    ) -> None:
        one_second = maintenance_respiration.forward(_state(), 0.0, 1.0)
        one_hour = maintenance_respiration.forward(_state(), 0.0, 3600.0)

        for key in OUTPUTS:
            torch.testing.assert_close(
                _value(one_second, key),
                _value(one_hour, key),
                rtol=0.0,
                atol=0.0,
            )
            assert one_hour[key].mode is UpdateMode.REPLACE
            assert one_hour[key].per_timestep is False

    @pytest.mark.parametrize("missing_key", REQUIRED_INPUTS)
    def test_required_inputs_fail_fast(
        self,
        maintenance_respiration: MaintenanceRespiration,
        missing_key: str,
    ) -> None:
        state = _state()
        state.pop(missing_key)
        with pytest.raises(KeyError, match=missing_key):
            maintenance_respiration.forward(state, 0.0, 1.0)

    @pytest.mark.parametrize(
        ("canonical_key", "legacy_key"),
        [
            ("air_temperature", "temperature"),
            ("biomass_leaf", "leaf_biomass"),
            ("biomass_stem", "stem_biomass"),
            ("biomass_root", "root_biomass"),
            ("soil_water_potential", "water_potential"),
        ],
    )
    def test_legacy_input_aliases_are_rejected(
        self,
        maintenance_respiration: MaintenanceRespiration,
        canonical_key: str,
        legacy_key: str,
    ) -> None:
        state = _state()
        value = state.pop(canonical_key)
        state[legacy_key] = value
        with pytest.raises(KeyError, match=canonical_key):
            maintenance_respiration.forward(state, 0.0, 1.0)

    @pytest.mark.parametrize(
        "key",
        REQUIRED_INPUTS + OPTIONAL_INPUTS,
    )
    def test_non_finite_inputs_fail_fast(
        self,
        maintenance_respiration: MaintenanceRespiration,
        key: str,
    ) -> None:
        with pytest.raises(ValueError, match=key):
            maintenance_respiration.forward(
                _state(**{key: torch.tensor(float("nan"))}),
                0.0,
                1.0,
            )

    @pytest.mark.parametrize(
        "key",
        ["biomass_leaf", "biomass_stem", "biomass_root", "soluble_sugar"],
    )
    def test_negative_carbon_inputs_fail_fast(
        self,
        maintenance_respiration: MaintenanceRespiration,
        key: str,
    ) -> None:
        with pytest.raises(ValueError, match=key):
            maintenance_respiration.forward(
                _state(**{key: torch.tensor(-1.0)}),
                0.0,
                1.0,
            )

    def test_parameter_units_are_loaded_from_yaml(
        self,
        maintenance_respiration: MaintenanceRespiration,
    ) -> None:
        manager = maintenance_respiration.get_parameter_manager()
        expected_units = {
            "rm_leaf_25": "s-1",
            "rm_stem_25": "s-1",
            "rm_root_25": "s-1",
            "q10_respiration": "dimensionless",
            "q10_leaf_factor": "dimensionless",
            "q10_stem_factor": "dimensionless",
            "q10_root_factor": "dimensionless",
            "acclimation_rate_coeff": "K-1",
            "km_substrate": "g C m-2",
            "water_stress_wilting_point": "MPa",
            "water_stress_field_capacity": "MPa",
        }
        assert set(maintenance_respiration.parameter_snapshots) == set(expected_units)
        for name, units in expected_units.items():
            spec = manager.get_parameter(f"biosphere.respiration.maintenance.{name}")
            assert spec is not None
            assert spec.units == units


class TestAcclimationHistoryMemory:
    """G08 memory contract: the acclimation window is replayable hidden state."""

    def test_window_mean_and_oldest_sample_drop(self) -> None:
        model = MaintenanceRespiration(enable_acclimation=True, acclimation_days=3)
        temperatures = [283.15, 293.15, 303.15, 313.15]
        seen = []
        for temperature in temperatures:
            output = model.forward(
                {k: v for k, v in _state().items() if k != "growing_temperature"}
                | {"air_temperature": torch.tensor(temperature)},
                0.0,
                3600.0,
            )
            seen.append(float(_value(output, "diagnostics.maintenance_respiration.growth_temperature")))
        assert seen[0] == pytest.approx(283.15)
        assert seen[1] == pytest.approx((283.15 + 293.15) / 2.0)
        assert seen[2] == pytest.approx((283.15 + 293.15 + 303.15) / 3.0)
        # The fourth sample evicts the oldest one from the three-day window.
        assert seen[3] == pytest.approx((293.15 + 303.15 + 313.15) / 3.0)
        torch.testing.assert_close(
            model.temperature_history,
            torch.tensor([293.15, 303.15, 313.15], dtype=torch.float64),
        )
        extra = model.get_extra_state()
        assert extra["count"] == 3 and extra["index"] == 1 and extra["window_days"] == 3

    def test_state_dict_roundtrip_restores_the_window(self) -> None:
        source = MaintenanceRespiration(enable_acclimation=True, acclimation_days=3)
        for temperature in (283.15, 293.15):
            source.forward(
                {k: v for k, v in _state().items() if k != "growing_temperature"}
                | {"air_temperature": torch.tensor(temperature)},
                0.0,
                3600.0,
            )
        restored = MaintenanceRespiration(enable_acclimation=True, acclimation_days=3)
        restored.load_state_dict(source.state_dict())
        torch.testing.assert_close(restored.temperature_history, source.temperature_history)
        next_state = {k: v for k, v in _state().items() if k != "growing_temperature"} | {
            "air_temperature": torch.tensor(303.15)
        }
        expected = _value(source.forward(next_state, 0.0, 3600.0), "maintenance_respiration_total")
        actual = _value(restored.forward(next_state, 0.0, 3600.0), "maintenance_respiration_total")
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize(
        "mutation, error",
        [
            (lambda s: "not a mapping", TypeError),
            (lambda s: {**s, "window_days": 5}, ValueError),
            (lambda s: {**s, "count": 4}, ValueError),
            (lambda s: {**s, "index": 3}, ValueError),
            (lambda s: {**s, "buffer": torch.zeros(4, dtype=torch.float64)}, TypeError),
            (lambda s: {**s, "format": "other"}, ValueError),
        ],
    )
    def test_invalid_extra_state_is_rejected(self, mutation, error) -> None:
        model = MaintenanceRespiration(enable_acclimation=True, acclimation_days=3)
        snapshot = model.get_extra_state()
        with pytest.raises(error):
            model.set_extra_state(mutation(snapshot))

    def test_governed_checkpoint_restart_continues_the_window(self, tmp_path) -> None:
        temperatures = [278.15, 288.15, 298.15, 308.15, 303.15, 293.15]

        def build():
            simulator = RELICSimulator(
                {
                    "device": "cpu",
                    "time_step": 3600.0,
                    "conservation_checks": False,
                    "strict_dependencies": True,
                    "strict_scale_detection": False,
                }
            )
            simulator.initialize_state(
                {k: v for k, v in _state().items() if k != "growing_temperature"}
            )
            functor = MaintenanceRespiration(enable_acclimation=True, acclimation_days=3)
            simulator.register_functor(functor, execution_group="fast")
            return simulator, functor

        def advance(simulator, temperature):
            simulator.state_manager.set_state(
                {"air_temperature": torch.tensor(temperature)}, validate=False
            )
            outputs = simulator.step(dt=3600.0)
            return float(outputs["diagnostics.maintenance_respiration.growth_temperature"])

        uninterrupted, _ = build()
        reference = [advance(uninterrupted, t) for t in temperatures]
        assert reference[3] == pytest.approx((288.15 + 298.15 + 308.15) / 3.0)

        source, source_functor = build()
        for temperature in temperatures[:3]:
            advance(source, temperature)
        checkpoint = tmp_path / "respiration.pt"
        source.create_checkpoint(checkpoint)
        assert source_functor.get_extra_state()["count"] == 3

        restored, restored_functor = build()
        restored.restore_checkpoint(checkpoint)
        torch.testing.assert_close(
            restored_functor.temperature_history, source_functor.temperature_history
        )
        continued = [advance(restored, t) for t in temperatures[3:]]
        assert continued == pytest.approx(reference[3:])


def test_maintenance_respiration_passes_and_assigns_no_bridge_label():
    """12.16: no algorithm reads the BiologicalProcess temperature_response or
    ph_dependence labels (the tissue Q10 factors call q10_temperature_response directly), so the module neither passes nor assigns them."""
    import inspect

    from relic.scientific.biosphere.respiration import maintenance_respiration

    source = inspect.getsource(maintenance_respiration)
    for label in ("temperature_response", "ph_dependence"):
        assert f"self.{label}" not in source
        assert f"{label}=" not in source
