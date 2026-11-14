"""L2 integration tests for the canonical biosphere growth chain."""

from __future__ import annotations

import pytest
import torch

from relic.core.state import StateManager, UpdateItem, UpdateMode
from relic.parameters import get_global_manager
from relic.scientific.biosphere.growth.biomass_allocation import BiomassAllocation
from relic.scientific.biosphere.growth.phenology_growth import PhenologyGrowth
from relic.scientific.biosphere.growth.structural_growth import StructuralGrowth


ALLOCATION_DIAGNOSTICS = {
    "diagnostics.biomass_allocation.total_growth_rate",
    "diagnostics.biomass_allocation.allocation_fraction_leaf",
    "diagnostics.biomass_allocation.allocation_fraction_stem",
    "diagnostics.biomass_allocation.allocation_fraction_root",
    "diagnostics.biomass_allocation.allocation_sum",
    "diagnostics.biomass_allocation.environmental_modifier",
    "diagnostics.biomass_allocation.phenology_modifier",
    "diagnostics.biomass_allocation.carbon_balance_residual",
}
ALLOCATION_RATE_KEYS = {
    "growth_rate_leaf",
    "growth_rate_stem",
    "growth_rate_root",
}
ALLOCATION_STATE_KEYS = {
    "leaf_biomass_c",
    "wood_biomass_c",
    "root_biomass_c",
}
STRUCTURAL_OUTPUTS = {
    "diagnostics.structural_growth.gross_leaf_area_growth_tendency",
    "diagnostics.structural_growth.leaf_area_senescence_tendency",
    "diagnostics.structural_growth.net_leaf_area_tendency",
    "diagnostics.structural_growth.step_leaf_area_delta",
}


def _value(result: dict[str, UpdateItem], key: str) -> torch.Tensor:
    """Return one explicit UpdateItem payload."""
    item = result[key]
    assert isinstance(item, UpdateItem)
    assert isinstance(item.value, torch.Tensor)
    return item.value


def _allocation_state(
    *,
    device: torch.device = torch.device("cpu"),
    npp: float = 1.0e-6,
) -> dict[str, torch.Tensor]:
    """Build one canonical allocation state in SI-style project units."""
    return {
        "net_primary_productivity": torch.tensor(npp, dtype=torch.float32, device=device),
        "air_temperature": torch.tensor(298.15, dtype=torch.float32, device=device),
        "water_stress_factor": torch.tensor(0.8, dtype=torch.float32, device=device),
        "leaf_area_index": torch.tensor(4.0, dtype=torch.float32, device=device),
        "phenology_phase": torch.tensor(1.0, dtype=torch.float32, device=device),
        "leaf_biomass_c": torch.tensor(0.1, dtype=torch.float32, device=device),
        "wood_biomass_c": torch.tensor(0.5, dtype=torch.float32, device=device),
        "root_biomass_c": torch.tensor(0.2, dtype=torch.float32, device=device),
    }


class TestGrowthModuleIntegration:
    """Verify canonical data flow across growth modules and StateManager."""

    @pytest.fixture
    def state_manager(self) -> StateManager:
        """Create an isolated CPU StateManager."""
        return StateManager(device=torch.device("cpu"))

    @pytest.fixture
    def initial_state(self) -> dict[str, torch.Tensor]:
        """Create a state containing allocation and phenology input ports."""
        state = _allocation_state()
        state.update(
            {
                "phenological_factor": torch.tensor(0.6, dtype=torch.float32),
            }
        )
        return state

    def test_biomass_allocation_state_flow(
        self,
        state_manager: StateManager,
        initial_state: dict[str, torch.Tensor],
    ) -> None:
        """Apply canonical rate and integrated biomass UpdateItems once."""
        allocation = BiomassAllocation()
        dt = 3600.0
        state_manager.initialize_state(initial_state)
        before = state_manager.get_state()

        result = allocation.forward(before, time=0.0, dt=dt)

        assert set(result) == ALLOCATION_RATE_KEYS | ALLOCATION_STATE_KEYS | ALLOCATION_DIAGNOSTICS
        for key in ALLOCATION_RATE_KEYS:
            item = result[key]
            assert item.mode is UpdateMode.REPLACE
            assert item.per_timestep is False
            assert item.unit == "kg C m-2 s-1"
            assert item.element is None
            assert item.source_functor == "biomass_allocation"
        for key in ALLOCATION_STATE_KEYS:
            item = result[key]
            assert item.mode is UpdateMode.ACCUMULATE
            assert item.per_timestep is True
            assert item.unit == "kg C m-2"
            assert item.element == "C"
            assert item.source_functor == "biomass_allocation"

        state_updates = {
            key: result[key]
            for key in ALLOCATION_RATE_KEYS | ALLOCATION_STATE_KEYS
        }
        updated = state_manager.apply_updates(
            state_updates,
            dt=dt,
            check_conservation=False,
        )
        for key in ALLOCATION_STATE_KEYS:
            torch.testing.assert_close(updated[key] - before[key], _value(result, key))
        for key in ALLOCATION_RATE_KEYS:
            torch.testing.assert_close(updated[key], _value(result, key))

    def test_phenology_growth_integration(
        self,
        state_manager: StateManager,
        initial_state: dict[str, torch.Tensor],
    ) -> None:
        """Observe activity from explicit selected-owner control scenarios."""
        phenology = PhenologyGrowth(phenology_model_owner="gsi_phenology")
        state_manager.initialize_state(initial_state)

        result = phenology.forward(state_manager.get_state(), time=0.0, dt=86400.0)
        assert set(result) == set(phenology.metadata.output_variables)
        assert "rate" not in result
        assert all(isinstance(item, UpdateItem) for item in result.values())
        assert all(item.mode is UpdateMode.REPLACE for item in result.values())
        assert all(item.per_timestep is False for item in result.values())
        assert all(item.unit == "dimensionless" for item in result.values())
        active_key = "diagnostics.phenology.growth_active"
        assert result[active_key].value.shape == torch.Size([])
        assert bool(result[active_key].value.item())

        early_state = dict(initial_state)
        early_state["phenological_factor"] = torch.tensor(0.0)
        early = phenology.forward(early_state, time=0.0, dt=86400.0)
        assert not bool(early[active_key].value.item())

        mature_state = dict(initial_state)
        mature_state["phenological_factor"] = torch.tensor(
            phenology.growth_active_threshold
        )
        mature = phenology.forward(mature_state, time=0.0, dt=86400.0)
        assert not bool(mature[active_key].value.item())

        late_state = dict(initial_state)
        late_state["phenological_factor"] = torch.tensor(1.0)
        late = phenology.forward(late_state, time=0.0, dt=86400.0)
        assert bool(late[active_key].value.item())

    def test_structural_growth_integration(
        self,
        initial_state: dict[str, torch.Tensor],
    ) -> None:
        """Verify StructuralGrowth is a four-output observer, not an LAI writer."""
        structural = StructuralGrowth()
        dt = 60.0
        state = {
            "growth_rate_leaf": torch.tensor(2.0e-7),
            "leaf_area_index": initial_state["leaf_area_index"],
        }

        result = structural.forward(state, time=0.0, dt=dt)

        assert set(result) == STRUCTURAL_OUTPUTS
        assert "leaf_area_index" not in result
        for key, item in result.items():
            assert item.mode is UpdateMode.REPLACE
            assert item.source_functor == "structural_growth"
            assert torch.isfinite(item.value).all()
            if key.endswith("step_leaf_area_delta"):
                assert item.per_timestep is True
                assert item.unit == "m2 m-2"
            else:
                assert item.per_timestep is False
                assert item.unit == "m2 m-2 s-1"
        net = _value(result, "diagnostics.structural_growth.net_leaf_area_tendency")
        step = _value(result, "diagnostics.structural_growth.step_leaf_area_delta")
        torch.testing.assert_close(step, net * dt)

    def test_module_chain_integration(
        self,
        state_manager: StateManager,
        initial_state: dict[str, torch.Tensor],
    ) -> None:
        """Run phenology, allocation, and the structural observer in data-flow order."""
        phenology = PhenologyGrowth(phenology_model_owner="gsi_phenology")
        allocation = BiomassAllocation()
        structural = StructuralGrowth()
        dt = 3600.0
        state_manager.initialize_state(initial_state)

        phenology_result = phenology.forward(
            state_manager.get_state(),
            time=0.0,
            dt=dt,
        )
        assert bool(
            phenology_result[
                "diagnostics.phenology.growth_active"
            ].value.item()
        )
        allocation_state = state_manager.get_state().copy()
        allocation_result = allocation.forward(allocation_state, time=0.0, dt=dt)
        structural_result = structural.forward(
            {
                "growth_rate_leaf": _value(allocation_result, "growth_rate_leaf"),
                "leaf_area_index": allocation_state["leaf_area_index"],
            },
            time=0.0,
            dt=dt,
        )

        total_growth = sum(_value(allocation_result, key) for key in ALLOCATION_RATE_KEYS)
        torch.testing.assert_close(total_growth, allocation_state["net_primary_productivity"])
        residual = _value(
            allocation_result,
            "diagnostics.biomass_allocation.carbon_balance_residual",
        )
        torch.testing.assert_close(residual, torch.zeros_like(residual))
        assert set(structural_result) == STRUCTURAL_OUTPUTS
        assert "leaf_area_index" not in structural_result

    def test_parameter_consistency_across_modules(self) -> None:
        """Verify all modules expose the common Functor parameter lifecycle."""
        modules = [
            BiomassAllocation(),
            PhenologyGrowth(phenology_model_owner="gsi_phenology"),
            StructuralGrowth(),
        ]
        manager = get_global_manager()

        assert manager is not None
        for module in modules:
            assert isinstance(module, torch.nn.Module)
            assert hasattr(module, "device")
            assert module.metadata.references
            assert module.metadata.parameters
            assert callable(module.get_parameter_value)

    def test_error_handling_missing_state(self) -> None:
        """Require every enabled BiomassAllocation input without defaults."""
        allocation = BiomassAllocation()
        valid_state = _allocation_state()

        for missing_key in allocation.get_required_inputs():
            state = dict(valid_state)
            state.pop(missing_key)
            with pytest.raises(KeyError, match=missing_key):
                allocation.forward(state, time=0.0, dt=60.0)

        allometric_allocation = BiomassAllocation(enable_allometry_control=True)
        for missing_key in ("leaf_biomass_c", "wood_biomass_c", "root_biomass_c"):
            state = dict(valid_state)
            state.pop(missing_key)
            with pytest.raises(KeyError, match=missing_key):
                allometric_allocation.forward(state, time=0.0, dt=60.0)

    def test_unit_conversion_integration(self) -> None:
        """Verify kg-carbon leaf flux maps to LAI tendency through SLA."""
        structural = StructuralGrowth()
        growth_rate_leaf = torch.tensor(1.0e-6, dtype=torch.float32)
        leaf_area_index = torch.tensor(3.0, dtype=torch.float32)
        dt = 300.0
        state = {
            "growth_rate_leaf": growth_rate_leaf,
            "leaf_area_index": leaf_area_index,
        }

        result = structural.forward(state, time=0.0, dt=dt)
        sla = torch.as_tensor(
            structural.get_parameter_value(
                "sla",
                namespace="biosphere.phenology.morphology",
                required=True,
            ),
            dtype=growth_rate_leaf.dtype,
        )
        turnover = torch.as_tensor(
            structural.get_parameter_value(
                "leaf_turnover_rate",
                namespace="biosphere.phenology.turnover",
                required=True,
            ),
            dtype=growth_rate_leaf.dtype,
        )
        expected_gross = growth_rate_leaf * sla
        expected_loss = leaf_area_index * turnover
        expected_net = expected_gross - expected_loss

        torch.testing.assert_close(
            _value(result, "diagnostics.structural_growth.gross_leaf_area_growth_tendency"),
            expected_gross,
        )
        torch.testing.assert_close(
            _value(result, "diagnostics.structural_growth.leaf_area_senescence_tendency"),
            expected_loss,
        )
        torch.testing.assert_close(
            _value(result, "diagnostics.structural_growth.net_leaf_area_tendency"),
            expected_net,
        )
        torch.testing.assert_close(
            _value(result, "diagnostics.structural_growth.step_leaf_area_delta"),
            expected_net * dt,
        )

    def test_conservation_validation(
        self,
        initial_state: dict[str, torch.Tensor],
    ) -> None:
        """Close both the allocation rate and integrated carbon ledgers."""
        allocation = BiomassAllocation()
        dt = 7200.0
        result = allocation.forward(initial_state, time=0.0, dt=dt)

        total_rate = sum(_value(result, key) for key in ALLOCATION_RATE_KEYS)
        total_delta = sum(_value(result, key) for key in ALLOCATION_STATE_KEYS)
        expected_delta = initial_state["net_primary_productivity"] * dt
        residual = _value(
            result,
            "diagnostics.biomass_allocation.carbon_balance_residual",
        )

        torch.testing.assert_close(total_rate, initial_state["net_primary_productivity"])
        torch.testing.assert_close(total_delta, expected_delta)
        torch.testing.assert_close(residual, torch.zeros_like(residual))
        assert "total_construction_respiration" not in result
        assert "effective_net_primary_production" not in result

    def test_dependency_resolution(self) -> None:
        """Verify dependency direction follows NPP and selected phenology owners."""
        allocation = BiomassAllocation()
        phenology = PhenologyGrowth(phenology_model_owner="gsi_phenology")
        structural = StructuralGrowth()

        assert allocation.get_dependencies() == [
            "autotrophic_respiration",
            "water_stress",
            "gsi_phenology",
        ]
        assert "farquhar_photosynthesis" not in allocation.get_dependencies()
        assert phenology.get_dependencies() == ["gsi_phenology"]
        assert structural.get_dependencies() == ["biomass_allocation"]
        assert allocation.get_required_inputs() == [
            "net_primary_productivity",
            "air_temperature",
            "water_stress_factor",
            "leaf_area_index",
            "phenology_phase",
        ]
        assert allocation.metadata.input_variables == [
            "net_primary_productivity",
            "air_temperature",
            "water_stress_factor",
            "leaf_area_index",
            "phenology_phase",
            "leaf_biomass_c",
            "wood_biomass_c",
            "root_biomass_c",
        ]
        assert phenology.get_required_inputs() == ["phenological_factor"]
        assert structural.get_required_inputs() == [
            "growth_rate_leaf",
            "leaf_area_index",
        ]


class TestGrowthModulePerformance:
    """Exercise batch, device, and numerical stability contracts."""

    def test_batch_processing(self) -> None:
        """Use deterministic non-negative NPP across a ten-cell batch."""
        allocation = BiomassAllocation()
        batch_size = 10
        state = {
            "net_primary_productivity": torch.linspace(
                1.0e-8,
                1.0e-6,
                batch_size,
                dtype=torch.float32,
            ),
            "air_temperature": torch.full((batch_size,), 298.15, dtype=torch.float32),
            "water_stress_factor": torch.full((batch_size,), 0.8, dtype=torch.float32),
            "leaf_area_index": torch.full((batch_size,), 3.0, dtype=torch.float32),
            "phenology_phase": torch.ones(batch_size, dtype=torch.float32),
            "leaf_biomass_c": torch.full((batch_size,), 0.1, dtype=torch.float32),
            "wood_biomass_c": torch.full((batch_size,), 0.5, dtype=torch.float32),
            "root_biomass_c": torch.full((batch_size,), 0.2, dtype=torch.float32),
        }

        result = allocation.forward(state, time=0.0, dt=60.0)

        for key in ALLOCATION_RATE_KEYS | ALLOCATION_STATE_KEYS:
            assert _value(result, key).shape == torch.Size([batch_size])
            assert torch.isfinite(_value(result, key)).all()
        total_rate = sum(_value(result, key) for key in ALLOCATION_RATE_KEYS)
        torch.testing.assert_close(total_rate, state["net_primary_productivity"])

    def test_device_consistency(self) -> None:
        """Keep all UpdateItem payloads on the explicitly selected device."""
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))

        for device in devices:
            allocation = BiomassAllocation(device=device)
            result = allocation.forward(
                _allocation_state(device=device),
                time=0.0,
                dt=60.0,
            )

            for item in result.values():
                assert isinstance(item, UpdateItem)
                assert item.value.device.type == device.type
                assert item.source_functor == "biomass_allocation"

    def test_numerical_stability(self) -> None:
        """Preserve finite ledgers for tiny and large non-negative NPP."""
        allocation = BiomassAllocation()

        for npp in (1.0e-12, 1.0e3):
            result = allocation.forward(
                _allocation_state(npp=npp),
                time=0.0,
                dt=1.0,
            )
            for item in result.values():
                assert torch.isfinite(item.value).all()
            total_rate = sum(_value(result, key) for key in ALLOCATION_RATE_KEYS)
            torch.testing.assert_close(
                total_rate,
                torch.tensor(npp, dtype=total_rate.dtype, device=total_rate.device),
            )
            allocation_sum = _value(
                result,
                "diagnostics.biomass_allocation.allocation_sum",
            )
            torch.testing.assert_close(allocation_sum, torch.ones_like(allocation_sum))

        with pytest.raises(ValueError, match="non-negative"):
            allocation.forward(
                _allocation_state(npp=-1.0e-6),
                time=0.0,
                dt=1.0,
            )
