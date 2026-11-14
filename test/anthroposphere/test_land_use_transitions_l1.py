#!/usr/bin/env python3
"""
L1 Unit Tests for LandUseTransitions
Tests the land use transition dynamics using Markov chain models

Test Coverage:
- Transition matrix initialization and validation
- Land use fraction dynamics and conservation
- Event-based transitions
- Parameter modifier calculations
- Error handling and edge cases
- Scientific accuracy and conservation laws
"""

import pytest
import torch
from pathlib import Path

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "relic"))

from relic.scientific.anthroposphere.land_use.transitions import (
    LandUseTransitions, LandUseType, TransitionEvent
)
from relic.core.abstractions import ScaleCoupling
from relic.core.state import UpdateItem, UpdateMode
from relic.utils.physical_constants import get_physical_constant


class TestLandUseTransitions:
    """Test suite for LandUseTransitions class"""

    OUTPUT_UNITS = {
        "land_use_fraction": "1",
        "diagnostics.land_use_transitions.delta_fraction": "1",
        "diagnostics.land_use_transitions.dominant_type_index": "1",
        "diagnostics.land_use_transitions.maximum_fraction": "1",
        "diagnostics.land_use_transitions.soil_carbon_input_rate_change": "kg C m-2 s-1",
        "diagnostics.land_use_transitions.canonical_iphs_runtime": "1",
    }

    @staticmethod
    def _unwrap_update(value):
        """Extract tensor payload from UpdateItem-like wrappers."""
        if hasattr(value, "value"):
            return value.value
        return value

    @pytest.fixture
    def land_use_transitions(self):
        """Create a LandUseTransitions instance for testing"""
        return LandUseTransitions(name="test_land_use_transitions")

    def test_initialization(self, land_use_transitions):
        """Test proper initialization of the land use transitions"""
        assert land_use_transitions is not None
        assert hasattr(land_use_transitions, 'name')
        assert land_use_transitions.name == "test_land_use_transitions"
        assert hasattr(land_use_transitions, 'land_use_types')
        assert isinstance(land_use_transitions.land_use_types, list)
        assert len(land_use_transitions.land_use_types) > 0
        assert hasattr(land_use_transitions, 'transition_matrix')
        assert isinstance(land_use_transitions.transition_matrix, torch.Tensor)
        assert not hasattr(land_use_transitions, "current_fractions")

    def test_inheritance(self, land_use_transitions):
        """Test that the class properly inherits from ScaleCoupling"""
        assert isinstance(land_use_transitions, ScaleCoupling)
        assert hasattr(land_use_transitions, 'get_dependencies')
        assert hasattr(land_use_transitions, 'forward')
        assert hasattr(land_use_transitions, 'upscale')
        assert hasattr(land_use_transitions, 'downscale')

    def test_parameter_loading(self, land_use_transitions):
        """Test parameter loading from ParameterManager"""
        # Test loading transition parameters
        param = land_use_transitions.get_parameter_value(
            "transition_rates.deforestation_rate",
            namespace="anthroposphere.land_use",
            validate=True,
            required=True,
            source_order=("manager",),
        )
        # Parameter might be a torch.Tensor, so accept both
        assert isinstance(param, (int, float, torch.Tensor))
        if isinstance(param, torch.Tensor):
            param_value = param.item()
        else:
            param_value = param
        assert param_value >= 0.0 and param_value <= 1.0

        seconds_per_year = get_physical_constant("seconds_per_year")
        assert float(seconds_per_year) > 3.0e7

    def test_land_use_types(self, land_use_transitions):
        """Test land use type definitions"""
        # Check that all required land use types are defined
        expected_types = [
            LandUseType.PRIMARY_FOREST,
            LandUseType.SECONDARY_FOREST,
            LandUseType.CROPLAND_C3,
            LandUseType.CROPLAND_C4,
            LandUseType.PASTURE,
            LandUseType.RANGELAND,
            LandUseType.URBAN,
            LandUseType.WATER,
            LandUseType.BARREN
        ]

        for lu_type in expected_types:
            assert lu_type in land_use_transitions.land_use_types

    def test_transition_matrix_properties(self, land_use_transitions):
        """Test transition matrix mathematical properties"""
        matrix = land_use_transitions.transition_matrix

        # Check dimensions
        n_types = len(land_use_transitions.land_use_types)
        assert matrix.shape == (n_types, n_types)

        # Check CTMC generator constraints
        row_sums = matrix.sum(dim=1)
        torch.testing.assert_close(row_sums, torch.zeros_like(row_sums), rtol=1e-6, atol=1e-6)
        for i in range(n_types):
            assert matrix[i, i] <= 0.0

        # Check that off-diagonal elements are non-negative
        for i in range(n_types):
            for j in range(n_types):
                if i != j:
                    assert matrix[i, j] >= 0.0

    def test_canonical_metadata_contract(self, land_use_transitions):
        """Metadata must expose one state port and namespaced observations."""
        assert land_use_transitions.metadata.input_variables == [
            "land_use_fraction",
            "time_index",
            "gdp_per_capita",
            "population_density",
            "market_accessibility",
            "policy_intensity",
        ]
        assert land_use_transitions.metadata.output_variables == list(self.OUTPUT_UNITS)
        assert land_use_transitions.get_dependencies() == []

    def test_fraction_length_mismatch_fails_fast(self, land_use_transitions):
        """A wrong class axis must not be padded or truncated."""
        bad_state = {"land_use_fraction": torch.ones(8) / 8.0}
        with pytest.raises(ValueError, match="exactly 9 land-use classes"):
            land_use_transitions.forward(bad_state, time=0.0, dt=1.0)

    def test_add_transition_event(self, land_use_transitions):
        """Test adding transition events"""
        initial_event_count = len(land_use_transitions.transition_events)

        # Create a test event
        event = TransitionEvent(
            from_type=LandUseType.PRIMARY_FOREST,
            to_type=LandUseType.CROPLAND_C3,
            area_fraction=0.5,
            time_index=10
        )

        land_use_transitions.add_transition_event(event)

        # Check that event was added
        assert len(land_use_transitions.transition_events) == initial_event_count + 1
        assert land_use_transitions.transition_events[-1] == event

    def test_transition_matrix_application(self, land_use_transitions):
        """Test transition matrix application"""
        # Create test fractions
        n_types = len(land_use_transitions.land_use_types)
        fractions = torch.ones(n_types) / n_types  # Equal fractions

        dt = 3600.0  # 1 hour

        delta = land_use_transitions._apply_transition_matrix(fractions, dt)

        # Check that output has correct dimensions
        assert delta.shape == (n_types,)

        # Check conservation (sum of changes should be zero for closed system)
        total_change = delta.sum()
        torch.testing.assert_close(total_change, torch.tensor(0.0), rtol=1e-10, atol=1e-10)

    def test_transition_events_application(self, land_use_transitions):
        """Test discrete transition events"""
        # Create test fractions with one dominant type
        n_types = len(land_use_transitions.land_use_types)
        fractions = torch.zeros(n_types)
        fractions[land_use_transitions.land_use_types.index(LandUseType.PRIMARY_FOREST)] = 0.8
        fractions[land_use_transitions.land_use_types.index(LandUseType.CROPLAND_C3)] = 0.2

        # Add transition event
        event = TransitionEvent(
            from_type=LandUseType.PRIMARY_FOREST,
            to_type=LandUseType.CROPLAND_C3,
            area_fraction=0.5,
            time_index=5
        )
        land_use_transitions.add_transition_event(event)

        delta = land_use_transitions._apply_transition_events(fractions, 5)

        # Check that changes occurred
        assert delta.shape == (n_types,)
        forest_idx = land_use_transitions.land_use_types.index(LandUseType.PRIMARY_FOREST)
        crop_idx = land_use_transitions.land_use_types.index(LandUseType.CROPLAND_C3)

        # Forest should decrease, crops should increase
        assert delta[forest_idx] < 0
        assert delta[crop_idx] > 0

    def test_canonical_output_update_contract(self, land_use_transitions):
        """Every runtime output must have one explicit observation/state semantic."""
        n_types = len(land_use_transitions.land_use_types)
        state = {"land_use_fraction": torch.ones(n_types) / n_types}
        outputs = land_use_transitions.forward(state, time=0.0, dt=3600.0)

        assert set(outputs) == set(self.OUTPUT_UNITS)
        for name, unit in self.OUTPUT_UNITS.items():
            item = outputs[name]
            assert isinstance(item, UpdateItem)
            assert item.mode is UpdateMode.REPLACE
            assert item.per_timestep is False
            assert item.unit == unit
            assert item.source_functor == land_use_transitions.name

        updated = outputs["land_use_fraction"].value
        delta = outputs["diagnostics.land_use_transitions.delta_fraction"].value
        torch.testing.assert_close(updated, state["land_use_fraction"] + delta)
        assert "land_use_fraction_vector" not in outputs
        assert "land_use_type_index" not in outputs
        assert "lai_modifier" not in outputs

    def test_optional_driver_validation(self, land_use_transitions):
        """Declared socioeconomic drivers must be Tensor, finite, and bounded."""
        n_types = len(land_use_transitions.land_use_types)
        base = torch.ones(n_types) / n_types
        with pytest.raises(TypeError, match="gdp_per_capita must be a torch.Tensor"):
            land_use_transitions.forward(
                {"land_use_fraction": base, "gdp_per_capita": 1000.0},
                time=0.0,
                dt=3600.0,
            )
        with pytest.raises(ValueError, match="policy_intensity must contain only finite"):
            land_use_transitions.forward(
                {
                    "land_use_fraction": base,
                    "policy_intensity": torch.tensor(float("nan")),
                },
                time=0.0,
                dt=3600.0,
            )

    def test_forward_method_interface(self, land_use_transitions):
        """Test the forward method interface"""
        # Create test state
        n_types = len(land_use_transitions.land_use_types)
        state = {
            "land_use_fraction": torch.ones(n_types) / n_types
        }
        time = 1000.0  # 1000 seconds
        dt = 3600.0    # 1 hour

        result = land_use_transitions.forward(state, time=time, dt=dt)
        assert isinstance(result, dict)
        assert set(result) == set(self.OUTPUT_UNITS)

        delta = result["diagnostics.land_use_transitions.delta_fraction"].value
        total_change = delta.sum()
        torch.testing.assert_close(total_change, torch.tensor(0.0), rtol=1e-10, atol=1e-10)

        new_fractions = result["land_use_fraction"].value
        assert torch.all(new_fractions >= 0.0)
        torch.testing.assert_close(new_fractions.sum(), torch.tensor(1.0), rtol=1e-6, atol=1e-10)

    def test_forward_is_stateless(self, land_use_transitions):
        """The fraction state must remain owned by StateManager, not the Functor."""
        fractions = torch.tensor([0.4, 0.2, 0.1, 0.05, 0.1, 0.05, 0.04, 0.03, 0.03])
        first = land_use_transitions.forward(
            {"land_use_fraction": fractions}, time=0.0, dt=3600.0
        )["land_use_fraction"].value
        second = land_use_transitions.forward(
            {"land_use_fraction": fractions}, time=3600.0, dt=3600.0
        )["land_use_fraction"].value
        torch.testing.assert_close(first, second)
        assert not hasattr(land_use_transitions, "current_fractions")

    def test_conservation_law(self, land_use_transitions):
        """Test that conservation laws are maintained"""
        n_types = len(land_use_transitions.land_use_types)

        # Test various initial fraction configurations
        test_fractions = [
            torch.ones(n_types) / n_types,  # Equal distribution
            torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # All primary forest
            torch.tensor([0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0])   # Mixed crops
        ]

        for fractions in test_fractions:
            # Normalize fractions
            normalized_fractions = fractions / fractions.sum()

            # Apply transitions
            state = {"land_use_fraction": normalized_fractions}
            result = land_use_transitions.forward(
                state,
                time=1000.0,
                dt=3600.0,
            )

            # Check conservation
            delta = result["diagnostics.land_use_transitions.delta_fraction"].value
            new_fractions = result["land_use_fraction"].value

            # Total should be conserved
            total_change = delta.sum()
            torch.testing.assert_close(total_change, torch.tensor(0.0), rtol=1e-10, atol=1e-10)

            # Final fractions should be positive and normalized
            assert torch.all(new_fractions >= 0.0)
            torch.testing.assert_close(new_fractions.sum(), torch.tensor(1.0), rtol=1e-6, atol=1e-10)

    def test_upscale_downscale(self, land_use_transitions):
        """Scale changes require explicitly bound typed chain maps."""
        fine_data = torch.tensor([0.2, 0.3, 0.25, 0.25])
        with pytest.raises(RuntimeError, match="explicit typed chain-map bridge"):
            land_use_transitions.upscale(fine_data)

        coarse_data = torch.tensor([0.2, 0.3, 0.25, 0.25])
        with pytest.raises(RuntimeError, match="explicit typed chain-map bridge"):
            land_use_transitions.downscale(coarse_data)

    def test_resolution_converter(self, land_use_transitions):
        """Test resolution conversion functionality"""
        n_types = len(land_use_transitions.land_use_types)

        # Test fraction data conversion
        fine_fractions = torch.tensor([0.3, 0.2, 0.25, 0.25])[:n_types]

        # Test if resolution_converter method exists
        if hasattr(land_use_transitions, 'resolution_converter'):
            # Test coarsening (same resolution)
            same_result = land_use_transitions.resolution_converter(
                fine_fractions, 1000.0, 1000.0, "conservative"
            )
            torch.testing.assert_close(same_result, fine_fractions)

            with pytest.raises(RuntimeError, match="explicit typed chain-map bridge"):
                land_use_transitions.resolution_converter(
                    fine_fractions, 1000.0, 2000.0, "conservative"
                )

            with pytest.raises(RuntimeError, match="explicit typed chain-map bridge"):
                land_use_transitions.resolution_converter(
                    fine_fractions, 2000.0, 500.0, "fraction_weighted"
                )
        else:
            # Skip test if method doesn't exist
            pytest.skip("resolution_converter method not implemented")

    def test_error_handling_negative_fractions(self, land_use_transitions):
        """Test error handling for negative fractions"""
        n_types = len(land_use_transitions.land_use_types)

        # Create state with negative fraction
        fractions = torch.zeros(n_types)
        fractions[0] = -0.1
        fractions[1] = 1.1
        state = {"land_use_fraction": fractions}

        with pytest.raises(ValueError, match="land_use_fraction must be non-negative"):
            land_use_transitions.forward(state, time=1000.0, dt=3600.0)

    def test_error_handling_invalid_time_step(self, land_use_transitions):
        """Test error handling for invalid time steps"""
        n_types = len(land_use_transitions.land_use_types)
        state = {"land_use_fraction": torch.ones(n_types) / n_types}

        result = land_use_transitions.forward(state, 1000.0, 0.0)
        torch.testing.assert_close(result["land_use_fraction"].value, state["land_use_fraction"])
        assert torch.all(
            result["diagnostics.land_use_transitions.delta_fraction"].value == 0.0
        )

        with pytest.raises(ValueError, match="dt must be non-negative"):
            land_use_transitions.forward(state, 1000.0, -3600.0)

    def test_edge_cases_extreme_transitions(self, land_use_transitions):
        """Test edge cases with extreme transition rates"""
        n_types = len(land_use_transitions.land_use_types)

        # Test very high transition rates
        state = {"land_use_fraction": torch.ones(n_types) / n_types}

        # Modify transition matrix with extreme values
        land_use_transitions.transition_matrix[:, :] = 0.5  # Very high rates

        try:
            result = land_use_transitions.forward(
                state,
                time=1000.0,
                dt=3600.0,
            )
            # Should still conserve total area
            total_change = result[
                "diagnostics.land_use_transitions.delta_fraction"
            ].value.sum()
            torch.testing.assert_close(total_change, torch.tensor(0.0), rtol=1e-10, atol=1e-10)
        except Exception as e:
            pytest.fail(f"Extreme transition rates failed: {e}")

    def test_spatial_mask_in_events(self, land_use_transitions):
        """Test spatial masking in transition events"""
        n_types = len(land_use_transitions.land_use_types)
        fractions = torch.zeros((n_types, 2, 2))
        fractions[land_use_transitions.land_use_types.index(LandUseType.PRIMARY_FOREST)] = 1.0

        # Create event with spatial mask
        spatial_mask = torch.tensor([[1.0, 0.0], [0.5, 0.0]])

        event = TransitionEvent(
            from_type=LandUseType.PRIMARY_FOREST,
            to_type=LandUseType.CROPLAND_C3,
            area_fraction=0.5,
            spatial_mask=spatial_mask
        )

        land_use_transitions.add_transition_event(event)
        delta = land_use_transitions._apply_transition_events(fractions, 5)

        # Check that spatial mask was applied
        assert delta.shape == fractions.shape
        forest_idx = land_use_transitions.land_use_types.index(LandUseType.PRIMARY_FOREST)
        crop_idx = land_use_transitions.land_use_types.index(LandUseType.CROPLAND_C3)
        torch.testing.assert_close(delta[forest_idx], -0.5 * spatial_mask)
        torch.testing.assert_close(delta[crop_idx], 0.5 * spatial_mask)

    def test_multiple_events_same_time(self, land_use_transitions):
        """Test multiple events occurring at the same time"""
        n_types = len(land_use_transitions.land_use_types)
        fractions = torch.zeros(n_types)
        fractions[land_use_transitions.land_use_types.index(LandUseType.PRIMARY_FOREST)] = 0.5
        fractions[land_use_transitions.land_use_types.index(LandUseType.SECONDARY_FOREST)] = 0.3
        fractions[land_use_transitions.land_use_types.index(LandUseType.CROPLAND_C3)] = 0.2

        # Add multiple events for the same time
        event1 = TransitionEvent(
            from_type=LandUseType.PRIMARY_FOREST,
            to_type=LandUseType.CROPLAND_C3,
            area_fraction=0.25,
            time_index=10
        )

        event2 = TransitionEvent(
            from_type=LandUseType.SECONDARY_FOREST,
            to_type=LandUseType.PASTURE,
            area_fraction=0.3,
            time_index=10
        )

        land_use_transitions.add_transition_event(event1)
        land_use_transitions.add_transition_event(event2)

        delta = land_use_transitions._apply_transition_events(fractions, 10)

        # Both events should have been applied
        primary_forest_idx = land_use_transitions.land_use_types.index(LandUseType.PRIMARY_FOREST)
        secondary_forest_idx = land_use_transitions.land_use_types.index(LandUseType.SECONDARY_FOREST)
        crop_idx = land_use_transitions.land_use_types.index(LandUseType.CROPLAND_C3)
        pasture_idx = land_use_transitions.land_use_types.index(LandUseType.PASTURE)

        # Forest should decrease, crops and pasture should increase
        assert delta[primary_forest_idx] < 0
        assert delta[secondary_forest_idx] < 0
        assert delta[crop_idx] > 0
        assert delta[pasture_idx] > 0

    def test_scientific_accuracy_transition_rates(self, land_use_transitions):
        """Test scientific accuracy of transition rates"""
        # Test realistic deforestation rates (should be small)
        deforest_rate = land_use_transitions.get_parameter_value(
            "transition_rates.deforestation_rate", 
            namespace="anthroposphere.land_use",
            required=True,
        )
        deforest_value = deforest_rate.item() if isinstance(deforest_rate, torch.Tensor) else deforest_rate
        assert isinstance(deforest_value, (int, float))
        assert 0.0 <= deforest_value <= 0.1  # Realistic range

        # Test urbanization rate (should be very small)
        urban_rate = land_use_transitions.get_parameter_value(
            "transition_rates.urbanization_rate", 
            namespace="anthroposphere.land_use",
            required=True,
        )
        urban_value = urban_rate.item() if isinstance(urban_rate, torch.Tensor) else urban_rate
        assert isinstance(urban_value, (int, float))
        assert 0.0 <= urban_value <= 0.05  # Realistic range

        # Check that rates are physically reasonable
        for rate in [deforest_value, urban_value]:
            assert rate >= 0.0
            assert rate <= 1.0

    def test_numerical_stability_large_timesteps(self, land_use_transitions):
        """Test numerical stability with large time steps"""
        n_types = len(land_use_transitions.land_use_types)
        state = {"land_use_fraction": torch.ones(n_types) / n_types}

        # Test with very large time step
        large_dt = float(get_physical_constant("seconds_per_year"))
        result = land_use_transitions.forward(
            state,
            time=1000.0,
            dt=large_dt,
        )

        # Should still maintain conservation
        total_change = result[
            "diagnostics.land_use_transitions.delta_fraction"
        ].value.sum()
        torch.testing.assert_close(total_change, torch.tensor(0.0), rtol=1e-8, atol=1e-8)

        # Fractions should remain positive and normalized
        new_fractions = result["land_use_fraction"].value
        assert torch.all(new_fractions >= 0.0)
        torch.testing.assert_close(new_fractions.sum(), torch.tensor(1.0), rtol=1e-6, atol=1e-6)

    def test_parameter_consistency(self, land_use_transitions):
        """Test parameter consistency across methods"""
        # Test that all parameter access uses consistent interface
        deforest_rate_1 = land_use_transitions.get_parameter_value(
            "transition_rates.deforestation_rate",
            namespace="anthroposphere.land_use"
        )
        deforest_rate_2 = land_use_transitions.get_parameter_value(
            "transition_rates.deforestation_rate", 
            namespace="anthroposphere.land_use",
            required=True,
        )

        deforest_rate_1 = deforest_rate_1.item() if isinstance(deforest_rate_1, torch.Tensor) else deforest_rate_1
        deforest_rate_2 = deforest_rate_2.item() if isinstance(deforest_rate_2, torch.Tensor) else deforest_rate_2
        assert deforest_rate_1 == deforest_rate_2

        # Test that parameter modifiers use consistent parameter system
        n_types = len(land_use_transitions.land_use_types)
        state = {"land_use_fraction": torch.ones(n_types) / n_types}
        result = land_use_transitions.forward(
            state,
            time=1000.0,
            dt=3600.0,
        )

        assert set(result) == set(self.OUTPUT_UNITS)
        assert all(torch.isfinite(item.value).all() for item in result.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

