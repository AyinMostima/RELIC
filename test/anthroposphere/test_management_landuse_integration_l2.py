#!/usr/bin/env python3
"""
L2 Integration Tests for Management and Land Use Modules
Tests the integration between anthroposphere management activities and land use changes

Test Coverage:
- Management impacts on land use transitions
- Spatial consistency between management and land use
- Parameter integration across management-land use modules
- Feedback loops between land use and management decisions
- Conservation laws in coupled management-land use system
- Temporal dynamics of management-land use interactions
"""

import pytest
import torch
from pathlib import Path
from datetime import datetime

# Import modules under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "relic"))

from relic.scientific.anthroposphere.management.coordinator import ManagementCoordinator
from relic.scientific.anthroposphere.land_use.transitions import LandUseTransitions
from relic.scientific.anthroposphere.land_use.mapping import LandUseParameterMapping
from relic.core.state import UpdateItem


def _with_management_inputs(state):
    """Add canonical ManagementCoordinator inputs using the state grid shape."""
    template = state["soil_moisture"]
    state.update(
        {
            "soil_nitrogen": torch.full_like(template, 0.01),
            "soil_organic_carbon": torch.full_like(template, 5.0),
            "crop_maturity": torch.full_like(template, 0.7),
            "aboveground_biomass": state["crop_biomass"].clone(),
            "air_temperature": torch.full_like(template, 298.15),
            "irrigation_flux": torch.zeros_like(template),
            "harvest_flux": torch.zeros_like(template),
            "grazing_flux": torch.zeros_like(template),
            "co2_emission": torch.zeros_like(template),
            "field_capacity": torch.full_like(template, 0.35),
            "wilting_point": torch.full_like(template, 0.15),
            "evapotranspiration_flux": torch.full_like(template, 3.0e-6),
            "precipitation_flux": torch.zeros_like(template),
        }
    )
    return state


def _apply_updates(state, updates):
    """Apply explicit update payloads in this test composition adapter."""
    result = dict(state)
    result.update(
        {
            key: value.value if isinstance(value, UpdateItem) else value
            for key, value in updates.items()
        }
    )
    return result


def _transition_state(state):
    """Select only the canonical LandUseTransitions inputs."""
    fractions = state["land_use_fraction"]
    if isinstance(fractions, UpdateItem):
        fractions = fractions.value
    result = {"land_use_fraction": fractions}
    for name in LandUseTransitions.DRIVER_INPUTS:
        value = state.get(name)
        if isinstance(value, UpdateItem):
            value = value.value
        if isinstance(value, torch.Tensor):
            result[name] = value
    return result


class TestManagementLandUseIntegration:
    """Test suite for management-land use integration"""

    @pytest.fixture
    def integrated_system(self):
        """Create integrated management and land use system"""
        management_config = {
                'calendar_type': 'noleap',
                'management_practices': ['fertilization', 'irrigation', 'tillage', 'harvest'],
                'coordination_rules': {
                    'conflict_resolution': 'priority',
                    'spatial_consistency': True,
                    'temporal_ordering': True
                },
                'default_parameters': {
                    'fertilization': {
                        'max_rate': 200.0,
                        'efficiency': 0.7
                    },
                    'irrigation': {
                        'max_rate': 1000.0,
                        'efficiency': 0.8
                    },
                    'tillage': {
                        'intensity': 0.5,
                        'depth': 0.3
                    },
                    'harvest': {
                        'efficiency': 0.9,
                        'residue_fraction': 0.3
                    }
                }
        }

        management_coordinator = ManagementCoordinator(
            name="test_management_coordinator",
            config=management_config,
        )
        land_use_transitions = LandUseTransitions(name="test_land_use_transitions")
        land_use_mapping = LandUseParameterMapping(name="test_land_use_mapping")

        return {
            'management': management_coordinator,
            'land_use_transitions': land_use_transitions,
            'land_use_mapping': land_use_mapping,
        }

    def test_parameter_consistency(self, integrated_system):
        """Test parameter consistency across management and land use modules"""
        management = integrated_system['management']
        land_use_transitions = integrated_system['land_use_transitions']
        land_use_mapping = integrated_system['land_use_mapping']

        # The coordinator schedules lagged management ports and does not own a grid.
        assert management.get_parameter_value(
            "spatial_resolution.lat",
        ) is None

        # Spatial parameters remain consistent across the two land-use modules.
        for module in [land_use_transitions, land_use_mapping]:
            spatial_res = module.get_parameter_value(
                "spatial_resolution.lat"
            )
            assert isinstance(spatial_res, (float, int, torch.Tensor))
            if isinstance(spatial_res, torch.Tensor):
                assert spatial_res.item() > 0
            else:
                assert spatial_res > 0

    def test_management_to_land_use_flow(self, integrated_system):
        """Test basic data flow from management to land use modules"""
        management = integrated_system['management']
        land_use_transitions = integrated_system['land_use_transitions']

        # Create simple test state
        state = _with_management_inputs({
            'soil_moisture': torch.tensor([0.3]),
            'crop_biomass': torch.tensor([2.0]),
            'land_use_fraction': torch.tensor([[0.3, 0.2, 0.4, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0]])
        })

        time = datetime(2020, 6, 1).timestamp()
        dt = 86400.0  # 1 day

        try:
            # Run management module
            management_updates = management.forward(state, time, dt)
            assert isinstance(management_updates, dict)

            # Check that management produces some output
            assert len(management_updates) > 0, "Management should produce some outputs"

            # Update state with management results
            updated_state = state.copy()
            updated_state.update(management_updates)

            # Run land use transitions
            transition_updates = land_use_transitions.forward(
                _transition_state(updated_state),
                time,
                dt,
            )
            assert isinstance(transition_updates, dict)

            # Check that transitions produce some output
            assert len(transition_updates) > 0, "Land use transitions should produce some outputs"

        except Exception as e:
            pytest.fail(f"Management to land use flow failed: {e}")

    def test_spatial_consistency_integration(self, integrated_system):
        """Test basic spatial consistency between management and land use"""
        management = integrated_system['management']
        land_use_transitions = integrated_system['land_use_transitions']

        # Create simple spatial test data
        state = _with_management_inputs({
            'soil_moisture': torch.tensor([0.3, 0.4]),
            'crop_biomass': torch.tensor([2.0, 1.5]),
            'land_use_fraction': torch.tensor([
                [0.3, 0.2, 0.4, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0],
                [0.2, 0.3, 0.45, 0.03, 0.02, 0.0, 0.0, 0.0, 0.0]
            ])
        })

        time = datetime(2020, 6, 1).timestamp()
        dt = 86400.0

        try:
            # Run management
            management_updates = management.forward(state, time, dt)
            assert isinstance(management_updates, dict)

            # Check that management outputs are reasonable
            for key, value in management_updates.items():
                if isinstance(value, torch.Tensor):
                    assert torch.isfinite(value).all(), f"Management output {key} should be finite"

            # Run land use transitions
            state_with_management = state.copy()
            state_with_management.update(management_updates)

            transition_updates = land_use_transitions.forward(
                _transition_state(state_with_management),
                time,
                dt,
            )
            assert isinstance(transition_updates, dict)

            # Check that land use outputs are reasonable
            for key, value in transition_updates.items():
                if isinstance(value, torch.Tensor):
                    assert torch.isfinite(value).all(), f"Land use output {key} should be finite"

        except Exception as e:
            pytest.fail(f"Spatial consistency integration test failed: {e}")

    def test_temporal_integration_dynamics(self, integrated_system):
        """Test basic temporal dynamics of management-land use integration"""
        management = integrated_system['management']
        land_use_transitions = integrated_system['land_use_transitions']

        # Create simple initial state
        state = _with_management_inputs({
            'soil_moisture': torch.tensor([0.3]),
            'crop_biomass': torch.tensor([2.0]),
            'land_use_fraction': torch.tensor([[0.3, 0.2, 0.4, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0]])
        })

        # Simulate a few time steps
        base_time = datetime(2020, 6, 1).timestamp()
        current_state = state.copy()

        try:
            for day in range(3):  # 3 days
                time = base_time + day * 86400.0

                # Run management
                management_updates = management.forward(current_state, time, 86400.0)
                current_state.update(management_updates)

                # Run land use transitions
                transition_updates = land_use_transitions.forward(
                    _transition_state(current_state),
                    time,
                    86400.0,
                )
                current_state.update(transition_updates)

            # Check that both modules ran successfully
            assert True, "Temporal integration should complete without errors"

        except Exception as e:
            pytest.fail(f"Temporal integration dynamics test failed: {e}")

    def test_feedback_loops(self, integrated_system):
        """Test feedback loops between land use and management decisions"""
        management = integrated_system['management']
        land_use_mapping = integrated_system['land_use_mapping']

        # Create states with different land use compositions
        cropland_dominant = _with_management_inputs({
            'soil_moisture': torch.tensor([0.25]),
            'crop_biomass': torch.tensor([3.0]),
            'land_use_fraction': torch.tensor([[0.05, 0.05, 0.85, 0.03, 0.02, 0.0, 0.0, 0.0, 0.0]])  # Cropland dominant
        })

        forest_dominant = _with_management_inputs({
            'soil_moisture': torch.tensor([0.4]),
            'crop_biomass': torch.tensor([1.0]),
            'land_use_fraction': torch.tensor([[0.85, 0.1, 0.02, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0]])  # Forest dominant
        })

        time = datetime(2020, 6, 1).timestamp()
        dt = 86400.0

        try:
            # Test management response to different land uses
            cropland_management = management.forward(cropland_dominant, time, dt)
            forest_management = management.forward(forest_dominant, time, dt)

            # Management should respond differently to different land uses
            cropland_keys = list(cropland_management.keys())
            forest_keys = list(forest_management.keys())

            # Both should produce some management outputs
            assert len(cropland_keys) > 0, "Should manage cropland"
            assert len(forest_keys) > 0, "Should manage forest"

            # Test land use mapping response to management
            if 'fertilizer_rate' in cropland_management:
                cropland_with_mgmt = cropland_dominant.copy()
                cropland_with_mgmt.update(cropland_management)

                mapping_response = land_use_mapping.forward(cropland_with_mgmt, time, dt)
                assert isinstance(mapping_response, dict), "Land use mapping should respond to management"

        except Exception as e:
            pytest.fail(f"Feedback loops test failed: {e}")

    def test_conservation_laws_integration(self, integrated_system):
        """Test conservation laws in coupled management-land use system"""
        management = integrated_system['management']
        land_use_transitions = integrated_system['land_use_transitions']

        # Create test state with known initial conditions
        initial_land_use = torch.tensor([[0.3, 0.2, 0.4, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0]])  # Sums to 1
        initial_carbon = torch.tensor([50.0])  # tC/ha
        initial_nitrogen = torch.tensor([5.0])  # tN/ha

        state = _with_management_inputs({
            'soil_moisture': torch.tensor([0.3]),
            'crop_biomass': torch.tensor([2.0]),
            'land_use_fraction': initial_land_use.clone(),
            'soil_carbon_stock': initial_carbon.clone(),
            'soil_nitrogen_stock': initial_nitrogen.clone()
        })

        time = datetime(2020, 6, 1).timestamp()
        dt = 86400.0

        try:
            # Run management
            management_updates = management.forward(state, time, dt)
            state_with_mgmt = _apply_updates(state, management_updates)

            # Run land use transitions
            transition_updates = land_use_transitions.forward(
                _transition_state(state_with_mgmt),
                time,
                dt,
            )
            final_state = _apply_updates(state_with_mgmt, transition_updates)

            # Check land use conservation (should sum to 1)
            if 'land_use_fraction' in final_state:
                final_land_use = final_state['land_use_fraction']
                land_use_sum = torch.sum(final_land_use, dim=-1)
                torch.testing.assert_close(
                    land_use_sum, torch.tensor([1.0]),
                    rtol=1e-6, atol=1e-10,
                    msg="Land use fractions should sum to 1"
                )

            # Check that changes are reasonable
            land_use_change = torch.abs(final_land_use - initial_land_use)
            assert torch.all(land_use_change < 0.1), "Land use changes should be gradual"

        except Exception as e:
            pytest.fail(f"Conservation laws integration test failed: {e}")

    def test_error_propagation_integration(self, integrated_system):
        """Test error propagation between management and land use modules"""
        management = integrated_system['management']
        land_use_transitions = integrated_system['land_use_transitions']

        # Create problematic state
        invalid_state = _with_management_inputs({
            'soil_moisture': torch.tensor([float('nan')]),  # Invalid soil moisture
            'crop_biomass': torch.tensor([2.0]),
            'land_use_fraction': torch.tensor([[0.3, 0.2, 0.4, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0]])
        })

        time = datetime(2020, 6, 1).timestamp()
        dt = 86400.0

        try:
            # Management should handle invalid state gracefully
            management_updates = management.forward(invalid_state, time, dt)

            # Check that outputs are reasonable despite invalid input
            for key, value in management_updates.items():
                if isinstance(value, torch.Tensor):
                    # Should not propagate NaN values
                    assert not torch.any(torch.isnan(value)), f"Management output {key} contains NaN"

            # Update state and test land use transitions
            state_with_mgmt = invalid_state.copy()
            state_with_mgmt.update(management_updates)

            # Land use transitions should also handle problematic state
            transition_updates = land_use_transitions.forward(
                _transition_state(state_with_mgmt),
                time,
                dt,
            )

            for key, value in transition_updates.items():
                if isinstance(value, torch.Tensor):
                    assert not torch.any(torch.isnan(value)), f"Land use output {key} contains NaN"

        except Exception as e:
            # Should fail gracefully with informative error
            assert "soil_moisture" in str(e).lower() or "nan" in str(e).lower(), \
                f"Error should mention problematic input: {e}"

    def test_performance_integration(self, integrated_system):
        """Test basic performance of integrated management-land use system"""
        management = integrated_system['management']
        land_use_transitions = integrated_system['land_use_transitions']

        # Create simple test problem
        state = _with_management_inputs({
            'soil_moisture': torch.rand(10) * 0.4 + 0.2,
            'crop_biomass': torch.rand(10) * 5.0,
            'land_use_fraction': torch.rand(10, 9),
        })
        state['land_use_fraction'] = state['land_use_fraction'] / state['land_use_fraction'].sum(dim=1, keepdim=True)

        base_time = datetime(2020, 6, 1).timestamp()
        dt = 86400.0

        # Test performance
        import time as time_module
        start_time = time_module.time()

        try:
            # Run a few iterations
            for iteration in range(3):
                management_updates = management.forward(state, base_time + iteration * dt, dt)
                state.update(management_updates)

                transition_updates = land_use_transitions.forward(
                    _transition_state(state),
                    base_time + iteration * dt,
                    dt,
                )
                state.update(transition_updates)

            end_time = time_module.time()
            processing_time = end_time - start_time

            # Should complete in reasonable time
            assert processing_time < 5.0, f"Integration too slow: {processing_time:.2f}s"

        except Exception as e:
            pytest.fail(f"Performance integration test failed: {e}")

    def test_configuration_integration(self, integrated_system):
        """Test configuration consistency across modules"""
        management = integrated_system['management']
        land_use_transitions = integrated_system['land_use_transitions']
        land_use_mapping = integrated_system['land_use_mapping']

        # Management coordination is grid-agnostic by contract.
        assert management.get_parameter_value(
            "spatial_resolution.lat",
        ) is None

        # The spatial land-use modules must share the same resolution.
        lu_spatial_res = land_use_transitions.get_parameter_value(
            "spatial_resolution.lat"
        )
        mapping_spatial_res = land_use_mapping.get_parameter_value(
            "spatial_resolution.lat"
        )

        if isinstance(lu_spatial_res, torch.Tensor):
            lu_res_val = lu_spatial_res.item()
        else:
            lu_res_val = lu_spatial_res

        if isinstance(mapping_spatial_res, torch.Tensor):
            mapping_res_val = mapping_spatial_res.item()
        else:
            mapping_res_val = mapping_spatial_res

        assert abs(mapping_res_val - lu_res_val) < 0.1, \
            f"Spatial resolutions should be consistent: {mapping_res_val} vs {lu_res_val}"

    def test_edge_cases_integration(self, integrated_system):
        """Test edge cases in integrated system"""
        management = integrated_system['management']
        land_use_transitions = integrated_system['land_use_transitions']

        # Test with minimal land use (one dominant type)
        minimal_state = _with_management_inputs({
            'soil_moisture': torch.tensor([0.3]),
            'crop_biomass': torch.tensor([2.0]),
            'land_use_fraction': torch.tensor([[0.98, 0.01, 0.005, 0.003, 0.002, 0.0, 0.0, 0.0, 0.0]])  # One dominant type
        })

        # Test with uniform land use
        uniform_state = _with_management_inputs({
            'soil_moisture': torch.tensor([0.3]),
            'crop_biomass': torch.tensor([2.0]),
            'land_use_fraction': torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0]])  # Uniform distribution
        })

        time = datetime(2020, 6, 1).timestamp()
        dt = 86400.0

        try:
            # Test minimal land use
            minimal_mgmt = management.forward(minimal_state, time, dt)
            minimal_trans = land_use_transitions.forward(
                _transition_state(_apply_updates(minimal_state, minimal_mgmt)),
                time,
                dt,
            )
            assert isinstance(minimal_trans, dict)

            # Test uniform land use
            uniform_mgmt = management.forward(uniform_state, time, dt)
            uniform_trans = land_use_transitions.forward(
                _transition_state(_apply_updates(uniform_state, uniform_mgmt)),
                time,
                dt,
            )
            assert isinstance(uniform_trans, dict)

            # Both should handle edge cases gracefully
            for updates in [minimal_mgmt, minimal_trans, uniform_mgmt, uniform_trans]:
                for key, value in updates.items():
                    if isinstance(value, torch.Tensor):
                        assert torch.isfinite(value).all(), f"Output {key} contains non-finite values"

        except Exception as e:
            pytest.fail(f"Edge cases integration test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


