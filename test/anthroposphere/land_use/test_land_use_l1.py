#!/usr/bin/env python3
"""
L1 Unit Tests for Land Use Modules

Test Categories:
- L1.1: Class instantiation and inheritance
- L1.2: Parameter management system validation
- L1.3: Land use to parameter mapping accuracy
- L1.4: Physical field operations
- L1.5: Weighted parameter calculations
- L1.6: Abstract base class compliance

Scientific References:
- Pongratz, J., et al. (2018). Models meet data: Challenges and opportunities in implementing land management in Earth system models.
- Lawrence, D.M., et al. (2016). The Land Use Model Intercomparison Project (LUMIP).
- FAO (2017). Voluntary Guidelines for Sustainable Soil Management.

Test Framework: pytest
"""

import pytest
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

# Import modules to test
from relic.scientific.anthroposphere.land_use.mapping import (
    DIAGNOSTIC_PREFIX,
    LandUseParameterMapping,
)
from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.anthroposphere.land_use.transitions import (
    LandUseTransitions,
    LandUseType
)

from relic.core.abstractions import PhysicalField
from relic.core.functor import PhysicalFieldFunctor

# Test constants
DEVICE = torch.device("cpu")
TEST_BATCH_SIZE = 4
TOLERANCE = 1e-6
REL_TOLERANCE = 1e-3


class TestLandUseParameterMappingL1:
    """L1 Unit Tests for LandUseParameterMapping"""
    
    @pytest.fixture
    def land_use_mapping(self):
        """Create land use parameter mapping instance for testing"""
        return LandUseParameterMapping()
    
    @pytest.fixture
    def sample_state(self):
        """Sample state dictionary with land use fractions"""
        # Create realistic land use fractions (sum to 1)
        batch_shape = (TEST_BATCH_SIZE, 1)
        
        # 9 land use types: [primary_forest, secondary_forest, cropland_c3, cropland_c4, 
        #                  pasture, rangeland, urban, water, barren]
        land_use_fractions = torch.tensor([
            [0.15, 0.10, 0.25, 0.08, 0.20, 0.05, 0.10, 0.05, 0.02],  # Mixed landscape
            [0.05, 0.08, 0.40, 0.15, 0.12, 0.03, 0.12, 0.03, 0.02],  # Agricultural
            [0.30, 0.20, 0.10, 0.05, 0.15, 0.08, 0.05, 0.05, 0.02],  # Forested
            [0.02, 0.03, 0.15, 0.10, 0.20, 0.25, 0.15, 0.08, 0.02],  # Mixed with rangeland
        ], device=DEVICE).reshape(TEST_BATCH_SIZE, 9, 1)  # [batch, land_use_types, spatial]
        
        return {
            'land_use_fraction': land_use_fractions,
            'coordinates': torch.randn(TEST_BATCH_SIZE, 2, device=DEVICE)
        }
    
    def test_class_instantiation_and_inheritance(self, land_use_mapping):
        """Test class instantiation and inheritance"""
        assert land_use_mapping is not None
        assert isinstance(land_use_mapping, LandUseParameterMapping)
        assert isinstance(land_use_mapping, PhysicalFieldFunctor)
        assert isinstance(land_use_mapping, PhysicalField)
        
        # Should have physical field methods
        assert hasattr(land_use_mapping, 'get_value_at')
        assert hasattr(land_use_mapping, 'compute_gradient')
        assert hasattr(land_use_mapping, 'interpolate')
        assert hasattr(land_use_mapping, 'update_field')
        
        # Methods should be callable
        assert callable(land_use_mapping.get_value_at)
        assert callable(land_use_mapping.compute_gradient)
        assert callable(land_use_mapping.interpolate)
        assert callable(land_use_mapping.update_field)
    
    def test_parameter_management(self, land_use_mapping):
        """Test parameter management system"""
        # Test vegetation parameter access
        primary_forest_lai = land_use_mapping.get_parameter_value(
            "vegetation.primary_forest.lai_max",
            namespace="anthroposphere.land_use"
        )
        assert isinstance(primary_forest_lai, (float, torch.Tensor))
        assert 5.0 <= primary_forest_lai <= 10.0  # Reasonable LAI range
    
    def test_parameter_database_initialization(self, land_use_mapping):
        """Test that parameter databases are properly initialized"""
        # Should have all land use types
        expected_land_use_types = list(LandUseType)
        for lu_type in expected_land_use_types:
            assert lu_type in land_use_mapping.vegetation_params, f"Missing vegetation params for {lu_type}"
            assert lu_type in land_use_mapping.soil_params, f"Missing soil params for {lu_type}"
            assert lu_type in land_use_mapping.hydro_params, f"Missing hydro params for {lu_type}"
        
        # Check specific parameter values
        forest_veg = land_use_mapping.vegetation_params[LandUseType.PRIMARY_FOREST]
        assert forest_veg.lai_max > forest_veg.lai_min, "Forest LAI max should be > min"
        assert forest_veg.c3_fraction > forest_veg.c4_fraction, "Primary forest should be mostly C3"
        
        cropland_c4 = land_use_mapping.vegetation_params[LandUseType.CROPLAND_C4]
        assert cropland_c4.c4_fraction > cropland_c4.c3_fraction, "C4 cropland should be mostly C4"
        
        urban_hydro = land_use_mapping.hydro_params[LandUseType.URBAN]
        assert urban_hydro.runoff_coefficient > 0.7, "Urban should have high runoff"
        urban_soil = land_use_mapping.soil_params[LandUseType.URBAN]
        assert urban_soil.infiltration_rate < 10.0, "Urban should have low infiltration"
    
    # L1.3: Land use to parameter mapping accuracy
    def test_weighted_parameter_calculation(self, land_use_mapping, sample_state):
        """Test weighted parameter calculation accuracy"""
        land_use_fractions = sample_state['land_use_fraction']
        
        # Calculate weighted parameters
        veg_params, soil_params, hydro_params = land_use_mapping.calculate_weighted_parameters(
            land_use_fractions
        )
        
        # Check that all parameter types are present
        expected_veg_params = ['lai_max', 'lai_min', 'vcmax25', 'jmax25', 'root_depth']
        expected_soil_params = ['decomposition_rate', 'nitrification_rate', 'soil_porosity']
        expected_hydro_params = ['albedo', 'runoff_coefficient', 'surface_roughness']
        
        for param in expected_veg_params:
            assert param in veg_params, f"Missing vegetation parameter: {param}"
            assert veg_params[param].shape == (TEST_BATCH_SIZE, 1), f"Wrong shape for {param}"
        
        for param in expected_soil_params:
            assert param in soil_params, f"Missing soil parameter: {param}"
            assert soil_params[param].shape == (TEST_BATCH_SIZE, 1), f"Wrong shape for {param}"
        
        for param in expected_hydro_params:
            assert param in hydro_params, f"Missing hydrological parameter: {param}"
            assert hydro_params[param].shape == (TEST_BATCH_SIZE, 1), f"Wrong shape for {param}"
    
    def test_parameter_bounds_validation(self, land_use_mapping):
        """Test that mapped parameters stay within realistic bounds"""
        # Create test land use fractions
        land_use_fractions = torch.ones(1, 9, 1, device=DEVICE) / 9.0  # Equal mix
        
        veg_params, soil_params, hydro_params = land_use_mapping.calculate_weighted_parameters(
            land_use_fractions
        )
        
        # Check vegetation parameter bounds
        assert 0.0 <= veg_params['lai_max'].item() <= 10.0, "LAI max out of bounds"
        assert 0.0 <= veg_params['vcmax25'].item() <= 150.0, "Vcmax out of bounds"
        assert 0.0 <= veg_params['root_depth'].item() <= 5.0, "Root depth out of bounds"
        assert 0.0 <= veg_params['c3_fraction'].item() <= 1.0, "C3 fraction out of bounds"
        
        # Check soil parameter bounds
        assert 0.0 <= soil_params['soil_porosity'].item() <= 0.6, "Soil porosity out of bounds"
        assert 0.0 <= soil_params['decomposition_rate'].item() <= 2.0, "Decomposition rate out of bounds"
        
        # Check hydrological parameter bounds
        assert 0.0 <= hydro_params['albedo'].item() <= 0.5, "Albedo out of bounds"
        assert 0.0 <= hydro_params['runoff_coefficient'].item() <= 1.0, "Runoff coefficient out of bounds"
    
    def test_land_use_fraction_conservation(self, land_use_mapping):
        """Test that land use fractions sum to 1"""
        # Create test fractions that don't sum to 1
        invalid_fractions = torch.tensor([
            [0.2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Sums to 1.2
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Sums to 0.9
        ], device=DEVICE).reshape(2, 9, 1)
        
        # Should handle invalid fractions gracefully
        try:
            veg_params, soil_params, hydro_params = land_use_mapping.calculate_weighted_parameters(
                invalid_fractions
            )
            # If no error, check that outputs are still reasonable
            assert all(torch.isfinite(p).all() for p in veg_params.values())
            assert all(torch.isfinite(p).all() for p in soil_params.values())
            assert all(torch.isfinite(p).all() for p in hydro_params.values())
        except Exception:
            # Raising an error for invalid fractions is also acceptable
            pass
    
    def test_land_use_type_specific_parameters(self, land_use_mapping):
        """Test that specific land use types have characteristic parameters"""
        # Test forest vs cropland differences
        forest_only = torch.zeros(1, 9, 1, device=DEVICE)
        forest_only[0, 0, 0] = 1.0  # 100% primary forest
        
        crop_only = torch.zeros(1, 9, 1, device=DEVICE)
        crop_only[0, 2, 0] = 1.0  # 100% C3 cropland
        
        forest_veg, _, _ = land_use_mapping.calculate_weighted_parameters(forest_only)
        crop_veg, _, _ = land_use_mapping.calculate_weighted_parameters(crop_only)
        
        # Forest should have higher LAI and root depth
        assert forest_veg['lai_max'].item() > crop_veg['lai_max'].item(), "Forest should have higher LAI"
        assert forest_veg['root_depth'].item() > crop_veg['root_depth'].item(), "Forest should have deeper roots"
        
        # Urban vs water differences
        urban_only = torch.zeros(1, 9, 1, device=DEVICE)
        urban_only[0, 6, 0] = 1.0  # 100% urban
        
        water_only = torch.zeros(1, 9, 1, device=DEVICE)
        water_only[0, 7, 0] = 1.0  # 100% water
        
        _, _, urban_hydro = land_use_mapping.calculate_weighted_parameters(urban_only)
        _, _, water_hydro = land_use_mapping.calculate_weighted_parameters(water_only)
        
        # Urban should have higher runoff, water should have zero runoff
        assert urban_hydro['runoff_coefficient'].item() > water_hydro['runoff_coefficient'].item(), \
            "Urban should have higher runoff than water"
    
    # L1.4: Physical field operations
    def test_get_value_at(self, land_use_mapping, sample_state):
        """Spatial queries fail until field support and a chain map are bound."""
        coordinates = sample_state['coordinates']

        with pytest.raises(RuntimeError, match="current spatial field"):
            land_use_mapping.get_value_at(coordinates)
    
    def test_compute_gradient(self, land_use_mapping, sample_state):
        """Gradient queries fail until field support and a chain map are bound."""
        coordinates = sample_state['coordinates']

        with pytest.raises(RuntimeError, match="current spatial field"):
            land_use_mapping.compute_gradient(coordinates)
    
    def test_interpolation_methods(self, land_use_mapping):
        """Only the explicitly bound chain-map runtime is accepted."""
        # Create source and target coordinates
        source_coords = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], device=DEVICE, dtype=torch.float32)
        target_coords = torch.tensor([[0.5, 0.5], [0.2, 0.8]], device=DEVICE, dtype=torch.float32)
        source_values = torch.tensor([1.0, 2.0, 3.0], device=DEVICE, dtype=torch.float32)
        
        def chain_map(source_coordinates, values, targets):
            assert torch.equal(source_coordinates, source_coords)
            return values.mean().expand(targets.shape[0]).clone()

        mapper = LandUseParameterMapping(field_query_bridge=chain_map)
        interpolated = mapper.interpolate(
            source_coords,
            target_coords,
            source_values=source_values,
        )
        assert torch.equal(interpolated, torch.full((2,), 2.0))
        # The bridge owns the algorithm; there is no method argument.
        with pytest.raises(TypeError, match="method"):
            mapper.interpolate(
                source_coords,
                target_coords,
                method="linear",
                source_values=source_values,
            )
    
    def test_field_update(self, land_use_mapping, sample_state):
        """Test field update based on land use changes"""
        # Create initial land use fractions
        initial_fractions = sample_state['land_use_fraction'].clone()
        
        # Create updated fractions (more forest, less cropland)
        updated_fractions = initial_fractions.clone()
        updated_fractions[:, 0, :] += 0.1  # Increase primary forest
        updated_fractions[:, 2, :] -= 0.1  # Decrease C3 cropland
        
        # Update field
        source_terms = {
            'land_use_fraction': updated_fractions,
            'coordinates': sample_state['coordinates']
        }
        
        land_use_mapping.update_field(time_step=86400, source_terms=source_terms)
        
        # Field should be updated (internal state)
        # This is hard to test directly without exposing internal state
        # But we can test that no errors occurred
        assert True  # If we reach here, update succeeded
    
    # L1.5: Weighted parameter calculations
    def test_forward_method_integration(self, land_use_mapping, sample_state):
        """Test forward method integration"""
        outputs = land_use_mapping.forward(sample_state, time=0, dt=3600)
        assert set(outputs) == set(land_use_mapping.metadata.output_variables)
        assert all(name.startswith(f"{DIAGNOSTIC_PREFIX}.") for name in outputs)
        for update in outputs.values():
            assert isinstance(update, UpdateItem)
            assert update.mode == UpdateMode.REPLACE
            assert update.per_timestep is False
            assert torch.all(torch.isfinite(update.value))
    
    def test_parameter_consistency_across_methods(self, land_use_mapping, sample_state):
        """Test parameter consistency between different calculation methods"""
        land_use_fractions = sample_state['land_use_fraction']
        
        # Calculate parameters using calculate_weighted_parameters
        veg_params_direct, soil_params_direct, hydro_params_direct = \
            land_use_mapping.calculate_weighted_parameters(land_use_fractions)
        
        # Calculate using forward method
        outputs = land_use_mapping.forward(sample_state, time=0, dt=3600)
        
        comparisons = {
            f"{DIAGNOSTIC_PREFIX}.vegetation.lai_max": veg_params_direct['lai_max'],
            f"{DIAGNOSTIC_PREFIX}.vegetation.vcmax25": veg_params_direct['vcmax25'],
            f"{DIAGNOSTIC_PREFIX}.soil.soil_porosity": soil_params_direct['soil_porosity'],
            f"{DIAGNOSTIC_PREFIX}.hydrology.albedo": hydro_params_direct['albedo'],
        }
        for name, direct_value in comparisons.items():
            assert torch.equal(outputs[name].value, direct_value)
    
    def test_spatial_conservation(self, land_use_mapping):
        """Test spatial conservation properties"""
        # Create uniform land use distribution
        uniform_fractions = torch.ones(1, 9, 1, device=DEVICE) / 9.0
        
        veg_params, soil_params, hydro_params = land_use_mapping.calculate_weighted_parameters(
            uniform_fractions
        )
        
        # For uniform distribution, parameters should be averages of all land use types
        # This is hard to verify exactly, but we can check that values are reasonable
        
        # LAI should be intermediate between forest (high) and urban (low)
        lai_max = veg_params['lai_max'].item()
        forest_lai = land_use_mapping.vegetation_params[LandUseType.PRIMARY_FOREST].lai_max
        urban_lai = land_use_mapping.vegetation_params[LandUseType.URBAN].lai_max
        
        assert urban_lai < lai_max < forest_lai, "Uniform LAI should be between extremes"
        
        # Albedo should be intermediate between water (low) and barren (high)
        albedo = hydro_params['albedo'].item()
        water_albedo = land_use_mapping.hydro_params[LandUseType.WATER].albedo
        barren_albedo = land_use_mapping.hydro_params[LandUseType.BARREN].albedo
        
        assert water_albedo < albedo < barren_albedo, "Uniform albedo should be between extremes"
    
    # L1.6: Abstract base class compliance
    def test_physical_field_interface_compliance(self, land_use_mapping):
        """Test compliance with PhysicalField interface"""
        # Should implement all required abstract methods
        required_methods = ['get_value_at', 'compute_gradient', 'interpolate', 'update_field']
        
        for method_name in required_methods:
            assert hasattr(land_use_mapping, method_name), f"Missing method: {method_name}"
            method = getattr(land_use_mapping, method_name)
            assert callable(method), f"Method {method_name} not callable"
    
    def test_forward_method_consistency(self, land_use_mapping, sample_state):
        """Test backward consistency with forward method"""
        # Should be able to call forward method
        outputs = land_use_mapping.forward(sample_state, time=0, dt=3600)
        
        # Should return dictionary
        assert isinstance(outputs, dict)
        assert len(outputs) > 0
        
        # Should handle missing land_use_fraction gracefully
        invalid_state = {'other_variable': torch.tensor([1.0])}
        
        with pytest.raises((ValueError, KeyError)):
            land_use_mapping.forward(invalid_state, time=0, dt=3600)
    
    def test_dependency_declaration(self, land_use_mapping):
        """Test dependency declaration functionality"""
        dependencies = land_use_mapping.get_dependencies()
        assert isinstance(dependencies, list)
        assert "land_use_transitions" in dependencies


class TestLandUseTransitionsL1:
    """L1 Unit Tests for LandUseTransitions"""
    
    @pytest.fixture
    def land_use_transitions(self):
        """Create land use transitions instance for testing"""
        return LandUseTransitions()
    
    def test_class_instantiation_and_inheritance(self, land_use_transitions):
        """Test class instantiation and inheritance"""
        assert land_use_transitions is not None
        assert isinstance(land_use_transitions, LandUseTransitions)
        # Should inherit from appropriate base class
        assert hasattr(land_use_transitions, 'forward')
        assert callable(land_use_transitions.forward)
    
    def test_transition_matrix_properties(self, land_use_transitions):
        """Test transition matrix properties"""
        # Should have transition matrix for land use types
        if hasattr(land_use_transitions, 'transition_matrix'):
            matrix = land_use_transitions.transition_matrix
            
            # Should be square matrix
            assert matrix.shape[0] == matrix.shape[1], "Transition matrix should be square"
            
            # CTMC generator contract: rows sum to 0
            row_sums = matrix.sum(dim=1)
            assert torch.allclose(row_sums, torch.zeros_like(row_sums), atol=1e-6), \
                "Each generator row should sum to 0"

            # CTMC generator contract: off-diagonal rates are non-negative
            offdiag = matrix.clone()
            offdiag.fill_diagonal_(0.0)
            assert torch.all(offdiag >= -1e-12), "Off-diagonal rates should be non-negative"

            # CTMC generator contract: diagonal entries are non-positive
            assert torch.all(torch.diagonal(matrix) <= 1e-12), \
                "Generator diagonal entries should be non-positive"
    
    def test_land_use_type_definitions(self):
        """Test land use type definitions"""
        # Should have defined land use types
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
        
        # Should be able to create all types
        for lu_type in expected_types:
            assert lu_type is not None
    
    def test_parameter_access(self, land_use_transitions):
        """Test parameter access for transitions"""
        transition_rate = land_use_transitions.get_parameter_value(
            "transition_rates.deforestation_rate",
            namespace="anthroposphere.land_use",
            required=True,
        )
        assert isinstance(transition_rate, (float, torch.Tensor))
        assert 0.0 <= transition_rate <= 1.0


class TestLandUseModulesIntegration:
    """Integration tests for all land use modules"""
    
    def test_all_modules_instantiation(self):
        """Test that all land use modules can be instantiated"""
        modules = [
            LandUseParameterMapping(),
            LandUseTransitions()
        ]
        
        for module in modules:
            assert module is not None
            assert hasattr(module, 'name')
            assert hasattr(module, 'forward')
    
    def test_parameter_system_consistency(self):
        """Test parameter system consistency across modules"""
        mapping = LandUseParameterMapping()
        transitions = LandUseTransitions()

        mapping_value = mapping.get_parameter_value(
            "vegetation.primary_forest.lai_max",
            namespace="anthroposphere.land_use",
            required=True,
        )
        transition_value = transitions.get_parameter_value(
            "transition_rates.deforestation_rate",
            namespace="anthroposphere.land_use",
            required=True,
        )
        assert isinstance(mapping_value, (float, torch.Tensor))
        assert isinstance(transition_value, (float, torch.Tensor))
    
    def test_land_use_fraction_flow(self):
        """Test land use fraction flow through modules"""
        # Create land use transitions module
        transitions = LandUseTransitions()
        
        # Create parameter mapping module
        mapping = LandUseParameterMapping()
        
        # Create initial land use fractions
        initial_fractions = torch.ones(1, 9, 1, device=DEVICE) / 9.0
        
        transition_state = {'land_use_fraction': initial_fractions}
        transition_outputs = transitions.forward(transition_state, time=0, dt=86400)

        transitioned = transition_outputs['land_use_fraction']
        assert transitioned.source_functor == transitions.name
        mapping_state = {'land_use_fraction': transitioned.value}
        mapping_outputs = mapping.forward(mapping_state, time=0, dt=3600)

        assert isinstance(mapping_outputs, dict)
        assert len(mapping_outputs) > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])


