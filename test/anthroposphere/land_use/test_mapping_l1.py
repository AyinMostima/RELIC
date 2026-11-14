#!/usr/bin/env python3
"""
L1 Unit Tests for Land Use Mapping Module

Test Categories:
- L1.1: Class instantiation and initialization
- L1.2: Parameter management system validation
- L1.3: Scientific calculation accuracy (parameter mapping)
- L1.4: Conservation laws validation
- L1.5: Land use type transitions
- L1.6: Abstract base class compliance

Scientific References:
- Hurtt, G.C., et al. (2011). Harmonization of land-use scenarios. Climatic Change 109, 23-33.
- Lawrence, D.M., et al. (2016). The Land Use Model Intercomparison Project (LUMIP). Journal of Geophysical Research 121, 1636-1659.
- Foley, J.A., et al. (2005). Global consequences of land use. Science 309, 570-574.

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
    LandUseParameterMapping,
)
from relic.scientific.anthroposphere.land_use.transitions import LandUseType
from relic.core.abstractions import PhysicalField
from relic.core.functor import PhysicalFieldFunctor
from relic.core.functor.functor_types import VariableKind
from relic.core.state import UpdateItem, UpdateMode

# Test constants
DEVICE = torch.device("cpu")
TEST_BATCH_SIZE = 5
TOLERANCE = 1e-6
REL_TOLERANCE = 1e-3

DIAGNOSTIC_PREFIX = "diagnostics.land_use_parameter_mapping"
EXPECTED_UNITS = {
    **{
        f"{DIAGNOSTIC_PREFIX}.vegetation.{name}": unit
        for name, unit in {
            "lai_max": "dimensionless",
            "lai_min": "dimensionless",
            # Counted species per the D03 rule (Farquhar et al. 1980).
            "vcmax25": "umol CO2 m-2 s-1",
            "jmax25": "umol e- m-2 s-1",
            "root_depth": "m",
            "canopy_height": "m",
            "wood_density": "kg m-3",
            "sla": "m2 kg-1",
            "leaf_lifespan": "days",
            "c3_fraction": "dimensionless",
            "c4_fraction": "dimensionless",
        }.items()
    },
    **{
        f"{DIAGNOSTIC_PREFIX}.soil.{name}": unit
        for name, unit in {
            "decomposition_rate": "dimensionless",
            "nitrification_rate": "dimensionless",
            "denitrification_rate": "dimensionless",
            "soil_porosity": "dimensionless",
            "infiltration_rate": "mm hr-1",
            "soil_carbon_input": "kg C m-2 yr-1",
            "tillage_factor": "dimensionless",
            "erosion_rate": "kg m-2 yr-1",
        }.items()
    },
    **{
        f"{DIAGNOSTIC_PREFIX}.hydrology.{name}": unit
        for name, unit in {
            "surface_roughness": "m",
            "albedo": "dimensionless",
            "emissivity": "dimensionless",
            "interception_capacity": "mm",
            "runoff_coefficient": "dimensionless",
            "evaporation_resistance": "s m-1",
        }.items()
    },
    f"{DIAGNOSTIC_PREFIX}.canonical_iphs_runtime": "dimensionless",
}
EXPECTED_OUTPUTS = set(EXPECTED_UNITS)


def _value(update: UpdateItem) -> torch.Tensor:
    assert isinstance(update, UpdateItem)
    return update.value


class TestLandUseParameterMappingL1:
    """L1 Unit Tests for LandUseParameterMapping"""
    
    @pytest.fixture
    def land_use_mapper(self):
        """Create land use mapper instance for testing"""
        return LandUseParameterMapping()
    
    @pytest.fixture
    def sample_land_use_fractions(self):
        """Sample land use fractions for testing"""
        # Shape after normalization: [batch_size, n_land_use_types, 1]
        base = torch.tensor([
            # [primary_forest, secondary_forest, cropland_c3, cropland_c4,
            #  pasture, rangeland, urban, water, barren]
            [0.05, 0.08, 0.35, 0.12, 0.20, 0.10, 0.08, 0.03, 0.02],  # Mixed agricultural
            [0.08, 0.12, 0.40, 0.15, 0.15, 0.05, 0.08, 0.03, 0.02],  # More cropland
            [0.25, 0.20, 0.10, 0.05, 0.22, 0.12, 0.05, 0.02, 0.01],  # More forest
            [0.35, 0.25, 0.08, 0.03, 0.12, 0.10, 0.05, 0.02, 0.00],  # Forest dominant
            [0.10, 0.15, 0.30, 0.14, 0.18, 0.07, 0.07, 0.06, 0.03]  # Moderate agriculture
        ], device=DEVICE)
        normalized = base / base.sum(dim=1, keepdim=True)
        return {
            'land_use_fraction': normalized.reshape(TEST_BATCH_SIZE, 9, 1)
        }
    
    # L1.1: Class instantiation and initialization tests
    def test_class_instantiation(self, land_use_mapper):
        """Test that LandUseParameterMapping can be instantiated"""
        assert land_use_mapper is not None
        assert isinstance(land_use_mapper, LandUseParameterMapping)
        assert isinstance(land_use_mapper, PhysicalFieldFunctor)
        assert isinstance(land_use_mapper, PhysicalField)
        assert hasattr(land_use_mapper, 'name')
        assert land_use_mapper.name == "land_use_mapping"
    
    def test_metadata_completeness(self, land_use_mapper):
        """Metadata exposes only canonical diagnostic observations."""
        assert land_use_mapper.name == "land_use_mapping"
        assert land_use_mapper.metadata.input_variables == ["land_use_fraction"]
        assert set(land_use_mapper.metadata.output_variables) == EXPECTED_OUTPUTS
        assert land_use_mapper.metadata.variable_kind == VariableKind.DIAGNOSTIC
        assert not any(
            name in land_use_mapper.metadata.output_variables
            for name in {
                "lai_max",
                "vcmax25",
                "root_depth",
                "albedo",
                "soil_porosity",
                "hydro_infiltration_rate",
            }
        )
    
    def test_initialization_with_config(self):
        """A canonical string name remains configurable without alias parsing."""
        mapper = LandUseParameterMapping(name="test_mapping", config={})
        assert mapper.name == "test_mapping"
        with pytest.raises(TypeError, match="name must be a string"):
            LandUseParameterMapping(name={"name": "legacy_mapping"})
    
    def test_land_use_type_definitions(self, land_use_mapper):
        """Test that land use types are properly defined"""
        # Check that mapper has parameter tables for different land use types
        assert hasattr(land_use_mapper, 'vegetation_params')
        assert hasattr(land_use_mapper, 'soil_params')
        assert hasattr(land_use_mapper, 'hydro_params')

        # Should have standard land use types
        standard_types = [
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

        for land_use_type in standard_types:
            # Check that each land use type has parameters defined
            assert land_use_type in land_use_mapper.vegetation_params, \
                f"Land use {land_use_type} should have vegetation parameters"
            assert land_use_type in land_use_mapper.soil_params, \
                f"Land use {land_use_type} should have soil parameters"
            assert land_use_type in land_use_mapper.hydro_params, \
                f"Land use {land_use_type} should have hydrological parameters"
    
    # L1.2: Parameter management system validation
    def test_parameter_access_through_mapper(self, land_use_mapper):
        """Test that parameters are accessible through ParameterManager"""
        lai_max = land_use_mapper.get_parameter_value(
            "lai_max",
            namespace="anthroposphere.land_use.vegetation.primary_forest",
            required=True,
        )
        assert isinstance(lai_max, (float, torch.Tensor))
    
    def test_parameter_units_validation(self, land_use_mapper):
        """Test that parameter units are properly validated"""
        root_depth = land_use_mapper.get_parameter_value(
            "root_depth",
            namespace="anthroposphere.land_use.vegetation.primary_forest",
            units="m",
            validate=True,
            required=True,
        )
        assert isinstance(root_depth, (float, torch.Tensor))
    
    def test_parameter_bounds_checking(self, land_use_mapper):
        """Test that parameter bounds are enforced"""
        albedo = land_use_mapper.get_parameter_value(
            "albedo",
            namespace="anthroposphere.land_use.hydrological.primary_forest",
            bounds=(0.0, 1.0),
            required=True,
        )
        assert 0.0 <= float(albedo) <= 1.0

    def test_fraction_padding_and_normalization(self, land_use_mapper):
        """Invalid fractions fail fast instead of being padded or normalized."""
        with pytest.raises(ValueError, match="exactly 9"):
            land_use_mapper._validate_land_use_fraction(torch.tensor([0.6, 0.3, 0.1]))
        with pytest.raises(ValueError, match="sum to one"):
            land_use_mapper._validate_land_use_fraction(torch.full((9,), 0.2))
        invalid = torch.full((9,), 1.0 / 9.0)
        invalid[0] = -0.1
        with pytest.raises(ValueError, match="non-negative"):
            land_use_mapper._validate_land_use_fraction(invalid)
        non_finite = torch.full((9,), 1.0 / 9.0)
        non_finite[0] = torch.nan
        with pytest.raises(ValueError, match="finite"):
            land_use_mapper._validate_land_use_fraction(non_finite)
    
    # L1.3: Scientific calculation accuracy (parameter mapping)
    def test_vegetation_parameter_mapping(self, land_use_mapper, sample_land_use_fractions):
        """Test vegetation parameter mapping accuracy"""
        land_use_fractions = sample_land_use_fractions['land_use_fraction']

        # Calculate weighted parameters using actual method
        vegetation_params, soil_params, hydro_params = land_use_mapper.calculate_weighted_parameters(
            land_use_fractions
        )
        
        # Check that vegetation parameters are mapped
        assert isinstance(vegetation_params, dict)
        assert len(vegetation_params) > 0

        # Should have key vegetation parameters (using actual parameter names from implementation)
        expected_params = ['lai_max', 'vcmax25', 'root_depth', 'c3_fraction', 'c4_fraction']
        found_params = []

        for param_name in expected_params:
            if param_name in vegetation_params:
                found_params.append(param_name)
                param_value = vegetation_params[param_name]

                # Should be physically reasonable
                if param_name == 'lai_max':
                    assert torch.all(param_value >= 0), f"{param_name} should be non-negative"
                elif param_name == 'vcmax25':
                    assert torch.all(param_value >= 0), f"{param_name} should be non-negative"
                elif param_name == 'root_depth':
                    assert torch.all(param_value >= 0), f"{param_name} should be non-negative"
                elif param_name in ['c3_fraction', 'c4_fraction']:
                    assert torch.all(param_value >= 0) and torch.all(param_value <= 1), \
                        f"{param_name} should be fraction between 0 and 1"

        assert len(found_params) > 0, "Should have at least some vegetation parameters"
    
    def test_surface_property_mapping(self, land_use_mapper, sample_land_use_fractions):
        """Test surface property mapping accuracy"""
        land_use_fractions = sample_land_use_fractions['land_use_fraction']

        # Calculate weighted parameters using actual method
        vegetation_params, soil_params, hydro_params = land_use_mapper.calculate_weighted_parameters(
            land_use_fractions
        )

        # Use hydro_params as surface properties (they include albedo, roughness, etc.)
        surface_params = hydro_params
        
        # Check that surface properties are mapped
        assert isinstance(surface_params, dict)
        assert len(surface_params) > 0

        # Should have key surface parameters (using actual parameter names from implementation)
        expected_params = ['albedo', 'surface_roughness', 'emissivity', 'runoff_coefficient']
        found_params = []

        for param_name in expected_params:
            if param_name in surface_params:
                found_params.append(param_name)
                param_value = surface_params[param_name]

                # Should be physically reasonable
                if param_name == 'albedo':
                    assert torch.all(param_value >= 0.05) and torch.all(param_value <= 0.4), \
                        f"{param_name} should be in albedo range"
                elif param_name == 'surface_roughness':
                    assert torch.all(param_value >= 0.001) and torch.all(param_value <= 2.0), \
                        f"{param_name} should be in roughness range"
                elif param_name == 'emissivity':
                    assert torch.all(param_value >= 0.9) and torch.all(param_value <= 1.0), \
                        f"{param_name} should be in emissivity range"
                elif param_name == 'runoff_coefficient':
                    assert torch.all(param_value >= 0.0) and torch.all(param_value <= 1.0), \
                        f"{param_name} should be in runoff range"

        assert len(found_params) > 0, "Should have at least some surface parameters"
    
    def test_hydrological_parameter_mapping(self, land_use_mapper, sample_land_use_fractions):
        """Test hydrological parameter mapping accuracy"""
        land_use_fractions = sample_land_use_fractions['land_use_fraction']

        # Calculate weighted parameters using actual method
        vegetation_params, soil_params, hydro_params = land_use_mapper.calculate_weighted_parameters(
            land_use_fractions
        )
        
        # Check that hydrological parameters are mapped
        assert isinstance(hydro_params, dict)
        assert len(hydro_params) > 0

        # Should have key hydrological parameters (using actual parameter names from implementation)
        expected_params = ['surface_roughness', 'albedo', 'emissivity', 'runoff_coefficient', 'interception_capacity']
        found_params = []

        for param_name in expected_params:
            if param_name in hydro_params:
                found_params.append(param_name)
                param_value = hydro_params[param_name]

                # Should be physically reasonable
                if param_name == 'runoff_coefficient':
                    assert torch.all(param_value >= 0.0) and torch.all(param_value <= 1.0), \
                        f"{param_name} should be in runoff range"
                elif param_name == 'albedo':
                    assert torch.all(param_value >= 0.05) and torch.all(param_value <= 0.4), \
                        f"{param_name} should be in albedo range"
                elif param_name == 'surface_roughness':
                    assert torch.all(param_value >= 0.001) and torch.all(param_value <= 2.0), \
                        f"{param_name} should be in roughness range"
                elif param_name == 'emissivity':
                    assert torch.all(param_value >= 0.9) and torch.all(param_value <= 1.0), \
                        f"{param_name} should be in emissivity range"
                elif param_name == 'interception_capacity':
                    assert torch.all(param_value >= 0.0) and torch.all(param_value <= 5.0), \
                        f"{param_name} should be in interception capacity range"

        assert len(found_params) > 0, "Should have at least some hydrological parameters"
    
    def test_weighted_parameter_calculation(self, land_use_mapper):
        """Test weighted parameter calculation accuracy"""
        # Create test land use fractions for 9 land use types with proper shape
        fractions = torch.zeros(9, 1, 1, device=DEVICE)
        fractions[0, 0, 0] = 0.3  # primary_forest
        fractions[1, 0, 0] = 0.4  # secondary_forest
        fractions[2, 0, 0] = 0.3  # cropland_c3

        # Calculate weighted parameters using actual method
        veg_params, soil_params, hydro_params = land_use_mapper.calculate_weighted_parameters(fractions)

        # Check that parameters are calculated
        assert isinstance(veg_params, dict)
        assert isinstance(soil_params, dict)
        assert isinstance(hydro_params, dict)
        assert len(veg_params) > 0
        assert len(soil_params) > 0
        assert len(hydro_params) > 0

        # Check specific parameter values
        if 'lai_max' in veg_params:
            lai_value = veg_params['lai_max']
            assert torch.all(lai_value >= 0), "LAI should be non-negative"
            # Should be weighted average of forest, secondary forest, and crop values
            expected_lai = 0.3 * 7.0 + 0.4 * 5.0 + 0.3 * 4.0  # Approximate values
            assert torch.abs(lai_value.mean() - expected_lai) < 2.0, "LAI should be reasonable weighted average"
    
    def test_interpolation_algorithm(self, land_use_mapper):
        """Spatial interpolation requires an explicitly bound chain map."""
        source_coords = torch.tensor([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]
        ], device=DEVICE)
        target_coords = torch.tensor([
            [0.5, 0.5], [0.25, 0.75], [0.75, 0.25]
        ], device=DEVICE)
        source_values = torch.tensor([1.0, 2.0, 3.0, 4.0], device=DEVICE)
        with pytest.raises(RuntimeError, match="typed chain-map bridge"):
            land_use_mapper.interpolate(
                source_coords,
                target_coords,
                source_values=source_values,
            )
    
    # L1.4: Conservation laws validation
    def test_area_conservation_mapping(self, land_use_mapper, sample_land_use_fractions):
        """Test area conservation in parameter mapping"""
        # Verify that land use fractions sum to 1
        land_use_fractions = sample_land_use_fractions['land_use_fraction']
        
        # Should sum to 1 for each location
        sum_check = land_use_fractions.sum(dim=1)
        expected_sum = torch.ones_like(sum_check)
        
        assert torch.allclose(sum_check, expected_sum, rtol=1e-6), \
            "Land use fractions should sum to 1"
        
        # The mapping is an observation, not a mass or energy conservation law.
        state_dict = {'land_use_fraction': sample_land_use_fractions['land_use_fraction']}
        outputs = land_use_mapper.forward(state_dict, time=0, dt=3600)
        lai_key = f"{DIAGNOSTIC_PREFIX}.vegetation.lai_max"
        expected_lai = land_use_mapper.calculate_weighted_parameters(
            land_use_fractions
        )[0]["lai_max"]
        assert torch.equal(_value(outputs[lai_key]), expected_lai)
    
    def test_energy_balance_albedo(self, land_use_mapper, sample_land_use_fractions):
        """Test energy balance through albedo mapping"""
        # Use forward method to get parameters
        state_dict = {'land_use_fraction': sample_land_use_fractions['land_use_fraction']}
        surface_params = land_use_mapper.forward(state_dict, time=0, dt=3600)

        albedo_key = f"{DIAGNOSTIC_PREFIX}.hydrology.albedo"
        albedo_values = _value(surface_params[albedo_key])
        land_use_fractions = sample_land_use_fractions['land_use_fraction']
            
        assert torch.all(albedo_values >= 0.05), "Albedo should not be too low"
        assert torch.all(albedo_values <= 0.4), "Albedo should not be too high"
            
        # The diagnostic should reflect land-use composition.
            
        forest_fraction = land_use_fractions[:, 0, 0] + land_use_fractions[:, 1, 0]
        high_forest_mask = forest_fraction > 0.5
        if torch.any(high_forest_mask):
            assert torch.all(albedo_values.squeeze()[high_forest_mask] <= 0.2)
    
    def test_water_balance_runoff(self, land_use_mapper, sample_land_use_fractions):
        """Test water balance through runoff coefficient mapping"""
        # Use forward method to get parameters
        state_dict = {'land_use_fraction': sample_land_use_fractions['land_use_fraction']}
        hydrological_params = land_use_mapper.forward(state_dict, time=0, dt=3600)
        
        runoff_key = f"{DIAGNOSTIC_PREFIX}.hydrology.runoff_coefficient"
        runoff_values = _value(hydrological_params[runoff_key])
        assert torch.all(runoff_values >= 0.0)
        assert torch.all(runoff_values <= 0.8)
    
    # L1.5: Land use type transitions
    def test_land_use_transition_modeling(self, land_use_mapper):
        """Test land use transition modeling"""
        # Test dependencies - this class depends on land_use_transitions
        dependencies = land_use_mapper.get_dependencies()
        assert isinstance(dependencies, list)
        assert "land_use_transitions" in dependencies

        # Note: Transition matrix functionality is in transitions module, not here
        # This class only maps parameters based on given fractions
    
    def test_dynamic_land_use_mapping(self, land_use_mapper):
        """Test dynamic land use mapping over time"""
        # Create test state with different land use fractions
        initial_fractions = torch.zeros(9, 1, 1, device=DEVICE)
        initial_fractions[0, 0, 0] = 0.3  # primary_forest
        initial_fractions[1, 0, 0] = 0.2  # secondary_forest
        initial_fractions[2, 0, 0] = 0.4  # cropland_c3
        initial_fractions[3, 0, 0] = 0.1  # cropland_c4

        # Test forward method with different time steps
        state_dict = {'land_use_fraction': initial_fractions}

        # First time step
        outputs1 = land_use_mapper.forward(state_dict, time=0, dt=3600)
        assert isinstance(outputs1, dict)
        assert len(outputs1) > 0

        # Second time step (same fractions should give same outputs)
        outputs2 = land_use_mapper.forward(state_dict, time=3600, dt=3600)
        assert isinstance(outputs2, dict)
        assert len(outputs2) > 0

        assert set(outputs1) == EXPECTED_OUTPUTS
        assert set(outputs2) == EXPECTED_OUTPUTS
        for key in EXPECTED_OUTPUTS:
            assert torch.equal(_value(outputs1[key]), _value(outputs2[key]))
    
    def test_climate_land_use_feedbacks(self, land_use_mapper):
        """Test climate-land use feedback mechanisms"""
        # This class doesn't directly implement climate adjustments
        # It only maps parameters based on land use fractions
        # Climate effects would be handled by other modules

        # Test that the mapper works with different environmental conditions
        land_use_fractions = torch.zeros(9, 1, 1, device=DEVICE)
        land_use_fractions[0, 0, 0] = 0.3  # primary_forest
        land_use_fractions[1, 0, 0] = 0.3  # secondary_forest
        land_use_fractions[2, 0, 0] = 0.2  # cropland_c3
        land_use_fractions[3, 0, 0] = 0.1  # cropland_c4
        land_use_fractions[4, 0, 0] = 0.05 # pasture
        land_use_fractions[5, 0, 0] = 0.05 # rangeland
        state_dict = {'land_use_fraction': land_use_fractions}

        # Forward method should work regardless of environmental conditions
        outputs = land_use_mapper.forward(state_dict, time=0, dt=3600)
        assert isinstance(outputs, dict)
        assert len(outputs) > 0

        assert set(outputs) == EXPECTED_OUTPUTS
        for update in outputs.values():
            value = _value(update)
            assert torch.all(torch.isfinite(value))
            assert torch.all(value >= 0)
    
    def test_spatial_heterogeneity_handling(self, land_use_mapper):
        """Test handling of spatial heterogeneity"""
        spatial_fractions = torch.zeros((9, 3, 3), device=DEVICE)
        spatial_fractions[0, :, 0] = 1.0
        spatial_fractions[2, :, 1] = 1.0
        spatial_fractions[6, :, 2] = 1.0
        outputs = land_use_mapper.forward(
            {'land_use_fraction': spatial_fractions}, time=0, dt=3600
        )
        lai_key = f"{DIAGNOSTIC_PREFIX}.vegetation.lai_max"
        lai_values = _value(outputs[lai_key])
        assert lai_values.shape == (3, 3)
        assert torch.all(torch.isfinite(lai_values))
    
    # L1.6: Abstract base class compliance
    def test_physical_field_interface(self, land_use_mapper, sample_land_use_fractions):
        """Test compliance with PhysicalField interface"""
        # Should implement required PhysicalField methods
        assert hasattr(land_use_mapper, 'get_value_at'), "Missing get_value_at method"
        assert hasattr(land_use_mapper, 'compute_gradient'), "Missing compute_gradient method"
        assert hasattr(land_use_mapper, 'interpolate'), "Missing interpolate method"
        assert hasattr(land_use_mapper, 'update_field'), "Missing update_field method"
        
        # Methods should be callable
        assert callable(land_use_mapper.get_value_at), "get_value_at should be callable"
        assert callable(land_use_mapper.compute_gradient), "compute_gradient should be callable"
        assert callable(land_use_mapper.interpolate), "interpolate should be callable"
        assert callable(land_use_mapper.update_field), "update_field should be callable"
    
    def test_field_value_retrieval(self, land_use_mapper, sample_land_use_fractions):
        """Test that undeclared spatial field queries fail fast."""
        coordinates = torch.tensor([[0.0, 0.0], [0.5, 0.5]], device=DEVICE)
        with pytest.raises(RuntimeError, match="current spatial field"):
            land_use_mapper.get_value_at(coordinates)
    
    def test_gradient_calculation(self, land_use_mapper, sample_land_use_fractions):
        """Test that undeclared spatial gradient queries fail fast."""
        coordinates = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]], device=DEVICE)
        with pytest.raises(RuntimeError, match="current spatial field"):
            land_use_mapper.compute_gradient(coordinates)
    
    def test_interpolation_functionality(self, land_use_mapper):
        """A bound typed chain map is the only interpolation runtime."""
        coordinates = torch.tensor([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]
        ], device=DEVICE)
        values = torch.tensor([1.0, 2.0, 3.0, 4.0], device=DEVICE)
        target_coords = torch.tensor([
            [0.5, 0.5], [0.25, 0.75], [0.75, 0.25]
        ], device=DEVICE)

        def chain_map(source_coordinates, source_values, targets):
            assert torch.equal(source_coordinates, coordinates)
            return source_values.mean().expand(targets.shape[0]).clone()

        mapper = LandUseParameterMapping(field_query_bridge=chain_map)
        interpolated = mapper.interpolate(
            coordinates,
            target_coords,
            source_values=values,
        )
        assert torch.equal(interpolated, torch.full((3,), 2.5))
        with pytest.raises(TypeError, match="method"):
            mapper.interpolate(
                coordinates,
                target_coords,
                method="linear",
                source_values=values,
            )
    
    def test_forward_method_consistency(self, land_use_mapper, sample_land_use_fractions):
        """Forward exposes the exact diagnostic contract and update semantics."""
        outputs = land_use_mapper.forward(
            sample_land_use_fractions, time=0, dt=3600
        )
        assert set(outputs) == EXPECTED_OUTPUTS
        assert set(outputs) == set(land_use_mapper.metadata.output_variables)
        for name, update in outputs.items():
            assert isinstance(update, UpdateItem)
            assert update.mode == UpdateMode.REPLACE
            assert update.per_timestep is False
            assert update.unit == EXPECTED_UNITS[name]
            assert update.source_functor == land_use_mapper.name
            assert torch.isfinite(update.value).all()
    
    def test_dependency_declaration(self, land_use_mapper):
        """Test dependency declaration functionality"""
        dependencies = land_use_mapper.get_dependencies()
        assert dependencies == ["land_use_transitions"]
        assert not hasattr(land_use_mapper, "optional_dependencies")
    
    # Additional validation tests
    def test_mapping_parameter_values(self, land_use_mapper):
        """Runtime units are taken from the mapped ParameterManager specs."""
        assert land_use_mapper.mapped_parameter_units == {
            name: unit
            for name, unit in EXPECTED_UNITS.items()
            if not name.endswith("canonical_iphs_runtime")
        }
    
    def test_photosynthetic_capacity_labels_carry_the_counted_species(self, land_use_mapper):
        """Vcmax25 counts carboxylated CO2 and Jmax25 transported electrons.

        Farquhar et al. (1980) give the RuP2-saturated carboxylation rate in
        umol m-2 s-1 of CO2 and the electron transport J in ueq m-2 s-1; the
        C4 Rubisco capacity (Collatz et al. 1992) is also a CO2 carboxylation
        rate. Every land-use class carries the annotated labels of the biosphere
        Farquhar parameters of the same quantity (D03), and the mapper's
        cross-class unit check and emitted diagnostics use those labels.
        """
        from pathlib import Path

        from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
        from relic.data.standards.unit_converter import unit_converter

        certificate = load_iphs_equivalence_certificate(
            Path(__file__).resolve().parents[3] / "docs" / "source" / "theory"
            / "iphs_certificates" / "anthroposphere_land_use_mapping_land_use_parameter_mapping.yaml"
        )
        manager = land_use_mapper.get_parameter_manager()
        expected = {"vcmax25": "umol CO2 m-2 s-1", "jmax25": "umol e- m-2 s-1"}
        for name, unit in expected.items():
            assert manager.get_parameter(f"biosphere.photosynthesis.{name}").units == unit, name
            for land_use_type in LandUseType:
                namespace = f"anthroposphere.land_use.vegetation.{land_use_type.value}"
                key = f"{namespace}.{name}"
                assert manager.get_parameter(key).units == unit, key
                # A read that requests the annotated label is an identity read,
                # not a conversion between differently annotated labels.
                stored = land_use_mapper.get_parameter_value(
                    name, namespace=namespace, source_order=("manager",), required=True
                )
                annotated = land_use_mapper.get_parameter_value(
                    name, namespace=namespace, units=unit,
                    source_order=("manager",), required=True,
                )
                assert float(annotated) == float(stored), key
            diagnostic = f"{DIAGNOSTIC_PREFIX}.vegetation.{name}"
            # Runtime label, YAML, certificate and EXPECTED_UNITS agree.
            assert land_use_mapper.mapped_parameter_units[diagnostic] == unit, diagnostic
            assert certificate.unit_map[diagnostic] == unit, diagnostic
            assert EXPECTED_UNITS[diagnostic] == unit, diagnostic
            assert unit_converter.can_convert(unit, unit, diagnostic), diagnostic
        # The unit SSOT refuses to turn carboxylated CO2 into transported electrons.
        assert not unit_converter.can_convert(
            expected["vcmax25"], expected["jmax25"], f"{DIAGNOSTIC_PREFIX}.vegetation.vcmax25"
        )

    def test_c4_cropland_vcmax_cites_the_registered_collatz_1998_record(self, land_use_mapper):
        """Collatz, Berry and Clark (1998), Oecologia 114, 441-454, is DOI
        10.1007/s004420050468 (Crossref); the former DOI does not resolve."""
        spec = land_use_mapper.get_parameter_manager().get_parameter(
            "anthroposphere.land_use.vegetation.cropland_c4.vcmax25"
        )
        references = " ".join(spec.references)
        assert "10.1007/s004420050468" in references
        assert "j.1365-3040.1998.00345.x" not in references

    def test_land_use_database_completeness(self, land_use_mapper):
        """Test land use database completeness"""
        # Check that parameter tables are initialized
        assert hasattr(land_use_mapper, 'vegetation_params'), "Should have vegetation parameters"
        assert hasattr(land_use_mapper, 'soil_params'), "Should have soil parameters"
        assert hasattr(land_use_mapper, 'hydro_params'), "Should have hydrological parameters"

        # Check that standard land use types are covered
        standard_types = [
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

        for land_use_type in standard_types:
            assert land_use_type in land_use_mapper.vegetation_params, \
                f"Land use {land_use_type} should have vegetation parameters"
            assert land_use_type in land_use_mapper.soil_params, \
                f"Land use {land_use_type} should have soil parameters"
            assert land_use_type in land_use_mapper.hydro_params, \
                f"Land use {land_use_type} should have hydrological parameters"
    
    def test_edge_cases_and_error_handling(self, land_use_mapper):
        """Required input failures are explicit and never substituted."""
        zero_fractions = torch.zeros(1, 9, 1, device=DEVICE)
        with pytest.raises(ValueError, match="sum to one"):
            land_use_mapper.forward(
                {'land_use_fraction': zero_fractions}, time=0, dt=3600
            )
        with pytest.raises(KeyError, match="land_use_fraction"):
            land_use_mapper.forward(
                {'other_key': torch.tensor([1.0])}, time=0, dt=3600
            )
        with pytest.raises(KeyError, match="land_use_fraction"):
            land_use_mapper.forward(
                {'land_use_fractions': torch.full((9,), 1.0 / 9.0)},
                time=0,
                dt=3600,
            )
        with pytest.raises(TypeError, match="torch.Tensor"):
            land_use_mapper.forward(
                {'land_use_fraction': [1.0 / 9.0] * 9}, time=0, dt=3600
            )
    
    def test_device_consistency(self, land_use_mapper):
        """Test device consistency across operations"""
        # Test on CPU device (most reliable)
        device = torch.device('cpu')

        # Create test data on device
        device_fractions = torch.zeros(9, 1, 1, device=device)
        device_fractions[0, 0, 0] = 0.3  # primary_forest
        device_fractions[1, 0, 0] = 0.4  # secondary_forest
        device_fractions[2, 0, 0] = 0.3  # cropland_c3

        state_dict = {'land_use_fraction': device_fractions}

        # Test forward method
        outputs = land_use_mapper.forward(state_dict, time=0, dt=3600)

        for update in outputs.values():
            assert isinstance(update, UpdateItem)
            assert update.value.device == device

        # Test field operations
        coords = torch.tensor([[0.0, 0.0]], device=device)
        with pytest.raises(RuntimeError, match="current spatial field"):
            land_use_mapper.get_value_at(coords)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])


