"""
Integration tests for RELIC biosphere modules.

Tests the basic functionality and integration of key biosphere components:
- BiomassAllocation
- OzoneStress  
- FarquharPhotosynthesis
- PFT system

This verifies that modules can be instantiated, have proper dependencies,
and produce reasonable outputs.
"""

import torch

from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.biosphere.growth.biomass_allocation import BiomassAllocation
from relic.scientific.biosphere.stress.ozone_stress import OzoneStress
from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis


class TestBiosphereModulesIntegration:
    """Test integration between biosphere modules."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.batch_size = 1
        
        # Common state variables for testing
        self.base_state = {
            'temperature': torch.tensor(298.15, device=self.device),
            'air_temperature': torch.tensor(298.15, device=self.device),
            'par': torch.tensor(800.0, device=self.device), 
            'ppfd': torch.tensor(800.0, device=self.device),
            'atmospheric_co2_ppm': torch.tensor(400.0, device=self.device),
            'vapor_pressure_deficit': torch.tensor(1.5, device=self.device),
            'leaf_area_index': torch.tensor(4.0, device=self.device),
            'net_primary_productivity': torch.tensor(2.0e-7, device=self.device),
            'water_stress_factor': torch.tensor(0.8, device=self.device),
            'phenology_phase': torch.tensor(2.0, device=self.device),  # Mature phase
            'phenological_factor': torch.tensor(1.0, device=self.device),
            'o3_mixing_ratio': torch.tensor(50.0e-9, device=self.device),  # mol mol-1 (50 ppb)
            'stomatal_conductance': torch.tensor(0.2, device=self.device),  # mol m-2 s-1
            'vcmax_scaling': torch.tensor(1.0, device=self.device),  # dimensionless scaling factor
            'leaf_biomass_c': torch.tensor(0.10, device=self.device),
            'wood_biomass_c': torch.tensor(0.20, device=self.device),
            'root_biomass_c': torch.tensor(0.15, device=self.device),
        }
    
    def test_biomass_allocation_instantiation(self):
        """Test BiomassAllocation can be instantiated and configured."""
        allocation = BiomassAllocation(
            name="test_allocation",
            device=self.device,
            enable_environmental_control=True,
            enable_phenology_control=True
        )
        
        assert allocation is not None
        assert allocation.name == "test_allocation"
        assert allocation.device == self.device
        assert allocation.enable_environmental_control is True
        assert allocation.enable_phenology_control is True
        
        # Check dependencies
        deps = allocation.get_dependencies()
        assert deps == ['autotrophic_respiration', 'water_stress', 'gsi_phenology']
        assert 'farquhar_photosynthesis' not in deps
        required_inputs = allocation.get_required_inputs()
        assert set(required_inputs) == {
            'net_primary_productivity',
            'air_temperature',
            'water_stress_factor',
            'leaf_area_index',
            'phenology_phase',
        }
        assert set(allocation.metadata.input_variables) == {
            'net_primary_productivity',
            'air_temperature',
            'water_stress_factor',
            'leaf_area_index',
            'phenology_phase',
            'leaf_biomass_c',
            'wood_biomass_c',
            'root_biomass_c',
        }
        
        # Check provides
        provides = allocation.get_provides()
        assert 'growth_rate_leaf' in provides
        assert 'growth_rate_stem' in provides
        assert 'growth_rate_root' in provides
        
    def test_biomass_allocation_forward_pass(self):
        """Test BiomassAllocation forward pass produces valid outputs."""
        allocation = BiomassAllocation(
            name="test_allocation",
            device=self.device
        )
        
        result = allocation.forward(self.base_state, time=0.0, dt=86400.0)
        
        # Check required outputs exist
        assert 'growth_rate_leaf' in result
        assert 'growth_rate_stem' in result  
        assert 'growth_rate_root' in result
        assert set(result) == set(allocation.metadata.output_variables)
        
        # Check allocation fractions sum to 1
        f_leaf = result[
            'diagnostics.biomass_allocation.allocation_fraction_leaf'
        ].value
        f_stem = result[
            'diagnostics.biomass_allocation.allocation_fraction_stem'
        ].value
        f_root = result[
            'diagnostics.biomass_allocation.allocation_fraction_root'
        ].value
        total_fraction = f_leaf + f_stem + f_root
        torch.testing.assert_close(
            total_fraction,
            torch.ones_like(total_fraction),
            rtol=1.0e-6,
            atol=1.0e-7,
        )
        
        # Check growth rates are non-negative
        rate_keys = ('growth_rate_leaf', 'growth_rate_stem', 'growth_rate_root')
        stock_keys = ('leaf_biomass_c', 'wood_biomass_c', 'root_biomass_c')
        for key in rate_keys:
            assert isinstance(result[key], UpdateItem)
            assert result[key].mode is UpdateMode.REPLACE
            assert result[key].per_timestep is False
            assert torch.all(result[key].value >= 0)
        for key in stock_keys:
            assert isinstance(result[key], UpdateItem)
            assert result[key].mode is UpdateMode.ACCUMULATE
            assert result[key].per_timestep is True
            assert torch.all(result[key].value >= 0)
        
        # Check carbon balance
        total_growth = sum(result[key].value for key in rate_keys)
        total_increment = sum(result[key].value for key in stock_keys)
        npp = self.base_state['net_primary_productivity']
        torch.testing.assert_close(
            total_growth,
            npp.to(dtype=total_growth.dtype),
            rtol=1.0e-6,
            atol=1.0e-12,
        )
        torch.testing.assert_close(
            total_increment,
            (npp * 86400.0).to(dtype=total_increment.dtype),
            rtol=1.0e-6,
            atol=1.0e-9,
        )
        assert 'total_construction_respiration' not in result
        
    def test_ozone_stress_instantiation(self):
        """Test OzoneStress can be instantiated."""
        ozone_stress = OzoneStress(
            name="test_ozone_stress",
            device=self.device
        )
        
        assert ozone_stress is not None
        assert ozone_stress.name == "test_ozone_stress"
        assert ozone_stress.device == self.device
        
        # Check dependencies
        from relic.core.dependency_graph import normalize_dependency_declaration

        deps = normalize_dependency_declaration(
            ozone_stress.get_dependencies(), owner=ozone_stress.name
        )
        assert 'atmospheric_chemistry' in deps.optional
        # stomatal_conductance is a caller-supplied forcing: no provider is declared.
        assert 'farquhar_photosynthesis' not in deps.required + deps.optional + deps.lagged
        assert 'o3_mixing_ratio' in ozone_stress.metadata.input_variables
        
        # Check provides
        provides = ozone_stress.get_provides()
        assert 'ozone_stress_factor' in provides or 'rate' in provides
        
    def test_ozone_stress_forward_pass(self):
        """Test OzoneStress forward pass produces valid outputs."""
        ozone_stress = OzoneStress(
            name="test_ozone_stress", 
            device=self.device,

        )
        
        result = ozone_stress.forward(self.base_state, time=0.0, dt=86400.0)
        
        # Check output structure
        assert 'rate' in result or 'ozone_stress_factor' in result
        
        # Get stress factor
        if 'ozone_stress_factor' in result:
            stress_factor = result['ozone_stress_factor'].value
        else:
            stress_factor = result['rate']
            
        # Stress factor should be between 0 and 1
        assert torch.all(stress_factor >= 0.0)
        assert torch.all(stress_factor <= 1.0)
        
    def test_farquhar_photosynthesis_basic(self):
        """Test basic FarquharPhotosynthesis functionality.""" 
        farquhar = FarquharPhotosynthesis(
            name="test_farquhar",
            device=self.device
        )
        
        assert farquhar is not None
        assert farquhar.name == "test_farquhar"
        
        # Check dependencies
        deps = farquhar.get_dependencies()
        # Farquhar has its own dependency structure
        assert len(deps) > 0, f"Farquhar should have dependencies, got: {deps}"
        
        # Test forward pass
        result = farquhar.forward(self.base_state, time=0.0, dt=86400.0)
        
        # Check key outputs
        assert 'gross_photosynthesis' in result or 'rate' in result
        assert 'net_photosynthesis' in result or 'net_carbon_assimilation' in result
        
    def test_module_consistency_chain(self):
        """Test that modules can work together in a simple chain."""
        # Create modules
        ozone_stress = OzoneStress(
            name="ozone_stress",
            device=self.device,

        )
        
        allocation = BiomassAllocation(
            name="allocation", 
            device=self.device,
            enable_environmental_control=True
        )
        
        # Run ozone stress
        ozone_result = ozone_stress.forward(self.base_state, time=0.0, dt=86400.0)
        
        assert 'ozone_stress_factor' in ozone_result
        assert 'ozone_stress_factor' not in allocation.get_required_inputs()

        # Allocation consumes canonical NPP and stress-control ports, not ozone aliases.
        allocation_result = allocation.forward(
            self.base_state, time=0.0, dt=86400.0
        )
        
        # Verify results are reasonable
        assert 'growth_rate_leaf' in allocation_result
        assert isinstance(allocation_result['growth_rate_leaf'], UpdateItem)
        assert torch.all(allocation_result['growth_rate_leaf'].value >= 0)
        
        print("[OK] Module consistency chain test passed")
        
    def test_tensor_device_consistency(self):
        """Test that all modules respect device placement."""
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            
            # Move state to GPU
            gpu_state = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in self.base_state.items()}
            
            allocation = BiomassAllocation(
                name="gpu_allocation",
                device=device
            )
            
            result = allocation.forward(gpu_state, time=0.0, dt=86400.0)
            
            # Check outputs are on correct device
            for key, value in result.items():
                assert isinstance(value, UpdateItem)
                assert value.value.device == device, (
                    f"Output {key} on wrong device: {value.value.device}"
                )
                    
        print("[OK] Device consistency test passed")

