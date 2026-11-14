"""
L1 Unit Tests for Photosynthesis Parameter System

Tests parameter loading, unit consistency, and namespace resolution
for the photosynthesis module.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
from relic.parameters import get_global_manager
from relic.core.state import UpdateItem


class TestPhotosynthesisParameterLoading:
    """Test parameter loading from the unified parameter system."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        # Reset parameter manager before each test
        self.param_manager = get_global_manager()
        # Load parameter database
        self.param_manager.load_from_database()
        
    def test_parameter_units_consistency(self):
        """Test that parameter units are correctly specified."""
        # Check jmax25 units are correct (electron flow rate)
        jmax_spec = self.param_manager.get_parameter("biosphere.photosynthesis.jmax25")
        assert jmax_spec is not None, "jmax25 parameter not found"
        assert jmax_spec.units == "umol e- m-2 s-1", f"jmax25 units incorrect: {jmax_spec.units}"
        
        # Check alpha units are correct (quantum efficiency)
        alpha_spec = self.param_manager.get_parameter("biosphere.photosynthesis.alpha")
        assert alpha_spec is not None, "alpha parameter not found"
        assert alpha_spec.units == "dimensionless", f"alpha units incorrect: {alpha_spec.units}"
        
    def test_validation_parameters_exist(self):
        """Test that all validation parameters exist in the database."""
        validation_params = [
            "validation.numerical.epsilon_divide",
            "validation.numerical.clamp_min_zero",
            "validation.numerical.clamp_min_temperature",
            "validation.numerical.clamp_max_temperature",
            "validation.numerical.max_ppfd_validation",
            "validation.numerical.min_vcmax_scaling_validation",
            "validation.numerical.max_vcmax_scaling_validation",
            "validation.numerical.vpd_response_d0",
            "validation.numerical.c4_electron_factor"
        ]
        
        for param_name in validation_params:
            spec = self.param_manager.get_parameter(param_name)
            assert spec is not None, f"Validation parameter {param_name} not found"
            assert spec.value is not None, f"Validation parameter {param_name} has no value"

        assert self.param_manager.get_parameter(
            "validation.numerical.max_par_validation"
        ) is None

        vpd_d0 = self.param_manager.get_parameter(
            "validation.numerical.vpd_response_d0"
        )
        assert vpd_d0.units == "Pa"
        assert vpd_d0.value == pytest.approx(1500.0)
            
    def test_respiration_rd_q10_parameter(self):
        """Test that rd_q10 parameter exists in respiration namespace."""
        rd_q10_spec = self.param_manager.get_parameter("biosphere.respiration.rd_q10")
        assert rd_q10_spec is not None, "rd_q10 parameter not found in respiration namespace"
        assert rd_q10_spec.units == "dimensionless", f"rd_q10 units incorrect: {rd_q10_spec.units}"
        assert 1.5 <= rd_q10_spec.value <= 3.5, f"rd_q10 value out of bounds: {rd_q10_spec.value}"
        
    def test_photosynthesis_model_initialization(self):
        """Test that FarquharPhotosynthesis initializes with correct parameters."""
        model = FarquharPhotosynthesis()
        
        # Check that key parameters are loaded
        assert model.vcmax25 is not None, "vcmax25 not initialized"
        assert model.jmax25 is not None, "jmax25 not initialized"
        assert model.rd25 is not None, "rd25 not initialized"
        assert model.rd_q10 is not None, "rd_q10 not initialized"
        
        # Check parameter values are within expected ranges
        vcmax25_value = model.vcmax25.item() if hasattr(model.vcmax25, 'item') else model.vcmax25
        assert 10.0 <= vcmax25_value <= 200.0, f"vcmax25 out of range: {vcmax25_value}"
        
        jmax25_value = model.jmax25.item() if hasattr(model.jmax25, 'item') else model.jmax25
        assert 20.0 <= jmax25_value <= 400.0, f"jmax25 out of range: {jmax25_value}"
        
    def test_parameter_namespace_resolution(self):
        """Test that parameters are correctly resolved from namespaces."""
        model = FarquharPhotosynthesis()
        
        # Test getting parameter with namespace
        vcmax_value = model.get_parameter_value(
            "vcmax25",
            namespace="biosphere.photosynthesis"
        )
        assert vcmax_value is not None, "Failed to resolve vcmax25 from namespace"
        
        # Test getting validation parameter
        epsilon = model.get_parameter_value(
            "epsilon_divide",
            namespace="validation.numerical"
        )
        assert epsilon is not None, "Failed to resolve epsilon_divide from validation namespace"
        if isinstance(epsilon, torch.Tensor):
            epsilon_value = float(epsilon.detach().cpu().item())
        else:
            epsilon_value = float(epsilon)
        assert epsilon_value == pytest.approx(1e-10, rel=1e-10), (
            f"epsilon_divide value incorrect: {epsilon_value}"
        )
        
    def test_required_forcings_are_not_cached_defaults(self):
        """Required DAG ports must not be reconstructed from parameter defaults."""
        model = FarquharPhotosynthesis()

        assert {
            "default_temperature",
            "nan_co2_default",
            "nan_par_default",
            "default_vpd",
        }.isdisjoint(model._param_cache)
            
    def test_parameter_bounds_validation(self):
        """Test that parameter bounds are enforced."""
        # Test vcmax25 bounds
        vcmax_spec = self.param_manager.get_parameter("biosphere.photosynthesis.vcmax25")
        assert vcmax_spec.bounds is not None, "vcmax25 has no bounds"
        assert len(vcmax_spec.bounds) == 2, "vcmax25 bounds not properly specified"
        assert vcmax_spec.bounds[0] <= vcmax_spec.value <= vcmax_spec.bounds[1], \
            f"vcmax25 value {vcmax_spec.value} outside bounds {vcmax_spec.bounds}"
        
    def test_pft_parameter_loading(self):
        """Test PFT-specific parameter loading mechanism."""
        # This tests that the PFT loading mechanism doesn't crash
        # Even if PFT parameters don't exist, it should fall back gracefully
        try:
            model = FarquharPhotosynthesis(pft_name="test_pft")
            # Should not crash, even with non-existent PFT
            assert model.vcmax25 is not None, "vcmax25 not set with PFT loading"
        except Exception as e:
            pytest.fail(f"PFT parameter loading failed: {e}")
            
    def test_required_missing_parameter_fails_without_registration(self):
        """Scientific parameters must exist in YAML before runtime access."""
        model = FarquharPhotosynthesis()

        with pytest.raises(ValueError, match="Required parameter"):
            model.get_parameter_value(
                "farquhar_required_missing_parameter",
                namespace="test.namespace",
                required=True,
            )

        assert self.param_manager.get_parameter(
            "test.namespace.farquhar_required_missing_parameter"
        ) is None


class TestParameterIntegration:
    """Test integration between parameters and model computation."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.param_manager = get_global_manager()
        self.param_manager.load_from_database()
        
    def test_temperature_response_parameters(self):
        """Test that temperature response parameters are properly loaded."""
        model = FarquharPhotosynthesis()
        
        # Check activation energies are loaded
        assert hasattr(model, 'vcmax_ha'), "vcmax_ha not loaded"
        assert hasattr(model, 'jmax_ha'), "jmax_ha not loaded"
        assert hasattr(model, 'kc_ha'), "kc_ha not loaded"
        assert hasattr(model, 'ko_ha'), "ko_ha not loaded"
        
        # Check values are reasonable
        vcmax_ha = model.vcmax_ha.item() if hasattr(model.vcmax_ha, 'item') else model.vcmax_ha
        assert 50000 <= vcmax_ha <= 80000, f"vcmax_ha out of range: {vcmax_ha}"
        
    def test_photosynthesis_computation_with_parameters(self):
        """Test that photosynthesis computation works with loaded parameters."""
        model = FarquharPhotosynthesis(validate_inputs=True)
        
        # Create test state
        state = {
            'air_temperature': torch.tensor(298.15),
            'atmospheric_co2_ppm': torch.tensor(400.0),
            'ppfd': torch.tensor(1000.0),
            'leaf_area_index': torch.tensor(1.0),
            'vapor_pressure_deficit': torch.tensor(1000.0),
            'vcmax_scaling': torch.tensor(1.0),
        }
        
        # Run forward pass
        output = model.forward(state, time=0.0, dt=1.0)

        assert set(output) == set(model.metadata.output_variables)
        assert all(isinstance(item, UpdateItem) for item in output.values())
        gross = output['gross_photosynthesis'].value.item()
        assert 0 <= gross <= 100, f"Gross photosynthesis out of range: {gross}"

        net = output['net_photosynthesis'].value.item()
        assert -10 <= net <= 100, f"Net photosynthesis out of range: {net}"
            
    def test_parameter_scaling_factors(self):
        """Test that scaling factors work correctly."""
        model = FarquharPhotosynthesis()
        
        # Test with different scaling factors
        state_base = {
            'air_temperature': torch.tensor(298.15),
            'atmospheric_co2_ppm': torch.tensor(400.0),
            'ppfd': torch.tensor(2000.0),
            'leaf_area_index': torch.tensor(1.0),
            'vapor_pressure_deficit': torch.tensor(1000.0),
            'vcmax_scaling': torch.tensor(1.0),
        }
        
        state_scaled = state_base.copy()
        state_scaled['vcmax_scaling'] = torch.tensor(0.5)
        
        output_base = model.forward(state_base, time=0.0, dt=1.0)
        output_scaled = model.forward(state_scaled, time=0.0, dt=1.0)
        
        # Scaled photosynthesis should be lower
        gpp_base = output_base['gross_photosynthesis'].value.item()
        gpp_scaled = output_scaled['gross_photosynthesis'].value.item()
        
        assert gpp_scaled < gpp_base, \
            f"Scaled GPP ({gpp_scaled}) not less than base GPP ({gpp_base})"
        
    def test_nan_required_forcing_fails_fast(self):
        """NaN required forcings must not be replaced by parameter defaults."""
        model = FarquharPhotosynthesis()
        
        # Create state with NaN values
        state = {
            'air_temperature': torch.tensor(float('nan')),
            'atmospheric_co2_ppm': torch.tensor(400.0),
            'ppfd': torch.tensor(1000.0),
            'leaf_area_index': torch.tensor(1.0),
            'vapor_pressure_deficit': torch.tensor(1000.0),
        }

        with pytest.raises(
            ValueError, match="air_temperature must contain only finite values"
        ):
            model.forward(state, time=0.0, dt=1.0)
        
    def test_validation_parameter_usage(self):
        """Test that validation parameters are actually used in computations."""
        model = FarquharPhotosynthesis(validate_inputs=True)
        
        # Test with an extreme PPFD value.
        state = {
            'air_temperature': torch.tensor(298.15),
            'atmospheric_co2_ppm': torch.tensor(400.0),
            'ppfd': torch.tensor(5000.0),
            'leaf_area_index': torch.tensor(1.0),
            'vapor_pressure_deficit': torch.tensor(1000.0),
        }
        
        # The ParameterManager-backed PPFD ceiling must fail fast.
        with pytest.raises(ValueError, match="ppfd exceeds supported maximum"):
            model.forward(state, time=0.0, dt=1.0)
            
    def test_c4_pathway_parameters(self):
        """Test that C4 pathway parameters are loaded correctly."""
        model = FarquharPhotosynthesis(config={'pathway': 'C4'})
        
        # Check C4-specific parameters are loaded
        assert hasattr(model, 'vpmax25'), "vpmax25 not loaded for C4"
        assert hasattr(model, 'kp25'), "kp25 not loaded for C4"
        assert hasattr(model, 'phi'), "phi not loaded for C4"
        
        # Check values are reasonable
        if hasattr(model.vpmax25, 'item'):
            vpmax25 = model.vpmax25.item()
        else:
            vpmax25 = model.vpmax25
        assert 20 <= vpmax25 <= 150, f"vpmax25 out of range: {vpmax25}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])

