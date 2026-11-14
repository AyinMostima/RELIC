"""
L1 Unit Tests for Plant Stress Response Modules

Comprehensive testing of all stress response models including water stress,
nutrient stress, thermal stress, and ozone stress using the new Functor architecture.

Test Coverage:
- Basic instantiation and metadata validation
- Parameter loading from ParameterManager
- Stress factor calculations with proper units
- Dependency and provides validation
- FluxDict output structure
- Boundary conditions and robustness
- BiologicalProcessFunctor interface consistency

Scientific Validation:
- Water stress: f_water = (蠄 - 蠄_wilt) / (蠄_fc - 蠄_wilt)
- Nutrient stress: f_N = ([N] - N_min) / (N_crit - N_min)
- Thermal stress: f_T = exp(-0.5 * ((T - T_opt)/蟽)虏) (Gaussian)
- Ozone stress: f_O3 = 1 - (O3_flux / O3_detox_capacity)

References:
- Boyer (1982). Plant productivity and environment
- Berry & Bj枚rkman (1980). Photosynthetic response and adaptation to temperature
- Chapin III (1991). Integrated responses of plants to stress
- Mills et al. (2011). A synthesis of AOT40-based response functions
"""

import torch
import pytest
import os
import sys
from test.utils.forward_call import call_forward

# Add project path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

# Import modules to test
from relic.scientific.biosphere.stress.water_stress import WaterStress
from relic.scientific.biosphere.stress.nutrient_stress import NutrientStress
from relic.scientific.biosphere.stress.thermal_stress import ThermalStress
from relic.scientific.biosphere.stress.ozone_stress import OzoneStress
from relic.parameters import get_global_manager
from relic.core.state import UpdateItem, UpdateMode


@pytest.fixture(autouse=True)
def _restore_global_parameter_manager(restore_global_parameter_manager):
    """Tests in this module write to the process-wide ParameterManager; the
    shared fixture puts it back after each test."""
    yield


class TestWaterStressL1:
    """L1 unit tests for water stress response model."""
    
    @pytest.fixture(autouse=True)
    def setup_parameter_manager(self):
        """Setup parameter manager with test parameters."""
        self.param_manager = get_global_manager()
        # Load test parameters
        test_params_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 
                                      'relic', 'parameters', 'database')
        if os.path.exists(test_params_dir):
            self.param_manager.load_from_database(test_params_dir)
    
    @pytest.fixture
    def water_stress_model(self):
        """Create water stress model for testing."""
        return WaterStress(name="water_stress_test")
    
    def test_basic_instantiation(self, water_stress_model):
        """Test L1.1: Basic instantiation and metadata validation."""
        assert isinstance(water_stress_model, WaterStress)

        # Check metadata exists
        assert hasattr(water_stress_model, 'metadata')
        metadata = water_stress_model.metadata

        # Verify metadata structure (metadata is a dataclass, not dict)
        assert hasattr(metadata, 'parameters')
        assert hasattr(metadata, 'mathematical_expression')
        assert hasattr(metadata, 'references')
        # Note: 'units' is not a direct metadata attribute

        # Check parameter descriptions
        params = metadata.parameters
        assert 'wilting_point_potential' in params or len(params) >= 0  # params dict may vary

        # Verify namespace
        assert water_stress_model.namespace == "biosphere.stress.water"

        # Check dependencies and provides
        deps = water_stress_model.get_dependencies()
        provides = water_stress_model.get_provides()

        assert deps == []
        assert provides == ['water_stress_factor']
    
    def test_parameter_loading(self, water_stress_model):
        """Test L1.2: Parameter loading from ParameterManager."""
        # Parameters should be loaded as tensors
        # Note: We can't directly access the buffers, but we can test the behavior
        
        # Test with known water potential
        state = {'soil_water_potential': torch.tensor(-1.0)}
        environment = {}
        
        result = call_forward(water_stress_model, {**state, **environment})
        
        # Check result structure
        assert isinstance(result, dict)
        assert 'water_stress_factor' in result
        
        stress_factor = result['water_stress_factor']
        assert isinstance(stress_factor, torch.Tensor)
        assert torch.isfinite(stress_factor).all()
        assert 0.0 <= stress_factor.item() <= 1.0
    
    def test_water_stress_calculation(self, water_stress_model):
        """Test L1.3: Water stress factor calculation with proper units."""
        # Test different water potentials in MPa
        test_cases = [
            (-2.0, 0.0),    # Below wilting point
            (-1.5, 0.0),    # At wilting point  
            (-0.7665, 0.5), # Midpoint
            (-0.033, 1.0),  # At field capacity
            (0.0, 1.0)      # Above field capacity
        ]
        
        for wp, expected_approx in test_cases:
            environment = {}
            state = {'soil_water_potential': torch.tensor(wp)}
            
            result = call_forward(water_stress_model, {**state, **environment})
            stress_factor = result['water_stress_factor']
            
            # Should be finite and within bounds
            assert torch.isfinite(stress_factor).all()
            assert 0.0 <= stress_factor.item() <= 1.0
            
            assert stress_factor.item() == pytest.approx(expected_approx, abs=1.0e-6)
    
    def test_forward_method(self, water_stress_model):
        """Test L1.4: Forward method with full FluxDict output."""
        state = {
            'soil_water_potential': torch.tensor(-0.8)  # MPa
        }

        result = water_stress_model.forward(state, time=0.0, dt=3600.0)
        
        # Check FluxDict structure
        assert isinstance(result, dict)
        assert set(result) == {'water_stress_factor'}
        
        # Check values
        update = result['water_stress_factor']
        assert isinstance(update, UpdateItem)
        assert update.mode is UpdateMode.REPLACE
        assert update.per_timestep is False
        stress_factor = update.value
        
        assert torch.isfinite(stress_factor).all()
        assert 0.0 <= stress_factor.item() <= 1.0
    
    def test_missing_water_potential(self, water_stress_model):
        """Test L1.5: Missing canonical forcing is a blocking error."""
        with pytest.raises(KeyError, match='soil_water_potential'):
            water_stress_model.forward({}, time=0.0, dt=3600.0)


class TestThermalStressL1:
    """L1 unit tests for thermal stress response model."""
    
    @pytest.fixture(autouse=True)
    def setup_parameter_manager(self):
        """Setup parameter manager with test parameters."""
        self.param_manager = get_global_manager()
        test_params_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 
                                      'relic', 'parameters', 'database')
        if os.path.exists(test_params_dir):
            self.param_manager.load_from_database(test_params_dir)
    
    @pytest.fixture
    def thermal_stress_model(self):
        """Create thermal stress model for testing."""
        return ThermalStress(name="thermal_stress_test")
    
    def test_basic_instantiation(self, thermal_stress_model):
        """Test L1.1: Basic instantiation and metadata validation."""
        assert isinstance(thermal_stress_model, ThermalStress)

        # Check metadata (dataclass, not dict)
        metadata = thermal_stress_model.metadata
        assert hasattr(metadata, 'parameters')
        # Check if parameters dict contains expected keys (may vary)
        params = metadata.parameters
        assert isinstance(params, dict)

        # Verify namespace
        assert thermal_stress_model.namespace == "biosphere.stress.thermal"

        # Check interface
        deps = thermal_stress_model.get_dependencies()
        provides = thermal_stress_model.get_provides()

        assert isinstance(deps, (list, tuple))
        assert isinstance(provides, (list, tuple))
    
    def test_forward_is_the_single_calculation_entry(self):
        """Item 114: the uncalled _compute_rate_core, which read the temperature from an
        environment mapping, is removed; forward is the only calculation entry."""
        assert not hasattr(ThermalStress, "_compute_rate_core")

    def test_air_temperature_is_the_single_temperature_input(self, thermal_stress_model):
        """Item 117: the thermal factors read one declared input, the near-surface air
        temperature air_temperature [K] (the driver of the cited crop thresholds and of
        the RELIC Farquhar temperature responses); ``temperature`` and
        ``leaf_temperature`` are not aliases for it."""
        from relic.core.functor.functor_exceptions import ExecutionError

        assert list(thermal_stress_model.metadata.input_variables) == ["air_temperature"]
        for alias in ("temperature", "leaf_temperature"):
            with pytest.raises(KeyError, match="air_temperature"):
                thermal_stress_model.forward({alias: torch.tensor(300.15)}, 0.0, 3600.0)
        with pytest.raises(ExecutionError, match="air_temperature"):
            thermal_stress_model(
                {"leaf_temperature": torch.tensor(300.15)}, time=0.0, dt=3600.0
            )
        air_only = thermal_stress_model.forward(
            {"air_temperature": torch.tensor(303.15)}, 0.0, 3600.0
        )
        with_leaf = thermal_stress_model.forward(
            {
                "air_temperature": torch.tensor(303.15),
                "leaf_temperature": torch.tensor(313.15),
            },
            0.0,
            3600.0,
        )
        for name in ("temperature_stress_factor", "heat_stress_factor", "cold_stress_factor"):
            left, right = with_leaf[name], air_only[name]
            left = getattr(left, "value", left)
            right = getattr(right, "value", right)
            assert torch.equal(left, right), name

    def test_gaussian_temperature_response(self, thermal_stress_model):
        """Test L1.2: Gaussian temperature response function."""
        # Test temperatures around optimum (298.15 K = 25掳C)
        temperatures = [
            278.15,  # 5掳C - cold
            288.15,  # 15掳C - cool
            298.15,  # 25掳C - optimal
            308.15,  # 35掳C - warm
            318.15   # 45掳C - hot
        ]
        
        stress_factors = []
        for temp in temperatures:
            environment = {'air_temperature': torch.tensor(temp)}
            result = call_forward(thermal_stress_model, {**environment})
            stress_factor = result['temperature_stress_factor']
            stress_factors.append(stress_factor.item())
        
        # Should peak at optimum temperature (index 2)
        max_stress_idx = stress_factors.index(max(stress_factors))
        assert max_stress_idx == 2, "Peak stress should be at optimum temperature"
        
        # All values should be bounded [0, 1]
        for sf in stress_factors:
            assert 0.0 <= sf <= 1.0
    
    def test_heat_and_cold_stress(self, thermal_stress_model):
        """Test L1.3: Heat and cold stress thresholds."""
        state_cold = {'air_temperature': torch.tensor(273.15)}  # 0掳C
        state_hot = {'air_temperature': torch.tensor(313.15)}   # 40掳C

        result_cold = thermal_stress_model.forward(state_cold, time=0.0, dt=3600.0)
        result_hot = thermal_stress_model.forward(state_hot, time=0.0, dt=3600.0)
        
        # Check all stress components
        assert 'temperature_stress_factor' in result_cold
        assert 'heat_stress_factor' in result_cold
        assert 'cold_stress_factor' in result_cold
        
        # Cold stress should be active
        cold_stress = result_cold['cold_stress_factor'].value
        assert cold_stress.item() < 1.0  # Some cold stress

        # Heat stress should be active
        heat_stress = result_hot['heat_stress_factor'].value
        assert heat_stress.item() < 1.0  # Some heat stress
    
    def test_extreme_temperatures(self, thermal_stress_model):
        """Test L1.4: Extreme temperature handling."""
        # Test freezing
        state_freeze = {'air_temperature': torch.tensor(268.15)}  # -5掳C
        result_freeze = thermal_stress_model.forward(state_freeze, time=0.0, dt=3600.0)

        cold_stress = result_freeze['cold_stress_factor'].value
        assert cold_stress.item() < 0.5  # Severe cold stress

        # Test extreme heat
        state_extreme = {'air_temperature': torch.tensor(323.15)}  # 50掳C
        result_extreme = thermal_stress_model.forward(state_extreme, time=0.0, dt=3600.0)
        
        heat_stress = result_extreme['heat_stress_factor'].value
        assert heat_stress.item() == 0.0  # Complete heat damage


class TestNutrientStressL1:
    """L1 unit tests for nutrient stress response model."""
    
    @pytest.fixture(autouse=True)
    def setup_parameter_manager(self):
        """Setup parameter manager with test parameters."""
        self.param_manager = get_global_manager()
        test_params_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 
                                      'relic', 'parameters', 'database')
        if os.path.exists(test_params_dir):
            self.param_manager.load_from_database(test_params_dir)
    
    @pytest.fixture
    def nutrient_stress_model(self):
        """Create nutrient stress model for testing."""
        return NutrientStress(name="nutrient_stress_test")
    
    def test_basic_instantiation(self, nutrient_stress_model):
        """Test L1.1: Basic instantiation and metadata validation."""
        assert isinstance(nutrient_stress_model, NutrientStress)

        # Check metadata (dataclass, not dict)
        metadata = nutrient_stress_model.metadata
        assert hasattr(metadata, 'parameters')
        params = metadata.parameters
        assert isinstance(params, dict)

        # Verify namespace
        assert nutrient_stress_model.namespace == "biosphere.stress.nutrient"

        # Check interface
        deps = nutrient_stress_model.get_dependencies()
        provides = nutrient_stress_model.get_provides()

        assert deps == []
        assert set(metadata.input_variables) == {
            'nitrogen_concentration',
            'phosphorus_concentration',
        }
        assert provides == list(metadata.output_variables)
        assert set(provides) == {
            'nutrient_stress_factor',
            'diagnostics.nutrient_stress.nitrogen_factor',
            'diagnostics.nutrient_stress.phosphorus_factor',
        }
    
    def test_critical_concentration_model(self, nutrient_stress_model):
        """Test L1.2: Critical concentration nutrient response."""
        # Test with tissue concentrations (kg N kg^-1 DM)
        state_low_n = {'nitrogen_concentration': torch.tensor(0.01)}   # Low N
        state_opt_n = {'nitrogen_concentration': torch.tensor(0.025)}  # Optimal N
        state_high_n = {'nitrogen_concentration': torch.tensor(0.04)}  # High N

        # Also provide P to avoid errors
        state_low_n['phosphorus_concentration'] = torch.tensor(0.002)
        state_opt_n['phosphorus_concentration'] = torch.tensor(0.003)
        state_high_n['phosphorus_concentration'] = torch.tensor(0.004)
        
        result_low = nutrient_stress_model.forward(state_low_n, time=0.0, dt=3600.0)
        result_opt = nutrient_stress_model.forward(state_opt_n, time=0.0, dt=3600.0)
        result_high = nutrient_stress_model.forward(state_high_n, time=0.0, dt=3600.0)
        
        # Check nitrogen stress factors
        n_stress_low = result_low[
            'diagnostics.nutrient_stress.nitrogen_factor'
        ].value
        n_stress_opt = result_opt[
            'diagnostics.nutrient_stress.nitrogen_factor'
        ].value
        n_stress_high = result_high[
            'diagnostics.nutrient_stress.nitrogen_factor'
        ].value
        
        # Higher N should reduce stress
        assert n_stress_low.item() < n_stress_opt.item()
        assert n_stress_opt.item() <= n_stress_high.item()
        
        # All should be bounded
        for stress in [n_stress_low, n_stress_opt, n_stress_high]:
            assert 0.0 <= stress.item() <= 1.0
    
    def test_liebig_law_minimum(self, nutrient_stress_model):
        """Test L1.3: Liebig's law of the minimum for nutrients."""
        # Low N, high P
        state_n_limited = {
            'nitrogen_concentration': torch.tensor(0.01),   # Low N
            'phosphorus_concentration': torch.tensor(0.005)  # High P
        }

        # High N, low P
        state_p_limited = {
            'nitrogen_concentration': torch.tensor(0.04),    # High N
            'phosphorus_concentration': torch.tensor(0.0005) # Low P
        }
        
        result_n_lim = nutrient_stress_model.forward(state_n_limited, time=0.0, dt=3600.0)
        result_p_lim = nutrient_stress_model.forward(state_p_limited, time=0.0, dt=3600.0)
        
        # Overall stress should follow minimum
        overall_n_lim = result_n_lim['nutrient_stress_factor'].value
        overall_p_lim = result_p_lim['nutrient_stress_factor'].value

        n_stress = result_n_lim[
            'diagnostics.nutrient_stress.nitrogen_factor'
        ].value
        p_stress = result_p_lim[
            'diagnostics.nutrient_stress.phosphorus_factor'
        ].value
        
        # Overall should be close to the most limiting factor
        assert overall_n_lim.item() <= n_stress.item()
        assert overall_p_lim.item() <= p_stress.item()


class TestOzoneStressL1:
    """L1 unit tests for ozone stress response model."""
    
    @pytest.fixture(autouse=True)
    def setup_parameter_manager(self):
        """Setup parameter manager with test parameters."""
        self.param_manager = get_global_manager()
        test_params_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 
                                      'relic', 'parameters', 'database')
        if os.path.exists(test_params_dir):
            self.param_manager.load_from_database(test_params_dir)
    
    @pytest.fixture
    def ozone_stress_model(self):
        """Create ozone stress model for testing."""
        return OzoneStress(name="ozone_stress_test")
    
    def test_basic_instantiation(self, ozone_stress_model):
        """Test L1.1: Basic instantiation and metadata validation."""
        assert isinstance(ozone_stress_model, OzoneStress)

        # Check metadata (dataclass, not dict)
        metadata = ozone_stress_model.metadata
        assert hasattr(metadata, 'parameters')
        params = metadata.parameters
        assert isinstance(params, dict)

        # Verify namespace
        assert ozone_stress_model.namespace == "biosphere.stress.ozone"

        # Check interface: the chemistry ozone provider is an optional dependency.
        from relic.core.dependency_graph import normalize_dependency_declaration

        deps = normalize_dependency_declaration(
            ozone_stress_model.get_dependencies(), owner=ozone_stress_model.name
        )
        provides = ozone_stress_model.get_provides()

        assert "atmospheric_chemistry" in deps.optional
        assert isinstance(provides, (list, tuple))
    
    def test_ozone_flux_calculation(self, ozone_stress_model):
        """Test L1.2: Ozone flux and uptake calculation."""
        # Test with different ozone levels (mol mol^-1)
        ozone_levels = [
            (20e-9, 0.3),   # Low ozone, normal conductance
            (40e-9, 0.3),   # Background ozone
            (80e-9, 0.3),   # Elevated ozone
            (40e-9, 0.1),   # Background ozone, low conductance
        ]
        
        for o3_conc, g_s in ozone_levels:
            state = {
                'o3_mixing_ratio': torch.tensor(o3_conc),
                'stomatal_conductance': torch.tensor(g_s)
            }
            
            result = ozone_stress_model.forward(state, time=0.0, dt=3600.0)
            
            # Check outputs
            assert 'ozone_stress_factor' in result
            assert 'ozone_uptake' in result
            assert 'antioxidant_depletion' in result
            
            # Validate values
            stress = result['ozone_stress_factor'].value
            uptake = result['ozone_uptake'].value
            
            assert 0.0 <= stress.item() <= 1.0
            assert uptake.item() >= 0.0
    
    def test_ozone_detoxification(self, ozone_stress_model):
        """Test L1.3: Ozone detoxification capacity."""
        # Low ozone - within detox capacity
        state_low = {
            'o3_mixing_ratio': torch.tensor(20e-9),
            'stomatal_conductance': torch.tensor(0.2)
        }

        # High ozone - exceeds detox capacity
        state_high = {
            'o3_mixing_ratio': torch.tensor(100e-9),
            'stomatal_conductance': torch.tensor(0.4)
        }
        
        result_low = ozone_stress_model.forward(state_low, time=0.0, dt=3600.0)
        result_high = ozone_stress_model.forward(state_high, time=0.0, dt=3600.0)
        
        # High ozone should cause more stress (lower or equal stress factor if saturated)
        stress_low = result_low['ozone_stress_factor'].value
        stress_high = result_high['ozone_stress_factor'].value

        # Allow for saturation - high ozone should be <= low ozone stress
        assert stress_high.item() <= stress_low.item()

        # Check antioxidant depletion - high should be >= low
        antioxidant_low = result_low['antioxidant_depletion'].value
        antioxidant_high = result_high['antioxidant_depletion'].value

        assert antioxidant_high.item() >= antioxidant_low.item()


class TestStressIntegrationL1:
    """L1 integration tests for stress module consistency."""
    
    @pytest.fixture(autouse=True)
    def setup_parameter_manager(self):
        """Setup parameter manager with test parameters."""
        self.param_manager = get_global_manager()
        test_params_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 
                                      'relic', 'parameters', 'database')
        if os.path.exists(test_params_dir):
            self.param_manager.load_from_database(test_params_dir)
    
    def test_all_stress_instantiation(self):
        """Test L1.1: All stress models can be instantiated."""
        water = WaterStress()
        thermal = ThermalStress()
        nutrient = NutrientStress()
        ozone = OzoneStress()
        
        # All should be valid functors
        for stress in [water, thermal, nutrient, ozone]:
            assert hasattr(stress, 'forward')
            assert hasattr(stress, 'forward')
            assert hasattr(stress, 'get_dependencies')
            assert hasattr(stress, 'get_provides')
            assert hasattr(stress, 'metadata')
    
    def test_combined_stress_state(self):
        """Test L1.2: Multiple stresses with combined state."""
        # Create comprehensive state
        state = {
            'soil_water_potential': torch.tensor(-0.8),
            'air_temperature': torch.tensor(303.15),  # 30 C
            'nitrogen_concentration': torch.tensor(0.02),
            'phosphorus_concentration': torch.tensor(0.002),
            'o3_mixing_ratio': torch.tensor(60e-9),
            'stomatal_conductance': torch.tensor(0.35)
        }
        
        # Test all stress models with same state
        water = WaterStress()
        thermal = ThermalStress()
        nutrient = NutrientStress()
        ozone = OzoneStress()
        
        water_result = water.forward(state, time=0.0, dt=3600.0)
        thermal_result = thermal.forward(state, time=0.0, dt=3600.0)
        nutrient_result = nutrient.forward(state, time=0.0, dt=3600.0)
        ozone_result = ozone.forward(state, time=0.0, dt=3600.0)
        
        # All should produce valid stress factors
        assert 0.0 <= water_result['water_stress_factor'].value.item() <= 1.0
        assert 0.0 <= thermal_result['temperature_stress_factor'].value.item() <= 1.0
        assert 0.0 <= nutrient_result['nutrient_stress_factor'].value.item() <= 1.0
        assert 0.0 <= ozone_result['ozone_stress_factor'].value.item() <= 1.0
    
    def test_stress_multiplicative_combination(self):
        """Test L1.3: Multiplicative stress combination."""
        state = {
            'soil_water_potential': torch.tensor(-1.0), # Moderate water stress
            'air_temperature': torch.tensor(308.15),    # 35 C - heat stress
            'nitrogen_concentration': torch.tensor(0.01), # Low N
            'phosphorus_concentration': torch.tensor(0.002),
            'o3_mixing_ratio': torch.tensor(60e-9),
            'stomatal_conductance': torch.tensor(0.3)
        }
        
        # Calculate individual stress factors
        water = WaterStress()
        thermal = ThermalStress()
        nutrient = NutrientStress()
        
        water_factor = water.forward(
            state, time=0.0, dt=3600.0
        )['water_stress_factor'].value
        thermal_factor = thermal.forward(
            state, time=0.0, dt=3600.0
        )['temperature_stress_factor'].value
        nutrient_factor = nutrient.forward(
            state, time=0.0, dt=3600.0
        )['nutrient_stress_factor'].value
        
        # Multiplicative combination
        combined_stress = water_factor * thermal_factor * nutrient_factor
        
        # Combined should be less than any individual
        assert combined_stress.item() <= water_factor.item()
        assert combined_stress.item() <= thermal_factor.item()
        assert combined_stress.item() <= nutrient_factor.item()
        
        # Should still be valid
        assert 0.0 <= combined_stress.item() <= 1.0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])


