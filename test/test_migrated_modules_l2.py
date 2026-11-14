"""
L2 Integration Tests for Migrated Scientific Modules

This module contains integration tests for scientific modules that have been
migrated to use the bridge base classes. Tests verify that the migration
preserved functionality while gaining the benefits of unified architecture.

Test Focus Areas:
1. Migrated module instantiation and configuration
2. Forward pass functionality preservation
3. Parameter access through unified interface
4. Dependency declaration consistency
5. Integration with StateManager and Simulator
6. Cross-module data flow and coupling
7. Performance and memory usage validation

Migrated Modules Under Test:
- FarquharPhotosynthesis: BiologicalProcessFunctor
- PenmanMonteithET: TransportMechanismFunctor
- EddyDiffusionBLMFunctor: TransportMechanismFunctor

References:
    - Farquhar, G.D. et al. (1980). Biochemical model of photosynthetic CO2 assimilation
    - Allen, R.G. et al. (1998). Crop evapotranspiration - FAO irrigation paper 56
    - Stull, R.B. (1988). An Introduction to Boundary Layer Meteorology
"""

import pytest
import torch
from collections.abc import Mapping

from relic.core import BiologicalProcessFunctor, TransportMechanismFunctor
from relic.core.dependency_graph import DependencyGraphResolver
from relic.core.functor import ProcessCategory
from relic.core.state import UpdateItem
from relic.scientific.atmosphere.environmental_forcing import (
    EnvironmentalForcingFunctor,
)
from relic.scientific.atmosphere.radiation import RadiationFunctor
from relic.scientific.biosphere.photosynthesis.farquhar import (
    FarquharPhotosynthesis,
)
from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import (
    PenmanMonteithET,
)


def _assert_finite_tensor_payload(value, output_name: str) -> None:
    """Validate tensor leaves in UpdateItem or structured diagnostics."""
    payload = value.value if isinstance(value, UpdateItem) else value
    if isinstance(payload, Mapping):
        assert payload, f"Structured diagnostic {output_name} is empty"
        for key, nested_value in payload.items():
            _assert_finite_tensor_payload(
                nested_value,
                f"{output_name}.{key}",
            )
        return
    assert isinstance(payload, torch.Tensor), f"Output {output_name} is not a tensor"
    assert torch.isfinite(payload).all(), f"Output {output_name} contains non-finite values"


class TestFarquharPhotosynthesisMigration:
    """Test Farquhar photosynthesis module migration to BiologicalProcessFunctor."""
    
    @pytest.fixture
    def farquhar_config(self):
        """Standard Farquhar photosynthesis configuration."""
        return {
            'parameters': {
                'vcmax25': 60.0,    # Max carboxylation rate at 25 degC [umol CO2 m-2 s-1]
                'jmax25': 120.0,    # Max electron transport at 25 degC [umol e- m-2 s-1]
                'rd25': 1.0,        # Dark respiration at 25 degC [umol CO2 m-2 s-1]
                'kc25': 404.0,      # Michaelis-Menten constant for CO2 at 25 degC [ubar]
                'ko25': 278000.0,   # Michaelis-Menten constant for O2 at 25 degC [ubar]
                'cp25': 42.75,      # CO2 compensation point at 25 degC [ubar]
                'ha_vcmax': 65330.0,  # Activation energy for Vcmax [J mol-1]
                'ha_jmax': 43540.0,   # Activation energy for Jmax [J mol-1]
                'ha_rd': 46390.0      # Activation energy for Rd [J mol-1]
            }
        }
    
    @pytest.fixture
    def standard_environment(self):
        """Standard environmental conditions for testing."""
        return {
            'air_temperature': torch.tensor(298.15),
            'ppfd': torch.tensor(1000.0),
            'atmospheric_co2_ppm': torch.tensor(400.0),
            'vapor_pressure_deficit': torch.tensor(1000.0),
            'atmospheric_pressure': torch.tensor(101325.0),  # Pa
            'relative_humidity': torch.tensor(0.7)     # [-]
        }
    
    @pytest.fixture  
    def standard_state(self):
        """Standard plant state for testing."""
        return {
            'leaf_area_index': torch.tensor(5.0),      # [-]
            'canopy_conductance': torch.tensor(0.1),   # [m s-1]
            'leaf_nitrogen': torch.tensor(2.0),        # [g N m-2]
            'water_potential': torch.tensor(-0.5)      # [MPa]
        }
    
    def test_migration_inheritance(self, farquhar_config):
        """Test that Farquhar correctly inherits from BiologicalProcessFunctor."""
        farquhar = FarquharPhotosynthesis(
            name='test_farquhar',
            config=farquhar_config,
        )
        
        # Test inheritance chain
        assert isinstance(farquhar, BiologicalProcessFunctor)
        assert isinstance(farquhar, torch.nn.Module)
        
        # Test category assignment
        assert farquhar.metadata.category == ProcessCategory.BIOLOGICAL_PROCESS
        
        # Test biological process specific attributes
        assert hasattr(farquhar, 'process_name')
        assert hasattr(farquhar, 'q10_value')
        
        # Test temperature function availability from bridge
        assert hasattr(farquhar, 'temperature_function')
        assert callable(farquhar.temperature_function)
    
    def test_parameter_access_unified(self, farquhar_config):
        """Test unified parameter access through bridge class."""
        farquhar = FarquharPhotosynthesis(
            name='test_farquhar',
            config=farquhar_config,
        )
        
        # Default precedence is YAML/ParameterManager-first; config can be requested explicitly
        vcmax25_config = farquhar.get_parameter_value('vcmax25', source_order=("config", "manager"))
        assert float(vcmax25_config) == pytest.approx(60.0)
        
        jmax25_config = farquhar.get_parameter_value('jmax25', source_order=("config", "manager"))
        assert float(jmax25_config) == pytest.approx(120.0)

        # Verify manager-first resolution returns the active database value
        vcmax25_manager = farquhar.get_parameter_value('vcmax25')
        assert vcmax25_manager is not None
        
        with pytest.raises(ValueError, match='Required parameter'):
            farquhar.get_parameter_value(
                'missing_param',
                source_order=('config',),
                required=True,
            )
        
        # Config-backed parameters should be accessible via explicit config-first lookup
        assert float(farquhar.get_parameter_value('vcmax25', source_order=("config", "manager"))) == pytest.approx(
            float(farquhar_config['parameters']['vcmax25'])
        )
    
    def test_forward_functionality_preserved(self, farquhar_config, standard_state, standard_environment):
        """Test that forward pass functionality is preserved after migration."""
        farquhar = FarquharPhotosynthesis(
            name='test_farquhar',
            config=farquhar_config,
        )
        
        # Test forward pass (merge environment into state for Farquhar)
        merged_state = {**standard_state, **standard_environment}
        result = farquhar.forward(
            state=merged_state,
            time=torch.tensor(0.0),
            dt=torch.tensor(3600.0)  # 1 hour
        )
        
        # Verify output structure
        assert isinstance(result, dict)
        
        expected_keys = {
            'gross_photosynthesis',
            'gross_primary_productivity',
            'net_photosynthesis',
        }
        assert expected_keys.issubset(result)
        assert {
            'gpp',
            'photosynthesis_rate',
            'an',
            'net_assimilation',
        }.isdisjoint(result)
        
        # Verify output values have finite tensor payloads.
        for key, value in result.items():
            _assert_finite_tensor_payload(value, key)
            
        assert result['gross_primary_productivity'].value > 0
    
    def test_temperature_response_is_owned_by_the_process(self, farquhar_config):
        """Farquhar evaluates its own temperature kinetics inside forward; the
        bridge offers no generic temperature law, so the abstract contract refuses."""
        farquhar = FarquharPhotosynthesis(
            name='test_farquhar',
            config=farquhar_config,
        )
        with pytest.raises(RuntimeError, match="concrete BiologicalProcessFunctor"):
            farquhar.temperature_function(torch.tensor([288.15, 298.15, 308.15]))
    
    def test_dependencies_preserved(self, farquhar_config):
        """Test that dependency declarations are preserved."""
        farquhar = FarquharPhotosynthesis(
            name='test_farquhar',
            config=farquhar_config,
        )
        
        # Test dependency declaration
        dependencies = farquhar.get_dependencies()
        assert isinstance(dependencies, list)
        
        # Dependencies should be valid (no None values)
        assert all(dep is not None for dep in dependencies)


class TestPenmanMonteithMigration:
    """Test Penman-Monteith ET module migration to TransportMechanismFunctor."""
    
    @pytest.fixture
    def pm_config(self):
        """Standard Penman-Monteith configuration."""
        return {
            'parameters': {
                'canopy_height': 0.5,           # [m]
                'roughness_length': 0.05,       # [m] 
                'displacement_height': 0.33,    # [m]
                'minimum_conductance': 0.001,   # [m s-1]
                'psychrometric_constant': 0.665, # [kPa K-1]
                'specific_heat_air': 1013.0,    # [J kg-1 K-1]
                'air_density': 1.225,           # [kg m-3]
                'latent_heat_vaporization': 2.45e6  # [J kg-1]
            }
        }
    
    @pytest.fixture
    def et_environment(self):
        """Standard environmental conditions for ET calculation."""
        return {
            'air_temperature': torch.tensor(298.15),     # K
            'net_radiation': torch.tensor(600.0),        # W m-2
            'wind_speed': torch.tensor(2.0),            # m s-1
            'relative_humidity': torch.tensor(0.6),      # [-]
            'atmospheric_pressure': torch.tensor(101325.0),  # Pa
            'soil_heat_flux': torch.tensor(50.0)         # W m-2
        }
    
    @pytest.fixture
    def et_state(self):
        """Standard vegetation state for ET calculation."""
        return {
            'surface_resistance': torch.tensor(70.0),    # s m-1
            'leaf_area_index': torch.tensor(3.5),        # [-]
            'canopy_conductance': torch.tensor(0.01),    # m s-1
            'root_zone_depth': torch.tensor(1.0)         # m
        }
    
    def test_migration_inheritance(self, pm_config):
        """Test that Penman-Monteith correctly inherits from TransportMechanismFunctor."""
        pm_et = PenmanMonteithET(
            name='test_pm_et',
            config={**pm_config, "canopy_resistance_source": "state"}
        )
        
        # Test inheritance chain
        assert isinstance(pm_et, TransportMechanismFunctor)
        assert isinstance(pm_et, torch.nn.Module)
        
        # Test category assignment
        assert pm_et.metadata.category == ProcessCategory.TRANSPORT_MECHANISM
        
        # Test transport mechanism specific attributes
        assert hasattr(pm_et, 'transport_type')
        assert hasattr(pm_et, 'diffusivity')
        
        # Test transport mechanism methods from bridge
        assert hasattr(pm_et, 'flux_to_updates')
        assert callable(pm_et.flux_to_updates)
    
    def test_et_calculation_functionality(self, pm_config, et_state, et_environment):
        """Test that ET calculation functionality is preserved."""
        pm_et = PenmanMonteithET(
            name='test_pm_et',
            config={**pm_config, "canopy_resistance_source": "state"}
        )
        
        # Test forward pass: the drivers are read from the state (the one input channel)
        result = pm_et.forward(
            state={**et_state, **et_environment},
            time=torch.tensor(0.0),
            dt=torch.tensor(3600.0),
        )
        
        # Verify output structure
        assert isinstance(result, dict)
        
        assert 'evapotranspiration_flux' in result
        assert {'et', 'evapotranspiration', 'et_flux', 'water_vapor_flux'}.isdisjoint(
            result
        )
        
        # Verify outputs are reasonable
        for key, value in result.items():
            _assert_finite_tensor_payload(value, key)

        et_flux = result['evapotranspiration_flux']
        et_payload = et_flux.value if isinstance(et_flux, UpdateItem) else et_flux
        assert et_payload >= 0
    
    def test_flux_to_updates_bridge(self, pm_config):
        """Test flux to updates conversion from bridge class."""
        pm_et = PenmanMonteithET(
            name='test_pm_et',
            config={**pm_config, "canopy_resistance_source": "state"}
        )
        
        # Test flux conversion
        fluxes = {
            'evapotranspiration_flux': torch.tensor([0.1, 0.2]),
            'sensible_heat_flux': torch.tensor([50.0, 60.0])
        }
        
        updates = pm_et.flux_to_updates(fluxes)
        
        # Default implementation should preserve fluxes
        assert isinstance(updates, dict)
        assert 'evapotranspiration_flux' in updates
        torch.testing.assert_close(updates['evapotranspiration_flux'], fluxes['evapotranspiration_flux'])
    
    def test_parameter_access_unified(self, pm_config):
        """Test unified parameter access through bridge."""
        pm_et = PenmanMonteithET(
            name='test_pm_et',
            config={**pm_config, "canopy_resistance_source": "state"}
        )
        
        # Test parameter access
        canopy_height = pm_et.get_parameter_value(
            'canopy_height',
            source_order=("config", "manager"),
            validate=False,
        )
        assert float(canopy_height.detach().cpu().item()) == 0.5
        
        with pytest.raises(ValueError, match='Required parameter'):
            pm_et.get_parameter_value(
                'missing_param',
                source_order=('config',),
                required=True,
            )


class TestModuleIntegration:
    """Integration tests for multiple migrated modules working together."""
    
    def test_photosynthesis_et_coupling(self):
        """Test coupling between photosynthesis and ET modules."""
        
        # Create both modules with consistent configurations
        farquhar_config = {
            'parameters': {
                'vcmax25': 60.0,
                'jmax25': 120.0,
                'rd25': 1.0
            }
        }
        
        pm_config = {
            'parameters': {
                'canopy_height': 0.5,
                'minimum_conductance': 0.001
            }
        }
        
        farquhar = FarquharPhotosynthesis(
            name='photosynthesis',
            config=farquhar_config,
        )
        pm_et = PenmanMonteithET(name='et', config={**pm_config, "canopy_resistance_source": "state"})
        
        # Test that they can work with shared state
        shared_environment = {
            'air_temperature': torch.tensor(298.15),
            'ppfd': torch.tensor(1000.0),
            'atmospheric_co2_ppm': torch.tensor(400.0),
            'vapor_pressure_deficit': torch.tensor(1000.0),
            'net_radiation': torch.tensor(600.0),
            'wind_speed': torch.tensor(2.0),
            'relative_humidity': torch.tensor(0.6),
            'atmospheric_pressure': torch.tensor(101325.0)
        }
        
        shared_state = {
            'leaf_area_index': torch.tensor(4.0),
            'canopy_conductance': torch.tensor(0.01),
            'surface_resistance': torch.tensor(70.0)
        }
        
        # Test forward passes (merge environment into state)
        merged_state = {**shared_state, **shared_environment}
        gpp_result = farquhar.forward(
            state=merged_state,
            time=torch.tensor(0.0),
            dt=torch.tensor(3600.0)
        )
        
        et_result = pm_et.forward(
            state=merged_state,
            time=torch.tensor(0.0),
            dt=torch.tensor(3600.0),
        )
        
        # Verify both modules produce reasonable outputs
        assert isinstance(gpp_result, dict) and len(gpp_result) > 0
        assert isinstance(et_result, dict) and len(et_result) > 0
        
        # Verify they don't interfere with each other
        for key, value in gpp_result.items():
            _assert_finite_tensor_payload(value, key)
        for key, value in et_result.items():
            _assert_finite_tensor_payload(value, key)
    
    def test_dependency_resolution_consistency(self):
        """Test that migrated modules work with dependency resolution."""
        farquhar = FarquharPhotosynthesis(
            name='photosynthesis',
            config={'parameters': {}},
        )
        pm_et = PenmanMonteithET(name='et', config={'parameters': {}, "canopy_resistance_source": "state"})
        environmental_forcing = EnvironmentalForcingFunctor(
            name='environmental_forcing'
        )
        atmospheric_radiation = RadiationFunctor(
            name='atmospheric_radiation',
            config={'enable_conservation_check': False},
        )

        farquhar_deps = farquhar.get_dependencies()
        et_deps = pm_et.get_dependencies()

        assert isinstance(farquhar_deps, list)
        assert isinstance(et_deps, Mapping)
        assert set(et_deps) == {'lagged'}
        assert set(et_deps['lagged']) == {
            'bucket_soil_moisture',
            'richards_soil_moisture',
        }

        graph = DependencyGraphResolver()
        graph.add_functor(environmental_forcing)
        graph.add_functor(atmospheric_radiation)
        graph.add_functor(farquhar)
        graph.add_functor(pm_et)

        sorted_nodes = graph.topological_sort()
        assert set(sorted_nodes) == {
            'environmental_forcing',
            'atmospheric_radiation',
            'photosynthesis',
            'et',
        }
        assert sorted_nodes.index('environmental_forcing') < sorted_nodes.index(
            'photosynthesis'
        )
        assert sorted_nodes.index('atmospheric_radiation') < sorted_nodes.index(
            'photosynthesis'
        )

    def test_pre_registration_cpu_placeholder_across_modules(self):
        """Test the documented pre-registration CPU placeholder."""
        device_obj = torch.device('cpu')
        
        farquhar = FarquharPhotosynthesis(
            name='photosynthesis',
            config={'parameters': {}},
        )
        pm_et = PenmanMonteithET(name='et', config={'parameters': {}, "canopy_resistance_source": "state"})
        
        state = {
            'leaf_area_index': torch.tensor(4.0, device=device_obj),
            'surface_resistance': torch.tensor(70.0, device=device_obj),
            'canopy_conductance': torch.tensor(0.01, device=device_obj)
        }
        
        environment = {
            'air_temperature': torch.tensor(298.15, device=device_obj),
            'atmospheric_co2_ppm': torch.tensor(400.0, device=device_obj),
            'ppfd': torch.tensor(1000.0, device=device_obj),
            'vapor_pressure_deficit': torch.tensor(1000.0, device=device_obj),
            'net_radiation': torch.tensor(600.0, device=device_obj),
            'wind_speed': torch.tensor(2.0, device=device_obj),
            'relative_humidity': torch.tensor(0.6, device=device_obj),
            'atmospheric_pressure': torch.tensor(101325.0, device=device_obj),
        }

        merged_state = {**state, **environment}
        gpp_result = farquhar.forward(
            merged_state,
            torch.tensor(0.0),
            torch.tensor(3600.0),
        )
        et_result = pm_et.forward(
            merged_state,
            torch.tensor(0.0),
            torch.tensor(3600.0),
        )

        assert farquhar.device == device_obj
        assert pm_et.device == device_obj
        for value in gpp_result.values():
            payload = value.value if isinstance(value, UpdateItem) else value
            if isinstance(payload, torch.Tensor):
                assert payload.device == device_obj

        for value in et_result.values():
            payload = value.value if isinstance(value, UpdateItem) else value
            if isinstance(payload, torch.Tensor):
                assert payload.device == device_obj


class TestPerformanceRegression:
    """Test that migration does not introduce performance regressions."""
    
    def test_forward_pass_timing(self):
        """Test that forward passes complete within reasonable time."""
        import time
        
        farquhar = FarquharPhotosynthesis(
            name='timing_test',
            config={'parameters': {'vcmax25': 60.0, 'jmax25': 120.0}},
        )
        
        state = {
            'leaf_area_index': torch.tensor(4.0),
            'canopy_conductance': torch.tensor(0.01)
        }
        
        forcing = {
            'air_temperature': torch.tensor(298.15),
            'atmospheric_co2_ppm': torch.tensor(400.0),
            'ppfd': torch.tensor(1000.0),
            'vapor_pressure_deficit': torch.tensor(1000.0),
        }
        merged_state = {**state, **forcing}
        
        # Warmup
        for _ in range(5):
            farquhar.forward(merged_state, torch.tensor(0.0), torch.tensor(3600.0))
        
        # Time actual execution
        start_time = time.time()
        for _ in range(10):
            result = farquhar.forward(
                merged_state,
                torch.tensor(0.0),
                torch.tensor(3600.0),
            )
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        
        # Should complete within reasonable time (generous limit for CI)
        assert avg_time < 0.1, f"Forward pass too slow: {avg_time:.4f} seconds"
        
        # Verify result is still valid
        assert isinstance(result, dict)
        assert len(result) > 0


if __name__ == '__main__':
    # Run tests directly
    pytest.main([__file__, '-v', '--tb=short'])


