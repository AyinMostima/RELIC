"""
Regression tests for feature flag behavior in RELIC system.

This test suite ensures that:
1. All feature flags are disabled by default (backward consistency)
2. Enabling features produces expected behavior changes
3. Disabling features restores baseline behavior
4. Feature interactions work correctly

Author: Yuchong Long
Date: 2024-08
"""

import pytest
import torch
from contextlib import contextmanager

from relic.core.feature_flags import FeatureFlagManager, feature_manager
from relic.scientific.hydrosphere.soil_moisture.richards import RichardsSoilMoisture
from relic.scientific.atmosphere.radiation import RadiationFunctor
from relic.scientific.biosphere.stress.ozone_stress import OzoneStress
from relic.core.state import StateManager
from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import PenmanMonteithET
from relic.parameters import ParameterSpec, get_global_manager


class TestFeatureFlagDefaults:
    """Test that all feature flags are disabled by default."""
    
    def test_all_flags_disabled_by_default(self):
        """Verify the declared off-by-default feature flags are disabled by default.

        Only declared FeatureFlagConfig fields are flags (an undeclared name
        raises); dynamic_snow_albedo is declared on by default.
        """
        manager = FeatureFlagManager()

        expected_flags = [
            'full_richards_solver',
            'o3_stress_enabled',
            'root_nutrient_uptake_enabled',
            'canopy_layers_enabled',
            'strict_conservation_enabled',
            'radiation_conservation',
            'advanced_numerical_solvers',
            'parameter_sensitivity_analysis',
            'canopy_interception_diagnostics',
            'external_model_coupling'
        ]
        
        for flag in expected_flags:
            assert not manager.is_enabled(flag), f"Flag '{flag}' should be disabled by default"
    
    def test_feature_flag_status_report(self):
        """Test feature flag status reporting."""
        manager = FeatureFlagManager()
        status = manager.get_all_flags()

        assert isinstance(status, dict)
        assert len(status) >= 10  # Should have at least 10 feature flags

        # Core flags should be False by default
        expected_flags = [
            'full_richards_solver',
            'o3_stress_enabled',
            'root_nutrient_uptake_enabled',
            'canopy_layers_enabled',
            'strict_conservation_enabled',
            'enzyme_kinetics_enabled',
            'atmospheric_chemistry_enabled',
            'multi_layer_snow_enabled',
            'canopy_interception_diagnostics',
            'dynamic_vegetation_enabled'
        ]
        for flag in expected_flags:
            if flag in status:
                assert status[flag] is False, f"Flag '{flag}' should be False by default"


class TestRichardsFeatureFlagRegression:
    """Regression tests for the canonical Richards runtime lock."""
    
    @pytest.fixture
    def state_factory(self):
        """Build the canonical Richards state for the selected model."""

        def _build(model):
            return {
                'soil_moisture': torch.full((model.soil_layers,), 0.3),
                'infiltration_flux': torch.tensor(0.0),
                'evapotranspiration_flux': torch.tensor(0.0),
            }

        return _build
    
    def test_richards_runtime_is_not_controlled_by_feature_flag(self, state_factory):
        """Test that feature flags cannot select a separate Richards process law."""
        with self._reset_feature_flags():
            feature_manager.disable('full_richards_solver')
            disabled_model = RichardsSoilMoisture()
            disabled_result = disabled_model.forward(
                state_factory(disabled_model), time=0.0, dt=60.0
            )

            feature_manager.enable('full_richards_solver')
            enabled_model = RichardsSoilMoisture()
            enabled_result = enabled_model.forward(
                state_factory(enabled_model), time=0.0, dt=60.0
            )

            assert disabled_model.runtime_path == "canonical_iphs_mixed_form_richards"
            assert enabled_model.runtime_path == "canonical_iphs_mixed_form_richards"
            assert not hasattr(disabled_model, "compute_mode")
            assert not hasattr(enabled_model, "compute_mode")
            assert set(disabled_result) == set(disabled_model.metadata.output_variables)
            assert set(enabled_result) == set(enabled_model.metadata.output_variables)
            torch.testing.assert_close(
                disabled_result['soil_moisture'].value,
                enabled_result['soil_moisture'].value,
            )
            torch.testing.assert_close(
                disabled_result['drainage_flux'].value,
                enabled_result['drainage_flux'].value,
            )

    def test_richards_obsolete_compute_mode_parameter_is_ignored(self, state_factory):
        """Test that the removed local mode key cannot select an alternate runtime."""
        with self._reset_feature_flags():
            manager = get_global_manager()
            snapshot = manager.snapshot()
            try:
                manager.register_parameter(ParameterSpec(
                    key="hydrosphere.soil_moisture.richards_compute_mode",
                    value="simplified",
                    units="dimensionless",
                    dtype="str",
                    description="Removed Richards compute mode supplied by a stale configuration", bounds=(-float("inf"), float("inf")), category="model_parameter",
                ))
                richards = RichardsSoilMoisture()
            finally:
                manager.restore(snapshot)

            assert richards.runtime_path == "canonical_iphs_mixed_form_richards"
            assert not hasattr(richards, "compute_mode")
            result = richards.forward(state_factory(richards), time=0.0, dt=60.0)
            assert set(result) == set(richards.metadata.output_variables)

    def test_richards_runtime_path_is_stable_under_removed_local_override(self, state_factory):
        """Test that obsolete local overrides do not alter the canonical runtime."""
        with self._reset_feature_flags():
            manager = get_global_manager()
            snapshot = manager.snapshot()
            try:
                manager.register_parameter(ParameterSpec(
                    key="hydrosphere.soil_moisture.richards_compute_mode",
                    value="full",
                    units="dimensionless",
                    dtype="str",
                    description="Removed Richards compute mode supplied by a stale configuration", bounds=(-float("inf"), float("inf")), category="model_parameter",
                ))
                richards = RichardsSoilMoisture()
            finally:
                manager.restore(snapshot)

            assert richards.runtime_path == "canonical_iphs_mixed_form_richards"
            result = richards.forward(state_factory(richards), time=0.0, dt=60.0)
            assert set(result) == set(richards.metadata.output_variables)

    @pytest.mark.parametrize(
        "runtime_override",
        [
            {"time_scheme": "crank_nicolson"},
            {"linearization": "picard"},
            {"max_picard_iterations": 1},
            {"convergence_tolerance": 1.0e-4},
            {"cfl_limit": 0.4},
        ],
        ids=[
            "time_scheme",
            "linearization",
            "max_iterations",
            "tolerance",
            "cfl_limit",
        ],
    )
    def test_richards_rejects_runtime_solver_overrides(
        self,
        state_factory,
        runtime_override,
    ):
        """Runtime kwargs cannot bypass the ParameterManager solver policy."""
        model = RichardsSoilMoisture()
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            model.forward(
                state_factory(model),
                time=0.0,
                dt=60.0,
                **runtime_override,
            )
    
    @contextmanager
    def _reset_feature_flags(self):
        """Context manager to reset feature flags after test."""
        try:
            yield
        finally:
            # Reset all flags to default (disabled)
            for flag in feature_manager.get_all_flags():
                feature_manager.disable(flag)


class TestCanopyRadiationRegression:
    """Regression tests for canopy radiation feature flag behavior."""
    
    @pytest.fixture
    def radiation_state(self):
        """Basic state for radiation testing."""
        return {
            'shortwave_down': torch.tensor(800.0),
            'incoming_shortwave_potential': torch.tensor(900.0),
            'air_temperature': torch.tensor(298.15),
            'cos_zenith': torch.tensor(0.8),
            'surface_temperature': torch.tensor(297.15),
            'relative_humidity': torch.tensor(0.6),
            'surface_albedo': torch.tensor(0.2),
            'surface_emissivity': torch.tensor(0.97),
            'cloud_fraction': torch.tensor(0.0),
            'leaf_area_index': torch.tensor(3.0),
            'canopy_height': torch.tensor(15.0)
        }
    
    def test_canopy_layers_disabled_by_default(self, radiation_state):
        """Test that canopy layers are disabled by default."""
        with self._reset_feature_flags():
            feature_manager.disable('canopy_layers_enabled')
            
            radiation = RadiationFunctor()
            
            # Should use single-layer approach by default
            result = radiation.forward(radiation_state, 298.15, 3600)
            
            assert 'net_radiation' in result
            assert 'shortwave_net' in result or 'shortwave_down' in result
            
            # Should not have multi-layer diagnostics when disabled
            multilayer_vars = [
                'canopy_radiation_profile', 
                'layer_par_absorption', 
                'layer_radiation_balance'
            ]
            for var in multilayer_vars:
                if var in result:
                    # If present, should be minimal (single value or empty)
                    assert len(result[var]) <= 1, f"Multi-layer variable {var} should be minimal when disabled"
    
    def test_canopy_layers_enabled_behavior(self, radiation_state):
        """Test canopy multi-layer behavior when enabled."""
        with self._reset_feature_flags():
            feature_manager.enable('canopy_layers_enabled')
            
            radiation = RadiationFunctor()
            result = radiation.forward(radiation_state, 298.15, 3600)
            
            # Should have standard outputs
            assert 'net_radiation' in result
            assert 'apar_layer' in result or 'fpar' in result
    
    @contextmanager
    def _reset_feature_flags(self):
        """Context manager to reset feature flags after test."""
        try:
            yield
        finally:
            for flag in feature_manager.get_all_flags():
                feature_manager.disable(flag)


class TestOzoneStressRegression:
    """Regression tests for ozone stress coupling behavior."""
    
    @pytest.fixture
    def ozone_state(self):
        """Basic state for ozone stress testing."""
        return {
            'air_temperature': torch.tensor(298.15),
            'o3_mixing_ratio': torch.tensor(60e-9),  # 60 ppb
            'vapor_pressure_deficit': torch.tensor(1500.0),
            'photosynthetically_active_radiation': torch.tensor(1500.0),
            'day_of_year': torch.tensor(180),  # Mid-summer
            'stomatal_conductance': torch.tensor(0.2)
        }
    
    def test_o3_stress_module_standalone(self, ozone_state):
        """Test that O3 stress module works independently."""
        ozone = OzoneStress()
        result = ozone.forward(ozone_state, 298.15, 3600)
        
        assert 'ozone_stress_factor' in result
        stress_factor = result['ozone_stress_factor'].value
        
        # Stress factor should be between 0 and 1
        assert 0.0 <= stress_factor <= 1.0
        assert not torch.isnan(stress_factor)
    
    def test_o3_coupling_flag_default_disabled(self):
        """Test that O3 coupling to photosynthesis is disabled by default."""
        with self._reset_feature_flags():
            assert not feature_manager.is_enabled('o3_stress_enabled')
    
    @contextmanager 
    def _reset_feature_flags(self):
        """Context manager to reset feature flags after test."""
        try:
            yield
        finally:
            for flag in feature_manager.get_all_flags():
                feature_manager.disable(flag)


class TestStateManagerConservationRegression:
    """Regression tests for StateManager strict conservation behavior."""
    
    def test_strict_conservation_disabled_by_default(self):
        """Test that strict conservation is disabled by default."""
        with self._reset_feature_flags():
            assert not feature_manager.is_enabled('strict_conservation_enabled')
            
            # StateManager should work without strict conservation checks
            manager = StateManager()
            
            # Basic functionality should work
            assert hasattr(manager, 'validate_state')
            assert hasattr(manager, 'apply_updates')
    
    def test_strict_conservation_when_enabled(self):
        """Test strict conservation behavior when enabled."""
        with self._reset_feature_flags():
            feature_manager.enable('strict_conservation_enabled')
            
            manager = StateManager()
            
            # Should have conservation checking capability
            assert hasattr(manager, 'validate_state')
            assert hasattr(manager, 'apply_updates')
    
    @contextmanager
    def _reset_feature_flags(self):
        """Context manager to reset feature flags after test."""
        try:
            yield
        finally:
            for flag in feature_manager.get_all_flags():
                feature_manager.disable(flag)


class TestCanopyInterceptionDiagnosticsRegression:
    """Regression tests for canopy interception diagnostics."""
    
    @pytest.fixture
    def et_state(self):
        """Basic state for evapotranspiration testing."""
        return {
            'net_radiation': torch.tensor(400.0),
            'air_temperature': torch.tensor(298.15),
            'relative_humidity': torch.tensor(0.6),
            'wind_speed': torch.tensor(2.0),
            'atmospheric_pressure': torch.tensor(101325.0),
            'surface_resistance': torch.tensor(50.0),
            'aerodynamic_resistance': torch.tensor(20.0),
            'precipitation_rate': torch.tensor(5e-6),
            'canopy_storage_capacity': torch.tensor(2.0),
            'canopy_storage': torch.tensor(0.5)
        }
    
    def test_canopy_diagnostics_disabled_by_default(self, et_state):
        """Test that canopy interception diagnostics are disabled by default."""
        with self._reset_feature_flags():
            assert not feature_manager.is_enabled('canopy_interception_diagnostics')
            
            et_model = PenmanMonteithET(config={"canopy_resistance_source": "state"})
            result = et_model.forward(et_state, 298.15, 3600)
            
            # Should have basic ET output
            assert 'evapotranspiration_flux' in result
    
    def test_canopy_diagnostics_when_enabled(self, et_state):
        """Test canopy diagnostics behavior when enabled."""
        with self._reset_feature_flags():
            feature_manager.enable('canopy_interception_diagnostics')
            
            et_model = PenmanMonteithET(config={"canopy_resistance_source": "state"})
            
            # Should not crash when enabled
            result = et_model.forward(et_state, 298.15, 3600)
            assert 'evapotranspiration_flux' in result
    
    @contextmanager
    def _reset_feature_flags(self):
        """Context manager to reset feature flags after test."""
        try:
            yield
        finally:
            for flag in feature_manager.get_all_flags():
                feature_manager.disable(flag)


class TestFeatureFlagInteractions:
    """Test interactions between multiple feature flags."""
    
    def test_no_feature_conflicts_default(self):
        """Test that no feature flags conflict in their default state."""
        with self._reset_feature_flags():
            # All should be disabled by default
            manager = FeatureFlagManager()
            status = manager.get_all_flags()
            
            # Verify all are False
            assert all(not enabled for enabled in status.values()), \
                "All feature flags should be disabled by default"
    
    def test_multiple_flag_enable_disable(self):
        """Test enabling and disabling multiple flags."""
        with self._reset_feature_flags():
            manager = FeatureFlagManager()
            
            # Enable multiple flags
            flags_to_test = ['canopy_layers_enabled', 'strict_conservation_enabled']
            
            for flag in flags_to_test:
                manager.enable(flag)
                assert manager.is_enabled(flag), f"Flag {flag} should be enabled"
            
            # Disable them
            for flag in flags_to_test:
                manager.disable(flag)
                assert not manager.is_enabled(flag), f"Flag {flag} should be disabled"
    
    @contextmanager
    def _reset_feature_flags(self):
        """Context manager to reset feature flags after test."""
        try:
            yield
        finally:
            manager = FeatureFlagManager()
            for flag in manager.get_all_flags():
                manager.disable(flag)


class TestBaselineBehaviorRegressionMatrix:
    """Comprehensive baseline behavior regression matrix."""
    
    @pytest.fixture(params=[
        'canonical_richards',  # Phase 8 Richards behavior
        'single_layer',  # Default radiation behavior  
        'basic_et',  # Default ET behavior
        'no_o3_stress',  # Default no O3 coupling
        'no_strict_conservation'  # Default loose conservation
    ])
    def baseline_scenario(self, request):
        """Parametrized baseline scenarios."""
        return request.param
    
    def test_baseline_behavior_unchanged(self, baseline_scenario):
        """Test that baseline behavior is unchanged across versions."""
        with self._reset_feature_flags():
            
            if baseline_scenario == 'canonical_richards':
                self._test_canonical_richards()
            elif baseline_scenario == 'single_layer':
                self._test_single_layer_radiation()
            elif baseline_scenario == 'basic_et':
                self._test_basic_et()
            elif baseline_scenario == 'no_o3_stress':
                self._test_no_o3_coupling()
            elif baseline_scenario == 'no_strict_conservation':
                self._test_loose_conservation()
    
    def _test_canonical_richards(self):
        """Test canonical Richards behavior."""
        state = {
            'soil_moisture': torch.tensor([0.3, 0.25, 0.2]),
            'soil_temperature': torch.tensor([293.15, 292.15, 291.15]),
        }
        
        richards = RichardsSoilMoisture()
        # Should not crash and produce reasonable output
        assert richards is not None
        assert richards.runtime_path == "canonical_iphs_mixed_form_richards"
    
    def _test_single_layer_radiation(self):
        """Test single-layer radiation behavior."""
        state = {
            'shortwave_down': torch.tensor(800.0),
            'air_temperature': torch.tensor(298.15),
            'incoming_shortwave_potential': torch.tensor(900.0),
            'cos_zenith': torch.tensor(0.8),
            'surface_temperature': torch.tensor(297.15),
            'relative_humidity': torch.tensor(0.6),
            'surface_albedo': torch.tensor(0.2),
            'surface_emissivity': torch.tensor(0.97),
            'cloud_fraction': torch.tensor(0.0),
        }
        
        radiation = RadiationFunctor()
        result = radiation.forward(state, 298.15, 3600)
        assert 'net_radiation' in result
    
    def _test_basic_et(self):
        """Test basic ET behavior."""
        state = {
            'net_radiation': torch.tensor(400.0),
            'air_temperature': torch.tensor(298.15),
            'relative_humidity': torch.tensor(0.6),
            'wind_speed': torch.tensor(2.0),
            'atmospheric_pressure': torch.tensor(101325.0),
            'surface_resistance': torch.tensor(150.0)
        }
        
        et_model = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        result = et_model.forward(state, 298.15, 3600)
        assert 'evapotranspiration_flux' in result
    
    def _test_no_o3_coupling(self):
        """Test that O3 coupling is not active by default."""
        assert not feature_manager.is_enabled('o3_stress_enabled')
    
    def _test_loose_conservation(self):
        """Test that strict conservation is not enforced by default."""
        assert not feature_manager.is_enabled('strict_conservation_enabled')
    
    @contextmanager
    def _reset_feature_flags(self):
        """Context manager to reset feature flags after test."""
        try:
            yield
        finally:
            manager = FeatureFlagManager()
            for flag in manager.get_all_flags():
                manager.disable(flag)


class TestEnvironmentVariableOverrides:
    """Test environment variable overrides for feature flags."""
    
    def test_environment_override_disabled_by_default(self):
        """Test that environment variable overrides work but are not set by default."""
        import os
        
        # Should not have override environment variables set
        relic_env_vars = [k for k in os.environ.keys() if k.startswith('RELIC_')]
        
        # If any are set, they should not interfere with defaults
        manager = FeatureFlagManager()
        
        # Most flags should still be disabled unless explicitly overridden
        status = manager.get_all_flags()
        disabled_count = sum(1 for enabled in status.values() if not enabled)
        
        # At least 80% should be disabled (allowing for some environment overrides)
        assert disabled_count >= 0.8 * len(status), \
            "Most feature flags should be disabled by default even with env overrides"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


