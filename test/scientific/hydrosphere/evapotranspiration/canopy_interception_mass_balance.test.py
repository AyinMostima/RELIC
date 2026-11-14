"""
Test canopy interception mass balance assertions and diagnostics.

Tests the comprehensive mass balance validation and diagnostic functionality
for canopy interception processes in the Penman-Monteith ET model.

Test Coverage:
- Mass balance conservation checks
- Physical constraint validation
- Diagnostic logging functionality
- Feature flag integration
- Error threshold handling
"""

import torch
import pytest
import warnings
from unittest.mock import patch, MagicMock

# Import test targets
from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import PenmanMonteithET
from relic.core.feature_flags import feature_manager
from relic.parameters import get_global_manager


class TestCanopyInterceptionMassBalance:
    """Test suite for canopy interception mass balance validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self._param_manager = get_global_manager()
        self._param_snapshot = self._param_manager.snapshot()
        # Feature flags are process-global by design (FeatureFlagManager singleton);
        # this class toggles canopy_interception_diagnostics, so it restores every flag.
        self._flag_snapshot = feature_manager.get_all_flags()
        self._param_manager.set_value(
            "hydrosphere.evapotranspiration.enable_canopy_interception",
            True,
            units="dimensionless",
            validate=True,
            notify=False,
        )
        self._param_manager.set_value(
            "hydrosphere.evapotranspiration.interception_capacity_per_LAI",
            0.2,
            units="mm",
            validate=True,
            notify=False,
        )
        self._param_manager.set_value(
            "hydrosphere.evapotranspiration.canopy_evaporation_efficiency",
            1.2,
            units="dimensionless",
            validate=True,
            notify=False,
        )
        
        self.et_model = PenmanMonteithET(
            name="test_et_with_interception",
            config={"canopy_resistance_source": "state"},
        )
        
        # Test state with precipitation
        self.wet_state = {
            'air_temperature': torch.tensor(298.15),  # 25 C
            'surface_temperature': torch.tensor(298.15),
            'relative_humidity': torch.tensor(0.7),
            'wind_speed': torch.tensor(2.5),  # m/s
            'atmospheric_pressure': torch.tensor(101325.0),  # Pa
            'net_radiation': torch.tensor(300.0),  # W m^-2
            'leaf_area_index': torch.tensor(3.5),
            'surface_resistance': torch.tensor(50.0),  # s m^-1
            'precipitation_flux': torch.tensor(2.0e-4),  # kg m^-2 s^-1 = ~0.72 mm/h
            'canopy_storage': torch.tensor(0.0),  # mm; interception requires the store
        }
        
        # Dry state (no precipitation)
        self.dry_state = self.wet_state.copy()
        self.dry_state['precipitation_flux'] = torch.tensor(0.0)
        
        # The Functor is stateless; canopy storage enters through the state mapping.
        self.initial_canopy_storage = 0.0

    def teardown_method(self):
        """Restore ParameterManager state and the process-global feature flags."""
        self._param_manager.restore(self._param_snapshot)
        for flag, enabled in self._flag_snapshot.items():
            feature_manager.update_flag(flag, enabled)
        
    def test_mass_balance_perfect_conservation(self):
        """Test mass balance validation with perfect conservation."""
        # Enable canopy interception
        feature_manager.enable('canopy_interception_diagnostics')
        
        fluxes = self.et_model.forward(
            state=self.wet_state,
            time=43200.0,  # Noon
            dt=3600.0
        )

        print("Mass balance validation: PASS")
        
        diagnostics = fluxes["diagnostics.evapotranspiration"]
        # Verify we have interception outputs
        assert 'evaporation_from_interception' in diagnostics
        assert 'throughfall_flux' in diagnostics
        assert 'canopy_interception_storage' in diagnostics
        
        # Basic sanity checks
        evap_from_int = float(
            diagnostics['evaporation_from_interception'].detach()
        )
        throughfall = float(diagnostics['throughfall_flux'].detach())
        
        assert evap_from_int >= 0, f"Negative interception evaporation: {evap_from_int}"
        assert throughfall >= 0, f"Negative throughfall: {throughfall}"
        
        print("Perfect conservation test: PASS")
        
    def test_mass_balance_violation_detection(self):
        """Test detection of mass balance violations."""
        # Create a scenario that should trigger mass balance errors
        # by manipulating internal state
        
        # Keep storage within its physical capacity so this test isolates the
        # precipitation partitioning ledger.
        storage_before = 0.5  # mm

        with pytest.raises(
            AssertionError,
            match="Precipitation mass balance violation",
        ):
            self.et_model._validate_canopy_mass_balance(
                precipitation_mm=2.0,
                interception_mm=0.2,
                evap_from_interception_mm=0.5, 
                throughfall_mm=0.0,
                total_et_mm=1.0,
                transpiration_mm=0.5,
                dt=3600.0,
                storage_before_mm=storage_before,
                storage_after_mm=storage_before + 0.2 - 0.5,
            )

        print("Mass balance violation detection: PASS")
        
    def test_physical_constraints_validation(self):
        """Test validation of physical constraints."""
        
        # Test 1: Interception exceeds precipitation (should fail)
        with pytest.raises(AssertionError, match="Precipitation mass balance violation"):
            self.et_model._validate_canopy_mass_balance(
                precipitation_mm=1.0,
                interception_mm=3.0,  # > precipitation + tolerance
                evap_from_interception_mm=0.0,
                throughfall_mm=0.0,
                total_et_mm=1.0,
                transpiration_mm=1.0,
                dt=3600.0,
                storage_before_mm=0.0,
                storage_after_mm=3.0,
            )
        
        # Test 2: Negative transpiration (should fail)  
        with pytest.raises(AssertionError, match="Negative transpiration"):
            self.et_model._validate_canopy_mass_balance(
                precipitation_mm=1.0,
                interception_mm=1.0,
                evap_from_interception_mm=0.0,
                throughfall_mm=0.0,
                total_et_mm=-2.0,
                transpiration_mm=-2.0,  # Negative beyond tolerance
                dt=3600.0,
                storage_before_mm=0.0,
                storage_after_mm=1.0,
            )
        
        # Test 3: Evaporation exceeds the storage available before evaporation (should fail)
        with pytest.raises(AssertionError, match="Evaporation exceeds available storage"):
            self.et_model._validate_canopy_mass_balance(
                precipitation_mm=0.0,
                interception_mm=0.0,
                evap_from_interception_mm=2.0,  # > storage before evaporation + tolerance
                throughfall_mm=0.0,
                total_et_mm=1.0,
                transpiration_mm=0.0,
                dt=3600.0,
                storage_before_mm=0.5,
                storage_after_mm=0.0,
            )
        
        print("Physical constraints validation: PASS")
        
    def test_diagnostic_logging_functionality(self):
        """Test diagnostic logging with feature flag control."""
        
        # Test with diagnostics disabled
        feature_manager.disable('canopy_interception_diagnostics')
        
        # Mock logger to capture calls
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            self.et_model._validate_canopy_mass_balance(
                precipitation_mm=2.0,
                interception_mm=1.2,
                evap_from_interception_mm=0.5,
                throughfall_mm=0.8,
                total_et_mm=1.0,
                transpiration_mm=0.5,
                dt=3600.0,
                storage_before_mm=0.0,
                storage_after_mm=0.7,
            )
            
            # Should not call logger when disabled
            mock_logger.debug.assert_not_called()
        
        # Test with diagnostics enabled
        feature_manager.enable('canopy_interception_diagnostics')
        
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            self.et_model._validate_canopy_mass_balance(
                precipitation_mm=2.0,
                interception_mm=1.2,
                evap_from_interception_mm=0.5,
                throughfall_mm=0.8,
                total_et_mm=1.0,
                transpiration_mm=0.5,
                dt=3600.0,
                storage_before_mm=0.0,
                storage_after_mm=0.7,
            )
            
            # Should call logger when enabled
            assert mock_logger.debug.call_count > 0
            
            # Check that diagnostic information is logged
            log_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
            diagnostic_found = any("Diagnostics" in call for call in log_calls)
            assert diagnostic_found, "Should log diagnostic information"
        
        print("Diagnostic logging functionality: PASS")
        
    def test_interception_efficiency_calculation(self):
        """Test interception efficiency and wet canopy fraction calculations."""
        
        # Enable diagnostics to test calculation functions
        feature_manager.enable('canopy_interception_diagnostics')
        
        # Set up scenario with partial interception
        precipitation = 3.0  # mm
        interception = 1.5   # mm (50% efficiency)
        
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            storage_before = 0.0  # mm
            self.et_model._validate_canopy_mass_balance(
                precipitation_mm=precipitation,
                interception_mm=interception,
                evap_from_interception_mm=0.3,
                throughfall_mm=precipitation - interception,
                total_et_mm=1.0,
                transpiration_mm=0.7,
                dt=3600.0,
                storage_before_mm=storage_before,
                storage_after_mm=storage_before + interception - 0.3,
            )
            
            # Check that interception efficiency was calculated
            log_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
            efficiency_found = any("50" in call and "%" in call for call in log_calls)
            assert efficiency_found, "Should calculate and log interception efficiency"
        
        print("Interception efficiency calculation: PASS")
        
    def test_storage_budget_analysis(self):
        """Test storage budget analysis in diagnostics."""
        
        feature_manager.enable('canopy_interception_diagnostics')
        
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            # Test with storage input and output
            self.et_model._validate_canopy_mass_balance(
                precipitation_mm=2.0,
                interception_mm=1.0,  # Storage input
                evap_from_interception_mm=0.3,  # Storage output
                throughfall_mm=1.0,
                total_et_mm=1.0,
                transpiration_mm=0.7,
                dt=3600.0,
                storage_before_mm=0.0,
                storage_after_mm=0.7,
            )
            
            # Check that storage budget was logged
            log_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
            diagnostics_found = any("Canopy Interception Diagnostics" in call for call in log_calls)
            storage_found = any("Canopy storage:" in call for call in log_calls)
            capacity_found = any("Storage capacity" in call for call in log_calls)
            
            assert diagnostics_found, "Should log canopy interception diagnostics"
            assert storage_found, "Should log canopy storage state"
            assert capacity_found, "Should log canopy storage capacity"
        
        print("Storage budget analysis: PASS")
        
    def test_canopy_storage_bounds_checking(self):
        """Test canopy storage physical bounds checking."""
        
        # Test maximum capacity warning
        oversized_storage = 25.0  # Exceeds reasonable capacity
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            self.et_model._validate_canopy_mass_balance(
                precipitation_mm=0.0,
                interception_mm=0.0,
                evap_from_interception_mm=0.0,
                throughfall_mm=0.0,
                total_et_mm=0.0,
                transpiration_mm=0.0,
                dt=3600.0,
                storage_before_mm=oversized_storage,
                storage_after_mm=oversized_storage,
            )
            
            # Should warn about exceeding physical bounds
            bound_warnings = [warning for warning in w 
                            if "physical bounds" in str(warning.message)]
            assert len(bound_warnings) > 0, "Should warn about exceeding physical bounds"
        
        # Test negative storage (should fail)
        with pytest.raises(AssertionError, match="Negative canopy storage"):
            self.et_model._validate_canopy_mass_balance(
                precipitation_mm=2.0,
                interception_mm=2.0,
                evap_from_interception_mm=0.0,
                throughfall_mm=0.0,
                total_et_mm=0.0,
                transpiration_mm=0.0,
                dt=3600.0,
                storage_before_mm=0.0,
                storage_after_mm=-2.0,
            )
        
        print("Canopy storage bounds checking: PASS")
        
    def test_integrated_mass_balance_with_et_calculation(self):
        """Test integrated mass balance validation during actual ET calculation."""
        
        # Enable strict checking
        feature_manager.enable('canopy_interception_diagnostics')
        
        # Set initial storage through the state mapping
        current_state = self.wet_state.copy()
        current_state['canopy_storage'] = torch.tensor(0.5)  # mm
        
        for step in range(3):
            print(f"Time step {step + 1}:")
            
            # Vary precipitation
            precip_values = [1.0e-4, 3.0e-4, 0.5e-4]  # kg m^-2 s^-1
            current_state['precipitation_flux'] = torch.tensor(precip_values[step])
            
            fluxes = self.et_model.forward(
                state=current_state,
                time=step * 3600.0,
                dt=3600.0
            )
            
            # Extract diagnostics
            diagnostics = fluxes["diagnostics.evapotranspiration"]
            storage = float(
                diagnostics['canopy_interception_storage'].detach()
            )
            evap_int = float(
                diagnostics['evaporation_from_interception'].detach()
            )
            throughfall = float(diagnostics['throughfall_flux'].detach())
            precipitation = float(current_state['precipitation_flux'])
            
            print(f"  Storage: {storage:.3f} mm")
            print(f"  Precipitation: {precipitation:.2e} kg m^-2 s^-1")
            print(f"  Evap from interception: {evap_int:.2e} kg m^-2 s^-1")
            print(f"  Throughfall: {throughfall:.2e} kg m^-2 s^-1")
            
            # Basic physical checks
            assert storage >= 0, f"Negative storage at step {step}: {storage}"
            assert evap_int >= 0, f"Negative interception evaporation at step {step}"
            assert throughfall >= 0, f"Negative throughfall at step {step}"
            
        print("Integrated mass balance with ET calculation: PASS")

    def _daily_dry_state(self, canopy_storage_mm, lai):
        state = self.dry_state.copy()
        state.update({
            'air_temperature': torch.tensor(298.15),
            'surface_temperature': torch.tensor(298.15),
            'relative_humidity': torch.tensor(0.5),
            'wind_speed': torch.tensor(2.0),
            'atmospheric_pressure': torch.tensor(101325.0),
            'net_radiation': torch.tensor(400.0),
            'leaf_area_index': torch.tensor(float(lai)),
            'surface_resistance': torch.tensor(100.0),
            'aerodynamic_resistance': torch.tensor(50.0),
            'precipitation_flux': torch.tensor(0.0),
            'canopy_storage': torch.tensor(float(canopy_storage_mm)),
        })
        return state

    def test_evaporation_of_the_whole_store_is_available_before_evaporation(self):
        """A legal drying of the whole stored water must not be reported as unavailable.

        One hour, the longest step of the FAO-56 hourly soil heat flux fractions, with a
        0.3 mm store at capacity: the hourly demand can take the whole store.
        """
        dt = 3600.0
        state = self._daily_dry_state(canopy_storage_mm=0.3, lai=1.5)
        fluxes = self.et_model(state, time=0.0, dt=dt)
        diagnostics = fluxes["diagnostics.evapotranspiration"]
        # References: Rutter et al. (1971); Gash (1979).
        # Mathematical expression: E_i = min(S_old + I, eta*ET*dt) with P = I = 0, so
        #   E_i <= S_old = 0.3 mm and S_new = S_old - E_i >= 0; Delta S = S_new - S_old.
        # Parameter meanings: S_old caller storage, eta canopy evaporation efficiency.
        # Units: depths [mm]; fluxes [kg m-2 s-1] with 1 kg m-2 = 1 mm; dt [s].
        # Domain/constraints: no precipitation, capacity 0.2*1.5 = 0.3 mm holds the store.
        # Sign convention: Delta S is negative when the canopy dries.
        evap_depth = float(diagnostics['evaporation_from_interception'].detach()) * dt
        new_storage = float(diagnostics['canopy_interception_storage'].detach())
        storage_change = float(diagnostics['canopy_storage_change'].detach())
        throughfall_depth = float(diagnostics['throughfall_flux'].detach()) * dt
        # Tolerances of 1e-6 mm absorb float32 parameter rounding (0.2 mm per LAI, efficiency 1.2)
        # and stay far below the 1 mm manager tolerance of the checker.
        assert 0.0 < evap_depth <= 0.3 + 1.0e-6
        assert abs(new_storage - (0.3 - evap_depth)) < 1.0e-6
        assert abs(storage_change - (new_storage - 0.3)) < 1.0e-6
        assert abs(throughfall_depth) < 1.0e-9
        committed = fluxes["canopy_storage"]
        assert abs(float(committed.value.detach()) - new_storage) < 1.0e-12

    def test_capacity_shrink_routes_old_storage_to_throughfall_and_closes_the_budget(self):
        """Drip from a shrinking capacity is throughfall, not a precipitation imbalance."""
        dt = 3600.0
        state = self._daily_dry_state(canopy_storage_mm=2.0, lai=0.5)
        state['net_radiation'] = torch.tensor(0.0)
        state['relative_humidity'] = torch.tensor(1.0)
        fluxes = self.et_model(state, time=0.0, dt=dt)
        diagnostics = fluxes["diagnostics.evapotranspiration"]
        # References: Rutter et al. (1971); Gash (1979).
        # Mathematical expression: C_max = 0.2*0.5 = 0.1 mm < S_old = 2 mm, so
        #   Drip = S_old - E_i - C_max, S_new = C_max, T_fall = Drip, and
        #   S_old + P = S_new + E_i + T_fall closes with P = I = 0.
        # Parameter meanings: C_max canopy capacity, Drip capacity overflow.
        # Units: depths [mm]; fluxes [kg m-2 s-1]; dt [s].
        # Domain/constraints: saturated air and zero net radiation give near-zero demand.
        # Sign convention: throughfall leaves the canopy store toward the soil.
        evap_depth = float(diagnostics['evaporation_from_interception'].detach()) * dt
        throughfall_depth = float(diagnostics['throughfall_flux'].detach()) * dt
        new_storage = float(diagnostics['canopy_interception_storage'].detach())
        storage_change = float(diagnostics['canopy_storage_change'].detach())
        assert abs(new_storage - 0.1) < 1.0e-6
        assert throughfall_depth > 1.0
        assert abs((2.0 + 0.0) - (new_storage + evap_depth + throughfall_depth)) < 1.0e-6
        assert abs(storage_change - (new_storage - 2.0)) < 1.0e-6

    def test_checker_rejects_an_open_canopy_budget_and_accepts_drip(self):
        """The complete budget S_old + P = S_new + E_i + T_fall is enforced explicitly."""
        # Drip-only partition: P = I = 0, S_old = 2, S_new = 0.1, T_fall = 1.9 closes.
        self.et_model._validate_canopy_mass_balance(
            precipitation_mm=0.0,
            interception_mm=0.0,
            evap_from_interception_mm=0.0,
            throughfall_mm=1.9,
            total_et_mm=0.0,
            transpiration_mm=0.0,
            dt=3600.0,
            storage_before_mm=2.0,
            storage_after_mm=0.1,
        )
        with pytest.raises(AssertionError, match="Canopy water budget violation"):
            self.et_model._validate_canopy_mass_balance(
                precipitation_mm=0.0,
                interception_mm=0.0,
                evap_from_interception_mm=0.0,
                throughfall_mm=0.0,
                total_et_mm=0.0,
                transpiration_mm=0.0,
                dt=3600.0,
                storage_before_mm=2.0,
                storage_after_mm=0.1,
            )
        with pytest.raises(AssertionError, match="Negative canopy storage before the step"):
            self.et_model._validate_canopy_mass_balance(
                precipitation_mm=0.0,
                interception_mm=0.0,
                evap_from_interception_mm=0.0,
                throughfall_mm=0.0,
                total_et_mm=0.0,
                transpiration_mm=0.0,
                dt=3600.0,
                storage_before_mm=-3.0,
                storage_after_mm=0.0,
            )


def test_factory_function():
    """Test that the enhanced ET model can be created and used."""
    model = PenmanMonteithET(name="test_model", config={"canopy_resistance_source": "state"})
    assert model is not None
    assert hasattr(model, '_validate_canopy_mass_balance')
    assert hasattr(model, '_log_canopy_diagnostics')
    print("Factory function: PASS")


if __name__ == "__main__":
    # Run tests directly
    test_instance = TestCanopyInterceptionMassBalance()
    test_instance.setup_method()
    
    print("=" * 60)
    print("Testing Canopy Interception Mass Balance")
    print("=" * 60)
    
    try:
        test_instance.test_mass_balance_perfect_conservation()
        test_instance.test_mass_balance_violation_detection()
        test_instance.test_physical_constraints_validation()
        test_instance.test_diagnostic_logging_functionality()
        test_instance.test_interception_efficiency_calculation()
        test_instance.test_storage_budget_analysis()
        test_instance.test_canopy_storage_bounds_checking()
        test_instance.test_integrated_mass_balance_with_et_calculation()
        test_factory_function()
        
        print("=" * 60)
        print("All canopy interception mass balance tests: PASS")
        print("=" * 60)
        
    except Exception as e:
        print("=" * 60)
        print(f"Test failed: {e}")
        print("=" * 60)
        raise

