"""
Comprehensive L1 Unit Tests for Hydrosphere Module

Tests all major hydrosphere components including:
- P0: Richards equation solver
- P1: Capillary rise and groundwater dynamics  
- P2: Canopy interception-ET coupling and snow albedo dynamics

This test suite ensures all modules work correctly individually
and validates the complete interface mapping system.
"""

import pytest
import torch
import sys
import os

# Add relic to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from relic.scientific.hydrosphere.soil_moisture.richards import RichardsSoilMoisture
from relic.scientific.hydrosphere.transport.capillary_rise import CapillaryRiseTransport
from relic.scientific.hydrosphere.groundwater.dynamics import GroundwaterDynamics
from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import PenmanMonteithET
from relic.scientific.hydrosphere.snow.albedo import SnowAlbedo
from relic.scientific.hydrosphere.interface_mapping import hydrosphere_mapper
from relic.core.state import UpdateItem
from relic.parameters import get_global_manager


def _unwrap_update(value):
    if isinstance(value, UpdateItem):
        return value.value
    return value


def _get_richards_diagnostics(result):
    diagnostics = result.get("diagnostics.soil_moisture_richards", {})
    if isinstance(diagnostics, dict):
        return diagnostics
    return {}


@pytest.fixture
def parameter_manager_snapshot():
    """Snapshot and restore ParameterManager state."""
    manager = get_global_manager()
    snapshot = manager.snapshot()
    yield manager
    manager.restore(snapshot)


class TestComprehensiveHydrosphere:
    """Comprehensive test suite for all hydrosphere components."""
    
    @pytest.fixture
    def standard_state(self):
        """Standard state variables for testing."""
        return {
            # Forcing variables
            "precipitation_flux": torch.tensor(1e-5),  # kg m-2 s-1 (light rain)
            "net_radiation": torch.tensor(200.0),  # W m-2
            "air_temperature": torch.tensor(293.15),  # K (20 C)
            "relative_humidity": torch.tensor(0.65),  # 65%
            "wind_speed": torch.tensor(2.5),  # m s-1
            "atmospheric_pressure": torch.tensor(101325.0),  # Pa
            "surface_resistance": torch.tensor(150.0),  # s m-1
            
            # State variables
            "soil_moisture": torch.tensor([0.25, 0.30, 0.35, 0.40, 0.38]),  # m3 m-3
            "soil_temperature": torch.tensor([293.15, 292.15, 291.15, 290.15, 289.15]),  # K
            "groundwater_storage": torch.tensor(150.0),  # mm
            "water_table_depth": torch.tensor(2.5),  # m
            "leaf_area_index": torch.tensor(3.0),  # dimensionless
            
            # Time parameters
            "time": 0.0,
            "dt": 3600.0  # 1 hour
        }
    
    @pytest.fixture
    def richards_overrides(self):
        """Parameter overrides for Richards equation tests."""
        return {
            "hydrosphere.soil_moisture.layer_count": (5, "dimensionless"),
            "hydrosphere.soil_moisture.soil_depth": (1.0, "m"),
            "hydrosphere.soil_moisture.residual_water_content": (0.05, "m3 m-3"),
            "hydrosphere.soil_moisture.porosity": (0.45, "m3 m-3"),
            "hydrosphere.soil_moisture.alpha_vg": (5.0, "m-1"),
            "hydrosphere.soil_moisture.n_vg": (1.5, "dimensionless"),
            "hydrosphere.soil_moisture.saturated_hydraulic_conductivity": (1e-5, "m s-1"),
        }
    
    @pytest.fixture
    def richards_model(self, parameter_manager_snapshot, richards_overrides):
        """Create Richards model instance."""
        manager = parameter_manager_snapshot
        for key, (value, units) in richards_overrides.items():
            manager.set_value(key, value, units=units, validate=True, notify=False)
        return RichardsSoilMoisture("test_richards")
    
    def test_richards_full_solution(self, richards_model, standard_state):
        """Test Richards equation full PDE solution (P0)."""
        richards = richards_model
        
        # Test canonical IPHS runtime path
        assert richards.runtime_path == "canonical_iphs_mixed_form_richards"
        assert not hasattr(richards, "enable_simplified_mode")
        
        richards_state = {
            **standard_state,
            "infiltration_flux": standard_state["precipitation_flux"],
            "evapotranspiration_flux": torch.tensor(0.0),
        }
        result = richards.forward(richards_state, 0.0, 3600.0)

        # Check required outputs
        assert "soil_moisture" in result
        assert "diagnostics.soil_moisture_richards" in result
        diagnostics = _get_richards_diagnostics(result)
        assert "pressure_head" in diagnostics
        assert "hydraulic_conductivity" in diagnostics
        assert "vertical_flux" in diagnostics
        assert "infiltration_capacity" in diagnostics
        assert "mass_conservation_error" in diagnostics
        assert "solver_converged" in diagnostics

        # Validate physical constraints
        pressure_head = diagnostics["pressure_head"]
        assert torch.all(pressure_head <= 0), "Pressure heads should be negative (unsaturated)"

        conductivity = diagnostics["hydraulic_conductivity"]
        assert torch.all(conductivity > 0), "Hydraulic conductivity should be positive"
        assert torch.all(conductivity <= richards.k_saturated), "K should not exceed K_sat"

        # Check mass conservation
        conservation_error = diagnostics["mass_conservation_error"]
        assert conservation_error < 0.01, f"Mass conservation error too large: {conservation_error:.2e}"

        # Check solver convergence
        assert diagnostics["solver_converged"] == True, "Solver should converge"
        
    def test_capillary_rise_integration(self, standard_state):
        """Test capillary rise transport (P1)."""
        capillary = CapillaryRiseTransport("test_capillary")

        capillary_state = dict(standard_state)
        assert "water_table_depth" in capillary_state
        result = capillary.forward(capillary_state, 0.0, 3600.0)
        
        # Check outputs
        assert "capillary_rise_flux" in result
        assert "diagnostics.capillary_rise" in result
        diagnostics = result["diagnostics.capillary_rise"]
        assert "capillary_rise_profile" in diagnostics
        assert "effective_capillary_height" in diagnostics
        assert "capillary_availability" in diagnostics

        # Validate flux magnitude and direction (upward)
        surface_flux = float(_unwrap_update(result["capillary_rise_flux"]))
        assert surface_flux >= 0, "Capillary flux should be upward (positive)"

        # Validate availability factor
        availability = float(_unwrap_update(diagnostics["capillary_availability"]))
        assert 0 <= availability <= 1, "Availability should be in [0,1]"

        # Check depth profile consistency
        profile = diagnostics["capillary_rise_profile"]
        assert torch.all(profile >= 0), "All layer fluxes should be non-negative"
        
    def test_groundwater_dynamics_integration(self, standard_state):
        """Test groundwater dynamics (P1)."""
        groundwater = GroundwaterDynamics("test_groundwater")
        capillary = CapillaryRiseTransport("test_groundwater_capillary")
        dt = 3600.0
        capillary_flux = _unwrap_update(
            capillary.forward(standard_state, 0.0, dt)["capillary_rise_flux"]
        )

        gw_state = {
            **standard_state,
            "drainage_flux": torch.tensor(5.0 / dt),
            "capillary_rise_flux": capillary_flux,
            "lateral_inflow": torch.tensor(2.0 / dt),
            "groundwater_pumping": torch.tensor(0.0),
        }
        
        result = groundwater.forward(gw_state, 0.0, dt)
        
        # Check required outputs
        assert set(result) == {
            "groundwater_storage",
            "water_table_depth",
            "baseflow_flux",
            "diagnostics.groundwater",
        }

        # Validate state constraints
        storage = float(_unwrap_update(result["groundwater_storage"]))
        assert storage >= 0, "Storage should be non-negative"

        wt_depth = float(_unwrap_update(result["water_table_depth"]))
        assert wt_depth >= 0.1, "Water table depth should be positive"
        assert wt_depth <= float(groundwater.bedrock_depth), "WT should not exceed bedrock"

        saturation = float(result["diagnostics.groundwater"]["saturation_fraction"])
        assert 0 <= saturation <= 1, "Saturation fraction should be in [0,1]"
        
    def test_canopy_interception_et_coupling(self, standard_state):
        """Test canopy interception-ET coupling (P2)."""
        # Enable canopy interception
        et_config = {
            "parameters": {
                "enable_canopy_interception": {"value": True, "units": "dimensionless"},
                "interception_capacity_per_LAI": {"value": 0.2, "units": "mm"},
                "canopy_evaporation_efficiency": {"value": 1.2, "units": "dimensionless"}
            }
        }
        
        penman_monteith = PenmanMonteithET("test_et", {**et_config, "canopy_resistance_source": "state"})
        # The constructor config does not switch interception on; enable it explicitly
        # and give the canopy store it partitions (no store is invented any more).
        penman_monteith.enable_canopy_interception = True

        result = penman_monteith.forward(
            {**standard_state, "canopy_storage": torch.tensor(0.0)}, 0.0, 3600.0
        )

        # Check enhanced outputs with interception
        assert "evapotranspiration_flux" in result
        diagnostics = result["diagnostics.evapotranspiration"]
        assert "evaporation_from_interception" in diagnostics
        assert "transpiration_flux" in diagnostics
        assert "throughfall_flux" in diagnostics
        assert "canopy_interception_storage" in diagnostics
        assert "canopy_storage_change" in diagnostics
        
        # Validate flux partitioning
        total_et = _unwrap_update(result["evapotranspiration_flux"])
        interception_evap = _unwrap_update(diagnostics["evaporation_from_interception"])
        transpiration = _unwrap_update(diagnostics["transpiration_flux"])
        
        # Mass balance: total ET = interception evap + transpiration
        balance_error = abs(total_et - (interception_evap + transpiration))
        relative_error = balance_error / (total_et + 1e-12)
        assert relative_error < 0.01, f"ET partitioning error: {relative_error:.2e}"
        
        # Check physical constraints
        assert total_et >= 0, "Total ET should be non-negative"
        assert interception_evap >= 0, "Interception evaporation should be non-negative"
        assert transpiration >= 0, "Transpiration should be non-negative"
        
        # Throughfall should be less than or equal to precipitation
        throughfall = diagnostics["throughfall_flux"]
        precipitation = standard_state["precipitation_flux"]
        assert torch.all(throughfall <= precipitation + 1e-12), "Throughfall cannot exceed precipitation"
        
    def test_snow_albedo_dynamics(self, standard_state):
        """Test snow albedo dynamics (P2)."""
        # Create snow conditions
        snow_state = {
            **standard_state,
            "snowfall": torch.tensor(0.01),  # kg m-2 s-1 (moderate snowfall)
            "snow_surface_temperature": torch.tensor(268.15),  # K (-5 C)
            "snow_water_equivalent": torch.tensor(50.0),  # mm
            # 50 mm SWE at a bulk density of 300 kg m-3 is 0.05 m * 1000 / 300 = 0.167 m.
            "snow_depth": torch.tensor(0.167),  # m
            "solar_zenith_angle": torch.tensor(0.8)  # radians
        }
        
        # Test both BATS and age-based schemes
        for scheme in ["bats", "age_based"]:
            albedo_config = {
                "parameters": {
                    "albedo_scheme": {"value": scheme, "units": "dimensionless"},
                    "fresh_snow_albedo": {"value": 0.85, "units": "dimensionless"},
                    "old_snow_albedo": {"value": 0.5, "units": "dimensionless"},
                    "snow_reset_threshold": {"value": 0.005, "units": "kg m-2 s-1"}
                }
            }
            
            snow_albedo = SnowAlbedo(f"test_albedo_{scheme}", albedo_config)
            
            # Fresh snow: the scheme-owned age state starts at zero (no state default).
            result = snow_albedo.forward(
                {**snow_state, snow_albedo.age_state_key: torch.tensor(0.0)}, 0.0, 86400.0
            )  # 1 day
            
            # Check outputs
            assert "surface_albedo" in result
            assert snow_albedo.age_state_key in result
            diagnostics = result["diagnostics.snow_albedo"]
            assert "aging_rate" in diagnostics
            assert "fresh_snow_reset" in diagnostics
            
            # Validate albedo range
            albedo = _unwrap_update(result["surface_albedo"])
            assert 0.1 <= albedo <= 0.95, f"Albedo out of range: {albedo:.3f}"
            
            # Check fresh snow reset
            fresh_reset = diagnostics["fresh_snow_reset"]
            assert isinstance(fresh_reset.item(), bool), "Fresh snow reset should be boolean"
            
            # Age should be non-negative
            age = _unwrap_update(result[snow_albedo.age_state_key])
            assert age >= 0, "Snow age should be non-negative"
            
            # Aging rate should be positive during aging
            aging_rate = diagnostics["aging_rate"]
            assert aging_rate >= 0, "Aging rate should be non-negative"
            
    def test_interface_mapping_consistency(self):
        """Test interface mapping system consistency."""
        # Test variable name resolution
        standard_names = [
            "precipitation_flux",
            "evapotranspiration_flux",
            "soil_moisture"
        ]
        
        for name in standard_names:
            standard_name = hydrosphere_mapper.get_standard_name(name)
            assert standard_name == name, f"Failed to resolve: {name}"
            
            info = hydrosphere_mapper.get_variable_info(name)
            assert info.units is not None, f"Missing units for: {name}"
            assert info.description is not None, f"Missing description for: {name}"
        
        # Test new P1/P2 variables
        new_variables = [
            "capillary_rise_flux",
            "water_table_depth",
            "lateral_inflow",
            "groundwater_pumping",
            "baseflow_flux",
            "evaporation_from_interception",
            "transpiration_flux",
            "surface_albedo",
            "snow_surface_age"
        ]
        
        for var in new_variables:
            try:
                standard_name = hydrosphere_mapper.get_standard_name(var)
                units = hydrosphere_mapper.get_units(var)
                category = hydrosphere_mapper.get_category(var)
                
                assert standard_name is not None, f"Failed to resolve new variable: {var}"
                assert units is not None, f"Missing units for new variable: {var}"
                assert category is not None, f"Missing category for new variable: {var}"
                
            except ValueError as e:
                pytest.fail(f"New variable not in mapping: {var} - {e}")
        
        # Test interface validation
        input_vars = ["precipitation_flux", "air_temperature", "net_radiation"]
        # The soil water content rate is reported under the soil water diagnostics group.
        output_vars = ["evapotranspiration_flux", "diagnostics.soil_moisture_richards"]
        
        issues = hydrosphere_mapper.validate_interface_consistency(input_vars, output_vars)
        assert len(issues) == 0, f"Interface validation issues: {issues}"
        
    def test_coupled_system_integration(self, standard_state):
        """Test coupled system with multiple components."""
        # Create models
        richards = RichardsSoilMoisture("coupled_richards")
        capillary = CapillaryRiseTransport("coupled_capillary")
        groundwater = GroundwaterDynamics("coupled_groundwater") 
        penman_monteith = PenmanMonteithET("coupled_et", config={"canopy_resistance_source": "state"})
        
        # Extended state for coupled system
        coupled_state = {
            **standard_state,
            "lateral_inflow": torch.tensor(1.0 / 3600.0),
            "groundwater_pumping": torch.tensor(0.0),
        }
        
        # Run components in sequence (simplified coupling)
        dt = 3600.0
        
        # 1. Capillary rise reads the accepted start-of-step groundwater depth.
        cap_result = capillary.forward(coupled_state, 0.0, dt)
        capillary_flux = _unwrap_update(cap_result["capillary_rise_flux"])
        richards.set_boundary_conditions(
            bottom={"type": "capillary_flux"}
        )
        richards_state = {
            **coupled_state,
            "capillary_rise_flux": capillary_flux,
        }
        richards_state["soil_moisture"] = torch.linspace(
            0.25,
            0.38,
            steps=richards.soil_layers,
        )
        richards_state["infiltration_flux"] = standard_state[
            "precipitation_flux"
        ]
        richards_state["evapotranspiration_flux"] = torch.tensor(0.0)

        # 2. Richards credits capillary inflow and emits canonical drainage.
        richards_result = richards.forward(richards_state, 0.0, dt)

        # 3. Groundwater debits the same capillary flux and its baseflow export.
        groundwater_state = {
            **coupled_state,
            "drainage_flux": _unwrap_update(richards_result["drainage_flux"]),
            "capillary_rise_flux": capillary_flux,
        }
        gw_result = groundwater.forward(groundwater_state, 0.0, dt)
        
        # 4. ET uses the updated Richards soil moisture.
        updated_state = dict(coupled_state)
        updated_state["soil_moisture"] = _unwrap_update(
            richards_result["soil_moisture"]
        )
        et_result = penman_monteith.forward(updated_state, 0.0, dt)
        
        # Validate coupled system outputs
        for result in [gw_result, cap_result, richards_result, et_result]:
            for val in result.values():
                val_unwrapped = _unwrap_update(val)
                if torch.is_tensor(val_unwrapped):
                    assert torch.all(torch.isfinite(val_unwrapped))
        
        # Check water balance components
        capillary_input = float(_unwrap_update(cap_result["capillary_rise_flux"]))
        et_output = _unwrap_update(et_result["evapotranspiration_flux"])
        soil_change = _unwrap_update(
            richards_result[
                "diagnostics.soil_moisture_richards.soil_moisture_change_rate"
            ]
        )
        
        # All fluxes should be physically reasonable
        assert capillary_input >= 0, "Capillary input should be non-negative"
        assert et_output >= 0, "ET should be non-negative"
        assert torch.all(torch.isfinite(soil_change)), "Soil moisture changes should be finite"
        
    def test_extreme_conditions(self):
        """Test model behavior under extreme conditions."""
        # Test very dry conditions
        dry_state = {
            "precipitation_flux": torch.tensor(0.0),
            "net_radiation": torch.tensor(400.0),  # High radiation
            "air_temperature": torch.tensor(313.15),  # 40 C
            "relative_humidity": torch.tensor(0.1),  # Very dry
            "wind_speed": torch.tensor(10.0),  # High wind
            "atmospheric_pressure": torch.tensor(101325.0),
            "surface_resistance": torch.tensor(500.0),  # High resistance
            "soil_moisture": torch.tensor([0.05, 0.06, 0.07, 0.08, 0.09]),  # Very dry
            "water_table_depth": torch.tensor(10.0),  # Very deep
            "groundwater_storage": torch.tensor(5.0),  # Very low
            "leaf_area_index": torch.tensor(0.5),  # Sparse vegetation
            "dt": 3600.0
        }
        
        # Models should handle extreme conditions gracefully
        et_model = PenmanMonteithET("extreme_et", config={"canopy_resistance_source": "state"})
        capillary_model = CapillaryRiseTransport("extreme_cap")
        
        # ET under dry conditions
        et_result = et_model.forward(dry_state, 0.0, 3600.0)
        et_flux = _unwrap_update(et_result["evapotranspiration_flux"])
        assert torch.isfinite(et_flux)
        assert et_flux >= 0
        
        # Capillary rise under deep water table
        cap_result = capillary_model.forward(dry_state, 0.0, 3600.0)
        cap_flux = float(_unwrap_update(cap_result["capillary_rise_flux"]))
        assert cap_flux == 0.0  # Should be zero for deep WT
        
        # Test very wet conditions
        wet_state = {
            "precipitation_flux": torch.tensor(0.01),  # Heavy rain
            "net_radiation": torch.tensor(50.0),  # Low radiation
            "air_temperature": torch.tensor(278.15),  # 5 C
            "relative_humidity": torch.tensor(0.95),  # Very humid
            "wind_speed": torch.tensor(0.5),  # Low wind
            "atmospheric_pressure": torch.tensor(101325.0),
            "surface_resistance": torch.tensor(50.0),  # Low resistance
            "soil_moisture": torch.tensor([0.42, 0.41, 0.40, 0.39, 0.38]),  # Near saturation
            "water_table_depth": torch.tensor(0.3),  # Very shallow
            "groundwater_storage": torch.tensor(300.0),  # High storage
            "leaf_area_index": torch.tensor(6.0),  # Dense vegetation
            "dt": 3600.0
        }
        
        et_result = et_model.forward(wet_state, 0.0, 3600.0)
        cap_result = capillary_model.forward(wet_state, 0.0, 3600.0)
        
        # All results should be finite and physically reasonable
        et_flux = _unwrap_update(et_result["evapotranspiration_flux"])
        assert torch.isfinite(et_flux)
        assert et_flux >= 0
        cap_flux = _unwrap_update(cap_result["capillary_rise_flux"])
        if isinstance(cap_flux, torch.Tensor):
            cap_flux = cap_flux.detach()
        assert float(cap_flux) > 0  # Should have capillary rise
        
    def test_units_and_conversions(self, standard_state):
        """Test unit consistency and conversions."""
        et_model = PenmanMonteithET("units_test", config={"canopy_resistance_source": "state"})
        
        result = et_model.forward(standard_state, 0.0, 3600.0)
        
        # ET flux should be in kg m-2 s-1
        et_flux = _unwrap_update(result["evapotranspiration_flux"])
        
        # Typical ET rates: 0.1-10 mm day-1 = 1e-6 to 1e-4 kg m-2 s-1
        assert 1e-7 <= et_flux <= 1e-3, f"ET flux outside reasonable range: {et_flux:.2e} kg m-2 s-1"
        
        # Convert to mm day-1 for interpretation
        et_mm_day = et_flux * 86400 * 1000 / 1000  # kg m-2 s-1 to mm day-1
        assert 0.01 <= et_mm_day <= 50, f"ET in mm day-1 outside range: {et_mm_day:.2f}"
        
        # Check vapor pressure deficit units (Pa)
        diagnostics = result["diagnostics.evapotranspiration"]
        vpd = diagnostics["vapor_pressure_deficit"]
        assert 0 <= vpd <= 5000, f"VPD outside range: {vpd:.1f} Pa"  # 0-5 kPa typical
        
        # Check conductances (m s-1)
        aerodynamic_conductance = diagnostics["aerodynamic_conductance"]
        canopy_conductance = diagnostics["canopy_conductance"]
        
        assert 0.001 <= aerodynamic_conductance <= 1.0, f"ga outside range: {aerodynamic_conductance:.3f}"
        assert 1e-6 <= canopy_conductance <= 0.1, f"gc outside range: {canopy_conductance:.3f}"


class TestModuleInteroperability:
    """Test interoperability between different hydrosphere modules."""
    
    def test_richards_capillary_coupling(self):
        """Test Richards equation with capillary rise boundary condition."""
        richards = RichardsSoilMoisture("test_richards")
        capillary = CapillaryRiseTransport("test_capillary")
        
        state = {
            "soil_moisture": torch.linspace(
                0.2,
                0.4,
                steps=richards.soil_layers,
            ),
            "water_table_depth": torch.tensor(1.5),
            "precipitation_flux": torch.tensor(1e-6)
        }
        
        # Get capillary rise flux
        cap_result = capillary.forward(state, 0.0, 3600.0)
        
        # Use the canonical capillary mass flux bottom boundary.
        capillary_flux = _unwrap_update(cap_result["capillary_rise_flux"])
        richards.set_boundary_conditions(
            bottom={"type": "capillary_flux"}
        )
        richards_state = {
            **state,
            "capillary_rise_flux": capillary_flux,
            "infiltration_flux": state["precipitation_flux"],
            "evapotranspiration_flux": torch.tensor(0.0),
        }
        
        richards_result = richards.forward(richards_state, 0.0, 3600.0)
        
        # Results should be physically consistent
        soil_change = _unwrap_update(
            richards_result[
                "diagnostics.soil_moisture_richards.soil_moisture_change_rate"
            ]
        )
        assert torch.all(torch.isfinite(soil_change))
        
    def test_et_interception_coupling(self):
        """Test ET model with interception coupling."""
        et_config = {
            "parameters": {
                "enable_canopy_interception": {"value": True, "units": "dimensionless"},
                "interception_capacity_per_LAI": {"value": 0.15, "units": "mm"}
            }
        }
        
        et_model = PenmanMonteithET("interception_test", {**et_config, "canopy_resistance_source": "state"})
        
        # Light rain with vegetation
        state = {
            "precipitation_flux": torch.tensor(3e-6),  # kg m-2 s-1 (light rain)
            "net_radiation": torch.tensor(150.0),
            "air_temperature": torch.tensor(288.15),
            "relative_humidity": torch.tensor(0.8),
            "wind_speed": torch.tensor(1.5),
            "atmospheric_pressure": torch.tensor(101325.0),
            "surface_resistance": torch.tensor(120.0),
            "leaf_area_index": torch.tensor(2.5)
        }
        
        result = et_model.forward(state, 0.0, 3600.0)

        # Should have both interception and transpiration
        diagnostics = result["diagnostics.evapotranspiration"]
        interception_evap = diagnostics["evaporation_from_interception"]
        transpiration = diagnostics["transpiration_flux"]
        
        assert interception_evap >= 0, "Interception evaporation should be non-negative"
        assert transpiration >= 0, "Transpiration should be non-negative"
        
        # Throughfall should be less than precipitation
        throughfall = diagnostics["throughfall_flux"]
        precipitation = state["precipitation_flux"]
        assert torch.all(throughfall <= precipitation + 1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
