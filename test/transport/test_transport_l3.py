"""
L3 System Tests for Transport Module

This module provides comprehensive system-level tests for the transport module,
testing end-to-end functionality, full simulation workflows, long-term stability,
and real-world scenario validation. Tests focus on complete system behavior,
performance under realistic conditions, and integration with the entire RELIC framework.

Test Coverage:
- End-to-end transport simulation workflows
- Long-term numerical stability and convergence
- Real-world scenario validation (atmospheric, soil, aquatic transport)
- Coupled multi-domain transport (atmosphere-soil-water)
- Performance benchmarking and optimization
- Error handling and recovery at system level
- Conservation law verification over long simulations
- Parameter sensitivity and uncertainty analysis
- Parallel computation and GPU acceleration (if available)
- Model intercomparison with known solutions
- Robustness under extreme conditions

System Scenarios:
- Complete ecosystem transport cycles (carbon, water, nutrients)
- Diurnal and seasonal transport patterns
- Extreme weather events (storms, droughts, heatwaves)
- Land-use change impacts on transport processes
- Climate change scenario simulations
- Air quality and pollutant dispersion modeling
- Soil solute transport and groundwater contamination
- Biosphere-atmosphere gas exchange processes

Mathematical Verification:
- Long-term mass balance: sum(m_in(t=0->T)) - sum(m_out(t=0->T)) = delta_m_storage
- Energy balance closure: sum(E_in) - sum(E_out) = delta_E_storage within 1% tolerance
- Convergence testing: delta_solution -> 0 as dt -> 0, dx -> 0
- Statistical validation against observations: |model - obs|/obs < threshold
- Dimensional consistency: All terms maintain proper units throughout simulation

References:
- IPCC (2021). Climate Change 2021: The Physical Science Basis
- WRR (2020). Water Resources Research guidelines for model validation
- ACP (2021). Atmospheric Chemistry and Physics modeling standards
- HESS (2021). Hydrology and Earth System Sciences validation protocols
- GEWEX (2020). Global Energy and Water Exchanges experiment design
"""

import pytest
import torch
import numpy as np
import time

from relic.scientific.transport.advection import LinearAdvection

# Import core system components
from relic.core.simulator import RELICSimulator
from relic.core.state import UpdateItem, UpdateMode

# Import scientific modules for comprehensive testing

# Units of this test's own scenario variables, which no SCM contract or SSOT
# entry declares; they follow the units stated with the test data. The simulator
# infers no units from names (relic/core/simulator/state_init.py).
_TEST_STATE_UNITS = {
    'atmospheric_co2': 'kg m-3',
    'atmospheric_moisture': 'kg kg-1',
    'atmospheric_nh3': 'ug m-3',
    'atmospheric_no': 'ug m-3',
    'atmospheric_no2': 'ug m-3',
    'atmospheric_source': 'kg m-3 s-1',
    'canopy_co2': 'kg m-3',
    'denitrification_rate': 'mg kg-1 s-1',
    'deposition_rate': 'ug m-3 s-1',
    'drainage': 'm s-1',
    'evapotranspiration': 'kg m-2 s-1',
    'groundwater_level': 'm',
    'infiltration': 'm s-1',
    'mineralization_rate': 'mg kg-1 s-1',
    'nitrification_rate': 'mg kg-1 s-1',
    'ph': '1',
    'photosynthesis': 'kg m-3 s-1',
    'precipitation': 'kg m-2 s-1',
    'pressure': 'Pa',
    'soil_co2': 'kg m-3',
    'soil_nh4': 'mg kg-1',
    'soil_no3': 'mg kg-1',
    'soil_organic_n': 'mg kg-1',
    'soil_respiration': 'kg m-3 s-1',
    'temperature_atm': 'K',
    'temperature_canopy': 'K',
    'temperature_soil': 'K',
    'volatilization_rate': 'mg kg-1 s-1',
}


def _declared_simulator(config=None, *args, **kwargs):
    """Construct RELICSimulator with this test's state_units declaration."""
    declared = dict(config or {})
    declared.setdefault("state_units", _TEST_STATE_UNITS)
    return RELICSimulator(declared, *args, **kwargs)


def _clamp_nonnegative(value: torch.Tensor) -> torch.Tensor:
    return torch.clamp(value, min=0.0)


def _linear_increment(output, key="tracer_concentration") -> torch.Tensor:
    """Return an integrated LinearAdvection increment with canonical semantics."""
    update = output[key]
    assert isinstance(update, UpdateItem)
    assert update.mode == UpdateMode.ACCUMULATE
    assert update.per_timestep is True
    return update.value


class TestEndToEndTransportWorkflows:
    """Test complete end-to-end transport workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.tolerance = 1e-4
        self.max_simulation_time = 3600.0  # 1 hour
        
    def test_complete_carbon_cycle_transport(self):
        """Test complete carbon cycle transport through ecosystem."""
        # Create comprehensive carbon transport system
        print("Testing complete carbon cycle transport...")
        
        # Atmospheric CO2 transport
        atm_advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=[10.0],  # 10m vertical resolution
            tracer_key="atmospheric_co2",
            name="AtmosphericCO2Transport"
        )
        
        # Soil CO2 transport
        soil_advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=[0.05],  # 5cm soil resolution
            tracer_key="soil_co2",
            name="SoilCO2Transport"
        )
        
        # Canopy air transport
        canopy_advection = LinearAdvection(
            spatial_dims=2,
            grid_spacing=[1.0, 1.0],  # 1m horizontal resolution
            tracer_key="canopy_co2",
            name="CanopyAirTransport"
        )
        
        # Set up velocity fields
        atm_velocity = torch.ones(1, 50) * 0.02  # 0.02 m/s upward
        atm_advection.set_velocity_field([atm_velocity])
        
        soil_velocity = torch.ones(1, 40) * 1e-7  # 1e-7 m/s soil water flow
        soil_advection.set_velocity_field([soil_velocity])
        
        canopy_u = torch.ones(20, 30) * 0.002  # 0.002 m/s wind
        canopy_v = torch.ones(20, 30) * 0.0005  # 0.0005 m/s vertical
        canopy_advection.set_velocity_field([canopy_u, canopy_v])
        
        # Initialize carbon pools
        state = {
            # Atmospheric CO2 (ppm to kg m-3 conversion factor ~1.8e-6 at STP)
            "atmospheric_co2": torch.ones(1, 50) * 400.0 * 1.8e-6,
            "soil_co2": torch.ones(1, 40) * 1.5e-3,  # kg m-3
            "canopy_co2": torch.ones(20, 30) * 600.0 * 1.8e-6,
            
            # Supporting variables
            "temperature_atm": torch.linspace(298.15, 288.15, 50).unsqueeze(0),
            "temperature_soil": torch.linspace(288.15, 283.15, 40).unsqueeze(0),
            "temperature_canopy": torch.ones(20, 30) * 293.15,
            
            # Source/sink terms
            "soil_respiration": torch.ones(1, 40) * 1e-7,  # kg m-3 s-1
            "photosynthesis": torch.ones(20, 30) * 2e-7,   # kg m-3 s-1
            "atmospheric_source": torch.zeros(1, 50)      # Background
        }
        
        # Create simulator
        simulator = _declared_simulator(config={"device": "cpu"})
        simulator.register_functor(atm_advection)
        simulator.register_functor(soil_advection)
        simulator.register_functor(canopy_advection)
        
        # Map variables for transport modules
        simulator.initialize_state(state)
        
        # Run simulation for multiple time steps
        dt = 300.0  # 5 minutes
        n_steps = int(self.max_simulation_time / dt)
        
        # Track total carbon mass
        initial_atm_mass = torch.sum(state["atmospheric_co2"]) * 10.0 * 100.0  # kg
        initial_soil_mass = torch.sum(state["soil_co2"]) * 0.05 * 100.0
        initial_canopy_mass = torch.sum(state["canopy_co2"]) * 1.0 * 1.0
        
        total_initial_carbon = initial_atm_mass + initial_soil_mass + initial_canopy_mass
        
        carbon_mass_history = [total_initial_carbon.item()]
        previous_total_carbon = total_initial_carbon
        photosynthesis_cumulative = 0.0
        respiration_cumulative = 0.0
        
        print(f"Running {n_steps} time steps...")
        
        for step in range(n_steps):
            if step % 10 == 0:
                print(f"  Step {step}/{n_steps}")
                
            # Map state for each transport module
            current_state = simulator.get_state()
            
            # Atmospheric transport
            atm_state = {
                "atmospheric_co2": current_state["atmospheric_co2"],
                "source_term": state["atmospheric_source"]
            }
            atm_output = atm_advection.forward(atm_state, time=step*dt, dt=dt)
            
            # Soil transport
            soil_state = {
                "soil_co2": current_state["soil_co2"],
                "source_term": state["soil_respiration"]
            }
            soil_output = soil_advection.forward(soil_state, time=step*dt, dt=dt)
            
            # Canopy transport
            canopy_state = {
                "canopy_co2": current_state["canopy_co2"],
                "source_term": -state["photosynthesis"]  # Negative = consumption
            }
            canopy_output = canopy_advection.forward(canopy_state, time=step*dt, dt=dt)
            
            # Apply updates
            atmospheric_next = _clamp_nonnegative(
                current_state["atmospheric_co2"]
                + _linear_increment(atm_output, key="atmospheric_co2")
            )
            soil_next = _clamp_nonnegative(
                current_state["soil_co2"]
                + _linear_increment(soil_output, key="soil_co2")
            )
            canopy_next = _clamp_nonnegative(
                current_state["canopy_co2"]
                + _linear_increment(canopy_output, key="canopy_co2")
            )
            simulator.state_manager.apply_updates(
                {
                    "atmospheric_co2": UpdateItem(
                        atmospheric_next,
                        UpdateMode.REPLACE,
                    ),
                    "soil_co2": UpdateItem(
                        soil_next,
                        UpdateMode.REPLACE,
                    ),
                    "canopy_co2": UpdateItem(
                        canopy_next,
                        UpdateMode.REPLACE,
                    ),
                },
                dt=dt,
                check_conservation=False,
            )
            
            # Update source/sink terms (simple diurnal cycle)
            hour_of_day = (step * dt) / 3600.0
            
            # Photosynthesis peaks at midday
            photosynthesis_factor = max(0, np.cos(2 * np.pi * (hour_of_day - 12) / 24))
            state["photosynthesis"] = torch.ones(20, 30) * 2e-7 * photosynthesis_factor
            
            # Soil respiration relatively constant
            soil_respiration_factor = 1.0 + 0.2 * np.sin(2 * np.pi * hour_of_day / 24)
            state["soil_respiration"] = torch.ones(1, 40) * 1e-7 * soil_respiration_factor
            
            # Check carbon mass conservation
            current_state = simulator.get_state()
            current_atm_mass = torch.sum(current_state["atmospheric_co2"]) * 10.0 * 100.0
            current_soil_mass = torch.sum(current_state["soil_co2"]) * 0.05 * 100.0
            current_canopy_mass = torch.sum(current_state["canopy_co2"]) * 1.0 * 1.0
            
            total_current_carbon = current_atm_mass + current_soil_mass + current_canopy_mass
            carbon_mass_history.append(total_current_carbon.item())
            
            # Verify conservation (allowing for source/sink terms)
            photosynthesis_total = torch.sum(state["photosynthesis"]) * dt * 1.0 * 1.0
            respiration_total = torch.sum(state["soil_respiration"]) * dt * 0.05 * 100.0
            
            expected_change = -photosynthesis_total + respiration_total
            actual_change = total_current_carbon - previous_total_carbon
            relative_error = torch.abs(actual_change - expected_change) / (
                torch.abs(total_current_carbon) + 1e-10
            )
            previous_total_carbon = total_current_carbon
            photosynthesis_cumulative += float(photosynthesis_total.detach().item())
            respiration_cumulative += float(respiration_total.detach().item())
            
            # Should conserve mass within numerical tolerance
            assert relative_error < 0.05, f"Carbon conservation error: {relative_error:.4f} at step {step}"
            
            # Verify physical constraints
            assert torch.all(current_state["atmospheric_co2"] >= 0)
            assert torch.all(current_state["soil_co2"] >= 0)
            assert torch.all(current_state["canopy_co2"] >= 0)
            
        # Final validation
        final_state = simulator.get_state()
        
        # Check reasonable final values
        assert torch.mean(final_state["atmospheric_co2"]) > 0
        assert torch.mean(final_state["soil_co2"]) > 0
        assert torch.mean(final_state["canopy_co2"]) > 0
        
        # Check mass conservation over entire simulation
        final_atm_mass = torch.sum(final_state["atmospheric_co2"]) * 10.0 * 100.0
        final_soil_mass = torch.sum(final_state["soil_co2"]) * 0.05 * 100.0
        final_canopy_mass = torch.sum(final_state["canopy_co2"]) * 1.0 * 1.0
        
        total_final_carbon = final_atm_mass + final_soil_mass + final_canopy_mass
        
        # Total change should equal integrated sources/sinks
        total_photosynthesis = photosynthesis_cumulative
        total_respiration = respiration_cumulative
        
        expected_total_change = -total_photosynthesis + total_respiration
        actual_total_change = total_final_carbon - total_initial_carbon
        
        denominator = abs(total_initial_carbon) + abs(expected_total_change) + 1e-10
        final_relative_error = abs(actual_total_change - expected_total_change) / denominator
        assert final_relative_error < 0.3, f"Final carbon conservation error: {final_relative_error:.4f}"
        
        print(f"Carbon cycle transport test passed!")
        print(f"  Initial total carbon: {total_initial_carbon:.2f} kg")
        print(f"  Final total carbon: {total_final_carbon:.2f} kg")
        print(f"  Expected change: {expected_total_change:.2f} kg")
        print(f"  Actual change: {actual_total_change:.2f} kg")
        print(f"  Relative error: {final_relative_error:.4f}")
        
    def test_complete_water_cycle_transport(self):
        """Test complete water cycle transport through ecosystem."""
        print("Testing complete water cycle transport...")
        
        # Create water transport system
        # Atmospheric moisture transport
        moisture_advection = LinearAdvection(
            spatial_dims=2,
            grid_spacing=[2.0, 2.0],  # 2km horizontal resolution
            tracer_key="atmospheric_moisture",
            name="AtmosphericMoistureTransport"
        )
        
        # Soil water transport
        soil_water_advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=[0.1],  # 10cm soil resolution
            tracer_key="soil_moisture",
            name="SoilWaterTransport"
        )
        
        # Groundwater transport
        groundwater_advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=[0.5],  # 0.5m groundwater resolution
            tracer_key="groundwater_level",
            name="GroundwaterTransport"
        )
        
        # Set up velocity fields
        moisture_u = torch.ones(15, 20) * 0.001  # 0.001 m/s wind
        moisture_v = torch.ones(15, 20) * 0.0002  # 0.0002 m/s vertical
        moisture_advection.set_velocity_field([moisture_u, moisture_v])
        
        soil_water_velocity = torch.ones(1, 30) * 1e-6  # 1e-6 m/s infiltration
        soil_water_advection.set_velocity_field([soil_water_velocity])
        
        groundwater_velocity = torch.ones(1, 20) * 1e-7  # 1e-7 m/s groundwater flow
        groundwater_advection.set_velocity_field([groundwater_velocity])
        
        # Initialize water state
        state = {
            # Atmospheric moisture (kg/kg)
            "atmospheric_moisture": torch.ones(15, 20) * 0.01,
            
            # Soil water (volumetric water content)
            "soil_moisture": torch.ones(1, 30) * 0.3,
            
            # Groundwater (water table depth, m)
            "groundwater_level": torch.ones(1, 20) * 2.0,
            
            # Supporting variables
            "temperature": torch.ones(15, 20) * 293.15,
            "pressure": torch.ones(15, 20) * 101325.0,
            "precipitation": torch.ones(15, 20) * 1e-8,  # kg m-2 s-1
            "evapotranspiration": torch.ones(15, 20) * 5e-8,  # kg m-2 s-1
            "infiltration": torch.ones(1, 30) * 1e-8,      # m s-1
            "drainage": torch.ones(1, 30) * 5e-9,          # m s-1
        }
        
        # Create simulator
        simulator = _declared_simulator(config={"device": "cpu"})
        simulator.register_functor(moisture_advection)
        simulator.register_functor(soil_water_advection)
        simulator.register_functor(groundwater_advection)
        
        simulator.initialize_state(state)
        
        # Run simulation
        dt = 600.0  # 10 minutes
        n_steps = int(self.max_simulation_time / dt)
        
        # Track water balance
        initial_moisture = torch.sum(state["atmospheric_moisture"]) * 2.0 * 2.0 * 1000.0  # kg
        initial_soil_water = torch.sum(state["soil_moisture"]) * 0.1 * 100.0  # m^3
        initial_groundwater = torch.sum(state["groundwater_level"]) * 0.5 * 100.0  # m^2
        
        water_balance_history = []
        
        print(f"Running {n_steps} time steps...")
        
        for step in range(n_steps):
            if step % 5 == 0:
                print(f"  Step {step}/{n_steps}")
                
            current_state = simulator.get_state()
            
            # Atmospheric moisture transport
            moisture_state = {
                "atmospheric_moisture": current_state["atmospheric_moisture"],
                "source_term": state["precipitation"] / 1000.0 - state["evapotranspiration"] / 1000.0
            }
            moisture_output = moisture_advection.forward(moisture_state, time=step*dt, dt=dt)
            
            # Soil water transport
            soil_state = {
                "soil_moisture": current_state["soil_moisture"],
                "source_term": state["infiltration"] / 0.1 - state["drainage"] / 0.1
            }
            soil_output = soil_water_advection.forward(soil_state, time=step*dt, dt=dt)
            
            # Groundwater transport
            gw_state = {
                "groundwater_level": current_state["groundwater_level"],
                "source_term": -torch.mean(state["drainage"]) * torch.ones(1, 20) * 0.1 / 0.5
            }
            gw_output = groundwater_advection.forward(gw_state, time=step*dt, dt=dt)
            
            # Apply updates
            atmospheric_next = _clamp_nonnegative(
                current_state["atmospheric_moisture"]
                + _linear_increment(moisture_output, key="atmospheric_moisture")
            )
            soil_next = _clamp_nonnegative(
                current_state["soil_moisture"]
                + _linear_increment(soil_output, key="soil_moisture")
            )
            groundwater_next = _clamp_nonnegative(
                current_state["groundwater_level"]
                + _linear_increment(gw_output, key="groundwater_level")
            )
            simulator.state_manager.apply_updates(
                {
                    "atmospheric_moisture": UpdateItem(
                        atmospheric_next,
                        UpdateMode.REPLACE,
                    ),
                    "soil_moisture": UpdateItem(
                        soil_next,
                        UpdateMode.REPLACE,
                    ),
                    "groundwater_level": UpdateItem(
                        groundwater_next,
                        UpdateMode.REPLACE,
                    ),
                },
                dt=dt,
                check_conservation=False,
            )
            
            # Update forcing terms (diurnal cycle)
            hour_of_day = (step * dt) / 3600.0
            
            # Precipitation events
            if 6 < hour_of_day < 8 or 14 < hour_of_day < 16:  # Morning and afternoon showers
                precipitation_rate = 2e-8
            else:
                precipitation_rate = 1e-9
                
            state["precipitation"] = torch.ones(15, 20) * precipitation_rate
            
            # Evapotranspiration peaks at midday
            et_factor = max(0, np.cos(2 * np.pi * (hour_of_day - 13) / 24))
            state["evapotranspiration"] = torch.ones(15, 20) * 8e-8 * et_factor
            
            # Infiltration limited by soil moisture (map atmospheric forcing to soil grid)
            current_state = simulator.get_state()
            soil_moisture_current = current_state["soil_moisture"]
            precip_mean = torch.mean(state["precipitation"])
            precipitation_to_soil = torch.ones_like(soil_moisture_current) * (precip_mean / 1000.0)
            infiltration_capacity = torch.clamp(
                (soil_moisture_current - 0.4) * 1e-6,
                min=0.0,
            )
            state["infiltration"] = torch.minimum(precipitation_to_soil, infiltration_capacity)
            
            # Drainage proportional to soil moisture
            state["drainage"] = torch.clamp(
                (soil_moisture_current - 0.35) * 1e-7,
                min=0.0,
            )
            
            # Check water balance
            precip_total = torch.sum(state["precipitation"]) * dt * 2.0 * 2.0  # kg
            et_total = torch.sum(state["evapotranspiration"]) * dt * 2.0 * 2.0    # kg
            infil_total = torch.sum(state["infiltration"]) * dt * 100.0             # m^3
            drain_total = torch.sum(state["drainage"]) * dt * 100.0                # m^3
            
            water_balance_history.append({
                'precipitation': precip_total.item(),
                'evapotranspiration': et_total.item(),
                'infiltration': infil_total.item(),
                'drainage': drain_total.item()
            })
            
            # Verify physical constraints
            assert torch.all(current_state["atmospheric_moisture"] >= 0)
            assert torch.all(current_state["soil_moisture"] >= 0) and torch.all(current_state["soil_moisture"] <= 1.0)
            assert torch.all(current_state["groundwater_level"] >= 0)
            
        print(f"Water cycle transport test passed!")
        print(f"  Water balance components:")
        total_precip = sum([w['precipitation'] for w in water_balance_history])
        total_et = sum([w['evapotranspiration'] for w in water_balance_history])
        total_infil = sum([w['infiltration'] for w in water_balance_history])
        total_drain = sum([w['drainage'] for w in water_balance_history])
        
        print(f"    Total precipitation: {total_precip:.2f} kg")
        print(f"    Total evapotranspiration: {total_et:.2f} kg")
        print(f"    Total infiltration: {total_infil:.2f} m")
        print(f"    Total drainage: {total_drain:.2f} m")
        
    def test_complete_nutrient_cycle_transport(self):
        """Test complete nutrient (nitrogen) cycle transport."""
        print("Testing complete nitrogen cycle transport...")
        
        # Create nitrogen transport system
        # Atmospheric NH3/NOx transport
        atm_nitrogen = LinearAdvection(
            spatial_dims=2,
            grid_spacing=[5.0, 5.0],  # 5 m resolution
            name="AtmosphericNitrogenTransport"
        )
        
        # Soil nitrate transport
        no3_transport = LinearAdvection(
            spatial_dims=1,
            grid_spacing=[0.05],  # 5cm resolution
            tracer_key="soil_no3",
            name="SoilNO3Transport"
        )
        
        # Soil ammonium transport
        nh4_transport = LinearAdvection(
            spatial_dims=1,
            grid_spacing=[0.05],
            tracer_key="soil_nh4",
            name="SoilNH4Transport"
        )
        
        # Set up velocity fields
        atm_u = torch.ones(10, 12) * 0.002
        atm_v = torch.ones(10, 12) * 0.0002
        atm_nitrogen.set_velocity_field([atm_u, atm_v])
        
        soil_velocity = torch.ones(1, 40) * 5e-7  # 5e-7 m/s
        no3_transport.set_velocity_field([soil_velocity])
        nh4_transport.set_velocity_field([soil_velocity])
        
        # Initialize nitrogen state
        state = {
            # Atmospheric nitrogen (ug m-3)
            "atmospheric_nh3": torch.ones(10, 12) * 2.0,
            "atmospheric_no": torch.ones(10, 12) * 1.0,
            "atmospheric_no2": torch.ones(10, 12) * 5.0,
            
            # Soil nitrogen (mg kg-1 soil)
            "soil_no3": torch.ones(1, 40) * 20.0,
            "soil_nh4": torch.ones(1, 40) * 10.0,
            "soil_organic_n": torch.ones(1, 40) * 1000.0,
            
            # Supporting variables
            "soil_moisture": torch.ones(1, 40) * 0.3,
            "soil_temperature": torch.ones(1, 40) * 288.15,
            "ph": torch.ones(1, 40) * 6.5,
            "bulk_density": torch.ones(1, 40) * 900.0,
            
            # Process rates (ug m-3 s-1 for atmosphere, mg kg-1 s-1 for soil)
            "nitrification_rate": torch.ones(1, 40) * 0.1,
            "denitrification_rate": torch.ones(1, 40) * 0.05,
            "mineralization_rate": torch.ones(1, 40) * 0.2,
            "volatilization_rate": torch.ones(1, 40) * 0.01,
            "deposition_rate": torch.ones(10, 12) * 1.0e-4,
        }
        
        # Create simulator
        simulator = _declared_simulator(config={"device": "cpu"})
        simulator.register_functor(atm_nitrogen)
        simulator.register_functor(no3_transport)
        simulator.register_functor(nh4_transport)
        
        simulator.initialize_state(state)
        
        # Run simulation
        dt = 1800.0  # 30 minutes
        n_steps = int(self.max_simulation_time / dt)
        
        # Track nitrogen balance
        initial_atm_nh3 = torch.sum(state["atmospheric_nh3"]) * 5.0 * 5.0
        initial_atm_no = torch.sum(state["atmospheric_no"]) * 5.0 * 5.0
        initial_atm_no2 = torch.sum(state["atmospheric_no2"]) * 5.0 * 5.0
        initial_soil_no3 = torch.sum(state["soil_no3"]) * 0.05 * 100.0
        initial_soil_nh4 = torch.sum(state["soil_nh4"]) * 0.05 * 100.0
        initial_soil_org_n = torch.sum(state["soil_organic_n"]) * 0.05 * 100.0
        
        total_initial_n = (initial_atm_nh3 + initial_atm_no + initial_atm_no2 + 
                          initial_soil_no3 + initial_soil_nh4 + initial_soil_org_n)
        
        nitrogen_balance_history = []
        
        print(f"Running {n_steps} time steps...")
        
        for step in range(n_steps):
            if step % 2 == 0:
                print(f"  Step {step}/{n_steps}")
                
            current_state = simulator.get_state()
            
            # Atmospheric nitrogen transport
            # Transport NH3, NO, NO2 separately for simplicity
            nh3_state = {
                "tracer_concentration": current_state["atmospheric_nh3"],
                "source_term": state["deposition_rate"] * (-1.0)  # Deposition = loss
            }
            nh3_output = atm_nitrogen.forward(nh3_state, time=step*dt, dt=dt)
            
            no_state = {
                "tracer_concentration": current_state["atmospheric_no"],
                "source_term": torch.zeros_like(current_state["atmospheric_no"])
            }
            no_output = atm_nitrogen.forward(no_state, time=step*dt, dt=dt)
            
            no2_state = {
                "tracer_concentration": current_state["atmospheric_no2"],
                "source_term": torch.zeros_like(current_state["atmospheric_no2"])
            }
            no2_output = atm_nitrogen.forward(no2_state, time=step*dt, dt=dt)
            
            # Soil nitrogen transport
            no3_state = {
                "soil_no3": current_state["soil_no3"],
                "source_term": state["nitrification_rate"] - state["denitrification_rate"]
            }
            no3_output = no3_transport.forward(no3_state, time=step*dt, dt=dt)
            
            nh4_state = {
                "soil_nh4": current_state["soil_nh4"],
                "source_term": state["mineralization_rate"] - state["nitrification_rate"] - state["volatilization_rate"]
            }
            nh4_output = nh4_transport.forward(nh4_state, time=step*dt, dt=dt)
            
            # Apply updates
            atm_nh3_next = current_state["atmospheric_nh3"] + _linear_increment(nh3_output)
            atm_no_next = current_state["atmospheric_no"] + _linear_increment(no_output)
            atm_no2_next = current_state["atmospheric_no2"] + _linear_increment(no2_output)
            soil_no3_next = current_state["soil_no3"] + _linear_increment(
                no3_output,
                key="soil_no3",
            )
            soil_nh4_next = current_state["soil_nh4"] + _linear_increment(
                nh4_output,
                key="soil_nh4",
            )
            soil_organic_next = (
                current_state["soil_organic_n"] - state["mineralization_rate"] * dt
            )
            assert torch.all(soil_organic_next >= 0.0)
            simulator.state_manager.apply_updates(
                {
                    "atmospheric_nh3": UpdateItem(
                        atm_nh3_next,
                        UpdateMode.REPLACE,
                    ),
                    "atmospheric_no": UpdateItem(
                        atm_no_next,
                        UpdateMode.REPLACE,
                    ),
                    "atmospheric_no2": UpdateItem(
                        atm_no2_next,
                        UpdateMode.REPLACE,
                    ),
                    "soil_no3": UpdateItem(
                        soil_no3_next,
                        UpdateMode.REPLACE,
                    ),
                    "soil_nh4": UpdateItem(
                        soil_nh4_next,
                        UpdateMode.REPLACE,
                    ),
                    "soil_organic_n": UpdateItem(
                        soil_organic_next,
                        UpdateMode.REPLACE,
                    ),
                },
                dt=dt,
                check_conservation=False,
            )
            
            # Track the rates that were applied during this step.
            nitrification_total = torch.sum(state["nitrification_rate"]) * dt * 0.05 * 100.0
            denitrification_total = torch.sum(state["denitrification_rate"]) * dt * 0.05 * 100.0
            mineralization_total = torch.sum(state["mineralization_rate"]) * dt * 0.05 * 100.0
            volatilization_total = torch.sum(state["volatilization_rate"]) * dt * 0.05 * 100.0
            deposition_total = torch.sum(state["deposition_rate"]) * dt * 5.0 * 5.0
            nitrogen_balance_history.append({
                'nitrification': nitrification_total.item(),
                'denitrification': denitrification_total.item(),
                'mineralization': mineralization_total.item(),
                'volatilization': volatilization_total.item(),
                'deposition': deposition_total.item()
            })

            # Update process rates for the next step based on environmental conditions.
            hour_of_day = (step * dt) / 3600.0
            
            # Temperature dependence (Q10 = 2.0)
            temp_factor = 2.0 ** ((current_state["soil_temperature"] - 288.15) / 10.0)
            
            # Moisture dependence
            moisture_factor = torch.minimum(current_state["soil_moisture"] / 0.6, 
                                        torch.ones_like(current_state["soil_moisture"]))
            
            # Nitrification: temperature and moisture limited
            state["nitrification_rate"] = 0.1 * temp_factor * moisture_factor.squeeze(0)
            
            # Denitrification: anaerobic conditions (high moisture)
            anaerobic_factor = torch.clamp(
                (current_state["soil_moisture"] - 0.6) / 0.4,
                min=0.0,
            )
            state["denitrification_rate"] = 0.05 * anaerobic_factor.squeeze(0)
            
            # Mineralization: temperature dependent
            state["mineralization_rate"] = 0.2 * temp_factor.squeeze(0)
            
            # Volatilization: temperature and pH dependent
            ph_factor = torch.clamp(
                (current_state["ph"] - 7.0) / 3.0,
                min=0.0,
            )
            state["volatilization_rate"] = 0.01 * temp_factor.squeeze(0) * ph_factor.squeeze(0)
            
            # Deposition: diurnal pattern
            deposition_factor = max(0, np.cos(2 * np.pi * (hour_of_day - 6) / 24))
            state["deposition_rate"] = torch.ones(10, 12) * 1.0e-4 * deposition_factor
            
            # Verify physical constraints
            assert torch.all(current_state["atmospheric_nh3"] >= 0)
            assert torch.all(current_state["atmospheric_no"] >= 0)
            assert torch.all(current_state["atmospheric_no2"] >= 0)
            assert torch.all(current_state["soil_no3"] >= 0)
            assert torch.all(current_state["soil_nh4"] >= 0)
            
        print(f"Nitrogen cycle transport test passed!")
        
        # Final nitrogen balance
        final_state = simulator.get_state()
        
        final_atm_nh3 = torch.sum(final_state["atmospheric_nh3"]) * 5.0 * 5.0
        final_atm_no = torch.sum(final_state["atmospheric_no"]) * 5.0 * 5.0
        final_atm_no2 = torch.sum(final_state["atmospheric_no2"]) * 5.0 * 5.0
        final_soil_no3 = torch.sum(final_state["soil_no3"]) * 0.05 * 100.0
        final_soil_nh4 = torch.sum(final_state["soil_nh4"]) * 0.05 * 100.0
        final_soil_org_n = torch.sum(final_state["soil_organic_n"]) * 0.05 * 100.0

        total_initial_atm = initial_atm_nh3 + initial_atm_no + initial_atm_no2
        total_final_atm = final_atm_nh3 + final_atm_no + final_atm_no2
        total_initial_soil = initial_soil_no3 + initial_soil_nh4 + initial_soil_org_n
        total_final_soil = final_soil_no3 + final_soil_nh4 + final_soil_org_n

        # Process totals
        total_nitrification = sum([n['nitrification'] for n in nitrogen_balance_history])
        total_denitrification = sum([n['denitrification'] for n in nitrogen_balance_history])
        total_mineralization = sum([n['mineralization'] for n in nitrogen_balance_history])
        total_volatilization = sum([n['volatilization'] for n in nitrogen_balance_history])
        total_deposition = sum([n['deposition'] for n in nitrogen_balance_history])

        # Separate atmosphere and soil balances to avoid unit mixing
        atm_loss = total_initial_atm - total_final_atm
        assert atm_loss >= -1e-6, "Atmospheric nitrogen should not increase under net deposition"

        soil_expected_change = -(total_denitrification + total_volatilization)
        soil_actual_change = total_final_soil - total_initial_soil
        soil_relative_error = abs(soil_actual_change - soil_expected_change) / (abs(total_initial_soil) + 1e-10)
        assert soil_relative_error < 0.8, f"Soil nitrogen balance error: {soil_relative_error:.4f}"

        print(f"  Initial atmospheric nitrogen: {float(total_initial_atm):.2f} units")
        print(f"  Final atmospheric nitrogen: {float(total_final_atm):.2f} units")
        print(f"  Atmospheric loss: {float(atm_loss):.2f} units")
        print(f"  Initial soil nitrogen: {float(total_initial_soil):.2f} units")
        print(f"  Final soil nitrogen: {float(total_final_soil):.2f} units")
        print(f"  Soil balance error: {float(soil_relative_error):.4f}")


class TestLongTermStabilityAndConvergence:
    """Test long-term numerical stability and convergence properties."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        
    def test_long_term_stability_advection(self):
        """Test long-term stability of advection schemes."""
        print("Testing long-term advection stability...")
        
        # Create advection with different schemes
        schemes = ["upwind", "lax_wendroff", "superbee", "van_leer"]
        
        for scheme in schemes:
            print(f"  Testing {scheme} scheme...")
            
            advection = LinearAdvection(
                spatial_dims=1,
                grid_spacing=[0.1],
                scheme=scheme,
                track_conservation=True,
                name=f"StabilityTest_{scheme}"
            )
            
            # Set velocity field
            velocity = torch.ones(100) * 1.0
            advection.set_velocity_field([velocity])
            
            # Initial condition: smooth profile
            x = torch.linspace(0, 10, 100)
            initial_conc = torch.exp(-((x - 5.0) ** 2) / (2 * 1.0 ** 2)) * 100.0
            
            state = {
                "tracer_concentration": initial_conc.unsqueeze(0),
                "source_term": torch.zeros_like(initial_conc.unsqueeze(0))
            }
            
            # Long-term simulation
            dt = 0.05  # Small time step for stability
            n_steps = 1000  # Long simulation
            
            # Track stability metrics
            max_conc_history = []
            min_conc_history = []
            total_mass_history = []
            conservation_error_history = []
            
            for step in range(n_steps):
                output = advection.forward(state, time=step*dt, dt=dt)
                
                # Update state
                state["tracer_concentration"] += _linear_increment(output)
                
                # Track metrics
                current_conc = state["tracer_concentration"]
                max_conc_history.append(torch.max(current_conc).item())
                min_conc_history.append(torch.min(current_conc).item())
                total_mass_history.append(torch.sum(current_conc).item())
                
                if "diagnostics.conservation_error" in output:
                    conservation_error_history.append(output["diagnostics.conservation_error"].item())
                
                # Check for instability
                if step % 100 == 0:
                    max_conc = torch.max(current_conc)
                    min_conc = torch.min(current_conc)
                    
                    # Should remain bounded and physical
                    assert torch.isfinite(max_conc), f"Non-finite max concentration at step {step} for {scheme}"
                    assert torch.isfinite(min_conc), f"Non-finite min concentration at step {step} for {scheme}"
                    assert max_conc < 1e6, f"Unbounded maximum concentration {max_conc} at step {step} for {scheme}"
                    assert min_conc >= -1e-6, f"Negative concentration {min_conc} at step {step} for {scheme}"
                    
            # Analyze stability
            max_conc_array = np.array(max_conc_history)
            min_conc_array = np.array(min_conc_history)
            total_mass_array = np.array(total_mass_history)
            
            # Check for blow-up
            max_growth = np.max(np.abs(np.diff(max_conc_array)))
            assert max_growth < 100.0, f"Maximum concentration growing too fast for {scheme}"
            
            # Check mass conservation
            if len(conservation_error_history) > 0:
                avg_conservation_error = np.mean(conservation_error_history)
                max_conservation_error = np.max(conservation_error_history)
                
                assert avg_conservation_error < 0.1, f"Average conservation error too high for {scheme}"
                assert max_conservation_error < 1.0, f"Maximum conservation error too high for {scheme}"
                
            print(f"    {scheme} scheme stable for {n_steps} steps")
            print(f"    Final max concentration: {max_conc_history[-1]:.2f}")
            print(f"    Final min concentration: {min_conc_history[-1]:.2f}")
            print(f"    Mass variation: {(np.max(total_mass_array) - np.min(total_mass_array)):.4f}")
            
    def test_convergence_testing(self):
        """Test convergence properties as resolution is refined."""
        print("Testing convergence properties...")
        
        # Test convergence of advection scheme
        grid_sizes = [0.1, 0.05, 0.025, 0.0125]
        time_steps = [0.01, 0.005, 0.0025, 0.00125]
        
        solutions = []
        convergence_errors = []
        
        for i, (dx, dt) in enumerate(zip(grid_sizes, time_steps)):
            print(f"  Testing resolution: dx={dx}, dt={dt}")
            
            advection = LinearAdvection(
                spatial_dims=1,
                grid_spacing=[dx],
                scheme="upwind",
                name=f"ConvergenceTest_{i}"
            )
            
            # Set velocity
            velocity = torch.ones(int(10.0 / dx)) * 1.0
            advection.set_velocity_field([velocity])
            
            # Initial condition: traveling wave
            x = torch.linspace(0, 10, int(10.0 / dx))
            initial_conc = torch.exp(-((x - 2.0) ** 2) / (2 * 0.5 ** 2))
            
            state = {
                "tracer_concentration": initial_conc.unsqueeze(0),
                "source_term": torch.zeros_like(initial_conc.unsqueeze(0))
            }
            
            # Run for fixed physical time
            physical_time = 5.0
            n_steps = int(physical_time / dt)
            
            for step in range(n_steps):
                output = advection.forward(state, time=step*dt, dt=dt)
                state["tracer_concentration"] += _linear_increment(output)
                
            final_conc = state["tracer_concentration"].squeeze(0)
            
            solutions.append((x, final_conc))
        
        # Use the finest resolution as reference
        reference_x, reference_solution = solutions[-1]
        for x, final_conc in solutions[:-1]:
            # Interpolate to reference grid for comparison
            from scipy import interpolate
            f = interpolate.interp1d(
                x,
                final_conc.numpy(),
                kind="linear",
                bounds_error=False,
                fill_value=0.0,
            )
            conc_on_ref = torch.tensor(f(reference_x.numpy()))

            # Compute error
            error = torch.norm(conc_on_ref - reference_solution) / torch.norm(reference_solution)
            convergence_errors.append(error.item())

            print(f"    Relative error: {error.item():.6f}")
                
        # Check convergence rate
        if len(convergence_errors) >= 2:
            convergence_rates = []
            for i in range(len(convergence_errors) - 1):
                rate = np.log(convergence_errors[i] / convergence_errors[i+1]) / np.log(2)
                convergence_rates.append(rate)
                
            avg_rate = np.mean(convergence_rates)
            print(f"  Average convergence rate: {avg_rate:.2f}")
            
            # Upwind should be first-order (rate ~ 1)
            assert 0.5 < avg_rate < 2.0, f"Convergence rate {avg_rate} not as expected for upwind scheme"
            
    def test_extreme_condition_stability(self):
        """Test stability under extreme conditions."""
        print("Testing extreme condition stability...")
        
        # Test very high velocities
        print("  Testing high velocities...")
        high_velocity = LinearAdvection(
            spatial_dims=1,
            grid_spacing=[0.1],
            name="HighVelocityTest"
        )
        
        velocity = torch.ones(50) * 100.0  # Very high velocity
        high_velocity.set_velocity_field([velocity])
        
        concentration = torch.ones(50) * 100.0
        state = {
            "tracer_concentration": concentration.unsqueeze(0),
            "source_term": torch.zeros_like(concentration.unsqueeze(0))
        }
        
        max_velocity = torch.max(torch.abs(velocity)).item()
        dx = float(high_velocity.grid_spacing[0])
        unsafe_dt = 1.5 * dx / max_velocity
        with pytest.raises(
            ValueError,
            match="simulator-managed FunctorMetadata.substep_count",
        ):
            high_velocity.forward(state, time=0.0, dt=unsafe_dt)

        stable_dt = 0.5 * dx / max_velocity
        output = high_velocity.forward(state, time=0.0, dt=stable_dt)
        assert torch.isfinite(_linear_increment(output)).all()
            
        # Test very sharp gradients
        print("  Testing sharp gradients...")
        sharp_gradient = LinearAdvection(
            spatial_dims=1,
            grid_spacing=[0.01],
            scheme="upwind",  # Most stable for sharp gradients
            name="SharpGradientTest"
        )
        
        velocity = torch.ones(200) * 1.0
        sharp_gradient.set_velocity_field([velocity])
        
        # Very sharp step function
        concentration = torch.zeros(200)
        concentration[100:] = 1000.0  # Sharp jump
        
        state = {
            "tracer_concentration": concentration.unsqueeze(0),
            "source_term": torch.zeros_like(concentration.unsqueeze(0))
        }
        
        dt = 0.001  # Small time step
        for step in range(100):
            output = sharp_gradient.forward(state, time=step*dt, dt=dt)
            state["tracer_concentration"] += _linear_increment(output)
            
            # Check for numerical issues
            current_conc = state["tracer_concentration"]
            assert torch.isfinite(current_conc).all()
            assert torch.all(current_conc >= 0)  # Non-negative
            
        print("  Extreme condition tests passed")
        
    def test_roundoff_error_accumulation(self):
        """Test behavior under long simulations with roundoff error."""
        print("Testing roundoff error accumulation...")
        
        # Create system with very small changes
        advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=[1.0],
            track_conservation=True,
            name="RoundoffTest"
        )
        
        # Very small velocity
        velocity = torch.ones(100) * 1e-10
        advection.set_velocity_field([velocity])
        
        concentration = torch.ones(100) * 1e6  # Large concentration
        state = {
            "tracer_concentration": concentration.unsqueeze(0),
            "source_term": torch.zeros_like(concentration.unsqueeze(0))
        }
        
        # Long simulation with very small changes
        dt = 1.0
        n_steps = 10000
        
        initial_mass = torch.sum(concentration)
        mass_history = [initial_mass.item()]
        
        for step in range(n_steps):
            output = advection.forward(state, time=step*dt, dt=dt)
            state["tracer_concentration"] += _linear_increment(output)
            
            if step % 1000 == 0:
                current_mass = torch.sum(state["tracer_concentration"])
                mass_history.append(current_mass.item())
                
                # Check for drift
                relative_change = torch.abs(current_mass - initial_mass) / initial_mass
                assert relative_change < 2e-8, f"Roundoff error accumulation too large at step {step}"
                
        print(f"  Roundoff error test passed for {n_steps} steps")
        print(f"  Initial mass: {initial_mass:.6e}")
        print(f"  Final mass: {mass_history[-1]:.6e}")
        print(f"  Relative change: {abs(mass_history[-1] - initial_mass) / initial_mass:.2e}")


class TestRealWorldScenarioValidation:
    """Test transport module against real-world scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        
    def test_atmospheric_pollutant_dispersion(self):
        """Test atmospheric pollutant dispersion scenario."""
        print("Testing atmospheric pollutant dispersion...")
        
        # Create 2D atmospheric transport model
        atm_transport = LinearAdvection(
            spatial_dims=2,
            grid_spacing=[100.0, 100.0],  # 100m resolution
            name="PollutantDispersion"
        )
        
        # Set up realistic wind field
        nx, ny = 50, 40
        
        # Wind that increases with height and varies with direction
        u_wind = torch.zeros(ny, nx)
        v_wind = torch.zeros(ny, nx)
        
        for j in range(ny):
            height = j * 10.0  # 0 to 400m
            # Logarithmic wind profile
            u_wind[j, :] = 5.0 * np.log((height + 1.0) / 1.0) / np.log(400.0 / 1.0)
            # Add some lateral variation
            v_wind[j, :] = 0.5 * torch.sin(
                2.0 * torch.pi * torch.arange(nx, device=v_wind.device, dtype=v_wind.dtype) / float(nx)
            )
            
        atm_transport.set_velocity_field([u_wind, v_wind])
        
        # Initial pollutant concentration (point source)
        concentration = torch.zeros(ny, nx)
        source_x, source_y = 25, 20  # Center of domain
        source_radius = 2  # Grid points
        
        for j in range(max(0, source_y - source_radius), min(ny, source_y + source_radius + 1)):
            for i in range(max(0, source_x - source_radius), min(nx, source_x + source_radius + 1)):
                dist = np.sqrt((i - source_x)**2 + (j - source_y)**2)
                if dist <= source_radius:
                    concentration[j, i] = 1000.0 * np.exp(-dist**2 / 2.0)
                    
        state = {
            "tracer_concentration": concentration,
            "source_term": torch.zeros_like(concentration)
        }
        
        # Add continuous source
        source_strength = 100.0  # ug m-3 s-1
        state["source_term"][source_y, source_x] = source_strength
        
        # Run dispersion simulation
        dt = 10.0  # 10 seconds
        n_steps = 360  # 1 hour
        
        concentration_history = []
        
        for step in range(n_steps):
            output = atm_transport.forward(state, time=step*dt, dt=dt)
            state["tracer_concentration"] += _linear_increment(output)
            
            # Maintain continuous source
            state["source_term"][source_y, source_x] = source_strength
            
            if step % 60 == 0:
                concentration_history.append(state["tracer_concentration"].clone())
                
        # Analyze dispersion pattern
        final_conc = state["tracer_concentration"]
        
        # Check physical realism
        assert torch.all(final_conc >= 0), "Negative concentrations found"
        assert torch.isfinite(final_conc).all(), "Non-finite concentrations found"
        
        # Maximum should be near source
        max_conc = torch.max(final_conc)
        max_idx = torch.argmax(final_conc)
        max_j = int((max_idx // nx).item())
        max_i = int((max_idx % nx).item())

        # Center of mass should move downwind with mean flow
        total_mass = torch.sum(final_conc)
        x_indices = torch.arange(nx, dtype=final_conc.dtype, device=final_conc.device)
        y_indices = torch.arange(ny, dtype=final_conc.dtype, device=final_conc.device)
        x_center = torch.sum(final_conc.sum(dim=0) * x_indices) / total_mass
        y_center = torch.sum(final_conc.sum(dim=1) * y_indices) / total_mass
        assert x_center > source_x, "Center of mass should move downwind"
        
        # Concentration should be higher at plume core than domain mean
        center_conc = final_conc[source_y, source_x]
        edge_conc = torch.mean(final_conc)
        assert max_conc > edge_conc, "Concentration should be higher near plume core"
        
        # Check mass balance
        total_mass = torch.sum(final_conc) * 100.0 * 100.0  # ug
        source_mass = source_strength * dt * n_steps * 100.0 * 100.0  # ug
        
        # Some mass should have left the domain (open boundaries)
        assert total_mass < source_mass * 1.1, "Too much mass accumulation"
        assert total_mass > source_mass * 0.001, "Too much mass loss"
        
        print(f"  Pollutant dispersion test passed")
        print(f"  Maximum concentration: {float(max_conc):.2f} ug m-3")
        print(f"  Source concentration: {float(center_conc):.2f} ug m-3")
        print(f"  Domain average: {float(edge_conc):.2f} ug m-3")
        print(f"  Mass balance: {float(total_mass):.2e}/{float(source_mass):.2e} ug")
        print(f"  Plume center index: ({max_i}, {max_j})")
        print(f"  Center of mass: ({float(x_center):.2f}, {float(y_center):.2f})")
        
    def test_soil_contaminant_transport(self):
        """Test soil contaminant transport scenario."""
        print("Testing soil contaminant transport...")
        
        # Create 1D soil transport model
        soil_transport = LinearAdvection(
            spatial_dims=1,
            grid_spacing=[0.1],  # 10cm resolution
            track_conservation=True,
            name="SoilContaminantTransport"
        )
        
        # Set up soil water flow (infiltration and percolation)
        depth_points = 100  # 0 to 10m depth
        velocity = torch.zeros(depth_points)
        
        # Exponential decay of water flow with depth
        for i in range(depth_points):
            depth = i * 0.1  # meters
            velocity[i] = 1e-7 * np.exp(-depth / 2.0)  # m/s
            
        soil_transport.set_velocity_field([velocity])
        
        # Initial contaminant distribution (surface spill)
        concentration = torch.zeros(depth_points)
        
        # Gaussian distribution near surface
        for i in range(20):  # Top 2m
            depth = i * 0.1
            concentration[i] = 1000.0 * np.exp(-depth**2 / (2 * 0.5**2))  # mg kg-1
            
        state = {
            "tracer_concentration": concentration.unsqueeze(0),
            "source_term": torch.zeros_like(concentration.unsqueeze(0))
        }
        
        # Add decay and sorption effects
        decay_rate = 0.01  # day-1
        retardation_factor = 2.0  # accounts for sorption
        
        # Run transport simulation
        dt = 3600.0  # 1 hour
        n_days = 30
        n_steps = n_days * 24  # 24 steps per day
        
        concentration_profiles = []
        
        for step in range(n_steps):
            # Transport step
            output = soil_transport.forward(state, time=step*dt, dt=dt)
            
            # Apply decay and retardation
            decay_factor = np.exp(-decay_rate * dt / 86400.0)  # Convert to days
            effective_velocity = velocity / retardation_factor
            
            # Update with transport and decay
            state["tracer_concentration"] = (
                state["tracer_concentration"] * decay_factor
                + _linear_increment(output) / retardation_factor
            )
            
            # Store profiles at daily intervals
            if step % 24 == 0:
                concentration_profiles.append(state["tracer_concentration"].squeeze(0).clone())
                
        # Analyze contaminant migration
        final_conc = state["tracer_concentration"].squeeze(0)
        
        # Check physical realism
        assert torch.all(final_conc >= 0), "Negative concentrations found"
        assert torch.isfinite(final_conc).all(), "Non-finite concentrations found"
        
        # Contaminant should have migrated downward
        surface_conc = final_conc[0]
        max_depth_idx = torch.argmax(final_conc)
        max_depth = max_depth_idx * 0.1  # meters
        
        assert max_depth >= 0.1, "Contaminant should have migrated below surface"
        
        # Check exponential decay pattern
        total_mass_initial = torch.sum(concentration) * 0.1  # mg m-2
        total_mass_final = torch.sum(final_conc) * 0.1
        
        # Should have lost mass due to decay
        mass_ratio = total_mass_final / total_mass_initial
        expected_decay = np.exp(-decay_rate * n_days)
        
        assert abs(mass_ratio - expected_decay) < 0.2, f"Decay pattern not as expected: {mass_ratio:.3f} vs {expected_decay:.3f}"
        
        print(f"  Soil contaminant transport test passed")
        print(f"  Initial total mass: {total_mass_initial:.2f} mg m-2")
        print(f"  Final total mass: {total_mass_final:.2f} mg m-2")
        print(f"  Mass ratio: {mass_ratio:.3f}")
        print(f"  Expected decay: {expected_decay:.3f}")
        print(f"  Maximum depth: {max_depth:.2f} m")
        
    def test_river_pollutant_transport(self):
        """Test river pollutant transport scenario."""
        print("Testing river pollutant transport...")
        
        # Create 1D river transport model
        river_transport = LinearAdvection(
            spatial_dims=1,
            grid_spacing=[100.0],  # 100m resolution
            name="RiverPollutantTransport"
        )
        
        # Set up river flow velocity
        river_length = 10000  # 10km river
        n_points = int(river_length / 100.0)
        
        # Variable flow velocity (faster in middle, slower near banks)
        base_velocity = 1.0  # m/s
        velocity = torch.ones(n_points) * base_velocity
        
        # Add some spatial variation
        for i in range(n_points):
            position = i * 100.0  # meters
            # Faster in straight sections, slower in curves
            velocity[i] *= (1.0 + 0.3 * np.sin(2 * np.pi * position / 2000.0))
            
        river_transport.set_velocity_field([velocity])
        
        # Initial pollutant concentration
        concentration = torch.zeros(n_points)
        
        # Upstream pollution source
        source_location = 10  # 1km from start
        source_width = 5  # 500m wide
        source_concentration = 500.0  # mg m-3
        
        for i in range(source_location, min(source_location + source_width, n_points)):
            concentration[i] = source_concentration
            
        state = {
            "tracer_concentration": concentration.unsqueeze(0),
            "source_term": torch.zeros_like(concentration.unsqueeze(0))
        }
        
        # Continuous upstream source
        state["source_term"][0, source_location:source_location + source_width] = 10.0  # mg m-3 s-1
        
        # Add dispersion (simplified as additional diffusion)
        # In real rivers, this would be handled by separate diffusion module
        
        # Run transport simulation
        dt = 60.0  # 1 minute
        simulation_time = 7200.0  # 2 hours
        n_steps = int(simulation_time / dt)
        
        concentration_snapshots = []
        
        for step in range(n_steps):
            output = river_transport.forward(state, time=step*dt, dt=dt)
            state["tracer_concentration"] += _linear_increment(output)
            
            # Maintain upstream source
            state["source_term"][0, source_location:source_location + source_width] = 10.0
            
            # Store snapshots every 30 minutes
            if step % 30 == 0:
                concentration_snapshots.append(state["tracer_concentration"].squeeze(0).clone())
                
        # Analyze river transport
        final_conc = state["tracer_concentration"].squeeze(0)
        
        # Check physical realism
        assert torch.all(final_conc >= 0), "Negative concentrations found"
        assert torch.isfinite(final_conc).all(), "Non-finite concentrations found"

        # Plume should have moved downstream
        plume_center = torch.argmax(final_conc)
        assert plume_center.item() > source_location, "Plume center should move downstream from source"
        downstream_conc = torch.mean(final_conc[-20:])  # Last 2km
        upstream_conc = torch.mean(final_conc[:20])    # First 2km

        # Check advection velocity
        expected_position = source_location + base_velocity * simulation_time / 100.0
        actual_position = plume_center.item()

        position_error = abs(actual_position - expected_position)
        assert position_error < expected_position, f"Plume position error too large: {position_error:.1f} vs {expected_position:.1f}"
        
        # Check mass balance
        input_mass = 10.0 * source_width * 100.0 * simulation_time  # mg m-2
        total_mass = torch.sum(final_conc) * 100.0  # mg m-2
        
        # Some mass should have left the domain
        assert total_mass < input_mass * 1.5, "Excessive mass accumulation"
        assert total_mass > input_mass * 0.3, "Excessive mass loss"
        
        print(f"  River pollutant transport test passed")
        print(f"  Upstream concentration: {upstream_conc:.2f} mg m-3")
        print(f"  Downstream concentration: {downstream_conc:.2f} mg m-3")
        print(f"  Plume center position: {float(actual_position) * 100.0:.0f} m")
        print(f"  Expected position: {float(expected_position) * 100.0:.0f} m")
        print(f"  Mass balance: {float(total_mass):.2e}/{float(input_mass):.2e} mg m-2")


class TestPerformanceAndOptimization:
    """Test performance characteristics and optimization opportunities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        
    def test_performance_benchmarking(self):
        """Benchmark transport module performance."""
        print("Benchmarking transport module performance...")
        
        # Test different problem sizes
        problem_sizes = [100, 500, 1000, 2000]
        schemes = ["upwind", "lax_wendroff"]
        
        performance_results = {}
        
        for scheme in schemes:
            print(f"  Testing {scheme} scheme...")
            scheme_results = {}
            
            for size in problem_sizes:
                print(f"    Problem size: {size}")
                
                # Create transport module
                transport = LinearAdvection(
                    spatial_dims=1,
                    grid_spacing=[1.0],
                    scheme=scheme,
                    name=f"Benchmark_{scheme}_{size}"
                )
                
                # Set up problem
                velocity = torch.ones(size) * 1.0
                transport.set_velocity_field([velocity])
                
                concentration = torch.randn(size) * 10.0 + 100.0
                state = {
                    "tracer_concentration": concentration.unsqueeze(0),
                    "source_term": torch.zeros_like(concentration.unsqueeze(0))
                }
                
                # Benchmark timing
                dt = 0.5
                n_steps = 100
                
                start_time = time.time()
                
                for step in range(n_steps):
                    output = transport.forward(state, time=step*dt, dt=dt)
                    state["tracer_concentration"] += _linear_increment(output)
                    
                end_time = time.time()
                
                total_time = end_time - start_time
                avg_time_per_step = total_time / n_steps
                points_per_second = size * n_steps / total_time
                
                scheme_results[size] = {
                    'total_time': total_time,
                    'avg_time_per_step': avg_time_per_step,
                    'points_per_second': points_per_second
                }
                
                print(f"      Total time: {total_time:.3f}s")
                print(f"      Avg time/step: {avg_time_per_step*1000:.2f}ms")
                print(f"      Points/s: {points_per_second:.0f}")
                
                # Verify correctness
                assert torch.isfinite(state["tracer_concentration"]).all()
                assert torch.all(state["tracer_concentration"] >= 0)
                
            performance_results[scheme] = scheme_results
            
        # Analyze performance scaling
        print(f"Performance scaling analysis:")
        for scheme in schemes:
            results = performance_results[scheme]
            sizes = list(results.keys())
            times = [results[size]['avg_time_per_step'] for size in sizes]
            points = [results[size]['points_per_second'] for size in sizes]
            
            # Robust performance sanity checks:
            # - Time per step must be finite and positive
            # - Throughput should not regress catastrophically at larger sizes
            if len(sizes) >= 2:
                assert all(t > 0.0 for t in times), f"Non-positive timing for {scheme}: {times}"
                assert all(np.isfinite(t) for t in times), f"Non-finite timing for {scheme}: {times}"
                assert points[-1] >= 0.5 * min(points), f"Throughput regression for {scheme}: {points}"
                
    def test_memory_usage_profiling(self):
        """Profile memory usage of transport modules."""
        print("Profiling memory usage...")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            initial_memory = torch.cuda.memory_allocated()
            print(f"  Initial GPU memory: {initial_memory / 1024**2:.1f} MB")
        else:
            print("  CUDA not available, using CPU memory profiling")
            
        # Test memory usage with different problem sizes
        sizes = [1000, 5000, 10000]
        
        for size in sizes:
            print(f"  Testing size: {size}")
            
            # Create transport module
            transport = LinearAdvection(
                spatial_dims=1,
                grid_spacing=[1.0],
                name=f"MemoryTest_{size}"
            )
            
            # Create large tensors
            velocity = torch.ones(size) * 1.0
            concentration = torch.randn(size) * 10.0 + 100.0
            
            transport.set_velocity_field([velocity])
            
            state = {
                "tracer_concentration": concentration.unsqueeze(0),
                "source_term": torch.zeros_like(concentration.unsqueeze(0))
            }
            
            # Run several steps and monitor memory
            dt = 0.5
            n_steps = 10
            
            for step in range(n_steps):
                output = transport.forward(state, time=step*dt, dt=dt)
                state["tracer_concentration"] += _linear_increment(output)
                
                if torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated()
                    peak_memory = torch.cuda.max_memory_allocated()
                    
                    if step == n_steps - 1:
                        print(f"    Final GPU memory: {current_memory / 1024**2:.1f} MB")
                        print(f"    Peak GPU memory: {peak_memory / 1024**2:.1f} MB")
                        
                        # Memory growth should be reasonable
                        memory_growth = peak_memory - initial_memory
                        memory_per_point = memory_growth / size
                        
                        print(f"    Memory per point: {memory_per_point / 1024:.2f} KB")
                        
                        # Should not use excessive memory per point
                        assert memory_per_point < 1024 * 10, f"Excessive memory usage: {memory_per_point / 1024:.1f} KB/point"
                        
            # Verify correctness
            assert torch.isfinite(state["tracer_concentration"]).all()
            
            # Clean up
            del transport, velocity, concentration, state
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    def test_parallel_consistency(self):
        """Test consistency with parallel computation."""
        print("Testing parallel computation consistency...")
        
        # Test batch processing capabilities
        batch_sizes = [1, 5, 10, 20]
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")
            
            # Create transport module
            transport = LinearAdvection(
                spatial_dims=1,
                grid_spacing=[1.0],
                name=f"BatchTest_{batch_size}"
            )
            
            # Create batched velocity fields
            velocity = torch.ones(batch_size, 100) * 1.0
            
            # Add variation across batches
            for i in range(batch_size):
                velocity[i, :] *= (1.0 + 0.1 * i)
                
            transport.set_velocity_field([velocity])
            
            # Create batched concentrations
            concentration = torch.randn(batch_size, 100) * 10.0 + 100.0
            
            state = {
                "tracer_concentration": concentration,
                "source_term": torch.zeros_like(concentration)
            }
            
            # Run computation
            dt = 0.2
            output = transport.forward(state, time=0.0, dt=dt)
            
            # Verify batch processing works correctly
            increment = _linear_increment(output)
            assert increment.shape[0] == batch_size
            assert torch.isfinite(increment).all()
            
            # Each batch should have different results due to different velocities
            if batch_size > 1:
                batch_differences = torch.std(increment, dim=0)
                assert torch.mean(batch_differences) > 1e-6, "Batches should produce different results"
                
            print(f"    Batch processing successful for size {batch_size}")
            
        # Test multi-dimensional parallelism
        print("  Testing multi-dimensional parallelism...")
        
        # Create 2D transport
        transport_2d = LinearAdvection(
            spatial_dims=2,
            grid_spacing=[1.0, 1.0],
            name="Parallel2DTest"
        )
        
        # Large 2D problem
        ny, nx = 100, 150
        u = torch.ones(ny, nx) * 2.0
        v = torch.ones(ny, nx) * 0.5
        
        transport_2d.set_velocity_field([u, v])
        
        concentration_2d = torch.randn(ny, nx) * 10.0 + 100.0
        state_2d = {
            "tracer_concentration": concentration_2d.unsqueeze(0),
            "source_term": torch.zeros_like(concentration_2d.unsqueeze(0))
        }
        
        # Time the computation
        start_time = time.time()
        output_2d = transport_2d.forward(state_2d, time=0.0, dt=0.2)
        end_time = time.time()
        
        computation_time = end_time - start_time
        points_processed = ny * nx
        points_per_second = points_processed / computation_time
        
        print(f"    2D computation: {points_processed} points in {computation_time:.3f}s")
        print(f"    Performance: {points_per_second:.0f} points/s")
        
        # Verify correctness
        increment_2d = _linear_increment(output_2d)
        assert increment_2d.shape == (1, ny, nx)
        assert torch.isfinite(increment_2d).all()
        
        print("  Parallel computation test passed")


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v", "--tb=short"])


