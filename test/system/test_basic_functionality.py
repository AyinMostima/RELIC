#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic L3 System Functionality Test

This is a simplified version of the L3 system tests to verify
basic functionality before running the full test suite.

Authors: Yuchong Long
License: MIT
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import time

# Units of this test's own scenario variables, which no SCM contract or SSOT
# entry declares; they follow the units stated with the test data. The simulator
# infers no units from names (relic/core/simulator/state_init.py).
_TEST_STATE_UNITS = {
    'no2_concentration': 'mol mol-1',
}


def _declared_simulator(config=None, *args, **kwargs):
    """Construct RELICSimulator with this test's state_units declaration."""
    from relic.core.simulator import RELICSimulator

    declared = dict(config or {})
    declared.setdefault("state_units", _TEST_STATE_UNITS)
    return RELICSimulator(declared, *args, **kwargs)


def test_basic_imports():
    """Test that all key modules can be imported."""
    from relic.core.simulator import RELICSimulator
    from relic.core.state import StateManager
    from relic.parameters import get_global_manager

    assert RELICSimulator is not None
    assert StateManager is not None
    assert callable(get_global_manager)
    print("[OK] Core runtime imports succeeded")
    
    # Test some scientific modules
    scientific_modules = [
        ("relic.scientific.atmosphere.radiation", "RadiationFunctor"),
        ("relic.scientific.biosphere.photosynthesis.farquhar", "FarquharPhotosynthesis"),
        ("relic.scientific.hydrosphere.evapotranspiration.penman_monteith", "PenmanMonteithET"),
    ]
    
    for module_name, class_name in scientific_modules:
        module = __import__(module_name, fromlist=[class_name])
        assert getattr(module, class_name) is not None
        print(f"[OK] {class_name} imported successfully")

def test_basic_simulator_creation():
    """Test basic simulator creation and initialization."""
    
    from relic.core.simulator import RELICSimulator
    
    # Create basic configuration
    config = {
        'time_step': 3600.0,  # 1 hour
        'max_time': 7200.0,   # 2 hours
        'enable_bootstrap_modules': False,
        'strict_conservation_enabled': False,
        'conservation_checks': False,
    }
    
    # Create simulator
    simulator = _declared_simulator(config=config)
    assert simulator is not None
    print("[OK] Simulator created successfully")
    
    # Create basic initial state
    initial_state = {
        'air_temperature': torch.tensor([293.15]),  # 20 C
        'air_pressure': torch.tensor([101325.0]),   # 1 atm
        'relative_humidity': torch.tensor([0.7]),   # 70%
        'atmospheric_co2_ppm': torch.tensor([415.0]), # 415 ppm
        'leaf_area_index': torch.tensor([3.0]),     # LAI
        'soil_moisture': torch.tensor([0.3]),       # 30%
    }
    
    # Initialize state
    simulator.initialize_state(initial_state)
    print("[OK] Simulator state initialized successfully")
    
    # Check that state was set
    current_state = simulator.state_manager.get_current_state()
    assert 'air_temperature' in current_state
    expected_air_temperature = initial_state['air_temperature'].to(
        device=current_state['air_temperature'].device,
        dtype=current_state['air_temperature'].dtype,
    )
    assert torch.allclose(current_state['air_temperature'], expected_air_temperature)
    print("[OK] State verification passed")

def test_basic_timestep_execution():
    """Test basic timestep execution."""
    
    from relic.core.simulator import RELICSimulator
    
    config = {
        'time_step': 3600.0,
        'max_time': 7200.0,
        'enable_bootstrap_modules': False,
        'strict_conservation_enabled': False,
        'conservation_checks': False,
    }
    
    simulator = _declared_simulator(config=config)
    
    # More comprehensive initial state
    initial_state = {
        'air_temperature': torch.tensor([293.15]),
        'air_pressure': torch.tensor([101325.0]),
        'relative_humidity': torch.tensor([0.7]),
        'wind_speed': torch.tensor([3.0]),
        'atmospheric_co2_ppm': torch.tensor([415.0]),
        'o3_mixing_ratio': torch.tensor([40e-9]),
        'no2_concentration': torch.tensor([20e-9]),
        'leaf_area_index': torch.tensor([3.0]),
        'soil_moisture': torch.tensor([0.3]),
        'soil_temperature': torch.tensor([290.15]),
        'incoming_shortwave_potential': torch.tensor([600.0]),
        'ppfd': torch.tensor([1200.0]),
        # Initialize flux variables (REQUIRED by bucket_soil_moisture)
        'precipitation_flux': torch.tensor([0.0]),  # kg m-2 s-1
        'evapotranspiration_flux': torch.tensor([0.0]),  # kg m-2 s-1
        'gross_primary_productivity': torch.tensor([0.0]),
        'net_radiation': torch.tensor([0.0]),
    }
    
    simulator.initialize_state(initial_state)
    
    # Run one timestep
    start_time = time.time()
    updates = simulator.step(3600.0)
    execution_time = time.time() - start_time
    
    print(f"[OK] Timestep executed successfully in {execution_time:.3f}s")
    
    # Check that some updates were generated
    print(f"[OK] Updates generated: {len(updates)} items")
    
    # Check final state
    final_state = simulator.state_manager.get_current_state()
    assert 'air_temperature' in final_state
    assert torch.isfinite(final_state['air_temperature']).all()
    print("[OK] Final state verification passed")
    
    # Performance check
    assert execution_time < 10.0, f"Timestep too slow: {execution_time:.3f}s"
    print("[OK] Performance check passed")

def test_scientific_module_basic_execution():
    """Test basic execution of individual scientific modules."""
    
    # Test state for modules
    state = {
        'air_temperature': torch.tensor([293.15]),
        'air_pressure': torch.tensor([101325.0]),
        'relative_humidity': torch.tensor([0.7]),
        'wind_speed': torch.tensor([3.0]),
        'atmospheric_co2_ppm': torch.tensor([415.0]),
        'ppfd': torch.tensor([1200.0]),
        'vapor_pressure_deficit': torch.tensor([1000.0]),
        'incoming_shortwave_potential': torch.tensor([900.0]),
        'cos_zenith': torch.tensor([0.7]),
        'surface_temperature': torch.tensor([292.15]),
        'surface_albedo': torch.tensor([0.2]),
        'surface_emissivity': torch.tensor([0.97]),
        'cloud_fraction': torch.tensor([0.0]),
        'leaf_area_index': torch.tensor([3.0]),
        'vcmax_scaling': torch.tensor([1.0]),
        'water_stress_factor': torch.tensor([1.0]),
        'temperature_stress_factor': torch.tensor([1.0]),
        'ozone_stress_factor': torch.tensor([1.0]),
    }
    
    # Test radiation module
    from relic.core.state import UpdateItem
    from relic.scientific.atmosphere.radiation import RadiationFunctor
    radiation = RadiationFunctor()
    rad_fluxes = radiation(state, time=0.0, dt=3600.0)
    assert isinstance(rad_fluxes, dict)
    assert isinstance(rad_fluxes['net_radiation'], UpdateItem)
    assert torch.isfinite(rad_fluxes['net_radiation'].value).all()
    print("[OK] Radiation module executed successfully")
    
    # Test photosynthesis module
    from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
    photosynthesis = FarquharPhotosynthesis()
    photo_fluxes = photosynthesis(state, time=0.0, dt=3600.0)
    assert isinstance(photo_fluxes, dict)
    print("[OK] Photosynthesis module executed successfully")
    
    # Test ET module
    from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import PenmanMonteithET

    et_state = state.copy()
    et_state.update({
        'net_radiation': torch.tensor([400.0]),
        'atmospheric_pressure': torch.tensor([101325.0]),
        'saturated_vapor_pressure': torch.tensor([2339.0]),
        'actual_vapor_pressure': torch.tensor([1637.0]),
        'surface_resistance': torch.tensor([70.0]),
    })

    et = PenmanMonteithET(config={"canopy_resistance_source": "state"})
    et_fluxes = et(et_state, time=0.0, dt=3600.0)
    assert isinstance(et_fluxes, dict)
    print("[OK] Evapotranspiration module executed successfully")

def test_basic_multi_timestep():
    """Test multiple timestep execution for stability."""
    
    from relic.core.simulator import RELICSimulator
    
    config = {
        'time_step': 3600.0,
        'max_time': 10 * 3600.0,  # 10 hours
        'enable_bootstrap_modules': False,
        'strict_conservation_enabled': False,
        'conservation_checks': False,
    }
    
    simulator = _declared_simulator(config=config)
    
    initial_state = {
        'air_temperature': torch.tensor([293.15]),
        'air_pressure': torch.tensor([101325.0]),
        'relative_humidity': torch.tensor([0.7]),
        'wind_speed': torch.tensor([3.0]),
        'atmospheric_co2_ppm': torch.tensor([415.0]),
        'leaf_area_index': torch.tensor([3.0]),
        'soil_moisture': torch.tensor([0.3]),
        'incoming_shortwave_potential': torch.tensor([600.0]),
        'ppfd': torch.tensor([1200.0]),
        # REQUIRED flux variables
        'precipitation_flux': torch.tensor([0.0]),
        'evapotranspiration_flux': torch.tensor([0.0]),
    }
    
    simulator.initialize_state(initial_state)
    
    # Run 5 timesteps
    for i in range(5):
        start_time = time.time()
        updates = simulator.step(3600.0)
        execution_time = time.time() - start_time
        
        # Check stability
        current_state = simulator.state_manager.get_current_state()
        temp = current_state['air_temperature']
        
        assert torch.isfinite(temp).all(), f"Non-finite temperature at step {i}"
        assert temp > 200, f"Temperature too low at step {i}: {temp}"
        assert temp < 400, f"Temperature too high at step {i}: {temp}"
        assert execution_time < 10.0, f"Step {i} too slow: {execution_time:.3f}s"
    
    print("[OK] Multi-timestep stability test passed")

if __name__ == '__main__':
    print("Running Basic L3 System Functionality Tests")
    print("=" * 50)
    
    try:
        test_basic_imports()
        print()
        
        test_basic_simulator_creation() 
        print()
        
        test_basic_timestep_execution()
        print()
        
        test_scientific_module_basic_execution()
        print()
        
        test_basic_multi_timestep()
        print()
        
        print("[SUCCESS] All basic functionality tests PASSED!")
        print("\nThe L3 system appears to be working correctly.")
        print("You can now run the full L3 test suite with:")
        print("  python run_l3_tests.py")
        
    except Exception as e:
        print(f"\n[ERROR] Basic functionality test FAILED: {e}")
        print("\nPlease resolve the issues before running the full L3 test suite.")
        raise


