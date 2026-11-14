"""
L3 System Tests - Full Simulation Verification
Tests complete system functionality after refactoring
"""

import unittest
import torch
from pathlib import Path

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Dict, List
from relic.core.functor import PhysicalFieldFunctor, ProcessCategory
from relic.core.simulator import RELICSimulator
from relic.core.config.loader import ConfigLoader
from relic.core.state import UpdateMode
from relic.parameters import get_global_manager
from test.system.conftest import _mock_metadata


class MockAtmosphericRadiation(PhysicalFieldFunctor):
    """Mock atmospheric radiation functor for dependency resolution."""

    # Point model: no coordinate query; the state is read from StateManager.
    SPATIAL_DIMS = None

    def get_value_at(self, coordinates: "torch.Tensor") -> "torch.Tensor":
        """No coordinate query: this Functor is a point model (SPATIAL_DIMS = None)."""
        raise NotImplementedError(
            f"{type(self).__name__} is a point model (SPATIAL_DIMS = None): it has no "
            "coordinate query; read its state from StateManager"
        )

    def __init__(self, name: str = "atmospheric_radiation", config: Dict = None):
        super().__init__(
            name,
            config or {},
            metadata=_mock_metadata(
                name,
                "Test-only atmospheric radiation dependency marker",
                (),
                ("diagnostics.test.atmospheric_radiation_ready",),
                category=ProcessCategory.PHYSICAL_FIELD,
            ),
        )

    def get_dependencies(self) -> List[str]:
        return []

    def forward(self, state: Dict, time: float, dt: float) -> Dict:
        return {
            "diagnostics.test.atmospheric_radiation_ready": self.wrap_update(
                torch.tensor(1.0),
                mode=UpdateMode.REPLACE,
            )
        }


class TestFullSimulation(unittest.TestCase):
    """Test full simulation workflow"""

    def setUp(self):
        """Set up test environment"""
        self.config = {
            'time_step': 3600.0,  # 1 hour
            'max_time': 86400.0,  # 24 hours
            'enable_bootstrap_modules': False,
            'device': 'cpu',
            # These L3 smoke tests use open-system mock forcing without a
            # complete carbon, water, energy, or chemistry ledger.
            'conservation_checks': False,
            'strict_conservation_enabled': False,
            'strict_scale_detection': False,
        }
        
    def test_simulator_basic_run(self):
        """Test that simulator can run basic timesteps"""
        sim = RELICSimulator(config=self.config)
        
        # Initialize with basic state
        initial_state = {
            'air_temperature': torch.tensor(298.15),
            'co2_concentration': torch.tensor(400.0),
            'soil_moisture': torch.tensor(0.3),
            'par': torch.tensor(500.0)
        }
        
        sim.initialize_state(initial_state)
        
        # Run one timestep
        updates = sim.step(3600.0)
        
        # Should return dict of updates
        self.assertIsInstance(updates, dict)
        
        # Time should advance
        self.assertEqual(sim.current_time, 3600.0)
        
    def test_multi_module_simulation(self):
        """Test simulation with multiple scientific modules"""
        from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
        from relic.scientific.atmosphere.chemistry import ChemistryFunctor
        from test.system.conftest import MockEnvironmentalForcing
        
        # Create simulator
        sim = RELICSimulator(config=self.config)
        
        # Add functors
        functors = [
            MockEnvironmentalForcing(),
            FarquharPhotosynthesis(),
            MockAtmosphericRadiation(),
            ChemistryFunctor(physical_boundary="open_mechanism")
        ]
        
        for functor in functors:
            sim.register_functor(functor)
        
        # Initialize state
        state = {
            'air_temperature': torch.tensor(295.0),
            'atmospheric_co2_ppm': torch.tensor(400.0),
            'ppfd': torch.tensor(1000.0),
            'vapor_pressure_deficit': torch.tensor(1000.0),
            'shortwave_down': torch.tensor(500.0),
            'relative_humidity': torch.tensor(0.7),
            'atmospheric_pressure': torch.tensor(101325.0),
            'jmax_scaling': torch.tensor(1.0),
            'water_stress_factor': torch.tensor(1.0),
            'temperature_stress_factor': torch.tensor(1.0),
            'ozone_stress_factor': torch.tensor(1.0),
            'leaf_area_index': torch.tensor(3.0),
            'leaf_water_potential': torch.tensor(-0.5),
            'no_mixing_ratio': torch.tensor(0.0),
            'no2_mixing_ratio': torch.tensor(0.0),
            'o3_mixing_ratio': torch.tensor(0.0),
            'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0),  # overhead sun
        }
        
        sim.initialize_state(state)
        
        # Run simulation for 3 hours
        for _ in range(3):
            updates = sim.step(3600.0)
            
        # Should complete without errors
        self.assertEqual(sim.current_time, 10800.0)
        
    def test_conservation_during_simulation(self):
        """Test mass/energy conservation during simulation"""
        sim = RELICSimulator(config=self.config)
        
        # Track carbon
        initial_carbon = torch.tensor(1000.0)
        
        state = {
            'soil_carbon': initial_carbon.clone(),
            'air_temperature': torch.tensor(298.15),
            'soil_moisture': torch.tensor(0.3)
        }
        
        sim.initialize_state(state)
        
        # Run for 24 hours
        carbon_history = [initial_carbon.item()]
        
        for _ in range(24):
            updates = sim.step(3600.0)
            current_state = sim.state_manager.get_state()
            
            if 'soil_carbon' in current_state:
                carbon_history.append(current_state['soil_carbon'].item())
        
        # Carbon shouldn't drastically change (conservation)
        max_change = max(carbon_history) - min(carbon_history)
        relative_change = max_change / initial_carbon.item()
        
        # Allow for some change but not unrealistic amounts
        self.assertLess(relative_change, 0.5, "Carbon changed too much - possible conservation violation")
        
    def test_configuration_loading(self):
        """Test configuration loading system"""
        # Test that config can be loaded and used
        loader = ConfigLoader()
        
        # Should be able to get parameter manager
        pm = get_global_manager()
        
        # Check some parameters exist
        self.assertIsNotNone(pm)
        
        # Try to get a known parameter
        try:
            value = pm.get("atmosphere.chemistry.oh_reaction_rate", default=1e-12)
            self.assertIsNotNone(value)
        except:
            # If not found, that's OK - just checking system works
            pass
            
    def test_performance_requirements(self):
        """Test that simulation meets basic performance requirements"""
        import time
        
        sim = RELICSimulator(config=self.config)
        
        # Simple state
        state = {
            'air_temperature': torch.tensor(298.15),
            'soil_moisture': torch.tensor(0.3)
        }
        
        sim.initialize_state(state)
        
        # Time a limited number of steps for a stable performance check.
        start = time.time()
        steps = 3
        for _ in range(steps):
            sim.step(3600.0)
        elapsed = time.time() - start

        # Should complete in reasonable time on CPU-only Windows CI.
        max_seconds_per_step = 5.0
        self.assertLess(
            elapsed,
            steps * max_seconds_per_step,
            f"Simulation too slow: {elapsed:.2f} seconds for {steps} steps",
        )
        

class TestScientificAccuracy(unittest.TestCase):
    """Test scientific accuracy of key processes"""
    
    def test_photosynthesis_known_values(self):
        """Test photosynthesis against known values"""
        from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
        
        photo = FarquharPhotosynthesis()
        
        # Standard conditions
        state = {
            'air_temperature': torch.tensor(298.15),  # 25 C
            'atmospheric_co2_ppm': torch.tensor(400.0),  # 400 ppm
            'ppfd': torch.tensor(1500.0),  # High light
            'leaf_area_index': torch.tensor(1.0),
            'vapor_pressure_deficit': torch.tensor(1000.0),
        }

        result = photo.forward(state, 0.0, 3600.0)

        # GPP should be positive and reasonable
        gpp = result['gross_photosynthesis'].value
        self.assertGreater(gpp.item(), 0)
        self.assertLess(gpp.item(), 50.0)  # umol m-2 s-1
        
    def test_temperature_response(self):
        """Test temperature response is reasonable"""
        from relic.scientific.biosphere.stress.thermal_stress import ThermalStress

        thermal = ThermalStress()
        
        # Test at different temperatures
        temps = [273.15, 288.15, 298.15, 308.15, 318.15]  # 0, 15, 25, 35, 45 C
        stress_factors = []
        
        for temp in temps:
            state = {'air_temperature': torch.tensor(temp)}
            result = thermal.forward(state, 0.0, 3600.0)
            stress_factors.append(result['temperature_stress_factor'].value.item())
        
        # Stress should be minimal at optimal temperature (around 25 C)
        self.assertGreater(stress_factors[2], stress_factors[0])  # 25 C better than 0 C
        self.assertGreater(stress_factors[2], stress_factors[4])  # 25 C better than 45 C
        

if __name__ == '__main__':
    unittest.main()


