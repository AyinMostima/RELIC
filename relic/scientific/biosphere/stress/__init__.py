"""
Plant Stress Response Processes for RELIC

This module provides a comprehensive framework for modeling plant responses to
environmental stresses including water stress, temperature stress, nutrient
limitations, and ozone stress.

Main Components
---------------
- WaterStress: Drought and water limitation responses using water potential
- ThermalStress: Temperature stress (heat and cold) with Gaussian response
- NutrientStress: Nitrogen and phosphorus limitations using critical concentrations  
- OzoneStress: Ozone stress using stomatal flux-based approaches
- StressIntegration: Multi-factor stress interactions (future development)

Mathematical Framework
----------------------
1. Water stress: f_water = (psi - psi_wilt) / (psi_fc - psi_wilt)
2. Temperature stress: f_temp = exp(-0.5 * ((T - T_opt)/sigma)^2)
3. Nutrient stress: f_N = ([N] - N_min) / (N_crit - N_min)
4. Ozone stress: F_O3 = g_s * [O3] * uptake_efficiency

Units
-----
- Water potential (psi): MPa
- Temperature: K
- Nutrient concentrations: kg element kg^-1 DM
- Ozone concentration: mol mol^-1
- Stress factors: dimensionless [0-1]

References
----------
- Boyer, J.S. (1982). Plant productivity and environment. Science 218, 443-448
- Berry, J. & Bjorkman, O. (1980). Photosynthetic response and adaptation to
  temperature. Annual Review of Plant Physiology 31, 491-543
- Chapin III, F.S. (1991). Integrated responses of plants to stress. 
  BioScience 41, 29-36
- Mills, G. et al. (2011). A synthesis of AOT40-based response functions.
  Atmospheric Environment 45, 1703-1713

Example Usage
-------------
    from relic.scientific.biosphere.stress import WaterStress, ThermalStress
    from relic.core.state import StateManager
    
    # Parameters are automatically loaded from YAML files via ParameterManager
    # No manual parameter passing required
    
    # Create stress functors
    water_stress = WaterStress(name="water_stress")
    thermal_stress = ThermalStress(name="thermal_stress")
    
    # Initialize state
    state_manager = StateManager()
    state_manager.initialize_state({
        'soil_water_potential': -0.8,  # MPa
        'air_temperature': 308.15,  # K (35 C)
        'nitrogen_concentration': 0.025,  # kg N kg^-1 DM
        'o3_mixing_ratio': 60e-9  # mol mol^-1
    })
    
    # Calculate stress factors
    water_result = water_stress.forward(
        state=state_manager.get_state(),
        time=0.0,
        dt=3600.0  # 1 hour timestep
    )
    
    # Access results
    water_stress_factor = water_result['water_stress_factor'].value
    water_stress_text = f"Water stress factor: {float(water_stress_factor):.3f}"

Integration with RELIC
------------------------
All stress functors inherit from BiologicalProcessFunctor and integrate
seamlessly with the RELIC framework:

1. Parameter management via ParameterManager
2. State management via StateManager  
3. Automatic differentiation via PyTorch
4. Conservation checking via FunctorBase
5. Dependency resolution via DependencyGraphResolver

Configuration
-------------
Stress parameters are stored in:
relic/parameters/database/relic/biosphere/stress.yaml

Parameters are organized by namespace:
- biosphere.stress.water.*
- biosphere.stress.thermal.*
- biosphere.stress.nutrient.*
- biosphere.stress.ozone.*
"""

# Import stress response modules
from .water_stress import WaterStress
from .thermal_stress import ThermalStress
from .nutrient_stress import NutrientStress
from .ozone_stress import OzoneStress

__all__ = [
    'WaterStress',
    'ThermalStress',
    'NutrientStress',
    'OzoneStress',
]
