"""
Shared fixtures for pedosphere tests.

This module provides common test fixtures for pedosphere module testing,
including mock states, environments, and parameter managers.
"""
import pytest
import torch
from relic.parameters import get_global_manager


@pytest.fixture
def parameter_manager():
    """The process-wide parameter manager of this test (read-only use).

    Function scope: a session-cached object could be a manager that another
    test module's fresh-manager fixture has since replaced.
    """
    return get_global_manager()


@pytest.fixture
def mock_state():
    """Mock state dictionary with typical soil variables."""
    return {
        'soil_temperature': torch.tensor([288.15, 293.15, 298.15]),  # K (15degC, 20degC, 25degC)
        'soil_moisture': torch.tensor([0.3, 0.4, 0.5]),  # m3/m3
        'water_filled_pore_space': torch.tensor([0.45, 0.5, 0.55]),  # dimensionless
        'soil_ph': torch.tensor([6.5, 6.8, 7.0]),  # dimensionless

        # Carbon pools
        'carbon_fast': torch.tensor([100.0, 120.0, 90.0]),  # g C/m2
        'carbon_slow': torch.tensor([500.0, 550.0, 480.0]),  # g C/m2
        'carbon_passive': torch.tensor([1000.0, 1100.0, 950.0]),  # g C/m2
        'metabolic_litter_c': torch.tensor([10.0, 12.0, 9.0]),  # g C/m2
        'structural_litter_c': torch.tensor([50.0, 55.0, 48.0]),  # g C/m2
        'fast_som_c': torch.tensor([150.0, 160.0, 140.0]),  # g C/m2
        'slow_som_c': torch.tensor([800.0, 850.0, 750.0]),  # g C/m2
        'passive_som_c': torch.tensor([2000.0, 2100.0, 1900.0]),  # g C/m2

        # Nitrogen pools
        'nitrogen_nh4': torch.tensor([5.0, 6.0, 4.5]),  # g N/m2
        'nitrogen_no3': torch.tensor([3.0, 3.5, 2.8]),  # g N/m2
        'nh4_nitrogen': torch.tensor([5.0, 6.0, 4.5]),  # Alternative naming
        'no3_nitrogen': torch.tensor([3.0, 3.5, 2.8]),  # Alternative naming    
        'organic_nitrogen': torch.tensor([100.0, 110.0, 95.0]),  # g N/m2       
        'microbial_nitrogen': torch.tensor([1.0, 1.2, 0.9]),  # g N/m2

        # Microbial pools
        'microbial_biomass_c': torch.tensor([10.0, 12.0, 9.0]),  # g C/m2       
        'microbial_biomass_n': torch.tensor([1.0, 1.2, 0.9]),  # g N/m2
    }


@pytest.fixture
def mock_environment():
    """Mock environment dictionary with forcing variables."""
    return {
        'soil_temperature': torch.tensor([288.15, 293.15, 298.15]),  # K
        'soil_moisture': torch.tensor([0.3, 0.4, 0.5]),  # m3/m3
        'precipitation': torch.tensor([0.0, 0.001, 0.0]),  # m/s
        'soil_ph': torch.tensor([6.5, 6.8, 7.0]),  # dimensionless
        'lignin_n_ratio': torch.tensor([0.3, 0.35, 0.28]),  # dimensionless
        'labile_carbon': torch.tensor([50.0, 55.0, 48.0]),  # g C/m2
    }


@pytest.fixture
def basic_config():
    """Basic configuration dictionary for Functors."""
    return {
        'name': 'test_functor',
        'enabled': True,
        'phase': 'MAIN_PROCESSES'
    }


@pytest.fixture
def single_state():
    """Single-point state for simple tests (scalar tensors)."""
    return {
        'soil_temperature': torch.tensor(293.15),  # 20degC
        'soil_moisture': torch.tensor(0.4),  # m3/m3
        'water_filled_pore_space': torch.tensor(0.5),  # dimensionless
        'soil_ph': torch.tensor(6.8),
        'carbon_fast': torch.tensor(100.0),
        'metabolic_litter_c': torch.tensor(10.0),
        'structural_litter_c': torch.tensor(50.0),
        'fast_som_c': torch.tensor(150.0),
        'slow_som_c': torch.tensor(800.0),
        'passive_som_c': torch.tensor(2000.0),
        'available_som_c': torch.tensor(50.0),  # Required for microbial_enzyme
        'nh4_nitrogen': torch.tensor(5.0),
        'no3_nitrogen': torch.tensor(3.0),
        'organic_nitrogen': torch.tensor(100.0),  # Required for mineralization
        'microbial_nitrogen': torch.tensor(1.5),  # Required for mineralization 
        'microbial_biomass_c': torch.tensor(10.0),
    }


@pytest.fixture
def single_environment():
    """Single-point environment for simple tests."""
    return {
        'soil_temperature': torch.tensor(293.15),
        'soil_moisture': torch.tensor(0.4),
        'soil_ph': torch.tensor(6.8),
        'lignin_n_ratio': torch.tensor(0.3),
        'labile_carbon': torch.tensor(50.0),
    }


@pytest.fixture
def cold_environment():
    """Cold temperature environment for testing temperature sensitivity."""
    return {
        'soil_temperature': torch.tensor(278.15),  # 5degC
        'soil_moisture': torch.tensor(0.4),
        'soil_ph': torch.tensor(6.8),
    }


@pytest.fixture
def warm_environment():
    """Warm temperature environment for testing temperature sensitivity."""
    return {
        'soil_temperature': torch.tensor(303.15),  # 30degC
        'soil_moisture': torch.tensor(0.4),
        'soil_ph': torch.tensor(6.8),
    }


@pytest.fixture
def dry_environment():
    """Dry environment for testing moisture limitation."""
    return {
        'soil_temperature': torch.tensor(293.15),
        'soil_moisture': torch.tensor(0.15),  # Below wilting point
        'soil_ph': torch.tensor(6.8),
    }


@pytest.fixture
def wet_environment():
    """Wet/saturated environment for testing anaerobic conditions."""
    return {
        'soil_temperature': torch.tensor(293.15),
        'soil_moisture': torch.tensor(0.7),  # High saturation
        'soil_ph': torch.tensor(6.8),
    }


@pytest.fixture
def acidic_environment():
    """Acidic pH environment."""
    return {
        'soil_temperature': torch.tensor(293.15),
        'soil_moisture': torch.tensor(0.4),
        'soil_ph': torch.tensor(5.0),  # Acidic
    }


@pytest.fixture
def alkaline_environment():
    """Alkaline pH environment."""
    return {
        'soil_temperature': torch.tensor(293.15),
        'soil_moisture': torch.tensor(0.4),
        'soil_ph': torch.tensor(8.5),  # Alkaline
    }

