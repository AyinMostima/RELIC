"""
Pytest Configuration and Fixtures for Hydrosphere Tests

Provides shared fixtures for testing hydrosphere modules.
"""

import pytest
import torch
from typing import Dict
from relic.parameters import get_global_manager


@pytest.fixture(scope="session", autouse=True)
def initialize_parameter_manager():
    """
    Initialize the global parameter manager for all tests.

    This fixture runs once per test session and ensures that all
    hydrosphere parameters are available before any tests execute.
    """
    param_manager = get_global_manager()

    yield param_manager


@pytest.fixture
def basic_state() -> Dict:
    """Minimal state variables for basic testing."""
    return {
        "air_temperature": torch.tensor(293.15),  # 20 degC in K
        "soil_moisture": torch.tensor(0.3),  # m3/m3
        "atmospheric_pressure": torch.tensor(101325.0),  # Pa (sea level)
    }


@pytest.fixture
def et_state() -> Dict:
    """State variables for evapotranspiration testing."""
    return {
        "net_radiation": torch.tensor(200.0),  # W m-2
        "air_temperature": torch.tensor(293.15),  # 20 degC in K
        "relative_humidity": torch.tensor(0.7),  # dimensionless
        "wind_speed": torch.tensor(2.0),  # m s-1
        "atmospheric_pressure": torch.tensor(101325.0),  # Pa
        "surface_resistance": torch.tensor(100.0),  # s m-1
    }


@pytest.fixture
def soil_moisture_state() -> Dict:
    """State variables for soil moisture testing."""
    return {
        "soil_moisture": torch.tensor(0.3),  # m3/m3
        "precipitation_flux": torch.tensor(0.001),  # kg m-2 s-1 (3.6 mm/h)
        "evapotranspiration_flux": torch.tensor(0.0001),  # kg m-2 s-1
        "air_temperature": torch.tensor(293.15),  # K
    }


@pytest.fixture
def runoff_state() -> Dict:
    """State variables for runoff testing."""
    return {
        "soil_moisture": torch.tensor(0.4),  # m3/m3
        "precipitation_flux": torch.tensor(0.005),  # kg m-2 s-1 (18 mm/h)
        "infiltration_capacity": torch.tensor(0.002),  # kg m-2 s-1
        "drainage_flux": torch.tensor(0.0001),  # kg m-2 s-1
    }


@pytest.fixture
def time_params() -> Dict:
    """Standard time parameters for testing."""
    return {
        "time": 0.0,  # s
        "dt": 3600.0,  # s (1 hour)
    }


@pytest.fixture
def tolerance() -> Dict:
    """Numerical tolerances for assertions."""
    return {
        "absolute": 1e-6,
        "relative": 1e-4,
        "flux": 1e-8,  # kg m-2 s-1
    }

