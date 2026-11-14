"""Shared fixtures for L2 integration tests."""
import pytest
import torch
import numpy as np
import xarray as xr
from datetime import datetime
from relic.data.standards.unit_converter import UnitConverter
from relic.parameters import get_global_manager


@pytest.fixture
def unit_converter():
    """Return a fresh UnitConverter instance."""
    return UnitConverter()


@pytest.fixture
def parameter_manager():
    """Return the global parameter manager."""
    return get_global_manager()


@pytest.fixture
def sample_xarray_dataset():
    """Return a sample xarray Dataset for testing adapters.

    This mimics the structure of climate model output (WRF, CESM, etc.).
    """
    times = [datetime(2020, 1, 1, i) for i in range(24)]
    lats = np.linspace(30, 40, 10)
    lons = np.linspace(-120, -110, 10)

    # Temperature in Celsius
    temp = 15 + 5 * np.random.randn(24, 10, 10)
    # Precipitation in mm day-1
    precip = np.abs(2 + 1 * np.random.randn(24, 10, 10))

    ds = xr.Dataset(
        {
            'air_temperature': (['time', 'lat', 'lon'], temp),
            'precipitation': (['time', 'lat', 'lon'], precip),
        },
        coords={
            'time': times,
            'lat': lats,
            'lon': lons,
        }
    )

    # Add CF-compliant attributes
    ds['air_temperature'].attrs['units'] = 'celsius'
    ds['air_temperature'].attrs['standard_name'] = 'air_temperature'

    ds['precipitation'].attrs['units'] = 'mm day-1'
    ds['precipitation'].attrs['standard_name'] = 'precipitation_flux'

    return ds


@pytest.fixture
def sample_state_dict():
    """Return a sample state dictionary for Functor tests."""
    return {
        'leaf_area_index': torch.tensor([2.5]),
        'air_temperature': torch.tensor([298.15]),  # K
        'soil_moisture': torch.tensor([0.3]),  # m3/m3
        'par': torch.tensor([1000.0]),  # umol/m2/s
    }


@pytest.fixture
def sample_environment_dict():
    """Return a sample environment dictionary."""
    return {
        'air_temperature': torch.tensor([298.15]),
        'relative_humidity': torch.tensor([0.6]),
        'wind_speed': torch.tensor([2.0]),
        'atmospheric_pressure': torch.tensor([101325.0]),
    }

