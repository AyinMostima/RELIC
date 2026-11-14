"""Shared fixtures for L1 unit tests."""
import sys
from pathlib import Path

# Ensure project root is on sys.path for local test runs.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import pytest
import torch
import numpy as np
from relic.data.standards.unit_converter import UnitConverter, ConversionContext


@pytest.fixture
def unit_converter():
    """Return a fresh UnitConverter instance."""
    return UnitConverter()


@pytest.fixture
def conversion_context():
    """Return a basic ConversionContext instance."""
    return ConversionContext()


@pytest.fixture
def sample_tensor():
    """Return a sample tensor for testing."""
    return torch.tensor([25.0, 30.0, 35.0])


@pytest.fixture
def sample_numpy():
    """Return a sample numpy array for testing."""
    return np.array([25.0, 30.0, 35.0])


@pytest.fixture
def sample_float():
    """Return a sample float for testing."""
    return 25.0


@pytest.fixture
def sample_int():
    """Return a sample integer for testing."""
    return 25


@pytest.fixture(params=[
    torch.tensor([25.0]),
    25.0,
    25,
    np.array([25.0])
])
def mixed_type_input(request):
    """Parametrized fixture providing different input types."""
    return request.param
