"""Shared fixtures for L3 benchmark/system tests."""
import pytest
import torch


@pytest.fixture
def large_scale_tensor():
    """Return a large-scale tensor for performance testing."""
    return torch.randn(1000, 100, 100)  # Simulating spatial grid


@pytest.fixture
def performance_threshold():
    """Return performance thresholds for unit conversion overhead."""
    return {
        'max_overhead_percent': 1.0,  # Max 1% overhead
        'max_latency_ms': 10.0,  # Max 10ms per conversion
    }

