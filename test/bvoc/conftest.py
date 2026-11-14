"""Pytest fixtures for BVOC test utilities."""

from unittest.mock import patch

import pytest

from test.bvoc_parameter_manager_helper import build_realistic_parameter_manager


@pytest.fixture
def bvoc_parameter_manager():
    """Provide a realistic ParameterManager proxy for tests."""
    return build_realistic_parameter_manager()


@pytest.fixture(autouse=True)
def patch_parameter_manager(bvoc_parameter_manager):
    """Automatically patch get_global_manager to return the realistic proxy."""
    with (
        patch('relic.parameters.manager.get_global_manager', return_value=bvoc_parameter_manager),
        patch('relic.parameters.get_global_manager', return_value=bvoc_parameter_manager),
    ):
        yield


@pytest.fixture
def param_manager(bvoc_parameter_manager):
    """Backward-consistent fixture name used by historical tests."""
    return bvoc_parameter_manager


