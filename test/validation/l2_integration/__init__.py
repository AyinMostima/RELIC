"""
L2 Integration Tests - Unified Validation Framework

This package contains L2 integration tests for the RELIC Unified Validation Framework.
L2 tests focus on integration between validation components, cross-domain validation,
conservation law verification, and validation engine orchestration.

Test Structure:
    - test_cf_compliance.py: CF-1.8 standard compliance and cross-domain integration
    - test_conservation_coupling.py: Mass/energy conservation and coupling contracts  
    - test_engine_integration.py: Validation engine orchestration and dependency resolution

Mathematical Framework:
    L2 integration validation encompasses:
    - Conservation equations: 鈭?inputs) - 鈭?outputs) - 鈭?storage_changes) 鈮?0
    - Interface contracts: variable consistency across domain boundaries
    - CF-1.8 compliance: standard_name 鈭?CFSR vocabulary, units 鈭?UDUNITS-2
    - Dependency resolution: topological ordering of validation rules

Coverage:
    - Cross-domain variable mapping and interface validation
    - Mass balance conservation (C, N, P, H2O)
    - Energy balance conservation (Rn - H - LE - G 鈮?0)
    - Biogeochemical coupling validation (carbon-water, nutrient cycles)
    - Rule dependency resolution and parallel execution
    - Report generation and aggregation

References:
    - CF-1.8 Convention: https://cfconventions.org/
    - Gneiting & Raftery (2007): Strictly proper scoring rules
    - Chapin et al. (2006): Reconciling carbon-cycle concepts
"""

# L2 Integration test modules
from . import test_cf_compliance
from . import test_conservation_coupling  
from . import test_engine_integration

__all__ = [
    'test_cf_compliance',
    'test_conservation_coupling',
    'test_engine_integration'
]

# Test configuration constants
L2_TEST_CONFIG = {
    'conservation_thresholds': {
        'strict': 1e-12,
        'tight': 1e-6, 
        'moderate': 1e-4,
        'loose': 1e-2
    },
    'cf_compliance': {
        'require_standard_names': True,
        'allow_unit_conversion': True,
        'check_coordinate_systems': True
    },
    'integration_settings': {
        'parallel_execution': True,
        'dependency_resolution': True,
        'comprehensive_reporting': True
    }
}

# Common test fixtures and utilities could be added here
def get_l2_test_config():
    """Get L2 integration test configuration."""
    return L2_TEST_CONFIG.copy()

