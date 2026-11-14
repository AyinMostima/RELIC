"""
Unified Validation Framework (UVF) Test Suite

This package contains comprehensive tests for the unified validation framework,
organized by validation layers (L1-L4) and testing different aspects:

Test Organization:
    L1 Unit Tests:
        - Core models validation (ValidationResultModel, ValidationReportModel)
        - Registry functionality (rule registration, discovery, dependencies)
        - Individual validation rules (parameters, stochastic scores, etc.)
        - Decorators and utility functions

    L2 Integration Tests:
        - Engine execution orchestration
        - Cross-domain rule interactions
        - Conservation and coupling validation
        - Reporter functionality

    L3 System Tests:  
        - End-to-end validation workflows
        - Performance and scalability
        - historical consistency
        - Error handling and recovery

Test Principles:
    - Scientific rigor: All tests validate mathematical correctness
    - CF-1.8 compliance: Ensure climate data standard adherence
    - Conservation laws: Verify mass/energy balance checks
    - Proper scoring rules: Validate probabilistic metrics (CRPS, PIT, Brier)
    
Usage:
    # Run all validation tests
    pytest test/validation/
    
    # Run specific layer tests
    pytest test/validation/l1_unit/
    pytest test/validation/l2_integration/
    pytest test/validation/l3_system/
    
    # Run domain-specific tests
    pytest test/validation/ -k "cross_domain"
    pytest test/validation/ -k "conservation"
"""

__version__ = "1.0.0"
__author__ = "Yuchong Long"


