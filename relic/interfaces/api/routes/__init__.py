"""
API Routes

Route modules for the RELIC REST API organized by functional area.

Available route modules:
    health      System health and diagnostics
    data        Data I/O operations and streaming
    compute     Computational optimization and benchmarking
    optimize    Parameter calibration, assimilation, sensitivity, UQ
    stochastic  Stochastic processes and uncertainty quantification
    config      Configuration management and validation
    graph       Graph orchestration validation and compilation

Each route module provides:
    - Structured request/response models
    - Comprehensive error handling
    - Async support for long-running operations
    - Consistent API response format
    - OpenAPI documentation

Authors: Yuchong Long
"""

__all__ = ['health', 'data', 'compute', 'optimize', 'stochastic', 'config', 'graph']
