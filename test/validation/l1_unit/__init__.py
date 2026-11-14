"""
L1 Unit Tests for Unified Validation Framework

Tests individual components and functions in isolation:
- Core data models (Pydantic validation, serialization)
- Registry operations (rule registration, lookup, dependencies)
- Individual validation rules (parameters, stochastic, cross-domain)
- Decorators and utility functions
- Mathematical correctness of scoring rules

Focus Areas:
1. Model Validation: ValidationResultModel, ValidationReportModel schema correctness
2. Registry Integrity: Rule registration, metadata handling, dependency resolution
3. Rule Logic: Individual validation rule mathematical correctness
4. Utility Functions: Decorators and helpers
5. Error Handling: Exception cases and edge conditions
"""

