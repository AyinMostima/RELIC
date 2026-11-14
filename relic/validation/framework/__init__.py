"""
Unified Validation Framework (UVF) for RELIC

This framework provides a unified approach to validation across all scientific domains
in RELIC, standardizing validation rules, result structures, and reporting formats.

Key Components:
    - ValidationResultModel: Unified result structure (Pydantic)
    - ValidationRegistry: Rule registration and discovery
    - ValidationEngine: Execution orchestration (L1-L4 layers)
    - ValidationDecorators: Rule standardization (@validated_rule)
    - ValidationReporters: Multi-format output (JSON, YAML, Markdown, NetCDF)

Architecture:
    L1_UNIT: Parameter specs, distributions, units, scoring rules
    L2_INTEGRATION: Interface contracts, conservation laws, coupling validation
    L3_BENCHMARK: Comparison with established models/baselines
    L4_SCIENTIFIC: Validation against observational data

Design Principles:
    - Rule composability: Validation rules can be combined and reused
    - CF metadata subset: scoped support for climate-data variable metadata
    - Scientific rigor: Based on proper scoring rules (Gneiting & Raftery 2007)

Usage:
    >>> from relic.validation.framework import ValidationEngine
    >>> engine = ValidationEngine()
    >>> report = engine.run(layers=["L2"], domains=["biosphere", "atmosphere"])
    >>> report_json = report.to_json()

References:
    - Gneiting, T. & Raftery, A.E. (2007). Strictly proper scoring rules,
      prediction, and estimation. J. Am. Stat. Assoc. 102, 359-378.
    - CF-1.8 Conventions: https://cfconventions.org/
"""

from .models import (
    ValidationLevel,
    ValidationCode,
    ValidationDetail,
    ValidationResultModel,
    ValidationReportModel,
    ValidationContext
)

from .registry import ValidationRegistry, RuleMetadata, get_global_registry
from .engine import ValidationEngine
from .decorators import validated_rule
from .reporters import ValidationReporter
from .layers import L1_UNIT, L2_INTEGRATION, L3_BENCHMARK, L4_SCIENTIFIC

def __getattr__(name: str) -> ValidationRegistry:
    """Resolve the resettable canonical registry without caching a stale alias."""
    if name == "validation_registry":
        return get_global_registry()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Version info
__version__ = "1.0.0"
__author__ = "Yuchong Long"

__all__ = [
    # Core models
    "ValidationLevel",
    "ValidationCode",
    "ValidationDetail",
    "ValidationResultModel",
    "ValidationReportModel",
    "ValidationContext",

    # Framework components
    "ValidationRegistry",
    "ValidationEngine",
    "ValidationReporter",
    "RuleMetadata",

    # Decorators
    "validated_rule",

    # Layer constants
    "L1_UNIT",
    "L2_INTEGRATION",
    "L3_BENCHMARK",
    "L4_SCIENTIFIC",

    # Instances
    "validation_registry"
]
