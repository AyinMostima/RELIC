"""
Data Quality Control Module for RELIC

This module provides quality control components for scientific data validation,
cleaning, and gap filling. Ensures data integrity before ingestion into the
model system.

Components:
    - validators: Data quality validators for scientific variables
    - cleaners: Data cleaning and filtering utilities
    - gap_fillers: Gap-filling algorithms for missing data

Key Features:
    - CF-1.8 convention compliance checking
    - Outlier detection and removal
    - Temporal consistency validation
    - Spatial consistency validation
    - Missing data interpolation
    - Quality flag generation

Usage:
    >>> from relic.data.quality import QualityValidator
    >>> validator = QualityValidator()
    >>> is_valid, report = validator.check_data(dataset)

Design Principles:
    - Non-destructive: Original data preserved with quality flags
    - Traceable: All operations logged with provenance metadata
    - Configurable: Thresholds and rules customizable per variable
    - Standards-compliant: Follows CF-1.8 and CMIP quality standards

References:
    - CF-1.8 Quality Control Conventions
    - CMIP Data Quality Guidelines
    - FluxNet Data Quality Control Protocols
"""

from .._placeholder import list_placeholder_exports, require_experimental_feature
from .validators import QualityValidator, RangeRule, ValidationIssue

require_experimental_feature(
    feature_name="data quality placeholder module",
    env_var="RELIC_EXPERIMENTAL_DATA_QUALITY",
    module_name="relic.data.quality",
)

__all__ = [
    "QualityValidator",
    "RangeRule",
    "ValidationIssue",
]

# Version info
__version__ = "0.1.0"
__author__ = "Yuchong Long"

get_available_components = lambda: list_placeholder_exports(__all__)

# Future extension notes:
# - cleaners: Outlier removal, spike detection, smoothing filters
# - gap_fillers: Linear interpolation, spline fitting, MDS gap filling
