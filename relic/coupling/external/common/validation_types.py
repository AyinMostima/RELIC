"""
Shared validation type definitions for external coupling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Union

PathLike = Union[str, Path]


@dataclass
class ValidationReport:
    """
    Structured validation report for external data.

    Contains diagnostic information about bounded external-adapter checks,
    variable completeness, and transformation results. It never certifies
    dataset-level CF conformance.
    """

    source_path: PathLike
    validation_time: str
    cf_compliant: bool = False
    cf_validation_scope: str = "external_adapter_checks_not_dataset_cf"
    dataset_conformance_checked: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    variable_mapping: Dict[str, str] = field(default_factory=dict)
    unit_conversions: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    missing_data_stats: Dict[str, float] = field(default_factory=dict)
    conservation_residuals: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.cf_compliant, bool):
            raise TypeError("cf_compliant must be bool")
        if self.cf_compliant:
            raise ValueError(
                "ValidationReport cannot claim dataset-level CF conformance"
            )
        if self.dataset_conformance_checked is not False:
            raise ValueError("dataset_conformance_checked must remain false")
