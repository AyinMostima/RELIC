"""Report/export helpers for CrossDomainValidator."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, cast

from .framework import ValidationContext, ValidationEngine, ValidationReporter
from .framework.models import ValidationCode, ValidationLevel, ValidationResultModel

logger = logging.getLogger(__name__)


def generate_validation_report_impl(validator: Any) -> str:
    """Generate plain-text validation report."""
    report = ["=" * 60]
    report.append("CROSS-DOMAIN VALIDATION REPORT")
    report.append("=" * 60)

    if validator.validation_errors:
        report.append("\nERRORS:")
        for error in validator.validation_errors:
            report.append(f"   {error}")
    else:
        report.append("\n No errors found")

    if validator.validation_warnings:
        report.append("\nWARNINGS:")
        for warning in validator.validation_warnings:
            report.append(f"   {warning}")
    else:
        report.append("\n No warnings")

    report.append("\nREGISTERED VARIABLES:")
    domains = sorted(
        {spec.domain for spec in validator.variable_registry.values()},
        key=lambda d: getattr(d, "value", str(d)),
    )
    for domain in domains:
        domain_vars = [v for v in validator.variable_registry.values() if v.domain == domain]
        if not domain_vars:
            continue
        domain_name = getattr(domain, "value", str(domain))
        report.append(f"\n  {domain_name.upper()}:")
        for var in domain_vars:
            report.append(f"    - {var.name} [{var.units}]")

    report.append("\nINTERFACE CONTRACTS:")
    for contract in validator.interface_contracts:
        report.append(f"\n  {contract.source_domain.value}  {contract.target_domain.value}:")
        for var in contract.variables:
            report.append(f"    - {var.name} [{var.units}]")

    report.append("\n" + "=" * 60)
    return "\n".join(report)


def to_report_impl(validator: Any, include_metadata: bool = True) -> Any:
    """Generate framework ValidationReportModel and append validator findings."""
    context = ValidationContext(
        state_manager=validator.state_manager,
        metadata={
            "validator_errors": validator.validation_errors.copy(),
            "validator_warnings": validator.validation_warnings.copy(),
            "variable_registry": validator.variable_registry,
            "interface_contracts": validator.interface_contracts,
        },
    )

    engine = ValidationEngine()
    report = engine.run(
        layers=["L1", "L2"],
        domains=["cf_conventions", "hydrosphere", "biosphere", "atmosphere", "pedosphere"],
        tags=["cf", "cross_domain", "units"],
        context=context,
        include_metrics=include_metadata,
    )

    for error in validator.validation_errors:
        report.results.append(
            ValidationResultModel(
                level=ValidationLevel.ERROR,
                code=ValidationCode.GENERIC_VALIDATION_FAILED,
                message=error,
                layer="L2",
                scope="cross_domain_validator",
                tags=["cross_domain"],
            )
        )

    for warning in validator.validation_warnings:
        report.results.append(
            ValidationResultModel(
                level=ValidationLevel.WARNING,
                code=ValidationCode.GENERIC_VALIDATION_FAILED,
                message=warning,
                layer="L2",
                scope="cross_domain_validator",
                tags=["cross_domain"],
            )
        )

    report._update_summary()
    return report


def to_json_impl(
    validator: Any,
    indent: int = 2,
    include_metadata: bool = True,
) -> str:
    """Export validation report as JSON."""
    report = to_report_impl(validator, include_metadata=include_metadata)
    reporter = ValidationReporter()
    return cast(str, reporter.to_json(report, indent=indent, include_metadata=include_metadata))


def to_markdown_impl(validator: Any, include_details: bool = True) -> str:
    """Export validation report as Markdown."""
    report = to_report_impl(validator, include_metadata=True)
    reporter = ValidationReporter()
    return cast(str, reporter.to_markdown(report, include_details=include_details))


def run_unified_validation_impl(
    validator: Any,
    layers: Optional[List[str]] = None,
    domains: Optional[List[str]] = None,
) -> Any:
    """Run framework validation with validator context."""
    if layers is None:
        layers = ["L1", "L2"]
    if domains is None:
        domains = ["cf_conventions", "hydrosphere", "biosphere", "atmosphere", "pedosphere"]

    context = ValidationContext(
        state_manager=validator.state_manager,
        metadata={
            "variable_registry": validator.variable_registry,
            "interface_contracts": validator.interface_contracts,
        },
    )
    engine = ValidationEngine()
    return engine.run(
        layers=layers,
        domains=domains,
        context=context,
        include_metrics=True,
    )
