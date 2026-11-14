#!/usr/bin/env python3
"""
Isotope Parameter Audit CLI Tool

Command-line interface for comprehensive parameter auditing and quality assessment
of the RELIC isotope tracking system.

Usage:
    python -m relic.scientific.isotopes.audit_cli audit [--output-dir OUTPUT_DIR]
    python -m relic.scientific.isotopes.audit_cli quality [--format FORMAT] [--output-dir OUTPUT_DIR]
    python -m relic.scientific.isotopes.audit_cli full [--output-dir OUTPUT_DIR]
    python -m relic.scientific.isotopes.audit_cli check PARAM_KEY
"""

import argparse
import logging
import sys
from pathlib import Path

from relic.parameters import get_global_manager, list_parameters

from .parameter_auditor import create_parameter_auditor, run_comprehensive_audit
from .parameter_quality_assessor import (
    QualityLevel,
    assess_parameter_quality,
    create_quality_assessor,
)

logger = logging.getLogger(__name__)


def _list_isotope_parameters() -> dict:
    """Read the isotope namespace through the canonical ParameterManager API."""
    manager = get_global_manager()
    return {
        key: manager.get(key, required=True)
        for key in list_parameters(namespace="isotopes")
    }


def _emit(message: str = "") -> None:
    """Emit user-facing CLI output through logger."""
    logger.info(message)


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="RELIC Isotope Parameter Audit and Quality Assessment Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run comprehensive parameter audit
  python audit_cli.py audit --output-dir ./reports

  # Generate quality assessment report
  python audit_cli.py quality --format markdown --output-dir ./reports

  # Run both audit and quality assessment
  python audit_cli.py full --output-dir ./reports

  # Check specific parameter
  python audit_cli.py check isotopes.temperature.min
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Audit command
    audit_parser = subparsers.add_parser('audit', help='Run parameter audit')
    audit_parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./audit_reports',
        help='Output directory for audit reports (default: ./audit_reports)'
    )

    # Quality command
    quality_parser = subparsers.add_parser('quality', help='Generate quality assessment')
    quality_parser.add_argument(
        '--format', '-f',
        choices=['markdown', 'json', 'html'],
        default='markdown',
        help='Output format (default: markdown)'
    )
    quality_parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./quality_reports',
        help='Output directory for quality reports (default: ./quality_reports)'
    )

    # Full command
    full_parser = subparsers.add_parser('full', help='Run both audit and quality assessment')
    full_parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./comprehensive_reports',
        help='Output directory for comprehensive reports (default: ./comprehensive_reports)'
    )
    full_parser.add_argument(
        '--quality-format', '-f',
        choices=['markdown', 'json', 'html'],
        default='markdown',
        help='Quality report format (default: markdown)'
    )

    # Check command
    check_parser = subparsers.add_parser('check', help='Check specific parameter')
    check_parser.add_argument(
        'parameter_key',
        type=str,
        help='Parameter key to check (e.g., isotopes.temperature.min)'
    )
    check_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    return parser


def run_audit_command(args) -> int:
    """Run parameter audit command."""
    logger.info("Starting parameter audit...")

    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run comprehensive audit
        audit_results = run_comprehensive_audit()

        if not audit_results:
            logger.error("No parameters found for audit")
            return 1

        # Generate and save audit report
        auditor = create_parameter_auditor()
        report_path = auditor.save_audit_report(audit_results, str(output_dir / "parameter_audit_report.md"))

        logger.info("Audit completed successfully")
        logger.info(f"Report saved to: {report_path}")

        # Print summary statistics
        total_params = len(audit_results)
        total_rules = sum(len(results) for results in audit_results.values())
        passed_rules = sum(
            1 for results in audit_results.values()
            for result in results
            if result.status == 'PASS'
        )
        error_rules = sum(
            1 for results in audit_results.values()
            for result in results
            if result.status == 'ERROR'
        )

        _emit("\nAudit Summary:")
        _emit(f"  Parameters audited: {total_params}")
        _emit(f"  Rules checked: {total_rules}")
        _emit(f"  Rules passed: {passed_rules} ({passed_rules/total_rules*100:.1f}%)")
        _emit(f"  Rules failed: {error_rules} ({error_rules/total_rules*100:.1f}%)")
        _emit(f"  Report: {report_path}")

        # A failed audit rule is a failed audit.
        if error_rules:
            logger.error(f"Audit failed: {error_rules} rule check(s) reported ERROR")
            return 1
        return 0

    except (OSError, KeyError, ValueError) as e:
        # CLI boundary: a file error or a parameter lookup or validation error ends the
        # command with a non-zero exit code; any other error propagates.
        logger.error(f"Audit failed: {e}")
        return 1


def run_quality_command(args) -> int:
    """Run quality assessment command."""
    logger.info("Starting quality assessment...")

    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get parameter metadata
        parameters_metadata = {}

        # Get all available parameters from ParameterManager
        available = _list_isotope_parameters()
        for param_name, value in available.items():
            param_metadata = {
                "key": param_name,
                "value": value,
                "units": "unknown",
                "description": f"Parameter {param_name}",
            }
            parameters_metadata[param_name] = param_metadata

        if not parameters_metadata:
            logger.error("No parameters found for quality assessment")
            return 1

        # Assess quality
        quality_assessor = create_quality_assessor()
        quality_metrics = quality_assessor.assess_all_parameters(parameters_metadata)

        # Generate and save quality report
        report_path = quality_assessor.save_quality_report(
            quality_metrics,
            str(output_dir / f"parameter_quality_report.{args.format}"),
            args.format
        )

        logger.info("Quality assessment completed successfully")
        logger.info(f"Report saved to: {report_path}")

        # Print summary statistics
        summary = quality_assessor.generate_quality_summary(quality_metrics)

        _emit("\nQuality Assessment Summary:")
        _emit(f"  Parameters assessed: {summary.total_parameters}")
        _emit(f"  Average completeness: {summary.average_completeness:.2%}")
        _emit(f"  Excellent quality: {summary.quality_distribution[QualityLevel.EXCELLENT]}")
        _emit(f"  Good quality: {summary.quality_distribution[QualityLevel.GOOD]}")
        _emit(f"  Fair quality: {summary.quality_distribution[QualityLevel.FAIR]}")
        _emit(f"  Poor quality: {summary.quality_distribution[QualityLevel.POOR]}")
        _emit(f"  Report: {report_path}")

        # Show critical issues if any
        if summary.critical_issues:
            _emit("\nCritical Issues:")
            for issue in summary.critical_issues:
                _emit(f"  - {issue}")

        return 0

    except (OSError, KeyError, ValueError) as e:
        # CLI boundary: a file error or a parameter lookup or validation error ends the
        # command with a non-zero exit code; any other error propagates.
        logger.error(f"Quality assessment failed: {e}")
        return 1


def run_full_command(args) -> int:
    """Run comprehensive audit and quality assessment."""
    logger.info("Starting comprehensive analysis...")

    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run audit
        logger.info("Running parameter audit...")
        audit_results = run_comprehensive_audit()

        # Get parameter metadata for quality assessment
        parameters_metadata = {}

        available = _list_isotope_parameters()
        for param_name, value in available.items():
            parameters_metadata[param_name] = {
                "key": param_name,
                "value": value,
                "units": "unknown",
                "description": f"Parameter {param_name}",
            }

        # Run quality assessment
        logger.info("Running quality assessment...")
        quality_assessor = create_quality_assessor()
        quality_metrics = quality_assessor.assess_all_parameters(parameters_metadata)

        # Save audit report
        auditor = create_parameter_auditor()
        audit_report_path = auditor.save_audit_report(
            audit_results,
            str(output_dir / "audit_report.md")
        )

        # Save quality report
        quality_report_path = quality_assessor.save_quality_report(
            quality_metrics,
            str(output_dir / f"quality_report.{args.quality_format}"),
            args.quality_format
        )

        logger.info("Comprehensive analysis completed successfully")

        # Print combined summary
        _emit("\nComprehensive Analysis Summary:")
        _emit(f"  Output directory: {output_dir}")
        _emit(f"  Audit report: {audit_report_path}")
        _emit(f"  Quality report: {quality_report_path}")

        # Audit summary
        if audit_results:
            total_params = len(audit_results)
            total_rules = sum(len(results) for results in audit_results.values())
            passed_rules = sum(
                1 for results in audit_results.values()
                for result in results
                if result.status == 'PASS'
            )
            _emit("\nAudit Results:")
            _emit(f"  Parameters audited: {total_params}")
            _emit(f"  Rules passed: {passed_rules}/{total_rules} ({passed_rules/total_rules*100:.1f}%)")

        # Quality summary
        if quality_metrics:
            summary = quality_assessor.generate_quality_summary(quality_metrics)
            _emit("\nQuality Results:")
            _emit(f"  Parameters assessed: {summary.total_parameters}")
            _emit(f"  Average completeness: {summary.average_completeness:.2%}")
            _emit(f"  Parameters needing improvement: {summary.quality_distribution[QualityLevel.POOR] + summary.quality_distribution[QualityLevel.FAIR]}")

        # A failed audit rule is a failed analysis.
        error_rules = sum(
            1 for results in audit_results.values()
            for result in results
            if result.status == 'ERROR'
        )
        if error_rules:
            logger.error(f"Comprehensive analysis failed: {error_rules} rule check(s) reported ERROR")
            return 1
        return 0

    except (OSError, KeyError, ValueError) as e:
        # CLI boundary: a file error or a parameter lookup or validation error ends the
        # command with a non-zero exit code; any other error propagates.
        logger.error(f"Comprehensive analysis failed: {e}")
        return 1


def run_check_command(args) -> int:
    """Run parameter check command."""
    try:
        parameter_key = args.parameter_key

        # Get parameter metadata
        try:
            param_value = get_global_manager().get(
                parameter_key,
                required=True,
            )
        except (KeyError, ValueError):
            _emit(f"Error: Parameter '{parameter_key}' not found")
            _emit("Available parameters:")
            available = _list_isotope_parameters()
            for param_name in sorted(list(available.keys())[:20]):  # Show first 20
                _emit(f"  - {param_name}")
            return 1

        # The declared metadata of the entry (units, dimension, dtype, bounds, ...).
        auditor = create_parameter_auditor()
        param_metadata = auditor.parameter_metadata(parameter_key)

        # Run audit on single parameter
        audit_results = auditor.audit_parameter(parameter_key, param_metadata)

        # Run quality assessment
        quality_metrics = assess_parameter_quality(parameter_key, param_metadata)

        # Display results
        _emit(f"\nParameter Check: {parameter_key}")
        _emit(f"Value: {param_value}")

        if args.verbose:
            _emit(f"\nMetadata: {param_metadata}")

        _emit("\nAudit Results:")
        for result in audit_results:
            status_icon = {'PASS': '[OK]', 'WARNING': '[WARN]', 'ERROR': '[ERROR]'}.get(result.status, '?')
            _emit(f"  {status_icon} {result.rule_name}: {result.status}")
            if result.status != 'PASS' or args.verbose:
                _emit(f"    {result.message}")

        _emit("\nQuality Assessment:")
        _emit(f"  Completeness: {quality_metrics.completeness_score:.2%}")
        _emit(f"  Scientific Confidence: {quality_metrics.scientific_confidence}")
        _emit(f"  Validation Status: {quality_metrics.validation_status}")
        _emit(f"  Quality Level: {quality_metrics.quality_level}")

        # A failed audit rule is a failed check.
        if any(result.status == 'ERROR' for result in audit_results):
            return 1
        return 0

    except (OSError, KeyError, ValueError) as e:
        # CLI boundary: a file error or a parameter lookup or validation error ends the
        # command with a non-zero exit code; any other error propagates.
        logger.error(f"Parameter check failed: {e}")
        return 1


def main() -> int:
    """Main CLI entry point."""
    # Logging is configured by this command-line entry point only; importing the
    # module (FUNCTOR_REGISTRY imports every scientific module) leaves the
    # process root logger untouched.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    parser = setup_argument_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Import QualityLevel for the quality commands

    # Route to appropriate command handler
    if args.command == 'audit':
        return run_audit_command(args)
    elif args.command == 'quality':
        return run_quality_command(args)
    elif args.command == 'full':
        return run_full_command(args)
    elif args.command == 'check':
        return run_check_command(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
