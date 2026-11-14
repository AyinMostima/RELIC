"""
Validation Report Generators

This module provides multi-format output capabilities for validation reports,
supporting JSON, YAML, Markdown, NetCDF metadata, and CLI summary formats.

Key Features:
    - CF-1.8 compliant NetCDF metadata generation
    - Markdown reports with statistical summaries
    - JSON/YAML structured output
    - CLI-friendly summary output
    - Customizable report templates
    - Integration with data standards

Mathematical Framework:
    Report R transforms ValidationReportModel V to format F:
    R(V, F) -> formatted_output where F in {JSON, YAML, MD, NetCDF, CLI}
    
    For CF-1.8 NetCDF: 
    NetCDF_attrs = CF_transform(validation_metadata, statistical_results)
    
References:
    - CF-1.8 Conventions: https://cfconventions.org/
    - NetCDF metadata standards
    - Markdown specification
"""

from typing import Dict, List, Any, Union
import json
import logging
from datetime import datetime
from pathlib import Path
import csv

import yaml

from .models import ValidationReportModel, ValidationResultModel, ValidationLevel
from ...data.standards.cf_conventions import CFConventions

HAS_YAML = True

logger = logging.getLogger(__name__)


class ValidationReporter:
    """
    Multi-format validation report generator.
    
    Provides methods to export validation reports in various formats
    suitable for different use cases: machine processing (JSON/YAML),
    human reading (Markdown), scientific data (NetCDF), and command line (CLI).
    
    The reporter summarizes bounded CF variable metadata without dataset claims
    and supports customizable templates for different report styles.
    """
    
    def __init__(self, cf_conventions = None):
        """
        Initialize validation reporter.
        
        Parameters:
            cf_conventions: CF conventions handler (uses default if None)
        """
        self.cf_conventions = cf_conventions or CFConventions()
    
    def to_json(self, 
               report: ValidationReportModel,
               indent: int = 2,
               include_metadata: bool = True) -> str:
        """
        Export report to JSON format.
        
        Parameters:
            report: Validation report to export
            indent: JSON indentation level
            include_metadata: Include execution metadata
            
        Returns:
            JSON string representation
        """
        report_dict = report.to_dict()
        
        if not include_metadata:
            report_dict.pop("execution_metadata", None)
        
        return json.dumps(report_dict, indent=indent, default=str)
    
    def to_yaml(self, 
               report: ValidationReportModel,
               include_metadata: bool = True) -> str:
        """
        Export report to YAML format.
        
        Parameters:
            report: Validation report to export
            include_metadata: Include execution metadata
            
        Returns:
            YAML string representation
            
        Raises:
            ImportError: If PyYAML not available
        """
        if not HAS_YAML:
            raise ImportError("PyYAML is required for YAML output. Install with: pip install pyyaml")
        
        report_dict = report.to_dict()
        
        if not include_metadata:
            report_dict.pop("execution_metadata", None)
        
        return yaml.dump(report_dict, default_flow_style=False, sort_keys=False)

    def to_csv(self,
               report: ValidationReportModel,
               include_details: bool = False) -> str:
        """
        Export report to CSV format (results-level rows).

        Parameters:
            report: Validation report to export
            include_details: Include details count field

        Returns:
            CSV string with header and rows for each ValidationResult
        """
        # Prepare rows
        header = [
            "level", "code", "layer", "scope", "passed", "message", "tags", "metrics", "details_count"
        ]
        rows: List[List[str]] = []
        for r in report.results:
            tags = ",".join(r.tags) if r.tags else ""
            # Metric values are validated numbers; a value that cannot be
            # serialized raises instead of being exported as an empty mapping.
            metrics_str = json.dumps(r.metrics) if r.metrics else "{}"
            details_count = str(len(r.details)) if include_details else ""
            row = [
                r.level.value,
                r.code.value,
                r.layer,
                r.scope,
                "true" if r.passed else "false",
                r.message.replace("\n", " ").strip(),
                tags,
                metrics_str,
                details_count,
            ]
            rows.append(row)

        # Write to in-memory CSV string
        from io import StringIO
        sio = StringIO()
        writer = csv.writer(sio)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
        return sio.getvalue()
    
    def to_markdown(self, 
                   report: ValidationReportModel,
                   include_details: bool = True,
                   max_detail_length: int = 100) -> str:
        """
        Export report to Markdown format.
        
        Parameters:
            report: Validation report to export
            include_details: Include detailed diagnostic information
            max_detail_length: Maximum length for detail truncation
            
        Returns:
            Markdown string representation
        """
        md_lines = []
        
        # Header
        md_lines.extend([
            "# RELIC Validation Report",
            "",
            f"**Generated:** {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Results:** {len(report.results)}",
            ""
        ])
        
        # Executive Summary
        md_lines.extend([
            "## Executive Summary",
            ""
        ])
        
        if report.summary:
            by_level = report.summary.get("by_level", {})
            by_layer = report.summary.get("by_layer", {})
            
            # Results by level
            md_lines.extend([
                "### Results by Level",
                "",
                "| Level | Count |",
                "|-------|-------|"
            ])
            
            for level in ["CRITICAL", "ERROR", "WARNING", "INFO"]:
                count = by_level.get(level, 0)
                emoji = self._get_level_emoji(level)
                md_lines.append(f"| {emoji} {level} | {count} |")
            
            md_lines.extend(["", "### Results by Layer", "", "| Layer | Count |", "|-------|-------|"])
            
            for layer, count in by_layer.items():
                md_lines.append(f"| {layer} | {count} |")
        
        md_lines.append("")
        
        # Bounded CF variable-metadata summary
        if report.cf_summary:
            md_lines.extend([
                "### CF Variable-Metadata Subset",
                "",
                f"- **Total CF Validations:** {report.cf_summary.get('total_cf_validations', 0)}",
                f"- **CF Errors:** {report.cf_summary.get('cf_errors', 0)}",  
                f"- **CF Warnings:** {report.cf_summary.get('cf_warnings', 0)}",
                ""
            ])
        
        # Conservation Summary
        if report.conservation_summary:
            md_lines.extend([
                "### Conservation Laws",
                "",
                f"- **Conservation Checks:** {report.conservation_summary.get('total_conservation_checks', 0)}",
                f"- **Conservation Failures:** {report.conservation_summary.get('conservation_failures', 0)}",
                f"- **Conservation Warnings:** {report.conservation_summary.get('conservation_warnings', 0)}",
                ""
            ])
        
        # Detailed Results
        if include_details:
            md_lines.extend(["## Detailed Results", ""])
            
            # Group results by level for better organization
            for level in [ValidationLevel.CRITICAL, ValidationLevel.ERROR, 
                         ValidationLevel.WARNING, ValidationLevel.INFO]:
                level_results = report.get_results_by_level(level)
                if level_results:
                    emoji = self._get_level_emoji(level.value)
                    md_lines.extend([f"### {emoji} {level.value} ({len(level_results)} items)", ""])
                    
                    for result in level_results:
                        md_lines.extend(self._format_result_markdown(result, max_detail_length))
        
        # Performance Metrics
        if report.execution_metadata:
            md_lines.extend([
                "## Performance Metrics",
                "",
                f"- **Execution Time:** {report.execution_metadata.get('execution_time_seconds', 0):.2f} seconds",
                f"- **Rules Executed:** {report.execution_metadata.get('total_rules_executed', 0)}",
                f"- **Rules Passed:** {report.execution_metadata.get('rules_passed', 0)}",
                f"- **Rules Failed:** {report.execution_metadata.get('rules_failed', 0)}",
                ""
            ])
            
            # Layer execution times
            layer_times = report.execution_metadata.get('layer_execution_times', {})
            if layer_times:
                md_lines.extend(["### Layer Execution Times", "", "| Layer | Time (s) |", "|-------|----------|"])
                for layer, time_sec in layer_times.items():
                    md_lines.append(f"| {layer} | {time_sec:.2f} |")
                md_lines.append("")
        
        return "\n".join(md_lines)
    
    def to_netcdf_dict(self, 
                      report: ValidationReportModel,
                      variable_prefix: str = "validation_") -> Dict[str, Any]:
        """
        Export report to NetCDF-consistent attribute dictionary.
        
        Parameters:
            report: Validation report to export
            variable_prefix: Prefix for validation variables
            
        Returns:
            Dictionary of NetCDF attributes and variables
        """
        netcdf_dict = {
            "global_attributes": {},
            "variables": {},
            "dimensions": {}
        }
        
        # Global attributes (CF-1.8 compliant)
        global_attrs = netcdf_dict["global_attributes"]
        global_attrs.update({
            "Conventions": "CF-1.8",
            "title": "RELIC Validation Results",
            "institution": "Yuchong Long",
            "source": "RELIC Unified Validation Framework",
            "history": f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "references": "Gneiting & Raftery (2007), CF-1.8 Conventions",
            "comment": f"Validation report containing {len(report.results)} results"
        })
        
        # Add validation summary as global attributes
        if report.summary:
            global_attrs.update({
                "validation_total_results": len(report.results),
                "validation_critical_count": report.summary.get("by_level", {}).get("CRITICAL", 0),
                "validation_error_count": report.summary.get("by_level", {}).get("ERROR", 0),
                "validation_warning_count": report.summary.get("by_level", {}).get("WARNING", 0),
                "validation_info_count": report.summary.get("by_level", {}).get("INFO", 0)
            })
        
        # Execution metadata
        if report.execution_metadata:
            global_attrs.update({
                "validation_execution_time_seconds": report.execution_metadata.get("execution_time_seconds", 0.0),
                "validation_rules_executed": report.execution_metadata.get("total_rules_executed", 0),
                "validation_timestamp": report.execution_metadata.get("start_time", "")
            })
        
        # Create validation result variables
        if report.results:
            # Dimensions
            n_results = len(report.results)
            netcdf_dict["dimensions"]["n_validation_results"] = n_results
            
            # Extract result data for variables
            levels = [r.level.value for r in report.results]
            codes = [r.code.value for r in report.results]
            layers = [r.layer for r in report.results]
            scopes = [r.scope for r in report.results]
            
            # Validation level variable
            netcdf_dict["variables"][f"{variable_prefix}level"] = {
                "dimensions": ("n_validation_results",),
                "data": levels,
                "attributes": {
                    "standard_name": "validation_severity_level",
                    "long_name": "Validation result severity level",
                    "units": "dimensionless",
                    "flag_values": "INFO WARNING ERROR CRITICAL",
                    "flag_meanings": "informational potential_issue definite_problem severe_issue"
                }
            }
            
            # Validation code variable
            netcdf_dict["variables"][f"{variable_prefix}code"] = {
                "dimensions": ("n_validation_results",),
                "data": codes,
                "attributes": {
                    "standard_name": "validation_result_code",
                    "long_name": "Standardized validation result code",
                    "units": "dimensionless"
                }
            }
            
            # Validation layer variable
            netcdf_dict["variables"][f"{variable_prefix}layer"] = {
                "dimensions": ("n_validation_results",),
                "data": layers,
                "attributes": {
                    "standard_name": "validation_layer",
                    "long_name": "Validation framework layer",
                    "units": "dimensionless",
                    "flag_values": "L1 L2 L3 L4", 
                    "flag_meanings": "unit_validation integration_validation benchmark_validation scientific_validation"
                }
            }
            
            # Validation scope variable
            netcdf_dict["variables"][f"{variable_prefix}scope"] = {
                "dimensions": ("n_validation_results",),
                "data": scopes,
                "attributes": {
                    "standard_name": "validation_scope",
                    "long_name": "Validation rule scope or module name",
                    "units": "dimensionless"
                }
            }
            
            # Add statistical metrics if available
            metric_names = set()
            for result in report.results:
                if result.metrics:
                    metric_names.update(result.metrics.keys())
            
            for metric_name in metric_names:
                metric_values = []
                for result in report.results:
                    value = result.metrics.get(metric_name, float('nan'))
                    metric_values.append(value)
                
                # Determine appropriate CF metadata based on metric name
                cf_attrs = self._get_metric_cf_attributes(metric_name)
                
                netcdf_dict["variables"][f"{variable_prefix}{metric_name}"] = {
                    "dimensions": ("n_validation_results",),
                    "data": metric_values,
                    "attributes": cf_attrs
                }
        
        return netcdf_dict
    
    def to_cli_summary(self, 
                      report: ValidationReportModel,
                      width: int = 80,
                      show_details: bool = False) -> str:
        """
        Export report to CLI-friendly summary format.
        
        Parameters:
            report: Validation report to export
            width: Terminal width for formatting
            show_details: Show detailed results
            
        Returns:
            CLI-formatted string
        """
        lines = []
        
        # Header
        separator = "=" * width
        lines.extend([
            separator,
            "RELIC VALIDATION SUMMARY".center(width),
            separator,
            ""
        ])
        
        # Quick stats
        if report.summary:
            by_level = report.summary.get("by_level", {})
            critical = by_level.get("CRITICAL", 0)
            errors = by_level.get("ERROR", 0)
            warnings = by_level.get("WARNING", 0)
            
            # Color-coded status
            if critical > 0:
                status = f"[CRITICAL] {critical} critical issues found"
            elif errors > 0:
                status = f"[ERROR] {errors} errors found"
            elif warnings > 0:
                status = f"[WARNING] {warnings} warnings found"
            else:
                status = "[PASS] All validations passed"
            
            lines.extend([
                f"Status: {status}",
                f"Total Results: {len(report.results)}",
                f"Critical: {critical}, Errors: {errors}, Warnings: {warnings}, Info: {by_level.get('INFO', 0)}",
                ""
            ])
        
        # Layer summary
        if report.summary and "by_layer" in report.summary:
            lines.extend(["Layer Summary:", "-" * 20])
            for layer, count in report.summary["by_layer"].items():
                lines.append(f"  {layer}: {count} results")
            lines.append("")
        
        # Conservation and CF summaries
        if report.conservation_summary:
            lines.extend([
                "Conservation Laws:",
                f"  Checks: {report.conservation_summary.get('total_conservation_checks', 0)}",
                f"  Failures: {report.conservation_summary.get('conservation_failures', 0)}",
                ""
            ])
        
        if report.cf_summary:
            lines.extend([
                "CF variable-metadata subset:",
                f"  Validations: {report.cf_summary.get('total_cf_validations', 0)}",
                f"  Errors: {report.cf_summary.get('cf_errors', 0)}",
                ""
            ])
        
        # Performance
        if report.execution_metadata:
            exec_time = report.execution_metadata.get("execution_time_seconds", 0)
            lines.extend([
                f"Execution Time: {exec_time:.2f}s",
                f"Rules Executed: {report.execution_metadata.get('total_rules_executed', 0)}",
                ""
            ])
        
        # Detailed results (if requested)
        if show_details:
            lines.extend(["-" * width, "DETAILED RESULTS", "-" * width, ""])
            
            for result in report.results:
                if result.level in [ValidationLevel.CRITICAL, ValidationLevel.ERROR]:
                    level_symbol = "" if result.level == ValidationLevel.ERROR else ""
                    lines.append(f"{level_symbol} [{result.level.value}] {result.message}")
                    lines.append(f"   Scope: {result.scope} | Layer: {result.layer}")
                    if result.details:
                        for detail in result.details[:2]:  # Limit to 2 details
                            if detail.variable_name:
                                lines.append(f"   - {detail.variable_name}: {detail.value}")
                    lines.append("")
        
        lines.append(separator)
        
        return "\n".join(lines)
    
    def save_report(self, 
                   report: ValidationReportModel,
                   file_path: Union[str, Path],
                   format: str = "auto",
                   **kwargs) -> None:
        """
        Save report to file in specified format.
        
        Parameters:
            report: Validation report to save
            file_path: Output file path
            format: Output format (json, yaml, markdown, auto)
            **kwargs: Additional format-specific arguments
        """
        path = Path(file_path)
        
        # Auto-detect format from extension
        if format == "auto":
            suffix = path.suffix.lower()
            if suffix == ".json":
                format = "json"
            elif suffix in [".yaml", ".yml"]:
                format = "yaml"
            elif suffix in [".md", ".markdown"]:
                format = "markdown"
            else:
                format = "json"  # Default
        
        # Generate content
        if format == "json":
            content = self.to_json(report, **kwargs)
        elif format == "yaml":
            content = self.to_yaml(report, **kwargs)
        elif format == "markdown":
            content = self.to_markdown(report, **kwargs)
        elif format == "csv":
            content = self.to_csv(report, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Write to file
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Validation report saved to {path} in {format} format")
    
    def _get_level_emoji(self, level: str) -> str:
        """Get ASCII label for validation level."""
        label_map = {
            "CRITICAL": "[CRITICAL]",
            "ERROR": "[ERROR]",
            "WARNING": "[WARNING]",
            "INFO": "[INFO]"
        }
        return label_map.get(level, "[NOTE]")
    
    def _format_result_markdown(self, 
                               result: ValidationResultModel, 
                               max_length: int) -> List[str]:
        """Format single validation result as Markdown."""
        lines = []
        
        # Main result
        emoji = self._get_level_emoji(result.level.value)
        lines.extend([
            f"#### {emoji} {result.message}",
            "",
            f"- **Code:** `{result.code.value}`",
            f"- **Layer:** {result.layer}",
            f"- **Scope:** {result.scope}"
        ])
        
        # Tags
        if result.tags:
            lines.append(f"- **Tags:** {', '.join(result.tags)}")
        
        # Details
        if result.details:
            lines.extend(["", "**Details:**"])
            for detail in result.details:
                detail_line = f"- {detail.variable_name or 'Variable'}"
                if detail.value is not None:
                    value_str = str(detail.value)
                    if len(value_str) > max_length:
                        value_str = value_str[:max_length] + "..."
                    detail_line += f": {value_str}"
                if detail.units:
                    detail_line += f" ({detail.units})"
                lines.append(detail_line)
        
        # Metrics
        if result.metrics:
            lines.extend(["", "**Metrics:**"])
            for metric, value in result.metrics.items():
                lines.append(f"- {metric}: {value}")
        
        lines.extend(["", "---", ""])
        return lines
    
    def _get_metric_cf_attributes(self, metric_name: str) -> Dict[str, Any]:
        """Get CF-1.8 compliant attributes for a metric."""
        # Default attributes
        attrs: Dict[str, Any] = {
            "units": "dimensionless",
            "long_name": metric_name.replace('_', ' ').title()
        }
        
        # Specific metric attributes
        if "crps" in metric_name.lower():
            attrs.update({
                "standard_name": "continuous_ranked_probability_score",
                "long_name": "Continuous Ranked Probability Score",
                "units": "same_as_observations",
                "references": "Matheson & Winkler (1976), Gneiting & Raftery (2007)"
            })
        elif "pit" in metric_name.lower():
            attrs.update({
                "standard_name": "probability_integral_transform",
                "long_name": "Probability Integral Transform statistic",
                "references": "Diebold et al. (1998)"
            })
        elif "pvalue" in metric_name:
            attrs.update({
                "standard_name": "statistical_significance_pvalue", 
                "long_name": "Statistical test p-value",
                "valid_range": [0.0, 1.0]
            })
        elif "residual" in metric_name:
            attrs.update({
                "standard_name": "validation_residual",
                "long_name": "Validation residual or difference"
            })
        elif "time" in metric_name:
            attrs.update({
                "units": "seconds",
                "long_name": "Execution time"
            })
        
        return attrs
