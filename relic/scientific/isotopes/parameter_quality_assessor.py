"""
Isotope Parameter Quality Assessor

This module provides comprehensive quality assessment functionality for isotope
tracking parameters. It evaluates parameter completeness, scientific rigor,
validation status, and overall quality levels.

Key Features:
- Parameter completeness scoring
- Scientific confidence assessment
- Quality level classification
- Validation status tracking
- Quality improvement recommendations
- Automated quality reporting
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json

from .parameter_auditor import (
    AuditResult,
    ParameterQualityMetrics,
    bounds_declaration_check,
    dtype_representation_check,
    unit_dimension_check,
)

logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """Parameter quality levels."""
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    FAIR = "FAIR"
    POOR = "POOR"


class ValidationStatus(Enum):
    """Parameter validation statuses."""
    VALIDATED = "VALIDATED"
    PENDING_VALIDATION = "PENDING_VALIDATION"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    DEPRECATED = "DEPRECATED"


@dataclass
class QualityRecommendation:
    """Quality improvement recommendation."""
    parameter_key: str
    recommendation_type: str  # 'add_metadata', 'improve_references', 'fix_constraints', etc.
    priority: str  # 'HIGH', 'MEDIUM', 'LOW'
    description: str
    specific_actions: List[str]
    expected_impact: str


@dataclass
class QualitySummary:
    """Summary of parameter quality assessment."""
    total_parameters: int
    quality_distribution: Dict[QualityLevel, int]
    validation_distribution: Dict[ValidationStatus, int]
    average_completeness: float
    confidence_distribution: Dict[str, int]
    critical_issues: List[str]
    improvement_recommendations: List[QualityRecommendation]
    assessment_date: str


class IsotopeParameterQualityAssessor:
    """
    Comprehensive quality assessor for isotope tracking parameters.

    Evaluates multiple dimensions of parameter quality and provides actionable
    recommendations for improvement.
    """

    def __init__(self):
        """Initialize the quality assessor."""
        self.logger = logging.getLogger(f"{__name__}.IsotopeParameterQualityAssessor")

        # Quality assessment criteria
        self.completeness_criteria = self._define_completeness_criteria()
        self.scientific_rigor_criteria = self._define_scientific_rigor_criteria()
        self.validation_criteria = self._define_validation_criteria()

        # Quality thresholds
        self.quality_thresholds = self._define_quality_thresholds()

        self.logger.info("Initialized isotope parameter quality assessor")

    def _define_completeness_criteria(self) -> Dict[str, float]:
        """Define completeness assessment criteria."""
        return {
            'required_fields': {
                'value': 1.0,
                'units': 1.0,
                'description': 1.0,
                'dtype': 0.8,
                'bounds': 0.8
            },
            'recommended_fields': {
                'references': 0.6,
                'dimension': 0.4,
                'category': 0.3,
                'tags': 0.2,
                'provenance': 0.3
            },
            'optional_fields': {
                'uncertainty': 0.2,
                'variability': 0.1,
                'spatial_scale': 0.1,
                'temporal_scale': 0.1
            }
        }

    def _define_scientific_rigor_criteria(self) -> Dict[str, Dict]:
        """Define scientific rigor assessment criteria."""
        return {
            'references': {
                'doi_count_weight': 0.4,
                'peer_reviewed_weight': 0.3,
                'recent_publications_weight': 0.2,
                'authoritative_sources_weight': 0.1
            },
            'parameter_traceability': {
                'provenance_documented': 0.3,
                'methodology_described': 0.3,
                'uncertainty_quantified': 0.2,
                'validation_performed': 0.2
            },
            'physical_constraints': {
                'bounds_physically_meaningful': 0.4,
                'units_consistent': 0.3,
                'dimensional_analysis': 0.3
            }
        }

    def _define_validation_criteria(self) -> Dict[str, Any]:
        """Define validation assessment criteria."""
        return {
            'validation_status_levels': {
                'VALIDATED': 1.0,
                'PENDING_VALIDATION': 0.6,
                'NEEDS_REVIEW': 0.3,
                'DEPRECATED': 0.0
            },
            'validation_methods': {
                'literature_comparison': 0.3,
                'experimental_data': 0.4,
                'expert_review': 0.2,
                'automated_testing': 0.1
            }
        }

    def _define_quality_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Define quality level thresholds."""
        return {
            'completeness': {
                'EXCELLENT': 0.9,
                'GOOD': 0.7,
                'FAIR': 0.5,
                'POOR': 0.0
            },
            'scientific_confidence': {
                'EXCELLENT': 0.85,
                'GOOD': 0.65,
                'FAIR': 0.45,
                'POOR': 0.0
            },
            'overall_quality': {
                'EXCELLENT': 0.85,
                'GOOD': 0.65,
                'FAIR': 0.45,
                'POOR': 0.0
            }
        }

    def assess_parameter_quality(self, param_key: str, metadata: Dict[str, Any],
                               audit_results: Optional[List[AuditResult]] = None) -> ParameterQualityMetrics:
        """
        Assess comprehensive quality of a single parameter.

        Parameters:
            param_key : str
                Parameter unique identifier
            metadata : Dict[str, Any]
                Parameter metadata
            audit_results : Optional[List[AuditResult]]
                Audit results for the parameter

        Returns:
            ParameterQualityMetrics: Comprehensive quality assessment
        """
        self.logger.debug(f"Assessing quality for parameter: {param_key}")

        # Assess completeness
        completeness_score = self._assess_completeness(metadata)

        # Assess scientific confidence
        scientific_confidence = self._assess_scientific_confidence(metadata)

        # Determine validation status
        validation_status = self._determine_validation_status(param_key, metadata, audit_results)

        # Calculate overall quality level
        quality_level = self._determine_quality_level(completeness_score, scientific_confidence, validation_status)

        return ParameterQualityMetrics(
            completeness_score=completeness_score,
            scientific_confidence=scientific_confidence,
            validation_status=validation_status.value,
            last_updated=datetime.now().isoformat(),
            quality_level=quality_level.value
        )

    def _assess_completeness(self, metadata: Dict[str, Any]) -> float:
        """Assess parameter completeness score.

        A field counts when it is present and not None; bounds: null is the complete
        declaration of a non-numeric parameter and is scored by the canonical rule."""
        total_score = 0.0
        max_score = 0.0

        for group in ('required_fields', 'recommended_fields', 'optional_fields'):
            for field, weight in self.completeness_criteria[group].items():
                max_score += weight
                present = field in metadata and (
                    metadata[field] is not None or field == 'bounds'
                )
                if present:
                    total_score += weight * self._assess_field_quality(field, metadata)

        return total_score / max_score if max_score > 0 else 0.0

    def _assess_field_quality(self, field: str, metadata: Dict[str, Any]) -> float:
        """Assess the quality of one field of a parameter's metadata.

        The value, units and bounds are scored with the checks of the auditor: the
        representation limits of the declared dtype, the SI dimensionality derivation of
        the unit against the declared dimension, and the canonical bounds rule (PASS
        1.0, WARNING 0.5, ERROR 0.0; a unit is 0.0 only when it fails)."""
        value = metadata.get(field)
        if field == 'value':
            status = dtype_representation_check(value, metadata.get('dtype'))[0]
            return {'PASS': 1.0, 'WARNING': 0.5, 'ERROR': 0.0}[status]

        elif field == 'units':
            status = unit_dimension_check(value, metadata.get('dimension'))[0]
            return 0.0 if status == 'ERROR' else 1.0

        elif field == 'description':
            # Check description length and quality
            if isinstance(value, str):
                if len(value) < 10:
                    return 0.3  # Too short
                elif len(value) < 50:
                    return 0.7  # Moderate
                else:
                    return 1.0  # Good description
            return 0.0

        elif field == 'bounds':
            status = bounds_declaration_check(value, metadata.get('dtype'))[0]
            return 1.0 if status == 'PASS' else 0.0

        elif field == 'references':
            # Check reference quality
            if isinstance(value, list):
                if not value:
                    return 0.0
                doi_count = sum(1 for ref in value if isinstance(ref, str) and ref.startswith('doi:'))
                return min(1.0, 0.5 + doi_count * 0.5)
            return 0.0

        # Default: if field exists, give basic credit
        return 0.8 if value is not None else 0.0

    def _assess_scientific_confidence(self, metadata: Dict[str, Any]) -> str:
        """Assess scientific confidence level."""
        confidence_score = 0.0

        # References assessment
        if 'references' in metadata:
            references = metadata['references']
            if isinstance(references, list) and references:
                doi_count = sum(1 for ref in references
                              if isinstance(ref, str) and ref.startswith('doi:'))
                peer_reviewed_count = sum(1 for ref in references
                                         if isinstance(ref, str) and
                                         any(indicator in ref.lower()
                                             for indicator in ['journal', 'nature', 'science']))

                confidence_score += doi_count * 0.4
                confidence_score += peer_reviewed_count * 0.3
                confidence_score += min(len(references) - doi_count - peer_reviewed_count, 2) * 0.1

        # Provenance assessment
        if 'provenance' in metadata and metadata['provenance']:
            confidence_score += 0.2

        # Uncertainty assessment
        if 'uncertainty' in metadata and metadata['uncertainty']:
            confidence_score += 0.1

        # Determine confidence level
        if confidence_score >= 0.8:
            return "HIGH"
        elif confidence_score >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"

    def _determine_validation_status(self, param_key: str, metadata: Dict[str, Any],
                                   audit_results: Optional[List[AuditResult]]) -> ValidationStatus:
        """Determine parameter validation status."""
        if audit_results is None:
            return ValidationStatus.PENDING_VALIDATION

        # Check for errors in audit results
        error_count = sum(1 for result in audit_results if result.status == 'ERROR')
        warning_count = sum(1 for result in audit_results if result.status == 'WARNING')

        if error_count > 0:
            return ValidationStatus.NEEDS_REVIEW
        elif warning_count > 2:
            return ValidationStatus.NEEDS_REVIEW
        elif warning_count > 0:
            return ValidationStatus.PENDING_VALIDATION
        else:
            return ValidationStatus.VALIDATED

    def _determine_quality_level(self, completeness: float, confidence: str, validation: ValidationStatus) -> QualityLevel:
        """Determine overall quality level."""
        # Convert confidence to numeric
        confidence_scores = {'HIGH': 0.9, 'MEDIUM': 0.6, 'LOW': 0.3}
        confidence_score = confidence_scores.get(confidence, 0.0)

        # Convert validation to numeric
        validation_scores = {
            ValidationStatus.VALIDATED: 1.0,
            ValidationStatus.PENDING_VALIDATION: 0.7,
            ValidationStatus.NEEDS_REVIEW: 0.3,
            ValidationStatus.DEPRECATED: 0.0
        }
        validation_score = validation_scores.get(validation, 0.0)

        # Calculate overall quality score
        overall_score = (completeness * 0.4 + confidence_score * 0.4 + validation_score * 0.2)

        # Determine quality level
        thresholds = self.quality_thresholds['overall_quality']
        if overall_score >= thresholds['EXCELLENT']:
            return QualityLevel.EXCELLENT
        elif overall_score >= thresholds['GOOD']:
            return QualityLevel.GOOD
        elif overall_score >= thresholds['FAIR']:
            return QualityLevel.FAIR
        else:
            return QualityLevel.POOR

    def assess_all_parameters(self, parameters_metadata: Dict[str, Dict[str, Any]],
                            audit_results: Optional[Dict[str, List[AuditResult]]] = None) -> Dict[str, ParameterQualityMetrics]:
        """
        Assess quality of all parameters.

        Parameters:
            parameters_metadata : Dict[str, Dict[str, Any]]
                Metadata for all parameters
            audit_results : Optional[Dict[str, List[AuditResult]]]
                Audit results for all parameters

        Returns:
            Dict[str, ParameterQualityMetrics]: Quality metrics by parameter
        """
        self.logger.info(f"Starting quality assessment of {len(parameters_metadata)} parameters")

        quality_metrics = {}

        for param_key, metadata in parameters_metadata.items():
            param_audit_results = audit_results.get(param_key) if audit_results else None
            quality_metrics[param_key] = self.assess_parameter_quality(param_key, metadata, param_audit_results)

        self.logger.info("Completed quality assessment")
        return quality_metrics

    def generate_quality_summary(self, quality_metrics: Dict[str, ParameterQualityMetrics]) -> QualitySummary:
        """
        Generate comprehensive quality summary.

        Parameters:
            quality_metrics : Dict[str, ParameterQualityMetrics]
                Quality metrics for all parameters

        Returns:
            QualitySummary: Comprehensive quality summary
        """
        total_parameters = len(quality_metrics)

        # Quality distribution
        quality_distribution = {}
        for level in QualityLevel:
            quality_distribution[level] = sum(1 for metrics in quality_metrics.values()
                                             if metrics.quality_level == level.value)

        # Validation distribution
        validation_distribution = {}
        for status in ValidationStatus:
            validation_distribution[status] = sum(1 for metrics in quality_metrics.values()
                                                if metrics.validation_status == status.value)

        # Average completeness
        average_completeness = sum(metrics.completeness_score for metrics in quality_metrics.values()) / total_parameters

        # Confidence distribution
        confidence_distribution = {}
        for metrics in quality_metrics.values():
            confidence_distribution[metrics.scientific_confidence] = confidence_distribution.get(metrics.scientific_confidence, 0) + 1

        # Identify critical issues
        critical_issues = self._identify_critical_issues(quality_metrics)

        # Generate recommendations
        recommendations = self._generate_recommendations(quality_metrics)

        return QualitySummary(
            total_parameters=total_parameters,
            quality_distribution=quality_distribution,
            validation_distribution=validation_distribution,
            average_completeness=average_completeness,
            confidence_distribution=confidence_distribution,
            critical_issues=critical_issues,
            improvement_recommendations=recommendations,
            assessment_date=datetime.now().isoformat()
        )

    def _identify_critical_issues(self, quality_metrics: Dict[str, ParameterQualityMetrics]) -> List[str]:
        """Identify critical quality issues."""
        issues = []

        # Parameters with poor quality
        poor_quality_params = [key for key, metrics in quality_metrics.items()
                             if metrics.quality_level == 'POOR']
        if poor_quality_params:
            issues.append(f"Parameters with poor quality: {len(poor_quality_params)}")

        # Parameters needing review
        needs_review_params = [key for key, metrics in quality_metrics.items()
                              if metrics.validation_status == 'NEEDS_REVIEW']
        if needs_review_params:
            issues.append(f"Parameters needing review: {len(needs_review_params)}")

        # Low completeness parameters
        low_completeness_params = [key for key, metrics in quality_metrics.items()
                                  if metrics.completeness_score < 0.5]
        if low_completeness_params:
            issues.append(f"Parameters with low completeness: {len(low_completeness_params)}")

        # Low confidence parameters
        low_confidence_params = [key for key, metrics in quality_metrics.items()
                                if metrics.scientific_confidence == 'LOW']
        if low_confidence_params:
            issues.append(f"Parameters with low scientific confidence: {len(low_confidence_params)}")

        return issues

    def _generate_recommendations(self, quality_metrics: Dict[str, ParameterQualityMetrics]) -> List[QualityRecommendation]:
        """Generate quality improvement recommendations."""
        recommendations = []

        for param_key, metrics in quality_metrics.items():
            if metrics.quality_level in ['POOR', 'FAIR']:
                param_recommendations = self._generate_parameter_recommendations(param_key, metrics)
                recommendations.extend(param_recommendations)

        # Sort by priority
        recommendations.sort(key=lambda x: {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}[x.priority])

        return recommendations

    def _generate_parameter_recommendations(self, param_key: str, metrics: ParameterQualityMetrics) -> List[QualityRecommendation]:
        """Generate recommendations for a specific parameter."""
        recommendations = []

        # Completeness recommendations
        if metrics.completeness_score < 0.5:
            recommendations.append(QualityRecommendation(
                parameter_key=param_key,
                recommendation_type='improve_completeness',
                priority='HIGH',
                description='Parameter has low completeness score',
                specific_actions=[
                    'Add missing required fields (units, description, bounds)',
                    'Include recommended fields (references, dimension)',
                    'Document uncertainty and provenance information'
                ],
                expected_impact='Significant improvement in overall quality'
            ))

        # Scientific confidence recommendations
        if metrics.scientific_confidence == 'LOW':
            recommendations.append(QualityRecommendation(
                parameter_key=param_key,
                recommendation_type='improve_scientific_rigor',
                priority='HIGH',
                description='Parameter has low scientific confidence',
                specific_actions=[
                    'Add peer-reviewed references with DOIs',
                    'Document measurement methodology',
                    'Include uncertainty quantification',
                    'Provide provenance information'
                ],
                expected_impact='Major improvement in scientific reliability'
            ))

        # Validation recommendations
        if metrics.validation_status == 'NEEDS_REVIEW':
            recommendations.append(QualityRecommendation(
                parameter_key=param_key,
                recommendation_type='address_validation_issues',
                priority='MEDIUM',
                description='Parameter has validation issues',
                specific_actions=[
                    'Review and fix audit failures',
                    'Verify physical constraints',
                    'Check numerical stability',
                    'Validate against literature values'
                ],
                expected_impact='Improved parameter reliability'
            ))

        return recommendations

    def generate_quality_report(self, quality_metrics: Dict[str, ParameterQualityMetrics],
                              output_format: str = 'markdown') -> str:
        """
        Generate comprehensive quality report.

        Parameters:
            quality_metrics : Dict[str, ParameterQualityMetrics]
                Quality metrics for all parameters
            output_format : str
                Output format ('markdown', 'html', 'json')

        Returns:
            str: Formatted quality report
        """
        summary = self.generate_quality_summary(quality_metrics)

        if output_format == 'json':
            return self._generate_json_report(quality_metrics, summary)
        elif output_format == 'html':
            return self._generate_html_report(quality_metrics, summary)
        else:
            return self._generate_markdown_report(quality_metrics, summary)

    def _generate_markdown_report(self, quality_metrics: Dict[str, ParameterQualityMetrics],
                                 summary: QualitySummary) -> str:
        """Generate markdown quality report."""
        lines = []

        # Header
        lines.append("# RELIC Isotope Parameter Quality Report\n")
        lines.append(f"Generated at: {summary.assessment_date}\n")

        # Executive summary
        lines.append("## Executive Summary\n")
        lines.append(f"- **Total parameters**: {summary.total_parameters}")
        lines.append(f"- **Average completeness**: {summary.average_completeness:.2%}")
        lines.append(f"- **Excellent parameters**: {summary.quality_distribution[QualityLevel.EXCELLENT]} ({summary.quality_distribution[QualityLevel.EXCELLENT]/summary.total_parameters*100:.1f}%)")
        lines.append(f"- **Good parameters**: {summary.quality_distribution[QualityLevel.GOOD]} ({summary.quality_distribution[QualityLevel.GOOD]/summary.total_parameters*100:.1f}%)")
        lines.append(f"- **Fair parameters**: {summary.quality_distribution[QualityLevel.FAIR]} ({summary.quality_distribution[QualityLevel.FAIR]/summary.total_parameters*100:.1f}%)")
        lines.append(f"- **Poor parameters**: {summary.quality_distribution[QualityLevel.POOR]} ({summary.quality_distribution[QualityLevel.POOR]/summary.total_parameters*100:.1f}%)\n")

        # Quality distribution
        lines.append("## Quality Distribution\n")
        lines.append("### Quality level distribution")
        lines.append("| Quality level | Count | Percent |")
        lines.append("|---------|---------|--------|")
        for level in QualityLevel:
            count = summary.quality_distribution[level]
            percentage = count / summary.total_parameters * 100
            lines.append(f"| {level.value} | {count} | {percentage:.1f}% |")
        lines.append("")

        # Validation status
        lines.append("### Validation status distribution")
        lines.append("| Validation status | Count | Percent |")
        lines.append("|---------|---------|--------|")
        for status in ValidationStatus:
            count = summary.validation_distribution[status]
            percentage = count / summary.total_parameters * 100
            lines.append(f"| {status.value} | {count} | {percentage:.1f}% |")
        lines.append("")

        # Scientific confidence
        lines.append("### Scientific confidence distribution")
        lines.append("| Confidence | Count | Percent |")
        lines.append("|-------|---------|--------|")
        for confidence, count in summary.confidence_distribution.items():
            percentage = count / summary.total_parameters * 100
            lines.append(f"| {confidence} | {count} | {percentage:.1f}% |")
        lines.append("")

        # Critical issues
        if summary.critical_issues:
            lines.append("## Critical issues\n")
            for issue in summary.critical_issues:
                lines.append(f"- **{issue}**")
            lines.append("")

        # Recommendations
        if summary.improvement_recommendations:
            lines.append("## Improvement recommendations\n")
            lines.append("### High priority recommendations\n")
            high_priority = [r for r in summary.improvement_recommendations if r.priority == 'HIGH']
            if high_priority:
                for rec in high_priority[:10]:  # Limit to top 10
                    lines.append(f"#### {rec.parameter_key}")
                    lines.append(f"**Issue**: {rec.description}")
                    lines.append("**Actions**:")
                    for action in rec.specific_actions:
                        lines.append(f"- {action}")
                    lines.append(f"**Expected impact**: {rec.expected_impact}")
                    lines.append("")
            else:
                lines.append("No high priority recommendations\n")

            lines.append("### Medium priority recommendations\n")
            medium_priority = [r for r in summary.improvement_recommendations if r.priority == 'MEDIUM']
            if medium_priority:
                for rec in medium_priority[:5]:  # Limit to top 5
                    lines.append(f"#### {rec.parameter_key}")
                    lines.append(f"**Issue**: {rec.description}")
                    lines.append(f"**Recommendation**: {rec.expected_impact}")
                    lines.append("")
            else:
                lines.append("No medium priority recommendations\n")

        # Detailed parameter assessment
        lines.append("## Detailed parameter assessment\n")
        lines.append("### Poor quality parameters")
        poor_params = {k: v for k, v in quality_metrics.items() if v.quality_level == 'POOR'}
        if poor_params:
            lines.append("| Parameter | Completeness | Scientific confidence | Validation status |")
            lines.append("|------|--------|-----------|----------|")
            for param_key, metrics in poor_params.items():
                lines.append(f"| {param_key} | {metrics.completeness_score:.2%} | {metrics.scientific_confidence} | {metrics.validation_status} |")
            lines.append("")
        else:
            lines.append("All parameters meet fair quality or better\n")

        return "\n".join(lines)

    def _generate_json_report(self, quality_metrics: Dict[str, ParameterQualityMetrics],
                            summary: QualitySummary) -> str:
        """Generate JSON quality report."""
        report_data = {
            'summary': {
                'total_parameters': summary.total_parameters,
                'quality_distribution': {k.value: v for k, v in summary.quality_distribution.items()},
                'validation_distribution': {k.value: v for k, v in summary.validation_distribution.items()},
                'average_completeness': summary.average_completeness,
                'confidence_distribution': summary.confidence_distribution,
                'critical_issues': summary.critical_issues,
                'assessment_date': summary.assessment_date
            },
            'recommendations': [
                {
                    'parameter_key': rec.parameter_key,
                    'type': rec.recommendation_type,
                    'priority': rec.priority,
                    'description': rec.description,
                    'specific_actions': rec.specific_actions,
                    'expected_impact': rec.expected_impact
                }
                for rec in summary.improvement_recommendations
            ],
            'parameters': {
                param_key: {
                    'completeness_score': metrics.completeness_score,
                    'scientific_confidence': metrics.scientific_confidence,
                    'validation_status': metrics.validation_status,
                    'quality_level': metrics.quality_level,
                    'last_updated': metrics.last_updated
                }
                for param_key, metrics in quality_metrics.items()
            }
        }

        return json.dumps(report_data, indent=2, ensure_ascii=True)

    def _generate_html_report(self, quality_metrics: Dict[str, ParameterQualityMetrics],
                            summary: QualitySummary) -> str:
        """Generate HTML quality report with inline styles."""
        lines = []
        lines.append("<!DOCTYPE html>")
        lines.append("<html lang='en'><head><meta charset='utf-8'>")
        lines.append("<title>Isotope Parameter Quality Report</title>")
        lines.append("<style>")
        lines.append("body{font-family:Arial,sans-serif;margin:2em;color:#333}")
        lines.append("h1{color:#1a5276;border-bottom:2px solid #2980b9;padding-bottom:8px}")
        lines.append("h2{color:#2471a3;margin-top:1.5em}h3{color:#2e86c1}")
        lines.append("table{border-collapse:collapse;width:100%;margin:1em 0}")
        lines.append("th,td{border:1px solid #d5dbdb;padding:8px;text-align:left}")
        lines.append("th{background:#2980b9;color:#fff}")
        lines.append("tr:nth-child(even){background:#f2f4f4}")
        lines.append(".tag-good{color:#27ae60;font-weight:bold}")
        lines.append(".tag-fair{color:#f39c12;font-weight:bold}")
        lines.append(".tag-poor{color:#e74c3c;font-weight:bold}")
        lines.append(".issue{background:#fdedec;padding:6px 10px;border-left:4px solid #e74c3c;margin:4px 0}")
        lines.append("</style></head><body>")

        # Header
        lines.append(f"<h1>Isotope Parameter Quality Report</h1>")
        lines.append(f"<p>Assessment date: <b>{summary.assessment_date}</b> "
                     f"| Total parameters: <b>{summary.total_parameters}</b></p>")

        # Summary statistics
        lines.append("<h2>Summary Statistics</h2>")
        lines.append(f"<p>Average completeness: <b>{summary.average_completeness:.1%}</b></p>")

        # Quality distribution table
        lines.append("<h3>Quality Distribution</h3>")
        lines.append("<table><tr><th>Quality Level</th><th>Count</th><th>Percent</th></tr>")
        for level in QualityLevel:
            count = summary.quality_distribution.get(level, 0)
            pct = count / max(summary.total_parameters, 1) * 100
            css = ("tag-good" if level in (QualityLevel.EXCELLENT, QualityLevel.GOOD)
                   else "tag-fair" if level == QualityLevel.FAIR else "tag-poor")
            lines.append(f"<tr><td class='{css}'>{level.value}</td>"
                         f"<td>{count}</td><td>{pct:.1f}%</td></tr>")
        lines.append("</table>")

        # Validation distribution table
        lines.append("<h3>Validation Status Distribution</h3>")
        lines.append("<table><tr><th>Status</th><th>Count</th><th>Percent</th></tr>")
        for status in ValidationStatus:
            count = summary.validation_distribution.get(status, 0)
            pct = count / max(summary.total_parameters, 1) * 100
            lines.append(f"<tr><td>{status.value}</td><td>{count}</td><td>{pct:.1f}%</td></tr>")
        lines.append("</table>")

        # Critical issues
        if summary.critical_issues:
            lines.append("<h2>Critical Issues</h2>")
            for issue in summary.critical_issues:
                lines.append(f"<div class='issue'>{issue}</div>")

        # Top recommendations
        if summary.improvement_recommendations:
            lines.append("<h2>Improvement Recommendations</h2>")
            high = [r for r in summary.improvement_recommendations if r.priority == 'HIGH']
            if high:
                lines.append("<h3>High Priority</h3>")
                lines.append("<table><tr><th>Parameter</th><th>Issue</th>"
                             "<th>Expected Impact</th></tr>")
                for rec in high[:10]:
                    lines.append(f"<tr><td><b>{rec.parameter_key}</b></td>"
                                 f"<td>{rec.description}</td>"
                                 f"<td>{rec.expected_impact}</td></tr>")
                lines.append("</table>")

        # Detailed parameter table
        lines.append("<h2>Parameter Details</h2>")
        lines.append("<table><tr><th>Parameter</th><th>Completeness</th>"
                     "<th>Confidence</th><th>Quality</th><th>Validation</th></tr>")
        for param_key, m in sorted(quality_metrics.items()):
            css = ("tag-good" if m.quality_level in ('EXCELLENT', 'GOOD')
                   else "tag-fair" if m.quality_level == 'FAIR' else "tag-poor")
            lines.append(f"<tr><td>{param_key}</td>"
                         f"<td>{m.completeness_score:.0%}</td>"
                         f"<td>{m.scientific_confidence}</td>"
                         f"<td class='{css}'>{m.quality_level}</td>"
                         f"<td>{m.validation_status}</td></tr>")
        lines.append("</table>")

        lines.append("</body></html>")
        return "\n".join(lines)

    def save_quality_report(self, quality_metrics: Dict[str, ParameterQualityMetrics],
                          output_path: Optional[str] = None,
                          output_format: str = 'markdown') -> str:
        """
        Save quality report to file.

        Parameters:
            quality_metrics : Dict[str, ParameterQualityMetrics]
                Quality metrics for all parameters
            output_path : Optional[str]
                Output file path
            output_format : str
                Output format

        Returns:
            str: Path to saved report file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            extension = '.md' if output_format == 'markdown' else '.json'
            output_path = f"isotope_parameter_quality_{timestamp}{extension}"

        report_content = self.generate_quality_report(quality_metrics, output_format)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)

            self.logger.info(f"Quality report saved to: {output_path}")
            return output_path

        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            self.logger.error(f"Failed to save quality report: {e}")
            raise


# Convenience functions
def create_quality_assessor() -> IsotopeParameterQualityAssessor:
    """Create and return a quality assessor instance."""
    return IsotopeParameterQualityAssessor()


def assess_parameter_quality(param_key: str, metadata: Dict[str, Any]) -> ParameterQualityMetrics:
    """Assess quality of a single parameter."""
    assessor = create_quality_assessor()
    return assessor.assess_parameter_quality(param_key, metadata)


def generate_quality_report(quality_metrics: Dict[str, ParameterQualityMetrics]) -> str:
    """Generate quality report from metrics."""
    assessor = create_quality_assessor()
    return assessor.generate_quality_report(quality_metrics)


# Export public interface
__all__ = [
    'IsotopeParameterQualityAssessor',
    'QualityLevel',
    'ValidationStatus',
    'QualityRecommendation',
    'QualitySummary',
    'ParameterQualityMetrics',
    'create_quality_assessor',
    'assess_parameter_quality',
    'generate_quality_report'
]