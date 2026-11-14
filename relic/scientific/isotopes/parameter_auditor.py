"""
Isotope Parameter Auditor

This module provides comprehensive parameter auditing functionality for the isotope
tracking system. It ensures parameter consistency, validates scientific references,
checks physical constraints, and maintains parameter quality standards.

Key Features:
- Comprehensive parameter validation against scientific standards
- Reference verification and completeness checking
- Declared-bounds validation (the YAML bounds are the only physical constraint)
- Numerical representation check against the declared dtype (torch.finfo)
- Dependency relationship verification
- Quality metrics calculation
- Automated audit report generation
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass

import torch

from relic.data.standards.dimensionality import (
    dimension_category_from_unit,
    is_declared_dimension_compatible,
)
from relic.parameters import get_global_manager
from relic.parameters.spec import parameter_bounds_error
from relic.utils.physical_constants import get_physical_constant

logger = logging.getLogger(__name__)


@dataclass
class AuditResult:
    """Parameter audit result data structure."""
    parameter_key: str
    rule_name: str
    status: str  # 'PASS', 'WARNING', 'ERROR'
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ParameterQualityMetrics:
    """Parameter quality assessment metrics."""
    completeness_score: float
    scientific_confidence: str
    validation_status: str
    last_updated: str
    quality_level: str


def bounds_declaration_check(bounds: Any, dtype: Any) -> Tuple[str, str, Dict[str, Any]]:
    """Status, message and details of a bounds declaration.

    The one rule for the bounds field is relic.parameters.spec.parameter_bounds_error
    (None only for a non-numeric dtype, otherwise a (lower, upper) pair with both sides
    declared); a degenerate interval [c, c] fixes a constant, and only lower > upper is
    an order error."""
    error = parameter_bounds_error(bounds, dtype)
    if error is not None:
        return 'ERROR', f'Invalid bounds declaration: {error}', {'bounds': bounds, 'dtype': dtype}
    if bounds is None:
        return 'PASS', 'No interval applies to this non-numeric parameter', {'dtype': dtype}
    if bounds[0] > bounds[1]:
        return 'ERROR', 'Bounds order error (min > max)', {'min': bounds[0], 'max': bounds[1]}
    return 'PASS', 'Bounds declaration is valid', {'bounds': bounds, 'dtype': dtype}


def unit_dimension_check(units: Any, dimension: Any) -> Tuple[str, str, Dict[str, Any]]:
    """Status, message and details of a unit against its declared dimension.

    The SI dimensionality owner (relic.data.standards.dimensionality) derives the unit;
    no unit list and no rule of its own is kept here."""
    declared = str(dimension or '').strip()
    canonical = dimension_category_from_unit(units)
    if canonical is None:
        return 'ERROR', f'Unit {units!r} has no SI dimensionality derivation', {'units': units}
    if not declared:
        return (
            'WARNING',
            'No declared dimension to compare the unit with',
            {'units': units, 'canonical_dimension': canonical},
        )
    details = {'units': units, 'dimension': declared, 'canonical_dimension': canonical}
    if not is_declared_dimension_compatible(units, declared):
        return (
            'ERROR',
            f'Unit {units!r} is inconsistent with the declared dimension {declared!r} '
            f'(canonical {canonical!r})',
            details,
        )
    return 'PASS', 'Unit derivation matches the declared dimension', details


def dtype_representation_check(value: Any, dtype: Any) -> Tuple[str, str, Dict[str, Any]]:
    """Status, message and details of a value against the representation limits of its
    declared floating-point dtype (torch.finfo); a non-floating dtype (text, integer,
    boolean) has no floating-point representation limit."""
    dtype_text = str(dtype or '').strip()
    if not dtype_text:
        return 'WARNING', 'No declared dtype to check the representation against', {'value': value}
    torch_dtype = getattr(torch, dtype_text.lower(), None)
    if not (isinstance(torch_dtype, torch.dtype) and torch_dtype.is_floating_point):
        return 'PASS', f'dtype {dtype_text!r} has no floating-point representation limit', {'dtype': dtype_text}
    try:
        values = torch.as_tensor(value, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError) as exc:
        # A value that is not numeric under a floating-point dtype is a data error.
        return 'ERROR', f'Value {value!r} is not numeric for dtype {dtype_text!r}: {exc}', {'value': value}

    # References: IEEE 754-2019 binary floating-point formats; torch.finfo.
    # Mathematical expression: v is reliably representable in dtype d when v is
    #   finite, |v| <= max_d, and v = 0 or |v| >= tiny_d; then the relative rounding
    #   error |fl_d(v) - v| / |v| <= eps_d / 2.
    # Parameter meanings: max_d largest finite value, tiny_d smallest positive
    #   normal value, eps_d machine epsilon of d; fl_d rounding to d.
    # Units: those of the parameter (the check is scale-free per value).
    # Domain/constraints: |v| > max_d overflows to inf (ERROR); 0 < |v| < tiny_d is
    #   subnormal, with relative error above eps_d (WARNING).
    # Sign convention: none (magnitudes only).
    info = torch.finfo(torch_dtype)
    magnitude = values.abs()
    rounded = values.to(torch_dtype).to(torch.float64)
    nonzero = magnitude > 0
    relative_error = float(
        ((rounded - values).abs()[nonzero] / magnitude[nonzero]).max()
    ) if bool(nonzero.any()) else 0.0
    details = {
        'value': value, 'dtype': dtype_text, 'max': info.max, 'tiny': info.tiny,
        'eps': info.eps, 'relative_rounding_error': relative_error,
    }
    if not bool(torch.all(torch.isfinite(values))) or bool(torch.any(magnitude > info.max)):
        return 'ERROR', f'Value is not finite in dtype {dtype_text!r} (|v| > {info.max:.3e})', details
    if bool(torch.any(nonzero & (magnitude < info.tiny))) or relative_error > info.eps:
        return (
            'WARNING',
            f'Value is subnormal in dtype {dtype_text!r} (|v| < {info.tiny:.3e}); its '
            f'relative rounding error {relative_error:.3e} exceeds eps {info.eps:.3e}',
            details,
        )
    return 'PASS', f'Value is representable in dtype {dtype_text!r}', details


class IsotopeParameterAuditor:
    """
    Comprehensive auditor for isotope tracking parameters.

    Performs systematic validation of all isotope-related parameters against
    scientific standards, physical constraints, and numerical stability requirements.
    """

    def __init__(self):
        """Initialize the parameter auditor."""
        self.logger = logging.getLogger(f"{__name__}.IsotopeParameterAuditor")

        # Build audit rules
        self.audit_rules = self._build_audit_rules()

        # Parameter dependency network
        self.dependency_network = self._build_dependency_network()

        self.logger.info("Initialized isotope parameter auditor")

    def _build_audit_rules(self) -> Dict[str, callable]:
        """Build dictionary of audit rules."""
        return {
            'bounds_consistency': self._check_bounds_consistency,
            'units_validity': self._check_units_validity,
            'references_completeness': self._check_references_completeness,
            'numerical_stability': self._check_numerical_stability,
            'dependency_completeness': self._check_dependency_completeness,
            'metadata_completeness': self._check_metadata_completeness,
            'scientific_rigor': self._check_scientific_rigor
        }

    def _build_dependency_network(self) -> Dict[str, Set[str]]:
        """Build parameter dependency relationship network."""
        return {
            'temperature_dependencies': {
                'isotopes.temperature.reference',
                'relic.physical.constants.universal.gas_constant',
                'isotopes.validation.numerical.temperature_diff_lower',
                'isotopes.validation.numerical.temperature_diff_upper'
            },
            'fractionation_dependencies': {
                'isotopes.fractionation.bounds.min',
                'isotopes.fractionation.bounds.max',
                'isotopes.standards.carbon_13.pdb_standard',
                'isotopes.standards.nitrogen_15.air_ratio',
                'isotopes.standards.oxygen_18.vsmow_standard'
            },
            'validation_dependencies': {
                'isotopes.validation.numerical.epsilon',
                'isotopes.validation.numerical.arrhenius_exponent_min',
                'isotopes.validation.numerical.arrhenius_exponent_max'
            },
            'pool_constraints': {
                'isotopes.pool.min_total',
                'isotopes.pool.max_ratio'
            }
        }

    def audit_parameter(self, param_key: str, param_metadata: Dict[str, Any]) -> List[AuditResult]:
        """
        Audit a single parameter against all validation rules.

        Parameters:
            param_key : str
                Parameter unique identifier
            param_metadata : Dict[str, Any]
                Parameter metadata dictionary

        Returns:
            List[AuditResult]: Audit results for all rules
        """
        results = []

        self.logger.debug(f"Auditing parameter: {param_key}")

        for rule_name, rule_func in self.audit_rules.items():
            try:
                result = rule_func(param_key, param_metadata)
                results.append(result)

                if result.status == 'ERROR':
                    self.logger.error(f"Parameter {param_key} failed {rule_name}: {result.message}")
                elif result.status == 'WARNING':
                    self.logger.warning(f"Parameter {param_key} warning in {rule_name}: {result.message}")

            except (TypeError, ValueError, KeyError) as e:
                # A rule that fails on the audited data (a value of the wrong type or
                # content, a missing metadata field) is an ERROR result of the audit; a
                # defect in a rule propagates.
                error_result = AuditResult(
                    parameter_key=param_key,
                    rule_name=rule_name,
                    status='ERROR',
                    message=f"Audit rule execution failed: {str(e)}",
                    details={'exception_type': type(e).__name__}
                )
                results.append(error_result)
                self.logger.error(f"Error executing {rule_name} for {param_key}: {e}")

        return results

    def audit_all_parameters(self) -> Dict[str, List[AuditResult]]:
        """
        Audit all available isotope parameters.

        Returns:
            Dict[str, List[AuditResult]]: Audit results by parameter key
        """
        self.logger.info("Starting comprehensive parameter audit")

        all_results: Dict[str, List[AuditResult]] = {}
        isotope_params = self._get_isotope_parameters_from_manager()
        for param_key, param_metadata in isotope_params.items():
            all_results[param_key] = self.audit_parameter(param_key, param_metadata)

        self.logger.info(f"Completed audit of {len(all_results)} parameters")
        return all_results

    @staticmethod
    def parameter_metadata(param_key: str) -> Dict[str, Any]:
        """The audited metadata of one ParameterManager entry: its value and the declared
        units, dimension, dtype, bounds, description and references."""
        manager = get_global_manager()
        spec = manager.get_parameter(param_key)
        if spec is None:
            raise KeyError(f"Parameter specification missing for '{param_key}'")
        return {
            "value": manager.get(param_key, required=True),
            "key": param_key,
            "units": spec.units,
            "dimension": spec.dimension,
            "dtype": spec.dtype,
            "bounds": spec.bounds,
            "description": spec.description,
            "references": spec.references,
        }

    def _get_isotope_parameters_from_manager(self) -> Dict[str, Dict[str, Any]]:
        """Every entry of the isotopes namespace, loaded through the manager (a listing of
        the already cached keys would omit the entries not read yet)."""
        specs = get_global_manager().get_namespace_parameters(
            namespace="isotopes", include_subnamespaces=True, return_specs=True
        )
        return {key: self.parameter_metadata(key) for key in specs}

    # Individual audit rule implementations
    def _check_bounds_consistency(self, param_key: str, metadata: Dict[str, Any]) -> AuditResult:
        """Check the declared bounds and the value against them.

        The bounds declared in the YAML entry are the only physical constraint: no rule
        guesses a constraint from the key, the dimension or the magnitude of a value
        (a temperature difference or a regression coefficient in K is not an absolute
        temperature). A value outside its declared bounds, or a non-finite value, is an
        ERROR.
        """
        if 'bounds' not in metadata:
            return AuditResult(
                parameter_key=param_key,
                rule_name='bounds_consistency',
                status='WARNING',
                message='Missing bounds definition',
                details={'missing_field': 'bounds'}
            )

        bounds = metadata['bounds']
        value = metadata.get('value')

        status, message, details = bounds_declaration_check(bounds, metadata.get('dtype'))
        if status != 'PASS' or bounds is None:
            return AuditResult(
                parameter_key=param_key,
                rule_name='bounds_consistency',
                status=status,
                message=message,
                details=details
            )

        min_val, max_val = bounds

        # Only a numeric value is compared with the interval; a text, list or mapping
        # value has no order against it, and a boolean switch is not a quantity.
        if isinstance(value, bool) or not isinstance(value, (int, float, torch.Tensor)):
            return AuditResult(
                parameter_key=param_key,
                rule_name='bounds_consistency',
                status='PASS',
                message='Non-numeric value; the interval is not compared',
                details={'bounds': bounds, 'type': type(value).__name__}
            )

        # The value must lie within its declared bounds (a NaN fails the comparison).
        values = torch.as_tensor(value, dtype=torch.float64)
        if not bool(torch.all((values >= min_val) & (values <= max_val))):
            return AuditResult(
                parameter_key=param_key,
                rule_name='bounds_consistency',
                status='ERROR',
                message='Value outside its declared bounds',
                details={'value': value, 'bounds': bounds}
            )

        return AuditResult(
            parameter_key=param_key,
            rule_name='bounds_consistency',
            status='PASS',
            message='Bounds consistency check passed',
            details={'bounds': bounds, 'value': value}
        )

    def _check_units_validity(self, param_key: str, metadata: Dict[str, Any]) -> AuditResult:
        """Check the declared unit through the SI dimensionality owner and against the
        declared dimension."""
        if 'units' not in metadata:
            return AuditResult(
                parameter_key=param_key,
                rule_name='units_validity',
                status='ERROR',
                message='Missing units definition',
                details={'missing_field': 'units'}
            )

        status, message, details = unit_dimension_check(metadata['units'], metadata.get('dimension'))
        return AuditResult(
            parameter_key=param_key,
            rule_name='units_validity',
            status=status,
            message=message,
            details=details
        )

    def _check_references_completeness(self, param_key: str, metadata: Dict[str, Any]) -> AuditResult:
        """Check parameter references completeness."""
        if 'references' not in metadata:
            return AuditResult(
                parameter_key=param_key,
                rule_name='references_completeness',
                status='WARNING',
                message='Missing references',
                details={'missing_field': 'references'}
            )

        references = metadata['references']

        if not references:
            return AuditResult(
                parameter_key=param_key,
                rule_name='references_completeness',
                status='WARNING',
                message='Empty references list',
                details={'references': references}
            )

        # Check reference formats
        valid_references = 0
        doi_references = 0

        for ref in references:
            if isinstance(ref, str):
                if ref.startswith('doi:'):
                    if self._validate_doi_format(ref[4:]):
                        doi_references += 1
                        valid_references += 1
                    else:
                        self.logger.warning(f"Invalid DOI format: {ref}")
                elif any(indicator in ref.lower() for indicator in ['journal', 'proceedings', 'book']):
                    valid_references += 1

        reference_quality = 'HIGH' if doi_references >= 1 else 'MEDIUM' if valid_references >= 1 else 'LOW'

        if valid_references == 0:
            return AuditResult(
                parameter_key=param_key,
                rule_name='references_completeness',
                status='WARNING',
                message='No valid references found',
                details={'references': references, 'quality': reference_quality}
            )

        return AuditResult(
            parameter_key=param_key,
            rule_name='references_completeness',
            status='PASS',
            message='References completeness check passed',
            details={
                'total_references': len(references),
                'valid_references': valid_references,
                'doi_references': doi_references,
                'quality': reference_quality
            }
        )

    def _validate_doi_format(self, doi: str) -> bool:
        """Validate DOI format."""
        doi_pattern = r'10\.\d+/.+'
        return bool(re.match(doi_pattern, doi))

    def _check_numerical_stability(self, param_key: str, metadata: Dict[str, Any]) -> AuditResult:
        """Check that the value is reliably representable in its declared dtype.

        The limits are those of the declared floating-point dtype (torch.finfo), not
        magnitudes guessed as problematic. A non-floating dtype (text, integer, boolean)
        has no floating-point representation limit to check."""
        status, message, details = dtype_representation_check(
            metadata.get('value'), metadata.get('dtype')
        )
        return AuditResult(
            parameter_key=param_key,
            rule_name='numerical_stability',
            status=status,
            message=message,
            details=details
        )

    def _check_dependency_completeness(self, param_key: str, metadata: Dict[str, Any]) -> AuditResult:
        """
        Check parameter dependency completeness.

        Verifies that all parameters listed in the dependency network
        for each dependency category that param_key belongs to are
        actually available in the parameter manager.

        Parameters
        ----------
        param_key : str
            Fully qualified parameter key
        metadata : Dict[str, Any]
            Parameter metadata dictionary

        Returns
        -------
        AuditResult
            PASS if all dependencies available, WARNING if some missing
        """
        missing_deps = []

        def dependency_is_available(dep_key: str) -> bool:
            """Check availability through the canonical owner for each key type.

            A parameter key is available when the ParameterManager holds it (a missing
            key reads as None). A physical constant goes through the constant SSOT, which
            raises ValueError for a missing constant; that error propagates, and the
            audit records it as an ERROR result, never as a missing dependency."""
            if dep_key.startswith("relic.physical.constants."):
                get_physical_constant(dep_key)
                return True
            return get_global_manager().get(dep_key, required=False) is not None

        # Check each dependency group that contains this parameter
        for group_name, dep_keys in self.dependency_network.items():
            if param_key in dep_keys:
                # This parameter belongs to this group; verify all group members exist
                for dep_key in dep_keys:
                    if dep_key == param_key:
                        continue
                    if not dependency_is_available(dep_key):
                        missing_deps.append(dep_key)

        # Also check 'dependencies' field in metadata if present
        explicit_deps = metadata.get('dependencies', [])
        for dep_key in explicit_deps:
            if not dependency_is_available(dep_key):
                missing_deps.append(dep_key)

        if missing_deps:
            return AuditResult(
                parameter_key=param_key,
                rule_name='dependency_completeness',
                status='WARNING',
                message=f"Missing {len(missing_deps)} dependency parameter(s)",
                details={'missing_dependencies': list(set(missing_deps))}
            )

        return AuditResult(
            parameter_key=param_key,
            rule_name='dependency_completeness',
            status='PASS',
            message='All dependency parameters available',
            details={'checked_groups': [
                g for g, keys in self.dependency_network.items()
                if param_key in keys
            ]}
        )

    def _check_metadata_completeness(self, param_key: str, metadata: Dict[str, Any]) -> AuditResult:
        """Check parameter metadata completeness."""
        required_fields = ['value', 'units', 'description']
        recommended_fields = ['bounds', 'references', 'dimension', 'dtype']

        missing_required = [field for field in required_fields if field not in metadata]
        missing_recommended = [field for field in recommended_fields if field not in metadata]

        if missing_required:
            return AuditResult(
                parameter_key=param_key,
                rule_name='metadata_completeness',
                status='ERROR',
                message=f'Missing required fields: {", ".join(missing_required)}',
                details={'missing_required': missing_required}
            )

        if missing_recommended:
            return AuditResult(
                parameter_key=param_key,
                rule_name='metadata_completeness',
                status='WARNING',
                message=f'Missing recommended fields: {", ".join(missing_recommended)}',
                details={'missing_recommended': missing_recommended}
            )

        return AuditResult(
            parameter_key=param_key,
            rule_name='metadata_completeness',
            status='PASS',
            message='Metadata completeness check passed',
            details={
                'total_fields': len(metadata),
                'has_required': len(required_fields),
                'has_recommended': len(recommended_fields) - len(missing_recommended)
            }
        )

    def _check_scientific_rigor(self, param_key: str, metadata: Dict[str, Any]) -> AuditResult:
        """Check scientific rigor of parameter definition."""
        score = 0
        max_score = 5

        # Has references (1 point)
        if 'references' in metadata and metadata['references']:
            score += 1

        # Has units (1 point)
        if 'units' in metadata:
            score += 1

        # Has bounds (1 point)
        if 'bounds' in metadata:
            score += 1

        # Has dimension (1 point)
        if 'dimension' in metadata:
            score += 1

        # Has DOI references (1 point)
        if 'references' in metadata:
            for ref in metadata['references']:
                if isinstance(ref, str) and ref.startswith('doi:'):
                    score += 1
                    break

        # Determine status based on score
        if score >= 4:
            status = 'PASS'
            message = 'High scientific rigor'
        elif score >= 2:
            status = 'WARNING'
            message = 'Moderate scientific rigor'
        else:
            status = 'ERROR'
            message = 'Low scientific rigor'

        return AuditResult(
            parameter_key=param_key,
            rule_name='scientific_rigor',
            status=status,
            message=message,
            details={'score': score, 'max_score': max_score}
        )

    def generate_audit_report(self, audit_results: Dict[str, List[AuditResult]]) -> str:
        """
        Generate comprehensive audit report.

        Parameters:
            audit_results : Dict[str, List[AuditResult]]
                Audit results by parameter key

        Returns:
            str: Formatted audit report
        """
        report_lines = []
        report_lines.append("# RELIC Isotope Parameter Audit Report\n")
        report_lines.append(f"Generated: {datetime.now().isoformat()}\n")

        # Calculate statistics
        total_params = len(audit_results)
        total_rules = sum(len(results) for results in audit_results.values())

        passed_rules = sum(
            1 for results in audit_results.values()
            for result in results
            if result.status == 'PASS'
        )
        warning_rules = sum(
            1 for results in audit_results.values()
            for result in results
            if result.status == 'WARNING'
        )
        error_rules = sum(
            1 for results in audit_results.values()
            for result in results
            if result.status == 'ERROR'
        )

        # Add statistics section
        report_lines.append("## Audit Statistics\n")
        report_lines.append(f"- Total parameters: {total_params}")
        report_lines.append(f"- Total rule checks: {total_rules}")
        report_lines.append(f"- Passed checks: {passed_rules} ({passed_rules/total_rules*100:.1f}%)")
        report_lines.append(f"- Warnings: {warning_rules} ({warning_rules/total_rules*100:.1f}%)")
        report_lines.append(f"- Errors: {error_rules} ({error_rules/total_rules*100:.1f}%)")
        report_lines.append("")

        # Add summary by parameter
        report_lines.append("## Parameter Audit Summary\n")

        for param_key, param_results in audit_results.items():
            passed = sum(1 for r in param_results if r.status == 'PASS')
            warnings = sum(1 for r in param_results if r.status == 'WARNING')
            errors = sum(1 for r in param_results if r.status == 'ERROR')

            status_icon = '[ERROR]' if errors else ('[WARN]' if warnings else '[OK]')

            report_lines.append(f"### {status_icon} {param_key}")
            report_lines.append(f"- Passed: {passed}, Warnings: {warnings}, Errors: {errors}")

            # List errors and warnings
            for result in param_results:
                if result.status != 'PASS':
                    icon = '[WARN]' if result.status == 'WARNING' else '[ERROR]'
                    report_lines.append(f"  - {icon} **{result.rule_name}**: {result.message}")

            report_lines.append("")

        # Add detailed results section
        report_lines.append("## Detailed Audit Results\n")

        for param_key, param_results in audit_results.items():
            report_lines.append(f"### {param_key}\n")

            for result in param_results:
                status_icon = {'PASS': '[OK]', 'WARNING': '[WARN]', 'ERROR': '[ERROR]'}.get(result.status, '?')
                report_lines.append(f"#### {status_icon} {result.rule_name}: {result.status}")
                report_lines.append(f"**Message**: {result.message}")

                if result.details:
                    report_lines.append("**Details**:")
                    for key, value in result.details.items():
                        report_lines.append(f"- {key}: {value}")

                report_lines.append(f"**Timestamp**: {result.timestamp}")
                report_lines.append("")

        return "\n".join(report_lines)

    def save_audit_report(self, audit_results: Dict[str, List[AuditResult]],
                         output_path: Optional[str] = None) -> str:
        """
        Save audit report to file.

        Parameters:
            audit_results : Dict[str, List[AuditResult]]
                Audit results
            output_path : Optional[str]
                Output file path (if None, generates default)

        Returns:
            str: Path to saved report file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"isotope_parameter_audit_{timestamp}.md"

        report_content = self.generate_audit_report(audit_results)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)

            self.logger.info(f"Audit report saved to: {output_path}")
            return output_path

        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            self.logger.error(f"Failed to save audit report: {e}")
            raise


# Convenience functions
def create_parameter_auditor() -> IsotopeParameterAuditor:
    """Create and return a parameter auditor instance."""
    return IsotopeParameterAuditor()


def run_comprehensive_audit() -> Dict[str, List[AuditResult]]:
    """Run comprehensive parameter audit and return results."""
    auditor = create_parameter_auditor()
    return auditor.audit_all_parameters()


def generate_audit_report(audit_results: Dict[str, List[AuditResult]]) -> str:
    """Generate audit report from results."""
    auditor = create_parameter_auditor()
    return auditor.generate_audit_report(audit_results)


# Export public interface
__all__ = [
    'IsotopeParameterAuditor',
    'AuditResult',
    'ParameterQualityMetrics',
    'create_parameter_auditor',
    'run_comprehensive_audit',
    'generate_audit_report'
]

