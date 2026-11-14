"""
Parameter Audit Data Structures for RELIC.

This module defines the data containers used by the parameter audit workflow,
including issue descriptors and structured audit reports. The implementation
follows best practices from workflow systems such as PEcAn (LeBauer et al.,
2013, Geosci. Model Dev.) where parameter completeness, provenance, and
uncertainty are tracked programmatically.

References:
    - LeBauer, D. et al. (2013). "Facilitating feedbacks between field data and
      ecosystem models." Geosci. Model Dev., 6, 437-447.
    - CF-1.8 Convention for Climate and Forecast Metadata (units, naming).
    - FAIR Guiding Principles for scientific data management (Wilkinson et al.,
      2016, Sci. Data).
"""

from __future__ import annotations

import builtins
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

from .serialization import YAML_AVAILABLE
from .spec import parameter_bounds_error


class AuditIssueType(str, Enum):
    """Enumeration of audit issue categories."""

    MISSING = "missing"
    ORPHAN = "orphan"
    DUPLICATE = "duplicate"
    VALIDATION_ERROR = "validation_error"
    UNIT_MISMATCH = "unit_mismatch"
    TYPE_MISMATCH = "type_mismatch"
    UNUSED = "unused"
    METADATA_GAP = "metadata_gap"


class AuditSeverity(str, Enum):
    """Severity levels for audit issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ParameterAuditIssue:
    """Structured description of a parameter audit issue.

    Parameters
    ----------
    key : str
        Parameter key expressed in dot notation.
    issue_type : AuditIssueType
        Category describing the issue nature.
    severity : AuditSeverity
        Severity level based on operational impact.
    message : str
        Human readable description of the issue.
    context : Dict[str, Any]
        Additional contextual information (e.g., source_file, namespace).
    references : List[str]
        Literature or documentation references backing the rule.

    Notes
    -----
    The audit output is designed to align with the FAIR principles, ensuring
    each issue can be traced back to provenance metadata and reproducible
    validation routines.
    """

    key: str
    issue_type: AuditIssueType
    severity: AuditSeverity
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    references: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert issue to a serializable dictionary."""

        return {
            "key": self.key,
            "issue_type": self.issue_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "context": self.context,
            "references": self.references,
        }


@dataclass
class ParameterAuditReport:
    r"""Aggregate report describing parameter registry health.

    Parameters
    ----------
    created_at : datetime
        Timestamp when the audit report was generated (UTC recommended).
    issues : List[ParameterAuditIssue]
        All issues discovered during the scan.
    scanned_namespaces : List[str]
        Namespaces inspected during the audit.
    expected_total : int
        Total number of expected parameters from references or inputs.
    observed_total : int
        Total number of observed/registered parameters.
    notes : List[str]
        Additional notes (e.g., assumptions, filters applied).

    Mathematical Expression
    -----------------------
    Let :math:`E` be the expected parameter set and :math:`O` the observed set.
    Missing parameters are :math:`E \setminus O`, while orphan parameters are
    :math:`O \setminus E`. Consistency checks ensure
    :math:`\forall p \in O`, units(p) adhere to CF-1.8 tables.
    """

    expected_total: int
    observed_total: int
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    issues: List[ParameterAuditIssue] = field(default_factory=list)
    scanned_namespaces: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def add_issue(self, issue: ParameterAuditIssue) -> None:
        """Add an issue to the report."""

        self.issues.append(issue)

    def extend(self, issues: Iterable[ParameterAuditIssue]) -> None:
        """Add multiple issues to the report."""

        for issue in issues:
            self.add_issue(issue)

    def summarize_by_type(self) -> Dict[str, int]:
        """Count issues grouped by issue type."""

        summary: Dict[str, int] = {}
        for issue in self.issues:
            summary.setdefault(issue.issue_type.value, 0)
            summary[issue.issue_type.value] += 1
        return summary

    def summarize_by_severity(self) -> Dict[str, int]:
        """Count issues grouped by severity."""

        summary: Dict[str, int] = {}
        for issue in self.issues:
            summary.setdefault(issue.severity.value, 0)
            summary[issue.severity.value] += 1
        return summary

    def filter_by_type(self, issue_type: AuditIssueType) -> List[ParameterAuditIssue]:
        """Return issues of a specific type."""

        return [issue for issue in self.issues if issue.issue_type == issue_type]

    def filter_by_severity(self, severity: AuditSeverity) -> List[ParameterAuditIssue]:
        """Return issues reaching a specific severity level."""

        return [issue for issue in self.issues if issue.severity == severity]

    def has_failures(self) -> bool:
        """Check whether any error or critical issues exist."""

        return any(
            issue.severity in {AuditSeverity.ERROR, AuditSeverity.CRITICAL}
            for issue in self.issues
        )

    @property
    def missing_keys(self) -> List[str]:
        """List missing parameter keys."""

        return [issue.key for issue in self.issues if issue.issue_type == AuditIssueType.MISSING]

    @property
    def orphan_keys(self) -> List[str]:
        """List orphan parameter keys."""

        return [issue.key for issue in self.issues if issue.issue_type == AuditIssueType.ORPHAN]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the report to a serializable dictionary."""

        return {
            "created_at": self.created_at.isoformat(),
            "issues": [issue.to_dict() for issue in self.issues],
            "summary": {
                "by_type": self.summarize_by_type(),
                "by_severity": self.summarize_by_severity(),
            },
            "scanned_namespaces": self.scanned_namespaces,
            "expected_total": self.expected_total,
            "observed_total": self.observed_total,
            "notes": self.notes,
        }

    def to_table_rows(self) -> List[Dict[str, Any]]:
        """Represent issues as table rows for CLI rendering."""

        return [
            {
                "key": issue.key,
                "issue_type": issue.issue_type.value,
                "severity": issue.severity.value,
                "message": issue.message,
                "source_file": issue.context.get("source_file"),
            }
            for issue in self.issues
        ]

    def summary_text(self) -> str:
        """Generate a concise textual summary."""

        type_summary = self.summarize_by_type()
        severity_summary = self.summarize_by_severity()
        parts = [
            f"Expected: {self.expected_total}",
            f"Observed: {self.observed_total}",
            f"Issues: {len(self.issues)}",
            f"By type: {type_summary}",
            f"By severity: {severity_summary}",
        ]
        return "; ".join(parts)


@dataclass(frozen=True)
class StructuralParameterAudit:
    """Compact RELIC-IPHS-DEC structural parameter audit summary.

    References:
      - RELIC-IPHS-DEC Typed Cochain Morphism Enhancement Plan.
      - Hirani (2003), Discrete Exterior Calculus.
      - Grmela and Oettinger (1997), GENERIC formalism.
    Mathematical expression:
      Legal generated structures satisfy H = H.T >= 0, M = M.T >= 0,
      J e_S = 0, and M e_E = 0 when the corresponding structures are declared.
    Parameter meanings:
      bounds_ok: all structural parameters declare valid bounds.
      units_ok: all structural parameters declare compatible units.
      references_ok: all structural parameters cite scientific references.
      provenance_ok: all structural parameters declare provenance.
    """

    bounds_ok: bool = True
    units_ok: bool = True
    hodge_psd_ok: bool = True
    hodge_symmetry_ok: bool = True
    dissipation_psd_ok: bool = True
    degeneracy_energy_ok: bool = True
    degeneracy_entropy_ok: bool = True
    references_ok: bool = True
    provenance_ok: bool = True
    errors: tuple[str, ...] = ()

    @classmethod
    def from_report(cls, report: ParameterAuditReport) -> "StructuralParameterAudit":
        """Build a compact structural audit summary from ParameterAuditReport."""

        field_failures = {
            str(issue.context.get("field", "")).strip().lower()
            for issue in report.issues
            if issue.severity in {AuditSeverity.ERROR, AuditSeverity.CRITICAL}
        }
        messages = tuple(
            issue.message
            for issue in report.issues
            if issue.severity in {AuditSeverity.ERROR, AuditSeverity.CRITICAL}
        )
        lower_messages = " | ".join(message.lower() for message in messages)

        return cls(
            bounds_ok="bounds" not in field_failures,
            units_ok="units" not in field_failures and "unit" not in lower_messages,
            hodge_psd_ok=not (
                "hodge" in lower_messages and "positive semidefinite" in lower_messages
            ),
            hodge_symmetry_ok=not (
                "hodge" in lower_messages and "symmetric" in lower_messages
            ),
            dissipation_psd_ok=not (
                "dissipation" in lower_messages
                and "positive semidefinite" in lower_messages
            ),
            degeneracy_energy_ok="energy_gradient" not in lower_messages,
            degeneracy_entropy_ok="entropy_gradient" not in lower_messages,
            references_ok="references" not in field_failures,
            provenance_ok="provenance" not in field_failures,
            errors=messages,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the structural audit summary to a serializable dictionary."""

        return {
            "bounds_ok": self.bounds_ok,
            "units_ok": self.units_ok,
            "hodge_psd_ok": self.hodge_psd_ok,
            "hodge_symmetry_ok": self.hodge_symmetry_ok,
            "dissipation_psd_ok": self.dissipation_psd_ok,
            "degeneracy_energy_ok": self.degeneracy_energy_ok,
            "degeneracy_entropy_ok": self.degeneracy_entropy_ok,
            "references_ok": self.references_ok,
            "provenance_ok": self.provenance_ok,
            "errors": list(self.errors),
        }


logger = logging.getLogger(__name__)


def _in_namespace(key: str, namespace: str) -> bool:
    """True for a key below "<namespace>." (the ParameterStorage.list rule).

    A sibling prefix ("testx.a" for namespace "test") and the namespace key
    itself are not members.
    """
    return key.startswith(f"{namespace}.")


def load_category(manager, category: str, subcategory: Optional[str] = None) -> List[str]:
    """
    Backward-consistent helper to preload a category/subcategory namespace.

    Notes
    -----
    The unified manager uses a lazy database loader. This helper preloads the
    requested namespace and returns the discovered keys.
    """
    if not isinstance(category, str) or not category.strip():
        raise ValueError("category must be a non-empty string")

    namespace = category.strip()
    if subcategory:
        namespace = f"{namespace}.{str(subcategory).strip()}"

    manager.database.load_namespace(namespace)

    return manager.list(namespace=namespace)


def run_registry_audit(
    manager,
    expected_params: Optional[List[str]] = None,
    namespace_filter: Optional[str] = None,
    include_unused: bool = True,
    normalize_key: Optional[Callable[[str], str]] = None,
) -> ParameterAuditReport:
    """
    Execute comprehensive parameter registry audit.

    Scans the entire parameter ecosystem to identify:
    - Missing parameters (requested but not found)
    - Orphan parameters (defined but never used)
    - Duplicate registrations
    - Unit/bounds inconsistencies
    - Validation errors
    """
    expected_set = _collect_expected_parameters(manager, expected_params, namespace_filter)
    observed_set = _collect_observed_parameters(manager, namespace_filter)
    report = ParameterAuditReport(
        expected_total=len(expected_set),
        observed_total=len(observed_set),
        created_at=datetime.now(timezone.utc),
        scanned_namespaces=[namespace_filter] if namespace_filter else ["all"],
    )

    missing_issues = _scan_missing_parameters(manager, expected_set, observed_set, namespace_filter)
    report.extend(missing_issues)

    if include_unused:
        orphan_issues = _scan_orphan_parameters(manager, expected_set, observed_set)
        report.extend(orphan_issues)

    duplicate_issues = _scan_duplicate_registrations(
        manager,
        normalize_key=normalize_key,
    )
    report.extend(duplicate_issues)

    validation_issues = _scan_validation_errors(manager, namespace_filter)
    report.extend(validation_issues)

    unit_issues = _scan_unit_consistency(manager)
    report.extend(unit_issues)

    report.notes.append(f"Audit completed at {datetime.now(timezone.utc).isoformat()}")
    if namespace_filter:
        report.notes.append(f"Filtered to namespace: {namespace_filter}")
    if not include_unused:
        report.notes.append("Unused parameter detection disabled")

    if getattr(manager, "_missing_requests", None):
        top_missing = sorted(
            manager._missing_requests.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:10]
        report.notes.append(
            f"Top missing requests: {', '.join(f'{k}({v})' for k, v in top_missing)}"
        )

    return report


def get_missing_requests(manager) -> Dict[str, int]:
    """Get statistics on missing parameter requests."""
    return manager._missing_requests.copy()


def clear_missing_requests(manager) -> None:
    """Clear the missing requests tracking."""
    manager._missing_requests.clear()


def _collect_expected_parameters(
    manager,
    explicit_list: Optional[List[str]] = None,
    namespace_filter: Optional[str] = None,
) -> Set[str]:
    """
    Collect the set of expected parameters.

    Sources (in priority order):
    1. Explicit list provided by user
    2. Parameters from database YAML files
    3. Parameters from Functor metadata (future)
    """
    expected: Set[str] = builtins.set()

    if explicit_list:
        expected.update(explicit_list)
        return expected

    # A database that cannot be loaded (unreadable file, duplicate key, invalid
    # entry) fails the audit; it must not shrink the expected set to a clean result.
    db_params = manager.database.load_all()
    for key in db_params.keys():
        if namespace_filter:
            if _in_namespace(key, namespace_filter):
                expected.add(key)
        else:
            expected.add(key)

    for key in manager.parameters.keys():
        if namespace_filter:
            if _in_namespace(key, namespace_filter):
                expected.add(key)
        else:
            expected.add(key)

    return expected


def _collect_observed_parameters(manager, namespace_filter: Optional[str] = None) -> Set[str]:
    """
    Collect the set of observed/available parameters.

    Sources:
    1. Currently registered parameters
    2. Storage cache
    3. Database entries
    """
    observed: Set[str] = builtins.set()

    for key in manager.parameters.keys():
        if namespace_filter:
            if _in_namespace(key, namespace_filter):
                observed.add(key)
        else:
            observed.add(key)

    if hasattr(manager, "storage") and manager.storage:
        storage_keys = manager.storage.list(namespace=namespace_filter)
        observed.update(storage_keys)

    return observed


def _scan_missing_parameters(
    manager,
    expected: Set[str],
    observed: Set[str],
    namespace_filter: Optional[str] = None,
) -> List[ParameterAuditIssue]:
    """Identify parameters that were expected but not found."""
    issues: List[ParameterAuditIssue] = []
    missing = expected - observed

    for key in missing:
        request_count = manager._missing_requests.get(key, 0)
        severity = AuditSeverity.ERROR if request_count > 0 else AuditSeverity.WARNING
        issue = ParameterAuditIssue(
            key=key,
            issue_type=AuditIssueType.MISSING,
            severity=severity,
            message=f"Parameter '{key}' is defined but not available in registry",
            context={
                "request_count": request_count,
                "defined_in": "database",
            },
        )
        issues.append(issue)

    for key, count in manager._missing_requests.items():
        if namespace_filter and not _in_namespace(key, namespace_filter):
            continue
        if key not in expected and key not in observed:
            issue = ParameterAuditIssue(
                key=key,
                issue_type=AuditIssueType.MISSING,
                severity=AuditSeverity.CRITICAL if count > 10 else AuditSeverity.ERROR,
                message=f"Parameter '{key}' requested {count} times but never defined",
                context={
                    "request_count": count,
                    "defined_in": "none",
                },
            )
            issues.append(issue)

    return issues


def _scan_orphan_parameters(
    manager,
    expected: Set[str],
    observed: Set[str],
) -> List[ParameterAuditIssue]:
    """Identify parameters that exist but are never accessed/used."""
    issues: List[ParameterAuditIssue] = []

    for key in observed:
        was_accessed = key in manager._access_history
        if not was_accessed:
            issue = ParameterAuditIssue(
                key=key,
                issue_type=AuditIssueType.ORPHAN,
                severity=AuditSeverity.WARNING,
                message=f"Parameter '{key}' exists but has never been accessed",
                context={
                    "source": "runtime" if key in manager.parameters else "storage",
                    "was_accessed": False,
                },
            )
            issues.append(issue)

    return issues


def _scan_duplicate_registrations(
    manager,
    normalize_key: Optional[Callable[[str], str]] = None,
) -> List[ParameterAuditIssue]:
    """Identify parameters with duplicate or conflicting registrations."""
    normalize_key = normalize_key or (lambda value: value)
    issues: List[ParameterAuditIssue] = []

    from collections import defaultdict

    canonical_map = defaultdict(list)
    for key in manager.parameters.keys():
        canonical_key = normalize_key(key)
        canonical_map[canonical_key].append(key)

    for canonical_key, keys in canonical_map.items():
        if len(keys) <= 1:
            continue


        specs = [manager.parameters[key] for key in keys]
        values = [spec.value for spec in specs]
        units = [spec.units for spec in specs]

        value_set = builtins.set(str(value) for value in values)
        unit_set = builtins.set(units)

        severity = AuditSeverity.WARNING
        if len(value_set) > 1 or len(unit_set) > 1:
            severity = AuditSeverity.ERROR

        issue = ParameterAuditIssue(
            key=canonical_key,
            issue_type=AuditIssueType.DUPLICATE,
            severity=severity,
            message=(
                f"Multiple parameter entries resolve to canonical key '{canonical_key}'. "
                "Please consolidate definitions to avoid ambiguity."
            ),
            context={
                "all_keys": keys,
                "values": {key: manager.parameters[key].value for key in keys},
                "units": {key: manager.parameters[key].units for key in keys},
            },
        )
        issues.append(issue)

    basename_check_enabled = True
    audit_toggle = manager.parameters.get("validation.numerical.audit.basename_duplicates_enabled")
    if audit_toggle is not None:
        basename_check_enabled = bool(audit_toggle.value)
    basename_policy = "strict"
    policy_spec = manager.parameters.get("validation.numerical.audit.basename_duplicates_policy")
    if policy_spec is not None:
        basename_policy = str(policy_spec.value).strip().lower()
    allow_domain_sets: List[Set[str]] = []
    allow_domain_spec = manager.parameters.get("validation.numerical.audit.basename_duplicates_allow_domain_sets")
    allow_prefix_groups: List[List[str]] = []
    allow_prefix_spec = manager.parameters.get("validation.numerical.audit.basename_duplicates_allow_prefix_groups")
    allow_basenames: List[str] = []
    allow_basenames_spec = manager.parameters.get("validation.numerical.audit.basename_duplicates_allow_basenames")

    if basename_check_enabled:
        def _normalize_list(value: Any) -> List[Any]:
            if value is None:
                return []
            if isinstance(value, str):
                cleaned = [item.strip() for item in value.split(";") if item.strip()]
                return cleaned
            if isinstance(value, (list, tuple, set)):
                return list(value)
            return [value]

        def _parse_domain_sets(value: Any) -> List[Set[str]]:
            parsed_sets = []
            for entry in _normalize_list(value):
                if isinstance(entry, str):
                    domains = [item.strip().lower() for item in entry.split(",") if item.strip()]
                else:
                    domains = [str(item).strip().lower() for item in _normalize_list(entry)]
                if domains:
                    parsed_sets.append(builtins.set(domains))
            return parsed_sets

        def _parse_prefix_groups(value: Any) -> List[List[str]]:
            parsed_groups = []
            for entry in _normalize_list(value):
                if isinstance(entry, str):
                    prefixes = [item.strip() for item in entry.split(",") if item.strip()]
                else:
                    prefixes = [str(item).strip() for item in _normalize_list(entry)]
                if prefixes:
                    parsed_groups.append(prefixes)
            return parsed_groups

        def _matches_prefix_group(candidate_keys: List[str], prefixes: List[str]) -> bool:
            for candidate in candidate_keys:
                matched = any(
                    candidate == prefix or candidate.startswith(prefix + ".")
                    for prefix in prefixes
                )
                if not matched:
                    return False
            return True

        def _common_prefix_len(candidate_keys: List[str]) -> int:
            split_keys = [key.split(".") for key in candidate_keys]
            min_len = min(len(parts) for parts in split_keys)
            prefix_len = 0
            for index in range(min_len):
                segment = split_keys[0][index]
                if all(parts[index] == segment for parts in split_keys[1:]):
                    prefix_len += 1
                else:
                    break
            return prefix_len

        def _is_structured_basename_group(candidate_keys: List[str]) -> bool:
            if _common_prefix_len(candidate_keys) >= 1:
                return True
            split_keys = [key.split(".") for key in candidate_keys]
            lengths = {len(parts) for parts in split_keys}
            if len(lengths) != 1:
                return False
            length = lengths.pop()
            varying_positions = []
            for index in range(length - 1):
                segments = {parts[index] for parts in split_keys}
                if len(segments) > 1:
                    varying_positions.append(index)
            return len(varying_positions) == 1

        if allow_domain_spec is not None:
            allow_domain_sets = _parse_domain_sets(allow_domain_spec.value)
        if allow_prefix_spec is not None:
            allow_prefix_groups = _parse_prefix_groups(allow_prefix_spec.value)
        if allow_basenames_spec is not None:
            allow_basenames = [str(item).strip().lower() for item in _normalize_list(allow_basenames_spec.value)]

        basename_map: Dict[str, List[str]] = defaultdict(list)
        for key in manager.parameters.keys():
            basename = key.split(".")[-1]
            basename_map[basename].append(key)

        for basename, keys in basename_map.items():
            if len(keys) <= 1:
                continue
            specs = [manager.parameters[key] for key in keys]
            values = [spec.value for spec in specs]
            units = [spec.units for spec in specs]

            value_set = builtins.set(str(value) for value in values)
            unit_set = builtins.set(units)
            if len(value_set) == 1 and len(unit_set) == 1:
                continue
            if basename_policy == "structured" and _is_structured_basename_group(keys):
                continue
            if allow_basenames and basename.lower() in allow_basenames:
                continue
            if allow_domain_sets:
                key_domains = builtins.set(key.split(".")[0].lower() for key in keys)
                if any(key_domains == allowed for allowed in allow_domain_sets):
                    continue
            if allow_prefix_groups:
                if any(_matches_prefix_group(keys, prefixes) for prefixes in allow_prefix_groups):
                    continue

            severity = AuditSeverity.ERROR if (len(value_set) > 1 or len(unit_set) > 1) else AuditSeverity.WARNING
            if basename_policy == "structured":
                severity = AuditSeverity.WARNING

            issue = ParameterAuditIssue(
                key=basename,
                issue_type=AuditIssueType.DUPLICATE,
                severity=severity,
                message=(
                    f"Parameter basename '{basename}' is registered in multiple namespaces with "
                    "conflicting definitions; consolidate to a single source of truth."
                ),
                context={
                    "all_keys": keys,
                    "values": {key: manager.parameters[key].value for key in keys},
                    "units": {key: manager.parameters[key].units for key in keys},
                },
            )
            issues.append(issue)

    return issues


def _scan_validation_errors(manager, namespace_filter: Optional[str] = None) -> List[ParameterAuditIssue]:
    """Scan for parameters that fail validation."""
    issues: List[ParameterAuditIssue] = []

    keys_to_check = manager.list(namespace=namespace_filter)
    validation_errors = manager.validate_parameters(keys_to_check)

    for error_dict in validation_errors:
        key = error_dict["parameter"]
        if namespace_filter and not _in_namespace(key, namespace_filter):
            continue
        error = error_dict["error"]

        if "nan" in error.lower() or "inf" in error.lower():
            severity = AuditSeverity.CRITICAL
        elif "below minimum" in error.lower() or "above maximum" in error.lower():
            severity = AuditSeverity.ERROR
        else:
            severity = AuditSeverity.WARNING

        issue = ParameterAuditIssue(
            key=key,
            issue_type=AuditIssueType.VALIDATION_ERROR,
            severity=severity,
            message=error,
            context={
                "current_value": manager.get(key),
                "spec": manager.parameters.get(key).to_dict() if key in manager.parameters else {},
            },
        )
        issues.append(issue)

    return issues


def _scan_unit_consistency(manager) -> List[ParameterAuditIssue]:
    """Check for unit consistency across related parameters."""
    from collections import defaultdict

    issues: List[ParameterAuditIssue] = []
    excluded_dimensions: Set[str] = {
        "configuration",
        "unit_conversion",
    }
    unit_aliases = {
        "year-1": "yr-1",
        "day-1": "day-1",
        "mm hr-1": "mm h-1",
    }
    equivalent_unit_families: Dict[str, Set[str]] = {
        "inverse_length": {"m-1", "cm-1", "mm-1"},
        "inverse_pressure": {"Pa-1", "kPa-1", "hPa-1", "MPa-1"},
    }

    by_dimension = defaultdict(list)
    for key, spec in manager.parameters.items():
        if spec.dimension:
            normalized_unit = unit_aliases.get(spec.units, spec.units)
            by_dimension[spec.dimension].append((key, normalized_unit))

    for dimension, params in by_dimension.items():
        if dimension in excluded_dimensions:
            continue

        units_seen = defaultdict(list)
        for key, unit in params:
            if unit:
                units_seen[unit].append(key)

        if len(units_seen) <= 1:
            continue

        if (
            dimension in equivalent_unit_families
            and set(units_seen.keys()).issubset(equivalent_unit_families[dimension])
        ):
            continue

        # Only flag clear binary inconsistencies where a dominant unit exists.
        if len(units_seen) != 2:
            continue

        dominant_unit, dominant_keys = max(
            units_seen.items(),
            key=lambda item: len(item[1]),
        )
        if len(dominant_keys) < 2:
            continue

        for unit, keys in units_seen.items():
            if unit == dominant_unit or len(keys) != 1:
                continue
            issue = ParameterAuditIssue(
                key=keys[0],
                issue_type=AuditIssueType.UNIT_MISMATCH,
                severity=AuditSeverity.WARNING,
                message=(
                    f"Unit '{unit}' differs from dominant unit '{dominant_unit}' "
                    f"in dimension '{dimension}'"
                ),
                context={
                    "dimension": dimension,
                    "common_units": [dominant_unit],
                    "parameters_with_unit": keys,
                    "dominant_unit_count": len(dominant_keys),
                },
            )
            issues.append(issue)

    return issues


def export_audit_report(
    report: ParameterAuditReport,
    format: str = "table",
    output_path: Optional[Path] = None,
) -> Optional[str]:
    """
    Export audit report in various formats.

    Parameters:
        report : ParameterAuditReport
            Report to export
        format : str
            Output format: 'table', 'json', 'yaml', 'markdown'
        output_path : Optional[Path]
            File to save, return string if None
    """
    if format == "table":
        lines = ["Parameter Audit Report", "=" * 60]
        lines.append(f"Created: {report.created_at.isoformat()}")
        lines.append(f"Expected: {report.expected_total}, Observed: {report.observed_total}")
        lines.append("")

        type_summary = report.summarize_by_type()
        if type_summary:
            lines.append("Issues by Type:")
            for issue_type, count in type_summary.items():
                lines.append(f"  {issue_type}: {count}")
            lines.append("")

        severity_summary = report.summarize_by_severity()
        if severity_summary:
            lines.append("Issues by Severity:")
            for severity, count in severity_summary.items():
                lines.append(f"  {severity}: {count}")
            lines.append("")

        if report.issues:
            lines.append("Detailed Issues:")
            lines.append("-" * 60)
            for issue in report.issues[:20]:
                lines.append(f"[{issue.severity.value.upper()}] {issue.key}")
                lines.append(f"  Type: {issue.issue_type.value}")
                lines.append(f"  Message: {issue.message}")
                lines.append("")

            if len(report.issues) > 20:
                lines.append(f"... and {len(report.issues) - 20} more issues")
                lines.append("")

        if report.notes:
            lines.append("Notes:")
            for note in report.notes:
                lines.append(f"  - {note}")

        formatted = "\n".join(lines)

    elif format == "json":
        formatted = json.dumps(report.to_dict(), indent=2)

    elif format == "yaml":
        if YAML_AVAILABLE:
            import yaml
            formatted = yaml.dump(report.to_dict(), default_flow_style=False)
        else:
            raise ImportError("PyYAML required for YAML export")

    elif format == "markdown":
        lines = ["# Parameter Audit Report", ""]
        lines.append(f"**Generated:** {report.created_at.isoformat()}")
        lines.append(f"**Parameters:** Expected {report.expected_total}, Observed {report.observed_total}")
        lines.append("")

        lines.append("## Summary")
        lines.append("")

        type_summary = report.summarize_by_type()
        if type_summary:
            lines.append("### Issues by Type")
            lines.append("| Type | Count |")
            lines.append("|------|-------|")
            for issue_type, count in type_summary.items():
                lines.append(f"| {issue_type} | {count} |")
            lines.append("")

        severity_summary = report.summarize_by_severity()
        if severity_summary:
            lines.append("### Issues by Severity")
            lines.append("| Severity | Count |")
            lines.append("|----------|-------|")
            for severity, count in severity_summary.items():
                lines.append(f"| {severity} | {count} |")
            lines.append("")

        critical = report.filter_by_severity(AuditSeverity.CRITICAL)
        if critical:
            lines.append("## Critical Issues")
            for issue in critical:
                lines.append(f"- **{issue.key}**: {issue.message}")
            lines.append("")

        if report.missing_keys:
            lines.append("## Missing Parameters")
            for key in report.missing_keys[:20]:
                lines.append(f"- `{key}`")
            if len(report.missing_keys) > 20:
                lines.append(f"- ... and {len(report.missing_keys) - 20} more")
            lines.append("")

        formatted = "\n".join(lines)

    else:
        raise ValueError(f"Unsupported format: {format}")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(formatted)
        return None
    else:
        return formatted


_STRUCTURAL_PARAMETER_TAGS = {
    "hodge",
    "hodge_star",
    "onsager",
    "mobility",
    "metric",
    "dec",
    "iphs",
    "port",
    "structural",
    "regime_number",
    "thermodynamic",
}


def _get_spec_field(spec: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(spec, dict):
        return spec.get(field_name, default)
    return getattr(spec, field_name, default)


def _spec_tags(spec: Any) -> Set[str]:
    tags = _get_spec_field(spec, "tags", set()) or set()
    if isinstance(tags, str):
        tags = {tags}
    return {str(tag).strip().lower() for tag in tags if str(tag).strip()}


def _spec_key(spec: Any) -> str:
    return str(_get_spec_field(spec, "key", _get_spec_field(spec, "name", "unknown_parameter")))


def _is_structural_parameter(spec: Any) -> bool:
    tags = _spec_tags(spec)
    if tags.intersection(_STRUCTURAL_PARAMETER_TAGS):
        return True
    category = _get_spec_field(spec, "category", "")
    category_value = getattr(category, "value", category)
    return str(category_value).strip().lower() in _STRUCTURAL_PARAMETER_TAGS


def _add_structural_metadata_issue(
    report: ParameterAuditReport,
    *,
    key: str,
    field_name: str,
    message: str,
    strict: bool,
) -> None:
    report.add_issue(
        ParameterAuditIssue(
            key=key,
            issue_type=AuditIssueType.METADATA_GAP,
            severity=AuditSeverity.ERROR if strict else AuditSeverity.WARNING,
            message=message,
            context={"field": field_name, "audit": "structural_parameter"},
            references=[
                "RELIC-IPHS-DEC Typed Cochain Morphism Enhancement Plan",
                "Wilkinson et al. (2016), FAIR Guiding Principles",
            ],
        )
    )


def _audit_structural_parameter_spec(
    report: ParameterAuditReport,
    spec: Any,
    *,
    strict: bool,
) -> None:
    key = _spec_key(spec)
    units = _get_spec_field(spec, "units")
    has_bounds = "bounds" in spec if isinstance(spec, dict) else hasattr(spec, "bounds")
    bounds_error = (
        parameter_bounds_error(_get_spec_field(spec, "bounds"), _get_spec_field(spec, "dtype"))
        if has_bounds
        else "no bounds field"
    )
    references = _get_spec_field(spec, "references", []) or []
    provenance = _get_spec_field(spec, "provenance", {}) or {}
    tags = _spec_tags(spec)

    if not units:
        _add_structural_metadata_issue(
            report,
            key=key,
            field_name="units",
            message=f"Structural parameter '{key}' must declare units.",
            strict=True,
        )
    if bounds_error is not None:
        _add_structural_metadata_issue(
            report,
            key=key,
            field_name="bounds",
            message=f"Structural parameter '{key}' should declare bounds: {bounds_error}.",
            strict=strict,
        )
    if not references:
        _add_structural_metadata_issue(
            report,
            key=key,
            field_name="references",
            message=f"Structural parameter '{key}' must declare references.",
            strict=strict,
        )
    if not provenance:
        _add_structural_metadata_issue(
            report,
            key=key,
            field_name="provenance",
            message=f"Structural parameter '{key}' must declare provenance.",
            strict=strict,
        )
    if "regime_number" in tags and str(units).strip().lower() != "dimensionless":
        report.add_issue(
            ParameterAuditIssue(
                key=key,
                issue_type=AuditIssueType.UNIT_MISMATCH,
                severity=AuditSeverity.ERROR,
                message=f"Dimensionless regime parameter '{key}' must use dimensionless units.",
                context={"field": "units", "audit": "structural_parameter"},
                references=["RELIC-IPHS-DEC structural parameter audit policy"],
            )
        )


def _audit_generated_structures(
    report: ParameterAuditReport,
    *,
    hodge_specs: Optional[Iterable[Any]] = None,
    iphs_specs: Optional[Iterable[Any]] = None,
) -> None:
    """Record a rejected generated structure as a CRITICAL audit issue.

    HodgeStarSpec.validate and IPHSStructureSpec.validate reject an invalid
    structure (asymmetric, indefinite, non-skew, degenerate) with ValueError;
    that rejection is the audit finding and marks the report failed. Any other
    error (an internal initialization failure, a malformed spec object) is not
    a structural finding and propagates.
    """
    for spec in hodge_specs or []:
        key = str(getattr(spec, "name", "hodge_structure"))
        try:
            spec.validate(passive=True)
        except ValueError as exc:
            report.add_issue(
                ParameterAuditIssue(
                    key=key,
                    issue_type=AuditIssueType.VALIDATION_ERROR,
                    severity=AuditSeverity.CRITICAL,
                    message=str(exc),
                    context={"structure": "hodge", "audit": "structural_parameter"},
                    references=["Hirani (2003), Discrete Exterior Calculus"],
                )
            )
    for spec in iphs_specs or []:
        key = str(getattr(spec, "name", "iphs_structure"))
        try:
            spec.validate()
        except ValueError as exc:
            report.add_issue(
                ParameterAuditIssue(
                    key=key,
                    issue_type=AuditIssueType.VALIDATION_ERROR,
                    severity=AuditSeverity.CRITICAL,
                    message=str(exc),
                    context={"structure": "iphs", "audit": "structural_parameter"},
                    references=["van der Schaft and Maschke (1995), Port-Hamiltonian systems"],
                )
            )


def run_structural_parameter_audit(
    parameter_specs: Iterable[Any],
    *,
    hodge_specs: Optional[Iterable[Any]] = None,
    iphs_specs: Optional[Iterable[Any]] = None,
    strict: bool = False,
) -> ParameterAuditReport:
    """Audit RELIC-IPHS-DEC structural parameters and generated structures.

    References:
      - LeBauer et al. (2013), PEcAn parameter provenance workflow.
      - Wilkinson et al. (2016), FAIR Guiding Principles.
    Mathematical expression:
      Structural validity requires parameter metadata completeness plus
      generated operator checks such as H = H^T >= 0 and M = M^T >= 0.
    Parameter meanings:
      parameter_specs: ParameterSpec-like objects or dictionaries.
      hodge_specs: generated Hodge operators assembled from parameters.
      iphs_specs: generated IPHS structures assembled from parameters.
      strict: escalate metadata gaps from warnings to errors.
    """

    structural_specs = [spec for spec in parameter_specs if _is_structural_parameter(spec)]
    report = ParameterAuditReport(
        expected_total=len(structural_specs),
        observed_total=len(structural_specs),
        created_at=datetime.now(timezone.utc),
        scanned_namespaces=["structural"],
    )
    for spec in structural_specs:
        _audit_structural_parameter_spec(report, spec, strict=strict)
    _audit_generated_structures(
        report,
        hodge_specs=hodge_specs,
        iphs_specs=iphs_specs,
    )
    report.notes.append("Structural parameter audit completed")
    return report


__all__ = [
    "AuditIssueType",
    "AuditSeverity",
    "ParameterAuditIssue",
    "ParameterAuditReport",
    "StructuralParameterAudit",
    "load_category",
    "run_registry_audit",
    "run_structural_parameter_audit",
    "get_missing_requests",
    "clear_missing_requests",
    "export_audit_report",
]
