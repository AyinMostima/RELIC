"""
Unified audit record management for coupling operations.

Provides consistent tracking and reporting of scaling operations,
conservation checks, and metadata updates.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import torch

if TYPE_CHECKING:
    from relic.parameters import UnifiedParameterManager

from relic.parameters.namespaces import ParameterNamespace, resolve_namespace
from relic.parameters import get_global_manager

logger = logging.getLogger(__name__)


def _get_parameter_manager_required(
    parameter_manager: Optional["UnifiedParameterManager"] = None,
) -> "UnifiedParameterManager":
    if parameter_manager is not None:
        return parameter_manager
    try:
        return get_global_manager()
    except RuntimeError as exc:
        raise RuntimeError(
            "ParameterManager is required for audit records but could not be resolved. "
            "Ensure the parameter system is initialized before constructing AuditRecord."
        ) from exc


from .conservation_checker import ConservationDiagnostics as _ConservationDiagnostics


class AuditLevel(Enum):
    """Audit detail levels."""
    MINIMAL = "minimal"      # Basic operation info
    STANDARD = "standard"    # Standard metrics
    DETAILED = "detailed"    # Full diagnostics


@dataclass
class AuditRecord:
    """
    Comprehensive audit record for scaling operations.
    
    Attributes:
        operation : str
            Operation type (aggregate, disaggregate, regrid)
        timestamp : datetime
            When the operation occurred
        source_scale : str
            Source scale name
        target_scale : str
            Target scale name
        variables : List[str]
            Variables processed
        mode : str
            Scaling mode used
        strategy : str
            Strategy applied
        conservation : Dict[str, _ConservationDiagnostics]
            Conservation diagnostics per variable
        mass_balance_error : float
            Maximum mass balance error
        relative_error : float
            Maximum relative error
        cf_compliant : bool
            Legacy-named dataset-level field; always false because not evaluated
        metadata_updates : Dict[str, Any]
            CF metadata changes
        warnings : List[str]
            Warning messages
        errors : List[str]
            Error messages
        performance : Dict[str, float]
            Performance metrics
        custom_data : Dict[str, Any]
            Additional custom data
    """
    operation: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_scale: str = ""
    target_scale: str = ""
    variables: List[str] = field(default_factory=list)
    mode: str = ""
    strategy: str = ""
    conservation: Dict[str, _ConservationDiagnostics] = field(default_factory=dict)
    mass_balance_error: float = 0.0
    relative_error: float = 0.0
    cf_compliant: bool = False
    metadata_updates: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    performance: Dict[str, float] = field(default_factory=dict)
    custom_data: Dict[str, Any] = field(default_factory=dict)
    parameter_manager: Optional["UnifiedParameterManager"] = field(default=None, repr=False)
    
    def __post_init__(self):
        if not isinstance(self.cf_compliant, bool):
            raise TypeError("cf_compliant must be bool")
        if self.cf_compliant:
            raise ValueError(
                "AuditRecord cannot claim dataset-level CF conformance"
            )
        if self.metadata_updates.get("dataset_conformance_checked") not in {
            None,
            False,
        }:
            raise ValueError(
                "AuditRecord dataset_conformance_checked must remain false"
            )
        self.metadata_updates.setdefault("dataset_conformance_checked", False)
        self.metadata_updates.setdefault("cf_validation_scope", "not_evaluated")
        self._param_manager = _get_parameter_manager_required(self.parameter_manager)
        self._warnings_default: Optional[int] = None
        self._errors_default: Optional[int] = None
        self._warnings_default = self._get_warnings_default()
        self._errors_default = self._get_errors_default()
    
    def _fetch_audit_default(self, key: str) -> int:
        manager = getattr(self, "_param_manager", None)
        if manager is None:
            manager = _get_parameter_manager_required()
            self._param_manager = manager
        resolved_ns = resolve_namespace(ParameterNamespace.COUPLING_AUDIT)
        try:
            value = manager.get(
                key,
                namespace=resolved_ns,
                required=True,
            )
            if isinstance(value, torch.Tensor):
                return int(value.detach().cpu().item())
            return int(value)
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:  # noqa: BLE001 - explicit audit trail
            raise RuntimeError(
                f"Audit default parameter lookup failed for {resolved_ns}.{key}: {exc}"
            ) from exc

    def _get_warnings_default(self) -> int:
        """Lazy accessor for warnings default to avoid import-time parameter lookups."""
        if self._warnings_default is None:
            self._warnings_default = self._fetch_audit_default("warnings_default")
        return self._warnings_default

    def _get_errors_default(self) -> int:
        """Lazy accessor for errors default to avoid import-time parameter lookups."""
        if self._errors_default is None:
            self._errors_default = self._fetch_audit_default("errors_default")
        return self._errors_default
    
    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(f"[{datetime.now(timezone.utc).isoformat()}] {message}")
    
    def add_error(self, message: str):
        """Add an error message."""
        self.errors.append(f"[{datetime.now(timezone.utc).isoformat()}] {message}")
    
    def add_conservation_check(
        self,
        variable: str,
        diagnostics: _ConservationDiagnostics
    ):
        """Add conservation diagnostics for a variable."""
        self.conservation[variable] = diagnostics
        
        # Update summary metrics
        if diagnostics.absolute_error > self.mass_balance_error:
            self.mass_balance_error = diagnostics.absolute_error
        if diagnostics.relative_error > self.relative_error:
            self.relative_error = diagnostics.relative_error
    
    def add_performance_metric(self, name: str, value: float):
        """Add a performance metric."""
        self.performance[name] = value
    
    def set_metadata_update(self, key: str, value: Any):
        """Record a metadata update."""
        self.metadata_updates[key] = value

    def add_topology_diagnostics(
        self,
        invariant_drift: Optional[float] = None,
        betti_number: Optional[int] = None,
        graph_rank: Optional[int] = None,
        graph_nullity: Optional[int] = None,
    ) -> None:
        """Store topology and Noether diagnostics in custom_data."""
        self.custom_data["topology"] = {
            "invariant_drift": invariant_drift,
            "betti_number": betti_number,
            "graph_rank": graph_rank,
            "graph_nullity": graph_nullity,
        }

    def add_identifiability_diagnostics(
        self,
        jacobian_rank: Optional[int] = None,
        jacobian_nullity: Optional[int] = None,
        condition_number: Optional[float] = None,
    ) -> None:
        """Store identifiability diagnostics in custom_data."""
        self.custom_data["identifiability"] = {
            "jacobian_rank": jacobian_rank,
            "jacobian_nullity": jacobian_nullity,
            "condition_number": condition_number,
        }

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """Convert values to JSON-safe primitives."""
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {k: AuditRecord._serialize_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [AuditRecord._serialize_value(v) for v in value]
        if isinstance(value, tuple):
            return [AuditRecord._serialize_value(v) for v in value]
        return value

    def to_dict(self) -> Dict[str, Any]:
        """Serialize audit record to a JSON-safe dictionary."""
        conservation_payload: Dict[str, Any] = {}
        for variable, diagnostics in self.conservation.items():
            conservation_payload[variable] = {
                "property_name": diagnostics.property_name,
                "is_conserved": diagnostics.is_conserved,
                "absolute_error": diagnostics.absolute_error,
                "relative_error": diagnostics.relative_error,
                "source_total": diagnostics.source_total,
                "target_total": diagnostics.target_total,
                "enforcement_applied": diagnostics.enforcement_applied,
                "correction_factor": diagnostics.correction_factor,
            }
        return {
            "operation": self.operation,
            "timestamp": self.timestamp.isoformat(),
            "source_scale": self.source_scale,
            "target_scale": self.target_scale,
            "variables": list(self.variables),
            "mode": self.mode,
            "strategy": self.strategy,
            "conservation": self._serialize_value(conservation_payload),
            "mass_balance_error": self._serialize_value(self.mass_balance_error),
            "relative_error": self._serialize_value(self.relative_error),
            "cf_compliant": self.cf_compliant,
            "metadata_updates": self._serialize_value(self.metadata_updates),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "performance": self._serialize_value(self.performance),
            "custom_data": self._serialize_value(self.custom_data),
        }
    
    def get_summary(self, level: AuditLevel = AuditLevel.STANDARD) -> Dict[str, Any]:
        """
        Get audit summary at specified detail level.
        
        Parameters:
            level : AuditLevel
                Detail level for summary
                
        Returns:
            Dict[str, Any]: Audit summary
        """
        if level == AuditLevel.MINIMAL:
            return {
                'operation': self.operation,
                'timestamp': self.timestamp.isoformat(),
                'success': len(self.errors) == 0,
                'n_warnings': len(self.warnings),
                'max_error': self.relative_error
            }
        
        elif level == AuditLevel.STANDARD:
            return {
                'operation': self.operation,
                'timestamp': self.timestamp.isoformat(),
                'scales': f"{self.source_scale} -> {self.target_scale}",
                'variables': self.variables,
                'mode': self.mode,
                'strategy': self.strategy,
                'conservation': {
                    'max_error': self.relative_error,
                    'mass_balance': self.mass_balance_error,
                    'n_enforced': sum(1 for d in self.conservation.values() 
                                    if d.enforcement_applied)
                },
                'cf_compliant': self.cf_compliant,
                'warnings': len(self.warnings),
                'errors': len(self.errors),
                'performance': self.performance
            }
        
        else:  # DETAILED
            return {
                'operation': self.operation,
                'timestamp': self.timestamp.isoformat(),
                'source_scale': self.source_scale,
                'target_scale': self.target_scale,
                'variables': self.variables,
                'mode': self.mode,
                'strategy': self.strategy,
                'conservation': {
                    var: {
                        'conserved': diag.is_conserved,
                        'abs_error': diag.absolute_error,
                        'rel_error': diag.relative_error,
                        'enforced': diag.enforcement_applied,
                        'correction': diag.correction_factor
                    }
                    for var, diag in self.conservation.items()
                },
                'mass_balance_error': self.mass_balance_error,
                'relative_error': self.relative_error,
                'cf_compliant': self.cf_compliant,
                'metadata_updates': self.metadata_updates,
                'warnings': self.warnings,
                'errors': self.errors,
                'performance': self.performance,
                'custom_data': self.custom_data
            }
    
    def to_string(self, level: AuditLevel = AuditLevel.STANDARD) -> str:
        """
        Format audit record as string.
        
        Parameters:
            level : AuditLevel
                Detail level
                
        Returns:
            str: Formatted audit record
        """
        summary = self.get_summary(level)
        
        lines = [
            f"=== Audit Record: {self.operation} ===",
            f"Timestamp: {summary['timestamp']}"
        ]
        
        if level != AuditLevel.MINIMAL:
            lines.extend([
                f"Scales: {summary.get('scales', 'N/A')}",
                f"Variables: {', '.join(summary.get('variables', []))}",
                f"Mode: {summary.get('mode', 'N/A')}",
                f"Strategy: {summary.get('strategy', 'N/A')}"
            ])
        
        if 'conservation' in summary:
            cons = summary['conservation']
            if isinstance(cons, dict) and 'max_error' in cons:
                lines.append(f"Max Conservation Error: {cons['max_error']:.2e}")
        
        warnings_count = summary.get('warnings', self._get_warnings_default())
        if warnings_count > 0:
            lines.append(f"Warnings: {warnings_count}")
        
        errors_count = summary.get('errors', self._get_errors_default())
        if errors_count > 0:
            lines.append(f"Errors: {errors_count}")
        
        return "\n".join(lines)


class AuditCollector:
    """
    Collects and manages audit records.
    
    Provides centralized audit trail management with
    querying and reporting capabilities.
    """
    
    def __init__(self, max_records: int = 1000):
        """
        Initialize audit collector.
        
        Audit records are read through this collector (get_recent, get_by_operation, get_errors,
        get_conservation_violations, summaries) and its optional JSONL sink; they are not sent
        into the StateManager diagnostics channel, whose stored diagnostics are real scalars.

        Parameters:
            max_records : int
                Maximum records to keep (FIFO)
        """
        self.records: List[AuditRecord] = []
        self.max_records = max_records
        self._jsonl_enabled = False
        self._jsonl_path: Optional[Path] = None
        self._configure_jsonl_sink()
    
    def add_record(self, record: AuditRecord):
        """Add an audit record."""
        self.records.append(record)
        
        # Maintain size limit
        if len(self.records) > self.max_records:
            self.records.pop(0)
        self._emit_jsonl(record)
    
    def get_recent(self, n: int = 10) -> List[AuditRecord]:
        """Get n most recent records."""
        return self.records[-n:] if self.records else []
    
    def get_by_operation(self, operation: str) -> List[AuditRecord]:
        """Get records for a specific operation type."""
        return [r for r in self.records if r.operation == operation]
    
    def get_errors(self) -> List[AuditRecord]:
        """Get records with errors."""
        return [r for r in self.records if r.errors]
    
    def get_warnings(self) -> List[AuditRecord]:
        """Get records with warnings."""
        return [r for r in self.records if r.warnings]
    
    def get_conservation_violations(
        self,
        threshold: float = 1e-9
    ) -> List[AuditRecord]:
        """Get records with conservation violations."""
        violations = []
        for record in self.records:
            if record.relative_error > threshold:
                violations.append(record)
        return violations
    
    def generate_report(
        self,
        level: AuditLevel = AuditLevel.STANDARD
    ) -> str:
        """
        Generate audit report.
        
        Parameters:
            level : AuditLevel
                Report detail level
                
        Returns:
            str: Formatted report
        """
        lines = [
            "=== Coupling Operations Audit Report ===",
            f"Total Records: {len(self.records)}",
            f"Report Generated: {datetime.now(timezone.utc).isoformat()}",
            ""
        ]
        
        # Summary statistics
        n_errors = sum(1 for r in self.records if r.errors)
        n_warnings = sum(1 for r in self.records if r.warnings)
        n_violations = len(self.get_conservation_violations())
        
        lines.extend([
            "Summary:",
            f"  Records with Errors: {n_errors}",
            f"  Records with Warnings: {n_warnings}",
            f"  Conservation Violations: {n_violations}",
            ""
        ])
        
        # Recent operations
        lines.append("Recent Operations:")
        for record in self.get_recent(5):
            lines.append(f"  - {record.timestamp.isoformat()}: {record.operation}")
        
        if level != AuditLevel.MINIMAL:
            # Add detailed sections
            if n_errors > 0:
                lines.extend(["", "Recent Errors:"])
                for record in self.get_errors()[-3:]:
                    lines.append(f"  {record.timestamp.isoformat()}: {record.errors[0]}")
            
            if n_violations > 0:
                lines.extend(["", "Conservation Violations:"])
                for record in self.get_conservation_violations()[-3:]:
                    lines.append(
                        f"  {record.timestamp.isoformat()}: "
                        f"{record.operation} (error={record.relative_error:.2e})"
                    )
        
        return "\n".join(lines)
    
    def clear(self):
        """Clear all audit records."""
        self.records.clear()

    def _configure_jsonl_sink(self) -> None:
        """Load JSONL sink configuration from ParameterManager."""
        manager = _get_parameter_manager_required()
        resolved_ns = resolve_namespace(ParameterNamespace.COUPLING_AUDIT)
        enabled_value = manager.get(
            "jsonl_enabled",
            namespace=resolved_ns,
            units="dimensionless",
            expected_type=bool,
            required=True,
        )
        path_value = manager.get(
            "jsonl_path",
            namespace=resolved_ns,
            units="dimensionless",
            expected_type=str,
            required=False,
        )

        self._jsonl_enabled = bool(enabled_value) if enabled_value is not None else False
        if path_value:
            self._jsonl_path = Path(path_value)

    def _emit_jsonl(self, record: AuditRecord) -> None:
        """Append audit record to JSONL sink when enabled."""
        # The enabled JSONL sink writes every record; a write failure raises (no dropped audit).
        if not self._jsonl_enabled:
            return
        if self._jsonl_path is None:
            logger.debug("Audit JSONL sink enabled but jsonl_path is not configured.")
            return
        self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        payload = record.to_dict()
        with self._jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def merge_audits(*audits: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple audit dictionaries.
    
    Parameters:
        *audits : Dict[str, Any]
            Audit dictionaries to merge
            
    Returns:
        Dict[str, Any]: Merged audit
    """
    merged = {
        'warnings': [],
        'errors': [],
        'conservation': {},
        'performance': {},
        'metadata': {}
    }
    
    for audit in audits:
        if not audit:
            continue
        
        # Merge lists
        merged['warnings'].extend(audit.get('warnings', []))
        merged['errors'].extend(audit.get('errors', []))
        
        # Merge dicts
        merged['conservation'].update(audit.get('conservation', {}))
        merged['performance'].update(audit.get('performance', {}))
        merged['metadata'].update(audit.get('metadata', {}))
    
    return merged




