"""Diagnostics helpers for RELICSimulator."""

from __future__ import annotations

import copy
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from typing import TYPE_CHECKING

from ..functor import FunctorBase
from ..abstractions import StateDict, FluxDict

if TYPE_CHECKING:
    from ...data.io.writer import WriteResult

logger = logging.getLogger(__name__)


def collect_functor_diagnostics(
    simulator: Any,
    functor: FunctorBase,
    state: StateDict,
    updates: FluxDict,
) -> None:
    """
    Collect diagnostics from individual functors for aggregation in StateManager.
    """
    if simulator.state_manager is None:
        return

    diagnose_conservation = getattr(functor, "diagnose_conservation", None)
    if callable(diagnose_conservation):
        # A failing diagnosis or ingestion propagates; an empty result stores
        # nothing (no availability or error sentinel is written).
        diagnostics: Any = diagnose_conservation(state, updates)
        if diagnostics:
            simulator.state_manager.ingest_module_diagnostics(functor.name, diagnostics)
            logger.debug("Collected %s diagnostics from %s", len(diagnostics), functor.name)
    else:
        logger.debug(
            "Functor %s does not provide diagnose_conservation method",
            functor.name,
        )


def capture_diagnostic_export(
    simulator: Any, *, output_path: Optional[Any] = None,
    reference_date: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Prepare and freeze diagnostic output through the existing I/O owners."""
    from .checkpoint import _decode_checkpoint_value, _encode_checkpoint_value
    from ...data.io.diagnostic_writer import DiagnosticWriter
    from ...data.io.writer import DataWriter, WriteConfig

    if simulator.state_manager is None:
        raise RuntimeError("Diagnostic export requires StateManager")
    target = output_path if output_path is not None else simulator.diagnostic_output_path
    if target is None:
        raise RuntimeError("Diagnostic export requires an output path")
    reference = reference_date if reference_date is not None else getattr(
        simulator, "diagnostic_reference_date", "1850-01-01",
    )
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("Diagnostic reference_date must be a nonempty string")
    history = simulator.state_manager.conservation_history
    if not history:
        return None
    writer = getattr(simulator, "diagnostic_writer", None)
    if writer is None:
        writer = DiagnosticWriter()
    dataset = writer.convert_summaries_to_dataset(
        list(history), [summary.timestamp for summary in history], reference,
    )
    # The summary conversion, CF epoch/metadata checks, and physical encoding
    # all complete while scientific rollback is still available. Keep only the
    # prepared values, using the original checkpoint grammar for persistence.
    prepared = DataWriter(WriteConfig(cf_sync=True, strict_cf=False, add_bounds=False)).prepare_dataset(
        dataset, Path(target),
    )
    return _decode_checkpoint_value(
        _encode_checkpoint_value(prepared, path="diagnostic_export"),
        path="diagnostic_export",
    )


def export_diagnostics(
    simulator: Any, *, snapshot: Optional[Dict[str, Any]] = None,
    delivery_key: Optional[str] = None,
) -> Optional["WriteResult"]:
    """Export live periodic diagnostics or one frozen accepted-step request."""
    if snapshot is None:
        if not simulator.enable_diagnostic_export:
            return None
        if simulator.diagnostic_writer is None:
            raise RuntimeError("Diagnostic export requires an initialized DiagnosticWriter")
        snapshot = capture_diagnostic_export(simulator)
        if snapshot is None:
            logger.debug("No conservation summaries to export")
            return None

    try:
        from ...data.io.writer import DataWriter
        result = DataWriter().write_dataset(
            None, snapshot["result"]["output_path"], snapshot=snapshot, delivery_key=delivery_key,
        )
        if not result.validation_passed:
            raise OSError(f"Diagnostic export failed: {'; '.join(result.warnings)}")
        logger.info("Diagnostics exported to %s", result.output_path)
        return result

    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        logger.error("Failed to export diagnostics: %s", exc)
        raise


__all__ = [
    "collect_functor_diagnostics",
    "capture_diagnostic_export",
    "export_diagnostics",
    "get_last_conservation_summary",
    "get_coupling_scales_statistics",
    "get_simulation_summary",
]


def get_last_conservation_summary(simulator: Any) -> Optional[Any]:
    """Return latest conservation summary from StateManager if available."""
    if simulator.state_manager is None:
        return None
    return simulator.state_manager.get_last_conservation_summary()


def get_coupling_scales_statistics(simulator: Any) -> Dict[str, Any]:
    """Compute summary statistics for automatic scale checks and regridding."""
    if not simulator.enable_auto_scale:
        return {"auto_scale_enabled": False}

    regrid_events = [
        check
        for check in simulator._scale_check_history
        if check.get("needs_regrid", False)
    ]
    total_checks = len(simulator._scale_check_history)

    return {
        "auto_scale_enabled": True,
        "total_scale_checks": total_checks,
        "regrid_events": len(regrid_events),
        "regrid_rate": len(regrid_events) / max(total_checks, 1),
        "last_regrid_time": (
            regrid_events[-1].get("timestamp") if regrid_events else None
        ),
        "recent_scales": list(
            set(
                check.get("target_scale")
                for check in regrid_events[-10:]
                if check.get("target_scale")
            )
        ),
    }


def get_simulation_summary(simulator: Any) -> Dict[str, Any]:
    """Build consolidated runtime summary for simulator status endpoints."""
    summary = {
        "state": simulator.state.value,
        "current_time": simulator.current_time,
        "step_count": simulator.step_count,
        "total_execution_time": simulator.total_execution_time,
        "num_functors": len(simulator.functors),
        "execution_order": simulator.execution_order,
        "time_step": simulator.time_step,
        "max_time": simulator.max_time,
        "progress_percent": (
            (simulator.current_time / simulator.max_time * 100)
            if simulator.max_time > 0
            else 0
        ),
        "average_step_time": (
            (simulator.total_execution_time / simulator.step_count)
            if simulator.step_count > 0
            else 0
        ),
        "groups": {
            "fast": len(simulator._order_fast),
            "medium": len(simulator._order_medium),
            "slow": len(simulator._order_slow),
        },
    }

    conservation_summary = get_last_conservation_summary(simulator)
    if conservation_summary is not None:
        summary["conservation"] = {
            "total_violations": conservation_summary.total_violations,
            "num_warnings": len(conservation_summary.warnings),
            "num_custom_checks": len(conservation_summary.custom_checks),
            "timestamp": conservation_summary.timestamp,
        }

    if simulator.enable_auto_scale:
        summary["coupling.scales"] = get_coupling_scales_statistics(simulator)

    performance = getattr(simulator, "_performance_diagnostics", None)
    if isinstance(performance, dict):
        summary["performance"] = copy.deepcopy(performance)

    return summary
