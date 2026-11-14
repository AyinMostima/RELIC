"""
Cross-domain validation helpers for RELICSimulator.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)
# Couplings whose step outputs carry the terms of their cross-domain checks. The
# biosphere-atmosphere coupling is not tracked: BiosphereAtmosphereCoupling emits only
# boundary_layer_conductance, so its water (ET = T + E_soil) and energy (Rn = H + LE + G)
# budgets can never be evaluated from its outputs.
_COUPLING_KEYWORDS = (
    "soil_vegetation",
    "water_carbon",
)


def _should_run_cross_domain_validation(self: Any, executed_groups: List[str]) -> bool:
    """Check whether cross-domain validation should run for this step."""
    if self.cross_domain_validator is None or self.state_manager is None:
        if len(executed_groups) > 1:
            logger.debug(
                "Cross-domain validation needed but disabled for groups: %s",
                executed_groups,
            )
        return False
    return len(executed_groups) > 1


def _has_functor_metadata(functor: Any) -> bool:
    """Return True when functor has usable metadata."""
    return bool(
        functor is not None
        and hasattr(functor, "metadata")
        and functor.metadata is not None
    )


def _is_tracked_coupling_functor(functor_name: str, functor: Any) -> bool:
    """Return True for coupling functors tracked by cross-domain validation."""
    if not _has_functor_metadata(functor):
        return False
    lowered_name = functor_name.lower()
    return any(keyword in lowered_name for keyword in _COUPLING_KEYWORDS)


def _collect_executed_coupling_modules(self: Any, executed_groups: List[str]) -> List[str]:
    """Collect coupling module names executed in current step."""
    coupling_modules: List[str] = []
    functor_registry = self.functors

    def _resolve_functor(name: str) -> Any:
        # `self.functors` may be a dict-like object or torch.nn.ModuleDict.
        getter = getattr(functor_registry, "get", None)
        if callable(getter):
            return getter(name)
        if name in functor_registry:
            return functor_registry[name]
        return None

    for group in executed_groups:
        for functor_name in self._groups.get(group, []):
            functor = _resolve_functor(functor_name)
            if _is_tracked_coupling_functor(functor_name, functor):
                coupling_modules.append(functor_name)
    return coupling_modules


def _resolve_coupling_type(coupling_name: str) -> str | None:
    """Resolve validator coupling type from functor name."""
    lowered_name = coupling_name.lower()
    if "soil_vegetation" in lowered_name:
        return "soil_vegetation"
    if "water_carbon" in lowered_name:
        return "water_carbon"
    return None


def _validate_single_coupling(
    self: Any,
    coupling_name: str,
    current_state: dict[str, Any],
) -> tuple[List[str], List[str]]:
    """Validate one coupling module and return collected errors and warnings."""
    functor = self.functors[coupling_name]
    last_updates = getattr(functor, "_last_updates", None)
    if not last_updates:
        return [], []

    coupling_type = _resolve_coupling_type(coupling_name)
    if coupling_type is None:
        return [], []

    validation_result = self.cross_domain_validator.validate_coupling_conservation(
        coupling_type,
        current_state,
        last_updates,
    )

    logger.debug(
        "Cross-domain validation for %s: %s",
        coupling_type,
        "PASSED" if validation_result["passed"] else "FAILED",
    )

    if validation_result["passed"]:
        return [], []
    return list(validation_result["errors"]), list(validation_result["warnings"])


def _log_cross_domain_validation_summary(
    validation_errors: List[str],
    validation_warnings: List[str],
    coupling_modules_executed: List[str],
) -> None:
    """Emit summary logs for cross-domain validation results."""
    if validation_errors:
        logger.error("Cross-domain validation errors detected: %s", validation_errors)
    if validation_warnings:
        logger.warning("Cross-domain validation warnings: %s", validation_warnings)
    if not validation_errors and not validation_warnings and coupling_modules_executed:
        logger.debug(
            "Cross-domain validation passed for %s coupling modules",
            len(coupling_modules_executed),
        )


def validate_cross_domain_coupling(self: Any, executed_groups: List[str]) -> None:
    """
    Run cross-domain coupling validation for executed groups.
    """
    if not _should_run_cross_domain_validation(self, executed_groups):
        return

    current_state = self.state_manager._state_variables.copy()
    coupling_modules_executed = _collect_executed_coupling_modules(self, executed_groups)

    errors_by_coupling: Dict[str, List[str]] = {}
    warnings_by_coupling: Dict[str, List[str]] = {}
    for coupling_name in coupling_modules_executed:
        errors, warnings = _validate_single_coupling(
            self,
            coupling_name,
            current_state,
        )
        if errors:
            errors_by_coupling[coupling_name] = list(errors)
        if warnings:
            warnings_by_coupling[coupling_name] = list(warnings)

    validation_errors = [message for messages in errors_by_coupling.values() for message in messages]
    validation_warnings = [message for messages in warnings_by_coupling.values() for message in messages]
    _log_cross_domain_validation_summary(
        validation_errors,
        validation_warnings,
        coupling_modules_executed,
    )

    # Every step with findings is recorded in the simulator report
    # (get_cross_domain_reports). ERROR findings raise unless the explicitly
    # configured mode is "report"; warnings never raise.
    mode = self.cross_domain_validation_mode
    if errors_by_coupling or warnings_by_coupling:
        self._cross_domain_reports.append(
            {
                "time": self.current_time,
                "step": self.step_count,
                "mode": mode,
                "couplings": list(coupling_modules_executed),
                "errors": errors_by_coupling,
                "warnings": warnings_by_coupling,
            }
        )
    if errors_by_coupling and mode == "raise":
        details = "; ".join(
            f"{name}: {', '.join(messages)}" for name, messages in errors_by_coupling.items()
        )
        raise RuntimeError(
            f"Cross-domain validation failed at t={self.current_time}: {details} "
            "(cross_domain_validation_mode='raise')"
        )
