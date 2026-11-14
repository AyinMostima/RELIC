"""
Temporal scaling strategies.

Formerly TemporalScalingOrchestrator. Encapsulates temporal alias aggregation paths.
"""

from typing import Any, Dict, List, Optional, Tuple

from torch import Tensor

from ..core.policy import ScalingMode
from .types import ScalingStrategy, ScalingAudit

class TemporalScalingStrategy:
    """Route scale aliases without inventing temporal calendar contracts."""

    def __init__(self, manager: Any) -> None:
        self.manager = manager

    def __getattr__(self, item: str) -> Any:
        return getattr(self.manager, item)

    def handle_alias_aggregation(
        self,
        data_dict: Dict[str, Tensor],
        source_scale: str,
        target_scale: str,
        variables: List[str],
        strategy: ScalingStrategy,
        mode: ScalingMode,
        metadata: Optional[Dict[str, Any]],
        var_metadata: Dict[str, Dict[str, Any]],
    ) -> None:
        """Temporal alias pairs fail closed; spatial and conservative pairs are not handled here."""
        alias_key = (source_scale.lower(), target_scale.lower())
        if alias_key in self._temporal_alias_up:
            self.aggregate_temporal_alias(
                data_dict=data_dict,
                source_scale=source_scale,
                target_scale=target_scale,
                variables=variables,
                strategy=strategy,
                mode=mode,
                metadata=metadata,
                var_metadata=var_metadata,
            )

    def handle_alias_disaggregation(
        self,
        data_dict: Dict[str, Tensor],
        source_scale: str,
        target_scale: str,
        variables: List[str],
        strategy: ScalingStrategy,
        mode: ScalingMode,
        metadata: Optional[Dict[str, Any]],
        var_metadata: Dict[str, Dict[str, Any]],
    ) -> None:
        """Temporal alias pairs fail closed; spatial and conservative pairs are not handled here."""
        alias_key = (source_scale.lower(), target_scale.lower())
        if alias_key in self._temporal_alias_down:
            self.disaggregate_temporal_alias(
                data_dict=data_dict,
                source_scale=source_scale,
                target_scale=target_scale,
                variables=variables,
                strategy=strategy,
                mode=mode,
                metadata=metadata,
                var_metadata=var_metadata,
            )

    def aggregate_temporal_alias(
        self,
        data_dict: Dict[str, Tensor],
        source_scale: str,
        target_scale: str,
        variables: List[str],
        strategy: ScalingStrategy,
        mode: ScalingMode,
        metadata: Optional[Dict[str, Any]],
        var_metadata: Dict[str, Dict[str, Any]],
    ) -> Tuple[Dict[str, Tensor], ScalingAudit]:
        """Fail closed until a calendar-owned alias period contract is supplied."""
        del (
            data_dict,
            source_scale,
            target_scale,
            variables,
            strategy,
            mode,
            metadata,
            var_metadata,
        )
        raise RuntimeError(
            "temporal_alias_periodization_unbound: temporal aliases require "
            "explicit calendar, time units, source/target bounds, and temporal "
            "semantics before canonical aggregation"
        )

    def disaggregate_temporal_alias(
        self,
        data_dict: Dict[str, Tensor],
        source_scale: str,
        target_scale: str,
        variables: List[str],
        strategy: ScalingStrategy,
        mode: ScalingMode,
        metadata: Optional[Dict[str, Any]],
        var_metadata: Dict[str, Dict[str, Any]],
    ) -> Tuple[Dict[str, Tensor], ScalingAudit]:
        """Fail closed until the canonical TemporalDisaggregator is fully bound."""
        del (
            data_dict,
            source_scale,
            target_scale,
            variables,
            strategy,
            mode,
            metadata,
            var_metadata,
        )
        raise RuntimeError(
            "temporal_alias_disaggregation_unbound: use TemporalDisaggregator "
            "with explicit period, support, unit, and conservation contracts"
        )
