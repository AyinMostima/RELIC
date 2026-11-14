"""Delegated hierarchy and audit helpers for ScaleManager."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

from torch import Tensor

from ....core.constants import get_scale_hierarchy
from ...core import AuditRecord, ConservationDiagnostics, ScaleHierarchyError
from ...core.policy import ScalingMode
from ..types import ScalingAudit, ScalingStrategy, SpatialQuantityBasis, resolve_spatial_basis

logger = logging.getLogger(__name__)


def validate_scale_transition_impl(self, source: str, target: str, direction: str) -> bool:
    hierarchy_config = ensure_hierarchy_config_impl(self)
    if "levels" in hierarchy_config:
        levels = hierarchy_config["levels"]
    else:
        levels = get_scale_hierarchy(self.parameter_manager)
    if not levels:
        raise ScaleHierarchyError(source, target, direction, get_scale_hierarchy(self.parameter_manager))
    if source not in levels or target not in levels:
        key = (source.lower(), target.lower(), direction.lower())
        return key in self._alias_transitions
    source_idx = levels.index(source)
    target_idx = levels.index(target)
    if direction == "up":
        return target_idx > source_idx
    return target_idx < source_idx


def ensure_hierarchy_config_impl(self) -> Dict[str, Any]:
    builder = self._require_topology_builder()
    self.hierarchy_config = builder.ensure_hierarchy_config()
    return self.hierarchy_config


def _metadata_basis(name: str, var_metadata: Dict[str, Any]) -> SpatialQuantityBasis:
    """Declared SpatialQuantityBasis of one metadata entry; nothing is inferred from names or units."""
    return resolve_spatial_basis(name, var_metadata)


def is_extensive_impl(self, var_metadata: Dict[str, Any]) -> bool:
    """True for amount bases (cell_total, area_density) declared in the metadata entry."""
    name = str(var_metadata.get("name") or var_metadata.get("variable_name") or "<unnamed>")
    return _metadata_basis(name, var_metadata).is_extensive


def select_strategy_impl(
    self,
    var_metadata: Dict[str, Dict[str, Any]],
    mode: ScalingMode,
) -> ScalingStrategy:
    bases = [_metadata_basis(str(name), meta) for name, meta in var_metadata.items()]
    all_extensive = all(basis.is_extensive for basis in bases)
    has_categorical = any(basis is SpatialQuantityBasis.CATEGORICAL for basis in bases)
    all_categorical = all(basis is SpatialQuantityBasis.CATEGORICAL for basis in bases)
    if mode in [ScalingMode.STRICT_CONSERVATIVE, ScalingMode.CONSERVATIVE]:
        if all_categorical:
            # Class codes are aggregated by their area-weighted mode, never averaged.
            return ScalingStrategy.MODE
        return ScalingStrategy.FLOW_CONSERVING if all_extensive else ScalingStrategy.CONSERVATIVE
    if mode == ScalingMode.BALANCED:
        if has_categorical:
            return ScalingStrategy.NEAREST
        if all_extensive:
            return ScalingStrategy.AREA_WEIGHTED_SUM
        return ScalingStrategy.AREA_WEIGHTED_MEAN
    return ScalingStrategy.NEAREST if has_categorical else ScalingStrategy.BILINEAR


def init_hierarchical_components_impl(self) -> None:
    from ..hierarchical.aggregator import HierarchicalAggregator
    from ..hierarchical.hierarchy_index import HierarchyIndex

    try:
        builder = self._require_topology_builder()
        hierarchy_cfg = ensure_hierarchy_config_impl(self)
        default_source = hierarchy_cfg["default_source_level"]
        default_target = hierarchy_cfg["default_target_level"]
        default_units = builder.infer_default_source_units(default_source, default_target)
        topology = builder.build_topology_from_parameters(default_source, default_target, default_units)

        self.hierarchy_index = HierarchyIndex(
            source_scale=default_source,
            target_scale=default_target,
            device=self.device,
            n_source_units=topology["n_source_units"],
            topology=topology,
        )

        precision = self._get_precision_policy(self.default_mode)
        self.hierarchy_index.hierarchy_tolerance = float(precision.tolerance)
        self.hierarchical_aggregator = HierarchicalAggregator(
            index=self.hierarchy_index,
            device=self.device,
            weight_type="area",
            rule=map_mode_to_rule_impl(self, self.default_mode),
            parameter_manager=self.parameter_manager,
        )
        self.hierarchical_aggregator.conservation_tolerance = float(precision.tolerance)
        register_hierarchical_components_impl(self, self.hierarchy_index, self.hierarchical_aggregator)
    except (RuntimeError, ValueError, KeyError) as exc:
        self.hierarchy_index = None
        self.hierarchical_aggregator = None
        logger.error("Error initializing hierarchical components: %s", exc)
        raise


def register_hierarchical_components_impl(self, index: Any, aggregator: Any) -> None:
    self.hierarchical_index = index
    self.hierarchical_aggregator = aggregator


def map_mode_to_rule_impl(self, mode: ScalingMode) -> str:
    if mode in [ScalingMode.STRICT_CONSERVATIVE, ScalingMode.CONSERVATIVE]:
        return "conservative"
    if mode == ScalingMode.BALANCED:
        return "area_weighted_sum"
    return "bilinear"


def compute_child_units_impl(
    self,
    coarse_scale: str,
    fine_scale: str,
    parent_units: int,
) -> Optional[int]:
    return self._require_topology_builder().compute_child_units(coarse_scale, fine_scale, parent_units)


def build_topology_from_parameters_impl(
    self,
    source_scale: str,
    target_scale: str,
    n_source_units: int,
    n_target_units: Optional[int] = None,
):
    return self._require_topology_builder().build_topology_from_parameters(
        source_scale,
        target_scale,
        n_source_units,
        n_target_units,
    )


def validate_topology_impl(self, topology: Dict[str, Any], operation: str) -> Dict[str, Any]:
    return self._require_topology_builder().validate_topology(topology, operation)


def get_temporal_coverage_threshold_impl(self) -> float:
    return float(
        self._get_param(
            "coverage_threshold",
            namespace="coupling.scales.temporal",
            units="dimensionless",
            expected_type=float,
        )
    )


def aggregate_temporal_alias_impl(self, **kwargs):
    return self.temporal_strategy.aggregate_temporal_alias(**kwargs)


def disaggregate_temporal_alias_impl(self, **kwargs):
    return self.temporal_strategy.disaggregate_temporal_alias(**kwargs)


def update_metadata_with_quality_impl(
    self,
    metadata: Dict[str, Any],
    quality: Dict[str, Any],
    diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    updated = metadata.copy()
    updated["quality"] = quality
    updated["diagnostics"] = diagnostics
    return updated


def collect_facade_warnings_impl(self) -> List[str]:
    warnings_list = self._facade_warnings.copy() + self._warnings.copy()
    self._facade_warnings.clear()
    self._warnings.clear()
    return warnings_list


def sync_cf_metadata_impl(
    self,
    operation: str,
    source_scale: str,
    target_scale: str,
    metadata: Any,
    var_metadata: Any,
    data: Any,
) -> Dict[str, Any]:
    """Propagate CF cell_methods and history of a governed hierarchical rescaling.

    Aggregation calls ``CFAttributeSynchronizer.sync_after_aggregation`` and disaggregation
    ``sync_after_disaggregation``; both record ``area: <method>`` and a history entry naming
    ``source_scale`` and ``target_scale``. ``data`` is the output tensor (all variables share
    one cell method) or the mapping of every variable to its output tensor. Variables of one
    call with one cell method return one metadata mapping; a call whose variables have
    different cell methods returns one mapping per variable.
    """
    base = dict(metadata) if isinstance(metadata, dict) else {}
    if self.cf_synchronizer is None:
        raise RuntimeError("cf_metadata_synchronizer_unbound")
    if operation not in ("aggregate", "disaggregate"):
        raise ValueError(f"CF metadata sync supports 'aggregate' or 'disaggregate', got {operation!r}")
    if isinstance(data, Mapping):
        outputs: Optional[Dict[str, Any]] = dict(data)
    elif isinstance(data, Tensor):
        outputs = None
    else:
        raise RuntimeError(f"cf_metadata_{operation}_data_unbound")
    if not isinstance(var_metadata, dict) or not var_metadata:
        raise RuntimeError("cf_metadata_variable_role_unbound")

    # References: CF Conventions 1.8 section 7.3 (cell_methods); docs/source/theory/
    #   scale_coupling.md section 2.9 (hierarchical aggregation and disaggregation).
    # Mathematical expression: method = "sum" for cell totals (parent sum X_j = sum_i X_i, or
    #   the area-proportional split X_i = X_j A_i / sum_k A_k), "mean" for intensive values
    #   and area densities (parent area mean, or the broadcast of the parent value), and
    #   "mode" for an aggregated categorical field (area-weighted mode).
    # Parameter meanings: X cell total, A cell area; the basis is the declared
    #   SpatialQuantityBasis of each variable of the call.
    # Units: cell_methods and history are text; the data units pass through unchanged.
    # Domain/constraints: a disaggregated categorical field has no CF rescaling contract and
    #   raises; variables with different methods are synchronized per variable and then need
    #   the output tensor of every variable.
    # Sign convention: not applicable (metadata only).
    methods: Dict[str, str] = {}
    for name, item in var_metadata.items():
        if not isinstance(item, dict):
            raise RuntimeError(f"cf_metadata_variable_role_unbound: metadata for {name!r} is not a mapping")
        basis = _metadata_basis(str(name), item)
        if basis is SpatialQuantityBasis.CATEGORICAL:
            if operation != "aggregate":
                raise RuntimeError(
                    f"cf_metadata_variable_role_unbound: categorical field {name!r} has no CF "
                    "disaggregation contract"
                )
            methods[name] = "mode"
        elif basis is SpatialQuantityBasis.CELL_TOTAL:
            methods[name] = "sum"
        else:
            methods[name] = "mean"

    def _output(name: str) -> Tensor:
        if outputs is None:
            return data
        tensor = outputs.get(name)
        if not isinstance(tensor, Tensor):
            raise RuntimeError(f"cf_metadata_{operation}_data_unbound: no output tensor for {name!r}")
        return tensor

    def _sync(method: str, tensor: Tensor) -> Dict[str, Any]:
        try:
            if operation == "aggregate":
                return self.cf_synchronizer.sync_after_aggregation(
                    data=tensor,
                    metadata=dict(base),
                    aggregation_method=method,
                    dimension="area",
                    source_label=source_scale,
                    target_label=target_scale,
                )
            return self.cf_synchronizer.sync_after_disaggregation(
                data=tensor,
                metadata=dict(base),
                disaggregation_method=method,
                dimension="area",
                source_label=source_scale,
                target_label=target_scale,
            )
        except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
            raise RuntimeError(
                f"CF metadata {operation} failed for {source_scale} -> "
                f"{target_scale} with method {method}: {exc}"
            ) from exc

    distinct = set(methods.values())
    if len(distinct) == 1:
        return _sync(distinct.pop(), _output(next(iter(methods))))
    if outputs is None:
        raise RuntimeError(
            "cf_metadata_variable_role_unbound: variables with different cell methods "
            f"{sorted(distinct)} need the output tensor of every variable"
        )
    return {name: _sync(method, _output(name)) for name, method in methods.items()}


def record_scaling_audit_impl(
    self,
    scaling_audit: ScalingAudit,
    conservation_details: Optional[Dict[str, Any]] = None,
    cf_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    if not self.audit_collector:
        return
    metadata_updates: Dict[str, Any] = {}
    if scaling_audit.metadata:
        metadata_updates.update(scaling_audit.metadata)
    if cf_metadata:
        metadata_updates.update(cf_metadata)
    record = AuditRecord(
        operation=scaling_audit.operation,
        source_scale=scaling_audit.source_scale,
        target_scale=scaling_audit.target_scale,
        variables=scaling_audit.variables,
        mode=str(scaling_audit.mode),
        strategy=str(scaling_audit.strategy),
        conservation={},
        mass_balance_error=scaling_audit.mass_balance_error,
        relative_error=scaling_audit.relative_error,
        cf_compliant=scaling_audit.cf_compliant,
        metadata_updates=metadata_updates,
        warnings=scaling_audit.warnings,
        errors=[],
        performance={},
        custom_data={"conservation_enforced": scaling_audit.conservation_enforced},
    )
    if conservation_details:
        for var, diag in conservation_details.items():
            if isinstance(diag, Mapping):
                is_conserved = bool(diag.get("conserved", False))
                absolute_error = float(diag.get("absolute_error", 0.0))
                relative_error = float(diag.get("relative_error", 0.0))
                source_total = float(diag.get("source_total", 0.0))
                target_total = float(diag.get("target_total", 0.0))
            else:
                is_conserved = bool(getattr(diag, "is_conserved", False))
                absolute_error = float(getattr(diag, "absolute_error", 0.0))
                relative_error = float(getattr(diag, "relative_error", 0.0))
                source_total = float(getattr(diag, "source_total", 0.0))
                target_total = float(getattr(diag, "target_total", 0.0))
            record.add_conservation_check(
                var,
                ConservationDiagnostics(
                    property_name=var,
                    is_conserved=is_conserved,
                    absolute_error=absolute_error,
                    relative_error=relative_error,
                    source_total=source_total,
                    target_total=target_total,
                ),
            )
    self.audit_collector.add_record(record)
