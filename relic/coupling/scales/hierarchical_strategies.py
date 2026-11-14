"""
Hierarchical scaling strategies.

Encapsulates aggregation/disaggregation logic.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from ...core.constants import get_scale_hierarchy
from .types import ScalingStrategy, ScalingAudit, SpatialQuantityBasis, resolve_spatial_basis
from ..core.conservation_checker import coarsest_floating_dtype
from ..core.policy import ScalingMode
from ..core.errors import ScaleHierarchyError

logger = logging.getLogger(__name__)


class HierarchicalScalingStrategy:
    """Encapsulate hierarchical aggregate/disaggregate logic for ScaleManager."""

    def __init__(self, manager: Any) -> None:
        self.manager = manager

    def _validate_transition(self, source_scale: str, target_scale: str, direction: str) -> None:
        manager = self.manager
        alias_key = (str(source_scale).lower(), str(target_scale).lower(), direction)
        if alias_key in manager._DEFAULT_ALIAS_TRANSITIONS:
            return
        if manager._validate_scale_transition(source_scale, target_scale, direction):
            return

        valid_hierarchy = get_scale_hierarchy()
        if direction == "down":
            valid_hierarchy = list(reversed(valid_hierarchy))
        raise ScaleHierarchyError(
            source_scale=source_scale,
            target_scale=target_scale,
            direction=direction,
            valid_hierarchy=valid_hierarchy,
        )

    def _ensure_hierarchical_components(self) -> None:
        if self.manager.hierarchical_aggregator is None:
            self.manager._init_hierarchical_components()

    def _normalize_input_payload(
        self,
        data: Union[Tensor, Dict[str, Tensor]],
        variables: Optional[List[str]],
    ) -> Tuple[Dict[str, Tensor], List[str]]:
        if isinstance(data, Tensor):
            if variables is None or len(variables) != 1:
                raise ValueError("Single tensor requires single variable name")
            return {variables[0]: data}, [variables[0]]

        data_dict = data
        resolved_variables = list(variables) if variables is not None else list(data_dict.keys())
        return data_dict, resolved_variables

    @staticmethod
    def _call_metadata_entry(
        var: str,
        variables: List[str],
        metadata: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Per-call metadata of one variable: ``metadata[var]`` or, for a single variable, the flat mapping."""
        if not isinstance(metadata, dict):
            return None
        entry = metadata.get(var)
        if isinstance(entry, dict):
            return dict(entry)
        if len(variables) == 1 and ("spatial_basis" in metadata or "variable_type" in metadata):
            return dict(metadata)
        return None

    def _resolve_variable_metadata(
        self,
        source_scale: str,
        variables: List[str],
        metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Catalog metadata per variable, else the caller's declared metadata; nothing is inferred.

        Every resolved entry must declare an unambiguous SpatialQuantityBasis (spatial_basis, or
        variable_type intensive/categorical); an undeclared variable re-raises the catalog error.
        """
        manager = self.manager
        resolved: Dict[str, Dict[str, Any]] = {}
        for var in variables:
            try:
                entry = manager.get_variable_metadata(var, scale=source_scale)
            except KeyError:
                entry = self._call_metadata_entry(var, variables, metadata)
                if entry is None:
                    raise
            entry = {"name": var, **entry}
            resolve_spatial_basis(var, entry)
            resolved[var] = entry
        return resolved

    def _empty_result(
        self,
        operation: str,
        source_scale: str,
        target_scale: str,
        mode: ScalingMode,
        emit_warning: bool = False,
    ) -> Tuple[Dict[str, Tensor], ScalingAudit]:
        if emit_warning:
            warnings.warn("Empty data provided for disaggregation")
        return {}, ScalingAudit(
            operation=operation,
            source_scale=source_scale,
            target_scale=target_scale,
            variables=[],
            mode=mode,
            strategy=ScalingStrategy.AREA_WEIGHTED_MEAN,
            mass_balance_error=0.0,
            relative_error=0.0,
            cf_compliant=False,
            conservation_enforced=False,
            physical_constraints_applied=False,
            warnings=["Empty data dictionary provided"],
            metadata={},
        )

    @staticmethod
    def _infer_n_source_units(data_dict: Dict[str, Tensor]) -> int:
        """Number of spatial units of the input: the size of its last axis.

        The hierarchical kernels map [..., n_source] to [..., n_target]; leading axes are
        batches (for example time slices), never spatial units.
        """
        first_var_data = next(iter(data_dict.values()))
        if first_var_data.dim() < 1:
            raise ValueError("Hierarchical scaling needs a field with its spatial units on the last axis.")
        return int(first_var_data.shape[-1])

    def _resolve_aggregate_topology(
        self,
        source_scale: str,
        target_scale: str,
        n_source_units: int,
        topology: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        manager = self.manager
        resolved_topology = topology
        if resolved_topology is None:
            resolved_topology = manager._build_topology_from_parameters(
                source_scale=source_scale,
                target_scale=target_scale,
                n_source_units=n_source_units,
            )
        if resolved_topology is None:
            raise RuntimeError(
                f"Missing hierarchy topology for {source_scale} -> {target_scale}; "
                "please supply topology or configure coupling.scales.hierarchy.* parameters."
            )

        resolved_topology = manager._validate_topology(resolved_topology, operation="aggregate")
        resolved_topology = resolved_topology.copy()
        resolved_topology.setdefault("n_source_units", n_source_units)
        return resolved_topology

    def _resolve_disaggregate_topology(
        self,
        source_scale: str,
        target_scale: str,
        n_input_units: int,
        topology: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], int]:
        manager = self.manager
        resolved_topology = topology.copy() if topology is not None else None
        parent_units = max(1, n_input_units)
        child_units = None
        if resolved_topology is None:
            child_units = manager._compute_child_units(
                coarse_scale=source_scale,
                fine_scale=target_scale,
                parent_units=parent_units,
            )
            if child_units is not None:
                resolved_topology = manager._build_topology_from_parameters(
                    source_scale=target_scale,
                    target_scale=source_scale,
                    n_source_units=child_units,
                    n_target_units=parent_units,
                )
        if resolved_topology is None:
            raise RuntimeError(
                f"Missing hierarchy topology for {target_scale} -> {source_scale} disaggregation; "
                "configure coupling.scales.hierarchy.* parameters or provide topology explicitly."
            )

        resolved_topology = manager._validate_topology(resolved_topology, operation="disaggregate")
        n_source_units = resolved_topology.get(
            "n_source_units",
            child_units if child_units is not None else parent_units,
        )
        return resolved_topology, n_source_units

    def _build_hierarchy_runtime(
        self,
        source_scale: str,
        target_scale: str,
        n_source_units: int,
        topology: Dict[str, Any],
        mode: ScalingMode,
    ) -> Tuple[Any, Any]:
        from .hierarchical.hierarchy_index import HierarchyIndex
        from .hierarchical.aggregator import HierarchicalAggregator

        manager = self.manager
        hierarchy_index = HierarchyIndex(
            source_scale=source_scale,
            target_scale=target_scale,
            device=manager.device,
            n_source_units=n_source_units,
            topology=topology,
        )

        # The call's mode supplies the tolerance of every check of this call (one source:
        # coupling.scales.defaults.precision.<mode>.tolerance, ScaleManager.ledger_tolerance).
        tolerance = manager.ledger_tolerance(mode)
        if hasattr(hierarchy_index, "hierarchy_tolerance"):
            hierarchy_index.hierarchy_tolerance = min(
                float(hierarchy_index.hierarchy_tolerance),
                tolerance,
            )

        aggregator = HierarchicalAggregator(
            index=hierarchy_index,
            device=manager.device,
            weight_type="area",
            rule=manager._map_mode_to_rule(mode),
            parameter_manager=manager.parameter_manager,
        )
        aggregator.conservation_tolerance = tolerance
        return hierarchy_index, aggregator

    def _build_conservation_audit(
        self,
        operation: str,
        variables: List[str],
        var_metadata: Dict[str, Dict[str, Any]],
        source_data: Dict[str, Tensor],
        target_data: Dict[str, Tensor],
        mode: ScalingMode,
        source_areas: Tensor,
        target_areas: Tensor,
        uncovered_source: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """Basis-aware, fail-closed conservation audit of the returned fields on the covered support."""
        # References: docs/source/theory/scale_coupling.md sections 2.3.2 and 2.9; Jones (1999)
        #   Mon. Weather Rev. 127:2204-2210 (conservative remapping ledgers).
        # Mathematical expression: the fields are [*batch, n] with the support on the last axis
        #   (the hierarchical kernels map [..., n_source] to [..., n_target]); for every batch
        #   index b: ledger L_b = sum_i X_(b,i) with X_(b,i) = x_(b,i) A_i for an area density and
        #   X_(b,i) as given for a cell total; L_src,b over the covered source cells (children
        #   without a parent excluded), L_tgt,b over the returned target cells;
        #   U_b = sum over the uncovered source cells of X_(b,i); r_b = L_tgt,b - L_src,b,
        #   e_b = |r_b|, epsilon_b = e_b / max(|L_src,b|, 1e-12) (ConservationEnforcer check with
        #   batch_ndim = ndim - 1, weights A for area densities and no weights for cell totals);
        #   accepted_b iff e_b <= max(tol max(|L_src,b|, 1e-12), gamma_(2n+2)(u_dtype) a_b) by the
        #   one shared rule conservation_checker.ledger_acceptance (Higham 2002 rounding bound of
        #   the coarsest floating dtype of the field, the result and the areas; a_b = sum_i
        #   |X_(b,i)|, n source cells); the call is conserved only when every batch is.
        # Parameter meanings: x/X source values; A_i, A_j source and target areas; b batch index
        #   over the leading axes; U_b the uncovered source amount of batch b; r_b its signed
        #   residual; tol = coupling.scales.defaults.precision.<mode>.tolerance of this call's
        #   mode (ScaleManager.ledger_tolerance; the default mode when the call passes none);
        #   u_dtype unit roundoff of the coarsest floating dtype of the source field, the returned
        #   field and the areas (a float32 field aggregated into a float64 result keeps float32
        #   rounding); a_b activity of the source terms of batch b.
        # Units: A [m2]; L, U, r, e and a in the amount unit of the field; epsilon, tol, gamma [1].
        # Domain/constraints: intensive and categorical fields are not audited (no amount
        #   ledger). The ledger of every amount is evaluated in every mode, per batch, from the
        #   returned field itself, so a wrong result can never report a zero residual and the
        #   residuals of different batches never cancel; in the strict path (enforce_conservation
        #   or STRICT_CONSERVATIVE) any rejected batch raises RuntimeError, and otherwise every
        #   batch residual and the failed batches are reported. The returned field is never
        #   rescaled and U_b is reported, never redistributed. The audit runs after the physical
        #   constraints, so clipping residuals are included.
        # Sign convention: signed amounts; r_b > 0 when the target holds more than the covered
        #   source (for example after clipping up to a lower bound); epsilon_b, e_b >= 0.
        manager = self.manager
        strict = bool(manager.enforce_conservation or mode == ScalingMode.STRICT_CONSERVATIVE)
        tolerance = manager.ledger_tolerance(mode)

        conservation_audit: Dict[str, Any] = {}
        for var in variables:
            basis = resolve_spatial_basis(var, var_metadata[var])
            if not basis.is_extensive:
                continue
            area_weighted = basis is SpatialQuantityBasis.AREA_DENSITY
            source_weights = source_areas if area_weighted else None
            target_weights = target_areas if area_weighted else None
            source_values = source_data[var]
            if source_values.dim() < 1:
                raise ValueError(
                    f"Hierarchical {operation} of '{var}' needs a field with its support on the last "
                    f"axis; got a 0-d tensor."
                )
            batch_ndim = source_values.dim() - 1
            uncovered_values = torch.zeros_like(source_values)
            if uncovered_source is not None:
                mask = uncovered_source.to(device=source_values.device)
                zeros = torch.zeros_like(source_values)
                uncovered_values = torch.where(mask, source_values, zeros)
                if source_weights is not None:
                    uncovered_values = uncovered_values * torch.as_tensor(
                        source_weights, device=source_values.device, dtype=source_values.dtype
                    )
                source_values = torch.where(mask, zeros, source_values)
            uncovered_per_batch = uncovered_values.detach().to(torch.float64).sum(dim=-1).cpu()

            diagnostics = manager.conservation_enforcer.check_conservation(
                source_data=source_values,
                target_data=target_data[var],
                property_name=var,
                source_weights=source_weights,
                target_weights=target_weights,
                rounding_dtype=coarsest_floating_dtype(
                    source_values.dtype,
                    target_data[var].dtype,
                    torch.as_tensor(source_areas).dtype,
                    torch.as_tensor(target_areas).dtype,
                ),
                tolerance=tolerance,
                batch_ndim=batch_ndim,
            )
            for ledger in diagnostics.metadata["batch_ledgers"]:
                ledger["uncovered_source_amount"] = float(uncovered_per_batch[ledger["batch_index"]].item())
            # One amount per batch: a float for a field without batch axes, else nested lists
            # shaped like the batch axes (amounts of different batches are never added).
            diagnostics.metadata["uncovered_source_amount"] = uncovered_per_batch.tolist()
            diagnostics.metadata["spatial_basis"] = basis.value
            if strict and not diagnostics.is_conserved:
                failed = [
                    ledger for ledger in diagnostics.metadata["batch_ledgers"] if not ledger["is_conserved"]
                ]
                listed = "; ".join(
                    f"batch {ledger['batch_index']}: residual {ledger['residual']:+.3e} (relative "
                    f"{ledger['relative_error']:.3e}, accepted {ledger['accepted_residual']:.3e}, "
                    f"uncovered source amount {ledger['uncovered_source_amount']:.6g})"
                    for ledger in failed[:8]
                )
                raise RuntimeError(
                    f"Hierarchical {operation} of '{var}' ({basis.value}) violates conservation on "
                    f"the covered support in {len(failed)} of "
                    f"{len(diagnostics.metadata['batch_ledgers'])} batch ledger(s): {listed} "
                    f"(tolerance {diagnostics.tolerance:.3e}); the result is not rescaled."
                )
            conservation_audit[var] = diagnostics

        return conservation_audit

    @staticmethod
    def _resolve_bounds_for_variable(
        metadata: Optional[Dict[str, Any]],
        var_meta: Dict[str, Any],
        var: str,
    ) -> Optional[Any]:
        bounds = None
        if metadata and isinstance(metadata, dict):
            if var in metadata and isinstance(metadata[var], dict):
                bounds = metadata[var].get("physical_bounds") or metadata[var].get("bounds")
            else:
                bounds = metadata.get("physical_bounds") or metadata.get("bounds")
        if bounds is None and isinstance(var_meta, dict):
            bounds = var_meta.get("physical_bounds") or var_meta.get("bounds")
        return bounds

    def _apply_constraints(
        self,
        data: Dict[str, Tensor],
        variables: List[str],
        metadata: Optional[Dict[str, Any]],
        var_metadata: Dict[str, Dict[str, Any]],
    ) -> bool:
        manager = self.manager
        if manager.constraint_manager is None:
            return False

        constraints_applied = False
        for var in variables:
            var_meta = var_metadata.get(var, {}) if isinstance(var_metadata, dict) else {}
            # The resolved metadata of the variable (call metadata, else its catalog entry) is its
            # declaration: its bounds apply, and a variable declared without bounds is not bounded
            # (no second catalog lookup); registered constraints apply either way.
            bounds = self._resolve_bounds_for_variable(metadata, var_meta, var)
            data[var] = manager.constraint_manager.apply_constraints(
                data[var],
                bounds=bounds,
                variable_name=var if bounds is not None else None,
            )
            constraints_applied = constraints_applied or bool(manager.constraint_manager._last_bounds_source)
        return constraints_applied

    def _collect_operation_warnings(
        self,
        operation_audit: Dict[str, Any],
        conservation_audit: Dict[str, Any],
    ) -> List[str]:
        all_warnings: List[str] = []
        all_warnings.extend(operation_audit.get("warnings", []))
        for diagnostics in conservation_audit.values():
            if hasattr(diagnostics, "warnings") and diagnostics.warnings:
                all_warnings.extend(diagnostics.warnings)
        all_warnings.extend(self.manager._collect_facade_warnings())
        return all_warnings

    @staticmethod
    def _max_conservation_metric(
        conservation_audit: Dict[str, Any],
        variables: List[str],
        metric_name: str,
    ) -> float:
        if not conservation_audit:
            return 0.0
        return max(
            float(getattr(conservation_audit.get(var), metric_name, 0.0) or 0.0)
            for var in variables
        )

    def _variable_strategies(
        self,
        var_metadata: Dict[str, Dict[str, Any]],
        mode: ScalingMode,
        strategy: Optional[ScalingStrategy],
    ) -> Dict[str, ScalingStrategy]:
        """The caller's strategy for every variable, else each variable's strategy from its declared basis."""
        if strategy is not None:
            return {var: strategy for var in var_metadata}
        return {
            var: self.manager._select_strategy({var: entry}, mode) for var, entry in var_metadata.items()
        }

    @staticmethod
    def _call_strategy(strategies: Dict[str, ScalingStrategy]) -> Optional[ScalingStrategy]:
        """The one strategy of a call, or None when its variables use different strategies."""
        distinct = set(strategies.values())
        return next(iter(distinct)) if len(distinct) == 1 else None

    def _build_scaling_audit(
        self,
        operation: str,
        source_scale: str,
        target_scale: str,
        variables: List[str],
        mode: ScalingMode,
        strategies: Dict[str, ScalingStrategy],
        operation_audit: Dict[str, Any],
        conservation_audit: Dict[str, Any],
        updated_metadata: Dict[str, Any],
        constraints_applied: bool,
    ) -> ScalingAudit:
        metadata_key = "aggregation_audit" if operation == "aggregate" else "disaggregation_audit"
        return ScalingAudit(
            operation=operation,
            source_scale=source_scale,
            target_scale=target_scale,
            variables=variables,
            mode=mode,
            strategy=self._call_strategy(strategies),
            mass_balance_error=self._max_conservation_metric(conservation_audit, variables, "absolute_error"),
            relative_error=self._max_conservation_metric(conservation_audit, variables, "relative_error"),
            cf_compliant=False,
            conservation_enforced=self.manager.enforce_conservation,
            physical_constraints_applied=constraints_applied,
            warnings=self._collect_operation_warnings(operation_audit, conservation_audit),
            metadata={
                metadata_key: operation_audit,
                "strategy_per_variable": {var: item.value for var, item in strategies.items()},
                "conservation_audit": conservation_audit,
                # Amount variables whose ledger was evaluated (intensive/categorical have none).
                "conservation_audited_variables": sorted(conservation_audit),
                "cf_metadata": updated_metadata,
                "cf_validation_scope": "not_evaluated",
                "dataset_conformance_checked": False,
            },
        )

    # ----------------------------- Public API -----------------------------
    def aggregate(
        self,
        data: Union[Tensor, Dict[str, Tensor]],
        source_scale: str,
        target_scale: str,
        variables: Optional[List[str]] = None,
        mode: Optional[ScalingMode] = None,
        strategy: Optional[ScalingStrategy] = None,
        metadata: Optional[Dict[str, Any]] = None,
        topology: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Union[Tensor, Dict[str, Tensor]], ScalingAudit]:
        manager = self.manager
        mode = mode or manager.default_mode
        self._validate_transition(source_scale, target_scale, "up")
        self._ensure_hierarchical_components()

        data_dict, variables = self._normalize_input_payload(data, variables)
        var_metadata = self._resolve_variable_metadata(
            source_scale=source_scale,
            variables=variables,
            metadata=metadata,
        )

        # Temporal alias pairs fail closed here; spatial and conservative alias pairs continue on
        # this path (declared topology, basis-aware kernels, ledger audit) like any other pair.
        manager._handle_alias_aggregation(
            data_dict=data_dict,
            source_scale=source_scale,
            target_scale=target_scale,
            variables=variables,
            strategy=strategy,
            mode=mode,
            metadata=metadata,
            var_metadata=var_metadata,
        )

        # One call may mix bases: each variable gets its own method from its declared basis.
        strategies = self._variable_strategies(var_metadata, mode, strategy)

        if not data_dict:
            logger.debug("Empty data provided for aggregation")
            return self._empty_result(
                operation="aggregate",
                source_scale=source_scale,
                target_scale=target_scale,
                mode=mode,
            )

        n_source_units = self._infer_n_source_units(data_dict)
        resolved_topology = self._resolve_aggregate_topology(
            source_scale=source_scale,
            target_scale=target_scale,
            n_source_units=n_source_units,
            topology=topology,
        )
        hierarchy_index, aggregator = self._build_hierarchy_runtime(
            source_scale=source_scale,
            target_scale=target_scale,
            n_source_units=resolved_topology["n_source_units"],
            topology=resolved_topology,
            mode=mode,
        )
        getattr(
            hierarchy_index,
            "n_target_units",
            resolved_topology.get("n_target_units", 1) if resolved_topology else 1,
        )

        try:
            # The aggregator receives the resolved variable metadata, so each field is
            # aggregated on its explicit spatial basis (catalog spatial_basis or declared type).
            aggregated_data, aggregation_audit = aggregator.aggregate(
                data=data_dict,
                variables=variables,
                strategy={var: item.value for var, item in strategies.items()},
                mode=mode.value,
                variable_metadata=var_metadata,
            )
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Hierarchical aggregation failed for {source_scale} -> {target_scale}: {exc}"
            ) from exc

        # Orphan children (no parent) are excluded from the covered-support ledger; their
        # amount is reported by the aggregator as uncovered_source_amount.
        orphan_children = (hierarchy_index.get_hierarchy_matrix() > 0).sum(dim=1) == 0
        constraints_applied = self._apply_constraints(
            data=aggregated_data,
            variables=variables,
            metadata=metadata,
            var_metadata=var_metadata,
        )
        conservation_audit = self._build_conservation_audit(
            operation="aggregate",
            variables=variables,
            var_metadata=var_metadata,
            source_data=data_dict,
            target_data=aggregated_data,
            mode=mode,
            source_areas=hierarchy_index.get_source_areas(),
            target_areas=hierarchy_index.get_target_areas(),
            uncovered_source=orphan_children,
        )

        updated_metadata = manager._sync_cf_metadata(
            operation="aggregate",
            source_scale=source_scale,
            target_scale=target_scale,
            metadata=metadata,
            var_metadata=var_metadata,
            data=aggregated_data,
        )
        audit = self._build_scaling_audit(
            operation="aggregate",
            source_scale=source_scale,
            target_scale=target_scale,
            variables=variables,
            mode=mode,
            strategies=strategies,
            operation_audit=aggregation_audit,
            conservation_audit=conservation_audit,
            updated_metadata=updated_metadata,
            constraints_applied=constraints_applied,
        )
        manager._record_scaling_audit(
            audit,
            conservation_details=conservation_audit,
            cf_metadata=updated_metadata,
        )

        if isinstance(data, Tensor):
            return aggregated_data[variables[0]], audit
        return aggregated_data, audit

    def disaggregate(
        self,
        data: Union[Tensor, Dict[str, Tensor]],
        source_scale: str,
        target_scale: str,
        variables: Optional[List[str]] = None,
        mode: Optional[ScalingMode] = None,
        strategy: Optional[ScalingStrategy] = None,
        metadata: Optional[Dict[str, Any]] = None,
        topology: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Union[Tensor, Dict[str, Tensor]], ScalingAudit]:
        manager = self.manager
        mode = mode or manager.default_mode
        self._validate_transition(source_scale, target_scale, "down")
        self._ensure_hierarchical_components()

        data_dict, variables = self._normalize_input_payload(data, variables)
        var_metadata = self._resolve_variable_metadata(
            source_scale=source_scale,
            variables=variables,
            metadata=metadata,
        )

        # One call may mix bases: each variable gets its own method from its declared basis.
        strategies = self._variable_strategies(var_metadata, mode, strategy)

        # Temporal alias pairs fail closed here; spatial and conservative alias pairs continue on
        # this path like any other pair.
        manager._handle_alias_disaggregation(
            data_dict=data_dict,
            source_scale=source_scale,
            target_scale=target_scale,
            variables=variables,
            strategy=self._call_strategy(strategies),
            mode=mode,
            metadata=metadata,
            var_metadata=var_metadata,
        )

        if not data_dict:
            return self._empty_result(
                operation="disaggregate",
                source_scale=source_scale,
                target_scale=target_scale,
                mode=mode,
                emit_warning=True,
            )

        n_input_units = self._infer_n_source_units(data_dict)
        resolved_topology, n_source_units = self._resolve_disaggregate_topology(
            source_scale=source_scale,
            target_scale=target_scale,
            n_input_units=n_input_units,
            topology=topology,
        )
        hierarchy_index, aggregator = self._build_hierarchy_runtime(
            source_scale=target_scale,
            target_scale=source_scale,
            n_source_units=n_source_units,
            topology=resolved_topology,
            mode=mode,
        )
        try:
            disaggregated_data, disaggregation_audit = aggregator.disaggregate(
                data=data_dict,
                variables=variables,
                strategy={var: item.value for var, item in strategies.items()},
                mode=mode.value,
                variable_metadata=var_metadata,
            )
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Hierarchical disaggregation failed for {target_scale} -> {source_scale}: {exc}"
            ) from exc

        constraints_applied = self._apply_constraints(
            data=disaggregated_data,
            variables=variables,
            metadata=metadata,
            var_metadata=var_metadata,
        )
        # Parents are the source of a disaggregation: their areas weight the source side.
        conservation_audit = self._build_conservation_audit(
            operation="disaggregate",
            variables=variables,
            var_metadata=var_metadata,
            source_data=data_dict,
            target_data=disaggregated_data,
            mode=mode,
            source_areas=hierarchy_index.get_target_areas(),
            target_areas=hierarchy_index.get_source_areas(),
        )

        updated_metadata = manager._sync_cf_metadata(
            operation="disaggregate",
            source_scale=source_scale,
            target_scale=target_scale,
            metadata=metadata,
            var_metadata=var_metadata,
            data=disaggregated_data,
        )
        audit = self._build_scaling_audit(
            operation="disaggregate",
            source_scale=source_scale,
            target_scale=target_scale,
            variables=variables,
            mode=mode,
            strategies=strategies,
            operation_audit=disaggregation_audit,
            conservation_audit=conservation_audit,
            updated_metadata=updated_metadata,
            constraints_applied=constraints_applied,
        )
        manager._record_scaling_audit(
            audit,
            conservation_details=conservation_audit,
            cf_metadata=updated_metadata,
        )

        if isinstance(data, Tensor):
            return disaggregated_data[variables[0]], audit
        return disaggregated_data, audit
