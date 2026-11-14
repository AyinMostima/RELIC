"""Delegated spatial regridding helpers for ScaleManager."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from ...core.policy import ScalingMode
from ..spatial.factory import check_regrid_method_for_basis
from ..spatial.grid_utils import GridDescriptor, content_fingerprint
from ..spatial.regrid_conservative import covered_support_acceptance
from ..types import (
    ScalingAudit,
    ScalingStrategy,
    SpatialQuantityBasis,
    convert_spatial_basis,
    resolve_spatial_basis,
)

logger = logging.getLogger(__name__)

# The spatial methods of ScaleManager.regrid_variable and the audit strategy of each: the one
# table read by the method check and by the audit (an unknown or missing method raises; none is
# assumed).
_METHOD_STRATEGIES: Dict[str, ScalingStrategy] = {
    "conservative": ScalingStrategy.CONSERVATIVE,
    "bilinear": ScalingStrategy.BILINEAR,
    "nearest": ScalingStrategy.NEAREST,
}


def _grid_descriptors_are_identical(
    source_grid: GridDescriptor,
    target_grid: GridDescriptor,
) -> bool:
    """Return whether two descriptors define the same discrete grid.

    References:
        Jones (1999), First- and second-order conservative remapping schemes.
    Mathematical expression:
        G_source = G_target implies R_G(x) = x.
    Parameter meanings:
        G_source and G_target are compared on every GridDescriptor field (topology,
        coordinates, bounds, mask, cell areas, coordinate units and CRS) through their
        content fingerprints (grid_utils.content_fingerprint, the one definition of equal
        geometry, also used by the regridder cache).
    """
    if source_grid is target_grid:
        return True
    return content_fingerprint(source_grid) == content_fingerprint(target_grid)


def validate_spatial_inputs_impl(
    self,
    var: Tensor,
    src_grid: GridDescriptor,
    tgt_grid: GridDescriptor,
    op: str,
) -> None:
    if not isinstance(var, Tensor):
        raise TypeError(f"{op}: input data must be a Tensor")
    if not isinstance(src_grid, GridDescriptor) or not isinstance(tgt_grid, GridDescriptor):
        raise TypeError(f"{op}: source_grid and target_grid must be GridDescriptor instances")
    if src_grid.dims:
        expected = math.prod(src_grid.dims)
        if var.numel() != expected:
            raise ValueError(f"{op}: data elements {var.numel()} do not match source grid cells {expected}")
    if not tgt_grid.dims:
        raise ValueError(f"{op}: target grid dimensions are missing")


def _identity_ledger(
    var_name: str,
    var_value: Tensor,
    grid: GridDescriptor,
    metadata: Optional[Dict[str, Any]],
    method: str,
    manager: Any,
    tolerance: float,
) -> Dict[str, Any]:
    """Covered-support ledger of an identity map (every cell onto itself) on the declared basis."""
    # References: docs/source/theory/scale_coupling.md section 2.3.1 (covered-support ledger);
    #   convert_spatial_basis in relic/coupling/scales/types.py (the basis owner).
    # Mathematical expression: per row b of the field, amounts X_i = convert(x_i, A_i, basis ->
    #   cell_total) for amounts and x_i A_i for intensive fields; covered = mapped = sum_i X_i;
    #   uncovered = 0; residual = 0; activity a_b = sum_i |X_i|; the report has the schema of
    #   ConservativeRegridder.last_regrid_report (rows, rounding dtype, term count 2 n + 8).
    # Parameter meanings: x_i field value; A_i cell area; X_i cell amount; n cells per row.
    # Units: A [m2]; X in the amount unit (value m2 for intensive fields).
    # Domain/constraints: a declared basis; cell areas for area densities and intensive fields; a
    #   categorical field is not remapped conservatively (check_regrid_method_for_basis); the
    #   field is returned unchanged, so its rounding dtype is its own dtype (float64 for an
    #   integer field).
    # Sign convention: amounts keep their sign.
    basis = resolve_spatial_basis(var_name, metadata or {})
    check_regrid_method_for_basis(method, basis, bool(manager.enforce_conservation))
    values = var_value.detach().to(torch.float64)
    if basis is SpatialQuantityBasis.CELL_TOTAL:
        amounts = values
    else:
        if grid.cell_areas is None:
            raise ValueError(
                f"The identity ledger of {var_name} ({basis.value}) needs the cell_areas of its grid."
            )
        areas = grid.cell_areas.to(device=values.device, dtype=torch.float64).reshape(values.shape)
        amounts = (
            convert_spatial_basis(values, areas, basis, SpatialQuantityBasis.CELL_TOTAL)
            if basis.is_extensive
            else values * areas
        )
    n_cells = int(math.prod(int(dim) for dim in grid.dims))
    rows_amounts = amounts.reshape(-1, n_cells)
    rows = [
        {
            "covered_source_amount": float(row.sum().item()),
            "absolute_residual": 0.0,
            "activity": float(row.abs().sum().item()),
        }
        for row in rows_amounts
    ]
    amount = float(rows_amounts.sum().item())
    rounding_dtype = var_value.dtype if var_value.is_floating_point() else torch.float64
    report: Dict[str, Any] = {
        "quantity_basis": basis.value,
        "covered_source_amount": amount,
        "mapped_target_amount": amount,
        "uncovered_source_amount": 0.0,
        "absolute_residual": 0.0,
        "relative_residual": 0.0,
        "activity": float(sum(row["activity"] for row in rows)),
        "rows": rows,
        "tolerance": tolerance,
        "rounding_dtype": str(rounding_dtype).replace("torch.", ""),
        "term_count": 2 * n_cells + 8,
    }
    report.update(covered_support_acceptance(report, tolerance))
    return report


def regrid_variable_impl(
    self,
    var_name: str,
    var_value: Tensor,
    source_grid: Union[GridDescriptor, Dict[str, Any]],
    target_grid: Union[GridDescriptor, Dict[str, Any]],
    strategy: str,
    metadata: Optional[Dict[str, Any]] = None,
    mode: Optional[ScalingMode] = None,
    source_scale: str = "unknown",
    target_scale: str = "unknown",
) -> Tuple[Tensor, Dict[str, Any]]:
    # The method is the caller's explicit choice (the simulator derives it from the declared
    # spatial basis, regrid_strategy_ops.determine_regrid_strategy); a missing or unknown method
    # raises instead of becoming the conservative method.
    if not isinstance(strategy, str) or strategy not in _METHOD_STRATEGIES:
        raise ValueError(
            f"regrid_variable({var_name!r}) needs the spatial method as one of "
            f"{sorted(_METHOD_STRATEGIES)}; got {strategy!r} (no method is assumed)."
        )
    self.ensure_ready()

    def _prepare_grid(grid_like: Union[GridDescriptor, Dict[str, Any]]) -> GridDescriptor:
        if isinstance(grid_like, GridDescriptor):
            return grid_like
        if isinstance(grid_like, dict):
            normalized = dict(grid_like)
            if "coordinate_system" in normalized and "coord_system" not in normalized:
                normalized["coord_system"] = normalized.pop("coordinate_system")
            if "spacing_override" in normalized and "spacing" not in normalized:
                normalized["spacing"] = normalized.pop("spacing_override")
            return GridDescriptor(**normalized)
        raise TypeError("source_grid/target_grid must be GridDescriptor or mapping")

    method = strategy
    src_grid = _prepare_grid(source_grid)
    tgt_grid = _prepare_grid(target_grid)

    if self.grid_manager:
        src_grid = self.grid_manager.ensure_areas(src_grid)
        tgt_grid = self.grid_manager.ensure_areas(tgt_grid)
    validate_spatial_inputs_impl(self, var_value, src_grid, tgt_grid, op="regrid_variable")

    factory_meta: Dict[str, Any] = {}
    identity_operation = _grid_descriptors_are_identical(src_grid, tgt_grid)
    # One source of the ledger tolerance: coupling.scales.defaults.precision.<mode>.tolerance of
    # this call's mode (the default mode when the call passes none); the conservative regridder of
    # the call is built with it, so its own checks and this audit use one tolerance.
    ledger_tolerance = self.ledger_tolerance(mode)
    # The declared basis decides the conservation ledger and, for a remap under enforcement, the
    # method rule; without caller metadata it comes from the YAML variable catalog (undeclared
    # variables raise). A non-conservative identity map needs neither.
    regridder_metadata = metadata
    if regridder_metadata is None and (
        method == "conservative" or (self.enforce_conservation and not identity_operation)
    ):
        regridder_metadata = self.get_variable_metadata(var_name, scale=source_scale)
    covered_support_report: Optional[Dict[str, Any]] = None
    if identity_operation:
        regridded = var_value
        factory_meta["operation"] = "identity"
        if method == "conservative":
            covered_support_report = _identity_ledger(
                var_name, var_value, src_grid, regridder_metadata, method, self, ledger_tolerance
            )
    elif hasattr(self.regridder_factory, "regrid_variable"):
        regridded, factory_meta = self.regridder_factory.regrid_variable(
            var_name=var_name,
            var_value=var_value,
            source_grid=src_grid,
            target_grid=tgt_grid,
            strategy=method,
            metadata=metadata,
            mode=mode or self.default_mode,
        )
    elif hasattr(self.regridder_factory, "create_regridder"):
        regridder_kwargs: Dict[str, Any] = {"method": method, "variable_metadata": regridder_metadata}
        if method == "conservative":
            regridder_kwargs["conservation_tolerance"] = ledger_tolerance
        regridder = self.regridder_factory.create_regridder(src_grid, tgt_grid, **regridder_kwargs)
        # The call maps the field from src_grid to tgt_grid: the regridder's upscale for every
        # regridder class (bilinear and conservative map source -> target in both kernels; the
        # nearest-neighbour downscale maps the other way), so no direction label is inferred from
        # cell counts and every kernel error propagates.
        regridded = regridder.upscale(var_value)
        factory_meta["method"] = method
        if method == "conservative":
            covered_support_report = getattr(regridder, "last_regrid_report", None)
        else:
            # Target coverage of the bilinear and nearest-neighbour regridders (uncovered targets
            # are NaN, never filled).
            coverage = getattr(regridder, "coverage_report", None)
            if coverage is not None:
                factory_meta["coverage"] = dict(coverage)
    else:
        raise AttributeError("regridder_factory must provide create_regridder or regrid_variable")

    if isinstance(regridded, dict):
        extra_meta = regridded.get("_regrid_metadata", {})
        if isinstance(extra_meta, dict):
            factory_meta.update(extra_meta)
        inline_meta = regridded.get("metadata", {})
        if isinstance(inline_meta, dict):
            factory_meta.update(inline_meta)

        tensor_payload = None
        for payload_key in ("regridded", "regridded_data", "data", "result"):
            candidate = regridded.get(payload_key)
            if isinstance(candidate, Tensor):
                tensor_payload = candidate
                break
        if tensor_payload is None:
            raise TypeError(
                "Regridding output dictionary must contain a Tensor under one of: "
                "'regridded', 'regridded_data', 'data', or 'result'."
            )
        regridded = tensor_payload

    grid_info_src = {"shape": src_grid.dims}
    grid_info_tgt = {"shape": tgt_grid.dims}
    variable_metadata = dict(metadata or {})
    for key, actual_shape, target in (
        ("source_grid", src_grid.dims, grid_info_src),
        ("target_grid", tgt_grid.dims, grid_info_tgt),
    ):
        declaration = variable_metadata.pop(key, None)
        if declaration is None:
            continue
        if not isinstance(declaration, dict):
            raise TypeError(f"metadata.{key} must be a mapping")
        declared_shape = tuple(declaration.get("shape", ()))
        if declared_shape != tuple(actual_shape):
            raise ValueError(
                f"metadata.{key}.shape must match the runtime grid dimensions"
            )
        target.update(declaration)

    updated_metadata: Dict[str, Any] = dict(variable_metadata)
    if factory_meta:
        updated_metadata.update(factory_meta)
    if self.cf_synchronizer and not identity_operation:
        sync_fn = getattr(self.cf_synchronizer, "sync_after_regridding", None)
        if sync_fn:
            updated_metadata = sync_fn(
                data=regridded,
                metadata=updated_metadata,
                regrid_method=method,
                source_grid=grid_info_src,
                target_grid=grid_info_tgt,
            )

    method_lower = method
    strategy_enum = _METHOD_STRATEGIES[method]
    conservation_requested = bool(
        self.enforce_conservation
        and strategy_enum == ScalingStrategy.CONSERVATIVE
    )
    conservation_evaluated = False
    conservation_passed: Optional[bool] = None
    conservation_failure: Optional[str] = None
    cf_compliant = False
    mass_balance_error = 0.0
    relative_error = 0.0
    warnings_list: List[str] = []
    conservation_details: Optional[Dict[str, Any]] = None

    if strategy_enum == ScalingStrategy.CONSERVATIVE:
        # References: Jones (1999) Mon. Weather Rev. 127:2204-2210 (covered-support ledger of a
        #   first-order conservative remap); docs/source/theory/scale_coupling.md Section 2.10;
        #   conservation_checker.ledger_acceptance through covered_support_acceptance (the one
        #   acceptance rule of every ledger).
        # Mathematical expression: per row b of the report, accepted_b iff
        #   e_b <= max(tol max(|L_cov,b|, 1e-12), gamma_k(u) a_b); the audit passes iff every row
        #   is accepted; relative error = max_b e_b / max(|L_cov,b|, 1e-12).
        # Parameter meanings: L_cov covered source amount; e the float64 residual of the returned
        #   field against it; a the activity sum_i |X_i| f_i; u the unit roundoff of the field
        #   dtype; k = 2 n + 8 rounded steps; tol the ledger tolerance of this call's mode
        #   (ScaleManager.ledger_tolerance).
        # Units: L, e, a in the amount unit of the field (value m2 for intensive fields); tol and
        #   gamma dimensionless.
        # Domain/constraints: the one audit is the covered-support ledger of the conservative
        #   regridder on the declared basis (ConservativeRegridder.last_regrid_report; the
        #   identity ledger for an identity map); partial coverage leaves the uncovered source
        #   amount out of the residual; with conservation requested a rejected ledger raises.
        # Sign convention: e and a are non-negative; amounts keep their sign.
        if covered_support_report is None:
            conservation_failure = (
                f"Conservative regridding of {var_name} produced no covered-support report; its "
                "conservation is unproven."
            )
            warnings_list.append(conservation_failure)
        else:
            conservation_evaluated = True
            acceptance = covered_support_acceptance(covered_support_report, ledger_tolerance)
            mass_balance_error = float(covered_support_report["absolute_residual"])
            relative_error = float(covered_support_report["relative_residual"])
            conservation_passed = bool(acceptance["accepted"])
            conservation_details = {
                var_name: {
                    "conserved": conservation_passed,
                    "absolute_error": mass_balance_error,
                    "relative_error": relative_error,
                    "source_total": float(covered_support_report["covered_source_amount"]),
                    "target_total": float(covered_support_report["mapped_target_amount"]),
                    "uncovered_source_amount": float(covered_support_report["uncovered_source_amount"]),
                    "quantity_basis": covered_support_report["quantity_basis"],
                }
            }
            if not conservation_passed:
                conservation_failure = (
                    f"Conservative regridding of {var_name} leaves a covered-support residual "
                    f"{mass_balance_error:.3e} (relative {relative_error:.3e}) above the accepted "
                    f"residual {acceptance['accepted_residual']:.3e} (tolerance {ledger_tolerance:.2e} of "
                    f"mode {(mode or self.default_mode).value}) on the "
                    f"{covered_support_report['quantity_basis']} basis."
                )
                warnings_list.append(conservation_failure)

    conservation_enforced = bool(
        conservation_requested
        and conservation_evaluated
        and conservation_passed
    )

    audit = ScalingAudit(
        operation="identity" if identity_operation else "regrid",
        source_scale=source_scale,
        target_scale=target_scale,
        variables=[var_name],
        mode=mode or self.default_mode,
        strategy=strategy_enum,
        mass_balance_error=mass_balance_error,
        relative_error=relative_error,
        cf_compliant=cf_compliant,
        conservation_enforced=conservation_enforced,
        physical_constraints_applied=False,
        warnings=warnings_list,
        metadata={
            "cf_metadata": updated_metadata,
            "method": method_lower,
            "cf_validation_scope": "not_evaluated",
            "dataset_conformance_checked": False,
            "conservation_requested": conservation_requested,
            "conservation_evaluated": conservation_evaluated,
            "conservation_passed": conservation_passed,
            "conservation_tolerance": (
                ledger_tolerance if strategy_enum == ScalingStrategy.CONSERVATIVE else None
            ),
        },
    )
    self._record_scaling_audit(
        audit,
        conservation_details=conservation_details,
        cf_metadata=updated_metadata,
    )

    if conservation_requested and conservation_failure is not None:
        raise RuntimeError(conservation_failure)

    return regridded, updated_metadata


def sync_aggregation_metadata_impl(
    self,
    var,
    metadata,
    method,
    rule,
    src,
    tgt,
    op,
    bounds=None,
):
    grid_src = {"shape": src.dims}
    grid_tgt = {"shape": tgt.dims}
    base_metadata = metadata.copy() if metadata else {}
    sync_fn = getattr(self.cf_synchronizer, "sync_after_regridding", None)
    updated = sync_fn(var, base_metadata, method, grid_src, grid_tgt) if sync_fn else base_metadata
    updated["coupling.scales"] = {"operation": op, "rule": rule, "method": method}
    if bounds:
        updated["bounds_applied"] = {"min": bounds[0], "max": bounds[1]}
    return updated
