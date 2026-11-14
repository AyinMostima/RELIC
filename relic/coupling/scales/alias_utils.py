"""Whole-support reduction and conservative remap utilities shared across scale adapters.

The module name is historical; no scale-name alias path remains. ``aggregate_spatial_alias``
is the one whole-support reduction (``SpatialAggregation`` and the graph-edge
``SupportAggregationAdapter``); ``remap_conservative_alias`` is the overlap remap of
``ConservativeRemap``.
"""

import math
from typing import Dict, FrozenSet, List, Optional, Tuple

import torch
from torch import Tensor

from ..core.conservation_checker import ConservationDiagnostics, check_mass_conservation
from ..core.policy import ScalingMode
from .types import ScalingStrategy, ScalingAudit, SpatialQuantityBasis, convert_spatial_basis

# Caller strategies that name the whole-support law of a declared basis. For one target support
# every listed strategy denotes the same law: the conservative and flow-conserving kernels of an
# area density are its area-weighted mean and those of a cell total are its sum
# (HierarchicalAggregator applies the same kernels per parent). Any other strategy conflicts with
# the basis and raises; a strategy never replaces the law of a basis.
_BASIS_STRATEGIES: Dict[SpatialQuantityBasis, FrozenSet[ScalingStrategy]] = {
    SpatialQuantityBasis.INTENSIVE: frozenset(
        {ScalingStrategy.AREA_WEIGHTED_MEAN, ScalingStrategy.CONSERVATIVE}
    ),
    SpatialQuantityBasis.AREA_DENSITY: frozenset(
        {
            ScalingStrategy.AREA_WEIGHTED_MEAN,
            ScalingStrategy.CONSERVATIVE,
            ScalingStrategy.FLOW_CONSERVING,
        }
    ),
    SpatialQuantityBasis.CELL_TOTAL: frozenset(
        {
            ScalingStrategy.AREA_WEIGHTED_SUM,
            ScalingStrategy.CONSERVATIVE,
            ScalingStrategy.FLOW_CONSERVING,
        }
    ),
    SpatialQuantityBasis.CATEGORICAL: frozenset({ScalingStrategy.MODE}),
}

# Name of the whole-support law of each basis, recorded in the audit.
_REDUCTION_NAMES: Dict[SpatialQuantityBasis, str] = {
    SpatialQuantityBasis.INTENSIVE: "area_weighted_mean",
    SpatialQuantityBasis.AREA_DENSITY: "area_weighted_mean",
    SpatialQuantityBasis.CELL_TOTAL: "sum",
    SpatialQuantityBasis.CATEGORICAL: "area_weighted_mode",
}


def aggregate_spatial_alias(
    data_dict: Dict[str, Tensor],
    variable_bases: Dict[str, SpatialQuantityBasis],
    source_scale: str,
    target_scale: str,
    mode: ScalingMode,
    device: torch.device,
    enforce_conservation: bool,
    tolerance: float,
    weights: Optional[Tensor] = None,
    strategy: Optional[ScalingStrategy] = None,
) -> Tuple[Dict[str, Tensor], ScalingAudit]:
    """Reduce every field over its whole support by its declared basis and audit the amounts.

    Every key of ``data_dict`` is reduced to one value by the law of its declared basis
    (``_reduce_on_basis``); ``strategy`` may only name that law (``_BASIS_STRATEGIES``). The
    ledger of every amount variable (cell_total, area_density) is evaluated from the returned
    value; the strict path (``enforce_conservation`` or ``ScalingMode.STRICT_CONSERVATIVE``)
    raises when the residual exceeds the accepted residual, and otherwise the residual is
    reported. Nothing is rescaled.
    """
    mode_enum = ScalingMode(mode) if isinstance(mode, str) else mode
    if strategy is not None and not isinstance(strategy, ScalingStrategy):
        raise TypeError(f"strategy must be a ScalingStrategy or None; got {strategy!r}.")
    tolerance_value = float(tolerance)
    if not math.isfinite(tolerance_value) or tolerance_value <= 0.0:
        raise ValueError(f"The conservation tolerance must be finite and positive; got {tolerance!r}.")
    strict = bool(enforce_conservation or mode_enum is ScalingMode.STRICT_CONSERVATIVE)

    reduced: Dict[str, Tensor] = {}
    ledgers: Dict[str, ConservationDiagnostics] = {}
    quantity_basis: Dict[str, str] = {}
    reductions: Dict[str, str] = {}
    weights_applied: Dict[str, bool] = {}
    warnings: List[str] = []
    for var, values in data_dict.items():
        if not isinstance(values, Tensor):
            raise TypeError(
                f"Variable {var!r} is not a tensor; the whole-support reduction needs one value per cell."
            )
        if var not in variable_bases:
            raise ValueError(f"Variable {var!r} has no declared spatial basis for the whole-support reduction.")
        if values.numel() == 0:
            raise ValueError(f"Variable {var!r} has no cells to reduce.")
        basis = SpatialQuantityBasis.parse(variable_bases[var])
        if strategy is not None and strategy not in _BASIS_STRATEGIES[basis]:
            allowed = sorted(item.value for item in _BASIS_STRATEGIES[basis])
            raise ValueError(
                f"Strategy {strategy.value!r} conflicts with the declared basis {basis.value!r} of "
                f"{var!r}: its whole-support law ({_REDUCTION_NAMES[basis]}) is named only by {allowed}."
            )
        field = values.to(device)
        field_weights = _validated_weights(weights, field, var)
        if basis is SpatialQuantityBasis.AREA_DENSITY and field_weights is None:
            raise ValueError(
                f"Variable {var!r} is an area density; its support mean sum_i x_i w_i / sum_i w_i "
                "needs the support weights (the cell areas A_i, or a normalized support measure) "
                "passed as weights."
            )
        result = _reduce_on_basis(field, field_weights, basis)
        reduced[var] = result
        quantity_basis[var] = basis.value
        reductions[var] = _REDUCTION_NAMES[basis]
        weights_applied[var] = field_weights is not None and basis is not SpatialQuantityBasis.CELL_TOTAL
        if not basis.is_extensive:
            continue
        diagnostics = _amount_ledger(var, basis, field, field_weights, result, tolerance_value)
        ledgers[var] = diagnostics
        if not diagnostics.is_conserved:
            message = (
                f"Whole-support reduction of {var!r} ({basis.value}) {source_scale}->{target_scale} "
                f"violates conservation: ledger residual {diagnostics.absolute_error:.3e} (relative "
                f"{diagnostics.relative_error:.3e}) exceeds the accepted residual "
                f"{diagnostics.metadata['accepted_residual']:.3e} (tolerance {tolerance_value:.3e}); "
                "the result is not rescaled."
            )
            if strict:
                raise RuntimeError(message)
            warnings.append(message)

    audit = ScalingAudit(
        operation="aggregate",
        source_scale=source_scale,
        target_scale=target_scale,
        variables=list(data_dict.keys()),
        mode=mode_enum,
        strategy=strategy,
        mass_balance_error=max((item.absolute_error for item in ledgers.values()), default=0.0),
        relative_error=max((item.relative_error for item in ledgers.values()), default=0.0),
        cf_compliant=False,
        # True only when an amount ledger was checked by the strict (raising) path.
        conservation_enforced=bool(strict and ledgers),
        physical_constraints_applied=False,
        warnings=warnings,
        metadata={
            "conservation_audit": ledgers,
            "conservation_audited_variables": sorted(ledgers),
            "quantity_basis": quantity_basis,
            "output_basis": dict(quantity_basis),
            "reduction": reductions,
            "weights_applied": weights_applied,
            "cf_validation_scope": "not_evaluated",
            "dataset_conformance_checked": False,
        },
    )
    return reduced, audit


def _validated_weights(weights: Optional[Tensor], field: Tensor, var: str) -> Optional[Tensor]:
    """Return the caller's support weights on the field's device, or None when none are given."""
    if weights is None:
        return None
    if not isinstance(weights, Tensor):
        raise TypeError(f"Spatial weights must be a tensor with the shape of {var!r}.")
    if tuple(weights.shape) != tuple(field.shape):
        raise ValueError(
            f"Spatial weights of shape {tuple(weights.shape)} do not match the shape "
            f"{tuple(field.shape)} of {var!r}; pass one weight per cell."
        )
    support = weights.to(device=field.device)
    if not bool(torch.isfinite(support).all()):
        raise ValueError(f"Spatial weights of {var!r} must be finite.")
    if bool((support < 0).any()):
        raise ValueError(f"Spatial weights of {var!r} must be non-negative measures of support.")
    return support


def _compute_dtype(values: Tensor) -> torch.dtype:
    """Floating dtype of the reduction arithmetic: the field's own, float64 for integer fields."""
    return values.dtype if torch.is_floating_point(values) else torch.float64


def _positive_weight_total(weights: Tensor) -> Tensor:
    """Return sum_i w_i for a weighted mean; the theory contract requires it to be positive."""
    # References: docs/source/theory/scale_coupling.md sections 2.2 and 2.7 (sum_i a_i > 0).
    # Mathematical expression: S = sum_i w_i; the mean is sum_i w_i x_i / S.
    # Parameter meanings: w_i caller-supplied weights (area, mass, volume or uniform).
    # Units: w_i in the caller's weight units; S shares them and cancels in the mean.
    # Domain/constraints: S > 0; a non-positive total means the weights carry no support and
    #   the mean is undefined, so it is rejected instead of being replaced by max(S, 1).
    # Sign convention: weights are non-negative measures of support.
    total = weights.sum()
    total_value = float(total.detach())
    if not math.isfinite(total_value) or total_value <= 0.0:
        raise ValueError(
            "Weighted mean requires a positive weight total; got "
            f"sum_i w_i = {total_value!r}."
        )
    return total


def _reduce_on_basis(
    values: Tensor,
    weights: Optional[Tensor],
    basis: SpatialQuantityBasis,
) -> Tensor:
    """The whole-support law of one declared basis: one value per field."""
    # References: Jones, P.W. (1999) Mon. Weather Rev. 127:2204-2210 (area-weighted cell
    #   means); CF-1.8 cell_methods "area: mean", "area: sum" and "area: mode"; Friedl and
    #   Brodley (1997) IEEE Trans. Geosci. Remote Sens. 35:716-724 (majority-area class);
    #   Kljun et al. (2015) Geosci. Model Dev. 8:3695-3713 (footprint-weighted flux
    #   F_fp = sum_i alpha_i F_i); docs/source/theory/scale_coupling.md sections 2.2 and 2.3.2.
    # Mathematical expression: intensive and area_density  phi = sum_i w_i x_i / sum_i w_i;
    #   cell_total  Q = sum_i X_i; categorical  c* = argmax_c sum_{i: x_i = c} w_i (ties to the
    #   smallest code, WeightOps.weighted_mode); w_i = 1 for intensive and categorical fields
    #   given without weights.
    # Parameter meanings: x_i value of cell i over the flattened index set; X_i amount of cell
    #   i; w_i support measure of cell i (its area, or a normalized support weight such as a
    #   flux-footprint weight); phi, Q and c* the value of the whole support.
    # Units: w in the caller's support-measure units (m2 for areas, 1 for normalized weights);
    #   phi keeps the units of x (an area density stays an amount per unit area); Q keeps the
    #   units of X; class codes are dimensionless.
    # Domain/constraints: the weights were validated by the caller (the field's shape, finite,
    #   non-negative); a mean or a mode needs sum_i w_i > 0; cell totals take no weights (every
    #   cell lies in the support); the arithmetic runs in the field's floating dtype (integer
    #   fields in float64; class codes are returned unchanged).
    # Sign convention: values keep their sign; weights are non-negative measures of support.
    flat = values.reshape(-1)
    compute_dtype = _compute_dtype(values)
    if basis is SpatialQuantityBasis.CATEGORICAL:
        from .hierarchical.weight_ops import WeightOps

        share = (
            torch.ones(flat.shape, dtype=compute_dtype, device=flat.device)
            if weights is None
            else weights.reshape(-1).to(device=flat.device, dtype=compute_dtype)
        )
        _positive_weight_total(share)
        return WeightOps.weighted_mode(flat, share.unsqueeze(1))[0]
    flat = flat.to(dtype=compute_dtype)
    if basis is SpatialQuantityBasis.CELL_TOTAL:
        return flat.sum()
    support = (
        torch.ones_like(flat)
        if weights is None
        else weights.reshape(-1).to(device=flat.device, dtype=compute_dtype)
    )
    return (flat * support).sum() / _positive_weight_total(support)


def _amount_ledger(
    var: str,
    basis: SpatialQuantityBasis,
    values: Tensor,
    weights: Optional[Tensor],
    result: Tensor,
    tolerance: float,
) -> ConservationDiagnostics:
    """Ledger of one amount variable, evaluated from the value the reduction returned."""
    # References: Jones, P.W. (1999) Mon. Weather Rev. 127:2204-2210 (conservative ledgers);
    #   Higham (2002) section 4.2 (rounding bound of the reduction); the shared
    #   check_mass_conservation and ledger_acceptance owner
    #   (relic/coupling/core/conservation_checker.py), the one acceptance rule of the
    #   hierarchical facade ledger as well.
    # Mathematical expression: area_density  L_src = sum_i x_i w_i, L_tgt = phi W with
    #   W = sum_i w_i; cell_total  L_src = sum_i X_i, L_tgt = Q; e = |L_tgt - L_src|,
    #   epsilon = e / max(|L_src|, 1e-12); activity a = sum_i |x_i w_i| (sum_i |X_i|);
    #   accepted iff e <= max(tol max(|L_src|, 1e-12), gamma_(2n+2) a) (ledger_acceptance with
    #   the dtype of the reduction and n cells).
    # Parameter meanings: phi, Q returned support value; x_i, X_i source values; w_i support
    #   measures; W support measure of the whole support (the target); tol parameter-backed
    #   relative tolerance (coupling.scales.defaults.conservation_tolerance); gamma the
    #   rounding allowance of a floating-point reduction of 2n + 2 rounded steps.
    # Units: L, e and a in the amount unit (units of x times units of w for an area density,
    #   units of X for a cell total); epsilon, tol and gamma dimensionless.
    # Domain/constraints: amount bases only; the ledger is evaluated in float64 from the
    #   returned value, so a wrong result is never reported with a zero residual; the
    #   rounding allowance only admits the rounding of the reduction's own dtype.
    # Sign convention: signed amounts; e, a and epsilon are non-negative.
    source = values.detach().reshape(-1).to(torch.float64)
    target = result.detach().reshape(1).to(torch.float64)
    support_measure: Optional[float] = None
    if basis is SpatialQuantityBasis.AREA_DENSITY:
        support = weights.detach().reshape(-1).to(device=source.device, dtype=torch.float64)
        measure = support.sum()
        checked = check_mass_conservation(
            source,
            target,
            source_areas=support,
            target_areas=measure.reshape(1),
            tolerance=tolerance,
            variable_name=var,
            rounding_dtype=_compute_dtype(values),
        )
        support_measure = float(measure.item())
    else:
        checked = check_mass_conservation(
            source,
            target,
            tolerance=tolerance,
            variable_name=var,
            rounding_dtype=_compute_dtype(values),
        )
    return ConservationDiagnostics(
        property_name=var,
        is_conserved=bool(checked["conserved"]),
        absolute_error=float(checked["absolute_error"]),
        relative_error=float(checked["relative_error"]),
        source_total=float(checked["source_total"]),
        target_total=float(checked["target_total"]),
        tolerance=tolerance,
        metadata={
            "spatial_basis": basis.value,
            "support_measure": support_measure,
            "activity": float(checked["activity"]),
            "rounding_allowance": float(checked["rounding_allowance"]),
            "accepted_residual": float(checked["accepted_residual"]),
        },
    )


def remap_conservative_alias(
    data_dict: Dict[str, Tensor],
    overlap_areas: Tensor,
    source_areas: Tensor,
    variable_bases: Dict[str, SpatialQuantityBasis],
    source_scale: str,
    target_scale: str,
    mode: ScalingMode,
    device: torch.device,
    enforce_conservation: bool,
    tolerance: float,
    operation: str = "remap",
    target_areas: Optional[Tensor] = None,
) -> Tuple[Dict[str, Tensor], ScalingAudit]:
    """First-order conservative remap on explicit overlap geometry (Jones, 1999)."""
    # References: Jones, P.W. (1999) Mon. Weather Rev. 127:2204-2210, first-order
    #   conservative remapping; SCRIP user guide (Jones, 1998) "fracarea" and "destarea"
    #   normalizations; docs/source/theory/scale_coupling.md sections 2.3 and 2.3.2.
    # Mathematical expression: with overlap areas A_iJ, source areas A_i, target areas A_J,
    #   covered target areas C_J = sum_i A_iJ and coverage fractions f_i = sum_J A_iJ / A_i:
    #   cell_total    Y_J = sum_i X_i A_iJ / A_i
    #   area_density  X_i = x_i A_i (shared convert_spatial_basis), Y_J as above,
    #                 y_J = Y_J / A_J = sum_i x_i A_iJ / A_J        (destarea)
    #   intensive     y_J = sum_i x_i A_iJ / C_J                    (fracarea)
    #   covered-support ledger  L_cov = sum_i X_i f_i  (extensive) | sum_i x_i A_i f_i (intensive)
    #   target ledger           L_tgt = sum_J Y_J      (extensive) | sum_J y_J C_J     (intensive)
    #   unmapped source amount  U = sum_i X_i (1 - f_i) (extensive) | sum_i x_i A_i (1 - f_i)
    #   residual e = |L_tgt - L_cov|, relative error e / max(|L_cov|, 1e-12).
    # Parameter meanings: X_i/Y_J cell totals; x_i/y_J area densities or intensive values;
    #   A_iJ overlap area of source i with target J; A_i source area; A_J target area;
    #   C_J covered target area; f_i fraction of source i inside the target support; U the
    #   source amount outside the target support, reported per variable and never routed
    #   into covered targets.
    # Units: A_iJ, A_i, A_J, C_J [m2]; f_i [1]; X, Y and U in the amount unit (for example
    #   kg); x and y of an area density in amount m-2; intensive values keep their units and
    #   their ledger is an area integral [value m2].
    # Domain/constraints: A_iJ >= 0; A_i > 0; sum_J A_iJ <= A_i (1 + tolerance); when A_J
    #   is supplied, A_J > 0 and C_J <= A_J (1 + tolerance); area_density needs A_J; an
    #   intensive field needs C_J > 0 for every target; categorical fields are rejected; every
    #   requested variable must be a tensor with n_source elements; a relative error above
    #   tolerance raises when enforce_conservation is set; no fallback reduction exists.
    # Sign convention: values pass through with their sign; U is signed like the field;
    #   nothing is clipped here.
    overlap = torch.as_tensor(overlap_areas).to(device=device, dtype=torch.float64)
    if overlap.dim() != 2:
        raise ValueError("overlap_areas must be a 2-D tensor [n_source, n_target].")
    n_source, n_target = overlap.shape
    if not torch.isfinite(overlap).all() or bool((overlap < 0).any()):
        raise ValueError("overlap_areas must be finite and non-negative.")
    areas = torch.as_tensor(source_areas).to(device=device, dtype=torch.float64).reshape(-1)
    if areas.numel() != n_source:
        raise ValueError(
            f"source_areas has {areas.numel()} entries but overlap_areas has {n_source} source rows."
        )
    if not torch.isfinite(areas).all() or bool((areas <= 0).any()):
        raise ValueError("source_areas must be finite and positive.")
    coverage = overlap.sum(dim=1)
    excess = coverage > areas * (1.0 + tolerance)
    if bool(excess.any()):
        offenders = torch.nonzero(excess).reshape(-1).tolist()
        raise ValueError(
            "overlap_areas row sums exceed source_areas for source cells "
            f"{offenders[:8]}; the overlap geometry is inconsistent."
        )
    covered_target = overlap.sum(dim=0)
    fraction = coverage / areas
    covered_source_fraction = float((coverage.sum() / areas.sum()).item())
    target_area_tensor: Optional[Tensor] = None
    if target_areas is not None:
        target_area_tensor = torch.as_tensor(target_areas).to(device=device, dtype=torch.float64).reshape(-1)
        if target_area_tensor.numel() != n_target:
            raise ValueError(
                f"target_areas has {target_area_tensor.numel()} entries but overlap_areas has {n_target} target columns."
            )
        if not torch.isfinite(target_area_tensor).all() or bool((target_area_tensor <= 0).any()):
            raise ValueError("target_areas must be finite and positive.")
        target_excess = covered_target > target_area_tensor * (1.0 + tolerance)
        if bool(target_excess.any()):
            offenders = torch.nonzero(target_excess).reshape(-1).tolist()
            raise ValueError(
                "overlap_areas column sums exceed target_areas for target cells "
                f"{offenders[:8]}; the overlap geometry is inconsistent."
            )

    remapped: Dict[str, Tensor] = {}
    mass_error = 0.0
    relative_error = 0.0
    warnings: List[str] = []
    basis_report: Dict[str, str] = {}
    uncovered_report: Dict[str, float] = {}
    for var, tensor in data_dict.items():
        if not isinstance(tensor, Tensor):
            raise ValueError(f"Variable {var} is not a tensor and cannot be remapped.")
        values = tensor.to(device=device, dtype=torch.float64).reshape(-1)
        if values.numel() != n_source:
            raise ValueError(
                f"Variable {var} has {values.numel()} source values but overlap_areas has {n_source} rows."
            )
        if var not in variable_bases:
            raise ValueError(f"Variable {var} has no declared spatial basis for the conservative remap.")
        basis = SpatialQuantityBasis.parse(variable_bases[var])
        if basis.is_extensive:
            if basis is SpatialQuantityBasis.AREA_DENSITY and target_area_tensor is None:
                raise ValueError(
                    f"Variable {var} is an area density; its target densities y_J = Y_J / A_J need "
                    "target_areas [n_target] (or source_grid and target_grid)."
                )
            totals = convert_spatial_basis(values, areas, basis, SpatialQuantityBasis.CELL_TOTAL)
            target_totals = torch.matmul((totals / areas).unsqueeze(0), overlap).squeeze(0)
            covered_ledger = (totals * fraction).sum()
            target_ledger = target_totals.sum()
            uncovered_amount = (totals * (1.0 - fraction)).sum()
            if basis is SpatialQuantityBasis.AREA_DENSITY:
                target = convert_spatial_basis(
                    target_totals, target_area_tensor, SpatialQuantityBasis.CELL_TOTAL, basis
                )
            else:
                target = target_totals
        elif basis is SpatialQuantityBasis.INTENSIVE:
            uncovered = covered_target <= 0
            if bool(uncovered.any()):
                offenders = torch.nonzero(uncovered).reshape(-1).tolist()
                raise ValueError(
                    f"Target cell(s) {offenders[:8]} receive no source overlap; the intensive "
                    f"field {var} is undefined there."
                )
            target = torch.matmul(values.unsqueeze(0), overlap).squeeze(0) / covered_target
            covered_ledger = (values * coverage).sum()
            target_ledger = (target * covered_target).sum()
            uncovered_amount = (values * (areas - coverage)).sum()
        else:
            raise ValueError(
                f"Variable {var} has spatial basis {basis.value!r}; the conservative remap is "
                "defined only for intensive, cell_total and area_density fields."
            )
        covered_value = float(covered_ledger.detach().item())
        diff = abs(covered_value - float(target_ledger.detach().item()))
        mass_error = max(mass_error, diff)
        relative_error = max(relative_error, diff / max(abs(covered_value), 1e-12))
        basis_report[var] = basis.value
        uncovered_report[var] = float(uncovered_amount.detach().item())
        out_dtype = tensor.dtype if torch.is_floating_point(tensor) else torch.float64
        remapped[var] = target.to(dtype=out_dtype)

    if covered_source_fraction < 1.0 - tolerance:
        warnings.append(
            f"only {covered_source_fraction:.6f} of the source area lies inside the target support; "
            f"the uncovered source amounts {uncovered_report} are not remapped and are reported "
            "as uncovered_source_amount"
        )
    if enforce_conservation and relative_error > tolerance:
        raise RuntimeError(
            f"Conservative remap {source_scale}->{target_scale} violates conservation: "
            f"relative error {relative_error:.3e} exceeds tolerance {tolerance:.3e}."
        )
    audit = ScalingAudit(
        operation=operation,
        source_scale=source_scale,
        target_scale=target_scale,
        variables=list(data_dict.keys()),
        mode=mode,
        strategy=ScalingStrategy.CONSERVATIVE,
        mass_balance_error=mass_error,
        relative_error=relative_error,
        cf_compliant=False,
        conservation_enforced=enforce_conservation,
        physical_constraints_applied=False,
        warnings=warnings,
        metadata={
            "alias": "conservative_remap",
            "normalization": {
                var: ("fracarea" if basis == SpatialQuantityBasis.INTENSIVE.value else "destarea")
                for var, basis in basis_report.items()
            },
            "n_source": int(n_source),
            "n_target": int(n_target),
            "covered_source_fraction": covered_source_fraction,
            "quantity_basis": basis_report,
            "uncovered_source_amount": uncovered_report,
            "cf_validation_scope": "not_evaluated",
            "dataset_conformance_checked": False,
        },
    )
    return remapped, audit
