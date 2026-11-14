"""Shared types for scale coupling facade."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from enum import Enum

import torch
from torch import Tensor

from ..core.policy import ScalingMode


class SpatialQuantityBasis(Enum):
    """Spatial representation of a field handed to a spatial scale operator.

    This is the single owner of the spatial quantity representation used by the
    extensive-capable spatial operators (ConservativeRemap, ConservativeRegridder,
    SpatialAggregation, HierarchicalAggregator). The older word ``extensive`` does
    not say whether a value is an amount per cell or an amount per unit area, so it
    is never accepted as a basis on its own.

    Members:
        INTENSIVE: a state or per-unit quantity (temperature, concentration,
            fraction); spatial operators form support-weighted means.
        CELL_TOTAL: an amount per cell X_i (for example kg per cell); the cell
            total over a support is sum_i X_i.
        AREA_DENSITY: an amount per unit horizontal area x_i (for example
            kg m-2 or W m-2); the amount of cell i is x_i A_i.
        CATEGORICAL: a class label; no weighted arithmetic is defined.
    """

    INTENSIVE = "intensive"
    CELL_TOTAL = "cell_total"
    AREA_DENSITY = "area_density"
    CATEGORICAL = "categorical"

    @property
    def is_extensive(self) -> bool:
        """True for the two amount representations (cell total and area density)."""
        return self in (SpatialQuantityBasis.CELL_TOTAL, SpatialQuantityBasis.AREA_DENSITY)

    @classmethod
    def parse(cls, value: Union["SpatialQuantityBasis", str]) -> "SpatialQuantityBasis":
        """Return the member for an enum or its exact string value.

        A bare ``extensive`` is rejected because it does not fix the area basis.
        """
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError(
                f"Spatial quantity basis must be a SpatialQuantityBasis or str, got {type(value).__name__}."
            )
        token = value.strip().lower()
        if token == "extensive":
            raise ValueError(
                "Spatial quantity basis 'extensive' is ambiguous: declare 'cell_total' "
                "(amount per cell) or 'area_density' (amount per unit area)."
            )
        for member in cls:
            if member.value == token:
                return member
        raise ValueError(
            f"Unknown spatial quantity basis {value!r}; expected one of "
            f"{[member.value for member in cls]}."
        )


_VARIABLE_TYPE_TO_BASIS = {
    "intensive": SpatialQuantityBasis.INTENSIVE,
    "categorical": SpatialQuantityBasis.CATEGORICAL,
}


def resolve_spatial_basis(
    variable: str,
    metadata: Optional[Mapping],
) -> SpatialQuantityBasis:
    """Resolve the explicit spatial basis of one variable from its metadata entry.

    Resolution order: ``spatial_basis`` (validated against ``variable_type`` when
    both are present), else ``variable_type`` ``intensive`` or ``categorical``.
    ``variable_type: extensive`` without ``spatial_basis`` and metadata without
    either key raise, because the area basis of the amount is then unknown.
    """
    if not isinstance(metadata, Mapping):
        raise ValueError(
            f"Variable {variable!r} has no metadata; declare spatial_basis "
            "(intensive, cell_total, area_density or categorical) for spatial operators."
        )
    raw_basis = metadata.get("spatial_basis")
    raw_type = metadata.get("variable_type")
    declared_type = raw_type.strip().lower() if isinstance(raw_type, str) else None
    if raw_basis is not None:
        basis = SpatialQuantityBasis.parse(raw_basis)
        if declared_type is not None:
            consistent = (
                (declared_type == "extensive" and basis.is_extensive)
                or _VARIABLE_TYPE_TO_BASIS.get(declared_type) is basis
            )
            if not consistent:
                raise ValueError(
                    f"Variable {variable!r} declares an inconsistent variable_type "
                    f"{raw_type!r} and spatial_basis {basis.value!r}."
                )
        return basis
    if declared_type == "extensive":
        raise ValueError(
            f"Variable {variable!r} is declared extensive without spatial_basis; this is "
            "ambiguous for spatial operators: declare spatial_basis 'cell_total' or 'area_density'."
        )
    if declared_type in _VARIABLE_TYPE_TO_BASIS:
        return _VARIABLE_TYPE_TO_BASIS[declared_type]
    raise ValueError(
        f"Variable {variable!r} does not declare spatial_basis or a recognised "
        f"variable_type (got {raw_type!r}); declare spatial_basis for spatial operators."
    )


def convert_spatial_basis(
    values: Tensor,
    areas: Tensor,
    source_basis: SpatialQuantityBasis,
    target_basis: SpatialQuantityBasis,
) -> Tensor:
    """Convert a field between the cell-total and area-density representations."""
    # References: docs/source/theory/scale_coupling.md section 2.3.2 (interface comparison of
    #   the extensive representations); Jones (1999) Mon. Weather Rev. 127:2204-2210, where
    #   conservative remapping integrates densities over cell areas.
    # Mathematical expression: cell_total -> area_density  x_i = X_i / A_i;
    #   area_density -> cell_total  X_i = x_i A_i; identical bases return the input unchanged.
    # Parameter meanings: X_i amount of cell i; x_i amount per unit area of cell i; A_i area
    #   of cell i (broadcast over leading batch dimensions of the field).
    # Units: A_i [m2]; X_i [amount, e.g. kg]; x_i [amount m-2]; the amount unit is carried
    #   through unchanged.
    # Domain/constraints: A_i finite and strictly positive and broadcastable to the field;
    #   intensive and categorical fields have no amount representation, so any conversion
    #   that involves them raises; no clipping or fallback is applied.
    # Sign convention: signs of X and x are identical because A_i > 0.
    source = SpatialQuantityBasis.parse(source_basis)
    target = SpatialQuantityBasis.parse(target_basis)
    if source is target:
        return values
    if not (source.is_extensive and target.is_extensive):
        raise ValueError(
            f"Cannot convert spatial basis {source.value!r} to {target.value!r}: intensive and "
            "categorical fields have no cell-total or area-density representation."
        )
    area_tensor = torch.as_tensor(areas).to(device=values.device, dtype=values.dtype)
    if not bool(torch.isfinite(area_tensor).all()) or bool((area_tensor <= 0).any()):
        raise ValueError("Cell areas for a spatial basis conversion must be finite and positive.")
    try:
        broadcast_shape = torch.broadcast_shapes(tuple(values.shape), tuple(area_tensor.shape))
    except RuntimeError as exc:
        raise ValueError(
            f"Cell areas of shape {tuple(area_tensor.shape)} do not broadcast to the field "
            f"shape {tuple(values.shape)}."
        ) from exc
    if tuple(broadcast_shape) != tuple(values.shape):
        raise ValueError(
            f"Cell areas of shape {tuple(area_tensor.shape)} do not broadcast to the field "
            f"shape {tuple(values.shape)} without enlarging it."
        )
    if source is SpatialQuantityBasis.CELL_TOTAL:
        return values / area_tensor
    return values * area_tensor


class ScalingStrategy(Enum):
    """Aggregation strategies for different variable types."""

    AREA_WEIGHTED_MEAN = "area_weighted_mean"
    AREA_WEIGHTED_SUM = "area_weighted_sum"
    FLOW_CONSERVING = "flow_conserving"
    MASS_WEIGHTED = "mass_weighted"
    VOLUME_WEIGHTED = "volume_weighted"
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    CONSERVATIVE = "conservative"
    MODE = "mode"  # Area-weighted mode of categorical fields (= AggregationMethod.MODE)


@dataclass
class ScalingAudit:
    """Audit information; scale operations never certify dataset CF conformance."""

    operation: str
    source_scale: str
    target_scale: str
    variables: List[str]
    mode: ScalingMode
    # None when the variables of one call use different strategies; the per-variable
    # strategies are then in metadata["strategy_per_variable"].
    strategy: Optional[ScalingStrategy]
    mass_balance_error: float
    relative_error: float
    cf_compliant: bool
    conservation_enforced: bool
    physical_constraints_applied: bool
    warnings: List[str]
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.cf_compliant, bool):
            raise TypeError("cf_compliant must be bool")
        if self.cf_compliant:
            raise ValueError(
                "ScalingAudit cannot claim dataset-level CF conformance"
            )
        if self.metadata.get("dataset_conformance_checked") not in {None, False}:
            raise ValueError(
                "ScalingAudit dataset_conformance_checked must remain false"
            )
        self.metadata.setdefault("dataset_conformance_checked", False)
        self.metadata.setdefault("cf_validation_scope", "not_evaluated")
