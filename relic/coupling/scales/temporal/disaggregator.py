"""Canonical temporal disaggregation adapter for RELIC.

The public disaggregation surface is retained, but executable interval mapping
is owned by :mod:`relic.coupling.scales.temporal.core`. Unsupported
reconstruction, profile, and stochastic selectors fail closed until a
canonical owner and executable scientific contract are available.

References
----------
CF Conventions 1.13, Sections 4.4, 7.1, 7.3, and Appendix E.
Jones, P. W. (1999), Mon. Weather Rev. 127, 2204-2210.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import torch
from torch import Tensor

from ....core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    ScaleCouplingFunctor,
    SpatialScale,
    TemporalScale,
    VariableKind,
)
from .core import (
    CFTimeProcessor,
    TemporalOverlapPlan,
    TemporalUnitContract,
    TimeCoordinate,
    TimeWeightGenerator,
    _materialize_device,
    apply_temporal_kernel,
    compute_temporal_mapping_residual,
    map_state_boundaries,
    update_cell_methods,
)
from ..types import SpatialQuantityBasis
from .cf_functions import identity_metadata, optional_text_argument, require_text_argument
from .types import (
    ScalingDirection,
    TemporalConfig,
    TemporalSemantics,
)

logger = logging.getLogger(__name__)


def _to_scalar(value: Any, *, param_name: str) -> float:
    """Return one finite real parameter value without a fallback."""
    if isinstance(value, Tensor):
        if value.numel() != 1 or not bool(torch.all(torch.isfinite(value)).item()):
            raise ValueError(f"Parameter '{param_name}' must be one finite scalar")
        value = value.detach().reshape(()).cpu().item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Parameter '{param_name}' must be a real scalar")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"Parameter '{param_name}' must be finite")
    return scalar


class DisaggregationMethod(Enum):
    """Public temporal reconstruction selectors."""

    UNIFORM = "uniform"
    LINEAR = "linear"
    CLIMATOLOGY_DIURNAL = "climatology_diurnal"
    SHAPE_PRESERVING_SPLINE = "spline"
    CUBIC_SPLINE = "cubic"
    EXTERNAL_DRIVER = "external_driver"
    STOCHASTIC = "stochastic"


@dataclass
class DisaggregationConfig:
    """Configuration for the retained disaggregation API.

    Every field is consumed by the piecewise-constant path. ``None`` values
    are resolved from ``coupling.scales.temporal.disaggregation.*``
    (``allow_negative``, ``use_physical_constraints``, ``time_dim``). The
    mapping tolerance is ``validation.conservation.conservation_tolerance`` and profile,
    smoothing, external-weight, and stochastic settings do not exist because
    no reconstruction path consumes them.
    """

    method: DisaggregationMethod = DisaggregationMethod.UNIFORM
    allow_negative: bool | None = None
    use_physical_constraints: bool | None = None
    time_dim: int | None = None


@dataclass
class DisaggregationResult:
    """Disaggregated values, metadata, and typed diagnostics.

    ``conservation_error`` is the maximum relative mapping residual over the
    checked output elements. It is NaN when a residual was computed but no
    output element had an accepted target, and 0.0 when the conservation level
    is ``none`` (no residual computed).
    """

    data: Tensor
    metadata: dict[str, Any]
    conservation_error: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _MappingArgumentNames:
    """Public argument names of one entry point, used in error messages.

    The internal mapping always reads ``source_*``/``target_*``; the entry
    points expose coarse/fine names whose roles depend on the direction.
    """

    source_time: str
    target_time: str
    source_bounds: str
    target_bounds: str


class TemporalDisaggregator(ScaleCouplingFunctor):
    """Thin direction-specific adapter over the canonical temporal core."""

    def __init__(
        self,
        config: TemporalConfig,
        disaggregation_config: DisaggregationConfig | None = None,
    ) -> None:
        metadata = FunctorMetadata(
            name="temporal_disaggregator",
            version="2.0.0",
            description="Canonical bounded temporal disaggregation adapter",
            category=ProcessCategory.SCALE_COUPLING,
            execution_phase=ExecutionPhase.COUPLING,
            mathematical_expression=(
                "O_ij=max(0,min(a_i+,b_j+)-max(a_i-,b_j-)); "
                "piecewise-constant refinement uses O_ij"
            ),
            references=[
                "CF Conventions Version 1.13, Sections 4.4, 7.1, and 7.3",
                "Jones, P.W. (1999), Mon. Weather Rev. 127:2204-2210",
            ],
            parameters={
                "coupling.scales.conservative.clamp_min": "Lower domain bound",
                "coupling.scales.conservative.clamp_max": "Upper domain bound",
                "validation.conservation.conservation_tolerance": (
                    "Relative tolerance of the temporal mapping amount ledger [dimensionless]; "
                    "the conservation level decides whether it is compared and whether a "
                    "violation blocks"
                ),
            },
            input_variables=["coarse_time_series"],
            output_variables=["fine_time_series"],
            conservation_laws=[],
            temporal_scale=TemporalScale.SECOND,
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.DIAGNOSTIC,
        )
        super().__init__(
            name=metadata.name,
            metadata=metadata,
            config={},
            source_scale=config.source_frequency,
            target_scale=config.target_frequency,
            coupling_type="temporal",
            device=config.device,
        )
        self.config = config
        self.disaggregation_config = disaggregation_config or DisaggregationConfig()
        conservative_ns = "coupling.scales.conservative"
        disagg_ns = "coupling.scales.temporal.disaggregation"
        self.conservative_clamp_min = self._required_float_parameter(
            "clamp_min", conservative_ns
        )
        self.conservative_clamp_max = self._required_float_parameter(
            "clamp_max", conservative_ns
        )
        if self.conservative_clamp_min > self.conservative_clamp_max:
            raise ValueError("conservative clamp_min must not exceed clamp_max")
        self._resolve_optional_configuration(disagg_ns)
        parameter_manager = self.get_parameter_manager()
        self.cf_time_processor = CFTimeProcessor(
            device=self.device, parameter_manager=parameter_manager
        )
        self.time_weight_generator = TimeWeightGenerator(
            device=self.device, parameter_manager=parameter_manager
        )

    def _required_float_parameter(self, key: str, namespace: str) -> float:
        value = self.get_parameter_value(
            key,
            namespace=namespace,
            units="dimensionless",
            validate=True,
            required=True,
        )
        return _to_scalar(value, param_name=f"{namespace}.{key}")

    def _resolve_optional_configuration(self, namespace: str) -> None:
        config = self.disaggregation_config
        if config.allow_negative is None:
            config.allow_negative = bool(
                self.get_parameter_value("allow_negative", namespace=namespace, required=True)
            )
        if config.use_physical_constraints is None:
            config.use_physical_constraints = bool(
                self.get_parameter_value(
                    "use_physical_constraints", namespace=namespace, required=True
                )
            )
        if config.time_dim is None:
            config.time_dim = int(
                self.get_parameter_value("time_dim", namespace=namespace, required=True)
            )

    def to_device(
        self, device: torch.device | str
    ) -> "TemporalDisaggregator":
        """Move the functor and all directly owned temporal helpers together."""
        target = _materialize_device(torch.device(device))
        super().to_device(target)
        self.cf_time_processor.device = target
        self.time_weight_generator.device = target
        self.time_weight_generator.cf_processor.device = target
        return self

    @staticmethod
    def _required_text(metadata: dict[str, Any], key: str) -> str:
        value = metadata.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be one explicit non-empty string")
        return value.strip()

    def _resolve_semantics(self, metadata: dict[str, Any]) -> TemporalSemantics:
        value = metadata.get("temporal_semantics", self.config.temporal_semantics)
        if value is None:
            raise ValueError("temporal_semantics is required by TemporalDisaggregator")
        try:
            semantics = (
                value
                if isinstance(value, TemporalSemantics)
                else TemporalSemantics(str(value))
            )
        except ValueError as exc:
            raise ValueError(f"Unsupported temporal_semantics: {value!r}") from exc
        if semantics in {TemporalSemantics.POINT_SAMPLE, TemporalSemantics.CATEGORICAL}:
            raise ValueError(f"{semantics.value} temporal disaggregation is unbound")
        if self.config.spatial_basis is SpatialQuantityBasis.CATEGORICAL:
            raise ValueError(
                "categorical temporal disaggregation is unbound "
                "(TemporalConfig.spatial_basis is categorical)"
            )
        return semantics

    def _require_direction(
        self,
        required: ScalingDirection,
        entry_point: str,
    ) -> None:
        """Reject a configuration declared for the opposite mapping direction."""
        if self.config.direction is not required:
            raise ValueError(
                f"TemporalConfig.direction is {self.config.direction.value!r}; "
                f"TemporalDisaggregator.{entry_point} requires {required.value!r}"
            )

    def _ensure_piecewise_constant_method(self) -> None:
        method = self.disaggregation_config.method
        if method is DisaggregationMethod.UNIFORM:
            return
        if method is DisaggregationMethod.STOCHASTIC:
            raise ValueError(
                "canonical RNG reconstruction is unavailable; stochastic "
                "disaggregation is fail-closed"
            )
        if method in {
            DisaggregationMethod.CLIMATOLOGY_DIURNAL,
            DisaggregationMethod.EXTERNAL_DRIVER,
        }:
            raise ValueError(
                "canonical profile reconstruction is unavailable; declared "
                "profile disaggregation is fail-closed"
            )
        raise ValueError(
            "canonical reconstruction is unavailable for "
            f"{method.value}; interpolation is fail-closed"
        )

    def _normalized_time_dim(self, data: Tensor) -> int:
        raw = self.disaggregation_config.time_dim
        if not isinstance(raw, int) or isinstance(raw, bool):
            raise TypeError("time_dim must be an integer")
        normalized = raw if raw >= 0 else data.dim() + raw
        if normalized < 0 or normalized >= data.dim():
            raise ValueError(f"time_dim {raw} is out of range for data rank {data.dim()}")
        return normalized

    @staticmethod
    def _validate_coordinate(coordinate: Tensor, name: str) -> None:
        if not isinstance(coordinate, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if coordinate.layout != torch.strided or coordinate.device.type == "meta":
            raise ValueError(f"{name} must be one materialized strided tensor")
        if coordinate.dim() != 1 or coordinate.numel() == 0:
            raise ValueError(f"{name} must be one non-empty one-dimensional tensor")
        if coordinate.dtype not in {torch.float32, torch.float64}:
            raise ValueError(f"{name} must use float32 or float64")
        if not bool(torch.all(torch.isfinite(coordinate)).item()):
            raise ValueError(f"{name} must contain finite values")
        if coordinate.numel() > 1 and bool(
            torch.any(coordinate[1:] <= coordinate[:-1]).item()
        ):
            raise ValueError(f"{name} must be strictly increasing")

    def _validate_inputs(
        self,
        data: Tensor,
        source_time: Tensor,
        target_time: Tensor,
        names: _MappingArgumentNames,
    ) -> int:
        if not isinstance(data, Tensor):
            raise TypeError("data must be a torch.Tensor")
        if data.layout != torch.strided or data.device.type == "meta":
            raise ValueError("data must be one materialized strided tensor")
        if data.dim() == 0 or data.numel() == 0:
            raise ValueError("data must have one non-empty time axis")
        if data.dtype not in {torch.float32, torch.float64}:
            raise ValueError("data must use float32 or float64")
        if bool(torch.any(torch.isinf(data)).item()):
            raise ValueError("data must not contain infinite or Inf values")
        if data.device != self.device:
            raise ValueError("data device must match TemporalDisaggregator device")
        self._validate_coordinate(source_time, names.source_time)
        self._validate_coordinate(target_time, names.target_time)
        time_dim = self._normalized_time_dim(data)
        if data.shape[time_dim] != source_time.numel():
            raise ValueError(
                f"data time axis must match {names.source_time} length"
            )
        return time_dim

    @staticmethod
    def _argument_names(
        metadata: dict[str, Any],
        *,
        source_time: str,
        target_time: str,
        source_alias: str,
        target_alias: str,
    ) -> _MappingArgumentNames:
        """Move entry-point bound aliases onto the canonical keys.

        ``metadata`` is modified in place: ``source_alias`` becomes
        ``source_bounds`` and ``target_alias`` becomes ``target_bounds``.
        Supplying both an alias and its canonical key is rejected instead of
        letting one silently overwrite the other. The returned names are the
        keys the caller actually used (or ``alias (canonical)`` when absent).
        """
        bound_names: list[str] = []
        for alias, canonical in (
            (source_alias, "source_bounds"),
            (target_alias, "target_bounds"),
        ):
            if alias in metadata and canonical in metadata:
                raise ValueError(
                    f"received both {alias} and {canonical}; pass exactly one"
                )
            if alias in metadata:
                metadata[canonical] = metadata.pop(alias)
                bound_names.append(alias)
            elif canonical in metadata:
                bound_names.append(canonical)
            else:
                bound_names.append(f"{alias} ({canonical})")
        return _MappingArgumentNames(
            source_time=source_time,
            target_time=target_time,
            source_bounds=bound_names[0],
            target_bounds=bound_names[1],
        )

    @staticmethod
    def _validated_bounds(bounds: Any, coordinate: Tensor, name: str) -> Tensor:
        if not isinstance(bounds, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if bounds.layout != torch.strided or bounds.device.type == "meta":
            raise ValueError(f"{name} must be one materialized strided tensor")
        if bounds.dtype not in {torch.float32, torch.float64}:
            raise ValueError(f"{name} must use float32 or float64")
        if bounds.shape != (coordinate.numel(), 2):
            raise ValueError(f"{name} must have shape ({coordinate.numel()}, 2)")
        if not bool(torch.all(torch.isfinite(bounds)).item()):
            raise ValueError(f"{name} must contain finite values")
        return bounds

    def _build_unit_contract(
        self,
        semantics: TemporalSemantics,
        source_units: str,
        metadata: dict[str, Any],
    ) -> TemporalUnitContract:
        target_units = self._required_text(metadata, "target_units")
        measure_units = self._required_text(metadata, "measure_units")
        integration_time_unit = metadata.get("integration_time_unit")
        if semantics is TemporalSemantics.INTERVAL_ACCUMULATION:
            if integration_time_unit is not None:
                raise ValueError(
                    "interval_accumulation must not declare integration_time_unit"
                )
        elif not isinstance(integration_time_unit, str) or not integration_time_unit.strip():
            raise ValueError(
                "integration_time_unit is required for interval means and rates"
            )
        return TemporalUnitContract(
            source_units=source_units,
            target_units=target_units,
            measure_units=measure_units,
            integration_time_unit=(
                None
                if integration_time_unit is None
                else integration_time_unit.strip()
            ),
        )

    def _build_plan(
        self,
        source_time: Tensor,
        target_time: Tensor,
        source_bounds: Tensor,
        target_bounds: Tensor,
        *,
        calendar: str,
        time_units: str,
    ) -> TemporalOverlapPlan:
        source_coordinate = self.cf_time_processor.parse_cf_time(
            source_time, units=time_units, calendar=calendar
        )
        target_coordinate = self.cf_time_processor.parse_cf_time(
            target_time, units=time_units, calendar=calendar
        )
        source_coordinate.bounds = source_bounds.to(
            device=source_coordinate.values.device,
            dtype=source_coordinate.values.dtype,
        )
        target_coordinate.bounds = target_bounds.to(
            device=target_coordinate.values.device,
            dtype=target_coordinate.values.dtype,
        )
        return self.time_weight_generator.build_overlap_plan(
            source_coordinate, target_coordinate
        )

    @staticmethod
    def _law_id(semantics: TemporalSemantics) -> str:
        if semantics is TemporalSemantics.INTERVAL_MEAN:
            return "temporal_observed_integral_consistency"
        if semantics is TemporalSemantics.RATE:
            return "temporal_rate_integral_consistency"
        return "temporal_interval_accumulation_partition"

    def _conservation_diagnostics(
        self,
        *,
        plan: TemporalOverlapPlan,
        source_data: Tensor,
        target_data: Tensor,
        semantics: TemporalSemantics,
        unit_contract: TemporalUnitContract,
        accepted_target_mask: Tensor,
        metadata: dict[str, Any],
        variable_name: str,
        time_dim: int,
    ) -> tuple[dict[str, Any], float]:
        level = self.config.resolve_conservation_level(self.name)
        diagnostics: dict[str, Any] = {
            "conservation_level": level.value,
            "conservation_computed": False,
            "conservation_checked": False,
            "conservation_passed": None,
            "conservation_violation": False,
            "physical_ledger_checked": False,
        }
        if not level.compute_residual:
            return diagnostics, 0.0
        q_ref_parameter = metadata.get("q_ref_parameter")
        if not isinstance(q_ref_parameter, str) or not q_ref_parameter.strip():
            raise ValueError(
                "q_ref_parameter is required when temporal conservation is enabled"
            )
        q_ref_name = q_ref_parameter.strip()
        q_ref = self.get_parameter_value(
            q_ref_name,
            units=unit_contract.measure_units,
            validate=True,
            required=True,
        )
        law_id = self._law_id(semantics)
        # An output element (one non-time series) without any accepted target
        # (every mapped value NaN, e.g. a fully missing source series) has no
        # support on which the mapping can be compared. It keeps its NaN output
        # and is excluded explicitly from the residual.
        excluded_output_mask = ~torch.any(accepted_target_mask, dim=time_dim)
        residual = compute_temporal_mapping_residual(
            plan,
            source_data,
            target_data,
            semantics=semantics,
            unit_contract=unit_contract,
            law_id=law_id,
            accepted_target_mask=accepted_target_mask,
            excluded_output_mask=excluded_output_mask,
            time_dim=time_dim,
            q_ref=q_ref,
            q_ref_provenance=q_ref_name,
        )
        diagnostics.update(
            {
                "conservation_computed": True,
                "mapping_law_id": law_id,
                "mapping_residual": residual.residual,
                "mapping_relative_residual": residual.relative_residual,
                "mapping_source_activity": residual.source_activity,
                "mapping_accepted_target_mask": residual.accepted_target_mask,
                "mapping_excluded_target_count": int(
                    torch.sum(~accepted_target_mask).detach().cpu().item()
                ),
                "mapping_excluded_output_mask": excluded_output_mask,
                "mapping_excluded_output_count": int(
                    torch.sum(excluded_output_mask).detach().cpu().item()
                ),
                "q_ref_provenance": q_ref_name,
            }
        )
        if residual.relative_residual is None:
            raise RuntimeError("Relative temporal mapping residual was not computed")
        checked_outputs = residual.checked
        if not bool(torch.any(checked_outputs).detach().cpu().item()):
            diagnostics["conservation_skip_reason"] = "no_accepted_target_support"
            logger.info(
                "Temporal mapping residual skipped for %s: no output element "
                "has an accepted target",
                variable_name or law_id,
            )
            return diagnostics, math.nan
        maximum_relative = torch.max(residual.relative_residual[checked_outputs])
        conservation_error = float(maximum_relative.detach().cpu().item())
        if not level.compare_tolerance:
            return diagnostics, conservation_error
        # References: LeVeque, R.J. (1992), Numerical Methods for Conservation Laws;
        #   RELIC conservation tolerance SSOT (validation/conservation.yaml).
        # Mathematical expression: passed = max_j |R_j| / max(A_j, q_ref) <= tol, with
        #   tol = validation.conservation.conservation_tolerance.
        # Parameter meanings: R_j mapping residual and A_j source activity of output
        #   element j; q_ref the declared reference magnitude; tol the general relative
        #   conservation tolerance, the one key TemporalDisaggregator declares and reads.
        # Units: R_j, A_j and q_ref in the mapped measure unit; tol dimensionless.
        # Domain/constraints: tol within its YAML bounds; the conservation level decides
        #   whether the comparison runs and whether a violation blocks, it selects no
        #   tolerance; there is no per-variable, precision-mode or scale-coupling fallback.
        # Sign convention: the comparison uses |R_j|; R_j > 0 when the target measure
        #   exceeds the source measure.
        tolerance = float(
            self.get_parameter_value(
                "validation.conservation.conservation_tolerance", required=True
            )
        )
        passed = bool((maximum_relative <= tolerance).item())
        diagnostics.update(
            {
                "conservation_checked": True,
                "conservation_passed": passed,
                "conservation_violation": not passed,
                "conservation_tolerance": tolerance,
                "maximum_relative_residual": maximum_relative,
            }
        )
        if not passed and level.block_on_violation:
            raise ValueError(
                "temporal conservation violation: maximum relative residual "
                f"{conservation_error:.6e} exceeds tolerance {tolerance:.6e}"
            )
        return diagnostics, conservation_error

    def _state_mapping(
        self,
        data: Tensor,
        source_time: Tensor,
        target_time: Tensor,
        *,
        calendar: str,
        time_units: str,
        target_units: str,
        source_units: str,
        metadata: dict[str, Any],
        time_dim: int,
    ) -> DisaggregationResult:
        level = self.config.resolve_conservation_level(self.name)
        if level.compute_residual:
            raise ValueError(
                "state_at_boundary conservation is unavailable because no state "
                "mapping residual law has been implemented"
            )
        if target_units != source_units:
            raise ValueError(
                "state_at_boundary requires identical source and target units"
            )
        policy = metadata.get("state_boundary_policy")
        if policy not in {"exact", "left", "right"}:
            raise ValueError(
                "state_boundary_policy must be explicitly exact, left, or right"
            )
        source_coordinate = self.cf_time_processor.parse_cf_time(
            source_time, units=time_units, calendar=calendar
        )
        target_coordinate = self.cf_time_processor.parse_cf_time(
            target_time, units=time_units, calendar=calendar
        )
        if policy == "exact":
            mapped = map_state_boundaries(
                source_coordinate, target_coordinate, data, time_dim=time_dim
            )
        else:
            # References: CF Conventions 1.13, Sections 4.4 and 7.1.
            # Mathematical expression: i_left=max{i:t_i<=t}; i_right=min{i:t_i>=t}.
            # Parameter meanings: t_i are declared source boundary instants.
            # Units: source and target coordinates share the declared time_units.
            # Domain/constraints: no extrapolation; source boundaries are increasing.
            # Sign convention: state values are selected without summation or repair.
            source_values = source_coordinate.values
            target_values = target_coordinate.values
            indices = torch.searchsorted(
                source_values, target_values, right=policy == "left"
            )
            if policy == "left":
                indices = indices - 1
            if bool(torch.any(indices < 0).item()) or bool(
                torch.any(indices >= source_values.numel()).item()
            ):
                raise ValueError("state boundary policy does not allow extrapolation")
            snapped = source_values.index_select(0, indices)
            unique_snapped, inverse = torch.unique_consecutive(
                snapped, return_inverse=True
            )
            unique_target = TimeCoordinate(
                values=unique_snapped,
                units=source_coordinate.units,
                calendar=calendar,
            )
            unique_values = map_state_boundaries(
                source_coordinate, unique_target, data, time_dim=time_dim
            )
            mapped = unique_values.index_select(time_dim, inverse)
        diagnostics = {
            "aggregation_method": "state_boundary",
            "temporal_semantics": TemporalSemantics.STATE_AT_BOUNDARY.value,
            "state_boundary_policy": policy,
            "state_boundary_mapping_verified": True,
            "conservation_level": level.value,
            "conservation_computed": False,
            "conservation_checked": False,
            "conservation_passed": None,
            "conservation_violation": False,
            "physical_ledger_checked": False,
        }
        updated = self._updated_metadata(
            variable_name=metadata["variable_name"],
            standard_name=metadata["standard_name"],
            target_units=target_units,
            cell_methods=metadata["cell_methods"],
            semantics=TemporalSemantics.STATE_AT_BOUNDARY,
            calendar=calendar,
            time_units=time_units,
            public_metadata=metadata,
            reconstruction=f"state_{policy}",
        )
        return DisaggregationResult(mapped, updated, 0.0, diagnostics)

    def _map_temporal(
        self,
        *,
        data: Tensor,
        source_time: Tensor,
        target_time: Tensor,
        variable_name: Optional[str],
        standard_name: Optional[str],
        units: Optional[str],
        cell_methods: Optional[str],
        metadata: dict[str, Any],
        names: _MappingArgumentNames,
    ) -> DisaggregationResult:
        owner = "TemporalDisaggregator"
        units = require_text_argument(units, "units", owner)
        variable_name = optional_text_argument(variable_name, "variable_name", owner)
        standard_name = optional_text_argument(standard_name, "standard_name", owner)
        cell_methods = optional_text_argument(cell_methods, "cell_methods", owner)
        time_dim = self._validate_inputs(data, source_time, target_time, names)
        semantics = self._resolve_semantics(metadata)
        calendar = self._required_text(metadata, "calendar")
        time_units = self._required_text(metadata, "time_units")
        if calendar not in {"noleap", "360_day"}:
            raise ValueError("calendar must be explicitly noleap or 360_day")
        target_units = self._required_text(metadata, "target_units")
        complete_metadata = {
            **metadata,
            "variable_name": variable_name,
            "standard_name": standard_name,
            "cell_methods": cell_methods,
        }
        if semantics is TemporalSemantics.STATE_AT_BOUNDARY:
            return self._state_mapping(
                data,
                source_time,
                target_time,
                calendar=calendar,
                time_units=time_units,
                target_units=target_units,
                source_units=units,
                metadata=complete_metadata,
                time_dim=time_dim,
            )
        self._ensure_piecewise_constant_method()
        missing_bounds = [
            name
            for key, name in (
                ("source_bounds", names.source_bounds),
                ("target_bounds", names.target_bounds),
            )
            if key not in metadata
        ]
        if missing_bounds:
            raise ValueError(
                "explicit interval bounds are required; missing "
                + ", ".join(missing_bounds)
            )
        source_bounds = self._validated_bounds(
            metadata["source_bounds"], source_time, names.source_bounds
        )
        target_bounds = self._validated_bounds(
            metadata["target_bounds"], target_time, names.target_bounds
        )
        if (
            self.disaggregation_config.use_physical_constraints
            and not self.disaggregation_config.allow_negative
            and bool(
                torch.any(
                    torch.isfinite(data) & (data < self.conservative_clamp_min)
                ).item()
            )
        ):
            raise ValueError(
                "nonnegative physical constraint is violated; values are not clamped"
            )
        plan = self._build_plan(
            source_time,
            target_time,
            source_bounds,
            target_bounds,
            calendar=calendar,
            time_units=time_units,
        )
        unit_contract = self._build_unit_contract(semantics, units, metadata)
        kernel_result = apply_temporal_kernel(
            plan,
            data,
            semantics=semantics,
            output_semantics=semantics,
            unit_contract=unit_contract,
            valid_mask=metadata.get("valid_mask"),
            time_dim=time_dim,
        )
        accepted_target_mask = torch.isfinite(kernel_result.values) & (
            kernel_result.valid_coverage > 0
        )
        conservation, conservation_error = self._conservation_diagnostics(
            plan=plan,
            source_data=data,
            target_data=kernel_result.values,
            semantics=semantics,
            unit_contract=unit_contract,
            accepted_target_mask=accepted_target_mask,
            metadata=metadata,
            variable_name=variable_name,
            time_dim=time_dim,
        )
        diagnostics: dict[str, Any] = {
            "disaggregation_method": self.disaggregation_config.method.value,
            "reconstruction": "piecewise_constant",
            "temporal_semantics": semantics.value,
            "bounds_relation": plan.relation.value,
            "geometric_coverage": kernel_result.geometric_coverage,
            "valid_coverage": kernel_result.valid_coverage,
            "complete_coverage": kernel_result.complete_coverage,
            "gap_kind": kernel_result.gap_diagnostic.kind,
            "gap_total_fraction": kernel_result.gap_diagnostic.total_gap_fraction,
            "observed_measure": kernel_result.observed_measure,
            **conservation,
        }
        updated = self._updated_metadata(
            variable_name=variable_name,
            standard_name=standard_name,
            target_units=target_units,
            cell_methods=cell_methods,
            semantics=semantics,
            calendar=calendar,
            time_units=time_units,
            public_metadata=metadata,
            reconstruction="piecewise_constant",
        )
        return DisaggregationResult(
            kernel_result.values, updated, conservation_error, diagnostics
        )

    @staticmethod
    def _updated_metadata(
        *,
        variable_name: Optional[str],
        standard_name: Optional[str],
        target_units: str,
        cell_methods: Optional[str],
        semantics: TemporalSemantics,
        calendar: str,
        time_units: str,
        public_metadata: dict[str, Any],
        reconstruction: str,
    ) -> dict[str, Any]:
        excluded = {
            "source_bounds",
            "target_bounds",
            "coarse_bounds",
            "fine_bounds",
            "temporal_semantics",
            "target_units",
            "measure_units",
            "integration_time_unit",
            "q_ref_parameter",
            "valid_mask",
            "state_boundary_policy",
            "variable_name",
            "standard_name",
            "cell_methods",
        }
        metadata = {
            key: value for key, value in public_metadata.items() if key not in excluded
        }
        cf_method = (
            "sum"
            if semantics is TemporalSemantics.INTERVAL_ACCUMULATION
            else "point"
            if semantics is TemporalSemantics.STATE_AT_BOUNDARY
            else "mean"
        )
        # Absent identity entries are omitted (never published as ""), and an absent cell_methods
        # means no earlier cell method, so only this operation's method is recorded.
        metadata.update(
            {
                **identity_metadata(variable_name, standard_name),
                "units": target_units,
                "calendar": calendar,
                "time_units": time_units,
                "cell_methods": update_cell_methods(cell_methods or "", cf_method, "time"),
                "temporal_reconstruction": reconstruction,
                "history": f"disaggregation using {reconstruction}",
            }
        )
        return metadata

    def disaggregate(
        self,
        data: Tensor,
        coarse_time: Tensor,
        fine_time: Tensor,
        variable_name: Optional[str] = None,
        standard_name: Optional[str] = None,
        units: Optional[str] = None,
        cell_methods: Optional[str] = None,
        **metadata_kwargs: Any,
    ) -> DisaggregationResult:
        """Map coarse intervals to fine intervals through the canonical core.

        Bounds are read from ``coarse_bounds``/``fine_bounds`` or from the
        canonical ``source_bounds``/``target_bounds`` (one name per bound).
        The configuration must declare ``direction=DOWNSCALE``. ``units`` is
        required; ``variable_name``, ``standard_name`` and ``cell_methods`` are
        optional: absent (None) means an unnamed field, no CF identity and no
        earlier cell method, and the absent names are not published (an empty
        string raises).
        """
        self._require_direction(ScalingDirection.DOWNSCALE, "disaggregate")
        metadata = dict(metadata_kwargs)
        names = self._argument_names(
            metadata,
            source_time="coarse_time",
            target_time="fine_time",
            source_alias="coarse_bounds",
            target_alias="fine_bounds",
        )
        return self._map_temporal(
            data=data,
            source_time=coarse_time,
            target_time=fine_time,
            variable_name=variable_name,
            standard_name=standard_name,
            units=units,
            cell_methods=cell_methods,
            metadata=metadata,
            names=names,
        )

    def downscale(self, coarse_data: Tensor, **kwargs: Any) -> Tensor:
        """Delegate bounded downscaling to :meth:`disaggregate`."""
        coarse_time = kwargs.get("coarse_time")
        fine_time = kwargs.get("fine_time")
        if coarse_time is None or fine_time is None:
            raise ValueError("downscale requires explicit coarse_time and fine_time")
        forwarded = {
            key: value
            for key, value in kwargs.items()
            if key not in {"coarse_time", "fine_time"}
        }
        return self.disaggregate(
            coarse_data, coarse_time, fine_time, **forwarded
        ).data

    def upscale(self, fine_data: Tensor, **kwargs: Any) -> Tensor:
        """Map fine intervals to coarse intervals without another Functor.

        The configuration must declare ``direction=UPSCALE``.
        """
        self._require_direction(ScalingDirection.UPSCALE, "upscale")
        fine_time = kwargs.get("fine_time")
        coarse_time = kwargs.get("coarse_time")
        if fine_time is None or coarse_time is None:
            raise ValueError("upscale requires explicit fine_time and coarse_time")
        metadata = {
            key: value
            for key, value in kwargs.items()
            if key
            not in {
                "fine_time",
                "coarse_time",
                "units",
                "variable_name",
                "standard_name",
                "cell_methods",
            }
        }
        names = self._argument_names(
            metadata,
            source_time="fine_time",
            target_time="coarse_time",
            source_alias="fine_bounds",
            target_alias="coarse_bounds",
        )
        return self._map_temporal(
            data=fine_data,
            source_time=fine_time,
            target_time=coarse_time,
            variable_name=kwargs.get("variable_name"),
            standard_name=kwargs.get("standard_name"),
            units=kwargs.get("units"),
            cell_methods=kwargs.get("cell_methods"),
            metadata=metadata,
            names=names,
        ).data

    def downscale_simple(self, coarse_data: Tensor) -> Tensor:
        """Retain the ratio-only signature while failing closed."""
        del coarse_data
        raise ValueError(
            "ratio-only downscale_simple is unbound without explicit interval bounds"
        )

    def forward(
        self,
        state: dict[str, Any],
        time: float,
        dt: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute disaggregation through the Functor lifecycle."""
        # References: CF Conventions 1.13, Sections 4.4, 7.1, 7.3, Appendix E; Jones (1999) (module
        #   docstring); RELIC ScaleCoupling contract (relic/core/functor/scale_coupling_bridge.py).
        # Mathematical expression: via self.disaggregate -> _map_temporal with the piecewise-constant
        #   kernel on O_ij = |I_i intersect J_j| (coarse i, fine j): interval_mean and rate
        #   x_j = sum_i v_i O_ij y_i / sum_i v_i O_ij; interval_accumulation x_j = sum_i (O_ij/|I_i|) a_i;
        #   state_at_boundary selects the exact, left, or right coarse instant. R = sum_j q_j - sum_i q_i.
        # Parameter meanings: y_i, a_i coarse values; x_j fine values; v_i finite mask; I_i, J_j
        #   intervals from coarse_bounds/fine_bounds; output keyed by the required variable_name
        #   plus "_temporal_metadata", "_temporal_diagnostics", "_conservation_error".
        # Units: means and rates map source units -> target_units; accumulations go through
        #   measure_units -> target_units via the TemporalUnitContract; bounds use time_units.
        # Domain/constraints: data, coarse_time, fine_time, a non-empty variable_name and units,
        #   temporal_semantics, calendar in {noleap, 360_day}, time_units, target_units, and bounds
        #   are required; standard_name and cell_methods are optional (absent: no CF identity, no
        #   earlier cell method; an empty string raises); only method "uniform" runs
        #   (others fail closed); point_sample/categorical rejected; values below conservative_clamp_min
        #   raise rather than clamp when negatives are forbidden; residual checked against tolerance
        #   over accepted fine intervals; an output series without accepted intervals stays NaN and
        #   is excluded explicitly (conservation_error NaN when no series is checked).
        # Sign convention: signed data preserved; O_ij >= 0; positive R means fine total exceeds coarse.
        del time, dt
        data = kwargs.get("data", state.get("data"))
        coarse_time = kwargs.get("coarse_time", state.get("coarse_time"))
        fine_time = kwargs.get("fine_time", state.get("fine_time"))
        if data is None or coarse_time is None or fine_time is None:
            raise ValueError(
                "TemporalDisaggregator.forward requires data, coarse_time, and fine_time"
            )
        def _argument(name: str) -> Any:
            # A keyword argument, else the state entry, else absent (None).
            if name in kwargs:
                return kwargs[name]
            return state[name] if name in state else None

        owner = "TemporalDisaggregator.forward"
        variable_name = require_text_argument(_argument("variable_name"), "variable_name", owner)
        units = require_text_argument(_argument("units"), "units", owner)
        standard_name = _argument("standard_name")
        cell_methods = _argument("cell_methods")
        metadata = {
            key: value
            for key, value in kwargs.items()
            if key
            not in {
                "data",
                "coarse_time",
                "fine_time",
                "variable_name",
                "standard_name",
                "units",
                "cell_methods",
            }
        }
        result = self.disaggregate(
            data,
            coarse_time,
            fine_time,
            variable_name=variable_name,
            standard_name=standard_name,
            units=units,
            cell_methods=cell_methods,
            **metadata,
        )
        return {
            variable_name: result.data,
            "_temporal_metadata": result.metadata,
            "_temporal_diagnostics": result.diagnostics,
            "_conservation_error": torch.tensor(
                result.conservation_error,
                dtype=result.data.dtype,
                device=result.data.device,
            ),
        }

    def update_metadata(
        self,
        metadata: dict[str, Any],
        operation: str = "disaggregation",
        method: str = "uniform",
    ) -> dict[str, Any]:
        """Publish a controlled CF cell-method token and reconstruction history."""
        del operation
        if method != DisaggregationMethod.UNIFORM.value:
            raise ValueError("canonical reconstruction is unavailable for metadata")
        semantics = self._resolve_semantics(metadata)
        cf_method = (
            "sum"
            if semantics is TemporalSemantics.INTERVAL_ACCUMULATION
            else "point"
            if semantics is TemporalSemantics.STATE_AT_BOUNDARY
            else "mean"
        )
        updated = metadata.copy()
        updated["cell_methods"] = update_cell_methods(
            str(updated.get("cell_methods") or ""), cf_method, "time"
        )
        updated["history"] = "disaggregation using piecewise_constant"
        updated["temporal_reconstruction"] = "piecewise_constant"
        return updated

    @staticmethod
    def _require_convenience_contract(meta: dict[str, Any]) -> None:
        required = ("coarse_bounds", "coarse_time", "fine_bounds", "fine_time")
        missing = [key for key in required if key not in meta]
        if missing:
            raise ValueError(
                "convenience disaggregation requires explicit coordinates and "
                "bounds; missing: " + ", ".join(missing)
            )

    def to_hourly(self, daily_data: Tensor, **meta: Any) -> DisaggregationResult:
        """Delegate daily-to-hourly mapping using caller-provided calendar bounds."""
        self._require_convenience_contract(meta)
        coarse_time = meta.pop("coarse_time")
        fine_time = meta.pop("fine_time")
        return self.disaggregate(daily_data, coarse_time, fine_time, **meta)

    def to_daily(self, monthly_data: Tensor, **meta: Any) -> DisaggregationResult:
        """Delegate monthly-to-daily mapping using caller-provided calendar bounds."""
        self._require_convenience_contract(meta)
        coarse_time = meta.pop("coarse_time")
        fine_time = meta.pop("fine_time")
        return self.disaggregate(monthly_data, coarse_time, fine_time, **meta)


__all__ = [
    "TemporalDisaggregator",
    "DisaggregationMethod",
    "DisaggregationConfig",
    "DisaggregationResult",
]
