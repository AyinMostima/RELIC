"""
Unified Temporal Aggregation Module

This module implements temporal upscaling (aggregation) for the RELIC framework.
It emits a bounded CF-1.13 cell_methods subset; dataset-level CF conformance,
coordinate bounds closure, and sampling-interval provenance are not claimed here.

Key Features:
- Controlled CF-1.13 cell-method metadata for supported aggregation methods
- Fail-closed metadata publication for variance and percentile
- Explicit temporal semantics without name or unit based method selection
- Per-output valid-duration coverage and quality flags
- Explicit noleap, 360_day, Gregorian and proleptic Gregorian interval execution
- Vectorized PyTorch operations for performance
- Fail-closed unbound conservation modes
- Comprehensive error handling and logging

Mathematical Framework:
    Target-fraction weights:
    w_{ij} = overlap(t_i, T_j) / duration(T_j)

    Missing-aware time-weighted mean:
    Mean_bj = sum_i(w_ij m_bi x_bi) / sum_i(w_ij m_bi)

    Coverage validation (valid coverage, m_bi = finite mask):
    C_bj = sum_i(w_ij m_bi) >= coupling.scales.temporal.coverage_threshold,
    otherwise the output is NaN and, when no target of an output element is
    accepted, that element is excluded explicitly from the mapping residual.

Classes:
    TemporalAggregator: Main aggregation class with full functionality
    AggregationConfig: Configuration dataclass
    AggregationResult: Result dataclass with metadata

References:
    - CF Conventions Version 1.13, Section 7.3 and Appendix E
    - IPCC Guidelines for Climate Data Processing
    - WMO Guide to Climatological Practices
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import torch
from cf_units import Unit
from torch import Tensor

from ....core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    ScaleCouplingFunctor,
    SpatialScale,
    VariableKind,
)
from ....data.standards.dimensionality import (
    Dimensionality,
    dimensionality_from_unit,
)
from ....parameters.namespaces import ParameterNamespace, resolve_namespace
from ... import DEFAULTS_NAMESPACE, DEFAULTS_NAMESPACE_STR
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

# Set up logging
logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """Enumeration of supported aggregation methods."""
    MEAN = "mean"                    # Arithmetic mean
    TIME_WEIGHTED_MEAN = "time_weighted_mean"  # Time-weighted mean (default for intensive)
    SUM = "sum"                      # Canonical rate or interval-amount integration
    MINIMUM = "min"                  # Minimum value
    MAXIMUM = "max"                  # Maximum value
    MEDIAN = "median"                # Median value
    VARIANCE = "variance"            # Population variance with squared units
    STANDARD_DEVIATION = "std"       # Standard deviation
    PERCENTILE = "percentile"        # [N] no general CF percentile token
    RANGE = "range"                  # Range (max - min)


# References: CF Conventions 1.13, Section 7.3 and Appendix E.
# Mathematical expression: cell_methods = "time: <controlled_method>".
# Parameter meanings: keys are RELIC aggregation methods; values are CF tokens.
# Units: mean, sum, extrema, median, standard deviation, and range retain u.
# Domain/constraints: variance requires u^2 and percentile has no general CF token.
# Sign convention: not applicable to descriptive temporal statistics.
_CF_CELL_METHOD_BY_AGGREGATION_METHOD: Dict[AggregationMethod, str] = {
    AggregationMethod.MEAN: "mean",
    AggregationMethod.TIME_WEIGHTED_MEAN: "mean",
    AggregationMethod.SUM: "sum",
    AggregationMethod.MINIMUM: "minimum",
    AggregationMethod.MAXIMUM: "maximum",
    AggregationMethod.MEDIAN: "median",
    AggregationMethod.VARIANCE: "variance",
    AggregationMethod.STANDARD_DEVIATION: "standard_deviation",
    AggregationMethod.RANGE: "range",
}


@dataclass
class AggregationConfig:
    """
    Configuration for temporal aggregation operations.
    
    Attributes:
        method : AggregationMethod
            Primary aggregation method
        time_dim : int
            Time dimension index in input tensor

    Interval labels and closure are not configurable: every interval is the
    explicit half-open bound pair [start, end) supplied by the caller, and the
    target coordinate is the caller's ``target_time``. Percentile aggregation
    is rejected, so no percentile parameter exists.
    """
    method: AggregationMethod = AggregationMethod.TIME_WEIGHTED_MEAN
    time_dim: int = 0


@dataclass
class QualityFlags:
    """Quality flags for aggregation results."""
    valid_fraction: Tensor
    num_samples: Tensor
    missing_fraction: Tensor
    insufficient_coverage_mask: Tensor
    conservation_error: Optional[float] = None
    range_preservation: Optional[float] = None


@dataclass
class AggregationResult:
    """
    Result of temporal aggregation with metadata.
    
    Attributes:
        data : Tensor
            Aggregated data tensor
        metadata : Dict[str, Any]
            Updated metadata including units, cell_methods, history
        quality_flags : QualityFlags
            Quality assessment flags
        diagnostics : Dict[str, float]
            Conservation and performance diagnostics
    """
    data: Tensor
    metadata: Dict[str, Any]
    quality_flags: QualityFlags
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class TemporalAggregator(ScaleCouplingFunctor):
    """
    Unified temporal aggregation with bounded CF cell-method metadata.
    
    Implements fixed-interval temporal statistics from explicit coordinates and
    bounds. It does not infer interval totals, calendar periods, variable roles,
    conservation semantics, or dataset-level CF conformance.
    
    Parameters
    ----------
    config : TemporalConfig
        Base temporal scaling configuration (frequencies, semantics, conservation level);
        ``direction`` must be ``UPSCALE`` and a categorical ``spatial_basis`` is rejected.
    aggregation_config : Optional[AggregationConfig]
        Aggregation-specific settings (method, time_dim), dimensionless selectors.
    parameter_manager : Optional[Any]
        Parameter manager instance; the coverage threshold is read only from
        coupling.scales.temporal.coverage_threshold (dimensionless).
            
    Mathematical Framework:
        w_ij = |I_i intersect J_j| / |J_j|
        y_bj = sum_i(w_ij m_bi x_bi) / sum_i(w_ij m_bi)
        c_bj = sum_i(w_ij m_bi)

        I_i and J_j are explicit source and target intervals. m_bi is one for
        finite data and zero for NaN. Inf is invalid rather than missing.

    Returns
    -------
    AggregationResult
        Contains aggregated data, metadata (units/cell_methods), quality flags, diagnostics.

    References
    ----------
    CF Conventions Version 1.13, Sections 4.4, 7.1, 7.3, and Appendix E.
    PyTorch Autograd Mechanics: graphs are rebuilt for each forward evaluation.
    """
    
    def __init__(
        self,
        config: TemporalConfig,
        aggregation_config: Optional[AggregationConfig] = None,
        parameter_manager: Optional[Any] = None,
        metadata: Optional[FunctorMetadata] = None,
    ):
        """
        Initialize temporal aggregator.

        Parameters:
            config : TemporalConfig
                Base temporal scaling configuration
            aggregation_config : Optional[AggregationConfig]
                Aggregation-specific configuration
            parameter_manager : Optional[Any]
                Parameter manager instance
            metadata : Optional[FunctorMetadata]
                Optional metadata override
        """
        if config.direction is not ScalingDirection.UPSCALE:
            raise ValueError(
                "TemporalAggregator maps source intervals onto target intervals "
                "(fine to coarse); TemporalConfig.direction must be 'upscale', "
                f"received {config.direction.value!r}. Use TemporalDisaggregator "
                "for a 'downscale' configuration."
            )
        self.temporal_config = config

        resolved_metadata = metadata or FunctorMetadata(
            name="temporal_aggregator",
            version="1.0.0",
            description="Temporal aggregation functor with bounded CF-1.13 cell-method metadata",
            category=ProcessCategory.SCALE_COUPLING,
            execution_phase=ExecutionPhase.COUPLING,
            mathematical_expression="U[phi](T_j)=sum_i w_ij phi(t_i); coverage_j=sum_i w_ij",
            references=[
                "CF Conventions Version 1.13, Sections 4.4, 7.1, 7.3, and Appendix E",
                "PyTorch Autograd Mechanics: How autograd encodes the history",
            ],
            parameters={
                "coverage_threshold": "coupling.scales.temporal.coverage_threshold (dimensionless)",
                "coverage_epsilon": f"{DEFAULTS_NAMESPACE_STR}.coverage_epsilon (dimensionless)",
                "validation.conservation.conservation_tolerance": (
                    "Relative tolerance of the temporal mapping amount ledger [dimensionless]; "
                    "the conservation level decides whether it is compared and whether a "
                    "violation blocks"
                ),
            },
            input_variables=["data", "source_time", "target_time"],
            output_variables=["temporal_aggregated"],
            conservation_laws=[],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.DIAGNOSTIC,
        )

        config_dict = {
            "temporal_config": {
                "source_frequency": config.source_frequency,
                "target_frequency": config.target_frequency,
                "direction": config.direction.value if isinstance(config.direction, Enum) else config.direction,
                "spatial_basis": config.spatial_basis.value,
                "conservation_level": config.conservation_level.value if isinstance(config.conservation_level, Enum) else config.conservation_level,
                "cf_validation_scope": "time_coordinate_and_cell_method_subset",
                "dataset_conformance_checked": False,
                "device": str(config.device) if config.device is not None else None,
            }
        }

        super().__init__(
            name="temporal_aggregator",
            config=config_dict,
            metadata=resolved_metadata,
            source_scale=config.source_frequency,
            target_scale=config.target_frequency,
            coupling_type="temporal",
            device=config.device,
            parameter_manager=parameter_manager,
        )
        self.parameter_manager = self.get_parameter_manager()

        self.aggregation_config = aggregation_config or AggregationConfig()
        # Read-only after construction (the coverage_threshold property).
        self._coverage_threshold = self._resolve_coverage_threshold()
        coverage_eps = self.get_parameter_value(
            "coverage_epsilon",
            namespace=DEFAULTS_NAMESPACE,
            units="dimensionless",
            validate=True,
        )
        if isinstance(coverage_eps, torch.Tensor):
            coverage_eps = float(coverage_eps.detach().cpu().item())
        self.coverage_minimum = coverage_eps

        self.cf_time_processor = CFTimeProcessor(
            device=self.device,
            parameter_manager=self.parameter_manager,
        )
        self.time_weight_generator = TimeWeightGenerator(
            device=self.device,
            parameter_manager=self.parameter_manager,
        )
        logger.info(f"Initialized TemporalAggregator with method={self.aggregation_config.method.value}")

    def to_device(
        self,
        device: str | torch.device,
    ) -> "TemporalAggregator":
        """Move the functor and its directly owned temporal helpers together."""
        target_device = _materialize_device(torch.device(device))
        super().to_device(target_device)
        self.cf_time_processor.device = target_device
        self.time_weight_generator.device = target_device
        self.time_weight_generator.cf_processor.device = target_device
        return self
    
    @property
    def coverage_threshold(self) -> float:
        """Minimum accepted coverage c_bj of a target interval [dimensionless], read-only.

        Resolved once at construction from its sole owner, the parameter
        coupling.scales.temporal.coverage_threshold, and checked to lie in
        (0, 1]. A different value for one aggregator is set on that parameter
        (the global ParameterManager or an injected parameter_manager) before
        construction, where it is validated; assigning this attribute raises
        AttributeError instead of silently bypassing the parameter.
        """
        return self._coverage_threshold

    def _resolve_coverage_threshold(self) -> float:
        """Resolve the coverage threshold from its sole ParameterManager owner."""
        namespace = resolve_namespace(ParameterNamespace.COUPLING_SCALES_TEMPORAL)
        if namespace is None:
            raise RuntimeError("Temporal parameter namespace could not be resolved")
        value = self.get_parameter_value(
            "coverage_threshold",
            namespace=namespace,
            units="dimensionless",
            validate=True,
            expected_type=float,
            required=True,
        )
        if isinstance(value, torch.Tensor):
            if value.numel() != 1 or not torch.all(torch.isfinite(value)):
                raise ValueError(
                    "coverage_threshold must be one finite scalar"
                )
            value = value.detach().reshape(()).item()
        if isinstance(value, bool):
            raise TypeError("coverage_threshold must be a real scalar")
        threshold = float(value)
        if not math.isfinite(threshold):
            raise ValueError("coverage_threshold must be finite")
        if threshold <= 0.0 or threshold > 1.0:
            raise ValueError(
                f"coverage_threshold must be in (0, 1]; received {threshold}"
            )
        return threshold

    def _resolve_temporal_semantics(
        self,
        declared_semantics: TemporalSemantics | str | None,
    ) -> TemporalSemantics:
        """Resolve an explicit temporal law without using identity metadata."""
        value = (
            declared_semantics
            if declared_semantics is not None
            else self.temporal_config.temporal_semantics
        )
        if value is None:
            raise ValueError("temporal_semantics is required for temporal aggregation")
        if isinstance(value, TemporalSemantics):
            semantics = value
        else:
            normalized = str(value).strip().lower()
            if normalized == "interval_total":
                raise ValueError(
                    "interval_total temporal semantics are unsupported; use the "
                    "canonical interval_accumulation contract"
                )
            try:
                semantics = TemporalSemantics(normalized)
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported temporal_semantics: {value!r}"
                ) from exc
        if (
            semantics is TemporalSemantics.CATEGORICAL
            or self.temporal_config.spatial_basis is SpatialQuantityBasis.CATEGORICAL
        ):
            raise ValueError("categorical temporal aggregation is [N]")
        if semantics is TemporalSemantics.POINT_SAMPLE:
            raise ValueError(
                "point_sample temporal aggregation requires an explicit quadrature "
                "or reconstruction contract"
            )
        return semantics

    @staticmethod
    def _required_metadata_text(metadata: Dict[str, Any], name: str) -> str:
        value = metadata.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be one explicit non-empty string")
        return value.strip()

    def _build_unit_contract(
        self,
        semantics: TemporalSemantics,
        source_units: str,
        metadata: Dict[str, Any],
    ) -> TemporalUnitContract:
        target_units = self._required_metadata_text(metadata, "target_units")
        measure_units = self._required_metadata_text(metadata, "measure_units")
        integration_time_unit = metadata.get("integration_time_unit")
        if semantics is TemporalSemantics.INTERVAL_ACCUMULATION:
            if integration_time_unit is not None:
                raise ValueError(
                    "interval_accumulation must not declare integration_time_unit"
                )
        elif not isinstance(integration_time_unit, str) or not (
            integration_time_unit.strip()
        ):
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

    @staticmethod
    def _resolve_output_semantics(
        semantics: TemporalSemantics,
        method: AggregationMethod,
    ) -> TemporalSemantics | None:
        if method in {
            AggregationMethod.MEAN,
            AggregationMethod.TIME_WEIGHTED_MEAN,
        }:
            if semantics in {
                TemporalSemantics.INTERVAL_MEAN,
                TemporalSemantics.RATE,
            }:
                return semantics
            raise ValueError(
                f"{method.value} is incompatible with {semantics.value}"
            )
        if method is AggregationMethod.SUM:
            if semantics in {
                TemporalSemantics.RATE,
                TemporalSemantics.INTERVAL_ACCUMULATION,
            }:
                return TemporalSemantics.INTERVAL_ACCUMULATION
            raise ValueError(f"sum is incompatible with {semantics.value}")
        if semantics is not TemporalSemantics.INTERVAL_MEAN:
            raise ValueError(
                f"Descriptive statistic {method.value} requires interval_mean semantics"
            )
        return None

    @staticmethod
    def _count_valid_samples(
        data: Tensor,
        plan: TemporalOverlapPlan,
        time_dim: int,
    ) -> Tensor:
        moved = data.movedim(time_dim, 0)
        non_time_shape = tuple(moved.shape[1:])
        source_size, target_size = plan.overlap.shape
        flat_valid = torch.isfinite(moved).reshape(source_size, -1)
        overlap_membership = plan.overlap > 0
        counts = torch.sum(
            overlap_membership[:, :, None] & flat_valid[:, None, :],
            dim=0,
        )
        restored = counts.reshape((target_size, *non_time_shape))
        return restored.movedim(0, time_dim)

    @staticmethod
    def _mapping_law_id(
        source_semantics: TemporalSemantics,
        target_semantics: TemporalSemantics,
    ) -> str:
        if source_semantics is TemporalSemantics.INTERVAL_MEAN:
            return "temporal_observed_integral_consistency"
        if source_semantics is TemporalSemantics.RATE:
            return (
                "temporal_rate_amount_consistency"
                if target_semantics is TemporalSemantics.INTERVAL_ACCUMULATION
                else "temporal_rate_integral_consistency"
            )
        return "temporal_interval_accumulation_partition"

    def _evaluate_mapping_conservation(
        self,
        *,
        plan: TemporalOverlapPlan,
        source_data: Tensor,
        target_data: Tensor,
        source_semantics: TemporalSemantics,
        target_semantics: TemporalSemantics,
        unit_contract: TemporalUnitContract,
        accepted_target_mask: Tensor,
        variable_name: str,
        metadata: Dict[str, Any],
        time_dim: int,
    ) -> Dict[str, Any]:
        level = self.temporal_config.resolve_conservation_level(self.name)
        diagnostics: Dict[str, Any] = {
            "conservation_level": level.value,
            "conservation_computed": False,
            "conservation_checked": False,
            "conservation_passed": None,
            "conservation_violation": False,
            "physical_ledger_checked": False,
        }
        if not level.compute_residual:
            return diagnostics

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
        law_id = self._mapping_law_id(source_semantics, target_semantics)
        # An output element (one non-time series) whose every target is
        # rejected (coverage below threshold, non-finite, or incomplete for an
        # accumulation) has no support on which the mapping can be compared.
        # It keeps its NaN output and is excluded explicitly from the residual.
        excluded_output_mask = ~torch.any(accepted_target_mask, dim=time_dim)
        residual = compute_temporal_mapping_residual(
            plan,
            source_data,
            target_data,
            semantics=source_semantics,
            output_semantics=target_semantics,
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
        checked_outputs = residual.checked
        if not bool(torch.any(checked_outputs).detach().cpu().item()):
            diagnostics["conservation_skip_reason"] = "no_accepted_target_support"
            logger.info(
                "Temporal mapping residual skipped for %s: no output element "
                "has an accepted target",
                variable_name or law_id,
            )
            return diagnostics
        if not level.compare_tolerance:
            return diagnostics

        if residual.relative_residual is None:
            raise RuntimeError("Relative mapping residual was not computed")
        # References: LeVeque, R.J. (1992), Numerical Methods for Conservation Laws;
        #   RELIC conservation tolerance SSOT (validation/conservation.yaml).
        # Mathematical expression: passed = max_j |R_j| / max(A_j, q_ref) <= tol, with
        #   tol = validation.conservation.conservation_tolerance.
        # Parameter meanings: R_j mapping residual and A_j source activity of output
        #   element j; q_ref the declared reference magnitude; tol the general relative
        #   conservation tolerance, the one key TemporalAggregator declares and reads.
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
        maximum_relative = torch.max(residual.relative_residual[checked_outputs])
        passed = bool(maximum_relative <= tolerance)
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
                f"{float(maximum_relative.detach().cpu().item()):.6e} exceeds "
                f"tolerance {tolerance:.6e}"
            )
        return diagnostics

    def _aggregate_state_boundary(
        self,
        *,
        data: Tensor,
        source_time: Tensor,
        target_time: Tensor,
        units: str,
        cell_methods: Optional[str],
        variable_name: Optional[str],
        standard_name: Optional[str],
        calendar: str,
        time_units: str,
        metadata: Dict[str, Any],
        time_dim: int,
    ) -> AggregationResult:
        level = self.temporal_config.resolve_conservation_level(self.name)
        if level.compute_residual:
            raise ValueError(
                "state_at_boundary conservation is unavailable because no "
                "state mapping residual law has been implemented"
            )
        target_units = self._required_metadata_text(metadata, "target_units")
        if target_units != units:
            raise ValueError(
                "state_at_boundary requires identical source and target value units"
            )
        source_coord = TimeCoordinate(
            values=source_time,
            units=time_units,
            calendar=calendar,
        )
        target_coord = TimeCoordinate(
            values=target_time,
            units=time_units,
            calendar=calendar,
        )
        mapped = map_state_boundaries(
            source_coord,
            target_coord,
            data,
            time_dim=time_dim,
        )
        valid_fraction = torch.ones_like(mapped, dtype=mapped.dtype)
        num_samples = torch.ones_like(mapped, dtype=torch.int64)
        quality_flags = self._create_quality_flags(
            valid_fraction,
            num_samples,
            torch.ones_like(mapped, dtype=torch.bool),
        )
        public_metadata = {
            key: value
            for key, value in metadata.items()
            if key
            not in {
                "temporal_semantics",
                "target_units",
                "measure_units",
                "integration_time_unit",
                "q_ref_parameter",
            }
        }
        updated_metadata = {
            **identity_metadata(variable_name, standard_name),
            "units": target_units,
            "calendar": calendar,
            "time_units": time_units,
            **public_metadata,
        }
        # An absent cell_methods means no earlier cell method: only this operation's is recorded.
        updated_metadata["cell_methods"] = update_cell_methods(
            cell_methods or "",
            "point",
            "time",
        )
        updated_metadata["history"] = "aggregation using exact state boundary mapping"
        diagnostics = {
            "aggregation_method": "state_boundary",
            "temporal_semantics": TemporalSemantics.STATE_AT_BOUNDARY.value,
            "state_boundary_mapping_verified": True,
            "conservation_level": level.value,
            "conservation_computed": False,
            "conservation_checked": False,
            "conservation_passed": None,
            "conservation_violation": False,
            "physical_ledger_checked": False,
        }
        return AggregationResult(
            data=mapped,
            metadata=updated_metadata,
            quality_flags=quality_flags,
            diagnostics=diagnostics,
        )

    def upscale(self, fine_data: Tensor, **kwargs) -> Tensor:
        """Delegate temporal upscaling to the single canonical aggregate path."""
        source_time = kwargs.get("source_time")
        target_time = kwargs.get("target_time")
        if source_time is None or target_time is None:
            raise ValueError(
                "TemporalAggregator.upscale requires source_time and target_time"
            )
        result = self.aggregate(
            data=fine_data,
            source_time=source_time,
            target_time=target_time,
            variable_name=kwargs.get("variable_name"),
            standard_name=kwargs.get("standard_name"),
            units=kwargs.get("units"),
            cell_methods=kwargs.get("cell_methods"),
            **{
                key: value
                for key, value in kwargs.items()
                if key
                not in {
                    "source_time",
                    "target_time",
                    "variable_name",
                    "standard_name",
                    "units",
                    "cell_methods",
                }
            },
        )
        return result.data

    
    def forward(self, state: Dict, time: float, dt: float, **kwargs) -> Dict:
        """
        Forward computation following Functor interface.
        
        Args:
            state: Current state variables
            time: Current simulation time [s]
            dt: Time step [s]
            **kwargs: Additional parameters
            
        Returns:
            Dict of computed fluxes/updates
        """
        # References: CF Conventions 1.13, Sections 4.4, 7.3, and Appendix E as cited in the module
        #   docstring; RELIC ScaleCoupling contract (relic/core/functor/scale_coupling_bridge.py).
        # Mathematical expression: via self.aggregate with O_ij = |I_i intersect J_j|. mean and
        #   time_weighted_mean (interval_mean, rate): y_j = sum_i v_i O_ij x_i / sum_i v_i O_ij;
        #   sum (rate): y_j = sum_i v_i O_ij r_i; sum (interval_accumulation): y_j = sum_i (O_ij/|I_i|) a_i;
        #   min/max/median/variance/std/range: w_ij = O_ij / |J_j| over finite samples;
        #   state_at_boundary: exact boundary instants, no arithmetic.
        # Parameter meanings: x_i, r_i, a_i source values; v_i finite mask; I_i, J_j source and target
        #   intervals from the explicit bounds; output keyed by the required variable_name (a
        #   standard_name never stands in for it) plus "_temporal_metadata",
        #   "_temporal_quality_flags", and "_temporal_diagnostics".
        # Units: means and descriptive statistics keep the declared units (target_units == units);
        #   variance reports variance_units = units^2; sum converts via the TemporalUnitContract.
        # Domain/constraints: requires data, source_time, target_time, a non-empty variable_name and
        #   units, calendar in {noleap, 360_day, gregorian, proleptic_gregorian},
        #   time_units, temporal_semantics, target_units, and
        #   explicit bounds; standard_name and cell_methods are optional (absent: no CF identity, no
        #   earlier cell method; an empty string raises);
        #   percentile is rejected; coverage < coverage_threshold -> NaN; residual vs. tolerance when enabled,
        #   over accepted targets only; an output element without accepted targets stays NaN and is
        #   excluded explicitly from the residual (reported, never counted as a zero residual).
        # Sign convention: signed data preserved; O_ij >= 0; residual = target minus source measure.
        data = kwargs.get("data")
        source_time = kwargs.get("source_time")
        target_time = kwargs.get("target_time")

        if data is None or source_time is None or target_time is None:
            raise ValueError(
                "TemporalAggregator.forward requires data, source_time, and target_time tensors provided via kwargs."
            )

        # The update is keyed by the variable name and the value units are required; the CF
        # identity and the earlier cell methods are optional (validated by aggregate).
        target_var = require_text_argument(
            kwargs.get("variable_name"), "variable_name", "TemporalAggregator.forward"
        )
        units = require_text_argument(kwargs.get("units"), "units", "TemporalAggregator.forward")
        variable_name = target_var
        standard_name = kwargs.get("standard_name")
        cell_methods = kwargs.get("cell_methods")

        result = self.aggregate(
            data=data,
            source_time=source_time,
            target_time=target_time,
            variable_name=variable_name,
            standard_name=standard_name,
            units=units,
            cell_methods=cell_methods,
            **{k: v for k, v in kwargs.items() if k not in {"data", "source_time", "target_time",
                                                            "variable_name", "standard_name", "units", "cell_methods"}}
        )

        return {
            target_var: result.data,
            "_temporal_metadata": result.metadata,
            "_temporal_quality_flags": result.quality_flags,
            "_temporal_diagnostics": result.diagnostics,
        }

    def aggregate(
        self,
        data: Tensor,
        source_time: Tensor,
        target_time: Tensor,
        variable_name: Optional[str] = None,
        standard_name: Optional[str] = None,
        units: Optional[str] = None,
        cell_methods: Optional[str] = None,
        **metadata_kwargs
    ) -> AggregationResult:
        """
        Perform temporal aggregation with bounded CF cell-method metadata.
        
        Parameters:
            data : Tensor
                Input data tensor with time as first or specified dimension
            source_time : Tensor
                Source time coordinates
            target_time : Tensor
                Target time coordinates
            variable_name : Optional[str]
                Variable name for metadata; absent (None) is an unnamed field and the result
                metadata then carry no variable_name
            standard_name : Optional[str]
                CF identity metadata; it never selects temporal semantics; absent (None) is a
                field without a CF identity and the result metadata then carry no standard_name
            units : str
                Variable units (required, non-empty)
            cell_methods : Optional[str]
                Current cell_methods attribute; absent (None) means no earlier cell method, so
                only this operation's method is recorded. An empty string is not accepted for
                any of the optional names (omit them instead).
            **metadata_kwargs : Any
                Additional metadata
                
        Returns:
            AggregationResult: Complete aggregation result with metadata
            
        Raises:
            ValidationError: If input validation fails
            RuntimeError: If aggregation operation fails
        """
        method = self.aggregation_config.method
        owner = "TemporalAggregator.aggregate"
        units = require_text_argument(units, "units", owner)
        variable_name = optional_text_argument(variable_name, "variable_name", owner)
        standard_name = optional_text_argument(standard_name, "standard_name", owner)
        cell_methods = optional_text_argument(cell_methods, "cell_methods", owner)

        # Fail closed before bounds processing or any metadata construction.
        self._resolve_cf_cell_method(method, metadata_kwargs)
        temporal_semantics = self._resolve_temporal_semantics(
            metadata_kwargs.get("temporal_semantics")
        )

        self._validate_fixed_interval_frequency(
            self.temporal_config.source_frequency,
            "source_frequency",
        )
        self._validate_fixed_interval_frequency(
            self.temporal_config.target_frequency,
            "target_frequency",
        )
        time_dim = self._validate_inputs(data, source_time, target_time)

        if "calendar" not in metadata_kwargs:
            raise ValueError("calendar is required for temporal aggregation")
        if "time_units" not in metadata_kwargs:
            raise ValueError("time_units is required for temporal aggregation")
        calendar = metadata_kwargs["calendar"]
        time_units = metadata_kwargs["time_units"]
        # References: CF Conventions, Sections 4.4 and 7.1; time_scaling.md.
        # Mathematical expression: O_ij = max(0, min(b_i,d_j)-max(a_i,c_j)).
        # Parameter meanings: [a_i,b_i] and [c_j,d_j] are explicit time bounds.
        # Units: coordinates use the declared CF time unit and reference epoch.
        # Domain/constraints: one admitted calendar; the original overlap owner
        # validates reference dates, positive durations and non-overlap.
        # Sign convention: overlap durations are nonnegative; values stay signed.
        if not isinstance(calendar, str) or calendar not in {
            "noleap", "360_day", "gregorian", "proleptic_gregorian",
        }:
            raise ValueError(
                "calendar must be explicitly declared as noleap, 360_day, "
                "gregorian or proleptic_gregorian"
            )
        if not isinstance(time_units, str) or not time_units.strip():
            raise ValueError("time_units must be a non-empty CF time-units string")

        if temporal_semantics is TemporalSemantics.STATE_AT_BOUNDARY:
            return self._aggregate_state_boundary(
                data=data,
                source_time=source_time,
                target_time=target_time,
                units=units,
                cell_methods=cell_methods,
                variable_name=variable_name,
                standard_name=standard_name,
                calendar=calendar,
                time_units=time_units,
                metadata=metadata_kwargs,
                time_dim=time_dim,
            )

        if "source_bounds" not in metadata_kwargs:
            raise ValueError("source_bounds are required for temporal aggregation")
        if "target_bounds" not in metadata_kwargs:
            raise ValueError("target_bounds are required for temporal aggregation")

        source_bounds = self._validate_bounds(
            metadata_kwargs["source_bounds"],
            source_time,
            "source_bounds",
        )
        target_bounds = self._validate_bounds(
            metadata_kwargs["target_bounds"],
            target_time,
            "target_bounds",
        )

        # References: CF Conventions 1.13, Sections 4.4 and 7.3.
        # Mathematical expression: w_ij = |I_i intersect J_j| / |J_j|.
        # Parameter meanings: I_i is a source interval; J_j is a target interval.
        # Units: interval endpoints and durations use the declared time_units.
        # Domain/constraints: explicit finite non-overlapping bounds; |J_j| > 0.
        # Sign convention: overlap duration is nonnegative.
        weights, coverage, target_durations, plan = self._compute_weights(
            source_time,
            target_time,
            source_bounds,
            target_bounds,
            time_units=time_units,
            calendar=calendar,
        )
        output_semantics = self._resolve_output_semantics(
            temporal_semantics,
            method,
        )
        unit_contract: TemporalUnitContract | None = None
        kernel_result = None
        if output_semantics is not None:
            unit_contract = self._build_unit_contract(
                temporal_semantics,
                units,
                metadata_kwargs,
            )
            kernel_result = apply_temporal_kernel(
                plan,
                data,
                semantics=temporal_semantics,
                output_semantics=output_semantics,
                unit_contract=unit_contract,
                time_dim=time_dim,
            )
            aggregated_data = kernel_result.values
            valid_fraction = kernel_result.valid_coverage
            num_samples = self._count_valid_samples(data, plan, time_dim)
            complete_coverage = kernel_result.complete_coverage
            gap_total_fraction = kernel_result.gap_diagnostic.total_gap_fraction
            observed_measure = kernel_result.observed_measure
        else:
            aggregated_data, valid_fraction, num_samples = self._perform_aggregation(
                data,
                weights,
                coverage,
                target_durations,
                method,
                time_dim,
            )
            complete_coverage = torch.abs(valid_fraction - 1.0) <= (
                torch.finfo(valid_fraction.dtype).eps * 64.0
            )
            gap_total_fraction = 1.0 - valid_fraction
            observed_measure = None

        valid_mask = valid_fraction >= self.coverage_threshold
        n_insufficient = int(torch.sum(~valid_mask).detach().cpu().item())
        output_mask = valid_mask
        if method in {
            AggregationMethod.MINIMUM,
            AggregationMethod.MAXIMUM,
            AggregationMethod.RANGE,
        }:
            # Extrema remain defined from any positive finite overlap. The
            # independent quality mask still records threshold insufficiency.
            output_mask = valid_fraction > 0
        aggregated_data = torch.where(
            output_mask,
            aggregated_data,
            torch.full_like(aggregated_data, float("nan")),
        ).to(data.dtype)

        quality_flags = self._create_quality_flags(
            valid_fraction,
            num_samples,
            valid_mask,
        )

        public_metadata = {
            key: value
            for key, value in metadata_kwargs.items()
            if key
            not in {
                "source_bounds",
                "target_bounds",
                "temporal_semantics",
                "target_units",
                "measure_units",
                "integration_time_unit",
                "q_ref_parameter",
                "variance_units",
            }
        }
        target_units = self._required_metadata_text(metadata_kwargs, "target_units")
        if output_semantics is None and target_units != units:
            raise ValueError(
                "descriptive temporal statistics require identical source and "
                "target units because no value conversion is applied"
            )
        result_units = target_units
        if method is AggregationMethod.VARIANCE:
            variance_units = self._required_metadata_text(
                metadata_kwargs,
                "variance_units",
            )
            target_dimension = dimensionality_from_unit(target_units)
            variance_dimension = dimensionality_from_unit(variance_units)
            if target_dimension is None:
                raise ValueError(
                    "target_units must have parseable dimensionality for variance"
                )
            if variance_dimension is None:
                raise ValueError(
                    "variance_units must have parseable squared dimensionality"
                )

            # References: CF Conventions 1.13 Appendix E; BIPM SI Brochure 9th ed.
            # Mathematical expression: [Var(X)] = [X]^2 and d_var = 2 d_x.
            # Parameter meanings: d_x and d_var are SI base-dimension exponent vectors.
            # Units: variance_units must encode the square of target_units.
            # Domain/constraints: both declarations must be parseable by the unit SSOT.
            # Sign convention: dimension exponents are doubled algebraically.
            squared_dimension = Dimensionality(
                M=2.0 * target_dimension.M,
                L=2.0 * target_dimension.L,
                T=2.0 * target_dimension.T,
                Theta=2.0 * target_dimension.Theta,
                Q=2.0 * target_dimension.Q,
                N=2.0 * target_dimension.N,
                J=2.0 * target_dimension.J,
            )
            if not variance_dimension.is_compatible(squared_dimension):
                raise ValueError(
                    "variance_units must have the squared dimensionality of "
                    "target_units"
                )
            try:
                natural_variance_unit = Unit(target_units) * Unit(target_units)
                declared_variance_unit = Unit(variance_units)
                variance_scale = float(
                    natural_variance_unit.convert(
                        1.0,
                        declared_variance_unit,
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "variance_units must be executable through the canonical "
                    "CF unit engine"
                ) from exc
            if not math.isfinite(variance_scale) or not math.isclose(
                variance_scale,
                1.0,
                rel_tol=math.ulp(1.0) * 64.0,
                abs_tol=0.0,
            ):
                raise ValueError(
                    "variance_units must use the same scale as target_units squared"
                )
            result_units = variance_units
        updated_metadata = self.update_metadata(
            {
                **identity_metadata(variable_name, standard_name),
                "units": result_units,
                "cell_methods": cell_methods or "",
                "calendar": calendar,
                "time_units": time_units,
                **public_metadata,
            },
            operation="aggregation",
            method=method.value,
        )

        diagnostics = {
            "aggregation_method": method.value,
            "temporal_semantics": temporal_semantics.value,
            "insufficient_output_count": n_insufficient,
            "geometric_coverage_min": float(
                torch.min(coverage).detach().cpu().item()
            ),
            "complete_coverage": complete_coverage,
            "gap_kind": "gap",
            "gap_total_fraction": gap_total_fraction,
            "observed_measure": observed_measure,
        }
        level = self.temporal_config.resolve_conservation_level(self.name)
        if output_semantics is None:
            if level.compute_residual:
                raise ValueError(
                    f"{method.value} has no temporal mapping conservation law"
                )
            diagnostics.update(
                {
                    "conservation_level": level.value,
                    "conservation_computed": False,
                    "conservation_checked": False,
                    "conservation_passed": None,
                    "conservation_violation": False,
                    "physical_ledger_checked": False,
                }
            )
        else:
            if unit_contract is None:
                raise RuntimeError("Canonical temporal unit contract is unavailable")
            accepted_target_mask = output_mask & torch.isfinite(aggregated_data)
            if output_semantics is TemporalSemantics.INTERVAL_ACCUMULATION:
                accepted_target_mask = accepted_target_mask & complete_coverage
            diagnostics.update(
                self._evaluate_mapping_conservation(
                    plan=plan,
                    source_data=data,
                    target_data=aggregated_data,
                    source_semantics=temporal_semantics,
                    target_semantics=output_semantics,
                    unit_contract=unit_contract,
                    accepted_target_mask=accepted_target_mask,
                    variable_name=variable_name,
                    metadata=metadata_kwargs,
                    time_dim=time_dim,
                )
            )

        logger.info(
            f"Aggregated {data.shape} -> {aggregated_data.shape} using {method.value}, "
            f"coverage_threshold={self.coverage_threshold}, "
            f"insufficient_periods={n_insufficient}"
        )
        
        return AggregationResult(
            data=aggregated_data,
            metadata=updated_metadata,
            quality_flags=quality_flags,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _validate_fixed_interval_frequency(frequency: str, field_name: str) -> None:
        """Reject calendar-derived period shortcuts handled by calendar owners."""
        frequency_text = str(frequency).strip().lower()
        if any(
            token in frequency_text
            for token in ("monthly", "seasonal", "annual", "yearly")
        ):
            raise ValueError(
                f"{field_name}={frequency!r} is [N] without PeriodIndexer or "
                "CalendarHandler periodization"
            )

    def _validate_inputs(
        self,
        data: Tensor,
        source_time: Tensor,
        target_time: Tensor,
    ) -> int:
        """Validate data and coordinate domains and return a normalized time axis."""
        if not isinstance(data, Tensor):
            raise TypeError("data must be a torch.Tensor")
        if data.layout != torch.strided or data.device.type == "meta":
            raise ValueError("data must be a materialized strided tensor")
        if data.dtype not in {torch.float32, torch.float64}:
            raise ValueError("data must use float32 or float64")
        if data.dim() < 1 or any(size == 0 for size in data.shape):
            raise ValueError("data must have non-empty dimensions")
        if torch.any(torch.isinf(data)):
            raise ValueError("data contains infinite values")

        time_dim = self.aggregation_config.time_dim
        if type(time_dim) is not int:
            raise ValueError("time_dim must be an integer axis")
        normalized_time_dim = time_dim if time_dim >= 0 else data.dim() + time_dim
        if normalized_time_dim < 0 or normalized_time_dim >= data.dim():
            raise ValueError(f"time_dim {time_dim} is outside the data rank")

        for name, coordinate in (
            ("source_time", source_time),
            ("target_time", target_time),
        ):
            if not isinstance(coordinate, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if coordinate.layout != torch.strided or coordinate.device.type == "meta":
                raise ValueError(f"{name} must be a materialized strided tensor")
            if coordinate.dim() != 1:
                raise ValueError(f"{name} must be one-dimensional")
            if coordinate.numel() == 0:
                raise ValueError(f"{name} must be non-empty")
            if coordinate.dtype not in {torch.float32, torch.float64}:
                raise ValueError(f"{name} must use float32 or float64")
            if not torch.all(torch.isfinite(coordinate)):
                raise ValueError(f"{name} must contain finite values")
            if coordinate.numel() > 1 and torch.any(coordinate[1:] <= coordinate[:-1]):
                raise ValueError(f"{name} must be strictly increasing")

        time_size = data.shape[normalized_time_dim]
        if time_size != source_time.numel():
            raise ValueError(
                f"Data time dimension size {time_size} does not match "
                f"source time length {source_time.numel()}"
            )
        return normalized_time_dim

    @staticmethod
    def _validate_bounds(bounds: Any, coordinate: Tensor, name: str) -> Tensor:
        """Validate one explicit interval-bound tensor without inferring bounds."""
        if not isinstance(bounds, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if bounds.layout != torch.strided or bounds.device.type == "meta":
            raise ValueError(f"{name} must be a materialized strided tensor")
        if bounds.dtype not in {torch.float32, torch.float64}:
            raise ValueError(f"{name} must use float32 or float64")
        if bounds.dim() != 2 or bounds.shape != (coordinate.numel(), 2):
            raise ValueError(
                f"{name} must have shape ({coordinate.numel()}, 2)"
            )
        if not torch.all(torch.isfinite(bounds)):
            raise ValueError(f"{name} must contain finite values")
        durations = bounds[:, 1] - bounds[:, 0]
        if torch.any(durations <= 0):
            raise ValueError(f"{name} durations must be strictly positive")
        if bounds.shape[0] > 1:
            scale = torch.maximum(
                torch.max(torch.abs(bounds)),
                torch.ones((), dtype=bounds.dtype, device=bounds.device),
            )
            tolerance = torch.finfo(bounds.dtype).eps * scale * 16.0
            if torch.any(bounds[1:, 0] < bounds[:-1, 1] - tolerance):
                raise ValueError(f"{name} intervals overlap")
        return bounds

    def _compute_weights(
        self,
        source_time: Tensor,
        target_time: Tensor,
        source_bounds: Tensor,
        target_bounds: Tensor,
        *,
        time_units: str,
        calendar: str,
    ) -> Tuple[Tensor, Tensor, Tensor, TemporalOverlapPlan]:
        """Build one canonical overlap plan and its target-fraction view.

        The kernel laws (means, rates, accumulations) read the plan directly;
        only the descriptive statistics consume ``weights``, and they always
        use the target fractions w_ij = O_ij / |J_j|. No variable-role string
        selects a weight view.
        """
        source_coord = self.cf_time_processor.parse_cf_time(
            source_time, units=time_units, calendar=calendar
        )
        target_coord = self.cf_time_processor.parse_cf_time(
            target_time, units=time_units, calendar=calendar
        )
        source_coord.bounds = source_bounds.to(
            device=source_coord.values.device,
            dtype=source_coord.values.dtype,
        )
        target_coord.bounds = target_bounds.to(
            device=target_coord.values.device,
            dtype=target_coord.values.dtype,
        )

        plan = self.time_weight_generator.build_overlap_plan(
            source_coord,
            target_coord,
        )
        weights = plan.target_fractions
        coverage = plan.geometric_target_coverage
        target_durations = plan.target_durations

        expected_shape = (source_time.numel(), target_time.numel())
        if weights.shape != expected_shape:
            raise ValueError(
                f"TimeWeightGenerator returned weights shape {tuple(weights.shape)}; "
                f"expected {expected_shape}"
            )
        if coverage.shape != (target_time.numel(),):
            raise ValueError("TimeWeightGenerator returned an invalid coverage shape")
        if not torch.all(torch.isfinite(weights)) or torch.any(weights < 0):
            raise ValueError("TimeWeightGenerator returned invalid overlap weights")
        if not torch.all(torch.isfinite(coverage)) or torch.any(coverage < 0):
            raise ValueError("TimeWeightGenerator returned invalid geometric coverage")
        tolerance = torch.finfo(coverage.dtype).eps * 64.0
        if torch.any(coverage > 1.0 + tolerance):
            raise ValueError("source_bounds overlap or duplicate target coverage")
        return weights, coverage, target_durations, plan
    
    @staticmethod
    def _resolve_cf_cell_method(
        method: AggregationMethod | str,
        metadata: Dict[str, Any],
    ) -> Tuple[AggregationMethod, str]:
        """Resolve one RELIC aggregation method to its bounded CF token."""
        try:
            aggregation_method = AggregationMethod(method)
        except (TypeError, ValueError):
            raise ValueError(
                f"Unsupported temporal aggregation metadata method: {method!r}"
            ) from None

        if aggregation_method == AggregationMethod.PERCENTILE:
            raise ValueError(
                "Temporal aggregation method 'percentile' is [N]: CF 1.13 "
                "defines no general percentile cell-method token"
            )

        if aggregation_method in {
            AggregationMethod.STANDARD_DEVIATION,
            AggregationMethod.RANGE,
        }:
            units_metadata = metadata.get("units_metadata")
            if units_metadata not in (None, "temperature: difference"):
                raise ValueError(
                    "Temporal standard deviation and range require explicit "
                    "units_metadata='temperature: difference' when temperature "
                    "semantics are declared"
                )

        cf_method = _CF_CELL_METHOD_BY_AGGREGATION_METHOD.get(aggregation_method)
        if cf_method is None:
            raise ValueError(
                f"Unsupported temporal aggregation metadata method: {aggregation_method.value}"
            )
        return aggregation_method, cf_method

    def update_metadata(
        self,
        metadata: Dict[str, Any],
        operation: str = "aggregation",
        method: str = "time_weighted_mean"
    ) -> Dict[str, Any]:
        """
        Update the bounded temporal cell-method metadata subset.
        
        Parameters:
            metadata : Dict[str, Any]
                Original metadata
            operation : str
                Operation type ("aggregation", "disaggregation")
            method : str
                Method used
                
        Returns:
            Dict[str, Any]: Updated metadata

        Notes:
            This method does not prove time-coordinate bounds or full-dataset CF
            conformance. Unsupported metadata contracts fail before mutation.
        """
        aggregation_method, cf_method = self._resolve_cf_cell_method(method, metadata)
        updated = metadata.copy()

        updated["cell_methods"] = update_cell_methods(
            str(updated.get("cell_methods") or ""),
            cf_method,
            "time",
        )

        history_entry = f"{operation} using {aggregation_method.value}"
        existing_history = str(updated.get("history") or "")
        updated["history"] = (
            f"{existing_history}; {history_entry}"
            if existing_history
            else history_entry
        )

        updated["coverage_threshold"] = getattr(self, "coverage_threshold", None)
        if "calendar" in metadata:
            updated["calendar"] = metadata.get("calendar")
        if "time_units" in metadata:
            updated["time_units"] = metadata.get("time_units")

        return updated
    
    def _perform_aggregation(
        self,
        data: Tensor,
        weights: Tensor,
        coverage: Tensor,
        target_durations: Tensor,
        method: AggregationMethod,
        time_dim: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Apply one missing-aware statistic independently to every output cell."""
        del target_durations
        moved = torch.movedim(data, time_dim, -1)
        batch_shape = moved.shape[:-1]
        source_count = moved.shape[-1]
        batch_size = moved.numel() // source_count
        flat_data = moved.reshape(batch_size, source_count)
        local_weights = weights.to(device=flat_data.device, dtype=flat_data.dtype)
        local_coverage = coverage.to(device=flat_data.device, dtype=flat_data.dtype)

        finite_mask = ~torch.isnan(flat_data)
        effective_weights = (
            finite_mask.to(flat_data.dtype).unsqueeze(-1)
            * local_weights.unsqueeze(0)
        )
        valid_coverage = torch.sum(effective_weights, dim=1)
        num_samples = torch.sum(effective_weights > 0, dim=1)
        tolerance = torch.finfo(flat_data.dtype).eps * 64.0
        if torch.any(valid_coverage > local_coverage.unsqueeze(0) + tolerance):
            raise ValueError("valid coverage exceeds geometric coverage")

        # References: CF Conventions 1.13, Section 7.3 and Appendix E.
        # Mathematical expression: y_bj = sum_i(w_ij m_bi x_bi) / sum_i(w_ij m_bi).
        # Parameter meanings: w_ij is normalized overlap; m_bi is the finite mask.
        # Units: median, extrema, range, and standard deviation retain data units.
        # Domain/constraints: NaN is missing; Inf was rejected; valid weight must be positive.
        # Sign convention: all overlap weights are nonnegative.
        if method == AggregationMethod.MINIMUM:
            flat_result = self._aggregate_extrema(flat_data, effective_weights, "min")
        elif method == AggregationMethod.MAXIMUM:
            flat_result = self._aggregate_extrema(flat_data, effective_weights, "max")
        elif method == AggregationMethod.MEDIAN:
            flat_result = self._aggregate_median(flat_data, effective_weights)
        elif method == AggregationMethod.VARIANCE:
            flat_result = self._aggregate_variance(flat_data, effective_weights)
        elif method == AggregationMethod.STANDARD_DEVIATION:
            flat_result = torch.sqrt(
                self._aggregate_variance(flat_data, effective_weights)
            )
        elif method == AggregationMethod.RANGE:
            minimum = self._aggregate_extrema(flat_data, effective_weights, "min")
            maximum = self._aggregate_extrema(flat_data, effective_weights, "max")
            flat_result = maximum - minimum
        else:
            raise ValueError(f"Unsupported aggregation method: {method.value}")

        target_count = local_weights.shape[1]
        output_shape = batch_shape + (target_count,)

        def restore(flat: Tensor) -> Tensor:
            restored = flat.reshape(output_shape)
            if time_dim != len(output_shape) - 1:
                restored = torch.movedim(restored, -1, time_dim)
            return restored

        return (
            restore(flat_result),
            restore(valid_coverage),
            restore(num_samples),
        )
    
    @staticmethod
    def _aggregate_extrema(
        data: Tensor,
        effective_weights: Tensor,
        operation: str,
    ) -> Tensor:
        """Return min or max over finite samples with positive overlap."""
        membership = effective_weights > 0
        expanded = data.unsqueeze(-1).expand_as(effective_weights)
        if operation == "min":
            candidate = torch.where(
                membership,
                expanded,
                torch.full_like(expanded, float("inf")),
            )
            result = torch.min(candidate, dim=1).values
        elif operation == "max":
            candidate = torch.where(
                membership,
                expanded,
                torch.full_like(expanded, float("-inf")),
            )
            result = torch.max(candidate, dim=1).values
        else:
            raise ValueError(f"Unsupported extrema operation: {operation}")
        return torch.where(
            torch.any(membership, dim=1),
            result,
            torch.full_like(result, float("nan")),
        )
    
    @staticmethod
    def _aggregate_median(data: Tensor, effective_weights: Tensor) -> Tensor:
        """Return the lower duration-weighted median for every output cell."""
        safe_data = torch.where(torch.isnan(data), torch.zeros_like(data), data)
        sorted_data, order = torch.sort(safe_data, dim=1)
        expanded_order = order.unsqueeze(-1).expand_as(effective_weights)
        sorted_weights = torch.gather(effective_weights, 1, expanded_order)
        cumulative = torch.cumsum(sorted_weights, dim=1)
        total = torch.sum(sorted_weights, dim=1)
        threshold = 0.5 * total
        selected = torch.argmax(
            (cumulative >= threshold.unsqueeze(1)).to(torch.int64),
            dim=1,
        )
        sorted_values = sorted_data.unsqueeze(-1).expand_as(effective_weights)
        result = torch.gather(
            sorted_values,
            1,
            selected.unsqueeze(1),
        ).squeeze(1)
        return torch.where(
            total > 0,
            result,
            torch.full_like(result, float("nan")),
        )
    
    def _aggregate_variance(
        self,
        data: Tensor,
        effective_weights: Tensor,
    ) -> Tensor:
        """Return duration-weighted population variance for internal std use."""
        denominator = torch.sum(effective_weights, dim=1)
        safe_data = torch.where(torch.isnan(data), torch.zeros_like(data), data)
        mean = torch.sum(
            safe_data.unsqueeze(-1) * effective_weights,
            dim=1,
        ) / torch.clamp(denominator, min=self.coverage_minimum)
        deviations = torch.where(
            effective_weights > 0,
            data.unsqueeze(-1) - mean.unsqueeze(1),
            torch.zeros_like(effective_weights),
        )
        variance = torch.sum(
            effective_weights * deviations.square(),
            dim=1,
        ) / torch.clamp(denominator, min=self.coverage_minimum)
        return torch.where(
            denominator > 0,
            variance,
            torch.full_like(variance, float("nan")),
        )
    
    def _create_quality_flags(
        self,
        valid_fraction: Tensor,
        num_samples: Tensor,
        valid_mask: Tensor,
    ) -> QualityFlags:
        """Create quality flags with the same shape as the aggregated output."""
        return QualityFlags(
            valid_fraction=valid_fraction,
            num_samples=num_samples,
            missing_fraction=torch.clamp(1.0 - valid_fraction, min=0.0, max=1.0),
            insufficient_coverage_mask=~valid_mask,
        )

    def downscale(self, coarse_data: Tensor) -> Tensor:
        """Fail closed because temporal disaggregation has a separate owner."""
        del coarse_data
        raise ValueError(
            "TemporalAggregator cannot downscale; use TemporalDisaggregator"
        )


# Export main classes and enums
__all__ = [
    'TemporalAggregator',
    'AggregationMethod',
    'AggregationConfig',
    'AggregationResult',
    'QualityFlags'
]




