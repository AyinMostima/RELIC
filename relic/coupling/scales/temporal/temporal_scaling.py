"""Direction-neutral temporal facade over the canonical interval kernels.

The frozen ``resample`` API supports only fixed SI durations. Calendar-dependent
month, quarter, season, and year mappings require explicit calendar bounds and
therefore fail closed on this ratio-only surface.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor

from relic.core.abstractions import TemporalScalingBase
from relic.core.functor import (
    ConflictResolutionPolicy,
    ExecutionPhase,
    FluxDict,
    FunctorMetadata,
    ProcessCategory,
    ScaleCouplingFunctor,
    SpatialScale,
    StateDict,
    TemporalScale,
    VariableKind,
)
from relic.coupling.scales.temporal.core import (
    TemporalUnitContract,
    TimeCoordinate,
    TimeWeightGenerator,
    _materialize_device,
    apply_temporal_kernel,
    compute_temporal_mapping_residual,
)
from relic.coupling.scales.temporal.types import ConservationLevel, TemporalSemantics


class TemporalScaling(ScaleCouplingFunctor, TemporalScalingBase):
    """Canonical direction-neutral temporal scaling facade.

    References:
        - CF Conventions 1.13, Sections 4.4, 7.1, and 7.3.
        - Jones, P.W. (1999), Mon. Weather Rev. 127:2204-2210.
    Mathematical expression:
        ``O_ij=max(0,min(a_i+,b_j+)-max(a_i-,b_j-))``. Fixed SI
        ``interval_mean`` and ``interval_accumulation`` mappings delegate to
        the shared duration-overlap kernel.
    Parameter meanings:
        ``source_resolution`` and ``target_resolution`` are positive fixed SI
        durations. ``method`` selects an explicit temporal semantics.
        ``pattern_type`` must be ``"constant"``: piecewise-constant
        reconstruction is the only one executed. Unknown keyword arguments
        raise ``TypeError``.
    Units:
        Ratio-only values are dimensionless; interval bounds use seconds.
    Domain/constraints:
        The source domain must partition exactly into complete target
        intervals. Calendar-dependent scales require an explicit-bounds API.
    Sign convention:
        Mapping residual is target measure minus source measure and never
        modifies the returned values.
    """

    _CALENDAR_DEPENDENT_TOKENS = (
        "month",
        "quarter",
        "season",
        "year",
        "annual",
    )
    # The only reconstruction the shared kernel executes is piecewise
    # constant over each interval; profile/driver patterns have no owner.
    _EXECUTED_PATTERN_TYPES = frozenset({"constant"})
    _ACCEPTED_KEYWORDS = frozenset(
        {
            "input_variables",
            "output_variables",
            "config",
            "device",
            "conservation_level",
            "q_ref_parameter",
        }
    )

    def __init__(
        self,
        source_scale: str,
        target_scale: str,
        source_resolution: float = 1.0,
        target_resolution: float = 1.0,
        *,
        method: str,
        pattern_type: str = "constant",
        name: str = "temporal_scaling",
        **kwargs: Any,
    ) -> None:
        unknown = sorted(set(kwargs) - self._ACCEPTED_KEYWORDS)
        if unknown:
            raise TypeError(
                "TemporalScaling got unexpected keyword arguments: "
                + ", ".join(unknown)
            )
        if pattern_type not in self._EXECUTED_PATTERN_TYPES:
            raise ValueError(
                f"pattern_type {pattern_type!r} has no executable reconstruction; "
                "only 'constant' (piecewise-constant) is implemented"
            )
        q_ref_parameter = kwargs.get("q_ref_parameter")
        if q_ref_parameter is not None:
            if not isinstance(q_ref_parameter, str):
                raise TypeError("q_ref_parameter must be a ParameterManager key string")
            if not q_ref_parameter.strip():
                raise ValueError("q_ref_parameter must be a non-empty ParameterManager key")
        initial_semantics = self._semantics_from_method(method)
        initial_law = self._law_id(initial_semantics)
        input_variables = list(kwargs.get("input_variables", ["time_series"]))
        output_variables = list(
            kwargs.get("output_variables", ["time_series_scaled"])
        )
        metadata = FunctorMetadata(
            name=name,
            version="2.0.0",
            description="Canonical fixed-SI temporal scaling facade",
            category=ProcessCategory.SCALE_COUPLING,
            execution_phase=ExecutionPhase.COUPLING,
            mathematical_expression=(
                "O_ij=max(0,min(a_i+,b_j+)-max(a_i-,b_j-)); "
                "mapping uses explicit interval semantics"
            ),
            references=[
                "CF Conventions Version 1.13, Sections 4.4, 7.1, and 7.3",
                "Jones, P.W. (1999), Mon. Weather Rev. 127:2204-2210",
            ],
            parameters={
                "validation.conservation.conservation_tolerance": (
                    "Relative tolerance of the temporal mapping amount ledger [dimensionless]; "
                    "the conservation level decides whether it is compared and whether a "
                    "violation blocks"
                ),
            },
            input_variables=input_variables,
            output_variables=output_variables,
            conservation_laws=[initial_law],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.SECOND,
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.DIAGNOSTIC,
            conflict_resolution_policy=ConflictResolutionPolicy.EXCLUSIVE,
        )
        super().__init__(
            name=name,
            metadata=metadata,
            config=kwargs.get("config", {}),
            device=kwargs.get("device"),
            source_scale=source_scale,
            target_scale=target_scale,
            coupling_type="temporal",
        )
        self.source_scale = source_scale
        self.target_scale = target_scale
        self.source_resolution = self._positive_duration(
            source_resolution,
            "source_resolution",
        )
        self.target_resolution = self._positive_duration(
            target_resolution,
            "target_resolution",
        )
        self.method = method
        self.pattern_type = pattern_type
        self.conservation_level = ConservationLevel.from_value(
            kwargs.get("conservation_level", ConservationLevel.NONE)
        )
        self.q_ref_parameter = (
            None if q_ref_parameter is None else q_ref_parameter.strip()
        )
        # No numerical threshold is read: a unity ratio is exact in the overlap kernel
        # (O_ij = delta_ij on coincident bounds; the partition check uses exact Fraction
        # arithmetic), so the removed shortcut |target_dt / source_dt - 1| < eps has no
        # successor.
        self.time_weight_generator = TimeWeightGenerator(
            device=self.device,
            parameter_manager=self.get_parameter_manager(),
        )
        self._active_semantics = initial_semantics
        self._active_law = initial_law
        self.metadata.conservation_laws = [initial_law]
        self._last_temporal_diagnostics: dict[str, Any] = {
            "active_law": initial_law,
            "temporal_semantics": initial_semantics.value,
            "conservation_level": self.conservation_level.value,
            "conservation_computed": False,
            "conservation_checked": False,
            "conservation_passed": None,
            "conservation_violation": False,
        }

    @staticmethod
    def _positive_duration(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be one real scalar")
        result = float(value)
        if not math.isfinite(result) or result <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
        return result

    @staticmethod
    def _semantics_from_method(method: Any) -> TemporalSemantics:
        normalized = str(method).strip().lower()
        aliases = {
            "mean": TemporalSemantics.INTERVAL_MEAN,
            "interval_mean": TemporalSemantics.INTERVAL_MEAN,
            "sum": TemporalSemantics.INTERVAL_ACCUMULATION,
            "total": TemporalSemantics.INTERVAL_ACCUMULATION,
            "extensive": TemporalSemantics.INTERVAL_ACCUMULATION,
            "interval_accumulation": TemporalSemantics.INTERVAL_ACCUMULATION,
            "rate": TemporalSemantics.RATE,
        }
        try:
            return aliases[normalized]
        except KeyError:
            raise ValueError(
                f"Unsupported fixed-SI temporal method: {method!r}"
            ) from None

    @staticmethod
    def _law_id(semantics: TemporalSemantics) -> str:
        if semantics is TemporalSemantics.INTERVAL_MEAN:
            return "temporal_observed_integral_consistency"
        if semantics is TemporalSemantics.RATE:
            return "temporal_rate_integral_consistency"
        return "temporal_interval_accumulation_partition"

    def _activate_semantics(self, semantics: TemporalSemantics) -> str:
        law_id = self._law_id(semantics)
        self._active_semantics = semantics
        self._active_law = law_id
        self.metadata.conservation_laws = [law_id]
        return law_id

    @property
    def last_temporal_diagnostics(self) -> dict[str, Any]:
        """Return the latest canonical diagnostics without recomputation."""
        return {
            key: value.clone() if isinstance(value, Tensor) else value
            for key, value in self._last_temporal_diagnostics.items()
        }

    def to_device(self, device: torch.device | str) -> "TemporalScaling":
        """Move the facade and its canonical overlap owner together."""
        target = _materialize_device(torch.device(device))
        super().to_device(target)
        self.time_weight_generator.device = target
        self.time_weight_generator.cf_processor.device = target
        return self

    def _reject_calendar_dependent_ratio(self) -> None:
        for scale in (self.source_scale, self.target_scale):
            normalized = str(scale).strip().lower()
            if any(token in normalized for token in self._CALENDAR_DEPENDENT_TOKENS):
                raise ValueError(
                    "calendar-dependent temporal scales require explicit bounds"
                )

    @staticmethod
    def _series_tensor(
        series: Union[Tensor, float, list],
        device: torch.device,
    ) -> Tensor:
        if isinstance(series, Tensor):
            data = series.to(device=device)
        else:
            data = torch.as_tensor(series, dtype=torch.float32, device=device)
        if data.dim() == 0:
            data = data.reshape(1)
        if data.numel() == 0 or data.shape[0] == 0:
            raise ValueError("series must have one non-empty time axis")
        if data.dtype not in {torch.float32, torch.float64}:
            raise ValueError("series must use float32 or float64")
        if not bool(torch.all(torch.isfinite(data)).item()):
            raise ValueError("series must contain finite values")
        return data

    @staticmethod
    def _fixed_bounds(
        source_count: int,
        source_dt: float,
        target_dt: float,
        *,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        source_step = Fraction(str(source_dt))
        target_step = Fraction(str(target_dt))
        total_duration = source_step * source_count
        target_count_fraction = total_duration / target_step
        if target_count_fraction.denominator != 1:
            raise ValueError(
                "source domain must contain complete target intervals"
            )
        target_count = int(target_count_fraction)
        if target_count <= 0:
            raise ValueError("target interval count must be positive")
        source_edges = torch.arange(
            source_count + 1,
            dtype=torch.float64,
            device=device,
        ) * float(source_step)
        target_edges = torch.arange(
            target_count + 1,
            dtype=torch.float64,
            device=device,
        ) * float(target_step)
        source_bounds = torch.stack((source_edges[:-1], source_edges[1:]), dim=1)
        target_bounds = torch.stack((target_edges[:-1], target_edges[1:]), dim=1)
        return source_bounds, target_bounds

    @staticmethod
    def _unit_contract(semantics: TemporalSemantics) -> TemporalUnitContract:
        if semantics is TemporalSemantics.INTERVAL_MEAN:
            return TemporalUnitContract(
                source_units="dimensionless",
                target_units="dimensionless",
                measure_units="dimensionless s",
                integration_time_unit="s",
            )
        if semantics is TemporalSemantics.RATE:
            return TemporalUnitContract(
                source_units="dimensionless s-1",
                target_units="dimensionless s-1",
                measure_units="dimensionless",
                integration_time_unit="s",
            )
        return TemporalUnitContract(
            source_units="dimensionless",
            target_units="dimensionless",
            measure_units="dimensionless",
            integration_time_unit=None,
        )

    def _evaluate_conservation(
        self,
        *,
        plan: Any,
        source_data: Tensor,
        target_data: Tensor,
        semantics: TemporalSemantics,
        unit_contract: TemporalUnitContract,
        accepted_target_mask: Tensor,
    ) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {
            "conservation_level": self.conservation_level.value,
            "conservation_computed": False,
            "conservation_checked": False,
            "conservation_passed": None,
            "conservation_violation": False,
        }
        if not self.conservation_level.compute_residual:
            return diagnostics
        if self.q_ref_parameter is None:
            raise ValueError(
                "q_ref_parameter is required when temporal conservation is enabled"
            )
        q_ref = self.get_parameter_value(
            self.q_ref_parameter,
            units=unit_contract.measure_units,
            validate=True,
            required=True,
        )
        residual = compute_temporal_mapping_residual(
            plan,
            source_data,
            target_data,
            semantics=semantics,
            unit_contract=unit_contract,
            law_id=self._active_law,
            accepted_target_mask=accepted_target_mask,
            time_dim=0,
            q_ref=q_ref,
            q_ref_provenance=self.q_ref_parameter,
        )
        diagnostics.update(
            {
                "conservation_computed": True,
                "mapping_residual": residual.residual,
                "mapping_relative_residual": residual.relative_residual,
                "q_ref_provenance": self.q_ref_parameter,
            }
        )
        if not self.conservation_level.compare_tolerance:
            return diagnostics
        if residual.relative_residual is None:
            raise RuntimeError("Relative temporal mapping residual was not computed")
        maximum_relative = torch.max(residual.relative_residual)
        # References: LeVeque, R.J. (1992), Numerical Methods for Conservation Laws;
        #   RELIC conservation tolerance SSOT (validation/conservation.yaml).
        # Mathematical expression: passed = max_j |R_j| / max(A_j, q_ref) <= tol, with
        #   tol = validation.conservation.conservation_tolerance.
        # Parameter meanings: R_j mapping residual and A_j source activity of output
        #   element j; q_ref the declared reference magnitude; tol the general relative
        #   conservation tolerance, the one key TemporalScaling declares and reads.
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
        if not passed and self.conservation_level.block_on_violation:
            raise ValueError(
                "temporal conservation violation: maximum relative residual "
                f"{float(maximum_relative.detach().cpu().item()):.6e} exceeds "
                f"tolerance {tolerance:.6e}"
            )
        return diagnostics

    def _run_core(
        self,
        data: Tensor,
        *,
        source_dt: float,
        target_dt: float,
        semantics: TemporalSemantics,
    ) -> Tensor:
        """Execute one fixed-SI mapping through the shared core.

        References: CF Conventions 1.13 and Jones (1999).
        Mathematical expression: target values use the canonical ``O_ij`` law.
        Parameter meanings: source_dt and target_dt define explicit SI bounds.
        Units: bounds are seconds; values use the semantics-specific contract.
        Domain/constraints: the source domain exactly tiles target intervals.
        Sign convention: residual is target measure minus source measure.
        """
        law_id = self._activate_semantics(semantics)
        source_bounds, target_bounds = self._fixed_bounds(
            data.shape[0],
            source_dt,
            target_dt,
            device=data.device,
        )
        source_coordinate = TimeCoordinate(
            values=(source_bounds[:, 0] + source_bounds[:, 1]) * 0.5,
            units="seconds since 2000-01-01",
            calendar="noleap",
            bounds=source_bounds,
        )
        target_coordinate = TimeCoordinate(
            values=(target_bounds[:, 0] + target_bounds[:, 1]) * 0.5,
            units="seconds since 2000-01-01",
            calendar="noleap",
            bounds=target_bounds,
        )
        plan = self.time_weight_generator.build_overlap_plan(
            source_coordinate,
            target_coordinate,
        )
        unit_contract = self._unit_contract(semantics)
        kernel_result = apply_temporal_kernel(
            plan,
            data,
            semantics=semantics,
            output_semantics=semantics,
            unit_contract=unit_contract,
            time_dim=0,
        )
        accepted = torch.isfinite(kernel_result.values) & (
            kernel_result.valid_coverage > 0
        )
        conservation = self._evaluate_conservation(
            plan=plan,
            source_data=data,
            target_data=kernel_result.values,
            semantics=semantics,
            unit_contract=unit_contract,
            accepted_target_mask=accepted,
        )
        self._last_temporal_diagnostics = {
            "active_law": law_id,
            "temporal_semantics": semantics.value,
            "bounds_relation": plan.relation.value,
            "observed_measure": kernel_result.observed_measure,
            "valid_coverage": kernel_result.valid_coverage,
            "complete_coverage": kernel_result.complete_coverage,
            "gap_total_fraction": kernel_result.gap_diagnostic.total_gap_fraction,
            **conservation,
        }
        return kernel_result.values

    def resample(  # type: ignore[override]
        self,
        series: Union[Tensor, float, list],
        *,
        source_dt: Optional[float] = None,
        target_dt: Optional[float] = None,
        method: str,
    ) -> Tensor:
        """Resample fixed-SI intervals through the canonical temporal core."""
        self._reject_calendar_dependent_ratio()
        data = self._series_tensor(series, self.device)
        resolved_source_dt = self._positive_duration(
            self.source_resolution if source_dt is None else source_dt,
            "source_dt",
        )
        resolved_target_dt = self._positive_duration(
            self.target_resolution if target_dt is None else target_dt,
            "target_dt",
        )
        semantics = self._semantics_from_method(method)
        return self._run_core(
            data,
            source_dt=resolved_source_dt,
            target_dt=resolved_target_dt,
            semantics=semantics,
        )

    def upscale(self, fine_data: Tensor, **kwargs: Any) -> Tensor:
        """Map source-resolution intervals to target-resolution intervals."""
        return self.resample(
            fine_data,
            source_dt=kwargs.get("source_dt", self.source_resolution),
            target_dt=kwargs.get("target_dt", self.target_resolution),
            method=kwargs.get("method", self.method),
        )

    def downscale(self, coarse_data: Tensor, **kwargs: Any) -> Tensor:
        """Map target-resolution intervals back to source-resolution intervals."""
        return self.resample(
            coarse_data,
            source_dt=kwargs.get("source_dt", self.target_resolution),
            target_dt=kwargs.get("target_dt", self.source_resolution),
            method=kwargs.get("method", self.method),
        )

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Expose only the latest canonical temporal mapping residual."""
        del input_state, output_updates
        residual = self._last_temporal_diagnostics.get("mapping_residual")
        if not isinstance(residual, Tensor):
            return {self._active_law: 0.0}
        magnitude = float(torch.max(torch.abs(residual)).detach().cpu().item())
        return {self._active_law: magnitude}

    def forward(
        self,
        state: StateDict,
        time: float,
        dt: float,
        **kwargs: Any,
    ) -> FluxDict:
        """Scale the declared input while preserving the output key."""
        # References: CF Conventions 1.13, Sections 4.4, 7.1, 7.3, and Jones (1999) as cited in the
        #   class docstring; RELIC ScaleCoupling contract (relic/core/functor/scale_coupling_bridge.py).
        # Mathematical expression: y = resample(x, source_dt = source_resolution, target_dt = dt, method)
        #   via _run_core on fixed bounds I_i = [i s, (i+1) s], J_j = [j dt, (j+1) dt] and
        #   O_ij = max(0, min(a_i+, b_j+) - max(a_i-, b_j-)): interval_mean and rate
        #   y_j = sum_i O_ij x_i / sum_i O_ij; interval_accumulation y_j = sum_i (O_ij / s) x_i.
        # Parameter meanings: x = state[input_variables[0]]; s = source_resolution; dt is the simulator
        #   step used as target interval; method tokens mean/interval_mean, sum/total/extensive/
        #   interval_accumulation, and rate select the semantics; y is written to output_variables[0].
        # Units: bounds in seconds; the dimensionless TemporalUnitContract (means and accumulations
        #   dimensionless, rates dimensionless s-1) leaves data units unchanged.
        # Domain/constraints: input variable must be in state (ValueError); series finite float32/64
        #   with a non-empty time axis; s and dt positive and finite; N s must be an integer multiple
        #   of dt; scale names containing month/quarter/season/year/annual are rejected; a residual
        #   above validation.conservation.conservation_tolerance raises when the conservation
        #   level blocks.
        # Sign convention: signed data preserved; O_ij >= 0; residual R = target minus source measure.
        del time, kwargs
        input_var = self.metadata.input_variables[0]
        output_var = self.metadata.output_variables[0]
        if input_var not in state:
            raise ValueError(
                f"Missing required state variable '{input_var}' for temporal scaling"
            )
        scaled = self.resample(
            state[input_var],
            source_dt=self.source_resolution,
            target_dt=dt,
            method=self.method,
        )
        return {output_var: scaled}


__all__ = ["TemporalScaling"]
