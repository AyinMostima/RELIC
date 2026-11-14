"""Three-height concentration observation operator for vertical columns."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from ...computation.operators.dec.forms import (
    DiagnosticSpec,
    ObservationOperatorSpec,
)
from ...core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    ScaleCouplingFunctor,
    SpatialScale,
    VariableKind,
)
from ...core.state import StateWriteMode, UpdateItem, UpdateMode
from ...utils.physical_constants import get_physical_constant


class TowerConcentrationProfileOperator(ScaleCouplingFunctor):
    """Map a one-dimensional mass profile to three tower ppbv values."""

    RAW_PPBV = (
        "diagnostics.observation.tower_concentration.raw_ppbv"
    )
    SUPPORT_WEIGHT_SUM = (
        "diagnostics.observation.tower_concentration.support_weight_sum"
    )
    WINDOW_COVERAGE = (
        "diagnostics.observation.tower_concentration.window_coverage"
    )
    WINDOW_COMPLETE = (
        "diagnostics.observation.tower_concentration.window_complete"
    )

    MEASUREMENT_HEIGHTS_KEY = (
        "coupling.observation.tower_concentration."
        "synthetic_measurement_heights"
    )
    SUPPORT_HALF_WIDTHS_KEY = (
        "coupling.observation.tower_concentration."
        "synthetic_support_half_widths"
    )
    WINDOW_DURATION_KEY = (
        "coupling.observation.tower_concentration.window_duration"
    )
    SUPPORT_CLOSURE_TOLERANCE_KEY = (
        "coupling.observation.tower_concentration."
        "support_closure_tolerance"
    )
    ISOPRENE_MOLAR_MASS_KEY = (
        "relic.physical.constants.molecular_weights.isoprene"
    )

    _SUPPORT_MODES = {"point_linear", "finite_layer_overlap"}
    _AIR_BASES = {"wet", "dry"}
    _HEIGHT_COUNT = 3

    def __init__(
        self,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        *,
        support_mode: str = "point_linear",
        air_basis: str = "wet",
    ) -> None:
        selected_support = str(support_mode).strip().lower()
        selected_basis = str(air_basis).strip().lower()
        if selected_support not in self._SUPPORT_MODES:
            raise ValueError(
                "support_mode must be 'point_linear' or "
                "'finite_layer_overlap'."
            )
        if selected_basis not in self._AIR_BASES:
            raise ValueError("air_basis must be 'wet' or 'dry'.")

        geometry_input = (
            "layer_center_height"
            if selected_support == "point_linear"
            else "layer_interface_height"
        )
        pressure_input = (
            "atmospheric_pressure"
            if selected_basis == "wet"
            else "dry_air_pressure"
        )
        functor_name = name or "tower_concentration_profile_operator"
        output_variables = [
            self.RAW_PPBV,
            self.SUPPORT_WEIGHT_SUM,
            self.WINDOW_COVERAGE,
            self.WINDOW_COMPLETE,
        ]
        parameter_metadata = {
            self.MEASUREMENT_HEIGHTS_KEY: (
                "Three synthetic receptor heights [m]"
            ),
            self.WINDOW_DURATION_KEY: (
                "Duration-weighted observation window [s]"
            ),
            self.SUPPORT_CLOSURE_TOLERANCE_KEY: (
                "Numerical support and window closure tolerance [1]"
            ),
            self.ISOPRENE_MOLAR_MASS_KEY: (
                "Isoprene molar mass physical constant [g mol-1]"
            ),
            "relic.physical.constants.universal.gas_constant": (
                "Universal molar gas constant [J mol-1 K-1]"
            ),
        }
        if selected_support == "finite_layer_overlap":
            parameter_metadata[self.SUPPORT_HALF_WIDTHS_KEY] = (
                "Synthetic finite-support half widths [m]"
            )
        support_expression = (
            "chi_point = 1e9 (W_z rho) R (W_z T) / "
            "(M_iso (W_z p_basis))"
            if selected_support == "point_linear"
            else (
                "chi_layer = 1e9 rho R T / (M_iso p_basis); "
                "chi_interval = W_z chi_layer"
            )
        )
        metadata = FunctorMetadata(
            name=functor_name,
            version="1.0.0",
            description=(
                "Read-only three-height concentration observation with "
                "explicit vertical and temporal support."
            ),
            category=ProcessCategory.SCALE_COUPLING,
            execution_phase=ExecutionPhase.POSTPROCESSING,
            mathematical_expression=(
                f"{support_expression}; "
                "y_bar = sum(y_n overlap_n) / sum(overlap_n)"
            ),
            references=[
                (
                    "Seinfeld and Pandis (2016), Atmospheric Chemistry "
                    "and Physics, 3rd ed."
                ),
                (
                    "Stull (1988), An Introduction to Boundary Layer "
                    "Meteorology."
                ),
                "CF Metadata Conventions 1.8, cell_methods.",
                "Patankar (1980), Numerical Heat Transfer and Fluid Flow.",
            ],
            parameters=parameter_metadata,
            input_variables=[
                "tracer_concentration",
                "air_temperature",
                pressure_input,
                geometry_input,
            ],
            output_variables=output_variables,
            conservation_laws=[],
            spatial_scale=SpatialScale.POINT,
            variable_kind=VariableKind.DIAGNOSTIC,
            coupling_interfaces=["vertical_profile_observation"],
        )
        super().__init__(
            name=functor_name,
            config=config or {},
            metadata=metadata,
            source_scale="vertical_column",
            target_scale="tower_height",
            coupling_type="hierarchical",
            conservation_properties=[],
        )
        self.support_mode = selected_support
        self.air_basis = selected_basis

        measurement_heights = self._load_vector_parameter(
            self.MEASUREMENT_HEIGHTS_KEY,
            expected_length=self._HEIGHT_COUNT,
            positive=True,
        )
        self.register_buffer(
            "measurement_heights",
            measurement_heights,
        )
        if self.support_mode == "finite_layer_overlap":
            support_half_widths = self._load_vector_parameter(
                self.SUPPORT_HALF_WIDTHS_KEY,
                expected_length=self._HEIGHT_COUNT,
                positive=True,
            )
            self.register_buffer(
                "support_half_widths",
                support_half_widths,
            )

        window_duration = self._load_scalar_parameter(
            self.WINDOW_DURATION_KEY,
            positive=True,
        )
        support_tolerance = self._load_scalar_parameter(
            self.SUPPORT_CLOSURE_TOLERANCE_KEY,
            positive=True,
        )
        if not support_tolerance < 1.0:
            raise ValueError(
                "Tower concentration support tolerance must be below one."
            )
        self.register_buffer(
            "window_duration",
            torch.tensor(window_duration, dtype=torch.float64),
        )
        self.register_buffer(
            "support_closure_tolerance",
            torch.tensor(support_tolerance, dtype=torch.float64),
        )

        gas_constant = get_physical_constant(
            "gas_constant",
            as_tensor=True,
            device=self.device,
        )
        molar_mass = get_physical_constant(
            self.ISOPRENE_MOLAR_MASS_KEY,
            as_tensor=True,
            device=self.device,
        )
        self.register_buffer(
            "gas_constant",
            torch.as_tensor(
                gas_constant,
                dtype=torch.float64,
                device=self.device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "isoprene_molar_mass_kg_per_mol",
            torch.as_tensor(
                molar_mass,
                dtype=torch.float64,
                device=self.device,
            )
            * 1.0e-3,
            persistent=False,
        )

        self.register_buffer(
            "_window_numerator",
            torch.zeros(self._HEIGHT_COUNT, dtype=torch.float64),
        )
        self.register_buffer(
            "_window_covered_duration",
            torch.zeros((), dtype=torch.float64),
        )
        self.register_buffer(
            "_active_window_bounds",
            torch.full((2,), float("nan"), dtype=torch.float64),
        )
        self.register_buffer(
            "_window_initialized",
            torch.zeros((), dtype=torch.bool),
        )
        self.register_buffer(
            "_last_step_end",
            torch.full((), float("nan"), dtype=torch.float64),
        )

    def _load_vector_parameter(
        self,
        key: str,
        *,
        expected_length: int,
        positive: bool,
    ) -> Tensor:
        value = self.get_parameter_value(key, required=True)
        tensor = torch.as_tensor(value, dtype=torch.float64)
        if tensor.ndim != 1 or tensor.numel() != expected_length:
            raise ValueError(
                f"Parameter '{key}' must contain {expected_length} values."
            )
        if not bool(torch.all(torch.isfinite(tensor))):
            raise ValueError(f"Parameter '{key}' must be finite.")
        if positive and bool(torch.any(tensor <= 0)):
            raise ValueError(f"Parameter '{key}' must be positive.")
        return tensor

    def _load_scalar_parameter(self, key: str, *, positive: bool) -> float:
        value = self.get_parameter_value(key, required=True)
        tensor = torch.as_tensor(value, dtype=torch.float64)
        if tensor.ndim != 0 or not bool(torch.isfinite(tensor)):
            raise ValueError(f"Parameter '{key}' must be a finite scalar.")
        scalar = float(tensor.detach().cpu().item())
        if positive and scalar <= 0:
            raise ValueError(f"Parameter '{key}' must be positive.")
        return scalar

    def required_parameters(self) -> tuple[str, ...]:
        """Return ParameterManager keys selected by this operator instance."""

        keys = [
            self.MEASUREMENT_HEIGHTS_KEY,
            self.WINDOW_DURATION_KEY,
            self.SUPPORT_CLOSURE_TOLERANCE_KEY,
        ]
        if self.support_mode == "finite_layer_overlap":
            keys.append(self.SUPPORT_HALF_WIDTHS_KEY)
        return tuple(keys)

    def observation_operator(self) -> ObservationOperatorSpec:
        """Return the typed three-height concentration observation map."""

        return ObservationOperatorSpec(
            name="tower_concentration_profile_observation",
            input_state_vars=tuple(self.metadata.input_variables),
            output_name=self.RAW_PPBV,
            output_units="ppbv",
            support="three_height_vertical_profile",
            averaging_window="duration_weighted_window",
            footprint_description=(
                "Three independently normalized point-linear or finite-"
                "overlap rows over the declared vertical column"
            ),
            error_model="external_instrument_or_site_error_spec",
            conservative_projection_required=False,
            metadata={
                "runtime_path": (
                    "tracer_concentration->vertical_support->"
                    "duration_weighted_window"
                ),
                "support_mode": self.support_mode,
                "air_basis": self.air_basis,
                "measurement_height_count": self._HEIGHT_COUNT,
                "physical_constant_keys": (
                    self.ISOPRENE_MOLAR_MASS_KEY,
                    "relic.physical.constants.universal.gas_constant",
                ),
                "qc_applied": False,
                "qc_stage": "post_observation",
                "dependency_delta_used": False,
            },
        )

    def diagnostic_operator(self) -> DiagnosticSpec:
        """Return the read-only support and temporal closure diagnostic."""

        return DiagnosticSpec(
            name="tower_concentration_support_diagnostic",
            input_state_vars=(self.RAW_PPBV,),
            output_units="1",
            diagnostic_kind="vertical_and_temporal_support_closure",
            writes_state=False,
            metadata={
                "outputs": tuple(self.metadata.output_variables),
                "invalid_cases": "fail_closed",
            },
        )

    @staticmethod
    def _require_profile(
        state: Mapping[str, Any],
        key: str,
        *,
        length: int | None = None,
    ) -> Tensor:
        if key not in state:
            raise KeyError(f"Required tower-concentration input '{key}' is missing.")
        value = state[key]
        if not isinstance(value, Tensor):
            raise TypeError(
                f"Tower-concentration input '{key}' must be a torch.Tensor."
            )
        if value.ndim != 1:
            raise ValueError(
                f"Tower-concentration input '{key}' must be one-dimensional."
            )
        if length is not None and value.numel() != length:
            raise ValueError(
                f"Tower-concentration input '{key}' must contain {length} values."
            )
        if not bool(torch.all(torch.isfinite(value))):
            raise ValueError(
                f"Tower-concentration input '{key}' must be finite."
            )
        return value

    @staticmethod
    def _require_column_profile(
        state: Mapping[str, Any],
        key: str,
    ) -> Tensor:
        """Return one vertical profile from a strict singleton column."""

        if key not in state:
            raise KeyError(
                f"Required tower-concentration input '{key}' is missing."
            )
        value = state[key]
        if not isinstance(value, Tensor):
            raise TypeError(
                f"Tower-concentration input '{key}' must be a torch.Tensor."
            )
        if value.ndim == 2:
            if value.shape[0] != 1:
                raise ValueError(
                    "Tower concentration supports exactly one horizontal cell."
                )
            value = value[0]
        elif value.ndim != 1:
            raise ValueError(
                f"Tower-concentration input '{key}' must have shape "
                "[layer] or [1, layer]."
            )
        if not bool(torch.all(torch.isfinite(value))):
            raise ValueError(
                f"Tower-concentration input '{key}' must be finite."
            )
        return value

    @classmethod
    def _require_thermodynamic_profile(
        cls,
        state: Mapping[str, Any],
        key: str,
        *,
        length: int,
    ) -> Tensor:
        """Broadcast one cell value or accept one explicit layer profile."""

        value = cls._require_profile(state, key)
        if value.numel() == 1:
            return value.expand(length)
        if value.numel() != length:
            raise ValueError(
                f"Tower-concentration input '{key}' must contain one or "
                f"{length} values."
            )
        return value

    @staticmethod
    def _require_strictly_increasing(values: Tensor, label: str) -> None:
        if values.numel() < 2 or not bool(torch.all(values[1:] > values[:-1])):
            raise ValueError(f"{label} must be strictly increasing.")

    def _point_linear_weights(
        self,
        centers: Tensor,
        heights: Tensor,
    ) -> Tensor:
        self._require_strictly_increasing(centers, "Layer-center heights")
        if bool(torch.any(heights < centers[0])) or bool(
            torch.any(heights > centers[-1])
        ):
            raise ValueError(
                "Measurement height is outside the modeled column."
            )
        rows = []
        for height in heights:
            if bool(torch.equal(height, centers[0])):
                row = torch.zeros_like(centers)
                row[0] = 1.0
            elif bool(torch.equal(height, centers[-1])):
                row = torch.zeros_like(centers)
                row[-1] = 1.0
            else:
                right = int(torch.searchsorted(centers, height).item())
                left = right - 1
                fraction = (
                    (height - centers[left])
                    / (centers[right] - centers[left])
                )
                row = torch.zeros_like(centers)
                row[left] = 1.0 - fraction
                row[right] = fraction
            rows.append(row)
        return torch.stack(rows)

    def _finite_overlap_weights(
        self,
        interfaces: Tensor,
        heights: Tensor,
    ) -> Tensor:
        self._require_strictly_increasing(
            interfaces,
            "Layer-interface heights",
        )
        half_widths = self.support_half_widths.to(
            dtype=interfaces.dtype,
            device=interfaces.device,
        )
        rows = []
        for height, half_width in zip(heights, half_widths, strict=True):
            support_lower = height - half_width
            support_upper = height + half_width
            if bool(support_lower < interfaces[0]) or bool(
                support_upper > interfaces[-1]
            ):
                raise ValueError(
                    "Finite observation support is outside the modeled column."
                )
            raw_overlap = torch.clamp(
                torch.minimum(support_upper, interfaces[1:])
                - torch.maximum(support_lower, interfaces[:-1]),
                min=0.0,
            )
            total_overlap = raw_overlap.sum()
            if not bool(total_overlap > 0):
                raise ValueError(
                    "Finite observation support has zero layer overlap."
                )
            rows.append(raw_overlap / total_overlap)
        return torch.stack(rows)

    def _support_weights(
        self,
        state: Mapping[str, Any],
        layer_count: int,
        reference: Tensor,
    ) -> Tensor:
        heights = self.measurement_heights.to(
            dtype=reference.dtype,
            device=reference.device,
        )
        if self.support_mode == "point_linear":
            geometry = self._require_profile(
                state,
                "layer_center_height",
                length=layer_count,
            ).to(dtype=reference.dtype, device=reference.device)
            weights = self._point_linear_weights(geometry, heights)
        else:
            geometry = self._require_profile(
                state,
                "layer_interface_height",
                length=layer_count + 1,
            ).to(dtype=reference.dtype, device=reference.device)
            weights = self._finite_overlap_weights(geometry, heights)

        tolerance = self.support_closure_tolerance.to(
            dtype=reference.dtype,
            device=reference.device,
        )
        row_sums = weights.sum(dim=1)
        if not bool(torch.all(torch.isfinite(weights))) or bool(
            torch.any(weights < 0)
        ):
            raise ValueError(
                "Vertical observation weights must be finite and non-negative."
            )
        if bool(torch.any(torch.abs(row_sums - 1.0) > tolerance)):
            raise ValueError(
                "Vertical observation weights must close to one."
            )
        return weights

    @staticmethod
    def _scalar_float(value: Any, label: str) -> float:
        tensor = torch.as_tensor(value)
        if tensor.ndim != 0 or not bool(torch.isfinite(tensor)):
            raise ValueError(f"{label} must be a finite scalar.")
        return float(tensor.detach().cpu().item())

    def _initialize_window(self, time: float) -> None:
        duration = float(self.window_duration.detach().cpu().item())
        start = math.floor(time / duration) * duration
        with torch.no_grad():
            self._active_window_bounds.copy_(
                torch.tensor(
                    [start, start + duration],
                    dtype=self._active_window_bounds.dtype,
                    device=self._active_window_bounds.device,
                )
            )
            self._window_numerator.zero_()
            self._window_covered_duration.zero_()
            self._window_initialized.fill_(True)

    def _persist_window(
        self,
        *,
        numerator: Tensor,
        covered_duration: float,
        bounds: tuple[float, float],
        step_end: float,
    ) -> None:
        with torch.no_grad():
            self._window_numerator.copy_(
                numerator.detach().to(
                    dtype=self._window_numerator.dtype,
                    device=self._window_numerator.device,
                )
            )
            self._window_covered_duration.fill_(covered_duration)
            self._active_window_bounds.copy_(
                torch.tensor(
                    bounds,
                    dtype=self._active_window_bounds.dtype,
                    device=self._active_window_bounds.device,
                )
            )
            self._last_step_end.fill_(step_end)
            self._window_initialized.fill_(True)

    def _duration_weighted_window(
        self,
        instantaneous: Tensor,
        *,
        time: Any,
        dt: Any,
    ) -> tuple[Tensor, Tensor, Tensor]:
        start_time = self._scalar_float(time, "time")
        step_duration = self._scalar_float(dt, "dt")
        if start_time < 0 or step_duration <= 0:
            raise ValueError("time must be non-negative and dt must be positive.")
        window_duration = float(
            self.window_duration.detach().cpu().item()
        )
        closure_tolerance = float(
            self.support_closure_tolerance.detach().cpu().item()
        )
        time_tolerance = window_duration * closure_tolerance
        if step_duration > window_duration + time_tolerance:
            raise ValueError(
                "A model step cannot span more than one observation window."
            )
        if not bool(self._window_initialized.detach().cpu().item()):
            self._initialize_window(start_time)

        bounds = self._active_window_bounds.detach().cpu().tolist()
        active_start, active_end = float(bounds[0]), float(bounds[1])
        last_step_end = float(self._last_step_end.detach().cpu().item())
        if math.isfinite(last_step_end) and (
            start_time < last_step_end - time_tolerance
        ):
            raise ValueError(
                "Tower concentration intervals must not overlap in time."
            )
        if start_time > active_end + time_tolerance:
            raise ValueError(
                "Tower concentration intervals must not skip an active "
                "observation window."
            )

        previous_numerator = self._window_numerator.to(
            dtype=instantaneous.dtype,
            device=instantaneous.device,
        )
        previous_coverage = float(
            self._window_covered_duration.detach().cpu().item()
        )
        step_end = start_time + step_duration
        overlap_end = min(step_end, active_end)
        overlap_start = max(start_time, active_start)
        overlap = max(0.0, overlap_end - overlap_start)
        current_numerator = (
            previous_numerator + instantaneous * overlap
        )
        current_coverage = previous_coverage + overlap
        if current_coverage <= 0:
            raise ValueError(
                "Tower concentration window has no modeled time coverage."
            )

        if step_end >= active_end:
            output_numerator = current_numerator
            output_coverage = current_coverage
            next_start = active_end
            next_end = active_end + window_duration
            remaining = max(0.0, step_end - active_end)
            next_numerator = instantaneous * remaining
            self._persist_window(
                numerator=next_numerator,
                covered_duration=remaining,
                bounds=(next_start, next_end),
                step_end=step_end,
            )
        else:
            output_numerator = current_numerator
            output_coverage = current_coverage
            self._persist_window(
                numerator=current_numerator,
                covered_duration=current_coverage,
                bounds=(active_start, active_end),
                step_end=step_end,
            )

        coverage_fraction = output_coverage / window_duration
        complete = (
            abs(coverage_fraction - 1.0) <= closure_tolerance
        )
        mean = output_numerator / output_coverage
        coverage = torch.full_like(mean, coverage_fraction)
        completion = torch.full_like(mean, 1.0 if complete else 0.0)
        return mean, coverage, completion

    @staticmethod
    def _derived(
        *,
        value: Tensor,
        unit: str,
        source_functor: str,
    ) -> UpdateItem:
        return UpdateItem(
            value=value,
            mode=UpdateMode.REPLACE,
            write_mode=StateWriteMode.DERIVED,
            unit=unit,
            per_timestep=False,
            source_functor=source_functor,
        )

    def forward(
        self,
        state: dict[str, Tensor],
        time: float,
        dt: Any,
        subprocess_outputs: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, UpdateItem]:
        """
        Observe the updated vertical concentration state.

        References:
            Seinfeld and Pandis (2016); Stull (1988); CF-1.8.
        Mathematical Expression:
            Point support: chi_ppbv = 1e9 (W_z rho) R (W_z T) /
            (M_iso (W_z p_basis)). Finite support:
            chi_ppbv = W_z[1e9 rho R T / (M_iso p_basis)].
            In both cases, y_bar = sum_n(y_n overlap_n) /
            sum_n(overlap_n).
        Parameters:
            rho is isoprene mass concentration [kg m-3]; W_z is a
            row-normalized vertical support [1]; R is the gas constant
            [J mol-1 K-1]; T is temperature [K]; M_iso is isoprene molar
            mass [kg mol-1]; p_basis is wet- or dry-air pressure [Pa].
        """
        # References: Seinfeld and Pandis (2016); Stull (1988); CF-1.8 cell_methods;
        #   Patankar (1980); ObservationOperatorSpec from observation_operator().
        # Mathematical expression: point_linear chi = 1e9 (W rho) R (W T) / (M (W p));
        #   finite_layer_overlap chi = W [1e9 rho R T / (M p)]; y_bar = sum_n(chi_n
        #   overlap_n) / sum_n(overlap_n); coverage = sum_n(overlap_n) / T_w;
        #   complete = (|coverage - 1| <= tol).
        # Parameter meanings: rho isoprene mass concentration; W 3 x N row-normalized
        #   vertical support; R gas constant; T air temperature; M isoprene molar mass;
        #   p wet or dry pressure basis; overlap_n step-window overlap; T_w window; tol.
        # Units: rho [kg m-3]; T [K]; p [Pa]; R [J mol-1 K-1]; M [kg mol-1]; geometry [m];
        #   time, dt, T_w [s]; raw_ppbv [ppbv]; weight sum, coverage, complete [1];
        #   outputs are REPLACE UpdateItems, StateWriteMode.DERIVED, per_timestep=False.
        # Domain/constraints: rho finite, >= 0, shape [N] or [1, N]; T > 0, p > 0 with 1
        #   or N values; geometry strictly increasing, receptors inside the column; W rows
        #   >= 0 summing to 1 within tol; time >= 0, 0 < dt <= T_w(1 + tol); read-only.
        # Sign convention: not applicable: chi is a non-negative mixing ratio; no flux direction.
        del subprocess_outputs, kwargs
        concentration = self._require_column_profile(
            state,
            "tracer_concentration",
        )
        if bool(torch.any(concentration < 0)):
            raise ValueError(
                "tracer_concentration must be non-negative."
            )
        layer_count = concentration.numel()
        temperature = self._require_thermodynamic_profile(
            state,
            "air_temperature",
            length=layer_count,
        ).to(dtype=concentration.dtype, device=concentration.device)
        pressure_key = (
            "atmospheric_pressure"
            if self.air_basis == "wet"
            else "dry_air_pressure"
        )
        pressure = self._require_thermodynamic_profile(
            state,
            pressure_key,
            length=layer_count,
        ).to(dtype=concentration.dtype, device=concentration.device)
        if bool(torch.any(temperature <= 0)):
            raise ValueError("air_temperature must be positive.")
        if bool(torch.any(pressure <= 0)):
            raise ValueError(f"{pressure_key} must be positive.")

        weights = self._support_weights(
            state,
            layer_count,
            concentration,
        )
        gas_constant = self.gas_constant.to(
            dtype=concentration.dtype,
            device=concentration.device,
        )
        molar_mass = self.isoprene_molar_mass_kg_per_mol.to(
            dtype=concentration.dtype,
            device=concentration.device,
        )

        if self.support_mode == "point_linear":
            supported_concentration = weights @ concentration
            supported_temperature = weights @ temperature
            supported_pressure = weights @ pressure

            # References: Seinfeld and Pandis (2016); CF Conventions 1.8.
            # Expression: chi(z_H) = 1e9*(W*rho)*R*(W*T)/(M*(W*p)).
            # Parameters: W reconstructs a point [1], rho is mass
            # concentration [kg m-3], T is temperature [K], p is pressure
            # [Pa], R is [J mol-1 K-1], and M is [kg mol-1].
            instantaneous_ppbv = (
                1.0e9
                * supported_concentration
                * gas_constant
                * supported_temperature
                / (molar_mass * supported_pressure)
            )
        else:
            # References: Seinfeld and Pandis (2016); CF Conventions 1.8.
            # Expression: chi_H = W[1e9*rho*R*T/(M*p_basis)].
            # Parameters: W is finite interval support [1], rho is mass
            # concentration [kg m-3], T is temperature [K], p is pressure
            # [Pa], R is [J mol-1 K-1], and M is [kg mol-1].
            layer_ppbv = (
                1.0e9
                * concentration
                * gas_constant
                * temperature
                / (molar_mass * pressure)
            )
            instantaneous_ppbv = weights @ layer_ppbv
        window_mean, coverage, completion = (
            self._duration_weighted_window(
                instantaneous_ppbv,
                time=time,
                dt=dt,
            )
        )
        support_weight_sum = weights.sum(dim=1)
        return {
            self.RAW_PPBV: self._derived(
                value=window_mean,
                unit="ppbv",
                source_functor=self.name,
            ),
            self.SUPPORT_WEIGHT_SUM: self._derived(
                value=support_weight_sum,
                unit="1",
                source_functor=self.name,
            ),
            self.WINDOW_COVERAGE: self._derived(
                value=coverage,
                unit="1",
                source_functor=self.name,
            ),
            self.WINDOW_COMPLETE: self._derived(
                value=completion,
                unit="1",
                source_functor=self.name,
            ),
        }


__all__ = ["TowerConcentrationProfileOperator"]
