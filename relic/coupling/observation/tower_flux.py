"""Flux-footprint observation operator for tower-equivalent diagnostics."""

from __future__ import annotations

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


class TowerFluxSupportOperator(ScaleCouplingFunctor):
    """Validate a flux footprint and expose its tower-equivalent flux."""

    FOOTPRINT_FLUX = (
        "diagnostics.observation.tower_flux."
        "footprint_weighted_isoprene_flux"
    )
    TOWER_FLUX = (
        "diagnostics.observation.tower_flux.tower_equivalent_flux"
    )
    SUPPORT_INTEGRAL = (
        "diagnostics.observation.tower_flux.support_integral"
    )
    SUPPORT_AREA = "diagnostics.observation.tower_flux.support_area"
    WEIGHT_SUM = "diagnostics.observation.tower_flux.weight_sum"
    COVERAGE_FRACTION = (
        "diagnostics.observation.tower_flux.coverage_fraction"
    )
    EXTRAPOLATION_FLAG = (
        "diagnostics.observation.tower_flux.support_extrapolation_flag"
    )
    SUPPORT_CLOSURE_TOLERANCE = (
        "coupling.observation.tower_flux.support_closure_tolerance"
    )

    _CONTEXT_INPUTS = (
        "footprint_weights",
        "footprint_kernel_mass",
        "footprint_cell_area",
        "footprint_data_coverage_fraction",
        "measurement_height",
        "time_bounds",
        "footprint_density_available",
        "flux_footprint_valid",
    )

    def __init__(
        self,
        name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        functor_name = name or "tower_flux_support_operator"
        output_variables = [
            self.TOWER_FLUX,
            self.SUPPORT_INTEGRAL,
            self.SUPPORT_AREA,
            self.WEIGHT_SUM,
            self.COVERAGE_FRACTION,
            self.EXTRAPOLATION_FLAG,
        ]
        metadata = FunctorMetadata(
            name=functor_name,
            version="1.0.0",
            description=(
                "Read-only identity observation operator for a validated "
                "flux-footprint support reduction."
            ),
            category=ProcessCategory.SCALE_COUPLING,
            execution_phase=ExecutionPhase.POSTPROCESSING,
            mathematical_expression=(
                "F_tower = H_flux(F_fp) = F_fp, conditional on "
                "sum(alpha_i)=1 and complete footprint-kernel closure"
            ),
            references=[
                (
                    "Kljun et al. (2015), Geosci. Model Dev. 8, "
                    "3695-3713, doi:10.5194/gmd-8-3695-2015"
                ),
                (
                    "Aubinet, Vesala and Papale (2012), Eddy Covariance: "
                    "A Practical Guide to Measurement and Data Analysis"
                ),
                (
                    "Finnigan (2004), Agricultural and Forest Meteorology "
                    "127, 117-129"
                ),
            ],
            parameters={
                self.SUPPORT_CLOSURE_TOLERANCE: (
                    "Dimensionless numerical tolerance for normalized "
                    "weight, kernel-mass, and valid-data closure"
                )
            },
            input_variables=[
                self.FOOTPRINT_FLUX,
                *self._CONTEXT_INPUTS,
            ],
            output_variables=output_variables,
            conservation_laws=[],
            spatial_scale=SpatialScale.POINT,
            variable_kind=VariableKind.DIAGNOSTIC,
            coupling_interfaces=["flux_footprint_observation"],
        )
        super().__init__(
            name=functor_name,
            config=config or {},
            metadata=metadata,
            source_scale="flux_footprint",
            target_scale="tower_point",
            coupling_type="spatial",
            conservation_properties=[],
        )
        # The weighted flux is required from the declared upstream graph
        # dependency, not from persistent StateManager storage. The forward
        # path remains fail-closed when that dependency output is absent.
        self.optional_input_variables = {self.FOOTPRINT_FLUX}

    def required_parameters(self) -> tuple[str, ...]:
        """Return the sole ParameterManager-backed numerical threshold."""

        return (self.SUPPORT_CLOSURE_TOLERANCE,)

    def observation_operator(self) -> ObservationOperatorSpec:
        """Return the structural contract for the flux observation."""

        return ObservationOperatorSpec(
            name="tower_flux_support_observation",
            input_state_vars=tuple(self.metadata.input_variables),
            output_name=self.TOWER_FLUX,
            output_units="mol m-2 s-1",
            support="flux_footprint",
            averaging_window="interval_mean",
            footprint_description=(
                "Validated cell-integrated flux-footprint kernel for the "
                "declared measurement height and observation window"
            ),
            error_model="external_observation_error_spec",
            conservative_projection_required=False,
            metadata={
                "runtime_path": (
                    "temporal_resample->support_aggregation->"
                    "tower_flux_support_operator"
                ),
                "parameter_keys": self.required_parameters(),
                "height_transfer_factor_applied": False,
                "weights_normalized": True,
                "raw_kernel_mass_required": True,
                "footprint_density_required": True,
                "qc_applied": False,
                "qc_stage": "post_observation",
            },
        )

    def diagnostic_operator(self) -> DiagnosticSpec:
        """Return the read-only support-closure diagnostic contract."""

        return DiagnosticSpec(
            name="tower_flux_support_closure_diagnostic",
            input_state_vars=(self.TOWER_FLUX,),
            output_units="1",
            diagnostic_kind="flux_footprint_support_closure",
            writes_state=False,
            metadata={
                "outputs": tuple(self.metadata.output_variables),
                "invalid_cases": "fail_closed",
            },
        )

    @staticmethod
    def _unwrap(value: Any) -> Any:
        return value.value if isinstance(value, UpdateItem) else value

    @staticmethod
    def _require_tensor(
        state: Mapping[str, Any],
        name: str,
    ) -> Tensor:
        if name not in state:
            raise KeyError(f"Required tower-flux input '{name}' is missing.")
        value = state[name]
        if not isinstance(value, Tensor):
            raise TypeError(
                f"Tower-flux input '{name}' must be a torch.Tensor."
            )
        return value

    def _resolve_footprint_flux(
        self,
        *,
        state: Mapping[str, Any],
        subprocess_outputs: Mapping[str, Any],
    ) -> Tensor:
        state_value = state.get(self.FOOTPRINT_FLUX)
        dependency_values = []
        for node_id, outputs in subprocess_outputs.items():
            if not isinstance(outputs, Mapping):
                continue
            if self.FOOTPRINT_FLUX in outputs:
                dependency_values.append(
                    (str(node_id), outputs[self.FOOTPRINT_FLUX])
                )
        if state_value is not None and dependency_values:
            raise RuntimeError(
                "The footprint-weighted flux input is ambiguous between "
                "StateManager and dependency outputs."
            )
        if len(dependency_values) > 1:
            owners = ", ".join(node_id for node_id, _ in dependency_values)
            raise RuntimeError(
                "The footprint-weighted flux input is ambiguous across "
                f"dependency outputs: {owners}."
            )
        if state_value is not None:
            resolved = state_value
        elif dependency_values:
            resolved = dependency_values[0][1]
        else:
            raise KeyError(
                "Required footprint-weighted flux is unavailable from the "
                "declared dependency."
            )
        resolved = self._unwrap(resolved)
        if not isinstance(resolved, Tensor):
            raise TypeError(
                "The footprint-weighted flux must be a torch.Tensor."
            )
        return resolved

    @staticmethod
    def _require_scalar_finite(value: Tensor, label: str) -> None:
        if value.ndim != 0:
            raise ValueError(f"{label} must be scalar.")
        if not bool(torch.isfinite(value)):
            raise ValueError(f"{label} must be finite.")

    @classmethod
    def _require_true_gate(cls, value: Tensor, label: str) -> None:
        cls._require_scalar_finite(value, label)
        expected = torch.ones((), dtype=value.dtype, device=value.device)
        if not bool(torch.equal(value, expected)):
            raise ValueError(f"{label} gate must be true.")

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
        Map a validated footprint-weighted source flux to tower flux.

        References:
            Kljun et al. (2015); Aubinet et al. (2012); Finnigan
            (2004).
        Mathematical Expression:
            w_i = integral_(A_i) f dA; alpha_i = w_i / sum_j w_j;
            F_fp = sum_i alpha_i F_i; H_flux(F_fp) = F_fp.
        Parameters:
            f is footprint density [m-2]; A_i is cell area [m2];
            alpha_i is normalized weight [1]; F_i and F_fp are flux
            densities [mol m-2 s-1]; support_closure_tolerance is the
            dimensionless numerical tolerance for sums that must equal one.
        """
        # References: Kljun et al. (2015); Aubinet, Vesala and Papale (2012); Finnigan
        #   (2004); ObservationOperatorSpec from observation_operator().
        # Mathematical expression: F_tower = H_flux(F_fp) = F_fp, accepted only if
        #   |sum_i alpha_i - 1| <= tol, |m_kernel - 1| <= tol and |c_data - 1| <= tol;
        #   A_support = sum_{i: alpha_i > 0} A_i; extrapolation_flag = 0.
        # Parameter meanings: F_fp footprint-weighted flux from the declared dependency
        #   (or state, never both); alpha_i normalized footprint weights; m_kernel kernel
        #   mass; c_data data coverage; A_i cell areas; tol closure tolerance; z_m height.
        # Units: F_fp, tower_equivalent_flux [mol m-2 s-1]; alpha, m_kernel, c_data,
        #   weight_sum, coverage_fraction, extrapolation_flag [1]; A_i, support_area [m2];
        #   z_m [m]; time_bounds [s]; outputs REPLACE, DERIVED, per_timestep=False.
        # Domain/constraints: F_fp scalar finite from one source; alpha 1-D finite >= 0;
        #   A_i same shape finite > 0; m_kernel, c_data scalar finite; z_m > 0; two strictly
        #   increasing time bounds; both gates exactly 1; 0 < tol < 1; else ValueError.
        # Sign convention: the sign of F_fp is passed through unchanged; this operator
        #   defines no convention of its own (upstream flux convention applies).
        del time, dt, kwargs
        dependency_outputs = subprocess_outputs or {}
        footprint_flux = self._resolve_footprint_flux(
            state=state,
            subprocess_outputs=dependency_outputs,
        )
        self._require_scalar_finite(
            footprint_flux,
            "Footprint-weighted flux",
        )

        weights = self._require_tensor(state, "footprint_weights").to(
            dtype=footprint_flux.dtype,
            device=footprint_flux.device,
        )
        if weights.ndim != 1 or weights.numel() == 0:
            raise ValueError(
                "Footprint weights must be a non-empty one-dimensional tensor."
            )
        if not bool(torch.all(torch.isfinite(weights))):
            raise ValueError("Footprint weights must be finite.")
        if bool(torch.any(weights < 0)):
            raise ValueError("Footprint weights must be non-negative.")

        cell_areas = self._require_tensor(
            state,
            "footprint_cell_area",
        ).to(dtype=footprint_flux.dtype, device=footprint_flux.device)
        if cell_areas.shape != weights.shape:
            raise ValueError(
                "Footprint cell-area shape must match footprint weights."
            )
        if not bool(torch.all(torch.isfinite(cell_areas))) or bool(
            torch.any(cell_areas <= 0)
        ):
            raise ValueError(
                "Footprint cell areas must be finite and positive."
            )

        kernel_mass = self._require_tensor(
            state,
            "footprint_kernel_mass",
        ).to(dtype=footprint_flux.dtype, device=footprint_flux.device)
        data_coverage = self._require_tensor(
            state,
            "footprint_data_coverage_fraction",
        ).to(dtype=footprint_flux.dtype, device=footprint_flux.device)
        measurement_height = self._require_tensor(
            state,
            "measurement_height",
        ).to(dtype=footprint_flux.dtype, device=footprint_flux.device)
        time_bounds = self._require_tensor(state, "time_bounds").to(
            dtype=footprint_flux.dtype,
            device=footprint_flux.device,
        )
        density_available = self._require_tensor(
            state,
            "footprint_density_available",
        ).to(dtype=footprint_flux.dtype, device=footprint_flux.device)
        footprint_valid = self._require_tensor(
            state,
            "flux_footprint_valid",
        ).to(dtype=footprint_flux.dtype, device=footprint_flux.device)

        self._require_scalar_finite(kernel_mass, "Footprint kernel mass")
        self._require_scalar_finite(data_coverage, "Footprint data coverage")
        self._require_scalar_finite(
            measurement_height,
            "Footprint measurement height",
        )
        if not bool(measurement_height > 0):
            raise ValueError(
                "Footprint measurement height must be positive."
            )
        if time_bounds.ndim != 1 or time_bounds.shape[0] != 2:
            raise ValueError(
                "Tower-flux support requires exactly two time bounds."
            )
        if not bool(torch.all(torch.isfinite(time_bounds))):
            raise ValueError("Tower-flux time bounds must be finite.")
        if not bool(time_bounds[1] > time_bounds[0]):
            raise ValueError(
                "Tower-flux time bounds must be strictly increasing."
            )
        self._require_true_gate(
            density_available,
            "Footprint density availability",
        )
        self._require_true_gate(
            footprint_valid,
            "Flux-footprint validity",
        )

        tolerance_value = self.get_parameter_value(
            self.SUPPORT_CLOSURE_TOLERANCE,
            required=True,
        )
        tolerance = torch.as_tensor(
            tolerance_value,
            dtype=footprint_flux.dtype,
            device=footprint_flux.device,
        )
        self._require_scalar_finite(
            tolerance,
            "Support closure tolerance",
        )
        if not bool(tolerance > 0) or not bool(tolerance < 1):
            raise ValueError(
                "Support closure tolerance must be between zero and one."
            )

        one = torch.ones(
            (),
            dtype=footprint_flux.dtype,
            device=footprint_flux.device,
        )
        weight_sum = weights.sum()
        if bool(torch.abs(weight_sum - one) > tolerance):
            raise ValueError(
                "Normalized footprint weights must sum to one within the "
                "support closure tolerance."
            )
        if bool(torch.abs(kernel_mass - one) > tolerance):
            raise ValueError(
                "Footprint kernel mass must close to one before normalized "
                "weights can represent a tower-equivalent flux."
            )
        if bool(torch.abs(data_coverage - one) > tolerance):
            raise ValueError(
                "Footprint data coverage must close to one for a "
                "tower-equivalent flux."
            )

        support_area = cell_areas[weights > 0].sum()
        extrapolation_flag = torch.zeros_like(kernel_mass)
        return {
            self.TOWER_FLUX: self._derived(
                value=footprint_flux,
                unit="mol m-2 s-1",
                source_functor=self.name,
            ),
            self.SUPPORT_INTEGRAL: self._derived(
                value=kernel_mass,
                unit="1",
                source_functor=self.name,
            ),
            self.SUPPORT_AREA: self._derived(
                value=support_area,
                unit="m2",
                source_functor=self.name,
            ),
            self.WEIGHT_SUM: self._derived(
                value=weight_sum,
                unit="1",
                source_functor=self.name,
            ),
            self.COVERAGE_FRACTION: self._derived(
                value=data_coverage,
                unit="1",
                source_functor=self.name,
            ),
            self.EXTRAPOLATION_FLAG: self._derived(
                value=extrapolation_flag,
                unit="1",
                source_functor=self.name,
            ),
        }


__all__ = ["TowerFluxSupportOperator"]
