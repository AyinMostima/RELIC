"""Managed state feedback from a direct observation (G11-OBS).

The functor closes the observation loop through the governed path: it reads
the model state and the observation from the StateManager, forms the
innovation with the identity observation operator, gates it against the
declared error model, and proposes the analysis as a REPLACE update that only
the StateManager commits. Read-only observation operators (tower flux and
concentration operators) stay read-only; this functor is the only feedback
writer and it must be listed as a tendency writer in the target variable's
SCM contract.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor

from ...core.abstractions import FluxDict, StateDict
from ...core.functor import ScaleCouplingFunctor
from ...core.functor.functor_types import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    SpatialScale,
    VariableKind,
)
from ...core.state.types import UpdateMode
from ...optimization.assimilation.base import IdentityOperator


class ObservationStateFeedbackFunctor(ScaleCouplingFunctor):
    """Scalar Kalman analysis of one state variable against its observation."""

    OBSERVATION_ERROR_VARIANCE = "coupling.observation.state_feedback.observation_error_variance"
    BACKGROUND_ERROR_VARIANCE = "coupling.observation.state_feedback.background_error_variance"
    MAX_NORMALIZED_INNOVATION = "coupling.observation.state_feedback.max_normalized_innovation"
    DIAG = "diagnostics.observation.state_feedback."
    DIAGNOSTICS = (
        DIAG + "innovation",
        DIAG + "normalized_innovation",
        DIAG + "kalman_gain",
        DIAG + "analysis_increment",
        DIAG + "analysis_error_variance",
    )

    def __init__(
        self,
        name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        config = dict(config or {})
        state_variable = str(config.get("state_variable", "")).strip()
        observation_variable = str(config.get("observation_variable", "")).strip()
        if not state_variable or not observation_variable:
            raise ValueError(
                "ObservationStateFeedbackFunctor requires config 'state_variable' and 'observation_variable'"
            )
        if state_variable == observation_variable:
            raise ValueError("state_variable and observation_variable must differ")
        self.state_variable = state_variable
        self.observation_variable = observation_variable
        self.state_units: Optional[str] = config.get("units")
        functor_name = name or "observation_state_feedback"
        metadata = FunctorMetadata(
            name=functor_name,
            version="1.0.0",
            description=(
                "Managed observation feedback: scalar Kalman analysis of one "
                "state variable proposed as a governed REPLACE update."
            ),
            category=ProcessCategory.SCALE_COUPLING,
            execution_phase=ExecutionPhase.COUPLING,
            mathematical_expression=(
                "d = y - H x, K = B / (B + R), x_a = x + K d, P_a = (1 - K) B, "
                "accepted only when |d| / sqrt(B + R) <= nu_max"
            ),
            references=[
                "Kalman (1960), Journal of Basic Engineering 82, 35-45",
                "Kalnay (2003), Atmospheric Modeling, Data Assimilation and Predictability, ch. 5",
                "Evensen (2003), Ocean Dynamics 53, 343-367",
            ],
            parameters={
                self.OBSERVATION_ERROR_VARIANCE: "Observation error variance R [YAML units: dimensionless; interpreted as the bound state units squared]",
                self.BACKGROUND_ERROR_VARIANCE: "Background error variance B [YAML units: dimensionless; interpreted as the bound state units squared]",
                self.MAX_NORMALIZED_INNOVATION: "Innovation gate nu_max [dimensionless]",
            },
            input_variables=[state_variable, observation_variable],
            output_variables=[state_variable, *self.DIAGNOSTICS],
            conservation_laws=[],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
            coupling_interfaces=["direct_state_observation"],
        )
        super().__init__(
            name=functor_name,
            config=config,
            metadata=metadata,
            source_scale="observation_support",
            target_scale="model_state_support",
            coupling_type="stochastic",
            conservation_properties=[],
        )
        self.optional_input_variables = []
        self._operator = IdentityOperator([state_variable])

    # ------------------------------------------------------------------
    # Scale coupling contract: observation and state share one support.
    # ------------------------------------------------------------------
    def upscale(self, fine_data: Tensor) -> Tensor:
        return fine_data

    def downscale(self, coarse_data: Tensor) -> Tensor:
        return coarse_data

    @staticmethod
    def analysis(x: Tensor, y: Tensor, background_variance: Tensor, observation_variance: Tensor) -> Dict[str, Tensor]:
        """Scalar Kalman analysis with the identity observation operator.

        References: Kalman (1960); Kalnay (2003) Eq. 5.3.11-5.3.13.
        Mathematical expression: d = y - x; K = B / (B + R); x_a = x + K d;
            P_a = (1 - K) B; nu = |d| / sqrt(B + R).
        Parameter meanings: x background state, y observation, B background
            error variance, R observation error variance, K gain, d innovation,
            x_a analysis, P_a analysis error variance, nu normalized innovation.
        Units: x, y, d, x_a in the state units; B, R, P_a in those units
            squared; K and nu dimensionless.
        Domain/constraints: B > 0 and R > 0 so that 0 < K < 1; all inputs
            finite; the gate on nu is applied by the caller.
        Sign convention: d > 0 means the observation exceeds the background,
            and the analysis moves toward the observation.
        """
        total = background_variance + observation_variance
        gain = background_variance / total
        innovation = y - x
        analysis = x + gain * innovation
        analysis_variance = (1.0 - gain) * background_variance
        normalized = torch.abs(innovation) / torch.sqrt(total)
        return {
            "innovation": innovation,
            "gain": gain,
            "analysis": analysis,
            "analysis_variance": analysis_variance,
            "normalized_innovation": normalized,
        }

    def forward(self, state: StateDict, time: float, dt: float, **kwargs: Any) -> FluxDict:
        # References: Kalman (1960); Kalnay (2003) Eq. 5.3.11-5.3.13; Evensen (2003);
        #   module docstring (G11-OBS); analysis() staticmethod above.
        # Mathematical expression: d = y - x (identity H); K = B / (B + R); x_a = x + K d;
        #   P_a = (1 - K) B; nu = |d| / sqrt(B + R); refused as a whole if any nu > nu_max.
        # Parameter meanings: x background state; y direct observation; B background
        #   error variance; R observation error variance; K Kalman gain; d innovation;
        #   x_a analysis; P_a analysis error variance; nu normalized innovation; nu_max gate.
        # Units: x, y, d, x_a, increment in the state units (config "units" is applied
        #   to the analysis UpdateItem when given); B, R, P_a in state units squared
        #   (YAML units: dimensionless); K, nu, nu_max [1]; analysis is a REPLACE
        #   UpdateItem with per_timestep=True; diagnostics are detached REPLACE items.
        # Domain/constraints: both variables required finite tensors; y.numel must equal
        #   x.numel and is reshaped to x.shape; B > 0 and R > 0; nu finite and
        #   nu <= nu_max elementwise else ValueError with no partial update.
        # Sign convention: d = y - x is observation minus model; d > 0 raises the
        #   analysis toward the observation; analysis_increment = x_a - x = K d.
        x = self._require_tensor(state, self.state_variable)
        y = self._require_tensor(state, self.observation_variable)
        if y.shape != x.shape and y.numel() != x.numel():
            raise ValueError(
                f"Observation '{self.observation_variable}' shape {tuple(y.shape)} does not match "
                f"state '{self.state_variable}' shape {tuple(x.shape)}"
            )
        y = y.reshape(x.shape).to(dtype=x.dtype, device=x.device)
        background = self.to_tensor(self.get_parameter_value(self.BACKGROUND_ERROR_VARIANCE, required=True), dtype=x.dtype)
        observation = self.to_tensor(self.get_parameter_value(self.OBSERVATION_ERROR_VARIANCE, required=True), dtype=x.dtype)
        gate = float(self.get_parameter_value(self.MAX_NORMALIZED_INNOVATION, required=True))
        if not (float(background) > 0.0 and float(observation) > 0.0):
            raise ValueError("background and observation error variances must be positive")
        result = self.analysis(x, self._operator.forward({self.state_variable: y}).reshape(x.shape), background, observation)
        normalized = result["normalized_innovation"]
        if not torch.isfinite(normalized).all():
            raise ValueError("observation feedback innovation is not finite")
        if bool((normalized > gate).any()):
            raise ValueError(
                "observation feedback refused: normalized innovation "
                f"{float(normalized.max()):.4g} exceeds the gate {gate:.4g}; no partial update is proposed"
            )
        analysis_item = self.wrap_update(result["analysis"], mode=UpdateMode.REPLACE, per_timestep=True)
        if self.state_units is not None:
            analysis_item.unit = str(self.state_units)
        outputs: Dict[str, Any] = {self.state_variable: analysis_item}
        diagnostics = {
            "innovation": result["innovation"],
            "normalized_innovation": normalized,
            "kalman_gain": result["gain"],
            "analysis_increment": result["analysis"] - x,
            "analysis_error_variance": result["analysis_variance"],
        }
        for key, value in diagnostics.items():
            outputs[self.DIAG + key] = self.wrap_update(value.detach(), mode=UpdateMode.REPLACE)
        return outputs

    @staticmethod
    def _require_tensor(state: StateDict, key: str) -> Tensor:
        if key not in state:
            raise KeyError(f"Observation feedback requires state variable '{key}'")
        value = state[key]
        if not isinstance(value, Tensor):
            raise TypeError(f"State variable '{key}' must be a tensor")
        if not torch.isfinite(value).all():
            raise ValueError(f"State variable '{key}' must be finite")
        return value


__all__ = ["ObservationStateFeedbackFunctor"]
