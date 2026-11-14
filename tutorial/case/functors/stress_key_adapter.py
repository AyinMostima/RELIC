"""Adapter that maps stress factor outputs to Farquhar/MEGAN stress keys."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from relic.core.abstractions import FluxDict, StateDict
from relic.core.functor import (
    BiologicalProcessFunctor,
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    SpatialScale,
    TemporalScale,
    VariableKind,
)


class StressKeyAdapter(BiologicalProcessFunctor):
    """Map `*_factor` stress outputs to canonical stress variable names."""

    def __init__(self, name: str = "water_stress", config: Optional[Dict[str, Any]] = None) -> None:
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Clamp and map stress factor keys into downstream photosynthesis/BVOC inputs.",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "f_T = clip(temperature_stress_factor, fmin, fmax); "
                "f_O3 = clip(ozone_stress_factor, fmin, fmax); "
                "f_W = clip((theta - theta_wp)/(theta_fc - theta_wp), fmin, fmax)"
            ),
            references=[
                "Farquhar, G.D. et al. (1980). Planta 149, 78-90.",
                "Emberson, L.D. et al. (2000). Environmental Pollution 109, 403-413.",
                "Berry, J. and Bjorkman, O. (1980). Annual Review of Plant Physiology 31, 491-543.",
            ],
            parameters={
                "tutorial.case.stress_adapter.clamp_min": "Lower clamp bound for stress factors [dimensionless]",
                "tutorial.case.stress_adapter.clamp_max": "Upper clamp bound for stress factors [dimensionless]",
                "hydrosphere.soil_moisture.wilting_point": "Wilting point moisture [m3 m-3]",
                "hydrosphere.soil_moisture.field_capacity": "Field capacity moisture [m3 m-3]",
            },
            input_variables=[
                "temperature_stress_factor",
                "ozone_stress_factor",
                "soil_moisture",
            ],
            output_variables=[
                "temperature_stress",
                "ozone_stress",
                "water_stress",
                "water_stress_factor",
            ],
            conservation_laws=[],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.HOUR,
            spatial_scale=SpatialScale.PATCH,
            variable_kind=VariableKind.DIAGNOSTIC,
        )
        super().__init__(name=name, config=config or {}, metadata=metadata)

    def get_dependencies(self) -> list[str]:
        return self._merge_declared_dependencies(
            ["thermal_stress", "ozone_stress", "bucket_soil_moisture"]
        )

    def _read_scalar(self, key: str, namespace: Optional[str] = None) -> float:
        value = self.get_parameter_value(key, namespace=namespace, required=True)
        if isinstance(value, torch.Tensor):
            return float(value.detach().reshape(-1)[0].tolist())
        return float(value)

    def forward(self, state: StateDict, time: float, dt: float, **kwargs: Any) -> FluxDict:
        # References:
        # - Berry and Bjorkman (1980), temperature stress normalization.
        # - Emberson et al. (2000), ozone stress factor usage in photosynthesis limitation.
        # - Manabe (1969), soil-moisture based water stress proxy.
        # Mathematical expression:
        # - f_T = clip(f_T_raw, fmin, fmax)
        # - f_O3 = clip(f_O3_raw, fmin, fmax)
        # - f_W = clip((theta - theta_wp)/(theta_fc - theta_wp), fmin, fmax)
        # Parameter meanings:
        # - theta soil_moisture [m3 m-3]
        # - theta_wp wilting_point [m3 m-3]
        # - theta_fc field_capacity [m3 m-3]
        # - fmin/fmax clamp bounds [dimensionless]
        clamp_min = self._read_scalar("clamp_min", namespace="tutorial.case.stress_adapter")
        clamp_max = self._read_scalar("clamp_max", namespace="tutorial.case.stress_adapter")

        temp_raw = state.get("temperature_stress_factor", 1.0)
        ozone_raw = state.get("ozone_stress_factor", 1.0)
        temp_factor = torch.clamp(self._ensure_tensor(temp_raw), min=clamp_min, max=clamp_max)
        ozone_factor = torch.clamp(self._ensure_tensor(ozone_raw), min=clamp_min, max=clamp_max)

        soil_moisture = self._ensure_tensor(state.get("soil_moisture", 0.3))
        wilting_point = self._read_scalar("wilting_point", namespace="hydrosphere.soil_moisture")
        field_capacity = self._read_scalar("field_capacity", namespace="hydrosphere.soil_moisture")
        eps = self._read_scalar("epsilon_divide", namespace="validation.numerical")
        water_factor = (soil_moisture - wilting_point) / max(field_capacity - wilting_point, eps)
        water_factor = torch.clamp(water_factor, min=clamp_min, max=clamp_max)

        return {
            "temperature_stress": temp_factor,
            "ozone_stress": ozone_factor,
            "water_stress": water_factor,
            "water_stress_factor": water_factor,
        }
