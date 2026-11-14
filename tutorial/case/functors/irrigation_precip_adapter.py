"""Adapter that normalizes precipitation units and computes throughfall forcing."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from relic.core.abstractions import FluxDict, StateDict
from relic.core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    SpatialScale,
    TemporalScale,
    TransportMechanismFunctor,
    VariableKind,
)
from relic.core.state.types import UpdateMode
from relic.utils.physical_constants import get_physical_constant


class PrecipThroughfallAdapter(TransportMechanismFunctor):
    """Convert gross precipitation to canopy throughfall forcing."""

    TRANSPORT_TYPE = "advection"

    def __init__(self, name: str = "precip_throughfall_adapter", config: Optional[Dict[str, Any]] = None) -> None:
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Normalize precipitation units and map gross precipitation to canopy throughfall.",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="P_eff = U(P) * (1 - clamp(k_int * LAI, f_min, f_max))",
            references=[
                "Allen, R.G. et al. (1998). FAO Irrigation and Drainage Paper 56.",
                "Rodriguez-Iturbe, I. and Porporato, A. (2004). Ecohydrology.",
                "Shuttleworth, W.J. (1988). Evaporation from Amazonian Rainforest.",
                "Cuartas, L.A. et al. (2007). Journal of Hydrology 342, 77-90.",
            ],
            parameters={
                "tutorial.case.precip_adapter.input_unit_mode": "Input unit mode: kg_m2_s, mm_h, auto",
                "tutorial.case.precip_adapter.auto_detect_threshold": "Auto-mode threshold for mm h-1 detection",
                "case.conversions.mm_to_m": "Depth conversion from mm to m",
                "tutorial.case.throughfall.interception_coefficient": "Canopy interception coefficient per LAI",
                "tutorial.case.throughfall.min_interception_fraction": "Lower bound of interception fraction",
                "tutorial.case.throughfall.max_interception_fraction": "Upper bound of interception fraction",
            },
            input_variables=["precipitation_flux", "leaf_area_index"],
            output_variables=[
                "precipitation_flux_effective",
                "precipitation_flux_gross",
                "throughfall_fraction",
                "canopy_interception_fraction",
                "canopy_interception_flux",
            ],
            conservation_laws=["water"],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.HOUR,
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
        )
        super().__init__(name=name, config=config or {}, metadata=metadata)

    def _read_value(self, key: str, *, namespace: str, required: bool = True) -> Any:
        value = self.get_parameter_value(key, namespace=namespace, required=required)
        if isinstance(value, torch.Tensor):
            return value.detach().reshape(-1)[0].tolist()
        return value

    def _normalize_precipitation(self, precipitation: torch.Tensor) -> torch.Tensor:
        # References:
        # - Allen et al. (1998), forcing units in land-surface water balance.
        # Mathematical expression:
        # - P_kg = P_mm_h * mm_to_m * rho_w / seconds_per_hour
        # Parameter meanings:
        # - mm_to_m from case.conversions.mm_to_m
        # - rho_w from physical constants water_density
        # - seconds_per_hour from physical constants time conversion
        mode_raw = self._read_value(
            "input_unit_mode",
            namespace="tutorial.case.precip_adapter",
            required=False,
        )
        mode = str(mode_raw).strip().lower() if mode_raw is not None else "kg_m2_s"
        mm_to_m = float(
            self._read_value(
                "case.conversions.mm_to_m",
                namespace="",
                required=True,
            )
        )
        water_density = float(get_physical_constant("water_density"))
        seconds_per_hour = float(get_physical_constant("seconds_per_hour"))
        conversion_factor = mm_to_m * water_density / seconds_per_hour

        if mode == "kg_m2_s":
            return precipitation
        if mode == "mm_h":
            factor_t = torch.as_tensor(conversion_factor, dtype=precipitation.dtype, device=precipitation.device)
            return precipitation * factor_t

        auto_threshold = float(
            self._read_value(
                "auto_detect_threshold",
                namespace="tutorial.case.precip_adapter",
                required=False,
            )
            or 0.01
        )
        abs_max = float(torch.max(torch.abs(precipitation.detach())).cpu().item())
        is_mmh = abs_max > auto_threshold
        if is_mmh:
            factor_t = torch.as_tensor(conversion_factor, dtype=precipitation.dtype, device=precipitation.device)
            return precipitation * factor_t
        return precipitation

    def forward(self, state: StateDict, time: float, dt: float, **kwargs: Any) -> FluxDict:
        # References:
        # - Allen et al. (1998), precipitation forcing unit consistency.
        # - Shuttleworth (1988), Amazon canopy interception and throughfall partitioning.
        # - Cuartas et al. (2007), Tapajos rainfall interception magnitude.
        # Mathematical expression:
        # - P_gross = U(P)
        # - f_int = clamp(k_int * LAI, f_min, f_max)
        # - P_eff = P_gross * (1 - f_int)
        # Parameter meanings:
        # - P precipitation_flux forcing input [kg m-2 s-1 or mm h-1]
        # - k_int interception coefficient [LAI-1]
        # - LAI leaf area index [m2 m-2]
        # - f_min/f_max bounds for interception fraction
        precipitation = state.get("precipitation_flux")
        if precipitation is None:
            precipitation = torch.zeros(1, 1, dtype=torch.float32, device=self.device)
        precipitation_t = self._ensure_tensor(precipitation)
        gross_precip = torch.clamp(self._normalize_precipitation(precipitation_t), min=0.0)

        lai_raw = state.get("leaf_area_index")
        if lai_raw is None:
            lai_default = float(
                self._read_value(
                    "case.biosphere.leaf_area_index",
                    namespace="",
                    required=True,
                )
            )
            lai = torch.full_like(gross_precip, fill_value=lai_default)
        else:
            lai = self._ensure_tensor(lai_raw).to(device=gross_precip.device, dtype=gross_precip.dtype)

        k_interception = float(
            self._read_value(
                "interception_coefficient",
                namespace="tutorial.case.throughfall",
                required=True,
            )
        )
        min_frac = float(
            self._read_value(
                "min_interception_fraction",
                namespace="tutorial.case.throughfall",
                required=True,
            )
        )
        max_frac = float(
            self._read_value(
                "max_interception_fraction",
                namespace="tutorial.case.throughfall",
                required=True,
            )
        )
        k_t = torch.as_tensor(k_interception, dtype=gross_precip.dtype, device=gross_precip.device)
        min_t = torch.as_tensor(min_frac, dtype=gross_precip.dtype, device=gross_precip.device)
        max_t = torch.as_tensor(max_frac, dtype=gross_precip.dtype, device=gross_precip.device)
        interception_fraction = torch.clamp(lai * k_t, min=min_t, max=max_t)
        throughfall_fraction = torch.clamp(1.0 - interception_fraction, min=0.0, max=1.0)
        effective = gross_precip * throughfall_fraction
        interception_flux = gross_precip - effective
        return {
            "precipitation_flux_effective": self.wrap_update(effective, mode=UpdateMode.REPLACE),
            "precipitation_flux_gross": self.wrap_update(gross_precip, mode=UpdateMode.REPLACE),
            "throughfall_fraction": self.wrap_update(throughfall_fraction, mode=UpdateMode.REPLACE),
            "canopy_interception_fraction": self.wrap_update(interception_fraction, mode=UpdateMode.REPLACE),
            "canopy_interception_flux": self.wrap_update(interception_flux, mode=UpdateMode.REPLACE),
        }


# Backward compatibility alias for existing imports.
IrrigationPrecipAdapter = PrecipThroughfallAdapter
