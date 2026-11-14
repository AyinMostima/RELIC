"""One-step lag stomatal conductance proxy for ozone uptake coupling."""

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
from relic.core.state import UpdateItem, UpdateMode
from relic.utils.unified_response_functions import beer_lambert_effective_leaf_area


class StomatalProxyFunctor(BiologicalProcessFunctor):
    """Recover representative leaf assimilation before evaluating the empirical proxy."""

    def __init__(self, name: str = "stomatal_proxy", config: Optional[Dict[str, Any]] = None) -> None:
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Empirical leaf conductance proxy with explicit Farquhar canopy-support inversion.",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=("L_eff=(1-exp(-k*LAI))/k; A_leaf=A_ground/L_eff; "
                                     "g_s=clip(g0+m*max(A_leaf,0),gmin,gmax)"),
            references=[
                "Ball, J.T. et al. (1987). Progress in Photosynthesis Research.",
                "Emberson, L.D. et al. (2000). Environmental Pollution 109, 403-413.",
                "de Pury and Farquhar (1997). Plant, Cell and Environment 20, 537-557.",
            ],
            parameters={
                "tutorial.case.stomatal_proxy.g0": "Baseline stomatal conductance [mol m-2 s-1]",
                "tutorial.case.stomatal_proxy.slope": "Slope linking net photosynthesis to g_s [mol m-2 s-1 per umol m-2 s-1]",
                "tutorial.case.stomatal_proxy.min": "Lower bound for stomatal conductance [mol m-2 s-1]",
                "tutorial.case.stomatal_proxy.max": "Upper bound for stomatal conductance [mol m-2 s-1]",
                "biosphere.photosynthesis.canopy_light_extinction": "Farquhar Beer-Lambert extinction coefficient [dimensionless]",
            },
            input_variables=["net_photosynthesis", "leaf_area_index"],
            output_variables=["stomatal_conductance"],
            conservation_laws=[],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.HOUR,
            spatial_scale=SpatialScale.PATCH,
            variable_kind=VariableKind.DIAGNOSTIC,
        )
        super().__init__(name=name, config=config or {}, metadata=metadata)

    def get_dependencies(self) -> list[str]:
        return self._merge_declared_dependencies(["farquhar_photosynthesis"])

    def forward(self, state: StateDict, time: float, dt: float, **kwargs: Any) -> FluxDict:
        # References:
        # - Ball et al. (1987), assimilation-conductance empirical linkage.
        # - Emberson et al. (2000), ozone uptake dependence on stomatal conductance.
        # - Monsi and Saeki (1953); de Pury and Farquhar (1997), Farquhar canopy support.
        # Mathematical expression:
        # - L_eff = (1 - exp(-k * LAI)) / k; A_leaf = A_ground / L_eff.
        # - g_s = clip(g0 + m * max(A_leaf, 0), g_min, g_max).
        # Parameter meanings:
        # - A_ground: public net_photosynthesis; A_leaf: representative leaf assimilation.
        # - LAI: total leaf area per ground area; k: shared Farquhar extinction coefficient.
        # - L_eff: effective illuminated leaf area per ground area, not total LAI.
        # - g0: baseline conductance; m: slope linking A_leaf to g_s.
        # - g_min, g_max: admissible conductance range (tutorial.case.stomatal_proxy.*)
        # Units:
        # - A_ground [umol CO2 m-2 ground s-1]; A_leaf [umol CO2 m-2 leaf s-1].
        # - LAI,L_eff [m2 leaf m-2 ground]; k [1]; g0,g_min,g_max,g_s [mol m-2 leaf s-1].
        # - m [mol H2O per umol CO2], with both rates on leaf area.
        # Domain/constraints:
        # - Finite A_ground and LAI > 0; zero LAI has no identifiable representative leaf.
        # - L_eff must remain positive in the input precision; no epsilon or default is substituted.
        # - Negative A_leaf gives the clipped g0; this is the declared proxy, not full Medlyn.
        # Sign convention:
        # - Positive assimilation increases conductance; the output is an instantaneous REPLACE.
        # - A one-step ozone lag depends on the caller's execution order, not on this kernel.
        net_photo = self._ensure_tensor(state["net_photosynthesis"])
        if not bool(torch.isfinite(net_photo).all()):
            raise ValueError("net_photosynthesis must be finite on ground-area support.")
        lai = self._ensure_tensor(state["leaf_area_index"], dtype=net_photo.dtype)
        if not bool(torch.isfinite(lai).all()) or bool(torch.any(lai <= 0)):
            raise ValueError("leaf_area_index must be finite and positive for representative-leaf inversion.")
        extinction = self.get_parameter_value("biosphere.photosynthesis.canopy_light_extinction", required=True)
        effective_lai = beer_lambert_effective_leaf_area(lai, extinction)
        if bool(torch.any(effective_lai <= 0)):
            raise ValueError("leaf_area_index gives unresolved zero effective leaf area in the input precision.")
        leaf_photo = net_photo / effective_lai
        g0, slope, g_min, g_max = (
            torch.as_tensor(self.get_parameter_value(key, namespace="tutorial.case.stomatal_proxy", required=True),
                            dtype=net_photo.dtype, device=net_photo.device)
            for key in ("g0", "slope", "min", "max")
        )
        gs_target = torch.clamp(g0 + slope * torch.relu(leaf_photo), min=g_min, max=g_max)
        return {"stomatal_conductance": UpdateItem(
            value=gs_target, mode=UpdateMode.REPLACE, unit="mol m-2 s-1", per_timestep=False)}
