"""Scale coupling functor providing hybrid methane gating weights."""

from typing import Any, Dict, Optional

import torch
from torch import Tensor

from relic.core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    ScaleCouplingFunctor,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from relic.core.state import UpdateItem, UpdateMode
from relic.core.abstractions import StateDict
from relic.scientific.pedosphere.api_contract import PedosphereContractMixin


class MethaneScaleCouplingFunctor(PedosphereContractMixin, ScaleCouplingFunctor):
    """Compute the hybrid weight blending the iMAPLE and calibrated methane branches.

    Scientific references:
        - Yue et al. (2024) doi:10.5194/gmd-17-4621-2024 (iMAPLE endpoint branch)
        - The weighted gate and the calibrated Rh-based branch it favours are RELIC
          conventions; the calibrated branch is not the DNDC substrate model.

    Gating formulation (ASCII form):
        score = alpha_wetland * f_wet + alpha_wfps * WFPS + alpha_data * score_data
        w = clip(score - threshold, 0, 1)
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'

    def __init__(
        self,
        name: str = "pedosphere_methane_scale",
        device: Optional[torch.device] = None,
    ) -> None:
        parameter_descriptions = {
            "pedosphere.methane.scales.gating.alpha_wetland": "Weight for wetland fraction [-]",
            "pedosphere.methane.scales.gating.alpha_wfps": "Weight for water-filled pore space [-]",
            "pedosphere.methane.scales.gating.alpha_data_score": "Weight for data availability score [-]",
            "pedosphere.methane.scales.gating.threshold": "Gating threshold [-]",
        }

        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Hybrid methane gating weight based on wetland fraction, soil moisture, and data quality",
            category=ProcessCategory.SCALE_COUPLING,
            execution_phase=ExecutionPhase.COUPLING,
            mathematical_expression="w = clip(alpha_wetland * f_wet + alpha_wfps * WFPS + alpha_data * score_data - threshold, 0, 1)",
            references=[
                "Yue et al. (2024) doi:10.5194/gmd-17-4621-2024",
            ],
            parameters=parameter_descriptions,
            input_variables=[
                "land_surface.wetland_fraction",
                "water_filled_pore_space",
                "methane_data_score",
            ],
            output_variables=[
                "methane_gating_weight",
                "methane_gating_score",
            ],
            conservation_laws=[],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.DIAGNOSTIC,
            spatial_dimensions=0,
            temporal_scale=TemporalScale.SECOND,
        )
        # The gating weights the process-branch representations (iMAPLE and calibrated)
        # into one grid-cell production control (model-hierarchy coupling).
        super().__init__(
            name=name,
            metadata=metadata,
            device=device,
            source_scale="process",
            target_scale="grid_cell",
            coupling_type="hierarchical",
        )

        self.alpha_wetland = self.get_parameter_value(
            "gating.alpha_wetland",
            namespace="pedosphere.methane.scales", required=True)
        self.alpha_wfps = self.get_parameter_value(
            "gating.alpha_wfps",
            namespace="pedosphere.methane.scales", required=True)
        self.alpha_data = self.get_parameter_value(
            "gating.alpha_data_score",
            namespace="pedosphere.methane.scales", required=True)
        self.threshold = self.get_parameter_value(
            "gating.threshold",
            namespace="pedosphere.methane.scales", required=True)

    def forward(
        # References: Yue et al. (2024) motivates the iMAPLE branch; the calibrated
        #   branch and this weighted switching rule are RELIC calibration closures.
        # Mathematical expression: w=clip(alpha_wet*f_wet+alpha_wfps*WFPS
        #   +alpha_data*data_score-threshold,0,1).
        # Parameter meanings: alpha values weight wetland fraction, water-filled pore
        #   space and data suitability; threshold shifts the selected mixture.
        # Units: all inputs, weights, threshold and output [1].
        # Domain/constraints: Required finite tensor fractions in [0,1]; this gate
        #   selects a model mixture and performs no conservative spatial remapping.
        # Sign convention: Increasing w favors the calibrated branch; REPLACE output
        #   does not advance CH4.
        self,
        state: StateDict,
        time: float,
        dt: Tensor | float,
        **kwargs: Any,
    ) -> Dict[str, UpdateItem]:
        wetland = self._require_fraction(state, "land_surface.wetland_fraction")
        wfps = self._require_fraction(state, "water_filled_pore_space")
        data_score = self._require_fraction(state, "methane_data_score")

        # References: Yue et al. (2024) describes the iMAPLE endpoint; the other
        # endpoint and this weighted routing rule are calibrated RELIC
        # phenomenological conventions rather than a source equation.
        # Expression: score = a_wet*f_wet + a_wfps*WFPS + a_data*s_data;
        # w = clip(score - threshold, 0, 1).
        # Parameters: the three a terms and threshold are dimensionless
        # ParameterManager values; all three state inputs are fractions in [0,1].
        score = self.alpha_wetland * wetland + self.alpha_wfps * wfps + self.alpha_data * data_score
        gating_weight = torch.clamp(score - self.threshold, min=0.0, max=1.0)

        updates: Dict[str, UpdateItem] = {}
        updates["methane_gating_weight"] = UpdateItem(
            value=gating_weight,
            mode=UpdateMode.REPLACE,
            unit="1",
            per_timestep=False,
            source_functor=self.name,
        )
        updates["methane_gating_score"] = UpdateItem(
            value=score,
            mode=UpdateMode.REPLACE,
            unit="1",
            per_timestep=False,
            source_functor=self.name,
        )

        self._last_updates = updates
        return updates

    def _require_fraction(
        self,
        state: StateDict,
        key: str,
    ) -> Tensor:
        """Return one required finite dimensionless fraction."""
        if key not in state:
            raise KeyError(f"Required methane scale input '{key}' is missing.")
        value = state[key]
        if not isinstance(value, Tensor):
            raise TypeError(f"Methane scale input '{key}' must be a torch.Tensor.")
        tensor = value.to(self.device)
        if not bool(torch.all(torch.isfinite(tensor))):
            raise ValueError(
                f"Methane scale input '{key}' must contain only finite values."
            )
        if bool(torch.any((tensor < 0.0) | (tensor > 1.0))):
            raise ValueError(f"Methane scale input '{key}' must be within [0, 1].")
        return tensor


__all__ = ["MethaneScaleCouplingFunctor"]
