"""
Water Stress Response Model for RELIC

Computes one bounded plant-available-water control from the canonical soil
water potential port. The process is an instantaneous biological diagnostic;
it does not advance soil water storage or duplicate root-uptake physics.

References
----------
- Boyer, J.S. (1982). Plant productivity and environment. Science 218, 443-448
- Hsiao, T.C. (1973). Plant responses to water stress. Annual Review of Plant 
  Physiology 24, 519-570
- Jones, H.G. (2007). Monitoring plant and soil water status. Journal of 
  Experimental Botany 58, 155-173

Mathematical Framework
----------------------
f_water = clamp((psi_soil - psi_wilt) / (psi_fc - psi_wilt), 0, 1)

Units
-----
- Water potential (psi): MPa
- Stress factors: dimensionless [0-1]
"""

from typing import Dict, Optional, Any
import torch
from torch import Tensor

from relic.core.functor import BiologicalProcessFunctor
from relic.core.abstractions import StateDict, FluxDict
from relic.core.state import UpdateItem, UpdateMode
from relic.core.functor import (
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
    SpatialScale,
    VariableKind,
    TemporalScale,
)


class WaterStress(BiologicalProcessFunctor):
    """
    Water Stress Response Model for RELIC.
    
    Implements a bounded empirical response to soil water potential. The
    result is a downstream control/observation, not a prognostic water state.
    
    Mathematical Framework
    ----------------------
    Linear water stress: f_water = (psi - psi_wilt) / (psi_fc - psi_wilt)
    where:
    - psi: current water potential (MPa)
    - psi_wilt: wilting point potential (MPa)  
    - psi_fc: field capacity potential (MPa)
    
    References
    ----------
    - Boyer, J.S. (1982). Plant productivity and environment. Science 218, 443-448
    - Hsiao, T.C. (1973). Plant responses to water stress. Annual Review of Plant 
      Physiology 24, 519-570
    - Jones, H.G. (2007). Monitoring plant and soil water status. Journal of 
      Experimental Botany 58, 155-173
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'
    
    def __init__(
        self,
        name: str = "water_stress",
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None
    ):
        # Initialize metadata for proper Functor integration
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description='Water stress response model based on water potential',
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=r'f_{water} = (psi - psi_wilt) / (psi_{fc} - psi_{wilt})',
            references=[
                'Boyer, J.S. (1982). Plant productivity and environment. Science 218, 443-448',
                'Hsiao, T.C. (1973). Plant responses to water stress. Annual Review of Plant Physiology 24, 519-570',
                'Jones, H.G. (2007). Monitoring plant and soil water status. Journal of Experimental Botany 58, 155-173'
            ],
            parameters={
                'wilting_point_potential': {
                    'description': 'Water potential at permanent wilting point',
                    'units': 'MPa',
                    'namespace': 'biosphere.stress.water'
                },
                'field_capacity_potential': {
                    'description': 'Water potential at field capacity',
                    'units': 'MPa',
                    'namespace': 'biosphere.stress.water'
                },
                'global.factor_min': {
                    'description': 'Lower bound for stress factors',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.stress'
                },
                'global.factor_max': {
                    'description': 'Upper bound for stress factors',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.stress'
                },
            },
            input_variables=['soil_water_potential'],
            output_variables=['water_stress_factor'],
            conservation_laws=[],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.HOUR,
            spatial_scale=SpatialScale.PATCH,
            variable_kind=VariableKind.DIAGNOSTIC,
        )
        
        # Initialize FunctorBase with proper metadata
        super().__init__(name, config or {}, metadata=metadata, device=device)
        
        # Unified namespace for all water stress parameters
        self.namespace = "biosphere.stress.water"
            
    def get_provides(self) -> list:
        """Define output variables provided by this Functor."""
        return ['water_stress_factor']
        
    def forward(self, state: StateDict, time: float, dt: float, **kwargs) -> FluxDict:
        """
        Main forward pass for water stress calculation.

        Parameters
        ----------
        state : StateDict
            Current system state
        time : float
            Current simulation time
        dt : float
            Time step
        **kwargs
            Additional arguments

        Returns
        -------
        FluxDict
            Dictionary containing all stress-related outputs
        """
        # References: Boyer (1982); Hsiao (1973); Cox et al. (1998).
        # Mathematical expression:
        # f_water = clamp((psi_soil - psi_wilt) / (psi_fc - psi_wilt), f_min, f_max).
        # Parameter meanings:
        # psi_soil, psi_wilt, psi_fc [MPa]; f_min, f_max, f_water [1].
        # Open/closed system route:
        # This read-only observation has no material source, sink, or state advance.
        if 'soil_water_potential' not in state:
            raise KeyError("soil_water_potential is required")

        water_potential = state['soil_water_potential']
        if not isinstance(water_potential, Tensor):
            raise TypeError("soil_water_potential must be a torch.Tensor")
        if not torch.is_floating_point(water_potential):
            raise TypeError("soil_water_potential must be a floating torch.Tensor")
        if not bool(torch.isfinite(water_potential).all().item()):
            raise ValueError("soil_water_potential must contain only finite values")
        if bool((water_potential > 0).any().item()):
            raise ValueError("soil_water_potential must be non-positive in MPa")

        wilting_point = self.get_parameter_value(
            "wilting_point_potential", namespace=self.namespace, required=True
        )
        field_capacity = self.get_parameter_value(
            "field_capacity_potential", namespace=self.namespace, required=True
        )
        stress_min = self.get_parameter_value(
            "global.factor_min", namespace="biosphere.stress", required=True
        )
        stress_max = self.get_parameter_value(
            "global.factor_max", namespace="biosphere.stress", required=True
        )

        target = {"device": water_potential.device, "dtype": water_potential.dtype}
        wilting_point_tensor = torch.as_tensor(wilting_point, **target)
        field_capacity_tensor = torch.as_tensor(field_capacity, **target)
        stress_min_tensor = torch.as_tensor(stress_min, **target)
        stress_max_tensor = torch.as_tensor(stress_max, **target)

        if not bool((field_capacity_tensor > wilting_point_tensor).item()):
            raise ValueError(
                "field_capacity_potential must exceed wilting_point_potential"
            )
        if not bool(
            (
                (stress_min_tensor >= 0)
                & (stress_min_tensor < stress_max_tensor)
                & (stress_max_tensor <= 1)
            ).item()
        ):
            raise ValueError("stress factor bounds must satisfy 0 <= min < max <= 1")

        stress_factor = (water_potential - wilting_point_tensor) / (
            field_capacity_tensor - wilting_point_tensor
        )
        stress_factor = torch.clamp(
            stress_factor, min=stress_min_tensor, max=stress_max_tensor
        )

        return {
            'water_stress_factor': UpdateItem(
                value=stress_factor,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit='1',
                source_functor=self.name,
            )
        }


__all__ = ['WaterStress']

