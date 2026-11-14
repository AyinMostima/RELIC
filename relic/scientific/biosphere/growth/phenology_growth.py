"""Growth-activity diagnostic derived from one selected phenology owner.

References:
    - Richardson et al. (2013). Climate change, phenology, and phenological
      control of vegetation feedbacks to the climate system.
    - Gao et al. (2025). pnetr: an R package for the PnET model family.

Mathematical expression:
    growth_active = phenological_factor > growth_active_threshold.

Parameter meanings:
    - phenological_factor: dimensionless control produced by exactly one
      compile-time-selected Static, GDD, GSI, or Crop phenology owner.
    - growth_active_threshold: ParameterManager-backed diagnostic threshold.

Units:
    The input, threshold, and Boolean-valued output are dimensionless.

Open system route:
    This phenomenological diagnostic does not advance LAI, carbon, nitrogen,
    water, energy, or hidden phenology history. The exact threshold is a
    calibratable observation rule, not a universal phenology law.
"""

import math
from typing import Any, Dict, Optional
import torch
from torch import Tensor

from relic.core.functor import (
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from relic.core.functor import BiologicalProcessFunctor
from relic.core.state import UpdateItem, UpdateMode


class PhenologyGrowth(BiologicalProcessFunctor):
    """Observe growth activity from one canonical phenology control port."""

    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'
    CANONICAL_NAME = "phenology_growth"
    DIAGNOSTIC_KEY = "diagnostics.phenology.growth_active"
    CANONICAL_PHENOLOGY_OWNERS = frozenset(
        {
            "static_phenology",
            "gdd_phenology",
            "gsi_phenology",
            "crop_phenology",
        }
    )
    
    def __init__(
        self,
        name: str = "phenology_growth",
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        metadata: Optional[FunctorMetadata] = None,
        phenology_model_owner: Optional[str] = None,
    ):
        if name != self.CANONICAL_NAME:
            raise ValueError(
                f"PhenologyGrowth requires canonical name '{self.CANONICAL_NAME}'."
            )
        if metadata is not None:
            raise ValueError("PhenologyGrowth custom metadata is not supported.")
        if phenology_model_owner not in self.CANONICAL_PHENOLOGY_OWNERS:
            allowed = ", ".join(sorted(self.CANONICAL_PHENOLOGY_OWNERS))
            raise ValueError(
                "phenology_model_owner must explicitly select one of: " + allowed
            )
        self.phenology_model_owner = str(phenology_model_owner)

        metadata = FunctorMetadata(
                name=name,
                version="3.0.0",
                description=(
                    "ParameterManager-backed phenomenological growth-activity "
                    "diagnostic from one selected canonical phenology owner."
                ),
                category=ProcessCategory.BIOLOGICAL_PROCESS,
                execution_phase=ExecutionPhase.MAIN_PROCESSES,
                mathematical_expression=(
                    "Observation only: diagnostics.phenology.growth_active = "
                    "phenological_factor > growth_active_threshold; no state "
                    "advancement and no strict IPHS claim."
                ),
                references=[
                    "Richardson et al. (2013). Climate change, phenology, and phenological control. Agric For Meteorol 169:156-173",
                    "Gao et al. (2025). pnetr: an R package for the PnET family of forest ecosystem models."
                ],
                parameters={
                    'growth_active_threshold': {
                        'description': 'Calibratable threshold for growth activity diagnosis',
                        'units': 'dimensionless',
                        'namespace': 'biosphere.growth.allocation'
                    }
                },
                input_variables=['phenological_factor'],
                output_variables=[self.DIAGNOSTIC_KEY],
                conservation_laws=[],
                spatial_dimensions=0,
                temporal_scale=TemporalScale.DAY,
                coupling_interfaces=['biosphere.phenology', 'biosphere.growth'],
                spatial_scale=SpatialScale.PATCH,
                variable_kind=VariableKind.DIAGNOSTIC,
            )
        
        # Initialize parent class
        super().__init__(
            name=name,
            config=config or {},
            metadata=metadata,
            device=device
        )
        
        self.growth_active_threshold = float(self.get_parameter_value(
            "growth_active_threshold",
            namespace="biosphere.growth.allocation", required=True))
        if not math.isfinite(self.growth_active_threshold) or not (
            0.0 <= self.growth_active_threshold <= 1.0
        ):
            raise ValueError(
                "growth_active_threshold must be finite and between 0 and 1"
            )
            
    def get_dependencies(self) -> list:
        """Return the single compile-time-selected phenology owner."""
        return self._merge_declared_dependencies([self.phenology_model_owner])
        
    def get_provides(self) -> list:
        return list(self.metadata.output_variables)

    def _required_factor(self, state: Dict[str, Tensor]) -> Tensor:
        """Return the finite canonical phenology factor without conversion."""
        key = "phenological_factor"
        if key not in state:
            raise KeyError(f"Missing required state '{key}'.")
        value = state[key]
        if not isinstance(value, Tensor):
            raise TypeError("phenological_factor must be a torch.Tensor")
        if not value.is_floating_point():
            raise TypeError("phenological_factor must be a floating torch.Tensor")
        factor = value.to(device=self.device)
        if not bool(torch.isfinite(factor).all()):
            raise ValueError("phenological_factor must be finite")
        if bool(((factor < 0.0) | (factor > 1.0)).any()):
            raise ValueError("phenological_factor must be between 0 and 1")
        return factor

    @staticmethod
    def _validate_runtime_scalar(value: Any, name: str, *, positive: bool) -> None:
        """Require one finite scalar runtime control."""
        tensor = torch.as_tensor(value)
        if tensor.numel() != 1 or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be a finite scalar")
        if positive and not bool(tensor > 0):
            raise ValueError(f"{name} must be positive")

    def forward(self, state, time, dt, **kwargs) -> Dict[str, UpdateItem]:
        """Return the selected-owner growth-activity diagnostic."""
        del kwargs
        self._validate_runtime_scalar(time, "time", positive=False)
        self._validate_runtime_scalar(dt, "dt", positive=True)
        factor = self._required_factor(state)

        # References: Richardson et al. (2013); Gao et al. (2025).
        # Mathematical expression: growth_active = phi > phi_active.
        # Parameter meanings: phi is the selected owner's phenological_factor;
        # phi_active is the ParameterManager-backed diagnostic threshold.
        # Units: phi, phi_active, and the Boolean observation are dimensionless.
        # Open system route: diagnostic only; no conserved state is written.
        threshold = torch.as_tensor(
            self.growth_active_threshold,
            dtype=factor.dtype,
            device=factor.device,
        )
        growth_active = factor > threshold

        return {
            self.DIAGNOSTIC_KEY: UpdateItem(
                value=growth_active,
                mode=UpdateMode.REPLACE,
                unit="dimensionless",
                per_timestep=False,
                source_functor=self.name,
            )
        }


__all__ = ['PhenologyGrowth']


