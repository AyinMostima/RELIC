"""
Infiltration Bridge Functor for Tutorial Case
===============================================

A lightweight bridge that copies ``infiltration_flux`` (output by
SurfaceRunoff) into ``precipitation_flux`` so that BucketSoilMoisture
can consume the infiltrated fraction without modification to the core
RELIC functor.

References
----------
- Design pattern: Adapter / Bridge (GoF)
- Physical basis: infiltration_flux = precipitation_flux - surface_runoff_flux
  (mass conservation, SCS 1972)

Mathematical expression
-----------------------
precipitation_flux_out = infiltration_flux   (simple key remapping)

Units: kg m-2 s-1 (passthrough, no conversion)
"""

from typing import Dict, Optional

import torch
from torch import Tensor

from relic.core.functor import (
    DependencyDeclaration,
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    TransportMechanismFunctor,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from relic.core.state import UpdateMode
from relic.core.abstractions import FluxDict, StateDict


class InfiltrationBridge(TransportMechanismFunctor):
    """Bridge ``infiltration_flux`` -> ``precipitation_flux`` for Bucket.

    SurfaceRunoff writes *infiltration_flux* (the portion of precipitation
    that actually enters the soil).  BucketSoilMoisture reads
    *precipitation_flux*.  This functor simply remaps the key so the two
    can work together without modifying core RELIC code.

    References
    ----------
    - SCS (1972) National Engineering Handbook, Section 4: Hydrology.
    - Horton, R.E. (1933). Trans. AGU 14(1), 446-460.

    Mathematical expression
    -----------------------
    precipitation_flux := infiltration_flux  (identity mapping)
    """

    TRANSPORT_TYPE = "advection"

    def __init__(
        self,
        name: str = "infiltration_bridge",
        config: Optional[Dict] = None,
    ) -> None:
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description=(
                "Bridge infiltration_flux to precipitation_flux "
                "for BucketSoilMoisture consumption"
            ),
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="precipitation_flux := infiltration_flux",
            references=[
                "SCS (1972) National Engineering Handbook, Section 4",
            ],
            parameters={},
            input_variables=["infiltration_flux", "precipitation_flux"],
            output_variables=["precipitation_flux"],
            conservation_laws=["mass"],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.SECOND,
            coupling_interfaces=["soil_moisture", "precipitation_flux"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
        )
        super().__init__(name=name, config=config or {}, metadata=metadata)
        self.optional_input_variables = {
            "infiltration_flux",
            "precipitation_flux",
        }
        # Transport mechanism required attributes
        self.peclet_number = None
        self.velocity_field = None

    def get_dependencies(self) -> DependencyDeclaration:
        """Depend on surface_runoff which produces infiltration_flux."""
        return self._merge_declared_dependencies(
            {"required": ["surface_runoff"], "optional": []}
        )

    def forward(
        self,
        state: StateDict,
        time: float,
        dt: float,
        **kwargs,
    ) -> FluxDict:
        """Remap infiltration_flux -> precipitation_flux.

        References
        ----------
        - SCS (1972): infiltration = precipitation - runoff (mass balance).

        Mathematical expression
        -----------------------
        precipitation_flux_out = infiltration_flux

        Parameters
        ----------
        state : StateDict
            Must contain ``infiltration_flux`` (kg m-2 s-1).
        time : float
            Current simulation time [s].
        dt : float
            Time step [s].

        Returns
        -------
        FluxDict
            ``precipitation_flux`` set to the infiltration value.
        """
        # Read infiltration_flux written by SurfaceRunoff
        infiltration = state.get("infiltration_flux")
        if infiltration is not None:
            infiltration_t = self._ensure_tensor(infiltration)
        else:
            # Fallback: pass through raw precipitation (no runoff computed)
            precip = state.get("precipitation_flux")
            if precip is not None:
                infiltration_t = self._ensure_tensor(precip)
            else:
                infiltration_t = torch.zeros(
                    1, 1, device=self.device, dtype=torch.float64
                )

        return {
            "precipitation_flux": self.wrap_update(
                infiltration_t, mode=UpdateMode.REPLACE
            ),
        }

    def _compute_rate_core(
        self,
        state: StateDict,
        environment: Optional[Dict] = None,
        time: float = 0.0,
        dt: Optional[float] = None,
    ) -> FluxDict:
        """Unified rate interface (delegates to forward)."""
        if dt is None:
            dt = 3600.0
        if environment is not None:
            combined = {**state, **environment}
        else:
            combined = state
        return self(combined, time=time, dt=dt)

    def _compute_flux_core(self, source, sink) -> FluxDict:
        """Transport interface stub (not used at runtime)."""
        src_t = self._ensure_tensor(source)
        return {"precipitation_flux": src_t}
