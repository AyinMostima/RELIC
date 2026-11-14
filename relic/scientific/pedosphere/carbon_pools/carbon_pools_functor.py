"""StateManager-backed observation of pedosphere carbon pool stocks.

The functor is state neutral. State initialization belongs to the simulator
initial-state contract; this observer only exposes a diagnostic pool vector and
its carbon total.
"""

from typing import Any, Dict, List

import torch
from torch import Tensor

from relic.core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from relic.core.functor import BiologicalProcessFunctor
from relic.core.abstractions import StateDict, FluxDict
from relic.scientific.pedosphere.api_contract import PedosphereContractMixin


class CarbonPoolsFunctor(PedosphereContractMixin, BiologicalProcessFunctor):
    """
    Pedosphere carbon pool observer over the complete runtime state basis.

    Mathematical Expression:
        C_total = sum(C_i) where C_i are pool stocks [g C m-2]

    References:
        - Parton, W.J. et al. (1987) Soil carbon pools (CENTURY)
        - Coleman, K. & Jenkinson, D.S. (1996) RothC transfer structure
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'

    def __init__(
        self,
        name: str = "carbon_pools_observer",
        config: Dict[str, Any] | None = None,
        device: torch.device | str | int | None = None,
    ) -> None:
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Observes carbon pool stocks without writing prognostic state",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.POSTPROCESSING,
            mathematical_expression="C_total = sum_i C_i",
            references=[
                "Parton, W.J. et al. (1987). Soil carbon pools (CENTURY model).",
                "Coleman, K. & Jenkinson, D.S. (1996). RothC soil carbon model.",
            ],
            parameters={},
            input_variables=[
                "metabolic_litter_c",
                "structural_litter_c",
                "fast_som_c",
                "slow_som_c",
                "passive_som_c",
            ],
            output_variables=[
                "diagnostics.carbon_pools_snapshot",
                "diagnostics.total_carbon_stock",
            ],
            conservation_laws=[],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.DIAGNOSTIC,
            temporal_scale=TemporalScale.SECOND,
        )
        super().__init__(name=name, config=config or {}, metadata=metadata, device=device)
        # This observation operator does not consume the generic response
        # parameters injected by the biological bridge.
        self.metadata.parameters.clear()
        self._pool_keys: List[str] = [
            "metabolic_litter_c",
            "structural_litter_c",
            "fast_som_c",
            "slow_som_c",
            "passive_som_c",
        ]

    def forward(
        # References: Parton et al. (1987), CENTURY; Coleman and Jenkinson (1996), RothC.
        # Mathematical expression: C_total = sum_i C_i over the declared pool basis.
        # Parameter meanings: C_i are the StateManager-owned soil carbon pools; C_total the
        #   observed stock.
        # Units: C_i and C_total [g C m-2].
        # Domain/constraints: observation only; the complete pool basis must be supplied; the
        #   sum runs over the pool axis only and yields one stock per grid cell; it is not an
        #   area-integrated regional mass.
        # Sign convention: not applicable (non-negative stocks).
        self,
        state: StateDict,
        time: float,
        dt: Tensor | float,
        **kwargs: Any,
    ) -> FluxDict:
        """
        Observe a complete carbon pool basis supplied by StateManager.

        Returns
        -------
        FluxDict
            Diagnostic pool vector and total carbon stock (g C m-2).
        """
        del time, dt, kwargs
        missing_pool_keys = [key for key in self._pool_keys if key not in state]
        if missing_pool_keys:
            raise ValueError(
                "CarbonPoolsFunctor requires a complete canonical pool basis; "
                f"missing state variables: {missing_pool_keys}"
            )

        pools: Dict[str, Tensor] = {}
        for key in self._pool_keys:
            value = state[key]
            if not isinstance(value, Tensor):
                raise TypeError(f"State variable '{key}' must be a torch.Tensor")
            if not torch.is_floating_point(value):
                raise TypeError(f"State variable '{key}' must use a floating dtype")
            value = value.to(device=self.device)
            if not torch.all(torch.isfinite(value)):
                raise ValueError(f"State variable '{key}' must be finite")
            if torch.any(value < 0):
                raise ValueError(f"State variable '{key}' must be non-negative")
            pools[key] = value

        pool_shapes = {tuple(value.shape) for value in pools.values()}
        if len(pool_shapes) != 1:
            raise ValueError("Canonical carbon pool state variables must have the same shape")

        # References: Parton et al. (1987), CENTURY; Coleman and Jenkinson (1996), RothC.
        # Mathematical expression: z_h = O_h x_h with z_h[i, c] = C_i(c), and
        #   C_total(c) = sum_i z_h[i, c] (sum over the pool axis i only).
        # Parameter meanings: C_i(c) is the canonical stock of pool i in grid cell c; O_h
        #   stacks the declared pools in canonical order without advancing any state;
        #   C_total(c) is the total litter and soil organic carbon stock of cell c.
        # Units: C_i(c), z_h and C_total(c) [g C m-2].
        # Domain/constraints: complete, finite, non-negative pool basis of one common shape;
        #   the pool axis is the stacking axis (dim 0). Areal stocks of different cells are
        #   never added, because a cross-cell total needs the cell areas.
        # Sign convention: not applicable (non-negative stocks; no flux is formed).
        pool_vector = torch.stack([pools[key] for key in self._pool_keys])
        total_carbon = pool_vector.sum(dim=0)

        return {
            "diagnostics.carbon_pools_snapshot": pool_vector,
            "diagnostics.total_carbon_stock": total_carbon,
        }


__all__ = ["CarbonPoolsFunctor"]
