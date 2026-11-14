"""Diagnostic structural-growth observer over the canonical growth ports.

References:
    - Wright et al. (2004), the worldwide leaf economics spectrum.
    - Yin et al. (2021), LAI development from leaf mass and specific leaf area.
    - Feinberg (2019), chemical reaction network stoichiometry.
Mathematical expression:
    r_growth = SLA * g_leaf; r_loss = k_leaf * LAI;
    r_net = r_growth - r_loss, lowered as y_dot = N_h r_h.
Parameter meanings:
    g_leaf is the canonical leaf-carbon growth flux in kg C m-2 s-1;
    SLA is m2 kg-1 C; k_leaf is the existing s-1 turnover parameter.
Units:
    All tendencies are m2 m-2 s-1 and the step diagnostic is m2 m-2.
Open system route:
    Results are observations only; the selected phenology closure owns LAI.
"""

from typing import Dict, Optional, Union, Any
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
from relic.computation.operators.iphs import advance_reaction_network

class StructuralGrowth(BiologicalProcessFunctor):
    """Observe leaf-carbon-driven LAI tendencies without writing LAI state."""
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'

    def __init__(
        self,
        name: str = "structural_growth",
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ):
        # Create comprehensive metadata
        metadata = FunctorMetadata(
            name=name,
            version="2.0.0",
            description=(
                "Diagnostic LAI tendency observer over the explicit leaf-growth "
                "and phenology-owned LAI ports."
            ),
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "Canonical IPHS:\n"
                f"  owner: {name}\n"
                "  state: one diagnostic leaf-area observation cochain\n"
                "  dynamics: y_dot = N_h r_h; delta_y = y_dot * dt\n"
                "  structure: N_h = [1,-1] maps leaf-area growth and "
                "senescence channels to the signed observation\n"
                "  ownership: no leaf_area_index state advancement"
            ),
            references=[
                "Wright et al. (2004). The worldwide leaf economics spectrum. Nature 428:821-827.",
                "Yin et al. (2021). Combining physiological principles and mathematics to improve crop models.",
                "Feinberg (2019). Foundations of Chemical Reaction Network Theory.",
            ],
            parameters={
                'sla': {
                    'description': 'Specific leaf area',
                    'units': 'm2 kg-1 C',
                    'namespace': 'biosphere.phenology.morphology'
                },
                'leaf_turnover_rate': {
                    'description': 'Leaf turnover rate coefficient',
                    'units': 's-1',
                    'namespace': 'biosphere.phenology.turnover'
                },
            },
            input_variables=['growth_rate_leaf', 'leaf_area_index'],
            output_variables=[
                'diagnostics.structural_growth.gross_leaf_area_growth_tendency',
                'diagnostics.structural_growth.leaf_area_senescence_tendency',
                'diagnostics.structural_growth.net_leaf_area_tendency',
                'diagnostics.structural_growth.step_leaf_area_delta',
            ],
            conservation_laws=[],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.SECOND,  # CF-1.8 standard time unit
            coupling_interfaces=['biosphere.growth.biomass_allocation', 'biosphere.phenology'],
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
        
    def get_dependencies(self) -> list:
        """Return Functor dependencies (not state variables)."""
        return self._merge_declared_dependencies(["biomass_allocation"])
        
    def get_provides(self) -> list:
        return list(self.metadata.output_variables)
        
    def _ensure_tensor(self, value: Union[Tensor, float], dtype: torch.dtype = torch.float32) -> Tensor:
        if isinstance(value, Tensor):
            return value.to(device=self.device, dtype=dtype)
        return torch.as_tensor(float(value), dtype=dtype, device=self.device)

    def _compute_rate_core(self, state: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Compute diagnostic LAI growth and senescence tendencies.

        References
        ----------
        - Wright et al. (2004). The worldwide leaf economics spectrum.
        - Yin et al. (2021). LAI development from leaf mass and SLA.

        Mathematical Expression
        ------------------------
        r_growth = SLA * growth_rate_leaf
        r_loss = k_leaf * leaf_area_index
        r_net = r_growth - r_loss

        Parameter Meaning
        -----------------
        - growth_rate_leaf: leaf carbon-growth port from BiomassAllocation.
        - leaf_area_index: StateManager-owned leaf area index.
        - sla: specific leaf area linking carbon growth to leaf area growth.
        - leaf_turnover_rate: ParameterManager rate already expressed in s-1.

        Units
        -----
        growth_rate_leaf is kg C m-2 s-1, SLA is m2 kg-1 C,
        leaf_area_index is m2 m-2, and all returned tendencies are
        m2 m-2 s-1.

        Open System Route
        -----------------
        The rates are routed to observations. This functor does not write the
        phenology-owned leaf_area_index state.
        """
        if "growth_rate_leaf" not in state:
            raise KeyError("Missing required state 'growth_rate_leaf'.")
        if "leaf_area_index" not in state:
            raise KeyError("Missing required state 'leaf_area_index'.")

        # References: PyTorch type promotion rules; RELIC runtime-precision contract.
        # Mathematical expression: dtype_run = promote_types over the floating input
        #   tensors (growth_rate_leaf, leaf_area_index), else torch.get_default_dtype().
        # Parameter meanings: the two declared inputs own the runtime precision; SLA and
        #   the turnover rate are cast to it.
        # Units: none (dtype resolution).
        # Domain/constraints: float32 parameters are widened to a float64 runtime and a
        #   float64 input is never truncated; integer inputs do not lower the precision.
        # Sign convention: not applicable.
        runtime_dtype = None
        for raw in (state["growth_rate_leaf"], state["leaf_area_index"]):
            if isinstance(raw, Tensor) and raw.is_floating_point():
                runtime_dtype = raw.dtype if runtime_dtype is None else torch.promote_types(runtime_dtype, raw.dtype)
        if runtime_dtype is None:
            runtime_dtype = torch.get_default_dtype()
        growth_leaf = self._ensure_tensor(state["growth_rate_leaf"], dtype=runtime_dtype)
        leaf_area_index = self._ensure_tensor(state["leaf_area_index"], dtype=runtime_dtype)
        for name, value in (
            ("growth_rate_leaf", growth_leaf),
            ("leaf_area_index", leaf_area_index),
        ):
            if not torch.all(torch.isfinite(value)):
                raise ValueError(f"{name} must be finite")
            if torch.any(value < 0):
                raise ValueError(f"{name} must be non-negative")

        sla_tensor = self._ensure_tensor(
            self.get_parameter_value(
                "sla",
                namespace="biosphere.phenology.morphology",
                required=True,
            ),
            dtype=runtime_dtype,
        )
        senescence_rate_tensor = self._ensure_tensor(
            self.get_parameter_value(
                "leaf_turnover_rate",
                namespace="biosphere.phenology.turnover",
                required=True,
            ),
            dtype=runtime_dtype,
        )

        for name, value in (
            ("sla", sla_tensor),
            ("leaf_turnover_rate", senescence_rate_tensor),
        ):
            if not torch.all(torch.isfinite(value)):
                raise ValueError(f"{name} must be finite")
            if torch.any(value < 0):
                raise ValueError(f"{name} must be non-negative")

        # References: Wright et al. (2004); Yin et al. (2021).
        # Mathematical expression: r_growth = SLA * growth_rate_leaf;
        # r_loss = k_leaf * LAI; r_net = r_growth - r_loss.
        # Parameter meanings: SLA maps leaf carbon to leaf area and k_leaf is
        # the ParameterManager turnover rate already represented in s-1.
        # Units: kg C m-2 s-1 * m2 kg-1 C = m2 m-2 s-1;
        # s-1 * m2 m-2 = m2 m-2 s-1.
        # Open system route: all values are observation-only diagnostics.
        lai_growth_rate = growth_leaf * sla_tensor
        lai_senescence = leaf_area_index * senescence_rate_tensor
        lai_change = lai_growth_rate - lai_senescence

        return {
            "gross_leaf_area_growth_tendency": lai_growth_rate,
            "leaf_area_senescence_tendency": lai_senescence,
            "net_leaf_area_tendency": lai_change,
        }

        
    def forward(self, state, time, dt, **kwargs) -> Dict[str, UpdateItem]:
        """Evaluate the signed observer balance through canonical N_h r_h."""
        # References:
        # - Wright et al. (2004); Yin et al. (2021); Feinberg (2019).
        # Mathematical expression:
        # - r_net = SLA * growth_rate_leaf - k_leaf * leaf_area_index.
        # - y_dot = N_h r_h and Delta_y = y_dot * dt.
        # Parameter meanings:
        # - N_h=[1,-1] maps gross growth and senescence to the net tendency.
        # - dt is the current positive time step in seconds.
        # Units: tendencies are m2 m-2 s-1; Delta_y is m2 m-2.
        # Open system route: observation; no leaf_area_index state is written.
        del time, kwargs
        rate_dict = self._compute_rate_core(state)
        prototype = rate_dict["net_leaf_area_tendency"]
        dt_tensor = torch.as_tensor(dt, device=prototype.device, dtype=prototype.dtype)
        if dt_tensor.numel() != 1 or not torch.all(torch.isfinite(dt_tensor)):
            raise ValueError("dt must be one finite scalar")
        if torch.any(dt_tensor <= 0):
            raise ValueError("dt must be positive")

        species = ("diagnostics.structural_growth.net_leaf_area_tendency",)
        reactions = (
            "leaf_area_growth",
            "leaf_area_senescence",
        )
        structural_network = advance_reaction_network(
            name=f"{self.name}.structural_growth_reaction_network",
            species=species,
            reactions=reactions,
            stoichiometric_N=torch.as_tensor(
                [[1.0, -1.0]],
                device=prototype.device,
                dtype=prototype.dtype,
            ),
            rates=(
                rate_dict["gross_leaf_area_growth_tendency"],
                rate_dict["leaf_area_senescence_tendency"],
            ),
            dt=dt_tensor,
            open_system_route="observation",
        )
        net_tendency = structural_network.species_rate(species[0])
        if structural_network.state_delta is None:
            raise RuntimeError("Structural growth observer did not produce a step delta")
        step_delta = structural_network.state_delta[0]

        def diagnostic(value: Tensor, unit: str, *, per_timestep: bool) -> UpdateItem:
            return UpdateItem(
                value=value,
                mode=UpdateMode.REPLACE,
                unit=unit,
                per_timestep=per_timestep,
                source_functor=self.name,
            )

        return {
            "diagnostics.structural_growth.gross_leaf_area_growth_tendency": diagnostic(
                rate_dict["gross_leaf_area_growth_tendency"],
                "m2 m-2 s-1",
                per_timestep=False,
            ),
            "diagnostics.structural_growth.leaf_area_senescence_tendency": diagnostic(
                rate_dict["leaf_area_senescence_tendency"],
                "m2 m-2 s-1",
                per_timestep=False,
            ),
            "diagnostics.structural_growth.net_leaf_area_tendency": diagnostic(
                net_tendency,
                "m2 m-2 s-1",
                per_timestep=False,
            ),
            "diagnostics.structural_growth.step_leaf_area_delta": diagnostic(
                step_delta,
                "m2 m-2",
                per_timestep=True,
            ),
        }


__all__ = ['StructuralGrowth']

