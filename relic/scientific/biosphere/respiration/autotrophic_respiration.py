"""Canonical state-driven autotrophic-respiration carbon ledger.

References
----------
- Amthor (2000), Annals of Botany 86, 1-20.
- Cannell and Thornley (2000), Annals of Botany 85, 45-54.
- Gifford (2003), Functional Plant Biology 30, 171-186.

The unique production ledger is ``R_a = R_m + R_g`` and
``NPP = GPP - R_a``. All terms use ``kg C m-2 s-1``. Component Functors
remain the sole owners of ``R_m`` and ``R_g``; this aggregator consumes their
committed state values and does not scale or re-emit them as public aliases.
"""

import torch
from torch import Tensor

from relic.computation.operators.iphs import advance_reaction_network
from relic.core.functor import BiologicalProcessFunctor
from relic.core.state import UpdateItem, UpdateMode


class AutotrophicRespiration(BiologicalProcessFunctor):
    """
    Unified Autotrophic Respiration Model.

    Aggregate maintenance and growth fluxes produced by scheduled Functors.

    Component process laws remain owned by their respective Functors. This
    aggregator declares them as dependencies and consumes their committed
    state outputs without direct Functor-to-Functor calls.
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'

    def __init__(
        self,
        name: str = "autotrophic_respiration",
        device: torch.device | None = None,
        pft_name: str | None = None
    ):
        """
        Initialize autotrophic respiration model.

        Parameters
        ----------
        name : str, optional
            Model name for identification
        device : torch.device, optional
            Computation device (CPU/CUDA)
        pft_name : str, optional
            Plant Functional Type name for specialized parameters
        """
        # Create metadata
        from relic.core.functor import (
            ExecutionPhase,
            FunctorMetadata,
            ProcessCategory,
            SpatialScale,
            VariableKind,
        )
        metadata = FunctorMetadata(
            name=name,
            version="2.0.0",
            description="Unique kg-C autotrophic respiration and NPP ledger",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "R_auto = R_maintenance + R_growth; NPP = GPP - R_auto"
            ),
            references=[
                "Amthor, J.S. (2000). Annals of Botany 86, 1-20.",
                "Cannell, M.G.R. and Thornley, J.H.M. (2000). Annals of Botany 85, 45-54.",
                "Gifford, R.M. (2003). Functional Plant Biology 30, 171-186.",
            ],
            parameters={
                "biosphere.validation.epsilon.flux": (
                    "Near-zero diagnostic fraction threshold [kg C m-2 s-1]"
                ),
            },
            input_variables=[
                "gross_primary_productivity",
                "maintenance_respiration_total",
                "growth_respiration_total",
            ],
            output_variables=[
                "total_autotrophic_respiration",
                "net_primary_productivity",
                "diagnostics.autotrophic_respiration.maintenance_component",
                "diagnostics.autotrophic_respiration.growth_component",
                "diagnostics.autotrophic_respiration.maintenance_fraction",
                "diagnostics.autotrophic_respiration.growth_fraction",
                "diagnostics.autotrophic_respiration.carbon_balance_residual",
            ],
            conservation_laws=["carbon"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
        )

        aggregate_parameters = dict(metadata.parameters)

        # Initialize BiologicalProcessFunctor
        super().__init__(
            name=name,
            metadata=metadata,
            device=device
        )
        self.set_dependencies(
            ["maintenance_respiration", "growth_respiration"]
        )
        # This aggregate consumes only its diagnostic threshold; the generic
        # biological response parameters injected by the bridge are not used.
        self.metadata.parameters.clear()
        self.metadata.parameters.update(aggregate_parameters)

        self.pft_name = pft_name

    def get_provides(self) -> list:
        """Return aggregate-owned outputs without re-emitting component state."""
        return list(self.metadata.output_variables)

    @staticmethod
    def _require_state_tensor(
        state: dict[str, Tensor | float],
        key: str,
    ) -> Tensor:
        """Read one declared tensor input without aliases or fallback values."""
        if key not in state:
            raise KeyError(f"Required state variable '{key}' is missing")
        value = state[key]
        if not torch.is_tensor(value):
            raise TypeError(f"Required state variable '{key}' must be a torch.Tensor")
        if not value.is_floating_point():
            raise TypeError(
                f"Required state variable '{key}' must be a floating-point torch.Tensor"
            )
        return value

    def forward(
        # References: Amthor (2000); Cannell and Thornley (2000); Gifford (2003).
        # Mathematical expression: [Ra,NPP]^T=[[0,1,1],[1,-1,-1]]*[GPP,Rm,Rg]^T; r=GPP-Ra-NPP.
        # Parameter meanings: Rm and Rg are the committed rates supplied by their scheduled
        #   component owners; this aggregator neither recomputes nor scales them.
        # Units: All carbon rate ports and raw residual [kg C m-2 s-1]; component fractions [1].
        # Domain/constraints: Finite nonnegative input rates; signed NPP is retained even if
        #   respiration exceeds GPP. Time-independent algebraic ports, not an autonomous biomass
        #   advance.
        # Sign convention: Respiration is positive carbon export; NPP is signed net production.
        self,
        state: dict[str, Tensor | float],
        time: float,
        dt: float,
        **kwargs
    ) -> dict[str, UpdateItem]:
        """
        Forward pass through unified respiration model.

        Parameters
        ----------
        state : Dict[str, Union[Tensor, float]]
            Input state containing required variables
        time : float
            Current simulation time [s]
        dt : float
            Time step [s]
        **kwargs
            Additional keyword arguments

        Returns
        -------
        dict[str, UpdateItem]
            Aggregate respiration, NPP, and namespaced diagnostics.
        """
        gpp = self._require_state_tensor(state, "gross_primary_productivity")
        maintenance_total = self._require_state_tensor(
            state,
            "maintenance_respiration_total",
        )
        growth_total = self._require_state_tensor(state, "growth_respiration_total")
        required_tensors = {
            "gross_primary_productivity": gpp,
            "maintenance_respiration_total": maintenance_total,
            "growth_respiration_total": growth_total,
        }
        for variable_name, value in required_tensors.items():
            if not torch.isfinite(value).all():
                raise ValueError(
                    f"Input '{variable_name}' must contain only finite values"
                )
            if torch.any(value < 0):
                raise ValueError(f"Input '{variable_name}' must be non-negative")

        maintenance_total = maintenance_total.to(device=gpp.device, dtype=gpp.dtype)
        growth_total = growth_total.to(device=gpp.device, dtype=gpp.dtype)

        # References: Amthor (2000); Cannell and Thornley (2000); Gifford (2003).
        # Mathematical expression: R_a=R_m+R_g and NPP=GPP-R_a.
        # Parameter meanings: GPP is gross carbon uptake; R_m and R_g are the
        # scheduler-produced maintenance and construction carbon exports.
        # Units: GPP, R_m, R_g, R_a, and NPP use kg C m-2 s-1.
        # Open-system route: GPP is an input port, and R_a is an atmosphere export.
        # IPHS wrapper: x_dot = N_h r_h with r_h=[GPP,R_m,R_g],
        # N_h(R_a)=[0,1,1], and N_h(NPP)=[1,-1,-1].
        respiration_network = advance_reaction_network(
            name=f"{self.name}.autotrophic_respiration_reaction_network",
            species=(
                "total_autotrophic_respiration",
                "net_primary_productivity",
            ),
            reactions=(
                "gross_primary_productivity_input",
                "maintenance_respiration_export",
                "growth_respiration_export",
            ),
            stoichiometric_N=torch.as_tensor(
                [
                    [0.0, 1.0, 1.0],
                    [1.0, -1.0, -1.0],
                ],
                device=gpp.device,
                dtype=gpp.dtype,
            ),
            rates=(gpp, maintenance_total, growth_total),
            open_system_route="port",
        )
        total_respiration = respiration_network.species_rate(
            "total_autotrophic_respiration"
        )
        net_primary_productivity = respiration_network.species_rate(
            "net_primary_productivity"
        )
        fraction_epsilon = torch.as_tensor(
            self.get_parameter_value(
                "flux",
                namespace="biosphere.validation.epsilon",
                required=True,
            ),
            device=total_respiration.device,
            dtype=total_respiration.dtype,
        )
        positive_total = total_respiration > fraction_epsilon
        safe_total = torch.where(
            positive_total,
            total_respiration,
            torch.ones_like(total_respiration),
        )
        maintenance_fraction = torch.where(
            positive_total,
            maintenance_total / safe_total,
            torch.zeros_like(total_respiration),
        )
        growth_fraction = torch.where(
            positive_total,
            growth_total / safe_total,
            torch.zeros_like(total_respiration),
        )
        carbon_balance_residual = gpp - total_respiration - net_primary_productivity

        def replace(value: Tensor, unit: str) -> UpdateItem:
            return UpdateItem(
                value=value,
                mode=UpdateMode.REPLACE,
                unit=unit,
                per_timestep=False,
                source_functor=self.name,
            )

        return {
            "total_autotrophic_respiration": replace(
                total_respiration,
                "kg C m-2 s-1",
            ),
            "net_primary_productivity": replace(
                net_primary_productivity,
                "kg C m-2 s-1",
            ),
            "diagnostics.autotrophic_respiration.maintenance_component": replace(
                maintenance_total,
                "kg C m-2 s-1",
            ),
            "diagnostics.autotrophic_respiration.growth_component": replace(
                growth_total,
                "kg C m-2 s-1",
            ),
            "diagnostics.autotrophic_respiration.maintenance_fraction": replace(
                maintenance_fraction,
                "1",
            ),
            "diagnostics.autotrophic_respiration.growth_fraction": replace(
                growth_fraction,
                "1",
            ),
            "diagnostics.autotrophic_respiration.carbon_balance_residual": replace(
                carbon_balance_residual,
                "kg C m-2 s-1",
            ),
        }


__all__ = ['AutotrophicRespiration']

