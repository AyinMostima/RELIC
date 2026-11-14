"""
Simple biosphere functor for RELIC.

Implements an explicitly selected lightweight canopy carbon closure with
mass-based productivity, respiration, and net-exchange ports.

Mathematical Expressions
------------------------
1. Gross Primary Productivity (GPP):
   ``GPP = G_max * (I / (I + K_I)) * exp(-((T - T_opt) / T_width)^2) * f_w * LAI``
2. Ecosystem Respiration (Reco):
   ``Reco = R_base * Q10 ** ((T - T_ref) / 10) * LAI``
3. Net Ecosystem Exchange (NEE):
   ``NEE = Reco - GPP`` (positive from ecosystem to atmosphere)

References
----------
- Baly, E. C. C., & Morgan, L. B. (1934). Kinetics of Photosynthesis and
  Allied Processes. Nature, 133, 414.
- Mahecha, M. D., et al. (2010). Global Convergence in the Temperature
  Sensitivity of Respiration at Ecosystem Level. Science, 329, 838-840.
- Chapin, F. S., et al. (2006). Reconciling Carbon-cycle Concepts,
  Terminology, and Methods. Ecosystems, 9, 1041-1050.
- Baldocchi, D. (2003). Assessing the eddy covariance technique for evaluating
  carbon dioxide exchange rates of ecosystems: past, present and future.
  Global Change Biology, 9, 479-492.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import torch
from torch import Tensor

from ....core.abstractions import FluxDict, StateDict
from ....core.functor import BiologicalProcessFunctor
from ....core.functor import ExecutionPhase, FunctorMetadata, ProcessCategory, SpatialScale, VariableKind
from ....core.state import UpdateItem, UpdateMode
from ....computation.operators.iphs import advance_reaction_network
from ....utils.physical_constants import get_physical_constant


class SimpleCanopyPhotosynthesis(BiologicalProcessFunctor):
    """
    Simple canopy photosynthesis and respiration functor.

    Provides smoothly varying mass-based canopy carbon fluxes as an explicit
    alternative to the Farquhar closure while remaining consistent with the
    FunctorBase lifecycle and parameter management pipeline.
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'

    def __init__(
        self,
        name: str = "simple_photosynthesis",
        config: Optional[Dict[str, Union[str, float]]] = None,
        device: Optional[torch.device] = None,
        **kwargs: Dict[str, Union[str, float]],
    ) -> None:
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Simple canopy photosynthesis and respiration functor.",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "GPP = G_max * (PPFD/(PPFD+K_PPFD)) * "
                "exp(-((T-T_opt)/T_width)^2) * f_w * LAI; "
                "Reco = R_base * Q10^((T-T_ref)/10) * LAI; NEE = Reco - GPP"
            ),
            references=[
                "Baly, E.C.C., Morgan, L.B. (1934). Nature 133:414.",
                "Mahecha, M.D., et al. (2010). Science 329:838-840.",
                "Chapin, F.S., et al. (2006). Ecosystems 9:1041-1050.",
                "Baldocchi, D. (2003). Global Change Biology 9:479-492.",
            ],
            parameters={
                "biosphere.photosynthesis.surrogate.max_gpp_rate": (
                    "Maximum gross primary productivity [kg C m-2 s-1]"
                ),
                "biosphere.photosynthesis.surrogate.ppfd_half_saturation": (
                    "Half-saturation PPFD of the light response [umol photons m-2 s-1]"
                ),
                "biosphere.photosynthesis.surrogate.temperature_optimum": (
                    "Optimum canopy temperature [K]"
                ),
                "biosphere.photosynthesis.surrogate.temperature_width": (
                    "Width of the temperature response [K]"
                ),
                "biosphere.photosynthesis.surrogate.base_respiration_rate": (
                    "Ecosystem respiration rate at the reference temperature [kg C m-2 s-1]"
                ),
                "biosphere.photosynthesis.surrogate.respiration_q10": (
                    "Q10 of the ecosystem respiration [dimensionless]"
                ),
                "biosphere.photosynthesis.surrogate.soil_moisture_optimum": (
                    "Optimum volumetric soil moisture of photosynthesis [m3 m-3]"
                ),
                "biosphere.photosynthesis.surrogate.soil_moisture_width": (
                    "Width of the soil moisture response [m3 m-3]"
                ),
            },
            input_variables=[
                "ppfd",
                "air_temperature",
                "leaf_area_index",
                "soil_moisture",
            ],
            output_variables=[
                "gross_primary_productivity",
                "ecosystem_respiration",
                "net_ecosystem_exchange",
                "diagnostics.photosynthesis.water_stress",
                "diagnostics.photosynthesis.temperature_factor",
            ],
            conservation_laws=["carbon"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
        )

        super().__init__(name=name, config=config or {}, metadata=metadata, device=device, **kwargs)
        # PPFD provider as an instance-level declaration merged with
        # config["dependencies"]; an explicit composition rebinds it.
        self.add_dependency("atmospheric_radiation")

        for parameter_name in (
            "max_gpp_rate",
            "ppfd_half_saturation",
            "temperature_optimum",
            "temperature_width",
            "base_respiration_rate",
            "respiration_q10",
            "soil_moisture_optimum",
            "soil_moisture_width",
        ):
            parameter_tensor = self._parameter_tensor(parameter_name)
            if parameter_name in self.__dict__:
                delattr(self, parameter_name)
            self.register_buffer(
                parameter_name,
                parameter_tensor,
                persistent=True,
            )

        self.register_buffer(
            "respiration_reference_temperature",
            self._to_scalar_tensor(
                get_physical_constant(
                    "reference_temperature",
                    as_tensor=True,
                    device=self.device,
                ),
                "reference_temperature",
            ),
            persistent=True,
        )

    # ------------------------------------------------------------------ helpers
    def _parameter_tensor(self, name: str) -> Tensor:
        value = self.get_parameter_value(
            name,
            namespace="biosphere.photosynthesis.surrogate",
            required=True,
        )
        return self._to_scalar_tensor(value, name)

    def _to_scalar_tensor(self, value: Union[Tensor, float], name: str) -> Tensor:
        tensor = self._to_tensor(value)
        if tensor.numel() != 1:
            raise ValueError(f"Expected scalar parameter for '{name}', got shape={tuple(tensor.shape)}")
        return tensor.reshape(())

    def _to_tensor(self, value: Union[Tensor, float], reference: Optional[Tensor] = None) -> Tensor:
        if isinstance(value, Tensor):
            tensor = value.to(device=self.device, dtype=torch.float32)
        else:
            tensor = torch.tensor(value, device=self.device, dtype=torch.float32)
        if reference is not None:
            tensor = tensor.expand_as(reference)
        return tensor

    def _require_state_tensor(self, state: StateDict, key: str) -> Tensor:
        """Return one finite floating tensor from the canonical state port."""
        if key not in state or state[key] is None:
            raise KeyError(f"Missing required state variable '{key}'")
        value = state[key]
        if not isinstance(value, Tensor) or not value.is_floating_point():
            raise TypeError(f"State variable '{key}' must be a floating torch.Tensor")
        if value.numel() == 0:
            raise ValueError(f"State variable '{key}' must not be empty")
        tensor = value.to(device=self.device)
        if not torch.all(torch.isfinite(tensor)):
            raise ValueError(f"State variable '{key}' must contain only finite values")
        return tensor

    @staticmethod
    def _validate_input_domains(
        ppfd: Tensor,
        air_temperature: Tensor,
        leaf_area_index: Tensor,
        soil_moisture: Tensor,
    ) -> None:
        """Validate physical domains without clipping invalid scientific input."""
        if torch.any(ppfd < 0):
            raise ValueError("State variable 'ppfd' must be non-negative")
        if torch.any(air_temperature <= 0):
            raise ValueError("State variable 'air_temperature' must be positive in K")
        if torch.any(leaf_area_index < 0):
            raise ValueError("State variable 'leaf_area_index' must be non-negative")
        if torch.any((soil_moisture < 0) | (soil_moisture > 1)):
            raise ValueError("State variable 'soil_moisture' must be within [0, 1]")

    # ------------------------------------------------------------------ forward
    # References: Baly and Morgan (1934); Mahecha et al. (2010); Chapin et al. (2006).
    # Math: GPP = G_max * (PPFD/(PPFD+K_PPFD)) * exp(-((T-T_opt)/T_width)^2) * f_w * LAI
    #       Reco = R_base * Q10^((T-T_ref)/10) * LAI
    #       NEE = Reco - GPP, positive from the ecosystem to the atmosphere.
    # Params: G_max=max_gpp_rate, K_PPFD=ppfd_half_saturation, T_opt=temperature_optimum,
    #         T_width=temperature_width, R_base=base_respiration_rate, Q10=respiration_q10,
    #         f_w=soil_moisture response, LAI=leaf_area_index, T_ref=reference_temperature.
    def forward(
        self,
        state: StateDict,
        time: float,
        dt: float,
        **kwargs: Dict[str, Union[str, float]],
    ) -> FluxDict:
        # References: Baly and Morgan (1934); Mahecha et al. (2010); Chapin et al. (2006).
        # Mathematical expression: GPP=G_max*Q/(Q+Q_half)*exp(-((T-T_opt)/w_T)^2)
        #   *exp(-((theta-theta_opt)/w_theta)^2)*LAI;
        #   Reco=R_ref*Q10^((T-T_ref)/10 K)*LAI; NEE=Reco-GPP.
        # Parameter meanings: Q is PPFD; theta is volumetric moisture; widths, optima,
        #   half-saturation, G_max, R_ref and Q10 belong to the existing surrogate.
        # Units: Q [umol m-2 s-1]; T [K]; theta [m3 m-3]; LAI [1]; carbon rates
        #   [kg C m-2 s-1]; temperature interval [K].
        # Domain/constraints: Explicit opt-in, finite broadcast-compatible inputs;
        #   this empirical canopy surrogate is not the Farquhar biochemical model.
        # Sign convention: GPP and Reco are positive magnitudes; NEE is positive toward
        #   atmosphere. REPLACE rates are not multiplied by dt or committed as stocks.
        ppfd_tensor = self._require_state_tensor(state, "ppfd")
        air_temp_tensor = self._require_state_tensor(state, "air_temperature")
        lai_tensor = self._require_state_tensor(state, "leaf_area_index")
        soil_tensor = self._require_state_tensor(state, "soil_moisture")

        try:
            ppfd_tensor, air_temp_tensor, lai_tensor, soil_tensor = torch.broadcast_tensors(
                ppfd_tensor,
                air_temp_tensor,
                lai_tensor,
                soil_tensor,
            )
        except RuntimeError as exc:
            raise ValueError("Simple canopy inputs must be broadcast-compatible") from exc
        self._validate_input_domains(
            ppfd_tensor,
            air_temp_tensor,
            lai_tensor,
            soil_tensor,
        )

        ppfd_factor = ppfd_tensor / (ppfd_tensor + self.ppfd_half_saturation)
        temp_factor = torch.exp(
            -((air_temp_tensor - self.temperature_optimum) / self.temperature_width) ** 2
        )
        soil_factor = torch.exp(
            -((soil_tensor - self.soil_moisture_optimum) / self.soil_moisture_width) ** 2
        )

        gpp = self.max_gpp_rate * ppfd_factor * temp_factor * soil_factor * lai_tensor
        resp_flux = (
            self.base_respiration_rate
            * torch.pow(self.respiration_q10, (air_temp_tensor - self.respiration_reference_temperature) / 10.0)
            * lai_tensor
        )
        if not torch.all(torch.isfinite(gpp)) or not torch.all(torch.isfinite(resp_flux)):
            raise ValueError("Simple canopy calculation produced a non-finite carbon flux")

        # References: Baly and Morgan (1934); Mahecha et al. (2010); Chapin et al. (2006).
        # Mathematical expression: x_dot = N_h r_h with r_h = [GPP, Reco].
        # Parameter meanings: N_h maps mass-based canopy carbon reaction channels
        # to gross productivity, respiration, and positive-to-atmosphere NEE ports.
        canopy_network = advance_reaction_network(
            name=f"{self.name}.canopy_carbon_reaction_network",
            species=(
                "gross_primary_productivity",
                "ecosystem_respiration",
                "net_ecosystem_exchange",
            ),
            reactions=(
                "gross_canopy_assimilation",
                "ecosystem_respiration",
            ),
            stoichiometric_N=torch.as_tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, 1.0],
                ],
                device=gpp.device,
                dtype=gpp.dtype,
            ),
            rates=(gpp, resp_flux),
            open_system_route="port",
        )
        gross_primary_productivity = canopy_network.species_rate("gross_primary_productivity")
        ecosystem_respiration = canopy_network.species_rate("ecosystem_respiration")
        net_ecosystem_exchange = canopy_network.species_rate("net_ecosystem_exchange")

        updates: FluxDict = {
            "gross_primary_productivity": UpdateItem(
                value=gross_primary_productivity,
                mode=UpdateMode.REPLACE,
                unit="kg C m-2 s-1",
                per_timestep=False,
            ),
            "ecosystem_respiration": UpdateItem(
                value=ecosystem_respiration,
                mode=UpdateMode.REPLACE,
                unit="kg C m-2 s-1",
                per_timestep=False,
            ),
            "net_ecosystem_exchange": UpdateItem(
                value=net_ecosystem_exchange,
                mode=UpdateMode.REPLACE,
                unit="kg C m-2 s-1",
                per_timestep=False,
            ),
            "diagnostics.photosynthesis.water_stress": soil_factor,
            "diagnostics.photosynthesis.temperature_factor": temp_factor,
        }

        return updates

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Return the positive-to-atmosphere canopy carbon ledger residual."""
        # References: Chapin et al. (2006), positive-to-atmosphere ecosystem exchange.
        # Mathematical expression: residual=max(abs(NEE-(Reco-GPP))).
        # Parameter meanings: GPP, Reco and NEE are the three declared public rates.
        # Units: all terms and residual [kg C m-2 s-1].
        # Domain/constraints: This checks a rate identity, not a plant-stock budget;
        #   missing outputs return no result and must not be described as a pass.
        # Sign convention: Reco contributes positive NEE and GPP negative NEE.
        del input_state
        required = (
            "gross_primary_productivity",
            "ecosystem_respiration",
            "net_ecosystem_exchange",
        )
        if any(key not in output_updates for key in required):
            return {}

        def _value(key: str) -> Tensor:
            item = output_updates[key]
            return item.value if isinstance(item, UpdateItem) else item

        gpp = _value("gross_primary_productivity")
        respiration = _value("ecosystem_respiration")
        nee = _value("net_ecosystem_exchange")
        residual = nee - (respiration - gpp)
        magnitude = torch.max(torch.abs(residual)).detach().cpu().item()
        return {"carbon": float(magnitude)}


__all__ = ["SimpleCanopyPhotosynthesis"]
