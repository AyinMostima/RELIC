"""Methane oxidation functor for wetland soils (CLM4Me double Michaelis-Menten kinetics)."""

from typing import Any, Dict, Optional

import torch
from torch import Tensor

from relic.core.functor import (
    BiologicalProcessFunctor,
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from relic.core.state import UpdateItem, UpdateMode
from relic.core.abstractions import StateDict
from relic.computation.operators.iphs import advance_reaction_network
from relic.scientific.pedosphere.api_contract import PedosphereContractMixin
from relic.utils.physical_constants import get_physical_constant


class MethaneOxidationFunctor(PedosphereContractMixin, BiologicalProcessFunctor):
    """Consume soil methane through methanotrophic oxidation.

    Scientific references:
        - Riley et al. (2011) doi:10.5194/bg-8-1925-2011, Eq. 7, Table 1 (wetland
          baseline values), Sect. 2.1.7 (demand scaling) and Sect. 2.7 (oxidation Q10)
        - Oleson et al. (2013) CLM4.5 Technical Note NCAR/TN-503+STR, Sect. 19.6,
          Eq. 19.11, Table 19.1 and Eq. 19.1 (concentration per water volume)
        - Walter and Heimann (2000) doi:10.1029/1999GB001204 (K_CH4 = 5 umol L-1)

    Governing equations (ASCII form):
        C_CH4 = M / (M_C * z * phi * WFPS)
        R     = R_max * C_CH4 / (K_CH4 + C_CH4) * C_O2 / (K_O2 + C_O2)
                * Q10 ** ((Ts - T_base) / dT10) * F,  F = 1
        O_kin = R * z * M_C
        O     = min(O_kin, M / dt)

    Parameter meanings:
        M: soil methane storage [g C m-2]
        z: methane layer depth [m]
        phi, WFPS: porosity and water-filled pore-space fraction [-]
        C_CH4, C_O2: CH4 and O2 concentrations per water volume [mol m-3]
        R_max: maximum oxidation rate per soil volume [mol m-3 s-1]
        K_CH4, K_O2: half-saturation concentrations [mol m-3]
        Q10, T_base: oxidation temperature sensitivity [-] and base temperature [K]
        F: soil moisture limitation factor [-]

    Notes:
        The layer is the saturated wetland soil below the water table, where the
        CLM4.5 moisture factor F applies only above the water table, so F = 1.
        The unsaturated-layer factor F = exp(-P / P_c) needs the soil matric
        potential P, which is not an input of the methane chain; it is not
        implemented. A demand above the available storage is scaled to the
        storage (Riley et al. 2011, Sect. 2.1.7); the unmet demand is reported as
        diagnostics.methane_oxidation.storage_limiter_residual.
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'

    def __init__(
        self,
        name: str = "pedosphere_methane_oxidation",
        device: Optional[torch.device] = None,
    ) -> None:
        parameter_descriptions = {
            "pedosphere.methane.oxidation.maximum_rate": "Maximum methane oxidation rate R_max per soil volume [mol m-3 s-1]",
            "pedosphere.methane.oxidation.half_saturation_ch4": "CH4 half-saturation concentration per water volume [mol m-3]",
            "pedosphere.methane.oxidation.half_saturation_o2": "O2 half-saturation concentration per water volume [mol m-3]",
            "pedosphere.methane.oxidation.q10": "Oxidation temperature sensitivity Q10 [-]",
            "pedosphere.methane.oxidation.temperature_base": "Base temperature of the oxidation Q10 factor [K]",
            "pedosphere.methane.transport.layer_mixing_depth": "Methane layer depth z [m]",
            "pedosphere.numerical.minimum_timestep": "Minimum timestep for methane calculations [s]",
            "pedosphere.numerical.epsilon_length": "Minimum layer depth for methane calculations [m]",
        }

        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description=(
                "Soil methane oxidation by double Michaelis-Menten kinetics in CH4 and O2 "
                "with a Q10 temperature factor (CLM4Me, Riley et al. 2011)"
            ),
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "O = min(M/dt, R_max * C_CH4/(K_CH4 + C_CH4) * C_O2/(K_O2 + C_O2) "
                "* Q10^((Ts - T_base)/dT10) * z * M_C), C_CH4 = M/(M_C * z * phi * WFPS)"
            ),
            references=[
                "Riley et al. (2011) doi:10.5194/bg-8-1925-2011",
                "Oleson et al. (2013) CLM4.5 Technical Note NCAR/TN-503+STR",
                "Walter and Heimann (2000) doi:10.1029/1999GB001204",
            ],
            parameters=parameter_descriptions,
            input_variables=[
                "ch4_soil_storage",
                "soil_temperature",
                "soil_o2_concentration",
                "porosity",
                "water_filled_pore_space",
            ],
            output_variables=[
                "ch4_oxidation_rate",
                "ch4_storage_tendency_oxid",
                "diagnostics.methane_oxidation.kinetic_rate",
                "diagnostics.methane_oxidation.storage_limiter_residual",
            ],
            conservation_laws=["mass"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
            spatial_dimensions=0,
            temporal_scale=TemporalScale.SECOND,
        )
        super().__init__(name=name, metadata=metadata, device=device)

        self.maximum_rate = self.get_parameter_value(
            "maximum_rate",
            namespace="pedosphere.methane.oxidation", required=True)
        self.half_saturation_ch4 = self.get_parameter_value(
            "half_saturation_ch4",
            namespace="pedosphere.methane.oxidation", required=True)
        self.half_saturation_o2 = self.get_parameter_value(
            "half_saturation_o2",
            namespace="pedosphere.methane.oxidation", required=True)
        self.q10 = self.get_parameter_value(
            "q10",
            namespace="pedosphere.methane.oxidation", required=True)
        self.temperature_base = self.get_parameter_value(
            "temperature_base",
            namespace="pedosphere.methane.oxidation", required=True)
        self.q10_interval = float(
            get_physical_constant(
                "relic.physical.constants.temperature.q10_interval"
            )
        )
        self.layer_depth = self.get_parameter_value(
            "layer_mixing_depth",
            namespace="pedosphere.methane.transport", required=True)
        self.dt_minimum = self.get_parameter_value(
            "minimum_timestep",
            namespace="pedosphere.numerical", required=True)
        self.minimum_layer_depth = self.get_parameter_value(
            "epsilon_length",
            namespace="pedosphere.numerical", required=True)
        self.carbon_molar_mass = float(
            get_physical_constant(
                "relic.physical.constants.molecular_weights.carbon"
            )
        )

    def forward(
        # References: Riley et al. (2011), doi:10.5194/bg-8-1925-2011, Eq. 7, Table 1,
        #   Sects. 2.1.7 and 2.7; Oleson et al. (2013) CLM4.5 Technical Note, Eqs. 19.1
        #   and 19.11, Table 19.1.
        # Mathematical expression: C_CH4 = M/(M_C z phi WFPS);
        #   O_kin = R_max C_CH4/(K_CH4 + C_CH4) C_O2/(K_O2 + C_O2) Q10^((Ts-T_base)/dT10) z M_C;
        #   O = min(O_kin, M/dt).
        # Parameter meanings: M methane-carbon storage; z layer depth; phi porosity;
        #   WFPS water-filled pore space; R_max maximum oxidation rate per soil volume;
        #   K_CH4 and K_O2 half-saturation concentrations; M_C carbon molar mass.
        # Units: O, O_kin [g C m-2 s-1]; M [g C m-2]; z [m]; C, K [mol m-3 water];
        #   R_max [mol m-3 s-1]; Ts, T_base, dT10 [K]; phi, WFPS, Q10 [1].
        # Domain/constraints: saturated layer (F = 1); phi and WFPS in (0, 1]; the
        #   demand is scaled to the storage available within dt.
        # Sign convention: oxidation is a positive sink of methane carbon.
        self,
        state: StateDict,
        time: float,
        dt: Tensor | float,
        **kwargs: Any,
    ) -> Dict[str, UpdateItem]:
        dt_value = (
            dt.to(self.device)
            if isinstance(dt, Tensor)
            else torch.as_tensor(dt, device=self.device)
        )
        if dt_value.numel() != 1 or not bool(torch.all(torch.isfinite(dt_value))):
            raise ValueError("Methane oxidation dt must be one finite scalar.")
        if bool(dt_value <= 0.0) or bool(dt_value < self.dt_minimum):
            raise ValueError(
                "Methane oxidation dt must be positive and meet the configured minimum."
            )

        storage = self._require_state_tensor(state, "ch4_soil_storage", lower_bound=0.0)
        temperature = self._require_state_tensor(state, "soil_temperature")
        oxygen = self._require_state_tensor(state, "soil_o2_concentration", lower_bound=0.0)
        porosity = self._require_positive_fraction(state, "porosity")
        water_filled = self._require_positive_fraction(state, "water_filled_pore_space")

        layer_depth = self._tensor(self.layer_depth)
        if not bool(torch.all(torch.isfinite(layer_depth))) or bool(
            torch.any(layer_depth < self.minimum_layer_depth)
        ):
            raise ValueError(
                "Methane oxidation layer_mixing_depth must be finite and meet "
                "the configured minimum."
            )

        # References: Oleson et al. (2013) CLM4.5 Technical Note, Eq. 19.1 (C per
        #   water volume); Riley et al. (2011) Sect. 2.1.1.
        # Mathematical expression: C_CH4 = M / (M_C * z * phi * WFPS).
        # Parameter meanings: M storage; M_C carbon molar mass; z layer depth;
        #   phi * WFPS the water volume per soil volume of the saturated layer.
        # Units: M [g C m-2]; M_C [g mol-1]; z [m]; C_CH4 [mol m-3 water].
        # Domain/constraints: phi and WFPS in (0, 1] are enforced on input.
        # Sign convention: concentrations are nonnegative.
        water_volume = layer_depth * porosity * water_filled
        methane_concentration = storage / (self.carbon_molar_mass * water_volume)

        # References: Riley et al. (2011), Eq. 7, Table 1 and Sect. 2.7 (Q10 = 1.9);
        #   Oleson et al. (2013) CLM4.5 Technical Note, Eq. 19.11 and Table 19.1
        #   (base temperature 12 C); Walter and Heimann (2000) for K_CH4.
        # Mathematical expression: R = R_max * C_CH4/(K_CH4 + C_CH4) * C_O2/(K_O2 + C_O2)
        #   * Q10^((Ts - T_base)/dT10) * F, with F = 1 below the water table;
        #   O_kin = R * z * M_C.
        # Parameter meanings: R_max maximum oxidation rate; K_CH4, K_O2 half-
        #   saturation concentrations; Q10 temperature sensitivity; T_base base
        #   temperature; dT10 the physical-SSOT Q10 interval; F moisture factor.
        # Units: R, R_max [mol m-3 s-1]; C, K [mol m-3 water]; Ts, T_base, dT10 [K];
        #   O_kin [g C m-2 s-1].
        # Domain/constraints: K_CH4, K_O2 > 0 by their bounds, so both Monod factors
        #   lie in [0, 1).
        # Sign convention: O_kin is a nonnegative oxidation demand.
        methane_factor = methane_concentration / (self.half_saturation_ch4 + methane_concentration)
        oxygen_factor = oxygen / (self.half_saturation_o2 + oxygen)
        temperature_factor = torch.pow(
            self.q10,
            (temperature - self.temperature_base) / self._tensor(self.q10_interval),
        )
        kinetic_rate = (
            self.maximum_rate
            * methane_factor
            * oxygen_factor
            * temperature_factor
            * layer_depth
            * self.carbon_molar_mass
        )
        if not bool(torch.all(torch.isfinite(kinetic_rate))) or bool(torch.any(kinetic_rate < 0.0)):
            raise ValueError("Methane oxidation kinetic rate must be finite and non-negative.")

        # References: Riley et al. (2011), Sect. 2.1.7 (a demand above the available
        #   amount is scaled to keep concentrations non-negative).
        # Mathematical expression: O = min(O_kin, M/dt); R_lim = O_kin - O >= 0.
        # Parameter meanings: O realized oxidation; R_lim the unmet oxidation demand.
        # Units: O, O_kin, R_lim [g C m-2 s-1]; M [g C m-2]; dt [s].
        # Domain/constraints: O <= M/dt, so oxidation never exceeds the storage; R_lim
        #   is a demand that did not occur, not carbon, and is reported as a diagnostic.
        # Sign convention: O and R_lim are nonnegative.
        max_possible = storage / dt_value
        oxidation_rate = torch.minimum(kinetic_rate, max_possible)
        storage_limiter_residual = kinetic_rate - oxidation_rate

        # References: Riley et al. (2011); Feinberg (2019).
        # Mathematical expression: x_dot = N_h r_h, with oxidation removing CH4 storage.
        # Parameter meanings: N_h maps methane oxidation to the export port and the
        # negative storage tendency; r_h is the storage-limited oxidation rate.
        oxidation_network = advance_reaction_network(
            name="methane_oxidation_network",
            species=("ch4_oxidation_rate", "ch4_storage_tendency_oxid"),
            reactions=("methane_oxidation",),
            stoichiometric_N=torch.tensor(
                [[1.0], [-1.0]],
                device=oxidation_rate.device,
                dtype=oxidation_rate.dtype,
            ),
            rates=(oxidation_rate,),
            **self._iphs_reaction_conservation_kwargs(open_system_route="port"),
        )
        oxidation_rate = oxidation_network.species_rate("ch4_oxidation_rate")
        oxidation_tendency = oxidation_network.species_rate("ch4_storage_tendency_oxid")

        updates: Dict[str, UpdateItem] = {}
        updates["ch4_oxidation_rate"] = UpdateItem(
            value=oxidation_rate,
            mode=UpdateMode.REPLACE,
            unit="g C m-2 s-1",
            per_timestep=False,
            element="C",
            source_functor=self.name,
        )
        updates["ch4_storage_tendency_oxid"] = UpdateItem(
            value=oxidation_tendency,
            mode=UpdateMode.REPLACE,
            unit="g C m-2 s-1",
            per_timestep=False,
            element="C",
            source_functor=self.name,
        )
        updates["diagnostics.methane_oxidation.kinetic_rate"] = UpdateItem(
            value=kinetic_rate,
            mode=UpdateMode.REPLACE,
            unit="g C m-2 s-1",
            per_timestep=False,
            source_functor=self.name,
        )
        updates["diagnostics.methane_oxidation.storage_limiter_residual"] = UpdateItem(
            value=storage_limiter_residual,
            mode=UpdateMode.REPLACE,
            unit="g C m-2 s-1",
            per_timestep=False,
            source_functor=self.name,
        )

        self._last_updates = updates
        return updates

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _tensor(self, value: Any) -> Tensor:
        if isinstance(value, Tensor):
            return value.to(self.device)
        return torch.as_tensor(value, device=self.device)

    def _require_state_tensor(
        self,
        state: StateDict,
        key: str,
        *,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
    ) -> Tensor:
        """Return one required finite tensor and enforce its declared domain."""
        if key not in state:
            raise KeyError(f"Required methane oxidation input '{key}' is missing.")
        value = state[key]
        if not isinstance(value, Tensor):
            raise TypeError(
                f"Methane oxidation input '{key}' must be a torch.Tensor."
            )
        tensor = value.to(self.device)
        if not bool(torch.all(torch.isfinite(tensor))):
            raise ValueError(
                f"Methane oxidation input '{key}' must contain only finite values."
            )
        if lower_bound is not None and bool(torch.any(tensor < lower_bound)):
            raise ValueError(
                f"Methane oxidation input '{key}' must be >= {lower_bound}."
            )
        if upper_bound is not None and bool(torch.any(tensor > upper_bound)):
            raise ValueError(
                f"Methane oxidation input '{key}' must be <= {upper_bound}."
            )
        return tensor

    def _require_positive_fraction(self, state: StateDict, key: str) -> Tensor:
        """Return a required fraction in (0, 1]; zero water volume has no concentration."""
        tensor = self._require_state_tensor(state, key, upper_bound=1.0)
        if bool(torch.any(tensor <= 0.0)):
            raise ValueError(f"Methane oxidation input '{key}' must be > 0.")
        return tensor

    @staticmethod
    def _tensor_from_update(value: Any) -> Tensor:
        """Return the full update tensor for conservation checks."""
        if hasattr(value, "value"):
            value = value.value
        if isinstance(value, Tensor):
            return value
        return torch.as_tensor(value)

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Check methane oxidation closure using the oxidation-rate increment.

        References:
            - Riley et al. (2011) doi:10.5194/bg-8-1925-2011
        Mathematical expression:
            violation["mass"] = |delta_O + delta_dS_oxid_dt|
        Parameter meanings:
            - ch4_oxidation_rate: update applied to the oxidation diagnostic [g C m-2 s-1]
            - ch4_storage_tendency_oxid: oxidation contribution to storage tendency [g C m-2 s-1]
        """
        if "mass" not in self.metadata.conservation_laws:
            return {}

        oxidation_update = output_updates.get("ch4_oxidation_rate")
        tendency_update = output_updates.get("ch4_storage_tendency_oxid")
        if oxidation_update is None or tendency_update is None:
            return {}

        oxidation_value = self._tensor_from_update(oxidation_update)
        tendency_value = self._tensor_from_update(tendency_update)
        residual = torch.abs(oxidation_value + tendency_value)
        return {"mass": float(torch.max(residual).detach().cpu().item())}


__all__ = ["MethaneOxidationFunctor"]
