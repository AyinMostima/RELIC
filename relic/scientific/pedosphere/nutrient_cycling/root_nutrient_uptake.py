"""
Root Nutrient Uptake Process

Implements root-mediated nutrient acquisition processes including nitrogen (N) and 
phosphorus (P) uptake based on Michaelis-Menten kinetics with root weight distribution,
plant demand regulation, and environmental modifiers.

Mathematical Models:
    Base Michaelis-Menten kinetics: V = Vmax * [S] / (Km + [S])
    Root weight distribution: W(z) = W0 * exp(-k * z)  [exponential decay]
    Plant demand closure: Uptake = min(Supply, Demand)
    Temperature response: fT(T) = Q10^((T-Tref)/10)
    Water stress response: fW(theta) = (theta-thetawilt)/(thetafc-thetawilt) for thetawilt <= theta <= thetafc

References:
    - Barber, S.A. (1995). Soil Nutrient Bioavailability: A Mechanistic Approach
    - Tinker, P.B. & Nye, P.H. (2000). Solute Movement in the Rhizosphere
    - Chapin, F.S. et al. (1993). Evolution of suites of traits in response to environmental stress
    - Jackson, R.B. et al. (1996). A global analysis of root distributions for terrestrial biomes
    - Craine, J.M. et al. (2007). The relationships among root and leaf traits of 76 grassland species
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import logging

from torch import Tensor
from relic.core.functor import BiologicalProcessFunctor, StateDict, FluxDict
from relic.core.state import UpdateItem, UpdateMode
from relic.computation.operators.iphs import advance_reaction_network
from relic.scientific.pedosphere.api_contract import (
    FluxDirection,
    FluxSpecification,
    FluxType,
    PedosphereContractMixin,
)
from relic.utils.physical_constants import get_physical_constant

logger = logging.getLogger(__name__)

class RootNutrientUptake(PedosphereContractMixin, BiologicalProcessFunctor):
    """
    Root Nutrient Uptake Process
    
    Models plant root nutrient acquisition for nitrogen (N) and phosphorus (P)
    using Michaelis-Menten kinetics combined with root distribution weighting,
    plant demand regulation, and environmental response functions.
    
    Mathematical Expression:
        Layer uptake rate: 
        U_i = Vmax_i * C_i / (Km + C_i) * f_T(T) * f_W(theta_i) * RootWeight_i
        
        Total uptake:
        U_total = Sigma_i U_i = Sigma_i [Vmax_i * C_i / (Km + C_i)] * f_env * W_i
        
        Demand closure:
        U_actual = min(U_total, PlantDemand - CurrentStorage)
        
        Root weight distribution:
        W_i = W0 * exp(-k_root * depth_i) * root_density_i
        
        Where:
        - U_i: layer uptake rate [mol m^-2 s^-1]
        - Vmax_i: maximum uptake rate in layer i [mol m^-2 s^-1]  
        - C_i: nutrient concentration in layer i [mol m^-3]
        - Km: Michaelis constant [mol m^-3]
        - f_T: temperature response function [dimensionless]
        - f_W: water stress response function [dimensionless] 
        - W_i: root weight in layer i [dimensionless]
        - PlantDemand: plant nutrient demand [mol m^-2 s^-1]
    
    Parameters:
        vmax_n : float
            Maximum N uptake rate [mol N m^-2 s^-1]
            Default: 1.0e-6
        vmax_p : float  
            Maximum P uptake rate [mol P m^-2 s^-1]
            Default: 1.0e-7
        km_n : float
            Michaelis constant for N [mol N m^-3]
            Default: 0.1
        km_p : float
            Michaelis constant for P [mol P m^-3] 
            Default: 0.01
        root_decay_coeff : float
            Root distribution decay coefficient [m^-1]
            Default: 2.0
        q10_value : float
            Temperature response Q10 coefficient [dimensionless]
            Default: 2.0
        reference_temperature : float
            Reference temperature for Q10 response [K]
            Default: 298.15
        water_stress_enabled : bool
            Whether to apply water stress response [dimensionless]
            Default: True
        demand_regulation_enabled : bool
            Whether to apply plant demand regulation [dimensionless] 
            Default: True
        max_storage_n : float
            Maximum plant N storage capacity [mol N m^-2]
            Default: 10.0
        max_storage_p : float
            Maximum plant P storage capacity [mol P m^-2]
            Default: 1.0
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'
    def get_dependencies(self) -> List[str]:
        """
        Get list of dependencies for this functor.

        Root nutrient uptake depends on nutrient availability from mineralization
        and nitrification processes, as well as soil environmental conditions.

        Returns:
            List[str]: List of dependency names
        """
        # Dependencies on nutrient cycling processes
        dependencies = ["mineralization", "nitrification"]

        return self._merge_declared_dependencies(dependencies)

    def forward(
        # References: Barber (1995); Tinker and Nye (2000); Chapin et al. (1993); Jackson et al. (1996).
        # Mathematical expression: U_i(z) = Vmax_i C_i(z)/(Km_i + C_i(z)) f_T f_W W_root(z);
        #   U_areal = sum_z U(z), bounded by pool availability m/(M dt), plant demand and storage
        #   capacity; x_dot = N_h r_h through advance_reaction_network; layered donor pools
        #   receive Delta_pool(z) = -M U(z) dt and plant storages the areal increment.
        # Parameter meanings: Vmax, Km kinetic constants; C_i(z) layer concentrations from the
        #   areal pools and layer depth; f_T, f_W environmental modifiers; W_root the root weight
        #   distribution; M the molar mass.
        # Units: U [mol m-2 s-1]; C [mol m-3]; pools and increments [g N m-2] or [g P m-2];
        #   plant_n_storage [mol N m-2]; dt [s].
        # Domain/constraints: finite non-negative canonical states; volumetric fractions in
        #   [0, 1]; field capacity above wilting point; donor increments keep the declared pool
        #   shape so sum_z Delta(z) equals the areal transfer.
        # Sign convention: soil pools lose mass (negative increments); plant storages gain it.
        self,
        state: StateDict,
        time: float,
        dt: Tensor | float,
        **kwargs,
    ) -> FluxDict:
        """
        Forward computation for nutrient uptake.

        Parameters:
            state : StateDict
                Current state variables
            time : float
                Current simulation time [s]
            dt : float
                Time step size [s]
            **kwargs : Dict
                Not read: the drivers and demands are declared state inputs (the
                one input channel).

        Returns:
            FluxDict: Pool deltas and diagnostic rates for state manager
            - no3_nitrogen, nh4_nitrogen: g N m-2 per step
            - available_phosphorus, plant_p_storage: g P m-2 per step
            - plant_n_storage: mol N m-2 per step
            - nitrogen_uptake_rate: mol N m-2 s-1
            - phosphorus_uptake_rate: g P m-2 s-1
        """
        required_inputs = tuple(self.metadata.input_variables)
        nonnegative_inputs = {
            "no3_nitrogen",
            "nh4_nitrogen",
            "available_phosphorus",
            "soil_water_content",
            "plant_n_storage",
            "plant_p_storage",
            "root_density",
            "plant_n_demand",
            "plant_p_demand",
            "field_capacity",
            "wilting_point",
        }
        validated_state: StateDict = dict(state)
        for variable in required_inputs:
            if variable not in state or state[variable] is None:
                raise ValueError(f"Required canonical state missing: {variable}")
            value = self._to_tensor(state[variable])
            if not bool(torch.all(torch.isfinite(value)).item()):
                raise ValueError(f"Canonical state must be finite: {variable}")
            if variable in nonnegative_inputs and bool(torch.any(value < 0.0).item()):
                raise ValueError(f"Canonical state must be non-negative: {variable}")
            validated_state[variable] = value

        if bool(torch.any(validated_state["soil_depth"] <= 0.0).item()):
            raise ValueError("Canonical state must be positive: soil_depth")
        for variable in ("soil_water_content", "field_capacity", "wilting_point"):
            value = validated_state[variable]
            if bool(torch.any(value > 1.0).item()):
                raise ValueError(f"Canonical volumetric fraction must be within [0, 1]: {variable}")
        if bool(
            torch.any(
                validated_state["field_capacity"]
                <= validated_state["wilting_point"]
            ).item()
        ):
            raise ValueError("field_capacity must be greater than wilting_point")

        rates = self._compute_rate_core(validated_state)

        rate_tensor = rates.get("nitrogen_uptake_rate")
        rate_dtype = rate_tensor.dtype if torch.is_tensor(rate_tensor) else None
        dt_tensor = self._to_tensor(dt, dtype=rate_dtype)
        if not bool(torch.all(torch.isfinite(dt_tensor)).item()):
            raise ValueError("Time step must be finite")
        if bool(torch.any(dt_tensor < 0.0).item()):
            raise ValueError("Time step must be non-negative")

        no3_uptake_by_layer = rates["no3_uptake_by_layer"]
        nh4_uptake_by_layer = rates["nh4_uptake_by_layer"]
        po4_uptake_by_layer_mol = rates["po4_uptake_by_layer"]

        # References: Barber (1995); Tinker and Nye (2000).
        # Formula: uptake_areal = sum(uptake_by_layer); delta_pool = -uptake_areal * molar_mass * dt.
        # Params: uptake_by_layer [mol m-2 s-1], molar_mass [g mol-1], dt [s].
        def _areal_from_profile(profile: Tensor) -> Tensor:
            if profile.dim() == 0:
                return profile
            return torch.sum(profile, dim=-1)

        n_molar_mass = self._to_tensor(self.atomic_weight_nitrogen, dtype=dt_tensor.dtype)
        p_molar_mass = self._to_tensor(self.atomic_weight_phosphorus, dtype=dt_tensor.dtype)

        # References: Barber (1995); Griffiths and York (2020); von Bloh et al. (2018).
        # Mathematical expressions:
        # U_NO3 = min(U_NO3,kin, m_NO3 / (M_N * dt)).
        # U_NH4 = min(U_NH4,kin, m_NH4 / (M_N * dt)).
        # U_N = min(U_NO3 + U_NH4, (S_N,max - S_N) / dt).
        # U_P = min(U_P,kin, P_available / (M_P * dt), P_capacity / dt).
        # Parameter meanings and units: U_N [mol N m-2 s-1], U_P [mol P m-2 s-1],
        # m_N [g N m-2], M_N [g N mol-1], P pools [g P m-2], M_P [g P mol-1].
        p_uptake_rate = rates["phosphorus_uptake_rate"]
        positive_dt = dt_tensor > 0.0
        safe_dt = torch.where(positive_dt, dt_tensor, torch.ones_like(dt_tensor))

        no3_available_mass = self._to_tensor(
            validated_state["no3_nitrogen"],
            dtype=no3_uptake_by_layer.dtype,
        )
        nh4_available_mass = self._to_tensor(
            validated_state["nh4_nitrogen"],
            dtype=nh4_uptake_by_layer.dtype,
        )
        no3_rate_limit = no3_available_mass / n_molar_mass / safe_dt
        nh4_rate_limit = nh4_available_mass / n_molar_mass / safe_dt
        no3_rate_limit = torch.where(positive_dt, no3_rate_limit, no3_uptake_by_layer)
        nh4_rate_limit = torch.where(positive_dt, nh4_rate_limit, nh4_uptake_by_layer)
        no3_uptake_by_layer = torch.minimum(no3_uptake_by_layer, no3_rate_limit)
        nh4_uptake_by_layer = torch.minimum(nh4_uptake_by_layer, nh4_rate_limit)

        n_uptake_rate = _areal_from_profile(
            no3_uptake_by_layer + nh4_uptake_by_layer
        )
        if self.demand_regulation_enabled:
            plant_n_storage = self._to_tensor(
                validated_state["plant_n_storage"],
                dtype=n_uptake_rate.dtype,
            )
            remaining_n_capacity = torch.clamp(
                self._to_tensor(self.max_storage_n, dtype=n_uptake_rate.dtype)
                - plant_n_storage,
                min=0.0,
            )
            n_capacity_rate_limit = remaining_n_capacity / safe_dt
            n_capacity_rate_limit = torch.where(
                positive_dt,
                n_capacity_rate_limit,
                n_uptake_rate,
            )
            limited_n_uptake_rate = torch.minimum(
                n_uptake_rate,
                n_capacity_rate_limit,
            )
            n_scale_factor = torch.where(
                n_uptake_rate > 0.0,
                limited_n_uptake_rate / n_uptake_rate,
                torch.zeros_like(limited_n_uptake_rate),
            )
            while n_scale_factor.dim() < no3_uptake_by_layer.dim():
                n_scale_factor = n_scale_factor.unsqueeze(-1)
            no3_uptake_by_layer = no3_uptake_by_layer * n_scale_factor
            nh4_uptake_by_layer = nh4_uptake_by_layer * n_scale_factor
        limited_n_uptake_rate = _areal_from_profile(
            no3_uptake_by_layer + nh4_uptake_by_layer
        )
        no3_uptake_areal_mol = _areal_from_profile(no3_uptake_by_layer)
        nh4_uptake_areal_mol = _areal_from_profile(nh4_uptake_by_layer)

        available_p_mass = self._to_tensor(
            validated_state["available_phosphorus"],
            dtype=p_uptake_rate.dtype,
        )
        available_rate_limit = available_p_mass / p_molar_mass / safe_dt
        available_rate_limit = torch.where(
            positive_dt,
            available_rate_limit,
            p_uptake_rate,
        )
        p_rate_limit = available_rate_limit
        if self.demand_regulation_enabled:
            plant_p_storage_mass = self._to_tensor(
                validated_state["plant_p_storage"],
                dtype=p_uptake_rate.dtype,
            )
            plant_p_storage_mol = plant_p_storage_mass / p_molar_mass
            remaining_capacity_mol = torch.clamp(
                self._to_tensor(self.max_storage_p, dtype=p_uptake_rate.dtype)
                - plant_p_storage_mol,
                min=0.0,
            )
            capacity_rate_limit = remaining_capacity_mol / safe_dt
            capacity_rate_limit = torch.where(
                positive_dt,
                capacity_rate_limit,
                p_uptake_rate,
            )
            p_rate_limit = torch.minimum(p_rate_limit, capacity_rate_limit)

        limited_p_uptake_rate = torch.minimum(p_uptake_rate, p_rate_limit)
        p_scale_factor = torch.where(
            p_uptake_rate > 0.0,
            limited_p_uptake_rate / p_uptake_rate,
            torch.zeros_like(limited_p_uptake_rate),
        )
        while p_scale_factor.dim() < po4_uptake_by_layer_mol.dim():
            p_scale_factor = p_scale_factor.unsqueeze(-1)
        po4_uptake_by_layer_mol = po4_uptake_by_layer_mol * p_scale_factor

        po4_uptake_areal_mol = _areal_from_profile(po4_uptake_by_layer_mol)
        po4_uptake_areal_g = po4_uptake_areal_mol * p_molar_mass
        po4_uptake_by_layer_g = po4_uptake_by_layer_mol * p_molar_mass

        # References: Barber (1995); Marschner (2012); Feinberg (2019).
        # Mathematical expression: x_dot = N_h r_h and delta_x = x_dot * dt.
        # Parameter meanings: N_h maps root uptake channels to soil-pool losses and
        # plant-storage gains; r_h contains Michaelis-Menten uptake rates.
        stoichiometric_n = torch.tensor(
            [
                [-self.atomic_weight_nitrogen, 0.0, 0.0],
                [0.0, -self.atomic_weight_nitrogen, 0.0],
                [0.0, 0.0, -1.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            device=dt_tensor.device,
            dtype=dt_tensor.dtype,
        )
        nutrient_conservation_matrix = torch.tensor(
            [
                [1.0 / self.atomic_weight_nitrogen, 1.0 / self.atomic_weight_nitrogen, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0],
            ],
            device=dt_tensor.device,
            dtype=dt_tensor.dtype,
        )
        uptake_network = advance_reaction_network(
            name="root_nutrient_uptake_network",
            species=(
                "no3_nitrogen",
                "nh4_nitrogen",
                "available_phosphorus",
                "plant_n_storage",
                "plant_p_storage",
            ),
            reactions=(
                "no3_root_transfer",
                "nh4_root_transfer",
                "po4_root_transfer",
            ),
            stoichiometric_N=stoichiometric_n,
            rates=(
                no3_uptake_areal_mol,
                nh4_uptake_areal_mol,
                po4_uptake_areal_g,
            ),
            dt=dt_tensor,
            **self._iphs_reaction_conservation_kwargs(
                conservation_matrix=nutrient_conservation_matrix,
                default_strict=True,
            ),
        )

        # References: Barber (1995); Tinker and Nye (2000); RELIC StateManager shape contract.
        # Mathematical expression: Delta_pool(z) = -M U(z) dt for every donor layer z, with
        #   sum_z Delta_pool(z) = (N_h r_h)_pool dt exactly because U_areal = sum_z U(z);
        #   the plant receiver keeps the areal network increment.
        # Parameter meanings: U(z) is the layer uptake rate after pool, demand and capacity
        #   limits; M the molar mass (already applied on the mass-basis P channel); dt the
        #   step; the areal increment is the advance_reaction_network state delta.
        # Units: U(z) [mol N m-2 s-1] or [g P m-2 s-1]; M [g mol-1]; dt [s]; Delta [g m-2].
        # Domain/constraints: a donor pool declared with a layer axis (more dims than the areal
        #   increment) receives per-layer increments so StateManager shapes match; scalar or
        #   batch-only pools receive the areal increment. dt >= 0 keeps Delta_pool <= 0.
        # Sign convention: donor increments are non-positive; plant increments are non-negative.
        def _donor_increment(pool_state: Tensor, layer_mass_rate: Tensor, species: str) -> Tensor:
            areal_increment = uptake_network.state_delta[uptake_network.species.index(species)]
            if pool_state.dim() > areal_increment.dim():
                return -layer_mass_rate.to(dtype=dt_tensor.dtype) * dt_tensor
            return areal_increment

        no3_increment = _donor_increment(
            validated_state["no3_nitrogen"], no3_uptake_by_layer * n_molar_mass, "no3_nitrogen"
        )
        nh4_increment = _donor_increment(
            validated_state["nh4_nitrogen"], nh4_uptake_by_layer * n_molar_mass, "nh4_nitrogen"
        )
        phosphorus_increment = _donor_increment(
            validated_state["available_phosphorus"], po4_uptake_by_layer_g, "available_phosphorus"
        )

        updates = {
            "no3_nitrogen": UpdateItem(
                value=no3_increment,
                mode=UpdateMode.ACCUMULATE,
                unit="g N m-2",
                basis="per_step",
                per_timestep=True,
                element="N",
                source_functor=self.name,
            ),
            "nh4_nitrogen": UpdateItem(
                value=nh4_increment,
                mode=UpdateMode.ACCUMULATE,
                unit="g N m-2",
                basis="per_step",
                per_timestep=True,
                element="N",
                source_functor=self.name,
            ),
            "available_phosphorus": UpdateItem(
                value=phosphorus_increment,
                mode=UpdateMode.ACCUMULATE,
                unit="g P m-2",
                basis="per_step",
                per_timestep=True,
                element="P",
                source_functor=self.name,
            ),
            "plant_n_storage": UpdateItem(
                value=uptake_network.state_delta[
                    uptake_network.species.index("plant_n_storage")
                ],
                mode=UpdateMode.ACCUMULATE,
                unit="mol N m-2",
                basis="per_step",
                per_timestep=True,
                element="N",
                source_functor=self.name,
            ),
            "plant_p_storage": UpdateItem(
                value=uptake_network.state_delta[
                    uptake_network.species.index("plant_p_storage")
                ],
                mode=UpdateMode.ACCUMULATE,
                unit="g P m-2",
                basis="per_step",
                per_timestep=True,
                element="P",
                source_functor=self.name,
            ),
            "nitrogen_uptake_rate": UpdateItem(
                value=limited_n_uptake_rate,
                mode=UpdateMode.REPLACE,
                unit="mol N m-2 s-1",
                basis="per_second",
                per_timestep=False,
                source_functor=self.name,
            ),
            "phosphorus_uptake_rate": UpdateItem(
                value=po4_uptake_areal_g,
                mode=UpdateMode.REPLACE,
                unit="g P m-2 s-1",
                basis="per_second",
                per_timestep=False,
                source_functor=self.name,
            ),
            "no3_uptake_by_layer": UpdateItem(
                value=no3_uptake_by_layer,
                mode=UpdateMode.REPLACE,
                unit="mol N m-2 s-1",
                basis="per_second",
                per_timestep=False,
                source_functor=self.name,
            ),
            "nh4_uptake_by_layer": UpdateItem(
                value=nh4_uptake_by_layer,
                mode=UpdateMode.REPLACE,
                unit="mol N m-2 s-1",
                basis="per_second",
                per_timestep=False,
                source_functor=self.name,
            ),
            "po4_uptake_by_layer": UpdateItem(
                value=po4_uptake_by_layer_g,
                mode=UpdateMode.REPLACE,
                unit="g P m-2 s-1",
                basis="per_second",
                per_timestep=False,
                source_functor=self.name,
            ),
        }
        return updates

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Return unit-consistent local nutrient transfer residuals."""
        nitrogen_residual = 0.0
        phosphorus_residual = 0.0
        no3_update = output_updates.get("no3_nitrogen")
        nh4_update = output_updates.get("nh4_nitrogen")
        plant_n_update = output_updates.get("plant_n_storage")
        if all(
            isinstance(update, UpdateItem)
            for update in (no3_update, nh4_update, plant_n_update)
        ):
            no3_delta = torch.as_tensor(no3_update.value)
            nh4_delta = torch.as_tensor(nh4_update.value)
            plant_delta = torch.as_tensor(plant_n_update.value)
            if no3_delta.dim() > plant_delta.dim():
                no3_delta = torch.sum(no3_delta, dim=-1)
            if nh4_delta.dim() > plant_delta.dim():
                nh4_delta = torch.sum(nh4_delta, dim=-1)
            residual = (
                no3_delta / self.atomic_weight_nitrogen
                + nh4_delta / self.atomic_weight_nitrogen
                + plant_delta
            )
            nitrogen_residual = float(
                torch.max(torch.abs(residual)).detach().cpu().item()
            )

        soil_p_update = output_updates.get("available_phosphorus")
        plant_p_update = output_updates.get("plant_p_storage")
        if isinstance(soil_p_update, UpdateItem) and isinstance(
            plant_p_update, UpdateItem
        ):
            soil_p_delta = torch.as_tensor(soil_p_update.value)
            plant_p_delta = torch.as_tensor(plant_p_update.value)
            # Layered donor increments are summed over the layer axis before closing
            # against the areal plant receiver (same rule as the nitrogen ledger).
            if soil_p_delta.dim() > plant_p_delta.dim():
                soil_p_delta = torch.sum(soil_p_delta, dim=-1)
            residual = soil_p_delta + plant_p_delta
            phosphorus_residual = float(
                torch.max(torch.abs(residual)).detach().cpu().item()
            )

        return {
            "nitrogen": nitrogen_residual,
            "phosphorus": phosphorus_residual,
        }

    
    def __init__(
        self,
        name: str = "root_nutrient_uptake",
        config: Optional[Dict] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize root nutrient uptake process.
        
        Mathematical Expression:
            Initialization: P_0 = {Vmax_N, Vmax_P, Km_N, Km_P, k_root, Q_10, ...}
            Parameter validation: Vmax > 0, Km > 0, k_root > 0, Q_10 > 1
        
        Parameters:
            name : str
                Functor name/identifier
            config : Optional[Dict]
                Configuration dictionary (optional)
            device : Optional[torch.device]
                Target torch device for tensors
        """
        config_dict = dict(config) if isinstance(config, dict) else {}
        # Prepare metadata with conservation laws
        from relic.core.functor import (
            ExecutionPhase,
            FunctorMetadata,
            ProcessCategory,
            SpatialScale,
            VariableKind,
        
    TemporalScale,
)
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Root nutrient uptake with Michaelis-Menten kinetics and environmental controls",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="U = Vmax * C/(Km + C) * f_T * f_W * W_root",
            references=[
                "Barber, S.A. (1995). Soil Nutrient Bioavailability: A Mechanistic Approach",
                "Tinker, P.B. & Nye, P.H. (2000). Solute Movement in the Rhizosphere",
                "Chapin, F.S. et al. (1993). Evolution of suites of traits",
                "Theory doc: docs/source/theory/root_nutrient_uptake.md"
            ],
            parameters={
                "pedosphere.nutrient_cycling.vmax_n": "Maximum N uptake rate [mol m-2 s-1]",
                "pedosphere.nutrient_cycling.vmax_p": "Maximum P uptake rate [mol m-2 s-1]",
                "pedosphere.nutrient_cycling.km_n": "Michaelis constant for N [mol m-3]",
                "pedosphere.nutrient_cycling.km_p": "Michaelis constant for P [mol m-3]",
                "pedosphere.nutrient_cycling.max_storage_n": "Maximum plant N storage [mol m-2]",
                "pedosphere.nutrient_cycling.max_storage_p": "Maximum plant P storage [mol m-2]",
                "pedosphere.nutrient_cycling.root_decay_coeff": "Root distribution decay [m-1]",
                "pedosphere.nutrient_cycling.q10_value": (
                    "Temperature Q10 coefficient [dimensionless]"
                ),
                "pedosphere.nutrient_cycling.reference_temperature": (
                    "Q10 reference temperature [K]"
                ),
                "pedosphere.nutrient_cycling.demand_regulation_enabled": (
                    "Switch of the plant-demand limit on nutrient uptake [dimensionless]"
                ),
                "pedosphere.nutrient_cycling.water_stress_enabled": (
                    "Switch of the water-stress effect on nutrient uptake [dimensionless]"
                ),
                "pedosphere.numerical.temperature_clamp_min": (
                    "Lower bound of the soil temperature in the uptake response [K]"
                ),
                "pedosphere.numerical.temperature_clamp_max": (
                    "Upper bound of the soil temperature in the uptake response [K]"
                ),
                "validation.conservation.strict_conservation": (
                    "Switch that makes a conservation violation raise [dimensionless]"
                ),
            },
            input_variables=[
                "no3_nitrogen",
                "nh4_nitrogen",
                "available_phosphorus",
                "soil_water_content",
                "soil_temperature",
                "soil_depth",
                "plant_n_storage",
                "plant_p_storage",
                "root_density",
                "plant_n_demand",
                "plant_p_demand",
                "field_capacity",
                "wilting_point",
            ],
            output_variables=[
                "no3_nitrogen",
                "nh4_nitrogen",
                "available_phosphorus",
                "plant_n_storage",
                "plant_p_storage",
                "nitrogen_uptake_rate",
                "phosphorus_uptake_rate",
                "no3_uptake_by_layer",
                "nh4_uptake_by_layer",
                "po4_uptake_by_layer",
            ],
            conservation_laws=["nitrogen", "phosphorus"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
            spatial_dimensions=1,
            temporal_scale=TemporalScale.SECOND,
            coupling_interfaces=["biosphere_interface"]
        )
        
        super().__init__(
            name=name,
            process_name="root_nutrient_uptake",
            config=config_dict,
            metadata=metadata,
            device=device,
        )
        self.optional_input_variables = []
        self.output_specifications = {
            output_name: FluxSpecification(
                name=output_name,
                units=units,
                direction=FluxDirection.OUTPUT,
                flux_type=FluxType.UPTAKE,
                description="Layer-resolved root nutrient uptake rate",
                is_rate=True,
            )
            for output_name, units in (
                ("no3_uptake_by_layer", "mol N m-2 s-1"),
                ("nh4_uptake_by_layer", "mol N m-2 s-1"),
                ("po4_uptake_by_layer", "g P m-2 s-1"),
            )
        }

        default_dtype = torch.get_default_dtype()

        def _numeric_value(value) -> float:
            if torch.is_tensor(value):
                scalar_value = torch.as_tensor(value).detach().reshape(-1)[0].tolist()
                return float(scalar_value)
            return float(value)

        def _as_tensor_or_parameter(value):
            if isinstance(value, nn.Parameter):
                return value
            return self._to_tensor(value, dtype=default_dtype)

        def _fetch_positive(namespace: str, key: str, message: str, min_value: float = 0.0):
            try:
                value = self.get_parameter_value(
                    key,
                    namespace=namespace,
                    required=True,
                )
            except ValueError as exc:
                raise ValueError(message) from exc
            if _numeric_value(value) <= min_value:
                raise ValueError(message)
            return value
        
        # Get parameters from parameter manager (no nn.Parameter usage)
        # Validation happens in get_parameter_value
        self.vmax_n = _as_tensor_or_parameter(
            _fetch_positive(
                "pedosphere.nutrient_cycling",
                "vmax_n",
                "Maximum uptake rates must be positive",
                min_value=0.0,
            )
        )
        self.vmax_p = _as_tensor_or_parameter(
            _fetch_positive(
                "pedosphere.nutrient_cycling",
                "vmax_p",
                "Maximum uptake rates must be positive",
                min_value=0.0,
            )
        )
        self.km_n = _as_tensor_or_parameter(
            _fetch_positive(
                "pedosphere.nutrient_cycling",
                "km_n",
                "Michaelis constants must be positive",
                min_value=0.0,
            )
        )
        self.km_p = _as_tensor_or_parameter(
            _fetch_positive(
                "pedosphere.nutrient_cycling",
                "km_p",
                "Michaelis constants must be positive",
                min_value=0.0,
            )
        )
        self.root_decay_coeff = _as_tensor_or_parameter(
            _fetch_positive(
                "pedosphere.nutrient_cycling",
                "root_decay_coeff",
                "Root decay coefficient must be positive",
                min_value=0.0,
            )
        )
        self.q10_value = _numeric_value(
            _fetch_positive(
                "pedosphere.nutrient_cycling",
                "q10_value",
                "Q10 value must be greater than 1",
                min_value=1.0,
            )
        )
        self.reference_temperature = _numeric_value(
            self.get_parameter_value(
                "reference_temperature",
                namespace="pedosphere.nutrient_cycling",
                required=True,
            )
        )
        self.q10_interval = float(
            get_physical_constant(
                "relic.physical.constants.temperature.q10_interval"
            )
        )
        self.water_stress_enabled = bool(
            self.get_parameter_value(
                "water_stress_enabled",
                namespace="pedosphere.nutrient_cycling",
                required=True,
            )
        )
        self.demand_regulation_enabled = bool(
            self.get_parameter_value(
                "demand_regulation_enabled",
                namespace="pedosphere.nutrient_cycling",
                required=True,
            )
        )
        self.max_storage_n = _numeric_value(
            self.get_parameter_value(
                "max_storage_n",
                namespace="pedosphere.nutrient_cycling",
                required=True,
            )
        )
        self.max_storage_p = _numeric_value(
            self.get_parameter_value(
                "max_storage_p",
                namespace="pedosphere.nutrient_cycling",
                required=True,
            )
        )
        
        # Get atomic weights and physical constants
        self.atomic_weight_nitrogen = float(
            get_physical_constant(
                "relic.physical.constants.molecular_weights.nitrogen"
            )
        )
        self.atomic_weight_phosphorus = float(
            get_physical_constant(
                "relic.physical.constants.molecular_weights.phosphorus"
            )
        )
        # Initialize diagnostic variables
        self._last_n_uptake = 0.0
        self._last_p_uptake = 0.0
        self._last_water_stress = 1.0
        self._last_temp_response = 1.0

    def _to_tensor(self, value, dtype=None):
        """
        Safely convert value to tensor, handling cases where value is already a tensor.
        
        Parameters:
            value : Union[float, int, Tensor]
                Value to convert
            dtype : Optional[torch.dtype]
                Target dtype (default: torch.float32)
                
        Returns:
            Tensor : Converted tensor on correct device
        """
        if isinstance(value, torch.Tensor):
            target_dtype = value.dtype if dtype is None else dtype
            return value.to(device=self.device, dtype=target_dtype)
        if dtype is None:
            dtype = torch.get_default_dtype()
        return torch.tensor(value, device=self.device, dtype=dtype)

    def _compute_rate_core(self, state: StateDict) -> FluxDict:
        """
        Compute root nutrient uptake rates for N and P.
        
        Mathematical Expression:
            Layer uptake: U_i = Vmax * C_i / (Km + C_i) * f_env * W_i
            Total uptake: U_total = Sigma_i U_i  
            Demand regulation: U_actual = min(U_total, Demand)
            
        Parameters:
            state : StateDict
                Current soil and plant state
                Required keys:
                - "no3_nitrogen" : Tensor [g N m^-2] - NO3-N pool by layer
                - "nh4_nitrogen" : Tensor [g N m^-2] - NH4-N pool by layer
                - "available_phosphorus" : Tensor [g P m^-2] - Available P pool
                - "soil_water_content" : Tensor [m^3 m^-3] - Soil moisture by layer
                - "soil_temperature" : Tensor [K] - Soil temperature
                - "plant_n_storage" : Tensor [mol N m^-2] - Current plant N storage
                - "plant_p_storage" : Tensor [g P m^-2] - Current plant P storage
                - "soil_depth" : Tensor [m] - Depth of soil layers
                - "root_density" : Tensor [kg m^-3] - Root density by layer
                - "plant_n_demand" : Tensor [mol N m^-2 s^-1]
                - "plant_p_demand" : Tensor [mol P m^-2 s^-1]
                - "field_capacity" : Tensor [m^3 m^-3]
                - "wilting_point" : Tensor [m^3 m^-3]
                The state is the one input channel (declared inputs only).
                
        Returns:
            Internal molar-rate mapping consumed by ``forward`` before public
            phosphorus ports are converted to their canonical mass basis:
                - "nitrogen_uptake_rate" : Tensor [mol N m^-2 s^-1] - Total N uptake rate
                - "phosphorus_uptake_rate" : Tensor [mol P m^-2 s^-1] - Total P uptake rate
                - "no3_uptake_by_layer" : Tensor [mol N m^-2 s^-1] - NO3 uptake by layer
                - "nh4_uptake_by_layer" : Tensor [mol N m^-2 s^-1] - NH4 uptake by layer
                - "po4_uptake_by_layer" : Tensor [mol P m^-2 s^-1] - PO4 uptake by layer
        """
        try:
            soil_depth = self._to_tensor(state["soil_depth"])
            # References: Barber (1995); NIST Atomic Data for Nitrogen.
            # Mathematical expression: C_N = m_N / (M_N * z).
            # Parameters: m_N [g N m-2], M_N [g N mol-1], z [m], C_N [mol N m-3].
            n_molweight = self._to_tensor(self.atomic_weight_nitrogen, dtype=soil_depth.dtype)
            p_molweight = self._to_tensor(self.atomic_weight_phosphorus, dtype=soil_depth.dtype)
            no3_pool = self._to_tensor(state["no3_nitrogen"], dtype=soil_depth.dtype)
            nh4_pool = self._to_tensor(state["nh4_nitrogen"], dtype=soil_depth.dtype)
            no3_conc = no3_pool / n_molweight / soil_depth
            nh4_conc = nh4_pool / n_molweight / soil_depth
            
            # Get canonical PO4 pool [g P/m2] and convert to concentration [mol P/m3]
            po4_pool = self._to_tensor(state["available_phosphorus"])
            po4_conc = po4_pool / p_molweight / soil_depth
            
            soil_water = self._to_tensor(state["soil_water_content"])
            temperature = self._to_tensor(state["soil_temperature"])
            root_density = self._to_tensor(state["root_density"])
            plant_n_demand = self._to_tensor(state["plant_n_demand"])
            plant_p_demand = self._to_tensor(state["plant_p_demand"])
            field_capacity = self._to_tensor(state["field_capacity"])
            wilting_point = self._to_tensor(state["wilting_point"])
            
            # Compute environmental response functions
            temp_response = self._compute_temperature_response(temperature)
            water_stress = self._compute_water_stress_response(soil_water, field_capacity, wilting_point)
            
            # Compute root weight distribution (normalized to sum=1)
            root_weights = self._compute_root_weight_distribution(soil_depth, root_density)
            
            # References: Barber (1995); Tinker and Nye (2000); Chapin et al. (1993).
            # Equations: U_i = Vmax * C_i / (Km + C_i) * f_T * f_W * W_i.
            # Parameters: Vmax (max uptake), Km (half-saturation), C_i (layer nutrient),
            #             f_T (temperature response), f_W (water stress), W_i (root weight).
            # Compute layer-by-layer uptake using Michaelis-Menten kinetics
            
            # Nitrogen uptake (NO3 + NH4)
            total_n_conc = no3_conc + nh4_conc  # [mol N m^-3]
            
            # Convert parameters to tensors for computation using helper to avoid warnings
            vmax_n_tensor = self._to_tensor(self.vmax_n)
            vmax_p_tensor = self._to_tensor(self.vmax_p)
            km_n_tensor = self._to_tensor(self.km_n)
            km_p_tensor = self._to_tensor(self.km_p)
            
            n_uptake_potential = self._michaelis_menten_uptake(
                vmax_n_tensor, total_n_conc, km_n_tensor
            ) * temp_response * water_stress * root_weights  # [mol N m^-2 s^-1]
            
            # Phosphorus uptake
            p_uptake_potential = self._michaelis_menten_uptake(
                vmax_p_tensor, po4_conc, km_p_tensor
            ) * temp_response * water_stress * root_weights  # [mol P m^-2 s^-1]
            
            # Sum across all layers
            total_n_uptake = torch.sum(n_uptake_potential, dim=-1)  # [mol N m^-2 s^-1]
            total_p_uptake = torch.sum(p_uptake_potential, dim=-1)  # [mol P m^-2 s^-1]
            
            # References: Barber (1995); Chapin et al. (1993).
            # Equation: U_actual = min(U_total, Demand).
            # Parameters: U_total and Demand [mol m-2 s-1]. Storage-capacity and
            # available-pool limits are applied on the time-step basis in forward.
            # Apply plant demand regulation
            if self.demand_regulation_enabled:
                n_uptake_actual = torch.minimum(total_n_uptake, plant_n_demand)
                p_uptake_actual = torch.minimum(total_p_uptake, plant_p_demand)
                
                # Scale down layer uptakes proportionally
                n_scale_factor = torch.where(
                    total_n_uptake > 0,
                    n_uptake_actual / total_n_uptake,
                    torch.zeros_like(n_uptake_actual)
                )
                p_scale_factor = torch.where(
                    total_p_uptake > 0,
                    p_uptake_actual / total_p_uptake,
                    torch.zeros_like(p_uptake_actual)
                )
                
                n_uptake_by_layer = n_uptake_potential * n_scale_factor.unsqueeze(-1)
                p_uptake_by_layer = p_uptake_potential * p_scale_factor.unsqueeze(-1)
            else:
                n_uptake_actual = total_n_uptake
                p_uptake_actual = total_p_uptake
                n_uptake_by_layer = n_uptake_potential
                p_uptake_by_layer = p_uptake_potential
            
            # Partition N uptake between NO3 and NH4 based on relative concentrations
            positive_n_concentration = total_n_conc > 0.0
            no3_fraction = torch.where(
                positive_n_concentration,
                no3_conc / torch.where(
                    positive_n_concentration,
                    total_n_conc,
                    torch.ones_like(total_n_conc),
                ),
                torch.zeros_like(no3_conc),
            )
            nh4_fraction = torch.where(
                positive_n_concentration,
                nh4_conc / torch.where(
                    positive_n_concentration,
                    total_n_conc,
                    torch.ones_like(total_n_conc),
                ),
                torch.zeros_like(nh4_conc),
            )
            
            no3_uptake_by_layer = n_uptake_by_layer * no3_fraction  # [mol N m^-2 s^-1]
            nh4_uptake_by_layer = n_uptake_by_layer * nh4_fraction  # [mol N m^-2 s^-1]
            
            # Store diagnostic information
            self._last_n_uptake = n_uptake_actual.detach()
            self._last_p_uptake = p_uptake_actual.detach()
            self._last_water_stress = water_stress.detach()
            self._last_temp_response = temp_response.detach()
            
            return {
                "nitrogen_uptake_rate": n_uptake_actual,
                "phosphorus_uptake_rate": p_uptake_actual,
                "no3_uptake_by_layer": no3_uptake_by_layer,
                "nh4_uptake_by_layer": nh4_uptake_by_layer, 
                "po4_uptake_by_layer": p_uptake_by_layer,
            }
            
        except ValueError:
            raise
        except KeyError as e:
            raise ValueError(f"Required state variable missing: {e}") from e
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            logger.error(f"Root nutrient uptake calculation failed: {e}")
            raise RuntimeError(f"Root nutrient uptake calculation failed: {e}") from e
    
    def _michaelis_menten_uptake(
        self, 
        vmax: Tensor, 
        concentration: Tensor, 
        km: Tensor
    ) -> Tensor:
        """
        Compute Michaelis-Menten uptake rate.
        
        Mathematical Expression:
            V = Vmax * [S] / (Km + [S])
            
        Parameters:
            vmax : Tensor [mol m^-2 s^-1]
                Maximum uptake rate
            concentration : Tensor [mol m^-3]
                Nutrient concentration
            km : Tensor [mol m^-3]
                Michaelis constant
                
        Returns:
            Tensor [mol m^-2 s^-1] : Uptake rate
        """
        # References: Barber (1995); Tinker and Nye (2000).
        # Equation: U = Vmax * C / (Km + C).
        # Parameters: Vmax (max uptake), C (concentration), Km (half-saturation).
        if not bool(torch.all(torch.isfinite(concentration)).item()):
            raise ValueError("Nutrient concentration must be finite")
        if bool(torch.any(concentration < 0.0).item()):
            raise ValueError("Nutrient concentration must be non-negative")
        if not bool(torch.all(torch.isfinite(km)).item()) or bool(torch.any(km <= 0.0).item()):
            raise ValueError("Michaelis constant must be finite and positive")
        return vmax * concentration / (km + concentration)
    
    def _compute_temperature_response(self, temperature: Tensor) -> Tensor:
        """
        Compute temperature response function using Q10 model.
        
        Mathematical Expression:
            f_T(T) = Q_1_0^((T - T_ref)/T_interval)
            
        Parameters:
            temperature : Tensor [K]
                Soil temperature
                
        Returns:
            Tensor [dimensionless] : Temperature response factor
        """
        if not bool(torch.all(torch.isfinite(temperature)).item()):
            raise ValueError("Soil temperature must be finite")
        minimum_temperature = self._to_tensor(
            self.get_parameter_value(
                "temperature_clamp_min",
                namespace="pedosphere.numerical",
                required=True,
            ),
            dtype=temperature.dtype,
        )
        maximum_temperature = self._to_tensor(
            self.get_parameter_value(
                "temperature_clamp_max",
                namespace="pedosphere.numerical",
                required=True,
            ),
            dtype=temperature.dtype,
        )
        if bool(
            torch.any(
                (temperature < minimum_temperature)
                | (temperature > maximum_temperature)
            ).item()
        ):
            raise ValueError("Soil temperature is outside the calibrated model domain")

        temp_diff = (temperature - self.reference_temperature) / self.q10_interval
        response = self.q10_value ** temp_diff
        if not bool(torch.all(torch.isfinite(response)).item()):
            raise ValueError("Q10 temperature response must be finite")
        return response
    
    def _compute_water_stress_response(
        self, 
        soil_water: Tensor, 
        field_capacity: Tensor, 
        wilting_point: Tensor
    ) -> Tensor:
        """
        Compute water stress response function.
        
        Mathematical Expression:
            f_W(theta) = max(0, min(1, (theta - theta_wilt) / (theta_fc - theta_wilt)))
            
        Parameters:
            soil_water : Tensor [m^3 m^-3]
                Soil water content
            field_capacity : Tensor [m^3 m^-3] 
                Field capacity
            wilting_point : Tensor [m^3 m^-3]
                Wilting point
                
        Returns:
            Tensor [dimensionless] : Water stress response factor
        """
        if not self.water_stress_enabled:
            return torch.ones_like(soil_water)
        
        # Calculate available water
        available_water_capacity = field_capacity - wilting_point
        available_water = soil_water - wilting_point
        
        if bool(torch.any(available_water_capacity <= 0.0).item()):
            raise ValueError("field_capacity must be greater than wilting_point")
        
        # Water stress response
        water_response = available_water / available_water_capacity
        
        # Clamp to [0, 1] range
        water_response = torch.clamp(water_response, min=0.0, max=1.0)
        
        return water_response
    
    def _compute_root_weight_distribution(
        self, 
        soil_depth: Tensor, 
        root_density: Tensor
    ) -> Tensor:
        """
        Compute root weight distribution with depth.
        
        Mathematical Expression:
            W(z) = W_0 * exp(-k_root * z) * rho_root(z)
            where W_0 is surface weight, normalized so Sigma W_i = 1
            
        Parameters:
            soil_depth : Tensor [m]
                Depth of soil layers (center depths)
            root_density : Tensor [kg m^-3]
                Root density by layer
                
        Returns:  
            Tensor [dimensionless] : Normalized root weight distribution
        """
        # Exponential decay with depth - use helper method to avoid warning
        root_decay_tensor = self._to_tensor(self.root_decay_coeff)
        decay_factor = torch.exp(-root_decay_tensor * soil_depth)
        
        # Include root density
        weight_unnormalized = decay_factor * root_density
        
        # Normalize weights to sum to 1
        total_weight = torch.sum(weight_unnormalized, dim=-1, keepdim=True)
        positive_weight = total_weight > 0.0
        safe_total_weight = torch.where(
            positive_weight,
            total_weight,
            torch.ones_like(total_weight),
        )
        weight_normalized = torch.where(
            positive_weight,
            weight_unnormalized / safe_total_weight,
            torch.zeros_like(weight_unnormalized),
        )
        
        return weight_normalized
    
    def get_diagnostic_info(self) -> Dict[str, float]:
        """
        Get diagnostic information about the last uptake calculation.
        
        Returns:
            Dict[str, float] : Diagnostic information
        """
        def _diag_scalar(value, reducer: str = "first") -> float:
            values = torch.as_tensor(value).detach().reshape(-1)
            if values.numel() == 0:
                return 0.0
            if reducer == "mean":
                mean_value = values.mean().tolist()
                return float(mean_value)
            first_value = values[0].tolist()
            return float(first_value)

        return {
            "last_n_uptake_mol_m2_s": _diag_scalar(self._last_n_uptake, reducer="first"),
            "last_p_uptake_mol_m2_s": _diag_scalar(self._last_p_uptake, reducer="first"),
            "last_water_stress_factor": _diag_scalar(self._last_water_stress, reducer="mean"),
            "last_temperature_response": _diag_scalar(self._last_temp_response, reducer="first"),
            "vmax_n_mol_m2_s": _diag_scalar(self.vmax_n, reducer="first"),
            "vmax_p_mol_m2_s": _diag_scalar(self.vmax_p, reducer="first"),
            "km_n_mol_m3": _diag_scalar(self.km_n, reducer="first"),
            "km_p_mol_m3": _diag_scalar(self.km_p, reducer="first"),
            "root_decay_coeff_m": _diag_scalar(self.root_decay_coeff, reducer="first"),
        }
