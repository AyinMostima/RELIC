"""
Denitrification Process Module for RELIC

This module implements denitrification (NO3- -> NO2- -> NO -> N2O -> N2)
using Michaelis-Menten kinetics with anaerobic controls.

Scientific Basis:
-----------------
Denitrification is an anaerobic respiratory process requiring organic carbon
as electron donor and nitrate as electron acceptor.

References:
- Parton, W.J. et al. (1996). Generalized model for N2 and N2O production
- Li, C. et al. (1992). DNDC denitrification formulations

Mathematical Expression:
-----------------------
R_denit = Vmax * [NO3] / (Km + [NO3]) * f_T * f_anaerobic * f_C * f_pH

Product distribution: N2O vs N2 depends on NO3 concentration and pH
"""

from typing import Any, Dict, List, Optional
import logging
import torch

from relic.core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    SpatialScale,
    SubFlowAccuracy,
    VariableKind,
    TemporalScale,
)
from relic.core.abstractions import StateDict, FluxDict
from relic.core.state import UpdateItem, UpdateMode

from relic.core.functor import BiologicalProcessFunctor
from relic.computation.operators.iphs import advance_reaction_network
from relic.utils.physical_constants import get_physical_constant
from relic.scientific.pedosphere.api_contract import (
    PedosphereContractMixin,
    FluxDirection,
    FluxSpecification,
    FluxType,
    StandardPedosphereOutputs,
    standardize_output_keys,
    validate_functor_output,
)

logger = logging.getLogger(__name__)


class DenitrificationProcess(PedosphereContractMixin, BiologicalProcessFunctor):
    """
    Denitrification process functor with anaerobic controls.
    
    References:
    - Parton et al. (1996) N2 and N2O production 
    - Li et al. (1992) DNDC denitrification model
    
    Mathematical Expression:
    -----------------------
    R_denit = Vmax * [NO3]/(Km + [NO3]) * f_T * f_anaerobic * f_C * f_pH
    
    N2O:N2 ratio depends on environmental conditions
    """

    PHASE8_RUNTIME_REPLACEMENT_STATUS = "canonical_iphs_runtime"
    
    def __init__(self, name: str = "denitrification", config: Optional[Dict] = None):
        if config is None:
            config = {}

        # Prepare metadata first
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Denitrification with anaerobic carbon requirements",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="R = Vmax * [NO3]/(Km + [NO3]) * f_anaerobic * f_C",
            references=[
                "Parton, W.J. et al. (1996). N2 and N2O production",
                "Li, C. et al. (1992). DNDC denitrification model"
            ],
            parameters={
                "pedosphere.denitrification.vmax_denitrification": (
                    "Maximum denitrification rate [g N m-2 s-1]"
                ),
                "pedosphere.denitrification.km_no3": "Half-saturation for NO3 [g N m-2]",
                "pedosphere.denitrification.carbon_half_saturation": (
                    "Half-saturation constant of labile carbon [g C m-2]"
                ),
                "pedosphere.physical.wfps_denitrification_threshold": (
                    "Water-filled pore space above which the anaerobic factor rises [dimensionless]"
                ),
                "pedosphere.denitrification.anaerobic_transition_width": (
                    "Width of the anaerobic transition in water-filled pore space [dimensionless]"
                ),
                "pedosphere.denitrification.minimum_anaerobic_factor": (
                    "Minimum anaerobic activity factor under aerobic conditions [dimensionless]"
                ),
                "pedosphere.denitrification.ph_minimum_factor": (
                    "Minimum pH response factor [dimensionless]"
                ),
                "pedosphere.denitrification.min_rate_threshold": (
                    "Lower bound of the temperature factor f_T [dimensionless]"
                ),
                "pedosphere.denitrification.max_rate_factor": (
                    "Maximum rate amplification factor [dimensionless]"
                ),
                "pedosphere.denitrification.base_n2o_fraction": (
                    "Minimum N2O fraction of the denitrified N [dimensionless]"
                ),
                "pedosphere.denitrification.max_n2o_fraction": (
                    "Maximum N2O fraction of the denitrified N [dimensionless]"
                ),
                "pedosphere.denitrification.n2o_no3_threshold": (
                    "NO3 threshold of the N2O enhancement [g N m-2]"
                ),
                "pedosphere.denitrification.n2o_no3_enhancement": (
                    "NO3 enhancement factor of the N2O fraction [dimensionless]"
                ),
                "pedosphere.denitrification.n2o_ph_reference": (
                    "Reference pH of the N2O fraction [dimensionless]"
                ),
                "pedosphere.denitrification.n2o_ph_sensitivity": (
                    "PH sensitivity of the N2O fraction [dimensionless]"
                ),
                "pedosphere.denitrification.n2o_ph_effect_min": (
                    "Minimum pH effect on the N2O fraction [dimensionless]"
                ),
                "pedosphere.denitrification.n2o_ph_effect_max": (
                    "Maximum pH effect on the N2O fraction [dimensionless]"
                ),
                "pedosphere.physical.q10_denitrification": (
                    "Q10 of denitrification [dimensionless]"
                ),
                "pedosphere.physical.reference_temperature": (
                    "Reference temperature of the Q10 response [K]"
                ),
                "pedosphere.physical.ph_optimum_denitrification": (
                    "Optimum pH of denitrification [dimensionless]"
                ),
                "pedosphere.physical.ph_response_width": (
                    "Width of the Gaussian pH response [dimensionless]"
                ),
                "validation.conservation.conservation_tolerance": (
                    "General relative tolerance of the conservation checks [dimensionless]"
                ),
                "validation.conservation.strict_conservation": (
                    "Switch that makes a conservation violation raise [dimensionless]"
                ),
            },
            input_variables=[
                "no3_nitrogen",
                "labile_carbon",
                "water_filled_pore_space",
                "soil_temperature",
                "soil_ph",
            ],
            output_variables=[
                "no3_nitrogen",
                "denitrification_flux",
                "n2o_flux",
                "n2_production",
                "diagnostics.anaerobic_factor",
                "diagnostics.carbon_factor",
                "diagnostics.n2o_fraction",
                "diagnostics.temperature_response",
                "diagnostics.ph_factor",
                "diagnostics.denitrification.step_seconds",
            ],
            conservation_laws=["nitrogen"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
            spatial_dimensions=0,
            temporal_scale=TemporalScale.SECOND,
            # References: Hairer, Norsett and Wanner (1993) Solving Ordinary Differential
            #   Equations I, 2nd ed., Sect. II.3 (explicit Euler converges with order 1);
            #   Hairer, Lubich and Wanner (2006) Geometric Numerical Integration, 2nd ed.,
            #   Sect. II.4-II.5 (the splitting order is bound to the sub-flow orders).
            # Mathematical expression: one call over [t, t + dt] is one explicit Euler
            #   step x(t + dt) = x(t) + N_h r_h(x(t)) dt of x_dot = N_h r_h(x),
            #   x = [NO3, N2O, N2], with the drivers held at their call-start values;
            #   while the donor limit is inactive, R_den(NO3) dt < NO3, i.e.
            #   Vmax f_T f_anoxic f_C f_pH dt < Km_NO3 + NO3, the local error is O(dt^2),
            #   so the declared sub-flow accuracy is ORDER_1.
            # Parameter meanings: x contains the NO3 pool and N2O/N2 export amounts;
            #   N_h is nitrogen-mass stoichiometry; r_h contains the N2O/N2 rates;
            #   R_den is the Michaelis-Menten rate of _compute_rate_core;
            #   Vmax is maximum rate and Km_NO3 is the nitrate half-saturation;
            #   f_T, f_anoxic, f_C, f_pH are temperature, anaerobic, carbon and pH
            #   modifiers; the clipped f_N2O is the N2O product fraction;
            #   dt the call length (the simulator sub-step length when it sub-steps).
            # Units: x, Km_NO3 [g N m-2]; r_h, R_den, Vmax [g N m-2 s-1];
            #   N_h, f_N2O and response factors [1];
            #   dt [s].
            # Domain/constraints: for positive NO3 the limit is inactive when
            #   Vmax f_T f_anoxic f_C f_pH dt < Km_NO3 and binds once
            #   NO3 < Vmax f_T f_anoxic f_C f_pH dt - Km_NO3; NO3 = 0 is an exact
            #   fixed point. A binding positive-pool call exhausts the numerical pool
            #   while the exact donor remains positive. T11 records a coarse-step
            #   equal-error plateau, not loss of asymptotic order (CR-A-04 T11,
            #   test/core/splitting_sub_flow_order.test.py). The clipped N2O fraction
            #   only partitions the released nitrogen between N2O and N2; the NO3
            #   sub-flow itself is smooth in NO3. Simulator sub-steps repeat the Euler
            #   step (order 1 in the sub-step length).
            # Sign convention: positive R_den removes NO3 and exports N2O/N2;
            #   the numerical order itself is dimensionless and signless.
            sub_flow_accuracy=SubFlowAccuracy.ORDER_1,
            coupling_interfaces=["anaerobic_respiration", "gas_emissions"]
        )
        # Initialize base functor
        super().__init__(name=name, config=config, metadata=metadata, device=None)
        self.metadata = metadata
        self._config_raw = dict(config) if config else {}
        self.logger = logger
        self._execution_count = 0
        # Every input is required: a missing labile carbon, water-filled pore space,
        # temperature or pH raises; no environment default is substituted.
        self.optional_input_variables = []

        # Load parameters from parameter manager  
        # References: Parton et al. (1996), Li et al. (1992), Tiedje (1988)
        # Mathematical expressions: Michaelis-Menten kinetics, Q10 temperature response, anaerobic controls
        # Parameters: vmax [g N m-2 s-1], Km [g N m-2], WFPS threshold [dimensionless], Q10 [dimensionless]
        
        self.vmax = self.require_scalar("vmax_denitrification", namespace="pedosphere.denitrification", units="g N m-2 s-1")
        
        self.km_no3 = self.require_scalar("km_no3", namespace="pedosphere.denitrification", units="g N m-2")
        
        self.wfps_threshold = self.require_scalar(
            "wfps_denitrification_threshold", namespace="pedosphere.physical", units="dimensionless"
        )

        self.q10 = self.require_scalar("q10_denitrification", namespace="pedosphere.physical", units="dimensionless")
        self.q10_interval = get_physical_constant(
            "relic.physical.constants.temperature.q10_interval"
        )
        
        # Additional environmental response parameters
        self.temperature_reference = self.require_scalar("reference_temperature", namespace="pedosphere.physical", units='K')
        
        self.anaerobic_transition_width = self.require_scalar("anaerobic_transition_width", namespace="pedosphere.denitrification",
            units="dimensionless",
        )

        self.minimum_anaerobic_factor = self.require_scalar("minimum_anaerobic_factor", namespace="pedosphere.denitrification",
            units="dimensionless",
        )

        self.carbon_half_saturation = self.require_scalar("carbon_half_saturation", namespace="pedosphere.denitrification", units="g C m-2")

        self.ph_optimum = self.require_scalar("ph_optimum_denitrification", namespace="pedosphere.physical",
            units="dimensionless",
        )

        self.ph_response_width = self.require_scalar("ph_response_width", namespace="pedosphere.physical", units="dimensionless")

        self.min_rate_threshold = self.require_scalar("min_rate_threshold", namespace="pedosphere.denitrification",
            units="dimensionless",
        )
        self.max_rate_factor = self.require_scalar("max_rate_factor", namespace="pedosphere.denitrification", units="dimensionless")
        self.ph_minimum_factor = self.require_scalar("ph_minimum_factor", namespace="pedosphere.denitrification", units="dimensionless")
        self.n2o_no3_threshold = self.require_scalar("n2o_no3_threshold", namespace="pedosphere.denitrification", units="g N m-2")
        self.n2o_ph_reference = self.require_scalar("n2o_ph_reference", namespace="pedosphere.denitrification",
            units="dimensionless",
        )
        self.n2o_ph_sensitivity = self.require_scalar("n2o_ph_sensitivity", namespace="pedosphere.denitrification",
            units="dimensionless",
        )
        self.n2o_ph_effect_min = self.require_scalar("n2o_ph_effect_min", namespace="pedosphere.denitrification",
            units="dimensionless",
        )
        self.n2o_ph_effect_max = self.require_scalar("n2o_ph_effect_max", namespace="pedosphere.denitrification",
            units="dimensionless",
        )
        self.base_n2o_fraction = self.require_scalar("base_n2o_fraction", namespace="pedosphere.denitrification", units="dimensionless")
        self.n2o_no3_enhancement = self.require_scalar("n2o_no3_enhancement", namespace="pedosphere.denitrification", units="dimensionless")
        self.max_n2o_fraction = self.require_scalar("max_n2o_fraction", namespace="pedosphere.denitrification", units="dimensionless")
        
        # Diagnostics totals are accumulated in tensor buffers to avoid
        # per-step Python scalar synchronization in the hot path.
        self._diagnostic_totals = {
            "total_denitrified": torch.zeros((), dtype=torch.float64),
            "total_n2o_produced": torch.zeros((), dtype=torch.float64),
            "total_n2_produced": torch.zeros((), dtype=torch.float64),
        }

    def _accumulate_diagnostic_total(self, key: str, value: Any) -> None:
        """Accumulate diagnostics without introducing graph-coupled scalar sync."""
        with torch.no_grad():
            target_device = self._diagnostic_totals[key].device
            if torch.is_tensor(value):
                delta = value.detach().to(device=target_device, dtype=torch.float64).sum()
            else:
                delta = torch.as_tensor(value, device=target_device, dtype=torch.float64)
            self._diagnostic_totals[key] = self._diagnostic_totals[key] + delta
    
    def get_dependencies(self) -> List[str]:
        """
        Get list of dependencies for this functor.

        Nitrification is the only named scheduler dependency. Labile carbon is
        consumed through the state interface so the compiled graph can select
        exactly one carbon decomposition closure.

        Returns:
            List[str]: List of dependency names
        """
        return self._merge_declared_dependencies(["nitrification"])

    @staticmethod
    def _ensure_tensor(
        value: Any,
        dtype: torch.dtype = torch.float32,
        template: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convert inputs to tensors and broadcast to template shapes when needed."""
        target_dtype = template.dtype if template is not None else dtype
        target_device = template.device if template is not None else None
        if isinstance(value, torch.Tensor):
            tensor = value.to(dtype=target_dtype, device=target_device)
        else:
            tensor = torch.as_tensor(value, dtype=target_dtype, device=target_device)
        if template is not None and tensor.shape != template.shape:
            try:
                tensor = torch.broadcast_to(tensor, template.shape)
            except RuntimeError as exc:
                raise ValueError(
                    "Environmental input shape "
                    f"{tuple(tensor.shape)} cannot broadcast to state shape "
                    f"{tuple(template.shape)}."
                ) from exc
        return tensor
    
# get_parameter_value removed - using BiologicalProcess implementation
    
    def _compute_rate_core(self, state: StateDict) -> FluxDict:
        """
        Compute denitrification rate using Michaelis-Menten kinetics.
        
        Parameters:
        -----------
        state : StateDict
            Current system state; the pools and the soil drivers are declared
            required state inputs (the one input channel)
            
        Returns:
        --------
        FluxDict
            Denitrification rates and products
        """
        # References: Tiedje (1988); Knowles (1982); Parton et al. (1996); Li et al. (1992).
        # Mathematical expression: R=Vmax*NO3/(Km_NO3+NO3)*f_T*f_anoxic*f_C*f_pH;
        #   f_T=clip(Q10^((T-T_ref)/dT10),fT_min,fT_max); f_C=C/(Km_C+C);
        #   f_anoxic=max(clip((WFPS-WFPS_thr)/width,0,1),f_anoxic_min);
        #   f_pH=clip(exp(-((pH-pH_opt)/w_pH)^2),f_pH_min,1).
        # Parameter meanings: Vmax is potential denitrification flux; Km values are
        #   substrate half-saturations; response parameters describe the reduced soil-driver law.
        # Units: R,Vmax [g N m-2 s-1]; NO3,Km_NO3 [g N m-2]; C,Km_C [g C m-2];
        #   T,T_ref,dT10 [K]; WFPS, pH coordinates and all response factors [1].
        # Domain/constraints: Positive denominators and physical substrate/driver
        #   inputs; the drivers are required state inputs (no default is substituted).
        # Sign convention: R is nitrate consumption magnitude; this helper computes rates only.
        # Extract pools and conditions (state variables must exist)
        no3_n = state['no3_nitrogen']
        missing = [
            key for key in ("labile_carbon", "water_filled_pore_space", "soil_temperature", "soil_ph")
            if key not in state or state[key] is None
        ]
        if missing:
            raise ValueError(
                "DenitrificationProcess requires the inputs " + ", ".join(missing)
                + "; no environment default is substituted"
            )
        labile_c = self._ensure_tensor(
            state["labile_carbon"],
            template=no3_n,
        )
        wfps = self._ensure_tensor(
            state["water_filled_pore_space"],
            template=no3_n,
        )
        soil_temp = self._ensure_tensor(
            state["soil_temperature"],
            template=no3_n,
        )
        soil_ph = self._ensure_tensor(
            state["soil_ph"],
            template=no3_n,
        )
        
        # Temperature response
        # References: Tiedje (1988), Q10 temperature dependence for denitrification.
        # Mathematical expression: f_T = clip(Q10^((T - T_ref)/T_interval), f_T,min, f_T,max).
        # Parameter meanings: Q10 = q10_denitrification; T_ref = reference_temperature;
        #   T_interval = the Q10 interval physical constant; f_T,min = min_rate_threshold and
        #   f_T,max = max_rate_factor, RELIC-convention bounds of the factor.
        # Units: T, T_ref, T_interval [K]; Q10, f_T, f_T,min, f_T,max [dimensionless].
        # Domain/constraints: f_T,min <= f_T <= f_T,max; with the default Q10 = 2.2 the floor
        #   1e-12 is inactive at every physical soil temperature.
        # Sign convention: f_T is a non-negative multiplier of the denitrification rate.
        q10_value = self._ensure_tensor(self.q10, template=soil_temp)
        reference_temp = self._ensure_tensor(self.temperature_reference, template=soil_temp)
        q10_interval = self._ensure_tensor(self.q10_interval, template=soil_temp)
        temp_response = torch.pow(
            q10_value,
            (soil_temp - reference_temp) / q10_interval,
        )

        # Apply parameter-managed bounds
        temp_min = self._ensure_tensor(self.min_rate_threshold, template=soil_temp)
        temp_max = self._ensure_tensor(self.max_rate_factor, template=soil_temp)
        temp_response = torch.clamp(temp_response, min=temp_min, max=temp_max)
        
        # Anaerobic conditions favor denitrification
        # References: Del Grosso et al. (2000) GBC 14:1045 and Parton et al. (1996) GBC
        #   10:401 (DAYCENT drives denitrification with water-filled pore space); Linn and
        #   Doran (1984) SSSAJ 48:1267 (aeration limits aerobic activity above 60 percent
        #   WFPS); Bateman and Baggs (2005) Biol. Fertil. Soils 41:379.
        # Mathematical expression: f_anoxic = max(clip((WFPS - WFPS_thr)/w, 0, 1), f_min).
        # Parameter meanings: WFPS the water-filled pore space; WFPS_thr =
        #   pedosphere.physical.wfps_denitrification_threshold; w = anaerobic_transition_width
        #   and f_min = minimum_anaerobic_factor, RELIC conventions (the linear ramp stands in
        #   for the DAYCENT arctangent WFPS function, which is not implemented).
        # Units: WFPS, WFPS_thr, w, f_min, f_anoxic [1].
        # Domain/constraints: 0 <= WFPS <= 1 (checked in forward); w > 0; f_anoxic is
        #   continuous and non-decreasing in WFPS, equal to f_min up to WFPS_thr + f_min w and
        #   to 1 from WFPS_thr + w.
        # Sign convention: f_anoxic in [f_min, 1] scales the nonnegative rate.
        wfps_threshold = self._ensure_tensor(self.wfps_threshold, template=wfps)
        transition_width = self._ensure_tensor(self.anaerobic_transition_width, template=wfps)
        anaerobic_floor = self._ensure_tensor(self.minimum_anaerobic_factor, template=wfps)
        anaerobic_factor = torch.maximum(
            torch.clamp((wfps - wfps_threshold) / transition_width, min=0.0, max=1.0),
            anaerobic_floor,
        )
        
        # Carbon availability (electron donor requirement)
        # References: Carbon limitation of denitrification, electron donor kinetics
        # Mathematical expression: f_C = C/(Km_C + C) Michaelis-Menten kinetics
        # Parameters: carbon half-saturation constant
        carbon_factor = labile_c / (self.carbon_half_saturation + labile_c)
        
        # pH effect (optimal around pH 6-8)
        # References: Knowles (1982), pH optimum for denitrifying bacteria
        # Mathematical expression: f_pH = exp(-((pH - pH_opt)/width)^2) with bounds
        # Parameters: pH optimum, response width, minimum factor
        ph_factor_raw = torch.exp(-((soil_ph - self.ph_optimum) / self.ph_response_width) ** 2)
        ph_factor = torch.clamp(
            ph_factor_raw,
            min=self._ensure_tensor(self.ph_minimum_factor, template=soil_ph),
            max=self._ensure_tensor(1.0, template=soil_ph),
        )
        
        # Denitrification rate
        # References: Parton et al. (1996); Li et al. (1992).
        # Equations: R_den = Vmax * NO3 / (Km_NO3 + NO3) * f_T * f_anaerobic * f_C * f_pH.
        # Parameters: Vmax (max denitrification rate), Km_NO3 (NO3 half-saturation),
        #             NO3 (nitrate pool), f_T (temperature modifier),
        #             f_anaerobic (anaerobic modifier), f_C (labile C modifier),
        #             f_pH (pH modifier).
        denitrification_rate = (self.vmax * no3_n / (self.km_no3 + no3_n)
                               * temp_response * anaerobic_factor * carbon_factor * ph_factor)
        
        # Product distribution (N2O vs N2)
        # References: Firestone & Davidson (1989), pH and NO3 control of N2O:N2 ratio
        # Mathematical expression: f_N2O = base + enhancement * NO3_effect * pH_effect
        # Parameters: base N2O fraction, NO3 enhancement, pH sensitivity, thresholds
        
        # NO3 concentration effect on N2O production
        no3_threshold = self._ensure_tensor(
            self.n2o_no3_threshold, template=no3_n
        )
        no3_effect = no3_n / (no3_n + no3_threshold)
        
        # pH effect on N2O:N2 ratio (lower pH favors N2O)
        ph_reference = self._ensure_tensor(
            self.n2o_ph_reference, template=soil_ph
        )
        ph_sensitivity = self._ensure_tensor(
            self.n2o_ph_sensitivity, template=soil_ph
        )
        ph_effect_min = self._ensure_tensor(
            self.n2o_ph_effect_min, template=soil_ph
        )
        ph_effect_max = self._ensure_tensor(
            self.n2o_ph_effect_max, template=soil_ph
        )
        ph_effect = torch.clamp(
            (ph_reference - soil_ph) / ph_sensitivity, min=ph_effect_min, max=ph_effect_max
        )
        
        # Calculate N2O fraction
        base_n2o = self._ensure_tensor(self.base_n2o_fraction, template=no3_n)
        no3_enhancement = self._ensure_tensor(self.n2o_no3_enhancement, template=no3_n)
        max_n2o = self._ensure_tensor(self.max_n2o_fraction, template=no3_n)

        n2o_fraction = base_n2o + no3_enhancement * no3_effect * ph_effect
        n2o_fraction = torch.clamp(n2o_fraction, min=base_n2o, max=max_n2o)
        
        n2o_production = denitrification_rate * n2o_fraction
        n2_production = denitrification_rate * (1.0 - n2o_fraction)
        
        return {
            'denitrification_rate': denitrification_rate,
            'n2o_production': n2o_production,
            'n2_production': n2_production,
            'no3_consumption': denitrification_rate,
            'anaerobic_factor': anaerobic_factor,
            'carbon_factor': carbon_factor,
            'ph_factor': ph_factor,
            'n2o_fraction': n2o_fraction,
            'temperature_response': temp_response
        }
    
    def forward(self, state: StateDict, time: float, dt: torch.Tensor | float, **kwargs) -> FluxDict:
        """
        Forward pass adapter - calls compute_rate and formats output for RELICSimulator.
        """
        # References: Li et al. (1992); Parton et al. (1996); Tiedje (1988).
        # Mathematical expression: R_exec=max(min(R_den,NO3/dt),0);
        #   r=[R_exec*f_N2O,R_exec*(1-f_N2O)];
        #   Delta[NO3,N2O,N2]=[[-1,-1],[1,0],[0,1]]*r*dt.
        # Parameter meanings: R_den and f_N2O come from the one rate core;
        #   NO3 is the beginning-step donor pool and dt is the positive step length.
        # Units: pools and internal reaction increments [g N m-2]; r and public gas
        #   outputs [g N m-2 s-1]; dt [s]; stoichiometric coefficients [1].
        # Domain/constraints: One finite positive scalar dt; the reaction network checks
        #   [1,1,1]*N=0. Gas outputs remain rates after the internal amount ledger is formed.
        # Sign convention: Nitrate loses R_exec*dt; gases leave the soil positively;
        #   nitrate ACCUMULATE is per_step while public gas rates are not stock advances.
        # References: SCM contract water_filled_pore_space ([0, 1]); RELIC fail-fast input
        #   policy (a reader never modifies the state value it reads).
        # Mathematical expression: raise unless 0 <= WFPS_i <= 1 for every element i.
        # Parameter meanings: WFPS the water-filled pore space of the anaerobic factor.
        # Units: WFPS [1].
        # Domain/constraints: entry protection of direct kernel calls and external states;
        #   NaN fails the check; no clipping; a missing WFPS raises in the rate core.
        # Sign convention: not applicable (a domain check, not a flux).
        if 'water_filled_pore_space' in state and state['water_filled_pore_space'] is not None:
            wfps_entry = torch.as_tensor(state['water_filled_pore_space'])
            if not bool(torch.all((wfps_entry >= 0.0) & (wfps_entry <= 1.0))):
                raise ValueError(
                    'DenitrificationProcess input water_filled_pore_space must lie in [0, 1]; '
                    'it is not clipped'
                )
        self._execution_count += 1
        
        # The soil drivers are required state inputs (metadata input_variables); the
        # rate core reads them from the state, which raises for a missing one (no
        # pedosphere.environment_defaults substitute, no second input channel).
        rates = self._compute_rate_core(state)
        
        # Limit by available NO3 and time step (state variable must exist)
        no3_n = state['no3_nitrogen']
        # Get denitrification rate from compute_rate
        denitrification_rate = rates['denitrification_rate']
        
        n2o_fraction = rates['n2o_fraction']

        dt_tensor = torch.as_tensor(dt, dtype=no3_n.dtype, device=no3_n.device)
        if dt_tensor.numel() != 1 or not bool(torch.isfinite(dt_tensor).all()):
            raise ValueError("DenitrificationProcess dt must be one finite scalar.")
        if bool(dt_tensor <= 0.0):
            raise ValueError("DenitrificationProcess dt must be positive.")
        safe_dt = dt_tensor

        # References: Li et al. (1992); Parton et al. (1996); Tiedje (1988).
        # Mathematical expression:
        #   x_dot = N_h r_h,
        #   x = [NO3, N2O, N2]^T,
        #   N_h = [[-1, -1], [1, 0], [0, 1]],
        #   r_h = [r_N2O, r_N2]^T,
        #   r_N2O = min(R_den, NO3 / dt) * f_N2O,
        #   r_N2 = min(R_den, NO3 / dt) * (1 - f_N2O).
        # Parameter meanings:
        #   R_den is the ParameterManager-backed denitrification rate from
        #   _compute_rate_core, f_N2O is the environment-dependent product
        #   fraction, dt is the simulator step, and NO3 is nitrate nitrogen.
        substrate_limited_rate = torch.minimum(denitrification_rate, no3_n / safe_dt)
        actual_denitrification_rate = torch.clamp(substrate_limited_rate, min=0.0)
        n2o_production = actual_denitrification_rate * n2o_fraction
        n2_production = actual_denitrification_rate * (1.0 - n2o_fraction)
        reaction_advance = advance_reaction_network(
            name=f"{self.name}.denitrification_reaction",
            species=("no3_nitrogen", "n2o_flux", "n2_production"),
            reactions=("no3_to_n2o", "no3_to_n2"),
            stoichiometric_N=(
                (-1.0, -1.0),
                (1.0, 0.0),
                (0.0, 1.0),
            ),
            rates=(n2o_production, n2_production),
            dt=safe_dt,
            **self._iphs_reaction_conservation_kwargs(
                conservation_matrix=torch.ones(
                    (1, 3),
                    device=no3_n.device,
                    dtype=no3_n.dtype,
                ),
            ),
        )
        delta_no3_actual = reaction_advance.state_delta[0]
        n2o_step = reaction_advance.state_delta[1]
        n2_step = reaction_advance.state_delta[2]

        # Update diagnostics totals (sum over spatial dimensions when needed).
        denitrified_step = -delta_no3_actual
        self._accumulate_diagnostic_total("total_denitrified", denitrified_step)
        self._accumulate_diagnostic_total("total_n2o_produced", n2o_step)
        self._accumulate_diagnostic_total("total_n2_produced", n2_step)
        
        # Prepare updates dictionary for conservation check
        # CONSERVATION FIX: Conservation check needs deltas, not rates
        updates = {
            'no3_nitrogen': delta_no3_actual,  # g N/m2 (delta for conservation)
            # Use declared output keys so metadata-first conservation aggregation
            # includes gaseous exports in local nitrogen closure:
            # dNO3 + dN2O + dN2 = 0
            'n2o_flux': n2o_production * dt,  # g N/m2 (delta)
            'n2_production': n2_production * dt,  # g N/m2 (delta)
        }
        self._enforce_forward_conservation(state, updates)
        
        output = {
            'no3_nitrogen': UpdateItem(
                value=delta_no3_actual,
                mode=UpdateMode.ACCUMULATE,
                unit="g N m-2",
                basis="per_step",
                per_timestep=True,
                element="N",
                source_functor=self.name,
            ),
            'denitrification_flux': UpdateItem(
                value=actual_denitrification_rate,
                mode=UpdateMode.REPLACE,
                unit="g N m-2 s-1",
                basis="per_second",
                per_timestep=False,
                source_functor=self.name,
            ),
            'n2o_flux': UpdateItem(
                value=n2o_production,
                mode=UpdateMode.ACCUMULATE,
                unit="g N m-2 s-1",
                basis="per_second",
                per_timestep=False,
                source_functor=self.name,
            ),
            'n2_production': UpdateItem(
                value=n2_production,
                mode=UpdateMode.REPLACE,
                unit="g N m-2 s-1",
                basis="per_second",
                per_timestep=False,
                source_functor=self.name,
            ),
            'diagnostics.anaerobic_factor': rates['anaerobic_factor'],
            'diagnostics.carbon_factor': rates['carbon_factor'],
            'diagnostics.n2o_fraction': rates['n2o_fraction'],
            'diagnostics.temperature_response': rates['temperature_response'],
            'diagnostics.ph_factor': rates['ph_factor'],
            # Declared step length so the public ledger can integrate the gas rates.
            'diagnostics.denitrification.step_seconds': safe_dt.detach().reshape(()),
        }

        output = standardize_output_keys(output)
        validation_errors = validate_functor_output(output, self.get_output_specifications())
        if validation_errors:
            self.logger.warning(
                "Output validation errors in %s: %s",
                self.name,
                "; ".join(validation_errors),
            )
        return output

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Audit the returned NO3/N2O/N2 proposal as one closed per-step nitrogen ledger.

        References:
          - Manzoni and Porporato (2009), process-level conservation diagnostics.
          - Li et al. (1992); Parton et al. (1996); Tiedje (1988), denitrification products.
        Mathematical expression:
          r = Delta_NO3 + (F_N2O + F_N2) * dt.
        Parameter meanings:
          Delta_NO3 is the returned per-step nitrate increment, F_N2O and F_N2 the returned gas rates; dt is the step length the proposal declares under
          diagnostics.denitrification.step_seconds.
        Units:
          pool increments and r [g N m-2]; exported rates [g N m-2 s-1]; dt [s].
        Domain/constraints:
          all listed outputs must be present as UpdateItems with the declared units and
          per_timestep flags; raw per-step deltas from the forward-internal ledger are
          summed directly. Missing keys raise ValueError instead of reporting zero.
        Sign convention:
          pool increments are positive when the pool grows; exports are positive out of
          the soil pool, so closure means r = 0.
        """
        residual = self.step_ledger_residual(
            output_updates,
            increments=("no3_nitrogen",),
            rates=("n2o_flux", "n2_production"),
            step_key="diagnostics.denitrification.step_seconds",
            increment_unit="g N m-2",
            rate_unit="g N m-2 s-1",
        )
        magnitude = float(residual.detach().abs().max().item())
        return {law: magnitude for law in self.metadata.conservation_laws}

    def get_output_specifications(self) -> Dict[str, Any]:
        return {
            "denitrification_flux": StandardPedosphereOutputs.DENITRIFICATION,
            "n2o_flux": StandardPedosphereOutputs.N2O_EMISSION,
            "n2_production": FluxSpecification(
                name="n2_production",
                units="g N m-2 s-1",
                direction=FluxDirection.OUTPUT,
                flux_type=FluxType.EMISSION,
                description="N2 production rate from denitrification",
                is_rate=True,
            ),
        }


__all__ = ['DenitrificationProcess']

