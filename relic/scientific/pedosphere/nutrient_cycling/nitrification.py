"""
Nitrification Process Module for RELIC

This module implements ammonia oxidation (nitrification) process using
Michaelis-Menten kinetics with environmental controls.

Scientific Basis:
-----------------
Nitrification is a two-step microbial process: NH4+ -> NO2- -> NO3-.
The model uses an effective single-step kinetics representation with
environmental modifiers.

References:
- Parton, W.J. et al. (1996). Generalized model for N2 and N2O production
- Li, C. et al. (1992). Model of nitrous oxide evolution from soil  
- Prosser, J.I. (1989). Autotrophic nitrification in bacteria

Mathematical Expression:
-----------------------
V = Vmax * [NH4] / (Km + [NH4]) * f_T * f_theta * f_O2 * f_pH

Where:
- V: nitrification rate [g N m^-2 s^-1]
- Vmax: maximum rate [g N m^-2 s^-1]
- [NH4]: ammonium concentration [g N m^-2]
- Km: half-saturation constant [g N m^-2]
- f_T, f_theta, f_O2, f_pH: environmental modifiers [dimensionless]
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
from relic.core.state import UpdateItem, UpdateMode
from relic.core.abstractions import StateDict, FluxDict

from relic.utils.unified_response_functions import (
    q10_temperature_response
)
from relic.core.functor import BiologicalProcessFunctor
from relic.computation.operators.iphs import advance_reaction_network
from relic.scientific.pedosphere.api_contract import (
    PedosphereContractMixin,
    StandardPedosphereOutputs,
    standardize_output_keys,
    validate_functor_output,
)

logger = logging.getLogger(__name__)


class NitrificationProcess(PedosphereContractMixin, BiologicalProcessFunctor):
    """
    Nitrification process functor using Michaelis-Menten kinetics.
    
    References:
    - Parton et al. (1996) N2 and N2O production model
    - Li et al. (1992) DNDC nitrification formulations
    - Prosser (1989) Autotrophic nitrification theory
    
    Mathematical Expression:
    -----------------------
    R_nit = Vmax * [NH4] / (Km + [NH4]) * f_T * f_O2(WFPS) * f_pH
    
    N2O production:
    R_n2o = R_nit * f_N2O (a fixed fraction, n2o_factor)
    
    Units:
    - Nitrification rate: g N m^-2 s^-1
    - NH4 concentration: g N m^-2
    """

    PHASE8_RUNTIME_REPLACEMENT_STATUS = "canonical_iphs_runtime"
    
    def __init__(self, name: str = "nitrification", config: Optional[Dict] = None):
        if config is None:
            config = {}
        # Prepare metadata first to avoid init warnings and pass to base class
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Nitrification process with environmental controls",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="V = Vmax * [NH4]/(Km + [NH4]) * f_env",
            references=[
                "Parton, W.J. et al. (1996). N2 and N2O production model",
                "Li, C. et al. (1992). DNDC nitrification formulations"
            ],
            parameters={
                "pedosphere.nitrification.vmax_nitrification": (
                    "Maximum nitrification rate [g N m-2 s-1]"
                ),
                "pedosphere.nitrification.km_nh4": "Half-saturation for NH4 [g N m-2]",
                "pedosphere.physical.wfps_nitrification_optimum": (
                    "Water-filled pore space above which oxygen limits nitrification [dimensionless]"
                ),
                "pedosphere.nitrification.minimum_oxygen_factor": (
                    "Floor of the oxygen response factor f_O2 [dimensionless]"
                ),
                "pedosphere.nitrification.extreme_temperature_factor": (
                    "Activity factor outside the temperature range [dimensionless]"
                ),
                "pedosphere.nitrification.temperature_minimum": (
                    "Lower temperature limit of nitrification [K]"
                ),
                "pedosphere.nitrification.temperature_maximum": (
                    "Upper temperature limit of nitrification [K]"
                ),
                "pedosphere.nitrification.ph_minimum_factor": (
                    "Minimum pH response factor [dimensionless]"
                ),
                "pedosphere.nitrification.min_rate_threshold": (
                    "Lower bound of the temperature factor f_T [dimensionless]"
                ),
                "pedosphere.nitrification.max_rate_factor": (
                    "Upper bound of the temperature factor f_T [dimensionless]"
                ),
                "pedosphere.nitrification.n2o_factor": (
                    "Fraction of nitrified N converted to N2O [dimensionless]"
                ),
                "pedosphere.physical.q10_nitrification": "Q10 of nitrification [dimensionless]",
                "pedosphere.physical.reference_temperature": (
                    "Reference temperature of the Q10 response [K]"
                ),
                "pedosphere.physical.ph_optimum_nitrification": (
                    "Optimum pH of nitrification [dimensionless]"
                ),
                "pedosphere.physical.ph_response_width": (
                    "Width of the Gaussian pH response [dimensionless]"
                ),
                "pedosphere.numerical.unity_multiplier": (
                    "Unit multiplier of the rate normalization [dimensionless]"
                ),
                "pedosphere.numerical.epsilon": (
                    "Denominator floor of the nitrification statistics [dimensionless]"
                ),
                "validation.conservation.conservation_tolerance": (
                    "General relative tolerance of the conservation checks [dimensionless]"
                ),
                "validation.conservation.strict_conservation": (
                    "Switch that makes a conservation violation raise [dimensionless]"
                ),
            },
            input_variables=[
                "nh4_nitrogen",
                "soil_temperature",
                "water_filled_pore_space",
                "soil_ph",
            ],
            output_variables=[
                "nh4_nitrogen",
                "no3_nitrogen",
                "nitrification_flux",
                "n2o_flux",
                "diagnostics.temperature_response",
                "diagnostics.oxygen_factor",
                "diagnostics.ph_factor",
                "diagnostics.nitrification.step_seconds",
                "diagnostics.nitrification.substrate_limiter_residual",
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
            #   x = [NH4, NO3, N2O], with the drivers held at their call-start values;
            #   while the donor limit is inactive, R_nit(NH4) dt < NH4, i.e.
            #   Vmax f_T f_O2 f_pH dt < Km + NH4, the local error is O(dt^2), so the
            #   declared sub-flow accuracy is ORDER_1.
            # Parameter meanings: x contains NH4/NO3 pools and the N2O export amount;
            #   N_h is the constant nitrogen-mass stoichiometry; r_h contains its
            #   NO3 and N2O reaction rates; R_nit is the _compute_rate_core rate;
            #   Vmax is the maximum rate; Km is the NH4 half-saturation pool;
            #   f_T, f_O2 and f_pH are temperature, oxygen and pH modifiers;
            #   dt is the call length (the sub-step length under sub-stepping).
            # Units: x, Km [g N m-2]; r_h, R_nit, Vmax [g N m-2 s-1];
            #   N_h and factors [1]; dt [s].
            # Domain/constraints: for positive NH4 the limit is inactive when
            #   Vmax f_T f_O2 f_pH dt < Km and binds when NH4 < Vmax f_T f_O2 f_pH dt - Km;
            #   NH4 = 0 is an exact fixed point. A binding positive-pool call empties
            #   the numerical pool while the exact donor
            #   remains positive. T11 records a coarse-step equal-error plateau
            #   (not a loss of asymptotic first-order convergence as dt -> 0;
            #   test/core/splitting_sub_flow_order.test.py). Simulator sub-steps repeat
            #   the Euler step (order 1 in the sub-step length).
            # Sign convention: positive R_nit removes NH4 and adds NO3/N2O;
            #   the accuracy order itself is dimensionless and signless.
            sub_flow_accuracy=SubFlowAccuracy.ORDER_1,
            coupling_interfaces=["soil_chemistry", "gas_emissions"]
        )
        # Initialize base functor with unified interface (no extra kwargs)
        super().__init__(name=name, config=config, metadata=metadata, device=None)
        self.logger = logger
        # Every input is required: a missing soil temperature, water-filled pore space
        # or pH raises; no environment default is substituted.
        self.optional_input_variables = []
        
        # Load parameters from parameter manager
        # References: Parton et al. (1996), Stark & Firestone (1995), Prosser (1989)
        # Mathematical expressions: Michaelis-Menten kinetics, Q10 temperature response, Gaussian pH response
        # Parameters: vmax [g N m-2 s-1], Km [g N m-2], Q10 [dimensionless], pH_opt [dimensionless]
        
        def _fetch(namespace: str, key: str, **kwargs):
            return self.get_parameter_value(
                key,
                namespace=namespace,
                required=True,
                **kwargs,
            )

        self.vmax = _fetch(
            "pedosphere.nitrification",
            "vmax_nitrification",
        )

        self.km_nh4 = _fetch(
            "pedosphere.nitrification",
            "km_nh4",
        )

        self.q10 = _fetch(
            "pedosphere.physical",
            "q10_nitrification",
        )

        self.ph_optimum = _fetch(
            "pedosphere.physical",
            "ph_optimum_nitrification",
        )

        self.wfps_optimum = _fetch(
            "pedosphere.physical",
            "wfps_nitrification_optimum",
        )

        self.n2o_factor = _fetch(
            "pedosphere.nitrification",
            "n2o_factor",
        )

        # Additional physical constants for improved parameterization
        self.temperature_reference = _fetch(
            "pedosphere.physical",
            "reference_temperature",
        )

        self.extreme_temp_factor = _fetch(
            "pedosphere.nitrification",
            "extreme_temperature_factor",
        )

        self.ph_response_width = _fetch(
            "pedosphere.physical",
            "ph_response_width",
        )

        self.minimum_oxygen_factor = _fetch(
            "pedosphere.nitrification",
            "minimum_oxygen_factor",
        )

        
        # Diagnostics totals are accumulated in tensor buffers to avoid
        # per-step Python scalar synchronization in the hot path.
        self._diagnostic_totals = {
            "total_nitrified": torch.zeros((), dtype=torch.float64),
            "total_n2o_produced": torch.zeros((), dtype=torch.float64),
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

        Nitrification depends on ammonium availability from mineralization
        and soil environmental conditions.

        Returns:
            List[str]: List of dependency names
        """
        # Dependencies on mineralization for ammonium supply
        dependencies = ["mineralization"]

        return self._merge_declared_dependencies(dependencies)
    
    def _compute_rate_core(self, state: StateDict) -> FluxDict:
        """
        Compute nitrification rate using Michaelis-Menten kinetics.
        
        Parameters:
        -----------
        state : StateDict
            Current system state; the pools and the soil drivers are declared
            required state inputs (the one input channel)
            
        Returns:
        --------
        FluxDict
            Nitrification rates and state changes
        """
        # References: Prosser (1989); Stark and Firestone (1995); Parton et al. (1996).
        # Mathematical expression: R=Vmax*NH4/(Km+NH4)*f_T*f_O2*f_pH;
        #   f_T=clip(Q10^((T-T_ref)/dT10)*f_extreme,fT_min,fT_max);
        #   f_O2=1 for WFPS<=WFPS_opt, else max((1-WFPS)/(1-WFPS_opt),f_O2_min);
        #   f_pH=clip(exp(-0.5*((pH-pH_opt)/w_pH)^2),f_pH_min,1).
        # Parameter meanings: f_extreme is the declared low/high-temperature activity
        #   modifier; Km is NH4 half-saturation and Vmax potential nitrification flux.
        # Units: NH4,Km [g N m-2]; R,Vmax [g N m-2 s-1]; temperatures [K];
        #   WFPS/pH coordinates, widths and response factors [1].
        # Domain/constraints: Positive half-saturation and response scales; this reduced
        #   empirical core requires soil temperature, water-filled pore space in [0, 1]
        #   and pH (no default).
        # Sign convention: R is NH4 consumption magnitude; f_N2O*R goes to N2O and
        #   (1-f_N2O)*R to NO3 before the forward donor limit is applied.
        # Extract pools and conditions (state variable must exist)
        nh4_n = state['nh4_nitrogen']

        missing = [
            key for key in ('soil_temperature', 'water_filled_pore_space', 'soil_ph')
            if key not in state or state[key] is None
        ]
        if missing:
            raise ValueError(
                'NitrificationProcess requires the inputs ' + ', '.join(missing)
                + '; no environment default is substituted'
            )
        soil_temp = self.to_tensor(state['soil_temperature'])
        wfps = self.to_tensor(state['water_filled_pore_space'])
        soil_ph = self.to_tensor(state['soil_ph'])
        
        # Temperature response with extreme temperature handling
        # References: Stark & Firestone (1995), Q10 temperature dependence
        # Mathematical expression: f_T = Q10^((T - T_ref)/10) with extreme temperature clipping
        # Parameters: Q10 coefficient, reference temperature, extreme temperature factors
        unity_multiplier = self.get_parameter_value(
            "unity_multiplier",
            namespace="pedosphere.numerical",
            source_order=("manager",),
            required=True,
        )
        # Device rule: ParameterManager-held parameters are references that stay on
        # the manager's device, so every constant combined with a driver is
        # materialized on the driver's device and dtype (not on a fixed CPU).
        q10_value = torch.as_tensor(
            self.q10, dtype=soil_temp.dtype, device=soil_temp.device
        )
        unity_multiplier_tensor = torch.as_tensor(
            unity_multiplier, dtype=soil_temp.dtype, device=soil_temp.device
        )
        temp_reference = self.temperature_reference
        if torch.is_tensor(temp_reference):
            if temp_reference.numel() == 1:
                temp_reference = float(temp_reference.detach().reshape(-1)[0].tolist())
            else:
                temp_reference = float(temp_reference.detach().reshape(-1).mean().tolist())
        else:
            temp_reference = float(temp_reference)
        temp_response = q10_temperature_response(
            unity_multiplier_tensor,
            soil_temp,
            q10_value,
            temp_reference,
        )
        
        # Additional extreme temperature limitations
        # Below 0degC or above 50degC, drastically reduce activity
        temp_min = self.get_parameter_value(
            "temperature_minimum",
            namespace="pedosphere.nitrification",
            source_order=("manager",),
            required=True,
        )
        temp_max = self.get_parameter_value(
            "temperature_maximum",
            namespace="pedosphere.nitrification",
            source_order=("manager",),
            required=True,
        )
        
        extreme_temp_value = torch.as_tensor(
            self.extreme_temp_factor, dtype=soil_temp.dtype, device=soil_temp.device
        )
        extreme_temp_factor = torch.where(
            (soil_temp < temp_min) | (soil_temp > temp_max),
            extreme_temp_value,
            unity_multiplier_tensor,
        )
        
        # Apply clipping with parameter-managed bounds
        temp_response_min = self.get_parameter_value(
            "min_rate_threshold",
            namespace="pedosphere.nitrification",
            source_order=("manager",),
            required=True,
        )
        temp_response_max = self.get_parameter_value(
            "max_rate_factor",
            namespace="pedosphere.nitrification",
            source_order=("manager",),
            required=True,
        )
        temp_response = temp_response * extreme_temp_factor
        temp_response = torch.clamp(
            temp_response,
            min=torch.as_tensor(
                temp_response_min, dtype=temp_response.dtype, device=temp_response.device
            ),
            max=torch.as_tensor(
                temp_response_max, dtype=temp_response.dtype, device=temp_response.device
            ),
        )
        
        # References: Linn and Doran (1984) SSSAJ 48:1267 (aerobic microbial activity
        #   peaks at 60 percent WFPS and declines above it through reduced aeration);
        #   Bateman and Baggs (2005) Biol. Fertil. Soils 41:379 (nitrification is the main
        #   N2O source at 35-60 percent WFPS, denitrification at 70 percent).
        # Mathematical expression: f_O2 = 1 for WFPS <= WFPS_opt, else
        #   max((1 - WFPS) / (1 - WFPS_opt), f_O2_min).
        # Parameter meanings: WFPS the water-filled pore space; WFPS_opt =
        #   pedosphere.physical.wfps_nitrification_optimum; f_O2_min the RELIC-convention
        #   floor of the dimensionless oxygen response (a response-shape floor, not a
        #   clipped mass). The linear decline to zero at saturation is a RELIC convention;
        #   the DAYCENT water function of Parton et al. (1996) is not implemented.
        # Units: WFPS, WFPS_opt, f_O2_min, f_O2 [1].
        # Domain/constraints: 0 <= WFPS <= 1 (checked in forward); 0 < WFPS_opt < 1; f_O2 is
        #   continuous at WFPS_opt; the constants follow the device and dtype of WFPS.
        # Sign convention: f_O2 in [f_O2_min, 1] scales the nonnegative rate.
        wfps_optimum = torch.as_tensor(
            self.wfps_optimum, dtype=wfps.dtype, device=wfps.device
        )
        oxygen_limited = (1.0 - wfps) / (1.0 - wfps_optimum)
        oxygen_factor = torch.where(
            wfps <= wfps_optimum,
            torch.ones_like(oxygen_limited),
            torch.clamp(
                oxygen_limited,
                min=torch.as_tensor(
                    self.minimum_oxygen_factor,
                    dtype=oxygen_limited.dtype,
                    device=oxygen_limited.device,
                ),
            ),
        )
        
        # pH response (optimal around pH 7-8)
        # References: Prosser (1989), pH optimum for ammonia-oxidizing bacteria
        # Mathematical expression: f_pH = exp(-0.5 * ((pH - pH_opt)/width)^2)
        # Parameters: pH optimum, response width, minimum pH factor
        ph_factor = torch.exp(
            -0.5 * ((soil_ph - self.ph_optimum) / self.ph_response_width) ** 2
        )
        ph_min_factor = self.get_parameter_value(
            "ph_minimum_factor",
            namespace="pedosphere.nitrification",
            source_order=("manager",),
            required=True,
        )
        ph_min_factor_value = torch.as_tensor(
            ph_min_factor, dtype=ph_factor.dtype, device=ph_factor.device
        )
        ph_max_factor = torch.ones_like(ph_factor)
        ph_factor = torch.clamp(ph_factor, min=ph_min_factor_value, max=ph_max_factor)
        
        # Michaelis-Menten kinetics
        # References: Prosser (1989); Parton et al. (1996).
        # Equations: R_nit = Vmax * NH4 / (Km + NH4) * f_T * f_O2 * f_pH.
        # Parameters: Vmax (max nitrification rate), Km (NH4 half-saturation),
        #             NH4 (ammonium pool), f_T (temperature modifier),
        #             f_O2 (oxygen modifier), f_pH (pH modifier).
        nitrification_rate = (self.vmax * nh4_n / (self.km_nh4 + nh4_n) 
                             * temp_response * oxygen_factor * ph_factor)
        
        # N2O production as byproduct
        n2o_production = nitrification_rate * self.n2o_factor
        
        return {
            'nitrification_rate': nitrification_rate,
            'n2o_production': n2o_production,
            'nh4_consumption': nitrification_rate,
            'no3_production': nitrification_rate - n2o_production,
            'temperature_response': temp_response,
            'oxygen_factor': oxygen_factor,
            'ph_factor': ph_factor
        }
    
# get_parameter_value removed - using BiologicalProcess implementation
    
    def forward(self, state: StateDict, time: float, dt: torch.Tensor | float, **kwargs) -> FluxDict:
        """
        Forward pass adapter - calls compute_rate and formats output for RELICSimulator.
        """
        # References: Li et al. (1992); Parton et al. (1996); Prosser (1989).
        # Mathematical expression: R_exec=max(min(R_nit,NH4/dt_safe),0);
        #   r=[R_exec*(1-f_N2O),R_exec*f_N2O];
        #   Delta[NH4,NO3,N2O]=[[-1,-1],[1,0],[0,1]]*r*dt_safe.
        # Parameter meanings: R_nit is the rate-core result; f_N2O is the N2O fraction;
        #   dt_safe is current forward's dtype-epsilon lower-bounded input step.
        # Units: reaction increments [g N m-2]; public process/gas rates [g N m-2 s-1]; dt [s].
        # Domain/constraints: Use the governed Functor call; invalid dt is rejected by
        #   existing call/reaction checks. The reaction network validates [1,1,1]*N=0.
        # Sign convention: NH4 decreases, NO3 and exported N2O increase; public N2O is
        #   a rate and must not be summed directly with the per-step mineral-pool amounts.
        # References: SCM contracts nh4_nitrogen (lower_bound 0.0) and
        #   water_filled_pore_space ([0, 1]); RELIC fail-fast input policy (a reader
        #   never modifies the state value it reads).
        # Mathematical expression: raise unless NH4_i is finite and NH4_i >= 0, and
        #   0 <= WFPS_i <= 1, for every element i.
        # Parameter meanings: NH4 the ammonium nitrogen pool; WFPS the water-filled pore
        #   space of the oxygen limitation.
        # Units: NH4 [g N m-2]; WFPS [1].
        # Domain/constraints: entry protection of direct kernel calls and external
        #   states; StateManager commits clip pools to the contract bounds, so a governed
        #   run does not reach this check with a negative pool. NaN fails both checks.
        #   No clipping here; a missing WFPS raises in the rate core.
        # Sign convention: not applicable (domain checks, not fluxes).
        nh4_entry = torch.as_tensor(state['nh4_nitrogen'])
        if not bool(torch.all(torch.isfinite(nh4_entry) & (nh4_entry >= 0.0))):
            raise ValueError(
                'NitrificationProcess input nh4_nitrogen must be finite and >= 0 g N m-2; '
                'a negative pool is not clipped'
            )
        if 'water_filled_pore_space' in state and state['water_filled_pore_space'] is not None:
            wfps_entry = torch.as_tensor(state['water_filled_pore_space'])
            if not bool(torch.all((wfps_entry >= 0.0) & (wfps_entry <= 1.0))):
                raise ValueError(
                    'NitrificationProcess input water_filled_pore_space must lie in [0, 1]; '
                    'it is not clipped'
                )
        self._execution_count += 1
        
        # The soil drivers are required state inputs (metadata input_variables); the
        # rate core reads them from the state, which raises for a missing one (no
        # pedosphere.environment_defaults substitute, no second input channel).
        rates = self._compute_rate_core(state)
        
        # Get nitrification rate from compute_rate
        nitrification_rate = rates['nitrification_rate']
        
        nh4_n = state['nh4_nitrogen']

        # The donor limit divides by dt: one finite positive scalar step is required
        # (checked, not clamped by a dimensionless machine epsilon).
        dt_tensor = torch.as_tensor(dt, dtype=nh4_n.dtype, device=nh4_n.device)
        if dt_tensor.numel() != 1 or not bool(torch.isfinite(dt_tensor).all()):
            raise ValueError("NitrificationProcess dt must be one finite scalar.")
        if bool(dt_tensor <= 0.0):
            raise ValueError("NitrificationProcess dt must be positive.")
        safe_dt = dt_tensor

        # References: Li et al. (1992); Parton et al. (1996); Prosser (1989).
        # Mathematical expression:
        #   x_dot = N_h r_h,
        #   x = [NH4, NO3, N2O]^T,
        #   N_h = [[-1, -1], [1, 0], [0, 1]],
        #   r_h = [r_NO3, r_N2O]^T,
        #   r_NO3 = min(R_nit, NH4 / dt) * (1 - f_N2O),
        #   r_N2O = min(R_nit, NH4 / dt) * f_N2O.
        # Parameter meanings:
        #   R_nit is the ParameterManager-backed Michaelis-Menten nitrification
        #   rate from _compute_rate_core, f_N2O is n2o_factor, dt is the
        #   simulator step size, and NH4 is the ammonium nitrogen pool.
        substrate_limited_rate = torch.minimum(nitrification_rate, nh4_n / safe_dt)
        actual_nitrification_rate = torch.clamp(substrate_limited_rate, min=0.0)
        # References: RELIC donor-availability limit of this module (the realized rate
        #   never exceeds the NH4 present within dt); recorded like methane oxidation.
        # Mathematical expression: R_lim = R_nit - max(min(R_nit, NH4/dt), 0).
        # Parameter meanings: R_nit the kinetic nitrification demand; the realized
        #   rate is limited by the NH4 donor available within dt.
        # Units: R_lim, R_nit [g N m-2 s-1]; NH4 [g N m-2]; dt [s].
        # Domain/constraints: for NH4 >= 0 and nonnegative factors R_nit >= 0 and
        #   NH4/dt >= 0, so the nonnegativity clamp is inactive and R_lim >= 0 is the
        #   unmet demand only; it is not nitrogen and enters no ledger.
        # Sign convention: R_lim is positive when the donor limit binds.
        substrate_limiter_residual = nitrification_rate - actual_nitrification_rate
        n2o_production = actual_nitrification_rate * self.n2o_factor
        no3_production = actual_nitrification_rate - n2o_production
        reaction_advance = advance_reaction_network(
            name=f"{self.name}.nitrification_reaction",
            species=("nh4_nitrogen", "no3_nitrogen", "n2o_flux"),
            reactions=("nh4_to_no3", "nh4_to_n2o"),
            stoichiometric_N=torch.tensor(
                (
                    (-1.0, -1.0),
                    (1.0, 0.0),
                    (0.0, 1.0),
                ),
                dtype=nh4_n.dtype,
                device=nh4_n.device,
            ),
            rates=(no3_production, n2o_production),
            dt=safe_dt,
            **self._iphs_reaction_conservation_kwargs(
                conservation_matrix=torch.ones(
                    (1, 3),
                    device=nh4_n.device,
                    dtype=nh4_n.dtype,
                ),
            ),
        )
        delta_nh4 = reaction_advance.state_delta[0]
        delta_no3 = reaction_advance.state_delta[1]
        n2o_step = reaction_advance.state_delta[2]

        # Update diagnostics totals (use actual consumption after clamping).
        nitrified_step = -delta_nh4
        self._accumulate_diagnostic_total("total_nitrified", nitrified_step)
        self._accumulate_diagnostic_total("total_n2o_produced", n2o_step)
        
        # Prepare updates dictionary for conservation check
        # CONSERVATION FIX: Conservation check needs deltas, not rates
        updates = {
            'nh4_nitrogen': delta_nh4,  # g N/m2 (delta for conservation)
            'no3_nitrogen': delta_no3,  # g N/m2 (delta)
            # Use declared output key so metadata-first conservation aggregation
            # includes gaseous export in local nitrogen closure:
            # dNH4 + dNO3 + dN2O = 0
            'n2o_flux': n2o_step,  # g N/m2 (delta)
        }
        self._enforce_forward_conservation(state, updates)
        
        output = {
            'nh4_nitrogen': UpdateItem(
                value=delta_nh4,
                mode=UpdateMode.ACCUMULATE,
                unit="g N m-2",
                basis="per_step",
                per_timestep=True,
                element="N",
                source_functor=self.name,
            ),
            'no3_nitrogen': UpdateItem(
                value=delta_no3,
                mode=UpdateMode.ACCUMULATE,
                unit="g N m-2",
                basis="per_step",
                per_timestep=True,
                element="N",
                source_functor=self.name,
            ),
            'nitrification_flux': UpdateItem(
                value=actual_nitrification_rate,
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
            'diagnostics.temperature_response': rates['temperature_response'],
            'diagnostics.oxygen_factor': rates['oxygen_factor'],
            'diagnostics.ph_factor': rates['ph_factor'],
            # Declared step length so the public ledger can integrate the N2O rate.
            'diagnostics.nitrification.step_seconds': safe_dt.detach().reshape(()),
            'diagnostics.nitrification.substrate_limiter_residual': UpdateItem(
                value=substrate_limiter_residual,
                mode=UpdateMode.REPLACE,
                unit="g N m-2 s-1",
                basis="per_second",
                per_timestep=False,
                source_functor=self.name,
            ),
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
        """Audit the returned NH4/NO3/N2O proposal as one closed per-step nitrogen ledger.

        References:
          - Manzoni and Porporato (2009), process-level conservation diagnostics.
          - Li et al. (1992); Parton et al. (1996), nitrification with N2O by-product.
        Mathematical expression:
          r = Delta_NH4 + Delta_NO3 + F_N2O * dt.
        Parameter meanings:
          Delta_NH4 and Delta_NO3 are the returned per-step pool increments, F_N2O the returned N2O rate; dt is the step length the proposal declares under
          diagnostics.nitrification.step_seconds.
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
            increments=("nh4_nitrogen", "no3_nitrogen"),
            rates=("n2o_flux",),
            step_key="diagnostics.nitrification.step_seconds",
            increment_unit="g N m-2",
            rate_unit="g N m-2 s-1",
        )
        magnitude = float(residual.detach().abs().max().item())
        return {law: magnitude for law in self.metadata.conservation_laws}

    def get_output_specifications(self) -> Dict[str, Any]:
        return {
            "nitrification_flux": StandardPedosphereOutputs.NITRIFICATION,
            "n2o_flux": StandardPedosphereOutputs.N2O_EMISSION,
        }
    
    def get_nitrification_statistics(self) -> Dict[str, float]:
        # References: Parton et al. (1996), process reporting context.
        # Mathematical expression: mean_per_call=total_nitrified/max(1,n_calls);
        #   N2O_yield=total_N2O/max(total_nitrified,epsilon).
        # Parameter meanings: Totals are detached sums of step amounts and n_calls
        #   is the execution counter; epsilon is the existing reporting denominator floor.
        # Units: totals and mean_per_call [g N m-2] as unweighted entry sums; yield [1].
        # Domain/constraints: The field named mean_nitrification_rate is a mean per call,
        #   not a per-second rate; no area or elapsed-time normalization is performed.
        # Sign convention: Positive accumulated consumption/emission raises the totals.
        epsilon_value = self.get_parameter_value(
            "epsilon",
            namespace="pedosphere.numerical",
            source_order=("manager",),
            required=True,
        )
        if torch.is_tensor(epsilon_value):
            if epsilon_value.numel() == 1:
                epsilon_value = float(epsilon_value.detach().reshape(-1)[0].tolist())
            else:
                epsilon_value = float(epsilon_value.detach().reshape(-1).mean().tolist())
        else:
            epsilon_value = float(epsilon_value)
        total_nitrified = float(self._diagnostic_totals['total_nitrified'].detach().reshape(-1)[0].tolist())
        total_n2o_produced = float(self._diagnostic_totals['total_n2o_produced'].detach().reshape(-1)[0].tolist())
        return {
            'total_nitrified': total_nitrified,
            'total_n2o_produced': total_n2o_produced,
            'execution_count': self._execution_count,
            'mean_nitrification_rate': total_nitrified / max(1, self._execution_count),
            'n2o_yield': total_n2o_produced / max(total_nitrified, epsilon_value)
        }


__all__ = ['NitrificationProcess']

