"""
Nitrogen Leaching Process Module for RELIC

This module implements nitrogen leaching via hydrological transport with
adsorption and mobility considerations.

Scientific Basis:
-----------------
Nitrogen leaching occurs when water-soluble nitrogen compounds (primarily 
NO3-) are transported from soil via water drainage. The process depends on
soil hydraulic properties, nitrogen concentration, and adsorption characteristics.

References:
- Li et al. (2006), Ecological Modelling 196, 116-130,
  doi:10.1016/j.ecolmodel.2006.02.007.
- van Genuchten and Wierenga (1986), Soil Science Society of America Journal.
- Biome-BGC version 4.2: nleaching.c implementation.

Mathematical Expression:
-----------------------
Nitrogen leaching flux:
F_N = Q_drain * C_NO3 * mobile_fraction

Retardation and mobile fraction (linear equilibrium sorption):
R = 1 + rho_b * K_d / theta
mobile_fraction = 1 / R

Conservative executable flux:
F_exec = min(F_N, N_NO3 / tau_leach, f_max * N_NO3 / dt)

Where:
- F_N: nitrogen leaching flux [g N m^-2 s^-1]
- Q_drain: water drainage flux [m s^-1]
- C_NO3: nitrate concentration in soil solution [g N m^-3]
- R: dimensionless retardation factor
- rho_b: soil bulk density [kg m^-3]
- K_d: adsorption coefficient [m^3 kg^-1]
- theta: soil volumetric water content [m^3 m^-3]
- tau_leach: minimum leaching time scale [s]
- f_max: maximum exported pool fraction per step [1]
"""

from typing import Dict, List, Mapping, Optional, Any
import logging
import torch

# Import core framework
from relic.computation.operators.dec import CochainSpaceSpec, HodgeStarSpec
from relic.computation.operators.iphs import PortSpecIPHS
from relic.core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from relic.core.abstractions import StateDict, FluxDict, TransportMechanism
from relic.core.state import UpdateItem, UpdateMode
from relic.utils.physical_constants import get_physical_constant

from relic.core.functor import TransportMechanismFunctor
from relic.scientific.pedosphere.api_contract import PedosphereContractMixin

logger = logging.getLogger(__name__)

# Import mass conservation and stability controls
from relic.scientific.pedosphere.mass_conservation import NumericalStabilityController


class NitrogenLeachingProcess(PedosphereContractMixin, TransportMechanismFunctor):
    """
    Nitrogen leaching process functor with hydrological coupling.
    
    Integrates with hydrosphere module via interface_mapping to transport
    water-soluble nitrogen from soil. Includes adsorption and mobility controls.
    
    References:
    - Biome-BGC nleaching.c: soilwater_nconc = MOBILEN_PROPORTION * sminn/soilw
    - CASA-CNP: NLeachingLoss/dt = fNleakHr * DIN
    - RHESSys compute_N_leached: complex adsorption parameterization
    
    Mathematical Expression:
    -----------------------
    Nitrogen leaching rate:
    F_N = Q_drain * C_NO3 * mobile_fraction
    
    Where Q_drain from hydrosphere interface, C_NO3 from soil N pools,
    mobile_fraction accounts for soil adsorption characteristics.
    
    Units:
    - Leaching flux: g N m^-2 s^-1
    - Water flux: kg m^-2 s^-1 (converted internally to m s^-1)
    - N concentration: g N m^-3
    """

    TRANSPORT_TYPE = "advection"
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'
    
    def __init__(self, name: str = "nitrogen_leaching", config: Optional[Dict] = None):
        # Prepare metadata first
        if config is None:
            config = {}
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Nitrogen leaching via hydrological transport with adsorption",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="F_N = Q_drain * C_NO3 * mobile_fraction",
            references=[
                "Li et al. (2006), doi:10.1016/j.ecolmodel.2006.02.007",
                "van Genuchten and Wierenga (1986), soil solute transport",
                "Biome-BGC nleaching.c: MOBILEN_PROPORTION formulation",
            ],
            parameters={
                "pedosphere.nutrient_cycling.leaching_time_constant": (
                    "Time constant limiting nitrate export [s]"
                ),
                "pedosphere.nutrient_cycling.min_no3_pool": (
                    "Minimum nitrate pool enforced for numerical stability [g N m-2]"
                ),
                "pedosphere.transport.nitrate_distribution_coefficient": (
                    "Nitrate sorption distribution coefficient K_d [L kg-1]"
                ),
                "pedosphere.nutrient_cycling.hodge_reference_soil_depth": (
                    "Reference soil depth of the static leaching Hodge contract [m]"
                ),
                "pedosphere.numerical.max_flux_fraction": (
                    "Maximum leaching fraction per time step [dimensionless]"
                ),
                "validation.conservation.conservation_tolerance": (
                    "General relative tolerance of the conservation checks [dimensionless]"
                ),
                "validation.conservation.strict_conservation": (
                    "Switch that makes a conservation violation raise [dimensionless]"
                ),
            },
            input_variables=[
                "no3_nitrogen", "soil_moisture", "drainage_flux", "soil_depth", "bulk_density",
            ],
            output_variables=[
                "nitrogen_leaching_flux",
                "no3_nitrogen",
                "diagnostics.nitrogen_leaching.nitrate_concentration",
                "diagnostics.nitrogen_leaching.mobile_fraction",
                "diagnostics.nitrogen_leaching.soil_water_volume",
                "diagnostics.nitrogen_leaching.step_seconds",
            ],
            conservation_laws=["nitrogen", "mass"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
            spatial_dimensions=2,
            temporal_scale=TemporalScale.SECOND
        )
        # Initialize base functor
        super().__init__(name=name, config=config, metadata=metadata, device=None)
        self.metadata = metadata
        self.logger = logger
        
        nutrient_namespace = "pedosphere.nutrient_cycling"
        numerical_namespace = "pedosphere.numerical"

        self.max_flux_fraction = float(
            self.get_parameter_value(
                "max_flux_fraction",
                namespace=numerical_namespace, required=True)
        )
        self.leaching_time_constant = float(
            self.get_parameter_value(
                "leaching_time_constant",
                namespace=nutrient_namespace, required=True)
        )
        self.water_density = float(get_physical_constant("water_density"))
        # References: van Genuchten and Wierenga (1986); Jury and Horton (2004)
        #   (retardation by linear equilibrium sorption); ParameterManager
        #   binding ownership, docs/source/architecture_overview.md Section 4.2.
        # Mathematical expression: R = 1 + rho_b * (K_d * c_L) / theta, with K_d
        #   read from the current binding at every evaluation.
        # Parameter meanings: K_d is the nitrate distribution coefficient
        #   (pedosphere.transport.nitrate_distribution_coefficient), c_L the
        #   liter-to-cubic-meter conversion, rho_b the bulk density, theta the
        #   volumetric water content, R the retardation factor.
        # Units: K_d [L kg-1]; c_L [m3 L-1]; rho_b [kg m-3]; theta [m3 m-3]; R [1].
        # Domain/constraints: K_d >= 0. K_d is a learnable binding held by
        #   reference and never cloned: the manager stays its one owner, the law
        #   follows a changed K_d, a checkpoint saves it in parameter_bindings,
        #   and no autograd graph is stored on the instance. The law never
        #   writes to it in place (the product K_d * c_L is a new tensor).
        # Sign convention: none (a nonnegative sorption coefficient).
        self.nitrate_distribution_coefficient = self.get_parameter_value(
            "nitrate_distribution_coefficient",
            namespace="pedosphere.transport",
            required=True,
        )
        # References: BIPM SI Brochure, 9th edition.
        # Mathematical expression: Kd_m3_kg = Kd_L_kg * c_L_to_m3.
        # Parameters: c_L_to_m3 is the liter-to-cubic-meter conversion [m3 L-1].
        self.liter_to_cubic_meter = float(
            get_physical_constant(
                "relic.physical.constants.conversions.liter_to_cubic_meter"
            )
        )
        self.min_no3_pool = float(
            self.get_parameter_value(
                "min_no3_pool",
                namespace="pedosphere.nutrient_cycling", required=True)
        )

        self.stability_controller = NumericalStabilityController(
            max_flux_fraction=self.max_flux_fraction,
            min_pool_size=self.min_no3_pool
        )
        self.optional_input_variables = []

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
        if dtype is None:
            dtype = torch.float32
            
        tensor_value = torch.as_tensor(value, dtype=dtype)
        if self.device is not None:
            tensor_value = tensor_value.to(self.device)
        return tensor_value.clone()

    def _require_input_tensor(
        self,
        state: Mapping[str, Any],
        key: str,
        *,
        dtype: Optional[torch.dtype] = None,
        strictly_positive: bool = False,
        upper_bound: Optional[float] = None,
    ) -> torch.Tensor:
        """Return a finite canonical input without aliases or hidden clipping."""
        if key not in state:
            raise KeyError(f"Required input '{key}' is missing.")
        try:
            value = self._to_tensor(state[key], dtype=dtype)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"Input '{key}' must be tensor-convertible.") from exc
        if value.numel() == 0 or not bool(torch.all(torch.isfinite(value))):
            raise ValueError(f"Input '{key}' must contain only finite values.")
        if strictly_positive:
            if bool(torch.any(value <= 0.0)):
                raise ValueError(f"Input '{key}' must be strictly positive.")
        elif bool(torch.any(value < 0.0)):
            raise ValueError(f"Input '{key}' must be non-negative.")
        if upper_bound is not None and bool(torch.any(value > upper_bound)):
            raise ValueError(f"Input '{key}' must not exceed {upper_bound}.")
        return value
    
    def get_dependencies(self) -> List[str]:
        """
        Get list of dependencies for this functor.

        Nitrogen leaching depends on nitrate availability from nitrification.

        Returns:
            List[str]: Required dependency names
        """
        return self._merge_declared_dependencies(["nitrification"])

    def declared_forms(self) -> tuple[CochainSpaceSpec, ...]:
        """Declare nitrate leaching cochain spaces for Phase 8 audit.

        References:
          - Biome-BGC nleaching.c mobile nitrogen formulation.
          - van Genuchten and Wierenga (1986), soil solute transport.
        Mathematical expression:
          x_dot = -D_h q_h + G_h u_h, q_h = star_Q C_NO3 f_mobile.
        Parameter meanings:
          no3_nitrogen is the nitrate pool, nitrate_solution_concentration is
          pool divided by water volume, and nitrogen_leaching_flux is the open
          drainage export flow.
        """

        return (
            CochainSpaceSpec(
                complex_id="pedosphere_nitrogen_leaching",
                degree=0,
                primality="primal",
                n_dofs=1,
                units="g N m-2",
                extensivity="extensive",
                physical_domain="pedosphere",
                semantic_name="no3_nitrogen",
                orientation_convention="outward_positive",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="pedosphere_nitrogen_leaching",
                degree=0,
                primality="primal",
                n_dofs=1,
                units="g N m-3",
                extensivity="intensive",
                physical_domain="pedosphere",
                semantic_name="nitrate_solution_concentration",
                orientation_convention="outward_positive",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="pedosphere_nitrogen_leaching",
                degree=2,
                primality="dual",
                n_dofs=1,
                units="g N m-2 s-1",
                extensivity="integral",
                physical_domain="pedosphere",
                semantic_name="nitrogen_leaching_flux",
                orientation_convention="outward_positive",
                boundary_role="boundary",
            ),
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Declare ParameterManager keys used by runtime and structural builders."""

        return (
            "pedosphere.numerical.max_flux_fraction",
            "pedosphere.nutrient_cycling.leaching_time_constant",
            "pedosphere.transport.nitrate_distribution_coefficient",
            "pedosphere.nutrient_cycling.hodge_reference_soil_depth",
            "pedosphere.nutrient_cycling.min_no3_pool",
        )

    def build_hodge_contract(
        self,
        n_dofs: int,
        *,
        strict: bool = True,
    ) -> HodgeStarSpec:
        """Build the nitrate leaching mobility Hodge contract.

        References:
          - Cameron and Haynes (1986), nitrate movement in soil.
          - Hirani (2003), Discrete Exterior Calculus.
        Mathematical expression:
          q_h = star_Q C_NO3 with star_Q >= 0.
        Parameter meanings:
          max_flux_fraction and leaching_time_constant define a conservative
          positive nominal export mobility when runtime drainage is absent.
        """
        # References: Cameron and Haynes (1986); Hirani (2003), structural context.
        # Mathematical expression: star_nominal=diag(max(f_max/tau_leach,0)).
        # Parameter meanings: f_max is maximum flux fraction; tau_leach is export time;
        #   n_dofs determines the repeated diagonal size.
        # Units: current diagonal [s-1]; declared input concentration [g N m-3];
        #   declared output [g N m-2 s-1] requires a length factor not present in this matrix.
        # Domain/constraints: Positive n_dofs; this nominal mapping requires a resolved
        #   support/metric contract before a physical or strict equivalence claim.
        # Sign convention: Nominal diagonal is nonnegative; direction is outward export.

        n_local = int(n_dofs)
        if n_local < 1:
            raise ValueError("Nitrogen leaching Hodge degrees of freedom must be positive.")
        nominal_mobility = max(self.max_flux_fraction / self.leaching_time_constant, 0.0)
        values = torch.full(
            (n_local,),
            float(nominal_mobility),
            device=self.device,
            dtype=torch.get_default_dtype(),
        )
        concentration_space = CochainSpaceSpec(
            complex_id="pedosphere_nitrogen_leaching",
            degree=0,
            primality="primal",
            n_dofs=n_local,
            units="g N m-3",
            extensivity="intensive",
            physical_domain="pedosphere",
            semantic_name="nitrate_solution_concentration",
            orientation_convention="outward_positive",
        )
        flux_space = CochainSpaceSpec(
            complex_id="pedosphere_nitrogen_leaching",
            degree=2,
            primality="dual",
            n_dofs=n_local,
            units="g N m-2 s-1",
            extensivity="integral",
            physical_domain="pedosphere",
            semantic_name="nitrogen_leaching_flux",
            orientation_convention="outward_positive",
            boundary_role="boundary",
        )
        hodge = HodgeStarSpec(
            name=f"{self.name}_nitrate_leaching_hodge",
            domain=concentration_space,
            codomain=flux_space,
            matrix=torch.diag(values),
            parameter_keys=(
                "pedosphere.numerical.max_flux_fraction",
                "pedosphere.nutrient_cycling.leaching_time_constant",
            ),
            # Static reference depth of this nominal contract; the runtime depth is the
            # soil_depth forcing input.
            metric_keys=("pedosphere.nutrient_cycling.hodge_reference_soil_depth",),
            material_law="nitrate_drainage_export_mobility",
        )
        if strict:
            hodge.validate(passive=True)
        return hodge

    def build_hodge_terms(
        self,
        complex_bundle: Any = None,
        resolved_parameters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[HodgeStarSpec, ...]:
        """Build nitrate leaching Hodge terms for Phase 8 assembly."""

        # The Hodge acts on the declared state layout unless the complex bundle
        # declares its own degrees of freedom.
        bundle_dofs = getattr(complex_bundle, "n_dofs", None) if complex_bundle is not None else None
        n_dofs = self.declared_forms()[0].n_dofs if bundle_dofs is None else bundle_dofs
        return (self.build_hodge_contract(n_dofs=n_dofs, strict=True),)

    def energy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return a structural nitrate storage-effort audit view."""
        # References: van der Schaft and Maschke (1995), structural audit context.
        # Mathematical expression: x=max(NO3,0); H_struct=0.5*sum(x^2); e_struct=x.
        # Parameter meanings: x is the flattened nonnegative nitrate areal pool.
        # Units: x,e_struct [g N m-2]; H_struct [(g N m-2)^2].
        # Domain/constraints: This is an unweighted structural quadratic, not established
        #   chemical free energy; missing runtime state returns descriptive strings only.
        # Sign convention: H_struct is nonnegative; the helper does not move nitrate.

        if "no3_nitrogen" not in state:
            return {
                "storage_functional": "H_h = 0.5 * x_h.T x_h",
                "effort": "e_h = delta H_h / delta x_h",
            }
        no3_pool = torch.clamp(self._to_tensor(state["no3_nitrogen"]), min=0.0)
        vector = no3_pool.reshape(-1)
        return {
            "storage_functional": 0.5 * torch.sum(vector * vector),
            "effort": no3_pool,
        }

    def entropy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return non-negative structural leaching dissipation terms."""
        # References: van der Schaft and Maschke (1995), structural audit context.
        # Mathematical expression: sigma_current=sum(f_mobile*C_NO3^2).
        # Parameter meanings: C_NO3 and f_mobile are evaluated by the canonical rate core.
        # Units: sigma_current [(g N m-3)^2]; f_mobile [1].
        # Domain/constraints: Current expression omits drainage q used by runtime M;
        #   it is neither the claimed C.T*M*C rate nor a proven thermodynamic entropy rate.
        # Sign convention: Squared concentration produces a nonnegative structural proxy.

        if "no3_nitrogen" not in state:
            return {
                "dissipation_rate": "sigma_h = C_NO3.T star_Q C_NO3",
                "thermodynamic_force": "C_NO3",
            }
        rates = self._compute_leaching_rates(dict(state))
        concentration = rates["nitrate_concentration"].reshape(-1)
        mobile_fraction = torch.clamp(rates["mobile_fraction"].reshape(-1), min=0.0)
        return {
            "dissipation_rate": torch.sum(mobile_fraction * concentration * concentration),
            "thermodynamic_force": concentration,
        }

    def reversible_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> None:
        """Return no reversible interconnection for open nitrate leaching."""

        return None

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the positive nitrate drainage mobility.

        References:
          - Cameron and Haynes (1986), nitrate mobility.
          - van der Schaft and Maschke (1995), dissipative IPHS form.
        Mathematical expression:
          M_h = diag(q_drain f_mobile), q_drain >= 0, f_mobile >= 0.
        Parameter meanings:
          q_drain is the hydrological drainage flux and f_mobile is the
          adsorption-limited mobile nitrate fraction.
        """
        # References: Cameron and Haynes (1986); van der Schaft and Maschke (1995).
        # Mathematical expression: M_runtime=diag(q*f_mobile), J=0;
        #   q=water_flux_mass/rho_water; missing state uses build_hodge_contract.
        # Parameter meanings: q is hydrological drainage velocity and f_mobile is the
        #   adsorption-controlled fraction returned by the one leaching rate core.
        # Units: runtime M [m s-1]; concentration [g N m-3]; export [g N m-2 s-1].
        # Domain/constraints: Nonnegative mobility; nominal/runtime units and the
        #   state-coordinate versus concentration metric require the separate structural audit.
        # Sign convention: Positive mobility represents outward nitrate drainage export.

        if "no3_nitrogen" in state:
            rates = self._compute_leaching_rates(dict(state))
            water_velocity = torch.clamp(
                rates["water_flux_mass"].reshape(-1) / self.water_density,
                min=0.0,
            )
            mobile_fraction = torch.clamp(rates["mobile_fraction"].reshape(-1), min=0.0)
            values = water_velocity * mobile_fraction
            if values.numel() == 0:
                values = torch.diag(
                    self.build_hodge_contract(
                        n_dofs=self.declared_forms()[0].n_dofs, strict=True
                    ).matrix
                )
        else:
            values = torch.diag(
                self.build_hodge_contract(
                    n_dofs=self.declared_forms()[0].n_dofs, strict=True
                ).matrix
            )
        if bool(torch.any(values < 0.0)):
            raise ValueError("Nitrogen leaching mobility must be non-negative in M_h.")
        matrix = torch.diag(values.to(device=self.device))
        return {
            "state_vars": tuple(f"no3_nitrogen[{idx}]" for idx in range(matrix.shape[0])),
            "J": torch.zeros_like(matrix),
            "M": matrix,
            "ports": self.ports(),
        }

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Declare nitrate drainage export port.

        References:
          - Biome-BGC nleaching.c mobile nitrogen export.
          - RHESSys nitrogen leaching boundary fluxes.
        Mathematical expression:
          P_port = e_h F_N, with F_N = q_drain C_NO3 f_mobile.
        Parameter meanings:
          e_h is nitrate solution concentration and F_N is nitrogen_leaching_flux.
        """

        effort_space = CochainSpaceSpec(
            complex_id="pedosphere_nitrogen_leaching",
            degree=0,
            primality="primal",
            n_dofs=1,
            units="g N m-3",
            extensivity="intensive",
            physical_domain="pedosphere",
            semantic_name="nitrate_solution_effort",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        flow_space = CochainSpaceSpec(
            complex_id="pedosphere_nitrogen_leaching",
            degree=2,
            primality="dual",
            n_dofs=1,
            units="g N m-2 s-1",
            extensivity="integral",
            physical_domain="pedosphere",
            semantic_name="nitrogen_leaching_flux",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        return (
            PortSpecIPHS(
                name="nitrate_drainage_export",
                effort_var="nitrate_concentration",
                flow_var="nitrogen_leaching_flux",
                effort_space=effort_space,
                flow_space=flow_space,
                power_units="g2 N2 m-5 s-1",
                positive_power_convention="out_of_system",
                port_kind="matter",
            ),
        )

    def compute(
        self,
        state: Mapping[str, Any],
        time: float,
        dt: float,
    ) -> Mapping[str, Any]:
        """Run the canonical calculation through the existing Functor lifecycle."""

        return self.forward(dict(state), time, dt)
    
    def _compute_leaching_rates(self, state: StateDict) -> FluxDict:
        """
        Compute nitrogen leaching rate using standardized J=q*C paradigm.
        
        Implements the core leaching formulation:
        J_N = q * C_NO3 * f_mobile
        
        Where:
        - J_N: nitrogen leaching flux [g N m^-2 s^-1]
        - q: water drainage flux [m s^-1] 
        - C_NO3: nitrate concentration [g N m^-3]
        - f_mobile: mobile fraction accounting for adsorption [dimensionless]
        
        Parameters:
        -----------
        state : StateDict
            Current system state with nitrogen pools, soil water and the drainage
            flux (the one input channel; declared inputs only)
            
        Returns:
        --------
        FluxDict
            Leaching rates and diagnostics following J=q*C paradigm
            
        References:
        -----------
        - Biome-BGC nleaching.c: Mobile nitrogen proportion approach
        - CASA-CNP testbed: fNleakHr * DIN formulation
        - Cameron & Haynes (1986): Nitrogen mobility principles
        """
        # soil_depth and bulk_density are the forcing inputs of their contracts
        # (soil_layer_geometry_input, soil_sorption_geometry_input); a missing input
        # raises and no *_default parameter stands in for it.
        missing = [name for name in self.get_required_inputs() if name not in state]
        if missing:
            raise ValueError(
                "NitrogenLeachingProcess requires the inputs " + ", ".join(missing)
            )
        no3_pool = self._require_input_tensor(state, "no3_nitrogen")
        soil_moisture = self._require_input_tensor(
            state,
            "soil_moisture",
            dtype=no3_pool.dtype,
            strictly_positive=True,
            upper_bound=1.0,
        )
        water_flux_mass = self._require_input_tensor(
            state,
            "drainage_flux",
            dtype=no3_pool.dtype,
        )
        
        # Convert mass flux to volumetric flux: q = F_water / rho_water [m s^-1]
        water_flux_volumetric = water_flux_mass / self.water_density  # m s^-1
        
        # References: Cameron and Haynes (1986); Jury and Horton (2004).
        # Mathematical expression: V_w = theta z; C_NO3 = N_NO3 / V_w.
        # Parameter meanings: theta soil moisture, z the soil_depth forcing input,
        #   N_NO3 the nitrate pool, V_w the soil water per unit area.
        # Units: theta [m3 m-3]; z [m]; V_w [m]; N_NO3 [g N m-2]; C_NO3 [g N m-3].
        # Domain/constraints: z > 0 and 0 < theta <= 1 (checked inputs); no default
        #   depth replaces a missing soil_depth.
        # Sign convention: magnitudes; leaching exports nitrate positively.
        soil_depth = self._require_input_tensor(
            state, "soil_depth", dtype=soil_moisture.dtype, strictly_positive=True
        )
        bulk_density = self._require_input_tensor(
            state, "bulk_density", dtype=soil_moisture.dtype, strictly_positive=True
        )
        soil_water_volume = soil_moisture * soil_depth  # m^3 m^-2 = m
        
        # NO3 concentration in soil solution [g N m^-3]
        no3_concentration = no3_pool / soil_water_volume
        
        # Calculate mobile fraction (accounting for adsorption)
        # References: Jury and Horton (2004); van Genuchten and Wierenga (1986).
        # Equations: R = 1 + rho_b * Kd / theta; f_mobile = 1 / R.
        # Parameters: rho_b (bulk density), Kd (distribution coefficient),
        #             theta (volumetric water content), f_mobile (mobile fraction).
        kd_tensor = self._to_tensor(
            self.nitrate_distribution_coefficient * self.liter_to_cubic_meter,
            dtype=soil_moisture.dtype,
        )
        bulk_density_tensor = bulk_density
        retardation_factor = 1.0 + (
            bulk_density_tensor * kd_tensor
        ) / soil_moisture
        mobile_fraction = torch.reciprocal(retardation_factor)
        
        # Apply J=q*C*f paradigm for nitrogen leaching flux
        nitrogen_leaching_flux = (water_flux_volumetric * 
                                 no3_concentration * 
                                 mobile_fraction)  # g N m^-2 s^-1
        
        # References: Li et al. (2006) and conservative explicit integration.
        # Mathematical expression: F_tau = min(q C / R, N_NO3 / tau_leach).
        # Parameters: tau_leach is the minimum nitrate export time scale [s].
        max_transport_rate = no3_pool / self.leaching_time_constant  # Conservative limit
        nitrogen_leaching_flux = torch.min(nitrogen_leaching_flux, max_transport_rate)
        
        return {
            # Primary output: standardized leaching flux
            'nitrogen_leaching_flux': nitrogen_leaching_flux,  # g N m^-2 s^-1
            
            # J=q*C paradigm components for diagnostics and coupling
            'water_flux_volumetric': water_flux_volumetric,   # q [m s^-1]
            'nitrate_concentration': no3_concentration,       # C [g N m^-3]  
            'mobile_fraction': mobile_fraction,               # f [dimensionless]
            
            # Additional diagnostics
            'water_flux_mass': water_flux_mass,              # Original flux [kg m^-2 s^-1]
            'soil_water_volume': soil_water_volume,          # Available water [m]
            'retardation_factor': retardation_factor, # R [dimensionless]
            'no3_pool': no3_pool,
        }

    def forward(self,
        # References: Cameron and Haynes (1986); van Genuchten and Wierenga (1986);
        #   Manzoni and Porporato (2009), explicit donor-limited accounting.
        # Mathematical expression: F_exec=limit_fluxes(F_potential,NO3,dt), with
        #   F_exec<=f_max*NO3/dt; DeltaNO3=-F_exec*dt.
        # Parameter meanings: f_max is the existing stability controller's maximum
        #   exported pool fraction; source_pool_by_flux names the nitrate donor explicitly.
        # Units: F_exec [g N m-2 s-1]; DeltaNO3 [g N m-2]; dt [s].
        # Domain/constraints: Positive finite scalar dt and valid rate-core inputs;
        #   reuse the existing limiter and compare pool loss with export on one time basis.
        # Sign convention: Nitrate ACCUMULATE amount is negative; public REPLACE export
        #   rate is positive and remains per_second rather than an accumulated stock.
                state: StateDict,
                time: float,
                dt: torch.Tensor | float,
                **kwargs) -> FluxDict:
        """
        Execute nitrogen leaching for one time step.

        Applies stability controls and limiting to prevent pool depletion.
        """
        # References: Cameron and Haynes (1986); van Genuchten and Wierenga (1986).
        # Mathematical expression: F_N = min(q_drain C_NO3 / R, N_NO3 / tau).
        # Parameter meanings: q_drain is drainage water flux, C_NO3 is soil-solution
        # nitrate concentration, R is retardation, and tau is the export time scale.
        rates = self._compute_leaching_rates(state)
        no3_nitrogen = rates["no3_pool"]
        nitrogen_leaching_flux = rates['nitrogen_leaching_flux']

        dt_tensor = self._to_tensor(dt, dtype=no3_nitrogen.dtype)
        if dt_tensor.numel() != 1 or not bool(torch.all(torch.isfinite(dt_tensor))):
            raise ValueError("dt must be a finite scalar.")
        if bool(torch.any(dt_tensor <= 0.0)):
            raise ValueError("dt must be strictly positive.")

        # References: Manzoni and Porporato (2009); Sierra et al. (2012).
        # Mathematical expression: F_exec <= f_max N_NO3 / dt.
        # Parameters: f_max is the maximum exported pool fraction per step [1].
        limited_fluxes = self.stability_controller.limit_fluxes(
            fluxes={'nitrogen_leaching_flux': nitrogen_leaching_flux},
            pools={'no3_nitrogen': no3_nitrogen},
            dt=dt_tensor,
            source_pool_by_flux={
                'nitrogen_leaching_flux': 'no3_nitrogen',
            },
        )
        nitrogen_leaching_flux = limited_fluxes['nitrogen_leaching_flux']
        
        # Calculate pool changes
        delta_no3_nitrogen = -nitrogen_leaching_flux * dt_tensor

        # References:
        #   - Parton et al. (1996) nitrogen transport process coupling
        #   - Li et al. (1992), doi:10.1029/91JD02705
        # Mathematical expression:
        #   delta_NO3 = -F_leach * dt
        # Parameter meanings:
        #   nitrogen_leaching_flux: nitrate leaching flux [g N m-2 s-1]
        #   delta_no3_nitrogen: nitrate pool change over dt [g N m-2]
        conservation_updates = {
            "no3_nitrogen": delta_no3_nitrogen,
            # Use declared output key for exported nitrogen so metadata-first
            # conservation aggregation includes both terms consistently.
            "nitrogen_leaching_flux": nitrogen_leaching_flux * dt_tensor,
        }
        self._enforce_forward_conservation(state, conservation_updates)

        return {
            # Primary output (standardized key name)
            'nitrogen_leaching_flux': UpdateItem(
                value=nitrogen_leaching_flux,
                mode=UpdateMode.REPLACE,
                unit="g N m-2 s-1",
                basis="per_second",
                per_timestep=False,
                source_functor=self.name,
            ),
            
            # Pool changes
            'no3_nitrogen': UpdateItem(
                value=delta_no3_nitrogen,
                mode=UpdateMode.ACCUMULATE,
                unit="g N m-2",
                basis="per_step",
                per_timestep=True,
                element="N",
                source_functor=self.name,
            ),
            
            # Diagnostics
            'diagnostics.nitrogen_leaching.nitrate_concentration': UpdateItem(
                value=rates['nitrate_concentration'],
                mode=UpdateMode.REPLACE,
                unit="g N m-3",
                basis="instantaneous",
                per_timestep=False,
                source_functor=self.name,
            ),
            'diagnostics.nitrogen_leaching.mobile_fraction': UpdateItem(
                value=rates['mobile_fraction'],
                mode=UpdateMode.REPLACE,
                unit="1",
                basis="instantaneous",
                per_timestep=False,
                source_functor=self.name,
            ),
            'diagnostics.nitrogen_leaching.soil_water_volume': UpdateItem(
                value=rates['soil_water_volume'],
                mode=UpdateMode.REPLACE,
                unit="m",
                basis="instantaneous",
                per_timestep=False,
                source_functor=self.name,
            ),
            # Declared step length so the public ledger can integrate the export rate.
            'diagnostics.nitrogen_leaching.step_seconds': UpdateItem(
                value=dt_tensor.detach().reshape(()),
                mode=UpdateMode.REPLACE,
                unit="s",
                basis="instantaneous",
                per_timestep=False,
                source_functor=self.name,
            ),
        }

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Audit the returned NO3/leaching-export proposal as one closed per-step nitrogen ledger.

        References:
          - Manzoni and Porporato (2009), process-level conservation diagnostics.
          - Cameron and Haynes (1986); van Genuchten and Wierenga (1986), nitrate export.
        Mathematical expression:
          r = Delta_NO3 + F_leach * dt.
        Parameter meanings:
          Delta_NO3 is the returned per-step nitrate increment, F_leach the returned boundary-export rate; dt is the step length the proposal declares under
          diagnostics.nitrogen_leaching.step_seconds.
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
            rates=("nitrogen_leaching_flux",),
            step_key="diagnostics.nitrogen_leaching.step_seconds",
            increment_unit="g N m-2",
            rate_unit="g N m-2 s-1",
        )
        magnitude = float(residual.detach().abs().max().item())
        return {law: magnitude for law in self.metadata.conservation_laws}

__all__ = ['NitrogenLeachingProcess']

# Register as virtual subclass for isinstance checks (avoids MRO issues)
TransportMechanism.register(NitrogenLeachingProcess)

