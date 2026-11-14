"""
Groundwater Dynamics Module for RELIC

This module implements groundwater storage dynamics with depth-dependent
porosity profiles and water table fluctuations. Based on RHESSys 
compute_delta_water.c methodology for handling groundwater-soil coupling.

References
----------
- RHESSys compute_delta_water.c implementation
- Beven, K.J. & Kirkby, M.J. (1979). A physically based, variable contributing 
  area model of hydrologic response incorporating variable source area concept.
  Hydrological Sciences Bulletin, 24(1), 43-69.
- Wigmosta, M.S. et al. (1994). A distributed hydrology-vegetation model for
  complex terrain. Water Resources Research, 30(6), 1665-1679.

Mathematical Expression
------------------------
Porosity profile: n(z) = n0 * exp(-z/p)
where:
- n0 = surface porosity [dimensionless]
- z = depth below surface [m] 
- p = decay length scale [m]

Storage balance: dS/dt = (F_drain + F_lat - F_pump - F_cap - F_base)
                         * 1000 / rho_w
Storage profile: DeltaS = integral[z_wt_new to z_wt_old] n(z) dz

Units Convention:
- groundwater_storage: mm (for consistency with runoff modules)
- water_table_depth: m (positive downward from surface)
- boundary water fluxes: kg m-2 s-1
- storage diagnostics: mm or dimensionless as documented in forward()
"""

from typing import Any, Dict, Mapping, Optional, Union, Tuple
import torch
from torch import Tensor

from relic.computation.operators.dec.forms import CochainSpaceSpec
from relic.computation.operators.dec.hodge import HodgeStarSpec
from relic.computation.operators.iphs import PortSpecIPHS
from relic.core.functor import (
    DependencyDeclaration,
    ExecutionPhase,
    FunctorMetadata,
    PhysicalFieldFunctor,
    ProcessCategory,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from relic.data.standards.unit_converter import unit_converter
from relic.core.abstractions import StateDict, FluxDict
from relic.core.state import UpdateItem, UpdateMode
from relic.core.units import UnitSystem
from relic.utils.physical_constants import get_physical_constant
from ..interface_mapping import HydrosphereInterfaceMixin


def _scalar_from_value(value: Any) -> float:
    """Convert scalar-like inputs to Python float for configuration and I/O bridges."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().reshape(-1)[0].tolist())
    return float(value)


class GroundwaterDynamics(HydrosphereInterfaceMixin, PhysicalFieldFunctor):
    """
    Groundwater Storage Dynamics
    
    Manages groundwater storage evolution, water table depth changes,
    and provides coupling variables for subsurface runoff and capillary rise.
    
    Implements exponential porosity decay with depth and handles bidirectional
    coupling with vadose zone soil moisture and saturated zone processes.
    
    Key Features:
    - Depth-dependent porosity profiles n(z) = n0 * exp(-z/p)
    - Water table depth tracking and storage deficit calculation
    - Saturation fraction for variable source area modeling
    - Mass balance with soil moisture and drainage inputs
    - consistent with TOPMODEL and similar distributed models
    
    Attributes
    ----------
    surface_porosity : float
        Porosity at soil surface [dimensionless]
    porosity_decay_length : float
        Exponential decay length scale [m]
    max_water_table_depth : float
        Maximum allowable water table depth [m]
    bedrock_depth : float
        Depth to impermeable bedrock [m]
    specific_yield : float
        Specific yield for water table fluctuations [dimensionless]
    """

    PHASE8_RUNTIME_REPLACEMENT_STATUS = "canonical_iphs_runtime"

    # Point model: no coordinate query; the state is read from StateManager.
    SPATIAL_DIMS = None
    
    def __init__(self, 
                 name: str = "groundwater_dynamics",
                 config: Optional[Dict] = None) -> None:
        """
        Initialize groundwater dynamics model.
        
        Parameters
        ----------
        name : str, optional
            Unique identifier for this functor instance
        config : Dict, optional
            Configuration parameters for groundwater properties
        """
        # Create metadata
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Groundwater storage dynamics with exponential porosity decay and water table tracking",
            category=ProcessCategory.PHYSICAL_FIELD,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "S_dot = G_gw u_gw - q_base; S(z) = integral n0 * exp(-z/p) dz; star_Sy >= 0"
            ),
            references=[
                "RHESSys compute_delta_water.c",
                "Beven, K.J. & Kirkby, M.J. (1979). Hydrological Sciences Bulletin, 24(1), 43-69",
                "Wigmosta, M.S. et al. (1994). Water Resources Research, 30(6), 1665-1679"
            ],
            parameters={
                "hydrosphere.groundwater.surface_porosity": "Porosity at the surface [1]",
                "hydrosphere.groundwater.porosity_decay_length": "Exponential porosity decay length [m]",
                "hydrosphere.groundwater.max_water_table_depth": "Maximum water-table depth [m]",
                "hydrosphere.groundwater.bedrock_depth": "Impermeable bedrock depth [m]",
                "hydrosphere.groundwater.specific_yield": "Groundwater specific yield [1]",
                "hydrosphere.groundwater.minimum_water_table_depth": "Minimum water-table depth [m]",
                "hydrosphere.groundwater.minimum_depth_factor": "Minimum baseflow depth factor [1]",
                "hydrosphere.groundwater.drainage_time_constant": "Baseflow recession time constant [s]",
                "hydrosphere.groundwater.enable_lateral_flow": "Lateral-inflow enable flag [1]",
                "hydrosphere.groundwater.topographic_index": "TOPMODEL-like topographic index [1]",
                "hydrosphere.numerical.minimum_timestep_tolerance": "Positive recession denominator floor [s]",
                "hydrosphere.numerical.epsilon_small": "Analytic inversion denominator tolerance [1]",
            },
            input_variables=[
                "groundwater_storage",
                "water_table_depth",
                "drainage_flux",
                "capillary_rise_flux",
                "lateral_inflow",
                "groundwater_pumping",
            ],
            output_variables=[
                "groundwater_storage",
                "water_table_depth",
                "baseflow_flux",
                "diagnostics.groundwater",
            ],
            conservation_laws=["mass"],
            spatial_dimensions=1,  # 1D vertical profile
            temporal_scale=TemporalScale.SUB_DAILY,
            coupling_interfaces=["soil_moisture", "subsurface_runoff", "capillary_rise"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
        )
        
        super().__init__(name=name, config=config or {}, metadata=metadata)

        # Get parameters from unified parameter system with full namespace
        self.surface_porosity = self.get_parameter_value(
            "surface_porosity",
            namespace="hydrosphere.groundwater", required=True)
        self.porosity_decay_length = self.get_parameter_value(
            "porosity_decay_length",
            namespace="hydrosphere.groundwater", required=True)
        self.max_water_table_depth = self.get_parameter_value(
            "max_water_table_depth",
            namespace="hydrosphere.groundwater", required=True)
        self.bedrock_depth = self.get_parameter_value(
            "bedrock_depth",
            namespace="hydrosphere.groundwater", required=True)
        self.specific_yield = self.get_parameter_value(
            "specific_yield",
            namespace="hydrosphere.groundwater", required=True)
        self.minimum_water_table_depth = self.get_parameter_value(
            "minimum_water_table_depth",
            namespace="hydrosphere.groundwater", required=True)
        self.minimum_depth_factor = self.get_parameter_value(
            "minimum_depth_factor",
            namespace="hydrosphere.groundwater", required=True)
        self.drainage_time_constant = self.get_parameter_value(
            "drainage_time_constant",
            namespace="hydrosphere.groundwater", required=True)
        self.enable_lateral_flow = self.get_parameter_value(
            "enable_lateral_flow",
            namespace="hydrosphere.groundwater", required=True)
        self.topographic_index = self.get_parameter_value(
            "topographic_index",
            namespace="hydrosphere.groundwater", required=True)
        self.water_density_reference = float(get_physical_constant("water_density"))
        self.minimum_timestep_tolerance = self.get_parameter_value(
            "minimum_timestep_tolerance",
            namespace="hydrosphere.numerical", required=True)
        self.epsilon_small = self.get_parameter_value(
            "epsilon_small",
            namespace="hydrosphere.numerical", required=True)
        self._validate_parameters()
        
    def _validate_parameters(self) -> None:
        """Validate parameter consistency and physical constraints."""
        if not (0 < self.surface_porosity <= 1):
            raise ValueError("Surface porosity must be in range (0, 1]")
        
        if self.porosity_decay_length <= 0:
            raise ValueError("Porosity decay length must be positive")
        
        if self.max_water_table_depth <= 0:
            raise ValueError("Maximum water table depth must be positive")
        
        if self.bedrock_depth <= 0:
            raise ValueError("Bedrock depth must be positive")
        
        if not (0 < self.specific_yield <= 1):
            raise ValueError("Specific yield must be in range (0, 1]")
        
    def _validate_state_contract(
        self,
        state: Mapping[str, Any],
        *,
        require_fluxes: bool,
    ) -> StateDict:
        """Validate explicit tensor state and boundary-port inputs."""

        required = ["groundwater_storage", "water_table_depth"]
        if require_fluxes:
            required.extend(
                [
                    "drainage_flux",
                    "capillary_rise_flux",
                    "lateral_inflow",
                    "groundwater_pumping",
                ]
            )

        normalized: StateDict = dict(state or {})
        for key in required:
            if key not in normalized or normalized[key] is None:
                raise KeyError(f"GroundwaterDynamics requires '{key}'.")
            value = normalized[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"GroundwaterDynamics input '{key}' must be a torch.Tensor.")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"GroundwaterDynamics input '{key}' must be finite.")
            if bool(torch.any(value < 0.0)):
                raise ValueError(f"GroundwaterDynamics input '{key}' must be non-negative.")

        bedrock_depth = self._ensure_tensor(self.bedrock_depth).to(
            device=normalized["water_table_depth"].device,
            dtype=normalized["water_table_depth"].dtype,
        )
        if bool(torch.any(normalized["water_table_depth"] > bedrock_depth)):
            raise ValueError("GroundwaterDynamics input 'water_table_depth' exceeds bedrock depth.")
        return normalized
    
    def get_dependencies(self) -> DependencyDeclaration:
        """
        Return list of dependencies for this functor.
        
        Returns
        -------
        DependencyDeclaration
            Dependencies: drainage inputs and optional lateral flows
        """
        # Dependencies are declared in terms of functor IDs (not variable names).
        # GroundwaterDynamics can accept external drivers via state, so we mark upstream
        # hydrosphere providers as optional to avoid hard failures when they are absent.
        optional = [
            "bucket_soil_moisture",
            "richards_soil_moisture",
            "vertical_percolation_transport",
            "capillary_rise_transport",
        ]
        return self._merge_declared_dependencies({"required": [], "optional": optional})

    def _structural_n_dofs(
        self,
        n_dofs: Optional[int] = None,
        complex_bundle: Any = None,
    ) -> int:
        """Resolve structural audit size without changing runtime tensor shape."""

        if n_dofs is not None:
            value = int(n_dofs)
        elif complex_bundle is not None and hasattr(complex_bundle, "n_dofs"):
            value = int(getattr(complex_bundle, "n_dofs"))
        else:
            value = 1
        if value < 1:
            raise ValueError("Structural degrees of freedom must be positive.")
        return value

    def declared_forms(self) -> tuple[CochainSpaceSpec, ...]:
        """Declare canonical cochain spaces for groundwater storage dynamics.

        References:
          - RHESSys compute_delta_water.c groundwater storage profile.
          - Beven and Kirkby (1979), variable contributing area hydrology.
          - Hirani (2003), Discrete Exterior Calculus.
        Mathematical expression:
          S_dot = G_gw u_gw - q_base, S(z) = integral n0 * exp(-z/p) dz.
        Parameter meanings:
          S is groundwater storage [mm], z is water table depth [m], and
          u_gw contains recharge, lateral inflow, pumping, and baseflow ports.
        """

        n_dofs = self._structural_n_dofs()
        return (
            CochainSpaceSpec(
                complex_id="hydrosphere_groundwater_dynamics",
                degree=0,
                primality="primal",
                n_dofs=n_dofs,
                units="mm",
                extensivity="integral",
                physical_domain="hydrosphere",
                semantic_name="groundwater_storage",
                orientation_convention="positive_storage",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="hydrosphere_groundwater_dynamics",
                degree=0,
                primality="primal",
                n_dofs=n_dofs,
                units="m",
                extensivity="intensive",
                physical_domain="hydrosphere",
                semantic_name="water_table_depth",
                orientation_convention="positive_downward_depth",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="hydrosphere_groundwater_dynamics",
                degree=1,
                primality="dual",
                n_dofs=n_dofs,
                units="kg m-2 s-1",
                extensivity="integral",
                physical_domain="hydrosphere",
                semantic_name="baseflow_flux",
                orientation_convention="outward_positive",
                boundary_role="port",
            ),
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Declare ParameterManager keys used by runtime and structural builders."""

        return (
            "hydrosphere.groundwater.surface_porosity",
            "hydrosphere.groundwater.porosity_decay_length",
            "hydrosphere.groundwater.max_water_table_depth",
            "hydrosphere.groundwater.bedrock_depth",
            "hydrosphere.groundwater.specific_yield",
            "hydrosphere.groundwater.minimum_water_table_depth",
            "hydrosphere.groundwater.minimum_depth_factor",
            "hydrosphere.groundwater.drainage_time_constant",
            "hydrosphere.groundwater.enable_lateral_flow",
            "hydrosphere.groundwater.topographic_index",
            "hydrosphere.numerical.minimum_timestep_tolerance",
            "hydrosphere.numerical.epsilon_small",
        )

    def build_hodge_contract(
        self,
        n_dofs: Optional[int] = None,
        *,
        strict: bool = True,
    ) -> HodgeStarSpec:
        """Build the groundwater storage Hodge contract.

        References:
          - RHESSys compute_delta_water.c storage integration.
          - Wigmosta et al. (1994), distributed hydrology-vegetation groundwater.
          - van der Schaft and Maschke (1995), dissipative port systems.
        Mathematical expression:
          star_Sy = diag(rho_w * S_y), with rho_w > 0 and S_y > 0.
        Parameter meanings:
          rho_w is water density [kg m-3]; S_y is specific yield [dimensionless].
        """

        n_dofs = self._structural_n_dofs(n_dofs=n_dofs)
        mobility_value = _scalar_from_value(self.water_density_reference) * _scalar_from_value(
            self.specific_yield
        )
        values = torch.full(
            (n_dofs,),
            mobility_value,
            dtype=torch.float32,
            device=getattr(self, "device", torch.device("cpu")),
        )
        if bool(torch.any(values <= 0.0)):
            raise ValueError("Groundwater storage mobility must be positive.")

        domain = CochainSpaceSpec(
            complex_id="hydrosphere_groundwater_dynamics",
            degree=0,
            primality="primal",
            n_dofs=n_dofs,
            units="m",
            extensivity="intensive",
            physical_domain="hydrosphere",
            semantic_name="water_table_head",
            orientation_convention="positive_downward_depth",
        )
        codomain = CochainSpaceSpec(
            complex_id="hydrosphere_groundwater_dynamics",
            degree=0,
            primality="dual",
            n_dofs=n_dofs,
            units="kg m-2",
            extensivity="integral",
            physical_domain="hydrosphere",
            semantic_name="groundwater_storage_mass",
            orientation_convention="positive_storage",
            boundary_role="interior",
        )
        hodge = HodgeStarSpec(
            name=f"{self.name}_groundwater_storage_hodge",
            domain=domain,
            codomain=codomain,
            matrix=torch.diag(values),
            parameter_keys=(
                "hydrosphere.groundwater.specific_yield",
            ),
            metric_keys=("hydrosphere.groundwater.porosity_decay_length",),
            material_law="exponential_porosity_groundwater_storage",
        )
        if strict:
            hodge.validate(passive=True)
        return hodge

    def build_hodge_terms(
        self,
        complex_bundle: Any = None,
        resolved_parameters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[HodgeStarSpec, ...]:
        """Build groundwater Hodge terms for IPHS assembly."""

        return (
            self.build_hodge_contract(
                n_dofs=self._structural_n_dofs(complex_bundle=complex_bundle),
                strict=True,
            ),
        )

    def energy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return storage functional and effort terms for non-strict IPHS audits."""

        if not self._has_groundwater_state(state):
            return {
                "storage_functional": "H_h = 0.5 * z_h.T star_Sy z_h",
                "effort": "e_h = delta H_h / delta z_h = star_Sy z_h",
            }
        normalized = self._validate_state_contract(state, require_fluxes=False)
        depth = normalized["water_table_depth"]
        return {
            "storage_functional": self._groundwater_storage_functional(depth),
            "effort": self._groundwater_effort(depth),
        }

    def entropy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return non-negative groundwater dissipation terms."""

        if not self._has_groundwater_state(state):
            return {
                "dissipation_rate": "sigma_h = z_h.T (star_Sy / tau_gw) z_h",
                "thermodynamic_force": "X_h = z_h",
            }
        normalized = self._validate_state_contract(state, require_fluxes=False)
        depth = normalized["water_table_depth"]
        return {
            "dissipation_rate": self._groundwater_dissipation_rate(depth),
            "thermodynamic_force": depth,
        }

    def reversible_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> None:
        """Return no reversible interconnection for scalar groundwater storage."""

        return None

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the positive groundwater storage mobility contribution.

        References:
          - RHESSys groundwater storage, Wigmosta et al. (1994), and van der Schaft and Maschke (1995).
        Mathematical expression:
          M_h = diag(rho_w * S_y), sigma = X_h.T M_h X_h >= 0.
        Parameter meanings:
          S_y is specific yield [dimensionless].
        """

        hodge = self.build_hodge_contract(
            n_dofs=self._structural_n_dofs(complex_bundle=complex_bundle),
            strict=True,
        )
        matrix = hodge.matrix
        return {
            "state_vars": tuple(
                f"groundwater_head[{idx}]" for idx in range(int(matrix.shape[0]))
            ),
            "J": torch.zeros_like(matrix),
            "M": matrix,
            "ports": self.ports(),
        }

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Declare every boundary term in the groundwater storage ledger."""

        effort_space = CochainSpaceSpec(
            complex_id="hydrosphere_groundwater_dynamics",
            degree=0,
            primality="primal",
            n_dofs=1,
            units="m",
            extensivity="intensive",
            physical_domain="hydrosphere",
            semantic_name="groundwater_head",
            orientation_convention="positive_downward_depth",
            boundary_role="port",
        )
        flow_kwargs = {
            "complex_id": "hydrosphere_groundwater_dynamics",
            "degree": 1,
            "primality": "dual",
            "n_dofs": 1,
            "units": "kg m-2 s-1",
            "extensivity": "integral",
            "physical_domain": "hydrosphere",
            "orientation_convention": "outward_positive",
            "boundary_role": "port",
        }
        def make_port(
            name: str,
            flow_var: str,
            semantic_name: str,
            convention: str,
        ) -> PortSpecIPHS:
            return PortSpecIPHS(
                name=name,
                effort_var="groundwater_head",
                flow_var=flow_var,
                effort_space=effort_space,
                flow_space=CochainSpaceSpec(
                    **flow_kwargs,
                    semantic_name=semantic_name,
                ),
                power_units="kg m-1 s-1",
                positive_power_convention=convention,
                port_kind="matter",
            )

        return (
            make_port(
                "groundwater_recharge_flux",
                "drainage_flux",
                "groundwater_recharge_mass_flux",
                "into_system",
            ),
            make_port(
                "groundwater_lateral_inflow_flux",
                "lateral_inflow",
                "groundwater_lateral_inflow_mass_flux",
                "into_system",
            ),
            make_port(
                "groundwater_pumping_flux",
                "groundwater_pumping",
                "groundwater_pumping_mass_flux",
                "out_of_system",
            ),
            make_port(
                "groundwater_capillary_rise_flux",
                "capillary_rise_flux",
                "groundwater_capillary_rise_mass_flux",
                "out_of_system",
            ),
            make_port(
                "groundwater_baseflow_flux",
                "baseflow_flux",
                "groundwater_baseflow_mass_flux",
                "out_of_system",
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
    
    def forward(self, 
        # References:
        #   Wigmosta et al. (1994); Beven and Kirkby (1979); Higham (2002).
        # Mathematical expression:
        #   dS_req=dt*(q_drain+q_lateral-q_pump-q_capillary-q_base);
        #   n(z)=n0*exp(-z/p); S_profile(z)=1000*integral_z^bedrock n(s) ds;
        #   S1=S_profile(z1)+(S0-S_profile(z0)); residual=(S1-S0)-dS_req.
        # Parameter meanings:
        #   q values are converted depth rates; z is water-table depth; n0 and p
        #   parameterize porosity. Depth/storage bounds may leave a disclosed residual.
        # Units:
        #   S and residual [mm]; z/p/bedrock [m]; q [mm s-1]; input mass rates
        #   [kg m-2 s-1]; dt [s].
        # Domain/constraints:
        #   Finite positive scalar dt and declared nonnegative boundary magnitudes;
        #   retain the original storage/profile bias and report bounds/roundoff effects.
        # Sign convention:
        #   Depth is positive downward. Recharge/lateral inflow increase storage and
        #   raise the water table; pumping/capillary/baseflow export storage.
                state: StateDict, 
                time: float, 
                dt: float,
                **kwargs) -> FluxDict:
        """
        Update groundwater storage and compute diagnostic variables.
        
        Parameters
        ----------
        state : StateDict
            Current system state including drainage inputs
        time : float
            Current simulation time [s]
        dt : float  
            Time step [s]
        **kwargs
            Additional parameters and forcing data
            
        Returns
        -------
        FluxDict
            Updated groundwater storage and diagnostic variables
            
        Raises
        ------
        ValueError
            If required state variables are missing or invalid
        """
        state = self._validate_state_contract(state, require_fluxes=True)
        current_storage = state["groundwater_storage"]
        current_wt_depth = state["water_table_depth"]

        dt_tensor = self._ensure_tensor(dt).to(
            device=current_storage.device, dtype=current_storage.dtype
        )
        if dt_tensor.numel() != 1 or not bool(torch.isfinite(dt_tensor).all()):
            raise ValueError("GroundwaterDynamics dt must be one finite scalar.")
        if bool(dt_tensor <= 0.0):
            raise ValueError("GroundwaterDynamics dt must be positive.")
        dt_safe = dt_tensor

        def as_storage_tensor(key: str) -> Tensor:
            return state[key].to(device=current_storage.device, dtype=current_storage.dtype)

        drainage_flux = as_storage_tensor("drainage_flux")
        capillary_flux = as_storage_tensor("capillary_rise_flux")
        lateral_flux = as_storage_tensor("lateral_inflow")
        pumping_flux = as_storage_tensor("groundwater_pumping")

        # References: Wigmosta et al. (1994); Beven and Kirkby (1979).
        # Mathematical expression:
        #   Delta S = [(F_drain + F_lat - F_pump - F_cap - F_base) / rho_w]
        #             * 1000 * dt.
        # Parameter meanings:
        #   F_* are non-negative boundary mass fluxes [kg m-2 s-1], rho_w is
        #   water density [kg m-3], S is storage [mm], and dt is time [s].
        #   The flux-to-depth-rate step is UnitConverter's single CF liquid-water-
        #   equivalent rule (d = F / rho_w, rho_w = water.density_liquid), selected by
        #   the declared water variable name.
        drainage_rate = unit_converter.convert(drainage_flux, "drainage_flux", "kg m-2 s-1", "mm s-1")
        capillary_rate = unit_converter.convert(capillary_flux, "capillary_rise_flux", "kg m-2 s-1", "mm s-1")
        lateral_inflow_rate = unit_converter.convert(lateral_flux, "lateral_inflow", "kg m-2 s-1", "mm s-1")
        pumping_rate = unit_converter.convert(pumping_flux, "groundwater_pumping", "kg m-2 s-1", "mm s-1")

        # Apply lateral flow enable flag as a multiplier (0/1) to avoid Python branching.
        lateral_flag = self._ensure_tensor(self.enable_lateral_flow).to(
            device=lateral_inflow_rate.device, dtype=lateral_inflow_rate.dtype
        )
        lateral_inflow_rate = lateral_inflow_rate * lateral_flag

        # Explicit Euler evaluates recession from the accepted start-of-step state.
        baseflow_depth_rate = self._compute_baseflow(current_storage, current_wt_depth)
        baseflow_flux = unit_converter.convert(baseflow_depth_rate, "baseflow_flux", "mm s-1", "kg m-2 s-1")

        # Canonical storage calculation advances storage and water table together.
        new_storage, new_wt_depth = self._update_storage_dynamics(
            current_storage=current_storage,
            current_wt_depth=current_wt_depth,
            drainage_rate=drainage_rate,
            lateral_rate=lateral_inflow_rate,
            pumping_rate=pumping_rate,
            capillary_rate=capillary_rate,
            baseflow_rate=baseflow_depth_rate,
            dt=dt_safe,
        )

        net_depth_rate = (
            drainage_rate
            + lateral_inflow_rate
            - pumping_rate
            - capillary_rate
            - baseflow_depth_rate
        )
        net_input = net_depth_rate * dt_safe
        
        # Compute diagnostic variables
        storage_deficit = self._compute_storage_deficit(new_wt_depth)
        saturation_fraction = self._compute_saturation_fraction(new_storage, storage_deficit)
        # Effective porosity at current water table
        effective_porosity = self._compute_porosity_at_depth(new_wt_depth)
        
        # Calculate storage increment (NOT absolute storage)
        storage_increment = new_storage - current_storage
        storage_balance_residual = storage_increment - net_input
        # References:
        #   - Higham (2002), Accuracy and Stability of Numerical Algorithms.
        # Mathematical expression:
        #   b_round = gamma_8 (|S_new| + |S_old| + |Delta S_port|),
        #   gamma_n = n eps / (1 - n eps).
        # Parameter meanings:
        #   eps is machine precision for the runtime dtype; n=8 bounds the
        #   floating additions, subtractions, and scaling in this ledger.
        #   b_round is a numerical audit bound [mm], not a scientific tolerance.
        machine_epsilon = torch.finfo(storage_balance_residual.dtype).eps
        ledger_operation_count = storage_balance_residual.new_tensor(8.0)
        roundoff_gamma = (
            ledger_operation_count * machine_epsilon
            / (1.0 - ledger_operation_count * machine_epsilon)
        )
        storage_balance_roundoff_bound = roundoff_gamma * (
            torch.abs(new_storage)
            + torch.abs(current_storage)
            + torch.abs(net_input)
        )
        bedrock_depth = self._ensure_tensor(self.bedrock_depth).to(
            device=new_wt_depth.device, dtype=new_wt_depth.dtype
        )
        minimum_depth = self._ensure_tensor(self.minimum_water_table_depth).to(
            device=new_wt_depth.device, dtype=new_wt_depth.dtype
        )
        water_table_bounds_residual = (
            torch.clamp(minimum_depth - new_wt_depth, min=0.0)
            + torch.clamp(new_wt_depth - bedrock_depth, min=0.0)
        )
        groundwater_storage_negative_residual = torch.clamp(-new_storage, min=0.0)
        iphs_audit = self._structured_iphs_audit_terms(
            current_wt_depth=current_wt_depth,
            new_wt_depth=new_wt_depth,
            drainage_rate=drainage_rate,
            lateral_inflow_rate=lateral_inflow_rate,
            pumping_rate=pumping_rate,
            capillary_rate=capillary_rate,
            baseflow_rate=baseflow_depth_rate,
            dt_safe=dt_safe,
        )
        
        return {
            "groundwater_storage": UpdateItem(
                value=new_storage,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="mm",
                source_functor=self.name,
            ),
            "water_table_depth": UpdateItem(
                value=new_wt_depth,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="m",
                source_functor=self.name,
            ),
            "baseflow_flux": UpdateItem(
                value=baseflow_flux,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="kg m-2 s-1",
                source_functor=self.name,
            ),
            "diagnostics.groundwater": {
                "effective_porosity": effective_porosity,  # dimensionless
                "groundwater_storage_increment": storage_increment,  # mm
                "groundwater_storage_change_rate": storage_increment / dt_safe,  # mm/s
                "water_table_change_rate": (new_wt_depth - current_wt_depth) / dt_safe,  # m/s
                "net_groundwater_flux": net_depth_rate,  # mm/s
                "storage_deficit": storage_deficit,  # mm
                "saturation_fraction": saturation_fraction,  # dimensionless
                "baseflow_depth_rate": baseflow_depth_rate,  # mm/s
                "storage_balance_residual": storage_balance_residual,  # mm
                "storage_balance_roundoff_bound": (
                    storage_balance_roundoff_bound
                ),  # mm
                "water_table_bounds_residual": water_table_bounds_residual,  # m
                "groundwater_storage_negative_residual": groundwater_storage_negative_residual,  # mm
                "energy_storage": iphs_audit["energy_storage"],
                "dissipation_rate": iphs_audit["dissipation_rate"],
                "dH_dt_audit": iphs_audit["dH_dt_audit"],
                "port_power_residual": iphs_audit["port_power_residual"],
            },
        }

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Evaluate the open five-port groundwater storage ledger.

        References:
          - Beven and Kirkby (1979), variable source area hydrology.
          - Wigmosta et al. (1994), distributed groundwater storage.
        Mathematical expression:
          Delta S = (F_drain + F_lateral - F_pumping - F_capillary
          - F_baseflow) * 1000 / rho_w * dt.
        Parameter meanings:
          Delta S is groundwater storage change [mm], the five F terms are
          boundary water mass fluxes [kg m-2 s-1], rho_w is water density
          [kg m-3], and dt is the step duration [s].
        Open-system route:
          Drainage and lateral flow enter the control volume; pumping,
          capillary rise, and baseflow leave it. The existing
          storage_balance_residual reports the complete ledger in mm.
        """

        del input_state
        diagnostics_key = "diagnostics.groundwater"
        if diagnostics_key not in output_updates:
            raise KeyError(f"Missing conservation output '{diagnostics_key}'")
        diagnostics = output_updates[diagnostics_key]
        if isinstance(diagnostics, UpdateItem):
            diagnostics = diagnostics.value
        if not isinstance(diagnostics, Mapping):
            raise TypeError("Groundwater conservation diagnostics must be a mapping")

        def _diagnostic_tensor(key: str) -> Tensor:
            if key not in diagnostics:
                raise KeyError(f"Missing conservation diagnostic '{key}'")
            value = diagnostics[key]
            if isinstance(value, UpdateItem):
                value = value.value
            if not isinstance(value, Tensor):
                raise TypeError(
                    f"Groundwater {key} must be a torch.Tensor"
                )
            if value.numel() == 0:
                raise ValueError(f"Groundwater {key} must be non-empty")
            if not value.is_floating_point():
                raise TypeError(
                    f"Groundwater {key} must use a floating dtype"
                )
            if not bool(torch.all(torch.isfinite(value))):
                raise ValueError(f"Groundwater {key} must be finite")
            return value

        residual = _diagnostic_tensor("storage_balance_residual")
        roundoff_bound = _diagnostic_tensor(
            "storage_balance_roundoff_bound"
        )
        if bool(torch.any(roundoff_bound < 0.0)):
            raise ValueError(
                "Groundwater storage_balance_roundoff_bound must be "
                "non-negative"
            )
        residual, roundoff_bound = torch.broadcast_tensors(
            residual,
            roundoff_bound,
        )
        magnitude = torch.max(
            torch.clamp(torch.abs(residual) - roundoff_bound, min=0.0)
        )
        return {"mass": float(magnitude.detach().item())}

    def _has_groundwater_state(self, state: Mapping[str, Any]) -> bool:
        """Return True when a mapping contains groundwater state variables."""

        return "groundwater_storage" in state or "water_table_depth" in state

    def _groundwater_storage_mobility(
        self,
        depth: Tensor,
        water_density: Optional[Tensor] = None,
    ) -> Tensor:
        """Return the positive storage Hodge diagonal value."""
        # References:
        #   van der Schaft and Maschke (1995); Wigmosta et al. (1994).
        # Mathematical expression:
        #   W=rho_w*Sy; H=0.5*sum(W*z^2); e=W*z; y=-z;
        #   sigma=sum(W*z^2/tau); Pres=(H1-H0)/dt+sigma-sum(y*u).
        # Parameter meanings:
        #   Sy is specific yield; u is signed net boundary mass input;
        #   z is positive-downward depth. These are existing non-strict audit terms.
        # Units:
        #   W [kg m-3], H [kg m-1], e [kg m-2], y [m];
        #   sigma, y*u and Pres [kg m-1 s-1], not W or thermodynamic entropy production.
        # Domain/constraints:
        #   Do not infer strict IPHS equivalence from these diagnostics; the residual
        #   remains explicit and does not repair the physical storage balance.
        # Sign convention:
        #   u is positive inward while y=-z; sigma is nonnegative and Pres is signed.

        if water_density is None:
            water_density = self._ensure_tensor(self.water_density_reference).to(
                device=depth.device, dtype=depth.dtype
            )
        specific_yield = self._ensure_tensor(self.specific_yield).to(
            device=depth.device, dtype=depth.dtype
        )
        return water_density * specific_yield

    def _groundwater_effort(self, depth: Tensor) -> Tensor:
        """Compute storage effort for the groundwater head cochain."""

        # References:
        #   van der Schaft and Maschke (1995); Wigmosta et al. (1994).
        # Mathematical expression:
        #   W=rho_w*Sy; H=0.5*sum(W*z^2); e=W*z; y=-z;
        #   sigma=sum(W*z^2/tau); Pres=(H1-H0)/dt+sigma-sum(y*u).
        # Parameter meanings:
        #   Sy is specific yield; u is signed net boundary mass input;
        #   z is positive-downward depth. These are existing non-strict audit terms.
        # Units:
        #   W [kg m-3], H [kg m-1], e [kg m-2], y [m];
        #   sigma, y*u and Pres [kg m-1 s-1], not W or thermodynamic entropy production.
        # Domain/constraints:
        #   Do not infer strict IPHS equivalence from these diagnostics; the residual
        #   remains explicit and does not repair the physical storage balance.
        # Sign convention:
        #   u is positive inward while y=-z; sigma is nonnegative and Pres is signed.
        depth_tensor = self._ensure_tensor(depth)
        mobility = self._groundwater_storage_mobility(depth_tensor)
        return mobility * depth_tensor

    def _groundwater_port_effort(self, depth: Tensor) -> Tensor:
        """Map storage effort to the conjugate positive-inflow port effort.

        References:
          - van der Schaft and Maschke (1995), port-Hamiltonian systems.
          - Pasumarthy et al. (2012), port-Hamiltonian open-channel flow.
        Mathematical expression:
          y = G.T e, e = rho_w S_y z, G = -1 / (rho_w S_y), so y = -z.
        Parameter meanings:
          z is positive-downward water-table depth [m], rho_w is water
          density [kg m-3], and S_y is specific yield [1].
        Units and route:
          G maps boundary mass flux [kg m-2 s-1] to depth rate [m s-1].
          The resulting y [m] pairs with the signed open-boundary mass flux
          to give head-weighted port power [kg m-1 s-1].
        """
        # References:
        #   van der Schaft and Maschke (1995); Wigmosta et al. (1994).
        # Mathematical expression:
        #   W=rho_w*Sy; H=0.5*sum(W*z^2); e=W*z; y=-z;
        #   sigma=sum(W*z^2/tau); Pres=(H1-H0)/dt+sigma-sum(y*u).
        # Parameter meanings:
        #   Sy is specific yield; u is signed net boundary mass input;
        #   z is positive-downward depth. These are existing non-strict audit terms.
        # Units:
        #   W [kg m-3], H [kg m-1], e [kg m-2], y [m];
        #   sigma, y*u and Pres [kg m-1 s-1], not W or thermodynamic entropy production.
        # Domain/constraints:
        #   Do not infer strict IPHS equivalence from these diagnostics; the residual
        #   remains explicit and does not repair the physical storage balance.
        # Sign convention:
        #   u is positive inward while y=-z; sigma is nonnegative and Pres is signed.

        depth_tensor = self._ensure_tensor(depth)
        state_effort = self._groundwater_effort(depth_tensor)
        mobility = self._groundwater_storage_mobility(depth_tensor)
        port_input_map = -torch.reciprocal(mobility)
        return port_input_map * state_effort

    def _groundwater_storage_functional(self, depth: Tensor) -> Tensor:
        """Compute the structured non-strict IPHS storage functional."""

        # References:
        #   van der Schaft and Maschke (1995); Wigmosta et al. (1994).
        # Mathematical expression:
        #   W=rho_w*Sy; H=0.5*sum(W*z^2); e=W*z; y=-z;
        #   sigma=sum(W*z^2/tau); Pres=(H1-H0)/dt+sigma-sum(y*u).
        # Parameter meanings:
        #   Sy is specific yield; u is signed net boundary mass input;
        #   z is positive-downward depth. These are existing non-strict audit terms.
        # Units:
        #   W [kg m-3], H [kg m-1], e [kg m-2], y [m];
        #   sigma, y*u and Pres [kg m-1 s-1], not W or thermodynamic entropy production.
        # Domain/constraints:
        #   Do not infer strict IPHS equivalence from these diagnostics; the residual
        #   remains explicit and does not repair the physical storage balance.
        # Sign convention:
        #   u is positive inward while y=-z; sigma is nonnegative and Pres is signed.
        depth_tensor = self._ensure_tensor(depth)
        effort = self._groundwater_effort(depth_tensor)
        return 0.5 * torch.sum(depth_tensor * effort)

    def _groundwater_dissipation_rate(self, depth: Tensor) -> Tensor:
        """Compute a non-negative groundwater recession dissipation audit."""

        # References:
        #   van der Schaft and Maschke (1995); Wigmosta et al. (1994).
        # Mathematical expression:
        #   W=rho_w*Sy; H=0.5*sum(W*z^2); e=W*z; y=-z;
        #   sigma=sum(W*z^2/tau); Pres=(H1-H0)/dt+sigma-sum(y*u).
        # Parameter meanings:
        #   Sy is specific yield; u is signed net boundary mass input;
        #   z is positive-downward depth. These are existing non-strict audit terms.
        # Units:
        #   W [kg m-3], H [kg m-1], e [kg m-2], y [m];
        #   sigma, y*u and Pres [kg m-1 s-1], not W or thermodynamic entropy production.
        # Domain/constraints:
        #   Do not infer strict IPHS equivalence from these diagnostics; the residual
        #   remains explicit and does not repair the physical storage balance.
        # Sign convention:
        #   u is positive inward while y=-z; sigma is nonnegative and Pres is signed.
        depth_tensor = self._ensure_tensor(depth)
        mobility = self._groundwater_storage_mobility(depth_tensor)
        tau = self._ensure_tensor(self.drainage_time_constant).to(
            device=depth_tensor.device, dtype=depth_tensor.dtype
        )
        tau = torch.clamp(tau, min=self._ensure_tensor(self.minimum_timestep_tolerance).to(
            device=depth_tensor.device, dtype=depth_tensor.dtype
        ))
        return torch.sum((mobility / tau) * depth_tensor * depth_tensor)

    def _structured_iphs_audit_terms(
        self,
        current_wt_depth: Tensor,
        new_wt_depth: Tensor,
        drainage_rate: Tensor,
        lateral_inflow_rate: Tensor,
        pumping_rate: Tensor,
        capillary_rate: Tensor,
        baseflow_rate: Tensor,
        dt_safe: Tensor,
    ) -> Dict[str, Tensor]:
        """Compute structured non-strict IPHS storage and port diagnostics."""

        # References:
        #   - Wigmosta et al. (1994), groundwater storage dynamics.
        #   - Beven and Kirkby (1979), variable contributing area hydrology.
        #   - van der Schaft and Maschke (1995), port-Hamiltonian systems.
        # Mathematical expression:
        #   dH/dt = -sigma_h + y_h.T u_h + P_residual.
        #   y_h = G_h.T e_h = -z_h for positive-downward depth.
        #   sigma_h = z_h.T (star_Sy / tau_gw) z_h >= 0.
        # Parameter meanings:
        #   u_h contains signed recharge, lateral, pumping, capillary, and
        #   baseflow mass fluxes [kg m-2 s-1]; y_h is hydraulic head [m].
        # Units and route:
        #   y_h.T u_h has units [kg m-1 s-1] and all five terms are explicit
        #   open or coupled boundary ports.
        storage_before = self._groundwater_storage_functional(current_wt_depth)
        storage_after = self._groundwater_storage_functional(new_wt_depth)
        dH_dt = (storage_after - storage_before) / dt_safe
        dissipation = self._groundwater_dissipation_rate(current_wt_depth)
        net_port_depth_rate = (
            drainage_rate
            + lateral_inflow_rate
            - pumping_rate
            - capillary_rate
            - baseflow_rate
        )
        net_port_mass_flux = unit_converter.convert(
            net_port_depth_rate, "water_flux", "mm s-1", "kg m-2 s-1"
        )
        port_effort = self._groundwater_port_effort(current_wt_depth)
        port_power = torch.sum(port_effort * net_port_mass_flux)
        port_power_residual = dH_dt + dissipation - port_power
        return {
            "energy_storage": storage_after,
            "dissipation_rate": dissipation,
            "dH_dt_audit": dH_dt,
            "port_power_residual": port_power_residual,
        }
    
    def _update_storage_dynamics(self, 
                               current_storage: Tensor,
                               current_wt_depth: Tensor, 
                               drainage_rate: Tensor,
                               lateral_rate: Tensor,
                               pumping_rate: Tensor,
                               capillary_rate: Tensor,
                               baseflow_rate: Tensor,
                               dt: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Update groundwater storage with explicit water table dynamics.
        
        Mathematical Expression
        -----------------------
        Storage change: DeltaS = integral[z_old to z_new] n(z) dz
        where n(z) = n0 * exp(-z/p)
        
        Solved analytically by inverting the exponential porosity integral;
        no iterative water-table approximation is used.
        """
        # References:
        #   Wigmosta et al. (1994); Beven and Kirkby (1979); Higham (2002).
        # Mathematical expression:
        #   dS_req=dt*(q_drain+q_lateral-q_pump-q_capillary-q_base);
        #   n(z)=n0*exp(-z/p); S_profile(z)=1000*integral_z^bedrock n(s) ds;
        #   S1=S_profile(z1)+(S0-S_profile(z0)); residual=(S1-S0)-dS_req.
        # Parameter meanings:
        #   q values are converted depth rates; z is water-table depth; n0 and p
        #   parameterize porosity. Depth/storage bounds may leave a disclosed residual.
        # Units:
        #   S and residual [mm]; z/p/bedrock [m]; q [mm s-1]; input mass rates
        #   [kg m-2 s-1]; dt [s].
        # Domain/constraints:
        #   Finite positive scalar dt and declared nonnegative boundary magnitudes;
        #   retain the original storage/profile bias and report bounds/roundoff effects.
        # Sign convention:
        #   Depth is positive downward. Recharge/lateral inflow increase storage and
        #   raise the water table; pumping/capillary/baseflow export storage.
        net_input = (
            drainage_rate
            + lateral_rate
            - pumping_rate
            - capillary_rate
            - baseflow_rate
        ) * dt

        bedrock_depth = self._ensure_tensor(self.bedrock_depth).to(
            device=current_storage.device, dtype=current_storage.dtype
        )
        min_depth = self._ensure_tensor(self.minimum_water_table_depth).to(
            device=current_storage.device, dtype=current_storage.dtype
        )
        z_current = torch.clamp(
            self._ensure_tensor(current_wt_depth).to(
                device=current_storage.device, dtype=current_storage.dtype
            ),
            min=min_depth,
            max=bedrock_depth,
        )

        z_new = self._solve_water_table_change(z_current, net_input)
        z_new = torch.clamp(z_new, min=min_depth, max=bedrock_depth)

        storage_current_profile = UnitSystem.convert(
            self._integrate_storage(z_current, bedrock_depth), "m", "mm"
        )
        storage_new_profile = UnitSystem.convert(
            self._integrate_storage(z_new, bedrock_depth), "m", "mm"
        )
        storage_bias = current_storage - storage_current_profile
        new_storage = torch.clamp(storage_new_profile + storage_bias, min=0.0)

        return new_storage, z_new
    
    def _solve_water_table_change(self, z_current: Tensor, net_input_mm: Tensor) -> Tensor:
        """
        Solve for new water table depth given cumulative water input or loss.

        Analytical solution of the exponential porosity profile integral:
            integral = n0 * p * [exp(-z_lower/p) - exp(-z_upper/p)]
        """
        # References:
        #   Wigmosta et al. (1994); RHESSys exponential-porosity storage formulation.
        # Mathematical expression:
        #   n(z)=n0*exp(-z/p); integral_a^b n(z) dz=n0*p*(exp(-a/p)-exp(-b/p));
        #   z1=-p*log(exp(-z0/p)+dS_m/(n0*p)), subject to declared bounds.
        # Parameter meanings:
        #   n0 is surface porosity; p is decay length; dS_m is the signed
        #   net storage input after the existing mm-to-m conversion.
        # Units:
        #   n0 [1]; z,a,b,p,dS_m [m]; integral [m of water].
        # Domain/constraints:
        #   Positive n0 and p; inversion argument and depth are bounded by the
        #   current soil/bedrock domain. The integral helper returns the nonnegative
        #   volume between its two ordered depths.
        # Sign convention:
        #   Positive dS_m reduces positive-downward z; the ordered interval integral
        #   is a capacity magnitude, not a signed transfer by itself.
        z_current = self._ensure_tensor(z_current)
        net_input_m = UnitSystem.convert(
            self._ensure_tensor(net_input_mm).to(
                device=z_current.device, dtype=z_current.dtype
            ),
            "mm",
            "m",
        )

        p = self._ensure_tensor(self.porosity_decay_length).to(
            device=z_current.device, dtype=z_current.dtype
        )
        n0 = self._ensure_tensor(self.surface_porosity).to(
            device=z_current.device, dtype=z_current.dtype
        )
        bedrock_depth = self._ensure_tensor(self.bedrock_depth).to(
            device=z_current.device, dtype=z_current.dtype
        )
        min_depth = self._ensure_tensor(self.minimum_water_table_depth).to(
            device=z_current.device, dtype=z_current.dtype
        )

        epsilon = self._ensure_tensor(self.epsilon_small).to(
            device=z_current.device, dtype=z_current.dtype
        )
        capacity = torch.clamp(n0 * p, min=epsilon)
        exp_current = torch.exp(-z_current / p)
        min_exp = torch.exp(-bedrock_depth / p)

        small = torch.abs(net_input_m) < epsilon

        # Recharge (water table moves upward, depth decreases).
        arg_pos = torch.clamp(exp_current + net_input_m / capacity, min=min_exp, max=1.0)
        z_new_pos = -p * torch.log(arg_pos)
        z_new_pos = torch.minimum(z_new_pos, z_current)
        z_new_pos = torch.clamp(z_new_pos, min=min_depth)

        # Discharge (water table moves downward, depth increases).
        deficit = -net_input_m
        arg_neg = torch.clamp(exp_current - deficit / capacity, min=min_exp)
        z_new_neg = -p * torch.log(arg_neg)
        z_new_neg = torch.maximum(z_new_neg, z_current)
        z_new_neg = torch.minimum(z_new_neg, bedrock_depth)

        z_new = torch.where(net_input_m >= 0.0, z_new_pos, z_new_neg)
        z_new = torch.where(small, z_current, z_new)
        return z_new
    
    # References: RHESSys compute_delta_water.c.
    # Formula: integral = n0 * p * (exp(-z1/p) - exp(-z2/p)).
    # Parameters: n0 (surface_porosity, dimensionless), p (porosity_decay_length, m), z1/z2 (m).
    def _integrate_storage(self, z_start: Tensor, z_end: Tensor) -> Tensor:
        """
        Integrate porosity profile between depths to get storage change.

        Mathematical Expression
        -----------------------
        integral[z1 to z2] n0 * exp(-z/p) dz = n0 * p * [exp(-z1/p) - exp(-z2/p)]

        Returns
        -------
        Tensor
            Storage change [m] (positive = more storage)
        """
        # References:
        #   Wigmosta et al. (1994); RHESSys exponential-porosity storage formulation.
        # Mathematical expression:
        #   n(z)=n0*exp(-z/p); integral_a^b n(z) dz=n0*p*(exp(-a/p)-exp(-b/p));
        #   z1=-p*log(exp(-z0/p)+dS_m/(n0*p)), subject to declared bounds.
        # Parameter meanings:
        #   n0 is surface porosity; p is decay length; dS_m is the signed
        #   net storage input after the existing mm-to-m conversion.
        # Units:
        #   n0 [1]; z,a,b,p,dS_m [m]; integral [m of water].
        # Domain/constraints:
        #   Positive n0 and p; inversion argument and depth are bounded by the
        #   current soil/bedrock domain. The integral helper returns the nonnegative
        #   volume between its two ordered depths.
        # Sign convention:
        #   Positive dS_m reduces positive-downward z; the ordered interval integral
        #   is a capacity magnitude, not a signed transfer by itself.
        z_start_t = self._ensure_tensor(z_start)
        z_end_t = self._ensure_tensor(z_end).to(device=z_start_t.device, dtype=z_start_t.dtype)
        z_lower = torch.minimum(z_start_t, z_end_t)
        z_upper = torch.maximum(z_start_t, z_end_t)

        p = self._ensure_tensor(self.porosity_decay_length).to(
            device=z_start_t.device, dtype=z_start_t.dtype
        )
        n0 = self._ensure_tensor(self.surface_porosity).to(
            device=z_start_t.device, dtype=z_start_t.dtype
        )
        term1 = torch.exp(-z_lower / p)
        term2 = torch.exp(-z_upper / p)
        integral = n0 * p * (term1 - term2)

        return torch.where(z_start_t == z_end_t, torch.zeros_like(integral), integral)
    
    # References: RHESSys compute_delta_water.c.
    # Formula: n(z) = n0 * exp(-z / p).
    # Parameters: n0 (surface_porosity, dimensionless), p (porosity_decay_length, m), z (m).
    def _compute_porosity_at_depth(self, depth: Tensor) -> Tensor:
        """
        Compute porosity at specified depth.

        Parameters
        ----------
        depth : Tensor
            Depth below surface [m]

        Returns
        -------
        Tensor
            Porosity at depth [dimensionless]
        """
        # References:
        #   Wigmosta et al. (1994); RHESSys exponential-porosity storage formulation.
        # Mathematical expression:
        #   n(z)=n0*exp(-z/p); integral_a^b n(z) dz=n0*p*(exp(-a/p)-exp(-b/p));
        #   z1=-p*log(exp(-z0/p)+dS_m/(n0*p)), subject to declared bounds.
        # Parameter meanings:
        #   n0 is surface porosity; p is decay length; dS_m is the signed
        #   net storage input after the existing mm-to-m conversion.
        # Units:
        #   n0 [1]; z,a,b,p,dS_m [m]; integral [m of water].
        # Domain/constraints:
        #   Positive n0 and p; inversion argument and depth are bounded by the
        #   current soil/bedrock domain. The integral helper returns the nonnegative
        #   volume between its two ordered depths.
        # Sign convention:
        #   Positive dS_m reduces positive-downward z; the ordered interval integral
        #   is a capacity magnitude, not a signed transfer by itself.
        depth_t = self._ensure_tensor(depth)
        p = self._ensure_tensor(self.porosity_decay_length).to(
            device=depth_t.device, dtype=depth_t.dtype
        )
        n0 = self._ensure_tensor(self.surface_porosity).to(
            device=depth_t.device, dtype=depth_t.dtype
        )
        return n0 * torch.exp(-depth_t / p)
    
    # References: RHESSys compute_delta_water.c.
    # Formula: deficit = S_max(z=0) - S_max(z=wt_depth).
    # Parameters: wt_depth (m), bedrock_depth (m), n0 (surface_porosity), p (porosity_decay_length).
    def _compute_storage_deficit(self, wt_depth: Tensor) -> Tensor:
        """
        Compute storage deficit (maximum possible storage - current storage).
        
        Parameters
        ----------  
        wt_depth : Tensor
            Current water table depth [m]
            
        Returns
        -------
        Tensor
            Storage deficit [mm]
        """
        # References:
        #   Beven and Kirkby (1979); Nathan and McMahon (1990).
        # Mathematical expression:
        #   D=S_profile(0)-S_profile(z); fsat=clip(S/(S+D),0,1);
        #   qbase=(S/tau)*clip(1-z/zmax,fmin,1)*exp(-TI/10).
        # Parameter meanings:
        #   D is profile capacity deficit; TI is the configured topographic-index
        #   factor; tau is the recession time and fmin the lower depth-response factor.
        # Units:
        #   S/D [mm]; qbase [mm s-1]; z/zmax [m]; tau [s]; fsat/TI/fmin [1].
        # Domain/constraints:
        #   This is the implemented empirical point recession closure, not a full
        #   TOPMODEL catchment solution. Zero capacity gives zero saturation fraction.
        # Sign convention:
        #   Baseflow is a nonnegative groundwater export; positive deficit is unfilled capacity.
        wt_depth_t = self._ensure_tensor(wt_depth)
        bedrock_depth = self._ensure_tensor(self.bedrock_depth).to(
            device=wt_depth_t.device, dtype=wt_depth_t.dtype
        )
        zero_depth = torch.zeros_like(wt_depth_t)
        max_storage = UnitSystem.convert(
            self._integrate_storage(zero_depth, bedrock_depth), "m", "mm"
        )
        current_max_storage = UnitSystem.convert(
            self._integrate_storage(wt_depth_t, bedrock_depth), "m", "mm"
        )

        deficit = max_storage - current_max_storage
        return torch.clamp(deficit, min=0.0)
    
    # References: Beven and Kirkby (1979).
    # Formula: f_sat = S / (S + deficit).
    # Parameters: S (storage, mm), deficit (mm).
    def _compute_saturation_fraction(self, storage: Tensor, deficit: Tensor) -> Tensor:
        """
        Compute saturation fraction for variable source area models.
        
        Parameters
        ----------
        storage : Tensor  
            Current groundwater storage [mm]
        deficit : Tensor
            Storage deficit [mm]
            
        Returns
        -------
        Tensor
            Saturation fraction [0-1]
        """
        # References:
        #   Beven and Kirkby (1979); Nathan and McMahon (1990).
        # Mathematical expression:
        #   D=S_profile(0)-S_profile(z); fsat=clip(S/(S+D),0,1);
        #   qbase=(S/tau)*clip(1-z/zmax,fmin,1)*exp(-TI/10).
        # Parameter meanings:
        #   D is profile capacity deficit; TI is the configured topographic-index
        #   factor; tau is the recession time and fmin the lower depth-response factor.
        # Units:
        #   S/D [mm]; qbase [mm s-1]; z/zmax [m]; tau [s]; fsat/TI/fmin [1].
        # Domain/constraints:
        #   This is the implemented empirical point recession closure, not a full
        #   TOPMODEL catchment solution. Zero capacity gives zero saturation fraction.
        # Sign convention:
        #   Baseflow is a nonnegative groundwater export; positive deficit is unfilled capacity.
        total_capacity = storage + deficit
        zero = torch.zeros_like(total_capacity)
        sat_fraction = torch.where(total_capacity > 0, storage / total_capacity, zero)

        return torch.clamp(sat_fraction, min=0.0, max=1.0)
    
    # References: Brutsaert and Nieber (1977); Tallaksen (1995).
    # Formula: Q = S / tau * depth_factor * exp(-TI / 10).
    # Parameters: S (storage, mm), tau (s), wt_depth (m), max_water_table_depth (m), topographic_index (dimensionless).
    def _compute_baseflow(self, storage: Tensor, wt_depth: Tensor) -> Tensor:
        """
        Compute baseflow contribution using exponential recession.
        
        Parameters
        ----------
        storage : Tensor
            Current groundwater storage [mm]
        wt_depth : Tensor
            Water table depth [m]
        Returns
        -------
        Tensor
            Baseflow rate [mm/s]
        """
        # References:
        #   Beven and Kirkby (1979); Nathan and McMahon (1990).
        # Mathematical expression:
        #   D=S_profile(0)-S_profile(z); fsat=clip(S/(S+D),0,1);
        #   qbase=(S/tau)*clip(1-z/zmax,fmin,1)*exp(-TI/10).
        # Parameter meanings:
        #   D is profile capacity deficit; TI is the configured topographic-index
        #   factor; tau is the recession time and fmin the lower depth-response factor.
        # Units:
        #   S/D [mm]; qbase [mm s-1]; z/zmax [m]; tau [s]; fsat/TI/fmin [1].
        # Domain/constraints:
        #   This is the implemented empirical point recession closure, not a full
        #   TOPMODEL catchment solution. Zero capacity gives zero saturation fraction.
        # Sign convention:
        #   Baseflow is a nonnegative groundwater export; positive deficit is unfilled capacity.
        tau = torch.clamp(
            self._ensure_tensor(self.drainage_time_constant).to(
                device=storage.device, dtype=storage.dtype
            ),
            min=1.0,
        )
        base_rate = storage / tau
        max_depth = self._ensure_tensor(self.max_water_table_depth).to(
            device=storage.device, dtype=storage.dtype
        )
        min_depth_factor = self._ensure_tensor(self.minimum_depth_factor).to(
            device=storage.device, dtype=storage.dtype
        )
        depth_factor = torch.clamp(1.0 - wt_depth / max_depth, min=min_depth_factor, max=1.0)
        topo_index = self._ensure_tensor(self.topographic_index).to(
            device=storage.device, dtype=storage.dtype
        )
        topo_factor = torch.exp(-topo_index / 10.0)
        baseflow_rate = base_rate * depth_factor * topo_factor

        return torch.clamp(baseflow_rate, min=0.0)
    
    # PhysicalField interface implementation
    def get_value_at(self, coordinates: Tensor) -> Tensor:
        """
        No coordinate query: GroundwaterDynamics is a point model.

        Notes
        -----
        Groundwater storage is prognostic state held by StateManager
        (SPATIAL_DIMS = None). Returning a parameter default would create a
        second state owner and conceal missing runtime state.
        """
        raise NotImplementedError(
            "GroundwaterDynamics is a point model (SPATIAL_DIMS = None): it has "
            "no coordinate query; read groundwater_storage from StateManager"
        )

    def compute_gradient(self, coordinates: Tensor) -> Tensor:
        """
        No spatial gradient: GroundwaterDynamics is a point model.

        A point model has no coordinate axis, so a gradient of any shape would
        be fabricated.
        """
        raise NotImplementedError(
            "GroundwaterDynamics is a point model (SPATIAL_DIMS = None): it has "
            "no spatial gradient"
        )

    def interpolate(self, source_coordinates: Tensor, target_coordinates: Tensor) -> Tensor:
        """
        No interpolation: GroundwaterDynamics is a point model.

        Notes
        -----
        Groundwater storage is prognostic state held by StateManager
        (SPATIAL_DIMS = None). Returning a parameter default would create a
        second state owner and conceal missing runtime state.
        """
        raise NotImplementedError(
            "GroundwaterDynamics is a point model (SPATIAL_DIMS = None): it has "
            "no interpolation; read groundwater_storage from StateManager"
        )
    
    def update_field(self, time_step: float, source_terms: Dict, boundary_conditions: Optional[Dict] = None) -> None:
        """
        Update groundwater field (implemented via forward method).
        
        Parameters
        ----------
        time_step : float
            Time step [s]
        source_terms : Dict
            Source terms (drainage, lateral flow, etc.)
        boundary_conditions : Optional[Dict]
            Boundary conditions (not used for point model)
        """
        # References: FunctorBase update adapter for PhysicalField.
        # Mathematical expression: field_update = forward(state, time, dt).
        # Parameter meanings: time_step [s], source_terms [dict], boundary_conditions [dict].
        super().update_field(time_step, source_terms, boundary_conditions)
        return None

    def _ensure_tensor(self, value: Union[float, Tensor]) -> Tensor:
        """
        Ensure input is a tensor with proper device and dtype.
        
        Parameters
        ----------
        value : Union[float, Tensor]
            Input value to convert
            
        Returns
        -------
        Tensor
            Tensor representation of input
        """
        if isinstance(value, torch.Tensor):
            return value
        return super()._ensure_tensor(value, dtype=torch.float32)
    
    def get_groundwater_metadata(self) -> Dict:
        """
        Get groundwater dynamics metadata for diagnostics.
        
        Returns
        -------
        Dict
            Groundwater metadata dictionary
        
        Notes
        -----
        Current storage and water table depth should be read from state,
        not from internal attributes.
        """
        return {
            "surface_porosity": self.surface_porosity,
            "porosity_decay_length": self.porosity_decay_length,
            "max_water_table_depth": self.max_water_table_depth,
            "bedrock_depth": self.bedrock_depth,
            "specific_yield": self.specific_yield,
            "enable_lateral_flow": self.enable_lateral_flow,
        }


# Export the main class
__all__ = ["GroundwaterDynamics"]

