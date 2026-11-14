"""
Surface Runoff Model for RELIC

This module implements surface runoff generation mechanisms
for the RELIC framework, including Hortonian overland flow
and saturation excess runoff based on soil moisture conditions.

References
----------
- Horton, R.E. (1933). The role of infiltration in the hydrologic cycle.
  Transactions, American Geophysical Union, 14(1), 446-460.
- Dunne, T. and Black, R.D. (1970). Partial area contributions to storm
  runoff in a small New England watershed. Water Resources Research, 6(5), 1296-1311.
- SCS (1972). National Engineering Handbook, Section 4: Hydrology.
  Soil Conservation Service, USDA, Washington, DC.
- USDA NRCS (2021). National Engineering Handbook, Part 650,
  Chapter 2: Estimating Runoff Volume and Peak Discharge.
- Chow, V.T., Maidment, D.R., and Mays, L.W. (1988). Applied Hydrology.
  McGraw-Hill, New York.

Mathematical Expressions
------------------------
SCS Curve Number method:
    Q = (P - Ia)^2 / (P - Ia + S)  for P > Ia
    Q = 0  for P <= Ia
    
Where:
    Q = surface runoff [mm]
    P = precipitation [mm] 
    Ia = initial abstraction = lambda_a*S [mm]
    S = potential maximum retention = 25400/CN - 254 [mm]
    CN = curve number [dimensionless, 30-100]
    lambda_a = initial abstraction ratio [dimensionless]

Hortonian runoff:
    Qh = max(0, P - Ic)
    
Where:
    Qh = Hortonian runoff [mm/h]
    P = precipitation rate [mm/h]
    Ic = infiltration capacity [mm/h]

Saturation excess runoff:
    Qs = P * As/A
    
Where:
    Qs = saturation excess runoff [mm/h]
    As = saturated area fraction [dimensionless]
    A = total area [dimensionless = 1]

Parameters
----------
curve_number : float
    SCS curve number [dimensionless, 30-100]
initial_abstraction_ratio : float
    Ratio of initial abstraction to S [dimensionless, typically 0.2]
infiltration_capacity : float
    Maximum infiltration rate [mm h-1]
saturated_area_threshold : float
    Soil moisture threshold for saturation excess [m3 m-3]
"""

from typing import Any, Dict, Mapping, Optional
import torch
from torch import Tensor

from relic.computation.operators.dec.forms import CochainSpaceSpec
from relic.computation.operators.dec.hodge import HodgeStarSpec
from relic.computation.operators.iphs import PortSpecIPHS
from relic.core.functor import (
    DependencyDeclaration,
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    TransportMechanismFunctor,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from relic.core.state import UpdateItem, UpdateMode
from relic.core.abstractions import FluxDict, StateDict
from relic.data.standards.unit_converter import unit_converter
from relic.utils.physical_constants import get_physical_constant
from ..interface_mapping import HydrosphereInterfaceMixin



class SurfaceRunoff(HydrosphereInterfaceMixin, TransportMechanismFunctor):
    """
    Surface Runoff Generation Model
    
    Implements multiple surface runoff mechanisms including:
    - SCS Curve Number method for event-based runoff
    - Hortonian overland flow (infiltration excess)
    - Saturation excess runoff based on soil moisture
    
    The model partitions precipitation into infiltration and
    surface runoff based on soil properties, antecedent moisture
    conditions, and precipitation intensity.
    
    Units Convention (CF-1.8 consistent):
        - surface_runoff_flux: kg m-2 s-1
        - precipitation: kg m-2 s-1
        - soil_moisture: m3 m-3
        - curve_number: dimensionless [30-100]
    
    Attributes
    ----------
    curve_number : float
        SCS curve number [dimensionless]
    initial_abstraction_ratio : float
        Initial abstraction ratio [dimensionless]
    infiltration_capacity : float
        Maximum infiltration rate [mm h-1]
    saturated_threshold : float
        Soil moisture threshold for saturation [m3 m-3]
    """

    TRANSPORT_TYPE = "advection"

    PHASE8_RUNTIME_REPLACEMENT_STATUS = "canonical_iphs_runtime"
    # Declared surface water input representations
    # (hydrosphere.runoff.surface_water_input_source, instance config override):
    #   "precipitation_flux": the total precipitation reaches the land surface
    #     (no snowpack Functor in the composition);
    #   "rain_flux_plus_snow_melt_flux": the snowpack partition is upstream; the rain
    #     reaching the ground plus the snowmelt form the input (E3SM ELM
    #     SnowHydrologyMod, column without snow layers:
    #     qflx_top_soil = qflx_rain_grnd + qflx_snomelt) and the snowpack Functor,
    #     the SCM owner of both fluxes, is a required dependency.
    # Exactly one representation is read; the state content never selects it.
    WATER_INPUT_SOURCES = ("precipitation_flux", "rain_flux_plus_snow_melt_flux")
    
    def __init__(self,
                 name: str = "surface_runoff",
                 config: Optional[Dict] = None) -> None:
        """
        Initialize Surface runoff model.

        Parameters
        ----------
        name : str, optional
            Unique identifier for this functor instance
        config : Dict, optional
            Configuration parameters for runoff mechanisms
        """
        # Create metadata
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Surface runoff generation using SCS, Hortonian, and saturation excess methods",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "m_dot = G_surf u_surf; u_surf = min(P, max(0, rho_w * Q(P, CN, theta, Ic)))"
            ),
            references=[
                "Horton, R.E. (1933). Transactions, American Geophysical Union, 14(1), 446-460",
                "Dunne, T. and Black, R.D. (1970). Water Resources Research, 6(5), 1296-1311",
                "SCS (1972). National Engineering Handbook, Section 4: Hydrology",
                "USDA NRCS (2021). National Engineering Handbook, Part 650, Chapter 2",
                "Chow, V.T. et al. (1988). Applied Hydrology"
            ],
            parameters={
                "hydrosphere.transport.surface_water_diffusivity": (
                    "Effective diffusivity for surface water transport processes [m2 s-1]"
                ),
                "hydrosphere.runoff.curve_number": (
                    "SCS Curve Number for runoff generation (forest land use, moderate "
                    "infiltration) [dimensionless]"
                ),
                "hydrosphere.runoff.initial_abstraction_ratio": (
                    "Initial abstraction ratio (Ia/S) in SCS method [dimensionless]"
                ),
                "hydrosphere.runoff.infiltration_capacity": (
                    "Maximum infiltration capacity for Hortonian runoff generation [mm h-1]"
                ),
                "hydrosphere.runoff.saturated_area_threshold": (
                    "Soil moisture threshold for saturation excess runoff generation [m3 m-3]"
                ),
                "hydrosphere.runoff.runoff_method": (
                    "Runoff generation method (scs, horton, saturation, or combined) "
                    "[dimensionless]"
                ),
                "hydrosphere.runoff.runoff_combination_method": (
                    "Combination rule (max or sum) of the combined runoff mechanisms "
                    "[dimensionless]"
                ),
                "hydrosphere.runoff.surface_water_input_source": (
                    "Declared surface water input of SurfaceRunoff [dimensionless]"
                ),
                "hydrosphere.runoff.antecedent_moisture_factor": (
                    "Antecedent moisture condition factor for SCS method (AMC I: 0.5-0.8, AMC II: "
                    "1.0, AMC III: 1.2-2.0) [dimensionless]"
                ),
                "hydrosphere.runoff.scs_storage_minimum": (
                    "Minimum SCS storage parameter S to avoid division issues [mm]"
                ),
                "hydrosphere.runoff.moisture_adjustment_factor": (
                    "Moisture adjustment coefficient for SCS retention scaling [dimensionless]"
                ),
                "hydrosphere.runoff.infiltration_moisture_reduction_factor": (
                    "Soil moisture reduction coefficient for infiltration capacity [dimensionless]"
                ),
                "hydrosphere.physical.scs_numerator_constant": (
                    "SCS Curve Number method numerator constant (1000 inches converted to mm) [mm]"
                ),
                "hydrosphere.physical.scs_offset_constant": (
                    "SCS Curve Number method offset constant (10 inches converted to mm) [mm]"
                ),
                "hydrosphere.numerical.minimum_flux_threshold": (
                    "Minimum flux threshold used in diagnostic ratio calculations [kg m-2 s-1]"
                ),
            },
            input_variables=[
                "precipitation_flux",
                "rain_flux",
                "snow_melt_flux",
                "soil_moisture",
            ],
            output_variables=[
                "surface_runoff_flux",
                "infiltration_flux",
                "diagnostics.surface_runoff.runoff_coefficient",
                "diagnostics.surface_runoff.precipitation_rate_mm_h",
                "diagnostics.surface_runoff.runoff_rate_mm_h",
                "diagnostics.surface_runoff.scs_retention",
                "diagnostics.surface_runoff.initial_abstraction",
                "diagnostics.surface_runoff.runoff_nonnegative_residual",
                "diagnostics.surface_runoff.precipitation_limit_residual",
                "diagnostics.surface_runoff.water_balance_residual",
            ],
            conservation_laws=["mass"],
            spatial_dimensions=0,  # Point-based model
            temporal_scale=TemporalScale.SECOND,
            coupling_interfaces=["soil_moisture", "precipitation_flux", "channel_routing"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
        )

        super().__init__(name=name, config=config or {}, metadata=metadata)
        self.optional_input_variables = {
            "precipitation_flux",
            "rain_flux",
            "snow_melt_flux",
        }

        # Add TransportMechanism functionality
        self.set_scalar_diffusivity(self.get_parameter_value(
            "surface_water_diffusivity",
            namespace="hydrosphere.transport", required=True))  # Surface water diffusivity [m2 s-1]
        self.peclet_number = None
        self.velocity_field = None
        # The surface-water diffusivity is the mobility of the static Hodge contract; the
        # runoff partition uses no tortuosity or porosity, so none is read.

        # Get parameters from unified parameter system
        self.curve_number = self.get_parameter_value(
            "curve_number",
            namespace="hydrosphere.runoff", required=True)
        self.initial_abstraction_ratio = self.get_parameter_value(
            "initial_abstraction_ratio",
            namespace="hydrosphere.runoff", required=True)
        self.infiltration_capacity = self.get_parameter_value(
            "infiltration_capacity",
            namespace="hydrosphere.runoff", required=True)
        self.saturated_threshold = self.get_parameter_value(
            "saturated_area_threshold",
            namespace="hydrosphere.runoff", required=True)
        self.runoff_method = self.get_parameter_value(
            "runoff_method",
            namespace="hydrosphere.runoff", required=True)
        self.runoff_combination_method = self.get_parameter_value(
            "runoff_combination_method",
            namespace="hydrosphere.runoff", required=True)
        self.antecedent_moisture_factor = self.get_parameter_value(
            "antecedent_moisture_factor",
            namespace="hydrosphere.runoff", required=True)
        self.scs_storage_minimum = self.get_parameter_value(
            "scs_storage_minimum",
            namespace="hydrosphere.runoff", required=True)
        self.moisture_adjustment_factor = self.get_parameter_value(
            "moisture_adjustment_factor",
            namespace="hydrosphere.runoff", required=True)
        self.infiltration_moisture_reduction_factor = self.get_parameter_value(  
            "infiltration_moisture_reduction_factor",
            namespace="hydrosphere.runoff", required=True)
        self.scs_numerator_constant = self.get_parameter_value(
            "scs_numerator_constant",
            namespace="hydrosphere.physical", required=True)
        self.scs_offset_constant = self.get_parameter_value(
            "scs_offset_constant",
            namespace="hydrosphere.physical", required=True)
        self.water_density = get_physical_constant("water_density")
        self.time_conversion = get_physical_constant("seconds_per_hour")
        self.unit_converter = unit_converter
        self.minimum_flux_threshold = self.get_parameter_value(
            "minimum_flux_threshold",
            namespace="hydrosphere.numerical", required=True)
        # Declared surface water input (YAML SSOT; an instance config key of the
        # same name overrides it); validated in _validate_parameters.
        self.surface_water_input_source = self.get_parameter_value(
            "surface_water_input_source",
            namespace="hydrosphere.runoff",
            source_order=("config", "manager"),
            required=True)
        
        # Validate parameters
        self._validate_parameters()
        if self.surface_water_input_source == "rain_flux_plus_snow_melt_flux":
            # The snowpack partition must run first in the same step, so its SCM
            # owner becomes a required dependency of this instance.
            self.add_dependency(self._snowpack_input_owner())
        
    def _to_scalar(self, value: Any) -> float:
        """Convert tensor-like values to float for validation checks."""
        try:
            tensor_value = (
                value.detach() if isinstance(value, Tensor) else torch.as_tensor(value)
            ).reshape(-1)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("Expected a numeric scalar configuration value.") from exc
        if tensor_value.numel() != 1:
            raise ValueError(
                "Expected a scalar configuration value; spatial averaging is not allowed."
            )
        return float(tensor_value.item())

    def _validate_parameters(self) -> None:
        """Validate parameter ranges and consistency."""
        curve_number = self._to_scalar(self.curve_number)
        initial_ratio = self._to_scalar(self.initial_abstraction_ratio)
        infiltration_capacity = self._to_scalar(self.infiltration_capacity)
        saturated_threshold = self._to_scalar(self.saturated_threshold)

        if not (30.0 <= curve_number <= 100.0):
            raise ValueError(f"Curve number must be in range [30, 100], got {curve_number}")
        
        if not (0.0 <= initial_ratio <= 0.5):
            raise ValueError(
                f"Initial abstraction ratio must be in range [0, 0.5], got {initial_ratio}"
            )
        
        if infiltration_capacity <= 0:
            raise ValueError("Infiltration capacity must be positive")
        
        if not (0.0 <= saturated_threshold <= 1.0):
            raise ValueError("Saturated area threshold must be in range [0, 1]")
        
        valid_methods = ["scs", "horton", "saturation", "combined"]
        if self.runoff_method not in valid_methods:
            raise ValueError(f"Runoff method must be one of {valid_methods}")

        valid_combinations = ["max", "sum"]
        if self.runoff_combination_method not in valid_combinations:
            raise ValueError(
                f"Runoff combination method must be one of {valid_combinations}"
            )

        if self.surface_water_input_source not in self.WATER_INPUT_SOURCES:
            raise ValueError(
                "surface_water_input_source must be one of "
                f"{list(self.WATER_INPUT_SOURCES)}, got {self.surface_water_input_source!r}"
            )

    @staticmethod
    def _snowpack_input_owner() -> str:
        """Return the one SCM owner of rain_flux and snow_melt_flux (the snowpack)."""
        from relic.contracts import load_contract_registry

        registry = load_contract_registry()
        owners = {registry.get_owner("rain_flux"), registry.get_owner("snow_melt_flux")}
        if len(owners) != 1 or None in owners:
            raise ValueError(
                "rain_flux and snow_melt_flux must share one SCM owner to be read as the "
                f"snowpack surface water input, got {sorted(map(str, owners))}"
            )
        return owners.pop()
    
    def get_dependencies(self) -> DependencyDeclaration:
        """
        Return graph dependencies for this functor.
        
        Returns
        -------
        DependencyDeclaration
            The partition reads the committed soil-moisture state and does not
            create a reverse execution edge to its selected writer. The
            downstream soil process owns the SurfaceRunoff dependency.
        """
        return self._merge_declared_dependencies({"required": [], "optional": []})

    def _structural_n_dofs(
        self,
        n_dofs: Optional[int] = None,
        complex_bundle: Any = None,
    ) -> int:
        """Resolve the structural audit size without changing runtime tensor shape."""

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
        """Declare canonical cochain spaces for surface runoff.

        References:
          - Horton (1933), Dunne and Black (1970), and SCS (1972).
          - Hirani (2003), Discrete Exterior Calculus.
        Mathematical expression:
          m_dot = G_surf u_surf, u_surf = min(P, max(0, rho_w * Q)).
        Parameter meanings:
          P is precipitation flux [kg m-2 s-1]; Q is runoff depth rate
          [mm h-1]; u_surf is surface runoff mass flux [kg m-2 s-1].
        """

        n_dofs = self._structural_n_dofs()
        return (
            CochainSpaceSpec(
                complex_id="hydrosphere_surface_runoff",
                degree=0,
                primality="primal",
                n_dofs=n_dofs,
                units="kg m-2 s-1",
                extensivity="intensive",
                physical_domain="hydrosphere",
                semantic_name="precipitation_flux",
                orientation_convention="downward_positive",
                boundary_role="boundary",
            ),
            CochainSpaceSpec(
                complex_id="hydrosphere_surface_runoff",
                degree=0,
                primality="primal",
                n_dofs=n_dofs,
                units="m3 m-3",
                extensivity="intensive",
                physical_domain="hydrosphere",
                semantic_name="soil_moisture",
                orientation_convention="unsigned",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="hydrosphere_surface_runoff",
                degree=2,
                primality="dual",
                n_dofs=n_dofs,
                units="kg m-2 s-1",
                extensivity="integral",
                physical_domain="hydrosphere",
                semantic_name="surface_runoff_flux",
                orientation_convention="outward_positive",
                boundary_role="port",
            ),
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Declare ParameterManager keys used by runtime and structural builders."""

        return (
            "hydrosphere.transport.surface_water_diffusivity",
            "hydrosphere.runoff.curve_number",
            "hydrosphere.runoff.initial_abstraction_ratio",
            "hydrosphere.runoff.infiltration_capacity",
            "hydrosphere.runoff.saturated_area_threshold",
            "hydrosphere.runoff.runoff_method",
            "hydrosphere.runoff.runoff_combination_method",
            "hydrosphere.runoff.surface_water_input_source",
            "hydrosphere.runoff.antecedent_moisture_factor",
            "hydrosphere.runoff.scs_storage_minimum",
            "hydrosphere.runoff.moisture_adjustment_factor",
            "hydrosphere.runoff.infiltration_moisture_reduction_factor",
            "hydrosphere.physical.scs_numerator_constant",
            "hydrosphere.physical.scs_offset_constant",
            "hydrosphere.numerical.minimum_flux_threshold",
        )

    def build_hodge_contract(
        self,
        n_dofs: Optional[int] = None,
        *,
        strict: bool = True,
    ) -> HodgeStarSpec:
        """Build the surface runoff routing Hodge contract.

        References:
          - Chow et al. (1988), overland flow routing.
          - van der Schaft and Maschke (1995), dissipative port variables.
        Mathematical expression:
          F_surf = star_surf X_surf;
          star_surf = diag(rho_w * D_surf), with X_surf [m-1].
        Parameter meanings:
          D_surf is hydrosphere.transport.surface_water_diffusivity [m2 s-1];
          rho_w is water density [kg m-3]; X_surf is a routing gradient [m-1];
          F_surf is the resulting mass flux [kg m-2 s-1].
        """

        n_dofs = self._structural_n_dofs(n_dofs=n_dofs)
        mobility_value = self._to_scalar(self.water_density) * self._to_scalar(self.diffusivity)
        values = torch.full(
            (n_dofs,),
            mobility_value,
            dtype=torch.float32,
            device=getattr(self, "device", torch.device("cpu")),
        )
        if bool(torch.any(values <= 0.0)):
            raise ValueError("Surface runoff mobility must be positive.")

        domain = CochainSpaceSpec(
            complex_id="hydrosphere_surface_runoff",
            degree=1,
            primality="primal",
            n_dofs=n_dofs,
            units="m-1",
            extensivity="intensive",
            physical_domain="hydrosphere",
            semantic_name="surface_routing_gradient",
            orientation_convention="outward_positive",
        )
        codomain = CochainSpaceSpec(
            complex_id="hydrosphere_surface_runoff",
            degree=2,
            primality="dual",
            n_dofs=n_dofs,
            units="kg m-2 s-1",
            extensivity="integral",
            physical_domain="hydrosphere",
            semantic_name="surface_runoff_flux",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        hodge = HodgeStarSpec(
            name=f"{self.name}_surface_runoff_hodge",
            domain=domain,
            codomain=codomain,
            matrix=torch.diag(values),
            parameter_keys=("hydrosphere.transport.surface_water_diffusivity",),
            metric_keys=("hydrosphere.runoff.scs_storage_minimum",),
            material_law="surface_runoff_partitioning",
        )
        if strict:
            hodge.validate(passive=True)
        return hodge

    def build_hodge_terms(
        self,
        complex_bundle: Any = None,
        resolved_parameters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[HodgeStarSpec, ...]:
        """Build surface runoff Hodge terms for IPHS assembly."""

        return (
            self.build_hodge_contract(
                n_dofs=self._structural_n_dofs(complex_bundle=complex_bundle),
                strict=True,
            ),
        )

    def energy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return optional energy terms for structural IPHS audits."""

        return {}

    def entropy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return optional entropy terms for structural IPHS audits."""

        return {}

    def reversible_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> None:
        """Return no reversible interconnection for event runoff partitioning."""

        return None

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the positive surface runoff mobility contribution.

        References:
          - Horton (1933), SCS (1972), and van der Schaft and Maschke (1995).
        Mathematical expression:
          F_h = M_h X_h; M_h = diag(rho_w * D_surf);
          sigma = X_h.T M_h X_h >= 0.
        Parameter meanings:
          D_surf is the surface water diffusivity [m2 s-1]; X_h is the
          surface routing gradient [m-1]; F_h is mass flux [kg m-2 s-1].
        """

        hodge = self.build_hodge_contract(
            n_dofs=self._structural_n_dofs(complex_bundle=complex_bundle),
            strict=True,
        )
        matrix = hodge.matrix
        return {
            "state_vars": tuple(
                f"surface_water_partition[{idx}]" for idx in range(int(matrix.shape[0]))
            ),
            "J": torch.zeros_like(matrix),
            "M": matrix,
            "ports": self.ports(),
        }

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Declare overland-flow and soil-infiltration matter ports.

        References:
          - Chow et al. (1988), overland flow boundary routing.
          - van der Schaft and Maschke (1995), port variables.
        Mathematical expression:
          P_port = e_h * f_h for each runoff or infiltration mass flux.
        Parameter meanings:
          e_h is hydraulic head [m]; f_h is runoff or infiltration mass flux
          [kg m-2 s-1].
        """

        effort_space = CochainSpaceSpec(
            complex_id="hydrosphere_surface_runoff",
            degree=0,
            primality="primal",
            n_dofs=1,
            units="m",
            extensivity="intensive",
            physical_domain="hydrosphere",
            semantic_name="surface_routing_head",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        flow_space = CochainSpaceSpec(
            complex_id="hydrosphere_surface_runoff",
            degree=2,
            primality="dual",
            n_dofs=1,
            units="kg m-2 s-1",
            extensivity="integral",
            physical_domain="hydrosphere",
            semantic_name="surface_runoff_mass_flux",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        infiltration_effort_space = CochainSpaceSpec(
            complex_id="surface_soil_water_interface",
            degree=0,
            primality="primal",
            n_dofs=1,
            units="m",
            extensivity="intensive",
            physical_domain="surface_soil_interface",
            semantic_name="surface_hydraulic_head",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        infiltration_flow_space = CochainSpaceSpec(
            complex_id="surface_soil_water_interface",
            degree=2,
            primality="dual",
            n_dofs=1,
            units="kg m-2 s-1",
            extensivity="integral",
            physical_domain="surface_soil_interface",
            semantic_name="infiltration_mass_flux",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        return (
            PortSpecIPHS(
                name="surface_runoff_boundary_flux",
                effort_var="surface_routing_head",
                flow_var="surface_runoff_flux",
                effort_space=effort_space,
                flow_space=flow_space,
                power_units="kg m-1 s-1",
                positive_power_convention="out_of_system",
                port_kind="matter",
            ),
            PortSpecIPHS(
                name="soil_infiltration_boundary",
                effort_var="surface_hydraulic_head",
                flow_var="infiltration_flux",
                effort_space=infiltration_effort_space,
                flow_space=infiltration_flow_space,
                power_units="kg m-1 s-1",
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
    
    # References: SCS (1972); Horton (1933); Dunne and Black (1970).
    # Mathematical expression: P = R_surf + I with R_surf in [0, P].
    # Parameter meanings: P, R_surf, and I are water mass fluxes
    # [kg m-2 s-1]; theta is soil moisture [m3 m-3]; dt is time [s].
    def forward(
        self,
        state: StateDict,
        time: float,
        dt: float,
        **kwargs,
    ) -> FluxDict:
        """
        Compute surface runoff generation.
        
        Parameters
        ----------
        state : StateDict
            Current system state containing precipitation and soil moisture
        time : float
            Current simulation time [s]
        dt : float  
            Time step [s]
        **kwargs
            Additional context variables
            
        Returns
        -------
        FluxDict
            Surface runoff flux and diagnostic variables
            
        Raises
        ------
        ValueError
            If required state variables are missing or invalid
        """
        # Extract required state variables
        try:
            soil_moisture_profile = self._ensure_tensor(
                state["soil_moisture"]
            )  # m3 m-3
        except KeyError as e:
            raise ValueError(f"Required state variable missing for Surface runoff: {e}")

        # References: Dunne and Black (1970); SCS (1972); Celia et al. (1990).
        # Mathematical expression: theta_surface = theta[0] for a vertical profile.
        # Parameter meanings: theta_surface is uppermost-layer water content
        # [m3 m-3]; index 0 is the top of the one-dimensional soil column.
        if soil_moisture_profile.ndim == 0:
            soil_moisture = soil_moisture_profile
        elif soil_moisture_profile.ndim == 1 and soil_moisture_profile.numel() > 0:
            soil_moisture = soil_moisture_profile[0]
        else:
            raise ValueError(
                "SurfaceRunoff soil_moisture must be a scalar or one-dimensional "
                "vertical profile."
            )

        # References: E3SM ELM SnowHydrologyMod, column without snow layers:
        #   qflx_top_soil = qflx_rain_grnd + qflx_snomelt; bibliographic references:
        #   Oleson et al. (2013), Technical Description of version 4.5 of the Community
        #   Land Model (CLM), NCAR/TN-503+STR; Neitsch et al. (2011), Soil and Water
        #   Assessment Tool Theoretical Documentation Version 2009; Anderson (1976),
        #   NOAA NWS 19.
        # Mathematical expression: P_in = P_total for the source
        #   "precipitation_flux"; P_in = P_rain + M for the source
        #   "rain_flux_plus_snow_melt_flux".
        # Parameter meanings: P_total is precipitation_flux; P_rain the rain reaching
        #   the ground (rain_flux) and M the snowmelt release (snow_melt_flux) of the
        #   snowpack partition; the source is surface_water_input_source.
        # Units: P_in, P_total, P_rain, M [kg m-2 s-1].
        # Domain/constraints: only the declared representation is read; the other one
        #   is ignored even when present, so precipitation already partitioned by the
        #   snowpack (snowfall held in the pack, rain, melt) is never counted twice; a
        #   missing declared variable raises.
        # Sign convention: non-negative downward mass fluxes into the land surface;
        #   P_in is partitioned into the runoff and soil-infiltration boundary ports.
        if self.surface_water_input_source == "precipitation_flux":
            if "precipitation_flux" not in state:
                raise ValueError(
                    "SurfaceRunoff with surface_water_input_source 'precipitation_flux' "
                    "requires precipitation_flux."
                )
            precipitation = self._ensure_tensor(state["precipitation_flux"])
        else:
            missing = [name for name in ("rain_flux", "snow_melt_flux") if name not in state]
            if missing:
                raise ValueError(
                    "SurfaceRunoff with surface_water_input_source "
                    f"'rain_flux_plus_snow_melt_flux' requires {missing}."
                )
            precipitation = self._ensure_tensor(state["rain_flux"]) + self._ensure_tensor(
                state["snow_melt_flux"]
            )
        
        # Validate the physical input domain without clamping or hidden defaults.
        if not bool(torch.all(torch.isfinite(precipitation))):
            raise ValueError("SurfaceRunoff precipitation forcing must be finite.")
        if bool(torch.any(precipitation < 0.0)):
            raise ValueError("SurfaceRunoff precipitation forcing must be non-negative.")
        if not bool(torch.all(torch.isfinite(soil_moisture_profile))):
            raise ValueError("SurfaceRunoff soil_moisture must be finite.")
        if torch.any(soil_moisture_profile < 0) or torch.any(
            soil_moisture_profile > 1
        ):
            raise ValueError("Soil moisture content must be in range [0, 1]")
        dt_tensor = torch.as_tensor(dt)
        if not bool(torch.all(torch.isfinite(dt_tensor))):
            raise ValueError("Time step must be finite")
        if bool(torch.any(dt_tensor <= 0)):
            raise ValueError("Time step must be positive")

        # Curve number is a ParameterManager-owned model parameter. State aliases
        # are intentionally ignored so runtime state cannot mutate the closure.
        curve_number_base = self._ensure_tensor(self.curve_number).to(
            device=soil_moisture.device, dtype=soil_moisture.dtype
        )
        curve_number = torch.ones_like(soil_moisture) * curve_number_base

        # References: CF-1.8 unit conversion for water depth rates.
        # Mathematical expression: rate_mm_h = (F / rho_w) * 1000 * seconds_per_hour.
        # Parameter meanings: F precipitation flux [kg m-2 s-1], rho_w water density [kg m-3],
        # seconds_per_hour [s].
        time_conversion = self._ensure_tensor(self.time_conversion).to(
            device=soil_moisture.device, dtype=soil_moisture.dtype
        )
        # The depth-rate step is UnitConverter's single CF liquid-water-equivalent rule
        # (rho_w = water.density_liquid), selected by the declared water variable name.
        converter = self.unit_converter
        precip_mm_s = converter.convert(precipitation, "precipitation_flux", "kg m-2 s-1", "mm s-1")
        precip_mm_h = precip_mm_s * time_conversion  # mm h-1

        # References: SCS (1972) with the configured antecedent-moisture modifier.
        # Mathematical expression: S0=max(25400/CN-254,S_min);
        # S=S0*f_amc*(1-alpha_theta*theta); Ia=lambda_a*S.
        # Parameter meanings: S0 and S [mm], CN [1], S_min [mm], f_amc [1],
        # alpha_theta [1], theta [m3 m-3], lambda_a [1], Ia [mm].
        scs_numerator = self._ensure_tensor(self.scs_numerator_constant).to(
            device=soil_moisture.device, dtype=soil_moisture.dtype
        )
        scs_offset = self._ensure_tensor(self.scs_offset_constant).to(
            device=soil_moisture.device, dtype=soil_moisture.dtype
        )
        scs_storage_minimum = self._ensure_tensor(self.scs_storage_minimum).to(
            device=soil_moisture.device, dtype=soil_moisture.dtype
        )
        retention_base = torch.clamp(
            scs_numerator / curve_number - scs_offset, min=scs_storage_minimum
        )
        moisture_adjustment_factor = self._ensure_tensor(
            self.moisture_adjustment_factor
        ).to(device=soil_moisture.device, dtype=soil_moisture.dtype)
        antecedent_factor = self._ensure_tensor(
            self.antecedent_moisture_factor
        ).to(device=soil_moisture.device, dtype=soil_moisture.dtype)
        moisture_factor = 1.0 - moisture_adjustment_factor * torch.clamp(
            soil_moisture, min=0.0, max=1.0
        )
        effective_retention = retention_base * antecedent_factor * moisture_factor
        initial_abstraction_ratio = self._ensure_tensor(self.initial_abstraction_ratio).to(
            device=soil_moisture.device, dtype=soil_moisture.dtype
        )
        initial_abstraction = torch.clamp(
            initial_abstraction_ratio * effective_retention,
            min=0.0,
        )
        
        # Compute runoff using selected method(s)
        if self.runoff_method == "scs":
            runoff_flux = self._compute_scs_runoff(
                precip_mm_h,
                dt,
                effective_retention=effective_retention,
                initial_abstraction=initial_abstraction,
            )
        elif self.runoff_method == "horton":
            runoff_flux = self._compute_hortonian_runoff(precip_mm_h, soil_moisture, dt)
        elif self.runoff_method == "saturation":
            runoff_flux = self._compute_saturation_runoff(precip_mm_h, soil_moisture, dt)
        elif self.runoff_method == "combined":
            runoff_flux = self._compute_combined_runoff(
                precip_mm_h,
                soil_moisture,
                dt,
                effective_retention=effective_retention,
                initial_abstraction=initial_abstraction,
            )
        else:
            raise ValueError(f"Unknown runoff method: {self.runoff_method}")
        
        # References: CF-1.8 unit conversion for water depth rates.
        # Mathematical expression: F = (rate_mm_h / seconds_per_hour) / 1000 * rho_w.
        # Parameter meanings: rate_mm_h runoff rate [mm h-1], rho_w water density [kg m-3],
        # seconds_per_hour [s].
        runoff_mm_s = runoff_flux / time_conversion
        surface_runoff_flux_raw = converter.convert(
            runoff_mm_s, "surface_runoff_flux", "mm s-1", "kg m-2 s-1"
        )
        
        # References: Horton (1933); SCS (1972); hydrologic mass balance.
        # Mathematical expression: F_runoff = min(P, max(0, F_raw)).
        # Parameter meanings: F_raw [kg m-2 s-1], P precipitation [kg m-2 s-1].
        runoff_nonnegative_residual = torch.clamp(-surface_runoff_flux_raw, min=0.0)
        surface_runoff_flux = torch.clamp(surface_runoff_flux_raw, min=0.0)
        precipitation_limit_residual = torch.clamp(
            surface_runoff_flux - precipitation,
            min=0.0,
        )
        surface_runoff_flux = torch.minimum(surface_runoff_flux, precipitation)
        
        # Compute diagnostic variables
        minimum_flux_threshold = self._ensure_tensor(self.minimum_flux_threshold).to(
            device=precipitation.device, dtype=precipitation.dtype
        )
        runoff_coefficient = torch.where(
            precipitation > minimum_flux_threshold,
            surface_runoff_flux / precipitation,
            torch.zeros_like(precipitation)
        )
        
        infiltration_flux = precipitation - surface_runoff_flux
        water_balance_residual = precipitation - surface_runoff_flux - infiltration_flux
        
        return {
            "surface_runoff_flux": UpdateItem(
                value=surface_runoff_flux,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="kg m-2 s-1",
                source_functor=self.name,
            ),
            "infiltration_flux": UpdateItem(
                value=infiltration_flux,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="kg m-2 s-1",
                source_functor=self.name,
            ),
            "diagnostics.surface_runoff.runoff_coefficient": runoff_coefficient,
            "diagnostics.surface_runoff.precipitation_rate_mm_h": precip_mm_h,
            "diagnostics.surface_runoff.runoff_rate_mm_h": runoff_flux,
            "diagnostics.surface_runoff.scs_retention": effective_retention,
            "diagnostics.surface_runoff.initial_abstraction": initial_abstraction,
            "diagnostics.surface_runoff.runoff_nonnegative_residual": (
                runoff_nonnegative_residual
            ),
            "diagnostics.surface_runoff.precipitation_limit_residual": (
                precipitation_limit_residual
            ),
            "diagnostics.surface_runoff.water_balance_residual": water_balance_residual,
        }

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Evaluate the explicit open-boundary water partition.

        References:
          - Horton (1933), infiltration-excess surface runoff.
          - Dunne and Black (1970), saturation-excess surface runoff.
          - SCS (1972), curve-number runoff partition.
        Mathematical expression:
          R_w = P - F_runoff - F_infiltration.
        Parameter meanings:
          P is precipitation, F_runoff is surface runoff, and
          F_infiltration is infiltration; all terms use kg m-2 s-1.
        Open-system route:
          Surface runoff exits through the runoff boundary port, while
          infiltration enters the soil-water graph through its typed port.
        """

        del input_state
        diagnostic_key = "diagnostics.surface_runoff.water_balance_residual"
        if diagnostic_key not in output_updates:
            raise KeyError(f"Missing conservation diagnostic '{diagnostic_key}'")
        residual = output_updates[diagnostic_key]
        if isinstance(residual, UpdateItem):
            residual = residual.value
        if not isinstance(residual, Tensor):
            raise TypeError("Surface-runoff mass residual must be a torch.Tensor")
        if residual.numel() == 0:
            raise ValueError("Surface-runoff mass residual must be non-empty")
        if not bool(torch.all(torch.isfinite(residual))):
            raise ValueError("Surface-runoff mass residual must be finite")
        magnitude = torch.max(torch.abs(residual))
        return {"mass": float(magnitude.detach().item())}
    
    def _compute_scs_runoff(self, 
                           precip_mm_h: Tensor, 
                           dt: float,
                           effective_retention: Tensor,
                           initial_abstraction: Tensor) -> Tensor:
        """
        Compute runoff using SCS Curve Number method.
        
        Parameters
        ----------
        precip_mm_h : Tensor
            Precipitation rate [mm h-1]
        dt : float
            Time step [s]
        effective_retention : Tensor
            Moisture-adjusted SCS retention [mm]
        initial_abstraction : Tensor
            Initial abstraction used by the SCS equation [mm]
            
        Returns
        -------
        Tensor
            Runoff rate [mm h-1]
        """
        # References:
        #   SCS (1972); USDA NRCS (1986), TR-55.
        # Mathematical expression:
        #   Pevent=P_rate*dt/seconds_per_hour; E=max(Pevent-Ia,0);
        #   Qevent=E^2/(E+S) when E>0, otherwise 0; Qrate=Qevent*seconds_per_hour/dt.
        # Parameter meanings:
        #   S is effective retention and Ia is initial abstraction supplied by forward;
        #   CN/antecedent/moisture factors have already been resolved there.
        # Units:
        #   Pevent/Ia/S/Qevent [mm]; P_rate/Qrate [mm h-1]; dt [s].
        # Domain/constraints:
        #   Positive dt and configured positive retention; apply the event formula to
        #   integrated depth and convert back exactly once.
        # Sign convention:
        #   Nonnegative Qrate is an outward surface runoff magnitude.
        time_conversion = self._ensure_tensor(self.time_conversion).to(
            device=precip_mm_h.device, dtype=precip_mm_h.dtype
        )
        precip_depth = precip_mm_h * dt / time_conversion  # mm
        
        # SCS runoff equation
        excess_precip = torch.clamp(
            precip_depth - initial_abstraction,
            min=0.0,
        )
        
        runoff_depth = torch.where(
            excess_precip > 0.0,
            torch.pow(excess_precip, 2)
            / (excess_precip + effective_retention),
            torch.zeros_like(excess_precip)
        )  # mm
        
        # Convert back to rate [mm h-1]
        runoff_rate = runoff_depth * time_conversion / dt
        
        return runoff_rate
    
    def _compute_hortonian_runoff(self, 
                                 precip_mm_h: Tensor, 
                                 soil_moisture: Tensor, 
                                 dt: float) -> Tensor:
        """
        Compute Hortonian (infiltration excess) runoff.
        
        Parameters
        ----------
        precip_mm_h : Tensor
            Precipitation rate [mm h-1]
        soil_moisture : Tensor
            Soil moisture content [m3 m-3]
        dt : float
            Time step [s]
            
        Returns
        -------
        Tensor
            Runoff rate [mm h-1]
        """
        # References:
        #   Horton (1933); Dunne and Black (1970); SCS (1972).
        # Mathematical expression:
        #   Ic_eff=Ic*(1-c_theta*clip(theta,0,1)); Qh=max(P-Ic_eff,0);
        #   Asat=clip((theta-theta_sat)/(1-theta_sat),0,1); Qs=P*Asat;
        #   Qcombined=min(P,max(Qscs,Qh,Qs)) or min(P,Qscs+Qh+Qs).
        # Parameter meanings:
        #   Ic is infiltration capacity; c_theta is its empirical moisture response;
        #   theta_sat is the area-fraction threshold; combination method is explicit.
        # Units:
        #   P/Ic/Qh/Qs/Qcombined [mm h-1]; theta, c_theta and Asat [1].
        # Domain/constraints:
        #   These are the configured point partition laws, not distributed routing;
        #   select exactly the declared combination method and preserve the input cap.
        # Sign convention:
        #   Positive rates are outward runoff; remaining forcing infiltrates.
        infiltration_reduction_factor = self._ensure_tensor(
            self.infiltration_moisture_reduction_factor
        ).to(device=soil_moisture.device, dtype=soil_moisture.dtype)
        moisture_reduction = 1.0 - infiltration_reduction_factor * torch.clamp(
            soil_moisture, min=0.0, max=1.0
        )
        infiltration_capacity = self._ensure_tensor(self.infiltration_capacity).to(
            device=soil_moisture.device, dtype=soil_moisture.dtype
        )
        infiltration_capacity = infiltration_capacity * moisture_reduction
        
        # Hortonian runoff occurs when precipitation exceeds infiltration capacity
        runoff_rate = torch.clamp(precip_mm_h - infiltration_capacity, min=0.0)  # Non-negative runoff
        
        return runoff_rate
    
    def _compute_saturation_runoff(self, 
                                  precip_mm_h: Tensor, 
                                  soil_moisture: Tensor, 
                                  dt: float) -> Tensor:
        """
        Compute saturation excess runoff.
        
        Parameters
        ----------
        precip_mm_h : Tensor
            Precipitation rate [mm h-1]
        soil_moisture : Tensor
            Soil moisture content [m3 m-3]
        dt : float
            Time step [s]
            
        Returns
        -------
        Tensor
            Runoff rate [mm h-1]
        """
        # References:
        #   Horton (1933); Dunne and Black (1970); SCS (1972).
        # Mathematical expression:
        #   Ic_eff=Ic*(1-c_theta*clip(theta,0,1)); Qh=max(P-Ic_eff,0);
        #   Asat=clip((theta-theta_sat)/(1-theta_sat),0,1); Qs=P*Asat;
        #   Qcombined=min(P,max(Qscs,Qh,Qs)) or min(P,Qscs+Qh+Qs).
        # Parameter meanings:
        #   Ic is infiltration capacity; c_theta is its empirical moisture response;
        #   theta_sat is the area-fraction threshold; combination method is explicit.
        # Units:
        #   P/Ic/Qh/Qs/Qcombined [mm h-1]; theta, c_theta and Asat [1].
        # Domain/constraints:
        #   These are the configured point partition laws, not distributed routing;
        #   select exactly the declared combination method and preserve the input cap.
        # Sign convention:
        #   Positive rates are outward runoff; remaining forcing infiltrates.
        saturated_threshold = self._ensure_tensor(self.saturated_threshold).to(
            device=soil_moisture.device, dtype=soil_moisture.dtype
        )
        saturated_area_fraction = torch.clamp(
            (soil_moisture - saturated_threshold) / (1.0 - saturated_threshold),
            min=0.0, max=1.0
        )
        
        # Saturation excess runoff occurs from saturated areas
        runoff_rate = precip_mm_h * saturated_area_fraction
        
        return runoff_rate
    
    def _compute_combined_runoff(self, 
                                precip_mm_h: Tensor, 
                                soil_moisture: Tensor, 
                                dt: float,
                                effective_retention: Tensor,
                                initial_abstraction: Tensor) -> Tensor:
        """
        Compute runoff using combined mechanisms.

        Parameters
        ----------
        precip_mm_h : Tensor
            Precipitation rate [mm h-1]
        soil_moisture : Tensor
            Soil moisture content [m3 m-3]
        dt : float
            Time step [s]
        effective_retention : Tensor
            Moisture-adjusted SCS retention parameter [mm]
        initial_abstraction : Tensor
            Initial abstraction used by the SCS equation [mm]
            
        Returns
        -------
        Tensor
            Runoff rate [mm h-1]

        Notes
        -----
        When runoff_combination_method == "max", the dominant mechanism is used
        and mechanisms are treated as mutually exclusive rather than additive.

        References
        ----------
        SCS (1972). National Engineering Handbook, Section 4: Hydrology.
        Horton, R.E. (1933). Trans. AGU 14(1), 446-460.
        Dunne, T. and Black, R.D. (1970). Water Resources Research, 6(5), 1296-1311.

        Mathematical expression
        -----------------------
        Q_combined = max(Q_scs, Q_horton, Q_saturation) or
        Q_combined = Q_scs + Q_horton + Q_saturation

        Parameter meanings
        ------------------
        precip_mm_h: precipitation rate [mm h-1]
        soil_moisture: volumetric soil water content [m3 m-3]
        dt: time step [s]
        effective_retention: moisture-adjusted SCS retention [mm]
        initial_abstraction: SCS initial abstraction [mm]
        """
        # References:
        #   Horton (1933); Dunne and Black (1970); SCS (1972).
        # Mathematical expression:
        #   Ic_eff=Ic*(1-c_theta*clip(theta,0,1)); Qh=max(P-Ic_eff,0);
        #   Asat=clip((theta-theta_sat)/(1-theta_sat),0,1); Qs=P*Asat;
        #   Qcombined=min(P,max(Qscs,Qh,Qs)) or min(P,Qscs+Qh+Qs).
        # Parameter meanings:
        #   Ic is infiltration capacity; c_theta is its empirical moisture response;
        #   theta_sat is the area-fraction threshold; combination method is explicit.
        # Units:
        #   P/Ic/Qh/Qs/Qcombined [mm h-1]; theta, c_theta and Asat [1].
        # Domain/constraints:
        #   These are the configured point partition laws, not distributed routing;
        #   select exactly the declared combination method and preserve the input cap.
        # Sign convention:
        #   Positive rates are outward runoff; remaining forcing infiltrates.
        scs_runoff = self._compute_scs_runoff(
            precip_mm_h,
            dt,
            effective_retention=effective_retention,
            initial_abstraction=initial_abstraction,
        )
        horton_runoff = self._compute_hortonian_runoff(precip_mm_h, soil_moisture, dt)
        saturation_runoff = self._compute_saturation_runoff(precip_mm_h, soil_moisture, dt)
        
        # Combined runoff uses configured rule to avoid implicit assumptions
        if self.runoff_combination_method == "max":
            # Dominant mechanism (non-additive)
            combined_runoff = torch.maximum(
                torch.maximum(scs_runoff, horton_runoff),
                saturation_runoff
            )
        elif self.runoff_combination_method == "sum":
            combined_runoff = scs_runoff + horton_runoff + saturation_runoff
        else:
            raise ValueError(
                f"Unsupported runoff_combination_method: {self.runoff_combination_method}"
            )
        
        # Apply upper bound (cannot exceed precipitation)
        combined_runoff = torch.minimum(combined_runoff, precip_mm_h)
        
        return combined_runoff
    
    def _ensure_tensor(self, value: Any) -> Tensor:
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
        return super()._ensure_tensor(value)


# Export the main class
__all__ = ["SurfaceRunoff"]
