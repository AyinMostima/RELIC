"""
Bucket Soil Moisture Model for RELIC

This module implements a simple bucket model for soil water balance
calculations in the RELIC framework. The bucket model provides
a simplified approach to soil moisture dynamics based on water 
balance principles and empirical relationships.

References
----------
- Manabe, S. (1969). Climate and the ocean circulation. I. The atmospheric 
  circulation and the hydrology of the earth's surface. Monthly Weather Review, 
  97(11), 739-774.
- Huang, J., van den Dool, H. M., and Georgakakos, K. P. (1996). Analysis of
  model-calculated soil moisture over the United States (1931-1993) and
  applications to long-range temperature forecasts. Journal of Climate, 9,
  1350-1362.
- Rodriguez-Iturbe, I. and Porporato, A. (2004). Ecohydrology of water-controlled
  ecosystems: soil moisture and plant dynamics. Cambridge University Press.

Mathematical Expressions
------------------------
Water balance equation:
    dtheta/dt = (P - ET - Q - D) / (rho_w * Z)

Where:
    theta = volumetric soil water content [m3 m-3]
    P = precipitation rate [kg m-2 s-1]
    ET = evapotranspiration rate [kg m-2 s-1]
    Q = surface runoff [kg m-2 s-1]
    D = drainage/deep percolation [kg m-2 s-1]
    Z = effective soil depth [m]
    rho_w = water density [kg m-3]

Runoff generation:
    theta_p = theta + (P / rho_w - ET_actual / rho_w) * dt / Z
    Q = max(theta_p - theta_sat, 0) * Z * rho_w / dt

Drainage:
    D_p = rho_w * f_D * K_sat * (theta_r/theta_sat)^b
    D = min(D_p, max(theta_r - theta_fc, 0) * Z * rho_w / dt)

Constraints:
    theta_wp <= theta <= theta_sat

Parameters
----------
soil_depth : float
    Effective soil layer depth [m]
field_capacity : float
    Field capacity [m3 m-3]
wilting_point : float
    Permanent wilting point [m3 m-3]
porosity : float
    Saturation water content (porosity) [m3 m-3]
saturated_hydraulic_conductivity : float
    Saturated hydraulic conductivity [m s-1]
drainage_exponent : float
    Drainage curve exponent [dimensionless]
"""

import math
from typing import Any, Dict, List, Mapping, Optional, Union
import torch
from torch import Tensor

from relic.computation.operators.dec.forms import CochainSpaceSpec
from relic.computation.operators.dec.hodge import HodgeStarSpec
from relic.computation.operators.iphs import PortSpecIPHS
from relic.core.functor import (
    ConflictResolutionPolicy,
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    TransportMechanismFunctor,
    SpatialScale,
    SubFlowAccuracy,
    VariableKind,
    TemporalScale,
)
from relic.utils.physical_constants import get_physical_constant
from relic.core.abstractions import FluxDict, StateDict
from relic.core.state import UpdateMode, UpdateItem
from ..interface_mapping import HydrosphereInterfaceMixin


class BucketSoilMoisture(HydrosphereInterfaceMixin, TransportMechanismFunctor):
    """
    Bucket Soil Moisture Model
    
    Implements a simple bucket model for soil water dynamics based on
    water balance principles. The model tracks soil moisture content
    in a single layer and partitions water inputs into evapotranspiration,
    runoff, and drainage based on soil hydraulic properties.
    
    The implementation follows CF-1.8 conventions for variable naming
    and units, and provides mass-conservative water balance calculations.
    
    Units Convention (CF-1.8 consistent):
        - soil_moisture: m3 m-3 (volumetric water content)
        - precipitation: kg m-2 s-1
        - evapotranspiration: kg m-2 s-1
        - runoff: kg m-2 s-1
        - drainage: kg m-2 s-1
        - soil_depth: m
    
    Attributes
    ----------
    soil_depth : float
        Effective soil layer depth [m]
    field_capacity : float
        Field capacity water content [m3 m-3]
    wilting_point : float
        Permanent wilting point [m3 m-3]
    saturation : float
        Saturation water content (porosity) [m3 m-3]
    k_sat : float
        Saturated hydraulic conductivity [m s-1]
    drainage_exponent : float
        Drainage curve exponent [dimensionless]
    """

    TRANSPORT_TYPE = "vertical_percolation"

    PHASE8_RUNTIME_REPLACEMENT_STATUS = "canonical_iphs_runtime"
    PHYSICAL_CONSTANT_KEYS = (
        "relic.physical.constants.water.density_liquid",
    )
    # Declared water input of the soil store
    # (hydrosphere.soil_moisture.water_input_source, instance config override).
    # References: E3SM ELM SnowHydrologyMod, column without snow layers:
    #   qflx_top_soil = qflx_rain_grnd + qflx_snomelt; Oleson et al. (2013),
    #   NCAR/TN-503+STR; Manabe (1969); Anderson (1976), NOAA NWS 19.
    # Mathematical expression: P_in = P_total for "precipitation_flux";
    #   P_in = P_rain + M for "rain_flux_plus_snow_melt_flux".
    # Parameter meanings: P_total is precipitation_flux (no snowpack Functor
    #   upstream); P_rain the rain reaching the ground (rain_flux) and M the
    #   snowmelt release (snow_melt_flux) of the snowpack partition.
    # Units: P_in, P_total, P_rain, M [kg m-2 s-1].
    # Domain/constraints: exactly one representation is declared and read; the
    #   state content never selects it. With the snowpack source the SCM owner of
    #   rain_flux and snow_melt_flux is a lagged provider: the bucket consumes the
    #   committed outflow (no same-step edge), which is the acknowledged
    #   zero-order hold when the snowpack runs in a slower execution group.
    # Sign convention: non-negative downward mass fluxes into the soil store.
    WATER_INPUT_SOURCES = ("precipitation_flux", "rain_flux_plus_snow_melt_flux")
    
    def __init__(self, 
                 name: str = "bucket_soil_moisture",
                 config: Optional[Dict] = None) -> None:
        """
        Initialize Bucket soil moisture model.
        
        Parameters
        ----------
        name : str, optional
            Unique identifier for this functor instance
        config : Dict, optional
            Configuration parameters for soil properties
        """
        # No default_config - all parameters from centralized parameter system
        # Merge with user config if provided
        merged_config = config or {}
            
        # Create metadata
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Bucket model for soil water balance with runoff and drainage partitioning",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "theta_dot = (P - ET - Q - D) / (rho_w * Z); q = star_K g_h with projection residual routing"
            ),
            references=[
                "Manabe, S. (1969). Monthly Weather Review, 97(11), 739-774",
                "Huang, J. et al. (1996). Journal of Climate, 9(6), 1350-1362",
                "Rodriguez-Iturbe, I. and Porporato, A. (2004). Ecohydrology of water-controlled ecosystems"
            ],
            parameters={
                "hydrosphere.soil_moisture.soil_depth": (
                    "Effective soil layer depth for bucket model water balance [m]"
                ),
                "hydrosphere.soil_moisture.field_capacity": (
                    "Soil field capacity (maximum water content at gravitational drainage "
                    "threshold) [m3 m-3]"
                ),
                "hydrosphere.soil_moisture.wilting_point": (
                    "Soil wilting point (minimum plant-available water content) [m3 m-3]"
                ),
                "hydrosphere.soil_moisture.porosity": (
                    "Total soil porosity (saturated water content) [m3 m-3]"
                ),
                "hydrosphere.soil_moisture.saturated_hydraulic_conductivity": (
                    "Saturated hydraulic conductivity for Richards equation [m s-1]"
                ),
                "hydrosphere.soil_moisture.drainage_factor": (
                    "Free-drainage factor f_d of K_eff = f_d K_sat [dimensionless]"
                ),
                "hydrosphere.soil_moisture.drainage_exponent": (
                    "Drainage curve exponent for bucket model (power law parameter) [dimensionless]"
                ),
                "hydrosphere.soil_moisture.water_input_source": (
                    "Declared water input of the bucket soil store [dimensionless]"
                ),
            },
            input_variables=[
                "soil_moisture", "precipitation_flux", "evapotranspiration_flux"
            ],
            output_variables=[
                "soil_moisture",
                "surface_runoff_flux",
                "drainage_flux",
                "diagnostics.soil_moisture_bucket",
            ],
            conservation_laws=["mass"],
            spatial_dimensions=0,  # Point-based model
            temporal_scale=TemporalScale.SECOND,
            coupling_interfaces=["evapotranspiration", "runoff", "groundwater"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
            # References: Manabe (1969); Rodriguez-Iturbe and Porporato (2004);
            #   Hairer, Norsett and Wanner (1993), Sect. II.3 (explicit Euler).
            # Mathematical expression: for unforced smooth drainage,
            #   theta_dot=-k*theta^b and theta1=theta-h*k*theta^b,
            #   k=Ks*fD/(Z*theta_sat^b). The local state error is
            #   -0.5*b*k^2*theta^(2*b-1)*h^2 + O(h^3), hence ORDER_1.
            #   The held-input P-ET/runoff/drainage sequence is first-order;
            #   the REPLACE state is already integrated over the call.
            # Parameter meanings: theta/theta1 are current/updated water contents;
            #   theta_fc/theta_sat are field capacity/porosity; Ks is saturated
            #   conductivity, fD the drainage factor, Z depth, b the exponent,
            #   k the drainage coefficient, and h the call duration;
            #   P/ET are precipitation/evapotranspiration mass fluxes.
            # Units: theta/theta1/theta_fc/theta_sat [m3 m-3]; Ks [m s-1]; Z [m];
            #   fD/b [1]; k [s-1]; h [s]; P/ET [kg m-2 s-1];
            #   local state error [m3 m-3].
            # Domain/constraints: fixed inputs and parameters on a smooth branch;
            #   Z,Ks>0, fD>=0, b>=1, and h>0 under the parameter/input contracts.
            #   T11 uses P=ET=0, theta_fc<theta<theta_sat and
            #   h*k*theta^b<theta-theta_fc with nonzero drainage. Active ET,
            #   runoff, drainage caps or projection are nonsmooth boundaries;
            #   their coarse-step behavior is not a higher-order claim. The S1
            #   fast snowpack-to-bucket flux-coupling refusal is unchanged.
            # Sign convention: positive drainage removes soil water; the Euler
            #   state error above is negative in the smooth unforced witness.
            sub_flow_accuracy=SubFlowAccuracy.ORDER_1,
            conflict_resolution_policy=ConflictResolutionPolicy.EXCLUSIVE,
        )
        
        # Initialize parent classes
        super().__init__(name=name, config=merged_config, metadata=metadata)
        
        def _to_tensor(value):
            return self._ensure_tensor(value)
        
        # Set TransportMechanism attributes. Gravity drainage of the bucket is vertical
        # percolation D = f_d K_sat (theta/theta_sat)^b with no diffusion term, so
        # diffusivity is None and no tortuosity is read.
        self.peclet_number = None
        self.velocity_field = None
        
        # Get parameters from centralized parameter system with full namespace
        self.soil_depth = _to_tensor(self.get_parameter_value(
            "soil_depth",
            namespace="hydrosphere.soil_moisture",
            validate=True,
            required=True,
        ))
        self.field_capacity = _to_tensor(self.get_parameter_value(
            "field_capacity",
            namespace="hydrosphere.soil_moisture",
            validate=True,
            required=True,
        ))
        self.wilting_point = _to_tensor(self.get_parameter_value(
            "wilting_point",
            namespace="hydrosphere.soil_moisture",
            validate=True,
            required=True,
        ))
        self.saturation = _to_tensor(self.get_parameter_value(
            "porosity",
            namespace="hydrosphere.soil_moisture",
            validate=True,
            required=True,
        ))
        self.k_sat = _to_tensor(self.get_parameter_value(
            "saturated_hydraulic_conductivity",
            namespace="hydrosphere.soil_moisture",
            validate=True,
            required=True,
        ))
        self.drainage_factor = _to_tensor(self.get_parameter_value(
            "drainage_factor",
            namespace="hydrosphere.soil_moisture",
            validate=True,
            required=True,
        ))
        self.drainage_exponent = _to_tensor(self.get_parameter_value(
            "drainage_exponent",
            namespace="hydrosphere.soil_moisture",
            validate=True,
            required=True,
        ))
        self.water_density = get_physical_constant(
            "water_density",
            as_tensor=True,
            device=getattr(self, "device", None),
        )
        # Declared water input (YAML SSOT; an instance config key of the same name
        # overrides it); see WATER_INPUT_SOURCES.
        self.water_input_source = self.get_parameter_value(
            "water_input_source",
            namespace="hydrosphere.soil_moisture",
            source_order=("config", "manager"),
            expected_type=str,
            required=True,
        )
        if self.water_input_source not in self.WATER_INPUT_SOURCES:
            raise ValueError(
                "water_input_source must be one of "
                f"{list(self.WATER_INPUT_SOURCES)}, got {self.water_input_source!r}"
            )
        self._snowpack_provider: Optional[str] = None
        if self.water_input_source == "rain_flux_plus_snow_melt_flux":
            # Only the declared representation is a declared input.
            self.metadata.input_variables.remove("precipitation_flux")
            self.metadata.input_variables.extend(["rain_flux", "snow_melt_flux"])
            self._snowpack_provider = self._snowpack_input_owner()
        
        # Validate parameter consistency
        self._validate_soil_parameters()

    @staticmethod
    def _snowpack_input_owner() -> str:
        """Return the one SCM owner of rain_flux and snow_melt_flux (the snowpack)."""
        from relic.contracts import load_contract_registry

        registry = load_contract_registry()
        owners = {registry.get_owner("rain_flux"), registry.get_owner("snow_melt_flux")}
        if len(owners) != 1 or None in owners:
            raise ValueError(
                "rain_flux and snow_melt_flux must share one SCM owner to be read as the "
                f"snowpack water input of the bucket, got {sorted(map(str, owners))}"
            )
        return owners.pop()

    def _validate_soil_parameters(self) -> None:
        """Validate soil parameter consistency and physical constraints."""
        def _to_scalar(value) -> float:
            return self._scalar_from_value(value)

        wilting_point = _to_scalar(self.wilting_point)
        field_capacity = _to_scalar(self.field_capacity)
        saturation = _to_scalar(self.saturation)
        soil_depth = _to_scalar(self.soil_depth)
        k_sat = _to_scalar(self.k_sat)

        if wilting_point >= field_capacity:
            raise ValueError(
                f"Wilting point ({wilting_point}) must be less than field capacity ({field_capacity})"
            )
        
        if field_capacity >= saturation:
            raise ValueError(
                f"Field capacity ({field_capacity}) must be less than saturation ({saturation})"
            )
        
        if wilting_point < 0 or saturation > 1:
            raise ValueError("Soil water contents must be in range [0, 1]")
        
        if soil_depth <= 0:
            raise ValueError("Soil depth must be positive")
        
        if k_sat <= 0:
            raise ValueError("Saturated hydraulic conductivity must be positive")
    
    def validate_state(self, state: Dict[str, torch.Tensor]) -> List[str]:
        """Validate required inputs without mutating the incoming state."""
        errors: List[str] = []
        if "soil_moisture" not in state:
            errors.append("Missing required state variable: soil_moisture")
        return errors

    def get_dependencies(self) -> Dict[str, List[str]]:
        """Return the canonical scheduling relation for ET boundary fluxes.

        Evapotranspiration may be supplied by the external StateManager adapter.
        When an in-model ET closure is selected, the previous committed value is
        consumed to avoid an algebraic cycle with the soil-water state. With the
        water input source rain_flux_plus_snow_melt_flux the snowpack owner of
        rain_flux and snow_melt_flux is a lagged provider as well: its committed
        outflow is consumed, held between its calls when it runs in a slower
        execution group.
        """

        lagged = ["penman_monteith_et", "priestley_taylor_et"]
        snowpack_provider = getattr(self, "_snowpack_provider", None)
        if snowpack_provider is not None:
            lagged.append(snowpack_provider)
        return self._merge_declared_dependencies({
            "required": [],
            "optional": [],
            "lagged": lagged,
        })

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
        """Declare canonical cochain spaces for the bucket water balance.

        References:
          - Manabe (1969), single-layer bucket land hydrology.
          - Rodriguez-Iturbe and Porporato (2004), soil moisture dynamics.
          - Hirani (2003), Discrete Exterior Calculus.
        Mathematical expression:
          theta_dot = (P - ET - Q - D) / (rho_w * Z).
        Parameter meanings:
          theta is soil water content [m3 m-3], P/ET/Q/D are matter fluxes
          [kg m-2 s-1], rho_w is water density [kg m-3], and Z is depth [m].
        """

        n_dofs = self._structural_n_dofs()
        return (
            CochainSpaceSpec(
                complex_id="hydrosphere_bucket_soil_moisture",
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
                complex_id="hydrosphere_bucket_soil_moisture",
                degree=2,
                primality="dual",
                n_dofs=n_dofs,
                units="kg m-2 s-1",
                extensivity="integral",
                physical_domain="hydrosphere",
                semantic_name="precipitation_flux",
                orientation_convention="downward_positive",
                boundary_role="boundary",
            ),
            CochainSpaceSpec(
                complex_id="hydrosphere_bucket_soil_moisture",
                degree=2,
                primality="dual",
                n_dofs=n_dofs,
                units="kg m-2 s-1",
                extensivity="integral",
                physical_domain="hydrosphere",
                semantic_name="evapotranspiration_flux",
                orientation_convention="upward_positive",
                boundary_role="port",
            ),
            CochainSpaceSpec(
                complex_id="hydrosphere_bucket_soil_moisture",
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
            CochainSpaceSpec(
                complex_id="hydrosphere_bucket_soil_moisture",
                degree=2,
                primality="dual",
                n_dofs=n_dofs,
                units="kg m-2 s-1",
                extensivity="integral",
                physical_domain="hydrosphere",
                semantic_name="drainage_flux",
                orientation_convention="downward_positive",
                boundary_role="port",
            ),
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Declare ParameterManager keys used by runtime and structural builders."""

        return (
            "hydrosphere.soil_moisture.soil_depth",
            "hydrosphere.soil_moisture.field_capacity",
            "hydrosphere.soil_moisture.wilting_point",
            "hydrosphere.soil_moisture.porosity",
            "hydrosphere.soil_moisture.saturated_hydraulic_conductivity",
            "hydrosphere.soil_moisture.drainage_factor",
            "hydrosphere.soil_moisture.drainage_exponent",
            "hydrosphere.soil_moisture.water_input_source",
        )

    def physical_constant_keys(self) -> tuple[str, ...]:
        """Return physical constants kept outside the domain parameter map."""

        return self.PHYSICAL_CONSTANT_KEYS

    def build_hodge_contract(
        self,
        n_dofs: Optional[int] = None,
        *,
        strict: bool = True,
    ) -> HodgeStarSpec:
        """Build the bucket hydraulic mobility Hodge contract.

        References:
          - Manabe (1969), bucket water balance.
          - Rodriguez-Iturbe and Porporato (2004), soil-water conductivity controls.
          - van der Schaft and Maschke (1995), dissipative port systems.
        Mathematical expression:
          star_K = diag(rho_w * K_sat), with rho_w > 0 and K_sat > 0.
        Parameter meanings:
          rho_w is liquid water density [kg m-3]; K_sat is saturated hydraulic
          conductivity [m s-1].
        """

        n_dofs = self._structural_n_dofs(n_dofs=n_dofs)
        mobility_value = self._scalar_from_value(self.water_density) * self._scalar_from_value(self.k_sat)
        values = torch.full(
            (n_dofs,),
            mobility_value,
            dtype=torch.float32,
            device=getattr(self, "device", torch.device("cpu")),
        )
        if bool(torch.any(values <= 0.0)):
            raise ValueError("Bucket soil moisture mobility must be positive.")

        domain = CochainSpaceSpec(
            complex_id="hydrosphere_bucket_soil_moisture",
            degree=1,
            primality="primal",
            n_dofs=n_dofs,
            units="m m-1",
            extensivity="intensive",
            physical_domain="hydrosphere",
            semantic_name="bucket_hydraulic_gradient",
            orientation_convention="downward_positive",
        )
        codomain = CochainSpaceSpec(
            complex_id="hydrosphere_bucket_soil_moisture",
            degree=2,
            primality="dual",
            n_dofs=n_dofs,
            units="kg m-2 s-1",
            extensivity="integral",
            physical_domain="hydrosphere",
            semantic_name="bucket_water_flux",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        hodge = HodgeStarSpec(
            name=f"{self.name}_bucket_hydraulic_hodge",
            domain=domain,
            codomain=codomain,
            matrix=torch.diag(values),
            parameter_keys=(
                "hydrosphere.soil_moisture.saturated_hydraulic_conductivity",
                "relic.physical.constants.water.density_liquid",
            ),
            metric_keys=("hydrosphere.soil_moisture.soil_depth",),
            material_law="single_layer_bucket_water_balance",
        )
        if strict:
            hodge.validate(passive=True)
        return hodge

    def build_hodge_terms(
        self,
        complex_bundle: Any = None,
        resolved_parameters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[HodgeStarSpec, ...]:
        """Build bucket Hodge terms for IPHS assembly."""

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
        """Return no reversible interconnection for bucket water partitioning."""

        return None

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the positive bucket hydraulic mobility contribution.

        References:
          - Manabe (1969), Eagleson (1978), and van der Schaft and Maschke (1995).
        Mathematical expression:
          M_h = diag(rho_w * K_sat), sigma = X_h.T M_h X_h >= 0.
        Parameter meanings:
          K_sat is saturated hydraulic conductivity [m s-1].
        """

        hodge = self.build_hodge_contract(
            n_dofs=self._structural_n_dofs(complex_bundle=complex_bundle),
            strict=True,
        )
        matrix = hodge.matrix
        return {
            "state_vars": tuple(
                f"bucket_soil_water_potential[{idx}]" for idx in range(int(matrix.shape[0]))
            ),
            "J": torch.zeros_like(matrix),
            "M": matrix,
            "ports": self.ports(),
        }

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Declare bucket water balance matter ports."""

        effort_space = CochainSpaceSpec(
            complex_id="hydrosphere_bucket_soil_moisture",
            degree=0,
            primality="primal",
            n_dofs=1,
            units="m",
            extensivity="intensive",
            physical_domain="hydrosphere",
            semantic_name="bucket_water_potential",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        flow_kwargs = {
            "complex_id": "hydrosphere_bucket_soil_moisture",
            "degree": 2,
            "primality": "dual",
            "n_dofs": 1,
            "units": "kg m-2 s-1",
            "extensivity": "integral",
            "physical_domain": "hydrosphere",
            "orientation_convention": "outward_positive",
            "boundary_role": "port",
        }
        return (
            PortSpecIPHS(
                name="bucket_evapotranspiration_flux",
                effort_var="bucket_water_potential",
                flow_var="evapotranspiration_flux",
                effort_space=effort_space,
                flow_space=CochainSpaceSpec(
                    **flow_kwargs,
                    semantic_name="bucket_evapotranspiration_mass_flux",
                ),
                power_units="kg m-1 s-1",
                positive_power_convention="out_of_system",
                port_kind="matter",
            ),
            PortSpecIPHS(
                name="bucket_surface_runoff_flux",
                effort_var="bucket_water_potential",
                flow_var="surface_runoff_flux",
                effort_space=effort_space,
                flow_space=CochainSpaceSpec(
                    **flow_kwargs,
                    semantic_name="bucket_surface_runoff_mass_flux",
                ),
                power_units="kg m-1 s-1",
                positive_power_convention="out_of_system",
                port_kind="matter",
            ),
            PortSpecIPHS(
                name="bucket_drainage_flux",
                effort_var="bucket_water_potential",
                flow_var="drainage_flux",
                effort_space=effort_space,
                flow_space=CochainSpaceSpec(
                    **flow_kwargs,
                    semantic_name="bucket_drainage_mass_flux",
                ),
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
    
    def _scalar_from_value(self, value: Union[float, int, Tensor]) -> float:
        """Convert numeric values to Python float without tensor.item/cpu sync API."""
        if isinstance(value, Tensor):
            detached = value.detach().reshape(-1)
            if detached.numel() == 0:
                raise ValueError("Cannot convert empty tensor to scalar.")
            return detached[:1].to(dtype=torch.float64).tolist()[0]
        return float(value)

    def forward(self, 
        # References:
        #   Manabe (1969); Rodriguez-Iturbe and Porporato (2004).
        # Mathematical expression:
        #   Pvol=P/rho_w; Evol=min(ET/rho_w,max((theta-theta_wp)*L/dt,0));
        #   theta_star=theta+(Pvol-Evol)*dt/L; Qvol=max(theta_star-theta_sat,0)*L/dt;
        #   Dpot=Ks*fD*(theta_after_runoff/theta_sat)^b above field capacity;
        #   Dvol=min(Dpot,max(theta_after_runoff-theta_fc,0)*L/dt);
        #   theta1=clip(theta_after_runoff-Dvol*dt/L,theta_wp,theta_sat).
        # Parameter meanings:
        #   L is soil depth; theta_wp/theta_fc/theta_sat are hydraulic limits;
        #   fD and b govern drainage. The actual code subsequently accounts for
        #   projection in the drainage correction and discloses any remaining residual.
        # Units:
        #   theta [m3 m-3], L [m], dt [s], P/ET [kg m-2 s-1];
        #   Pvol/Evol/Qvol/Dvol/Ks [m s-1], soil-water storage [kg m-2].
        # Domain/constraints:
        #   Finite floating inputs and positive dt; actual ET cannot exceed available
        #   water. A diagnostic clipping residual discloses imbalance rather than
        #   closing it physically. REPLACE theta1 is already an integrated state.
        # Sign convention:
        #   Precipitation enters; ET, runoff and drainage leave the soil column.
                state: StateDict, 
                time: float, 
                dt: float,
                **kwargs) -> FluxDict:
        """
        Compute soil moisture change using bucket model water balance.
        
        Parameters
        ----------
        state : StateDict
            Current system state containing water balance variables
        time : float
            Current simulation time [s]
        dt : float  
            Time step [s]
        **kwargs
            Additional context variables
            
        Returns
        -------
        FluxDict
            Soil moisture change and water balance components
            
        Raises
        ------
        ValueError
            If required state variables are missing or invalid
        """
        if not isinstance(state, Mapping):
            raise TypeError("Bucket soil moisture state must be a mapping.")
        local_state = dict(state)

        # Only the declared water input representation is required and read.
        required_state = tuple(self.metadata.input_variables)
        for state_key in required_state:
            if state_key not in local_state or local_state[state_key] is None:
                raise ValueError(
                    f"Missing required state variable: {state_key}"
                )
            state_value = local_state[state_key]
            if not isinstance(state_value, Tensor):
                raise TypeError(f"{state_key} must be a torch.Tensor")
            if not state_value.is_floating_point():
                raise TypeError(f"{state_key} must use a floating-point dtype")
            if not bool(torch.all(torch.isfinite(state_value))):
                raise ValueError(f"{state_key} must be finite")

        dt_value = self._scalar_from_value(dt)
        if not math.isfinite(dt_value) or dt_value <= 0.0:
            raise ValueError("Time step must be finite and positive")

        # References:
        #   Manabe (1969); Rodriguez-Iturbe and Porporato (2004).
        # Mathematical expression:
        #   0 <= theta_wp < theta_fc < theta_sat <= 1, L > 0, Ks > 0.
        # Parameter meanings:
        #   theta_wp, theta_fc and theta_sat are the wilting point, field capacity
        #   and porosity used by this call; L is soil depth and Ks the saturated
        #   hydraulic conductivity.
        # Units:
        #   theta_wp, theta_fc, theta_sat [m3 m-3], L [m], Ks [m s-1].
        # Domain/constraints:
        #   A learnable binding can change after construction, and each value can
        #   lie inside its own owner bounds while the order fails. The order checked
        #   at construction is therefore checked again on the values of this call;
        #   a violation raises ValueError before the water balance and before
        #   (theta - theta_wp) / (theta_fc - theta_wp) is formed.
        # Sign convention:
        #   Not applicable (ordering of water-content thresholds).
        self._validate_soil_parameters()

        soil_moisture_raw = local_state["soil_moisture"].to(device=self.device)
        # References: E3SM ELM SnowHydrologyMod, column without snow layers:
        #   qflx_top_soil = qflx_rain_grnd + qflx_snomelt; Oleson et al. (2013),
        #   NCAR/TN-503+STR; Anderson (1976), NOAA NWS 19.
        # Mathematical expression: P = P_total for the source "precipitation_flux";
        #   P = P_rain + M for the source "rain_flux_plus_snow_melt_flux".
        # Parameter meanings: P is the water input of the bucket balance below;
        #   P_total is precipitation_flux; P_rain the rain reaching the ground
        #   (rain_flux) and M the snowmelt release (snow_melt_flux) of the snowpack.
        # Units: P, P_total, P_rain, M [kg m-2 s-1].
        # Domain/constraints: only the declared representation is read (the other
        #   one is ignored even when present), so precipitation already partitioned
        #   by a snowpack (snowfall held in the pack, rain, melt) is never counted
        #   twice; P must be finite and non-negative (checked below).
        # Sign convention: non-negative downward mass flux into the soil store.
        if self.water_input_source == "precipitation_flux":
            water_input = local_state["precipitation_flux"]
        else:
            water_input = local_state["rain_flux"] + local_state["snow_melt_flux"].to(
                device=local_state["rain_flux"].device,
                dtype=local_state["rain_flux"].dtype,
            )
        precipitation = water_input.to(
            device=self.device,
            dtype=soil_moisture_raw.dtype,
        )
        evapotranspiration = local_state["evapotranspiration_flux"].to(
            device=self.device,
            dtype=soil_moisture_raw.dtype,
        )
        
        # Unit conversion: Handle CF-1.8 vs internal scientific units
        soil_depth_value = self._ensure_tensor(self.soil_depth, dtype=precipitation.dtype).to(
            device=precipitation.device,
            dtype=precipitation.dtype,
        )
        water_density_value = self._ensure_tensor(self.water_density, dtype=precipitation.dtype).to(
            device=precipitation.device,
            dtype=precipitation.dtype,
        )
        
        # Canonical soil moisture is always volumetric [m3 m-3]. Unit conversion
        # belongs to the central adapter and UnitConverter before StateManager input.
        soil_moisture = soil_moisture_raw

        target_device = soil_moisture.device
        target_dtype = soil_moisture.dtype
        soil_depth_tensor = self._ensure_tensor(self.soil_depth, dtype=target_dtype).to(
            device=target_device, dtype=target_dtype
        )
        field_capacity_tensor = self._ensure_tensor(self.field_capacity, dtype=target_dtype).to(
            device=target_device, dtype=target_dtype
        )
        wilting_point_tensor = self._ensure_tensor(self.wilting_point, dtype=target_dtype).to(
            device=target_device, dtype=target_dtype
        )
        saturation_tensor = self._ensure_tensor(self.saturation, dtype=target_dtype).to(
            device=target_device, dtype=target_dtype
        )
        k_sat_tensor = self._ensure_tensor(self.k_sat, dtype=target_dtype).to(
            device=target_device, dtype=target_dtype
        )
        drainage_exponent_tensor = self._ensure_tensor(self.drainage_exponent, dtype=target_dtype).to(
            device=target_device, dtype=target_dtype
        )
        water_density_tensor = self._ensure_tensor(self.water_density, dtype=target_dtype).to(
            device=target_device, dtype=target_dtype
        )
        
        # Validate input ranges
        if torch.any(soil_moisture < 0) or torch.any(soil_moisture > 1):
            raise ValueError("Soil moisture content must be in range [0, 1]")
        
        if bool(torch.any(precipitation < 0.0)):
            raise ValueError("precipitation_flux must be non-negative")
        if bool(torch.any(evapotranspiration < 0.0)):
            raise ValueError("evapotranspiration_flux must be non-negative")
        
        # References:
        # - Manabe (1969), sequential single-bucket partitioning.
        # - Rodriguez-Iturbe and Porporato (2004), physically constrained soil-water flux partition.
        # Mathematical expression:
        # - Step 1: ET_act = min(ET, max((theta - theta_wilt) * Z / dt, 0))
        # - Step 2: theta_p = theta + (P - ET_act) * dt / Z
        # - Step 3: R = max(theta_p - theta_sat, 0) * Z / dt
        # - Step 4: D = min(f_d Ksat (theta_r/theta_sat)^b, max(theta_r - theta_fc, 0) Z / dt)
        #   for theta_r > theta_fc, and D = 0 at or below field capacity
        # - Step 5: theta_new = clamp(theta_r - D*dt/Z, theta_wilt, theta_sat)
        # Parameter meanings:
        # - theta: volumetric soil moisture; theta_r after runoff removal
        # - theta_wilt/theta_fc/theta_sat: wilting/field-capacity/saturation
        # - Z: soil depth, dt: timestep; f_d drainage_factor; b drainage_exponent
        # - P, ET_act, R, D: volumetric fluxes
        # Units: theta [m3 m-3]; Z [m]; dt [s]; Ksat and fluxes [m s-1]; f_d, b [1].
        # Domain/constraints: dt > 0 and Z > 0; drainage only above field capacity and
        #   bounded by the storage above it (no minimum drainage floor exists: below
        #   field capacity the storage bound is zero).
        # Sign convention: P adds water; ET_act, R and D remove water from the bucket.
        # 1) Convert input fluxes to volumetric units.
        precip_vol = precipitation / water_density_value  # m s-1
        et_vol = evapotranspiration / water_density_value  # m s-1

        # 2) Limit ET by available water above wilting point.
        max_et_vol = torch.clamp(
            (soil_moisture - wilting_point_tensor) * soil_depth_tensor / dt_value,
            min=0.0,
        )
        actual_et_vol = torch.minimum(et_vol, max_et_vol)
        net_input_rate = precip_vol - actual_et_vol

        # 3) Update provisional moisture after P-ET.
        theta_provisional = (
            soil_moisture + net_input_rate * dt_value / soil_depth_tensor
        )

        # 4) Saturation excess generates runoff first.
        runoff_excess = torch.clamp(theta_provisional - saturation_tensor, min=0.0)
        runoff_vol = runoff_excess * soil_depth_tensor / dt_value
        theta_after_runoff = theta_provisional - runoff_excess

        # 5) Drainage only from moisture above field capacity, after runoff removal.
        relative_moisture = torch.clamp(
            theta_after_runoff / saturation_tensor,
            min=0.0,
            max=1.0,
        )
        drainage_factor = self._ensure_tensor(
            self.drainage_factor,
            dtype=soil_moisture.dtype,
        )
        k_sat_effective = k_sat_tensor * drainage_factor
        potential_drainage_vol = torch.where(
            theta_after_runoff > field_capacity_tensor,
            k_sat_effective * torch.pow(relative_moisture, drainage_exponent_tensor),
            torch.zeros_like(theta_after_runoff),
        )
        drainage_excess = torch.clamp(theta_after_runoff - field_capacity_tensor, min=0.0)
        max_drainage = drainage_excess * soil_depth_tensor / dt_value
        drainage_vol = torch.minimum(potential_drainage_vol, max_drainage)

        # 6) Final moisture state with strict physical bounds.
        theta_after_drainage = (
            theta_after_runoff
            - drainage_vol * dt_value / soil_depth_tensor
        )
        new_soil_moisture = torch.clamp(
            theta_after_drainage,
            min=wilting_point_tensor,
            max=saturation_tensor,
        )

        # 7) Conservation closure: absorb clamp residual into drainage.
        # This keeps reported fluxes consistent with actual state change.
        actual_change = new_soil_moisture - soil_moisture
        provisional_change = net_input_rate * dt_value / soil_depth_tensor
        residual_theta = (
            provisional_change
            - runoff_excess
            - (drainage_vol * dt_value / soil_depth_tensor)
            - actual_change
        )
        drainage_correction = residual_theta * soil_depth_tensor / dt_value
        corrected_drainage_vol = torch.clamp(drainage_vol + drainage_correction, min=0.0)
        residual_after_adjust = (
            residual_theta
            - (corrected_drainage_vol - drainage_vol)
            * dt_value
            / soil_depth_tensor
        )
        clipped_mass_residual = (
            residual_after_adjust
            * soil_depth_tensor
            * water_density_tensor
            / dt_value
        )

        # Final non-negative physical fluxes.
        final_et = torch.clamp(actual_et_vol, min=0.0)
        final_runoff = torch.clamp(runoff_vol, min=0.0)
        final_drainage = torch.clamp(corrected_drainage_vol, min=0.0)
        net_input = precip_vol - final_et - final_runoff - final_drainage
        actual_moisture_change = (
            new_soil_moisture - soil_moisture
        ) / dt_value

        # Convert back to mass fluxes [kg m-2 s-1]
        surface_runoff = final_runoff * water_density_tensor
        subsurface_drainage = final_drainage * water_density_tensor
        adjusted_et = final_et * water_density_tensor
        
        # 6. Compute diagnostic variables
        # Available water capacity
        available_water = torch.clamp(
            soil_moisture - wilting_point_tensor, min=0.0
        )
        
        # Relative water content
        relative_water_content = torch.clamp(
            (soil_moisture - wilting_point_tensor) / (field_capacity_tensor - wilting_point_tensor),
            min=0.0, max=1.0
        )
        
        # Water stress factor (1 = no stress, 0 = maximum stress)
        water_stress_factor = relative_water_content
        
        # Keep core state in volumetric units.
        output_soil_moisture = new_soil_moisture  # m3 m-3
        # References: Manabe (1969); CF-1.8 water content equivalence.
        # Mathematical expression: W = theta * Z * rho_w.
        # Parameter meanings: theta [m3 m-3], Z [m], rho_w [kg m-3].
        soil_water_storage = new_soil_moisture * soil_depth_value * water_density_value
        
        diagnostics = {
            "water_stress_factor": water_stress_factor,
            "saturation_deficit": saturation_tensor - new_soil_moisture,
            "available_water_capacity": available_water,
            "relative_water_content": relative_water_content,
            "drainage_rate": final_drainage,
            "runoff_rate": final_runoff,
            "net_water_input": net_input,
            "adjusted_evapotranspiration": adjusted_et,
            "soil_moisture_change_rate": actual_moisture_change,
            "soil_water_storage": soil_water_storage,
            "clipped_mass_residual": clipped_mass_residual,
            "drainage_limiter_residual": potential_drainage_vol - corrected_drainage_vol,
        }

        updates = {
            "soil_moisture": UpdateItem(
                value=output_soil_moisture,
                mode=UpdateMode.REPLACE,
                unit="m3 m-3",
                per_timestep=False,
                source_functor=self.name,
            ),
            "surface_runoff_flux": UpdateItem(
                value=surface_runoff,
                mode=UpdateMode.REPLACE,
                unit="kg m-2 s-1",
                per_timestep=False,
                source_functor=self.name,
            ),
            "drainage_flux": UpdateItem(
                value=subsurface_drainage,
                mode=UpdateMode.REPLACE,
                unit="kg m-2 s-1",
                per_timestep=False,
                source_functor=self.name,
            ),
            "diagnostics.soil_moisture_bucket": diagnostics,
        }
        return updates

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Evaluate the explicit open-bucket water ledger.

        References:
          - Manabe (1969), single-layer bucket water balance.
          - Rodriguez-Iturbe and Porporato (2004), open soil-water storage.
        Mathematical expression:
          R_w = rho_w Z [(theta_new - theta_old) / dt]
                - (P - ET - Q - D).
        Parameter meanings:
          theta is volumetric soil moisture [m3 m-3], rho_w is water density
          [kg m-3], Z is soil depth [m], and P, ET, Q, D are boundary mass
          fluxes [kg m-2 s-1].
        Open-system route:
          Precipitation is the input port; evapotranspiration, surface runoff,
          and drainage are the declared water-export ports.
        """

        del input_state
        diagnostic_key = "diagnostics.soil_moisture_bucket"
        if diagnostic_key not in output_updates:
            raise KeyError(f"Missing conservation diagnostic '{diagnostic_key}'")
        diagnostics = output_updates[diagnostic_key]
        if isinstance(diagnostics, UpdateItem):
            diagnostics = diagnostics.value
        if not isinstance(diagnostics, Mapping):
            raise TypeError("Bucket conservation diagnostics must be a mapping")
        residual = diagnostics.get("clipped_mass_residual")
        if residual is None:
            raise KeyError("Missing conservation diagnostic 'clipped_mass_residual'")
        if isinstance(residual, UpdateItem):
            residual = residual.value
        if not isinstance(residual, Tensor):
            raise TypeError("Bucket mass residual must be a torch.Tensor")
        if not bool(torch.all(torch.isfinite(residual))):
            raise ValueError("Bucket mass residual must be finite")
        magnitude = torch.max(torch.abs(residual))
        return {"mass": float(magnitude.detach().item())}
    

    def get_soil_properties(self) -> Dict[str, float]:
        """
        Get soil hydraulic properties for diagnostics and coupling.
        
        Returns
        -------
        Dict[str, float]
            Soil hydraulic parameters
        """
        def _to_scalar(value) -> float:
            return self._scalar_from_value(value)

        return {
            "soil_depth": _to_scalar(self.soil_depth),
            "field_capacity": _to_scalar(self.field_capacity),
            "wilting_point": _to_scalar(self.wilting_point),
            "saturation": _to_scalar(self.saturation),
            "saturated_hydraulic_conductivity": _to_scalar(self.k_sat),
            "drainage_exponent": _to_scalar(self.drainage_exponent),
            "water_density": _to_scalar(self.water_density)
        }
    
# Export the main class
__all__ = ["BucketSoilMoisture"]

