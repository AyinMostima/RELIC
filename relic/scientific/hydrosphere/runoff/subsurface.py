"""
Subsurface Runoff Model for RELIC

This module implements subsurface runoff generation mechanisms
for the RELIC framework, including baseflow, interflow, and
groundwater discharge based on topographic and soil moisture conditions.

References
----------
- Beven, K.J. and Kirkby, M.J. (1979). A physically based, variable
  contributing area model of basin hydrology. Hydrological Sciences
  Bulletin, 24(1), 43-69.
- Brutsaert, W. and Nieber, J.L. (1977). Regionalized drought flow
  hydrographs from a mature glaciated plateau. Water Resources Research,
  13(3), 637-643.
- Tallaksen, L.M. (1995). A review of baseflow recession analysis.
  Journal of Hydrology, 165(1-4), 349-370.
- Nathan, R.J. and McMahon, T.A. (1990). Evaluation of automated
  techniques for base flow and recession analyses. Water Resources
  Research, 26(7), 1465-1473.

Mathematical Expressions
------------------------
TOPMODEL baseflow:
    q = q0 * exp(-lambda * (f - f_bar))
    f = ln(a/tan(beta))

Exponential recession:
    Qt = Q0 * exp(-alpha * t)

Linear reservoir:
    dS/dt = I - Q
    Q = k * S

Threshold-based interflow:
    Qi = ki * max(0, theta - theta_t)^ni

Open control-volume balance:
    S1 = S0 + R * dt - Q_total * dt

Where:
    q = specific discharge per unit area [m s-1]
    q0 = baseflow at mean topographic index [m s-1]
    lambda = recession parameter [m-1]
    f = topographic index [dimensionless]
    f_bar = catchment-mean topographic index [dimensionless]
    a = upslope contributing area per unit contour length [m]
    beta = local slope angle [rad]
    Qt = discharge at time t [m3 s-1]
    Q0 = initial discharge [m3 s-1]
    alpha = recession coefficient [s-1]
    S = storage [m3]
    I = inflow [m3 s-1]
    k = recession constant [s-1]
    theta = soil moisture [m3 m-3]
    theta_t = threshold moisture content [m3 m-3]
    R = incoming drainage depth rate [mm s-1]
    Q_total = regular runoff plus routed capacity overflow [mm s-1]

Parameters
----------
recession_coefficient : float
    Baseflow recession coefficient [s-1]
topographic_index : float
    Mean topographic index [dimensionless]
transmissivity : float
    Soil transmissivity [m2 s-1]
threshold_moisture : float
    Threshold for interflow generation [m3 m-3]
interflow_coefficient : float
    Interflow recession coefficient [s-1]
"""

import math
from typing import Any, Dict, Mapping, Optional, Union
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
from ..interface_mapping import HydrosphereInterfaceMixin
from relic.data.standards.unit_converter import ConversionContext, unit_converter
from relic.utils.physical_constants import get_physical_constant

class SubsurfaceRunoff(HydrosphereInterfaceMixin, TransportMechanismFunctor):
    """
    Subsurface Runoff Generation Model
    
    Implements multiple subsurface runoff mechanisms including:
    - TOPMODEL-based baseflow with topographic index
    - Exponential recession for groundwater discharge
    - Linear reservoir model for baseflow
    - Threshold-based interflow from shallow groundwater
    
    The model simulates the delayed response of subsurface water
    contributions to streamflow based on soil moisture conditions,
    topography, and groundwater storage.

    This is an open-boundary closure. ``drainage_flux`` is the required
    boundary inflow, while ``subsurface_runoff_flux`` is the unique boundary
    export and includes the capacity-overflow component. The retained
    ``groundwater_storage`` output is a complete replacement state.
    
    Units Convention (CF-1.8 consistent):
        - subsurface_runoff_flux: kg m-2 s-1
        - soil_moisture: m3 m-3
        - groundwater_storage: mm
        - transmissivity: m2 s-1
        - recession_coefficient: s-1
    
    Attributes
    ----------
    recession_coefficient : float
        Baseflow recession coefficient [s-1]
    topographic_index : float
        Mean topographic index [dimensionless]
    transmissivity : float
        Soil transmissivity [m2 s-1]
    threshold_moisture : float
        Threshold for interflow [m3 m-3]
    interflow_coefficient : float
        Interflow recession coefficient [s-1]
    """

    TRANSPORT_TYPE = "advection"

    PHASE8_RUNTIME_REPLACEMENT_STATUS = "canonical_iphs_runtime"
    
    def __init__(self,
                 name: str = "subsurface_runoff",
                 config: Optional[Dict] = None) -> None:
        """
        Initialize Subsurface runoff model.

        Parameters
        ----------
        name : str, optional
            Unique identifier for this functor instance
        config : Dict, optional
            Configuration parameters for subsurface flow mechanisms
        """
        # Create metadata
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Subsurface runoff generation using TOPMODEL, exponential recession, and interflow",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "S_dot = I_drain - D_h q_total; "
                "q_total = star_T g_h + q_interflow + q_capacity; star_T >= 0"
            ),
            references=[
                "Beven, K.J. and Kirkby, M.J. (1979). Hydrological Sciences Bulletin, 24(1), 43-69",
                "Brutsaert, W. and Nieber, J.L. (1977). Water Resources Research, 13(3), 637-643",
                "Tallaksen, L.M. (1995). Journal of Hydrology, 165(1-4), 349-370",
                "Nathan, R.J. and McMahon, T.A. (1990). Water Resources Research, 26(7), 1465-1473"
            ],
            parameters={
                "hydrosphere.runoff.baseflow_recession_constant": (
                    "Baseflow recession constant for groundwater discharge [s-1]"
                ),
                "hydrosphere.runoff.topographic_index_mean": (
                    "Mean topographic index ln(a/tan(beta)) of the watershed [dimensionless]"
                ),
                "hydrosphere.runoff.transmissivity": (
                    "Soil transmissivity for lateral subsurface flow [m2 s-1]"
                ),
                "hydrosphere.runoff.threshold_moisture": (
                    "Threshold soil moisture for interflow generation (field capacity) [m3 m-3]"
                ),
                "hydrosphere.runoff.return_flow_fraction": (
                    "Fraction of subsurface flow that returns as baseflow [dimensionless]"
                ),
                "hydrosphere.runoff.interflow_scaling_factor": (
                    "Scaling factor converting return flow fraction to daily interflow coefficient "
                    "[dimensionless]"
                ),
                "hydrosphere.runoff.storage_capacity": (
                    "Maximum groundwater storage capacity for subsurface runoff model [mm]"
                ),
                "hydrosphere.runoff.baseflow_method": (
                    "Baseflow calculation method (exponential, topmodel, or linear_reservoir) "
                    "[dimensionless]"
                ),
                "hydrosphere.runoff.topmodel_max_deficit": (
                    "Maximum storage deficit used in simplified TOPMODEL baseflow [mm]"
                ),
                "hydrosphere.runoff.topmodel_m_scaling_factor": (
                    "Scaling factor mapping max storage deficit to TOPMODEL m parameter "
                    "[dimensionless]"
                ),
                "hydrosphere.runoff.interflow_exponent": (
                    "Exponent controlling interflow sensitivity to excess soil moisture "
                    "[dimensionless]"
                ),
                "hydrosphere.runoff.interflow_rate_scaling_factor": (
                    "Multiplier of the moisture-gated interflow recession coefficient "
                    "[dimensionless]"
                ),
                "hydrosphere.runoff.topmodel_moisture_reference": (
                    "Reference soil moisture for TOPMODEL moisture scaling [m3 m-3]"
                ),
                "hydrosphere.runoff.topmodel_moisture_factor_min": (
                    "Minimum moisture scaling factor for TOPMODEL baseflow [dimensionless]"
                ),
                "hydrosphere.runoff.topmodel_moisture_factor_max": (
                    "Maximum moisture scaling factor for TOPMODEL baseflow [dimensionless]"
                ),
                "hydrosphere.runoff.water_table_depth_scale": (
                    "Diagnostic scaling factor for groundwater table depth estimation [m]"
                ),
                "hydrosphere.runoff.water_table_depth_max": (
                    "Maximum diagnostic groundwater table depth [m]"
                ),
                "hydrosphere.numerical.minimum_storage_outflow": (
                    "Minimum storage outflow used to avoid division by zero in storage scaling [mm]"
                ),
                "hydrosphere.numerical.recession_coefficient_minimum": (
                    "Minimum recession coefficient used to avoid division by zero [s-1]"
                ),
            },
            input_variables=["soil_moisture", "drainage_flux", "groundwater_storage"],
            output_variables=[
                "subsurface_runoff_flux",
                "groundwater_storage",
                "diagnostics.subsurface_runoff.baseflow_flux",
                "diagnostics.subsurface_runoff.interflow_flux",
                "diagnostics.subsurface_runoff.storage_fraction",
                "diagnostics.subsurface_runoff.water_table_depth",
                "diagnostics.subsurface_runoff.recession_time_constant_days",
                "diagnostics.subsurface_runoff.drainage_rate_mm_s",
                "diagnostics.subsurface_runoff.total_runoff_rate_mm_s",
                "diagnostics.subsurface_runoff.storage_capacity_overflow",
                "diagnostics.subsurface_runoff.outflow_limiter_residual",
                "diagnostics.subsurface_runoff.water_table_depth_clipping_residual",
                "diagnostics.subsurface_runoff.water_balance_residual",
            ],
            conservation_laws=["mass"],
            spatial_dimensions=0,  # Point-based model
            temporal_scale=TemporalScale.SECOND,
            coupling_interfaces=["soil_moisture", "surface_runoff", "channel_routing"],
            spatial_scale=SpatialScale.PATCH,
            variable_kind=VariableKind.STATE,
        )
        
        # Initialize parent classes
        super().__init__(name=name, config=config, metadata=metadata)
        self.optional_input_variables = set()
        
        # Set TransportMechanism attributes.
        # References: Beven and Kirkby (1979), Hydrological Sciences Bulletin 24(1), 43-69;
        #   Tallaksen (1995), Journal of Hydrology 165, 349-370 (recession analysis).
        # Mathematical expression: TOPMODEL q = T_0 tan(beta) exp(-D/m) and the linear
        #   reservoir Q = S / k are storage-discharge laws of lateral Darcy flow in the
        #   kinematic-wave approximation (hydraulic gradient = local slope tan(beta)).
        # Parameter meanings: T_0 lateral transmissivity, beta local slope, D storage
        #   deficit, m decay parameter, S storage, k recession time constant.
        # Units: q [m2 s-1] per unit contour; T_0 [m2 s-1]; D, m [m]; S [m]; k [s].
        # Domain/constraints: the law has no diffusion term, so transport_type is
        #   advection and diffusivity is None; no tortuosity or porosity enters it.
        # Sign convention: positive discharge leaves the soil column laterally.
        self.peclet_number = None
        self.velocity_field = None
        
        # Get parameters from unified parameter manager with full namespace
        # Recession coefficient is stored as s-1
        recession_coeff_s = self.get_parameter_value(
            "baseflow_recession_constant",
            namespace="hydrosphere.runoff", required=True)
        self.topographic_index = self.get_parameter_value(
            "topographic_index_mean",
            namespace="hydrosphere.runoff", required=True)
        self.transmissivity = self.get_parameter_value(
            "transmissivity",
            namespace="hydrosphere.runoff", required=True)
        # Threshold moisture for interflow: use a typical field capacity value
        self.threshold_moisture = self.get_parameter_value(
            "threshold_moisture",
            namespace="hydrosphere.runoff", required=True)
        # Interflow coefficient: derived at use from the return flow fraction
        # (see the interflow_coefficient and interflow_coeff_s properties). The
        # learnable return-flow binding is held by reference, so the manager
        # stays its one owner and a checkpoint saves it in parameter_bindings.
        self.return_flow_fraction = self.get_parameter_value(
            "return_flow_fraction",
            namespace="hydrosphere.runoff", required=True)
        self.interflow_scaling_factor = self.get_parameter_value(
            "interflow_scaling_factor",
            namespace="hydrosphere.runoff", required=True)
        # Storage capacity: use reasonable value based on soil depth
        self.storage_capacity = self.get_parameter_value(
            "storage_capacity",
            namespace="hydrosphere.runoff", required=True)
        # Physical constants are resolved only through the physical SSOT API.
        self.water_density = float(get_physical_constant("water_density"))
        # Baseflow method: read from config or use exponential as default
        self.baseflow_method = self.get_parameter_value(
            "baseflow_method",
            namespace="hydrosphere.runoff", required=True)
        self.topmodel_max_deficit = self.get_parameter_value(
            "topmodel_max_deficit",
            namespace="hydrosphere.runoff", required=True)
        self.topmodel_m_scaling_factor = self.get_parameter_value(
            "topmodel_m_scaling_factor",
            namespace="hydrosphere.runoff", required=True)
        self.interflow_exponent = self.get_parameter_value(
            "interflow_exponent",
            namespace="hydrosphere.runoff", required=True)
        self.interflow_rate_scaling_factor = self.get_parameter_value(
            "interflow_rate_scaling_factor",
            namespace="hydrosphere.runoff", required=True)
        self.topmodel_moisture_reference = self.get_parameter_value(
            "topmodel_moisture_reference",
            namespace="hydrosphere.runoff", required=True)
        self.topmodel_moisture_factor_min = self.get_parameter_value(
            "topmodel_moisture_factor_min",
            namespace="hydrosphere.runoff", required=True)
        self.topmodel_moisture_factor_max = self.get_parameter_value(
            "topmodel_moisture_factor_max",
            namespace="hydrosphere.runoff", required=True)
        self.water_table_depth_scale = self.get_parameter_value(
            "water_table_depth_scale",
            namespace="hydrosphere.runoff", required=True)
        self.water_table_depth_max = self.get_parameter_value(
            "water_table_depth_max",
            namespace="hydrosphere.runoff", required=True)
        self.minimum_storage_outflow = self.get_parameter_value(
            "minimum_storage_outflow",
            namespace="hydrosphere.numerical", required=True)
        self.recession_coefficient_minimum = self.get_parameter_value(
            "recession_coefficient_minimum",
            namespace="hydrosphere.numerical", required=True)
        self.unit_converter = unit_converter
        self.unit_conversion_context = ConversionContext(
            water_density=float(self._to_scalar(self.water_density))
        )

        # Compute derived parameters
        self.time_conversion = float(get_physical_constant("seconds_per_day"))
        # Store recession coefficient already in s-1
        self.recession_coefficient = recession_coeff_s
        self.recession_coeff_s = recession_coeff_s  # Already in s-1
        # The interflow coefficient is a property derived from the held
        # return-flow binding at every read; nothing derived from a learnable
        # binding is stored on the instance.

        # Validate parameters
        self._validate_parameters()

    @property
    def interflow_coefficient(self) -> Tensor:
        """Interflow coefficient per day [d-1] from the current return-flow binding."""
        # References: Beven and Kirkby (1979), Hydrol. Sci. Bull. 24(1), 43-69;
        #   Brutsaert and Nieber (1977), Water Resour. Res. 13(3), 637-643
        #   (storage-discharge recession of a lateral subsurface store).
        # Mathematical expression: k_d = f_r * c_s.
        # Parameter meanings: f_r is the return-flow fraction
        #   (hydrosphere.runoff.return_flow_fraction), c_s the interflow scaling
        #   factor (hydrosphere.runoff.interflow_scaling_factor), k_d the
        #   interflow coefficient per day.
        # Units: f_r and c_s [1]; k_d [d-1].
        # Domain/constraints: k_d > 0 (checked at construction by
        #   _validate_parameters). f_r is a learnable binding held by reference,
        #   so k_d is derived at every read and never stored: it follows a
        #   changed f_r, and no autograd graph is carried from one call into
        #   the next.
        # Sign convention: k_d is a nonnegative rate coefficient of an outward
        #   lateral export.
        return self.return_flow_fraction * self.interflow_scaling_factor

    @property
    def interflow_coeff_s(self) -> Tensor:
        """Interflow coefficient per second [s-1] from the current return-flow binding."""
        # References: BIPM (2019), SI Brochure, 9th edition (day as a non-SI
        #   unit accepted for use with the SI); the interflow law of
        #   _compute_interflow.
        # Mathematical expression: k_i = k_d / s_day.
        # Parameter meanings: k_d is the interflow coefficient per day
        #   (interflow_coefficient), s_day the seconds per day of the physical
        #   constant SSOT, k_i the coefficient of Q_i = k_i c_i max(theta -
        #   theta_t, 0)^n S.
        # Units: k_d [d-1]; s_day [s d-1]; k_i [s-1].
        # Domain/constraints: s_day > 0; derived at every read like k_d, never
        #   stored.
        # Sign convention: k_i is a nonnegative rate coefficient of an outward
        #   lateral export.
        return self.interflow_coefficient / self.time_conversion
        
    def _to_scalar(self, value: Any) -> float:
        """Convert tensor-like values to float for validation checks."""
        if isinstance(value, Tensor):
            detached = value.detach().reshape(-1)
            values = detached.tolist()
            if not values:
                raise ValueError("Expected non-empty tensor value for scalar conversion.")
            if len(values) == 1:
                return float(values[0])
            return float(value.detach().mean().reshape(-1).tolist()[0])
        try:
            return float(value)
        except (TypeError, ValueError):
            tensor_value = torch.as_tensor(value).detach().reshape(-1)
            values = tensor_value.tolist()
            if not values:
                raise ValueError("Expected non-empty value for scalar conversion.")
            return float(values[0])

    def _validate_parameters(self) -> None:
        """Validate parameter ranges and consistency."""
        recession_coeff = self._to_scalar(self.recession_coefficient)
        transmissivity = self._to_scalar(self.transmissivity)
        threshold_moisture = self._to_scalar(self.threshold_moisture)
        interflow_coeff = self._to_scalar(self.interflow_coefficient)
        storage_capacity = self._to_scalar(self.storage_capacity)

        if recession_coeff <= 0:
            raise ValueError("Recession coefficient must be positive")
        
        if transmissivity <= 0:
            raise ValueError("Transmissivity must be positive")
        
        if not (0.0 <= threshold_moisture <= 1.0):
            raise ValueError("Threshold moisture must be in range [0, 1]")
        
        if interflow_coeff <= 0:
            raise ValueError("Interflow coefficient must be positive")
        
        valid_methods = ["exponential", "topmodel", "linear_reservoir"]
        if self.baseflow_method not in valid_methods:
            raise ValueError(f"Baseflow method must be one of {valid_methods}")
        
        if storage_capacity <= 0:
            raise ValueError("Storage capacity must be positive")
    
    def get_dependencies(self) -> DependencyDeclaration:
        """
        Return candidate upstream soil-water closures.
        
        Returns
        -------
        DependencyDeclaration
            Optional noncompeting providers of both soil moisture and drainage.

        Notes
        -----
        The three providers are compile-time alternatives, so none can be an
        unconditional required functor. External adapters may also inject all
        three canonical state inputs. GroundwaterDynamics is deliberately not
        a dependency because it is an alternative full-state owner of
        groundwater_storage.
        """
        return self._merge_declared_dependencies({
            "required": [],
            "optional": [
                "bucket_soil_moisture",
                "richards_soil_moisture",
                "vertical_percolation_transport",
            ],
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
        """Declare canonical cochain spaces for subsurface runoff.

        References:
          - Beven and Kirkby (1979), TOPMODEL runoff generation.
          - Brutsaert and Nieber (1977), baseflow recession analysis.
          - Hirani (2003), Discrete Exterior Calculus.
        Mathematical expression:
          S_dot = I_drain - D_h q_total,
          q_total = star_T g_h + q_interflow + q_capacity.
        Parameter meanings:
          S is groundwater storage [mm], I_drain is drainage input
          [kg m-2 s-1], and q_total is the unique lateral runoff export,
          including routed capacity overflow [kg m-2 s-1].
        """

        n_dofs = self._structural_n_dofs()
        return (
            CochainSpaceSpec(
                complex_id="hydrosphere_subsurface_runoff",
                degree=0,
                primality="primal",
                n_dofs=n_dofs,
                units="mm",
                extensivity="intensive",
                physical_domain="hydrosphere",
                semantic_name="groundwater_storage",
                orientation_convention="unsigned",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="hydrosphere_subsurface_runoff",
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
                complex_id="hydrosphere_subsurface_runoff",
                degree=2,
                primality="dual",
                n_dofs=n_dofs,
                units="kg m-2 s-1",
                extensivity="integral",
                physical_domain="hydrosphere",
                semantic_name="subsurface_runoff_flux",
                orientation_convention="outward_positive",
                boundary_role="port",
            ),
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Declare ParameterManager keys used by runtime and structural builders.

        Physical constants are resolved through get_physical_constant() and are
        intentionally excluded from the ParameterManager dependency contract.
        """

        return (
            "hydrosphere.runoff.baseflow_recession_constant",
            "hydrosphere.runoff.topographic_index_mean",
            "hydrosphere.runoff.transmissivity",
            "hydrosphere.runoff.threshold_moisture",
            "hydrosphere.runoff.return_flow_fraction",
            "hydrosphere.runoff.interflow_scaling_factor",
            "hydrosphere.runoff.storage_capacity",
            "hydrosphere.runoff.baseflow_method",
            "hydrosphere.runoff.topmodel_max_deficit",
            "hydrosphere.runoff.topmodel_m_scaling_factor",
            "hydrosphere.runoff.interflow_exponent",
            "hydrosphere.runoff.interflow_rate_scaling_factor",
            "hydrosphere.runoff.topmodel_moisture_reference",
            "hydrosphere.runoff.topmodel_moisture_factor_min",
            "hydrosphere.runoff.topmodel_moisture_factor_max",
            "hydrosphere.runoff.water_table_depth_scale",
            "hydrosphere.runoff.water_table_depth_max",
            "hydrosphere.numerical.minimum_storage_outflow",
            "hydrosphere.numerical.recession_coefficient_minimum",
        )

    def build_hodge_contract(
        self,
        n_dofs: Optional[int] = None,
        *,
        strict: bool = True,
    ) -> HodgeStarSpec:
        """Build the subsurface mobility Hodge contract.

        References:
          - Darcy (1856), groundwater flow law.
          - Beven and Kirkby (1979), TOPMODEL baseflow scale Q_0 = T_0 exp(-lambda).
          - van der Schaft and Maschke (1995), dissipative port systems.
        Mathematical expression:
          star_T = diag(rho_w * T_0 * exp(-lambda)), with rho_w > 0, T_0 > 0.
        Parameter meanings:
          rho_w is liquid water density; T_0 is the saturated lateral
          transmissivity; lambda is the mean topographic index ln(a/tan(beta))
          with the upslope area per unit contour length a in metres.
        Units:
          rho_w [kg m-3]; T_0 [m2 s-1]; exp(-lambda) carries [m-1] because a
          is in metres; star_T [kg m-2 s-1] maps the dimensionless lateral
          gradient to the declared mass-flux codomain.
        Domain/constraints:
          strictly positive diagonal; the same T_0 exp(-lambda) scale is the
          runtime TOPMODEL baseflow coefficient, so this nominal Hodge is
          dimensionally consistent with the declared codomain.
        Sign convention:
          outward-positive lateral export; passivity follows from positivity.
        """

        n_dofs = self._structural_n_dofs(n_dofs=n_dofs)
        mobility_value = (
            self._to_scalar(self.water_density)
            * self._to_scalar(self.transmissivity)
            * math.exp(-self._to_scalar(self.topographic_index))
        )
        values = torch.full(
            (n_dofs,),
            mobility_value,
            dtype=torch.float32,
            device=getattr(self, "device", torch.device("cpu")),
        )
        if bool(torch.any(values <= 0.0)):
            raise ValueError("Subsurface runoff mobility must be positive.")

        domain = CochainSpaceSpec(
            complex_id="hydrosphere_subsurface_runoff",
            degree=1,
            primality="primal",
            n_dofs=n_dofs,
            units="m m-1",
            extensivity="intensive",
            physical_domain="hydrosphere",
            semantic_name="subsurface_hydraulic_gradient",
            orientation_convention="down_gradient_positive",
        )
        codomain = CochainSpaceSpec(
            complex_id="hydrosphere_subsurface_runoff",
            degree=2,
            primality="dual",
            n_dofs=n_dofs,
            units="kg m-2 s-1",
            extensivity="integral",
            physical_domain="hydrosphere",
            semantic_name="subsurface_runoff_flux",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        hodge = HodgeStarSpec(
            name=f"{self.name}_subsurface_runoff_hodge",
            domain=domain,
            codomain=codomain,
            matrix=torch.diag(values),
            parameter_keys=(
                "hydrosphere.runoff.transmissivity",
                "hydrosphere.runoff.topographic_index_mean",
            ),
            metric_keys=("hydrosphere.runoff.topographic_index_mean",),
            material_law="topmodel_linear_reservoir_subsurface_routing",
        )
        if strict:
            hodge.validate(passive=True)
        return hodge

    def build_hodge_terms(
        self,
        complex_bundle: Any = None,
        resolved_parameters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[HodgeStarSpec, ...]:
        """Build subsurface runoff Hodge terms for IPHS assembly."""

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
        """Return no reversible interconnection for this dissipative storage loss."""

        return None

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the positive subsurface runoff mobility contribution.

        References:
          - Darcy (1856), Brutsaert and Nieber (1977), and van der Schaft and Maschke (1995).
        Mathematical expression:
          M_h = diag(rho_w * T_0 * exp(-lambda)), sigma = X_h.T M_h X_h >= 0.
        Parameter meanings:
          T_0 is transmissivity [m2 s-1]; lambda is the mean topographic index
          with a in metres; rho_w is water density [kg m-3]; M_h [kg m-2 s-1].
        """

        hodge = self.build_hodge_contract(
            n_dofs=self._structural_n_dofs(complex_bundle=complex_bundle),
            strict=True,
        )
        matrix = hodge.matrix
        return {
            "state_vars": tuple(
                f"subsurface_storage_potential[{idx}]" for idx in range(int(matrix.shape[0]))
            ),
            "J": torch.zeros_like(matrix),
            "M": matrix,
            "ports": self.ports(),
        }

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Declare the lateral subsurface matter port.

        References:
          - Beven and Kirkby (1979), lateral saturated contributing area flow.
          - van der Schaft and Maschke (1995), port variables.
        Mathematical expression:
          P_port = e_h * f_h, where f_h is subsurface_runoff_flux.
        Parameter meanings:
          e_h is hydraulic head potential; f_h is lateral water mass flux.
        """

        effort_space = CochainSpaceSpec(
            complex_id="hydrosphere_subsurface_runoff",
            degree=0,
            primality="primal",
            n_dofs=1,
            units="m",
            extensivity="intensive",
            physical_domain="hydrosphere",
            semantic_name="subsurface_hydraulic_head",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        flow_space = CochainSpaceSpec(
            complex_id="hydrosphere_subsurface_runoff",
            degree=2,
            primality="dual",
            n_dofs=1,
            units="kg m-2 s-1",
            extensivity="integral",
            physical_domain="hydrosphere",
            semantic_name="subsurface_runoff_mass_flux",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        return (
            PortSpecIPHS(
                name="subsurface_runoff_boundary_flux",
                effort_var="subsurface_hydraulic_head",
                flow_var="subsurface_runoff_flux",
                effort_space=effort_space,
                flow_space=flow_space,
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
    
    def forward(self,
        # References:
        #   Beven and Kirkby (1979); Brutsaert and Nieber (1977); Tallaksen (1995).
        # Mathematical expression:
        #   S* = min(S0 + R dt, Smax); O = max(S0 + R dt - Smax, 0);
        #   Qb = selected baseflow law (exponential, TOPMODEL T0 exp(-lambda) exp(-D/m),
        #   or linear reservoir); Qi = k_i c_i max(theta - theta_t, 0)^n S*;
        #   Vout = min((Qb + Qi) dt, S*); Qout = (Vout + O)/dt; S1 = S* - Vout.
        # Parameter meanings:
        #   S0/S1 storage before/after the step; R converted recharge; Smax capacity;
        #   D = Smax - S* the deficit; m the TOPMODEL scale; lambda the catchment-mean
        #   ln(a/tan beta) with a in metres; O the capacity overflow routed to the same
        #   lateral outlet.
        # Units:
        #   S0, S*, S1, O, Vout [mm]; R, Qb, Qi, Qout [mm s-1]; T0 [m2 s-1];
        #   exp(-lambda) [m-1]; exported flux F_out [kg m-2 s-1] = Qout / 1000 * rho_w.
        # Domain/constraints:
        #   Finite non-negative inputs and positive dt; the open ledger
        #   S1 - S0 - R dt + Qout dt is routed as water_balance_residual [mm] and
        #   audited by check_conservation_laws as a relative closure.
        # Sign convention:
        #   Recharge is inward; subsurface runoff is outward and decreases storage.
                state: StateDict,
                time: float,
                dt: float,
                **kwargs) -> FluxDict:
        """
        Compute subsurface runoff generation.

        Parameters
        ----------
        state : StateDict
            Current system state containing soil moisture and drainage inputs
        time : float
            Current simulation time [s]
        dt : float
            Time step [s]
        **kwargs
            Additional context variables

        Returns
        -------
        FluxDict
            Subsurface runoff flux and diagnostic variables

        Raises
        ------
        ValueError
            If required state variables are missing or invalid
        """
        # References: Beven and Kirkby (1979); Brutsaert and Nieber (1977).
        # Mathematical expression: S1 = S0 + R * dt - Q_total * dt.
        # Parameter meanings: S is groundwater storage [mm], R is drainage_flux
        # [kg m-2 s-1] converted to [mm s-1], Q is runoff [mm s-1], and
        # Q_total includes regular baseflow/interflow and routed capacity
        # overflow [mm s-1].
        # Units: storage and residuals [mm], fluxes [kg m-2 s-1].
        # Open route: drainage_flux enters the groundwater control volume and
        # subsurface_runoff_flux is the unique outlet and includes capacity
        # overflow; the overflow diagnostic reports its routed component.
        required_inputs = ("soil_moisture", "drainage_flux", "groundwater_storage")
        for key in required_inputs:
            if key not in state:
                raise ValueError(f"Required state variable missing: {key}")

        soil_moisture = self._ensure_tensor(state["soil_moisture"])
        drainage_flux = self._ensure_tensor(state["drainage_flux"])
        current_storage = self._ensure_tensor(state["groundwater_storage"])

        if not bool(torch.all(torch.isfinite(soil_moisture))):
            raise ValueError("soil_moisture must contain only finite values")
        if bool(torch.any(soil_moisture < 0.0)) or bool(torch.any(soil_moisture > 1.0)):
            raise ValueError("soil_moisture must be in range [0, 1]")
        if not bool(torch.all(torch.isfinite(drainage_flux))):
            raise ValueError("drainage_flux must contain only finite values")
        if bool(torch.any(drainage_flux < 0.0)):
            raise ValueError("drainage_flux must be non-negative")
        if not bool(torch.all(torch.isfinite(current_storage))):
            raise ValueError("groundwater_storage must contain only finite values")
        if bool(torch.any(current_storage < 0.0)):
            raise ValueError("groundwater_storage must be non-negative")
        dt_tensor = torch.as_tensor(dt)
        if not bool(torch.isfinite(dt_tensor).all()) or bool(dt_tensor <= 0):
            raise ValueError("Time step must be finite and positive")

        # References: CF Standard Name Table v94 (liquid water equivalent).
        # Mathematical expression: rate_mm_s = (F / rho_w) * 1000, evaluated by the single
        # UnitConverter liquid-water-equivalent rule selected by the declared name.
        # Parameter meanings: F drainage flux [kg m-2 s-1], rho_w = water.density_liquid [kg m-3].
        converter = self.unit_converter
        drainage_mm_s = converter.convert(drainage_flux, "drainage_flux", "kg m-2 s-1", "mm s-1")

        # 1. Update groundwater storage with drainage input
        storage_input = drainage_mm_s * dt  # mm
        new_storage_unbounded = current_storage + storage_input
        
        # Apply storage capacity constraint
        storage_capacity = self._ensure_tensor(self.storage_capacity).to(
            device=new_storage_unbounded.device, dtype=new_storage_unbounded.dtype
        )
        min_storage = torch.zeros_like(storage_capacity)
        storage_capacity_overflow = torch.clamp(
            new_storage_unbounded - storage_capacity,
            min=0.0,
        )
        new_storage = torch.clamp(
            new_storage_unbounded,
            min=min_storage,
            max=storage_capacity,
        )
        
        # 2. Compute baseflow based on method
        if self.baseflow_method == "exponential":
            baseflow_flux = self._compute_exponential_baseflow(new_storage, dt)
        elif self.baseflow_method == "topmodel":
            baseflow_flux = self._compute_topmodel_baseflow(new_storage, soil_moisture, dt)
        elif self.baseflow_method == "linear_reservoir":
            baseflow_flux = self._compute_linear_reservoir_baseflow(new_storage, dt)
        else:
            raise ValueError(f"Unknown baseflow method: {self.baseflow_method}")
        
        # 3. Compute interflow as a moisture-gated drainage of the projected store
        interflow_flux = self._compute_interflow(soil_moisture, new_storage, dt)
        
        # 4. Total subsurface runoff
        total_subsurface_flux = baseflow_flux + interflow_flux  # mm s-1
        
        # References: linear reservoir continuity and finite-volume flux limiting.
        # Mathematical expression: Q_actual * dt = min(Q_candidate * dt, S_available).
        # Parameter meanings: Q is total runoff [mm s-1], dt is [s], and
        # S_available is the projected groundwater storage [mm].
        # Units: all storage terms are [mm].
        # Open route: rejected candidate outflow is an observation diagnostic;
        # only the limited actual runoff crosses the boundary port.
        candidate_storage_outflow = total_subsurface_flux * dt
        actual_storage_outflow = torch.minimum(candidate_storage_outflow, new_storage)
        outflow_limiter_residual = torch.clamp(
            candidate_storage_outflow - actual_storage_outflow,
            min=0.0,
        )
        minimum_storage_outflow = self._ensure_tensor(self.minimum_storage_outflow).to(
            device=new_storage.device, dtype=new_storage.dtype
        )
        flux_reduction = torch.where(
            candidate_storage_outflow > 0.0,
            actual_storage_outflow
            / torch.clamp(candidate_storage_outflow, min=minimum_storage_outflow),
            torch.ones_like(candidate_storage_outflow),
        )
        baseflow_flux = baseflow_flux * flux_reduction
        interflow_flux = interflow_flux * flux_reduction
        regular_runoff_rate = actual_storage_outflow / dt
        capacity_overflow_rate = storage_capacity_overflow / dt
        total_subsurface_flux = regular_runoff_rate + capacity_overflow_rate
        final_storage = new_storage - actual_storage_outflow

        # References: CF-1.8 unit conversion for water depth rates.
        # Mathematical expression: F = (rate_mm_s / 1000) * rho_w.
        # Parameter meanings: rate_mm_s [mm s-1], rho_w water density [kg m-3].
        subsurface_runoff_flux = converter.convert(
            total_subsurface_flux, "subsurface_runoff_flux", "mm s-1", "kg m-2 s-1"
        )
        baseflow_mass_flux = converter.convert(baseflow_flux, "baseflow_flux", "mm s-1", "kg m-2 s-1")
        interflow_mass_flux = converter.convert(interflow_flux, "interflow_flux", "mm s-1", "kg m-2 s-1")
        
        # 6. Compute diagnostic variables
        storage_fraction = final_storage / storage_capacity
        
        # Recession time constant
        # Ensure recession coefficient is clamped properly
        recession_coeff = self._ensure_tensor(self.recession_coefficient)
        recession_coeff_minimum = self._ensure_tensor(self.recession_coefficient_minimum).to(
            device=recession_coeff.device, dtype=recession_coeff.dtype
        )
        recession_time_seconds = 1.0 / torch.clamp(
            recession_coeff, min=recession_coeff_minimum
        )
        seconds_per_day = self._ensure_tensor(self.time_conversion).to(
            device=recession_time_seconds.device, dtype=recession_time_seconds.dtype
        )
        recession_time_days = recession_time_seconds / seconds_per_day
        
        # Groundwater table depth approximation (simplified)
        water_table_depth_scale = self._ensure_tensor(self.water_table_depth_scale).to(
            device=storage_fraction.device, dtype=storage_fraction.dtype
        )
        water_table_depth_max = self._ensure_tensor(self.water_table_depth_max).to(
            device=storage_fraction.device, dtype=storage_fraction.dtype
        )
        water_table_depth_unbounded = water_table_depth_scale * (1.0 - storage_fraction)
        water_table_depth_clipping_residual = (
            torch.clamp(-water_table_depth_unbounded, min=0.0)
            + torch.clamp(water_table_depth_unbounded - water_table_depth_max, min=0.0)
        )
        water_table_depth = torch.clamp(
            water_table_depth_unbounded,
            min=0.0, max=water_table_depth_max
        )

        water_balance_residual = (
            final_storage
            - current_storage
            - storage_input
            + total_subsurface_flux * dt
        )

        # Every retained output has explicit runtime semantics and provenance.
        return {
            "subsurface_runoff_flux": UpdateItem(
                value=subsurface_runoff_flux,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="kg m-2 s-1",
                source_functor=self.name,
            ),
            "groundwater_storage": UpdateItem(
                value=final_storage,
                mode=UpdateMode.REPLACE,
                per_timestep=True,
                unit="mm",
                source_functor=self.name,
            ),
            "diagnostics.subsurface_runoff.baseflow_flux": UpdateItem(
                value=baseflow_mass_flux,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="kg m-2 s-1",
                source_functor=self.name,
            ),
            "diagnostics.subsurface_runoff.interflow_flux": UpdateItem(
                value=interflow_mass_flux,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="kg m-2 s-1",
                source_functor=self.name,
            ),
            "diagnostics.subsurface_runoff.storage_fraction": UpdateItem(
                value=storage_fraction,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="dimensionless",
                source_functor=self.name,
            ),
            "diagnostics.subsurface_runoff.water_table_depth": UpdateItem(
                value=water_table_depth,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="m",
                source_functor=self.name,
            ),
            "diagnostics.subsurface_runoff.recession_time_constant_days": UpdateItem(
                value=recession_time_days,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="d",
                source_functor=self.name,
            ),
            "diagnostics.subsurface_runoff.drainage_rate_mm_s": UpdateItem(
                value=drainage_mm_s,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="mm s-1",
                source_functor=self.name,
            ),
            "diagnostics.subsurface_runoff.total_runoff_rate_mm_s": UpdateItem(
                value=total_subsurface_flux,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="mm s-1",
                source_functor=self.name,
            ),
            "diagnostics.subsurface_runoff.storage_capacity_overflow": UpdateItem(
                value=storage_capacity_overflow,
                mode=UpdateMode.REPLACE,
                per_timestep=True,
                unit="mm",
                source_functor=self.name,
            ),
            "diagnostics.subsurface_runoff.outflow_limiter_residual": UpdateItem(
                value=outflow_limiter_residual,
                mode=UpdateMode.REPLACE,
                per_timestep=True,
                unit="mm",
                source_functor=self.name,
            ),
            "diagnostics.subsurface_runoff.water_table_depth_clipping_residual": UpdateItem(
                value=water_table_depth_clipping_residual,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="m",
                source_functor=self.name,
            ),
            "diagnostics.subsurface_runoff.water_balance_residual": UpdateItem(
                value=water_balance_residual,
                mode=UpdateMode.REPLACE,
                per_timestep=True,
                unit="mm",
                source_functor=self.name,
            ),
        }

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Evaluate the explicit open subsurface-water control volume.

        References:
          - Beven and Kirkby (1979), variable contributing-area hydrology.
          - Brutsaert and Nieber (1977), groundwater recession flow.
          - Higham (2002), Accuracy and Stability of Numerical Algorithms, relative error.
        Mathematical expression:
          epsilon_S = S1 - S0 - R * dt + Q_total * dt,
          r_S = |epsilon_S| / max(|S0|, |S1|, |R dt|, |Q_total dt|) when that scale is
          positive, and r_S = |epsilon_S| when every term is zero.
        Parameter meanings:
          S0 and S1 are groundwater storage before and after the step, R is
          incoming drainage, Q_total is lateral runoff including capacity
          overflow, dt is the time step; r_S is the relative closure reported
          to the dimensionless FunctorBase tolerance.
        Units:
          S0, S1, epsilon_S, R dt, Q_total dt [mm]; R, Q_total [mm s-1]; dt [s]; r_S [1].
        Domain/constraints:
          diagnostics.subsurface_runoff.water_balance_residual (epsilon_S) and the
          storage, drainage and runoff terms must be present and finite; the residual
          is compared in floating precision of the runtime state, so the relative form
          avoids reporting float32 rounding of an 80 mm store as a violation.
        Sign convention:
          r_S >= 0; positive epsilon_S means the store gained more than the ledger
          explains.
        Open-system route:
          Drainage enters the storage control volume and subsurface runoff
          exits through the declared matter port.
        """

        del input_state
        diagnostic_key = "diagnostics.subsurface_runoff.water_balance_residual"
        if diagnostic_key not in output_updates:
            raise KeyError(f"Missing conservation diagnostic '{diagnostic_key}'")
        residual = output_updates[diagnostic_key]
        if isinstance(residual, UpdateItem):
            residual = residual.value
        if not isinstance(residual, Tensor):
            raise TypeError("Subsurface-runoff mass residual must be a torch.Tensor")
        if residual.numel() == 0:
            raise ValueError("Subsurface-runoff mass residual must be non-empty")
        if not bool(torch.all(torch.isfinite(residual))):
            raise ValueError("Subsurface-runoff mass residual must be finite")
        scale_terms = []
        for key in (
            "groundwater_storage",
            "diagnostics.subsurface_runoff.storage_ledger_terms",
        ):
            if key in output_updates:
                term = output_updates[key]
                if isinstance(term, UpdateItem):
                    term = term.value
                if isinstance(term, Tensor):
                    scale_terms.append(torch.abs(term).to(dtype=residual.dtype, device=residual.device))
        if not scale_terms:
            raise KeyError("Subsurface-runoff conservation scale requires groundwater_storage")
        scale = scale_terms[0]
        for term in scale_terms[1:]:
            scale = torch.maximum(scale, term)
        scale = torch.max(scale)
        absolute = torch.max(torch.abs(residual))
        relative = torch.where(scale > 0.0, absolute / torch.where(scale > 0.0, scale, torch.ones_like(scale)), absolute)
        return {"mass": float(relative.detach().item())}
    

    # References: Brutsaert and Nieber (1977); Tallaksen (1995).
    # Formula: Q = alpha * S.
    # Parameters: alpha (recession_coeff_s, s-1), S (storage, mm).
    def _compute_exponential_baseflow(self, storage: Tensor, dt: float) -> Tensor:
        """
        Compute baseflow using exponential recession.
        
        Mathematical expression: Q = alpha * S
        
        Parameters
        ----------
        storage : Tensor
            Current groundwater storage [mm]
        dt : float
            Time step [s]
            
        Returns
        -------
        Tensor
            Baseflow rate [mm s-1]
        """
        # Exponential recession: dS/dt = -alpha*S, so Q = alpha*S
        recession_coeff = self._ensure_tensor(self.recession_coeff_s).to(
            device=storage.device, dtype=storage.dtype
        )
        baseflow_rate = recession_coeff * storage
        
        return baseflow_rate
    
    # References: Beven and Kirkby (1979); Beven (1997), TOPMODEL: a critique.
    # Mathematical expression: Q_b = 1000 * T_0 * exp(-lambda) * exp(-S_def/m) * f_theta,
    #   the catchment-mean TOPMODEL baseflow Q_0 exp(-S_def/m) with Q_0 = T_0 exp(-lambda),
    #   multiplied by the retained empirical moisture factor f_theta.
    # Parameter meanings: T_0 saturated lateral transmissivity; lambda mean
    #   topographic index ln(a/tan(beta)) with a in metres; S_def storage deficit;
    #   m = c_m * S_def_max deficit scale; f_theta = clip(theta/theta_ref, f_min, f_max).
    # Units: T_0 [m2 s-1]; exp(-lambda) [m-1] because a is in metres; Q_0 [m s-1];
    #   1000 converts m to mm; S_def, m [mm]; f_theta [1]; Q_b [mm s-1].
    # Domain/constraints: S_def >= 0 by clamping; m > 0; the 1/lambda form is not
    #   TOPMODEL and is not dimensionally a depth rate; point application of the
    #   catchment-mean relation, not a distributed solution.
    # Sign convention: Q_b is a nonnegative outward lateral export from storage.
    def _compute_topmodel_baseflow(self, 
                                  storage: Tensor, 
                                  soil_moisture: Tensor, 
                                  dt: float) -> Tensor:
        """
        Compute baseflow using TOPMODEL approach.
        
        Mathematical expression:
        Q_b = 1000 * T_0 * exp(-lambda) * exp(-S_def / m) * f_theta

        T_0 exp(-lambda) is the catchment-mean TOPMODEL baseflow scale Q_0
        [m s-1] because lambda = mean ln(a/tan(beta)) is evaluated with the
        upslope area per unit contour length a in metres. This is a point
        application of the catchment-mean relation with the retained empirical
        moisture factor; it is not a distributed TOPMODEL solution.
        
        Parameters
        ----------
        storage : Tensor
            Current groundwater storage [mm]
        soil_moisture : Tensor
            Soil moisture content [m3 m-3]
        dt : float
            Time step [s]
            
        Returns
        -------
        Tensor
            Baseflow rate [mm s-1]
        """
        converter = self.unit_converter
        conversion_context = self.unit_conversion_context
        # Storage deficit (simplified)
        max_deficit = self._ensure_tensor(self.topmodel_max_deficit).to(
            device=storage.device, dtype=storage.dtype
        )
        storage_deficit = torch.clamp(max_deficit - storage, min=0.0)  # Non-negative deficit
        
        # TOPMODEL scaling parameter
        m_param = self._ensure_tensor(self.topmodel_m_scaling_factor).to(
            device=storage.device, dtype=storage.dtype
        ) * max_deficit  # Scaling parameter [mm]
        
        # Transmissivity factor based on storage deficit
        transmissivity_factor = torch.exp(-storage_deficit / m_param)
        
        # Base transmissivity [mm s-1]
        transmissivity = self._ensure_tensor(self.transmissivity).to(
            device=storage.device, dtype=storage.dtype
        )
        base_transmissivity = converter.meter_to_millimeter(
            transmissivity, conversion_context
        )
        
        # Catchment-mean topographic factor exp(-lambda) [m-1]; lambda is ln(a/tan(beta))
        # with a in metres, so T_0 exp(-lambda) is the TOPMODEL Q_0 in m s-1.
        topographic_index = self._ensure_tensor(self.topographic_index).to(
            device=storage.device, dtype=storage.dtype
        )
        topographic_factor = torch.exp(-topographic_index)
        
        # TOPMODEL baseflow
        baseflow_rate = base_transmissivity * transmissivity_factor * topographic_factor
        
        # Scale by soil moisture (higher moisture increases baseflow)
        moisture_reference = self._ensure_tensor(self.topmodel_moisture_reference).to(
            device=storage.device, dtype=storage.dtype
        )
        moisture_factor_min = self._ensure_tensor(self.topmodel_moisture_factor_min).to(
            device=storage.device, dtype=storage.dtype
        )
        moisture_factor_max = self._ensure_tensor(self.topmodel_moisture_factor_max).to(
            device=storage.device, dtype=storage.dtype
        )
        moisture_factor = torch.clamp(
            soil_moisture / moisture_reference,
            min=moisture_factor_min,
            max=moisture_factor_max,
        )
        baseflow_rate = baseflow_rate * moisture_factor
        
        return baseflow_rate
    
    # References: Nathan and McMahon (1990).
    # Formula: Q = k * S.
    # Parameters: k (recession_coeff_s, s-1), S (storage, mm).
    def _compute_linear_reservoir_baseflow(self, storage: Tensor, dt: float) -> Tensor:
        """
        Compute baseflow using linear reservoir model.
        
        Mathematical expression: Q = k * S, where k is recession constant
        
        Parameters
        ----------
        storage : Tensor
            Current groundwater storage [mm]
        dt : float
            Time step [s]
            
        Returns
        -------
        Tensor
            Baseflow rate [mm s-1]
        """
        # Linear reservoir coefficient
        k_reservoir = self._ensure_tensor(self.recession_coeff_s).to(
            device=storage.device, dtype=storage.dtype
        )
        
        # Linear reservoir baseflow
        baseflow_rate = k_reservoir * storage
        
        return baseflow_rate
    
    # References: Beven and Kirkby (1979); Brutsaert and Nieber (1977).
    # Mathematical expression: Q_i = k_i * c_i * max(0, theta - theta_t)^n * S,
    #   a moisture-gated recession of the same groundwater store S that the
    #   forward pass drains, so Q_i vanishes for an empty store.
    # Parameter meanings: k_i interflow coefficient; c_i dimensionless
    #   multiplier; theta soil moisture; theta_t interflow threshold; n exponent;
    #   S groundwater storage available to the lateral export.
    # Units: k_i [s-1]; c_i, n [1]; theta, theta_t [m3 m-3]; S [mm]; Q_i [mm s-1].
    # Domain/constraints: the excess theta - theta_t is clamped at zero; S >= 0;
    #   without the store depth the s-1 coefficient alone is not a depth rate.
    # Sign convention: Q_i is a nonnegative outward lateral export from storage.
    def _compute_interflow(self, soil_moisture: Tensor, storage: Tensor, dt: float) -> Tensor:
        """
        Compute interflow as a moisture-gated drainage of the groundwater store.
        
        Mathematical expression: Q_i = k_i * c_i * max(0, theta - theta_t)^n * S
        
        Parameters
        ----------
        soil_moisture : Tensor
            Soil moisture content [m3 m-3]
        storage : Tensor
            Groundwater storage available to lateral export [mm]
        dt : float
            Time step [s]
            
        Returns
        -------
        Tensor
            Interflow rate [mm s-1]
        """
        # Excess moisture above threshold
        threshold = self._ensure_tensor(self.threshold_moisture).to(
            device=soil_moisture.device, dtype=soil_moisture.dtype
        )
        excess_moisture = torch.clamp(soil_moisture - threshold, min=0.0)  # Non-negative excess
        
        # Interflow exponent (typically 1-3)
        interflow_exponent = self._ensure_tensor(self.interflow_exponent).to(
            device=soil_moisture.device, dtype=soil_moisture.dtype
        )
        interflow_coeff = self._ensure_tensor(self.interflow_coeff_s).to(
            device=soil_moisture.device, dtype=soil_moisture.dtype
        )
        scaling_factor = self._ensure_tensor(self.interflow_rate_scaling_factor).to(
            device=soil_moisture.device, dtype=soil_moisture.dtype
        )
        
        # Interflow rate: moisture-gated recession coefficient times the store depth
        storage_depth = storage.to(device=soil_moisture.device, dtype=soil_moisture.dtype)
        interflow_rate = (
            interflow_coeff
            * scaling_factor
            * torch.pow(excess_moisture, interflow_exponent)
            * storage_depth
        )
        
        return interflow_rate
    
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
        return super()._ensure_tensor(value, dtype=torch.float32)


# Export the main class
__all__ = ["SubsurfaceRunoff"]

