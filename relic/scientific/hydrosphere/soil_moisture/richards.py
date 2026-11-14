"""
Richards Equation Soil Moisture Model for RELIC

This module provides a framework for implementing the Richards equation
for soil water flow in the RELIC system. The implementation includes
numerical discretization, boundary condition interfaces, and coupling
with the unified response function library.

References
----------
- Richards, L.A. (1931). Capillary conduction of liquids through porous mediums.
  Journal of Applied Physics, 1(5), 318-333.
- Van Genuchten, M.T. (1980). A closed-form equation for predicting the hydraulic
  conductivity of unsaturated soils. Soil Science Society of America Journal, 
  44(5), 892-898.
- Mualem, Y. (1976). A new model for predicting the hydraulic conductivity of
  unsaturated porous media. Water Resources Research, 12(3), 513-522.

Mathematical Expressions
------------------------
Richards equation (mixed form):
    dtheta/dt = d/dz[K(theta)(dh/dz - 1)] - S(theta)

Alternative forms:
    theta-based: dtheta/dt = d/dz[D(theta)dtheta/dz] + dK(theta)/dz - S(theta)
    h-based: C(h)dh/dt = d/dz[K(h)(dh/dz - 1)] - S(h)

Where:
    theta = volumetric water content [m3 m-3]
    h = pressure head [m]
    K(theta) = hydraulic conductivity function [m s-1]
    D(theta) = diffusivity function [m2 s-1]
    C(h) = specific water capacity [m-1]
    S(theta) = sink/source term [s-1]
    z = depth coordinate (positive downward) [m]

Van Genuchten-Mualem model:
    theta(h) = thetar + (thetas - thetar) / [1 + (alpha|h|)?]^m
    K(theta) = Ks * Se^l * [1 - (1 - Se^(1/m))^m]2
    Se = (theta - thetar) / (thetas - thetar)
    m = 1 - 1/n

Parameters
----------
soil_layers : int
    Number of soil layers for discretization
layer_thickness : float
    Thickness of each soil layer [m]
theta_residual : float
    Residual water content [m3 m-3]
theta_saturated : float
    Saturated water content [m3 m-3]
alpha_vg : float
    Van Genuchten alpha parameter [m-1]
n_vg : float
    Van Genuchten n parameter [dimensionless]
k_saturated : float
    Saturated hydraulic conductivity [m s-1]
"""

from typing import Any, Dict, Mapping, Optional, Union
import torch
from torch import Tensor

from relic.computation.operators.dec.forms import CochainSpaceSpec
from relic.computation.operators.dec.hodge import HodgeStarSpec
from relic.computation.operators.iphs import PortSpecIPHS
from relic.core.functor import (
    ConflictResolutionPolicy,
    DependencyDeclaration,
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    TransportMechanismFunctor,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from relic.core.abstractions import FluxDict, StateDict
from relic.core.state import UpdateItem, UpdateMode
from relic.utils.physical_constants import get_physical_constant
from relic.utils.soil_hydraulics import van_genuchten_m_parameter
from ..solvers import RichardsNumericalSolver
from ..solvers.richards_solver import harmonic_interface_conductivity, on_device_of
from ..interface_mapping import HydrosphereInterfaceMixin
from relic.computation.solvers import SolverFactory


class RichardsSoilMoisture(HydrosphereInterfaceMixin, TransportMechanismFunctor):
    """
    Richards Equation Soil Moisture Model Framework
    
    Provides a framework for implementing the Richards equation for
    unsaturated soil water flow. This implementation includes:
    - Van Genuchten-Mualem hydraulic functions
    - Flexible boundary condition interface
    - Numerical discretization framework
    - Mass conservation enforcement
    
    Uses one canonical mixed-form Richards stepping path for the Phase 8 IPHS
    runtime. Literature approximations are retained only in tests and theory
    documents, not as executable runtime branches.
    
    Units Convention (CF-1.8 consistent):
        - soil_moisture: m3 m-3
        - pressure_head: m
        - hydraulic_conductivity: m s-1
        - vertical_flux: kg m-2 s-1
        - water_flux_profile: kg m-2 s-1
        - soil_depth: m
    
    Attributes
    ----------
    soil_layers : int
        Number of discretized soil layers
    layer_thickness : float
        Thickness of each layer [m]
    theta_residual : float
        Residual water content [m3 m-3]
    theta_saturated : float
        Saturated water content [m3 m-3]
    alpha_vg : float
        Van Genuchten alpha parameter [m-1]
    n_vg : float
        Van Genuchten n parameter [dimensionless]
    """

    TRANSPORT_TYPE = "diffusion"

    PHASE8_RUNTIME_REPLACEMENT_STATUS = "canonical_iphs_runtime"
    _REQUIRED_PARAMETER_KEYS = (
        "hydrosphere.soil_moisture.layer_count",
        "hydrosphere.soil_moisture.soil_depth",
        "hydrosphere.soil_moisture.residual_water_content",
        "hydrosphere.soil_moisture.porosity",
        "hydrosphere.soil_moisture.alpha_vg",
        "hydrosphere.soil_moisture.n_vg",
        "hydrosphere.soil_moisture.saturated_hydraulic_conductivity",
        "hydrosphere.soil_moisture.tortuosity_parameter",
        "hydrosphere.physical.field_capacity_fraction",
        "hydrosphere.physical.wilting_point_fraction",
        "hydrosphere.numerical.richards_max_picard_iterations",
        "hydrosphere.numerical.richards_max_newton_iterations",
        "hydrosphere.numerical.richards_convergence_tolerance",
        "hydrosphere.numerical.richards_newton_absolute_tolerance",
        "hydrosphere.numerical.richards_newton_relative_tolerance",
        "hydrosphere.soil_moisture.richards_enable_newton_raphson",
        "hydrosphere.soil_moisture.richards_enable_mass_conservation_check",
        "hydrosphere.numerical.mass_conservation_tolerance",
        "hydrosphere.numerical.richards_cfl_coefficient",
        "hydrosphere.numerical.saturation_epsilon",
        "hydrosphere.numerical.minimum_pressure_term",
        "hydrosphere.numerical.minimum_se_term",
        "hydrosphere.numerical.minimum_term1",
        "hydrosphere.numerical.minimum_timestep",
        "hydrosphere.numerical.harmonic_mean_factor",
        "hydrosphere.numerical.tridiagonal_tolerance",
        "hydrosphere.numerical.line_search_initial_alpha",
        "hydrosphere.numerical.line_search_max_backtracks",
        "hydrosphere.numerical.line_search_armijo_constant",
        "hydrosphere.numerical.line_search_alpha_reduction_factor",
        "hydrosphere.numerical.line_search_min_alpha",
        "hydrosphere.numerical.characteristic_pressure_head",
    )
    
    def __init__(self, 
                 name: str = "richards_soil_moisture",
                 config: Optional[Dict] = None) -> None:
        """
        Initialize Richards equation soil moisture model.
        
        Parameters
        ----------
        name : str, optional
            Unique identifier for this functor instance
        config : Dict, optional
            Functor orchestration settings. Scientific and numerical values
            are resolved exclusively through ParameterManager.
        """
        if config is not None and not isinstance(config, dict):
            raise TypeError("RichardsSoilMoisture config must be a dictionary.")
        config_dict = dict(config or {})
        clean_config: Dict[str, Any] = {}
        for setting_key in (
            "dependencies",
            "strict_validation",
            "strict_validation_errors",
        ):
            if setting_key in config_dict:
                clean_config[setting_key] = config_dict.pop(setting_key)
        if config_dict:
            raise ValueError(
                "Unsupported RichardsSoilMoisture config keys: "
                + ", ".join(sorted(config_dict))
            )

        # Create metadata
        config = clean_config

        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Richards equation framework for unsaturated soil water flow",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "theta_dot = -D_z q_z + s_z; q_z = star_K (d h - g_z); star_K >= 0"
            ),
            references=[
                "Richards, L.A. (1931). Journal of Applied Physics, 1(5), 318-333",
                "Van Genuchten, M.T. (1980). Soil Science Society of America Journal, 44(5), 892-898",
                "Mualem, Y. (1976). Water Resources Research, 12(3), 513-522",
                "Feddes et al. (1978). Simulation of Field Water Use and Crop Yield",
            ],
            parameters={
                key: {
                    "description": (
                        "Canonical ParameterManager specification for "
                        f"{key.rsplit('.', 1)[-1].replace('_', ' ')}."
                    ),
                    "namespace": key.rsplit(".", 1)[0],
                    "required": True,
                }
                for key in type(self)._REQUIRED_PARAMETER_KEYS
            },
            input_variables=[
                "soil_moisture",
                "infiltration_flux",
                "evapotranspiration_flux",
                "capillary_rise_flux",
            ],
            output_variables=[
                "soil_moisture",
                "diagnostics.soil_moisture_richards.soil_moisture_change_rate",
                "diagnostics.soil_moisture_richards.water_flux_profile",
                "soil_matric_potential",
                "diagnostics.soil_moisture_richards.relative_moisture",
                "drainage_flux",
                "diagnostics.soil_moisture_richards",
            ],
            # The soil column is open through surface and drainage ports.
            # The process-specific checker reads the integrated relative
            # residual and avoids the generic mixed-unit output sum.
            conservation_laws=["mass"],
            spatial_dimensions=1,  # 1D vertical profile
            temporal_scale=TemporalScale.SECOND,
            coupling_interfaces=["evapotranspiration", "runoff", "groundwater"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
            conflict_resolution_policy=ConflictResolutionPolicy.EXCLUSIVE,
        )
        
        # Initialize base classes without passing transport-specific kwargs
        # Note: Passing unknown kwargs down to nn.Module.__init__ will raise TypeError.
        # Set transport-specific attributes explicitly after initialization instead.
        super().__init__(name=name, config=config, metadata=metadata)
        self.optional_input_variables = {"capillary_rise_flux"}
        # References: Richards (1931), Physics 1, 318-333; Celia et al. (1990), Water
        #   Resources Research 26(7), 1483-1496.
        # Mathematical expression: d(theta)/dt = d/dz [D(theta) d(theta)/dz - K(theta)] with
        #   D(theta) = K(theta) dh/dtheta.
        # Parameter meanings: theta water content, h pressure head, K hydraulic
        #   conductivity, D soil-water diffusivity.
        # Units: theta [m3 m-3]; h [m]; K [m s-1]; D [m2 s-1].
        # Domain/constraints: the law is diffusive, but D(theta) is state dependent and
        #   computed in the solver; no scalar diffusivity exists, so diffusivity is None.
        # Sign convention: z positive upward; the gravity term drains downward.

        self.peclet_number = None
        self.velocity_field = None

        def _numerical_param(
            name: str,
            *,
            cast: Optional[type] = None,
        ) -> Any:
            value = self.get_parameter_value(
                name,
                namespace="hydrosphere.numerical", required=True)
            if cast is not None:
                return cast(value)
            return value

        # Discretization parameters from ParameterManager
        self.soil_layers = int(
            self.get_parameter_value(
                "layer_count",
                namespace="hydrosphere.soil_moisture", required=True)
        )
        self.total_soil_depth = self.get_parameter_value(
            "soil_depth",
            namespace="hydrosphere.soil_moisture", required=True)
        self.theta_residual = self.get_parameter_value(
            "residual_water_content",
            namespace="hydrosphere.soil_moisture", required=True)
        # References: Clapp and Hornberger (1978); Cosby et al. (1984).
        # Mathematical expression: theta_saturated = porosity.
        # Parameter meanings: theta_saturated [m3 m-3] mapped from porosity parameter.
        self.theta_saturated = self.get_parameter_value(
            "porosity",
            namespace="hydrosphere.soil_moisture", required=True)
        self.alpha_vg = self.get_parameter_value(
            "alpha_vg",
            namespace="hydrosphere.soil_moisture", required=True)
        self.n_vg = self.get_parameter_value(
            "n_vg",
            namespace="hydrosphere.soil_moisture", required=True)
        self.k_saturated = self.get_parameter_value(
            "saturated_hydraulic_conductivity",
            namespace="hydrosphere.soil_moisture", required=True)
        self.mualem_tortuosity = self.get_parameter_value(
            "tortuosity_parameter",
            namespace="hydrosphere.soil_moisture", required=True)
        self.tortuosity = self.mualem_tortuosity

        self.water_density = get_physical_constant("water_density")
        self.gravity = get_physical_constant("gravity_acceleration")
        self.field_capacity_fraction = self.get_parameter_value(
            "field_capacity_fraction",
            namespace="hydrosphere.physical", required=True)
        self.wilting_point_fraction = self.get_parameter_value(
            "wilting_point_fraction",
            namespace="hydrosphere.physical", required=True)

        self.runtime_path = "canonical_iphs_mixed_form_richards"

        # Numerical solver parameters from ParameterManager
        self.max_picard_iterations = int(
            _numerical_param(
                "richards_max_picard_iterations",
                cast=int,
            )
        )
        self.max_newton_iterations = int(
            _numerical_param(
                "richards_max_newton_iterations",
                cast=int,
            )
        )
        self.convergence_tolerance = float(
            _numerical_param(
                "richards_convergence_tolerance",
            )
        )
        self.atol = float(
            _numerical_param(
                "richards_newton_absolute_tolerance",
            )
        )
        self.rtol = float(
            _numerical_param(
                "richards_newton_relative_tolerance",
            )
        )

        # Solver control flags from ParameterManager
        self.enable_newton_raphson = bool(
            self.get_parameter_value(
                "richards_enable_newton_raphson",
                namespace="hydrosphere.soil_moisture", required=True)
        )
        self.enable_mass_conservation_check = bool(
            self.get_parameter_value(
                "richards_enable_mass_conservation_check",
                namespace="hydrosphere.soil_moisture", required=True)
        )
        self.mass_conservation_tolerance = float(
            _numerical_param(
                "mass_conservation_tolerance",
            )
        )
        self.richards_cfl_coefficient = float(
            _numerical_param(
                "richards_cfl_coefficient",
            )
        )
        self.saturation_epsilon = float(
            _numerical_param(
                "saturation_epsilon",
            )
        )
        self.minimum_pressure_term = float(
            _numerical_param(
                "minimum_pressure_term",
            )
        )
        self.minimum_se_term = float(
            _numerical_param(
                "minimum_se_term",
            )
        )
        self.minimum_term1 = float(
            _numerical_param(
                "minimum_term1",
            )
        )
        self.minimum_timestep = float(
            _numerical_param(
                "minimum_timestep",
            )
        )
        self.harmonic_mean_factor = float(
            _numerical_param(
                "harmonic_mean_factor",
            )
        )
        self.tridiagonal_tolerance = float(
            _numerical_param(
                "tridiagonal_tolerance",
            )
        )
        self.line_search_initial_alpha = float(
            _numerical_param(
                "line_search_initial_alpha",
            )
        )
        self.line_search_max_backtracks = int(
            _numerical_param(
                "line_search_max_backtracks",
                cast=int,
            )
        )
        self.line_search_armijo_constant = float(
            _numerical_param(
                "line_search_armijo_constant",
            )
        )
        self.line_search_alpha_reduction = float(
            _numerical_param(
                "line_search_alpha_reduction_factor",
            )
        )
        self.line_search_min_alpha = float(
            _numerical_param(
                "line_search_min_alpha",
            )
        )
        self.characteristic_pressure_head = float(
            _numerical_param(
                "characteristic_pressure_head",
            )
        )
        
        # Derived parameters: layer_thickness and m_vg are properties computed
        # from the held bindings at every read; nothing derived from a learnable
        # binding is stored on the instance.
        
        # Boundary conditions storage
        self._top_boundary: Optional[Dict] = None
        self._bottom_boundary: Optional[Dict] = None
        
        # Initialize layer properties
        self._initialize_layers()
        
        # Validate parameters
        self._validate_parameters()

        # Create unified Thomas algorithm solver for tridiagonal systems
        # Used in Picard iteration and Newton-Raphson linear system solving
        self.thomas_solver = SolverFactory.create(
            solver_name="thomas_algorithm",
            name=f"{name}_tridiag_solver",
            config={
                "check_stability": True,
                "rtol": self.rtol,
                "atol": self.atol
            }
        )

        # Delegate solver routines to a dedicated numerical solver helper.
        self._numerical_solver = RichardsNumericalSolver(self)

    @property
    def layer_thickness(self) -> Tensor:
        """Uniform layer thickness [m] from the current soil-depth binding."""
        # References: Celia et al. (1990), Water Resources Research 26(7),
        #   1483-1496 (uniform node spacing of the mixed-form discretization).
        # Mathematical expression: dz = L / N.
        # Parameter meanings: L is the profile depth
        #   (hydrosphere.soil_moisture.soil_depth), N the number of layers
        #   (hydrosphere.soil_moisture.layer_count), dz the layer thickness.
        # Units: dz and L [m]; N [1].
        # Domain/constraints: L > 0 and N >= 3. L is a learnable binding held by
        #   reference, so dz is derived at every read and never stored: it
        #   follows a changed L, and no autograd graph is carried from one call
        #   into the next. It has the device of the binding; the kernel aligns
        #   it with the state through on_device_of.
        # Sign convention: dz > 0; depth is positive downward.
        return self.total_soil_depth / self.soil_layers

    @property
    def m_vg(self) -> Tensor:
        """van Genuchten exponent m [1] from the current n binding."""
        # References: van Genuchten (1980), Soil Sci. Soc. Am. J. 44, 892-898
        #   (Mualem restriction); Mualem (1976), Water Resources Research
        #   12(3), 513-522.
        # Mathematical expression: m = 1 - 1 / n.
        # Parameter meanings: n is the pore-size distribution index
        #   (hydrosphere.soil_moisture.n_vg); m the derived exponent of the
        #   retention and conductivity curves.
        # Units: n and m [1].
        # Domain/constraints: n > 1, so 0 < m < 1. n is a learnable binding held
        #   by reference, so m is derived at every read and never stored: the
        #   retention and conductivity laws always use one consistent pair
        #   (n, m), and no autograd graph is carried from one call into the
        #   next. It has the device of the binding; the kernel aligns it with
        #   the state through on_device_of.
        # Sign convention: none (a dimensionless shape exponent).
        return van_genuchten_m_parameter(self.n_vg)

    def _initialize_layers(self) -> None:
        """Initialize soil layer properties and discretization grid."""
        # Depth coordinates (positive downward, layer centers)
        self.layer_depths = torch.linspace(
            self.layer_thickness / 2,
            self.total_soil_depth - self.layer_thickness / 2,
            self.soil_layers,
            dtype=torch.float32
        )
        
        # Interface depths (for flux calculations)
        self.interface_depths = torch.linspace(
            0.0, self.total_soil_depth, self.soil_layers + 1,
            dtype=torch.float32
        )
    
    def _validate_parameters(self) -> None:
        """Validate parameter consistency and physical constraints."""
        if self.theta_residual >= self.theta_saturated:
            raise ValueError(f"Residual water content ({self.theta_residual}) must be less than saturated ({self.theta_saturated})")
        
        if self.theta_residual < 0 or self.theta_saturated > 1:
            raise ValueError("Water contents must be in range [0, 1]")
        
        if self.n_vg <= 1:
            raise ValueError("Van Genuchten n parameter must be > 1")
        
        if self.alpha_vg <= 0 or self.k_saturated <= 0:
            raise ValueError("Hydraulic parameters must be positive")
        
        if self.soil_layers < 3:
            raise ValueError("Minimum 3 soil layers required for discretization")
    
    def get_dependencies(self) -> DependencyDeclaration:
        """
        Return same-step optional producer dependencies for this functor.

        Returns
        -------
        DependencyDeclaration
            Optional producer ordering for surface runoff and the mutually
            exclusive evapotranspiration closures. External StateManager
            boundary inputs remain valid when a producer is absent.
        """
        return self._merge_declared_dependencies({
            "required": [],
            "optional": [
                "surface_runoff",
                "penman_monteith_et",
                "priestley_taylor_et",
                "capillary_rise_transport",
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
            value = int(self.soil_layers)
        if value < 1:
            raise ValueError("Structural degrees of freedom must be positive.")
        return value

    def declared_forms(self) -> tuple[CochainSpaceSpec, ...]:
        """Declare canonical cochain spaces for Richards soil water flow.

        References:
          - Richards (1931), unsaturated porous-media flow.
          - van Genuchten (1980) and Mualem (1976), hydraulic constitutive laws.
          - Hirani (2003), Discrete Exterior Calculus.
        Mathematical expression:
          theta_dot = -D_z q_z + s_z, q_z = star_K (d h - g_z).
        Parameter meanings:
          theta is volumetric water content [m3 m-3], h is pressure head [m],
          K is hydraulic conductivity [m s-1], and q_z is water flux [kg m-2 s-1].
        """

        n_dofs = self._structural_n_dofs()
        return (
            CochainSpaceSpec(
                complex_id="hydrosphere_richards_soil_moisture",
                degree=0,
                primality="primal",
                n_dofs=n_dofs,
                units="m3 m-3",
                extensivity="intensive",
                physical_domain="hydrosphere",
                semantic_name="soil_moisture",
                orientation_convention="positive_downward_depth",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="hydrosphere_richards_soil_moisture",
                degree=0,
                primality="primal",
                n_dofs=n_dofs,
                units="m",
                extensivity="intensive",
                physical_domain="hydrosphere",
                semantic_name="pressure_head",
                orientation_convention="positive_downward_depth",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="hydrosphere_richards_soil_moisture",
                degree=1,
                primality="dual",
                n_dofs=n_dofs,
                units="kg m-2 s-1",
                extensivity="integral",
                physical_domain="hydrosphere",
                semantic_name="water_flux_profile",
                orientation_convention="downward_positive",
                boundary_role="port",
            ),
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Declare ParameterManager keys used by runtime and structural builders.

        Physical constants are resolved through get_physical_constant() and are
        intentionally excluded from the ParameterManager dependency contract.
        """

        return self._REQUIRED_PARAMETER_KEYS

    def build_hodge_contract(
        self,
        n_dofs: Optional[int] = None,
        *,
        strict: bool = True,
    ) -> HodgeStarSpec:
        """Build the Richards hydraulic conductivity Hodge contract.

        References:
          - Richards (1931), Darcy-continuity form.
          - Mualem (1976) and van Genuchten (1980), conductivity law.
          - van der Schaft and Maschke (1995), dissipative port systems.
        Mathematical expression:
          star_K = diag(rho_w * K_s), with rho_w > 0 and K_s > 0.
        Parameter meanings:
          rho_w is water density [kg m-3]; K_s is saturated conductivity [m s-1].
        """

        n_dofs = self._structural_n_dofs(n_dofs=n_dofs)
        def _scalar(value: Any) -> float:
            if isinstance(value, Tensor):
                flattened = value.detach().reshape(-1)
                if flattened.numel() == 0:
                    raise ValueError("Cannot convert empty tensor to scalar.")
                return float(flattened[:1].to(dtype=torch.float64).tolist()[0])
            return float(value)

        mobility_value = _scalar(self.water_density) * _scalar(self.k_saturated)
        values = torch.full(
            (n_dofs,),
            mobility_value,
            dtype=torch.float32,
            device=getattr(self, "device", torch.device("cpu")),
        )
        if bool(torch.any(values <= 0.0)):
            raise ValueError("Richards hydraulic mobility must be positive.")

        domain = CochainSpaceSpec(
            complex_id="hydrosphere_richards_soil_moisture",
            degree=1,
            primality="primal",
            n_dofs=n_dofs,
            units="m m-1",
            extensivity="intensive",
            physical_domain="hydrosphere",
            semantic_name="richards_hydraulic_gradient",
            orientation_convention="downward_positive",
        )
        codomain = CochainSpaceSpec(
            complex_id="hydrosphere_richards_soil_moisture",
            degree=1,
            primality="dual",
            n_dofs=n_dofs,
            units="kg m-2 s-1",
            extensivity="integral",
            physical_domain="hydrosphere",
            semantic_name="water_flux_profile",
            orientation_convention="downward_positive",
            boundary_role="port",
        )
        hodge = HodgeStarSpec(
            name=f"{self.name}_richards_hydraulic_hodge",
            domain=domain,
            codomain=codomain,
            matrix=torch.diag(values),
            parameter_keys=(
                "hydrosphere.soil_moisture.saturated_hydraulic_conductivity",
                "relic.physical.constants.water.density_liquid",
            ),
            metric_keys=("hydrosphere.soil_moisture.soil_depth",),
            material_law="van_genuchten_mualem_richards_flow",
        )
        if strict:
            hodge.validate(passive=True)
        return hodge

    def build_hodge_terms(
        self,
        complex_bundle: Any = None,
        resolved_parameters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[HodgeStarSpec, ...]:
        """Build Richards Hodge terms for IPHS assembly."""

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
        """Return no reversible interconnection for scalar vertical water flow."""

        return None

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the positive Richards hydraulic mobility contribution.

        References:
          - Richards (1931), Mualem (1976), and van der Schaft and Maschke (1995).
        Mathematical expression:
          M_h = diag(rho_w * K_s), sigma = X_h.T M_h X_h >= 0.
        Parameter meanings:
          K_s is saturated hydraulic conductivity [m s-1].
        """

        hodge = self.build_hodge_contract(
            n_dofs=self._structural_n_dofs(complex_bundle=complex_bundle),
            strict=True,
        )
        matrix = hodge.matrix
        return {
            "state_vars": tuple(
                f"richards_pressure_head[{idx}]" for idx in range(int(matrix.shape[0]))
            ),
            "J": torch.zeros_like(matrix),
            "M": matrix,
            "ports": self.ports(),
        }

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Declare Richards surface, capillary, and drainage matter ports."""

        effort_space = CochainSpaceSpec(
            complex_id="hydrosphere_richards_soil_moisture",
            degree=0,
            primality="primal",
            n_dofs=1,
            units="m",
            extensivity="intensive",
            physical_domain="hydrosphere",
            semantic_name="richards_pressure_head",
            orientation_convention="downward_positive",
            boundary_role="port",
        )
        flow_kwargs = {
            "complex_id": "hydrosphere_richards_soil_moisture",
            "degree": 1,
            "primality": "dual",
            "n_dofs": 1,
            "units": "kg m-2 s-1",
            "extensivity": "integral",
            "physical_domain": "hydrosphere",
            "orientation_convention": "downward_positive",
            "boundary_role": "port",
        }
        surface_effort_space = CochainSpaceSpec(
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
        surface_flow_kwargs = {
            "complex_id": "surface_soil_water_interface",
            "degree": 2,
            "primality": "dual",
            "n_dofs": 1,
            "units": "kg m-2 s-1",
            "extensivity": "integral",
            "physical_domain": "surface_soil_interface",
            "orientation_convention": "outward_positive",
            "boundary_role": "port",
        }
        capillary_effort_space = CochainSpaceSpec(
            complex_id="hydrosphere_capillary_rise",
            degree=0,
            primality="primal",
            n_dofs=1,
            units="m",
            extensivity="intensive",
            physical_domain="hydrosphere",
            semantic_name="water_table_boundary_head",
            orientation_convention="upward_positive",
            boundary_role="port",
        )
        capillary_flow_space = CochainSpaceSpec(
            complex_id="hydrosphere_capillary_rise",
            degree=2,
            primality="dual",
            n_dofs=1,
            units="kg m-2 s-1",
            extensivity="integral",
            physical_domain="hydrosphere",
            semantic_name="capillary_boundary_mass_flux",
            orientation_convention="upward_positive",
            boundary_role="port",
        )
        return (
            PortSpecIPHS(
                name="soil_infiltration_boundary",
                effort_var="surface_hydraulic_head",
                flow_var="infiltration_flux",
                effort_space=surface_effort_space,
                flow_space=CochainSpaceSpec(
                    **surface_flow_kwargs,
                    semantic_name="infiltration_mass_flux",
                ),
                power_units="kg m-1 s-1",
                positive_power_convention="into_system",
                port_kind="matter",
            ),
            PortSpecIPHS(
                name="soil_evapotranspiration_boundary",
                effort_var="surface_hydraulic_head",
                flow_var="evapotranspiration_flux",
                effort_space=surface_effort_space,
                flow_space=CochainSpaceSpec(
                    **surface_flow_kwargs,
                    semantic_name="evapotranspiration_mass_flux",
                ),
                power_units="kg m-1 s-1",
                positive_power_convention="out_of_system",
                port_kind="matter",
            ),
            PortSpecIPHS(
                name="soil_capillary_boundary",
                effort_var="water_table_boundary_head",
                flow_var="capillary_rise_flux",
                effort_space=capillary_effort_space,
                flow_space=capillary_flow_space,
                power_units="kg m-1 s-1",
                positive_power_convention="into_system",
                port_kind="matter",
            ),
            PortSpecIPHS(
                name="richards_drainage_flux",
                effort_var="richards_pressure_head",
                flow_var="drainage_flux",
                effort_space=effort_space,
                flow_space=CochainSpaceSpec(
                    **flow_kwargs,
                    semantic_name="richards_drainage_mass_flux",
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
    
    def forward(self, 
        # References:
        #   Richards (1931); Celia et al. (1990); van Genuchten (1980).
        # Mathematical expression:
        #   q=-K(theta)*(dh/dz-1); dtheta/dt=-dq/dz;
        #   theta1-theta0=dt_sub*(q_top-q_bottom)/dz in each accepted substep;
        #   q_mean=sum(q_sub*dt_sub)/dt, with sum(dt_sub)=dt.
        # Parameter meanings:
        #   theta is the layer profile; h is pressure head; K is conductivity;
        #   external boundary mass rates are divided by rho_w. Picard or Newton
        #   solves the same implicit residual on the selected boundary contract.
        # Units:
        #   theta [m3 m-3], h/dz [m], K/q [m s-1], input/output mass rates
        #   [kg m-2 s-1], dt/dt_sub [s], matric pressure rho_w*g*h [Pa].
        # Domain/constraints:
        #   Exactly soil_layers entries; finite canonical boundary fluxes; internal
        #   substeps cover the full external dt. Nonconvergence and checked mass errors
        #   reject the step. Prescribed-head boundary rates are reconciled with storage.
        # Sign convention:
        #   z and q are positive downward; top infiltration is inward, ET subtracts
        #   from top input, and positive capillary supply is a negative bottom q.
                state: StateDict, 
                time: float, 
                dt: float) -> FluxDict:
        """
        Compute soil moisture dynamics using Richards equation framework.
        
        Uses the canonical Phase 8 IPHS mixed-form Richards runtime path.
        
        Parameters
        ----------
        state : StateDict
            Current system state containing soil moisture profile
        time : float
            Current simulation time [s]
        dt : float  
            Time step [s]
        Returns
        -------
        FluxDict
            Soil moisture changes and diagnostic variables
            
        Raises
        ------
        ValueError
            If required state variables are missing or invalid
        """
        return self._compute_full_solution(state, time, dt)
    

    def _compute_full_solution(self, 
                              state: StateDict, 
                              time: float, 
                              dt: float) -> FluxDict:
        """
        Complete numerical solution of Richards equation using finite differences.
        
        Implements implicit time integration with Picard iteration for linearization
        and Thomas algorithm for solving the tridiagonal linear system.
        
        Parameters
        ----------
        state : StateDict
            Current system state containing soil moisture profile
        time : float
            Current simulation time [s]
        dt : float  
            Time step [s]
        Returns
        -------
        FluxDict
            Soil moisture changes and diagnostic variables
            
        Mathematical Expression
        -----------------------
        Richards equation (mixed form):
        dtheta/dt = d/dz[K(theta)(dh/dz - 1)] - S(theta)
        
        Implicit discretization:
        (theta_i^(n+1) - theta_i^n)/dt = [K_{i+1/2}(dh/dz)_{i+1/2} - K_{i-1/2}(dh/dz)_{i-1/2}]/dz - S_i
        
        References
        ----------
        - Celia et al. (1990). A general mass-conservative numerical solution for the unsaturated flow equation
        - Van Genuchten (1980). A closed-form equation for predicting the hydraulic conductivity
        - Mualem (1976). A new model for predicting the hydraulic conductivity
        """
        # References:
        #   Richards (1931); Celia et al. (1990); van Genuchten (1980).
        # Mathematical expression:
        #   q=-K(theta)*(dh/dz-1); dtheta/dt=-dq/dz;
        #   theta1-theta0=dt_sub*(q_top-q_bottom)/dz in each accepted substep;
        #   q_mean=sum(q_sub*dt_sub)/dt, with sum(dt_sub)=dt.
        # Parameter meanings:
        #   theta is the layer profile; h is pressure head; K is conductivity;
        #   external boundary mass rates are divided by rho_w. Picard or Newton
        #   solves the same implicit residual on the selected boundary contract.
        # Units:
        #   theta [m3 m-3], h/dz [m], K/q [m s-1], input/output mass rates
        #   [kg m-2 s-1], dt/dt_sub [s], matric pressure rho_w*g*h [Pa].
        # Domain/constraints:
        #   Exactly soil_layers entries; finite canonical boundary fluxes; internal
        #   substeps cover the full external dt. Nonconvergence and checked mass errors
        #   reject the step. Prescribed-head boundary rates are reconciled with storage.
        # Sign convention:
        #   z and q are positive downward; top infiltration is inward, ET subtracts
        #   from top input, and positive capillary supply is a negative bottom q.
        soil_moisture_input = state.get("soil_moisture")
        if (
            not isinstance(soil_moisture_input, Tensor)
            or soil_moisture_input.ndim != 1
            or soil_moisture_input.numel() != self.soil_layers
        ):
            raise ValueError(
                "soil_moisture must be a one-dimensional Tensor profile with "
                f"exactly {self.soil_layers} layers."
            )
        soil_moisture = soil_moisture_input

        # References: Richards (1931); Celia et al. (1990).
        # Mathematical expression: F_surface = F_infiltration - F_ET.
        # Parameter meanings: both boundary mass fluxes are [kg m-2 s-1];
        # positive infiltration enters and positive evapotranspiration leaves.
        boundary_fluxes = {}
        for key in ("infiltration_flux", "evapotranspiration_flux"):
            if key not in state:
                raise ValueError(
                    f"Richards full solution requires canonical '{key}'."
                )
            if not isinstance(state[key], Tensor):
                raise ValueError(f"{key} must be a scalar Tensor.")
            value = state[key].to(
                device=soil_moisture.device,
                dtype=soil_moisture.dtype,
            )
            if value.numel() != 1:
                raise ValueError(f"{key} must be a scalar boundary mass flux.")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{key} must be finite.")
            if bool(torch.any(value < 0.0)):
                raise ValueError(f"{key} must be non-negative.")
            boundary_fluxes[key] = value.reshape(())
        surface_flux = (
            boundary_fluxes["infiltration_flux"]
            - boundary_fluxes["evapotranspiration_flux"]
        )

        bottom_boundary_type = (
            self._bottom_boundary.get("type")
            if self._bottom_boundary is not None
            else None
        )
        has_capillary_flux = "capillary_rise_flux" in state
        if bottom_boundary_type == "capillary_flux":
            if not has_capillary_flux:
                raise ValueError(
                    "Richards capillary lower boundary requires canonical "
                    "'capillary_rise_flux'."
                )
            capillary_flux_input = state["capillary_rise_flux"]
            if not isinstance(capillary_flux_input, Tensor):
                raise ValueError("capillary_rise_flux must be a scalar Tensor.")
            capillary_mass_flux = capillary_flux_input.to(
                device=soil_moisture.device,
                dtype=soil_moisture.dtype,
            )
            if capillary_mass_flux.numel() != 1:
                raise ValueError(
                    "capillary_rise_flux must be a scalar boundary mass flux."
                )
            if not bool(torch.isfinite(capillary_mass_flux).all()):
                raise ValueError("capillary_rise_flux must be finite.")
            if bool(torch.any(capillary_mass_flux < 0.0)):
                raise ValueError("capillary_rise_flux must be non-negative.")
            capillary_mass_flux = capillary_mass_flux.reshape(())
        else:
            if has_capillary_flux:
                raise ValueError(
                    "capillary_rise_flux requires the explicit 'capillary_flux' "
                    "lower boundary selection."
                )
            capillary_mass_flux = None

        # Validate inputs
        if not torch.isfinite(soil_moisture).all():
            raise ValueError("Soil moisture content must be finite")
        if torch.any(soil_moisture < 0) or torch.any(soil_moisture > 1):
            raise ValueError("Soil moisture content must be in range [0, 1]")
        
        # ParameterManager is the only production nonlinear-solver policy owner.
        linearization = (
            "newton_raphson" if self.enable_newton_raphson else "picard"
        )
        max_iterations = (
            self.max_newton_iterations
            if self.enable_newton_raphson
            else self.max_picard_iterations
        )
        convergence_tolerance = self.convergence_tolerance
        cfl_limit = self.richards_cfl_coefficient
        
        # Held hydraulic values are read on the device of the state profile
        # (see on_device_of); their values and dtypes are unchanged.
        layer_thickness = on_device_of(self.layer_thickness, soil_moisture)
        total_soil_depth = on_device_of(self.total_soil_depth, soil_moisture)
        theta_residual = on_device_of(self.theta_residual, soil_moisture)
        theta_saturated = on_device_of(self.theta_saturated, soil_moisture)

        # References: CF-1.8 water-flux conversion and Richards (1931).
        # Mathematical expression: q_v = F_w / rho_w.
        # Parameter meanings: F_w [kg m-2 s-1], rho_w [kg m-3], q_v [m s-1].
        water_density = self._ensure_tensor(self.water_density).to(
            device=soil_moisture.device,
            dtype=soil_moisture.dtype,
        )
        gravity = self._ensure_tensor(self.gravity).to(
            device=soil_moisture.device,
            dtype=soil_moisture.dtype,
        )
        surface_flux_vol = surface_flux / water_density
        # References: Richards (1931); HYDRUS technical documentation for
        # variable lower-boundary water fluxes.
        # Mathematical expression: q_bottom = -F_capillary / rho_w.
        # Parameter meanings: F_capillary is the upward-positive capillary
        # mass flux [kg m-2 s-1], rho_w is water density [kg m-3], and
        # q_bottom is the downward-positive Richards flux [m s-1].
        bottom_flux_vol = (
            -capillary_mass_flux / water_density
            if capillary_mass_flux is not None
            else None
        )
        
        # Initial pressure head and hydraulic conductivity
        theta_old = soil_moisture.clone()
        theta_new = soil_moisture.clone()  # Initial guess for new time step

        top_is_prescribed_head = (
            self._top_boundary is not None
            and self._top_boundary["type"] == "prescribed_head"
        )
        bottom_is_prescribed_head = (
            self._bottom_boundary is not None
            and self._bottom_boundary["type"] == "prescribed_head"
        )
        if top_is_prescribed_head or bottom_is_prescribed_head:
            # References: Celia et al. (1990); Paniconi and Putti (1994),
            # nonlinear initialization for Richards boundary-value problems.
            # Mathematical expression: h_i^0 = h_top + zeta_i(h_bottom-h_top),
            # theta_i^0 = theta_vG(h_i^0).
            # Parameter meanings: zeta_i is normalized depth [1], h_top and
            # h_bottom are either prescribed heads or the corresponding old
            # boundary heads [m], and theta_vG is the canonical van Genuchten
            # retention relation [m3 m-3]. This is an initial guess only; both
            # solvers still evaluate the same mixed residual and no second
            # state-advance formula is introduced.
            old_pressure_head = self._compute_pressure_head(theta_old)
            top_initial_head = (
                self._ensure_tensor(self._top_boundary["value"]).to(
                    device=theta_new.device,
                    dtype=theta_new.dtype,
                )
                if top_is_prescribed_head
                else old_pressure_head[0]
            )
            bottom_initial_head = (
                self._ensure_tensor(self._bottom_boundary["value"]).to(
                    device=theta_new.device,
                    dtype=theta_new.dtype,
                )
                if bottom_is_prescribed_head
                else old_pressure_head[-1]
            )
            depth_fraction = torch.linspace(
                0.0,
                1.0,
                steps=self.soil_layers,
                device=theta_new.device,
                dtype=theta_new.dtype,
            )
            head_initial = top_initial_head + depth_fraction * (
                bottom_initial_head - top_initial_head
            )
            theta_new = self._compute_theta_from_head(head_initial)
        
        # Adaptive time stepping for stability
        dt_effective = self._compute_adaptive_timestep(theta_old, dt, cfl_limit)
        dt_ratio = self._ensure_tensor(dt).to(device=theta_old.device, dtype=theta_old.dtype) / dt_effective
        # .item used only for loop count, not part of the physical computation.
        n_sub_steps_tensor = torch.clamp(torch.ceil(torch.max(dt_ratio)), min=1.0)
        # Convert scalar tensor to Python int for loop control only.
        n_sub_steps = int(n_sub_steps_tensor.detach())
        dt_sub = dt / n_sub_steps
        integrated_layer_fluxes = theta_old.new_zeros(self.soil_layers + 1)
        integrated_clipping_residual = theta_old.new_zeros(theta_old.shape)
        
        # Sub-stepping for large time steps
        for sub_step in range(n_sub_steps):
            theta_substep_old = theta_old.clone()
            converged = False
            accepted_clipping_residual = theta_old.new_zeros(theta_old.shape)
            accepted_pressure_head = self._compute_pressure_head(theta_new)
            accepted_conductivity = self._compute_hydraulic_conductivity(theta_new)

            # References: Celia et al. (1990), mixed-form mass conservation.
            # Mathematical expression:
            #   |sum_i R_i dz| / W_old <= epsilon_mass,
            #   therefore ||R||_inf <= epsilon_mass * W_old / L.
            # Parameter meanings: R is the layer water-content residual
            # [m3 m-3], dz is layer thickness [m], W_old is stored water depth
            # [m], L is profile depth [m], and epsilon_mass is the configured
            # dimensionless conservation tolerance.
            # Units: R and the residual tolerance [m3 m-3]; W_old, dz, L [m];
            #   epsilon_mass [1].
            # Domain/constraints: uniform layer thickness with L = N dz, so
            #   |sum_i R_i dz| <= L ||R||_inf; W_old >= 0 for nonnegative theta.
            # Sign convention: a magnitude bound; the sign of R is not used.
            stored_water_depth = (
                torch.sum(theta_old) * layer_thickness
            )
            ledger_residual_tolerance = (
                self.mass_conservation_tolerance
                * stored_water_depth
                / total_soil_depth
            )
            
            for iteration in range(max_iterations):
                theta_iter = theta_new.clone()
                h_new = self._compute_pressure_head(theta_iter)
                k_new = self._compute_hydraulic_conductivity(theta_iter)

                current_residual = self._compute_newton_residual(
                    theta_iter,
                    theta_old,
                    h_new,
                    k_new,
                    dt_sub,
                    surface_flux_vol,
                    bottom_flux=bottom_flux_vol,
                )
                residual = torch.max(torch.abs(current_residual))
                if linearization == "newton_raphson":
                    state_scale = torch.max(
                        torch.maximum(torch.abs(theta_old), torch.abs(theta_iter))
                    )
                    nonlinear_tolerance = self.atol + self.rtol * state_scale
                else:
                    nonlinear_tolerance = self._ensure_tensor(
                        convergence_tolerance
                    ).to(device=theta_iter.device, dtype=theta_iter.dtype)
                if self.enable_mass_conservation_check:
                    nonlinear_tolerance = torch.minimum(
                        nonlinear_tolerance,
                        ledger_residual_tolerance,
                    )
                if bool(residual <= nonlinear_tolerance):
                    converged = True
                    accepted_pressure_head = h_new
                    accepted_conductivity = k_new
                    break
                
                if linearization == 'picard':
                    theta_candidate, precision_converged = (
                        self._solve_linearized_system_picard(
                            theta_old,
                            theta_iter,
                            h_new,
                            k_new,
                            dt_sub,
                            surface_flux_vol,
                            bottom_flux=bottom_flux_vol,
                        )
                    )
                else:
                    theta_candidate, precision_converged = (
                        self._solve_linearized_system_newton(
                            theta_old,
                            theta_iter,
                            h_new,
                            k_new,
                            dt_sub,
                            surface_flux_vol,
                            bottom_flux=bottom_flux_vol,
                        )
                    )
                
                theta_new = torch.clamp(
                    theta_candidate,
                    min=theta_residual,
                    max=theta_saturated,
                )
                accepted_clipping_residual = theta_new - theta_candidate

                accepted_pressure_head = self._compute_pressure_head(theta_new)
                accepted_conductivity = self._compute_hydraulic_conductivity(
                    theta_new
                )
                accepted_residual = self._compute_newton_residual(
                    theta_new,
                    theta_old,
                    accepted_pressure_head,
                    accepted_conductivity,
                    dt_sub,
                    surface_flux_vol,
                    bottom_flux=bottom_flux_vol,
                )
                residual = torch.max(torch.abs(accepted_residual))
                if bool(residual <= nonlinear_tolerance) or precision_converged:
                    converged = True
                    break
            
            if not converged:
                raise RuntimeError(
                    "Richards solver did not converge in "
                    f"{max_iterations} iterations at substep {sub_step}. "
                    f"Residual: {float(residual.detach()):.2e}"
                )

            integrated_clipping_residual = (
                integrated_clipping_residual + accepted_clipping_residual
            )

            # References: Celia et al. (1990), mass-conservative mixed form.
            # Mathematical expression:
            #   q_bar = (1 / dt) * sum_s(q_s * dt_s),
            #   Delta W = (q_top_bar - q_bottom_bar) * dt.
            # Parameter meanings:
            #   q_s is a downward-positive substep water flux [m s-1],
            #   dt_s is the substep duration [s], and W is water depth [m].
            # Units and route:
            #   The time-averaged boundary flux is the open-system port value
            #   consumed by StateManager for the enclosing model step.
            substep_fluxes = self._compute_inter_layer_fluxes(
                theta_new,
                accepted_pressure_head,
                accepted_conductivity,
            )
            if self._top_boundary is None:
                substep_fluxes[0] = surface_flux_vol
            elif self._top_boundary["type"] == "prescribed_flux":
                substep_fluxes[0] = self._ensure_tensor(
                    self._top_boundary["value"]
                ).to(device=theta_old.device, dtype=theta_old.dtype)
            elif self._top_boundary["type"] == "atmospheric":
                substep_fluxes[0] = surface_flux_vol
            else:
                top_storage_rate = (
                    (theta_new[0] - theta_substep_old[0])
                    * layer_thickness
                    / dt_sub
                )
                substep_fluxes[0] = top_storage_rate + substep_fluxes[1]

            if (
                self._bottom_boundary is not None
                and self._bottom_boundary["type"] == "prescribed_flux"
            ):
                bottom_flux_value = self._bottom_boundary["value"]
                substep_fluxes[-1] = self._ensure_tensor(bottom_flux_value).to(
                    device=theta_old.device,
                    dtype=theta_old.dtype,
                )
            elif bottom_boundary_type == "capillary_flux":
                substep_fluxes[-1] = bottom_flux_vol
            top_is_head = (
                self._top_boundary is not None
                and self._top_boundary["type"] == "prescribed_head"
            )
            bottom_is_head = (
                self._bottom_boundary is not None
                and self._bottom_boundary["type"] == "prescribed_head"
            )
            substep_storage_rate = (
                torch.sum(theta_new - theta_substep_old)
                * layer_thickness
                / dt_sub
            )
            if top_is_head and not bottom_is_head:
                substep_fluxes[0] = substep_storage_rate + substep_fluxes[-1]
            if bottom_is_head:
                substep_fluxes[-1] = substep_fluxes[0] - substep_storage_rate

            integrated_layer_fluxes = (
                integrated_layer_fluxes + substep_fluxes * dt_sub
            )
            
            # Update for next sub-step
            theta_old = theta_new.clone()
        
        # Compute final diagnostic variables
        final_pressure_head = self._compute_pressure_head(theta_new)
        final_conductivity = self._compute_hydraulic_conductivity(theta_new)
        
        # Flux outputs are time averages over all adaptive substeps.
        layer_fluxes = integrated_layer_fluxes / dt

        # Mass conservation check
        mass_error = self._check_mass_conservation(
            theta_old=soil_moisture,
            theta_new=theta_new,
            surface_flux=layer_fluxes[0],
            bottom_flux=layer_fluxes[-1],
            dt=dt,
        )
        if (
            self.enable_mass_conservation_check
            and bool(mass_error > self.mass_conservation_tolerance)
        ):
            raise RuntimeError(
                "Richards mass conservation error exceeds tolerance: "
                f"{float(mass_error.detach()):.6e} > "
                f"{self.mass_conservation_tolerance:.6e}."
            )
        moisture_clipping_residual = integrated_clipping_residual

        # Compute moisture change rate
        moisture_changes = (theta_new - soil_moisture) / dt
        
        # Compute diagnostic variables
        total_water_content = torch.sum(theta_new) * layer_thickness
        average_moisture = torch.mean(theta_new)
        surface_moisture = theta_new[0]
        
        # References:
        #   - van Genuchten (1980), Soil Sci. Soc. Am. J. 44, 892-898.
        #   - Feddes et al. (1978), Simulation of Field Water Use and Crop Yield.
        # Mathematical expression:
        #   f_rel = clamp((theta_bar - theta_wp) / (theta_fc - theta_wp), 0, 1).
        # Parameter meanings:
        #   theta_bar is profile-mean water content [m3 m-3].
        #   theta_fc and theta_wp are diagnostic wetness thresholds [m3 m-3].
        # Units:
        #   f_rel is dimensionless [1].
        # Domain/constraints:
        #   theta_fc > theta_wp is required for a finite quotient; the clamp keeps
        #   f_rel in [0, 1].
        # Sign convention:
        #   f_rel increases with wetness; it is a state ratio, not a flux.
        # Open-system route:
        #   Read-only observation; it neither advances storage nor represents
        #   the pressure-head-dependent plant uptake response alpha(h).
        field_capacity = (
            on_device_of(self.field_capacity_fraction, theta_new)
            * theta_saturated
        )
        wilting_point = (
            on_device_of(self.wilting_point_fraction, theta_new)
            * theta_residual
        )
        relative_moisture = torch.clamp(
            (average_moisture - wilting_point) / (field_capacity - wilting_point),
            min=0.0, max=1.0
        )
        
        # Infiltration capacity (maximum flux surface can accept)
        infiltration_capacity = self._compute_infiltration_capacity(theta_new[0], final_conductivity[0])

        # References:
        #   - van Genuchten (1980), Soil Sci. Soc. Am. J. 44, 892-898.
        #   - Celia et al. (1990), Water Resources Research 26, 1483-1496.
        # Mathematical expression:
        #   Se = (theta - theta_r) / (theta_s - theta_r).
        #   h = -(1 / alpha) * (Se^(-1 / m) - 1)^(1 / n).
        #   psi_m = rho_w * g * h.
        # Parameter meanings:
        #   theta is volumetric water content [m3 m-3]; theta_r and theta_s
        #   are residual and saturated contents [m3 m-3]; alpha is the
        #   inverse air-entry scale [m-1]; n and m are shape factors [1];
        #   h is pressure head [m]; rho_w is water density [kg m-3]; g is
        #   gravitational acceleration [m s-2]; psi_m is matric pressure [Pa].
        # Open-system route:
        #   This full-profile pressure is derived from the retained water
        #   state and has no independent water source, sink, or storage route.
        soil_matric_potential = final_pressure_head * gravity * water_density
        diagnostics = {
            "total_water_content": total_water_content,
            "solver_converged": bool(converged),
            "solver_iterations": int(iteration + 1),
            "pressure_head": final_pressure_head,
            "hydraulic_conductivity": final_conductivity,
            "vertical_flux": layer_fluxes * water_density,
            "average_soil_moisture": average_moisture,
            "surface_soil_moisture": surface_moisture,
            "infiltration_capacity": infiltration_capacity * water_density,
            "mass_conservation_error": mass_error,
            "moisture_clipping_residual": moisture_clipping_residual,
        }

        return {
            "soil_moisture": UpdateItem(
                value=theta_new,
                mode=UpdateMode.REPLACE,
                unit="m3 m-3",
                per_timestep=False,
                source_functor=self.name,
            ),
            "diagnostics.soil_moisture_richards.soil_moisture_change_rate": UpdateItem(
                value=moisture_changes,
                mode=UpdateMode.REPLACE,
                unit="m3 m-3 s-1",
                per_timestep=False,
                source_functor=self.name,
            ),
            "soil_matric_potential": UpdateItem(
                value=soil_matric_potential,
                mode=UpdateMode.REPLACE,
                unit="Pa",
                per_timestep=False,
                source_functor=self.name,
            ),
            "drainage_flux": UpdateItem(
                value=torch.clamp(layer_fluxes[-1] * water_density, min=0.0),
                mode=UpdateMode.REPLACE,
                unit="kg m-2 s-1",
                per_timestep=False,
                source_functor=self.name,
            ),
            "diagnostics.soil_moisture_richards.relative_moisture": UpdateItem(
                value=relative_moisture,
                mode=UpdateMode.REPLACE,
                unit="dimensionless",
                per_timestep=False,
                source_functor=self.name,
            ),
            "diagnostics.soil_moisture_richards.water_flux_profile": UpdateItem(
                value=layer_fluxes[1:] * water_density,
                mode=UpdateMode.REPLACE,
                unit="kg m-2 s-1",
                per_timestep=False,
                source_functor=self.name,
            ),
            "diagnostics.soil_moisture_richards": diagnostics,
        }

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Evaluate the integrated open-column water ledger.

        References:
          - Richards (1931), unsaturated porous-medium water flow.
          - Celia et al. (1990), mass-conservative mixed-form solution.
        Mathematical expression:
          epsilon_m = abs(Delta W - (q_top - q_bottom) * dt) / W_old.
        Parameter meanings:
          W is stored water depth [m], q_top and q_bottom are signed boundary
          water fluxes [m s-1], and epsilon_m is dimensionless.
        Open-system route:
          Infiltration and selected capillary rise enter the soil column;
          evapotranspiration and drainage leave through typed matter ports.
        """

        del input_state
        diagnostic_key = "diagnostics.soil_moisture_richards"
        if diagnostic_key not in output_updates:
            raise KeyError(f"Missing conservation diagnostics '{diagnostic_key}'")
        diagnostics = output_updates[diagnostic_key]
        if isinstance(diagnostics, UpdateItem):
            diagnostics = diagnostics.value
        if not isinstance(diagnostics, Mapping):
            raise TypeError("Richards conservation diagnostics must be a mapping")
        residual_key = "mass_conservation_error"
        if residual_key not in diagnostics:
            raise KeyError(f"Missing conservation diagnostic '{residual_key}'")
        residual = diagnostics[residual_key]
        if isinstance(residual, UpdateItem):
            residual = residual.value
        if not isinstance(residual, Tensor):
            raise TypeError("Richards mass residual must be a torch.Tensor")
        if residual.numel() == 0:
            raise ValueError("Richards mass residual must be non-empty")
        if not bool(torch.all(torch.isfinite(residual))):
            raise ValueError("Richards mass residual must be finite")
        magnitude = torch.max(torch.abs(residual))
        return {"mass": float(magnitude.detach().item())}
    
    def _compute_pressure_head(self, theta: Tensor) -> Tensor:
        """
        Compute pressure head from water content using Van Genuchten model.
        
        Mathematical expression:
            h = -1/alpha * [(thetas-thetar)/(theta-thetar)]^(1/m) - 1]^(1/n)
        
        Parameters
        ----------
        theta : Tensor
            Volumetric water content [m3 m-3]
            
        Returns
        -------
        Tensor
            Pressure head [m] (negative for unsaturated conditions)
        """
        # References:
        #   van Genuchten (1980); Mualem (1976).
        # Mathematical expression:
        #   Se=clip((theta-theta_r)/(theta_s-theta_r),eps,1);
        #   h=-(Se^(-1/m)-1)^(1/n)/alpha;
        #   K=Ks*Se^l*[1-(1-Se^(1/m))^m]^2; m=1-1/n;
        #   theta(h)=theta_r+(theta_s-theta_r)*(1+(alpha*abs(h))^n)^(-m).
        # Parameter meanings:
        #   theta_r/theta_s bound water content; alpha,n,m describe retention;
        #   l is tortuosity and Ks is saturated conductivity. Nonnegative head
        #   returns saturated water content in the inverse function.
        # Units:
        #   theta/Se/n/m/l [1], h [m], alpha [m-1], K/Ks [m s-1].
        # Domain/constraints:
        #   Use configured endpoint floors for the retention inversions; require
        #   theta_s>theta_r, alpha>0, n>1 and positive Ks. These are constitutive
        #   properties, not an additional state advance.
        # Sign convention:
        #   Unsaturated h is nonpositive; K is a nonnegative conductance magnitude.
        theta_residual = on_device_of(self.theta_residual, theta)
        theta_saturated = on_device_of(self.theta_saturated, theta)
        alpha_vg = on_device_of(self.alpha_vg, theta)
        n_vg = on_device_of(self.n_vg, theta)
        m_vg = on_device_of(self.m_vg, theta)
        se = torch.clamp(
            (theta - theta_residual) / (theta_saturated - theta_residual),
            min=self.saturation_epsilon,
            max=1.0,
        )
        
        # Van Genuchten pressure head relation
        # h = -1/alpha * [(Se^(-1/m) - 1)]^(1/n)
        term1 = torch.pow(se, -1.0 / m_vg) - 1.0
        min_pressure_term = self.minimum_pressure_term
        term2 = torch.pow(torch.clamp(term1, min=min_pressure_term), 1.0 / n_vg)
        pressure_head = -term2 / alpha_vg
        
        return pressure_head
    
    def _compute_hydraulic_conductivity(self, theta: Tensor) -> Tensor:
        """
        Compute hydraulic conductivity from water content using Mualem model.
        
        Mathematical expression:
            K(theta) = Ks * Se^l * [1 - (1 - Se^(1/m))^m]2
        
        Parameters
        ----------
        theta : Tensor
            Volumetric water content [m3 m-3]
            
        Returns
        -------
        Tensor
            Hydraulic conductivity [m s-1]
        """
        # References:
        #   van Genuchten (1980); Mualem (1976).
        # Mathematical expression:
        #   Se=clip((theta-theta_r)/(theta_s-theta_r),eps,1);
        #   h=-(Se^(-1/m)-1)^(1/n)/alpha;
        #   K=Ks*Se^l*[1-(1-Se^(1/m))^m]^2; m=1-1/n;
        #   theta(h)=theta_r+(theta_s-theta_r)*(1+(alpha*abs(h))^n)^(-m).
        # Parameter meanings:
        #   theta_r/theta_s bound water content; alpha,n,m describe retention;
        #   l is tortuosity and Ks is saturated conductivity. Nonnegative head
        #   returns saturated water content in the inverse function.
        # Units:
        #   theta/Se/n/m/l [1], h [m], alpha [m-1], K/Ks [m s-1].
        # Domain/constraints:
        #   Use configured endpoint floors for the retention inversions; require
        #   theta_s>theta_r, alpha>0, n>1 and positive Ks. These are constitutive
        #   properties, not an additional state advance.
        # Sign convention:
        #   Unsaturated h is nonpositive; K is a nonnegative conductance magnitude.
        theta_residual = on_device_of(self.theta_residual, theta)
        theta_saturated = on_device_of(self.theta_saturated, theta)
        tortuosity = on_device_of(self.tortuosity, theta)
        m_vg = on_device_of(self.m_vg, theta)
        k_saturated = on_device_of(self.k_saturated, theta)
        se = torch.clamp(
            (theta - theta_residual) / (theta_saturated - theta_residual),
            min=self.saturation_epsilon,
            max=1.0,
        )
        
        # Mualem hydraulic conductivity model
        term1 = torch.pow(se, tortuosity)
        term2 = 1.0 - torch.pow(se, 1.0 / m_vg)
        term3 = torch.pow(torch.clamp(term2, min=0.0), m_vg)
        term4 = torch.pow(1.0 - term3, 2.0)
        
        hydraulic_conductivity = k_saturated * term1 * term4
        
        return hydraulic_conductivity
    
    def set_boundary_conditions(self, 
                               top: Optional[Dict] = None, 
                               bottom: Optional[Dict] = None) -> None:
        """
        Set boundary conditions for the Richards equation.
        
        Parameters
        ----------
        top : Dict, optional
            Top boundary conditions (surface)
            Keys: 
            - "type": "prescribed_flux", "prescribed_head", "atmospheric"
            - "value": boundary value [m s-1 for flux, m for head]
        bottom : Dict, optional
            Bottom boundary conditions (depth)
            Keys:
            - "type": "prescribed_flux", "prescribed_head", "free_drainage",
              "capillary_flux"
            - "value": boundary value [m s-1 for flux, m for head]
        Notes
        -----
        Atmospheric mode consumes the canonical ``infiltration_flux`` and
        ``evapotranspiration_flux`` state inputs. Prescribed modes require an
        explicit finite scalar value; no ParameterManager boundary fallback
        is used by the production runtime.
        """
        if top is not None:
            if not isinstance(top, Mapping):
                raise ValueError("Top boundary must be a mapping.")
            if "type" not in top:
                raise ValueError("Top boundary must declare an explicit 'type'.")
            boundary_type = top["type"]
            if boundary_type not in {
                "prescribed_flux",
                "prescribed_head",
                "atmospheric",
            }:
                raise ValueError(f"Unknown top boundary condition type: {boundary_type}")
            self._top_boundary = self._validate_boundary_configuration(
                side="Top",
                boundary=top,
                requires_value=boundary_type in {
                    "prescribed_flux",
                    "prescribed_head",
                },
            )

        if bottom is not None:
            if not isinstance(bottom, Mapping):
                raise ValueError("Bottom boundary must be a mapping.")
            if "type" not in bottom:
                raise ValueError("Bottom boundary must declare an explicit 'type'.")
            boundary_type = bottom["type"]
            if boundary_type not in {
                "prescribed_flux",
                "prescribed_head",
                "free_drainage",
                "capillary_flux",
            }:
                raise ValueError(f"Unknown bottom boundary condition type: {boundary_type}")
            self._bottom_boundary = self._validate_boundary_configuration(
                side="Bottom",
                boundary=bottom,
                requires_value=boundary_type in {
                    "prescribed_flux",
                    "prescribed_head",
                },
            )

    def _validate_boundary_configuration(
        self,
        *,
        side: str,
        boundary: Mapping[str, Any],
        requires_value: bool,
    ) -> Dict[str, Any]:
        """Return one explicit, finite boundary configuration."""

        normalized: Dict[str, Any] = {"type": boundary["type"]}
        if requires_value:
            if "value" not in boundary or boundary["value"] is None:
                raise ValueError(
                    f"{side} boundary type '{boundary['type']}' requires a value."
                )
            value = self._ensure_tensor(boundary["value"])
            if value.numel() != 1 or not bool(torch.isfinite(value).all()):
                raise ValueError(
                    f"{side} boundary value must be a finite scalar."
                )
            normalized["value"] = boundary["value"]
        elif "value" in boundary:
            raise ValueError(
                f"{side} boundary type '{boundary['type']}' does not accept a value."
            )
        return normalized
    
    def get_hydraulic_properties(self) -> Dict[str, Union[float, Tensor]]:
        """
        Get soil hydraulic properties for diagnostics and coupling.
        
        Returns
        -------
        Dict[str, Union[float, Tensor]]
            Hydraulic parameters and functions
        """
        return {
            "soil_layers": self.soil_layers,
            "total_soil_depth": self.total_soil_depth,
            "layer_thickness": self.layer_thickness,
            "theta_residual": self.theta_residual,
            "theta_saturated": self.theta_saturated,
            "alpha_vg": self.alpha_vg,
            "n_vg": self.n_vg,
            "m_vg": self.m_vg,
            "k_saturated": self.k_saturated,
            "tortuosity": self.tortuosity,
            "layer_depths": self.layer_depths,
            "interface_depths": self.interface_depths
        }
    
    def _compute_adaptive_timestep(self, theta: Tensor, dt_requested: float, cfl_limit: float = None) -> Tensor:
        """
        Compute adaptive time step based on CFL stability condition.
        
        Parameters
        ----------
        theta : Tensor
            Current soil moisture profile
        dt_requested : float
            Requested time step [s]
        cfl_limit : float
            CFL stability limit [dimensionless]
            
        Returns
        -------
        Tensor
            Adaptive time step [s]
            
        Mathematical Expression
        -----------------------
        CFL condition: dt <= min(dz^2 / (2*D), dz / |v|)
        where D is diffusivity, v is advective velocity
        """
        # References: Richards (1931); Hillel (1998).
        # Mathematical expression: dt <= min(dz^2/(2D), dz/|v|).
        # Parameter meanings: D diffusivity [m2 s-1], v velocity [m s-1], dz layer thickness [m].
        # Units: dt [s]; D [m2 s-1] estimated as K h_char / (theta_s - theta_r); v [m s-1] taken as K.
        # Domain/constraints: theta_s > theta_r and K > 0 somewhere in the profile; the result is
        #   min(dt_stable, dt_requested) raised to minimum_timestep; a stability heuristic.
        # Sign convention: magnitudes only; the direction of flow is not used.
        # Compute hydraulic diffusivity D = K * dh/dtheta
        k = self._compute_hydraulic_conductivity(theta)
        
        # Diffusivity estimate for CFL control.
        # D ~= K * h_char / theta_char
        h_characteristic = self.characteristic_pressure_head  # m (characteristic pressure head scale)
        theta_characteristic = (
            on_device_of(self.theta_saturated, theta)
            - on_device_of(self.theta_residual, theta)
        )
        diffusivity = k * h_characteristic / theta_characteristic
        
        cfl_limit_t = self._ensure_tensor(cfl_limit).to(device=theta.device, dtype=theta.dtype)
        layer_thickness = self._ensure_tensor(self.layer_thickness).to(
            device=theta.device, dtype=theta.dtype
        )

        # Diffusion stability limit
        dt_diffusion = cfl_limit_t * (layer_thickness ** 2) / (2 * torch.max(diffusivity))
        
        # Advection stability limit (gravity term)
        gravity_velocity = k  # Approximate gravity-driven velocity
        dt_advection = cfl_limit_t * layer_thickness / torch.max(gravity_velocity)
        
        # Take minimum of all constraints
        dt_requested_t = self._ensure_tensor(dt_requested).to(device=theta.device, dtype=theta.dtype)
        dt_stable = torch.minimum(dt_diffusion, dt_advection)
        dt_stable = torch.minimum(dt_stable, dt_requested_t)
        min_dt = self._ensure_tensor(self.minimum_timestep).to(device=theta.device, dtype=theta.dtype)

        return torch.clamp(dt_stable, min=min_dt)
    
    def _solve_linearized_system_picard(self, 
                                      theta_old: Tensor,
                                      theta_iter: Tensor, 
                                      h_iter: Tensor,
                                      k_iter: Tensor,
                                      dt: float,
                                      surface_flux: float,
                                      bottom_flux: Optional[Tensor] = None) -> tuple[Tensor, bool]:
        """Delegate Picard solver to RichardsNumericalSolver."""
        # References:
        #   Celia et al. (1990); Kelley (2003).
        # Mathematical expression:
        #   F(theta)=theta-theta_old-dt*div_in(q(theta));
        #   J=dF/dtheta; Newton solves J*delta=-F; Picard solves its frozen linearization.
        # Parameter meanings:
        #   theta_old/iterate are layer contents; h and K are the same constitutive
        #   closures; boundary q and dt belong to this substep.
        # Units:
        #   F/theta/delta [m3 m-3], dt [s], q [m s-1], h/dz [m].
        # Domain/constraints:
        #   Delegate to the existing RichardsNumericalSolver; do not copy or
        #   re-implement its residual, Jacobian, linear solve, or convergence policy.
        # Sign convention:
        #   Positive div_in adds layer storage; the Newton correction cancels F.
        return self._numerical_solver._solve_linearized_system_picard(
            theta_old,
            theta_iter,
            h_iter,
            k_iter,
            dt,
            surface_flux,
            bottom_flux=bottom_flux,
        )

    def _solve_linearized_system_newton(self,
                                      theta_old: Tensor,
                                      theta_iter: Tensor,
                                      h_iter: Tensor, 
                                      k_iter: Tensor,
                                      dt: float,
                                      surface_flux: float,
                                      bottom_flux: Optional[Tensor] = None) -> tuple[Tensor, bool]:
        """Delegate Newton solver to RichardsNumericalSolver."""
        # References:
        #   Celia et al. (1990); Kelley (2003).
        # Mathematical expression:
        #   F(theta)=theta-theta_old-dt*div_in(q(theta));
        #   J=dF/dtheta; Newton solves J*delta=-F; Picard solves its frozen linearization.
        # Parameter meanings:
        #   theta_old/iterate are layer contents; h and K are the same constitutive
        #   closures; boundary q and dt belong to this substep.
        # Units:
        #   F/theta/delta [m3 m-3], dt [s], q [m s-1], h/dz [m].
        # Domain/constraints:
        #   Delegate to the existing RichardsNumericalSolver; do not copy or
        #   re-implement its residual, Jacobian, linear solve, or convergence policy.
        # Sign convention:
        #   Positive div_in adds layer storage; the Newton correction cancels F.
        return self._numerical_solver._solve_linearized_system_newton(
            theta_old,
            theta_iter,
            h_iter,
            k_iter,
            dt,
            surface_flux,
            bottom_flux=bottom_flux,
        )

    def _compute_newton_residual(self, 
                                theta: Tensor,
                                theta_old: Tensor,
                                h_iter: Tensor,
                                k_iter: Tensor,
                                dt: float,
                                surface_flux: float,
                                bottom_flux: Optional[Tensor] = None) -> Tensor:
        """Delegate Newton residual computation to RichardsNumericalSolver."""
        # References:
        #   Celia et al. (1990); Kelley (2003).
        # Mathematical expression:
        #   F(theta)=theta-theta_old-dt*div_in(q(theta));
        #   J=dF/dtheta; Newton solves J*delta=-F; Picard solves its frozen linearization.
        # Parameter meanings:
        #   theta_old/iterate are layer contents; h and K are the same constitutive
        #   closures; boundary q and dt belong to this substep.
        # Units:
        #   F/theta/delta [m3 m-3], dt [s], q [m s-1], h/dz [m].
        # Domain/constraints:
        #   Delegate to the existing RichardsNumericalSolver; do not copy or
        #   re-implement its residual, Jacobian, linear solve, or convergence policy.
        # Sign convention:
        #   Positive div_in adds layer storage; the Newton correction cancels F.
        return self._numerical_solver._compute_newton_residual(
            theta,
            theta_old,
            h_iter,
            k_iter,
            dt,
            surface_flux,
            bottom_flux=bottom_flux,
        )

    def _compute_newton_jacobian(self,
                                theta: Tensor,
                                theta_old: Tensor,
                                dt: float,
                                surface_flux: float,
                                bottom_flux: Optional[Tensor] = None) -> Tensor:
        """Delegate Newton Jacobian computation to RichardsNumericalSolver."""
        # References:
        #   Celia et al. (1990); Kelley (2003).
        # Mathematical expression:
        #   F(theta)=theta-theta_old-dt*div_in(q(theta));
        #   J=dF/dtheta; Newton solves J*delta=-F; Picard solves its frozen linearization.
        # Parameter meanings:
        #   theta_old/iterate are layer contents; h and K are the same constitutive
        #   closures; boundary q and dt belong to this substep.
        # Units:
        #   F/theta/delta [m3 m-3], dt [s], q [m s-1], h/dz [m].
        # Domain/constraints:
        #   Delegate to the existing RichardsNumericalSolver; do not copy or
        #   re-implement its residual, Jacobian, linear solve, or convergence policy.
        # Sign convention:
        #   Positive div_in adds layer storage; the Newton correction cancels F.
        return self._numerical_solver._compute_newton_jacobian(
            theta,
            theta_old,
            dt,
            surface_flux,
            bottom_flux=bottom_flux,
        )

    def _compute_dh_dtheta(self, theta: Tensor) -> Tensor:
        """
        Compute derivative dh/dtheta using Van Genuchten model.
        
        Mathematical Expression
        -----------------------
        dh/dtheta = dh/dSe * dSe/dtheta
        where Se = (theta - theta_r)/(theta_s - theta_r)
        and h = -1/alpha * [(Se^(-1/m) - 1)]^(1/n)
        """
        # References:
        #   van Genuchten (1980); Mualem (1976).
        # Mathematical expression:
        #   dh/dtheta=(dh/dSe)/(theta_s-theta_r);
        #   dK/dtheta=(dK/dSe)/(theta_s-theta_r), differentiating the retained closures.
        # Parameter meanings:
        #   Se is effective saturation; alpha,n,m,l,Ks are the same hydraulic
        #   parameters used by the primal pressure-head and conductivity functions.
        # Units:
        #   dh/dtheta [m]; dK/dtheta [m s-1] per unit volumetric water fraction.
        # Domain/constraints:
        #   These helpers use the existing endpoint regularizers; do not advertise
        #   a global smooth derivative through clipping boundaries.
        # Sign convention:
        #   For the admissible unsaturated branch, wetter soil raises h and K.
        theta_residual = on_device_of(self.theta_residual, theta)
        theta_saturated = on_device_of(self.theta_saturated, theta)
        alpha_vg = on_device_of(self.alpha_vg, theta)
        n_vg = on_device_of(self.n_vg, theta)
        m_vg = on_device_of(self.m_vg, theta)
        se = torch.clamp(
            (theta - theta_residual) / (theta_saturated - theta_residual),
            min=self.saturation_epsilon, max=1.0
        )
        
        # dSe/dtheta
        dse_dtheta = 1.0 / (theta_saturated - theta_residual)
        
        # dh/dSe using Van Genuchten relation
        # h = -1/alpha * [(Se^(-1/m) - 1)]^(1/n)
        se_term = torch.pow(se, -1.0/m_vg) - 1.0
        min_se_term = self.minimum_se_term
        se_term = torch.clamp(se_term, min=min_se_term)  # Avoid numerical issues
        
        # Chain rule: dh/dSe = dh/d(se_term) * d(se_term)/dSe
        dh_dse_term = -1.0/alpha_vg * (1.0/n_vg) * torch.pow(se_term, (1.0/n_vg - 1.0))
        dse_term_dse = (-1.0/m_vg) * torch.pow(se, (-1.0/m_vg - 1.0))
        
        dh_dse = dh_dse_term * dse_term_dse
        
        # Final derivative
        dh_dtheta = dh_dse * dse_dtheta
        
        return dh_dtheta
    
    def _compute_dk_dtheta(self, theta: Tensor) -> Tensor:
        """
        Compute derivative dK/dtheta using Mualem model.
        
        Mathematical Expression
        -----------------------
        K(theta) = Ks * Se^l * [1 - (1 - Se^(1/m))^m]^2
        dK/dtheta = dK/dSe * dSe/dtheta
        """
        # References:
        #   van Genuchten (1980); Mualem (1976).
        # Mathematical expression:
        #   dh/dtheta=(dh/dSe)/(theta_s-theta_r);
        #   dK/dtheta=(dK/dSe)/(theta_s-theta_r), differentiating the retained closures.
        # Parameter meanings:
        #   Se is effective saturation; alpha,n,m,l,Ks are the same hydraulic
        #   parameters used by the primal pressure-head and conductivity functions.
        # Units:
        #   dh/dtheta [m]; dK/dtheta [m s-1] per unit volumetric water fraction.
        # Domain/constraints:
        #   These helpers use the existing endpoint regularizers; do not advertise
        #   a global smooth derivative through clipping boundaries.
        # Sign convention:
        #   For the admissible unsaturated branch, wetter soil raises h and K.
        theta_residual = on_device_of(self.theta_residual, theta)
        theta_saturated = on_device_of(self.theta_saturated, theta)
        tortuosity = on_device_of(self.tortuosity, theta)
        m_vg = on_device_of(self.m_vg, theta)
        k_saturated = on_device_of(self.k_saturated, theta)
        se = torch.clamp(
            (theta - theta_residual) / (theta_saturated - theta_residual),
            min=self.saturation_epsilon, max=1.0
        )
        
        # dSe/dtheta
        dse_dtheta = 1.0 / (theta_saturated - theta_residual)
        
        # Compute terms for Mualem model
        se_power = torch.pow(se, 1.0/m_vg)
        term1 = 1.0 - se_power
        term2 = torch.pow(torch.clamp(term1, min=0.0), m_vg)
        term3 = 1.0 - term2
        
        # dK/dSe using product rule and chain rule
        # K = Ks * Se^l * [1 - (1 - Se^(1/m))^m]^2
        
        # First term: d/dSe[Se^l] = l * Se^(l-1)
        dterm1_dse = tortuosity * torch.pow(se, tortuosity - 1.0)
        
        # Second term: d/dSe[[1 - (1 - Se^(1/m))^m]^2]
        # Chain rule: 2 * [1 - (1 - Se^(1/m))^m] * d/dSe[1 - (1 - Se^(1/m))^m]
        #           = 2 * term3 * (-1) * m * (1 - Se^(1/m))^(m-1) * (-1/m) * Se^(1/m - 1)
        #           = 2 * term3 * Se^(1/m - 1) * (1 - Se^(1/m))^(m-1)
        
        if m_vg != 1.0:  # Avoid division by zero
            se_power_deriv = (1.0/m_vg) * torch.pow(se, (1.0/m_vg - 1.0))
            term1_deriv = -se_power_deriv
            min_term1 = self.minimum_term1
            term2_deriv = m_vg * torch.pow(torch.clamp(term1, min=min_term1), (m_vg - 1.0)) * term1_deriv
            term3_deriv = -term2_deriv
            dterm2_dse = 2.0 * term3 * term3_deriv
        else:
            dterm2_dse = torch.zeros_like(se)
        
        # Combine using product rule: d/dSe[f * g] = f' * g + f * g'
        se_power_l = torch.pow(se, tortuosity)
        term3_squared = torch.pow(term3, 2.0)
        
        dk_dse = k_saturated * (dterm1_dse * term3_squared + se_power_l * dterm2_dse)
        
        # Final derivative
        dk_dtheta = dk_dse * dse_dtheta
        
        return dk_dtheta
    
    def _solve_newton_linear_system(self, jacobian: Tensor, rhs: Tensor) -> Tensor:
        """Delegate Newton linear solver to RichardsNumericalSolver."""
        # References: Kelley (2003); Press et al. (2007).
        # Mathematical expression: J * delta = rhs (linearized Newton system).
        # Parameter meanings: jacobian J [dimensionless], rhs [dimensionless].
        return self._numerical_solver._solve_newton_linear_system(jacobian, rhs)

    def _is_nearly_tridiagonal(self, matrix: Tensor, tol: float = None) -> bool:
        """Delegate tridiagonal check to RichardsNumericalSolver."""
        # References: Press et al. (2007) Numerical Recipes.
        # Mathematical expression: |a_ij| <= tol for |i-j| > 1 (tridiagonal check).
        # Parameter meanings: matrix [dimensionless], tol [dimensionless].
        return self._numerical_solver._is_nearly_tridiagonal(matrix, tol=tol)

    def _newton_line_search(self,
                           theta_current: Tensor,
                           delta_theta: Tensor,
                           theta_old: Tensor,
                           dt: float,
                           surface_flux: float,
                           residual_norm_old: float,
                           bottom_flux: Optional[Tensor] = None) -> float:
        """Delegate Newton line search to RichardsNumericalSolver."""
        # References: Nocedal & Wright (2006); Kelley (2003).
        # Mathematical expression: alpha = argmin ||F(theta + alpha * delta)|| with Armijo rule.
        # Parameter meanings: theta_current/theta_old [m3 m-3], delta_theta [m3 m-3], dt [s], surface_flux [m s-1].
        return self._numerical_solver._newton_line_search(
            theta_current,
            delta_theta,
            theta_old,
            dt,
            surface_flux,
            residual_norm_old,
            bottom_flux=bottom_flux,
        )

    def _compute_inter_layer_fluxes(self, theta: Tensor, h: Tensor, k: Tensor) -> Tensor:
        """
        Compute water fluxes between soil layers using Darcy's law.
        
        Parameters
        ----------
        theta : Tensor
            Soil moisture profile [m3 m-3]
        h : Tensor
            Pressure head profile [m]
        k : Tensor
            Hydraulic conductivity profile [m s-1]
            
        Returns
        -------
        Tensor
            Inter-layer fluxes [m s-1], shape [n_layers + 1]
            Positive flux is downward
            
        Mathematical Expression
        -----------------------
        Darcy flux: q = -K * (dh/dz - 1)
        Interface conductivity: K_interface = harmonic_mean(K_i, K_{i+1})
        """
        # References:
        #   Darcy (1856); Richards (1931); van Genuchten (1980); Belfort et al.
        #   (2013), J. Hydrol. 505:202-217, Table 2 (harmonic interblock mean).
        # Mathematical expression:
        #   Kface=f*K_i*K_j/(K_i+K_j), Kface=0 if K_i=K_j=0, evaluated by the
        #   solver helper harmonic_interface_conductivity that also builds the
        #   residual and the Picard matrix; the former literal 2.0 is replaced
        #   by the configured factor f (identical at the default f=2);
        #   qface=-Kface*((h_j-h_i)/dz-1);
        #   Icap=Ksurface*(1+(theta_s-theta_surface)/(theta_s-theta_r)).
        # Parameter meanings:
        #   Neighbor values define harmonic interface conductivity; f is
        #   hydrosphere.numerical.harmonic_mean_factor. No constant in m s-1 is
        #   added to K_i+K_j. Icap is the configured surface diagnostic.
        # Units:
        #   Kface/qface/Icap [m s-1], h/dz [m], f and moisture ratios [1].
        # Domain/constraints:
        #   Interior faces use the one-dimensional depth ordering. The provisional
        #   outer faces are overwritten/reconciled by the selected boundary conditions
        #   in the full solution; they are not independent default physical ports.
        # Sign convention:
        #   Positive interface flow is downward; negative flow is upward.
        n = self.soil_layers
        dz = on_device_of(self.layer_thickness, theta)
        fluxes = theta.new_zeros(n + 1)
        
        # Surface flux (boundary condition)
        fluxes[0] = 0.0  # Will be set by boundary conditions
        
        # Inter-layer fluxes
        k_faces = harmonic_interface_conductivity(k, self.harmonic_mean_factor)
        for i in range(n - 1):
            # Interface hydraulic conductivity (harmonic mean, see above)
            k_interface = k_faces[i]
            
            # Pressure gradient
            dh_dz = (h[i+1] - h[i]) / dz
            
            # Darcy flux (positive downward)
            fluxes[i+1] = -k_interface * (dh_dz - 1.0)
        
        # Bottom flux (boundary condition)
        if self._bottom_boundary is not None and self._bottom_boundary["type"] == "free_drainage":
            # Free drainage has zero pressure-head gradient.
            # References: Celia et al. (1990).
            # Mathematical expression: q_bottom = -K(0 - 1) = K.
            # Parameter meanings: q_bottom and K [m s-1], positive downward.
            fluxes[n] = k[n-1]
        else:
            # The canonical default is the same downward free-drainage closure.
            fluxes[n] = k[n-1]
        
        return fluxes
    
    def _check_mass_conservation(self, theta_old: Tensor, theta_new: Tensor,
                               surface_flux: float, bottom_flux: float, dt: float) -> Tensor:
        """
        Check mass conservation for the Richards equation solution.
        
        Parameters
        ----------
        theta_old : Tensor
            Previous moisture content [m3 m-3]
        theta_new : Tensor
            New moisture content [m3 m-3]
        surface_flux : float
            Surface flux [m s-1]
        bottom_flux : float
            Bottom flux [m s-1]  
        dt : float
            Time step [s]
            
        Returns
        -------
        Tensor
            Relative mass conservation error [dimensionless]
            
        Mathematical Expression
        -----------------------
        Mass balance: d(theta)/dt = (flux_in - flux_out)/dz
        Conservation error = |sum(d_theta) - net_flux_dt| / |sum(theta_old)|
        """
        # References: Richards (1931); mass continuity for soil water.
        # Mathematical expression: error = |delta_theta - net_flux| / total_water.
        # Parameter meanings: theta [m3 m-3], surface_flux/bottom_flux [m s-1], dz [m], dt [s].
        # Units: delta_theta, net_flux and total_water are water depths [m]; error [1].
        # Domain/constraints: uniform dz; total_water > 0, otherwise ValueError.
        # Sign convention: surface_flux positive into the profile and bottom_flux positive out
        #   of it, net_flux = (surface_flux - bottom_flux) dt; the error is a magnitude.
        dz = on_device_of(self.layer_thickness, theta_new)
        
        # Total moisture change
        delta_theta = torch.sum(theta_new - theta_old) * dz  # [m]
        
        # Net flux into domain
        net_flux = (surface_flux - bottom_flux) * dt  # [m]
        
        # Conservation error
        total_water = torch.sum(theta_old) * dz
        if total_water <= 0:
            raise ValueError("Richards mass ledger requires positive initial water storage")
        conservation_error = torch.abs(delta_theta - net_flux) / total_water
        
        return conservation_error
    
    def _compute_infiltration_capacity(self, surface_theta: float, surface_k: float) -> float:
        """
        Compute maximum infiltration capacity at soil surface.
        
        Parameters
        ----------
        surface_theta : float
            Surface moisture content [m3 m-3]
        surface_k : float
            Surface hydraulic conductivity [m s-1]
            
        Returns
        -------
        float
            Infiltration capacity [m s-1]
        """
        # References: Richards (1931); Green and Ampt (1911) infiltration capacity concept.
        # Mathematical expression: infil_cap = K_surface * (1 + saturation_deficit).
        # Parameter meanings: surface_theta [m3 m-3], surface_k [m s-1].
        # Units: infil_cap and surface_k [m s-1]; saturation_deficit [1].
        # Domain/constraints: theta_r <= surface_theta <= theta_s gives a deficit in [0, 1] and
        #   a capacity in [K_surface, 2 K_surface]; an approximation, not the Green-Ampt solution.
        # Sign convention: a positive capacity is a downward flux into the soil.
        # Simple approximation: infiltration capacity is related to conductivity
        # and degree of saturation
        theta_residual = on_device_of(self.theta_residual, surface_theta)
        theta_saturated = on_device_of(self.theta_saturated, surface_theta)
        saturation_deficit = (theta_saturated - surface_theta) / (theta_saturated - theta_residual)
        infiltration_capacity = surface_k * (1.0 + saturation_deficit)
        
        return infiltration_capacity
    
    def _compute_theta_from_head(
        self,
        h: Union[float, Tensor],
    ) -> Tensor:
        """
        Compute moisture content from pressure head using Van Genuchten model.
        
        Parameters
        ----------
        h : float or Tensor
            Pressure head [m]
            
        Returns
        -------
        Tensor
            Moisture content [m3 m-3] on the input or parameter device.
        """
        # References:
        #   van Genuchten (1980); Mualem (1976).
        # Mathematical expression:
        #   Se=clip((theta-theta_r)/(theta_s-theta_r),eps,1);
        #   h=-(Se^(-1/m)-1)^(1/n)/alpha;
        #   K=Ks*Se^l*[1-(1-Se^(1/m))^m]^2; m=1-1/n;
        #   theta(h)=theta_r+(theta_s-theta_r)*(1+(alpha*abs(h))^n)^(-m).
        # Parameter meanings:
        #   theta_r/theta_s bound water content; alpha,n,m describe retention;
        #   l is tortuosity and Ks is saturated conductivity. Nonnegative head
        #   returns saturated water content in the inverse function.
        # Units:
        #   theta/Se/n/m/l [1], h [m], alpha [m-1], K/Ks [m s-1].
        # Domain/constraints:
        #   Use configured endpoint floors for the retention inversions; require
        #   theta_s>theta_r, alpha>0, n>1 and positive Ks. These are constitutive
        #   properties, not an additional state advance.
        # Sign convention:
        #   Unsaturated h is nonpositive; K is a nonnegative conductance magnitude.
        if isinstance(h, Tensor):
            head = h
        else:
            reference = self._ensure_tensor(self.theta_saturated)
            head = torch.as_tensor(
                h,
                device=reference.device,
                dtype=reference.dtype,
            )
        theta_residual = on_device_of(self.theta_residual, head)
        theta_saturated = on_device_of(self.theta_saturated, head)
        alpha_vg = on_device_of(self.alpha_vg, head)
        n_vg = on_device_of(self.n_vg, head)
        m_vg = on_device_of(self.m_vg, head)
        term = (alpha_vg * torch.abs(head)) ** n_vg
        effective_saturation = (1.0 + term) ** (-m_vg)
        unsaturated_theta = (
            theta_residual
            + (theta_saturated - theta_residual)
            * effective_saturation
        )
        saturated_theta = torch.as_tensor(
            theta_saturated,
            device=head.device,
            dtype=head.dtype,
        )
        return torch.where(head >= 0.0, saturated_theta, unsaturated_theta)

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
__all__ = ["RichardsSoilMoisture"]
