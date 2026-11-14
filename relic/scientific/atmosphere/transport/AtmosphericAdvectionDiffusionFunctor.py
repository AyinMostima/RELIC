"""
Atmospheric Advection-Diffusion Transport Functor

This module implements atmospheric transport processes including advection and diffusion
for scalar quantities in the atmosphere. It follows the sphere-based organization principle
and provides specialized implementations for atmospheric transport scenarios.

Mathematical Expression:
    Atmospheric transport equation: dc/dt + grad*(v c) = grad*(K gradc) + S
    where:
    - c is concentration [kg m-3]
    - v is wind velocity [m s-1]
    - K is eddy diffusivity tensor [m2 s-1]
    - S is source/sink term [kg m-3 s-1]

References:
    - Stull, R.B. (1988). An Introduction to Boundary Layer Meteorology. Kluwer.
    - Jacobson, M.Z. (2005). Fundamentals of Atmospheric Modeling. Cambridge University Press.
    - Seinfeld, J.H. & Pandis, S.N. (2016). Atmospheric Chemistry and Physics. Wiley.
"""

from typing import Dict, Mapping, Optional, Tuple, Any
import math
import torch
import logging

from ....computation.operators.dec import CochainSpaceSpec, HodgeStarSpec
from ....computation.operators.functional import pack_outward_boundary_flux_cochain
from ....computation.operators.iphs import PortSpecIPHS
from ....core.functor import (
    DependencyDeclaration,
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from ....core.functor import TransportMechanismFunctor
from ....core.state import UpdateItem, UpdateMode
from relic.utils.physical_constants import get_physical_constant
from ....utils.unified_response_functions import validate_unity_range
from ...hydrosphere.utils.psychrometrics import (
    compute_moist_air_density,
    compute_saturation_vapor_pressure,
)
from ..most import monin_obukhov_length, richardson_number, zeta
# Import unified transport operators for spatial calculations
from ....computation.operators import OperatorFactory

from ...transport.boundary_conditions import (
    BoundaryCondition,
    BoundaryConditionManager,
    BoundaryType,
)

logger = logging.getLogger(__name__)

# Type aliases
Tensor = torch.Tensor
StateDict = Dict[str, Tensor]
EnvironmentDict = Dict[str, Tensor]
FluxDict = Dict[str, Tensor]


def _to_float(value: Any) -> float:
    """Convert parameter manager outputs to Python float."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().reshape(-1)[0].tolist())
    return float(value)


class AtmosphericAdvectionDiffusionFunctor(TransportMechanismFunctor):
    """
    Atmospheric advection-diffusion transport functor.
    
    Implements combined advection and diffusion processes for atmospheric
    transport of scalar quantities (e.g., pollutants, tracers, BVOCs).
    Uses atmospheric-specific parameterizations and boundary conditions.
    
    Mathematical Expression:
        dc/dt + grad*(v c) = grad*(K gradc) + S
        
        where K = K_molecular + K_eddy
        K_eddy depends on atmospheric stability and turbulence
        
    Parameters:
        spatial_dims : int
            Number of spatial dimensions, 2 or 3 (required; no default)
        grid_spacing : Tuple[float, ...]
            Physical cell widths (dx, dy[, dz]) [m]; canonical operators
            receive the reversed tensor storage order internally
        stability_scheme : str
            Atmospheric stability parameterization: "monin-obukhov", "richardson" or
            "constant" (required keyword; no default)
        device : Optional[torch.device]
            Computation device
    """

    TRANSPORT_TYPE = "advection_diffusion"
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'
    EXTRA_STATE_FORMAT = "relic.atmospheric_advection_diffusion.boundary_descriptors"
    EXTRA_STATE_VERSION = 1
    _FACE_ORDER = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
    _CHECKPOINTABLE_BOUNDARY_TYPES = (
        BoundaryType.NEUMANN,
        BoundaryType.PERIODIC,
        BoundaryType.DIRICHLET,
    )
    
    def __init__(
        self,
        spatial_dims: int,
        grid_spacing: Optional[Tuple[float, ...]] = None,
        *,
        stability_scheme: str,
        boundary_layer_height: Optional[float] = None,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> None:
        """Initialize atmospheric advection-diffusion functor."""
        config = kwargs.get('config', {})

        # Set up metadata
        metadata = FunctorMetadata(
            name="atmospheric_advection_diffusion",
            version="1.0.0",
            description="Atmospheric advection-diffusion transport with stability-dependent eddy diffusivity",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="dc/dt + grad*(v c) = grad*(K gradc) + S",
            references=[
                "Stull, R.B. (1988). An Introduction to Boundary Layer Meteorology.",
                "Jacobson, M.Z. (2005). Fundamentals of Atmospheric Modeling.",
                "Seinfeld, J.H. & Pandis, S.N. (2016). Atmospheric Chemistry and Physics."
            ],
            parameters={
                "atmosphere.grid.dx": "Grid spacing in the x direction [m]",
                "atmosphere.grid.dy": "Grid spacing in the y direction [m]",
                "atmosphere.grid.dz": "Grid spacing in the vertical direction [m]",
                "atmosphere.boundary_layer.boundary_layer_height": (
                    "Atmospheric boundary layer height [m]"
                ),
                "atmosphere.turbulence.eddy_diffusivity_max": "Maximum eddy diffusivity [m2 s-1]",
                "atmosphere.turbulence.transport_neutral_diffusivity_fraction": (
                    "Neutral fraction K0 / K_max of the transport eddy-diffusivity closure "
                    "[dimensionless]"
                ),
                "atmosphere.turbulence.transport_unstable_enhancement_coefficient": (
                    "Coefficient a of K = K0 (1 + a s / (1 + s)) for unstable stratification "
                    "[dimensionless]"
                ),
                "atmosphere.turbulence.transport_stable_suppression_coefficient": (
                    "Coefficient b of K = K0 / (1 + b s) for stable stratification "
                    "[dimensionless]"
                ),
                "atmosphere.turbulence.transport_stability_limit": (
                    "Symmetric bound of the stability index of the transport closure "
                    "[dimensionless]"
                ),
                "atmosphere.turbulence.minimum_friction_velocity": (
                    "Minimum friction velocity [m s-1]"
                ),
                "atmosphere.stability.potential_temperature_gradient": (
                    "Potential-temperature gradient [K m-1]"
                ),
                "atmosphere.molecular_diffusion.molecular_diffusivity": (
                    "Molecular diffusivity of the scalar in air [m2 s-1]"
                ),
                "transport.numerical.cfl_advection_max": (
                    "Largest accepted advective Courant number [dimensionless]"
                ),
                "transport.numerical.cfl_diffusion_max": (
                    "Largest accepted diffusion number (von Neumann) [dimensionless]"
                ),
            },
            input_variables=[
                "tracer_concentration",
                "wind_u", "wind_v", "wind_w",
                "temperature", "sensible_heat_flux",
                "atmospheric_pressure", "relative_humidity",
                "source_term",
            ],
            output_variables=[
                "tracer_concentration",
                "diagnostics.atmospheric_advection_diffusion.cfl_number",
                "diagnostics.stability_number",
                "diagnostics.advection_rate",
                "diagnostics.diffusion_rate",
                "diagnostics.eddy_diffusivity",
                "diagnostics.stability_parameter",
                "diagnostics.conservation_error",
                "diagnostics.boundary_flux_cochain",
            ],
            optional_output_variables=["diagnostics.boundary_flux_cochain"],
            conservation_laws=["mass"],
            spatial_dimensions=spatial_dims,
            temporal_scale=TemporalScale.SECOND,
            coupling_interfaces=["atmosphere.wind", "atmosphere.turbulence"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
        )
        
        super().__init__(name="atmospheric_advection_diffusion", config=config, metadata=metadata)
        
        # Validate inputs
        if spatial_dims not in [2, 3]:
            raise ValueError("Atmospheric models typically use 2D or 3D")
        if stability_scheme not in ["monin-obukhov", "richardson", "constant"]:
            raise ValueError("Invalid stability scheme")
        
        self.spatial_dims = spatial_dims
        self.stability_scheme = stability_scheme
        # The monin-obukhov scheme reads the air temperature T, the surface
        # upward sensible heat flux H, and the pressure p and relative humidity
        # RH of the moist-air density in the Obukhov length; the richardson and
        # constant schemes read none of them, so they are optional inputs there.
        self.optional_input_variables = (
            set()
            if stability_scheme == "monin-obukhov"
            else {
                "temperature", "sensible_heat_flux",
                "atmospheric_pressure", "relative_humidity",
            }
        )
        # A 2-D field reads the horizontal wind components only.
        if spatial_dims == 2:
            self.optional_input_variables = self.optional_input_variables | {"wind_w"}
        # The volumetric source port is optional: its absence is the closed
        # no-source case (AdvectionDiffusion._prepare_source_term).
        self.optional_input_variables = self.optional_input_variables | {"source_term"}
        self.gravity_acceleration = _to_float(
            get_physical_constant("gravity_acceleration")
        )
        self.reference_temperature = _to_float(
            get_physical_constant("reference_temperature")
        )
        self.von_karman_constant = _to_float(
            get_physical_constant("von_karman_constant")
        )
        self.air_specific_heat = _to_float(get_physical_constant("air_specific_heat_cp"))
        # Psychrometric SSOT constants of the saturation vapour pressure in the
        # moist-air density of the monin-obukhov scheme (no fixed air density).
        self._tetens_constants = {
            "tetens_base_pressure": _to_float(get_physical_constant(
                "relic.physical.constants.psychrometrics.tetens_base_pressure")),
            "tetens_coeff_a": _to_float(get_physical_constant(
                "relic.physical.constants.psychrometrics.tetens_coefficient_a")),
            "tetens_coeff_b": _to_float(get_physical_constant(
                "relic.physical.constants.psychrometrics.tetens_coefficient_b")),
            "kelvin_offset": _to_float(get_physical_constant("celsius_to_kelvin")),
        }
        self._mixing_coefficients = {
            name: _to_float(self.get_parameter_value(
                name, namespace="atmosphere.turbulence", source_order=("manager",), required=True,
            ))
            for name in (
                "transport_neutral_diffusivity_fraction",
                "transport_unstable_enhancement_coefficient",
                "transport_stable_suppression_coefficient",
                "transport_stability_limit",
            )
        }
        
        # References: LeVeque (2002), axis-aligned finite-volume grids.
        # Mathematical expression: h_physical=(dx,dy[,dz]); h_storage=reverse(h_physical).
        # Parameter meanings: dx,dy,dz are the physical x,y,z cell widths; the
        #   public constructor and ParameterManager keys use physical order.
        # Units: every width is in meters [m].
        # Domain/constraints: finite positive widths, exactly one per axis;
        #   tensors are stored as (batch, [z,] y, x), so canonical operators
        #   receive the reversed tuple inside forward.
        # Sign convention: widths are positive lengths along +x,+y,+z.
        if grid_spacing is None:
            grid_spacing = tuple(
                _to_float(
                    self.get_parameter_value(
                        axis,
                        namespace='atmosphere.grid',
                        source_order=("manager",), required=True)
                )
                for axis in ("dx", "dy", "dz")[:spatial_dims]
            )
        self.grid_spacing = tuple(_to_float(val) for val in grid_spacing)
        if len(self.grid_spacing) != spatial_dims:
            raise ValueError(
                "grid_spacing must provide exactly one physical width per spatial axis"
            )
        if any(not math.isfinite(width) or width <= 0.0 for width in self.grid_spacing):
            raise ValueError("grid_spacing widths must be finite and positive")
        
        # Atmospheric parameters
        if boundary_layer_height is None:
            boundary_layer_height = self.get_parameter_value(
                'boundary_layer_height',
                namespace='atmosphere.boundary_layer',
                source_order=("manager",), required=True)
        self.boundary_layer_height = _to_float(boundary_layer_height)

        eddy_diffusivity_max = self.get_parameter_value(
            'eddy_diffusivity_max',
            namespace='atmosphere.turbulence',
            source_order=("manager",), required=True)
        self.eddy_diffusivity_max = _to_float(eddy_diffusivity_max)

        molecular_diffusivity = self.get_parameter_value(
            'molecular_diffusivity',
            namespace='atmosphere.molecular_diffusion',
            source_order=("manager",), required=True)
        self.molecular_diffusivity = _to_float(molecular_diffusivity)

        # Numerical limits: the explicit step is bounded before the update by
        # transport.numerical.cfl_advection_max and cfl_diffusion_max (forward).
        
        # Initialize boundary condition manager
        self.boundary_manager = BoundaryConditionManager(spatial_dims, self.device)
        self.boundary_manager.initialize()
        self.boundary_manager.validate()
        if device is not None:
            self.to_device(device)

    def to_device(
        self,
        device: torch.device | str,
    ) -> "AtmosphericAdvectionDiffusionFunctor":
        """Move the Functor and its nested boundary owner together."""
        super().to_device(device)
        if hasattr(self, "boundary_manager"):
            self.boundary_manager.to_device(self.device)
        return self

    def get_dependencies(self) -> DependencyDeclaration:
        """Declare the optional same-step producer of the surface sensible heat flux.

        atmospheric_turbulence writes sensible_heat_flux; when it is registered
        it runs before this Functor in the same step, otherwise the monin-obukhov
        scheme needs sensible_heat_flux as explicit forcing.
        """
        return self._merge_declared_dependencies({"optional": ["atmospheric_turbulence"]})

    def declared_forms(self) -> tuple[CochainSpaceSpec, ...]:
        """Declare atmospheric tracer transport cochain spaces.

        References:
          - Stull (1988), boundary-layer tracer transport.
          - Hirani (2003), Discrete Exterior Calculus.
        Mathematical expression:
          c_dot = -D_h q_h + G_h u_h + s_h,
          q_h = v_h c_h - star_K d_h c_h.
        Parameter meanings:
          spatial_dims selects the number of gradient and flux components;
          tracer_concentration is the transported concentration cochain.
        """

        return (
            CochainSpaceSpec(
                complex_id="atmospheric_advection_diffusion",
                degree=0,
                primality="primal",
                n_dofs=1,
                units="kg m-3",
                extensivity="intensive",
                physical_domain="atmosphere",
                semantic_name="tracer_concentration",
                orientation_convention="outward_positive",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="atmospheric_advection_diffusion",
                degree=1,
                primality="primal",
                n_dofs=self.spatial_dims,
                units="kg m-4",
                extensivity="intensive",
                physical_domain="atmosphere",
                semantic_name="atmospheric_tracer_gradient",
                orientation_convention="outward_positive",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="atmospheric_advection_diffusion",
                degree=2,
                primality="dual",
                n_dofs=self.spatial_dims,
                units="kg m-2 s-1",
                extensivity="integral",
                physical_domain="atmosphere",
                semantic_name="atmospheric_tracer_flux",
                orientation_convention="outward_positive",
                boundary_role="boundary",
            ),
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Declare ParameterManager keys used by runtime and structural builders."""

        keys = [
            "atmosphere.grid.dx",
            "atmosphere.grid.dy",
            "atmosphere.boundary_layer.boundary_layer_height",
            "atmosphere.turbulence.eddy_diffusivity_max",
            "atmosphere.turbulence.transport_neutral_diffusivity_fraction",
            "atmosphere.turbulence.transport_unstable_enhancement_coefficient",
            "atmosphere.turbulence.transport_stable_suppression_coefficient",
            "atmosphere.turbulence.transport_stability_limit",
            "atmosphere.turbulence.minimum_friction_velocity",
            "atmosphere.stability.potential_temperature_gradient",
            "atmosphere.molecular_diffusion.molecular_diffusivity",
            "transport.numerical.cfl_advection_max",
            "transport.numerical.cfl_diffusion_max",
        ]
        if self.spatial_dims == 3:
            keys.insert(2, "atmosphere.grid.dz")
        return tuple(keys)

    def build_hodge_contract(
        self,
        n_dofs: Optional[int] = None,
        *,
        strict: bool = True,
    ) -> HodgeStarSpec:
        """Build the atmospheric tracer diffusion Hodge contract.

        References:
          - Stull (1988), K-theory eddy diffusion closure.
          - Hirani (2003), Discrete Exterior Calculus.
        Mathematical expression:
          q_diff_h = -star_K d_h c_h with star_K = diag(K) and K >= 0.
        Parameter meanings:
          molecular_diffusivity and eddy_diffusivity_max are ParameterManager
          values that bound the passive diffusion mobility.
        """

        # References: Stull (1988), K-theory; Hirani (2003), material Hodge maps.
        # Mathematical expression: K_nom=max(K_molecular+K_eddy_max,0), star=K_nom*I.
        # Parameter meanings: n_local counts directional audit coordinates;
        #   K_nom is an upper-bound material declaration, not runtime K(zeta).
        # Units: gradient [kg m-4], matrix [m2 s-1], flux [kg m-2 s-1].
        # Domain/constraints: positive n_local; this nominal matrix alone does
        #   not establish equivalence to the executed grid transport operator.
        # Sign convention: the associated constitutive diffusive flux is -star*grad(c).
        n_local = int(n_dofs) if n_dofs is not None else self.spatial_dims
        if n_local < 1:
            raise ValueError("Atmospheric Hodge degrees of freedom must be positive.")
        nominal_diffusivity = max(
            self.molecular_diffusivity + self.eddy_diffusivity_max,
            0.0,
        )
        values = torch.full(
            (n_local,),
            float(nominal_diffusivity),
            device=self.device,
            dtype=torch.get_default_dtype(),
        )
        gradient_space = CochainSpaceSpec(
            complex_id="atmospheric_advection_diffusion",
            degree=1,
            primality="primal",
            n_dofs=n_local,
            units="kg m-4",
            extensivity="intensive",
            physical_domain="atmosphere",
            semantic_name="atmospheric_tracer_gradient",
            orientation_convention="outward_positive",
        )
        flux_space = CochainSpaceSpec(
            complex_id="atmospheric_advection_diffusion",
            degree=2,
            primality="dual",
            n_dofs=n_local,
            units="kg m-2 s-1",
            extensivity="integral",
            physical_domain="atmosphere",
            semantic_name="atmospheric_tracer_flux",
            orientation_convention="outward_positive",
            boundary_role="boundary",
        )
        hodge = HodgeStarSpec(
            name=f"{self.name}_tracer_hodge",
            domain=gradient_space,
            codomain=flux_space,
            matrix=torch.diag(values),
            parameter_keys=(
                "atmosphere.molecular_diffusion.molecular_diffusivity",
                "atmosphere.turbulence.eddy_diffusivity_max",
            ),
            metric_keys=tuple(f"atmosphere.grid.d{axis}" for axis in "xyz"[: self.spatial_dims]),
            material_law="atmospheric_k_theory_tracer_diffusion",
        )
        if strict:
            hodge.validate(passive=True)
        return hodge

    def build_hodge_terms(
        self,
        complex_bundle: Any = None,
        resolved_parameters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[HodgeStarSpec, ...]:
        """Build atmospheric Hodge terms for Phase 8 IPHS assembly."""

        n_dofs = getattr(complex_bundle, "n_dofs", None) if complex_bundle is not None else None
        return (self.build_hodge_contract(n_dofs=n_dofs, strict=True),)

    def energy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return structural tracer storage-effort audit terms."""

        # References: Hirani (2003), finite cochain quadratic forms.
        # Mathematical expression: H_proxy=sum(c_i**2)/2, e_proxy=c.
        # Parameter meanings: c is the flattened tracer concentration.
        # Units: H_proxy [kg2 m-6], e_proxy [kg m-3]; these are not joules.
        # Domain/constraints: structural unweighted proxy, not a thermodynamic
        #   storage or a governed strict-process energy certificate.
        # Sign convention: H_proxy is nonnegative.
        if "tracer_concentration" not in state:
            return {
                "storage_functional": "H_h = 0.5 * c_h.T c_h",
                "effort": "e_h = delta H_h / delta c_h = c_h",
            }
        concentration = self._as_tensor(state["tracer_concentration"])
        vector = concentration.reshape(-1)
        return {
            "storage_functional": 0.5 * torch.sum(vector * vector),
            "effort": concentration,
        }

    def entropy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return a structural nonnegative quadratic proxy, not physical entropy."""

        # References: Hirani (2003), positive material quadratic forms.
        # Mathematical expression: r=c[:n]-mean(c), Q_proxy=r.T*star_nom*r;
        #   r=0 when fewer than n=spatial_dims concentration values exist.
        # Parameter meanings: r is a concentration deviation, not a spatial
        #   derivative; star_nom is the nominal material audit matrix.
        # Units: r [kg m-3], star_nom [m2 s-1], Q_proxy [kg2 m-4 s-1].
        # Domain/constraints: the Hodge audit dtype is used before differences;
        #   no thermodynamic entropy or physical production-rate claim follows.
        # Sign convention: the positive quadratic form is nonnegative.
        if "tracer_concentration" not in state:
            return {
                "dissipation_rate": "Q_proxy = r.T star_nom r",
                "thermodynamic_force": "r = c[:spatial_dims] - mean(c); r=0 if c is shorter",
            }
        hodge = self.build_hodge_contract(strict=True).matrix
        concentration = self._as_tensor(state["tracer_concentration"]).to(dtype=hodge.dtype).reshape(-1)
        if concentration.numel() < self.spatial_dims:
            gradient = concentration.new_zeros((self.spatial_dims,))
        else:
            gradient = concentration[: self.spatial_dims] - torch.mean(concentration)
        return {
            "dissipation_rate": gradient @ hodge @ gradient,
            "thermodynamic_force": gradient,
        }

    def reversible_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return a zero structural declaration, not the runtime advection operator."""

        # References: van der Schaft and Maschke (1995), skew structure declarations.
        # Mathematical expression: J=0 on n=spatial_dims audit coordinates.
        # Parameter meanings: n is the directional declaration size.
        # Units: this zero declaration defines no physical advective rate.
        # Domain/constraints: it is not the runtime advection operator and
        #   cannot prove its boundary power or conservation by itself.
        # Sign convention: zero contributes no reversible audit drift.
        size = self.spatial_dims
        return {
            "state_vars": tuple(f"tracer_concentration[{idx}]" for idx in range(size)),
            "J": torch.zeros((size, size), device=self.device),
        }

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the nominal positive atmospheric material audit matrix.

        References:
          - Stull (1988), local K-theory closure.
          - van der Schaft and Maschke (1995), dissipative port-Hamiltonian form.
        Mathematical expression:
          M_h = diag(K_nom), K_nom >= 0, Q_proxy = r.T M_h r >= 0.
        Parameter meanings:
          K is bounded by molecular_diffusivity plus eddy_diffusivity_max.
        """

        # References: Stull (1988); van der Schaft and Maschke (1995).
        # Mathematical expression: M=diag(K_nom), r.T*M*r>=0 for K_nom>=0.
        # Parameter meanings: K_nom is the nominal upper-bound material scale.
        # Units: M [m2 s-1]; its quadratic proxy is not thermodynamic entropy.
        # Domain/constraints: structural declaration only; runtime uses the
        #   existing transport operator with stability-dependent K instead.
        # Sign convention: positive semidefinite M gives a nonnegative proxy.
        hodge = self.build_hodge_contract(strict=True).matrix
        if bool(torch.any(hodge < 0.0)):
            raise ValueError("Atmospheric tracer mobility must be non-negative in M_h.")
        return {
            "state_vars": tuple(
                f"tracer_concentration[{idx}]" for idx in range(hodge.shape[0])
            ),
            "J": torch.zeros_like(hodge),
            "M": hodge,
            "ports": self.ports(),
        }

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Declare atmospheric tracer source and boundary matter ports.

        References:
          - Jacobson (2005), atmospheric tracer transport boundary fluxes.
          - Seinfeld and Pandis (2016), atmospheric source and sink terms.
        Mathematical expression:
          P_port = e_h f_h for boundary flux and source-term flow variables.
        Parameter meanings:
          e_h is tracer concentration and f_h is either the packed outward face
          flux ledger diagnostics.boundary_flux_cochain [kg m-2 s-1] or the
          source_term input [kg m-3 s-1].
        """

        effort_space = CochainSpaceSpec(
            complex_id="atmospheric_advection_diffusion",
            degree=0,
            primality="primal",
            n_dofs=1,
            units="kg m-3",
            extensivity="intensive",
            physical_domain="atmosphere",
            semantic_name="atmospheric_tracer_effort",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        boundary_flow_space = CochainSpaceSpec(
            complex_id="atmospheric_advection_diffusion",
            degree=2,
            primality="dual",
            n_dofs=1,
            units="kg m-2 s-1",
            extensivity="integral",
            physical_domain="atmosphere",
            semantic_name="atmospheric_tracer_boundary_flux",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        source_flow_space = CochainSpaceSpec(
            complex_id="atmospheric_advection_diffusion",
            degree=3,
            primality="dual",
            n_dofs=1,
            units="kg m-3 s-1",
            extensivity="density",
            physical_domain="atmosphere",
            semantic_name="atmospheric_tracer_source",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        return (
            PortSpecIPHS(
                name="atmosphere_tracer_boundary_export",
                effort_var="tracer_concentration",
                flow_var="diagnostics.boundary_flux_cochain",
                effort_space=effort_space,
                flow_space=boundary_flow_space,
                power_units="kg2 m-5 s-1",
                positive_power_convention="out_of_system",
                port_kind="matter",
            ),
            PortSpecIPHS(
                name="atmosphere_tracer_source_port",
                effort_var="tracer_concentration",
                flow_var="source_term",
                effort_space=effort_space,
                flow_space=source_flow_space,
                power_units="kg2 m-6 s-1",
                positive_power_convention="into_system",
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
    
    def forward(
        self,
        state: StateDict,
        time: float,
        dt: float,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward time step for atmospheric advection-diffusion.
        
        Parameters:
            state : StateDict
                Current atmospheric state
                Required keys: "tracer_concentration" [kg m-3]; for the
                monin-obukhov scheme also "temperature" [K],
                "sensible_heat_flux" [W m-2] (output of atmospheric_turbulence
                or explicit forcing), "atmospheric_pressure" [Pa] and
                "relative_humidity" [1] of the moist-air density; the wind
                components "wind_u", "wind_v" and, in 3-D, "wind_w" [m s-1]
                (no defaults: a missing input raises)
            time : float
                Current time [s]
            dt : float
                Time step [s]
            
        Returns:
            increments : Dict[str, Any]
                Explicit integrated tracer increment (UpdateItem, ACCUMULATE,
                per_timestep=True) plus diagnostics, including the packed
                outward face flux ledger when exterior faces exist.
        """
        # References: Jacobson (2005), scalar advection-diffusion; Stull (1988).
        # Mathematical expression: c_t=-div(v*c)+div(K*grad(c))+S;
        #   delta_c is the existing transport operator's accepted interval update.
        # Parameter meanings: c tracer concentration, v wind, K total diffusivity,
        #   S external source, dt the scheduler-owned integration interval.
        # Units: c [kg m-3], v [m s-1], K [m2 s-1], S [kg m-3 s-1], dt [s].
        # Domain/constraints: configured grid/boundaries and explicit stability
        #   limits; this Functor must not silently shorten the requested interval.
        # Sign convention: positive S and delta_c add tracer to the cell.
        state = dict(state or {})

        # Get tracer concentration
        if "tracer_concentration" in state:
            tracer_concentration = state["tracer_concentration"]
        else:
            raise ValueError("State must contain 'tracer_concentration'")
        
        concentration = tracer_concentration.to(self.device)
        remove_batch_dimension = False
        if concentration.dim() == self.spatial_dims:
            concentration = concentration.unsqueeze(0)
            remove_batch_dimension = True
        
        # The wind components [m s-1] are required inputs with no
        # ParameterManager fallback (wind_w only in 3-D): a missing one raises.
        wind_keys = ("wind_u", "wind_v", "wind_w")[: self.spatial_dims]
        missing_winds = [key for key in wind_keys if state.get(key) is None]
        if missing_winds:
            raise ValueError(
                f"{self.name} needs the wind components {', '.join(wind_keys)} "
                "[m s-1] as forcing or state; missing: "
                + ", ".join(missing_winds) + "; no default is applied."
            )
        wind_u = self._broadcast_field(state["wind_u"], concentration)
        wind_v = self._broadcast_field(state["wind_v"], concentration)
        wind_w = (
            self._broadcast_field(state["wind_w"], concentration)
            if self.spatial_dims == 3
            else None
        )
        
        # References: Hundsdorfer and Verwer (2003); Jacobson (2005), tracer sources.
        # Mathematical expression: S = source_term when the optional volumetric
        #   source port is supplied, otherwise S = 0 (closed no-source case).
        # Parameter meanings: S the volumetric tracer source rate of the port
        #   atmosphere_tracer_source_port.
        # Units: S [kg m-3 s-1].
        # Domain/constraints: a supplied field must broadcast to the tracer field;
        #   no ParameterManager background source is applied.
        # Sign convention: positive S adds tracer to the cell (into the system).
        if state.get("source_term") is None:
            source_term = torch.zeros_like(concentration)
        else:
            source_term = self._broadcast_field(state["source_term"], concentration)
        
        # Compute atmospheric stability
        stability = self._compute_atmospheric_stability(state)
        
        # Compute eddy diffusivity based on stability
        eddy_diffusivity = self._compute_eddy_diffusivity(state, stability)
        
        # Compute total diffusivity (molecular + eddy)
        total_diffusivity = self.molecular_diffusivity + eddy_diffusivity
        
        # References: LeVeque (2002), axis-aligned finite-volume transport.
        # Mathematical expression: h_storage=reverse((dx,dy[,dz])).
        # Parameter meanings: u,v,w are physical components on (...,z,y,x)
        #   tensors; h_storage is the same width tuple in storage-axis order.
        # Units: every width is in meters; the permutation changes no value.
        # Domain/constraints: the canonical CFL, diffusion-stability and
        #   transport operators require storage order; boundary descriptors
        #   keep their physical face names.
        # Sign convention: component directions stay +x,+y,+z; only widths reorder.
        operator_grid_spacing = tuple(reversed(self.grid_spacing))
        velocity = (wind_u, wind_v) if wind_w is None else (wind_u, wind_v, wind_w)

        boundary_method, advection_boundary_state, diffusion_boundary_state = (
            self._resolve_canonical_boundary_law(concentration)
        )

        # Check stability constraints using unified operators
        cfl_operator = OperatorFactory.create(
            "diagnostics.cfl_calculator",
            name="cfl_checker",
            device=self.device,
            dtype=concentration.dtype,
        )
        cfl_number = cfl_operator.apply(
            velocity=velocity,
            grid_spacing=operator_grid_spacing,
            time_step=dt,
        )

        stability_operator = OperatorFactory.create(
            "diagnostics.stability_checker",
            name="stability_checker",
            device=self.device,
            dtype=concentration.dtype,
        )
        stability_number = stability_operator.apply(
            diffusivity=total_diffusivity,
            grid_spacing=operator_grid_spacing,
            time_step=dt,
        )

        # Get maximum allowed values from parameters
        max_cfl = _to_float(
            self.get_parameter_value(
                'cfl_advection_max',
                namespace='transport.numerical',
                source_order=("manager",), required=True)
        )
        max_stability = _to_float(
            self.get_parameter_value(
                'cfl_diffusion_max',
                namespace='transport.numerical',
                source_order=("manager",), required=True)
        )

        # References: LeVeque (2002), explicit transport stability bounds.
        # Mathematical expression: dt_bound=dt*min(Cmax/C,Dmax/D).
        # Parameter meanings: C,D are maximum advection/diffusion diagnostics;
        #   Cmax,Dmax are their ParameterManager-owned limits.
        # Units: diagnostics [1]; requested and recommended intervals [s].
        # Domain/constraints: only positive diagnostic ratios constrain dt;
        #   the canonical combined operator may impose a stricter joint bound.
        # Sign convention: a rejected interval produces no concentration update.
        max_cfl_val = torch.max(cfl_number)
        max_stability_val = torch.max(stability_number)

        if max_cfl_val > max_cfl or max_stability_val > max_stability:
            admissible_ratios = [
                limit / float(value)
                for value, limit in ((max_cfl_val, max_cfl), (max_stability_val, max_stability))
                if float(value) > 0.0
            ]
            dt_stable = float(dt) * min(admissible_ratios)
            raise ValueError(
                f"Requested time step {float(dt):.8g} s exceeds transport stability limits; "
                f"individual-limit interval bound is {dt_stable:.8g} s; "
                "the combined operator may require a smaller interval."
            )

        # References: Patankar (1980); LeVeque (2002); Hundsdorfer and Verwer (2003).
        # Mathematical expression: c_new=c+dt*(-div(v c)+div(K grad c)+S) on the
        #   caller's actual cells and delta_c=c_new-c; the same operator returns
        #   the outward face ledger j_out, so
        #   sum(delta_c)dV-dt*sum(S)dV+dt*sum_f(j_out,f*A_f)=0 up to roundoff.
        # Parameter meanings: c caller concentration, v wind, K total diffusivity,
        #   S source, j_out outward-positive face flux, A_f face area, dV cell volume.
        # Units: c,delta_c [kg m-3]; S [kg m-3 s-1]; j_out [kg m-2 s-1]; dt [s].
        # Domain/constraints: no real cell is projected before or after the solve;
        #   the resolved homogeneous face law is the only boundary input, and the
        #   operator rejects unstable or negative results before returning.
        # Sign convention: positive delta_c adds tracer; positive j_out exports it.
        advection_diffusion_operator = OperatorFactory.create(
            "transport.advection_diffusion",
            name="atmospheric_transport",
            device=self.device,
            dtype=concentration.dtype,
            config={
                "splitting_method": "combined",
                "advection_scheme": "upwind",
                "boundary_method": boundary_method,
                "check_stability": True,
                "enforce_non_negative": True,
            },
        )
        transport_result = advection_diffusion_operator.apply(
            concentration=concentration,
            velocity=velocity,
            diffusivity=total_diffusivity,
            diffusivity_layout="scalar_field",
            grid_spacing=operator_grid_spacing,
            time_step=dt,
            spatial_dims=self.spatial_dims,
            source_term=source_term,
            boundary_method=boundary_method,
            advection_boundary_state=advection_boundary_state,
            diffusion_boundary_state=diffusion_boundary_state,
        )
        concentration_new = transport_result['concentration']
        conservation_error = transport_result.get('conservation_error')
        if conservation_error is None:
            raise RuntimeError(
                "Canonical advection-diffusion operator omitted the finite-volume "
                "conservation ledger required by atmospheric transport"
            )
        boundary_fluxes = transport_result.get('boundary_fluxes') or {}

        delta_concentration = concentration_new - concentration
        advection_rate = transport_result.get('advection_rate')
        diffusion_rate = transport_result.get('diffusion_rate')
        if advection_rate is None or diffusion_rate is None:
            raise RuntimeError(
                "Canonical advection-diffusion operator omitted the advection or "
                "diffusion rate required by atmospheric transport diagnostics"
            )
        # Shape convention: every per-cell output (the increment and the
        # advection and diffusion rates) has the caller's tracer shape; the
        # leading batch axis added for an unbatched field is removed from all.
        if remove_batch_dimension:
            delta_concentration = delta_concentration.squeeze(0)
            advection_rate = advection_rate.squeeze(0)
            diffusion_rate = diffusion_rate.squeeze(0)

        # References: Hundsdorfer and Verwer (2003); RELIC StateManager increment contract.
        # Mathematical expression: tracer_concentration <- tracer_concentration + delta_c.
        # Parameter meanings: delta_c is the interval increment already integrated over dt.
        # Units: delta_c [kg m-3]; per_timestep=True forbids a second dt scaling.
        # Domain/constraints: exactly one governed state advance per call; no
        #   flux-tolerance filter or unit round trip alters the returned increment.
        # Sign convention: positive delta_c increases the stored tracer.
        increments: Dict[str, Any] = {
            'tracer_concentration': UpdateItem(
                value=delta_concentration,
                mode=UpdateMode.ACCUMULATE,
                unit="kg m-3",
                per_timestep=True,
                source_functor=self.name,
            ),
            'diagnostics.conservation_error': conservation_error,
            # Cell-centred Courant sum max(dt * sum_k |u_k| / dx_k) of the
            # pre-step limit check, named apart from diagnostics.cfl_number (the
            # finite-volume outward-face sum of the generic transport Functors).
            'diagnostics.atmospheric_advection_diffusion.cfl_number': max_cfl_val,
            'diagnostics.stability_number': max_stability_val,
            'diagnostics.advection_rate': advection_rate,
            'diagnostics.diffusion_rate': diffusion_rate,
            'diagnostics.eddy_diffusivity': torch.mean(eddy_diffusivity),
            'diagnostics.stability_parameter': stability,
        }
        if boundary_fluxes:
            # References: Patankar (1980); LeVeque (2002).
            # Mathematical expression:
            #   q_boundary=[q_xmin,q_xmax,q_ymin,q_ymax(,q_zmin,q_zmax)].
            # Parameter meanings: each q_face is the operator's outward-positive
            #   advective plus diffusive face flux for the returned increment.
            # Units: [kg m-2 s-1]; a 2D ledger is per unit omitted depth.
            # Domain/constraints: diagnostic disclosure packed by the existing
            #   computation-layer packer; periodic domains have no exterior
            #   faces and omit this optional output; it is not a StateManager
            #   flux writer.
            # Sign convention: positive values export tracer from the domain.
            increments['diagnostics.boundary_flux_cochain'] = (
                pack_outward_boundary_flux_cochain(
                    boundary_fluxes,
                    squeeze_batch=remove_batch_dimension,
                )
            )
        return increments
    
    def _compute_atmospheric_stability(self, state: StateDict) -> Tensor:
        """
        Compute atmospheric stability parameter.

        References:
            - Businger et al. (1971) Flux-profile relationships in the atmospheric surface layer.
            - Hass et al. (1991) Boundary-layer parameterizations in regional transport models.
            - CMAQ Science Algorithms (Chs. 6-7, US EPA, 1999) for HDIFF/VDIFF decomposition.
        Mathematical expressions:
            - Monin-Obukhov length: L = -rho*cp*T*u_*^3/(kappa*g*H).
            - Bulk Richardson number: Ri = (g/theta_ref)*(dtheta/dz)*z^2/|U|^2.
        Parameters:
            u_* (m s-1): friction velocity (minimum value from atmosphere.turbulence).
            kappa (dimensionless): von Karman constant.
            g (m s-2): gravitational acceleration (physical constant).
            T / theta_ref (K): input temperature / reference potential temperature.
            H (W m-2): surface upward sensible heat flux sensible_heat_flux, the
                atmospheric_turbulence output or explicit forcing (no default).
            rho (kg m-3): moist-air density of T, atmospheric_pressure p [Pa] and
                relative_humidity RH [1] (no defaults).
            dtheta/dz (K m-1): prescribed potential temperature gradient.
        """
        # References: Monin and Obukhov (1954); Richardson (1920); Stull (1988).
        # Mathematical expression: zeta=z/L, L=-rho*cp*T*u_star^3/(kappa*g*H);
        #   Ri=(g/theta_ref)*theta_z/(U/z)^2, with z=boundary_layer_height/2.
        # Parameter meanings: rho the moist-air density of T, p and RH (see the
        #   density block below), cp heat capacity, H the surface
        #   upward sensible heat flux read from state as sensible_heat_flux (the
        #   atmospheric_turbulence output H = rho cp u* theta*, or explicit
        #   forcing; CF surface_upward_sensible_heat_flux), theta_z prescribed
        #   gradient; minimum u_star is a reduced closure.
        # Units: H [W m-2], rho [kg m-3], cp [J kg-1 K-1], T,theta_ref [K],
        #   theta_z [K m-1], U,u_star [m s-1], z,L [m]; zeta,Ri [1].
        # Domain/constraints: canonical MOST safeguards apply; T, H, p and RH have
        #   no default and a missing one raises; bulk Ri assumes linear theta and
        #   a nonzero U/z shear proxy, not resolved wind shear.
        # Sign convention: upward heat gives negative zeta; positive theta_z stable Ri.
        stability_limit = self._mixing_coefficients["transport_stability_limit"]
        if self.stability_scheme == "constant":
            template = state.get("tracer_concentration")
            return torch.zeros((), device=self.device, dtype=template.dtype if isinstance(template, Tensor) else torch.get_default_dtype())
        
        elif self.stability_scheme == "monin-obukhov":
            missing = [
                key
                for key in (
                    "temperature", "sensible_heat_flux",
                    "atmospheric_pressure", "relative_humidity",
                )
                if state.get(key) is None
            ]
            if missing:
                raise ValueError(
                    "The monin-obukhov stability scheme needs the air 'temperature' [K], "
                    "the surface upward sensible heat flux 'sensible_heat_flux' [W m-2], "
                    "written by atmospheric_turbulence or supplied as forcing, and the "
                    "'atmospheric_pressure' [Pa] and 'relative_humidity' [1] of the "
                    "moist-air density; missing: " + ", ".join(missing)
                    + "; no default is applied."
                )
            fields = [
                self._as_tensor(state[key])
                for key in (
                    "temperature", "sensible_heat_flux",
                    "atmospheric_pressure", "relative_humidity",
                )
            ]
            dtype = fields[0].dtype
            for field in fields[1:]:
                dtype = torch.promote_types(dtype, field.dtype)
            (
                temperature, sensible_heat_flux, atmospheric_pressure, relative_humidity,
            ) = torch.broadcast_tensors(*(field.to(dtype=dtype) for field in fields))

            # References: Tetens (1930); Allen et al. (1998), FAO-56 Eqs. 11 and 19;
            #   Holton and Hakim (2012), Eqs. 2.63-2.64 (compute_moist_air_density).
            # Mathematical expression: e = RH * e_s(T), e_s = e_0 exp(a Tc / (Tc + b)),
            #   Tc = T - K_0; rho = (p - (1 - eps) e) / (R_d T).
            # Parameter meanings: RH relative humidity, T air temperature, p
            #   atmospheric pressure, e actual and e_s saturation vapour pressure,
            #   e_0/a/b/K_0 psychrometric physical constants, eps = M_w / M_d, R_d
            #   dry-air gas constant.
            # Units: RH [1]; T, K_0 [K]; Tc, b [degC]; p, e, e_s, e_0 [Pa]; a [1];
            #   rho [kg m-3].
            # Domain/constraints: 0 <= RH <= 1 raises otherwise; T > 0, p > 0 and
            #   0 <= e < p (compute_moist_air_density raises); no defaults for p, RH.
            # Sign convention: rho is a positive state property, not a flux.
            validate_unity_range(relative_humidity, "relative_humidity")
            vapor_pressure = relative_humidity * compute_saturation_vapor_pressure(
                temperature, **self._tetens_constants
            )
            air_density = compute_moist_air_density(
                temperature, atmospheric_pressure, vapor_pressure
            )
            
            # Obukhov length (reduced-order)
            u_star = _to_float(
                self.get_parameter_value(
                    'minimum_friction_velocity',
                    namespace='atmosphere.turbulence',
                    source_order=("manager",), required=True)
            )
            length = monin_obukhov_length(
                u_star=torch.full_like(temperature, u_star), rho=air_density,
                cp=self.air_specific_heat, T=temperature, H=sensible_heat_flux,
                k=self.von_karman_constant, g=self.gravity_acceleration,
            )
            height = length.new_tensor(self.boundary_layer_height / 2.0)
            return torch.clamp(zeta(height, length), min=-stability_limit, max=stability_limit)
        
        elif self.stability_scheme == "richardson":
            # Richardson number
            template = state["tracer_concentration"].to(self.device)
            wind_u = self._broadcast_field(state["wind_u"], template)
            wind_v = self._broadcast_field(state["wind_v"], template)

            wind_speed = torch.sqrt(wind_u**2 + wind_v**2)
            
            potential_temp_gradient = _to_float(
                self.get_parameter_value(
                    'potential_temperature_gradient',
                    namespace='atmosphere.stability',
                    source_order=("manager",), required=True)
            )
            
            height = wind_speed.new_tensor(self.boundary_layer_height / 2.0)
            Ri = richardson_number(
                delta_theta=height * potential_temp_gradient, delta_u=wind_speed,
                delta_z=height, theta_mean=wind_speed.new_tensor(self.reference_temperature),
                g=self.gravity_acceleration,
            )
            return torch.clamp(Ri, min=-stability_limit, max=stability_limit)
        
        return torch.zeros((), device=self.device)
    
    def _as_tensor(self, value: Any) -> Tensor:
        """Ensure value is a tensor on the correct device."""
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        return torch.tensor(_to_float(value), device=self.device)
    
    def _broadcast_field(self, value: Any, template: Tensor) -> Tensor:
        """
        Broadcast a scalar or tensor field onto the template shape.

        Parameters
        ----------
        value : Any
            Candidate field (tensor or scalar); None raises, since no field has
            a ParameterManager default.
        template : Tensor
            Tensor providing target shape and dtype.

        Returns
        -------
        Tensor
            Field tensor matching template shape.
        """
        if value is None:
            raise ValueError("Required forcing field is missing and has no default")
        if isinstance(value, torch.Tensor):
            value = value.to(device=self.device, dtype=template.dtype)
            try:
                return value.reshape(()).expand_as(template) if value.numel() == 1 else value.expand_as(template)
            except RuntimeError as exc:
                raise ValueError("Forcing field shape inconsistent with tracer concentration") from exc
        else:
            fill_value = _to_float(value)

        return torch.full_like(template, fill_value, device=self.device)
    
    def _compute_eddy_diffusivity(self, state: StateDict, stability: Tensor) -> Tensor:
        """Compute eddy diffusivity based on atmospheric stability."""
        # References: Stull (1988), K-theory; RELIC's existing empirical closure.
        # Mathematical expression: K0=f*Kmax; K=K0*(1+a*s/(1+s)) for s=-zeta>0,
        #   otherwise K=K0/(1+b*zeta); cap K at Kmax.
        # Parameter meanings: f,a,b and the cap are canonical YAML coefficients;
        #   the Richardson option supplies its declared bulk stability proxy.
        # Units: K,K0,Kmax [m2 s-1]; f,a,b,s,zeta [1].
        # Domain/constraints: pointwise stable/unstable half-lines; compute each
        #   branch on its own safe domain so inactive divisions stay finite.
        # Sign convention: negative stability enhances positive diffusive mixing.
        K_base = self.eddy_diffusivity_max * self._mixing_coefficients["transport_neutral_diffusivity_fraction"]
        
        if self.stability_scheme == "constant":
            return torch.full_like(stability, K_base)
        
        unstable = torch.clamp(-stability, min=0.0)
        stable = torch.clamp(stability, min=0.0)
        enhancement = 1.0 + self._mixing_coefficients["transport_unstable_enhancement_coefficient"] * unstable / (1.0 + unstable)
        suppression = 1.0 / (1.0 + self._mixing_coefficients["transport_stable_suppression_coefficient"] * stable)
        diffusivity = K_base * torch.where(stability < 0, enhancement, suppression)
        return torch.clamp(diffusivity, max=self.eddy_diffusivity_max)
    
      
    def _active_faces(self) -> Tuple[str, ...]:
        """Return the physical face names owned by this Functor's spatial dimension."""
        return self._FACE_ORDER[: 2 * self.spatial_dims]

    @staticmethod
    def _scalar_float(value: Any) -> float:
        """Read a static scalar datum as float without touching an autograd graph."""
        if isinstance(value, torch.Tensor):
            return float(value.detach().reshape(()).item())
        return float(value)

    @staticmethod
    def _static_scalar_boundary_value(face: str, value: Any) -> Any:
        """Validate one finite static scalar boundary datum.

        References:
          - Patankar (1980), finite-volume boundary faces.
        Mathematical expression:
          g_face is one real number or a one-element real floating tensor.
        Parameter meanings:
          face names the physical boundary; value is the configured datum.
        Units:
          [kg m-3] for a Dirichlet concentration, [kg m-4] for a Neumann gradient.
        Domain/constraints:
          finite, non-boolean, non-complex, exactly one element; callables,
          bools, integers-as-flags and multi-element tensors are rejected.
        Sign convention:
          none; the datum is configuration, not a signed flux.
        """
        if isinstance(value, bool):
            raise TypeError(f"Boundary face '{face}' value must be numeric, not bool")
        if isinstance(value, torch.Tensor):
            if value.is_complex() or not value.is_floating_point():
                raise TypeError(
                    f"Boundary face '{face}' tensor value must be real floating point"
                )
            if value.numel() != 1:
                raise ValueError(
                    f"Boundary face '{face}' tensor value must hold exactly one element; "
                    "spatially varying face data has no canonical combined-operator contract"
                )
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"Boundary face '{face}' value must be finite")
            return value
        if isinstance(value, (int, float)):
            scalar = float(value)
            if not math.isfinite(scalar):
                raise ValueError(f"Boundary face '{face}' value must be finite")
            return scalar
        raise TypeError(
            f"Boundary face '{face}' value of type {type(value).__name__} is not a "
            "static scalar datum"
        )

    def _resolve_canonical_boundary_law(
        self,
        concentration: Tensor,
    ) -> Tuple[str, Optional[Tensor], Optional[Tensor]]:
        """Lower the configured face descriptors to the canonical operator law.

        References:
          - Patankar (1980), finite-volume boundary faces.
          - LeVeque (2002), ghost-state closure of explicit transport stencils.
        Mathematical expression:
          zero Neumann: dc/dn|face=0 -> j_diff,out=0 and advective ghost
            c_ghost=c_adjacent;
          periodic: c(x+L)=c(x) on every axis;
          Dirichlet: c_face=g on every face, j_diff,out=-K(g-c_adjacent)/(h/2)
            on the outward normal and advective inflow ghost c_ghost=g.
        Parameter meanings:
          g is one finite static scalar concentration shared by all faces;
          h is the face-normal cell width; K is the total diffusivity.
        Units:
          g,c [kg m-3]; K [m2 s-1]; h [m]; j [kg m-2 s-1].
        Domain/constraints:
          all active faces must carry one common law; nonzero Neumann, Robin,
          wall, open, mixed, callable, time-dependent, non-scalar or unequal
          Dirichlet data have no canonical combined-operator contract and are
          rejected before any numerical work or state mutation.
        Sign convention:
          outward-positive face fluxes; zero Neumann still allows advective
          outflow with nonzero normal wind and is not a closed wall.
        """
        faces = self._active_faces()
        configured = self.boundary_manager.boundary_conditions
        if set(configured) != set(faces):
            raise ValueError(
                "Atmospheric boundary descriptors must cover exactly the active faces "
                f"{list(faces)}; configured {sorted(configured)}"
            )
        descriptors = [configured[face] for face in faces]
        laws = {descriptor.bc_type for descriptor in descriptors}
        if len(laws) != 1:
            raise ValueError(
                "Mixed boundary laws are not representable by the canonical combined "
                f"transport operator: {sorted(law.value for law in laws)}"
            )
        law = next(iter(laws))
        for face, descriptor in zip(faces, descriptors):
            if descriptor.time_dependent or callable(descriptor.value):
                raise ValueError(
                    f"Boundary face '{face}' uses a callable or time-dependent value; "
                    "no restart-safe forcing contract exists for it"
                )
        if law == BoundaryType.PERIODIC:
            return "periodic", None, None
        if law == BoundaryType.NEUMANN:
            for face, descriptor in zip(faces, descriptors):
                gradient = self._static_scalar_boundary_value(face, descriptor.value)
                if self._scalar_float(gradient) != 0.0:
                    raise ValueError(
                        f"Boundary face '{face}' prescribes a nonzero Neumann gradient; "
                        "the canonical combined operator has no prescribed-flux contract"
                    )
            return "neumann", None, None
        if law == BoundaryType.DIRICHLET:
            values = [
                self._static_scalar_boundary_value(face, descriptor.value)
                for face, descriptor in zip(faces, descriptors)
            ]
            reference = values[0]
            for face, value in zip(faces[1:], values[1:]):
                if self._scalar_float(value) != self._scalar_float(reference):
                    raise ValueError(
                        f"Boundary face '{face}' Dirichlet value differs from face "
                        f"'{faces[0]}'; the canonical combined operator accepts one "
                        "static scalar shared by all faces"
                    )
            if isinstance(reference, torch.Tensor):
                boundary_state = reference.reshape(()).to(
                    device=concentration.device, dtype=concentration.dtype,
                )
            else:
                boundary_state = concentration.new_tensor(reference)
            return "dirichlet", boundary_state, boundary_state
        raise ValueError(
            f"Boundary law '{law.value}' has no canonical finite-volume contract in the "
            "combined atmospheric transport operator (supported: zero-gradient neumann, "
            "periodic, uniform static scalar dirichlet)"
        )

    def get_extra_state(self) -> Dict[str, Any]:
        """Return checkpoint-safe boundary descriptors for durable restart.

        References:
          - PyTorch Module extra-state contract.
          - RELIC simulator checkpoint owner, strict functor state_dict grammar.
        Mathematical expression:
          z_boundary={face:(law,g_face)} over the active faces.
        Parameter meanings:
          law is the BoundaryType value; g_face is the static scalar datum.
        Units:
          g_face [kg m-3] for Dirichlet, [kg m-4] for Neumann, unused for periodic.
        Domain/constraints:
          only neumann, periodic and dirichlet descriptors with finite static
          scalar data are serializable; callable, time-dependent, robin, wall
          and open descriptors are rejected instead of being sampled once.
        Sign convention:
          none; descriptors are configuration, not fluxes.
        """
        faces = self._active_faces()
        configured = self.boundary_manager.boundary_conditions
        if set(configured) != set(faces):
            raise ValueError(
                "Atmospheric boundary descriptors must cover exactly the active faces "
                f"{list(faces)} before checkpointing; configured {sorted(configured)}"
            )
        descriptors: Dict[str, Dict[str, Any]] = {}
        for face in faces:
            descriptor = configured[face]
            if descriptor.bc_type not in self._CHECKPOINTABLE_BOUNDARY_TYPES:
                raise ValueError(
                    f"Boundary face '{face}' law '{descriptor.bc_type.value}' cannot be "
                    "checkpointed; no canonical combined-operator contract exists for it"
                )
            if descriptor.time_dependent or callable(descriptor.value):
                raise ValueError(
                    f"Boundary face '{face}' uses a callable or time-dependent value; "
                    "a single sample is not a restart-safe time law"
                )
            value = self._static_scalar_boundary_value(face, descriptor.value)
            if isinstance(value, torch.Tensor):
                value = value.detach().clone()
            descriptors[face] = {"type": descriptor.bc_type.value, "value": value}
        return {
            "format": self.EXTRA_STATE_FORMAT,
            "version": self.EXTRA_STATE_VERSION,
            "spatial_dims": int(self.spatial_dims),
            "boundary_conditions": descriptors,
        }

    def set_extra_state(self, state: Any) -> None:
        """Restore boundary descriptors after validating the complete payload."""
        if not isinstance(state, Mapping):
            raise TypeError("Atmospheric transport extra state must be a mapping.")
        expected_keys = {"format", "version", "spatial_dims", "boundary_conditions"}
        if set(state) != expected_keys:
            raise ValueError(
                "Atmospheric transport extra-state keys do not match the required schema."
            )
        if state["format"] != self.EXTRA_STATE_FORMAT:
            raise ValueError("Atmospheric transport extra-state format is unsupported.")
        if state["version"] != self.EXTRA_STATE_VERSION:
            raise ValueError("Atmospheric transport extra-state version is unsupported.")
        spatial_dims = state["spatial_dims"]
        if (
            isinstance(spatial_dims, bool)
            or not isinstance(spatial_dims, int)
            or spatial_dims != self.spatial_dims
        ):
            raise ValueError(
                "Atmospheric transport extra-state spatial_dims does not match this Functor."
            )
        faces = self._active_faces()
        descriptors = state["boundary_conditions"]
        if not isinstance(descriptors, Mapping) or set(descriptors) != set(faces):
            raise ValueError(
                "Atmospheric transport extra-state boundary faces must equal "
                f"{list(faces)}."
            )
        restored: Dict[str, BoundaryCondition] = {}
        for face in faces:
            entry = descriptors[face]
            if not isinstance(entry, Mapping) or set(entry) != {"type", "value"}:
                raise ValueError(
                    f"Atmospheric transport extra-state face '{face}' must map type and value."
                )
            try:
                law = BoundaryType(entry["type"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Atmospheric transport extra-state face '{face}' law is invalid."
                ) from exc
            if law not in self._CHECKPOINTABLE_BOUNDARY_TYPES:
                raise ValueError(
                    f"Atmospheric transport extra-state face '{face}' law "
                    f"'{law.value}' is not restorable."
                )
            value = self._static_scalar_boundary_value(face, entry["value"])
            if isinstance(value, torch.Tensor):
                value = value.detach().clone().to(self.boundary_manager.device)
            restored[face] = BoundaryCondition(law, value, face)
        self.boundary_manager.boundary_conditions.clear()
        self.boundary_manager.boundary_conditions.update(restored)

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: Dict[str, Any],
    ) -> Dict[str, float]:
        """Report the open-system finite-volume ledger residual of the returned increment.

        References:
          - Patankar (1980), Numerical Heat Transfer and Fluid Flow.
          - Hundsdorfer and Verwer (2003), Time-Dependent ADR Equations.
          - RELIC conservation architecture, per-Functor local hook.
        Mathematical expression:
          r=max|DeltaM-dt*integral(S)dV+dt*sum_f(j_out,f*A_f)|
            /max(M_scale,|dt*integral(S)dV|,|dt*sum_f(j_out,f*A_f)|,eps),
          evaluated by the canonical operator on the caller cells c and on the
          c_new whose difference this Functor returns as delta_c.
        Parameter meanings:
          DeltaM stored-mass change, S source rate, j_out outward face flux,
          A_f face area, dV cell volume, M_scale the larger absolute inventory.
        Units:
          r [1]; DeltaM [kg] per unit omitted depth in 2D; S [kg m-3 s-1].
        Domain/constraints:
          the ledger diagnostic must be present, real, finite and scalar; a raw
          sum of increments is not used because sources and boundary exchange
          legitimately change an open inventory; a missing ledger is an error.
        Sign convention:
          r is nonnegative and reported as the violation of each declared law.
        """
        if not self.metadata.conservation_laws:
            return {}
        if "tracer_concentration" not in output_updates:
            raise ValueError(
                f"{self.name}: conservation check requires the tracer_concentration increment"
            )
        ledger = output_updates.get("diagnostics.conservation_error")
        if isinstance(ledger, UpdateItem):
            ledger = ledger.value
        if ledger is None:
            raise ValueError(
                f"{self.name}: conservation check requires diagnostics.conservation_error"
            )
        ledger_tensor = torch.as_tensor(ledger)
        if (
            ledger_tensor.numel() != 1
            or ledger_tensor.is_complex()
            or not bool(torch.isfinite(ledger_tensor).all())
        ):
            raise ValueError(
                f"{self.name}: diagnostics.conservation_error must be one finite real scalar"
            )
        residual = abs(float(ledger_tensor.detach().reshape(()).item()))
        return {law: residual for law in self.metadata.conservation_laws}


# Export
__all__ = ['AtmosphericAdvectionDiffusionFunctor']

